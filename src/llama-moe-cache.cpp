#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache-share.h"
#include "llama-moe-cache-policy.h"

#include "ggml-cpp.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace {

static std::vector<ggml_tensor *> layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_up_gate_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t) res.push_back(t);
    }
    return res;
}

static bool is_host_weight(const ggml_tensor * t) {
    return t && t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

static bool same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) return false;
    }
    return true;
}

} // namespace

struct llama_moe_cache::impl {
    struct device {
        ggml_backend * backend = nullptr;
        ggml_backend_buffer_type * buft = nullptr;
        int32_t model_device = -1;
        double split = 0.0;
        size_t host_bytes = 0;
        size_t buffer_size = 0;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
    };
    struct group {
        int32_t device = -1;
        std::vector<ggml_tensor *> ref;
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks;
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        int32_t zero_slots = 0;
        bool cpu_share_enabled = false;
        llama_moe_cache_policy_detail::state cache;
    };
    struct layer {
        int32_t group = -1;
        ggml_tensor * slot_map = nullptr;
        ggml_tensor * gpu_map = nullptr;
        ggml_tensor * cpu_map = nullptr;
        ggml_tensor * gpu_bias_map = nullptr;
        ggml_tensor * cpu_bias_map = nullptr;
        llama_moe_cache_policy_detail::state::cpu_share_plan share_plan;
        bool share_planned = false;
        std::vector<ggml_tensor *> experts;
    };
    struct binding {
        int32_t layer;
        int32_t part;
        ggml_tensor * cached;
    };
    struct stats {
        size_t hits = 0;
        size_t gpu_fills = 0;
        size_t cpu_serviced_unique_misses = 0;
        size_t evictions = 0;
        size_t logical_uploaded_bytes = 0;
    };
    static constexpr int64_t max_batch = llama_moe_cache_policy_detail::state::max_batch;

    int32_t n_expert_used;
    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings;
    std::unordered_map<const ggml_tensor *, int32_t> layer_of;
    std::unordered_map<const ggml_tensor *, int32_t> share_map_of;
    std::vector<int32_t> ids;
    std::vector<llama_moe_cache_policy_detail::state::fill> fills;
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;
    stats stats_small, stats_large;
    size_t requested_size;
    uint32_t cpu_share_q16 = 0;
    double cpu_share = 0.0;
    bool token_geometry_maps = false;
    mutable size_t large_batch_skips = 0;
    mutable size_t incompatible_layer_skips = 0;

    impl(const llama_model & model, const std::vector<ggml_backend *> & backends,
            const std::vector<ggml_backend_buffer_type *> & bufts, size_t size,
            enum llama_moe_cache_policy policy, float share)
            : n_expert_used(model.hparams.n_expert_used), layers(model.layers.size()), requested_size(size), cpu_share(share) {
        if (!llama_moe_cache_policy_detail::cpu_share_q16(share, cpu_share_q16)) {
            throw std::invalid_argument("MoE cache CPU share must be finite and in [0,1]");
        }
        if (cpu_share_q16 == 0) cpu_share = 0.0;
        token_geometry_maps = policy == LLAMA_MOE_CACHE_POLICY_CLASSLRU || cpu_share_q16 != 0;
        if (model.hparams.n_expert == 0 || n_expert_used <= 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }
        if (!model.rpc_servers.empty()) throw std::runtime_error("MoE cache does not support RPC devices");
        if (backends.size() != bufts.size()) throw std::runtime_error("MoE cache backend metadata is inconsistent");

        for (size_t i = 0; i < backends.size(); ++i) {
            if (!ggml_backend_is_cpu(backends[i])) {
                device d;
                d.backend = backends[i];
                d.buft = bufts[i];
                devices.push_back(std::move(d));
            }
        }
        if (devices.empty()) throw std::runtime_error("MoE cache requires a GPU backend");

        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), is_host_weight)) continue;
            if (!std::all_of(experts.begin(), experts.end(), [&](const ggml_tensor * t) {
                    return t->ne[2] == experts[0]->ne[2];
                })) continue;
            if (il >= model.default_layer_device.size()) continue;
            const int32_t model_device = model.default_layer_device[il];
            if (model_device < 0 || (size_t) model_device >= model.devices.size()) continue;
            const auto * layer_buft = model.default_buffer_type_offload(model_device);
            auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return d.buft == layer_buft; });
            if (it_dev == devices.end()) continue;
            const int32_t id = (int32_t) (it_dev - devices.begin());
            it_dev->model_device = model_device;
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.device == id && same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->device = id;
                it->ref = experts;
            }
            it->layers.push_back((int32_t) il);
            for (ggml_tensor * t : experts) {
                it->host_bytes += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no complete host-resident expert layer is eligible. MoE cache is inactive\n", __func__);
            return;
        }

        std::vector<bool> eligible(model.splits.size(), false);
        for (const device & d : devices) {
            if (d.host_bytes && d.model_device >= 0 && (size_t) d.model_device < eligible.size()) {
                eligible[d.model_device] = true;
            }
        }
        const std::vector<double> shares = llama_moe_cache_detail::device_shares(model.splits, eligible);
        for (device & d : devices) {
            d.split = d.model_device >= 0 && (size_t) d.model_device < shares.size() ? shares[d.model_device] : 0.0;
        }

        double split_sum = 0.0;
        for (const device & d : devices) if (d.host_bytes) split_sum += d.split;
        if (split_sum <= 0.0) {
            for (device & d : devices) if (d.host_bytes) d.split = 1.0;
            split_sum = std::count_if(devices.begin(), devices.end(), [](const device & d) { return d.host_bytes != 0; });
        }
        auto alloc_size = [&](const group & g, int32_t slots, int32_t zeros) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.device].buft);
            size_t total = 0;
            for (const ggml_tensor * t : g.ref) {
                const size_t rows = (size_t) slots + (size_t) zeros + 1;
                if (rows > SIZE_MAX / t->nb[2]) return SIZE_MAX;
                const size_t bytes = t->nb[2] * rows;
                if (bytes > SIZE_MAX - alignment) return SIZE_MAX;
                const size_t padded = GGML_PAD(bytes, alignment);
                if (padded > SIZE_MAX - total) return SIZE_MAX;
                total += padded;
            }
            return total;
        };

        size_t host_layer_count = 0;
        std::vector<size_t> device_tensor_counts(devices.size(), 0);
        for (group & g : groups) {
            const device & d = devices[g.device];
            const size_t budget = (size_t) ((double) size * d.split / split_sum * g.host_bytes / d.host_bytes);
            const int32_t n_expert = (int32_t) g.ref[0]->ne[2];
            const int32_t max_slots = (int32_t) std::min<size_t>(INT32_MAX, g.layers.size() * (size_t) n_expert);
            if (cpu_share_q16 != 0 && n_expert_used <= llama_moe_cache_policy_detail::state::max_zero_slots) {
                g.zero_slots = n_expert_used;
                while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1, g.zero_slots) <= budget) ++g.n_slots;
                g.cpu_share_enabled = g.n_slots >= n_expert_used;
            }
            if (!g.cpu_share_enabled) {
                g.zero_slots = 0;
                g.n_slots = 0;
                while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1, 0) <= budget) ++g.n_slots;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: cache budget leaves %zu expert layers uncached because they cannot hold one token\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.cache.init((int32_t) layers.size(), n_expert, g.n_slots,
                    policy == LLAMA_MOE_CACHE_POLICY_CLASSLRU ? llama_moe_cache_policy_detail::kind::classlru : llama_moe_cache_policy_detail::kind::lru);
            device_tensor_counts[g.device] += g.ref.size() * (1 + g.layers.size());
            host_layer_count += g.layers.size() * (g.cpu_share_enabled ? 5 : 1);
        }
        if (host_layer_count == 0) throw std::runtime_error("MoE cache budget is too small to hold one token");

        auto make_ctx = [](size_t n_tensors) {
            ggml_init_params params = { n_tensors * ggml_tensor_overhead(), nullptr, true };
            ggml_context_ptr ctx(ggml_init(params));
            if (!ctx) throw std::runtime_error("failed to create MoE cache tensor metadata");
            return ctx;
        };
        for (size_t id = 0; id < devices.size(); ++id) if (device_tensor_counts[id]) devices[id].ctx = make_ctx(device_tensor_counts[id]);
        ctx_host = make_ctx(host_layer_count);
        const auto buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (group & g : groups) {
            if (g.n_slots == 0) continue;
            auto & d = devices[g.device];
            for (ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(d.ctx.get(), t->type, t->ne[0], t->ne[1],
                        g.n_slots + g.zero_slots + 1);
                if (bank->nb[2] != t->nb[2]) throw std::runtime_error("MoE cache expert layout is not contiguous by expert");
                ggml_format_name(bank, "moe_cache.%s", t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.group = (int32_t) (&g - groups.data());
                l.experts = layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(d.ctx.get(), bank, bank->ne[0], bank->ne[1],
                            g.n_slots + g.zero_slots,
                            bank->nb[1], bank->nb[2], 0);
                    if (cached->type == GGML_TYPE_NVFP4_KS) {
                        ggml_set_input_scale(cached, ggml_get_input_scale(l.experts[ip]));
                    }
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (g.cpu_share_enabled) {
                    l.gpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    l.gpu_bias_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    l.cpu_bias_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.gpu_map, "moe_cache.gpu_map-%d", il);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    ggml_format_name(l.gpu_bias_map, "moe_cache.gpu_bias_map-%d", il);
                    ggml_format_name(l.cpu_bias_map, "moe_cache.cpu_bias_map-%d", il);
                    share_map_of[l.gpu_map] = il;
                    share_map_of[l.cpu_map] = il;
                    share_map_of[l.gpu_bias_map] = il;
                    share_map_of[l.cpu_bias_map] = il;
                    for (ggml_tensor * map : {l.gpu_map, l.cpu_map, l.gpu_bias_map, l.cpu_bias_map}) {
                        buf_host_size += GGML_PAD(ggml_nbytes(map), alignment_host);
                    }
                }
            }
            d.buffer_size += alloc_size(g, g.n_slots, g.zero_slots);
        }

        for (device & d : devices) {
            if (!d.ctx) continue;
            d.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
            if (!d.buffer) throw std::runtime_error("failed to allocate MoE cache buffers");
            ggml_backend_buffer_clear(d.buffer.get(), 0);
            d.buffer_size = ggml_backend_buffer_get_size(d.buffer.get());
        }
        buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
        if (!buf_host) throw std::runtime_error("failed to allocate MoE cache slot maps");
        ggml_backend_buffer_clear(buf_host.get(), 0xff);
        buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
        for (group & g : groups) for (int32_t il : g.layers) g.cache.slot_map[il] = (int32_t *) layers[il].slot_map->data;
        for (layer & l : layers) if (l.gpu_bias_map) {
            const int32_t n_expert = (int32_t) l.gpu_bias_map->ne[1];
            std::fill((int32_t *) l.gpu_bias_map->data, (int32_t *) l.gpu_bias_map->data + n_expert, n_expert);
            std::fill((int32_t *) l.cpu_bias_map->data, (int32_t *) l.cpu_bias_map->data + n_expert, n_expert);
        }
        for (device & d : devices) if (d.buffer) ggml_backend_buffer_set_usage(d.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (buf_host) ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (size_t i = 0; i < devices.size(); ++i) {
            const device & d = devices[i];
            if (!d.host_bytes || !d.buffer_size) continue;
            LLAMA_LOG_INFO("%s: %s MoE cache uses %.2f MiB for %.2f MiB of host experts\n", __func__,
                    ggml_backend_buft_name(d.buft), d.buffer_size / 1048576.0, d.host_bytes / 1048576.0);
            for (const group & g : groups) if (g.device == (int32_t) i && g.n_slots) {
                LLAMA_LOG_INFO("%s: %zu layer(s), %d expert slots (%.1f%% coverage)\n", __func__,
                        g.layers.size(), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                if (g.cpu_share_enabled) {
                    LLAMA_LOG_INFO("%s: CPU-share bank rows per layer: real=%d zero=%d guard=1; requested effective share=%.6f\n",
                            __func__, g.n_slots, g.zero_slots, cpu_share_q16 / 65536.0);
                }
            }
        }
        LLAMA_LOG_INFO("%s: active MoE cache policy: %s\n", __func__,
                policy == LLAMA_MOE_CACHE_POLICY_CLASSLRU ? "classlru" : "lru");
    }

    ~impl() { log_stats(); }

    ggml_backend * backend(int32_t il) const {
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) return nullptr;
        return devices[groups[layers[il].group].device].backend;
    }

    bool requires_token_geometry_maps() const { return token_geometry_maps; }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) {
            ++incompatible_layer_skips;
            return nullptr;
        }
        const layer & l = layers[il];
        const group & g = groups[l.group];
        if (n_tokens <= 0 || n_tokens > max_batch) {
            ++large_batch_skips;
            return nullptr;
        }
        if (!llama_moe_cache_policy_detail::slot_map_eligible(n_tokens, n_expert_used,
                l.slot_map->ne[1], g.n_slots, max_batch)) {
            ++incompatible_layer_skips;
            return nullptr;
        }
        return l.slot_map;
    }

    llama_moe_cache::cpu_share_maps get_cpu_share_maps(int32_t il, int64_t n_tokens,
            int64_t n_expert_used) const {
        if (cpu_share_q16 == 0 || il < 0 || (size_t) il >= layers.size()) return {};
        const layer & l = layers[il];
        if (!l.gpu_map || l.group < 0) return {};
        const group & g = groups[l.group];
        if (!g.cpu_share_enabled || n_tokens <= 0 || n_tokens > max_batch ||
                n_expert_used > g.zero_slots ||
                !llama_moe_cache_policy_detail::slot_map_eligible(n_tokens, n_expert_used,
                    l.slot_map->ne[1], g.n_slots, max_batch)) return {};
        return {l.gpu_map, l.cpu_map, l.gpu_bias_map, l.cpu_bias_map};
    }

    void begin_compute() {
        for (layer & l : layers) l.share_planned = false;
    }

    ggml_tensor * get_experts(const ggml_tensor * t) const {
        auto it = bindings.find(t);
        return it == bindings.end() ? nullptr : it->second.cached;
    }

    size_t copy_expert_range(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst,
            int32_t first_expert, size_t max_experts, int64_t n_tokens) {
#ifdef GGML_USE_CUDA
        if (!backend || !src || !dst || !src->buffer || !dst->buffer ||
                !ggml_backend_is_cuda(backend) || n_tokens <= max_batch) return 0;
        auto it_binding = bindings.find(src);
        if (it_binding == bindings.end()) return 0;
        const binding & b = it_binding->second;
        if (b.layer < 0 || (size_t) b.layer >= layers.size()) return 0;
        const layer & l = layers[b.layer];
        if (l.group < 0 || (size_t) l.group >= groups.size() || b.part < 0 ||
                (size_t) b.part >= l.experts.size() || l.experts[b.part] != src) return 0;
        const group & g = groups[l.group];
        if (g.device < 0 || (size_t) g.device >= devices.size() || (size_t) b.part >= g.banks.size()) return 0;
        const device & d = devices[g.device];
        if (backend != d.backend || d.model_device < 0 ||
                d.buft != ggml_backend_cuda_buffer_type(d.model_device) ||
                ggml_backend_get_default_buffer_type(backend) != d.buft ||
                ggml_backend_buffer_get_type(dst->buffer) != d.buft) return 0;
        ggml_tensor * bank = g.banks[b.part];
        if (!bank || !bank->buffer || ggml_backend_buffer_get_type(bank->buffer) != d.buft ||
                !ggml_backend_buffer_is_host(src->buffer) ||
                ggml_backend_buffer_get_usage(src->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) return 0;
        if (src->ne[2] != g.cache.n_expert || src->ne[3] != 1 ||
                first_expert < 0 || first_expert >= g.cache.n_expert || max_experts == 0) return 0;
        const size_t available = (size_t) (g.cache.n_expert - first_expert);
        max_experts = std::min(max_experts, available);

        if (src->type != dst->type || src->ne[0] != dst->ne[0] || src->ne[1] != dst->ne[1] ||
                src->ne[2] != dst->ne[2] || src->ne[3] != dst->ne[3]) return 0;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (src->nb[i] != dst->nb[i]) return 0;
        }
        const size_t row_bytes = ggml_row_size(src->type, src->ne[0]);
        const size_t type_bytes = ggml_type_size(src->type);
        const int64_t block = ggml_blck_size(src->type);
        if (row_bytes == 0 || type_bytes == 0 || block <= 0 || src->ne[0] % block != 0 ||
                src->ne[1] <= 0 || src->ne[2] <= 0 || bank->ne[2] <= 0 ||
                src->nb[0] != type_bytes || src->nb[1] != row_bytes) return 0;
        const size_t expert_size = src->nb[2];
        const size_t bank_extent = ggml_nbytes(bank);
        const size_t dst_extent = ggml_nbytes(dst);
        if (expert_size == 0 || bank->type != src->type ||
                bank->ne[0] != src->ne[0] || bank->ne[1] != src->ne[1] ||
                bank->ne[2] < (int64_t) ((size_t) g.cache.n_slots + (size_t) g.zero_slots + 1) || bank->ne[3] != 1 ||
                bank->nb[0] != src->nb[0] || bank->nb[1] != src->nb[1] || bank->nb[2] != expert_size) return 0;
        if (!src->data || !dst->data || !bank->data || !g.cache.slot_map[b.layer]) return 0;

        const int32_t * slots = g.cache.slot_map[b.layer];
        const int32_t first_slot = slots[first_expert];
        if (first_slot < 0 || first_slot >= g.cache.n_slots) return 0;
        size_t n = 1;
        while (n < max_experts && (int64_t) slots[first_expert + (int32_t) n] == (int64_t) first_slot + n) ++n;
        if ((size_t) first_slot + n > (size_t) g.cache.n_slots) return 0;
        const size_t src_offset = (size_t) first_slot * expert_size;
        const size_t dst_offset = (size_t) first_expert * expert_size;
        const size_t copy_size = n * expert_size;
        if (src_offset > bank_extent || copy_size > bank_extent - src_offset ||
                dst_offset > dst_extent || copy_size > dst_extent - dst_offset ||
                copy_size > (size_t) INT64_MAX) return 0;

        ggml_tensor src_bytes = *bank;
        ggml_tensor dst_bytes = *dst;
        auto set_byte_view = [](ggml_tensor & view, const ggml_tensor & parent, size_t offset, size_t bytes) {
            view = parent;
            view.type = GGML_TYPE_I8;
            view.ne[0] = (int64_t) bytes;
            view.ne[1] = view.ne[2] = view.ne[3] = 1;
            view.nb[0] = 1;
            view.nb[1] = view.nb[2] = view.nb[3] = bytes;
            view.data = (uint8_t *) parent.data + offset;
            view.view_src = nullptr;
        };
        set_byte_view(src_bytes, *bank, src_offset, copy_size);
        set_byte_view(dst_bytes, *dst, dst_offset, copy_size);
        ggml_backend_tensor_copy_async(backend, backend, &src_bytes, &dst_bytes);
        return n;
#else
        GGML_UNUSED(backend);
        GGML_UNUSED(src);
        GGML_UNUSED(dst);
        GGML_UNUSED(first_expert);
        GGML_UNUSED(max_experts);
        GGML_UNUSED(n_tokens);
        return 0;
#endif
    }


    bool copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
            ggml_backend_sched_copy_phase phase) {
        auto it_share = share_map_of.find(src);
        if (it_share != share_map_of.end()) {
            if (phase == GGML_BACKEND_SCHED_COPY_PHASE_QUERY) return true;
            const int32_t il = it_share->second;
            layer & l = layers[il];
            group & g = groups[l.group];
            const device & d = devices[g.device];
            if (!l.share_planned) {
                if (backend != d.backend) {
                    GGML_ABORT("CPU-share map callback ran outside its cache backend for layer %d", il);
                }
                const ggml_tensor * lookup = nullptr;
                const ggml_tensor * repeated_map = nullptr;
                for (int i = 0; i < ggml_graph_n_nodes(graph) && !lookup; ++i) {
                    const ggml_tensor * node = ggml_graph_node(graph, i);
                    if (node->op == GGML_OP_REPEAT && node->src[0] == dst) repeated_map = node;
                    if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) lookup = node;
                    if (node->op == GGML_OP_GET_ROWS && repeated_map && node->src[0] == repeated_map) lookup = node;
                }
                GGML_ASSERT(lookup && "MoE CPU-share map must feed GET_ROWS");
                const ggml_tensor * selected = lookup->src[1];
                auto view_root = [](const ggml_tensor * tensor) {
                    while (tensor && tensor->view_src) tensor = tensor->view_src;
                    return tensor;
                };
                const ggml_tensor * selected_root = view_root(selected);
                for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                    const ggml_tensor * node = ggml_graph_node(graph, i);
                    const bool produces_selected = node == selected_root && node->op != GGML_OP_NONE;
                    const bool is_view = node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE;
                    const bool writes_selected = node != selected_root && !is_view &&
                            view_root(node) == selected_root;
                    if (produces_selected || writes_selected) {
                        GGML_ABORT("MoE experts for layer %d are selected in the same scheduler split as the cache lookup", il);
                    }
                }
                int64_t n_tokens = 0;
                int64_t selected_per_token = 0;
                if (!ggml_is_contiguous(selected) ||
                        !llama_moe_cache_policy_detail::recover_selected_geometry(selected,
                            g.cache.n_expert, selected_per_token, n_tokens) || n_tokens > max_batch) {
                    GGML_ABORT("Unable to recover MoE CPU-share token geometry for layer %d", il);
                }
                const ggml_tensor * selected_storage = selected->view_src ? selected->view_src : selected;
                if (!selected_storage->buffer) GGML_ABORT("CPU-share selected IDs have no backing buffer for layer %d", il);
                ids.resize(ggml_nelements(selected));
                const auto * selected_buft = ggml_backend_buffer_get_type(selected_storage->buffer);
                if (selected_buft == d.buft) {
                    ggml_backend_tensor_get_async(d.backend, selected, ids.data(), 0, ggml_nbytes(selected));
                    ggml_backend_synchronize(d.backend);
                } else if (ggml_backend_buffer_is_host(selected_storage->buffer)) {
                    ggml_backend_tensor_get(selected, ids.data(), 0, ggml_nbytes(selected));
                } else {
                    GGML_ABORT("CPU-share selected IDs use unsupported buffer type for layer %d", il);
                }
                if (!g.cache.plan_with_cpu_share(il, ids.data(), ids.size(), n_tokens, selected_per_token,
                        cpu_share, g.zero_slots, l.share_plan)) {
                    GGML_ABORT("Unable to plan MoE CPU-share assignments for layer %d", il);
                }
                auto store_map = [&](ggml_tensor * tensor, const std::vector<int32_t> & values) {
                    GGML_ASSERT(tensor && values.size() == (size_t) tensor->ne[1]);
                    std::memcpy(tensor->data, values.data(), ggml_nbytes(tensor));
                };
                store_map(l.gpu_map, l.share_plan.gpu_map);
                store_map(l.cpu_map, l.share_plan.cpu_map);
                store_map(l.gpu_bias_map, l.share_plan.gpu_bias_map);
                store_map(l.cpu_bias_map, l.share_plan.cpu_bias_map);
                l.share_planned = true;

                size_t bytes = 0;
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    ggml_tensor * bank = g.banks[ip];
                    const size_t expert_size = w->nb[2];
                    for (size_t i = 0; i < l.share_plan.fills.size();) {
                        size_t n = 1;
                        while (i+n < l.share_plan.fills.size() &&
                                l.share_plan.fills[i+n].expert == l.share_plan.fills[i].expert + (int32_t) n &&
                                l.share_plan.fills[i+n].slot == l.share_plan.fills[i].slot + (int32_t) n) ++n;
                        ggml_backend_tensor_set_async(d.backend, bank,
                                (const uint8_t *) w->data + l.share_plan.fills[i].expert*expert_size,
                                l.share_plan.fills[i].slot*expert_size, n*expert_size);
                        bytes += n*expert_size;
                        i += n;
                    }
                }
                stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
                st.hits += l.share_plan.n_hit;
                st.gpu_fills += l.share_plan.fills.size();
                st.cpu_serviced_unique_misses += l.share_plan.cpu.size();
                st.evictions += l.share_plan.n_eviction;
                st.logical_uploaded_bytes += bytes;
            }
            const ggml_tensor * map = src;
            GGML_ASSERT(ggml_nbytes(map) == ggml_nbytes(dst));
            if (dst != map) ggml_backend_tensor_set_async(backend, dst, map->data, 0, ggml_nbytes(map));
            return true;
        }
        auto it = layer_of.find(src);
        if (it == layer_of.end()) return false;
        if (phase == GGML_BACKEND_SCHED_COPY_PHASE_QUERY) return true;
        const int32_t il = it->second;
        layer & l = layers[il];
        group & g = groups[l.group];
        const device & d = devices[g.device];
        GGML_ASSERT(backend == d.backend);
        const ggml_tensor * lookup = nullptr;
        const ggml_tensor * repeated_map = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(graph) && !lookup; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_REPEAT && node->src[0] == dst) repeated_map = node;
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) lookup = node;
            if (node->op == GGML_OP_GET_ROWS && repeated_map && node->src[0] == repeated_map) lookup = node;
        }
        GGML_ASSERT(lookup && "MoE cache slot map must feed GET_ROWS");
        const ggml_tensor * selected = lookup->src[1];
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == selected || node == selected->view_src) {
                GGML_ABORT("MoE experts for layer %d are selected in the same scheduler split as the cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(selected));
        int64_t n_tokens = 1;
        int64_t selected_per_token = 0;
        if (g.cache.policy == llama_moe_cache_policy_detail::kind::classlru &&
                (!llama_moe_cache_policy_detail::recover_selected_geometry(selected,
                    g.cache.n_expert, selected_per_token, n_tokens) || n_tokens > max_batch)) {
            GGML_ABORT("Unable to recover MoE cache token geometry for layer %d", il);
        }
        ids.resize(ggml_nelements(selected));
        ggml_backend_tensor_get_async(backend, selected, ids.data(), 0, ggml_nbytes(selected));
        ggml_backend_synchronize(backend);
        size_t n_hit = 0;
        size_t n_eviction = 0;
        if (!g.cache.plan(il, ids.data(), ids.size(), n_tokens, fills, n_hit, n_eviction)) GGML_ABORT("MoE cache slot count is smaller than the routed expert set");
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w = l.experts[ip];
            ggml_tensor * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i+n < fills.size() && fills[i+n].expert == fills[i].expert + (int32_t)n &&
                        fills[i+n].slot == fills[i].slot + (int32_t)n) ++n;
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size,
                        fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }
        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits += n_hit;
        st.gpu_fills += fills.size();
        st.evictions += n_eviction;
        st.logical_uploaded_bytes += bytes;
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }

    std::map<ggml_backend_buffer_type *, size_t> memory_breakdown() const {
        std::map<ggml_backend_buffer_type *, size_t> res;
        for (const device & d : devices) {
            if (d.buffer) res[d.buft] += ggml_backend_buffer_get_size(d.buffer.get());
        }
        if (buf_host) res[ggml_backend_buffer_get_type(buf_host.get())] += ggml_backend_buffer_get_size(buf_host.get());
        return res;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & s) {
            const size_t unique_keys = s.hits + s.gpu_fills + s.cpu_serviced_unique_misses;
            if (unique_keys) LLAMA_LOG_INFO("llama_moe_cache: %s hits=%zu gpu_fills=%zu cpu_serviced_unique_misses=%zu unique_keys=%zu unique_key_hit_rate=%.2f%% logical_uploaded_bytes=%zu\n",
                    name, s.hits, s.gpu_fills, s.cpu_serviced_unique_misses, unique_keys,
                    100.0*s.hits/unique_keys, s.logical_uploaded_bytes);
        };
        size_t actual_size = buf_host_size;
        for (const device & d : devices) actual_size += d.buffer_size;
        LLAMA_LOG_INFO("llama_moe_cache: requested %.2f MiB, allocated %.2f MiB\n",
                requested_size / 1048576.0, actual_size / 1048576.0);
        log("ubatch <= 8", stats_small);
        log("ubatch 9-32", stats_large);
        if (large_batch_skips || incompatible_layer_skips) {
            LLAMA_LOG_INFO("llama_moe_cache: skipped %zu large-batch and %zu ineligible layer accesses\n",
                    large_batch_skips, incompatible_layer_skips);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model,
        const std::vector<ggml_backend *> & backends,
        const std::vector<ggml_backend_buffer_type *> & bufts, size_t size,
        enum llama_moe_cache_policy policy, float cpu_share)
        : pimpl(new impl(model, backends, bufts, size, policy, cpu_share)) {}

llama_moe_cache::~llama_moe_cache() = default;
ggml_backend * llama_moe_cache::backend(int32_t il) const { return pimpl->backend(il); }
bool llama_moe_cache::requires_token_geometry_maps() const { return pimpl->requires_token_geometry_maps(); }
ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const { return pimpl->get_slot_map(il, n_tokens, n_expert_used); }
llama_moe_cache::cpu_share_maps llama_moe_cache::get_cpu_share_maps(int32_t il, int64_t n_tokens, int64_t n_expert_used) const { return pimpl->get_cpu_share_maps(il, n_tokens, n_expert_used); }
void llama_moe_cache::begin_compute() { pimpl->begin_compute(); }
ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * t) const { return pimpl->get_experts(t); }
bool llama_moe_cache::copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
        ggml_backend_sched_copy_phase phase) { return pimpl->copy(backend, src, dst, graph, phase); }
size_t llama_moe_cache::copy_expert_range(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst,
        int32_t first_expert, size_t max_experts, int64_t n_tokens) {
    return pimpl->copy_expert_range(backend, src, dst, first_expert, max_experts, n_tokens);
}
int64_t llama_moe_cache::copy_expert_range_min_tokens() const {
    return llama_moe_cache_policy_detail::state::max_batch + 1;
}
std::map<ggml_backend_buffer_type *, size_t> llama_moe_cache::memory_breakdown() const { return pimpl->memory_breakdown(); }
