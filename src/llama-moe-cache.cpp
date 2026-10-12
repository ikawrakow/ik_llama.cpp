#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache-share.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace {

struct moe_cache_lru {
    struct fill { int32_t expert; int32_t slot; };

    int32_t n_expert = 0;
    int32_t n_slots = 0;
    std::vector<int32_t *> slot_map;
    std::vector<int32_t> key_of;
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    std::vector<uint32_t> seen;
    std::vector<int32_t> uniq;
    uint32_t seen_gen = 0;
    int32_t head = -1;
    int32_t tail = -1;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
    }

    void touch(int32_t s) {
        if (s == tail) return;
        if (prev[s] >= 0) next[prev[s]] = next[s]; else head = next[s];
        prev[next[s]] = prev[s];
        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit, size_t & n_eviction) {
        fills.clear();
        n_hit = 0;
        n_eviction = 0;
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) return false;
        int32_t * slots = slot_map[il];
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                ++n_hit;
            }
        }
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) continue;
            const int32_t s = head;
            if (key_of[s] >= 0) {
                const int32_t old = key_of[s];
                slot_map[old / n_expert][old % n_expert] = -1;
                ++n_eviction;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s });
        }
        return true;
    }
};

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
        moe_cache_lru lru;
    };
    struct layer {
        int32_t group = -1;
        ggml_tensor * slot_map = nullptr;
        std::vector<ggml_tensor *> experts;
    };
    struct binding {
        int32_t layer;
        int32_t part;
        ggml_tensor * cached;
    };
    struct stats { size_t hits = 0; size_t misses = 0; size_t evictions = 0; size_t bytes = 0; };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;
    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings;
    std::unordered_map<const ggml_tensor *, int32_t> layer_of;
    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;
    stats stats_small, stats_large;
    size_t requested_size;
    mutable size_t large_batch_skips = 0;
    mutable size_t incompatible_layer_skips = 0;

    impl(const llama_model & model, const std::vector<ggml_backend *> & backends,
            const std::vector<ggml_backend_buffer_type *> & bufts, size_t size)
            : n_expert_used(model.hparams.n_expert_used), layers(model.layers.size()), requested_size(size) {
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
        auto alloc_size = [&](const group & g, int32_t slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.device].buft);
            size_t total = 0;
            for (const ggml_tensor * t : g.ref) {
                const size_t bytes = t->nb[2] * (size_t) (slots + 1);
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
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) ++g.n_slots;
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: cache budget leaves %zu expert layers uncached because they cannot hold one token\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init((int32_t) layers.size(), n_expert, g.n_slots);
            device_tensor_counts[g.device] += g.ref.size() * (1 + g.layers.size());
            host_layer_count += g.layers.size();
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
                ggml_tensor * bank = ggml_new_tensor_3d(d.ctx.get(), t->type, t->ne[0], t->ne[1], g.n_slots + 1);
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
                    ggml_tensor * cached = ggml_view_3d(d.ctx.get(), bank, bank->ne[0], bank->ne[1], g.n_slots,
                            bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            d.buffer_size += alloc_size(g, g.n_slots);
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
        for (group & g : groups) for (int32_t il : g.layers) g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
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
            }
        }
    }

    ~impl() { log_stats(); }

    ggml_backend * backend(int32_t il) const {
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) return nullptr;
        return devices[groups[layers[il].group].device].backend;
    }

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
        if (std::min<int64_t>(n_tokens*n_expert_used, l.slot_map->ne[1]) > g.n_slots) {
            ++incompatible_layer_skips;
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * t) const {
        auto it = bindings.find(t);
        return it == bindings.end() ? nullptr : it->second.cached;
    }

    bool copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
            ggml_backend_sched_copy_phase phase) {
        auto it = layer_of.find(src);
        if (it == layer_of.end()) return false;
        if (phase == GGML_BACKEND_SCHED_COPY_PHASE_QUERY) return true;
        const int32_t il = it->second;
        layer & l = layers[il];
        group & g = groups[l.group];
        const device & d = devices[g.device];
        GGML_ASSERT(backend == d.backend);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(graph) && !lookup; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) lookup = node;
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
        ids.resize(ggml_nelements(selected));
        ggml_backend_tensor_get_async(backend, selected, ids.data(), 0, ggml_nbytes(selected));
        ggml_backend_synchronize(backend);
        size_t n_hit = 0;
        size_t n_eviction = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit, n_eviction)) GGML_ABORT("MoE cache slot count is smaller than the routed expert set");
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
        st.misses += fills.size();
        st.evictions += n_eviction;
        st.bytes += bytes;
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & s) {
            const size_t n = s.hits + s.misses;
            if (n) LLAMA_LOG_INFO("llama_moe_cache: %s hits=%zu misses=%zu evictions=%zu hit-rate=%.2f%% uploaded=%.2f MiB\n",
                    name, s.hits, s.misses, s.evictions, 100.0*s.hits/n, s.bytes/1048576.0);
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
        const std::vector<ggml_backend_buffer_type *> & bufts, size_t size)
        : pimpl(new impl(model, backends, bufts, size)) {}

llama_moe_cache::~llama_moe_cache() = default;
ggml_backend * llama_moe_cache::backend(int32_t il) const { return pimpl->backend(il); }
ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const { return pimpl->get_slot_map(il, n_tokens, n_expert_used); }
ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * t) const { return pimpl->get_experts(t); }
bool llama_moe_cache::copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
        ggml_backend_sched_copy_phase phase) { return pimpl->copy(backend, src, dst, graph, phase); }
