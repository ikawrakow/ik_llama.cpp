#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache-share.h"
#include "llama-moe-cache-residency.h"

#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "ggml-moe-provider.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace {

bool backend_is_cuda(ggml_backend * backend) {
#ifdef GGML_USE_CUDA
    return ggml_backend_is_cuda(backend);
#else
    (void) backend;
    return false;
#endif
}

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

static enum ggml_unary_op moe_fused_unary(const ggml_tensor * op) {
    int32_t value = 0;
    memcpy(&value, op->op_params, sizeof(value));
    return (enum ggml_unary_op) value;
}

} // namespace

struct llama_moe_cache::impl {
    enum class fill_enqueue_result { accepted, coalesced, unavailable, invalid, occupied,
        no_free_slot, job_limit, byte_limit, no_victim };
    struct binding {
        int32_t layer;
        int32_t part;
        ggml_tensor * cached;
    };
    struct provider_row {
        size_t row = 0;
        int32_t slot = -1;
    };
    struct provider_request {
        binding weight = { -1, -1, nullptr };
        binding second_weight = { -1, -1, nullptr };
        const ggml_tensor * op = nullptr;
        std::array<provider_row, GGML_MOE_PROVIDER_MAX_ROWS> rows{};
        std::array<llama_moe_cache_residency::key, GGML_MOE_PROVIDER_MAX_ROWS> route_keys{};
        std::array<int32_t, GGML_MOE_PROVIDER_MAX_ROWS> pinned_slots{};
        std::array<llama_moe_cache_residency::pin, GGML_MOE_PROVIDER_MAX_ROWS> pins{};
        std::array<llama_moe_cache_residency::key, GGML_MOE_PROVIDER_MAX_ROWS> miss_candidates{};
        size_t n_rows = 0;
        size_t n_hit = 0;
        size_t n_pins = 0;
        size_t n_valid = 0;
        size_t n_miss_candidates = 0;
        size_t n_unique_activations = 0;
        std::array<int32_t, GGML_MOE_PROVIDER_MAX_ROWS> activation_map{};
        std::array<const float *, GGML_MOE_PROVIDER_MAX_ROWS> unique_activations{};
        bool fused = false;
        bool active = false;
        bool submitted = false;
        bool completed = false;
        bool admission_plan_valid = false;
        bool admission_driver = false;
    };
    struct direct_workspace {
        ggml_backend_buffer_ptr device_buffer;
        ggml_backend_buffer_ptr staging_buffer;
        ggml_backend_event_t event = nullptr;
        size_t input_capacity = 0;
        size_t q8_capacity = 0;
        size_t output_capacity = 0;
    };
    struct direct_buffers {
        uint8_t * host = nullptr;
        uint8_t * device = nullptr;
        uint8_t * device_q8 = nullptr;
        float * device_output = nullptr;
        float * host_output = nullptr;
        size_t input_bytes = 0;
        size_t output_bytes = 0;
    };
    static constexpr size_t max_fill_jobs = 128;
    static constexpr size_t max_fill_parts = 4;
    static constexpr size_t max_fill_bytes = 512u * 1024u * 1024u;
    struct fill_part {
        ggml_tensor * source = nullptr;
        ggml_tensor * destination = nullptr;
        size_t source_offset = 0;
        size_t destination_offset = 0;
        size_t bytes = 0;
    };
    struct fill_job {
        int32_t group = -1;
        int32_t slot = -1;
        llama_moe_cache_residency::key owner;
        uint64_t generation = 0;
        std::array<fill_part, max_fill_parts> parts{};
        size_t n_parts = 0;
        size_t bytes = 0;
    };
    struct fill_counters {
        size_t accepted = 0;
        size_t completed = 0;
        size_t failures = 0;
    };
    struct group_admission_state {
        std::mutex mutex;
        std::vector<uint8_t> demand;
        std::vector<int32_t> lru_prev;
        std::vector<int32_t> lru_next;
        std::vector<uint8_t> lru_member;
        int32_t lru_head = -1;
        int32_t lru_tail = -1;
        void initialize(size_t n_keys, size_t n_slots) {
            demand.assign(n_keys, 0);
            lru_prev.assign(n_slots, -1);
            lru_next.assign(n_slots, -1);
            lru_member.assign(n_slots, 0);
        }
        void remove_lru(int32_t slot) {
            if (slot < 0 || (size_t)slot >= lru_member.size() || !lru_member[slot]) return;
            const int32_t prev = lru_prev[slot], next = lru_next[slot];
            if (prev >= 0) lru_next[prev] = next; else lru_head = next;
            if (next >= 0) lru_prev[next] = prev; else lru_tail = prev;
            lru_prev[slot] = lru_next[slot] = -1;
            lru_member[slot] = 0;
        }
        void insert_mru(int32_t slot) {
            if (slot < 0 || (size_t)slot >= lru_member.size()) return;
            remove_lru(slot);
            lru_prev[slot] = lru_tail;
            lru_next[slot] = -1;
            if (lru_tail >= 0) lru_next[lru_tail] = slot; else lru_head = slot;
            lru_tail = slot;
            lru_member[slot] = 1;
        }
        void touch_lru(int32_t slot) {
            if (slot < 0 || (size_t)slot >= lru_member.size() || !lru_member[slot]) return;
            if (lru_tail != slot) insert_mru(slot);
        }
    };
    struct fill_engine {
        ggml_backend * backend = nullptr;
        std::thread worker;
        std::mutex mutex;
        std::condition_variable changed;
        std::array<fill_job, max_fill_jobs> queue{};
        size_t head = 0;
        size_t queued = 0;
        size_t outstanding = 0;
        size_t outstanding_bytes = 0;
        bool stopping = false;
        bool inflight = false;
        bool inflight_cancelled = false;
        bool publishing = false;
        fill_job inflight_job;
        fill_counters counters;
    };
    struct device {
        ggml_backend * backend = nullptr;
        ggml_backend_buffer_type * buft = nullptr;
        int32_t model_device = -1;
        int32_t cuda_ordinal = -1;
        double split = 0.0;
        size_t host_bytes = 0;
        size_t buffer_size = 0;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
        ggml_backend * provider_backend = nullptr;
        std::unique_ptr<fill_engine> fills;
        provider_request request;
        direct_workspace direct;
    };
    struct group {
        int32_t device = -1;
        std::vector<ggml_tensor *> ref;
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks;
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
        std::unique_ptr<llama_moe_cache_residency> residency;
        std::unique_ptr<group_admission_state> admission;
    };
    struct layer {
        int32_t group = -1;
        int32_t admission_part = -1;
        ggml_tensor * slot_map = nullptr;
        std::vector<ggml_tensor *> experts;
    };
    struct stats { size_t hits = 0; size_t misses = 0; size_t evictions = 0; size_t bytes = 0; };


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
    bool async_mode;
    struct ggml_moe_provider moe_provider = {};
    std::atomic_size_t active_provider_requests{0};
    struct async_runtime_stats { size_t operations = 0, eligible_rows = 0, committed_gpu_rows = 0; } async_stats;
    std::mutex fill_enqueue_mutex;

    static bool provider_begin(void * user_data, const ggml_tensor * op, void ** request);
    static bool provider_plan(void * user_data, void * request, const ggml_tensor * op, uint8_t * gpu_rows, size_t n_rows);
    static bool provider_dispatch(void * user_data, void * request, ggml_tensor * op, const uint8_t * gpu_rows, size_t n_rows);
    static bool provider_collect(void * user_data, void * request, ggml_tensor * op, const uint8_t * gpu_rows, size_t n_rows);
    static void provider_end(void * user_data, void * request);
    void touch_ready_pins_locked(const group & g, const llama_moe_cache_residency::pin * pins, size_t n_pins);
    bool ensure_direct_workspace(device & d, size_t input_bytes, size_t q8_bytes, size_t output_bytes);
    bool prepare_direct_inputs(provider_request & req, const ggml_tensor * source,
            const ggml_tensor * ids, size_t k, size_t m, int64_t padded_k, direct_buffers & buffers);
    impl(const llama_model & model, const std::vector<ggml_backend *> & backends,
            const std::vector<ggml_backend_buffer_type *> & bufts, size_t size, bool async_mode)
            : n_expert_used(model.hparams.n_expert_used), layers(model.layers.size()), requested_size(size), async_mode(async_mode) {
        moe_provider.user_data = this;
        moe_provider.begin = provider_begin;
        moe_provider.plan = provider_plan;
        moe_provider.dispatch = provider_dispatch;
        moe_provider.collect = provider_collect;
        moe_provider.end = provider_end;

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
                if (backend_is_cuda(d.backend)) d.cuda_ordinal = cuda_ordinal_for_buft(d.buft);
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
            const auto * layer_buft = model.default_buffer_type_offload(model.devices[model_device]);
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
            if (async_mode) throw std::runtime_error("asynchronous MoE cache requires host-resident eligible expert weights");
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

        size_t cache_layer_count = 0;
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
            if (!async_mode) g.lru.init((int32_t) layers.size(), n_expert, g.n_slots);
            if (async_mode) {
                g.residency = std::make_unique<llama_moe_cache_residency>(
                        (size_t) g.n_slots, layers.size(), (size_t) n_expert);
                if ((size_t) n_expert > SIZE_MAX / layers.size()) {
                    throw std::runtime_error("MoE admission key set is too large");
                }
                g.admission = std::make_unique<group_admission_state>();
                g.admission->initialize(layers.size() * (size_t) n_expert, (size_t)g.n_slots);
            }
            device_tensor_counts[g.device] += g.ref.size() * (1 + (async_mode ? 0 : g.layers.size()));
            cache_layer_count += g.layers.size();
        }
        if (cache_layer_count == 0) throw std::runtime_error("MoE cache budget is too small to hold one token");

        auto make_ctx = [](size_t n_tensors) {
            ggml_init_params params = { n_tensors * ggml_tensor_overhead(), nullptr, true };
            ggml_context_ptr ctx(ggml_init(params));
            if (!ctx) throw std::runtime_error("failed to create MoE cache tensor metadata");
            return ctx;
        };
        for (size_t id = 0; id < devices.size(); ++id) if (device_tensor_counts[id]) devices[id].ctx = make_ctx(device_tensor_counts[id]);
        const auto buft_host = ggml_backend_cpu_buffer_type();
        size_t alignment_host = 0;
        if (!async_mode) {
            ctx_host = make_ctx(cache_layer_count);
            alignment_host = ggml_backend_buft_get_alignment(buft_host);
        }

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
                    ggml_tensor * cached = nullptr;
                    if (!async_mode) {
                        ggml_tensor * bank = g.banks[ip];
                        cached = ggml_view_3d(d.ctx.get(), bank, bank->ne[0], bank->ne[1], g.n_slots,
                                bank->nb[1], bank->nb[2], 0);
                        ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    }
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                    if (l.admission_part < 0) l.admission_part = bindings[l.experts[ip]].part;
                }
                if (!async_mode) {
                    l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                    layer_of[l.slot_map] = il;
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                }
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
        if (!async_mode) {
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) throw std::runtime_error("failed to allocate MoE cache slot maps");
            ggml_backend_buffer_clear(buf_host.get(), 0xff);
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
            for (group & g : groups) for (int32_t il : g.layers) g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
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
            }
        }
        if (async_mode) {
            for (size_t i = 0; i < devices.size(); ++i) {
                device & d = devices[i];
                const bool has_admission_group = std::any_of(groups.begin(), groups.end(), [&](const group & g) {
                    return g.device == (int32_t) i && g.n_slots > 0 && g.admission;
                });
                if (!has_admission_group) continue;
                if (!ensure_fill_engine(i)) {
                    LLAMA_LOG_WARN("%s: live MoE fill admission is disabled for CUDA device %d because setup failed\n",
                            __func__, d.cuda_ordinal);
                }
            }
        }
    }

    static int32_t cuda_ordinal_for_buft(ggml_backend_buffer_type * buft) {
#ifdef GGML_USE_CUDA
        for (int32_t i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            if (ggml_backend_cuda_buffer_type(i) == buft) return i;
        }
#else
        (void) buft;
#endif
        return -1;
    }

    void set_expert_bank_async(device & d, ggml_tensor * bank, const void * src, size_t offset, size_t size) {
        ggml_backend_tensor_set_async(d.backend, bank, src, offset, size);
    }

    bool ensure_fill_engine(size_t device_id) {
        if (device_id >= devices.size()) return false;
        device & d = devices[device_id];
        if (d.fills) return true;
#ifndef GGML_USE_CUDA
        return false;
#else
        if (d.cuda_ordinal < 0) return false;
        std::unique_ptr<fill_engine> engine(new (std::nothrow) fill_engine());
        if (!engine) return false;
        engine->backend = ggml_backend_cuda_init(d.cuda_ordinal, nullptr, nullptr);
        if (!engine->backend) return false;
        d.fills = std::move(engine);
        try {
            d.fills->worker = std::thread([this, device_id] { fill_worker(device_id); });
        } catch (...) {
            ggml_backend_free(d.fills->backend);
            d.fills.reset();
            return false;
        }
        return true;
#endif
    }

    void fill_worker(size_t device_id) {
        device & d = devices[device_id];
        fill_engine & engine = *d.fills;
        for (;;) {
            fill_job job;
            {
                std::unique_lock<std::mutex> lock(engine.mutex);
                engine.changed.wait(lock, [&] { return engine.stopping || engine.queued != 0; });
                if (engine.stopping && engine.queued == 0) break;
                job = engine.queue[engine.head];
                engine.queue[engine.head] = {};
                engine.head = (engine.head + 1) % max_fill_jobs;
                --engine.queued;
                engine.inflight = true;
                engine.inflight_cancelled = false;
                engine.publishing = false;
                engine.inflight_job = job;
                engine.changed.notify_all();
            }

            bool failed = false;
            size_t submitted_bytes = 0;
            if (!failed) {
                for (size_t ip = 0; ip < job.n_parts; ++ip) {
                    const fill_part & part = job.parts[ip];
                    if (!part.source || !part.destination || !part.source->data || !part.destination->data ||
                            part.source_offset > ggml_nbytes(part.source) ||
                            part.bytes > ggml_nbytes(part.source) - part.source_offset ||
                            part.destination_offset > ggml_nbytes(part.destination) ||
                            part.bytes > ggml_nbytes(part.destination) - part.destination_offset) {
                        failed = true;
                        break;
                    }
#ifdef GGML_USE_CUDA
                    if (!ggml_cuda_moe_cache_try_tensor_set_async(engine.backend, part.destination,
                                (const uint8_t *) part.source->data + part.source_offset,
                                part.destination_offset, part.bytes)) {
                        failed = true;
                        break;
                    }
#else
                    failed = true;
                    break;
#endif
                    submitted_bytes += part.bytes;
                }
            }

            bool cancelled = false;
            if (!failed) {
                std::unique_lock<std::mutex> lock(engine.mutex);
                cancelled = engine.inflight_cancelled || engine.stopping;
            }

            if (submitted_bytes != 0) {
#ifdef GGML_USE_CUDA
                if (!ggml_cuda_moe_cache_try_synchronize(engine.backend)) failed = true;
#else
                failed = true;
#endif
            }
            if (!failed && !cancelled) {
                std::lock_guard<std::mutex> lock(engine.mutex);
                cancelled = engine.inflight_cancelled || engine.stopping;
                engine.publishing = !cancelled;
            }

            group & g = groups[job.group];
            bool published = false;
            if (failed) {
                g.residency->fail((size_t) job.slot, job.owner, job.generation);
            } else if (cancelled) {
                g.residency->cancel((size_t) job.slot, job.owner, job.generation);
            } else {
                GGML_ASSERT(g.admission);
                std::lock_guard<std::mutex> lock(g.admission->mutex);
                published = g.residency->complete((size_t) job.slot, job.owner, job.generation);
                if (published) {
                    const size_t key_index = (size_t) job.owner.layer*(size_t) g.ref[0]->ne[2] +
                            (size_t) job.owner.expert;
                    g.admission->demand[key_index] = 0;
                    g.admission->insert_mru(job.slot);
                }
            }

            {
                std::lock_guard<std::mutex> lock(engine.mutex);
                if (failed || cancelled || !published) {
                    ++engine.counters.failures;
                } else {
                    ++engine.counters.completed;
                }
                GGML_ASSERT(engine.outstanding > 0 && engine.outstanding_bytes >= job.bytes);
                --engine.outstanding;
                engine.outstanding_bytes -= job.bytes;
                engine.inflight = false;
                engine.inflight_cancelled = false;
                engine.publishing = false;
                engine.inflight_job = {};
                engine.changed.notify_all();
            }
        }
    }

    void stop_fill_engine(device & d) {
        if (!d.fills) return;
        fill_engine & engine = *d.fills;
        std::array<fill_job, max_fill_jobs> cancelled{};
        size_t n_cancelled = 0;
        fill_job inflight_job;
        bool retire_inflight = false;
        {
            std::lock_guard<std::mutex> lock(engine.mutex);
            engine.stopping = true;
            while (engine.queued != 0) {
                cancelled[n_cancelled] = engine.queue[engine.head];
                engine.queue[engine.head] = {};
                engine.head = (engine.head + 1) % max_fill_jobs;
                --engine.queued;
                --engine.outstanding;
                engine.outstanding_bytes -= cancelled[n_cancelled].bytes;
                ++engine.counters.failures;
                ++n_cancelled;
            }
            if (engine.inflight && !engine.publishing) {
                engine.inflight_cancelled = true;
                inflight_job = engine.inflight_job;
                retire_inflight = true;
            }
            engine.changed.notify_all();
        }
        for (size_t i = 0; i < n_cancelled; ++i) {
            const fill_job & job = cancelled[i];
            groups[job.group].residency->cancel((size_t) job.slot, job.owner, job.generation);
        }
        if (retire_inflight) {
            groups[inflight_job.group].residency->retire(
                    (size_t) inflight_job.slot, inflight_job.owner, inflight_job.generation);
        }
        if (engine.worker.joinable()) engine.worker.join();
        GGML_ASSERT(engine.outstanding == 0 && engine.queued == 0 && !engine.inflight);
    }

    fill_enqueue_result build_fill_job(int32_t group_id, int32_t il, int32_t expert,
            int32_t slot, fill_job & job) {
        using result = fill_enqueue_result;
        if (group_id < 0 || (size_t)group_id >= groups.size() ||
                il < 0 || (size_t)il >= layers.size() || expert < 0 || slot < -1) return result::invalid;
        const layer & l = layers[il];
        const group & g = groups[group_id];
        if (l.group != group_id || l.experts.empty()) return result::unavailable;
        if (expert >= g.ref[0]->ne[2] || (slot >= 0 && slot >= g.n_slots)) return result::invalid;
        if (l.experts.size() != g.banks.size() || l.experts.size() > max_fill_parts) return result::unavailable;

        job = {};
        job.group = group_id;
        job.slot = slot;
        job.owner = { il, expert };
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            ggml_tensor * src = l.experts[ip];
            ggml_tensor * dst = g.banks[ip];
            if (!src || !dst || !src->data || !dst->data || src->ne[2] != g.ref[0]->ne[2] ||
                    src->nb[2] == 0 || src->nb[2] != dst->nb[2] ||
                    (size_t)expert > SIZE_MAX/src->nb[2] ||
                    (slot >= 0 && (size_t)slot > SIZE_MAX/dst->nb[2]) ||
                    (slot < 0 && (size_t)g.n_slots > SIZE_MAX/dst->nb[2])) return result::invalid;
            fill_part & part = job.parts[job.n_parts++];
            part.source = src;
            part.destination = dst;
            part.source_offset = (size_t)expert*src->nb[2];
            part.destination_offset = slot < 0 ? 0 : (size_t)slot*dst->nb[2];
            part.bytes = src->nb[2];
            if (part.bytes > SIZE_MAX-job.bytes) return result::invalid;
            job.bytes += part.bytes;
        }
        return job.n_parts != 0 && job.bytes != 0 ? result::accepted : result::invalid;
    }

    fill_enqueue_result check_fill_queue_locked(fill_engine & engine, size_t bytes) {
        using result = fill_enqueue_result;
        if (engine.stopping) return result::unavailable;
        if (engine.outstanding >= max_fill_jobs) {
            return result::job_limit;
        }
        if (bytes > max_fill_bytes || engine.outstanding_bytes > max_fill_bytes-bytes) {
            return result::byte_limit;
        }
        return result::accepted;
    }

    void publish_fill_locked(fill_engine & engine, const fill_job & job) {
        const size_t tail = (engine.head+engine.queued)%max_fill_jobs;
        engine.queue[tail] = job;
        ++engine.queued;
        ++engine.outstanding;
        engine.outstanding_bytes += job.bytes;
        ++engine.counters.accepted;
        engine.changed.notify_all();
    }

    fill_enqueue_result resolve_fill_target(int32_t il, int32_t expert,
            layer *& l, group *& g, device *& d) {
        using result = fill_enqueue_result;
        l = nullptr;
        g = nullptr;
        d = nullptr;
        if (!async_mode || il < 0 || (size_t) il >= layers.size() || expert < 0) return result::invalid;
        l = &layers[il];
        if (l->group < 0 || l->experts.empty()) return result::unavailable;
        g = &groups[l->group];
        if (!g->residency || (size_t) expert >= (size_t) g->ref[0]->ne[2]) return result::invalid;
        d = &devices[g->device];
        if (!backend_is_cuda(d->backend) || d->cuda_ordinal < 0 || !d->fills ||
                l->experts.size() != g->banks.size() || l->experts.size() > max_fill_parts) return result::unavailable;
        return result::accepted;
    }

    fill_enqueue_result coalesce_fill_owner(group & g, fill_engine & engine,
            llama_moe_cache_residency::key owner) {
        using result = fill_enqueue_result;
        const auto existing = g.residency->find(owner);
        if (existing.state == llama_moe_cache_residency::slot_state::ready ||
                existing.state == llama_moe_cache_residency::slot_state::copying) {
            std::lock_guard<std::mutex> lock(engine.mutex);
            return result::coalesced;
        }
        return existing.state == llama_moe_cache_residency::slot_state::free ? result::accepted : result::occupied;
    }

    fill_enqueue_result prepare_fill_job(int32_t group_id, int32_t il, int32_t expert,
            int32_t slot, fill_job & job) {
        const auto result = build_fill_job(group_id, il, expert, slot, job);
        if (result != fill_enqueue_result::accepted) return result;
        return job.bytes <= max_fill_bytes ? result : fill_enqueue_result::byte_limit;
    }

    fill_enqueue_result queue_reserved_fill(group & g, fill_engine & engine, const fill_job & job) {
        fill_enqueue_result status;
        {
            std::lock_guard<std::mutex> lock(engine.mutex);
            status = check_fill_queue_locked(engine, job.bytes);
            if (status == fill_enqueue_result::accepted) publish_fill_locked(engine, job);
        }
        if (status != fill_enqueue_result::accepted) {
            g.residency->cancel((size_t) job.slot, job.owner, job.generation);
        }
        return status;
    }

    fill_enqueue_result enqueue_fill_locked(int32_t il, int32_t expert) {
        using result = fill_enqueue_result;
        layer * l;
        group * g;
        device * d;
        const auto resolved = resolve_fill_target(il, expert, l, g, d);
        if (resolved != result::accepted) return resolved;
        fill_engine & engine = *d->fills;
        const llama_moe_cache_residency::key owner = { il, expert };
        const auto state = coalesce_fill_owner(*g, engine, owner);
        if (state != result::accepted) return state;

        int32_t slot = -1;
        for (int32_t candidate = 0; candidate < g->n_slots; ++candidate) {
            if (g->residency->reusable((size_t) candidate)) {
                slot = candidate;
                break;
            }
        }
        if (slot < 0) return result::no_free_slot;

        fill_job job;
        const auto prepared = prepare_fill_job(l->group, il, expert, slot, job);
        if (prepared != result::accepted) return prepared;
        if (!g->residency->reserve((size_t) slot, owner, job.generation)) {
            const auto current = coalesce_fill_owner(*g, engine, owner);
            return current == result::accepted ? result::occupied : current;
        }
        return queue_reserved_fill(*g, engine, job);
    }

    fill_enqueue_result enqueue_free_slot(int32_t il, int32_t expert) {
        std::unique_lock<std::mutex> lock(fill_enqueue_mutex, std::try_to_lock);
        if (!lock.owns_lock()) return fill_enqueue_result::unavailable;
        return enqueue_fill_locked(il, expert);
    }

    fill_enqueue_result enqueue_replacement(int32_t il, int32_t expert,
            const int32_t * excluded_slots, size_t n_excluded_slots, int32_t * accepted_slot) {
        using result = fill_enqueue_result;
        if (accepted_slot) *accepted_slot = -1;
        // Lock order: enqueue -> admission -> fill queue -> residency.
        std::unique_lock<std::mutex> enqueue_lock(fill_enqueue_mutex, std::try_to_lock);
        if (!enqueue_lock.owns_lock()) return result::unavailable;
        layer * l;
        group * g;
        device * d;
        const auto resolved = resolve_fill_target(il, expert, l, g, d);
        if (resolved != result::accepted) return resolved;
        if (!g->admission) return result::unavailable;

        fill_job job;
        const auto prepared = prepare_fill_job(l->group, il, expert, -1, job);
        if (prepared != result::accepted) return prepared;

        group_admission_state & admission = *g->admission;
        std::lock_guard<std::mutex> admission_lock(admission.mutex);
        fill_engine & engine = *d->fills;
        const auto state = coalesce_fill_owner(*g, engine, job.owner);
        if (state != result::accepted) return state;

        for (int32_t slot = 0; slot < g->n_slots; ++slot) {
            if (g->residency->reusable((size_t)slot)) return result::no_free_slot;
        }

        std::lock_guard<std::mutex> queue_lock(engine.mutex);
        const result queue_status = check_fill_queue_locked(engine, job.bytes);
        if (queue_status != result::accepted) return queue_status;

        int32_t victim_slot = admission.lru_head;
        uint64_t new_generation = 0;
        llama_moe_cache_residency::snapshot victim;
        for (int32_t slot = admission.lru_head; slot >= 0; slot = admission.lru_next[slot]) {
            bool excluded = false;
            for (size_t i = 0; excluded_slots && i < n_excluded_slots; ++i) excluded |= excluded_slots[i] == slot;
            if (excluded) continue;
            const auto candidate = g->residency->inspect((size_t)slot);
            if (candidate.state != llama_moe_cache_residency::slot_state::ready || candidate.pins != 0 ||
                    candidate.slot != slot || candidate.owner.layer < 0 || candidate.owner.expert < 0 ||
                    (size_t) slot >= admission.lru_member.size() || !admission.lru_member[slot]) continue;
            if (!g->residency->replace_ready((size_t)slot, candidate.owner, candidate.generation,
                        job.owner, new_generation)) continue;
            victim_slot = slot;
            victim = candidate;
            break;
        }
        if (victim.generation == 0) {
            return result::no_victim;
        }

        admission.remove_lru(victim_slot);
        job.slot = victim_slot;
        job.generation = new_generation;
        if (accepted_slot) *accepted_slot = victim_slot;
        for (size_t ip = 0; ip < job.n_parts; ++ip) {
            job.parts[ip].destination_offset = (size_t)victim_slot*job.parts[ip].destination->nb[2];
        }
        publish_fill_locked(engine, job);
        return result::accepted;
    }

    void rollback_miss_demand(group & g, llama_moe_cache_residency::key key,
            size_t demand_index, bool incremented) {
        group_admission_state & admission = *g.admission;
        std::lock_guard<std::mutex> lock(admission.mutex);
        const auto existing = g.residency->find(key);
        if (existing.state == llama_moe_cache_residency::slot_state::ready) {
            admission.demand[demand_index] = 0;
        } else if (incremented && admission.demand[demand_index] != 0) {
            --admission.demand[demand_index];
        }
    }

    void admit_provider_misses(provider_request & req) {
        if (!async_mode || !req.admission_plan_valid) return;
        const int32_t il = req.weight.layer;
        if (il < 0 || (size_t) il >= layers.size()) return;
        group & g = groups[layers[il].group];
        if (!g.admission) return;
        group_admission_state & admission = *g.admission;

        const size_t n_experts = (size_t) g.ref[0]->ne[2];
        const size_t full_key_count = g.layers.size() * n_experts;
        const bool first_miss_admits = (size_t) g.n_slots >= full_key_count;
        size_t accepted_jobs = 0;
        std::array<int32_t, 8> replacement_slots{};
        size_t n_replacement_slots = 0;

        for (size_t i = 0; i < req.n_miss_candidates; ++i) {
            const auto key = req.miss_candidates[i];
            if (key.layer < 0 || (size_t) key.layer >= layers.size() || key.expert < 0 ||
                    (size_t) key.expert >= n_experts) continue;
            const size_t demand_index = (size_t) key.layer*n_experts + (size_t) key.expert;
            if (demand_index >= admission.demand.size()) continue;

            bool has_free_slot = false;
            bool demand_incremented = false;
            {
                std::lock_guard<std::mutex> lock(admission.mutex);
                if (!req.admission_driver) continue;
                // Admission and publication share this lock, keeping READY/COPYING stable here
                const auto existing = g.residency->find(key);
                if (existing.state == llama_moe_cache_residency::slot_state::ready) {
                    admission.demand[demand_index] = 0;
                    continue;
                }
                if (existing.state == llama_moe_cache_residency::slot_state::copying) continue;

                uint8_t demand = admission.demand[demand_index];
                if (demand < 8) {
                    ++demand;
                    demand_incremented = true;
                }
                admission.demand[demand_index] = demand;
                for (int32_t slot = 0; slot < g.n_slots; ++slot) {
                    if (g.residency->reusable((size_t)slot)) {
                        has_free_slot = true;
                        break;
                    }
                }
                const uint8_t threshold = has_free_slot ? (first_miss_admits ? 1 : 2) : 8;
                if (demand < threshold) {
                    continue;
                }
            }

            if (accepted_jobs >= 8) {
                continue;
            }

            int32_t replacement_slot = -1;
            const auto result = has_free_slot ? enqueue_free_slot(key.layer, key.expert) :
                    enqueue_replacement(key.layer, key.expert, replacement_slots.data(),
                            n_replacement_slots, &replacement_slot);
            if (result == fill_enqueue_result::accepted) {
                ++accepted_jobs;
                if (!has_free_slot && replacement_slot >= 0 && n_replacement_slots < replacement_slots.size()) {
                    replacement_slots[n_replacement_slots++] = replacement_slot;
                }
            } else if (result == fill_enqueue_result::coalesced) {
                // A concurrent enqueue won, undo only this observation.
                rollback_miss_demand(g, key, demand_index, demand_incremented);
            }
        }
    }

    ~impl() {
        for (device & d : devices) stop_fill_engine(d);
        GGML_ASSERT(active_provider_requests.load(std::memory_order_acquire) == 0);
        for (const group & g : groups) {
            GGML_ASSERT(!g.residency || g.residency->live_pins() == 0);
        }
        log_stats();
        for (device & d : devices) {
            if (d.request.submitted && !d.request.completed && d.direct.event) {
                ggml_backend_event_synchronize(d.direct.event);
                d.request.completed = true;
            }
            if (d.direct.event) ggml_backend_event_free(d.direct.event);
            d.direct.event = nullptr;
            d.direct.staging_buffer.reset();
            d.direct.device_buffer.reset();
            if (d.provider_backend) ggml_backend_free(d.provider_backend);
            if (d.fills && d.fills->backend) ggml_backend_free(d.fills->backend);
            d.fills.reset();
        }
    }

    ggml_backend * backend(int32_t il) const {
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) return nullptr;
        return devices[groups[layers[il].group].device].backend;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) return nullptr;
        const layer & l = layers[il];
        const group & g = groups[l.group];
        if (n_tokens <= 0 || n_tokens > GGML_MOE_PROVIDER_MAX_BATCH) return nullptr;
        if (std::min<int64_t>(n_tokens*n_expert_used, l.slot_map->ne[1]) > g.n_slots) return nullptr;
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * t) const {
        auto it = bindings.find(t);
        return it == bindings.end() ? nullptr : it->second.cached;
    }

    bool copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
            ggml_backend_sched_copy_phase phase) {
        if (async_mode) return false;
        auto it = layer_of.find(src);
        if (it == layer_of.end()) return false;
        if (phase == GGML_BACKEND_SCHED_COPY_PHASE_QUERY) return true;
        const int32_t il = it->second;
        layer & l = layers[il];
        group & g = groups[l.group];
        device & d = devices[g.device];
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
                set_expert_bank_async(d, bank, (const uint8_t *) w->data + fills[i].expert*expert_size,
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
        size_t allocated = buf_host_size;
        for (const device & d : devices) allocated += d.buffer_size;
        LLAMA_LOG_INFO("llama_moe_cache: requested %.2f MiB, allocated %.2f MiB\n",
                requested_size / 1048576.0, allocated / 1048576.0);
        auto log_upload = [](const char * name, const stats & s) {
            const size_t accesses = s.hits + s.misses;
            if (accesses) LLAMA_LOG_INFO("llama_moe_cache: %s hits=%zu misses=%zu evictions=%zu hit-rate=%.2f%% uploaded=%.2f MiB\n",
                    name, s.hits, s.misses, s.evictions, 100.0*s.hits/accesses, s.bytes/1048576.0);
        };
        log_upload("ubatch <= 8", stats_small);
        log_upload("ubatch 9-32", stats_large);
        if (!async_mode || async_stats.operations == 0) return;
        size_t accepted = 0, completed = 0, failures = 0;
        for (const device & d : devices) if (d.fills) {
            std::lock_guard<std::mutex> lock(d.fills->mutex);
            const auto & c = d.fills->counters;
            accepted += c.accepted;
            completed += c.completed;
            failures += c.failures;
        }
        LLAMA_LOG_INFO("llama_moe_cache: async operation_rows_committed/eligible=%zu/%zu fills_completed/accepted=%zu/%zu fills_failed_cancelled_stale=%zu\n",
                async_stats.committed_gpu_rows, async_stats.eligible_rows,
                completed, accepted, failures);
    }
};

bool llama_moe_cache::impl::provider_begin(void * user_data, const ggml_tensor * op, void ** request) {
    impl * self = (impl *) user_data;
    *request = nullptr;
    if (!self->async_mode || !op || !op->src[0]) return false;
    const bool fused = op->op == GGML_OP_MOE_FUSED_UP_GATE;
    if (!fused && op->op != GGML_OP_MUL_MAT_ID) return false;
    const ggml_tensor * ids = fused ? op->src[3] : op->src[2];
    const ggml_tensor * input = fused ? op->src[2] : op->src[1];
    if (!ids || !input || ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
            input->type != GGML_TYPE_F32 || input->nb[0] != sizeof(float) || input->ne[0] <= 0 ||
            input->ne[1] <= 0 || input->ne[2] != ids->ne[1] || input->ne[3] != 1 ||
            ids->ne[0] <= 0 || ids->ne[1] <= 0 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            op->ne[0] <= 0 || op->ne[1] != ids->ne[0] || op->ne[2] != ids->ne[1] || op->ne[3] != 1) return false;
    if (fused) {
        if (input->ne[1] != 1) return false;
        // Fused bias layouts fall back until direct CUDA dispatch supports them.
        if (op->src[4] || op->src[5] || !ggml_is_quantized(op->src[0]->type) ||
                (op->src[1] && (op->src[1]->type != op->src[0]->type ||
                        !ggml_are_same_shape(op->src[0], op->src[1]))) ||
                (op->src[1] ? op->src[0]->ne[1] : op->src[0]->ne[1]/2) != op->ne[0] ||
                (!op->src[1] && (op->src[0]->ne[1] % 2 != 0)) ||
                op->src[0]->ne[0] != input->ne[0] || op->src[0]->ne[2] <= 0 ||
                op->src[0]->ne[3] != 1 ||
                (moe_fused_unary(op) != GGML_UNARY_OP_SILU &&
                        moe_fused_unary(op) != GGML_UNARY_OP_SWIGLU_OAI)) return false;
    } else if ((input->ne[1] != 1 && input->ne[1] != ids->ne[0]) ||
            !input->ne[2] || op->src[0]->ne[0] <= 0 || op->src[0]->ne[1] <= 0 ||
            op->src[0]->ne[2] <= 0 || op->ne[0] != op->src[0]->ne[1]) {
        return false;
    }
#ifdef GGML_USE_CUDA
    if (!ggml_cuda_moe_cache_type_supported(op->src[0]->type) ||
            (fused && !ggml_is_quantized(op->src[0]->type))) return false;
#endif
    if ((input->buffer && !ggml_backend_buffer_is_host(input->buffer)) ||
            (ids->buffer && !ggml_backend_buffer_is_host(ids->buffer)) ||
            (op->buffer && !ggml_backend_buffer_is_host(op->buffer))) return false;
    if (fused && op->src[1] && op->src[1]->buffer && !ggml_backend_buffer_is_host(op->src[1]->buffer)) return false;

    const auto it = self->bindings.find(op->src[0]);
    if (it == self->bindings.end()) return false;
    const binding selected = it->second;
    if (selected.layer < 0 || (size_t) selected.layer >= self->layers.size()) return false;
    const layer & l = self->layers[selected.layer];
    if (l.group < 0 || selected.part < 0 || (size_t) selected.part >= l.experts.size()) return false;
    const group & g = self->groups[l.group];
    device & d = self->devices[g.device];
    binding selected_second = { -1, -1, nullptr };
    if (fused && op->src[1]) {
        const auto second_it = self->bindings.find(op->src[1]);
        if (second_it == self->bindings.end()) return false;
        selected_second = second_it->second;
        if (selected_second.layer != selected.layer || selected_second.part < 0 ||
                (size_t) selected_second.part >= g.banks.size() ||
                self->layers[selected_second.layer].group != l.group) return false;
    }
    if (!backend_is_cuda(d.backend) || d.cuda_ordinal < 0 || d.request.active) return false;
    const int32_t cache_backend_ordinal = cuda_ordinal_for_buft(ggml_backend_get_default_buffer_type(d.backend));
    if (cache_backend_ordinal != d.cuda_ordinal) return false;
    ggml_tensor * bank = g.banks[selected.part];
    if (!bank->buffer) return false;
    const int32_t bank_ordinal = cuda_ordinal_for_buft(ggml_backend_buffer_get_type(bank->buffer));
    if (bank_ordinal != d.cuda_ordinal) return false;
    if (fused && op->src[1]) {
        ggml_tensor * bank2 = g.banks[selected_second.part];
        if (!bank2->buffer || cuda_ordinal_for_buft(ggml_backend_buffer_get_type(bank2->buffer)) != d.cuda_ordinal) return false;
    }

    provider_request * req = &d.request;
    req->weight = selected;
    req->second_weight = selected_second;
    req->op = op;
    req->n_rows = 0;
    req->n_hit = 0;
    req->n_valid = 0;
    req->n_miss_candidates = 0;
    req->n_pins = 0;
    req->fused = fused;
    req->n_unique_activations = 0;
    req->submitted = false;
    req->completed = false;
    req->admission_plan_valid = false;
    req->admission_driver = fused && l.admission_part >= 0
            ? (selected.part == l.admission_part || selected_second.part == l.admission_part)
            : (!fused && l.admission_part >= 0 && selected.part == l.admission_part);
    req->active = true;
    self->active_provider_requests.fetch_add(1, std::memory_order_release);
    *request = req;
    return true;
}

bool llama_moe_cache::impl::ensure_direct_workspace(
        device & d, size_t input_bytes, size_t q8_bytes, size_t output_bytes) {
#ifndef GGML_USE_CUDA
    GGML_UNUSED(d);
    GGML_UNUSED(input_bytes);
    GGML_UNUSED(q8_bytes);
    GGML_UNUSED(output_bytes);
    return false;
#else
    if (input_bytes == 0 || output_bytes == 0 || d.cuda_ordinal < 0) return false;
    if (!d.provider_backend) {
        d.provider_backend = ggml_backend_cuda_init(d.cuda_ordinal, "graphs=0", this);
        if (!d.provider_backend) return false;
    }
    if (!backend_is_cuda(d.provider_backend)) return false;
    const int32_t provider_ordinal = cuda_ordinal_for_buft(
            ggml_backend_get_default_buffer_type(d.provider_backend));
    if (provider_ordinal != d.cuda_ordinal) return false;

    direct_workspace & ws = d.direct;
    const auto grow = [](size_t current, size_t required) {
        size_t result = current ? current : 4096;
        while (result < required && result <= SIZE_MAX/2) result *= 2;
        return result < required ? required : result;
    };
    const size_t input_capacity = grow(ws.input_capacity, input_bytes);
    const size_t q8_capacity = grow(ws.q8_capacity, std::max<size_t>(q8_bytes, 1));
    const size_t output_capacity = grow(ws.output_capacity, output_bytes);
    if (input_capacity > SIZE_MAX - q8_capacity ||
        input_capacity + q8_capacity > SIZE_MAX - output_capacity ||
        input_capacity > SIZE_MAX - output_capacity) return false;

    const bool needs_buffers = !ws.device_buffer || !ws.staging_buffer ||
            input_capacity != ws.input_capacity || q8_capacity != ws.q8_capacity ||
            output_capacity != ws.output_capacity;
    if (needs_buffers) {
        const size_t device_size = input_capacity + q8_capacity + output_capacity;
        const size_t host_size = input_capacity + output_capacity;
        ggml_backend_buffer_ptr device_buffer(ggml_backend_buft_alloc_buffer(
                ggml_backend_get_default_buffer_type(d.provider_backend), device_size));
        ggml_backend_buffer_ptr staging_buffer(ggml_backend_buft_alloc_buffer(
                ggml_backend_cuda_host_buffer_type(), host_size));
        if (!device_buffer || !staging_buffer || !ggml_backend_buffer_get_base(device_buffer.get()) ||
            !ggml_backend_buffer_get_base(staging_buffer.get())) return false;
        ws.device_buffer = std::move(device_buffer);
        ws.staging_buffer = std::move(staging_buffer);
        ws.input_capacity = input_capacity;
        ws.q8_capacity = q8_capacity;
        ws.output_capacity = output_capacity;
    }
    if (!ws.event) {
        ws.event = ggml_backend_event_new(d.provider_backend);
        if (!ws.event) return false;
    }
    return true;
#endif
}

bool llama_moe_cache::impl::prepare_direct_inputs(provider_request & req,
        const ggml_tensor * source, const ggml_tensor * ids, size_t k, size_t m,
        int64_t padded_k, direct_buffers & buffers) {
#ifndef GGML_USE_CUDA
    GGML_UNUSED(req);
    GGML_UNUSED(source);
    GGML_UNUSED(ids);
    GGML_UNUSED(k);
    GGML_UNUSED(m);
    GGML_UNUSED(padded_k);
    GGML_UNUSED(buffers);
    return false;
#else
    if (!source || !source->data || !ids || ids->ne[0] <= 0 || req.n_hit == 0 ||
            req.n_hit > GGML_MOE_PROVIDER_MAX_ROWS || k == 0 || m == 0 ||
            k > SIZE_MAX/sizeof(float)/req.n_hit || m > SIZE_MAX/sizeof(float)/req.n_hit ||
            req.n_hit > SIZE_MAX/(2*sizeof(int32_t))) return false;

    req.n_unique_activations = 0;
    for (size_t i = 0; i < req.n_hit; ++i) {
        const size_t row = req.rows[i].row;
        const size_t token = row/(size_t)ids->ne[0];
        size_t offset = token*source->nb[2];
        if (!req.fused) {
            const size_t selection = row%(size_t)ids->ne[0];
            if (source->ne[1] > 1) offset += selection*source->nb[1];
        }
        const float * activation = (const float *)((const char *)source->data + offset);
        size_t unique = 0;
        while (unique < req.n_unique_activations && req.unique_activations[unique] != activation) ++unique;
        if (unique == req.n_unique_activations) {
            req.unique_activations[unique] = activation;
            ++req.n_unique_activations;
        }
        req.activation_map[i] = (int32_t)unique;
    }

    if (req.n_unique_activations == 0 || k > SIZE_MAX/sizeof(float)/req.n_unique_activations) return false;
    const size_t ids_bytes = req.n_hit*2*sizeof(int32_t);
    const size_t activation_bytes = req.n_unique_activations*k*sizeof(float);
    if (activation_bytes > SIZE_MAX-ids_bytes ||
            req.n_hit > SIZE_MAX/m/sizeof(float)) return false;
    buffers.input_bytes = ids_bytes+activation_bytes;
    buffers.output_bytes = req.n_hit*m*sizeof(float);

    const ggml_type type = req.op->src[0]->type;
    size_t q8_bytes = 0;
    if (req.fused || type != GGML_TYPE_F16) {
        const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_1, padded_k);
        if (q8_row_bytes == 0 || req.n_unique_activations > SIZE_MAX/q8_row_bytes) return false;
        q8_bytes = req.n_unique_activations*q8_row_bytes;
    }
    const layer & l = layers[req.weight.layer];
    device & d = devices[groups[l.group].device];
    if (!ensure_direct_workspace(d, buffers.input_bytes, q8_bytes, buffers.output_bytes)) return false;

    direct_workspace & ws = d.direct;
    buffers.host = (uint8_t *)ggml_backend_buffer_get_base(ws.staging_buffer.get());
    buffers.device = (uint8_t *)ggml_backend_buffer_get_base(ws.device_buffer.get());
    if (!buffers.host || !buffers.device) return false;
    int32_t * host_slots = (int32_t *)buffers.host;
    int32_t * host_activation_map = host_slots+req.n_hit;
    float * host_activations = (float *)(host_activation_map+req.n_hit);
    for (size_t i = 0; i < req.n_hit; ++i) {
        host_slots[i] = req.rows[i].slot;
        host_activation_map[i] = req.activation_map[i];
    }
    for (size_t i = 0; i < req.n_unique_activations; ++i) {
        memcpy(host_activations+i*k, req.unique_activations[i], k*sizeof(float));
    }
    buffers.device_q8 = buffers.device+ws.input_capacity;
    buffers.device_output = (float *)(buffers.device_q8+ws.q8_capacity);
    buffers.host_output = (float *)(buffers.host+ws.input_capacity);
    return true;
#endif
}

void llama_moe_cache::impl::touch_ready_pins_locked(const group & g,
        const llama_moe_cache_residency::pin * pins, size_t n_pins) {
    if (!g.admission || !pins) return;
    for (size_t i = 0; i < n_pins; ++i) {
        const auto & pin = pins[i];
        if (pin.slot < g.admission->lru_member.size() && g.admission->lru_member[pin.slot]) {
            g.admission->touch_lru((int32_t)pin.slot);
        }
    }
}

bool llama_moe_cache::impl::provider_plan(void * user_data, void * request, const ggml_tensor * op,
        uint8_t * gpu_rows, size_t n_rows) {
    impl * self = (impl *) user_data;
    provider_request * req = (provider_request *) request;
    if (!req || !req->active || op != req->op) return false;
    const ggml_tensor * ids = req->fused ? op->src[3] : op->src[2];
    if (!ids || ids->ne[0] <= 0 || ids->ne[1] <= 0 || (size_t) ids->ne[0] > SIZE_MAX / (size_t) ids->ne[1] ||
            n_rows > GGML_MOE_PROVIDER_MAX_ROWS || n_rows != (size_t) ids->ne[0] * (size_t) ids->ne[1]) return false;

    const layer & l = self->layers[req->weight.layer];
    const group & g = self->groups[l.group];
    req->n_rows = n_rows;
    req->n_hit = 0;
    const int64_t n_selected = ids->ne[0];
    for (int64_t token = 0; token < ids->ne[1]; ++token) {
        for (int64_t selection = 0; selection < n_selected; ++selection) {
            const size_t row = (size_t) token*n_selected + selection;
            const int32_t expert = *(const int32_t *) ((const char *) ids->data + token*ids->nb[1] + selection*ids->nb[0]);
            req->route_keys[row] = {};
            req->pinned_slots[row] = -1;
            if (expert < 0 || expert >= g.ref[0]->ne[2]) continue;
            ++req->n_valid;
            req->route_keys[row] = { req->weight.layer, expert };
        }
    }
    if (!g.residency) {
        ++self->async_stats.operations;
        self->async_stats.eligible_rows += req->n_valid;
        return true;
    }
    if (!g.residency->pin_rows(req->route_keys.data(), n_rows,
                req->pinned_slots.data(),
                req->pins.data(), req->pins.size(), req->n_pins)) return false;
    const size_t n_tokens = (size_t) ids->ne[1];
    if (self->async_mode && g.admission) {
        req->admission_plan_valid = true;
        if (n_tokens <= 8) {
            for (size_t row = 0; row < n_rows; ++row) {
                const auto key = req->route_keys[row];
                if (key.layer < 0 || req->pinned_slots[row] >= 0) continue;
                bool duplicate = false;
                for (size_t i = 0; i < req->n_miss_candidates; ++i) {
                    if (req->miss_candidates[i] == key) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) req->miss_candidates[req->n_miss_candidates++] = key;
            }
        }
    }
    for (size_t row = 0; row < n_rows; ++row) {
            const int32_t slot = req->pinned_slots[row];
            if (slot < 0) continue;
            gpu_rows[row] = 1;
            GGML_ASSERT(req->n_hit < req->rows.size());
            req->rows[req->n_hit++] = { row, slot };
    }
    if (g.admission && req->n_hit != 0) {
        std::lock_guard<std::mutex> lock(g.admission->mutex);
        self->touch_ready_pins_locked(g, req->pins.data(), req->n_pins);
    }
    ++self->async_stats.operations;
    self->async_stats.eligible_rows += req->n_valid;
    return true;
}

bool llama_moe_cache::impl::provider_dispatch(void * user_data, void * request, ggml_tensor * op,
        const uint8_t * gpu_rows, size_t n_rows) {
    impl * self = (impl *) user_data;
#ifndef GGML_USE_CUDA
    (void) self;
#endif
    provider_request * req = (provider_request *) request;
    const ggml_tensor * ids = req && req->fused ? op->src[3] : (req ? op->src[2] : nullptr);
    if (!req || !req->active || op != req->op || req->n_hit == 0 || req->n_hit > req->rows.size() ||
            !ids || n_rows != req->n_rows || n_rows != (size_t) ids->ne[0]*(size_t) ids->ne[1]) return false;
    for (size_t i = 0; i < req->n_hit; ++i) if (!gpu_rows[req->rows[i].row]) return false;

#ifdef GGML_USE_CUDA
    const layer & l = self->layers[req->weight.layer];
    const group & g = self->groups[l.group];
    device & d = self->devices[g.device];
    const size_t n_hit = req->n_hit;
#endif
    if (req->fused) {
#ifndef GGML_USE_CUDA
        return false;
#else
        const ggml_tensor * source = op->src[2];
        const size_t k = (size_t) source->ne[0];
        const size_t m = (size_t) op->ne[0];
        const ggml_type type = op->src[0]->type;
        if (k == 0 || m == 0 || n_hit > GGML_MOE_PROVIDER_MAX_ROWS ||
                k > (size_t)INT64_MAX-511) return false;
        const int64_t padded_k = GGML_PAD((int64_t)k, 512);
        direct_buffers buffers;
        if (!self->prepare_direct_inputs(*req, source, ids, k, m, padded_k, buffers)) return false;

        direct_workspace & ws = d.direct;

        ggml_tensor * up_bank = g.banks[req->weight.part];
        ggml_tensor * gate_bank = nullptr;
        size_t expert_bytes = up_bank->nb[2];
        const void * up_weights = up_bank->data;
        const void * gate_weights = nullptr;
        if (req->second_weight.part >= 0) {
            gate_bank = g.banks[req->second_weight.part];
            if (gate_bank->nb[2] != expert_bytes) return false;
            gate_weights = gate_bank->data;
        } else {
            if (expert_bytes % 2 != 0 || up_bank->ne[1] != (int64_t)(2*m)) return false;
            gate_weights = up_bank->data;
            up_weights = (const char *)up_bank->data + expert_bytes/2;
        }
        float limit = 0.0f;
        memcpy(&limit, (const char *)op->op_params+sizeof(int32_t), sizeof(limit));
        if (!ggml_cuda_moe_cache_dispatch(d.provider_backend, type, up_weights,
                buffers.host, buffers.device, buffers.device_q8, buffers.device_output, buffers.host_output,
                buffers.input_bytes, buffers.output_bytes, (int64_t)k, (int64_t)m, expert_bytes,
                (int32_t)n_hit, (int32_t)req->n_unique_activations, padded_k,
                gate_weights, moe_fused_unary(op), limit)) return false;
        ggml_backend_event_record(ws.event);
#endif
    } else {
#ifndef GGML_USE_CUDA
        return false;
#else
        const ggml_tensor * source = op->src[1];
        const size_t k = (size_t) source->ne[0];
        const size_t m = (size_t) op->ne[0];
        const ggml_type type = op->src[0]->type;
        if (k == 0 || m == 0 || (type != GGML_TYPE_F16 && k > (size_t) INT64_MAX - 511)) return false;
        const int64_t padded_k = type == GGML_TYPE_F16 ? 0 : GGML_PAD((int64_t) k, 512);
        direct_buffers buffers;
        if (!self->prepare_direct_inputs(*req, source, ids, k, m, padded_k, buffers)) return false;
        direct_workspace & ws = d.direct;

        ggml_tensor * bank = g.banks[req->weight.part];
        if (!ggml_cuda_moe_cache_dispatch(d.provider_backend, type, bank->data,
                buffers.host, buffers.device, buffers.device_q8, buffers.device_output, buffers.host_output,
                buffers.input_bytes, buffers.output_bytes, (int64_t) k, (int64_t) m, bank->nb[2],
                (int32_t) n_hit, (int32_t) req->n_unique_activations, padded_k,
                nullptr, GGML_UNARY_OP_COUNT, 0.0f)) return false;
        ggml_backend_event_record(ws.event);
#endif
    }
    req->submitted = true;
    req->completed = false;
    return true;
}

bool llama_moe_cache::impl::provider_collect(void * user_data, void * request, ggml_tensor * op,
        const uint8_t * gpu_rows, size_t n_rows) {
    impl * self = (impl *) user_data;
    provider_request * req = (provider_request *) request;
    if (!req || !req->active || op != req->op || !req->submitted || n_rows != req->n_rows ||
            req->n_hit == 0) return false;
    const layer & l = self->layers[req->weight.layer];
    device & d = self->devices[self->groups[l.group].device];
    if (!d.direct.event) return false;
    ggml_backend_event_synchronize(d.direct.event);
    req->completed = true;
    const auto * staging = (const uint8_t *) ggml_backend_buffer_get_base(d.direct.staging_buffer.get());
    if (!staging) return false;
    const size_t row_bytes = (size_t) op->ne[0]*sizeof(float);
    for (size_t i = 0; i < req->n_hit; ++i) {
        const size_t row = req->rows[i].row;
        const ggml_tensor * ids = req->fused ? op->src[3] : op->src[2];
        const size_t token = row/(size_t) ids->ne[0];
        const size_t selection = row%(size_t) ids->ne[0];
        void * dst = (char *) op->data + token*op->nb[2] + selection*op->nb[1];
        memcpy(dst, staging + d.direct.input_capacity + i*row_bytes, row_bytes);
    }
    self->async_stats.committed_gpu_rows += req->n_hit;
    GGML_UNUSED(gpu_rows);
    return true;
}

void llama_moe_cache::impl::provider_end(void * user_data, void * request) {
    impl * self = (impl *) user_data;
    provider_request * req = (provider_request *) request;
    if (!req || !req->active) return;
    const layer & l = self->layers[req->weight.layer];
    device & d = self->devices[self->groups[l.group].device];
    if (req->submitted && !req->completed && d.direct.event) {
        ggml_backend_event_synchronize(d.direct.event);
        req->completed = true;
    }
    if (req->n_pins != 0) {
        const group & g = self->groups[l.group];
        const bool released = g.residency->release_pins(req->pins.data(), req->n_pins);
        GGML_ASSERT(released && "provider pin release must match its request snapshot");
        req->n_pins = 0;
    }
    self->active_provider_requests.fetch_sub(1, std::memory_order_release);
    self->admit_provider_misses(*req);
    req->submitted = false;
    req->completed = true;
    req->active = false;
    req->op = nullptr;
    req->n_rows = req->n_hit = req->n_valid = 0;
    req->second_weight = { -1, -1, nullptr };
    req->fused = false;
    req->admission_plan_valid = req->admission_driver = false;
    req->n_miss_candidates = 0;
}

llama_moe_cache::llama_moe_cache(const llama_model & model,
        const std::vector<ggml_backend *> & backends,
        const std::vector<ggml_backend_buffer_type *> & bufts, size_t size, bool async_mode)
        : pimpl(new impl(model, backends, bufts, size, async_mode)) {}

llama_moe_cache::~llama_moe_cache() = default;
ggml_backend * llama_moe_cache::backend(int32_t il) const { return pimpl->backend(il); }
ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const { return pimpl->get_slot_map(il, n_tokens, n_expert_used); }
ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * t) const { return pimpl->get_experts(t); }
bool llama_moe_cache::copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
        ggml_backend_sched_copy_phase phase) { return pimpl->copy(backend, src, dst, graph, phase); }
const ggml_moe_provider * llama_moe_cache::provider() const { return &pimpl->moe_provider; }
bool llama_moe_cache::has_eligible_layers() const {
    return std::any_of(pimpl->layers.begin(), pimpl->layers.end(), [&](const impl::layer & l) {
        return l.group >= 0 && pimpl->groups[l.group].n_slots > 0;
    });
}
