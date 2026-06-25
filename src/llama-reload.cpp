#include "llama-reload-info.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <map>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstring>


// ------------------------------------------------------------------
// Debug helpers
// ------------------------------------------------------------------
static void log_tensor_state(const char * ctx, struct ggml_tensor * t) {
#ifndef NDEBUG
    if (!t) {
        LLAMA_LOG_DEBUG("%s: tensor=NULL\n", ctx);
        return;
    }
    const char * buft_name = "null";
    if (t->buffer) {
        auto buft = ggml_backend_buffer_get_type(t->buffer);
        if (buft) buft_name = ggml_backend_buft_name(buft);
    }
    LLAMA_LOG_DEBUG("%s: tensor='%s' type=%s ne={%ld,%ld,%ld,%ld} nb={%zu,%zu,%zu,%zu} "
                    "buffer=%p data=%p extra=%p buft=%s\n",
        ctx, t->name, ggml_type_name(t->type),
        (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3],
        t->nb[0], t->nb[1], t->nb[2], t->nb[3],
        (void*)t->buffer, t->data, (void*)t->extra, buft_name);
#else
    (void)ctx;
    (void)t;
#endif
}

static void log_split_state(const char * ctx, struct ggml_tensor * t) {
#ifndef NDEBUG
    if (!t || !t->extra) {
        LLAMA_LOG_DEBUG("%s: no splits (extra=%p)\n", ctx, (void*)(t ? t->extra : nullptr));
        return;
    }
    auto extra = (ggml_split_tensor_t *)t->extra;
    LLAMA_LOG_DEBUG("%s: tensor='%s' n_device=%d split_dim=%d\n",
            ctx, t->name, extra->n_device, extra->split_dim);
    for (int i = 0; i < extra->n_device; ++i) {
        if (!extra->splits[i]) {
            LLAMA_LOG_DEBUG("%s:   split[%d]=NULL\n", ctx, i);
            continue;
        }
        const char * split_buft_name = "null";
        if (extra->splits[i]->buffer) {
            auto buft = ggml_backend_buffer_get_type(extra->splits[i]->buffer);
            if (buft) split_buft_name = ggml_backend_buft_name(buft);
        }
        LLAMA_LOG_DEBUG("%s:   split[%d] type=%s ne={%ld,%ld,%ld,%ld} nb={%zu,%zu,%zu,%zu} "
                        "buffer=%p data=%p buft=%s\n",
            ctx, i, ggml_type_name(extra->splits[i]->type),
            (long)extra->splits[i]->ne[0], (long)extra->splits[i]->ne[1],
            (long)extra->splits[i]->ne[2], (long)extra->splits[i]->ne[3],
            extra->splits[i]->nb[0], extra->splits[i]->nb[1],
            extra->splits[i]->nb[2], extra->splits[i]->nb[3],
            (void*)extra->splits[i]->buffer, extra->splits[i]->data, split_buft_name);
    }
#else
    (void)ctx;
    (void)t;
#endif
}

// ------------------------------------------------------------------
// GGUF header parser (reuses llama.cpp / ggml GGUF loader)
// ------------------------------------------------------------------
static bool gguf_find_tensor_meta(const char * path, const char * target_name,
                                  size_t & out_offset, size_t & out_nbytes,
                                  ggml_type & out_type,
                                  int64_t out_ne[GGML_MAX_DIMS])
{
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };
    struct gguf_context * gguf = gguf_init_from_file(path, params);
    if (!gguf) {
        return false;
    }
    const int idx = gguf_find_tensor(gguf, target_name);
    if (idx < 0) {
        ggml_free(ctx);
        gguf_free(gguf);
        return false;
    }
    struct ggml_tensor * tensor = ggml_get_tensor(ctx, target_name);
    if (!tensor) {
        ggml_free(ctx);
        gguf_free(gguf);
        return false;
    }

    out_offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, idx);
    out_nbytes = ggml_nbytes(tensor);
    out_type   = tensor->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        out_ne[i] = tensor->ne[i];
    }

    ggml_free(ctx);
    gguf_free(gguf);
    return true;
}

// ------------------------------------------------------------------
// Buffer census helper
// ------------------------------------------------------------------
static size_t count_buffer_users(
        const std::vector<std::pair<std::string, struct ggml_tensor *>> & tensors_by_name,
        ggml_backend_buffer_t buf)
{
    if (!buf) return 0;
    size_t n = 0;
    for (auto & p : tensors_by_name) {
        if (p.second->buffer == buf) ++n;
    }
    return n;
}

static bool is_original_snapshot_buffer(llama_model & model, ggml_backend_buffer_t buf) {
    if (!buf) return false;
    if (!model.reload) return false;
    for (const auto & kv : model.reload->tensor_reload_sources) {
        const auto & src = kv.second;
        if (buf == src.original_buffer) return true;
        for (const auto & os : src.original_splits) {
            if (buf == os.buffer) return true;
        }
    }
    return false;
}

// ------------------------------------------------------------------
// Final size estimator
// ------------------------------------------------------------------
static size_t llama_model_compute_final_nbytes(struct ggml_tensor * tensor, ggml_type new_type) {
    if (new_type == tensor->type) {
        return ggml_nbytes(tensor);
    }
    return ggml_row_size(new_type, tensor->ne[0]) * ggml_nrows(tensor);
}

// ------------------------------------------------------------------
// Fallback allocator
// ------------------------------------------------------------------
static ggml_backend_buffer_t alloc_buffer_fallback(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, size);
    if (buf) {
        LLAMA_LOG_DEBUG("%s: allocated %zu bytes on backend '%s'\n",
                __func__, size, ggml_backend_buft_name(buft));
        return buf;
    }

    auto cpu_buft = ggml_backend_cpu_buffer_type();
    if (buft == cpu_buft) {
        LLAMA_LOG_WARN("%s: CPU alloc failed (%zu bytes)\n", __func__, size);
        return nullptr;
    }

    LLAMA_LOG_WARN("%s: backend alloc failed (%zu bytes on '%s'), trying CPU fallback\n",
            __func__, size, ggml_backend_buft_name(buft));

    buf = ggml_backend_buft_alloc_buffer(cpu_buft, size);
    if (!buf) {
        LLAMA_LOG_WARN("%s: CPU fallback alloc failed (%zu bytes)\n", __func__, size);
        return nullptr;
    }
    LLAMA_LOG_DEBUG("%s: allocated %zu bytes on CPU fallback\n", __func__, size);
    return buf;
}

// ------------------------------------------------------------------
// MoE sibling resync
// ------------------------------------------------------------------
// MoE layers have three weight tensors per block: gate, up, down.
// The CUDA split backend distributes each tensor across GPUs by splitting
// one dimension (usually dim 0 or 1). Split boundaries must be multiples
// of the quantization block size (e.g. 256 for IQ1_KT). If the reference
// tensor changes quantization type, its block size changes, which changes
// the valid split boundaries. ALL siblings in the same layer MUST adopt
// the SAME per-device split dimensions, otherwise the backend dispatches
// rows to the wrong devices and corrupts inference.
//
// When the reference tensor is back on its original snapshot, siblings
// can simply be reattached to their original snapshots too -- no data
// movement or allocation is required.
// ------------------------------------------------------------------


// ------------------------------------------------------------------
// Sibling name registration
// ------------------------------------------------------------------
static void populate_moe_siblings(const char * name, tensor_reload_source & src) {
    LLAMA_LOG_DEBUG("%s: name='%s'\n", __func__, name);

    static const char * suffixes[] = {
        ".ffn_down_exps.weight",
        ".ffn_up_exps.weight",
        ".ffn_gate_exps.weight",
    };
    std::string n(name);
    for (const char * sfx : suffixes) {
        size_t pos = n.find(sfx);
        if (pos == std::string::npos) continue;
        std::string base = n.substr(0, pos);
        for (const char * other : suffixes) {
            if (strcmp(other, sfx) != 0) {
                src.sibling_names.push_back(base + other);
                LLAMA_LOG_DEBUG("%s: registered sibling '%s' for '%s'\n",
                        __func__, (base + other).c_str(), name);
            }
        }
        return;
    }
    LLAMA_LOG_DEBUG("%s: '%s' no MoE suffix matched\n", __func__, name);
}

// ------------------------------------------------------------------
// Snapshot helper
// ------------------------------------------------------------------
static void snapshot_tensor_source(struct ggml_tensor * tensor,
                                   tensor_reload_source & src)
{
    if (!tensor || src.original_buffer != nullptr) return;

    src.original_buffer = tensor->buffer;
    src.original_data   = tensor->data;
    src.original_nbytes = ggml_nbytes(tensor);
    src.original_type   = tensor->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        src.original_ne[i] = tensor->ne[i];
        src.original_nb[i] = tensor->nb[i];
    }
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    if (extra) {
        src.original_extra = extra;
        src.original_splits.clear();
        for (int i = 0; i < extra->n_device; ++i) {
            tensor_reload_source::split_info si;
            if (extra->splits[i]) {
                for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                    si.ne[j] = extra->splits[i]->ne[j];
                    si.nb[j] = extra->splits[i]->nb[j];
                }
                si.data   = extra->splits[i]->data;
                si.buffer = extra->splits[i]->buffer;
                si.tensor = extra->splits[i];
            }
            src.original_splits.push_back(si);
        }
    }
    populate_moe_siblings(ggml_get_name(tensor), src);
    src.state = tensor_reload_source::reload_state::ON_ORIGINAL;
    log_tensor_state("snapshot_tensor_source", tensor);
}

// ------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------
reload_info::reload_info(const llama_model_loader & ml) {
    for (const auto & w : ml.weights) {
        if (!w.tensor || w.idx >= (int)ml.files.size()) continue;

        struct stat st;
        if (stat(ml.files[w.idx]->get_path().c_str(), &st) != 0) continue;

        tensor_reload_source src;
        src.path        = ml.files[w.idx]->get_path();
        src.data_offset = w.offs;
        src.nbytes      = ggml_nbytes(w.tensor);
        src.last_mtime  = st.st_mtime;
#ifdef __linux__
        src.last_mtime_ns = st.st_mtim.tv_nsec;
#endif
        tensor_reload_sources[ggml_get_name(w.tensor)] = std::move(src);
    }
}

// ------------------------------------------------------------------
// Eager snapshot
// ------------------------------------------------------------------
void reload_info::snapshot_all_reload_tensors(llama_model & model) {
    if (this->reload_snapshots_done.exchange(true)) return;

    LLAMA_LOG_INFO("%s: eager snapshot of all reload tensors + siblings\n", __func__);

    for (auto & kv : tensor_reload_sources) {
        struct ggml_tensor * tensor = nullptr;
        for (auto & p : model.tensors_by_name) {
            if (p.first == kv.first) { tensor = p.second; break; }
        }
        if (!tensor) continue;
        snapshot_tensor_source(tensor, kv.second);
    }

    for (auto & kv : tensor_reload_sources) {
        auto & src = kv.second;
        for (const auto & sib_name : src.sibling_names) {
            auto it = this->tensor_reload_sources.find(sib_name);
            if (it == this->tensor_reload_sources.end()) continue;
            if (it->second.original_buffer != nullptr) continue;

            struct ggml_tensor * sib = nullptr;
            for (auto & p : model.tensors_by_name) {
                if (p.first == sib_name) { sib = p.second; break; }
            }
            if (!sib) continue;
            snapshot_tensor_source(sib, it->second);
        }
    }
}

// ------------------------------------------------------------------
// Re-attachment helper
// ------------------------------------------------------------------
static bool reattach_split_tensor_to_shared(llama_model & model, const char * name) {
    auto it = model.reload->tensor_reload_sources.find(name);
    if (it == model.reload->tensor_reload_sources.end()) return false;
    auto & src = it->second;

    if (!src.original_buffer) return false;

    struct ggml_tensor * tensor = nullptr;
    for (auto & p : model.tensors_by_name) {
        if (p.first == name) { tensor = p.second; break; }
    }
    if (!tensor) return false;
    if (tensor->buffer == src.original_buffer) {
        log_tensor_state("reattach_split_tensor_to_shared", tensor);
        src.state = tensor_reload_source::reload_state::ON_ORIGINAL;
        return true;
    }

    if (tensor->buffer && src.state != tensor_reload_source::reload_state::ON_ORIGINAL) {
        ggml_backend_buffer_free(tensor->buffer);
    }
    tensor->buffer = nullptr;
    tensor->data   = nullptr;

    tensor->buffer = src.original_buffer;
    tensor->data   = src.original_data;
    tensor->type   = src.original_type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        tensor->ne[i] = src.original_ne[i];
        tensor->nb[i] = src.original_nb[i];
    }

    if (src.original_extra) {
        tensor->extra = src.original_extra;
        auto extra = (ggml_split_tensor_t *)tensor->extra;
        for (int i = 0; i < extra->n_device && i < (int)src.original_splits.size(); ++i) {
            auto & os = src.original_splits[i];
            if (!extra->splits[i] && os.tensor) {
                extra->splits[i] = os.tensor;
            }
            if (extra->splits[i]) {
                if (extra->splits[i]->buffer && extra->splits[i]->buffer != os.buffer &&
                    src.state != tensor_reload_source::reload_state::ON_ORIGINAL) {
                    ggml_backend_buffer_free(extra->splits[i]->buffer);
                }
                extra->splits[i]->data   = os.data;
                extra->splits[i]->buffer = os.buffer;
                extra->splits[i]->type   = src.original_type;
                for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                    extra->splits[i]->ne[j] = os.ne[j];
                    extra->splits[i]->nb[j] = os.nb[j];
                }
            }
        }
    }

    src.state = tensor_reload_source::reload_state::ON_ORIGINAL;
    return true;
}

// ------------------------------------------------------------------
// MoE sibling resync
// ------------------------------------------------------------------
static void resync_moe_sibling_splits(
        llama_model & model,
        struct ggml_context * /*ctx_tmp*/,
        struct ggml_tensor * ref_tensor,
        const char * ref_name)
{
    std::string name_str(ref_name);
    std::string layer_prefix;
    std::vector<std::string> suffixes;

    if (name_str.find(".ffn_down_exps.weight") != std::string::npos) {
        layer_prefix = name_str.substr(0, name_str.find(".ffn_down_exps.weight"));
        suffixes = {".ffn_up_exps.weight", ".ffn_gate_exps.weight"};
    } else if (name_str.find(".ffn_up_exps.weight") != std::string::npos) {
        layer_prefix = name_str.substr(0, name_str.find(".ffn_up_exps.weight"));
        suffixes = {".ffn_down_exps.weight", ".ffn_gate_exps.weight"};
    } else if (name_str.find(".ffn_gate_exps.weight") != std::string::npos) {
        layer_prefix = name_str.substr(0, name_str.find(".ffn_gate_exps.weight"));
        suffixes = {".ffn_up_exps.weight", ".ffn_down_exps.weight"};
    } else {
        return;
    }

    auto ref_extra = (ggml_split_tensor_t *)ref_tensor->extra;
    if (!ref_extra) return;

    auto it_ref_src = model.reload->tensor_reload_sources.find(ref_name);
    if (it_ref_src != model.reload->tensor_reload_sources.end() && ref_tensor->buffer == it_ref_src->second.original_buffer) {
        for (const auto & suffix : suffixes) {
            reattach_split_tensor_to_shared(model, (layer_prefix + suffix).c_str());
        }
        return;
    }

    struct sibling_job {
        std::string name;
        struct ggml_tensor * tensor;
        ggml_split_tensor_t * extra;
        std::vector<char> host_buf;
        bool needs_resync = false;
    };
    std::vector<sibling_job> jobs;

    for (const auto & suffix : suffixes) {
        std::string sib_name = layer_prefix + suffix;
        struct ggml_tensor * sib = nullptr;
        for (auto & p : model.tensors_by_name) {
            if (p.first == sib_name) { sib = p.second; break; }
        }
        if (!sib || !sib->extra || sib == ref_tensor) continue;

        auto sib_extra = (ggml_split_tensor_t *)sib->extra;
        if (sib_extra->n_device != ref_extra->n_device) continue;

        int sib_dim = sib_extra->split_dim < 0 ? 0 : sib_extra->split_dim;
        int ref_dim = ref_extra->split_dim < 0 ? 0 : ref_extra->split_dim;

        bool need = false;
        for (int i = 0; i < ref_extra->n_device; ++i) {
            bool rh = ref_extra->splits[i] != nullptr;
            bool sh = sib_extra->splits[i] != nullptr;
            if (rh != sh) { need = true; break; }
            if (rh && sh && sib_extra->splits[i]->ne[sib_dim] != ref_extra->splits[i]->ne[ref_dim]) {
                need = true; break;
            }
        }
        if (!need) continue;

        size_t nbytes = ggml_nbytes(sib);
        std::vector<char> buf(nbytes);
        ggml_backend_tensor_get(sib, buf.data(), 0, nbytes);
        jobs.push_back({sib_name, sib, sib_extra, std::move(buf), true});
    }

    if (jobs.empty()) return;
    log_split_state("resync_moe_sibling_splits", ref_tensor);

    // Phase A: Detach / free old buffers, allocate new main handles
    for (auto & job : jobs) {
        auto sib = job.tensor;

        ggml_backend_buffer_type_t buft = sib->buffer
            ? ggml_backend_buffer_get_type(sib->buffer)
            : ggml_backend_cpu_buffer_type();

        auto it = model.reload->tensor_reload_sources.find(job.name);
        bool was_orig = (it != model.reload->tensor_reload_sources.end() && it->second.state == tensor_reload_source::reload_state::ON_ORIGINAL);

        if (sib->buffer) {
            if (!was_orig) ggml_backend_buffer_free(sib->buffer);
            sib->buffer = nullptr;
            sib->data   = nullptr;
        }

        size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, sib);
        ggml_backend_buffer_t new_buf = alloc_buffer_fallback(buft, alloc_size);
        if (!new_buf) {
            job.needs_resync = false;
            continue;
        }
        sib->buffer = new_buf;
        sib->data   = (void*)0x1; // dummy; split backend uses extra->splits

        if (it != model.reload->tensor_reload_sources.end()) {
            it->second.state = tensor_reload_source::reload_state::DETACHED;
        }
    }

    // Phase B: Propagate dimensions & recompute strides
    for (auto & job : jobs) {
        if (!job.needs_resync) continue;
        auto sib = job.tensor;
        auto sib_extra = job.extra;

        for (int i = 0; i < ref_extra->n_device; ++i) {
            if (!ref_extra->splits[i]) {
                if (sib_extra->splits[i]) sib_extra->splits[i] = nullptr;
                continue;
            }
            if (!sib_extra->splits[i]) continue;
            sib_extra->splits[i]->ne[sib_extra->split_dim < 0 ? 0 : sib_extra->split_dim] =
                ref_extra->splits[i]->ne[ref_extra->split_dim < 0 ? 0 : ref_extra->split_dim];
        }

        int n_dims = 0;
        for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
            if (sib->ne[i] != 1) { n_dims = i + 1; break; }
        }
        size_t ctx_size = ggml_tensor_overhead() * (sib_extra->n_device + 4);
        if (ctx_size < 16384) ctx_size = 16384;
        struct ggml_init_params p = { ctx_size, NULL, true };
        struct ggml_context * ctx = ggml_init(p);
        if (ctx) {
            for (int i = 0; i < sib_extra->n_device; ++i) {
                if (!sib_extra->splits[i]) continue;
                auto tmp = ggml_new_tensor(ctx, sib->type, n_dims, sib_extra->splits[i]->ne);
                if (tmp) {
                    for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                        sib_extra->splits[i]->nb[j] = tmp->nb[j];
                    }
                }
            }
            ggml_free(ctx);
        }
    }

    // Phase C: Allocate GPU split buffers
    bool gpu_failed = false;
#ifdef GGML_USE_CUDA
    for (auto & job : jobs) {
        if (!job.needs_resync) continue;
        auto sib_extra = job.extra;

        for (int i = 0; i < sib_extra->n_device; ++i) {
            if (!sib_extra->splits[i]) continue;
            size_t need = ggml_nbytes(sib_extra->splits[i]);
            auto buft = ggml_backend_cuda_buffer_type(i);
            auto b = ggml_backend_buft_alloc_buffer(buft, need);
            if (!b) { gpu_failed = true; break; }
            sib_extra->splits[i]->buffer = b;
            sib_extra->splits[i]->data   = ggml_backend_buffer_get_base(b);
        }
        if (gpu_failed) break;
    }
#else
    // Without CUDA support, force CPU fallback for any resync jobs
    for (auto & job : jobs) {
        if (job.needs_resync) { gpu_failed = true; break; }
    }
#endif

    // Phase D: If any GPU alloc failed, move entire layer to CPU
    if (gpu_failed) {
        for (auto & job : jobs) {
            if (!job.needs_resync) continue;
            auto sib = job.tensor;
            auto sib_extra = job.extra;

            for (int i = 0; i < sib_extra->n_device; ++i) {
                if (sib_extra->splits[i] && sib_extra->splits[i]->buffer) {
                    auto it = model.reload->tensor_reload_sources.find(job.name);
                    bool is_orig = false;
                    if (it != model.reload->tensor_reload_sources.end() && i < (int)it->second.original_splits.size()) {
                        is_orig = (sib_extra->splits[i]->buffer == it->second.original_splits[i].buffer);
                    }
                    if (!is_orig) ggml_backend_buffer_free(sib_extra->splits[i]->buffer);
                    sib_extra->splits[i]->buffer = nullptr;
                    sib_extra->splits[i]->data   = nullptr;
                }
            }

            if (sib->buffer) {
                auto it = model.reload->tensor_reload_sources.find(job.name);
                bool is_orig = (it != model.reload->tensor_reload_sources.end() && it->second.state == tensor_reload_source::reload_state::ON_ORIGINAL);
                if (!is_orig) ggml_backend_buffer_free(sib->buffer);
                sib->buffer = nullptr;
                sib->data   = nullptr;
            }

            size_t need = ggml_nbytes(sib);
            auto cpu = alloc_buffer_fallback(ggml_backend_cpu_buffer_type(), need);
            if (cpu) {
                sib->buffer = cpu;
                sib->data   = ggml_backend_buffer_get_base(cpu);
                auto it = model.reload->tensor_reload_sources.find(job.name);
                if (it != model.reload->tensor_reload_sources.end()) it->second.state = tensor_reload_source::reload_state::FALLBACK_CPU;
            }
        }
    }

    // Phase E: Write data back
    for (auto & job : jobs) {
        if (!job.needs_resync) continue;
        ggml_backend_tensor_set(job.tensor, job.host_buf.data(), 0, job.host_buf.size());
    }
}

// ------------------------------------------------------------------
// reload_tensor_split_path
// ------------------------------------------------------------------
static bool reload_tensor_split_path(
        llama_model & model,
        struct ggml_tensor * tensor,
        tensor_reload_source & src,
        const std::vector<char> & host_buf,
        ggml_type curr_type,
        bool returning_to_original,
        ggml_backend_buffer_t old_buf)
{
		(void)curr_type;
    const char * name = ggml_get_name(tensor);

    if (returning_to_original) {
        if (old_buf && src.state != tensor_reload_source::reload_state::ON_ORIGINAL) {
            ggml_backend_buffer_free(old_buf);
        }
        tensor->buffer = nullptr;
        tensor->data   = nullptr;

        if (!reattach_split_tensor_to_shared(model, name)) return false;
        for (const auto & sib : src.sibling_names) {
            reattach_split_tensor_to_shared(model, sib.c_str());
        }
        return true;
    }

    ggml_backend_buffer_type_t buft = old_buf
        ? ggml_backend_buffer_get_type(old_buf)
        : ggml_backend_cpu_buffer_type();

    if (old_buf && src.state != tensor_reload_source::reload_state::ON_ORIGINAL) {
        ggml_backend_buffer_free(old_buf);
    }
    tensor->buffer = nullptr;
    tensor->data   = nullptr;

    size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t new_buf = alloc_buffer_fallback(buft, alloc_size);
    if (!new_buf) return false;

    ggml_backend_tensor_alloc(new_buf, tensor, ggml_backend_buffer_get_base(new_buf));
    //ggml_backend_buffer_init_tensor(tensor->buffer, tensor);

    ggml_backend_tensor_set(tensor, host_buf.data(), 0, host_buf.size());
    log_tensor_state("reload_tensor_split_path", tensor);
    if (tensor->extra) resync_moe_sibling_splits(model, nullptr, tensor, name);

    src.state = tensor_reload_source::reload_state::DETACHED;
    return true;
}

// ------------------------------------------------------------------
// reload_tensor_non_split_path
// ------------------------------------------------------------------
static bool reload_tensor_non_split_path(
        llama_model & model,
        struct ggml_tensor * tensor,
        tensor_reload_source & src,
        const std::vector<char> & host_buf,
        ggml_type curr_type,
        bool returning_to_original,
        ggml_backend_buffer_t old_buf)
{
		(void)model;
		(void)curr_type;
#ifndef NDEBUG
    const char * name = ggml_get_name(tensor);
#endif

    if (returning_to_original) {
        if (old_buf && src.state != tensor_reload_source::reload_state::ON_ORIGINAL) {
            ggml_backend_buffer_free(old_buf);
        }
        tensor->buffer = src.original_buffer;
        tensor->data   = src.original_data;
        tensor->type   = src.original_type;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            tensor->ne[i] = src.original_ne[i];
            tensor->nb[i] = src.original_nb[i];
        }
        src.state = tensor_reload_source::reload_state::ON_ORIGINAL;
        return true;
    }

    ggml_backend_buffer_type_t buft = old_buf
        ? ggml_backend_buffer_get_type(old_buf)
        : ggml_backend_cpu_buffer_type();

    if (old_buf && src.state != tensor_reload_source::reload_state::ON_ORIGINAL) {
        ggml_backend_buffer_free(old_buf);
#ifndef NDEBUG
    } else if (old_buf) {
        LLAMA_LOG_DEBUG("detaching from original snapshot buffer %p for '%s'\n", (void*)old_buf, name);
#endif
    }
    tensor->buffer = nullptr;
    tensor->data   = nullptr;

    size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t new_buf = alloc_buffer_fallback(buft, alloc_size);
    if (!new_buf) return false;

    ggml_backend_tensor_alloc(new_buf, tensor, ggml_backend_buffer_get_base(new_buf));
    ggml_backend_tensor_set(tensor, host_buf.data(), 0, host_buf.size());

    src.state = tensor_reload_source::reload_state::DETACHED;
    return true;
}

// ------------------------------------------------------------------
// apply_tensor_type_change
// ------------------------------------------------------------------
static bool apply_tensor_type_change(
        llama_model & /*model*/,
        struct ggml_tensor * tensor,
        tensor_reload_source & /*src*/,
        ggml_type curr_type)
{
#ifndef NDEBUG
    const char * name = ggml_get_name(tensor);
    (void)name;
#endif
    tensor->type = curr_type;

    int n_dims = 0;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        if (tensor->ne[i] != 1) { n_dims = i + 1; break; }
    }

    size_t ctx_size = ggml_tensor_overhead() * (1 + (tensor->extra ? ((ggml_split_tensor_t*)tensor->extra)->n_device : 0))
                    + ggml_graph_overhead_custom(1, false);
    struct ggml_init_params p = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(p);
    if (!ctx) return false;

    auto tmp = ggml_new_tensor(ctx, curr_type, n_dims, tensor->ne);
    if (!tmp) { ggml_free(ctx); return false; }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) tensor->nb[i] = tmp->nb[i];

    if (tensor->extra) {
        auto extra = (ggml_split_tensor_t *)tensor->extra;
        auto tt = ggml_internal_get_type_traits(curr_type);

        if (tt.blck_size > 1 && extra->split_dim == 0) {
            int64_t bs = tt.blck_size;
            int n = extra->n_device;
            std::vector<int64_t> bounds(n, 0);
            int64_t acc = 0;
            for (int i = 0; i < n; ++i) {
                if (extra->splits[i]) acc += extra->splits[i]->ne[0];
                bounds[i] = acc;
            }
            for (int i = 0; i < n - 1; ++i) {
                if (bounds[i] > 0) {
                    bounds[i] = ((bounds[i] + bs - 1) / bs) * bs;
                }
            }
            bounds[n - 1] = tensor->ne[0];
            for (int i = 1; i < n; ++i) {
                if (bounds[i] < bounds[i - 1]) bounds[i] = bounds[i - 1];
            }
            int64_t prev = 0;
            for (int i = 0; i < n; ++i) {
                if (extra->splits[i]) {
                    int64_t ne0 = bounds[i] - prev;
                    if (ne0 <= 0) {
                        extra->splits[i] = nullptr;
                    } else {
                        extra->splits[i]->ne[0] = ne0;
                    }
                }
                prev = bounds[i];
            }
        }

        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            split->type = curr_type;
            auto t = ggml_new_tensor(ctx, curr_type, n_dims, split->ne);
            if (t) {
                for (int j = 0; j < GGML_MAX_DIMS; ++j) split->nb[j] = t->nb[j];
            }
        }

        int64_t sum = 0;
        for (int i = 0; i < extra->n_device; ++i) {
            if (extra->splits[i]) sum += extra->splits[i]->ne[0];
        }
        GGML_ASSERT(sum == tensor->ne[0]);
    }

    ggml_free(ctx);
    return true;
}

// ------------------------------------------------------------------
// reload_tensor
// ------------------------------------------------------------------
bool reload_info::reload_tensor(const char * name, llama_model & model) {
    auto it = tensor_reload_sources.find(name);
    if (it == tensor_reload_sources.end()) return false;
    auto & src = it->second;

    struct stat st;
    if (stat(src.path.c_str(), &st) != 0) return false;

    bool changed = (st.st_mtime != src.last_mtime);
#ifdef __linux__
    changed = changed || (st.st_mtim.tv_nsec != src.last_mtime_ns);
#endif
    if (!changed) return false;

    size_t off = 0, file_nbytes = 0;
    ggml_type curr_type = GGML_TYPE_COUNT;
    int64_t file_ne[GGML_MAX_DIMS];
    if (!gguf_find_tensor_meta(src.path.c_str(), name, off, file_nbytes, curr_type, file_ne)) return false;

    std::ifstream file(src.path, std::ios::binary);
    if (!file) return false;
    file.seekg((std::streamoff)off);
    if (!file) return false;

    struct ggml_tensor * tensor = nullptr;
    for (auto & p : model.tensors_by_name) {
        if (p.first == name) { tensor = p.second; break; }
    }
    if (!tensor || !src.original_buffer) return false;

    // Refuse to swap if the on-disk shape differs from the model tensor
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] != file_ne[i]) {
            LLAMA_LOG_INFO("reload_tensor: dimension mismatch for '%s': model ne[%d]=%ld, file ne[%d]=%ld — refusing swap\n",
                           name, i, (long)tensor->ne[i], i, (long)file_ne[i]);
            return false;
        }
    }

    ggml_backend_buffer_t old_buf = tensor->buffer;
    bool returning = (curr_type == src.original_type);

    std::vector<char> host_buf;
    if (!returning) {
        if (curr_type != tensor->type) {
            if (!apply_tensor_type_change(model, tensor, src, curr_type)) return false;
        }
        size_t need = ggml_nbytes(tensor);
        if (file_nbytes < need) return false;
        host_buf.resize(need);
        file.read(host_buf.data(), (std::streamsize)need);
        if (!file || (size_t)file.gcount() != need) return false;
    }

    bool ok = false;
    if (tensor->extra) {
        ok = reload_tensor_split_path(model, tensor, src, host_buf, curr_type, returning, old_buf);
    } else {
        ok = reload_tensor_non_split_path(model, tensor, src, host_buf, curr_type, returning, old_buf);
    }

    if (ok) {
        src.last_mtime = st.st_mtime;
#ifdef __linux__
        src.last_mtime_ns = st.st_mtim.tv_nsec;
#endif
    }
    return ok;
}

// ------------------------------------------------------------------
// reload_changed_tensors
// ------------------------------------------------------------------
bool reload_info::reload_changed_tensors(llama_model & model) {
    snapshot_all_reload_tensors(model);

    struct job { const char * name; bool returning; };
    std::vector<job> jobs;

    for (auto & kv : tensor_reload_sources) {
        auto & src = kv.second;
        struct stat st;
        if (stat(src.path.c_str(), &st) != 0) continue;

        bool changed = (st.st_mtime != src.last_mtime);
#ifdef __linux__
        changed = changed || (st.st_mtim.tv_nsec != src.last_mtime_ns);
#endif
        if (!changed) continue;

        size_t off = 0, nbytes = 0;
        ggml_type t = GGML_TYPE_COUNT;
        int64_t file_ne[GGML_MAX_DIMS];
        if (!gguf_find_tensor_meta(src.path.c_str(), kv.first.c_str(), off, nbytes, t, file_ne)) continue;

        struct ggml_tensor * tensor = nullptr;
        for (auto & p : model.tensors_by_name) {
            if (p.first == kv.first) { tensor = p.second; break; }
        }
        if (!tensor) continue;

        bool dims_ok = true;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (tensor->ne[i] != file_ne[i]) {
                LLAMA_LOG_INFO("reload_changed_tensors: dimension mismatch for '%s': model ne[%d]=%ld, file ne[%d]=%ld — skipping\n",
                               kv.first.c_str(), i, (long)tensor->ne[i], i, (long)file_ne[i]);
                dims_ok = false;
                break;
            }
        }
        if (!dims_ok) continue;

        bool returning = (t == src.original_type);
        jobs.push_back({kv.first.c_str(), returning});
    }

    std::sort(jobs.begin(), jobs.end(), [](const job & a, const job & b) {
        return a.returning > b.returning;
    });

    bool r = false;
    for (auto & j : jobs) {
        if (reload_tensor(j.name, model)) {
            r = true;
            LLAMA_LOG_INFO("reloaded tensor '%s'\n", j.name);
        }
    }

    if (r) {
#ifdef GGML_USE_CUDA
        ggml_backend_cuda_invalidate_graphs(&model);
#endif
    }
    return r;
}

