#include "llama-dflash.h"

#include "llama-impl.h"
#include "llama-build-context.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-spec-features.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

static ggml_backend_buffer_type_t llama_dflash_kv_cache_layer_buft(const llama_context & lctx, int32_t il) {
    if (il >= 0 && il < (int32_t) lctx.model.buft_layer.size() && lctx.model.buft_layer[il].buft != nullptr) {
        return lctx.model.buft_layer[il].buft;
    }

    if (il >= 0 && il < (int32_t) lctx.model.layers.size()) {
        const ggml_tensor * wk = lctx.model.layers[il].wk;
        if (wk != nullptr && wk->buffer != nullptr) {
            return ggml_backend_buffer_get_type(wk->buffer);
        }
    }

    return llama_default_buffer_type_cpu(true);
}

static ggml_backend_t llama_backend_for_tensor(const llama_context & lctx, const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return nullptr;
    }

    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buf == nullptr) {
        return nullptr;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
    for (ggml_backend_t backend : lctx.backends) {
        ggml_backend_buffer_type_t backend_buft = ggml_backend_is_cpu(backend)
                ? llama_default_buffer_type_cpu(true)
                : ggml_backend_get_default_buffer_type(backend);
        if (backend_buft == buft) {
            return backend;
        }
    }

    return nullptr;
}

bool llama_context::ensure_dflash_kv_cache_tensors(int32_t cross_ctx) {
    const int32_t target_cross_ctx = std::max<int32_t>(1, cross_ctx);
    const int32_t target_token_capacity = std::max<int32_t>(
            1,
            std::max<int32_t>((int32_t) model.hparams.dflash_block_size, (int32_t) cparams.n_ubatch));
    const int32_t target_cache_n_kv_total = GGML_PAD(target_cross_ctx + target_token_capacity, cparams.flash_attn ? 256 : 32);
    const ggml_type target_cache_type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
    const int32_t n_layer = model.hparams.n_layer;
    const int64_t n_embd_head_k = model.hparams.n_embd_head_k(0);
    const int64_t n_embd_head_v = model.hparams.n_embd_head_v(0);
    const int64_t n_head_kv = model.hparams.n_head_kv();

    if (dflash.kv.cache_ctx != nullptr &&
        (int32_t) dflash.kv.k_ctx_cache.size() == n_layer &&
        (int32_t) dflash.kv.cache_pos.size() == target_cross_ctx &&
        (int32_t) dflash.kv.cache_slot_valid.size() == target_cross_ctx) {
        const bool cache_matches =
                dflash.kv.k_ctx_cache.front() != nullptr &&
                dflash.kv.v_ctx_cache.front() != nullptr &&
                dflash.kv.k_ctx_cache.front()->type == target_cache_type &&
                dflash.kv.v_ctx_cache.front()->type == target_cache_type &&
                (int32_t) dflash.kv.k_ctx_cache.front()->ne[1] == target_cache_n_kv_total &&
                (int32_t) dflash.kv.v_ctx_cache.front()->ne[1] == target_cache_n_kv_total;
        if (cache_matches) {
            return true;
        }

        free_dflash_kv_cache_tensors();
        if (dflash.kv.cache_sched != nullptr) {
            ggml_backend_sched_free(dflash.kv.cache_sched);
            dflash.kv.cache_sched = nullptr;
        }
        dflash.kv.cache_graph = nullptr;
        dflash.kv.cache_graph_rows = 0;
        dflash.kv.cache_graph_write_pos = 0;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ (size_t) (4 * std::max(1, n_layer)) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    dflash.kv.cache_ctx = ggml_init(params);
    if (dflash.kv.cache_ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to allocate DFlash K/V cache context\n", __func__);
        return false;
    }

    dflash.kv.k_ctx_cache.resize((size_t) n_layer);
    dflash.kv.v_ctx_cache.resize((size_t) n_layer);
    dflash.kv.cache_pos.assign((size_t) target_cross_ctx, 0);
    dflash.kv.cache_slot_valid.assign((size_t) target_cross_ctx, 0);
    dflash.kv.cache_bufs.clear();
    dflash.kv.cache_bufs.reserve((size_t) std::max(1, n_layer) * 2);
    for (int32_t il = 0; il < n_layer; ++il) {
        ggml_backend_buffer_type_t layer_buft = llama_dflash_kv_cache_layer_buft(*this, il);
        ggml_tensor *& k_ctx_cache = dflash.kv.k_ctx_cache[il];
        ggml_tensor *& v_ctx_cache = dflash.kv.v_ctx_cache[il];

        auto alloc_kv_input = [&](ggml_tensor *& tensor, const char * tensor_tag, const char * tensor_name,
                                  ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) -> bool {
            tensor = ggml_new_tensor_3d(dflash.kv.cache_ctx, type, ne0, ne1, ne2);
            if (tensor == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to create %s for layer %d\n", __func__, tensor_tag, il);
                return false;
            }

            ggml_set_input(tensor);
            ggml_format_name(tensor, tensor_name, il);

            const size_t tensor_bytes = ggml_backend_buft_get_alloc_size(layer_buft, tensor);
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(layer_buft, tensor_bytes);
            if (buf == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to allocate %s buffer for layer %d (%zu bytes)\n",
                        __func__, tensor_tag, il, tensor_bytes);
                return false;
            }

            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            ggml_backend_tensor_alloc(buf, tensor, ggml_backend_buffer_get_base(buf));
            ggml_backend_buffer_clear(buf, 0);
            dflash.kv.cache_bufs.push_back(buf);

            return true;
        };

        if (!alloc_kv_input(k_ctx_cache, "dflash_k_ctx_cache", "dflash_k_ctx_cache_%d",
                    target_cache_type, n_embd_head_k, target_cache_n_kv_total, n_head_kv) ||
            !alloc_kv_input(v_ctx_cache, "dflash_v_ctx_cache", "dflash_v_ctx_cache_%d",
                    target_cache_type, n_embd_head_v, target_cache_n_kv_total, n_head_kv)) {
            free_dflash_kv_cache_tensors();
            return false;
        }
    }

    llama_reset_dflash_kv_cache_state(this);

    return true;
}

void llama_context::free_dflash_kv_cache_tensors() {
    auto release_vector = [](auto & v) {
        using vec_type = std::decay_t<decltype(v)>;
        vec_type().swap(v);
    };

    release_vector(dflash.kv.k_ctx_cache);
    release_vector(dflash.kv.v_ctx_cache);
    release_vector(dflash.kv.cache_pos);
    release_vector(dflash.kv.cache_slot_valid);
    dflash.kv.cache_write_pos = 0;
    dflash.kv.cache_n_filled = 0;
    dflash.kv.cache_update_rows = 0;
    dflash.kv.cache_reserved_rows = 0;
    dflash.kv.cache_view_write_pos = 0;
    dflash.kv.cache_view_n_filled = 0;
    dflash.kv.cache_applied_window_version = 0;
    dflash.kv.cache_valid = false;
    dflash.kv.cache_view_valid = false;
    dflash.kv.cache_graph = nullptr;
    dflash.kv.cache_graph_rows = 0;
    dflash.kv.cache_graph_write_pos = 0;
    dflash.kv.cache_input_target_features = nullptr;
    dflash.kv.cache_input_pos_ctx = nullptr;
    dflash.kv.kq_mask_tensor = nullptr;
    dflash.kv.kq_mask_swa_tensor = nullptr;
    dflash.kv.draft_tail_rows_tensor = nullptr;

    for (ggml_backend_buffer_t buf : dflash.kv.cache_bufs) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
    }
    release_vector(dflash.kv.cache_bufs);
    release_vector(dflash.kv.cache_compute_meta);
    if (dflash.kv.cache_ctx != nullptr) {
        ggml_free(dflash.kv.cache_ctx);
        dflash.kv.cache_ctx = nullptr;
    }
}

static void llama_graph_compute_sched(
        llama_context & lctx,
        ggml_backend_sched_t sched,
          ggml_cgraph * gf,
                  int   n_threads) {
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(lctx.backend_metal)) {
        ggml_backend_metal_set_n_cb(lctx.backend_metal, n_threads);
    }
#endif

    if (lctx.backend_cpu != nullptr) {
        ggml_backend_cpu_set_n_threads(lctx.backend_cpu, n_threads);
        ggml_backend_cpu_set_abort_callback(lctx.backend_cpu, lctx.abort_callback, lctx.abort_callback_data);
    }

    ggml_backend_sched_graph_compute_async(sched, gf);
}

static bool dflash_layer_has_attention_bias(const llama_layer & layer) {
    // build_dflash() now applies separate q/k/v/o attention biases (bq/bk/bv/bo).
    // Only fused qkv-style biases remain unimplemented in the DFlash graph.
    return layer.bqkv != nullptr ||
           layer.bqk  != nullptr ||
           layer.bkv  != nullptr;
}

static bool validate_dflash_graph_contract(const llama_context & lctx) {
    const auto & model = lctx.model;
    const auto & hparams = model.hparams;

    auto rope_dim_for_layer = [&hparams](int32_t il) -> uint32_t {
        if (hparams.rope_dim_per_layer[il] != 0) {
            return hparams.rope_dim_per_layer[il];
        }

        return hparams.swa_layers[il] ? hparams.n_rot_swa : hparams.n_rot;
    };

    auto rope_base_for_layer = [&hparams](int32_t il) -> float {
        if (hparams.has_rope_freq_base_per_layer) {
            return hparams.rope_freq_base_per_layer[il];
        }

        return hparams.swa_layers[il] ? hparams.rope_freq_base_train_swa : hparams.rope_freq_base_train;
    };

    auto rope_scale_for_layer = [&hparams](int32_t il) -> float {
        return hparams.swa_layers[il] ? hparams.rope_freq_scale_train_swa : hparams.rope_freq_scale_train;
    };

    const uint32_t ref_n_head = hparams.n_head(0);
    const uint32_t ref_n_head_kv = hparams.n_head_kv(0);
    const uint32_t ref_n_embd_head_k = hparams.n_embd_head_k(0);
    const uint32_t ref_n_embd_head_v = hparams.n_embd_head_v(0);
    const uint32_t ref_rope_dim = rope_dim_for_layer(0);
    const float ref_rope_base = rope_base_for_layer(0);
    const float ref_rope_scale = rope_scale_for_layer(0);

    for (int32_t il = 0; il < (int32_t) hparams.n_layer; ++il) {
        if (hparams.n_head((uint32_t) il) != ref_n_head ||
                hparams.n_head_kv((uint32_t) il) != ref_n_head_kv ||
                hparams.n_embd_head_k(il) != ref_n_embd_head_k ||
                hparams.n_embd_head_v(il) != ref_n_embd_head_v) {
            LLAMA_LOG_ERROR("%s: DFlash graph assumes layer-invariant head config, but layer %d differs (n_head=%u/%u n_head_kv=%u/%u head_k=%u/%u head_v=%u/%u)\n",
                    __func__,
                    il,
                    hparams.n_head((uint32_t) il), ref_n_head,
                    hparams.n_head_kv((uint32_t) il), ref_n_head_kv,
                    hparams.n_embd_head_k(il), ref_n_embd_head_k,
                    hparams.n_embd_head_v(il), ref_n_embd_head_v);
            return false;
        }

        const uint32_t rope_dim = rope_dim_for_layer(il);
        const float rope_base = rope_base_for_layer(il);
        const float rope_scale = rope_scale_for_layer(il);
        if (rope_dim != ref_rope_dim || std::fabs(rope_base - ref_rope_base) > 1e-6f || std::fabs(rope_scale - ref_rope_scale) > 1e-6f) {
            LLAMA_LOG_ERROR("%s: DFlash graph assumes layer-invariant RoPE config, but layer %d differs (dim=%u/%u base=%g/%g scale=%g/%g)\n",
                    __func__,
                    il,
                    rope_dim, ref_rope_dim,
                    (double) rope_base, (double) ref_rope_base,
                    (double) rope_scale, (double) ref_rope_scale);
            return false;
        }

            if (model.layers[il].attn_norm == nullptr ||
                model.layers[il].attn_q_norm == nullptr ||
                model.layers[il].attn_k_norm == nullptr) {
                LLAMA_LOG_ERROR("%s: DFlash graph requires attn_norm, attn_q_norm, and attn_k_norm weights, but layer %d is missing one or more of them\n",
                    __func__, il);
                return false;
            }

        const bool has_q_norm = model.layers[il].attn_q_norm != nullptr;
        const bool has_k_norm = model.layers[il].attn_k_norm != nullptr;
        if (has_q_norm != has_k_norm) {
            LLAMA_LOG_ERROR("%s: DFlash graph requires symmetric Q/K norm presence, but layer %d has q_norm=%d k_norm=%d\n",
                    __func__, il, (int) has_q_norm, (int) has_k_norm);
            return false;
        }

            if (model.layers[il].attn_norm_b != nullptr ||
                model.layers[il].attn_q_norm_b != nullptr ||
                model.layers[il].attn_k_norm_b != nullptr) {
                LLAMA_LOG_ERROR("%s: DFlash graph does not implement norm-bias tensors, but layer %d requires attn_norm_b/q_norm_b/k_norm_b\n",
                    __func__, il);
                return false;
            }

        if (dflash_layer_has_attention_bias(model.layers[il])) {
            LLAMA_LOG_ERROR("%s: DFlash graph implements only separate q/k/v/o attention bias; layer %d uses an unsupported fused qkv bias\n",
                    __func__, il);
            return false;
        }
    }

    return true;
}

bool llama_prepare_dflash_graph_inputs(
        struct llama_context & lctx,
        uint32_t n_tokens) {
    const int32_t cross_ctx = lctx.dflash.visible_cross_ctx > 0
            ? lctx.dflash.visible_cross_ctx
            : std::max<int32_t>(1, (int32_t) lctx.cparams.n_ctx - (int32_t) lctx.model.hparams.dflash_block_size);
    ggml_tensor * kq_mask = lctx.dflash.kv.kq_mask_tensor;
    ggml_tensor * kq_mask_swa = lctx.dflash.kv.kq_mask_swa_tensor;

    // An all-SWA draft has no full mask; an all-full draft has no SWA mask. Both masks share the
    // same dimensions, so use whichever one is live to derive shape.
    ggml_tensor * mask_dims = kq_mask != nullptr ? kq_mask : kq_mask_swa;
    if (mask_dims == nullptr) {
        LLAMA_LOG_ERROR("%s: DFlash graph inputs are not initialized\n", __func__);
        return false;
    }

    if (!validate_dflash_graph_contract(lctx)) {
        return false;
    }

    if (!lctx.ensure_dflash_kv_cache_tensors(cross_ctx) || lctx.dflash.kv.k_ctx_cache.empty() || lctx.dflash.kv.v_ctx_cache.empty()) {
        LLAMA_LOG_ERROR("%s: DFlash K/V cache inputs are not initialized\n", __func__);
        return false;
    }

    const float * src = lctx.dflash.target.features;
    const float * append_src = lctx.dflash.target.append_features;
    const llama_pos * src_pos = lctx.dflash.target.positions;
    const size_t total_floats = lctx.dflash.target.features_n_floats;
    const size_t append_floats = lctx.dflash.target.append_features_n_floats;
    const size_t total_positions = lctx.dflash.target.positions_n;
    const int32_t n_rows = lctx.dflash.target.features_n_rows;
    const int32_t append_rows_available = lctx.dflash.target.append_features_n_rows;
    const int32_t width = (int32_t) lctx.model.hparams.dflash_n_target_features;
    const int32_t graph_cross_ctx = (int32_t) lctx.dflash.kv.cache_pos.size();
    const int32_t n_mask_tokens = (int32_t) mask_dims->ne[1];
    const int32_t n_kv_total = (int32_t) mask_dims->ne[0];
    ggml_tensor * draft_tail_rows = lctx.dflash.kv.draft_tail_rows_tensor;

    if (graph_cross_ctx != cross_ctx) {
        LLAMA_LOG_ERROR("%s: DFlash graph cross_ctx drift (graph=%d configured=%d)\n",
                __func__, graph_cross_ctx, cross_ctx);
        return false;
    }
    if (n_rows <= 0) {
        LLAMA_LOG_ERROR("%s: missing DFlash target feature rows\n", __func__);
        return false;
    }

    const bool have_full_src = src != nullptr && total_floats == (size_t) n_rows * (size_t) width;
    if (n_rows > cross_ctx || (src != nullptr && !have_full_src)) {
        LLAMA_LOG_ERROR("%s: invalid DFlash target feature shape (rows=%d width=%d floats=%zu cross_ctx=%d)\n",
                __func__, n_rows, width, total_floats, cross_ctx);
        return false;
    }

    if (n_kv_total < cross_ctx + (int32_t) n_tokens) {
        LLAMA_LOG_ERROR("%s: invalid DFlash mask shape (n_kv_total=%d < cross_ctx+n_tokens=%d)\n",
                __func__, n_kv_total, cross_ctx + (int32_t) n_tokens);
        return false;
    }
    if (draft_tail_rows == nullptr || draft_tail_rows->type != GGML_TYPE_I32 || draft_tail_rows->ne[0] != (int64_t) n_tokens) {
        LLAMA_LOG_ERROR("%s: DFlash draft tail row input is not initialized for n_tokens=%u\n", __func__, n_tokens);
        return false;
    }

    lctx.dflash.target.pos_ctx_data.resize((size_t) cross_ctx);
    std::fill(lctx.dflash.target.pos_ctx_data.begin(), lctx.dflash.target.pos_ctx_data.end(), 0);
    if (src_pos == nullptr || total_positions != (size_t) n_rows) {
        LLAMA_LOG_ERROR("%s: missing DFlash target positions (rows=%d positions=%zu cross_ctx=%d)\n",
                __func__, n_rows, total_positions, cross_ctx);
        return false;
    }

    const llama_pos last_target_pos = src_pos[n_rows - 1];
    for (int32_t i = 1; i < n_rows; ++i) {
        if (src_pos[i] <= src_pos[i - 1]) {
            LLAMA_LOG_ERROR("%s: DFlash target positions are not strictly increasing (rows=%d first=%d last=%d)\n",
                    __func__, n_rows, (int) src_pos[0], (int) src_pos[n_rows - 1]);
            return false;
        }
    }

    const llama_dflash_kv_cache_transition cache_plan = llama_plan_dflash_kv_cache_transition(
        cross_ctx,
        lctx.dflash.kv.cache_n_filled,
        lctx.dflash.kv.cache_write_pos,
        lctx.dflash.kv.cache_valid,
        lctx.dflash.kv.cache_applied_window_version,
        lctx.dflash.target.version,
        lctx.dflash.target.keep_rows,
        lctx.dflash.target.append_rows,
        lctx.dflash.target.replace,
        n_rows);

    const bool have_append_src = append_src != nullptr &&
        append_rows_available == cache_plan.append_rows &&
        append_floats == (size_t) cache_plan.append_rows * (size_t) width;

    const int32_t update_rows = cache_plan.cache_up_to_date
            ? 0
        : (cache_plan.rebuild_cache ? n_rows : cache_plan.append_rows);
    const size_t max_nodes = lctx.model.max_nodes((int) std::max<int32_t>(1, cross_ctx)) + 24 * lctx.model.hparams.n_layer;
    const size_t meta_size = ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    if (lctx.dflash.kv.cache_compute_meta.size() != meta_size) {
        lctx.dflash.kv.cache_compute_meta.resize(meta_size);
    }

    if (lctx.dflash.kv.cache_sched == nullptr || lctx.dflash.kv.cache_reserved_rows != cross_ctx) {
        std::vector<ggml_backend_buffer_type_t> backend_buft;
        backend_buft.reserve(lctx.backends.size());
        for (auto * backend : lctx.backends) {
            if (ggml_backend_is_cpu(backend)) {
                backend_buft.push_back(llama_default_buffer_type_cpu(true));
            } else {
                backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
            }
        }

        if (lctx.dflash.kv.cache_sched != nullptr) {
            ggml_backend_sched_free(lctx.dflash.kv.cache_sched);
            lctx.dflash.kv.cache_sched = nullptr;
        }
        lctx.dflash.kv.cache_graph = nullptr;
        lctx.dflash.kv.cache_graph_rows = 0;
        lctx.dflash.kv.cache_graph_write_pos = 0;

        const int32_t saved_update_rows = lctx.dflash.kv.cache_update_rows;
        lctx.dflash.kv.cache_update_rows = cross_ctx;
        ggml_cgraph * gf_reserve = llm_build_context::llama_build_graph_dflash_kv_cache(lctx);
        lctx.dflash.kv.cache_update_rows = saved_update_rows;
        if (gf_reserve == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to build DFlash K/V cache reserve graph\n", __func__);
            return false;
        }

        lctx.dflash.kv.cache_sched = ggml_backend_sched_new(lctx.backends.data(), backend_buft.data(), lctx.backends.size(), max_nodes, false);
        const bool reserved = lctx.dflash.kv.cache_sched != nullptr && ggml_backend_sched_reserve(lctx.dflash.kv.cache_sched, gf_reserve);
        if (!reserved) {
            LLAMA_LOG_ERROR("%s: failed to initialize DFlash K/V scheduler\n", __func__);
            return false;
        }
        lctx.dflash.kv.cache_reserved_rows = cross_ctx;
    }

    if (update_rows > 0) {
        const float * update_src = nullptr;
        if (have_append_src && update_rows == cache_plan.append_rows) {
            update_src = append_src;
        } else if (have_full_src) {
            update_src = src + (size_t) (n_rows - update_rows) * (size_t) width;
        }
        const llama_pos * update_pos = src_pos + (n_rows - update_rows);

        if (update_src == nullptr) {
            LLAMA_LOG_ERROR("%s: missing DFlash appended target features for cached update (rows=%d append_rows=%d floats=%zu)\n",
                    __func__, n_rows, update_rows, append_floats);
            return false;
        }

        if (cache_plan.rebuild_cache) {
            llama_reset_dflash_kv_cache_state(&lctx);
        }

        const int32_t cache_write_start = lctx.dflash.kv.cache_write_pos;
        lctx.dflash.kv.cache_update_rows = update_rows;
        ggml_cgraph * gf_kv = nullptr;
        const bool can_reuse_kv_graph = lctx.dflash.kv.cache_graph != nullptr &&
                lctx.dflash.kv.cache_graph_rows == update_rows &&
                lctx.dflash.kv.cache_graph_write_pos == lctx.dflash.kv.cache_write_pos;
        if (can_reuse_kv_graph) {
            gf_kv = lctx.dflash.kv.cache_graph;
        } else {
            gf_kv = llm_build_context::llama_build_graph_dflash_kv_cache(lctx);
            if (gf_kv == nullptr || lctx.dflash.kv.cache_input_target_features == nullptr || lctx.dflash.kv.cache_input_pos_ctx == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to build DFlash K/V cache graph\n", __func__);
                return false;
            }

            ggml_backend_sched_reset(lctx.dflash.kv.cache_sched);
            ggml_backend_sched_alloc_graph(lctx.dflash.kv.cache_sched, gf_kv);

            lctx.dflash.kv.cache_graph = gf_kv;
            lctx.dflash.kv.cache_graph_rows = update_rows;
            lctx.dflash.kv.cache_graph_write_pos = lctx.dflash.kv.cache_write_pos;
        }

        ggml_backend_t kv_feature_backend = llama_backend_for_tensor(lctx, lctx.dflash.kv.cache_input_target_features);
        if (kv_feature_backend != nullptr) {
            ggml_backend_tensor_set_async(kv_feature_backend, lctx.dflash.kv.cache_input_target_features, update_src, 0, ggml_nbytes(lctx.dflash.kv.cache_input_target_features));
        } else {
            ggml_backend_tensor_set(lctx.dflash.kv.cache_input_target_features, update_src, 0, ggml_nbytes(lctx.dflash.kv.cache_input_target_features));
        }

        ggml_backend_t kv_pos_backend = llama_backend_for_tensor(lctx, lctx.dflash.kv.cache_input_pos_ctx);
        if (kv_pos_backend != nullptr) {
            ggml_backend_tensor_set_async(kv_pos_backend, lctx.dflash.kv.cache_input_pos_ctx, update_pos, 0, ggml_nbytes(lctx.dflash.kv.cache_input_pos_ctx));
        } else {
            ggml_backend_tensor_set(lctx.dflash.kv.cache_input_pos_ctx, update_pos, 0, ggml_nbytes(lctx.dflash.kv.cache_input_pos_ctx));
        }
        llama_graph_compute_sched(lctx, lctx.dflash.kv.cache_sched, gf_kv, lctx.cparams.n_threads);
        ggml_backend_sched_synchronize(lctx.dflash.kv.cache_sched);

        if ((int32_t) lctx.dflash.kv.cache_pos.size() != cross_ctx) {
            lctx.dflash.kv.cache_pos.assign((size_t) cross_ctx, 0);
        }
        if ((int32_t) lctx.dflash.kv.cache_slot_valid.size() != cross_ctx) {
            lctx.dflash.kv.cache_slot_valid.assign((size_t) cross_ctx, 0);
        }
        for (int32_t i = 0; i < update_rows; ++i) {
            const int32_t slot = (cache_write_start + i) % cross_ctx;
            lctx.dflash.kv.cache_pos[(size_t) slot] = update_pos[i];
            lctx.dflash.kv.cache_slot_valid[(size_t) slot] = 1;
        }

        lctx.dflash.kv.cache_n_filled = std::min(cross_ctx, lctx.dflash.kv.cache_n_filled + update_rows);
        lctx.dflash.kv.cache_write_pos = (lctx.dflash.kv.cache_write_pos + update_rows) % cross_ctx;
        lctx.dflash.kv.cache_applied_window_version = lctx.dflash.target.version;
        lctx.dflash.kv.cache_valid = true;
        lctx.dflash.kv.cache_view_n_filled = lctx.dflash.kv.cache_n_filled;
        lctx.dflash.kv.cache_view_write_pos = lctx.dflash.kv.cache_write_pos;
        lctx.dflash.kv.cache_view_valid = true;
    }

    if ((int32_t) lctx.dflash.kv.cache_pos.size() != cross_ctx ||
            (int32_t) lctx.dflash.kv.cache_slot_valid.size() != cross_ctx) {
        LLAMA_LOG_ERROR("%s: DFlash physical cache slot map is not initialized\n", __func__);
        return false;
    }

    for (int32_t i = 0; i < cross_ctx; ++i) {
        if (lctx.dflash.kv.cache_slot_valid[(size_t) i]) {
            lctx.dflash.target.pos_ctx_data[(size_t) i] = lctx.dflash.kv.cache_pos[(size_t) i];
        }
    }

    std::vector<int32_t> draft_tail_rows_data((size_t) n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        draft_tail_rows_data[(size_t) i] = cross_ctx + (int32_t) i;
    }
    ggml_backend_tensor_set(draft_tail_rows, draft_tail_rows_data.data(), 0, ggml_nbytes(draft_tail_rows));

    const size_t mask_elems = (size_t) n_kv_total * (size_t) n_mask_tokens;
    if (kq_mask == nullptr) {
        // all-SWA draft: the full mask was not created (no non-SWA layer consumes it); only the
        // SWA mask below is populated.
    } else if (kq_mask->type == GGML_TYPE_F16) {
        const ggml_fp16_t h_inf = ggml_fp32_to_fp16(-INFINITY);
        const ggml_fp16_t h_zero = ggml_fp32_to_fp16(0.0f);
        std::vector<ggml_fp16_t> mask_f16(mask_elems, h_inf);
        std::vector<ggml_fp16_t> row_f16((size_t) n_kv_total, h_inf);
        for (int32_t i = 0; i < cross_ctx; ++i) {
            if (lctx.dflash.kv.cache_slot_valid[(size_t) i]) {
                row_f16[(size_t) i] = h_zero;
            }
        }
        std::fill(row_f16.begin() + cross_ctx, row_f16.begin() + cross_ctx + n_tokens, h_zero);
        for (uint32_t j = 0; j < n_tokens; ++j) {
            std::memcpy(mask_f16.data() + (size_t) j * (size_t) n_kv_total, row_f16.data(), (size_t) n_kv_total * sizeof(ggml_fp16_t));
        }
        ggml_backend_tensor_set(kq_mask, mask_f16.data(), 0, ggml_nbytes(kq_mask));
    } else {
        lctx.dflash.target.kq_mask_data.assign(mask_elems, -INFINITY);
        std::vector<float> row_f32((size_t) n_kv_total, -INFINITY);
        for (int32_t i = 0; i < cross_ctx; ++i) {
            if (lctx.dflash.kv.cache_slot_valid[(size_t) i]) {
                row_f32[(size_t) i] = 0.0f;
            }
        }
        std::fill(row_f32.begin() + cross_ctx, row_f32.begin() + cross_ctx + n_tokens, 0.0f);
        for (uint32_t j = 0; j < n_tokens; ++j) {
            std::memcpy(lctx.dflash.target.kq_mask_data.data() + (size_t) j * (size_t) n_kv_total, row_f32.data(), (size_t) n_kv_total * sizeof(float));
        }
        ggml_backend_tensor_set(kq_mask, lctx.dflash.target.kq_mask_data.data(), 0, ggml_nbytes(kq_mask));
    }

    if (kq_mask_swa != nullptr) {
        const int32_t swa_window = (int32_t) lctx.model.hparams.n_swa;
        const int32_t draft_pos_base = (int32_t) last_target_pos;

        if (kq_mask_swa->type == GGML_TYPE_F16) {
            const ggml_fp16_t h_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t h_zero = ggml_fp32_to_fp16(0.0f);
            std::vector<ggml_fp16_t> mask_swa_f16(mask_elems, h_inf);
            for (uint32_t j = 0; j < n_tokens; ++j) {
                ggml_fp16_t * row = mask_swa_f16.data() + (size_t) j * (size_t) n_kv_total;
                const int32_t q_pos = draft_pos_base + (int32_t) j;

                for (int32_t k = 0; k < cross_ctx; ++k) {
                    if (!lctx.dflash.kv.cache_slot_valid[(size_t) k]) {
                        continue;
                    }
                    const int32_t k_pos = (int32_t) lctx.dflash.target.pos_ctx_data[(size_t) k];
                    if (q_pos - k_pos < swa_window) {
                        row[k] = h_zero;
                    }
                }

                for (int32_t k = cross_ctx; k < cross_ctx + (int32_t) n_tokens; ++k) {
                    const int32_t block_k = k - cross_ctx;
                    // intra-block draft tokens are contiguous from draft_pos_base, so the
                    // SWA distance is (j - block_k); apply the same window bound as the
                    // cross-context section above (causal AND within n_swa).
                    if (block_k <= (int32_t) j && ((int32_t) j - block_k) < swa_window) {
                        row[k] = h_zero;
                    }
                }
            }
            ggml_backend_tensor_set(kq_mask_swa, mask_swa_f16.data(), 0, ggml_nbytes(kq_mask_swa));
        } else {
            lctx.dflash.target.kq_mask_swa_data.assign(mask_elems, -INFINITY);
            for (uint32_t j = 0; j < n_tokens; ++j) {
                float * row = lctx.dflash.target.kq_mask_swa_data.data() + (size_t) j * (size_t) n_kv_total;
                const int32_t q_pos = draft_pos_base + (int32_t) j;

                for (int32_t k = 0; k < cross_ctx; ++k) {
                    if (!lctx.dflash.kv.cache_slot_valid[(size_t) k]) {
                        continue;
                    }
                    const int32_t k_pos = (int32_t) lctx.dflash.target.pos_ctx_data[(size_t) k];
                    if (q_pos - k_pos < swa_window) {
                        row[k] = 0.0f;
                    }
                }

                for (int32_t k = cross_ctx; k < cross_ctx + (int32_t) n_tokens; ++k) {
                    const int32_t block_k = k - cross_ctx;
                    // intra-block draft tokens are contiguous from draft_pos_base, so the
                    // SWA distance is (j - block_k); apply the same window bound as the
                    // cross-context section above (causal AND within n_swa).
                    if (block_k <= (int32_t) j && ((int32_t) j - block_k) < swa_window) {
                        row[k] = 0.0f;
                    }
                }
            }
            ggml_backend_tensor_set(kq_mask_swa, lctx.dflash.target.kq_mask_swa_data.data(), 0, ggml_nbytes(kq_mask_swa));
        }
    }

    return true;
}
