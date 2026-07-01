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

using llama_dflash_layout = llama_context::dflash_runtime::llama_dflash_kv_layout;

static bool llama_dflash_has_any_swa_layers(const llama_hparams & hparams) {
    if (hparams.n_swa <= 0) {
        return false;
    }

    for (int32_t il = 0; il < (int32_t) hparams.n_layer; ++il) {
        if (hparams.swa_layers[il]) {
            return true;
        }
    }

    return false;
}

static bool llama_dflash_has_any_full_layers(const llama_hparams & hparams) {
    for (int32_t il = 0; il < (int32_t) hparams.n_layer; ++il) {
        if (!hparams.swa_layers[il] || hparams.n_swa <= 0) {
            return true;
        }
    }

    return false;
}

static int32_t llama_dflash_effective_swa_capacity(const llama_hparams & hparams, int32_t cross_ctx) {
    if (!llama_dflash_has_any_swa_layers(hparams)) {
        return 0;
    }

    return std::min<int32_t>(std::max<int32_t>(1, cross_ctx), std::max<int32_t>(1, (int32_t) hparams.n_swa));
}

static bool llama_dflash_share_swa_with_full(const llama_hparams & hparams, int32_t cross_ctx) {
    return llama_dflash_has_any_swa_layers(hparams) &&
           llama_dflash_has_any_full_layers(hparams) &&
           hparams.n_swa > 0 &&
           hparams.n_swa >= cross_ctx;
}

static llama_dflash_layout & llama_dflash_select_layout(llama_context::dflash_runtime::kv_runtime_state & kv, bool use_swa) {
    return (use_swa && !kv.share_swa_with_full) ? kv.swa_layout : kv.full_layout;
}

static const llama_dflash_layout & llama_dflash_select_layout(const llama_context::dflash_runtime::kv_runtime_state & kv, bool use_swa) {
    return (use_swa && !kv.share_swa_with_full) ? kv.swa_layout : kv.full_layout;
}

static void llama_dflash_resize_layout(llama_dflash_layout & layout, int32_t capacity, bool allocate_slots) {
    layout.capacity = capacity;
    layout.n_filled = 0;
    layout.write_pos = 0;
    layout.applied_window_version = 0;
    layout.valid = false;
    if (allocate_slots && capacity > 0) {
        layout.positions.assign((size_t) capacity, 0);
        layout.slot_valid.assign((size_t) capacity, 0);
    } else {
        layout.positions.clear();
        layout.slot_valid.clear();
    }
}

static void llama_dflash_assign_layout_state(
        llama_dflash_layout & layout,
        int32_t n_filled,
        int32_t write_pos,
        uint64_t applied_window_version,
        bool valid) {
    layout.n_filled = std::clamp(n_filled, 0, std::max<int32_t>(0, layout.capacity));
    layout.write_pos = layout.capacity > 0 ? ((write_pos % layout.capacity) + layout.capacity) % layout.capacity : 0;
    layout.applied_window_version = applied_window_version;
    layout.valid = valid;
}

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
    const ggml_type target_cache_type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
    const int32_t n_layer = model.hparams.n_layer;
    const int64_t n_embd_head_k = model.hparams.n_embd_head_k(0);
    const int64_t n_embd_head_v = model.hparams.n_embd_head_v(0);
    const int64_t n_head_kv = model.hparams.n_head_kv();
    const bool has_full_layers = llama_dflash_has_any_full_layers(model.hparams);
    const bool has_swa_layers = llama_dflash_has_any_swa_layers(model.hparams);
    const bool share_swa_with_full = llama_dflash_share_swa_with_full(model.hparams, target_cross_ctx);
    const int32_t target_swa_capacity = llama_dflash_effective_swa_capacity(model.hparams, target_cross_ctx);
    std::vector<int32_t> target_layer_history_capacity((size_t) n_layer, target_cross_ctx);
    std::vector<uint8_t> target_layer_uses_swa((size_t) n_layer, 0);

    for (int32_t il = 0; il < n_layer; ++il) {
        const bool use_swa_layout = model.hparams.swa_layers[il] && target_swa_capacity > 0;
        target_layer_uses_swa[(size_t) il] = use_swa_layout ? 1 : 0;
        target_layer_history_capacity[(size_t) il] = use_swa_layout ? target_swa_capacity : target_cross_ctx;
    }

    if (dflash.kv.cache_ctx != nullptr &&
        (int32_t) dflash.kv.k_ctx_cache.size() == n_layer &&
        (int32_t) dflash.kv.layer_history_capacity.size() == n_layer &&
        (int32_t) dflash.kv.layer_uses_swa_layout.size() == n_layer &&
        dflash.kv.has_full_layers == has_full_layers &&
        dflash.kv.has_swa_layers == has_swa_layers &&
        dflash.kv.share_swa_with_full == share_swa_with_full &&
        dflash.kv.full_layout.capacity == (has_full_layers ? target_cross_ctx : 0) &&
        dflash.kv.swa_layout.capacity == (has_swa_layers ? target_swa_capacity : 0)) {
        bool cache_matches = true;
        for (int32_t il = 0; il < n_layer && cache_matches; ++il) {
            const int32_t layer_history_capacity = target_layer_history_capacity[(size_t) il];
            const int32_t target_cache_n_kv_total = GGML_PAD(layer_history_capacity + target_token_capacity, cparams.flash_attn ? 256 : 32);
            cache_matches =
                    dflash.kv.k_ctx_cache[(size_t) il] != nullptr &&
                    dflash.kv.v_ctx_cache[(size_t) il] != nullptr &&
                    dflash.kv.k_ctx_cache[(size_t) il]->type == target_cache_type &&
                    dflash.kv.v_ctx_cache[(size_t) il]->type == target_cache_type &&
                    (int32_t) dflash.kv.k_ctx_cache[(size_t) il]->ne[1] == target_cache_n_kv_total &&
                    (int32_t) dflash.kv.v_ctx_cache[(size_t) il]->ne[1] == target_cache_n_kv_total &&
                    dflash.kv.layer_history_capacity[(size_t) il] == layer_history_capacity &&
                    dflash.kv.layer_uses_swa_layout[(size_t) il] == target_layer_uses_swa[(size_t) il];
        }
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
        dflash.kv.cache_graph_full_update_rows = 0;
        dflash.kv.cache_graph_swa_update_rows = 0;
        dflash.kv.cache_graph_full_source_row_offset = 0;
        dflash.kv.cache_graph_swa_source_row_offset = 0;
        dflash.kv.cache_graph_full_write_pos = 0;
        dflash.kv.cache_graph_swa_write_pos = 0;
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
    dflash.kv.layer_history_capacity = target_layer_history_capacity;
    dflash.kv.layer_uses_swa_layout = target_layer_uses_swa;
    dflash.kv.has_full_layers = has_full_layers;
    dflash.kv.has_swa_layers = has_swa_layers;
    dflash.kv.share_swa_with_full = share_swa_with_full;
    llama_dflash_resize_layout(dflash.kv.full_layout, has_full_layers ? target_cross_ctx : 0, has_full_layers);
    llama_dflash_resize_layout(dflash.kv.swa_layout, has_swa_layers ? target_swa_capacity : 0, has_swa_layers && !share_swa_with_full);
    dflash.kv.full_update_rows = 0;
    dflash.kv.swa_update_rows = 0;
    dflash.kv.full_source_row_offset = 0;
    dflash.kv.swa_source_row_offset = 0;
    dflash.kv.full_update_write_pos = 0;
    dflash.kv.swa_update_write_pos = 0;
    dflash.kv.cache_bufs.clear();
    dflash.kv.cache_bufs.reserve((size_t) std::max(1, n_layer) * 2);
    for (int32_t il = 0; il < n_layer; ++il) {
        const int32_t layer_history_capacity = target_layer_history_capacity[(size_t) il];
        const int32_t target_cache_n_kv_total = GGML_PAD(layer_history_capacity + target_token_capacity, cparams.flash_attn ? 256 : 32);
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
    release_vector(dflash.kv.layer_history_capacity);
    release_vector(dflash.kv.layer_uses_swa_layout);
    llama_dflash_resize_layout(dflash.kv.full_layout, 0, false);
    llama_dflash_resize_layout(dflash.kv.swa_layout, 0, false);
    dflash.kv.has_full_layers = false;
    dflash.kv.has_swa_layers = false;
    dflash.kv.share_swa_with_full = false;
    dflash.kv.cache_update_rows = 0;
    dflash.kv.full_update_rows = 0;
    dflash.kv.swa_update_rows = 0;
    dflash.kv.full_source_row_offset = 0;
    dflash.kv.swa_source_row_offset = 0;
    dflash.kv.full_update_write_pos = 0;
    dflash.kv.swa_update_write_pos = 0;
    dflash.kv.cache_reserved_rows = 0;
    dflash.kv.cache_graph = nullptr;
    dflash.kv.cache_graph_rows = 0;
    dflash.kv.cache_graph_full_update_rows = 0;
    dflash.kv.cache_graph_swa_update_rows = 0;
    dflash.kv.cache_graph_full_source_row_offset = 0;
    dflash.kv.cache_graph_swa_source_row_offset = 0;
    dflash.kv.cache_graph_full_write_pos = 0;
    dflash.kv.cache_graph_swa_write_pos = 0;
    dflash.kv.cache_input_target_features = nullptr;
    dflash.kv.cache_input_pos_ctx = nullptr;
    dflash.kv.full_kq_mask_tensor = nullptr;
    dflash.kv.swa_kq_mask_tensor = nullptr;
    dflash.kv.full_draft_tail_rows_tensor = nullptr;
    dflash.kv.swa_draft_tail_rows_tensor = nullptr;

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
    ggml_tensor * kq_mask = lctx.dflash.kv.full_kq_mask_tensor;
    ggml_tensor * kq_mask_swa = lctx.dflash.kv.swa_kq_mask_tensor;
    ggml_tensor * draft_tail_rows_full = lctx.dflash.kv.full_draft_tail_rows_tensor;
    ggml_tensor * draft_tail_rows_swa = lctx.dflash.kv.swa_draft_tail_rows_tensor;

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
    const auto & kv = lctx.dflash.kv;
    const llama_dflash_layout & full_layout = kv.full_layout;
    const llama_dflash_layout & swa_layout = llama_dflash_select_layout(kv, true);
    const int32_t full_capacity = full_layout.capacity;
    const int32_t swa_capacity = swa_layout.capacity;
    const int32_t max_layout_capacity = std::max(full_capacity, swa_capacity);

    if (kv.has_full_layers && full_capacity != cross_ctx) {
        LLAMA_LOG_ERROR("%s: DFlash full-layout cross_ctx drift (layout=%d configured=%d)\n",
                __func__, full_capacity, cross_ctx);
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

    const llama_dflash_kv_cache_transition full_plan = kv.has_full_layers
            ? llama_plan_dflash_kv_cache_transition(
                    full_capacity,
                    full_layout.n_filled,
                    full_layout.write_pos,
                    full_layout.valid,
                    full_layout.applied_window_version,
                    lctx.dflash.target.version,
                    lctx.dflash.target.keep_rows,
                    lctx.dflash.target.append_rows,
                    lctx.dflash.target.replace,
                    n_rows)
            : llama_dflash_kv_cache_transition{};
    const llama_dflash_kv_cache_transition swa_plan = kv.has_swa_layers
            ? llama_plan_dflash_kv_cache_transition(
                    swa_capacity,
                    swa_layout.n_filled,
                    swa_layout.write_pos,
                    swa_layout.valid,
                    swa_layout.applied_window_version,
                    lctx.dflash.target.version,
                    lctx.dflash.target.keep_rows,
                    lctx.dflash.target.append_rows,
                    lctx.dflash.target.replace,
                    n_rows)
            : llama_dflash_kv_cache_transition{};

    const int32_t full_update_rows = kv.has_full_layers && !full_plan.cache_up_to_date ? full_plan.update_rows : 0;
    const int32_t swa_update_rows = kv.has_swa_layers && !kv.share_swa_with_full && !swa_plan.cache_up_to_date ? swa_plan.update_rows : 0;
    const bool cache_up_to_date = full_update_rows == 0 && swa_update_rows == 0;
    GGML_UNUSED(cache_up_to_date);
    const int32_t source_rows = std::max(full_update_rows, swa_update_rows);
    const bool requires_materialized_window =
            (full_update_rows > 0 && full_plan.requires_materialized_window) ||
            (swa_update_rows > 0 && swa_plan.requires_materialized_window);
    const bool have_append_src = append_src != nullptr &&
            append_rows_available > 0 &&
            append_floats == (size_t) append_rows_available * (size_t) width;
    const size_t max_nodes = lctx.model.max_nodes((int) std::max<int32_t>(1, cross_ctx)) + 24 * lctx.model.hparams.n_layer;
    const size_t meta_size = ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    if (lctx.dflash.kv.cache_compute_meta.size() != meta_size) {
        lctx.dflash.kv.cache_compute_meta.resize(meta_size);
    }

    if (lctx.dflash.kv.cache_sched == nullptr || lctx.dflash.kv.cache_reserved_rows != max_layout_capacity) {
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
        lctx.dflash.kv.cache_graph_full_update_rows = 0;
        lctx.dflash.kv.cache_graph_swa_update_rows = 0;
        lctx.dflash.kv.cache_graph_full_source_row_offset = 0;
        lctx.dflash.kv.cache_graph_swa_source_row_offset = 0;
        lctx.dflash.kv.cache_graph_full_write_pos = 0;
        lctx.dflash.kv.cache_graph_swa_write_pos = 0;

        const int32_t saved_update_rows = lctx.dflash.kv.cache_update_rows;
        const int32_t saved_full_update_rows = lctx.dflash.kv.full_update_rows;
        const int32_t saved_swa_update_rows = lctx.dflash.kv.swa_update_rows;
        const int32_t saved_full_source_row_offset = lctx.dflash.kv.full_source_row_offset;
        const int32_t saved_swa_source_row_offset = lctx.dflash.kv.swa_source_row_offset;
        const int32_t saved_full_update_write_pos = lctx.dflash.kv.full_update_write_pos;
        const int32_t saved_swa_update_write_pos = lctx.dflash.kv.swa_update_write_pos;
        lctx.dflash.kv.cache_update_rows = std::max(1, max_layout_capacity);
        lctx.dflash.kv.full_update_rows = lctx.dflash.kv.full_layout.capacity > 0 ? lctx.dflash.kv.full_layout.capacity : 0;
        lctx.dflash.kv.swa_update_rows = (!kv.share_swa_with_full && lctx.dflash.kv.swa_layout.capacity > 0) ? lctx.dflash.kv.swa_layout.capacity : 0;
        lctx.dflash.kv.full_source_row_offset = 0;
        lctx.dflash.kv.swa_source_row_offset = std::max(0, lctx.dflash.kv.cache_update_rows - lctx.dflash.kv.swa_update_rows);
        lctx.dflash.kv.full_update_write_pos = lctx.dflash.kv.full_layout.capacity > 1 ? 1 : 0;
        lctx.dflash.kv.swa_update_write_pos = lctx.dflash.kv.swa_layout.capacity > 1 ? 1 : 0;
        ggml_cgraph * gf_reserve = llm_build_context::llama_build_graph_dflash_kv_cache(lctx);
        lctx.dflash.kv.cache_update_rows = saved_update_rows;
        lctx.dflash.kv.full_update_rows = saved_full_update_rows;
        lctx.dflash.kv.swa_update_rows = saved_swa_update_rows;
        lctx.dflash.kv.full_source_row_offset = saved_full_source_row_offset;
        lctx.dflash.kv.swa_source_row_offset = saved_swa_source_row_offset;
        lctx.dflash.kv.full_update_write_pos = saved_full_update_write_pos;
        lctx.dflash.kv.swa_update_write_pos = saved_swa_update_write_pos;
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
        lctx.dflash.kv.cache_reserved_rows = max_layout_capacity;
    }

    if (source_rows > 0) {
        const float * update_src = nullptr;
        if (requires_materialized_window) {
            if (have_full_src) {
                update_src = src + (size_t) (n_rows - source_rows) * (size_t) width;
            }
        } else if (have_append_src && append_rows_available >= source_rows) {
            update_src = append_src + (size_t) (append_rows_available - source_rows) * (size_t) width;
        }
        const llama_pos * update_pos = src_pos + (n_rows - source_rows);

        if (update_src == nullptr) {
            LLAMA_LOG_ERROR("%s: missing DFlash source features for cached update (rows=%d source_rows=%d append_rows=%d append_floats=%zu materialized=%d)\n",
                    __func__, n_rows, source_rows, append_rows_available, append_floats, (int) requires_materialized_window);
            return false;
        }

        auto reset_layout_metadata = [](llama_dflash_layout & layout) {
            layout.n_filled = 0;
            layout.write_pos = 0;
            layout.applied_window_version = 0;
            layout.valid = false;
            std::fill(layout.slot_valid.begin(), layout.slot_valid.end(), 0);
        };

        if (full_update_rows > 0 && full_plan.rebuild_cache) {
            reset_layout_metadata(lctx.dflash.kv.full_layout);
        }
        if (swa_update_rows > 0 && swa_plan.rebuild_cache && !kv.share_swa_with_full) {
            reset_layout_metadata(lctx.dflash.kv.swa_layout);
        }

        lctx.dflash.kv.cache_update_rows = source_rows;
        lctx.dflash.kv.full_update_rows = full_update_rows;
        lctx.dflash.kv.swa_update_rows = swa_update_rows;
        lctx.dflash.kv.full_source_row_offset = full_update_rows > 0 ? source_rows - full_update_rows : 0;
        lctx.dflash.kv.swa_source_row_offset = swa_update_rows > 0 ? source_rows - swa_update_rows : 0;
        lctx.dflash.kv.full_update_write_pos = full_update_rows > 0
                ? (full_plan.rebuild_cache ? 0 : lctx.dflash.kv.full_layout.write_pos)
                : 0;
        lctx.dflash.kv.swa_update_write_pos = swa_update_rows > 0
                ? (swa_plan.rebuild_cache ? 0 : lctx.dflash.kv.swa_layout.write_pos)
                : 0;

        ggml_cgraph * gf_kv = nullptr;
        const bool can_reuse_kv_graph = lctx.dflash.kv.cache_graph != nullptr &&
                lctx.dflash.kv.cache_graph_rows == source_rows &&
                lctx.dflash.kv.cache_graph_full_update_rows == full_update_rows &&
                lctx.dflash.kv.cache_graph_swa_update_rows == swa_update_rows &&
                lctx.dflash.kv.cache_graph_full_source_row_offset == lctx.dflash.kv.full_source_row_offset &&
                lctx.dflash.kv.cache_graph_swa_source_row_offset == lctx.dflash.kv.swa_source_row_offset &&
                lctx.dflash.kv.cache_graph_full_write_pos == lctx.dflash.kv.full_update_write_pos &&
                lctx.dflash.kv.cache_graph_swa_write_pos == lctx.dflash.kv.swa_update_write_pos;
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
            lctx.dflash.kv.cache_graph_rows = source_rows;
            lctx.dflash.kv.cache_graph_full_update_rows = full_update_rows;
            lctx.dflash.kv.cache_graph_swa_update_rows = swa_update_rows;
            lctx.dflash.kv.cache_graph_full_source_row_offset = lctx.dflash.kv.full_source_row_offset;
            lctx.dflash.kv.cache_graph_swa_source_row_offset = lctx.dflash.kv.swa_source_row_offset;
            lctx.dflash.kv.cache_graph_full_write_pos = lctx.dflash.kv.full_update_write_pos;
            lctx.dflash.kv.cache_graph_swa_write_pos = lctx.dflash.kv.swa_update_write_pos;
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

        auto apply_layout_update = [&](llama_dflash_layout & layout, const llama_dflash_kv_cache_transition & plan, int32_t layout_update_rows) {
            if (layout.capacity <= 0 || layout_update_rows <= 0) {
                return;
            }

            if ((int32_t) layout.positions.size() != layout.capacity) {
                layout.positions.assign((size_t) layout.capacity, 0);
            }
            if ((int32_t) layout.slot_valid.size() != layout.capacity) {
                layout.slot_valid.assign((size_t) layout.capacity, 0);
            }

            const int32_t write_start = plan.rebuild_cache ? 0 : layout.write_pos;
            const llama_pos * layout_pos = update_pos + (source_rows - layout_update_rows);
            for (int32_t i = 0; i < layout_update_rows; ++i) {
                const int32_t slot = (write_start + i) % layout.capacity;
                layout.positions[(size_t) slot] = layout_pos[i];
                layout.slot_valid[(size_t) slot] = 1;
            }

            llama_dflash_assign_layout_state(layout, plan.next_n_filled, plan.next_write_pos, lctx.dflash.target.version, true);
        };

        if (kv.has_full_layers) {
            apply_layout_update(lctx.dflash.kv.full_layout, full_plan, full_update_rows);
        }
        if (kv.has_swa_layers && !kv.share_swa_with_full) {
            apply_layout_update(lctx.dflash.kv.swa_layout, swa_plan, swa_update_rows);
        }
    } else {
        lctx.dflash.kv.cache_update_rows = 0;
        lctx.dflash.kv.full_update_rows = 0;
        lctx.dflash.kv.swa_update_rows = 0;
        lctx.dflash.kv.full_source_row_offset = 0;
        lctx.dflash.kv.swa_source_row_offset = 0;
        lctx.dflash.kv.full_update_write_pos = 0;
        lctx.dflash.kv.swa_update_write_pos = 0;
    }

    auto set_tail_rows = [&](ggml_tensor * tensor, int32_t history_capacity, const char * tag) -> bool {
        if (tensor == nullptr) {
            return true;
        }
        if (tensor->type != GGML_TYPE_I32 || tensor->ne[0] != (int64_t) n_tokens) {
            LLAMA_LOG_ERROR("%s: DFlash %s tail row input is not initialized for n_tokens=%u\n", __func__, tag, n_tokens);
            return false;
        }
        std::vector<int32_t> draft_tail_rows_data((size_t) n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            draft_tail_rows_data[(size_t) i] = history_capacity + (int32_t) i;
        }
        ggml_backend_tensor_set(tensor, draft_tail_rows_data.data(), 0, ggml_nbytes(tensor));
        return true;
    };

    auto build_full_mask = [&](ggml_tensor * tensor, const llama_dflash_layout & layout) -> bool {
        if (tensor == nullptr) {
            return true;
        }
        const int32_t logical_width = layout.capacity + (int32_t) n_tokens;
        const int32_t mask_width = (int32_t) tensor->ne[0];
        const int32_t n_mask_tokens = (int32_t) tensor->ne[1];
        if (mask_width < logical_width) {
            LLAMA_LOG_ERROR("%s: DFlash full mask width drift (mask=%d expected>=%d)\n",
                    __func__, mask_width, logical_width);
            return false;
        }
        const size_t mask_elems = (size_t) mask_width * (size_t) n_mask_tokens;
        if (tensor->type == GGML_TYPE_F16) {
            const ggml_fp16_t h_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t h_zero = ggml_fp32_to_fp16(0.0f);
            std::vector<ggml_fp16_t> mask_f16(mask_elems, h_inf);
            std::vector<ggml_fp16_t> row_f16((size_t) mask_width, h_inf);
            for (int32_t i = 0; i < layout.capacity; ++i) {
                if (layout.slot_valid[(size_t) i]) {
                    row_f16[(size_t) i] = h_zero;
                }
            }
            std::fill(row_f16.begin() + layout.capacity, row_f16.begin() + logical_width, h_zero);
            for (uint32_t j = 0; j < n_tokens; ++j) {
                std::memcpy(mask_f16.data() + (size_t) j * (size_t) mask_width, row_f16.data(), (size_t) mask_width * sizeof(ggml_fp16_t));
            }
            ggml_backend_tensor_set(tensor, mask_f16.data(), 0, ggml_nbytes(tensor));
        } else {
            lctx.dflash.target.kq_mask_data.assign(mask_elems, -INFINITY);
            std::vector<float> row_f32((size_t) mask_width, -INFINITY);
            for (int32_t i = 0; i < layout.capacity; ++i) {
                if (layout.slot_valid[(size_t) i]) {
                    row_f32[(size_t) i] = 0.0f;
                }
            }
            std::fill(row_f32.begin() + layout.capacity, row_f32.begin() + logical_width, 0.0f);
            for (uint32_t j = 0; j < n_tokens; ++j) {
                std::memcpy(lctx.dflash.target.kq_mask_data.data() + (size_t) j * (size_t) mask_width, row_f32.data(), (size_t) mask_width * sizeof(float));
            }
            ggml_backend_tensor_set(tensor, lctx.dflash.target.kq_mask_data.data(), 0, ggml_nbytes(tensor));
        }
        return true;
    };

    auto build_swa_mask = [&](ggml_tensor * tensor, const llama_dflash_layout & layout) -> bool {
        if (tensor == nullptr) {
            return true;
        }
        const int32_t logical_width = layout.capacity + (int32_t) n_tokens;
        const int32_t mask_width = (int32_t) tensor->ne[0];
        const int32_t n_mask_tokens = (int32_t) tensor->ne[1];
        if (mask_width < logical_width) {
            LLAMA_LOG_ERROR("%s: DFlash SWA mask width drift (mask=%d expected>=%d)\n",
                    __func__, mask_width, logical_width);
            return false;
        }
        const int32_t swa_window = (int32_t) lctx.model.hparams.n_swa;
        const int32_t draft_pos_base = (int32_t) last_target_pos;
        const size_t mask_elems = (size_t) mask_width * (size_t) n_mask_tokens;
        if (tensor->type == GGML_TYPE_F16) {
            const ggml_fp16_t h_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t h_zero = ggml_fp32_to_fp16(0.0f);
            std::vector<ggml_fp16_t> mask_swa_f16(mask_elems, h_inf);
            for (uint32_t j = 0; j < n_tokens; ++j) {
                ggml_fp16_t * row = mask_swa_f16.data() + (size_t) j * (size_t) mask_width;
                const int32_t q_pos = draft_pos_base + (int32_t) j;
                for (int32_t k = 0; k < layout.capacity; ++k) {
                    if (!layout.slot_valid[(size_t) k]) {
                        continue;
                    }
                    const int32_t k_pos = (int32_t) layout.positions[(size_t) k];
                    if (q_pos - k_pos < swa_window) {
                        row[k] = h_zero;
                    }
                }
                for (int32_t k = layout.capacity; k < logical_width; ++k) {
                    const int32_t block_k = k - layout.capacity;
                    if (block_k <= (int32_t) j && ((int32_t) j - block_k) < swa_window) {
                        row[k] = h_zero;
                    }
                }
            }
            ggml_backend_tensor_set(tensor, mask_swa_f16.data(), 0, ggml_nbytes(tensor));
        } else {
            lctx.dflash.target.kq_mask_swa_data.assign(mask_elems, -INFINITY);
            for (uint32_t j = 0; j < n_tokens; ++j) {
                float * row = lctx.dflash.target.kq_mask_swa_data.data() + (size_t) j * (size_t) mask_width;
                const int32_t q_pos = draft_pos_base + (int32_t) j;
                for (int32_t k = 0; k < layout.capacity; ++k) {
                    if (!layout.slot_valid[(size_t) k]) {
                        continue;
                    }
                    const int32_t k_pos = (int32_t) layout.positions[(size_t) k];
                    if (q_pos - k_pos < swa_window) {
                        row[k] = 0.0f;
                    }
                }
                for (int32_t k = layout.capacity; k < logical_width; ++k) {
                    const int32_t block_k = k - layout.capacity;
                    if (block_k <= (int32_t) j && ((int32_t) j - block_k) < swa_window) {
                        row[k] = 0.0f;
                    }
                }
            }
            ggml_backend_tensor_set(tensor, lctx.dflash.target.kq_mask_swa_data.data(), 0, ggml_nbytes(tensor));
        }
        return true;
    };

    if (!set_tail_rows(draft_tail_rows_full, full_capacity, "full")) {
        return false;
    }
    if (draft_tail_rows_swa != draft_tail_rows_full && !set_tail_rows(draft_tail_rows_swa, swa_capacity, "swa")) {
        return false;
    }
    if (!build_full_mask(kq_mask, full_layout)) {
        return false;
    }
    if (!build_swa_mask(kq_mask_swa, swa_layout)) {
        return false;
    }

    return true;
}
