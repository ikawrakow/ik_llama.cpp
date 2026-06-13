#include "llama-dflash.h"

#include "llama-impl.h"
#include "llama-build-context.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-spec-features.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

static bool llama_env_flag_enabled_local(const char * name) {
    const char * env = std::getenv(name);
    return env != nullptr && *env != '\0' &&
            std::strcmp(env, "0") != 0 &&
            std::strcmp(env, "false") != 0 &&
            std::strcmp(env, "off") != 0;
}

static bool llama_dflash_stats_log_enabled() {
    return llama_env_flag_enabled_local("IK_DFLASH_STATS_LOG");
}

enum llama_dflash_kv_node_kind {
    LLAMA_DFLASH_KV_NODE_NONE = 0,
    LLAMA_DFLASH_KV_NODE_FUSED_TARGET,
    LLAMA_DFLASH_KV_NODE_K_PROJ,
    LLAMA_DFLASH_KV_NODE_K_NORM,
    LLAMA_DFLASH_KV_NODE_K_ROPE,
    LLAMA_DFLASH_KV_NODE_V_PROJ,
    LLAMA_DFLASH_KV_NODE_K_STORE,
    LLAMA_DFLASH_KV_NODE_V_STORE,
};

enum llama_dflash_main_node_kind {
    LLAMA_DFLASH_MAIN_NODE_NONE = 0,
    LLAMA_DFLASH_MAIN_NODE_QCUR,
    LLAMA_DFLASH_MAIN_NODE_K_DRAFT,
    LLAMA_DFLASH_MAIN_NODE_V_DRAFT,
    LLAMA_DFLASH_MAIN_NODE_K_CTX_VIEW,
    LLAMA_DFLASH_MAIN_NODE_V_CTX_VIEW,
    LLAMA_DFLASH_MAIN_NODE_K_CONCAT,
    LLAMA_DFLASH_MAIN_NODE_V_CONCAT,
    LLAMA_DFLASH_MAIN_NODE_K_PAD,
    LLAMA_DFLASH_MAIN_NODE_V_PAD,
    LLAMA_DFLASH_MAIN_NODE_K_PERM_CONT,
    LLAMA_DFLASH_MAIN_NODE_V_PERM_CONT,
    LLAMA_DFLASH_MAIN_NODE_FLASH_ATTN,
    LLAMA_DFLASH_MAIN_NODE_ATTN_OUT,
    LLAMA_DFLASH_MAIN_NODE_FFN,
    LLAMA_DFLASH_MAIN_NODE_RESULT_ROWS,
    LLAMA_DFLASH_MAIN_NODE_RESULT_NORM,
    LLAMA_DFLASH_MAIN_NODE_RESULT,
};

struct llama_dflash_kv_node_profiler {
    llama_dflash_profile_stats * profile = nullptr;
    int64_t t_start_us = 0;
    llama_dflash_kv_node_kind active_kind = LLAMA_DFLASH_KV_NODE_NONE;
};

struct llama_dflash_main_node_profiler {
    llama_dflash_profile_stats * profile = nullptr;
    ggml_backend_sched_eval_callback prev_callback = nullptr;
    void * prev_user_data = nullptr;
    bool prev_active = false;
    int64_t t_start_us = 0;
    llama_dflash_main_node_kind active_kind = LLAMA_DFLASH_MAIN_NODE_NONE;
};

static bool llama_dflash_tensor_name_has_prefix(const struct ggml_tensor * tensor, const char * prefix) {
    if (tensor == nullptr || prefix == nullptr || prefix[0] == '\0') {
        return false;
    }

    return std::strncmp(tensor->name, prefix, std::strlen(prefix)) == 0;
}

static bool llama_dflash_tensor_name_matches_label(const struct ggml_tensor * tensor, const char * label) {
    if (!llama_dflash_tensor_name_has_prefix(tensor, label)) {
        return false;
    }

    const size_t label_len = std::strlen(label);
    const char next = tensor->name[label_len];
    return next == '\0' || next == '-';
}

static llama_dflash_kv_node_kind llama_dflash_kv_node_kind_from_tensor(const struct ggml_tensor * tensor) {
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_kv_fused_target")) {
        return LLAMA_DFLASH_KV_NODE_FUSED_TARGET;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_kv_k_proj")) {
        return LLAMA_DFLASH_KV_NODE_K_PROJ;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_kv_k_norm")) {
        return LLAMA_DFLASH_KV_NODE_K_NORM;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_kv_k_rope")) {
        return LLAMA_DFLASH_KV_NODE_K_ROPE;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_kv_v_proj")) {
        return LLAMA_DFLASH_KV_NODE_V_PROJ;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_kv_k_store")) {
        return LLAMA_DFLASH_KV_NODE_K_STORE;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_kv_v_store")) {
        return LLAMA_DFLASH_KV_NODE_V_STORE;
    }

    return LLAMA_DFLASH_KV_NODE_NONE;
}

static void llama_dflash_kv_node_profile_add(
        llama_dflash_profile_stats & profile,
        llama_dflash_kv_node_kind kind,
        uint64_t elapsed_us) {
    switch (kind) {
        case LLAMA_DFLASH_KV_NODE_FUSED_TARGET:
            profile.graph_kv_node_fused_target_calls++;
            profile.graph_kv_node_fused_target_us += elapsed_us;
            break;
        case LLAMA_DFLASH_KV_NODE_K_PROJ:
            profile.graph_kv_node_k_proj_calls++;
            profile.graph_kv_node_k_proj_us += elapsed_us;
            break;
        case LLAMA_DFLASH_KV_NODE_K_NORM:
            profile.graph_kv_node_k_norm_calls++;
            profile.graph_kv_node_k_norm_us += elapsed_us;
            break;
        case LLAMA_DFLASH_KV_NODE_K_ROPE:
            profile.graph_kv_node_k_rope_calls++;
            profile.graph_kv_node_k_rope_us += elapsed_us;
            break;
        case LLAMA_DFLASH_KV_NODE_V_PROJ:
            profile.graph_kv_node_v_proj_calls++;
            profile.graph_kv_node_v_proj_us += elapsed_us;
            break;
        case LLAMA_DFLASH_KV_NODE_K_STORE:
            profile.graph_kv_node_k_store_calls++;
            profile.graph_kv_node_k_store_us += elapsed_us;
            break;
        case LLAMA_DFLASH_KV_NODE_V_STORE:
            profile.graph_kv_node_v_store_calls++;
            profile.graph_kv_node_v_store_us += elapsed_us;
            break;
        case LLAMA_DFLASH_KV_NODE_NONE:
            break;
    }
}

static llama_dflash_main_node_kind llama_dflash_main_node_kind_from_tensor(const struct ggml_tensor * tensor) {
    if (llama_dflash_tensor_name_has_prefix(tensor, "Qcur")) {
        return LLAMA_DFLASH_MAIN_NODE_QCUR;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "Kcur_noise")) {
        return LLAMA_DFLASH_MAIN_NODE_K_DRAFT;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "Vcur_noise")) {
        return LLAMA_DFLASH_MAIN_NODE_V_DRAFT;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "Kcur_ctx_cache")) {
        return LLAMA_DFLASH_MAIN_NODE_K_CTX_VIEW;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "Vcur_ctx_cache")) {
        return LLAMA_DFLASH_MAIN_NODE_V_CTX_VIEW;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_main_k_concat")) {
        return LLAMA_DFLASH_MAIN_NODE_K_CONCAT;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_main_v_concat")) {
        return LLAMA_DFLASH_MAIN_NODE_V_CONCAT;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_main_k_pad")) {
        return LLAMA_DFLASH_MAIN_NODE_K_PAD;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_main_v_pad")) {
        return LLAMA_DFLASH_MAIN_NODE_V_PAD;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_main_k_perm_cont")) {
        return LLAMA_DFLASH_MAIN_NODE_K_PERM_CONT;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "dflash_main_v_perm_cont")) {
        return LLAMA_DFLASH_MAIN_NODE_V_PERM_CONT;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "flash_attn_reshaped")) {
        return LLAMA_DFLASH_MAIN_NODE_NONE;
    }
    if (llama_dflash_tensor_name_matches_label(tensor, "flash_attn")) {
        return LLAMA_DFLASH_MAIN_NODE_FLASH_ATTN;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "kqv_out")) {
        return LLAMA_DFLASH_MAIN_NODE_ATTN_OUT;
    }
    if (llama_dflash_tensor_name_has_prefix(tensor, "ffn_out")) {
        return LLAMA_DFLASH_MAIN_NODE_FFN;
    }
    if (llama_dflash_tensor_name_matches_label(tensor, "result_output_rows")) {
        return LLAMA_DFLASH_MAIN_NODE_RESULT_ROWS;
    }
    if (llama_dflash_tensor_name_matches_label(tensor, "result_norm")) {
        return LLAMA_DFLASH_MAIN_NODE_RESULT_NORM;
    }
    if (llama_dflash_tensor_name_matches_label(tensor, "output")) {
        return LLAMA_DFLASH_MAIN_NODE_RESULT;
    }
    if (llama_dflash_tensor_name_matches_label(tensor, "result_output")) {
        return LLAMA_DFLASH_MAIN_NODE_RESULT;
    }

    return LLAMA_DFLASH_MAIN_NODE_NONE;
}

static void llama_dflash_main_node_profile_add(
        llama_dflash_profile_stats & profile,
        llama_dflash_main_node_kind kind,
        uint64_t elapsed_us) {
    switch (kind) {
        case LLAMA_DFLASH_MAIN_NODE_QCUR:
            profile.graph_main_node_qcur_calls++;
            profile.graph_main_node_qcur_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_K_DRAFT:
            profile.graph_main_node_k_draft_calls++;
            profile.graph_main_node_k_draft_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_V_DRAFT:
            profile.graph_main_node_v_draft_calls++;
            profile.graph_main_node_v_draft_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_K_CTX_VIEW:
            profile.graph_main_node_k_ctx_view_calls++;
            profile.graph_main_node_k_ctx_view_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_V_CTX_VIEW:
            profile.graph_main_node_v_ctx_view_calls++;
            profile.graph_main_node_v_ctx_view_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_K_CONCAT:
            profile.graph_main_node_k_concat_calls++;
            profile.graph_main_node_k_concat_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_V_CONCAT:
            profile.graph_main_node_v_concat_calls++;
            profile.graph_main_node_v_concat_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_K_PAD:
            profile.graph_main_node_k_pad_calls++;
            profile.graph_main_node_k_pad_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_V_PAD:
            profile.graph_main_node_v_pad_calls++;
            profile.graph_main_node_v_pad_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_K_PERM_CONT:
            profile.graph_main_node_k_perm_cont_calls++;
            profile.graph_main_node_k_perm_cont_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_V_PERM_CONT:
            profile.graph_main_node_v_perm_cont_calls++;
            profile.graph_main_node_v_perm_cont_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_FLASH_ATTN:
            profile.graph_main_node_flash_attn_calls++;
            profile.graph_main_node_flash_attn_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_ATTN_OUT:
            profile.graph_main_node_attn_out_calls++;
            profile.graph_main_node_attn_out_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_FFN:
            profile.graph_main_node_ffn_calls++;
            profile.graph_main_node_ffn_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_RESULT_ROWS:
            profile.graph_main_node_result_rows_calls++;
            profile.graph_main_node_result_rows_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_RESULT_NORM:
            profile.graph_main_node_result_norm_calls++;
            profile.graph_main_node_result_norm_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_RESULT:
            profile.graph_main_node_result_calls++;
            profile.graph_main_node_result_us += elapsed_us;
            break;
        case LLAMA_DFLASH_MAIN_NODE_NONE:
            break;
    }
}

static bool llama_dflash_kv_node_eval_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * profiler = static_cast<llama_dflash_kv_node_profiler *>(user_data);
    if (profiler == nullptr || profiler->profile == nullptr) {
        return false;
    }

    const llama_dflash_kv_node_kind kind = llama_dflash_kv_node_kind_from_tensor(tensor);
    if (ask) {
        if (kind == LLAMA_DFLASH_KV_NODE_NONE) {
            return false;
        }

        profiler->active_kind = kind;
        profiler->t_start_us = ggml_time_us();
        return true;
    }

    if (kind != LLAMA_DFLASH_KV_NODE_NONE && profiler->active_kind == kind && profiler->t_start_us > 0) {
        llama_dflash_kv_node_profile_add(*profiler->profile, kind, (uint64_t) (ggml_time_us() - profiler->t_start_us));
    }

    profiler->active_kind = LLAMA_DFLASH_KV_NODE_NONE;
    profiler->t_start_us = 0;
    return true;
}

static bool llama_dflash_main_node_eval_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * profiler = static_cast<llama_dflash_main_node_profiler *>(user_data);
    if (profiler == nullptr || profiler->profile == nullptr) {
        return false;
    }

    const llama_dflash_main_node_kind kind = llama_dflash_main_node_kind_from_tensor(tensor);
    if (ask) {
        profiler->prev_active = profiler->prev_callback != nullptr
                ? profiler->prev_callback(tensor, ask, profiler->prev_user_data)
                : false;

        if (kind == LLAMA_DFLASH_MAIN_NODE_NONE) {
            profiler->active_kind = LLAMA_DFLASH_MAIN_NODE_NONE;
            profiler->t_start_us = 0;
            return profiler->prev_active;
        }

        profiler->active_kind = kind;
        profiler->t_start_us = ggml_time_us();
        return true;
    }

    bool prev_result = false;
    if (profiler->prev_active && profiler->prev_callback != nullptr) {
        prev_result = profiler->prev_callback(tensor, ask, profiler->prev_user_data);
    }

    const bool tracked = kind != LLAMA_DFLASH_MAIN_NODE_NONE &&
            profiler->active_kind == kind &&
            profiler->t_start_us > 0;
    if (tracked) {
        llama_dflash_main_node_profile_add(*profiler->profile, kind, (uint64_t) (ggml_time_us() - profiler->t_start_us));
    }

    profiler->prev_active = false;
    profiler->active_kind = LLAMA_DFLASH_MAIN_NODE_NONE;
    profiler->t_start_us = 0;
    return prev_result || tracked;
}

void llama_sync_dflash_workspace_if_pending(struct llama_context & lctx) {
    if (!lctx.dflash_kv_workspace_sync_pending || lctx.dflash_workspace_sched == nullptr) {
        return;
    }

    const int64_t t_workspace_sync_us = ggml_time_us();
    ggml_backend_sched_synchronize(lctx.dflash_workspace_sched);
    lctx.dflash_profile.graph_kv_workspace_sync_us += (uint64_t) (ggml_time_us() - t_workspace_sync_us);
    lctx.dflash_kv_workspace_sync_pending = false;
}

static ggml_backend_buffer_type_t llama_dflash_kv_cache_layer_buft(const llama_context & lctx, int32_t il) {
    if (il >= 0 && (size_t) il < lctx.model.buft_layer.size() && lctx.model.buft_layer[(size_t) il].buft != nullptr) {
        return lctx.model.buft_layer[(size_t) il].buft;
    }

    if (il >= 0 && (size_t) il < lctx.model.layers.size()) {
        const ggml_tensor * wk = lctx.model.layers[(size_t) il].wk;
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
    const int32_t target_token_capacity = std::max<int32_t>(1, (int32_t) model.hparams.dflash_block_size);
    const int32_t target_workspace_n_kv_total = GGML_PAD(target_cross_ctx + target_token_capacity, cparams.flash_attn ? 256 : 32);
    const int32_t n_layer = model.hparams.n_layer;
    const int64_t n_embd_head_k = model.hparams.n_embd_head_k(0);
    const int64_t n_embd_head_v = model.hparams.n_embd_head_v(0);
    const int64_t n_head_kv = model.hparams.n_head_kv();

    if (dflash_cache_ctx != nullptr && !dflash_k_ctx_cache.empty()) {
    const bool cache_matches = (int32_t) dflash_k_ctx_cache.size() == n_layer &&
        dflash_k_ctx_cache.front() != nullptr &&
        (int32_t) dflash_k_ctx_cache.front()->ne[2] == target_cross_ctx;
    const bool workspace_matches = (int32_t) dflash_k_ctx_workspace.size() == n_layer &&
        dflash_k_ctx_workspace.front() != nullptr &&
        (int32_t) dflash_k_ctx_workspace.front()->ne[1] == target_workspace_n_kv_total;

    if (cache_matches && workspace_matches) {
            return true;
        }

        free_dflash_kv_cache_tensors();
        if (dflash_sched != nullptr) {
            ggml_backend_sched_free(dflash_sched);
            dflash_sched = nullptr;
        }
        if (dflash_workspace_sched != nullptr) {
            ggml_backend_sched_free(dflash_workspace_sched);
            dflash_workspace_sched = nullptr;
        }
        dflash_kv_graph = nullptr;
        dflash_kv_workspace_graph = nullptr;
        dflash_kv_graph_rows = 0;
        dflash_kv_graph_write_pos = 0;
        dflash_kv_workspace_graph_rows = 0;
        dflash_kv_workspace_graph_write_pos = 0;
        dflash_kv_workspace_reserved_rows = 0;
        dflash_buf_compute_meta.clear();
        dflash_workspace_buf_compute_meta.clear();
    }

    ggml_init_params params = {
        /*.mem_size   =*/ (size_t) (4 * std::max(1, n_layer)) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    dflash_cache_ctx = ggml_init(params);
    if (dflash_cache_ctx == nullptr) {
        return false;
    }

    dflash_k_ctx_cache.resize((size_t) n_layer);
    dflash_v_ctx_cache.resize((size_t) n_layer);
    dflash_k_ctx_workspace.clear();
    dflash_v_ctx_workspace.clear();
    dflash_k_ctx_workspace.resize((size_t) n_layer);
    dflash_v_ctx_workspace.resize((size_t) n_layer);
    dflash_cache_bufs.clear();
    dflash_cache_bufs.reserve((size_t) std::max(1, n_layer) * 4);
    int32_t host_layers = 0;
    const char * first_buft_name = nullptr;
    const char * last_buft_name = nullptr;
    for (int32_t il = 0; il < n_layer; ++il) {
        ggml_backend_buffer_type_t layer_buft = llama_dflash_kv_cache_layer_buft(*this, il);
        if (ggml_backend_buft_is_host(layer_buft)) {
            host_layers++;
        }
        if (first_buft_name == nullptr) {
            first_buft_name = ggml_backend_buft_name(layer_buft);
        }
        last_buft_name = ggml_backend_buft_name(layer_buft);

        dflash_k_ctx_cache[(size_t) il] = ggml_new_tensor_3d(dflash_cache_ctx, GGML_TYPE_F32, n_embd_head_k, n_head_kv, target_cross_ctx);
        dflash_v_ctx_cache[(size_t) il] = ggml_new_tensor_3d(dflash_cache_ctx, GGML_TYPE_F32, n_embd_head_v, n_head_kv, target_cross_ctx);
        if (dflash_k_ctx_cache[(size_t) il] == nullptr || dflash_v_ctx_cache[(size_t) il] == nullptr) {
            free_dflash_kv_cache_tensors();
            return false;
        }

        ggml_set_input(dflash_k_ctx_cache[(size_t) il]);
        ggml_set_input(dflash_v_ctx_cache[(size_t) il]);
        ggml_format_name(dflash_k_ctx_cache[(size_t) il], "dflash_k_ctx_cache_%d", il);
        ggml_format_name(dflash_v_ctx_cache[(size_t) il], "dflash_v_ctx_cache_%d", il);

        const size_t k_bytes = ggml_backend_buft_get_alloc_size(layer_buft, dflash_k_ctx_cache[(size_t) il]);
        ggml_backend_buffer_t k_buf = ggml_backend_buft_alloc_buffer(layer_buft, k_bytes);
        if (k_buf == nullptr) {
            free_dflash_kv_cache_tensors();
            return false;
        }
        ggml_backend_buffer_set_usage(k_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_tensor_alloc(k_buf, dflash_k_ctx_cache[(size_t) il], ggml_backend_buffer_get_base(k_buf));
        ggml_backend_buffer_clear(k_buf, 0);
        dflash_cache_bufs.push_back(k_buf);

        const size_t v_bytes = ggml_backend_buft_get_alloc_size(layer_buft, dflash_v_ctx_cache[(size_t) il]);
        ggml_backend_buffer_t v_buf = ggml_backend_buft_alloc_buffer(layer_buft, v_bytes);
        if (v_buf == nullptr) {
            free_dflash_kv_cache_tensors();
            return false;
        }
        ggml_backend_buffer_set_usage(v_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_tensor_alloc(v_buf, dflash_v_ctx_cache[(size_t) il], ggml_backend_buffer_get_base(v_buf));
        ggml_backend_buffer_clear(v_buf, 0);
        dflash_cache_bufs.push_back(v_buf);

        dflash_k_ctx_workspace[(size_t) il] = ggml_new_tensor_3d(dflash_cache_ctx, GGML_TYPE_F32, n_embd_head_k, target_workspace_n_kv_total, n_head_kv);
        dflash_v_ctx_workspace[(size_t) il] = ggml_new_tensor_3d(dflash_cache_ctx, GGML_TYPE_F32, n_embd_head_v, target_workspace_n_kv_total, n_head_kv);
        if (dflash_k_ctx_workspace[(size_t) il] == nullptr || dflash_v_ctx_workspace[(size_t) il] == nullptr) {
            free_dflash_kv_cache_tensors();
            return false;
        }

        ggml_set_input(dflash_k_ctx_workspace[(size_t) il]);
        ggml_set_input(dflash_v_ctx_workspace[(size_t) il]);
        ggml_format_name(dflash_k_ctx_workspace[(size_t) il], "dflash_k_ctx_workspace_%d", il);
        ggml_format_name(dflash_v_ctx_workspace[(size_t) il], "dflash_v_ctx_workspace_%d", il);

        const size_t k_workspace_bytes = ggml_backend_buft_get_alloc_size(layer_buft, dflash_k_ctx_workspace[(size_t) il]);
        ggml_backend_buffer_t k_workspace_buf = ggml_backend_buft_alloc_buffer(layer_buft, k_workspace_bytes);
        if (k_workspace_buf == nullptr) {
            free_dflash_kv_cache_tensors();
            return false;
        }
        ggml_backend_buffer_set_usage(k_workspace_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_tensor_alloc(k_workspace_buf, dflash_k_ctx_workspace[(size_t) il], ggml_backend_buffer_get_base(k_workspace_buf));
        ggml_backend_buffer_clear(k_workspace_buf, 0);
        dflash_cache_bufs.push_back(k_workspace_buf);

        const size_t v_workspace_bytes = ggml_backend_buft_get_alloc_size(layer_buft, dflash_v_ctx_workspace[(size_t) il]);
        ggml_backend_buffer_t v_workspace_buf = ggml_backend_buft_alloc_buffer(layer_buft, v_workspace_bytes);
        if (v_workspace_buf == nullptr) {
            free_dflash_kv_cache_tensors();
            return false;
        }
        ggml_backend_buffer_set_usage(v_workspace_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_tensor_alloc(v_workspace_buf, dflash_v_ctx_workspace[(size_t) il], ggml_backend_buffer_get_base(v_workspace_buf));
        ggml_backend_buffer_clear(v_workspace_buf, 0);
        dflash_cache_bufs.push_back(v_workspace_buf);
    }

    dflash_profile.last_kv_cache_host_layers = host_layers;
    dflash_kv_workspace_token_capacity = target_token_capacity;
    dflash_kv_workspace_n_kv_total = target_workspace_n_kv_total;
    llama_reset_dflash_kv_cache_state(this);
    if (llama_dflash_stats_log_enabled()) {
        LLAMA_LOG_INFO("%s: DFlash K/V cache placement cross_ctx=%d host_layers=%d/%d first=%s last=%s\n",
                __func__,
                target_cross_ctx,
                host_layers,
                n_layer,
                first_buft_name != nullptr ? first_buft_name : "(none)",
                last_buft_name != nullptr ? last_buft_name : "(none)");
    }

    return true;
}

void llama_context::free_dflash_kv_cache_tensors() {
    dflash_k_ctx_cache.clear();
    dflash_v_ctx_cache.clear();
    dflash_k_ctx_workspace.clear();
    dflash_v_ctx_workspace.clear();
    dflash_kv_cache_write_pos = 0;
    dflash_kv_cache_n_filled = 0;
    dflash_kv_cache_update_rows = 0;
    dflash_kv_cache_reserved_rows = 0;
    dflash_kv_cache_view_write_pos = 0;
    dflash_kv_cache_view_n_filled = 0;
    dflash_kv_cache_applied_window_version = 0;
    dflash_kv_cache_valid = false;
    dflash_kv_cache_view_valid = false;
    dflash_kv_workspace_write_pos = 0;
    dflash_kv_workspace_n_filled = 0;
    dflash_kv_workspace_reserved_rows = 0;
    dflash_kv_workspace_token_capacity = 0;
    dflash_kv_workspace_n_kv_total = 0;
    dflash_kv_workspace_applied_window_version = 0;
    dflash_kv_workspace_valid = false;
    dflash_kv_workspace_sync_pending = false;
    dflash_kv_graph = nullptr;
    dflash_kv_workspace_graph = nullptr;
    dflash_kv_graph_rows = 0;
    dflash_kv_graph_write_pos = 0;
    dflash_kv_workspace_graph_rows = 0;
    dflash_kv_workspace_graph_write_pos = 0;
    dflash_kv_input_target_features = nullptr;
    dflash_kv_input_pos_ctx = nullptr;
    dflash_kq_mask_tensor = nullptr;
    dflash_kq_mask_swa_tensor = nullptr;

    if (dflash_workspace_sched != nullptr) {
        ggml_backend_sched_synchronize(dflash_workspace_sched);
        ggml_backend_sched_free(dflash_workspace_sched);
        dflash_workspace_sched = nullptr;
    }

    for (ggml_backend_buffer_t buf : dflash_cache_bufs) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
    }
    dflash_cache_bufs.clear();
    if (dflash_cache_ctx != nullptr) {
        ggml_free(dflash_cache_ctx);
        dflash_cache_ctx = nullptr;
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
#ifdef GGML_USE_BLAS
    if (lctx.backend_blas != nullptr) {
        ggml_backend_blas_set_n_threads(lctx.backend_blas, n_threads);
    }
#endif

    ggml_backend_sched_graph_compute_async(sched, gf);
}

static bool dflash_layer_has_attention_bias(const llama_layer & layer) {
    return layer.bq != nullptr ||
           layer.bk != nullptr ||
           layer.bv != nullptr ||
           layer.bo != nullptr ||
           layer.bqkv != nullptr ||
           layer.bqk != nullptr ||
           layer.bkv != nullptr;
}

static bool validate_dflash_graph_contract(const llama_context & lctx) {
    const auto & model = lctx.model;
    const auto & hparams = model.hparams;

    auto rope_dim_for_layer = [&hparams](int32_t il) -> uint32_t {
        if (hparams.rope_dim_per_layer[(size_t) il] != 0) {
            return hparams.rope_dim_per_layer[(size_t) il];
        }

        return hparams.swa_layers[(size_t) il] ? hparams.n_rot_swa : hparams.n_rot;
    };

    auto rope_base_for_layer = [&hparams](int32_t il) -> float {
        if (hparams.has_rope_freq_base_per_layer) {
            return hparams.rope_freq_base_per_layer[(size_t) il];
        }

        return hparams.swa_layers[(size_t) il] ? hparams.rope_freq_base_train_swa : hparams.rope_freq_base_train;
    };

    auto rope_scale_for_layer = [&hparams](int32_t il) -> float {
        return hparams.swa_layers[(size_t) il] ? hparams.rope_freq_scale_train_swa : hparams.rope_freq_scale_train;
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

            if (model.layers[(size_t) il].attn_norm == nullptr ||
                model.layers[(size_t) il].attn_q_norm == nullptr ||
                model.layers[(size_t) il].attn_k_norm == nullptr) {
                LLAMA_LOG_ERROR("%s: DFlash graph requires attn_norm, attn_q_norm, and attn_k_norm weights, but layer %d is missing one or more of them\n",
                    __func__, il);
                return false;
            }

        const bool has_q_norm = model.layers[(size_t) il].attn_q_norm != nullptr;
        const bool has_k_norm = model.layers[(size_t) il].attn_k_norm != nullptr;
        if (has_q_norm != has_k_norm) {
            LLAMA_LOG_ERROR("%s: DFlash graph requires symmetric Q/K norm presence, but layer %d has q_norm=%d k_norm=%d\n",
                    __func__, il, (int) has_q_norm, (int) has_k_norm);
            return false;
        }

            if (model.layers[(size_t) il].attn_norm_b != nullptr ||
                model.layers[(size_t) il].attn_q_norm_b != nullptr ||
                model.layers[(size_t) il].attn_k_norm_b != nullptr) {
                LLAMA_LOG_ERROR("%s: DFlash graph does not implement norm-bias tensors, but layer %d requires attn_norm_b/q_norm_b/k_norm_b\n",
                    __func__, il);
                return false;
            }

        if (dflash_layer_has_attention_bias(model.layers[(size_t) il])) {
            LLAMA_LOG_ERROR("%s: DFlash graph does not implement attention bias tensors, but layer %d requires them\n",
                    __func__, il);
            return false;
        }
    }

    return true;
}

bool llama_prepare_dflash_graph_inputs(
        struct llama_context & lctx,
        uint32_t n_tokens) {
    const bool kv_node_timing = llama_env_flag_enabled_local("IK_DFLASH_KV_NODE_TIMING");
    auto & profile = lctx.dflash_profile;
    const int32_t cross_ctx = lctx.dflash_visible_cross_ctx > 0
            ? lctx.dflash_visible_cross_ctx
            : std::max<int32_t>(1, (int32_t) lctx.cparams.n_ctx - (int32_t) lctx.model.hparams.dflash_block_size);
    ggml_tensor * kq_mask = lctx.dflash_kq_mask_tensor;
    ggml_tensor * kq_mask_swa = lctx.dflash_kq_mask_swa_tensor;

    if (kq_mask == nullptr) {
        LLAMA_LOG_ERROR("%s: DFlash graph inputs are not initialized\n", __func__);
        return false;
    }

    if (!validate_dflash_graph_contract(lctx)) {
        profile.graph_shape_failures++;
        return false;
    }

    if (!lctx.ensure_dflash_kv_cache_tensors(cross_ctx) || lctx.dflash_k_ctx_cache.empty() || lctx.dflash_v_ctx_cache.empty()) {
        LLAMA_LOG_ERROR("%s: DFlash K/V cache inputs are not initialized\n", __func__);
        return false;
    }

    const float * src = lctx.dflash_target_features;
    const float * append_src = lctx.dflash_target_append_features;
    const llama_pos * src_pos = lctx.dflash_target_positions;
    const size_t total_floats = lctx.dflash_target_features_n_floats;
    const size_t append_floats = lctx.dflash_target_append_features_n_floats;
    const size_t total_positions = lctx.dflash_target_positions_n;
    const int32_t n_rows = lctx.dflash_target_features_n_rows;
    const int32_t append_rows_available = lctx.dflash_target_append_features_n_rows;
    const int32_t width = (int32_t) lctx.model.hparams.dflash_n_target_features;
    const int32_t graph_cross_ctx = lctx.dflash_k_ctx_cache.front() != nullptr
            ? (int32_t) lctx.dflash_k_ctx_cache.front()->ne[2]
            : 0;
    const int32_t n_mask_tokens = (int32_t) kq_mask->ne[1];
    const int32_t n_kv_total = (int32_t) kq_mask->ne[0];
    const int64_t t_total_us = ggml_time_us();

    profile.graph_prepare_calls++;
    profile.last_n_rows = n_rows;
    profile.last_width = width;
    profile.last_cross_ctx = cross_ctx;
    profile.last_n_tokens = (int32_t) n_tokens;
    profile.last_n_kv_total = n_kv_total;

    llama_sync_dflash_workspace_if_pending(lctx);

    if (graph_cross_ctx != cross_ctx) {
        profile.graph_shape_failures++;

        LLAMA_LOG_ERROR("%s: DFlash graph cross_ctx drift (graph=%d configured=%d)\n",
                __func__, graph_cross_ctx, cross_ctx);
        return false;
    }
    if (n_rows <= 0) {
        profile.graph_shape_failures++;
        LLAMA_LOG_ERROR("%s: missing DFlash target feature rows\n", __func__);
        return false;
    }

    const bool have_full_src = src != nullptr && total_floats == (size_t) n_rows * (size_t) width;
    if (n_rows > cross_ctx || (src != nullptr && !have_full_src)) {
        profile.graph_shape_failures++;
        LLAMA_LOG_ERROR("%s: invalid DFlash target feature shape (rows=%d width=%d floats=%zu cross_ctx=%d)\n",
                __func__, n_rows, width, total_floats, cross_ctx);
        return false;
    }

    if (n_kv_total < cross_ctx + (int32_t) n_tokens) {
        profile.graph_mask_overflow++;
        LLAMA_LOG_ERROR("%s: invalid DFlash mask shape (n_kv_total=%d < cross_ctx+n_tokens=%d)\n",
                __func__, n_kv_total, cross_ctx + (int32_t) n_tokens);
        return false;
    }

    const int32_t left_pad = cross_ctx - n_rows;
    profile.last_left_pad = left_pad;

    const int64_t t_pos_us = ggml_time_us();
    lctx.dflash_pos_ctx_data.resize((size_t) cross_ctx);
    std::fill(lctx.dflash_pos_ctx_data.begin(), lctx.dflash_pos_ctx_data.end(), 0);
    if (src_pos == nullptr || total_positions != (size_t) n_rows) {
        profile.graph_pos_fallbacks++;
        profile.graph_shape_failures++;
        profile.last_pos_first = -1;
        profile.last_pos_last = -1;
        if (profile.graph_pos_fallbacks <= 3) {
            LLAMA_LOG_ERROR("%s: missing DFlash target positions (rows=%d positions=%zu cross_ctx=%d)\n",
                    __func__, n_rows, total_positions, cross_ctx);
        }
        return false;
    }

    profile.last_pos_first = src_pos[0];
    profile.last_pos_last = src_pos[n_rows - 1];
    for (int32_t i = 1; i < n_rows; ++i) {
        if (src_pos[i] <= src_pos[i - 1]) {
            profile.graph_pos_non_monotonic++;
            profile.graph_shape_failures++;
            if (profile.graph_pos_non_monotonic <= 3) {
                LLAMA_LOG_ERROR("%s: DFlash target positions are not strictly increasing (rows=%d first=%d last=%d)\n",
                        __func__, n_rows, (int) src_pos[0], (int) src_pos[n_rows - 1]);
            }
            return false;
        }
    }
    std::copy(src_pos, src_pos + n_rows, lctx.dflash_pos_ctx_data.begin() + (ptrdiff_t) left_pad);
    profile.graph_pos_copy_us += (uint64_t) (ggml_time_us() - t_pos_us);
    profile.graph_pos_bytes += lctx.dflash_pos_ctx_data.size() * sizeof(llama_pos);

    const llama_dflash_kv_cache_transition cache_plan = llama_plan_dflash_kv_cache_transition(
        cross_ctx,
        lctx.dflash_kv_cache_n_filled,
        lctx.dflash_kv_cache_write_pos,
        lctx.dflash_kv_cache_valid,
        lctx.dflash_kv_cache_applied_window_version,
        lctx.dflash_target_window_version,
        lctx.dflash_target_window_keep_rows,
        lctx.dflash_target_window_append_rows,
        lctx.dflash_target_window_replace,
        n_rows);

    const bool have_append_src = append_src != nullptr &&
        append_rows_available == cache_plan.append_rows &&
        append_floats == (size_t) cache_plan.append_rows * (size_t) width;

    const int32_t update_rows = cache_plan.cache_up_to_date
            ? 0
        : (cache_plan.rebuild_cache ? n_rows : cache_plan.append_rows);
    const size_t max_nodes = lctx.model.max_nodes((int) std::max<int32_t>(1, cross_ctx)) + 24 * lctx.model.hparams.n_layer;
    const size_t meta_size = ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    if (lctx.dflash_buf_compute_meta.size() != meta_size) {
        lctx.dflash_buf_compute_meta.resize(meta_size);
    }

    if (lctx.dflash_sched == nullptr || lctx.dflash_kv_cache_reserved_rows != cross_ctx) {
        std::vector<ggml_backend_buffer_type_t> backend_buft;
        backend_buft.reserve(lctx.backends.size());
        for (auto * backend : lctx.backends) {
            if (ggml_backend_is_cpu(backend)) {
                backend_buft.push_back(llama_default_buffer_type_cpu(true));
            } else {
                backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
            }
        }

        if (lctx.dflash_sched != nullptr) {
            ggml_backend_sched_free(lctx.dflash_sched);
            lctx.dflash_sched = nullptr;
        }
        lctx.dflash_kv_graph = nullptr;
        lctx.dflash_kv_graph_rows = 0;
        lctx.dflash_kv_graph_write_pos = 0;

        const int32_t saved_update_rows = lctx.dflash_kv_cache_update_rows;
        lctx.dflash_kv_cache_update_rows = cross_ctx;
        const int64_t t_build_us = ggml_time_us();
        ggml_cgraph * gf_reserve = llm_build_context::llama_build_graph_dflash_kv_cache(lctx);
        profile.graph_kv_cache_build_us += (uint64_t) (ggml_time_us() - t_build_us);
        lctx.dflash_kv_cache_update_rows = saved_update_rows;
        if (gf_reserve == nullptr) {
            profile.graph_shape_failures++;
            LLAMA_LOG_ERROR("%s: failed to build DFlash K/V cache reserve graph\n", __func__);
            return false;
        }

        const int64_t t_reserve_us = ggml_time_us();
        lctx.dflash_sched = ggml_backend_sched_new(lctx.backends.data(), backend_buft.data(), lctx.backends.size(), max_nodes, false);
        const bool reserved = lctx.dflash_sched != nullptr && ggml_backend_sched_reserve(lctx.dflash_sched, gf_reserve);
        profile.graph_kv_cache_reserve_us += (uint64_t) (ggml_time_us() - t_reserve_us);
        if (!reserved) {
            profile.graph_shape_failures++;
            LLAMA_LOG_ERROR("%s: failed to initialize DFlash K/V scheduler\n", __func__);
            return false;
        }
        lctx.dflash_kv_cache_reserved_rows = cross_ctx;
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
            profile.graph_shape_failures++;
            LLAMA_LOG_ERROR("%s: missing DFlash appended target features for cached update (rows=%d append_rows=%d floats=%zu)\n",
                    __func__, n_rows, update_rows, append_floats);
            return false;
        }

        if (cache_plan.rebuild_cache) {
            llama_reset_dflash_kv_cache_state(&lctx);
        }

        lctx.dflash_kv_cache_update_rows = update_rows;
        ggml_cgraph * gf_kv = nullptr;
        const bool can_reuse_kv_graph = lctx.dflash_kv_graph != nullptr &&
                lctx.dflash_kv_graph_rows == update_rows &&
                lctx.dflash_kv_graph_write_pos == lctx.dflash_kv_cache_write_pos;
        if (can_reuse_kv_graph) {
            gf_kv = lctx.dflash_kv_graph;
        } else {
            const int64_t t_build_us = ggml_time_us();
            gf_kv = llm_build_context::llama_build_graph_dflash_kv_cache(lctx);
            profile.graph_kv_cache_build_us += (uint64_t) (ggml_time_us() - t_build_us);
            if (gf_kv == nullptr || lctx.dflash_kv_input_target_features == nullptr || lctx.dflash_kv_input_pos_ctx == nullptr) {
                profile.graph_shape_failures++;
                LLAMA_LOG_ERROR("%s: failed to build DFlash K/V cache graph\n", __func__);
                return false;
            }

            const int64_t t_reset_us = ggml_time_us();
            ggml_backend_sched_reset(lctx.dflash_sched);
            profile.graph_kv_cache_reset_us += (uint64_t) (ggml_time_us() - t_reset_us);

            const int64_t t_alloc_us = ggml_time_us();
            ggml_backend_sched_alloc_graph(lctx.dflash_sched, gf_kv);
            profile.graph_kv_cache_alloc_us += (uint64_t) (ggml_time_us() - t_alloc_us);

            lctx.dflash_kv_graph = gf_kv;
            lctx.dflash_kv_graph_rows = update_rows;
            lctx.dflash_kv_graph_write_pos = lctx.dflash_kv_cache_write_pos;
        }

        ggml_backend_t kv_feature_backend = llama_backend_for_tensor(lctx, lctx.dflash_kv_input_target_features);
        const int64_t t_feature_upload_us = ggml_time_us();
        if (kv_feature_backend != nullptr) {
            ggml_backend_tensor_set_async(kv_feature_backend, lctx.dflash_kv_input_target_features, update_src, 0, ggml_nbytes(lctx.dflash_kv_input_target_features));
        } else {
            ggml_backend_tensor_set(lctx.dflash_kv_input_target_features, update_src, 0, ggml_nbytes(lctx.dflash_kv_input_target_features));
        }
        profile.graph_kv_cache_feature_upload_us += (uint64_t) (ggml_time_us() - t_feature_upload_us);
        profile.graph_feature_bytes += (size_t) update_rows * (size_t) width * sizeof(float);

        ggml_backend_t kv_pos_backend = llama_backend_for_tensor(lctx, lctx.dflash_kv_input_pos_ctx);
        const int64_t t_pos_upload_us = ggml_time_us();
        if (kv_pos_backend != nullptr) {
            ggml_backend_tensor_set_async(kv_pos_backend, lctx.dflash_kv_input_pos_ctx, update_pos, 0, ggml_nbytes(lctx.dflash_kv_input_pos_ctx));
        } else {
            ggml_backend_tensor_set(lctx.dflash_kv_input_pos_ctx, update_pos, 0, ggml_nbytes(lctx.dflash_kv_input_pos_ctx));
        }
        profile.graph_kv_cache_pos_upload_us += (uint64_t) (ggml_time_us() - t_pos_upload_us);

        const int64_t t_kv_cache_us = ggml_time_us();
        llama_dflash_kv_node_profiler kv_node_profiler;
        if (kv_node_timing) {
            kv_node_profiler.profile = &profile;
            ggml_backend_sched_set_eval_callback(lctx.dflash_sched, llama_dflash_kv_node_eval_callback, &kv_node_profiler);
        }
        llama_graph_compute_sched(lctx, lctx.dflash_sched, gf_kv, lctx.cparams.n_threads);
        if (kv_node_timing) {
            ggml_backend_sched_set_eval_callback(lctx.dflash_sched, nullptr, nullptr);
        }
        profile.graph_kv_cache_compute_us += (uint64_t) (ggml_time_us() - t_kv_cache_us);

        const int64_t t_sync_us = ggml_time_us();
        ggml_backend_sched_synchronize(lctx.dflash_sched);
        profile.graph_kv_cache_sync_us += (uint64_t) (ggml_time_us() - t_sync_us);
        profile.graph_kv_cache_calls++;

        lctx.dflash_kv_cache_n_filled = std::min(cross_ctx, lctx.dflash_kv_cache_n_filled + update_rows);
        lctx.dflash_kv_cache_write_pos = (lctx.dflash_kv_cache_write_pos + update_rows) % cross_ctx;
        lctx.dflash_kv_cache_applied_window_version = lctx.dflash_target_window_version;
        lctx.dflash_kv_cache_valid = true;
        lctx.dflash_kv_cache_view_n_filled = lctx.dflash_kv_cache_n_filled;
        lctx.dflash_kv_cache_view_write_pos = lctx.dflash_kv_cache_write_pos;
        lctx.dflash_kv_cache_view_valid = true;
    }

    if (lctx.dflash_kv_cache_view_valid &&
            !lctx.dflash_k_ctx_workspace.empty() && !lctx.dflash_v_ctx_workspace.empty()) {
        const bool need_workspace_refresh = !lctx.dflash_kv_workspace_valid ||
                lctx.dflash_kv_workspace_n_filled != lctx.dflash_kv_cache_view_n_filled ||
                lctx.dflash_kv_workspace_write_pos != lctx.dflash_kv_cache_view_write_pos ||
                lctx.dflash_kv_workspace_applied_window_version != lctx.dflash_kv_cache_applied_window_version;

        if (need_workspace_refresh) {
            const size_t max_nodes = lctx.model.max_nodes((int) std::max<int32_t>(1, cross_ctx)) + 16 * lctx.model.hparams.n_layer;
            const size_t meta_size = ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false);
            if (lctx.dflash_workspace_buf_compute_meta.size() != meta_size) {
                lctx.dflash_workspace_buf_compute_meta.resize(meta_size);
            }

            ggml_cgraph * gf_workspace = nullptr;
            const bool can_reuse_workspace_graph = lctx.dflash_kv_workspace_graph != nullptr &&
                    lctx.dflash_kv_workspace_graph_rows == lctx.dflash_kv_cache_view_n_filled &&
                    lctx.dflash_kv_workspace_graph_write_pos == lctx.dflash_kv_cache_view_write_pos;

            if (can_reuse_workspace_graph) {
                gf_workspace = lctx.dflash_kv_workspace_graph;
            } else {
                const int64_t t_build_us = ggml_time_us();
                gf_workspace = llm_build_context::llama_build_graph_dflash_kv_workspace(lctx);
                profile.graph_kv_workspace_build_us += (uint64_t) (ggml_time_us() - t_build_us);
                if (gf_workspace == nullptr) {
                    profile.graph_shape_failures++;
                    LLAMA_LOG_ERROR("%s: failed to build DFlash K/V workspace graph\n", __func__);
                    return false;
                }

                std::vector<ggml_backend_buffer_type_t> backend_buft;
                backend_buft.reserve(lctx.backends.size());
                for (auto * backend : lctx.backends) {
                    if (ggml_backend_is_cpu(backend)) {
                        backend_buft.push_back(llama_default_buffer_type_cpu(true));
                    } else {
                        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
                    }
                }

                if (lctx.dflash_workspace_sched == nullptr) {
                    lctx.dflash_workspace_sched = ggml_backend_sched_new(lctx.backends.data(), backend_buft.data(), lctx.backends.size(), max_nodes, false);
                }

                if (lctx.dflash_kv_workspace_reserved_rows != cross_ctx) {
                    const bool saved_view_valid = lctx.dflash_kv_cache_view_valid;
                    const int32_t saved_view_rows = lctx.dflash_kv_cache_view_n_filled;
                    const int32_t saved_view_write_pos = lctx.dflash_kv_cache_view_write_pos;

                    lctx.dflash_kv_cache_view_valid = true;
                    lctx.dflash_kv_cache_view_n_filled = cross_ctx;
                    lctx.dflash_kv_cache_view_write_pos = cross_ctx > 1 ? 1 : 0;

                    const int64_t t_reserve_build_us = ggml_time_us();
                    ggml_cgraph * gf_workspace_reserve = llm_build_context::llama_build_graph_dflash_kv_workspace(lctx);
                    profile.graph_kv_workspace_build_us += (uint64_t) (ggml_time_us() - t_reserve_build_us);

                    lctx.dflash_kv_cache_view_valid = saved_view_valid;
                    lctx.dflash_kv_cache_view_n_filled = saved_view_rows;
                    lctx.dflash_kv_cache_view_write_pos = saved_view_write_pos;

                    const int64_t t_reserve_us = ggml_time_us();
                    const bool reserved = lctx.dflash_workspace_sched != nullptr &&
                            gf_workspace_reserve != nullptr &&
                            ggml_backend_sched_reserve(lctx.dflash_workspace_sched, gf_workspace_reserve);
                    profile.graph_kv_workspace_reserve_us += (uint64_t) (ggml_time_us() - t_reserve_us);
                    if (!reserved) {
                        profile.graph_shape_failures++;
                        LLAMA_LOG_ERROR("%s: failed to initialize DFlash K/V workspace scheduler\n", __func__);
                        return false;
                    }

                    lctx.dflash_kv_workspace_reserved_rows = cross_ctx;
                }

                const int64_t t_reset_us = ggml_time_us();
                ggml_backend_sched_reset(lctx.dflash_workspace_sched);
                profile.graph_kv_workspace_reset_us += (uint64_t) (ggml_time_us() - t_reset_us);

                const int64_t t_alloc_us = ggml_time_us();
                ggml_backend_sched_alloc_graph(lctx.dflash_workspace_sched, gf_workspace);
                profile.graph_kv_workspace_alloc_us += (uint64_t) (ggml_time_us() - t_alloc_us);

                lctx.dflash_kv_workspace_graph = gf_workspace;
                lctx.dflash_kv_workspace_graph_rows = lctx.dflash_kv_cache_view_n_filled;
                lctx.dflash_kv_workspace_graph_write_pos = lctx.dflash_kv_cache_view_write_pos;
            }

            const int64_t t_workspace_us = ggml_time_us();
            llama_graph_compute_sched(lctx, lctx.dflash_workspace_sched, gf_workspace, lctx.cparams.n_threads);
            profile.graph_kv_workspace_compute_us += (uint64_t) (ggml_time_us() - t_workspace_us);
            lctx.dflash_kv_workspace_sync_pending = true;
            profile.graph_kv_workspace_calls++;

            lctx.dflash_kv_workspace_n_filled = lctx.dflash_kv_cache_view_n_filled;
            lctx.dflash_kv_workspace_write_pos = lctx.dflash_kv_cache_view_write_pos;
            lctx.dflash_kv_workspace_applied_window_version = lctx.dflash_kv_cache_applied_window_version;
            lctx.dflash_kv_workspace_valid = true;
        }
    }

    const int64_t t_mask_us = ggml_time_us();
    const int32_t full_visible_first = left_pad;
    const int32_t full_visible_last = cross_ctx + (int32_t) n_tokens - 1;
    lctx.dflash_kq_mask_data.assign((size_t) n_kv_total * (size_t) n_mask_tokens, -INFINITY);
    int32_t visible_kv_max = 0;
    for (uint32_t j = 0; j < n_tokens; ++j) {
        float * row = lctx.dflash_kq_mask_data.data() + (size_t) j * (size_t) n_kv_total;
        const int32_t visible_kv = cross_ctx + (int32_t) n_tokens;
        visible_kv_max = std::max(visible_kv_max, visible_kv);
        profile.graph_visible_kv_sum += (uint64_t) visible_kv;
        for (int32_t i = full_visible_first; i <= full_visible_last; ++i) {
            row[i] = 0.0f;
        }
    }
    ggml_backend_tensor_set(kq_mask, lctx.dflash_kq_mask_data.data(), 0, ggml_nbytes(kq_mask));
    profile.graph_mask_build_us += (uint64_t) (ggml_time_us() - t_mask_us);
    profile.graph_mask_bytes += ggml_nbytes(kq_mask);

    if (kq_mask_swa != nullptr) {
        lctx.dflash_kq_mask_swa_data.assign((size_t) n_kv_total * (size_t) n_mask_tokens, -INFINITY);
        const int32_t swa_window = (int32_t) lctx.model.hparams.n_swa;
        const int32_t draft_pos_base = (int32_t) profile.last_pos_last;
        for (uint32_t j = 0; j < n_tokens; ++j) {
            float * row = lctx.dflash_kq_mask_swa_data.data() + (size_t) j * (size_t) n_kv_total;
            const int32_t q_pos = draft_pos_base + (int32_t) j;

            for (int32_t k = left_pad; k < cross_ctx; ++k) {
                const int32_t k_pos = (int32_t) lctx.dflash_pos_ctx_data[(size_t) k];
                if (q_pos - k_pos < swa_window) {
                    row[k] = 0.0f;
                }
            }

            for (int32_t k = cross_ctx; k < cross_ctx + (int32_t) n_tokens; ++k) {
                const int32_t block_k = k - cross_ctx;
                if (block_k <= (int32_t) j) {
                    row[k] = 0.0f;
                }
            }
        }

        ggml_backend_tensor_set(kq_mask_swa, lctx.dflash_kq_mask_swa_data.data(), 0, ggml_nbytes(kq_mask_swa));
        profile.graph_mask_bytes += ggml_nbytes(kq_mask_swa);
    }

    profile.graph_visible_kv_max = std::max<uint64_t>(profile.graph_visible_kv_max, (uint64_t) visible_kv_max);
    profile.graph_prepare_total_us += (uint64_t) (ggml_time_us() - t_total_us);

    if (profile.graph_prepare_calls == 1 && llama_dflash_stats_log_enabled()) {
        int32_t n_swa_layers = 0;
        for (int32_t il = 0; il < lctx.model.hparams.n_layer; ++il) {
            n_swa_layers += lctx.model.hparams.swa_layers[(size_t) il] ? 1 : 0;
        }

        LLAMA_LOG_INFO("%s: DFlash graph contract rows=%d width=%d cross_ctx=%d n_tokens=%u left_pad=%d n_kv_total=%d draft_n_ctx=%u pos=%s [%d..%d] full_mask=[%d..%d] swa_window=%u swa_layers=%d\n",
            __func__, n_rows, width, cross_ctx, n_tokens, left_pad, n_kv_total, lctx.cparams.n_ctx,
                (src_pos != nullptr && total_positions == (size_t) n_rows) ? "target" : "synthetic",
                (int) profile.last_pos_first, (int) profile.last_pos_last,
                full_visible_first, full_visible_last,
                lctx.model.hparams.n_swa,
                n_swa_layers);
    }

    return true;
}
