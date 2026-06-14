#pragma once

#include <cstdint>
#include <cstring>

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

static inline bool llama_dflash_tensor_name_has_prefix(const struct ggml_tensor * tensor, const char * prefix) {
    if (tensor == nullptr || prefix == nullptr || prefix[0] == '\0') {
        return false;
    }

    return std::strncmp(tensor->name, prefix, std::strlen(prefix)) == 0;
}

static inline bool llama_dflash_tensor_name_matches_label(const struct ggml_tensor * tensor, const char * label) {
    if (!llama_dflash_tensor_name_has_prefix(tensor, label)) {
        return false;
    }

    const size_t label_len = std::strlen(label);
    const char next = tensor->name[label_len];
    return next == '\0' || next == '-';
}

static inline llama_dflash_kv_node_kind llama_dflash_kv_node_kind_from_tensor(const struct ggml_tensor * tensor) {
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

static inline void llama_dflash_kv_node_profile_add(
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

static inline llama_dflash_main_node_kind llama_dflash_main_node_kind_from_tensor(const struct ggml_tensor * tensor) {
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

static inline void llama_dflash_main_node_profile_add(
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

static inline bool llama_dflash_kv_node_eval_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
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

static inline bool llama_dflash_main_node_eval_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
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
