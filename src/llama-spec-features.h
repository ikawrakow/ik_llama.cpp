#pragma once

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <vector>

struct llama_context;

enum llama_spec_feature_kind {
    LLAMA_SPEC_FEATURE_NONE,
    LLAMA_SPEC_FEATURE_HIDDEN_STATE,
};

struct llama_spec_feature_row_view {
    llama_seq_id seq_id = 0;
    llama_pos pos = -1;
    const float * data = nullptr;
};

struct llama_spec_feature_view {
    llama_spec_feature_kind kind = LLAMA_SPEC_FEATURE_NONE;
    int32_t width = 0;
    std::vector<llama_spec_feature_row_view> rows;
};

struct llama_dflash_profile_stats {
        uint64_t decode_internal_chunks = 0;
        uint64_t decode_graph_rebuilds = 0;
        uint64_t decode_sync_profile_points = 0;
        uint64_t decode_prelude_us = 0;
        uint64_t decode_sched_reset_us = 0;
        uint64_t decode_build_graph_us = 0;
        uint64_t decode_sched_alloc_graph_us = 0;
        uint64_t decode_set_inputs_us = 0;
        uint64_t decode_graph_compute_us = 0;
        uint64_t decode_result_us = 0;
        uint64_t decode_embedding_us = 0;
        uint64_t decode_final_sched_reset_us = 0;

        uint64_t decode_output_reserve_calls = 0;
        uint64_t decode_output_reserve_us = 0;
        uint64_t decode_output_reserve_reallocs = 0;
        uint64_t decode_output_reserve_realloc_bytes = 0;
        uint64_t decode_prepare_calls = 0;
        uint64_t decode_prepare_us = 0;
        uint64_t decode_prepare_failures = 0;

        uint64_t set_target_copy_calls = 0;
        uint64_t set_target_copy_us = 0;
        uint64_t set_target_rows = 0;
        uint64_t set_target_copy_bytes = 0;
        uint64_t set_target_missing_positions = 0;
        uint64_t set_target_non_monotonic_positions = 0;

        uint64_t capture_prepare_calls = 0;
        uint64_t capture_prepare_sync_us = 0;
        uint64_t capture_prepare_failures = 0;
        uint64_t capture_layer_shape_mismatch = 0;
        uint64_t capture_layer_batch_mismatch = 0;
        uint64_t capture_prompt_batches = 0;
        uint64_t capture_prompt_shape_changes = 0;
        uint64_t capture_verify_batches = 0;
        uint64_t capture_verify_shape_changes = 0;
        uint64_t capture_materialize_calls = 0;
        uint64_t capture_materialize_rows = 0;
        uint64_t capture_materialize_bytes = 0;
        uint64_t capture_materialize_us = 0;
        uint64_t capture_materialize_failures = 0;

        uint64_t graph_prepare_calls = 0;
        uint64_t graph_prepare_total_us = 0;
        uint64_t graph_feature_copy_us = 0;
        uint64_t graph_pos_copy_us = 0;
        uint64_t graph_mask_build_us = 0;
        uint64_t graph_kv_cache_build_us = 0;
        uint64_t graph_kv_cache_reserve_us = 0;
        uint64_t graph_kv_cache_reset_us = 0;
        uint64_t graph_kv_cache_alloc_us = 0;
        uint64_t graph_kv_cache_feature_upload_us = 0;
        uint64_t graph_kv_cache_pos_upload_us = 0;
        uint64_t graph_kv_cache_compute_us = 0;
        uint64_t graph_kv_cache_sync_us = 0;
        uint64_t graph_kv_cache_read_concat_pad_us = 0;
        uint64_t graph_kv_cache_read_concat_pad_calls = 0;
        uint64_t graph_kv_cache_cached_bytes = 0;
        uint64_t graph_kv_cache_calls = 0;
        uint64_t graph_kv_node_fused_target_calls = 0;
        uint64_t graph_kv_node_fused_target_us = 0;
        uint64_t graph_kv_node_k_proj_calls = 0;
        uint64_t graph_kv_node_k_proj_us = 0;
        uint64_t graph_kv_node_k_norm_calls = 0;
        uint64_t graph_kv_node_k_norm_us = 0;
        uint64_t graph_kv_node_k_rope_calls = 0;
        uint64_t graph_kv_node_k_rope_us = 0;
        uint64_t graph_kv_node_v_proj_calls = 0;
        uint64_t graph_kv_node_v_proj_us = 0;
        uint64_t graph_kv_node_k_store_calls = 0;
        uint64_t graph_kv_node_k_store_us = 0;
        uint64_t graph_kv_node_v_store_calls = 0;
        uint64_t graph_kv_node_v_store_us = 0;
        uint64_t graph_feature_bytes = 0;
        uint64_t graph_pos_bytes = 0;
        uint64_t graph_mask_bytes = 0;
        uint64_t graph_visible_kv_sum = 0;
        uint64_t graph_visible_kv_max = 0;
        uint64_t graph_pos_fallbacks = 0;
        uint64_t graph_pos_non_monotonic = 0;
        uint64_t graph_shape_failures = 0;
        uint64_t graph_mask_overflow = 0;

        int32_t last_n_rows = 0;
        int32_t last_width = 0;
        int32_t last_cross_ctx = 0;
        int32_t last_left_pad = 0;
        int32_t last_n_tokens = 0;
        int32_t last_n_kv_total = 0;
        int32_t last_kv_cache_host_layers = 0;
        int32_t capture_prompt_last_rows = 0;
        int32_t capture_prompt_last_width = 0;
        int32_t capture_verify_last_rows = 0;
        int32_t capture_verify_last_width = 0;
        llama_pos last_pos_first = -1;
        llama_pos last_pos_last = -1;
};

struct llama_dflash_window_update {
        uint64_t version = 0;
        int32_t keep_rows = 0;
        int32_t append_rows = 0;
        bool replace = false;
        const float * append_features = nullptr;
        size_t append_floats = 0;
};

struct llama_dflash_kv_cache_transition {
        bool cache_up_to_date = false;
        bool rebuild_cache = false;
        int32_t append_rows = 0;
        int32_t next_n_filled = 0;
        int32_t next_write_pos = 0;
};

static inline llama_dflash_kv_cache_transition llama_plan_dflash_kv_cache_transition(
                int32_t cross_ctx,
                int32_t current_n_filled,
                int32_t current_write_pos,
                bool cache_valid,
                uint64_t applied_window_version,
                uint64_t target_window_version,
                int32_t keep_rows,
                int32_t append_rows,
                bool replace,
                int32_t n_rows) {
        llama_dflash_kv_cache_transition plan;

        const int32_t safe_cross_ctx = std::max<int32_t>(1, cross_ctx);
        const int32_t bounded_n_filled = std::clamp(current_n_filled, 0, safe_cross_ctx);
        const int32_t bounded_append_rows = std::clamp(append_rows, 0, n_rows);
        const int32_t bounded_keep_rows = std::clamp(keep_rows, 0, n_rows);
        const int32_t expected_keep_rows = std::min(bounded_n_filled, std::max<int32_t>(0, safe_cross_ctx - bounded_append_rows));

        plan.cache_up_to_date = cache_valid && applied_window_version == target_window_version;
        plan.rebuild_cache = !cache_valid || replace || bounded_append_rows <= 0 || bounded_append_rows > n_rows;
        if (!plan.rebuild_cache && bounded_keep_rows != expected_keep_rows) {
                plan.rebuild_cache = true;
        }

        plan.append_rows = bounded_append_rows;
        if (plan.cache_up_to_date) {
                plan.next_n_filled = bounded_n_filled;
                plan.next_write_pos = safe_cross_ctx > 0
                                ? ((current_write_pos % safe_cross_ctx) + safe_cross_ctx) % safe_cross_ctx
                                : 0;
        } else if (plan.rebuild_cache) {
                plan.next_n_filled = std::min(safe_cross_ctx, n_rows);
                plan.next_write_pos = plan.next_n_filled % safe_cross_ctx;
        } else {
                plan.next_n_filled = std::min(safe_cross_ctx, bounded_n_filled + bounded_append_rows);
                plan.next_write_pos = (current_write_pos + bounded_append_rows) % safe_cross_ctx;
        }

        return plan;
}

llama_dflash_kv_cache_transition llama_plan_dflash_kv_cache_transition_for_ctx(
                const struct llama_context * ctx,
                const llama_dflash_window_update & window_update,
                int32_t n_rows);

uint32_t llama_mtp_state_n_embd(const struct llama_context * ctx);

void llama_dflash_profile_reset(struct llama_context * ctx);

void llama_reset_dflash_kv_cache_state(struct llama_context * ctx);

void llama_set_dflash_visible_cross_ctx(
        struct llama_context * ctx,
        int32_t cross_ctx);

int32_t llama_get_dflash_visible_cross_ctx(
        const struct llama_context * ctx);

bool llama_dflash_profile_get_stats(
                const struct llama_context * ctx,
                llama_dflash_profile_stats * stats);

int32_t llama_model_dflash_block_size(const struct llama_model * model);

int32_t llama_model_dflash_mask_token_id(const struct llama_model * model);

int32_t llama_model_dflash_n_target_layers(const struct llama_model * model);

int32_t llama_model_dflash_n_target_features(const struct llama_model * model);

int32_t llama_model_dflash_target_layer_ids(
        const struct llama_model * model,
        int32_t * layer_ids,
        int32_t capacity);

enum llama_dflash_io_mode {
    LLAMA_DFLASH_IO_MODE_INVALID = 0,
    LLAMA_DFLASH_IO_MODE_SHARED,
    LLAMA_DFLASH_IO_MODE_SELF_CONTAINED,
    LLAMA_DFLASH_IO_MODE_MIXED,
};

int32_t llama_model_dflash_target_mask_token_id(const struct llama_model * model);

int32_t llama_model_dflash_io_mode(
        const struct llama_model * draft_model,
        const struct llama_model * target_model);

bool llama_model_dflash_io_tensors_match(
        const struct llama_model * draft_model,
        int32_t n_embd,
        int32_t n_vocab);

bool llama_model_share_dflash_io_tensors(
        struct llama_model * draft_model,
        const struct llama_model * target_model);

bool llama_set_draft_input_hidden_state_copy(
        struct llama_context * ctx,
        const float * hidden_state,
        size_t n_floats);

bool llama_set_dflash_target_features_copy(
        struct llama_context * ctx,
        const float * target_features,
        size_t n_floats,
        int32_t n_rows,
        const llama_pos * target_positions,
        const llama_dflash_window_update * window_update = nullptr);

bool llama_set_dflash_target_features_view(
        struct llama_context * ctx,
        const float * target_features,
        size_t n_floats,
        int32_t n_rows,
        const llama_pos * target_positions,
        const llama_dflash_window_update * window_update = nullptr);

bool llama_set_dflash_capture_layers(
        struct llama_context * ctx,
        const int32_t * layer_ids,
        int32_t n_layers);

void llama_clear_dflash_capture(struct llama_context * ctx);

void llama_begin_dflash_capture_batch(struct llama_context * ctx);

void llama_finish_dflash_capture_batch(
        struct llama_context * ctx,
        bool is_prompt_warmup);

bool llama_spec_get_hidden_feature_view(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_spec_feature_view & view);

bool llama_spec_get_dflash_feature_view(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_spec_feature_view & view);

bool llama_spec_get_dflash_feature_view_for_seq(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_seq_id             seq_id,
        llama_spec_feature_view & view);

bool llama_spec_get_hidden_feature_view_for_seq(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_seq_id             seq_id,
        llama_spec_feature_view & view);

bool llama_spec_get_hidden_feature_view_from_output_index(
        struct llama_context   * ctx,
        int32_t                  output_index,
        llama_seq_id             seq_id,
        llama_pos                pos,
        llama_spec_feature_view & view);

bool llama_spec_copy_hidden_rows_from_output_indices(
        struct llama_context * ctx,
        const std::vector<int32_t> & output_indices,
        std::vector<float> & hidden_rows);

bool llama_spec_copy_dflash_rows_from_output_indices(
        struct llama_context * ctx,
        const std::vector<int32_t> & output_indices,
        std::vector<float> & hidden_rows);
