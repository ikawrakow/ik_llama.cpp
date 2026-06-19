#pragma once

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <vector>

struct llama_context;
struct llama_model;
struct ggml_tensor;
struct llama_spec_feature_view;

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

void llama_reset_dflash_kv_cache_state(struct llama_context * ctx);
void llama_set_dflash_visible_cross_ctx(struct llama_context * ctx, int32_t cross_ctx);
int32_t llama_get_dflash_visible_cross_ctx(const struct llama_context * ctx);

int32_t llama_model_dflash_block_size(const struct llama_model * model);
int32_t llama_model_dflash_mask_token_id(const struct llama_model * model);
int32_t llama_model_dflash_n_target_layers(const struct llama_model * model);
int32_t llama_model_dflash_n_target_features(const struct llama_model * model);
int32_t llama_model_dflash_target_layer_ids(const struct llama_model * model, int32_t * layer_ids, int32_t capacity);
int32_t llama_model_dflash_target_mask_token_id(const struct llama_model * model);

enum llama_dflash_io_mode {
    LLAMA_DFLASH_IO_MODE_INVALID = 0,
    LLAMA_DFLASH_IO_MODE_SHARED,
    LLAMA_DFLASH_IO_MODE_SELF_CONTAINED,
    LLAMA_DFLASH_IO_MODE_MIXED,
};

int32_t llama_model_dflash_io_mode(const struct llama_model * draft_model, const struct llama_model * target_model);
bool llama_model_dflash_io_tensors_match(const struct llama_model * draft_model, int32_t n_embd, int32_t n_vocab);
bool llama_model_share_dflash_io_tensors(struct llama_model * draft_model, const struct llama_model * target_model);

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

bool llama_set_dflash_capture_layers(struct llama_context * ctx, const int32_t * layer_ids, int32_t n_layers);
void llama_clear_dflash_capture(struct llama_context * ctx);
void llama_begin_dflash_capture_batch(struct llama_context * ctx);
void llama_finish_dflash_capture_batch(struct llama_context * ctx, bool is_prompt_warmup);

bool llama_spec_get_dflash_feature_view(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_spec_feature_view & view);

bool llama_spec_get_dflash_feature_view_for_seq(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_seq_id             seq_id,
        llama_spec_feature_view & view);

bool llama_spec_copy_dflash_rows_from_output_indices(
        struct llama_context * ctx,
        const std::vector<int32_t> & output_indices,
        std::vector<float> & hidden_rows);
