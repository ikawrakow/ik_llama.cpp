#pragma once

#include "llama.h"

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

uint32_t llama_mtp_state_n_embd(const struct llama_context * ctx);

int32_t llama_model_dflash_block_size(const struct llama_model * model);

int32_t llama_model_dflash_mask_token_id(const struct llama_model * model);

int32_t llama_model_dflash_n_target_layers(const struct llama_model * model);

int32_t llama_model_dflash_n_target_features(const struct llama_model * model);

int32_t llama_model_dflash_target_layer_ids(
        const struct llama_model * model,
        int32_t * layer_ids,
        int32_t capacity);

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
        const llama_pos * target_positions);

bool llama_set_dflash_capture_layers(
        struct llama_context * ctx,
        const int32_t * layer_ids,
        int32_t n_layers);

void llama_clear_dflash_capture(struct llama_context * ctx);

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
