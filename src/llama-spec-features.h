#pragma once

#include "llama.h"

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

#include "llama-spec-features-dflash.h"

uint32_t llama_mtp_state_n_embd(const struct llama_context * ctx);

bool llama_set_draft_input_hidden_state_copy(
        struct llama_context * ctx,
        const float * hidden_state,
        size_t n_floats);

bool llama_spec_get_hidden_feature_view(
        struct llama_context   * ctx,
        const llama_batch      & batch,
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
