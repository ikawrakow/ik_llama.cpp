#pragma once

#include "llama.h"
#include "llama-spec-features.h"
#include "common.h"
#include "spec-tuner.h"

struct common_speculative;

using common_speculative_feature_kind = llama_spec_feature_kind;
using common_speculative_feature_row_view = llama_spec_feature_row_view;
using common_speculative_feature_view = llama_spec_feature_view;

static constexpr common_speculative_feature_kind COMMON_SPECULATIVE_FEATURE_NONE = LLAMA_SPEC_FEATURE_NONE;
static constexpr common_speculative_feature_kind COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE = LLAMA_SPEC_FEATURE_HIDDEN_STATE;

struct common_speculative_draft_span
{
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;
    common_speculative_stage_role role = COMMON_SPECULATIVE_STAGE_ROLE_AUTO;
    size_t token_offset = 0;
    size_t n_tokens = 0;
    size_t impl_index = 0;
    bool mutates_companion_state = false;
};

struct common_speculative_draft_result
{
    llama_tokens tokens;
    std::vector<common_speculative_draft_span> spans;
    bool combined = false;

    void clear()
    {
        tokens.clear();
        spans.clear();
        combined = false;
    }
};

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

common_speculative_draft_result common_speculative_draft_ex(
    common_speculative *spec,
    common_params_speculative &params,
    const llama_tokens &prompt,
    llama_token id_last,
    llama_pos draft_base_pos = -1,
    llama_seq_id draft_seq_id = 0);

// sample up to n_draft tokens and add them to the batch using the draft model
// draft_base_pos/draft_seq_id override the MTP position for id_last
llama_tokens common_speculative_draft(
                     common_speculative * spec,
                     common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                            llama_pos     draft_base_pos = -1,
                            llama_seq_id  draft_seq_id = 0);

common_speculative_type common_speculative_draft_result_primary_type(const common_speculative_draft_result &result);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

bool common_speculative_ensure_sequence_hidden(
    common_speculative * spec,
    llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos pos);

bool common_speculative_capture_output_hidden(
    common_speculative * spec,
    llama_context * ctx,
    int32_t output_index,
    llama_seq_id seq_id,
    llama_pos pos);

bool common_speculative_copy_output_hidden_rows(
    const common_speculative * spec,
    llama_context * ctx,
    const std::vector<int32_t> & output_indices,
    std::vector<float> & hidden_rows);

bool common_speculative_commit_accepted_hidden_rows(
    common_speculative * spec,
    common_speculative_type spec_type_used,
    llama_seq_id seq_id,
    llama_pos pos_base,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    const std::vector<float> & hidden_rows);

bool common_speculative_commit_accepted_output(
    common_speculative * spec,
    llama_context * ctx,
    common_speculative_type spec_type_used,
    llama_seq_id seq_id,
    llama_pos pos_base,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    const std::vector<int32_t> & output_indices);

bool common_speculative_has_sequence_hidden(const common_speculative * spec, llama_seq_id seq_id);

void common_speculative_clear_sequence_hidden(common_speculative * spec, llama_seq_id seq_id);

llama_context * common_speculative_get_companion_ctx(common_speculative * spec);

int32_t common_speculative_on_target_seq_batch(
    common_speculative * spec,
    llama_context * ctx,
    const llama_batch & batch,
    llama_seq_id seq_id,
    bool is_prompt_warmup);

int32_t common_speculative_on_target_batch(
    common_speculative * spec,
    const llama_batch & batch,
    const common_speculative_feature_view & features,
    bool is_prompt_warmup);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec, double slot_tps = 0.0, int n_decoded = 0, int n_past = 0, common_params_speculative * active_params = nullptr);

common_speculative_type common_speculative_current_type(const common_speculative * spec);

// Context shift for MTP to match how server handle main model
void common_speculative_context_shift(
        common_speculative * spec,
        llama_seq_id         seq_id,
        llama_pos            kv_keep,
        llama_pos            kv_discard,
        llama_pos            kv_past);
