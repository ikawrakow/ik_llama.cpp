#pragma once

#include "llama.h"
#include "llama-spec-features.h"
#include "common.h"
#include "spec-tuner.h"

struct common_speculative;

enum common_speculative_init_status {
    COMMON_SPECULATIVE_INIT_SKIPPED,
    COMMON_SPECULATIVE_INIT_READY,
    COMMON_SPECULATIVE_INIT_ERR_RECURRENT,
    COMMON_SPECULATIVE_INIT_ERR_MTP,
    COMMON_SPECULATIVE_INIT_ERR_GENERIC,
};

using common_speculative_feature_kind = llama_spec_feature_kind;
using common_speculative_feature_row_view = llama_spec_feature_row_view;
using common_speculative_feature_view = llama_spec_feature_view;

static constexpr common_speculative_feature_kind COMMON_SPECULATIVE_FEATURE_NONE = LLAMA_SPEC_FEATURE_NONE;
static constexpr common_speculative_feature_kind COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE = LLAMA_SPEC_FEATURE_HIDDEN_STATE;

struct common_speculative_checkpoint {
    bool valid = false;
    bool per_step_enabled = false;
    llama_pos n_past = 0;
    llama_token sampled = LLAMA_TOKEN_NULL;
    common_sampler * sampler = nullptr;

    void clear();
};

struct common_speculative_draft_result {
    llama_tokens tokens;
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;
    bool target_only = false;
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

common_speculative_init_status common_speculative_try_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        common_speculative      ** out_spec);

void common_speculative_prepare_startup(
        gpt_params & params_base,
        bool         allow_parallel_mtp = true);

bool common_speculative_finalize_startup(
        gpt_params        & params_base,
        const llama_model * model);

bool common_speculative_load_draft_model(
        common_params_speculative & params,
        const gpt_params         & params_base);

bool common_speculative_prepare_mtp_runtime(
        common_params_speculative & params,
        const gpt_params         & params_base,
        const llama_model        * model,
        bool                       has_external_mtp);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// apply per-request runtime parameters before prompt warmup can touch companion state
void common_speculative_prepare_request(common_speculative * spec, common_params_speculative & params);

// true when the active request drafts with more MTP heads than the cached prefix was
// warmed with; the caller must then reprocess the prompt from position 0
bool common_speculative_mtp_requires_fresh_warmup(const common_speculative * spec);

// sample up to n_draft tokens and add them to the batch using the draft model
// draft_base_pos/draft_seq_id override the MTP position for id_last
llama_tokens common_speculative_draft(
                     common_speculative * spec,
                     common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                            llama_pos     draft_base_pos = -1,
                            llama_seq_id  draft_seq_id = 0);

common_speculative_draft_result common_speculative_draft_ex(
                     common_speculative * spec,
                     llama_context * ctx,
                     common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                            llama_pos     draft_base_pos = -1,
                            llama_seq_id  draft_seq_id = 0);

int common_speculative_get_configured_n_max(const common_speculative * spec);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

bool common_speculative_before_draft(
    common_speculative * spec,
    llama_model * model,
    llama_context * ctx,
    common_sampler * sampler_src,
    const common_params_sampling & sparams,
    llama_seq_id seq_id,
    llama_pos n_past,
    llama_token sampled,
    int max_tokens,
    int ckpt_mode);

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

const common_speculative_checkpoint * common_speculative_get_checkpoint(const common_speculative * spec);

void common_speculative_checkpoint_discard(
    common_speculative_checkpoint & ckpt,
    llama_context * ctx);

void common_speculative_checkpoint_restore(
    common_speculative_checkpoint & ckpt,
    common_speculative * spec,
    llama_context * ctx,
    common_sampler * sampler_dst,
    llama_seq_id seq_id,
    common_speculative_type spec_type_used,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    int n_draft,
    const std::vector<float> & mtp_hidden_state_pre,
    int32_t mtp_n_past_base);

void common_speculative_commit(
    common_speculative * spec,
    llama_context * ctx,
    common_sampler * sampler_dst,
    llama_seq_id seq_id,
    llama_token sampled_before,
    const std::vector<llama_token> & ids,
    int n_draft,
    llama_pos pos_base,
    const std::vector<int32_t> & accepted_output_indices);

bool common_speculative_has_sequence_hidden(const common_speculative * spec, llama_seq_id seq_id);

void common_speculative_clear_sequence_hidden(common_speculative * spec, llama_seq_id seq_id);

void common_speculative_clear_sequence(
    common_speculative * spec,
    llama_seq_id seq_id,
    bool clear_companion_ctx = false);

bool common_speculative_trim_sequence(
    common_speculative * spec,
    llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos pos_begin);

void common_speculative_clear_sequence_kv(
    common_speculative * spec,
    llama_context * ctx,
    llama_seq_id seq_id);

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
