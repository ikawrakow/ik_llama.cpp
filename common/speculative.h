#pragma once

#include "llama.h"
#include "common.h"
#include "spec-tuner.h"

struct common_speculative;

enum common_speculative_companion_kind {
    COMMON_SPECULATIVE_COMPANION_NONE,
    COMMON_SPECULATIVE_COMPANION_MODEL,
};

enum common_speculative_feature_kind {
    COMMON_SPECULATIVE_FEATURE_NONE,
    COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE,
};

struct common_speculative_traits {
    std::vector<common_speculative_type> configured_types;
    common_speculative_type active_type = COMMON_SPECULATIVE_TYPE_NONE;
    common_speculative_companion_kind companion_kind = COMMON_SPECULATIVE_COMPANION_NONE;
};

struct common_speculative_feature_request {
    common_speculative_feature_kind kind = COMMON_SPECULATIVE_FEATURE_NONE;
};

struct common_speculative_feature_row_view {
    llama_seq_id seq_id = 0;
    llama_pos pos = -1;
    const float * data = nullptr;
};

struct common_speculative_feature_view {
    common_speculative_feature_kind kind = COMMON_SPECULATIVE_FEATURE_NONE;
    int32_t width = 0;
    std::vector<common_speculative_feature_row_view> rows;
};

struct common_speculative_span {
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;
    uint32_t stage_index = 0;
    uint32_t token_offset = 0;
    uint32_t n_tokens = 0;
};

struct common_speculative_draft_result {
    llama_tokens tokens;
    std::vector<common_speculative_span> spans;
    common_speculative_type active_type = COMMON_SPECULATIVE_TYPE_NONE;
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

// sample up to n_draft tokens and add them to the batch using the draft model
// draft_base_pos/draft_seq_id override the MTP position for id_last
common_speculative_draft_result common_speculative_draft_ex(
            common_speculative * spec,
            common_params_speculative & params,
            const llama_tokens & prompt,
                llama_token   id_last,
                llama_pos     draft_base_pos = -1,
                llama_seq_id  draft_seq_id = 0);

llama_tokens common_speculative_draft(
                     common_speculative * spec,
                     common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                            llama_pos     draft_base_pos = -1,
                            llama_seq_id  draft_seq_id = 0);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

bool common_speculative_has_type(const common_speculative * spec, common_speculative_type type);
common_speculative_traits common_speculative_get_traits(const common_speculative * spec);
std::vector<common_speculative_feature_request> common_speculative_get_feature_requests(const common_speculative * spec);
bool common_speculative_feature_view_copy_seq_rows(
    const common_speculative_feature_view & view,
    llama_seq_id seq_id,
    std::vector<float> * first_row,
    std::vector<float> * last_row);
bool common_speculative_collect_target_batch_features(
    const common_speculative * spec,
    llama_context * ctx,
    const llama_batch & batch,
    common_speculative_feature_view & features);
bool common_speculative_collect_target_seq_batch_features(
    const common_speculative * spec,
    llama_context * ctx,
    const llama_batch & batch,
    llama_seq_id seq_id,
    common_speculative_feature_view & features);
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
bool common_speculative_capture_target_features(common_speculative * spec, const common_speculative_feature_view & features);
bool common_speculative_has_sequence_hidden(const common_speculative * spec, llama_seq_id seq_id);
bool common_speculative_copy_sequence_hidden(const common_speculative * spec, llama_seq_id seq_id, std::vector<float> & hidden);
void common_speculative_restore_sequence_hidden(common_speculative * spec, llama_seq_id seq_id, const std::vector<float> & hidden);
void common_speculative_clear_sequence_hidden(common_speculative * spec, llama_seq_id seq_id);
llama_context * common_speculative_get_companion_ctx(common_speculative * spec);
int32_t common_speculative_on_target_batch(
    common_speculative * spec,
    const llama_batch & batch,
    const common_speculative_feature_view & features,
    bool is_prompt_warmup,
    const float * seed_hidden = nullptr);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec, double slot_tps = 0.0, int n_decoded = 0, int n_past = 0, common_params_speculative * active_params = nullptr);

// get the MTP context from the speculative object (nullptr if not MTP type)
llama_context * common_speculative_get_mtp_ctx(common_speculative * spec);
common_speculative_type common_speculative_current_type(const common_speculative * spec);

// Context shift for MTP to match how server handle main model
void common_speculative_context_shift(
        common_speculative * spec,
        llama_seq_id         seq_id,
        llama_pos            kv_keep,
        llama_pos            kv_discard,
        llama_pos            kv_past);
