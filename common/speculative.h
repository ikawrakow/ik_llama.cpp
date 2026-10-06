#pragma once

#include "llama.h"
#include "llama-spec-features.h"
#include "common.h"
#include "spec-tuner.h"

struct common_speculative;

bool common_speculative_needs_checkpoint(const llama_model * model);

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

struct common_speculative_token_dist {
    llama_tokens ids;
    std::vector<float> probs;
};

struct common_speculative_checkpoint {
    bool valid = false;
    int mode = LLAMA_SPEC_CKPT_NONE;
    llama_pos n_past = 0;
    llama_token sampled = LLAMA_TOKEN_NULL;
    common_sampler * sampler = nullptr;

    void clear();
};

// Multi-branch speculative draft tree, verified in one target pass.
// nodes[0] = root (last accepted token); `path` = branch id (one llama_seq_id per branch).
struct common_speculative_tree_node {
    llama_token tok    = LLAMA_TOKEN_NULL;
    int32_t     parent = -1;   // index into nodes, -1 for the root
    int32_t     path   = -1;   // branch id (0..n_paths-1), -1 for the root
    int32_t     depth  = 0;    // 0 = root
    float       score  = 0.0f; // confidence (e.g. ngram map count)
};

struct common_speculative_tree {
    std::vector<common_speculative_tree_node> nodes; // nodes[0] = root
    int n_paths = 0;                                 // number of branches (paths)

    bool empty() const { return nodes.size() <= 1; }
};

// each branch gets one llama_seq_id; the caller's batch must reserve these extra ids per token
constexpr int COMMON_SPECULATIVE_TREE_MAX_PATHS = 4;

struct common_speculative_tree_layout {
    llama_seq_id     tree_seq_base = 0;
    int              n_paths       = 0;
    llama_pos        root_pos      = 0;
    std::vector<int> node_batch_index;
};

// add the tree to the batch, delta-tagging the shared prefix with the branch seq ids
common_speculative_tree_layout common_speculative_tree_add(
        common_speculative * spec,
        llama_context * ctx,
        llama_batch & batch,
        llama_pos n_past,
        llama_seq_id seq_id,
        const common_speculative_tree & tree,
        int root_batch_index = -1);

// Elect the most likely root->leaf path using the target's greedy token at every node.
void common_speculative_tree_elect(
        llama_context * ctx,
        const common_speculative_tree & tree,
        const common_speculative_tree_layout & layout,
        llama_tokens & winner,
        std::vector<int> & winner_indices,
        llama_seq_id & winner_path);

// drop branch cells at positions >= n_past, keep the prefix membership
void common_speculative_tree_drop_branches(
        llama_context * ctx,
        const common_speculative_tree_layout & layout,
        llama_pos n_past);

void common_speculative_tree_mark_tagged(common_speculative * spec, llama_pos pos);

struct common_speculative_draft_result {
    llama_tokens tokens;
    std::vector<common_speculative_token_dist> proposal_dists; // Sparse proposal distributions populated by stochastic DFlash2
    common_speculative_tree tree;                              // non-empty => verify as a tree
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;
    bool target_only = false;
};

// Result of verifying a draft (linear chain or tree) against the target model.
struct common_speculative_verify_result {
    llama_tokens             ids;                     // accepted tokens
    std::vector<int32_t>     accepted_output_indices; // indices into the verify batch
    int                      n_elected = 0;           // length of the elected draft (for commit)
    bool                     used_tree = false;
    bool                     ok = true;
    std::string              error;
};

// verify a draft (linear, or tree when draft_result.tree is set) in one target decode
common_speculative_verify_result common_speculative_verify(
        common_speculative * spec,
        llama_context * ctx,
        common_sampler * sampler,
        const common_speculative_draft_result & draft_result,
        llama_token sampled_before,
        llama_pos n_past,
        llama_seq_id seq_id);

// learn a token->successors transition from the target logits at batch index `logits_idx`
void common_speculative_observe(common_speculative * spec, llama_context * ctx, int logits_idx, llama_token tok);

struct common_speculative_metrics_stage_snapshot {
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;

    uint64_t n_call_begin = 0;
    uint64_t n_call_draft = 0;
    uint64_t n_call_accept = 0;

    uint64_t n_gen_drafts = 0;
    uint64_t n_acc_drafts = 0;
    uint64_t n_gen_tokens = 0;
    uint64_t n_acc_tokens = 0;

    // Position zero represents speculative position 1.
    std::vector<uint64_t> drafted_by_position;
    std::vector<uint64_t> accepted_by_position;

    int64_t t_begin_us = 0;
    int64_t t_draft_us = 0;
    int64_t t_accept_us = 0;
};

struct common_speculative_metrics_snapshot {
    std::vector<common_speculative_metrics_stage_snapshot> stages;
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

bool common_speculative_prepare_startup(
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
                            llama_seq_id  draft_seq_id = 0,
                            const common_params_sampling * sampling = nullptr);

int common_speculative_get_configured_n_max(const common_speculative * spec);

// apply -td/--threads-draft to a draft params copy
void common_speculative_apply_draft_threads(const common_params_speculative & spec, gpt_params & out);

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

bool common_speculative_checkpoint_restore(
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

bool common_speculative_commit(
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

common_speculative_metrics_snapshot common_speculative_get_metrics_snapshot(const common_speculative * spec);

// Context shift for MTP to match how server handle main model
void common_speculative_context_shift(
        common_speculative * spec,
        llama_seq_id         seq_id,
        llama_pos            kv_keep,
        llama_pos            kv_discard,
        llama_pos            kv_past);

struct common_speculative_round_result {
    bool attempted = false;
    bool sampled_before_ready = false;
    bool sampled_before_from_carry = false;
    bool used_speculative = false;
    bool failed = false;
    std::string error;
    llama_token sampled_before = LLAMA_TOKEN_NULL;
    llama_tokens ids;
};

common_speculative_round_result common_speculative_run_round(
    common_speculative * spec,
    llama_model * model,
    llama_context * ctx,
    common_sampler * sampler,
    llama_context * ctx_guidance,
    common_params_speculative params,
    const common_params_sampling & sparams,
    llama_seq_id seq_id,
    llama_pos n_past,
    int n_predict_budget,
    bool have_carry,
    const llama_tokens & draft_history,
    llama_token carry_token);
