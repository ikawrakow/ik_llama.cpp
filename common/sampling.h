#pragma once

#include "llama.h"
#include "llama-grammar.h"
#include <set>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// sampler types
enum class llama_sampler_type : char {
    DRY         = 'd',
    TOP_K       = 'k',
    TOP_P       = 'p',
    MIN_P       = 'm',
    TFS_Z       = 'f',
    XTC         = 'x',
    TOP_N_SIGMA = 'n',
    TYPICAL_P   = 'y',
    TEMPERATURE = 't',
    ADAPTIVE_P  = 'w',
    DIST     = 's',
};

enum common_grammar_trigger_type {
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};

struct common_grammar_trigger {
    common_grammar_trigger_type type;
    std::string value;
    llama_token token = LLAMA_TOKEN_NULL;

    // T can only be nlohmann::ordered_json
    template <class T> T to_json() const;
    template <class T> static common_grammar_trigger from_json(const T& in);
};



// sampling parameters
typedef struct common_params_sampling {
    int32_t     n_prev                = 64;                 // number of previous tokens to remember
    int32_t     n_probs               = 0;                  // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t     min_keep              = 0;                  // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t     top_k                 = 40;                 // <= 0 to use vocab size
    float       top_p                 = 0.95f;              // 1.0 = disabled
    float       min_p                 = 0.05f;              // 0.0 = disabled
    float       tfs_z                 = 1.00f;              // 1.0 = disabled
    float       typical_p             = 1.00f;              // 1.0 = disabled
    float       temp                  = 0.80f;              // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float       dynatemp_range        = 0.00f;              // 0.0 = disabled
    float       dynatemp_exponent     = 1.00f;              // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t   penalty_last_n        = 64;                 // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float       penalty_repeat        = 1.00f;              // 1.0 = disabled
    float       penalty_freq          = 0.00f;              // 0.0 = disabled
    float       penalty_present       = 0.00f;              // 0.0 = disabled
    float       dry_multiplier = 0.0f;  // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
    float       dry_base = 1.75f; // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
    int32_t   dry_allowed_length = 2;     // tokens extending repetitions beyond this receive penalty
    int32_t   dry_penalty_last_n = -1;    // how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
    int32_t   total_context_size = 16840;
    int32_t   mirostat              = 0;                  // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float       mirostat_tau          = 5.00f;              // target entropy
    float       mirostat_eta          = 0.10f;              // learning rate
    float       xtc_probability       = 0.0f;               // xtc probability
    float       xtc_threshold         = 1.0f;               // xtc threshold, disabled if > 0.5
    float       top_n_sigma           = 0.0f;               // top-n-sigma
    float       adaptive_target       = -1.0f;              // select tokens near this probability (valid range 0.0 to 1.0; <0 = disabled)
    float       adaptive_decay        = 0.90f;              // decay rate for target adaptation over time. lower values -> faster but less stable adaptation. (valid range 0.0 to 1.0; ≤0 = no adaptation)
    bool        adaptive_updt_w_cur   = false;              // update state with current probability
    bool        penalize_nl           = false;              // consider newlines as a repeatable token
    uint32_t    seed                  = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampling_context

    std::vector<std::string> dry_sequence_breakers = { "\n", ":", "\"", "*" };     // default sequence breakers for DRY

    std::vector<llama_sampler_type> samplers_sequence = {
        llama_sampler_type::DRY,
        llama_sampler_type::TOP_K,
        llama_sampler_type::TFS_Z,
        llama_sampler_type::TYPICAL_P,
        llama_sampler_type::TOP_P,
        llama_sampler_type::MIN_P,
        llama_sampler_type::XTC,
        llama_sampler_type::TOP_N_SIGMA,
        llama_sampler_type::TEMPERATURE,
        llama_sampler_type::ADAPTIVE_P,
        llama_sampler_type::DIST,
    };


    std::string grammar;  // optional BNF-like grammar to constrain sampling
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers; // optional triggers (for lazy grammars)
    std::set<llama_token>               preserved_tokens;
    // Classifier-Free Guidance
    // https://arxiv.org/abs/2306.17806
    std::string cfg_negative_prompt; // string to help guidance
    float       cfg_scale     = 1.f; // how strong is guidance

    std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens

    std::vector<llama_token> penalty_prompt_tokens;
    bool                     use_penalty_prompt_tokens = false;
} llama_sampling_params;

// general sampler context
// TODO: move to llama.h
struct common_sampler {
    // parameters that will be used for sampling
    common_params_sampling params;

    // mirostat sampler state
    float mirostat_mu;

    std::string grammar_str;
    std::string grammar_root;

    llama_grammar * grammar;

    // TODO: replace with ring-buffer
    std::vector<llama_token>      prev;
    std::vector<llama_token_data> cur;
    llama_sampler_dry* smpl;

    llama_sampler_adaptive_p * adapt_p_ctx;    // adaptive p sampler

    size_t n_valid; // Number of correct top tokens with correct probabilities.

    llama_token_data_array cur_p; // current candidates

    std::mt19937 rng;
};



// Create a new sampling context instance.
struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * ctx);

// Reset the sampler context
// - clear prev tokens
// - reset grammar
void common_sampler_reset(common_sampler * ctx);

// Set the sampler seed
void llama_sampling_set_rng_seed(struct common_sampler * ctx, uint32_t seed);

// Copy the sampler context
void common_sampler_clone(common_sampler * src, common_sampler * dst);

// Get the last sampled token
llama_token llama_sampling_last(common_sampler * ctx);

// Get a string representation of the last sampled tokens
std::string llama_sampling_prev_str(common_sampler * ctx_sampling, llama_context * ctx_main, int n);

// Print sampling parameters into a string
std::string llama_sampling_print(const common_params_sampling & params);

// Print sampling order into a string
std::string llama_sampling_order_print(const common_params_sampling & params);

std::string llama_sampling_type_to_str(llama_sampler_type sampler_type);

std::vector<llama_sampler_type> llama_sampling_types_from_names(const std::vector<std::string> & names, bool allow_alt_names);
std::vector<llama_sampler_type> llama_sampling_types_from_chars(const std::string & names_string);

// this is a common sampling function used across the examples for convenience
// it can serve as a starting point for implementing your own sampling function
// Note: When using multiple sequences, it is the caller's responsibility to call
//       common_sampler_reset when a sequence ends
//
// required:
//  - ctx_main:     context to use for sampling
//  - ctx_sampling: sampling-specific context
//
// optional:
//  - ctx_cfg:      context to use for classifier-free guidance
//  - idx:          sample from llama_get_logits_ith(ctx, idx)
//
// returns:
//  - token:      sampled token
//  - candidates: vector of candidate tokens
//
llama_token common_sampler_sample_legacy(
        struct common_sampler * ctx_sampling,
        struct llama_context * ctx_main,
        struct llama_context * ctx_cfg,
        int idx = -1);

llama_token common_sampler_sample(
    struct common_sampler * ctx_sampling,
    struct llama_context * ctx_main,
    int idx = -1,
    bool grammar_first = false);

// Prepares and adjusts the set of token candidates for sampling based on penalties, biases, and sampling parameters.
llama_token_data_array llama_sampling_prepare(
        struct common_sampler * ctx_sampling,
        struct llama_context * ctx_main,
        struct llama_context * ctx_cfg,
        int idx = 0,
        bool apply_grammar = true,
        std::vector<float> * original_logits = nullptr);

void common_sampler_accept(
        struct common_sampler * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token id,
        bool apply_grammar);

// returns at least 1 token, up to draft.size()
// access the internal list of current candidate tokens
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * ctx_sampling, bool do_sort = false);

std::vector<llama_token> llama_sampling_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<llama_token> & draft);

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const std::vector<llama_token> & draft, bool grammar_first = false);

llama_grammar* llama_sampler_init_llg(const llama_vocab* vocab,
    const char* grammar_kind, const char* grammar_data);
