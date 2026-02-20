#pragma once

#include "llama-impl.h"
#include <unordered_map>
struct llama_sampling {
    llama_sampling(int32_t n_vocab) : n_vocab(n_vocab) {}

    std::mt19937 rng;

    int32_t n_vocab = 0;

    mutable int64_t t_sample_us = 0;
    mutable int32_t n_sample = 0;

    void reset_timings() const {
        t_sample_us = 0;
        n_sample = 0;
    }
};

//
// internal API
//

void llama_set_rng_seed_impl(struct llama_sampling * smpl, uint32_t seed);

void llama_sample_softmax_impl  (struct llama_sampling * smpl, llama_token_data_array * candidates, bool normalize = true);
void llama_sample_top_k_impl    (struct llama_sampling * smpl, llama_token_data_array * candidates, int32_t k, size_t min_keep);
void llama_sample_top_p_impl    (struct llama_sampling * smpl, llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sample_min_p_impl    (struct llama_sampling * smpl, llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sample_tail_free_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float z, size_t min_keep);
void llama_sample_typical_impl  (struct llama_sampling * smpl, llama_token_data_array * candidates, float p, size_t min_keep);
void llama_sample_entropy_impl  (struct llama_sampling * smpl, llama_token_data_array * candidates, float min_temp, float max_temp, float exponent_val);
void llama_sample_temp_impl     (struct llama_sampling * smpl, llama_token_data_array * candidates, float temp);
void llama_sample_xtc_impl      (struct llama_sampling * smpl, llama_token_data_array * candidates, float probability, float threshold, size_t min_keep);
void llama_sample_top_n_sigma_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float top_n_sigma);

struct llama_sampler_dry {
    int32_t total_context_size;

    const float   dry_multiplier;
    const float   dry_base;
    const int32_t dry_allowed_length;
    const int32_t dry_penalty_last_n;

    std::unordered_multimap<llama_token, std::vector<llama_token>> dry_processed_breakers;
    std::vector<int> dry_repeat_count;
    std::unordered_map<llama_token, int> dry_max_token_repeat;
    ring_buffer<llama_token> last_tokens;
};

struct llama_sampler_dry * llama_sampler_init_dry_impl(
        const struct llama_vocab &  vocab,
                         int32_t    context_size,
                           float    dry_multiplier,
                           float    dry_base,
                         int32_t    dry_allowed_length,
                         int32_t    dry_penalty_last_n,
                      const char ** seq_breakers,
                          size_t    num_breakers);

void llama_sampler_dry_apply(struct llama_sampler_dry* smpl, llama_token_data_array* cur_p);


// maintains an exponential moving average of the *ORIGINAL* probabilities of selected tokens
// used to compute an adapted target at each sampling step.
// see llama.h for a full description of the sampler
struct llama_sampler_adaptive_p {
    const float target;     // target probability (0.0 - 1.0; negative = disabled)
    const float decay;      // EMA decay; history ≈ 1/(1-decay) tokens (0.0 - 0.99)
    const bool updt_w_cur;  // false=original, true=current
    std::mt19937 rng;       // RNG
    float weighted_sum;     // sum(p_n * decay^N)
    float total_weight;     // sum(decay^i), converges to 1/(1-decay)

    // first referenced in prep
    std::vector<float> orig_prob;   // for storing the original proibabilities
    float cum_orig_prob;    // for normalizing orig_prob in sample_token

    // first referenced in sample
    float cum_cur_p;        // cumulative sum of current probabilities
    float max_xform_logit;  // maximum logit found during transform

    // first referenced in sample_token
    std::vector<float> cum_probs;   // cumulative probability distribution

    // recorded states for rewinding
    float recd_weighted_sum;
    float recd_total_weight;
};

struct llama_sampler_adaptive_p * llama_init_adaptive_p_impl(int n_vocab,
       const float target,
       const float decay,
        const bool updt_w_cur,
    const uint32_t seed);

void llama_prep_adaptive_p_impl(
              struct llama_sampling * smpl,
             llama_token_data_array * candidates,
    struct llama_sampler_adaptive_p * adapt_p_ctx);

void llama_sample_adaptive_p_impl(
              struct llama_sampling * smpl,
             llama_token_data_array * candidates,
    struct llama_sampler_adaptive_p * adapt_p_ctx);

void llama_review_adaptive_p_impl(llama_sampler_adaptive_p * adapt_p_ctx, const bool record, const bool rewind);


void llama_sample_repetition_penalties_impl(
        struct llama_sampling * smpl,
       llama_token_data_array * candidates,
            const llama_token * last_tokens,
                       size_t   penalty_last_n,
                        float   penalty_repeat,
                        float   penalty_freq,
                        float   penalty_present);

void llama_sample_apply_guidance_impl(
        struct llama_sampling * smpl,
                        float * logits,
                        float * logits_guidance,
                        float   scale);

llama_token llama_sample_token_mirostat_impl   (struct llama_sampling * smpl, llama_token_data_array * candidates, float tau, float eta, int32_t m, float * mu);
llama_token llama_sample_token_mirostat_v2_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float tau, float eta, float * mu);
llama_token llama_sample_token_greedy_impl     (struct llama_sampling * smpl, llama_token_data_array * candidates);
llama_token llama_sample_token_with_rng_impl   (struct llama_sampling * smpl, llama_token_data_array * candidates, std::mt19937 & rng);
llama_token llama_sample_token_impl            (struct llama_sampling * smpl, llama_token_data_array * candidates);
llama_token llama_sample_token_adaptive_p_impl (struct llama_sampling * smpl, llama_token_data_array * candidates, struct llama_sampler_adaptive_p * adapt_p_ctx);


