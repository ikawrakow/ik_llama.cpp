#pragma once

#include "llama.h"

#include <vector>

struct llama_speculative;

struct llama_speculative_params {
    int n_draft = 16;  // max drafted tokens
    int n_reuse = 256;

    float p_min = 0.75f; // min probability required to accept a token in the draft
};

struct mtp_kv_update_data {
    llama_token id;
    int32_t n_past;
    int32_t tok_idx;
};

struct llama_speculative * llama_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft
);

void llama_speculative_free(struct llama_speculative * spec);

void llama_speculative_add_replacement_tgt_dft(
        struct llama_speculative * spec,
        const char *source, const char *dest);

bool llama_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft);

// sample up to n_draft tokens and add them to the batch using the draft model
std::vector<llama_token> llama_speculative_gen_draft(
                struct llama_speculative * spec,
         struct llama_speculative_params   params,
          const std::vector<llama_token> & prompt,
                             llama_token   id_last);

// Generates speculative draft tokens using the Multi-Token Prediction (MTP) architecture.
std::vector<llama_token> mtp_speculative_gen_draft(
    struct llama_sampling_context * smpl,
    struct llama_context * ctx,
    struct llama_speculative_params params,
    llama_token id_last,
    int32_t n_past,
    llama_seq_id seq_id);

void mtp_update_kv_cache(struct llama_context * ctx, const llama_batch& batch, bool is_prompt_warmup);

void mtp_accept_tokens(
    struct llama_context * ctx,
    const std::vector<llama_token> & ids,
    int32_t n_past_base,
    llama_seq_id seq_id
);