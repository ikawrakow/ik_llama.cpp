#pragma once

#include "llama.h"

#include <vector>

struct llama_speculative;

struct llama_speculative_params {
    int n_draft = 16;  // max drafted tokens
    int n_reuse = 256;

    float p_min = 0.75f; // min probability required to accept a token in the draft
};

struct llama_speculative * llama_speculative_init(struct llama_context * ctx_dft);

void llama_speculative_free(struct llama_speculative * spec);

bool llama_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft);

// sample up to n_draft tokens and add them to the batch using the draft model
std::vector<llama_token> llama_speculative_gen_draft(
                struct llama_speculative * spec,
         struct llama_speculative_params   params,
          const std::vector<llama_token> & prompt,
                             llama_token   id_last);
