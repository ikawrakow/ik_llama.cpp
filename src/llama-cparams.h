#pragma once

#include "llama-impl.h"

#include <cstdint>

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_threads;       // number of threads to use for generation
    uint32_t n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    float defrag_thold;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    int  mla_attn;
    int  attn_max_batch;
    bool fused_moe_up_gate;
    bool fused_up_gate;
    int  min_experts;
    float thresh_experts;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
