#pragma once

#include "llama-build-context.h"

#include <utility>

struct delta_net {
    delta_net(llama_context & lctx, const llama_batch & batch);
    ~delta_net();

    static std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(ggml_context * ctx0,
                      ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                      ggml_tensor * g, ggml_tensor * beta, ggml_tensor * state,
                      ggml_tensor * causal_mask, ggml_tensor * identity,
                      ggml_tensor * diag_mask, int il, const llm_build_cb & cb);

    static std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(ggml_context * ctx0,
                      ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                      ggml_tensor * g, ggml_tensor * beta, ggml_tensor * state,
                      int il, const llm_build_cb & cb);

    std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(ggml_context * ctx0, ggml_tensor * input, int il, const llm_build_cb & cb) const;

    ggml_tensor * build_layer_attn_linear_core(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * causal_mask, ggml_tensor * identity,
            ggml_tensor * diag_mask, ggml_tensor * inp_s_seq_qnext,
            uint32_t state_seq_id_local, bool reset_state_local, int il, const llm_build_cb & cb) const;

    ggml_tensor * build_layer_attn_linear(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * causal_mask, ggml_tensor * identity,
            ggml_tensor * diag_mask, int il, const llm_build_cb & cb) const;

private:

    llama_context     & lctx;
    const llama_batch & batch;
    std::vector<llama_seq_id> token_seq_ids;
    bool all_same_seq;
    bool has_unique_seq_ids;

};
