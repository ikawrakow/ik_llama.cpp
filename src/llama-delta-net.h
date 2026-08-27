#pragma once

#include "llama-build-context.h"

#include <utility>

struct delta_net {
    delta_net(llama_context & lctx, const llama_batch & batch);
    ~delta_net();

    // Used for speculative decoding to enable per-step state checkpoint restoration.
    bool save_per_step_states = false;

    static std::pair<ggml_tensor *, ggml_tensor *> build_fused_delta_net(ggml_context * ctx0,
                      ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                      ggml_tensor * g, ggml_tensor * beta, ggml_tensor * state,
                      int il, const llm_build_cb & cb, int repeat_type,
                      ggml_tensor * per_step_ckpt = nullptr);

    ggml_tensor * build_layer_attn_linear_core(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * inp_s_seq_qnext, ggml_tensor * inp_out_ids,
            uint32_t state_seq_id_local, bool reset_state_local, int il, const llm_build_cb & cb,
            bool external_residual = false, ggml_unary_op gate_op = GGML_UNARY_OP_SILU) const;

    // external_residual: the caller has already normalised the input and owns the residual
    // add, as a hyper-connection stack does. The layer then returns just its own output.
    ggml_tensor * build_layer_attn_linear(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * inp_out_ids, int il, const llm_build_cb & cb,
            bool external_residual = false, ggml_unary_op gate_op = GGML_UNARY_OP_SILU) const;

    ggml_tensor * build_layer_attn_kda(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * inp_out_ids, int il, const llm_build_cb & cb) const;

    // which recurrent state slot a token owns. Other per-sequence state in the same row shares
    // the slot, so callers read it here rather than resolving the batch again
    bool     batch_shares_one_seq()  const { return all_same_seq; }
    uint32_t state_slot(int32_t i)   const { return (uint32_t) token_seq_ids[i]; }

private:

    llama_context     & lctx;
    const llama_batch & batch;
    std::vector<llama_seq_id> token_seq_ids;
    bool all_same_seq;
    bool has_unique_seq_ids;

    static std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(llama_context & lctx, ggml_context * ctx0,
            ggml_tensor * wqkv, ggml_tensor * wqkv_gate, ggml_tensor * input, int il, const llm_build_cb & cb,
            ggml_cgraph * gf);

    static std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(llama_context & lctx, ggml_context * ctx0, ggml_tensor * ssm_in,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, ggml_tensor * input, int il,
            const llm_build_cb & cb);

    static std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(llama_context & lctx, ggml_context * ctx0,
            ggml_tensor * wqkv, ggml_tensor * wqkv_gate, ggml_tensor * ssm_in,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, ggml_tensor * input,
            int il, const llm_build_cb & cb, ggml_cgraph * gf);

    static std::pair<ggml_tensor *, ggml_tensor *> build_beta_gate(llama_context & lctx, ggml_context * ctx0,
            ggml_tensor * ssm_beta_alpha, ggml_tensor * ssm_beta, ggml_tensor * ssm_alpha,
            ggml_tensor * ssm_dt, ggml_tensor * ssm_a, int64_t num_k_heads, int64_t num_v_heads, int64_t n_seqs,
            ggml_tensor * cur, int il, const llm_build_cb & cb, ggml_cgraph * gf);

    static ggml_tensor * build_qkv(ggml_context * ctx0, ggml_tensor * state_storage, ggml_tensor * ssm_conv1d,
            ggml_tensor * qkv_mixed, ggml_tensor * inp_s_seq_qnext, ggml_tensor * beta, ggml_tensor * gate,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, int64_t ssm_d_conv,
            int64_t state_seq_id_local, uint32_t qnext_state_slots, bool reset_state_local,
            float eps_norm, int repeat_type, int il, const llm_build_cb & cb, ggml_cgraph * gf,
            ggml_tensor * per_step_ssm = nullptr, ggml_tensor * per_step_conv = nullptr);

    // gate_op selects the output gate: SiLU for the Qwen3-Next family, SIGMOID for qwen4exp
    static ggml_tensor * build_gated_output(llama_context & lctx, ggml_context * ctx0, ggml_tensor * ssm_norm, ggml_tensor * ssm_out,
            ggml_tensor * output, ggml_tensor * z, int64_t head_v_dim, int64_t num_v_heads, int64_t n_tok, int il, const llm_build_cb & cb,
            ggml_unary_op gate_op = GGML_UNARY_OP_SILU);

    ggml_tensor * build_layer_attn_kda_core(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * inp_s_seq_qnext, ggml_tensor * inp_out_ids,
            uint32_t state_seq_id_local, bool reset_state_local, int il, const llm_build_cb & cb) const;
};
