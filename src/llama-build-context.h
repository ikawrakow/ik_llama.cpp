#pragma once

#include "llama-impl.h"
#include "llama-hparams.h"

#include <cstdint>
#include <functional>
#include <tuple>

struct llama_model;
struct llama_context;
struct llama_cparams;
struct llama_batch;
struct llama_kv_cache;

struct ggml_cgraph;
struct ggml_tensor;

using llm_build_cb = std::function<void(struct ggml_tensor * cur, const char * name, int nl)>;

enum llm_ffn_op_type {
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
    LLM_FFN_SWIGLU_OAI_MOE,
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
};

struct llm_build_context {
    const llama_model    & model;
          llama_context  & lctx;
    const llama_hparams  & hparams;
    const llama_cparams  & cparams;
    const llama_batch    & batch;
    const llama_kv_cache & kv_self;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int32_t n_tokens;
    const int32_t n_kv;     // size of KV cache to consider (n_kv <= kv_self.size)
    const int32_t n_outputs;
    const int32_t n_outputs_enc;
    const int32_t kv_head;  // index of where we store new KV data in the cache
    const int32_t n_ctx_orig;

    const bool flash_attn;
    const int  mla_attn;
    const int  attn_max_batch;
    const bool fused_moe_up_gate;
    const bool grouped_expert_routing;
    const bool fused_up_gate;
    const bool fused_mmad;
    const bool rope_cache;
    const bool k_cache_hadamard;
    const bool split_mode_graph_scheduling;
    const int  min_experts;
    const float thresh_experts;

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    const llm_build_cb & cb;

    std::vector<uint8_t> & buf_compute_meta;

    struct ggml_context * ctx0 = nullptr;

    // TODO: consider making the entire interface noexcept
    llm_build_context(
        llama_context  & lctx,
    const llama_batch  & batch,
    const llm_build_cb & cb,
    bool   worst_case,
    bool   warmup);

    void init();

    void free();

    ggml_cgraph * build_k_shift();

    ggml_cgraph * build_s_copy();

    ggml_cgraph * build_defrag(const std::vector<uint32_t> & ids);

    ggml_tensor * build_inp_pos();

    ggml_tensor * build_input_scale(int n_tokens);

    ggml_tensor * build_rope_factors(int il);

    ggml_tensor * build_inp_out_ids();

    ggml_tensor * build_inp_KQ_mask(bool causal = true);

    ggml_tensor * build_inp_KQ_mask_swa(bool causal = true);

    ggml_tensor * build_inp_mean();

    ggml_tensor * build_inp_cls();

    ggml_tensor * build_inp_s_copy();

    ggml_tensor * build_inp_s_mask();

    ggml_tensor * build_inp_s_seq();

    ggml_cgraph * append_pooling(struct ggml_cgraph * gf);

    ggml_tensor * llm_build_pos_bucket(bool causal);

    ggml_tensor * llm_build_pos_bias(struct ggml_tensor * pos_bucket, struct ggml_tensor * attn_rel_b);

    ggml_tensor * llm_build_inp_embd_enc();

    ggml_tensor * llm_build_inp_KQ_mask_cross();

    std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*> llm_build_mul_mat_qkv(ggml_cgraph * gf, ggml_tensor * cur,
            ggml_tensor * wq, ggml_tensor * bq,
            ggml_tensor * wk, ggml_tensor * bk,
            ggml_tensor * wv, ggml_tensor * bv,
            float attention_scale, int il, bool add_graph_split = false) const;

    std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*> llm_build_mul_mat_qkv(ggml_cgraph * gf, ggml_tensor * cur,
            ggml_tensor * wqkv, ggml_tensor * bqkv,
            ggml_tensor * wqk, ggml_tensor * bqk,
            ggml_tensor * wq, ggml_tensor * bq,
            ggml_tensor * wk, ggml_tensor * bk,
            ggml_tensor * wv, ggml_tensor * bv,
            ggml_tensor * q_norm, ggml_tensor * k_norm, float attention_scale, int il, bool add_graph_split = false) const;

    ggml_cgraph * build_llama();

    ggml_cgraph * build_mistral3();

    ggml_cgraph * build_deci();

    ggml_cgraph * build_baichuan();

    ggml_cgraph * build_xverse();

    ggml_cgraph * build_falcon();

    ggml_cgraph * build_grok();

    ggml_cgraph * build_dbrx();

    ggml_cgraph * build_starcoder();

    ggml_cgraph * build_refact();

    ggml_cgraph * build_bert();

    ggml_cgraph * build_bloom();

    ggml_cgraph * build_mpt();

    ggml_cgraph * build_stablelm();

    ggml_cgraph * build_qwen();

    ggml_cgraph * build_qwen2();

    ggml_cgraph * build_qwen2vl();

    ggml_cgraph * build_qwen2moe();

    ggml_cgraph * build_qwen3();

    ggml_cgraph * build_qwen3vl();

    ggml_cgraph * build_qwen3moe();

    ggml_cgraph * build_qwen3vlmoe();

    ggml_cgraph * build_phi2();

    ggml_cgraph * build_phi3();

    ggml_cgraph * build_plamo();

    ggml_cgraph * build_gpt2();

    ggml_cgraph * build_codeshell();

    ggml_cgraph * build_orion();

    ggml_cgraph * build_internlm2();

    ggml_cgraph * build_minicpm();

    ggml_cgraph * build_gemma();

    ggml_cgraph * build_gemma2();

    ggml_cgraph * build_gemma3();

    ggml_cgraph * build_starcoder2();

    ggml_cgraph * build_mamba();

    ggml_cgraph * build_command_r();

    ggml_cgraph * build_olmo();

    ggml_cgraph * build_openelm();

    ggml_cgraph * build_gptneox();

    ggml_cgraph * build_arctic();

    ggml_cgraph * build_deepseek2();

    ggml_cgraph * build_glm4_moe();

    ggml_cgraph * build_bitnet();

    ggml_cgraph * build_bitnet_158();

    ggml_cgraph * build_cohere2();

    ggml_cgraph * build_t5_encoder();

    ggml_cgraph * build_t5_decoder();

    ggml_cgraph * build_jais();

    ggml_cgraph * build_chatglm();

    ggml_cgraph * build_glm4();

    ggml_cgraph * build_dots1();

    ggml_cgraph * build_ernie4_5();

    ggml_cgraph * build_ernie4_5_moe();

    ggml_cgraph * build_hunyuan_moe();

    ggml_cgraph * build_openai_moe();

    ggml_cgraph * build_bailingmoe2();

    ggml_cgraph * build_minimaxm2();

    ggml_cgraph * build_smollm3();

    ggml_cgraph * build_mimo2();

    //
    static ggml_tensor * llm_build_lora_mm(llama_context & lctx, ggml_context * ctx0,
            ggml_tensor * w, ggml_tensor * cur);

    static ggml_tensor * llm_build_lora_mm_id(llama_context & lctx, ggml_context * ctx0,
          ggml_tensor * w, ggml_tensor * cur, ggml_tensor * ids);

    static ggml_tensor * llm_build_inp_embd(ggml_context * ctx, llama_context & lctx,
        const llama_hparams & hparams,
          const llama_batch & batch,
         struct ggml_tensor * tok_embd,
         const llm_build_cb & cb);

    static ggml_tensor * llm_build_norm(ggml_context * ctx, ggml_tensor * cur,
         const llama_hparams & hparams,
         ggml_tensor * mw,
         ggml_tensor * mb,
         llm_norm_type   type,
         const llm_build_cb & cb, int il, float scale_eps = 1);

    static void llm_build_kv_store(llama_context & lctx, ggml_context * ctx, const llama_hparams & hparams,
        const llama_cparams & cparams,
       const llama_kv_cache & kv,
         ggml_cgraph * graph,
         ggml_tensor * k_cur,
         ggml_tensor * v_cur,
         int32_t   n_tokens,
         int32_t   kv_head,
         const llm_build_cb & cb, int64_t il);

    static struct ggml_tensor * llm_build_kv(ggml_context * ctx, llama_context & lctx,
       const llama_kv_cache & kv,
         ggml_cgraph * graph,
         ggml_tensor * wo,
         ggml_tensor * wo_b,
         ggml_tensor * k_cur,
         ggml_tensor * v_cur,
         ggml_tensor * q_cur,
         ggml_tensor * kq_mask,
                    int32_t   n_tokens,
                    int32_t   kv_head,
                    int32_t   n_kv,
                    float     kq_scale,
         const llm_build_cb & cb, int il, ggml_tensor * sinks = nullptr, int n_swa = 0);

    static ggml_tensor * llm_build_ffn(ggml_context * ctx, llama_context & lctx, ggml_tensor * ffn_norm,
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
            llm_ffn_op_type   type_op,
          llm_ffn_gate_type   type_gate,
         const llm_build_cb & cb, int il, ggml_cgraph * graph = nullptr, bool add_input = false,
         bool is_norm = false, ggml_tensor * add_extra = nullptr);

    static ggml_tensor * llm_build_moe_ffn(ggml_context * ctx, llama_context & lctx,
         ggml_tensor * cur,
         ggml_tensor * gate_inp,   ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,    ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,  ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,  ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
                    int64_t   n_expert,
                    int64_t   n_expert_used,
            llm_ffn_op_type   type_op,
                       bool   norm_w,
                       bool   scale_w,
                      float   w_scale,
llm_expert_gating_func_type   gating_op,
         const llm_build_cb & cb, int il, ggml_cgraph * graph = nullptr, bool add_input = false,
         ggml_tensor * up_gate_exps = nullptr, ggml_tensor * up_gate_exps_b = nullptr);

    static ggml_tensor * llm_build_moe_ffn(ggml_context * ctx, llama_context & lctx,
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
                    int64_t   n_expert,
                    int64_t   n_expert_used,
            llm_ffn_op_type   type_op,
                       bool   norm_w,
                       bool   scale_w,
                      float   w_scale,
llm_expert_gating_func_type   gating_op,
         const llm_build_cb & cb, int il, ggml_cgraph * graph = nullptr, bool add_input = false,
         ggml_tensor * up_gate_exps = nullptr, ggml_tensor * up_gate_exps_b = nullptr) {
        return llm_build_moe_ffn(ctx, lctx, cur,
                gate_inp,   nullptr,
                up_exps,    nullptr,
                gate_exps,  nullptr,
                down_exps,  nullptr,
                exp_probs_b,
                n_expert, n_expert_used,
                type_op, norm_w, scale_w, w_scale,
                gating_op, cb, il, graph, add_input, up_gate_exps, up_gate_exps_b);
    }

    static ggml_tensor * llm_build_std_moe_ffn(ggml_context * ctx, llama_context & lctx,
         ggml_tensor * ffn_norm,
         ggml_tensor * input,
         ggml_tensor * gate_inp,   ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,    ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,  ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,  ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
         ggml_tensor * up_shexp,   ggml_tensor * up_b_shexp,
         ggml_tensor * gate_shexp, ggml_tensor * gate_b_shexp,
         ggml_tensor * down_shexp, ggml_tensor * down_b_shexp,
                    int64_t   n_expert,
                    int64_t   n_expert_used,
            llm_ffn_op_type   type_op,
                       bool   norm_w,
                       bool   scale_w,
                      float   w_scale,
llm_expert_gating_func_type   gating_op,
            llm_ffn_op_type   type_op_shexp,
         const llm_build_cb & cb, int il, ggml_cgraph * graph, bool add_input = false,
         ggml_tensor * up_gate_exps = nullptr, ggml_tensor * up_gate_exps_b = nullptr);

    static ggml_cgraph * llama_build_graph_defrag(llama_context & lctx, const std::vector<uint32_t> & ids);

    static ggml_cgraph * llama_build_graph_k_shift(llama_context & lctx);

    static ggml_cgraph * llama_build_graph_s_copy(llama_context & lctx);

    static ggml_cgraph * llama_build_graph(llama_context & lctx, const llama_batch & batch, bool worst_case);

    ggml_tensor * build_std_attention(ggml_cgraph * gf, ggml_tensor * attn_norm, ggml_tensor * cur, ggml_tensor * inp_pos, ggml_tensor * rope_factors,
            ggml_tensor * KQ_mask, ggml_tensor * sinks, ggml_tensor * inp_attn_scale, float KQ_scale, float f_attn_scale,
            int n_swa, int il, bool do_rope = true, bool add_graph_split = false, bool add_input = false, bool is_norm = false,
            bool is_multi = false);

};
