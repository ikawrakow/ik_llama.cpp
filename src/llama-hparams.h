#pragma once

#include "llama-impl.h"

#include <cstdint>
#include <array>
#include <cmath>

#define LLAMA_MAX_LAYERS  512

enum llm_expert_gating_func_type {
    LLM_EXPERT_GATING_FUNC_TYPE_NONE             = 0,
    LLM_EXPERT_GATING_FUNC_SOFTMAX               = 1,
    LLM_EXPERT_GATING_FUNC_SIGMOID               = 2,
    LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT = 3,
};

struct llama_hparams {
    bool vocab_only;
    bool rope_finetuned;
    bool use_par_res;

    uint32_t n_vocab;
    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_layer;
    int32_t n_layer_kv_from_start = -1; // if non-negative, the first n_layer_kv_from_start layers have KV cache
    uint32_t n_rot;
    uint32_t n_swa = 0; // sliding window attention (SWA)
    uint32_t n_swa_pattern = 1; // by default, all layers use non-sliding-window attention
    uint32_t n_embd_head_k; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v; // dimension of values (d_v) aka n_embd_head
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_vocab_type = 0; // for BERT-style token types
    uint32_t n_rel_attn_bkts = 0;

    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

    uint32_t n_layer_dense_lead = 0;
    uint32_t n_lora_q           = 0;
    uint32_t n_lora_kv          = 0;
    uint32_t n_ff_exp           = 0;
    uint32_t n_ff_shexp         = 0;
    uint32_t n_expert_shared    = 0;
    uint32_t n_norm_groups      = 0;
    uint32_t n_expert_groups    = 0;
    uint32_t n_group_used       = 0;
    uint32_t n_group_experts    = 0;

    float    expert_group_scale   = 0.05f;
    float    expert_weights_scale = 0.0f;
    bool     expert_weights_norm  = false;
    uint32_t expert_gating_func   = LLM_EXPERT_GATING_FUNC_SOFTMAX;
    uint32_t moe_every_n_layers   = 0;
    uint32_t nextn_predict_layers = 0;

    float f_norm_eps;
    float f_norm_rms_eps;
    float f_norm_group_eps;

    float f_attn_logit_softcapping   = 50.0f;
    float f_router_logit_softcapping = 30.0f;
    float f_final_logit_softcapping  = 30.0f;

    float    rope_attn_factor = 1.0f;
    float    rope_freq_base_train;
    float    rope_freq_base_train_swa;
    float    rope_freq_scale_train;
    float    rope_freq_scale_train_swa;
    uint32_t rope_scaling_apply_mask = 0x1;
    bool     has_rope_freq_base_per_layer = false;
    uint32_t n_ctx_orig_yarn;
    float    rope_yarn_log_mul = 0.0f;

    float    yarn_ext_factor  = -1.0f;
    float    yarn_attn_factor =  1.0f;
    float    yarn_beta_fast   = 32.0f;
    float    yarn_beta_slow   =  1.0f;

    std::array<int, 4> rope_sections;
    std::array<float,    LLAMA_MAX_LAYERS> rope_freq_base_per_layer;
    std::array<uint32_t, LLAMA_MAX_LAYERS> rope_dim_per_layer;

    // for State Space Models
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    uint32_t ssm_n_group = 0;

    // for hybrid state-space models (e.g. qwen3next)
    std::array<bool, LLAMA_MAX_LAYERS> recurrent_layer_arr;

    float f_clamp_kqv      = 0.0f;
    float f_max_alibi_bias = 0.0f;
    float f_logit_scale    = 0.0f;

    // Additional scale factors (Granite/Granite MoE)
    float f_residual_scale  = 0.0f;
    float f_embedding_scale = 0.0f;
    float f_attention_scale = 0.0f;

    // grok-2
    float    f_attn_out_scale = 0.0f;
    uint32_t attn_temp_length = 0;

    bool causal_attn   = true;
    bool use_alibi     = false;
    bool attn_soft_cap = false;

    uint32_t n_moe_layer_step        = 0;
    bool     use_kq_norm             = true;
    uint32_t n_attn_chunk            = 0;
    // values below seems to be fixed on llama4
    uint32_t n_no_rope_layer_step    = 4;
    uint32_t n_attn_temp_floor_scale = 8192;
    float    f_attn_temp_scale       = 0.1;

    // DSA (deepseek sparse attention)
    uint32_t indexer_n_head    = 0;
    uint32_t indexer_head_size = 0;
    uint32_t indexer_top_k     = 0;

	// qwen3vl deepstack
    uint32_t n_deepstack_layers = 0;

    // needed by encoder-decoder models (e.g. T5, FLAN-T5)
    // ref: https://github.com/ggerganov/llama.cpp/pull/8141
    llama_token dec_start_token_id = -1;

    enum llama_pooling_type      pooling_type            = LLAMA_POOLING_TYPE_NONE;
    enum llama_rope_type         rope_type               = LLAMA_ROPE_TYPE_NONE;
    enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;

    std::array<uint32_t, LLAMA_MAX_LAYERS> swa_layers;

    std::array<float, LLAMA_MAX_LAYERS> swiglu_limits;
    std::array<float, LLAMA_MAX_LAYERS> swiglu_limits_shared;

    bool operator!=(const llama_hparams & other) const {
        if (this->vocab_only    != other.vocab_only)    return true;
        if (this->n_vocab       != other.n_vocab)       return true;
        if (this->n_ctx_train   != other.n_ctx_train)   return true;
        if (this->n_embd        != other.n_embd)        return true;
        if (this->n_layer       != other.n_layer)       return true;
        if (this->n_rot         != other.n_rot)         return true;
        if (this->n_swa         != other.n_swa)         return true;
        if (this->n_swa_pattern != other.n_swa_pattern) return false;
        if (this->n_embd_head_k != other.n_embd_head_k) return true;
        if (this->n_embd_head_v != other.n_embd_head_v) return true;
        if (this->n_expert      != other.n_expert)      return true;
        if (this->n_expert_used != other.n_expert_used) return true;

        if (this->n_head_arr    != other.n_head_arr)    return true;
        if (this->n_head_kv_arr != other.n_head_kv_arr) return true;
        if (this->n_ff_arr      != other.n_ff_arr)      return true;

        if (this->n_rel_attn_bkts    != other.n_rel_attn_bkts)    return true;
        if (this->n_layer_dense_lead != other.n_layer_dense_lead) return true;
        if (this->n_lora_q           != other.n_lora_q)           return true;
        if (this->n_lora_kv          != other.n_lora_kv)          return true;
        if (this->n_ff_exp           != other.n_ff_exp)           return true;
        if (this->n_ff_shexp         != other.n_ff_shexp)         return true;
        if (this->n_expert_shared    != other.n_expert_shared)    return true;

        if (this->rope_finetuned  != other.rope_finetuned)  return true;
        if (this->n_ctx_orig_yarn != other.n_ctx_orig_yarn) return true;

        if (this->ssm_d_conv  != other.ssm_d_conv)  return true;
        if (this->ssm_d_inner != other.ssm_d_inner) return true;
        if (this->ssm_d_state != other.ssm_d_state) return true;
        if (this->ssm_dt_rank != other.ssm_dt_rank) return true;
        if (this->ssm_n_group != other.ssm_n_group) return true;
        if (this->recurrent_layer_arr != other.recurrent_layer_arr) return true;

        if (this->dec_start_token_id != other.dec_start_token_id) return true;

        const float EPSILON = 1e-9f;

        if (!is_float_close(this->f_norm_eps,            other.f_norm_eps,            EPSILON)) return true;
        if (!is_float_close(this->f_norm_rms_eps,        other.f_norm_rms_eps,        EPSILON)) return true;
        if (!is_float_close(this->rope_attn_factor,      other.rope_attn_factor,      EPSILON)) return true;
        if (!is_float_close(this->rope_freq_base_train,  other.rope_freq_base_train,  EPSILON)) return true;
        if (!is_float_close(this->rope_freq_scale_train, other.rope_freq_scale_train, EPSILON)) return true;
        if (!is_float_close(this->expert_weights_scale,  other.expert_weights_scale,  EPSILON)) return true;
        if (!is_float_close(this->rope_yarn_log_mul,     other.rope_yarn_log_mul,     EPSILON)) return true;
        if (!is_float_close(this->f_residual_scale,      other.f_residual_scale,      EPSILON)) return true;
        if (!is_float_close(this->f_embedding_scale,     other.f_embedding_scale,     EPSILON)) return true;
        if (!is_float_close(this->f_attention_scale,     other.f_attention_scale,     EPSILON)) return true;

        return false;
    }

    uint32_t n_head(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_arr[il];
        }
        printf("%s: Oops, il = %d\n", __func__, il);
        GGML_ABORT("fatal error");
    }

    uint32_t n_head_kv(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_kv_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    uint32_t n_embd_inp() const {
        uint32_t n_embd_inp = n_embd;

        if (n_deepstack_layers > 0) {
            n_embd_inp += n_embd * n_deepstack_layers;
        }

        return n_embd_inp;
    }

    uint32_t n_ff(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_ff_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    uint32_t n_gqa(uint32_t il = 0) const {
        const uint32_t n_head    = this->n_head(il);
        const uint32_t n_head_kv = this->n_head_kv(il);

        if (n_head_kv == 0) {
            return 0;
        }

        return n_head/n_head_kv;
    }

    uint32_t n_embd_k_gqa(uint32_t il = 0) const { // dimension of key embeddings across all k-v heads
        const uint32_t n_head_kv = this->n_head_kv(il);

        return n_embd_head_k * n_head_kv;
    }

    uint32_t n_embd_v_gqa(uint32_t il = 0) const { // dimension of value embeddings across all k-v heads
        const uint32_t n_head_kv = this->n_head_kv(il);

        return n_embd_head_v * n_head_kv;
    }

    uint32_t n_embd_k_s() const { // dimension of the rolling state embeddings
        if (ssm_n_group > 0) {
            // qwen3next keeps all recurrent state in the V-cache tail
            return 0;
        }
        // corresponds to Mamba's conv_states size
        // TODO: maybe support other convolution strides than 1
        // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
        return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * ssm_d_inner;
    }

    uint32_t n_embd_v_s() const { // dimension of the recurrent state embeddings
        if (ssm_n_group > 0) {
            // qwen3next recurrent state packs:
            // 1) conv state: (d_conv - 1) * (2 * key_dim + value_dim)
            // 2) delta-net state: head_v_dim * head_v_dim * num_v_heads
            const uint32_t key_dim        = ssm_d_state * ssm_n_group;
            const uint32_t value_dim      = ssm_d_inner;
            const uint32_t conv_dim       = 2 * key_dim + value_dim;
            const uint32_t conv_state_dim = (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * conv_dim;
            const uint32_t head_v_dim     = ssm_dt_rank > 0 ? ssm_d_inner / ssm_dt_rank : 0;
            const uint32_t ssm_state_dim  = head_v_dim * head_v_dim * ssm_dt_rank;
            return conv_state_dim + ssm_state_dim;
        }
        // corresponds to Mamba's ssm_states size
        return ssm_d_state * ssm_d_inner;
    }

    bool is_recurrent(uint32_t il) const {
        return il < n_layer ? recurrent_layer_arr[il] : false;
    }

    static bool is_float_close(float a, float b, float abs_tol) {
        // Check for non-negative tolerance
        if (abs_tol < 0.0) {
            throw std::invalid_argument("Tolerance must be non-negative");
        }

        // Exact equality check
        if (a == b) {
            return true;
        }

        // Check for infinities
        if (std::isinf(a) || std::isinf(b)) {
            return false;
        }

        // Regular comparison using the provided absolute tolerance
        return std::fabs(b - a) <= abs_tol;
    }

    uint32_t rope_n_rot(uint32_t il) const {
        const uint32_t v = rope_dim_per_layer[il];
        return v ? v : n_rot;
    }

    static const char * rope_scaling_type_name(llama_rope_scaling_type);

};

static_assert(std::is_trivially_copyable<llama_hparams>::value, "llama_hparams must be trivially copyable");
