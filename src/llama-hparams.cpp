
#include "llama-hparams.h"
#include "llama-model-loader.h"
#include "llama-model.h"

#include <map>

#define LLAMA_MAX_EXPERTS 512  // Qwen3 Next

static const std::map<llama_rope_scaling_type, const char *> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,   "none"   },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR, "linear" },
    { LLAMA_ROPE_SCALING_TYPE_YARN,   "yarn"   },
};

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string & name) {
    for (const auto & kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type) kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

const char * llama_hparams::rope_scaling_type_name(llama_rope_scaling_type type) {
    return LLAMA_ROPE_SCALING_TYPES.at(type);
}

static inline const char * llm_expert_gating_func_name(llm_expert_gating_func_type type) {
    switch (type) {
        case LLM_EXPERT_GATING_FUNC_SOFTMAX: return "softmax";
        case LLM_EXPERT_GATING_FUNC_SIGMOID: return "sigmoid";
        case LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT: return "weight";
        default: return "none";
    }
}


void llm_load_hparams(
        llama_model_loader & ml,
        llama_model & model) {
    auto & hparams = model.hparams;
    const gguf_context * ctx = ml.meta;

    // get metadata as string
    for (int i = 0; i < gguf_get_n_kv(ctx); i++) {
        enum gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        const char * name = gguf_get_key(ctx, i);
        const std::string value = gguf_kv_to_str(ctx, i);
        model.gguf_kv.emplace(name, value);
    }

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, model.name, false);

    // get hparams kv
    ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, hparams.n_vocab);

    // everything past this point is not vocab-related
    if (hparams.vocab_only) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH,    hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH,  hparams.n_embd);
    ml.get_key(LLM_KV_BLOCK_COUNT,       hparams.n_layer);
    ml.get_key(LLM_KV_EXPERT_COUNT,      hparams.n_expert,      false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used, false);

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
    } else {
        GGML_ASSERT(hparams.n_expert_used == 0);
    }

    // zero-out the per-layer hparams
    std::fill(hparams.n_head_arr.begin(),    hparams.n_head_arr.end(),    0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(),      hparams.n_ff_arr.end(),      0);
    std::fill(hparams.recurrent_layer_arr.begin(), hparams.recurrent_layer_arr.end(), false);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH,  hparams.n_ff_arr,   hparams.n_layer);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;

    // by default assume that the sliding-window layers use the same scaling type as the non-sliding-window layers
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k, false);

        hparams.n_embd_head_v = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v, false);

        // sanity check for n_rot (optional)
        hparams.n_rot = hparams.n_embd_head_k;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);

        if (model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_FALCON || model.arch == LLM_ARCH_BITNET_25 || model.arch == LLM_ARCH_BITNET_B158 || model.arch == LLM_ARCH_DECI) {
            if (hparams.n_rot != hparams.n_embd_head_k) {
                throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.n_embd_head_k));
            }
        }
    } else {
        hparams.n_rot = 0;
        hparams.n_embd_head_k = 0;
        hparams.n_embd_head_v = 0;
    }

    // arch-specific KVs
    switch (model.arch) {
        case LLM_ARCH_LLAMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                if (hparams.n_expert == 8) {
                    switch (hparams.n_layer) {
                        case 32: model.type = e_model::MODEL_8x7B; break;
                        case 56: model.type = e_model::MODEL_8x22B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                } else {
                    switch (hparams.n_layer) {
                        case 22: model.type = e_model::MODEL_1B; break;
                        case 26: model.type = e_model::MODEL_3B; break;
                        // granite uses a vocab with len 49152
                        case 32: model.type = hparams.n_vocab == 49152 ? e_model::MODEL_3B : (hparams.n_vocab < 40000 ? e_model::MODEL_7B : e_model::MODEL_8B); break;
                        case 36: model.type = e_model::MODEL_8B; break; // granite
                        case 40: model.type = e_model::MODEL_13B; break;
                        case 48: model.type = e_model::MODEL_34B; break;
                        case 60: model.type = e_model::MODEL_30B; break;
                        case 80: model.type = hparams.n_head() == hparams.n_head_kv() ? e_model::MODEL_65B : e_model::MODEL_70B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                }
            } break;
        case LLM_ARCH_LLAMA4:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,   hparams.n_moe_layer_step);
                hparams.n_swa_pattern = 4;    // pattern: 3 chunked - 1 full
                hparams.n_attn_chunk  = 8192; // should this be a gguf kv? currently it's the same for Scout and Maverick
                hparams.n_swa = 1; // TODO @ngxson : this is added to trigger the SWA branch (we store the chunked attn mask in the SWA tensor), will need to clean this up later

                switch (hparams.n_expert) {
                    case 16:  model.type = MODEL_17B_16E; break;
                    case 128: model.type = MODEL_17B_128E; break;
                    default:  model.type = MODEL_UNKNOWN;
                }

                if (model.type == MODEL_17B_128E) {
                    hparams.use_kq_norm = false;
                    }
            } break;
        case LLM_ARCH_DECI:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
	            case 162: model.type = e_model::MODEL_405B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MINICPM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_2B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GROK:
            {
                // defaults for old GGUFs
                hparams.yarn_beta_fast = 8.0f;
                hparams.f_logit_scale = 0.5773502691896257f;
                hparams.f_embedding_scale = 78.38367176906169f;
                hparams.f_attn_out_scale = 0.08838834764831845f;
                hparams.f_attn_logit_softcapping = 30.0f;
                hparams.f_router_logit_softcapping = 30.0f;
                // no final_logit_softcapping in grok-1
                hparams.f_final_logit_softcapping = 0.0f;

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale, false);
                ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale, false);
                ml.get_key(LLM_KV_ATTENTION_OUTPUT_SCALE, hparams.f_attn_out_scale, false);
                ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING, hparams.f_attn_logit_softcapping, false);
                ml.get_key(LLM_KV_ROUTER_LOGIT_SOFTCAPPING, hparams.f_router_logit_softcapping, false);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

                ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_LENGTH, hparams.attn_temp_length, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR, hparams.yarn_ext_factor, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR, hparams.yarn_attn_factor, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, hparams.yarn_beta_fast, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW, hparams.yarn_beta_slow, false);

                switch (hparams.n_layer) {
                    case 64: model.type = e_model::MODEL_314B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_FALCON:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 60: model.type = e_model::MODEL_40B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BAICHUAN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                if (model.type == e_model::MODEL_13B) {
                    // TODO: become GGUF KV parameter
                    hparams.f_max_alibi_bias = 8.0f;
                }
            } break;
        case LLM_ARCH_STARCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 36: model.type = e_model::MODEL_3B; break;
                    case 42: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_15B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_REFACT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_1B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type, false);

                switch (hparams.n_layer) {
                    case 3:
                        model.type = e_model::MODEL_17M; break; // bge-micro
                    case 6:
                        model.type = e_model::MODEL_22M; break; // MiniLM-L6
                    case 12:
                        switch (hparams.n_embd) {
                            case 384: model.type = e_model::MODEL_33M; break; // MiniLM-L12, bge-small
                            case 768: model.type = e_model::MODEL_109M; break; // bge-base
                        } break;
                    case 24:
                        model.type = e_model::MODEL_335M; break; // bge-large
                }
            } break;
        case LLM_ARCH_JINA_BERT_V2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type);
                hparams.f_max_alibi_bias = 8.0f;

                switch (hparams.n_layer) {
                    case 4: model.type = e_model::MODEL_33M; break; // jina-embeddings-small
                    case 12: model.type = e_model::MODEL_137M; break; // jina-embeddings-base
                }
            } break;
        case LLM_ARCH_NOMIC_BERT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,    hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CAUSAL,           hparams.causal_attn);
                ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
                ml.get_key(LLM_KV_POOLING_TYPE,               hparams.pooling_type);

                if (hparams.n_layer == 12 && hparams.n_embd == 768) {
                    model.type = e_model::MODEL_137M;
                }
            } break;
        case LLM_ARCH_BLOOM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 30:
                        switch (hparams.n_embd) {
                            case 2560: model.type = e_model::MODEL_3B; break;
                            case 4096: model.type = e_model::MODEL_7B; break;
                        } break;
                }

                // TODO: become GGUF KV parameter
                hparams.f_max_alibi_bias = 8.0f;
            } break;
        case LLM_ARCH_MPT:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,      hparams.f_clamp_kqv, false);
                ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 48: model.type = e_model::MODEL_30B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_STABLELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = e_model::MODEL_12B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_QWEN:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN2VL:
            {
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
            }
            // fall through
        case LLM_ARCH_QWEN2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = hparams.n_embd == 1024 ? e_model::MODEL_0_5B : e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = hparams.n_head() == 20 ? e_model::MODEL_4B : e_model::MODEL_13B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN2MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_A2_7B; break;
                    case 28: model.type = e_model::MODEL_57B_A14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;

        case LLM_ARCH_QWEN3:
            {
                ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: model.type = hparams.n_embd == 1024 ? e_model::MODEL_0_6B : e_model::MODEL_1_7B; break;
                    case 36: model.type = hparams.n_embd == 2560 ? e_model::MODEL_4B : e_model::MODEL_8B; break;
                    case 40: model.type = e_model::MODEL_14B; break;
                    case 64: model.type = e_model::MODEL_32B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN3VL:
            {
                ml.get_key(LLM_KV_NUM_DEEPSTACK_LAYERS, hparams.n_deepstack_layers, false);
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: model.type = e_model::MODEL_1_7B; break;
                    case 36: model.type = hparams.n_embd == 2560 ? e_model::MODEL_4B : e_model::MODEL_8B; break;
                    case 64: model.type = e_model::MODEL_32B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
                // since vision model stacks deepstack features along feature dim
                // we also create a fake "n_embd" for text model to be the main embd + deepstack embds
                hparams.n_embd *= hparams.n_deepstack_layers + 1;
            } break;
        case LLM_ARCH_QWEN3MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_30B_A3B; break;
                    case 94: model.type = e_model::MODEL_235B_A22B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN3NEXT:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
                ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

                // Upstream convention: every 4th layer is full attention, others are recurrent.
                for (uint32_t i = 0; i < hparams.n_layer; ++i) {
                    hparams.recurrent_layer_arr[i] = ((i + 1) % 4 != 0);
                }

                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_80B_A3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_QWEN3VLMOE:
            {
                ml.get_key(LLM_KV_NUM_DEEPSTACK_LAYERS, hparams.n_deepstack_layers, false);
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_30B_A3B; break;
                    case 94: model.type = e_model::MODEL_235B_A22B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
                // since vision model stacks deepstack features along feature dim
                // we also create a fake "n_embd" for text model to be the main embd + deepstack embds
                hparams.n_embd *= hparams.n_deepstack_layers + 1;
            } break;
        case LLM_ARCH_PHI2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_PHI3:
            {
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                // for backward compatibility ; see: https://github.com/ggerganov/llama.cpp/pull/8931
                if ((hparams.n_layer == 32 || hparams.n_layer == 40) && hparams.n_ctx_train == 4096) {
                    // default value for Phi-3-mini-4k-instruct and Phi-3-medium-4k-instruct
                    hparams.n_swa = 2047;
                } else if (hparams.n_layer == 32 && hparams.n_head_kv(0) == 32 && hparams.n_ctx_train == 131072) {
                    // default value for Phi-3-mini-128k-instruct
                    hparams.n_swa = 262144;
                } else if (hparams.n_layer == 40 && hparams.n_ctx_train == 131072) {
                    // default value for Phi-3-medium-128k-instruct
                    hparams.n_swa = 131072;
                }
                bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                if (!found_swa && hparams.n_swa == 0) {
                    throw std::runtime_error("invalid value for sliding_window");
                }
            } break;
        case LLM_ARCH_PLAMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GPT2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 12: model.type = e_model::MODEL_SMALL; break;
                    case 24: model.type = e_model::MODEL_MEDIUM; break;
                    case 36: model.type = e_model::MODEL_LARGE; break;
                    case 48: model.type = e_model::MODEL_XL; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CODESHELL:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 42: model.type = e_model::MODEL_7B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ORION:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_INTERNLM2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 48: model.type = e_model::MODEL_20B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GEMMA:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 18: model.type = e_model::MODEL_2B; break;
                    case 28: model.type = e_model::MODEL_7B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GEMMA2:
            {
                hparams.n_swa = 4096; // default value of gemma 2
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING, hparams.f_attn_logit_softcapping, false);
                ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);
                hparams.attn_soft_cap = true;

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_2B; break;
                    case 42: model.type = e_model::MODEL_9B; break;
                    case 46: model.type = e_model::MODEL_27B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_GEMMA3:
            {
                hparams.n_swa_pattern = 6;

                hparams.rope_freq_base_train_swa  = 10000.0f;
                hparams.rope_freq_scale_train_swa = 1.0f;

                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_2B; break;
                    case 34: model.type = e_model::MODEL_4B; break;
                    case 48: model.type = e_model::MODEL_12B; break;
                    case 62: model.type = e_model::MODEL_27B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

                hparams.f_attention_scale = model.type == e_model::MODEL_27B
                    ? 1.0f / std::sqrt(float(hparams.n_embd / hparams.n_head(0)))
                    : 1.0f / std::sqrt(float(hparams.n_embd_head_k));
            } break;
        case LLM_ARCH_STARCODER2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 30: model.type = e_model::MODEL_3B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_15B; break;
                    case 52: model.type = e_model::MODEL_20B; break; // granite
                    case 88: model.type = e_model::MODEL_34B; break; // granite
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MAMBA:
            {
                ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
                ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
                ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
                ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);

                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 24:
                        switch (hparams.n_embd) {
                            case 768: model.type = e_model::MODEL_SMALL; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 48:
                        switch (hparams.n_embd) {
                            case 1024: model.type = e_model::MODEL_MEDIUM; break;
                            case 1536: model.type = e_model::MODEL_LARGE; break;
                            case 2048: model.type = e_model::MODEL_XL; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 64:
                        switch (hparams.n_embd) {
                            case 2560: model.type = e_model::MODEL_3B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_XVERSE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    case 80: model.type = e_model::MODEL_65B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COMMAND_R:
            {
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_35B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DBRX:
        {
            ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,  hparams.f_norm_eps);
            ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,      hparams.f_clamp_kqv);

            switch (hparams.n_layer) {
                case 40: model.type = e_model::MODEL_16x12B; break;
                default: model.type = e_model::MODEL_UNKNOWN;
            }
        } break;
        case LLM_ARCH_OLMO:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV,     hparams.f_clamp_kqv, false);

                switch (hparams.n_layer) {
                    case 22: model.type = e_model::MODEL_1B; break;
                    case 32: model.type = e_model::MODEL_7B; break;
                    case 80: model.type = e_model::MODEL_70B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OPENELM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                case 16: model.type = e_model::MODEL_270M; break;
                case 20: model.type = e_model::MODEL_450M; break;
                case 28: model.type = e_model::MODEL_1B; break;
                case 36: model.type = e_model::MODEL_3B; break;
                default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GPTNEOX:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_USE_PARALLEL_RESIDUAL, hparams.use_par_res);
                switch (hparams.n_layer) {
                    case 6:
                        switch (hparams.n_ff()) {
                            case 512: model.type = e_model::MODEL_14M; break;
                            case 2048: model.type = e_model::MODEL_70M; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 12:
                        switch (hparams.n_ff()) {
                            case 3072: model.type = e_model::MODEL_160M; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 16:
                        switch (hparams.n_ff()) {
                            case 8192: model.type = e_model::MODEL_1B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_ff()) {
                            case 4096: model.type = e_model::MODEL_410M; break;
                            case 8192: model.type = e_model::MODEL_1_4B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 32:
                        switch (hparams.n_ff()) {
                            case 10240: model.type = e_model::MODEL_2_8B; break;
                            case 16384: model.type = e_model::MODEL_6_9B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 36:
                        switch (hparams.n_ff()) {
                            case 20480: model.type = e_model::MODEL_12B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 44:
                        switch (hparams.n_ff()) {
                            case 24576: model.type = e_model::MODEL_20B; break;
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ARCTIC:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                if (hparams.n_expert == 128) {
                    switch (hparams.n_layer) {
                        case 35: model.type = e_model::MODEL_10B_128x3_66B; break;
                        default: model.type = e_model::MODEL_UNKNOWN;
                    }
                } else {
                    model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_DEEPSEEK2:
            {
                if (hparams.n_head_kv() == 1) {
                    int n_nead_kv = hparams.n_gqa();
                    if (n_nead_kv%4 != 0 || hparams.n_embd_head_k != 576 || hparams.n_embd_head_v != 512 ||
                        hparams.n_rot != 64) {
                        printf("==========================================================================\n");
                        printf("Detected incompatible DeepSeek model without a known way to fix it.\n");
                        printf("Consider making your own ik_llama.cpp compatible model or\n");
                        printf("ask the model provider to make one for you,\n\n");
                        printf("Sorry, uknown model => cannot fix it => bailing out\n");
                        printf("==========================================================================\n");
                        GGML_ABORT("Fatal error");
                    }
                    printf("================= Adjusted mainline llama.cpp MLA tensors to ik_llama.cpp\n");
                    for (auto& item : hparams.n_head_kv_arr) item = n_nead_kv;
                    hparams.n_embd_head_k = 192;
                    hparams.n_embd_head_v = 128;
                    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k);
                    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v);
                }
                bool is_lite = (hparams.n_layer == 27 || hparams.n_layer == 26);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
                if (!is_lite) {
                    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
                }
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM, hparams.expert_weights_norm, false);
                hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_TYPE_NONE;
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    // Older DeepSeek models from the 2.0/2.5 series may not have the experts gating function recorded in the GGUF.
                    // Such models use SOFTMAX as the experts gating function.
                    // The new (new as of this commit) GLM-4.7-Flash may also be missing the experts gating function.
                    // GLM-4.7-Flash uses SIGMOID as the experts gating function.
                    // Hence, we make the LLM_KV_EXPERT_GATING_FUNC entry optional, and set here if missing.
                    // We distinguish between GLM-4.7-Flash and DeepSeek-2/2.5 models by the number of layers.
                    // GLM-4.7-Flash has 47 layers (or 48, if an MTP layer is included in the GGUF).
                    hparams.expert_gating_func = hparams.n_layer == 47 || hparams.n_layer == 48 ?
                        LLM_EXPERT_GATING_FUNC_SIGMOID : LLM_EXPERT_GATING_FUNC_SOFTMAX;
                    printf("================= Missing experts gating function -> set to %s\n",
                            llm_expert_gating_func_name(llm_expert_gating_func_type(hparams.expert_gating_func)));
                }
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, false);

                switch (hparams.n_layer) {
                    case 27: model.type = e_model::MODEL_16B; break;
                    case 47: model.type = e_model::MODEL_30B_A3B; break; // GLM-4.7-Flash
                    case 60: model.type = e_model::MODEL_236B; break;
                    case 61: model.type = e_model::MODEL_671B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_CHATGLM:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 28: model.type = e_model::MODEL_6B; break;
                    case 40: model.type = e_model::MODEL_9B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GLM4:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 40: model.type = e_model::MODEL_9B; break;
                    case 61: model.type = e_model::MODEL_32B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GLM4_MOE:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                // MoE parameters
                ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
                ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

                // Expert gating function (GLM4_MOE uses sigmoid)
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == 0) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }

                // NextN/MTP parameters
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);

                switch (hparams.n_layer) {
                    case 47: model.type = e_model::MODEL_106B_A12B; break; // GLM-4.5-Air (46 layers + 1 NextN layer)
                    case 93: model.type = e_model::MODEL_355B_A32B; break; // GLM-4.5 (92 layers + 1 NextN layer)
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BITNET:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BITNET_B158:
        case LLM_ARCH_BITNET_25:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                switch (hparams.n_layer) {
                    case 30: model.type = e_model::MODEL_2B; break; // bitnet2b_2501
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_T5:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);

                uint32_t dec_start_token_id;
                if (ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, dec_start_token_id, false)) {
                    hparams.dec_start_token_id = dec_start_token_id;
                }

                switch (hparams.n_layer) {
                    case 6:  model.type = e_model::MODEL_60M;  break; // t5-small
                    case 8:  model.type = e_model::MODEL_80M;  break; // flan-t5-small
                    case 12:
                        switch (hparams.n_ff()) {
                            case 3072: model.type = e_model::MODEL_220M; break; // t5-base
                            case 2048: model.type = e_model::MODEL_250M; break; // flan-t5-base
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    case 24:
                        switch (hparams.n_ff()) {
                            case 4096:  model.type = e_model::MODEL_770M; break; // t5-large
                            case 2816:  model.type = e_model::MODEL_780M; break; // flan-t5-large
                            case 16384: model.type = e_model::MODEL_3B;   break; // t5-3b
                            case 5120:  model.type = e_model::MODEL_3B;   break; // flan-t5-xl
                            case 65536: model.type = e_model::MODEL_11B;  break; // t5-11b
                            case 10240: model.type = e_model::MODEL_11B;  break; // flan-t5-xxl
                            default: model.type = e_model::MODEL_UNKNOWN;
                        } break;
                    default: model.type = e_model::MODEL_UNKNOWN;
               }
            } break;
        case LLM_ARCH_T5ENCODER:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);
                model.type = e_model::MODEL_UNKNOWN;
            } break;
        case LLM_ARCH_JAIS:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

                switch (hparams.n_layer) {
                    case 24: model.type = e_model::MODEL_1_3B; break;
                    case 40: model.type = e_model::MODEL_13B; break;
                    /* TODO: add variants */
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
                ml.get_key(LLM_KV_RESIDUAL_SCALE, hparams.f_residual_scale);
                ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale);
                ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_3B; break;
                    case 40: model.type = e_model::MODEL_3B; break;
                    // Add additional layer/vocab/etc checks here for other model sizes
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_COHERE2:
            {
                hparams.n_swa_pattern = 4;
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_8B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_BAILINGMOE2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,                hparams.n_expert_groups);
                ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT,           hparams.n_group_used);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);

                // TODO: when MTP is implemented, this should probably be updated if needed
                hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;

                switch (hparams.n_layer) {
                    case 20: model.type = MODEL_16B_A1B; break;
                    case 21: model.type = MODEL_16B_A1B; break;
                    case 32: model.type = MODEL_100B_A6B; break;
                    case 33: model.type = MODEL_100B_A6B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
	    case LLM_ARCH_DOTS1:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                switch (hparams.n_layer) {
                    case 62: model.type = e_model::MODEL_142B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        {
            ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
            if (model.arch == LLM_ARCH_ERNIE4_5_MOE) {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_INTERLEAVE_MOE_LAYER_STEP, hparams.n_moe_layer_step);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
            }

            switch (hparams.n_layer) {
            case 18: model.type = e_model::MODEL_0_3B; break;
            case 28: model.type = e_model::MODEL_21B_A3B; break;
            case 54: model.type = e_model::MODEL_300B_A47B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            }
        } break;
        case LLM_ARCH_HUNYUAN_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp);

                switch (hparams.n_layer) {
                    case 32: model.type = e_model::MODEL_80B_A13B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_OPENAI_MOE:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

                //TODO OAI_MOE: SWA
                //hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                //hparams.set_swa_pattern(2);

                // TODO: switch (hparams.n_layer)

            } break;
        case LLM_ARCH_MINIMAX_M2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func, false);

                switch (hparams.n_layer) {
                    case 62: model.type = e_model::MODEL_230B_A10B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_SMOLLM3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                hparams.n_no_rope_layer_step = 4;

                switch (hparams.n_layer) {
                    case 36: model.type = e_model::MODEL_3B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MISTRAL3:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                ml.get_key(LLM_KV_ATTENTION_TEMPERATURE_SCALE, hparams.f_attn_temp_scale, false);

                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_FAST,   hparams.yarn_beta_fast, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   hparams.yarn_beta_slow, false);
                ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL,     hparams.rope_yarn_log_mul, false);

                if (hparams.f_attn_temp_scale != 0.0f) {
                    hparams.n_attn_temp_floor_scale = hparams.n_ctx_orig_yarn;
                    if (hparams.n_attn_temp_floor_scale == 0) {
                        throw std::runtime_error("invalid n_ctx_orig_yarn for attention temperature scaling");
                    }
                }

                // TODO: this seems to be correct with the case of mscale == mscale_all_dims == 1.0f
                //       but may need further verification with other values
                if (hparams.rope_yarn_log_mul != 0.0f) {
                    float factor = 1.0f / hparams.rope_freq_scale_train;
                    float mscale = 1.0f;
                    float mscale_all_dims = hparams.rope_yarn_log_mul;
                    static auto get_mscale = [](float scale, float mscale) {
                        return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
                    };
                    hparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);
                }

                switch (hparams.n_layer) {
                    case 26: model.type = e_model::MODEL_3B; break;
                    case 34: model.type = e_model::MODEL_8B; break;
                    case 40: model.type = e_model::MODEL_14B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_MIMO2:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,   hparams.n_swa);
                ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,         hparams.rope_freq_base_train_swa);
                //TODO
                //hparams.swa_type = LLAMA_SWA_TYPE_STANDARD; // which is the same as OpenAI
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

                switch (hparams.n_layer) {
                    case 48: model.type = e_model::MODEL_310B_A15B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }

            } break;
        case LLM_ARCH_SEED_OSS:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                switch (hparams.n_layer) {
                    case 64: model.type = e_model::MODEL_36B; break;
                    default: model.type = e_model::MODEL_UNKNOWN;
                }
            } break;
        case LLM_ARCH_STEP35:
            {
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
                //hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
                // MoE + SWA parameters
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
                // Step35 uses sigmoid gating by default (if not set in GGUF)
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }
                ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
                bool have_rfb_train_swa = ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);
                ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);
                if (!ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_COUNT_PER_LAYER, hparams.rope_dim_per_layer, hparams.n_layer, false)) {
                    for (int i = 0; i < hparams.n_layer; ++i) {
                        hparams.rope_dim_per_layer[i] = hparams.swa_layers[i] ? hparams.n_rot : hparams.n_rot/2;
                    }
                }
                // The following two parameters: one of the two versions must be present in the GGUF
                if (!ml.get_key_or_arr(LLM_KV_SWIGLU_LIMITS,        hparams.swiglu_limits,        hparams.n_layer, false)) {
                     ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_limits, hparams.n_layer, true);
                }
                if (!ml.get_key_or_arr(LLM_KV_SWIGLU_LIMITS_SHARED, hparams.swiglu_limits_shared, hparams.n_layer, false)) {
                     ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,   hparams.swiglu_limits_shared, hparams.n_layer, true);
                }
                // Optional: Step35-only gating for applying rope scaling (HF: yarn_only_types).
                // Default is 3 (apply on all layers) if the key is absent.
                //ml.get_key(format("%s.rope.scaling.apply_mask", ml.get_arch_name().c_str()),
                //    hparams.rope_scaling_apply_mask, false);
                //hparams.has_rope_freq_base_per_layer = ml.get_key_or_arr(
                //    format("%s.rope.freq_base_per_layer", ml.get_arch_name().c_str()),
                //    hparams.rope_freq_base_per_layer, hparams.n_layer, false);
                ml.get_key(format("%s.rope.scaling.apply_mask", ml.get_arch_name().c_str()),
                    hparams.rope_scaling_apply_mask, false);
                hparams.has_rope_freq_base_per_layer = ml.get_key_or_arr(LLM_KV_ROPE_FREQ_BASE_PER_LAYER,
                    hparams.rope_freq_base_per_layer, hparams.n_layer, false);
                GGML_ASSERT(hparams.has_rope_freq_base_per_layer || have_rfb_train_swa);
            } break;
        case LLM_ARCH_GLM_DSA:
            {
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,     hparams.n_ff_exp);
                ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
                ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

                // MoE parameters
                ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
                ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
                ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
                ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

                // deepseek MLA parameters
                ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
                ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
                //ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
                //ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);
                ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
                ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);

                // DSA parameters
                ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
                ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

                // Expert gating function (GLM-4.5 uses sigmoid)
                ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
                if (hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_NONE) {
                    hparams.expert_gating_func = LLM_EXPERT_GATING_FUNC_SIGMOID;
                }

                // NextN/MTP parameters
                ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,        hparams.nextn_predict_layers, false);

                // TODO: when MTP is implemented, this should probably be updated if needed
                hparams.n_layer_kv_from_start = hparams.n_layer - hparams.nextn_predict_layers;

                switch (hparams.n_layer) {
                    case 79: model.type = MODEL_744B_A40B; break;
                    default: model.type = MODEL_UNKNOWN;
                }
                if (hparams.n_head_kv() == 1) {
                    int n_nead_kv = hparams.n_gqa();
                    if (n_nead_kv%4 != 0 || hparams.n_embd_head_k != 576 || hparams.n_embd_head_v != 512 ||
                        hparams.n_rot != 64) {
                        printf("==========================================================================\n");
                        printf("Detected incompatible DeepSeek model without a known way to fix it.\n");
                        printf("Sorry, uknown model => cannot fix it => bailing out\n");
                        printf("==========================================================================\n");
                        GGML_ABORT("Fatal error");
                    }
                    printf("================= Adjusted mainline llama.cpp MLA tensors to ik_llama.cpp\n");
                    for (auto& item : hparams.n_head_kv_arr) item = n_nead_kv;
                    hparams.n_embd_head_k = 192;
                    hparams.n_embd_head_v = 128;
                    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k);
                    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v);
                }
            } break;
        default: (void)0;
    }

    model.ftype = ml.ftype;

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_rope_type(&model);
}
