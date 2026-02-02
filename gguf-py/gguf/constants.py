from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import Any

#
# constants
#

GGUF_MAGIC             = 0x46554747  # "GGUF"
GGUF_VERSION           = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGML_QUANT_VERSION     = 2  # GGML_QNT_VERSION from ggml.h

#
# metadata keys
#


class Keys:
    class General:
        TYPE                       = "general.type"
        ARCHITECTURE               = "general.architecture"
        QUANTIZATION_VERSION       = "general.quantization_version"
        ALIGNMENT                  = "general.alignment"
        FILE_TYPE                  = "general.file_type"

        # Authorship Metadata
        NAME                       = "general.name"
        AUTHOR                     = "general.author"
        VERSION                    = "general.version"
        ORGANIZATION               = "general.organization"

        FINETUNE                   = "general.finetune"
        BASENAME                   = "general.basename"

        DESCRIPTION                = "general.description"
        QUANTIZED_BY               = "general.quantized_by"

        SIZE_LABEL                 = "general.size_label"

        # Licensing details
        LICENSE                    = "general.license"
        LICENSE_NAME               = "general.license.name"
        LICENSE_LINK               = "general.license.link"

        # Typically represents the converted GGUF repo (Unless native)
        URL                        = "general.url" # Model Website/Paper
        DOI                        = "general.doi"
        UUID                       = "general.uuid"
        REPO_URL                   = "general.repo_url" # Model Source Repository (git/svn/etc...)

        # Model Source during conversion
        SOURCE_URL                 = "general.source.url" # Model Website/Paper
        SOURCE_DOI                 = "general.source.doi"
        SOURCE_UUID                = "general.source.uuid"
        SOURCE_REPO_URL            = "general.source.repo_url" # Model Source Repository (git/svn/etc...)

        # Base Model Source. There can be more than one source if it's a merged
        # model like with 'Mistral-7B-Merge-14-v0.1'. This will assist in
        # tracing linage of models as it is finetuned or merged over time.
        BASE_MODEL_COUNT           = "general.base_model.count"
        BASE_MODEL_NAME            = "general.base_model.{id}.name"
        BASE_MODEL_AUTHOR          = "general.base_model.{id}.author"
        BASE_MODEL_VERSION         = "general.base_model.{id}.version"
        BASE_MODEL_ORGANIZATION    = "general.base_model.{id}.organization"
        BASE_MODEL_URL             = "general.base_model.{id}.url" # Model Website/Paper
        BASE_MODEL_DOI             = "general.base_model.{id}.doi"
        BASE_MODEL_UUID            = "general.base_model.{id}.uuid"
        BASE_MODEL_REPO_URL        = "general.base_model.{id}.repo_url" # Model Source Repository (git/svn/etc...)

        # Array based KV stores
        TAGS                       = "general.tags"
        LANGUAGES                  = "general.languages"
        DATASETS                   = "general.datasets"

    class LLM:
        VOCAB_SIZE                        = "{arch}.vocab_size"
        CONTEXT_LENGTH                    = "{arch}.context_length"
        EMBEDDING_LENGTH                  = "{arch}.embedding_length"
        BLOCK_COUNT                       = "{arch}.block_count"
        LEADING_DENSE_BLOCK_COUNT         = "{arch}.leading_dense_block_count"
        FEED_FORWARD_LENGTH               = "{arch}.feed_forward_length"
        EXPERT_FEED_FORWARD_LENGTH        = "{arch}.expert_feed_forward_length"
        EXPERT_SHARED_FEED_FORWARD_LENGTH = "{arch}.expert_shared_feed_forward_length"
        USE_PARALLEL_RESIDUAL             = "{arch}.use_parallel_residual"
        TENSOR_DATA_LAYOUT                = "{arch}.tensor_data_layout"
        EXPERT_COUNT                      = "{arch}.expert_count"
        EXPERT_USED_COUNT                 = "{arch}.expert_used_count"
        EXPERT_SHARED_COUNT               = "{arch}.expert_shared_count"
        EXPERT_GROUP_COUNT                = "{arch}.expert_group_count"
        EXPERT_GROUP_USED_COUNT           = "{arch}.expert_group_used_count"
        EXPERT_WEIGHTS_SCALE              = "{arch}.expert_weights_scale"
        EXPERT_WEIGHTS_NORM               = "{arch}.expert_weights_norm"
        EXPERT_GATING_FUNC                = "{arch}.expert_gating_func"
        NEXTN_PREDICT_LAYERS              = "{arch}.nextn_predict_layers"
        POOLING_TYPE                      = "{arch}.pooling_type"
        LOGIT_SCALE                       = "{arch}.logit_scale"
        DECODER_START_TOKEN_ID            = "{arch}.decoder_start_token_id"
        ATTN_LOGIT_SOFTCAPPING            = "{arch}.attn_logit_softcapping"
        FINAL_LOGIT_SOFTCAPPING           = "{arch}.final_logit_softcapping"
        ROUTER_LOGIT_SOFTCAPPING          = "{arch}.router_logit_softcapping"

    class Attention:
        HEAD_COUNT        = "{arch}.attention.head_count"
        HEAD_COUNT_KV     = "{arch}.attention.head_count_kv"
        MAX_ALIBI_BIAS    = "{arch}.attention.max_alibi_bias"
        CLAMP_KQV         = "{arch}.attention.clamp_kqv"
        KEY_LENGTH        = "{arch}.attention.key_length"
        VALUE_LENGTH      = "{arch}.attention.value_length"
        LAYERNORM_EPS     = "{arch}.attention.layer_norm_epsilon"
        LAYERNORM_RMS_EPS = "{arch}.attention.layer_norm_rms_epsilon"
        CAUSAL            = "{arch}.attention.causal"
        Q_LORA_RANK       = "{arch}.attention.q_lora_rank"
        KV_LORA_RANK      = "{arch}.attention.kv_lora_rank"
        REL_BUCKETS_COUNT = "{arch}.attention.relative_buckets_count"
        SLIDING_WINDOW    = "{arch}.attention.sliding_window"
        OUTPUT_SCALE                 = "{arch}.attention.output_scale"
        TEMPERATURE_LENGTH           = "{arch}.attention.temperature_length"

    class Rope:
        DIMENSION_COUNT          = "{arch}.rope.dimension_count"
        FREQ_BASE                = "{arch}.rope.freq_base"
        SCALING_TYPE             = "{arch}.rope.scaling.type"
        SCALING_FACTOR           = "{arch}.rope.scaling.factor"
        SCALING_ATTN_FACTOR      = "{arch}.rope.scaling.attn_factor"
        SCALING_ORIG_CTX_LEN     = "{arch}.rope.scaling.original_context_length"
        SCALING_FINETUNED        = "{arch}.rope.scaling.finetuned"
        SCALING_YARN_LOG_MUL     = "{arch}.rope.scaling.yarn_log_multiplier"
        SCALING_YARN_EXT_FACTOR  = "{arch}.rope.scaling.yarn_ext_factor"
        SCALING_YARN_ATTN_FACTOR = "{arch}.rope.scaling.yarn_attn_factor"
        SCALING_YARN_BETA_FAST   = "{arch}.rope.scaling.yarn_beta_fast"
        SCALING_YARN_BETA_SLOW   = "{arch}.rope.scaling.yarn_beta_slow"

    class Split:
        LLM_KV_SPLIT_NO            = "split.no"
        LLM_KV_SPLIT_COUNT         = "split.count"
        LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count"

    class SSM:
        CONV_KERNEL    = "{arch}.ssm.conv_kernel"
        INNER_SIZE     = "{arch}.ssm.inner_size"
        STATE_SIZE     = "{arch}.ssm.state_size"
        TIME_STEP_RANK = "{arch}.ssm.time_step_rank"

    class Tokenizer:
        MODEL                = "tokenizer.ggml.model"
        PRE                  = "tokenizer.ggml.pre"
        LIST                 = "tokenizer.ggml.tokens"
        TOKEN_TYPE           = "tokenizer.ggml.token_type"
        TOKEN_TYPE_COUNT     = "tokenizer.ggml.token_type_count"  # for BERT-style token types
        SCORES               = "tokenizer.ggml.scores"
        MERGES               = "tokenizer.ggml.merges"
        BOS_ID               = "tokenizer.ggml.bos_token_id"
        EOS_ID               = "tokenizer.ggml.eos_token_id"
        UNK_ID               = "tokenizer.ggml.unknown_token_id"
        SEP_ID               = "tokenizer.ggml.seperator_token_id"
        PAD_ID               = "tokenizer.ggml.padding_token_id"
        CLS_ID               = "tokenizer.ggml.cls_token_id"
        MASK_ID              = "tokenizer.ggml.mask_token_id"
        ADD_BOS              = "tokenizer.ggml.add_bos_token"
        ADD_EOS              = "tokenizer.ggml.add_eos_token"
        ADD_PREFIX           = "tokenizer.ggml.add_space_prefix"
        REMOVE_EXTRA_WS      = "tokenizer.ggml.remove_extra_whitespaces"
        PRECOMPILED_CHARSMAP = "tokenizer.ggml.precompiled_charsmap"
        HF_JSON              = "tokenizer.huggingface.json"
        RWKV                 = "tokenizer.rwkv.world"
        CHAT_TEMPLATE        = "tokenizer.chat_template"
        CHAT_TEMPLATE_N      = "tokenizer.chat_template.{name}"
        CHAT_TEMPLATES       = "tokenizer.chat_templates"
        # FIM/Infill special tokens constants
        FIM_PRE_ID           = "tokenizer.ggml.fim_pre_token_id"
        FIM_SUF_ID           = "tokenizer.ggml.fim_suf_token_id"
        FIM_MID_ID           = "tokenizer.ggml.fim_mid_token_id"
        FIM_PAD_ID           = "tokenizer.ggml.fim_pad_token_id"
        FIM_REP_ID           = "tokenizer.ggml.fim_rep_token_id"
        FIM_SEP_ID           = "tokenizer.ggml.fim_sep_token_id"
        # FIM/Infill special tokens constants
        PREFIX_ID            = "tokenizer.ggml.prefix_token_id"
        SUFFIX_ID            = "tokenizer.ggml.suffix_token_id"
        MIDDLE_ID            = "tokenizer.ggml.middle_token_id"
        EOT_ID               = "tokenizer.ggml.eot_token_id"
        EOM_ID               = "tokenizer.ggml.eom_token_id"

    class Adapter:
        TYPE       = "adapter.type"
        LORA_ALPHA = "adapter.lora.alpha"

#
# recommended mapping of model tensor names for storage in gguf
#


class GGUFType:
    MODEL   = "model"
    ADAPTER = "adapter"


class MODEL_ARCH(IntEnum):
    LLAMA        = auto()
    DECI         = auto()
    FALCON       = auto()
    BAICHUAN     = auto()
    GROK         = auto()
    GPT2         = auto()
    GPTJ         = auto()
    GPTNEOX      = auto()
    MPT          = auto()
    STARCODER    = auto()
    REFACT       = auto()
    BERT         = auto()
    NOMIC_BERT   = auto()
    JINA_BERT_V2 = auto()
    BLOOM        = auto()
    STABLELM     = auto()
    QWEN         = auto()
    QWEN2        = auto()
    QWEN2MOE     = auto()
    QWEN3        = auto()
    QWEN3MOE     = auto()
    PHI2         = auto()
    PHI3         = auto()
    PLAMO        = auto()
    CODESHELL    = auto()
    ORION        = auto()
    INTERNLM2    = auto()
    MINICPM      = auto()
    GEMMA        = auto()
    GEMMA2       = auto()
    GEMMA3       = auto()
    STARCODER2   = auto()
    MAMBA        = auto()
    XVERSE       = auto()
    COMMAND_R    = auto()
    DBRX         = auto()
    OLMO         = auto()
    OPENELM      = auto()
    ARCTIC       = auto()
    DEEPSEEK2    = auto()
    GLM4_MOE     = auto()
    CHATGLM      = auto()
    BITNET       = auto()
    BITNET_25    = auto()
    T5           = auto()
    T5ENCODER    = auto()
    JAIS         = auto()
    DOTS1        = auto()
    ERNIE4_5     = auto()
    ERNIE4_5_MOE = auto()
    BAILINGMOE2  = auto()
    MINIMAXM2    = auto()
    SMOLLM3      = auto()
    SEED_OSS     = auto()

class MODEL_TENSOR(IntEnum):
    TOKEN_EMBD           = auto()
    TOKEN_EMBD_NORM      = auto()
    TOKEN_TYPES          = auto()
    POS_EMBD             = auto()
    OUTPUT               = auto()
    OUTPUT_NORM          = auto()
    ROPE_FREQS           = auto()
    ROPE_FACTORS_LONG    = auto()
    ROPE_FACTORS_SHORT   = auto()
    ATTN_Q               = auto()
    ATTN_K               = auto()
    ATTN_V               = auto()
    ATTN_QKV             = auto()
    ATTN_OUT             = auto()
    ATTN_NORM            = auto()
    ATTN_NORM_2          = auto()
    ATTN_OUT_NORM        = auto()
    ATTN_POST_NORM       = auto()
    ATTN_ROT_EMBD        = auto()
    FFN_GATE_INP         = auto()
    FFN_GATE_INP_SHEXP   = auto()
    FFN_NORM             = auto()
    FFN_PRE_NORM         = auto()
    FFN_POST_NORM        = auto()
    FFN_GATE             = auto()
    FFN_DOWN             = auto()
    FFN_UP               = auto()
    FFN_ACT              = auto()
    FFN_NORM_EXP         = auto()
    FFN_GATE_EXP         = auto()
    FFN_DOWN_EXP         = auto()
    FFN_UP_EXP           = auto()
    FFN_GATE_SHEXP       = auto()
    FFN_DOWN_SHEXP       = auto()
    FFN_UP_SHEXP         = auto()
    FFN_EXP_PROBS_B      = auto()
    ATTN_Q_NORM          = auto()
    ATTN_K_NORM          = auto()
    LAYER_OUT_NORM       = auto()
    SSM_IN               = auto()
    SSM_CONV1D           = auto()
    SSM_X                = auto()
    SSM_DT               = auto()
    SSM_A                = auto()
    SSM_D                = auto()
    SSM_OUT              = auto()
    ATTN_Q_A             = auto()
    ATTN_Q_B             = auto()
    ATTN_KV_A_MQA        = auto()
    ATTN_KV_B            = auto()
    ATTN_K_B             = auto()
    ATTN_V_B             = auto()
    ATTN_Q_A_NORM        = auto()
    ATTN_KV_A_NORM       = auto()
    FFN_SUB_NORM         = auto()
    ATTN_SUB_NORM        = auto()
    DEC_ATTN_NORM        = auto()
    DEC_ATTN_Q           = auto()
    DEC_ATTN_K           = auto()
    DEC_ATTN_V           = auto()
    DEC_ATTN_OUT         = auto()
    DEC_ATTN_REL_B       = auto()
    DEC_CROSS_ATTN_NORM  = auto()
    DEC_CROSS_ATTN_Q     = auto()
    DEC_CROSS_ATTN_K     = auto()
    DEC_CROSS_ATTN_V     = auto()
    DEC_CROSS_ATTN_OUT   = auto()
    DEC_CROSS_ATTN_REL_B = auto()
    DEC_FFN_NORM         = auto()
    DEC_FFN_GATE         = auto()
    DEC_FFN_DOWN         = auto()
    DEC_FFN_UP           = auto()
    DEC_OUTPUT_NORM      = auto()
    ENC_ATTN_NORM        = auto()
    ENC_ATTN_Q           = auto()
    ENC_ATTN_K           = auto()
    ENC_ATTN_V           = auto()
    ENC_ATTN_OUT         = auto()
    ENC_ATTN_REL_B       = auto()
    ENC_FFN_NORM         = auto()
    ENC_FFN_GATE         = auto()
    ENC_FFN_DOWN         = auto()
    ENC_FFN_UP           = auto()
    ENC_OUTPUT_NORM      = auto()
    NEXTN_EH_PROJ        = auto()   # nextn tensors (glm4moe)
    NEXTN_EMBED_TOKENS   = auto()   # nextn tensors (glm4moe)
    NEXTN_ENORM          = auto()   # nextn tensors (glm4moe)
    NEXTN_HNORM          = auto()   # nextn tensors (glm4moe)
    NEXTN_SHARED_HEAD_HEAD = auto() # nextn tensors (glm4moe)
    NEXTN_SHARED_HEAD_NORM = auto() # nextn tensors (glm4moe)


MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    MODEL_ARCH.LLAMA:          "llama",
    MODEL_ARCH.DECI:           "deci",
    MODEL_ARCH.FALCON:         "falcon",
    MODEL_ARCH.BAICHUAN:       "baichuan",
    MODEL_ARCH.GROK:           "grok",
    MODEL_ARCH.GPT2:           "gpt2",
    MODEL_ARCH.GPTJ:           "gptj",
    MODEL_ARCH.GPTNEOX:        "gptneox",
    MODEL_ARCH.MPT:            "mpt",
    MODEL_ARCH.STARCODER:      "starcoder",
    MODEL_ARCH.REFACT:         "refact",
    MODEL_ARCH.BERT:           "bert",
    MODEL_ARCH.NOMIC_BERT:     "nomic-bert",
    MODEL_ARCH.JINA_BERT_V2:   "jina-bert-v2",
    MODEL_ARCH.BLOOM:          "bloom",
    MODEL_ARCH.STABLELM:       "stablelm",
    MODEL_ARCH.QWEN:           "qwen",
    MODEL_ARCH.QWEN2:          "qwen2",
    MODEL_ARCH.QWEN2MOE:       "qwen2moe",
    MODEL_ARCH.QWEN3:          "qwen3",
    MODEL_ARCH.QWEN3MOE:       "qwen3moe",
    MODEL_ARCH.PHI2:           "phi2",
    MODEL_ARCH.PHI3:           "phi3",
    MODEL_ARCH.PLAMO:          "plamo",
    MODEL_ARCH.CODESHELL:      "codeshell",
    MODEL_ARCH.ORION:          "orion",
    MODEL_ARCH.INTERNLM2:      "internlm2",
    MODEL_ARCH.MINICPM:        "minicpm",
    MODEL_ARCH.GEMMA:          "gemma",
    MODEL_ARCH.GEMMA2:         "gemma2",
    MODEL_ARCH.GEMMA3:         "gemma3",
    MODEL_ARCH.STARCODER2:     "starcoder2",
    MODEL_ARCH.MAMBA:          "mamba",
    MODEL_ARCH.XVERSE:         "xverse",
    MODEL_ARCH.COMMAND_R:      "command-r",
    MODEL_ARCH.DBRX:           "dbrx",
    MODEL_ARCH.OLMO:           "olmo",
    MODEL_ARCH.OPENELM:        "openelm",
    MODEL_ARCH.ARCTIC:         "arctic",
    MODEL_ARCH.DEEPSEEK2:      "deepseek2",
    MODEL_ARCH.CHATGLM:        "chatglm",
    MODEL_ARCH.GLM4_MOE:       "glm4moe",
    MODEL_ARCH.BITNET:         "bitnet",
    MODEL_ARCH.BITNET_25:      "bitnet-25",
    MODEL_ARCH.T5:             "t5",
    MODEL_ARCH.T5ENCODER:      "t5encoder",
    MODEL_ARCH.JAIS:           "jais",
    MODEL_ARCH.DOTS1:          "dots1",
    MODEL_ARCH.ERNIE4_5:       "ernie4_5",
    MODEL_ARCH.ERNIE4_5_MOE:   "ernie4_5-moe",
    MODEL_ARCH.BAILINGMOE2:    "bailingmoe2",
    MODEL_ARCH.MINIMAXM2:      "minimax-m2",
    MODEL_ARCH.SMOLLM3:        "smollm3",
    MODEL_ARCH.SEED_OSS:       "seed_oss",
}

TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
    MODEL_TENSOR.TOKEN_EMBD:           "token_embd",
    MODEL_TENSOR.TOKEN_EMBD_NORM:      "token_embd_norm",
    MODEL_TENSOR.TOKEN_TYPES:          "token_types",
    MODEL_TENSOR.POS_EMBD:             "position_embd",
    MODEL_TENSOR.OUTPUT_NORM:          "output_norm",
    MODEL_TENSOR.OUTPUT:               "output",
    MODEL_TENSOR.ROPE_FREQS:           "rope_freqs",
    MODEL_TENSOR.ROPE_FACTORS_LONG:    "rope_factors_long",
    MODEL_TENSOR.ROPE_FACTORS_SHORT:   "rope_factors_short",
    MODEL_TENSOR.ATTN_NORM:            "blk.{bid}.attn_norm",
    MODEL_TENSOR.ATTN_NORM_2:          "blk.{bid}.attn_norm_2",
    MODEL_TENSOR.ATTN_QKV:             "blk.{bid}.attn_qkv",
    MODEL_TENSOR.ATTN_Q:               "blk.{bid}.attn_q",
    MODEL_TENSOR.ATTN_K:               "blk.{bid}.attn_k",
    MODEL_TENSOR.ATTN_V:               "blk.{bid}.attn_v",
    MODEL_TENSOR.ATTN_OUT:             "blk.{bid}.attn_output",
    MODEL_TENSOR.ATTN_ROT_EMBD:        "blk.{bid}.attn_rot_embd",
    MODEL_TENSOR.ATTN_Q_NORM:          "blk.{bid}.attn_q_norm",
    MODEL_TENSOR.ATTN_K_NORM:          "blk.{bid}.attn_k_norm",
    MODEL_TENSOR.ATTN_OUT_NORM:        "blk.{bid}.attn_output_norm",
    MODEL_TENSOR.ATTN_POST_NORM:       "blk.{bid}.post_attention_norm",
    MODEL_TENSOR.FFN_GATE_INP:         "blk.{bid}.ffn_gate_inp",
    MODEL_TENSOR.FFN_GATE_INP_SHEXP:   "blk.{bid}.ffn_gate_inp_shexp",
    MODEL_TENSOR.FFN_NORM:             "blk.{bid}.ffn_norm",
    MODEL_TENSOR.FFN_PRE_NORM:         "blk.{bid}.ffn_norm",
    MODEL_TENSOR.FFN_POST_NORM:        "blk.{bid}.post_ffw_norm",
    MODEL_TENSOR.FFN_GATE:             "blk.{bid}.ffn_gate",
    MODEL_TENSOR.FFN_DOWN:             "blk.{bid}.ffn_down",
    MODEL_TENSOR.FFN_UP:               "blk.{bid}.ffn_up",
    MODEL_TENSOR.FFN_GATE_SHEXP:       "blk.{bid}.ffn_gate_shexp",
    MODEL_TENSOR.FFN_DOWN_SHEXP:       "blk.{bid}.ffn_down_shexp",
    MODEL_TENSOR.FFN_UP_SHEXP:         "blk.{bid}.ffn_up_shexp",
    MODEL_TENSOR.FFN_ACT:              "blk.{bid}.ffn",
    MODEL_TENSOR.FFN_NORM_EXP:         "blk.{bid}.ffn_norm_exps",
    MODEL_TENSOR.FFN_GATE_EXP:         "blk.{bid}.ffn_gate_exps",
    MODEL_TENSOR.FFN_DOWN_EXP:         "blk.{bid}.ffn_down_exps",
    MODEL_TENSOR.FFN_UP_EXP:           "blk.{bid}.ffn_up_exps",
    MODEL_TENSOR.FFN_EXP_PROBS_B:      "blk.{bid}.exp_probs_b",
    MODEL_TENSOR.LAYER_OUT_NORM:       "blk.{bid}.layer_output_norm",
    MODEL_TENSOR.SSM_IN:               "blk.{bid}.ssm_in",
    MODEL_TENSOR.SSM_CONV1D:           "blk.{bid}.ssm_conv1d",
    MODEL_TENSOR.SSM_X:                "blk.{bid}.ssm_x",
    MODEL_TENSOR.SSM_DT:               "blk.{bid}.ssm_dt",
    MODEL_TENSOR.SSM_A:                "blk.{bid}.ssm_a",
    MODEL_TENSOR.SSM_D:                "blk.{bid}.ssm_d",
    MODEL_TENSOR.SSM_OUT:              "blk.{bid}.ssm_out",
    MODEL_TENSOR.ATTN_Q_A:             "blk.{bid}.attn_q_a",
    MODEL_TENSOR.ATTN_Q_B:             "blk.{bid}.attn_q_b",
    MODEL_TENSOR.ATTN_KV_A_MQA:        "blk.{bid}.attn_kv_a_mqa",
    MODEL_TENSOR.ATTN_KV_B:            "blk.{bid}.attn_kv_b",
    MODEL_TENSOR.ATTN_K_B:             "blk.{bid}.attn_k_b",
    MODEL_TENSOR.ATTN_V_B:             "blk.{bid}.attn_v_b",
    MODEL_TENSOR.ATTN_Q_A_NORM:        "blk.{bid}.attn_q_a_norm",
    MODEL_TENSOR.ATTN_KV_A_NORM:       "blk.{bid}.attn_kv_a_norm",
    MODEL_TENSOR.ATTN_SUB_NORM:        "blk.{bid}.attn_sub_norm",
    MODEL_TENSOR.FFN_SUB_NORM:         "blk.{bid}.ffn_sub_norm",
    MODEL_TENSOR.DEC_ATTN_NORM:        "dec.blk.{bid}.attn_norm",
    MODEL_TENSOR.DEC_ATTN_Q:           "dec.blk.{bid}.attn_q",
    MODEL_TENSOR.DEC_ATTN_K:           "dec.blk.{bid}.attn_k",
    MODEL_TENSOR.DEC_ATTN_V:           "dec.blk.{bid}.attn_v",
    MODEL_TENSOR.DEC_ATTN_OUT:         "dec.blk.{bid}.attn_o",
    MODEL_TENSOR.DEC_ATTN_REL_B:       "dec.blk.{bid}.attn_rel_b",
    MODEL_TENSOR.DEC_CROSS_ATTN_NORM:  "dec.blk.{bid}.cross_attn_norm",
    MODEL_TENSOR.DEC_CROSS_ATTN_Q:     "dec.blk.{bid}.cross_attn_q",
    MODEL_TENSOR.DEC_CROSS_ATTN_K:     "dec.blk.{bid}.cross_attn_k",
    MODEL_TENSOR.DEC_CROSS_ATTN_V:     "dec.blk.{bid}.cross_attn_v",
    MODEL_TENSOR.DEC_CROSS_ATTN_OUT:   "dec.blk.{bid}.cross_attn_o",
    MODEL_TENSOR.DEC_CROSS_ATTN_REL_B: "dec.blk.{bid}.cross_attn_rel_b",
    MODEL_TENSOR.DEC_FFN_NORM:         "dec.blk.{bid}.ffn_norm",
    MODEL_TENSOR.DEC_FFN_GATE:         "dec.blk.{bid}.ffn_gate",
    MODEL_TENSOR.DEC_FFN_DOWN:         "dec.blk.{bid}.ffn_down",
    MODEL_TENSOR.DEC_FFN_UP:           "dec.blk.{bid}.ffn_up",
    MODEL_TENSOR.DEC_OUTPUT_NORM:      "dec.output_norm",
    MODEL_TENSOR.ENC_ATTN_NORM:        "enc.blk.{bid}.attn_norm",
    MODEL_TENSOR.ENC_ATTN_Q:           "enc.blk.{bid}.attn_q",
    MODEL_TENSOR.ENC_ATTN_K:           "enc.blk.{bid}.attn_k",
    MODEL_TENSOR.ENC_ATTN_V:           "enc.blk.{bid}.attn_v",
    MODEL_TENSOR.ENC_ATTN_OUT:         "enc.blk.{bid}.attn_o",
    MODEL_TENSOR.ENC_ATTN_REL_B:       "enc.blk.{bid}.attn_rel_b",
    MODEL_TENSOR.ENC_FFN_NORM:         "enc.blk.{bid}.ffn_norm",
    MODEL_TENSOR.ENC_FFN_GATE:         "enc.blk.{bid}.ffn_gate",
    MODEL_TENSOR.ENC_FFN_DOWN:         "enc.blk.{bid}.ffn_down",
    MODEL_TENSOR.ENC_FFN_UP:           "enc.blk.{bid}.ffn_up",
    MODEL_TENSOR.ENC_OUTPUT_NORM:      "enc.output_norm",
    # NextN/MTP
    MODEL_TENSOR.NEXTN_EH_PROJ:             "blk.{bid}.nextn.eh_proj",
    MODEL_TENSOR.NEXTN_EMBED_TOKENS:        "blk.{bid}.nextn.embed_tokens",
    MODEL_TENSOR.NEXTN_ENORM:               "blk.{bid}.nextn.enorm",
    MODEL_TENSOR.NEXTN_HNORM:               "blk.{bid}.nextn.hnorm",
    MODEL_TENSOR.NEXTN_SHARED_HEAD_HEAD:    "blk.{bid}.nextn.shared_head_head",
    MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM:    "blk.{bid}.nextn.shared_head_norm",
}

MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.DECI: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.GROK: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_POST_NORM,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.GPTNEOX: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.FALCON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.STARCODER: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BERT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.NOMIC_BERT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.JINA_BERT_V2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.MPT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_ACT,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.POS_EMBD,
    ],
    MODEL_ARCH.GPTJ: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.REFACT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BLOOM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.STABLELM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
    ],
    MODEL_ARCH.QWEN: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.QWEN2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.QWEN2MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_INP_SHEXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
    ],
    MODEL_ARCH.QWEN3: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.QWEN3MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.PLAMO: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPT2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.PHI2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.PHI3: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FACTORS_LONG,
        MODEL_TENSOR.ROPE_FACTORS_SHORT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.CODESHELL: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.ORION: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.INTERNLM2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.MINICPM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.GEMMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_NORM,
    ],
    MODEL_ARCH.GEMMA2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.FFN_PRE_NORM,
        MODEL_TENSOR.FFN_POST_NORM,
    ],
    MODEL_ARCH.STARCODER2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.MAMBA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.SSM_IN,
        MODEL_TENSOR.SSM_CONV1D,
        MODEL_TENSOR.SSM_X,
        MODEL_TENSOR.SSM_DT,
        MODEL_TENSOR.SSM_A,
        MODEL_TENSOR.SSM_D,
        MODEL_TENSOR.SSM_OUT,
    ],
    MODEL_ARCH.XVERSE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.COMMAND_R: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_Q_NORM,
    ],
    MODEL_ARCH.DBRX: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.OLMO: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.OPENELM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.ARCTIC: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_NORM_EXP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.DEEPSEEK2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_A,
        MODEL_TENSOR.ATTN_Q_B,
        MODEL_TENSOR.ATTN_KV_A_MQA,
        MODEL_TENSOR.ATTN_KV_B,
        MODEL_TENSOR.ATTN_K_B,
        MODEL_TENSOR.ATTN_V_B,
        MODEL_TENSOR.ATTN_Q_A_NORM,
        MODEL_TENSOR.ATTN_KV_A_NORM,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
        MODEL_TENSOR.FFN_EXP_PROBS_B
    ],
    MODEL_ARCH.CHATGLM : [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GLM4_MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
        # NextN/MTP tensors - preserved but unused
        MODEL_TENSOR.NEXTN_EH_PROJ,
        MODEL_TENSOR.NEXTN_EMBED_TOKENS,
        MODEL_TENSOR.NEXTN_ENORM,
        MODEL_TENSOR.NEXTN_HNORM,
        MODEL_TENSOR.NEXTN_SHARED_HEAD_HEAD,
        MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM,
    ],
    MODEL_ARCH.BITNET: [
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_SUB_NORM,
        MODEL_TENSOR.FFN_SUB_NORM,
    ],
    MODEL_ARCH.BITNET_25: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.ATTN_SUB_NORM,
        MODEL_TENSOR.FFN_SUB_NORM,
    ],
    MODEL_ARCH.T5: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.DEC_ATTN_NORM,
        MODEL_TENSOR.DEC_ATTN_Q,
        MODEL_TENSOR.DEC_ATTN_K,
        MODEL_TENSOR.DEC_ATTN_V,
        MODEL_TENSOR.DEC_ATTN_OUT,
        MODEL_TENSOR.DEC_ATTN_REL_B,
        MODEL_TENSOR.DEC_CROSS_ATTN_NORM,
        MODEL_TENSOR.DEC_CROSS_ATTN_Q,
        MODEL_TENSOR.DEC_CROSS_ATTN_K,
        MODEL_TENSOR.DEC_CROSS_ATTN_V,
        MODEL_TENSOR.DEC_CROSS_ATTN_OUT,
        MODEL_TENSOR.DEC_CROSS_ATTN_REL_B,
        MODEL_TENSOR.DEC_FFN_NORM,
        MODEL_TENSOR.DEC_FFN_GATE,
        MODEL_TENSOR.DEC_FFN_DOWN,
        MODEL_TENSOR.DEC_FFN_UP,
        MODEL_TENSOR.DEC_OUTPUT_NORM,
        MODEL_TENSOR.ENC_ATTN_NORM,
        MODEL_TENSOR.ENC_ATTN_Q,
        MODEL_TENSOR.ENC_ATTN_K,
        MODEL_TENSOR.ENC_ATTN_V,
        MODEL_TENSOR.ENC_ATTN_OUT,
        MODEL_TENSOR.ENC_ATTN_REL_B,
        MODEL_TENSOR.ENC_FFN_NORM,
        MODEL_TENSOR.ENC_FFN_GATE,
        MODEL_TENSOR.ENC_FFN_DOWN,
        MODEL_TENSOR.ENC_FFN_UP,
        MODEL_TENSOR.ENC_OUTPUT_NORM,
    ],
    MODEL_ARCH.T5ENCODER: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ENC_ATTN_NORM,
        MODEL_TENSOR.ENC_ATTN_Q,
        MODEL_TENSOR.ENC_ATTN_K,
        MODEL_TENSOR.ENC_ATTN_V,
        MODEL_TENSOR.ENC_ATTN_OUT,
        MODEL_TENSOR.ENC_ATTN_REL_B,
        MODEL_TENSOR.ENC_FFN_NORM,
        MODEL_TENSOR.ENC_FFN_GATE,
        MODEL_TENSOR.ENC_FFN_DOWN,
        MODEL_TENSOR.ENC_FFN_UP,
        MODEL_TENSOR.ENC_OUTPUT_NORM,
    ],
    MODEL_ARCH.JAIS: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.DOTS1: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
    ],
    MODEL_ARCH.ERNIE4_5: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.ERNIE4_5_MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
    ],
    MODEL_ARCH.BAILINGMOE2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
        MODEL_TENSOR.NEXTN_EH_PROJ,
        MODEL_TENSOR.NEXTN_EMBED_TOKENS,
        MODEL_TENSOR.NEXTN_ENORM,
        MODEL_TENSOR.NEXTN_HNORM,
        MODEL_TENSOR.NEXTN_SHARED_HEAD_HEAD,
        MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.MINIMAXM2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
    ],
    MODEL_ARCH.SMOLLM3: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.SEED_OSS: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
    ],
    # TODO
}

# tensors that will not be serialized
MODEL_TENSOR_SKIP: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.DECI: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.QWEN: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.CODESHELL: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.ORION: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.STARCODER2: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.XVERSE: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.DEEPSEEK2: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.CHATGLM: [
        MODEL_TENSOR.ROPE_FREQS,
    ],
}

#
# types
#


class TokenType(IntEnum):
    NORMAL       = 1
    UNKNOWN      = 2
    CONTROL      = 3
    USER_DEFINED = 4
    UNUSED       = 5
    BYTE         = 6


class RopeScalingType(Enum):
    NONE   = 'none'
    LINEAR = 'linear'
    YARN   = 'yarn'


class PoolingType(IntEnum):
    NONE = 0
    MEAN = 1
    CLS  = 2


class GGMLQuantizationType(IntEnum):
    F32       =   0
    F16       =   1
    Q4_0      =   2
    Q4_1      =   3
    Q5_0      =   6
    Q5_1      =   7
    Q8_0      =   8
    Q8_1      =   9
    Q2_K      =  10
    Q3_K      =  11
    Q4_K      =  12
    Q5_K      =  13
    Q6_K      =  14
    Q8_K      =  15
    IQ2_XXS   =  16
    IQ2_XS    =  17
    IQ3_XXS   =  18
    IQ1_S     =  19
    IQ4_NL    =  20
    IQ3_S     =  21
    IQ2_S     =  22
    IQ4_XS    =  23
    I8        =  24
    I16       =  25
    I32       =  26
    I64       =  27
    F64       =  28
    IQ1_M     =  29
    BF16      =  30
    Q4_0_4_4  =  31
    Q4_0_4_8  =  32
    Q4_0_8_8  =  33
    I2_S      =  36
    MXFP4     =  39
    Q8_0_X4   =  97
    Q8_1_X4   =  98
    Q8_2_X4   =  99
    Q6_0      = 133
    IQ1_BN    = 134
    IQ2_BN    = 135
    Q8_K64    = 136
    IQ2_K     = 137
    IQ3_K     = 138
    IQ4_K     = 139
    IQ5_K     = 140
    IQ6_K     = 141
    IQ4_KS    = 144
    IQ2_KS    = 145
    IQ4_KSS   = 146
    Q8_K16    = 147
    Q8_K32    = 148
    Q8_KR8    = 149
    Q8_K128   = 150
    Q8_KV     = 151
    IQ5_KS    = 152
    IQ2_KT    = 153
    IQ3_KT    = 154
    IQ4_KT    = 155
    IQ3_KS    = 156
    IQ2_KL    = 157
    IQ1_KT    = 158
    Q4_0_R8   = 202
    Q5_0_R4   = 206
    Q8_0_R8   = 208
    Q2_K_R4   = 210
    Q3_K_R4   = 211
    Q4_K_R4   = 212
    Q5_K_R4   = 213
    Q6_K_R4   = 214
    IQ2_XXS_R4= 216
    IQ2_XS_R4 = 217
    IQ3_XXS_R4= 218
    IQ1_S_R4  = 219
    IQ4_NL_R4 = 220
    IQ3_S_R4  = 221
    IQ2_S_R4  = 222
    IQ4_XS_R8 = 223
    IQ1_M_R4  = 229
    BF16_R16  = 230
    Q6_0_R4   = 233
    IQ2_BN_R4 = 335
    IQ2_K_R4  = 337
    IQ3_K_R4  = 338
    IQ4_K_R4  = 339
    IQ5_K_R4  = 340
    IQ4_KS_R4 = 344
    IQ5_KS_R4 = 352
    Q8_KV_R8  = 398
    Q8_K_R8   = 399


class ExpertGatingFuncType(IntEnum):
    SOFTMAX  = 1
    SIGMOID  = 2


# TODO: add GGMLFileType from ggml_ftype in ggml.h


# from llama_ftype in llama.h
# ALL VALUES SHOULD BE THE SAME HERE AS THEY ARE OVER THERE.
class LlamaFileType(IntEnum):
    ALL_F32              = 0
    MOSTLY_F16           = 1   #except 1d tensors
    MOSTLY_Q4_0          = 2   #except 1d tensors
    MOSTLY_Q4_1          = 3   #except 1d tensors
    MOSTLY_Q8_0          = 7   #except 1d tensors
    MOSTLY_Q5_0          = 8   #except 1d tensors
    MOSTLY_Q5_1          = 9   #except 1d tensors
    MOSTLY_Q2_K          = 10  #except 1d tensors
    MOSTLY_Q3_K_S        = 11  #except 1d tensors
    MOSTLY_Q3_K_M        = 12  #except 1d tensors
    MOSTLY_Q3_K_L        = 13  #except 1d tensors
    MOSTLY_Q4_K_S        = 14  #except 1d tensors
    MOSTLY_Q4_K_M        = 15  #except 1d tensors
    MOSTLY_Q5_K_S        = 16  #except 1d tensors
    MOSTLY_Q5_K_M        = 17  #except 1d tensors
    MOSTLY_Q6_K          = 18  #except 1d tensors
    MOSTLY_IQ2_XXS       = 19  #except 1d tensors
    MOSTLY_IQ2_XS        = 20  #except 1d tensors
    MOSTLY_Q2_K_S        = 21  #except 1d tensors
    MOSTLY_IQ3_XS        = 22  #except 1d tensors
    MOSTLY_IQ3_XXS       = 23  #except 1d tensors
    MOSTLY_IQ1_S         = 24  #except 1d tensors
    MOSTLY_IQ4_NL        = 25  #except 1d tensors
    MOSTLY_IQ3_S         = 26  #except 1d tensors
    MOSTLY_IQ3_M         = 27  #except 1d tensors
    MOSTLY_IQ2_S         = 28  #except 1d tensors
    MOSTLY_IQ2_M         = 29  #except 1d tensors
    MOSTLY_IQ4_XS        = 30  #except 1d tensors
    MOSTLY_IQ1_M         = 31  #except 1d tensors
    MOSTLY_BF16          = 32  #except 1d tensors
    MOSTLY_Q4_0_4_4      = 33  #except 1d tensors
    MOSTLY_Q4_0_4_8      = 34  #except 1d tensors
    MOSTLY_Q4_0_8_8      = 35  #except 1d tensors
    MOSTLY_MXFP4         = 38  #except 1d tensors, 38 to be compatible with mainline

    MOSTLY_Q6_0          = 135 #except 1d tensors
    MOSTLY_IQ1_BN        = 136 #except 1d tensors
    MOSTLY_IQ2_BN        = 137 #except 1d tensors
    MOSTLY_IQ2_K         = 138 #except 1d tensors
    MOSTLY_IQ3_K         = 139 #except 1d tensors
    MOSTLY_IQ4_K         = 140 #except 1d tensors
    MOSTLY_IQ5_K         = 141 #except 1d tensors
    MOSTLY_IQ6_K         = 142 #except 1d tensors
    MOSTLY_IQ4_KS        = 145 #except 1d tensors
    MOSTLY_IQ3_KL        = 146 #except 1d tensors
    MOSTLY_IQ2_KS        = 147 #except 1d tensors
    MOSTLY_IQ4_KSS       = 148 #except 1d tensors
    MOSTLY_Q8_KV         = 149 #except 1d tensors
    MOSTLY_IQ5_KS        = 150 #except 1d tensors
    MOSTLY_IQ2_KT        = 151 #except 1d tensors
    MOSTLY_IQ3_KT        = 152 #except 1d tensors
    MOSTLY_IQ4_KT        = 153 #except 1d tensors
    MOSTLY_IQ3_KS        = 154 #except 1d tensors
    MOSTLY_IQ2_KL        = 155 #except 1d tensors
    MOSTLY_IQ1_KT        = 156 #except 1d tensors

    MOSTLY_Q4_0_R8       = 202 #except 1d tensors
    MOSTLY_Q8_0_R8       = 207 #except 1d tensors
    MOSTLY_Q5_0_R4       = 208 #except 1d tensors
    MOSTLY_Q2_K_R4       = 210 #except 1d tensors
    MOSTLY_Q3_K_R4       = 211 #except 1d tensors
    MOSTLY_Q4_K_R4       = 214 #except 1d tensors
    MOSTLY_Q5_K_R4       = 216 #except 1d tensors
    MOSTLY_Q6_K_R4       = 218 #except 1d tensors
    MOSTLY_IQ2_XXS_R4    = 219 #except 1d tensors
    MOSTLY_IQ2_XS_R4     = 220 #except 1d tensors
    MOSTLY_IQ3_XXS_R4    = 223 #except 1d tensors
    MOSTLY_IQ1_S_R4      = 224 #except 1d tensors
    MOSTLY_IQ4_NL_R4     = 225 #except 1d tensors
    MOSTLY_IQ3_S_R4      = 226 #except 1d tensors
    MOSTLY_IQ2_M_R4      = 229 #except 1d tensors
    MOSTLY_IQ4_XS_R8     = 230 #except 1d tensors
    MOSTLY_IQ1_M_R4      = 231 #except 1d tensors
    MOSTLY_Q6_0_R4       = 335 #except 1d tensors
    MOSTLY_BF16_R16      = 232 #except 1d tensors
    MOSTLY_IQ2_BN_R4     = 337 #except 1d tensors
    MOSTLY_IQ2_K_R4      = 338 #except 1d tensors
    MOSTLY_IQ3_K_R4      = 339 #except 1d tensors
    MOSTLY_IQ4_K_R4      = 340 #except 1d tensors
    MOSTLY_IQ5_K_R4      = 341 #except 1d tensors
    MOSTLY_IQ4_KS_R4     = 345 #except 1d tensors
    MOSTLY_IQ5_KS_R4     = 350 #except 1d tensors
    MOSTLY_Q8_KV_R8      = 398 #except 1d tensors
    MOSTLY_Q8_K_R8       = 399 #except 1d tensors

    GUESSED              = 1024  # not specified in the model file


class GGUFEndian(IntEnum):
    LITTLE = 0
    BIG = 1


class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

    @staticmethod
    def get_type(val: Any) -> GGUFValueType:
        if isinstance(val, (str, bytes, bytearray)):
            return GGUFValueType.STRING
        elif isinstance(val, list):
            return GGUFValueType.ARRAY
        elif isinstance(val, float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, bool):
            return GGUFValueType.BOOL
        elif isinstance(val, int):
            return GGUFValueType.INT32
        # TODO: need help with 64-bit types in Python
        else:
            raise ValueError(f"Unknown type: {type(val)}")


# Items here are (block size, type size)
QK_K = 256

#Values generated programatically
GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32         : (   1,    4),
    GGMLQuantizationType.F16         : (   1,    2),
    GGMLQuantizationType.Q4_0        : (  32,   18),
    GGMLQuantizationType.Q4_1        : (  32,   20),
    GGMLQuantizationType.Q5_0        : (  32,   22),
    GGMLQuantizationType.Q5_1        : (  32,   24),
    GGMLQuantizationType.Q8_0        : (  32,   34),
    GGMLQuantizationType.Q8_1        : (  32,   36),
    GGMLQuantizationType.Q2_K        : ( 256,   84),
    GGMLQuantizationType.Q3_K        : ( 256,  110),
    GGMLQuantizationType.Q4_K        : ( 256,  144),
    GGMLQuantizationType.Q5_K        : ( 256,  176),
    GGMLQuantizationType.Q6_K        : ( 256,  210),
    GGMLQuantizationType.Q8_K        : ( 256,  292),
    GGMLQuantizationType.IQ2_XXS     : ( 256,   66),
    GGMLQuantizationType.IQ2_XS      : ( 256,   74),
    GGMLQuantizationType.IQ3_XXS     : ( 256,   98),
    GGMLQuantizationType.IQ1_S       : ( 256,   50),
    GGMLQuantizationType.IQ4_NL      : (  32,   18),
    GGMLQuantizationType.IQ3_S       : ( 256,  110),
    GGMLQuantizationType.IQ2_S       : ( 256,   82),
    GGMLQuantizationType.IQ4_XS      : ( 256,  136),
    GGMLQuantizationType.I8          : (   1,    1),
    GGMLQuantizationType.I16         : (   1,    2),
    GGMLQuantizationType.I32         : (   1,    4),
    GGMLQuantizationType.I64         : (   1,    8),
    GGMLQuantizationType.F64         : (   1,    8),
    GGMLQuantizationType.IQ1_M       : ( 256,   56),
    GGMLQuantizationType.BF16        : (   1,    2),
    GGMLQuantizationType.MXFP4       : (  32,   17),
    GGMLQuantizationType.Q4_0_4_4    : (  32,   18),
    GGMLQuantizationType.Q4_0_4_8    : (  32,   18),
    GGMLQuantizationType.Q4_0_8_8    : (  32,   18),
    GGMLQuantizationType.I2_S        : (   1,    1),
    GGMLQuantizationType.Q8_0_X4     : (  32,   34),
    GGMLQuantizationType.Q8_1_X4     : (  32,   36),
    GGMLQuantizationType.Q8_2_X4     : (  32,   36),
    GGMLQuantizationType.Q6_0        : (  32,   26),
    GGMLQuantizationType.IQ1_BN      : (  64,   13),
    GGMLQuantizationType.IQ2_BN      : (  64,   16),
    GGMLQuantizationType.Q8_K64      : (  64,   68),
    GGMLQuantizationType.IQ2_K       : ( 256,   76),
    GGMLQuantizationType.IQ3_K       : ( 256,  110),
    GGMLQuantizationType.IQ4_K       : ( 256,  144),
    GGMLQuantizationType.IQ5_K       : ( 256,  176),
    GGMLQuantizationType.IQ6_K       : ( 256,  212),
    GGMLQuantizationType.IQ4_KS      : ( 256,  136),
    GGMLQuantizationType.IQ2_KS      : ( 256,   70),
    GGMLQuantizationType.IQ4_KSS     : ( 256,  128),
    GGMLQuantizationType.Q8_K16      : (  64,   64),
    GGMLQuantizationType.Q8_K32      : ( 256,  292),
    GGMLQuantizationType.Q8_KR8      : ( 256,  292),
    GGMLQuantizationType.Q8_K128     : ( 128,  140),
    GGMLQuantizationType.Q8_KV       : (  32,   32),
    GGMLQuantizationType.IQ5_KS      : ( 256,  168),
    GGMLQuantizationType.IQ2_KT      : ( 256,   68),
    GGMLQuantizationType.IQ3_KT      : ( 256,  100),
    GGMLQuantizationType.IQ4_KT      : ( 256,  128),
    GGMLQuantizationType.IQ3_KS      : ( 256,  102),
    GGMLQuantizationType.IQ2_KL      : ( 256,   86),
    GGMLQuantizationType.IQ1_KT      : ( 256,   56),
    GGMLQuantizationType.Q4_0_R8     : (  32,   18),
    GGMLQuantizationType.Q5_0_R4     : (  32,   22),
    GGMLQuantizationType.Q8_0_R8     : (  32,   34),
    GGMLQuantizationType.Q2_K_R4     : ( 256,   84),
    GGMLQuantizationType.Q3_K_R4     : ( 256,  110),
    GGMLQuantizationType.Q4_K_R4     : ( 256,  144),
    GGMLQuantizationType.Q5_K_R4     : ( 256,  176),
    GGMLQuantizationType.Q6_K_R4     : ( 256,  210),
    GGMLQuantizationType.IQ2_XXS_R4  : ( 256,   66),
    GGMLQuantizationType.IQ2_XS_R4   : ( 256,   74),
    GGMLQuantizationType.IQ3_XXS_R4  : ( 256,   98),
    GGMLQuantizationType.IQ1_S_R4    : (  32,    6),
    GGMLQuantizationType.IQ4_NL_R4   : (  32,   18),
    GGMLQuantizationType.IQ3_S_R4    : ( 256,  110),
    GGMLQuantizationType.IQ2_S_R4    : ( 256,   82),
    GGMLQuantizationType.IQ4_XS_R8   : ( 256,  136),
    GGMLQuantizationType.IQ1_M_R4    : (  32,    7),
    GGMLQuantizationType.BF16_R16    : (   1,    2),
    GGMLQuantizationType.Q6_0_R4     : (  32,   26),
    GGMLQuantizationType.IQ2_BN_R4   : (  64,   16),
    GGMLQuantizationType.IQ2_K_R4    : ( 256,   76),
    GGMLQuantizationType.IQ3_K_R4    : ( 256,  110),
    GGMLQuantizationType.IQ4_K_R4    : ( 256,  144),
    GGMLQuantizationType.IQ5_K_R4    : ( 256,  176),
    GGMLQuantizationType.IQ4_KS_R4   : ( 256,  136),
    GGMLQuantizationType.IQ5_KS_R4   : ( 256,  168),
    GGMLQuantizationType.Q8_KV_R8    : (  32,   32),
    GGMLQuantizationType.Q8_K_R8     : ( 256,  258),
}


# Aliases for backward compatibility.

# general
KEY_GENERAL_ARCHITECTURE         = Keys.General.ARCHITECTURE
KEY_GENERAL_QUANTIZATION_VERSION = Keys.General.QUANTIZATION_VERSION
KEY_GENERAL_ALIGNMENT            = Keys.General.ALIGNMENT
KEY_GENERAL_NAME                 = Keys.General.NAME
KEY_GENERAL_AUTHOR               = Keys.General.AUTHOR
KEY_GENERAL_URL                  = Keys.General.URL
KEY_GENERAL_DESCRIPTION          = Keys.General.DESCRIPTION
KEY_GENERAL_LICENSE              = Keys.General.LICENSE
KEY_GENERAL_SOURCE_URL           = Keys.General.SOURCE_URL
KEY_GENERAL_FILE_TYPE            = Keys.General.FILE_TYPE

# LLM
KEY_VOCAB_SIZE            = Keys.LLM.VOCAB_SIZE
KEY_CONTEXT_LENGTH        = Keys.LLM.CONTEXT_LENGTH
KEY_EMBEDDING_LENGTH      = Keys.LLM.EMBEDDING_LENGTH
KEY_BLOCK_COUNT           = Keys.LLM.BLOCK_COUNT
KEY_FEED_FORWARD_LENGTH   = Keys.LLM.FEED_FORWARD_LENGTH
KEY_USE_PARALLEL_RESIDUAL = Keys.LLM.USE_PARALLEL_RESIDUAL
KEY_TENSOR_DATA_LAYOUT    = Keys.LLM.TENSOR_DATA_LAYOUT

# attention
KEY_ATTENTION_HEAD_COUNT        = Keys.Attention.HEAD_COUNT
KEY_ATTENTION_HEAD_COUNT_KV     = Keys.Attention.HEAD_COUNT_KV
KEY_ATTENTION_MAX_ALIBI_BIAS    = Keys.Attention.MAX_ALIBI_BIAS
KEY_ATTENTION_CLAMP_KQV         = Keys.Attention.CLAMP_KQV
KEY_ATTENTION_LAYERNORM_EPS     = Keys.Attention.LAYERNORM_EPS
KEY_ATTENTION_LAYERNORM_RMS_EPS = Keys.Attention.LAYERNORM_RMS_EPS

# RoPE
KEY_ROPE_DIMENSION_COUNT      = Keys.Rope.DIMENSION_COUNT
KEY_ROPE_FREQ_BASE            = Keys.Rope.FREQ_BASE
KEY_ROPE_SCALING_TYPE         = Keys.Rope.SCALING_TYPE
KEY_ROPE_SCALING_FACTOR       = Keys.Rope.SCALING_FACTOR
KEY_ROPE_SCALING_ORIG_CTX_LEN = Keys.Rope.SCALING_ORIG_CTX_LEN
KEY_ROPE_SCALING_FINETUNED    = Keys.Rope.SCALING_FINETUNED

# SSM
KEY_SSM_CONV_KERNEL    = Keys.SSM.CONV_KERNEL
KEY_SSM_INNER_SIZE     = Keys.SSM.INNER_SIZE
KEY_SSM_STATE_SIZE     = Keys.SSM.STATE_SIZE
KEY_SSM_TIME_STEP_RANK = Keys.SSM.TIME_STEP_RANK

# tokenization
KEY_TOKENIZER_MODEL      = Keys.Tokenizer.MODEL
KEY_TOKENIZER_PRE        = Keys.Tokenizer.PRE
KEY_TOKENIZER_LIST       = Keys.Tokenizer.LIST
KEY_TOKENIZER_TOKEN_TYPE = Keys.Tokenizer.TOKEN_TYPE
KEY_TOKENIZER_SCORES     = Keys.Tokenizer.SCORES
KEY_TOKENIZER_MERGES     = Keys.Tokenizer.MERGES
KEY_TOKENIZER_BOS_ID     = Keys.Tokenizer.BOS_ID
KEY_TOKENIZER_EOS_ID     = Keys.Tokenizer.EOS_ID
KEY_TOKENIZER_UNK_ID     = Keys.Tokenizer.UNK_ID
KEY_TOKENIZER_SEP_ID     = Keys.Tokenizer.SEP_ID
KEY_TOKENIZER_PAD_ID     = Keys.Tokenizer.PAD_ID
KEY_TOKENIZER_CLS_ID     = Keys.Tokenizer.CLS_ID
KEY_TOKENIZER_MASK_ID    = Keys.Tokenizer.MASK_ID
KEY_TOKENIZER_HF_JSON    = Keys.Tokenizer.HF_JSON
KEY_TOKENIZER_RWKV       = Keys.Tokenizer.RWKV

KEY_TOKENIZER_FIM_PRE_ID = Keys.Tokenizer.FIM_PRE_ID
KEY_TOKENIZER_FIM_SUF_ID = Keys.Tokenizer.FIM_SUF_ID
KEY_TOKENIZER_FIM_MID_ID = Keys.Tokenizer.FIM_MID_ID
KEY_TOKENIZER_FIM_PAD_ID = Keys.Tokenizer.FIM_PAD_ID
KEY_TOKENIZER_FIM_REP_ID = Keys.Tokenizer.FIM_REP_ID
KEY_TOKENIZER_FIM_SEP_ID = Keys.Tokenizer.FIM_SEP_ID

KEY_TOKENIZER_PREFIX_ID  = Keys.Tokenizer.PREFIX_ID
KEY_TOKENIZER_SUFFIX_ID  = Keys.Tokenizer.SUFFIX_ID
KEY_TOKENIZER_MIDDLE_ID  = Keys.Tokenizer.MIDDLE_ID
KEY_TOKENIZER_EOT_ID     = Keys.Tokenizer.EOT_ID
KEY_TOKENIZER_EOM_ID     = Keys.Tokenizer.EOM_ID
