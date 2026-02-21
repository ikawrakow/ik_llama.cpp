#include "llama-arch.h"
#include "llama-impl.h"

#include <map>

static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,           "llama"        },
    { LLM_ARCH_LLAMA4,          "llama4"       },
    { LLM_ARCH_DECI,            "deci"         },
    { LLM_ARCH_FALCON,          "falcon"       },
    { LLM_ARCH_GROK,            "grok"         },
    { LLM_ARCH_GPT2,            "gpt2"         },
    { LLM_ARCH_GPTJ,            "gptj"         },
    { LLM_ARCH_GPTNEOX,         "gptneox"      },
    { LLM_ARCH_MPT,             "mpt"          },
    { LLM_ARCH_BAICHUAN,        "baichuan"     },
    { LLM_ARCH_STARCODER,       "starcoder"    },
    { LLM_ARCH_REFACT,          "refact"       },
    { LLM_ARCH_BERT,            "bert"         },
    { LLM_ARCH_NOMIC_BERT,      "nomic-bert"   },
    { LLM_ARCH_JINA_BERT_V2,    "jina-bert-v2" },
    { LLM_ARCH_BLOOM,           "bloom"        },
    { LLM_ARCH_STABLELM,        "stablelm"     },
    { LLM_ARCH_QWEN,            "qwen"         },
    { LLM_ARCH_QWEN2,           "qwen2"        },
    { LLM_ARCH_QWEN2MOE,        "qwen2moe"     },
    { LLM_ARCH_QWEN2VL,         "qwen2vl"      },
    { LLM_ARCH_QWEN3,           "qwen3"        },
    { LLM_ARCH_QWEN3MOE,        "qwen3moe"     },
    { LLM_ARCH_QWEN3NEXT,       "qwen3next"    },
    { LLM_ARCH_QWEN3VL,         "qwen3vl"      },
    { LLM_ARCH_QWEN3VLMOE,      "qwen3vlmoe"   },
    { LLM_ARCH_QWEN35MOE,       "qwen35moe"    },
    { LLM_ARCH_PHI2,            "phi2"         },
    { LLM_ARCH_PHI3,            "phi3"         },
    { LLM_ARCH_PLAMO,           "plamo"        },
    { LLM_ARCH_CODESHELL,       "codeshell"    },
    { LLM_ARCH_ORION,           "orion"        },
    { LLM_ARCH_INTERNLM2,       "internlm2"    },
    { LLM_ARCH_MINICPM,         "minicpm"      },
    { LLM_ARCH_GEMMA,           "gemma"        },
    { LLM_ARCH_GEMMA2,          "gemma2"       },
    { LLM_ARCH_GEMMA3,          "gemma3"       },
    { LLM_ARCH_STARCODER2,      "starcoder2"   },
    { LLM_ARCH_MAMBA,           "mamba"        },
    { LLM_ARCH_XVERSE,          "xverse"       },
    { LLM_ARCH_COMMAND_R,       "command-r"    },
    { LLM_ARCH_DBRX,            "dbrx"         },
    { LLM_ARCH_OLMO,            "olmo"         },
    { LLM_ARCH_OPENELM,         "openelm"      },
    { LLM_ARCH_ARCTIC,          "arctic"       },
    { LLM_ARCH_DEEPSEEK2,       "deepseek2"    },
    { LLM_ARCH_CHATGLM,         "chatglm"      },
    { LLM_ARCH_GLM4,            "glm4"         },
    { LLM_ARCH_GLM4_MOE,        "glm4moe"      },
    { LLM_ARCH_BITNET,          "bitnet"       },
    { LLM_ARCH_BITNET_25,       "bitnet-25"    },
    { LLM_ARCH_BITNET_B158,     "bitnet-b1.58" },
    { LLM_ARCH_T5,              "t5"           },
    { LLM_ARCH_T5ENCODER,       "t5encoder"    },
    { LLM_ARCH_JAIS,            "jais"         },
    { LLM_ARCH_GRANITE,         "granite"      },
    { LLM_ARCH_GRANITE_MOE,     "granitemoe"   },
    { LLM_ARCH_COHERE2,         "cohere2"      },
    { LLM_ARCH_DOTS1,           "dots1"        },
    { LLM_ARCH_ERNIE4_5,        "ernie4_5"     },
    { LLM_ARCH_ERNIE4_5_MOE,    "ernie4_5-moe" },
    { LLM_ARCH_HUNYUAN_MOE,     "hunyuan-moe"  },
    { LLM_ARCH_OPENAI_MOE,      "gpt-oss"      },
    { LLM_ARCH_BAILINGMOE2,     "bailingmoe2"  },
    { LLM_ARCH_MINIMAX_M2,      "minimax-m2"   },
    { LLM_ARCH_SMOLLM3,         "smollm3"      },
    { LLM_ARCH_MISTRAL3,        "mistral3"     },
    { LLM_ARCH_MIMO2,           "mimo2"        },
    { LLM_ARCH_SEED_OSS,        "seed_oss"     },
    { LLM_ARCH_STEP35,          "step35"       },
    { LLM_ARCH_GLM_DSA,         "glm-dsa"      },
    { LLM_ARCH_UNKNOWN,         "(unknown)"    },
};

llm_arch llm_arch_from_string(const std::string & name) {
    for (const auto & kv : LLM_ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    return LLM_ARCH_UNKNOWN;
}

static const std::map<llm_kv, const char *> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_TYPE,                  "general.type"                          },
    { LLM_KV_GENERAL_ARCHITECTURE,          "general.architecture"                  },
    { LLM_KV_GENERAL_QUANTIZATION_VERSION,  "general.quantization_version"          },
    { LLM_KV_GENERAL_ALIGNMENT,             "general.alignment"                     },
    { LLM_KV_GENERAL_NAME,                  "general.name"                          },
    { LLM_KV_GENERAL_AUTHOR,                "general.author"                        },
    { LLM_KV_GENERAL_VERSION,               "general.version"                       },
    { LLM_KV_GENERAL_URL,                   "general.url"                           },
    { LLM_KV_GENERAL_DESCRIPTION,           "general.description"                   },
    { LLM_KV_GENERAL_LICENSE,               "general.license"                       },
    { LLM_KV_GENERAL_SOURCE_URL,            "general.source.url"                    },
    { LLM_KV_GENERAL_SOURCE_HF_REPO,        "general.source.huggingface.repository" },

    { LLM_KV_VOCAB_SIZE,                        "%s.vocab_size"                        },
    { LLM_KV_CONTEXT_LENGTH,                    "%s.context_length"                    },
    { LLM_KV_EMBEDDING_LENGTH,                  "%s.embedding_length"                  },
    { LLM_KV_BLOCK_COUNT,                       "%s.block_count"                       },
    { LLM_KV_LEADING_DENSE_BLOCK_COUNT,         "%s.leading_dense_block_count"         },
    { LLM_KV_FEED_FORWARD_LENGTH,               "%s.feed_forward_length"               },
    { LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        "%s.expert_feed_forward_length"        },
    { LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, "%s.expert_shared_feed_forward_length" },
    { LLM_KV_USE_PARALLEL_RESIDUAL,             "%s.use_parallel_residual"             },
    { LLM_KV_TENSOR_DATA_LAYOUT,                "%s.tensor_data_layout"                },
    { LLM_KV_EXPERT_COUNT,                      "%s.expert_count"                      },
    { LLM_KV_EXPERT_USED_COUNT,                 "%s.expert_used_count"                 },
    { LLM_KV_EXPERT_SHARED_COUNT,               "%s.expert_shared_count"               },
    { LLM_KV_EXPERT_GROUP_COUNT,                "%s.expert_group_count"                },
    { LLM_KV_EXPERT_GROUP_USED_COUNT,           "%s.expert_group_used_count"           },
    { LLM_KV_EXPERT_WEIGHTS_SCALE,              "%s.expert_weights_scale"              },
    { LLM_KV_EXPERT_WEIGHTS_NORM,               "%s.expert_weights_norm"               },
    { LLM_KV_EXPERT_GATING_FUNC,                "%s.expert_gating_func"                },
    { LLM_KV_NEXTN_PREDICT_LAYERS,              "%s.nextn_predict_layers"              },
    { LLM_KV_NUM_DEEPSTACK_LAYERS,              "%s.n_deepstack_layers"                },
    { LLM_KV_POOLING_TYPE,                      "%s.pooling_type"                      },
    { LLM_KV_LOGIT_SCALE,                       "%s.logit_scale"                       },
    { LLM_KV_DECODER_START_TOKEN_ID,            "%s.decoder_start_token_id"            },
    { LLM_KV_ATTN_LOGIT_SOFTCAPPING,            "%s.attn_logit_softcapping"            },
    { LLM_KV_ROUTER_LOGIT_SOFTCAPPING,          "%s.router_logit_softcapping"          },
    { LLM_KV_FINAL_LOGIT_SOFTCAPPING,           "%s.final_logit_softcapping"           },
    { LLM_KV_RESIDUAL_SCALE,                    "%s.residual_scale"                    },
    { LLM_KV_EMBEDDING_SCALE,                   "%s.embedding_scale"                   },
    { LLM_KV_TOKEN_SHIFT_COUNT,                 "%s.token_shift_count"                 },
    { LLM_KV_INTERLEAVE_MOE_LAYER_STEP,         "%s.interleave_moe_layer_step"         },
    { LLM_KV_SWIGLU_LIMITS,                     "%s.swiglu_limits"                     },
    { LLM_KV_SWIGLU_LIMITS_SHARED,              "%s.swiglu_limits_shared"              },
    { LLM_KV_SWIGLU_CLAMP_EXP,                  "%s.swiglu_clamp_exp"                  },
    { LLM_KV_SWIGLU_CLAMP_SHEXP,                "%s.swiglu_clamp_shexp"                },

    { LLM_KV_ATTENTION_HEAD_COUNT,             "%s.attention.head_count"             },
    { LLM_KV_ATTENTION_HEAD_COUNT_KV,          "%s.attention.head_count_kv"          },
    { LLM_KV_ATTENTION_MAX_ALIBI_BIAS,         "%s.attention.max_alibi_bias"         },
    { LLM_KV_ATTENTION_CLAMP_KQV,              "%s.attention.clamp_kqv"              },
    { LLM_KV_ATTENTION_KEY_LENGTH,             "%s.attention.key_length"             },
    { LLM_KV_ATTENTION_VALUE_LENGTH,           "%s.attention.value_length"           },
    { LLM_KV_ATTENTION_LAYERNORM_EPS,          "%s.attention.layer_norm_epsilon"     },
    { LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      "%s.attention.layer_norm_rms_epsilon" },
    { LLM_KV_ATTENTION_CAUSAL,                 "%s.attention.causal"                 },
    { LLM_KV_ATTENTION_Q_LORA_RANK,            "%s.attention.q_lora_rank"            },
    { LLM_KV_ATTENTION_KV_LORA_RANK,           "%s.attention.kv_lora_rank"           },
    { LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, "%s.attention.relative_buckets_count" },
    { LLM_KV_ATTENTION_SLIDING_WINDOW,         "%s.attention.sliding_window"         },
    { LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, "%s.attention.sliding_window_pattern" },
    { LLM_KV_ATTENTION_SCALE,                  "%s.attention.scale"                  },
    { LLM_KV_ATTENTION_OUTPUT_SCALE,           "%s.attention.output_scale"           },
    { LLM_KV_ATTENTION_TEMPERATURE_LENGTH,     "%s.attention.temperature_length"     },
    { LLM_KV_ATTENTION_TEMPERATURE_SCALE,      "%s.attention.temperature_scale"      },
    { LLM_KV_ATTENTION_KEY_LENGTH_MLA,         "%s.attention.key_length_mla"         },
    { LLM_KV_ATTENTION_VALUE_LENGTH_MLA,       "%s.attention.value_length_mla"       },
    { LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,     "%s.attention.indexer.head_count"     },
    { LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,     "%s.attention.indexer.key_length"     },
    { LLM_KV_ATTENTION_INDEXER_TOP_K,          "%s.attention.indexer.top_k"          },
    { LLM_KV_FULL_ATTENTION_INTERVAL,          "%s.full_attention_interval"          },


    { LLM_KV_ROPE_DIMENSION_COUNT,          "%s.rope.dimension_count"                 },
    { LLM_KV_ROPE_DIMENSION_COUNT_PER_LAYER,"%s.rope.dimension_count_per_layer"       },
    { LLM_KV_ROPE_DIMENSION_SECTIONS,       "%s.rope.dimension_sections"              },
    { LLM_KV_ROPE_FREQ_BASE,                "%s.rope.freq_base"                       },
    { LLM_KV_ROPE_FREQ_BASE_PER_LAYER,      "%s.rope.freq_base_per_layer"             },
    { LLM_KV_ROPE_FREQ_BASE_SWA,            "%s.rope.freq_base_swa"                   },
    { LLM_KV_ROPE_SCALE_LINEAR,             "%s.rope.scale_linear"                    },
    { LLM_KV_ROPE_SCALING_TYPE,             "%s.rope.scaling.type"                    },
    { LLM_KV_ROPE_SCALING_FACTOR,           "%s.rope.scaling.factor"                  },
    { LLM_KV_ROPE_SCALING_ATTN_FACTOR,      "%s.rope.scaling.attn_factor"             },
    { LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,     "%s.rope.scaling.original_context_length" },
    { LLM_KV_ROPE_SCALING_FINETUNED,        "%s.rope.scaling.finetuned"               },
    { LLM_KV_ROPE_SCALING_YARN_LOG_MUL,     "%s.rope.scaling.yarn_log_multiplier"     },
    { LLM_KV_ROPE_SCALING_YARN_EXT_FACTOR,  "%s.rope.scaling.yarn_ext_factor"         },
    { LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR, "%s.rope.scaling.yarn_attn_factor"        },
    { LLM_KV_ROPE_SCALING_YARN_BETA_FAST,   "%s.rope.scaling.yarn_beta_fast"          },
    { LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   "%s.rope.scaling.yarn_beta_slow"          },

    { LLM_KV_SPLIT_NO,                      "split.no"            },
    { LLM_KV_SPLIT_COUNT,                   "split.count"         },
    { LLM_KV_SPLIT_TENSORS_COUNT,           "split.tensors.count" },

    { LLM_KV_SSM_CONV_KERNEL,               "%s.ssm.conv_kernel"    },
    { LLM_KV_SSM_INNER_SIZE,                "%s.ssm.inner_size"     },
    { LLM_KV_SSM_STATE_SIZE,                "%s.ssm.state_size"     },
    { LLM_KV_SSM_TIME_STEP_RANK,            "%s.ssm.time_step_rank" },
    { LLM_KV_SSM_GROUP_COUNT,               "%s.ssm.group_count"    },

    { LLM_KV_TOKENIZER_MODEL,                "tokenizer.ggml.model"                    },
    { LLM_KV_TOKENIZER_PRE,                  "tokenizer.ggml.pre"                      },
    { LLM_KV_TOKENIZER_LIST,                 "tokenizer.ggml.tokens"                   },
    { LLM_KV_TOKENIZER_TOKEN_TYPE,           "tokenizer.ggml.token_type"               },
    { LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,     "tokenizer.ggml.token_type_count"         },
    { LLM_KV_TOKENIZER_SCORES,               "tokenizer.ggml.scores"                   },
    { LLM_KV_TOKENIZER_MERGES,               "tokenizer.ggml.merges"                   },
    { LLM_KV_TOKENIZER_BOS_ID,               "tokenizer.ggml.bos_token_id"             },
    { LLM_KV_TOKENIZER_EOS_ID,               "tokenizer.ggml.eos_token_id"             },
    { LLM_KV_TOKENIZER_UNK_ID,               "tokenizer.ggml.unknown_token_id"         },
    { LLM_KV_TOKENIZER_SEP_ID,               "tokenizer.ggml.seperator_token_id"       },
    { LLM_KV_TOKENIZER_PAD_ID,               "tokenizer.ggml.padding_token_id"         },
    { LLM_KV_TOKENIZER_CLS_ID,               "tokenizer.ggml.cls_token_id"             },
    { LLM_KV_TOKENIZER_MASK_ID,              "tokenizer.ggml.mask_token_id"            },
    { LLM_KV_TOKENIZER_ADD_BOS,              "tokenizer.ggml.add_bos_token"            },
    { LLM_KV_TOKENIZER_ADD_EOS,              "tokenizer.ggml.add_eos_token"            },
    { LLM_KV_TOKENIZER_ADD_SEP,              "tokenizer.ggml.add_sep_token"            },
    { LLM_KV_TOKENIZER_ADD_PREFIX,           "tokenizer.ggml.add_space_prefix"         },
    { LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,      "tokenizer.ggml.remove_extra_whitespaces" },
    { LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP, "tokenizer.ggml.precompiled_charsmap"     },
    { LLM_KV_TOKENIZER_HF_JSON,              "tokenizer.huggingface.json"              },
    { LLM_KV_TOKENIZER_RWKV,                 "tokenizer.rwkv.world"                    },
    { LLM_KV_TOKENIZER_CHAT_TEMPLATE,        "tokenizer.chat_template"                 },
    { LLM_KV_TOKENIZER_CHAT_TEMPLATE_N,      "tokenizer.chat_template.%s"              },
    { LLM_KV_TOKENIZER_FIM_PRE_ID,           "tokenizer.ggml.fim_pre_token_id"         },
    { LLM_KV_TOKENIZER_FIM_SUF_ID,           "tokenizer.ggml.fim_suf_token_id"         },
    { LLM_KV_TOKENIZER_FIM_MID_ID,           "tokenizer.ggml.fim_mid_token_id"         },
    { LLM_KV_TOKENIZER_FIM_PAD_ID,           "tokenizer.ggml.fim_pad_token_id"         },
    { LLM_KV_TOKENIZER_FIM_REP_ID,           "tokenizer.ggml.fim_rep_token_id"         },
    { LLM_KV_TOKENIZER_FIM_SEP_ID,           "tokenizer.ggml.fim_sep_token_id"         },

    { LLM_KV_TOKENIZER_PREFIX_ID,            "tokenizer.ggml.prefix_token_id"          },
    { LLM_KV_TOKENIZER_SUFFIX_ID,            "tokenizer.ggml.suffix_token_id"          },
    { LLM_KV_TOKENIZER_MIDDLE_ID,            "tokenizer.ggml.middle_token_id"          },
    { LLM_KV_TOKENIZER_EOT_ID,               "tokenizer.ggml.eot_token_id"             },
    { LLM_KV_TOKENIZER_EOM_ID,               "tokenizer.ggml.eom_token_id"             },

    { LLM_KV_ADAPTER_TYPE,                  "adapter.type"       },
    { LLM_KV_ADAPTER_LORA_ALPHA,            "adapter.lora.alpha" },
};

LLM_KV::LLM_KV(llm_arch arch, const char* suffix) : arch(arch), suffix(suffix) {}

std::string LLM_KV::operator()(llm_kv kv) const {
    return suffix ? ::format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch), suffix)
        : ::format(LLM_KV_NAMES.at(kv), LLM_ARCH_NAMES.at(arch));
}

const char * llama_model_arch_name(llm_arch arch) {
    auto it = LLM_ARCH_NAMES.find(arch);
    if (it == LLM_ARCH_NAMES.end()) {
        return "unknown";
    }
    return it->second;
}

bool llm_arch_is_recurrent(const llm_arch & arch) {
    switch (arch) {
    case LLM_ARCH_MAMBA:
        return true;
    default:
        return false;
    }
}

bool llm_arch_is_hybrid(const llm_arch & arch) {
    switch (arch) {
    case LLM_ARCH_QWEN3NEXT:
    case LLM_ARCH_QWEN3MOE: 
        return true;
    default:
        return false;
    }
}

