#include "llama-model.h"

#include <map>

static const std::map<llm_arch, std::map<llm_tensor, std::string>> LLM_TENSOR_NAMES = {
    {
        LLM_ARCH_LLAMA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_DECI,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_LLAMA4,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_SHEXP,  "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,  "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,    "blk.%d.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_BAICHUAN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_FALCON,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_NORM_2,     "blk.%d.attn_norm_2" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_GROK,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_POST_NORM,   "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.%d.layer_output_norm" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.%d.attn_output_norm" },
        },
    },
    {
        LLM_ARCH_GPT2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_GPTJ,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        },
    },
    {
        LLM_ARCH_GPTNEOX,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_MPT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output"},
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_ACT,         "blk.%d.ffn.act" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm"},
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm"},
        },
    },
    {
        LLM_ARCH_STARCODER,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_REFACT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_BERT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_TOKEN_TYPES,     "token_types" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.%d.attn_output_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.%d.layer_output_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_NOMIC_BERT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_TOKEN_TYPES,     "token_types" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.%d.attn_output_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.%d.layer_output_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_JINA_BERT_V2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_TOKEN_TYPES,     "token_types" },
            { LLM_TENSOR_ATTN_NORM_2,     "blk.%d.attn_norm_2" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.%d.attn_output_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.%d.layer_output_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_BLOOM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_STABLELM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
        },
    },
    {
        LLM_ARCH_QWEN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN2MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_QWEN2VL,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN3,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN3MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_QWEN3NEXT,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_POST_NORM,     "blk.%d.post_attention_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_QKV,           "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_GATE,          "blk.%d.attn_gate" },
            { LLM_TENSOR_SSM_CONV1D,         "blk.%d.ssm_conv1d" },
            { LLM_TENSOR_SSM_DT,             "blk.%d.ssm_dt" },
            { LLM_TENSOR_SSM_A_NOSCAN,       "blk.%d.ssm_a" },
            { LLM_TENSOR_SSM_BETA_ALPHA,     "blk.%d.ssm_ba" },
            { LLM_TENSOR_SSM_IN,             "blk.%d.ssm_in" },
            { LLM_TENSOR_SSM_NORM,           "blk.%d.ssm_norm" },
            { LLM_TENSOR_SSM_OUT,            "blk.%d.ssm_out" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_QWEN3VL,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN3VLMOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_PHI2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_PHI3,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ROPE_FACTORS_LONG,  "rope_factors_long" },
            { LLM_TENSOR_ROPE_FACTORS_SHORT, "rope_factors_short" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,           "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_PLAMO,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_CODESHELL,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_ORION,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_INTERNLM2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_MINICPM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
        },
    },
    {
        LLM_ARCH_GEMMA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_GEMMA2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,  "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_POST_NORM,   "blk.%d.post_ffw_norm" },
        },
    },
    {
        LLM_ARCH_GEMMA3,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,  "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_POST_NORM,   "blk.%d.post_ffw_norm" },
        },
    },
    {
        LLM_ARCH_STARCODER2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_MAMBA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_SSM_IN,          "blk.%d.ssm_in" },
            { LLM_TENSOR_SSM_CONV1D,      "blk.%d.ssm_conv1d" },
            { LLM_TENSOR_SSM_X,           "blk.%d.ssm_x" },
            { LLM_TENSOR_SSM_DT,          "blk.%d.ssm_dt" },
            { LLM_TENSOR_SSM_A,           "blk.%d.ssm_a" },
            { LLM_TENSOR_SSM_D,           "blk.%d.ssm_d" },
            { LLM_TENSOR_SSM_OUT,         "blk.%d.ssm_out" },
        },
    },
    {
        LLM_ARCH_XVERSE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_COMMAND_R,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
        },
    },
    {
        LLM_ARCH_DBRX,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.%d.attn_output_norm" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_OLMO,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_OPENELM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_ARCTIC,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_NORM_EXPS,   "blk.%d.ffn_norm_exps" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_DEEPSEEK2,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q_A_NORM,      "blk.%d.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_KV_A_NORM,     "blk.%d.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_A,           "blk.%d.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_B,           "blk.%d.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_A_MQA,      "blk.%d.attn_kv_a_mqa" },
            { LLM_TENSOR_ATTN_KQ_A_MQA,      "blk.%d.attn_kq_a_mqa" },
            { LLM_TENSOR_ATTN_KV_B,          "blk.%d.attn_kv_b" },
            { LLM_TENSOR_ATTN_K_B,           "blk.%d.attn_k_b" },
            { LLM_TENSOR_ATTN_V_B,           "blk.%d.attn_v_b" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
	    { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_CHATGLM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_GLM4,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_ATTN_POST_NORM,  "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_POST_NORM,   "blk.%d.post_ffw_norm" },
        },
    },
    {
        LLM_ARCH_GLM4_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_POST_NORM,     "blk.%d.post_attention_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
            // NextN/MTP tensors - preserved but unused (in final layer, dynamic layer number)
            { LLM_TENSOR_NEXTN_EH_PROJ,      "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_EMBED_TOKENS, "blk.%d.nextn.embed_tokens" },
            { LLM_TENSOR_NEXTN_ENORM,        "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,        "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "blk.%d.nextn.shared_head_head" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
        },
    },
    {
        LLM_ARCH_BITNET,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_SUB_NORM,      "blk.%d.attn_sub_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_SUB_NORM,       "blk.%d.ffn_sub_norm" },
        },
    },
    {
        LLM_ARCH_BITNET_25,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_ATTN_SUB_NORM,   "blk.%d.attn_sub_norm" },
            { LLM_TENSOR_FFN_SUB_NORM,    "blk.%d.ffn_sub_norm" },
        },
    },
    {
        LLM_ARCH_BITNET_B158,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_ATTN_SUB_NORM,   "blk.%d.attn_sub_norm" },
            { LLM_TENSOR_FFN_SUB_NORM,    "blk.%d.ffn_sub_norm" },
        },
    },
    {
        LLM_ARCH_T5,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_DEC_OUTPUT_NORM,      "dec.output_norm" },
            { LLM_TENSOR_DEC_ATTN_NORM,        "dec.blk.%d.attn_norm" },
            { LLM_TENSOR_DEC_ATTN_Q,           "dec.blk.%d.attn_q" },
            { LLM_TENSOR_DEC_ATTN_K,           "dec.blk.%d.attn_k" },
            { LLM_TENSOR_DEC_ATTN_V,           "dec.blk.%d.attn_v" },
            { LLM_TENSOR_DEC_ATTN_OUT,         "dec.blk.%d.attn_o" },
            { LLM_TENSOR_DEC_ATTN_REL_B,       "dec.blk.%d.attn_rel_b" },
            { LLM_TENSOR_DEC_CROSS_ATTN_NORM,  "dec.blk.%d.cross_attn_norm" },
            { LLM_TENSOR_DEC_CROSS_ATTN_Q,     "dec.blk.%d.cross_attn_q" },
            { LLM_TENSOR_DEC_CROSS_ATTN_K,     "dec.blk.%d.cross_attn_k" },
            { LLM_TENSOR_DEC_CROSS_ATTN_V,     "dec.blk.%d.cross_attn_v" },
            { LLM_TENSOR_DEC_CROSS_ATTN_OUT,   "dec.blk.%d.cross_attn_o" },
            { LLM_TENSOR_DEC_CROSS_ATTN_REL_B, "dec.blk.%d.cross_attn_rel_b" },
            { LLM_TENSOR_DEC_FFN_NORM,         "dec.blk.%d.ffn_norm" },
            { LLM_TENSOR_DEC_FFN_GATE,         "dec.blk.%d.ffn_gate" },
            { LLM_TENSOR_DEC_FFN_DOWN,         "dec.blk.%d.ffn_down" },
            { LLM_TENSOR_DEC_FFN_UP,           "dec.blk.%d.ffn_up" },
            { LLM_TENSOR_ENC_OUTPUT_NORM,      "enc.output_norm" },
            { LLM_TENSOR_ENC_ATTN_NORM,        "enc.blk.%d.attn_norm" },
            { LLM_TENSOR_ENC_ATTN_Q,           "enc.blk.%d.attn_q" },
            { LLM_TENSOR_ENC_ATTN_K,           "enc.blk.%d.attn_k" },
            { LLM_TENSOR_ENC_ATTN_V,           "enc.blk.%d.attn_v" },
            { LLM_TENSOR_ENC_ATTN_OUT,         "enc.blk.%d.attn_o" },
            { LLM_TENSOR_ENC_ATTN_REL_B,       "enc.blk.%d.attn_rel_b" },
            { LLM_TENSOR_ENC_FFN_NORM,         "enc.blk.%d.ffn_norm" },
            { LLM_TENSOR_ENC_FFN_GATE,         "enc.blk.%d.ffn_gate" },
            { LLM_TENSOR_ENC_FFN_DOWN,         "enc.blk.%d.ffn_down" },
            { LLM_TENSOR_ENC_FFN_UP,           "enc.blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_T5ENCODER,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_ENC_OUTPUT_NORM,      "enc.output_norm" },
            { LLM_TENSOR_ENC_ATTN_NORM,        "enc.blk.%d.attn_norm" },
            { LLM_TENSOR_ENC_ATTN_Q,           "enc.blk.%d.attn_q" },
            { LLM_TENSOR_ENC_ATTN_K,           "enc.blk.%d.attn_k" },
            { LLM_TENSOR_ENC_ATTN_V,           "enc.blk.%d.attn_v" },
            { LLM_TENSOR_ENC_ATTN_OUT,         "enc.blk.%d.attn_o" },
            { LLM_TENSOR_ENC_ATTN_REL_B,       "enc.blk.%d.attn_rel_b" },
            { LLM_TENSOR_ENC_FFN_NORM,         "enc.blk.%d.ffn_norm" },
            { LLM_TENSOR_ENC_FFN_GATE,         "enc.blk.%d.ffn_gate" },
            { LLM_TENSOR_ENC_FFN_DOWN,         "enc.blk.%d.ffn_down" },
            { LLM_TENSOR_ENC_FFN_UP,           "enc.blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_JAIS,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
        },
    },
    {
        LLM_ARCH_GRANITE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_GRANITE_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_COHERE2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_DOTS1,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
        }
    },
    {
        LLM_ARCH_ERNIE4_5,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_ERNIE4_5_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_HUNYUAN_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_SHEXP,  "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,  "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,    "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_OPENAI_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_POST_NORM,     "blk.%d.post_attention_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_SINKS,         "blk.%d.attn_sinks" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_BAILINGMOE2,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_QKV,           "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_NEXTN_EH_PROJ,      "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_EMBED_TOKENS, "blk.%d.nextn.embed_tokens" },
            { LLM_TENSOR_NEXTN_ENORM,        "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,        "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "blk.%d.nextn.shared_head_head" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
            { LLM_TENSOR_LAYER_OUT_NORM,     "blk.%d.layer_output_norm" },
        },
    },
    {
        LLM_ARCH_MINIMAX_M2,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.%d.attn_k_norm" },
            { LLM_TENSOR_FFN_NORM,           "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_SMOLLM3,
        {
            { LLM_TENSOR_TOKEN_EMBD,     "token_embd"            },
            { LLM_TENSOR_OUTPUT_NORM,    "output_norm"           },
            { LLM_TENSOR_OUTPUT,         "output"                },
            { LLM_TENSOR_ATTN_NORM,      "blk.%d.attn_norm"      },
            { LLM_TENSOR_ATTN_Q,         "blk.%d.attn_q"         },
            { LLM_TENSOR_ATTN_K,         "blk.%d.attn_k"         },
            { LLM_TENSOR_ATTN_V,         "blk.%d.attn_v"         },
            { LLM_TENSOR_ATTN_OUT,       "blk.%d.attn_output"    },
            { LLM_TENSOR_FFN_NORM,       "blk.%d.ffn_norm"       },
            { LLM_TENSOR_FFN_GATE,       "blk.%d.ffn_gate"       },
            { LLM_TENSOR_FFN_DOWN,       "blk.%d.ffn_down"       },
            { LLM_TENSOR_FFN_UP,         "blk.%d.ffn_up"         },
        },
    },
    {
        LLM_ARCH_MISTRAL3,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.%d.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.%d.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.%d.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.%d.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_MIMO2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_SINKS,      "blk.%d.attn_sinks" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
	        { LLM_TENSOR_FFN_EXP_PROBS_B, "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_SEED_OSS,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,  "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
        },
    },
    {
        LLM_ARCH_STEP35,
        {
            {   LLM_TENSOR_TOKEN_EMBD,        "token_embd" },
            {   LLM_TENSOR_OUTPUT_NORM,       "output_norm" },
            {   LLM_TENSOR_OUTPUT,            "output" },
            {   LLM_TENSOR_ROPE_FREQS,        "rope_freqs" },
            {   LLM_TENSOR_ROPE_FACTORS_LONG, "rope_factors_long" },
            {   LLM_TENSOR_ROPE_FACTORS_SHORT,"rope_factors_short" },
            {   LLM_TENSOR_ATTN_NORM,         "blk.%d.attn_norm" },
            {   LLM_TENSOR_ATTN_Q,            "blk.%d.attn_q" },
            {   LLM_TENSOR_ATTN_Q_NORM,       "blk.%d.attn_q_norm" },
            {   LLM_TENSOR_ATTN_K,            "blk.%d.attn_k" },
            {   LLM_TENSOR_ATTN_K_NORM,       "blk.%d.attn_k_norm" },
            {   LLM_TENSOR_ATTN_V,            "blk.%d.attn_v" },
            {   LLM_TENSOR_ATTN_GATE,         "blk.%d.attn_gate" },
            {   LLM_TENSOR_ATTN_OUT,          "blk.%d.attn_output" },
            {   LLM_TENSOR_FFN_NORM,          "blk.%d.ffn_norm" },
            {   LLM_TENSOR_FFN_GATE,          "blk.%d.ffn_gate" },
            {   LLM_TENSOR_FFN_DOWN,          "blk.%d.ffn_down" },
            {   LLM_TENSOR_FFN_UP,            "blk.%d.ffn_up" },
            {   LLM_TENSOR_FFN_GATE_INP,      "blk.%d.ffn_gate_inp" },
            {   LLM_TENSOR_FFN_GATE_EXPS,     "blk.%d.ffn_gate_exps" },
            {   LLM_TENSOR_FFN_DOWN_EXPS,     "blk.%d.ffn_down_exps" },
            {   LLM_TENSOR_FFN_UP_EXPS,       "blk.%d.ffn_up_exps" },
            {   LLM_TENSOR_FFN_GATE_SHEXP,    "blk.%d.ffn_gate_shexp" },
            {   LLM_TENSOR_FFN_DOWN_SHEXP,    "blk.%d.ffn_down_shexp" },
            {   LLM_TENSOR_FFN_UP_SHEXP,      "blk.%d.ffn_up_shexp" },
	        {   LLM_TENSOR_FFN_EXP_PROBS_B,   "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_GLM_DSA,
        {
            { LLM_TENSOR_TOKEN_EMBD,             "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,            "output_norm" },
            { LLM_TENSOR_OUTPUT,                 "output" },
            { LLM_TENSOR_ATTN_NORM,              "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q_A_NORM,          "blk.%d.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_KV_A_NORM,         "blk.%d.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_Q,                 "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_A,               "blk.%d.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_B,               "blk.%d.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_A_MQA,          "blk.%d.attn_kv_a_mqa" },
            { LLM_TENSOR_ATTN_KQ_A_MQA,          "blk.%d.attn_kq_a_mqa" },
            { LLM_TENSOR_ATTN_KV_B,              "blk.%d.attn_kv_b" },
            { LLM_TENSOR_ATTN_K_B,               "blk.%d.attn_k_b" },
            { LLM_TENSOR_ATTN_V_B,               "blk.%d.attn_v_b" },
            { LLM_TENSOR_ATTN_OUT,               "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_NORM,               "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,               "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_UP,                 "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,               "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_GATE_INP,           "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,          "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,          "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,            "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP,     "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,         "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,         "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,           "blk.%d.ffn_up_shexp" },
	        { LLM_TENSOR_FFN_EXP_PROBS_B,        "blk.%d.exp_probs_b" },
            { LLM_TENSOR_INDEXER_K_NORM,         "blk.%d.indexer.k_norm" },
            { LLM_TENSOR_INDEXER_PROJ,           "blk.%d.indexer.proj" },
            { LLM_TENSOR_INDEXER_ATTN_K,         "blk.%d.indexer.attn_k" },
            { LLM_TENSOR_INDEXER_ATTN_Q_B,       "blk.%d.indexer.attn_q_b" },
            { LLM_TENSOR_NEXTN_EH_PROJ,          "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_EMBED_TOKENS,     "blk.%d.nextn.embed_tokens" },
            { LLM_TENSOR_NEXTN_ENORM,            "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,            "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "blk.%d.nextn.shared_head_head" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },

        },
    },
    {
        LLM_ARCH_UNKNOWN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        },
    },
};

std::string LLM_TN::operator()(llm_tensor tensor) const {
    auto& map = LLM_TENSOR_NAMES.at(arch);
    if (auto it = map.find(tensor); it != map.end()) {
        return it->second;
    }
    return "__missing__";
    //if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
    //    return "__missing__";
    //}
    //return LLM_TENSOR_NAMES.at(arch).at(tensor);
}

std::string LLM_TN::operator()(llm_tensor tensor, const std::string & suffix) const {
    if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
        return "__missing__";
    }
    return LLM_TENSOR_NAMES.at(arch).at(tensor) + "." + suffix;
}

std::string LLM_TN::operator()(llm_tensor tensor, int bid) const {
    if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
        return "__missing__";
    }
    return ::format(LLM_TENSOR_NAMES.at(arch).at(tensor).c_str(), bid);
}

std::string LLM_TN::operator()(llm_tensor tensor, const std::string & suffix, int bid) const {
    if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
        return "__missing__";
    }
    return ::format(LLM_TENSOR_NAMES.at(arch).at(tensor).c_str(), bid) + "." + suffix;
}

std::string LLM_TN::operator()(llm_tensor tensor, const std::string & suffix, int bid, int xid) const {
    if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
        return "__missing__";
    }
    return ::format(LLM_TENSOR_NAMES.at(arch).at(tensor).c_str(), bid, xid) + "." + suffix;
}

void llama_model::set_tensor_overrides(const llama_model_params& params) {
    tensor_overrides = params.tensor_buft_overrides && params.tensor_buft_overrides[0].pattern;
}

std::string llama_model_ftype_name(llama_ftype ftype) {
    if (ftype & LLAMA_FTYPE_GUESSED) {
        return llama_model_ftype_name((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) + " (guessed)";
    }

    switch (ftype) {
        case LLAMA_FTYPE_ALL_F32:         return "all F32";
        case LLAMA_FTYPE_MOSTLY_F16:      return "F16";
        case LLAMA_FTYPE_MOSTLY_BF16:     return "BF16";
        case LLAMA_FTYPE_MOSTLY_BF16_R16: return "BF16_R16";
        case LLAMA_FTYPE_MOSTLY_Q4_0:     return "Q4_0";
        case LLAMA_FTYPE_MOSTLY_Q4_1:     return "Q4_1";
        case LLAMA_FTYPE_MOSTLY_Q5_0:     return "Q5_0";
        case LLAMA_FTYPE_MOSTLY_Q5_1:     return "Q5_1";
        case LLAMA_FTYPE_MOSTLY_Q6_0:     return "Q6_0";
        case LLAMA_FTYPE_MOSTLY_Q8_0:     return "Q8_0";
        case LLAMA_FTYPE_MOSTLY_Q8_KV:    return "Q8_KV";
        case LLAMA_FTYPE_MOSTLY_Q2_K:     return "Q2_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q2_K_R4:  return "Q2_K_R4";
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:   return "Q2_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:   return "Q3_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:   return "Q3_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:   return "Q3_K - Large";
        case LLAMA_FTYPE_MOSTLY_Q3_K_R4:  return "Q3_K_R4";
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:   return "Q4_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q4_K_R4:  return "Q4_K_R4";
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:   return "Q4_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:   return "Q5_K - Small";
        case LLAMA_FTYPE_MOSTLY_Q5_K_R4:  return "Q5_K_R4";
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:   return "Q5_K - Medium";
        case LLAMA_FTYPE_MOSTLY_Q6_K:     return "Q6_K";
        case LLAMA_FTYPE_MOSTLY_Q6_K_R4:  return "Q6_K_R4";
        case LLAMA_FTYPE_MOSTLY_Q8_K_R8:  return "Q8_K_R8";
        case LLAMA_FTYPE_MOSTLY_Q8_KV_R8: return "Q8_KV_R8";
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:  return "IQ2_XXS - 2.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4:return "IQ2_XXS_R4 - 2.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:   return "IQ2_XS - 2.3125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_XS_R4:return "IQ2_XS_R4 - 2.3125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_KS:   return "IQ2_KS - 2.1875 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_S:    return "IQ2_S - 2.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_M:    return "IQ2_M - 2.7 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_M_R4: return "IQ2_M_R4 - 2.7 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:   return "IQ3_XS - 3.3 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:  return "IQ3_XXS - 3.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_KT:   return "IQ1_KT - 1.75 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_KT:   return "IQ2_KT - 2.125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_KT:   return "IQ3_KT - 3.125 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_KT:   return "IQ4_KT - 4.0 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4: return "IQ3_XXS_R4 - 3.0625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_S:    return "IQ1_S - 1.5625 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_S_R4: return "IQ1_S_R4 - 1.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_M_R4: return "IQ1_M_R4 - 1.75 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_M:    return "IQ1_M - 1.75 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:   return "IQ4_NL - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_NL_R4:return "IQ4_NL_R4 - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_XS_R8:return "IQ4_XS_R8 - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_Q4_0_R8:  return "Q4_0_R8 - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_Q5_0_R4:  return "Q5_0_R4 - 5.5 bpw";
        case LLAMA_FTYPE_MOSTLY_Q6_0_R4:  return "Q6_0_R4 - 6.5 bpw";
        case LLAMA_FTYPE_MOSTLY_Q8_0_R8:  return "Q8_0_R8 - 8.5 bpw";
        case LLAMA_FTYPE_MOSTLY_MXFP4:    return "MXFP4 - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:   return "IQ4_XS - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_KS:   return "IQ4_KS - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_KS_R4:return "IQ4_KS_R4 - 4.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ5_KS_R4:return "IQ5_KS_R4 - 5.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_KSS:  return "IQ4_KSS - 4.0 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ5_KS:   return "IQ5_KS - 5.25 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_K:    return "IQ2_K - 2.375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_K_R4: return "IQ2_K_R4 - 2.375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_KS:   return "IQ3_KS - 3.1875 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ2_KL:   return "IQ2_KL - 2.6875 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_K:    return "IQ3_K - 3.4325 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_K_R4: return "IQ3_K_R4 - 3.4325 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_KL:   return "IQ3_KL - 4 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_K:    return "IQ4_K - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ4_K_R4: return "IQ4_K_R4 - 4.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ5_K:    return "IQ5_K - 5.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ5_K_R4: return "IQ5_K_R4 - 5.5 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ6_K:    return "IQ6_K - 6.6 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ1_BN:   return "IQ1_BN - 1.625 bpw Bitnet";
        case LLAMA_FTYPE_MOSTLY_IQ2_BN:   return "IQ2_BN - 2.00 bpw Bitnet";
        case LLAMA_FTYPE_MOSTLY_IQ2_BN_R4:return "IQ2_BN_R4 - 2.00 bpw Bitnet";
        case LLAMA_FTYPE_MOSTLY_IQ3_S:    return "IQ3_S - 3.4375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_S_R4: return "IQ3_S_R4 - 3.4375 bpw";
        case LLAMA_FTYPE_MOSTLY_IQ3_M:    return "IQ3_S mix - 3.66 bpw";
        case LLAMA_FTYPE_MOSTLY_Q4_0_4_4: return "Q4_0_4_4";
        case LLAMA_FTYPE_MOSTLY_Q4_0_4_8: return "Q4_0_4_8";
        case LLAMA_FTYPE_MOSTLY_Q4_0_8_8: return "Q4_0_8_8";

        default: return "unknown, may not work";
    }
}

const char * llama_model_type_name(e_model type) {
    switch (type) {
        case MODEL_14M:           return "14M";
        case MODEL_17M:           return "17M";
        case MODEL_22M:           return "22M";
        case MODEL_33M:           return "33M";
        case MODEL_60M:           return "60M";
        case MODEL_70M:           return "70M";
        case MODEL_80M:           return "80M";
        case MODEL_109M:          return "109M";
        case MODEL_137M:          return "137M";
        case MODEL_140M:          return "140M";
        case MODEL_160M:          return "160M";
        case MODEL_190M:          return "190M";
        case MODEL_220M:          return "220M";
        case MODEL_250M:          return "250M";
        case MODEL_256M:          return "256M";
        case MODEL_270M:          return "270M";
        case MODEL_335M:          return "335M";
        case MODEL_350M:          return "350M";
        case MODEL_360M:          return "360M";
        case MODEL_410M:          return "410M";
        case MODEL_450M:          return "450M";
        case MODEL_475M:          return "475M";
        case MODEL_558M:          return "558M";
        case MODEL_700M:          return "700M";
        case MODEL_770M:          return "770M";
        case MODEL_780M:          return "780M";
        case MODEL_950M:          return "950M";
        case MODEL_0_3B:          return "0.3B";
        case MODEL_0_5B:          return "0.5B";
        case MODEL_0_6B:          return "0.6B";
        case MODEL_1B:            return "1B";
        case MODEL_1_2B:          return "1.2B";
        case MODEL_1_3B:          return "1.3B";
        case MODEL_1_4B:          return "1.4B";
        case MODEL_1_5B:          return "1.5B";
        case MODEL_1_6B:          return "1.6B";
        case MODEL_1_7B:          return "1.7B";
        case MODEL_1_8B:          return "1.8B";
        case MODEL_2B:            return "2B";
        case MODEL_2_6B:          return "2.6B";
        case MODEL_2_8B:          return "2.8B";
        case MODEL_2_9B:          return "2.9B";
        case MODEL_3B:            return "3B";
        case MODEL_4B:            return "4B";
        case MODEL_6B:            return "6B";
        case MODEL_6_9B:          return "6.9B";
        case MODEL_7B:            return "7B";
        case MODEL_8B:            return "8B";
        case MODEL_9B:            return "9B";
        case MODEL_11B:           return "11B";
        case MODEL_12B:           return "12B";
        case MODEL_13B:           return "13B";
        case MODEL_14B:           return "14B";
        case MODEL_15B:           return "15B";
        case MODEL_16B:           return "16B";
        case MODEL_20B:           return "20B";
        case MODEL_27B:           return "27B";
        case MODEL_30B:           return "30B";
        case MODEL_32B:           return "32B";
        case MODEL_34B:           return "34B";
        case MODEL_35B:           return "35B";
        case MODEL_36B:           return "36B";
        case MODEL_40B:           return "40B";
        case MODEL_65B:           return "65B";
        case MODEL_70B:           return "70B";
        case MODEL_120B:          return "120B";
        case MODEL_142B:          return "142B";
        case MODEL_236B:          return "236B";
        case MODEL_290B:          return "290B";
        case MODEL_314B:          return "314B";
        case MODEL_405B:          return "405B";
        case MODEL_671B:          return "671B";
        case MODEL_SMALL:         return "0.1B";
        case MODEL_MEDIUM:        return "0.4B";
        case MODEL_LARGE:         return "0.8B";
        case MODEL_XL:            return "1.5B";
        case MODEL_A1_7B:         return "A1.7B";
        case MODEL_A2_7B:         return "A2.7B";
        case MODEL_8x7B:          return "8x7B";
        case MODEL_8x22B:         return "8x22B";
        case MODEL_16x12B:        return "16x12B";
        case MODEL_16x3_8B:       return "16x3.8B";
        case MODEL_10B_128x3_66B: return "10B+128x3.66B";
        case MODEL_57B_A14B:      return "57B.A14B";
        case MODEL_17B_16E:       return "17Bx16E (Scout)";
        case MODEL_17B_128E:      return "17Bx128E (Maverick)";
        case MODEL_A13B:          return "A13B";
        case MODEL_7B_A1B:        return "7B.A1B";
        case MODEL_8B_A1B:        return "8B.A1B";
        case MODEL_16B_A1B:       return "16B.A1B";
        case MODEL_21B_A3B:       return "21B.A3B";
        case MODEL_30B_A3B:       return "30B.A3B";
        case MODEL_80B_A3B:       return "80B.A3B";
        case MODEL_80B_A13B:      return "80B.A13B";
        case MODEL_100B_A6B:      return "100B.A6B";
        case MODEL_106B_A12B:     return "106B.A12B";
        case MODEL_230B_A10B:     return "230B.A10B";
        case MODEL_235B_A22B:     return "235B.A22B";
        case MODEL_310B_A15B:     return "310B.A15B";
        case MODEL_300B_A47B:     return "300B.A47B";
        case MODEL_355B_A32B:     return "355B.A32B";
        case MODEL_744B_A40B:     return "744B.A40B";
        case MODEL_E2B:           return "E2B";
        case MODEL_E4B:           return "E4B";
        default:                     return "?B";
    }
}
