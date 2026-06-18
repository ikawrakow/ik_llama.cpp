#include "llama-model.h"
#include "llama-cparams.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <map>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstring>

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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
        },
    },
    {
        LLM_ARCH_MELLUM,
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_QWEN35MOE,
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
            { LLM_TENSOR_SSM_BETA,           "blk.%d.ssm_beta" },
            { LLM_TENSOR_SSM_ALPHA,          "blk.%d.ssm_alpha" },
            //{ LLM_TENSOR_SSM_IN,             "blk.%d.ssm_in" },
            { LLM_TENSOR_SSM_NORM,           "blk.%d.ssm_norm" },
            { LLM_TENSOR_SSM_OUT,            "blk.%d.ssm_out" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
            { LLM_TENSOR_NEXTN_EH_PROJ,      "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_ENORM,        "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,        "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
        },
    },
    {
        LLM_ARCH_QWEN35,
        {
            { LLM_TENSOR_TOKEN_EMBD,             "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,            "output_norm" },
            { LLM_TENSOR_OUTPUT,                 "output" },
            { LLM_TENSOR_ATTN_NORM,              "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_POST_NORM,         "blk.%d.post_attention_norm" },
            { LLM_TENSOR_ATTN_Q,                 "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,            "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,                 "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,            "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,                 "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,               "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_QKV,               "blk.%d.attn_qkv" },
            { LLM_TENSOR_ATTN_GATE,              "blk.%d.attn_gate" },
            { LLM_TENSOR_SSM_CONV1D,             "blk.%d.ssm_conv1d" },
            { LLM_TENSOR_SSM_DT,                 "blk.%d.ssm_dt" },
            { LLM_TENSOR_SSM_A_NOSCAN,           "blk.%d.ssm_a" },
            { LLM_TENSOR_SSM_BETA,               "blk.%d.ssm_beta" },
            { LLM_TENSOR_SSM_ALPHA,              "blk.%d.ssm_alpha" },
            { LLM_TENSOR_SSM_NORM,               "blk.%d.ssm_norm" },
            { LLM_TENSOR_SSM_OUT,                "blk.%d.ssm_out" },
            { LLM_TENSOR_FFN_GATE,               "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,               "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,                 "blk.%d.ffn_up" },
            { LLM_TENSOR_NEXTN_EH_PROJ,          "blk.%d.nextn.eh_proj" },
            { LLM_TENSOR_NEXTN_ENORM,            "blk.%d.nextn.enorm" },
            { LLM_TENSOR_NEXTN_HNORM,            "blk.%d.nextn.hnorm" },
            { LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "blk.%d.nextn.shared_head_norm" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
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
        LLM_ARCH_GEMMA4,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,          "output_norm" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_ROPE_FREQS,           "rope_freqs" },
            { LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "per_layer_token_embd" },
            { LLM_TENSOR_PER_LAYER_MODEL_PROJ, "per_layer_model_proj" },
            { LLM_TENSOR_PER_LAYER_PROJ_NORM,  "per_layer_proj_norm" },
            { LLM_TENSOR_ATTN_NORM,            "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,               "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,          "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,               "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,          "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,               "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,             "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,       "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,             "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,             "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,             "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,               "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,     "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,        "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_GATE_INP,         "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_POST_NORM,        "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_FFN_POST_NORM_1,      "blk.%d.post_ffw_norm_1" },
            { LLM_TENSOR_FFN_POST_NORM_2,      "blk.%d.post_ffw_norm_2" },
            { LLM_TENSOR_FFN_PRE_NORM_2,       "blk.%d.pre_ffw_norm_2" },
            { LLM_TENSOR_LAYER_OUT_SCALE,      "blk.%d.layer_output_scale" },
            { LLM_TENSOR_PER_LAYER_INP_GATE,   "blk.%d.inp_gate" },
            { LLM_TENSOR_PER_LAYER_PROJ,       "blk.%d.proj" },
            { LLM_TENSOR_PER_LAYER_POST_NORM,  "blk.%d.post_norm" },
        },
    },
    {
        LLM_ARCH_GEMMA4_MTP,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,          "output_norm" },
            { LLM_TENSOR_ATTN_NORM,            "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,               "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,          "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_OUT,             "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,       "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,             "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,             "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,             "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,               "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_POST_NORM,        "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_LAYER_OUT_SCALE,      "blk.%d.layer_output_scale" },
            { LLM_TENSOR_MTP_PRE_PROJ,         "mtp_pre_proj" },
            { LLM_TENSOR_MTP_POST_PROJ,        "mtp_post_proj" },
            { LLM_TENSOR_MTP_TOKEN_ORDERING,   "mtp_token_ordering" },
            { LLM_TENSOR_MTP_CENTROIDS,        "mtp_centroids" },
        },
    },
    {
        LLM_ARCH_DFLASH_DRAFT,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,          "output_norm" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_ATTN_NORM,            "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,               "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,          "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,               "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,          "blk.%d.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,               "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,             "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,       "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_GATE,             "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,             "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,               "blk.%d.ffn_up" },
            { LLM_TENSOR_DFLASH_FC,            "dflash_fc" },
            { LLM_TENSOR_DFLASH_HIDDEN_NORM,   "dflash_hidden_norm" },
        },
    },
    {
        LLM_ARCH_GEMMA4_ASSISTANT,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,          "output_norm" },
            { LLM_TENSOR_ATTN_NORM,            "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,               "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,          "blk.%d.attn_q_norm" },
            { LLM_TENSOR_ATTN_OUT,             "blk.%d.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,       "blk.%d.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,             "blk.%d.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,             "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,             "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,               "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_POST_NORM,        "blk.%d.post_ffw_norm" },
            { LLM_TENSOR_LAYER_OUT_SCALE,      "blk.%d.layer_output_scale" },
            { LLM_TENSOR_MTP_PRE_PROJ,         "mtp_pre_proj" },
            { LLM_TENSOR_MTP_POST_PROJ,        "mtp_post_proj" },
            { LLM_TENSOR_MTP_TOKEN_ORDERING,   "mtp_token_ordering" },
            { LLM_TENSOR_MTP_CENTROIDS,        "mtp_centroids" },
            { LLM_TENSOR_ROPE_FREQS,           "rope_freqs" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.%d.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
	        { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_MISTRAL4,
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
        LLM_ARCH_COHERE2_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.%d.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.%d.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.%d.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.%d.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.%d.attn_output" },
            { LLM_TENSOR_FFN_GATE,        "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,   "blk.%d.ffn_gate_up_exps" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_MINIMAX_M3,
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
            { LLM_TENSOR_FFN_GATE,           "blk.%d.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.%d.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.%d.ffn_up" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.%d.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.%d.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.%d.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.%d.ffn_up_exps" },
            { LLM_TENSOR_FFN_EXP_PROBS_B,    "blk.%d.exp_probs_b" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.%d.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.%d.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.%d.ffn_up_shexp" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,"blk.%d.ffn_gate_up_exps" },
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
            {   LLM_TENSOR_FFN_GATE_UP_EXPS,  "blk.%d.ffn_gate_up_exps" },
            {   LLM_TENSOR_FFN_GATE_SHEXP,    "blk.%d.ffn_gate_shexp" },
            {   LLM_TENSOR_FFN_DOWN_SHEXP,    "blk.%d.ffn_down_shexp" },
            {   LLM_TENSOR_FFN_UP_SHEXP,      "blk.%d.ffn_up_shexp" },
	        {   LLM_TENSOR_FFN_EXP_PROBS_B,   "blk.%d.exp_probs_b" },
        },
    },
    {
        LLM_ARCH_LAGUNA,
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
            {   LLM_TENSOR_FFN_GATE_UP_EXPS,  "blk.%d.ffn_gate_up_exps" },
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
            { LLM_TENSOR_FFN_GATE_UP_EXPS,       "blk.%d.ffn_gate_up_exps" },
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
        case LLAMA_FTYPE_MOSTLY_Q1_0_G128:return "Q1_0_G128 - 1.125 bpw";
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
        case MODEL_0_8B:          return "0.8B";
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
        case MODEL_12B_A2_5B:     return "12B.A2.5B";
        case MODEL_16B_A1B:       return "16B.A1B";
        case MODEL_21B_A3B:       return "21B.A3B";
        case MODEL_30B_A3B:       return "30B.A3B";
        case MODEL_33B_A3B:       return "33B.A3B";
        case MODEL_35B_A3B:       return "35B.A3B";
        case MODEL_80B_A3B:       return "80B.A3B";
        case MODEL_80B_A13B:      return "80B.A13B";
        case MODEL_100B_A6B:      return "100B.A6B";
        case MODEL_106B_A12B:     return "106B.A12B";
        case MODEL_119B_A6B:      return "119B.A6B";
        case MODEL_122B_A10B:     return "122B.A10B";
        case MODEL_230B_A10B:     return "230B.A10B";
        case MODEL_235B_A22B:     return "235B.A22B";
        case MODEL_310B_A15B:     return "310B.A15B";
        case MODEL_300B_A47B:     return "300B.A47B";
        case MODEL_355B_A32B:     return "355B.A32B";
        case MODEL_397B_A17B:     return "397B.A17B";
        case MODEL_744B_A40B:     return "744B.A40B";
        case MODEL_E2B:           return "E2B";
        case MODEL_E4B:           return "E4B";
        default:                  return "?B";
    }
}

bool llama_model_is_recurrent(const llama_model * model) {
    return llm_arch_is_recurrent(model->arch);
}

bool llama_model_is_hybrid(const llama_model * model) {
    return llm_arch_is_hybrid(model->arch);
}

bool llama_model_has_recurrent(const llama_model * model) {
    return llm_arch_is_hybrid(model->arch) || llm_arch_is_recurrent(model->arch);
}

bool llama_model_is_gemma4_mtp_assistant(const llama_model * model) {
    return model && (model->arch == LLM_ARCH_GEMMA4_MTP || model->arch == LLM_ARCH_GEMMA4_ASSISTANT);
}

bool llama_is_gemma4_mtp_file(const char * path) {
    if (!path || !*path) return false;
    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * ctx = gguf_init_from_file(path, params);
    if (!ctx) return false;
    bool result = false;
    const int key_id = gguf_find_key(ctx, "general.architecture");
    if (key_id >= 0) {
        const char * arch = gguf_get_val_str(ctx, key_id);
        if (arch && strcmp(arch, "gemma4_mtp") == 0) {
            result = true;
        }
    }
    gguf_free(ctx);
    return result;
}

bool llama_model_is_split_mode_graph(const struct llama_model * model) {
    return model && (model->split_mode == LLAMA_SPLIT_MODE_GRAPH || model->split_mode == LLAMA_SPLIT_MODE_ATTN);
}

llm_tensor llm_tensor_type(llm_arch arch, const std::string & tensor_name, int il) {
    auto it = LLM_TENSOR_NAMES.find(arch);
    if (it == LLM_TENSOR_NAMES.end()) {
        printf("%s: Oops, did not find arch\n", __func__);
        return LLM_TENSOR_UNKNOWN;
    }
    if (il < 0) {
        for (auto & entry : it->second) {
            if (tensor_name.find(entry.second) == 0) {
                return entry.first;
            }
        }
        return LLM_TENSOR_UNKNOWN;
    }
    for (auto & entry : it->second) {
        auto base_name = ::format(entry.second.c_str(), il);
        auto this_name = base_name + ".weight";
        if (tensor_name == this_name) {
            return entry.first;
        }
        this_name = base_name + ".bias";
        if (tensor_name == this_name) {
            return entry.first;
        }
        if (tensor_name == base_name) {
            return entry.first;
        }
    }
    return LLM_TENSOR_UNKNOWN;
}

size_t llama_model::cache_size(int il, ggml_type type_k, ggml_type type_v, uint32_t kv_size, int mla_attn, int n_seq_max, bool flash_attn) const {
    if (il < 0 || il >= hparams.n_layer) return 0;
    if (hparams.recurrent_layer_arr[il]) {
        auto state_sots = std::min<uint32_t>(std::max<uint32_t>(1, n_seq_max), kv_size);
        return hparams.n_embd_v_s() * state_sots * sizeof(float);
    }
    bool is_mla_attn = arch == LLM_ARCH_DEEPSEEK2 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_MISTRAL4;
    if (is_mla_attn && mla_attn) {
        auto n_embd_head_qk_rope = hparams.n_rot;
        auto kv_lora_rank = hparams.n_lora_kv;
        if (flash_attn) {
            return ggml_row_size(type_k, kv_lora_rank + n_embd_head_qk_rope) * kv_size;
        }
        auto kv_type = mla_attn == 1 ? type_k : type_v;
        auto size = ggml_row_size(kv_type, kv_lora_rank + n_embd_head_qk_rope) * kv_size;
        if (mla_attn == 1) {
            size += ggml_row_size(type_v, kv_lora_rank*kv_size);
        }
        return size;
    }
    auto n_head_kv = hparams.n_head_kv(il);
    auto k_size = ggml_row_size(type_k, hparams.n_embd_head_k(il)) * n_head_kv*kv_size;
    auto v_size = ggml_row_size(type_v, hparams.n_embd_v_gqa(il)) * kv_size;
    return k_size + v_size;
}

// ------------------------------------------------------------------
// Debug helpers
// ------------------------------------------------------------------
static void log_tensor_state(const char * ctx, struct ggml_tensor * t) {
#ifndef NDEBUG
    if (!t) {
        LLAMA_LOG_DEBUG("%s: tensor=NULL\n", ctx);
        return;
    }
    const char * buft_name = "null";
    if (t->buffer) {
        auto buft = ggml_backend_buffer_get_type(t->buffer);
        if (buft) buft_name = ggml_backend_buft_name(buft);
    }
    LLAMA_LOG_DEBUG("%s: tensor='%s' type=%s ne={%ld,%ld,%ld,%ld} nb={%zu,%zu,%zu,%zu} "
                    "buffer=%p data=%p extra=%p buft=%s\n",
        ctx, t->name, ggml_type_name(t->type),
        (long)t->ne[0], (long)t->ne[1], (long)t->ne[2], (long)t->ne[3],
        t->nb[0], t->nb[1], t->nb[2], t->nb[3],
        (void*)t->buffer, t->data, (void*)t->extra, buft_name);
#else
    (void)ctx;
    (void)t;
#endif
}

static void log_split_state(const char * ctx, struct ggml_tensor * t) {
#ifndef NDEBUG
    if (!t || !t->extra) {
        LLAMA_LOG_DEBUG("%s: no splits (extra=%p)\n", ctx, (void*)(t ? t->extra : nullptr));
        return;
    }
    auto extra = (ggml_split_tensor_t *)t->extra;
    LLAMA_LOG_DEBUG("%s: tensor='%s' n_device=%d split_dim=%d\n",
            ctx, t->name, extra->n_device, extra->split_dim);
    for (int i = 0; i < extra->n_device; ++i) {
        if (!extra->splits[i]) {
            LLAMA_LOG_DEBUG("%s:   split[%d]=NULL\n", ctx, i);
            continue;
        }
        const char * split_buft_name = "null";
        if (extra->splits[i]->buffer) {
            auto buft = ggml_backend_buffer_get_type(extra->splits[i]->buffer);
            if (buft) split_buft_name = ggml_backend_buft_name(buft);
        }
        LLAMA_LOG_DEBUG("%s:   split[%d] type=%s ne={%ld,%ld,%ld,%ld} nb={%zu,%zu,%zu,%zu} "
                        "buffer=%p data=%p buft=%s\n",
            ctx, i, ggml_type_name(extra->splits[i]->type),
            (long)extra->splits[i]->ne[0], (long)extra->splits[i]->ne[1],
            (long)extra->splits[i]->ne[2], (long)extra->splits[i]->ne[3],
            extra->splits[i]->nb[0], extra->splits[i]->nb[1],
            extra->splits[i]->nb[2], extra->splits[i]->nb[3],
            (void*)extra->splits[i]->buffer, extra->splits[i]->data, split_buft_name);
    }
#else
    (void)ctx;
    (void)t;
#endif
}

// ------------------------------------------------------------------
// Portable GGUF header parser
// ------------------------------------------------------------------
static bool gguf_find_tensor_meta(const char * path, const char * target_name,
                                  size_t & out_offset, size_t & out_nbytes,
                                  ggml_type & out_type)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    auto read_u32 = [&](uint32_t & out) -> bool {
        char b[4]; file.read(b, 4);
        out = static_cast<uint8_t>(b[0])
            | (static_cast<uint32_t>(static_cast<uint8_t>(b[1])) << 8)
            | (static_cast<uint32_t>(static_cast<uint8_t>(b[2])) << 16)
            | (static_cast<uint32_t>(static_cast<uint8_t>(b[3])) << 24);
        return file.good();
    };
    auto read_u64 = [&](uint64_t & out) -> bool {
        char b[8]; file.read(b, 8);
        out = 0;
        for (int i = 0; i < 8; ++i)
            out |= static_cast<uint64_t>(static_cast<uint8_t>(b[i])) << (8*i);
        return file.good();
    };
    auto skip = [&](size_t n) -> bool {
        file.seekg(static_cast<std::streamoff>(n), std::ios::cur);
        return file.good();
    };

    char magic[4];
    file.read(magic, 4);
    if (!file || magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F')
        return false;

    uint32_t version;
    if (!read_u32(version) || version != 3) return false;

    uint64_t n_tensors, n_kv;
    if (!read_u64(n_tensors) || !read_u64(n_kv)) return false;

    for (uint64_t i = 0; i < n_kv; ++i) {
        uint64_t klen; if (!read_u64(klen)) return false;
        if (klen == 0 || klen > 4096) return false;
        if (!skip(klen)) return false;
        uint32_t vtype; if (!read_u32(vtype)) return false;

        if (vtype == GGUF_TYPE_ARRAY) {
            uint32_t itype; uint64_t alen;
            if (!read_u32(itype) || !read_u64(alen)) return false;
            int esize = 0;
            switch (itype) {
                case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL: esize = 1; break;
                case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: esize = 2; break;
                case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: esize = 4; break;
                case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: esize = 8; break;
                case GGUF_TYPE_STRING: {
                    for (uint64_t j = 0; j < alen; ++j) {
                        uint64_t slen; if (!read_u64(slen)) return false;
                        if (!skip(slen)) return false;
                    }
                    continue;
                }
                default: return false;
            }
            if (!skip(alen * esize)) return false;
        } else {
            switch (vtype) {
                case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL:  if (!skip(1)) return false; break;
                case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: if (!skip(2)) return false; break;
                case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: if (!skip(4)) return false; break;
                case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: if (!skip(8)) return false; break;
                case GGUF_TYPE_STRING: {
                    uint64_t slen; if (!read_u64(slen)) return false;
                    if (!skip(slen)) return false;
                    break;
                }
                default: return false;
            }
        }
    }

    struct TInfo { char name[256]; uint64_t offset; ggml_type type; };
    std::vector<TInfo> tinfos;
    tinfos.reserve(n_tensors);

    for (uint64_t ti = 0; ti < n_tensors; ++ti) {
        uint64_t nlen; if (!read_u64(nlen)) return false;
        if (nlen == 0 || nlen > 255) return false;
        char tname[256];
        file.read(tname, static_cast<std::streamsize>(nlen));
        if (!file) return false;
        tname[nlen] = '\0';

        uint32_t ndims; if (!read_u32(ndims)) return false;
        for (uint32_t d = 0; d < ndims; ++d) {
            uint64_t dim; if (!read_u64(dim)) return false;
            (void)dim;
        }

        uint32_t ty; if (!read_u32(ty)) return false;
        uint64_t toffs; if (!read_u64(toffs)) return false;

        TInfo info = {};
        info.offset = toffs;
        info.type   = static_cast<ggml_type>(ty);
        memcpy(info.name, tname, nlen);
        info.name[nlen] = '\0';
        tinfos.push_back(info);
    }

    if (tinfos.empty()) return false;

    std::sort(tinfos.begin(), tinfos.end(),
              [](const TInfo & a, const TInfo & b) { return a.offset < b.offset; });

    uint64_t raw_header_size = static_cast<uint64_t>(file.tellg());
    uint64_t header_size     = (raw_header_size + 31) / 32 * 32;
    for (auto & t : tinfos) t.offset += header_size;

    file.seekg(0, std::ios::end);
    uint64_t file_size = static_cast<uint64_t>(file.tellg());

    for (size_t i = 0; i < tinfos.size(); ++i) {
        if (strcmp(tinfos[i].name, target_name) != 0) continue;
        uint64_t tsize = (i + 1 < tinfos.size())
                         ? (tinfos[i+1].offset - tinfos[i].offset)
                         : (file_size - tinfos[i].offset);
        out_offset = static_cast<size_t>(tinfos[i].offset);
        out_nbytes = static_cast<size_t>(tsize);
        out_type   = tinfos[i].type;
        return true;
    }
    return false;
}

// ------------------------------------------------------------------
// Buffer census helper
// ------------------------------------------------------------------
static size_t count_buffer_users(
        const std::vector<std::pair<std::string, struct ggml_tensor *>> & tensors_by_name,
        ggml_backend_buffer_t buf)
{
    if (!buf) return 0;
    size_t n = 0;
    for (auto & p : tensors_by_name) {
        if (p.second->buffer == buf) ++n;
    }
    return n;
}

static bool is_original_snapshot_buffer(llama_model & model, ggml_backend_buffer_t buf) {
    if (!buf) return false;
    for (const auto & kv : model.tensor_reload_sources) {
        const auto & src = kv.second;
        if (buf == src.original_buffer) return true;
        for (const auto & os : src.original_splits) {
            if (buf == os.buffer) return true;
        }
    }
    return false;
}

// ------------------------------------------------------------------
// Final size estimator
// ------------------------------------------------------------------
static size_t llama_model_compute_final_nbytes(struct ggml_tensor * tensor, ggml_type new_type) {
    if (new_type == tensor->type) {
        return ggml_nbytes(tensor);
    }

    int n_dims = 0;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        if (tensor->ne[i] != 1) { n_dims = i + 1; break; }
    }

    size_t ctx_mem_size = ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false);
    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_mem_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };
    struct ggml_context * ctx_tmp = ggml_init(params);
    if (!ctx_tmp) return SIZE_MAX;

    struct ggml_tensor * tmp = ggml_new_tensor(ctx_tmp, new_type, n_dims, tensor->ne);
    if (!tmp) {
        ggml_free(ctx_tmp);
        return SIZE_MAX;
    }
    size_t nbytes = ggml_nbytes(tmp);
    ggml_free(ctx_tmp);
    return nbytes;
}

// ------------------------------------------------------------------
// Fallback allocator
// ------------------------------------------------------------------
static ggml_backend_buffer_t alloc_buffer_fallback(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, size);
    if (buf) {
        LLAMA_LOG_DEBUG("%s: allocated %zu bytes on backend '%s'\n",
                __func__, size, ggml_backend_buft_name(buft));
        return buf;
    }

    LLAMA_LOG_WARN("%s: backend alloc failed (%zu bytes on '%s'), trying CPU fallback\n",
            __func__, size, ggml_backend_buft_name(buft));
    buft = ggml_backend_cpu_buffer_type();
    if (!buft) return nullptr;

    buf = ggml_backend_buft_alloc_buffer(buft, size);
    if (!buf) {
        LLAMA_LOG_WARN("%s: CPU fallback alloc failed (%zu bytes)\n", __func__, size);
        return nullptr;
    }
    LLAMA_LOG_DEBUG("%s: allocated %zu bytes on CPU fallback\n", __func__, size);
    return buf;
}

// ------------------------------------------------------------------
// MoE sibling resync
// ------------------------------------------------------------------
// MoE layers have three weight tensors per block: gate, up, down.
// The CUDA split backend distributes each tensor across GPUs by splitting
// one dimension (usually dim 0 or 1). Split boundaries must be multiples
// of the quantization block size (e.g. 256 for IQ1_KT). If the reference
// tensor changes quantization type, its block size changes, which changes
// the valid split boundaries. ALL siblings in the same layer MUST adopt
// the SAME per-device split dimensions, otherwise the backend dispatches
// rows to the wrong devices and corrupts inference.
//
// When the reference tensor is back on its original snapshot, siblings
// can simply be reattached to their original snapshots too -- no data
// movement or allocation is required.
// ------------------------------------------------------------------


// ------------------------------------------------------------------
// Sibling name registration
// ------------------------------------------------------------------
static void populate_moe_siblings(const char * name, llama_model::tensor_reload_source & src) {
    LLAMA_LOG_DEBUG("%s: name='%s'\n", __func__, name);

    static const char * suffixes[] = {
        ".ffn_down_exps.weight",
        ".ffn_up_exps.weight",
        ".ffn_gate_exps.weight",
    };
    std::string n(name);
    for (const char * sfx : suffixes) {
        size_t pos = n.find(sfx);
        if (pos == std::string::npos) continue;
        std::string base = n.substr(0, pos);
        for (const char * other : suffixes) {
            if (strcmp(other, sfx) != 0) {
                src.sibling_names.push_back(base + other);
                LLAMA_LOG_DEBUG("%s: registered sibling '%s' for '%s'\n",
                        __func__, (base + other).c_str(), name);
            }
        }
        return;
    }
    LLAMA_LOG_DEBUG("%s: '%s' no MoE suffix matched\n", __func__, name);
}

// ------------------------------------------------------------------
// Snapshot helper
// ------------------------------------------------------------------
static void snapshot_tensor_source(struct ggml_tensor * tensor,
                                   llama_model::tensor_reload_source & src)
{
    if (!tensor || src.original_buffer != nullptr) return;

    src.original_buffer = tensor->buffer;
    src.original_data   = tensor->data;
    src.original_nbytes = ggml_nbytes(tensor);
    src.original_type   = tensor->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        src.original_ne[i] = tensor->ne[i];
        src.original_nb[i] = tensor->nb[i];
    }
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    if (extra) {
        src.original_extra = extra;
        src.original_splits.clear();
        for (int i = 0; i < extra->n_device; ++i) {
            llama_model::tensor_reload_source::split_info si;
            if (extra->splits[i]) {
                for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                    si.ne[j] = extra->splits[i]->ne[j];
                    si.nb[j] = extra->splits[i]->nb[j];
                }
                si.data   = extra->splits[i]->data;
                si.buffer = extra->splits[i]->buffer;
                si.tensor = extra->splits[i];
            }
            src.original_splits.push_back(si);
        }
    }
    populate_moe_siblings(ggml_get_name(tensor), src);
    src.state = llama_model::reload_state::ON_ORIGINAL;
    log_tensor_state("snapshot_tensor_source", tensor);
}

// ------------------------------------------------------------------
// Eager snapshot
// ------------------------------------------------------------------
void llama_model::snapshot_all_reload_tensors() {
    if (reload_snapshots_done.exchange(true)) return;

    LLAMA_LOG_INFO("%s: eager snapshot of all reload tensors + siblings\n", __func__);

    for (auto & kv : tensor_reload_sources) {
        struct ggml_tensor * tensor = nullptr;
        for (auto & p : tensors_by_name) {
            if (p.first == kv.first) { tensor = p.second; break; }
        }
        if (!tensor) continue;
        snapshot_tensor_source(tensor, kv.second);
    }

    for (auto & kv : tensor_reload_sources) {
        auto & src = kv.second;
        for (const auto & sib_name : src.sibling_names) {
            auto it = tensor_reload_sources.find(sib_name);
            if (it == tensor_reload_sources.end()) continue;
            if (it->second.original_buffer != nullptr) continue;

            struct ggml_tensor * sib = nullptr;
            for (auto & p : tensors_by_name) {
                if (p.first == sib_name) { sib = p.second; break; }
            }
            if (!sib) continue;
            snapshot_tensor_source(sib, it->second);
        }
    }
}

// ------------------------------------------------------------------
// Re-attachment helper
// ------------------------------------------------------------------
static bool reattach_split_tensor_to_shared(llama_model & model, const char * name) {
    auto it = model.tensor_reload_sources.find(name);
    if (it == model.tensor_reload_sources.end()) return false;
    auto & src = it->second;

    if (!src.original_buffer) return false;

    struct ggml_tensor * tensor = nullptr;
    for (auto & p : model.tensors_by_name) {
        if (p.first == name) { tensor = p.second; break; }
    }
    if (!tensor) return false;
    if (tensor->buffer == src.original_buffer) {
        log_tensor_state("reattach_split_tensor_to_shared", tensor);
        src.state = llama_model::reload_state::ON_ORIGINAL;
        return true;
    }

    if (tensor->buffer && src.state != llama_model::reload_state::ON_ORIGINAL) {
        ggml_backend_buffer_free(tensor->buffer);
    }
    tensor->buffer = nullptr;
    tensor->data   = nullptr;

    tensor->buffer = src.original_buffer;
    tensor->data   = src.original_data;
    tensor->type   = src.original_type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        tensor->ne[i] = src.original_ne[i];
        tensor->nb[i] = src.original_nb[i];
    }

    if (src.original_extra) {
        tensor->extra = src.original_extra;
        auto extra = (ggml_split_tensor_t *)tensor->extra;
        for (int i = 0; i < extra->n_device && i < (int)src.original_splits.size(); ++i) {
            auto & os = src.original_splits[i];
            if (!extra->splits[i] && os.tensor) {
                extra->splits[i] = os.tensor;
            }
            if (extra->splits[i]) {
                if (extra->splits[i]->buffer && extra->splits[i]->buffer != os.buffer &&
                    src.state != llama_model::reload_state::ON_ORIGINAL) {
                    ggml_backend_buffer_free(extra->splits[i]->buffer);
                }
                extra->splits[i]->data   = os.data;
                extra->splits[i]->buffer = os.buffer;
                extra->splits[i]->type   = src.original_type;
                for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                    extra->splits[i]->ne[j] = os.ne[j];
                    extra->splits[i]->nb[j] = os.nb[j];
                }
            }
        }
    }

    src.state = llama_model::reload_state::ON_ORIGINAL;
    return true;
}

// ------------------------------------------------------------------
// MoE sibling resync
// ------------------------------------------------------------------
static void resync_moe_sibling_splits(
        llama_model & model,
        struct ggml_context * /*ctx_tmp*/,
        struct ggml_tensor * ref_tensor,
        const char * ref_name)
{
    std::string name_str(ref_name);
    std::string layer_prefix;
    std::vector<std::string> suffixes;

    if (name_str.find(".ffn_down_exps.weight") != std::string::npos) {
        layer_prefix = name_str.substr(0, name_str.find(".ffn_down_exps.weight"));
        suffixes = {".ffn_up_exps.weight", ".ffn_gate_exps.weight"};
    } else if (name_str.find(".ffn_up_exps.weight") != std::string::npos) {
        layer_prefix = name_str.substr(0, name_str.find(".ffn_up_exps.weight"));
        suffixes = {".ffn_down_exps.weight", ".ffn_gate_exps.weight"};
    } else if (name_str.find(".ffn_gate_exps.weight") != std::string::npos) {
        layer_prefix = name_str.substr(0, name_str.find(".ffn_gate_exps.weight"));
        suffixes = {".ffn_up_exps.weight", ".ffn_down_exps.weight"};
    } else {
        return;
    }

    auto ref_extra = (ggml_split_tensor_t *)ref_tensor->extra;
    if (!ref_extra) return;

    auto it_ref_src = model.tensor_reload_sources.find(ref_name);
    if (it_ref_src != model.tensor_reload_sources.end() && ref_tensor->buffer == it_ref_src->second.original_buffer) {
        for (const auto & suffix : suffixes) {
            reattach_split_tensor_to_shared(model, (layer_prefix + suffix).c_str());
        }
        return;
    }

    struct sibling_job {
        std::string name;
        struct ggml_tensor * tensor;
        ggml_split_tensor_t * extra;
        std::vector<char> host_buf;
        bool needs_resync = false;
    };
    std::vector<sibling_job> jobs;

    for (const auto & suffix : suffixes) {
        std::string sib_name = layer_prefix + suffix;
        struct ggml_tensor * sib = nullptr;
        for (auto & p : model.tensors_by_name) {
            if (p.first == sib_name) { sib = p.second; break; }
        }
        if (!sib || !sib->extra || sib == ref_tensor) continue;

        auto sib_extra = (ggml_split_tensor_t *)sib->extra;
        if (sib_extra->n_device != ref_extra->n_device) continue;

        int sib_dim = sib_extra->split_dim < 0 ? 0 : sib_extra->split_dim;
        int ref_dim = ref_extra->split_dim < 0 ? 0 : ref_extra->split_dim;

        bool need = false;
        for (int i = 0; i < ref_extra->n_device; ++i) {
            bool rh = ref_extra->splits[i] != nullptr;
            bool sh = sib_extra->splits[i] != nullptr;
            if (rh != sh) { need = true; break; }
            if (rh && sh && sib_extra->splits[i]->ne[sib_dim] != ref_extra->splits[i]->ne[ref_dim]) {
                need = true; break;
            }
        }
        if (!need) continue;

        size_t nbytes = ggml_nbytes(sib);
        std::vector<char> buf(nbytes);
        ggml_backend_tensor_get(sib, buf.data(), 0, nbytes);
        jobs.push_back({sib_name, sib, sib_extra, std::move(buf), true});
    }

    if (jobs.empty()) return;
    log_split_state("resync_moe_sibling_splits", ref_tensor);

    // Phase A: Detach / free old buffers, allocate new main handles
    for (auto & job : jobs) {
        auto sib = job.tensor;

        ggml_backend_buffer_type_t buft = sib->buffer
            ? ggml_backend_buffer_get_type(sib->buffer)
            : ggml_backend_cpu_buffer_type();

        auto it = model.tensor_reload_sources.find(job.name);
        bool was_orig = (it != model.tensor_reload_sources.end() && it->second.state == llama_model::reload_state::ON_ORIGINAL);

        if (sib->buffer) {
            if (!was_orig) ggml_backend_buffer_free(sib->buffer);
            sib->buffer = nullptr;
            sib->data   = nullptr;
        }

        size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, sib);
        ggml_backend_buffer_t new_buf = alloc_buffer_fallback(buft, alloc_size);
        if (!new_buf) {
            job.needs_resync = false;
            continue;
        }
        sib->buffer = new_buf;
        sib->data   = (void*)0x1; // dummy; split backend uses extra->splits

        if (it != model.tensor_reload_sources.end()) {
            it->second.state = llama_model::reload_state::DETACHED;
        }
    }

    // Phase B: Propagate dimensions & recompute strides
    for (auto & job : jobs) {
        if (!job.needs_resync) continue;
        auto sib = job.tensor;
        auto sib_extra = job.extra;

        for (int i = 0; i < ref_extra->n_device; ++i) {
            if (!ref_extra->splits[i]) {
                if (sib_extra->splits[i]) sib_extra->splits[i] = nullptr;
                continue;
            }
            if (!sib_extra->splits[i]) continue;
            sib_extra->splits[i]->ne[sib_extra->split_dim < 0 ? 0 : sib_extra->split_dim] =
                ref_extra->splits[i]->ne[ref_extra->split_dim < 0 ? 0 : ref_extra->split_dim];
        }

        int n_dims = 0;
        for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
            if (sib->ne[i] != 1) { n_dims = i + 1; break; }
        }
        size_t ctx_size = ggml_tensor_overhead() * (sib_extra->n_device + 4);
        if (ctx_size < 16384) ctx_size = 16384;
        struct ggml_init_params p = { ctx_size, NULL, true };
        struct ggml_context * ctx = ggml_init(p);
        if (ctx) {
            for (int i = 0; i < sib_extra->n_device; ++i) {
                if (!sib_extra->splits[i]) continue;
                auto tmp = ggml_new_tensor(ctx, sib->type, n_dims, sib_extra->splits[i]->ne);
                if (tmp) {
                    for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                        sib_extra->splits[i]->nb[j] = tmp->nb[j];
                    }
                }
            }
            ggml_free(ctx);
        }
    }

    // Phase C: Allocate GPU split buffers
    bool gpu_failed = false;
    for (auto & job : jobs) {
        if (!job.needs_resync) continue;
        auto sib_extra = job.extra;

        for (int i = 0; i < sib_extra->n_device; ++i) {
            if (!sib_extra->splits[i]) continue;
            size_t need = ggml_nbytes(sib_extra->splits[i]);
            auto buft = ggml_backend_cuda_buffer_type(i);
            auto b = ggml_backend_buft_alloc_buffer(buft, need);
            if (!b) { gpu_failed = true; break; }
            sib_extra->splits[i]->buffer = b;
            sib_extra->splits[i]->data   = ggml_backend_buffer_get_base(b);
        }
        if (gpu_failed) break;
    }

    // Phase D: If any GPU alloc failed, move entire layer to CPU
    if (gpu_failed) {
        for (auto & job : jobs) {
            if (!job.needs_resync) continue;
            auto sib = job.tensor;
            auto sib_extra = job.extra;

            for (int i = 0; i < sib_extra->n_device; ++i) {
                if (sib_extra->splits[i] && sib_extra->splits[i]->buffer) {
                    auto it = model.tensor_reload_sources.find(job.name);
                    bool is_orig = false;
                    if (it != model.tensor_reload_sources.end() && i < (int)it->second.original_splits.size()) {
                        is_orig = (sib_extra->splits[i]->buffer == it->second.original_splits[i].buffer);
                    }
                    if (!is_orig) ggml_backend_buffer_free(sib_extra->splits[i]->buffer);
                    sib_extra->splits[i]->buffer = nullptr;
                    sib_extra->splits[i]->data   = nullptr;
                }
            }

            if (sib->buffer) {
                auto it = model.tensor_reload_sources.find(job.name);
                bool is_orig = (it != model.tensor_reload_sources.end() && it->second.state == llama_model::reload_state::ON_ORIGINAL);
                if (!is_orig) ggml_backend_buffer_free(sib->buffer);
                sib->buffer = nullptr;
                sib->data   = nullptr;
            }

            size_t need = ggml_nbytes(sib);
            auto cpu = alloc_buffer_fallback(ggml_backend_cpu_buffer_type(), need);
            if (cpu) {
                sib->buffer = cpu;
                sib->data   = ggml_backend_buffer_get_base(cpu);
                auto it = model.tensor_reload_sources.find(job.name);
                if (it != model.tensor_reload_sources.end()) it->second.state = llama_model::reload_state::FALLBACK_CPU;
            }
        }
    }

    // Phase E: Write data back
    for (auto & job : jobs) {
        if (!job.needs_resync) continue;
        ggml_backend_tensor_set(job.tensor, job.host_buf.data(), 0, job.host_buf.size());
    }
}

// ------------------------------------------------------------------
// reload_tensor_split_path
// ------------------------------------------------------------------
static bool reload_tensor_split_path(
        llama_model & model,
        struct ggml_tensor * tensor,
        llama_model::tensor_reload_source & src,
        const std::vector<char> & host_buf,
        ggml_type curr_type,
        bool returning_to_original,
        ggml_backend_buffer_t old_buf)
{
		(void)curr_type;
    const char * name = ggml_get_name(tensor);

    if (returning_to_original) {
        if (old_buf && src.state != llama_model::reload_state::ON_ORIGINAL) {
            ggml_backend_buffer_free(old_buf);
        }
        tensor->buffer = nullptr;
        tensor->data   = nullptr;

        if (!reattach_split_tensor_to_shared(model, name)) return false;
        for (const auto & sib : src.sibling_names) {
            reattach_split_tensor_to_shared(model, sib.c_str());
        }
        return true;
    }

    ggml_backend_buffer_type_t buft = old_buf
        ? ggml_backend_buffer_get_type(old_buf)
        : ggml_backend_cpu_buffer_type();

    if (old_buf && src.state != llama_model::reload_state::ON_ORIGINAL) {
        ggml_backend_buffer_free(old_buf);
    }
    tensor->buffer = nullptr;
    tensor->data   = nullptr;

    size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t new_buf = alloc_buffer_fallback(buft, alloc_size);
    if (!new_buf) return false;

    ggml_backend_tensor_alloc(new_buf, tensor, ggml_backend_buffer_get_base(new_buf));
    //ggml_backend_buffer_init_tensor(tensor->buffer, tensor);

    ggml_backend_tensor_set(tensor, host_buf.data(), 0, host_buf.size());
    log_tensor_state("reload_tensor_split_path", tensor);
    if (tensor->extra) resync_moe_sibling_splits(model, nullptr, tensor, name);

    src.state = llama_model::reload_state::DETACHED;
    return true;
}

// ------------------------------------------------------------------
// reload_tensor_non_split_path
// ------------------------------------------------------------------
static bool reload_tensor_non_split_path(
        llama_model & model,
        struct ggml_tensor * tensor,
        llama_model::tensor_reload_source & src,
        const std::vector<char> & host_buf,
        ggml_type curr_type,
        bool returning_to_original,
        ggml_backend_buffer_t old_buf)
{
		(void)model;
		(void)curr_type;
#ifndef NDEBUG
    const char * name = ggml_get_name(tensor);
#endif

    if (returning_to_original) {
        if (old_buf && src.state != llama_model::reload_state::ON_ORIGINAL) {
            ggml_backend_buffer_free(old_buf);
        }
        tensor->buffer = src.original_buffer;
        tensor->data   = src.original_data;
        tensor->type   = src.original_type;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            tensor->ne[i] = src.original_ne[i];
            tensor->nb[i] = src.original_nb[i];
        }
        src.state = llama_model::reload_state::ON_ORIGINAL;
        return true;
    }

    ggml_backend_buffer_type_t buft = old_buf
        ? ggml_backend_buffer_get_type(old_buf)
        : ggml_backend_cpu_buffer_type();

    if (old_buf && src.state != llama_model::reload_state::ON_ORIGINAL) {
        ggml_backend_buffer_free(old_buf);
#ifndef NDEBUG
    } else if (old_buf) {
        LLAMA_LOG_DEBUG("detaching from original snapshot buffer %p for '%s'\n", (void*)old_buf, name);
#endif
    }
    tensor->buffer = nullptr;
    tensor->data   = nullptr;

    size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);
    ggml_backend_buffer_t new_buf = alloc_buffer_fallback(buft, alloc_size);
    if (!new_buf) return false;

    ggml_backend_tensor_alloc(new_buf, tensor, ggml_backend_buffer_get_base(new_buf));
    ggml_backend_tensor_set(tensor, host_buf.data(), 0, host_buf.size());

    src.state = llama_model::reload_state::DETACHED;
    return true;
}

// ------------------------------------------------------------------
// apply_tensor_type_change
// ------------------------------------------------------------------
static bool apply_tensor_type_change(
        llama_model & /*model*/,
        struct ggml_tensor * tensor,
        llama_model::tensor_reload_source & /*src*/,
        ggml_type curr_type)
{
#ifndef NDEBUG
    const char * name = ggml_get_name(tensor);
    (void)name;
#endif
    tensor->type = curr_type;

    int n_dims = 0;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        if (tensor->ne[i] != 1) { n_dims = i + 1; break; }
    }

    size_t ctx_size = ggml_tensor_overhead() * (1 + (tensor->extra ? ((ggml_split_tensor_t*)tensor->extra)->n_device : 0))
                    + ggml_graph_overhead_custom(1, false);
    struct ggml_init_params p = { ctx_size, NULL, true };
    struct ggml_context * ctx = ggml_init(p);
    if (!ctx) return false;

    auto tmp = ggml_new_tensor(ctx, curr_type, n_dims, tensor->ne);
    if (!tmp) { ggml_free(ctx); return false; }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) tensor->nb[i] = tmp->nb[i];

    if (tensor->extra) {
        auto extra = (ggml_split_tensor_t *)tensor->extra;
        auto tt = ggml_internal_get_type_traits(curr_type);

        if (tt.blck_size > 1 && extra->split_dim == 0) {
            int64_t bs = tt.blck_size;
            int n = extra->n_device;
            std::vector<int64_t> bounds(n, 0);
            int64_t acc = 0;
            for (int i = 0; i < n; ++i) {
                if (extra->splits[i]) acc += extra->splits[i]->ne[0];
                bounds[i] = acc;
            }
            for (int i = 0; i < n - 1; ++i) {
                if (bounds[i] > 0) {
                    bounds[i] = ((bounds[i] + bs - 1) / bs) * bs;
                }
            }
            bounds[n - 1] = tensor->ne[0];
            for (int i = 1; i < n; ++i) {
                if (bounds[i] < bounds[i - 1]) bounds[i] = bounds[i - 1];
            }
            int64_t prev = 0;
            for (int i = 0; i < n; ++i) {
                if (extra->splits[i]) {
                    int64_t ne0 = bounds[i] - prev;
                    if (ne0 <= 0) {
                        extra->splits[i] = nullptr;
                    } else {
                        extra->splits[i]->ne[0] = ne0;
                    }
                }
                prev = bounds[i];
            }
        }

        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            split->type = curr_type;
            auto t = ggml_new_tensor(ctx, curr_type, n_dims, split->ne);
            if (t) {
                for (int j = 0; j < GGML_MAX_DIMS; ++j) split->nb[j] = t->nb[j];
            }
        }

        int64_t sum = 0;
        for (int i = 0; i < extra->n_device; ++i) {
            if (extra->splits[i]) sum += extra->splits[i]->ne[0];
        }
        GGML_ASSERT(sum == tensor->ne[0]);
    }

    ggml_free(ctx);
    return true;
}

// ------------------------------------------------------------------
// reload_tensor
// ------------------------------------------------------------------
bool llama_model::reload_tensor(const char * name) {
    auto it = tensor_reload_sources.find(name);
    if (it == tensor_reload_sources.end()) return false;
    auto & src = it->second;

    struct stat st;
    if (stat(src.path.c_str(), &st) != 0) return false;

    bool changed = (st.st_mtime != src.last_mtime);
#ifdef __linux__
    changed = changed || (st.st_mtim.tv_nsec != src.last_mtime_ns);
#endif
    if (!changed) return false;

    size_t off = 0, file_nbytes = 0;
    ggml_type curr_type = GGML_TYPE_COUNT;
    if (!gguf_find_tensor_meta(src.path.c_str(), name, off, file_nbytes, curr_type)) return false;

    std::ifstream file(src.path, std::ios::binary);
    if (!file) return false;
    file.seekg((std::streamoff)off);
    if (!file) return false;

    struct ggml_tensor * tensor = nullptr;
    for (auto & p : tensors_by_name) {
        if (p.first == name) { tensor = p.second; break; }
    }
    if (!tensor || !src.original_buffer) return false;

    ggml_backend_buffer_t old_buf = tensor->buffer;
    bool returning = (curr_type == src.original_type);

    std::vector<char> host_buf;
    if (!returning) {
        if (curr_type != tensor->type) {
            if (!apply_tensor_type_change(*this, tensor, src, curr_type)) return false;
        }
        size_t need = ggml_nbytes(tensor);
        if (file_nbytes < need) return false;
        host_buf.resize(need);
        file.read(host_buf.data(), (std::streamsize)need);
        if (!file || (size_t)file.gcount() != need) return false;
    }

    bool ok = false;
    if (tensor->extra) {
        ok = reload_tensor_split_path(*this, tensor, src, host_buf, curr_type, returning, old_buf);
    } else {
        ok = reload_tensor_non_split_path(*this, tensor, src, host_buf, curr_type, returning, old_buf);
    }

    if (ok) {
        src.last_mtime = st.st_mtime;
#ifdef __linux__
        src.last_mtime_ns = st.st_mtim.tv_nsec;
#endif
    }
    return ok;
}

// ------------------------------------------------------------------
// reload_changed_tensors
// ------------------------------------------------------------------
bool llama_model::reload_changed_tensors() {
    snapshot_all_reload_tensors();

    struct job { const char * name; bool returning; };
    std::vector<job> jobs;

    for (auto & kv : tensor_reload_sources) {
        auto & src = kv.second;
        struct stat st;
        if (stat(src.path.c_str(), &st) != 0) continue;

        bool changed = (st.st_mtime != src.last_mtime);
#ifdef __linux__
        changed = changed || (st.st_mtim.tv_nsec != src.last_mtime_ns);
#endif
        if (!changed) continue;

        size_t off = 0, nbytes = 0;
        ggml_type t = GGML_TYPE_COUNT;
        if (!gguf_find_tensor_meta(src.path.c_str(), kv.first.c_str(), off, nbytes, t)) continue;

        struct ggml_tensor * tensor = nullptr;
        for (auto & p : tensors_by_name) {
            if (p.first == kv.first) { tensor = p.second; break; }
        }
        if (!tensor) continue;

        bool returning = (t == src.original_type);
        jobs.push_back({kv.first.c_str(), returning});
    }

    std::sort(jobs.begin(), jobs.end(), [](const job & a, const job & b) {
        return a.returning > b.returning;
    });

    bool r = false;
    for (auto & j : jobs) {
        if (reload_tensor(j.name)) {
            r = true;
            LLAMA_LOG_INFO("reloaded tensor '%s'\n", j.name);
        }
    }

    if (r) {
#ifdef GGML_USE_CUDA
        ggml_backend_cuda_invalidate_graphs();
#endif
        ++graph_generation;
    }
    return r;
}
