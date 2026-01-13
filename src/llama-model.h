#pragma once

#include "llama-impl.h"
#include "llama-arch.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-hparams.h"

#include "ggml-backend.h"

#include <vector>
#include <unordered_map>
#include <set>

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_14M,
    MODEL_17M,
    MODEL_22M,
    MODEL_33M,
    MODEL_60M,
    MODEL_70M,
    MODEL_80M,
    MODEL_109M,
    MODEL_137M,
    MODEL_140M,
    MODEL_160M,
    MODEL_190M,
    MODEL_220M,
    MODEL_250M,
    MODEL_256M,
    MODEL_270M,
    MODEL_335M,
    MODEL_350M,
    MODEL_360M,
    MODEL_410M,
    MODEL_450M,
    MODEL_475M,
    MODEL_558M,
    MODEL_700M,
    MODEL_770M,
    MODEL_780M,
    MODEL_950M,
    MODEL_0_3B,
    MODEL_0_5B,
    MODEL_0_6B,
    MODEL_1B,
    MODEL_1_2B,
    MODEL_1_3B,
    MODEL_1_4B,
    MODEL_1_5B,
    MODEL_1_6B,
    MODEL_1_7B,
    MODEL_1_8B,
    MODEL_2B,
    MODEL_2_6B,
    MODEL_2_8B,
    MODEL_2_9B,
    MODEL_3B,
    MODEL_4B,
    MODEL_6B,
    MODEL_6_9B,
    MODEL_7B,
    MODEL_8B,
    MODEL_9B,
    MODEL_11B,
    MODEL_12B,
    MODEL_13B,
    MODEL_14B,
    MODEL_15B,
    MODEL_16B,
    MODEL_20B,
    MODEL_27B,
    MODEL_30B,
    MODEL_32B,
    MODEL_34B,
    MODEL_35B,
    MODEL_36B,
    MODEL_40B,
    MODEL_65B,
    MODEL_70B,
    MODEL_120B,
    MODEL_142B,
    MODEL_236B,
    MODEL_290B,
    MODEL_314B,
    MODEL_405B,
    MODEL_671B,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
    MODEL_XL,
    MODEL_A1_7B,
    MODEL_A2_7B,
    MODEL_8x7B,
    MODEL_8x22B,
    MODEL_16x12B,
    MODEL_16x3_8B,
    MODEL_10B_128x3_66B,
    MODEL_57B_A14B,
    MODEL_17B_16E,
    MODEL_17B_128E,
    MODEL_A13B,
    MODEL_7B_A1B,
    MODEL_8B_A1B,
    MODEL_16B_A1B,
    MODEL_21B_A3B, // Ernie MoE small
    MODEL_30B_A3B,
    MODEL_80B_A13B,
    MODEL_100B_A6B,
    MODEL_106B_A12B,
    MODEL_230B_A10B, // Minimax M2
    MODEL_235B_A22B,
    MODEL_310B_A15B,
    MODEL_300B_A47B, // Ernie MoE big
    MODEL_355B_A32B,
    MODEL_E2B,
    MODEL_E4B,
};


struct llama_layer_nextn {
    struct ggml_tensor * eh_proj          = nullptr;
    struct ggml_tensor * embed_tokens     = nullptr;
    struct ggml_tensor * enorm            = nullptr;
    struct ggml_tensor * hnorm            = nullptr;
    struct ggml_tensor * shared_head_head = nullptr;
    struct ggml_tensor * shared_head_norm = nullptr;
};

// TODO: separate into "llama_layer_enc" and "llama_layer_dec"
struct llama_layer {
    // normalization
    struct ggml_tensor * attn_norm = nullptr;
    struct ggml_tensor * attn_norm_b = nullptr;
    struct ggml_tensor * attn_norm_2 = nullptr;
    struct ggml_tensor * attn_norm_2_b = nullptr;
    struct ggml_tensor * attn_q_norm = nullptr;
    struct ggml_tensor * attn_q_norm_b = nullptr;
    struct ggml_tensor * attn_k_norm = nullptr;
    struct ggml_tensor * attn_k_norm_b = nullptr;
    struct ggml_tensor * attn_out_norm = nullptr;
    struct ggml_tensor * attn_out_norm_b = nullptr;
    struct ggml_tensor * attn_q_a_norm = nullptr;
    struct ggml_tensor * attn_kv_a_norm = nullptr;
    struct ggml_tensor * attn_sub_norm = nullptr;
    struct ggml_tensor * attn_post_norm = nullptr;
    struct ggml_tensor * ffn_sub_norm = nullptr;
    struct ggml_tensor * attn_norm_cross = nullptr;
    struct ggml_tensor * attn_norm_enc = nullptr;

    // attention
    struct ggml_tensor * wq = nullptr;
    struct ggml_tensor * wk = nullptr;
    struct ggml_tensor * wv = nullptr;
    struct ggml_tensor * wo = nullptr;
    struct ggml_tensor * wqkv = nullptr;
    struct ggml_tensor * wqk  = nullptr;
    struct ggml_tensor * wkv  = nullptr;
    struct ggml_tensor * wq_a = nullptr;
    struct ggml_tensor * wq_b = nullptr;
    struct ggml_tensor * wkv_a_mqa = nullptr;
    struct ggml_tensor * wkq_a_mqa = nullptr;
    struct ggml_tensor * wkv_b = nullptr;
    struct ggml_tensor * wk_b = nullptr;
    struct ggml_tensor * wv_b = nullptr;
    struct ggml_tensor * wq_cross = nullptr;
    struct ggml_tensor * wk_cross = nullptr;
    struct ggml_tensor * wv_cross = nullptr;
    struct ggml_tensor * wo_cross = nullptr;
    struct ggml_tensor * wq_enc = nullptr;
    struct ggml_tensor * wk_enc = nullptr;
    struct ggml_tensor * wv_enc = nullptr;
    struct ggml_tensor * wo_enc = nullptr;
    struct ggml_tensor * attn_sinks = nullptr;

    // attention bias
    struct ggml_tensor * bq = nullptr;
    struct ggml_tensor * bk = nullptr;
    struct ggml_tensor * bv = nullptr;
    struct ggml_tensor * bo = nullptr;
    struct ggml_tensor * bqkv = nullptr;
    struct ggml_tensor * bqk  = nullptr;
    struct ggml_tensor * bkv  = nullptr;

    llama_split_tensor split_attn_norm;
    llama_split_tensor split_attn_sinks;
    llama_split_tensor split_wq;
    llama_split_tensor split_wk;
    llama_split_tensor split_wv;
    llama_split_tensor split_wo;
    llama_split_tensor split_wqkv;
    llama_split_tensor split_wqk;
    llama_split_tensor split_wkv;
    llama_split_tensor split_bq;
    llama_split_tensor split_bk;
    llama_split_tensor split_bv;
    llama_split_tensor split_bo;
    llama_split_tensor split_bqkv;
    llama_split_tensor split_bqk;
    llama_split_tensor split_bkv;
    llama_split_tensor split_q_norm;
    llama_split_tensor split_k_norm;
    llama_split_tensor split_sinks;

    // relative position bias
    struct ggml_tensor * attn_rel_b = nullptr;
    struct ggml_tensor * attn_rel_b_enc = nullptr;
    struct ggml_tensor * attn_rel_b_cross = nullptr;

    // normalization
    struct ggml_tensor * ffn_norm = nullptr;
    struct ggml_tensor * ffn_norm_b = nullptr;
    struct ggml_tensor * ffn_post_norm = nullptr;
    struct ggml_tensor * layer_out_norm = nullptr;
    struct ggml_tensor * layer_out_norm_b = nullptr;
    struct ggml_tensor * ffn_norm_exps = nullptr;
    struct ggml_tensor * ffn_norm_enc = nullptr;

    // ff
    struct ggml_tensor * ffn_gate = nullptr; // w1
    struct ggml_tensor * ffn_down = nullptr; // w2
    struct ggml_tensor * ffn_up = nullptr;   // w3
    struct ggml_tensor * ffn_gate_enc = nullptr;
    struct ggml_tensor * ffn_down_enc = nullptr;
    struct ggml_tensor * ffn_up_enc = nullptr;

    llama_split_tensor split_ffn_up;
    llama_split_tensor split_ffn_gate;
    llama_split_tensor split_ffn_down;
    llama_split_tensor split_ffn_norm;

    // ff MoE
    struct ggml_tensor * ffn_gate_inp = nullptr;
    struct ggml_tensor * ffn_gate_exps = nullptr;
    struct ggml_tensor * ffn_down_exps = nullptr;
    struct ggml_tensor * ffn_up_exps  = nullptr;
    struct ggml_tensor * ffn_up_gate_exps  = nullptr;

    llama_split_tensor split_ffn_gate_inp;
    llama_split_tensor split_ffn_up_exps;
    llama_split_tensor split_ffn_gate_exps;
    llama_split_tensor split_ffn_down_exps;

    // ff MoE bias
    struct ggml_tensor * ffn_gate_inp_b = nullptr;
    struct ggml_tensor * ffn_gate_exps_b = nullptr;
    struct ggml_tensor * ffn_down_exps_b = nullptr;
    struct ggml_tensor * ffn_up_exps_b = nullptr;
    struct ggml_tensor * ffn_up_gate_exps_b = nullptr;
    struct ggml_tensor * ffn_gate_exps_b_dup = nullptr;
    struct ggml_tensor * ffn_down_exps_b_dup = nullptr;
    struct ggml_tensor * ffn_up_exps_b_dup = nullptr;

    // ff shared expert (shexp)
    struct ggml_tensor * ffn_gate_inp_shexp = nullptr;
    struct ggml_tensor * ffn_gate_shexp = nullptr;
    struct ggml_tensor * ffn_down_shexp = nullptr;
    struct ggml_tensor * ffn_up_shexp = nullptr;

    llama_split_tensor split_ffn_up_shexp;
    llama_split_tensor split_ffn_gate_shexp;
    llama_split_tensor split_ffn_down_shexp;

    llama_split_tensor split_ffn_gate_inp_b;
    llama_split_tensor split_ffn_gate_exps_b;
    llama_split_tensor split_ffn_down_exps_b;
    llama_split_tensor split_ffn_up_exps_b;

    // ff bias
    struct ggml_tensor * ffn_gate_b = nullptr;
    struct ggml_tensor * ffn_down_b = nullptr; // b2
    struct ggml_tensor * ffn_up_b   = nullptr; // b3
    struct ggml_tensor * ffn_act = nullptr;
    struct ggml_tensor * ffn_exp_probs_b = nullptr;

    llama_split_tensor split_ffn_gate_b;
    llama_split_tensor split_ffn_down_b;
    llama_split_tensor split_ffn_up_b;
    llama_split_tensor split_ffn_act;
    llama_split_tensor split_ffn_exp_probs_b;

    // mamba proj
    struct ggml_tensor * ssm_in = nullptr;
    struct ggml_tensor * ssm_x = nullptr;
    struct ggml_tensor * ssm_dt = nullptr;
    struct ggml_tensor * ssm_out = nullptr;

    // mamba
    struct ggml_tensor * ssm_conv1d = nullptr;
    struct ggml_tensor * ssm_a = nullptr;
    struct ggml_tensor * ssm_d = nullptr;

    // mamba bias
    struct ggml_tensor * ssm_conv1d_b = nullptr;
    struct ggml_tensor * ssm_dt_b = nullptr;

    // long rope factors
    struct ggml_tensor * rope_long  = nullptr;
    struct ggml_tensor * rope_short = nullptr;
    struct ggml_tensor * rope_freqs = nullptr;

    llama_split_tensor split_rope_freqs;

    // bitnet scale
    struct ggml_tensor * wq_scale = nullptr;
    struct ggml_tensor * wk_scale = nullptr;
    struct ggml_tensor * wv_scale = nullptr;
    struct ggml_tensor * wo_scale = nullptr;
    struct ggml_tensor * ffn_gate_scale = nullptr;
    struct ggml_tensor * ffn_up_scale = nullptr;
    struct ggml_tensor * ffn_down_scale = nullptr;

    struct llama_layer_nextn nextn;

    std::unique_ptr<ggml_tensor> computed_wk_b;
    std::unique_ptr<ggml_tensor> computed_wv_b;
    std::unique_ptr<ggml_tensor> computed_wkv_b;
};

struct llama_lora_adapter;

struct rpc_device {
    std::string endpoint;
    uint32_t device;
};

struct llama_model {
    e_model     type  = MODEL_UNKNOWN;
    llm_arch    arch  = LLM_ARCH_UNKNOWN;
    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    struct ggml_tensor * tok_embd;
    struct ggml_tensor * type_embd;
    struct ggml_tensor * pos_embd;
    struct ggml_tensor * tok_norm;
    struct ggml_tensor * tok_norm_b;

    struct ggml_tensor * output_norm;
    struct ggml_tensor * output_norm_b;
    struct ggml_tensor * output;
    struct ggml_tensor * output_b;
    struct ggml_tensor * output_norm_enc;

    llama_split_tensor split_output;
    llama_split_tensor split_output_norm;

    std::vector<llama_layer> layers;

    llama_split_mode split_mode;
    int main_gpu;
    int max_gpu = 0; // max. number of GPUs to use per layer for aplit mode "graph"
    int n_gpu_layers;

    std::vector<rpc_device> rpc_servers;
    std::vector<int32_t> devices;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // layer -> buffer type mapping
    struct layer_buft {
        layer_buft() : buft_matrix(nullptr), buft(nullptr) {}
        layer_buft(ggml_backend_buffer_type_t matrix) : buft_matrix(matrix), buft(matrix) {}
        layer_buft(ggml_backend_buffer_type_t matrix, ggml_backend_buffer_type_t other) : buft_matrix(matrix), buft(other) {}

        ggml_backend_buffer_type_t buft_matrix; // matrices only - used by split buffers and backends that support only matrix multiplication
        ggml_backend_buffer_type_t buft;        // everything else
    };

    layer_buft buft_input;
    layer_buft buft_output;
    std::vector<layer_buft> buft_layer;

    // contexts where the model tensors metadata is stored
    std::vector<struct ggml_context *> ctxs;

    // the model memory buffers for the tensor data
    std::vector<ggml_backend_buffer_t> bufs;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    // keep track of loaded lora adapters
    std::set<llama_lora_adapter *> lora_adapters;

    bool tensor_overrides;

    ~llama_model();

    // Not actually needed, but left in place for now
    size_t max_nodes() const { return 65536; }

    bool has_tensor_overrides() const {
        return tensor_overrides;
    }

    void set_tensor_overrides(const llama_model_params& params);

    int device_count() const;
    ggml_backend_buffer_type_t default_buffer_type_offload(int device) const;

    std::vector<float> splits;
    ggml_backend_buffer_type_t split_buft = nullptr;
};

struct llama_lora_weight {
    struct ggml_tensor * a = nullptr;
    struct ggml_tensor * b = nullptr;
    llama_lora_weight() = default;
    llama_lora_weight(struct ggml_tensor * a, struct ggml_tensor * b): a(a), b(b) {}
};

struct llama_lora_adapter {
    llama_model * base_model;
    // map tensor name to lora_a_b
    std::unordered_map<std::string, struct llama_lora_weight> ab_map;
    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    float alpha;

    llama_lora_adapter(struct llama_model * base_model): base_model(base_model) {
        base_model->lora_adapters.insert(this);
    }

    llama_lora_weight * get_weight(struct ggml_tensor * w) {
        std::string name(w->name);
        auto pos = ab_map.find(name);
        if (ab_map.find(name) != ab_map.end()) {
            return &pos->second;
        }
        return nullptr;
    }

    ~llama_lora_adapter() {
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
        auto pos = base_model->lora_adapters.find(this);
        if (pos != base_model->lora_adapters.end()) {
            base_model->lora_adapters.erase(pos);
        }
    }
};

// helper to handle gguf constants
// usage:
//
//   const auto tn = LLM_TN(LLM_ARCH_LLAMA);
//
//   std::string name = tn(LLM_TENSOR_OUTPUT);                     -> "output"
//   std::string name = tn(LLM_TENSOR_TOKEN_EMBD, "bias");         -> "token_embd.bias"
//   std::string name = tn(LLM_TENSOR_ATTN_NORM, "weight", 3);     -> "blk.3.attn_norm.weight"
//
struct LLM_TN {
    LLM_TN(llm_arch arch) : arch(arch) {}

    llm_arch arch;

    std::string operator()(llm_tensor tensor) const;

    std::string operator()(llm_tensor tensor, const std::string & suffix) const;

    std::string operator()(llm_tensor tensor, int bid) const;

    std::string operator()(llm_tensor tensor, const std::string & suffix, int bid) const;

    std::string operator()(llm_tensor tensor, const std::string & suffix, int bid, int xid) const;
};

std::string llama_model_ftype_name(llama_ftype ftype);

const char * llama_model_type_name(e_model type);
