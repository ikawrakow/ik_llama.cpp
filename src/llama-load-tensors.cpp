#include "llama-model-loader.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "ggml.h"


#include <set>
#include <map>
#include <array>
#include <future>
#include <regex>
#include <unordered_set>

#define LLAMA_API_INTERNAL

struct create_tensors_helper : public create_tensors_helper_interface {

    create_tensors_helper(llama_model_loader & ml, llama_model & model);
    ~create_tensors_helper() = default;

    //virtual std::map<ggml_backend_buffer_type_t, int> & buft_layer_count_map() override {
    //    return buft_layer_count;
    //}

    virtual std::map<ggml_backend_buffer_type_t, ggml_context *> & get_ctx_map() override {
        return ctx_map;
    }

    virtual size_t get_ctx_size() const override { return ctx_size; }

    bool merge_qkv(const LLM_TN & tn, int i, int bias, bool ignore_attn_scale = false);

    bool merge_up_gate_exps(const LLM_TN & tn, int i, int bias);

    bool create_tensors() override;

    bool create_llama_tensors(const LLM_TN & tn);

    bool create_deci_tensors(const LLM_TN & tn);

    bool create_llama4_tensors(const LLM_TN & tn);

    bool create_grok_tensors(const LLM_TN & tn);

    bool create_dbrx_tensors(const LLM_TN & tn);

    bool create_baichuan_tensors(const LLM_TN & tn, bool with_ffn_norm = true);

    bool create_falcon_tensors(const LLM_TN & tn);

    bool create_starcoder_tensors(const LLM_TN & tn);

    bool create_bert_tensors(const LLM_TN & tn);

    bool create_jina_bert2_tensors(const LLM_TN & tn);

    bool create_bloom_tensors(const LLM_TN & tn);

    bool create_mpt_tensors(const LLM_TN & tn);

    bool create_stablelm_tensors(const LLM_TN & tn);

    bool create_qwen_tensors(const LLM_TN & tn);

    bool create_qwen2_tensors(const LLM_TN & tn);

    bool create_qwen2_moe_tensors(const LLM_TN & tn);

    bool create_qwen3_tensors(const LLM_TN & tn);

    bool create_qwen3_moe_tensors(const LLM_TN & tn);

    bool create_phi2_tensors(const LLM_TN & tn);

    bool create_phi3_tensors(const LLM_TN & tn);

    bool create_gpt2_tensors(const LLM_TN & tn);

    bool create_codeshell_tensors(const LLM_TN & tn);

    bool create_orion_tensors(const LLM_TN & tn);

    bool create_internlm_tensors(const LLM_TN & tn);

    bool create_gemma_tensors(const LLM_TN & tn, int version);

    bool create_starcoder2_tensors(const LLM_TN & tn);

    bool create_mamba_tensors(const LLM_TN & tn);

    bool create_xverse_tensors(const LLM_TN & tn);

    bool create_command_r_tensors(const LLM_TN & tn);

    bool create_olmo_tensors(const LLM_TN & tn);

    bool create_openelm_tensors(const LLM_TN & tn);

    bool create_gptneox_tensors(const LLM_TN & tn);

    bool create_arctix_tensors(const LLM_TN & tn);

    bool create_deepseek2_tensors(const LLM_TN & tn);

    bool create_glm4_tensors(const LLM_TN & tn);

    bool create_glm4_moe_tensors(const LLM_TN & tn);

    bool create_bitnet_tensors(const LLM_TN & tn);

    bool create_bitnet2_tensors(const LLM_TN & tn);

    bool create_t5_tensors(const LLM_TN & tn);

    bool create_tsencoder_tensors(const LLM_TN & tn);

    bool create_jais_tensors(const LLM_TN & tn);

    bool create_chatglm_tensors(const LLM_TN & tn);

    bool create_cohere2_tensors(const LLM_TN & tn);

    bool create_dots1_tensors(const LLM_TN & tn);

    bool create_ernie45_tensors(const LLM_TN & tn);

    bool create_hunyuan_tensors(const LLM_TN & tn);

    bool create_openai_moe_tensors(const LLM_TN & tn);

    bool create_bailingmoe2_tensors(const LLM_TN & tn);

    bool create_minimaxm2_tensors(const LLM_TN & tn);

    bool create_smollm3_tensors(const LLM_TN & tn);

    bool create_mimo2_tensors(const LLM_TN & tn);

    llama_model_loader & ml;
    llama_model        & model;

    ggml_tensor * create_tensor(ggml_context * ctx, const std::string & name, const std::vector<int64_t> & ne, int flags = 0,
            ggml_context ** actual_ctx = nullptr);

    ggml_context * get_context_for_tensor(ggml_context * ctx, const std::string & name);

    void create_default_embd_output(const LLM_TN & tn, int n_embd, int n_vocab, bool norm_bias);
    void create_embd_output(const LLM_TN & tn, int n_embd, int n_vocab, bool has_norm = true, bool use_ctx_split = false);

    void create_std_attn(int i, const LLM_TN & tn, llama_layer & layer, int n_embd, int n_embd_gqa, ggml_context * ctx_split);
    void create_std_ffn(int i, const LLM_TN & tn, llama_layer & layer, int n_ff, int n_embd, ggml_context * ctx_split);

    inline ggml_context * ctx_for_layer(int i) const {
        return ctx_map.at(model.buft_layer[i].buft);
    }
    inline ggml_context * ctx_for_layer_split(int i) const {
        return ctx_map.at(model.buft_layer[i].buft_matrix);
    }

    std::map<ggml_backend_buffer_type_t, int> buft_layer_count;
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    ggml_context * split_ctx = nullptr;
    size_t ctx_size;

    ggml_context * ctx_input;
    ggml_context * ctx_output;
    ggml_context * ctx_output_split;

    std::unordered_set<ggml_tensor *> split_tensors;

    inline ggml_context * ctx_for_buft(ggml_backend_buffer_type_t buft) {
        if (auto it = ctx_map.find(buft); it != ctx_map.end()) return it->second;

        ggml_init_params params = { /*.mem_size   =*/ ctx_size, /*.mem_buffer =*/ NULL, /*.no_alloc   =*/ true, };

        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error(format("failed to create ggml context"));
        }

        ctx_map[buft] = ctx;
        model.ctxs.emplace_back(ctx);

        return ctx;

    }
};

create_tensors_helper::create_tensors_helper(llama_model_loader & _ml, llama_model & _model) : ml(_ml), model(_model) {

    const int n_layer = model.hparams.n_layer;
    buft_layer_count[model.buft_input.buft]++;
    buft_layer_count[model.buft_input.buft_matrix]++;
    buft_layer_count[model.buft_output.buft]++;
    buft_layer_count[model.buft_output.buft_matrix]++;
    for (int i = 0; i < n_layer; ++i) {
        buft_layer_count[model.buft_layer[i].buft]++;
        buft_layer_count[model.buft_layer[i].buft_matrix]++;
    }

    auto n_tensors = ml.n_tensors;
    if (ml.merge_qkv) n_tensors += n_layer;
    if (ml.merge_up_gate_exps) n_tensors += n_layer;
    ctx_size = ggml_tensor_overhead()*(n_tensors + 1); // +1 for models where tok_embd is duplicated as output
    ctx_size += ggml_tensor_overhead()*n_layer*3;         // for moe merged tensors

    if (model.splits.size() > 1) {
        ctx_size += ggml_tensor_overhead()*n_layer*4;    // for KV cache
        ctx_size *= (model.splits.size() + 1);
    }

    for (auto & it : buft_layer_count) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error(format("failed to create context"));
        }
        ctx_map[it.first] = ctx;
        model.ctxs.push_back(ctx);
    }
    if (model.split_buft) {
        if (auto it = ctx_map.find(model.split_buft); it != ctx_map.end()) {
            split_ctx = it->second;
        }
    }
}

static std::vector<int> create_split(int nr, int granularity, const std::vector<float> & splits, const std::vector<size_t> & mem_used,
        bool verbose = false) {
    GGML_ASSERT(nr % granularity == 0);
    GGML_ASSERT(!splits.empty());
    if (granularity < 0) return std::vector<int>(splits.size(), nr);
    GGML_ASSERT(mem_used.size() == splits.size());
    size_t tot_memory_used = 1;
    for (auto & mem : mem_used) tot_memory_used += mem;
    int nchunk = nr / granularity;
    std::vector<int> result(splits.size());
    float last_split = 0;
    int sum = 0;
    if (verbose) printf("--- %s: %d chunks\n", __func__, nchunk);
    for (int i = 0; i < (int)splits.size(); ++i) {
        float p = splits[i] - last_split;
        float p0 = p;
        p += (p - 1.f*mem_used[i]/tot_memory_used);
        result[i] = roundf(p*nchunk);
        if (result[i] < 0) result[i] = 0;
        if (verbose) printf("i = %d, p0 = %g, p = %g, result = %d\n", i, p0, p, result[i]);
        sum += result[i];
        last_split = splits[i];
    }
    while (sum > nchunk) {
        last_split = 0;
        float best_err = -INFINITY;
        int ibest = -1;
        for (int i = 0; i < (int)splits.size(); ++i) {
            if (result[i] > 0) {
                float p = splits[i] - last_split;
                p += (p - 1.f*mem_used[i]/tot_memory_used);
                float n_want = p*nchunk;
                float err = result[i] - n_want;
                if (err > best_err) {
                    best_err = err; ibest = i;
                }
            }
            last_split = splits[i];
        }
        GGML_ASSERT(ibest >= 0 && result[ibest] > 0);
        --result[ibest];
        --sum;
    }
    while (sum < nchunk) {
        last_split = 0;
        float best_err = -INFINITY;
        int ibest = -1;
        for (int i = 0; i < (int)splits.size(); ++i) {
            float p = splits[i] - last_split;
            p += (p - 1.f*mem_used[i]/tot_memory_used);
            float n_want = p*nchunk;
            float err = n_want - result[i];
            if (err > best_err) {
                best_err = err; ibest = i;
            }
            last_split = splits[i];
        }
        GGML_ASSERT(ibest >= 0);
        ++result[ibest];
        ++sum;
    }
    for (auto & r : result) r *= granularity;
    return result;
}

ggml_context * create_tensors_helper::get_context_for_tensor(ggml_context * ctx, const std::string & name) {
    if (ml.tensor_buft_overrides) {
        for (const auto * overrides = ml.tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
            std::regex pattern(overrides->pattern);
            if (std::regex_search(name, pattern)) {
                LLAMA_LOG_INFO("Tensor %s buffer type overriden to %s\n", name.c_str(), ggml_backend_buft_name(overrides->buft));
                ctx = ctx_for_buft(overrides->buft);
                break;
            }
        }
    }
    return ctx;
}

ggml_tensor * create_tensors_helper::create_tensor(ggml_context * ctx, const std::string & name, const std::vector<int64_t> & ne,
        int flags, ggml_context ** actual_context) {
    ctx = get_context_for_tensor(ctx, name);
    if (actual_context) *actual_context = ctx;
    auto tensor = ml.create_tensor(ctx, name, ne, flags);
    if (tensor && ctx == split_ctx) {
        //printf("%s: adding tensor %s to split tensors\n", __func__, tensor->name);
        split_tensors.insert(tensor);
    }
    return tensor;
}

#define LOADING_PRELUDE \
        [[maybe_unused]] const auto & hparams = model.hparams; \
        [[maybe_unused]] const int64_t n_layer       = hparams.n_layer; \
        [[maybe_unused]] const int64_t n_head        = hparams.n_head(); \
        [[maybe_unused]] const int64_t n_head_kv     = hparams.n_head_kv(); \
        [[maybe_unused]] const int64_t n_embd        = hparams.n_embd / (hparams.n_deepstack_layers + 1); /* For Qwen3-VL we need to divide by the number of deepstack layers + 1, for other models n_deepstack_layers value is 0 by default */ \
        [[maybe_unused]] const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(); \
        [[maybe_unused]] const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(); \
        [[maybe_unused]] const int64_t n_embd_head_k = hparams.n_embd_head_k; \
        [[maybe_unused]] const int64_t n_embd_head_v = hparams.n_embd_head_v; \
        [[maybe_unused]] const int64_t n_ff          = hparams.n_ff(); \
        [[maybe_unused]] const int64_t n_embd_gqa    = n_embd_v_gqa; \
        [[maybe_unused]] const int64_t n_vocab       = hparams.n_vocab; \
        [[maybe_unused]] const int64_t n_vocab_type  = hparams.n_vocab_type; \
        [[maybe_unused]] const int64_t n_rot         = hparams.n_rot; \
        [[maybe_unused]] const int64_t n_expert      = hparams.n_expert; \
        [[maybe_unused]] const int64_t n_expert_used = hparams.n_expert_used; \
        [[maybe_unused]] const int64_t n_ctx_train   = hparams.n_ctx_train; \
        if (n_expert > 0 && hparams.n_expert_used == 0) { \
            throw std::runtime_error("model has expert layers but no expert layers are used"); \
        } \
        ctx_input        = ctx_map.at(model.buft_input.buft); \
        ctx_output       = ctx_map.at(model.buft_output.buft); \
        ctx_output_split = ctx_map.at(model.buft_output.buft_matrix); \
        model.layers.resize(n_layer);\
        bool use_mmap_buffer = true;


void create_tensors_helper::create_embd_output(const LLM_TN & tn, int n_embd, int n_vocab, bool has_norm, bool use_ctx_split) {
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    auto out_ctx = use_ctx_split ? ctx_output_split : ctx_output;
    if (has_norm) {
        model.output_norm = create_tensor(out_ctx, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
    }
    model.output = create_tensor(out_ctx, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (model.output == NULL) {
        model.output = create_tensor(out_ctx, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
    }
}

void create_tensors_helper::create_std_attn(int i, const LLM_TN & tn, llama_layer & layer, int n_embd, int n_embd_gqa, ggml_context * ctx_split) {
    layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
    layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
    layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
    layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
}

void create_tensors_helper::create_std_ffn(int i, const LLM_TN & tn, llama_layer & layer, int n_ff, int n_embd, ggml_context * ctx_split) {
    layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
    layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
    layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
}

bool create_tensors_helper::create_llama_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    create_embd_output(tn, n_embd, n_vocab, true, false); //true);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        use_mmap_buffer &= !merge_qkv(tn, i, 1);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

        // optional bias tensors
        layer.bo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? ctx_split : ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.rope_freqs = create_tensor(ctx_split, tn(LLM_TENSOR_ROPE_FREQS, "weight"), {n_embd/n_head/2}, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));

        if (n_expert == 0) {
            create_std_ffn(i, tn, layer, n_ff, n_embd, model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? ctx_split : ctx_layer);

            // optional MLP bias
            layer.ffn_gate_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_down_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_up_b   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        } else {
            layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});

            layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);
            if (layer.ffn_gate_exps) {
                layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert});
                layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert});
            } else {
                // merge split expert into a single tensor for compatibility with older models
                // requires disabling mmap
                use_mmap_buffer = false;

                ggml_type type_gate = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, 0).c_str())->type;
                ggml_type type_down = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, 0).c_str())->type;
                ggml_type type_up   = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, 0).c_str())->type;

                layer.ffn_gate_exps = ggml_new_tensor_3d(ctx_split, type_gate, n_embd,   n_ff, n_expert);
                layer.ffn_down_exps = ggml_new_tensor_3d(ctx_split, type_down,   n_ff, n_embd, n_expert);
                layer.ffn_up_exps   = ggml_new_tensor_3d(ctx_split, type_up,   n_embd,   n_ff, n_expert);

                ggml_set_name(layer.ffn_gate_exps, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i).c_str());
                ggml_set_name(layer.ffn_down_exps, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i).c_str());
                ggml_set_name(layer.ffn_up_exps,   tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i).c_str());

                for (uint32_t x = 0; x < n_expert; ++x) {
                    // the individual experts are loaded into a view of the merged tensor
                    ml.create_tensor_as_view(ctx_split, layer.ffn_gate_exps, tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, x), { n_embd, n_ff }, layer.ffn_gate_exps->nb[2]*x);
                    ml.create_tensor_as_view(ctx_split, layer.ffn_down_exps, tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, x), { n_ff, n_embd }, layer.ffn_down_exps->nb[2]*x);
                    ml.create_tensor_as_view(ctx_split, layer.ffn_up_exps,   tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, x), { n_embd, n_ff }, layer.ffn_up_exps->nb[2]*x);
                }
            }
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_deci_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];
        const int64_t n_embd_k_gqa  = hparams.n_embd_k_gqa(i);
        const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa(i);
        const int64_t n_embd_gqa    = hparams.n_embd_v_gqa(i);
        const int64_t n_ff          = hparams.n_ff(i);
        const int64_t n_head        = hparams.n_head(i);
        const int64_t n_head_kv     = hparams.n_head_kv(i);

        if (n_head_kv == 0 && n_head > 0) {
            // linear attention for DeciLMCausalModel
            layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
            layer.wo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        }
        else if (n_head_kv > 0) {
            layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

            layer.wq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head});
            layer.wk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
            layer.wv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
            layer.wo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});
        }

        // optional bias tensors


        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);
        if (n_ff > 0) {
            layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        }

        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight"), { n_rot/2 }, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight"), { n_rot/2 }, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
        }
        else {
            layer.rope_freqs = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FREQS, "weight"), {n_rot/2}, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
        }

        if (n_ff > 0) {
            create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
        }

        // optional MLP bias
        layer.ffn_gate_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_up_b   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_llama4_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    create_embd_output(tn, n_embd, n_vocab);

    GGML_ASSERT(hparams.n_moe_layer_step > 0 && "Llama 4 requires n_moe_layer_step > 0");
    for (int i = 0; i < n_layer; ++i) {
        bool is_moe_layer = (i + 1) % hparams.n_moe_layer_step == 0;
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        use_mmap_buffer &= !merge_qkv(tn, i, 0);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.rope_freqs = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2},
                llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));

        if (is_moe_layer) {
            int n_ff_exp = hparams.n_ff_exp;

            layer.ffn_gate_inp  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);

            // Shared expert
            const int64_t n_ff_shexp = n_ff_exp;
            layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {    n_embd, n_ff_shexp}, 0);
            layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd    }, 0);
            layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {    n_embd, n_ff_shexp}, 0);
        } else {
            create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_grok_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    if (n_expert == 0) {
        throw std::runtime_error("Grok model cannot have zero experts");
    }

    create_embd_output(tn, n_embd, n_vocab);

    const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff/* / n_expert_used*/; // grok-1 n_ff_exp == n_ff
    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.attn_out_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff,   n_embd }, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_up = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_gate_inp  = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert});
        layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);

        if (layer.ffn_gate_exps) {
            layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert});
            layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), { n_embd,   n_ff_exp, n_expert });
        } else {
            // merge split expert into a single tensor for compatibility with older models
            // requires disabling mmap
            use_mmap_buffer = false;

            ggml_type type_gate = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, 0).c_str())->type;
            ggml_type type_down = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, 0).c_str())->type;
            ggml_type type_up   = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, 0).c_str())->type;

            layer.ffn_gate_exps = ggml_new_tensor_3d(ctx_split, type_gate, n_embd,   n_ff, n_expert);
            layer.ffn_down_exps = ggml_new_tensor_3d(ctx_split, type_down,   n_ff, n_embd, n_expert);
            layer.ffn_up_exps   = ggml_new_tensor_3d(ctx_split, type_up,   n_embd,   n_ff, n_expert);

            ggml_set_name(layer.ffn_gate_exps, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i).c_str());
            ggml_set_name(layer.ffn_down_exps, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i).c_str());
            ggml_set_name(layer.ffn_up_exps,   tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i).c_str());

            for (uint32_t x = 0; x < n_expert; ++x) {
                // the individual experts are loaded into a view of the merged tensor
                ml.create_tensor_as_view(ctx_split, layer.ffn_gate_exps, tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, x), { n_embd, n_ff }, layer.ffn_gate_exps->nb[2]*x);
                ml.create_tensor_as_view(ctx_split, layer.ffn_down_exps, tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, x), { n_ff, n_embd }, layer.ffn_down_exps->nb[2]*x);
                ml.create_tensor_as_view(ctx_split, layer.ffn_up_exps,   tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, x), { n_embd, n_ff }, layer.ffn_up_exps->nb[2]*x);
            }
        }

        layer.ffn_post_norm = create_tensor(ctx_layer,tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), { n_embd }, llama_model_loader::TENSOR_NOT_REQUIRED);
        if (!layer.ffn_post_norm) {
            layer.ffn_post_norm = create_tensor(ctx_layer,tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), { n_embd }, 0);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_dbrx_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    if (n_expert == 0) {
        throw std::runtime_error("DBRX model cannot have zero experts");
    }

    create_default_embd_output(tn, n_embd, n_vocab, false);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.attn_out_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd});

        layer.ffn_gate_inp  = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert});
        layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff,   n_expert});
        layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff,   n_embd, n_expert});
        layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff,   n_expert});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_baichuan_tensors(const LLM_TN & tn, bool with_ffn_norm) {
    LOADING_PRELUDE
    create_default_embd_output(tn, n_embd, n_vocab, false);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        if (with_ffn_norm) {
            layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        }

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_falcon_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});

        model.output = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        if (!model.output) {
            model.output = create_tensor(ctx_output_split, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED); // needs to be on GPU
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.attn_norm_2   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_norm_2_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM_2, "bias", i),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_starcoder_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
    model.pos_embd = create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train});

    // output
    {
        model.output_norm   = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
        model.output        = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        if (!model.output) {
            // needs to be on GPU
            model.output = create_tensor(ctx_output_split, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }

    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i),   {n_embd, n_ff});
        layer.ffn_up_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP, "bias", i),     {n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_bert_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd     = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab});
    model.type_embd    = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_vocab_type});

    if (model.arch == LLM_ARCH_BERT) {
        model.pos_embd = create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,    "weight"), {n_embd, n_ctx_train});
    }

    model.tok_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd});
    model.tok_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd});

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        if (model.arch == LLM_ARCH_BERT) {
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
            layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i),   {n_embd});

            layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
            layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i),   {n_embd_gqa});

            layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
            layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i),   {n_embd_gqa});
        } else {
            layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        }

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {n_embd, n_embd});

        layer.attn_out_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd});
        layer.attn_out_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "bias", i),   {n_embd});

        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,        "weight", i), {n_embd, n_ff});
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN,      "weight", i), {n_ff, n_embd});

        if (model.arch == LLM_ARCH_BERT) {
            layer.bo         = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd});
            layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff});
            layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd});
        } else {
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff});
        }

        layer.layer_out_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd});
        layer.layer_out_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_LAYER_OUT_NORM, "bias", i),   {n_embd});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_jina_bert2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd  = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}); // word_embeddings
    model.type_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_TYPES, "weight"), {n_embd, n_vocab_type}); // token_type_embeddings

    model.tok_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd}); // LayerNorm
    model.tok_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd}); //LayerNorm bias

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i]; // JinaBertLayer

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd});
        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q, "bias", i),   {n_embd});

        layer.attn_q_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_q_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "bias",   i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa});
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K, "bias",   i), {n_embd_gqa});

        layer.attn_k_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_k_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "bias",   i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa});
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V, "bias",   i), {n_embd_gqa});

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}); //output_dens
        layer.bo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd}); //output_dens

        layer.attn_out_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}); //output_norm
        layer.attn_out_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT_NORM, "bias",   i), {n_embd});

        layer.attn_norm_2   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_norm_2_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM_2, "bias",   i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
        layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd});

        layer.layer_out_norm   = create_tensor(ctx_split, tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd});
        layer.layer_out_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_LAYER_OUT_NORM, "bias",   i), {n_embd});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_bloom_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd   = create_tensor(ctx_input,  tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab});
    model.tok_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), {n_embd});
    model.tok_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"),   {n_embd});

    // output
    {
        model.output_norm   = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
        model.output        = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias",   i), {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias",   i), {n_embd + 2*n_embd_gqa});

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias",   i), {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP, "bias",   i), {n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_mpt_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
    model.pos_embd = create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train}, llama_model_loader::TENSOR_NOT_REQUIRED);

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        model.output        = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        if (!model.output) {
            model.output = create_tensor(ctx_output_split, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED); // needs to be on GPU
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.attn_q_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_q_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "bias",   i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.attn_k_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_k_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "bias",   i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        // AWQ ScaleActivation layer
        layer.ffn_act = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_ACT, "scales", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_stablelm_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_default_embd_output(tn, n_embd, n_vocab, true);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm =   create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i), {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        // optional bias tensors, present in Stable LM 2 1.6B
        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);

        // optional q and k layernorms, present in StableLM 2 12B
        layer.attn_q_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k, n_head},    llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.attn_k_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k, n_head_kv}, llama_model_loader::TENSOR_NOT_REQUIRED);

        // optional FFN norm, not present in StableLM 2 12B which uses parallel residual
        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);

        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd*3});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd*3});
        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        create_std_ffn(i, tn, layer, n_ff/2, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        model.output_b    = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);

        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        // optional bias tensors
        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd});
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa});
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen2_moe_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        // optional bias tensors
        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd});
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa});
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});

        if (n_expert == 0) {
            throw std::runtime_error("n_expert must be > 0 for QWEN2MOE");
        }
        if (n_expert_used == 0) {
            throw std::runtime_error("n_expert_used must be > 0 for QWEN2MOE");
        }


        // MoE branch
        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

        layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert});
        layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert});
        layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert});

        // Shared expert branch
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        layer.ffn_gate_inp_shexp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), {n_embd});
        layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {    n_embd, n_ff_shexp});
        layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp,     n_embd});
        layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {    n_embd, n_ff_shexp});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen3_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        use_mmap_buffer &= !merge_qkv(tn, i, 0);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

        layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k});
        layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k});

        layer.ffn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_qwen3_moe_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output_split, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        use_mmap_buffer &= !merge_qkv(tn, i, 0);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

        layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k});
        layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k});

        auto ffn_ctx = model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? ctx_split : ctx_layer;
        layer.ffn_norm = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.ffn_gate_inp = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});

        if (n_expert == 0) {
            throw std::runtime_error("n_expert must be > 0 for QWEN3MOE");
        }
        if (n_expert_used == 0) {
            throw std::runtime_error("n_expert_used must be > 0 for QWEN3MOE");
        }

        // MoE branch
        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

        bool merged = ml.merge_up_gate_exps && merge_up_gate_exps(tn, i, 0);
        if (merged) {
            use_mmap_buffer = false;
        } else {
            layer.ffn_up_exps   = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert});
            layer.ffn_gate_exps = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert});
        }
        layer.ffn_down_exps = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_mimo2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab, true, false); //true);

    for (int i = 0; i < n_layer; ++i) {
        uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i);
        uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i);
        uint32_t n_head = hparams.n_head(i);
        //printf("Layer %2d: n_head = %u, n_embd_head_k = %d, n_embd_head_v = %d, n_embd_k_gqa = %d, n_embd_v_gqa = %d\n", i, n_head, (int)n_embd_head_k, (int)n_embd_head_v, n_embd_k_gqa, n_embd_v_gqa);

        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm  = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM,  "weight", i), {n_embd});
        layer.attn_sinks = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_v * n_head, n_embd }, 0);

        auto ffn_ctx = model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? ctx_split : ctx_layer;
        layer.ffn_norm = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        // non-MoE branch
        layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);

        // MoE branch
        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;
        layer.ffn_gate_inp  = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_gate_exps = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down_exps = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_up_exps   = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_exp_probs_b = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_phi2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm   = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
        model.output        = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
        model.output_b      = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);

        if (layer.wqkv == nullptr) {
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd});
            layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q, "bias", i),   {n_embd});

            layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa});
            layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K, "bias", i),   {n_embd_gqa});

            layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa});
            layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V, "bias", i),   {n_embd_gqa});
        }

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_phi3_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    const int64_t n_embd_head = n_embd / n_head;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd });

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd });

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd });

        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd });
        layer.ffn_up = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, 2 * n_ff });

        layer.rope_long  = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight"), { n_embd_head/2 }, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
        layer.rope_short = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight"), { n_embd_head/2 }, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_gpt2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});
    model.pos_embd = create_tensor(ctx_input, tn(LLM_TENSOR_POS_EMBD,   "weight"), {n_embd, n_ctx_train});

    // output
    {
        model.output_norm   = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
        model.output        = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_codeshell_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    create_default_embd_output(tn, n_embd, n_vocab, true);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "weight", i),   {n_embd, n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP, "bias", i),     {n_ff});
    }
    return use_mmap_buffer;
}

void create_tensors_helper::create_default_embd_output(const LLM_TN & tn, int n_embd, int n_vocab, bool norm_bias) {
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    model.output_norm   = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
    if (norm_bias) {
    model.output_norm_b = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
    }
    model.output        = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
}

bool create_tensors_helper::create_orion_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    create_default_embd_output(tn, n_embd, n_vocab, true);
    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_internlm_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    create_default_embd_output(tn, n_embd, n_vocab, false);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        // layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_gemma_tensors(const LLM_TN & tn, int version) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab},
            llama_model_loader::TENSOR_NOT_REQUIRED);
    if (!model.output) {
        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab},
                llama_model_loader::TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        use_mmap_buffer &= !merge_qkv(tn, i, 0, true);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});
        if (version > 1) {
            layer.attn_post_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd});
        }
        if (version > 2) {
            layer.attn_k_norm    = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head_k});
            layer.attn_q_norm    = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head_k});
        }

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
        if (version > 1) {
            layer.ffn_post_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd});
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_starcoder2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm   = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});

        model.output = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }

    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        // optional bias tensors
        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd});
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa});
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa});
        layer.bo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});

        // optional bias tensors
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP ,  "bias", i), {  n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_mamba_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t dt_rank = hparams.ssm_dt_rank;

    // only an expansion factor of 2 is supported for now
    GGML_ASSERT(2 * n_embd == d_inner);

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});

        model.output = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed, duplicated to allow offloading
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output_split, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        // norm
        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.ssm_in = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_IN, "weight", i), {n_embd, 2*d_inner});

        layer.ssm_conv1d = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {d_conv, d_inner});
        layer.ssm_conv1d_b = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_CONV1D, "bias", i), {d_inner});

        layer.ssm_x = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_X, "weight", i), {d_inner, dt_rank + 2*d_state});

        layer.ssm_dt = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_DT, "weight", i), {dt_rank, d_inner});
        layer.ssm_dt_b = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner});

        // no "weight" suffix for these
        layer.ssm_a = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_A, i), {d_state, d_inner});
        layer.ssm_d = create_tensor(ctx_layer, tn(LLM_TENSOR_SSM_D, i), {d_inner});

        // out_proj
        layer.ssm_out = create_tensor(ctx_split, tn(LLM_TENSOR_SSM_OUT, "weight", i), {d_inner, n_embd});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_xverse_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_command_r_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        // init output from the input tok embed
        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        if (n_layer >= 64){
            layer.attn_q_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k, n_head});
            layer.attn_k_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k, n_head_kv});
        }

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_olmo_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab, false);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        create_std_attn(i, tn, layer, n_embd, n_embd_gqa, ctx_split);
        create_std_ffn (i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_openelm_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        // init output from the input tok embed
        model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        const int64_t n_head      =   hparams.n_head(i);
        const int64_t n_head_qkv  = 2*hparams.n_head_kv(i) + n_head;
        const int64_t n_ff        =   hparams.n_ff(i);

        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_head_qkv*n_embd_head_k});
        layer.attn_q_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k});
        layer.attn_k_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head*n_embd_head_k, n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_gptneox_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm   = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
        model.output        = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "bias", i),   {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_arctix_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        create_std_attn(i, tn, layer, n_embd, n_embd_gqa, ctx_split);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        create_std_ffn (i, tn, layer, n_embd, n_embd, ctx_split);

        layer.ffn_gate_inp  = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert});
        layer.ffn_norm_exps = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM_EXPS, "weight", i), {n_embd});
        layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, false);
        layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert});
        layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_deepseek2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const bool is_lite = (hparams.n_layer == 27 || hparams.n_layer == 26);

    const int64_t n_embd_head_qk_rope = hparams.n_rot;
    const int64_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        if (!is_lite) {
            layer.attn_q_a_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank});
        }

        layer.attn_kv_a_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank});

        bool merged = false;
        if (ml.merge_qkv) {
            auto q_name = is_lite ? tn(LLM_TENSOR_ATTN_Q, "weight", i) : tn(LLM_TENSOR_ATTN_Q_A, "weight", i);
            auto k_name = tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i);
            auto wq = ml.require_tensor_meta(q_name.c_str());
            auto wk = ml.require_tensor_meta(k_name.c_str());
            GGML_ASSERT(wq && wk);
            if (wq->type == wk->type) {
                GGML_ASSERT(wq->ne[0] == wk->ne[0]);
                layer.wkq_a_mqa = ggml_new_tensor_2d(ctx_split, wq->type, wq->ne[0], wq->ne[1] + wk->ne[1]);
                snprintf(layer.wkq_a_mqa->name, GGML_MAX_NAME, "blk.%d.attn_qk_a_mqa.weight", i);
                if (is_lite) {
                    layer.wq = ml.create_tensor_as_view(ctx_split, layer.wkq_a_mqa, q_name.c_str(), { wq->ne[0], wq->ne[1] }, 0);
                } else {
                    layer.wq_a = ml.create_tensor_as_view(ctx_split, layer.wkq_a_mqa, q_name.c_str(), { wq->ne[0], wq->ne[1] }, 0);
                }
                layer.wkv_a_mqa = ml.create_tensor_as_view(ctx_split, layer.wkq_a_mqa, k_name.c_str(), { wk->ne[0], wk->ne[1] }, wq->ne[1]*wq->nb[1]);
                merged = true;
                use_mmap_buffer = false;
                printf("============== Merged %s (%ld x %ld) and %s (%ld x %ld)\n", q_name.c_str(),
                        wq->ne[0], wq->ne[1], k_name.c_str(), wk->ne[0], wk->ne[1]);
            }
        }

        if (!is_lite) {
            if (!merged) {
                layer.wq_a = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank});
            }
            layer.wq_b = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k});
        } else if (!merged) {
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_k_gqa});
        }

        if (!merged) {
            layer.wkv_a_mqa = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i),{n_embd, kv_lora_rank + (n_embd_head_qk_rope)});
        }

        layer.wkv_b     = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_KV_B,     "weight", i),
                {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)}, llama_model_loader::TENSOR_NOT_REQUIRED);
        if (!layer.wkv_b) {
            // Incompatible mainline model. Let's see if we can still load it
            layer.wk_b = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, 0);
            layer.wv_b = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v, n_head}, 0);

        } else {
            layer.wk_b      = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_B,      "weight", i), {n_embd_head_qk_nope, n_head * kv_lora_rank}, 1);
            layer.wv_b      = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V_B,      "weight", i), {kv_lora_rank, n_head * n_embd_head_v}, 1);
        }
        layer.wo        = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT,      "weight", i), {              n_head * (                      n_embd_head_v), n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
            layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
            layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});
        } else {
            layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});
            layer.ffn_exp_probs_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, 1);

            GGML_ASSERT(n_expert      > 0);
            GGML_ASSERT(n_expert_used > 0);

            // MoE branch
            layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert});
            layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert});
            layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert});

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared});
            layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd});
            layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared});
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_glm4_moe_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t n_expert_shared = hparams.n_expert_shared;

    GGML_ASSERT(hparams.n_expert > 0 && "n_expert must be > 0 for GLM4_MOE MoE layers");
    GGML_ASSERT(hparams.n_expert_used > 0 && "n_expert_used must be > 0 for GLM4_MOE MoE layers");

    create_embd_output(tn, n_embd, n_vocab, true, false); //true);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        int flags = 0;
        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
            // skip all tensors in the NextN layers
            flags |= llama_model_loader::TENSOR_SKIP;
        }

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);

        // GLM-style attention with bias terms
        if (!flags) {
            use_mmap_buffer &= !merge_qkv(tn, i, 2);
        } else {
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, flags);
            layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, flags);
            layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, flags);
            layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd_head_k * n_head }, flags);
            layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_k_gqa }, flags);
            layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_v_gqa }, flags);
        }

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, flags);

        // K/Q norm tensors (optional for GLM-4.5 355B variant)
        layer.attn_q_norm = create_tensor(ctx_split,
                tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, llama_model_loader::TENSOR_NOT_REQUIRED | flags);
        layer.attn_k_norm = create_tensor(ctx_split,
                tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, llama_model_loader::TENSOR_NOT_REQUIRED | flags);

        auto ffn_ctx = model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? ctx_split : ctx_layer;

        // Why are we adding an additional tensor type?
        // attn_post_norm is the exact same thing as ffn_norm
        //layer.attn_post_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, flags);
        layer.ffn_norm = create_tensor(ffn_ctx, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, flags);

        // Check if this layer uses MoE or dense FFN based on n_layer_dense_lead
        // GLM 4.5 uses hybrid architecture: layer 0 is dense, layers 1+ are MoE
        const bool use_moe = (static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead);

        if (use_moe) {
            // MoE layers
            layer.ffn_gate_inp = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, flags);
            // gate bias
            layer.ffn_exp_probs_b = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), { n_expert }, flags);

            // MoE branch
            const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

            layer.ffn_gate_exps = create_tensor(ffn_ctx,
                    tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, flags);
            layer.ffn_down_exps = create_tensor(ffn_ctx,
                    tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, flags);
            layer.ffn_up_exps = create_tensor(ffn_ctx,
                    tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, flags);

            // Shared expert
            if (n_expert_shared > 0) {
                const int64_t n_ff_shexp = n_ff_exp * n_expert_shared;
                layer.ffn_gate_shexp     = create_tensor(ffn_ctx,
                        tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_shexp }, flags);
                layer.ffn_down_shexp = create_tensor(ffn_ctx,
                        tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_shexp, n_embd }, flags);
                layer.ffn_up_shexp = create_tensor(ffn_ctx,
                        tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), { n_embd, n_ff_shexp }, flags);
            }
        } else {
            // Dense layers (first k layers) - GLM uses separate gate/up projections
            layer.ffn_gate = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, flags);
            layer.ffn_down = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, flags);
            layer.ffn_up   = create_tensor(ffn_ctx, tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, flags);
        }
        // --- NextN / MTP tensors (preserved but unused), on the final layer ---
        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
            const int final_layer = n_layer - 1;
            // EH_PROJ: [2*embd, embd]
            layer.nextn.eh_proj          = create_tensor(ctx_for_layer(final_layer),
                    tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", final_layer),
                    { 2*n_embd, n_embd },
                    flags);
            // EMBED_TOKENS: [embd, vocab]
            layer.nextn.embed_tokens     = create_tensor(ctx_for_layer(final_layer),
                    tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", final_layer),
                    { n_embd, n_vocab },
                    flags | llama_model_loader::TENSOR_NOT_REQUIRED);
            // ENORM, HNORM: [embd]
            layer.nextn.enorm            = create_tensor(ctx_for_layer(final_layer),
                    tn(LLM_TENSOR_NEXTN_ENORM, "weight", final_layer),
                    { n_embd },
                    flags);
            layer.nextn.hnorm            = create_tensor(ctx_for_layer(final_layer),
                    tn(LLM_TENSOR_NEXTN_HNORM, "weight", final_layer),
                    { n_embd },
                    flags);
            // SHARED_HEAD_HEAD: [embd, vocab]
            layer.nextn.shared_head_head = create_tensor(ctx_for_layer(final_layer),
                    tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", final_layer),
                    { n_embd, n_vocab },
                    flags | llama_model_loader::TENSOR_NOT_REQUIRED);
            // SHARED_HEAD_NORM: [embd]
            layer.nextn.shared_head_norm = create_tensor(ctx_for_layer(final_layer),
                    tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", final_layer),
                    { n_embd },
                    flags | llama_model_loader::TENSOR_NOT_REQUIRED);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_bitnet_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED); // same as tok_embd, duplicated to allow offloading
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});
        layer.attn_sub_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_SUB_NORM, "weight", i), {n_embd});

        layer.wq       = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd});
        layer.wq_scale = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.wk       = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_gqa});
        layer.wk_scale = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.wv       = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_gqa});
        layer.wv_scale = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.wo       = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.wo_scale = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_sub_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_SUB_NORM, "weight", i), {n_ff});

        layer.ffn_gate       = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff});
        layer.ffn_gate_scale = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down       = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_scale = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_up         = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
        layer.ffn_up_scale   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP, "scale", i), {1},  llama_model_loader::TENSOR_NOT_REQUIRED);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_bitnet2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output      = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);

        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.attn_sub_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_SUB_NORM, "weight", i), {n_embd});
        layer.ffn_sub_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_SUB_NORM, "weight", i), {n_ff});

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd});

        // optional bias tensors
        layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd},     llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.rope_freqs = create_tensor(ctx_layer, tn(LLM_TENSOR_ROPE_FREQS, "weight"), {n_rot/2}, llama_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llama_model_loader::TENSOR_DUPLICATED : 0));

        if (n_expert == 0) {
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff});
            layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd});
            layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff});

            // optional MLP bias
            layer.ffn_gate_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        } else {
            layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert});

            layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);
            if (layer.ffn_gate_exps) {
                layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert});
                layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert});
            } else {
                // merge split expert into a single tensor for compatibility with older models
                // requires disabling mmap
                use_mmap_buffer = false;

                ggml_type type_gate = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, 0).c_str())->type;
                ggml_type type_down = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, 0).c_str())->type;
                ggml_type type_up   = ml.require_tensor_meta(tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, 0).c_str())->type;

                layer.ffn_gate_exps = ggml_new_tensor_3d(ctx_split, type_gate, n_embd,   n_ff, n_expert);
                layer.ffn_down_exps = ggml_new_tensor_3d(ctx_split, type_down,   n_ff, n_embd, n_expert);
                layer.ffn_up_exps   = ggml_new_tensor_3d(ctx_split, type_up,   n_embd,   n_ff, n_expert);

                ggml_set_name(layer.ffn_gate_exps, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i).c_str());
                ggml_set_name(layer.ffn_down_exps, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i).c_str());
                ggml_set_name(layer.ffn_up_exps,   tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i).c_str());

                for (uint32_t x = 0; x < n_expert; ++x) {
                    // the individual experts are loaded into a view of the merged tensor
                    ml.create_tensor_as_view(ctx_split, layer.ffn_gate_exps, tn(LLM_TENSOR_FFN_GATE_EXP, "weight", i, x), { n_embd, n_ff }, layer.ffn_gate_exps->nb[2]*x);
                    ml.create_tensor_as_view(ctx_split, layer.ffn_down_exps, tn(LLM_TENSOR_FFN_DOWN_EXP, "weight", i, x), { n_ff, n_embd }, layer.ffn_down_exps->nb[2]*x);
                    ml.create_tensor_as_view(ctx_split, layer.ffn_up_exps,   tn(LLM_TENSOR_FFN_UP_EXP,   "weight", i, x), { n_embd, n_ff }, layer.ffn_up_exps->nb[2]*x);
                }
            }
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_t5_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    const auto n_rel_attn_bkts = hparams.n_rel_attn_bkts;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm_enc = create_tensor(ctx_output, tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm     = create_tensor(ctx_output, tn(LLM_TENSOR_DEC_OUTPUT_NORM, "weight"), {n_embd});

        model.output = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm_enc  = create_tensor(ctx_layer, tn(LLM_TENSOR_ENC_ATTN_NORM,  "weight", i), {n_embd});
        layer.attn_rel_b_enc = create_tensor(ctx_input, tn(LLM_TENSOR_ENC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wq_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wk_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wv_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
        layer.wo_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd});

        layer.ffn_norm_enc = create_tensor(ctx_layer, tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_gate_enc = create_tensor(ctx_layer, tn(LLM_TENSOR_ENC_FFN_GATE, "weight", i), {n_embd,   n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), {  n_ff, n_embd});
        layer.ffn_up_enc   = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_FFN_UP,   "weight", i), {n_embd,   n_ff});

        layer.attn_norm  = create_tensor(ctx_layer, tn(LLM_TENSOR_DEC_ATTN_NORM,  "weight", i), {n_embd});
        layer.attn_rel_b = create_tensor(ctx_input, tn(LLM_TENSOR_DEC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd});

        layer.attn_norm_cross  = create_tensor(ctx_layer, tn(LLM_TENSOR_DEC_CROSS_ATTN_NORM,  "weight", i), {n_embd});
        // this tensor seems to be unused in HF transformers implementation
        layer.attn_rel_b_cross = create_tensor(ctx_input, tn(LLM_TENSOR_DEC_CROSS_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wq_cross = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_CROSS_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wk_cross = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_CROSS_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wv_cross = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_CROSS_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
        layer.wo_cross = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_CROSS_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd});

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_DEC_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_gate = create_tensor(ctx_layer, tn(LLM_TENSOR_DEC_FFN_GATE, "weight", i), {n_embd,   n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_FFN_DOWN, "weight", i), {  n_ff, n_embd});
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_DEC_FFN_UP,   "weight", i), {n_embd,   n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_tsencoder_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    const auto n_rel_attn_bkts = hparams.n_rel_attn_bkts;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm_enc = create_tensor(ctx_output, tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd});
        model.output = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed
        if (model.output == NULL) {
            model.output = create_tensor(ctx_output, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
        }
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm_enc  = create_tensor(ctx_layer, tn(LLM_TENSOR_ENC_ATTN_NORM,  "weight", i), {n_embd});
        layer.attn_rel_b_enc = create_tensor(ctx_input, tn(LLM_TENSOR_ENC_ATTN_REL_B, "weight", i), {n_head, n_rel_attn_bkts}, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.wq_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_Q,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wk_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa});
        layer.wv_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa});
        layer.wo_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_v_gqa, n_embd});

        layer.ffn_norm_enc = create_tensor(ctx_layer, tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_gate_enc = create_tensor(ctx_layer, tn(LLM_TENSOR_ENC_FFN_GATE, "weight", i), {n_embd,   n_ff}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.ffn_down_enc = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), {  n_ff, n_embd});
        layer.ffn_up_enc   = create_tensor(ctx_split, tn(LLM_TENSOR_ENC_FFN_UP,   "weight", i), {n_embd,   n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_jais_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab});

    // Output
    {
        model.output_norm   = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output_norm_b = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd});
        model.output        = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "weight", i), {n_embd});
        layer.attn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM,   "bias", i),   {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa});

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});
        layer.bo = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_OUT, "bias", i),   {n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});
        layer.ffn_norm_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "bias", i),   {n_embd});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
        layer.ffn_down_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_DOWN, "bias", i),   {n_embd});

        layer.ffn_gate   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE,   "weight", i), {n_embd, n_ff});
        layer.ffn_gate_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE,   "bias", i),   {n_ff});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff});
        layer.ffn_up_b   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_UP,   "bias", i),   {n_ff});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_chatglm_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd   = create_tensor(ctx_input,  tn(LLM_TENSOR_TOKEN_EMBD,      "weight"), {n_embd, n_vocab});

    // output
    {
        model.output_norm   = create_tensor(ctx_output,       tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd});
        model.output        = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab});
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd});

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + (hparams.n_embd_head_k << 2)});
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + (hparams.n_embd_head_k << 2)});

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd});

        layer.ffn_norm   = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd});

        layer.ffn_up     = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff * 2});

        layer.ffn_down   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd});
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_cohere2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab, true, false); //true);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = model.layers[i];
        ggml_context * ctx_split = ctx_for_layer_split(i);

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        create_std_attn(i, tn, layer, n_embd, n_embd_gqa, ctx_split);
        create_std_ffn (i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_glm4_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE
    model.tok_embd   = create_tensor(ctx_input,  tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output      = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (model.output == NULL) {
        model.output = create_tensor(ctx_output_split, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, llama_model_loader::TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bqkv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_QKV, "bias", i),   {n_embd + 2*n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);

        if (layer.wqkv == nullptr) {
            layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
            layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
            layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
            layer.bq = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {n_embd}, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.bk = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {n_embd_gqa}, llama_model_loader::TENSOR_NOT_REQUIRED);
        }

        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.attn_post_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff * 2}, 0);

        layer.ffn_post_norm  = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_dots1_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output      = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = model.layers[i];
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, llama_model_loader::TENSOR_NOT_REQUIRED);
            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }
            // MoE branch
            layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_bailingmoe2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output      = create_tensor(ctx_output_split, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    GGML_ASSERT(n_expert > 0 && "n_expert must be > 0 for bailingmoe2");
    GGML_ASSERT(n_expert_used > 0 && "n_expert_used must be > 0 for bailingmoe2");

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = model.layers[i];
        ggml_context * ctx_layer = ctx_for_layer(i);
        ggml_context * ctx_split = ctx_for_layer_split(i);

        int flags = 0;
        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
            // skip all tensors in the NextN layers
            flags |= llama_model_loader::TENSOR_SKIP;
        }

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);

        layer.wqkv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, flags);
        layer.wo   = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, flags);

        layer.attn_q_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, flags);
        layer.attn_k_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, flags);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        if (static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead) { // MoE layers
            const int64_t n_ff_shexp = (hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff_exp) * n_expert_shared;

            layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert},
                    llama_model_loader::TENSOR_NOT_REQUIRED | flags);

            layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, flags);
            layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, flags);
        } else { // Dense layers
            layer.ffn_gate = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
            layer.ffn_down = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
        }

        // NextN/MTP tensors (preserved but unused) - conditionally load for last nextn_predict_layers
        if (hparams.nextn_predict_layers > 0 && static_cast<uint32_t>(i) >= n_layer - hparams.nextn_predict_layers) {
            layer.nextn.eh_proj          = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, flags);
            layer.nextn.embed_tokens     = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab },
                    llama_model_loader::TENSOR_NOT_REQUIRED | flags);
            layer.nextn.enorm            = create_tensor(ctx_layer, tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
            layer.nextn.hnorm            = create_tensor(ctx_layer, tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);
            layer.nextn.shared_head_head = create_tensor(ctx_split, tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, llama_model_loader::TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, llama_model_loader::TENSOR_NOT_REQUIRED | flags);
            layer.layer_out_norm         = create_tensor(ctx_layer, tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, flags);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_ernie45_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        auto& layer = model.layers[i];
        ggml_context* ctx_split = ctx_for_layer_split(i);

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        // optional bias tensors
        layer.bq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd }, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_gqa }, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_gqa }, llama_model_loader::TENSOR_NOT_REQUIRED);
        layer.bo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, llama_model_loader::TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

        if (model.arch == LLM_ARCH_ERNIE4_5_MOE && static_cast<uint32_t>(i) >= hparams.n_layer_dense_lead) { // MoE layers
            int n_ff_exp = hparams.n_ff_exp;

            layer.ffn_gate_inp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);
            layer.ffn_exp_probs_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), { n_expert }, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd,   n_ff_exp, n_expert }, llama_model_loader::TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp, n_embd, n_expert }, 0);
            layer.ffn_up_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd,   n_ff_exp, n_expert }, 0);

            // Shared expert (if present)
            if (hparams.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, hparams.n_ff_shexp }, 0);
                layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { hparams.n_ff_shexp, n_embd }, 0);
                layer.ffn_up_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), { n_embd, hparams.n_ff_shexp }, 0);
            }
        }
        else { // Dense layers
            create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_hunyuan_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_split = ctx_for_layer_split(i);

        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_inp  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
        layer.ffn_up_shexp   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
        layer.ffn_down_shexp = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_openai_moe_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    const int64_t n_ff_exp = hparams.n_ff_exp;

    model.tok_embd = create_tensor(ctx_input, tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    model.output_norm = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    model.output      = create_tensor(ctx_output, tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context * ctx_split = ctx_for_layer_split(i);
        ggml_context * ctx_layer = ctx_for_layer(i);
        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_norm  = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        use_mmap_buffer &= !merge_qkv(tn, i, 2);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_rot, n_embd}, 0);
        layer.bo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd}, 0);

        layer.attn_sinks = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, 0);

        ggml_context *ctx_ffn_gate, *ctx_ffn_up, *ctx_ffn_down;
        layer.ffn_gate_inp  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {  n_embd, n_expert}, 0);
        bool merged = ml.merge_up_gate_exps && merge_up_gate_exps(tn, i, 2);
        use_mmap_buffer &= !merged;
        if (merged) {
            ctx_ffn_gate = ctx_ffn_up = ctx_split;
        } else {
            layer.ffn_up_exps   = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i),
                    {  n_embd, n_ff_exp, n_expert}, 0, &ctx_ffn_up);
            layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i),
                    {  n_embd, n_ff_exp, n_expert}, 0, &ctx_ffn_gate);
        }
        layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i),
                {n_ff_exp, n_embd, n_expert}, 0, &ctx_ffn_down);

        // bias
        layer.ffn_gate_inp_b  = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_INP,  "bias", i), {n_expert}, 0);
        ggml_context *ctx_ffn_gate_b, *ctx_ffn_up_b, *ctx_ffn_down_b;
        auto ctx_gate_b = ctx_ffn_gate == ctx_split ? ctx_split : ctx_layer;
        auto ctx_down_b = ctx_ffn_down == ctx_split ? ctx_split : ctx_layer;
        auto ctx_up_b   = ctx_ffn_up   == ctx_split ? ctx_split : ctx_layer;
        if (!merged) {
            layer.ffn_up_exps_b   = create_tensor(ctx_up_b,   tn(LLM_TENSOR_FFN_UP_EXPS,   "bias", i), {n_ff_exp, n_expert}, 0, &ctx_ffn_up_b);
            layer.ffn_gate_exps_b = create_tensor(ctx_gate_b, tn(LLM_TENSOR_FFN_GATE_EXPS, "bias", i), {n_ff_exp, n_expert}, 0, &ctx_ffn_gate_b);
        }
        layer.ffn_down_exps_b = create_tensor(ctx_down_b, tn(LLM_TENSOR_FFN_DOWN_EXPS, "bias", i), {  n_embd, n_expert}, 0, &ctx_ffn_down_b);

        if (!merged && ctx_ffn_gate_b != ctx_ffn_gate) {
            layer.ffn_gate_exps_b_dup = create_tensor(ctx_ffn_gate, tn(LLM_TENSOR_FFN_GATE_EXPS, "bias", i), {n_ff_exp, n_expert},
                    llama_model_loader::TENSOR_DUPLICATED);
        }
        if (!merged && ctx_ffn_up_b != ctx_ffn_up) {
            layer.ffn_up_exps_b_dup = create_tensor(ctx_ffn_up, tn(LLM_TENSOR_FFN_UP_EXPS,   "bias", i), {n_ff_exp, n_expert},
                    llama_model_loader::TENSOR_DUPLICATED);
        }
        if (ctx_ffn_down_b != ctx_ffn_down) {
            layer.ffn_down_exps_b_dup = create_tensor(ctx_ffn_down, tn(LLM_TENSOR_FFN_DOWN_EXPS, "bias", i), {  n_embd, n_expert},
                    llama_model_loader::TENSOR_DUPLICATED);
        }
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_minimaxm2_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context* ctx_layer = ctx_for_layer(i);
        ggml_context* ctx_split = ctx_for_layer_split(i);
        auto& layer = model.layers[i];

        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
        layer.attn_q_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k * n_head }, 0);
        layer.attn_k_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_k_gqa }, 0);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

        layer.ffn_gate_inp = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);
        layer.ffn_gate_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff,   n_expert }, 0);
        layer.ffn_down_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff,   n_embd, n_expert }, 0);
        layer.ffn_up_exps = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd, n_ff,   n_expert }, 0);
        layer.ffn_exp_probs_b = create_tensor(ctx_split, tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), { n_expert }, 0);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::create_smollm3_tensors(const LLM_TN & tn) {
    LOADING_PRELUDE

    create_embd_output(tn, n_embd, n_vocab);

    for (int i = 0; i < n_layer; ++i) {
        ggml_context* ctx_layer = ctx_for_layer(i);
        ggml_context* ctx_split = ctx_for_layer_split(i);
        auto & layer = model.layers[i];

        layer.attn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        use_mmap_buffer &= !merge_qkv(tn, i, 0);

        layer.wo = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.ffn_norm = create_tensor(ctx_layer, tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        create_std_ffn(i, tn, layer, n_ff, n_embd, ctx_split);
    }
    return use_mmap_buffer;
}

bool create_tensors_helper::merge_up_gate_exps(const LLM_TN & tn, int i, int bias) {
    ggml_context * ctx_split = ctx_for_layer_split(i);

    auto & layer = model.layers[i];

    auto u_name = tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i);
    auto g_name = tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i);
    auto u_meta = ml.require_tensor_meta(u_name.c_str());
    auto g_meta = ml.require_tensor_meta(g_name.c_str());

    if (u_meta->type != g_meta->type || u_meta->ne[0] != g_meta->ne[0] || u_meta->ne[2] != g_meta->ne[2]) {
        printf("%s: not merging because up/fate meta info is different\n", __func__);
        return false;
    }

    auto u_ctx = get_context_for_tensor(ctx_split, u_name);
    auto g_ctx = get_context_for_tensor(ctx_split, g_name);

    if (u_ctx != g_ctx) {
        printf("%s: not merging because of context\n", __func__);
        return false;
    }

    if (bias && (u_ctx != ctx_split || g_ctx != ctx_split)) {
        printf("%s: not merging because of context\n", __func__);
        return false;
    }

    printf("%s: merging up/gate in layer %d\n", __func__, i);

    layer.ffn_up_gate_exps = ggml_new_tensor_3d(u_ctx, u_meta->type, u_meta->ne[0], u_meta->ne[1] + g_meta->ne[1], u_meta->ne[2]);
    snprintf(layer.ffn_up_gate_exps->name, GGML_MAX_NAME, "blk.%d.ffn_up_gate_exps.weight", i);
    layer.ffn_up_exps   = ml.create_tensor_as_view(u_ctx, layer.ffn_up_gate_exps, u_name.c_str(),
            { u_meta->ne[0], u_meta->ne[1], u_meta->ne[2] }, 0);
    layer.ffn_gate_exps = ml.create_tensor_as_view(u_ctx, layer.ffn_up_gate_exps, g_name.c_str(),
            { g_meta->ne[0], g_meta->ne[1], g_meta->ne[2] }, ggml_nbytes(layer.ffn_up_exps) ); //u_meta->ne[1]*u_meta->nb[1] );

    if (!bias) return true;

    auto u_name_b = tn(LLM_TENSOR_FFN_UP_EXPS,   "bias", i);
    auto g_name_b = tn(LLM_TENSOR_FFN_GATE_EXPS, "bias", i);
    auto u_meta_b = ml.get_tensor_meta(u_name_b.c_str());
    auto g_meta_b = ml.get_tensor_meta(g_name_b.c_str());
    if (bias == 2) {
        GGML_ASSERT(u_meta_b && g_meta_b);
        GGML_ASSERT(u_meta_b->type == g_meta_b->type);
        GGML_ASSERT(u_meta_b->ne[1] == g_meta_b->ne[1]);
    } else {
        GGML_ASSERT(!u_meta_b && !g_meta_b);
        return true;
    }

    GGML_ASSERT(u_meta->ne[1] == u_meta_b->ne[0]);
    GGML_ASSERT(g_meta->ne[1] == g_meta_b->ne[0]);

    layer.ffn_up_gate_exps_b = ggml_new_tensor_2d(ctx_split, u_meta_b->type, u_meta_b->ne[0] + g_meta_b->ne[0], u_meta->ne[1]);
    snprintf(layer.ffn_up_gate_exps_b->name, GGML_MAX_NAME, "blk.%d.ffn_up_gate_exps.bias", i);
    layer.ffn_up_exps_b   = ml.create_tensor_as_view(ctx_split, layer.ffn_up_gate_exps_b, u_name_b.c_str(),
            { u_meta_b->ne[0], u_meta_b->ne[1] }, 0);
    layer.ffn_gate_exps_b = ml.create_tensor_as_view(ctx_split, layer.ffn_up_gate_exps_b, g_name_b.c_str(),
            { g_meta_b->ne[0], g_meta_b->ne[1] }, ggml_nbytes(layer.ffn_up_exps_b) ); //u_meta->nb[1]);

    return true;
}

bool create_tensors_helper::merge_qkv(const LLM_TN & tn, int i, int bias, bool ignore_attn_scale) {
    auto& hparams = model.hparams;
    const int64_t n_head        = hparams.n_head();
    const int64_t n_head_kv     = hparams.n_head_kv();
    const int64_t n_embd        = hparams.n_embd / (hparams.n_deepstack_layers + 1); // For Qwen3-VL we need to divide by the number of deepstack layers + 1, for other models n_deepstack_layers value is 0 by default
    const int64_t n_embd_v_gqa  = hparams.n_embd_v_gqa();
    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    const int64_t n_embd_gqa    = n_embd_v_gqa;

    ggml_context * ctx_layer = ctx_for_layer(i);
    ggml_context * ctx_split = ctx_for_layer_split(i);

    auto & layer = model.layers[i];

    auto wq_name = tn(LLM_TENSOR_ATTN_Q, "weight", i);
    auto wk_name = tn(LLM_TENSOR_ATTN_K, "weight", i);
    auto wv_name = tn(LLM_TENSOR_ATTN_V, "weight", i);
    auto wq = ml.require_tensor_meta(wq_name.c_str());
    auto wk = ml.require_tensor_meta(wk_name.c_str());
    auto wv = ml.require_tensor_meta(wv_name.c_str());
    GGML_ASSERT(wq && wk && wv);

    bool fused_qkv = false;
    if (ml.merge_qkv && wq->type == wk->type && wq->type == wv->type && (ignore_attn_scale || hparams.f_attention_scale == 0.0f)) {
        GGML_ASSERT(wq->ne[0] == n_embd && wq->ne[1] == n_head * n_embd_head_k);
        GGML_ASSERT(wk->ne[0] == n_embd && wk->ne[1] == n_embd_gqa);
        GGML_ASSERT(wv->ne[0] == n_embd && wv->ne[1] == n_embd_gqa);
        layer.wqkv = ggml_new_tensor_2d(ctx_split, wq->type, n_embd, n_embd_head_k * (n_head + n_head_kv + n_head_kv));
        snprintf(layer.wqkv->name, GGML_MAX_NAME, "blk.%d.attn_qkv.weight", i);
        // This does not work. If we are doing this merge manually, it basically means that the arch does not have
        // an LLM_TENSOR_ATTN_QKV entry, so we will get __missing__ as the tensor name.
        //ggml_set_name(layer.wqkv, tn(LLM_TENSOR_ATTN_QKV, "weight", i).c_str());
        layer.wq = ml.create_tensor_as_view(ctx_split, layer.wqkv, wq_name.c_str(), { wq->ne[0], wq->ne[1] }, 0);
        layer.wk = ml.create_tensor_as_view(ctx_split, layer.wqkv, wk_name.c_str(), { wk->ne[0], wk->ne[1] }, wq->ne[1]*wq->nb[1]);
        layer.wv = ml.create_tensor_as_view(ctx_split, layer.wqkv, wv_name.c_str(), { wv->ne[0], wv->ne[1] }, wq->ne[1]*wq->nb[1] + wk->ne[1]*wk->nb[1] );
        fused_qkv = true;
        if (bias) {
            auto bq_name = tn(LLM_TENSOR_ATTN_Q, "bias", i);
            auto bk_name = tn(LLM_TENSOR_ATTN_K, "bias", i);
            auto bv_name = tn(LLM_TENSOR_ATTN_V, "bias", i);
            auto bq = ml.get_tensor_meta(bq_name.c_str());
            auto bk = ml.get_tensor_meta(bk_name.c_str());
            auto bv = ml.get_tensor_meta(bv_name.c_str());
            if (bias == 2) {
                GGML_ASSERT(bq && bk && bv);
            } else {
                GGML_ASSERT(!bq && !bk && !bv);
            }
            if (bq && bk && bv) {
                GGML_ASSERT(bq->type == GGML_TYPE_F32 && bk->type == GGML_TYPE_F32 && bv->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_nrows(bq) == 1 && bq->ne[0] == wq->ne[1]);
                GGML_ASSERT(ggml_nrows(bk) == 1 && bk->ne[0] == wk->ne[1]);
                GGML_ASSERT(ggml_nrows(bv) == 1 && bv->ne[0] == wv->ne[1]);
                layer.bqkv = ggml_new_tensor_1d(ctx_layer, bq->type, n_embd_head_k * (n_head + n_head_kv + n_head_kv));
                snprintf(layer.bqkv->name, GGML_MAX_NAME, "blk.%d.attn_qkv.bias", i);
                layer.bq = ml.create_tensor_as_view(ctx_layer, layer.bqkv, bq_name.c_str(), { bq->ne[0] }, 0);
                layer.bk = ml.create_tensor_as_view(ctx_layer, layer.bqkv, bk_name.c_str(), { bk->ne[0] }, bq->ne[0]*bq->nb[0]);
                layer.bv = ml.create_tensor_as_view(ctx_layer, layer.bqkv, bv_name.c_str(), { bv->ne[0] }, bq->ne[0]*bq->nb[0] + bk->ne[0]*bk->nb[0] );
            }
        }
    }
    if (!fused_qkv && ml.merge_qkv && wq->type == wk->type && (ignore_attn_scale || hparams.f_attention_scale == 0.0f)) {
        GGML_ASSERT(wq->ne[0] == n_embd && wq->ne[1] == n_head * n_embd_head_k);
        GGML_ASSERT(wk->ne[0] == n_embd && wk->ne[1] == n_embd_gqa);
        layer.wqk = ggml_new_tensor_2d(ctx_split, wq->type, n_embd, n_embd_head_k * (n_head + n_head_kv));
        snprintf(layer.wqk->name, GGML_MAX_NAME, "blk.%d.attn_qk.weight", i);
        layer.wq = ml.create_tensor_as_view(ctx_split, layer.wqk, wq_name.c_str(), { wq->ne[0], wq->ne[1] }, 0);
        layer.wk = ml.create_tensor_as_view(ctx_split, layer.wqk, wk_name.c_str(), { wk->ne[0], wk->ne[1] }, wq->ne[1]*wq->nb[1]);
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        printf("====================== Merged only Q and K in layer %d because V is of different type\n", i);
        fused_qkv = true;
        if (bias) {
            auto bq_name = tn(LLM_TENSOR_ATTN_Q, "bias", i);
            auto bk_name = tn(LLM_TENSOR_ATTN_K, "bias", i);
            auto bv_name = tn(LLM_TENSOR_ATTN_V, "bias", i);
            auto bq = ml.get_tensor_meta(bq_name.c_str());
            auto bk = ml.get_tensor_meta(bk_name.c_str());
            auto bv = ml.get_tensor_meta(bv_name.c_str());
            if (bias == 2) {
                GGML_ASSERT(bq && bk && bv);
            } else {
                GGML_ASSERT(!bq && !bk && !bv);
            }
            if (bq && bk && bv) {
                GGML_ASSERT(bq->type == GGML_TYPE_F32 && bk->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_nrows(bq) == 1 && bq->ne[0] == wq->ne[1]);
                GGML_ASSERT(ggml_nrows(bk) == 1 && bk->ne[0] == wk->ne[1]);
                layer.bqk = ggml_new_tensor_1d(ctx_layer, bq->type, n_embd_head_k * (n_head + n_head_kv));
                snprintf(layer.bqk->name, GGML_MAX_NAME, "blk.%d.attn_qk.bias", i);
                layer.bq = ml.create_tensor_as_view(ctx_layer, layer.bqk, bq_name.c_str(), { bq->ne[0] }, 0);
                layer.bk = ml.create_tensor_as_view(ctx_layer, layer.bqk, bk_name.c_str(), { bk->ne[0] }, bq->ne[0]*bq->nb[0]);
                layer.bv = create_tensor(ctx_layer, tn(LLM_TENSOR_ATTN_V,   "bias", i), {layer.wv->ne[1]});
            }
        }
    }

    if (!fused_qkv) {
        if (ml.merge_qkv) {
            printf("%s: did not merge Q, K, V in layer %d because %d, %d, %d\n", __func__, i,
                    wq->type == wk->type, wq->type == wv->type, (ignore_attn_scale || hparams.f_attention_scale == 0.0f));
        }
        layer.wq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head});
        layer.wk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa});
        layer.wv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa});
        if (bias) {
            auto flags = bias == 1 ? llama_model_loader::TENSOR_NOT_REQUIRED : 0;
            layer.bq = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_Q,   "bias", i), {layer.wq->ne[1]}, flags);
            layer.bk = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_K,   "bias", i), {layer.wk->ne[1]}, flags);
            layer.bv = create_tensor(ctx_split, tn(LLM_TENSOR_ATTN_V,   "bias", i), {layer.wv->ne[1]}, flags);
        }
    }

    return fused_qkv;
}

static void prepare_split_tensors(int split_dim, ggml_context * ctx, ggml_tensor * tensor, llama_split_tensor & split_tensor,
        const std::vector<int> & splits, std::vector<size_t> & mem_used) {
    GGML_ASSERT(split_dim <= 1);
    GGML_ASSERT(splits.size() > 1);
    std::string name{tensor->name};
    split_tensor.tensor_splits.resize(splits.size());
    if (split_dim < 0) {
        for (int i = 0; i < int(splits.size()); ++i) {
            if (splits[i] > 0) {
                split_tensor.tensor_splits[i] = ggml_new_tensor_3d(ctx, tensor->type, tensor->ne[0], tensor->ne[1], tensor->ne[2]);
                auto name_i = name + '.' + std::to_string(i);
                ggml_set_name(split_tensor.tensor_splits[i], name_i.c_str());
            } else {
                split_tensor.tensor_splits[i] = nullptr;
            }
        }
    }
    else if (split_dim == 1) {
        for (int i = 0; i < int(splits.size()); ++i) {
            if (splits[i] > 0) {
                split_tensor.tensor_splits[i] = ggml_new_tensor_3d(ctx, tensor->type, tensor->ne[0], splits[i], tensor->ne[2]);
                auto name_i = name + '.' + std::to_string(i);
                ggml_set_name(split_tensor.tensor_splits[i], name_i.c_str());
            } else {
                split_tensor.tensor_splits[i] = nullptr;
            }
        }
    } else {
        for (int i = 0; i < int(splits.size()); ++i) {
            if (splits[i] > 0) {
                split_tensor.tensor_splits[i] = ggml_new_tensor_3d(ctx, tensor->type, splits[i], tensor->ne[1], tensor->ne[2]);
                auto name_i = name + '.' + std::to_string(i);
                ggml_set_name(split_tensor.tensor_splits[i], name_i.c_str());
            } else {
                split_tensor.tensor_splits[i] = nullptr;
            }
        }
    }
    split_tensor.ggml.n_device  = splits.size();
    split_tensor.ggml.split_dim = split_dim;
    split_tensor.ggml.splits    = split_tensor.tensor_splits.data();
    tensor->extra = (void *)&split_tensor.ggml;
    GGML_ASSERT(mem_used.size() >= splits.size());
    for (int i = 0; i < split_tensor.ggml.n_device; ++i) {
        if (split_tensor.ggml.splits[i]) {
            //auto nbytes = ggml_nbytes(split_tensor.ggml.splits[i]);
            //printf("mem_used(%s): %8.2f, total: %8.2f\n", split_tensor.ggml.splits[i]->name, nbytes/1024./1024., (mem_used[i] + nbytes)/1024./1024.);
            mem_used[i] += ggml_nbytes(split_tensor.ggml.splits[i]);
        }
    }
}

static void adjust_split(std::vector<float> & split, const std::vector<size_t> & mem_used, int max_gpu) {
    if (max_gpu < 1 || max_gpu >= int(split.size()) || split.size() != mem_used.size()) {
        return;
    }
    size_t tot_mem_used = 1;
    for (auto & mem : mem_used) tot_mem_used += mem;
    for (int i = split.size() - 1; i > 0; --i) split[i] -= split[i-1];
    std::vector<std::pair<float, int>> sorted(split.size());
    for (int i = 0; i < int(split.size()); ++i) {
        float mem_ideal = split[i]*tot_mem_used;
        float err = mem_ideal - mem_used[i];
        sorted[i] = {err, i};
    }
    std::partial_sort(sorted.begin(), sorted.begin() + max_gpu, sorted.end(), std::greater<std::pair<float,int>>{});
    for (auto & p : split) p = 0;
    for (int j = 0; j < max_gpu; ++j) split[sorted[j].second] = 1;
    float sum = 0;
    for (auto & p : split) {
        sum += p/max_gpu;
        p = sum;
    }
}

bool create_tensors_helper::create_tensors() {
    const auto tn = LLM_TN(model.arch);
    bool use_mmap_buffer = true;
    if (ml.merge_qkv && (model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN)) {
        LLAMA_LOG_WARN("\n========================================================\n");
        LLAMA_LOG_WARN("merge_qkv is not compatible with split mode 'graph'\n");
        LLAMA_LOG_WARN("  => turning off merge_qkv\n");
        LLAMA_LOG_WARN("========================================================\n\n");
        ml.merge_qkv = false;
    }
    if (ml.merge_up_gate_exps && (model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN)) {
        LLAMA_LOG_WARN("\n========================================================\n");
        LLAMA_LOG_WARN("merge_up_gate_exps is not compatible with split mode 'graph'\n");
        LLAMA_LOG_WARN("  => turning off merge_up_gate_exps\n");
        LLAMA_LOG_WARN("========================================================\n\n");
        ml.merge_up_gate_exps = false;
    }
    switch (model.arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
            use_mmap_buffer = create_llama_tensors(tn); break;
        case LLM_ARCH_DECI:
            use_mmap_buffer = create_deci_tensors(tn); break;
        case LLM_ARCH_LLAMA4:
            use_mmap_buffer = create_llama4_tensors(tn); break;
        case LLM_ARCH_GROK:
            use_mmap_buffer = create_grok_tensors(tn); break;
        case LLM_ARCH_DBRX:
            use_mmap_buffer = create_dbrx_tensors(tn); break;
        case LLM_ARCH_BAICHUAN:
            use_mmap_buffer = create_baichuan_tensors(tn); break;
        case LLM_ARCH_FALCON:
            use_mmap_buffer = create_falcon_tensors(tn); break;
        case LLM_ARCH_STARCODER:
            use_mmap_buffer = create_starcoder_tensors(tn); break;
        case LLM_ARCH_BERT:
        case LLM_ARCH_NOMIC_BERT:
            use_mmap_buffer = create_bert_tensors(tn); break;
        case LLM_ARCH_JINA_BERT_V2:
            use_mmap_buffer = create_jina_bert2_tensors(tn); break;
        case LLM_ARCH_BLOOM:
            use_mmap_buffer = create_bloom_tensors(tn); break;
        case LLM_ARCH_MPT:
            use_mmap_buffer = create_mpt_tensors(tn); break;
        case LLM_ARCH_STABLELM:
            use_mmap_buffer = create_stablelm_tensors(tn); break;
        case LLM_ARCH_QWEN:
            use_mmap_buffer = create_qwen_tensors(tn); break;
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_QWEN2VL:
            use_mmap_buffer = create_qwen2_tensors(tn); break;
        case LLM_ARCH_QWEN2MOE:
            use_mmap_buffer = create_qwen2_moe_tensors(tn); break;
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3VL:
            use_mmap_buffer = create_qwen3_tensors(tn); break;
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3VLMOE:
            use_mmap_buffer = create_qwen3_moe_tensors(tn); break;
        case LLM_ARCH_PHI2:
            use_mmap_buffer = create_phi2_tensors(tn); break;
        case LLM_ARCH_PHI3:
            use_mmap_buffer = create_phi3_tensors(tn); break;
        case LLM_ARCH_PLAMO:
            use_mmap_buffer = create_baichuan_tensors(tn, false); break;
        case LLM_ARCH_GPT2:
            use_mmap_buffer = create_gpt2_tensors(tn); break;
        case LLM_ARCH_CODESHELL:
            use_mmap_buffer = create_codeshell_tensors(tn); break;
        case LLM_ARCH_ORION:
            use_mmap_buffer = create_orion_tensors(tn); break;
        case LLM_ARCH_INTERNLM2:
            use_mmap_buffer = create_internlm_tensors(tn); break;
        case LLM_ARCH_GEMMA:
            use_mmap_buffer = create_gemma_tensors(tn, 1); break;
        case LLM_ARCH_GEMMA2:
            use_mmap_buffer = create_gemma_tensors(tn, 2); break;
        case LLM_ARCH_GEMMA3:
            use_mmap_buffer = create_gemma_tensors(tn, 3); break;
        case LLM_ARCH_STARCODER2:
            use_mmap_buffer = create_starcoder2_tensors(tn); break;
        case LLM_ARCH_MAMBA:
            use_mmap_buffer = create_mamba_tensors(tn); break;
        case LLM_ARCH_XVERSE:
            use_mmap_buffer = create_xverse_tensors(tn); break;
        case LLM_ARCH_COMMAND_R:
            use_mmap_buffer = create_command_r_tensors(tn); break;
        case LLM_ARCH_OLMO:  // adapted from LLM_ARCH_LLAMA with norm params removed
            use_mmap_buffer = create_olmo_tensors(tn); break;
        case LLM_ARCH_OPENELM:
            use_mmap_buffer = create_openelm_tensors(tn); break;
        case LLM_ARCH_GPTNEOX:
            use_mmap_buffer = create_gptneox_tensors(tn); break;
        case LLM_ARCH_ARCTIC:
            use_mmap_buffer = create_arctix_tensors(tn); break;
        case LLM_ARCH_DEEPSEEK2:
            use_mmap_buffer = create_deepseek2_tensors(tn); break;
        case LLM_ARCH_GLM4_MOE:
            use_mmap_buffer = create_glm4_moe_tensors(tn); break;
        case LLM_ARCH_BITNET:
            use_mmap_buffer = create_bitnet_tensors(tn); break;
        case LLM_ARCH_BITNET_B158:
        case LLM_ARCH_BITNET_25:
            use_mmap_buffer = create_bitnet2_tensors(tn); break;
        case LLM_ARCH_T5:
            use_mmap_buffer = create_t5_tensors(tn); break;
        case LLM_ARCH_T5ENCODER:
            use_mmap_buffer = create_tsencoder_tensors(tn); break;
        case LLM_ARCH_JAIS:
            use_mmap_buffer = create_jais_tensors(tn); break;
        case LLM_ARCH_CHATGLM:
            use_mmap_buffer = create_chatglm_tensors(tn); break;
        case LLM_ARCH_COHERE2:
            use_mmap_buffer = create_cohere2_tensors(tn); break;
        case LLM_ARCH_GLM4:
            use_mmap_buffer = create_glm4_tensors(tn); break;
        case LLM_ARCH_DOTS1:
            use_mmap_buffer = create_dots1_tensors(tn); break;
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
            use_mmap_buffer = create_ernie45_tensors(tn); break;
        case LLM_ARCH_HUNYUAN_MOE:
            use_mmap_buffer = create_hunyuan_tensors(tn); break;
        case LLM_ARCH_OPENAI_MOE:
            use_mmap_buffer = create_openai_moe_tensors(tn); break;
        case LLM_ARCH_BAILINGMOE2:
            use_mmap_buffer = create_bailingmoe2_tensors(tn); break;
        case LLM_ARCH_MINIMAX_M2:
            use_mmap_buffer = create_minimaxm2_tensors(tn); break;
        case LLM_ARCH_SMOLLM3:
            use_mmap_buffer = create_smollm3_tensors(tn); break;
        case LLM_ARCH_MIMO2:
            use_mmap_buffer = create_mimo2_tensors(tn); break;
        default:
            throw std::runtime_error("unknown architecture");
    }
    if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN) {
        const int n_layer = model.layers.size() - model.hparams.nextn_predict_layers;
        printf("================================ max_gpu = %d\n", model.max_gpu);
        std::vector<size_t> mem_used(model.splits.size(), 0);
        const auto & hparams = model.hparams;
        auto cur_splits = model.splits;
        int adjust_step = std::max(1, int(n_layer / (2*model.splits.size())));
        if (model.max_gpu > 1 && model.max_gpu < int(cur_splits.size())) {
            bool equal_split = true;
            for (int i = 0; i < int(cur_splits.size()); ++i) {
                float p = i > 0 ? cur_splits[i] - cur_splits[i-1] : cur_splits[i];
                if (std::abs(p*cur_splits.size() - 1.f) > 0.02f) {
                    equal_split = false; break;
                }
            }
            if (equal_split) {
                if (cur_splits.size() % model.max_gpu == 0) {
                    int nadj = cur_splits.size()/model.max_gpu;
                    adjust_step = (n_layer + nadj - 1) / nadj;
                } else {
                    adjust_step = (n_layer + cur_splits.size() - 1)/cur_splits.size();
                }
            }
        }
        for (int il = 0; il < n_layer; ++il) {
            int gqa_ratio = hparams.n_head(il) / hparams.n_head_kv(il);
            if (ggml_backend_buft_is_host(model.buft_layer[il].buft_matrix)) {
                LLAMA_LOG_INFO("%s: not splitting layer %d because buffer type is host\n", __func__, il);
                continue;
            }
            if (model.max_gpu > 0 && model.max_gpu < int(model.splits.size()) && il % adjust_step == 0) {
                cur_splits = model.splits;
                adjust_split(cur_splits, mem_used, model.max_gpu);
                printf("Adjusted split at layer %2d:", il);
                float last_split = 0;
                for (auto & p : cur_splits) {
                    printf(" %g", p - last_split);
                    last_split = p;
                }
                printf("\n");
            }
            //printf("=== Layer %2d. Mem used so far:", il);
            //for (auto mem : mem_used) printf(" %g", mem/1024./1024.);
            //printf("\n");
            auto & layer = model.layers[il];
            auto ctx_split = ctx_for_layer_split(il);
            if (layer.attn_norm) {
                auto split = create_split(ggml_nrows(layer.attn_norm), -1, cur_splits, mem_used);
                prepare_split_tensors(-1, ctx_split, layer.attn_norm, layer.split_attn_norm, split, mem_used);
            }
            if (layer.rope_freqs) {
                auto split = create_split(ggml_nrows(layer.rope_freqs), -1, cur_splits, mem_used);
                prepare_split_tensors(-1, ctx_split, layer.rope_freqs, layer.split_rope_freqs, split, mem_used);
            }
            if (layer.wo && layer.wq && layer.wk && layer.wv) {
                auto granularity_kq = hparams.n_embd_head_k * gqa_ratio;
                auto granularity_vo = hparams.n_embd_head_v * gqa_ratio;
                if (ggml_is_quantized(layer.wo->type)) {
                    auto tt = ggml_internal_get_type_traits(layer.wo->type);
                    if (tt.blck_size > granularity_vo) granularity_vo = tt.blck_size;
                    GGML_ASSERT(granularity_vo % hparams.n_embd_head_v == 0);
                }
                auto split_vo = create_split(layer.wo->ne[0], granularity_vo, cur_splits, mem_used); //, true);
                auto split_kq = create_split(layer.wq->ne[1], granularity_kq, cur_splits, mem_used); //, true);
                //printf("  split_vo:"); for (auto s : split_vo) printf(" %d", s);
                //printf("\n");
                //printf("  split_kq:"); for (auto s : split_kq) printf(" %d", s);
                //printf("\n");
                prepare_split_tensors(0, ctx_split, layer.wo, layer.split_wo, split_vo, mem_used);
                prepare_split_tensors(1, ctx_split, layer.wq, layer.split_wq, split_kq, mem_used);
                if (layer.bo) {
                    prepare_split_tensors(-1, ctx_split, layer.bo, layer.split_bo, split_vo, mem_used);
                }
                if (layer.bq) {
                    prepare_split_tensors(0, ctx_split, layer.bq, layer.split_bq, split_kq, mem_used);
                }
                if (layer.attn_q_norm) {
                    prepare_split_tensors(-1, ctx_split, layer.attn_q_norm, layer.split_q_norm, split_kq, mem_used);
                }
                if (layer.attn_sinks) {
                    auto split_sinks = split_kq;
                    for (auto & s : split_sinks) {
                        s /= hparams.n_embd_head_k;
                    }
                    prepare_split_tensors(0, ctx_split, layer.attn_sinks, layer.split_sinks, split_sinks, mem_used);
                }
                for (auto & s : split_kq) s /= gqa_ratio;
                for (auto & s : split_vo) s /= gqa_ratio;
                prepare_split_tensors(1, ctx_split, layer.wk, layer.split_wk, split_kq, mem_used);
                prepare_split_tensors(1, ctx_split, layer.wv, layer.split_wv, split_vo, mem_used);
                if (layer.bk) {
                    prepare_split_tensors(0, ctx_split, layer.bk, layer.split_bk, split_kq, mem_used);
                }
                if (layer.bv) {
                    prepare_split_tensors(0, ctx_split, layer.bv, layer.split_bv, split_vo, mem_used);
                }
                if (layer.attn_k_norm) {
                    prepare_split_tensors(-1, ctx_split, layer.attn_k_norm, layer.split_k_norm, split_kq, mem_used);
                }
            }

            if (layer.ffn_norm) {
                if (auto it = split_tensors.find(layer.ffn_norm); it != split_tensors.end()) {
                    auto split = create_split(ggml_nrows(layer.ffn_norm), -1, cur_splits, mem_used);
                    prepare_split_tensors(-1, ctx_split, layer.ffn_norm, layer.split_ffn_norm, split, mem_used);
                }
            }

            if (layer.ffn_down && layer.ffn_up && layer.ffn_gate) {
                bool use_split = split_tensors.find(layer.ffn_down) != split_tensors.end() &&
                                 split_tensors.find(layer.ffn_gate) != split_tensors.end() &&
                                 split_tensors.find(layer.ffn_up)   != split_tensors.end();
                if (use_split) {
                    int ffn_granularity = 16;
                    if (ggml_is_quantized(layer.ffn_down->type)) {
                        auto tt = ggml_internal_get_type_traits(layer.ffn_down->type);
                        if (tt.blck_size > ffn_granularity) ffn_granularity = tt.blck_size;
                    }
                    auto split = create_split(layer.ffn_down->ne[0], ffn_granularity, cur_splits, mem_used);
                    prepare_split_tensors(0, ctx_split, layer.ffn_down, layer.split_ffn_down, split, mem_used);
                    prepare_split_tensors(1, ctx_split, layer.ffn_up,   layer.split_ffn_up,   split, mem_used);
                    prepare_split_tensors(1, ctx_split, layer.ffn_gate, layer.split_ffn_gate, split, mem_used);
                }
            }

            //bool any_ffn_split = false;
            if (layer.ffn_down_shexp && layer.ffn_up_shexp && layer.ffn_gate_shexp) {
                bool use_split = split_tensors.find(layer.ffn_down_shexp) != split_tensors.end() &&
                                 split_tensors.find(layer.ffn_gate_shexp) != split_tensors.end() &&
                                 split_tensors.find(layer.ffn_up_shexp)   != split_tensors.end();
                if (use_split) {
                    //any_ffn_split = true;
                    int ffn_granularity = 16;
                    if (ggml_is_quantized(layer.ffn_down_shexp->type)) {
                        auto tt = ggml_internal_get_type_traits(layer.ffn_down_shexp->type);
                        if (tt.blck_size > ffn_granularity) ffn_granularity = tt.blck_size;
                    }
                    auto split = create_split(layer.ffn_down_shexp->ne[0], ffn_granularity, cur_splits, mem_used);
                    prepare_split_tensors(0, ctx_split, layer.ffn_down_shexp, layer.split_ffn_down_shexp, split, mem_used);
                    prepare_split_tensors(1, ctx_split, layer.ffn_up_shexp,   layer.split_ffn_up_shexp,   split, mem_used);
                    prepare_split_tensors(1, ctx_split, layer.ffn_gate_shexp, layer.split_ffn_gate_shexp, split, mem_used);
                }
            }

            if (layer.ffn_down_exps && layer.ffn_up_exps && layer.ffn_gate_exps) {
                bool use_split = split_tensors.find(layer.ffn_down_exps) != split_tensors.end() &&
                                 split_tensors.find(layer.ffn_gate_exps) != split_tensors.end() &&
                                 split_tensors.find(layer.ffn_up_exps)   != split_tensors.end();

                if (use_split) {
                    //any_ffn_split = true;
                    int ffn_granularity = 16;
                    if (ggml_is_quantized(layer.ffn_down_exps->type)) {
                        auto tt = ggml_internal_get_type_traits(layer.ffn_down_exps->type);
                        if (tt.blck_size > ffn_granularity) ffn_granularity = tt.blck_size;
                    }
                    auto split = create_split(layer.ffn_down_exps->ne[0], ffn_granularity, cur_splits, mem_used);
                    //printf("split(%2d):", il); for (auto & s : split) printf(" %d", s); printf("\n");
                    prepare_split_tensors(0, ctx_split, layer.ffn_down_exps, layer.split_ffn_down_exps, split, mem_used);
                    prepare_split_tensors(1, ctx_split, layer.ffn_up_exps,   layer.split_ffn_up_exps,   split, mem_used);
                    prepare_split_tensors(1, ctx_split, layer.ffn_gate_exps, layer.split_ffn_gate_exps, split, mem_used);
                    if (layer.ffn_down_exps_b) {
                        prepare_split_tensors(-1, ctx_split, layer.ffn_down_exps_b, layer.split_ffn_down_exps_b, split, mem_used);
                    }
                    if (layer.ffn_up_exps_b) {
                        prepare_split_tensors( 0, ctx_split, layer.ffn_up_exps_b, layer.split_ffn_up_exps_b, split, mem_used);
                    }
                    if (layer.ffn_gate_exps_b) {
                        prepare_split_tensors( 0, ctx_split, layer.ffn_gate_exps_b, layer.split_ffn_gate_exps_b, split, mem_used);
                    }
                }
            }

            if (layer.ffn_gate_inp) {
                if (auto it = split_tensors.find(layer.ffn_gate_inp); it != split_tensors.end()) {
                    auto shared_split = create_split(ggml_nrows(layer.ffn_gate_inp), -1, cur_splits, mem_used);
                    prepare_split_tensors(-1, ctx_split, layer.ffn_gate_inp, layer.split_ffn_gate_inp, shared_split, mem_used);
                }
            }
            if (layer.ffn_gate_inp_b) {
                if (auto it = split_tensors.find(layer.ffn_gate_inp_b); it != split_tensors.end()) {
                    auto shared_split = create_split(ggml_nrows(layer.ffn_gate_inp_b), -1, cur_splits, mem_used);
                    prepare_split_tensors(-1, ctx_split, layer.ffn_gate_inp_b, layer.split_ffn_gate_inp_b, shared_split, mem_used);
                }
            }
            if (layer.ffn_exp_probs_b) {
                if (auto it = split_tensors.find(layer.ffn_exp_probs_b); it != split_tensors.end()) {
                    auto shared_split = create_split(ggml_nrows(layer.ffn_exp_probs_b), -1, cur_splits, mem_used);
                    prepare_split_tensors(-1, ctx_split, layer.ffn_exp_probs_b, layer.split_ffn_exp_probs_b, shared_split, mem_used);
                }
            }
        }

        if (model.output) {
            if (auto it = split_tensors.find(model.output); it != split_tensors.end()) {
                if (ggml_backend_buft_is_host(model.buft_output.buft_matrix)) {
                    LLAMA_LOG_INFO("%s: not splitting output tensor becausee buffer is host\n", __func__);
                } else {
                    auto ctx_split = ctx_map[model.buft_output.buft_matrix];
                    auto split = create_split(model.output->ne[1], 16, model.splits, mem_used);
                    prepare_split_tensors(1, ctx_split, model.output, model.split_output, split, mem_used);
                    if (auto it = split_tensors.find(model.output_norm); it != split_tensors.end() && !ggml_backend_buft_is_host(model.buft_output.buft_matrix)) {
                        auto ctx_split = ctx_map[model.buft_output.buft_matrix];
                        prepare_split_tensors(-1, ctx_split, model.output_norm, model.split_output_norm, split, mem_used);
                    }
                }
            }
        }

        LLAMA_LOG_INFO("Estimated model buffer size per device:\n");
        for (int i = 0; i < int(mem_used.size()); ++i) {
            LLAMA_LOG_INFO("    Device %d:  %8.2f MiB\n", i, mem_used[i]/1024./1024.);
        }
    }
    return use_mmap_buffer;
}

std::unique_ptr<create_tensors_helper_interface> create_tensors_helper_interface::instance(llama_model_loader & ml, llama_model & model) {
    return std::make_unique<create_tensors_helper>(ml, model);
}
