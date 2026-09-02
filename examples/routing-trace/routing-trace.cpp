// routing-trace: dump MoE routing (ffn_moe_topk expert IDs) per token per layer to JSONL.
// Optional TRACE_ACT=1: also dump downsampled ffn_moe_gate_par activations.
// Usage: llama-routing-trace -m model.gguf -f corpus.txt   (env: TRACE_OUT, N_GEN, TRACE_ACT)
// Corpus format: prompts separated by lines starting with ###PROMPT
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

struct trace_data {
    FILE * out = nullptr;
    std::vector<uint8_t> buf;
    int prompt_idx = -1;
    int call_idx = 0;
    bool trace_act = false;
};

static int routing_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * td = (trace_data *) user_data;
    bool is_topk = strncmp(t->name, "ffn_moe_topk", 12) == 0;
    bool is_act = td->trace_act && strncmp(t->name, "ffn_moe_gate_par", 16) == 0;
    bool is_pregate = strncmp(t->name, "pregate_topk", 12) == 0;
    if (!is_topk && !is_act && !is_pregate) {
        return ask ? 0 : 1;
    }
    if (ask) return 1;

    const char * dash = strrchr(t->name, '-');
    int layer = dash ? atoi(dash + 1) : -1;
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (!is_host) {
        td->buf.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, td->buf.data(), 0, ggml_nbytes(t));
    }
    const uint8_t * data = is_host ? (const uint8_t *) t->data : td->buf.data();

    if (is_pregate) {
        // [n_expert_used, n_tokens] i32 — prediction FOR layer+1
        int64_t n_used = t->ne[0], n_tok = t->ne[1];
        fprintf(td->out, "{\"prompt\":%d,\"call\":%d,\"layer\":%d,\"pregate\":[",
                td->prompt_idx, td->call_idx, layer);
        for (int64_t j = 0; j < n_tok; ++j) {
            fprintf(td->out, "[");
            for (int64_t i = 0; i < n_used; ++i) {
                size_t idx = j * t->nb[1] + i * t->nb[0];
                int32_t e = *(const int32_t *)(data + idx);
                fprintf(td->out, "%s%d", i ? "," : "", e);
            }
            fprintf(td->out, "]%s", j + 1 < n_tok ? "," : "");
        }
        fprintf(td->out, "]}\n");
        fflush(td->out);
        return 1;
    }

    if (is_act) {
        // [n_embd, n_used, n_tokens] — dump downsampled magnitudes per (token, expert-slot)
        int64_t n0 = t->ne[0], n1 = t->ne[1], n2 = t->ne[2];
        int64_t stride = std::max<int64_t>(1, n0 / 128);
        fprintf(td->out, "{\"prompt\":%d,\"call\":%d,\"layer\":%d,\"act\":[",
                td->prompt_idx, td->call_idx, layer);
        bool first_tok = true;
        for (int64_t k = 0; k < n2; ++k) {
            for (int64_t j = 0; j < n1; ++j) {
                fprintf(td->out, "%s[", first_tok ? "" : ",");
                first_tok = false;
                bool first = true;
                for (int64_t i = 0; i < n0; i += stride) {
                    size_t idx = k * t->nb[2] + j * t->nb[1] + i * t->nb[0];
                    float v = t->type == GGML_TYPE_F16
                        ? ggml_fp16_to_fp32(*(const ggml_fp16_t *)(data + idx))
                        : *(const float *)(data + idx);
                    fprintf(td->out, "%s%.6g", first ? "" : ",", v);
                    first = false;
                }
                fprintf(td->out, "]");
            }
        }
        fprintf(td->out, "]}\n");
        fflush(td->out);
        return 1;
    }

    // topk: [n_expert_used, n_tokens] i32
    int64_t n_used = t->ne[0], n_tok = t->ne[1];
    fprintf(td->out, "{\"prompt\":%d,\"call\":%d,\"layer\":%d,\"experts\":[",
            td->prompt_idx, td->call_idx, layer);
    for (int64_t j = 0; j < n_tok; ++j) {
        fprintf(td->out, "[");
        for (int64_t i = 0; i < n_used; ++i) {
            size_t idx = j * t->nb[1] + i * t->nb[0];
            int32_t e = *(const int32_t *)(data + idx);
            fprintf(td->out, "%s%d", i ? "," : "", e);
        }
        fprintf(td->out, "]%s", j + 1 < n_tok ? "," : "");
    }
    fprintf(td->out, "]}\n");
    fflush(td->out);
    return 1;
}

int main(int argc, char ** argv) {
    trace_data td;
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }
    const char * trace_out = getenv("TRACE_OUT");
    if (!trace_out) trace_out = "trace.jsonl";
    td.out = fopen(trace_out, "w");
    if (!td.out) { fprintf(stderr, "cannot open %s\n", trace_out); return 1; }
    td.trace_act = getenv("TRACE_ACT") && strcmp(getenv("TRACE_ACT"), "1") == 0;
    const int n_gen = getenv("N_GEN") ? atoi(getenv("N_GEN")) : 32;

    std::mt19937 rng(params.seed);
    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = routing_cb;
    params.cb_eval_user_data = &td;
    params.warmup = false;

    llama_init_result llama_init = llama_init_from_gpt_params(params);
    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;
    if (!model || !ctx) { fprintf(stderr, "init failed\n"); return 1; }
    common_sampler * smpl = common_sampler_init(model, params.sparams);
    if (!smpl) { fprintf(stderr, "sampler init failed\n"); return 1; }

    std::vector<std::string> prompts;
    {
        const std::string & path = params.prompt_file.empty() ? params.prompt : params.prompt_file;
        std::ifstream in(path);
        if (!in) { fprintf(stderr, "cannot open corpus %s\n", path.c_str()); return 1; }
        std::string line, cur;
        while (std::getline(in, line)) {
            if (line.rfind("###PROMPT", 0) == 0) {
                if (!cur.empty()) { prompts.push_back(cur); cur.clear(); }
            } else { cur += line + "\n"; }
        }
        if (!cur.empty()) prompts.push_back(cur);
    }
    fprintf(stderr, "corpus: %zu prompts, n_gen=%d, act=%d\n",
            prompts.size(), n_gen, (int) td.trace_act);

    const bool add_bos = llama_should_add_bos_token(model);
    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        td.prompt_idx = (int) pi;
        td.call_idx = 0;
        llama_kv_cache_clear(ctx);
        common_sampler_reset(smpl);
        std::vector<llama_token> tokens = ::common_tokenize(ctx, prompts[pi], add_bos);
        if (tokens.empty()) continue;
        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), 0, 0))) {
            fprintf(stderr, "prefill failed prompt %zu\n", pi); continue;
        }
        td.call_idx++;
        llama_token tok = common_sampler_sample(smpl, ctx, -1);
        for (int g = 0; g < n_gen && !llama_vocab_is_eog(llama_model_get_vocab(model), tok); ++g) {
            if (llama_decode(ctx, llama_batch_get_one(&tok, 1, tokens.size() + g, 0))) break;
            td.call_idx++;
            tok = common_sampler_sample(smpl, ctx, -1);
        }
        fprintf(stderr, "prompt %zu done (%d calls)\n", pi, td.call_idx);
    }
    fclose(td.out);
    common_sampler_free(smpl);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
