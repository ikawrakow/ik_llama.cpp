// quillama.cpp: the fast grammar mask (cached decoding + first code point filter) must reject exactly
// the same tokens as the reference algorithm, over the whole vocabulary, along random valid paths.
#include "llama.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "../src/llama-grammar.h"
#include "../src/llama-vocab.h"

#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using json = nlohmann::ordered_json;

static std::vector<float> mask(const llama_grammar * g, const llama_vocab * vocab, bool reference) {
    if (reference) setenv("QUILL_GRAMMAR_REFERENCE", "1", 1); else unsetenv("QUILL_GRAMMAR_REFERENCE");
    const int n = vocab->n_tokens();
    std::vector<llama_token_data> cand(n);
    for (int i = 0; i < n; ++i) cand[i] = { i, 0.0f, 0.0f };
    llama_token_data_array arr = { cand.data(), (size_t) n, false };
    llama_grammar_sample_impl(g, vocab, nullptr, &arr);
    std::vector<float> m(n);
    for (int i = 0; i < n; ++i) m[i] = cand[i].logit;
    return m;
}

int main(int argc, char ** argv) {
    const char * vocab_path = argc > 1 ? argv[1] : "models/ggml-vocab-qwen2.gguf";
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(vocab_path, mparams);
    if (!model) { fprintf(stderr, "failed to load %s\n", vocab_path); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * schemas[] = {
        R"({"type":"object","properties":{"name":{"type":"string","maxLength":12},"age":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}}},"required":["name","age","tags"],"additionalProperties":false})",
        R"({"type":"object","properties":{"kind":{"enum":["alpha","beta","gamma"]},"x":{"type":"number"},"ok":{"type":"boolean"},"note":{"type":["string","null"]}},"required":["kind","x","ok","note"]})",
        R"({"type":"object","properties":{"code":{"type":"string","pattern":"^[A-Z]{3}-[0-9]{2}$"},"uni":{"type":"string"}},"required":["code","uni"]})",
    };
    std::mt19937 rng(7);
    bool ok = true;
    int n_steps_total = 0;
    for (const char * sch : schemas) {
        const std::string gbnf = json_schema_to_grammar(json::parse(sch));
        for (int walk = 0; walk < 3; ++walk) {
            llama_grammar * g = llama_grammar_init_impl(vocab, gbnf.c_str(), "root", false, nullptr, 0, nullptr, 0);
            if (!g) { fprintf(stderr, "grammar init failed\n"); return 1; }
            for (int step = 0; step < 120; ++step) {
                const auto a = mask(g, vocab, true);
                const auto b = mask(g, vocab, false);
                std::vector<llama_token> valid;
                for (size_t i = 0; i < a.size(); ++i) {
                    if ((a[i] == -INFINITY) != (b[i] == -INFINITY)) {
                        fprintf(stderr, "MISMATCH at step %d token %zu '%s': reference %s, fast %s\n", step, i,
                                vocab->token_to_piece((llama_token) i).c_str(), a[i] == -INFINITY ? "rejects" : "accepts",
                                b[i] == -INFINITY ? "rejects" : "accepts");
                        ok = false;
                    }
                    if (a[i] != -INFINITY && !vocab->is_eog((llama_token) i)) valid.push_back((llama_token) i);
                }
                ++n_steps_total;
                if (valid.empty() || !ok) break;
                const llama_token t = valid[rng() % valid.size()];
                llama_grammar_accept_impl(*g, vocab, nullptr, t);
            }
            llama_grammar_free_impl(g);
        }
    }
    printf("%d grammar states compared over %u tokens: %s\n", n_steps_total, vocab->n_tokens(), ok ? "identical masks" : "MISMATCH");
    llama_free_model(model);
    return ok ? 0 : 1;
}
