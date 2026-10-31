// Verifies that a batch which interleaves several sequences gets regrouped before
// the ubatch loop, instead of degrading to one token per graph.
//
// Hybrid architectures (delta-net based) need each sequence's tokens in one
// contiguous run per ubatch. Callers interleave sequences token by token, which
// used to make the decoder fall back to single-token graphs: same output, but
// prefill collapses by roughly 40x.
//
// The decoder counts those fallbacks in the internal llama_context field
// n_single_token_fallbacks, read directly here so the public API stays untouched.
// A 128-token batch over two sequences must need zero of them, whether the tokens
// arrive grouped or interleaved. Before the fix the interleaved case needed 128.
//
// Run with: ./tests/test-hybrid-batch-order <model.gguf>
// or set LLAMACPP_TEST_MODELFILE. Skips when no model is available.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "llama.h"
#include "get-model.h"

// internal header, for the fallback counter only; no public accessor exists
#include "../src/llama-context.h"

static const int32_t N_TOK     = 64;
static const int32_t N_SEQ     = 2;
static const int32_t N_UBATCH  = 32;
static const int32_t N_TOKENS  = N_TOK * N_SEQ;
static const int32_t N_CTX     = 256;
static const float   LOGIT_TOL = 1e-3f;

struct run_result {
    uint64_t fallbacks = 0;
    std::vector<float> logits[N_SEQ];
};

// Adds one token to the batch at the next free slot.
static void add_token(llama_batch & batch, int32_t i, llama_token tok, llama_pos pos, llama_seq_id seq, bool want_logits) {
    batch.token    [i] = tok;
    batch.pos      [i] = pos;
    batch.n_seq_id [i] = 1;
    batch.seq_id[i][0] = seq;
    batch.logits   [i] = want_logits ? 1 : 0;
}

// Fills the batch grouped (seq0 tokens, then seq1 tokens) or interleaved (token-major).
static void fill_batch(llama_batch & batch, const std::vector<llama_token> & toks, bool interleaved) {
    int32_t i = 0;
    if (!interleaved) {
        for (llama_seq_id s = 0; s < N_SEQ; ++s) {
            for (int32_t t = 0; t < N_TOK; ++t) {
                add_token(batch, i++, toks[t], t, s, t == N_TOK - 1);
            }
        }
    } else {
        for (int32_t t = 0; t < N_TOK; ++t) {
            for (llama_seq_id s = 0; s < N_SEQ; ++s) {
                add_token(batch, i++, toks[t], t, s, t == N_TOK - 1);
            }
        }
    }
    batch.n_tokens = i;
}

// Decodes one batch on a clean cache and records fallback count plus last-token logits per sequence.
static bool run_batch(llama_context * ctx, int32_t n_vocab, const std::vector<llama_token> & toks, bool interleaved, run_result & out) {
    llama_batch batch = llama_batch_init(N_TOKENS, 0, N_SEQ);
    fill_batch(batch, toks, interleaved);

    llama_kv_cache_clear(ctx);

    const uint64_t before = ctx->n_single_token_fallbacks;
    const bool     ok     = llama_decode(ctx, batch) == 0;
    out.fallbacks = ctx->n_single_token_fallbacks - before;

    bool have_logits = false;
    if (ok) {
        const float * l = llama_get_logits(ctx);
        if (l) {
            for (int32_t s = 0; s < N_SEQ; ++s) {
                out.logits[s].assign(l + (size_t) s * n_vocab, l + (size_t) (s + 1) * n_vocab);
            }
            have_logits = true;
        }
    }

    llama_batch_free(batch);
    return ok && have_logits;
}

int main(int argc, char ** argv) {
    const char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path);
        return 1;
    }

    const int32_t n_vocab = llama_n_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = N_CTX;
    cparams.n_batch         = N_TOKENS;
    cparams.n_ubatch        = N_UBATCH;
    cparams.n_seq_max       = N_SEQ;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_free_model(model);
        return 1;
    }

    // Build a prompt long enough for N_TOK tokens.
    std::string text;
    while (text.size() < 4096) {
        text += "The quick brown fox jumps over the lazy dog. ";
    }

    std::vector<llama_token> tokens(4096);
    int32_t n = llama_tokenize(model, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), true, false);
    if (n < N_TOK) {
        fprintf(stderr, "error: tokenizer produced %d tokens, need %d\n", n, N_TOK);
        llama_free(ctx);
        llama_free_model(model);
        return 1;
    }
    tokens.resize(N_TOK);

    run_result grouped;
    run_result interleaved;

    if (!run_batch(ctx, n_vocab, tokens, false, grouped) ||
        !run_batch(ctx, n_vocab, tokens, true,  interleaved)) {
        fprintf(stderr, "error: llama_decode failed\n");
        llama_free(ctx);
        llama_free_model(model);
        return 1;
    }

    bool ok = true;

    printf("grouped:     %llu single-token fallbacks (expected 0)\n", (unsigned long long) grouped.fallbacks);
    printf("interleaved: %llu single-token fallbacks (expected 0)\n", (unsigned long long) interleaved.fallbacks);

    if (grouped.fallbacks != 0) {
        fprintf(stderr, "FAIL: grouped batch needed %llu fallbacks\n", (unsigned long long) grouped.fallbacks);
        ok = false;
    }
    if (interleaved.fallbacks != 0) {
        fprintf(stderr, "FAIL: interleaved batch needed %llu fallbacks, expected 0\n", (unsigned long long) interleaved.fallbacks);
        ok = false;
    }

    for (int32_t s = 0; s < N_SEQ; ++s) {
        if (grouped.logits[s].size() != (size_t) n_vocab || interleaved.logits[s].size() != (size_t) n_vocab) {
            fprintf(stderr, "FAIL: missing logits for sequence %d\n", s);
            ok = false;
            continue;
        }
        float max_diff = 0.0f;
        for (int32_t v = 0; v < n_vocab; ++v) {
            max_diff = std::fmax(max_diff, std::fabs(grouped.logits[s][v] - interleaved.logits[s][v]));
        }
        printf("seq %d logits: max |diff| = %.6f\n", s, max_diff);
        if (max_diff > LOGIT_TOL) {
            fprintf(stderr, "FAIL: sequence %d logits differ between grouped and interleaved batches (%.6f)\n", s, max_diff);
            ok = false;
        }
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}