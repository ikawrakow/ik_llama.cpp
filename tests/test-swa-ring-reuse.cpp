// Pins graph reuse for the SWA ring KV cache during single-token decode.
//
// update_cache_copies() routes every ring layer to patch_ring_copies, which reads the
// per-run copy records that llm_build_ring_store leaves in llama_context::ring_copies.
// A ring layer that registers itself anywhere else is invisible to the patcher: reuse
// is refused, and llama_decode rebuilds the whole graph for every generated token.
// That is silent -- output stays correct, only throughput collapses (measured on
// Laguna-XS/L40: -10% at ring 1024 cells, -21% at ring 4608) -- so nothing but a
// direct count of rebuilds catches it.
//
// The bound is relative, not absolute: the dense path legitimately rebuilds now and
// then (kv_self.n grows in pad-sized steps), so the ring is only required not to
// rebuild materially more often than dense over the same decode run.
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <random>   // llama-context.h -> llama-sampling.h needs std::mt19937
#include <vector>

#include "llama-model.h"
#include "llama-context.h"
#include "get-model.h"

static int g_fails = 0;

static void check(bool ok, const char * what) {
    printf("  %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) {
        ++g_fails;
    }
}

struct ctx_cfg {
    bool     swa_compress;
    uint32_t n_ctx;
    uint32_t n_ubatch;
    bool     flash_attn;
};

static llama_context * make_ctx(llama_model * model, const ctx_cfg & cfg) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx         = cfg.n_ctx;
    cparams.n_batch       = cfg.n_ctx;   // the prompt is submitted as one batch
    cparams.n_ubatch      = cfg.n_ubatch;
    cparams.n_seq_max     = 1;
    cparams.n_threads       = 2;
    cparams.n_threads_batch = 2;
    cparams.swa_compress  = cfg.swa_compress;
    cparams.flash_attn    = cfg.flash_attn;
    return llama_init_from_model(model, cparams);
}

static bool decode_range(llama_context * ctx, const int32_t * toks, int32_t n, llama_pos p0) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = p0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i == n - 1;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

// rebuilds attributable to the single-token decode phase alone
static uint64_t decode_phase_rebuilds(llama_model * model, const ctx_cfg & cfg,
                                      const std::vector<int32_t> & prompt, int32_t n_gen,
                                      bool * ring_engaged, uint64_t * prefill_rebuilds) {
    llama_context * ctx = make_ctx(model, cfg);
    if (ctx == nullptr) {
        return UINT64_MAX;
    }
    *ring_engaged = llama_kv_self_is_swa_ring(ctx);

    if (!decode_range(ctx, prompt.data(), (int32_t) prompt.size(), 0)) {
        llama_free(ctx);
        return UINT64_MAX;
    }

    // count only from here: prefill ubatches legitimately rebuild
    const uint64_t before = llama_context_n_graph_rebuilds(ctx);
    *prefill_rebuilds = before;
    llama_pos pos = (llama_pos) prompt.size();
    for (int32_t i = 0; i < n_gen; ++i) {
        const int32_t tok = prompt[i % prompt.size()];
        if (!decode_range(ctx, &tok, 1, pos++)) {
            llama_free(ctx);
            return UINT64_MAX;
        }
    }
    const uint64_t after = llama_context_n_graph_rebuilds(ctx);
    llama_free(ctx);
    return after - before;
}

// Next-token logits after a prompt LONGER than the window, so which ring rows the SWA
// mask exposes actually decides the answer. This is a gross-error detector, not a
// bit-parity assertion: ring and dense reduce over the same values in a different row
// order, so exact equality is not expected (and on a tiny random-weight fixture the
// logits are nearly flat, which is why nothing here asserts on the argmax token).
static std::vector<float> last_logits(llama_model * model, const ctx_cfg & cfg,
                                      const std::vector<int32_t> & prompt, int32_t n_vocab) {
    llama_context * ctx = make_ctx(model, cfg);
    if (ctx == nullptr || !decode_range(ctx, prompt.data(), (int32_t) prompt.size(), 0)) {
        if (ctx) {
            llama_free(ctx);
        }
        return {};
    }
    const float * l = llama_get_logits_ith(ctx, -1);
    std::vector<float> out(l ? n_vocab : 0);
    for (int32_t i = 0; i < (int32_t) out.size(); ++i) {
        out[i] = l[i];
    }
    llama_free(ctx);
    return out;
}

int main(int argc, char * argv[]) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = (llama_model *) llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model at '%s'\n", model_path);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    const auto & hparams = model->hparams;
    bool has_swa_layer = false, has_full_layer = false;
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        has_swa_layer  = has_swa_layer  ||  hparams.swa_layers[il];
        has_full_layer = has_full_layer || !hparams.swa_layers[il];
    }
    if (hparams.n_swa == 0 || !has_swa_layer || !has_full_layer) {
        fprintf(stderr, "model needs n_swa > 0 and both SWA and full-attention layers to test the ring\n");
        llama_free_model(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    const uint32_t n_ubatch = 48;
    const ctx_cfg  cfg_ring  = { true,  4 * (hparams.n_swa + n_ubatch) + 512, n_ubatch, true };
    const ctx_cfg  cfg_dense = { false, cfg_ring.n_ctx,                       n_ubatch, true };

    // long enough that the ring has wrapped before generation starts, so the decode
    // phase exercises wrapped row runs and not just the pristine first pass
    const int32_t n_prompt = (int32_t) (hparams.n_swa + n_ubatch) * 2;
    const int32_t n_gen    = 32;

    std::vector<int32_t> prompt(n_prompt);
    for (int32_t i = 0; i < n_prompt; ++i) {
        prompt[i] = 1 + (i % (llama_n_vocab(model) - 1));
    }

    bool ring_engaged = false, dense_engaged = true;
    uint64_t ring_prefill = 0, dense_prefill = 0;
    const uint64_t ring_rebuilds  = decode_phase_rebuilds(model, cfg_ring,  prompt, n_gen, &ring_engaged,  &ring_prefill);
    const uint64_t dense_rebuilds = decode_phase_rebuilds(model, cfg_dense, prompt, n_gen, &dense_engaged, &dense_prefill);

    // If the counter never moved at all the comparison below would pass no matter what
    // the ring does. Prefill walks several ubatches of changing shape, so it must rebuild.
    check(ring_prefill > 0 && dense_prefill > 0,
          "rebuild counter actually counts (prefill rebuilt at least once)");

    check(ring_rebuilds  != UINT64_MAX, "ring context prefilled and generated");
    check(dense_rebuilds != UINT64_MAX, "dense context prefilled and generated");
    check(ring_engaged,   "ring KV cache engaged (test is not vacuous)");
    check(!dense_engaged, "dense reference context has no ring");

    printf("     %d generated tokens: ring rebuilt the graph %llu time(s), dense %llu time(s)\n",
           n_gen, (unsigned long long) ring_rebuilds, (unsigned long long) dense_rebuilds);

    // The defect this pins made the ring rebuild on EVERY generated token, so the
    // failing value is n_gen. Allow a small margin over dense for ring-specific
    // boundaries (a run that wraps its stripe splits in two and cannot be patched).
    check(ring_rebuilds < (uint64_t) n_gen,
          "ring does not rebuild the graph on every generated token");
    check(ring_rebuilds <= dense_rebuilds + 4,
          "ring reuses the graph about as often as dense");

    // The SWA mask decides which ring rows a query may attend to. Whatever it costs to
    // build, it has to select the same window the dense cache selects, or the ring is
    // simply attending to the wrong keys.
    const int32_t n_vocab = llama_n_vocab(model);
    const std::vector<float> lr = last_logits(model, cfg_ring,  prompt, n_vocab);
    const std::vector<float> ld = last_logits(model, cfg_dense, prompt, n_vocab);
    check(!lr.empty() && lr.size() == ld.size(), "both contexts produced logits");
    if (!lr.empty() && lr.size() == ld.size()) {
        double max_abs = 0.0;
        for (size_t i = 0; i < lr.size(); ++i) {
            const double d = lr[i] - ld[i];
            max_abs = d < 0 ? (-d > max_abs ? -d : max_abs) : (d > max_abs ? d : max_abs);
        }
        printf("     ring vs dense next-token logits: max |delta| = %.3e\n", max_abs);
        check(max_abs < 1e-2, "ring selects the same attention window as dense");
    }

    llama_free_model(model);
    llama_backend_free();

    printf("%s: %d failure(s)\n", g_fails ? "FAILED" : "OK", g_fails);
    return g_fails ? EXIT_FAILURE : EXIT_SUCCESS;
}
