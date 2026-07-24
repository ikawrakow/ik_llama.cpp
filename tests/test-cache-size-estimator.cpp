// Unit test for llama_model::cache_size()'s SWA-ring-aware sizing formula
// (src/llama-model.cpp), independent of the --fit multi-device split-mode-graph
// path that is its only real caller (get_layer_sizes() in src/llama.cpp, gated
// behind 2+ devices) -- that path can't be exercised on a single-device/CPU-only
// build, so this pins the formula directly against a loaded tiny model instead.
#include "llama.h"
#include "llama-model.h"
#include "llama-cparams.h"
#include "get-model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char * argv[]) {
    auto * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();
    auto params = llama_model_params{};
    params.use_mmap = false;
    auto * model = (llama_model *) llama_model_load_from_file(model_path, params);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model at '%s'\n", model_path);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    const auto & hparams = model->hparams;
    int il_swa = -1, il_full = -1;
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        if (hparams.swa_layers[i] && il_swa < 0)  il_swa  = (int) i;
        if (!hparams.swa_layers[i] && il_full < 0) il_full = (int) i;
    }
    if (il_swa < 0 || il_full < 0) {
        fprintf(stderr, "model needs at least one SWA layer and one full-attention layer to test\n");
        llama_backend_free();
        return EXIT_FAILURE;
    }

    const uint32_t kv_size   = 768;
    const uint32_t n_ubatch  = 48;
    // Derived from the SAME shared helper the real allocation site (src/llama.cpp)
    // and cache_size() itself both call -- not a re-derived literal -- so this pins
    // "estimator matches allocator", not just "estimator matches its own formula".
    const uint32_t expect_pad = std::max<uint32_t>(llama_kv_pad_granularity(false), 256u);
    const uint32_t expect_swa_cells = GGML_PAD(hparams.n_swa + n_ubatch, expect_pad);

    auto size = [&](int il, bool swa_compress, int n_seq_max) {
        return model->cache_size(il, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F16,
                                  kv_size, /*mla_attn=*/0, n_seq_max,
                                  /*flash_attn=*/false, n_ubatch, swa_compress);
    };

    const size_t swa_on  = size(il_swa, true, 1);
    const size_t swa_off = size(il_swa, false, 1);
    const size_t full_on  = size(il_full, true, 1);
    const size_t full_off = size(il_full, false, 1);
    // The runtime falls back to full-size dense KV whenever n_seq_max > 1 (ring
    // requires single-sequence append-only use). The estimator must agree, or
    // --fit under-budgets a multi-slot (-np > 1) run: src/llama.cpp's runtime
    // gate logs "SWA ring KV needs n_seq_max == 1" and skips the shrink in that
    // case; this pins that cache_size() does the same.
    const size_t swa_on_multiseq = size(il_swa, true, 2);

    int rc = EXIT_SUCCESS;

    // swa_compress must be a strict no-op on a non-SWA layer.
    if (full_on != full_off) {
        fprintf(stderr, "FAIL: swa_compress changed size of a non-SWA layer (%zu vs %zu)\n", full_on, full_off);
        rc = EXIT_FAILURE;
    }

    // swa_compress must be a no-op when n_seq_max > 1, matching the runtime's fallback.
    if (swa_on_multiseq != swa_off) {
        fprintf(stderr, "FAIL: swa_compress shrank an SWA layer despite n_seq_max=2 (%zu vs dense %zu)\n",
                swa_on_multiseq, swa_off);
        rc = EXIT_FAILURE;
    }

    // On the SWA layer, swa_compress must shrink to exactly GGML_PAD(n_swa+n_ubatch,256)
    // cells' worth -- i.e. the same ratio as clamping kv_size down to expect_swa_cells.
    const double expect_ratio = (double) expect_swa_cells / (double) kv_size;
    const double actual_ratio = (double) swa_on / (double) swa_off;
    if (std::fabs(actual_ratio - expect_ratio) > 1e-9) {
        fprintf(stderr, "FAIL: SWA layer size ratio %.9f != expected %.9f (n_swa=%u n_ubatch=%u -> %u cells)\n",
                actual_ratio, expect_ratio, hparams.n_swa, n_ubatch, expect_swa_cells);
        rc = EXIT_FAILURE;
    }

    if (rc == EXIT_SUCCESS) {
        fprintf(stderr, "cache_size() estimator OK: swa layer %d shrinks %zu -> %zu bytes (ratio %.4f), "
                        "full layer %d unaffected (%zu bytes)\n",
                il_swa, swa_off, swa_on, actual_ratio, il_full, full_off);
    }

    llama_backend_free();
    return rc;
}
