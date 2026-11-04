// NVFP4 (type 40) self-test: quantize -> mul_mat round trip against the float reference.
//
// NVFP4 is a ModelOpt export format (64-value super-blocks, four UE4M3 scales of 16), so the
// interesting properties are: the block geometry matches the GGUF layout, the round-to-nearest
// quantizer lands in the right error band, and the dequantize + vec_dot path used by
// ggml_mul_mat agrees with the float reference. Everything here is public ggml API.

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

int main() {
    const int64_t nrows = 4;
    const int64_t n     = 256; // 4 NVFP4 super-blocks per row

    // block geometry
    const size_t row_size = ggml_row_size(GGML_TYPE_NVFP4, n);
    CHECK(row_size == 4 * 36);                    // 4 super-blocks of 36 bytes
    CHECK(ggml_blck_size(GGML_TYPE_NVFP4) == 64);
    CHECK(ggml_type_size(GGML_TYPE_NVFP4) == 36);
    CHECK(ggml_is_quantized(GGML_TYPE_NVFP4));

    // deterministic source data, comfortably inside the E2M1 range
    std::vector<float> src((size_t) nrows * n);
    for (int64_t i = 0; i < nrows * n; ++i) {
        src[(size_t) i] = 6.0f * std::sin(0.07f * (float) i) + 0.5f * std::cos(0.31f * (float) i);
    }

    std::vector<uint8_t> q((size_t) nrows * row_size);
    const size_t written = ggml_quantize_chunk(GGML_TYPE_NVFP4, src.data(), q.data(), 0,
            nrows, n, nullptr, nullptr);
    CHECK(written == (size_t) nrows * row_size);

    // round trip through ggml_mul_mat: sum_j q[i][j] * 1 == row sum of the dequantized row
    struct ggml_init_params ip = { 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, n, nrows);
    memcpy(a->data, q.data(), q.size());

    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
    for (int64_t j = 0; j < n; ++j) {
        ((float *) b->data)[j] = 1.0f;
    }

    struct ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    struct ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const float * got = (const float *) out->data;
    double worst_rel = 0.0;
    for (int64_t i = 0; i < nrows; ++i) {
        double ref = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            ref += src[(size_t) (i * n + j)];
        }
        const double rel = std::fabs((double) got[i] - ref) / (std::fabs(ref) + 1e-9);
        worst_rel = std::max(worst_rel, rel);
        if (rel > 0.02) {
            fprintf(stderr, "row %3d: got %12.6f, reference %12.6f (%.3f%%)\n",
                    (int) i, (double) got[i], ref, 100.0 * rel);
        }
    }
    // 4-bit E2M1 with one scale per 16 values: per-element error averages out over 256 terms
    CHECK(worst_rel < 0.02);

    // all-zero rows must stay exactly zero (the UE4M3 zero scale is special)
    std::vector<float> zeros((size_t) n, 0.0f);
    std::vector<uint8_t> qz(row_size);
    ggml_quantize_chunk(GGML_TYPE_NVFP4, zeros.data(), qz.data(), 0, 1, n, nullptr, nullptr);
    bool all_zero = true;
    for (uint8_t v : qz) {
        all_zero = all_zero && v == 0;
    }
    CHECK(all_zero);

    ggml_free(ctx);

    if (n_fail == 0) {
        printf("NVFP4: all checks passed (worst row-sum error %.4f%%)\n", 100.0 * worst_rel);
        return 0;
    }
    fprintf(stderr, "NVFP4: %d check(s) failed\n", n_fail);
    return 1;
}
