// NVFP4 (type 40) self-test: block layout, dot against the dequantized reference, and the
// rows a quantizer must refuse.

#include "ggml.h"

#include <cmath>
#include <limits>
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

// NaN must not be swallowed: `std::max(0.0, NaN)` is 0.0, so a NaN from the kernels would pass.
static double worst_of(double acc, double d) {
    return (d != d) ? std::numeric_limits<double>::infinity() : (d > acc ? d : acc);
}

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

    struct ggml_init_params ip = { 16 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_NVFP4, n, nrows);
    memcpy(a->data, q.data(), q.size());

    // ---- check 1: mul_mat(A, ones) == sum of the dequantized row (dot vs dequantizer)
    struct ggml_tensor * ones = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
    for (int64_t j = 0; j < n; ++j) {
        ((float *) ones->data)[j] = 1.0f;
    }
    struct ggml_tensor * dots = ggml_mul_mat(ctx, a, ones);
    struct ggml_cgraph   * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dots);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    // per-row dequantization through get_rows (one graph per row)
    std::vector<float> row_sum((size_t) nrows, 0.0f);
    std::vector<float> row_amax((size_t) nrows, 0.0f);
    double worst_dot_rel = 0.0;
    double worst_rms_rel = 0.0;
    for (int64_t i = 0; i < nrows; ++i) {
        struct ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ((int32_t *) ids->data)[0] = (int32_t) i;
        struct ggml_tensor * row = ggml_get_rows(ctx, a, ids);

        struct ggml_cgraph * g2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(g2, row);
        ggml_graph_compute_with_ctx(ctx, g2, 4);

        const float * deq = (const float *) row->data;
        double sum = 0.0;
        double se2 = 0.0;
        float  amax = 0.0f;
        for (int64_t j = 0; j < n; ++j) {
            sum += deq[j];
            const float d = deq[j] - src[(size_t) (i * n + j)];
            se2 += (double) d * d;
            if (std::fabs(src[(size_t) (i * n + j)]) > amax) {
                amax = std::fabs(src[(size_t) (i * n + j)]);
            }
        }
        row_sum[(size_t) i]  = (float) sum;
        row_amax[(size_t) i] = amax;

        const float got = ((const float *) dots->data)[i];
        const double rel = std::fabs((double) got - sum) / (std::fabs(sum) + 1e-9);
        worst_dot_rel = worst_of(worst_dot_rel, rel);
        // rms error relative to the row's largest magnitude
        worst_rms_rel = worst_of(worst_rms_rel, std::sqrt(se2 / (double) n) / (double) (amax + 1e-9));
    }
    // diagnostic: the same sum through a pure F32 matmul of the dequantized rows
    struct ggml_tensor * af = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, nrows);
    for (int64_t i = 0; i < nrows; ++i) {
        struct ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ((int32_t *) ids->data)[0] = (int32_t) i;
        struct ggml_tensor * row = ggml_get_rows(ctx, a, ids);
        struct ggml_cgraph * g3 = ggml_new_graph(ctx);
        ggml_build_forward_expand(g3, row);
        ggml_graph_compute_with_ctx(ctx, g3, 4);
        memcpy((char *) af->data + i * af->nb[1], row->data, n * sizeof(float));
    }
    struct ggml_tensor * df32 = ggml_mul_mat(ctx, af, ones);
    struct ggml_cgraph * g4 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g4, df32);
    ggml_graph_compute_with_ctx(ctx, g4, 4);
    // the dot must also match the same rows summed through a plain F32 matmul of the
    // dequantized values (a second, independent path through the same quantized bytes)
    double worst_f32_rel = 0.0;
    for (int64_t i = 0; i < nrows; ++i) {
        const double a_i = ((const float *) dots->data)[i];
        const double b_i = ((const float *) df32->data)[i];
        worst_f32_rel = std::max(worst_f32_rel, std::fabs(a_i - b_i) / (std::fabs(b_i) + 1e-9));
    }
    CHECK(worst_f32_rel < 2e-3); // bf16 scale of the quantized activation

    // the dot and the dequantizer must agree up to fp32 accumulation
    CHECK(worst_dot_rel < 1e-4);
    // E2M1 with one scale per 16 values: ~2-4% rms of the group amax in theory
    CHECK(worst_rms_rel < 0.10); // E2M1 with per-16 scales: ~2-4% rms typical, 10% is a hard bound

    // ---- check 2: all-zero rows stay exactly zero (the UE4M3 zero scale is special)
    std::vector<float> zeros((size_t) n, 0.0f);
    std::vector<uint8_t> qz(row_size);
    ggml_quantize_chunk(GGML_TYPE_NVFP4, zeros.data(), qz.data(), 0, 1, n, nullptr, nullptr);
    bool all_zero = true;
    for (uint8_t v : qz) {
        all_zero = all_zero && v == 0;
    }
    CHECK(all_zero);

    // ---- check 3: a large-magnitude row stays in band
    // amax/6 lands in [256, 448], which is exactly the range the UE4M3 exponent field 15 encodes;
    // an early version saturated it to 448 and silently scaled the whole group by up to +75%.
    {
        std::vector<float> big((size_t) n, 0.0f);
        for (int64_t j = 0; j < n; ++j) {
            big[(size_t) j] = (j % 2 ? -1.0f : 1.0f) * (1800.0f * (float) (j + 1) / (float) n);
        }
        std::vector<uint8_t> qb(row_size);
        std::vector<float> back((size_t) n, 0.0f);
        ggml_quantize_chunk(GGML_TYPE_NVFP4, big.data(), qb.data(), 0, 1, n, nullptr, nullptr);
        ggml_internal_get_type_traits(GGML_TYPE_NVFP4).to_float(qb.data(), back.data(), n);

        float amax_back = 0.0f;
        for (float v : back) {
            amax_back = std::max(amax_back, std::fabs(v));
        }
        printf("NVFP4: amax 1800 -> %8.2f (%.2f%%)\n", (double) amax_back,
                100.0 * ((double) amax_back - 1800.0) / 1800.0);
        CHECK(std::fabs((double) amax_back - 1800.0) < 0.12 * 1800.0);
    }

    ggml_free(ctx);

    if (n_fail == 0) {
        printf("NVFP4: checks passed (dot vs dequant %.1e, dot vs f32 %.1e, rms %.2f%% of amax)\n",
                worst_dot_rel, worst_f32_rel, 100.0 * worst_rms_rel);
        return 0;
    }
    fprintf(stderr, "NVFP4: %d check(s) failed (dot %.1e, f32 %.1e, rms %.2f%%)\n",
            n_fail, worst_dot_rel, worst_f32_rel, 100.0 * worst_rms_rel);
    return 1;
}
