// Verify ggml_scale_row_inplace() for every round-trippable quant type.

#include "ggml.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void generate_data(size_t n, float * dst) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = 0.1f + 2.0f * cosf((float) i);
    }
}

static double relative_error(const std::vector<float> & after, const std::vector<float> & before, float scale) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < after.size(); ++i) {
        const double expected = before[i] * scale;
        num += std::fabs(after[i] - expected);
        den += std::fabs(expected);
    }
    return den > 0.0 ? num / den : num;
}

static bool test_f32() {
    constexpr int64_t ne0 = 256;
    constexpr float scale = 0.37f;
    std::vector<float> data(ne0);
    generate_data(data.size(), data.data());

    std::vector<float> after = data;
    if (!ggml_type_can_scale_row(GGML_TYPE_F32)) {
        printf("f32: can_scale_row returned false\n");
        return false;
    }
    if (!ggml_scale_row_inplace(GGML_TYPE_F32, after.data(), ne0, scale)) {
        printf("f32: scale_row_inplace returned false\n");
        return false;
    }
    for (int64_t i = 0; i < ne0; ++i) {
        if (after[i] != data[i] * scale) {
            printf("f32: mismatch at %lld\n", (long long) i);
            return false;
        }
    }
    printf("f32: OK\n");
    return true;
}

static bool test_f16_bf16() {
    constexpr int64_t ne0 = 256;
    constexpr float scale = 0.37f;
    std::vector<float> data(ne0);
    generate_data(data.size(), data.data());

    {
        std::vector<ggml_fp16_t> row(ne0);
        for (int64_t i = 0; i < ne0; ++i) {
            row[i] = ggml_fp32_to_fp16(data[i]);
        }
        if (!ggml_type_can_scale_row(GGML_TYPE_F16) ||
            !ggml_type_scales_exactly(GGML_TYPE_F16) ||
            !ggml_scale_row_inplace(GGML_TYPE_F16, row.data(), ne0, scale)) {
            printf("f16: scaling failed\n");
            return false;
        }
        double num = 0.0, den = 0.0;
        for (int64_t i = 0; i < ne0; ++i) {
            const double got = ggml_fp16_to_fp32(row[i]);
            const double want = ggml_fp16_to_fp32(ggml_fp32_to_fp16(data[i])) * scale;
            num += std::fabs(got - want);
            den += std::fabs(want);
        }
        if (den > 0.0 && num / den > 0.005) {
            printf("f16: error %f\n", num / den);
            return false;
        }
    }

    {
        std::vector<ggml_bf16_t> row(ne0);
        for (int64_t i = 0; i < ne0; ++i) {
            row[i] = ggml_fp32_to_bf16(data[i]);
        }
        if (!ggml_type_can_scale_row(GGML_TYPE_BF16) ||
            !ggml_type_scales_exactly(GGML_TYPE_BF16) ||
            !ggml_scale_row_inplace(GGML_TYPE_BF16, row.data(), ne0, scale)) {
            printf("bf16: scaling failed\n");
            return false;
        }
        double num = 0.0, den = 0.0;
        for (int64_t i = 0; i < ne0; ++i) {
            const double got = ggml_bf16_to_fp32(row[i]);
            const double want = ggml_bf16_to_fp32(ggml_fp32_to_bf16(data[i])) * scale;
            num += std::fabs(got - want);
            den += std::fabs(want);
        }
        if (den > 0.0 && num / den > 0.01) {
            printf("bf16: error %f\n", num / den);
            return false;
        }
    }

    printf("f16/bf16: OK\n");
    return true;
}

int main() {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    int n_tested = 0;
    int n_failed = 0;

    if (!test_f32()) {
        ++n_failed;
    }
    if (!test_f16_bf16()) {
        ++n_failed;
    }

    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const auto type = (ggml_type) i;
        const ggml_type_traits_t tr = ggml_internal_get_type_traits(type);
        if (!tr.type_name || tr.type_name[0] == '\0' || tr.blck_size == 0) {
            continue;
        }
        if (!tr.to_float || !tr.from_float) {
            continue;
        }

        const char * name = tr.type_name;
        const bool repacked = strstr(name, "_r4") || strstr(name, "_r8") || strstr(name, "_r16") ||
                              strstr(name, "_x4") || strstr(name, "_x8");
        const bool implicit_scale = type == GGML_TYPE_IQ1_BN || type == GGML_TYPE_IQ2_BN;
        const bool e8m0_scale = type == GGML_TYPE_MXFP4;
        if (repacked || implicit_scale || e8m0_scale) {
            printf("%-16s: skip (%s)\n", name, repacked ? "repacked after load" :
                    implicit_scale ? "implicit scale" : "E8M0 scale");
            continue;
        }

        const int64_t ne0 = tr.blck_size * 8;
        std::vector<float> data(ne0);
        generate_data(data.size(), data.data());

        std::vector<uint8_t> row(ggml_row_size(type, ne0));
        ggml_quantize_init(type);
        tr.from_float(data.data(), row.data(), ne0);

        std::vector<float> before(ne0);
        std::vector<float> after(ne0);
        tr.to_float(row.data(), before.data(), ne0);

        constexpr float scale = 0.37f;
        bool ok = ggml_type_can_scale_row(type);
        const bool exact = ggml_type_scales_exactly(type);
        ok = ok && ggml_scale_row_inplace(type, row.data(), ne0, scale);
        if (ok) {
            tr.to_float(row.data(), after.data(), ne0);
        }

        const double err = ok ? relative_error(after, before, scale) : 1.0;
        // Exact layouts: 0.5% (catches a missing dmin); fallback: 20%.
        const double tol = exact ? 0.005 : 0.20;
        const bool failed = !ok || err > tol;
        ++n_tested;
        n_failed += failed;
        if (failed) {
            printf("%-16s: FAILED (err %.6f%s)\n", tr.type_name, err, ok ? "" : ", unsupported");
        } else {
            printf("%-16s: OK (%s, err %.6f)\n", tr.type_name, exact ? "exact" : "fallback", err);
        }
    }

    ggml_free(ctx);

    printf("%d types tested, %d failed\n", n_tested, n_failed);
    return n_failed > 0 ? 1 : 0;
}
