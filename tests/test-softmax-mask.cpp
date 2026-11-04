// soft_max_ext with broadcast masks: 2-D (per-row) and 3-D (per-head) biases.
//
// The series relaxes the "mask must be a matrix" assert so that a per-head bias can be applied
// without materialising it over the rows (the LiLiCorr lattice attention needs that). The CPU
// kernel broadcasts along ne[1] and takes `head % mask->ne[2]`, so this test pins that contract
// against a double-precision reference for three mask shapes, plus the no-mask case.

#include "ggml.h"

#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// NaN must not be swallowed by the maxima: `std::max(0.0, NaN)` returns 0.0, so a kernel that
// emits NaN would pass every tolerance below. Treat a NaN difference as infinitely bad.
static double worst_of(double acc, double d) {
    return (d != d) ? std::numeric_limits<double>::infinity() : (d > acc ? d : acc);
}

static float rnd(int64_t i) {
    return 1.5f * std::sin(0.31f * (float) i + 0.7f) + 0.4f * std::cos(0.77f * (float) i);
}

// runs soft_max_ext(a, mask) and returns the result; mask may be null
static std::vector<float> run(const std::vector<float> & a_data,
                              const std::vector<float> & m_data,
                              const int64_t ne_a[4], const int64_t ne_m[4], bool has_mask,
                              float scale) {
    struct ggml_init_params ip = { 4 * 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0], ne_a[1], ne_a[2], ne_a[3]);
    memcpy(a->data, a_data.data(), a_data.size() * sizeof(float));

    struct ggml_tensor * mask = nullptr;
    if (has_mask) {
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_m[0], ne_m[1], ne_m[2], ne_m[3]);
        memcpy(mask->data, m_data.data(), m_data.size() * sizeof(float));
    }

    struct ggml_tensor * out = ggml_soft_max_ext(ctx, a, mask, scale, 0.0f);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 2);

    std::vector<float> res((size_t) ggml_nelements(out));
    memcpy(res.data(), out->data, res.size() * sizeof(float));
    ggml_free(ctx);
    return res;
}

static double check(const char * name,
                    const std::vector<float> & a_data, const std::vector<float> & m_data,
                    const int64_t ne_a[4], const int64_t ne_m[4], bool has_mask, float scale) {
    const std::vector<float> got = run(a_data, m_data, ne_a, ne_m, has_mask, scale);

    const int64_t ncols = ne_a[0], nrows = ne_a[1], nhead = ne_a[2], nseq = ne_a[3];
    double worst = 0.0;
    for (int64_t is = 0; is < nseq; ++is) {
        for (int64_t ih = 0; ih < nhead; ++ih) {
            for (int64_t ir = 0; ir < nrows; ++ir) {
                // reference: softmax over the columns of (a*scale + mask)
                std::vector<double> x((size_t) ncols);
                double maxv = -1e30;
                for (int64_t ic = 0; ic < ncols; ++ic) {
                    const size_t ia = (size_t) (ic + ncols * (ir + nrows * (ih + nhead * is)));
                    double v = (double) a_data[ia] * (double) scale;
                    if (has_mask) {
                        const int64_t im_r = ir % ne_m[1];
                        const int64_t im_h = ih % ne_m[2];
                        const int64_t im_s = is % ne_m[3];
                        const size_t im = (size_t) (ic + ne_m[0] * (im_r + ne_m[1] * (im_h + ne_m[2] * im_s)));
                        v += (double) m_data[im];
                    }
                    x[(size_t) ic] = v;
                    if (v > maxv) maxv = v;
                }
                double sum = 0.0;
                for (int64_t ic = 0; ic < ncols; ++ic) {
                    x[(size_t) ic] = std::exp(x[(size_t) ic] - maxv);
                    sum += x[(size_t) ic];
                }
                for (int64_t ic = 0; ic < ncols; ++ic) {
                    const size_t io = (size_t) (ic + ncols * (ir + nrows * (ih + nhead * is)));
                    const double ref = x[(size_t) ic] / sum;
                    worst = worst_of(worst, std::fabs((double) got[io] - ref));
                }
            }
        }
    }
    CHECK(worst < 1e-6);
    printf("soft_max mask %-22s ne_m = {%lld,%lld,%lld,%lld}: worst |dp| %.2e\n",
           has_mask ? name : "none", (long long) ne_m[0], (long long) ne_m[1],
           (long long) ne_m[2], (long long) ne_m[3], worst);
    return worst;
}

int main() {
    const int64_t ncols = 8, nrows = 3, nhead = 4, nseq = 1;
    const int64_t ne_a[4] = { ncols, nrows, nhead, nseq };

    std::vector<float> a((size_t) (ncols * nrows * nhead * nseq));
    for (size_t i = 0; i < a.size(); ++i) a[i] = rnd((int64_t) i);

    const float scale = 0.125f;

    // 1) no mask
    {
        const int64_t ne_m[4] = { 1, 1, 1, 1 };
        check("none", a, {}, ne_a, ne_m, false, scale);
    }
    // 2) per-row 2-D mask {ncols, nrows}
    {
        const int64_t ne_m[4] = { ncols, nrows, 1, 1 };
        std::vector<float> m((size_t) (ncols * nrows));
        for (size_t i = 0; i < m.size(); ++i) m[i] = rnd((int64_t) i + 500);
        check("2-D per-row", a, m, ne_a, ne_m, true, scale);
    }
    // 3) per-head 3-D mask {ncols, nrows, nhead}: the case the series added. ne[1] must be at
    //    least the row count (the op keeps that requirement); the head axis may be narrower.
    {
        const int64_t ne_m[4] = { ncols, nrows, nhead, 1 };
        std::vector<float> m((size_t) (ncols * nrows * nhead));
        for (size_t i = 0; i < m.size(); ++i) m[i] = rnd((int64_t) i + 900);
        check("3-D per-head", a, m, ne_a, ne_m, true, scale);
    }
    // 4) narrower head count -> the kernel takes head % ne[2]
    {
        const int64_t ne_m[4] = { ncols, nrows, 2, 1 };
        std::vector<float> m((size_t) (ncols * nrows * 2));
        for (size_t i = 0; i < m.size(); ++i) m[i] = rnd((int64_t) i + 1300);
        check("3-D two heads", a, m, ne_a, ne_m, true, scale);
    }

    if (n_fail == 0) {
        printf("soft_max masks: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "soft_max masks: %d check(s) failed\n", n_fail);
    return 1;
}
