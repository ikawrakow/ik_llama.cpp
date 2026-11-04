// GGML_OP_FUSED_MUL_UNARY on the CPU backend: the op must agree with the two-node form
// (unary then mul) that the graph builders could use instead.

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
    // mixed signs, magnitudes around 1, plus the occasional larger value
    return 3.0f * std::sin(0.43f * (float) i + 0.7f) * std::cos(0.09f * (float) i);
}

struct result {
    std::vector<float> y;
    float max_abs_diff = 0.0f;  // fused vs unary-then-mul
    int   n_exact      = 0;
};

// runs two graphs over the same inputs:
//   ref:   y = mul(unary(a), b)
//   fused: y = fused_mul_unary(a, b, op)
static result run_case(enum ggml_unary_op op, int64_t n, bool same_inputs) {
    const size_t sz = (size_t) (n * sizeof(float));

    std::vector<float> a(n), b(n);
    for (int64_t i = 0; i < n; ++i) {
        a[i] = rnd(i);
        b[i] = same_inputs ? a[i] : rnd(i + 5000);
    }

    result res;
    res.y.resize(n);

    {
        struct ggml_init_params ip = { 4 * 1024 * 1024, nullptr, false };
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * ta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        struct ggml_tensor * tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        memcpy(ta->data, a.data(), sz);
        memcpy(tb->data, b.data(), sz);

        struct ggml_tensor * u = ggml_unary(ctx, ta, op);
        struct ggml_tensor * r = ggml_mul(ctx, u, tb);
        struct ggml_tensor * f = ggml_fused_mul_unary(ctx, ta, tb, op);

        struct ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, r);
        ggml_build_forward_expand(g, f);
        ggml_graph_compute_with_ctx(ctx, g, 2);

        const float * pr = (const float *) r->data;
        const float * pf = (const float *) f->data;
        for (int64_t i = 0; i < n; ++i) {
            res.y[i] = pf[i];
            const float d = std::fabs(pr[i] - pf[i]);
            res.max_abs_diff = (float) worst_of((double) res.max_abs_diff, (double) d);
            if (memcmp(&pr[i], &pf[i], sizeof(float)) == 0) res.n_exact++;
        }
        ggml_free(ctx);
    }
    return res;
}

int main() {
    const int64_t n = 4093; // deliberately not a multiple of the kernel block size

    struct { enum ggml_unary_op op; const char * name; } ops[] = {
        { GGML_UNARY_OP_RELU,    "relu"    },
        { GGML_UNARY_OP_GELU,    "gelu"    },
        { GGML_UNARY_OP_SILU,    "silu"    },
        { GGML_UNARY_OP_SIGMOID, "sigmoid" },
    };

    for (const auto & e : ops) {
        const result r = run_case(e.op, n, false);
        printf("fused_mul_unary %-7s x*mul: max |diff| %.3e, bit-identical %d/%lld\n",
               e.name, (double) r.max_abs_diff, r.n_exact, (long long) n);

        // relu and gelu are implemented as the unary vector op followed by a multiply, i.e. the
        // same operations in the same order: they must agree bit for bit.
        if (e.op == GGML_UNARY_OP_RELU || e.op == GGML_UNARY_OP_GELU) {
            CHECK(r.n_exact == n);
        } else {
            // silu and sigmoid have dedicated fused vector routines; allow rounding differences
            CHECK(r.max_abs_diff < 1e-6f);
        }
    }

    // the relu^2 case the FFN builder relies on: fused(x, x, relu) vs sqr(relu(x))
    {
        const result r = run_case(GGML_UNARY_OP_RELU, n, true);
        printf("fused_mul_unary relu   x^2  : max |diff| %.3e, bit-identical %d/%lld\n",
               (double) r.max_abs_diff, r.n_exact, (long long) n);
        CHECK(r.max_abs_diff == 0.0f); // equal in value; signs of zero may differ
    }

    if (n_fail == 0) {
        printf("fused_mul_unary: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "fused_mul_unary: %d check(s) failed\n", n_fail);
    return 1;
}
