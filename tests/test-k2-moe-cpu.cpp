// Unit test for the K2 trellis (t16) MoE matmul CPU kernel.
//
// ggml/src/iqk/iqk_cpu_ops.cpp implements iqk_t16_mul_mat_id() twice: a scalar
// reference (k2_t16_moe_ref, the specification of the op) and an AVX2 kernel
// (k2_t16_moe_avx2) that iqk_t16_mul_mat_id dispatches to when the shapes allow it.
// This test diffs the two on random trellis tensors for K = 2..6 and the (in,out)
// shapes of the 35B MoE pack, and reports the max abs/rel error.
//
// The reference below is a verbatim copy of k2_t16_moe_ref(); it has to be kept in
// step with iqk_cpu_ops.cpp by hand if the op semantics ever change. The fp16
// conversions go through ggml_fp16_to_fp32()/ggml_fp32_to_fp16(), i.e. the exact
// same helpers (GGML_FP16_TO_FP32/GGML_FP32_TO_FP16) that the library uses.
//
// Build (the normal way):  cmake --build build-cuda --target test-k2-moe-cpu
// Run:                     ./build-cuda/bin/test-k2-moe-cpu

#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined __F16C__
#include <immintrin.h>
#endif

extern "C" void iqk_t16_mul_mat_id(struct ggml_tensor * dst, int ith, int nth);

namespace {

// ---------------------------------------------------------------------------
// scalar reference: verbatim copy of k2_t16_moe_ref() in ggml/src/iqk/iqk_cpu_ops.cpp
// ---------------------------------------------------------------------------

static const uint8_t k2_t16_perm[256] = {
      0,  16, 128, 144,   8,  24, 136, 152,  32,  48, 160, 176,  40,  56, 168, 184,
     64,  80, 192, 208,  72,  88, 200, 216,  96, 112, 224, 240, 104, 120, 232, 248,
      1,  17, 129, 145,   9,  25, 137, 153,  33,  49, 161, 177,  41,  57, 169, 185,
     65,  81, 193, 209,  73,  89, 201, 217,  97, 113, 225, 241, 105, 121, 233, 249,
      2,  18, 130, 146,  10,  26, 138, 154,  34,  50, 162, 178,  42,  58, 170, 186,
     66,  82, 194, 210,  74,  90, 202, 218,  98, 114, 226, 242, 106, 122, 234, 250,
      3,  19, 131, 147,  11,  27, 139, 155,  35,  51, 163, 179,  43,  59, 171, 187,
     67,  83, 195, 211,  75,  91, 203, 219,  99, 115, 227, 243, 107, 123, 235, 251,
      4,  20, 132, 148,  12,  28, 140, 156,  36,  52, 164, 180,  44,  60, 172, 188,
     68,  84, 196, 212,  76,  92, 204, 220, 100, 116, 228, 244, 108, 124, 236, 252,
      5,  21, 133, 149,  13,  29, 141, 157,  37,  53, 165, 181,  45,  61, 173, 189,
     69,  85, 197, 213,  77,  93, 205, 221, 101, 117, 229, 245, 109, 125, 237, 253,
      6,  22, 134, 150,  14,  30, 142, 158,  38,  54, 166, 182,  46,  62, 174, 190,
     70,  86, 198, 214,  78,  94, 206, 222, 102, 118, 230, 246, 110, 126, 238, 254,
      7,  23, 135, 151,  15,  31, 143, 159,  39,  55, 167, 183,  47,  63, 175, 191,
     71,  87, 199, 215,  79,  95, 207, 223, 103, 119, 231, 247, 111, 127, 239, 255,
};

static inline uint32_t k2_t16_fshr(uint32_t lo, uint32_t hi, uint32_t s) {
    s &= 31;
    return s == 0 ? lo : (hi << (32 - s)) | (lo >> s);
}

// Same conversions the library is compiled with (ggml-impl.h: with __F16C__,
// GGML_FP16_TO_FP32(x) is _cvtsh_ss and GGML_FP32_TO_FP16(x) is _cvtss_sh(x, 0)).
// The non-F16C fallback goes through the public helpers, which need ggml_init()
// (they are table based) - main() calls it.
#if defined __F16C__
static inline float k2_fp16_to_fp32(uint16_t h) { return _cvtsh_ss(h); }
static inline uint16_t k2_fp32_to_fp16(float f) { return _cvtss_sh(f, 0); }
#else
static inline float k2_fp16_to_fp32(uint16_t h) { return ggml_fp16_to_fp32((ggml_fp16_t) h); }
static inline uint16_t k2_fp32_to_fp16(float f) { return (uint16_t) ggml_fp32_to_fp16(f); }
#endif

static inline float k2_t16_mul1(uint16_t w) {
    const uint32_t x = uint32_t(w) * 0x83DCD12Du;
    const uint32_t s = 0x6400u + ((x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24));
    const float h = k2_fp16_to_fp32((uint16_t) s);
    const float v = h * 0.00676727294921875f + (-10.3828125f);
    return k2_fp16_to_fp32(k2_fp32_to_fp16(v));
}

static void fast_ht(int n, float * values) {
    constexpr float ksqrt2 = 0.707106781f;
    float scale = 1;
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                float x = values[j], y = values[j+h];
                values[j+0] = x + y;
                values[j+h] = x - y;
            }
        }
        scale *= ksqrt2;
    }
    for (int i = 0; i < n; ++i) values[i] *= scale;
}

static void ref_mul_mat_id(struct ggml_tensor * dst, int ith, int nth) {
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const ggml_tensor * su  = dst->src[3];
    const ggml_tensor * sv  = dst->src[4];

    const int K = int(w->ne[0] / 16);
    const int J = int(w->ne[1]);
    const int I = int(w->ne[2]);
    (void) int(w->ne[3]);   // E is unused here (the real reference asserts on it)
    const int in = 16*I, out = 16*J;
    const int n = int(x->ne[1]);
    const int n_used = int(ids->ne[0]);
    const int nw32 = 8*K;

    std::vector<float> xp((size_t) in);
    std::vector<float> yp((size_t) out);

    const int64_t total = (int64_t) n * n_used;
    for (int64_t tu = ith; tu < total; tu += nth) {
        const int u = int(tu % n_used);
        const int t = int(tu / n_used);
        const int32_t e = *(const int32_t *)((const char *) ids->data + u*ids->nb[0] + t*ids->nb[1]);

        const float * xcol = (const float *)((const char *) x->data + t * x->nb[1]);
        for (int i = 0; i < in; ++i) {
            float v = xcol[i];
            if (su) {
                v *= *(const float *)((const char *) su->data + (int64_t) e * su->nb[1] + (int64_t) i * 4);
            }
            xp[i] = v;
        }
        for (int blk = 0; blk < in/128; ++blk) { fast_ht(128, xp.data() + (size_t) blk * 128); }

        std::fill(yp.begin(), yp.end(), 0.0f);
        const char * wbase = (const char *) w->data + (size_t) e * w->nb[3];
        for (int j = 0; j < J; ++j) {
            for (int i = 0; i < I; ++i) {
                const uint32_t * tile = (const uint32_t *)(wbase + (size_t) i * w->nb[2] + (size_t) j * w->nb[1]);
                uint16_t win[256];
                for (int tt = 0; tt < 128; ++tt) {
                    const int b0 = tt*2*K + K - 16 + 256*K;
                    const int b2 = b0 + K + 16;
                    const int i0 = (b0 / 32) % nw32;
                    const int i1u = (b2 - 1) / 32;
                    const int i1 = i1u % nw32;
                    const uint32_t w1 = k2_t16_fshr(tile[i1], tile[i0], uint32_t((i1u + 1)*32 - b2));
                    win[2*tt]     = uint16_t((w1 >> K) & 0xFFFFu);
                    win[2*tt + 1] = uint16_t(w1 & 0xFFFFu);
                }
                float m[256];
                for (int tt = 0; tt < 256; ++tt) {
                    m[k2_t16_perm[tt]] = k2_t16_mul1(win[tt]);
                }
                for (int rr = 0; rr < 16; ++rr) {
                    const float xv = xp[(size_t) 16*i + rr];
                    const float * mr = m + 16*rr;
                    float * yo = yp.data() + (size_t) 16*j;
                    for (int c = 0; c < 16; ++c) {
                        yo[c] += mr[c] * xv;
                    }
                }
            }
        }
        for (int blk = 0; blk < out/128; ++blk) { fast_ht(128, yp.data() + (size_t) blk * 128); }
        float * ycol = (float *)((char *) dst->data + u*dst->nb[1] + t*dst->nb[2]);
        for (int o = 0; o < out; ++o) {
            float v = yp[o];
            if (sv) {
                v *= *(const float *)((const char *) sv->data + (int64_t) e * sv->nb[1] + (int64_t) o * 4);
            }
            ycol[o] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

struct Tensor {
    ggml_tensor t;
    std::vector<uint8_t> buf;

    void init(ggml_type type, std::vector<int64_t> ne) {
        std::memset(&t, 0, sizeof(t));
        t.type = type;
        size_t n = 1;
        for (size_t i = 0; i < ne.size(); ++i) {
            t.ne[i] = ne[i];
            n *= (size_t) ne[i];
        }
        const size_t es = (type == GGML_TYPE_I16) ? 2 : 4;
        t.nb[0] = es;
        for (int i = 1; i < 4; ++i) t.nb[i] = t.nb[i-1] * (size_t) t.ne[i-1];
        buf.assign(n * es, 0);
        t.data = buf.data();
    }
    template <typename T> T * ptr() { return reinterpret_cast<T *>(t.data); }
    template <typename T> const T * cptr() const { return reinterpret_cast<const T *>(t.data); }
};

struct Result {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double scale = 0.0;      // max |reference| of the whole tensor
    long long n_exact = 0;
    long long n_total = 0;
};

Result compare(const float * a, const float * b, size_t n) {
    Result r;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        const double s = std::fabs((double) b[i]);
        if (d > r.max_abs) r.max_abs = d;
        if (s > r.scale) r.scale = s;
        if (s > 1e-4) {
            const double rel = d / s;
            if (rel > r.max_rel) r.max_rel = rel;
        }
        if (a[i] == b[i]) ++r.n_exact;
        ++r.n_total;
    }
    return r;
}

struct Case {
    int K, in, out, E, n, n_used, nth;
    bool use_su_sv;
    const char * tag;
};

bool run_case(const Case & c, std::mt19937 & rng) {
    const int I = c.in / 16, J = c.out / 16;
    Tensor w, x, ids, su, sv, y_new, y_ref;
    w.init(GGML_TYPE_I16, {16*c.K, J, I, c.E});
    x.init(GGML_TYPE_F32, {c.in, c.n});
    ids.init(GGML_TYPE_I32, {c.n_used, c.n});
    su.init(GGML_TYPE_F32, {c.in, c.E});
    sv.init(GGML_TYPE_F32, {c.out, c.E});
    y_new.init(GGML_TYPE_F32, {c.out, c.n_used, c.n});
    y_ref.init(GGML_TYPE_F32, {c.out, c.n_used, c.n});

    // every bit pattern of the packed trellis is reachable from random words
    auto * wd = w.ptr<uint16_t>();
    for (size_t i = 0; i < w.buf.size() / 2; ++i) wd[i] = (uint16_t) rng();
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    for (int i = 0; i < c.in * c.n; ++i) x.ptr<float>()[i] = nd(rng);
    for (int i = 0; i < c.in * c.E; ++i) su.ptr<float>()[i] = ud(rng);
    for (int i = 0; i < c.out * c.E; ++i) sv.ptr<float>()[i] = ud(rng);
    std::uniform_int_distribution<int> ed(0, c.E - 1);
    for (int i = 0; i < c.n_used * c.n; ++i) ids.ptr<int32_t>()[i] = ed(rng);

    y_new.t.src[0] = &w.t; y_new.t.src[1] = &x.t; y_new.t.src[2] = &ids.t;
    y_new.t.src[3] = c.use_su_sv ? &su.t : nullptr;
    y_new.t.src[4] = c.use_su_sv ? &sv.t : nullptr;
    std::memcpy(&y_ref.t, &y_new.t, sizeof(ggml_tensor));
    y_ref.t.data = y_ref.buf.data();

    for (int ith = 0; ith < c.nth; ++ith) iqk_t16_mul_mat_id(&y_new.t, ith, c.nth);
    for (int ith = 0; ith < c.nth; ++ith) ref_mul_mat_id  (&y_ref.t, ith, c.nth);

    const Result r = compare(y_new.cptr<float>(), y_ref.cptr<float>(), y_new.buf.size() / 4);
    // the two paths are expected to agree bit for bit; the tolerance only exists so the
    // test is still meaningful on a build that contracts the reference's mul+add into FMA
    const bool ok = r.max_abs <= 1e-5 * r.scale;
    std::printf("K=%d in=%5d out=%5d E=%d n=%d n_used=%d nth=%d su/sv=%d | max_abs=%.3e max_rel=%.3e bit-exact=%lld/%lld %s\n",
            c.K, c.in, c.out, c.E, c.n, c.n_used, c.nth, c.use_su_sv ? 1 : 0,
            r.max_abs, r.max_rel, r.n_exact, r.n_total, ok ? "OK" : "FAIL");
    return ok;
}

} // namespace

int main() {
    std::mt19937 rng(1234);

    // only needed for the non-F16C fallback conversions
    ggml_init_params ip = { 1u << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    const int shapes[][2] = { {2048, 512}, {2048, 2048}, {1536, 8960} };
    bool ok = true;

    // 1) all K on the three shapes of the 35B MoE pack, single token, 2 experts
    for (int K = 2; K <= 6; ++K) {
        for (auto & s : shapes) {
            Case c{K, s[0], s[1], 4, 1, 2, 1, true, "decode"};
            ok &= run_case(c, rng);
        }
    }

    // 2) su/sv == NULL (identity), several tokens, several threads
    for (int K = 2; K <= 6; ++K) {
        Case c{K, 2048, 512, 6, 3, 3, 3, false, "no-su/sv, batched, nth=3"};
        ok &= run_case(c, rng);
    }

    // 3) prefill-shaped batch: many tokens per expert (exercises the NT=4 grouping)
    for (int K : {2, 4, 6}) {
        Case c{K, 2048, 512, 8, 17, 4, 2, true, "prefill"};
        ok &= run_case(c, rng);
    }

    // 4) more threads than work items: idle threads must not corrupt anything
    for (int K : {2, 6}) {
        Case c{K, 2048, 512, 4, 1, 2, 6, true, "nth > work"};
        ok &= run_case(c, rng);
    }

    std::printf("%s\n", ok ? "all cases passed" : "FAILURES");
    ggml_free(ctx);
    return ok ? 0 : 1;
}
