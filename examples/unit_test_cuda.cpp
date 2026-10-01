//
// unit_test_cuda.cpp - byte-for-byte CUDA block-quant verification.
// Order after Joel's KT: Q8_0, Q6_0, Q5_0, Q4_0 (+imatrix where CPU uses it).
// Dispatched in production through Joel's single ggml_cuda_quantize() entry
// (kt-encoder.cu lead); this harness also exercises the per-type helpers
// directly (ggml/src/ggml-cuda/quantize_gguf.cu).
//
// For each type, three producers of GGUF quantized bytes are compared on
// identical input:
//
//   1. GPU : ggml_cuda_quantize(0,type,...,nslice=1,imatrix) (Joel single
//            entry -> ggml/src/ggml-cuda/quantize_gguf.cu helpers)
//   2. CPU : ggml_quantize_chunk       (the fork's real llama-quantize path;
//            for Q4_0/Q5_0 without imatrix/symmetric this is the vanilla ref;
//            Q6_0 always OLS make_qx; Q8_0 ignores imatrix)
//   3. REF : local copy of the quantize_row_*_ref implementations
//            (ggml/src/ggml-quants.c q8_0, q4_0, q5_0) with the
//            fp16 step done by __float2half_rn, no ggml internals
//
// TODO (HEAD adaptation): local REF copies must include fudge factors
// (FP16(fudge*d), Q8_0 uses Q6_0 fudge quirk) and Q6_0 OLS (make_qx +
// fudge, weight=x*x without imatrix) to stay byte-exact vs HEAD CPU.
// Q8_0+imatrix must expect plain (CPU ignores imatrix). Q4_0 symmetric
// stays on CPU. Until updated, gpu==cpu is the binding check; gpu==ref
// may diverge on fudge != 1 or OLS types.
//
// The triggering scenario is llama-quantize --cuda-quantize bf16 -> q8_0
// producing garbage perplexity (no NaN). gpu vs cpu answers "is the kernel
// byte-exact?"; cpu vs ref answers "is the fork's CPU path still the vanilla
// reference?"; gpu vs ref ties the two together with an independent baseline.
//
// Layout note: the 32-value quant blocks tile the flat row-major F32 buffer
// contiguously (n_per_row % 32 == 0), so rows need no bookkeeping and per-slice
// calls can be concatenated. test_slices reproduces do_quantize's ne[2]
// slicing exactly: the CUDA branch and the CPU branch both quantize one expert
// slice at a time into consecutive slots, and both must equal a single
// whole-tensor call.
//
// Edge cases (docs/cuda-quantize.md §6): all-zero, single outlier, ±max,
// exact .5 rounding ties, denormals, huge/tiny magnitudes, mixed signs.
// A >1<<20-block tensor exercises the chunk loop in the CUDA host wrapper.
//
// Build (must run with GGML_CUDA on):
//   cmake --build build --target unit_test_cuda -j
// Run:
//   unit_test_cuda [--seed N] [--device N] [--all-devices] [--big] [--huge] [--quick]
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <vector>
#include <random>
#include <algorithm>

#include "ggml.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

static int  g_seed  = 12345;
static int  g_failures = 0;
static bool g_quick = false;
static std::mt19937 g_rng(g_seed);

// ---------------------------------------------------------------------------
// Per-type plumbing
// ---------------------------------------------------------------------------

struct quant_spec {
    const char * name;
    ggml_type    type;
    int64_t      qk;
    size_t       blk_size;
    size_t (*cuda_quantize)(const float *, void *, int64_t, int64_t);
    void (*ref)(void *, const float *, int64_t, int64_t);
    bool         imatrix;
    size_t (*cuda_quantize_imatrix)(const float *, void *, int64_t, int64_t, const float *);
    void (*ref_imatrix)(void *, const float *, int64_t, int64_t, const float *);
    bool         nan_d_equal; // treat fp16 NaN scale (d) as equal even if sign/payload differs
};

// Local copy of quantize_row_q8_0_ref (ggml/src/ggml-quants.c:943). The fp16
// conversion uses __float2half_rn (nearest-even), what GGML_FP32_TO_FP16 maps
// to and what the CUDA kernel stores via __half_as_ushort.
static void ref_quantize_q8_0(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const int64_t nb = (nrows*n_per_row)/QK8_0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = src + ib*QK8_0;
        block_q8_0 *  yb = (block_q8_0 *)dst + ib;

        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            amax = fmaxf(amax, fabsf(xb[j]));
        }

        const float d  = amax/127.0f;
        const float id = d ? 1.0f/d : 0.0f;

        yb->d = (ggml_half)__half_as_ushort(__float2half_rn(d));
        for (int j = 0; j < QK8_0; ++j) {
            yb->qs[j] = (int8_t)roundf(xb[j]*id);
        }
    }
}

// Local copy of quantize_row_q4_0_ref (ggml/src/ggml-quants.c:673).
static void ref_quantize_q4_0(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const int64_t nb = (nrows*n_per_row)/QK4_0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = src + ib*QK4_0;
        block_q4_0 *  yb = (block_q4_0 *)dst + ib;

        // signed value with max |x|; first occurrence wins |x| ties
        float amax = 0.0f;
        float max  = 0.0f;
        for (int j = 0; j < QK4_0; ++j) {
            const float v = xb[j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max/-8.0f;
        const float id = d ? 1.0f/d : 0.0f;

        yb->d = (ggml_half)__half_as_ushort(__float2half_rn(d));

        // MIN(15, (int8_t)(x*id + 8.5f)): truncation toward zero, then clamp.
        // |x*id| <= 8 so the truncation is always in [0, 16] -> nibble [0, 15].
        for (int j = 0; j < QK4_0/2; ++j) {
            const uint8_t xi0 = (uint8_t)std::min(15, (int)(int8_t)(xb[j]*id + 8.5f));
            const uint8_t xi1 = (uint8_t)std::min(15, (int)(int8_t)(xb[j + QK4_0/2]*id + 8.5f));
            yb->qs[j] = (uint8_t)(xi0 | (xi1 << 4));
        }
    }
}

// ---------------------------------------------------------------------------
// Q4_0 imatrix path: make_qx_quants (ggml-quants.c:1786) + quantize_row_q4_0_impl
// ---------------------------------------------------------------------------

// Verbatim copy of ggml-quants.c:1779 (round half to even via magic constant).
static inline int ref_nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// MAX(-nmax, MIN(nmax-1, l)) as in make_qx_quants
static inline int ref_clamp_l(int l, int nmax) {
    return l > nmax-1 ? nmax-1 : (l < -nmax ? -nmax : l);
}

// Verbatim copy of make_qx_quants (ggml-quants.c:1786), restricted to the
// Q4_0 imatrix use (rmse_type == 1, qw always provided). Host code compiled
// with the same flags as the llama lib CPU path, so it must reproduce it
// byte-for-byte.
static float ref_make_qx_quants(int n, int nmax, const float * x, int8_t * L, int rmse_type, const float * qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < 1e-15f) { // GROUP_MAX_EPS: all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = ref_nearest_int(iscale * x[i]);
            L[i] = nmax + ref_clamp_l(l, nmax);
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = ref_nearest_int(iscale * x[i]);
        l = ref_clamp_l(l, nmax);
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    float best_sumlx = sumlx, best_suml2 = suml2;
    for (int is = -9; is <= 9; ++is) {
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = ref_nearest_int(iscale * x[i]);
            l = ref_clamp_l(l, nmax);
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = ref_nearest_int(iscale * x[i]);
                L[i] = nmax + ref_clamp_l(l, nmax);
            }
            scale = sumlx/suml2; best = scale*sumlx;
            best_sumlx = sumlx; best_suml2 = suml2;
        }
        iscale = (nmax-1 + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = ref_nearest_int(iscale * x[i]);
            l = ref_clamp_l(l, nmax);
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = ref_nearest_int(iscale * x[i]);
                L[i] = nmax + ref_clamp_l(l, nmax);
            }
            scale = sumlx/suml2; best = scale*sumlx;
            best_sumlx = sumlx; best_suml2 = suml2;
        }
    }

    sumlx = best_sumlx; suml2 = best_suml2;
    for (int iter = 0; iter < n*(2*nmax-1); ++iter) {
        float abs_gmax = 0, gmax = 0;
        int best_j = -1;
        for (int j = 0; j < n; ++j) {
            float w = qw ? qw[j] : rmse_type == 1 ? x[j] * x[j] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[j]) : sqrtf(fabsf(x[j]));
            int l = L[j] - nmax;
            float g = scale * w * (x[j] - scale*l);
            if ((g > 0 && l < nmax-1) || (g < 0 && l > -nmax)) {
                float ag = fabsf(g);
                if (ag > abs_gmax) {
                    abs_gmax = ag; gmax = g; best_j = j;
                }
            }
        }
        if (best_j < 0) break;

        float new_sumlx = sumlx, new_suml2 = suml2;
        float w = qw ? qw[best_j] : rmse_type == 1 ? x[best_j] * x[best_j] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[best_j]) : sqrtf(fabsf(x[best_j]));
        int l = L[best_j] - nmax;
        if (gmax > 0) {
            new_sumlx += w*x[best_j];
            new_suml2 += w*(2*l + 1);
            l += 1;
        } else {
            new_sumlx -= w*x[best_j];
            new_suml2 -= w*(2*l - 1);
            l -= 1;
        }
        if (new_suml2 > 0 && new_sumlx*new_sumlx > best*new_suml2) {
            sumlx = new_sumlx; suml2 = new_suml2;
            scale = sumlx/suml2; best = scale*sumlx;
            L[best_j] = l + nmax;
        }
        else {
            break;
        }
    }
    return scale;
}

// Host port of the CPU reference's fp16 conversion ggml_compute_fp32_to_fp16
// (ggml/src/ggml-impl.h:595, used when __F16C__ is off as in this build). The
// device kernel uses the identical bit-mask path (fp32_to_fp16_ggml), so NaN
// scales (possible from make_qx_quants on degenerate blocks) are byte-exact
// too: the CPU encodes NaN as sign|0x7e00, not the hardware conversion's 0x7fff.
static uint16_t fp32_to_fp16_ggml_host(float f) {
    const float scale_to_inf  = [](){ uint32_t u = 0x77800000u; float v; memcpy(&v, &u, 4); return v; }();
    const float scale_to_zero = [](){ uint32_t u = 0x08800000u; float v; memcpy(&v, &u, 4); return v; }();
    float base = (fabsf(f)*scale_to_inf)*scale_to_zero;

    uint32_t w; memcpy(&w, &f, 4);
    const uint32_t shl1_w = w + w;
    const uint32_t sign   = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) {
        bias = 0x71000000u;
    }
    float badd; uint32_t u2 = (bias >> 1) + 0x07800000u; memcpy(&badd, &u2, 4);
    base = badd + base;

    uint32_t bits; memcpy(&bits, &base, 4);
    const uint32_t exp_bits      = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign       = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

// Local copy of quantize_row_q4_0_impl (ggml-quants.c:3429). The imatrix holds
// n_per_row weights and is reused for every row, exactly like the CPU path.
static void ref_quantize_q4_0_imatrix(void * dst, const float * src, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * x = src + irow*n_per_row;
        block_q4_0 * y = (block_q4_0 *)dst + irow*(n_per_row/QK4_0);

        float sum_x2 = 0;
        for (int64_t j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
        float sigma2 = sum_x2/n_per_row;

        float weight[QK4_0];
        int8_t L[QK4_0];
        const int64_t nb = n_per_row/QK4_0;
        for (int64_t ib = 0; ib < nb; ++ib) {
            const float * xb = x + QK4_0 * ib;
            const float * qw = imatrix + QK4_0 * ib;
            for (int j = 0; j < QK4_0; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            float d = ref_make_qx_quants(QK4_0, 8, xb, L, 1, weight);
            y[ib].d = (ggml_half)fp32_to_fp16_ggml_host(d);
            for (int j = 0; j < QK4_0/2; ++j) {
                y[ib].qs[j] = (uint8_t)(L[j] | (L[j + QK4_0/2] << 4));
            }
        }
    }
}

// Local copy of quantize_row_q5_0_ref (ggml/src/ggml-quants.c:757). The 5-th
// bit of every quant goes into the 4-byte LE qh bitmap, which the ref memcpys
// from a native uint32.
static void ref_quantize_q5_0(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const int64_t nb = (nrows*n_per_row)/QK5_0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = src + ib*QK5_0;
        block_q5_0 *  yb = (block_q5_0 *)dst + ib;

        float amax = 0.0f;
        float max  = 0.0f;
        for (int j = 0; j < QK5_0; ++j) {
            const float v = xb[j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max/-16.0f;
        const float id = d ? 1.0f/d : 0.0f;

        yb->d = (ggml_half)__half_as_ushort(__float2half_rn(d));

        uint32_t qh = 0;
        for (int j = 0; j < QK5_0/2; ++j) {
            const uint8_t xi0 = (uint8_t)std::min(31, (int)(int8_t)(xb[j]*id + 16.5f));
            const uint8_t xi1 = (uint8_t)std::min(31, (int)(int8_t)(xb[j + QK5_0/2]*id + 16.5f));
            yb->qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
            qh |= ((uint32_t)((xi0 & 0x10u) >> 4)) << (j + 0);
            qh |= ((uint32_t)((xi1 & 0x10u) >> 4)) << (j + QK5_0/2);
        }
        memcpy(yb->qh, &qh, sizeof(qh));
    }
}

// Local copy of quantize_row_q6_0_ref (ggml/src/ggml-quants.c:853). Bits 4-5
// of every 6-bit quant go into the 8-byte qh: byte k = h_k | (h_{k+8} << 4)
// with h_e = (q_e>>4) | ((q_{e+16}>>4) << 2), matching `qh[j%8] |= h<<4*(j/8)`.
static void ref_quantize_q6_0(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const int64_t nb = (nrows*n_per_row)/QK6_0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = src + ib*QK6_0;
        block_q6_0 *  yb = (block_q6_0 *)dst + ib;

        float amax = 0.0f;
        float max  = 0.0f;
        for (int j = 0; j < QK6_0; ++j) {
            const float v = xb[j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max  = v;
            }
        }

        const float d  = max/-32.0f;
        const float id = d ? 1.0f/d : 0.0f;

        yb->d = (ggml_half)__half_as_ushort(__float2half_rn(d));

        memset(yb->qh, 0, QK6_0/4);
        for (int j = 0; j < QK6_0/2; ++j) {
            const uint8_t xi0 = (uint8_t)std::min(63, (int)(int8_t)(xb[j]*id + 32.5f));
            const uint8_t xi1 = (uint8_t)std::min(63, (int)(int8_t)(xb[j + QK6_0/2]*id + 32.5f));
            yb->qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
            const uint8_t h = (xi0 >> 4) | ((xi1 >> 4) << 2);
            yb->qh[j%(QK6_0/4)] |= (uint8_t)(h << 4*(j/(QK6_0/4)));
        }
    }
}

// Local copy of quantize_row_q5_0_impl (ggml-quants.c:3577): make_qx_quants
// with nmax == 16, plus the qh 5th-bit bitmap packing.
static void ref_quantize_q5_0_imatrix(void * dst, const float * src, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * x = src + irow*n_per_row;
        block_q5_0 * y = (block_q5_0 *)dst + irow*(n_per_row/QK5_0);

        float sum_x2 = 0;
        for (int64_t j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
        float sigma2 = sum_x2/n_per_row;

        float weight[QK5_0];
        int8_t L[QK5_0];
        const int64_t nb = n_per_row/QK5_0;
        for (int64_t ib = 0; ib < nb; ++ib) {
            const float * xb = x + QK5_0 * ib;
            const float * qw = imatrix + QK5_0 * ib;
            for (int j = 0; j < QK5_0; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            float d = ref_make_qx_quants(QK5_0, 16, xb, L, 1, weight);
            y[ib].d = (ggml_half)fp32_to_fp16_ggml_host(d);

            uint32_t qh = 0;
            for (int j = 0; j < QK5_0/2; ++j) {
                const uint8_t xi0 = (uint8_t)L[j];
                const uint8_t xi1 = (uint8_t)L[j + QK5_0/2];
                y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
                qh |= ((uint32_t)((xi0 & 0x10u) >> 4)) << (j + 0);
                qh |= ((uint32_t)((xi1 & 0x10u) >> 4)) << (j + QK5_0/2);
            }
            memcpy(y[ib].qh, &qh, sizeof(qh));
        }
    }
}

// Local copy of quantize_row_q6_0_impl (ggml-quants.c:3697): make_qx_quants
// with nmax == 32, plus the 2-bit qh packing (6-bit quants).
static void ref_quantize_q6_0_imatrix(void * dst, const float * src, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * x = src + irow*n_per_row;
        block_q6_0 * y = (block_q6_0 *)dst + irow*(n_per_row/QK6_0);

        float sum_x2 = 0;
        for (int64_t j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
        float sigma2 = sum_x2/n_per_row;

        float weight[QK6_0];
        int8_t L[QK6_0];
        const int64_t nb = n_per_row/QK6_0;
        for (int64_t ib = 0; ib < nb; ++ib) {
            const float * xb = x + QK6_0 * ib;
            const float * qw = imatrix + QK6_0 * ib;
            for (int j = 0; j < QK6_0; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            float d = ref_make_qx_quants(QK6_0, 32, xb, L, 1, weight);
            y[ib].d = (ggml_half)fp32_to_fp16_ggml_host(d);

            memset(y[ib].qh, 0, QK6_0/4);
            for (int j = 0; j < QK6_0/2; ++j) {
                const uint8_t xi0 = (uint8_t)L[j];
                const uint8_t xi1 = (uint8_t)L[j + QK6_0/2];
                y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
                const uint8_t h = (xi0 >> 4) | ((xi1 >> 4) << 2);
                y[ib].qh[j%(QK6_0/4)] |= (uint8_t)(h << 4*(j/(QK6_0/4)));
            }
        }
    }
}

// Local copy of quantize_row_q8_0_impl (ggml-quants.c:3799): a weighted
// least-squares scale refinement over the plain signed-8-bit roundf codes
// (Q8_0 cannot use make_qx_quants; see the CPU impl). The per-element weight
// and the sums match the CPU/GPU exactly.
static void ref_quantize_q8_0_imatrix(void * dst, const float * src, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * x = src + irow*n_per_row;
        block_q8_0 * y = (block_q8_0 *)dst + irow*(n_per_row/QK8_0);

        float sum_x2 = 0;
        for (int64_t j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
        float sigma2 = sum_x2/n_per_row;

        float weight[QK8_0];
        const int64_t nb = n_per_row/QK8_0;
        for (int64_t ib = 0; ib < nb; ++ib) {
            const float * xb = x + QK8_0 * ib;
            const float * qw = imatrix + QK8_0 * ib;
            for (int j = 0; j < QK8_0; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);

            float amax = 0.0f;
            for (int j = 0; j < QK8_0; ++j) amax = fmaxf(amax, fabsf(xb[j]));

            const float d0  = amax/127.0f;
            const float id0 = d0 ? 1.0f/d0 : 0.0f;

            float sumqx = 0, sumq2 = 0;
            for (int j = 0; j < QK8_0; ++j) {
                const float v = xb[j];
                const int8_t q = (int8_t)roundf(v*id0);
                y[ib].qs[j] = q;
                sumqx += weight[j]*q*v;
                sumq2 += weight[j]*q*q;
            }
            const float d = sumq2 > 0.0f ? sumqx/sumq2 : d0;
            y[ib].d = (ggml_half)fp32_to_fp16_ggml_host(d);
        }
    }
}

static void fill_random_uniform(float * dst, int64_t n) {
    std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
    for (int64_t i = 0; i < n; ++i) dst[i] = dist(g_rng);
}

// Weight-like: 90% of 32-value blocks tight around zero (N(0,0.05)), 10% wide
// (N(0,1)) - covers near-amax-scale diversity without denormal-range noise.
static void fill_random_weight_like(float * dst, int64_t n) {
    std::normal_distribution<float> tight(0.0f, 0.05f);
    std::normal_distribution<float> wide(0.0f, 1.0f);
    for (int64_t ib = 0; ib < n/QK8_0; ++ib) {
        const bool w = (g_rng() % 10) == 0;
        for (int j = 0; j < QK8_0; ++j) {
            dst[ib*QK8_0 + j] = w ? wide(g_rng) : tight(g_rng);
        }
    }
}

// Crafted edge-case blocks, one pattern per block (QK8_0 == QK4_0 == 32).
static void fill_edge_cases(float * dst, int64_t n) {
    const int64_t nb = n/QK8_0;
    const int64_t pat = 8;
    for (int64_t ib = 0; ib < nb; ++ib) {
        float * xb = dst + ib*QK8_0;
        switch (ib % pat) {
            case 0: // all zeros (d == 0 -> id == 0 path)
                for (int j = 0; j < QK8_0; ++j) xb[j] = 0.0f;
                break;
            case 1: // single outlier, rest tiny
                for (int j = 0; j < QK8_0; ++j) xb[j] = 0.001f;
                xb[g_rng() % QK8_0] = 1.0e6f;
                break;
            case 2: // values at exactly ±amax
                for (int j = 0; j < QK8_0; ++j) xb[j] = (j & 1) ? 1000.0f : -1000.0f;
                break;
            case 3: // exact .5 rounding ties (amax == 127 -> d == 1 -> id == 1)
                for (int j = 0; j < QK8_0; ++j) {
                    const float v = (float)(j % 5) + 0.5f; // 0.5 1.5 2.5 3.5 4.5
                    xb[j] = (j & 1) ? -v : v;
                }
                xb[QK8_0-1] = 127.0f;
                break;
            case 4: // denormals / subnormals
                for (int j = 0; j < QK8_0; ++j) xb[j] = (j & 1) ? 1.0e-38f : 1.0e-30f;
                break;
            case 5: // huge magnitudes
                for (int j = 0; j < QK8_0; ++j) xb[j] = (j & 1) ? 1.0e30f : 1.0e38f;
                break;
            case 6: // mixed signs, small (amax from a negative value)
                for (int j = 0; j < QK8_0; ++j) xb[j] = (j & 1) ? -0.4f : 0.1f;
                break;
            default: // moderate varied magnitudes
                for (int j = 0; j < QK8_0; ++j) xb[j] = (float)((g_rng() % 2001) - 1000)/8.0f;
                break;
        }
    }
}

// Q5_0 boundary blocks: exercises the 5-th bit / qh bitmap and the
// (int32_t)(x*id + 16.5f) truncation ties.
static void fill_q5_0_boundary(float * dst, int64_t n) {
    const int64_t nb = n/QK5_0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        float * xb = dst + ib*QK5_0;
        switch (ib % 6) {
            case 0: // x*id = -16..15 by 1 -> q = 0..31, every qh bit set
                for (int j = 0; j < QK5_0; ++j) xb[j] = (float)j - 16.0f;
                break;
            case 1: // half-integer q values; amax = -16 (id = 1)
                for (int j = 0; j < QK5_0; ++j) xb[j] = 0.5f*(float)(j % 8) + (float)(j % 3) - 10.0f;
                xb[QK5_0-1] = -16.0f;
                break;
            case 2: // q clamps at both extremes: x = ±16 -> q = 31 / 0
                for (int j = 0; j < QK5_0; ++j) xb[j] = (j & 1) ? 16.0f : -16.0f;
                break;
            case 3: // qh 16-bit boundary: q crosses 16 at x*id = -0.5
                for (int j = 0; j < QK5_0; ++j) {
                    const float v = (float)(j % 8) - 0.5f; // -0.5f 0.5f .. 7.5f
                    xb[j] = (j & 1) ? v : -v;
                }
                xb[QK5_0-1] = -16.0f;
                break;
            case 4: // q samples spread over [0, 32)
                for (int j = 0; j < QK5_0; ++j) xb[j] = ((g_rng() % 33) - 16) + 0.25f;
                break;
            default: // all zeros (d == 0 -> id == 0 path)
                for (int j = 0; j < QK5_0; ++j) xb[j] = 0.0f;
                break;
        }
    }
}

// Q6_0 boundary blocks: exercises the 6-bit levels / 2-bit qh and the
// (int32_t)(x*id + 32.5f) truncation ties.
static void fill_q6_0_boundary(float * dst, int64_t n) {
    const int64_t nb = n/QK6_0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        float * xb = dst + ib*QK6_0;
        switch (ib % 7) {
            case 0: // x*id = -32..31 by 1 -> q = 0..63, every qh 2-bit pair set
                for (int j = 0; j < QK6_0; ++j) xb[j] = (float)j - 32.0f;
                break;
            case 1: // half-integer q values; amax = -32 (id = 1)
                for (int j = 0; j < QK6_0; ++j) xb[j] = 0.5f*(float)(j % 8) + (float)(j % 3) - 20.0f;
                xb[QK6_0-1] = -32.0f;
                break;
            case 2: // q clamps at the top: x = ±32 -> q = 63 / 0
                for (int j = 0; j < QK6_0; ++j) xb[j] = (j & 1) ? 32.0f : -32.0f;
                break;
            case 3: // qh bit4 boundary: q crosses 32 at x*id = -0.5
                for (int j = 0; j < QK6_0; ++j) {
                    const float v = (float)(j % 8) - 0.5f; // -0.5f 0.5f .. 7.5f
                    xb[j] = (j & 1) ? v : -v;
                }
                xb[QK6_0-1] = -32.0f;
                break;
            case 4: // qh bit5 boundary: q crosses 64 at x*id = 31.5
                for (int j = 0; j < QK6_0; ++j) xb[j] = (float)(j % 4) + 29.5f;
                break;
            case 5: // q samples spread over [0, 64)
                for (int j = 0; j < QK6_0; ++j) xb[j] = ((g_rng() % 65) - 32) + 0.25f;
                break;
            default: // all zeros (d == 0 -> id == 0 path)
                for (int j = 0; j < QK6_0; ++j) xb[j] = 0.0f;
                break;
        }
    }
}

// Synthetic importance matrix: positive weights spanning ~5 decades, with
// every 11th column zero (exercises the w == 0 / suml2 == 0 paths).
static void fill_imatrix(float * dst, int64_t n_per_row) {
    std::uniform_real_distribution<float> dist(-2.5f, 2.5f);
    for (int64_t j = 0; j < n_per_row; ++j) {
        dst[j] = (j % 11 == 5) ? 0.0f : powf(10.0f, dist(g_rng));
    }
}

static void dump_block(const char * who, const uint8_t * blk, size_t blk_size) {
    printf("    %s: ", who);
    for (size_t j = 0; j < blk_size; ++j) printf("%02x", blk[j]);
    printf("\n");
}

// Report the first differing quant block plus the diff count.
// fp16 exponent all-ones + nonzero mantissa => NaN
static bool is_fp16_nan(uint16_t v) {
    return ((v >> 10) & 0x1f) == 0x1f && (v & 0x03ff) != 0;
}

static int64_t compare_buffers(const char * tag, const uint8_t * a, const uint8_t * b, size_t n, size_t blk_size,
        bool nan_d_equal = false) {
    int64_t ndiff = 0;
    int64_t first_blk = -1;
    for (size_t i = 0; i < n; ++i) {
        // The block scale d is the first 2 bytes (little-endian) of every
        // quant block. The imatrix quantizer can legitimately produce a NaN
        // scale for degenerate blocks (e.g. huge |x| makes x*x overflow so the
        // weights become 0*inf/inf). The sign/payload of that NaN differs
        // between x86 SSE (sign bit from the operands) and NVIDIA hardware
        // (default quiet NaN), a CPU-vendor-level semantic difference that even
        // CPU-only llama.cpp does not normalize. For such specs treat any NaN
        // d as equal so the byte comparison stays strict for every finite block.
        if (nan_d_equal && i % blk_size == 0 && i + 1 < n) {
            const uint16_t da = (uint16_t)a[i] | (uint16_t)(a[i+1] << 8);
            const uint16_t db = (uint16_t)b[i] | (uint16_t)(b[i+1] << 8);
            if (is_fp16_nan(da) && is_fp16_nan(db)) {
                i += 1;
                continue;
            }
        }
        if (a[i] != b[i]) {
            ++ndiff;
            if (first_blk < 0) first_blk = (int64_t)i / blk_size;
        }
    }
    if (first_blk >= 0) {
        const uint8_t * ra = a + first_blk*blk_size;
        const uint8_t * rb = b + first_blk*blk_size;
        printf("  [FAIL] %s: %lld/%zu bytes differ; first differing quant block %lld\n",
               tag, (long long)ndiff, n, (long long)first_blk);
        dump_block("ref", ra, blk_size);
        dump_block("gpu", rb, blk_size);
    }
    return ndiff;
}

// ---------------------------------------------------------------------------
// Byte-exact comparison of the three producers
// ---------------------------------------------------------------------------

static void test_one(const char * tag, int64_t nrows, int64_t n_per_row,
        void (*fill)(float *, int64_t), int device, const quant_spec & spec) {
    if (n_per_row % spec.qk != 0 || nrows <= 0) return;

    const int64_t nelements = nrows*n_per_row;
    std::vector<float> src(nelements);
    fill(src.data(), nelements);

    // Q4_0 imatrix specs: synthetic weights, one value per column, reused for
    // every row (matches the CPU quantize_row_q4_0_impl contract).
    std::vector<float> imat;
    if (spec.imatrix) {
        imat.resize(n_per_row);
        fill_imatrix(imat.data(), n_per_row);
    }
    const float * imatrix = spec.imatrix ? imat.data() : nullptr;

    const size_t row_size = ggml_row_size(spec.type, n_per_row);
    const size_t out_size = nrows*row_size;

    std::vector<uint8_t> out_cpu(out_size);
    std::vector<uint8_t> out_gpu(out_size);
    std::vector<uint8_t> out_ref(out_size);

    // CPU: real llama-quantize path
    const size_t nb_cpu = ggml_quantize_chunk(spec.type, src.data(), out_cpu.data(),
            0, nrows, n_per_row, imatrix, nullptr);

    // REF: local vanilla copy
    if (spec.imatrix) {
        spec.ref_imatrix(out_ref.data(), src.data(), nrows, n_per_row, imatrix);
    } else {
        spec.ref(out_ref.data(), src.data(), nrows, n_per_row);
    }
    const size_t nb_ref = out_size;

    // GPU: real CUDA path (always runs on device 0 in this POC)
    cudaSetDevice(device);
    const size_t nb_gpu = spec.imatrix
        ? spec.cuda_quantize_imatrix(src.data(), out_gpu.data(), nrows, n_per_row, imatrix)
        : spec.cuda_quantize(src.data(), out_gpu.data(), nrows, n_per_row);

    if (nb_cpu != nb_ref || nb_gpu != nb_ref) {
        printf("  [FAIL] %s: size mismatch cpu=%zu gpu=%zu ref=%zu\n", tag, nb_cpu, nb_gpu, nb_ref);
        ++g_failures;
        return;
    }

    const int64_t d_gpu_cpu = compare_buffers(tag, out_cpu.data(), out_gpu.data(), out_size, spec.blk_size, spec.nan_d_equal);
    const int64_t d_cpu_ref = compare_buffers(tag, out_ref.data(), out_cpu.data(), out_size, spec.blk_size, spec.nan_d_equal);

    if (d_gpu_cpu == 0 && d_cpu_ref == 0) {
        printf("  [OK]   %s nrows=%-6lld n_per_row=%-5lld : gpu==cpu==ref\n",
               tag, (long long)nrows, (long long)n_per_row);
    } else {
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// do_quantize slice reproduction (multi-slice tensors, ne[2] > 1)
// ---------------------------------------------------------------------------

// Reproduces do_quantize (src/llama-quantize.cpp) for a tensor with ne[2]
// expert slices: the CUDA branch and the threaded CPU branch both quantize each
// slice into consecutive slots. Also checks a single whole-tensor call gives
// the same bytes (the quantizer is contiguous, so it must).
static void test_slices(int device, int64_t ne0, int64_t ne1, int64_t ne2, const quant_spec & spec) {
    char tag[96];
    snprintf(tag, sizeof(tag), "slices ne0=%lld ne1=%lld ne2=%lld",
             (long long)ne0, (long long)ne1, (long long)ne2);

    std::vector<float> src(ne0*ne1*ne2);
    fill_random_uniform(src.data(), src.size());

    std::vector<float> imat;
    if (spec.imatrix) {
        imat.resize(ne0);
        fill_imatrix(imat.data(), ne0);
    }
    const float * imatrix = spec.imatrix ? imat.data() : nullptr;

    const size_t row_size    = ggml_row_size(spec.type, ne0);
    const size_t matrix_size = row_size*ne1;

    std::vector<uint8_t> cpu(ne2*matrix_size);
    std::vector<uint8_t> cpu_whole(ne2*matrix_size);
    std::vector<uint8_t> gpu(ne2*matrix_size);

    // CPU per-slice (do_quantize threaded path)
    for (int64_t i02 = 0; i02 < ne2; ++i02) {
        ggml_quantize_chunk(spec.type, src.data() + i02*ne0*ne1, cpu.data() + i02*matrix_size,
                0, ne1, ne0, imatrix, nullptr);
    }

    // CPU whole-tensor single call (do_quantize single-thread path)
    ggml_quantize_chunk(spec.type, src.data(), cpu_whole.data(), 0, ne1*ne2, ne0, imatrix, nullptr);

    // GPU per-slice (do_quantize CUDA branch)
    cudaSetDevice(device);
    size_t nb_gpu_total = 0;
    for (int64_t i02 = 0; i02 < ne2; ++i02) {
        nb_gpu_total += spec.imatrix
            ? spec.cuda_quantize_imatrix(src.data() + i02*ne0*ne1, gpu.data() + i02*matrix_size, ne1, ne0, imatrix)
            : spec.cuda_quantize(src.data() + i02*ne0*ne1, gpu.data() + i02*matrix_size, ne1, ne0);
    }

    if (nb_gpu_total != cpu.size()) {
        printf("  [FAIL] %s: gpu bytes %zu != cpu bytes %zu\n", tag, nb_gpu_total, cpu.size());
        ++g_failures;
        return;
    }

    const int64_t d_whole   = compare_buffers(tag, cpu.data(), cpu_whole.data(), cpu.size(), spec.blk_size);
    const int64_t d_gpu_cpu = compare_buffers(tag, cpu.data(), gpu.data(), cpu.size(), spec.blk_size);

    if (d_whole == 0 && d_gpu_cpu == 0) {
        printf("  [OK]   %s : gpu-slices==cpu-slices==cpu-whole\n", tag);
    } else {
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// Device diagnostics
// ---------------------------------------------------------------------------

static int print_devices(void) {
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) {
        printf("  [FAIL] no CUDA device available\n");
        ++g_failures;
        return 0;
    }
    for (int i = 0; i < nd; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            printf("  [INFO] device %d: %s (sm_%d%d, %d MiB)\n",
                   i, prop.name, prop.major, prop.minor, (int)(prop.totalGlobalMem >> 20));
        }
    }
    return nd;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    int device = -1; // default: first enumerated device
    bool all_devices = false;
    bool big = false;
    bool huge = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if      (arg == "--seed"  && i+1 < argc) g_seed = atoi(argv[++i]);
        else if (arg == "--device" && i+1 < argc) device = atoi(argv[++i]);
        else if (arg == "--all-devices") all_devices = true;
        else if (arg == "--big")   big = true;
        else if (arg == "--huge")  huge = true;
        else if (arg == "--quick") g_quick = true;
        else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return 1;
        }
    }
    g_rng.seed(g_seed);
    printf("=== unit_test_cuda ===\n");
    printf("seed %d%s\n", g_seed, g_quick ? ", quick mode" : "");

    const int nd = print_devices();
    if (nd == 0) return 1;

    const int devices[2] = { device >= 0 ? device : 0, 0 };
    const int ntest_dev  = all_devices ? nd : 1;

    // quantize always runs on device 0 in this POC; report which one is real.
    printf("  [INFO] ggml_cuda_quantize runs on device 0; testing device %d\n", devices[0]);

    // Joel-lead wrappers: exercise the production single entry
    // ggml_cuda_quantize(device,type,...,nslice=1,imatrix) with the spec's
    // (src,dst,nrows,n_per_row) signature. Order after KT: Q8_0, Q6_0, Q5_0, Q4_0.
    struct cuda_wraps {
        static size_t q8_0(const float * s, void * d, int64_t r, int64_t n) {
            return ggml_cuda_quantize(0, GGML_TYPE_Q8_0, s, d, r, n, 1, nullptr);
        }
        static size_t q8_0_imatrix(const float * s, void * d, int64_t r, int64_t n, const float * im) {
            return ggml_cuda_quantize(0, GGML_TYPE_Q8_0, s, d, r, n, 1, im);
        }
        static size_t q6_0(const float * s, void * d, int64_t r, int64_t n) {
            return ggml_cuda_quantize(0, GGML_TYPE_Q6_0, s, d, r, n, 1, nullptr);
        }
        static size_t q6_0_imatrix(const float * s, void * d, int64_t r, int64_t n, const float * im) {
            return ggml_cuda_quantize(0, GGML_TYPE_Q6_0, s, d, r, n, 1, im);
        }
        // --- Removable Q5_0 wrappers (delete to drop Q5_0) ---
        static size_t q5_0(const float * s, void * d, int64_t r, int64_t n) {
            return ggml_cuda_quantize(0, GGML_TYPE_Q5_0, s, d, r, n, 1, nullptr);
        }
        static size_t q5_0_imatrix(const float * s, void * d, int64_t r, int64_t n, const float * im) {
            return ggml_cuda_quantize(0, GGML_TYPE_Q5_0, s, d, r, n, 1, im);
        }
        // --- Removable Q4_0 wrappers (delete to drop Q4_0) ---
        static size_t q4_0(const float * s, void * d, int64_t r, int64_t n) {
            return ggml_cuda_quantize(0, GGML_TYPE_Q4_0, s, d, r, n, 1, nullptr);
        }
        static size_t q4_0_imatrix(const float * s, void * d, int64_t r, int64_t n, const float * im) {
            return ggml_cuda_quantize(0, GGML_TYPE_Q4_0, s, d, r, n, 1, im);
        }
    };

    const quant_spec specs[] = {
        { "q8_0", GGML_TYPE_Q8_0, QK8_0, sizeof(block_q8_0), cuda_wraps::q8_0, ref_quantize_q8_0,
                false, nullptr, nullptr },
        { "q8_0-imatrix", GGML_TYPE_Q8_0, QK8_0, sizeof(block_q8_0), cuda_wraps::q8_0, ref_quantize_q8_0,
                true, cuda_wraps::q8_0_imatrix, ref_quantize_q8_0_imatrix, true },
        { "q6_0", GGML_TYPE_Q6_0, QK6_0, sizeof(block_q6_0), cuda_wraps::q6_0, ref_quantize_q6_0,
                false, nullptr, nullptr },
        { "q6_0-imatrix", GGML_TYPE_Q6_0, QK6_0, sizeof(block_q6_0), cuda_wraps::q6_0, ref_quantize_q6_0,
                true, cuda_wraps::q6_0_imatrix, ref_quantize_q6_0_imatrix, true },
        // --- Removable Q5_0 specs ---
        { "q5_0", GGML_TYPE_Q5_0, QK5_0, sizeof(block_q5_0), cuda_wraps::q5_0, ref_quantize_q5_0,
                false, nullptr, nullptr },
        { "q5_0-imatrix", GGML_TYPE_Q5_0, QK5_0, sizeof(block_q5_0), cuda_wraps::q5_0, ref_quantize_q5_0,
                true, cuda_wraps::q5_0_imatrix, ref_quantize_q5_0_imatrix, true },
        // --- Removable Q4_0 specs ---
        { "q4_0", GGML_TYPE_Q4_0, QK4_0, sizeof(block_q4_0), cuda_wraps::q4_0, ref_quantize_q4_0,
                false, nullptr, nullptr },
        { "q4_0-imatrix", GGML_TYPE_Q4_0, QK4_0, sizeof(block_q4_0), cuda_wraps::q4_0, ref_quantize_q4_0,
                true, cuda_wraps::q4_0_imatrix, ref_quantize_q4_0_imatrix, true },
    };
    const size_t nspec = sizeof(specs)/sizeof(specs[0]);

    static const int64_t ns_npr[]   = { 32, 64, 256, 512, 2048, 4096 };
    static const int64_t ns_nrows[] = { 1, 17, 128, 1000, 16384 };
    const int64_t npr_cnt = g_quick ? 3 : (int64_t)(sizeof(ns_npr)/sizeof(ns_npr[0]));
    const int64_t nrow_cnt = g_quick ? 2 : (int64_t)(sizeof(ns_nrows)/sizeof(ns_nrows[0]));

    // cap nelements so the largest default case stays ~256 MiB of f32
    const int64_t cap = g_quick ? (1<<24) : (1<<26);

    for (size_t s = 0; s < nspec; ++s) {
        const quant_spec & spec = specs[s];

        printf("\n=== type %s ===\n", spec.name);

        printf("\n--- Test: gpu vs cpu vs ref, random + weight-like + edge fills ---\n");
        for (int di = 0; di < ntest_dev; ++di) {
            const int dev = all_devices ? di : devices[0];
            for (int64_t k = 0; k < npr_cnt; ++k) {
                for (int64_t r = 0; r < nrow_cnt; ++r) {
                    const int64_t n_per_row = ns_npr[k];
                    const int64_t nrows     = std::min(ns_nrows[r], cap/n_per_row);
                    test_one("random-uniform", nrows, n_per_row, fill_random_uniform, dev, spec);
                    test_one("weight-like",   nrows, n_per_row, fill_random_weight_like, dev, spec);
                    test_one("edge-cases",    std::min<int64_t>(nrows, 1024), n_per_row, fill_edge_cases, dev, spec);
                    test_one("q5-boundary",  std::min<int64_t>(nrows, 1024), n_per_row, fill_q5_0_boundary, dev, spec);
                    test_one("q6-boundary",  std::min<int64_t>(nrows, 1024), n_per_row, fill_q6_0_boundary, dev, spec);
                }
            }
        }

        printf("\n--- Test: chunk-loop boundary (>1<<20 quant blocks) ---\n");
        test_one("chunk-boundary", 16385, 2048, fill_random_uniform, devices[0], spec);

        printf("\n--- Test: do_quantize ne[2] slice reproduction ---\n");
        test_slices(devices[0], 32,   17,   4, spec);
        test_slices(devices[0], 512,  128,  4, spec);
        test_slices(devices[0], 2048, 128,  4, spec);
        test_slices(devices[0], 2048, 128, 17, spec);

        if (big) {
            printf("\n--- Test: big tensor (token-embd scale) ---\n");
            test_one("big", 65536, 2048, fill_random_uniform, devices[0], spec);
        }
        if (huge) {
            printf("\n--- Test: huge tensor (Llama-3.2-1B token_embd 128256x2048) ---\n");
            test_one("huge-token_embd", 128256, 2048, fill_random_uniform, devices[0], spec);
        }
    }

    printf("\n=== %s ===\n", g_failures == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
