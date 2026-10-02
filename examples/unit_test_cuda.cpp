//
// unit_test_cuda.cpp - byte-exact CUDA vs CPU vs REF quant check; KT order Q8_0,Q6_0,Q5_0,Q4_0,Q5_1,Q4_1,IQ4_NL,IQ4_XS (+imatrix).
// Producers: GPU ggml_cuda_quantize (Joel single entry, nslice=1) vs CPU ggml_quantize_chunk vs local REF copies of ggml-quants.c.
// Layout: 32-val blocks tile rows contiguously; test_slices reproduces do_quantize ne[2] slicing; edge cases per docs/cuda-quantize.md S6.
// Build (GGML_CUDA on): cmake --build build --target unit_test_cuda -j
// Run: unit_test_cuda [--seed N] [--device N] [--all-devices] [--big] [--huge] [--quick]
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
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
static bool g_debug_inputs = false; // --debug-inputs: dump exact inputs of cpu/ref-divergent blocks
static int  g_cuda_device = 0; // --device N, forwarded to ggml_cuda_quantize
static std::mt19937 g_rng(g_seed);

// Per-type plumbing
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
    bool         nan_block_equal; // skip the whole block when both d are non-finite
};

// Joel single-entry wrappers (nslice=1); order Q8_0,Q6_0,Q5_0,Q4_0,Q5_1,Q4_1,IQ4_NL,IQ4_XS.
template<ggml_type T>
static size_t cuda_plain(const float * s, void * d, int64_t r, int64_t n) {
    return ggml_cuda_quantize(g_cuda_device, T, s, d, r, n, 1, nullptr);
}
template<ggml_type T>
static size_t cuda_imatrix(const float * s, void * d, int64_t r, int64_t n, const float * im) {
    return ggml_cuda_quantize(g_cuda_device, T, s, d, r, n, 1, im);
}

// FP_CONTRACT OFF: match CPU (ggml-quants.c); no FMA so OLS ties stay bit-exact.
#pragma STDC FP_CONTRACT OFF

// REF quantize_row_q8_0_ref (ggml-quants.c); matches CPU Q6_0-fudge quirk via __float2half_rn.
static void ref_quantize_q8_0(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0); // match CPU quirk
    const int64_t nb = (nrows*n_per_row)/QK8_0;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = src + ib*QK8_0;
        block_q8_0 *  yb = (block_q8_0 *)dst + ib;

        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            amax = fmaxf(amax, fabsf(xb[j]));
        }

        const float d  = fudge*(amax/127.0f);
        const float id = d ? 1.0f/d : 0.0f;

        yb->d = (ggml_half)__half_as_ushort(__float2half_rn(d));
        for (int j = 0; j < QK8_0; ++j) {
            yb->qs[j] = (int8_t)roundf(xb[j]*id);
        }
    }
}

// Q8_0 ignores imatrix on the CPU (quantize_q8_0): ref routes to plain.
static void ref_quantize_q8_0_imatrix_plain(void * dst, const float * src, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    (void) imatrix;
    ref_quantize_q8_0(dst, src, nrows, n_per_row);
}

// Local copy of quantize_row_q4_0_ref (ggml/src/ggml-quants.c:673).
// HEAD CPU stores FP16(fudge*d); codes use id=1/(fudge*d).
static void ref_quantize_q4_0(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q4_0);
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

        const float d  = fudge*(max/-8.0f);
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

// Q4_0 imatrix: make_qx_quants + quantize_row_q4_0_impl (ggml-quants.c:1786,3429).

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

// REF make_qx_quants (ggml-quants.c:1786) for Q4_0 imatrix (rmse_type==1); byte-exact host port.
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

// Host port of ggml_compute_fp32_to_fp16 (ggml-impl.h:595); matches device NaN encoding (0x7e00).
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

// REF quantize_row_q4_0_impl (ggml-quants.c:3429); stores FP16(fudge*d), imatrix reused per row.
static void ref_quantize_q4_0_imatrix(void * dst, const float * src, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q4_0);
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
            y[ib].d = (ggml_half)fp32_to_fp16_ggml_host(fudge*d);
            for (int j = 0; j < QK4_0/2; ++j) {
                y[ib].qs[j] = (uint8_t)(L[j] | (L[j + QK4_0/2] << 4));
            }
        }
    }
}

// Local copy of quantize_row_q4_1_ref (ggml-quants.c:718). No fudge, MIN(15)
// clamp. Q4_1 ignores symmetric_q4_0 on the CPU.
static void ref_quantize_q4_1(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const int64_t nb = (nrows*n_per_row)/QK4_1;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = src + ib*QK4_1;
        block_q4_1 *  yb = (block_q4_1 *)dst + ib;

        float mn = FLT_MAX;
        float mx = -FLT_MAX;
        for (int j = 0; j < QK4_1; ++j) {
            if (xb[j] < mn) mn = xb[j];
            if (xb[j] > mx) mx = xb[j];
        }

        const float d  = (mx - mn)/15.0f;
        const float id = d ? 1.0f/d : 0.0f;

        yb->d = (ggml_half)__half_as_ushort(__float2half_rn(d));
        yb->m = (ggml_half)__half_as_ushort(__float2half_rn(mn));

        for (int j = 0; j < QK4_1/2; ++j) {
            const uint8_t xi0 = (uint8_t)std::min(15, (int)(int8_t)((xb[j] - mn)*id + 0.5f));
            const uint8_t xi1 = (uint8_t)std::min(15, (int)(int8_t)((xb[j + QK4_1/2] - mn)*id + 0.5f));
            yb->qs[j] = (uint8_t)(xi0 | (xi1 << 4));
        }
    }
}

// REF quantize_row_q5_0_ref (ggml-quants.c:757); 5th bit packs into LE qh bitmap.
static void ref_quantize_q5_0(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q5_0);
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

        const float d  = fudge*(max/-16.0f);
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

// Local copy of quantize_row_q6_0_impl without weights (ggml-quants.c):
// OLS is KEPT. weight = x*x, d = make_qx_quants * fudge.
static void ref_quantize_q6_0(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0);
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * x = src + irow*n_per_row;
        block_q6_0 * y = (block_q6_0 *)dst + irow*(n_per_row/QK6_0);

        float weight[QK6_0];
        int8_t L[QK6_0];
        const int64_t nb = n_per_row/QK6_0;
        for (int64_t ib = 0; ib < nb; ++ib) {
            const float * xb = x + QK6_0*ib;
            for (int j = 0; j < QK6_0; ++j) weight[j] = xb[j]*xb[j];
            float d = ref_make_qx_quants(QK6_0, 32, xb, L, 1, weight);
            y[ib].d = (ggml_half)fp32_to_fp16_ggml_host(fudge*d);

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
            y[ib].d = (ggml_half)fp32_to_fp16_ggml_host(ggml_get_quantize_fudge_factor(GGML_TYPE_Q5_0)*d);

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

// Local copy of quantize_row_q5_1_ref (ggml-quants.c:809). No fudge, no clamp
// (bug-compatible). min/max values are order-independent.
static void ref_quantize_q5_1(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const int64_t nb = (nrows*n_per_row)/QK5_1;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = src + ib*QK5_1;
        block_q5_1 *  yb = (block_q5_1 *)dst + ib;

        float mn = FLT_MAX;
        float mx = -FLT_MAX;
        for (int j = 0; j < QK5_1; ++j) {
            if (xb[j] < mn) mn = xb[j];
            if (xb[j] > mx) mx = xb[j];
        }

        const float d  = (mx - mn)/31.0f;
        const float id = d ? 1.0f/d : 0.0f;

        yb->d = (ggml_half)__half_as_ushort(__float2half_rn(d));
        yb->m = (ggml_half)__half_as_ushort(__float2half_rn(mn));

        uint32_t qh = 0;
        for (int j = 0; j < QK5_1/2; ++j) {
            const uint8_t xi0 = (uint8_t)((xb[j] - mn)*id + 0.5f);
            const uint8_t xi1 = (uint8_t)((xb[j + QK5_1/2] - mn)*id + 0.5f);
            yb->qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
            qh |= ((uint32_t)((xi0 & 0x10u) >> 4)) << (j + 0);
            qh |= ((uint32_t)((xi1 & 0x10u) >> 4)) << (j + QK5_1/2);
        }
        memcpy(yb->qh, &qh, sizeof(qh));
    }
}

// REF make_qkx3_quants (ggml-quants.c:2211) for Q4_1/Q5_1 imatrix; verbatim order, double accumulators.
// optimize off (clang only): force strict source order for tie-sensitive reductions (test-only).
#ifdef __clang__
#pragma clang optimize off
#endif
static inline int ref_kx3_nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float ref_make_qkx3_quants(int n, int nmax, const float * x, const float * weights,
        uint8_t * L, float * the_min, uint8_t * Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    double sum_w = weights ? (double)weights[0] : (double)(x[0]*x[0]);
    double sum_x = sum_w * (double)x[0];
    double sum_x2 = sum_w * (double)x[0] * (double)x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i]*x[i];
        sum_w += (double)w;
        sum_x += (double)w * (double)x[i];
        sum_x2 += (double)w * (double)x[i] * (double)x[i];
    }
    if (min > 0) {
        min = 0;
    }
    if (max - min < 1e-10f) {
        memset(L, 0, n);
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    double best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = ref_kx3_nearest_int(iscale*(x[i] - min));
        l = l > nmax ? nmax : (l < 0 ? 0 : l);
        L[i] = (uint8_t)l;
        double diff = (double)scale * L[i] + (double)min - (double)x[i];
        diff = use_mad ? fabs(diff) : diff*diff;
        double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        double sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = ref_kx3_nearest_int(iscale*(x[i] - min));
            l = l > nmax ? nmax : (l < 0 ? 0 : l);
            Laux[i] = (uint8_t)l;
            float w = weights ? weights[i] : x[i]*x[i];
            sum_l  += (double)w*l;
            sum_l2 += (double)w*l*l;
            sum_xl += (double)w*l*(double)x[i];
        }
        double D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            double this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            double this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            double mad = 0;
            if (use_mad) {
                for (int i = 0; i < n; ++i) {
                    double diff = (double)this_scale * Laux[i] + (double)this_min - (double)x[i];
                    diff = fabs(diff);
                    double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
                    mad += w * diff;
                }
            } else {
                mad = sum_x2 - 2*this_scale*sum_xl - 2*this_min*sum_x + 2*this_scale*this_min*sum_l
                    + this_scale*this_scale*sum_l2 + this_min*this_min*sum_w;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = (float)this_scale;
                min = (float)this_min;
            }
        }
    }
    if (use_mad) {
        *the_min = -min;
        return scale;
    }

    double sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
        int l = L[i];
        double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
        sum_l  += w*l;
        sum_l2 += w*l*l;
        sum_xl += w*l*(double)x[i];
    }
    double best = 2*(double)scale*sum_xl + 2*(double)min*sum_x - 2*(double)scale*(double)min*sum_l
                - (double)scale*(double)scale*sum_l2 - (double)min*(double)min*sum_w;
    int last_j = -1, last_dir = 0;
    for (int itry = 0; itry < nmax*n; ++itry) {
        float gmax = 0;
        int best_j = -1, dir = 0;
        for (int j = 0; j < n; ++j) {
            float g = x[j] - scale*L[j] - min;
            if (g > 0 && L[j] < nmax && g > gmax) {
                gmax = g; best_j = j; dir = 1;
            }
            else if (g < 0 && L[j] > 0 && -g > gmax) {
                gmax = -g; best_j = j; dir = -1;
            }
        }
        if (best_j < 0 || (best_j == last_j && dir == -last_dir)) break;
        double w = weights ? (double)weights[best_j] : (double)(x[best_j]*x[best_j]);
        sum_l  += w*dir;
        sum_l2 += w*(2*L[best_j]*dir + 1);
        sum_xl += w*(double)x[best_j]*dir;
        double D = (double)sum_w * sum_l2 - sum_l * sum_l;
        if (D <= 0) break;
        double this_scale = ((double)sum_w * sum_xl - (double)sum_x * sum_l)/D;
        double this_min   = (sum_l2 * (double)sum_x - sum_l * sum_xl)/D;
        if (this_min > 0) {
            this_min = 0;
            this_scale = sum_xl / sum_l2;
        }
        if (this_scale < 0) break;
        double score = 2*this_scale*sum_xl + 2*this_min*(double)sum_x - 2*this_scale*this_min*sum_l
                     - this_scale*this_scale*sum_l2 - this_min*this_min*(double)sum_w;
        if (score <= best) break;
        best = score;
        scale = (float)this_scale;
        min = (float)this_min;
        L[best_j] += dir;
        last_j = best_j; last_dir = dir;
    }
    *the_min = -min;
    return scale;
}

// REF quantize_row_q5_1_impl: make_qkx3 (nmax=31), no fudge.
static void ref_quantize_q5_1_imatrix(void * dst, const float * src, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * x = src + irow*n_per_row;
        block_q5_1 * y = (block_q5_1 *)dst + irow*(n_per_row/QK5_1);

        float sum_x2 = 0;
        for (int64_t j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
        float sigma2 = sum_x2/n_per_row;

        float weight[QK5_1];
        uint8_t L[QK5_1], Laux[QK5_1];
        const int64_t nb = n_per_row/QK5_1;
        for (int64_t ib = 0; ib < nb; ++ib) {
            const float * xb = x + QK5_1*ib;
            const float * qw = imatrix + QK5_1*ib;
            for (int j = 0; j < QK5_1; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            float the_min;
            float d = ref_make_qkx3_quants(QK5_1, 31, xb, weight, L, &the_min, Laux, -0.9f, 0.05f, 36, false);
            // Bit-twiddle FP16 so NaN payload matches ggml_compute_fp32_to_fp16.
            y[ib].d = (ggml_half)fp32_to_fp16_ggml_host(d);
            y[ib].m = (ggml_half)fp32_to_fp16_ggml_host(-the_min);

            uint32_t qh = 0;
            for (int j = 0; j < QK5_1/2; ++j) {
                const uint8_t xi0 = L[j];
                const uint8_t xi1 = L[j + QK5_1/2];
                y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
                qh |= ((uint32_t)((xi0 & 0x10u) >> 4)) << (j + 0);
                qh |= ((uint32_t)((xi1 & 0x10u) >> 4)) << (j + QK5_1/2);
            }
            memcpy(y[ib].qh, &qh, sizeof(qh));
        }
    }
}

// Local copy of quantize_row_q4_1_impl: make_qkx3 (nmax=15), no fudge.
static void ref_quantize_q4_1_imatrix(void * dst, const float * src, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * x = src + irow*n_per_row;
        block_q4_1 * y = (block_q4_1 *)dst + irow*(n_per_row/QK4_1);

        float sum_x2 = 0;
        for (int64_t j = 0; j < n_per_row; ++j) sum_x2 += x[j]*x[j];
        float sigma2 = sum_x2/n_per_row;

        float weight[QK4_1];
        uint8_t L[QK4_1], Laux[QK4_1];
        const int64_t nb = n_per_row/QK4_1;
        for (int64_t ib = 0; ib < nb; ++ib) {
            const float * xb = x + QK4_1*ib;
            const float * qw = imatrix + QK4_1*ib;
            for (int j = 0; j < QK4_1; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            float the_min;
            float d = ref_make_qkx3_quants(QK4_1, 15, xb, weight, L, &the_min, Laux, -0.9f, 0.05f, 36, false);
            // bit-twiddle FP16 (see Q5_1 imatrix ref): NaN payload must match.
            y[ib].d = (ggml_half)fp32_to_fp16_ggml_host(d);
            y[ib].m = (ggml_half)fp32_to_fp16_ggml_host(-the_min);
            for (int j = 0; j < QK4_1/2; ++j) {
                y[ib].qs[j] = (uint8_t)(L[j] | (L[j + QK4_1/2] << 4));
            }
        }
    }
}
#pragma STDC FP_CONTRACT ON
#ifdef __clang__
#pragma clang optimize on
#endif

// Local copy of quantize_iq4_nl plain path (qw == NULL, ntry = 7): codebook
// grid search + hill-climb with w = x*x, dh = FP16(scale).
static const int8_t ref_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static int ref_best_index_iq4nl(const int8_t * values, float x);

static const int ref_iq4nl_index[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};

static int ref_best_index_iq4nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    // entries >= 16 name the boundary pair (ix-16, ix-15); ties go high
    ix = ref_iq4nl_index[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}

// IQ4_NL/XS 32-val block optimizer (ntry=7, verbatim CPU order); weight from caller, returns LS scale.
static float ref_iq4nl_opt_block(const float * xb, float * weight, uint8_t * L) {
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < 32; ++j) {
        float ax = fabsf(xb[j]);
        if (ax > amax) { amax = ax; max = xb[j]; }
    }
    if (amax < 1e-15f) {
        return 0.0f;
    }
    float d = -max/ref_kvalues_iq4nl[0];
    float id = 1/d;
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int j = 0; j < 32; ++j) {
        int l = ref_best_index_iq4nl(ref_kvalues_iq4nl, id*xb[j]);
        L[j] = (uint8_t)l;
        float q = (float)ref_kvalues_iq4nl[l];
        sumqx += weight[j]*q*xb[j];
        sumq2 += weight[j]*q*q;
    }
    d = sumqx/sumq2;
    float best = d*sumqx;
    float best_sumqx = sumqx, best_sumq2 = sumq2;
    for (int itry = -7; itry <= 7; ++itry) {
        id = (itry + ref_kvalues_iq4nl[0])/max;
        sumqx = sumq2 = 0.0f;
        for (int j = 0; j < 32; ++j) {
            int l = ref_best_index_iq4nl(ref_kvalues_iq4nl, id*xb[j]);
            float q = (float)ref_kvalues_iq4nl[l];
            sumqx += weight[j]*q*xb[j];
            sumq2 += weight[j]*q*q;
        }
        if (sumq2 > 0.0f && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
            best_sumqx = sumqx; best_sumq2 = sumq2;
            for (int j = 0; j < 32; ++j) {
                L[j] = (uint8_t)ref_best_index_iq4nl(ref_kvalues_iq4nl, id*xb[j]);
            }
        }
        id = (itry + ref_kvalues_iq4nl[15])/max;
        sumqx = sumq2 = 0.0f;
        for (int j = 0; j < 32; ++j) {
            int l = ref_best_index_iq4nl(ref_kvalues_iq4nl, id*xb[j]);
            float q = (float)ref_kvalues_iq4nl[l];
            sumqx += weight[j]*q*xb[j];
            sumq2 += weight[j]*q*q;
        }
        if (sumq2 > 0.0f && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
            best_sumqx = sumqx; best_sumq2 = sumq2;
            for (int j = 0; j < 32; ++j) {
                L[j] = (uint8_t)ref_best_index_iq4nl(ref_kvalues_iq4nl, id*xb[j]);
            }
        }
    }
    sumqx = best_sumqx; sumq2 = best_sumq2;
    for (int iter = 0; iter < 32*32; ++iter) {
        float min_step = INFINITY;
        int best_j = -1, dir = 0;
        for (int j = 0; j < 32; ++j) {
            float g = d*weight[j]*(xb[j] - d*ref_kvalues_iq4nl[L[j]]);
            if (g > 0.0f && L[j] < 15) {
                float step = (ref_kvalues_iq4nl[L[j]+1] - ref_kvalues_iq4nl[L[j]])/g;
                if (step < min_step) { min_step = step; best_j = j; dir = 1; }
            }
            else if (g < 0.0f && L[j] > 0) {
                float step = (ref_kvalues_iq4nl[L[j]-1] - ref_kvalues_iq4nl[L[j]])/g;
                if (step < min_step) { min_step = step; best_j = j; dir = -1; }
            }
        }
        if (best_j < 0) break;
        float new_sumqx = sumqx + weight[best_j]*xb[best_j]*(ref_kvalues_iq4nl[L[best_j]+dir] - ref_kvalues_iq4nl[L[best_j]]);
        float new_sumq2 = sumq2 + weight[best_j]*(ref_kvalues_iq4nl[L[best_j]+dir]*ref_kvalues_iq4nl[L[best_j]+dir] - ref_kvalues_iq4nl[L[best_j]]*ref_kvalues_iq4nl[L[best_j]]);
        if (new_sumq2 > 0.0f && new_sumqx*new_sumqx > best*new_sumq2) {
            sumqx = new_sumqx; sumq2 = new_sumq2;
            d = sumqx/sumq2; best = d*sumqx;
            L[best_j] += dir;
        } else break;
    }
    return d;
}

static void ref_quantize_iq4_nl(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const int64_t nb = (nrows*n_per_row)/QK4_NL;
    for (int64_t ib = 0; ib < nb; ++ib) {
        const float * xb = src + ib*QK4_NL;
        block_iq4_nl * yb = (block_iq4_nl *)dst + ib;

        float weight[QK4_NL];
        uint8_t L[QK4_NL];
        for (int j = 0; j < QK4_NL; ++j) weight[j] = xb[j]*xb[j];

        float scale = ref_iq4nl_opt_block(xb, weight, L);

        // bit-twiddle FP16 (see Q5_1 imatrix ref): NaN payload must match.
        yb->d = (ggml_half)fp32_to_fp16_ggml_host(scale);
        {
            float idf = scale ? 1/scale : 0.0f;
            for (int j = 0; j < QK4_NL; ++j) {
                L[j] = (uint8_t)ref_best_index_iq4nl(ref_kvalues_iq4nl, idf*xb[j]);
            }
        }
        for (int j = 0; j < QK4_NL/2; ++j) {
            yb->qs[j] = (uint8_t)(L[j] | (L[j + QK4_NL/2] << 4));
        }
    }
}

// REF quantize_iq4_nl with imatrix: same optimizer, weight=qw*sqrt(sigma2+x*x) per 32-block.
static void ref_quantize_iq4_nl_imatrix(void * dst, const float * src, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    for (int64_t irow = 0; irow < nrows; ++irow) {
        const float * x = src + irow*n_per_row;
        block_iq4_nl * y = (block_iq4_nl *)dst + irow*(n_per_row/QK4_NL);

        float weight[QK4_NL];
        uint8_t L[QK4_NL];
        const int64_t nb = n_per_row/QK4_NL;
        for (int64_t ib = 0; ib < nb; ++ib) {
            const float * xb = x + QK4_NL*ib;
            float sum = 0.0f;
            for (int j = 0; j < QK4_NL; ++j) sum += xb[j]*xb[j];
            const float sigma2 = sum*2.0f/QK4_NL;
            const float * qw = imatrix + QK4_NL*ib;
            for (int j = 0; j < QK4_NL; ++j) weight[j] = qw[j]*sqrtf(sigma2 + xb[j]*xb[j]);

            float scale = ref_iq4nl_opt_block(xb, weight, L);

            // bit-twiddle FP16 (see Q5_1 imatrix ref): NaN payload must match.
            y[ib].d = (ggml_half)fp32_to_fp16_ggml_host(scale);
            {
                float idf = scale ? 1/scale : 0.0f;
                for (int j = 0; j < QK4_NL; ++j) {
                    L[j] = (uint8_t)ref_best_index_iq4nl(ref_kvalues_iq4nl, idf*xb[j]);
                }
            }
            for (int j = 0; j < QK4_NL/2; ++j) {
                y[ib].qs[j] = (uint8_t)(L[j] | (L[j + QK4_NL/2] << 4));
            }
        }
    }
}

// IQ4_XS superblock 32-val optimizer (ntry=7); weight from caller, *eps when amax<1e-15.
static float ref_iq4_block_opt(const float * xb, float * weight, uint8_t * Lb, bool * eps) {
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < 32; ++j) {
        float ax = fabsf(xb[j]);
        if (ax > amax) { amax = ax; max = xb[j]; }
    }
    if (amax < 1e-15f) { *eps = true; return 0.0f; } // Lb left for re-quant
    *eps = false;
    float d = -max/ref_kvalues_iq4nl[0];
    float id = 1/d;
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int j = 0; j < 32; ++j) {
        int l = ref_best_index_iq4nl(ref_kvalues_iq4nl, id*xb[j]);
        Lb[j] = (uint8_t)l;
        float q = (float)ref_kvalues_iq4nl[l];
        sumqx += weight[j]*q*xb[j];
        sumq2 += weight[j]*q*q;
    }
    d = sumqx/sumq2;
    float best = d*sumqx;
    float best_sumqx = sumqx, best_sumq2 = sumq2;
    for (int itry = -7; itry <= 7; ++itry) {
        id = (itry + ref_kvalues_iq4nl[0])/max;
        sumqx = sumq2 = 0.0f;
        for (int j = 0; j < 32; ++j) {
            int l = ref_best_index_iq4nl(ref_kvalues_iq4nl, id*xb[j]);
            float q = (float)ref_kvalues_iq4nl[l];
            sumqx += weight[j]*q*xb[j];
            sumq2 += weight[j]*q*q;
        }
        if (sumq2 > 0.0f && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
            best_sumqx = sumqx; best_sumq2 = sumq2;
            for (int j = 0; j < 32; ++j) {
                Lb[j] = (uint8_t)ref_best_index_iq4nl(ref_kvalues_iq4nl, id*xb[j]);
            }
        }
        id = (itry + ref_kvalues_iq4nl[15])/max;
        sumqx = sumq2 = 0.0f;
        for (int j = 0; j < 32; ++j) {
            int l = ref_best_index_iq4nl(ref_kvalues_iq4nl, id*xb[j]);
            float q = (float)ref_kvalues_iq4nl[l];
            sumqx += weight[j]*q*xb[j];
            sumq2 += weight[j]*q*q;
        }
        if (sumq2 > 0.0f && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
            best_sumqx = sumqx; best_sumq2 = sumq2;
            for (int j = 0; j < 32; ++j) {
                Lb[j] = (uint8_t)ref_best_index_iq4nl(ref_kvalues_iq4nl, id*xb[j]);
            }
        }
    }
    sumqx = best_sumqx; sumq2 = best_sumq2;
    for (int iter = 0; iter < 32*32; ++iter) {
        float min_step = INFINITY;
        int best_j = -1, dir = 0;
        for (int j = 0; j < 32; ++j) {
            float g = d*weight[j]*(xb[j] - d*ref_kvalues_iq4nl[Lb[j]]);
            if (g > 0.0f && Lb[j] < 15) {
                float step = (ref_kvalues_iq4nl[Lb[j]+1] - ref_kvalues_iq4nl[Lb[j]])/g;
                if (step < min_step) { min_step = step; best_j = j; dir = 1; }
            }
            else if (g < 0.0f && Lb[j] > 0) {
                float step = (ref_kvalues_iq4nl[Lb[j]-1] - ref_kvalues_iq4nl[Lb[j]])/g;
                if (step < min_step) { min_step = step; best_j = j; dir = -1; }
            }
        }
        if (best_j < 0) break;
        float new_sumqx = sumqx + weight[best_j]*xb[best_j]*(ref_kvalues_iq4nl[Lb[best_j]+dir] - ref_kvalues_iq4nl[Lb[best_j]]);
        float new_sumq2 = sumq2 + weight[best_j]*(ref_kvalues_iq4nl[Lb[best_j]+dir]*ref_kvalues_iq4nl[Lb[best_j]+dir] - ref_kvalues_iq4nl[Lb[best_j]]*ref_kvalues_iq4nl[Lb[best_j]]);
        if (new_sumq2 > 0.0f && new_sumqx*new_sumqx > best*new_sumq2) {
            sumqx = new_sumqx; sumq2 = new_sumq2;
            d = sumqx/sumq2; best = d*sumqx;
            Lb[best_j] += dir;
        } else break;
    }
    return d;
}

static int ref_iq4xs_nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// Local copy of quantize_iq4_xs plain path: 8 block optimizers + global scale
// fit + re-quant, w = x*x throughout.
// One IQ4_XS superblock shared by plain/imatrix refs; qs == nullptr → w = x*x,
// else w = qw*sqrt(sigma2 + x*x) with superblock sigma2 (CPU order).
static void ref_iq4xs_superblock(const float * xs, const float * qs, block_iq4_xs * yb) {
    float sigma2 = 0.0f;
    if (qs != nullptr) {
        float sum = 0.0f;
        for (int j = 0; j < QK_K; ++j) sum += xs[j]*xs[j];
        sigma2 = sum*2.0f/QK_K;
    }
    float weight[32];
    uint8_t L[QK_K];
    float scales[8];
    float max_scale = 0.0f, amax_scale = 0.0f;
    for (int ib = 0; ib < 8; ++ib) {
        const float * xb = xs + ib*32;
        if (qs == nullptr) {
            for (int j = 0; j < 32; ++j) weight[j] = xb[j]*xb[j];
        } else {
            const float * qb = qs + ib*32;
            for (int j = 0; j < 32; ++j) weight[j] = qb[j]*sqrtf(sigma2 + xb[j]*xb[j]);
        }
        bool eps = false;
        float d = ref_iq4_block_opt(xb, weight, L + ib*32, &eps);
        if (eps) { scales[ib] = 0.0f; continue; } // matches CPU exactly
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) { amax_scale = abs_d; max_scale = d; }
    }

    float gd = -max_scale/32.0f;
    // bit-twiddle FP16 (see Q5_1 imatrix ref): NaN payload must match.
    yb->d = (ggml_half)fp32_to_fp16_ggml_host(gd);
    float gid = gd ? 1/gd : 0.0f;
    uint16_t scales_h = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int l = isfinite(scales[ib]) ? ref_iq4xs_nearest_int(gid*scales[ib]) : 0; // deterministic degenerate path
        l = l > 31 ? 31 : (l < -32 ? -32 : l);
        float dl = gd*l;
        float idl = dl ? 1/dl : 0.0f;
        uint8_t * Lb = L + ib*32;
        const float * xb = xs + ib*32;
        for (int j = 0; j < 32; ++j) {
            Lb[j] = (uint8_t)ref_best_index_iq4nl(ref_kvalues_iq4nl, idl*xb[j]);
        }
        l += 32;
        uint8_t l_l = (uint8_t)(l & 0xf);
        uint8_t l_h = (uint8_t)((unsigned)l >> 4);
        if (ib % 2 == 0) yb->scales_l[ib/2] = l_l;
        else yb->scales_l[ib/2] |= (uint8_t)(l_l << 4);
        scales_h |= (uint16_t)(l_h << (2*(ib % 8)));
    }
    yb->scales_h = scales_h;
    for (int i = 0; i < QK_K/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            yb->qs[16*i + j] = (uint8_t)(L[32*i + j] | (L[32*i + 16 + j] << 4));
        }
    }
}

static void ref_quantize_iq4_xs(void * dst, const float * src, int64_t nrows, int64_t n_per_row) {
    const int64_t nb = (nrows*n_per_row)/QK_K;
    for (int64_t sb = 0; sb < nb; ++sb) {
        ref_iq4xs_superblock(src + sb*QK_K, nullptr, (block_iq4_xs *)dst + sb);
    }
}

// REF quantize_iq4_xs with imatrix: same replay, weight=qw*sqrt(sigma2+x*x) per superblock.
static void ref_quantize_iq4_xs_imatrix(void * dst, const float * src, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    const int64_t sb_per_row = n_per_row/QK_K;
    const int64_t nb = nrows*sb_per_row;
    for (int64_t sb = 0; sb < nb; ++sb) {
        ref_iq4xs_superblock(src + sb*QK_K, imatrix + (sb % sb_per_row)*QK_K, (block_iq4_xs *)dst + sb);
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
            y[ib].d = (ggml_half)fp32_to_fp16_ggml_host(ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0)*d);

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

// Random-uniform fills.
static void fill_random_uniform(float * dst, int64_t n) {
    std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
    for (int64_t i = 0; i < n; ++i) dst[i] = dist(g_rng);
}

// Weight-like: 90% tight N(0,0.05) + 10% wide N(0,1) blocks.
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

// Q5_0 boundary blocks: 5th-bit/qh bitmap and truncation ties.
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

// Q6_0 boundary blocks: 6-bit levels/2-bit qh and truncation ties.
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

// Synthetic imatrix: ~5 decades, every 11th column zero (w==0 path).
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

// fp16 NaN: exponent all-ones + nonzero mantissa.
static bool is_fp16_nan(uint16_t v) {
    return ((v >> 10) & 0x1f) == 0x1f && (v & 0x03ff) != 0;
}
// fp16 nonfinite (Inf/NaN): payload from UB casts differs by platform.
static bool is_fp16_nonfinite(uint16_t v) {
    return ((v >> 10) & 0x1f) == 0x1f;
}
// Compare buffers; report first differing block + diff count.
// Nonfinite-d skips whole block, NaN-d treats any NaN scale as equal (UB payload differs x86 vs CUDA).
static int64_t compare_buffers(const char * tag, const uint8_t * a, const uint8_t * b, size_t n, size_t blk_size,
        bool nan_d_equal = false, bool nan_block_equal = false) {
    int64_t ndiff = 0;
    int64_t first_blk = -1;
    for (size_t blk = 0; blk < n; blk += blk_size) {
        // Skip degenerate overflow blocks: UB payload differs x86 vs CUDA.
        if (nan_block_equal && blk + 1 < n) {
            const uint16_t da = (uint16_t)a[blk] | (uint16_t)(a[blk+1] << 8);
            const uint16_t db = (uint16_t)b[blk] | (uint16_t)(b[blk+1] << 8);
            if (is_fp16_nonfinite(da) && is_fp16_nonfinite(db)) {
                continue;
            }
        }
        for (size_t i = blk; i < blk + blk_size && i < n; ++i) {
        // NaN scale: sign/payload differs x86 SSE vs NVIDIA, treat any NaN d as equal.
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
    }
    if (first_blk >= 0) {
        const uint8_t * ra = a + first_blk*blk_size;
        const uint8_t * rb = b + first_blk*blk_size;
        printf("  [FAIL] %s: %lld/%zu bytes differ; first differing quant block %lld\n",
               tag, (long long)ndiff, n, (long long)first_blk);
        dump_block("a", ra, blk_size);
        dump_block("b", rb, blk_size);
    }
    return ndiff;
}

// Compare GPU vs CPU vs REF producers.
static void test_one(const char * tag, int64_t nrows, int64_t n_per_row,
        void (*fill)(float *, int64_t), int device, const quant_spec & spec) {
    if (n_per_row % spec.qk != 0 || nrows <= 0) return;

    const int64_t nelements = nrows*n_per_row;
    std::vector<float> src(nelements);
    fill(src.data(), nelements);

    // Synthetic imatrix: one weight per column, reused per row (CPU contract).
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

    char tag_gc[128], tag_cr[128];
    snprintf(tag_gc, sizeof(tag_gc), "%s gpu/cpu", tag);
    snprintf(tag_cr, sizeof(tag_cr), "%s cpu/ref", tag);
    const int64_t d_gpu_cpu = compare_buffers(tag_gc, out_gpu.data(), out_cpu.data(), out_size, spec.blk_size, spec.nan_d_equal, spec.nan_block_equal);
    const int64_t d_cpu_ref = compare_buffers(tag_cr, out_cpu.data(), out_ref.data(), out_size, spec.blk_size, spec.nan_d_equal, spec.nan_block_equal);

    if (d_gpu_cpu == 0 && d_cpu_ref == 0) {
        printf("  [OK]   %s nrows=%-6lld n_per_row=%-5lld : gpu==cpu==ref\n",
               tag, (long long)nrows, (long long)n_per_row);
    } else {
        ++g_failures;
        // Dump first cpu/ref-divergent block inputs (%a) to root-cause codegen ghosts.
        if (g_debug_inputs && d_cpu_ref != 0) {
            const size_t blk_size = spec.blk_size;
            size_t first = 0;
            for (; first < out_size; ++first) {
                if (out_cpu[first] != out_ref[first]) break;
            }
            const size_t fblk = first/blk_size; // d_cpu_ref != 0 guarantees first < out_size
            const int64_t vals_per_blk = spec.qk == QK_K ? QK_K : 32;
            printf("  [DEBUG] %s/%s nrows=%lld npr=%lld blk=%zu vals:",
                    spec.name, tag, (long long)nrows, (long long)n_per_row, fblk);
            for (int64_t j = 0; j < vals_per_blk; ++j) {
                printf(" %a", (double)src[fblk*vals_per_blk + j]);
            }
            printf("\n  [DEBUG] imatrix:");
            if (imatrix) {
                for (int64_t j = 0; j < vals_per_blk; ++j) {
                    printf(" %a", (double)imatrix[(fblk*vals_per_blk + j) % n_per_row]);
                }
            } else {
                printf(" (none)");
            }
            printf("\n");
        }
    }
}

// Reproduces do_quantize ne[2] slicing: per-slice CPU/GPU into consecutive slots.
// Also checks whole-tensor call matches (quantizer is contiguous).
static void test_slices(int device, int64_t ne0, int64_t ne1, int64_t ne2, const quant_spec & spec) {
    if (ne0 % spec.qk != 0) return; // e.g. IQ4_XS needs ne0 % 256 == 0
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

// Device diagnostics

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

// main

#pragma STDC FP_CONTRACT ON // local refs above replay CPU bit-for-bit; re-enable contraction for the harness below

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
        else if (arg == "--debug-inputs") g_debug_inputs = true;
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
    g_cuda_device = devices[0];

    printf("  [INFO] ggml_cuda_quantize runs on device %d\n", g_cuda_device);

    const quant_spec specs[] = {
        { "q8_0", GGML_TYPE_Q8_0, QK8_0, sizeof(block_q8_0), cuda_plain<GGML_TYPE_Q8_0>, ref_quantize_q8_0,
                false, nullptr, nullptr },
        { "q8_0-imatrix", GGML_TYPE_Q8_0, QK8_0, sizeof(block_q8_0), cuda_plain<GGML_TYPE_Q8_0>, ref_quantize_q8_0,
                true, cuda_imatrix<GGML_TYPE_Q8_0>, ref_quantize_q8_0_imatrix_plain, false },
        { "q6_0", GGML_TYPE_Q6_0, QK6_0, sizeof(block_q6_0), cuda_plain<GGML_TYPE_Q6_0>, ref_quantize_q6_0,
                false, nullptr, nullptr, true }, // OLS can yield NaN scale on degenerate blocks (payload is vendor-defined)
        { "q6_0-imatrix", GGML_TYPE_Q6_0, QK6_0, sizeof(block_q6_0), cuda_plain<GGML_TYPE_Q6_0>, ref_quantize_q6_0,
                true, cuda_imatrix<GGML_TYPE_Q6_0>, ref_quantize_q6_0_imatrix, true },
        // --- Removable Q5_0 specs ---
        { "q5_0", GGML_TYPE_Q5_0, QK5_0, sizeof(block_q5_0), cuda_plain<GGML_TYPE_Q5_0>, ref_quantize_q5_0,
                false, nullptr, nullptr },
        { "q5_0-imatrix", GGML_TYPE_Q5_0, QK5_0, sizeof(block_q5_0), cuda_plain<GGML_TYPE_Q5_0>, ref_quantize_q5_0,
                true, cuda_imatrix<GGML_TYPE_Q5_0>, ref_quantize_q5_0_imatrix, true },
        // --- Removable Q4_0 specs ---
        { "q4_0", GGML_TYPE_Q4_0, QK4_0, sizeof(block_q4_0), cuda_plain<GGML_TYPE_Q4_0>, ref_quantize_q4_0,
                false, nullptr, nullptr },
        { "q4_0-imatrix", GGML_TYPE_Q4_0, QK4_0, sizeof(block_q4_0), cuda_plain<GGML_TYPE_Q4_0>, ref_quantize_q4_0,
                true, cuda_imatrix<GGML_TYPE_Q4_0>, ref_quantize_q4_0_imatrix, true },
        { "q5_1", GGML_TYPE_Q5_1, QK5_1, sizeof(block_q5_1), cuda_plain<GGML_TYPE_Q5_1>, ref_quantize_q5_1,
                false, nullptr, nullptr },
        { "q5_1-imatrix", GGML_TYPE_Q5_1, QK5_1, sizeof(block_q5_1), cuda_plain<GGML_TYPE_Q5_1>, ref_quantize_q5_1,
                true, cuda_imatrix<GGML_TYPE_Q5_1>, ref_quantize_q5_1_imatrix, true, true },
        { "q4_1", GGML_TYPE_Q4_1, QK4_1, sizeof(block_q4_1), cuda_plain<GGML_TYPE_Q4_1>, ref_quantize_q4_1,
                false, nullptr, nullptr },
        { "q4_1-imatrix", GGML_TYPE_Q4_1, QK4_1, sizeof(block_q4_1), cuda_plain<GGML_TYPE_Q4_1>, ref_quantize_q4_1,
                true, cuda_imatrix<GGML_TYPE_Q4_1>, ref_quantize_q4_1_imatrix, true, true },
        // IQ4_NL (+imatrix): optimizer replay with x*x / qw*sqrt weights
        { "iq4_nl", GGML_TYPE_IQ4_NL, QK4_NL, sizeof(block_iq4_nl), cuda_plain<GGML_TYPE_IQ4_NL>, ref_quantize_iq4_nl,
                false, nullptr, nullptr, false, true },
        { "iq4_nl-imatrix", GGML_TYPE_IQ4_NL, QK4_NL, sizeof(block_iq4_nl), cuda_plain<GGML_TYPE_IQ4_NL>, ref_quantize_iq4_nl,
                true, cuda_imatrix<GGML_TYPE_IQ4_NL>, ref_quantize_iq4_nl_imatrix, false, true },
        // IQ4_XS (+imatrix): superblock replay with x*x / qw*sqrt weights
        { "iq4_xs", GGML_TYPE_IQ4_XS, QK_K, sizeof(block_iq4_xs), cuda_plain<GGML_TYPE_IQ4_XS>, ref_quantize_iq4_xs,
                false, nullptr, nullptr, false, true },
        { "iq4_xs-imatrix", GGML_TYPE_IQ4_XS, QK_K, sizeof(block_iq4_xs), cuda_plain<GGML_TYPE_IQ4_XS>, ref_quantize_iq4_xs,
                true, cuda_imatrix<GGML_TYPE_IQ4_XS>, ref_quantize_iq4_xs_imatrix, false, true },
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
