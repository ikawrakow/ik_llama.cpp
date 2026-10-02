//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//
// Bit-exact CUDA GGUF quants (RN intrinsics, no FMA; host sigma2 in CPU order), via Joel's ggml_cuda_quantize() (kt-encoder.cu).
// Order: Q8_0, Q6_0, Q5_0, Q4_0. Q4_0/Q5_0 removable (delete kernel+helpers+cases).
// Q8_0 uses Q6_0 fudge quirk (ggml-quants.c:915), ignores imatrix; Q6_0 OLS kept (make_qx+fudge).

#include "quantize_gguf.cuh"

#include <cinttypes>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

// --- kernels ---

static __global__ void quantize_q8_0_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK8_0

    // grid-stride over quant blocks (nblocks can exceed UINT_MAX on big models)
    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK8_0 + lane];

        // exact, order-independent max reduction (max is associative/exact)
        float amax = fabsf(xi);
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        }

        // __fdiv_rn for exact rounding under -use_fast_math; 1-ulp flips roundf() on k+0.5 ties.
        const float d  = __fmul_rn(fudge, __fdiv_rn(amax, 127.0f));
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        block_q8_0 * y = (block_q8_0 *)vy;
        y[ib].qs[lane] = (int8_t)roundf(xi*id);

        if (lane == 0) {
            // Assign __half directly (ushort assignment corrupts bits, e.g. 0x29f2->0x713e).
            y[ib].d = __float2half_rn(d);
        }
    }
}

static __global__ void quantize_q4_0_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK4_0

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK4_0 + lane];

        // argmax |x|, lowest index wins ties (matches ref scan; fixes sign of d).
        float   bval = fabsf(xi);
        int32_t bidx = lane;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float   oval = __shfl_xor_sync(0xffffffffu, bval, m);
            const int32_t oidx = __shfl_xor_sync(0xffffffffu, bidx, m);
            if (oval > bval || (oval == bval && oidx < bidx)) {
                bval = oval;
                bidx = oidx;
            }
        }

        const float max = __shfl_sync(0xffffffffu, xi, bidx);
        // __fdiv_rn for 1/d (max/-8 exact pow2); CPU stores FP16(fudge*d).
        const float d  = __fmul_rn(fudge, __fdiv_rn(max, -8.0f));
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // Truncation toward zero + clamp; __fmul_rn avoids FMA (1-ulp flips truncation).
        const float   t = __fmul_rn(xi, id);
        const int32_t v = (int32_t)(t + 8.5f);
        const int32_t q = v > 15 ? 15 : v;

        block_q4_0 * y = (block_q4_0 *)vy;
        if (lane == 0) {
            y[ib].d = __float2half_rn(d); // store the __half, see Q8_0 kernel
        }

        // Nibble pack: lane j<16 writes byte j from pair j+16.
        const uint32_t my   = (uint32_t)q & 0xF;
        const uint32_t pair = __shfl_xor_sync(0xffffffffu, my, 16);
        if (lane < 16) {
            y[ib].qs[lane] = (uint8_t)(my | (pair << 4));
        }
    }
}

// --- Q5_0 (quantize_row_q5_0_ref) ---

static __global__ void quantize_q5_0_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK5_0

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK5_0 + lane];
        float   bval = fabsf(xi);
        int32_t bidx = lane;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            const float   oval = __shfl_xor_sync(0xffffffffu, bval, m);
            const int32_t oidx = __shfl_xor_sync(0xffffffffu, bidx, m);
            if (oval > bval || (oval == bval && oidx < bidx)) {
                bval = oval;
                bidx = oidx;
            }
        }

        const float max = __shfl_sync(0xffffffffu, xi, bidx);
        // __fdiv_rn for 1/d (max/-16 exact pow2).
        const float d  = __fmul_rn(fudge, __fdiv_rn(max, -16.0f));
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // Truncation + clamp; __fmul_rn avoids FMA (matches CPU roundings).
        const float   t = __fmul_rn(xi, id);
        const int32_t v = (int32_t)(t + 16.5f);
        const int32_t q = v > 31 ? 31 : v;

        block_q5_0 * y = (block_q5_0 *)vy;
        if (lane == 0) {
            y[ib].d = __float2half_rn(d); // store the __half, see Q8_0 kernel
        }

        const uint32_t my   = (uint32_t)q & 0xF;
        const uint32_t pair = __shfl_xor_sync(0xffffffffu, my, 16);
        if (lane < 16) {
            y[ib].qs[lane] = (uint8_t)(my | (pair << 4));
        }

        // qh via __ballot_sync (must stay uniform), LE uint32.
        const uint32_t qh = __ballot_sync(0xffffffffu, (q >> 4) & 1);
        if (lane == 0) {
            y[ib].qh[0] = (uint8_t)(qh >>  0);
            y[ib].qh[1] = (uint8_t)(qh >>  8);
            y[ib].qh[2] = (uint8_t)(qh >> 16);
            y[ib].qh[3] = (uint8_t)(qh >> 24);
        }
    }
}

// --- Q5_1 (quantize_row_q5_1_ref, no fudge, no clamp) ---

static __global__ void quantize_q5_1_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    (void) fudge; // no fudge for Q5_1 (matches CPU ref)
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK5_1

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK5_1 + lane];

        // min/max reductions (values only; order-independent).
        float bmin = xi;
        float bmax = xi;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            bmin = fminf(bmin, __shfl_xor_sync(0xffffffffu, bmin, m));
            bmax = fmaxf(bmax, __shfl_xor_sync(0xffffffffu, bmax, m));
        }

        const float d  = __fdiv_rn(bmax - bmin, 31.0f);
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // (x-min)*id + 0.5f truncation toward zero, no clamp (matches ref).
        const float t = __fmul_rn(__fsub_rn(xi, bmin), id);
        const uint32_t q = (uint32_t)(int32_t)(t + 0.5f);

        block_q5_1 * y = (block_q5_1 *)vy;
        if (lane == 0) {
            // CUDA packs d/m as half2 dm (ggml-common.h); one RN conversion each.
            y[ib].dm = __floats2half2_rn(d, bmin);
        }
        const uint32_t my   = q & 0xF;
        const uint32_t pair = __shfl_xor_sync(0xffffffffu, my, 16);
        if (lane < 16) {
            y[ib].qs[lane] = (uint8_t)(my | (pair << 4));
        }

        // 5th bit -> qh bitmap, LE uint32 (same as Q5_0 kernel).
        const uint32_t qhb = __ballot_sync(0xffffffffu, (q >> 4) & 1);
        if (lane == 0) {
            y[ib].qh[0] = (uint8_t)(qhb >>  0);
            y[ib].qh[1] = (uint8_t)(qhb >>  8);
            y[ib].qh[2] = (uint8_t)(qhb >> 16);
            y[ib].qh[3] = (uint8_t)(qhb >> 24);
        }
    }
}

// --- Q4_1 (quantize_row_q4_1_ref, no fudge, MIN(15)) ---

static __global__ void quantize_q4_1_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    (void) fudge; // no fudge for Q4_1 (matches CPU ref)
    const int32_t lane = threadIdx.x; // 0 .. 31 == QK4_1

    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const float xi = x[ib*QK4_1 + lane];

        // min/max reductions (values only; order-independent).
        float bmin = xi;
        float bmax = xi;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            bmin = fminf(bmin, __shfl_xor_sync(0xffffffffu, bmin, m));
            bmax = fmaxf(bmax, __shfl_xor_sync(0xffffffffu, bmax, m));
        }

        const float d  = __fdiv_rn(bmax - bmin, 15.0f);
        const float id = d ? __fdiv_rn(1.0f, d) : 0.0f;

        // MIN(15, (int8_t)((x-min)*id + 0.5f)) truncation toward zero.
        const float t = __fmul_rn(__fsub_rn(xi, bmin), id);
        const int32_t v = (int32_t)(t + 0.5f);
        const uint32_t q = (uint32_t)(v > 15 ? 15 : v);

        block_q4_1 * y = (block_q4_1 *)vy;
        if (lane == 0) {
            // CUDA packs d/m as half2 dm (ggml-common.h); one RN conversion each.
            y[ib].dm = __floats2half2_rn(d, bmin);
        }
        const uint32_t my   = q & 0xF;
        const uint32_t pair = __shfl_xor_sync(0xffffffffu, my, 16);
        if (lane < 16) {
            y[ib].qs[lane] = (uint8_t)(my | (pair << 4));
        }
    }
}

// --- Q6_0 OLS kept (make_qx_quants+fudge; plain max/-32 unused) ---

// --- make_qx_quants device port ---

// Bit-twiddle FP16 matching ggml_compute_fp32_to_fp16; NaN payload is vendor-specific (harness nan_d_equal).
static __device__ __forceinline__ uint16_t fp32_to_fp16_ggml(float f) {
    const float scale_to_inf  = __int_as_float(0x77800000u);
    const float scale_to_zero = __int_as_float(0x08800000u);
    float base = __fmul_rn(__fmul_rn(fabsf(f), scale_to_inf), scale_to_zero);

    const uint32_t w      = __float_as_uint(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign   = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) {
        bias = 0x71000000u;
    }

    base = __fadd_rn(__int_as_float((bias >> 1) + 0x07800000u), base);
    const uint32_t bits          = __float_as_uint(base);
    const uint32_t exp_bits      = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign       = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

// round-half-to-even via the 2^23 + 2^22 magic constant (ggml-quants.c:1779)
static __device__ int nearest_int_device(float fval) {
    const unsigned int u = __float_as_uint(__fadd_rn(fval, 12582912.0f));
    return (int)((u & 0x007fffffu) - 0x00400000u);
}

// clamp_l_device(l, nmax) as in make_qx_quants
static __device__ int clamp_l_device(int l, int nmax) {
    return l > nmax-1 ? nmax-1 : (l < -nmax ? -nmax : l);
}

// make_qx_quants port (rmse_type==1 only); RN intrinsics, exact CPU order, no FMA.
static __device__ float make_qx_quants_device(int n, int nmax, const float * x, int8_t * L, const float * qw) {
    float max  = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < 1e-15f) { // GROUP_MAX_EPS: all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.0f;
    }
    float iscale = __fdiv_rn(-(float)nmax, max);

    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_device(__fmul_rn(iscale, x[i]));
        l = clamp_l_device(l, nmax);
        L[i] = (int8_t)(l + nmax);
        const float w = qw[i]; // rmse_type == 1, qw always provided
        sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
        suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
    }
    float scale = suml2 != 0.0f ? __fdiv_rn(sumlx, suml2) : 0.0f;
    float best = __fmul_rn(scale, sumlx);
    float best_sumlx = sumlx, best_suml2 = suml2;

    for (int is = -9; is <= 9; ++is) {
        iscale = __fdiv_rn(-__fadd_rn((float)nmax, __fmul_rn(0.1f, (float)is)), max);
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, x[i]));
            l = clamp_l_device(l, nmax);
            const float w = qw[i];
            sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
            suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
        }
        if (suml2 > 0.0f && __fmul_rn(sumlx, sumlx) > __fmul_rn(best, suml2)) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int_device(__fmul_rn(iscale, x[i]));
                L[i] = (int8_t)(nmax + clamp_l_device(l, nmax));
            }
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            best_sumlx = sumlx; best_suml2 = suml2;
        }
        iscale = __fdiv_rn(__fadd_rn((float)(nmax-1), __fmul_rn(0.1f, (float)is)), max);
        sumlx = suml2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, x[i]));
            l = clamp_l_device(l, nmax);
            const float w = qw[i];
            sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
            suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
        }
        if (suml2 > 0.0f && __fmul_rn(sumlx, sumlx) > __fmul_rn(best, suml2)) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int_device(__fmul_rn(iscale, x[i]));
                L[i] = (int8_t)(nmax + clamp_l_device(l, nmax));
            }
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            best_sumlx = sumlx; best_suml2 = suml2;
        }
    }

    // coordinate descent; identical step sequence to the reference
    sumlx = best_sumlx; suml2 = best_suml2;
    for (int iter = 0; iter < n*(2*nmax-1); ++iter) {
        float abs_gmax = 0.0f, gmax = 0.0f;
        int best_j = -1;
        for (int j = 0; j < n; ++j) {
            const float w = qw[j];
            const int l = (int)L[j] - nmax;
            // g = scale*w*(x[j] - scale*l), each op rounded separately
            const float g = __fmul_rn(__fmul_rn(scale, w),
                    __fadd_rn(x[j], -__fmul_rn(scale, (float)l)));
            if ((g > 0.0f && l < nmax-1) || (g < 0.0f && l > -nmax)) {
                const float ag = fabsf(g);
                if (ag > abs_gmax) { abs_gmax = ag; gmax = g; best_j = j; }
            }
        }
        if (best_j < 0) break;

        float new_sumlx = sumlx, new_suml2 = suml2;
        const float w = qw[best_j];
        int l = (int)L[best_j] - nmax;
        if (gmax > 0.0f) {
            new_sumlx = __fadd_rn(new_sumlx, __fmul_rn(w, x[best_j]));
            new_suml2 = __fadd_rn(new_suml2, __fmul_rn(w, (float)(2*l + 1)));
            l += 1;
        } else {
            new_sumlx = __fsub_rn(new_sumlx, __fmul_rn(w, x[best_j]));
            new_suml2 = __fsub_rn(new_suml2, __fmul_rn(w, (float)(2*l - 1)));
            l -= 1;
        }
        if (new_suml2 > 0.0f && __fmul_rn(new_sumlx, new_sumlx) > __fmul_rn(best, new_suml2)) {
            sumlx = new_sumlx; suml2 = new_suml2;
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
            L[best_j] = (int8_t)(l + nmax);
        } else {
            break;
        }
    }
    return scale;
}

// --- make_qkx3_quants device port (Q4_1/Q5_1; double RN intrinsics, FP_CONTRACT OFF) ---
static __device__ float make_qkx3_quants_device(int n, int nmax, const float * x, const float * weights,
        uint8_t * L, float * the_min, uint8_t * Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    double sum_w = weights ? (double)weights[0] : (double)(x[0]*x[0]);
    double sum_x = sum_w * (double)x[0];
    double sum_x2 = sum_w * (double)x[0] * (double)x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i]*x[i];
        sum_w = __dadd_rn(sum_w, (double)w);
        sum_x = __dadd_rn(sum_x, __dmul_rn((double)w, (double)x[i]));
        sum_x2 = __dadd_rn(sum_x2, __dmul_rn(__dmul_rn((double)w, (double)x[i]), (double)x[i]));
    }
    if (min > 0) {
        min = 0;
    }
    if (max - min < 1e-10f) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = __fdiv_rn((float)nmax, __fsub_rn(max, min));
    float scale = __fdiv_rn(1.0f, iscale);
    double best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_device(__fmul_rn(iscale, __fsub_rn(x[i], min)));
        l = l > nmax ? nmax : (l < 0 ? 0 : l);
        L[i] = (uint8_t)l;
        double diff = __dadd_rn(__dmul_rn((double)scale, (double)L[i]), (double)min);
        diff = __dsub_rn(diff, (double)x[i]);
        diff = use_mad ? fabs(diff) : __dmul_rn(diff, diff);
        double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
        best_mad = __dadd_rn(best_mad, __dmul_rn(w, diff));
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = __fdiv_rn(__fadd_rn(__fadd_rn(rmin, __fmul_rn(rdelta, (float)is)), (float)nmax), __fsub_rn(max, min));
        double sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_device(__fmul_rn(iscale, __fsub_rn(x[i], min)));
            l = l > nmax ? nmax : (l < 0 ? 0 : l);
            Laux[i] = (uint8_t)l;
            float w = weights ? weights[i] : x[i]*x[i];
            sum_l  = __dadd_rn(sum_l, __dmul_rn((double)w, (double)l));
            sum_l2 = __dadd_rn(sum_l2, __dmul_rn(__dmul_rn((double)w, (double)l), (double)l));
            sum_xl = __dadd_rn(sum_xl, __dmul_rn(__dmul_rn((double)w, (double)l), (double)x[i]));
        }
        double D = __dsub_rn(__dmul_rn(sum_w, sum_l2), __dmul_rn(sum_l, sum_l));
        if (D > 0) {
            double this_scale = __ddiv_rn(__dsub_rn(__dmul_rn(sum_w, sum_xl), __dmul_rn(sum_x, sum_l)), D);
            double this_min   = __ddiv_rn(__dsub_rn(__dmul_rn(sum_l2, sum_x), __dmul_rn(sum_l, sum_xl)), D);
            if (this_min > 0) {
                this_min = 0;
                this_scale = __ddiv_rn(sum_xl, sum_l2);
            }
            double mad = 0;
            if (use_mad) {
                for (int i = 0; i < n; ++i) {
                    double diff = __dadd_rn(__dmul_rn((double)this_scale, (double)Laux[i]), (double)this_min);
                    diff = __dsub_rn(diff, (double)x[i]);
                    diff = fabs(diff);
                    double w = weights ? (double)weights[i] : (double)(x[i]*x[i]);
                    mad = __dadd_rn(mad, __dmul_rn(w, diff));
                }
            } else {
                mad = __dsub_rn(sum_x2, __dmul_rn(2*this_scale, sum_xl));
                mad = __dsub_rn(mad, __dmul_rn(2*this_min, sum_x));
                mad = __dadd_rn(mad, __dmul_rn(__dmul_rn(2*this_scale, this_min), sum_l));
                mad = __dadd_rn(mad, __dmul_rn(__dmul_rn(this_scale, this_scale), sum_l2));
                mad = __dadd_rn(mad, __dmul_rn(__dmul_rn(this_min, this_min), sum_w));
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
        sum_l  = __dadd_rn(sum_l, __dmul_rn(w, (double)l));
        sum_l2 = __dadd_rn(sum_l2, __dmul_rn(__dmul_rn(w, (double)l), (double)l));
        sum_xl = __dadd_rn(sum_xl, __dmul_rn(__dmul_rn(w, (double)l), (double)x[i]));
    }
    double best = __dsub_rn(__dadd_rn(__dmul_rn(__dmul_rn(2.0, (double)scale), sum_xl), __dmul_rn(__dmul_rn(2.0, (double)min), sum_x)),
        __dmul_rn(__dmul_rn(__dmul_rn(2.0, (double)scale), (double)min), sum_l));
    best = __dsub_rn(best, __dmul_rn(__dmul_rn((double)scale, (double)scale), sum_l2));
    best = __dsub_rn(best, __dmul_rn(__dmul_rn((double)min, (double)min), sum_w));
    int last_j = -1, last_dir = 0;
    for (int itry = 0; itry < nmax*n; ++itry) {
        float gmax = 0;
        int best_j = -1, dir = 0;
        for (int j = 0; j < n; ++j) {
            // NOTE float (not double): CPU computes g in float32.
            const float g = __fsub_rn(__fsub_rn(x[j], __fmul_rn(scale, (float)L[j])), min);
            if (g > 0 && L[j] < nmax && g > gmax) {
                gmax = g; best_j = j; dir = 1;
            }
            else if (g < 0 && L[j] > 0 && -g > gmax) {
                gmax = -g; best_j = j; dir = -1;
            }
        }
        if (best_j < 0 || (best_j == last_j && dir == -last_dir)) break;
        double w = weights ? (double)weights[best_j] : (double)(x[best_j]*x[best_j]);
        sum_l  = __dadd_rn(sum_l, __dmul_rn(w, (double)dir));
        sum_l2 = __dadd_rn(sum_l2, __dmul_rn(w, (double)(2*L[best_j]*dir + 1)));
        sum_xl = __dadd_rn(sum_xl, __dmul_rn(__dmul_rn(w, (double)x[best_j]), (double)dir));
        double D = __dsub_rn(__dmul_rn(sum_w, sum_l2), __dmul_rn(sum_l, sum_l));
        if (D <= 0) break;
        double this_scale = __ddiv_rn(__dsub_rn(__dmul_rn(sum_w, sum_xl), __dmul_rn(sum_x, sum_l)), D);
        double this_min   = __ddiv_rn(__dsub_rn(__dmul_rn(sum_l2, sum_x), __dmul_rn(sum_l, sum_xl)), D);
        if (this_min > 0) {
            this_min = 0;
            this_scale = __ddiv_rn(sum_xl, sum_l2);
        }
        if (this_scale < 0) break;
        double score = __dadd_rn(__dmul_rn(2*this_scale, sum_xl), __dmul_rn(2*this_min, (double)sum_x));
        score = __dsub_rn(score, __dmul_rn(__dmul_rn(2*this_scale, this_min), sum_l));
        score = __dsub_rn(score, __dmul_rn(__dmul_rn(this_scale, this_scale), sum_l2));
        score = __dsub_rn(score, __dmul_rn(__dmul_rn(this_min, this_min), sum_w));
        if (score <= best) break;
        best = score;
        scale = (float)this_scale;
        min = (float)this_min;
        L[best_j] += (uint8_t)dir;
        last_j = best_j; last_dir = dir;
    }
    *the_min = -min;
    return scale;
}

// --- Removable Q4_0 imatrix kernel (one thread/block; base keeps chunk indexing exact) ---
static __global__ void quantize_q4_0_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK4_0;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK4_0;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK4_0];
    int8_t L[QK4_0];
    for (int j = 0; j < QK4_0; ++j) {
        // __fsqrt_rn keeps sqrt exact under -use_fast_math, no FMA.
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    const float d = make_qx_quants_device(QK4_0, 8, xb, L, weight);

    block_q4_0 * y = (block_q4_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, d)));
    for (int j = 0; j < QK4_0/2; ++j) {
        y[ib].qs[j] = (uint8_t)(L[j] | (L[j + QK4_0/2] << 4));
    }
}

// --- Q4_1 imatrix via make_qkx3 (nmax=15, no fudge) ---
static __global__ void quantize_q4_1_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    (void) fudge; // no fudge for Q4_1 (matches CPU)
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK4_1;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK4_1;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK4_1];
    uint8_t L[QK4_1], Laux[QK4_1];
    for (int j = 0; j < QK4_1; ++j) {
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    float the_min;
    const float d = make_qkx3_quants_device(QK4_1, 15, xb, weight, L, &the_min, Laux, -0.9f, 0.05f, 36, false);

    block_q4_1 * y = (block_q4_1 *)vy;
    // Bit-twiddle FP16 (NaN payload is CPU-vendor semantics).
    y[ib].dm = __halves2half2(__ushort_as_half(fp32_to_fp16_ggml(d)),
                              __ushort_as_half(fp32_to_fp16_ggml(-the_min)));
    for (int j = 0; j < QK4_1/2; ++j) {
        y[ib].qs[j] = (uint8_t)(L[j] | (L[j + QK4_1/2] << 4));
    }
}

// --- IQ4_NL (ntry=7, w=x*x) ---

static __device__ const int8_t kvalues_iq4nl_dev[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// LUT from ggml-quants.c:14823; >=16 names boundary pair (ix-16, ix-15).
static __device__ const int kIq4nlIndex[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};

static __device__ int best_index_iq4nl_device(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = kIq4nlIndex[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}

// --- IQ4 shared block optimizer (ntry=7 grid+hill-climb, exact CPU order) ---
static __device__ float iq4nl_opt_block_device(const float * xb, const float * weight, uint8_t * L) {
    float amax = 0.0f, max = 0.0f;
    for (int j = 0; j < 32; ++j) {
        const float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax; max = xb[j];
        }
    }
    if (amax < 1e-15f) {
        return 0.0f;
    }
    float d = __fdiv_rn(-max, (float)kvalues_iq4nl_dev[0]);
    const float id0 = __fdiv_rn(1.0f, d);
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int j = 0; j < 32; ++j) {
        const int l = best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(id0, xb[j]));
        L[j] = (uint8_t)l;
        const float q = (float)kvalues_iq4nl_dev[l];
        const float w = weight[j];
        sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(w, q), xb[j]));
        sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(w, q), q));
    }
    d = __fdiv_rn(sumqx, sumq2);
    float best = __fmul_rn(d, sumqx);
    float best_sumqx = sumqx, best_sumq2 = sumq2;
    for (int itry = -7; itry <= 7; ++itry) {
        float id = __fdiv_rn((float)(itry + kvalues_iq4nl_dev[0]), max);
        sumqx = sumq2 = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const int l = best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(id, xb[j]));
            const float q = (float)kvalues_iq4nl_dev[l];
            const float w = weight[j];
            sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(w, q), xb[j]));
            sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(w, q), q));
        }
        if (sumq2 > 0.0f && __fmul_rn(sumqx, sumqx) > __fmul_rn(best, sumq2)) {
            d = __fdiv_rn(sumqx, sumq2); best = __fmul_rn(d, sumqx);
            best_sumqx = sumqx; best_sumq2 = sumq2;
            for (int j = 0; j < 32; ++j) {
                L[j] = (uint8_t)best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(id, xb[j]));
            }
        }
        id = __fdiv_rn((float)(itry + kvalues_iq4nl_dev[15]), max);
        sumqx = sumq2 = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const int l = best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(id, xb[j]));
            const float q = (float)kvalues_iq4nl_dev[l];
            const float w = weight[j];
            sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(w, q), xb[j]));
            sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(w, q), q));
        }
        if (sumq2 > 0.0f && __fmul_rn(sumqx, sumqx) > __fmul_rn(best, sumq2)) {
            d = __fdiv_rn(sumqx, sumq2); best = __fmul_rn(d, sumqx);
            best_sumqx = sumqx; best_sumq2 = sumq2;
            for (int j = 0; j < 32; ++j) {
                L[j] = (uint8_t)best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(id, xb[j]));
            }
        }
    }
    sumqx = best_sumqx; sumq2 = best_sumq2;
    for (int iter = 0; iter < 32*32; ++iter) {
        float min_step = INFINITY;
        int best_j = -1, dir = 0;
        for (int j = 0; j < 32; ++j) {
            const float w = weight[j];
            const float g = __fmul_rn(__fmul_rn(d, w), __fsub_rn(xb[j], __fmul_rn(d, (float)kvalues_iq4nl_dev[L[j]])));
            if (g > 0.0f && L[j] < 15) {
                const float step = __fdiv_rn((float)(kvalues_iq4nl_dev[L[j]+1] - kvalues_iq4nl_dev[L[j]]), g);
                if (step < min_step) {
                    min_step = step; best_j = j; dir = 1;
                }
            }
            else if (g < 0.0f && L[j] > 0) {
                const float step = __fdiv_rn((float)(kvalues_iq4nl_dev[L[j]-1] - kvalues_iq4nl_dev[L[j]]), g);
                if (step < min_step) {
                    min_step = step; best_j = j; dir = -1;
                }
            }
        }
        if (best_j < 0) break;

        const float w = weight[best_j];
        const int l0 = L[best_j];
        const int l1 = l0 + dir;
        float new_sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(w, xb[best_j]), (float)(kvalues_iq4nl_dev[l1] - kvalues_iq4nl_dev[l0])));
        const int q1sq = kvalues_iq4nl_dev[l1]*kvalues_iq4nl_dev[l1];
        const int q0sq = kvalues_iq4nl_dev[l0]*kvalues_iq4nl_dev[l0];
        float new_sumq2 = __fadd_rn(sumq2, __fmul_rn(w, (float)(q1sq - q0sq)));
        if (new_sumq2 > 0.0f && __fmul_rn(new_sumqx, new_sumqx) > __fmul_rn(best, new_sumq2)) {
            sumqx = new_sumqx; sumq2 = new_sumq2;
            d = __fdiv_rn(sumqx, sumq2); best = __fmul_rn(d, sumqx);
            L[best_j] = (uint8_t)l1;
        }
        else {
            break;
        }
    }
    return d;
}

static __global__ void quantize_iq4_nl_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    (void) fudge; // no fudge for IQ4_NL (matches CPU)
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const float * xb = x + ib*QK4_NL;

    float weight[QK4_NL];
    uint8_t L[QK4_NL];
    for (int j = 0; j < QK4_NL; ++j) {
        weight[j] = __fmul_rn(xb[j], xb[j]);
    }

    const float scale = iq4nl_opt_block_device(xb, weight, L);

    // Finalize: re-quant with id=0 when scale=0; bit-twiddle FP16 for NaN payload.
    block_iq4_nl * y = (block_iq4_nl *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(scale));
    {
        const float idf = scale ? __fdiv_rn(1.0f, scale) : 0.0f;
        for (int j = 0; j < QK4_NL; ++j) {
            L[j] = (uint8_t)best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(idf, xb[j]));
        }
    }
    for (int j = 0; j < QK4_NL/2; ++j) {
        y[ib].qs[j] = (uint8_t)(L[j] | (L[j + QK4_NL/2] << 4));
    }
}

// --- IQ4_NL imatrix (same optimizer, qw*sqrt(sigma2+x*x)) ---
static __global__ void quantize_iq4_nl_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    (void) fudge; // no fudge for IQ4_NL (matches CPU)
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK4_NL;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK4_NL;
    const float s2   = sigma2[gb];

    float weight[QK4_NL];
    uint8_t L[QK4_NL];
    for (int j = 0; j < QK4_NL; ++j) {
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    const float scale = iq4nl_opt_block_device(xb, weight, L);

    block_iq4_nl * y = (block_iq4_nl *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(scale));
    {
        const float idf = scale ? __fdiv_rn(1.0f, scale) : 0.0f;
        for (int j = 0; j < QK4_NL; ++j) {
            L[j] = (uint8_t)best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(idf, xb[j]));
        }
    }
    for (int j = 0; j < QK4_NL/2; ++j) {
        y[ib].qs[j] = (uint8_t)(L[j] | (L[j + QK4_NL/2] << 4));
    }
}

// --- IQ4_XS (super_block=256, ntry=7) ---

static __global__ void quantize_iq4_xs_kernel(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    (void) fudge; // no fudge for IQ4_XS (matches CPU)
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const float * xs = x + sb*QK_K;

    float weight[32];
    uint8_t L[QK_K];
    float scales[8];
    float max_scale = 0.0f, amax_scale = 0.0f;

    for (int ib = 0; ib < 8; ++ib) {
        const float * xb = xs + ib*32;
        uint8_t * Lb = L + ib*32;
        for (int j = 0; j < 32; ++j) {
            weight[j] = __fmul_rn(xb[j], xb[j]);
        }
        // Shared optimizer (eps 0 is no-op, matches CPU continue).
        const float d = iq4nl_opt_block_device(xb, weight, Lb);
        scales[ib] = d;
        const float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    // Global scale + re-quant (CPU order, fudge=1; bit-twiddle FP16 for NaN).
    block_iq4_xs * y = (block_iq4_xs *)vy;
    const float gd = -max_scale/32.0f;
    y[sb].d = __ushort_as_half(fp32_to_fp16_ggml(gd));
    const float gid = gd ? __fdiv_rn(1.0f, gd) : 0.0f;
    uint16_t scales_h = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int l = isfinite(scales[ib]) ? nearest_int_device(__fmul_rn(gid, scales[ib])) : 0; // deterministic degenerate path (see ggml-quants.c)
        l = l > 31 ? 31 : (l < -32 ? -32 : l);
        const float dl = __fmul_rn(gd, (float)l);
        const float idl = dl ? __fdiv_rn(1.0f, dl) : 0.0f;
        uint8_t * Lb = L + ib*32;
        const float * xb = xs + ib*32;
        for (int j = 0; j < 32; ++j) {
            Lb[j] = best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(idl, xb[j]));
        }
        l += 32;
        const uint8_t l_l = (uint8_t)(l & 0xf);
        const uint8_t l_h = (uint8_t)((unsigned)l >> 4);
        if (ib % 2 == 0) {
            y[sb].scales_l[ib/2] = l_l;
        } else {
            y[sb].scales_l[ib/2] |= (uint8_t)(l_l << 4);
        }
        scales_h |= (uint16_t)(l_h << (2*(ib % 8)));
    }
    y[sb].scales_h = scales_h;
    for (int i = 0; i < QK_K/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            y[sb].qs[16*i + j] = (uint8_t)(L[32*i + j] | (L[32*i + 16 + j] << 4));
        }
    }
}

// --- IQ4_XS imatrix (same replay, qw*sqrt(sigma2+x*x)) ---
static __global__ void quantize_iq4_xs_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t sb_per_row,
        const float fudge) {
    (void) fudge; // no fudge for IQ4_XS (matches CPU)
    const int64_t sb = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (sb >= nblocks) {
        return;
    }
    const int64_t gb = base + sb;
    const float * xs = x + sb*QK_K;
    const float * qs = qw + (gb % sb_per_row)*QK_K;
    const float s2   = sigma2[gb];

    float weight[32];
    uint8_t L[QK_K];
    float scales[8];
    float max_scale = 0.0f, amax_scale = 0.0f;

    for (int ib = 0; ib < 8; ++ib) {
        const float * xb = xs + ib*32;
        const float * qb = qs + ib*32;
        uint8_t * Lb = L + ib*32;
        for (int j = 0; j < 32; ++j) {
            weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
        }
        const float d = iq4nl_opt_block_device(xb, weight, Lb);
        scales[ib] = d;
        const float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    block_iq4_xs * y = (block_iq4_xs *)vy;
    const float gd = -max_scale/32.0f;
    y[sb].d = __ushort_as_half(fp32_to_fp16_ggml(gd));
    const float gid = gd ? __fdiv_rn(1.0f, gd) : 0.0f;
    uint16_t scales_h = 0;
    for (int ib = 0; ib < 8; ++ib) {
        int l = isfinite(scales[ib]) ? nearest_int_device(__fmul_rn(gid, scales[ib])) : 0; // deterministic degenerate path (see ggml-quants.c)
        l = l > 31 ? 31 : (l < -32 ? -32 : l);
        const float dl = __fmul_rn(gd, (float)l);
        const float idl = dl ? __fdiv_rn(1.0f, dl) : 0.0f;
        uint8_t * Lb = L + ib*32;
        const float * xb = xs + ib*32;
        for (int j = 0; j < 32; ++j) {
            Lb[j] = best_index_iq4nl_device(kvalues_iq4nl_dev, __fmul_rn(idl, xb[j]));
        }
        l += 32;
        const uint8_t l_l = (uint8_t)(l & 0xf);
        const uint8_t l_h = (uint8_t)((unsigned)l >> 4);
        if (ib % 2 == 0) {
            y[sb].scales_l[ib/2] = l_l;
        } else {
            y[sb].scales_l[ib/2] |= (uint8_t)(l_l << 4);
        }
        scales_h |= (uint16_t)(l_h << (2*(ib % 8)));
    }
    y[sb].scales_h = scales_h;
    for (int i = 0; i < QK_K/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            y[sb].qs[16*i + j] = (uint8_t)(L[32*i + j] | (L[32*i + 16 + j] << 4));
        }
    }
}

// --- Removable Q5_0 imatrix (nmax=16 + qh bitmap) ---
static __global__ void quantize_q5_0_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK5_0;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK5_0;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK5_0];
    int8_t L[QK5_0];
    for (int j = 0; j < QK5_0; ++j) {
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    const float d = make_qx_quants_device(QK5_0, 16, xb, L, weight);

    block_q5_0 * y = (block_q5_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, d)));

    // qs+qh pack (L in [0,31]).
    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const uint8_t xi0 = (uint8_t)L[j];
        const uint8_t xi1 = (uint8_t)L[j + QK5_0/2];
        y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
        qh |= ((uint32_t)((xi0 & 0x10u) >> 4)) << (j + 0);
        qh |= ((uint32_t)((xi1 & 0x10u) >> 4)) << (j + QK5_0/2);
    }
    y[ib].qh[0] = (uint8_t)(qh >>  0);
    y[ib].qh[1] = (uint8_t)(qh >>  8);
    y[ib].qh[2] = (uint8_t)(qh >> 16);
    y[ib].qh[3] = (uint8_t)(qh >> 24);
}

// --- Q5_1 imatrix via make_qkx3 (nmax=31, no fudge) ---
static __global__ void quantize_q5_1_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    (void) fudge; // no fudge for Q5_1 (matches CPU)
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK5_1;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK5_1;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK5_1];
    uint8_t L[QK5_1], Laux[QK5_1];
    for (int j = 0; j < QK5_1; ++j) {
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    float the_min;
    const float d = make_qkx3_quants_device(QK5_1, 31, xb, weight, L, &the_min, Laux, -0.9f, 0.05f, 36, false);

    block_q5_1 * y = (block_q5_1 *)vy;
    // bit-twiddle FP16 (see Q4_1 imatrix kernel): NaN-scale payload must match.
    y[ib].dm = __halves2half2(__ushort_as_half(fp32_to_fp16_ggml(d)),
                              __ushort_as_half(fp32_to_fp16_ggml(-the_min)));

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const uint8_t xi0 = L[j];
        const uint8_t xi1 = L[j + QK5_1/2];
        y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
        qh |= ((uint32_t)((xi0 & 0x10u) >> 4)) << (j + 0);
        qh |= ((uint32_t)((xi1 & 0x10u) >> 4)) << (j + QK5_1/2);
    }
    y[ib].qh[0] = (uint8_t)(qh >>  0);
    y[ib].qh[1] = (uint8_t)(qh >>  8);
    y[ib].qh[2] = (uint8_t)(qh >> 16);
    y[ib].qh[3] = (uint8_t)(qh >> 24);
}

// --- Q6_0 imatrix (nmax=32, 2-bit qh) ---
static __global__ void quantize_q6_0_imatrix_kernel(
        const float * __restrict__ x, const float * __restrict__ qw, const float * __restrict__ sigma2,
        void * __restrict__ vy, const int64_t base, const int64_t nblocks, const int32_t blocks_per_row,
        const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const int64_t gb = base + ib;
    const float * xb = x + ib*QK6_0;
    const float * qb = qw + (int32_t)(gb % blocks_per_row)*QK6_0;
    const float s2   = sigma2[gb / blocks_per_row];

    float weight[QK6_0];
    int8_t L[QK6_0];
    for (int j = 0; j < QK6_0; ++j) {
        weight[j] = __fmul_rn(qb[j], __fsqrt_rn(__fadd_rn(s2, __fmul_rn(xb[j], xb[j]))));
    }

    const float d = make_qx_quants_device(QK6_0, 32, xb, L, weight);

    block_q6_0 * y = (block_q6_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, d)));

    // qs low nibbles + 2-bit qh (L in [0,63]).
    for (int j = 0; j < QK6_0/2; ++j) {
        const uint8_t xi0 = (uint8_t)L[j];
        const uint8_t xi1 = (uint8_t)L[j + QK6_0/2];
        y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
    }
    for (int k = 0; k < QK6_0/4; ++k) {
        const uint8_t a0 = (uint8_t)L[k];
        const uint8_t a1 = (uint8_t)L[k + QK6_0/2];
        const uint8_t b0 = (uint8_t)L[k + QK6_0/4];
        const uint8_t b1 = (uint8_t)L[k + 3*(QK6_0/4)];
        y[ib].qh[k] = (uint8_t)((a0 >> 4) | ((a1 >> 4) << 2)
                              | (((b0 >> 4) | ((b1 >> 4) << 2)) << 4));
    }
}

// --- Q6_0 OLS kernel (w=x*x, make_qx+fudge) ---
static __global__ void quantize_q6_0_ols_kernel(
        const float * __restrict__ x,
        void * __restrict__ vy, const int64_t nblocks, const float fudge) {
    const int64_t ib = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (ib >= nblocks) {
        return;
    }
    const float * xb = x + ib*QK6_0;

    float weight[QK6_0];
    int8_t L[QK6_0];
    for (int j = 0; j < QK6_0; ++j) {
        weight[j] = __fmul_rn(xb[j], xb[j]);
    }

    const float d = make_qx_quants_device(QK6_0, 32, xb, L, weight);

    block_q6_0 * y = (block_q6_0 *)vy;
    y[ib].d = __ushort_as_half(fp32_to_fp16_ggml(__fmul_rn(fudge, d)));

    for (int j = 0; j < QK6_0/2; ++j) {
        const uint8_t xi0 = (uint8_t)L[j];
        const uint8_t xi1 = (uint8_t)L[j + QK6_0/2];
        y[ib].qs[j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
    }
    for (int k = 0; k < QK6_0/4; ++k) {
        const uint8_t a0 = (uint8_t)L[k];
        const uint8_t a1 = (uint8_t)L[k + QK6_0/2];
        const uint8_t b0 = (uint8_t)L[k + QK6_0/4];
        const uint8_t b1 = (uint8_t)L[k + 3*(QK6_0/4)];
        y[ib].qh[k] = (uint8_t)((a0 >> 4) | ((a1 >> 4) << 2)
                              | (((b0 >> 4) | ((b1 >> 4) << 2)) << 4));
    }
}

// --- generic chunked host drivers (~128 MiB, shared) ---

using quantize_kernel_t = void (*)(const float *, void *, int64_t, float);

static size_t ggml_cuda_quantize_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        int64_t qk, size_t blk_size, quantize_kernel_t kernel, const char * name, float fudge) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % qk == 0);

    const int64_t nblocks_total = nrows*(n_per_row/qk);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    // Runs on the ambient CUDA device selected by the ggml_cuda_quantize
    // dispatcher (ggml_cuda_set_device); device 0 when called directly.

    // Chunked ~128 MiB F32 input so large tensors avoid huge VRAM allocs.
    const int64_t chunk_blocks = 1 << 20;                  // quant blocks per chunk
    const int64_t chunk_x      = chunk_blocks*qk;          // floats per chunk
    const int64_t chunk_y      = chunk_blocks*blk_size;    // bytes per chunk

    float   * x_dev = nullptr;
    uint8_t * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(x_dev, %" PRId64 "): %s\n",
                __func__, name, (int64_t)(chunk_x*sizeof(float)), cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(y_dev, %" PRId64 "): %s\n",
                __func__, name, (int64_t)chunk_y, cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }

    // one warp per quant block
    const int64_t block_size = qk;

    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*qk, nblocks*qk*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpy H2D: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }

        kernel<<<(unsigned)nblocks, (unsigned)block_size>>>(x_dev, y_dev, nblocks, fudge);

        err = cudaMemcpy((char *)dst + base*blk_size, y_dev, nblocks*blk_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpy D2H: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*blk_size;
}

using imatrix_kernel_t = void (*)(const float *, const float *, const float *, void *, int64_t, int64_t, int32_t, float);

// Shared chunked driver for all imatrix kernels. sigma covers sigma_len values
// per entry: a full row (sum/n_per_row, CPU row order) or one quant unit
// (sum*(2/qk), CPU block order); kernels index it per row or per block.
static size_t ggml_cuda_quantize_imatrix_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix, int64_t qk, size_t blk_size, imatrix_kernel_t kernel, const char * name, float fudge,
        bool per_block_sigma) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % qk == 0);

    const int64_t nblocks_total = nrows*(n_per_row/qk);
    const int32_t blocks_per_row = (int32_t)(n_per_row/qk);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    // Runs on the ambient device (dispatcher selects it).

    const int64_t nsigma = per_block_sigma ? nblocks_total : nrows;
    const int64_t sigma_len = per_block_sigma ? qk : n_per_row;
    std::vector<float> sigma2(nsigma);
    for (int64_t i = 0; i < nsigma; ++i) {
        const float * xr = src + i*sigma_len;
        float sum = 0.0f;
        for (int64_t j = 0; j < sigma_len; ++j) {
            sum += xr[j]*xr[j];
        }
        sigma2[i] = per_block_sigma ? sum*(2.0f/(float)qk) : sum/(float)n_per_row;
    }

    const int64_t chunk_blocks = std::max<int64_t>(1, (128ll << 20)/(qk*(int64_t)sizeof(float)));
    const int64_t chunk_x      = chunk_blocks*qk;
    const int64_t chunk_y      = chunk_blocks*(int64_t)blk_size;

    float   * x_dev = nullptr;
    float   * q_dev = nullptr;
    float   * s_dev = nullptr;
    uint8_t * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(x_dev): %s\n", __func__, name, cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&q_dev, n_per_row*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(q_dev): %s\n", __func__, name, cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }
    err = cudaMalloc(&s_dev, nsigma*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(s_dev): %s\n", __func__, name, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(y_dev): %s\n", __func__, name, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        return 0;
    }

    err = cudaMemcpy(q_dev, imatrix, n_per_row*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMemcpy imatrix H2D: %s\n", __func__, name, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }
    err = cudaMemcpy(s_dev, sigma2.data(), nsigma*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMemcpy sigma2 H2D: %s\n", __func__, name, cudaGetErrorString(err));
        cudaFree(x_dev);
        cudaFree(q_dev);
        cudaFree(s_dev);
        cudaFree(y_dev);
        return 0;
    }

    const unsigned int block_size = 256;
    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*qk, nblocks*qk*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpy H2D: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }

        kernel<<<(unsigned)((nblocks + block_size - 1)/block_size), block_size>>>(
                x_dev, q_dev, s_dev, y_dev, base, nblocks, blocks_per_row, fudge);

        err = cudaMemcpy((char *)dst + base*blk_size, y_dev, nblocks*blk_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpy D2H: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(q_dev);
    cudaFree(s_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*blk_size;
}

// Order Q8_0/Q6_0/Q5_0/Q4_0; Q8_0 keeps Q6_0-fudge quirk (ggml-quants.c:915).
size_t ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0); // match CPU quirk
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK8_0, sizeof(block_q8_0), quantize_q8_0_kernel, "q8_0", fudge);
}


// --- Removable Q5_0 section (delete kernel+helpers+cases to drop) ---
size_t ggml_cuda_quantize_q5_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q5_0);
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK5_0, sizeof(block_q5_0), quantize_q5_0_kernel, "q5_0", fudge);
}

// --- Removable Q4_0 section (delete kernel+helpers+cases to drop) ---
size_t ggml_cuda_quantize_q4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    const float fudge = ggml_get_quantize_fudge_factor(GGML_TYPE_Q4_0);
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK4_0, sizeof(block_q4_0), quantize_q4_0_kernel, "q4_0", fudge);
}

// --- Q4_1 (quantize_row_q4_1_ref, no fudge) ---
size_t ggml_cuda_quantize_q4_1(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK4_1, sizeof(block_q4_1), quantize_q4_1_kernel, "q4_1", 1.0f);
}

// --- Removable Q4_0 imatrix (reuses row imatrix like CPU) ---
size_t ggml_cuda_quantize_q4_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK4_0, sizeof(block_q4_0),
            quantize_q4_0_imatrix_kernel, "q4_0_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_Q4_0), false);
}

// --- Removable Q5_0 imatrix (mirror of Q4_0 imatrix) ---
size_t ggml_cuda_quantize_q5_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK5_0, sizeof(block_q5_0),
            quantize_q5_0_imatrix_kernel, "q5_0_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_Q5_0), false);
}

// --- Q6_0 OLS imatrix (mirror of Q4_0 imatrix, nmax=32) ---
size_t ggml_cuda_quantize_q6_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK6_0, sizeof(block_q6_0),
            quantize_q6_0_imatrix_kernel, "q6_0_imatrix", ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0), false);
}

// --- Q5_1 (quantize_row_q5_1_ref, no fudge) ---
size_t ggml_cuda_quantize_q5_1(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_generic(src, dst, nrows, n_per_row,
            QK5_1, sizeof(block_q5_1), quantize_q5_1_kernel, "q5_1", 1.0f);
}

// --- Q5_1 imatrix via make_qkx3 (no fudge) ---
size_t ggml_cuda_quantize_q5_1_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK5_1, sizeof(block_q5_1),
            quantize_q5_1_imatrix_kernel, "q5_1_imatrix", 1.0f, false);
}

// --- Q4_1 imatrix via make_qkx3 (no fudge) ---
size_t ggml_cuda_quantize_q4_1_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK4_1, sizeof(block_q4_1),
            quantize_q4_1_imatrix_kernel, "q4_1_imatrix", 1.0f, false);
}

// --- IQ4_NL (no imatrix, ntry=7) ---
using ols_kernel_t = void (*)(const float *, void *, int64_t, float);

// Shared chunked driver for one-thread-per-block kernels (Q6_0 OLS, IQ4_NL/XS).
static size_t ggml_cuda_quantize_ols_generic(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        int64_t qk, size_t blk_size, ols_kernel_t kernel, const char * name, float fudge) {
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(n_per_row % qk == 0);

    const int64_t nblocks_total = nrows*(n_per_row/qk);

    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        return 0;
    }
    // Runs on the ambient device (dispatcher selects it).

    const int64_t chunk_blocks = std::max<int64_t>(1, (128ll << 20)/(qk*(int64_t)sizeof(float)));
    const int64_t chunk_x      = chunk_blocks*qk;
    const int64_t chunk_y      = chunk_blocks*(int64_t)blk_size;

    float   * x_dev = nullptr;
    uint8_t * y_dev = nullptr;

    cudaError_t err = cudaMalloc(&x_dev, chunk_x*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(x_dev): %s\n", __func__, name, cudaGetErrorString(err));
        return 0;
    }
    err = cudaMalloc(&y_dev, chunk_y);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s: cudaMalloc(y_dev): %s\n", __func__, name, cudaGetErrorString(err));
        cudaFree(x_dev);
        return 0;
    }

    const unsigned int block_size = 256;
    for (int64_t base = 0; base < nblocks_total; base += chunk_blocks) {
        const int64_t nblocks = std::min(chunk_blocks, nblocks_total - base);

        err = cudaMemcpy(x_dev, src + base*qk, nblocks*qk*sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpy H2D: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }

        kernel<<<(unsigned)((nblocks + block_size - 1)/block_size), block_size>>>(
                x_dev, y_dev, nblocks, fudge);

        err = cudaMemcpy((char *)dst + base*blk_size, y_dev, nblocks*blk_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "%s: %s: cudaMemcpy D2H: %s\n", __func__, name, cudaGetErrorString(err));
            break;
        }
    }

    cudaFree(x_dev);
    cudaFree(y_dev);

    if (err != cudaSuccess) {
        return 0;
    }

    return nblocks_total*blk_size;
}

size_t ggml_cuda_quantize_iq4_nl(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_ols_generic(src, dst, nrows, n_per_row, QK4_NL, sizeof(block_iq4_nl),
            quantize_iq4_nl_kernel, "iq4_nl", 1.0f);
}

// --- IQ4_NL imatrix (qw*sqrt(sigma2+x*x), host sigma2 per 32-block) ---
size_t ggml_cuda_quantize_iq4_nl_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK4_NL, sizeof(block_iq4_nl),
            quantize_iq4_nl_imatrix_kernel, "iq4_nl_imatrix", 1.0f, true);
}

// --- IQ4_XS (no imatrix, ntry=7) ---
size_t ggml_cuda_quantize_iq4_xs(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_ols_generic(src, dst, nrows, n_per_row, QK_K, sizeof(block_iq4_xs),
            quantize_iq4_xs_kernel, "iq4_xs", 1.0f);
}

// --- IQ4_XS imatrix (host sigma2 per superblock) ---
size_t ggml_cuda_quantize_iq4_xs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    return ggml_cuda_quantize_imatrix_generic(src, dst, nrows, n_per_row, imatrix, QK_K, sizeof(block_iq4_xs),
            quantize_iq4_xs_imatrix_kernel, "iq4_xs_imatrix", 1.0f, true);
}

// --- Q6_0 OLS driver (w=x*x, make_qx+fudge) ---
size_t ggml_cuda_quantize_q6_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row) {
    return ggml_cuda_quantize_ols_generic(src, dst, nrows, n_per_row, QK6_0, sizeof(block_q6_0),
            quantize_q6_0_ols_kernel, "q6_0", ggml_get_quantize_fudge_factor(GGML_TYPE_Q6_0));
}

// Q8_0 imatrix routes to plain (CPU ignores imatrix).
size_t ggml_cuda_quantize_q8_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix) {
    (void) imatrix; // CPU ignores it; match exactly
    return ggml_cuda_quantize_q8_0(src, dst, nrows, n_per_row);
}
