//
// Copyright (C) 2026 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

// Packed KV-cache storage types - CUDA device code.

#pragma once

#include "common.cuh"

// elements per packed unit: 2 for the e2m1 types, 1 for the e4m3 one
#define QR_FP4 2
#define QR_FP8 1

// e2m1 magnitudes: the reference grid (kvalues_mxfp4 above is doubled)
static const __device__ float kvalues_e2m1_kv[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

// the e4m3 subnormal grid: m * 2^-9 for m = 0..7
static const __device__ float kvalues_e4m3_sub_kv[8] = {
    0.0f, 0x1p-9f, 0x1p-8f, 0x1.8p-8f, 0x1p-7f, 0x1.4p-7f, 0x1.8p-7f, 0x1.cp-7f,
};

// 2^i for i >= -126 (the e8m0 byte is never 0: the per-type amax floors keep
// the exponent at -126 or above, exactly as on the CPU)
static __device__ __forceinline__ float kv_pow2_dev(int i) {
    return __int_as_float((i + 127) << 23);
}

// ceil(log2(x)) for a normal, positive float, via the exponent field
static __device__ __forceinline__ int kv_log2_ceil_dev(float x) {
    const uint32_t u = __float_as_uint(x);
    return (int)((u >> 23) & 0xFFu) - 127 + ((u & 0x7FFFFFu) != 0u);
}

// e4m3fn -> fp32 (subnormal grid m * 2^-9).
static __device__ __forceinline__ float kv_e4m3_to_fp32_dev(uint8_t v) {
    const uint32_t s = (uint32_t)(v >> 7) << 31;
    const uint32_t e = ((uint32_t)v >> 3) & 0xFu;
    const uint32_t m =  (uint32_t)v       & 0x7u;
    if (e == 0u) {
        const float f = kvalues_e4m3_sub_kv[m];
        return s ? -f : f;
    }
    return __uint_as_float(s | ((e + 120u) << 23) | (m << 20));
}

// fp32 -> e4m3fn, round to nearest with ties to even, saturating at +-448.
static __device__ __forceinline__ uint8_t kv_fp32_to_e4m3_dev(float x) {
    const uint32_t u    = __float_as_uint(x);
    const uint32_t sign = (u >> 24) & 0x80u;
    const uint32_t mag  = u & 0x7FFFFFFFu;

    if (mag >= 0x7F800000u || mag > 0x43E00000u) {   // NaN/inf, or |x| > 448
        return (uint8_t)(sign | 0x7Eu);              // saturate at 448
    }
    if (mag < 0x3C800000u) {                         // |x| < 2^-6: the subnormal grid
        const float t    = fabsf(x) * 512.0f;        // exact: a power-of-two scaling
        const float fl   = floorf(t);
        uint32_t    m    = (uint32_t)fl;
        const float frac = t - fl;
        if (frac > 0.5f || (frac == 0.5f && (m & 1u))) {
            ++m;
        }
        if (m == 8u) {                               // 8 * 2^-9 is 2^-6: the smallest normal
            return (uint8_t)(sign | 0x08u);
        }
        return (uint8_t)(sign | m);
    }
    uint32_t       e    = (mag >> 23) - 127u;
    uint32_t       m    = (mag >> 20) & 0x7u;
    const uint32_t rest = mag & 0xFFFFFu;            // the bits below the 3 kept ones
    if (rest > 0x80000u || (rest == 0x80000u && (m & 1u))) {
        if (++m == 8u) { m = 0u; ++e; }
    }
    const uint32_t e_enc = e + 7u;
    if (e_enc > 15u || (e_enc == 15u && m > 6u)) {   // 480 would be NaN in e4m3fn
        return (uint8_t)(sign | 0x7Eu);              // saturate at 448
    }
    return (uint8_t)(sign | (e_enc << 3) | m);
}

// Round |q| (already clamped to 6) onto the e2m1 grid, ties to even.
static __device__ __forceinline__ uint32_t kv_e2m1_index_dev(float aq) {
    if (aq <= 0.25f) return 0;
    if (aq <  0.75f) return 1;
    if (aq <= 1.25f) return 2;
    if (aq <  1.75f) return 3;
    if (aq <= 2.50f) return 4;
    if (aq <  3.50f) return 5;
    if (aq <= 5.00f) return 6;
    return 7;
}

static __device__ __forceinline__ uint8_t kv_fp4_code_dev(float x, float d) {
    const float q  = x / d;                          // divide first, then clamp
    float       aq = fabsf(q);
    if (aq > 6.0f) {
        aq = 6.0f;
    }
    return (uint8_t)(kv_e2m1_index_dev(aq) | (q < 0.0f ? 8u : 0u));
}

// ---------------------------------------------------------------------------
// the write side (ggml_set_rows destination): quantize a block of F32 values
// ---------------------------------------------------------------------------

static __device__ void quantize_f32_fp4_b16_e4m3_block(const float * __restrict__ x, block_fp4_b16_e4m3 * __restrict__ y) {
    float amax = 0.0f;
    for (int j = 0; j < QK_FP4_B16; ++j) {
        amax = fmaxf(amax, fabsf(x[j]));
    }
    if (amax < 6.0f * 0x1p-9f) {                     // 6 * 2^-9
        amax = 6.0f * 0x1p-9f;
    }
    const uint8_t e = kv_fp32_to_e4m3_dev(amax / 6.0f);
    y->d = e;
    const float d = kv_e4m3_to_fp32_dev(e);
    for (int j = 0; j < QK_FP4_B16/2; ++j) {
        y->qs[j] = (uint8_t)(kv_fp4_code_dev(x[2*j], d) | (kv_fp4_code_dev(x[2*j + 1], d) << 4));
    }
}

static __device__ void quantize_f32_fp4_b32_e8m0_block(const float * __restrict__ x, block_fp4_b32_e8m0 * __restrict__ y) {
    float amax = 0.0f;
    for (int j = 0; j < QK_FP4_B32; ++j) {
        amax = fmaxf(amax, fabsf(x[j]));
    }
    if (amax < 6.0f * 0x1p-126f) {                   // 6 * 2^-126
        amax = 6.0f * 0x1p-126f;
    }
    const int l = kv_log2_ceil_dev(amax * (1.0f/6.0f));
    y->d = (uint8_t)(l + 127);                       // the e8m0 byte
    const float d = kv_pow2_dev(l);
    for (int j = 0; j < QK_FP4_B32/2; ++j) {
        y->qs[j] = (uint8_t)(kv_fp4_code_dev(x[2*j], d) | (kv_fp4_code_dev(x[2*j + 1], d) << 4));
    }
}

static __device__ void quantize_f32_fp8_b32_e8m0_block(const float * __restrict__ x, block_fp8_b32_e8m0 * __restrict__ y) {
    float amax = 0.0f;
    for (int j = 0; j < QK_FP8_B32; ++j) {
        amax = fmaxf(amax, fabsf(x[j]));
    }
    if (amax < 1e-4f) {
        amax = 1e-4f;
    }
    const int l = kv_log2_ceil_dev(amax * (1.0f/448.0f));
    y->d = (uint8_t)(l + 127);                       // the e8m0 byte
    const float d = kv_pow2_dev(l);
    for (int j = 0; j < QK_FP8_B32; ++j) {
        float q = x[j] / d;
        if (q >  448.0f) q =  448.0f;
        if (q < -448.0f) q = -448.0f;
        y->qs[j] = kv_fp32_to_e4m3_dev(q);
    }
}

// ---------------------------------------------------------------------------
// the read side (ggml_get_rows): one (v.x, v.y) pair per call, v.x at iqs and
// v.y at iqs + qk/2
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void dequantize_fp4_b16_e4m3(const void * vx, const int64_t ib, const int iqs, dfloat2 & v) {
    const block_fp4_b16_e4m3 * x = (const block_fp4_b16_e4m3 *) vx + ib;
    const float d = kv_e4m3_to_fp32_dev(x->d);
    const uint8_t b0 = x->qs[iqs/2];
    const uint8_t b1 = x->qs[iqs/2 + QK_FP4_B16/4];
    const uint8_t c0 = (b0 >> (4*(iqs%2))) & 0x0Fu;
    const uint8_t c1 = (b1 >> (4*(iqs%2))) & 0x0Fu;
    v.x = ((c0 & 0x08u) ? -d : d) * kvalues_e2m1_kv[c0 & 0x07u];
    v.y = ((c1 & 0x08u) ? -d : d) * kvalues_e2m1_kv[c1 & 0x07u];
}

static __device__ __forceinline__ void dequantize_fp4_b32_e8m0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v) {
    const block_fp4_b32_e8m0 * x = (const block_fp4_b32_e8m0 *) vx + ib;
    const float d = kv_pow2_dev((int)x->d - 127);
    const uint8_t b0 = x->qs[iqs/2];
    const uint8_t b1 = x->qs[iqs/2 + QK_FP4_B32/4];
    const uint8_t c0 = (b0 >> (4*(iqs%2))) & 0x0Fu;
    const uint8_t c1 = (b1 >> (4*(iqs%2))) & 0x0Fu;
    v.x = ((c0 & 0x08u) ? -d : d) * kvalues_e2m1_kv[c0 & 0x07u];
    v.y = ((c1 & 0x08u) ? -d : d) * kvalues_e2m1_kv[c1 & 0x07u];
}

static __device__ __forceinline__ void dequantize_fp8_b32_e8m0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v) {
    const block_fp8_b32_e8m0 * x = (const block_fp8_b32_e8m0 *) vx + ib;
    const float d = kv_pow2_dev((int)x->d - 127);
    v.x = d * kv_e4m3_to_fp32_dev(x->qs[iqs    ]);
    v.y = d * kv_e4m3_to_fp32_dev(x->qs[iqs + 1]);
}