//
// Copyright (C) 2026 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

// CPU quantisers for the packed KV-cache storage types.
// See ggml-kv-quants.h for what these types are and where the rule comes from.

#include "ggml-kv-quants.h"

#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// bit tricks
// ---------------------------------------------------------------------------

// ceil(log2(x)) for a normal, positive float, via the exponent field.
// x >= 2^-126 and finite: guaranteed by the per-type amax floor below.
static inline int32_t kv_log2_ceil(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (int32_t)((u >> 23) & 0xFFu) - 127 + ((u & 0x7FFFFFu) != 0u);
}

static inline float kv_pow2(int32_t i) {
    const uint32_t u = (uint32_t)(i + 127) << 23;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// 2^ceil(log2(x * max_inv)) - the reference's fast_round_scale
static inline float kv_round_scale(float x, float max_inv) {
    return kv_pow2(kv_log2_ceil(x * max_inv));
}

// ---------------------------------------------------------------------------
// e4m3 (e4m3fn) <-> fp32: direct bit construction; encode rounds to nearest,
// ties to even, saturating at +-448.
// ---------------------------------------------------------------------------

// e4m3 subnormals: m * 2^-9 for m = 0..7
static const float kv_e4m3_sub[8] = {
    0.0f, 0x1p-9f, 0x1p-8f, 0x1.8p-8f, 0x1p-7f, 0x1.4p-7f, 0x1.8p-7f, 0x1.cp-7f,
};

static inline float kv_e4m3_to_fp32(uint8_t v) {
    const uint32_t s = (uint32_t)(v >> 7) << 31;
    const uint32_t e = ((uint32_t)v >> 3) & 0xFu;
    const uint32_t m =  (uint32_t)v       & 0x7u;
    if (e == 0u) {
        // subnormal: m * 2^-9
        const float f = kv_e4m3_sub[m];
        return s ? -f : f;
    }
    const uint32_t u = s | ((e + 120u) << 23) | (m << 20);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint8_t kv_fp32_to_e4m3(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    const uint32_t sign = (u >> 24) & 0x80u;
    uint32_t       mag  = u & 0x7FFFFFFFu;

    if (mag >= 0x7F800000u || mag > 0x43E00000u) {   // NaN/inf, or |x| > 448
        return (uint8_t)(sign | 0x7Eu);              // saturate at 448
    }
    if (mag < 0x3C800000u) {                         // |x| < 2^-6: the subnormal grid
        // t = |x| * 2^9 is exact (a power-of-two scaling), so this rounding is exact
        const float t    = fabsf(x) * 512.0f;
        const float fl   = floorf(t);
        uint32_t    m    = (uint32_t)fl;
        const float frac = t - fl;
        if (frac > 0.5f || (frac == 0.5f && (m & 1u))) {
            ++m;
        }
        if (m == 8u) {                               // 8 * 2^-9 is 2^-6: the smallest normal
            return (uint8_t)(sign | 0x08u);
        }
        return (uint8_t)(sign | m);                  // 0 <= m <= 7: subnormal, or zero
    }
    // |x| >= 2^-6: round the 3-bit mantissa (ties to even) and carry
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

// ---------------------------------------------------------------------------
// e2m1: code bits 2..0 are the magnitude, bit 3 the sign.
// ---------------------------------------------------------------------------

static const float kv_e2m1[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

// Round |q| (already clamped to 6) onto the e2m1 grid, ties to even.
static inline uint32_t kv_e2m1_index(float aq) {
    if (aq <= 0.25f) return 0;
    if (aq <  0.75f) return 1;
    if (aq <= 1.25f) return 2;
    if (aq <  1.75f) return 3;
    if (aq <= 2.50f) return 4;
    if (aq <  3.50f) return 5;
    if (aq <= 5.00f) return 6;
    return 7;
}

// One packed code for x against the block scale d.
static inline uint8_t kv_fp4_code(float x, float d) {
    const float q  = x / d;                          // the reference divides, then clamps
    float       aq = fabsf(q);
    if (aq > 6.0f) {
        aq = 6.0f;
    }
    return (uint8_t)(kv_e2m1_index(aq) | (q < 0.0f ? 8u : 0u));
}

// ---------------------------------------------------------------------------
// the shared cores
// ---------------------------------------------------------------------------

// e2m1 blocks with a one-byte scale: block 16 with an e4m3 scale, block 32
// with an e8m0 one.
static void kv_quantize_fp4(const float * x, void * vy, int64_t k, int block, int scale_e4m3) {
    assert(k % block == 0);
    uint8_t *     y      = (uint8_t *) vy;
    const int64_t nblock = k / block;
    const size_t  bs     = 1 + block/2;

    for (int64_t ib = 0; ib < nblock; ++ib) {
        const float * xb = x + ib*block;
        uint8_t *     yb = y + ib*bs;

        float amax = 0.0f;
        for (int j = 0; j < block; ++j) {
            const float ax = fabsf(xb[j]);
            if (ax > amax) amax = ax;
        }

        float d;
        if (scale_e4m3) {
            if (amax < 6.0f * 0x1p-9f) {             // 6 * 2^-9
                amax = 6.0f * 0x1p-9f;
            }
            const uint8_t e = kv_fp32_to_e4m3(amax / 6.0f);
            yb[0] = e;
            d     = kv_e4m3_to_fp32(e);
        } else {
            if (amax < 6.0f * 0x1p-126f) {           // 6 * 2^-126
                amax = 6.0f * 0x1p-126f;
            }
            const int32_t l = kv_log2_ceil(amax * (1.0f/6.0f));
            yb[0] = (uint8_t)(l + 127);              // the e8m0 byte
            d     = kv_pow2(l);
        }

        // reference packing: element 2j is the low nibble, element 2j+1 the high one
        for (int j = 0; j < block/2; ++j) {
            const uint8_t c0 = kv_fp4_code(xb[2*j    ], d);
            const uint8_t c1 = kv_fp4_code(xb[2*j + 1], d);
            yb[1 + j] = (uint8_t)(c0 | (c1 << 4));
        }
    }
}

static void kv_dequantize_fp4(const void * vx, float * y, int64_t k, int block, int scale_e4m3) {
    const uint8_t * x      = (const uint8_t *) vx;
    const int64_t   nblock = k / block;
    const size_t    bs     = 1 + block/2;

    for (int64_t ib = 0; ib < nblock; ++ib) {
        const uint8_t * xb = x + ib*bs;
        const float d = scale_e4m3 ? kv_e4m3_to_fp32(xb[0]) : kv_pow2((int32_t)xb[0] - 127);
        float * yb = y + ib*block;
        for (int j = 0; j < block/2; ++j) {
            // the code is 4 bits: bit 3 is the sign, bits 2..0 are the magnitude
            const uint8_t c0 =  xb[1 + j]       & 0x0Fu;
            const uint8_t c1 = (xb[1 + j] >> 4) & 0x0Fu;
            yb[2*j    ] = ((c0 & 0x08u) ? -d : d) * kv_e2m1[c0 & 0x07u];
            yb[2*j + 1] = ((c1 & 0x08u) ? -d : d) * kv_e2m1[c1 & 0x07u];
        }
    }
}

static void kv_quantize_fp8(const float * x, void * vy, int64_t k, int block) {
    assert(k % block == 0);
    uint8_t *     y      = (uint8_t *) vy;
    const int64_t nblock = k / block;
    const size_t  bs     = 1 + block;

    for (int64_t ib = 0; ib < nblock; ++ib) {
        const float * xb = x + ib*block;
        uint8_t *     yb = y + ib*bs;

        float amax = 0.0f;
        for (int j = 0; j < block; ++j) {
            const float ax = fabsf(xb[j]);
            if (ax > amax) amax = ax;
        }
        if (amax < 1e-4f) {
            amax = 1e-4f;
        }
        const int32_t l = kv_log2_ceil(amax * (1.0f/448.0f));
        yb[0] = (uint8_t)(l + 127);                  // the e8m0 byte
        const float d = kv_pow2(l);

        for (int j = 0; j < block; ++j) {
            float q = xb[j] / d;
            if (q >  448.0f) q =  448.0f;
            if (q < -448.0f) q = -448.0f;
            yb[1 + j] = kv_fp32_to_e4m3(q);
        }
    }
}

static void kv_dequantize_fp8(const void * vx, float * y, int64_t k, int block) {
    const uint8_t * x      = (const uint8_t *) vx;
    const int64_t   nblock = k / block;
    const size_t    bs     = 1 + block;

    for (int64_t ib = 0; ib < nblock; ++ib) {
        const uint8_t * xb = x + ib*bs;
        const float d = kv_pow2((int32_t)xb[0] - 127);
        float * yb = y + ib*block;
        for (int j = 0; j < block; ++j) {
            yb[j] = d * kv_e4m3_to_fp32(xb[1 + j]);
        }
    }
}

// ---------------------------------------------------------------------------
// the three types
// ---------------------------------------------------------------------------

void quantize_row_fp4_b16_e4m3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    kv_quantize_fp4(x, y, k, QK_FP4_B16, /*scale_e4m3*/ 1);
}

void dequantize_row_fp4_b16_e4m3(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    kv_dequantize_fp4(x, y, k, QK_FP4_B16, /*scale_e4m3*/ 1);
}

void quantize_row_fp4_b32_e8m0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    kv_quantize_fp4(x, y, k, QK_FP4_B32, /*scale_e4m3*/ 0);
}

void dequantize_row_fp4_b32_e8m0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    kv_dequantize_fp4(x, y, k, QK_FP4_B32, /*scale_e4m3*/ 0);
}

void quantize_row_fp8_b32_e8m0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    kv_quantize_fp8(x, y, k, QK_FP8_B32);
}

void dequantize_row_fp8_b32_e8m0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    kv_dequantize_fp8(x, y, k, QK_FP8_B32);
}