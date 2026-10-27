// Step-5 E4M3FN helpers: F32 -> BF16 (RNE) -> E4M3FN (RN, satfinite) -> F32; shared by
// the ggml.c CPU forward and test probes. E4M3FN: bias 7, no inf, 0x7F = NaN, max 448.
#pragma once

#include "ggml-impl.h"

static inline uint8_t ggml_f32_to_e4m3fn_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    const uint32_t s = (u >> 31) << 7;
    const uint32_t e32 = (u >> 23) & 0xFF;
    const uint32_t m32 = u & 0x7FFFFF;
    if (e32 == 0xFF && m32) return (uint8_t)(s | 0x7F);        // NaN in -> NaN out
    if (e32 == 0) return (uint8_t) s;                          // f32 subnormal << min subnormal 2^-9
    if (e32 == 0xFF) return (uint8_t)(s | 0x7E);               // inf -> satfinite maxnorm 448
    const uint32_t mant = m32 | 0x800000;                      // 24-bit significand (e32 > 0)
    const int e4 = (int) e32 - 120;                            // value = mant * 2^(e4-30)
    if (e4 >= 1) {
        // normal: RNE the 24-bit significand to 3 fraction bits; carry bumps the exponent
        const uint32_t rem = mant & 0xFFFFF;
        uint32_t q4 = mant >> 20;
        if (rem > 0x80000 || (rem == 0x80000 && (q4 & 1))) q4++;   // RNE
        int E = e4;
        int m3 = (int) q4 & 7;
        if (q4 == 16) { E++; m3 = 0; }
        if (E > 15 || (E == 15 && m3 == 7)) return (uint8_t)(s | 0x7E); // satfinite -> 448 (480 is the NaN code)
        return (uint8_t)(s | ((E << 3) | m3));
    }
    // subnormal (e4 <= 0): q = rne(mant, 21 - e4)
    const int sh = 21 - e4;
    if (sh >= 25) return (uint8_t) s;                          // < 2^-10 even after RNE
    uint32_t rem = mant & ((1u << sh) - 1);
    int q = (int)(mant >> sh);
    if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (q & 1))) q++; // RNE
    if (q >= 8) return (uint8_t)(s | 0x08);                    // carry into the smallest normal (2^-6)
    return (uint8_t)(s | q);                                   // q 0..7 subnormal
}

static inline float ggml_e4m3fn_bits_to_f32(uint8_t b) {
    const float sgn = (b & 0x80) ? -1.0f : 1.0f;
    const int em = b & 0x7F;
    if (em == 0x7F) return NAN;
    const int E = em >> 3;
    const int m = em & 7;
    if (E == 0) return sgn * m * 0x1p-9f;                      // subnormal: m * 2^-9
    return sgn * (1.0f + m / 8.0f) * ldexpf(1.0f, E - 7);      // E==15, m<7 valid (max 448)
}

static inline float ggml_csa_round_bf16_e4m3(float x) {
    return ggml_e4m3fn_bits_to_f32(ggml_f32_to_e4m3fn_bits(GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(x))));
}

