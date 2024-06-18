#pragma once

#include <stdint.h>

typedef union {
    float    f;
    uint32_t i;
} iq1bn_scale_t;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef BITNET_IQ1BN_4x4
static inline float iq1bn_min_value(void) { return 1.9074e-06f; }
static inline float iq1bn_max_value(void) { return 0.12109f; }
#else
static inline float iq1bn_min_value(void) { return 0.000488281f; }
static inline float iq1bn_max_value(void) { return 0.123047f; }
#endif

static inline uint8_t iq1bn_float_to_fp8(float f) {
    if (f <= iq1bn_min_value()) return 0;
    if (f >= iq1bn_max_value()) return 255;
    iq1bn_scale_t s;
    s.f = f;
#ifdef BITNET_IQ1BN_4x4
    return ((((s.i >> 23) + 132) & 0xf) << 4) | ((s.i >> 19) & 0xf);
#else
    return ((s.i >> 18) & 0x1f) | (((s.i >> 23) - 116) << 5);
#endif
}

static inline float iq1bn_fp8_to_float(uint8_t fp8) {
    iq1bn_scale_t s;
#ifdef BITNET_IQ1BN_4x4
    s.i = ((((fp8 >> 4) | 0xf0) - 132) << 23) | ((fp8 & 0x0f) << 19);
#else
    s.i = (((fp8 >> 5) + 116) << 23) | ((fp8 & 0x1f) << 18);
#endif
    return s.f;
}

#ifdef __cplusplus
}
#endif
