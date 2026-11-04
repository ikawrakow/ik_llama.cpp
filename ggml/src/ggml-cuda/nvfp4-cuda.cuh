#pragma once

// NVFP4 device-side helpers, ported from upstream llama.cpp.
// (ik does not carry mainline's PDL/fp8 feature macros; the software path is exact
// and matches the CPU ggml_ue4m3_to_fp32 semantics.)

#include <cstdint>

static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return raw * 0.5f;
}
