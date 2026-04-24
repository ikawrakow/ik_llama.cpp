#pragma once

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
static const uint3 init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    uint32_t d = (uint32_t)d_64;

    // compute L = ceil(log2(d));
    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    // pack divisor as well to reduce error surface
    return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n, const uint3 fastdiv_values) {
    // expects fastdiv_values to contain <mp, L, divisor> in <x, y, z>
    // fastdiv_values.z is unused and optimized away by the compiler.
    // Compute high 32 bits of n * mp
    const uint32_t hi = __umulhi(n, fastdiv_values.x);
    // add n, apply bit shift
    return (hi + n) >> fastdiv_values.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z;
}

// Calculate both division and modulo at once, returns <n/divisor, n%divisor>
static __device__ __forceinline__ uint2 fast_div_modulo(uint32_t n, const uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z;
    return make_uint2(div_val, mod_val);
}

