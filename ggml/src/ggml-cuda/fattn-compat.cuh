#pragma once

#define FLASH_ATTN_AVAILABLE

#include "common.cuh"

#if defined GGML_USE_HIPBLAS && !defined GGML_USE_HIP
#define GGML_USE_HIP
#endif

static bool amd_wmma_available(const int cc) {
    return (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3(cc));
}

static bool volta_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == CC_VOLTA;
}

static bool turing_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= CC_TURING;
}

static bool ampere_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= CC_AMPERE;
}

static bool blackwell_mma_available(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= CC_BLACKWELL &&
           ggml_cuda_highest_compiled_arch(cc) < CC_RUBIN;
}

// Maximum number of bytes that can be copied in a single instruction.
static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#ifdef GGML_USE_HIP
    return 16;
#else
#if __CUDA_ARCH__ >= CC_VOLTA
    return 16;
#else
    return 8;
#endif // __CUDA_ARCH__ >= CC_VOLTA
#endif // GGML_USE_HIP
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD.
// Important: do not use this function if dst and src both point at registers.
//     Due to the strict aliasing rule the compiler can do incorrect optimizations if src and dst have different types.
//     The function is intended for copies between registers and SRAM/VRAM to make the compiler emit the right instructions.
//     If dst and src point at different address spaces then they are guaranteed to not be aliased.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    static_assert(
        nbytes <= ggml_cuda_get_max_cpy_bytes() || alignment == 0,
        "You are misusing the alignment parameter for ggml_cuda_memcpy_1. "
        "The intent is for the parameter is only as a workaround if either one of the pointers is not properly aligned. "
        "If you use it to do more bytes per copy than ggml_cuda_max_cpy_bytes() the reads and writes may not be coalesced. "
        "Call ggml_cuda_memcpy_1 in a loop instead.");
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float v, const float u) {
    acc += v*u;
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float2 v, const float2 u) {
    acc += v.x*u.x;
    acc += v.y*u.y;
}

#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))
#define V_DOT2_F32_F16_AVAILABLE
#endif // defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(__gfx906__) || defined(CDNA))

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const half2 v, const half2 u) {
#ifdef V_DOT2_F32_F16_AVAILABLE
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
#else
#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(v*u);
    acc += tmp.x + tmp.y;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    acc += tmpv.x * tmpu.x;
    acc += tmpv.y * tmpu.y;
#endif // FAST_FP16_AVAILABLE
#endif // V_DOT2_F32_F16_AVAILABLE
}

static __device__ __forceinline__ void ggml_cuda_mad(half2 & acc, const half2 v, const half2 u) {
#ifdef FAST_FP16_AVAILABLE
    acc += v*u;
#else
    const float2 tmpv = __half22float2(v);
    const float2 tmpu = __half22float2(u);
    float2 tmpacc = __half22float2(acc);
    tmpacc.x += tmpv.x * tmpu.x;
    tmpacc.y += tmpv.y * tmpu.y;
    acc = make_half2(tmpacc.x, tmpacc.y);
#endif // FAST_FP16_AVAILABLE
}

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

template<typename... Args>
__host__ __device__ constexpr inline void ggml_unused_vars_impl(Args&&...) noexcept {}
#define GGML_UNUSED_VARS(...) ggml_unused_vars_impl(__VA_ARGS__)

#if defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)
#define AMD_MFMA_AVAILABLE
#endif // defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)

#if defined(GGML_USE_HIP) && (defined(RDNA4) || defined(RDNA3))
#define AMD_WMMA_AVAILABLE
#endif // defined(GGML_USE_HIP) && defined(RDNA4)

// The Volta instructions are in principle available on Turing or newer but they are effectively unusable:
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ == CC_VOLTA
#define VOLTA_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ == CC_VOLTA

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= CC_TURING
#define TURING_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= CC_TURING

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= CC_AMPERE

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= CC_BLACKWELL && __CUDA_ARCH__ < CC_RUBIN
#    define BLACKWELL_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= CC_BLACKWELL

#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= CC_AMPERE
#define CP_ASYNC_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= CC_AMPERE

#if defined(TURING_MMA_AVAILABLE)
#define LDMATRIX_TRANS_AVAILABLE
#endif // defined(TURING_MMA_AVAILABLE)

template <int n>
struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

static inline ggml_prec ggml_flash_attn_ext_get_prec(const ggml_tensor * t) {
    return ggml_prec(t->op_params[3]);
}
