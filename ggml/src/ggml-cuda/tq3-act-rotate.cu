#include "tq3-act-rotate.cuh"

// Forward WHT rotation with per-lane sign flip + normalization by 1/sqrt(32).
//
// Optimization notes:
//   * Out-of-place variant (src != dst) fuses the cudaMemcpyAsync that
//     previously copied src1->data into src1_tq3_rot.get() before calling the
//     in-place kernel. One global-memory read-write round-trip saved per call.
//   * Branchless butterfly: the XOR (lane & step) pattern that used to gate
//     `val = other ± val` is replaced by a sign-bit flip on `val` (bitwise
//     XOR on the float32 mantissa), followed by an unconditional add. nvcc
//     folds this into a single FMA-style ISEL-free add.
//   * 1/sqrt(32) is a compile-time constant, not a runtime sqrtf() call.
static __device__ __forceinline__ float tq3_wht32_butterfly(float val, int lane) {
    // Hadamard butterfly: for step = 1, 2, 4, 8, 16,
    //    val = val_xor_partner + ((lane & step) ? -val : val)
    // Re-expressed with a branchless sign flip via XOR on the IEEE-754 sign bit.
    #pragma unroll
    for (int step = 1; step < QK_TQ3_0; step <<= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, step, 32);
        const uint32_t flip = (lane & step) ? 0x80000000u : 0u;
        val = other + __uint_as_float(__float_as_uint(val) ^ flip);
    }
    return val;
}

static __global__ void tq3_rotate_act_kernel_inplace(float * __restrict__ x, int64_t n) {
    const int64_t base = (int64_t)blockIdx.x * QK_TQ3_0;
    if (base >= n) return;
    const int lane = threadIdx.x;
    float val = x[base + lane] * ggml_cuda_tq3_sign(lane);
    val = tq3_wht32_butterfly(val, lane);
    constexpr float inv_sqrt_32 = 0.17677669529663688f; // 1 / sqrt(32)
    x[base + lane] = val * inv_sqrt_32;
}

static __global__ void tq3_rotate_act_kernel_copy(
        const float * __restrict__ src, float * __restrict__ dst, int64_t n) {
    const int64_t base = (int64_t)blockIdx.x * QK_TQ3_0;
    if (base >= n) return;
    const int lane = threadIdx.x;
    float val = src[base + lane] * ggml_cuda_tq3_sign(lane);
    val = tq3_wht32_butterfly(val, lane);
    constexpr float inv_sqrt_32 = 0.17677669529663688f; // 1 / sqrt(32)
    dst[base + lane] = val * inv_sqrt_32;
}

void ggml_cuda_tq3_rotate_act(float * x, int64_t n, cudaStream_t stream) {
    tq3_rotate_act_kernel_inplace<<<n / QK_TQ3_0, QK_TQ3_0, 0, stream>>>(x, n);
}

void ggml_cuda_tq3_rotate_act_copy(const float * src, float * dst, int64_t n, cudaStream_t stream) {
    tq3_rotate_act_kernel_copy<<<n / QK_TQ3_0, QK_TQ3_0, 0, stream>>>(src, dst, n);
}
