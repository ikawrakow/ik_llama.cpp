#include "tq3-act-rotate.cuh"

static __global__ void tq3_rotate_act_kernel(float * __restrict__ x, int64_t n) {
    const int64_t base = (int64_t)blockIdx.x * QK_TQ3_0;
    if (base >= n) return;
    const int lane = threadIdx.x;
    float val = x[base + lane] * ggml_cuda_tq3_sign(lane);
    for (int step = 1; step < QK_TQ3_0; step <<= 1) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, step, 32);
        val = (lane & step) ? (other - val) : (other + val);
    }
    x[base + lane] = val / sqrtf((float)QK_TQ3_0);
}

void ggml_cuda_tq3_rotate_act(float * x, int64_t n, cudaStream_t stream) {
    tq3_rotate_act_kernel<<<n / QK_TQ3_0, QK_TQ3_0, 0, stream>>>(x, n);
}
