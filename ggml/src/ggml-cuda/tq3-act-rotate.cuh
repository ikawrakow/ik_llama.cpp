//
// TurboQuant TQ3 activation WHT rotation (Phase 2c)
// Port from turbo-tan/llama.cpp-tq3 ggml-cuda/tq3-native.{cu,cuh}
//
// Weights are stored with forward WHT + signs applied. At inference
// time the activations must be rotated with the same forward WHT
// before being dot-producted against the weight centroids, because
// the vec_dot_tq3_*_q8_1 kernel computes a naive dot in the rotated
// basis (the inverse WHT is NOT applied on the weight side in MMVQ).
//

#pragma once

#include "common.cuh"

static __device__ __forceinline__ float ggml_cuda_tq3_sign(const int i) {
    // Sign pattern matching TQ3_0_SIGNS[32] — golden-ratio hash,
    // bit 31 of (i * 0x9E3779B9) → -1 if set else +1.
    return ((((unsigned) i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}

void ggml_cuda_tq3_rotate_act(float * x, int64_t n, cudaStream_t stream);
