//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.cuh"

#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_MULTI_ADD_BLOCK_SIZE 256

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_mul_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_fused_mul_unary(ggml_backend_cuda_context & ctx, ggml_unary_op op,
        int64_t nelements, const float * x, const float * y, float * z);

void ggml_cuda_op_multi_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_swiglu_oai_cuda_f32(const float * x, const float * g, float * dst, const int64_t k, const int64_t n,
        const int64_t o0, const int64_t o1, const float alpha, const float limit, cudaStream_t stream);

