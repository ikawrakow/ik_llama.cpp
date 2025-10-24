//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.cuh"

#define CUDA_MULTI_ADD_BLOCK_SIZE 256

void ggml_cuda_op_multi_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_mul_multi_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
