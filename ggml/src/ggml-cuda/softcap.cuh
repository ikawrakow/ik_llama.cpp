//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.cuh"

#define CUDA_SOFTCAP_BLOCK_SIZE 256

void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
