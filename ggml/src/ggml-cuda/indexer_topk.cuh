//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
#include "common.cuh"

void ggml_cuda_op_indexer_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_indexer_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
