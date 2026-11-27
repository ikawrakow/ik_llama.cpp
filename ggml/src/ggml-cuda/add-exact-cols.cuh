#pragma once
#include "common.cuh"

void ggml_cuda_op_add_exact_cols(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
