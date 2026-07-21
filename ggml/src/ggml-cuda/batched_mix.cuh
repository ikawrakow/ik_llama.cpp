#pragma once

#include "common.cuh"

bool ggml_cuda_batched_mix_is_supported(const ggml_tensor * op);
void ggml_cuda_op_batched_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
