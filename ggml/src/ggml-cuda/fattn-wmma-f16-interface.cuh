#pragma once

#include "common.cuh"

void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_fattn_wmma_f16_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
