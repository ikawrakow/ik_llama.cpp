#include "common.cuh"

void ggml_cuda_flash_attn_ext_tile_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_fattn_tile_f32_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
