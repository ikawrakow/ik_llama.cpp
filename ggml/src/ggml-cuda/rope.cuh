#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256
#define CUDA_ROPE_MAX_GRID_Y 65535

bool ggml_cuda_rope_offset_is_supported(const ggml_tensor * op);
void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fast(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_op_fused_rope_fast(ggml_backend_cuda_context & ctx, ggml_tensor * dst1, ggml_tensor * dst2);

bool ggml_cuda_op_fused_rms_rope_fast(ggml_backend_cuda_context & ctx, ggml_tensor * dst1, ggml_tensor * dst2);

bool ggml_cuda_op_rope_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst1, ggml_tensor * dst2);
