#include "common.cuh"

void ggml_cuda_op_fma(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_op_fma_sus(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
