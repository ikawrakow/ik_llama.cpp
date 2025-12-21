#include "common.cuh"

#define CUDA_REDUCE_BLOCK_SIZE 256

void ggml_cuda_op_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fake_cpy(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
