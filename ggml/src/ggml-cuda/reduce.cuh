#include "common.cuh"

#define CUDA_REDUCE_BLOCK_SIZE 256

void ggml_cuda_op_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
