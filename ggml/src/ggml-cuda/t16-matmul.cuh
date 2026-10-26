#include "common.cuh"

// K2 (t16) trellis matmul: pure tile matmul, no transforms/scales
void ggml_cuda_op_t16_matmul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_t16_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
