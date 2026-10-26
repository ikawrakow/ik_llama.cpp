#include "common.cuh"

// K2E trellis embedding lookup (row rings, 256-groups, signs)
void ggml_cuda_op_t16_embed_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
