#include "common.cuh"

void ggml_cuda_op_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
