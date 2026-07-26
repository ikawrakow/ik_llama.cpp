#include "common.cuh"

bool ggml_cuda_latent_attn_is_supported(const ggml_tensor * op);
void ggml_cuda_op_latent_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
