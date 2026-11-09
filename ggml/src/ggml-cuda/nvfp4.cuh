#pragma once

#include "common.cuh"

bool ggml_cuda_nvfp4_available(int cc);
bool ggml_cuda_can_mul_mat_nvfp4(const ggml_tensor * src0, const ggml_tensor * src1, int cc);
bool ggml_cuda_nvfp4_same_input_scale(const ggml_tensor * a, const ggml_tensor * b);
int ggml_cuda_mul_mat_nvfp4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, const ggml_cgraph * cgraph, int node_n);
void ggml_cuda_mul_mat_nvfp4_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * src1, float * dst_up, float * dst_gate);
void ggml_cuda_up_gate_nvfp4(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * src1, ggml_tensor * dst, float limit);

bool ggml_cuda_can_mul_mat_nvfp4_id(const ggml_tensor * src0, const ggml_tensor * src1, int cc);
struct nvfp4_input {
    int k, n;
    ggml_cuda_pool_alloc<uint8_t> q, s;
    ggml_cuda_pool_alloc<float> r;

    nvfp4_input(ggml_backend_cuda_context & ctx);
    nvfp4_input(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, float scale);
    void quantize(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, float scale);
    void mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, float * dst);
};

void ggml_cuda_gemm_nvfp4_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, const int32_t * ids_src, const int32_t * ids_dst, const int32_t * bounds,
        const nvfp4_input & input, bool fused = false, const ggml_tensor * gate = nullptr, float alpha = 0.0f, float limit = 0.0f);
