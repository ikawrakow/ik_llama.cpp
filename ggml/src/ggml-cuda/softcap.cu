#include "softcap.cuh"

static __global__ void softcap_f32(const float * x, float * dst, const float * scales, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = scales[1] * tanh(scales[0]*x[i]);
}

static void softcap_f32_cuda(const float * x, float * dst, const float * scales, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    softcap_f32<<<num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream>>>(x, dst, scales, k);
}

void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scales[2];
    memcpy(scales, dst->op_params, sizeof(scales));

    softcap_f32_cuda(src0_d, dst_d, scales, ggml_nelements(src0), stream);
}
