#include "multiadd.cuh"

static __global__ void multi_add_f32(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, const char * src0, char * dst) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;
    int64_t k = ne0*ne1;
    if (i >= k) {
        return;
    }
    int i1 = i / ne0;
    int i0 = i % ne0;
    float * result = (float *)(dst + i1*nb1);
    const float * s = (const float *)(src0 + i1*nb01) + i0;
    if (nused == 1) {
        result[i0] = s[0];
    } else {
        float sum = s[0] + s[ne0];
        for (int j = 2; j < nused; ++j) sum += s[j*ne0];
        result[i0] = sum;
    }
}

static void multi_add_f32_cuda(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, const char * src0, char * dst, cudaStream_t stream) {
    int64_t k = ne0 * ne1;
    const int num_blocks = (k + CUDA_MULTI_ADD_BLOCK_SIZE - 1) / CUDA_MULTI_ADD_BLOCK_SIZE;
    multi_add_f32<<<num_blocks, CUDA_MULTI_ADD_BLOCK_SIZE, 0, stream>>>(nused, ne0, ne1, nb1, nb01, src0, dst);
}

void ggml_cuda_op_multi_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[2] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    int nused = dst->op_params[0];
    GGML_ASSERT(nused >= 1);
    const char * src0 = (const char *)dst->src[0]->data;
    cudaStream_t stream = ctx.stream();
    multi_add_f32_cuda(nused, dst->ne[0], dst->ne[1], dst->nb[1], dst->src[0]->nb[1], src0, (char *)dst->data, stream);
}


static __global__ void mul_multi_add_f32(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, int64_t nb02, int64_t nb11, int64_t nb12, const char * src0, const char * src1, char * dst) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;
    int64_t k = ne0*ne1;
    if (i >= k) {
        return;
    }
    int i1 = i / ne0;
    int i0 = i % ne0;
    float * result = (float *)(dst + i1*nb1);

    auto c0 = src0 + i1*nb02;
    auto c1 = src1 + i1*nb12;

    float sum = 0;
    for (int j = 0; j < nused; ++j) {
        auto x0 = (const float *)c0;
        auto x1 = (const float *)c1;
        sum += x0[i0] * x1[0];
        c0 += nb01;
        c1 += nb11;
    }
    result[i0] = sum;
}

static void mul_multi_add_f32_cuda(int nused, int64_t ne0, int64_t ne1, int64_t nb1, int64_t nb01, int64_t nb02, int64_t nb11, int64_t nb12,
        const char * src0, const char * src1, char * dst, cudaStream_t stream) {
    int64_t k = ne0 * ne1;
    const int num_blocks = (k + CUDA_MULTI_ADD_BLOCK_SIZE - 1) / CUDA_MULTI_ADD_BLOCK_SIZE;
    mul_multi_add_f32<<<num_blocks, CUDA_MULTI_ADD_BLOCK_SIZE, 0, stream>>>(nused, ne0, ne1, nb1, nb01, nb02, nb11, nb12, src0, src1, dst);
}

void ggml_cuda_op_mul_multi_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto src0 = dst->src[0];
    auto src1 = dst->src[1];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] ==  dst->ne[0]);
    GGML_ASSERT(src0->ne[2] ==  dst->ne[1]);
    GGML_ASSERT(src0->ne[1] == src1->ne[1]);
    GGML_ASSERT(src0->ne[2] == src1->ne[2]);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]);
    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[0] == 1);

    mul_multi_add_f32_cuda(src0->ne[1], dst->ne[0], dst->ne[1], dst->nb[1], src0->nb[1], src0->nb[2], src1->nb[1], src1->nb[2],
            (const char *)src0->data, (const char *)src1->data, (char *)dst->data, ctx.stream());
}
