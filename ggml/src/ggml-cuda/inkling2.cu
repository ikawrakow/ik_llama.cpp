#include "inkling2.cuh"

static __global__ void k_ingkling2_clip(const int ne0, const float * __restrict__ src, float * __restrict__ dst,
                                       size_t nb01, size_t nb02, size_t nb03,
                                       size_t nb1,  size_t nb2,  size_t nb3,
                                       float scale, float eps, float limit, float clip_min, float clip_max) {
    src += blockIdx.x*nb02 + blockIdx.y*nb01 + blockIdx.z*nb03;
    dst += blockIdx.x*nb2  + blockIdx.y*nb1  + blockIdx.z*nb3;

    float sum2 = 0;
    for (int j = threadIdx.x; j < ne0; j += blockDim.x) sum2 += src[j]*src[j];
    sum2 = warp_reduce_sum(sum2);
    float norm = limit/(scale*sqrtf(sum2) + eps);
    norm = fmaxf(clip_min, fminf(clip_max, norm));
    for (int j = threadIdx.x; j < ne0; j += blockDim.x) dst[j] = norm*src[j];
}

void ggml_cuda_op_inkling2_clip(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    auto src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

    const float * params = (const float *)dst->op_params;

    dim3 grid(src0->ne[2], src0->ne[1], src0->ne[3]);
    k_ingkling2_clip<<<grid, WARP_SIZE, 0, ctx.stream()>>>(src0->ne[0], (const float *)src0->data, (float *)dst->data,
            src0->nb[1]/sizeof(float), src0->nb[2]/sizeof(float), src0->nb[3]/sizeof(float),
             dst->nb[1]/sizeof(float),  dst->nb[2]/sizeof(float),  dst->nb[3]/sizeof(float),
             params[0], params[1], params[2], params[3], params[4]);
}
