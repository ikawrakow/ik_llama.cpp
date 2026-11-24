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

static __global__ void k_ingkling2_control(int64_t nelem, float * __restrict__ y,
        const float * __restrict__ x0l, const float * __restrict__ x1l, const float * __restrict__ x0r, const float * __restrict__ x1r) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    float exp_l = expf(2*x0l[i]);
    float exp_r = expf(2*x0r[i]);
    y[i] = 2.0f*x1l[i]/(1.0f + exp_l) + x1r[i]*(exp_r - 1)/(exp_r + 1);
}

bool ggml_cuda_op_inkling2_control(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!dst || dst->type != GGML_TYPE_F32 || dst->op != GGML_OP_ADD) return false;
    auto add0 = dst->src[0];
    auto add1 = dst->src[1];
    if (add0->type != GGML_TYPE_F32 || add0->op != GGML_OP_MUL) return false;
    if (add1->type != GGML_TYPE_F32 || add1->op != GGML_OP_MUL) return false;
    auto src1l = add0->src[1];
    auto src1r = add1->src[1];
    if (src1l->type != GGML_TYPE_F32 || src1r->type != GGML_TYPE_F32) return false;
    auto scale = add0->src[0];
    if (scale->op != GGML_OP_SCALE) return false;
    auto fparams = (const float *)scale->op_params;
    if (fparams[0] != -1.0f || fparams[1] != 1.0f) return false;
    auto tanh0 = scale->src[0];
    if (tanh0->op != GGML_OP_UNARY || ggml_unary_op(tanh0->op_params[0]) != GGML_UNARY_OP_TANH) return false;
    auto tanh1 = add1->src[0];
    if (tanh1->op != GGML_OP_UNARY || ggml_unary_op(tanh1->op_params[0]) != GGML_UNARY_OP_TANH) return false;
    auto src0l = tanh0->src[0];
    auto src0r = tanh1->src[0];
    if (src0l->type != GGML_TYPE_F32 || src0r->type != GGML_TYPE_F32) return false;
    if (!ggml_are_same_shape(src0l, src0r)) return false;
    if (!ggml_are_same_shape(src0l, dst)) return false;
    if (!ggml_are_same_shape(src0l, src1l)) return false;
    if (!ggml_are_same_shape(src0r, src1r)) return false;
    // strictly speaking not necessary, just lazy to deal with non-contiguous variants
    if (!ggml_is_contiguous(src0l) || !ggml_is_contiguous(src0r) ||
        !ggml_is_contiguous(src1l) || !ggml_is_contiguous(src1r)) return false;
    int64_t nelem = ggml_nelements(dst);
    int64_t nsm = ggml_cuda_info().devices[ctx.device].nsm;
    if (nsm*WARP_SIZE <= nelem) {
        constexpr int k_block_size = WARP_SIZE;
        int nblock = (nelem + k_block_size - 1)/k_block_size;
        k_ingkling2_control<<<nblock, k_block_size, 0, ctx.stream()>>>(nelem, (float *)dst->data,
                (const float *)src0l->data, (const float *)src1l->data,
                (const float *)src0r->data, (const float *)src1r->data);
    }
    else if (2*nsm*WARP_SIZE <= nelem) {
        constexpr int k_block_size = 2*WARP_SIZE;
        int nblock = (nelem + k_block_size - 1)/k_block_size;
        k_ingkling2_control<<<nblock, k_block_size, 0, ctx.stream()>>>(nelem, (float *)dst->data,
                (const float *)src0l->data, (const float *)src1l->data,
                (const float *)src0r->data, (const float *)src1r->data);
    }
    else if (4*nsm*WARP_SIZE <= nelem) {
        constexpr int k_block_size = 4*WARP_SIZE;
        int nblock = (nelem + k_block_size - 1)/k_block_size;
        k_ingkling2_control<<<nblock, k_block_size, 0, ctx.stream()>>>(nelem, (float *)dst->data,
                (const float *)src0l->data, (const float *)src1l->data,
                (const float *)src0r->data, (const float *)src1r->data);
    }
    else {
        constexpr int k_block_size = 8*WARP_SIZE;
        int nblock = (nelem + k_block_size - 1)/k_block_size;
        k_ingkling2_control<<<nblock, k_block_size, 0, ctx.stream()>>>(nelem, (float *)dst->data,
                (const float *)src0l->data, (const float *)src1l->data,
                (const float *)src0r->data, (const float *)src1r->data);
    }
    return true;
}
