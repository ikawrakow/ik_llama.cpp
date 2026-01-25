#include "fattn-mma-f16.cuh"
#include "fattn-mma-f16-interface.cuh"

static __global__ void k_repack_q(int nelements, int ne0, int ne0_1, const float * src, float * dst1, float * dst2) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nelements) {
        return;
    }
    int row = i / ne0;
    int i0  = i % ne0;
    if (i0 < ne0_1) {
        dst1[row*ne0_1 + i0] = src[i];
    } else {
        dst2[row*(ne0 - ne0_1) + i0 - ne0_1] = src[i];
    }
}

static __global__ void k_pack_fa(const float * x, const float * y, float * dst, int ne0, int ne00, int nelem) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nelem) {
        return;
    }

    int row = i / ne0;
    int i0  = i % ne0;

    if (i0 < ne00) {
        dst[row*ne0 + i0] = x[row*ne00 + i0];
    } else {
        dst[row*ne0 + i0] = y[row*(ne0 - ne00) + i0 - ne00];
    }
}


static void repack_q(const ggml_tensor * q, float * dst, int nhead1, int nhead2, int nek2, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    GGML_ASSERT((nhead1 + nhead2)*nek2 == q->ne[2]);
    int ne0 = q->ne[0] * (nhead1 + nhead2); // we know that Q is contiguous along the second dimension
    int ne0_1 = q->ne[0] * nhead1;
    int nelements = ne0 * q->ne[1] * q->ne[3] * nek2;
    int nblocks = (nelements + kBlockSize - 1)/kBlockSize;
    auto dst1 = dst;
    auto dst2 = dst + ne0_1 * q->ne[1] * q->ne[3] * nek2;
    k_repack_q<<<nblocks, kBlockSize, 0, stream>>>(nelements, ne0, ne0_1, (const float *)q->data, dst1, dst2);
}

static void pack_glm45_result(const ggml_tensor * fa1, const ggml_tensor * fa2, ggml_tensor * dst, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    GGML_ASSERT(dst->ne[1] % 12 == 0);
    GGML_ASSERT(fa1->ne[0] == fa2->ne[0] && fa1->ne[0] == dst->ne[0]);
    GGML_ASSERT(fa1->ne[1] + fa2->ne[1] == dst->ne[1]);
    GGML_ASSERT(fa1->ne[2] == fa2->ne[2] && fa1->ne[2] == dst->ne[2]);
    GGML_ASSERT(fa1->ne[3] == fa2->ne[3] && fa1->ne[3] == dst->ne[3]);
    GGML_ASSERT(fa1->type == GGML_TYPE_F32 && fa2->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    int ne0  = dst->ne[0] * 12;
    int ne00 = dst->ne[0] *  8;
    int nelem = ne0 * dst->ne[1]/12 * dst->ne[2] * dst->ne[3];
    int nblocks = (nelem + kBlockSize - 1)/kBlockSize;
    k_pack_fa<<<nblocks, kBlockSize, 0, stream>>>((const float *)fa1->data, (const float *)fa2->data, (float *)dst->data, ne0, ne00, nelem);
}

static inline ggml_tensor get_float_tensor(int ne0, int ne1, int ne2, int ne3) {
    return {GGML_TYPE_F32, {}, nullptr, {ne0, ne1, ne2, ne3},
        {sizeof(float), ne0*sizeof(float), ne0*ne1*sizeof(float), ne0*ne1*ne2*sizeof(float)},
        GGML_OP_NONE, {}, 0, nullptr, {}, nullptr, 0, nullptr, {}, nullptr};
}
static inline void permute_21(ggml_tensor & t) {
    auto tmp1 = t.ne[1]; t.ne[1] = t.ne[2]; t.ne[2] = tmp1;
    auto tmp2 = t.nb[1]; t.nb[1] = t.nb[2]; t.nb[2] = tmp2;
}

static void glm45_flash_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto Q = dst->src[0];
    auto K = dst->src[1];
    auto V = dst->src[2];
    GGML_ASSERT(Q->ne[2] / K->ne[2] == 12);

    ggml_cuda_pool_alloc<float> q_data(ctx.pool(), ggml_nelements(Q));
    ggml_cuda_pool_alloc<float> dst_data(ctx.pool(), ggml_nelements(dst));
    ggml_cuda_pool_alloc<half>  k_data(ctx.pool());
    ggml_cuda_pool_alloc<half>  v_data(ctx.pool());

    repack_q(Q, q_data.get(), 8, 4, K->ne[2], ctx.stream());

    auto local_Q1 = get_float_tensor(Q->ne[0], 8*K->ne[2], Q->ne[1], Q->ne[3]);
    permute_21(local_Q1);
    local_Q1.data = q_data.get();

    auto local_Q2 = get_float_tensor(Q->ne[0], 4*K->ne[2], Q->ne[1], Q->ne[3]);
    permute_21(local_Q2);
    local_Q2.data = q_data.get() + ggml_nelements(&local_Q1);

    GGML_ASSERT(ggml_nelements(Q) == ggml_nelements(&local_Q1) + ggml_nelements(&local_Q2));

    auto local_K = *K;
    auto local_V = *V;

    if (K->type != GGML_TYPE_F16) {
        auto nelem = ggml_nelements(K);
        k_data.alloc(nelem);
        auto to_fp_16 = ggml_get_to_fp16_cuda(K->type);
        to_fp_16(K->data, k_data.get(), 1, nelem, ctx.stream());
        local_K.type = GGML_TYPE_F16;
        local_K.data = k_data.get();
        auto ts = ggml_type_size(K->type);
        auto bs = ggml_blck_size(K->type);
        local_K.nb[0] = sizeof(half);
        local_K.nb[1] = sizeof(half)*bs * local_K.nb[1]/ts;
        local_K.nb[2] = sizeof(half)*bs * local_K.nb[2]/ts;
        local_K.nb[3] = sizeof(half)*bs * local_K.nb[3]/ts;
    }
    if (V->type != GGML_TYPE_F16) {
        auto nelem = ggml_nelements(V);
        v_data.alloc(nelem);
        auto to_fp_16 = ggml_get_to_fp16_cuda(V->type);
        to_fp_16(V->data, v_data.get(), 1, nelem, ctx.stream());
        local_V.type = GGML_TYPE_F16;
        local_V.data = v_data.get();
        auto ts = ggml_type_size(V->type);
        auto bs = ggml_blck_size(V->type);
        local_V.nb[0] = sizeof(half);
        local_V.nb[1] = sizeof(half)*bs * local_V.nb[1]/ts;
        local_V.nb[2] = sizeof(half)*bs * local_V.nb[2]/ts;
        local_V.nb[3] = sizeof(half)*bs * local_V.nb[3]/ts;
    }

    constexpr int n_op_params = GGML_MAX_OP_PARAMS / sizeof(int);

    auto fa1 = get_float_tensor(V->ne[0], local_Q1.ne[2], local_Q1.ne[1], local_Q1.ne[3]);
    fa1.data = dst_data.get();
    fa1.op   = GGML_OP_FLASH_ATTN_EXT;
    fa1.src[0] = &local_Q1;
    fa1.src[1] = &local_K;
    fa1.src[2] = &local_V;
    for (int i = 3; i < GGML_MAX_SRC; ++i) fa1.src[i] = dst->src[i];
    for (int i = 0; i < n_op_params; ++i) fa1.op_params[i] = dst->op_params[i];

    auto fa2 = get_float_tensor(V->ne[0], local_Q2.ne[2], local_Q2.ne[1], local_Q2.ne[3]);
    fa2.data = dst_data.get() + ggml_nelements(&fa1);
    fa2.op   = GGML_OP_FLASH_ATTN_EXT;
    fa2.src[0] = &local_Q2;
    fa2.src[1] = &local_K;
    fa2.src[2] = &local_V;
    for (int i = 3; i < GGML_MAX_SRC; ++i) fa2.src[i] = dst->src[i];
    for (int i = 0; i < n_op_params; ++i) fa2.op_params[i] = dst->op_params[i];

    ggml_cuda_flash_attn_ext_mma_f16(ctx, &fa1);
    ggml_cuda_flash_attn_ext_mma_f16(ctx, &fa2);
    pack_glm45_result(&fa1, &fa2, dst, ctx.stream());
}

template <int D, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    if (Q->ne[1] <= 8/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<D, 8/ncols2, ncols2>(ctx, dst);
        return;
    }

    if (Q->ne[1] <= 16/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<D, 16/ncols2, ncols2>(ctx, dst);
        return;
    }

    if (Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<D, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<D, 64/ncols2, ncols2>(ctx, dst);
}

template <int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_hs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    switch (Q->ne[0]) {
        case 64:
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1< 64, ncols2>(ctx, dst);
            break;
        case 80:
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1< 80, ncols2>(ctx, dst);
            break;
        case 96:
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1< 96, ncols2>(ctx, dst);
            break;
        case 112:
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<112, ncols2>(ctx, dst);
            break;
        case 128:
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<128, ncols2>(ctx, dst);
            break;
        case 192:
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, ncols2>(ctx, dst);
            break;
        case 256:
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<256, ncols2>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    const bool use_gqa_opt = mask && max_bias == 0.0f;

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (gqa_ratio == 12 && Q->ne[1] == 1 && K->ne[1]*K->ne[2] >= 65536) {
        // This is a hack to improve GLM-4.5/4.6/4.7/AIR TG performance
        glm45_flash_attention(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_hs<8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_hs<4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_hs<2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_switch_hs<1>(ctx, dst);
}

bool ggml_cuda_fattn_mma_f16_is_supported([[maybe_unused]] ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    auto K = dst->src[1];
    auto V = dst->src[1];
    if (K->ne[0] != V->ne[0]) return false;
    return K->ne[0] == 64 || K->ne[0] == 80 || K->ne[0] == 96 || K->ne[0] == 112 || K->ne[0] == 128 || K->ne[0] == 192 || K->ne[0] == 256;
}
