#include "fattn-mma-f16.cuh"
#include "fattn-mma-f16-interface.cuh"

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
