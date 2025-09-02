#include "fattn-wmma-f16.cuh"
#include "fattn-wmma-f16-interface.cuh"

void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * V   = dst->src[2];

    if (Q->ne[0] != V->ne[0]) {
        if (!((Q->ne[0] == 192 && V->ne[0] == 128) || (Q->ne[0] == 576 && V->ne[0] == 512))) {
            fprintf(stderr, "======================= %s: Unhandled head size combination %d, %d\n", __func__, (int)Q->ne[0], (int)V->ne[0]);
            GGML_ABORT("fatal error");
        }
    }

    const int32_t precision = KQV->op_params[3];

    if (precision != GGML_PREC_DEFAULT) {
        if (Q->ne[1] <= 32 || Q->ne[0] > 128) {
            constexpr int cols_per_block = 16;
            switch (Q->ne[0]) {
                case 64:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 64, 64, cols_per_block, float>(ctx, dst);
                    break;
                case 80:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 80, 80, cols_per_block, float>(ctx, dst);
                    break;
                case 96:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 96, 96, cols_per_block, float>(ctx, dst);
                    break;
                case 112:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<112, 112, cols_per_block, float>(ctx, dst);
                    break;
                case 128:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<128, 128, cols_per_block, float>(ctx, dst);
                    break;
                case 256:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<256, 256, cols_per_block, float>(ctx, dst);
                    break;
                case 192:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<192, 128, cols_per_block, float>(ctx, dst);
                    break;
                default:
                    fprintf(stderr, "======================= %s: Unhandled head size %d\n", __func__, (int)Q->ne[0]);
                    GGML_ABORT("fatal error");
                    break;
            }
        } else {
            constexpr int cols_per_block = 32;
            switch (Q->ne[0]) {
                case 64:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 64, 64, cols_per_block, float>(ctx, dst);
                    break;
                case 80:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 80, 80, cols_per_block, float>(ctx, dst);
                    break;
                case 96:
                    ggml_cuda_flash_attn_ext_wmma_f16_case< 96, 96, cols_per_block, float>(ctx, dst);
                    break;
                case 112:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<112, 112, cols_per_block, float>(ctx, dst);
                    break;
                case 128:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<128, 128, cols_per_block, float>(ctx, dst);
                    break;
                case 192:
                    ggml_cuda_flash_attn_ext_wmma_f16_case<192, 128, cols_per_block, float>(ctx, dst);
                    break;
                // case 256:
                //     ggml_cuda_flash_attn_ext_wmma_f16_case<128, cols_per_block, float>(ctx, dst);
                //     break;
                default:
                    fprintf(stderr, "======================= %s: Unhandled head size %d\n", __func__, (int)Q->ne[0]);
                    GGML_ABORT("fatal error");
                    break;
            }
        }
        return;
    }

    if (Q->ne[1] <= 8 && Q->ne[0] % WARP_SIZE == 0) {
        constexpr int cols_per_block = 8;
        switch (Q->ne[0]) {
            case 64:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 64, 64, cols_per_block, half>(ctx, dst);
                break;
            case 96:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 96, 96, cols_per_block, half>(ctx, dst);
                break;
            case 128:
                ggml_cuda_flash_attn_ext_wmma_f16_case<128, 128, cols_per_block, half>(ctx, dst);
                break;
            case 192:
                ggml_cuda_flash_attn_ext_wmma_f16_case<192, 128, cols_per_block, half>(ctx, dst);
                break;
            case 256:
                ggml_cuda_flash_attn_ext_wmma_f16_case<256, 256, cols_per_block, half>(ctx, dst);
                break;
            default:
                fprintf(stderr, "======================= %s: Unhandled head size %d\n", __func__, (int)Q->ne[0]);
                GGML_ABORT("fatal error");
                break;
        }
        return;
    }

    if (Q->ne[1] <= 32) {
        constexpr int cols_per_block = 16;
        switch (Q->ne[0]) {
            case 64:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 64, 64, cols_per_block, half>(ctx, dst);
                break;
            case 80:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 80, 80, cols_per_block, half>(ctx, dst);
                break;
            case 96:
                ggml_cuda_flash_attn_ext_wmma_f16_case< 96, 96, cols_per_block, half>(ctx, dst);
                break;
            case 112:
                ggml_cuda_flash_attn_ext_wmma_f16_case<112, 112, cols_per_block, half>(ctx, dst);
                break;
            case 128:
                ggml_cuda_flash_attn_ext_wmma_f16_case<128, 128, cols_per_block, half>(ctx, dst);
                break;
            case 192:
                ggml_cuda_flash_attn_ext_wmma_f16_case<192, 128, cols_per_block, half>(ctx, dst);
                break;
            case 256:
                ggml_cuda_flash_attn_ext_wmma_f16_case<256, 256, cols_per_block, half>(ctx, dst);
                break;
            default:
                fprintf(stderr, "======================= %s: Unhandled head size %d\n", __func__, (int)Q->ne[0]);
                GGML_ABORT("fatal error");
                break;
        }
        return;
    }
    constexpr int cols_per_block = 32;
    switch (Q->ne[0]) {
        case 64:
            ggml_cuda_flash_attn_ext_wmma_f16_case< 64, 64, cols_per_block, half>(ctx, dst);
            break;
        case 80:
            ggml_cuda_flash_attn_ext_wmma_f16_case< 80, 80, cols_per_block, half>(ctx, dst);
            break;
        case 96:
            ggml_cuda_flash_attn_ext_wmma_f16_case< 96, 96, cols_per_block, half>(ctx, dst);
            break;
        case 112:
            ggml_cuda_flash_attn_ext_wmma_f16_case<112, 112, cols_per_block, half>(ctx, dst);
            break;
        case 128:
            ggml_cuda_flash_attn_ext_wmma_f16_case<128, 128, cols_per_block, half>(ctx, dst);
            break;
        case 192:
            ggml_cuda_flash_attn_ext_wmma_f16_case<192, 128, cols_per_block, half>(ctx, dst);
            break;
        case 256:
            ggml_cuda_flash_attn_ext_wmma_f16_case<256, 256, cols_per_block, half>(ctx, dst);
            break;
        default:
            fprintf(stderr, "======================= %s: Unhandled head size %d\n", __func__, (int)Q->ne[0]);
            GGML_ABORT("fatal error");
            break;
    }
}

