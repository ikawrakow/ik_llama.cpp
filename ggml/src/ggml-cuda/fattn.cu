//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "fattn-tile-f16.cuh"
#include "fattn-tile-f32.cuh"
#include "fattn-vec-f16-interface.cuh"
#include "fattn-vec-f32-interface.cuh"
#include "fattn-wmma-f16-interface.cuh"
#include "fattn-mma-f16-interface.cuh"
#include "fattn-new-mma.cuh"
#include "fattn.cuh"
#include "convert.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE 256

static inline bool mma_better_than_turing(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) > CC_TURING;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_cuda_set_device(ctx.device);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int32_t precision = KQV->op_params[3];
    const int32_t n_swa = KQV->op_params[4];

    ggml_tensor local_dst, Kl, Vl, Ml;
    if (n_swa > 0) {
        int ntokens = std::max(FATTN_KQ_STRIDE, int(Q->ne[1]));
        int nton = FATTN_KQ_STRIDE*((ntokens + n_swa + FATTN_KQ_STRIDE - 1)/FATTN_KQ_STRIDE);
        int first = K->ne[1] - nton;
        if (first > 0) {
            local_dst = *dst;
            Kl = *K; Kl.ne[1] = nton; Kl.data = (char *)K->data + K->nb[1]*first;
            Vl = *V; Vl.ne[1] = nton; Vl.data = (char *)V->data + V->nb[1]*first;
            Ml = *mask; Ml.ne[0] = nton; Ml.data = (char *)mask->data + mask->nb[0]*first;
            local_dst.src[1] = &Kl;
            local_dst.src[2] = &Vl;
            local_dst.src[3] = &Ml;
            local_dst.op_params[4] = 0;
            dst = &local_dst;
        }
    }

    // On AMD the tile kernels perform poorly, use the vec kernel instead:
    if (cc >= CC_OFFSET_AMD) {
        if (precision == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        }
        return;
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
        }
        return;
    }

    if (!fp16_mma_available(cc)) {
        if (precision == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            }
        }
        return;
    }

    if (new_mma_available(cc) && K->ne[0] == 128 && V->ne[0] == 128 && Q->ne[0] == 128 && Q->ne[1] == 1 &&
            (Q->ne[2] / K->ne[2] == 12 || Q->ne[2] / K->ne[2] == 6 || Q->ne[2] / K->ne[2] == 10)) {
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    // So, not sure why in mainline they thought that for CC_ADA_LOVELACE or when KV cache is not f16 the vector kernels are faster.
    // On my GPU (RTX-4080) MMA is efinitely faster for GQA, both for f16 and for quantized KV cache.
    //const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    //const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        return;
    }

    //
    // It turns out the new new MMA implementation is slower than the
    // previous MMA implementation.
    // Hence, we use it only for DeepSeek with MLA enabled, where head sizes are 576, 512,
    // so no other implementation works.
    //

    if (new_mma_available(cc) && ((K->ne[0] == 576 && V->ne[0] == 512) || (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
        //printf("Using ggml_cuda_flash_attn_ext_mma_new\n");
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    //
    // We need this because I haven't adapted new MMA kernels to work for different
    // K and V head sizes.
    // We also need it if the new MMA is not available
    //
    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
        return;
    }

    // As mentioned above, the new-new MMA is slower then the new MMA.
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
    //ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
}

bool ggml_cuda_fattn_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int32_t precision = KQV->op_params[3];
    const int32_t n_swa = KQV->op_params[4];
    if (cc >= CC_OFFSET_AMD) {
        return precision == GGML_PREC_DEFAULT ? ggml_cuda_fattn_vec_f16_is_supported(ctx, dst)
                                              : ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
        } else {
            return ggml_cuda_fattn_tile_f32_is_supported(ctx, dst);
        }
    }

    if (!fp16_mma_available(cc)) {
        if (precision == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                return ggml_cuda_fattn_vec_f16_is_supported(ctx, dst);
            } else {
                return ggml_cuda_fattn_tile_f16_is_supported(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
            } else {
                return ggml_cuda_fattn_tile_f32_is_supported(ctx, dst);
            }
        }
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    // So, not sure why in mainline they thought that for CC_ADA_LOVELACE or when KV cache is not f16 the vector kernels are faster.
    // On my GPU (RTX-4080) MMA is efinitely faster for GQA, both for f16 and for quantized KV cache.
    //const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    //const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
    }

    if (new_mma_available(cc) && (Q->ne[0] == 576 || (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
        if (Q->ne[0] == 576) {
            int gqa_ratio = Q->ne[2]/K->ne[2];
            return (gqa_ratio % 4) == 0;
        }
        return true;
    }

    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        return ggml_cuda_fattn_wmma_f16_is_supported(ctx, dst);
    }

    return ggml_cuda_fattn_mma_f16_is_supported(ctx, dst);
}
