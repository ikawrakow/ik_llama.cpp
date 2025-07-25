//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "mmq.cuh"

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t nb01 = src0->nb[1];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();
    const int compute_capability = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const mmq_args args = {src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, nb01, src1_padded_row_size, src1_ncols, ne11, nrows_dst};

    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_0:
            mul_mat_q_case<GGML_TYPE_Q6_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S_R4:
            mul_mat_q_case<GGML_TYPE_IQ1_S_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_KL:
            mul_mat_q_case<GGML_TYPE_IQ2_KL>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_KS:
            mul_mat_q_case<GGML_TYPE_IQ3_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KSS:
            mul_mat_q_case<GGML_TYPE_IQ4_KSS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KS:
            mul_mat_q_case<GGML_TYPE_IQ4_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KS_R4:
            mul_mat_q_case<GGML_TYPE_IQ4_KS_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KT:
            mul_mat_q_case<GGML_TYPE_IQ4_KT>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_KT:
            mul_mat_q_case<GGML_TYPE_IQ1_KT>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_KT:
            mul_mat_q_case<GGML_TYPE_IQ2_KT>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_KT:
            mul_mat_q_case<GGML_TYPE_IQ3_KT>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_KS:
            mul_mat_q_case<GGML_TYPE_IQ5_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_KS_R4:
            mul_mat_q_case<GGML_TYPE_IQ5_KS_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_KS:
            mul_mat_q_case<GGML_TYPE_IQ2_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_K:
            mul_mat_q_case<GGML_TYPE_IQ2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_K:
            mul_mat_q_case<GGML_TYPE_IQ3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_K:
            mul_mat_q_case<GGML_TYPE_IQ4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_K:
            mul_mat_q_case<GGML_TYPE_IQ5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ6_K:
            mul_mat_q_case<GGML_TYPE_IQ6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_K_R4:
            mul_mat_q_case<GGML_TYPE_IQ2_K_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_K_R4:
            mul_mat_q_case<GGML_TYPE_IQ3_K_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_K_R4:
            mul_mat_q_case<GGML_TYPE_IQ4_K_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_K_R4:
            mul_mat_q_case<GGML_TYPE_IQ5_K_R4>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q2_K: mmq_supported = ne11 < 384; break;
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
            mmq_supported = ne11 < 1536;
            break;
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_K_R4:
            mmq_supported = ne11 < 2048;
            break;
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ5_K_R4:
            mmq_supported = ne11 < 1024;
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (int8_mma_available(cc)) {
        return true;
    }
    if (type == GGML_TYPE_IQ1_S_R4) {
        return false;
    }

    if (cc < MIN_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (cc < CC_OFFSET_AMD) {
        return cc < CC_VOLTA || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    return cc < CC_RDNA3 || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
