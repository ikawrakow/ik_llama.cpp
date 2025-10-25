//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "iqk_mmvq.cuh"
#include "iqk_mmvq_templates.cuh"

void iqk_mul_mat_vec_q(ggml_type type, const mmvq_args & args, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_IQ1_BN:
            mul_mat_vec_iq1_bn_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_BN:
            mul_mat_vec_iq2_bn_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_K:
            mul_mat_vec_iq2_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_K:
            mul_mat_vec_iq3_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_KL:
            mul_mat_vec_iq2_kl_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_KS:
            mul_mat_vec_iq3_ks_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_K:
            mul_mat_vec_iq4_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_KS:
            mul_mat_vec_iq4_ks_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_KSS:
            mul_mat_vec_iq4_kss_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ1_KT:
            mul_mat_vec_iq1_kt_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_KT:
            mul_mat_vec_iq2_kt_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_KT:
            mul_mat_vec_iq3_kt_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_KT:
            mul_mat_vec_iq4_kt_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_KS:
            mul_mat_vec_iq2_ks_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ5_K:
            mul_mat_vec_iq5_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ5_KS:
            mul_mat_vec_iq5_ks_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ6_K:
            mul_mat_vec_iq6_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_K_R4:
            mul_mat_vec_iq2_k_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_K_R4:
            mul_mat_vec_iq3_k_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_K_R4:
            mul_mat_vec_iq4_k_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_KS_R4:
            mul_mat_vec_iq4_ks_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ5_K_R4:
            mul_mat_vec_iq5_k_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ5_KS_R4:
            mul_mat_vec_iq5_ks_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ1_S_R4:
            mul_mat_vec_iq1_s_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ1_M_R4:
            mul_mat_vec_iq1_m_r4_q8_1_cuda(args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
