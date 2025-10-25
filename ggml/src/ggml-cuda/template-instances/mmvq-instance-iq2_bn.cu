#include "../iqk_mmvq_templates.cuh"

void mul_mat_vec_iq2_bn_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_BN, 1, vec_dot_iq2_bn_q8_1>(args, stream);
}

