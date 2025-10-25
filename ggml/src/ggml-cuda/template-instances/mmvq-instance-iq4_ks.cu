#include "../iqk_mmvq_templates.cuh"

void mul_mat_vec_iq4_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KS, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq4_ks_q8_1>(args, stream);
}

