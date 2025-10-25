#include "../iqk_mmvq_templates.cuh"

void mul_mat_vec_iq5_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ5_K_R4, 2, vec_dot_iq5_k_r4_q8_1, 4>(args, stream);
}

