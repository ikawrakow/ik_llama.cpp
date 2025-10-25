#include "../mmvq-templates.cuh"

void mul_mat_vec_iq3_s_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    mul_mat_vec_q_cuda<GGML_TYPE_IQ3_S>(args, stream);
}

