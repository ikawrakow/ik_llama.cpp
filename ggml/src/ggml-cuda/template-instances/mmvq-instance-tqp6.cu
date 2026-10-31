#include "../iqk_mmvq_templates.cuh"
#include "../tq.cuh"

void mul_mat_vec_tqp6_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_TQP6, VDR_TQ_Q8_1_MMVQ, vec_dot_tq_q8_1<GGML_TYPE_TQP6>>(args, stream);
}

