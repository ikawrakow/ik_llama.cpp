#include "../iqk_mmvq_templates.cuh"
#include "../nt-mmvq.cuh"

__device__ __forceinline__ void vec_dot_iq4_nt_q8_1_unused(
    const void * __restrict__, const block_q8_1 * __restrict__, const int &, const int &, float *) {
}

void mul_mat_vec_iq4_nt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    if (nt_mul_mat_vec_1_cuda(args, stream)) return;
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_NT, 8, vec_dot_iq4_nt_q8_1_unused>(args, stream);
}
