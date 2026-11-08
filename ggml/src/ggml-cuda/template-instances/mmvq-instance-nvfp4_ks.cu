#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_nvfp4_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const float * dptr = (const float *)vbq;
    const block_nvfp4 * bq4 = (const block_nvfp4 *)(dptr + 1) + kbx;

    const int ib32 = iqs/4;
    const int * q8 = (const int *)bq8_1[ib32].qs;
    const int * q4 = (const int *)bq4->qs + 4*ib32;
    float sumf = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int2 v1 = get_int_from_table_16(q4[2*l+0], kvalues_mxfp4);
        const int2 v2 = get_int_from_table_16(q4[2*l+1], kvalues_mxfp4);
        int sumi = ggml_cuda_dp4a(v1.x, q8[4*l+0], 0);
        sumi = ggml_cuda_dp4a(v2.x, q8[4*l+1], sumi);
        sumi = ggml_cuda_dp4a(v1.y, q8[4*l+2], sumi);
        sumi = ggml_cuda_dp4a(v2.y, q8[4*l+3], sumi);
        sumf += ggml_cuda_ue4m3_to_fp32(bq4->d[2*ib32+l]) * sumi;
    }
    *result += dptr[0] * 0.5f * __low2float(bq8_1[ib32].ds) * sumf;
}

void mul_mat_vec_nvfp4_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_NVFP4_KS, VDR_NVFP4_KS_Q8_1_MMVQ, vec_dot_nvfp4_ks_q8_1>(args, stream);
}
