#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq1_m_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const half * dptr = (const half *)vbq;
    const block_iq1_m_r4 * bq1 = (const block_iq1_m_r4 *)(dptr + 4) + kbx;

    // iqs is 0 or 2
    const float d8 = __low2float(bq8_1->ds);
    const int32_t  * q8 = (const int *)bq8_1->qs;

    int32_t grid32[2];
    const int * igrid = (const int *)grid32;

    int minus1 = ggml_cuda_dp4a(0x01010101, q8[4*(iqs/2)+0], ggml_cuda_dp4a(0x01010101, q8[4*(iqs/2)+1], 0));
    int minus2 = ggml_cuda_dp4a(0x01010101, q8[4*(iqs/2)+2], ggml_cuda_dp4a(0x01010101, q8[4*(iqs/2)+3], 0));

    for (int i = 0; i < 4; ++i) {
        float dl = __half2float(dptr[i])*((bq1->scales[i] >> 4*(iqs/2)) & 0xf) * d8;
        float ml1 = dl * (bq1->qh[4*(iqs/2)+i] & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA);
        float ml2 = dl * (bq1->qh[4*(iqs/2)+i] & 0x80 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA);
        grid32[0] = iq1s_grid_gpu[bq1->qs[4*iqs+i] | ((bq1->qh[4*(iqs/2)+i] & 0x07) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        int sumi = ggml_cuda_dp4a(igrid[0], q8[4*(iqs/2)+0], ggml_cuda_dp4a(igrid[1], q8[4*(iqs/2)+1], 0));
        grid32[0] = iq1s_grid_gpu[bq1->qs[4*iqs+i+4] | ((bq1->qh[4*(iqs/2)+i] & 0x70) << 4)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        sumi = ggml_cuda_dp4a(igrid[0], q8[4*(iqs/2)+2], ggml_cuda_dp4a(igrid[1], q8[4*(iqs/2)+3], sumi));
        result[i] += dl * sumi + ml1 * minus1 + ml2*minus2;
    }
}

void mul_mat_vec_iq1_m_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ1_M_R4, 2, vec_dot_iq1_m_r4_q8_1, 4>(args, stream);
}

