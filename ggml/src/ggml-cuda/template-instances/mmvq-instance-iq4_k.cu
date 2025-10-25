#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq4_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq4_k * bq4 = (const block_iq4_k *) vbq + kbx;
    const uint8_t * all_values = (const uint8_t *)iq4k_values;

    // iqs is 0...28
    const int ib32 = iqs/4;
    // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint16_t * q4 = (const uint16_t *)bq4->qs + 8*ib32;
    const uint16_t extra = bq4->extra >> 2*ib32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        const uint32_t aux32 = q4[2*j+0] | (q4[2*j+1] << 16);
        get_int_from_table_16_shift(aux32, extra, all_values, v1, v2);
        sumi1 = ggml_cuda_dp4a(v1, q8[j+0], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8[j+4], sumi2);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1[ib32].ds);
    const uint8_t sh = bq4->scales_h[ib32/2] >> 4*(ib32%2);
    const int ls1 = ((bq4->scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32;
    const int ls2 = ((bq4->scales_l[ib32] >>  4) | ((sh << 2) & 0x30)) - 32;
    *result += d * (sumi1 * ls1 + sumi2 * ls2);
}

void mul_mat_vec_iq4_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_K, VDR_IQ4_K_Q8_1_MMVQ, vec_dot_iq4_k_q8_1>(args, stream);
}

