#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq6_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq6_k * bq6 = (const block_iq6_k *) vbq + kbx;
    const uint8_t * all_values = (const uint8_t *)iq6nl_values;

    int i4 = iqs/4;  // 0...7.  Blocks of 16 index is 4*(i4/2) + (i4%2) + (0 and 2)
                     //         Blocks of 32 index is 2*(i4/2) + 0 or 1

    const int32_t  * q8_1 = (const int *)bq8_1[2*(i4/2)+0].qs + 4*(i4%2);
    const int32_t  * q8_2 = (const int *)bq8_1[2*(i4/2)+1].qs + 4*(i4%2);
    const uint32_t * q4 = (const uint32_t *)bq6->qs + 8*(i4/2) + 4*(i4%2);
    const uint32_t * qh = (const uint32_t *)bq6->qh + 8*(i4/4) + 4*(i4%2);
    const uint16_t extra = bq6->extra >> (4*(i4/2) + (i4%2));
    const uint8_t * values1 = all_values + 64*(extra & 1);
    const uint8_t * values2 = all_values + 16*(extra & 4);
    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t h = qh[j] >> 4*((i4/2)%2);
        aux32[0] = ((q4[j] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x30303030);
        aux32[1] = ((q4[j] >> 4) & 0x0f0f0f0f) | ((h << 2) & 0x30303030);
        v1 = int_from_table(a8+0, values1);
        v2 = int_from_table(a8+4, values2);
        sumi1 = ggml_cuda_dp4a(v1, q8_1[j], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8_2[j], sumi2);
    }
    const float d6 = __half2float(bq6->d);
    *result += d6 * (__low2float(bq8_1[2*(i4/2)+0].ds) * sumi1 * bq6->scales[4*(i4/2)+(i4%2)] + __low2float(bq8_1[2*(i4/2)+1].ds) * sumi2 * bq6->scales[4*(i4/2)+(i4%2)+2]);
}

void mul_mat_vec_iq6_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ6_K, VDR_IQ6_K_Q8_1_MMVQ, vec_dot_iq6_k_q8_1>(args, stream);
}

