#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq3_k_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq3_k_r4 * bq3 = (const block_iq3_k_r4 *)vbq + kbx;

    // iqs is 0...30 in steps of 2
    const int ib16 = iqs/2;
    const float d8 = __low2float(bq8_1[ib16/2].ds);
    const int32_t  * q8 = (const int *)bq8_1[ib16/2].qs + 4*(ib16%2);

    int ib32 = ib16/2;
    int is   = ib16%2;
    int scales[2];
    const uint32_t * scales_l = (const uint32_t *)bq3->scales_l;
    const uint32_t * scales_h = (const uint32_t *)bq3->scales_h;

    scales[0] = (((scales_l[2*(ib32%4)+is] >> 4*(ib32/4)) & 0x0f0f0f0f) << 1) | 0x01010101;
    scales[1] = (scales_h[is] >> ib32) & 0x01010101;
    // This is not faster. Why?
    //scales[1] = __vcmpeq4((scales_h[is] >> ib32) & 0x01010101, 0x01010101);
    //scales[0] = __vsub4(scales[0] ^ scales[1], scales[1]);
    const int8_t * s8 = (const int8_t *)scales;
    const uint32_t * q2 = (const uint32_t *)bq3->qs + 8*ib32 + 4*is;
    const uint32_t * qh = (const uint32_t *)bq3->qh + 4*ib32;
    for (int i = 0; i < 4; ++i) {
        uint32_t extra32 = uint32_t((bq3->extra[i+4*is] >> ib32) & 1) * 0x88888888;

        int sumi1 = 0; 
        uint32_t h = qh[i] >> 4*is;
        uint32_t val1 = ((q2[i] >> 0) & 0x33333333) | extra32 | ((h << 2) & 0x04040404) | ((h << 4) & 0x40404040);
        uint32_t val2 = ((q2[i] >> 2) & 0x33333333) | extra32 | ((h << 1) & 0x04040404) | ((h << 3) & 0x40404040);
        int2 v1 = get_int_from_table_16(val1, iq3nl_values);
        int2 v2 = get_int_from_table_16(val2, iq3nl_values);
        sumi1 = ggml_cuda_dp4a(v1.x, q8[0], ggml_cuda_dp4a(v2.x, q8[1], sumi1));
        sumi1 = ggml_cuda_dp4a(v1.y, q8[2], ggml_cuda_dp4a(v2.y, q8[3], sumi1));
        const float d = __half2float(bq3->d[i]) * d8;
        result[i] += d * sumi1 * s8[i] * (s8[i+4] ? -1 : 1);
    }
}

void mul_mat_vec_iq3_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_K_R4, 2, vec_dot_iq3_k_r4_q8_1, 4>(args, stream);
}

