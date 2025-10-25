#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq4_k_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq4_k_r4 * bq4 = (const block_iq4_k_r4 *)vbq + kbx;

    // iqs is 0...28 in steps of 2
    const int ib16 = iqs/2;
    const float d8 = __low2float(bq8_1[ib16/2].ds);
    const int32_t  * q8 = (const int *)bq8_1[ib16/2].qs + 4*(ib16%2);

    int ib32 = ib16/2;
    int is   = ib16%2;
    int scales;
    const uint32_t * scales_l = (const uint32_t *)bq4->scales_l;
    const uint32_t * scales_h = (const uint32_t *)bq4->scales_h;
    scales = __vsub4(((scales_l[2*(ib32%4)+is] >> 4*(ib32/4)) & 0x0f0f0f0f) | (((scales_h[2*(ib32%2)+is] >> 2*(ib32/2)) & 0x03030303) << 4), 0x20202020);
    const int8_t * s8 = (const int8_t *)&scales;
    int2 val1;
    const int * q4 = (const int *)bq4->qs + 16*ib32;
    for (int i = 0; i < 4; ++i) {
        auto values1 = iq4k_values + (((bq4->extra[i+4*is] >> ib32) & 1) << 4);
        int sumi1 = 0;
        val1  = get_int_from_table_16(q4[i+4*is+0], values1);
        sumi1 = ggml_cuda_dp4a(val1.x, q8[0], ggml_cuda_dp4a(val1.y, q8[2], sumi1));
        val1  = get_int_from_table_16(q4[i+4*is+8], values1);
        sumi1 = ggml_cuda_dp4a(val1.x, q8[1], ggml_cuda_dp4a(val1.y, q8[3], sumi1));
        const float d = __half2float(bq4->d[i]) * d8;
        result[i] += d * sumi1 * s8[i];
    }
}

void mul_mat_vec_iq4_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_K_R4, 2, vec_dot_iq4_k_r4_q8_1, 4>(args, stream);
}

