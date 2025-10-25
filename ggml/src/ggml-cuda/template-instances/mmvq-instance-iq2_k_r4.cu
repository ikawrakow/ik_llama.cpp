#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq2_k_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq2_k_r4 * bq2 = (const block_iq2_k_r4 *)vbq + kbx;

    // iqs is 0...30 in steps of 2
    const int ib16 = iqs/2;
    const float d8 = __low2float(bq8_1[ib16/2].ds);
    const int32_t  * q8 = (const int *)bq8_1[ib16/2].qs + 4*(ib16%2);

    int ib32 = ib16/2;
    int is   = ib16%2;
    const int * scales_l = (const int *)bq2->scales;

    int scales = __vsub4(((scales_l[2*(ib32%4)+is] >> 4*(ib32/4)) & 0x0f0f0f0f), 0x08080808);
    const int8_t * s8 = (const int8_t *)&scales;

    const int * q2 = (const int *)bq2->qs + 8*ib32 + 4*is;

#ifdef __CUDA_ARCH__

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t extra32 = uint32_t((bq2->extra[i+4*is] >> ib32) & 1) * 0x04040404;
        extra32 |= (extra32 << 4);
        uint32_t val1 = ((q2[i] >> 0) & 0x33333333) | extra32;
        uint32_t val2 = ((q2[i] >> 2) & 0x33333333) | extra32;
        int2 v1 = get_int_from_table_8(val1, iq2nl_values);
        int2 v2 = get_int_from_table_8(val2, iq2nl_values);
        int sumi = 0;
        sumi = ggml_cuda_dp4a(v1.x, q8[0], ggml_cuda_dp4a(v2.x, q8[1], sumi));
        sumi = ggml_cuda_dp4a(v1.y, q8[2], ggml_cuda_dp4a(v2.y, q8[3], sumi));
        const float d = __half2float(bq2->d[i]) * d8;
        result[i] += d * sumi * s8[i];
    }

#else
    const int * all_values = (const int *)iq2k_table;
    int2 val1;
    int aux32[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        auto values1 = all_values + (((bq2->extra[i+4*is] >> ib32) & 1) << 8);
        int sumi1 = 0;
        aux32[0] = ((q2[i] >> 0) & 0x03030303);
        aux32[1] = ((q2[i] >> 2) & 0x03030303);
        val1.x  = int_from_table_4(aux32[0], values1);
        val1.y  = int_from_table_4(aux32[1], values1);
        sumi1 = ggml_cuda_dp4a(val1.x, q8[0], ggml_cuda_dp4a(val1.y, q8[1], sumi1));
        aux32[0] = ((q2[i] >> 4) & 0x03030303);
        aux32[1] = ((q2[i] >> 6) & 0x03030303);
        val1.x  = int_from_table_4(aux32[0], values1);
        val1.y  = int_from_table_4(aux32[1], values1);
        sumi1 = ggml_cuda_dp4a(val1.x, q8[2], ggml_cuda_dp4a(val1.y, q8[3], sumi1));
        const float d = __half2float(bq2->d[i]) * d8;
        result[i] += d * sumi1 * s8[i];
    }
#endif
}

void mul_mat_vec_iq2_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_K_R4, 2, vec_dot_iq2_k_r4_q8_1, 4>(args, stream);
}

