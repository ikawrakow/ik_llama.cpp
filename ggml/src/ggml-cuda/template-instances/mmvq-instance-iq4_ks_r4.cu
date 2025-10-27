#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq4_ks_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const float * dptr = (const float *)vbq;
    const block_iq4_ks_r4 * bq4 = (const block_iq4_ks_r4 *)(dptr + 4) + kbx;

    // iqs is 0...28 in steps of 2
    const int ib16 = iqs/2;
    const float d8 = __low2float(bq8_1[ib16/2].ds);
    const int32_t  * q8 = (const int *)bq8_1[ib16/2].qs + 4*(ib16%2);

    int ib32 = ib16/2;
    int is   = ib16%2;
    const uint32_t * scales32 = (const uint32_t *)bq4->scales;
    int scales = __vsub4(scales32[ib32] & 0xfefefefe, 0x7f7f7f7f);
    const int8_t * s8 = (const int8_t *)&scales;
    int2 val;
    const int * q4 = (const int *)bq4->qs + 16*ib32;
    for (int i = 0; i < 4; ++i) {
        auto values = iq4k_values + ((bq4->scales[4*ib32+i] & 1) << 4);
        int sumi = 0;
        val  = get_int_from_table_16(q4[i+4*is+0], values);
        sumi = ggml_cuda_dp4a(val.x, q8[0], ggml_cuda_dp4a(val.y, q8[2], sumi));
        val  = get_int_from_table_16(q4[i+4*is+8], values);
        sumi = ggml_cuda_dp4a(val.x, q8[1], ggml_cuda_dp4a(val.y, q8[3], sumi));
        const float d = dptr[i] * d8;
        result[i] += d * sumi * s8[i];
    }
}

void mul_mat_vec_iq4_ks_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KS_R4, 2, vec_dot_iq4_ks_r4_q8_1, 4>(args, stream);
}

