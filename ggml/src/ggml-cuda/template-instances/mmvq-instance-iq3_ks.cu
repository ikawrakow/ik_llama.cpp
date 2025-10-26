#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq3_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs, float * result) {

    float d = __half2float(*(const half *)vbq);
    const block_iq3_ks * bq3 = (const block_iq3_ks *)((const char *)vbq + sizeof(half)) + kbx;

    int iqs = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1. 0 works on quants 0...127, 1 on quants 128...255
                              // Each thread processes 8 quants in each of the 4 32-blocks
    const int il8   = iqs%4;  // 0...3. 0 works on quants 0...7, 1 on quants 8...15, 2 on 16...23, 3 on 24...31

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    uint16_t extra = bq3->extra >> 4*ib128;
    uint32_t extra_v = uint32_t(extra >> 8) * 0x01010101;

    uint32_t extra32_1 = ((extra_v << 3) & 0x08080808) | ((extra_v << 5) & 0x80808080);
    uint32_t extra32_2 = ((extra_v << 2) & 0x08080808) | ((extra_v << 4) & 0x80808080);

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
    for (int i = 0; i < 2; ++i) {
        uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
        uint32_t vh = ((qh[2*i+0] | (qh[2*i+1] << 16)) >> 4*ib128);

        uint32_t val1 = ((vl >> 0) & 0x33333333) | extra32_1 | ((vh << 2) & 0x04040404) | ((vh << 4) & 0x40404040);
        uint32_t val2 = ((vl >> 2) & 0x33333333) | extra32_2 | ((vh << 1) & 0x04040404) | ((vh << 3) & 0x40404040);
        int2 v1 = get_int_from_table_16(val1, iq3nl_values);
        int2 v2 = get_int_from_table_16(val2, iq3nl_values);

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        sumi[0] = ggml_cuda_dp4a(v1.x, q8[i], sumi[0]);

        q8 += sizeof(block_q8_1)/4;
        sumi[1] = ggml_cuda_dp4a(v2.x, q8[i], sumi[1]);

        q8 += sizeof(block_q8_1)/4;
        sumi[2] = ggml_cuda_dp4a(v1.y, q8[i], sumi[2]);

        q8 += sizeof(block_q8_1)/4;
        sumi[3] = ggml_cuda_dp4a(v2.y, q8[i], sumi[3]);
    }
    const uint16_t * sl16 = (const uint16_t *)bq3->scales;
    int32_t aux32 = __vsub4(((sl16[0] | (sl16[1] << 16)) >> 4*ib128) & 0x0f0f0f0f, 0x10101010);
    const int8_t * a8 = (const int8_t *)&aux32;
    *result += d * (__low2float(bq8_1[4*ib128+0].ds) * (a8[0] + ((extra << 4) & 0x10)) * sumi[0] +
                    __low2float(bq8_1[4*ib128+1].ds) * (a8[1] + ((extra << 3) & 0x10)) * sumi[1] +
                    __low2float(bq8_1[4*ib128+2].ds) * (a8[2] + ((extra << 2) & 0x10)) * sumi[2] +
                    __low2float(bq8_1[4*ib128+3].ds) * (a8[3] + ((extra << 1) & 0x10)) * sumi[3]);

}

void mul_mat_vec_iq3_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_KS, VDR_IQ3_K_Q8_1_MMVQ, vec_dot_iq3_ks_q8_1>(args, stream);
}

