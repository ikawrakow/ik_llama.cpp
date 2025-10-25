#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq2_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const half *)vbq;
    const block_iq2_ks * bq2 = (const block_iq2_ks *)((const char *)vbq + sizeof(half)) + kbx;

    int i4 = iqs/4;  // 0...7. We will process q8 blocks 4*(i4/4), 4*(i4/4)+1, 4*(i4/4)+2, 4*(i4/4)+3
    const int32_t  * q8_1 = (const int *)bq8_1[4*(i4/4)+0].qs + 2*(i4%4);
    const int32_t  * q8_2 = (const int *)bq8_1[4*(i4/4)+1].qs + 2*(i4%4);
    const int32_t  * q8_3 = (const int *)bq8_1[4*(i4/4)+2].qs + 2*(i4%4);
    const int32_t  * q8_4 = (const int *)bq8_1[4*(i4/4)+3].qs + 2*(i4%4);

    const uint16_t * q2 = (const uint16_t *)bq2->qs + 16*(i4/4) + 4*(i4%4);
    const uint16_t extra = bq2->extra >> 4*(i4/4);

    uint32_t val1 = q2[0] | (q2[1] << 16), val2 = q2[2] | (q2[3] << 16);

    int32_t scales32;
    const uint16_t * scales16 = (const uint16_t *)bq2->scales;
    scales32 = __vsub4((scales16[i4/4] | (scales16[i4/4] << 12)) & 0x0f0f0f0f, 0x10101010);
    int8_t * s8 = (int8_t *)&scales32;
    s8[0] += ((extra >> 4) & 0x10);
    s8[1] += ((extra >> 6) & 0x10);
    s8[2] += ((extra >> 5) & 0x10);
    s8[3] += ((extra >> 7) & 0x10);

#ifdef __CUDA_ARCH__

    uint32_t extra32 = uint32_t(extra & 0xf) * 0x01010101;

    uint32_t this_extra = ((extra32 << 2) & 0x04040404) | ((extra32 << 4) & 0x40404040);
    uint32_t idx1 = ((val1 >> 0) & 0x33333333) | this_extra;
    uint32_t idx2 = ((val2 >> 0) & 0x33333333) | this_extra;
    int2 v1 = get_int_from_table_8(idx1, iq2nl_values);
    int2 v2 = get_int_from_table_8(idx2, iq2nl_values);

    int sumi1 = ggml_cuda_dp4a(v2.x, q8_1[1], ggml_cuda_dp4a(v1.x, q8_1[0], 0)) * s8[0];
    int sumi3 = ggml_cuda_dp4a(v2.y, q8_3[1], ggml_cuda_dp4a(v1.y, q8_3[0], 0)) * s8[1];

    this_extra = ((extra32 << 1) & 0x04040404) | ((extra32 << 3) & 0x40404040);
    idx1 = ((val1 >> 2) & 0x33333333) | this_extra;
    idx2 = ((val2 >> 2) & 0x33333333) | this_extra;
    v1 = get_int_from_table_8(idx1, iq2nl_values);
    v2 = get_int_from_table_8(idx2, iq2nl_values);

    int sumi2 = ggml_cuda_dp4a(v2.x, q8_2[1], ggml_cuda_dp4a(v1.x, q8_2[0], 0)) * s8[2];
    int sumi4 = ggml_cuda_dp4a(v2.y, q8_4[1], ggml_cuda_dp4a(v1.y, q8_4[0], 0)) * s8[3];

#else
    uint32_t aux32[2];
    int v1, v2;
    const int * all_values = (const int *)iq2k_table;
    const int * values;

    aux32[0] = ((val1 >> 0) & 0x03030303); aux32[1] = ((val2 >> 0) & 0x03030303); values = all_values + ((extra & 0x01) << 8);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi1 = ggml_cuda_dp4a(v2, q8_1[1], ggml_cuda_dp4a(v1, q8_1[0], 0)) * s8[0];

    aux32[0] = ((val1 >> 2) & 0x03030303); aux32[1] = ((val2 >> 2) & 0x03030303); values = all_values + ((extra & 0x02) << 7);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi2 = ggml_cuda_dp4a(v2, q8_2[1], ggml_cuda_dp4a(v1, q8_2[0], 0)) * s8[2];

    aux32[0] = ((val1 >> 4) & 0x03030303); aux32[1] = ((val2 >> 4) & 0x03030303); values = all_values + ((extra & 0x04) << 6);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi3 = ggml_cuda_dp4a(v2, q8_3[1], ggml_cuda_dp4a(v1, q8_3[0], 0)) * s8[1];

    aux32[0] = ((val1 >> 6) & 0x03030303); aux32[1] = ((val2 >> 6) & 0x03030303); values = all_values + ((extra & 0x08) << 5);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi4 = ggml_cuda_dp4a(v2, q8_4[1], ggml_cuda_dp4a(v1, q8_4[0], 0)) * s8[3];
#endif

    *result += scale * (__low2float(bq8_1[4*(i4/4)+0].ds) * sumi1
                     +  __low2float(bq8_1[4*(i4/4)+1].ds) * sumi2
                     +  __low2float(bq8_1[4*(i4/4)+2].ds) * sumi3
                     +  __low2float(bq8_1[4*(i4/4)+3].ds) * sumi4);
}

void mul_mat_vec_iq2_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_KS, VDR_IQ2_KS_Q8_1_MMVQ, vec_dot_iq2_ks_q8_1>(args, stream);
}

