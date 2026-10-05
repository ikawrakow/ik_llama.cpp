#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq4_ks_r16_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const float * dptr = (const float *)vbq;
    const auto bq4 = (const block_iq4_ks_r16 *)(dptr + 16) + kbx;

    const int ir = 4*(blockIdx.x%4);
    const int il = iqs / 8;  // 0 or 1

    auto d8 = __half22float2(bq8_1->ds);
    const int32_t * q8 = (const int *)bq8_1->qs + 2*il;

    auto sas = (const uint32_t *)(bq4->scales + ir);
    uint32_t shifts[1];
    int32_t  scales[1];
    #pragma unroll
    for (int j = 0; j < 1; ++j) {
        shifts[j] = (sas[j] & 0x01010101) << 1;
        scales[j] = __vsub4(sas[j] & 0xfefefefe, 0x7f7f7f7f);
    }
    auto shifts8 = (const uint8_t *)shifts;
    auto scales8 = (const  int8_t *)scales;
    int32_t isum[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        auto q4 = (const int *)bq4->qs + 32*il + ir + j;
        int sumi = 0;
        int2 val = get_int_from_table_16(q4[0], iq4k_values);
        sumi = ggml_cuda_dp4a(val.x, q8[0], ggml_cuda_dp4a(val.y, q8[4], sumi));
        val = get_int_from_table_16(q4[16], iq4k_values);
        sumi = ggml_cuda_dp4a(val.x, q8[1], ggml_cuda_dp4a(val.y, q8[5], sumi));
        isum[j] = sumi * scales8[j];
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) result[j] += dptr[ir+j] * (d8.x * isum[j] + d8.y * shifts8[j] * scales8[j]);
}

// We tell the framework that we have 4 interleaved rows. The actual interleaving is recovered
// via ggml_cuda_actual_row0<type>
void mul_mat_vec_iq4_ks_r16_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KS_R16, 8, vec_dot_iq4_ks_r16_q8_1, 4>(args, stream);
}

