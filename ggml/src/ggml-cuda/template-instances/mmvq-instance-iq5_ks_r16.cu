#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq5_ks_r16_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const float * dptr = (const float *)vbq;
    const auto bq5 = (const block_iq5_ks_r16 *)(dptr + 16) + kbx;

    auto values = (const uint8_t *)iq5nl_values;

    const int ir = 4*(blockIdx.x%4) + 2*(iqs/8);

    auto d8 = __half22float2(bq8_1->ds);
    const int32_t * q8 = (const int *)bq8_1->qs;

    auto sas = (const uint16_t *)(bq5->scales + ir);
    uint16_t shifts = (sas[0] & 0x0101) << 1;
    uint16_t scales = __vsub2(sas[0] & 0xfefe, 0x7f7f);
    auto shifts8 = (const uint8_t *)&shifts;
    auto scales8 = (const  int8_t *)&scales;

    auto qs = (const uint32_t *)bq5->qs + ir;
    auto qh = (const uint32_t *)bq5->qh + ir;
    int32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        int sumi = 0;
        uint32_t h = qh[j];

        aux32[0] = (qs[j] & 0x0f0f0f0f) | ((h << 4) & 0x10101010);
        aux32[1] = ((qs[j] >> 4) & 0x0f0f0f0f) | (h & 0x10101010);
        sumi = ggml_cuda_dp4a(int_from_table(a8, values), q8[0], ggml_cuda_dp4a(int_from_table(a8+4, values), q8[4], sumi));

        aux32[0] = (qs[j+16] & 0x0f0f0f0f) | ((h << 3) & 0x10101010);
        aux32[1] = ((qs[j+16] >> 4) & 0x0f0f0f0f) | ((h >> 1) & 0x10101010);
        sumi = ggml_cuda_dp4a(int_from_table(a8, values), q8[1], ggml_cuda_dp4a(int_from_table(a8+4, values), q8[5], sumi));

        aux32[0] = (qs[j+32] & 0x0f0f0f0f) | ((h << 2) & 0x10101010);
        aux32[1] = ((qs[j+32] >> 4) & 0x0f0f0f0f) | ((h >> 2) & 0x10101010);
        sumi = ggml_cuda_dp4a(int_from_table(a8, values), q8[2], ggml_cuda_dp4a(int_from_table(a8+4, values), q8[6], sumi));

        aux32[0] = (qs[j+48] & 0x0f0f0f0f) | ((h << 1) & 0x10101010);
        aux32[1] = ((qs[j+48] >> 4) & 0x0f0f0f0f) | ((h >> 3) & 0x10101010);
        sumi = ggml_cuda_dp4a(int_from_table(a8, values), q8[3], ggml_cuda_dp4a(int_from_table(a8+4, values), q8[7], sumi));

        result[2*(iqs/8)+j] += dptr[ir+j] * (d8.x * sumi + d8.y * shifts8[j]) * scales8[j];
    }
}

// We tell the framework that we have 4 interleaved rows. The actual interleaving is recovered
// via ggml_cuda_actual_row0<type>
void mul_mat_vec_iq5_ks_r16_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ5_KS_R16, 8, vec_dot_iq5_ks_r16_q8_1, 4>(args, stream);
}

