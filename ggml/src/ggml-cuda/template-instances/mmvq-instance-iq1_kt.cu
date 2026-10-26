#include "../iqk_mmvq_templates.cuh"

static __device__ __forceinline__ void vec_dot_iq1_kt_q8_1_impl(
    float scale, const uint8_t * sh, const uint8_t * ql, const uint8_t * qh, const block_q8_1 * __restrict__ bq8_1, const int ib32, float * result) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const int ls = iq4k_values[sh[ib32] & 0xf];
    const float dl = scale * ls;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t val = ql[4*ib32+j] + 4096 + ((qh[4*(ib32%4)+j] << (8 - 4*(ib32/4))) & 0xf00) + ((sh[ib32] << (8 - j)) & 0x1000);
        int v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v4 |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        sumi = ggml_cuda_dp4a(v4, q8[2*j+0], sumi);
        v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v4 |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        sumi = ggml_cuda_dp4a(v4, q8[2*j+1], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

__device__ __forceinline__ void vec_dot_iq1_kt_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq1_kt * bq1 = (const block_iq1_kt *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    vec_dot_iq1_kt_q8_1_impl(scale, bq1->sh, bq1->ql, bq1->qh, bq8_1, iqs/4, result);
}

__device__ __forceinline__ void vec_dot_iq1_kt_q8_1_tail(
    const void * __restrict__ vbq, const void * __restrict__ bq8_1_v, const int & kbx, const int & iqs, const int & nt, float * result) {

    const int ib32 = iqs/4;
    if (ib32 >= nt) return;

    float scale = *(const float *)vbq;
    const block_iq1_kt * bq1 = (const block_iq1_kt *)((const char *)vbq + sizeof(float)) + kbx;
    vec_dot_iq1_kt_q8_1_impl(scale, bq1->sh, bq1->ql, bq1->ql + 4*nt, (const block_q8_1 *)bq8_1_v, ib32, result);
}

void mul_mat_vec_iq1_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    if (args.ncols_x % QK_K == 0) {
        iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ1_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq1_kt_q8_1, 1, nullptr>(args, stream);
    } else {
        iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ1_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq1_kt_q8_1, 1, vec_dot_iq1_kt_q8_1_tail>(args, stream);
    }
}

