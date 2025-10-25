#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq2_kt_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

    float scale = *(const float *)vbq;
    const block_iq2_kt * bq2 = (const block_iq2_kt *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4;
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const int ls = iq4k_values[(bq2->scales[ib32%4] >> 4*(ib32/4)) & 0xf];
    const float dl = scale * ls * 1.05f;
    auto ql = (const uint16_t *)bq2->ql;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t val = ql[4*ib32+j] + 4096;
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

void mul_mat_vec_iq2_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq2_kt_q8_1>(args, stream);
}

