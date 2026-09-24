#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq3_kt_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq3_kt * bq3 = (const block_iq3_kt *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4;
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const int ls = (bq3->scales[ib32%4] >> 4*(ib32/4)) & 0xf;
    const float dl = scale * ls * 1.01f;
    auto ql = (const uint16_t *)bq3->ql;
    uint32_t mask = 0x01010101 << ib32;
    const uint32_t * qh = (const uint32_t *)bq3->qh;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t val = ql[4*ib32+j] + 4096;
        sumi = kt_dot4_signed(val, __vcmpne4(qh[2*j+0] & mask, 0), q8[2*j+0], sumi);
        sumi = kt_dot4_signed(val, __vcmpne4(qh[2*j+1] & mask, 0), q8[2*j+1], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

__device__ __forceinline__ void vec_dot_iq3_kt_q8_1_tail(
    const void * __restrict__ vbq, const void * __restrict__ bq8_1_v, const int & kbx, const int & iqs, const int & nt, float * result) {

    const block_q8_1 * __restrict__ bq8_1 = (const block_q8_1 *) bq8_1_v;
    const int ib32 = iqs/4;
    if (ib32 >= nt) return;

    float scale = *(const float *)vbq;
    const uint8_t * tail = (const uint8_t *)((const block_iq3_kt *)((const char *)vbq + sizeof(float)) + kbx);
    const uint8_t * ql = tail + 8*ib32;
    const uint8_t * qh = tail + 8*nt + 4*ib32;
    const uint8_t * scales = tail + 12*nt;

    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const int ls = (scales[ib32/2] >> 4*(ib32&1)) & 0xf;
    const float dl = scale * ls * 1.01f;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t val = (ql[2*j] | (ql[2*j+1] << 8)) + 4096;
        const uint32_t sb = qh[j];
        sumi = kt_dot4_signed(val, __vcmpne4(((sb & 0x0f) * 0x00204081) & 0x01010101, 0), q8[2*j+0], sumi);
        sumi = kt_dot4_signed(val, __vcmpne4(((sb >>   4) * 0x00204081) & 0x01010101, 0), q8[2*j+1], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

void mul_mat_vec_iq3_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    if (args.ncols_x % QK_K == 0) {
        iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq3_kt_q8_1, 1, nullptr>(args, stream);
    } else {
        iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq3_kt_q8_1, 1, vec_dot_iq3_kt_q8_1_tail>(args, stream);
    }
}
