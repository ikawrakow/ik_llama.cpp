#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq4_kt_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq4_kt * bq4 = (const block_iq4_kt *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4; // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    //const int8_t  * q8 = bq8_1[ib32].qs;
    const int ls = (bq4->qs[ib32] & 0xff) >> 1;
    const float dl = scale * (ls - 64);
    const uint32_t idx0 = ((bq4->qs[ib32] & 1) << 15) + 4096;
    auto ql = (const uint8_t *)(bq4->qs + 8);
    auto qh = ql + 64;
    ql += 8*ib32;
    qh += 8*(ib32%4);
    const int shift1 = 8 - 4*(ib32/4);
    int sumi = 0;
    for (int j = 0; j < 8; ++j) {
        const uint32_t sh = bq4->qs[ib32] >> (8 + 3*j);
        uint32_t val = ql[j] + ((qh[j] << shift1) & 0xf00) + ((sh & 7) << 12) + idx0;
        sumi = kt_dot4(val, q8[j], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

__device__ __forceinline__ void vec_dot_iq4_kt_q8_1_tail(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, const int & nt, float * result) {

    const int ib32 = iqs/4;
    if (ib32 >= nt) return;

    float scale = *(const float *)vbq;
    const uint32_t * shb = (const uint32_t *)((const block_iq4_kt *)((const char *)vbq + sizeof(float)) + kbx);
    const uint8_t * ql = (const uint8_t *)(shb + nt) + 8*ib32;
    const uint8_t * qh = (const uint8_t *)(shb + nt) + 8*nt + 4*ib32;

    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const int ls = (shb[ib32] & 0xff) >> 1;
    const float dl = scale * (ls - 64);
    const uint32_t idx0 = ((shb[ib32] & 1) << 15) + 4096;
    int sumi = 0;
    for (int j = 0; j < 8; ++j) {
        const uint32_t sh = shb[ib32] >> (8 + 3*j);
        uint32_t val = ql[j] + (((qh[j/2] >> 4*(j&1)) & 0xf) << 8) + ((sh & 7) << 12) + idx0;
        sumi = kt_dot4(val, q8[j], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

void mul_mat_vec_iq4_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    if (args.ncols_x % QK_K == 0) {
        iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq4_kt_q8_1, 1, nullptr>(args, stream);
    } else {
        iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq4_kt_q8_1, 1, vec_dot_iq4_kt_q8_1_tail>(args, stream);
    }
}
