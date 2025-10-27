#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq2_bn_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq2_bn * bq2 = (const block_iq2_bn *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0 or 1

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    auto qs  = (const int *)bq2->qs + 2*iqs;
    auto q8l = (const int *)bq8_1[0].qs + 2*iqs;
    auto q8h = (const int *)bq8_1[1].qs + 2*iqs;
    int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
    for (int j = 0; j < 2; ++j) {
        int vl = qs[j];
        int vh = qs[j] >> 4;
        sumi1 = __dp4a(vl & 0x03030303, q8l[j+0], sumi1);
        sumi2 = __dp4a(vl & 0x0c0c0c0c, q8l[j+4], sumi2);
        sumi3 = __dp4a(vh & 0x03030303, q8h[j+0], sumi3);
        sumi4 = __dp4a(vh & 0x0c0c0c0c, q8h[j+4], sumi4);
    }
    auto d8l = __half22float2(bq8_1[0].ds);
    auto d8h = __half22float2(bq8_1[1].ds);
    *result += scale * (d8l.x * (sumi1 + 0.25f*sumi2) + d8h.x * (sumi3 + 0.25f * sumi4) - 0.5f*d8l.y - 0.5f*d8h.y);
#else
    int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
    auto q8l = bq8_1[0].qs + 8*iqs;
    auto q8h = bq8_1[1].qs + 8*iqs;
    auto qs  = bq2->qs + 8*iqs;
    for (int j = 0; j < 8; ++j) {
        sumi1 += q8l[j+ 0] * (qs[j] & 0x03);
        sumi2 += q8l[j+16] * (qs[j] & 0x0c);
        sumi3 += q8h[j+ 0] * (qs[j] & 0x30);
        sumi4 += q8h[j+16] * (qs[j] & 0xc0);
    }
    auto d8l = __half22float2(bq8_1[0].ds);
    auto d8h = __half22float2(bq8_1[1].ds);
    *result += scale * (d8l.x * (sumi1 + 0.25f*sumi2) + 0.0625f * d8h.x*(sumi3 + 0.25f*sumi4) - 0.5f*d8l.y - 0.5f*d8h.y);
#endif
}

void mul_mat_vec_iq2_bn_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_BN, 1, vec_dot_iq2_bn_q8_1>(args, stream);
}

