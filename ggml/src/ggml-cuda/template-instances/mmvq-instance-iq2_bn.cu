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
        sumi1 = ggml_cuda_dp4a(vl & 0x03030303, q8l[j+0], sumi1);
        sumi2 = ggml_cuda_dp4a(vl & 0x0c0c0c0c, q8l[j+4], sumi2);
        sumi3 = ggml_cuda_dp4a(vh & 0x03030303, q8h[j+0], sumi3);
        sumi4 = ggml_cuda_dp4a(vh & 0x0c0c0c0c, q8h[j+4], sumi4);
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

__device__ __forceinline__ void vec_dot_pq2_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_pq2_0 * bq = (const block_pq2_0 *) vbq + kbx;

    // iqs selects which of the 4 chunks of 32 elements of the 128-element block to process (0-3)
    const float       d2 = bq->d;
    const int16_t   * qs = (const int16_t *) bq->qs + iqs * 4;
    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q  = qs[j];
        const int u  = ((const int *) bq8_1_chunk->qs)[2*j + 0];
        const int v  = ((const int *) bq8_1_chunk->qs)[2*j + 1];

        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);

        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    *result += d2 * d8 * sumi;
#else
    int sumi = 0;
    for (int j = 0; j < 8; ++j) {
        sumi += bq8_1_chunk->qs[4*j + 0] * (((bq->qs[8*iqs + j] >> 0) & 0x03) - 1);
        sumi += bq8_1_chunk->qs[4*j + 1] * (((bq->qs[8*iqs + j] >> 2) & 0x03) - 1);
        sumi += bq8_1_chunk->qs[4*j + 2] * (((bq->qs[8*iqs + j] >> 4) & 0x03) - 1);
        sumi += bq8_1_chunk->qs[4*j + 3] * (((bq->qs[8*iqs + j] >> 6) & 0x03) - 1);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    *result += d2 * d8 * sumi;
#endif
}

void mul_mat_vec_iq2_bn_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_BN, 1, vec_dot_iq2_bn_q8_1>(args, stream);
}

void mul_mat_vec_pq2_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    //printf("========================= nx, ny = %d, %d\n", args.ncols_x, args.ncols_y);
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_PQ2_0, 1, vec_dot_pq2_0_q8_1>(args, stream);
}

