#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq2_kl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs, float * result) {

    float d = __half2float(*(const half *)vbq);
    const block_iq2_kl * bq2 = (const block_iq2_kl *)((const char *)vbq + sizeof(half)) + kbx;

    int iqs = iiqs/4;
    const int ib64 = iqs/2;  // 0...3. 0 works on quants 0...63, 1 on quants 64...127, etc.
                             // Each thread processes 16 quants in each of the 2 32-blocks
    const int il16 = iqs%2;  // 0...3. 0 works on quants 0...7, 1 on quants 8...15, 2 on 16...23, 3 on 24...31

    const uint16_t * ql = (const uint16_t *)bq2->qs + 8*ib64 + 4*il16;
    const uint16_t * qh = (const uint16_t *)bq2->qh + 4*il16;

    int32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    const int * q8l = (const int *)bq8_1[2*ib64+0].qs + 4*il16;
    const int * q8h = (const int *)bq8_1[2*ib64+1].qs + 4*il16;

    int sumi1 = 0, sumi2 = 0;
    int v1, v2;
    for (int i = 0; i < 2; ++i) {
        uint32_t vl =  ql[2*i+0] | (ql[2*i+1] << 16);
        uint32_t vh = (qh[2*i+0] | (qh[2*i+1] << 16)) >> 2*ib64;

        aux32 = (vl & 0x0f0f0f0f) | ((vh << 4) & 0x10101010);
        v1 = iq2kl_values[aux8[0]] | (iq2kl_values[aux8[1]] << 16);
        v2 = iq2kl_values[aux8[2]] | (iq2kl_values[aux8[3]] << 16);
        sumi1 = ggml_cuda_dp4a(v1, q8l[2*i+0], ggml_cuda_dp4a(v2, q8l[2*i+1], sumi1));

        aux32 = ((vl >> 4) & 0x0f0f0f0f) | ((vh << 3) & 0x10101010);
        v1 = iq2kl_values[aux8[0]] | (iq2kl_values[aux8[1]] << 16);
        v2 = iq2kl_values[aux8[2]] | (iq2kl_values[aux8[3]] << 16);
        sumi2 = ggml_cuda_dp4a(v1, q8h[2*i+0], ggml_cuda_dp4a(v2, q8h[2*i+1], sumi2));
    }

    auto sh = bq2->scales_h >> 4*ib64;
    int ls1 = int(((bq2->scales_l[(2*ib64+0)%4] >> 4*(ib64/2)) & 0xf) | ((sh << 4) & 0x30)) - 32;
    int ls2 = int(((bq2->scales_l[(2*ib64+1)%4] >> 4*(ib64/2)) & 0xf) | ((sh << 2) & 0x30)) - 32;

    *result += d * (__low2float(bq8_1[2*ib64+0].ds) * ls1 * sumi1 + __low2float(bq8_1[2*ib64+1].ds) * ls2 * sumi2);

}

void mul_mat_vec_iq2_kl_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_KL, VDR_IQ3_K_Q8_1_MMVQ, vec_dot_iq2_kl_q8_1>(args, stream);
}

