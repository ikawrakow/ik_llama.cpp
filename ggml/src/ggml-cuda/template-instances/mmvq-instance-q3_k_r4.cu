#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_q3_k_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_q3_k_r4 * bq3 = (const block_q3_k_r4 *)vbq + kbx;

    // iqs is 0...30 in steps of 2 -> ib16 = iqs/2 is 0...15
    const int ib16 = iqs/2;
    const int ib32 = ib16/2;   // 0...7  -> 32-column block within the super-block
    const int is   = ib16%2;   // 0...1  -> 16-column half within the 32-column block

    const float d8 = __low2float(bq8_1[ib16/2].ds);
    const int * q8 = (const int *)bq8_1[ib16/2].qs + 4*(ib16%2);

    // Per-row scales: scale index is = 8*ib32 + 4*is + k (k = row).
    // All 4 rows live in consecutive bytes of scales_l / scales_h, so one
    // uint32 read gives the scale nibble/field for each row in its own byte.
    const uint32_t * scales_l = (const uint32_t *)bq3->scales_l;
    const uint32_t * scales_h = (const uint32_t *)bq3->scales_h;

    int sc = ((scales_l[2*(ib32%4)+is] >> 4*(ib32/4)) & 0x0f0f0f0f)
           | (((scales_h[(2*ib32+is)%4] >> 2*((2*ib32+is)/4)) & 0x03030303) << 4);
    // 6-bit scale in [0, 63] -> (scale - 32) in [-32, 31], one byte per row
    sc = __vsub4(sc, 0x20202020);
    const int8_t * s8 = (const int8_t *)&sc;

    const uint32_t * q2 = (const uint32_t *)bq3->qs + 8*ib32 + 4*is; // word i -> row i, half is
    const uint32_t * qh = (const uint32_t *)bq3->qh + 4*ib32;        // word i -> row i

    for (int i = 0; i < 4; ++i) {
        const uint32_t q = q2[i];
        const uint32_t h = qh[i] >> 4*is; // bits 0...3: high bits of the 16 columns of this half

        int sumi = 0;
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            // byte j = 3-bit value (low 2 bits | high bit << 2) of column 16*is + 4*m + j
            const uint32_t v = ((q >> 2*m) & 0x03030303) | (((h >> m) & 0x01010101) << 2);
            // q3_K values are index - 4 -> signed bytes in [-4, 3]
            sumi = ggml_cuda_dp4a(__vsub4(v, 0x04040404), q8[m], sumi);
        }
        result[i] += __half2float(bq3->d[i]) * d8 * s8[i] * sumi;
    }
}

void mul_mat_vec_q3_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_Q3_K_R4, 2, vec_dot_q3_k_r4_q8_1, 4>(args, stream);
}
