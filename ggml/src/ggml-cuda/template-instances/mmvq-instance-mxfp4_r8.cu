#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_mxfp4_r8_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_mxfp4_r8 * bq4 = (const block_mxfp4_r8 *)vbq + kbx;

    const int * q8 = (const int *)bq8_1->qs + iqs;
    const float d8 = __low2float(bq8_1->ds);

    // One block holds 8 rows. Row ir's 32 columns live in uint32 words 8*j + ir
    // (j = 0..3 column sub-block); its scale is e[ir].
#pragma unroll
    for (int ir = 0; ir < 8; ++ir) {
        int2 sumi = {0, 0};
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            // word 8*(iqs+l) + ir holds row ir, column sub-block (iqs+l); read it
            // directly (vecdotq.cuh helpers are not in the mmvq include chain).
            const int aux_q4 = ((const uint32_t *)bq4->qs)[8*(iqs + l) + ir];
            const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
            sumi.x = ggml_cuda_dp4a(v.x, q8[l + 0], sumi.x);
            sumi.y = ggml_cuda_dp4a(v.y, q8[l + 4], sumi.y);
        }

        union { float f; uint32_t u; } helper;
        helper.u = bq4->e[ir] ? uint32_t(bq4->e[ir]) << 23u : 0x00400000;

        result[ir] += 0.5f * helper.f * d8 * (sumi.x + sumi.y);
    }
}

void mul_mat_vec_mxfp4_r8_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_MXFP4_R8, 2, vec_dot_mxfp4_r8_q8_1, 8>(args, stream);
}