#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq4_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq4_ks * bq4 = (const block_iq4_ks *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4; // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint32_t * q4 = (const uint32_t *)bq4->qs + 4*ib32;
    const float dl = scale * ((bq4->scales[ib32] & 254) - 127);
    auto values = iq4k_values + ((bq4->scales[ib32] & 1) << 4);
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        auto v = get_int_from_table_16(q4[j], values);
        sumi = ggml_cuda_dp4a(v.x, q8[j+0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[j+4], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

void mul_mat_vec_iq4_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KS, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq4_ks_q8_1>(args, stream);
}

