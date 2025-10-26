#include "../iqk_mmvq_templates.cuh"

__device__ __forceinline__ void vec_dot_iq1_bn_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    half d16; memcpy(&d16, vbq, sizeof(d16));
    float scale = d16;
    const block_iq1_bn * bq1 = (const block_iq1_bn *)((const char *)vbq + sizeof(d16)) + kbx;

    // iqs is 0 or 1

    int sumi = 0;
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    uint16_t mult[2];
    mult[1] = iqs == 0 ? 27 : 3;
    mult[0] = mult[1] + (mult[1] << 1);
    const int * q8 = (const int *)bq8_1[iqs].qs;
    int val[4];
    for (int l = 0; l < 2; ++l) {
        int8_t * a = (int8_t *)val;
        const int i16 = 2*iqs + l;
        for (int k = 0; k < 3; ++k) {
            uint16_t q = bq1->ql[3*i16+k];
            for (int j = 4; j >= 0; --j) {
                uint16_t v = q & 0xff;
                v += v << 1;
                a[j] = v >> 8;
                q += q << 1;
            }
            a += 5;
        }
        uint16_t v = (mult[l]*bq1->extra) & 0xff;
        v += v << 1;
        *a = v >> 8;
        sumi = __dp4a(val[0], q8[4*l+0], __dp4a(val[1], q8[4*l+1], __dp4a(val[2], q8[4*l+2], __dp4a(val[3], q8[4*l+3], sumi))));
    }
    float2 d8 = __half22float2(bq8_1[iqs].ds);
    *result += scale * (d8.x * sumi - d8.y);
#else
    static const uint16_t k_mult[5] = {81, 27, 9, 3, 1};
    const int8_t * q8 = bq8_1[iqs].qs;
    for (int l = 0; l < 2; ++l) {
        const int i16 = 2*iqs + l;
        for (int k = 0; k < 3; ++k) {
            uint8_t q = bq1->ql[3*i16+k];
            for (int j = 0; j < 5; ++j) {
                uint8_t v = k_mult[j]*q;
                int8_t vs = (v + (v >> 1)) >> 7;
                sumi += q8[j]*(vs - 1);
            }
            q8 += 5;
        }
        uint8_t v = k_mult[i16]*bq1->extra;
        int8_t vs = (v + (v >> 1)) >> 7;
        sumi += q8[0]*(vs - 1);
        q8++;
    }
    *result += scale * __low2float(bq8_1[iqs].ds) * sumi;
#endif
}

void mul_mat_vec_iq1_bn_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ1_BN, 1, vec_dot_iq1_bn_q8_1>(args, stream);
}

