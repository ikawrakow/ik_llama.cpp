#include "iqk_common.h"
#include "iqk_gemm_ktquants.h"
#include "ggml.h"

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#ifdef __x86_64__

namespace {

static inline uint32_t trellis_next(uint32_t& val) {
    constexpr uint32_t ka = 89226354;
    constexpr uint32_t kb = 64248484;
    constexpr uint32_t kmask = 0x8fff8fff;
    constexpr uint32_t km32 = 0x3b603b60;
    val = val*ka + kb;
    return (val & kmask) ^ km32;
}

static inline float trellis_gen(uint32_t& val, uint32_t* s) {
    const ggml_fp16_t * h = (const ggml_fp16_t *)s;
    s[0] = trellis_next(val);
    return GGML_FP16_TO_FP32(h[0]) + GGML_FP16_TO_FP32(h[1]);
}

struct Trellis1 {
    constexpr static uint32_t kmask = 0x8fff8fff;
    constexpr static uint32_t km32 = 0x3b603b60;
    constexpr static uint32_t ka = 89226354;
    constexpr static uint32_t kb = 64248484;
    constexpr static uint32_t ka1 = ka*ka;
    constexpr static uint32_t kb1 = kb*ka+kb;
    constexpr static uint32_t ka2 = ka1*ka;
    constexpr static uint32_t kb2 = kb1*ka+kb;
    constexpr static uint32_t ka3 = ka2*ka;
    constexpr static uint32_t kb3 = kb2*ka+kb;
    constexpr static uint32_t ka4 = ka3*ka;
    constexpr static uint32_t kb4 = kb3*ka+kb;
    constexpr static uint32_t ka5 = ka4*ka;
    constexpr static uint32_t kb5 = kb4*ka+kb;
    constexpr static uint32_t ka6 = ka5*ka;
    constexpr static uint32_t kb6 = kb5*ka+kb;
    constexpr static uint32_t ka7 = ka6*ka;
    constexpr static uint32_t kb7 = kb6*ka+kb;
    const __m256i mka = _mm256_setr_epi32(ka, ka1, ka2, ka3, ka4, ka5, ka6, ka7);
    const __m256i mkb = _mm256_setr_epi32(kb, kb1, kb2, kb3, kb4, kb5, kb6, kb7);
    const __m256i mask1 = _mm256_set1_epi32(kmask);
    const __m256i mask2 = _mm256_set1_epi32(km32);

    inline __m256i next8(uint32_t val) const {
        auto mval = _mm256_set1_epi32(val);
        auto mres = _mm256_add_epi32(_mm256_mullo_epi32(mval, mka), mkb);
        return _mm256_xor_si256(_mm256_and_si256(mres, mask1), mask2);
    }
};

static inline __m256 trellis_gen8(__m256i i8) {
    // split upper and lower bits of each 32-bit lane into two 8xfloat16 `hlo`, `hhi`
    __m256i low_16_bits_mask = _mm256_set1_epi32(0x0000FFFF);
    __m256i lower_halves_lanes32 = _mm256_and_si256(i8, low_16_bits_mask);
    __m256i upper_halves_lanes32 = _mm256_srli_epi32(i8, 16);
    // 00L0, 00L1, 00L2, 00L3, 00H0, 00H1, 00H2, 00H3, 00L4, 00L5, 00L6, 00L7, 00H4, 00H5, 00H6, 00H7
    auto iv = _mm256_packus_epi32(lower_halves_lanes32, upper_halves_lanes32);
    // 00L0, 00L1, 00L2, 00L3, 00L4, 00L5, 00L6, 00L7, 00H0, 00H1, 00H2, 00H3, 00H4, 00H5, 00H6, 00H7
    iv = _mm256_permute4x64_epi64(iv, 0xd8);
    auto fv1 = _mm256_cvtph_ps(_mm256_extracti128_si256(iv, 0));
    auto fv2 = _mm256_cvtph_ps(_mm256_extracti128_si256(iv, 1));
    return _mm256_add_ps(fv1, fv2);
}

struct Trellis2 {
    constexpr static uint32_t kmask = 0x8fff8fff;
    constexpr static uint32_t km32 = 0x3b603b60;
    constexpr static uint32_t ka = 89226354;
    constexpr static uint32_t kb = 64248484;
    constexpr static uint32_t ka1 = ka*ka;
    constexpr static uint32_t kb1 = kb*ka+kb;
    constexpr static uint32_t ka2 = ka1*ka;
    constexpr static uint32_t kb2 = kb1*ka+kb;
    constexpr static uint32_t ka3 = ka2*ka;
    constexpr static uint32_t kb3 = kb2*ka+kb;
    __m256i mka = _mm256_setr_epi32(ka, ka1, ka2, ka3, ka, ka1, ka2, ka3);
    __m256i mkb = _mm256_setr_epi32(kb, kb1, kb2, kb3, kb, kb1, kb2, kb3);
    const __m256i mask1 = _mm256_set1_epi32(kmask);
    const __m256i mask2 = _mm256_set1_epi32(km32);

    inline __m256i next8(uint32_t val1, uint32_t val2) {
        __m256i mval = MM256_SET_M128I(_mm_set1_epi32(val2), _mm_set1_epi32(val1));
        //__m256i mval = _mm256_setr_epi32(val1, val1, val1, val1, val2, val2, val2, val2);
        __m256i mres = _mm256_add_epi32(_mm256_mullo_epi32(mval, mka), mkb);
        return _mm256_xor_si256(_mm256_and_si256(mres, _mm256_set1_epi32(kmask)), _mm256_set1_epi32(km32));
    }
};

template <int nrc_y>
static void mul_mat_iq2_kt_F32_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis1 trellis;

    auto shifts = _mm_set_epi32(0, 0, 4, 0);
    auto values = _mm_loadu_si128((const __m128i *)iq4k_values);

    union { __m256 vec; float val[8]; } s_helper;

    constexpr int k_acc = nrc_y == 1 ? 2 : nrc_y;
    __m256  accd[k_acc];
    const float * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const float *)info.src1_row(iy);

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        const float d = *dptr * 31.75f * 1.05f;
        const block_iq2_kt * x = (const block_iq2_kt *)(dptr + 1);

        for (int iy = 0; iy < k_acc; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            const uint16_t * ql = (const uint16_t *)x[i].ql;
            auto s8 = _mm_set1_epi32(*(const uint32_t *)x[i].scales);
            s8 = _mm_and_si128(_mm_srlv_epi32(s8, shifts), _mm_set1_epi8(0xf));
            s8 = _mm_shuffle_epi8(values, s8);
            auto s32 = _mm256_cvtepi8_epi32(s8);
            s_helper.vec = _mm256_cvtepi32_ps(s32);
            for (int ib = 0; ib < QK_K/64; ++ib) {
                auto scale1 = _mm256_set1_ps(s_helper.val[2*ib+0]);
                auto scale2 = _mm256_set1_ps(s_helper.val[2*ib+1]);
                for (int j = 0; j < 4; ++j) {
                    auto xval1 = _mm256_mul_ps(scale1, trellis_gen8(trellis.next8(ql[8*ib+j+0]+4096)));
                    auto xval2 = _mm256_mul_ps(scale2, trellis_gen8(trellis.next8(ql[8*ib+j+4]+4096)));
                    if constexpr (nrc_y == 1) {
                        accd[0] = _mm256_fmadd_ps(_mm256_load_ps(y[0] + i*QK_K + 64*ib + 8*j +  0), xval1, accd[0]);
                        accd[1] = _mm256_fmadd_ps(_mm256_load_ps(y[0] + i*QK_K + 64*ib + 8*j + 32), xval2, accd[1]);
                    } else {
                        for (int iy = 0; iy < nrc_y; ++iy) {
                            accd[iy] = _mm256_fmadd_ps(_mm256_load_ps(y[iy] + i*QK_K + 64*ib + 8*j +  0), xval1, accd[iy]);
                            accd[iy] = _mm256_fmadd_ps(_mm256_load_ps(y[iy] + i*QK_K + 64*ib + 8*j + 32), xval2, accd[iy]);
                        }
                    }
                }
            }
        }

        if constexpr (nrc_y == 1) {
            __m256 res = _mm256_mul_ps(_mm256_set1_ps(d), _mm256_add_ps(accd[0], accd[1]));
            info.store(ix, 0, hsum_float_8(res));
        } else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                __m256 res = _mm256_mul_ps(_mm256_set1_ps(d), accd[iy]);
                info.store(ix, iy, hsum_float_8(res));
            }
        }
    }
}

static inline __m256 abs_ps(__m256 vals) {
    // Clear sign-bit of all the 32-bit floats in vals
    __m256 sign_bit = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(sign_bit, vals);
}

template <int nrc_y>
static void mul_mat_iq3_kt_F32_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis1 trellis;

    union { __m256 vec; float val[8]; } s_helper;

    auto shifts = _mm_set_epi32(0, 0, 4, 0);

    __m256i all_signs[4];
    auto mask1 = _mm256_set1_epi32(0x01);
    auto mask2 = _mm256_set1_epi32(0x10);

    __m256  accd[nrc_y];
    const float * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const float *)info.src1_row(iy);

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        const float d = *dptr * 31.75f * 1.015f;
        const block_iq3_kt * x = (const block_iq3_kt *)(dptr + 1);

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            const uint16_t * ql = (const uint16_t *)x[i].ql;
            const uint8_t * qh = x[i].qh;
            auto s8 = _mm_set1_epi32(*(const uint32_t *)x[i].scales);
            s8 = _mm_and_si128(_mm_srlv_epi32(s8, shifts), _mm_set1_epi8(0xf));
            auto s32 = _mm256_cvtepi8_epi32(s8);
            s_helper.vec = _mm256_cvtepi32_ps(s32);
            for (int j = 0; j < 4; ++j) all_signs[j] = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qh + 8*j)));
            for (int ib = 0; ib < 4; ++ib) {
                auto scale1 = _mm256_set1_ps(s_helper.val[ib+0]);
                auto scale2 = _mm256_set1_ps(s_helper.val[ib+4]);
                for (int j = 0; j < 4; ++j) {
                    uint32_t val1 = ql[4*ib+j   ] + 4096;
                    uint32_t val2 = ql[4*ib+j+16] + 4096;
                    auto sign1 = _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_and_si256(all_signs[j], mask1), mask1), _mm256_set1_epi32(0x80000000));
                    auto sign2 = _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_and_si256(all_signs[j], mask2), mask2), _mm256_set1_epi32(0x80000000));
                    all_signs[j] = _mm256_srli_epi32(all_signs[j], 1);
                    auto x_val1 = abs_ps(trellis_gen8(trellis.next8(val1)));
                    auto x_val2 = abs_ps(trellis_gen8(trellis.next8(val2)));
                    x_val1 = _mm256_mul_ps(scale1, _mm256_xor_ps(x_val1, _mm256_castsi256_ps(sign1)));
                    x_val2 = _mm256_mul_ps(scale2, _mm256_xor_ps(x_val2, _mm256_castsi256_ps(sign2)));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        accd[iy] = _mm256_fmadd_ps(_mm256_load_ps(y[iy] + i*QK_K+32*ib+8*j    ), x_val1, accd[iy]);
                        accd[iy] = _mm256_fmadd_ps(_mm256_load_ps(y[iy] + i*QK_K+32*ib+8*j+128), x_val2, accd[iy]);
                    }
                }
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            __m256 res = _mm256_mul_ps(_mm256_set1_ps(d), accd[iy]);
            info.store(ix, iy, hsum_float_8(res));
        }
    }
}

template <int nrc_y>
static void mul_mat_iq4_kt_F32_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;
    constexpr int kNumGroups = 64;

    Trellis2 trellis;

    union { __m256  vec; float    val[8]; } s_helper;
    union { __m256i vec; uint32_t val[8]; } o_helper;

    constexpr int k_acc = nrc_y == 1 ? 2 : nrc_y;

    __m256  accd[k_acc];
    const float * y[nrc_y];
    float row_sum[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = (const float *)info.src1_row(iy);
        auto sum = _mm256_setzero_ps();
        for (int i = 0; i < n/8; ++i) sum = _mm256_add_ps(sum, _mm256_loadu_ps(y[iy] + 8*i));
        row_sum[iy] = hsum_float_8(sum);
    }

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        auto d = _mm256_set1_ps(dptr[0] * 31.75f * 1.01f);
        auto dav = dptr[1];
        const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 2);

        for (int iy = 0; iy < k_acc; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            auto vshb = _mm256_loadu_si256((const __m256i *)x[i].qs);
            const uint32_t * shb = x[i].qs;
            const uint8_t * ql = (const uint8_t *)(shb + 8);
            const uint8_t * qh = ql + kNumGroups;
            auto iscales = _mm256_srli_epi32(_mm256_and_si256(vshb, _mm256_set1_epi32(0xff)), 1);
            s_helper.vec = _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_sub_epi32(iscales, _mm256_set1_epi32(64))));
            o_helper.vec = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(vshb, _mm256_set1_epi32(1)), 15), _mm256_set1_epi32(4096));
            for (int ib = 0; ib < 4; ++ib) {
                auto scale1 = _mm256_set1_ps(s_helper.val[ib+0]);
                auto scale2 = _mm256_set1_ps(s_helper.val[ib+4]);
                for (int j = 0; j < 4; ++j) {
                    const uint32_t sh1 = shb[ib+0] >> (8 + 6*j);
                    const uint32_t sh2 = shb[ib+4] >> (8 + 6*j);
                    uint32_t val1 = ql[8*ib+2*j+ 0] + ((qh[8*ib+2*j+0] << 8) & 0xf00) + ((sh1 & 7) << 12) + o_helper.val[ib+0];
                    uint32_t val2 = ql[8*ib+2*j+32] + ((qh[8*ib+2*j+0] << 4) & 0xf00) + ((sh2 & 7) << 12) + o_helper.val[ib+4];
                    uint32_t val3 = ql[8*ib+2*j+ 1] + ((qh[8*ib+2*j+1] << 8) & 0xf00) + ((sh1 & 56) << 9) + o_helper.val[ib+0];
                    uint32_t val4 = ql[8*ib+2*j+33] + ((qh[8*ib+2*j+1] << 4) & 0xf00) + ((sh2 & 56) << 9) + o_helper.val[ib+4];
                    auto x_val1 = _mm256_mul_ps(scale1, trellis_gen8(trellis.next8(val1, val3)));
                    auto x_val2 = _mm256_mul_ps(scale2, trellis_gen8(trellis.next8(val2, val4)));
                    if constexpr (nrc_y == 1) {
                        auto y1 = _mm256_load_ps(y[0] + i*QK_K+32*ib+8*j+  0);
                        auto y2 = _mm256_load_ps(y[0] + i*QK_K+32*ib+8*j+128);
                        accd[0] = _mm256_fmadd_ps(y1, x_val1, accd[0]);
                        accd[1] = _mm256_fmadd_ps(y2, x_val2, accd[1]);
                    } else {
                        for (int iy = 0; iy < nrc_y; ++iy) {
                            auto y1 = _mm256_load_ps(y[iy] + i*QK_K+32*ib+8*j+  0);
                            auto y2 = _mm256_load_ps(y[iy] + i*QK_K+32*ib+8*j+128);
                            accd[iy] = _mm256_fmadd_ps(y1, x_val1, accd[iy]);
                            accd[iy] = _mm256_fmadd_ps(y2, x_val2, accd[iy]);
                        }
                    }
                }
            }
        }

        if constexpr (nrc_y == 1) {
            info.store(ix, 0, hsum_float_8(_mm256_add_ps(accd[0], accd[1])) + dav*row_sum[0]);
        } else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                info.store(ix, iy, hsum_float_8(accd[iy]) + dav*row_sum[iy]);
            }
        }
    }
}

} // namespace

bool iqk_set_kernels_ktquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {

    if (ne00%QK_K != 0 || ggml_type(typeB) != GGML_TYPE_F32) {
        return false;
    }

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_IQ2_KT:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_kt_F32_T, kernels);
            break;
        case GGML_TYPE_IQ3_KT:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq3_kt_F32_T, kernels);
            break;
        case GGML_TYPE_IQ4_KT:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_kt_F32_T, kernels);
            break;
        default:
            return false;
    }

    return true;

}

#else // !__x86_64__

namespace {

struct Trellis1 {
    constexpr static uint32_t kmask = 0x8fff8fff;
    constexpr static uint32_t km32 = 0x3b603b60;
    constexpr static uint32_t ka = 89226354;
    constexpr static uint32_t kb = 64248484;
    constexpr static uint32_t ka1 = ka*ka;
    constexpr static uint32_t kb1 = kb*ka+kb;
    constexpr static uint32_t ka2 = ka1*ka;
    constexpr static uint32_t kb2 = kb1*ka+kb;
    constexpr static uint32_t ka3 = ka2*ka;
    constexpr static uint32_t kb3 = kb2*ka+kb;
    constexpr static uint32_t ka4 = ka3*ka;
    constexpr static uint32_t kb4 = kb3*ka+kb;
    constexpr static uint32_t ka5 = ka4*ka;
    constexpr static uint32_t kb5 = kb4*ka+kb;
    constexpr static uint32_t ka6 = ka5*ka;
    constexpr static uint32_t kb6 = kb5*ka+kb;
    constexpr static uint32_t ka7 = ka6*ka;
    constexpr static uint32_t kb7 = kb6*ka+kb;
    const uint32x4x2_t mka = {uint32x4_t{ka, ka1, ka2, ka3}, uint32x4_t{ka4, ka5, ka6, ka7}};
    const uint32x4x2_t mkb = {uint32x4_t{kb, kb1, kb2, kb3}, uint32x4_t{kb4, kb5, kb6, kb7}};
    const uint32x4_t mask1 = vdupq_n_u32(kmask);
    const uint32x4_t mask2 = vdupq_n_u32(km32);

    inline uint32x4x2_t next8(uint32_t val) const {
        auto mval = vdupq_n_u32(val);
        uint32x4x2_t mres;
        // This does not seem to be faster
        //mres.val[0] = vmlaq_u32(mkb.val[0], mka.val[0], mval);
        //mres.val[1] = vmlaq_u32(mkb.val[1], mka.val[1], mval);
        mres.val[0] = vaddq_u32(vmulq_u32(mval, mka.val[0]), mkb.val[0]);
        mres.val[1] = vaddq_u32(vmulq_u32(mval, mka.val[1]), mkb.val[1]);
        mres.val[0] = veorq_u32(vandq_u32(mres.val[0], mask1), mask2);
        mres.val[1] = veorq_u32(vandq_u32(mres.val[1], mask1), mask2);
        return mres;
    }
    inline uint32x4x2_t next8(uint32_t val1, uint32_t val2) const {
        auto mval1 = vdupq_n_u32(val1);
        auto mval2 = vdupq_n_u32(val2);
        uint32x4x2_t mres;
        // This does not seem to be faster
        //mres.val[0] = vmlaq_u32(mkb.val[0], mka.val[0], mval1);
        //mres.val[1] = vmlaq_u32(mkb.val[0], mka.val[0], mval2);
        mres.val[0] = vaddq_u32(vmulq_u32(mval1, mka.val[0]), mkb.val[0]);
        mres.val[1] = vaddq_u32(vmulq_u32(mval2, mka.val[0]), mkb.val[0]);
        mres.val[0] = veorq_u32(vandq_u32(mres.val[0], mask1), mask2);
        mres.val[1] = veorq_u32(vandq_u32(mres.val[1], mask1), mask2);
        return mres;
    }
    static inline float16x8_t gen8(const uint32x4x2_t& i8) {
        auto fv1 = vreinterpretq_f16_u32(i8.val[0]);
        auto fv2 = vreinterpretq_f16_u32(i8.val[1]);
        return vpaddq_f16(fv1, fv2);
    }
    inline float16x8_t gen8(uint32_t val) const { return gen8(next8(val)); }
    inline float16x8_t gen8(uint32_t val1, uint32_t val2) const { return gen8(next8(val1, val2)); }
    inline float32x4x2_t gen8_f32(uint32_t val1, uint32_t val2) const {
        auto x16 = gen8(val1, val2);
        return { vcvt_f32_f16(vget_low_f16(x16)), vcvt_f32_f16(vget_high_f16(x16)) };
    }
    inline float32x4x2_t gen8_f32(uint32_t val1, uint32_t val2, float16x8_t scale) const {
        auto x16 = vmulq_f16(gen8(val1, val2), scale);
        return { vcvt_f32_f16(vget_low_f16(x16)), vcvt_f32_f16(vget_high_f16(x16)) };
    }
};

template <int nrc_y>
static void mul_mat_iq2_kt_F16_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis1 trellis;

    auto values = vld1q_s8(iq4k_values);

    union { float16x8_t vec; float16_t val[8]; } s_helper;

    constexpr int k_acc = nrc_y == 1 ? 2 : nrc_y;
    float16x8_t accd[k_acc];
    const float16_t * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const float16_t *)info.src1_row(iy);

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        const float d = *dptr * 31.75f * 1.05f;
        const block_iq2_kt * x = (const block_iq2_kt *)(dptr + 1);

        for (int iy = 0; iy < k_acc; ++iy) accd[iy] = vdupq_n_f16(0);

        for (int i = 0; i < nb; ++i) {
            const uint16_t * ql = (const uint16_t *)x[i].ql;
            auto u32 = *(const uint32_t *)x[i].scales;
            auto s8_u32 = uint32x2_t{u32, u32 >> 4};
            s8_u32 = vand_u8(s8_u32, vdup_n_u32(0x0f0f0f0f));
            auto s8 = vqtbl1_s8(values, vreinterpret_u8_u32(s8_u32));
            auto s16 = vmovl_s8(s8);
            s_helper.vec = vcvtq_f16_s16(s16);
            for (int ib = 0; ib < QK_K/64; ++ib) {
                auto scale1 = vdupq_n_f16(s_helper.val[2*ib+0]);
                auto scale2 = vdupq_n_f16(s_helper.val[2*ib+1]);
                for (int j = 0; j < 4; ++j) {
                    auto xval1 = vmulq_f16(scale1, trellis.gen8(ql[8*ib+j+0]+4096));
                    auto xval2 = vmulq_f16(scale2, trellis.gen8(ql[8*ib+j+4]+4096));
                    if constexpr (nrc_y == 1) {
                        accd[0] = vfmaq_f16(accd[0], xval1, vld1q_f16(y[0] + i*QK_K + 64*ib + 8*j +  0));
                        accd[1] = vfmaq_f16(accd[1], xval2, vld1q_f16(y[0] + i*QK_K + 64*ib + 8*j + 32));
                    } else {
                        for (int iy = 0; iy < nrc_y; ++iy) {
                            accd[iy] = vfmaq_f16(accd[iy], xval1, vld1q_f16(y[iy] + i*QK_K + 64*ib + 8*j +  0));
                            accd[iy] = vfmaq_f16(accd[iy], xval2, vld1q_f16(y[iy] + i*QK_K + 64*ib + 8*j + 32));
                        }
                    }
                }
            }
        }

        if constexpr (nrc_y == 1) {
            auto res16 = vpaddq_f16(accd[0], accd[1]);
            auto res = vaddq_f32(vcvt_f32_f16(vget_low_f16(res16)), vcvt_f32_f16(vget_high_f16(res16)));
            info.store(ix, 0, vaddvq_f32(res)*d);
        } else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto res = vaddq_f32(vcvt_f32_f16(vget_low_f16(accd[iy])), vcvt_f32_f16(vget_high_f16(accd[iy])));
                info.store(ix, iy, vaddvq_f32(res)*d);
            }
        }
    }
}

template <int nrc_y>
static void mul_mat_iq3_kt_F16_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis1 trellis;

    union { float16x8_t vec; float16_t val[8]; } s_helper;

    uint16x8_t all_signs[4];
    auto mask1 = vdupq_n_u16(0x01);
    auto mask2 = vdupq_n_u16(0x10);

    float16x8_t accd[nrc_y];
    const float16_t * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const float16_t *)info.src1_row(iy);

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        const float d = *dptr * 31.75f * 1.015f;
        const block_iq3_kt * x = (const block_iq3_kt *)(dptr + 1);

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = vdupq_n_f16(0);

        for (int i = 0; i < nb; ++i) {
            const uint16_t * ql = (const uint16_t *)x[i].ql;
            const uint8_t * qh = x[i].qh;
            auto u32 = *(const uint32_t *)x[i].scales;
            auto s8_u32 = uint32x2_t{u32, u32 >> 4};
            s8_u32 = vand_u8(s8_u32, vdup_n_u32(0x0f0f0f0f));
            auto s16 = vmovl_s8(vreinterpret_s8_u32(s8_u32));
            s_helper.vec = vcvtq_f16_s16(s16);
            for (int j = 0; j < 4; ++j) all_signs[j] = vmovl_u8(vld1_u8(qh + 8*j));
            for (int ib = 0; ib < 4; ++ib) {
                auto scale1 = vdupq_n_f16(s_helper.val[ib+0]);
                auto scale2 = vdupq_n_f16(s_helper.val[ib+4]);
                for (int j = 0; j < 4; ++j) {
                    uint32_t val1 = ql[4*ib+j   ] + 4096;
                    uint32_t val2 = ql[4*ib+j+16] + 4096;
                    auto sign1 = vshlq_n_u16(vandq_u16(all_signs[j], mask1), 15);
                    auto sign2 = vshlq_n_u16(vandq_u16(all_signs[j], mask2), 11);
                    all_signs[j] = vshrq_n_u16(all_signs[j], 1);
                    auto x_val1 = vabsq_f16(trellis.gen8(val1));
                    auto x_val2 = vabsq_f16(trellis.gen8(val2));
                    x_val1 = vmulq_f16(scale1, vreinterpretq_f16_u16(vorrq_u16(vreinterpretq_u16_f16(x_val1), sign1)));
                    x_val2 = vmulq_f16(scale2, vreinterpretq_f16_u16(vorrq_u16(vreinterpretq_u16_f16(x_val2), sign2)));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        accd[iy] = vfmaq_f16(accd[iy], x_val1, vld1q_f16(y[iy] + i*QK_K+32*ib+8*j    ));
                        accd[iy] = vfmaq_f16(accd[iy], x_val2, vld1q_f16(y[iy] + i*QK_K+32*ib+8*j+128));
                    }
                }
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            auto res = vaddq_f32(vcvt_f32_f16(vget_low_f16(accd[iy])), vcvt_f32_f16(vget_high_f16(accd[iy])));
            info.store(ix, iy, d*vaddvq_f32(res));
        }
    }
}

template <int nrc_y>
static void mul_mat_iq4_kt_F16_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;
    constexpr int kNumGroups = 64;

    Trellis1 trellis;

    union { float16x8_t vec; float16_t val[8]; } s_helper;
    union { uint16x8_t  vec; uint16_t  val[8]; } o_helper;

    constexpr int k_acc = nrc_y == 1 ? 2 : nrc_y;

    float16x8_t accd[k_acc];
    const float16_t * y[nrc_y];
    float row_sum[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = (const float16_t *)info.src1_row(iy);
        auto sum = vdupq_n_f16(0);
        for (int i = 0; i < n/8; ++i) sum = vaddq_f16(sum, vld1q_f16(y[iy] + 8*i));
        auto sum32 = vaddq_f32(vcvt_f32_f16(vget_low_f16(sum)), vcvt_f32_f16(vget_high_f16(sum)));
        //auto sum32 = vdupq_n_f32(0);
        //for (int i = 0; i < n/4; ++i) sum32 = vaddq_f32(sum32, vcvt_f32_f16(vld1_f16(y[iy] + 4*i)));
        row_sum[iy] = vaddvq_f32(sum32);
    }

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        auto d = dptr[0] * 31.75f * 1.01f;
        auto dav = dptr[1];
        const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 2);

        for (int iy = 0; iy < k_acc; ++iy) accd[iy] = vdupq_n_f16(0);

        for (int i = 0; i < nb; ++i) {
            const uint32_t * shb = x[i].qs;
            auto vshb = vld1q_u32_x2(shb);
            auto vshb16 = vcombine_u16(vmovn_u32(vandq_u32(vshb.val[0], vdupq_n_u32(0xff))), vmovn_u32(vandq_u32(vshb.val[1], vdupq_n_u32(0xff))));
            const uint8_t * ql = (const uint8_t *)(shb + 8);
            const uint8_t * qh = ql + kNumGroups;
            auto iscales = vsubq_s16(vreinterpretq_s16_u16(vshrq_n_u16(vshb16, 1)), vdupq_n_s16(64));
            s_helper.vec = vcvtq_f16_s16(iscales);
            o_helper.vec = vaddq_u16(vshlq_n_u16(vandq_u16(vshb16, vdupq_n_u16(1)), 15), vdupq_n_u16(4096));
            for (int ib = 0; ib < 4; ++ib) {
                auto scale1 = vdupq_n_f16(s_helper.val[ib+0]);
                auto scale2 = vdupq_n_f16(s_helper.val[ib+4]);
                for (int j = 0; j < 4; ++j) {
                    const uint32_t sh1 = shb[ib+0] >> (8 + 6*j);
                    const uint32_t sh2 = shb[ib+4] >> (8 + 6*j);
                    uint32_t val1 = ql[8*ib+2*j+ 0] + ((qh[8*ib+2*j+0] << 8) & 0xf00) + ((sh1 & 7) << 12) + o_helper.val[ib+0];
                    uint32_t val2 = ql[8*ib+2*j+32] + ((qh[8*ib+2*j+0] << 4) & 0xf00) + ((sh2 & 7) << 12) + o_helper.val[ib+4];
                    uint32_t val3 = ql[8*ib+2*j+ 1] + ((qh[8*ib+2*j+1] << 8) & 0xf00) + ((sh1 & 56) << 9) + o_helper.val[ib+0];
                    uint32_t val4 = ql[8*ib+2*j+33] + ((qh[8*ib+2*j+1] << 4) & 0xf00) + ((sh2 & 56) << 9) + o_helper.val[ib+4];
                    auto x_val1 = vmulq_f16(scale1, trellis.gen8(val1, val3));
                    auto x_val2 = vmulq_f16(scale2, trellis.gen8(val2, val4));
                    if constexpr (nrc_y == 1) {
                        auto y1 = vld1q_f16(y[0] + i*QK_K+32*ib+8*j+  0);
                        auto y2 = vld1q_f16(y[0] + i*QK_K+32*ib+8*j+128);
                        accd[0] = vfmaq_f16(accd[0], y1, x_val1);
                        accd[1] = vfmaq_f16(accd[1], y2, x_val2);
                    } else {
                        for (int iy = 0; iy < nrc_y; ++iy) {
                            auto y1 = vld1q_f16(y[iy] + i*QK_K+32*ib+8*j+  0);
                            auto y2 = vld1q_f16(y[iy] + i*QK_K+32*ib+8*j+128);
                            accd[iy] = vfmaq_f16(accd[iy], y1, x_val1);
                            accd[iy] = vfmaq_f16(accd[iy], y2, x_val2);
                        }
                    }
                }
            }
        }

        if constexpr (nrc_y == 1) {
            auto sum16 = vaddq_f16(accd[0], accd[1]);
            auto sum = vaddq_f32(vcvt_f32_f16(vget_low_f16(sum16)), vcvt_f32_f16(vget_high_f16(sum16)));
            info.store(ix, 0, d*vaddvq_f32(sum) + dav*row_sum[0]);
        } else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto sum = vaddq_f32(vcvt_f32_f16(vget_low_f16(accd[iy])), vcvt_f32_f16(vget_high_f16(accd[iy])));
                info.store(ix, iy, d*vaddvq_f32(sum) + dav*row_sum[iy]);
            }
        }
    }
}

template <int nrc_y>
static void mul_mat_iq4_kt_F32_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;
    constexpr int kNumGroups = 64;

    Trellis1 trellis;

    float32x4_t accd[nrc_y * 2];
    const float * y[nrc_y];
    float row_sum[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = (const float *)info.src1_row(iy);
        auto sum = vdupq_n_f32(0);
        for (int i = 0; i < n/4; ++i) sum = vaddq_f32(sum, vld1q_f32(y[iy] + 4*i));
        row_sum[iy] = vaddvq_f32(sum);
    }

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        const float d = dptr[0] * 31.75f * 1.01f;
        const float row_av = dptr[1];
        const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 2);

        for (int iy = 0; iy < nrc_y * 2; ++iy) accd[iy] = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; ++i) {
            const uint32_t * shb = x[i].qs;
            const uint8_t * ql = (const uint8_t *)(shb + 8);
            const uint8_t * qh = ql + kNumGroups;

            for (int ib = 0; ib < 4; ++ib) {
                const uint16_t x_scale1 = (int16_t)((shb[ib+0] & 0xff) >> 1) - 64;
                const uint16_t x_scale2 = (int16_t)((shb[ib+4] & 0xff) >> 1) - 64;
                const float16x8_t scale1 = vcvtq_f16_s16(vdupq_n_s16(x_scale1));
                const float16x8_t scale2 = vcvtq_f16_s16(vdupq_n_s16(x_scale2));
                const uint32_t offset1 = 4096 + ((shb[ib+0] & 1) << 15);
                const uint32_t offset2 = 4096 + ((shb[ib+4] & 1) << 15);

                uint32_t sh1 = shb[ib+0] >> 8;
                uint32_t sh2 = shb[ib+4] >> 8;

                for (int j = 0; j < 4; ++j) {

                    uint32_t val1 = ql[8*ib+2*j+ 0] + ((qh[8*ib+2*j+0] << 8) & 0xf00) + ((sh1 & 7) << 12) + offset1;
                    uint32_t val2 = ql[8*ib+2*j+32] + ((qh[8*ib+2*j+0] << 4) & 0xf00) + ((sh2 & 7) << 12) + offset2;
                    uint32_t val3 = ql[8*ib+2*j+ 1] + ((qh[8*ib+2*j+1] << 8) & 0xf00) + ((sh1 & 56) << 9) + offset1;
                    uint32_t val4 = ql[8*ib+2*j+33] + ((qh[8*ib+2*j+1] << 4) & 0xf00) + ((sh2 & 56) << 9) + offset2;

                    sh1 >>= 6;
                    sh2 >>= 6;

                    auto x1 = trellis.gen8_f32(val1, val3, scale1);
                    auto x2 = trellis.gen8_f32(val2, val4, scale2);

                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y1 = vld1q_f32_x2(y[iy] + i*QK_K + 32*ib + 8*j);
                        auto y2 = vld1q_f32_x2(y[iy] + i*QK_K + 32*ib + 8*j + 128);

                        accd[iy*2 + 0] = vfmaq_f32(accd[iy*2 + 0], y1.val[0], x1.val[0]);
                        accd[iy*2 + 1] = vfmaq_f32(accd[iy*2 + 1], y1.val[1], x1.val[1]);
                        accd[iy*2 + 0] = vfmaq_f32(accd[iy*2 + 0], y2.val[0], x2.val[0]);
                        accd[iy*2 + 1] = vfmaq_f32(accd[iy*2 + 1], y2.val[1], x2.val[1]);
                    }
                }

            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            float32x4_t sum1 = vaddq_f32(accd[iy*2], accd[iy*2 + 1]);
            float result = d*vaddvq_f32(sum1) + row_av*row_sum[iy];
            info.store(ix, iy, result);
        }
    }
}

}

bool iqk_set_kernels_ktquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {

    if (ne00%QK_K == 0 && ggml_type(typeB) == GGML_TYPE_F32 && ggml_type(typeA) == GGML_TYPE_IQ4_KT) {
        IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_kt_F32_T, kernels);
        func16 = nullptr;
        return true;
    }

    if (ne00%QK_K != 0 || ggml_type(typeB) != GGML_TYPE_F16) {
        return false;
    }

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_IQ2_KT:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_kt_F16_T, kernels);
            break;
        case GGML_TYPE_IQ3_KT:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq3_kt_F16_T, kernels);
            break;
        case GGML_TYPE_IQ4_KT:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_kt_F16_T, kernels);
            break;
        default:
            return false;
    }

    return true;
}

#endif

#endif
