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

static inline __m256i trellis_next8(uint32_t val) {
    constexpr uint32_t kmask = 0x8fff8fff;
    constexpr uint32_t km32 = 0x3b603b60;
    constexpr uint32_t ka = 89226354;
    constexpr uint32_t kb = 64248484;
    constexpr uint32_t ka1 = ka*ka;
    constexpr uint32_t kb1 = kb*ka+kb;
    constexpr uint32_t ka2 = ka1*ka;
    constexpr uint32_t kb2 = kb1*ka+kb;
    constexpr uint32_t ka3 = ka2*ka;
    constexpr uint32_t kb3 = kb2*ka+kb;
    constexpr uint32_t ka4 = ka3*ka;
    constexpr uint32_t kb4 = kb3*ka+kb;
    constexpr uint32_t ka5 = ka4*ka;
    constexpr uint32_t kb5 = kb4*ka+kb;
    constexpr uint32_t ka6 = ka5*ka;
    constexpr uint32_t kb6 = kb5*ka+kb;
    constexpr uint32_t ka7 = ka6*ka;
    constexpr uint32_t kb7 = kb6*ka+kb;
    __m256i mka = _mm256_setr_epi32(ka, ka1, ka2, ka3, ka4, ka5, ka6, ka7);
    __m256i mkb = _mm256_setr_epi32(kb, kb1, kb2, kb3, kb4, kb5, kb6, kb7);
    __m256i mval = _mm256_set1_epi32(val);
    __m256i mres = _mm256_add_epi32(_mm256_mullo_epi32(mval, mka), mkb);
    return _mm256_and_si256(mres, _mm256_set1_epi32(kmask)) ^ _mm256_set1_epi32(km32);
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
        return _mm256_and_si256(mres, mask1) ^ mask2;
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
        __m256i mval = _mm256_setr_epi32(val1, val1, val1, val1, val2, val2, val2, val2);
        __m256i mres = _mm256_add_epi32(_mm256_mullo_epi32(mval, mka), mkb);
        return _mm256_and_si256(mres, _mm256_set1_epi32(kmask)) ^ _mm256_set1_epi32(km32);
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

// Negates 32-bit float lanes of an 8x32-bit vector
// based on 8x8-bit condition var. For float lane i, if byte i of
// `condition` is nonzero, the float will be negated.
static inline __m256 conditional_negate_ps(__m256 vals, __m128i condition_bytes) {
    // Make `should_negate_byte_mask` where byte i == 0xFF if byte i in condition_bytes is zero,
    // else 0x00 (upper bytes are meaningless)
    __m128i zeros = _mm_setzero_si128();
    __m128i is_zero_byte_mask = _mm_cmpeq_epi8(condition_bytes, zeros);
    __m128i should_negate_byte_mask = _mm_cmpeq_epi8(is_zero_byte_mask, zeros);
    // Widen lower 8x8 bits of `should_negate_byte_mask` to 8x32 bits by padding zeros
    // expanded_mask_epi32[j] will be 0x000000FF if vals[j] should be negated, zero otherwise
    __m256i expanded_mask_epi32 = _mm256_cvtepu8_epi32(should_negate_byte_mask);
    // Same as above but with all 32 bits of lane j set if vals[j] should be negated (use to make XOR mask)
    __m256i full_dword_negate_mask = _mm256_cmpgt_epi32(expanded_mask_epi32, _mm256_setzero_si256());
    // Negate via XOR on sign bits of each 32-bit float
    __m256i sign_bit_pattern = _mm256_set1_epi32(0x80000000); // MSB set for a 32-bit value
    __m256i xor_mask_epi32 = _mm256_and_si256(full_dword_negate_mask, sign_bit_pattern);
    __m256 xor_mask_ps = _mm256_castsi256_ps(xor_mask_epi32);
    return _mm256_xor_ps(vals, xor_mask_ps);
}

template <int nrc_y>
static void mul_mat_iq3_kt_F32_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis1 trellis;

    union { __m256 vec; float val[8]; } s_helper;

    auto shifts = _mm_set_epi32(0, 0, 4, 0);
    //auto sign_mask1 = _mm256_set1_epi32(0x01);
    //auto sign_mask2 = _mm256_set1_epi32(0x10);
    //auto sign_bit   = _mm256_set1_ps(-0.0f);

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
            //auto mask1 = _mm_set1_epi8(1);
            //auto mask2 = _mm_slli_epi16(mask1, 4);
            for (int j = 0; j < 4; ++j) all_signs[j] = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qh + 8*j)));
            for (int ib = 0; ib < 4; ++ib) {
                //auto sign_bits = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(x[i].qh + 8*ib)));
                auto scale1 = _mm256_set1_ps(s_helper.val[ib+0]);
                auto scale2 = _mm256_set1_ps(s_helper.val[ib+4]);
                //uint64_t mask1 = 0x0101010101010101 << ib; //(j/32);
                //uint64_t mask2 = mask1 << 4;
                for (int j = 0; j < 4; ++j) {
                    uint32_t val1 = ql[4*ib+j   ] + 4096;
                    uint32_t val2 = ql[4*ib+j+16] + 4096;
                    auto sign1 = _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_and_si256(all_signs[j], mask1), mask1), _mm256_set1_epi32(0x80000000));
                    auto sign2 = _mm256_and_si256(_mm256_cmpeq_epi32(_mm256_and_si256(all_signs[j], mask2), mask2), _mm256_set1_epi32(0x80000000));
                    all_signs[j] = _mm256_srli_epi32(all_signs[j], 1);
                    //auto signs = _mm_loadl_epi64((const __m128i *)(qh + j));
                    //auto sign1 = _mm_and_si128(signs, mask1);
                    //auto sign2 = _mm_and_si128(signs, mask2);
                    //const uint64_t signs = *((const uint64_t *)(qh + j));
                    auto x_val1 = abs_ps(trellis_gen8(trellis.next8(val1)));
                    auto x_val2 = abs_ps(trellis_gen8(trellis.next8(val2)));
                    x_val1 = _mm256_mul_ps(scale1, _mm256_xor_ps(x_val1, _mm256_castsi256_ps(sign1)));
                    x_val2 = _mm256_mul_ps(scale2, _mm256_xor_ps(x_val2, _mm256_castsi256_ps(sign2)));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        accd[iy] = _mm256_fmadd_ps(_mm256_load_ps(y[iy] + i*QK_K+32*ib+8*j    ), x_val1, accd[iy]);
                        accd[iy] = _mm256_fmadd_ps(_mm256_load_ps(y[iy] + i*QK_K+32*ib+8*j+128), x_val2, accd[iy]);
                    }
                }
                //mask1 = _mm_slli_epi16(mask1, 1);
                //mask2 = _mm_slli_epi16(mask2, 1);
                //for (int j = 0; j < 4; ++j) {
                //    //auto signs1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(sign_bits, sign_mask1), 27));
                //    //auto signs2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_and_si256(sign_bits, sign_mask2), 23));
                //    //sign_bits   = _mm256_srli_epi32(sign_bits, 1);
                //    auto smask1 = _mm256_cmpeq_epi32(_mm256_and_si256(sign_bits, sign_mask1), sign_mask1);
                //    auto smask2 = _mm256_cmpeq_epi32(_mm256_and_si256(sign_bits, sign_mask2), sign_mask2);
                //    sign_bits   = _mm256_srli_epi32(sign_bits, 1);
                //    auto signs1 = _mm256_and_ps(_mm256_castsi256_ps(smask1), sign_bit);
                //    auto signs2 = _mm256_and_ps(_mm256_castsi256_ps(smask2), sign_bit);
                //    auto a_val1 = _mm256_andnot_ps(sign_bit, trellis_gen8(trellis.next8(ql[4*ib+j+ 0]+4096)));
                //    auto a_val2 = _mm256_andnot_ps(sign_bit, trellis_gen8(trellis.next8(ql[4*ib+j+16]+4096)));
                //    auto x_val1 = _mm256_mul_ps(scale1, _mm256_xor_ps(a_val1, signs1));
                //    auto x_val2 = _mm256_mul_ps(scale2, _mm256_xor_ps(a_val2, signs2));
                //    for (int iy = 0; iy < nrc_y; ++iy) {
                //        accd[iy] = _mm256_fmadd_ps(_mm256_loadu_ps(y[iy] + i*QK_K + 32*ib + 8*j +   0), x_val1, accd[iy]);
                //        accd[iy] = _mm256_fmadd_ps(_mm256_loadu_ps(y[iy] + i*QK_K + 32*ib + 8*j + 128), x_val2, accd[iy]);
                //    }
                //}
            }
            //for (int j = 0; j < 128; j+=8) {
            //    uint64_t mask1 = 0x0101010101010101 << (j/32);
            //    uint64_t mask2 = mask1 << 4;
            //    uint32_t val1 = ql[j/8] + 4096;
            //    uint32_t val2 = ql[j/8+16] + 4096;
            //    const uint64_t signs = *((const uint64_t *)(qh + (j%32)));
            //    const float x_scale1 = (x[i].scales[j/32] & 0xf);
            //    const float x_scale2 = (x[i].scales[j/32] >> 4);
            //    const __m256 x_val1 = abs_ps(trellis_gen8(trellis.next8(val1)));
            //    const __m256 x_val2 = abs_ps(trellis_gen8(trellis.next8(val2)));
            //    for (int iy = 0; iy < nrc_y; ++iy) {
            //        accd[iy] = _mm256_fmadd_ps(
            //            conditional_negate_ps(
            //                _mm256_load_ps(y[iy] + i*QK_K+j), signs & mask1
            //            ),
            //            _mm256_mul_ps(_mm256_set1_ps(x_scale1), x_val1),
            //            accd[iy]
            //        );
            //        accd[iy] = _mm256_fmadd_ps(
            //            conditional_negate_ps(
            //                _mm256_load_ps(y[iy] + i*QK_K+j+128), signs & mask2
            //            ),
            //            _mm256_mul_ps(_mm256_set1_ps(x_scale2), x_val2),
            //            accd[iy]
            //        );
            //    }
            //}
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

    __m256  accd[nrc_y];
    __m256  accd2[nrc_y];
    const float * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const float *)info.src1_row(iy);

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        const float d = dptr[0] * 31.75f * 1.01f;
        const float row_av = dptr[1];
        const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 2);

        for (int iy = 0; iy < nrc_y; ++iy) {
            accd[iy] = _mm256_setzero_ps();
            accd2[iy] = _mm256_setzero_ps();
        }

        for (int i = 0; i < nb; ++i) {
            const uint32_t * shb = x[i].qs;
            const uint8_t * ql = (const uint8_t *)(shb + 8);
            const uint8_t * qh = ql + kNumGroups;
            for (int j = 0; j < 128; j+=8) {
                const uint32_t offset1 = 4096 + ((shb[j/32+0] & 1) << 15);
                const uint32_t offset2 = 4096 + ((shb[j/32+4] & 1) << 15);
                const float x_scale1 = (int)((shb[j/32+0] & 0xff) >> 1) - 64;
                const float x_scale2 = (int)((shb[j/32+4] & 0xff) >> 1) - 64;
                const uint32_t sh1 = shb[j/32+0] >> (8 + 6*((j/8)%4));
                const uint32_t sh2 = shb[j/32+4] >> (8 + 6*((j/8)%4));
                uint32_t val1 = ql[j/4+ 0] + ((qh[j/4+0] << 8) & 0xf00) + ((sh1 & 7) << 12) + offset1;
                uint32_t val2 = ql[j/4+32] + ((qh[j/4+0] << 4) & 0xf00) + ((sh2 & 7) << 12) + offset2;
                uint32_t val3 = ql[j/4+ 1] + ((qh[j/4+1] << 8) & 0xf00) + ((sh1 & 56) << 9) + offset1;
                uint32_t val4 = ql[j/4+33] + ((qh[j/4+1] << 4) & 0xf00) + ((sh2 & 56) << 9) + offset2;
                const __m256 x_val1 = trellis_gen8(trellis.next8(val1, val3));
                const __m256 x_val2 = trellis_gen8(trellis.next8(val2, val4));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    accd[iy] = _mm256_fmadd_ps(
                        _mm256_load_ps(y[iy] + i*QK_K+j), 
                        _mm256_mul_ps(_mm256_set1_ps(x_scale1), x_val1),
                        accd[iy]
                    );
                    accd[iy] = _mm256_fmadd_ps(
                        _mm256_load_ps(y[iy] + i*QK_K+j+128), 
                        _mm256_mul_ps(_mm256_set1_ps(x_scale2), x_val2),
                        accd[iy]
                    );
                    accd2[iy] = _mm256_add_ps(
                        _mm256_load_ps(y[iy] + i*QK_K+j),
                        accd2[iy]
                    );
                    accd2[iy] = _mm256_add_ps(
                        _mm256_load_ps(y[iy] + i*QK_K+j+128),
                        accd2[iy]
                    );
                }
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            __m256 res = _mm256_mul_ps(_mm256_set1_ps(d), accd[iy]);
            __m256 res2 = _mm256_mul_ps(_mm256_set1_ps(row_av), accd2[iy]);
            info.store(ix, iy, hsum_float_8(res) + hsum_float_8(res2));
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
            assert (ne00 % QK_K == 0);
            kernels[0] = mul_mat_iq2_kt_F32_T<1>;
            kernels[1] = mul_mat_iq2_kt_F32_T<2>;
            kernels[2] = mul_mat_iq2_kt_F32_T<3>;
            kernels[3] = mul_mat_iq2_kt_F32_T<4>;
            kernels[4] = mul_mat_iq2_kt_F32_T<5>;
            kernels[5] = mul_mat_iq2_kt_F32_T<6>;
            kernels[6] = mul_mat_iq2_kt_F32_T<7>;
            kernels[7] = mul_mat_iq2_kt_F32_T<8>;
            break;
        case GGML_TYPE_IQ3_KT:
            assert (ne00 % QK_K == 0);
            kernels[0] = mul_mat_iq3_kt_F32_T<1>;
            kernels[1] = mul_mat_iq3_kt_F32_T<2>;
            kernels[2] = mul_mat_iq3_kt_F32_T<3>;
            kernels[3] = mul_mat_iq3_kt_F32_T<4>;
            kernels[4] = mul_mat_iq3_kt_F32_T<5>;
            kernels[5] = mul_mat_iq3_kt_F32_T<6>;
            kernels[6] = mul_mat_iq3_kt_F32_T<7>;
            kernels[7] = mul_mat_iq3_kt_F32_T<8>;
            break;
        case GGML_TYPE_IQ4_KT:
            assert (ne00 % QK_K == 0);
            kernels[0] = mul_mat_iq4_kt_F32_T<1>;
            kernels[1] = mul_mat_iq4_kt_F32_T<2>;
            kernels[2] = mul_mat_iq4_kt_F32_T<3>;
            kernels[3] = mul_mat_iq4_kt_F32_T<4>;
            kernels[4] = mul_mat_iq4_kt_F32_T<5>;
            kernels[5] = mul_mat_iq4_kt_F32_T<6>;
            kernels[6] = mul_mat_iq4_kt_F32_T<7>;
            kernels[7] = mul_mat_iq4_kt_F32_T<8>;
            break;
        default:
            return false;
    }

    return true;

}

#else // !__x86_64__

bool iqk_set_kernels_ktquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {
    return false;
}

#endif

#endif
