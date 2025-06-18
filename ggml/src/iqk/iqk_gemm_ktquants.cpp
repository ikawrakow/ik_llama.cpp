#include "iqk_common.h"
#include "iqk_gemm_ktquants.h"
#include "ggml.h"

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#ifdef __x86_64__

namespace {

inline uint32_t trellis_next(uint32_t& val) {
    constexpr uint32_t ka = 89226354;
    constexpr uint32_t kb = 64248484;
    constexpr uint32_t kmask = 0x8fff8fff;
    constexpr uint32_t km32 = 0x3b603b60;
    val = val*ka + kb;
    return (val & kmask) ^ km32;
}

inline float trellis_gen(uint32_t& val, uint32_t* s) {
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

inline __m256 trellis_gen8(__m256i i8) {
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


template <bool is_8 = false, bool is_abs = false>
struct Trellis3 {
    constexpr static uint32_t ka = 0xCBAC1FED;
    constexpr static uint32_t ka1 = ka*ka;
    constexpr static uint32_t ka2 = ka1*ka;
    constexpr static uint32_t ka3 = ka2*ka;
    constexpr static uint32_t ka4 = ka3*ka;
    constexpr static uint32_t ka5 = ka4*ka;
    constexpr static uint32_t ka6 = ka5*ka;
    constexpr static uint32_t ka7 = ka6*ka;
    const __m256i mka = is_8 ? _mm256_setr_epi32(ka, ka1, ka2, ka3, ka4, ka5, ka6, ka7) : _mm256_setr_epi32(ka, ka1, ka2, ka3, ka, ka1, ka2, ka3);
    const __m256i shuffle = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    inline __m256i next8(uint32_t val1, uint32_t val2) const {
        __m256i mval = MM256_SET_M128I(_mm_set1_epi32(val2), _mm_set1_epi32(val1));
        return _mm256_mullo_epi32(mval, mka);
    }
    inline __m256i next8(uint32_t val) const {
        __m256i mval = _mm256_set1_epi32(val);
        return _mm256_mullo_epi32(mval, mka);
    }
    inline __m256 gen8(uint32_t val1, uint32_t val2) const {
        auto v8 = _mm256_and_si256(next8(val1, val2), _mm256_set1_epi32(0x3f3f3f3f));
#ifdef HAVE_FANCY_SIMD
        auto i8 = _mm256_dpbusd_epi32(_mm256_set1_epi32(-126), _mm256_set1_epi32(0x01010101), v8);
#else
        auto dot = _mm256_maddubs_epi16(v8, _mm256_set1_epi32(0x01010101));
        auto i8  = _mm256_add_epi32(_mm256_set1_epi32(-126), _mm256_madd_epi16(dot, _mm256_set1_epi16(1)));
#endif
        if constexpr (is_abs) {
            return _mm256_cvtepi32_ps(_mm256_sign_epi32(i8, i8));
        } else {
            return _mm256_cvtepi32_ps(i8);
        }
    }
    inline __m256 gen8(uint32_t val) const {
        auto v8 = _mm256_and_si256(next8(val), _mm256_set1_epi32(0x3f3f3f3f));
#ifdef HAVE_FANCY_SIMD
        auto i8 = _mm256_dpbusd_epi32(_mm256_set1_epi32(-126), _mm256_set1_epi32(0x01010101), v8);
#else
        auto dot = _mm256_maddubs_epi16(v8, _mm256_set1_epi32(0x01010101));
        auto i8  = _mm256_add_epi32(_mm256_set1_epi32(-126), _mm256_madd_epi16(dot, _mm256_set1_epi16(1)));
#endif
        if constexpr (is_abs) {
            return _mm256_cvtepi32_ps(_mm256_sign_epi32(i8, i8));
        } else {
            return _mm256_cvtepi32_ps(i8);
        }
    }
    inline __m256i next32(const uint32_t * val) const {
        const __m256i offset = _mm256_set1_epi32(-126);
        __m256i aux[4];
        for (int i = 0; i < 4; ++i) {
            auto i8 = _mm256_and_si256(next8(val[2*i+0], val[2*i+1]), _mm256_set1_epi32(0x3f3f3f3f));
#ifdef HAVE_FANCY_SIMD
            aux[i] = _mm256_dpbusd_epi32(offset, _mm256_set1_epi32(0x01010101), i8);
#else
            auto dot = _mm256_maddubs_epi16(i8, _mm256_set1_epi32(0x01010101));
            aux[i] = _mm256_add_epi32(offset, _mm256_madd_epi16(dot, _mm256_set1_epi16(1)));
#endif
        }
        aux[0] = _mm256_packs_epi32(aux[0], aux[1]); //  0,  1,  2,  3,  8,  9, 10, 11,  4,  5,  6,  7, 12, 13, 14, 15
        aux[2] = _mm256_packs_epi32(aux[2], aux[3]); // 16, 17, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 30, 31
        aux[0] = _mm256_packs_epi16(aux[0], aux[2]); //  0,  1,  2,  3,  8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
                                                     //  4,  5,  6,  7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
        if constexpr (is_abs) {
            auto result = _mm256_permutevar8x32_epi32(aux[0], shuffle);
            return _mm256_sign_epi8(result, result);
        } else {
            return _mm256_permutevar8x32_epi32(aux[0], shuffle);
        }
    }
    inline __m256i next32(const uint16_t * val, uint32_t v0) const {
        const __m256i offset = _mm256_set1_epi32(-126);
        __m256i aux[4];
        for (int i = 0; i < 4; ++i) {
            auto i8 = _mm256_and_si256(next8(v0 + val[i]), _mm256_set1_epi32(0x3f3f3f3f));
#ifdef HAVE_FANCY_SIMD
            aux[i] = _mm256_dpbusd_epi32(offset, _mm256_set1_epi32(0x01010101), i8);
#else
            auto dot = _mm256_maddubs_epi16(i8, _mm256_set1_epi32(0x01010101));
            aux[i] = _mm256_add_epi32(offset, _mm256_madd_epi16(dot, _mm256_set1_epi16(1)));
#endif
        }
        aux[0] = _mm256_packs_epi32(aux[0], aux[1]); //  0,  1,  2,  3,  8,  9, 10, 11,  4,  5,  6,  7, 12, 13, 14, 15
        aux[2] = _mm256_packs_epi32(aux[2], aux[3]); // 16, 17, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 30, 31
        aux[0] = _mm256_packs_epi16(aux[0], aux[2]); //  0,  1,  2,  3,  8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
                                                     //  4,  5,  6,  7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
        if constexpr (is_abs) {
            auto result = _mm256_permutevar8x32_epi32(aux[0], shuffle);
            return _mm256_sign_epi8(result, result);
        } else {
            return _mm256_permutevar8x32_epi32(aux[0], shuffle);
        }
    }
    inline void next64(const uint32_t * val, __m256i * result) const {
        const __m256i offset = _mm256_set1_epi32(-126);
        auto vka3 = _mm256_set1_epi32(ka3);
        __m256i aux[8];
        for (int i = 0; i < 4; ++i) {
            auto i8_1 = next8(val[2*i+0], val[2*i+1]);
            auto i8_2 = _mm256_mullo_epi32(i8_1, vka3);
            i8_1 = _mm256_and_si256(i8_1, _mm256_set1_epi32(0x3f3f3f3f));
            i8_2 = _mm256_and_si256(i8_2, _mm256_set1_epi32(0x3f3f3f3f));
#ifdef HAVE_FANCY_SIMD
            aux[i+0] = _mm256_dpbusd_epi32(offset, _mm256_set1_epi32(0x01010101), i8_1);
            aux[i+4] = _mm256_dpbusd_epi32(offset, _mm256_set1_epi32(0x01010101), i8_2);
#else
            auto dot1 = _mm256_maddubs_epi16(i8_1, _mm256_set1_epi32(0x01010101));
            auto dot2 = _mm256_maddubs_epi16(i8_2, _mm256_set1_epi32(0x01010101));
            aux[i+0] = _mm256_add_epi32(offset, _mm256_madd_epi16(dot1, _mm256_set1_epi16(1)));
            aux[i+4] = _mm256_add_epi32(offset, _mm256_madd_epi16(dot2, _mm256_set1_epi16(1)));
#endif
        }
        for (int k = 0; k < 2; ++k) {
            aux[4*k+0] = _mm256_packs_epi32(aux[4*k+0], aux[4*k+1]); //  0,  1,  2,  3,  8,  9, 10, 11,  4,  5,  6,  7, 12, 13, 14, 15
            aux[4*k+2] = _mm256_packs_epi32(aux[4*k+2], aux[4*k+3]); // 16, 17, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 30, 31
            aux[4*k+0] = _mm256_packs_epi16(aux[4*k+0], aux[4*k+2]); //  0,  1,  2,  3,  8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
                                                                     //  4,  5,  6,  7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
            result[k] = _mm256_permutevar8x32_epi32(aux[4*k+0], shuffle);
            if constexpr (is_abs) {
                result[k] = _mm256_sign_epi8(result[k], result[k]);
            }
        }
    }
};

void iqk_dequantize_iq2_kt(int n, const void * vx, size_t bx, float * y, size_t stride_y, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis1 trellis;

    auto shifts = _mm_set_epi32(0, 0, 4, 0);
    auto values = _mm_loadu_si128((const __m128i *)iq4k_values);

    union { __m256 vec; float val[8]; } s_helper;

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        auto d = _mm256_set1_ps(*dptr * 31.75f * 1.05f);
        const block_iq2_kt * x = (const block_iq2_kt *)(dptr + 1);

        for (int i = 0; i < nb; ++i) {
            const uint16_t * ql = (const uint16_t *)x[i].ql;
            auto s8 = _mm_set1_epi32(*(const uint32_t *)x[i].scales);
            s8 = _mm_and_si128(_mm_srlv_epi32(s8, shifts), _mm_set1_epi8(0xf));
            s8 = _mm_shuffle_epi8(values, s8);
            auto s32 = _mm256_cvtepi8_epi32(s8);
            s_helper.vec = _mm256_mul_ps(d, _mm256_cvtepi32_ps(s32));
            for (int ib = 0; ib < QK_K/64; ++ib) {
                auto scale1 = _mm256_set1_ps(s_helper.val[2*ib+0]);
                auto scale2 = _mm256_set1_ps(s_helper.val[2*ib+1]);
                for (int j = 0; j < 4; ++j) {
                    auto xval1 = _mm256_mul_ps(scale1, trellis_gen8(trellis.next8(ql[8*ib+j+0]+4096)));
                    auto xval2 = _mm256_mul_ps(scale2, trellis_gen8(trellis.next8(ql[8*ib+j+4]+4096)));
                    _mm256_storeu_ps(y + i*QK_K + 64*ib + 8*j +  0, xval1);
                    _mm256_storeu_ps(y + i*QK_K + 64*ib + 8*j + 32, xval2);
                }
            }
        }

        y += stride_y;
    }
}

void iqk_dequantize_iq2_kt_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);
    const int nb = n/QK_K;

    Trellis3 trellis;

    auto shifts = _mm_set_epi32(0, 0, 4, 0);
    auto values = _mm_loadu_si128((const __m128i *)iq4k_values);

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const block_iq2_kt * x8[8];
    float dkt[8];
    float ls[8];
    float ls_all[64];
    uint32_t idx[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char*)vx + (ix+k)*bx);
            dkt[k] = dptr[0];
            x8[k] = (const block_iq2_kt *)(dptr + 1);
        }
        auto vd = _mm256_mul_ps(_mm256_set1_ps(1.05f), _mm256_loadu_ps(dkt));

        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                auto s8 = _mm_set1_epi32(*(const uint32_t *)x8[k][i].scales);
                s8 = _mm_and_si128(_mm_srlv_epi32(s8, shifts), _mm_set1_epi8(0xf));
                s8 = _mm_shuffle_epi8(values, s8);
                auto s32 = _mm256_cvtepi8_epi32(s8);
                _mm256_storeu_ps(ls_all + 8*k, _mm256_cvtepi32_ps(s32));
            }
            for (int ib = 0; ib < QK_K/32; ++ib) {
                for (int k = 0; k < 8; ++k) ls[k] = ls_all[8*k+ib];
                auto scales = _mm256_mul_ps(vd, _mm256_loadu_ps(ls));
                _mm_storeu_si128((__m128i *)y[ib].d, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
                for (int j = 0; j < 4; ++j) {
                    for (int k = 0; k < 8; ++k) {
                        const uint16_t * ql = (const uint16_t *)x8[k][i].ql;
                        idx[k] = ql[4*ib+j] + 4096;
                    }
                    __m256i packed[2];
                    trellis.next64(idx, packed);
                    _mm256_storeu_si256((__m256i *)y[ib].qs+2*j+0, packed[0]);
                    _mm256_storeu_si256((__m256i *)y[ib].qs+2*j+1, packed[1]);
                }
            }
            y += 8; // = QK_K/32;
        }
    }
}

template <int nrc_y>
void mul_mat_iq2_kt_F32_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
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

template <int nrc_y>
void mul_mat_iq2_kt_q8_2_x4_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis3<true> trellis;

    auto shifts = _mm_set_epi32(0, 0, 4, 0);
    auto values = _mm_loadu_si128((const __m128i *)iq4k_values);

    constexpr int k_acc = nrc_y;

    __m256  accd[k_acc];
    const block_q8_2_x4 * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = (const block_q8_2_x4 *)info.src1_row(iy);
    }

    __m256i  xv[4], dot[4];
    __m256   scales[2];

    auto sum_4 = [&dot] () {
        // dot[k] has 8 values from block k
        // 0 1 0 1 0 1 0 1
        dot[0] = _mm256_add_epi32(_mm256_unpacklo_epi32(dot[0], dot[1]), _mm256_unpackhi_epi32(dot[0], dot[1]));
        // 2 3 2 3 2 3 2 3
        dot[2] = _mm256_add_epi32(_mm256_unpacklo_epi32(dot[2], dot[3]), _mm256_unpackhi_epi32(dot[2], dot[3]));
        // 0 1 2 3 0 1 2 3
        dot[0] = _mm256_add_epi32(_mm256_unpacklo_epi64(dot[0], dot[2]), _mm256_unpackhi_epi64(dot[0], dot[2]));
        return _mm256_cvtepi32_ps(dot[0]);
    };

    auto compute_dot = [&dot, &xv] (const int8_t * y) {
        for (int k = 0; k < 4; ++k) {
            auto yv = _mm256_loadu_si256((const __m256i *)y + k);
#ifdef HAVE_FANCY_SIMD
            //dot[k] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), xv[k], yv);
            dot[k] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(xv[k], xv[k]), _mm256_sign_epi8(yv, xv[k]));
#else
            auto p = _mm256_maddubs_epi16(_mm256_sign_epi8(xv[k], xv[k]), _mm256_sign_epi8(yv, xv[k]));
            dot[k] = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
#endif
        }
    };

    //auto m126 = _mm256_set1_ps(-126.f);

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        auto d = _mm256_set1_ps(dptr[0] * 1.05f);
        const block_iq2_kt * x = (const block_iq2_kt *)(dptr + 1);

        for (int iy = 0; iy < k_acc; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            const uint16_t * ql = (const uint16_t *)x[i].ql;
            auto s8 = _mm_set1_epi32(*(const uint32_t *)x[i].scales);
            s8 = _mm_and_si128(_mm_srlv_epi32(s8, shifts), _mm_set1_epi8(0xf));
            s8 = _mm_shuffle_epi8(values, s8);
            auto s32 = _mm256_cvtepi8_epi32(s8);
            auto all_scales = _mm256_mul_ps(d, _mm256_cvtepi32_ps(s32));
            auto scales_l = _mm256_castps256_ps128(all_scales);
            auto scales_h = _mm256_extractf128_ps(all_scales, 1);
            scales[0] = _mm256_set_m128(scales_l, scales_l);
            scales[1] = _mm256_set_m128(scales_h, scales_h);
            for (int i128 = 0; i128 < 2; ++i128) {
                //for (int k = 0; k < 4; ++k) xv[k] = trellis.next32<true>(values + 32*i128 + 8*k);
                for (int k = 0; k < 4; ++k) xv[k] = trellis.next32(ql + 16*i128 + 4*k, 4096);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const block_q8_2_x4& yb = y[iy][2*i+i128];
                    auto dy = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)yb.d)), 16));
                    dy = _mm256_mul_ps(scales[i128], dy);
                    auto d8 = _mm256_set_m128(_mm256_castps256_ps128(dy), _mm256_castps256_ps128(dy));
                    //auto m8 = _mm256_set_m128(_mm256_extractf128_ps(dy, 1), _mm256_extractf128_ps(dy, 1));
                    compute_dot(yb.qs);
                    accd[iy] = _mm256_fmadd_ps(d8, sum_4(), accd[iy]);
                    //accd[iy] = _mm256_fmadd_ps(m8, m126,    accd[iy]);
                }
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }
    }
}

void iqk_dequantize_iq3_kt_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);
    const int nb = n/QK_K;

    Trellis3<false, true> trellis;

    auto shifts = _mm_set_epi32(0, 0, 4, 0);

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const block_iq3_kt * x8[8];
    float dkt[8];
    float ls[8];
    float ls_all[64];
    uint32_t idx[8];
    uint32_t sign_bits[16];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char*)vx + (ix+k)*bx);
            dkt[k] = dptr[0];
            x8[k] = (const block_iq3_kt *)(dptr + 1);
        }
        auto vd = _mm256_mul_ps(_mm256_set1_ps(1.01f), _mm256_loadu_ps(dkt));

        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                auto s8 = _mm_set1_epi32(*(const uint32_t *)x8[k][i].scales);
                s8 = _mm_and_si128(_mm_srlv_epi32(s8, shifts), _mm_set1_epi8(0xf));
                auto s32 = _mm256_cvtepi8_epi32(s8);
                _mm256_storeu_ps(ls_all + 8*k, _mm256_cvtepi32_ps(s32));
            }
            auto mask = _mm256_set1_epi8(1);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                for (int k = 0; k < 8; ++k) ls[k] = ls_all[8*k+ib];
                auto scales = _mm256_mul_ps(vd, _mm256_loadu_ps(ls));
                _mm_storeu_si128((__m128i *)y[ib].d, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
                for (int j = 0; j < 4; ++j) {
                    for (int k = 0; k < 8; ++k) {
                        const uint16_t * ql = (const uint16_t *)x8[k][i].ql;
                        idx[k] = ql[4*ib+j] + 4096;
                        auto qh = (const uint32_t *)x8[k][i].qh;
                        sign_bits[k+0] = qh[2*j+0];
                        sign_bits[k+8] = qh[2*j+1];
                    }
                    __m256i packed[2];
                    trellis.next64(idx, packed);
                    auto signs1 = _mm256_loadu_si256((const __m256i *)sign_bits+0);
                    auto signs2 = _mm256_loadu_si256((const __m256i *)sign_bits+1);
                    signs1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(signs1, mask), mask), _mm256_set1_epi8(1));
                    signs2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(signs2, mask), mask), _mm256_set1_epi8(1));
                    packed[0] = _mm256_sign_epi8(packed[0], signs1);
                    packed[1] = _mm256_sign_epi8(packed[1], signs2);
                    _mm256_storeu_si256((__m256i *)y[ib].qs+2*j+0, packed[0]);
                    _mm256_storeu_si256((__m256i *)y[ib].qs+2*j+1, packed[1]);
                }
                mask = _mm256_slli_epi16(mask, 1);
            }
            y += 8; // = QK_K/32;
        }
    }
}

template <int nrc_y>
void mul_mat_iq3_kt_q8_2_x4_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis3<true, true> trellis;

    auto shifts = _mm_set_epi32(0, 0, 4, 0);

    constexpr int k_acc = nrc_y;

    __m256  accd[k_acc];
    const block_q8_2_x4 * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = (const block_q8_2_x4 *)info.src1_row(iy);
    }

    __m256i  xv[4], sv[4], dot[4];
    __m256   scales[2];

    auto sum_4 = [&dot] () {
        // dot[k] has 8 values from block k
        // 0 1 0 1 0 1 0 1
        dot[0] = _mm256_add_epi32(_mm256_unpacklo_epi32(dot[0], dot[1]), _mm256_unpackhi_epi32(dot[0], dot[1]));
        // 2 3 2 3 2 3 2 3
        dot[2] = _mm256_add_epi32(_mm256_unpacklo_epi32(dot[2], dot[3]), _mm256_unpackhi_epi32(dot[2], dot[3]));
        // 0 1 2 3 0 1 2 3
        dot[0] = _mm256_add_epi32(_mm256_unpacklo_epi64(dot[0], dot[2]), _mm256_unpackhi_epi64(dot[0], dot[2]));
        return _mm256_cvtepi32_ps(dot[0]);
    };

    auto compute_dot = [&dot, &xv, &sv] (const int8_t * y) {
        for (int k = 0; k < 4; ++k) {
            auto yv = _mm256_loadu_si256((const __m256i *)y + k);
#ifdef HAVE_FANCY_SIMD
            //dot[k] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), xv[k], yv);
            dot[k] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), xv[k], _mm256_sign_epi8(yv, sv[k]));
#else
            auto p = _mm256_maddubs_epi16(xv[k], _mm256_sign_epi8(yv, sv[k]));
            dot[k] = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
#endif
        }
    };

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        auto d = _mm256_set1_ps(dptr[0] * 1.01f);
        const block_iq3_kt * x = (const block_iq3_kt *)(dptr + 1);

        for (int iy = 0; iy < k_acc; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            auto ql = (const uint16_t *)x[i].ql;
            auto sign_bits = _mm256_loadu_si256((const __m256i *)x[i].qh);
            auto s8 = _mm_set1_epi32(*(const uint32_t *)x[i].scales);
            s8 = _mm_and_si128(_mm_srlv_epi32(s8, shifts), _mm_set1_epi8(0xf));
            auto s32 = _mm256_cvtepi8_epi32(s8);
            auto all_scales = _mm256_mul_ps(d, _mm256_cvtepi32_ps(s32));
            auto scales_l = _mm256_castps256_ps128(all_scales);
            auto scales_h = _mm256_extractf128_ps(all_scales, 1);
            scales[0] = _mm256_set_m128(scales_l, scales_l);
            scales[1] = _mm256_set_m128(scales_h, scales_h);
            auto mask = _mm256_set1_epi8(1);
            for (int i128 = 0; i128 < 2; ++i128) {
                for (int k = 0; k < 4; ++k) {
                    xv[k] = trellis.next32(ql + 16*i128 + 4*k, 4096);
                    sv[k] = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sign_bits, mask), mask), _mm256_set1_epi8(1));
                    mask = _mm256_slli_epi16(mask, 1);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const block_q8_2_x4& yb = y[iy][2*i+i128];
                    auto dy = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)yb.d)), 16));
                    dy = _mm256_mul_ps(scales[i128], dy);
                    auto d8 = _mm256_set_m128(_mm256_castps256_ps128(dy), _mm256_castps256_ps128(dy));
                    compute_dot(yb.qs);
                    accd[iy] = _mm256_fmadd_ps(d8, sum_4(), accd[iy]);
                }
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }
    }
}

inline __m256 abs_ps(__m256 vals) {
    // Clear sign-bit of all the 32-bit floats in vals
    __m256 sign_bit = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(sign_bit, vals);
}

void iqk_dequantize_iq3_kt(int n, const void * vx, size_t bx, float * y, size_t stride_y, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis1 trellis;

    union { __m256 vec; float val[8]; } s_helper;

    auto shifts = _mm_set_epi32(0, 0, 4, 0);

    __m256i all_signs[4];
    auto mask1 = _mm256_set1_epi32(0x01);
    auto mask2 = _mm256_set1_epi32(0x10);

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        auto d  = _mm256_set1_ps(*dptr * 31.75f * 1.015f);
        const block_iq3_kt * x = (const block_iq3_kt *)(dptr + 1);

        for (int i = 0; i < nb; ++i) {
            const uint16_t * ql = (const uint16_t *)x[i].ql;
            const uint8_t * qh = x[i].qh;
            auto s8 = _mm_set1_epi32(*(const uint32_t *)x[i].scales);
            s8 = _mm_and_si128(_mm_srlv_epi32(s8, shifts), _mm_set1_epi8(0xf));
            auto s32 = _mm256_cvtepi8_epi32(s8);
            s_helper.vec = _mm256_mul_ps(d, _mm256_cvtepi32_ps(s32));
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
                    _mm256_storeu_ps(y + i*QK_K+32*ib+8*j    , x_val1);
                    _mm256_storeu_ps(y + i*QK_K+32*ib+8*j+128, x_val2);
                }
            }
        }
        y += stride_y;
    }
}

template <int nrc_y>
void mul_mat_iq3_kt_F32_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
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

void iqk_dequantize_iq4_kt_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);
    const int nb = n/QK_K;
    constexpr int kNumGroups = 64;

    Trellis3 trellis;

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const block_iq4_kt * x8[8];
    float dkt[8];
    int32_t ls[8];
    uint32_t idx0[8], idx[16];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char*)vx + (ix+k)*bx);
            dkt[k] = dptr[0];
            x8[k] = (const block_iq4_kt *)(dptr + 1);
        }
        auto vd = _mm256_loadu_ps(dkt);

        for (int i = 0; i < nb; ++i) {
            for (int ib = 0; ib < QK_K/32; ++ib) {
                for (int k = 0; k < 8; ++k) {
                    ls[k] = ((x8[k][i].qs[ib] & 0xff) >> 1) - 64;
                    idx0[k] = ((x8[k][i].qs[ib] & 1) << 15) + 4096;
                }
                auto scales = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)ls)));
                _mm_storeu_si128((__m128i *)y[ib].d, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
                int shift1 = 8 - 4*(ib/4);
                for (int j = 0; j < 8; ++j) {
                    for (int k = 0; k < 8; ++k) {
                        const uint8_t * ql = (const uint8_t *)(x8[k][i].qs + 8);
                        const uint8_t * qh = ql + kNumGroups;
                        const uint32_t sh = x8[k][i].qs[ib] >> (8 + 3*j);
                        idx[k+0] = ql[8*ib+j] + ((qh[8*(ib%4)+j] << shift1) & 0xf00) + ((sh & 7) << 12) + idx0[k];
                    }
                    _mm256_storeu_si256((__m256i *)y[ib].qs+j, trellis.next32(idx));
                }
            }
            y += 8; // = QK_K/32;
        }

    }
}

void iqk_dequantize_iq4_kt(int n, const void * vx, size_t bx, float * y, size_t stride_y, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    const int nb = n/QK_K;
    constexpr int kNumGroups = 64;

    Trellis3 trellis;

    union { __m256  vec; float    val[8]; } s_helper;
    union { __m256i vec; uint32_t val[8]; } o_helper;

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        auto d = _mm256_set1_ps(dptr[0]);
        auto dav = _mm256_set1_ps(dptr[1]);
        const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 2);

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
                    auto x_val1 = _mm256_fmadd_ps(scale1, trellis.gen8(val1, val3), dav);
                    auto x_val2 = _mm256_fmadd_ps(scale2, trellis.gen8(val2, val4), dav);

                    _mm256_storeu_ps(y + i*QK_K + 32*ib + 8*j,          x_val1);
                    _mm256_storeu_ps(y + i*QK_K + 32*ib + 8*j + QK_K/2, x_val2);

                }
            }
        }

        y += stride_y;

    }
}

template <int nrc_y>
void mul_mat_iq4_kt_q8_2_x4_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;
    constexpr int kNumGroups = 64;

    Trellis3 trellis;

    union { __m256i vec; uint32_t val[8]; } o_helper;

    constexpr int k_acc = nrc_y;

    __m256  accd[k_acc];
    const block_q8_2_x4 * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = (const block_q8_2_x4 *)info.src1_row(iy);
    }

    uint32_t values[64];
    __m256i  xv[4], dot[4];
    __m256   scales[2];

    auto sum_4 = [&dot] () {
        // dot[k] has 8 values from block k
        // 0 1 0 1 0 1 0 1
        dot[0] = _mm256_add_epi32(_mm256_unpacklo_epi32(dot[0], dot[1]), _mm256_unpackhi_epi32(dot[0], dot[1]));
        // 2 3 2 3 2 3 2 3
        dot[2] = _mm256_add_epi32(_mm256_unpacklo_epi32(dot[2], dot[3]), _mm256_unpackhi_epi32(dot[2], dot[3]));
        // 0 1 2 3 0 1 2 3
        dot[0] = _mm256_add_epi32(_mm256_unpacklo_epi64(dot[0], dot[2]), _mm256_unpackhi_epi64(dot[0], dot[2]));
        return _mm256_cvtepi32_ps(dot[0]);
    };

    auto compute_dot = [&dot, &xv] (const int8_t * y) {
        for (int k = 0; k < 4; ++k) {
            auto yv = _mm256_loadu_si256((const __m256i *)y + k);
#ifdef HAVE_FANCY_SIMD
            //dot[k] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), xv[k], yv);
            dot[k] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_sign_epi8(xv[k], xv[k]), _mm256_sign_epi8(yv, xv[k]));
#else
            auto p = _mm256_maddubs_epi16(_mm256_sign_epi8(xv[k], xv[k]), _mm256_sign_epi8(yv, xv[k]));
            dot[k] = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
#endif
        }
    };

    //auto m126 = _mm256_set1_ps(-126.f);

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        auto d = _mm256_set1_ps(dptr[0]);
        const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 1);

        for (int iy = 0; iy < k_acc; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {
            auto vshb = _mm256_loadu_si256((const __m256i *)x[i].qs);
            const uint32_t * shb = x[i].qs;
            const uint8_t * ql = (const uint8_t *)(shb + 8);
            const uint8_t * qh = ql + kNumGroups;
            auto iscales = _mm256_srli_epi32(_mm256_and_si256(vshb, _mm256_set1_epi32(0xff)), 1);
            iscales = _mm256_sub_epi32(iscales, _mm256_set1_epi32(64));
            auto all_scales = _mm256_mul_ps(d, _mm256_cvtepi32_ps(iscales));
            auto scales_l = _mm256_castps256_ps128(all_scales);
            auto scales_h = _mm256_extractf128_ps(all_scales, 1);
            scales[0] = _mm256_set_m128(scales_l, scales_l);
            scales[1] = _mm256_set_m128(scales_h, scales_h);
            o_helper.vec = _mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(vshb, _mm256_set1_epi32(1)), 15), _mm256_set1_epi32(4096));
            for (int ib = 0; ib < 4; ++ib) {
                for (int j = 0; j < 4; ++j) {
                    const uint32_t sh1 = shb[ib+0] >> (8 + 6*j);
                    const uint32_t sh2 = shb[ib+4] >> (8 + 6*j);
                    values[8*ib+2*j+ 0] = ql[8*ib+2*j+ 0] + ((qh[8*ib+2*j+0] << 8) & 0xf00) + ((sh1 & 7) << 12) + o_helper.val[ib+0];
                    values[8*ib+2*j+ 1] = ql[8*ib+2*j+ 1] + ((qh[8*ib+2*j+1] << 8) & 0xf00) + ((sh1 & 56) << 9) + o_helper.val[ib+0];
                    values[8*ib+2*j+32] = ql[8*ib+2*j+32] + ((qh[8*ib+2*j+0] << 4) & 0xf00) + ((sh2 & 7) << 12) + o_helper.val[ib+4];
                    values[8*ib+2*j+33] = ql[8*ib+2*j+33] + ((qh[8*ib+2*j+1] << 4) & 0xf00) + ((sh2 & 56) << 9) + o_helper.val[ib+4];
                }
            }
            for (int i128 = 0; i128 < 2; ++i128) {
                //for (int k = 0; k < 4; ++k) xv[k] = trellis.next32<true>(values + 32*i128 + 8*k);
                for (int k = 0; k < 4; ++k) xv[k] = trellis.next32(values + 32*i128 + 8*k);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const block_q8_2_x4& yb = y[iy][2*i+i128];
                    auto dy = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)yb.d)), 16));
                    dy = _mm256_mul_ps(scales[i128], dy);
                    auto d8 = _mm256_set_m128(_mm256_castps256_ps128(dy), _mm256_castps256_ps128(dy));
                    //auto m8 = _mm256_set_m128(_mm256_extractf128_ps(dy, 1), _mm256_extractf128_ps(dy, 1));
                    compute_dot(yb.qs);
                    accd[iy] = _mm256_fmadd_ps(d8, sum_4(), accd[iy]);
                    //accd[iy] = _mm256_fmadd_ps(m8, m126,    accd[iy]);
                }
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }
    }
}

template <int nrc_y>
void mul_mat_iq4_kt_F32_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;
    constexpr int kNumGroups = 64;

    Trellis3 trellis;

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
        auto d = _mm256_set1_ps(dptr[0]);
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
                    auto x_val1 = _mm256_mul_ps(scale1, trellis.gen8(val1, val3));
                    auto x_val2 = _mm256_mul_ps(scale2, trellis.gen8(val2, val4));
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

    if (ne00%QK_K != 0) return false;

    func16 = nullptr;

    if (typeA == GGML_TYPE_IQ4_KT) {
        if (typeB == GGML_TYPE_Q8_2_X4) {
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_kt_q8_2_x4_T, kernels);
            return true;
        }
        return false;
    }

    if (typeA == GGML_TYPE_IQ2_KT) {
        if (typeB == GGML_TYPE_Q8_2_X4) {
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_kt_q8_2_x4_T, kernels);
            return true;
        }
        return false;
    }

    if (typeA == GGML_TYPE_IQ3_KT) {
        if (typeB == GGML_TYPE_Q8_2_X4) {
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq3_kt_q8_2_x4_T, kernels);
            return true;
        }
        return false;
    }

    if (ggml_type(typeB) != GGML_TYPE_F32) {
        return false;
    }

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

bool iqk_dequantize_ktquants(int type, int n, const void * vx, size_t bx, void * y, [[maybe_unused]] size_t stride_y, int nrc_x) {
    switch (type) {
        case GGML_TYPE_IQ2_KT: iqk_dequantize_iq2_kt_q80_r8(n, vx, bx, y, nrc_x); break;
        case GGML_TYPE_IQ3_KT: iqk_dequantize_iq3_kt_q80_r8(n, vx, bx, y, nrc_x); break;
        case GGML_TYPE_IQ4_KT: iqk_dequantize_iq4_kt_q80_r8(n, vx, bx, y, nrc_x); break;
        default: return false;
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

void iqk_dequantize_iq2_kt(int n, const void * vx, size_t bx, float16_t * y, size_t stride_y, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis1 trellis;

    auto values = vld1q_s8(iq4k_values);

    union { float16x8_t vec; float16_t val[8]; } s_helper;

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        const float d = *dptr * 31.75f * 1.05f;
        auto vd = vdupq_n_f32(d);
        const block_iq2_kt * x = (const block_iq2_kt *)(dptr + 1);

        for (int i = 0; i < nb; ++i) {
            const uint16_t * ql = (const uint16_t *)x[i].ql;
            auto u32 = *(const uint32_t *)x[i].scales;
            auto s8_u32 = uint32x2_t{u32, u32 >> 4};
            s8_u32 = vand_u8(s8_u32, vdup_n_u32(0x0f0f0f0f));
            auto s8 = vqtbl1_s8(values, vreinterpret_u8_u32(s8_u32));
            auto s16 = vmovl_s8(s8);
            auto s32l = vmovl_s16(vget_low_s16 (s16));
            auto s32h = vmovl_s16(vget_high_s16(s16));
            auto f32l = vmulq_f32(vd, vcvtq_f32_s32(s32l));
            auto f32h = vmulq_f32(vd, vcvtq_f32_s32(s32h));
            s_helper.vec = vcombine_f16(vcvt_f16_f32(f32l), vcvt_f16_f32(f32h));
            for (int ib = 0; ib < QK_K/64; ++ib) {
                auto scale1 = vdupq_n_f16(s_helper.val[2*ib+0]);
                auto scale2 = vdupq_n_f16(s_helper.val[2*ib+1]);
                for (int j = 0; j < 4; ++j) {
                    auto xval1 = vmulq_f16(scale1, trellis.gen8(ql[8*ib+j+0]+4096));
                    auto xval2 = vmulq_f16(scale2, trellis.gen8(ql[8*ib+j+4]+4096));
                    vst1q_f16(y + i*QK_K + 64*ib + 8*j +  0, xval1);
                    vst1q_f16(y + i*QK_K + 64*ib + 8*j + 32, xval2);
                }
            }
        }

        y += stride_y;
    }
}

template <int nrc_y>
void mul_mat_iq2_kt_F16_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
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

void iqk_dequantize_iq3_kt(int n, const void * vx, size_t bx, float16_t * y, size_t stride_y, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis1 trellis;

    union { float16x8_t vec; float16_t val[8]; } s_helper;

    uint16x8_t all_signs[4];
    auto mask1 = vdupq_n_u16(0x01);
    auto mask2 = vdupq_n_u16(0x10);

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        const float d = *dptr * 31.75f * 1.015f;
        auto vd = vdupq_n_f32(d);
        const block_iq3_kt * x = (const block_iq3_kt *)(dptr + 1);

        for (int i = 0; i < nb; ++i) {
            const uint16_t * ql = (const uint16_t *)x[i].ql;
            const uint8_t * qh = x[i].qh;
            auto u32 = *(const uint32_t *)x[i].scales;
            auto s8_u32 = uint32x2_t{u32, u32 >> 4};
            s8_u32 = vand_u8(s8_u32, vdup_n_u32(0x0f0f0f0f));
            auto s16 = vmovl_s8(vreinterpret_s8_u32(s8_u32));
            auto s32l = vmovl_s16(vget_low_s16 (s16));
            auto s32h = vmovl_s16(vget_high_s16(s16));
            auto f32l = vmulq_f32(vd, vcvtq_f32_s32(s32l));
            auto f32h = vmulq_f32(vd, vcvtq_f32_s32(s32h));
            s_helper.vec = vcombine_f16(vcvt_f16_f32(f32l), vcvt_f16_f32(f32h));
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
                    vst1q_f16(y + i*QK_K+32*ib+8*j    , x_val1);
                    vst1q_f16(y + i*QK_K+32*ib+8*j+128, x_val2);
                }
            }
        }
        y += stride_y;
    }
}

template <int nrc_y>
void mul_mat_iq3_kt_F16_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
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

void iqk_dequantize_iq4_kt(int n, const void * vx, size_t bx, float16_t * y, size_t stride_y, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    const int nb = n/QK_K;
    constexpr int kNumGroups = 64;

    Trellis1 trellis;

    union { float16x8_t vec; float16_t val[8]; } s_helper;
    union { uint16x8_t  vec; uint16_t  val[8]; } o_helper;

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        auto d = dptr[0] * 31.75f * 1.01f;
        //auto dav = dptr[1];
        // Something goes wrong when we add the average. Why?
        //auto vav = std::abs(dav) > 0.00006103515625f ? vdupq_n_f16(GGML_FP32_TO_FP16(dav)) : vdupq_n_f16(0);
        auto vd = vdupq_n_f32(d);
        const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 2);

        for (int i = 0; i < nb; ++i) {
            const uint32_t * shb = x[i].qs;
            auto vshb = vld1q_u32_x2(shb);
            auto vshb16 = vcombine_u16(vmovn_u32(vandq_u32(vshb.val[0], vdupq_n_u32(0xff))), vmovn_u32(vandq_u32(vshb.val[1], vdupq_n_u32(0xff))));
            const uint8_t * ql = (const uint8_t *)(shb + 8);
            const uint8_t * qh = ql + kNumGroups;
            auto iscales = vsubq_s16(vreinterpretq_s16_u16(vshrq_n_u16(vshb16, 1)), vdupq_n_s16(64));
            auto s32l = vmovl_s16(vget_low_s16(iscales));
            auto s32h = vmovl_s16(vget_high_s16(iscales));
            auto f32l = vmulq_f32(vd, vcvtq_f32_s32(s32l));
            auto f32h = vmulq_f32(vd, vcvtq_f32_s32(s32h));
            s_helper.vec = vcombine_f16(vcvt_f16_f32(f32l), vcvt_f16_f32(f32h));
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
                    //auto x_val1 = vfmaq_f16(vav, scale1, trellis.gen8(val1, val3));
                    //auto x_val2 = vfmaq_f16(vav, scale2, trellis.gen8(val2, val4));
                    auto x_val1 = vmulq_f16(scale1, trellis.gen8(val1, val3));
                    auto x_val2 = vmulq_f16(scale2, trellis.gen8(val2, val4));
                    vst1q_f16(y + i*QK_K+32*ib+8*j+  0, x_val1);
                    vst1q_f16(y + i*QK_K+32*ib+8*j+128, x_val2);
                }
            }
        }
        y += stride_y;
    }
}

template <int nrc_y>
void mul_mat_iq4_kt_F16_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
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
        row_sum[iy] = vaddvq_f32(sum32);
    }

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        auto d = dptr[0] * 31.75f * 1.01f;
        auto dav = dptr[1];
        auto vd = vdupq_n_f32(d);
        const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 2);

        for (int iy = 0; iy < k_acc; ++iy) accd[iy] = vdupq_n_f16(0);

        for (int i = 0; i < nb; ++i) {
            const uint32_t * shb = x[i].qs;
            auto vshb = vld1q_u32_x2(shb);
            auto vshb16 = vcombine_u16(vmovn_u32(vandq_u32(vshb.val[0], vdupq_n_u32(0xff))), vmovn_u32(vandq_u32(vshb.val[1], vdupq_n_u32(0xff))));
            const uint8_t * ql = (const uint8_t *)(shb + 8);
            const uint8_t * qh = ql + kNumGroups;
            auto iscales = vsubq_s16(vreinterpretq_s16_u16(vshrq_n_u16(vshb16, 1)), vdupq_n_s16(64));
            auto s32l = vmovl_s16(vget_low_s16(iscales));
            auto s32h = vmovl_s16(vget_high_s16(iscales));
            auto f32l = vmulq_f32(vd, vcvtq_f32_s32(s32l));
            auto f32h = vmulq_f32(vd, vcvtq_f32_s32(s32h));
            s_helper.vec = vcombine_f16(vcvt_f16_f32(f32l), vcvt_f16_f32(f32h));
            //s_helper.vec = vcvtq_f16_s16(iscales);
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
            info.store(ix, 0, vaddvq_f32(sum) + dav*row_sum[0]);
        } else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto sum = vaddq_f32(vcvt_f32_f16(vget_low_f16(accd[iy])), vcvt_f32_f16(vget_high_f16(accd[iy])));
                info.store(ix, iy, vaddvq_f32(sum) + dav*row_sum[iy]);
            }
        }
    }
}

template <int nrc_y>
void mul_mat_iq4_kt_F32_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
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

struct Trellis3 {
    constexpr static uint32_t ka = ;0xCBAC1FED;
    constexpr static uint32_t ka1 = ka*ka;
    constexpr static uint32_t ka2 = ka1*ka;
    constexpr static uint32_t ka3 = ka2*ka;
    const uint32x4_t mka = uint32x4_t{ka, ka1, ka2, ka3};
    const uint8x16_t shuffle = load_shuffle();

    inline uint32x4x2_t next8(uint32_t val1, uint32_t val2) const {
        uint32x4x2_t result{vdupq_n_u32(val1), vdupq_n_u32(val2)};
        result.val[0] = vmulq_u32(mka, result.val[0]);
        result.val[1] = vmulq_u32(mka, result.val[1]);
        return result;
    }
    inline int8x16x2_t next32(const uint32_t * val) const {
        int8x16x2_t result = {vdupq_n_s8(-126), vdupq_n_s8(-126)};
        for (int i = 0; i < 2; ++i) {
            auto i8 = next8(val[4*i+0], val[4*i+1]);
            i8.val[0] = vandq_u32(i8.val[0], vdupq_n_u32(0x3f3f3f3f));
            i8.val[1] = vandq_u32(i8.val[1], vdupq_n_u32(0x3f3f3f3f));
            auto s1 = vpaddq_s8(vreinterpretq_s8_u32(i8.val[0]), vreinterpretq_s8_u32(i8.val[1]));
            i8 = next8(val[4*i+2], val[4*i+3]);
            i8.val[0] = vandq_u32(i8.val[0], vdupq_n_u32(0x3f3f3f3f));
            i8.val[1] = vandq_u32(i8.val[1], vdupq_n_u32(0x3f3f3f3f));
            auto s2 = vpaddq_s8(vreinterpretq_s8_u32(i8.val[0]), vreinterpretq_s8_u32(i8.val[1]));
            result.val[i] = vaddq_s8(result.val[i], vpaddq_s8(s1, s2));
        }
        return result;
    }
    inline int8x16x2_t next32(const uint16_t * val, uint32_t v0) const {
        auto vka3 = vdupq_n_u32(ka3), vkb3 = vdupq_n_u32(kb3);
        int8x16x2_t result = {vdupq_n_s8(-126), vdupq_n_s8(-126)};
        int8x16x2_t i8;
        for (int i = 0; i < 2; ++i) {
            i8.val[0] = vmulq_u32(mka, vdupq_n_u32(val[2*i+0]+v0));
            i8.val[1] = vmlaq_u32(vkb3, vka3, i8.val[0]);
            i8.val[0] = vandq_u32(i8.val[0], vdupq_n_u32(0x3f3f3f3f));
            i8.val[1] = vandq_u32(i8.val[1], vdupq_n_u32(0x3f3f3f3f));
            auto s1 = vpaddq_s8(vreinterpretq_s8_u32(i8.val[0]), vreinterpretq_s8_u32(i8.val[1]));
            i8.val[0] = vmulq_u32(mka, vdupq_n_u32(val[2*i+1]+v0));
            i8.val[1] = vmlaq_u32(vkb3, vka3, i8.val[0]);
            i8.val[0] = vandq_u32(i8.val[0], vdupq_n_u32(0x3f3f3f3f));
            i8.val[1] = vandq_u32(i8.val[1], vdupq_n_u32(0x3f3f3f3f));
            auto s2 = vpaddq_s8(vreinterpretq_s8_u32(i8.val[0]), vreinterpretq_s8_u32(i8.val[1]));
            result.val[i] = vaddq_s8(result.val[i], vpaddq_s8(s1, s2));
        }
        return result;
    }
    inline int8x16x4_t next64(const uint32_t * val) const {
        auto vka3 = vdupq_n_u32(ka3), vkb3 = vdupq_n_u32(kb3);
        int8x16x4_t result = {vdupq_n_s8(-126), vdupq_n_s8(-126), vdupq_n_s8(-126), vdupq_n_s8(-126)};
        for (int i = 0; i < 2; ++i) {
            auto i8_1 = next8(val[4*i+0], val[4*i+1]);
            int8x16x2_t i8_2{vmlaq_u32(vkb3, vka3, i8_1.val[0]), vmlaq_u32(vkb3, vka3, i8_1.val[1])};
            i8_1.val[0] = vandq_u32(i8_1.val[0], vdupq_n_u32(0x3f3f3f3f));
            i8_1.val[1] = vandq_u32(i8_1.val[1], vdupq_n_u32(0x3f3f3f3f));
            i8_2.val[0] = vandq_u32(i8_2.val[0], vdupq_n_u32(0x3f3f3f3f));
            i8_2.val[1] = vandq_u32(i8_2.val[1], vdupq_n_u32(0x3f3f3f3f));
            auto s1_1 = vpaddq_s8(vreinterpretq_s8_u32(i8_1.val[0]), vreinterpretq_s8_u32(i8_1.val[1]));
            auto s1_2 = vpaddq_s8(vreinterpretq_s8_u32(i8_2.val[0]), vreinterpretq_s8_u32(i8_2.val[1]));
            i8_1 = next8(val[4*i+2], val[4*i+3]);
            i8_2.val[0] = vmlaq_u32(vkb3, vka3, i8_1.val[0]);
            i8_2.val[1] = vmlaq_u32(vkb3, vka3, i8_1.val[1]);
            i8_1.val[0] = vandq_u32(i8_1.val[0], vdupq_n_u32(0x3f3f3f3f));
            i8_1.val[1] = vandq_u32(i8_1.val[1], vdupq_n_u32(0x3f3f3f3f));
            i8_2.val[0] = vandq_u32(i8_2.val[0], vdupq_n_u32(0x3f3f3f3f));
            i8_2.val[1] = vandq_u32(i8_2.val[1], vdupq_n_u32(0x3f3f3f3f));
            auto s2_1 = vpaddq_s8(vreinterpretq_s8_u32(i8_1.val[0]), vreinterpretq_s8_u32(i8_1.val[1]));
            auto s2_2 = vpaddq_s8(vreinterpretq_s8_u32(i8_2.val[0]), vreinterpretq_s8_u32(i8_2.val[1]));
            result.val[i+0] = vaddq_s8(result.val[i+0], vpaddq_s8(s1_1, s2_1));
            result.val[i+2] = vaddq_s8(result.val[i+2], vpaddq_s8(s1_2, s2_2));
        }
        return result;
    }
    static uint8x16_t load_shuffle() {
        static const uint8_t k_shuffle[16] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60};
        return vld1q_u8(k_shuffle);
    }
};

void iqk_dequantize_iq4_kt_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);
    const int nb = n/QK_K;
    constexpr int kNumGroups = 64;

    Trellis3 trellis;

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const block_iq4_kt * x8[8];
    float dkt[8];
    int32_t ls[8];
    uint32_t idx0[8], idx[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char*)vx + (ix+k)*bx);
            dkt[k] = dptr[0];
            x8[k] = (const block_iq4_kt *)(dptr + 1);
        }
        auto vd = vld1q_f32_x2(dkt);

        for (int i = 0; i < nb; ++i) {
            for (int ib = 0; ib < QK_K/32; ++ib) {
                for (int k = 0; k < 8; ++k) {
                    ls[k] = ((x8[k][i].qs[ib] & 0xff) >> 1) - 64;
                    idx0[k] = ((x8[k][i].qs[ib] & 1) << 15) + 4096;
                }
                auto scales1 = vmulq_f32(vd.val[0], vcvtq_f32_s32(vld1q_s32(ls+0)));
                auto scales2 = vmulq_f32(vd.val[1], vcvtq_f32_s32(vld1q_s32(ls+4)));
                vst1_f16((float16_t *)y[ib].d+0, vcvt_f16_f32(scales1));
                vst1_f16((float16_t *)y[ib].d+4, vcvt_f16_f32(scales2));
                int shift1 = 8 - 4*(ib/4);
                for (int j = 0; j < 8; ++j) {
                    for (int k = 0; k < 8; ++k) {
                        const uint8_t * ql = (const uint8_t *)(x8[k][i].qs + 8);
                        const uint8_t * qh = ql + kNumGroups;
                        const uint32_t sh = x8[k][i].qs[ib] >> (8 + 3*j);
                        idx[k+0] = ql[8*ib+j] + ((qh[8*(ib%4)+j] << shift1) & 0xf00) + ((sh & 7) << 12) + idx0[k];
                    }
                    vst1q_s8_x2(y[ib].qs+32*j, trellis.next32(idx));
                }
            }
            y += 8; // = QK_K/32;
        }
    }
}

template <int nrc_y>
void mul_mat_iq4_kt_q8_0_x4_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;
    constexpr int kNumGroups = 64;

    Trellis3 trellis;

    union { uint32x4x2_t vec; uint32_t val[8]; } o_helper;

    constexpr int k_acc = nrc_y == 1 ? 2 : nrc_y;

    float32x4_t accd[k_acc];

    const block_q8_0_x4 * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = (const block_q8_0_x4 *)info.src1_row(iy);
    }

    uint32_t values[16];
    int8x16x2_t xv[8];
    int32x4x4_t dot;

    auto compute_dot = [&dot] (const int8_t * y, const int8x16x2_t * xv) {
        for (int k = 0; k < 4; ++k) {
            auto yv = vld1q_s8_x2(y + 32*k);
            dot.val[k] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), xv[k].val[0], yv.val[0]), xv[k].val[1], yv.val[1]);
        }
        dot.val[0] = vpaddq_s32(dot.val[0], dot.val[1]);
        dot.val[2] = vpaddq_s32(dot.val[2], dot.val[3]);
        return vpaddq_s32(dot.val[0], dot.val[2]);
    };

    int32x4x2_t shifts = {int32x4_t{4, 1, -2, -5}, int32x4_t{-8, -11, -14, -17}};

    float32x4x2_t scales;

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        auto d = vdupq_n_f32(dptr[0]);
        const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 1);

        for (int iy = 0; iy < k_acc; ++iy) accd[iy] = vdupq_n_f32(0);

        for (int i = 0; i < nb; ++i) {
            auto vshb = vld1q_u32_x2(x[i].qs);
            const uint32_t * shb = x[i].qs;
            const uint8_t * ql = (const uint8_t *)(shb + 8);
            const uint8_t * qh = ql + kNumGroups;
            auto iscales1 = vreinterpretq_s32_u32(vshrq_n_u32(vandq_u32(vshb.val[0], vdupq_n_u32(0xff)), 1));
            auto iscales2 = vreinterpretq_s32_u32(vshrq_n_u32(vandq_u32(vshb.val[1], vdupq_n_u32(0xff)), 1));
            iscales1 = vaddq_s32(iscales1, vdupq_n_s32(-64));
            iscales2 = vaddq_s32(iscales2, vdupq_n_s32(-64));
            scales.val[0] = vmulq_f32(d, vcvtq_f32_s32(iscales1));
            scales.val[1] = vmulq_f32(d, vcvtq_f32_s32(iscales2));
            o_helper.vec.val[0] = vaddq_u32(vshlq_n_u32(vandq_u32(vshb.val[0], vdupq_n_u32(1)), 15), vdupq_n_u32(4096));
            o_helper.vec.val[1] = vaddq_u32(vshlq_n_u32(vandq_u32(vshb.val[1], vdupq_n_u32(1)), 15), vdupq_n_u32(4096));
            for (int ib = 0; ib < 4; ++ib) {
                auto vql1 = vmovl_u8(vld1_u8(ql+8*ib));
                auto vql2 = vmovl_u8(vld1_u8(ql+8*ib+32));
                auto vqh  = vmovl_u8(vld1_u8(qh+8*ib));
                vql1 = vaddq_u16(vql1, vandq_u16(vdupq_n_u16(0xf00), vshlq_n_u16(vqh, 8)));
                vql2 = vaddq_u16(vql2, vandq_u16(vdupq_n_u16(0xf00), vshlq_n_u16(vqh, 4)));
                auto sh1_u32 = vdupq_n_u32(shb[ib+0]);
                auto sh2_u32 = vdupq_n_u32(shb[ib+4]);
                auto sh1 = vcombine_u16(vmovn_u32(vshlq_u32(sh1_u32, shifts.val[0])), vmovn_u32(vshlq_u32(sh1_u32, shifts.val[1])));
                auto sh2 = vcombine_u16(vmovn_u32(vshlq_u32(sh2_u32, shifts.val[0])), vmovn_u32(vshlq_u32(sh2_u32, shifts.val[1])));
                vql1 = vaddq_u16(vql1, vandq_u16(vdupq_n_u16(0x7000), sh1));
                vql2 = vaddq_u16(vql2, vandq_u16(vdupq_n_u16(0x7000), sh2));
                auto oh1 = vdupq_n_u32(o_helper.val[ib+0]);
                auto oh2 = vdupq_n_u32(o_helper.val[ib+4]);
                vst1q_u32(values +0, vaddq_u32(vmovl_u16(vget_low_u16 (vql1)), oh1));
                vst1q_u32(values +4, vaddq_u32(vmovl_u16(vget_high_u16(vql1)), oh1));
                vst1q_u32(values +8, vaddq_u32(vmovl_u16(vget_low_u16 (vql2)), oh2));
                vst1q_u32(values+12, vaddq_u32(vmovl_u16(vget_high_u16(vql2)), oh2));
                xv[ib+0] = trellis.next32(values+0);
                xv[ib+4] = trellis.next32(values+8);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                const block_q8_0_x4& ybl = y[iy][2*i+0];
                const block_q8_0_x4& ybh = y[iy][2*i+1];
                auto dyl = vmulq_f32(scales.val[0], vcvt_f32_f16(vld1_f16((const float16_t *)ybl.d)));
                auto dyh = vmulq_f32(scales.val[1], vcvt_f32_f16(vld1_f16((const float16_t *)ybh.d)));
                auto sumil = compute_dot(ybl.qs, xv+0);
                auto sumih = compute_dot(ybh.qs, xv+4);
                if constexpr (nrc_y == 1) {
                    accd[2*iy+0] = vfmaq_f32(accd[2*iy+0], dyl, vcvtq_f32_s32(sumil));
                    accd[2*iy+1] = vfmaq_f32(accd[2*iy+1], dyh, vcvtq_f32_s32(sumih));
                } else {
                    accd[iy] = vfmaq_f32(accd[iy], dyl, vcvtq_f32_s32(sumil));
                    accd[iy] = vfmaq_f32(accd[iy], dyh, vcvtq_f32_s32(sumih));
                }
            }
        }

        if constexpr (nrc_y == 1) {
            info.store(ix, 0, vaddvq_f32(vaddq_f32(accd[0], accd[1])));
        } else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                info.store(ix, iy, vaddvq_f32(accd[iy]));
            }
        }
    }
}

void iqk_dequantize_iq2_kt_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);
    const int nb = n/QK_K;

    Trellis3 trellis;

    auto values = vld1q_s8(iq4k_values);

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const block_iq2_kt * x8[8];
    float dkt[8];
    float ls[8], ls_all[64];
    uint32_t idx[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char*)vx + (ix+k)*bx);
            dkt[k] = dptr[0] * 1.05f;
            x8[k] = (const block_iq2_kt *)(dptr + 1);
        }
        auto vd = vld1q_f32_x2(dkt);

        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                auto u32 = *(const uint32_t *)x8[k][i].scales;
                auto s8_u32 = uint32x2_t{u32, u32 >> 4};
                s8_u32 = vand_u8(s8_u32, vdup_n_u32(0x0f0f0f0f));
                auto s8 = vqtbl1_s8(values, vreinterpret_u8_u32(s8_u32));
                auto s16 = vmovl_s8(s8);
                vst1q_f32(ls_all + 8*k + 0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16))));
                vst1q_f32(ls_all + 8*k + 4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16))));
            }
            for (int ib = 0; ib < QK_K/32; ++ib) {
                for (int k = 0; k < 8; ++k) ls[k] = ls_all[8*k+ib];
                auto scales1 = vmulq_f32(vd.val[0], vld1q_f32(ls+0));
                auto scales2 = vmulq_f32(vd.val[1], vld1q_f32(ls+4));
                vst1_f16((float16_t *)y[ib].d+0, vcvt_f16_f32(scales1));
                vst1_f16((float16_t *)y[ib].d+4, vcvt_f16_f32(scales2));
                for (int j = 0; j < 4; ++j) {
                    for (int k = 0; k < 8; ++k) {
                        const uint16_t * ql = (const uint16_t *)x8[k][i].ql;
                        idx[k] = ql[4*ib+j] + 4096;
                    }
                    vst1q_s8_x4(y[ib].qs+64*j, trellis.next64(idx));
                }
            }
            y += 8; // = QK_K/32;
        }
    }
}

template <int nrc_y>
void mul_mat_iq2_kt_q8_0_x4_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Trellis3 trellis;

    auto values = vld1q_s8(iq4k_values);

    constexpr int k_acc = nrc_y == 1 ? 2 : nrc_y;

    float32x4_t accd[k_acc];

    const block_q8_0_x4 * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        y[iy] = (const block_q8_0_x4 *)info.src1_row(iy);
    }

    int8x16x2_t xv[8];
    int32x4x4_t dot;

    auto compute_dot = [&dot] (const int8_t * y, const int8x16x2_t * xv) {
        for (int k = 0; k < 4; ++k) {
            auto yv = vld1q_s8_x2(y + 32*k);
            dot.val[k] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), xv[k].val[0], yv.val[0]), xv[k].val[1], yv.val[1]);
        }
        dot.val[0] = vpaddq_s32(dot.val[0], dot.val[1]);
        dot.val[2] = vpaddq_s32(dot.val[2], dot.val[3]);
        return vpaddq_s32(dot.val[0], dot.val[2]);
    };

    float32x4x2_t scales;

    for (int ix = 0; ix < nrc_x; ++ix) {
        const float * dptr = (const float *)((const char*)vx + ix*bx);
        auto d = vdupq_n_f32(dptr[0]*1.05f);
        const block_iq2_kt * x = (const block_iq2_kt *)(dptr + 1);

        for (int iy = 0; iy < k_acc; ++iy) accd[iy] = vdupq_n_f32(0);

        for (int i = 0; i < nb; ++i) {
            auto u32 = *(const uint32_t *)x[i].scales;
            auto s8_u32 = uint32x2_t{u32, u32 >> 4};
            s8_u32 = vand_u8(s8_u32, vdup_n_u32(0x0f0f0f0f));
            auto s8 = vqtbl1_s8(values, vreinterpret_u8_u32(s8_u32));
            auto s16 = vmovl_s8(s8);
            scales.val[0] = vmulq_f32(d, vcvtq_f32_s32(vmovl_s16(vget_low_s16 (s16))));
            scales.val[1] = vmulq_f32(d, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16))));
            const uint16_t * ql = (const uint16_t *)x[i].ql;
            if constexpr (nrc_y == 1) {
                const block_q8_0_x4& ybl = y[0][2*i+0];
                const block_q8_0_x4& ybh = y[0][2*i+1];
                auto dyl = vmulq_f32(scales.val[0], vcvt_f32_f16(vld1_f16((const float16_t *)ybl.d)));
                auto dyh = vmulq_f32(scales.val[1], vcvt_f32_f16(vld1_f16((const float16_t *)ybh.d)));
                int32x4x4_t suml = {};
                int32x4x4_t sumh = {};
                for (int ib = 0; ib < 4; ++ib) {
                    auto xl = trellis.next32(ql + 4*ib +  0, 4096);
                    auto xh = trellis.next32(ql + 4*ib + 16, 4096);
                    auto yl = vld1q_s8_x2(ybl.qs + 32*ib);
                    auto yh = vld1q_s8_x2(ybh.qs + 32*ib);
                    suml.val[ib] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), xl.val[0], yl.val[0]), xl.val[1], yl.val[1]);
                    sumh.val[ib] = vdotq_s32(vdotq_s32(vdupq_n_s32(0), xh.val[0], yh.val[0]), xh.val[1], yh.val[1]);
                }
                auto sl1 = vpaddq_s32(suml.val[0], suml.val[1]);
                auto sl2 = vpaddq_s32(suml.val[2], suml.val[3]);
                auto sl  = vpaddq_s32(sl1, sl2);
                auto sh1 = vpaddq_s32(sumh.val[0], sumh.val[1]);
                auto sh2 = vpaddq_s32(sumh.val[2], sumh.val[3]);
                auto sh  = vpaddq_s32(sh1, sh2);
                accd[0] = vfmaq_f32(accd[0], dyl, vcvtq_f32_s32(sl));
                accd[1] = vfmaq_f32(accd[1], dyh, vcvtq_f32_s32(sh));
            } else {
            for (int k = 0; k < 8; ++k) xv[k] = trellis.next32(ql + 4*k, 4096);
            for (int iy = 0; iy < nrc_y; ++iy) {
                const block_q8_0_x4& ybl = y[iy][2*i+0];
                const block_q8_0_x4& ybh = y[iy][2*i+1];
                auto dyl = vmulq_f32(scales.val[0], vcvt_f32_f16(vld1_f16((const float16_t *)ybl.d)));
                auto dyh = vmulq_f32(scales.val[1], vcvt_f32_f16(vld1_f16((const float16_t *)ybh.d)));
                auto sumil = compute_dot(ybl.qs, xv+0);
                auto sumih = compute_dot(ybh.qs, xv+4);
                if constexpr (nrc_y == 1) {
                    accd[2*iy+0] = vfmaq_f32(accd[2*iy+0], dyl, vcvtq_f32_s32(sumil));
                    accd[2*iy+1] = vfmaq_f32(accd[2*iy+1], dyh, vcvtq_f32_s32(sumih));
                } else {
                    accd[iy] = vfmaq_f32(accd[iy], dyl, vcvtq_f32_s32(sumil));
                    accd[iy] = vfmaq_f32(accd[iy], dyh, vcvtq_f32_s32(sumih));
                }
            }
            }
        }

        if constexpr (nrc_y == 1) {
            info.store(ix, 0, vaddvq_f32(vaddq_f32(accd[0], accd[1])));
        } else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                info.store(ix, iy, vaddvq_f32(accd[iy]));
            }
        }
    }
}

}

bool iqk_set_kernels_ktquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {


    if (ne00%QK_K !=  0) return false;

    if (ggml_type(typeA) == GGML_TYPE_IQ4_KT) {
        if (ggml_type(typeB) == GGML_TYPE_Q8_0_X4) {
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_kt_q8_0_x4_T, kernels);
            func16 = nullptr;
            return true;
        }
        return false;
    }

    if (ggml_type(typeA) == GGML_TYPE_IQ2_KT) {
        if (ggml_type(typeB) == GGML_TYPE_Q8_0_X4) {
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_kt_q8_0_x4_T, kernels);
            func16 = nullptr;
            return true;
        }
        return false;
    }

    if (ggml_type(typeB) != GGML_TYPE_F16) {
        return false;
    }

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

    func16 = nullptr;

    return true;
}

bool iqk_dequantize_ktquants(int type, int n, const void * vx, size_t bx, void * y, size_t stride_y, int nrc_x) {
    switch (type) {
        case GGML_TYPE_IQ2_KT: iqk_dequantize_iq2_kt_q80_r8(n, vx, bx, y, nrc_x); break;
        case GGML_TYPE_IQ3_KT: iqk_dequantize_iq3_kt(n, vx, bx, (float16_t *)y, stride_y, nrc_x); break;
        case GGML_TYPE_IQ4_KT: iqk_dequantize_iq4_kt_q80_r8(n, vx, bx, y, nrc_x); break;
        default: return false;
    }

    return true;
}

#endif

#endif
