#include "iqk_gemm_legacy_quants.h"

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

//
// ============================== Legacy quants
//

#ifdef __x86_64__

namespace {

struct DotHelper {
    const __m256i m1 = _mm256_set1_epi16(1);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    inline __m256i dot(__m256i x, __m256i y) const {
        return _mm256_dpbusd_epi32(_mm256_setzero_si256(), x, y);
    }
#else
    inline __m256i dot(__m256i x, __m256i y) const {
        return _mm256_madd_epi16(m1, _mm256_maddubs_epi16(x, y));
    }
#endif
};

struct SignedDot {
    DotHelper helper;
    inline __m256i compute(__m256i x, __m256i y) const {
        return helper.dot(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x));
    }
};
struct UnsignedDot {
    DotHelper helper;
    inline __m256i compute(__m256i x, __m256i y) const {
        return helper.dot(x, y);
    }
};

template <typename Q8, typename Q8x4, typename Dot, bool can_pack = true> struct Sum4 {
    Dot dot;
    inline __m256i compute(const __m256i * qx, const Q8 * y) const {
        const Q8x4 * y4 = (const Q8x4 *)y;
        const __m256i p0 = dot.compute(qx[0], _mm256_loadu_si256((const __m256i *)y4->qs+0)); // 8x block 0
        const __m256i p1 = dot.compute(qx[1], _mm256_loadu_si256((const __m256i *)y4->qs+1)); // 8x block 1
        const __m256i p2 = dot.compute(qx[2], _mm256_loadu_si256((const __m256i *)y4->qs+2)); // 8x block 2
        const __m256i p3 = dot.compute(qx[3], _mm256_loadu_si256((const __m256i *)y4->qs+3)); // 8x block 3
        if constexpr (can_pack) {
            const __m256i p01 = _mm256_madd_epi16(dot.helper.m1, _mm256_packs_epi32(p0, p1));    // 0,0, 1,1, 0,0, 1,1
            const __m256i p23 = _mm256_madd_epi16(dot.helper.m1, _mm256_packs_epi32(p2, p3));    // 2,2, 3,3, 2,2, 3,3
            return _mm256_madd_epi16(dot.helper.m1, _mm256_packs_epi32(p01, p23)); // 0,1,2,3, 0,1,2,3
        } else {
            // Note to myself: this is much faster than using _mm256_hadd_epi32()
            auto p01 = _mm256_add_epi32(_mm256_unpacklo_epi32(p0, p1), _mm256_unpackhi_epi32(p0, p1)); // 0,1, 0,1, 0,1, 0,1
            auto p23 = _mm256_add_epi32(_mm256_unpacklo_epi32(p2, p3), _mm256_unpackhi_epi32(p2, p3)); // 2,3, 2,3, 2,3, 2,3
            return _mm256_add_epi32(_mm256_unpacklo_epi64(p01, p23), _mm256_unpackhi_epi64(p01, p23)); // 0,1,2,3, 0,1,2,3
        }
    }
    inline __m256i compute(__m256i x, __m256i y) const { return dot.compute(x, y); }
};

template <typename Q8, typename Q8x4> struct Sum4q4 {
    inline __m256i compute(const __m256i * qx, const Q8 * y) const {
        const Q8x4 * y4 = (const Q8x4 *)y;
        auto p0 = _mm256_maddubs_epi16(qx[0], _mm256_loadu_si256((const __m256i *)y4->qs+0)); // 16x block 0
        auto p1 = _mm256_maddubs_epi16(qx[1], _mm256_loadu_si256((const __m256i *)y4->qs+1)); // 16x block 1
        auto p2 = _mm256_maddubs_epi16(qx[2], _mm256_loadu_si256((const __m256i *)y4->qs+2)); // 16x block 2
        auto p3 = _mm256_maddubs_epi16(qx[3], _mm256_loadu_si256((const __m256i *)y4->qs+3)); // 16x block 3
        auto p01 = _mm256_add_epi16(_mm256_unpacklo_epi32(p0, p1), _mm256_unpackhi_epi32(p0, p1)); // 0,0, 1,1, 0,0, 1,1, 0,0, 1,1, 0,0, 1,1
        auto p23 = _mm256_add_epi16(_mm256_unpacklo_epi32(p2, p3), _mm256_unpackhi_epi32(p2, p3)); // 2,2, 3,3, 2,2, 3,3, 2,2, 3,3, 2,2, 3,3
        auto p0123 = _mm256_add_epi16(_mm256_unpacklo_epi64(p01, p23), _mm256_unpackhi_epi64(p01, p23)); // 0,0, 1,1, 2,2, 3,3, 0,0, 1,1, 2,2, 3,3
        return _mm256_madd_epi16(_mm256_set1_epi16(1), p0123);
    }
    inline __m256i compute(__m256i x, __m256i y) const { return _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(x, y)); }
};

struct ScaleHelperQ8_0 {
    inline __m128 prepare4(const block_q8_0 * y) {
        const block_q8_0_x4 * y4 = (const block_q8_0_x4 *)y;
        return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)y4->d));
    }
    inline __m128 prepare4(__m128 other_scales, const block_q8_0 * y) {
        return _mm_mul_ps(other_scales, prepare4(y));
    }
    template <typename Q> inline float prepare1(const Q * y) const { return GGML_FP16_TO_FP32(y->d); }
    template <typename Q> inline float prepare1(float d, const Q * y) const { return d*prepare1(y); }
};

struct ScaleHelperQ_0 {
    ggml_half scales8[4];
    template <typename Q>
    inline __m128 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) scales8[j] = y[j].d;
        return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)scales8));
    }
    template <typename Q>
    inline __m128 prepare4(__m128 other_scales, const Q * y) {
        return _mm_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline float prepare1(const Q * y) const { return GGML_FP16_TO_FP32(y->d); }
    template <typename Q> inline float prepare1(float d, const Q * y) const { return d*prepare1(y); }
};

template <int min_value>
struct ScaleHelperQ_0_1 {
    ggml_half scales8[4];
    template <typename Q>
    inline __m256 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) scales8[j] = y[j].d;
        auto s4 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)scales8));
        return _mm256_set_m128(_mm_mul_ps(s4, min), s4);
    }
    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm_mul256_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const Q * y) const {
        float d = GGML_FP16_TO_FP32(y->d);
        return std::make_pair(d, -d*float(min_value));
    }
    std::pair<float, float> inline prepare1(const std::pair<float, float>& dm, const block_q8_1 * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->s));
    }
    const __m128 min = _mm_set1_ps(float(-min_value));
};

//template <int min_value>
//struct ScaleHelperQ_0_2 {
//    ggml_bf16_t scales8[4];
//    template <typename Q>
//    inline __m256 prepare4(const Q * y) {
//        for (int j = 0; j < 4; ++j) scales8[j] = y[j].d;
//        auto s4 = _mm_castsi128_ps(_mm_slli_epi16(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)scales8)), 16));
//        return _mm256_set_m128(_mm_mul_ps(s4, min), s4);
//    }
//    template <typename Q>
//    inline __m256 prepare4(__m256 other_scales, const Q * y) {
//        return _mm_mul256_ps(other_scales, prepare4<Q>(y));
//    }
//    template <typename Q> inline std::pair<float, float> prepare1(const Q * y) const {
//        float d = GGML_BF16_TO_FP32(y->d);
//        return std::make_pair(d, -d*float(min_value));
//    }
//    std::pair<float, float> inline prepare1(const std::pair<float, float>& dm, const block_q8_1 * y) const {
//        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->s));
//    }
//    const __m128 min = _mm_set1_ps(float(-min_value));
//};

struct ScaleHelperQ8_1 {
    template <typename Q>
    inline __m256 prepare4(const Q * y) {
        const block_q8_1_x4 * y4 = (const block_q8_1_x4 *)y;
        return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)y4->d));
    }
    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm256_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const Q * y) const {
        return std::make_pair(GGML_FP16_TO_FP32(y->d), GGML_FP16_TO_FP32(y->m));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const std::pair<float, float>& dm, const Q * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->m));
    }
    std::pair<float, float> inline prepare1(const std::pair<float, float>& dm, const block_q8_1 * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->s));
    }
};

struct ScaleHelperQ8_2 {
    template <typename Q>
    inline __m256 prepare4(const Q * y) {
        const block_q8_2_x4 * y4 = (const block_q8_2_x4 *)y;
        auto aux = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)y4->d));
        return _mm256_castsi256_ps(_mm256_slli_epi32(aux, 16));
    }
    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm256_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const Q * y) const {
        return std::make_pair(GGML_BF16_TO_FP32(y->d), GGML_BF16_TO_FP32(y->m));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const std::pair<float, float>& dm, const Q * y) const {
        ggml_bf16_t d, s; d.bits = y->d; s.bits = y->s;
        return std::make_pair(dm.first*GGML_BF16_TO_FP32(d), dm.second*GGML_BF16_TO_FP32(s));
    }
    std::pair<float, float> inline prepare1(const std::pair<float, float>& dm, const block_q8_2 * y) const {
        ggml_bf16_t d, s; d.bits = y->d; s.bits = y->s;
        return std::make_pair(dm.first*GGML_BF16_TO_FP32(d), dm.second*GGML_BF16_TO_FP32(s));
    }
};

struct ScaleHelperQ_1 {
    uint32_t scales8[4];
    const __m128i shuffle = _mm_set_epi16(0x0f0e, 0x0b0a, 0x0706, 0x0302, 0x0d0c, 0x0908, 0x0504, 0x0100);

    template <typename Q>
    inline __m256 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) {
            // it is slightly faster to directly dereference (const uint32 *)&y[j].d, but some compilers
            // complain that this breaks strict-aliasing rules.
            memcpy(scales8 + j, &y[j].d, sizeof(uint32_t));
        }
        return _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)scales8), shuffle));
    }

    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm256_mul_ps(other_scales, prepare4<Q>(y));
    }

    template <typename Q> inline std::pair<float, float> prepare1(const Q * y) const {
        return std::make_pair(GGML_FP16_TO_FP32(y->d), GGML_FP16_TO_FP32(y->m));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const std::pair<float, float>& dm, const Q * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->m));
    }
    std::pair<float, float> inline prepare1(const std::pair<float, float>& dm, const block_q8_1 * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->s));
    }
};

struct MinusType0 {
    inline __m256 compute(__m128 d, int) const { return _mm256_set_m128(d, d); }
    inline float compute(float d, int) const { return d; }
    inline float result(__m256 acc, int) const { return hsum_float_8(acc); }
    inline __m256 vresult(__m256 acc, int) const { return acc; }
};

template <int nrc_y> struct MinusType1 {
    __m128 accm[nrc_y];
    MinusType1() { for (int iy = 0; iy < nrc_y; ++iy) accm[iy] = _mm_setzero_ps(); }
    inline __m256 compute(__m256 dm, int iy) {
        const __m128 d = _mm256_castps256_ps128(dm);
        const __m128 m = _mm256_extractf128_ps(dm, 1);
        accm[iy] = _mm_add_ps(accm[iy], m);
        return _mm256_set_m128(d, d);
    }
    inline float compute(const std::pair<float, float>& dm, int iy) {
        accm[iy] = _mm_add_ps(accm[iy], _mm_set1_ps(dm.second*0.25f));
        return dm.first;
    }
    inline float result(__m256 acc, int iy) const {
        const __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        return hsum_float_4(_mm_add_ps(sum, accm[iy]));
    }
    inline __m256 vresult(__m256 acc, int iy) const {
        return _mm256_add_ps(acc, _mm256_insertf128_ps(_mm256_setzero_ps(), accm[iy], 0));
    }
};

template <typename Minus, int nrc_y, bool is_multiple_of_4> struct AccumT {
    __m256 acc[nrc_y];
    Minus accm;
    AccumT() {  for (int iy = 0; iy < nrc_y; ++iy) acc[iy] = _mm256_setzero_ps(); }
    template <typename Unpacker, typename Scales, typename Sum, typename Q8>
    inline void compute(int nb, Unpacker& unp, Scales& scales, Sum& sum, const Q8 ** y, const DataInfo& info, int ix) {
        auto qx = unp.quants();
        __m256 dall[nrc_y];
        for (int i = 0; i < nb/4; ++i) {
            auto other_scales = unp.set_block_4(i);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto s12 = scales.prepare4(other_scales, y[iy] + 4*i);
                dall[iy] = accm.compute(s12, iy);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto pall = sum.compute(qx, y[iy] + 4*i);
                acc[iy] = _mm256_fmadd_ps(dall[iy], _mm256_cvtepi32_ps(pall), acc[iy]);
            }
        }
        if (!is_multiple_of_4) {
            for (int i = 4*(nb/4); i < nb; ++i) {
                auto other_scales = unp.set_block(i);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto s12 = scales.prepare1(other_scales, y[iy] + i);
                    auto d = accm.compute(s12, iy);
                    const __m256i p0 = sum.compute(qx[0], _mm256_loadu_si256((const __m256i *)y[iy][i].qs));
                    acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, accm.result(acc[iy], iy));
        }
    }
    template <typename Unpacker, typename Scales, typename Sum, typename Q8>
    inline void compute(int nb, Unpacker& unp, Scales& scales, Sum& sum, const Q8 ** y, __m256 * result) {
        auto qx = unp.quants();
        __m256 dall[nrc_y];
        for (int i = 0; i < nb/4; ++i) {
            auto other_scales = unp.set_block_4(i);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto s12 = scales.prepare4(other_scales, y[iy] + 4*i);
                dall[iy] = accm.compute(s12, iy);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto pall = sum.compute(qx, y[iy] + 4*i);
                acc[iy] = _mm256_fmadd_ps(dall[iy], _mm256_cvtepi32_ps(pall), acc[iy]);
            }
        }
        if (!is_multiple_of_4) {
            for (int i = 4*(nb/4); i < nb; ++i) {
                auto other_scales = unp.set_block(i);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto s12 = scales.prepare1(other_scales, y[iy] + i);
                    auto d = accm.compute(s12, iy);
                    const __m256i p0 = sum.compute(qx[0], _mm256_loadu_si256((const __m256i *)y[iy][i].qs));
                    acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            result[iy] = accm.vresult(acc[iy], iy);
        }
    }
};

template <int nrc_y, bool is_multiple_of_4>
using AccumType0 = AccumT<MinusType0, nrc_y, is_multiple_of_4>;

template <int nrc_y, bool is_multiple_of_4>
using AccumType1 = AccumT<MinusType1<nrc_y>, nrc_y, is_multiple_of_4>;

using Sum4TypeQ80 = Sum4<block_q8_0, block_q8_0_x4, SignedDot, false>;
using Sum4TypeQ82 = Sum4<block_q8_2, block_q8_2_x4, UnsignedDot, false>;

template <typename Unpacker, typename AccumType, typename Scales, typename Q8, int nrc_y>
void mul_mat_qX_q8_Helper(int nb, const void * vx, size_t bx, const DataInfo& info, const Q8 ** y, int nrc_x) {
    Unpacker unp(vx, bx);
    typename Unpacker::Sum4T sum4;
    Scales scales;
    for (int ix = 0; ix < nrc_x; ++ix) {
        unp.set_row(ix);
        AccumType accum;
        accum.compute(nb, unp, scales, sum4, y, info, ix);
    }
}

template <typename Unpacker, typename AccumType, typename Scales, typename Q8, int nrc_y>
void mul_mat_qX_q8_Helper_x2(int nb, const void * vx, size_t bx, const DataInfo& info, const Q8 ** y, int nrc_x) {
    GGML_ASSERT(nrc_x%2 == 0);
    Unpacker unp(vx, bx);
    typename Unpacker::Sum4T sum4;
    Scales scales;
    for (int ix = 0; ix < nrc_x; ix += 2) {
        unp.set_row(ix);
        AccumType accum;
        accum.compute(nb, unp, scales, sum4, y, info, ix);
    }
}

template <typename Unpacker, int nrc_y>
void mul_mat_qX_0_q8_0_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_0> q8(info);
    int nb = n/Unpacker::block_size();
    if (nb%4 == 0) {
        mul_mat_qX_q8_Helper<Unpacker, AccumType0<nrc_y, true>, ScaleHelperQ8_0, block_q8_0, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    } else {
        mul_mat_qX_q8_Helper<Unpacker, AccumType0<nrc_y, false>, ScaleHelperQ8_0, block_q8_0, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    }
}

template <typename Unpacker, int nrc_y, int nrc_x>
void mul_mat_qX_0_q8_0_Tx(int n, const void * vx, size_t bx, const DataInfo& info, int) {
    static_assert(8%nrc_y == 0);
    Q8<nrc_y, block_q8_0> q8(info);
    int nb = n/Unpacker::block_size();
    Unpacker unp(vx, bx);
    typename Unpacker::Sum4T sum4;
    ScaleHelperQ8_0 scales;
    __m256 result[8];
    auto store = [&info, &result] (int ix0) {
        if constexpr (nrc_y == 1) {
            info.store(ix0, 0, hsum_float_8x8(result));
        }
        else if constexpr (nrc_y == 2) {
            auto value = hsum_float_8x8(result);
            auto value1 = _mm256_extractf128_ps(value, 1);
            info.store(ix0, 0, _mm_shuffle_ps(_mm256_castps256_ps128(value), value1, 0x88));
            info.store(ix0, 1, _mm_shuffle_ps(_mm256_castps256_ps128(value), value1, 0xdd));
        }
        else {
            float val[8];
            _mm256_storeu_ps(val, hsum_float_8x8(result));
            for (int iy = 0; iy < nrc_y; ++iy) for (int ix = 0; ix < 8/nrc_y; ++ix) info.store(ix0+ix, iy, val[nrc_y*ix+iy]);
        }
    };
    if (nb%4 == 0) {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 8/nrc_y) {
            for (int ix = 0; ix < 8/nrc_y; ++ix) {
                unp.set_row(ix0 + ix);
                AccumType0<nrc_y, true> accum;
                accum.compute(nb, unp, scales, sum4, q8.y, result + nrc_y*ix);
            }
            store(ix0);
        }
    } else {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 8/nrc_y) {
            for (int ix = 0; ix < 8/nrc_y; ++ix) {
                unp.set_row(ix0 + ix);
                AccumType0<nrc_y, false> accum;
                accum.compute(nb, unp, scales, sum4, q8.y, result + nrc_y*ix);
            }
            store(ix0);
        }
    }
}

template <typename Unpacker, int nrc_y>
void mul_mat_qX_1_q8_1_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_1> q8(info);
    int nb = n/Unpacker::block_size();
    if (nb%4 == 0) {
        mul_mat_qX_q8_Helper<Unpacker, AccumType1<nrc_y, true>, ScaleHelperQ8_1, block_q8_1, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    } else {
        mul_mat_qX_q8_Helper<Unpacker, AccumType1<nrc_y, false>, ScaleHelperQ8_1, block_q8_1, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    }
}

template <typename Unpacker, int nrc_y>
void mul_mat_qX_1_q8_2_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_2> q8(info);
    int nb = n/Unpacker::block_size();
    if (nb%4 == 0) {
        mul_mat_qX_q8_Helper<Unpacker, AccumType1<nrc_y, true>, ScaleHelperQ8_2, block_q8_2, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    } else {
        mul_mat_qX_q8_Helper<Unpacker, AccumType1<nrc_y, false>, ScaleHelperQ8_2, block_q8_2, nrc_y>(
                nb, vx, bx, info, q8.y, nrc_x
        );
    }
}

template <typename Unpacker, int nrc_y, int nrc_x>
void mul_mat_qX_0_q8_2_Tx(int n, const void * vx, size_t bx, const DataInfo& info, int) {
    static_assert(8%nrc_y == 0);
    Q8<nrc_y, block_q8_2> q8(info);
    int nb = n/Unpacker::block_size();
    Unpacker unp(vx, bx);
    typename Unpacker::Sum4T sum4;
    ScaleHelperQ8_2 scales;
    __m256 result[8];
    auto store = [&info, &result] (int ix0) {
        if constexpr (nrc_y == 1) {
            info.store(ix0, 0, hsum_float_8x8(result));
        }
        else if constexpr (nrc_y == 2) {
            auto value = hsum_float_8x8(result);
            auto value1 = _mm256_extractf128_ps(value, 1);
            info.store(ix0, 0, _mm_shuffle_ps(_mm256_castps256_ps128(value), value1, 0x88));
            info.store(ix0, 1, _mm_shuffle_ps(_mm256_castps256_ps128(value), value1, 0xdd));
        }
        else {
            float val[8];
            _mm256_storeu_ps(val, hsum_float_8x8(result));
            for (int iy = 0; iy < nrc_y; ++iy) for (int ix = 0; ix < 8/nrc_y; ++ix) info.store(ix0+ix, iy, val[nrc_y*ix+iy]);
        }
    };
    if (nb%4 == 0) {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 8/nrc_y) {
            for (int ix = 0; ix < 8/nrc_y; ++ix) {
                unp.set_row(ix0 + ix);
                AccumType1<nrc_y, true> accum;
                accum.compute(nb, unp, scales, sum4, q8.y, result + nrc_y*ix);
            }
            store(ix0);
        }
    } else {
        for (int ix0 = 0; ix0 < nrc_x; ix0 += 8/nrc_y) {
            for (int ix = 0; ix < 8/nrc_y; ++ix) {
                unp.set_row(ix0 + ix);
                AccumType1<nrc_y, false> accum;
                accum.compute(nb, unp, scales, sum4, q8.y, result + nrc_y*ix);
            }
            store(ix0);
        }
    }
}

struct Dequantizer4bit {
    const __m256i m4 = _mm256_set1_epi8(0xf);
    inline __m256i dequant(const uint8_t * qs) const {
        const __m128i aux128 = _mm_loadu_si128((const __m128i *)qs);
        return _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(aux128, 4), aux128), m4);
    }
};

struct Q8_0_Dequantizer {
    inline __m256i dequant(const block_q8_0 * x) const {
        return _mm256_loadu_si256((const __m256i *)x->qs);
    }
};

struct Q8_0_1_Dequantizer {
    inline __m256i dequant(const block_q8_0 * x) const {
        return _mm256_add_epi8(_mm256_set1_epi8(127), _mm256_loadu_si256((const __m256i *)x->qs));
    }
};

struct Q4_0_Dequantizer {
    Dequantizer4bit b4;
    const __m256i m8 = _mm256_set1_epi8(-8);
    inline __m256i dequant(const block_q4_0 * x) const {
        return _mm256_add_epi8(b4.dequant(x->qs), m8);
    }
};

struct Q4_0_1_Dequantizer {
    Dequantizer4bit b4;
    inline __m256i dequant(const block_q4_0 * x) const {
        return b4.dequant(x->qs);
    }
};

struct IQ4_NL_Dequantizer {
    Dequantizer4bit b4;
#ifdef HAVE_FANCY_SIMD
    const __m256i values = load_iq4nl_values_256();
#else
    const __m256i values = load_iq4k_values_256();
#endif
    inline __m256i dequant(const block_iq4_nl * x) const {
        return _mm256_shuffle_epi8(values, b4.dequant(x->qs));
    }
};

struct Q4_1_Dequantizer {
    Dequantizer4bit b4;
    inline __m256i dequant(const block_q4_1 * x) const {
        return b4.dequant(x->qs);
    }
};

struct HBitDequantizer {
    const __m256i shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    const __m256i mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    const __m256i minus1 = _mm256_set1_epi64x(-1);
    inline __m256i to_bytes(const uint8_t * bits) const {
        // Note: Data in all ggml quants is at least 2-byte aligned.
        // => we can cast to uint16_t and use or on two consecutive entries
        // which is faster than memcpy
        const uint16_t * aux16 = (const uint16_t *)bits;
        const uint32_t aux32 = aux16[0] | (aux16[1] << 16);
        //uint32_t aux32; memcpy(&aux32, bits, sizeof(uint32_t));
        __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(aux32), shuffle);
        bytes = _mm256_or_si256(bytes, mask);
        return _mm256_cmpeq_epi8(bytes, minus1);
    }
};

struct Q5_0_Dequantizer {
    Dequantizer4bit b4;
    HBitDequantizer hbit;
    const __m256i mh = _mm256_set1_epi8((char)0xF0);
    inline __m256i dequant(const block_q5_0 * x) const {
        const __m256i vqh = _mm256_andnot_si256(hbit.to_bytes(x->qh), mh);
        return _mm256_or_si256(b4.dequant(x->qs), vqh);
    }
};

template <typename Q5>
struct Q5_1_Dequantizer {
    Dequantizer4bit b4;
    HBitDequantizer hbit;
    const __m256i mh = _mm256_set1_epi8(0x10);
    inline __m256i dequant(const Q5 * x) const {
        const __m256i vqh = _mm256_and_si256(hbit.to_bytes(x->qh), mh);
        return _mm256_or_si256(b4.dequant(x->qs), vqh);
    }
};
struct Q6_0_1_Dequantizer {
    Dequantizer4bit b4;
    const __m256i mh = _mm256_set1_epi8(0x30);
    const __m256i shift1 = _mm256_set_epi64x(0, 2, 0, 4);
    const __m256i shift2 = _mm256_set_epi64x(2, 0, 0, 0);
    inline __m256i dequant(const block_q6_0 * x) const {
        uint64_t aux64; std::memcpy(&aux64, x->qh, 8);
        auto h256 = _mm256_sllv_epi64(_mm256_set1_epi64x(aux64), shift1);
        return _mm256_or_si256(b4.dequant(x->qs), _mm256_and_si256(_mm256_srlv_epi64(h256, shift2), mh));
    }
};

template <typename Q, typename Scales, typename Dequantizer>
struct Q_Unpacker {
    Q_Unpacker(const void * vx, size_t bx) : cx_0((const char *)vx), x((const Q*)cx_0), bx(bx) {}

    const char * cx_0;
    const Q    * x;
    size_t       bx;

    Scales scales;
    Dequantizer deq;

    __m256i qx[4];

    inline const __m256i* quants() const { return qx; }

    inline void set_row(int ix) { x = (const Q*)(cx_0 + ix*bx); }

    inline auto set_block_4(int i) {
        for (int j = 0; j < 4; ++j) {
            qx[j] = deq.dequant(x + 4*i + j);
        }
        return scales.prepare4(x + 4*i);
    }
    inline auto set_block(int i) {
        qx[0] = deq.dequant(x + i);
        return scales.prepare1(x + i);
    }
};

struct Q8_0_Unpacker final : public Q_Unpacker<block_q8_0, ScaleHelperQ_0, Q8_0_Dequantizer> {
    Q8_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ80;
    inline static int block_size() { return QK8_0; }
};
struct Q8_0_1_Unpacker final : public Q_Unpacker<block_q8_0, ScaleHelperQ_0_1<127>, Q8_0_1_Dequantizer> {
    Q8_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK8_0; }
};
struct Q4_0_Unpacker final : public Q_Unpacker<block_q4_0, ScaleHelperQ_0, Q4_0_Dequantizer> {
    Q4_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ80;
    inline static int block_size() { return QK4_0; }
};
struct Q4_0_1_Unpacker final : public Q_Unpacker<block_q4_0, ScaleHelperQ_0_1<8>, Q4_0_1_Dequantizer> {
    Q4_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    //using Sum4T = Sum4TypeQ82;
    using Sum4T = Sum4q4<block_q8_2, block_q8_2_x4>;
    inline static int block_size() { return QK4_0; }
};
#ifdef HAVE_FANCY_SIMD
struct IQ4_NL_Unpacker final : public Q_Unpacker<block_iq4_nl, ScaleHelperQ_0_1<128>, IQ4_NL_Dequantizer> {
    IQ4_NL_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK4_NL; }
};
#else
struct IQ4_NL_Unpacker final : public Q_Unpacker<block_iq4_nl, ScaleHelperQ_0, IQ4_NL_Dequantizer> {
    IQ4_NL_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ80;
    inline static int block_size() { return QK4_NL; }
};
#endif
struct Q5_0_Unpacker final : public Q_Unpacker<block_q5_0, ScaleHelperQ_0, Q5_0_Dequantizer> {
    Q5_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ80;
    inline static int block_size() { return QK5_0; }
};
struct Q5_0_1_Unpacker final : public Q_Unpacker<block_q5_0, ScaleHelperQ_0_1<16>, Q5_1_Dequantizer<block_q5_0>> {
    Q5_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK5_0; }
};
struct Q4_1_Unpacker final : public Q_Unpacker<block_q4_1, ScaleHelperQ_1, Q4_1_Dequantizer> {
    Q4_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK4_1; }
};
struct Q5_1_Unpacker final : public Q_Unpacker<block_q5_1, ScaleHelperQ_1, Q5_1_Dequantizer<block_q5_1>> {
    Q5_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK5_1; }
};
struct Q6_0_1_Unpacker final : public Q_Unpacker<block_q6_0, ScaleHelperQ_0_1<32>, Q6_0_1_Dequantizer> {
    Q6_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK6_0; }
};

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_iq4_nl_r4_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm512_set1_epi8(0xf);
    auto values = load_iq4nl_values_512();
    int nb = n / QK4_NL;
    __m512  acc[2*nrc_y] = {};
    __m512i qx[4];
    float d8[8*nrc_y];
    auto prepare = [&qx, &m4, &values] (const block_iq4_nl_r4& iq4l, const block_iq4_nl_r4& iq4h) {
        auto scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq4l.d));
        auto scales1 = _mm256_set_m128(scales128, scales128);
        scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq4h.d));
        auto scales2 = _mm256_set_m128(scales128, scales128);
        auto scales = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
        auto bits1 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq4l.qs+0)),
                                                               _mm256_loadu_si256((const __m256i *)iq4h.qs+0), 1);
        auto bits2 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq4l.qs+1)),
                                                               _mm256_loadu_si256((const __m256i *)iq4h.qs+1), 1);
        qx[0] = _mm512_shuffle_epi8(values, _mm512_and_si512(bits1, m4));
        qx[1] = _mm512_shuffle_epi8(values, _mm512_and_si512(bits2, m4));
        qx[2] = _mm512_shuffle_epi8(values, _mm512_and_si512(_mm512_srli_epi16(bits1, 4), m4));
        qx[3] = _mm512_shuffle_epi8(values, _mm512_and_si512(_mm512_srli_epi16(bits2, 4), m4));
        return scales;
    };
    auto dot = [&qx] (__m256i y8) {
        auto y = _mm512_inserti32x8(_mm512_castsi256_si512(y8), y8, 1);
        auto sumi = _mm512_setzero_si512();
        sumi = _mm512_dpbusd_epi32(sumi, qx[0], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[1], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[2], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[3], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
        return sumi;
    };
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_iq4_nl_r4 * iq4l = (const block_iq4_nl_r4 *)((const char *)vx + (ix+0)*bx);
        const block_iq4_nl_r4 * iq4h = (const block_iq4_nl_r4 *)((const char *)vx + (ix+4)*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto aux = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d)), 16);
                _mm256_storeu_ps(d8+8*iy, _mm256_castsi256_ps(aux));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq4l[4*ib4+k], iq4h[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(_mm256_loadu_si256((const __m256i*)q8.y[iy][ib4].qs+k));
                    auto dy = _mm512_set1_ps(d8[8*iy+k]);
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(d8[8*iy+k+4]), acc[2*iy+1]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq4l[ib], iq4h[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(_mm256_loadu_si256((const __m256i*)qy[ib].qs));
                ggml_bf16_t d, s; d.bits = qy[ib].d; s.bits = qy[ib].s;
                auto dy = _mm512_set1_ps(GGML_BF16_TO_FP32(d));
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(GGML_BF16_TO_FP32(s)), acc[2*iy+1]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum512 = _mm512_fmadd_ps(_mm512_set1_ps(-64.f), acc[2*iy+1], acc[2*iy+0]);
            acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_ps();
            auto sum1 = _mm_add_ps(_mm512_extractf32x4_ps(sum512, 0), _mm512_extractf32x4_ps(sum512, 1));
            auto sum2 = _mm_add_ps(_mm512_extractf32x4_ps(sum512, 2), _mm512_extractf32x4_ps(sum512, 3));
            info.store(ix+0, iy, sum1);
            info.store(ix+4, iy, sum2);
        }
    }
}
#else
template <int nrc_y>
static void mul_mat_iq4_nl_r4_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto m1 = _mm256_set1_epi16(1);
    auto values128 = _mm_loadu_si128((const __m128i *)iq4k_values);
    auto values = MM256_SET_M128I(values128, values128);
    int nb = n / QK4_NL;
    __m256 acc[nrc_y] = {};
    __m256i qs[4];
    float d8[4*nrc_y];
    auto prepare = [&qs, &values, &m4] (const block_iq4_nl_r4& iq4) {
        auto scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq4.d));
        auto scales = _mm256_set_m128(scales128, scales128);
        auto bits1 = _mm256_loadu_si256((const __m256i *)iq4.qs+0);
        auto bits2 = _mm256_loadu_si256((const __m256i *)iq4.qs+1);
        qs[0] = _mm256_shuffle_epi8(values, _mm256_and_si256(bits1, m4));
        qs[1] = _mm256_shuffle_epi8(values, _mm256_and_si256(bits2, m4));
        qs[2] = _mm256_shuffle_epi8(values, _mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4));
        qs[3] = _mm256_shuffle_epi8(values, _mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4));
        return scales;
    };
    auto dot = [&qs, &m1] (__m256i y) {
        auto u1 = _mm256_sign_epi8(qs[0], qs[0]);
        auto u2 = _mm256_sign_epi8(qs[1], qs[1]);
        auto sumi1 = _mm256_add_epi32(
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(u1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qs[0]))),
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(u2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qs[1]))));
        u1 = _mm256_sign_epi8(qs[2], qs[2]);
        u2 = _mm256_sign_epi8(qs[3], qs[3]);
        auto sumi2 = _mm256_add_epi32(
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(u1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qs[2]))),
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(u2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qs[3]))));
        return _mm256_add_epi32(sumi1, sumi2);
    };
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_iq4_nl_r4 * iq4 = (const block_iq4_nl_r4 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto aux = _mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8.y[iy][ib4].d)), 16);
                _mm_storeu_ps(d8+4*iy, _mm_castsi128_ps(aux));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq4[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(_mm256_loadu_si256((const __m256i*)q8.y[iy][ib4].qs+k));
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq4[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(_mm256_loadu_si256((const __m256i*)qy[ib].qs));
                ggml_bf16_t d{qy[ib].d};
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(d)));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, sum);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}
#endif

inline void prepare_q4_0_quants_avx2(const uint8_t * qs, __m256i * v, const __m256i& m4) {
    auto bits1 = _mm256_loadu_si256((const __m256i *)qs+0);
    auto bits2 = _mm256_loadu_si256((const __m256i *)qs+1);
    auto bits3 = _mm256_loadu_si256((const __m256i *)qs+2);
    auto bits4 = _mm256_loadu_si256((const __m256i *)qs+3);
    v[0] = _mm256_and_si256(bits1, m4);
    v[1] = _mm256_and_si256(bits2, m4);
    v[2] = _mm256_and_si256(bits3, m4);
    v[3] = _mm256_and_si256(bits4, m4);
    v[4] = _mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4);
    v[5] = _mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4);
    v[6] = _mm256_and_si256(_mm256_srli_epi16(bits3, 4), m4);
    v[7] = _mm256_and_si256(_mm256_srli_epi16(bits4, 4), m4);
}

inline __m256i accum_q4_0_quants(const __m256i * v, const int8_t * qs) {
    auto y4l = _mm_loadu_si128((const __m128i*)qs+0);
    auto y4h = _mm_loadu_si128((const __m128i*)qs+1);
    auto yl  = MM256_SET_M128I(y4l, y4l);
    auto yh  = MM256_SET_M128I(y4h, y4h);
#ifdef HAVE_FANCY_SIMD
    auto sumi = _mm256_setzero_si256();
    sumi = _mm256_dpbusd_epi32(sumi, v[0], _mm256_shuffle_epi32(yl, 0x00));
    sumi = _mm256_dpbusd_epi32(sumi, v[1], _mm256_shuffle_epi32(yl, 0x55));
    sumi = _mm256_dpbusd_epi32(sumi, v[2], _mm256_shuffle_epi32(yl, 0xaa));
    sumi = _mm256_dpbusd_epi32(sumi, v[3], _mm256_shuffle_epi32(yl, 0xff));
    sumi = _mm256_dpbusd_epi32(sumi, v[4], _mm256_shuffle_epi32(yh, 0x00));
    sumi = _mm256_dpbusd_epi32(sumi, v[5], _mm256_shuffle_epi32(yh, 0x55));
    sumi = _mm256_dpbusd_epi32(sumi, v[6], _mm256_shuffle_epi32(yh, 0xaa));
    sumi = _mm256_dpbusd_epi32(sumi, v[7], _mm256_shuffle_epi32(yh, 0xff));
#else
    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(v[0], _mm256_shuffle_epi32(yl, 0x00)),
                                  _mm256_maddubs_epi16(v[1], _mm256_shuffle_epi32(yl, 0x55)));
    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(v[2], _mm256_shuffle_epi32(yl, 0xaa)),
                                  _mm256_maddubs_epi16(v[3], _mm256_shuffle_epi32(yl, 0xff)));
    auto sumi3 = _mm256_add_epi16(_mm256_maddubs_epi16(v[4], _mm256_shuffle_epi32(yh, 0x00)),
                                  _mm256_maddubs_epi16(v[5], _mm256_shuffle_epi32(yh, 0x55)));
    auto sumi4 = _mm256_add_epi16(_mm256_maddubs_epi16(v[6], _mm256_shuffle_epi32(yh, 0xaa)),
                                  _mm256_maddubs_epi16(v[7], _mm256_shuffle_epi32(yh, 0xff)));
    auto sumi = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_add_epi16(_mm256_add_epi16(sumi1, sumi2), _mm256_add_epi16(sumi3, sumi4)));
#endif
    return sumi;
}

template <int nrc_y>
static void mul_mat_q4_0_r8_q8_2_avx2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_1_x4> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    int nb = n / QK4_NL;
    __m256i v[8];
    GGML_ASSERT(nb%4 == 0);
    if constexpr (nrc_y == 1) {
        union { __m256 vec; float val[8]; } helper;
        for (int ix = 0; ix < nrc_x; ix += 8) {
            const block_iq4_nl_r8 * iq4 = (const block_iq4_nl_r8 *)((const char *)vx + ix*bx);
            auto acc1 = _mm256_setzero_ps();
            auto acc2 = _mm256_setzero_ps();
            for (int ib4 = 0; ib4 < nb/4; ++ib4) {
                helper.vec = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)q8.y[0][ib4].d)), 16));
                for (int k = 0; k < 4; ++k) {
                    auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[4*ib4+k].d));
                    prepare_q4_0_quants_avx2(iq4[4*ib4+k].qs, v, m4);
                    auto sumi = accum_q4_0_quants(v, q8.y[0][ib4].qs+32*k);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(helper.val[k]));
                    acc1 = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc1);
                    acc2 = _mm256_fmadd_ps(scales, _mm256_set1_ps(helper.val[k+4]), acc2);
                }
            }
            for (int ib = 4*(nb/4); ib < nb; ++ib) {
                auto qy = (const block_q8_1 *)q8.y[0];
                auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[ib].d));
                prepare_q4_0_quants_avx2(iq4[ib].qs, v, m4);
                auto sumi = accum_q4_0_quants(v, qy[ib].qs);
                ggml_bf16_t d{qy[ib].d}, s{qy[ib].s};
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(d)));
                acc1 = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc1);
                acc2 = _mm256_fmadd_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(s)), acc2);
            }
            acc1 = _mm256_fmadd_ps(acc2, _mm256_set1_ps(-8.f), acc1);
            info.store(ix, 0, acc1);
        }
    }
    else {
    __m256 acc[nrc_y] = {};
    float d8[8*nrc_y];
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_iq4_nl_r8 * iq4 = (const block_iq4_nl_r8 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            {
                __m256 d4[4];
                for (int k = 0; k < 4; ++k) {
                    d4[k] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[4*ib4+k].d));
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto scales = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d)), 16));
                    _mm256_storeu_ps(d8 + 8*iy, scales);
                    auto m4 = _mm256_extractf128_ps(scales, 1);
                    auto m8 = _mm256_set_m128(m4, m4);
                    auto sumf = _mm256_mul_ps(d4[0], _mm256_shuffle_ps(m8, m8, 0x00));
                    sumf = _mm256_fmadd_ps(d4[1], _mm256_shuffle_ps(m8, m8, 0x55), sumf);
                    sumf = _mm256_fmadd_ps(d4[2], _mm256_shuffle_ps(m8, m8, 0xaa), sumf);
                    sumf = _mm256_fmadd_ps(d4[3], _mm256_shuffle_ps(m8, m8, 0xff), sumf);
                    acc[iy] = _mm256_fmadd_ps(sumf, _mm256_set1_ps(-8.f), acc[iy]);
                }
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[4*ib4+k].d));
                prepare_q4_0_quants_avx2(iq4[4*ib4+k].qs, v, m4);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = accum_q4_0_quants(v, q8.y[iy][ib4].qs+32*k);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[8*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[ib].d));
            auto scales_m = _mm256_mul_ps(scales, _mm256_set1_ps(-8.f));
            prepare_q4_0_quants_avx2(iq4[ib].qs, v, m4);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = accum_q4_0_quants(v, qy[ib].qs);
                ggml_bf16_t d{qy[ib].d}, s{qy[ib].s};
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(d)));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                acc[iy] = _mm256_fmadd_ps(scales_m, _mm256_set1_ps(GGML_BF16_TO_FP32(s)), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
    }
}

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_q4_0_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if constexpr (nrc_y == 1) {
        mul_mat_q4_0_r8_q8_2_avx2<1>(n, vx, bx, info, nrc_x);
        return;
    }
    GGML_ASSERT(nrc_x%16 == 0);
    Q8<nrc_y, block_q8_1_x4> q8(info);
    auto m4 = _mm512_set1_epi8(0xf);
    int nb = n / QK4_NL;
    __m512  acc[2*nrc_y] = {};
    __m512i qx[8];
    auto prepare = [&qx, &m4] (const block_iq4_nl_r8& iq4l, const block_iq4_nl_r8& iq4h) {
        auto scales1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4l.d));
        auto scales2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4h.d));
        auto scales = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
        for (int j = 0; j < 4; ++j) {
            auto bits = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq4l.qs+j)),
                    _mm256_loadu_si256((const __m256i *)iq4h.qs+j), 1);
            qx[j+0] = _mm512_and_si512(bits, m4);
            qx[j+4] = _mm512_and_si512(_mm512_srli_epi16(bits, 4), m4);
        }
        return scales;
    };
    auto dot = [&qx] (const int8_t * qy) {
        auto y4l = _mm_loadu_si128((const __m128i*)qy+0);
        auto y4h = _mm_loadu_si128((const __m128i*)qy+1);
        auto y8l = MM256_SET_M128I(y4l, y4l);
        auto y8h = MM256_SET_M128I(y4h, y4h);
        auto yl = _mm512_inserti32x8(_mm512_castsi256_si512(y8l), y8l, 1);
        auto yh = _mm512_inserti32x8(_mm512_castsi256_si512(y8h), y8h, 1);
        auto sumi = _mm512_setzero_si512();
        sumi = _mm512_dpbusd_epi32(sumi, qx[0], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0x00)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[1], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0x55)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[2], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0xaa)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[3], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0xff)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[4], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0x00)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[5], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0x55)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[6], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0xaa)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[7], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0xff)));
        return sumi;
    };
    float d8[8*nrc_y];
    for (int ix = 0; ix < nrc_x; ix += 16) {
        const block_iq4_nl_r8 * iq4l = (const block_iq4_nl_r8 *)((const char *)vx + (ix+0)*bx);
        const block_iq4_nl_r8 * iq4h = (const block_iq4_nl_r8 *)((const char *)vx + (ix+8)*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto aux = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d)), 16);
                _mm256_storeu_ps(d8+8*iy, _mm256_castsi256_ps(aux));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq4l[4*ib4+k], iq4h[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(q8.y[iy][ib4].qs+32*k);
                    auto dy = _mm512_set1_ps(d8[8*iy+k]);
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(d8[8*iy+k+4]), acc[2*iy+1]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq4l[ib], iq4h[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(qy[ib].qs);
                ggml_bf16_t d{qy[ib].d}, s{qy[ib].s};
                auto dy = _mm512_set1_ps(GGML_BF16_TO_FP32(d));
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(GGML_BF16_TO_FP32(s)), acc[2*iy+1]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm512_fmadd_ps(_mm512_set1_ps(-8.f), acc[2*iy+1], acc[2*iy+0]);
            acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_ps();
            info.store(ix, iy, sum);
        }
    }
}
#else
template <int nrc_y>
static void mul_mat_q4_0_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    mul_mat_q4_0_r8_q8_2_avx2<nrc_y>(n, vx, bx, info, nrc_x);
}
#endif

template <int nrc_y>
static void mul_mat_q5_0_r4_q8_2_avx2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto m5 = _mm256_set1_epi8(0x10);
#ifndef HAVE_FANCY_SIMD
    auto m1 = _mm256_set1_epi16(1);
#endif
    auto mscale = _mm256_set_m128(_mm_set1_ps(-8.f), _mm_set1_ps(1.f));
    int nb = n / QK5_0;
    __m256 acc[nrc_y] = {};
    __m256i qx[4];
    float d8[8*nrc_y];
    auto prepare = [&qx, &m4, &m5] (const block_q5_0_r4& iq5) {
        auto scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq5.d));
        auto scales = _mm256_set_m128(scales128, scales128);
        auto bits1 = _mm256_loadu_si256((const __m256i *)iq5.qs+0);
        auto bits2 = _mm256_loadu_si256((const __m256i *)iq5.qs+1);
        auto hbits = _mm_loadu_si128((const __m128i *)iq5.qh);
        auto hb = MM256_SET_M128I(_mm_srli_epi16(hbits, 1), hbits);
        qx[0] = _mm256_or_si256(_mm256_and_si256(bits1, m4), _mm256_and_si256(_mm256_slli_epi16(hb, 4), m5));
        qx[1] = _mm256_or_si256(_mm256_and_si256(bits2, m4), _mm256_and_si256(_mm256_slli_epi16(hb, 2), m5));
        qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4), _mm256_and_si256(hb, m5));
        qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4), _mm256_and_si256(_mm256_srli_epi16(hb, 2), m5));;
        return scales;
    };
#ifdef HAVE_FANCY_SIMD
    auto dot = [&qx] (__m256i y) {
        auto sumi = _mm256_setzero_si256();
        sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
        sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
        sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
        sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
        return sumi;
    };
#else
    auto dot = [&qx, &m1] (__m256i y) {
        auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                      _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
        auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                      _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
        auto sumi = _mm256_madd_epi16(m1, _mm256_add_epi16(sumi1, sumi2));
        return sumi;
    };
#endif
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q5_0_r4 * iq5 = (const block_q5_0_r4 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto scales = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d)), 16));
                _mm256_storeu_ps(d8 + 8*iy, _mm256_mul_ps(mscale, scales));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq5[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(_mm256_loadu_si256((const __m256i*)q8.y[iy][ib4].qs+k));
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[8*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(scales, _mm256_set1_ps(d8[8*iy+k+4]), acc[iy]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq5[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(_mm256_loadu_si256((const __m256i*)qy[ib].qs));
                ggml_bf16_t d{qy[ib].d}, s{qy[ib].s};
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(d)));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                acc[iy] = _mm256_fmadd_ps(scales, _mm256_set1_ps(-8.f*GGML_BF16_TO_FP32(s)), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, sum);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_q5_0_r4_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if constexpr (nrc_y == 1) {
        mul_mat_q5_0_r4_q8_2_avx2<1>(n, vx, bx, info, nrc_x);
    } else {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm512_set1_epi8(0xf);
    auto m5 = _mm512_set1_epi8(0x10);
    int nb = n / QK5_0;
    __m512  acc[2*nrc_y] = {};
    __m512i qx[4];
    float d8[8*nrc_y];
    auto prepare = [&qx, &m4, &m5] (const block_q5_0_r4& iq5l, const block_q5_0_r4& iq5h) {
        auto scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq5l.d));
        auto scales1 = _mm256_set_m128(scales128, scales128);
        scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq5h.d));
        auto scales2 = _mm256_set_m128(scales128, scales128);
        auto scales = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
        auto bits1 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq5l.qs+0)),
                _mm256_loadu_si256((const __m256i *)iq5h.qs+0), 1);
        auto bits2 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq5l.qs+1)),
                _mm256_loadu_si256((const __m256i *)iq5h.qs+1), 1);
        auto hbits1 = _mm_loadu_si128((const __m128i *)iq5l.qh);
        auto hbits2 = _mm_loadu_si128((const __m128i *)iq5h.qh);
        auto hb1 = MM256_SET_M128I(_mm_srli_epi16(hbits1, 1), hbits1);
        auto hb2 = MM256_SET_M128I(_mm_srli_epi16(hbits2, 1), hbits2);
        auto hb = _mm512_inserti32x8(_mm512_castsi256_si512(hb1), hb2, 1);
        qx[0] = _mm512_or_si512(_mm512_and_si512(bits1, m4), _mm512_and_si512(_mm512_slli_epi16(hb, 4), m5));
        qx[1] = _mm512_or_si512(_mm512_and_si512(bits2, m4), _mm512_and_si512(_mm512_slli_epi16(hb, 2), m5));
        qx[2] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(bits1, 4), m4), _mm512_and_si512(hb, m5));
        qx[3] = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(bits2, 4), m4), _mm512_and_si512(_mm512_srli_epi16(hb, 2), m5));
        return scales;
    };
    auto dot = [&qx] (__m256i y8) {
        auto y = _mm512_inserti32x8(_mm512_castsi256_si512(y8), y8, 1);
        auto sumi = _mm512_setzero_si512();
        sumi = _mm512_dpbusd_epi32(sumi, qx[0], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[1], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[2], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[3], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
        return sumi;
    };
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q5_0_r4 * iq5l = (const block_q5_0_r4 *)((const char *)vx + (ix+0)*bx);
        const block_q5_0_r4 * iq5h = (const block_q5_0_r4 *)((const char *)vx + (ix+4)*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                _mm256_storeu_ps(d8+8*iy, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d)), 16)));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq5l[4*ib4+k], iq5h[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(_mm256_loadu_si256((const __m256i*)q8.y[iy][ib4].qs+k));
                    auto dy = _mm512_set1_ps(d8[8*iy+k]);
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(d8[8*iy+k+4]), acc[2*iy+1]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq5l[ib], iq5h[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(_mm256_loadu_si256((const __m256i*)qy[ib].qs));
                ggml_bf16_t d{qy[ib].d}, s{qy[ib].s};
                auto dy = _mm512_set1_ps(GGML_BF16_TO_FP32(d));
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(GGML_BF16_TO_FP32(s)), acc[2*iy+1]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum512 = _mm512_fmadd_ps(_mm512_set1_ps(-8.f), acc[2*iy+1], acc[2*iy+0]);
            acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_ps();
            auto sum1 = _mm_add_ps(_mm512_extractf32x4_ps(sum512, 0), _mm512_extractf32x4_ps(sum512, 1));
            auto sum2 = _mm_add_ps(_mm512_extractf32x4_ps(sum512, 2), _mm512_extractf32x4_ps(sum512, 3));
            info.store(ix+0, iy, sum1);
            info.store(ix+4, iy, sum2);
        }
    }
    }
}
#else
template <int nrc_y>
static void mul_mat_q5_0_r4_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    mul_mat_q5_0_r4_q8_2_avx2<nrc_y>(n, vx, bx, info, nrc_x);
}
#endif

template <int nrc_y>
static void mul_mat_q6_0_r4_q8_2_avx2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto m6 = _mm256_set1_epi8(0x30);
    auto mscale = _mm256_set_m128(_mm_set1_ps(-16.f), _mm_set1_ps(1.f));
#ifndef HAVE_FANCY_SIMD
    auto m1 = _mm256_set1_epi16(1);
#endif
    int nb = n / QK6_0;
    __m256 acc[nrc_y] = {};
    float d8[8*nrc_y];
    __m256i qx[4];
    auto prepare = [&qx, &m4, &m6] (const block_q6_0_r4& iq6) {
        auto scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq6.d));
        auto scales = _mm256_set_m128(scales128, scales128);
        auto bits1 = _mm256_loadu_si256((const __m256i *)iq6.qs+0);
        auto bits2 = _mm256_loadu_si256((const __m256i *)iq6.qs+1);
        auto hbits = _mm256_loadu_si256((const __m256i *)iq6.qh);
        qx[0] = _mm256_or_si256(_mm256_and_si256(bits1, m4), _mm256_and_si256(_mm256_slli_epi16(hbits, 4), m6));
        qx[1] = _mm256_or_si256(_mm256_and_si256(bits2, m4), _mm256_and_si256(_mm256_slli_epi16(hbits, 2), m6));
        qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4), _mm256_and_si256(hbits, m6));
        qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4), _mm256_and_si256(_mm256_srli_epi16(hbits, 2), m6));
        return scales;
    };
#ifdef HAVE_FANCY_SIMD
    auto dot = [&qx] (__m256i y) {
        auto sumi = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[0], _mm256_shuffle_epi32(y, 0x00));
        sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
        sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
        sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
        return sumi;
    };
#else
    auto dot = [&qx, &m1] (__m256i y) {
        auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                      _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
        auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                      _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
        auto sumi = _mm256_add_epi32(_mm256_madd_epi16(m1, sumi1), _mm256_madd_epi16(m1, sumi2));
        return sumi;
    };
#endif
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q6_0_r4 * iq6 = (const block_q6_0_r4 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto scales = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d)), 16));
                _mm256_storeu_ps(d8 + 8*iy, _mm256_mul_ps(scales, mscale));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq6[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(_mm256_loadu_si256((const __m256i*)q8.y[iy][ib4].qs+k));
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[8*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(scales, _mm256_set1_ps(d8[8*iy+k+4]), acc[iy]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq6[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(_mm256_loadu_si256((const __m256i*)qy[ib].qs));
                ggml_bf16_t d{qy[ib].d}, s{qy[ib].s};
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(d)));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                acc[iy] = _mm256_fmadd_ps(scales, _mm256_set1_ps(-16.f*GGML_BF16_TO_FP32(s)), acc[iy]);
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, sum);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_q6_0_r4_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if constexpr (nrc_y == 1) {
        mul_mat_q6_0_r4_q8_2_avx2<1>(n, vx, bx, info, nrc_x);
    } else {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m4 = _mm512_set1_epi8(0xf);
    auto m6 = _mm512_set1_epi8(0x30);
    int nb = n / QK6_0;
    __m512  acc[2*nrc_y] = {};
    __m512i qx[4];
    float d8[8*nrc_y];
    auto prepare = [&qx, &m4, &m6] (const block_q6_0_r4& iq6l, const block_q6_0_r4& iq6h) {
        auto scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq6l.d));
        auto scales1 = _mm256_set_m128(scales128, scales128);
        scales128 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq6h.d));
        auto scales2 = _mm256_set_m128(scales128, scales128);
        auto scales = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
        auto bits1 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq6l.qs+0)),
                                                               _mm256_loadu_si256((const __m256i *)iq6h.qs+0), 1);
        auto bits2 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq6l.qs+1)),
                                                               _mm256_loadu_si256((const __m256i *)iq6h.qs+1), 1);
        auto hbits1 = _mm256_loadu_si256((const __m256i *)iq6l.qh);
        auto hbits2 = _mm256_loadu_si256((const __m256i *)iq6h.qh);
        auto hb = _mm512_inserti32x8(_mm512_castsi256_si512(hbits1), hbits2, 1);
        qx[0] = _mm512_and_si512(bits1, m4) | _mm512_and_si512(_mm512_slli_epi16(hb, 4), m6);
        qx[1] = _mm512_and_si512(bits2, m4) | _mm512_and_si512(_mm512_slli_epi16(hb, 2), m6);;
        qx[2] = _mm512_and_si512(_mm512_srli_epi16(bits1, 4), m4) | _mm512_and_si512(hb, m6);
        qx[3] = _mm512_and_si512(_mm512_srli_epi16(bits2, 4), m4) | _mm512_and_si512(_mm512_srli_epi16(hb, 2), m6);
        return scales;
    };
    auto dot = [&qx] (__m256i y8) {
        auto y = _mm512_inserti32x8(_mm512_castsi256_si512(y8), y8, 1);
        auto sumi = _mm512_setzero_si512();
        sumi = _mm512_dpbusd_epi32(sumi, qx[0], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[1], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[2], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
        sumi = _mm512_dpbusd_epi32(sumi, qx[3], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
        return sumi;
    };
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q6_0_r4 * iq6l = (const block_q6_0_r4 *)((const char *)vx + (ix+0)*bx);
        const block_q6_0_r4 * iq6h = (const block_q6_0_r4 *)((const char *)vx + (ix+4)*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto scales = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d)), 16));
                _mm256_storeu_ps(d8 + 8*iy, scales);
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = prepare(iq6l[4*ib4+k], iq6h[4*ib4+k]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(_mm256_loadu_si256((const __m256i*)q8.y[iy][ib4].qs+k));
                    auto dy = _mm512_set1_ps(d8[8*iy+k]);
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(d8[8*iy+k+4]), acc[2*iy+1]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = prepare(iq6l[ib], iq6h[ib]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(_mm256_loadu_si256((const __m256i*)qy[ib].qs));
                ggml_bf16_t d{qy[ib].d}, s{qy[ib].s};
                auto dy = _mm512_set1_ps(GGML_BF16_TO_FP32(d));
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(GGML_BF16_TO_FP32(s)), acc[2*iy+1]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum512 = _mm512_fmadd_ps(_mm512_set1_ps(-16.f), acc[2*iy+1], acc[2*iy+0]);
            acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_ps();
            auto sum1 = _mm_add_ps(_mm512_extractf32x4_ps(sum512, 0), _mm512_extractf32x4_ps(sum512, 1));
            auto sum2 = _mm_add_ps(_mm512_extractf32x4_ps(sum512, 2), _mm512_extractf32x4_ps(sum512, 3));
            info.store(ix+0, iy, sum1);
            info.store(ix+4, iy, sum2);
        }
    }
    }
}
#else
template <int nrc_y>
static void mul_mat_q6_0_r4_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    mul_mat_q6_0_r4_q8_2_avx2<nrc_y>(n, vx, bx, info, nrc_x);
}
#endif

#ifdef HAVE_FANCY_SIMD
inline __m512i qx_r8_q8_dot_product(const __m512i * qx, const int8_t * y) {
    auto y4l = _mm_loadu_si128((const __m128i*)y+0);
    auto y4h = _mm_loadu_si128((const __m128i*)y+1);
    auto y8l = MM256_SET_M128I(y4l, y4l);
    auto y8h = MM256_SET_M128I(y4h, y4h);
    auto yl  = _mm512_inserti32x8(_mm512_castsi256_si512(y8l), y8l, 1);
    auto yh  = _mm512_inserti32x8(_mm512_castsi256_si512(y8h), y8h, 1);
    auto sumi = _mm512_setzero_si512();
    sumi = _mm512_dpbusd_epi32(sumi, qx[0], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0x00)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[1], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0x55)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[2], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0xaa)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[3], _mm512_shuffle_epi32(yl, _MM_PERM_ENUM(0xff)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[4], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0x00)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[5], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0x55)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[6], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0xaa)));
    sumi = _mm512_dpbusd_epi32(sumi, qx[7], _mm512_shuffle_epi32(yh, _MM_PERM_ENUM(0xff)));
    return sumi;
}
inline __m256i qx_r8_q8_dot_product(const __m256i * qx, const int8_t * y) {
    auto y4l = _mm_loadu_si128((const __m128i*)y+0);
    auto y4h = _mm_loadu_si128((const __m128i*)y+1);
    auto yl  = MM256_SET_M128I(y4l, y4l);
    auto yh  = MM256_SET_M128I(y4h, y4h);
    auto sumi = _mm256_setzero_si256();
    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(yl, 0x00));
    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(yl, 0x55));
    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(yl, 0xaa));
    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(yl, 0xff));
    sumi = _mm256_dpbusd_epi32(sumi, qx[4], _mm256_shuffle_epi32(yh, 0x00));
    sumi = _mm256_dpbusd_epi32(sumi, qx[5], _mm256_shuffle_epi32(yh, 0x55));
    sumi = _mm256_dpbusd_epi32(sumi, qx[6], _mm256_shuffle_epi32(yh, 0xaa));
    sumi = _mm256_dpbusd_epi32(sumi, qx[7], _mm256_shuffle_epi32(yh, 0xff));
    return sumi;
}
inline __m256i q8_0_r8_dot_product(const uint8_t * x, const int8_t * y, __m256i * qx) {
    for (int i = 0; i < 8; ++i) {
        qx[i] = _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)x+i), _mm256_set1_epi8(127));
    }
    return qx_r8_q8_dot_product(qx, y);
}
template <int nrc_y>
static void mul_mat_q8_0_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%16 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    int nb = n / QK8_0;
    if constexpr (nrc_y == 1) {
        __m256 acc[2] = {};
        __m256i qx[8];
        float d8[8];
        for (int ix = 0; ix < nrc_x; ix += 8) {
            const block_q8_0_r8 * iq8 = (const block_q8_0_r8 *)((const char *)vx + ix*bx);
            for (int ib4 = 0; ib4 < nb/4; ++ib4) {
                auto aux = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)q8.y[0][ib4].d)), 16);
                _mm256_storeu_ps(d8, _mm256_castsi256_ps(aux));
                for (int k = 0; k < 4; ++k) {
                    auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*ib4+k].d));
                    auto sumi = q8_0_r8_dot_product((const uint8_t *)iq8[4*ib4+k].qs, q8.y[0][ib4].qs+32*k, qx);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[k]));
                    acc[0] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[0]);
                    acc[1] = _mm256_fmadd_ps(scales, _mm256_set1_ps(d8[k+4]), acc[1]);
                }
            }
            if (4*(nb/4) < nb) {
                auto qy = (const block_q8_1 *)q8.y[0];
                for (int ib = 4*(nb/4); ib < nb; ++ib) {
                    auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[ib].d));
                    auto sumi = q8_0_r8_dot_product((const uint8_t *)iq8[ib].qs, qy[ib].qs, qx);
                    ggml_bf16_t d, s; d.bits = qy[ib].d; s.bits = qy[ib].s;
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(d)));
                    acc[0] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[0]);
                    acc[1] = _mm256_fmadd_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(s)), acc[1]);
                }
            }
            info.store(ix, 0, _mm256_fmadd_ps(_mm256_set1_ps(-127.f), acc[1], acc[0]));
            acc[0] = acc[1] = _mm256_setzero_ps();
        }
    } else {
        __m512  acc[2*nrc_y] = {};
        __m512i qx[8];
        float d8[8*nrc_y];
        for (int ix = 0; ix < nrc_x; ix += 16) {
            const block_q8_0_r8 * q8l = (const block_q8_0_r8 *)((const char *)vx + (ix+0)*bx);
            const block_q8_0_r8 * q8h = (const block_q8_0_r8 *)((const char *)vx + (ix+8)*bx);
            for (int ib4 = 0; ib4 < nb/4; ++ib4) {
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto aux = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d)), 16);
                    _mm256_storeu_ps(d8+8*iy, _mm256_castsi256_ps(aux));
                }
                for (int k = 0; k < 4; ++k) {
                    auto scales1  = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8l[4*ib4+k].d));
                    auto scales2  = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8h[4*ib4+k].d));
                    auto scales   = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
                    for (int j = 0; j < 8; ++j) {
                        qx[j] = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)q8l[4*ib4+k].qs+j)),
                                                                          _mm256_loadu_si256((const __m256i *)q8h[4*ib4+k].qs+j), 1);
                        qx[j] = _mm512_add_epi8(qx[j], _mm512_set1_epi8(127));
                    }
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto sumi = qx_r8_q8_dot_product(qx, q8.y[iy][ib4].qs+32*k);
                        auto dy = _mm512_set1_ps(d8[8*iy+k]);
                        acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                        acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(d8[8*iy+k+4]), acc[2*iy+1]);
                    }
                }
            }
            for (int ib = 4*(nb/4); ib < nb; ++ib) {
                auto scales1  = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8l[ib].d));
                auto scales2  = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8h[ib].d));
                auto scales   = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
                for (int j = 0; j < 8; ++j) {
                    qx[j] = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)q8l[ib].qs+j)),
                                                                      _mm256_loadu_si256((const __m256i *)q8h[ib].qs+j), 1);
                    qx[j] = _mm512_add_epi8(qx[j], _mm512_set1_epi8(127));
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto qy = (const block_q8_1 *)q8.y[iy];
                    auto sumi = qx_r8_q8_dot_product(qx, qy[ib].qs);
                    ggml_bf16_t d, s; d.bits = qy[ib].d; s.bits = qy[ib].s;
                    auto dy = _mm512_set1_ps(GGML_BF16_TO_FP32(d));
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(GGML_BF16_TO_FP32(s)), acc[2*iy+1]);
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto sum512 = _mm512_fmadd_ps(_mm512_set1_ps(-127.f), acc[2*iy+1], acc[2*iy+0]);
                info.store(ix, iy, sum512);
                acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_ps();
            }
        }
    }
}
#else
template <int nrc_y>
static void mul_mat_q8_0_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    auto m1 = _mm256_set1_epi16(1);
    int nb = n / QK8_0;
    __m256 acc[nrc_y] = {};
    float d8[4*nrc_y];
    __m256i qx[4], sx[4];
    auto dot = [&qx, &sx, &m1] (const int8_t * qy) {
        auto y128 = _mm_loadu_si128((const __m128i*)qy);
        auto y = MM256_SET_M128I(y128, y128);
        auto sumi1 = _mm256_add_epi32(
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[0], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]))),
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[1], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1])))
        );
        auto sumi2 = _mm256_add_epi32(
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[2], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]))),
                _mm256_madd_epi16(m1, _mm256_maddubs_epi16(sx[3], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3])))
        );
        return _mm256_add_epi32(sumi1, sumi2);
    };
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q8_0_r8 * iq8 = (const block_q8_0_r8 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto scales = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8.y[iy][ib4].d)), 16));
                _mm_storeu_ps(d8 + 4*iy, scales);
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*ib4+k].d));
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*ib4+k].qs+j);
                    sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(q8.y[iy][ib4].qs+32*k);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*ib4+k].qs+4+j);
                    sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(q8.y[iy][ib4].qs+32*k+16);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+k]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[ib].d));
            for (int j = 0; j < 4; ++j) {
                qx[j] = _mm256_loadu_si256((const __m256i *)iq8[ib].qs+j);
                sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_2 *)q8.y[iy];
                auto sumi = dot(qy[ib].qs);
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(ggml_bf16_t{qy[ib].d})));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
            }
            for (int j = 0; j < 4; ++j) {
                qx[j] = _mm256_loadu_si256((const __m256i *)iq8[ib].qs+4+j);
                sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_2 *)q8.y[iy];
                auto sumi = dot(qy[ib].qs+16);
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_BF16_TO_FP32(ggml_bf16_t{qy[ib].d})));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}
#endif

template <typename Dequantizer> void set_functions(std::array<mul_mat_t, IQK_MAX_NY>& funcs) {
    if constexpr (std::is_same_v<Dequantizer, Q4_0_Unpacker> || std::is_same_v<Dequantizer, Q5_0_Unpacker> ||
            std::is_same_v<Dequantizer, Q8_0_Unpacker>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0_T, Dequantizer, funcs)
    }
    else if constexpr (std::is_same_v<Dequantizer, Q4_1_Unpacker> || std::is_same_v<Dequantizer, Q5_1_Unpacker>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_2_T, Dequantizer, funcs)
    }
    else if constexpr (std::is_same_v<Dequantizer, IQ4_NL_Unpacker>) {
#ifdef HAVE_FANCY_SIMD
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_2_T, Dequantizer, funcs)
#else
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0_T, Dequantizer, funcs)
#endif
    }
    else if constexpr (std::is_same_v<Dequantizer, Q8_0_1_Unpacker> || std::is_same_v<Dequantizer, Q4_0_1_Unpacker> ||
                       std::is_same_v<Dequantizer, Q5_0_1_Unpacker> || std::is_same_v<Dequantizer, Q6_0_1_Unpacker>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_2_T, Dequantizer, funcs)
    }
}

} // namespace

bool iqk_set_kernels_legacy_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {

    if (ne00%QK8_0 != 0) return false;

    auto expected_typeB = GGML_TYPE_Q8_2_X4;

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_Q4_0:
            set_functions<Q4_0_1_Unpacker>(kernels);
            break;
        case GGML_TYPE_Q4_1:
            set_functions<Q4_1_Unpacker>(kernels);
            break;
        case GGML_TYPE_Q5_0:
            set_functions<Q5_0_1_Unpacker>(kernels);
            break;
        case GGML_TYPE_Q5_1:
            set_functions<Q5_1_Unpacker>(kernels);
            break;
        case GGML_TYPE_Q6_0:
            set_functions<Q6_0_1_Unpacker>(kernels);
            break;
        case GGML_TYPE_Q8_0:
#ifdef HAVE_FANCY_SIMD
            set_functions<Q8_0_1_Unpacker>(kernels);
#else
            set_functions<Q8_0_Unpacker>(kernels);
            expected_typeB = GGML_TYPE_Q8_0_X4;
#endif
            break;
        case GGML_TYPE_IQ4_NL:
            set_functions<IQ4_NL_Unpacker>(kernels);
#ifndef HAVE_FANCY_SIMD
            expected_typeB = GGML_TYPE_Q8_0_X4;
#endif
            break;
        case GGML_TYPE_Q4_0_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q4_0_r8_q8_2, kernels)
#ifdef HAVE_FANCY_SIMD
            func16 = mul_mat_q4_0_r8_q8_2<16>;
#endif
            break;
        case GGML_TYPE_Q5_0_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q5_0_r4_q8_2, kernels)
            break;
        case GGML_TYPE_Q6_0_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q6_0_r4_q8_2, kernels)
            break;
        case GGML_TYPE_Q8_0_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_0_r8_q8_2, kernels)
            break;
        case GGML_TYPE_IQ4_NL_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_nl_r4_q8_2, kernels)
            break;
        default:
            return false;
    }

    return ggml_type(typeB) == expected_typeB;
}

#else
// ---------------------------- __aarch64__ ----------------------------------------------

#endif

#endif
