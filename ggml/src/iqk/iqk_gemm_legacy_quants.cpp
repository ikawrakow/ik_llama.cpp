#include "iqk_gemm_legacy_quants.h"

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

//
// ============================== Legacy quants
//

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

template <typename Dequantizer> void set_functions(std::array<mul_mat_t, IQK_MAX_NY>& funcs) {
    if constexpr (std::is_same_v<Dequantizer, Q4_0_Unpacker> || std::is_same_v<Dequantizer, Q5_0_Unpacker> ||
            std::is_same_v<Dequantizer, Q8_0_Unpacker>) {
        funcs[0] = mul_mat_qX_0_q8_0_T<Dequantizer, 1>;
        funcs[1] = mul_mat_qX_0_q8_0_T<Dequantizer, 2>;
        funcs[2] = mul_mat_qX_0_q8_0_T<Dequantizer, 3>;
        funcs[3] = mul_mat_qX_0_q8_0_T<Dequantizer, 4>;
        funcs[4] = mul_mat_qX_0_q8_0_T<Dequantizer, 5>;
        funcs[5] = mul_mat_qX_0_q8_0_T<Dequantizer, 6>;
        funcs[6] = mul_mat_qX_0_q8_0_T<Dequantizer, 7>;
        funcs[7] = mul_mat_qX_0_q8_0_T<Dequantizer, 8>;
    }
    else if constexpr (std::is_same_v<Dequantizer, Q4_1_Unpacker> || std::is_same_v<Dequantizer, Q5_1_Unpacker>) {
        funcs[0] = mul_mat_qX_1_q8_2_T<Dequantizer, 1>;
        funcs[1] = mul_mat_qX_1_q8_2_T<Dequantizer, 2>;
        funcs[2] = mul_mat_qX_1_q8_2_T<Dequantizer, 3>;
        funcs[3] = mul_mat_qX_1_q8_2_T<Dequantizer, 4>;
        funcs[4] = mul_mat_qX_1_q8_2_T<Dequantizer, 5>;
        funcs[5] = mul_mat_qX_1_q8_2_T<Dequantizer, 6>;
        funcs[6] = mul_mat_qX_1_q8_2_T<Dequantizer, 7>;
        funcs[7] = mul_mat_qX_1_q8_2_T<Dequantizer, 8>;
    }
    else if constexpr (std::is_same_v<Dequantizer, IQ4_NL_Unpacker>) {
#ifdef HAVE_FANCY_SIMD
        funcs[0] = mul_mat_qX_1_q8_2_T<Dequantizer, 1>;
        funcs[1] = mul_mat_qX_1_q8_2_T<Dequantizer, 2>;
        funcs[2] = mul_mat_qX_1_q8_2_T<Dequantizer, 3>;
        funcs[3] = mul_mat_qX_1_q8_2_T<Dequantizer, 4>;
        funcs[4] = mul_mat_qX_1_q8_2_T<Dequantizer, 5>;
        funcs[5] = mul_mat_qX_1_q8_2_T<Dequantizer, 6>;
        funcs[6] = mul_mat_qX_1_q8_2_T<Dequantizer, 7>;
        funcs[7] = mul_mat_qX_1_q8_2_T<Dequantizer, 8>;
#else
        funcs[0] = mul_mat_qX_0_q8_0_T<Dequantizer, 1>;
        funcs[1] = mul_mat_qX_0_q8_0_T<Dequantizer, 2>;
        funcs[2] = mul_mat_qX_0_q8_0_T<Dequantizer, 3>;
        funcs[3] = mul_mat_qX_0_q8_0_T<Dequantizer, 4>;
        funcs[4] = mul_mat_qX_0_q8_0_T<Dequantizer, 5>;
        funcs[5] = mul_mat_qX_0_q8_0_T<Dequantizer, 6>;
        funcs[6] = mul_mat_qX_0_q8_0_T<Dequantizer, 7>;
        funcs[7] = mul_mat_qX_0_q8_0_T<Dequantizer, 8>;
#endif
    }
    else if constexpr (std::is_same_v<Dequantizer, Q8_0_1_Unpacker> || std::is_same_v<Dequantizer, Q4_0_1_Unpacker> ||
                       std::is_same_v<Dequantizer, Q5_0_1_Unpacker> || std::is_same_v<Dequantizer, Q6_0_1_Unpacker>) {
        funcs[0] = mul_mat_qX_1_q8_2_T<Dequantizer, 1>;
        funcs[1] = mul_mat_qX_1_q8_2_T<Dequantizer, 2>;
        funcs[2] = mul_mat_qX_1_q8_2_T<Dequantizer, 3>;
        funcs[3] = mul_mat_qX_1_q8_2_T<Dequantizer, 4>;
        funcs[4] = mul_mat_qX_1_q8_2_T<Dequantizer, 5>;
        funcs[5] = mul_mat_qX_1_q8_2_T<Dequantizer, 6>;
        funcs[6] = mul_mat_qX_1_q8_2_T<Dequantizer, 7>;
        funcs[7] = mul_mat_qX_1_q8_2_T<Dequantizer, 8>;
    }
}

} // namespace

bool iqk_set_kernels_legacy_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels) {

    if (ne00%QK8_0 != 0) return false;

    auto expected_typeB = GGML_TYPE_Q8_2_X4;

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
        default:
            return false;
    }

    return ggml_type(typeB) == expected_typeB;
}

#endif
