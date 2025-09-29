#include "iqk_gemm_legacy_quants.h"
#include <type_traits>

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

inline __m256 convert_scales(const uint16_t * scales) {
    auto aux_d = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)scales)), 16));
    auto aux_m = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(scales+4))));
    return _mm256_set_m128(_mm_mul_ps(aux_d, aux_m), aux_d);
}

inline __m128 convert_scales_s(const uint16_t * scales) {
    return _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)scales)), 16));
}

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

struct ScaleHelperQ8_2S {
    template <typename Q>
    inline __m128 prepare4(const Q * y) {
        const block_q8_2_x4 * y4 = (const block_q8_2_x4 *)y;
        return convert_scales_s((const uint16_t *)y4->d);
    }
    template <typename Q>
    inline __m128 prepare4(__m128 other_scales, const Q * y) {
        return _mm_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> static inline float prepare1(const Q * y) { return GGML_BF16_TO_FP32(ggml_bf16_t{y->d}); }
    template <typename Q> static inline float prepare1(float d, const Q * y) { return d*prepare1(y); }
};

struct ScaleHelperQ_0_MXFP4 {
    float scales[4];
    template <typename Q>
    inline __m128 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) scales[j] = GGML_E8M0_TO_FP32_HALF(y[j].e);
        return _mm_loadu_ps(scales);
    }
    template <typename Q>
    inline __m128 prepare4(__m128 other_scales, const Q * y) {
        return _mm_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline float prepare1(const Q * y) const { return GGML_E8M0_TO_FP32_HALF(y->e); }
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

template <int min_value>
struct ScaleHelperQ_0_1_MXFP4 {
    float scales[4];
    template <typename Q>
    inline __m256 prepare4(const Q * y) {
        for (int j = 0; j < 4; ++j) scales[j] = GGML_E8M0_TO_FP32_HALF(y[j].e);
        auto s4 = _mm_loadu_ps(scales);
        return _mm256_set_m128(_mm_mul_ps(s4, min), s4);
    }
    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm_mul256_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> inline std::pair<float, float> prepare1(const Q * y) const {
        float d = GGML_E8M0_TO_FP32_HALF(y->e);
        return std::make_pair(d, -d*float(min_value));
    }
    std::pair<float, float> inline prepare1(const std::pair<float, float>& dm, const block_q8_1 * y) const {
        return std::make_pair(dm.first*GGML_FP16_TO_FP32(y->d), dm.second*GGML_FP16_TO_FP32(y->s));
    }
    const __m128 min = _mm_set1_ps(float(-min_value));
};

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
        return convert_scales((const uint16_t *)y4->d);
    }
    template <typename Q>
    inline __m256 prepare4(__m256 other_scales, const Q * y) {
        return _mm256_mul_ps(other_scales, prepare4<Q>(y));
    }
    template <typename Q> static inline std::pair<float, float> prepare1(const Q * y) {
        float   d = GGML_BF16_TO_FP32(ggml_bf16_t{y->d});
        int16_t m = *(const int16_t *)&y->s;
        return std::make_pair(d, d*m);
    }
    static inline std::pair<float, float> prepare1(const std::pair<float, float>& dm, const block_q8_2 * y) {
        auto d = prepare1(y);
        return std::make_pair(dm.first*d.first, dm.second*d.second);
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
using Sum4TypeQ82S = Sum4<block_q8_2, block_q8_2_x4, SignedDot, false>;

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

template <typename Unpacker, int nrc_y, typename Block = block_q8_0>
void mul_mat_qX_0_q8_0_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, Block> q8(info);
    int nb = n/Unpacker::block_size();
    if constexpr (std::is_same_v<Block, block_q8_2>) {
        if (nb%4 == 0) {
            mul_mat_qX_q8_Helper<Unpacker, AccumType0<nrc_y, true>, ScaleHelperQ8_2S, Block, nrc_y>(
                    nb, vx, bx, info, q8.y, nrc_x);
        } else {
            mul_mat_qX_q8_Helper<Unpacker, AccumType0<nrc_y, false>, ScaleHelperQ8_2S, Block, nrc_y>(
                    nb, vx, bx, info, q8.y, nrc_x);
        }
    }
    else {
        if (nb%4 == 0) {
            mul_mat_qX_q8_Helper<Unpacker, AccumType0<nrc_y, true>, ScaleHelperQ8_0, Block, nrc_y>(
                    nb, vx, bx, info, q8.y, nrc_x);
        } else {
            mul_mat_qX_q8_Helper<Unpacker, AccumType0<nrc_y, false>, ScaleHelperQ8_0, Block, nrc_y>(
                    nb, vx, bx, info, q8.y, nrc_x);
        }
    }
}

template <typename Unpacker, int nrc_y>
void mul_mat_qX_0_q8_2_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%Unpacker::block_size() == 0);
    Q8<nrc_y, block_q8_2> q8(info);
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
    Q8<nrc_y, block_q8_2> q8(info);
    int nb = n/Unpacker::block_size();
    Unpacker unp(vx, bx);
    typename Unpacker::Sum4T sum4;
    ScaleHelperQ8_2S scales;
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

struct IQ4_NL_DequantizerU {
    Dequantizer4bit b4;
    const __m256i values = load_iq4nl_values_256();
    inline __m256i dequant(const block_iq4_nl * x) const {
        return _mm256_shuffle_epi8(values, b4.dequant(x->qs));
    }
};

struct IQ4_NL_DequantizerS {
    Dequantizer4bit b4;
    const __m256i values = load_iq4k_values_256();
    inline __m256i dequant(const block_iq4_nl * x) const {
        return _mm256_shuffle_epi8(values, b4.dequant(x->qs));
    }
};

//=============================
static inline __m128i load_unsigned_mxfp4_values_128() {
    static const uint8_t kvalues_mxfp4_unsigned[16] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};
    return _mm_loadu_si128((const __m128i *)kvalues_mxfp4_unsigned);
}

static inline __m256i load_unsigned_mxfp4_values_256() {
    auto val128 = load_unsigned_mxfp4_values_128();
    return MM256_SET_M128I(val128, val128);
}

#ifdef HAVE_FANCY_SIMD
static inline __m512i load_unsigned_mxfp4_values_512() {
    auto val256 = load_unsigned_mxfp4_values_256();
    return _mm512_inserti32x8(_mm512_castsi256_si512(val256), val256, 1);
}
#endif

static inline __m128i load_mxfp4_values_128() {
    return _mm_loadu_si128((const __m128i *)kvalues_mxfp4);
}

static inline __m256i load_mxfp4_values_256() {
    auto val128 = load_mxfp4_values_128();
    return MM256_SET_M128I(val128, val128);
}

struct MXFP4_Dequantizer {
    Dequantizer4bit b4;
    const __m256i values = load_unsigned_mxfp4_values_256();
    inline __m256i dequant(const block_mxfp4 * x) const {
        return _mm256_shuffle_epi8(values, b4.dequant(x->qs));
    }
};

struct MXFP40_Dequantizer {
    Dequantizer4bit b4;
    const __m256i values = load_mxfp4_values_256();
    inline __m256i dequant(const block_mxfp4 * x) const {
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
struct Q6_0_Dequantizer {
    Q6_0_1_Dequantizer deq;
    inline __m256i dequant(const block_q6_0 * x) const {
        return _mm256_add_epi8(deq.dequant(x), _mm256_set1_epi8(-32));
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
    using Sum4T = Sum4TypeQ82S;
    inline static int block_size() { return QK8_0; }
};
struct Q8_0_1_Unpacker final : public Q_Unpacker<block_q8_0, ScaleHelperQ_0_1<127>, Q8_0_1_Dequantizer> {
    Q8_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK8_0; }
};
struct Q8_0_2_Unpacker final : public Q_Unpacker<block_q8_0, ScaleHelperQ_0, Q8_0_Dequantizer> {
    Q8_0_2_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
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
struct MXFP4_Unpacker final : public Q_Unpacker<block_mxfp4, ScaleHelperQ_0_1_MXFP4<12>, MXFP4_Dequantizer> {
    MXFP4_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK4_NL; }
};
struct IQ4_NL_UnpackerU final : public Q_Unpacker<block_iq4_nl, ScaleHelperQ_0_1<128>, IQ4_NL_DequantizerU> {
    IQ4_NL_UnpackerU(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82;
    inline static int block_size() { return QK4_NL; }
};
struct IQ4_NL_UnpackerS final : public Q_Unpacker<block_iq4_nl, ScaleHelperQ_0, IQ4_NL_DequantizerS> {
    IQ4_NL_UnpackerS(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ82S;
    inline static int block_size() { return QK4_NL; }
};
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
                _mm256_storeu_ps(d8+8*iy, convert_scales((const uint16_t *)q8.y[iy][ib4].d));
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
                helper.vec = convert_scales((const uint16_t *)q8.y[0][ib4].d);
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
                    auto scales = convert_scales((const uint16_t *)q8.y[iy][ib4].d);
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
                _mm256_storeu_ps(d8+8*iy, convert_scales((const uint16_t *)q8.y[iy][ib4].d));
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
                auto scales = convert_scales((const uint16_t *)q8.y[iy][ib4].d);
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
                _mm256_storeu_ps(d8+8*iy, convert_scales((const uint16_t *)q8.y[iy][ib4].d));
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
                auto scales = convert_scales((const uint16_t *)q8.y[iy][ib4].d);
                _mm256_storeu_ps(d8 + 8*iy,  _mm256_mul_ps(scales, mscale));
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
                auto scales = convert_scales((const uint16_t *)q8.y[iy][ib4].d);
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
                _mm256_storeu_ps(d8, convert_scales((const uint16_t *)q8.y[0][ib4].d));
                for (int k = 0; k < 4; ++k) {
                    auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*ib4+k].d));
                    auto sumi = q8_0_r8_dot_product((const uint8_t *)iq8[4*ib4+k].qs, q8.y[0][ib4].qs+32*k, qx);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[k]));
                    acc[0] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[0]);
                    acc[1] = _mm256_fmadd_ps(scales, _mm256_set1_ps(d8[k+4]), acc[1]);
                }
            }
            if (4*(nb/4) < nb) {
                auto qy = (const block_q8_2 *)q8.y[0];
                for (int ib = 4*(nb/4); ib < nb; ++ib) {
                    auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[ib].d));
                    auto sumi = q8_0_r8_dot_product((const uint8_t *)iq8[ib].qs, qy[ib].qs, qx);
                    auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8));
                    acc[0] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[0]);
                    acc[1] = _mm256_fmadd_ps(scales, _mm256_set1_ps(m8), acc[1]);
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
                    _mm256_storeu_ps(d8+8*iy, convert_scales((const uint16_t *)q8.y[iy][ib4].d));
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
                    auto qy = (const block_q8_2 *)q8.y[iy];
                    auto sumi = qx_r8_q8_dot_product(qx, qy[ib].qs);
                    auto [d8, m8] = ScaleHelperQ8_2::prepare1(qy + ib);
                    auto dy = _mm512_set1_ps(d8);
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(m8), acc[2*iy+1]);
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

typedef struct {
    ggml_half d[16];
    uint8_t   qs[256];
} block_q8_1_r8;

template <int nrc_y>
static void mul_mat_q8_1_r8_q8_2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_2_x4> q8(info);
    int nb = n / QK8_0;
    __m256 acc[nrc_y] = {};
    float d8[4*nrc_y];
    __m256i qx[4];
    auto dot = [&qx] (const int8_t * qy) {
        auto y128 = _mm_loadu_si128((const __m128i*)qy);
        auto y = MM256_SET_M128I(y128, y128);
#ifdef HAVE_FANCY_SIMD
        auto sumi = _mm256_setzero_si256();
        sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
        sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
        sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
        sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
        return sumi;
#else
        auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                      _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
        auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                      _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
        return _mm256_add_epi32(_mm256_madd_epi16(_mm256_set1_epi16(1), sumi1), _mm256_madd_epi16(_mm256_set1_epi16(1), sumi2));
#endif
    };
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q8_1_r8 * iq8 = (const block_q8_1_r8 *)((const char *)vx + ix*bx);
        for (int i4 = 0; i4 < nb/4; ++i4) {
            {
                __m256 mx[4];
                for (int ib32 = 0; ib32 < 4; ++ib32) mx[ib32] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*i4+ib32].d+1));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto scales = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8.y[iy][i4].d)), 16));
                    _mm_storeu_ps(d8 + 4*iy + 0, scales);
                    auto bsums4 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][i4].d+4))));
                    bsums4 = _mm_mul_ps(bsums4, scales);
                    auto bsums  = _mm256_set_m128(bsums4, bsums4);
                    acc[iy] = _mm256_fmadd_ps(mx[0], _mm256_shuffle_ps(bsums, bsums, 0x00), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(mx[1], _mm256_shuffle_ps(bsums, bsums, 0x55), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(mx[2], _mm256_shuffle_ps(bsums, bsums, 0xaa), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(mx[3], _mm256_shuffle_ps(bsums, bsums, 0xff), acc[iy]);
                }
            }
            for (int ib32 = 0; ib32 < 4; ++ib32) {
                auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*i4+ib32].d));
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*i4+ib32].qs+j);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(q8.y[iy][i4].qs+32*ib32);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+ib32]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
                for (int j = 0; j < 4; ++j) {
                    qx[j] = _mm256_loadu_si256((const __m256i *)iq8[4*i4+ib32].qs+4+j);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = dot(q8.y[iy][i4].qs+32*ib32+16);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[4*iy+ib32]));
                    acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

void iqk_convert_q80_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    static_assert(QK4_0 == QK8_0);
    GGML_ASSERT(n%QK4_0 == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    const int nb = n/QK4_0;

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const block_q8_0 * x8[8];

    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {

        for (int k = 0; k < 8; ++k) x8[k] = (const block_q8_0 *)((const char *)vx + (ix + k)*bx);

        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                y[i].d[k] = x8[k][i].d;
                _mm256_storeu_si256((__m256i *)block, _mm256_loadu_si256((const __m256i *)x8[k][i].qs));
                auto qs = (uint32_t *)y[i].qs;
                for (int l = 0; l < 4; ++l) {
                    qs[8*l + k +  0] = block[l + 0];
                    qs[8*l + k + 32] = block[l + 4];
                }
            }
        }
        y += nb;
    }
}

template <typename Block, typename Dequantizer>
void iqk_convert_qX_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK4_0 == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    const int nb = n/QK8_0;

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const Block * x8[8];

    uint32_t block[8];

    Dequantizer deq;

    for (int ix = 0; ix < nrc_x; ix += 8) {

        for (int k = 0; k < 8; ++k) x8[k] = (const Block *)((const char *)vx + (ix + k)*bx);

        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                if constexpr (std::is_same_v<Dequantizer, MXFP40_Dequantizer>) {
                    y[i].d[k] = GGML_FP32_TO_FP16(GGML_E8M0_TO_FP32_HALF(x8[k][i].e));
                } else {
                    y[i].d[k] = x8[k][i].d;
                }
                _mm256_storeu_si256((__m256i *)block, deq.dequant(x8[k] + i));
                auto qs = (uint32_t *)y[i].qs;
                for (int l = 0; l < 4; ++l) {
                    qs[8*l + k +  0] = block[l + 0];
                    qs[8*l + k + 32] = block[l + 4];
                }
            }
        }
        y += nb;
    }
}

template <typename Block, typename Dequantizer>
void iqk_convert_qX_1_q8_1_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK8_0 == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK8_0;

    const Block * x8[8];

    block_q8_1_r8 * y = (block_q8_1_r8 *)vy;

    uint32_t block[8];

    Dequantizer deq;

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const Block *)((const char *)vx + (ix + k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                y[i].d[k+0] = x8[k][i].d;
                y[i].d[k+8] = x8[k][i].m;
                _mm256_storeu_si256((__m256i *)block, deq.dequant(x8[k]+i));
                auto qs = (uint32_t *)y[i].qs;
                for (int l = 0; l < 4; ++l) {
                    qs[8*l + k +  0] = block[l + 0];
                    qs[8*l + k + 32] = block[l + 4];
                }
            }
        }
        y += nb;
    }
}

template <typename Dequantizer> void set_functions(std::array<mul_mat_t, IQK_MAX_NY>& funcs) {
    if constexpr (std::is_same_v<Dequantizer, Q4_0_Unpacker> || std::is_same_v<Dequantizer, Q5_0_Unpacker>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0_T, Dequantizer, funcs)
    }
    else if constexpr (std::is_same_v<Dequantizer, Q8_0_Unpacker>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T2(mul_mat_qX_0_q8_0_T, Dequantizer, block_q8_2, funcs)
    }
    else if constexpr (std::is_same_v<Dequantizer, Q4_1_Unpacker> || std::is_same_v<Dequantizer, Q5_1_Unpacker>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_2_T, Dequantizer, funcs)
    }
    else if constexpr (std::is_same_v<Dequantizer, IQ4_NL_UnpackerU>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_2_T, Dequantizer, funcs)
    }
    else if constexpr (std::is_same_v<Dequantizer, IQ4_NL_UnpackerS>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T2(mul_mat_qX_0_q8_0_T, Dequantizer, block_q8_2, funcs)
    }
    else if constexpr (std::is_same_v<Dequantizer, Q8_0_1_Unpacker> || std::is_same_v<Dequantizer, Q4_0_1_Unpacker> ||
                       std::is_same_v<Dequantizer, Q5_0_1_Unpacker> || std::is_same_v<Dequantizer, Q6_0_1_Unpacker> ||
                       std::is_same_v<Dequantizer, MXFP4_Unpacker>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_2_T, Dequantizer, funcs)
    }
}

} // namespace

bool iqk_convert_legacy_quants_q8_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    switch (type) {
        case GGML_TYPE_Q4_0  : iqk_convert_qX_q80_r8<block_q4_0, Q4_0_Dequantizer>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q4_1  : iqk_convert_qX_1_q8_1_r8<block_q4_1, Q4_1_Dequantizer>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q5_0  : iqk_convert_qX_q80_r8<block_q5_0, Q5_0_Dequantizer>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q5_1  : iqk_convert_qX_1_q8_1_r8<block_q5_1, Q5_1_Dequantizer<block_q5_1>>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q6_0  : iqk_convert_qX_q80_r8<block_q6_0, Q6_0_Dequantizer>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_NL: iqk_convert_qX_q80_r8<block_iq4_nl, IQ4_NL_DequantizerS>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q8_0  : iqk_convert_q80_q80_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_MXFP4 : iqk_convert_qX_q80_r8<block_mxfp4, MXFP40_Dequantizer>(n, vx, bx, vy, nrc_x); break;
        default: return false;
    }
    return true;
}

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
#endif
            break;
        case GGML_TYPE_IQ4_NL:
#ifdef HAVE_FANCY_SIMD
            set_functions<IQ4_NL_UnpackerU>(kernels);
#else
            set_functions<IQ4_NL_UnpackerS>(kernels);
#endif
            break;
        case GGML_TYPE_MXFP4:
            set_functions<MXFP4_Unpacker>(kernels);
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
        case GGML_TYPE_Q8_1: // Note: we are misusing the Q8_1 type for Q8_1_R8
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_1_r8_q8_2, kernels)
            break;
        default:
            return false;
    }

    return ggml_type(typeB) == expected_typeB;
}

#else
// ---------------------------- __aarch64__ ----------------------------------------------

namespace {

template <typename Block>
inline float16x4_t load_scales_q0(const Block * x, ggml_half * aux) {
    for (int k = 0; k < 4; ++k) aux[k] = x[k].d;
    return vld1_f16((const float16_t *)aux);
}

template <typename Block>
inline float16x8_t load_scales_q1(const Block * x, ggml_half * aux) {
    if constexpr (std::is_same_v<Block, block_q8_1>) {
        for (int k = 0; k < 4; ++k) { aux[k] = x[k].d; aux[k+4] = x[k].s; }
    } else {
        for (int k = 0; k < 4; ++k) { aux[k] = x[k].d; aux[k+4] = x[k].m; }
    }
    return vld1q_f16((const float16_t *)aux);
}

struct Q4LegacyBits {
    template <typename Block>
    inline void prepare(const Block * x) {
        for (int i = 0; i < 4; ++i) {
            auto q4bits = vld1q_u8(x[i].qs);
            b[2*i+0] = vreinterpretq_s8_u8(vandq_u8(q4bits, m4b));
            b[2*i+1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits, 4));
        }
    }
    inline void prepare1(const uint8_t * qs, int8x16_t * q) const {
        auto q4bits = vld1q_u8(qs);
        q[0] = vreinterpretq_s8_u8(vandq_u8(q4bits, m4b));
        q[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits, 4));
    }
    inline void prepare1(const uint8_t * qs) {
        prepare1(qs, b);
    }
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    int8x16_t b[8];
};

// One would think this commented out version would do better than the one below
// because it offers more opportunities to execute instructions in parallel.
// Instead, it runs significantly slower. Why? If the compiler is running out of vector registers
// cannot it just do the sequential version below on its own?
//inline int32x4_t sum_4_blocks(const int8x16_t * b, const int8_t * qs) {
//    const auto q8b_1 = vld1q_s8_x2(qs + 0);
//    auto p12 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[0], q8b_1.val[0]), b[1], q8b_1.val[1]);
//    const auto q8b_2 = vld1q_s8_x2(qs + 32);
//    auto p34 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[2], q8b_2.val[0]), b[3], q8b_2.val[1]);
//    auto p1234 = vpaddq_s32(p12, p34);
//    const auto q8b_3 = vld1q_s8_x2(qs + 64);
//    auto p56 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[4], q8b_3.val[0]), b[5], q8b_3.val[1]);
//    const auto q8b_4 = vld1q_s8_x2(qs + 96);
//    auto p78 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[6], q8b_4.val[0]), b[7], q8b_4.val[1]);
//    return vpaddq_s32(p1234, vpaddq_s32(p56, p78));
//}

inline int32x4_t sum_4_blocks(const int8x16_t * b, const int8_t * qs) {
    auto q8b = vld1q_s8_x2(qs + 0);
    auto p12 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[0], q8b.val[0]), b[1], q8b.val[1]);
    q8b = vld1q_s8_x2(qs + 32);
    auto p34 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[2], q8b.val[0]), b[3], q8b.val[1]);
    auto p1234 = vpaddq_s32(p12, p34);
    q8b = vld1q_s8_x2(qs + 64);
    auto p56 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[4], q8b.val[0]), b[5], q8b.val[1]);
    q8b = vld1q_s8_x2(qs + 96);
    auto p78 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b[6], q8b.val[0]), b[7], q8b.val[1]);
    return vpaddq_s32(p1234, vpaddq_s32(p56, p78));
}

inline int32x4x2_t sum_4_blocks(const int8x16_t * b1, const int8x16_t * b2, const int8_t * qs) {
    auto q8b = vld1q_s8_x2(qs + 0);
    auto p12_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b1[0], q8b.val[0]), b1[1], q8b.val[1]);
    auto p12_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b2[0], q8b.val[0]), b2[1], q8b.val[1]);
    q8b = vld1q_s8_x2(qs + 32);
    auto p34_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b1[2], q8b.val[0]), b1[3], q8b.val[1]);
    auto p34_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b2[2], q8b.val[0]), b2[3], q8b.val[1]);
    auto p1234_1 = vpaddq_s32(p12_1, p34_1);
    auto p1234_2 = vpaddq_s32(p12_2, p34_2);
    q8b = vld1q_s8_x2(qs + 64);
    auto p56_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b1[4], q8b.val[0]), b1[5], q8b.val[1]);
    auto p56_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b2[4], q8b.val[0]), b2[5], q8b.val[1]);
    q8b = vld1q_s8_x2(qs + 96);
    auto p78_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b1[6], q8b.val[0]), b1[7], q8b.val[1]);
    auto p78_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), b2[6], q8b.val[0]), b2[7], q8b.val[1]);
    auto p5678_1 = vpaddq_s32(p56_1, p78_1);
    auto p5678_2 = vpaddq_s32(p56_2, p78_2);
    return { vpaddq_s32(p1234_1, p5678_1), vpaddq_s32(p1234_2, p5678_2)};
}

template <int nrc> struct Q80 {

    constexpr static int nrc_y = nrc;

    Q80(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8_0 *)info.src1_row(iy);
    }

    inline const int8_t * quant_data(int iy, int i) const {
        const block_q8_0_x4 * y4 = (const block_q8_0_x4 *)y[iy] + i;
        return y4->qs;
    }

    inline float16x4_t load_scales(int iy, int i) const {
        const block_q8_0_x4 * y4 = (const block_q8_0_x4 *)y[iy] + i;
        return vld1_f16((const float16_t *)y4->d);
    }

    template <typename Dequantizer>
    inline void process_scales(int i, Dequantizer& deq, float16x4_t * sc16, float32x4_t * /*acc*/) const {
        auto qx_scales = deq.new_block(i);
        for (int iy = 0; iy < nrc; ++iy) {
            auto q8_scales = load_scales(iy, i);
            sc16[iy] = vmul_f16(qx_scales, q8_scales);
        }
    }

    template <typename Dequantizer>
    inline void process_scales(int i, Dequantizer& deq1, Dequantizer& deq2, float16x4_t * sc16, float32x4_t * /*acc*/) const {
        auto qx_scales_1 = deq1.new_block(i);
        auto qx_scales_2 = deq2.new_block(i);
        for (int iy = 0; iy < nrc; ++iy) {
            auto q8_scales = load_scales(iy, i);
            sc16[iy      ] = vmul_f16(qx_scales_1, q8_scales);
            sc16[iy+nrc_y] = vmul_f16(qx_scales_2, q8_scales);
        }
    }

    template <typename Dequantizer>
    inline void process_1_block(int i, Dequantizer& deq, float32x4_t * acc) const {
        deq.prepare1(i);
        float d = deq.block_scale(i);
        for (int iy = 0; iy < nrc; ++iy) {
            auto q8b = vld1q_s8_x2(y[iy][i].qs);
            auto p = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), deq.bits.b[0], q8b.val[0]), deq.bits.b[1], q8b.val[1]);
            acc[iy] = vmlaq_f32(acc[iy], vdupq_n_f32(d*GGML_FP16_TO_FP32(y[iy][i].d)), vcvtq_f32_s32(p));
        }
    }

    const block_q8_0 * y[nrc_y];
};

template <int nrc> struct Q81 {

    constexpr static int nrc_y = nrc;

    Q81(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8_1 *)info.src1_row(iy);
    }

    inline const int8_t * quant_data(int iy, int i) const {
        const block_q8_1_x4 * y4 = (const block_q8_1_x4 *)y[iy] + i;
        return y4->qs;
    }

    inline float16x8_t load_scales(int iy, int i) const {
        const block_q8_1_x4 * y4 = (const block_q8_1_x4 *)y[iy] + i;
        return vld1q_f16((const float16_t *)y4->d);
    }

    template <typename Dequantizer>
    inline void process_scales(int i, Dequantizer& deq, float16x4_t * sc16, float32x4_t * acc) const {
        auto qx_scales = deq.new_block(i);
        for (int iy = 0; iy < nrc; ++iy) {
            auto q8_scales = load_scales(iy, i);
            auto m = vmul_f16(vget_high_f16(qx_scales), vget_high_f16(q8_scales));
            acc[iy] = vaddq_f32(acc[iy], vcvt_f32_f16(m));
            sc16[iy] = vmul_f16(vget_low_f16(qx_scales), vget_low_f16(q8_scales));
        }
    }

    template <typename Dequantizer>
    inline void process_scales(int i, Dequantizer& deq1, Dequantizer& deq2, float16x4_t * sc16, float32x4_t * acc) const {
        auto qx_scales_1 = deq1.new_block(i);
        auto qx_scales_2 = deq2.new_block(i);
        for (int iy = 0; iy < nrc; ++iy) {
            auto q8_scales = load_scales(iy, i);
            auto q8_scales_l = vget_low_f16(q8_scales);
            auto q8_scales_h = vget_high_f16(q8_scales);
            auto m1 = vmul_f16(vget_high_f16(qx_scales_1), q8_scales_h);
            auto m2 = vmul_f16(vget_high_f16(qx_scales_2), q8_scales_h);
            acc[iy       ] = vaddq_f32(acc[iy      ], vcvt_f32_f16(m1));
            acc[iy+nrc_y ] = vaddq_f32(acc[iy+nrc_y], vcvt_f32_f16(m2));
            sc16[iy      ] = vmul_f16(vget_low_f16(qx_scales_1), q8_scales_l);
            sc16[iy+nrc_y] = vmul_f16(vget_low_f16(qx_scales_2), q8_scales_l);
        }
    }

    template <typename Dequantizer>
    inline void process_1_block(int i, Dequantizer& deq, float32x4_t * acc) const {
        deq.prepare1(i);
        float d = GGML_FP16_TO_FP32(deq.x[i].d), m = 0.25f*GGML_FP16_TO_FP32(deq.x[i].m);
        for (int iy = 0; iy < nrc; ++iy) {
            auto q8b = vld1q_s8_x2(y[iy][i].qs);
            auto p = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), deq.bits.b[0], q8b.val[0]), deq.bits.b[1], q8b.val[1]);
            acc[iy] = vmlaq_f32(acc[iy], vdupq_n_f32(d*GGML_FP16_TO_FP32(y[iy][i].d)), vcvtq_f32_s32(p));
            acc[iy] = vaddq_f32(acc[iy], vdupq_n_f32(m*GGML_FP16_TO_FP32(y[iy][i].s)));
        }
    }

    const block_q8_1 * y[nrc_y];
};

template <typename block_q>
struct BaseLegacyDequantizer {

    BaseLegacyDequantizer(const void * vx, size_t bx) : vx(vx), x(nullptr), bx(bx) {}

    inline void new_row(int ix) { x = (const block_q *)((const char *)vx + bx*ix); }

    Q4LegacyBits bits;

    const void * vx;
    const block_q * x;
    size_t bx;
};

struct DequantizerQ40 final : public BaseLegacyDequantizer<block_q4_0> {

    DequantizerQ40(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i, int8x16_t * q) const {
        bits.prepare1(x[i].qs, q);
        q[0] = vaddq_s8(q[0], m8);
        q[1] = vaddq_s8(q[1], m8);
    }
    inline void prepare1(int i) {
        prepare1(i, bits.b);
    }

    inline float16x4_t new_block(int i) {
        ggml_half aux[4];
        for (int k = 0; k < 4; ++k) {
            aux[k] = x[4*i+k].d;
            prepare1(4*i+k, bits.b + 2*k);
        }
        return vld1_f16((const float16_t *)aux);
    }

    inline float block_scale(int i) const { return GGML_FP16_TO_FP32(x[i].d); }

    const int8x16_t m8 = vdupq_n_s8(-8);
    //ggml_half aux[4];
};

struct DequantizerQ60 final : public BaseLegacyDequantizer<block_q6_0> {

    DequantizerQ60(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i, int8x16_t * q) const {
        bits.prepare1(x[i].qs, q);
        auto qh8 = vld1_u8(x[i].qh);
        auto qh  = vcombine_u8(vshl_n_u8(qh8, 4), qh8);
        q[0] = vaddq_s8(vorrq_u8(q[0], vandq_u8(qh, hmask)), m32);
        q[1] = vaddq_s8(vorrq_u8(q[1], vandq_u8(vshrq_n_u8(qh, 2), hmask)), m32);
    }
    inline void prepare1(int i) {
        prepare1(i, bits.b);
    }

    inline float16x4_t new_block(int i) {
        ggml_half aux[4];
        for (int k = 0; k < 4; ++k) {
            aux[k] = x[4*i+k].d;
            prepare1(4*i+k, bits.b + 2*k);
        }
        return vld1_f16((const float16_t *)aux);
    }
    inline float block_scale(int i) const { return GGML_FP16_TO_FP32(x[i].d); }

    const int8x16_t m32 = vdupq_n_s8(-32);
    const uint8x16_t hmask = vdupq_n_u8(0x30);
};

struct DequantizerIQ4NL final : public BaseLegacyDequantizer<block_iq4_nl> {

    DequantizerIQ4NL(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i, int8x16_t * q) const {
        bits.prepare1(x[i].qs, q);
        q[0] = vqtbl1q_s8(values, q[0]);
        q[1] = vqtbl1q_s8(values, q[1]);
    }
    inline void prepare1(int i) {
        prepare1(i, bits.b);
    }

    inline float16x4_t new_block(int i) {
        ggml_half aux[4];
        for (int k = 0; k < 4; ++k) {
            aux[k] = x[4*i+k].d;
            prepare1(4*i+k, bits.b + 2*k);
        }
        return vld1_f16((const float16_t *)aux);
    }
    static int8x16_t load_values() {
        static const int8_t iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        return vld1q_s8(iq4nl_values);
    }
    inline float block_scale(int i) const { return GGML_FP16_TO_FP32(x[i].d); }

    const int8x16_t values = load_values();
};

struct DequantizerMXFP4 final : public BaseLegacyDequantizer<block_mxfp4> {

    DequantizerMXFP4(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i, int8x16_t * q) const {
        bits.prepare1(x[i].qs, q);
        q[0] = vqtbl1q_s8(values, q[0]);
        q[1] = vqtbl1q_s8(values, q[1]);
    }
    inline void prepare1(int i) {
        prepare1(i, bits.b);
    }

    inline float16x4_t new_block(int i) {
        float aux[4];
        for (int k = 0; k < 4; ++k) {
            aux[k] = GGML_E8M0_TO_FP32_HALF(x[4*i+k].e);
            prepare1(4*i+k, bits.b + 2*k);
        }
        return vcvt_f16_f32(vld1q_f32(aux));
    }
    static int8x16_t load_values() {
        return vld1q_s8(kvalues_mxfp4);
    }
    inline float block_scale(int i) const { return GGML_E8M0_TO_FP32_HALF(x[i].e); }

    const int8x16_t values = load_values();
};

struct DequantizerQ41 : public BaseLegacyDequantizer<block_q4_1> {

    DequantizerQ41(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i) {
        bits.prepare1(x[i].qs);
    }

    inline float16x8_t new_block(int i) {
        uint32_t aux32[4];
        const uint32_t * s32 = (const uint32_t *)&x[4*i].d;
        for (int k = 0; k < 4; ++k) {
            aux32[k] = *s32; s32 += sizeof(block_q4_1)/4;
            bits.prepare1(x[4*i+k].qs, bits.b + 2*k);
        }
        return vreinterpretq_f16_u8(vqtbl1q_u8(vld1q_u8((const uint8_t *)aux32), vreinterpretq_u8_u64(shuffle)));
    }
    // Leaving this commented out attempt to be reminded that I already tried this.
    // It has basically the same performance as the version above.
    //inline float16x8_t new_block(int i) {
    //    uint32x4_t scales = {};
    //    const block_q4_1 * xi = x + 4*i;
    //    const uint32_t * s32 = (const uint32_t *)&xi->d;
    //    scales = vsetq_lane_u32(*s32, scales, 0); s32 += sizeof(block_q4_1)/4;
    //    bits.prepare1(xi[0].qs, bits.b + 0);
    //    scales = vsetq_lane_u32(*s32, scales, 1); s32 += sizeof(block_q4_1)/4;
    //    bits.prepare1(xi[1].qs, bits.b + 2);
    //    scales = vsetq_lane_u32(*s32, scales, 2); s32 += sizeof(block_q4_1)/4;
    //    bits.prepare1(xi[2].qs, bits.b + 4);
    //    scales = vsetq_lane_u32(*s32, scales, 3);
    //    bits.prepare1(xi[3].qs, bits.b + 6);
    //    return vreinterpretq_f16_u8(vqtbl1q_u8(vreinterpretq_u8_u32(scales), vreinterpretq_u8_u64(shuffle)));
    //}

    const uint64x2_t shuffle = {0x0d0c090805040100, 0x0f0e0b0a07060302};
};

struct HighBit5Legacy {
    inline uint8x16_t to_bytes(const uint8_t * qh) const {
        uint8x16_t h = vqtbl1q_u8(vreinterpretq_u8_u16(vdupq_n_u16(*(const uint16_t *)qh)), shuffle);
        return vceqq_u8(vandq_u8(h, vreinterpretq_u8_u64(mask)), vreinterpretq_u8_u64(mask));
    }
    inline uint8x16_t to_negated_bytes(const uint8_t * qh) const {
        uint8x16_t h = vqtbl1q_u8(vreinterpretq_u8_u16(vdupq_n_u16(*(const uint16_t *)qh)), shuffle);
        return vceqq_u8(vandq_u8(h, vreinterpretq_u8_u64(mask)), vdupq_n_u8(0));
    }
    const uint64x2_t mask = vdupq_n_u64(0x8040201008040201);
    const uint8x16_t shuffle = vcombine_u8(vdup_n_u8(0), vdup_n_u8(1));
};

struct DequantizerQ50 final : public BaseLegacyDequantizer<block_q5_0> {

    DequantizerQ50(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i, int8x16_t * q) const {
        bits.prepare1(x[i].qs, q);
        auto qh = x[i].qh;
        q[0] = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(q[0]), vandq_u8(mh, hbits.to_negated_bytes(qh+0))));
        q[1] = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(q[1]), vandq_u8(mh, hbits.to_negated_bytes(qh+2))));
    }
    inline void prepare1(int i) {
        prepare1(i, bits.b);
    }

    inline float16x4_t new_block(int i) {
        ggml_half aux[4];
        for (int k = 0; k < 4; ++k) {
            aux[k] = x[4*i+k].d;
            prepare1(4*i+k, bits.b + 2*k);
        }
        return vld1_f16((const float16_t *)aux);
    }
    inline float block_scale(int i) const { return GGML_FP16_TO_FP32(x[i].d); }

    HighBit5Legacy hbits;

    const uint8x16_t mh = vdupq_n_u8(0xf0);

};

struct DequantizerQ80 final : public BaseLegacyDequantizer<block_q8_0> {

    DequantizerQ80(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i) {
        bits.b[0] = vld1q_s8(x[i].qs);
        bits.b[1] = vld1q_s8(x[i].qs+16);
    }

    inline float16x4_t new_block(int i) {
        ggml_half aux[4];
        for (int k = 0; k < 4; ++k) {
            aux[k] = x[4*i+k].d;
            bits.b[2*k+0] = vld1q_s8(x[4*i+k].qs);
            bits.b[2*k+1] = vld1q_s8(x[4*i+k].qs+16);
        }
        return vld1_f16((const float16_t *)aux);
    }
    inline float block_scale(int i) const { return GGML_FP16_TO_FP32(x[i].d); }

};

// TODO: handle case where row size is not a multiple of 128
struct DequantizerQ80_x4 final : public BaseLegacyDequantizer<block_q8_0_x4> {

    DequantizerQ80_x4(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i) {
        bits.b[0] = vld1q_s8(x[i].qs);
        bits.b[1] = vld1q_s8(x[i].qs+16);
    }

    inline float16x4_t new_block(int i) {
        auto scale = vld1_f16((const float16_t *)x[i].d);
        for (int k = 0; k < 4; ++k) {
            bits.b[2*k+0] = vld1q_s8(x[i].qs+32*k);
            bits.b[2*k+1] = vld1q_s8(x[i].qs+32*k+16);
        }
        return scale;
    }

};

struct DequantizerQ51 final : public BaseLegacyDequantizer<block_q5_1> {

    DequantizerQ51(const void * vx, size_t bx) : BaseLegacyDequantizer(vx, bx) {}

    inline void prepare1(int i, int8x16_t * q) const {
        bits.prepare1(x[i].qs, q);
        auto qh = x[i].qh;
        q[0] = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(q[0]), vandq_u8(mh, hbits.to_bytes(qh+0))));
        q[1] = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(q[1]), vandq_u8(mh, hbits.to_bytes(qh+2))));
    }
    inline void prepare1(int i) {
        bits.prepare1(x[i].qs, bits.b);
    }

    inline float16x8_t new_block(int i) {
        uint32_t aux32[4];
        const uint32_t * s32 = (const uint32_t *)&x[4*i].d;
        for (int k = 0; k < 4; ++k) {
            aux32[k] = *s32; s32 += sizeof(block_q5_1)/4;
            prepare1(4*i+k, bits.b + 2*k);
        }
        return vreinterpretq_f16_u8(vqtbl1q_u8(vld1q_u8((const uint8_t *)aux32), vreinterpretq_u8_u64(shuffle)));
    }

    HighBit5Legacy hbits;

    const uint8x16_t mh = vdupq_n_u8(0x10);
    const uint64x2_t shuffle = {0x0d0c090805040100, 0x0f0e0b0a07060302};

};

template <typename Dequantizer, typename Q8>
inline void sum_4(int i, Dequantizer& deq, const Q8& q8, const float16x4_t * sc16, float32x4_t * acc) {
    for (int iy = 0; iy < Q8::nrc_y; ++iy) {
        auto pall = sum_4_blocks(deq.bits.b, q8.quant_data(iy, i));
        auto scale = vcvt_f32_f16(sc16[iy]);
        acc[iy] = vmlaq_f32(acc[iy], scale, vcvtq_f32_s32(pall));
    }
}

template <typename Dequantizer, typename Q8>
inline void sum_4(int i, Dequantizer& deq1, Dequantizer& deq2, const Q8& q8, const float16x4_t * sc16, float32x4_t * acc) {
    for (int iy = 0; iy < Q8::nrc_y; ++iy) {
        auto pall = sum_4_blocks(deq1.bits.b, deq2.bits.b, q8.quant_data(iy, i));
        auto scale1 = vcvt_f32_f16(sc16[iy]);
        auto scale2 = vcvt_f32_f16(sc16[iy+Q8::nrc_y]);
        acc[iy] = vmlaq_f32(acc[iy], scale1, vcvtq_f32_s32(pall.val[0]));
        acc[iy+Q8::nrc_y] = vmlaq_f32(acc[iy+Q8::nrc_y], scale2, vcvtq_f32_s32(pall.val[1]));
    }
}

template <typename Dequantizer, typename Q8>
inline void mul_mat_qX_Y_q8_Y(int n, Dequantizer& deq, Q8& q8, const DataInfo& info, int nrc_x) {
    const int nb = n / QK4_1;

    float16x4_t sc16[Q8::nrc_y];

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);

        float32x4_t acc[Q8::nrc_y];
        for (int iy = 0; iy < Q8::nrc_y; ++iy) acc[iy] = vdupq_n_f32(0.f);

        for (int i = 0; i < nb/4; ++i) {
            q8.process_scales(i, deq, sc16, acc);
            sum_4(i, deq, q8, sc16, acc);
        }
        for (int i = 4*(nb/4); i < nb; ++i) {
            q8.process_1_block(i, deq, acc);
        }

        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            info.store(ix, iy, vaddvq_f32(acc[iy]));
        }
    }
}

template <typename Dequantizer, typename Q8>
inline void mul_mat_qX_Y_q8_Y_IK(int n, Dequantizer& deq1, Dequantizer& deq2, Q8& q8, const DataInfo& info, int nrc_x) {
    const int nb = n / QK4_1;

    float16x4_t sc16[2*Q8::nrc_y];
    float32x4_t acc[2*Q8::nrc_y];

    for (int ix = 0; ix < nrc_x; ix += 2) {

        deq1.new_row(ix+0);
        deq2.new_row(ix+1);

        for (int iy = 0; iy < 2*Q8::nrc_y; ++iy) acc[iy] = vdupq_n_f32(0.f);

        for (int i = 0; i < nb/4; ++i) {
            q8.process_scales(i, deq1, deq2, sc16, acc);
            sum_4(i, deq1, deq2, q8, sc16, acc);
        }
        //for (int i = 4*(nb/4); i < nb; ++i) {
        //    q8.process_1_block(i, deq, acc);
        //}

        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            info.store(ix+0, iy, vaddvq_f32(acc[iy]));
            info.store(ix+1, iy, vaddvq_f32(acc[iy+Q8::nrc_y]));
        }
    }
}

template <typename Dequantizer, typename Q8>
inline void mul_mat_qX_Y_q8_Y_1(int n, Dequantizer& deq1, Dequantizer& deq2, Q8& q8, const DataInfo& info, int nrc_x) {
    const int nb = n / QK4_1;

    float16x4_t sc16[2];

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq1.new_row(ix);
        deq2.new_row(ix);

        float32x4_t acc[2] = { vdupq_n_f32(0.f), vdupq_n_f32(0.f) };

        for (int i = 0; i < nb/8; ++i) {
            q8.process_scales(2*i+0, deq1, sc16+0, acc+0);
            q8.process_scales(2*i+1, deq2, sc16+1, acc+1);
            sum_4(2*i+0, deq1, q8, sc16+0, acc+0);
            sum_4(2*i+1, deq2, q8, sc16+1, acc+1);
        }
        for (int i = 2*(nb/8); i < nb/4; ++i) {
            q8.process_scales(i, deq1, sc16, acc);
            sum_4(i, deq1, q8, sc16, acc);
        }
        //for (int i = 4*(nb/4); i < nb; ++i) {
        //    q8.process_1_block(i, deq1, acc);
        //}

        info.store(ix, 0, vaddvq_f32(vaddq_f32(acc[0], acc[1])));
    }
}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_1_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    Q81<nrc_y> q8(info);
    if constexpr (nrc_y == 1) {
        Dequantizer deq1(vx, bx), deq2(vx, bx);
        mul_mat_qX_Y_q8_Y_1(n, deq1, deq2, q8, info, nrc_x);
    } else {
        if (nrc_x%2 == 0 && n%128 == 0) {
            Dequantizer deq1(vx, bx), deq2(vx, bx);
            mul_mat_qX_Y_q8_Y_IK(n, deq1, deq2, q8, info, nrc_x);
        } else {
            Dequantizer deq(vx, bx);
            mul_mat_qX_Y_q8_Y(n, deq, q8, info, nrc_x);
        }
    }
}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_0_q8_0(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    Q80<nrc_y> q8(info);
    if constexpr (nrc_y == 1) {
        Dequantizer deq1(vx, bx), deq2(vx, bx);
        mul_mat_qX_Y_q8_Y_1(n, deq1, deq2, q8, info, nrc_x);
    } else {
        if (nrc_x%2 == 0 && n%128 == 0) {
            Dequantizer deq1(vx, bx), deq2(vx, bx);
            mul_mat_qX_Y_q8_Y_IK(n, deq1, deq2, q8, info, nrc_x);
        } else {
            Dequantizer deq(vx, bx);
            mul_mat_qX_Y_q8_Y(n, deq, q8, info, nrc_x);
        }
    }
}

template <typename Dequantizer>
static void mul_mat_qX_1_q8_1_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    Dequantizer deq1(vx, bx), deq2(vx, bx);
    Q81<1> q8(info);
    mul_mat_qX_Y_q8_Y_1(n, deq1, deq2, q8, info, nrc_x);
}

template <typename Dequantizer>
static void mul_mat_qX_0_q8_0_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    Dequantizer deq1(vx, bx), deq2(vx, bx);
    Q80<1> q8(info);
    mul_mat_qX_Y_q8_Y(n, deq1, deq2, q8, info, nrc_x);
}

template <typename Dequantizer, int nrc_y>
void mul_mat_qx_r4_q8_0(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_0_x4> q8(info);
    Dequantizer deq(vx, bx);
    int nb = n / QK4_NL;
    int8x16_t qx[8];
    float d8[4*nrc_y];
    float32x4_t acc[nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 4) {
        deq.new_row(ix);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                vst1q_f32(d8+4*iy, vcvt_f32_f16(vld1_f16((const float16_t *)q8.y[iy][ib4].d)));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = deq.prepare(4*ib4+k, qx);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = vld1q_s8_x2(q8.y[iy][ib4].qs+32*k);
                    auto sumi = interleaved_dotq(qx, y);
                    auto d4d8 = vmulq_f32(scales, vdupq_n_f32(d8[4*iy+k]));
                    acc[iy] = vfmaq_f32(acc[iy], d4d8, vcvtq_f32_s32(sumi));
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = deq.prepare(ib, qx);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_0 *)q8.y[iy];
                auto y = vld1q_s8_x2(qy[ib].qs);
                auto sumi = interleaved_dotq(qx, y);
                auto d4d8 = vmulq_f32(scales, vdupq_n_f32(GGML_FP16_TO_FP32(qy[ib].d)));
                acc[iy] = vfmaq_f32(acc[iy], d4d8, vcvtq_f32_s32(sumi));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, deq.result(acc[iy]));
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

template <typename Dequantizer, int nrc_y>
void mul_mat_qx_r8_q8_0(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_0_x4> q8(info);
    Dequantizer deq(vx, bx);
    int nb = n / QK4_NL;
    int8x16_t qx[16];
    float d8[4*nrc_y];
    float32x4_t acc[2*nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 8) {
        deq.new_row(ix);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                vst1q_f32(d8+4*iy, vcvt_f32_f16(vld1_f16((const float16_t *)q8.y[iy][ib4].d)));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales = deq.prepare(ib4, k, qx);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = vld1q_s8_x2(q8.y[iy][ib4].qs+32*k);
                    auto sumi1 = interleaved_dotq(qx+0, y);
                    auto sumi2 = interleaved_dotq(qx+8, y);
                    auto dy = vdupq_n_f32(d8[4*iy+k]);
                    acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], vmulq_f32(scales.val[0], dy), vcvtq_f32_s32(sumi1));
                    acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], vmulq_f32(scales.val[1], dy), vcvtq_f32_s32(sumi2));
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = deq.prepare(ib, 0, qx);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_0 *)q8.y[iy];
                auto y = vld1q_s8_x2(qy[ib].qs);
                auto sumi1 = interleaved_dotq(qx+0, y);
                auto sumi2 = interleaved_dotq(qx+8, y);
                auto dy = vdupq_n_f32(GGML_FP16_TO_FP32(qy[ib].d));
                acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], vmulq_f32(scales.val[0], dy), vcvtq_f32_s32(sumi1));
                acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], vmulq_f32(scales.val[1], dy), vcvtq_f32_s32(sumi2));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix+0, iy, deq.result(acc[2*iy+0]));
            info.store(ix+4, iy, deq.result(acc[2*iy+1]));
            acc[2*iy] = acc[2*iy+1] = vdupq_n_f32(0.f);
        }
    }
}

struct IQ4_NL_R4_Dequantizer {
    IQ4_NL_R4_Dequantizer(const void * vx, size_t bx) : cx((const char *)vx), bx(bx), values(vld1q_s8(iq4k_values)) {}
    inline void new_row(int ix) { iq4 = (const block_iq4_nl_r4 *)(cx + ix*bx); }
    inline float32x4_t prepare(int ib, int8x16_t * qx) const {
        auto scales = vcvt_f32_f16(vld1_f16((const float16_t *)iq4[ib].d));
        auto bits   = vld1q_u8_x4(iq4[ib].qs);
        prepare_iq4_nl_quants(values, m4, bits, qx);
        return scales;
    }
    inline float32x4_t result(float32x4_t acc) const {
        return acc;
    }

    const char * cx;
    const size_t bx;
    const block_iq4_nl_r4 * iq4;
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const int8x16_t values;
};

struct Q4_0_R8_Dequantizer {
    Q4_0_R8_Dequantizer(const void * vx, size_t bx) : cx((const char *)vx), bx(bx) {}
    inline void new_row(int ix) { iq4 = (const block_iq4_nl_r8 *)(cx + ix*bx); }
    inline float32x4x2_t prepare(int ib4, int k, int8x16_t * qx) const {
        auto scales16 = vld1q_f16((const float16_t *)iq4[4*ib4+k].d);
        float32x4x2_t scales = { vcvt_f32_f16(vget_low_f16(scales16)), vcvt_f32_f16(vget_high_f16(scales16)) };
        for (int j = 0; j < 4; ++j) {
            auto bits = vld1q_u8_x2(iq4[4*ib4+k].qs + 32*j);
            bits.val[0] = veorq_u8(m88, bits.val[0]);
            bits.val[1] = veorq_u8(m88, bits.val[1]);
            qx[2*j+0] = vshlq_n_u8(bits.val[0], 4);
            qx[2*j+1] = vandq_u8(bits.val[0], m4);
            qx[2*j+8] = vshlq_n_u8(bits.val[1], 4);
            qx[2*j+9] = vandq_u8(bits.val[1], m4);
        }
        return scales;
    }
    inline float32x4_t result(float32x4_t acc) const {
        return vmulq_f32(norm, acc);
    }

    const char * cx;
    const size_t bx;
    const block_iq4_nl_r8 * iq4;
    const uint8x16_t m4 = vdupq_n_u8(0xf0);
    const uint8x16_t m88 = vdupq_n_u8(0x88);
    const float32x4_t norm = vdupq_n_f32(1.f/16);
};

struct Q5_0_R4_Dequantizer {
    Q5_0_R4_Dequantizer(const void * vx, size_t bx) : cx((const char *)vx), bx(bx) {}
    inline void new_row(int ix) { iq5 = (const block_q5_0_r4 *)(cx + ix*bx); }
    inline float32x4_t prepare(int ib, int8x16_t * qx) const {
        auto scales = vcvt_f32_f16(vld1_f16((const float16_t *)iq5[ib].d));
        auto lbits   = vld1q_u8_x4(iq5[ib].qs);
        auto hbits   = vld1q_u8(iq5[ib].qh);
        qx[0] = vaddq_s8(vandq_u8(lbits.val[0], m4) | vandq_u8(vshlq_n_u8(hbits, 4), m5), m16); //  0...3
        qx[1] = vaddq_s8(vandq_u8(lbits.val[1], m4) | vandq_u8(vshlq_n_u8(hbits, 3), m5), m16); // 16..19
        qx[2] = vaddq_s8(vandq_u8(lbits.val[2], m4) | vandq_u8(vshlq_n_u8(hbits, 2), m5), m16); //  4...7
        qx[3] = vaddq_s8(vandq_u8(lbits.val[3], m4) | vandq_u8(vshlq_n_u8(hbits, 1), m5), m16); // 20..23
        qx[4] = vaddq_s8(vshrq_n_u8(lbits.val[0], 4)| vandq_u8(hbits, m5), m16);                //  8..11
        qx[5] = vaddq_s8(vshrq_n_u8(lbits.val[1], 4)| vandq_u8(vshrq_n_u8(hbits, 1), m5), m16); // 24..27
        qx[6] = vaddq_s8(vshrq_n_u8(lbits.val[2], 4)| vandq_u8(vshrq_n_u8(hbits, 2), m5), m16); // 12..15
        qx[7] = vaddq_s8(vshrq_n_u8(lbits.val[3], 4)| vandq_u8(vshrq_n_u8(hbits, 3), m5), m16); // 28..31
        return scales;
    }
    inline float32x4_t result(float32x4_t acc) const {
        return acc;
    }

    const char * cx;
    const size_t bx;
    const block_q5_0_r4 * iq5;
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const uint8x16_t m5 = vdupq_n_u8(0x10);
    const int8x16_t m16 = vdupq_n_s8(-16);
};

struct Q6_0_R4_Dequantizer {
    Q6_0_R4_Dequantizer(const void * vx, size_t bx) : cx((const char *)vx), bx(bx) {}
    inline void new_row(int ix) { iq6 = (const block_q6_0_r4 *)(cx + ix*bx); }
    inline float32x4_t prepare(int ib, int8x16_t * qx) const {
        auto scales = vcvt_f32_f16(vld1_f16((const float16_t *)iq6[ib].d));
        auto lbits   = vld1q_u8_x4(iq6[ib].qs);
        auto hbits   = vld1q_u8_x2(iq6[ib].qh);
        qx[0] = vaddq_s8(vandq_u8(lbits.val[0], m4) | vandq_u8(vshlq_n_u8(hbits.val[0], 4), m6), m32); //  0...3
        qx[1] = vaddq_s8(vandq_u8(lbits.val[1], m4) | vandq_u8(vshlq_n_u8(hbits.val[1], 4), m6), m32); // 16..19
        qx[2] = vaddq_s8(vandq_u8(lbits.val[2], m4) | vandq_u8(vshlq_n_u8(hbits.val[0], 2), m6), m32); //  4...7
        qx[3] = vaddq_s8(vandq_u8(lbits.val[3], m4) | vandq_u8(vshlq_n_u8(hbits.val[1], 2), m6), m32); // 20..23
        qx[4] = vaddq_s8(vshrq_n_u8(lbits.val[0], 4)| vandq_u8(hbits.val[0], m6), m32);                //  8..11
        qx[5] = vaddq_s8(vshrq_n_u8(lbits.val[1], 4)| vandq_u8(hbits.val[1], m6), m32);                // 24..27
        qx[6] = vaddq_s8(vshrq_n_u8(lbits.val[2], 4)| vandq_u8(vshrq_n_u8(hbits.val[0], 2), m6), m32); // 12..15
        qx[7] = vaddq_s8(vshrq_n_u8(lbits.val[3], 4)| vandq_u8(vshrq_n_u8(hbits.val[1], 2), m6), m32); // 28..31
        return scales;
    }
    inline float32x4_t result(float32x4_t acc) const {
        return acc;
    }

    const char * cx;
    const size_t bx;
    const block_q6_0_r4 * iq6;
    const uint8x16_t m4 = vdupq_n_u8(0x0f);
    const uint8x16_t m6 = vdupq_n_u8(0x30);
    const int8x16_t m32 = vdupq_n_s8(-32);
};

inline void qx_0_q8_0_dot(const int8x16_t * qx, const int8_t * qy, int32x4_t& sumi1, int32x4_t& sumi2) {
    auto y = vld1q_s8_x2(qy);
    sumi1 = sumi2 = vdupq_n_s32(0);
    sumi1 = vdotq_laneq_s32(sumi1, qx[0], y.val[0], 0);
    sumi2 = vdotq_laneq_s32(sumi2, qx[1], y.val[0], 0);
    sumi1 = vdotq_laneq_s32(sumi1, qx[2], y.val[0], 1);
    sumi2 = vdotq_laneq_s32(sumi2, qx[3], y.val[0], 1);
    sumi1 = vdotq_laneq_s32(sumi1, qx[4], y.val[0], 2);
    sumi2 = vdotq_laneq_s32(sumi2, qx[5], y.val[0], 2);
    sumi1 = vdotq_laneq_s32(sumi1, qx[6], y.val[0], 3);
    sumi2 = vdotq_laneq_s32(sumi2, qx[7], y.val[0], 3);
    sumi1 = vdotq_laneq_s32(sumi1, qx[8+0], y.val[1], 0);
    sumi2 = vdotq_laneq_s32(sumi2, qx[8+1], y.val[1], 0);
    sumi1 = vdotq_laneq_s32(sumi1, qx[8+2], y.val[1], 1);
    sumi2 = vdotq_laneq_s32(sumi2, qx[8+3], y.val[1], 1);
    sumi1 = vdotq_laneq_s32(sumi1, qx[8+4], y.val[1], 2);
    sumi2 = vdotq_laneq_s32(sumi2, qx[8+5], y.val[1], 2);
    sumi1 = vdotq_laneq_s32(sumi1, qx[8+6], y.val[1], 3);
    sumi2 = vdotq_laneq_s32(sumi2, qx[8+7], y.val[1], 3);
}

template <int nrc_y>
void mul_mat_q8_0_r8_q8_0(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_0_x4> q8(info);
    int nb = n / QK8_0;
    float32x4_t acc[2*nrc_y] = {};
    int8x16_t qx[16];
    float d8[4*nrc_y];
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q8_0_r8 * iq8 = (const block_q8_0_r8 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                vst1q_f32(d8+4*iy, vcvt_f32_f16(vld1_f16((const float16_t *)q8.y[iy][ib4].d)));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales16 = vld1q_f16((const float16_t *)iq8[4*ib4+k].d);
                auto scales1 = vcvt_f32_f16(vget_low_f16 (scales16));
                auto scales2 = vcvt_f32_f16(vget_high_f16(scales16));
                for (int j = 0; j < 16; ++j) qx[j] = vld1q_s8(iq8[4*ib4+k].qs + 16*j);
                int32x4_t sumi1, sumi2;
                for (int iy = 0; iy < nrc_y; ++iy) {
                    qx_0_q8_0_dot(qx, q8.y[iy][ib4].qs+32*k, sumi1, sumi2);
                    auto dy = vdupq_n_f32(d8[4*iy+k]);
                    acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], vmulq_f32(scales1, dy), vcvtq_f32_s32(sumi1));
                    acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], vmulq_f32(scales2, dy), vcvtq_f32_s32(sumi2));
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales16 = vld1q_f16((const float16_t *)iq8[ib].d);
            auto scales1 = vcvt_f32_f16(vget_low_f16 (scales16));
            auto scales2 = vcvt_f32_f16(vget_high_f16(scales16));
            for (int j = 0; j < 16; ++j) qx[j] = vld1q_s8(iq8[ib].qs + 16*j);
            int32x4_t sumi1, sumi2;
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_0 *)q8.y[iy];
                qx_0_q8_0_dot(qx, qy[ib].qs, sumi1, sumi2);
                auto dy = vdupq_n_f32(GGML_FP16_TO_FP32(qy[ib].d));
                acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], vmulq_f32(scales1, dy), vcvtq_f32_s32(sumi1));
                acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], vmulq_f32(scales2, dy), vcvtq_f32_s32(sumi2));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix+0, iy, acc[2*iy+0]);
            info.store(ix+4, iy, acc[2*iy+1]);
            acc[2*iy] = acc[2*iy+1] = vdupq_n_f32(0.f);
        }
    }
}

typedef struct {
    ggml_half d[16];
    int8_t    qs[256];
} block_q8_1_r8;

template <int nrc_y>
void mul_mat_q8_1_r8_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_1_x4> q8(info);
    int nb = n / QK8_0;
    float32x4_t acc[2*nrc_y] = {};
    int8x16_t qx[16];
    float d8[8*nrc_y];
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q8_1_r8 * iq8 = (const block_q8_1_r8 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                vst1q_f32(d8+8*iy+0, vcvt_f32_f16(vld1_f16((const float16_t *)q8.y[iy][ib4].d+0)));
                vst1q_f32(d8+8*iy+4, vcvt_f32_f16(vld1_f16((const float16_t *)q8.y[iy][ib4].d+4)));
            }
            for (int k = 0; k < 4; ++k) {
                auto scales16 = vld1q_f16((const float16_t *)iq8[4*ib4+k].d);
                auto scales1 = vcvt_f32_f16(vget_low_f16 (scales16));
                auto scales2 = vcvt_f32_f16(vget_high_f16(scales16));
                auto m16 = vld1q_f16((const float16_t *)iq8[4*ib4+k].d+8);
                auto m1 = vcvt_f32_f16(vget_low_f16 (m16));
                auto m2 = vcvt_f32_f16(vget_high_f16(m16));
                for (int j = 0; j < 16; ++j) qx[j] = vld1q_s8(iq8[4*ib4+k].qs + 16*j);
                int32x4_t sumi1, sumi2;
                for (int iy = 0; iy < nrc_y; ++iy) {
                    qx_0_q8_0_dot(qx, q8.y[iy][ib4].qs+32*k, sumi1, sumi2);
                    auto dy = vdupq_n_f32(d8[8*iy+k]);
                    acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], vmulq_f32(scales1, dy), vcvtq_f32_s32(sumi1));
                    acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], vmulq_f32(scales2, dy), vcvtq_f32_s32(sumi2));
                    auto my = vdupq_n_f32(d8[8*iy+k+4]);
                    acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], m1, my);
                    acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], m2, my);
                }
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales16 = vld1q_f16((const float16_t *)iq8[ib].d);
            auto scales1 = vcvt_f32_f16(vget_low_f16 (scales16));
            auto scales2 = vcvt_f32_f16(vget_high_f16(scales16));
            auto m16 = vld1q_f16((const float16_t *)iq8[ib].d+8);
            auto m1 = vcvt_f32_f16(vget_low_f16 (m16));
            auto m2 = vcvt_f32_f16(vget_high_f16(m16));
            for (int j = 0; j < 16; ++j) qx[j] = vld1q_s8(iq8[ib].qs + 16*j);
            int32x4_t sumi1, sumi2;
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                qx_0_q8_0_dot(qx, qy[ib].qs, sumi1, sumi2);
                auto dy = vdupq_n_f32(GGML_FP16_TO_FP32(qy[ib].d));
                acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], vmulq_f32(scales1, dy), vcvtq_f32_s32(sumi1));
                acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], vmulq_f32(scales2, dy), vcvtq_f32_s32(sumi2));
                auto my = vdupq_n_f32(GGML_FP16_TO_FP32(qy[ib].s));
                acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], m1, my);
                acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], m2, my);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix+0, iy, acc[2*iy+0]);
            info.store(ix+4, iy, acc[2*iy+1]);
            acc[2*iy] = acc[2*iy+1] = vdupq_n_f32(0.f);
        }
    }
}

struct DeqQ40 {
    const int8x16_t m8  = vdupq_n_s8(-8);
    const uint8x16_t ml = vdupq_n_s8(0xf);
    inline int8x16x2_t dequant(const block_q4_0& x) const {
        auto bits = vld1q_u8(x.qs);
        return { vaddq_s8(vreinterpretq_s8_u8(vandq_u8(bits, ml)), m8), vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(bits, 4)), m8) };
    }
};

struct DeqQ41 {
    const uint8x16_t ml = vdupq_n_s8(0xf);
    inline int8x16x2_t dequant(const block_q4_1& x) const {
        auto bits = vld1q_u8(x.qs);
        return { vreinterpretq_s8_u8(vandq_u8(bits, ml)), vreinterpretq_s8_u8(vshrq_n_u8(bits, 4)) };
    }
};

struct DeqIQ4NL {
    const int8x16_t mt  = load_values();
    const uint8x16_t ml = vdupq_n_s8(0xf);
    inline int8x16x2_t dequant(const block_iq4_nl& x) const {
        auto bits = vld1q_u8(x.qs);
        return { vqtbl1q_s8(mt, vandq_u8(bits, ml)), vqtbl1q_s8(mt, vshrq_n_u8(bits, 4)) };
    }
    static inline int8x16_t load_values() { return vld1q_s8(iq4k_values); }
};

struct DeqMXFP4 {
    const int8x16_t mt  = load_values();
    const uint8x16_t ml = vdupq_n_s8(0xf);
    inline int8x16x2_t dequant(const block_mxfp4& x) const {
        auto bits = vld1q_u8(x.qs);
        return { vqtbl1q_s8(mt, vandq_u8(bits, ml)), vqtbl1q_s8(mt, vshrq_n_u8(bits, 4)) };
    }
    static inline int8x16_t load_values() { return vld1q_s8(kvalues_mxfp4); }
};

struct DeqQ50 {

    inline int8x16x2_t dequant(const block_q5_0& x) const {
        int8x16x2_t r;
        bits.prepare1(x.qs, r.val);
        auto qh = x.qh;
        r.val[0] = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(r.val[0]), vandq_u8(mh, hbits.to_negated_bytes(qh+0))));
        r.val[1] = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(r.val[1]), vandq_u8(mh, hbits.to_negated_bytes(qh+2))));
        return r;
    }

    Q4LegacyBits bits;
    HighBit5Legacy hbits;
    const uint8x16_t mh = vdupq_n_u8(0xf0);
};

struct DeqQ51 {

    inline int8x16x2_t dequant(const block_q5_1& x) const {
        int8x16x2_t r;
        bits.prepare1(x.qs, r.val);
        auto qh = x.qh;
        r.val[0] = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(r.val[0]), vandq_u8(mh, hbits.to_bytes(qh+0))));
        r.val[1] = vreinterpretq_s8_u8(vorrq_u8(vreinterpretq_u8_s8(r.val[1]), vandq_u8(mh, hbits.to_bytes(qh+2))));
        return r;
    }

    Q4LegacyBits bits;
    HighBit5Legacy hbits;
    const uint8x16_t mh = vdupq_n_u8(0x10);
};

struct DeqQ60 {

    inline int8x16x2_t dequant(const block_q6_0& x) const {
        int8x16x2_t r;
        bits.prepare1(x.qs, r.val);
        auto qh8 = vld1_u8(x.qh);
        auto qh  = vcombine_u8(vshl_n_u8(qh8, 4), qh8);
        r.val[0] = vaddq_s8(vorrq_u8(r.val[0], vandq_u8(qh, hmask)), m32);
        r.val[1] = vaddq_s8(vorrq_u8(r.val[1], vandq_u8(vshrq_n_u8(qh, 2), hmask)), m32);
        return r;
    }

    Q4LegacyBits bits;
    const int8x16_t m32 = vdupq_n_s8(-32);
    const uint8x16_t hmask = vdupq_n_u8(0x30);
};

struct DeqQ80 {
    inline int8x16x2_t dequant(const block_q8_0& x) const {
        return vld1q_s8_x2(x.qs);
    }
};

template <typename Block, typename Dequantizer>
void iqk_convert_qX_q80_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK4_0 == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    const int nb = n/QK8_0;

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    const Block * x8[8];

    uint32_t block[8];

    Dequantizer deq;

    for (int ix = 0; ix < nrc_x; ix += 8) {

        for (int k = 0; k < 8; ++k) x8[k] = (const Block *)((const char *)vx + (ix + k)*bx);

        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                if constexpr (std::is_same_v<Dequantizer, DeqMXFP4>) {
                    y[i].d[k] = GGML_FP32_TO_FP16(GGML_E8M0_TO_FP32_HALF(x8[k][i].e));
                } else {
                    y[i].d[k] = x8[k][i].d;
                }
                vst1q_s8_x2((int8_t *)block, deq.dequant(x8[k][i]));
                auto qs = (uint32_t *)y[i].qs;
                for (int l = 0; l < 4; ++l) {
                    qs[8*l + k +  0] = block[l + 0];
                    qs[8*l + k + 32] = block[l + 4];
                }
            }
        }
        y += nb;
    }
}

template <typename Block, typename Dequantizer>
void iqk_convert_qX_1_q8_1_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK4_0 == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    const int nb = n/QK8_0;

    block_q8_1_r8 * y = (block_q8_1_r8 *)vy;

    const Block * x8[8];

    uint32_t block[8];

    Dequantizer deq;

    for (int ix = 0; ix < nrc_x; ix += 8) {

        for (int k = 0; k < 8; ++k) x8[k] = (const Block *)((const char *)vx + (ix + k)*bx);

        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                y[i].d[k+0] = x8[k][i].d;
                y[i].d[k+8] = x8[k][i].m;
                vst1q_s8_x2((int8_t *)block, deq.dequant(x8[k][i]));
                auto qs = (uint32_t *)y[i].qs;
                for (int l = 0; l < 4; ++l) {
                    qs[8*l + k +  0] = block[l + 0];
                    qs[8*l + k + 32] = block[l + 4];
                }
            }
        }
        y += nb;
    }
}

}

bool iqk_convert_legacy_quants_q8_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    switch (type) {
        case GGML_TYPE_Q4_0  : iqk_convert_qX_q80_r8<block_q4_0, DeqQ40>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q4_1  : iqk_convert_qX_1_q8_1_r8<block_q4_1, DeqQ41>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q5_0  : iqk_convert_qX_q80_r8<block_q5_0, DeqQ50>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q5_1  : iqk_convert_qX_1_q8_1_r8<block_q5_1, DeqQ51>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q6_0  : iqk_convert_qX_q80_r8<block_q6_0, DeqQ60>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_NL: iqk_convert_qX_q80_r8<block_iq4_nl, DeqIQ4NL>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_MXFP4 : iqk_convert_qX_q80_r8<block_mxfp4, DeqMXFP4>(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_Q8_0  : iqk_convert_qX_q80_r8<block_q8_0, DeqQ80>(n, vx, bx, vy, nrc_x); break;
        default: return false;
    }
    return true;
}

bool iqk_set_kernels_legacy_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {

    if (ne00%QK8_0 != 0) return false;

    auto etypeA = ggml_type(typeA);
    auto expected_typeB = etypeA == GGML_TYPE_Q4_1 || etypeA == GGML_TYPE_Q5_1 || etypeA == GGML_TYPE_Q8_1 ? GGML_TYPE_Q8_1_X4 : GGML_TYPE_Q8_0_X4;
    if (ggml_type(typeB) != expected_typeB) return false;

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_Q4_0:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0, DequantizerQ40, kernels);
            break;
        case GGML_TYPE_Q4_1:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_1, DequantizerQ41, kernels);
            break;
        case GGML_TYPE_Q5_0:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0, DequantizerQ50, kernels);
            break;
        case GGML_TYPE_Q5_1:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_1_q8_1, DequantizerQ51, kernels);
            break;
        case GGML_TYPE_Q6_0:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0, DequantizerQ60, kernels);
            break;
        case GGML_TYPE_Q8_0:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0, DequantizerQ80, kernels);
            break;
        case GGML_TYPE_IQ4_NL:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0, DequantizerIQ4NL, kernels);
            break;
        case GGML_TYPE_MXFP4:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_0_q8_0, DequantizerMXFP4, kernels);
            break;
        case GGML_TYPE_Q4_0_R8:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qx_r8_q8_0, Q4_0_R8_Dequantizer, kernels);
            break;
        case GGML_TYPE_Q5_0_R4:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qx_r4_q8_0, Q5_0_R4_Dequantizer, kernels);
            break;
        case GGML_TYPE_Q6_0_R4:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qx_r4_q8_0, Q6_0_R4_Dequantizer, kernels);
            break;
        case GGML_TYPE_Q8_0_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_0_r8_q8_0, kernels);
            break;
        case GGML_TYPE_Q8_1:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_1_r8_q8_1, kernels);
            break;
        case GGML_TYPE_IQ4_NL_R4:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qx_r4_q8_0, IQ4_NL_R4_Dequantizer, kernels);
            break;
        default:
            return false;
    }

    return true;
}

#endif

namespace {
template <int k_step>
inline std::pair<mul_mat_t, int> mul_mat_kernel(int int_typeA, int nq) {
    auto typeA = ggml_type(int_typeA);
    constexpr int kMaxQ = 8;
#define MAKE_FUNCS(mul_mat, n) \
    if (n >= kMaxQ) return std::make_pair(mul_mat, kMaxQ>, kMaxQ);\
    else {\
        switch (n) {\
            case 1: return std::make_pair(mul_mat, 1>, 1);\
            case 2: return std::make_pair(mul_mat, 2>, 2);\
            case 3: return std::make_pair(mul_mat, 3>, 3);\
            case 4: return std::make_pair(mul_mat, 4>, 4);\
            case 5: return std::make_pair(mul_mat, 5>, 5);\
            case 6: return std::make_pair(mul_mat, 6>, 6);\
            case 7: return std::make_pair(mul_mat, 7>, 7);\
        }\
    }
#define MAKE_FUNCS2(mul_mat, block, n) \
    if (n >= kMaxQ) return std::make_pair(mul_mat, kMaxQ, block>, kMaxQ);\
    else {\
        switch (n) {\
            case 1: return std::make_pair(mul_mat, 1, block>, 1);\
            case 2: return std::make_pair(mul_mat, 2, block>, 2);\
            case 3: return std::make_pair(mul_mat, 3, block>, 3);\
            case 4: return std::make_pair(mul_mat, 4, block>, 4);\
            case 5: return std::make_pair(mul_mat, 5, block>, 5);\
            case 6: return std::make_pair(mul_mat, 6, block>, 6);\
            case 7: return std::make_pair(mul_mat, 7, block>, 7);\
        }\
    }
#define MAKE_FUNCS_ONLY_NRC(mul_mat, n) \
    if (n >= kMaxQ) return std::make_pair(mul_mat<kMaxQ>, kMaxQ);\
    else {\
        switch (n) {\
            case 1: return std::make_pair(mul_mat<1>, 1);\
            case 2: return std::make_pair(mul_mat<2>, 2);\
            case 3: return std::make_pair(mul_mat<3>, 3);\
            case 4: return std::make_pair(mul_mat<4>, 4);\
            case 5: return std::make_pair(mul_mat<5>, 5);\
            case 6: return std::make_pair(mul_mat<6>, 6);\
            case 7: return std::make_pair(mul_mat<7>, 7);\
        }\
    }
    if (typeA == GGML_TYPE_Q8_0) {
#ifdef __aarch64__
        MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerQ80, nq);
#else
#ifdef HAVE_FANCY_SIMD
        if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q8_0_1_Unpacker, 1, k_step>, 1);
        if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q8_0_1_Unpacker, 2, k_step>, 2);
        if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q8_0_1_Unpacker, 4, k_step>, 4);
        MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q8_0_1_Unpacker, nq);
#else
        //if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_0_Tx<Q8_0_Unpacker, 1, k_step>, 1);
        //if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_0_Tx<Q8_0_Unpacker, 2, k_step>, 2);
        //if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_0_Tx<Q8_0_Unpacker, 4, k_step>, 4);
        if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 1, block_q8_2>, 1);
        if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 2, block_q8_2>, 2);
        if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 4, block_q8_2>, 4);
        if (nq == 3) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 3, block_q8_2>, 3);
        if (nq == 5) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 5, block_q8_2>, 5);
        if (nq == 6) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 6, block_q8_2>, 6);
        if (nq == 7) return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 7, block_q8_2>, 7);
        return std::make_pair(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, kMaxQ, block_q8_2>, kMaxQ);
#endif
#endif
    }
    else if (typeA == GGML_TYPE_Q8_0_R8) {
#ifdef __aarch64__
        MAKE_FUNCS_ONLY_NRC(mul_mat_q8_0_r8_q8_0, nq);
#else
        MAKE_FUNCS_ONLY_NRC(mul_mat_q8_0_r8_q8_2, nq);
#endif
    }
    else if (typeA == GGML_TYPE_Q6_0) {
#ifdef __aarch64__
        MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerQ60, nq);
#else
        if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q6_0_1_Unpacker, 1, k_step>, 1);
        if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q6_0_1_Unpacker, 2, k_step>, 2);
        if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q6_0_1_Unpacker, 4, k_step>, 4);
        MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q6_0_1_Unpacker, nq);
#endif
    }
    else if (typeA == GGML_TYPE_Q4_0) {
#ifdef __aarch64__
        MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerQ40, nq);
#else
        if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q4_0_1_Unpacker, 1, k_step>, 1);
        if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q4_0_1_Unpacker, 2, k_step>, 2);
        if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q4_0_1_Unpacker, 4, k_step>, 4);
        MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q4_0_1_Unpacker, nq);
#endif
    }
#if GGML_IQK_FA_ALL_QUANTS
    else if (typeA == GGML_TYPE_Q4_1) {
#ifdef __aarch64__
        MAKE_FUNCS(mul_mat_qX_1_q8_1<DequantizerQ41, nq);
#else
        MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q4_1_Unpacker, nq);
#endif
    }
    else if (typeA == GGML_TYPE_IQ4_NL) {
#ifdef __aarch64__
       MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerIQ4NL, nq);
#else
#ifdef HAVE_FANCY_SIMD
       MAKE_FUNCS(mul_mat_qX_1_q8_2_T<IQ4_NL_UnpackerU, nq);
#else
       MAKE_FUNCS2(mul_mat_qX_0_q8_0_T<IQ4_NL_UnpackerS, block_q8_2, nq);
#endif
#endif
    }
#endif
    else {
        GGML_ASSERT(false);
    }
    return std::make_pair<mul_mat_t, int>(nullptr, 0);
}

inline std::pair<mul_mat_t, int> mul_mat_kernel(int int_typeA, int nq, int k_step) {
    switch (k_step) {
        case  32: return mul_mat_kernel< 32>(int_typeA, nq);
        case  64: return mul_mat_kernel< 64>(int_typeA, nq);
        case 128: return mul_mat_kernel<128>(int_typeA, nq);
        default: GGML_ABORT("Fatal error");
    }
}
}

void iqk_gemm_legacy_fa(int D, int nq, int type_k, const char * k, size_t stride_k, DataInfo& info, int k_step) {
    auto [mul_mat, nrc_q] = mul_mat_kernel(type_k, nq, k_step);
    for (int iq = 0; iq < nq/nrc_q; ++iq) {
        mul_mat(D, k, stride_k, info, k_step);
        info.cur_y += nrc_q;
    }
    int iq = nrc_q*(nq/nrc_q);
    if (iq < nq) {
        auto [mul_mat1, nrc_q1] = mul_mat_kernel(type_k, nq - iq, k_step);
        GGML_ASSERT(nrc_q1 == nq - iq);
        mul_mat1(D, k, stride_k, info, k_step);
    }
}

#endif
