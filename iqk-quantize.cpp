#include "ggml-quants.h"
#include "ggml-impl.h"
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cmath>
#include <array>
#include <algorithm>
#include <cstring>
#include <mutex>

namespace {

inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

struct IQ1BNData {
    IQ1BNData();
    std::vector<std::pair<int16_t, bool>> map;
    std::vector<uint16_t> rmap;
};

const IQ1BNData& get_iq1bn_data() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static IQ1BNData iq1bn;
    return iq1bn;
}

IQ1BNData::IQ1BNData() {
    map.resize(1 << 16, {int16_t(-1), false});
    uint64_t aux64;
    uint8_t * aux8 = (uint8_t *)&aux64;
    std::vector<uint64_t> values;
    values.reserve(6561);
    rmap.reserve(6561);
    for (int i = 0; i < (1 << 16); ++i) {
        bool is_good = true;
        for (int j = 0; j < 8; ++j) {
            aux8[j] = (i >> 2*j) & 3;
            if (aux8[j] == 3u) { is_good = false; break; }
        }
        if (!is_good) continue;
        auto orig = aux64;
        for (int j = 0; j < 8; ++j) aux8[j] = 2 - aux8[j];
        int k = 0;
        for (; k < int(values.size()); ++k) {
            if (values[k] == aux64) break;
        }
        if (k < int(values.size())) {
            map[i] = {k, true};
        } else {
            map[i].first = values.size();
            values.push_back(orig);
            rmap.push_back(i);
        }
    }
    printf("==================== %s: initialized %d grid points\n", __func__, int(rmap.size()));
}

struct IQ1BNQuantizer {
    typedef union {
        float f;
        uint32_t i;
    } scale_t;
    constexpr static int block_size = QK_IQ1BN;
    int8_t L[QK_IQ1BN];
    void quantize_one_row(const float * src, block_iq1_bn * y, int n_per_row, const float * imatrix);
};

void IQ1BNQuantizer::quantize_one_row(const float * src, block_iq1_bn * y, int n_per_row, const float * imatrix) {

    (void)imatrix;

    constexpr int Nk = block_size/8;

    const int nblock = n_per_row/QK_IQ1BN;

    const auto& iq1bn = get_iq1bn_data();

    float max_in_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        float ax = fabsf(src[j]);
        max_in_row = std::max(max_in_row, ax);
    }

    max_in_row *= 1.03125f; // i.e., round to nearest in our fp8 representation
    scale_t s;
    uint8_t u = 0;
    if (max_in_row > 1.9074e-06f && max_in_row < 0.12109f) {
        s.f = max_in_row;
        u = ((((s.i >> 23) + 132) & 0xf) << 4) | ((s.i >> 19) & 0xf);
        s.i = ((((u >> 4) | 0xf0) - 132) << 23) | ((u & 0x0f) << 19);
    } else {
        // outside the allowed range. Small values we can habdle via quants set to zero, so we only warn about too large values
        if (max_in_row >= 0.12109f) {
            u = 255;
            fprintf(stderr, "%s: found scale %g, which is outside the range of out fp8 representation\n", __func__, max_in_row);
        } else{
            u = 0;
        }
    }

    for (int ib = 0; ib < nblock; ++ib) {
        std::memset(&y[ib], 0, sizeof(block_iq1_bn));
        auto xb = src + QK_IQ1BN*ib;
        for (int j = 0; j < QK_IQ1BN; ++j) {
            L[j] = fabsf(xb[j]) < 1e-6f ? 1 : xb[j] < 0 ? 0 : 2;
        }
        auto ql = y[ib].ql;
        auto qh = y[ib].qh;
        uint16_t extra = 0;
        for (int k = 0; k < Nk; ++k) {
            auto Lk = L + 8*k;
            uint16_t u = 0;
            for (int j = 0; j < 8; ++j) u |= (Lk[j] << 2*j);
            auto& val = iq1bn.map[u];
            GGML_ASSERT(val.first >= 0);
            ql[k] = val.first & 255;
            qh[k/2] |= (val.first >> 8) << 4*(k%2);
            if (val.second) extra |= (1 << k);
        }

        y[ib].extra = u | (extra << 8);

    }
}
}

void iq1bn_init_impl(void) {
    get_iq1bn_data();
}

size_t quantize_iq1_bn(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    IQ1BNQuantizer iq1bn;
    int nblock = n_per_row/QK_IQ1BN;
    block_iq1_bn * y = (block_iq1_bn *)dst;
    for (int row = 0; row < nrows; ++row) {
        iq1bn.quantize_one_row(src + row*n_per_row, y, n_per_row, imatrix);
        y += nblock;
    }
    return sizeof(block_iq1_bn)*nblock*nrows;
}

void quantize_row_iq1_bn_reference(const float * x, block_iq1_bn * y, int64_t k) {
    quantize_iq1_bn(x, y, 1, k, nullptr);
}

void quantize_row_iq1_bn(const float * x, void * y, int64_t k) {
    quantize_iq1_bn(x, y, 1, k, nullptr);
}

void dequantize_row_iq1_bn(const block_iq1_bn * x, float * y, int64_t k) {
    assert(k%QK_IQ1BN == 0);
    int nblock = k / QK_IQ1BN;

    IQ1BNQuantizer::scale_t s;

    for (int i = 0; i < nblock; ++i) {
        uint16_t u = x[i].extra & 0xff;
        s.i = ((((u >> 4) | 0xf0) - 132) << 23) | ((u & 0x0f) << 19);
        float d = s.f;
        uint8_t extra = x[i].extra >> 8;
        auto qh = x[i].qh;
        auto ql = x[i].ql;
        for (int k = 0; k < QK_IQ1BN/8; ++k) {
            uint16_t idx = ql[k] | ((qh[k/2] << (8 - 4*(k%2))) & 0x0f00);
            uint16_t val = iq1bn_grid_u16[idx];
            float dls = extra & (1 << k) ? -d : d;
            for (int j = 0; j < 8; ++j) y[j] = dls * (((val >> 2*j) & 3) - 1);
            y += 8;
        }
    }
}

#if __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
#endif

void ggml_vec_dot_iq1_bn_q8_0 (int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    static_assert(QK_IQ1BN == 64, "This dot product implementation for iq1_bn requires a block size of 64");

    const block_iq1_bn * x = (const block_iq1_bn *)vx;
    const block_q8_0   * y = (const block_q8_0 *)vy;
    int nblock = n / QK_IQ1BN;

    float sumf = 0;
    IQ1BNQuantizer::scale_t scale;

#if defined __AVX2__

    const auto m1_8   = _mm256_set1_epi8(1);
    const auto shuff1 = _mm256_set_epi64x(0x0808080808080808, 0x0000000000000000, 0x0808080808080808, 0x0000000000000000);
    const auto shuff2 = _mm256_add_epi8(shuff1, m1_8);
    const auto shuff3 = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    const auto shuff4 = _mm256_set_epi64x(0x0707070707070707, 0x0606060606060606, 0x0505050505050505, 0x0404040404040404);
    const auto mask1  = _mm256_set1_epi64x(0x8040201008040201);
#if !(defined __AVX512VNNI__ && defined __AVX512VL__)
    const auto m1_16  = _mm256_set1_epi16(1);
#endif

    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();

    // All scales are the same in BitNet!
    uint16_t u = x[0].extra & 0xff;
    scale.i = ((((u >> 4) | 0xf0) - 132) << 23) | ((u & 0x0f) << 19);

    for (int i = 0; i < nblock; ++i) {
        // We would uncomment this if we wanted to use this implementation for a model that has per block scales
        //uint16_t u = x[i].extra & 0xff;
        //scale.i = ((((u >> 4) | 0xf0) - 132) << 23) | ((u & 0x0f) << 19);
        auto signs = _mm256_set1_epi8(x[i].extra >> 8);
        // signs for groups of 8 ordered as 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, ...
        // To use these to sign the q8 values we need
        //  0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 amd the same for 4...7
        signs = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(signs, mask1), mask1), m1_8);
        auto q8_1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)y[2*i+0].qs), _mm256_shuffle_epi8(signs, shuff3));
        auto q8_2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)y[2*i+1].qs), _mm256_shuffle_epi8(signs, shuff4));

        auto ql = x[i].ql;
        auto qh = x[i].qh;
        auto aux1 = _mm256_set_epi64x(iq1bn_grid_xxx[ql[3] | ((qh[1] << 4) & 0x0f00)], iq1bn_grid_xxx[ql[2] | ((qh[1] << 8) & 0x0f00)],
                                      iq1bn_grid_xxx[ql[1] | ((qh[0] << 4) & 0x0f00)], iq1bn_grid_xxx[ql[0] | ((qh[0] << 8) & 0x0f00)]);
        auto aux2 = _mm256_set_epi64x(iq1bn_grid_xxx[ql[7] | ((qh[3] << 4) & 0x0f00)], iq1bn_grid_xxx[ql[6] | ((qh[3] << 8) & 0x0f00)],
                                      iq1bn_grid_xxx[ql[5] | ((qh[2] << 4) & 0x0f00)], iq1bn_grid_xxx[ql[4] | ((qh[2] << 8) & 0x0f00)]);

        auto v1_p = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(aux1, shuff1), mask1), mask1);
        auto v1_m = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(aux1, shuff2), mask1), mask1);
        auto v2_p = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(aux2, shuff1), mask1), mask1);
        auto v2_m = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(aux2, shuff2), mask1), mask1);

        auto dot1 = _mm256_sub_epi8(_mm256_sign_epi8(q8_1, v1_m), _mm256_sign_epi8(q8_1, v1_p));
        auto dot2 = _mm256_sub_epi8(_mm256_sign_epi8(q8_2, v2_m), _mm256_sign_epi8(q8_2, v2_p));

#if defined __AVX512VNNI__ && defined __AVX512VL__
        dot1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), m1_8, dot1);
        dot2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), m1_8, dot2);
#else
        dot1 = _mm256_madd_epi16(m1_16, _mm256_maddubs_epi16(m1_8, dot1));
        dot2 = _mm256_madd_epi16(m1_16, _mm256_maddubs_epi16(m1_8, dot2));
#endif

        // We would uncomment this if we wanted to use this implementation for a model that has per block scales
        //acc1 = _mm256_fmadd_ps(_mm256_set1_ps(scale.f*GGML_FP16_TO_FP32(y[2*i+0].d)), _mm256_cvtepi32_ps(dot1), acc1);
        //acc2 = _mm256_fmadd_ps(_mm256_set1_ps(scale.f*GGML_FP16_TO_FP32(y[2*i+1].d)), _mm256_cvtepi32_ps(dot2), acc2);
        // All scales are the same for BitNet!
        // This is slower
        //uint32_t aux32 = y[2*i+0].d | (y[2*i+1].d << 16);
        //auto d8 = _mm256_cvtph_ps(_mm_set1_epi32(aux32));
        //acc1 = _mm256_fmadd_ps(_mm256_permute_ps(d8, 0x00), _mm256_cvtepi32_ps(dot1), acc1);
        //acc2 = _mm256_fmadd_ps(_mm256_permute_ps(d8, 0x55), _mm256_cvtepi32_ps(dot2), acc2);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(GGML_FP16_TO_FP32(y[2*i+0].d)), _mm256_cvtepi32_ps(dot1), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(GGML_FP16_TO_FP32(y[2*i+1].d)), _mm256_cvtepi32_ps(dot2), acc2);

    }

    //sumf = hsum_float_8(_mm256_add_ps(acc1, acc2));
    sumf = scale.f * hsum_float_8(_mm256_add_ps(acc1, acc2));

#else

    for (int i = 0; i < nblock; ++i) {
        uint16_t u = x[i].extra & 0xff;
        scale.i = ((((u >> 4) | 0xf0) - 132) << 23) | ((u & 0x0f) << 19);
        uint8_t extra = x[i].extra >> 8;
        auto qh = x[i].qh;
        auto ql = x[i].ql;
        auto q8 = y[2*i+0].qs;
        int16_t sumi1 = 0;
        for (int k = 0; k < 4; ++k) {
            uint16_t idx = ql[k] | ((qh[k/2] << (8 - 4*(k%2))) & 0x0f00);
            uint16_t val = iq1bn_grid_u16[idx];
            int16_t sl = 0;
            for (int j = 0; j < 8; ++j) sl += q8[j] * (((val >> 2*j) & 3) - 1);
            sumi1 += extra & (1 << k) ? -sl : sl;
            q8 += 8;
        }
        q8 = y[2*i+1].qs;
        int16_t sumi2 = 0;
        for (int k = 4; k < 8; ++k) {
            uint16_t idx = ql[k] | ((qh[k/2] << (8 - 4*(k%2))) & 0x0f00);
            uint16_t val = iq1bn_grid_u16[idx];
            int16_t sl = 0;
            for (int j = 0; j < 8; ++j) sl += q8[j] * (((val >> 2*j) & 3) - 1);
            sumi2 += extra & (1 << k) ? -sl : sl;
            q8 += 8;
        }
        sumf += scale.f * (GGML_FP16_TO_FP32(y[2*i+0].d) * sumi1 + GGML_FP16_TO_FP32(y[2*i+1].d) * sumi2);
    }

#endif

    *s = sumf;

}

void ggml_vec_dot_iq1_bn_q8_K64(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    static_assert(QK_IQ1BN == 64, "This dot product implementation for iq1_bn requires a block size of 64");

    const block_iq1_bn * x = (const block_iq1_bn *)vx;
    const block_q8_K64 * y = (const block_q8_K64 *)vy;
    int nblock = n / QK_IQ1BN;

    float sumf = 0;
    IQ1BNQuantizer::scale_t scale;

#if defined __AVX2__

    const auto m1_8   = _mm256_set1_epi8(1);
    const auto shuff1 = _mm256_set_epi64x(0x0808080808080808, 0x0000000000000000, 0x0808080808080808, 0x0000000000000000);
    const auto shuff2 = _mm256_add_epi8(shuff1, m1_8);
    const auto shuff3 = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    const auto shuff4 = _mm256_set_epi64x(0x0707070707070707, 0x0606060606060606, 0x0505050505050505, 0x0404040404040404);
    const auto mask1  = _mm256_set1_epi64x(0x8040201008040201);
#if !(defined __AVX512VNNI__ && defined __AVX512VL__)
    const auto m1_16  = _mm256_set1_epi16(1);
#endif

    __m256 acc = _mm256_setzero_ps();

    // All scales are the same in BitNet!
    uint16_t u = x[0].extra & 0xff;
    scale.i = ((((u >> 4) | 0xf0) - 132) << 23) | ((u & 0x0f) << 19);

    for (int i = 0; i < nblock; ++i) {
        // We would uncomment this if we wanted to use this implementation for a model that has per block scales
        //uint16_t u = x[i].extra & 0xff;
        //scale.i = ((((u >> 4) | 0xf0) - 132) << 23) | ((u & 0x0f) << 19);
        auto signs = _mm256_set1_epi8(x[i].extra >> 8);
        // signs for groups of 8 ordered as 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, ...
        // To use these to sign the q8 values we need
        //  0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 amd the same for 4...7
        signs = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(signs, mask1), mask1), m1_8);
        auto q8_1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)y[i].qs+0), _mm256_shuffle_epi8(signs, shuff3));
        auto q8_2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)y[i].qs+1), _mm256_shuffle_epi8(signs, shuff4));

        auto ql = x[i].ql;
        auto qh = x[i].qh;
        auto aux1 = _mm256_set_epi64x(iq1bn_grid_xxx[ql[3] | ((qh[1] << 4) & 0x0f00)], iq1bn_grid_xxx[ql[2] | ((qh[1] << 8) & 0x0f00)],
                                      iq1bn_grid_xxx[ql[1] | ((qh[0] << 4) & 0x0f00)], iq1bn_grid_xxx[ql[0] | ((qh[0] << 8) & 0x0f00)]);
        auto aux2 = _mm256_set_epi64x(iq1bn_grid_xxx[ql[7] | ((qh[3] << 4) & 0x0f00)], iq1bn_grid_xxx[ql[6] | ((qh[3] << 8) & 0x0f00)],
                                      iq1bn_grid_xxx[ql[5] | ((qh[2] << 4) & 0x0f00)], iq1bn_grid_xxx[ql[4] | ((qh[2] << 8) & 0x0f00)]);

        auto v1_p = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(aux1, shuff1), mask1), mask1);
        auto v1_m = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(aux1, shuff2), mask1), mask1);
        auto v2_p = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(aux2, shuff1), mask1), mask1);
        auto v2_m = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(aux2, shuff2), mask1), mask1);

        auto dot1 = _mm256_sub_epi8(_mm256_sign_epi8(q8_1, v1_m), _mm256_sign_epi8(q8_1, v1_p));
        auto dot2 = _mm256_sub_epi8(_mm256_sign_epi8(q8_2, v2_m), _mm256_sign_epi8(q8_2, v2_p));

#if defined __AVX512VNNI__ && defined __AVX512VL__
        dot1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), m1_8, dot1);
        dot2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), m1_8, dot2);
#else
        dot1 = _mm256_madd_epi16(m1_16, _mm256_maddubs_epi16(m1_8, dot1));
        dot2 = _mm256_madd_epi16(m1_16, _mm256_maddubs_epi16(m1_8, dot2));
#endif

        // We would uncomment this if we wanted to use this implementation for a model that has per block scales
        //acc1 = _mm256_fmadd_ps(_mm256_set1_ps(scale.f*GGML_FP16_TO_FP32(y[2*i+0].d)), _mm256_cvtepi32_ps(dot1), acc1);
        //acc2 = _mm256_fmadd_ps(_mm256_set1_ps(scale.f*GGML_FP16_TO_FP32(y[2*i+1].d)), _mm256_cvtepi32_ps(dot2), acc2);
        // All scales are the same for BitNet!
        // This is slower
        //uint32_t aux32 = y[2*i+0].d | (y[2*i+1].d << 16);
        //auto d8 = _mm256_cvtph_ps(_mm_set1_epi32(aux32));
        //acc1 = _mm256_fmadd_ps(_mm256_permute_ps(d8, 0x00), _mm256_cvtepi32_ps(dot1), acc1);
        //acc2 = _mm256_fmadd_ps(_mm256_permute_ps(d8, 0x55), _mm256_cvtepi32_ps(dot2), acc2);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(y[i].d), _mm256_cvtepi32_ps(_mm256_add_epi32(dot1, dot2)), acc);

    }

    sumf = scale.f * hsum_float_8(acc);

#else

    uint16_t u = x[0].extra & 0xff;
    scale.i = ((((u >> 4) | 0xf0) - 132) << 23) | ((u & 0x0f) << 19);
    for (int i = 0; i < nblock; ++i) {
        uint8_t extra = x[i].extra >> 8;
        auto qh = x[i].qh;
        auto ql = x[i].ql;
        auto q8 = y[i].qs;
        int sumi = 0;
        for (int k = 0; k < 4; ++k) {
            uint16_t idx = ql[k] | ((qh[k/2] << (8 - 4*(k%2))) & 0x0f00);
            uint16_t val = iq1bn_grid_u16[idx];
            int16_t sl = 0;
            for (int j = 0; j < 8; ++j) sl += q8[j] * (((val >> 2*j) & 3) - 1);
            sumi += extra & (1 << k) ? -sl : sl;
            q8 += 8;
        }
        for (int k = 4; k < 8; ++k) {
            uint16_t idx = ql[k] | ((qh[k/2] << (8 - 4*(k%2))) & 0x0f00);
            uint16_t val = iq1bn_grid_u16[idx];
            int16_t sl = 0;
            for (int j = 0; j < 8; ++j) sl += q8[j] * (((val >> 2*j) & 3) - 1);
            sumi += extra & (1 << k) ? -sl : sl;
            q8 += 8;
        }
        sumf += scale.f * (y[i].d) * sumi;
    }

#endif

    *s = sumf;

}
