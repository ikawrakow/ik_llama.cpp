#include "iqk_gemm_iqk_quants.h"

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

namespace {

#ifdef HAVE_FANCY_SIMD

__m512i inline load_iq4nl_values_512() {
    auto val256 = load_iq4nl_values_256();
    return _mm512_inserti32x8(_mm512_castsi256_si512(val256), val256, 1);
}

struct IQXKScales {
    IQXKScales(uint8_t shift, int8_t min_val) : eshift(_mm256_set1_epi16(shift)), min(_mm256_set1_epi16(min_val)) {}
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m128i scales8, const Q8& q8, __m256 * accm, __m512i * scales) const {
        auto scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales8, scale_shuffle));
        scales16 = _mm256_mullo_epi16(scales16, _mm256_mask_add_epi16(min, extra, min, eshift));
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i prod  = _mm256_madd_epi16(scales16, q8.load_bsums(iy, i));
            accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accm[iy]);
        }
        scales16 = MM256_SET_M128I(scales8, scales8);
        scales[0] = _mm512_cvtepi8_epi16(_mm256_shuffle_epi8(scales16, shuffle1));
        scales[1] = _mm512_cvtepi8_epi16(_mm256_shuffle_epi8(scales16, shuffle2));
    }
    const __m256i eshift;
    const __m256i min;
    const __m128i scale_shuffle = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
    const __m128i emask         = _mm_set_epi32(0x80804040, 0x20201010, 0x08080404, 0x02020101);
    const __m128i eshuffle      = _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
    const __m256i shuffle1      = _mm256_set_epi64x(0x0b0b0b0b09090909, 0x0303030301010101, 0x0a0a0a0a08080808, 0x0202020200000000);
    const __m256i shuffle2      = _mm256_set_epi64x(0x0f0f0f0f0d0d0d0d, 0x0707070705050505, 0x0e0e0e0e0c0c0c0c, 0x0606060604040404);
};

struct IQXKScales2 {
    IQXKScales2(uint8_t shift, int8_t min_val) : eshift(_mm256_set1_epi16(shift)), min(_mm256_set1_epi16(min_val)) {}
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m128i scales8, const Q8& q8, __m256 * accm, __m512i * scales) const {
        process(i, d, extra, _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales8, scale_shuffle)), q8, accm, scales);
    }
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m256i scales16, const Q8& q8, __m256 * accm, __m512i * scales) const {
        auto scales_s = _mm256_mullo_epi16(scales16, _mm256_mask_add_epi16(min, extra, min, eshift));
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i prod  = _mm256_madd_epi16(scales_s, q8.load_bsums(iy, i));
            accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accm[iy]);
        }
        auto aux_1 = MM256_SET_M128I(_mm256_castsi256_si128(scales16), _mm256_castsi256_si128(scales16));
        auto aux_2 = MM256_SET_M128I(_mm256_extracti128_si256(scales16, 1), _mm256_extracti128_si256(scales16, 1));
        auto scales16_1 = _mm512_inserti32x8(_mm512_castsi256_si512(aux_1), aux_1, 1);
        auto scales16_2 = _mm512_inserti32x8(_mm512_castsi256_si512(aux_2), aux_2, 1);
        scales[0] = _mm512_shuffle_epi8(scales16_1, shuffles[0]);
        scales[1] = _mm512_shuffle_epi8(scales16_1, shuffles[1]);
        scales[2] = _mm512_shuffle_epi8(scales16_2, shuffles[0]);
        scales[3] = _mm512_shuffle_epi8(scales16_2, shuffles[1]);
    }
    const __m256i eshift;
    const __m256i min;
    const __m128i scale_shuffle = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
    const __m128i emask         = _mm_set_epi32(0x80804040, 0x20201010, 0x08080404, 0x02020101);
    const __m128i eshuffle      = _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
    const __m512i shuffles[2] = {
        _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_setzero_si512(),
                            _mm_set1_epi16(0x0100), 0), _mm_set1_epi16(0x0302), 1), _mm_set1_epi16(0x0504), 2), _mm_set1_epi16(0x0706), 3),
        _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_setzero_si512(),
                            _mm_set1_epi16(0x0908), 0), _mm_set1_epi16(0x0b0a), 1), _mm_set1_epi16(0x0d0c), 2), _mm_set1_epi16(0x0f0e), 3)
    };
};

struct DequantizerIQ2KS final : public BaseDequantizer<block_iq2_ks, true, true> {
    DequantizerIQ2KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
    template <typename Q8>
    inline void compute_block(int i, const Q8& q8, __m512 * acc) {
        prepare(x[i].qs);
        auto scales128 = make_scales(x[i].scales, x[i].extra >> 8);
        auto shifts = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi8(x[i].extra), hmask), hmask), m5);
        auto mins128 = _mm_mullo_epi16(scales128, _mm_cvtepi8_epi16(_mm_add_epi8(m32, shifts)));
        auto mins = MM256_SET_M128I(_mm_shuffle_epi8(mins128, s8k.shuffles[1]), _mm_shuffle_epi8(mins128, s8k.shuffles[0]));
        auto scales256 = MM256_SET_M128I(scales128, scales128);
        auto all_scales = _mm512_inserti32x8(_mm512_castsi256_si512(scales256), scales256, 1);
        __m512i scales[4];
        for (int k = 0; k < 4; ++k) scales[k] = _mm512_shuffle_epi8(all_scales, shuffles[k]);
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            auto q8s = q8.load_bsums(iy, i);
            auto prod = _mm256_madd_epi16(mins, q8s);
            auto sumi = _mm512_inserti32x8(_mm512_setzero_si512(), prod, 0);
            for (int k = 0; k < 4; ++k) {
                auto p = _mm512_maddubs_epi16(bits.values[k], q8.load_quants64(iy, i, k));
                sumi = _mm512_dpwssd_epi32(sumi, p, scales[k]);
            }
            acc[iy] = _mm512_fmadd_ps(_mm512_set1_ps(d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi), acc[iy]);
        }
    }
    inline void prepare(const uint8_t * q2) {
        bits.prepare(q2);
        bits.values[0] = _mm512_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm512_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm512_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm512_shuffle_epi8(values, bits.values[3]);
    }
    static inline __m512i load_values() {
        static const uint8_t kvalues_iq2nl[16] = {1, 19, 33, 49, 0, 0, 0, 0,  6, 24, 38, 54, 0, 0, 0, 0};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq2nl);
        auto val256 = MM256_SET_M128I(val128, val128);
        return _mm512_inserti32x8(_mm512_castsi256_si512(val256), val256, 1);
    }
    inline __m128i make_scales(const uint8_t * scales_l, uint8_t scales_h) const {
        const uint16_t * scales = (const uint16_t *)scales_l;
        uint32_t aux32 = scales[0] | (uint32_t(scales[1]) << 16);
        auto scl = _mm_srlv_epi32(_mm_set1_epi32(aux32), shift);
        scl = _mm_and_si128(_mm_shuffle_epi8(scl, shuffle), _mm_set1_epi8(0xf));
        auto sch = _mm_set1_epi8(scales_h);
        sch = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(sch, hmask), _mm_setzero_si128()), m16);
        return _mm_cvtepi8_epi16(_mm_add_epi8(scl, sch));
    }
    Q2Bits bits;
    Scales8KBase s8k;

    const __m512i values;
    const __m128i m16 = _mm_set1_epi8(-16);
    const __m128i m5  = _mm_set1_epi8(5);
    const __m128i m32 = _mm_set1_epi8(-32);
    const __m128i hmask = _mm_set1_epi64x(0x8040201008040201);
    const __m128i shuffle = _mm_set1_epi64x(0x0703060205010400);
    const __m128i shift = _mm_set_epi32(0, 0, 4, 0);
    const __m512i shuffles[4] = {
        _mm512_inserti32x8(_mm512_set1_epi16(0x0100), _mm256_set1_epi16(0x0302), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0504), _mm256_set1_epi16(0x0706), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0908), _mm256_set1_epi16(0x0b0a), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0d0c), _mm256_set1_epi16(0x0f0e), 1),
    };
};

struct DequantizerIQ2K final : public BaseDequantizer<block_iq2_k> {
    DequantizerIQ2K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(IQXKScales(5, -32)), values(load_values()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        prepare(x[i].qs);
        iqxk.process(i, d, x[i].extra, make_scales(x[i].scales), q8, accm, scales);
    }
    inline void prepare(const uint8_t * q2) {
        bits.prepare(q2);
        bits.values[0] = _mm512_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm512_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm512_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm512_shuffle_epi8(values, bits.values[3]);
    }
    static inline __m512i load_values() {
        static const uint8_t kvalues_iq2nl[16] = {1, 19, 33, 49, 0, 0, 0, 0,  6, 24, 38, 54, 0, 0, 0, 0};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq2nl);
        auto val256 = MM256_SET_M128I(val128, val128);
        return _mm512_inserti32x8(_mm512_castsi256_si512(val256), val256, 1);
    }
    inline __m128i make_scales(const uint8_t * scales_l) const {
        uint64_t aux64; std::memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
        return _mm_add_epi8(scl, m8);
    }
    Q2Bits bits;
    const IQXKScales iqxk;

    const __m512i values;
    const __m128i m8 = _mm_set1_epi8(-8);
};

struct DequantizerIQ3K final : public BaseDequantizer<block_iq3_k> {
    DequantizerIQ3K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(4, -64), values(load_values()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        prepare(x[i].qs, x[i].qh);
        iqxk.process(i, d, x[i].extra, make_scales(x[i].scales_h, x[i].scales_l), q8, accm, scales);
    }
    inline void prepare(const uint8_t * q2, const uint8_t * qh) {
        bits.prepare(q2);
        auto h256 = _mm256_loadu_si256((const __m256i *)qh);
        auto hbits = _mm512_inserti32x8(_mm512_castsi256_si512(h256), _mm256_srli_epi16(h256, 1), 1);
        bits.values[0] = _mm512_or_si512(bits.values[0], _mm512_and_si512(_mm512_slli_epi16(hbits, 2), hmask));
        bits.values[1] = _mm512_or_si512(bits.values[1], _mm512_and_si512(hbits, hmask));
        bits.values[2] = _mm512_or_si512(bits.values[2], _mm512_and_si512(_mm512_srli_epi16(hbits, 2), hmask));
        bits.values[3] = _mm512_or_si512(bits.values[3], _mm512_and_si512(_mm512_srli_epi16(hbits, 4), hmask));
        bits.values[0] = _mm512_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm512_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm512_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm512_shuffle_epi8(values, bits.values[3]);
    }
    static inline __m512i load_values() {
        static const uint8_t kvalues_iq3nl[16] = {1, 24, 41, 54, 65, 77, 92, 111, 5, 28, 45, 58, 69, 81, 96, 115};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq3nl);
        auto val256 = MM256_SET_M128I(val128, val128);
        return _mm512_inserti32x8(_mm512_castsi256_si512(val256), val256, 1);
    }
    inline __m128i make_scales(uint16_t signs, const uint8_t * scales_l) const {
        uint64_t aux64; std::memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
        scl = _mm_add_epi8(_mm_slli_epi16(scl, 1), m1);
        const __m128i sc_signs = _mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi16(signs), sign_mask), sign_mask);
        const __m128i sch = _mm_shuffle_epi8(_mm_or_si128(sc_signs, _mm_set1_epi8(1)), hshuff);
        return _mm_sign_epi8(scl, sch);
    }
    Q2Bits bits;
    const IQXKScales2 iqxk;

    const __m512i values;
    const __m512i hmask = _mm512_set1_epi8(4);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m128i sign_mask = _mm_set_epi64x(0x8080404020201010, 0x0808040402020101);
    const __m128i hshuff = _mm_loadu_si128((const __m128i*)k_shuff);
    constexpr static uint8_t k_shuff[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
};

struct DequantizerIQ4KSS final : public BaseDequantizer<block_iq4_kss, true> {
    DequantizerIQ4KSS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_iq4nl_values_512()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        uint32_t aux32[2];
        auto b1 = _mm512_loadu_si512((const __m512i *)x[i].qs + 0);
        auto b2 = _mm512_loadu_si512((const __m512i *)x[i].qs + 1);
        auto bs1 = _mm512_and_si512(b1, mask15);
        bs1 = _mm512_xor_si512(bs1, _mm512_srli_epi16(bs1, 1));
        auto bs2 = _mm512_and_si512(b2, mask15);
        bs2 = _mm512_xor_si512(bs2, _mm512_srli_epi16(bs2, 1));
        bits.values[0] = _mm512_and_si512(bs1, bits.ml);
        bits.values[1] = _mm512_and_si512(_mm512_srli_epi16(bs1, 4), bits.ml);
        bits.values[2] = _mm512_and_si512(bs2, bits.ml);
        bits.values[3] = _mm512_and_si512(_mm512_srli_epi16(bs2, 4), bits.ml);
        auto tmp = _mm512_permutex2var_epi64(bits.values[0], permute1, bits.values[1]);
        bits.values[1] = _mm512_shuffle_epi8(values, _mm512_permutex2var_epi64(bits.values[0], permute2, bits.values[1]));
        bits.values[0] = _mm512_shuffle_epi8(values, tmp);
        tmp = _mm512_permutex2var_epi64(bits.values[2], permute1, bits.values[3]);
        bits.values[3] = _mm512_shuffle_epi8(values, _mm512_permutex2var_epi64(bits.values[2], permute2, bits.values[3]));
        bits.values[2] = _mm512_shuffle_epi8(values, tmp);
        //
        // Now the more difficult part - prepare the scales
        //
        aux32[0] = _mm512_cmpeq_epi16_mask(_mm512_and_si512(b1, mask1), mask1);
        aux32[1] = _mm512_cmpeq_epi16_mask(_mm512_and_si512(b2, mask1), mask1);

        auto scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)aux32));
        auto m1 = _mm512_castsi512_si128(mask1);
        auto shifts = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(scales128, m1), m1), m4);
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, mask), m127);
        auto scales_s = _mm_mullo_epi16(scales128, _mm_add_epi16(m128, shifts));
        s8k.accum_mins(scales_s, q8, i, d, accm);
        auto scales256 = MM256_SET_M128I(scales128, scales128);
        auto all_scales = _mm512_inserti32x8(_mm512_castsi256_si512(scales256), scales256, 1);
        scales[0] = _mm512_shuffle_epi8(all_scales, shuffles[0]);
        scales[1] = _mm512_shuffle_epi8(all_scales, shuffles[1]);
        scales[2] = _mm512_shuffle_epi8(all_scales, shuffles[2]);
        scales[3] = _mm512_shuffle_epi8(all_scales, shuffles[3]);
    }

    Q4Bits bits;
    Scales8KBase s8k;
    const __m512i values;
    const __m512i mask15   = _mm512_set1_epi16(-2); // value is 0xfffe, but to shut up the stupid compiler warning we use the signed value
    const __m512i mask1    = _mm512_set1_epi16(1);
    const __m512i permute1 = _mm512_set_epi64(11, 10, 3, 2,  9,  8, 1, 0);
    const __m512i permute2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    const __m128i mask     = _mm_set1_epi16(254);
    const __m128i m127     = _mm_set1_epi16(-127);
    const __m128i m128     = _mm_set1_epi16(-128);
    const __m128i m4       = _mm_set1_epi16(4);
    const __m512i shuffles[4] = {
        _mm512_inserti32x8(_mm512_set1_epi16(0x0100), _mm256_set1_epi16(0x0302), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0504), _mm256_set1_epi16(0x0706), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0908), _mm256_set1_epi16(0x0b0a), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0d0c), _mm256_set1_epi16(0x0f0e), 1),
    };
};

struct DequantizerIQ4KS final : public BaseDequantizer<block_iq4_ks, true> {
    DequantizerIQ4KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_iq4nl_values_512()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        auto scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
        auto shifts = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(scales128, m1), m1), m4);
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, mask), m127);
        auto scales_s = _mm_mullo_epi16(scales128, _mm_add_epi16(m128, shifts));
        s8k.accum_mins(scales_s, q8, i, d, accm);
        auto scales256 = MM256_SET_M128I(scales128, scales128);
        auto all_scales = _mm512_inserti32x8(_mm512_castsi256_si512(scales256), scales256, 1);
        scales[0] = _mm512_shuffle_epi8(all_scales, shuffles[0]);
        scales[1] = _mm512_shuffle_epi8(all_scales, shuffles[1]);
        scales[2] = _mm512_shuffle_epi8(all_scales, shuffles[2]);
        scales[3] = _mm512_shuffle_epi8(all_scales, shuffles[3]);
        prepare(x[i].qs);
    }
    template <typename Q8>
    inline void compute_block(int i, const Q8& q8, __m512 * acc) {
        auto scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
        auto shifts = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(scales128, m1), m1), m4);
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, mask), m127);
        auto mins128 = _mm_mullo_epi16(scales128, _mm_add_epi16(m128, shifts));
        auto mins = MM256_SET_M128I(_mm_shuffle_epi8(mins128, s8k.shuffles[1]), _mm_shuffle_epi8(mins128, s8k.shuffles[0]));
        auto scales256 = MM256_SET_M128I(scales128, scales128);
        auto all_scales = _mm512_inserti32x8(_mm512_castsi256_si512(scales256), scales256, 1);
        __m512i scales[4];
        for (int k = 0; k < 4; ++k) scales[k] = _mm512_shuffle_epi8(all_scales, shuffles[k]);
        prepare(x[i].qs);
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            auto q8s = q8.load_bsums(iy, i);
            auto prod = _mm256_madd_epi16(mins, q8s);
            auto sumi = _mm512_inserti32x8(_mm512_setzero_si512(), prod, 0);
            for (int k = 0; k < 4; ++k) {
                auto p = _mm512_maddubs_epi16(bits.values[k], q8.load_quants64(iy, i, k));
                sumi = _mm512_dpwssd_epi32(sumi, p, scales[k]);
            }
            acc[iy] = _mm512_fmadd_ps(_mm512_set1_ps(d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi), acc[iy]);
        }
    }
    inline void prepare(const uint8_t * q4) {
        bits.prepare64(q4);
        // We now have in bits.valuse[0]: 0...15, 32...47, 64...79, 96...111
        //                bits.valuse[1]: 16..31, 48...63, 80...95, 112..127
        //                etc.
        auto tmp = _mm512_permutex2var_epi64(bits.values[0], permute1, bits.values[1]);
        bits.values[1] = _mm512_shuffle_epi8(values, _mm512_permutex2var_epi64(bits.values[0], permute2, bits.values[1]));
        bits.values[0] = _mm512_shuffle_epi8(values, tmp);
        tmp = _mm512_permutex2var_epi64(bits.values[2], permute1, bits.values[3]);
        bits.values[3] = _mm512_shuffle_epi8(values, _mm512_permutex2var_epi64(bits.values[2], permute2, bits.values[3]));
        bits.values[2] = _mm512_shuffle_epi8(values, tmp);
    }

    Q4Bits bits;
    Scales8KBase s8k;
    const __m512i values;
    const __m512i permute1 = _mm512_set_epi64(11, 10, 3, 2,  9,  8, 1, 0);
    const __m512i permute2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    const __m128i mask     = _mm_set1_epi16(254);
    const __m128i m127     = _mm_set1_epi16(-127);
    const __m128i m128     = _mm_set1_epi16(-128);
    const __m128i m1       = _mm_set1_epi16(1);
    const __m128i m4       = _mm_set1_epi16(4);
    const __m512i shuffles[4] = {
        _mm512_inserti32x8(_mm512_set1_epi16(0x0100), _mm256_set1_epi16(0x0302), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0504), _mm256_set1_epi16(0x0706), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0908), _mm256_set1_epi16(0x0b0a), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0d0c), _mm256_set1_epi16(0x0f0e), 1),
    };
};

struct DequantizerIQ4K final : public BaseDequantizer<block_iq4_k> {
    DequantizerIQ4K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(4, -128), values(load_iq4nl_values_512()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        prepare(x[i].qs);
        iqxk.process(i, d, x[i].extra, make_scales(x[i].scales_l, (const uint16_t *)x[i].scales_h), q8, accm, scales);
    }
    inline void prepare(const uint8_t * q4) {
        bits.prepare64(q4);
        // We now have in bits.valuse[0]: 0...15, 32...47, 64...79, 96...111
        //                bits.valuse[1]: 16..31, 48...63, 80...95, 112..127
        //                etc.
        auto tmp = _mm512_permutex2var_epi64(bits.values[0], permute1, bits.values[1]);
        bits.values[1] = _mm512_shuffle_epi8(values, _mm512_permutex2var_epi64(bits.values[0], permute2, bits.values[1]));
        bits.values[0] = _mm512_shuffle_epi8(values, tmp);
        tmp = _mm512_permutex2var_epi64(bits.values[2], permute1, bits.values[3]);
        bits.values[3] = _mm512_shuffle_epi8(values, _mm512_permutex2var_epi64(bits.values[2], permute2, bits.values[3]));
        bits.values[2] = _mm512_shuffle_epi8(values, tmp);
    }
    __m128i make_scales(const uint8_t * scales_l, const uint16_t * scales_h) const {
        uint64_t aux64;
        memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), maskl);
        const uint32_t aux32 = scales_h[0] | (scales_h[1] << 16);
        auto aux = _mm_and_si128(_mm_set_epi32(aux32 >> 2, aux32, aux32 << 2, aux32 << 4), maskh);
        auto sch = _mm_shuffle_epi8(aux, iqxk.scale_shuffle);
        return _mm_add_epi8(_mm_or_si128(scl, sch), m32);
    }

    Q4Bits bits;
    const IQXKScales2 iqxk;
    const __m512i values;
    const __m512i permute1 = _mm512_set_epi64(11, 10, 3, 2,  9,  8, 1, 0);
    const __m512i permute2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    const __m128i maskl    = _mm_set1_epi8(0xf);
    const __m128i maskh    = _mm_set1_epi8(0x30);
    const __m128i m32      = _mm_set1_epi8(-32);
};

struct DequantizerIQ5KS final : public BaseDequantizer<block_iq5_ks, true> {
    DequantizerIQ5KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(values); }
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        auto scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
        auto shifts = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(scales128, m1), m1), m2);
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, mask), m127);
        auto scales_s = _mm_mullo_epi16(scales128, _mm_add_epi16(m128, shifts));
        s8k.accum_mins(scales_s, q8, i, d, accm);
        auto scales256 = MM256_SET_M128I(scales128, scales128);
        auto all_scales = _mm512_inserti32x8(_mm512_castsi256_si512(scales256), scales256, 1);
        scales[0] = _mm512_shuffle_epi8(all_scales, shuffles[0]);
        scales[1] = _mm512_shuffle_epi8(all_scales, shuffles[1]);
        scales[2] = _mm512_shuffle_epi8(all_scales, shuffles[2]);
        scales[3] = _mm512_shuffle_epi8(all_scales, shuffles[3]);
        prepare(x[i].qs, x[i].qh);
    }
    template <typename Q8>
    inline void compute_block(int i, const Q8& q8, __m512 * acc) {
        auto scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
        auto shifts = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(scales128, m1), m1), m2);
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, mask), m127);
        auto mins128 = _mm_mullo_epi16(scales128, _mm_add_epi16(m128, shifts));
        auto mins = MM256_SET_M128I(_mm_shuffle_epi8(mins128, s8k.shuffles[1]), _mm_shuffle_epi8(mins128, s8k.shuffles[0]));
        auto scales256 = MM256_SET_M128I(scales128, scales128);
        auto all_scales = _mm512_inserti32x8(_mm512_castsi256_si512(scales256), scales256, 1);
        __m512i scales[4];
        for (int k = 0; k < 4; ++k) scales[k] = _mm512_shuffle_epi8(all_scales, shuffles[k]);
        prepare(x[i].qs, x[i].qh);
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            auto q8s = q8.load_bsums(iy, i);
            auto prod = _mm256_madd_epi16(mins, q8s);
            auto sumi = _mm512_inserti32x8(_mm512_setzero_si512(), prod, 0);
            for (int k = 0; k < 4; ++k) {
                auto p = _mm512_maddubs_epi16(bits.values[k], q8.load_quants64(iy, i, k));
                sumi = _mm512_dpwssd_epi32(sumi, p, scales[k]);
            }
            acc[iy] = _mm512_fmadd_ps(_mm512_set1_ps(d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi), acc[iy]);
        }
    }
    inline void prepare(const uint8_t * q4, const uint8_t * qh) {
        bits.prepare64a(q4);
        auto h256 = _mm256_loadu_si256((const __m256i *)qh);
        auto hbits = _mm512_inserti32x8(_mm512_castsi256_si512(h256), _mm256_srli_epi16(h256, 1), 1);
        auto m1 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(hbits, hmask1), hmask1);
        auto m2 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(hbits, hmask2), hmask2);
        bits.values[0] = _mm512_mask_shuffle_epi8(_mm512_maskz_shuffle_epi8(_knot_mask64(m1), values[0], bits.values[0]), m1, values[1], bits.values[0]);
        bits.values[1] = _mm512_mask_shuffle_epi8(_mm512_maskz_shuffle_epi8(_knot_mask64(m2), values[0], bits.values[1]), m2, values[1], bits.values[1]);
        hbits = _mm512_srli_epi16(hbits, 4);
        m1 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(hbits, hmask1), hmask1);
        m2 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(hbits, hmask2), hmask2);
        bits.values[2] = _mm512_mask_shuffle_epi8(_mm512_maskz_shuffle_epi8(_knot_mask64(m1), values[0], bits.values[2]), m1, values[1], bits.values[2]);
        bits.values[3] = _mm512_mask_shuffle_epi8(_mm512_maskz_shuffle_epi8(_knot_mask64(m2), values[0], bits.values[3]), m2, values[1], bits.values[3]);
    }
    static void load_values(__m512i * values) {
        static const uint8_t kvalues_iq5nl[32] = {
            2,  14,  25,  36,  45,  54,  63,  71,  78,  85,  92,  98, 104, 110, 116, 122, 127,
            133, 139, 145, 151, 157, 164, 171, 179, 187, 196, 205, 215, 225, 237, 249,
        };
        auto values128_1 = _mm_loadu_si128((const __m128i *)kvalues_iq5nl + 0);
        auto values128_2 = _mm_loadu_si128((const __m128i *)kvalues_iq5nl + 1);
        auto values256_1 = MM256_SET_M128I(values128_1, values128_1);
        auto values256_2 = MM256_SET_M128I(values128_2, values128_2);
        values[0] = _mm512_inserti32x8(_mm512_castsi256_si512(values256_1), values256_1, 1);
        values[1] = _mm512_inserti32x8(_mm512_castsi256_si512(values256_2), values256_2, 1);
    }

    Q4Bits bits;
    Scales8KBase s8k;
    __m512i values[2];
    const __m512i hmask1   = _mm512_set1_epi8(1);
    const __m512i hmask2   = _mm512_set1_epi8(4);
    const __m128i m127     = _mm_set1_epi16(-127);
    const __m128i m128     = _mm_set1_epi16(-128);
    const __m128i mask     = _mm_set1_epi16(254);
    const __m128i m1       = _mm_set1_epi16(1);
    const __m128i m2       = _mm_set1_epi16(2);
    const __m512i shuffles[4] = {
        _mm512_inserti32x8(_mm512_set1_epi16(0x0100), _mm256_set1_epi16(0x0302), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0504), _mm256_set1_epi16(0x0706), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0908), _mm256_set1_epi16(0x0b0a), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0d0c), _mm256_set1_epi16(0x0f0e), 1),
    };
};

struct DequantizerIQ5K final : public BaseDequantizer<block_iq5_k> {
    DequantizerIQ5K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(2, -128) { load_values(values); }
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        prepare(x[i].qs, x[i].qh);
        iqxk.process(i, d, x[i].extra, make_scales(x[i].scales_l, (const uint16_t *)x[i].scales_h), q8, accm, scales);
    }
    inline void prepare(const uint8_t * q4, const uint8_t * qh) {
        bits.prepare64(q4);
        auto h256 = _mm256_loadu_si256((const __m256i *)qh);
        auto hbits = _mm512_inserti32x8(_mm512_castsi256_si512(h256), _mm256_srli_epi16(h256, 2), 1);
        auto m1 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(hbits, hmask1), hmask1);
        auto m2 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(hbits, hmask2), hmask2);
        bits.values[0] = _mm512_mask_shuffle_epi8(_mm512_maskz_shuffle_epi8(_knot_mask64(m1), values[0], bits.values[0]), m1, values[1], bits.values[0]);
        bits.values[1] = _mm512_mask_shuffle_epi8(_mm512_maskz_shuffle_epi8(_knot_mask64(m2), values[0], bits.values[1]), m2, values[1], bits.values[1]);
        hbits = _mm512_srli_epi16(hbits, 4);
        m1 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(hbits, hmask1), hmask1);
        m2 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(hbits, hmask2), hmask2);
        bits.values[2] = _mm512_mask_shuffle_epi8(_mm512_maskz_shuffle_epi8(_knot_mask64(m1), values[0], bits.values[2]), m1, values[1], bits.values[2]);
        bits.values[3] = _mm512_mask_shuffle_epi8(_mm512_maskz_shuffle_epi8(_knot_mask64(m2), values[0], bits.values[3]), m2, values[1], bits.values[3]);
        // We now have in bits.valuse[0]: 0...31, 64...95
        //                bits.valuse[1]: 32..63, 96..127
        //                etc.
        auto tmp = _mm512_permutex2var_epi64(bits.values[0], permute1, bits.values[1]);
        bits.values[1] = _mm512_permutex2var_epi64(bits.values[0], permute2, bits.values[1]);
        bits.values[0] = tmp;
        tmp = _mm512_permutex2var_epi64(bits.values[2], permute1, bits.values[3]);
        bits.values[3] = _mm512_permutex2var_epi64(bits.values[2], permute2, bits.values[3]);
        bits.values[2] = tmp;
    }
    __m128i make_scales(const uint8_t * scales_l, const uint16_t * scales_h) const {
        uint64_t aux64;
        memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), maskl);
        const uint32_t aux32 = scales_h[0] | (scales_h[1] << 16);
        auto aux = _mm_and_si128(_mm_set_epi32(aux32 >> 2, aux32, aux32 << 2, aux32 << 4), maskh);
        auto sch = _mm_shuffle_epi8(aux, iqxk.scale_shuffle);
        return _mm_add_epi8(_mm_or_si128(scl, sch), m32);
    }
    static void load_values(__m512i * values) {
        static const uint8_t kvalues_iq5nl[32] = {
            2,  14,  25,  36,  45,  54,  63,  71,  78,  85,  92,  98, 104, 110, 116, 122, 127,
            133, 139, 145, 151, 157, 164, 171, 179, 187, 196, 205, 215, 225, 237, 249,
        };
        auto values128_1 = _mm_loadu_si128((const __m128i *)kvalues_iq5nl + 0);
        auto values128_2 = _mm_loadu_si128((const __m128i *)kvalues_iq5nl + 1);
        auto values256_1 = MM256_SET_M128I(values128_1, values128_1);
        auto values256_2 = MM256_SET_M128I(values128_2, values128_2);
        values[0] = _mm512_inserti32x8(_mm512_castsi256_si512(values256_1), values256_1, 1);
        values[1] = _mm512_inserti32x8(_mm512_castsi256_si512(values256_2), values256_2, 1);
    }

    Q4Bits bits;
    const IQXKScales2 iqxk;
    __m512i values[2];
    const __m512i hmask1   = _mm512_set1_epi8(1);
    const __m512i hmask2   = _mm512_set1_epi8(2);
    const __m512i permute1 = _mm512_set_epi64(11, 10,  9,  8, 3, 2, 1, 0);
    const __m512i permute2 = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
    const __m128i maskl    = _mm_set1_epi8(0xf);
    const __m128i maskh    = _mm_set1_epi8(0x30);
    const __m128i m32      = _mm_set1_epi8(-32);
};

struct DequantizerIQ6K final : public BaseDequantizer<block_iq6_k> {
    DequantizerIQ6K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(1, -128) { load_values(values); }
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        prepare(x[i].qs, x[i].qh);
        auto scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
        iqxk.process(i, d, x[i].extra, _mm256_cvtepi8_epi16(scales8), q8, accm, scales);
    }
    inline __m512i make_one(__m512i l, __m512i h) const {
        auto p = _mm512_shuffle_epi8(values[0], l);
        p = _mm512_mask_shuffle_epi8(p, _mm512_cmpeq_epi8_mask(_mm512_and_si512(h, masks[0]), masks[0]), values[1], l);
        p = _mm512_mask_shuffle_epi8(p, _mm512_cmpeq_epi8_mask(_mm512_and_si512(h, masks[1]), masks[1]), values[2], l);
        p = _mm512_mask_shuffle_epi8(p, _mm512_cmpeq_epi8_mask(_mm512_and_si512(h, masks[2]), masks[2]), values[3], l);
        return p;
    }
    inline void prepare(const uint8_t * q4, const uint8_t * qh) {
        bits.prepare64(q4);
        auto h256_1 = _mm256_loadu_si256((const __m256i *)qh + 0);
        auto h256_2 = _mm256_loadu_si256((const __m256i *)qh + 1);
        auto h1 = _mm512_inserti32x8(_mm512_castsi256_si512(h256_1), _mm256_srli_epi16(h256_1, 4), 1);
        auto h2 = _mm512_inserti32x8(_mm512_castsi256_si512(h256_2), _mm256_srli_epi16(h256_2, 4), 1);
        bits.values[0] = make_one(bits.values[0], h1);
        bits.values[1] = make_one(bits.values[1], _mm512_srli_epi16(h1, 2));
        bits.values[2] = make_one(bits.values[2], h2);
        bits.values[3] = make_one(bits.values[3], _mm512_srli_epi16(h2, 2));
        // We now have in bits.valuse[0]: 0...31, 64...95
        //                bits.valuse[1]: 32..63, 96..127
        //                etc.
        auto tmp = _mm512_permutex2var_epi64(bits.values[0], permute1, bits.values[1]);
        bits.values[1] = _mm512_permutex2var_epi64(bits.values[0], permute2, bits.values[1]);
        bits.values[0] = tmp;
        tmp = _mm512_permutex2var_epi64(bits.values[2], permute1, bits.values[3]);
        bits.values[3] = _mm512_permutex2var_epi64(bits.values[2], permute2, bits.values[3]);
        bits.values[2] = tmp;
    }
    static void load_values(__m512i * values) {
        static const uint8_t kvalues_iq6nl[64] = {
               1,    7,   13,   19,   24,   30,   35,   40,   44,   49,   54,   58,   62,   66,   70,   74,
              77,   81,   84,   88,   91,   94,   97,  100,  103,  106,  109,  112,  115,  117,  120,  123,
             126,  128,  131,  134,  137,  140,  142,  145,  148,  151,  155,  158,  161,  164,  168,  172,
             175,  179,  183,  187,  191,  196,  200,  205,  210,  215,  220,  226,  231,  237,  243,  249,
        };
        for (int k = 0; k < 4; ++k) {
            auto values128 = _mm_loadu_si128((const __m128i *)kvalues_iq6nl + k);
            auto values256 = MM256_SET_M128I(values128, values128);
            values[k] = _mm512_inserti32x8(_mm512_castsi256_si512(values256), values256, 1);
        }
    }

    Q4Bits bits;
    IQXKScales2 iqxk;
    __m512i values[4];
    __m512i masks[3] = { _mm512_set1_epi8(0x01), _mm512_set1_epi8(0x02), _mm512_set1_epi8(0x03) };
    const __m512i permute1 = _mm512_set_epi64(11, 10,  9,  8, 3, 2, 1, 0);
    const __m512i permute2 = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
};

template <typename Dequantizer, int nrc_y>
static void mul_mat_iqX_k_q8_K_AVX512(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y> q8(info);

    Dequantizer deq(vx, bx);

    __m256  accm[nrc_y];
    __m512  accd[nrc_y];
    __m512i scales[4];

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm512_setzero_ps();
        for (int iy = 0; iy < nrc_y; ++iy) accm[iy] = _mm256_setzero_ps();

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            deq.new_block(i, q8, accm, scales);

            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m512i p1 = _mm512_maddubs_epi16(deq.bits.values[0], q8.load_quants64(iy, i, 0));
                const __m512i p2 = _mm512_maddubs_epi16(deq.bits.values[1], q8.load_quants64(iy, i, 1));
                const __m512i p3 = _mm512_maddubs_epi16(deq.bits.values[2], q8.load_quants64(iy, i, 2));
                const __m512i p4 = _mm512_maddubs_epi16(deq.bits.values[3], q8.load_quants64(iy, i, 3));
                auto sumi = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(_mm512_dpwssd_epi32(_mm512_dpwssd_epi32(_mm512_setzero_si512(),
                                    p1, scales[0]), p2, scales[1]), p3, scales[2]), p4, scales[3]);
                accd[iy] = _mm512_fmadd_ps(_mm512_set1_ps(deq.d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi), accd[iy]);
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum256 = _mm256_add_ps(_mm512_castps512_ps256(accd[iy]), _mm512_extractf32x8_ps(accd[iy], 1));
            info.store(ix, iy, hsum_float_8(_mm256_add_ps(accm[iy], sum256)));
        }

    }
}

template <typename Dequantizer, int nrc_y>
static void mul_mat_iqX_k_q8_K_AVX512_new(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y> q8(info);

    Dequantizer deq(vx, bx);

    __m512  accd[nrc_y];

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm512_setzero_ps();

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {
            deq.compute_block(i, q8, accd);
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, _mm512_reduce_add_ps(accd[iy]));
        }

    }
}

template <typename Q8>
inline void compute_block(int iy, int i, float d, const Q8& q8, const __m512i * values, const __m512i * scales, __m512 * accd) {
    const __m512i p1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), values[0], q8.load_quants64(iy, i, 0));
    const __m512i p2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), values[1], q8.load_quants64(iy, i, 1));
    const __m512i p3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), values[2], q8.load_quants64(iy, i, 2));
    const __m512i p4 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), values[3], q8.load_quants64(iy, i, 3));
    auto sumi = _mm512_dpwssd_epi32(_mm512_setzero_si512(), scales[0], _mm512_packs_epi32(p1, p2));
    sumi = _mm512_dpwssd_epi32(sumi, scales[1], _mm512_packs_epi32(p3, p4));
    accd[iy] = _mm512_fmadd_ps(_mm512_set1_ps(d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi), accd[iy]);
}

template <typename Dequantizer>
static void mul_mat_qX_K_q8_K_AVX512_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    constexpr int k_nx = 2;

    Q8<1> q8(info);

    Dequantizer deq1(vx, bx);
    Dequantizer deq2(vx, bx);

    Dequantizer * deq[k_nx];
    deq[0] = &deq1;
    deq[1] = &deq2;

    __m512i scales[2*k_nx];

    for (int ix = 0; ix < nrc_x; ++ix) {

        auto accd = _mm512_setzero_ps();
        auto accm = _mm256_setzero_ps();

        for (int kx = 0; kx < k_nx; ++kx) deq[kx]->new_row(ix);

        for (int i = 0; i < nb/k_nx; ++i) {

            for (int kx = 0; kx < k_nx; ++kx) deq[kx]->new_block(k_nx*i+kx, q8, &accm, scales+2*kx);

            for (int kx = 0; kx < k_nx; ++kx) {
                compute_block(0, k_nx*i+kx, deq[kx]->d, q8, deq[kx]->bits.values, scales+2*kx, &accd);
            }

        }
        if (2*(nb/2) < nb) {
            int i0 = 2*(nb/2);
            deq[0]->new_block(i0, q8, &accm, scales);
            compute_block(0, i0, deq[0]->d, q8, deq[0]->bits.values, scales, &accd);
        }

        auto sum256 = _mm256_add_ps(_mm512_castps512_ps256(accd), _mm512_extractf32x8_ps(accd, 1));
        info.store(ix, 0, hsum_float_8(_mm256_add_ps(accm, sum256)));
    }
}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_K_q8_K_AVX512(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y> q8(info);

    Dequantizer deq(vx, bx);

    __m256  accm[nrc_y];
    __m512  accd[nrc_y];
    __m512i scales[2];

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm512_setzero_ps();
        for (int iy = 0; iy < nrc_y; ++iy) accm[iy] = _mm256_setzero_ps();

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            deq.new_block(i, q8, accm, scales);

            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m512i p1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), deq.bits.values[0], q8.load_quants64(iy, i, 0));
                const __m512i p2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), deq.bits.values[1], q8.load_quants64(iy, i, 1));
                const __m512i p3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), deq.bits.values[2], q8.load_quants64(iy, i, 2));
                const __m512i p4 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), deq.bits.values[3], q8.load_quants64(iy, i, 3));
                auto sumi = _mm512_dpwssd_epi32(_mm512_setzero_si512(), scales[0], _mm512_packs_epi32(p1, p2));
                sumi = _mm512_dpwssd_epi32(sumi, scales[1], _mm512_packs_epi32(p3, p4));
                accd[iy] = _mm512_fmadd_ps(_mm512_set1_ps(deq.d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi), accd[iy]);
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum256 = _mm256_add_ps(_mm512_castps512_ps256(accd[iy]), _mm512_extractf32x8_ps(accd[iy], 1));
            info.store(ix, iy, hsum_float_8(_mm256_add_ps(accm[iy], sum256)));
        }

    }
}

#else

inline void prepare_scales_16(const __m256i& all_scales, __m256i * scales) {
    const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
    const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
    scales[0] = MM256_SET_M128I(l_scales, l_scales);
    scales[1] = MM256_SET_M128I(h_scales, h_scales);
}

struct IQXKScales {
    IQXKScales(int8_t shift, int8_t min_val) : min(_mm256_set1_epi16(min_val)), eshift(_mm_set1_epi8(shift)) {}
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m128i scales8, const Q8& q8, __m256 * accm, __m256i * scales) const {
        auto scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales8, hshuff));
        process(i, d, extra, scales16, q8, accm, scales);
    }
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m256i scales16, const Q8& q8, __m256 * accm, __m256i * scales) const {
        auto extra128 = _mm_set1_epi16(extra);
        extra128 = _mm_cmpeq_epi8(_mm_and_si128(extra128, emask), emask);
        extra128 = _mm_and_si128(extra128, eshift);
        extra128 = _mm_shuffle_epi8(extra128, eshuffle);
        auto scales_s = _mm256_mullo_epi16(scales16, _mm256_add_epi16(min, _mm256_cvtepi8_epi16(extra128)));
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i prod  = _mm256_madd_epi16(scales_s, q8.load_bsums(iy, i));
            accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accm[iy]);
        }
        prepare_scales_16(scales16, scales);
    }

    const __m256i min;
    const __m128i eshift;
    const __m128i hshuff   = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
    const __m128i emask    = _mm_set_epi32(0x80804040, 0x20201010, 0x08080404, 0x02020101);
    const __m128i eshuffle = _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
};

struct DequantizerIQ2KS final : public BaseDequantizer<block_iq2_ks, true, true> {
    DequantizerIQ2KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accm) {
        auto scales128 = make_scales(x[i].scales, x[i].extra >> 8);
        auto shifts = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi8(x[i].extra), hmask), hmask), m5);
        auto scales_s = _mm_mullo_epi16(scales128, _mm_cvtepi8_epi16(_mm_add_epi8(m32, shifts)));
        s8k.accum_mins(scales_s, q8, i, d, accm);
        return MM256_SET_M128I(scales128, scales128);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        bits.values[0] = _mm256_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values, bits.values[3]);
    }
    static inline __m256i load_values() {
        static const uint8_t kvalues_iq2nl[16] = {1, 19, 33, 49, 0, 0, 0, 0,  6, 24, 38, 54, 0, 0, 0, 0};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq2nl);
        return MM256_SET_M128I(val128, val128);
    }
    inline __m128i make_scales(const uint8_t * scales_l, uint8_t scales_h) const {
        const uint16_t * scales = (const uint16_t *)scales_l;
        uint32_t aux32 = scales[0] | (uint32_t(scales[1]) << 16);
        auto scl = _mm_srlv_epi32(_mm_set1_epi32(aux32), shift);
        scl = _mm_and_si128(_mm_shuffle_epi8(scl, shuffle), _mm_set1_epi8(0xf));
        auto sch = _mm_set1_epi8(scales_h);
        sch = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(sch, hmask), _mm_setzero_si128()), m16);
        return _mm_cvtepi8_epi16(_mm_add_epi8(scl, sch));
    }
    Q2Bits bits;
    Scales8KBase s8k;

    const __m256i values;
    const __m128i m16 = _mm_set1_epi8(-16);
    const __m128i m5  = _mm_set1_epi8(5);
    const __m128i m32 = _mm_set1_epi8(-32);
    const __m128i hmask = _mm_set1_epi64x(0x8040201008040201);
    const __m128i shuffle = _mm_set1_epi64x(0x0703060205010400);
    const __m128i shift = _mm_set_epi32(0, 0, 4, 0);
};

struct DequantizerIQ2K final : public BaseDequantizer<block_iq2_k> {
    DequantizerIQ2K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(5, -32), values(load_values()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        iqxk.process(i, d, x[i].extra, make_scales(x[i].scales), q8, accm, scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        bits.values[0] = _mm256_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values, bits.values[3]);
    }
    static inline __m256i load_values() {
        static const uint8_t kvalues_iq2nl[16] = {1, 19, 33, 49, 0, 0, 0, 0,  6, 24, 38, 54, 0, 0, 0, 0};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq2nl);
        return MM256_SET_M128I(val128, val128);
    }
    inline __m128i make_scales(const uint8_t * scales_l) const {
        uint64_t aux64; std::memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), maskl);
        return _mm_add_epi8(scl, m8);
    }

    Q2Bits bits;
    const IQXKScales iqxk;
    const __m256i values;
    const __m128i m8       = _mm_set1_epi8(-8);
    const __m128i maskl    = _mm_set1_epi8(0xf);
};

struct DequantizerIQ3K final : public BaseDequantizer<block_iq3_k> {
    DequantizerIQ3K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(4, -64), values(load_values()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        iqxk.process(i, d, x[i].extra, make_scales(x[i].scales_h, x[i].scales_l), q8, accm, scales);
        hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        auto h256 = j == 0 ? hbits : _mm256_srli_epi16(hbits, 4);
        bits.values[0] = _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(h256, 2), hmask));
        bits.values[1] = _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(h256, 1), hmask));
        bits.values[2] = _mm256_or_si256(bits.values[2], _mm256_and_si256(h256, hmask));
        bits.values[3] = _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_srli_epi16(h256, 1), hmask));
        bits.values[0] = _mm256_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values, bits.values[3]);
    }
    static inline __m256i load_values() {
        static const uint8_t kvalues_iq3nl[16] = {1, 24, 41, 54, 65, 77, 92, 111, 5, 28, 45, 58, 69, 81, 96, 115};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq3nl);
        return MM256_SET_M128I(val128, val128);
    }
    inline __m128i make_scales(uint16_t signs, const uint8_t * scales_l) const {
        uint64_t aux64; std::memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
        scl = _mm_add_epi8(_mm_slli_epi16(scl, 1), m1);
        const __m128i sc_signs = _mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi16(signs), sign_mask), sign_mask);
        const __m128i sch = _mm_shuffle_epi8(_mm_or_si128(sc_signs, _mm_set1_epi8(1)), hshuff);
        return _mm_sign_epi8(scl, sch);
    }

    Q2Bits bits;
    const IQXKScales iqxk;
    const __m256i values;
    __m256i hbits;
    const __m256i hmask  = _mm256_set1_epi8(4);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m128i sign_mask = _mm_set_epi64x(0x8080404020201010, 0x0808040402020101);
    const __m128i hshuff = _mm_loadu_si128((const __m128i*)k_shuff);
    constexpr static uint8_t k_shuff[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
};

struct DequantizerIQ4KSS final : public BaseDequantizer<block_iq4_kss, true> {
    DequantizerIQ4KSS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_iq4nl_values_256()) {}
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
        union { __m256i vec; uint16_t val[16]; } helper;
        for (int k = 0; k < 4; ++k) {
            data[k] = _mm256_loadu_si256((const __m256i *)x[i].qs + k);
            auto p = _mm256_and_si256(_mm256_cmpeq_epi16(_mm256_and_si256(data[k], m1), m1), smask);
            p = _mm256_add_epi32(_mm256_unpackhi_epi64(p, p), p);
            p = _mm256_add_epi32(_mm256_shuffle_epi32(p, _MM_SHUFFLE(2, 3, 0, 1)), p);
            helper.vec = _mm256_hadd_epi16(p, p);
            aux[2*k+0] = helper.val[0];
            aux[2*k+1] = helper.val[8];
            data[k] = _mm256_and_si256(data[k], bmask);
            data[k] = _mm256_xor_si256(data[k], _mm256_srli_epi16(data[k], 1));
        }
        auto scales128 = _mm_loadu_si128((const __m128i *)aux);
        auto shifts = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(scales128, _mm256_castsi256_si128(m1)), _mm256_castsi256_si128(m1)), m4);
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, mask), m127);
        auto scales_s = _mm_mullo_epi16(scales128, _mm_add_epi16(m128, shifts));
        s8k.accum_mins(scales_s, q8, i, d, accd);
        return MM256_SET_M128I(scales128, scales128);
    }
    inline void prepare(int, int j) {
        for (int k = 0; k < 2; ++k) {
            auto p1 = _mm256_castsi256_si128(data[2*j+k]);
            auto p2 = _mm256_extractf128_si256(data[2*j+k], 1);
            bits.values[2*k+0] = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(p1, 4), p1), bits.ml);
            bits.values[2*k+0] = _mm256_shuffle_epi8(values, bits.values[2*k+0]);
            bits.values[2*k+1] = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(p2, 4), p2), bits.ml);
            bits.values[2*k+1] = _mm256_shuffle_epi8(values, bits.values[2*k+1]);
        }
    }

    Q4Bits bits;
    Scales8KBase s8k;
    const __m256i values;
    __m256i data[4];
    const __m256i smask    = _mm256_set_epi64x(0x0080004000200010, 0x0008000400020001, 0x0080004000200010, 0x0008000400020001);
    const __m256i bmask    = _mm256_set1_epi16(-2); // 0xfffe;
    const __m128i mask     = _mm_set1_epi16(254);
    const __m128i m127     = _mm_set1_epi16(-127);
    const __m128i m128     = _mm_set1_epi16(-128);
    const __m256i m1       = _mm256_set1_epi16(1);
    const __m128i m4       = _mm_set1_epi16(4);
    uint16_t aux[8];
};

struct DequantizerIQ4KS final : public BaseDequantizer<block_iq4_ks, true> {
    DequantizerIQ4KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(); }
    template <typename Q8>
    inline __m256i new_block(int i, [[maybe_unused]] const Q8& q8, [[maybe_unused]] __m256 * accd) {
        auto scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, mask), m127);
        return MM256_SET_M128I(scales128, scales128);
    }
    inline void prepare(int i, int j) {
        bits.prepare16(x[i].qs, j);
        bits.values[0] = _mm256_shuffle_epi8(values[x[i].scales[4*j+0] & 1], bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values[x[i].scales[4*j+1] & 1], bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values[x[i].scales[4*j+2] & 1], bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values[x[i].scales[4*j+3] & 1], bits.values[3]);
    }
    void load_values() {
        auto v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
        values[0] = MM256_SET_M128I(v1, v1);
        values[1] = MM256_SET_M128I(v2, v2);
    }


    Q4Bits bits;
    __m256i values[2];
    const __m128i mask     = _mm_set1_epi16(254);
    const __m128i m127     = _mm_set1_epi16(-127);
};

struct DequantizerIQ4K final : public BaseDequantizer<block_iq4_k> {
    DequantizerIQ4K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(); }
    template <typename Q8>
    inline void new_block(int i, [[maybe_unused]] const Q8& q8, [[maybe_unused]] __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        auto scales8 = make_scales(x[i].scales_l, (const uint16_t *)x[i].scales_h);
        auto scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales8, hshuff));
        prepare_scales_16(scales16, scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare16(x[i].qs, j);
        auto extra = x[i].extra >> 8*j;
        bits.values[0] = _mm256_shuffle_epi8(values[extra & 3], bits.values[0]); extra >>= 2;
        bits.values[1] = _mm256_shuffle_epi8(values[extra & 3], bits.values[1]); extra >>= 2;
        bits.values[2] = _mm256_shuffle_epi8(values[extra & 3], bits.values[2]); extra >>= 2;
        bits.values[3] = _mm256_shuffle_epi8(values[extra & 3], bits.values[3]);
    }
    __m128i make_scales(const uint8_t * scales_l, const uint16_t * scales_h) const {
        uint64_t aux64;
        memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), maskl);
        const uint32_t aux32 = scales_h[0] | (scales_h[1] << 16);
        auto aux = _mm_and_si128(_mm_set_epi32(aux32 >> 2, aux32, aux32 << 2, aux32 << 4), maskh);
        auto sch = _mm_shuffle_epi8(aux, hshuff);
        return _mm_add_epi8(_mm_or_si128(scl, sch), m32);
    }
    void load_values() {
        auto v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
        values[0] = MM256_SET_M128I(v1, v1);
        values[1] = MM256_SET_M128I(v1, v2);
        values[2] = MM256_SET_M128I(v2, v1);
        values[3] = MM256_SET_M128I(v2, v2);
    }

    Q4Bits bits;
    const __m128i maskl    = _mm_set1_epi8(0xf);
    const __m128i maskh    = _mm_set1_epi8(0x30);
    const __m128i m32      = _mm_set1_epi8(-32);
    const __m128i hshuff   = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
    __m256i values[4];
};

struct DequantizerIQ5KS final : public BaseDequantizer<block_iq5_ks, true> {
    DequantizerIQ5KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(values); }
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
        hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
        auto scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
        auto shifts = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(scales128, m1), m1), m2);
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, mask), m127);
        auto scales_s = _mm_mullo_epi16(scales128, _mm_add_epi16(m128, shifts));
        s8k.accum_mins(scales_s, q8, i, d, accd);
        return MM256_SET_M128I(scales128, scales128);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        auto h = j == 0 ? hbits : _mm256_srli_epi16(hbits, 4);
        for (int k = 0; k < 4; ++k) {
            auto qh = _mm256_and_si256(_mm256_slli_epi16(h, 7-k), mh);
            auto q5vl = _mm256_or_si256(bits.values[k], qh);
            auto q5vh = _mm256_or_si256(bits.values[k], _mm256_xor_si256(qh, mh));
            bits.values[k] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
        }
    }
    static void load_values(__m256i * values) {
        static const uint8_t kvalues_iq5nl[32] = {
            2,  14,  25,  36,  45,  54,  63,  71,  78,  85,  92,  98, 104, 110, 116, 122, 127,
            133, 139, 145, 151, 157, 164, 171, 179, 187, 196, 205, 215, 225, 237, 249,
        };
        auto values128_1 = _mm_loadu_si128((const __m128i *)kvalues_iq5nl + 0);
        auto values128_2 = _mm_loadu_si128((const __m128i *)kvalues_iq5nl + 1);
        values[0] = MM256_SET_M128I(values128_1, values128_1);
        values[1] = MM256_SET_M128I(values128_2, values128_2);
    }

    Q4Bits bits;
    Scales8KBase s8k;
    __m256i hbits;
    __m256i values[2];
    const __m128i maskl    = _mm_set1_epi8(0xf);
    const __m128i maskh    = _mm_set1_epi8(0x30);
    const __m256i mh       = _mm256_set1_epi8(-128); // to avoid stupid warning about 0x80 overflowing
    const __m128i mask     = _mm_set1_epi16(254);
    const __m128i m127     = _mm_set1_epi16(-127);
    const __m128i m128     = _mm_set1_epi16(-128);
    const __m128i m1       = _mm_set1_epi16(1);
    const __m128i m2       = _mm_set1_epi16(2);
};

struct DequantizerIQ5K final : public BaseDequantizer<block_iq5_k> {
    DequantizerIQ5K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(2, 0) { load_values(values); }
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        iqxk.process(i, d, x[i].extra, make_scales(x[i].scales_l, (const uint16_t *)x[i].scales_h), q8, accm, scales);
        hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        auto h = j == 0 ? hbits : _mm256_srli_epi16(hbits, 4);
        for (int k = 0; k < 4; ++k) {
            auto qh = _mm256_and_si256(_mm256_slli_epi16(h, 7-k), mh);
            auto q5vl = _mm256_or_si256(bits.values[k], qh);
            auto q5vh = _mm256_or_si256(bits.values[k], _mm256_xor_si256(qh, mh));
            bits.values[k] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
        }
    }
    __m128i make_scales(const uint8_t * scales_l, const uint16_t * scales_h) const {
        uint64_t aux64;
        memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), maskl);
        const uint32_t aux32 = scales_h[0] | (scales_h[1] << 16);
        auto aux = _mm_and_si128(_mm_set_epi32(aux32 >> 2, aux32, aux32 << 2, aux32 << 4), maskh);
        auto sch = _mm_shuffle_epi8(aux, iqxk.hshuff);
        return _mm_add_epi8(_mm_or_si128(scl, sch), m32);
    }
    static void load_values(__m256i * values) {
        auto values128_1 = _mm_loadu_si128((const __m128i *)iq5nl_values + 0);
        auto values128_2 = _mm_loadu_si128((const __m128i *)iq5nl_values + 1);
        values[0] = MM256_SET_M128I(values128_1, values128_1);
        values[1] = MM256_SET_M128I(values128_2, values128_2);
    }

    Q4Bits bits;
    const IQXKScales iqxk;
    __m256i hbits;
    __m256i values[2];
    const __m128i maskl    = _mm_set1_epi8(0xf);
    const __m128i maskh    = _mm_set1_epi8(0x30);
    const __m128i m32      = _mm_set1_epi8(-32);
    const __m256i mh       = _mm256_set1_epi8(-128); // to avoid stupid warning about 0x80 overflowing
};

struct DequantizerIQ6K final : public BaseDequantizer<block_iq6_k> {
    DequantizerIQ6K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(1, 0) { load_values(values); }
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        auto scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
        auto scales16 = _mm256_cvtepi8_epi16(scales8);
        iqxk.process(i, d, x[i].extra, scales16, q8, accm, scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        auto hbits = _mm256_loadu_si256((const __m256i *)x[i].qh + j);
        for (int k = 0; k < 4; ++k) {
            bits.values[k] = make_one(bits.values[k], hbits);
            hbits = _mm256_srli_epi16(hbits, 2);
        }
    }
    inline __m256i make_one(__m256i l, __m256i hbits) const {
        auto mask4 = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, mh3), mh3);
        auto h1 = _mm256_andnot_si256(mask4, hbits);
        auto mask2 = _mm256_cmpeq_epi8(_mm256_and_si256(h1, mh1), mh1);
        auto mask3 = _mm256_cmpeq_epi8(_mm256_and_si256(h1, mh2), mh2);
        auto mask1 = _mm256_andnot_si256(_mm256_or_si256(mask4, _mm256_or_si256(mask2, mask3)), _mm256_set1_epi8(-1)); // 0xff;
        return _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(mask1, _mm256_shuffle_epi8(values[0], l)),
                                               _mm256_and_si256(mask2, _mm256_shuffle_epi8(values[1], l))),
                               _mm256_or_si256(_mm256_and_si256(mask3, _mm256_shuffle_epi8(values[2], l)),
                                               _mm256_and_si256(mask4, _mm256_shuffle_epi8(values[3], l))));
    }
    static void load_values(__m256i * values) {
        for (int k = 0; k < 4; ++k) {
            auto values128 = _mm_loadu_si128((const __m128i *)iq6nl_values + k);
            values[k] = MM256_SET_M128I(values128, values128);
        }
    }

    Q4Bits bits;
    const IQXKScales iqxk;
    __m256i values[4];
    const __m256i mh1 = _mm256_set1_epi8(1);
    const __m256i mh2 = _mm256_set1_epi8(2);
    const __m256i mh3 = _mm256_set1_epi8(3);
    const __m256i mh       = _mm256_set1_epi8(-128); // to avoid stupid warning about 0x80 overflowing
};

inline __m256i get_scale_shuffle_16(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

inline void set_scales_16(const __m256i& all_scales, __m256i * scales) {
    scales[0] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(0));
    scales[1] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(1));
    scales[2] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(2));
    scales[3] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(3));
}

inline __m256i get_scale_shuffle_8(int i) {
    return _mm256_set1_epi16((2*i) | ((2*i+1) << 8));
}

inline void set_scales_8(const __m256i& all_scales, int j, __m256i * scales) {
    scales[0] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+0));
    scales[1] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+1));
    scales[2] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+2));
    scales[3] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+3));
}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qY_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Q8<nrc_y> q8(info);

    __m256i all_scales[2];
    __m256i scales[4];
    __m256  accd[nrc_y];

    Dequantizer deq(vx, bx);

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        for (int i = 0; i < nb; ++i) {

            deq.new_block(i, q8, accd, all_scales);

            __m256i sumi[nrc_y];

            for (int j = 0; j < QK_K/128; ++j) {
                deq.prepare(i, j);
                set_scales_16(all_scales[j], scales);
                if constexpr (std::is_same_v<Dequantizer, DequantizerIQ4K> ||
                              std::is_same_v<Dequantizer, DequantizerIQ5K> ||
                              std::is_same_v<Dequantizer, DequantizerIQ6K>) {
                    multiply_add_avx2(deq.bits, scales, j, i, q8, sumi);
                } else {
                    multiply_add(deq.bits, scales, j, i, q8, sumi);
                }
            }

            for (int iy = 0; iy < nrc_y; ++iy) {
                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8.scale(iy, i)), _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }

    }

}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y> q8(info);

    Dequantizer deq(vx, bx);

    __m256  accd[nrc_y];
    __m256i scales[4];

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            auto all_scales = deq.new_block(i, q8, accd);

            __m256i sumi[nrc_y];

            for (int j = 0; j < QK_K/128; ++j) {

                deq.prepare(i, j);

                set_scales_8(all_scales, j, scales);

                if constexpr (std::is_same_v<Dequantizer, DequantizerIQ4KS>) {
                    multiply_add_avx2(deq.bits, scales, j, i, q8, sumi);
                } else {
                    multiply_add(deq.bits, scales, j, i, q8, sumi);
                }

            }

            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(deq.d*q8.scale(iy, i));
                accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }

        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }

    }
}

#endif


template <typename Dequantizer> void set_functions(std::array<mul_mat_t, IQK_MAX_NY>& funcs) {
#ifdef HAVE_FANCY_SIMD
    if constexpr (std::is_same_v<Dequantizer, DequantizerIQ2KS> ||
                  std::is_same_v<Dequantizer, DequantizerIQ4KS> ||
                  std::is_same_v<Dequantizer, DequantizerIQ5KS>) {
        funcs[0] = mul_mat_iqX_k_q8_K_AVX512_new<Dequantizer, 1>;
        funcs[1] = mul_mat_iqX_k_q8_K_AVX512_new<Dequantizer, 2>;
        funcs[2] = mul_mat_iqX_k_q8_K_AVX512_new<Dequantizer, 3>;
        funcs[3] = mul_mat_iqX_k_q8_K_AVX512_new<Dequantizer, 4>;
        funcs[4] = mul_mat_iqX_k_q8_K_AVX512_new<Dequantizer, 5>;
        funcs[5] = mul_mat_iqX_k_q8_K_AVX512_new<Dequantizer, 6>;
        funcs[6] = mul_mat_iqX_k_q8_K_AVX512_new<Dequantizer, 7>;
        funcs[7] = mul_mat_iqX_k_q8_K_AVX512_new<Dequantizer, 8>;
    } else if constexpr (std::is_same_v<Dequantizer, DequantizerIQ2K>) {
        funcs[0] = mul_mat_qX_K_q8_K_AVX512_1<Dequantizer>;
        funcs[1] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 2>;
        funcs[2] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 3>;
        funcs[3] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 4>;
        funcs[4] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 5>;
        funcs[5] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 6>;
        funcs[6] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 7>;
        funcs[7] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 8>;
    } else {
        funcs[0] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 1>;
        funcs[1] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 2>;
        funcs[2] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 3>;
        funcs[3] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 4>;
        funcs[4] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 5>;
        funcs[5] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 6>;
        funcs[6] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 7>;
        funcs[7] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 8>;
    }
#else
    if constexpr (std::is_same_v<Dequantizer, DequantizerIQ2K>||
                  std::is_same_v<Dequantizer, DequantizerIQ3K>||
                  std::is_same_v<Dequantizer, DequantizerIQ4K>||
                  std::is_same_v<Dequantizer, DequantizerIQ5K>||
                  std::is_same_v<Dequantizer, DequantizerIQ6K>) {
        funcs[0] = mul_mat_qY_K_q8_K_T<Dequantizer, 1>;
        funcs[1] = mul_mat_qY_K_q8_K_T<Dequantizer, 2>;
        funcs[2] = mul_mat_qY_K_q8_K_T<Dequantizer, 3>;
        funcs[3] = mul_mat_qY_K_q8_K_T<Dequantizer, 4>;
        funcs[4] = mul_mat_qY_K_q8_K_T<Dequantizer, 5>;
        funcs[5] = mul_mat_qY_K_q8_K_T<Dequantizer, 6>;
        funcs[6] = mul_mat_qY_K_q8_K_T<Dequantizer, 7>;
        funcs[7] = mul_mat_qY_K_q8_K_T<Dequantizer, 8>;
    } else {
        funcs[0] = mul_mat_qX_K_q8_K_T<Dequantizer, 1>;
        funcs[1] = mul_mat_qX_K_q8_K_T<Dequantizer, 2>;
        funcs[2] = mul_mat_qX_K_q8_K_T<Dequantizer, 3>;
        funcs[3] = mul_mat_qX_K_q8_K_T<Dequantizer, 4>;
        funcs[4] = mul_mat_qX_K_q8_K_T<Dequantizer, 5>;
        funcs[5] = mul_mat_qX_K_q8_K_T<Dequantizer, 6>;
        funcs[6] = mul_mat_qX_K_q8_K_T<Dequantizer, 7>;
        funcs[7] = mul_mat_qX_K_q8_K_T<Dequantizer, 8>;
    }

#endif
}

} // namespace

bool iqk_set_kernels_iqk_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels) {

    if (ne00%QK_K != 0 || ggml_type(typeB) != GGML_TYPE_Q8_K) {
        return false;
    }

    switch (typeA) {
        case GGML_TYPE_IQ2_KS:
            set_functions<DequantizerIQ2KS>(kernels);
            break;
        case GGML_TYPE_IQ2_K:
            set_functions<DequantizerIQ2K>(kernels);
            break;
        case GGML_TYPE_IQ3_K:
            set_functions<DequantizerIQ3K>(kernels);
            break;
        case GGML_TYPE_IQ4_KSS:
            set_functions<DequantizerIQ4KSS>(kernels);
            break;
       case GGML_TYPE_IQ4_KS:
            set_functions<DequantizerIQ4KS>(kernels);
            break;
        case GGML_TYPE_IQ4_K:
            set_functions<DequantizerIQ4K>(kernels);
            break;
        case GGML_TYPE_IQ5_KS:
            set_functions<DequantizerIQ5KS>(kernels);
            break;
        case GGML_TYPE_IQ5_K:
            set_functions<DequantizerIQ5K>(kernels);
            break;
        case GGML_TYPE_IQ6_K:
            set_functions<DequantizerIQ6K>(kernels);
            break;
        default:
            return false;
    }

    return true;

}

#endif
