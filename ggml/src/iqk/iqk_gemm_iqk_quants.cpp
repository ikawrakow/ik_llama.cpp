#include "iqk_gemm_iqk_quants.h"
#include <cstring>

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#ifdef __x86_64__

namespace {

#ifdef HAVE_FANCY_SIMD

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

struct DequantizerIQ3KS final : public BaseDequantizer<block_iq3_ks, true, true> {
    DequantizerIQ3KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
    template <typename Q8>
    inline void compute_block(int i, const Q8& q8, __m512 * acc) {
        uint32_t aux32; std::memcpy(&aux32, x[i].scales, 4);
        auto scl = _mm_srlv_epi32(_mm_set1_epi32(aux32), _mm_set_epi32(0, 0, 4, 0));
        auto scales128 = _mm_cvtepu8_epi16(_mm_and_si128(scl, _mm_set1_epi8(0xf)));
        scales128 = _mm_mask_add_epi16(scales128, __mmask8(x[i].extra & 0xff), scales128, _mm_set1_epi16(16));
        scales128 = _mm_sub_epi16(scales128, _mm_set1_epi16(16));
        auto shifts = _mm_mask_add_epi16(m64, __mmask8(x[i].extra >> 8), m64, _mm_set1_epi16(4));
        auto mins128 = _mm_mullo_epi16(scales128, shifts);
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

    Q2Bits bits;
    Scales8KBase s8k;

    const __m128i m64 = _mm_set1_epi16(-64);
    const __m512i values;
    const __m512i hmask = _mm512_set1_epi8(4);
    const __m512i shuffles[4] = {
        _mm512_inserti32x8(_mm512_set1_epi16(0x0100), _mm256_set1_epi16(0x0302), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0504), _mm256_set1_epi16(0x0706), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0908), _mm256_set1_epi16(0x0b0a), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0d0c), _mm256_set1_epi16(0x0f0e), 1),
    };
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

struct DequantizerIQ2KL final : public BaseDequantizer<block_iq2_kl, true, true> {
    DequantizerIQ2KL(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(); }
    inline __m128i make_scales(int i) const {
        uint32_t aux32; std::memcpy(&aux32, x[i].scales_l, 4);
        auto scl = _mm_cvtepu8_epi16(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(aux32), _mm_set_epi32(0, 0, 4, 0)), _mm_set1_epi8(0xf)));
        auto sch = _mm_srlv_epi16(_mm_sllv_epi64(_mm_set1_epi16(x[i].scales_h), _mm_set_epi64x(0, 8)), _mm_set1_epi64x(0x000a000800060004));
        auto scales128 = _mm_sub_epi16(_mm_or_si128(scl, _mm_and_si128(sch, _mm_set1_epi16(0x30))), _mm_set1_epi16(32));
        return scales128;
    }
    template <typename Q8>
    inline void compute_block(int i, const Q8& q8, __m512 * acc) {
        auto scales128 = make_scales(i);
        auto mins128 = _mm_mullo_epi16(scales128, _mm_set1_epi16(-64));
        auto mins = MM256_SET_M128I(_mm_shuffle_epi8(mins128, s8k.shuffles[1]), _mm_shuffle_epi8(mins128, s8k.shuffles[0]));
        auto scales256 = MM256_SET_M128I(scales128, scales128);
        auto all_scales = _mm512_inserti32x8(_mm512_castsi256_si512(scales256), scales256, 1);
        __m512i scales[4];
        for (int k = 0; k < 4; ++k) scales[k] = _mm512_shuffle_epi8(all_scales, shuffles[k]);
        prepare(i);
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
    inline void prepare(int i) {
        __m512i ql[2], qs[4];
        __mmask64 mask[2];
        auto lbits = _mm512_loadu_si512((const __m512i *)x[i].qs);
        ql[0] = _mm512_and_si512(lbits, m4);
        ql[1] = _mm512_and_si512(_mm512_srli_epi16(lbits, 4), m4);
        auto tmp = _mm512_permutex2var_epi64(ql[0], permute1, ql[1]);
        ql[1] = _mm512_permutex2var_epi64(ql[0], permute2, ql[1]);
        ql[0] = tmp;
        auto h128 = _mm_loadu_si128((const __m128i *)x[i].qh);
        auto h256 = MM256_SET_M128I(_mm_srli_epi16(h128, 1), h128);
        auto h512 = _mm512_inserti32x8(_mm512_castsi256_si512(h256), _mm256_srli_epi16(h256, 2), 1);
        mask[0] = _mm512_cmpeq_epi8_mask(_mm512_and_si512(h512, m01), m01);
        mask[1] = _mm512_cmpeq_epi8_mask(_mm512_and_si512(h512, m10), m10);

        for (int k = 0; k < 2; ++k) {
            // qs[0]: even quants when hbits is not set (so pair index is in  0...15)
            // qs[1]: even quants when hbits is set     (so pair index is in 16...31)
            // qs[2]:  odd quants when hbits is not set (so pair index is in  0...15)
            // qs[3]:  odd quants when hbits is set     (so pair index is in 16...31)
            // if we blend qs[0] and qs[1] with the hbit mask, we get the correct even quants -> q1
            // if we blend qs[2] and qs[3] with the hbit mask, we get the correct odd  quants -> q2
            // If we convert q1 and q2 to int16_t, shift q2 left by 8 bits, and or them, we get the quants in the correct order
            for (int l = 0; l < 4; ++l) qs[l] = _mm512_shuffle_epi8(values[l], ql[k]);
            auto q1 = _mm512_mask_blend_epi8(mask[k], qs[0], qs[1]);
            auto q2 = _mm512_mask_blend_epi8(mask[k], qs[2], qs[3]);
            auto t1 = _mm512_unpacklo_epi8(q1, q2); //  0...15, 32...47, 64...79,  96...111
            auto t2 = _mm512_unpackhi_epi8(q1, q2); // 16...31, 48...63, 80...95, 112...127
            bits.values[2*k+0] = _mm512_permutex2var_epi64(t1, permute1, t2);
            bits.values[2*k+1] = _mm512_permutex2var_epi64(t1, permute2, t2);
        }
    }
    void load_values() {
        static const uint8_t k_values[64] = {
            1, 1, 24, 24, 24, 24, 41, 41, 41, 41, 41, 54, 54, 54, 54, 65, 65, 65, 65, 65, 77, 77, 77, 77, 77, 92, 92, 92, 92, 92, 111, 111,
            41, 77, 1, 54, 77, 111, 24, 41, 65, 77, 92, 1, 65, 77, 111, 41, 54, 65, 77, 92, 24, 41, 54, 65, 77, 1, 41, 65, 92, 111, 41, 77,
        };
        for (int k = 0; k < 4; ++k) {
            auto v128 = _mm_loadu_si128((const __m128i *)k_values + k);
            auto v256 = MM256_SET_M128I(v128, v128);
            values[k] = _mm512_inserti32x8(_mm512_castsi256_si512(v256), v256, 1);
        }
    }

    struct { __m512i values[4]; } bits;
    Scales8KBase s8k;
    const __m512i m01 = _mm512_set1_epi8(0x01);
    const __m512i m10 = _mm512_set1_epi8(0x10);
    const __m512i m4  = _mm512_set1_epi8(0xf);
    const __m512i permute1 = _mm512_set_epi64(11, 10, 3, 2,  9,  8, 1, 0);
    const __m512i permute2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    __m512i values[4];
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

struct DequantizerIQ2KL final : public BaseDequantizer<block_iq2_kl, true, true> {
    DequantizerIQ2KL(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(); }
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accm) {
        auto hbits128 = _mm_loadu_si128((const __m128i *)x[i].qh);
        hbits = MM256_SET_M128I(_mm_srli_epi16(hbits128, 1), hbits128);
        auto scales128 = make_scales(i);
        auto scales_s = _mm_mullo_epi16(scales128, _mm_set1_epi16(-64));
        s8k.accum_mins(scales_s, q8, i, d, accm);
        return MM256_SET_M128I(scales128, scales128);
    }
    inline void prepare(int i, int j) {
        __m256i ql[2], mask[2];
        auto b1 = _mm_loadu_si128((const __m128i *)x[i].qs+2*j+0);
        auto b2 = _mm_loadu_si128((const __m128i *)x[i].qs+2*j+1);
        ql[0] = _mm256_and_si256(_mm256_set1_epi8(0xf), MM256_SET_M128I(_mm_srli_epi16(b1, 4), b1));
        ql[1] = _mm256_and_si256(_mm256_set1_epi8(0xf), MM256_SET_M128I(_mm_srli_epi16(b2, 4), b2));
        mask[0] = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, _mm256_set1_epi8(0x1)), _mm256_set1_epi8(0x1));
        mask[1] = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, _mm256_set1_epi8(0x4)), _mm256_set1_epi8(0x4));
        for (int k = 0; k < 2; ++k) {
            auto v0 = _mm256_shuffle_epi8(values[0], ql[k]);
            auto v1 = _mm256_shuffle_epi8(values[1], ql[k]);
            auto v2 = _mm256_shuffle_epi8(values[2], ql[k]);
            auto v3 = _mm256_shuffle_epi8(values[3], ql[k]);
            auto q1 = _mm256_or_si256(_mm256_and_si256(mask[k], v1), _mm256_andnot_si256(mask[k], v0));
            auto q2 = _mm256_or_si256(_mm256_and_si256(mask[k], v3), _mm256_andnot_si256(mask[k], v2));
            auto q1l = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(q1));
            auto q1h = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(q1, 1));
            auto q2l = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(q2));
            auto q2h = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(q2, 1));
            bits.values[2*k+0] = _mm256_or_si256(q1l, _mm256_slli_epi16(q2l, 8));
            bits.values[2*k+1] = _mm256_or_si256(q1h, _mm256_slli_epi16(q2h, 8));
        }
        hbits = _mm256_srli_epi16(hbits, 4);
    }
    inline __m128i make_scales(int i) const {
        uint32_t aux32; std::memcpy(&aux32, x[i].scales_l, 4);
        auto scl = _mm_cvtepu8_epi16(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(aux32), shift), _mm_set1_epi8(0xf)));
        auto sch = _mm_srlv_epi32(_mm_set1_epi16(x[i].scales_h), _mm_set_epi32(12, 8, 4, 0));
        sch = _mm_and_si128(sch, _mm_set1_epi32(0x000c0003));
        sch = _mm_mullo_epi16(sch, _mm_set1_epi32(0x00040010));
        auto scales128 = _mm_sub_epi16(_mm_or_si128(scl, sch), _mm_set1_epi16(32));
        return scales128;
    }
    void load_values() {
        static const uint8_t k_values[64] = {
            1, 1, 24, 24, 24, 24, 41, 41, 41, 41, 41, 54, 54, 54, 54, 65, 65, 65, 65, 65, 77, 77, 77, 77, 77, 92, 92, 92, 92, 92, 111, 111,
            41, 77, 1, 54, 77, 111, 24, 41, 65, 77, 92, 1, 65, 77, 111, 41, 54, 65, 77, 92, 24, 41, 54, 65, 77, 1, 41, 65, 92, 111, 41, 77,
        };
        for (int k = 0; k < 4; ++k) {
            auto v128 = _mm_loadu_si128((const __m128i *)k_values + k);
            values[k] = MM256_SET_M128I(v128, v128);
        }
    }
    struct { __m256i values[4]; } bits;
    Scales8KBase s8k;

    __m256i values[4];
    __m256i hbits;
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

struct DequantizerIQ3KS final : public BaseDequantizer<block_iq3_ks, true, true> {
    DequantizerIQ3KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
    template <typename Q8>
    inline __m256i new_block(int i, [[maybe_unused]] const Q8& q8, [[maybe_unused]] __m256 * accd) {
        uint32_t aux32; std::memcpy(&aux32, x[i].scales, 4);
        auto scl = _mm_cvtepi8_epi16(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(aux32), _mm_set_epi32(0, 0, 4, 0)), _mm_set1_epi8(0xf)));
        auto sch = _mm_cmpeq_epi16(_mm_and_si128(_mm_set1_epi16(x[i].extra), mask), mask);
        auto scales128 = _mm_add_epi16(scl, _mm_and_si128(sch, _mm_set1_epi16(16)));
        scales128 = _mm_sub_epi16(scales128, _mm_set1_epi16(16));
        return MM256_SET_M128I(scales128, scales128);
    }
    inline void prepare(int i, int j) {
        uint8_t extra = x[i].extra >> (8 + 4*j);
        hbits = j == 0 ? _mm256_loadu_si256((const __m256i *)x[i].qh) : _mm256_srli_epi16(hbits, 4);
        bits.prepare(x[i].qs, j);
        bits.values[0] = _mm256_add_epi8(_mm256_set1_epi8((extra << 3) & 8), _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh)));
        bits.values[1] = _mm256_add_epi8(_mm256_set1_epi8((extra << 2) & 8), _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(hbits, 1), mh)));
        bits.values[2] = _mm256_add_epi8(_mm256_set1_epi8((extra << 1) & 8), _mm256_or_si256(bits.values[2], _mm256_and_si256(hbits, mh)));
        bits.values[3] = _mm256_add_epi8(_mm256_set1_epi8((extra << 0) & 8), _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_srli_epi16(hbits, 1), mh)));
        for (int k = 0; k < 4; ++k) bits.values[k] = _mm256_shuffle_epi8(values, bits.values[k]);
    }
    inline __m256i load_values() {
        auto v = _mm_loadu_si128((const __m128i *)iq3nl_values);
        return MM256_SET_M128I(v, v);
    }


    Q2Bits bits;
    __m256i hbits;
    const __m256i values;
    const __m256i mh   = _mm256_set1_epi8(4);
    const __m128i mask = _mm_setr_epi16(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
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

                if constexpr (std::is_same_v<Dequantizer, DequantizerIQ4KS> || std::is_same_v<Dequantizer, DequantizerIQ3KS>) {
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

template <int nrc_y>
//IQK_ALWAYS_INLINE void iq234_k_accum_mins(int ibl, __m256i i8scales1, __m256i i8scales2, const Q8<nrc_y, block_q8_K>& q8, __m256i shuff,
inline void iq234_k_accum_mins(int ibl, __m256i i8scales1, __m256i i8scales2, const Q8<nrc_y, block_q8_K>& q8, __m256i shuff,
        __m256i * isum, int16_t min) {
    auto t1 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales1, 0)), shuff); // blocks  0,  1,  2,  3 for each row
    auto t2 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales1, 1)), shuff); // blocks  4,  5,  6,  7 for each row
    auto t3 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales2, 0)), shuff); // blocks  8,  9, 10, 11 for each row
    auto t4 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales2, 1)), shuff); // blocks 12, 13, 14, 15 for each row
    if constexpr (nrc_y == 1) {
        auto s1 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 0), _mm256_extracti128_si256(t1, 0)); // blocks 0, 1,  8, 9
        auto s2 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 1), _mm256_extracti128_si256(t1, 1)); // blocks 2, 3, 10, 11
        auto s3 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 0), _mm256_extracti128_si256(t2, 0)); // blocks 4, 5, 12, 13
        auto s4 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 1), _mm256_extracti128_si256(t2, 1)); // blocks 6, 7, 14, 15
        auto sumi = _mm256_setzero_si256();
        auto bsums = q8.load_bsums(0, ibl);
#ifdef HAVE_FANCY_SIMD
        sumi = _mm256_dpwssd_epi32(sumi, s1, _mm256_shuffle_epi32(bsums, 0x00));
        sumi = _mm256_dpwssd_epi32(sumi, s2, _mm256_shuffle_epi32(bsums, 0x55));
        sumi = _mm256_dpwssd_epi32(sumi, s3, _mm256_shuffle_epi32(bsums, 0xaa));
        sumi = _mm256_dpwssd_epi32(sumi, s4, _mm256_shuffle_epi32(bsums, 0xff));
#else
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s1, _mm256_shuffle_epi32(bsums, 0x00)));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s2, _mm256_shuffle_epi32(bsums, 0x55)));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s3, _mm256_shuffle_epi32(bsums, 0xaa)));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s4, _mm256_shuffle_epi32(bsums, 0xff)));
#endif
        isum[0] = _mm256_mullo_epi32(sumi, _mm256_set1_epi32(min));

    } else {
        auto s1 = _mm256_mullo_epi16(_mm256_set1_epi16(min), MM256_SET_M128I(_mm256_extracti128_si256(t3, 0), _mm256_extracti128_si256(t1, 0))); // blocks 0, 1,  8, 9
        auto s2 = _mm256_mullo_epi16(_mm256_set1_epi16(min), MM256_SET_M128I(_mm256_extracti128_si256(t3, 1), _mm256_extracti128_si256(t1, 1))); // blocks 2, 3, 10, 11
        auto s3 = _mm256_mullo_epi16(_mm256_set1_epi16(min), MM256_SET_M128I(_mm256_extracti128_si256(t4, 0), _mm256_extracti128_si256(t2, 0))); // blocks 4, 5, 12, 13
        auto s4 = _mm256_mullo_epi16(_mm256_set1_epi16(min), MM256_SET_M128I(_mm256_extracti128_si256(t4, 1), _mm256_extracti128_si256(t2, 1))); // blocks 6, 7, 14, 15
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto bsums = q8.load_bsums(iy, ibl);
#ifdef HAVE_FANCY_SIMD
            isum[iy] = _mm256_dpwssd_epi32(isum[iy], s1, _mm256_shuffle_epi32(bsums, 0x00));
            isum[iy] = _mm256_dpwssd_epi32(isum[iy], s2, _mm256_shuffle_epi32(bsums, 0x55));
            isum[iy] = _mm256_dpwssd_epi32(isum[iy], s3, _mm256_shuffle_epi32(bsums, 0xaa));
            isum[iy] = _mm256_dpwssd_epi32(isum[iy], s4, _mm256_shuffle_epi32(bsums, 0xff));
#else
            isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s1, _mm256_shuffle_epi32(bsums, 0x00)));
            isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s2, _mm256_shuffle_epi32(bsums, 0x55)));
            isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s3, _mm256_shuffle_epi32(bsums, 0xaa)));
            isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s4, _mm256_shuffle_epi32(bsums, 0xff)));
#endif
        }
    }
}

template <int nrc_y>
inline void iq2345_k_accum_mins(int ibl, __m256i i8scales1, __m256i i8scales2, const Q8<nrc_y, block_q8_K>& q8, __m256i shuff,
        __m256i extra, __m256i * isum, int8_t min, int8_t delta) {
    auto mask = _mm256_set_epi64x(0x0808080808080808, 0x0404040404040404, 0x0202020202020202, 0x0101010101010101);
    auto vdelta = _mm256_set1_epi8(delta);
    auto vmin   = _mm256_set1_epi8(min);
    auto min1 = _mm256_add_epi8(vmin, _mm256_and_si256(vdelta, _mm256_cmpeq_epi8(_mm256_and_si256(extra, mask), mask)));
    auto min2 = _mm256_add_epi8(vmin, _mm256_and_si256(vdelta, _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_srli_epi16(extra, 4), mask), mask)));
    auto t1 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales1, 0)), shuff); // blocks  0,  1,  2,  3 for each row
    auto t2 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales1, 1)), shuff); // blocks  4,  5,  6,  7 for each row
    auto t3 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales2, 0)), shuff); // blocks  8,  9, 10, 11 for each row
    auto t4 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales2, 1)), shuff); // blocks 12, 13, 14, 15 for each row
    auto m1 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(min1, 0)), shuff); // blocks  0,  1,  2,  3 for each row
    auto m2 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(min1, 1)), shuff); // blocks  4,  5,  6,  7 for each row
    auto m3 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(min2, 0)), shuff); // blocks  8,  9, 10, 11 for each row
    auto m4 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(min2, 1)), shuff); // blocks 12, 13, 14, 15 for each row
    auto s1 = _mm256_mullo_epi16(MM256_SET_M128I(_mm256_extracti128_si256(m3, 0), _mm256_extracti128_si256(m1, 0)),
                                 MM256_SET_M128I(_mm256_extracti128_si256(t3, 0), _mm256_extracti128_si256(t1, 0))); // blocks 0, 1,  8, 9
    auto s2 = _mm256_mullo_epi16(MM256_SET_M128I(_mm256_extracti128_si256(m3, 1), _mm256_extracti128_si256(m1, 1)),
                                 MM256_SET_M128I(_mm256_extracti128_si256(t3, 1), _mm256_extracti128_si256(t1, 1))); // blocks 2, 3, 10, 11
    auto s3 = _mm256_mullo_epi16(MM256_SET_M128I(_mm256_extracti128_si256(m4, 0), _mm256_extracti128_si256(m2, 0)),
                                 MM256_SET_M128I(_mm256_extracti128_si256(t4, 0), _mm256_extracti128_si256(t2, 0))); // blocks 4, 5, 12, 13
    auto s4 = _mm256_mullo_epi16(MM256_SET_M128I(_mm256_extracti128_si256(m4, 1), _mm256_extracti128_si256(m2, 1)),
                                 MM256_SET_M128I(_mm256_extracti128_si256(t4, 1), _mm256_extracti128_si256(t2, 1))); // blocks 6, 7, 14, 15
    for (int iy = 0; iy < nrc_y; ++iy) {
        auto bsums = q8.load_bsums(iy, ibl);
#ifdef HAVE_FANCY_SIMD
        isum[iy] = _mm256_dpwssd_epi32(isum[iy], s1, _mm256_shuffle_epi32(bsums, 0x00));
        isum[iy] = _mm256_dpwssd_epi32(isum[iy], s2, _mm256_shuffle_epi32(bsums, 0x55));
        isum[iy] = _mm256_dpwssd_epi32(isum[iy], s3, _mm256_shuffle_epi32(bsums, 0xaa));
        isum[iy] = _mm256_dpwssd_epi32(isum[iy], s4, _mm256_shuffle_epi32(bsums, 0xff));
#else
        isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s1, _mm256_shuffle_epi32(bsums, 0x00)));
        isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s2, _mm256_shuffle_epi32(bsums, 0x55)));
        isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s3, _mm256_shuffle_epi32(bsums, 0xaa)));
        isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s4, _mm256_shuffle_epi32(bsums, 0xff)));
#endif
    }
}

template <int nrc_y>
static void mul_mat_iq2_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto ms  = _mm256_set1_epi8(4);
    auto m03 = _mm256_set1_epi8(0x03);
    auto shift_shuffle = _mm256_set_epi64x(0x0707070706060606, 0x0505050504040404, 0x0303030302020202, 0x0101010100000000);
    static const uint8_t kvalues_iq2nl[32] = {1, 19, 33, 49, 6, 24, 38, 54, 1, 19, 33, 49, 6, 24, 38, 54, 1, 19, 33, 49, 6, 24, 38, 54, 1, 19, 33, 49, 6, 24, 38, 54};
    auto values = _mm256_loadu_si256((const __m256i*)kvalues_iq2nl);
    static const uint8_t k_shuff[32] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    auto shuff = _mm256_loadu_si256((const __m256i *)k_shuff);
#ifndef HAVE_FANCY_SIMD
    auto s_shuffle = _mm256_set_epi64x(0x0f0e0f0e0d0c0d0c, 0x0b0a0b0a09080908, 0x0706070605040504, 0x0302030201000100);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y] = {};
    __m256i qx[4];
    uint64_t stored_scales[8];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_iq2_k_r4 * iq2 = (const block_iq2_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq2[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto extra = _mm256_set1_epi64x(*(const uint64_t *)iq2[ibl].extra);
            auto slbits = _mm256_loadu_si256((const __m256i *)iq2[ibl].scales);
            auto i8scales1 = _mm256_add_epi8(_mm256_and_si256(slbits, m4), _mm256_set1_epi8(-8));
            auto i8scales2 = _mm256_add_epi8(_mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4), _mm256_set1_epi8(-8));
            _mm256_storeu_si256((__m256i *)stored_scales+0, i8scales1);
            _mm256_storeu_si256((__m256i *)stored_scales+1, i8scales2);
            __m256i isum[nrc_y] = {};
            iq234_k_accum_mins(ibl, i8scales1, i8scales2, q8, shuff, isum, -32);
            for (int ib = 0; ib < QK_K/32; ++ib) {
#ifdef HAVE_FANCY_SIMD
                auto scales = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(stored_scales + ib)));
#else
                auto scales = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_set1_epi64x(stored_scales[ib])), s_shuffle);
#endif
                auto lb = _mm256_loadu_si256((const __m256i *)iq2[ibl].qs+ib);
                auto shift = _mm256_and_si256(ms, _mm256_slli_epi16(extra, 2)); extra = _mm256_srli_epi16(extra, 1);
                shift = _mm256_shuffle_epi8(shift, shift_shuffle);
                qx[0] = _mm256_and_si256(lb, m03);
                qx[1] = _mm256_and_si256(_mm256_srli_epi16(lb, 2), m03);
                qx[2] = _mm256_and_si256(_mm256_srli_epi16(lb, 4), m03);
                qx[3] = _mm256_and_si256(_mm256_srli_epi16(lb, 6), m03);
                qx[0] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[0], shift));
                qx[1] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[1], shift));
                qx[2] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[2], shift));
                qx[3] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[3], shift));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales, sumi));
#else
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                  _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                  _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(scales, sumi1), _mm256_madd_epi16(scales, sumi2)));
#endif
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

template <int nrc_y>
static void mul_mat_iq3_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto ms  = _mm256_set1_epi8(8);
    auto m03 = _mm256_set1_epi8(0x03);
    auto m04 = _mm256_set1_epi8(0x04);
    auto smask = _mm256_set_epi64x(0x0808080808080808, 0x0404040404040404, 0x0202020202020202, 0x0101010101010101);
    auto shift_shuffle = _mm256_set_epi64x(0x0707070706060606, 0x0505050504040404, 0x0303030302020202, 0x0101010100000000);
    auto values128 = _mm_loadu_si128((const __m128i *)iq3nl_values);
    auto values = MM256_SET_M128I(values128, values128);
    values = _mm256_add_epi8(values, _mm256_set1_epi8(64));
    static const uint8_t k_shuff[32] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    auto shuff = _mm256_loadu_si256((const __m256i *)k_shuff);
#ifndef HAVE_FANCY_SIMD
    auto s_shuffle = _mm256_set_epi64x(0x0f0e0f0e0d0c0d0c, 0x0b0a0b0a09080908, 0x0706070605040504, 0x0302030201000100);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y] = {};
    __m256i qx[4];
    uint64_t stored_scales[8];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_iq3_k_r4 * iq3 = (const block_iq3_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq3[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto extra = _mm256_set1_epi64x(*(const uint64_t *)iq3[ibl].extra);
            auto slbits = _mm256_loadu_si256((const __m256i *)iq3[ibl].scales_l);
            auto sl1 = _mm256_add_epi8(_mm256_slli_epi16(_mm256_and_si256(slbits, m4), 1), _mm256_set1_epi8(1));
            auto sl2 = _mm256_add_epi8(_mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4), 1), _mm256_set1_epi8(1));
            auto sh = _mm256_set1_epi64x(((const uint64_t *)iq3[ibl].scales_h)[0]);
            auto sh1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sh, smask), smask), _mm256_set1_epi8(1));
            auto sh2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_srli_epi16(sh, 4), smask), smask), _mm256_set1_epi8(1));
            auto i8scales1 = _mm256_sign_epi8(sl1, sh1);
            auto i8scales2 = _mm256_sign_epi8(sl2, sh2);
            _mm256_storeu_si256((__m256i *)stored_scales+0, i8scales1);
            _mm256_storeu_si256((__m256i *)stored_scales+1, i8scales2);
            __m256i isum[nrc_y] = {};
            iq234_k_accum_mins(ibl, i8scales1, i8scales2, q8, shuff, isum, -64);
            for (int ib = 0; ib < QK_K/32; ++ib) {
#ifdef HAVE_FANCY_SIMD
                auto scales = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(stored_scales + ib)));
#else
                auto scales = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_set1_epi64x(stored_scales[ib])), s_shuffle);
#endif
                auto lb = _mm256_loadu_si256((const __m256i *)iq3[ibl].qs+ib);
                auto hbits = _mm_loadu_si128((const __m128i *)iq3[ibl].qh+ib);
                auto hb = MM256_SET_M128I(hbits, _mm_slli_epi16(hbits, 4));
                auto shift = _mm256_and_si256(ms, _mm256_slli_epi16(extra, 3)); extra = _mm256_srli_epi16(extra, 1);
                shift = _mm256_shuffle_epi8(shift, shift_shuffle);
                qx[0] = _mm256_or_si256(_mm256_and_si256(lb, m03),                       _mm256_and_si256(m04, _mm256_srli_epi16(hb, 2)));
                qx[1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 2), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 3)));
                qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 4), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 4)));
                qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 6), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 5)));
                qx[0] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[0], shift));
                qx[1] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[1], shift));
                qx[2] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[2], shift));
                qx[3] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[3], shift));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales, sumi));
#else
                    auto sumi1 = _mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00));
                    auto sumi2 = _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55));
                    auto sumi3 = _mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    auto sumi4 = _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(scales, sumi1), _mm256_madd_epi16(scales, sumi2)));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(scales, sumi3), _mm256_madd_epi16(scales, sumi4)));
#endif
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

template <int nrc_y>
static void mul_mat_iq4_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto m30 = _mm256_set1_epi8(0x30);
    auto m32 = _mm256_set1_epi8(32);
    auto ms  = _mm256_set1_epi8(4);
    auto shift_shuffle = _mm256_set_epi64x(0x0707070706060606, 0x0505050504040404, 0x0303030302020202, 0x0101010100000000);
#ifdef HAVE_FANCY_SIMD
    auto values = load_iq4nl_values_256();
    static const uint8_t k_shuff[32] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    auto shuff = _mm256_loadu_si256((const __m256i *)k_shuff);
#else
    auto s_shuffle = _mm256_set_epi64x(0x0f0e0f0e0d0c0d0c, 0x0b0a0b0a09080908, 0x0706070605040504, 0x0302030201000100);
    auto values128 = _mm_loadu_si128((const __m128i *)iq4k_values);
    auto values = MM256_SET_M128I(values128, values128);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y] = {};
    __m256i qx[4];
    uint64_t stored_scales[8];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_iq4_k_r4 * iq4 = (const block_iq4_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq4[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto extra = _mm256_set1_epi64x(*(const uint64_t *)iq4[ibl].extra);
            auto slbits = _mm256_loadu_si256((const __m256i *)iq4[ibl].scales_l);
            auto sl1 = _mm256_and_si256(slbits, m4);
            auto sl2 = _mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4);
            auto shbits = _mm_loadu_si128((const __m128i*)iq4[ibl].scales_h);
            auto sh = MM256_SET_M128I(_mm_srli_epi16(shbits, 2), shbits);
            auto i8scales1 = _mm256_sub_epi8(_mm256_or_si256(sl1, _mm256_and_si256(m30, _mm256_slli_epi16(sh, 4))), m32);
            auto i8scales2 = _mm256_sub_epi8(_mm256_or_si256(sl2, _mm256_and_si256(m30, sh)), m32);
            _mm256_storeu_si256((__m256i *)stored_scales+0, i8scales1);
            _mm256_storeu_si256((__m256i *)stored_scales+1, i8scales2);
            __m256i isum[nrc_y] = {};
#ifdef HAVE_FANCY_SIMD
            iq234_k_accum_mins(ibl, i8scales1, i8scales2, q8, shuff, isum, -128);
#endif
            for (int ib = 0; ib < QK_K/32; ++ib) {
#ifdef HAVE_FANCY_SIMD
                auto scales = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(stored_scales + ib)));
#else
                auto scales = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_set1_epi64x(stored_scales[ib])), s_shuffle);
#endif
                auto bits1 = _mm256_loadu_si256((const __m256i *)iq4[ibl].qs+2*ib+0);
                auto bits2 = _mm256_loadu_si256((const __m256i *)iq4[ibl].qs+2*ib+1);
                auto shift = _mm256_and_si256(ms, _mm256_slli_epi16(extra, 2)); extra = _mm256_srli_epi16(extra, 1);
                shift = _mm256_shuffle_epi8(shift, shift_shuffle);
                qx[0] = _mm256_add_epi8(shift, _mm256_shuffle_epi8(values, _mm256_and_si256(bits1, m4)));
                qx[1] = _mm256_add_epi8(shift, _mm256_shuffle_epi8(values, _mm256_and_si256(bits2, m4)));
                qx[2] = _mm256_add_epi8(shift, _mm256_shuffle_epi8(values, _mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4)));
                qx[3] = _mm256_add_epi8(shift, _mm256_shuffle_epi8(values, _mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4)));
#ifndef HAVE_FANCY_SIMD
                auto s1 = _mm256_sign_epi8(qx[0], qx[0]);
                auto s2 = _mm256_sign_epi8(qx[1], qx[1]);
                auto s3 = _mm256_sign_epi8(qx[2], qx[2]);
                auto s4 = _mm256_sign_epi8(qx[3], qx[3]);
#endif
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales, sumi));
#else
                    auto sumi1 = _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]));
                    auto sumi2 = _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1]));
                    auto sumi3 = _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]));
                    auto sumi4 = _mm256_maddubs_epi16(s4, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3]));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(scales, sumi1), _mm256_madd_epi16(scales, sumi2)));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(scales, sumi3), _mm256_madd_epi16(scales, sumi4)));
#endif
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

static inline __m256i prepare_5bit_quants(const __m256i * values, __m256i ql, __m256i qh, __m256i mask) {
    auto q5vl = _mm256_shuffle_epi8(values[0], ql);
    auto q5vh = _mm256_shuffle_epi8(values[1], ql);
#ifdef HAVE_FANCY_SIMD
    return _mm256_mask_blend_epi8(_mm256_cmpeq_epi8_mask(_mm256_and_si256(qh, mask), mask), q5vl, q5vh);
#else
    return _mm256_blendv_epi8(q5vl, q5vh, _mm256_cmpeq_epi8(_mm256_and_si256(qh, mask), mask));
#endif
}

template <int nrc_y>
static void mul_mat_iq5_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto m30 = _mm256_set1_epi8(0x30);
    auto m32 = _mm256_set1_epi8(32);
    auto ms  = _mm256_set1_epi8(2);
    auto shift_shuffle = _mm256_set_epi64x(0x0707070706060606, 0x0505050504040404, 0x0303030302020202, 0x0101010100000000);
    __m256i values[2];
    {
        auto val1 = _mm_loadu_si128((const __m128i *)iq5nl_values+0);
        auto val2 = _mm_loadu_si128((const __m128i *)iq5nl_values+1);
        values[0] = MM256_SET_M128I(val1, val1);
        values[1] = MM256_SET_M128I(val2, val2);
#ifdef HAVE_FANCY_SIMD
        values[0] = _mm256_sub_epi8(values[0], _mm256_set1_epi8(-128));
        values[1] = _mm256_sub_epi8(values[1], _mm256_set1_epi8(-128));
#endif
    }
#ifdef HAVE_FANCY_SIMD
    static const uint8_t k_shuff[32] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    auto shuff = _mm256_loadu_si256((const __m256i *)k_shuff);
#else
    auto s_shuffle = _mm256_set_epi64x(0x0f0e0f0e0d0c0d0c, 0x0b0a0b0a09080908, 0x0706070605040504, 0x0302030201000100);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y] = {};
    __m256i qx[4];
    uint64_t stored_scales[8];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_iq5_k_r4 * iq5 = (const block_iq5_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq5[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto extra = _mm256_set1_epi64x(*(const uint64_t *)iq5[ibl].extra);
            auto slbits = _mm256_loadu_si256((const __m256i *)iq5[ibl].scales_l);
            auto sl1 = _mm256_and_si256(slbits, m4);
            auto sl2 = _mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4);
            auto shbits = _mm_loadu_si128((const __m128i*)iq5[ibl].scales_h);
            auto sh = MM256_SET_M128I(_mm_srli_epi16(shbits, 2), shbits);
            auto i8scales1 = _mm256_sub_epi8(_mm256_or_si256(sl1, _mm256_and_si256(m30, _mm256_slli_epi16(sh, 4))), m32);
            auto i8scales2 = _mm256_sub_epi8(_mm256_or_si256(sl2, _mm256_and_si256(m30, sh)), m32);
            _mm256_storeu_si256((__m256i *)stored_scales+0, i8scales1);
            _mm256_storeu_si256((__m256i *)stored_scales+1, i8scales2);
            __m256i isum[nrc_y] = {};
#ifdef HAVE_FANCY_SIMD
            if constexpr (nrc_y == 1) {
                iq234_k_accum_mins(ibl, i8scales1, i8scales2, q8, shuff, isum, -128);
            } else {
                iq2345_k_accum_mins(ibl, i8scales1, i8scales2, q8, shuff, extra, isum, -128, 2);
            }
#endif
            for (int ib = 0; ib < QK_K/32; ++ib) {
#ifdef HAVE_FANCY_SIMD
                auto scales = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(stored_scales + ib)));
#else
                auto scales = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_set1_epi64x(stored_scales[ib])), s_shuffle);
#endif
                auto lbits1 = _mm256_loadu_si256((const __m256i *)iq5[ibl].qs+2*ib+0);
                auto lbits2 = _mm256_loadu_si256((const __m256i *)iq5[ibl].qs+2*ib+1);
                auto hbits  = _mm_loadu_si128((const __m128i *)iq5[ibl].qh+ib);
                auto hb     = MM256_SET_M128I(_mm_srli_epi16(hbits, 2), hbits);
                qx[0] = _mm256_and_si256(lbits1, m4);
                qx[1] = _mm256_and_si256(lbits2, m4);
                qx[2] = _mm256_and_si256(_mm256_srli_epi16(lbits1, 4), m4);
                qx[3] = _mm256_and_si256(_mm256_srli_epi16(lbits2, 4), m4);

                qx[0] = prepare_5bit_quants(values, qx[0], hb, _mm256_set1_epi8(0x01));
                qx[1] = prepare_5bit_quants(values, qx[1], hb, _mm256_set1_epi8(0x10));
                qx[2] = prepare_5bit_quants(values, qx[2], hb, _mm256_set1_epi8(0x02));
                qx[3] = prepare_5bit_quants(values, qx[3], hb, _mm256_set1_epi8(0x20));
#ifdef HAVE_FANCY_SIMD
                if constexpr (nrc_y == 1) {
                    auto shift = _mm256_and_si256(ms, _mm256_slli_epi16(extra, 1)); extra = _mm256_srli_epi16(extra, 1);
                    shift = _mm256_shuffle_epi8(shift, shift_shuffle);
                    qx[0] = _mm256_add_epi8(qx[0], shift);
                    qx[1] = _mm256_add_epi8(qx[1], shift);
                    qx[2] = _mm256_add_epi8(qx[2], shift);
                    qx[3] = _mm256_add_epi8(qx[3], shift);
                }
#else
                auto shift = _mm256_and_si256(ms, _mm256_slli_epi16(extra, 1)); extra = _mm256_srli_epi16(extra, 1);
                shift = _mm256_shuffle_epi8(shift, shift_shuffle);
                qx[0] = _mm256_add_epi8(qx[0], shift);
                qx[1] = _mm256_add_epi8(qx[1], shift);
                qx[2] = _mm256_add_epi8(qx[2], shift);
                qx[3] = _mm256_add_epi8(qx[3], shift);
                auto s1 = _mm256_sign_epi8(qx[0], qx[0]);
                auto s2 = _mm256_sign_epi8(qx[1], qx[1]);
                auto s3 = _mm256_sign_epi8(qx[2], qx[2]);
                auto s4 = _mm256_sign_epi8(qx[3], qx[3]);
#endif
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales, sumi));
#else
                    auto sumi1 = _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]));
                    auto sumi2 = _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1]));
                    auto sumi3 = _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]));
                    auto sumi4 = _mm256_maddubs_epi16(s4, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3]));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(scales, sumi1), _mm256_madd_epi16(scales, sumi2)));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(scales, sumi3), _mm256_madd_epi16(scales, sumi4)));
#endif
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

template <int nrc_y>
static void mul_mat_iq4_ks_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
#ifndef HAVE_FANCY_SIMD
    auto s_shuffle = _mm256_set_epi64x(0x0f0e0f0e0d0c0d0c, 0x0b0a0b0a09080908, 0x0706070605040504, 0x0302030201000100);
    auto values128 = _mm_loadu_si128((const __m128i *)iq4k_values);
    auto values = MM256_SET_M128I(values128, values128);
#else
    auto values = load_iq4nl_values_256();
#endif
    int nbl = n / QK_K;
    using helper_t = union { __m256i vec; uint32_t val[8]; };
#ifndef HAVE_FANCY_SIMD
    helper_t h, h_shift;
#else
    using helper512_t = union { __m512i vec; uint64_t val[8]; };
    helper_t h;
    helper512_t h_shift;
#endif
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto dptr = (const float *)((const char *)vx + (ix+0)*bx);
        const block_iq4_ks_r4 * iq4 = (const block_iq4_ks_r4 *)(dptr + 4);
        auto d4 = _mm_loadu_ps(dptr);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto scales = _mm256_loadu_si256((const __m256i *)iq4[ibl].scales);
            h.vec = _mm256_sub_epi8(_mm256_and_si256(scales, _mm256_set1_epi8(-2)), _mm256_set1_epi8(127));
#ifndef HAVE_FANCY_SIMD
            h_shift.vec = _mm256_slli_epi16(_mm256_and_si256(scales, _mm256_set1_epi8(1)), 2);
            {
                __m256 v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h.val[4])),       _mm_cvtepi8_epi32(_mm_set1_epi32(h.val[0])))),
                                          _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[4])), _mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[0])))));
                __m256 v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h.val[5])),       _mm_cvtepi8_epi32(_mm_set1_epi32(h.val[1])))),
                                          _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[5])), _mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[1])))));
                __m256 v3 = _mm256_mul_ps(_mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h.val[6])),       _mm_cvtepi8_epi32(_mm_set1_epi32(h.val[2])))),
                                          _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[6])), _mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[2])))));
                __m256 v4 = _mm256_mul_ps(_mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h.val[7])),       _mm_cvtepi8_epi32(_mm_set1_epi32(h.val[3])))),
                                          _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[7])), _mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[3])))));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto m8 = _mm256_loadu_ps((const float *)q8.y[iy][ibl].bsums);
                    acc[iy] = _mm256_fmadd_ps(v1, _mm256_shuffle_ps(m8, m8, 0x00), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(v2, _mm256_shuffle_ps(m8, m8, 0x55), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(v3, _mm256_shuffle_ps(m8, m8, 0xaa), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(v4, _mm256_shuffle_ps(m8, m8, 0xff), acc[iy]);
                }
            }
#else
            auto shift = _mm256_add_epi8(_mm256_set1_epi8(-64), _mm256_slli_epi16(_mm256_and_si256(scales, _mm256_set1_epi8(1)), 1));
            h_shift.vec = _mm512_mullo_epi16(_mm512_cvtepi8_epi16(shift), _mm512_cvtepi8_epi16(h.vec));
#endif
            for (int ib = 0; ib < QK_K/32; ++ib) {
#ifdef HAVE_FANCY_SIMD
                auto iscales = _mm256_cvtepi8_epi32(_mm_set1_epi32(h.val[ib]));
                auto ishifts = _mm256_cvtepi16_epi32(_mm_set1_epi64x(h_shift.val[ib]));
                auto scales_m = _mm256_cvtepi32_ps(ishifts);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    float m8 = ((const float *)q8.y[iy][ibl].bsums)[ib];
                    acc[iy] = _mm256_fmadd_ps(scales_m, _mm256_set1_ps(m8), acc[iy]);
                }
#endif
                auto bits1 = _mm256_loadu_si256((const __m256i *)iq4[ibl].qs+2*ib+0);
                auto bits2 = _mm256_loadu_si256((const __m256i *)iq4[ibl].qs+2*ib+1);
                qx[0] = _mm256_shuffle_epi8(values, _mm256_and_si256(bits1, m4));
                qx[1] = _mm256_shuffle_epi8(values, _mm256_and_si256(bits2, m4));
                qx[2] = _mm256_shuffle_epi8(values, _mm256_and_si256(_mm256_srli_epi16(bits1, 4), m4));
                qx[3] = _mm256_shuffle_epi8(values, _mm256_and_si256(_mm256_srli_epi16(bits2, 4), m4));
#ifndef HAVE_FANCY_SIMD
                auto iscales = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_set1_epi32(h.val[ib])), s_shuffle);
                auto s1 = _mm256_sign_epi8(qx[0], qx[0]);
                auto s2 = _mm256_sign_epi8(qx[1], qx[1]);
                auto s3 = _mm256_sign_epi8(qx[2], qx[2]);
                auto s4 = _mm256_sign_epi8(qx[3], qx[3]);
#endif
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(iscales, sumi));
#else
                    auto sumi1 = _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]));
                    auto sumi2 = _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1]));
                    auto sumi3 = _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]));
                    auto sumi4 = _mm256_maddubs_epi16(s4, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3]));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(iscales, sumi1), _mm256_madd_epi16(iscales, sumi2)));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(iscales, sumi3), _mm256_madd_epi16(iscales, sumi4)));
#endif
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(q8.scale(iy, ibl)), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, _mm_mul_ps(d4, sum));
        }
    }
}

template <int nrc_y>
static void mul_mat_iq5_ks_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    __m256i values[2];
    {
        auto val1 = _mm_loadu_si128((const __m128i *)iq5nl_values+0);
        auto val2 = _mm_loadu_si128((const __m128i *)iq5nl_values+1);
        values[0] = MM256_SET_M128I(val1, val1);
        values[1] = MM256_SET_M128I(val2, val2);
#ifdef HAVE_FANCY_SIMD
        values[0] = _mm256_sub_epi8(values[0], _mm256_set1_epi8(-128));
        values[1] = _mm256_sub_epi8(values[1], _mm256_set1_epi8(-128));
#endif
    }
    int nbl = n / QK_K;
    using helper_t = union { __m256i vec; uint32_t val[8]; };
#ifndef HAVE_FANCY_SIMD
    helper_t h, h_shift;
    auto s_shuffle = _mm256_set_epi64x(0x0f0e0f0e0d0c0d0c, 0x0b0a0b0a09080908, 0x0706070605040504, 0x0302030201000100);
#else
    using helper512_t = union { __m512i vec; uint64_t val[8]; };
    helper_t h;
    helper512_t h_shift;
#endif
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto dptr = (const float *)((const char *)vx + (ix+0)*bx);
        const block_iq5_ks_r4 * iq5 = (const block_iq5_ks_r4 *)(dptr + 4);
        auto d4 = _mm_loadu_ps(dptr);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto scales = _mm256_loadu_si256((const __m256i *)iq5[ibl].scales);
            h.vec = _mm256_sub_epi8(_mm256_and_si256(scales, _mm256_set1_epi8(-2)), _mm256_set1_epi8(127));
#ifndef HAVE_FANCY_SIMD
            h_shift.vec = _mm256_slli_epi16(_mm256_and_si256(scales, _mm256_set1_epi8(1)), 1);
            {
                __m256 v1 = _mm256_mul_ps(_mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h.val[4])),       _mm_cvtepi8_epi32(_mm_set1_epi32(h.val[0])))),
                                          _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[4])), _mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[0])))));
                __m256 v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h.val[5])),       _mm_cvtepi8_epi32(_mm_set1_epi32(h.val[1])))),
                                          _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[5])), _mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[1])))));
                __m256 v3 = _mm256_mul_ps(_mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h.val[6])),       _mm_cvtepi8_epi32(_mm_set1_epi32(h.val[2])))),
                                          _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[6])), _mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[2])))));
                __m256 v4 = _mm256_mul_ps(_mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h.val[7])),       _mm_cvtepi8_epi32(_mm_set1_epi32(h.val[3])))),
                                          _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[7])), _mm_cvtepi8_epi32(_mm_set1_epi32(h_shift.val[3])))));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto m8 = _mm256_loadu_ps((const float *)q8.y[iy][ibl].bsums);
                    acc[iy] = _mm256_fmadd_ps(v1, _mm256_shuffle_ps(m8, m8, 0x00), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(v2, _mm256_shuffle_ps(m8, m8, 0x55), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(v3, _mm256_shuffle_ps(m8, m8, 0xaa), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(v4, _mm256_shuffle_ps(m8, m8, 0xff), acc[iy]);
                }
            }
#else
            auto shift = _mm256_add_epi8(_mm256_set1_epi8(-64), _mm256_and_si256(scales, _mm256_set1_epi8(1)));
            h_shift.vec = _mm512_mullo_epi16(_mm512_cvtepi8_epi16(shift), _mm512_cvtepi8_epi16(h.vec));
#endif
            for (int ib = 0; ib < QK_K/32; ++ib) {
#ifdef HAVE_FANCY_SIMD
                auto iscales = _mm256_cvtepi8_epi32(_mm_set1_epi32(h.val[ib]));
                auto ishifts = _mm256_cvtepi16_epi32(_mm_set1_epi64x(h_shift.val[ib]));
                auto scales_m = _mm256_cvtepi32_ps(ishifts);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    float m8 = ((const float *)q8.y[iy][ibl].bsums)[ib];
                    acc[iy] = _mm256_fmadd_ps(scales_m, _mm256_set1_ps(m8), acc[iy]);
                }
#endif
                auto lbits1 = _mm256_loadu_si256((const __m256i *)iq5[ibl].qs+2*ib+0);
                auto lbits2 = _mm256_loadu_si256((const __m256i *)iq5[ibl].qs+2*ib+1);
                auto hbits  = _mm_loadu_si128((const __m128i *)iq5[ibl].qh+ib);
                auto hb     = MM256_SET_M128I(_mm_srli_epi16(hbits, 2), hbits);
                qx[0] = _mm256_and_si256(lbits1, m4);
                qx[1] = _mm256_and_si256(lbits2, m4);
                qx[2] = _mm256_and_si256(_mm256_srli_epi16(lbits1, 4), m4);
                qx[3] = _mm256_and_si256(_mm256_srli_epi16(lbits2, 4), m4);

                qx[0] = prepare_5bit_quants(values, qx[0], hb, _mm256_set1_epi8(0x01));
                qx[1] = prepare_5bit_quants(values, qx[1], hb, _mm256_set1_epi8(0x10));
                qx[2] = prepare_5bit_quants(values, qx[2], hb, _mm256_set1_epi8(0x02));
                qx[3] = prepare_5bit_quants(values, qx[3], hb, _mm256_set1_epi8(0x20));

#ifndef HAVE_FANCY_SIMD
                auto iscales = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_set1_epi32(h.val[ib])), s_shuffle);
                auto s1 = _mm256_sign_epi8(qx[0], qx[0]);
                auto s2 = _mm256_sign_epi8(qx[1], qx[1]);
                auto s3 = _mm256_sign_epi8(qx[2], qx[2]);
                auto s4 = _mm256_sign_epi8(qx[3], qx[3]);
#endif
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(iscales, sumi));
#else
                    auto sumi1 = _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]));
                    auto sumi2 = _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1]));
                    auto sumi3 = _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]));
                    auto sumi4 = _mm256_maddubs_epi16(s4, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3]));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(iscales, sumi1), _mm256_madd_epi16(iscales, sumi2)));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(iscales, sumi3), _mm256_madd_epi16(iscales, sumi4)));
#endif
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(q8.scale(iy, ibl)), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, _mm_mul_ps(d4, sum));
        }
    }
}


template <typename Dequantizer> void set_functions(std::array<mul_mat_t, IQK_MAX_NY>& funcs) {
#ifdef HAVE_FANCY_SIMD
    if constexpr (std::is_same_v<Dequantizer, DequantizerIQ2KS> ||
                  std::is_same_v<Dequantizer, DequantizerIQ2KL> ||
                  std::is_same_v<Dequantizer, DequantizerIQ3KS> ||
                  std::is_same_v<Dequantizer, DequantizerIQ4KS> ||
                  std::is_same_v<Dequantizer, DequantizerIQ5KS>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_iqX_k_q8_K_AVX512_new, Dequantizer, funcs)
    } else if constexpr (std::is_same_v<Dequantizer, DequantizerIQ2K>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_AVX512, Dequantizer, funcs);
        funcs[0] = mul_mat_qX_K_q8_K_AVX512_1<Dequantizer>;
    } else {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_iqX_k_q8_K_AVX512, Dequantizer, funcs);
    }
#else
    if constexpr (std::is_same_v<Dequantizer, DequantizerIQ2K>||
                  std::is_same_v<Dequantizer, DequantizerIQ3K>||
                  std::is_same_v<Dequantizer, DequantizerIQ4K>||
                  std::is_same_v<Dequantizer, DequantizerIQ5K>||
                  std::is_same_v<Dequantizer, DequantizerIQ6K>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qY_K_q8_K_T, Dequantizer, funcs);
    } else {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, Dequantizer, funcs);
    }

#endif
}

inline float convert_to_q8_k_r8(int k, float d0, const __m256i * qx, const int16_t * scales, uint32_t * block, int8_t * q8_k) {
    auto max_i16 = _mm256_setzero_si256();
    __m256i qs[16];
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        qs[2*ib32+0] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qx[ib32]));
        qs[2*ib32+1] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qx[ib32], 1));
        qs[2*ib32+0] = _mm256_mullo_epi16(qs[2*ib32+0], _mm256_set1_epi16(scales[2*ib32+0]));
        qs[2*ib32+1] = _mm256_mullo_epi16(qs[2*ib32+1], _mm256_set1_epi16(scales[2*ib32+1]));
        max_i16 = _mm256_max_epi16(max_i16, _mm256_sign_epi16(qs[2*ib32+0], qs[2*ib32+0]));
        max_i16 = _mm256_max_epi16(max_i16, _mm256_sign_epi16(qs[2*ib32+1], qs[2*ib32+1]));
    }
    auto max_q32 = _mm256_cvtepi16_epi32(_mm_max_epi16(_mm256_castsi256_si128(max_i16), _mm256_extracti128_si256(max_i16, 1)));
    auto imax4 = _mm_max_epi32(_mm256_castsi256_si128(max_q32), _mm256_extracti128_si256(max_q32, 1));
    auto max4  = _mm_cvtepi32_ps(imax4);
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    bool needs_scaling = true;
    float dnew = _mm_cvtss_f32(max4) * d0;
    if (dnew < 1.f) {
        dnew = 1.f; needs_scaling = false;
    }
    auto scale = _mm256_set1_ps(std::abs(dnew) > 1e-9f ? 1/dnew : 0.f);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        if (needs_scaling) {
            auto i0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(qs[2*ib32+0]));
            auto i1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(qs[2*ib32+0], 1));
            auto i2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(qs[2*ib32+1]));
            auto i3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(qs[2*ib32+1], 1));
            i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i0)), _MM_ROUND_NEAREST));
            i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i1)), _MM_ROUND_NEAREST));
            i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i2)), _MM_ROUND_NEAREST));
            i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i3)), _MM_ROUND_NEAREST));
            i0 = _mm256_packs_epi32(i0, i1);
            i2 = _mm256_packs_epi32(i2, i3);
            i0 = _mm256_packs_epi16(i0, i2);
            i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256((__m256i *)block, i0);
        } else {
            // 0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31
            auto i0 = _mm256_packs_epi16(qs[2*ib32+0], qs[2*ib32+1]);
            auto i0_l = _mm256_castsi256_si128(i0);
            auto i0_h = _mm256_extracti128_si256(i0, 1);
            _mm_storeu_si128((__m128i *)block+0, _mm_unpacklo_epi64(i0_l, i0_h));
            _mm_storeu_si128((__m128i *)block+1, _mm_unpackhi_epi64(i0_l, i0_h));
        }
        auto qs = (uint32_t *)q8_k + 64*ib32;
        for (int l = 0; l < 8; ++l) {
            qs[8*l + k] = block[l];
        }
    }
    return dnew;
}

void iqk_convert_iq2_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq2_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values;
    {
        auto v = _mm_loadl_epi64((const __m128i *)iq2nl_values);
        values = MM256_SET_M128I(v, v);
    }

    ggml_half dh[8];
    float     dnew[8];
    uint32_t  block[8];
    int16_t   ls[16];

    __m256i  xv[8];

    auto ml = _mm256_set1_epi8(0x03);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const ggml_half * dptr = (const ggml_half *)((const char *)vx + (ix+k)*bx);
            dh[k] = dptr[0];
            x8[k] = (const block_iq2_ks *)(dptr + 1);
        }
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                auto extra = x8[k][i].extra;
                for (int i128 = 0; i128 < 2; ++i128) {
                    ls[8*i128+0] = ls[8*i128+1] = ((x8[k][i].scales[2*i128+0] & 0xf) | ((extra >> 4) & 0x10)) - 16;
                    ls[8*i128+2] = ls[8*i128+3] = ((x8[k][i].scales[2*i128+0] >>  4) | ((extra >> 5) & 0x10)) - 16;
                    ls[8*i128+4] = ls[8*i128+5] = ((x8[k][i].scales[2*i128+1] & 0xf) | ((extra >> 6) & 0x10)) - 16;
                    ls[8*i128+6] = ls[8*i128+7] = ((x8[k][i].scales[2*i128+1] >>  4) | ((extra >> 7) & 0x10)) - 16;
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+i128);
                    xv[4*i128+0] = _mm256_and_si256(bits, ml);
                    xv[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), ml);
                    xv[4*i128+2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), ml);
                    xv[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), ml);
                    xv[4*i128+0] = _mm256_add_epi8(xv[4*i128+0], _mm256_set1_epi8((extra << 2) & 0x04));
                    xv[4*i128+1] = _mm256_add_epi8(xv[4*i128+1], _mm256_set1_epi8((extra << 1) & 0x04));
                    xv[4*i128+2] = _mm256_add_epi8(xv[4*i128+2], _mm256_set1_epi8((extra >> 0) & 0x04));
                    xv[4*i128+3] = _mm256_add_epi8(xv[4*i128+3], _mm256_set1_epi8((extra >> 1) & 0x04));
                    xv[4*i128+0] = _mm256_shuffle_epi8(values, xv[4*i128+0]);
                    xv[4*i128+1] = _mm256_shuffle_epi8(values, xv[4*i128+1]);
                    xv[4*i128+2] = _mm256_shuffle_epi8(values, xv[4*i128+2]);
                    xv[4*i128+3] = _mm256_shuffle_epi8(values, xv[4*i128+3]);
                    extra >>= 4;
                }
                dnew[k] = convert_to_q8_k_r8(k, 1.f/125, xv, ls, block, y[i].qs);
            }
            auto vd = _mm256_mul_ps(_mm256_loadu_ps(dnew), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh)));
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(vd, _MM_ROUND_NEAREST));
        }
        y += nb;
    }
}

void iqk_convert_iq2_kl_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq2_kl * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values[4];
    {
        static const int8_t k_values[64] = {
            -63, -63, -40, -40, -40, -40, -23, -23, -23, -23, -23, -10, -10, -10, -10, 1, 1, 1, 1, 1, 13, 13, 13, 13, 13, 28, 28, 28, 28, 28, 47, 47,
            -23, 13, -63, -10, 13, 47, -40, -23, 1, 13, 28, -63, 1, 13, 47, -23, -10, 1, 13, 28, -40, -23, -10, 1, 13, -63, -23, 1, 28, 47, -23, 13,
        };
        for (int k = 0; k < 4; ++k) {
            auto v = _mm_loadu_si128((const __m128i *)k_values + k);
            values[k] = MM256_SET_M128I(v, v);
        }
    }

    ggml_half dh[8];
    float     dnew[8];
    uint32_t  block[8];
    int16_t   ls[16];

    __m256i  xv[8];
    __m256i  ql[2];
    __m256i  mask[2];

    uint32_t sl32;
    const auto sl8 = (const int8_t *)&sl32;

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const ggml_half * dptr = (const ggml_half *)((const char *)vx + (ix+k)*bx);
            dh[k] = dptr[0];
            x8[k] = (const block_iq2_kl *)(dptr + 1);
        }
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                uint32_t aux32;
                std::memcpy(&aux32, x8[k][i].scales_l, 4);
                auto sh = x8[k][i].scales_h;
                auto hbits128 = _mm_loadu_si128((const __m128i *)x8[k][i].qh);
                auto hbits = MM256_SET_M128I(_mm_srli_epi16(hbits128, 1), hbits128);
                //auto sl = _mm_and_si128(_mm_cvtepu8_epi16(_mm_srlv_epi32(_mm_set1_epi32(aux32), _mm_set_epi32(0, 0, 4, 0))), _mm_set1_epi16(0xf));
                for (int i128 = 0; i128 < 2; ++i128) {
                    sl32 = aux32 & 0x0f0f0f0f;
                    ls[8*i128+0] = ls[8*i128+1] = (sl8[0] | ((sh << 4) & 0x30)) - 32;
                    ls[8*i128+2] = ls[8*i128+3] = (sl8[1] | ((sh << 2) & 0x30)) - 32;
                    ls[8*i128+4] = ls[8*i128+5] = (sl8[2] | ((sh >> 0) & 0x30)) - 32;
                    ls[8*i128+6] = ls[8*i128+7] = (sl8[3] | ((sh >> 2) & 0x30)) - 32;
                    aux32 >>= 4; sh >>= 8;
                    {
                        auto b1 = _mm_loadu_si128((const __m128i *)x8[k][i].qs+2*i128+0);
                        auto b2 = _mm_loadu_si128((const __m128i *)x8[k][i].qs+2*i128+1);
                        ql[0] = _mm256_and_si256(_mm256_set1_epi8(0xf), MM256_SET_M128I(_mm_srli_epi16(b1, 4), b1));
                        ql[1] = _mm256_and_si256(_mm256_set1_epi8(0xf), MM256_SET_M128I(_mm_srli_epi16(b2, 4), b2));
                    }
                    mask[0] = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, _mm256_set1_epi8(0x1)), _mm256_set1_epi8(0x1));
                    mask[1] = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, _mm256_set1_epi8(0x4)), _mm256_set1_epi8(0x4));
                    for (int k = 0; k < 2; ++k) {
                        auto v0 = _mm256_shuffle_epi8(values[0], ql[k]);
                        auto v1 = _mm256_shuffle_epi8(values[1], ql[k]);
                        auto v2 = _mm256_shuffle_epi8(values[2], ql[k]);
                        auto v3 = _mm256_shuffle_epi8(values[3], ql[k]);
                        auto q1 = _mm256_or_si256(_mm256_and_si256(mask[k], v1), _mm256_andnot_si256(mask[k], v0));
                        auto q2 = _mm256_or_si256(_mm256_and_si256(mask[k], v3), _mm256_andnot_si256(mask[k], v2));
                        auto q1l = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(q1));
                        auto q1h = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(q1, 1));
                        auto q2l = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(q2));
                        auto q2h = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(q2, 1));
                        xv[4*i128+2*k+0] = _mm256_or_si256(q1l, _mm256_slli_epi16(q2l, 8));
                        xv[4*i128+2*k+1] = _mm256_or_si256(q1h, _mm256_slli_epi16(q2h, 8));
                    }
                    hbits = _mm256_srli_epi16(hbits, 4);
                }
                dnew[k] = convert_to_q8_k_r8(k, 1.f/125, xv, ls, block, y[i].qs);
            }
            auto vd = _mm256_mul_ps(_mm256_loadu_ps(dnew), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh)));
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(vd, _MM_ROUND_NEAREST));
        }
        y += nb;
    }
}

void iqk_convert_iq2_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq2_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values;
    {
        auto v = _mm_loadl_epi64((const __m128i *)iq2nl_values);
        values = MM256_SET_M128I(v, v);
    }

    __m256i  xv[8];
    uint32_t block[8];

    const __m128i scale_shuffle = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);

    union { __m256i vec; int16_t val[16]; } helper;

    auto ml = _mm256_set1_epi8(0x03);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq2_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                uint64_t aux64; std::memcpy(&aux64, x8[k][i].scales, 8);
                auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
                scl = _mm_add_epi8(scl, _mm_set1_epi8(-8));
                helper.vec = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scl, scale_shuffle));
                auto extra = x8[k][i].extra;
                for (int i128 = 0; i128 < 2; ++i128) {
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+i128);
                    xv[4*i128+0] = _mm256_and_si256(bits, ml);
                    xv[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), ml);
                    xv[4*i128+2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), ml);
                    xv[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), ml);
                    auto shift1 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x02) << 1), _mm_set1_epi8((extra & 0x01) << 2));
                    auto shift2 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x08) >> 1), _mm_set1_epi8((extra & 0x04) >> 0));
                    auto shift3 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x20) >> 3), _mm_set1_epi8((extra & 0x10) >> 2));
                    auto shift4 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x80) >> 5), _mm_set1_epi8((extra & 0x40) >> 4));
                    xv[4*i128+0] = _mm256_add_epi8(xv[4*i128+0], shift1);
                    xv[4*i128+1] = _mm256_add_epi8(xv[4*i128+1], shift2);
                    xv[4*i128+2] = _mm256_add_epi8(xv[4*i128+2], shift3);
                    xv[4*i128+3] = _mm256_add_epi8(xv[4*i128+3], shift4);
                    xv[4*i128+0] = _mm256_shuffle_epi8(values, xv[4*i128+0]);
                    xv[4*i128+1] = _mm256_shuffle_epi8(values, xv[4*i128+1]);
                    xv[4*i128+2] = _mm256_shuffle_epi8(values, xv[4*i128+2]);
                    xv[4*i128+3] = _mm256_shuffle_epi8(values, xv[4*i128+3]);
                    extra >>= 8;
                }
                float dnew = convert_to_q8_k_r8(k, 1.f/120, xv, helper.val, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
}

void iqk_convert_iq3_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq3_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values;
    {
        auto v = _mm_loadu_si128((const __m128i *)iq3nl_values);
        values = MM256_SET_M128I(v, v);
    }

    __m256i  xv[8];
    uint32_t block[8];

    constexpr static uint8_t k_shuff[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    const __m128i sign_mask = _mm_set_epi64x(0x8080404020201010, 0x0808040402020101);
    const __m128i hshuff = _mm_loadu_si128((const __m128i*)k_shuff);
    const __m128i scale_shuffle = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);

    union { __m256i vec; int16_t val[16]; } helper;

    auto ml = _mm256_set1_epi8(0x03);
    auto hmask  = _mm256_set1_epi8(4);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq3_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                uint64_t aux64; std::memcpy(&aux64, x8[k][i].scales_l, 8);
                auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
                scl = _mm_add_epi8(_mm_slli_epi16(scl, 1), _mm_set1_epi8(1));
                auto sc_signs = _mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi16(x8[k][i].scales_h), sign_mask), sign_mask);
                auto sch   = _mm_shuffle_epi8(_mm_or_si128(sc_signs, _mm_set1_epi8(1)), hshuff);
                helper.vec = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(_mm_sign_epi8(scl, sch), scale_shuffle));
                auto extra = x8[k][i].extra;
                auto hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                for (int i128 = 0; i128 < 2; ++i128) {
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+i128);
                    xv[4*i128+0] = _mm256_and_si256(bits, ml);
                    xv[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), ml);
                    xv[4*i128+2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), ml);
                    xv[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), ml);
                    xv[4*i128+0] = _mm256_or_si256(xv[4*i128+0], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), hmask));
                    xv[4*i128+1] = _mm256_or_si256(xv[4*i128+1], _mm256_and_si256(_mm256_slli_epi16(hbits, 1), hmask));
                    xv[4*i128+2] = _mm256_or_si256(xv[4*i128+2], _mm256_and_si256(hbits, hmask));
                    xv[4*i128+3] = _mm256_or_si256(xv[4*i128+3], _mm256_and_si256(_mm256_srli_epi16(hbits, 1), hmask));
                    auto shift1 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x02) << 2), _mm_set1_epi8((extra & 0x01) << 3));
                    auto shift2 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x08) << 0), _mm_set1_epi8((extra & 0x04) << 1));
                    auto shift3 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x20) >> 2), _mm_set1_epi8((extra & 0x10) >> 1));
                    auto shift4 = MM256_SET_M128I(_mm_set1_epi8((extra & 0x80) >> 4), _mm_set1_epi8((extra & 0x40) >> 3));
                    xv[4*i128+0] = _mm256_add_epi8(xv[4*i128+0], shift1);
                    xv[4*i128+1] = _mm256_add_epi8(xv[4*i128+1], shift2);
                    xv[4*i128+2] = _mm256_add_epi8(xv[4*i128+2], shift3);
                    xv[4*i128+3] = _mm256_add_epi8(xv[4*i128+3], shift4);
                    xv[4*i128+0] = _mm256_shuffle_epi8(values, xv[4*i128+0]);
                    xv[4*i128+1] = _mm256_shuffle_epi8(values, xv[4*i128+1]);
                    xv[4*i128+2] = _mm256_shuffle_epi8(values, xv[4*i128+2]);
                    xv[4*i128+3] = _mm256_shuffle_epi8(values, xv[4*i128+3]);
                    hbits = _mm256_srli_epi16(hbits, 4);
                    extra >>= 8;
                }
                float dnew = convert_to_q8_k_r8(k, 1.f/127, xv, helper.val, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
}

void iqk_convert_iq3_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq3_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values;
    {
        auto v = _mm_loadu_si128((const __m128i *)iq3nl_values);
        values = MM256_SET_M128I(v, v);
    }

    ggml_half drow[8];
    float dnew[8];
    int16_t ls[16];

    __m256i xv[8];
    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const ggml_half * dptr = (const ggml_half *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq3_ks *)(dptr + 1);
        }
        auto vd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)drow));
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                auto hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                auto extra = x8[k][i].extra;
                uint8_t extra_v = extra >> 8;
                for (int j = 0; j < 4; ++j) {
                    ls[2*j+0] = ls[2*j+1] = ((x8[k][i].scales[j] & 0xf) | ((extra << 4) & 0x10)) - 16;
                    ls[2*j+8] = ls[2*j+9] = ((x8[k][i].scales[j] >>  4) | ((extra << 0) & 0x10)) - 16;
                    extra >>= 1;
                }
                for (int i128 = 0; i128 < QK_K/128; ++i128) {
                    auto lbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs + i128);
                    for (int j = 0; j < 4; ++j) {
                        xv[4*i128+j] = _mm256_or_si256(_mm256_and_si256(lbits, _mm256_set1_epi8(3)), _mm256_and_si256(_mm256_slli_epi16(hbits, 2), _mm256_set1_epi8(4)));
                        xv[4*i128+j] = _mm256_add_epi8(xv[4*i128+j], _mm256_set1_epi8((extra_v & 1) << 3));
                        xv[4*i128+j] = _mm256_shuffle_epi8(values, xv[4*i128+j]);
                        extra_v >>= 1;
                        lbits = _mm256_srli_epi16(lbits, 2);
                        hbits = _mm256_srli_epi16(hbits, 1);
                    }
                }
                dnew[k] = convert_to_q8_k_r8(k, 1.f/127, xv, ls, block, y[i].qs);
            }
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(_mm256_mul_ps(vd, _mm256_loadu_ps(dnew)), _MM_ROUND_NEAREST));
        }
        y += nb;
    }
}

void iqk_convert_iq4_kss_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq4_kss * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values[2];
    {
        auto v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
        values[0] = MM256_SET_M128I(v1, v1);
        values[1] = MM256_SET_M128I(v2, v2);
    }

    float drow[8];
    float dnew[8];
    int16_t ls[16];

    __m256i xv[8];
    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq4_kss *)(dptr + 1);
        }
        auto vd = _mm256_loadu_ps(drow);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    auto val = _mm_loadu_si128((const __m128i *)x8[k][i].qs+ib32);
                    auto val_q = _mm_and_si128(val, _mm_set1_epi32(0xfffefffe));
                    val_q = _mm_xor_si128(val_q, _mm_srli_epi16(val_q, 1));
                    xv[ib32] = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(val_q, 4), val_q), _mm256_set1_epi8(0xf));
                    auto q4 = x8[k][i].qs + 4*ib32;
                    uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
                    uint8_t s8 = (s32 | (s32 >> 15)) & 0xff;
                    //auto val_s = _mm_madd_epi16(_mm_and_si128(val, _mm_set1_epi32(0x00010001)), _mm_set1_epi64x(0x0008000400020001));
                    ls[2*ib32+0] = ls[2*ib32+1] = ((s8 & 254) - 127);
                    xv[ib32] = _mm256_shuffle_epi8(values[s8 & 1], xv[ib32]);
                }
                dnew[k] = convert_to_q8_k_r8(k, 1.f/127, xv, ls, block, y[i].qs);
            }
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(_mm256_mul_ps(vd, _mm256_loadu_ps(dnew)), _MM_ROUND_NEAREST));
        }
        y += nb;
    }
}

void iqk_convert_iq4_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq4_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values[2];
    {
        auto v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
        values[0] = MM256_SET_M128I(v1, v1);
        values[1] = MM256_SET_M128I(v2, v2);
    }

    float drow[8];
    float dnew[8];
    int16_t ls[16];

    __m256i xv[8];
    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq4_ks *)(dptr + 1);
        }
        auto vd = _mm256_loadu_ps(drow);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    ls[2*ib32+0] = (x8[k][i].scales[ib32] & 254) - 127;
                    ls[2*ib32+1] = ls[2*ib32+0];
                    auto aux128 = _mm_loadu_si128((const __m128i *)x8[k][i].qs+ib32);
                    xv[ib32] = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(aux128, 4), aux128), _mm256_set1_epi8(0xf));
                    xv[ib32] = _mm256_shuffle_epi8(values[x8[k][i].scales[ib32] & 1], xv[ib32]);
                }
                dnew[k] = convert_to_q8_k_r8(k, 1.f/127, xv, ls, block, y[i].qs);
            }
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(_mm256_mul_ps(vd, _mm256_loadu_ps(dnew)), _MM_ROUND_NEAREST));
        }
        y += nb;
    }
}

void iqk_convert_iq4_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq4_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values[4];
    {
        auto v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
        values[0] = MM256_SET_M128I(v1, v1);
        values[1] = MM256_SET_M128I(v1, v2);
        values[2] = MM256_SET_M128I(v2, v1);
        values[3] = MM256_SET_M128I(v2, v2);
    }

    __m256i  xv[8];
    uint32_t block[8];
    int16_t  ls[16];

    //auto hshuff = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);

    //union { __m256i vec; int16_t val[16]; } helper;

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq4_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                auto extra = x8[k][i].extra;
                //uint64_t aux64;
                //memcpy(&aux64, x8[k][i].scales_l, 8);
                //auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
                //const uint32_t aux32 = *(const uint32_t *)x8[k][i].scales_h;
                //auto aux = _mm_and_si128(_mm_set_epi32(aux32 >> 2, aux32, aux32 << 2, aux32 << 4), _mm_set1_epi8(0x30));
                //auto sch = _mm_shuffle_epi8(aux, hshuff);
                //aux = _mm_add_epi8(_mm_or_si128(scl, sch), _mm_set1_epi8(-32));
                //helper.vec = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(aux, hshuff));
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const uint8_t sh = x8[k][i].scales_h[ib32/2] >> 4*(ib32%2);
                    ls[2*ib32+0] = ((x8[k][i].scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32;
                    ls[2*ib32+1] = ((x8[k][i].scales_l[ib32] >>  4) | ((sh << 2) & 0x30)) - 32;
                    auto bits = _mm_loadu_si128((const __m128i *)x8[k][i].qs+ib32);
                    xv[ib32]  = _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(bits, 4), bits), _mm256_set1_epi8(0xf));
                    xv[ib32]  = _mm256_shuffle_epi8(values[extra & 3], xv[ib32]); extra >>= 2;
                }
                //float dnew = convert_to_q8_k_r8(k, 1.f/127, xv, helper.val, block, y[i].qs);
                float dnew = convert_to_q8_k_r8(k, 1.f/127, xv, ls, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
}

void iqk_convert_iq5_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq5_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values[2];
    {
        auto v1 = _mm_loadu_si128((const __m128i *)iq5nl_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq5nl_values+1);
        values[0] = MM256_SET_M128I(v1, v1);
        values[1] = MM256_SET_M128I(v2, v2);
    }

    float drow[8];
    float dnew[8];
    int16_t ls[16];

    __m256i xv[8];
    uint32_t block[8];

    auto mh = _mm256_set1_epi8(-128); // to avoid stupid warning about 0x80 overflowing

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq5_ks *)(dptr + 1);
        }
        auto vd = _mm256_loadu_ps(drow);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                auto hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                for (int ib64 = 0; ib64 < 4; ++ib64) {
                    ls[4*ib64+0] = (x8[k][i].scales[2*ib64+0] & 254) - 127;
                    ls[4*ib64+1] = ls[4*ib64+0];
                    ls[4*ib64+2] = (x8[k][i].scales[2*ib64+1] & 254) - 127;
                    ls[4*ib64+3] = ls[4*ib64+2];
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+ib64);
                    xv[2*ib64+0] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[2*ib64+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    auto qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 7), mh);
                    auto q5vl = _mm256_or_si256(xv[2*ib64+0], qh);
                    auto q5vh = _mm256_or_si256(xv[2*ib64+0], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+0] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 6), mh);
                    q5vl = _mm256_or_si256(xv[2*ib64+1], qh);
                    q5vh = _mm256_or_si256(xv[2*ib64+1], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+1] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    auto shift1 = _mm256_set1_epi8((x8[k][i].scales[2*ib64+0] & 1) << 1);
                    auto shift2 = _mm256_set1_epi8((x8[k][i].scales[2*ib64+1] & 1) << 1);
                    xv[2*ib64+0] = _mm256_add_epi8(xv[2*ib64+0], shift1);
                    xv[2*ib64+1] = _mm256_add_epi8(xv[2*ib64+1], shift2);
                    hbits = _mm256_srli_epi16(hbits, 2);
                }
                dnew[k] = convert_to_q8_k_r8(k, 1.f/127, xv, ls, block, y[i].qs);
            }
            _mm_storeu_si128((__m128i *)y[i].d, _mm256_cvtps_ph(_mm256_mul_ps(vd, _mm256_loadu_ps(dnew)), _MM_ROUND_NEAREST));
        }
        y += nb;
    }
}

void iqk_convert_iq5_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq5_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values[2];
    {
        auto v1 = _mm_loadu_si128((const __m128i *)iq5nl_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq5nl_values+1);
        values[0] = MM256_SET_M128I(v1, v1);
        values[1] = MM256_SET_M128I(v2, v2);
    }

    __m256i  xv[8];
    uint32_t block[8];
    int16_t  ls[16];

    auto mh = _mm256_set1_epi8(-128); // to avoid stupid warning about 0x80 overflowing

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq5_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                auto extra = x8[k][i].extra;
                auto hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                for (int ib64 = 0; ib64 < 4; ++ib64) {
                    ls[4*ib64+0] = ((x8[k][i].scales_l[2*ib64+0] & 0xf) | ((x8[k][i].scales_h[ib64] << 4) & 0x30)) - 32;
                    ls[4*ib64+1] = ((x8[k][i].scales_l[2*ib64+0] >>  4) | ((x8[k][i].scales_h[ib64] << 2) & 0x30)) - 32;
                    ls[4*ib64+2] = ((x8[k][i].scales_l[2*ib64+1] & 0xf) | ((x8[k][i].scales_h[ib64] >> 0) & 0x30)) - 32;
                    ls[4*ib64+3] = ((x8[k][i].scales_l[2*ib64+1] >>  4) | ((x8[k][i].scales_h[ib64] >> 2) & 0x30)) - 32;
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+ib64);
                    xv[2*ib64+0] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[2*ib64+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    auto qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 7), mh);
                    auto q5vl = _mm256_or_si256(xv[2*ib64+0], qh);
                    auto q5vh = _mm256_or_si256(xv[2*ib64+0], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+0] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 6), mh);
                    q5vl = _mm256_or_si256(xv[2*ib64+1], qh);
                    q5vh = _mm256_or_si256(xv[2*ib64+1], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+1] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    auto shift1 = MM256_SET_M128I(_mm_set1_epi8((extra & 2) << 0), _mm_set1_epi8((extra & 1) << 1));
                    auto shift2 = MM256_SET_M128I(_mm_set1_epi8((extra & 8) >> 2), _mm_set1_epi8((extra & 4) >> 1));
                    xv[2*ib64+0] = _mm256_add_epi8(xv[2*ib64+0], shift1);
                    xv[2*ib64+1] = _mm256_add_epi8(xv[2*ib64+1], shift2);
                    hbits = _mm256_srli_epi16(hbits, 2);
                    extra >>= 4;
                }
                float dnew = convert_to_q8_k_r8(k, 1.f/127, xv, ls, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
}

void iqk_convert_iq5_k_q8_0_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq5_k * x8[8];

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    __m256i values[2];
    {
        auto v1 = _mm_loadu_si128((const __m128i *)iq5nl_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq5nl_values+1);
        values[0] = MM256_SET_M128I(v1, v1);
        values[1] = MM256_SET_M128I(v2, v2);
    }

    __m256i  xv[8];
    uint32_t block[8];
    int16_t  ls[16];
    float    all_s[64];

    auto mh = _mm256_set1_epi8(-128); // to avoid stupid warning about 0x80 overflowing

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq5_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                auto extra = x8[k][i].extra;
                auto hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh);
                for (int ib64 = 0; ib64 < 4; ++ib64) {
                    ls[4*ib64+0] = ((x8[k][i].scales_l[2*ib64+0] & 0xf) | ((x8[k][i].scales_h[ib64] << 4) & 0x30)) - 32;
                    ls[4*ib64+1] = ((x8[k][i].scales_l[2*ib64+0] >>  4) | ((x8[k][i].scales_h[ib64] << 2) & 0x30)) - 32;
                    ls[4*ib64+2] = ((x8[k][i].scales_l[2*ib64+1] & 0xf) | ((x8[k][i].scales_h[ib64] >> 0) & 0x30)) - 32;
                    ls[4*ib64+3] = ((x8[k][i].scales_l[2*ib64+1] >>  4) | ((x8[k][i].scales_h[ib64] >> 2) & 0x30)) - 32;
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+ib64);
                    xv[2*ib64+0] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[2*ib64+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    auto qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 7), mh);
                    auto q5vl = _mm256_or_si256(xv[2*ib64+0], qh);
                    auto q5vh = _mm256_or_si256(xv[2*ib64+0], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+0] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    qh = _mm256_and_si256(_mm256_slli_epi16(hbits, 6), mh);
                    q5vl = _mm256_or_si256(xv[2*ib64+1], qh);
                    q5vh = _mm256_or_si256(xv[2*ib64+1], _mm256_xor_si256(qh, mh));
                    xv[2*ib64+1] = _mm256_or_si256(_mm256_shuffle_epi8(values[0], q5vl), _mm256_shuffle_epi8(values[1], q5vh));
                    auto shift1 = MM256_SET_M128I(_mm_set1_epi8((extra & 2) << 0), _mm_set1_epi8((extra & 1) << 1));
                    auto shift2 = MM256_SET_M128I(_mm_set1_epi8((extra & 8) >> 2), _mm_set1_epi8((extra & 4) >> 1));
                    xv[2*ib64+0] = _mm256_add_epi8(xv[2*ib64+0], shift1);
                    xv[2*ib64+1] = _mm256_add_epi8(xv[2*ib64+1], shift2);
                    hbits = _mm256_srli_epi16(hbits, 2);
                    extra >>= 4;
                }
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    // We have two blocks of 16 with different scales
                    // We multiply the quants with the scales, find the max value, and convert to 8-bit quants with a single block scale.
                    auto q16_l = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(xv[ib32]));
                    auto q16_h = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xv[ib32], 1));
                    q16_l = _mm256_mullo_epi16(q16_l, _mm256_set1_epi16(ls[2*ib32+0]));
                    q16_h = _mm256_mullo_epi16(q16_h, _mm256_set1_epi16(ls[2*ib32+1]));
                    auto abs_q16_l = _mm256_sign_epi16(q16_l, q16_l);
                    auto abs_q16_h = _mm256_sign_epi16(q16_h, q16_h);
                    auto max_q16 = _mm256_max_epi16(abs_q16_l, abs_q16_h);
                    auto max_q32 = _mm256_cvtepi16_epi32(_mm_max_epi16(_mm256_castsi256_si128(max_q16), _mm256_extracti128_si256(max_q16, 1)));
                    auto imax4 = _mm_max_epi32(_mm256_castsi256_si128(max_q32), _mm256_extracti128_si256(max_q32, 1));
                    auto max4  = _mm_cvtepi32_ps(imax4);
                    max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
                    max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
                    float max = _mm_cvtss_f32(max4) / 127;
                    all_s[8*ib32+k] = d*max;
                    if (max > 1e-9f) {
                        auto scale = _mm256_set1_ps(1/max);
                        auto i0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16_l));
                        auto i1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16_l, 1));
                        auto i2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16_h));
                        auto i3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16_h, 1));
                        i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i0)), _MM_ROUND_NEAREST));
                        i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i1)), _MM_ROUND_NEAREST));
                        i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i2)), _MM_ROUND_NEAREST));
                        i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(i3)), _MM_ROUND_NEAREST));
                        i0 = _mm256_packs_epi32(i0, i1);
                        i2 = _mm256_packs_epi32(i2, i3);
                        i0 = _mm256_packs_epi16(i0, i2);
                        i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
                        _mm256_storeu_si256((__m256i *)block, i0);
                    } else {
                        _mm256_storeu_si256((__m256i *)block, _mm256_setzero_si256());
                    }
                    auto qs = (uint32_t *)y[ib32].qs;
                    for (int l = 0; l < 4; ++l) {
                        qs[8*l + k +  0] = block[l + 0];
                        qs[8*l + k + 32] = block[l + 4];
                    }
                }
            }
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                _mm_storeu_si128((__m128i *)y[ib32].d, _mm256_cvtps_ph(_mm256_loadu_ps(all_s + 8*ib32), _MM_FROUND_TO_NEAREST_INT));
            }
            y += QK_K/32;
        }
    }
}

void iqk_convert_iq6_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq6_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    __m256i values[4];
    for (int k = 0; k < 4; ++k) {
        auto values128 = _mm_loadu_si128((const __m128i *)iq6nl_values + k);
        values[k] = MM256_SET_M128I(values128, values128);
    }

    __m256i  xv[8];
    uint32_t block[8];

    union { __m256i vec; int16_t val[16]; } helper;

    auto mh1 = _mm256_set1_epi8(1);
    auto mh2 = _mm256_set1_epi8(2);
    auto mh3 = _mm256_set1_epi8(3);

    auto make_one = [&values, &mh1, &mh2, &mh3] (__m256i l, __m256i hbits) {
        auto mask4 = _mm256_cmpeq_epi8(_mm256_and_si256(hbits, mh3), mh3);
        auto h1 = _mm256_andnot_si256(mask4, hbits);
        auto mask2 = _mm256_cmpeq_epi8(_mm256_and_si256(h1, mh1), mh1);
        auto mask3 = _mm256_cmpeq_epi8(_mm256_and_si256(h1, mh2), mh2);
        auto mask1 = _mm256_andnot_si256(_mm256_or_si256(mask4, _mm256_or_si256(mask2, mask3)), _mm256_set1_epi8(-1)); // 0xff;
        return _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(mask1, _mm256_shuffle_epi8(values[0], l)),
                                               _mm256_and_si256(mask2, _mm256_shuffle_epi8(values[1], l))),
                               _mm256_or_si256(_mm256_and_si256(mask3, _mm256_shuffle_epi8(values[2], l)),
                                               _mm256_and_si256(mask4, _mm256_shuffle_epi8(values[3], l))));
    };

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq6_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                helper.vec = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)x8[k][i].scales));
                auto extra = x8[k][i].extra;
                for (int i128 = 0; i128 < 2; ++i128) {
                    auto hbits = _mm256_loadu_si256((const __m256i *)x8[k][i].qh+i128);
                    auto bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+2*i128+0);
                    xv[4*i128+0] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[4*i128+1] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    bits = _mm256_loadu_si256((const __m256i *)x8[k][i].qs+2*i128+1);
                    xv[4*i128+2] = _mm256_and_si256(bits, _mm256_set1_epi8(0xf));
                    xv[4*i128+3] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), _mm256_set1_epi8(0xf));
                    for (int k = 0; k < 4; ++k) {
                        xv[4*i128+k] = make_one(xv[4*i128+k], hbits);
                        hbits = _mm256_srli_epi16(hbits, 2);
                    }
                    auto shift1 = MM256_SET_M128I(_mm_set1_epi8((extra >> 1) & 1), _mm_set1_epi8((extra >> 0) & 1));
                    auto shift2 = MM256_SET_M128I(_mm_set1_epi8((extra >> 3) & 1), _mm_set1_epi8((extra >> 2) & 1));
                    auto shift3 = MM256_SET_M128I(_mm_set1_epi8((extra >> 5) & 1), _mm_set1_epi8((extra >> 4) & 1));
                    auto shift4 = MM256_SET_M128I(_mm_set1_epi8((extra >> 7) & 1), _mm_set1_epi8((extra >> 6) & 1));
                    xv[4*i128+0] = _mm256_add_epi8(xv[4*i128+0], shift1);
                    xv[4*i128+1] = _mm256_add_epi8(xv[4*i128+1], shift2);
                    xv[4*i128+2] = _mm256_add_epi8(xv[4*i128+2], shift3);
                    xv[4*i128+3] = _mm256_add_epi8(xv[4*i128+3], shift4);
                    extra >>= 8;
                }
                float dnew = convert_to_q8_k_r8(k, 1.f/127, xv, helper.val, block, y[i].qs);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
}

} // namespace

bool iqk_convert_iqk_quants_q80_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    if (n%QK_K != 0 || nrc_x%8 != 0) return false;
    switch (ggml_type(type)) {
        case GGML_TYPE_IQ2_KS : iqk_convert_iq2_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ2_K  : iqk_convert_iq2_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ2_KL : iqk_convert_iq2_kl_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ3_KS : iqk_convert_iq3_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ3_K  : iqk_convert_iq3_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_KSS: iqk_convert_iq4_kss_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_KS : iqk_convert_iq4_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_K  : iqk_convert_iq4_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ5_KS : iqk_convert_iq5_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ5_K  : iqk_convert_iq5_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ6_K  : iqk_convert_iq6_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        default: return false;
    }
    return true;
}

bool iqk_set_kernels_iqk_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {

    auto etypeA = ggml_type(typeA);
    auto expected_type_B = etypeA == GGML_TYPE_IQ4_KS_R4 || etypeA == GGML_TYPE_IQ5_KS_R4 ? GGML_TYPE_Q8_K32 : GGML_TYPE_Q8_K;
    if (ne00%QK_K != 0 || ggml_type(typeB) != expected_type_B) {
        return false;
    }

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_IQ2_KS:
            set_functions<DequantizerIQ2KS>(kernels);
            break;
        case GGML_TYPE_IQ2_K:
            set_functions<DequantizerIQ2K>(kernels);
            break;
        case GGML_TYPE_IQ2_KL:
            set_functions<DequantizerIQ2KL>(kernels);
#ifdef HAVE_FANCY_SIMD
            func16 = mul_mat_iqX_k_q8_K_AVX512_new<DequantizerIQ2KL, 16>;
#endif
            break;
        case GGML_TYPE_IQ3_KS:
            set_functions<DequantizerIQ3KS>(kernels);
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
        case GGML_TYPE_IQ2_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_k_r4_q8_k, kernels);
            break;
        case GGML_TYPE_IQ3_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq3_k_r4_q8_k, kernels);
#ifdef HAVE_FANCY_SIMD
            func16 = mul_mat_iq3_k_r4_q8_k<16>;
#endif
            break;
        case GGML_TYPE_IQ4_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_k_r4_q8_k, kernels);
            func16  = mul_mat_iq4_k_r4_q8_k<16>;
            break;
        case GGML_TYPE_IQ4_KS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_ks_r4_q8_k, kernels);
#ifndef HAVE_FANCY_SIMD
            // For some reason Zen4 does not like this particular function
            func16 = mul_mat_iq4_ks_r4_q8_k<16>;
#endif
            break;
        case GGML_TYPE_IQ5_KS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq5_ks_r4_q8_k, kernels);
#ifndef HAVE_FANCY_SIMD
            // For some reason Zen4 does not like this particular function
            func16 = mul_mat_iq5_ks_r4_q8_k<16>;
#endif
            break;
        case GGML_TYPE_IQ5_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq5_k_r4_q8_k, kernels);
            func16  = mul_mat_iq5_k_r4_q8_k<16>;
            break;
        default:
            return false;
    }

    return true;

}

#else
// ----------------------------------------- __aarch64__ ---------------------------------------------

namespace {

inline int32x4x4_t make_wider(const int16x8x2_t& scales16) {
    int32x4x4_t scales = {
        vmovl_s16(vget_low_s16 (scales16.val[0])),
        vmovl_s16(vget_high_s16(scales16.val[0])),
        vmovl_s16(vget_low_s16 (scales16.val[1])),
        vmovl_s16(vget_high_s16(scales16.val[1])),
    };
    return scales;
}

inline int32x4x4_t make_wider_8(const int8x16_t& scales8) {
    int16x8x2_t scales16{vmovl_s8(vget_low_s8(scales8)), vmovl_s8(vget_high_s8(scales8))};
    return make_wider(scales16);
}

template <typename Q8>
inline void accum_mins_16(const int16x8x2_t& mins, const Q8& q8, float32x4_t * acc, int i, float c) {
    for (int iy = 0; iy < Q8::nrc_y; ++iy) {
        auto q8s = q8.load_bsums(iy, i);
        int32x4_t b1 = vmull_s16(vget_low_s16 (mins.val[0]), vget_low_s16 (q8s.val[0]));
        int32x4_t b2 = vmull_s16(vget_high_s16(mins.val[0]), vget_high_s16(q8s.val[0]));
        int32x4_t b3 = vmull_s16(vget_low_s16 (mins.val[1]), vget_low_s16 (q8s.val[1]));
        int32x4_t b4 = vmull_s16(vget_high_s16(mins.val[1]), vget_high_s16(q8s.val[1]));
        float32x4_t prod = vcvtq_f32_s32(vaddq_s32(vaddq_s32(b1, b2), vaddq_s32(b3, b4)));
        acc[iy] = vmlaq_f32(acc[iy], prod, vdupq_n_f32(c*q8.scale(iy, i)));
    }
}

struct Scale16Extra {
    template <typename Q8>
    static inline int32x4x4_t new_block(int i, float d, uint16_t extra, uint8_t val,
            const int8x16_t& scales8, const Q8& q8, float32x4_t * acc) {
        uint8x16_t e8 = vreinterpretq_u8_u16(vdupq_n_u16(extra));
        e8 = vceqq_u8(vandq_u8(e8, emask), emask);
        e8 = vqtbl1q_u8(vandq_u8(e8, vdupq_n_u8(val)), eshuff);
        int16x8x2_t extra16 = {vmull_s8(vget_low_s8 (e8), vget_low_s8 (scales8)),
                               vmull_s8(vget_high_s8(e8), vget_high_s8(scales8))};
        accum_mins_16(extra16, q8, acc, i, d);
        return make_wider_8(scales8);
    }

    constexpr static uint32x4_t emask  = {0x02020101, 0x08080404, 0x20201010, 0x80804040};
    constexpr static uint32x4_t eshuff = {0x06040200, 0x0e0c0a08, 0x07050301, 0x0f0d0b09};
};

// Note: on ARM_NEON we cannot use the values shifted into the uint8_t range because
//       the ARM_NEON only has vdotq_s32 or vdotq_u32, where both operands need to
//       be signed or unsigned. As the Q8_K quants are signed, we need to have the
//       iq4_s quants also signed. We can only use unsigned values in k-quants
//       because they are all within the valid int8_t range.
struct DequantizerIQ4K final : public BaseDequantizer<block_iq4_k> {
    DequantizerIQ4K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc), values(vld1q_s8(iq4k_values)) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        return Scale16Extra::new_block(i, d, x[i].extra, 4, make_scales(x[i].scales_l, x[i].scales_h), q8, acc);
    }
    inline void prepare(int i, int j) {
        bits.prepare16(x[i].qs+64*j);
        for (int k = 0; k < 4; ++k) {
            bits.b1.val[k] = vqtbl1q_s8(values, bits.b1.val[k]);
            bits.b2.val[k] = vqtbl1q_s8(values, bits.b2.val[k]);
        }
    }
    inline int8x16_t make_scales(const uint8_t * scales_l, const uint8_t * scales_h) const {
        uint8x8_t aux = vld1_u8(scales_l);
        uint8x16_t scl8 = vandq_u8(vcombine_u8(aux, vshr_n_u8(aux, 4)), vdupq_n_u8(0xf));
        const uint32_t * aux32 = (const uint32_t *)scales_h;
        uint32x4_t sch_32 = {aux32[0] << 4, aux32[0] << 2, aux32[0], aux32[0] >> 2};
        uint8x16_t sch8 = vandq_u8(vreinterpretq_u8_u32(sch_32), vdupq_n_u8(0x30));
        int8x16_t scales8 = vorrq_u8(scl8, vqtbl1q_u8(sch8, hshuff));
        return vaddq_s8(vqtbl1q_s8(scales8, hshuff), vdupq_n_s8(-32));
    }

    Q4bits bits;
    const int8x16_t values;
    const uint8x16_t hshuff = vreinterpretq_u8_u32(uint32x4_t{0x09010800, 0x0b030a02, 0x0d050c04, 0x0f070e06});

};

struct DequantizerIQ5K final : public BaseDequantizer<block_iq5_k> {
    DequantizerIQ5K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc), values(vld1q_s8_x2(iq5nl_values)) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        hbits = vld1q_u8_x2(x[i].qh); // hbits.val[0] holds 0....15, 32...47, 64...79, 96...111, 128...143, 160...175, 192...207, 224...239
                                      // hbits.val[1] holds 16...31, 48...63, 80...95, 112..127, 144...159, 176...191, 208...223, 240...255
        return Scale16Extra::new_block(i, d, x[i].extra, 2, make_scales(x[i].scales_l, x[i].scales_h), q8, acc);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs+64*j);
        if (j == 1) {
            for (int k = 0; k < 2; ++k) hbits.val[k] = vshrq_n_u8(hbits.val[k], 4);
        }
        bits.b1.val[0] = vorrq_u8(bits.b1.val[0], vandq_u8(vshlq_n_u8(hbits.val[0], 4), hm));
        bits.b1.val[1] = vorrq_u8(bits.b1.val[1], vandq_u8(vshlq_n_u8(hbits.val[1], 4), hm));
        bits.b1.val[2] = vorrq_u8(bits.b1.val[2], vandq_u8(vshlq_n_u8(hbits.val[0], 3), hm));
        bits.b1.val[3] = vorrq_u8(bits.b1.val[3], vandq_u8(vshlq_n_u8(hbits.val[1], 3), hm));
        bits.b2.val[0] = vorrq_u8(bits.b2.val[0], vandq_u8(vshlq_n_u8(hbits.val[0], 2), hm));
        bits.b2.val[1] = vorrq_u8(bits.b2.val[1], vandq_u8(vshlq_n_u8(hbits.val[1], 2), hm));
        bits.b2.val[2] = vorrq_u8(bits.b2.val[2], vandq_u8(vshlq_n_u8(hbits.val[0], 1), hm));
        bits.b2.val[3] = vorrq_u8(bits.b2.val[3], vandq_u8(vshlq_n_u8(hbits.val[1], 1), hm));
        for (int k = 0; k < 4; ++k) {
            bits.b1.val[k] = vqtbl2q_s8(values, bits.b1.val[k]);
            bits.b2.val[k] = vqtbl2q_s8(values, bits.b2.val[k]);
        }
    }
    inline int8x16_t make_scales(const uint8_t * scales_l, const uint8_t * scales_h) const {
        uint8x8_t aux = vld1_u8(scales_l);
        uint8x16_t scl8 = vandq_u8(vcombine_u8(aux, vshr_n_u8(aux, 4)), vdupq_n_u8(0xf));
        const uint32_t * aux32 = (const uint32_t *)scales_h;
        uint32x4_t sch_32 = {aux32[0] << 4, aux32[0] << 2, aux32[0], aux32[0] >> 2};
        uint8x16_t sch8 = vandq_u8(vreinterpretq_u8_u32(sch_32), vdupq_n_u8(0x30));
        int8x16_t scales8 = vorrq_u8(scl8, vqtbl1q_u8(sch8, hshuff));
        return vaddq_s8(vqtbl1q_s8(scales8, hshuff), vdupq_n_s8(-32));
    }

    Q4bits bits;
    const int8x16x2_t values;
    const uint8x16_t hshuff = vreinterpretq_u8_u32(uint32x4_t{0x09010800, 0x0b030a02, 0x0d050c04, 0x0f070e06});
    const uint8x16_t hm = vdupq_n_u8(0x10);
    uint8x16x2_t hbits;

};

struct DequantizerIQ6K final : public BaseDequantizer<block_iq6_k> {
    DequantizerIQ6K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc), values(vld1q_s8_x4(iq6nl_values)) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        return Scale16Extra::new_block(i, d, x[i].extra, 1, vld1q_s8(x[i].scales), q8, acc);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs+64*j);
        auto hbits = vld1q_u8_x2(x[i].qh + 32*j);
        bits.b1.val[0] = vorrq_u8(bits.b1.val[0], vandq_u8(vshlq_n_u8(hbits.val[0], 4), hm));
        bits.b1.val[1] = vorrq_u8(bits.b1.val[1], vandq_u8(vshlq_n_u8(hbits.val[1], 4), hm));
        bits.b1.val[2] = vorrq_u8(bits.b1.val[2], vandq_u8(vshlq_n_u8(hbits.val[0], 2), hm));
        bits.b1.val[3] = vorrq_u8(bits.b1.val[3], vandq_u8(vshlq_n_u8(hbits.val[1], 2), hm));
        bits.b2.val[0] = vorrq_u8(bits.b2.val[0], vandq_u8(hbits.val[0], hm));
        bits.b2.val[1] = vorrq_u8(bits.b2.val[1], vandq_u8(hbits.val[1], hm));
        bits.b2.val[2] = vorrq_u8(bits.b2.val[2], vandq_u8(vshrq_n_u8(hbits.val[0], 2), hm));
        bits.b2.val[3] = vorrq_u8(bits.b2.val[3], vandq_u8(vshrq_n_u8(hbits.val[1], 2), hm));
        for (int k = 0; k < 4; ++k) {
            bits.b1.val[k] = vqtbl4q_s8(values, bits.b1.val[k]);
            bits.b2.val[k] = vqtbl4q_s8(values, bits.b2.val[k]);
        }
    }

    Q4bits bits;
    const int8x16x4_t values;
    const uint8x16_t hm = vdupq_n_u8(0x30);

};

struct DequantizerIQ2K final : public BaseDequantizer<block_iq2_k> {
    DequantizerIQ2K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        return Scale16Extra::new_block(i, d, x[i].extra, 5, make_scales(x[i].scales), q8, acc);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs+32*j);
        for (int k = 0; k < 4; ++k) {
            bits.b1.val[k] = vqtbl1q_s8(values, bits.b1.val[k]);
            bits.b2.val[k] = vqtbl1q_s8(values, bits.b2.val[k]);
        }
    }
    inline int8x16_t make_scales(const uint8_t * scales_l) const {
        uint8x8_t aux = vld1_u8(scales_l);
        uint8x16_t scl8 = vandq_u8(vcombine_u8(aux, vshr_n_u8(aux, 4)), vdupq_n_u8(0xf));
        int8x16_t scales = vaddq_s8(vreinterpretq_s8_u8(scl8), vdupq_n_s8(-8));
        return vqtbl1q_s8(scales, hshuff);
    }

    Q2bits bits;
    const int8x16_t values = vreinterpretq_s8_u64(vdupq_n_u64(0x000000001101f3e1));
    const uint8x16_t hshuff = vreinterpretq_u8_u32(uint32x4_t{0x09010800, 0x0b030a02, 0x0d050c04, 0x0f070e06});

};

struct DequantizerIQ3K final : public BaseDequantizer<block_iq3_k> {
    DequantizerIQ3K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        return Scale16Extra::new_block(i, d, x[i].extra, 4, make_scales(x[i].scales_h, x[i].scales_l), q8, acc);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs+32*j);
        if (j == 0) {
            hbits = vld1q_u8_x2(x[i].qh);
        }
        else {
            hbits.val[0] = vshrq_n_u8(hbits.val[0], 4);
            hbits.val[1] = vshrq_n_u8(hbits.val[1], 4);
        }
        bits.b1.val[0] = vorrq_u8(bits.b1.val[0], vandq_u8(vshlq_n_u8(hbits.val[0], 2), hmask));
        bits.b1.val[1] = vorrq_u8(bits.b1.val[1], vandq_u8(vshlq_n_u8(hbits.val[1], 2), hmask));
        bits.b1.val[2] = vorrq_u8(bits.b1.val[2], vandq_u8(vshlq_n_u8(hbits.val[0], 1), hmask));
        bits.b1.val[3] = vorrq_u8(bits.b1.val[3], vandq_u8(vshlq_n_u8(hbits.val[1], 1), hmask));
        bits.b2.val[0] = vorrq_u8(bits.b2.val[0], vandq_u8(hbits.val[0], hmask));
        bits.b2.val[1] = vorrq_u8(bits.b2.val[1], vandq_u8(hbits.val[1], hmask));
        bits.b2.val[2] = vorrq_u8(bits.b2.val[2], vandq_u8(vshrq_n_u8(hbits.val[0], 1), hmask));
        bits.b2.val[3] = vorrq_u8(bits.b2.val[3], vandq_u8(vshrq_n_u8(hbits.val[1], 1), hmask));
        for (int k = 0; k < 4; ++k) {
            bits.b1.val[k] = vqtbl1q_s8(values, bits.b1.val[k]);
            bits.b2.val[k] = vqtbl1q_s8(values, bits.b2.val[k]);
        }
    }
    inline int8x16_t make_scales(uint16_t sign_bits, const uint8_t * scales_l) const {
        uint8x8_t aux = vld1_u8(scales_l);
        uint8x16_t scl8 = vandq_u8(vcombine_u8(aux, vshr_n_u8(aux, 4)), vdupq_n_u8(0xf));
        int8x16_t scales = vaddq_s8(vreinterpretq_s8_u8(vshlq_n_u8(scl8, 1)), vdupq_n_s8(1));
        uint8x16_t signs = vceqq_u8(vandq_u8(vreinterpretq_u8_u16(vdupq_n_u16(sign_bits)), sign_mask), sign_mask);
        signs = vorrq_u8(signs, vdupq_n_u8(1));
        // scales are 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
        // signs  are 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
        scales = vmulq_s8(scales, vreinterpretq_s8_u8(vqtbl1q_u8(signs, sign_shuffle)));
        return vqtbl1q_s8(scales, hshuff);
    }
    inline static uint8x16_t load_sign_shuffle() {
        static uint8_t k_shuff[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
        return vld1q_u8(k_shuff);
    }

    Q2bits bits;
    uint8x16x2_t hbits;
    const int8x16_t values = vreinterpretq_s8_u64(vdupq_n_u64(0x2f1c0d01f6e9d8c1));
    const uint8x16_t hshuff = vreinterpretq_u8_u32(uint32x4_t{0x09010800, 0x0b030a02, 0x0d050c04, 0x0f070e06});
    const uint8x16_t hmask = vdupq_n_u8(4);
    const uint8x16_t sign_mask = vreinterpretq_u8_u64(uint64x2_t{0x0808040402020101, 0x8080404020201010});
    const uint8x16_t sign_shuffle = load_sign_shuffle();

};

struct DequantizerIQ3KS final : public BaseDequantizer<block_iq3_ks, true, true> {

    DequantizerIQ3KS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc), values(load_values()) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        (void)q8;
        (void)acc;
        uint32_t aux32; std::memcpy(&aux32, x[i].scales, 4);
        auto scl8 = vand_s8(vreinterpret_s8_u32(uint32x2_t{aux32, aux32 >> 4}), vdup_n_s8(0xf));
        auto sch8 = vdup_n_u8(x[i].extra & 0xff);
        sch8 = vand_u8(vceq_u8(vand_u8(sch8, shmask), shmask), vdup_n_u8(16));
        scl8 = vsub_s8(vadd_s8(scl8, vreinterpret_s8_u8(sch8)), vdup_n_s8(16));
        auto scales16 = vmovl_s8(scl8);
        int32x4x2_t scales = {vmovl_s16(vget_low_s16(scales16)), vmovl_s16(vget_high_s16(scales16))};
        return scales;
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs+32*j);
        if (j == 0) {
            hbits = vld1q_u8_x2(x[i].qh);
        }
        else {
            hbits.val[0] = vshrq_n_u8(hbits.val[0], 4);
            hbits.val[1] = vshrq_n_u8(hbits.val[1], 4);
        }
        uint8_t extra = x[i].extra >> (8 + 4*j);
        bits.b1.val[0] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b1.val[0], vandq_u8(vshlq_n_u8(hbits.val[0], 2), hmask)));
        bits.b1.val[1] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b1.val[1], vandq_u8(vshlq_n_u8(hbits.val[1], 2), hmask))); extra >>= 1;
        bits.b1.val[2] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b1.val[2], vandq_u8(vshlq_n_u8(hbits.val[0], 1), hmask)));
        bits.b1.val[3] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b1.val[3], vandq_u8(vshlq_n_u8(hbits.val[1], 1), hmask))); extra >>= 1;
        bits.b2.val[0] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b2.val[0], vandq_u8(hbits.val[0], hmask)));
        bits.b2.val[1] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b2.val[1], vandq_u8(hbits.val[1], hmask))); extra >>= 1;
        bits.b2.val[2] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b2.val[2], vandq_u8(vshrq_n_u8(hbits.val[0], 1), hmask)));
        bits.b2.val[3] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b2.val[3], vandq_u8(vshrq_n_u8(hbits.val[1], 1), hmask)));
    }
    static int8x16x2_t load_values() {
        auto v1 = vld1_s8(iq3nl_values + 0);
        auto v2 = vld1_s8(iq3nl_values + 8);
        return { vcombine_s8(v1, v1), vcombine_s8(v2, v2) };
    }

    Q2bits bits;
    uint8x16x2_t hbits;
    const int8x16x2_t values;
    const uint8x16_t hmask = vdupq_n_u8(4);
    const uint8x8_t shmask = vreinterpret_u8_u64(vdup_n_u64(0x8040201008040201));
};

struct DequantizerIQ4KS final : public BaseDequantizer<block_iq4_ks, true> {

    DequantizerIQ4KS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc), values(vld1q_s8_x2(iq4k_values)) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        (void)q8;
        (void)acc;
        auto scales16 = vaddq_s16(vreinterpretq_s16_u16(vandq_u16(vmovl_u8(vld1_u8(x[i].scales)), mask)), m127);
        int32x4x2_t scales = {vmovl_s16(vget_low_s16(scales16)), vmovl_s16(vget_high_s16(scales16))};
        return scales;
    }
    inline void prepare(int i, int j) {
        bits.prepare16(x[i].qs+64*j);
        const uint32_t * scales32 = (const uint32_t *)x[i].scales;
        uint32_t aux32 = scales32[j] & 0x01010101;
        const uint8_t * aux8 = (const uint8_t *)&aux32;
        for (int k = 0; k < 4; ++k) {
            bits.b1.val[k] = vreinterpretq_u8_s8(vqtbl1q_s8(values.val[aux8[k/2+0]], bits.b1.val[k]));
            bits.b2.val[k] = vreinterpretq_u8_s8(vqtbl1q_s8(values.val[aux8[k/2+2]], bits.b2.val[k]));
        }
    }

    Q4bits bits;
    const int8x16x2_t values;
    const uint16x8_t  mask = vdupq_n_u16(254);
    const int16x8_t   m127 = vdupq_n_s16(-127);
};

struct DequantizerIQ5KS final : public BaseDequantizer<block_iq5_ks, true> {
    DequantizerIQ5KS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc),
        values(vld1q_s8_x4(iq5nl_values)) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        (void)q8;
        (void)acc;
        auto sas8 = vld1_u8(x[i].scales);
        auto scales16 = vaddq_s16(vreinterpretq_s16_u16(vandq_u16(vmovl_u8(sas8), mask)), m127);
        hbits = vld1q_u8_x2(x[i].qh);
        sas = vcombine_u8(sas8, sas8);
        sas = vshlq_n_u8(vandq_u8(sas, vdupq_n_u8(1)), 5);
        int32x4x2_t scales = {vmovl_s16(vget_low_s16(scales16)), vmovl_s16(vget_high_s16(scales16))};
        return scales;
    }

    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs+64*j);
        if (j == 1) {
            for (int k = 0; k < 2; ++k) hbits.val[k] = vshrq_n_u8(hbits.val[k], 4);
        }
        auto shift = vdupq_n_u8((x[i].scales[4*j+0] & 1) << 5);
        bits.b1.val[0] = vaddq_u8(shift, vorrq_u8(bits.b1.val[0], vandq_u8(vshlq_n_u8(hbits.val[0], 4), hm)));
        bits.b1.val[1] = vaddq_u8(shift, vorrq_u8(bits.b1.val[1], vandq_u8(vshlq_n_u8(hbits.val[1], 4), hm)));
        shift = vdupq_n_u8((x[i].scales[4*j+1] & 1) << 5);
        bits.b1.val[2] = vaddq_u8(shift, vorrq_u8(bits.b1.val[2], vandq_u8(vshlq_n_u8(hbits.val[0], 3), hm)));
        bits.b1.val[3] = vaddq_u8(shift, vorrq_u8(bits.b1.val[3], vandq_u8(vshlq_n_u8(hbits.val[1], 3), hm)));
        for (int k = 0; k < 4; ++k) bits.b1.val[k] = vqtbl4q_s8(values, bits.b1.val[k]);
        shift = vdupq_n_u8((x[i].scales[4*j+2] & 1) << 5);
        bits.b2.val[0] = vaddq_u8(shift, vorrq_u8(bits.b2.val[0], vandq_u8(vshlq_n_u8(hbits.val[0], 2), hm)));
        bits.b2.val[1] = vaddq_u8(shift, vorrq_u8(bits.b2.val[1], vandq_u8(vshlq_n_u8(hbits.val[1], 2), hm)));
        shift = vdupq_n_u8((x[i].scales[4*j+3] & 1) << 5);
        bits.b2.val[2] = vaddq_u8(shift, vorrq_u8(bits.b2.val[2], vandq_u8(vshlq_n_u8(hbits.val[0], 1), hm)));
        bits.b2.val[3] = vaddq_u8(shift, vorrq_u8(bits.b2.val[3], vandq_u8(vshlq_n_u8(hbits.val[1], 1), hm)));
        for (int k = 0; k < 4; ++k) bits.b2.val[k] = vqtbl4q_s8(values, bits.b2.val[k]);
    }

    Q4bits bits;
    const int8x16x4_t values;
    const uint8x16_t hm = vdupq_n_u8(0x10);
    const uint16x8_t  mask = vdupq_n_u16(254);
    const int16x8_t   m127 = vdupq_n_s16(-127);
    uint8x16x2_t hbits;
    uint8x16_t   sas;

};

struct DequantizerIQ4KSS final : public BaseDequantizer<block_iq4_kss, true> {

    DequantizerIQ4KSS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc), values(vld1q_s8_x2(iq4k_values)) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        (void)q8;
        (void)acc;
        auto q4bits_1 = vld1q_u16_x4((const uint16_t *)x[i].qs);
        q4bits_2 = vld1q_u16_x4((const uint16_t *)x[i].qs + 32);
        for (int k = 0; k < 4; ++k) {
            aux[k+0] = vaddvq_s16(vshlq_s16(vandq_u16(q4bits_1.val[k], m1), shift));
            aux[k+4] = vaddvq_s16(vshlq_s16(vandq_u16(q4bits_2.val[k], m1), shift));
            q4bits_1.val[k] = vandq_u16(q4bits_1.val[k], bmask);
            q4bits_1.val[k] = veorq_u16(q4bits_1.val[k], vshrq_n_u16(q4bits_1.val[k], 1));
            q4bits_2.val[k] = vandq_u16(q4bits_2.val[k], bmask);
            q4bits_2.val[k] = veorq_u16(q4bits_2.val[k], vshrq_n_u16(q4bits_2.val[k], 1));
        }
        make_quants(q4bits_1, bits, aux);
        auto scales16 = vld1q_s16(aux);
        scales16 = vaddq_s16(vandq_s16(scales16, mask), m127);
        int32x4x2_t scales = {vmovl_s16(vget_low_s16(scales16)), vmovl_s16(vget_high_s16(scales16))};
        return scales;
    }
    inline void make_quants(uint16x8x4_t& q4bits, Q4bits& bits, const int16_t * aux) const {
        bits.b1.val[0] = vqtbl1q_s8(values.val[aux[0] & 1], vandq_u8(q4bits.val[0], bits.m4b));
        bits.b1.val[1] = vqtbl1q_s8(values.val[aux[0] & 1], vshrq_n_u8(q4bits.val[0], 4));
        bits.b1.val[2] = vqtbl1q_s8(values.val[aux[1] & 1], vandq_u8(q4bits.val[1], bits.m4b));
        bits.b1.val[3] = vqtbl1q_s8(values.val[aux[1] & 1], vshrq_n_u8(q4bits.val[1], 4));
        bits.b2.val[0] = vqtbl1q_s8(values.val[aux[2] & 1], vandq_u8(q4bits.val[2], bits.m4b));
        bits.b2.val[1] = vqtbl1q_s8(values.val[aux[2] & 1], vshrq_n_u8(q4bits.val[2], 4));
        bits.b2.val[2] = vqtbl1q_s8(values.val[aux[3] & 1], vandq_u8(q4bits.val[3], bits.m4b));
        bits.b2.val[3] = vqtbl1q_s8(values.val[aux[3] & 1], vshrq_n_u8(q4bits.val[3], 4));
    }
    inline void prepare([[maybe_unused]] int i, int j) {
        if (j == 0) return;
        make_quants(q4bits_2, bits, aux+4);
    }
    static int16x8_t load_shift() {
        static const int16_t k_shift[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        return vld1q_s16(k_shift);
    }

    Q4bits bits;
    const int8x16x2_t values;
    const uint16x8_t  mask = vdupq_n_s16(254);
    const uint16x8_t  bmask = vdupq_n_u16(0xfffe);
    const uint16x8_t  m1   = vdupq_n_u16(1);
    const int16x8_t   shift = load_shift();
    const int16x8_t   m127 = vdupq_n_s16(-127);
    uint16x8x4_t q4bits_2;
    int16_t aux[8];
};

struct DequantizerIQ2KS final : public BaseDequantizer<block_iq2_ks, true, true> {
    DequantizerIQ2KS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, [[maybe_unused]] const Q8& q8, [[maybe_unused]] float32x4_t * acc) {
        const uint16_t * sc16 = (const uint16_t *)x[i].scales;
        uint32_t aux32 = sc16[0] | (sc16[1] << 16);
        uint8x8_t scales8 = vreinterpret_u8_u32(vdup_n_u32(aux32));
        scales8 = vand_u8(vzip1_u8(scales8, vshr_n_u8(scales8, 4)), vdup_n_u8(0xf));
        uint8x8_t sh = vand_u8(vceq_u8(vand_u8(vdup_n_u8(x[i].extra >> 8), hmask), vdup_n_u8(0)), vdup_n_u8(16));
        int16x8_t scales16 = vmovl_s8(vsub_s8(vreinterpret_s8_u8(scales8), vreinterpret_s8_u8(sh)));
        int32x4x2_t scales = {vmovl_s16(vget_low_s16(scales16)), vmovl_s16(vget_high_s16(scales16))};
        return scales;
    }
    inline void prepare(int i, int j) {
        uint8_t extra = x[i].extra >> 4*j;
        bits.prepare(x[i].qs+32*j);
        bits.b1.val[0] = vqtbl1q_s8(values.val[extra & 1], bits.b1.val[0]);
        bits.b1.val[1] = vqtbl1q_s8(values.val[extra & 1], bits.b1.val[1]); extra >>= 1;
        bits.b1.val[2] = vqtbl1q_s8(values.val[extra & 1], bits.b1.val[2]);
        bits.b1.val[3] = vqtbl1q_s8(values.val[extra & 1], bits.b1.val[3]); extra >>= 1;
        bits.b2.val[0] = vqtbl1q_s8(values.val[extra & 1], bits.b2.val[0]);
        bits.b2.val[1] = vqtbl1q_s8(values.val[extra & 1], bits.b2.val[1]); extra >>= 1;
        bits.b2.val[2] = vqtbl1q_s8(values.val[extra & 1], bits.b2.val[2]);
        bits.b2.val[3] = vqtbl1q_s8(values.val[extra & 1], bits.b2.val[3]);
    }

    Q2bits bits;
    const uint8x8_t hmask = vreinterpret_u8_u64(vdup_n_u64(0x8040201008040201));
    const int8x16x2_t values = { vreinterpretq_s8_u64(vdupq_n_u64(0x1101f3e1)), vreinterpretq_s8_u64(vdupq_n_u64(0x1606f8e6)) };

};

struct DequantizerIQ2KL final : public BaseDequantizer<block_iq2_kl, true, true> {
    DequantizerIQ2KL(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc), shuff(load_shuffle()), shifts(load_shift()) { load_values(values); }

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, [[maybe_unused]] const Q8& q8, [[maybe_unused]] float32x4_t * acc) {
        uint32_t aux32; std::memcpy(&aux32, x[i].scales_l, 4);
        auto scl = vand_u8(vdup_n_u8(0xf), vreinterpret_u8_u32(uint32x2_t{aux32, aux32 >> 4}));
        auto sch = vandq_u16(vshlq_u16(vdupq_n_u16(x[i].scales_h), shifts), vdupq_n_u16(0x30));
        auto scales16 = vsubq_s16(vreinterpretq_s16_u16(vorrq_u16(sch, vmovl_u8(scl))), vdupq_n_s16(32));
        int32x4x2_t scales = {vmovl_s16(vget_low_s16(scales16)), vmovl_s16(vget_high_s16(scales16))};
        return scales;
    }
    inline void process_pair(uint8x16_t x, uint8x16_t * val) const {
        uint8x16x2_t aux{ vqtbl2q_s8(values[0], x), vqtbl2q_s8(values[1], x) };
        val[0] = vqtbl2q_u8(aux, shuff.val[0]);
        val[1] = vqtbl2q_u8(aux, shuff.val[1]);
    }
    inline void prepare(int i, int j) {
        hbits = j == 0 ? vld1q_u8(x[i].qh) : vshrq_n_u8(hbits, 4);
        auto lbits = vld1q_u8_x2(x[i].qs+32*j);

        uint8x16x4_t aux;
        aux.val[0] = vorrq_u8(vandq_u8(m10, vshlq_n_u8(hbits, 4)), vandq_u8(lbits.val[0], vdupq_n_u8(0xf)));
        aux.val[1] = vorrq_u8(vandq_u8(m10, vshlq_n_u8(hbits, 3)), vshrq_n_u8(lbits.val[0], 4));
        aux.val[2] = vorrq_u8(vandq_u8(m10, vshlq_n_u8(hbits, 2)), vandq_u8(lbits.val[1], vdupq_n_u8(0xf)));
        aux.val[3] = vorrq_u8(vandq_u8(m10, vshlq_n_u8(hbits, 1)), vshrq_n_u8(lbits.val[1], 4));

        process_pair(aux.val[0], bits.b1.val+0);
        process_pair(aux.val[1], bits.b1.val+2);
        process_pair(aux.val[2], bits.b2.val+0);
        process_pair(aux.val[3], bits.b2.val+2);

        // The compiler crashes the moment I try to use vzip2q_u8!!!
        //bits.b1.val[0] = vandq_u8(lbits.val[0], vdupq_n_u8(0xf));
        //bits.b1.val[2] = vshrq_n_u8(lbits.val[0], 4);
        //bits.b2.val[0] = vandq_u8(lbits.val[1], vdupq_n_u8(0xf));
        //bits.b2.val[2] = vshrq_n_u8(lbits.val[1], 4);
        //bits.b1.val[0] = vorrq_u8(bits.b1.val[0], vandq_u8(m10, vshlq_n_u8(hbits, 4)));
        //bits.b1.val[2] = vorrq_u8(bits.b1.val[2], vandq_u8(m10, vshlq_n_u8(hbits, 3)));
        //bits.b2.val[0] = vorrq_u8(bits.b2.val[0], vandq_u8(m10, vshlq_n_u8(hbits, 2)));
        //bits.b2.val[2] = vorrq_u8(bits.b2.val[2], vandq_u8(m10, vshlq_n_u8(hbits, 1)));

        //auto t1 = vqtbl2q_s8(values[0], bits.b1.val[0]);
        //auto t2 = vqtbl2q_s8(values[1], bits.b1.val[0]);
        //bits.b1.val[0] = vzip1q_s8(t1, t2);
        ////bits.b1.val[1] = vzip2q_u8(t1, t2);
        //t1 = vqtbl2q_s8(values[0], bits.b1.val[2]);
        //t2 = vqtbl2q_s8(values[1], bits.b1.val[2]);
        //bits.b1.val[2] = vzip1q_s8(t1, t2);
        ////bits.b1.val[3] = vzip2q_s8(t1, t2);

        //t1 = vqtbl2q_s8(values[0], bits.b2.val[0]);
        //t2 = vqtbl2q_s8(values[1], bits.b2.val[0]);
        //bits.b2.val[0] = vzip1q_s8(t1, t2);
        ////bits.b2.val[1] = vzip2q_s8(t1, t2);
        //t1 = vqtbl2q_s8(values[0], bits.b2.val[2]);
        //t2 = vqtbl2q_s8(values[1], bits.b2.val[2]);
        //bits.b2.val[2] = vzip1q_s8(t1, t2);
        ////bits.b2.val[3] = vzip2q_s8(t1, t2);
    }
    static inline int16x8_t load_shift() {
        static const int16_t k_shift[8] = {4, 2, 0, -2, -4, -6, -8, -10};
        return vld1q_s16(k_shift);
    }
    static inline void load_values(int8x16x2_t * values) {
        static const int8_t k_values[64] = {
            -63, -63, -40, -40, -40, -40, -23, -23, -23, -23, -23, -10, -10, -10, -10, 1, 1, 1, 1, 1, 13, 13, 13, 13, 13, 28, 28, 28, 28, 28, 47, 47,
            -23, 13, -63, -10, 13, 47, -40, -23, 1, 13, 28, -63, 1, 13, 47, -23, -10, 1, 13, 28, -40, -23, -10, 1, 13, -63, -23, 1, 28, 47, -23, 13,
        };
        values[0] = vld1q_s8_x2(k_values+ 0);
        values[1] = vld1q_s8_x2(k_values+32);
    }
    static uint8x16x2_t load_shuffle() {
        static const uint8_t k_shuff[32] = {
            0, 16, 1, 17,  2, 18,  3, 19,  4, 20,  5, 21,  6, 22,  7, 23,
            8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
        };
        return vld1q_u8_x2(k_shuff);
    }

    struct { uint8x16x4_t b1, b2; } bits;
    uint8x16_t hbits;
    const uint8x16x2_t shuff;
    const int16x8_t shifts;
    const uint8x16_t m10 = vdupq_n_u8(0x10);
    int8x16x2_t values[2];

};

template <int nrc_y>
void mul_mat_iq4_ks_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = vdupq_n_u8(0xf);
    auto values = vld1q_s8(iq4k_values);
    int nbl = n / QK_K;
    int8x16_t qx[8];
    int16x8x4_t iscales;
    int32x4x4_t scales;
    float32x4_t acc[nrc_y] = {};
    int32x4_t isum[nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto dptr = (const float *)((const char *)vx + ix*bx);
        auto d4 = vld1q_f32(dptr);
        const block_iq4_ks_r4 * iq4 = (const block_iq4_ks_r4 *)(dptr + 4);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            auto sas = vld1q_u8_x2(iq4[ibl].scales);
            auto scale = vandq_u8(sas.val[0], vdupq_n_u8(254));
            iscales.val[0] = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8 (scale))), vdupq_n_s16(-127));
            iscales.val[1] = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(scale))), vdupq_n_s16(-127));
            scale = vandq_u8(sas.val[1], vdupq_n_u8(254));
            iscales.val[2] = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8 (scale))), vdupq_n_s16(-127));
            iscales.val[3] = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(scale))), vdupq_n_s16(-127));
            // Adding the block shifts costs us ~9% in performance drop.
            // Is there a better way?
            sas.val[0] = vshlq_n_u8(vandq_u8(sas.val[0], vdupq_n_u8(1)), 2);
            sas.val[1] = vshlq_n_u8(vandq_u8(sas.val[1], vdupq_n_u8(1)), 2);
            {
                auto s16_1 = vmulq_s16(iscales.val[0], vmovl_u8(vget_low_u8 (sas.val[0])));
                auto s16_2 = vmulq_s16(iscales.val[1], vmovl_u8(vget_high_u8(sas.val[0])));
                auto s16_3 = vmulq_s16(iscales.val[2], vmovl_u8(vget_low_u8 (sas.val[1])));
                auto s16_4 = vmulq_s16(iscales.val[3], vmovl_u8(vget_high_u8(sas.val[1])));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto bsums = vld1q_s16_x2(q8.y[iy][ibl].bsums);
                    auto bs = vpaddq_s16(bsums.val[0], bsums.val[1]);
                    auto b8 = vget_low_s16(bs);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_1), b8, 0);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_1), b8, 1);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_2), b8, 2);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_2), b8, 3);
                    b8 = vget_high_s16(bs);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_3), b8, 0);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_3), b8, 1);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_4), b8, 2);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_4), b8, 3);
                }
            }
            for (int is = 0; is < 2; ++is) {
                scales.val[0] = vmovl_s16(vget_low_s16 (iscales.val[2*is+0]));
                scales.val[1] = vmovl_s16(vget_high_s16(iscales.val[2*is+0]));
                scales.val[2] = vmovl_s16(vget_low_s16 (iscales.val[2*is+1]));
                scales.val[3] = vmovl_s16(vget_high_s16(iscales.val[2*is+1]));
                for (int ib = 0; ib < 4; ++ib) {
                    auto bits = vld1q_u8_x4(iq4[ibl].qs + 256*is + 64*ib);
                    prepare_iq4_nl_quants(values, m4, bits, qx);
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8_x2(q8.y[iy][ibl].qs+128*is+32*ib);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales.val[ib], sumi);
                    }
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vfmaq_f32(acc[iy], vdupq_n_f32(q8.scale(iy, ibl)), vcvtq_f32_s32(isum[iy]));
                isum[iy] = vdupq_n_s32(0);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, vmulq_f32(d4, acc[iy]));
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

template <int nrc_y>
void mul_mat_iq5_ks_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = vdupq_n_u8(0xf);
    auto m10 = vdupq_n_u8(0x10);
    auto values = vld1q_s8_x2(iq5nl_values);
    int nbl = n / QK_K;
    int8x16_t qx[8];
    int16x8x4_t iscales;
    int32x4x4_t scales;
    float32x4_t acc[nrc_y] = {};
    int32x4_t isum[nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto dptr = (const float *)((const char *)vx + ix*bx);
        auto d4 = vld1q_f32(dptr);
        const block_iq5_ks_r4 * iq5 = (const block_iq5_ks_r4 *)(dptr + 4);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            auto sas = vld1q_u8_x2(iq5[ibl].scales);
            auto scale = vandq_u8(sas.val[0], vdupq_n_u8(254));
            iscales.val[0] = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8 (scale))), vdupq_n_s16(-127));
            iscales.val[1] = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(scale))), vdupq_n_s16(-127));
            scale = vandq_u8(sas.val[1], vdupq_n_u8(254));
            iscales.val[2] = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8 (scale))), vdupq_n_s16(-127));
            iscales.val[3] = vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(scale))), vdupq_n_s16(-127));
            // Adding the block shifts costs us ~9% in performance drop.
            // Is there a better way?
            sas.val[0] = vshlq_n_u8(vandq_u8(sas.val[0], vdupq_n_u8(1)), 1);
            sas.val[1] = vshlq_n_u8(vandq_u8(sas.val[1], vdupq_n_u8(1)), 1);
            {
                auto s16_1 = vmulq_s16(iscales.val[0], vmovl_u8(vget_low_u8 (sas.val[0])));
                auto s16_2 = vmulq_s16(iscales.val[1], vmovl_u8(vget_high_u8(sas.val[0])));
                auto s16_3 = vmulq_s16(iscales.val[2], vmovl_u8(vget_low_u8 (sas.val[1])));
                auto s16_4 = vmulq_s16(iscales.val[3], vmovl_u8(vget_high_u8(sas.val[1])));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto bsums = vld1q_s16_x2(q8.y[iy][ibl].bsums);
                    auto bs = vpaddq_s16(bsums.val[0], bsums.val[1]);
                    auto b8 = vget_low_s16(bs);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_1), b8, 0);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_1), b8, 1);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_2), b8, 2);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_2), b8, 3);
                    b8 = vget_high_s16(bs);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_3), b8, 0);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_3), b8, 1);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_4), b8, 2);
                    isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_4), b8, 3);
                }
            }
            for (int is = 0; is < 2; ++is) {
                scales.val[0] = vmovl_s16(vget_low_s16 (iscales.val[2*is+0]));
                scales.val[1] = vmovl_s16(vget_high_s16(iscales.val[2*is+0]));
                scales.val[2] = vmovl_s16(vget_low_s16 (iscales.val[2*is+1]));
                scales.val[3] = vmovl_s16(vget_high_s16(iscales.val[2*is+1]));
                for (int ib = 0; ib < 4; ++ib) {
                    auto lbits = vld1q_u8_x4(iq5[ibl].qs + 256*is + 64*ib);
                    auto hbits = vld1q_u8(iq5[ibl].qh + 64*is + 16*ib);
                    qx[0] = vorrq_u8(vandq_u8(lbits.val[0],  m4), vandq_u8(m10, vshlq_n_u8(hbits, 4)));
                    qx[1] = vorrq_u8(vandq_u8(lbits.val[1],  m4), vandq_u8(m10, vshlq_n_u8(hbits, 2)));
                    qx[2] = vorrq_u8(vandq_u8(lbits.val[2],  m4), vandq_u8(m10, hbits));
                    qx[3] = vorrq_u8(vandq_u8(lbits.val[3],  m4), vandq_u8(m10, vshrq_n_u8(hbits, 2)));
                    qx[4] = vorrq_u8(vshrq_n_u8(lbits.val[0], 4), vandq_u8(m10, vshlq_n_u8(hbits, 3)));
                    qx[5] = vorrq_u8(vshrq_n_u8(lbits.val[1], 4), vandq_u8(m10, vshlq_n_u8(hbits, 1)));
                    qx[6] = vorrq_u8(vshrq_n_u8(lbits.val[2], 4), vandq_u8(m10, vshrq_n_u8(hbits, 1)));
                    qx[7] = vorrq_u8(vshrq_n_u8(lbits.val[3], 4), vandq_u8(m10, vshrq_n_u8(hbits, 3)));
                    for (int l = 0; l < 8; ++l) qx[l] = vqtbl2q_s8(values, qx[l]);
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8_x2(q8.y[iy][ibl].qs+128*is+32*ib);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales.val[ib], sumi);
                    }
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vfmaq_f32(acc[iy], vdupq_n_f32(q8.scale(iy, ibl)), vcvtq_f32_s32(isum[iy]));
                isum[iy] = vdupq_n_s32(0);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, vmulq_f32(d4, acc[iy]));
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

template <int nrc_y, int k_shift>
inline void iq3_4_add_shift(int ibl, const Q8<nrc_y, block_q8_K>& q8, const int8x16x4_t& i8scales, uint8x16_t extra,
        int32x4_t * isum) {
    auto ms = vdupq_n_s8(k_shift);
    int8x16_t s8_1, s8_2;
    if constexpr (k_shift == 5) {
        auto m1 = vdupq_n_u8(1);
        s8_1 = vmulq_s8(i8scales.val[0], vandq_s8(ms, vceqq_u8(vandq_u8(extra, m1), m1))); extra = vshrq_n_u8(extra, 2);
        s8_2 = vmulq_s8(i8scales.val[1], vandq_s8(ms, vceqq_u8(vandq_u8(extra, m1), m1))); extra = vshrq_n_u8(extra, 2);
    } else {
        if constexpr (k_shift == 4) {
            s8_1 = vmulq_s8(i8scales.val[0], vandq_u8(ms, vshlq_n_u8(extra, 2)));
            s8_2 = vmulq_s8(i8scales.val[1], vandq_u8(ms, extra));
        } else {
            s8_1 = vmulq_s8(i8scales.val[0], vandq_u8(ms, vshlq_n_u8(extra, 1)));
            s8_2 = vmulq_s8(i8scales.val[1], vandq_u8(ms, vshrq_n_u8(extra, 1)));
        }
    }
    auto s16_1 = vmovl_s8(vget_low_s8 (s8_1));
    auto s16_2 = vmovl_s8(vget_high_s8(s8_1));
    auto s16_3 = vmovl_s8(vget_low_s8 (s8_2));
    auto s16_4 = vmovl_s8(vget_high_s8(s8_2));
    for (int iy = 0; iy < nrc_y; ++iy) {
        auto b8 = vld1_s16(q8.y[iy][ibl].bsums);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_1), b8, 0);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_1), b8, 1);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_2), b8, 2);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_2), b8, 3);
        b8 = vld1_s16(q8.y[iy][ibl].bsums+4);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_3), b8, 0);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_3), b8, 1);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_4), b8, 2);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_4), b8, 3);
    }
    if constexpr (k_shift == 5) {
        auto m1 = vdupq_n_u8(1);
        s8_1 = vmulq_s8(i8scales.val[2], vandq_s8(ms, vceqq_u8(vandq_u8(extra, m1), m1))); extra = vshrq_n_u8(extra, 2);
        s8_2 = vmulq_s8(i8scales.val[3], vandq_s8(ms, vceqq_u8(vandq_u8(extra, m1), m1))); extra = vshrq_n_u8(extra, 2);
    } else {
        if constexpr (k_shift == 4) {
            s8_1 = vmulq_s8(i8scales.val[2], vandq_u8(ms, vshrq_n_u8(extra, 2)));
            s8_2 = vmulq_s8(i8scales.val[3], vandq_u8(ms, vshrq_n_u8(extra, 4)));
        } else {
            s8_1 = vmulq_s8(i8scales.val[2], vandq_u8(ms, vshrq_n_u8(extra, 3)));
            s8_2 = vmulq_s8(i8scales.val[3], vandq_u8(ms, vshrq_n_u8(extra, 5)));
        }
    }
    s16_1 = vmovl_s8(vget_low_s8 (s8_1));
    s16_2 = vmovl_s8(vget_high_s8(s8_1));
    s16_3 = vmovl_s8(vget_low_s8 (s8_2));
    s16_4 = vmovl_s8(vget_high_s8(s8_2));
    for (int iy = 0; iy < nrc_y; ++iy) {
        auto b8 = vld1_s16(q8.y[iy][ibl].bsums+8);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_1), b8, 0);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_1), b8, 1);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_2), b8, 2);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_2), b8, 3);
        b8 = vld1_s16(q8.y[iy][ibl].bsums+12);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_3), b8, 0);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_3), b8, 1);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_low_s16 (s16_4), b8, 2);
        isum[iy] = vmlal_lane_s16(isum[iy], vget_high_s16(s16_4), b8, 3);
    }
}

template <int nrc_y>
void mul_mat_iq2_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = vdupq_n_u8(0xf);
    auto m03 = vdupq_n_u8(0x03);
    auto ms = vdupq_n_u8(4);
    uint8x16x2_t shift_shuffle = {
        vreinterpretq_u8_u64(uint64x2_t{0x0101010100000000, 0x0303030302020202}),
        vreinterpretq_u8_u64(uint64x2_t{0x0505050504040404, 0x0707070706060606})
    };
    auto values8 = vld1_s8(iq2nl_values);
    auto values = vcombine_s8(values8, values8);
    int nbl = n / QK_K;
    int8x16_t qx[4];
    int8x16x4_t i8scales;
    int16x8x4_t i16scales;
    float32x4_t acc[nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_iq2_k_r4 * iq2 = (const block_iq2_k_r4 *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq2[ibl].d));
            auto extra8 = vld1_u8(iq2[ibl].extra);
            uint8x16_t extra;
            if constexpr (nrc_y == 1) {
                extra = vcombine_u8(extra8, vshr_n_u8(extra8,1));
            } else {
                extra = vcombine_u8(extra8, extra8);
            }
            auto sl = vld1q_u8_x2(iq2[ibl].scales);
            i8scales.val[0] = vaddq_s8(vandq_u8(sl.val[0],  m4), vdupq_n_s8(-8));
            i8scales.val[1] = vaddq_s8(vandq_u8(sl.val[1],  m4), vdupq_n_s8(-8));
            i8scales.val[2] = vaddq_s8(vshrq_n_u8(sl.val[0], 4), vdupq_n_s8(-8));
            i8scales.val[3] = vaddq_s8(vshrq_n_u8(sl.val[1], 4), vdupq_n_s8(-8));
            int32x4_t isum[nrc_y] = {};
            if constexpr (nrc_y == 1) {
                iq3_4_add_shift<nrc_y, 5>(ibl, q8, i8scales, extra, isum);
            }
            for (int is = 0; is < 2; ++is) {
                i16scales.val[0] = vmovl_s8(vget_low_s8 (i8scales.val[2*is+0]));
                i16scales.val[1] = vmovl_s8(vget_high_s8(i8scales.val[2*is+0]));
                i16scales.val[2] = vmovl_s8(vget_low_s8 (i8scales.val[2*is+1]));
                i16scales.val[3] = vmovl_s8(vget_high_s8(i8scales.val[2*is+1]));
                for (int ib = 0; ib < 4; ++ib) {
                    auto scales = vmovl_s16(vget_low_s16 (i16scales.val[ib]));
                    auto bits = vld1q_u8_x2(iq2[ibl].qs + 128*is + 32*ib);
                    qx[0] = vandq_u8(           bits.val[0],     m03);
                    qx[1] = vandq_u8(vshrq_n_u8(bits.val[0], 2), m03);
                    qx[2] = vandq_u8(vshrq_n_u8(bits.val[0], 4), m03);
                    qx[3] = vandq_u8(vshrq_n_u8(bits.val[0], 6), m03);
                    uint8x16_t shifts;
                    if constexpr (nrc_y == 1) {
                        qx[0] = vqtbl1q_s8(values, qx[0]);  //  0...3 from the 4 rows
                        qx[1] = vqtbl1q_s8(values, qx[1]);  //  4...7
                        qx[2] = vqtbl1q_s8(values, qx[2]);  //  8..11
                        qx[3] = vqtbl1q_s8(values, qx[3]);  // 12..15
                    } else {
                        shifts = vandq_u8(ms, vshlq_n_u8(extra, 2));
                        auto shift = vqtbl1q_u8(shifts, shift_shuffle.val[0]);
                        extra = vshrq_n_u8(extra, 1);
                        qx[0] = vqtbl1q_s8(values, vaddq_u8(shift, qx[0]));  //  0...3 from the 4 rows
                        qx[1] = vqtbl1q_s8(values, vaddq_u8(shift, qx[1]));  //  4...7
                        qx[2] = vqtbl1q_s8(values, vaddq_u8(shift, qx[2]));  //  8..11
                        qx[3] = vqtbl1q_s8(values, vaddq_u8(shift, qx[3]));  // 12..15
                    }
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8(q8.y[iy][ibl].qs+128*is+32*ib);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                    }
                    qx[0] = vandq_u8(           bits.val[1],     m03);
                    qx[1] = vandq_u8(vshrq_n_u8(bits.val[1], 2), m03);
                    qx[2] = vandq_u8(vshrq_n_u8(bits.val[1], 4), m03);
                    qx[3] = vandq_u8(vshrq_n_u8(bits.val[1], 6), m03);
                    if constexpr (nrc_y == 1) {
                        qx[0] = vqtbl1q_s8(values, qx[0]);  //  0...3 from the 4 rows
                        qx[1] = vqtbl1q_s8(values, qx[1]);  //  4...7
                        qx[2] = vqtbl1q_s8(values, qx[2]);  //  8..11
                        qx[3] = vqtbl1q_s8(values, qx[3]);  // 12..15
                    } else {
                        auto shift = vqtbl1q_u8(shifts, shift_shuffle.val[1]);
                        qx[0] = vqtbl1q_s8(values, vaddq_u8(shift, qx[0]));  //  0...3 from the 4 rows
                        qx[1] = vqtbl1q_s8(values, vaddq_u8(shift, qx[1]));  //  4...7
                        qx[2] = vqtbl1q_s8(values, vaddq_u8(shift, qx[2]));  //  8..11
                        qx[3] = vqtbl1q_s8(values, vaddq_u8(shift, qx[3]));  // 12..15
                    }
                    scales = vmovl_s16(vget_high_s16(i16scales.val[ib]));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8(q8.y[iy][ibl].qs+128*is+32*ib+16);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                    }
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vfmaq_f32(acc[iy], vmulq_f32(d4, vdupq_n_f32(q8.scale(iy, ibl))), vcvtq_f32_s32(isum[iy]));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

template <int nrc_y>
void mul_mat_iq3_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = vdupq_n_u8(0xf);
    auto ms = nrc_y == 1 ? vdupq_n_u8(4) : vdupq_n_u8(8);
    auto m03 = vdupq_n_u8(0x03);
    auto m04 = vdupq_n_u8(0x04);
    uint8x16x2_t shift_shuffle = {
        vreinterpretq_u8_u64(uint64x2_t{0x0101010100000000, 0x0303030302020202}),
        vreinterpretq_u8_u64(uint64x2_t{0x0505050504040404, 0x0707070706060606})
    };
    uint8x16x2_t smask = { vcombine_u8(vdup_n_u8(1), vdup_n_u8(2)), vcombine_u8(vdup_n_u8(4), vdup_n_u8(8)) };
    auto values = vld1q_s8(iq3nl_values);
    int nbl = n / QK_K;
    int8x16_t qx[4];
    int8x16x4_t i8scales;
    int16x8x4_t i16scales;
    float32x4_t acc[nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_iq3_k_r4 * iq3 = (const block_iq3_k_r4 *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq3[ibl].d));
            auto extra8 = vld1_u8(iq3[ibl].extra);
            uint8x16_t extra;
            if constexpr (nrc_y == 1) {
                extra = vcombine_u8(extra8, vshr_n_u8(extra8,1));
            } else {
                extra = vcombine_u8(extra8, extra8);
            }
            auto sl = vld1q_u8_x2(iq3[ibl].scales_l);
            auto sh8 = vld1_u8(iq3[ibl].scales_h);
            auto sh = vcombine_u8(sh8, sh8);
            i8scales.val[0] = vaddq_s8(vshlq_n_u8(vandq_u8(sl.val[0],  m4), 1), vdupq_n_s8(1));
            i8scales.val[1] = vaddq_s8(vshlq_n_u8(vandq_u8(sl.val[1],  m4), 1), vdupq_n_s8(1));
            i8scales.val[2] = vaddq_s8(vshlq_n_u8(vshrq_n_u8(sl.val[0], 4), 1), vdupq_n_s8(1));
            i8scales.val[3] = vaddq_s8(vshlq_n_u8(vshrq_n_u8(sl.val[1], 4), 1), vdupq_n_s8(1));
            i8scales.val[0] = vmulq_s8(i8scales.val[0], vorrq_u8(vceqq_u8(vandq_u8(sh, smask.val[0]), smask.val[0]), vdupq_n_u8(1)));
            i8scales.val[1] = vmulq_s8(i8scales.val[1], vorrq_u8(vceqq_u8(vandq_u8(sh, smask.val[1]), smask.val[1]), vdupq_n_u8(1)));
            sh = vshrq_n_u8(sh, 4);
            i8scales.val[2] = vmulq_s8(i8scales.val[2], vorrq_u8(vceqq_u8(vandq_u8(sh, smask.val[0]), smask.val[0]), vdupq_n_u8(1)));
            i8scales.val[3] = vmulq_s8(i8scales.val[3], vorrq_u8(vceqq_u8(vandq_u8(sh, smask.val[1]), smask.val[1]), vdupq_n_u8(1)));
            int32x4_t isum[nrc_y] = {};
            if constexpr (nrc_y == 1) {
                iq3_4_add_shift<nrc_y, 4>(ibl, q8, i8scales, extra, isum);
            }
            for (int is = 0; is < 2; ++is) {
                i16scales.val[0] = vmovl_s8(vget_low_s8 (i8scales.val[2*is+0]));
                i16scales.val[1] = vmovl_s8(vget_high_s8(i8scales.val[2*is+0]));
                i16scales.val[2] = vmovl_s8(vget_low_s8 (i8scales.val[2*is+1]));
                i16scales.val[3] = vmovl_s8(vget_high_s8(i8scales.val[2*is+1]));
                for (int ib = 0; ib < 4; ++ib) {
                    auto scales = vmovl_s16(vget_low_s16 (i16scales.val[ib]));
                    auto lbits = vld1q_u8_x2(iq3[ibl].qs + 128*is + 32*ib);
                    auto hbits = vld1q_u8(iq3[ibl].qh + 64*is + 16*ib);
                    qx[0] = vorrq_u8(vandq_u8(           lbits.val[0],     m03), vandq_u8(m04, vshlq_n_u8(hbits, 2)));
                    qx[1] = vorrq_u8(vandq_u8(vshrq_n_u8(lbits.val[0], 2), m03), vandq_u8(m04, vshlq_n_u8(hbits, 1)));
                    qx[2] = vorrq_u8(vandq_u8(vshrq_n_u8(lbits.val[0], 4), m03), vandq_u8(m04, hbits));
                    qx[3] = vorrq_u8(vandq_u8(vshrq_n_u8(lbits.val[0], 6), m03), vandq_u8(m04, vshrq_n_u8(hbits, 1)));
                    uint8x16_t shifts;
                    if constexpr (nrc_y == 1) {
                        qx[0] = vqtbl1q_s8(values, qx[0]);  //  0...3 from the 4 rows
                        qx[1] = vqtbl1q_s8(values, qx[1]);  //  4...7
                        qx[2] = vqtbl1q_s8(values, qx[2]);  //  8..11
                        qx[3] = vqtbl1q_s8(values, qx[3]);  // 12..15
                    } else {
                        shifts = vandq_u8(ms, vshlq_n_u8(extra, 3));
                        auto shift = vqtbl1q_u8(shifts, shift_shuffle.val[0]);
                        extra = vshrq_n_u8(extra, 1);
                        qx[0] = vqtbl1q_s8(values, vaddq_u8(shift, qx[0]));  //  0...3 from the 4 rows
                        qx[1] = vqtbl1q_s8(values, vaddq_u8(shift, qx[1]));  //  4...7
                        qx[2] = vqtbl1q_s8(values, vaddq_u8(shift, qx[2]));  //  8..11
                        qx[3] = vqtbl1q_s8(values, vaddq_u8(shift, qx[3]));  // 12..15
                    }
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8(q8.y[iy][ibl].qs+128*is+32*ib);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                    }
                    qx[0] = vorrq_u8(vandq_u8(           lbits.val[1],     m03), vandq_u8(m04, vshrq_n_u8(hbits, 2)));
                    qx[1] = vorrq_u8(vandq_u8(vshrq_n_u8(lbits.val[1], 2), m03), vandq_u8(m04, vshrq_n_u8(hbits, 3)));
                    qx[2] = vorrq_u8(vandq_u8(vshrq_n_u8(lbits.val[1], 4), m03), vandq_u8(m04, vshrq_n_u8(hbits, 4)));
                    qx[3] = vorrq_u8(vandq_u8(vshrq_n_u8(lbits.val[1], 6), m03), vandq_u8(m04, vshrq_n_u8(hbits, 5)));
                    if constexpr (nrc_y == 1) {
                        qx[0] = vqtbl1q_s8(values, qx[0]);  //  0...3 from the 4 rows
                        qx[1] = vqtbl1q_s8(values, qx[1]);  //  4...7
                        qx[2] = vqtbl1q_s8(values, qx[2]);  //  8..11
                        qx[3] = vqtbl1q_s8(values, qx[3]);  // 12..15
                    } else {
                        auto shift = vqtbl1q_u8(shifts, shift_shuffle.val[1]);
                        qx[0] = vqtbl1q_s8(values, vaddq_u8(shift, qx[0]));  //  0...3 from the 4 rows
                        qx[1] = vqtbl1q_s8(values, vaddq_u8(shift, qx[1]));  //  4...7
                        qx[2] = vqtbl1q_s8(values, vaddq_u8(shift, qx[2]));  //  8..11
                        qx[3] = vqtbl1q_s8(values, vaddq_u8(shift, qx[3]));  // 12..15
                    }
                    scales = vmovl_s16(vget_high_s16(i16scales.val[ib]));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8(q8.y[iy][ibl].qs+128*is+32*ib+16);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                    }
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vfmaq_f32(acc[iy], vmulq_f32(d4, vdupq_n_f32(q8.scale(iy, ibl))), vcvtq_f32_s32(isum[iy]));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

template <int nrc_y>
void mul_mat_iq4_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = vdupq_n_u8(0xf);
    auto m3 = vdupq_n_u8(0x30);
    auto ms = vdupq_n_u8(4);
    auto m32 = vdupq_n_s8(-32);
    uint8x16x2_t shift_shuffle = {
        vreinterpretq_u8_u64(uint64x2_t{0x0101010100000000, 0x0303030302020202}),
        vreinterpretq_u8_u64(uint64x2_t{0x0505050504040404, 0x0707070706060606})
    };
    auto values = vld1q_s8(iq4k_values);
    int nbl = n / QK_K;
    int8x16_t qx[4];
    int8x16x4_t i8scales;
    int16x8x4_t i16scales;
    float32x4_t acc[nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_iq4_k_r4 * iq4 = (const block_iq4_k_r4 *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq4[ibl].d));
            auto extra8 = vld1_u8(iq4[ibl].extra);
            uint8x16_t extra;
            if constexpr (nrc_y == 1) {
                extra = vcombine_u8(extra8, vshr_n_u8(extra8,1));
            } else {
                extra = vcombine_u8(extra8, extra8);
            }
            auto sl = vld1q_u8_x2(iq4[ibl].scales_l);
            auto sh = vld1q_u8(iq4[ibl].scales_h);
            i8scales.val[0] = vaddq_s8(vorrq_u8(vandq_u8(sl.val[0],  m4), vandq_u8(vshlq_n_u8(sh, 4), m3)), m32);
            i8scales.val[1] = vaddq_s8(vorrq_u8(vandq_u8(sl.val[1],  m4), vandq_u8(vshlq_n_u8(sh, 2), m3)), m32);
            i8scales.val[2] = vaddq_s8(vorrq_u8(vshrq_n_u8(sl.val[0], 4), vandq_u8(sh, m3)), m32);
            i8scales.val[3] = vaddq_s8(vorrq_u8(vshrq_n_u8(sl.val[1], 4), vandq_u8(vshrq_n_u8(sh, 2), m3)), m32);
            int32x4_t isum[nrc_y] = {};
            if constexpr (nrc_y == 1) {
                iq3_4_add_shift<nrc_y, 4>(ibl, q8, i8scales, extra, isum);
            }
            for (int is = 0; is < 2; ++is) {
                i16scales.val[0] = vmovl_s8(vget_low_s8 (i8scales.val[2*is+0]));
                i16scales.val[1] = vmovl_s8(vget_high_s8(i8scales.val[2*is+0]));
                i16scales.val[2] = vmovl_s8(vget_low_s8 (i8scales.val[2*is+1]));
                i16scales.val[3] = vmovl_s8(vget_high_s8(i8scales.val[2*is+1]));
                for (int ib = 0; ib < 4; ++ib) {
                    auto bits = vld1q_u8_x4(iq4[ibl].qs + 256*is + 64*ib);
                    uint8x16_t shifts;
                    if constexpr (nrc_y == 1) {
                        qx[0] = vqtbl1q_s8(values, vandq_u8(bits.val[0], m4));   //  0...3 from the 4 rows
                        qx[1] = vqtbl1q_s8(values, vandq_u8(bits.val[2], m4));   //  4...7
                        qx[2] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[0], 4));  //  8..11
                        qx[3] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[2], 4));  // 12..15
                    } else {
                        shifts = vandq_u8(ms, vshlq_n_u8(extra, 2));
                        auto shift = vqtbl1q_u8(shifts, shift_shuffle.val[0]);
                        extra = vshrq_n_u8(extra, 1);
                        qx[0] = vaddq_s8(shift, vqtbl1q_s8(values, vandq_u8(bits.val[0], m4)));   //  0...3 from the 4 rows
                        qx[1] = vaddq_s8(shift, vqtbl1q_s8(values, vandq_u8(bits.val[2], m4)));   //  4...7
                        qx[2] = vaddq_s8(shift, vqtbl1q_s8(values, vshrq_n_u8(bits.val[0], 4)));  //  8..11
                        qx[3] = vaddq_s8(shift, vqtbl1q_s8(values, vshrq_n_u8(bits.val[2], 4)));  // 12..15
                    }
                    auto scales = vmovl_s16(vget_low_s16 (i16scales.val[ib]));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8(q8.y[iy][ibl].qs+128*is+32*ib);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                    }
                    if constexpr (nrc_y == 1) {
                        qx[0] = vqtbl1q_s8(values, vandq_u8(bits.val[1], m4));   // 16..19
                        qx[1] = vqtbl1q_s8(values, vandq_u8(bits.val[3], m4));   // 20..23
                        qx[2] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[1], 4));  // 24..27
                        qx[3] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[3], 4));  // 28..31
                    } else {
                        auto shift = vqtbl1q_u8(shifts, shift_shuffle.val[1]);
                        qx[0] = vaddq_s8(shift, vqtbl1q_s8(values, vandq_u8(bits.val[1], m4)));   // 16..19
                        qx[1] = vaddq_s8(shift, vqtbl1q_s8(values, vandq_u8(bits.val[3], m4)));   // 20..23
                        qx[2] = vaddq_s8(shift, vqtbl1q_s8(values, vshrq_n_u8(bits.val[1], 4)));  // 24..27
                        qx[3] = vaddq_s8(shift, vqtbl1q_s8(values, vshrq_n_u8(bits.val[3], 4)));  // 28..31
                    }
                    scales = vmovl_s16(vget_high_s16(i16scales.val[ib]));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8(q8.y[iy][ibl].qs+128*is+32*ib+16);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                    }
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vfmaq_f32(acc[iy], vmulq_f32(d4, vdupq_n_f32(q8.scale(iy, ibl))), vcvtq_f32_s32(isum[iy]));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

template <int nrc_y>
void mul_mat_iq5_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = vdupq_n_u8(0xf);
    auto m3 = vdupq_n_u8(0x30);
    auto ms = vdupq_n_u8(2);
    auto m32 = vdupq_n_s8(-32);
    auto m10 = vdupq_n_u8(0x10);
    uint8x16x2_t shift_shuffle = {
        vreinterpretq_u8_u64(uint64x2_t{0x0101010100000000, 0x0303030302020202}),
        vreinterpretq_u8_u64(uint64x2_t{0x0505050504040404, 0x0707070706060606})
    };
    auto values = vld1q_s8_x2(iq5nl_values);
    int nbl = n / QK_K;
    int8x16_t qx[4];
    int8x16x4_t i8scales;
    int16x8x4_t i16scales;
    float32x4_t acc[nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_iq5_k_r4 * iq5 = (const block_iq5_k_r4 *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq5[ibl].d));
            auto extra8 = vld1_u8(iq5[ibl].extra);
            uint8x16_t extra;
            if constexpr (nrc_y == 1) {
                extra = vcombine_u8(extra8, vshr_n_u8(extra8,1));
            } else {
                extra = vcombine_u8(extra8, extra8);
            }
            auto sl = vld1q_u8_x2(iq5[ibl].scales_l);
            auto sh = vld1q_u8(iq5[ibl].scales_h);
            i8scales.val[0] = vaddq_s8(vorrq_u8(vandq_u8(sl.val[0],  m4), vandq_u8(vshlq_n_u8(sh, 4), m3)), m32);
            i8scales.val[1] = vaddq_s8(vorrq_u8(vandq_u8(sl.val[1],  m4), vandq_u8(vshlq_n_u8(sh, 2), m3)), m32);
            i8scales.val[2] = vaddq_s8(vorrq_u8(vshrq_n_u8(sl.val[0], 4), vandq_u8(sh, m3)), m32);
            i8scales.val[3] = vaddq_s8(vorrq_u8(vshrq_n_u8(sl.val[1], 4), vandq_u8(vshrq_n_u8(sh, 2), m3)), m32);
            int32x4_t isum[nrc_y] = {};
            if constexpr (nrc_y == 1) {
                iq3_4_add_shift<nrc_y, 2>(ibl, q8, i8scales, extra, isum);
            }
            for (int is = 0; is < 2; ++is) {
                i16scales.val[0] = vmovl_s8(vget_low_s8 (i8scales.val[2*is+0]));
                i16scales.val[1] = vmovl_s8(vget_high_s8(i8scales.val[2*is+0]));
                i16scales.val[2] = vmovl_s8(vget_low_s8 (i8scales.val[2*is+1]));
                i16scales.val[3] = vmovl_s8(vget_high_s8(i8scales.val[2*is+1]));
                for (int ib = 0; ib < 4; ++ib) {
                    auto lbits = vld1q_u8_x4(iq5[ibl].qs + 256*is + 64*ib);
                    auto hbits = vld1q_u8(iq5[ibl].qh + 64*is + 16*ib);
                    qx[0] = vorrq_u8(vandq_u8(lbits.val[0],  m4), vandq_u8(m10, vshlq_n_u8(hbits, 4))); // aligns with 1st half of qx[0] in AVX2
                    qx[1] = vorrq_u8(vandq_u8(lbits.val[2],  m4), vandq_u8(m10, hbits));                // aligns with 1st half of qx[1] in AVX2
                    qx[2] = vorrq_u8(vshrq_n_u8(lbits.val[0], 4), vandq_u8(m10, vshlq_n_u8(hbits, 3))); // aligns with 1st half of qx[2] in AVX2
                    qx[3] = vorrq_u8(vshrq_n_u8(lbits.val[2], 4), vandq_u8(m10, vshrq_n_u8(hbits, 1))); // aligns with 1st half of qx[3] in AVX2
                    uint8x16_t shifts;
                    if constexpr (nrc_y == 1) {
                        qx[0] = vqtbl2q_s8(values, qx[0]);  //  0...3 from the 4 rows
                        qx[1] = vqtbl2q_s8(values, qx[1]);  //  4...7
                        qx[2] = vqtbl2q_s8(values, qx[2]);  //  8..11
                        qx[3] = vqtbl2q_s8(values, qx[3]);  // 12..15
                    } else {
                        shifts = vandq_u8(ms, vshlq_n_u8(extra, 1));
                        auto shift = vqtbl1q_u8(shifts, shift_shuffle.val[0]);
                        extra = vshrq_n_u8(extra, 1);
                        qx[0] = vaddq_s8(shift, vqtbl2q_s8(values, qx[0]));  //  0...3 from the 4 rows
                        qx[1] = vaddq_s8(shift, vqtbl2q_s8(values, qx[1]));  //  4...7
                        qx[2] = vaddq_s8(shift, vqtbl2q_s8(values, qx[2]));  //  8..11
                        qx[3] = vaddq_s8(shift, vqtbl2q_s8(values, qx[3]));  // 12..15
                    }
                    auto scales = vmovl_s16(vget_low_s16 (i16scales.val[ib]));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8(q8.y[iy][ibl].qs+128*is+32*ib);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                    }
                    qx[0] = vorrq_u8(vandq_u8(lbits.val[1],  m4), vandq_u8(m10, vshlq_n_u8(hbits, 2))); // aligns with 2nd half of qx[0] in AVX2
                    qx[1] = vorrq_u8(vandq_u8(lbits.val[3],  m4), vandq_u8(m10, vshrq_n_u8(hbits, 2))); // aligns with 2nd half of qx[1] in AVX2
                    qx[2] = vorrq_u8(vshrq_n_u8(lbits.val[1], 4), vandq_u8(m10, vshlq_n_u8(hbits, 1))); // aligns with 2nd half of qx[2] in AVX2
                    qx[3] = vorrq_u8(vshrq_n_u8(lbits.val[3], 4), vandq_u8(m10, vshrq_n_u8(hbits, 3))); // aligns with 2nd half of qx[3] in AVX2
                    if constexpr (nrc_y == 1) {
                        qx[0] = vqtbl2q_s8(values, qx[0]);  //  0...3 from the 4 rows
                        qx[1] = vqtbl2q_s8(values, qx[1]);  //  4...7
                        qx[2] = vqtbl2q_s8(values, qx[2]);  //  8..11
                        qx[3] = vqtbl2q_s8(values, qx[3]);  // 12..15
                    } else {
                        auto shift = vqtbl1q_u8(shifts, shift_shuffle.val[1]);
                        qx[0] = vaddq_s8(shift, vqtbl2q_s8(values, qx[0]));  //  0...3 from the 4 rows
                        qx[1] = vaddq_s8(shift, vqtbl2q_s8(values, qx[1]));  //  4...7
                        qx[2] = vaddq_s8(shift, vqtbl2q_s8(values, qx[2]));  //  8..11
                        qx[3] = vaddq_s8(shift, vqtbl2q_s8(values, qx[3]));  // 12..15
                    }
                    scales = vmovl_s16(vget_high_s16(i16scales.val[ib]));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8(q8.y[iy][ibl].qs+128*is+32*ib+16);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                    }
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vfmaq_f32(acc[iy], vmulq_f32(d4, vdupq_n_f32(q8.scale(iy, ibl))), vcvtq_f32_s32(isum[iy]));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

inline float convert_to_q8_k_r8(float d0, const int8x16x2_t * qx, const int8_t * scales, uint32_t * block, uint32_t * q8_k) {
    auto max_i16 = vdupq_n_u16(0);
    int16x8x4_t q[8];
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        auto scale_l = vdup_n_s8(scales[2*ib32+0]);
        auto scale_h = vdup_n_s8(scales[2*ib32+1]);
        q[ib32].val[0] = vmull_s8(scale_l, vget_low_s8 (qx[ib32].val[0]));
        q[ib32].val[1] = vmull_s8(scale_l, vget_high_s8(qx[ib32].val[0]));
        q[ib32].val[2] = vmull_s8(scale_h, vget_low_s8 (qx[ib32].val[1]));
        q[ib32].val[3] = vmull_s8(scale_h, vget_high_s8(qx[ib32].val[1]));
        max_i16 = vmaxq_u16(max_i16, vmaxq_u16(vabsq_s16(q[ib32].val[0]), vabsq_s16(q[ib32].val[1])));
        max_i16 = vmaxq_u16(max_i16, vmaxq_u16(vabsq_s16(q[ib32].val[2]), vabsq_s16(q[ib32].val[3])));
    }
    uint16_t imax = vmaxvq_u16(max_i16);
    if (!imax) {
        for (int ib32 = 0; ib32 < 8; ++ib32) for (int l = 0; l < 8; ++l) q8_k[64*ib32 + 8*l] = 0;
        return 0.f;
    }
    float dnew = float(imax) * d0;
    //auto max_u32 = vmaxq_u32(vmovl_u16(vget_low_u16(max_i16)), vmovl_u16(vget_high_u16(max_i16)));
    //auto max_f32 = vcvtq_f32_u32(max_u32);
    //auto dnew = vmaxvq_f32(max_f32) * d0;
    bool needs_scaling = true;
    if (dnew <= 1.f) {
        dnew = 1.f; needs_scaling = false;
    }
    auto scale = vdupq_n_f32(1/dnew);
    for (int ib32 = 0; ib32 < 8; ++ib32) {
        if (needs_scaling) {
            for (int l = 0; l < 4; ++l) {
                auto i1 = vcvtnq_s32_f32(vmulq_f32(scale, vcvtq_f32_s32(vmovl_s16(vget_low_s16 (q[ib32].val[l])))));
                auto i2 = vcvtnq_s32_f32(vmulq_f32(scale, vcvtq_f32_s32(vmovl_s16(vget_high_s16(q[ib32].val[l])))));
                q[ib32].val[l] = vcombine_s16(vmovn_s32(i1), vmovn_s32(i2));
            }
        }
        for (int l = 0; l < 2; ++l) {
            auto s8 = vcombine_s8(vmovn_s16(q[ib32].val[2*l+0]), vmovn_s16(q[ib32].val[2*l+1]));
            vst1q_s8((int8_t *)block + 16*l, s8);
        }
        auto qb = q8_k + 64*ib32;
        for (int l = 0; l < 8; ++l) {
            qb[8*l] = block[l];
        }
    }
    return dnew;
}

void iqk_convert_iq2_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq2_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    int8x16_t values;
    {
        auto v = vld1_s8(iq2nl_values);
        values = vcombine_s8(v, v);
    }

    ggml_half dh[8];
    float     dnew[8];
    uint32_t  block[8];
    int8_t    ls[16];

    int8x16x2_t xv[8];

    auto ml = vdupq_n_u8(0x03);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const ggml_half * dptr = (const ggml_half *)((const char *)vx + (ix+k)*bx);
            dh[k] = dptr[0];
            x8[k] = (const block_iq2_ks *)(dptr + 1);
        }
        float32x4x2_t vd{vcvt_f32_f16(vld1_f16((const float16_t *)dh+0)), vcvt_f32_f16(vld1_f16((const float16_t *)dh+4))};
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                auto extra = x8[k][i].extra;
                for (int i128 = 0; i128 < 2; ++i128) {
                    ls[8*i128+0] = ls[8*i128+1] = ((x8[k][i].scales[2*i128+0] & 0xf) | ((extra >> 4) & 0x10)) - 16;
                    ls[8*i128+2] = ls[8*i128+3] = ((x8[k][i].scales[2*i128+0] >>  4) | ((extra >> 5) & 0x10)) - 16;
                    ls[8*i128+4] = ls[8*i128+5] = ((x8[k][i].scales[2*i128+1] & 0xf) | ((extra >> 6) & 0x10)) - 16;
                    ls[8*i128+6] = ls[8*i128+7] = ((x8[k][i].scales[2*i128+1] >>  4) | ((extra >> 7) & 0x10)) - 16;
                    auto bits = vld1q_u8_x2(x8[k][i].qs+32*i128);
                    xv[4*i128+0].val[0] = vandq_u8(bits.val[0], ml);
                    xv[4*i128+0].val[1] = vandq_u8(bits.val[1], ml);
                    xv[4*i128+1].val[0] = vandq_u8(vshrq_n_u8(bits.val[0], 2), ml);
                    xv[4*i128+1].val[1] = vandq_u8(vshrq_n_u8(bits.val[1], 2), ml);
                    xv[4*i128+2].val[0] = vandq_u8(vshrq_n_u8(bits.val[0], 4), ml);
                    xv[4*i128+2].val[1] = vandq_u8(vshrq_n_u8(bits.val[1], 4), ml);
                    xv[4*i128+3].val[0] = vshrq_n_u8(bits.val[0], 6);
                    xv[4*i128+3].val[1] = vshrq_n_u8(bits.val[1], 6);
                    auto shift = vdupq_n_u8((extra << 2) & 0x04);
                    xv[4*i128+0].val[0] = vqtbl1q_s8(values, vaddq_u8(xv[4*i128+0].val[0], shift));
                    xv[4*i128+0].val[1] = vqtbl1q_s8(values, vaddq_u8(xv[4*i128+0].val[1], shift));
                    shift = vdupq_n_u8((extra << 1) & 0x04);
                    xv[4*i128+1].val[0] = vqtbl1q_s8(values, vaddq_u8(xv[4*i128+1].val[0], shift));
                    xv[4*i128+1].val[1] = vqtbl1q_s8(values, vaddq_u8(xv[4*i128+1].val[1], shift));
                    shift = vdupq_n_u8((extra >> 0) & 0x04);
                    xv[4*i128+2].val[0] = vqtbl1q_s8(values, vaddq_u8(xv[4*i128+2].val[0], shift));
                    xv[4*i128+2].val[1] = vqtbl1q_s8(values, vaddq_u8(xv[4*i128+2].val[1], shift));
                    shift = vdupq_n_u8((extra >> 1) & 0x04);
                    xv[4*i128+3].val[0] = vqtbl1q_s8(values, vaddq_u8(xv[4*i128+3].val[0], shift));
                    xv[4*i128+3].val[1] = vqtbl1q_s8(values, vaddq_u8(xv[4*i128+3].val[1], shift));
                    extra >>= 4;
                }
                dnew[k] = convert_to_q8_k_r8(1.f/125, xv, ls, block, (uint32_t *)y[i].qs + k);
            }
            auto d = vld1q_f32_x2(dnew);
            d.val[0] = vmulq_f32(d.val[0], vd.val[0]);
            d.val[1] = vmulq_f32(d.val[1], vd.val[1]);
            vst1_f16((float16_t *)y[i].d + 0,vcvt_f16_f32(d.val[0]));
            vst1_f16((float16_t *)y[i].d + 4,vcvt_f16_f32(d.val[1]));
        }
        y += nb;
    }
}

//struct DequantizerIQ2KL final : public BaseDequantizer<block_iq2_kl, true, true> {
//    DequantizerIQ2KL(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc), shuff(load_shuffle()), shifts(load_shift()) { load_values(values); }
//
//    constexpr static int num_blocks() { return 8; }
//    constexpr static bool should_scale_quants() { return false; }
//
//    template <typename Q8>
//    inline int32x4x2_t new_block(int i, [[maybe_unused]] const Q8& q8, [[maybe_unused]] float32x4_t * acc) {
//        uint32_t aux32; std::memcpy(&aux32, x[i].scales_l, 4);
//        auto scl = vand_u8(vdup_n_u8(0xf), vreinterpret_u8_u32(uint32x2_t{aux32, aux32 >> 4}));
//        auto sch = vandq_u16(vshlq_u16(vdupq_n_u16(x[i].scales_h), shifts), vdupq_n_u16(0x30));
//        auto scales16 = vsubq_s16(vreinterpretq_s16_u16(vorrq_u16(sch, vmovl_u8(scl))), vdupq_n_s16(32));
//        int32x4x2_t scales = {vmovl_s16(vget_low_s16(scales16)), vmovl_s16(vget_high_s16(scales16))};
//        return scales;
//    }
//    inline void prepare(int i, int j) {
//        hbits = j == 0 ? vld1q_u8(x[i].qh) : vshrq_n_u8(hbits, 4);
//        auto lbits = vld1q_u8_x2(x[i].qs+32*j);
//
//        uint8x16x4_t aux;
//        aux.val[0] = vorrq_u8(vandq_u8(m10, vshlq_n_u8(hbits, 4)), vandq_u8(lbits.val[0], vdupq_n_u8(0xf)));
//        aux.val[1] = vorrq_u8(vandq_u8(m10, vshlq_n_u8(hbits, 3)), vshrq_n_u8(lbits.val[0], 4));
//        aux.val[2] = vorrq_u8(vandq_u8(m10, vshlq_n_u8(hbits, 2)), vandq_u8(lbits.val[1], vdupq_n_u8(0xf)));
//        aux.val[3] = vorrq_u8(vandq_u8(m10, vshlq_n_u8(hbits, 1)), vshrq_n_u8(lbits.val[1], 4));
//
//        process_pair(aux.val[0], bits.b1.val+0);
//        process_pair(aux.val[1], bits.b1.val+2);
//        process_pair(aux.val[2], bits.b2.val+0);
//        process_pair(aux.val[3], bits.b2.val+2);
//
//    }
//    static inline int16x8_t load_shift() {
//    }
//    static inline void load_values(int8x16x2_t * values) {
//    }
//    static uint8x16x2_t load_shuffle() {
//        return vld1q_u8_x2(k_shuff);
//    }
//
//    struct { uint8x16x4_t b1, b2; } bits;
//    uint8x16_t hbits;
//    const uint8x16x2_t shuff;
//    const int16x8_t shifts;
//    const uint8x16_t m10 = vdupq_n_u8(0x10);
//    int8x16x2_t values[2];
//
//};

void iqk_convert_iq2_kl_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq2_kl * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    ggml_half dh[8];
    float     dnew[8];
    uint32_t  block[8];
    int8_t    ls[16];

    int8x16x2_t xv[8];

    const uint8x16_t m10 = vdupq_n_u8(0x10);
    static const uint8_t k_shuff[32] = {
        0, 16, 1, 17,  2, 18,  3, 19,  4, 20,  5, 21,  6, 22,  7, 23,
        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
    };
    auto shuff = vld1q_u8_x2(k_shuff);

    int8x16x2_t values[2];
    static const int8_t k_values[64] = {
        -63, -63, -40, -40, -40, -40, -23, -23, -23, -23, -23, -10, -10, -10, -10, 1, 1, 1, 1, 1, 13, 13, 13, 13, 13, 28, 28, 28, 28, 28, 47, 47,
        -23, 13, -63, -10, 13, 47, -40, -23, 1, 13, 28, -63, 1, 13, 47, -23, -10, 1, 13, 28, -40, -23, -10, 1, 13, -63, -23, 1, 28, 47, -23, 13,
    };
    values[0] = vld1q_s8_x2(k_values+ 0);
    values[1] = vld1q_s8_x2(k_values+32);

    auto process_pair = [&values, &shuff] (uint8x16_t x, int8x16_t * val) {
        int8x16x2_t aux{ vqtbl2q_s8(values[0], x), vqtbl2q_s8(values[1], x) };
        val[0] = vqtbl2q_s8(aux, shuff.val[0]);
        val[1] = vqtbl2q_s8(aux, shuff.val[1]);
    };

    uint32_t sl32;
    auto s8 = (const int8_t *)&sl32;

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const ggml_half * dptr = (const ggml_half *)((const char *)vx + (ix+k)*bx);
            dh[k] = dptr[0];
            x8[k] = (const block_iq2_kl *)(dptr + 1);
        }
        float32x4x2_t vd{vcvt_f32_f16(vld1_f16((const float16_t *)dh+0)), vcvt_f32_f16(vld1_f16((const float16_t *)dh+4))};
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                uint32_t aux32; std::memcpy(&aux32, x8[k][i].scales_l, 4);
                auto sh = x8[k][i].scales_h;
                auto hbits = vld1q_u8(x8[k][i].qh);
                for (int i128 = 0; i128 < 2; ++i128) {

                    sl32 = aux32 & 0x0f0f0f0f;
                    ls[8*i128+0] = ls[8*i128+1] = (s8[0] | ((sh << 4) & 0x30)) - 32;
                    ls[8*i128+2] = ls[8*i128+3] = (s8[1] | ((sh << 2) & 0x30)) - 32;
                    ls[8*i128+4] = ls[8*i128+5] = (s8[2] | ((sh >> 0) & 0x30)) - 32;
                    ls[8*i128+6] = ls[8*i128+7] = (s8[3] | ((sh >> 2) & 0x30)) - 32;
                    sh >>= 8; aux32 >>= 4;

                    auto lbits = vld1q_u8_x2(x8[k][i].qs+32*i128);

                    uint8x16x4_t aux;
                    aux.val[0] = vorrq_u8(vandq_u8(m10, vshlq_n_u8(hbits, 4)), vandq_u8(lbits.val[0], vdupq_n_u8(0xf)));
                    aux.val[1] = vorrq_u8(vandq_u8(m10, vshlq_n_u8(hbits, 3)), vshrq_n_u8(lbits.val[0], 4));
                    aux.val[2] = vorrq_u8(vandq_u8(m10, vshlq_n_u8(hbits, 2)), vandq_u8(lbits.val[1], vdupq_n_u8(0xf)));
                    aux.val[3] = vorrq_u8(vandq_u8(m10, vshlq_n_u8(hbits, 1)), vshrq_n_u8(lbits.val[1], 4));
                    hbits = vshrq_n_u8(hbits, 4);

                    process_pair(aux.val[0], xv[4*i128+0].val);
                    process_pair(aux.val[1], xv[4*i128+1].val);
                    process_pair(aux.val[2], xv[4*i128+2].val);
                    process_pair(aux.val[3], xv[4*i128+3].val);
                }
                dnew[k] = convert_to_q8_k_r8(1.f/125, xv, ls, block, (uint32_t *)y[i].qs + k);
            }
            auto d = vld1q_f32_x2(dnew);
            d.val[0] = vmulq_f32(d.val[0], vd.val[0]);
            d.val[1] = vmulq_f32(d.val[1], vd.val[1]);
            vst1_f16((float16_t *)y[i].d + 0,vcvt_f16_f32(d.val[0]));
            vst1_f16((float16_t *)y[i].d + 4,vcvt_f16_f32(d.val[1]));
        }
        y += nb;
    }
}

void iqk_convert_iq4_kss_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq4_kss * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    auto values = vld1q_s8_x2(iq4k_values);

    float drow[8];
    float dnew[8];
    int8_t ls[16];

    int8x16x2_t xv[8];
    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq4_kss *)(dptr + 1);
        }
        auto vd = vld1q_f32_x2(drow);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    auto q4 = x8[k][i].qs + 4*ib32;
                    uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
                    uint8_t s8 = (s32 | (s32 >> 15)) & 0xff;
                    ls[2*ib32+0] = ls[2*ib32+1] = ((s8 & 254) - 127);
                    auto val16 = vandq_u16(vld1q_u16((const uint16_t *)q4), vdupq_n_u16(0xfffe));
                    auto val8 = vreinterpretq_u8_u16(veorq_u16(val16, vshrq_n_u16(val16, 1)));
                    auto& block_values = values.val[s8 & 1];
                    xv[ib32].val[0] = vqtbl1q_s8(block_values, vandq_u8(val8, vdupq_n_u8(0xf)));
                    xv[ib32].val[1] = vqtbl1q_s8(block_values, vshrq_n_u8(val8, 4));
                }
                dnew[k] = convert_to_q8_k_r8(1.f/127, xv, ls, block, (uint32_t *)y[i].qs + k);
            }
            auto d = vld1q_f32_x2(dnew);
            d.val[0] = vmulq_f32(d.val[0], vd.val[0]);
            d.val[1] = vmulq_f32(d.val[1], vd.val[1]);
            vst1_f16((float16_t *)y[i].d + 0, vcvt_f16_f32(d.val[0]));
            vst1_f16((float16_t *)y[i].d + 4, vcvt_f16_f32(d.val[1]));
        }
        y += nb;
    }
}

void iqk_convert_iq4_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq4_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    auto values = vld1q_s8_x2(iq4k_values);

    float drow[8];
    float dnew[8];
    int8_t ls[16];

    int8x16x2_t xv[8];
    uint32_t block[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq4_ks *)(dptr + 1);
        }
        auto vd = vld1q_f32_x2(drow);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    ls[2*ib32+0] = ((x8[k][i].scales[ib32] & 254) - 127);
                    ls[2*ib32+1] = ls[2*ib32+0];
                    auto aux128 = vld1q_u8(x8[k][i].qs+16*ib32);
                    auto& block_values = values.val[x8[k][i].scales[ib32] & 1];
                    xv[ib32].val[0] = vqtbl1q_s8(block_values, vandq_u8(aux128, vdupq_n_u8(0xf)));
                    xv[ib32].val[1] = vqtbl1q_s8(block_values, vshrq_n_u8(aux128, 4));
                }
                dnew[k] = convert_to_q8_k_r8(1.f/127, xv, ls, block, (uint32_t *)y[i].qs + k);
            }
            auto d = vld1q_f32_x2(dnew);
            d.val[0] = vmulq_f32(d.val[0], vd.val[0]);
            d.val[1] = vmulq_f32(d.val[1], vd.val[1]);
            vst1_f16((float16_t *)y[i].d + 0, vcvt_f16_f32(d.val[0]));
            vst1_f16((float16_t *)y[i].d + 4, vcvt_f16_f32(d.val[1]));
        }
        y += nb;
    }
}

void iqk_convert_iq5_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq5_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    int8x16x2_t values[2];
    {
        values[0] = vld1q_s8_x2(iq5nl_values);
        values[1] = vld1q_s8_x2(iq5nl_values + 32);
    }

    float drow[8];
    float dnew[8];
    int8_t ls[16];

    int8x16x2_t xv[8];
    uint32_t block[8];

    auto ml = vdupq_n_u8(0x0f);
    auto mh = vdupq_n_u8(0x10);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            const float * dptr = (const float *)((const char *)vx + (ix + k)*bx);
            drow[k] = dptr[0];
            x8[k] = (const block_iq5_ks *)(dptr + 1);
        }
        auto vd = vld1q_f32_x2(drow);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                auto hbits = vld1q_u8_x2(x8[k][i].qh);
                for (int ib64 = 0; ib64 < 4; ++ib64) {
                    ls[4*ib64+0] = ls[4*ib64+1] = (x8[k][i].scales[2*ib64+0] & 254) - 127;
                    ls[4*ib64+2] = ls[4*ib64+3] = (x8[k][i].scales[2*ib64+1] & 254) - 127;
                    auto bits = vld1q_u8_x2(x8[k][i].qs+32*ib64);
                    auto& val1 = values[x8[k][i].scales[2*ib64+0] & 1];
                    auto& val2 = values[x8[k][i].scales[2*ib64+1] & 1];
                    xv[2*ib64+0].val[0] = vqtbl2q_s8(val1, vorrq_u8(vandq_u8(bits.val[0], ml),  vandq_u8(vshlq_n_u8(hbits.val[0], 4), mh)));
                    xv[2*ib64+0].val[1] = vqtbl2q_s8(val1, vorrq_u8(vandq_u8(bits.val[1], ml),  vandq_u8(vshlq_n_u8(hbits.val[1], 4), mh)));
                    xv[2*ib64+1].val[0] = vqtbl2q_s8(val2, vorrq_u8(vshrq_n_u8(bits.val[0], 4), vandq_u8(vshlq_n_u8(hbits.val[0], 3), mh)));
                    xv[2*ib64+1].val[1] = vqtbl2q_s8(val2, vorrq_u8(vshrq_n_u8(bits.val[1], 4), vandq_u8(vshlq_n_u8(hbits.val[1], 3), mh)));
                    hbits.val[0] = vshrq_n_u8(hbits.val[0], 2);
                    hbits.val[1] = vshrq_n_u8(hbits.val[1], 2);
                }
                dnew[k] = convert_to_q8_k_r8(1.f/127, xv, ls, block, (uint32_t *)y[i].qs + k);
            }
            auto d = vld1q_f32_x2(dnew);
            d.val[0] = vmulq_f32(d.val[0], vd.val[0]);
            d.val[1] = vmulq_f32(d.val[1], vd.val[1]);
            vst1_f16((float16_t *)y[i].d + 0, vcvt_f16_f32(d.val[0]));
            vst1_f16((float16_t *)y[i].d + 4, vcvt_f16_f32(d.val[1]));
        }
        y += nb;
    }
}

void iqk_convert_iq2_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq2_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    int8x16x2_t values;
    {
        auto v1 = vld1_s8(iq2nl_values);
        auto v2 = vreinterpret_s8_u64(vshr_n_u64(vreinterpret_u64_s8(v1), 32));
        values.val[0] = vcombine_s8(v1, v1);
        values.val[1] = vcombine_s8(v2, v2);
    }

    int8x16x2_t xv[8];
    uint32_t block[8];
    int8_t   ls[16];

    auto ml = vdupq_n_u8(0x03);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq2_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                auto extra = x8[k][i].extra;
                for (int i128 = 0; i128 < 2; ++i128) {

                    ls[8*i128+0] = ((x8[k][i].scales[4*i128+0] & 0xf) - 8);
                    ls[8*i128+1] = ((x8[k][i].scales[4*i128+0] >>  4) - 8);
                    ls[8*i128+2] = ((x8[k][i].scales[4*i128+1] & 0xf) - 8);
                    ls[8*i128+3] = ((x8[k][i].scales[4*i128+1] >>  4) - 8);
                    ls[8*i128+4] = ((x8[k][i].scales[4*i128+2] & 0xf) - 8);
                    ls[8*i128+5] = ((x8[k][i].scales[4*i128+2] >>  4) - 8);
                    ls[8*i128+6] = ((x8[k][i].scales[4*i128+3] & 0xf) - 8);
                    ls[8*i128+7] = ((x8[k][i].scales[4*i128+3] >>  4) - 8);

                    auto bits = vld1q_u8_x2(x8[k][i].qs+32*i128);
                    xv[4*i128+0].val[0] = vqtbl1q_s8(values.val[extra & 1], vandq_u8(bits.val[0], ml)); extra >>= 1;
                    xv[4*i128+0].val[1] = vqtbl1q_s8(values.val[extra & 1], vandq_u8(bits.val[1], ml)); extra >>= 1;
                    xv[4*i128+1].val[0] = vqtbl1q_s8(values.val[extra & 1], vandq_u8(vshrq_n_u8(bits.val[0], 2), ml)); extra >>= 1;
                    xv[4*i128+1].val[1] = vqtbl1q_s8(values.val[extra & 1], vandq_u8(vshrq_n_u8(bits.val[1], 2), ml)); extra >>= 1;
                    xv[4*i128+2].val[0] = vqtbl1q_s8(values.val[extra & 1], vandq_u8(vshrq_n_u8(bits.val[0], 4), ml)); extra >>= 1;
                    xv[4*i128+2].val[1] = vqtbl1q_s8(values.val[extra & 1], vandq_u8(vshrq_n_u8(bits.val[1], 4), ml)); extra >>= 1;
                    xv[4*i128+3].val[0] = vqtbl1q_s8(values.val[extra & 1], vshrq_n_u8(bits.val[0], 6)); extra >>= 1;
                    xv[4*i128+3].val[1] = vqtbl1q_s8(values.val[extra & 1], vshrq_n_u8(bits.val[1], 6)); extra >>= 1;
                }
                float dnew = convert_to_q8_k_r8(1.f/127, xv, ls, block, (uint32_t *)y[i].qs + k);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
}

void iqk_convert_iq3_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq3_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    int8x16x2_t values;
    {
        auto v1 = vld1_s8(iq3nl_values+0);
        auto v2 = vld1_s8(iq3nl_values+8);
        values.val[0] = vcombine_s8(v1, v1);
        values.val[1] = vcombine_s8(v2, v2);
    }

    ggml_half dh[8];
    int8x16x2_t xv[8];
    uint32_t block[8];
    int8_t   ls[16];

    auto ml = vdupq_n_u8(0x03);
    auto mh = vdupq_n_u8(0x04);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            auto dptr = (const ggml_half *)((const char *)vx + (ix+k)*bx);
            dh[k] = dptr[0];
            x8[k] = (const block_iq3_ks *)(dptr + 1);
        }
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                auto extra = x8[k][i].extra;
                auto extra_v = extra >> 8;
                auto hbits = vld1q_u8_x2(x8[k][i].qh);
                for (int i128 = 0; i128 < 2; ++i128) {

                    ls[8*i128+0] = ls[8*i128+1] = int8_t(((x8[k][i].scales[0] >> 4*i128) & 0xf) | ((extra << 4) & 0x10)) - 16;
                    ls[8*i128+2] = ls[8*i128+3] = int8_t(((x8[k][i].scales[1] >> 4*i128) & 0xf) | ((extra << 3) & 0x10)) - 16;
                    ls[8*i128+4] = ls[8*i128+5] = int8_t(((x8[k][i].scales[2] >> 4*i128) & 0xf) | ((extra << 2) & 0x10)) - 16;
                    ls[8*i128+6] = ls[8*i128+7] = int8_t(((x8[k][i].scales[3] >> 4*i128) & 0xf) | ((extra << 1) & 0x10)) - 16;

                    auto bits = vld1q_u8_x2(x8[k][i].qs+32*i128);
                    xv[4*i128+0].val[0] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vandq_u8(bits.val[0], ml), vandq_u8(vshlq_n_u8(hbits.val[0], 2), mh)));
                    xv[4*i128+0].val[1] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vandq_u8(bits.val[1], ml), vandq_u8(vshlq_n_u8(hbits.val[1], 2), mh))); extra_v >>= 1;
                    xv[4*i128+1].val[0] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vandq_u8(vshrq_n_u8(bits.val[0], 2), ml), vandq_u8(vshlq_n_u8(hbits.val[0], 1), mh)));
                    xv[4*i128+1].val[1] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vandq_u8(vshrq_n_u8(bits.val[1], 2), ml), vandq_u8(vshlq_n_u8(hbits.val[1], 1), mh))); extra_v >>= 1;
                    xv[4*i128+2].val[0] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vandq_u8(vshrq_n_u8(bits.val[0], 4), ml), vandq_u8(hbits.val[0], mh)));
                    xv[4*i128+2].val[1] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vandq_u8(vshrq_n_u8(bits.val[1], 4), ml), vandq_u8(hbits.val[1], mh))); extra_v >>= 1;
                    xv[4*i128+3].val[0] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vshrq_n_u8(bits.val[0], 6), vandq_u8(vshrq_n_u8(hbits.val[0], 1), mh)));
                    xv[4*i128+3].val[1] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vshrq_n_u8(bits.val[1], 6), vandq_u8(vshrq_n_u8(hbits.val[1], 1), mh))); extra_v >>= 1;
                    hbits.val[0] = vshrq_n_u8(hbits.val[0], 4);
                    hbits.val[1] = vshrq_n_u8(hbits.val[1], 4);
                    extra >>= 4;
                }
                float dnew = convert_to_q8_k_r8(1.f/127, xv, ls, block, (uint32_t *)y[i].qs + k);
                y[i].d[k] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(dh[k])*dnew);
            }
        }
        y += nb;
    }
}

void iqk_convert_iq3_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq3_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    int8x16x2_t values;
    {
        auto v1 = vld1_s8(iq3nl_values+0);
        auto v2 = vld1_s8(iq3nl_values+8);
        values.val[0] = vcombine_s8(v1, v1);
        values.val[1] = vcombine_s8(v2, v2);
    }

    int8x16x2_t xv[8];
    uint32_t block[8];
    int8_t   ls[16];

    auto ml = vdupq_n_u8(0x03);
    auto mh = vdupq_n_u8(0x04);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq3_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                auto extra = x8[k][i].extra;
                auto hbits = vld1q_u8_x2(x8[k][i].qh);
                uint16_t sh = x8[k][i].scales_h;
                for (int i128 = 0; i128 < 2; ++i128) {

                    ls[8*i128+0] = ((2*(x8[k][i].scales_l[4*i128+0] & 0xf) + 1) * ((sh & 0x01) ? -1 : 1));
                    ls[8*i128+1] = ((2*(x8[k][i].scales_l[4*i128+0] >>  4) + 1) * ((sh & 0x02) ? -1 : 1));
                    ls[8*i128+2] = ((2*(x8[k][i].scales_l[4*i128+1] & 0xf) + 1) * ((sh & 0x04) ? -1 : 1));
                    ls[8*i128+3] = ((2*(x8[k][i].scales_l[4*i128+1] >>  4) + 1) * ((sh & 0x08) ? -1 : 1));
                    ls[8*i128+4] = ((2*(x8[k][i].scales_l[4*i128+2] & 0xf) + 1) * ((sh & 0x10) ? -1 : 1));
                    ls[8*i128+5] = ((2*(x8[k][i].scales_l[4*i128+2] >>  4) + 1) * ((sh & 0x20) ? -1 : 1));
                    ls[8*i128+6] = ((2*(x8[k][i].scales_l[4*i128+3] & 0xf) + 1) * ((sh & 0x40) ? -1 : 1));
                    ls[8*i128+7] = ((2*(x8[k][i].scales_l[4*i128+3] >>  4) + 1) * ((sh & 0x80) ? -1 : 1));

                    auto bits = vld1q_u8_x2(x8[k][i].qs+32*i128);
                    xv[4*i128+0].val[0] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(vandq_u8(bits.val[0], ml), vandq_u8(vshlq_n_u8(hbits.val[0], 2), mh))); extra >>= 1;
                    xv[4*i128+0].val[1] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(vandq_u8(bits.val[1], ml), vandq_u8(vshlq_n_u8(hbits.val[1], 2), mh))); extra >>= 1;
                    xv[4*i128+1].val[0] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(vandq_u8(vshrq_n_u8(bits.val[0], 2), ml), vandq_u8(vshlq_n_u8(hbits.val[0], 1), mh))); extra >>= 1;
                    xv[4*i128+1].val[1] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(vandq_u8(vshrq_n_u8(bits.val[1], 2), ml), vandq_u8(vshlq_n_u8(hbits.val[1], 1), mh))); extra >>= 1;
                    xv[4*i128+2].val[0] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(vandq_u8(vshrq_n_u8(bits.val[0], 4), ml), vandq_u8(hbits.val[0], mh))); extra >>= 1;
                    xv[4*i128+2].val[1] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(vandq_u8(vshrq_n_u8(bits.val[1], 4), ml), vandq_u8(hbits.val[1], mh))); extra >>= 1;
                    xv[4*i128+3].val[0] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(vshrq_n_u8(bits.val[0], 6), vandq_u8(vshrq_n_u8(hbits.val[0], 1), mh))); extra >>= 1;
                    xv[4*i128+3].val[1] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(vshrq_n_u8(bits.val[1], 6), vandq_u8(vshrq_n_u8(hbits.val[1], 1), mh))); extra >>= 1;
                    hbits.val[0] = vshrq_n_u8(hbits.val[0], 4);
                    hbits.val[1] = vshrq_n_u8(hbits.val[1], 4);
                    sh >>= 8;
                }
                float dnew = convert_to_q8_k_r8(1.f/127, xv, ls, block, (uint32_t *)y[i].qs + k);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
}

void iqk_convert_iq4_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq4_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    int8x16x2_t values = vld1q_s8_x2(iq4k_values);

    int8x16x2_t xv[8];
    uint32_t block[8];
    int8_t   ls[16];

    auto ml = vdupq_n_u8(0xf);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq4_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                auto extra = x8[k][i].extra;
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const uint8_t sh = x8[k][i].scales_h[ib32/2] >> 4*(ib32%2);
                    ls[2*ib32+0] = ((x8[k][i].scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32;
                    ls[2*ib32+1] = ((x8[k][i].scales_l[ib32] >>  4) | ((sh << 2) & 0x30)) - 32;
                    auto bits = vld1q_u8(x8[k][i].qs+16*ib32);
                    xv[ib32].val[0] = vqtbl1q_s8(values.val[extra & 1], vandq_u8(bits,  ml)); extra >>= 1;
                    xv[ib32].val[1] = vqtbl1q_s8(values.val[extra & 1], vshrq_n_u8(bits, 4)); extra >>= 1;
                }
                float dnew = convert_to_q8_k_r8(1.f/127, xv, ls, block, (uint32_t *)y[i].qs + k);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
}

void iqk_convert_iq5_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq5_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    int8x16x2_t values[2];
    values[0] = vld1q_s8_x2(iq5nl_values);
    values[1] = vld1q_s8_x2(iq5nl_values + 32);

    int8x16x2_t xv[8];
    uint32_t block[8];
    int8_t   ls[16];

    auto ml = vdupq_n_u8(0x0f);
    auto mh = vdupq_n_u8(0x10);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq5_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                auto extra = x8[k][i].extra;
                auto hbits = vld1q_u8_x2(x8[k][i].qh);
                for (int ib64 = 0; ib64 < 4; ++ib64) {
                    ls[4*ib64+0] = ((x8[k][i].scales_l[2*ib64+0] & 0xf) | ((x8[k][i].scales_h[ib64] << 4) & 0x30)) - 32;
                    ls[4*ib64+1] = ((x8[k][i].scales_l[2*ib64+0] >>  4) | ((x8[k][i].scales_h[ib64] << 2) & 0x30)) - 32;
                    ls[4*ib64+2] = ((x8[k][i].scales_l[2*ib64+1] & 0xf) | ((x8[k][i].scales_h[ib64] >> 0) & 0x30)) - 32;
                    ls[4*ib64+3] = ((x8[k][i].scales_l[2*ib64+1] >>  4) | ((x8[k][i].scales_h[ib64] >> 2) & 0x30)) - 32;
                    auto bits = vld1q_u8_x2(x8[k][i].qs+32*ib64);
                    xv[2*ib64+0].val[0] = vqtbl2q_s8(values[(extra >> 0) & 1], vorrq_u8(vandq_u8(bits.val[0], ml),  vandq_u8(vshlq_n_u8(hbits.val[0], 4), mh)));
                    xv[2*ib64+0].val[1] = vqtbl2q_s8(values[(extra >> 1) & 1], vorrq_u8(vandq_u8(bits.val[1], ml),  vandq_u8(vshlq_n_u8(hbits.val[1], 4), mh)));
                    xv[2*ib64+1].val[0] = vqtbl2q_s8(values[(extra >> 2) & 1], vorrq_u8(vshrq_n_u8(bits.val[0], 4), vandq_u8(vshlq_n_u8(hbits.val[0], 3), mh)));
                    xv[2*ib64+1].val[1] = vqtbl2q_s8(values[(extra >> 3) & 1], vorrq_u8(vshrq_n_u8(bits.val[1], 4), vandq_u8(vshlq_n_u8(hbits.val[1], 3), mh)));
                    hbits.val[0] = vshrq_n_u8(hbits.val[0], 2);
                    hbits.val[1] = vshrq_n_u8(hbits.val[1], 2);
                    extra >>= 4;
                }
                float dnew = convert_to_q8_k_r8(1.f/127, xv, ls, block, (uint32_t *)y[i].qs + k);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
}

void iqk_convert_iq6_k_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq6_k * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    int8x16x4_t values[2];
    values[0] = vld1q_s8_x4(iq6nl_values);
    values[1] = vld1q_s8_x4(iq6nl_values+64);

    int8x16x2_t xv[8];
    uint32_t block[8];

    auto ml = vdupq_n_u8(0x0f);
    auto mh = vdupq_n_u8(0x30);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq6_k *)((const char *)vx + (ix+k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                float d = GGML_FP16_TO_FP32(x8[k][i].d);
                auto extra = x8[k][i].extra;
                for (int i128 = 0; i128 < 2; ++i128) {
                    auto hbits = vld1q_u8_x2(x8[k][i].qh+i128*32);
                    auto bits  = vld1q_u8_x4(x8[k][i].qs+i128*64);
                    xv[4*i128+0].val[0] = vqtbl4q_s8(values[(extra >> 0) & 1], vorrq_u8(vandq_u8(bits.val[0], ml),  vandq_u8(vshlq_n_u8(hbits.val[0], 4), mh)));
                    xv[4*i128+0].val[1] = vqtbl4q_s8(values[(extra >> 1) & 1], vorrq_u8(vandq_u8(bits.val[1], ml),  vandq_u8(vshlq_n_u8(hbits.val[1], 4), mh)));
                    xv[4*i128+1].val[0] = vqtbl4q_s8(values[(extra >> 2) & 1], vorrq_u8(vshrq_n_u8(bits.val[0], 4), vandq_u8(vshlq_n_u8(hbits.val[0], 2), mh)));
                    xv[4*i128+1].val[1] = vqtbl4q_s8(values[(extra >> 3) & 1], vorrq_u8(vshrq_n_u8(bits.val[1], 4), vandq_u8(vshlq_n_u8(hbits.val[1], 2), mh)));
                    xv[4*i128+2].val[0] = vqtbl4q_s8(values[(extra >> 4) & 1], vorrq_u8(vandq_u8(bits.val[2], ml),  vandq_u8(hbits.val[0], mh)));
                    xv[4*i128+2].val[1] = vqtbl4q_s8(values[(extra >> 5) & 1], vorrq_u8(vandq_u8(bits.val[3], ml),  vandq_u8(hbits.val[1], mh)));
                    xv[4*i128+3].val[0] = vqtbl4q_s8(values[(extra >> 6) & 1], vorrq_u8(vshrq_n_u8(bits.val[2], 4), vandq_u8(vshrq_n_u8(hbits.val[0], 2), mh)));
                    xv[4*i128+3].val[1] = vqtbl4q_s8(values[(extra >> 7) & 1], vorrq_u8(vshrq_n_u8(bits.val[3], 4), vandq_u8(vshrq_n_u8(hbits.val[1], 2), mh)));
                    extra >>= 8;
                }
                float dnew = convert_to_q8_k_r8(1.f/127, xv, x8[k][i].scales, block, (uint32_t *)y[i].qs + k);
                y[i].d[k] = GGML_FP32_TO_FP16(d*dnew);
            }
        }
        y += nb;
    }
}

}

bool iqk_convert_iqk_quants_q80_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    if (n%QK_K != 0 || nrc_x%8 != 0) return false;
    switch (ggml_type(type)) {
        case GGML_TYPE_IQ2_KS : iqk_convert_iq2_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ2_K  : iqk_convert_iq2_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ2_KL : iqk_convert_iq2_kl_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ3_KS : iqk_convert_iq3_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ3_K  : iqk_convert_iq3_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_KSS: iqk_convert_iq4_kss_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_KS : iqk_convert_iq4_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_K  : iqk_convert_iq4_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ5_KS : iqk_convert_iq5_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ5_K  : iqk_convert_iq5_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ6_K  : iqk_convert_iq6_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        default: return false;
    }
    return true;
}

bool iqk_set_kernels_iqk_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, [[maybe_unused]] mul_mat_t& func16) {

    if (ne00%QK_K != 0 || ggml_type(typeB) != GGML_TYPE_Q8_K) {
        return false;
    }

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_IQ2_KS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ2KS, kernels);
            break;
        case GGML_TYPE_IQ2_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ2K, kernels);
            break;
        case GGML_TYPE_IQ2_KL:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ2KL, kernels);
            break;
        case GGML_TYPE_IQ3_KS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ3KS, kernels);
            break;
        case GGML_TYPE_IQ3_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ3K, kernels);
            break;
        case GGML_TYPE_IQ4_KSS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ4KSS, kernels);
            break;
       case GGML_TYPE_IQ4_KS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ4KS, kernels);
            break;
        case GGML_TYPE_IQ4_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ4K, kernels);
            break;
        case GGML_TYPE_IQ5_KS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ5KS, kernels);
            break;
        case GGML_TYPE_IQ5_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ5K, kernels);
            break;
        case GGML_TYPE_IQ6_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ6K, kernels);
            break;
        case GGML_TYPE_IQ2_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_k_r4_q8_k, kernels);
            break;
        case GGML_TYPE_IQ3_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq3_k_r4_q8_k, kernels);
            break;
        case GGML_TYPE_IQ4_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_k_r4_q8_k, kernels);
            break;
        case GGML_TYPE_IQ4_KS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_ks_r4_q8_k, kernels);
            break;
        case GGML_TYPE_IQ5_KS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq5_ks_r4_q8_k, kernels);
            break;
        case GGML_TYPE_IQ5_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq5_k_r4_q8_k, kernels);
            break;
        default:
            return false;
    }

    return true;

}

#endif

#endif
