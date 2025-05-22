#include "iqk_gemm_kquants.h"

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#ifdef __x86_64__

namespace {

// Handles q4_K and q5_K scales/mins
struct Scales8K {
    template <typename Q8>
    inline __m256i process_mins_and_scales(const uint8_t * data, float c, int i, const Q8& q8, __m256 * accd) {
        make_q4_scales(data, utmp);
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
        accum_mins(mins128, q8, i, c, accd);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        return MM256_SET_M128I(sc128, sc128);
    }
#ifdef HAVE_FANCY_SIMD
    template <typename Q8>
    inline __m512i process_mins_and_scales_64(const uint8_t * data, float c, int i, const Q8& q8, __m256 * accd) {
        auto scales = process_mins_and_scales(data, c, i, q8, accd);
        return _mm512_inserti32x8(_mm512_castsi256_si512(scales), scales, 1);
    }
#endif
    template <typename Q8>
    inline void accum_mins(const __m128i& mins128, const Q8& q8, int i, float c, __m256 * accd) const {
        base.accum_mins(mins128, q8, i, c, accd);
    }
#ifdef HAVE_FANCY_SIMD
    const __m512i shuffles512[2] = {
        _mm512_set_epi64(0x0706070607060706, 0x0302030203020302, 0x0706070607060706, 0x0302030203020302,
                         0x0504050405040504, 0x0100010001000100, 0x0504050405040504, 0x0100010001000100),
        _mm512_set_epi64(0x0f0e0f0e0f0e0f0e, 0x0b0a0b0a0b0a0b0a, 0x0f0e0f0e0f0e0f0e, 0x0b0a0b0a0b0a0b0a,
                         0x0d0c0d0c0d0c0d0c, 0x0908090809080908, 0x0d0c0d0c0d0c0d0c, 0x0908090809080908)
    };
#endif
    Scales8KBase base;

    uint32_t utmp[4];
};

template <typename Q8>
inline void process_mins_16(const __m256i& all_scales, const Q8& q8, int i, float d, __m256 * accm) {
    for (int iy = 0; iy < Q8::nrc_y; ++iy) {
        const __m256i prod  = _mm256_madd_epi16(all_scales, q8.load_bsums(iy, i));
        accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accm[iy]);
    }
}
inline void prepare_scales_16(const __m256i& all_scales, __m256i * scales) {
    const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
    const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
    scales[0] = MM256_SET_M128I(l_scales, l_scales);
    scales[1] = MM256_SET_M128I(h_scales, h_scales);
}

// Handles q3_K scales
struct ScaleQ3 {
    inline __m128i make_scales(const uint16_t * s8) const {
        const uint16_t * scales16 = (const uint16_t *)s8;
        uint32_t aux0 = scales16[0] | (scales16[1] << 16);
        uint32_t aux1 = scales16[2] | (scales16[3] << 16);
        uint32_t aux2 = scales16[4] | (scales16[5] << 16);
        __m128i scales128 = _mm_set_epi32(
            ((aux1 >> 4) & 0x0f0f0f0f) | ((aux2 >> 2) & 0x30303030),
            ((aux0 >> 4) & 0x0f0f0f0f) | ((aux2 >> 0) & 0x30303030),
             (aux1       & 0x0f0f0f0f) | ((aux2 << 2) & 0x30303030),
             (aux0       & 0x0f0f0f0f) | ((aux2 << 4) & 0x30303030));
        return _mm_add_epi8(scales128, m32);
    }
    const __m128i m32 = _mm_set1_epi8(-32);
};

struct Scale16 {
    inline void make_scales(const __m128i& scales8, __m512i * scales) const {
        auto all_scales8 = MM256_SET_M128I(scales8, scales8);
        auto scales1 = _mm256_shuffle_epi8(all_scales8, shuffle1);
        auto scales2 = _mm256_shuffle_epi8(all_scales8, shuffle2);
        scales[0] = _mm512_cvtepi8_epi16(scales1);
        scales[1] = _mm512_cvtepi8_epi16(scales2);
    }
    template <typename Q8>
    inline void process_mins_and_scales(int i, float c, const __m128i& mins8, const __m128i& scales8,
        const Q8& q8, __m256 * accm, __m512i * scales) const {
        process_mins_16(_mm256_cvtepi8_epi16(mins8), q8, i, c, accm);
        make_scales(scales8, scales);
    }
    const __m256i shuffle1 = _mm256_set_epi32(0x07070707, 0x03030303, 0x06060606, 0x02020202,
                                              0x05050505, 0x01010101, 0x04040404, 0x00000000);
    const __m256i shuffle2 = _mm256_set_epi32(0x0f0f0f0f, 0x0b0b0b0b, 0x0e0e0e0e, 0x0a0a0a0a,
                                              0x0d0d0d0d, 0x09090909, 0x0c0c0c0c, 0x08080808);
};

template <typename Q8>
inline void process_mins_and_scales_16(const __m128i& scales128, const Q8& q8, int i, float d,
    __m256 * accm, __m256i * scales) {
    const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
    process_mins_16(all_scales, q8, i, d, accm);
    prepare_scales_16(all_scales, scales);
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

struct ScaleIQ4XS {
    inline __m128i make_scales(const uint32_t scales_l, const uint16_t scales_h) {
        uint32_t tmp32 = scales_h | (scales_h << 14);
        const __m128i sh = _mm_slli_epi16(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(tmp32), hshift), hmask), 4);
        const __m128i sl = _mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(scales_l), lshift), lmask);
        return _mm_add_epi16(_mm_or_si128(sh, _mm_cvtepi8_epi16(_mm_shuffle_epi8(sl, lshuffle))), m32);
    }
    const __m128i hshift = _mm_set_epi32(12, 8, 4, 0);
    const __m128i lshift = _mm_set_epi32(4, 0, 4, 0);
    const __m128i hmask  = _mm_set1_epi16(0x03);
    const __m128i lmask  = _mm_set1_epi8(0xf);
    const __m128i lshuffle = _mm_set_epi32(0x07030602, 0x05010400, 0x07030602, 0x05010400);
    const __m128i m32 = _mm_set1_epi16(-32);
};

#ifdef HAVE_FANCY_SIMD
//====================================== Zen4 ==================================================

struct HighBit5 {
    inline void apply(const uint8_t * h, Q4Bits& bits) {
        auto hbits256 = _mm256_loadu_si256((const __m256i *)h);
        auto hbits = _mm512_inserti32x8(_mm512_castsi256_si512(hbits256), _mm256_srli_epi16(hbits256, 1), 1);
        bits.values[0] = _mm512_or_si512(bits.values[0], _mm512_and_si512(_mm512_slli_epi16(hbits, 4), mh));
        bits.values[1] = _mm512_or_si512(bits.values[1], _mm512_and_si512(_mm512_slli_epi16(hbits, 2), mh));
        bits.values[2] = _mm512_or_si512(bits.values[2], _mm512_and_si512(hbits, mh));
        bits.values[3] = _mm512_or_si512(bits.values[3], _mm512_and_si512(_mm512_srli_epi16(hbits, 2), mh));
    }
    const __m512i mh = _mm512_set1_epi8(0x10);
};

struct HighBit3 {
    inline void apply(const uint8_t * h, Q2Bits& bits) {
        auto hbits256 = _mm256_loadu_si256((const __m256i *)h);
        auto hbits = _mm512_inserti32x8(_mm512_castsi256_si512(hbits256), _mm256_srli_epi16(hbits256, 1), 1);
        bits.values[0] = _mm512_or_si512(bits.values[0], _mm512_and_si512(_mm512_slli_epi16(hbits, 2), mh));
        bits.values[1] = _mm512_or_si512(bits.values[1], _mm512_and_si512(hbits, mh));
        bits.values[2] = _mm512_or_si512(bits.values[2], _mm512_and_si512(_mm512_srli_epi16(hbits, 2), mh));
        bits.values[3] = _mm512_or_si512(bits.values[3], _mm512_and_si512(_mm512_srli_epi16(hbits, 4), mh));
    }
    const __m512i mh = _mm512_set1_epi8(0x04);
};


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

struct DequantizerQ2K final : public BaseDequantizer<block_q2_K> {
    DequantizerQ2K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        bits.prepare(x[i].qs);
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        sc16.process_mins_and_scales(i, -GGML_FP16_TO_FP32(x[i].dmin), mins8, scales8, q8, accm, scales);
    }

    Q2Bits bits;
    Scale16 sc16;
    const __m128i m4 = _mm_set1_epi8(0xf);

};

struct DequantizerQ3K final : public BaseDequantizer<block_q3_K> {
    DequantizerQ3K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        bits.prepare(x[i].qs);
        hbits.apply(x[i].hmask, bits);
        auto scales128 = sc3.make_scales((const uint16_t *)x[i].scales);
        sc16.process_mins_and_scales(i, -4.f*d, scales128, scales128, q8, accm, scales);
    }

    Q2Bits bits;
    HighBit3 hbits;
    ScaleQ3 sc3;
    Scale16 sc16;
    const __m128i m4  = _mm_set1_epi8(0xf);
    const __m128i m32 = _mm_set1_epi8(-32);
};

struct DequantizerQ4K final : public BaseDequantizer<block_q4_K> {
    DequantizerQ4K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accd, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        bits.prepare(x[i].qs);
        auto all_scales = s8k.process_mins_and_scales_64(x[i].scales, -GGML_FP16_TO_FP32(x[i].dmin), i, q8, accd);
        scales[0] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[0]);
        scales[1] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[1]);
    }

    Q4Bits bits;
    Scales8K s8k;
};

struct DequantizerQ5K final : public BaseDequantizer<block_q5_K> {
    DequantizerQ5K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accd, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        bits.prepare(x[i].qs);
        hbits.apply(x[i].qh, bits);
        auto all_scales = s8k.process_mins_and_scales_64(x[i].scales, -GGML_FP16_TO_FP32(x[i].dmin), i, q8, accd);
        scales[0] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[0]);
        scales[1] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[1]);
    }

    Q4Bits bits;
    HighBit5 hbits;
    Scales8K s8k;
};

struct DequantizerQ6K final : public BaseDequantizer<block_q6_K> {
    DequantizerQ6K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        bits.prepare64(x[i].ql);
        add_high_bits(x[i].qh, bits);
        auto scales128 = _mm_loadu_si128((const __m128i *)x[i].scales);
        sc16.process_mins_and_scales(i, -32.f*d, scales128, scales128, q8, accm, scales);
    }

    inline void add_high_bits(const uint8_t * qh, Q4Bits& bits) const {
        auto hbits = _mm512_loadu_si512((const __m512i *)qh);
        auto tmp1 = _mm512_and_si512(_mm512_slli_epi16(hbits, 4), mh);
        auto tmp2 = _mm512_and_si512(_mm512_slli_epi16(hbits, 2), mh);
        bits.values[0] = _mm512_or_si512(bits.values[0], _mm512_permutex2var_epi64(tmp1, bits.perm.permute1, tmp2));
        bits.values[2] = _mm512_or_si512(bits.values[2], _mm512_permutex2var_epi64(tmp1, bits.perm.permute2, tmp2));
        tmp1 = _mm512_and_si512(hbits, mh);
        tmp2 = _mm512_and_si512(_mm512_srli_epi16(hbits, 2), mh);
        bits.values[1] = _mm512_or_si512(bits.values[1], _mm512_permutex2var_epi64(tmp1, bits.perm.permute1, tmp2));
        bits.values[3] = _mm512_or_si512(bits.values[3], _mm512_permutex2var_epi64(tmp1, bits.perm.permute2, tmp2));
    }

    Q4Bits bits;
    HighBit3 hbits;
    Scale16 sc16;

    const __m512i mh = _mm512_set1_epi8(0x30);

};

struct DequantizerIQ4XS final : public BaseDequantizer<block_iq4_xs> {
    DequantizerIQ4XS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_iq4nl_values_512()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accd, __m512i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        prepare(x[i].qs);
        auto scales128 = siq4.make_scales(*(const uint32_t *)x[i].scales_l, x[i].scales_h);
        s8k.accum_mins(scales128, q8, i, -128.f*d, accd);
        auto scales256 = MM256_SET_M128I(scales128, scales128);
        auto all_scales = _mm512_inserti32x8(_mm512_castsi256_si512(scales256), scales256, 1);
        scales[0] = _mm512_shuffle_epi8(all_scales, shuffles[0]);
        scales[1] = _mm512_shuffle_epi8(all_scales, shuffles[1]);
        scales[2] = _mm512_shuffle_epi8(all_scales, shuffles[2]);
        scales[3] = _mm512_shuffle_epi8(all_scales, shuffles[3]);
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
    ScaleIQ4XS siq4;
    const __m512i values;
    const __m512i permute1 = _mm512_set_epi64(11, 10, 3, 2,  9,  8, 1, 0);
    const __m512i permute2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    const __m512i shuffles[4] = {
        _mm512_inserti32x8(_mm512_set1_epi16(0x0100), _mm256_set1_epi16(0x0302), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0504), _mm256_set1_epi16(0x0706), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0908), _mm256_set1_epi16(0x0b0a), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0d0c), _mm256_set1_epi16(0x0f0e), 1),
    };
};

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

#else
//====================================== AVX2 ==================================================

struct HighBit5 {
    inline void load(const uint8_t * h) { hbits = _mm256_loadu_si256((const __m256i *)h); }
    inline void apply(Q4Bits& bits, bool do_shift) {
        bits.values[0] = _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(hbits, 4), mh));
        bits.values[1] = _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(hbits, 3), mh));
        bits.values[2] = _mm256_or_si256(bits.values[2], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh));
        bits.values[3] = _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_slli_epi16(hbits, 1), mh));
        if (do_shift) {
            hbits = _mm256_srli_epi16(hbits, 4);
        }
    }
    const __m256i mh = _mm256_set1_epi8(0x10);
    __m256i hbits;
};

struct HighBit3 {
    inline void load(const uint8_t * h) { hbits = _mm256_loadu_si256((const __m256i *)h); }
    inline void apply(Q2Bits& bits, bool do_shift) {
        bits.values[0] = _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh));
        bits.values[1] = _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(hbits, 1), mh));
        bits.values[2] = _mm256_or_si256(bits.values[2], _mm256_and_si256(hbits, mh));
        bits.values[3] = _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_srli_epi16(hbits, 1), mh));
        if (do_shift) {
            hbits = _mm256_srli_epi16(hbits, 4);
        }
    }
    const __m256i mh = _mm256_set1_epi8(0x04);
    __m256i hbits;
};

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

struct DequantizerQ2K final : public BaseDequantizer<block_q2_K> {
    DequantizerQ2K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}

    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        process_mins_16(_mm256_cvtepi8_epi16(mins8), q8, i, -GGML_FP16_TO_FP32(x[i].dmin), accm);
        prepare_scales_16(_mm256_cvtepi8_epi16(scales8), scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
    }

    Q2Bits  bits;

    const __m128i m4 = _mm_set1_epi8(0xf);
};

struct DequantizerQ3K final : public BaseDequantizer<block_q3_K> {
    DequantizerQ3K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}

    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        hbits.load(x[i].hmask);
        process_mins_and_scales_16(sc3.make_scales((const uint16_t *)x[i].scales), q8, i, -4.f*d, accm, scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        hbits.apply(bits, j == 0);
    }

    Q2Bits  bits;
    HighBit3 hbits;
    ScaleQ3 sc3;

    const __m128i m32 = _mm_set1_epi8(-32);
};

struct DequantizerQ4K final : public BaseDequantizer<block_q4_K> {
    DequantizerQ4K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
        d = GGML_FP16_TO_FP32(x[i].d);
        return s8k.process_mins_and_scales(x[i].scales, -GGML_FP16_TO_FP32(x[i].dmin), i, q8, accd);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
    }

    Q4Bits bits;
    Scales8K s8k;
};

struct DequantizerQ5K final : public BaseDequantizer<block_q5_K> {
    DequantizerQ5K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
        d = GGML_FP16_TO_FP32(x[i].d);
        hbits.load(x[i].qh);
        return s8k.process_mins_and_scales(x[i].scales, -GGML_FP16_TO_FP32(x[i].dmin), i, q8, accd);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        hbits.apply(bits, j == 0);
    }

    Q4Bits  bits;
    HighBit5 hbits;
    Scales8K s8k;
};

struct DequantizerQ6K final : public BaseDequantizer<block_q6_K> {
    DequantizerQ6K(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        process_mins_and_scales_16(_mm_loadu_si128((const __m128i *)x[i].scales), q8, i, -32.f*d, accm, scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare64(x[i].ql, j);
        auto hbits = _mm256_loadu_si256((const __m256i *)x[i].qh + j);
        bits.values[0] = _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(hbits, 4), mh));
        bits.values[1] = _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh));
        bits.values[2] = _mm256_or_si256(bits.values[2], _mm256_and_si256(hbits, mh));
        bits.values[3] = _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_srli_epi16(hbits, 2), mh));
    }

    Q4Bits  bits;
    const __m256i mh = _mm256_set1_epi8(0x30);
};

struct DequantizerIQ4XS final : public BaseDequantizer<block_iq4_xs> {
    DequantizerIQ4XS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_iq4nl_values_256()) {}
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
        d = GGML_FP16_TO_FP32(x[i].d);
        auto scales128 = siq4.make_scales(*(const uint32_t *)x[i].scales_l, x[i].scales_h);
        s8k.accum_mins(scales128, q8, i, -128.f*d, accd);
        return MM256_SET_M128I(scales128, scales128);
    }
    inline void prepare(int i, int j) {
        bits.prepare16(x[i].qs, j);
        bits.values[0] = _mm256_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values, bits.values[3]);
    }

    Q4Bits bits;
    Scales8K s8k;
    ScaleIQ4XS siq4;
    const __m256i values;
};

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

                multiply_add(deq.bits, scales, j, i, q8, sumi);

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
                multiply_add(deq.bits, scales, j, i, q8, sumi);
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

#endif

template <int nrc_y>
static void mul_mat_iq4_xs_r8_q8_k_avx2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto m30 = _mm256_set1_epi8(0x30);
    auto m32 = _mm256_set1_epi8(32);
#ifndef HAVE_FANCY_SIMD
    auto s_shuffle = _mm256_set_epi64x(0x0f0e0f0e0d0c0d0c, 0x0b0a0b0a09080908, 0x0706070605040504, 0x0302030201000100);
    auto values128 = _mm_loadu_si128((const __m128i *)iq4k_values);
    auto values = MM256_SET_M128I(values128, values128);
#else
    auto values = load_iq4nl_values_256();
#endif
    int nbl = n / QK_K;
    using helper_t = union { __m256i vec[2]; uint64_t val[8]; };
    helper_t h;
    __m256  acc[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_iq4_xs_r8 * iq4 = (const block_iq4_xs_r8 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto d4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[ibl].d));
            auto slbits = _mm256_loadu_si256((const __m256i *)iq4[ibl].scales_l);
            auto sl1 = _mm256_and_si256(slbits, m4);
            auto sl2 = _mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4);
            auto shbits = _mm_loadu_si128((const __m128i*)iq4[ibl].scales_h);
            auto sh = MM256_SET_M128I(_mm_srli_epi16(shbits, 2), shbits);
            h.vec[0] = _mm256_sub_epi8(_mm256_or_si256(sl1, _mm256_and_si256(_mm256_slli_epi16(sh, 4), m30)), m32);
            h.vec[1] = _mm256_sub_epi8(_mm256_or_si256(sl2, _mm256_and_si256(sh, m30)), m32);
            __m256i isum[nrc_y] = {};
            for (int ib = 0; ib < QK_K/32; ++ib) {
#ifdef HAVE_FANCY_SIMD
                auto iscales = _mm256_cvtepi8_epi32(_mm_set1_epi64x(h.val[ib]));
                auto scales  = _mm256_mul_ps(d4, _mm256_cvtepi32_ps(iscales));
                auto scales_m = _mm256_mul_ps(scales, _mm256_set1_ps(-128.f));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    float m8 = ((const float *)q8.y[iy][ibl].bsums)[ib];
                    acc[iy] = _mm256_fmadd_ps(scales_m, _mm256_set1_ps(m8), acc[iy]);
                }
#else
                auto iscales = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_set1_epi64x(h.val[ib])), s_shuffle);
#endif
                auto bits1 = _mm256_loadu_si256((const __m256i *)iq4[ibl].qs+4*ib+0);
                auto bits2 = _mm256_loadu_si256((const __m256i *)iq4[ibl].qs+4*ib+1);
                qx[0] = _mm256_shuffle_epi8(values, _mm256_and_si256(m4, bits1));
                qx[1] = _mm256_shuffle_epi8(values, _mm256_and_si256(m4, _mm256_srli_epi16(bits1, 4)));
                qx[2] = _mm256_shuffle_epi8(values, _mm256_and_si256(m4, bits2));
                qx[3] = _mm256_shuffle_epi8(values, _mm256_and_si256(m4, _mm256_srli_epi16(bits2, 4)));
#ifndef HAVE_FANCY_SIMD
                auto s1 = _mm256_sign_epi8(qx[0], qx[0]);
                auto s2 = _mm256_sign_epi8(qx[1], qx[1]);
                auto s3 = _mm256_sign_epi8(qx[2], qx[2]);
                auto s4 = _mm256_sign_epi8(qx[3], qx[3]);
#endif
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y128 = _mm_loadu_si128((const __m128i*)q8.y[iy][ibl].qs+2*ib+0);
                    auto y = MM256_SET_M128I(y128, y128);
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
                    auto sumi  = _mm256_add_epi32(_mm256_add_epi32(_mm256_madd_epi16(iscales, sumi1), _mm256_madd_epi16(iscales, sumi2)),
                                                  _mm256_add_epi32(_mm256_madd_epi16(iscales, sumi3), _mm256_madd_epi16(iscales, sumi4)));
                    isum[iy] = _mm256_add_epi32(isum[iy], sumi);
#endif
                }
                bits1 = _mm256_loadu_si256((const __m256i *)iq4[ibl].qs+4*ib+2);
                bits2 = _mm256_loadu_si256((const __m256i *)iq4[ibl].qs+4*ib+3);
                qx[0] = _mm256_shuffle_epi8(values, _mm256_and_si256(m4, bits1));
                qx[1] = _mm256_shuffle_epi8(values, _mm256_and_si256(m4, _mm256_srli_epi16(bits1, 4)));
                qx[2] = _mm256_shuffle_epi8(values, _mm256_and_si256(m4, bits2));
                qx[3] = _mm256_shuffle_epi8(values, _mm256_and_si256(m4, _mm256_srli_epi16(bits2, 4)));
#ifndef HAVE_FANCY_SIMD
                s1 = _mm256_sign_epi8(qx[0], qx[0]);
                s2 = _mm256_sign_epi8(qx[1], qx[1]);
                s3 = _mm256_sign_epi8(qx[2], qx[2]);
                s4 = _mm256_sign_epi8(qx[3], qx[3]);
#endif
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y128 = _mm_loadu_si128((const __m128i*)q8.y[iy][ibl].qs+2*ib+1);
                    auto y = MM256_SET_M128I(y128, y128);
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
                    auto sumi  = _mm256_add_epi32(_mm256_add_epi32(_mm256_madd_epi16(iscales, sumi1), _mm256_madd_epi16(iscales, sumi2)),
                                                  _mm256_add_epi32(_mm256_madd_epi16(iscales, sumi3), _mm256_madd_epi16(iscales, sumi4)));
                    isum[iy] = _mm256_add_epi32(isum[iy], sumi);
#endif
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_iq4_xs_r8_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    mul_mat_iq4_xs_r8_q8_k_avx2<nrc_y>(n, vx, bx, info, nrc_x);
    return;
    if constexpr (nrc_y == 1){
        mul_mat_iq4_xs_r8_q8_k_avx2<1>(n, vx, bx, info, nrc_x);
    } else {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm512_set1_epi8(0xf);
    auto values = load_iq4nl_values_512();
    int nbl = n / QK_K;
    using helper_t = union { __m512i vec; uint32_t val[16]; };
    helper_t h;
    __m512  acc[nrc_y] = {};
    __m512i isum[nrc_y] = {};
    __m512i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_iq4_xs_r8 * iq4l = (const block_iq4_xs_r8 *)((const char *)vx + (ix+0)*bx);
        const block_iq4_xs_r8 * iq4h = (const block_iq4_xs_r8 *)((const char *)vx + (ix+4)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq4l[ibl].d));
            auto dh = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq4h[ibl].d));
            auto d4 = _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set_m128(dl, dl)), _mm256_set_m128(dh, dh), 1);
            auto d4x64 = _mm512_mul_ps(d4, _mm512_set1_ps(-64.f));
            auto slbits_l = _mm_loadu_si128((const __m128i *)iq4l[ibl].scales_l);
            auto shbits_l = _mm_loadu_si128((const __m128i *)iq4h[ibl].scales_l);
            auto sl_l = MM256_SET_M128I(_mm_srli_epi16(slbits_l, 4), slbits_l);
            auto sh_l = MM256_SET_M128I(_mm_srli_epi16(shbits_l, 4), shbits_l);
            auto slb = _mm512_and_si512(_mm512_inserti32x8(_mm512_castsi256_si512(sl_l), sh_l, 1), m4);
            auto aux64 = (const uint64_t *)iq4l[ibl].scales_h;
            auto slbits_h = _mm_set_epi64x(aux64[0] >> 2, aux64[0]);
            aux64 = (const uint64_t *)iq4h[ibl].scales_h;
            auto shbits_h = _mm_set_epi64x(aux64[0] >> 2, aux64[0]);
            auto sl_h = MM256_SET_M128I(slbits_h, _mm_slli_epi16(slbits_h, 4));
            auto sh_h = MM256_SET_M128I(shbits_h, _mm_slli_epi16(shbits_h, 4));
            auto shb = _mm512_and_si512(_mm512_inserti32x8(_mm512_castsi256_si512(sl_h), sh_h, 1), _mm512_set1_epi8(0x30));
            h.vec = _mm512_sub_epi8(_mm512_or_si512(slb, shb), _mm512_set1_epi8(32));
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto iscales = _mm512_cvtepi8_epi32(_mm_blend_epi32(_mm_set1_epi32(h.val[ib+0]), _mm_set1_epi32(h.val[ib+8]), 0x0c));
                auto scales  = _mm512_cvtepi32_ps(iscales);
                auto scales_m = _mm512_mul_ps(scales, d4x64);
                auto bits1 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq4l[ibl].qs+2*ib+0)),
                                                                       _mm256_loadu_si256((const __m256i *)iq4h[ibl].qs+2*ib+0), 1);
                auto bits2 = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)iq4l[ibl].qs+2*ib+1)),
                                                                       _mm256_loadu_si256((const __m256i *)iq4h[ibl].qs+2*ib+1), 1);
                qx[0] = _mm512_shuffle_epi8(values, _mm512_and_si512(bits1, m4));
                qx[1] = _mm512_shuffle_epi8(values, _mm512_and_si512(bits2, m4));
                qx[2] = _mm512_shuffle_epi8(values, _mm512_and_si512(_mm512_srli_epi16(bits1, 4), m4));
                qx[3] = _mm512_shuffle_epi8(values, _mm512_and_si512(_mm512_srli_epi16(bits2, 4), m4));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y8 = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
                    auto y = _mm512_inserti32x8(_mm512_castsi256_si512(y8), y8, 1);
                    auto sumi = _mm512_setzero_si512();
                    sumi = _mm512_dpbusd_epi32(sumi, qx[0], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
                    sumi = _mm512_dpbusd_epi32(sumi, qx[1], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
                    sumi = _mm512_dpbusd_epi32(sumi, qx[2], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
                    sumi = _mm512_dpbusd_epi32(sumi, qx[3], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
                    isum[iy] = _mm512_add_epi32(isum[iy], _mm512_mullo_epi32(iscales, sumi));
                    float m8 = ((const float *)q8.y[iy][ibl].bsums)[ib];
                    acc[iy] = _mm512_fmadd_ps(scales_m, _mm512_set1_ps(m8), acc[iy]);
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm512_fmadd_ps(_mm512_mul_ps(d4, _mm512_set1_ps(q8.scale(iy, ibl))), _mm512_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm512_setzero_si512();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum1 = _mm_add_ps(_mm512_extractf32x4_ps(acc[iy], 0), _mm512_extractf32x4_ps(acc[iy], 1));
            auto sum2 = _mm_add_ps(_mm512_extractf32x4_ps(acc[iy], 2), _mm512_extractf32x4_ps(acc[iy], 3));
            info.store(ix+0, iy, sum1);
            info.store(ix+4, iy, sum2);
            acc[iy] = _mm512_setzero_ps();
        }
    }
    }
}
#else
template <int nrc_y>
static void mul_mat_iq4_xs_r8_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    mul_mat_iq4_xs_r8_q8_k_avx2<nrc_y>(n, vx, bx, info, nrc_x);
}
#endif

template <int nrc_y>
static void mul_mat_q2_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto mxf = _mm256_set1_epi8(0xf);
    auto m03 = _mm256_set1_epi8(0x03);
    static const uint8_t k_shuff[32] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    auto shuff = _mm256_loadu_si256((const __m256i *)k_shuff);
#ifdef HAVE_FANCY_SIMD
    __m256i isum[nrc_y] = {};
#else
    auto m1 = _mm256_set1_epi16(1);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y] = {};
    __m256i qx[4];
    int8_t scales[64];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q2_k_r4 * iq2 = (const block_q2_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dm = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq2[ibl].d));
            auto d4 = _mm256_set_m128(_mm256_castps256_ps128(dm), _mm256_castps256_ps128(dm));
            auto m4 = _mm256_set_m128(_mm256_extractf128_ps(dm, 1), _mm256_extractf128_ps(dm, 1));
            m4 = _mm256_mul_ps(m4, _mm256_set1_ps(-1.f));
            auto all_scales1 = _mm256_loadu_si256((const __m256i *)iq2[ibl].scales+0);
            auto all_scales2 = _mm256_loadu_si256((const __m256i *)iq2[ibl].scales+1);
            auto scales1 = _mm256_and_si256(_mm256_srli_epi16(all_scales1, 4), mxf);
            auto scales2 = _mm256_and_si256(_mm256_srli_epi16(all_scales2, 4), mxf);
            {
                auto t1 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales1, 0)), shuff); // blocks  0,  1,  2,  3 for each row
                auto t2 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales1, 1)), shuff); // blocks  4,  5,  6,  7 for each row
                auto t3 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales2, 0)), shuff); // blocks  8,  9, 10, 11 for each row
                auto t4 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales2, 1)), shuff); // blocks 12, 13, 14, 15 for each row
                auto s1 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 0), _mm256_extracti128_si256(t1, 0)); // blocks 0, 1,  8, 9
                auto s2 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 1), _mm256_extracti128_si256(t1, 1)); // blocks 2, 3, 10, 11
                auto s3 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 0), _mm256_extracti128_si256(t2, 0)); // blocks 4, 5, 12, 13
                auto s4 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 1), _mm256_extracti128_si256(t2, 1)); // blocks 6, 7, 14, 15
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto bsums = q8.load_bsums(iy, ibl);
                    auto sumi = _mm256_setzero_si256();
#ifdef HAVE_FANCY_SIMD
                    sumi = _mm256_dpwssd_epi32(sumi, s1, _mm256_shuffle_epi32(bsums, 0x00));
                    sumi = _mm256_dpwssd_epi32(sumi, s2, _mm256_shuffle_epi32(bsums, 0x55));
                    sumi = _mm256_dpwssd_epi32(sumi, s3, _mm256_shuffle_epi32(bsums, 0xaa));
                    sumi = _mm256_dpwssd_epi32(sumi, s4, _mm256_shuffle_epi32(bsums, 0xff));
                    auto d8 = _mm256_set1_ps(q8.scale(iy, ibl));
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(m4, d8), _mm256_cvtepi32_ps(sumi), acc[iy]);
#else
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s1, _mm256_shuffle_epi32(bsums, 0x00)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s2, _mm256_shuffle_epi32(bsums, 0x55)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s3, _mm256_shuffle_epi32(bsums, 0xaa)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s4, _mm256_shuffle_epi32(bsums, 0xff)));
                    auto d8 = _mm256_set1_ps(q8.scale(iy, ibl));
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(m4, d8), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    if constexpr (nrc_y == 1) {
                        d4 = _mm256_mul_ps(d4, d8);
                    }
#endif
                }
            }
            all_scales1 = _mm256_and_si256(all_scales1, mxf);
            all_scales2 = _mm256_and_si256(all_scales2, mxf);
            _mm256_storeu_si256((__m256i *)scales+0, all_scales1);
            _mm256_storeu_si256((__m256i *)scales+1, all_scales2);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto iscales = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(scales + 8*ib)));
#ifndef HAVE_FANCY_SIMD
                auto scales  = _mm256_mul_ps(d4, _mm256_cvtepi32_ps(iscales));
#endif
                auto lb = _mm256_loadu_si256((const __m256i *)iq2[ibl].qs+ib);
                qx[0] = _mm256_and_si256(lb, m03);
                qx[1] = _mm256_and_si256(_mm256_srli_epi16(lb, 2), m03);
                qx[2] = _mm256_and_si256(_mm256_srli_epi16(lb, 4), m03);
                qx[3] = _mm256_and_si256(_mm256_srli_epi16(lb, 6), m03);
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
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                  _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                  _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    // Quants are in 0...3, so we can add add up all of them as int16_t without overflowing
                    auto sumi = _mm256_madd_epi16(m1, _mm256_add_epi16(sumi1, sumi2));
                    if constexpr (nrc_y == 1) {
                        acc[iy] = _mm256_fmadd_ps(scales, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    } else {
                        acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(scales, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    }
#endif
                }
            }
#ifdef HAVE_FANCY_SIMD
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl)));
                acc[iy] = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
            }
#endif
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

template <int nrc_y>
static void mul_mat_q3_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto m30 = _mm256_set1_epi8(0x30);
    auto m32 = _mm256_set1_epi8(32);
    auto m03 = _mm256_set1_epi8(0x03);
    auto m04 = _mm256_set1_epi8(0x04);
    static const uint8_t k_shuff[32] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    auto shuff = _mm256_loadu_si256((const __m256i *)k_shuff);
#ifdef HAVE_FANCY_SIMD
    __m256i isum[nrc_y];
#else
    auto m1 = _mm256_set1_epi16(1);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y] = {};
    __m256i qx[4];
    int8_t scales[64];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q3_k_r4 * iq3 = (const block_q3_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq3[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
#ifndef HAVE_FANCY_SIMD
            if constexpr (nrc_y == 1) {
                d4 = _mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(0, ibl)));
            }
#endif
            auto slb = _mm256_loadu_si256((const __m256i *)iq3[ibl].scales_l);
            auto shbits = _mm_loadu_si128((const __m128i *)iq3[ibl].scales_h);
            auto shb = MM256_SET_M128I(_mm_srli_epi16(shbits, 2), shbits);
            auto scales1 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(slb, m4), _mm256_and_si256(_mm256_slli_epi16(shb, 4), m30)), m32);
            auto scales2 = _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(slb, 4), m4), _mm256_and_si256(shb, m30)), m32);
            _mm256_storeu_si256((__m256i *)scales+0, scales1);
            _mm256_storeu_si256((__m256i *)scales+1, scales2);
            {
#ifndef HAVE_FANCY_SIMD
                auto min = _mm256_mul_ps(d4, _mm256_set1_ps(-4.f));
#endif
                auto t1 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales1, 0)), shuff); // blocks  0,  1,  2,  3 for each row
                auto t2 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales1, 1)), shuff); // blocks  4,  5,  6,  7 for each row
                auto t3 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales2, 0)), shuff); // blocks  8,  9, 10, 11 for each row
                auto t4 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(scales2, 1)), shuff); // blocks 12, 13, 14, 15 for each row
                auto s1 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 0), _mm256_extracti128_si256(t1, 0)); // blocks 0, 1,  8, 9
                auto s2 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 1), _mm256_extracti128_si256(t1, 1)); // blocks 2, 3, 10, 11
                auto s3 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 0), _mm256_extracti128_si256(t2, 0)); // blocks 4, 5, 12, 13
                auto s4 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 1), _mm256_extracti128_si256(t2, 1)); // blocks 6, 7, 14, 15
#ifdef HAVE_FANCY_SIMD
                s1 = _mm256_mullo_epi16(s1, _mm256_set1_epi16(-4));
                s2 = _mm256_mullo_epi16(s2, _mm256_set1_epi16(-4));
                s3 = _mm256_mullo_epi16(s3, _mm256_set1_epi16(-4));
                s4 = _mm256_mullo_epi16(s4, _mm256_set1_epi16(-4));
#endif
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto bsums = q8.load_bsums(iy, ibl);
                    auto sumi = _mm256_setzero_si256();
#ifdef HAVE_FANCY_SIMD
                    sumi = _mm256_dpwssd_epi32(sumi, s1, _mm256_shuffle_epi32(bsums, 0x00));
                    sumi = _mm256_dpwssd_epi32(sumi, s2, _mm256_shuffle_epi32(bsums, 0x55));
                    sumi = _mm256_dpwssd_epi32(sumi, s3, _mm256_shuffle_epi32(bsums, 0xaa));
                    sumi = _mm256_dpwssd_epi32(sumi, s4, _mm256_shuffle_epi32(bsums, 0xff));
                    isum[iy] = sumi;
#else
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s1, _mm256_shuffle_epi32(bsums, 0x00)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s2, _mm256_shuffle_epi32(bsums, 0x55)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s3, _mm256_shuffle_epi32(bsums, 0xaa)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s4, _mm256_shuffle_epi32(bsums, 0xff)));
                    if constexpr (nrc_y == 1) {
                        acc[iy] = _mm256_fmadd_ps(min, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    } else {
                        acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(min, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    }
#endif
                }
            }
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto iscales = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(scales + 8*ib)));
#ifndef HAVE_FANCY_SIMD
                auto scales  = _mm256_mul_ps(d4, _mm256_cvtepi32_ps(iscales));
#endif
                auto lb = _mm256_loadu_si256((const __m256i *)iq3[ibl].qs+ib);
                auto hbits = _mm_loadu_si128((const __m128i *)iq3[ibl].qh+ib);
                auto hb = MM256_SET_M128I(hbits, _mm_slli_epi16(hbits, 4));
                qx[0] = _mm256_or_si256(_mm256_and_si256(lb, m03),                       _mm256_and_si256(m04, _mm256_srli_epi16(hb, 2)));
                qx[1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 2), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 3)));
                qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 4), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 4)));
                qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 6), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 5)));
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
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                  _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                  _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    // Quants are in 0...8, so we can add add up all of them as int16_t without overflowing
                    auto sumi = _mm256_madd_epi16(m1, _mm256_add_epi16(sumi1, sumi2));
                    if constexpr (nrc_y == 1) {
                        acc[iy] = _mm256_fmadd_ps(scales, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    } else {
                        acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(scales, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    }
#endif

                }
            }
#ifdef HAVE_FANCY_SIMD
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl)));
                acc[iy] = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
            }
#endif
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

template <int nrc_y>
inline void process_min_r4_b32(int ibl, __m256 m4, __m256i mins, const Q8<nrc_y, block_q8_K>& q8, __m256 * acc) {
    auto mins_l = _mm256_castsi256_si128(mins);
    auto mins_h = _mm256_extracti128_si256(mins, 1);
    auto aux1   = _mm_unpacklo_epi32(mins_l, mins_h);
    auto aux2   = _mm_unpackhi_epi32(mins_l, mins_h);
    auto ic1 = _mm256_cvtepi8_epi32(aux1);
    auto ic2 = _mm256_cvtepi8_epi32(_mm_shuffle_epi32(aux1, 0xee));
    auto ic3 = _mm256_cvtepi8_epi32(aux2);
    auto ic4 = _mm256_cvtepi8_epi32(_mm_shuffle_epi32(aux2, 0xee));
    if constexpr (nrc_y == 1) {
        auto bs = _mm256_loadu_ps((const float *)q8.y[0][ibl].bsums);
        auto sumf = _mm256_mul_ps(_mm256_cvtepi32_ps(ic1), _mm256_shuffle_ps(bs, bs, 0x00));
        sumf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ic2), _mm256_shuffle_ps(bs, bs, 0x55), sumf);
        sumf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ic3), _mm256_shuffle_ps(bs, bs, 0xaa), sumf);
        sumf = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ic4), _mm256_shuffle_ps(bs, bs, 0xff), sumf);
        acc[0] = _mm256_fmadd_ps(m4, sumf, acc[0]);
    } else {
        auto c1 = _mm256_mul_ps(m4, _mm256_cvtepi32_ps(ic1));
        auto c2 = _mm256_mul_ps(m4, _mm256_cvtepi32_ps(ic2));
        auto c3 = _mm256_mul_ps(m4, _mm256_cvtepi32_ps(ic3));
        auto c4 = _mm256_mul_ps(m4, _mm256_cvtepi32_ps(ic4));
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto bs = _mm256_loadu_ps((const float *)q8.y[iy][ibl].bsums);
            acc[iy] = _mm256_fmadd_ps(c1, _mm256_shuffle_ps(bs, bs, 0x00), acc[iy]);
            acc[iy] = _mm256_fmadd_ps(c2, _mm256_shuffle_ps(bs, bs, 0x55), acc[iy]);
            acc[iy] = _mm256_fmadd_ps(c3, _mm256_shuffle_ps(bs, bs, 0xaa), acc[iy]);
            acc[iy] = _mm256_fmadd_ps(c4, _mm256_shuffle_ps(bs, bs, 0xff), acc[iy]);
        }
    }
}

template <int nrc_y>
static void mul_mat_q4_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto mf = _mm256_set1_epi8(0xf);
    auto m3 = _mm256_set1_epi8(0x30);
    int nbl = n / QK_K;
    union { __m256i vec; uint32_t val[8]; } hd;
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q4_k_r4 * iq4 = (const block_q4_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq4[ibl].d));
            auto d4 = _mm256_set_m128(_mm256_castps256_ps128(dl), _mm256_castps256_ps128(dl));
            auto m4 = _mm256_mul_ps(_mm256_set1_ps(-1.0f), _mm256_set_m128(_mm256_extractf128_ps(dl, 1), _mm256_extractf128_ps(dl, 1)));
            auto lbits = _mm256_loadu_si256((const __m256i *)iq4[ibl].scales_l);
            auto hbits128 = _mm_loadu_si128((const __m128i *)iq4[ibl].scales_h);
            auto hbits = MM256_SET_M128I(hbits128, _mm_slli_epi16(hbits128, 4));
            hd.vec = _mm256_or_si256(_mm256_and_si256(lbits, mf), _mm256_and_si256(hbits, m3));
            auto mins = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits, 4), mf), _mm256_and_si256(_mm256_srli_epi16(hbits, 2), m3));
            process_min_r4_b32(ibl, m4, mins, q8, acc);
            for (int ib = 0; ib < QK_K/32; ++ib) {
#ifdef HAVE_FANCY_SIMD
                auto scales_d = _mm256_cvtepi8_epi32(_mm_set1_epi32(hd.val[ib]));
#else
                auto aux = _mm_set1_epi32(hd.val[ib]);
                aux = _mm_cvtepu8_epi16(_mm_unpacklo_epi8(aux, aux));
                auto scales_d = MM256_SET_M128I(aux, aux);
#endif
                auto bits1 = _mm256_loadu_si256((const __m256i *)iq4[ibl].qs+2*ib+0);
                auto bits2 = _mm256_loadu_si256((const __m256i *)iq4[ibl].qs+2*ib+1);
                qx[0] = _mm256_and_si256(bits1, mf);
                qx[1] = _mm256_and_si256(bits2, mf);
                qx[2] = _mm256_and_si256(_mm256_srli_epi16(bits1, 4), mf);
                qx[3] = _mm256_and_si256(_mm256_srli_epi16(bits2, 4), mf);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales_d, sumi));
#else
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                  _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                  _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(scales_d, _mm256_add_epi16(sumi1, sumi2)));
#endif
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
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
static void mul_mat_q5_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto mf = _mm256_set1_epi8(0xf);
    auto m10 = _mm256_set1_epi8(0x10);
    auto m30 = _mm256_set1_epi8(0x30);
    int nbl = n / QK_K;
    union { __m256i vec; uint32_t val[8]; } hd;
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q5_k_r4 * iq5 = (const block_q5_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq5[ibl].d));
            auto d4 = _mm256_set_m128(_mm256_castps256_ps128(dl), _mm256_castps256_ps128(dl));
            auto m4 = _mm256_mul_ps(_mm256_set1_ps(-1.0f), _mm256_set_m128(_mm256_extractf128_ps(dl, 1), _mm256_extractf128_ps(dl, 1)));
            auto lbits = _mm256_loadu_si256((const __m256i *)iq5[ibl].scales_l);
            auto hbits128 = _mm_loadu_si128((const __m128i *)iq5[ibl].scales_h);
            auto hbits = MM256_SET_M128I(hbits128, _mm_slli_epi16(hbits128, 4));
            hd.vec = _mm256_or_si256(_mm256_and_si256(lbits, mf), _mm256_and_si256(hbits, m30));
            auto mins = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits, 4), mf), _mm256_and_si256(_mm256_srli_epi16(hbits, 2), m30));
            process_min_r4_b32(ibl, m4, mins, q8, acc);
            for (int ib = 0; ib < QK_K/32; ++ib) {
#ifdef HAVE_FANCY_SIMD
                auto scales_d = _mm256_cvtepi8_epi32(_mm_set1_epi32(hd.val[ib]));
#else
                auto aux = _mm_set1_epi32(hd.val[ib]);
                aux = _mm_cvtepu8_epi16(_mm_unpacklo_epi8(aux, aux));
                auto scales_d = MM256_SET_M128I(aux, aux);
#endif
                auto lbits1 = _mm256_loadu_si256((const __m256i *)iq5[ibl].qs+2*ib+0);
                auto lbits2 = _mm256_loadu_si256((const __m256i *)iq5[ibl].qs+2*ib+1);
                auto hbits128 = _mm_loadu_si128((const __m128i*)iq5[ibl].qh + ib);
                auto hbits = MM256_SET_M128I(hbits128, _mm_slli_epi16(hbits128, 4));
                qx[0] = _mm256_or_si256(_mm256_and_si256(lbits1, mf), _mm256_and_si256(m10, hbits));
                qx[1] = _mm256_or_si256(_mm256_and_si256(lbits2, mf), _mm256_and_si256(m10, _mm256_srli_epi16(hbits, 2)));
                qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits1, 4), mf), _mm256_and_si256(m10, _mm256_srli_epi16(hbits, 1)));
                qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits2, 4), mf), _mm256_and_si256(m10, _mm256_srli_epi16(hbits, 3)));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales_d, sumi));
#else
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                  _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                  _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    // To avoid overflow, we can only add up to 4 q5 x q8 products.
                    auto sumi = _mm256_add_epi32(_mm256_madd_epi16(scales_d, sumi1), _mm256_madd_epi16(scales_d, sumi2));
                    isum[iy] = _mm256_add_epi32(isum[iy], sumi);
#endif
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
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
static void mul_mat_q6_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto m3 = _mm256_set1_epi8(0x30);
    static const uint8_t k_shuff[32] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    auto shuff = _mm256_loadu_si256((const __m256i *)k_shuff);
#ifdef HAVE_FANCY_SIMD
    __m256i isum[nrc_y];
#else
    auto m1 = _mm256_set1_epi16(1);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q6_k_r4 * iq6 = (const block_q6_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq6[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
#ifndef HAVE_FANCY_SIMD
            if constexpr (nrc_y == 1) {
                d4 = _mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(0, ibl)));
            }
#endif
            {
#ifndef HAVE_FANCY_SIMD
                auto min = _mm256_mul_ps(d4, _mm256_set1_ps(-32.f));
#endif
                auto t1 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)iq6[ibl].scales+0)), shuff); // blocks  0,  1,  2,  3 for each row
                auto t2 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)iq6[ibl].scales+1)), shuff); // blocks  4,  5,  6,  7 for each row
                auto t3 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)iq6[ibl].scales+2)), shuff); // blocks  8,  9, 10, 11 for each row
                auto t4 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)iq6[ibl].scales+3)), shuff); // blocks 12, 13, 14, 15 for each row
                auto s1 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 0), _mm256_extracti128_si256(t1, 0)); // blocks 0, 1,  8, 9
                auto s2 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 1), _mm256_extracti128_si256(t1, 1)); // blocks 2, 3, 10, 11
                auto s3 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 0), _mm256_extracti128_si256(t2, 0)); // blocks 4, 5, 12, 13
                auto s4 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 1), _mm256_extracti128_si256(t2, 1)); // blocks 6, 7, 14, 15
#ifdef HAVE_FANCY_SIMD
                s1 = _mm256_mullo_epi16(s1, _mm256_set1_epi16(-32));
                s2 = _mm256_mullo_epi16(s2, _mm256_set1_epi16(-32));
                s3 = _mm256_mullo_epi16(s3, _mm256_set1_epi16(-32));
                s4 = _mm256_mullo_epi16(s4, _mm256_set1_epi16(-32));
#endif
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto bsums = q8.load_bsums(iy, ibl);
                    auto sumi = _mm256_setzero_si256();
#ifdef HAVE_FANCY_SIMD
                    sumi = _mm256_dpwssd_epi32(sumi, s1, _mm256_shuffle_epi32(bsums, 0x00));
                    sumi = _mm256_dpwssd_epi32(sumi, s2, _mm256_shuffle_epi32(bsums, 0x55));
                    sumi = _mm256_dpwssd_epi32(sumi, s3, _mm256_shuffle_epi32(bsums, 0xaa));
                    sumi = _mm256_dpwssd_epi32(sumi, s4, _mm256_shuffle_epi32(bsums, 0xff));
                    isum[iy] = sumi;
#else
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s1, _mm256_shuffle_epi32(bsums, 0x00)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s2, _mm256_shuffle_epi32(bsums, 0x55)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s3, _mm256_shuffle_epi32(bsums, 0xaa)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s4, _mm256_shuffle_epi32(bsums, 0xff)));
                    if constexpr (nrc_y == 1) {
                        acc[iy] = _mm256_fmadd_ps(min, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    } else {
                        acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(min, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    }
#endif
                }
            }
            const uint32_t * scales = (const uint32_t *)iq6[ibl].scales;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto iscales = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(scales + 2*ib)));
#ifndef HAVE_FANCY_SIMD
                auto scales  = _mm256_mul_ps(d4, _mm256_cvtepi32_ps(iscales));
#endif
                auto lbits1 = _mm256_loadu_si256((const __m256i *)iq6[ibl].ql+2*ib+0);
                auto lbits2 = _mm256_loadu_si256((const __m256i *)iq6[ibl].ql+2*ib+1);
                auto hbits  = _mm256_loadu_si256((const __m256i *)iq6[ibl].qh+ib);
                qx[0] = _mm256_or_si256(_mm256_and_si256(lbits1, m4), _mm256_and_si256(m3, _mm256_slli_epi16(hbits, 4)));
                qx[1] = _mm256_or_si256(_mm256_and_si256(lbits2, m4), _mm256_and_si256(m3, hbits));
                qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits1, 4), m4), _mm256_and_si256(m3, _mm256_slli_epi16(hbits, 2)));
                qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lbits2, 4), m4), _mm256_and_si256(m3, _mm256_srli_epi16(hbits, 2)));
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
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                                  _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                                  _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                    // Quants are in 0...63, so we can add at most 4 as int16_t to be sure of no int16_t overflow
                    auto sumi = _mm256_add_epi32(_mm256_madd_epi16(m1, sumi1), _mm256_madd_epi16(m1, sumi2));
                    if constexpr (nrc_y == 1) {
                        acc[iy] = _mm256_fmadd_ps(scales, _mm256_cvtepi32_ps(sumi), acc[iy]);
                    } else {
                        acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(scales, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    }
#endif
                }
            }
#ifdef HAVE_FANCY_SIMD
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl)));
                acc[iy] = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
            }
#endif
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

template <typename Dequantizer> void set_functions(std::array<mul_mat_t, IQK_MAX_NY>& funcs) {
#ifdef HAVE_FANCY_SIMD
    if constexpr (std::is_same_v<Dequantizer, DequantizerIQ4XS>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_iqX_k_q8_K_AVX512, Dequantizer, funcs)
    } else {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_AVX512, Dequantizer, funcs)
        funcs[0] = mul_mat_qX_K_q8_K_AVX512_1<Dequantizer>;
    }
#else
    if constexpr (std::is_same_v<Dequantizer, DequantizerQ2K> ||
                  std::is_same_v<Dequantizer, DequantizerQ3K> ||
                  std::is_same_v<Dequantizer, DequantizerQ6K>) {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qY_K_q8_K_T, Dequantizer, funcs)
    } else {
        IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, Dequantizer, funcs)
    }
#endif
}

// The HAVE_FANCY_SIMD should only be #if defined(__AVX512_VNNI__ && defined(__AVX512VL__)
template <int nrc_y>
static void mul_mat_q8_k_r8_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
#ifndef HAVE_FANCY_SIMD
    auto m1 = _mm256_set1_epi16(1);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q8_k_r8 * iq8 = (const block_q8_k_r8 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto d4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[ibl].d));
            for (int ib = 0; ib < QK_K/16; ++ib) {
                qx[0] = _mm256_loadu_si256((const __m256i *)iq8[ibl].qs+4*ib+0);
                qx[1] = _mm256_loadu_si256((const __m256i *)iq8[ibl].qs+4*ib+1);
                qx[2] = _mm256_loadu_si256((const __m256i *)iq8[ibl].qs+4*ib+2);
                qx[3] = _mm256_loadu_si256((const __m256i *)iq8[ibl].qs+4*ib+3);
#ifndef HAVE_FANCY_SIMD
                auto s0 = _mm256_sign_epi8(qx[0], qx[0]);
                auto s1 = _mm256_sign_epi8(qx[1], qx[1]);
                auto s2 = _mm256_sign_epi8(qx[2], qx[2]);
                auto s3 = _mm256_sign_epi8(qx[3], qx[3]);
#else
                qx[0] = _mm256_add_epi8(qx[0], _mm256_set1_epi8(127));
                qx[1] = _mm256_add_epi8(qx[1], _mm256_set1_epi8(127));
                qx[2] = _mm256_add_epi8(qx[2], _mm256_set1_epi8(127));
                qx[3] = _mm256_add_epi8(qx[3], _mm256_set1_epi8(127));
#endif
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y128 = _mm_loadu_si128((const __m128i*)q8.y[iy][ibl].qs+ib);
                    auto y = MM256_SET_M128I(y128, y128);
#ifdef HAVE_FANCY_SIMD
                    isum[iy] = _mm256_dpbusd_epi32(isum[iy], qx[0], _mm256_shuffle_epi32(y, 0x00));
                    isum[iy] = _mm256_dpbusd_epi32(isum[iy], qx[1], _mm256_shuffle_epi32(y, 0x55));
                    isum[iy] = _mm256_dpbusd_epi32(isum[iy], qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    isum[iy] = _mm256_dpbusd_epi32(isum[iy], qx[3], _mm256_shuffle_epi32(y, 0xff));
#else
                    auto sumi1 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0])));
                    auto sumi2 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1])));
                    auto sumi3 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2])));
                    auto sumi4 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3])));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(sumi1, sumi2));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(sumi3, sumi4));
#endif
                }
            }
#ifdef HAVE_FANCY_SIMD
            auto m4 = _mm256_mul_ps(d4, _mm256_set1_ps(-128.f));
#endif
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto d4y = _mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl)));
                acc[iy] = _mm256_fmadd_ps(d4y, _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
#ifdef HAVE_FANCY_SIMD
                auto bsums = (const float *)q8.y[iy][ibl].bsums;
                acc[iy] = _mm256_fmadd_ps(m4, _mm256_set1_ps(bsums[0]), acc[iy]);
#endif
                isum[iy] = _mm256_setzero_si256();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

template <int nrc_y>
static void mul_mat_q8_KV_q8_KV(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    GGML_ASSERT(n%32 == 0);
    __m256i qx[4];
#ifndef HAVE_FANCY_SIMD
    __m256i sx[4];
    auto m1 = _mm256_set1_epi16(1);
#endif
    __m256i acc[nrc_y] = {};
    float dy[nrc_y];
#ifdef HAVE_FANCY_SIMD
    int32_t sy[nrc_y];
#endif
    const int8_t * q8y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        auto dptr = (const float *)info.src1_row(iy);
        dy[iy] = dptr[0];
#ifdef HAVE_FANCY_SIMD
        auto iptr = (const int32_t *)(dptr + 1);
        sy[iy] = -127*iptr[0];
#endif
        q8y[iy] = (const int8_t *)(dptr + 2);
    }
    const int8_t * q8x[4];
    float dx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        for (int kx = 0; kx < 4; ++kx) {
            auto dptr = (const float *)((const char *)vx + (ix+kx)*bx);
            dx[kx] = dptr[0];
            q8x[kx] = (const int8_t *)(dptr + 2);
        }
        for (int i = 0; i < n/32; ++i) {
            for (int kx = 0; kx < 4; ++kx) qx[kx] = _mm256_loadu_si256((const __m256i *)q8x[kx] + i);
            auto t0 = _mm256_unpacklo_epi32(qx[0], qx[1]);
            auto t1 = _mm256_unpacklo_epi32(qx[2], qx[3]);
            auto t2 = _mm256_unpackhi_epi32(qx[0], qx[1]);
            auto t3 = _mm256_unpackhi_epi32(qx[2], qx[3]);
#ifdef HAVE_FANCY_SIMD
            qx[0] = _mm256_add_epi8(_mm256_unpacklo_epi64(t0, t1), _mm256_set1_epi8(127));
            qx[1] = _mm256_add_epi8(_mm256_unpackhi_epi64(t0, t1), _mm256_set1_epi8(127));
            qx[2] = _mm256_add_epi8(_mm256_unpacklo_epi64(t2, t3), _mm256_set1_epi8(127));
            qx[3] = _mm256_add_epi8(_mm256_unpackhi_epi64(t2, t3), _mm256_set1_epi8(127));
#else
            qx[0] = _mm256_unpacklo_epi64(t0, t1); sx[0] = _mm256_sign_epi8(qx[0], qx[0]);
            qx[1] = _mm256_unpackhi_epi64(t0, t1); sx[1] = _mm256_sign_epi8(qx[1], qx[1]);
            qx[2] = _mm256_unpacklo_epi64(t2, t3); sx[2] = _mm256_sign_epi8(qx[2], qx[2]);
            qx[3] = _mm256_unpackhi_epi64(t2, t3); sx[3] = _mm256_sign_epi8(qx[3], qx[3]);
#endif
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = _mm256_loadu_si256((const __m256i *)q8y[iy] + i);
#ifdef HAVE_FANCY_SIMD
                acc[iy] = _mm256_dpbusd_epi32(acc[iy], qx[0], _mm256_shuffle_epi32(y, 0x00));
                acc[iy] = _mm256_dpbusd_epi32(acc[iy], qx[1], _mm256_shuffle_epi32(y, 0x55));
                acc[iy] = _mm256_dpbusd_epi32(acc[iy], qx[2], _mm256_shuffle_epi32(y, 0xaa));
                acc[iy] = _mm256_dpbusd_epi32(acc[iy], qx[3], _mm256_shuffle_epi32(y, 0xff));
#else
                auto dot1 = _mm256_maddubs_epi16(sx[0], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]));
                auto dot2 = _mm256_maddubs_epi16(sx[1], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1]));
                auto dot3 = _mm256_maddubs_epi16(sx[2], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]));
                auto dot4 = _mm256_maddubs_epi16(sx[3], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3]));
                auto dot12 = _mm256_add_epi32(_mm256_madd_epi16(m1, dot1), _mm256_madd_epi16(m1, dot2));
                auto dot34 = _mm256_add_epi32(_mm256_madd_epi16(m1, dot3), _mm256_madd_epi16(m1, dot4));
                acc[iy] = _mm256_add_epi32(acc[iy], _mm256_add_epi32(dot12, dot34));
#endif
            }
        }
        auto scales_x = _mm_loadu_ps(dx);
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sumi = _mm_add_epi32(_mm256_castsi256_si128(acc[iy]), _mm256_extracti128_si256(acc[iy], 1));
#ifdef HAVE_FANCY_SIMD
            sumi = _mm_add_epi32(sumi, _mm_set1_epi32(sy[iy]));
#endif
            auto scale = _mm_mul_ps(scales_x, _mm_set1_ps(dy[iy]));
            info.store(ix, iy, _mm_mul_ps(scale, _mm_cvtepi32_ps(sumi)));
            acc[iy] = _mm256_setzero_si256();
        }
    }
}

// The HAVE_FANCY_SIMD should only be #if defined(__AVX512_VNNI__ && defined(__AVX512VL__)
template <int nrc_y>
static void mul_mat_q8_KV_r8_q8_KV(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%32 == 0);
    GGML_ASSERT(nrc_x%8 == 0);
#ifndef HAVE_FANCY_SIMD
    auto m1 = _mm256_set1_epi16(1);
#endif
    int nb = n / 16;
    __m256i acc[nrc_y] = {};
    __m256i qx[4];
    float dy[nrc_y];
#ifdef HAVE_FANCY_SIMD
    float sy[nrc_y];
#endif
    const int8_t * q8y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        auto dptr = (const float *)info.src1_row(iy);
        dy[iy] = dptr[0];
#ifdef HAVE_FANCY_SIMD
        auto iptr = (const int32_t *)(dptr + 1);
        sy[iy] = -127*iptr[0];
#endif
        q8y[iy] = (const int8_t *)(dptr + 2);
    }
    for (int ix = 0; ix < nrc_x; ix += 8) {
        auto dptr = (const float *)((const char *)vx + ix*bx);
        auto dx = _mm256_loadu_ps(dptr);
        auto q8x = (const int8_t *)(dptr + 8);
        for (int ib = 0; ib < nb; ++ib) { // Blocks of 16 for 8 interleaved rows
            qx[0] = _mm256_loadu_si256((const __m256i *)q8x+4*ib+0);
            qx[1] = _mm256_loadu_si256((const __m256i *)q8x+4*ib+1);
            qx[2] = _mm256_loadu_si256((const __m256i *)q8x+4*ib+2);
            qx[3] = _mm256_loadu_si256((const __m256i *)q8x+4*ib+3);
#ifndef HAVE_FANCY_SIMD
            auto s0 = _mm256_sign_epi8(qx[0], qx[0]);
            auto s1 = _mm256_sign_epi8(qx[1], qx[1]);
            auto s2 = _mm256_sign_epi8(qx[2], qx[2]);
            auto s3 = _mm256_sign_epi8(qx[3], qx[3]);
#else
            qx[0] = _mm256_add_epi8(qx[0], _mm256_set1_epi8(127));
            qx[1] = _mm256_add_epi8(qx[1], _mm256_set1_epi8(127));
            qx[2] = _mm256_add_epi8(qx[2], _mm256_set1_epi8(127));
            qx[3] = _mm256_add_epi8(qx[3], _mm256_set1_epi8(127));
#endif
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y128 = _mm_loadu_si128((const __m128i*)q8y[iy]+ib);
                auto y = MM256_SET_M128I(y128, y128);
#ifdef HAVE_FANCY_SIMD
                acc[iy] = _mm256_dpbusd_epi32(acc[iy], qx[0], _mm256_shuffle_epi32(y, 0x00));
                acc[iy] = _mm256_dpbusd_epi32(acc[iy], qx[1], _mm256_shuffle_epi32(y, 0x55));
                acc[iy] = _mm256_dpbusd_epi32(acc[iy], qx[2], _mm256_shuffle_epi32(y, 0xaa));
                acc[iy] = _mm256_dpbusd_epi32(acc[iy], qx[3], _mm256_shuffle_epi32(y, 0xff));
#else
                auto sumi1 = _mm256_maddubs_epi16(s0, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), qx[0]));
                auto sumi2 = _mm256_maddubs_epi16(s1, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), qx[1]));
                auto sumi3 = _mm256_maddubs_epi16(s2, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), qx[2]));
                auto sumi4 = _mm256_maddubs_epi16(s3, _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), qx[3]));
                auto sumi12 = _mm256_add_epi32(_mm256_madd_epi16(m1, sumi1), _mm256_madd_epi16(m1, sumi2));
                auto sumi34 = _mm256_add_epi32(_mm256_madd_epi16(m1, sumi3), _mm256_madd_epi16(m1, sumi4));
                acc[iy] = _mm256_add_epi32(acc[iy], _mm256_add_epi32(sumi12, sumi34));
#endif
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto scale = _mm256_mul_ps(dx, _mm256_set1_ps(dy[iy]));
#ifdef HAVE_FANCY_SIMD
            acc[iy] = _mm256_add_epi32(acc[iy], _mm256_set1_epi32(sy[iy]));
#endif
            info.store(ix, iy, _mm256_mul_ps(scale, _mm256_cvtepi32_ps(acc[iy])));
            acc[iy] = _mm256_setzero_si256();
        }
    }
}

} // namespace

bool iqk_set_kernels_kquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {

    auto etypeA = ggml_type(typeA);
    auto expected_type_B = etypeA == GGML_TYPE_IQ4_XS_R8 || etypeA == GGML_TYPE_Q4_K_R4 || etypeA == GGML_TYPE_Q5_K_R4 ? GGML_TYPE_Q8_K32
                         : etypeA == GGML_TYPE_Q8_K_R8 ? GGML_TYPE_Q8_KR8
                         : etypeA == GGML_TYPE_Q8_KV || etypeA == GGML_TYPE_Q8_KV_R8 ? GGML_TYPE_Q8_KV
                         : GGML_TYPE_Q8_K;

    if (ne00%QK_K != 0 || ggml_type(typeB) != expected_type_B) {
        return false;
    }

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_Q2_K:
            set_functions<DequantizerQ2K>(kernels);
            break;
        case GGML_TYPE_Q3_K:
            set_functions<DequantizerQ3K>(kernels);
            break;
        case GGML_TYPE_Q4_K:
            set_functions<DequantizerQ4K>(kernels);
            break;
        case GGML_TYPE_Q5_K:
            set_functions<DequantizerQ5K>(kernels);
            break;
        case GGML_TYPE_Q6_K:
            set_functions<DequantizerQ6K>(kernels);
            break;
        case GGML_TYPE_IQ4_XS:
            set_functions<DequantizerIQ4XS>(kernels);
            break;
        case GGML_TYPE_Q2_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q2_k_r4_q8_k, kernels)
            break;
        case GGML_TYPE_Q3_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q3_k_r4_q8_k, kernels)
            break;
        case GGML_TYPE_Q4_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q4_k_r4_q8_k, kernels)
            break;
        case GGML_TYPE_Q5_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q5_k_r4_q8_k, kernels)
            break;
        case GGML_TYPE_Q6_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q6_k_r4_q8_k, kernels)
            break;
        case GGML_TYPE_IQ4_XS_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_xs_r8_q8_k_avx2, kernels)
            break;
        case GGML_TYPE_Q8_K_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_k_r8_q8_k, kernels)
#ifdef HAVE_FANCY_SIMD
            func16 = mul_mat_q8_k_r8_q8_k<16>;
#endif
            break;
        case GGML_TYPE_Q8_KV:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_KV_q8_KV, kernels)
#ifdef HAVE_FANCY_SIMD
            func16 = mul_mat_q8_KV_q8_KV<16>;
#endif
            break;
        case GGML_TYPE_Q8_KV_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_KV_r8_q8_KV, kernels);
            break;
        default:
            return false;
    }

    return true;

}

#else
// --------------------------------- __aarch64__ --------------------------------------

namespace {

template <typename Q8>
inline void accum_mins_8(const int16x8_t& mins, const Q8& q8, float32x4_t * acc, int i, float c) {
    for (int iy = 0; iy < Q8::nrc_y; ++iy) {
        auto q8s = q8.load_bsums8(iy, i);
        int32x4_t b1 = vmull_s16(vget_low_s16(mins), vget_low_s16(q8s));
        int32x4_t b2 = vmull_s16(vget_high_s16(mins), vget_high_s16(q8s));
        float32x4_t prod = vcvtq_f32_s32(vaddq_s32(b1, b2));
        acc[iy] = vmlaq_f32(acc[iy], prod, vdupq_n_f32(c*q8.scale(iy, i)));
    }
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

struct Scales8 {
    uint32_t utmp[4];
    const uint8_t * sc8 = (const uint8_t *)utmp;
    template <typename Q8, typename Qx>
    inline int32x4x2_t process_scales_mins(const Qx& x, const Q8& q8, int i, float32x4_t * acc) {
        make_q4_scales(x.scales, utmp);
        int16x8_t mins = vmovl_s8(vld1_s8((const int8_t *)sc8 + 8));
        accum_mins_8(mins, q8, acc, i, -GGML_FP16_TO_FP32(x.dmin));

        uint8x8_t scales8 = vld1_u8(sc8);
        uint16x8_t scales16 = vmovl_u8(scales8);
        int32x4x2_t scales = {vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales16))),
                              vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales16)))};
        return scales;
    }
};

struct DequantizerQ4K final : public BaseDequantizer<block_q4_K> {
    DequantizerQ4K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        return s8.process_scales_mins(x[i], q8, i, acc);
    }
    inline void prepare(int i, int j) {
        if (nrc == 1) bits.prepare_v2(x[i].qs+64*j);
        else bits.prepare(x[i].qs+64*j);
    }

    Q4bits bits;
    Scales8 s8;

};

struct HighBit5 {
    const uint8x16_t mhb = vdupq_n_u8(0x10);
    uint8x16x2_t bits;
    inline void apply(uint8x16x4_t& b1, uint8x16x4_t& b2, bool do_shift) {
        b1.val[0] = vorrq_u8(b1.val[0], vandq_u8(vshlq_n_u8(bits.val[0], 4), mhb));
        b1.val[1] = vorrq_u8(b1.val[1], vandq_u8(vshlq_n_u8(bits.val[1], 4), mhb));
        b1.val[2] = vorrq_u8(b1.val[2], vandq_u8(vshlq_n_u8(bits.val[0], 3), mhb));
        b1.val[3] = vorrq_u8(b1.val[3], vandq_u8(vshlq_n_u8(bits.val[1], 3), mhb));

        b2.val[0] = vorrq_u8(b2.val[0], vandq_u8(vshlq_n_u8(bits.val[0], 2), mhb));
        b2.val[1] = vorrq_u8(b2.val[1], vandq_u8(vshlq_n_u8(bits.val[1], 2), mhb));
        b2.val[2] = vorrq_u8(b2.val[2], vandq_u8(vshlq_n_u8(bits.val[0], 1), mhb));
        b2.val[3] = vorrq_u8(b2.val[3], vandq_u8(vshlq_n_u8(bits.val[1], 1), mhb));

        if (do_shift) {
            bits.val[0] = vshrq_n_u8(bits.val[0], 4);
            bits.val[1] = vshrq_n_u8(bits.val[1], 4);
        }
    }
};

struct HighBit3 {
    const uint8x16_t mhb = vdupq_n_u8(0x04);
    uint8x16x2_t bits;
    inline void apply(uint8x16x4_t& b1, uint8x16x4_t& b2, bool do_shift) {
        b1.val[0] = vorrq_u8(b1.val[0], vandq_u8(vshlq_n_u8(bits.val[0], 2), mhb));
        b1.val[1] = vorrq_u8(b1.val[1], vandq_u8(vshlq_n_u8(bits.val[1], 2), mhb));
        b1.val[2] = vorrq_u8(b1.val[2], vandq_u8(vshlq_n_u8(bits.val[0], 1), mhb));
        b1.val[3] = vorrq_u8(b1.val[3], vandq_u8(vshlq_n_u8(bits.val[1], 1), mhb));

        b2.val[0] = vorrq_u8(b2.val[0], vandq_u8(bits.val[0], mhb));
        b2.val[1] = vorrq_u8(b2.val[1], vandq_u8(bits.val[1], mhb));
        b2.val[2] = vorrq_u8(b2.val[2], vandq_u8(vshrq_n_u8(bits.val[0], 1), mhb));
        b2.val[3] = vorrq_u8(b2.val[3], vandq_u8(vshrq_n_u8(bits.val[1], 1), mhb));

        if (do_shift) {
            bits.val[0] = vshrq_n_u8(bits.val[0], 4);
            bits.val[1] = vshrq_n_u8(bits.val[1], 4);
        }
    }
};

struct DequantizerQ5K final : public BaseDequantizer<block_q5_K> {
    DequantizerQ5K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        h.bits = vld1q_u8_x2(x[i].qh);
        return s8.process_scales_mins(x[i], q8, i, acc);
    }
    inline void prepare(int i, int j) {
        if (nrc == 1) bits.prepare_v2(x[i].qs+64*j);
        else bits.prepare(x[i].qs+64*j);
        h.apply(bits.b1, bits.b2, j == 0);
    }

    Q4bits bits;
    HighBit5 h;
    Scales8 s8;

    uint8x16x2_t hbits;

};

inline int32x4x4_t make_wider(const int16x8x2_t& scales16) {
    int32x4x4_t scales = {
        vmovl_s16(vget_low_s16 (scales16.val[0])),
        vmovl_s16(vget_high_s16(scales16.val[0])),
        vmovl_s16(vget_low_s16 (scales16.val[1])),
        vmovl_s16(vget_high_s16(scales16.val[1])),
    };
    return scales;
}

template <typename Q8>
inline int32x4x4_t process_scales_mins_16(const int8x16_t& scales8, const Q8& q8, float32x4_t * acc, int i, float c) {
    int16x8x2_t scales16;
    scales16.val[0] = vmovl_s8(vget_low_s8(scales8));
    scales16.val[1] = vmovl_s8(vget_high_s8(scales8));
    accum_mins_16(scales16, q8, acc, i, c);
    return make_wider(scales16);
}

struct DequantizerQ6K final : public BaseDequantizer<block_q6_K> {
    DequantizerQ6K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        return process_scales_mins_16(vld1q_s8(x[i].scales), q8, acc, i, -32.f*d);
    }
    inline void prepare(int i, int j) {

        auto hbits = vld1q_u8_x2(x[i].qh + 32*j);

        bits.prepare64(x[i].ql+64*j);
        bits.b1.val[0] = vorrq_u8(bits.b1.val[0], vandq_u8(vshlq_n_u8(hbits.val[0], 4), mhb));
        bits.b1.val[1] = vorrq_u8(bits.b1.val[1], vandq_u8(vshlq_n_u8(hbits.val[1], 4), mhb));
        bits.b1.val[2] = vorrq_u8(bits.b1.val[2], vandq_u8(vshlq_n_u8(hbits.val[0], 2), mhb));
        bits.b1.val[3] = vorrq_u8(bits.b1.val[3], vandq_u8(vshlq_n_u8(hbits.val[1], 2), mhb));

        bits.b2.val[0] = vorrq_u8(bits.b2.val[0], vandq_u8(hbits.val[0], mhb));
        bits.b2.val[1] = vorrq_u8(bits.b2.val[1], vandq_u8(hbits.val[1], mhb));
        bits.b2.val[2] = vorrq_u8(bits.b2.val[2], vandq_u8(vshrq_n_u8(hbits.val[0], 2), mhb));
        bits.b2.val[3] = vorrq_u8(bits.b2.val[3], vandq_u8(vshrq_n_u8(hbits.val[1], 2), mhb));

    }

    Q4bits bits;

    const uint8x16_t mhb = vdupq_n_u8(0x30);

};

struct DequantizerQ3K final : public BaseDequantizer<block_q3_K> {
    DequantizerQ3K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        h.bits = vld1q_u8_x2(x[i].hmask);
        mask = vdupq_n_u8(0x01);
        const uint16_t * sc16 = (const uint16_t *)x[i].scales;
        uint32_t aux0 = sc16[0] | (sc16[1] << 16);
        uint32_t aux1 = sc16[2] | (sc16[3] << 16);
        uint32_t aux2 = sc16[4] | (sc16[5] << 16);
        aux32[0] =  (aux0       & 0x0f0f0f0f) | ((aux2 << 4) & 0x30303030);
        aux32[1] =  (aux1       & 0x0f0f0f0f) | ((aux2 << 2) & 0x30303030);
        aux32[2] = ((aux0 >> 4) & 0x0f0f0f0f) | ((aux2 >> 0) & 0x30303030);
        aux32[3] = ((aux1 >> 4) & 0x0f0f0f0f) | ((aux2 >> 2) & 0x30303030);
        auto scales8 = vaddq_s8(vld1q_s8((const int8_t *)aux32), vdupq_n_s8(-32));
        if (nrc > 1) {
            return process_scales_mins_16(scales8, q8, acc, i, -4.f*d);
        }
        int16x8x2_t scales16;
        scales16.val[0] = vmovl_s8(vget_low_s8(scales8));
        scales16.val[1] = vmovl_s8(vget_high_s8(scales8));
        return make_wider(scales16);
    }

    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs+32*j);
        if (nrc > 1) {
            h.apply(bits.b1, bits.b2, j == 0);
        } else {
            auto minus4 = vdupq_n_u8(0xfc);
            auto zero = vdupq_n_u8(0);
            bits.b1.val[0] = vorrq_u8(bits.b1.val[0], vandq_u8(minus4, vceqq_u8(vandq_u8(h.bits.val[0], mask), zero)));
            bits.b1.val[1] = vorrq_u8(bits.b1.val[1], vandq_u8(minus4, vceqq_u8(vandq_u8(h.bits.val[1], mask), zero)));
            mask = vshlq_n_u8(mask, 1);
            bits.b1.val[2] = vorrq_u8(bits.b1.val[2], vandq_u8(minus4, vceqq_u8(vandq_u8(h.bits.val[0], mask), zero)));
            bits.b1.val[3] = vorrq_u8(bits.b1.val[3], vandq_u8(minus4, vceqq_u8(vandq_u8(h.bits.val[1], mask), zero)));
            mask = vshlq_n_u8(mask, 1);
            bits.b2.val[0] = vorrq_u8(bits.b2.val[0], vandq_u8(minus4, vceqq_u8(vandq_u8(h.bits.val[0], mask), zero)));
            bits.b2.val[1] = vorrq_u8(bits.b2.val[1], vandq_u8(minus4, vceqq_u8(vandq_u8(h.bits.val[1], mask), zero)));
            mask = vshlq_n_u8(mask, 1);
            bits.b2.val[2] = vorrq_u8(bits.b2.val[2], vandq_u8(minus4, vceqq_u8(vandq_u8(h.bits.val[0], mask), zero)));
            bits.b2.val[3] = vorrq_u8(bits.b2.val[3], vandq_u8(minus4, vceqq_u8(vandq_u8(h.bits.val[1], mask), zero)));
            mask = vshlq_n_u8(mask, 1);
        }
    }

    uint32_t aux32[4];

    Q2bits bits;

    uint8x16_t mask;
    HighBit3 h;

};

struct DequantizerQ2K final : public BaseDequantizer<block_q2_K> {
    DequantizerQ2K(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return true; }

    template <typename Q8>
    inline void process_scales(int i, const Q8& q8, float32x4_t * acc) {
        d = GGML_FP16_TO_FP32(x[i].d);
        auto scales_and_mins = vld1q_u8(x[i].scales);
        auto mins8 = vreinterpretq_s8_u8(vshrq_n_u8(scales_and_mins, 4));
        int16x8x2_t scales16;
        scales16.val[0] = vmovl_s8(vget_low_s8(mins8));
        scales16.val[1] = vmovl_s8(vget_high_s8(mins8));
        accum_mins_16(scales16, q8, acc, i, -GGML_FP16_TO_FP32(x[i].dmin));

        scales8 = vandq_u8(scales_and_mins, vdupq_n_u8(0xf));
    }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        process_scales(i, q8, acc);
        int16x8x2_t scales16;
        scales16.val[0] = vmovl_s8(vget_low_s8(vreinterpretq_s8_u8(scales8)));
        scales16.val[1] = vmovl_s8(vget_high_s8(vreinterpretq_s8_u8(scales8)));
        return make_wider(scales16);
    }

    template <typename Q8>
    inline void compute(const Q8& q8, int i, int j, int32x4_t * sumi) {
        auto m1 = vdupq_n_u8(1);
        auto shuffle = vdupq_n_u8(8*j);
        bits.b1.val[0] = vmulq_u8(bits.b1.val[0], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b1.val[1] = vmulq_u8(bits.b1.val[1], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b1.val[2] = vmulq_u8(bits.b1.val[2], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b1.val[3] = vmulq_u8(bits.b1.val[3], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b2.val[0] = vmulq_u8(bits.b2.val[0], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b2.val[1] = vmulq_u8(bits.b2.val[1], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b2.val[2] = vmulq_u8(bits.b2.val[2], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        bits.b2.val[3] = vmulq_u8(bits.b2.val[3], vqtbl1q_u8(scales8, shuffle)); shuffle = vaddq_u8(shuffle, m1);
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            auto q8b_1 = q8.load_quants(iy, i, 4*j+0);
            sumi[iy] = ggml_vdotq_s32(ggml_vdotq_s32(sumi[iy], vreinterpretq_s8_u8(bits.b1.val[0]), q8b_1.val[0]),
                    vreinterpretq_s8_u8(bits.b1.val[1]), q8b_1.val[1]);

            auto q8b_2 = q8.load_quants(iy, i, 4*j+1);
            sumi[iy] = ggml_vdotq_s32(ggml_vdotq_s32(sumi[iy], vreinterpretq_s8_u8(bits.b1.val[2]), q8b_2.val[0]),
                    vreinterpretq_s8_u8(bits.b1.val[3]), q8b_2.val[1]);

            auto q8b_3 = q8.load_quants(iy, i, 4*j+2);
            sumi[iy] = ggml_vdotq_s32(ggml_vdotq_s32(sumi[iy], vreinterpretq_s8_u8(bits.b2.val[0]), q8b_3.val[0]),
                    vreinterpretq_s8_u8(bits.b2.val[1]), q8b_3.val[1]);

            auto q8b_4 = q8.load_quants(iy, i, 4*j+3);
            sumi[iy] = ggml_vdotq_s32(ggml_vdotq_s32(sumi[iy], vreinterpretq_s8_u8(bits.b2.val[2]), q8b_4.val[0]),
                    vreinterpretq_s8_u8(bits.b2.val[3]), q8b_4.val[1]);
        }
    }

    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs+32*j);
    }

    uint32_t aux32[4];

    uint8x16_t scales8;

    Q2bits bits;

};

struct DequantizerIQ4XS final : public BaseDequantizer<block_iq4_xs> {

    static int8x16_t load_values() {
        static const int8_t iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        return vld1q_s8(iq4nl_values);
    }

    DequantizerIQ4XS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc), values(load_values()) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    inline void new_row(int ix) { x = (const block_iq4_xs *)((const char *)vx + bx*ix); }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        (void)q8;
        (void)acc;
        d = GGML_FP16_TO_FP32(x[i].d);
        const uint16_t scales_h = x[i].scales_h;
        const uint16_t * scales_l = (const uint16_t *)x[i].scales_l;
        aux32[0] = scales_l[0] | (scales_l[1] << 16);
        aux32[1] = aux32[0] >> 4;
        // scl is ordered as 0, 2, 4, 6, 1, 3, 5, 7
        uint8x8_t scl8 = vand_u8(vld1_u8((const uint8_t *)aux32), vdup_n_u8(0xf));
        uint16_t * aux16 = (uint16_t *)aux32;
        aux16[0] = scales_h << 4; aux16[1] = scales_h << 2; aux16[2] = scales_h; aux16[3] = scales_h >> 2;
        // sch is ordered as 0, 4, 1, 5, 2, 6, 3, 7
        uint8x8_t sch8 = vand_u8(vld1_u8((const uint8_t *)aux16), vdup_n_u8(0x30));
        int8x8_t scales8 = vadd_s8(vreinterpret_s8_u8(vorr_u8(scl8, vtbl1_u8(sch8, vreinterpret_u8_u32(hshuff)))), vdup_n_s8(-32));
        // shuffle 0, 2, 4, 6, 1, 3, 5, 7 -> 0, 1, 2, 3, 4, 5, 6, 7
        scales8 = vtbl1_s8(scales8, vreinterpret_s8_u32(hshuff));
        int16x8_t scales16 = vmovl_s8(scales8);
        int32x4x2_t scales = {vmovl_s16(vget_low_s16(scales16)), vmovl_s16(vget_high_s16(scales16))};
        return scales;
    }
    inline void prepare(int i, int j) {
        bits.prepare16(x[i].qs+64*j);
        //if (nrc == 1) {
        //    bits.prepare16_v2(x[i].qs+64*j);
        //} else {
        //    bits.prepare16(x[i].qs+64*j);
        //}
        for (int k = 0; k < 4; ++k) {
            bits.b1.val[k] = vreinterpretq_u8_s8(vqtbl1q_s8(values, bits.b1.val[k]));
            bits.b2.val[k] = vreinterpretq_u8_s8(vqtbl1q_s8(values, bits.b2.val[k]));
        }
    }

    Q4bits bits;
    const int8x16_t values;
    uint32_t aux32[2];

    constexpr static uint32x2_t hshuff = {0x05010400, 0x07030602};

};

IQK_ALWAYS_INLINE void prepare_q4_k_quants(const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
    qx[0] = vandq_u8(bits.val[0], m4);   //  0...3 from the 4 rows
    qx[1] = vandq_u8(bits.val[1], m4);   // 16..19
    qx[2] = vandq_u8(bits.val[2], m4);   //  4...7
    qx[3] = vandq_u8(bits.val[3], m4);   // 20..23
    qx[4] = vshrq_n_u8(bits.val[0], 4);  //  8..11
    qx[5] = vshrq_n_u8(bits.val[1], 4);  // 24..27
    qx[6] = vshrq_n_u8(bits.val[2], 4);  // 12..15
    qx[7] = vshrq_n_u8(bits.val[3], 4);  // 28..31
}

template <int nrc_y>
void mul_mat_q2_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto mf = vdupq_n_u8(0x0f);
    auto m03 = vdupq_n_u8(0x03);
    int nbl = n / QK_K;
    int8x16_t qx[4];
    float32x4_t acc[nrc_y] = {};
    int16x8x4_t i16scales;
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q2_k_r4 * iq2 = (const block_q2_k_r4 *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            int32x4_t isum[nrc_y] = {};
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq2[ibl].d));
            auto m4 = vmulq_f32(vdupq_n_f32(-1.f), vcvt_f32_f16(vld1_f16((const float16_t *)iq2[ibl].d+4)));
            for (int is = 0; is < 2; ++is) {
                auto sl = vld1q_u8_x2(iq2[ibl].scales + 32*is);
                auto m = vshrq_n_u8(sl.val[0], 4);
                i16scales.val[0] = vmovl_u8(vget_low_u8 (m));
                i16scales.val[1] = vmovl_u8(vget_high_u8(m));
                m = vshrq_n_u8(sl.val[1], 4);
                i16scales.val[2] = vmovl_u8(vget_low_u8 (m));
                i16scales.val[3] = vmovl_u8(vget_high_u8(m));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto sumi = vdupq_n_s32(0);
                    auto bsums = vld1q_s16(q8.y[iy][ibl].bsums + 8*is);
                    auto b8 = vget_low_s16(bsums);
                    //auto bsums = q8.load_bsums(iy, ibl);
                    //auto b8 = vget_low_s16(bsums.val[0]);
                    sumi = vmlal_lane_s16(sumi, vget_low_s16 (i16scales.val[0]), b8, 0);
                    sumi = vmlal_lane_s16(sumi, vget_high_s16(i16scales.val[0]), b8, 1);
                    sumi = vmlal_lane_s16(sumi, vget_low_s16 (i16scales.val[1]), b8, 2);
                    sumi = vmlal_lane_s16(sumi, vget_high_s16(i16scales.val[1]), b8, 3);
                    b8 = vget_high_s16(bsums);
                    sumi = vmlal_lane_s16(sumi, vget_low_s16 (i16scales.val[2]), b8, 0);
                    sumi = vmlal_lane_s16(sumi, vget_high_s16(i16scales.val[2]), b8, 1);
                    sumi = vmlal_lane_s16(sumi, vget_low_s16 (i16scales.val[3]), b8, 2);
                    sumi = vmlal_lane_s16(sumi, vget_high_s16(i16scales.val[3]), b8, 3);
                    acc[iy] = vfmaq_f32(acc[iy], vmulq_f32(m4, vdupq_n_f32(q8.scale(iy, ibl))), vcvtq_f32_s32(sumi));
                }
                m = vandq_u8(sl.val[0], mf);
                i16scales.val[0] = vmovl_u8(vget_low_u8 (m));
                i16scales.val[1] = vmovl_u8(vget_high_u8(m));
                m = vandq_u8(sl.val[1], mf);
                i16scales.val[2] = vmovl_u8(vget_low_u8 (m));
                i16scales.val[3] = vmovl_u8(vget_high_u8(m));
                for (int ib = 0; ib < 4; ++ib) {
                    auto bits = vld1q_u8_x2(iq2[ibl].qs + 128*is + 32*ib);
                    auto scales = vmovl_s16(vget_low_s16 (i16scales.val[ib]));
                    qx[0] = vreinterpretq_s8_u8(vandq_u8(           bits.val[0],     m03));
                    qx[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(bits.val[0], 2), m03));
                    qx[2] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(bits.val[0], 4), m03));
                    qx[3] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(bits.val[0], 6), m03));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8(q8.y[iy][ibl].qs+128*is+32*ib);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                    }
                    scales = vmovl_s16(vget_high_s16(i16scales.val[ib]));
                    qx[0] = vreinterpretq_s8_u8(vandq_u8(           bits.val[1],     m03));
                    qx[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(bits.val[1], 2), m03));
                    qx[2] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(bits.val[1], 4), m03));
                    qx[3] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(bits.val[1], 6), m03));
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
void mul_mat_q3_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto mf = vdupq_n_u8(0x0f);
    auto m30 = vdupq_n_u8(0x30);
    auto m32 = vdupq_n_s8(-32);
    auto m03 = vdupq_n_u8(0x03);
    auto m04 = vdupq_n_u8(0x04);
    int nbl = n / QK_K;
    int8x16_t qx[4];
    float32x4_t acc[nrc_y] = {};
    int8x16x4_t i8scales;
    int16x8x4_t i16scales;
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q3_k_r4 * iq3 = (const block_q3_k_r4 *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            int32x4_t isum[nrc_y] = {};
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq3[ibl].d));
            auto sl = vld1q_u8_x2(iq3[ibl].scales_l);
            auto sh = vld1q_u8(iq3[ibl].scales_h);
            i8scales.val[0] = vaddq_s8(m32, vorrq_u8(vandq_u8(sl.val[0],  mf), vandq_u8(vshlq_n_u8(sh, 4), m30)));
            i8scales.val[1] = vaddq_s8(m32, vorrq_u8(vandq_u8(sl.val[1],  mf), vandq_u8(vshlq_n_u8(sh, 2), m30)));
            i8scales.val[2] = vaddq_s8(m32, vorrq_u8(vshrq_n_u8(sl.val[0], 4), vandq_u8(sh, m30)));
            i8scales.val[3] = vaddq_s8(m32, vorrq_u8(vshrq_n_u8(sl.val[1], 4), vandq_u8(vshrq_n_u8(sh, 2), m30)));
            for (int is = 0; is < 2; ++is) {
                i16scales.val[0] = vmovl_s8(vget_low_s8 (i8scales.val[2*is+0]));
                i16scales.val[1] = vmovl_s8(vget_high_s8(i8scales.val[2*is+0]));
                i16scales.val[2] = vmovl_s8(vget_low_s8 (i8scales.val[2*is+1]));
                i16scales.val[3] = vmovl_s8(vget_high_s8(i8scales.val[2*is+1]));
                for (int ib = 0; ib < 4; ++ib) {
                    auto lbits = vld1q_u8_x2(iq3[ibl].qs + 128*is + 32*ib);
                    auto hbits = vld1q_u8(iq3[ibl].qh + 64*is + 16*ib);
                    hbits = veorq_u8(hbits, vdupq_n_u8(0xff));
                    auto scales = vmovl_s16(vget_low_s16 (i16scales.val[ib]));
                    qx[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(           lbits.val[0],     m03)), vreinterpretq_s8_u8(vandq_u8(m04, vshlq_n_u8(hbits, 2))));
                    qx[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(lbits.val[0], 2), m03)), vreinterpretq_s8_u8(vandq_u8(m04, vshlq_n_u8(hbits, 1))));
                    qx[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(lbits.val[0], 4), m03)), vreinterpretq_s8_u8(vandq_u8(m04, hbits)));
                    qx[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(lbits.val[0], 6), m03)), vreinterpretq_s8_u8(vandq_u8(m04, vshrq_n_u8(hbits, 1))));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8(q8.y[iy][ibl].qs+128*is+32*ib);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                    }
                    scales = vmovl_s16(vget_high_s16(i16scales.val[ib]));
                    qx[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(           lbits.val[1],     m03)), vreinterpretq_s8_u8(vandq_u8(m04, vshrq_n_u8(hbits, 2))));
                    qx[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(lbits.val[1], 2), m03)), vreinterpretq_s8_u8(vandq_u8(m04, vshrq_n_u8(hbits, 3))));
                    qx[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(lbits.val[1], 4), m03)), vreinterpretq_s8_u8(vandq_u8(m04, vshrq_n_u8(hbits, 4))));
                    qx[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(lbits.val[1], 6), m03)), vreinterpretq_s8_u8(vandq_u8(m04, vshrq_n_u8(hbits, 5))));
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
void mul_mat_q4_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto mf = vdupq_n_u8(0xf);
    auto m3 = vdupq_n_u8(0x30);
    int nbl = n / QK_K;
    int8x16_t qx[8];
    int8x16x2_t iscales;
    int32x4x4_t scales;
    float32x4_t acc[nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q4_k_r4 * iq4 = (const block_q4_k_r4 *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq4[ibl].d));
            auto m4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq4[ibl].d+4));
            m4 = vmulq_f32(m4, vdupq_n_f32(-1.f));
            auto sl = vld1q_u8_x2(iq4[ibl].scales_l);
            auto sh = vld1q_u8(iq4[ibl].scales_h);
            iscales.val[0] = vorrq_u8(vshrq_n_u8(sl.val[0], 4), vandq_u8(vshlq_n_u8(sh, 2), m3));
            iscales.val[1] = vorrq_u8(vshrq_n_u8(sl.val[1], 4), vandq_u8(vshrq_n_u8(sh, 2), m3));
            for (int is = 0; is < 2; ++is) {
                auto iscales16_1 = vmovl_s8(vget_low_s8(iscales.val[is]));
                auto iscales16_2 = vmovl_s8(vget_high_s8(iscales.val[is]));
                float32x4x4_t fscales;
                fscales.val[0] = vmulq_f32(m4, vcvtq_f32_s32(vmovl_s16(vget_low_s16(iscales16_1))));
                fscales.val[1] = vmulq_f32(m4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(iscales16_1))));
                fscales.val[2] = vmulq_f32(m4, vcvtq_f32_s32(vmovl_s16(vget_low_s16(iscales16_2))));
                fscales.val[3] = vmulq_f32(m4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(iscales16_2))));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto m8 = vld1q_f32((const float *)q8.y[iy][ibl].bsums + 4*is);
                    acc[iy] = vmlaq_laneq_f32(acc[iy], fscales.val[0], m8, 0);
                    acc[iy] = vmlaq_laneq_f32(acc[iy], fscales.val[1], m8, 1);
                    acc[iy] = vmlaq_laneq_f32(acc[iy], fscales.val[2], m8, 2);
                    acc[iy] = vmlaq_laneq_f32(acc[iy], fscales.val[3], m8, 3);
                }
            }
            iscales.val[0] = vorrq_u8(vandq_u8(sl.val[0], mf), vandq_u8(vshlq_n_u8(sh, 4), m3));
            iscales.val[1] = vorrq_u8(vandq_u8(sl.val[1], mf), vandq_u8(sh, m3));
            int32x4_t isum[nrc_y] = {};
            for (int is = 0; is < 2; ++is) {
                auto iscales16_1 = vmovl_s8(vget_low_s8(iscales.val[is]));
                auto iscales16_2 = vmovl_s8(vget_high_s8(iscales.val[is]));
                scales.val[0] = vmovl_s16(vget_low_s16(iscales16_1));
                scales.val[1] = vmovl_s16(vget_high_s16(iscales16_1));
                scales.val[2] = vmovl_s16(vget_low_s16(iscales16_2));
                scales.val[3] = vmovl_s16(vget_high_s16(iscales16_2));
                for (int ib = 0; ib < 4; ++ib) {
                    auto bits = vld1q_u8_x4(iq4[ibl].qs + 256*is + 64*ib);
                    prepare_q4_k_quants(mf, bits, qx);
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8_x2(q8.y[iy][ibl].qs+128*is+32*ib);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales.val[ib], sumi);
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
void mul_mat_q5_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto mf = vdupq_n_u8(0xf);
    auto m30 = vdupq_n_u8(0x30);
    auto m10 = vdupq_n_u8(0x10);
    int nbl = n / QK_K;
    int8x16_t qx[8];
    int8x16x2_t iscales;
    int32x4x4_t scales;
    float32x4_t acc[nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q5_k_r4 * iq5 = (const block_q5_k_r4 *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq5[ibl].d));
            auto m4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq5[ibl].d+4));
            m4 = vmulq_f32(m4, vdupq_n_f32(-1.f));
            auto sl = vld1q_u8_x2(iq5[ibl].scales_l);
            auto sh = vld1q_u8(iq5[ibl].scales_h);
            iscales.val[0] = vorrq_u8(vshrq_n_u8(sl.val[0], 4), vandq_u8(vshlq_n_u8(sh, 2), m30));
            iscales.val[1] = vorrq_u8(vshrq_n_u8(sl.val[1], 4), vandq_u8(vshrq_n_u8(sh, 2), m30));
            for (int is = 0; is < 2; ++is) {
                auto iscales16_1 = vmovl_s8(vget_low_s8(iscales.val[is]));
                auto iscales16_2 = vmovl_s8(vget_high_s8(iscales.val[is]));
                float32x4x4_t fscales;
                fscales.val[0] = vmulq_f32(m4, vcvtq_f32_s32(vmovl_s16(vget_low_s16(iscales16_1))));
                fscales.val[1] = vmulq_f32(m4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(iscales16_1))));
                fscales.val[2] = vmulq_f32(m4, vcvtq_f32_s32(vmovl_s16(vget_low_s16(iscales16_2))));
                fscales.val[3] = vmulq_f32(m4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(iscales16_2))));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto m8 = vld1q_f32((const float *)q8.y[iy][ibl].bsums + 4*is);
                    acc[iy] = vmlaq_laneq_f32(acc[iy], fscales.val[0], m8, 0);
                    acc[iy] = vmlaq_laneq_f32(acc[iy], fscales.val[1], m8, 1);
                    acc[iy] = vmlaq_laneq_f32(acc[iy], fscales.val[2], m8, 2);
                    acc[iy] = vmlaq_laneq_f32(acc[iy], fscales.val[3], m8, 3);
                }
            }
            iscales.val[0] = vorrq_u8(vandq_u8(sl.val[0], mf), vandq_u8(vshlq_n_u8(sh, 4), m30));
            iscales.val[1] = vorrq_u8(vandq_u8(sl.val[1], mf), vandq_u8(sh, m30));
            int32x4_t isum[nrc_y] = {};
            for (int is = 0; is < 2; ++is) {
                auto iscales16_1 = vmovl_s8(vget_low_s8(iscales.val[is]));
                auto iscales16_2 = vmovl_s8(vget_high_s8(iscales.val[is]));
                scales.val[0] = vmovl_s16(vget_low_s16(iscales16_1));
                scales.val[1] = vmovl_s16(vget_high_s16(iscales16_1));
                scales.val[2] = vmovl_s16(vget_low_s16(iscales16_2));
                scales.val[3] = vmovl_s16(vget_high_s16(iscales16_2));
                for (int ib = 0; ib < 4; ++ib) {
                    auto lbits = vld1q_u8_x4(iq5[ibl].qs + 256*is + 64*ib);
                    auto hbits2 = vld1q_u8(iq5[ibl].qh + 64*is + 16*ib);
                    auto hbits1 = vshlq_n_u8(hbits2, 4);
                    prepare_q4_k_quants(mf, lbits, qx);
                    qx[0] = vorrq_u8(qx[0], vandq_u8(m10, hbits1));
                    qx[1] = vorrq_u8(qx[1], vandq_u8(m10, hbits2));
                    qx[2] = vorrq_u8(qx[2], vandq_u8(m10, vshrq_n_u8(hbits1, 2)));
                    qx[3] = vorrq_u8(qx[3], vandq_u8(m10, vshrq_n_u8(hbits2, 2)));
                    qx[4] = vorrq_u8(qx[4], vandq_u8(m10, vshrq_n_u8(hbits1, 1)));
                    qx[5] = vorrq_u8(qx[5], vandq_u8(m10, vshrq_n_u8(hbits2, 1)));
                    qx[6] = vorrq_u8(qx[6], vandq_u8(m10, vshrq_n_u8(hbits1, 3)));
                    qx[7] = vorrq_u8(qx[7], vandq_u8(m10, vshrq_n_u8(hbits2, 3)));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8_x2(q8.y[iy][ibl].qs+128*is+32*ib);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales.val[ib], sumi);
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
void mul_mat_q6_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto mf = vdupq_n_u8(0x0f);
    auto m3 = vdupq_n_u8(0x30);
    auto m32 = vdupq_n_s8(-32);
    int nbl = n / QK_K;
    int8x16_t qx[4];
    float32x4_t acc[nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_q6_k_r4 * iq6 = (const block_q6_k_r4 *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq6[ibl].d));
            int32x4_t isum[nrc_y] = {};
            for (int is = 0; is < 2; ++is) {
                for (int ib = 0; ib < 4; ++ib) {
                    auto lbits = vld1q_u8_x4(iq6[ibl].ql + 256*is + 64*ib);
                    auto hbits = vld1q_u8(iq6[ibl].qh + 128*is + 32*ib);
                    auto iscales = vmovl_s8(vld1_s8(iq6[ibl].scales + 32*is + 8*ib));
                    auto scales = vmovl_s16(vget_low_s16(iscales));
                    qx[0] = vaddq_s8(m32, vorrq_u8(vandq_u8 (lbits.val[0], mf), vandq_u8(m3, vshlq_n_u8(hbits, 4))));
                    qx[1] = vaddq_s8(m32, vorrq_u8(vandq_u8 (lbits.val[2], mf), vandq_u8(m3, hbits)));
                    qx[2] = vaddq_s8(m32, vorrq_u8(vshrq_n_u8(lbits.val[0], 4), vandq_u8(m3, vshlq_n_u8(hbits, 2))));
                    qx[3] = vaddq_s8(m32, vorrq_u8(vshrq_n_u8(lbits.val[2], 4), vandq_u8(m3, vshrq_n_u8(hbits, 2))));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8(q8.y[iy][ibl].qs+128*is+32*ib);
                        auto sumi = interleaved_dotq(qx, y);
                        isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                    }
                    scales = vmovl_s16(vget_high_s16(iscales));
                    hbits = vld1q_u8(iq6[ibl].qh + 128*is + 32*ib + 16);
                    qx[0] = vaddq_s8(m32, vorrq_u8(vandq_u8 (lbits.val[1], mf), vandq_u8(m3, vshlq_n_u8(hbits, 4))));
                    qx[1] = vaddq_s8(m32, vorrq_u8(vandq_u8 (lbits.val[3], mf), vandq_u8(m3, hbits)));
                    qx[2] = vaddq_s8(m32, vorrq_u8(vshrq_n_u8(lbits.val[1], 4), vandq_u8(m3, vshlq_n_u8(hbits, 2))));
                    qx[3] = vaddq_s8(m32, vorrq_u8(vshrq_n_u8(lbits.val[3], 4), vandq_u8(m3, vshrq_n_u8(hbits, 2))));
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
void mul_mat_q8_k_r8_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
    float32x4_t acc[2*nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_q8_k_r8 * iq8 = (const block_q8_k_r8 *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            auto d4l = vcvt_f32_f16(vld1_f16((const float16_t *)iq8[ibl].d+0));
            auto d4h = vcvt_f32_f16(vld1_f16((const float16_t *)iq8[ibl].d+4));
            int32x4_t isum[2*nrc_y] = {};
            for (int ib = 0; ib < QK_K/16; ++ib) {
                auto q1 = vld1q_s8_x4(iq8[ibl].qs + 128*ib +  0);
                auto q2 = vld1q_s8_x4(iq8[ibl].qs + 128*ib + 64);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = vld1q_s8(q8.y[iy][ibl].qs+16*ib);
                    isum[2*iy+0] = vdotq_laneq_s32(isum[2*iy+0], q1.val[0], y, 0);
                    isum[2*iy+1] = vdotq_laneq_s32(isum[2*iy+1], q1.val[1], y, 0);
                    isum[2*iy+0] = vdotq_laneq_s32(isum[2*iy+0], q1.val[2], y, 1);
                    isum[2*iy+1] = vdotq_laneq_s32(isum[2*iy+1], q1.val[3], y, 1);
                    isum[2*iy+0] = vdotq_laneq_s32(isum[2*iy+0], q2.val[0], y, 2);
                    isum[2*iy+1] = vdotq_laneq_s32(isum[2*iy+1], q2.val[1], y, 2);
                    isum[2*iy+0] = vdotq_laneq_s32(isum[2*iy+0], q2.val[2], y, 3);
                    isum[2*iy+1] = vdotq_laneq_s32(isum[2*iy+1], q2.val[3], y, 3);
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto d8 = vdupq_n_f32(q8.scale(iy, ibl));
                const float * bsum = (const float *)q8.y[iy][ibl].bsums;
                auto m8 = vdupq_n_f32(-128.f*bsum[0]);
                acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], vmulq_f32(d4l, d8), vcvtq_f32_s32(isum[2*iy+0]));
                acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], vmulq_f32(d4h, d8), vcvtq_f32_s32(isum[2*iy+1]));
                acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], d4l, m8);
                acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], d4l, m8);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix+0, iy, acc[2*iy+0]);
            info.store(ix+4, iy, acc[2*iy+1]);
            acc[2*iy+0] = acc[2*iy+1] = vdupq_n_f32(0.f);
        }
    }
}

template <int nrc_y>
void mul_mat_iq4_xs_r8_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = vdupq_n_u8(0xf);
    auto m3 = vdupq_n_u8(0x30);
    auto m32 = vdupq_n_s8(-32);
    auto values = vld1q_s8(iq4k_values);
    int nbl = n / QK_K;
    int8x16_t qx[8];
    int8x16x4_t iscales;
    int32x4x2_t scales;
    float32x4_t acc[2*nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const block_iq4_xs_r8 * iq4 = (const block_iq4_xs_r8 *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) {
            auto d4_f16 = vld1q_f16((const float16_t *)iq4[ibl].d);
            auto d4l = vcvt_f32_f16(vget_low_f16 (d4_f16));
            auto d4h = vcvt_f32_f16(vget_high_f16(d4_f16));
            auto sl = vld1q_u8_x2(iq4[ibl].scales_l);
            auto sh = vld1q_u8(iq4[ibl].scales_h);
            iscales.val[0] = vaddq_s8(vorrq_u8(vandq_u8(sl.val[0], m4), vandq_u8(vshlq_n_u8(sh, 4), m3)), m32);
            iscales.val[1] = vaddq_s8(vorrq_u8(vandq_u8(sl.val[1], m4), vandq_u8(vshlq_n_u8(sh, 2), m3)), m32);
            iscales.val[2] = vaddq_s8(vorrq_u8(vshrq_n_u8(sl.val[0], 4), vandq_u8(sh, m3)), m32);
            iscales.val[3] = vaddq_s8(vorrq_u8(vshrq_n_u8(sl.val[1], 4), vandq_u8(vshrq_n_u8(sh, 2), m3)), m32);
            int32x4_t isum[nrc_y] = {};
            for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
                auto iscales16_1 = vmovl_s8(vget_low_s8(iscales.val[ib64]));
                auto iscales16_2 = vmovl_s8(vget_high_s8(iscales.val[ib64]));
                scales.val[0] = vmovl_s16(vget_low_s16(iscales16_1));
                scales.val[1] = vmovl_s16(vget_low_s16(iscales16_2));
                for (int l = 0; l < 2; ++l) {
                    uint8x16x2_t bits;
                    bits.val[0] = vld1q_u8(iq4[ibl].qs + 256*ib64 + 128*l);
                    bits.val[1] = vld1q_u8(iq4[ibl].qs + 256*ib64 + 128*l + 32);
                    prepare_iq4_nl_quants_r8(values, m4, bits, qx+0);
                    bits.val[0] = vld1q_u8(iq4[ibl].qs + 256*ib64 + 128*l + 64);
                    bits.val[1] = vld1q_u8(iq4[ibl].qs + 256*ib64 + 128*l + 96);
                    prepare_iq4_nl_quants_r8(values, m4, bits, qx+4);
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8_x2(q8.y[iy][ibl].qs+64*ib64+32*l);
                        auto sumi = vdupq_n_s32(0);
                        sumi = vdotq_laneq_s32(sumi, qx[0], y.val[0], 0);
                        sumi = vdotq_laneq_s32(sumi, qx[1], y.val[0], 1);
                        sumi = vdotq_laneq_s32(sumi, qx[2], y.val[0], 2);
                        sumi = vdotq_laneq_s32(sumi, qx[3], y.val[0], 3);
                        sumi = vdotq_laneq_s32(sumi, qx[4], y.val[1], 0);
                        sumi = vdotq_laneq_s32(sumi, qx[5], y.val[1], 1);
                        sumi = vdotq_laneq_s32(sumi, qx[6], y.val[1], 2);
                        sumi = vdotq_laneq_s32(sumi, qx[7], y.val[1], 3);
                        isum[iy] = vmlaq_s32(isum[iy], sumi, scales.val[l]);
                    }
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto d8 = vdupq_n_f32(q8.scale(iy, ibl));
                acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], vmulq_f32(d4l, d8), vcvtq_f32_s32(isum[iy]));
                isum[iy] = vdupq_n_s32(0);
            }
            for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
                auto iscales16_1 = vmovl_s8(vget_low_s8(iscales.val[ib64]));
                auto iscales16_2 = vmovl_s8(vget_high_s8(iscales.val[ib64]));
                scales.val[0] = vmovl_s16(vget_high_s16(iscales16_1));
                scales.val[1] = vmovl_s16(vget_high_s16(iscales16_2));
                for (int l = 0; l < 2; ++l) {
                    uint8x16x2_t bits;
                    bits.val[0] = vld1q_u8(iq4[ibl].qs + 256*ib64 + 128*l + 16);
                    bits.val[1] = vld1q_u8(iq4[ibl].qs + 256*ib64 + 128*l + 48);
                    prepare_iq4_nl_quants_r8(values, m4, bits, qx+0);
                    bits.val[0] = vld1q_u8(iq4[ibl].qs + 256*ib64 + 128*l + 80);
                    bits.val[1] = vld1q_u8(iq4[ibl].qs + 256*ib64 + 128*l +112);
                    prepare_iq4_nl_quants_r8(values, m4, bits, qx+4);
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8_x2(q8.y[iy][ibl].qs+64*ib64+32*l);
                        auto sumi = vdupq_n_s32(0);
                        sumi = vdotq_laneq_s32(sumi, qx[0], y.val[0], 0);
                        sumi = vdotq_laneq_s32(sumi, qx[1], y.val[0], 1);
                        sumi = vdotq_laneq_s32(sumi, qx[2], y.val[0], 2);
                        sumi = vdotq_laneq_s32(sumi, qx[3], y.val[0], 3);
                        sumi = vdotq_laneq_s32(sumi, qx[4], y.val[1], 0);
                        sumi = vdotq_laneq_s32(sumi, qx[5], y.val[1], 1);
                        sumi = vdotq_laneq_s32(sumi, qx[6], y.val[1], 2);
                        sumi = vdotq_laneq_s32(sumi, qx[7], y.val[1], 3);
                        isum[iy] = vmlaq_s32(isum[iy], sumi, scales.val[l]);
                    }
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto d8 = vdupq_n_f32(q8.scale(iy, ibl));
                acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], vmulq_f32(d4h, d8), vcvtq_f32_s32(isum[iy]));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix+0, iy, acc[2*iy+0]);
            info.store(ix+4, iy, acc[2*iy+1]);
            acc[2*iy+0] = acc[2*iy+1] = vdupq_n_f32(0.f);
        }
    }
}

static void mul_mat_q8_KV_q8_KV_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%32 == 0);
    int32x4_t acc[4] = {};
    auto dptr = (const float *)info.src1_row(0);
    const float dy = dptr[0];
    auto q8y = (const int8_t *)(dptr + 2);
    for (int ix = 0; ix < nrc_x; ++ix) {
        auto dx  = (const float *)((const char *)vx + ix*bx);
        auto q8x = (const int8_t *)(dx + 2);
        for (int i = 0; i < n/64; ++i) {
            auto qx = vld1q_s8_x4(q8x + 64*i);
            for (int j = 0; j < 4; ++j) {
                acc[j] = ggml_vdotq_s32(acc[j], qx.val[j], vld1q_s8(q8y + 64*i + 16*j));
            }
        }
        if (int i = 2*(n/64); i < n/32) {
            auto qx = vld1q_s8_x2(q8x + 32*i);
            for (int j = 0; j < 2; ++j) {
                acc[j] = ggml_vdotq_s32(acc[j], qx.val[j], vld1q_s8(q8y + 32*i + 16*j));
            }
        }
        acc[0] = vaddq_s32(acc[0], acc[1]);
        acc[2] = vaddq_s32(acc[2], acc[3]);
        acc[0] = vaddq_s32(acc[0], acc[2]);
        info.store(ix, 0, dx[0]*dy*vaddvq_s32(acc[0]));
        acc[0] = acc[1] = acc[2] = acc[3] = vdupq_n_s32(0);
    }
}

template <int nrc_y>
static void mul_mat_q8_KV_q8_KV(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    GGML_ASSERT(n%16 == 0);
    int8x16_t qx[4];
    int32x4_t acc[nrc_y] = {};
    float dy[nrc_y];
    const int8_t * q8y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        auto dptr = (const float *)info.src1_row(iy);
        dy[iy] = dptr[0];
        q8y[iy] = (const int8_t *)(dptr + 2);
    }
    const int8_t * q8x[4];
    float dx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        for (int kx = 0; kx < 4; ++kx) {
            auto dptr = (const float *)((const char *)vx + (ix+kx)*bx);
            dx[kx] = dptr[0];
            q8x[kx] = (const int8_t *)(dptr + 2);
        }
        for (int i = 0; i < n/16; ++i) {
            for (int kx = 0; kx < 4; ++kx) qx[kx] = vld1q_s8(q8x[kx] + 16*i);
            auto row01 = vtrnq_s32(qx[0], qx[1]);
            auto row23 = vtrnq_s32(qx[2], qx[3]);
            qx[0] = vtrn1q_s64(row01.val[0], row23.val[0]);
            qx[1] = vtrn1q_s64(row01.val[1], row23.val[1]);
            qx[2] = vtrn2q_s64(row01.val[0], row23.val[0]);
            qx[3] = vtrn2q_s64(row01.val[1], row23.val[1]);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = vld1q_s8(q8y[iy] + 16*i);
                acc[iy] = vdotq_laneq_s32(acc[iy], qx[0], y, 0);
                acc[iy] = vdotq_laneq_s32(acc[iy], qx[1], y, 1);
                acc[iy] = vdotq_laneq_s32(acc[iy], qx[2], y, 2);
                acc[iy] = vdotq_laneq_s32(acc[iy], qx[3], y, 3);
            }
        }
        auto scales_x = vld1q_f32(dx);
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto scale = vmulq_f32(scales_x, vdupq_n_f32(dy[iy]));
            info.store(ix, iy, vmulq_f32(scale, vcvtq_f32_s32(acc[iy])));
            acc[iy] = vdupq_n_s32(0);
        }
    }
}

template <int nrc_y>
void mul_mat_q8_KV_r8_q8_KV(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    int32x4_t acc[2*nrc_y] = {};
    float dy[nrc_y];
    const int8_t * q8y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        auto dptr = (const float *)info.src1_row(iy);
        dy[iy] = dptr[0];
        q8y[iy] = (const int8_t *)(dptr + 2);
    }
    for (int ix = 0; ix < nrc_x; ix += 8) {
        const float * dptr = (const float *)((const char *)vx + ix*bx);
        auto q8x = (const int8_t *)(dptr + 8);
        for (int ib = 0; ib < n/16; ++ib) {
            auto q1 = vld1q_s8_x4(q8x + 128*ib +  0);
            auto q2 = vld1q_s8_x4(q8x + 128*ib + 64);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = vld1q_s8(q8y[iy]+16*ib);
                acc[2*iy+0] = vdotq_laneq_s32(acc[2*iy+0], q1.val[0], y, 0);
                acc[2*iy+1] = vdotq_laneq_s32(acc[2*iy+1], q1.val[1], y, 0);
                acc[2*iy+0] = vdotq_laneq_s32(acc[2*iy+0], q1.val[2], y, 1);
                acc[2*iy+1] = vdotq_laneq_s32(acc[2*iy+1], q1.val[3], y, 1);
                acc[2*iy+0] = vdotq_laneq_s32(acc[2*iy+0], q2.val[0], y, 2);
                acc[2*iy+1] = vdotq_laneq_s32(acc[2*iy+1], q2.val[1], y, 2);
                acc[2*iy+0] = vdotq_laneq_s32(acc[2*iy+0], q2.val[2], y, 3);
                acc[2*iy+1] = vdotq_laneq_s32(acc[2*iy+1], q2.val[3], y, 3);
            }
        }
        auto scale1_x = vld1q_f32(dptr+0);
        auto scale2_x = vld1q_f32(dptr+4);
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto scale_y = vdupq_n_f32(dy[iy]);
            auto scale1 = vmulq_f32(scale1_x, scale_y);
            auto scale2 = vmulq_f32(scale2_x, scale_y);
            info.store(ix+0, iy, vmulq_f32(scale1, vcvtq_f32_s32(acc[2*iy+0])));
            info.store(ix+4, iy, vmulq_f32(scale2, vcvtq_f32_s32(acc[2*iy+1])));
            acc[2*iy+0] = acc[2*iy+1] = vdupq_n_s32(0.f);
        }
    }
}

}

bool iqk_set_kernels_kquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, [[maybe_unused]] mul_mat_t& func16) {

    auto etypeA = ggml_type(typeA);
    auto expected_type_B = etypeA == GGML_TYPE_IQ4_XS_R8 || etypeA == GGML_TYPE_Q4_K_R4 || etypeA == GGML_TYPE_Q5_K_R4 ? GGML_TYPE_Q8_K32
                         : etypeA == GGML_TYPE_Q8_K_R8 ? GGML_TYPE_Q8_KR8
                         : etypeA == GGML_TYPE_Q8_KV || etypeA == GGML_TYPE_Q8_KV_R8 ? GGML_TYPE_Q8_KV
                         : GGML_TYPE_Q8_K;

    if (ne00%QK_K != 0 || ggml_type(typeB) != expected_type_B) {
        return false;
    }

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_Q2_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerQ2K, kernels)
            break;
        case GGML_TYPE_Q3_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerQ3K, kernels)
            break;
        case GGML_TYPE_Q4_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerQ4K, kernels)
            break;
        case GGML_TYPE_Q5_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerQ5K, kernels)
            break;
        case GGML_TYPE_Q6_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerQ6K, kernels)
            break;
        case GGML_TYPE_IQ4_XS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ4XS, kernels)
            break;
        case GGML_TYPE_Q2_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q2_k_r4_q8_k, kernels)
            break;
        case GGML_TYPE_Q3_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q3_k_r4_q8_k, kernels)
            break;
        case GGML_TYPE_Q4_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q4_k_r4_q8_k, kernels)
            break;
        case GGML_TYPE_Q5_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q5_k_r4_q8_k, kernels)
            break;
        case GGML_TYPE_Q6_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q6_k_r4_q8_k, kernels)
            break;
        case GGML_TYPE_IQ4_XS_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_xs_r8_q8_k, kernels)
            break;
        case GGML_TYPE_Q8_K_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_k_r8_q8_k, kernels)
            break;
        case GGML_TYPE_Q8_KV:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_KV_q8_KV, kernels)
            kernels[0] = mul_mat_q8_KV_q8_KV_1;
            func16 = mul_mat_q8_KV_q8_KV<16>;
            break;
        case GGML_TYPE_Q8_KV_R8:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_KV_r8_q8_KV, kernels);
            break;
        default:
            return false;
    }

    return true;

}

#endif

namespace {

#ifdef __AVX2__
template <int nrc_y>
static void mul_mat_q8_KV_q8_KV_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%32 == 0);
    if (nrc_y == 1 && nrc_x == 1) { 
        auto dx = (const float *)vx;
        auto dy = (const float *)info.src1_row(0);
#ifdef HAVE_FANCY_SIMD
        auto sy = (const int32_t *)(dy + 1);
        auto x = (const int8_t *)(dx + 2);
        auto y = (const int8_t *)(dy + 2);
        auto isum = _mm512_setzero_si512();
        for (int i = 0; i < n/64; ++i) {
            auto qx = _mm512_loadu_si512((const __m512i *)x + i);
            auto qy = _mm512_loadu_si512((const __m512i *)y + i);
            isum = _mm512_dpbusd_epi32(isum, _mm512_add_epi8(qx, _mm512_set1_epi8(127)), qy); 
        }
        auto isum256 = _mm256_add_epi32(_mm512_castsi512_si256(isum), _mm512_extracti32x8_epi32(isum, 1)); 
        for (int i = 2*(n/64); i < n/32; ++i) {
            auto qx = _mm256_loadu_si256((const __m256i *)x + i);
            auto qy = _mm256_loadu_si256((const __m256i *)y + i);
            isum256 = _mm256_dpbusd_epi32(isum256, _mm256_add_epi8(qx, _mm256_set1_epi8(127)), qy); 
        }
        info.store(0, 0, dx[0]*dy[0]*(hsum_i32_8(isum256) - 127*sy[0]));
#else
        auto x = (const int8_t *)(dx + 2);
        auto y = (const int8_t *)(dy + 2);
        auto isum = _mm256_setzero_si256();
        for (int i = 0; i < n/32; ++i) {
            auto qx = _mm256_loadu_si256((const __m256i *)x + i);
            auto qy = _mm256_loadu_si256((const __m256i *)y + i);
            auto dot = _mm256_maddubs_epi16(_mm256_sign_epi8(qx, qx), _mm256_sign_epi8(qy, qx));
            isum = _mm256_add_epi32(isum, _mm256_madd_epi16(_mm256_set1_epi16(1), dot));
        }
        info.store(0, 0, dx[0]*dy[0]*hsum_i32_8(isum));
#endif
        return;
    }
    __m256i qx[2];
    __m256i acc[2*nrc_y] = {};
    float   dy[nrc_y];
#ifdef HAVE_FANCY_SIMD
    int32_t sy[nrc_y];
#else
    __m256i sx[2];
    auto m1 = _mm256_set1_epi16(1);
#endif
    const int8_t * q8y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        auto dptr = (const float *)info.src1_row(iy);
        dy[iy] = dptr[0];
#ifdef HAVE_FANCY_SIMD
        auto iptr = (const int32_t *)(dptr+1);
        sy[iy] = -127*iptr[0];
#endif
        q8y[iy] = (const int8_t *)(dptr + 2);
    }
    for (int ix = 0; ix < nrc_x; ++ix) {
        auto dx  = (const float *)((const char *)vx + ix*bx);
        auto q8x = (const int8_t *)(dx + 2);
        for (int i = 0; i < n/64; ++i) {
            for (int j = 0; j < 2; ++j) {
#ifdef HAVE_FANCY_SIMD
                qx[j] = _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)q8x + 2*i + j), _mm256_set1_epi8(127));
#else
                qx[j] = _mm256_loadu_si256((const __m256i *)q8x + 2*i + j);
                sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
#endif
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                for (int j = 0; j < 2; ++j) {
#ifdef HAVE_FANCY_SIMD
                    acc[2*iy+j] = _mm256_dpbusd_epi32(acc[2*iy+j], qx[j], _mm256_loadu_si256((const __m256i *)q8y[iy] + 2*i + j));
#else
                    auto dot = _mm256_maddubs_epi16(sx[j], _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)q8y[iy] + 2*i + j), qx[j]));
                    acc[2*iy+j] = _mm256_add_epi32(acc[2*iy+j], _mm256_madd_epi16(m1, dot));
#endif
                }
            }
        }
        if (int i = 2*(n/64); i < n/32) {
#ifdef HAVE_FANCY_SIMD
            qx[0] = _mm256_add_epi8(_mm256_loadu_si256((const __m256i *)q8x + i), _mm256_set1_epi8(127));
#else
            qx[0] = _mm256_loadu_si256((const __m256i *)q8x + i);
            sx[0] = _mm256_sign_epi8(qx[0], qx[0]);
#endif
            for (int iy = 0; iy < nrc_y; ++iy) {
#ifdef HAVE_FANCY_SIMD
                acc[2*iy] = _mm256_dpbusd_epi32(acc[2*iy], qx[0], _mm256_loadu_si256((const __m256i *)q8y[iy] + i));
#else
                auto dot = _mm256_maddubs_epi16(sx[0], _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)q8y[iy] + i), qx[0]));
                acc[2*iy] = _mm256_add_epi32(acc[2*iy], _mm256_madd_epi16(m1, dot));
#endif
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sumi = hsum_i32_8(_mm256_add_epi32(acc[2*iy], acc[2*iy+1]));
#ifdef HAVE_FANCY_SIMD
            info.store(ix, iy, dx[0]*dy[iy]*(sumi+sy[iy]));
#else
            info.store(ix, iy, dx[0]*dy[iy]*sumi);
#endif
            acc[2*iy] = acc[2*iy+1] = _mm256_setzero_si256();
        }
    }
}

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_q8_KV_q8_KV_8(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    GGML_ASSERT(n%32 == 0);
    __m512i qx[4];
    __m512i acc[nrc_y <= 4 ? 2*nrc_y : nrc_y] = {};
    float dy[nrc_y];
    int32_t sy[nrc_y];
    const int8_t * q8y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        auto dptr = (const float *)info.src1_row(iy);
        dy[iy] = dptr[0];
        auto iptr = (const int32_t *)(dptr + 1);
        sy[iy] = -64*iptr[0];
        q8y[iy] = (const int8_t *)(dptr + 2);
    }
    const int8_t * q8x[8];
    float dx[8];
    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int kx = 0; kx < 8; ++kx) {
            auto dptr = (const float *)((const char *)vx + (ix+kx)*bx);
            dx[kx] = dptr[0];
            q8x[kx] = (const int8_t *)(dptr + 2);
        }
        for (int i = 0; i < n/32; ++i) {
            for (int kx = 0; kx < 4; ++kx) {
                qx[kx] = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)q8x[kx+0] + i)),
                                                                   _mm256_loadu_si256((const __m256i *)q8x[kx+4] + i), 1);
            }
            auto t0 = _mm512_unpacklo_epi32(qx[0], qx[1]);
            auto t1 = _mm512_unpacklo_epi32(qx[2], qx[3]);
            auto t2 = _mm512_unpackhi_epi32(qx[0], qx[1]);
            auto t3 = _mm512_unpackhi_epi32(qx[2], qx[3]);
            qx[0] = _mm512_xor_si512(_mm512_unpacklo_epi64(t0, t1), _mm512_set1_epi8(-128));
            qx[1] = _mm512_xor_si512(_mm512_unpackhi_epi64(t0, t1), _mm512_set1_epi8(-128));
            qx[2] = _mm512_xor_si512(_mm512_unpacklo_epi64(t2, t3), _mm512_set1_epi8(-128));
            qx[3] = _mm512_xor_si512(_mm512_unpackhi_epi64(t2, t3), _mm512_set1_epi8(-128));
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y256 = _mm256_loadu_si256((const __m256i *)q8y[iy] + i);
                auto y = _mm512_inserti32x8(_mm512_castsi256_si512(y256), y256, 1);
                if constexpr (nrc_y <= 4) {
                    acc[2*iy+0] = _mm512_dpbusd_epi32(acc[2*iy+0], qx[0], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
                    acc[2*iy+1] = _mm512_dpbusd_epi32(acc[2*iy+1], qx[1], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
                    acc[2*iy+0] = _mm512_dpbusd_epi32(acc[2*iy+0], qx[2], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
                    acc[2*iy+1] = _mm512_dpbusd_epi32(acc[2*iy+1], qx[3], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
                } else {
                    acc[iy] = _mm512_dpbusd_epi32(acc[iy], qx[0], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
                    acc[iy] = _mm512_dpbusd_epi32(acc[iy], qx[1], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
                    acc[iy] = _mm512_dpbusd_epi32(acc[iy], qx[2], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
                    acc[iy] = _mm512_dpbusd_epi32(acc[iy], qx[3], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
                }
            }
        }
        auto scales_x = _mm256_loadu_ps(dx);
        for (int iy = 0; iy < nrc_y; ++iy) {
            if constexpr (nrc_y <= 4) {
                auto ss = _mm512_add_epi32(_mm512_add_epi32(acc[2*iy+0], acc[2*iy+1]), _mm512_set1_epi32(sy[iy]));
                auto sum1 = _mm_add_epi32(_mm512_extracti32x4_epi32(ss, 0), _mm512_extracti32x4_epi32(ss, 1));
                auto sum2 = _mm_add_epi32(_mm512_extracti32x4_epi32(ss, 2), _mm512_extracti32x4_epi32(ss, 3));
                auto scale = _mm256_mul_ps(scales_x, _mm256_set1_ps(dy[iy]));
                info.store(ix+0, iy, _mm_mul_ps(_mm256_castps256_ps128(scale),   _mm_cvtepi32_ps(sum1)));
                info.store(ix+4, iy, _mm_mul_ps(_mm256_extractf128_ps(scale, 1), _mm_cvtepi32_ps(sum2)));
                acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_si512();
            } else {
                acc[iy] = _mm512_add_epi32(acc[iy], _mm512_set1_epi32(sy[iy]));
                auto sum1 = _mm_add_epi32(_mm512_extracti32x4_epi32(acc[iy], 0), _mm512_extracti32x4_epi32(acc[iy], 1));
                auto sum2 = _mm_add_epi32(_mm512_extracti32x4_epi32(acc[iy], 2), _mm512_extracti32x4_epi32(acc[iy], 3));
                auto scale = _mm256_mul_ps(scales_x, _mm256_set1_ps(dy[iy]));
                info.store(ix+0, iy, _mm_mul_ps(_mm256_castps256_ps128(scale),   _mm_cvtepi32_ps(sum1)));
                info.store(ix+4, iy, _mm_mul_ps(_mm256_extractf128_ps(scale, 1), _mm_cvtepi32_ps(sum2)));
                acc[iy] = _mm512_setzero_si512();
            }
        }
    }
}
#endif
#endif

template <int k_step>
inline std::pair<mul_mat_t, int> mul_mat_kernel([[maybe_unused]] int D, int int_typeA, int nq) {
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
    if (typeA == GGML_TYPE_Q8_KV) {
#ifdef __aarch64__
        if (nq%16 == 0) return std::make_pair(mul_mat_q8_KV_q8_KV<16>, 16);
        if (nq == 1) return std::make_pair(mul_mat_q8_KV_q8_KV_1, 1);
        MAKE_FUNCS_ONLY_NRC(mul_mat_q8_KV_q8_KV, nq);
#else
        if (nq == 1) return std::make_pair(mul_mat_q8_KV_q8_KV_1<1>, 1);
#ifdef HAVE_FANCY_SIMD
        if (D%32 == 0 && k_step%8 == 0) {
            if (nq%16 == 0) return std::make_pair(mul_mat_q8_KV_q8_KV_8<16>, 16);
            MAKE_FUNCS_ONLY_NRC(mul_mat_q8_KV_q8_KV_8, nq);
        } else {
            if (nq%16 == 0) return std::make_pair(mul_mat_q8_KV_q8_KV<16>, 16);
        }
#endif
        MAKE_FUNCS_ONLY_NRC(mul_mat_q8_KV_q8_KV, nq);
#endif
    }
    else if (typeA == GGML_TYPE_Q8_KV_R8) {
        MAKE_FUNCS_ONLY_NRC(mul_mat_q8_KV_r8_q8_KV, nq);
    }
    GGML_ABORT("Fatal error");
}

inline std::pair<mul_mat_t, int> mul_mat_kernel(int D, int int_typeA, int nq, int k_step) {
    switch (k_step) {
        case  32: return mul_mat_kernel< 32>(D, int_typeA, nq);
        case  64: return mul_mat_kernel< 64>(D, int_typeA, nq);
        case 128: return mul_mat_kernel<128>(D, int_typeA, nq);
        default: GGML_ABORT("Fatal error");
    }
}

}

void iqk_gemm_q8kv_fa(int D, int nq, int type_k, const char * k, size_t stride_k, DataInfo& info, int k_step) {
    auto [mul_mat, nrc_q] = mul_mat_kernel(D, type_k, nq, k_step);
    for (int iq = 0; iq < nq/nrc_q; ++iq) {
        mul_mat(D, k, stride_k, info, k_step);
        info.cur_y += nrc_q;
    }
    int iq = nrc_q*(nq/nrc_q);
    if (iq < nq) {
        auto [mul_mat1, nrc_q1] = mul_mat_kernel(D, type_k, nq - iq, k_step);
        GGML_ASSERT(nrc_q1 == nq - iq);
        mul_mat1(D, k, stride_k, info, k_step);
    }
}

#endif
