#include "iqk_gemm_iquants.h"

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#ifdef __x86_64__

namespace {

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

// TODO: find the bug that causes this to be called without HAVE_FANCY_SIMD, which triggers
//       writing 4 vvalues into scales, which is of size 2.
inline void set_scales_8_iq(int j, const __m256i& all_scales, __m256i * scales) {
//#ifdef HAVE_FANCY_SIMD
    auto shuffle = j == 0 ? _mm256_set_epi64x(0x0302030203020302, 0x0100010001000100, 0x0302030203020302, 0x0100010001000100)
                          : _mm256_set_epi64x(0x0b0a0b0a0b0a0b0a, 0x0908090809080908, 0x0b0a0b0a0b0a0b0a, 0x0908090809080908);
    scales[0] = _mm256_shuffle_epi8(all_scales, shuffle);
    scales[1] = _mm256_shuffle_epi8(all_scales, _mm256_add_epi8(shuffle, _mm256_set1_epi8(4)));
//#else
//    set_scales_8(all_scales, j, scales);
//#endif
}

inline void set_scales_16_iq(const __m256i& all_scales, __m256i * scales) {
#ifdef HAVE_FANCY_SIMD
    auto shuffle = _mm256_set_epi64x(0x0706070607060706, 0x0302030203020302, 0x0504050405040504, 0x0100010001000100);
    scales[0] = _mm256_shuffle_epi8(all_scales, shuffle);
    scales[1] = _mm256_shuffle_epi8(all_scales, _mm256_add_epi8(shuffle, _mm256_set1_epi8(8)));
#else
    set_scales_16(all_scales, scales);
#endif
}

struct SimpleBits {
    __m256i values[4];
};

struct EvenSignHelper {
#if defined HAVE_FANCY_SIMD && defined __AVX512VPOPCNTDQ__
    union sbits_t {
        __m128i vec;
        __mmask32 mask[4];
    };
    IQK_ALWAYS_INLINE void sign_2_values(__m256i aux, __m256i * values) const {
        aux = _mm256_and_si256(_mm256_srlv_epi32(aux, shifts), mask);
        auto pcnt = _mm256_popcnt_epi32(aux);
        sbits_t sbits;
        sbits.vec = _mm256_cvtepi32_epi8(_mm256_or_si256(aux, _mm256_slli_epi32(_mm256_and_si256(pcnt, mone), 7)));
        values[0] = _mm256_mask_sub_epi8(values[0], sbits.mask[0], _mm256_setzero_si256(), values[0]);
        values[1] = _mm256_mask_sub_epi8(values[1], sbits.mask[1], _mm256_setzero_si256(), values[1]);
        //auto sign_bits = _mm256_cvtepi32_epi8(_mm256_or_si256(aux, _mm256_slli_epi32(_mm256_and_si256(pcnt, mone), 7)));
        //const __mmask32 * m32 = (const __mmask32 *)&sign_bits;
        //values[0] = _mm256_mask_sub_epi8(values[0], m32[0], _mm256_setzero_si256(), values[0]);
        //values[1] = _mm256_mask_sub_epi8(values[1], m32[1], _mm256_setzero_si256(), values[1]);
    }
    const __m256i shifts = _mm256_set_epi32(21, 14, 7, 0, 21, 14, 7, 0);
    const __m256i mask   = _mm256_set1_epi32(127);
    const __m256i mone   = _mm256_set1_epi32(1);
#endif
    inline void sign_value(uint32_t aux32, __m256i& value) const {
        auto signs = _mm256_set_epi64x(keven_signs[(aux32 >> 21) & 127], keven_signs[(aux32 >> 14) & 127],
                                       keven_signs[(aux32 >>  7) & 127], keven_signs[(aux32 >>  0) & 127]);
        value = _mm256_sign_epi8(value, signs);
    }
};

struct SignHelper {
    inline __m256i make_signs(uint32_t sign_bits) const {
        auto aux256 = _mm256_set1_epi32(sign_bits);
        aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
        return _mm256_or_si256(_mm256_cmpeq_epi8(aux256, mask2), mone);
    }
//    inline __m256i make_signs(const uint16_t * sign_bits) const {
//#ifdef HAVE_FANCY_SIMD
//#else
//        return make_signs(sign_bits[0] | (sign_bits[1] << 16));
//#endif
//    }
    inline __m256i sign_value(const uint16_t * sign_bits, const __m256i& value) const {
#ifdef HAVE_FANCY_SIMD
        const __mmask32 * mask = (const __mmask32 *)sign_bits;
        return _mm256_mask_sub_epi8(value, mask[0], _mm256_setzero_si256(), value);
#else
        return _mm256_sign_epi8(value, make_signs(sign_bits[0] | (sign_bits[1] << 16)));
#endif
    }
    IQK_ALWAYS_INLINE void sign_4_values(const uint16_t * sign_bits, __m256i * values) const {
        // Somehow the FANCY_SIMD version has become 50% slower for TG???
#ifdef z_HAVE_FANCY_SIMD
        //__mmask32 mask[4]; std::memcpy(mask, sign_bits, 4*sizeof(__mmask32));
        const __mmask32 * mask = (const __mmask32 *)sign_bits;
        values[0] = _mm256_mask_sub_epi8(values[0], mask[0], _mm256_setzero_si256(), values[0]);
        values[1] = _mm256_mask_sub_epi8(values[1], mask[1], _mm256_setzero_si256(), values[1]);
        values[2] = _mm256_mask_sub_epi8(values[2], mask[2], _mm256_setzero_si256(), values[2]);
        values[3] = _mm256_mask_sub_epi8(values[3], mask[3], _mm256_setzero_si256(), values[3]);
#else
        auto s128 = _mm_loadu_si128((const __m128i *)sign_bits);
        auto s256 = MM256_SET_M128I(s128, s128);
        __m256i aux256;
        auto shuffle = mask1;
        auto step = _mm256_set1_epi8(4);
        aux256 = _mm256_and_si256(_mm256_shuffle_epi8(s256, shuffle), mask2); shuffle = _mm256_add_epi8(shuffle, step);
        values[0] = _mm256_sign_epi8(values[0], _mm256_or_si256(_mm256_cmpeq_epi8(aux256, mask2), mone));
        aux256 = _mm256_and_si256(_mm256_shuffle_epi8(s256, shuffle), mask2); shuffle = _mm256_add_epi8(shuffle, step);
        values[1] = _mm256_sign_epi8(values[1], _mm256_or_si256(_mm256_cmpeq_epi8(aux256, mask2), mone));
        aux256 = _mm256_and_si256(_mm256_shuffle_epi8(s256, shuffle), mask2); shuffle = _mm256_add_epi8(shuffle, step);
        values[2] = _mm256_sign_epi8(values[2], _mm256_or_si256(_mm256_cmpeq_epi8(aux256, mask2), mone));
        aux256 = _mm256_and_si256(_mm256_shuffle_epi8(s256, shuffle), mask2); shuffle = _mm256_add_epi8(shuffle, step);
        values[3] = _mm256_sign_epi8(values[3], _mm256_or_si256(_mm256_cmpeq_epi8(aux256, mask2), mone));
#endif
    }
    const __m256i mask1 = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    const __m256i mask2 = _mm256_set1_epi64x(0x8040201008040201ull);
    const __m256i mone  = _mm256_set1_epi8(1);
};

//        for (int i = 0; i < nb; ++i) {
//
//            __m256i sumi[nrc_y], all_scales;
//            //for (int iy = 0; iy < nrc_y; ++iy) sumi[iy] = _mm256_setzero_si256();
//            __m256i mins;
//            float dmin = deq.new_block(i, &all_scales, mins);
//            for (int iy = 0; iy < nrc_y; ++iy) {
//                auto bsums = q8.load_bsums(iy, i);
//                auto prod  = _mm256_madd_epi16(mins, bsums);
//                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(dmin*q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accd[iy]);
//            }
//
//            for (int j = 0; j < QK_K/128; ++j) {
//                deq.prepare(i, j);
//                set_scales_8(&all_scales, j, scales);
//                //multiply_add_iq(deq.bits, scales, j, i, q8, sumi);
//                multiply_add(deq.bits, scales, j, i, q8, sumi);
//            }
//            for (int iy = 0; iy < nrc_y; ++iy) {
//                const __m256 vd = _mm256_set1_ps(deq.d*q8.scale(iy, i));
//                accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
//            }
//        }
//
//        for (int iy = 0; iy < nrc_y; ++iy) {
//            info.store(ix, iy, hsum_float_8(accd[iy]));
//        }
//    }

struct DequantizerIQ2XXS final : public BaseDequantizer<block_iq2_xxs> {
    DequantizerIQ2XXS(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}

    constexpr static int num_blocks = 8;

    union Data {
        __m256i vec;
        uint32_t val[8];
    };

    inline __m128i load_scales(int i) {
        d = 0.125f * GGML_FP16_TO_FP32(x[i].d);
        const uint16_t * a16 = (const uint16_t *)x[i].qs;
        auto scales = _mm_srli_epi16(_mm_set_epi16(a16[31], a16[27], a16[23], a16[19], a16[15], a16[11], a16[7], a16[3]), 12);
        return _mm_or_si128(_mm_slli_epi16(scales, 1), _mm_set1_epi16(1));
    }

    inline void new_block(int i, __m256i * scales) {
        auto sc16 = load_scales(i);
        scales[0] = MM256_SET_M128I(sc16, sc16);
    }
    inline void new_block_f(int i, __m256 * scales) {
        auto sc16 = load_scales(i);
        auto scf  = _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sc16)));
        auto scf_l = _mm256_castps256_ps128(scf);
        auto scf_h = _mm256_extractf128_ps(scf, 1);
        scales[0] = _mm256_set_m128(scf_l, scf_l);
        scales[1] = _mm256_set_m128(scf_h, scf_h);
        scales[2] = _mm256_mul_ps(scf, _mm256_set1_ps(-minv));
    }

    inline float new_block(int i, __m256i * scales, __m256i& mins) {
        auto sc16 = load_scales(i);
        mins = scb.shuffle(sc16);
        scales[0] = MM256_SET_M128I(sc16, sc16);
        return -d*minv;
    }

    inline static void make4(const uint32_t * aux32, __m256i * values) {
        const uint8_t * aux8 = (const uint8_t *)aux32;
        values[0] = _mm256_set_epi64x(iq2xxs_grid[aux8[ 3]], iq2xxs_grid[aux8[ 2]], iq2xxs_grid[aux8[ 1]], iq2xxs_grid[aux8[ 0]]);
        values[1] = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[ 9]], iq2xxs_grid[aux8[ 8]]);
        values[2] = _mm256_set_epi64x(iq2xxs_grid[aux8[19]], iq2xxs_grid[aux8[18]], iq2xxs_grid[aux8[17]], iq2xxs_grid[aux8[16]]);
        values[3] = _mm256_set_epi64x(iq2xxs_grid[aux8[27]], iq2xxs_grid[aux8[26]], iq2xxs_grid[aux8[25]], iq2xxs_grid[aux8[24]]);
    }

    IQK_ALWAYS_INLINE void sign_values(const uint32_t * aux32, __m256i * values) const {
#if defined HAVE_FANCY_SIMD && defined __AVX512VPOPCNTDQ__
        esh.sign_2_values(MM256_SET_M128I(_mm_set1_epi32(aux32[3]), _mm_set1_epi32(aux32[1])), values+0);
        esh.sign_2_values(MM256_SET_M128I(_mm_set1_epi32(aux32[7]), _mm_set1_epi32(aux32[5])), values+2);
#else
        esh.sign_value(aux32[1], values[0]);
        esh.sign_value(aux32[3], values[1]);
        esh.sign_value(aux32[5], values[2]);
        esh.sign_value(aux32[7], values[3]);
#endif
    }
    inline void make4_signed(const uint32_t * aux32, const __m256i& min_value, __m256i * values) const {
        make4(aux32, values);
        sign_values(aux32, values);
        for (int k = 0; k < 4; ++k) values[k] = _mm256_add_epi8(values[k], min_value);
    }
    inline void make4(const uint32_t * aux32, __m256i * values, __m256i * q8) const {
        make4(aux32, values);
        sign_values(aux32, q8);
    }
    inline void prepare(int i, int j) {
        Data data; data.vec = _mm256_loadu_si256((const __m256i *)x[i].qs + j);
        make4_signed(data.val, min_value, bits.values);
    }
    inline void prepare(int i, int j, const Q8<1>& q8, __m256i * q8_quants) {
        for (int k = 0; k < 4; ++k) q8_quants[k] = q8.load_quants(0, i, 4*j+k);
        Data data; data.vec = _mm256_loadu_si256((const __m256i *)x[i].qs + j); 
        make4(data.val, bits.values, q8_quants);
    }

    constexpr static int minv = 43;
    SimpleBits bits;
    Scales8KBase scb;
    EvenSignHelper esh;
    const __m256i min_value = _mm256_set1_epi8(minv);
    const __m256i shuffle = _mm256_set_epi32(7, 5, 3, 1, 7, 5, 3, 1);
};

struct DequantizerIQ2XS final : public BaseDequantizer<block_iq2_xs> {
    DequantizerIQ2XS(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}

    constexpr static int num_blocks = 16;

    inline __m256i load_scales(int i) {
        d = 0.125f * GGML_FP16_TO_FP32(x[i].d);
        auto tmp = _mm_loadl_epi64((const __m128i *)x[i].scales);
        auto all = _mm_and_si128(_mm_unpacklo_epi8(tmp, _mm_srli_epi16(tmp, 4)), _mm_set1_epi8(0xf));
        auto scales8 = _mm_or_si128(_mm_slli_epi16(all, 1), _mm_set1_epi8(1));
        return _mm256_cvtepi8_epi16(scales8);
    }
    inline static void prepare_scales(const __m256i& all, __m256i * scales) {
        auto scales_l = _mm256_castsi256_si128(all);
        auto scales_h = _mm256_extractf128_si256(all, 1);
        scales[0] = MM256_SET_M128I(scales_l, scales_l);
        scales[1] = MM256_SET_M128I(scales_h, scales_h);
    }

    inline void new_block(int i, __m256i * scales) {
        prepare_scales(load_scales(i), scales);
    }
    inline float new_block(int i, __m256i * scales, __m256i& mins) {
        mins = load_scales(i);
        prepare_scales(mins, scales);
        return -d*minv;
    }

    struct Helper {
        const __m256i mone = _mm256_set1_epi8(1);
        const __m256i mask = _mm256_set1_epi64x(0x8040201008040201);
        //const __m256i bhelper = _mm256_set_epi64x(0x8000008000808000, 0x0080800080000080, 0x8000008000808000, 0x0080800080000080);
        const __m256i bhelper = load_bhelper();
        const __m256i shuff1  = _mm256_set_epi64x(0x0606060606060606, 0x0404040404040404, 0x0202020202020202, 0x0000000000000000);
        const __m256i shuff2  = _mm256_set_epi64x(0x0e0e0e0e0e0e0e0e, 0x0c0c0c0c0c0c0c0c, 0x0a0a0a0a0a0a0a0a, 0x0808080808080808);
        static __m256i load_bhelper() {
            static const uint8_t k_bit_helper[32] = {
                0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
                0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
            };
            return _mm256_loadu_si256((const __m256i*)k_bit_helper);
        }
    };

    union index_t {
        __m256i vec;
        uint16_t val[8];
    };

    inline static void make4(const __m256i& data, const __m256i& mask, __m256i * values) {
        index_t idx;
        idx.vec = _mm256_and_si256(data, mask);
        values[0] = _mm256_set_epi64x(iq2xs_grid[idx.val[ 3]], iq2xs_grid[idx.val[ 2]], iq2xs_grid[idx.val[ 1]], iq2xs_grid[idx.val[ 0]]);
        values[1] = _mm256_set_epi64x(iq2xs_grid[idx.val[ 7]], iq2xs_grid[idx.val[ 6]], iq2xs_grid[idx.val[ 5]], iq2xs_grid[idx.val[ 4]]);
        values[2] = _mm256_set_epi64x(iq2xs_grid[idx.val[11]], iq2xs_grid[idx.val[10]], iq2xs_grid[idx.val[ 9]], iq2xs_grid[idx.val[ 8]]);
        values[3] = _mm256_set_epi64x(iq2xs_grid[idx.val[15]], iq2xs_grid[idx.val[14]], iq2xs_grid[idx.val[13]], iq2xs_grid[idx.val[12]]);
    }
    inline static void sign_value(const __m256i& sign_bits, const __m256i& shuffle, const __m256i& mask,
            const __m256i& mone, __m256i& value) {
        auto signs = _mm256_shuffle_epi8(sign_bits, shuffle);
        signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, mask), mask);
        value = _mm256_sign_epi8(value, _mm256_or_si256(signs, mone));
    }
    inline void sign_values(const __m256i& data, __m256i * values) const {
#if defined HAVE_FANCY_SIMD && defined __AVX512VPOPCNTDQ__
        auto partial_bits = _mm256_cvtepi16_epi8(_mm256_srli_epi16(data,  9));
        auto pcnt = _mm_popcnt_epi8(partial_bits);
        auto full_bits = _mm_or_si128(partial_bits, _mm_slli_epi16(_mm_and_si128(pcnt, _mm_set1_epi8(1)), 7));
        const __mmask32 * m32 = (const __mmask32 *)&full_bits;
        auto zero = _mm256_setzero_si256();
        values[0] = _mm256_mask_sub_epi8(values[0], m32[0], zero, values[0]);
        values[1] = _mm256_mask_sub_epi8(values[1], m32[1], zero, values[1]);
        values[2] = _mm256_mask_sub_epi8(values[2], m32[2], zero, values[2]);
        values[3] = _mm256_mask_sub_epi8(values[3], m32[3], zero, values[3]);
#else
        auto psb1 = _mm256_srli_epi16(data,  9);
        auto psb2 = _mm256_srli_epi16(data, 13);
        auto psbc = _mm256_xor_si256(psb1, psb2);
        auto oddb = _mm256_shuffle_epi8(helper.bhelper, psbc);
        auto full = _mm256_or_si256(psb1, oddb);
        auto full_l = _mm256_castsi256_si128(full);
        auto full_h = _mm256_extractf128_si256(full, 1);
        auto full_1 = MM256_SET_M128I(full_l, full_l);
        auto full_2 = MM256_SET_M128I(full_h, full_h);
        sign_value(full_1, helper.shuff1, helper.mask, helper.mone, values[0]);
        sign_value(full_1, helper.shuff2, helper.mask, helper.mone, values[1]);
        sign_value(full_2, helper.shuff1, helper.mask, helper.mone, values[2]);
        sign_value(full_2, helper.shuff2, helper.mask, helper.mone, values[3]);
#endif
    }
    inline void make4_signed(const uint16_t * qs, const __m256i& m511,
            const __m256i& min_value, __m256i * values) const {
        auto q2 = _mm256_loadu_si256((const __m256i *)qs);
        make4(q2, m511, values);
        sign_values(q2, values);
        for (int k = 0; k < 4; ++k) values[k] = _mm256_add_epi8(values[k], min_value);
    }
    inline void make4(const uint16_t * qs, const __m256i& m511, __m256i * values, __m256i * q8) const {
        auto q2 = _mm256_loadu_si256((const __m256i *)qs);
        make4(q2, m511, values);
        sign_values(q2, q8);
    }

    inline void prepare(int i, int j) {
        make4_signed(x[i].qs + 16*j, idx_mask, min_value, bits.values);
    }
    inline void prepare(int i, int j, const Q8<1>& q8, __m256i * q8_quants) {
        for (int k = 0; k < 4; ++k) q8_quants[k] = q8.load_quants(0, i, 4*j+k);
        make4(x[i].qs + 16*j, idx_mask, bits.values, q8_quants);
    }

    constexpr static int minv = 43;

    SimpleBits bits;
#if !(defined HAVE_FANCY_SIMD && defined __AVX512VPOPCNTDQ__)
    Helper helper;
#endif
    const __m256i idx_mask  = _mm256_set1_epi16(511);
    const __m256i min_value = _mm256_set1_epi8(minv);

};

struct DequantizerIQ2S final : public BaseDequantizer<block_iq2_s> {
    DequantizerIQ2S(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}

    constexpr static int num_blocks = 16;

    inline __m256i load_scales(int i) {
        d = 0.125f * GGML_FP16_TO_FP32(x[i].d);
        auto tmp = _mm_loadl_epi64((const __m128i *)x[i].scales);
        auto all = _mm_and_si128(_mm_unpacklo_epi8(tmp, _mm_srli_epi16(tmp, 4)), _mm_set1_epi8(0xf));
        auto scales8 = _mm_or_si128(_mm_slli_epi16(all, 1), _mm_set1_epi8(1));
        return _mm256_cvtepi8_epi16(scales8);
    }
    inline static void prepare_scales(const __m256i& all, __m256i * scales) {
        auto scales_l = _mm256_castsi256_si128(all);
        auto scales_h = _mm256_extractf128_si256(all, 1);
        scales[0] = MM256_SET_M128I(scales_l, scales_l);
        scales[1] = MM256_SET_M128I(scales_h, scales_h);
    }

    inline void new_block(int i, __m256i * scales) {
        prepare_scales(load_scales(i), scales);
    }
    inline float new_block(int i, __m256i * scales, __m256i& mins) {
        mins = load_scales(i);
        prepare_scales(mins, scales);
        return -d*minv;
    }

    union index_t {
        __m256i vec;
        uint32_t val[8];
    };

    inline static void make2(const uint8_t * qs, const uint8_t * qh, const __m256i& idx_shift, const __m256i& idx_mask, __m256i * values) {
        auto idx_l = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)qs));
        auto idx_h = MM256_SET_M128I(_mm_set1_epi32(qh[1]), _mm_set1_epi32(qh[0]));
        index_t idx;
        idx.vec = _mm256_or_si256(idx_l, _mm256_and_si256(_mm256_sllv_epi32(idx_h, idx_shift), idx_mask));
        values[0] = _mm256_set_epi64x(iq2s_grid[idx.val[3]], iq2s_grid[idx.val[2]], iq2s_grid[idx.val[1]], iq2s_grid[idx.val[0]]);
        values[1] = _mm256_set_epi64x(iq2s_grid[idx.val[7]], iq2s_grid[idx.val[6]], iq2s_grid[idx.val[5]], iq2s_grid[idx.val[4]]);
    }
    inline static void make2_signed(const SignHelper& sh, const uint8_t * qs, const uint8_t * qh, const uint16_t * sidx,
            const __m256i& idx_shift, const __m256i& idx_mask, const __m256i& min_value, __m256i * values) {
        make2(qs, qh, idx_shift, idx_mask, values);
        values[0] = _mm256_add_epi8(sh.sign_value(sidx+0, values[0]), min_value);
        values[1] = _mm256_add_epi8(sh.sign_value(sidx+2, values[1]), min_value);
    }

    inline void prepare(int i, int j) {
        auto qs = x[i].qs + 16*j;
        auto qh = x[i].qh +  4*j;
        const uint16_t * signs = (const uint16_t *)(x[i].qs + QK_K/8) + 8*j;
        make2_signed(sh, qs+0, qh+0, signs+0, idx_shift, idx_mask, min_value, bits.values+0);
        make2_signed(sh, qs+8, qh+2, signs+4, idx_shift, idx_mask, min_value, bits.values+2);
    }
    inline void prepare(int i, int j, const Q8<1>& q8, __m256i * q8_quants) {
        auto qs = x[i].qs + 16*j;
        auto qh = x[i].qh +  4*j;
        const uint16_t * signs = (const uint16_t *)(x[i].qs + QK_K/8) + 8*j;
        make2(qs+0, qh+0, idx_shift, idx_mask, bits.values+0);
        make2(qs+8, qh+2, idx_shift, idx_mask, bits.values+2);
        q8_quants[0] = _mm256_sign_epi8(q8.load_quants(0, i, 4*j+0), sh.make_signs(signs[0] | (signs[1] << 16)));
        q8_quants[1] = _mm256_sign_epi8(q8.load_quants(0, i, 4*j+1), sh.make_signs(signs[2] | (signs[3] << 16)));
        q8_quants[2] = _mm256_sign_epi8(q8.load_quants(0, i, 4*j+2), sh.make_signs(signs[4] | (signs[5] << 16)));
        q8_quants[3] = _mm256_sign_epi8(q8.load_quants(0, i, 4*j+3), sh.make_signs(signs[6] | (signs[7] << 16)));
    }

    constexpr static int minv = 43;

    SimpleBits bits;
    SignHelper sh;
    const __m256i idx_shift = _mm256_set_epi32(2, 4, 6, 8, 2, 4, 6, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(0x300);
    const __m256i min_value = _mm256_set1_epi8(minv);

};

struct DequantizerIQ3XXS final : public BaseDequantizer<block_iq3_xxs> {
    DequantizerIQ3XXS(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}

    constexpr static int num_blocks = 8;

    inline __m128i prepare_scales(int i) {
        d = 0.25f * GGML_FP16_TO_FP32(x[i].d);
        auto tmp = _mm256_loadu_si256((const __m256i *)(x[i].qs + QK_K/4));
        auto scales32 = _mm256_srli_epi32(tmp, 28);
        scales32 = _mm256_or_si256(_mm256_slli_epi32(scales32, 1), _mm256_set1_epi32(1));
        return _mm_packs_epi32(_mm256_castsi256_si128(scales32), _mm256_extractf128_si256(scales32, 1));
    }

    inline void new_block(int i, __m256i * scales) {
        auto scales16 = prepare_scales(i);
        scales[0] = MM256_SET_M128I(scales16, scales16);
    }
    inline void new_block_f(int i, __m256 * scales) {
        auto sc16 = prepare_scales(i);
        auto scf  = _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sc16)));
        auto scf_l = _mm256_castps256_ps128(scf);
        auto scf_h = _mm256_extractf128_ps(scf, 1);
        scales[0] = _mm256_set_m128(scf_l, scf_l);
        scales[1] = _mm256_set_m128(scf_h, scf_h);
        scales[2] = _mm256_mul_ps(scf, _mm256_set1_ps(-minv));
    }
    inline float new_block(int i, __m256i * scales, __m256i& mins) {
        auto scales16 = prepare_scales(i);
        mins = scb.shuffle(scales16);
        scales[0] = MM256_SET_M128I(scales16, scales16);
        return -d*minv;
    }

    inline static __m256i make_quants(const uint8_t * qs) {
        return _mm256_set_epi32(iq3xxs_grid[qs[7]], iq3xxs_grid[qs[6]], iq3xxs_grid[qs[5]], iq3xxs_grid[qs[4]],
                                iq3xxs_grid[qs[3]], iq3xxs_grid[qs[2]], iq3xxs_grid[qs[1]], iq3xxs_grid[qs[0]]);
    }
    inline static void make4_unsigned(const uint8_t * qs, __m256i * values) {
        values[0] = make_quants(qs+ 0);
        values[1] = make_quants(qs+ 8);
        values[2] = make_quants(qs+16);
        values[3] = make_quants(qs+24);
    }

    IQK_ALWAYS_INLINE void sign_2_values(const uint16_t * signs, __m256i * values) const {
#if defined HAVE_FANCY_SIMD && defined __AVX512VPOPCNTDQ__
        esh.sign_2_values(MM256_SET_M128I(_mm_set1_epi32(signs[2] | (signs[3] << 16)), _mm_set1_epi32(signs[0] | (signs[1] << 16))), values);
#else
        esh.sign_value(signs[0] | (signs[1] << 16), values[0]);
        esh.sign_value(signs[2] | (signs[3] << 16), values[1]);
#endif
    }

    inline void prepare(int i, int j) {
        auto qs = x[i].qs + 32*j;
        const uint16_t * signs = (const uint16_t *)(x[i].qs + QK_K/4) + 8*j;
        make4_unsigned(qs, bits.values);
        sign_2_values(signs+0, bits.values+0);
        sign_2_values(signs+4, bits.values+2);
        for (int k = 0; k < 4; ++k) bits.values[k] = _mm256_add_epi32(bits.values[k], min_value);
    }
    inline void prepare(int i, int j, const Q8<1>& q8, __m256i * q8_quants) {
        for (int k = 0; k < 4; ++k) q8_quants[k] = q8.load_quants(0, i, 4*j+k);
        auto qs = x[i].qs + 32*j;
        const uint16_t * signs = (const uint16_t *)(x[i].qs + QK_K/4) + 8*j;
        make4_unsigned(qs, bits.values);
        sign_2_values(signs+0, q8_quants+0);
        sign_2_values(signs+4, q8_quants+2);
    }

    constexpr static int minv = 64;

    SimpleBits bits;
    Scales8KBase scb;
    EvenSignHelper esh;
    const __m256i min_value = _mm256_set1_epi8(minv);

};

#ifdef z_HAVE_FANCY_SIMD
// Strangely enough, the following implementation makes PP ~6% slower and TG ~6% faster
// compared to the vanilla AVX2 version below.
struct IndexHelperIQ3S {
    union index_t {
        __m256i  vec;
        uint16_t val[16];
    };
    inline void make2(const uint8_t * qs, const uint8_t * qh, __m256i * values) const {
        auto idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs));
        const __mmask16 * m16 = (const __mmask16 *)qh;
        index_t idx;
        idx.vec = _mm256_mask_add_epi16(idx_l, m16[0], idx_l, offset);
        values[0] = _mm256_set_epi32(iq3s_grid[idx.val[ 7]], iq3s_grid[idx.val[ 6]], iq3s_grid[idx.val[ 5]], iq3s_grid[idx.val[ 4]],
                                     iq3s_grid[idx.val[ 3]], iq3s_grid[idx.val[ 2]], iq3s_grid[idx.val[ 1]], iq3s_grid[idx.val[ 0]]);
        values[1] = _mm256_set_epi32(iq3s_grid[idx.val[15]], iq3s_grid[idx.val[14]], iq3s_grid[idx.val[13]], iq3s_grid[idx.val[12]],
                                     iq3s_grid[idx.val[11]], iq3s_grid[idx.val[10]], iq3s_grid[idx.val[ 9]], iq3s_grid[idx.val[ 8]]);
    }
    const __m256i offset = _mm256_set1_epi16(256);
};
#else
struct IndexHelperIQ3S {
    union index_t {
        __m256i  vec;
        uint32_t val[8];
    };
    inline void make2(const uint8_t * qs, const uint8_t * qh, __m256i * values) const {
        index_t idx;
        auto idx_l = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)qs));
        auto idx_h = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[0]), idx_shift), idx_mask);
        idx.vec = _mm256_or_si256(idx_h, idx_l);
        values[0] = _mm256_set_epi32(iq3s_grid[idx.val[7]], iq3s_grid[idx.val[6]], iq3s_grid[idx.val[5]], iq3s_grid[idx.val[4]],
                                     iq3s_grid[idx.val[3]], iq3s_grid[idx.val[2]], iq3s_grid[idx.val[1]], iq3s_grid[idx.val[0]]);
        idx_l = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qs+8)));
        idx_h = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[1]), idx_shift), idx_mask);
        idx.vec = _mm256_or_si256(idx_h, idx_l);
        values[1] = _mm256_set_epi32(iq3s_grid[idx.val[7]], iq3s_grid[idx.val[6]], iq3s_grid[idx.val[5]], iq3s_grid[idx.val[4]],
                                     iq3s_grid[idx.val[3]], iq3s_grid[idx.val[2]], iq3s_grid[idx.val[1]], iq3s_grid[idx.val[0]]);
    }
    const __m256i idx_mask = _mm256_set1_epi32(256);
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
};
#endif

struct DequantizerIQ3S final : public BaseDequantizer<block_iq3_s> {
    DequantizerIQ3S(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}

    constexpr static int num_blocks = 8;

    inline __m128i make_scales(int i, float& dd) const {
        dd = GGML_FP16_TO_FP32(x[i].d);
        uint32_t aux32[2];
        std::memcpy(aux32, x[i].scales, 4);
        aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
        aux32[0] &= 0x0f0f0f0f;
        auto scales8 = _mm_shuffle_epi8(_mm_loadl_epi64((const __m128i *)aux32), _mm_set1_epi64x(0x0703060205010400));
        auto scales16 = _mm256_castsi256_si128(_mm256_cvtepi8_epi16(scales8));
        return _mm_or_si128(_mm_slli_epi16(scales16, 1), _mm_set1_epi16(1));
    }
    inline void new_block(int i, __m256i * scales) {
        auto scales16 = make_scales(i, d);
        scales[0] = MM256_SET_M128I(scales16, scales16);
    }
    inline void new_block_f(int i, __m256 * scales) {
        auto sc16 = make_scales(i, d);
        auto scf  = _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sc16)));
        auto scf_l = _mm256_castps256_ps128(scf);
        auto scf_h = _mm256_extractf128_ps(scf, 1);
        scales[0] = _mm256_set_m128(scf_l, scf_l);
        scales[1] = _mm256_set_m128(scf_h, scf_h);
        scales[2] = _mm256_mul_ps(scf, _mm256_set1_ps(-minv));
    }
    inline float new_block(int i, __m256i * scales, __m256i& mins) {
        auto scales16 = make_scales(i, d);
        mins = scb.shuffle(scales16);
        scales[0] = MM256_SET_M128I(scales16, scales16);
        return -minv*d;
    }

    inline void prepare(int i, int j) {
        prepare_unsigned(i, j);
        sh.sign_4_values((const uint16_t *)x[i].signs + 8*j, bits.values);
        for (int k = 0; k < 4; ++k) bits.values[k] = _mm256_add_epi8(bits.values[k], min_value);
    }
    inline void prepare(int i, int j, const Q8<1>& q8, __m256i * q8_quants) {
        prepare_unsigned(i, j);
        for (int k = 0; k < 4; ++k) q8_quants[k] = q8.load_quants(0, i, 4*j+k);
        sh.sign_4_values((const uint16_t *)x[i].signs + 8*j, q8_quants);
    }

    inline void prepare_unsigned(int i, int j) {
        auto qs = x[i].qs + 32*j;
        auto qh = x[i].qh +  4*j;
        helper.make2(qs+ 0, qh+0, bits.values+0);
        helper.make2(qs+16, qh+2, bits.values+2);
    }

    constexpr static int minv = 16;

    SimpleBits bits;
    SignHelper sh;
    Scales8KBase scb;
    IndexHelperIQ3S helper;
    const __m256i min_value = _mm256_set1_epi8(minv);

};

template <typename Bits>
inline void multiply_add_1(int j, const Bits& bits, const __m256i * scales, const __m256i * q8, __m256i * sumi) {
    if (j == 0) {
#ifdef HAVE_FANCY_SIMD
        auto p1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), bits.values[0], q8[0]);
        auto p2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), bits.values[1], q8[1]);
        auto p3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), bits.values[2], q8[2]);
        auto p4 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), bits.values[3], q8[3]);
        sumi[0] = _mm256_dpwssd_epi32(_mm256_setzero_si256(), scales[0], _mm256_packs_epi32(p1, p2));
        sumi[1] = _mm256_dpwssd_epi32(_mm256_setzero_si256(), scales[1], _mm256_packs_epi32(p3, p4));
#else
        const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(bits.values[0], q8[0]));
        const __m256i p2 = _mm256_madd_epi16(scales[1], _mm256_maddubs_epi16(bits.values[1], q8[1]));
        const __m256i p3 = _mm256_madd_epi16(scales[2], _mm256_maddubs_epi16(bits.values[2], q8[2]));
        const __m256i p4 = _mm256_madd_epi16(scales[3], _mm256_maddubs_epi16(bits.values[3], q8[3]));
        sumi[0] = _mm256_add_epi32(p1, p3);
        sumi[1] = _mm256_add_epi32(p2, p4);
#endif
    } else {
#ifdef HAVE_FANCY_SIMD
        auto p1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), bits.values[0], q8[0]);
        auto p2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), bits.values[1], q8[1]);
        auto p3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), bits.values[2], q8[2]);
        auto p4 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), bits.values[3], q8[3]);
        sumi[0] = _mm256_dpwssd_epi32(sumi[0], scales[0], _mm256_packs_epi32(p1, p2));
        sumi[1] = _mm256_dpwssd_epi32(sumi[1], scales[1], _mm256_packs_epi32(p3, p4));
#else
        const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(bits.values[0], q8[0]));
        const __m256i p2 = _mm256_madd_epi16(scales[1], _mm256_maddubs_epi16(bits.values[1], q8[1]));
        const __m256i p3 = _mm256_madd_epi16(scales[2], _mm256_maddubs_epi16(bits.values[2], q8[2]));
        const __m256i p4 = _mm256_madd_epi16(scales[3], _mm256_maddubs_epi16(bits.values[3], q8[3]));
        sumi[0] = _mm256_add_epi32(sumi[0], _mm256_add_epi32(p1, p3));
        sumi[1] = _mm256_add_epi32(sumi[1], _mm256_add_epi32(p2, p4));
#endif
    }
}

template <typename Dequantizer>
static void mul_mat_qX_K_q8_K_IQ_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const int nb = n / QK_K;
    Q8<1> q8(info);
    Dequantizer deq(vx, bx);
    __m256i scales[2];
    __m256i q8_quants[4];
    for (int ix = 0; ix < nrc_x; ++ix) {

        __m256 accd = _mm256_setzero_ps();
        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            __m256i sumi[2], all_scales[Dequantizer::num_blocks/8];
            deq.new_block(i, all_scales);

            for (int j = 0; j < QK_K/128; ++j) {
                deq.prepare(i, j, q8, q8_quants);
                if constexpr (Dequantizer::num_blocks == 8) {
                    set_scales_8_iq(j, all_scales[0], scales);
                } else {
                    set_scales_16_iq(all_scales[j], scales);
                }
                multiply_add_1(j, deq.bits, scales, q8_quants, sumi);
            }
            accd = _mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8.scale(0, i)), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi[0], sumi[1])), accd);
        }

        info.store(ix, 0, hsum_float_8(accd));
    }
}

// So, if I uncomment this function and the call to it in mul_mat_qX_K_q8_K_IQ_N() below,
// PP performance improves by ~2-3% (when we have __AVX512VNNI__ and __AVX512VL__).
// But TG performance for iq3_xs drops by 35%. Seriously? I mean, c'mon,
// what does the compilation of mul_mat_qX_K_q8_K_IQ_1 (which gets invoked during TG)
// have to do with the compilation of mul_mat_qX_K_q8_K_IQ_N (invoked during PP)?
//template <typename Q8, typename Bits>
//inline void multiply_add_iq(const Bits& bits, const __m256i * scales, int j, int i, const Q8& q8, __m256i * sumi) {
//#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
//    for (int iy = 0; iy < Q8::nrc_y; ++iy) {
//        sumi[iy] = _mm256_dpwssd_epi32(sumi[iy], scales[0], _mm256_maddubs_epi16(bits.values[0], q8.load_quants(iy, i, 4*j+0)));
//        sumi[iy] = _mm256_dpwssd_epi32(sumi[iy], scales[1], _mm256_maddubs_epi16(bits.values[1], q8.load_quants(iy, i, 4*j+1)));
//        sumi[iy] = _mm256_dpwssd_epi32(sumi[iy], scales[2], _mm256_maddubs_epi16(bits.values[2], q8.load_quants(iy, i, 4*j+2)));
//        sumi[iy] = _mm256_dpwssd_epi32(sumi[iy], scales[3], _mm256_maddubs_epi16(bits.values[3], q8.load_quants(iy, i, 4*j+3)));
//    }
//#else
//    for (int iy = 0; iy < Q8::nrc_y; ++iy) {
//        const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(bits.values[0], q8.load_quants(iy, i, 4*j+0)));
//        const __m256i p2 = _mm256_madd_epi16(scales[1], _mm256_maddubs_epi16(bits.values[1], q8.load_quants(iy, i, 4*j+1)));
//        const __m256i p3 = _mm256_madd_epi16(scales[2], _mm256_maddubs_epi16(bits.values[2], q8.load_quants(iy, i, 4*j+2)));
//        const __m256i p4 = _mm256_madd_epi16(scales[3], _mm256_maddubs_epi16(bits.values[3], q8.load_quants(iy, i, 4*j+3)));
//        sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p1, p3));
//        sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p2, p4));
//    }
//#endif
//}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_K_q8_K_IQ_N(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const int nb = n / QK_K;
    Q8<nrc_y> q8(info);
    Dequantizer deq(vx, bx);
    __m256i scales[4];
    __m256  accd[nrc_y];

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            __m256i sumi[nrc_y], all_scales[Dequantizer::num_blocks/8];
            //for (int iy = 0; iy < nrc_y; ++iy) sumi[iy] = _mm256_setzero_si256();
            __m256i mins;
            float dmin = deq.new_block(i, all_scales, mins);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto bsums = q8.load_bsums(iy, i);
                auto prod  = _mm256_madd_epi16(mins, bsums);
                accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(dmin*q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accd[iy]);
            }

            for (int j = 0; j < QK_K/128; ++j) {
                deq.prepare(i, j);
                if constexpr (Dequantizer::num_blocks == 8) {
                    set_scales_8(all_scales[0], j, scales);
                } else {
                    set_scales_16(all_scales[j], scales);
                }
                //multiply_add_iq(deq.bits, scales, j, i, q8, sumi);
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
static void mul_mat_qX_K_q8_2_IQ_N(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    static_assert(Dequantizer::num_blocks == 8);
    const int nb = n / QK_K;
    Q8<nrc_y, block_q8_2_x4> q8(info);
    Dequantizer deq(vx, bx);
    __m256  scales[3];
    __m256  accd[nrc_y];
    __m256i sumi[4];

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            deq.new_block_f(i, scales);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto my1 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][2*i+0].d + 4)));
                auto my2 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(q8.y[iy][2*i+1].d + 4)));
                auto my  = _mm256_castsi256_ps(_mm256_slli_epi32(MM256_SET_M128I(my2, my1), 16));
                accd[iy] = _mm256_fmadd_ps(scales[2], my, accd[iy]);
            }

            for (int j = 0; j < QK_K/128; ++j) {
                deq.prepare(i, j);
                auto& values = deq.bits.values;
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto qs = q8.y[iy][2*i+j].qs;
#ifdef HAVE_FANCY_SIMD
                    sumi[0] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), values[0], _mm256_loadu_si256((const __m256i*)qs+0));
                    sumi[1] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), values[1], _mm256_loadu_si256((const __m256i*)qs+1));
                    sumi[2] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), values[2], _mm256_loadu_si256((const __m256i*)qs+2));
                    sumi[3] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), values[3], _mm256_loadu_si256((const __m256i*)qs+3));
#else
                    sumi[0] = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(values[0], _mm256_loadu_si256((const __m256i*)qs+0)));
                    sumi[1] = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(values[1], _mm256_loadu_si256((const __m256i*)qs+1)));
                    sumi[2] = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(values[2], _mm256_loadu_si256((const __m256i*)qs+2)));
                    sumi[3] = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(values[3], _mm256_loadu_si256((const __m256i*)qs+3)));
#endif
                    sumi[0] = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi[0], sumi[1]), _mm256_unpackhi_epi32(sumi[0], sumi[1]));
                    sumi[2] = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi[2], sumi[3]), _mm256_unpackhi_epi32(sumi[2], sumi[3]));
                    sumi[0] = _mm256_add_epi32(_mm256_unpacklo_epi64(sumi[0], sumi[2]), _mm256_unpackhi_epi64(sumi[0], sumi[2]));
                    auto d4 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)q8.y[iy][2*i+j].d)), 16));
                    auto dy = _mm256_set_m128(d4, d4);
                    accd[iy] = _mm256_fmadd_ps(_mm256_mul_ps(scales[j], dy), _mm256_cvtepi32_ps(sumi[0]), accd[iy]);
                }
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }
    }
}

template <int nrc_y>
static void mul_mat_iq2_xxs_q8_2_IQ_N(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const int nb = n / QK_K;
    __m256  scales[2];
    __m256  accd[nrc_y];
    __m256i sumi[4];
    __m256i xv[4];
    EvenSignHelper esh;

    for (int ix = 0; ix < nrc_x; ++ix) {

        for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_ps();

        const block_iq2_xxs * x = (const block_iq2_xxs *)((const char *)vx + ix*bx);

        for (int i = 0; i < nb; ++i) {
            const float d = GGML_FP16_TO_FP32(x[i].d)*0.125f;
            const uint16_t * a16 = x[i].qs;
            auto sc16 = _mm_set_epi16(a16[31], a16[27], a16[23], a16[19], a16[15], a16[11], a16[7], a16[3]);
            sc16 = _mm_or_si128(_mm_slli_epi16(_mm_srli_epi16(sc16, 12), 1), _mm_set1_epi16(1));
            auto sc32 = _mm256_cvtepi16_epi32(sc16);
            auto all_scales = _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sc32));
            auto all_mins = _mm256_mul_ps(all_scales, _mm256_set1_ps(-43.f));
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = (const block_q8_2_x4 *)info.src1_row(iy);
                auto my1 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(y[2*i+0].d + 4)));
                auto my2 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)(y[2*i+1].d + 4)));
                auto my  = _mm256_castsi256_ps(_mm256_slli_epi32(MM256_SET_M128I(my2, my1), 16));
                accd[iy] = _mm256_fmadd_ps(all_mins, my, accd[iy]);
            }
            auto scales_l = _mm256_castps256_ps128(all_scales);
            auto scales_h = _mm256_extractf128_ps(all_scales, 1);
            scales[0] = _mm256_set_m128(scales_l, scales_l);
            scales[1] = _mm256_set_m128(scales_h, scales_h);

            for (int j = 0; j < QK_K/128; ++j) {
                const uint8_t * a8 = (const uint8_t *)(a16 + 16*j);
                for (int k = 0; k < 4; ++k) {
                    auto a8k = a8 + 8*k;
                    xv[k] = _mm256_set_epi64x(iq2xxs_grid[a8k[3]], iq2xxs_grid[a8k[2]], iq2xxs_grid[a8k[1]], iq2xxs_grid[a8k[0]]);
                    uint32_t aux32; std::memcpy(&aux32, a8k+4, sizeof(uint32_t));
                    esh.sign_value(aux32, xv[k]);
                    xv[k] = _mm256_add_epi8(xv[k], _mm256_set1_epi8(43));
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = (const block_q8_2_x4 *)info.src1_row(iy);
                    sumi[0] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), xv[0], _mm256_loadu_si256((const __m256i*)y[2*i+j].qs+0));
                    sumi[1] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), xv[1], _mm256_loadu_si256((const __m256i*)y[2*i+j].qs+1));
                    sumi[2] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), xv[2], _mm256_loadu_si256((const __m256i*)y[2*i+j].qs+2));
                    sumi[3] = _mm256_dpbusd_epi32(_mm256_setzero_si256(), xv[3], _mm256_loadu_si256((const __m256i*)y[2*i+j].qs+3));
                    sumi[0] = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi[0], sumi[1]), _mm256_unpackhi_epi32(sumi[0], sumi[1]));
                    sumi[2] = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi[2], sumi[3]), _mm256_unpackhi_epi32(sumi[2], sumi[3]));
                    sumi[0] = _mm256_add_epi32(_mm256_unpacklo_epi64(sumi[0], sumi[2]), _mm256_unpackhi_epi64(sumi[0], sumi[2]));
                    auto d4 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i *)y[2*i+j].d)), 16));
                    auto dy = _mm256_set_m128(d4, d4);
                    accd[iy] = _mm256_fmadd_ps(_mm256_mul_ps(scales[j], dy), _mm256_cvtepi32_ps(sumi[0]), accd[iy]);
                }
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, hsum_float_8(accd[iy]));
        }
    }
}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_K_q8_K_IQ(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
#ifdef HAVE_FANCY_SIMD
    if constexpr (nrc_y == 1) {
        mul_mat_qX_K_q8_K_IQ_1<Dequantizer>(n, vx, bx, info, nrc_x);
    } else {
        mul_mat_qX_K_q8_K_IQ_N<Dequantizer, nrc_y>(n, vx, bx, info, nrc_x);
    }
#else
    mul_mat_qX_K_q8_K_IQ_N<Dequantizer, nrc_y>(n, vx, bx, info, nrc_x);
#endif
}

template <int nrc_y>
static void mul_mat_iq2_xxs_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
#ifndef HAVE_FANCY_SIMD
    auto smask = _mm256_set1_epi64x(0x8040201008040201);
    auto sign_shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    auto m4 = _mm256_set1_epi8(4);
    auto m1 = _mm256_set1_epi16(1);
#endif
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq2 = (const block_iq2_xxs_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq2[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto qs = iq2[ibl].qs;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                qx[0] = _mm256_set_epi64x(iq2xxs_grid[qs[ 3]], iq2xxs_grid[qs[ 2]], iq2xxs_grid[qs[ 1]], iq2xxs_grid[qs[ 0]]);
                qx[1] = _mm256_set_epi64x(iq2xxs_grid[qs[ 7]], iq2xxs_grid[qs[ 6]], iq2xxs_grid[qs[ 5]], iq2xxs_grid[qs[ 4]]);
                qx[2] = _mm256_set_epi64x(iq2xxs_grid[qs[11]], iq2xxs_grid[qs[10]], iq2xxs_grid[qs[ 9]], iq2xxs_grid[qs[ 8]]);
                qx[3] = _mm256_set_epi64x(iq2xxs_grid[qs[15]], iq2xxs_grid[qs[14]], iq2xxs_grid[qs[13]], iq2xxs_grid[qs[12]]);
                qs += 16;
                auto sas = _mm_loadu_si128((const __m128i *)iq2[ibl].sas + ib);
                auto scales = _mm_and_si128(sas, _mm_set1_epi8(1));
#ifdef HAVE_FANCY_SIMD
                scales = _mm_dpbusd_epi32(_mm_set1_epi32(1), scales, _mm_set1_epi32(0x10080402));
#else
                scales = _mm_maddubs_epi16(scales, _mm_set1_epi32(0x10080402));
                scales = _mm_add_epi32(_mm_madd_epi16(_mm_set1_epi16(1), scales), _mm_set1_epi32(1));
#endif
                auto scales32 = MM256_SET_M128I(scales, scales);
                auto signs128 = _mm_and_si128(sas, _mm_set1_epi8(-2)); // 0xfe = -2 as signed. Needed to shutup compiler warning.
                signs128 = _mm_xor_si128(signs128, _mm_srli_epi16(signs128, 1));
#ifdef HAVE_FANCY_SIMD
                auto mask = (const __mmask32 *)&signs128;
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[0], _mm256_mask_sub_epi8(y, mask[0], _mm256_setzero_si256(), y));
                    auto sumi2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[1], _mm256_mask_sub_epi8(y, mask[1], _mm256_setzero_si256(), y));
                    auto sumi3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[2], _mm256_mask_sub_epi8(y, mask[2], _mm256_setzero_si256(), y));
                    auto sumi4 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[3], _mm256_mask_sub_epi8(y, mask[3], _mm256_setzero_si256(), y));
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales32, sumi));
                }
#else
                auto signs = MM256_SET_M128I(signs128, signs128);
                auto shuffle = sign_shuffle;
                auto s1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s3 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s4 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1)));
                    auto sumi2 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2)));
                    auto sumi3 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3)));
                    auto sumi4 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4)));
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales32, sumi));
                }
#endif
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, _mm_mul_ps(_mm_set1_ps(0.125f), sum));
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

template <int nrc_y>
static void mul_mat_iq2_xs_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
#ifndef HAVE_FANCY_SIMD
    auto smask = _mm256_set1_epi64x(0x8040201008040201);
    auto sign_shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    auto m4 = _mm256_set1_epi8(4);
#endif
    __m256  acc[nrc_y] = {};
#ifdef HAVE_FANCY_SIMD
    __m256i shuffles[2] = {
        _mm256_set_epi64x(0x0706070607060706, 0x0302030203020302, 0x0504050405040504, 0x0100010001000100),
        _mm256_set_epi64x(0x0f0e0f0e0f0e0f0e, 0x0b0a0b0a0b0a0b0a, 0x0d0c0d0c0d0c0d0c, 0x0908090809080908)
    };
    __m256i isum[2*nrc_y] = {};
#else
    __m256i shuffles[4] = {
        MM256_SET_M128I(_mm_set1_epi16(0x0302), _mm_set1_epi16(0x0100)),
        MM256_SET_M128I(_mm_set1_epi16(0x0706), _mm_set1_epi16(0x0504)),
        MM256_SET_M128I(_mm_set1_epi16(0x0b0a), _mm_set1_epi16(0x0908)),
        MM256_SET_M128I(_mm_set1_epi16(0x0f0e), _mm_set1_epi16(0x0d0c)),
    };
    __m256i isum[nrc_y == 1 ? 4 : nrc_y] = {};
#endif
    auto s_shuffle = _mm_set_epi64x(0x0f0d0b0907050301, 0x0e0c0a0806040200);
    __m256i qx[4];
    union { __m256i vec; uint16_t val[16]; } helper;
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq2 = (const block_iq2_xs_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq2[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto s32 = (const uint32_t *)iq2[ibl].scales;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto val = _mm256_loadu_si256((const __m256i *)iq2[ibl].qs + ib);
                helper.vec = _mm256_and_si256(val, _mm256_set1_epi16(511));
                qx[0] = _mm256_set_epi64x(iq2xs_grid[helper.val[ 3]], iq2xs_grid[helper.val[ 2]], iq2xs_grid[helper.val[ 1]], iq2xs_grid[helper.val[ 0]]);
                qx[1] = _mm256_set_epi64x(iq2xs_grid[helper.val[ 7]], iq2xs_grid[helper.val[ 6]], iq2xs_grid[helper.val[ 5]], iq2xs_grid[helper.val[ 4]]);
                qx[2] = _mm256_set_epi64x(iq2xs_grid[helper.val[11]], iq2xs_grid[helper.val[10]], iq2xs_grid[helper.val[ 9]], iq2xs_grid[helper.val[ 8]]);
                qx[3] = _mm256_set_epi64x(iq2xs_grid[helper.val[15]], iq2xs_grid[helper.val[14]], iq2xs_grid[helper.val[13]], iq2xs_grid[helper.val[12]]);
                auto signs16 = _mm256_srli_epi16(val, 9);
                signs16 = _mm256_xor_si256(signs16, _mm256_slli_epi16(signs16, 1));
                auto signs128 = _mm_or_si128(_mm256_castsi256_si128(signs16), _mm_slli_epi16(_mm256_extracti128_si256(signs16, 1), 8));
                signs128 = _mm_shuffle_epi8(signs128, s_shuffle);
                auto scales = _mm_set1_epi32(s32[ib]);
                scales = _mm_and_si128(_mm_unpacklo_epi8(scales, _mm_srli_epi16(scales, 4)), _mm_set1_epi8(0xf));
                scales = _mm_or_si128(_mm_slli_epi16(scales, 1), _mm_set1_epi8(1));
                auto scales16 = _mm256_cvtepi8_epi16(scales);  // 0...7, 0...7
#ifdef HAVE_FANCY_SIMD
                __m256i scs[2] = { _mm256_shuffle_epi8(scales16, shuffles[0]), _mm256_shuffle_epi8(scales16, shuffles[1]) };
                auto mask = (const __mmask32 *)&signs128;
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[0], _mm256_mask_sub_epi8(y, mask[0], _mm256_setzero_si256(), y)); // blocks: 0,0,0,0,  1,1,1,1, row 0
                    auto sumi2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[1], _mm256_mask_sub_epi8(y, mask[1], _mm256_setzero_si256(), y)); // blocks: 2,2,2,2,  3,3,3,3, row 1
                    auto sumi3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[2], _mm256_mask_sub_epi8(y, mask[2], _mm256_setzero_si256(), y)); // blocks: 4,4,4,4,  5,5,5,5, row 2
                    auto sumi4 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[3], _mm256_mask_sub_epi8(y, mask[3], _mm256_setzero_si256(), y)); // blocks: 6,6,6,6,  7,7,7,7, row 3
                    auto s12 = _mm256_packs_epi32(sumi1, sumi2);  // 0,0,0,0, 2,2,2,2,  1,1,1,1, 3,3,3,3
                    auto s34 = _mm256_packs_epi32(sumi3, sumi4);  // 4,4,4,4, 6,6,6,6,  5,5,5,5, 7,7,7,7
                    isum[2*iy+0] = _mm256_add_epi32(isum[2*iy+0], _mm256_madd_epi16(scs[0], s12));
                    isum[2*iy+1] = _mm256_add_epi32(isum[2*iy+1], _mm256_madd_epi16(scs[1], s34));
                }
#else
                auto signs = MM256_SET_M128I(signs128, signs128);
                auto shuffle = sign_shuffle;
                auto s1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s3 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s4 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                __m256i scs[4] = {
                    _mm256_shuffle_epi8(scales16, shuffles[0]), _mm256_shuffle_epi8(scales16, shuffles[1]),
                    _mm256_shuffle_epi8(scales16, shuffles[2]), _mm256_shuffle_epi8(scales16, shuffles[3]),
                };
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    if constexpr (nrc_y == 1) {
                        isum[0] = _mm256_add_epi32(isum[0], _mm256_madd_epi16(scs[0], _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1))));
                        isum[1] = _mm256_add_epi32(isum[1], _mm256_madd_epi16(scs[1], _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2))));
                        isum[2] = _mm256_add_epi32(isum[2], _mm256_madd_epi16(scs[2], _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3))));
                        isum[3] = _mm256_add_epi32(isum[3], _mm256_madd_epi16(scs[3], _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4))));
                    } else {
                        auto sumi1 = _mm256_madd_epi16(scs[0], _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1))); // blocks 4x0, 4x1, row 0
                        auto sumi2 = _mm256_madd_epi16(scs[1], _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2))); // blocks 4x2, 4x3, row 1
                        auto sumi3 = _mm256_madd_epi16(scs[2], _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3))); // blocks 4x4, 4x5, row 2
                        auto sumi4 = _mm256_madd_epi16(scs[3], _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4))); // blocks 4x6, 4x7, row 3
                        auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                        auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                        auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                        isum[iy] = _mm256_add_epi32(isum[iy], sumi);
                    }
                }
#endif
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
#ifdef HAVE_FANCY_SIMD
                auto sumi = _mm256_hadd_epi32(isum[2*iy+0], isum[2*iy+1]);
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                isum[2*iy+0] = isum[2*iy+1] = _mm256_setzero_si256();
#else
                if constexpr (nrc_y == 1) {
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(isum[0], isum[1]), _mm256_unpackhi_epi32(isum[0], isum[1]));
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(isum[2], isum[3]), _mm256_unpackhi_epi32(isum[2], isum[3]));
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34));
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    isum[0] = isum[1] = isum[2] = isum[3] = _mm256_setzero_si256();
                } else {
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                    isum[iy] = _mm256_setzero_si256();
                }
#endif
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, _mm_mul_ps(_mm_set1_ps(0.125f), sum));
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

static void mul_mat_iq2_xs_r4_q8_k_16(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    constexpr int nrc_y = 16;
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
#ifndef HAVE_FANCY_SIMD
    auto smask = _mm256_set1_epi64x(0x8040201008040201);
    auto sign_shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    auto m4 = _mm256_set1_epi8(4);
#endif
    __m256  acc[nrc_y] = {};
#ifdef HAVE_FANCY_SIMD
    __m256i shuffles[2] = {
        _mm256_set_epi64x(0x0706070607060706, 0x0302030203020302, 0x0504050405040504, 0x0100010001000100),
        _mm256_set_epi64x(0x0f0e0f0e0f0e0f0e, 0x0b0a0b0a0b0a0b0a, 0x0d0c0d0c0d0c0d0c, 0x0908090809080908)
    };
    __m256i isum[2*nrc_y] = {};
#else
    __m256i shuffles[4] = {
        MM256_SET_M128I(_mm_set1_epi16(0x0302), _mm_set1_epi16(0x0100)),
        MM256_SET_M128I(_mm_set1_epi16(0x0706), _mm_set1_epi16(0x0504)),
        MM256_SET_M128I(_mm_set1_epi16(0x0b0a), _mm_set1_epi16(0x0908)),
        MM256_SET_M128I(_mm_set1_epi16(0x0f0e), _mm_set1_epi16(0x0d0c)),
    };
    __m256i isum[nrc_y == 1 ? 4 : nrc_y] = {};
#endif
    auto s_shuffle = _mm_set_epi64x(0x0f0d0b0907050301, 0x0e0c0a0806040200);
    __m256i qx[4];
    union { __m256i vec; uint16_t val[16]; } helper;
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq2 = (const block_iq2_xs_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq2[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto s32 = (const uint32_t *)iq2[ibl].scales;
            {
                auto scale_bits = _mm256_loadu_si256((const __m256i *)iq2[ibl].scales);
                auto scales1 = _mm256_and_si256(scale_bits, _mm256_set1_epi8(0xf));
                auto scales2 = _mm256_and_si256(_mm256_srli_epi16(scale_bits, 4), _mm256_set1_epi8(0xf));
                scales1 = _mm256_or_si256(_mm256_slli_epi16(scales1, 1), _mm256_set1_epi8(1));
                scales2 = _mm256_or_si256(_mm256_slli_epi16(scales2, 1), _mm256_set1_epi8(1));
                auto s1_8 = _mm256_unpacklo_epi8(scales1, scales2); // blocks 0...15, 32...47  (0...3, 8...11 from each row)
                auto s2_8 = _mm256_unpackhi_epi8(scales1, scales2); // blocks 16..31, 48...63  (4...7, 12..15 from each row)
                auto s1_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s1_8));       //  0...15 (0...3 from each row)
                auto s2_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(s1_8, 1));  // 32...47 (8..11 from each row)
                auto s3_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s2_8));       // 16...31 (4...7 from each row)
                auto s4_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(s2_8, 1));  // 48...63 (12.15 from each row)
                auto t1 = MM256_SET_M128I(_mm256_castsi256_si128(s2_16), _mm256_castsi256_si128(s1_16));            // 0,1 and  8,9 from each row
                auto t2 = MM256_SET_M128I(_mm256_extracti128_si256(s2_16, 1), _mm256_extracti128_si256(s1_16, 1));  // 2,3 and 10,11 from each row
                auto t3 = MM256_SET_M128I(_mm256_castsi256_si128(s4_16), _mm256_castsi256_si128(s3_16));            // 4,5 and 12,13 from each row
                auto t4 = MM256_SET_M128I(_mm256_extracti128_si256(s4_16, 1), _mm256_extracti128_si256(s3_16, 1));  // 6,7 and 14,15 from each row
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto bsums = q8.load_bsums(iy, ibl);
                    auto sumi = _mm256_setzero_si256();
#ifdef HAVE_FANCY_SIMD
                    sumi = _mm256_dpwssd_epi32(sumi, t1, _mm256_shuffle_epi32(bsums, 0x00));
                    sumi = _mm256_dpwssd_epi32(sumi, t2, _mm256_shuffle_epi32(bsums, 0x55));
                    sumi = _mm256_dpwssd_epi32(sumi, t3, _mm256_shuffle_epi32(bsums, 0xaa));
                    sumi = _mm256_dpwssd_epi32(sumi, t4, _mm256_shuffle_epi32(bsums, 0xff));
#else
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(t1, _mm256_shuffle_epi32(bsums, 0x00)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(t2, _mm256_shuffle_epi32(bsums, 0x55)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(t3, _mm256_shuffle_epi32(bsums, 0xaa)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(t4, _mm256_shuffle_epi32(bsums, 0xff)));
#endif
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(-64.f*q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto val = _mm256_loadu_si256((const __m256i *)iq2[ibl].qs + ib);
                helper.vec = _mm256_and_si256(val, _mm256_set1_epi16(511));
                qx[0] = _mm256_set_epi64x(iq2xs_grid[helper.val[ 3]], iq2xs_grid[helper.val[ 2]], iq2xs_grid[helper.val[ 1]], iq2xs_grid[helper.val[ 0]]);
                qx[1] = _mm256_set_epi64x(iq2xs_grid[helper.val[ 7]], iq2xs_grid[helper.val[ 6]], iq2xs_grid[helper.val[ 5]], iq2xs_grid[helper.val[ 4]]);
                qx[2] = _mm256_set_epi64x(iq2xs_grid[helper.val[11]], iq2xs_grid[helper.val[10]], iq2xs_grid[helper.val[ 9]], iq2xs_grid[helper.val[ 8]]);
                qx[3] = _mm256_set_epi64x(iq2xs_grid[helper.val[15]], iq2xs_grid[helper.val[14]], iq2xs_grid[helper.val[13]], iq2xs_grid[helper.val[12]]);
                auto signs16 = _mm256_srli_epi16(val, 9);
                signs16 = _mm256_xor_si256(signs16, _mm256_slli_epi16(signs16, 1));
                auto signs128 = _mm_or_si128(_mm256_castsi256_si128(signs16), _mm_slli_epi16(_mm256_extracti128_si256(signs16, 1), 8));
                signs128 = _mm_shuffle_epi8(signs128, s_shuffle);
                auto scales = _mm_set1_epi32(s32[ib]);
                scales = _mm_and_si128(_mm_unpacklo_epi8(scales, _mm_srli_epi16(scales, 4)), _mm_set1_epi8(0xf));
                scales = _mm_or_si128(_mm_slli_epi16(scales, 1), _mm_set1_epi8(1));
                auto scales16 = _mm256_cvtepi8_epi16(scales);  // 0...7, 0...7
#ifdef HAVE_FANCY_SIMD
                __m256i scs[2] = { _mm256_shuffle_epi8(scales16, shuffles[0]), _mm256_shuffle_epi8(scales16, shuffles[1]) };
                auto mask = (const __mmask32 *)&signs128;
                qx[0] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_mask_sub_epi8(qx[0], mask[0], _mm256_setzero_si256(), qx[0]));
                qx[1] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_mask_sub_epi8(qx[1], mask[1], _mm256_setzero_si256(), qx[1]));
                qx[2] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_mask_sub_epi8(qx[2], mask[2], _mm256_setzero_si256(), qx[2]));
                qx[3] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_mask_sub_epi8(qx[3], mask[3], _mm256_setzero_si256(), qx[3]));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[0], y); // blocks: 0,0,0,0,  1,1,1,1, row 0
                    auto sumi2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[1], y); // blocks: 2,2,2,2,  3,3,3,3, row 1
                    auto sumi3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[2], y); // blocks: 4,4,4,4,  5,5,5,5, row 2
                    auto sumi4 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[3], y); // blocks: 6,6,6,6,  7,7,7,7, row 3
                    auto s12 = _mm256_packs_epi32(sumi1, sumi2);  // 0,0,0,0, 2,2,2,2,  1,1,1,1, 3,3,3,3
                    auto s34 = _mm256_packs_epi32(sumi3, sumi4);  // 4,4,4,4, 6,6,6,6,  5,5,5,5, 7,7,7,7
                    isum[2*iy+0] = _mm256_add_epi32(isum[2*iy+0], _mm256_madd_epi16(scs[0], s12));
                    isum[2*iy+1] = _mm256_add_epi32(isum[2*iy+1], _mm256_madd_epi16(scs[1], s34));
                }
#else
                auto signs = MM256_SET_M128I(signs128, signs128);
                auto shuffle = sign_shuffle;
                auto s = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                qx[0] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_sign_epi8(qx[0], s));
                s = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                qx[1] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_sign_epi8(qx[1], s));
                s = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                qx[2] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_sign_epi8(qx[2], s));
                s = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                qx[3] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_sign_epi8(qx[3], s));
                __m256i scs[4] = {
                    _mm256_shuffle_epi8(scales16, shuffles[0]), _mm256_shuffle_epi8(scales16, shuffles[1]),
                    _mm256_shuffle_epi8(scales16, shuffles[2]), _mm256_shuffle_epi8(scales16, shuffles[3]),
                };
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_madd_epi16(scs[0], _mm256_maddubs_epi16(qx[0], y)); // blocks 4x0, 4x1, row 0
                    auto sumi2 = _mm256_madd_epi16(scs[1], _mm256_maddubs_epi16(qx[1], y)); // blocks 4x2, 4x3, row 1
                    auto sumi3 = _mm256_madd_epi16(scs[2], _mm256_maddubs_epi16(qx[2], y)); // blocks 4x4, 4x5, row 2
                    auto sumi4 = _mm256_madd_epi16(scs[3], _mm256_maddubs_epi16(qx[3], y)); // blocks 4x6, 4x7, row 3
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                    isum[iy] = _mm256_add_epi32(isum[iy], sumi);
                }
#endif
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
#ifdef HAVE_FANCY_SIMD
                auto sumi = _mm256_hadd_epi32(isum[2*iy+0], isum[2*iy+1]);
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                isum[2*iy+0] = isum[2*iy+1] = _mm256_setzero_si256();
#else
                if constexpr (nrc_y == 1) {
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(isum[0], isum[1]), _mm256_unpackhi_epi32(isum[0], isum[1]));
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(isum[2], isum[3]), _mm256_unpackhi_epi32(isum[2], isum[3]));
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34));
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    isum[0] = isum[1] = isum[2] = isum[3] = _mm256_setzero_si256();
                } else {
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                    isum[iy] = _mm256_setzero_si256();
                }
#endif
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, _mm_mul_ps(_mm_set1_ps(0.125f), sum));
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

template <int nrc_y>
static void mul_mat_iq2_s_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
#ifndef HAVE_FANCY_SIMD
    auto smask = _mm256_set1_epi64x(0x8040201008040201);
    auto sign_shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    auto m4 = _mm256_set1_epi8(4);
#endif
    __m256  acc[nrc_y] = {};
#ifdef HAVE_FANCY_SIMD
    __m256i shuffles[2] = {
        _mm256_set_epi64x(0x0706070607060706, 0x0302030203020302, 0x0504050405040504, 0x0100010001000100),
        _mm256_set_epi64x(0x0f0e0f0e0f0e0f0e, 0x0b0a0b0a0b0a0b0a, 0x0d0c0d0c0d0c0d0c, 0x0908090809080908)
    };
    __m256i isum[2*nrc_y] = {};
#else
    __m256i shuffles[4] = {
        MM256_SET_M128I(_mm_set1_epi16(0x0302), _mm_set1_epi16(0x0100)),
        MM256_SET_M128I(_mm_set1_epi16(0x0706), _mm_set1_epi16(0x0504)),
        MM256_SET_M128I(_mm_set1_epi16(0x0b0a), _mm_set1_epi16(0x0908)),
        MM256_SET_M128I(_mm_set1_epi16(0x0f0e), _mm_set1_epi16(0x0d0c)),
    };
    __m256i isum[nrc_y == 1 ? 4 : nrc_y] = {};
#endif
    __m256i qx[4];
    auto grid = iq2s_grid;
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq2 = (const block_iq2_s_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq2[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto s32 = (const uint32_t *)iq2[ibl].scales;
            auto ql = iq2[ibl].qs;
            auto qh = iq2[ibl].qh;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                qx[0] = _mm256_set_epi64x(grid[ql[ 3] | ((qh[0] << 2) & 0x300)], grid[ql[ 2] | ((qh[0] << 4) & 0x300)], grid[ql[ 1] | ((qh[0] << 6) & 0x300)], grid[ql[ 0] | ((qh[0] << 8) & 0x300)]);
                qx[1] = _mm256_set_epi64x(grid[ql[ 7] | ((qh[1] << 2) & 0x300)], grid[ql[ 6] | ((qh[1] << 4) & 0x300)], grid[ql[ 5] | ((qh[1] << 6) & 0x300)], grid[ql[ 4] | ((qh[1] << 8) & 0x300)]);
                qx[2] = _mm256_set_epi64x(grid[ql[11] | ((qh[2] << 2) & 0x300)], grid[ql[10] | ((qh[2] << 4) & 0x300)], grid[ql[ 9] | ((qh[2] << 6) & 0x300)], grid[ql[ 8] | ((qh[2] << 8) & 0x300)]);
                qx[3] = _mm256_set_epi64x(grid[ql[15] | ((qh[3] << 2) & 0x300)], grid[ql[14] | ((qh[3] << 4) & 0x300)], grid[ql[13] | ((qh[3] << 6) & 0x300)], grid[ql[12] | ((qh[3] << 8) & 0x300)]);
                ql += 16; qh += 4;
                auto signs128 = _mm_loadu_si128((const __m128i*)iq2[ibl].signs + ib);
                auto scales = _mm_set1_epi32(s32[ib]);
                scales = _mm_and_si128(_mm_unpacklo_epi8(scales, _mm_srli_epi16(scales, 4)), _mm_set1_epi8(0xf));
                scales = _mm_or_si128(_mm_slli_epi16(scales, 1), _mm_set1_epi8(1));
                auto scales16 = _mm256_cvtepi8_epi16(scales);  // 0...7, 0...7
#ifdef HAVE_FANCY_SIMD
                __m256i scs[2] = { _mm256_shuffle_epi8(scales16, shuffles[0]), _mm256_shuffle_epi8(scales16, shuffles[1]) };
                auto mask = (const __mmask32 *)&signs128;
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[0], _mm256_mask_sub_epi8(y, mask[0], _mm256_setzero_si256(), y)); // blocks: 0,0,0,0,  1,1,1,1, row 0
                    auto sumi2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[1], _mm256_mask_sub_epi8(y, mask[1], _mm256_setzero_si256(), y)); // blocks: 2,2,2,2,  3,3,3,3, row 1
                    auto sumi3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[2], _mm256_mask_sub_epi8(y, mask[2], _mm256_setzero_si256(), y)); // blocks: 4,4,4,4,  5,5,5,5, row 2
                    auto sumi4 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[3], _mm256_mask_sub_epi8(y, mask[3], _mm256_setzero_si256(), y)); // blocks: 6,6,6,6,  7,7,7,7, row 3
                    auto s12 = _mm256_packs_epi32(sumi1, sumi2);  // 0,0,0,0, 2,2,2,2,  1,1,1,1, 3,3,3,3
                    auto s34 = _mm256_packs_epi32(sumi3, sumi4);  // 4,4,4,4, 6,6,6,6,  5,5,5,5, 7,7,7,7
                    isum[2*iy+0] = _mm256_add_epi32(isum[2*iy+0], _mm256_madd_epi16(scs[0], s12));
                    isum[2*iy+1] = _mm256_add_epi32(isum[2*iy+1], _mm256_madd_epi16(scs[1], s34));
                }
#else
                auto signs = MM256_SET_M128I(signs128, signs128);
                auto shuffle = sign_shuffle;
                auto s1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s3 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s4 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                __m256i scs[4] = {
                    _mm256_shuffle_epi8(scales16, shuffles[0]), _mm256_shuffle_epi8(scales16, shuffles[1]),
                    _mm256_shuffle_epi8(scales16, shuffles[2]), _mm256_shuffle_epi8(scales16, shuffles[3]),
                };
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    if constexpr (nrc_y == 1) {
                        isum[0] = _mm256_add_epi32(isum[0], _mm256_madd_epi16(scs[0], _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1))));
                        isum[1] = _mm256_add_epi32(isum[1], _mm256_madd_epi16(scs[1], _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2))));
                        isum[2] = _mm256_add_epi32(isum[2], _mm256_madd_epi16(scs[2], _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3))));
                        isum[3] = _mm256_add_epi32(isum[3], _mm256_madd_epi16(scs[3], _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4))));
                    } else {
                        auto sumi1 = _mm256_madd_epi16(scs[0], _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1))); // blocks 4x0, 4x1, row 0
                        auto sumi2 = _mm256_madd_epi16(scs[1], _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2))); // blocks 4x2, 4x3, row 1
                        auto sumi3 = _mm256_madd_epi16(scs[2], _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3))); // blocks 4x4, 4x5, row 2
                        auto sumi4 = _mm256_madd_epi16(scs[3], _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4))); // blocks 4x6, 4x7, row 3
                        auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                        auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                        auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                        isum[iy] = _mm256_add_epi32(isum[iy], sumi);
                    }
                }
#endif
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
#ifdef HAVE_FANCY_SIMD
                auto sumi = _mm256_hadd_epi32(isum[2*iy+0], isum[2*iy+1]);
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                isum[2*iy+0] = isum[2*iy+1] = _mm256_setzero_si256();
#else
                if constexpr (nrc_y == 1) {
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(isum[0], isum[1]), _mm256_unpackhi_epi32(isum[0], isum[1]));
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(isum[2], isum[3]), _mm256_unpackhi_epi32(isum[2], isum[3]));
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34));
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    isum[0] = isum[1] = isum[2] = isum[3] = _mm256_setzero_si256();
                } else {
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                    isum[iy] = _mm256_setzero_si256();
                }
#endif
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, _mm_mul_ps(_mm_set1_ps(0.125f), sum));
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

static void mul_mat_iq2_s_r4_q8_k_16(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    constexpr int nrc_y = 16;
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
#ifndef HAVE_FANCY_SIMD
    auto smask = _mm256_set1_epi64x(0x8040201008040201);
    auto sign_shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    auto m4 = _mm256_set1_epi8(4);
#endif
    __m256  acc[nrc_y] = {};
#ifdef HAVE_FANCY_SIMD
    __m256i shuffles[2] = {
        _mm256_set_epi64x(0x0706070607060706, 0x0302030203020302, 0x0504050405040504, 0x0100010001000100),
        _mm256_set_epi64x(0x0f0e0f0e0f0e0f0e, 0x0b0a0b0a0b0a0b0a, 0x0d0c0d0c0d0c0d0c, 0x0908090809080908)
    };
    __m256i isum[2*nrc_y] = {};
#else
    __m256i shuffles[4] = {
        MM256_SET_M128I(_mm_set1_epi16(0x0302), _mm_set1_epi16(0x0100)),
        MM256_SET_M128I(_mm_set1_epi16(0x0706), _mm_set1_epi16(0x0504)),
        MM256_SET_M128I(_mm_set1_epi16(0x0b0a), _mm_set1_epi16(0x0908)),
        MM256_SET_M128I(_mm_set1_epi16(0x0f0e), _mm_set1_epi16(0x0d0c)),
    };
    __m256i isum[nrc_y == 1 ? 4 : nrc_y] = {};
#endif
    __m256i qx[4];
    auto grid = iq2s_grid;
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq2 = (const block_iq2_s_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq2[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto s32 = (const uint32_t *)iq2[ibl].scales;
            auto ql = iq2[ibl].qs;
            auto qh = iq2[ibl].qh;
            {
                auto scale_bits = _mm256_loadu_si256((const __m256i *)iq2[ibl].scales);
                auto scales1 = _mm256_and_si256(scale_bits, _mm256_set1_epi8(0xf));
                auto scales2 = _mm256_and_si256(_mm256_srli_epi16(scale_bits, 4), _mm256_set1_epi8(0xf));
                scales1 = _mm256_or_si256(_mm256_slli_epi16(scales1, 1), _mm256_set1_epi8(1));
                scales2 = _mm256_or_si256(_mm256_slli_epi16(scales2, 1), _mm256_set1_epi8(1));
                auto s1_8 = _mm256_unpacklo_epi8(scales1, scales2); // blocks 0...15, 32...47  (0...3, 8...11 from each row)
                auto s2_8 = _mm256_unpackhi_epi8(scales1, scales2); // blocks 16..31, 48...63  (4...7, 12..15 from each row)
                auto s1_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s1_8));       //  0...15 (0...3 from each row)
                auto s2_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(s1_8, 1));  // 32...47 (8..11 from each row)
                auto s3_16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s2_8));       // 16...31 (4...7 from each row)
                auto s4_16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(s2_8, 1));  // 48...63 (12.15 from each row)
                auto t1 = MM256_SET_M128I(_mm256_castsi256_si128(s2_16), _mm256_castsi256_si128(s1_16));            // 0,1 and  8,9 from each row
                auto t2 = MM256_SET_M128I(_mm256_extracti128_si256(s2_16, 1), _mm256_extracti128_si256(s1_16, 1));  // 2,3 and 10,11 from each row
                auto t3 = MM256_SET_M128I(_mm256_castsi256_si128(s4_16), _mm256_castsi256_si128(s3_16));            // 4,5 and 12,13 from each row
                auto t4 = MM256_SET_M128I(_mm256_extracti128_si256(s4_16, 1), _mm256_extracti128_si256(s3_16, 1));  // 6,7 and 14,15 from each row
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto bsums = q8.load_bsums(iy, ibl);
                    auto sumi = _mm256_setzero_si256();
#ifdef HAVE_FANCY_SIMD
                    sumi = _mm256_dpwssd_epi32(sumi, t1, _mm256_shuffle_epi32(bsums, 0x00));
                    sumi = _mm256_dpwssd_epi32(sumi, t2, _mm256_shuffle_epi32(bsums, 0x55));
                    sumi = _mm256_dpwssd_epi32(sumi, t3, _mm256_shuffle_epi32(bsums, 0xaa));
                    sumi = _mm256_dpwssd_epi32(sumi, t4, _mm256_shuffle_epi32(bsums, 0xff));
#else
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(t1, _mm256_shuffle_epi32(bsums, 0x00)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(t2, _mm256_shuffle_epi32(bsums, 0x55)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(t3, _mm256_shuffle_epi32(bsums, 0xaa)));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(t4, _mm256_shuffle_epi32(bsums, 0xff)));
#endif
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(-64.f*q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                }
            }
            for (int ib = 0; ib < QK_K/32; ++ib) {
                qx[0] = _mm256_set_epi64x(grid[ql[ 3] | ((qh[0] << 2) & 0x300)], grid[ql[ 2] | ((qh[0] << 4) & 0x300)], grid[ql[ 1] | ((qh[0] << 6) & 0x300)], grid[ql[ 0] | ((qh[0] << 8) & 0x300)]);
                qx[1] = _mm256_set_epi64x(grid[ql[ 7] | ((qh[1] << 2) & 0x300)], grid[ql[ 6] | ((qh[1] << 4) & 0x300)], grid[ql[ 5] | ((qh[1] << 6) & 0x300)], grid[ql[ 4] | ((qh[1] << 8) & 0x300)]);
                qx[2] = _mm256_set_epi64x(grid[ql[11] | ((qh[2] << 2) & 0x300)], grid[ql[10] | ((qh[2] << 4) & 0x300)], grid[ql[ 9] | ((qh[2] << 6) & 0x300)], grid[ql[ 8] | ((qh[2] << 8) & 0x300)]);
                qx[3] = _mm256_set_epi64x(grid[ql[15] | ((qh[3] << 2) & 0x300)], grid[ql[14] | ((qh[3] << 4) & 0x300)], grid[ql[13] | ((qh[3] << 6) & 0x300)], grid[ql[12] | ((qh[3] << 8) & 0x300)]);
                ql += 16; qh += 4;
                auto signs128 = _mm_loadu_si128((const __m128i*)iq2[ibl].signs + ib);
                auto scales = _mm_set1_epi32(s32[ib]);
                scales = _mm_and_si128(_mm_unpacklo_epi8(scales, _mm_srli_epi16(scales, 4)), _mm_set1_epi8(0xf));
                scales = _mm_or_si128(_mm_slli_epi16(scales, 1), _mm_set1_epi8(1));
                auto scales16 = _mm256_cvtepi8_epi16(scales);  // 0...7, 0...7
#ifdef HAVE_FANCY_SIMD
                __m256i scs[2] = { _mm256_shuffle_epi8(scales16, shuffles[0]), _mm256_shuffle_epi8(scales16, shuffles[1]) };
                auto mask = (const __mmask32 *)&signs128;
                qx[0] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_mask_sub_epi8(qx[0], mask[0], _mm256_setzero_si256(), qx[0]));
                qx[1] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_mask_sub_epi8(qx[1], mask[1], _mm256_setzero_si256(), qx[1]));
                qx[2] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_mask_sub_epi8(qx[2], mask[2], _mm256_setzero_si256(), qx[2]));
                qx[3] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_mask_sub_epi8(qx[3], mask[3], _mm256_setzero_si256(), qx[3]));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[0], y); // blocks: 0,0,0,0,  1,1,1,1, row 0
                    auto sumi2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[1], y); // blocks: 2,2,2,2,  3,3,3,3, row 1
                    auto sumi3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[2], y); // blocks: 4,4,4,4,  5,5,5,5, row 2
                    auto sumi4 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[3], y); // blocks: 6,6,6,6,  7,7,7,7, row 3
                    auto s12 = _mm256_packs_epi32(sumi1, sumi2);  // 0,0,0,0, 2,2,2,2,  1,1,1,1, 3,3,3,3
                    auto s34 = _mm256_packs_epi32(sumi3, sumi4);  // 4,4,4,4, 6,6,6,6,  5,5,5,5, 7,7,7,7
                    isum[2*iy+0] = _mm256_add_epi32(isum[2*iy+0], _mm256_madd_epi16(scs[0], s12));
                    isum[2*iy+1] = _mm256_add_epi32(isum[2*iy+1], _mm256_madd_epi16(scs[1], s34));
                }
#else
                auto signs = MM256_SET_M128I(signs128, signs128);
                auto shuffle = sign_shuffle;
                auto s = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                qx[0] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_sign_epi8(qx[0], s));
                s = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                qx[1] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_sign_epi8(qx[1], s));
                s = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                qx[2] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_sign_epi8(qx[2], s));
                s = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                qx[3] = _mm256_add_epi8(_mm256_set1_epi8(64), _mm256_sign_epi8(qx[3], s));
                __m256i scs[4] = {
                    _mm256_shuffle_epi8(scales16, shuffles[0]), _mm256_shuffle_epi8(scales16, shuffles[1]),
                    _mm256_shuffle_epi8(scales16, shuffles[2]), _mm256_shuffle_epi8(scales16, shuffles[3]),
                };
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_madd_epi16(scs[0], _mm256_maddubs_epi16(qx[0], y)); // blocks 4x0, 4x1, row 0
                    auto sumi2 = _mm256_madd_epi16(scs[1], _mm256_maddubs_epi16(qx[1], y)); // blocks 4x2, 4x3, row 1
                    auto sumi3 = _mm256_madd_epi16(scs[2], _mm256_maddubs_epi16(qx[2], y)); // blocks 4x4, 4x5, row 2
                    auto sumi4 = _mm256_madd_epi16(scs[3], _mm256_maddubs_epi16(qx[3], y)); // blocks 4x6, 4x7, row 3
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                    isum[iy] = _mm256_add_epi32(isum[iy], sumi);
                }
#endif
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
#ifdef HAVE_FANCY_SIMD
                auto sumi = _mm256_hadd_epi32(isum[2*iy+0], isum[2*iy+1]);
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(sumi), acc[iy]);
                isum[2*iy+0] = isum[2*iy+1] = _mm256_setzero_si256();
#else
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
#endif
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, _mm_mul_ps(_mm_set1_ps(0.125f), sum));
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

template <int nrc_y>
static void mul_mat_iq3_xxs_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
#ifndef HAVE_FANCY_SIMD
    auto smask = _mm256_set1_epi64x(0x8040201008040201);
    auto sign_shuffle = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    auto m4 = _mm256_set1_epi8(4);
    auto m1 = _mm256_set1_epi16(1);
#endif
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y] = {};
    __m256i qx[4];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq3 = (const block_iq3_xxs_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_mul_ps(_mm_set1_ps(0.25f), _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq3[ibl].d))); // TODO: absorb the 0.25 factor into d when quantizing/repacking
            auto d4 = _mm256_set_m128(dl, dl);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                qx[0] = _mm256_set_epi32(iq3xxs_grid[iq3[ibl].qs[32*ib+ 7]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 6]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 5]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 4]],
                                         iq3xxs_grid[iq3[ibl].qs[32*ib+ 3]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 2]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 1]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 0]]);
                qx[1] = _mm256_set_epi32(iq3xxs_grid[iq3[ibl].qs[32*ib+15]], iq3xxs_grid[iq3[ibl].qs[32*ib+14]], iq3xxs_grid[iq3[ibl].qs[32*ib+13]], iq3xxs_grid[iq3[ibl].qs[32*ib+12]],
                                         iq3xxs_grid[iq3[ibl].qs[32*ib+11]], iq3xxs_grid[iq3[ibl].qs[32*ib+10]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 9]], iq3xxs_grid[iq3[ibl].qs[32*ib+ 8]]);
                qx[2] = _mm256_set_epi32(iq3xxs_grid[iq3[ibl].qs[32*ib+23]], iq3xxs_grid[iq3[ibl].qs[32*ib+22]], iq3xxs_grid[iq3[ibl].qs[32*ib+21]], iq3xxs_grid[iq3[ibl].qs[32*ib+20]],
                                         iq3xxs_grid[iq3[ibl].qs[32*ib+19]], iq3xxs_grid[iq3[ibl].qs[32*ib+18]], iq3xxs_grid[iq3[ibl].qs[32*ib+17]], iq3xxs_grid[iq3[ibl].qs[32*ib+16]]);
                qx[3] = _mm256_set_epi32(iq3xxs_grid[iq3[ibl].qs[32*ib+31]], iq3xxs_grid[iq3[ibl].qs[32*ib+30]], iq3xxs_grid[iq3[ibl].qs[32*ib+29]], iq3xxs_grid[iq3[ibl].qs[32*ib+28]],
                                         iq3xxs_grid[iq3[ibl].qs[32*ib+27]], iq3xxs_grid[iq3[ibl].qs[32*ib+26]], iq3xxs_grid[iq3[ibl].qs[32*ib+25]], iq3xxs_grid[iq3[ibl].qs[32*ib+24]]);
                auto sas = _mm_loadu_si128((const __m128i *)iq3[ibl].sas + ib);
                auto scales = _mm_and_si128(sas, _mm_set1_epi8(1));
#ifdef HAVE_FANCY_SIMD
                scales = _mm_dpbusd_epi32(_mm_set1_epi32(1), scales, _mm_set1_epi32(0x10080402));
#else
                scales = _mm_maddubs_epi16(scales, _mm_set1_epi32(0x10080402));
                scales = _mm_add_epi32(_mm_madd_epi16(_mm_set1_epi16(1), scales), _mm_set1_epi32(1));
                //auto t1 = _mm_or_si128(_mm_and_si128(scales, _mm_set1_epi32(0x00000001)), _mm_srli_epi32(_mm_and_si128(scales, _mm_set1_epi32(0x00000100)), 7));
                //auto t2 = _mm_or_si128(_mm_srli_epi32(_mm_and_si128(scales, _mm_set1_epi32(0x00010000)), 14), _mm_srli_epi32(_mm_and_si128(scales, _mm_set1_epi32(0x01000000)), 21));
                //scales = _mm_or_si128(_mm_slli_epi32(_mm_or_si128(t1, t2), 1), _mm_set1_epi32(1));
#endif
                auto scales32 = MM256_SET_M128I(scales, scales);
                auto signs128 = _mm_and_si128(sas, _mm_set1_epi8(-2)); // 0xfe = -2 as signed. Needed to shutup compiler warning.
                signs128 = _mm_xor_si128(signs128, _mm_srli_epi16(signs128, 1));
#ifdef HAVE_FANCY_SIMD
                auto mask = (const __mmask32 *)&signs128;
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[0], _mm256_mask_sub_epi8(y, mask[0], _mm256_setzero_si256(), y));
                    auto sumi2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[1], _mm256_mask_sub_epi8(y, mask[1], _mm256_setzero_si256(), y));
                    auto sumi3 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[2], _mm256_mask_sub_epi8(y, mask[2], _mm256_setzero_si256(), y));
                    auto sumi4 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[3], _mm256_mask_sub_epi8(y, mask[3], _mm256_setzero_si256(), y));
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales32, sumi));
                }
#else
                auto signs = MM256_SET_M128I(signs128, signs128);
                auto shuffle = sign_shuffle;
                auto s1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s3 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                shuffle = _mm256_add_epi8(shuffle, m4);
                auto s4 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(signs, shuffle), smask), smask), _mm256_set1_epi8(1));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi1 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(y, s1)));
                    auto sumi2 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(y, s2)));
                    auto sumi3 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(y, s3)));
                    auto sumi4 = _mm256_madd_epi16(m1, _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(y, s4)));
                    auto s12 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi1, sumi2), _mm256_unpackhi_epi32(sumi1, sumi2)); // 0,1, 0,1, 0,1, 0,1
                    auto s34 = _mm256_add_epi32(_mm256_unpacklo_epi32(sumi3, sumi4), _mm256_unpackhi_epi32(sumi3, sumi4)); // 2,3, 2,3, 2,3, 2,3
                    auto sumi = _mm256_add_epi32(_mm256_unpacklo_epi64(s12, s34), _mm256_unpackhi_epi64(s12, s34)); // 0,1,2,3, 0,1,2,3
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales32, sumi));
                }
#endif
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, sum);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

template <int nrc_y>
static void mul_mat_iq3_s_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
    auto smask = _mm256_set1_epi8(1);
    union { __m256i vec; uint32_t val[8]; } helper;
    union { __m128i vec; uint16_t val[8]; } hidx;
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y] = {};
    __m256i qx[4];
#ifdef HAVE_FANCY_SIMD
    __mmask32 mask[4];
#endif
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq3 = (const block_iq3_s_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq3[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto qs = iq3[ibl].qs;
            auto qh = iq3[ibl].qh;
            auto scale_bits = _mm_loadu_si128((const __m128i *)iq3[ibl].scales);
            auto scales8 = MM256_SET_M128I(_mm_srli_epi16(scale_bits, 4), scale_bits);
            helper.vec = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(scales8, _mm256_set1_epi8(0xf)), 1), _mm256_set1_epi8(1));
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto qh32 = (const uint32_t *)qh;
                auto idx_h = _mm_sllv_epi64(_mm_cvtepu8_epi16(_mm_set1_epi32(qh32[0])), _mm_set_epi64x(4, 8));
                for (int i = 0; i < 4; ++i) {
                    auto idx_l = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(qs + 8*i)));
                    hidx.vec = _mm_or_si128(idx_l, _mm_and_si128(idx_h, _mm_set1_epi16(0x100))); idx_h = _mm_srli_epi16(idx_h, 1);
                    qx[i] = _mm256_set_epi32(iq3s_grid[hidx.val[7]], iq3s_grid[hidx.val[6]], iq3s_grid[hidx.val[5]], iq3s_grid[hidx.val[4]],
                                             iq3s_grid[hidx.val[3]], iq3s_grid[hidx.val[2]], iq3s_grid[hidx.val[1]], iq3s_grid[hidx.val[0]]);
                }
                qs += 32; qh += 4;
                auto signs128 = _mm_loadu_si128((const __m128i*)iq3[ibl].signs + ib);
                auto signs = MM256_SET_M128I(_mm_srli_epi16(signs128, 4), signs128);
#ifdef HAVE_FANCY_SIMD
                auto scales = _mm256_cvtepi8_epi32(_mm_set1_epi32(helper.val[ib]));
                mask[0] = _mm256_cmpeq_epi8_mask(_mm256_and_si256(signs, smask), smask); signs = _mm256_srli_epi16(signs, 1);
                mask[1] = _mm256_cmpeq_epi8_mask(_mm256_and_si256(signs, smask), smask); signs = _mm256_srli_epi16(signs, 1);
                mask[2] = _mm256_cmpeq_epi8_mask(_mm256_and_si256(signs, smask), smask); signs = _mm256_srli_epi16(signs, 1);
                mask[3] = _mm256_cmpeq_epi8_mask(_mm256_and_si256(signs, smask), smask);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi = _mm256_setzero_si256();
                    auto ys = _mm256_shuffle_epi32(y, 0x00);
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_mask_sub_epi8(ys, mask[0], _mm256_setzero_si256(), ys));
                    ys = _mm256_shuffle_epi32(y, 0x55);
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_mask_sub_epi8(ys, mask[1], _mm256_setzero_si256(), ys));
                    ys = _mm256_shuffle_epi32(y, 0xaa);
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_mask_sub_epi8(ys, mask[2], _mm256_setzero_si256(), ys));
                    ys = _mm256_shuffle_epi32(y, 0xff);
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_mask_sub_epi8(ys, mask[3], _mm256_setzero_si256(), ys));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(sumi, scales));
                }
#else
                auto scales16 = _mm256_cvtepi8_epi16(_mm_set1_epi32(helper.val[ib]));
                auto scales = _mm256_unpacklo_epi16(scales16, scales16);
                auto s1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(signs, smask), smask), smask); signs = _mm256_srli_epi16(signs, 1);
                auto s2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(signs, smask), smask), smask); signs = _mm256_srli_epi16(signs, 1);
                auto s3 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(signs, smask), smask), smask); signs = _mm256_srli_epi16(signs, 1);
                auto s4 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(signs, smask), smask), smask);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ibl].qs + ib);
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_add_epi16(sumi, _mm256_maddubs_epi16(qx[0], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x00), s1)));
                    sumi = _mm256_add_epi16(sumi, _mm256_maddubs_epi16(qx[1], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0x55), s2)));
                    sumi = _mm256_add_epi16(sumi, _mm256_maddubs_epi16(qx[2], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xaa), s3)));
                    sumi = _mm256_add_epi16(sumi, _mm256_maddubs_epi16(qx[3], _mm256_sign_epi8(_mm256_shuffle_epi32(y, 0xff), s4)));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(scales, sumi));
                }
#endif
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, sum);
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

void iqk_convert_iq2_xxs_q8_0_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq2_xxs * x8[8];

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    ggml_half dh[8];
    uint16_t all_ls[64];
    EvenSignHelper esh;

    uint32_t block[8];
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq2_xxs *)((const char *)vx + (ix + k)*bx);
        for (int i = 0; i < nb; ++i) {
            // TODO: simdify
            for (int k = 0; k < 8; ++k) {
                dh[k] = x8[k][i].d;
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    std::memcpy(aux32, x8[k][i].qs + 4*ib32, 2*sizeof(uint32_t));
                    all_ls[8*ib32 + k] = (2*(aux32[1] >> 28) + 1);
                    auto value = _mm256_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
                    esh.sign_value(aux32[1], value);
                    _mm256_storeu_si256((__m256i *)block, value);
                    auto qs = (uint32_t *)y[ib32].qs;
                    for (int l = 0; l < 4; ++l) {
                        qs[8*l + k +  0] = block[l + 0];
                        qs[8*l + k + 32] = block[l + 4];
                    }
                }
            }
            auto vd = _mm256_mul_ps(_mm256_set1_ps(0.125f), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh)));
            for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                auto iscales16 = _mm_loadu_si128((const __m128i *)all_ls + ib32);
                auto iscales32 = _mm256_cvtepi16_epi32(iscales16);
                auto scales = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(iscales32));
                _mm_storeu_si128((__m128i *)y[ib32].d, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
            }
            y += QK_K/32;
        }
    }
}

void iqk_convert_iq3_xxs_q8_0_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq3_xxs * x8[8];

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    ggml_half dh[8];
    uint16_t all_ls[64];
    EvenSignHelper esh;

    uint32_t block[8];
    uint32_t aux32;

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq3_xxs *)((const char *)vx + (ix + k)*bx);
        for (int i = 0; i < nb; ++i) {
            // TODO: simdify
            for (int k = 0; k < 8; ++k) {
                dh[k] = x8[k][i].d;
                auto qs  = x8[k][i].qs;
                auto sas = qs + QK_K/4;
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    std::memcpy(&aux32, sas + 4*ib32, sizeof(uint32_t));
                    all_ls[8*ib32 + k] = (2*(aux32 >> 28) + 1);
                    auto value = _mm256_set_epi32(iq3xxs_grid[qs[7]], iq3xxs_grid[qs[6]], iq3xxs_grid[qs[5]], iq3xxs_grid[qs[4]],
                                                  iq3xxs_grid[qs[3]], iq3xxs_grid[qs[2]], iq3xxs_grid[qs[1]], iq3xxs_grid[qs[0]]);
                    esh.sign_value(aux32, value);
                    _mm256_storeu_si256((__m256i *)block, value);
                    auto q8 = (uint32_t *)y[ib32].qs;
                    for (int l = 0; l < 4; ++l) {
                        q8[8*l + k +  0] = block[l + 0];
                        q8[8*l + k + 32] = block[l + 4];
                    }
                    qs += 8;
                }
            }
            auto vd = _mm256_mul_ps(_mm256_set1_ps(0.25f), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh)));
            for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                auto iscales16 = _mm_loadu_si128((const __m128i *)all_ls + ib32);
                auto iscales32 = _mm256_cvtepi16_epi32(iscales16);
                auto scales = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(iscales32));
                _mm_storeu_si128((__m128i *)y[ib32].d, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
            }
            y += QK_K/32;
        }
    }
}

void iqk_convert_iq3_s_q8_0_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq3_s * x8[8];

    block_q8_0_r8 * y = (block_q8_0_r8 *)vy;

    ggml_half dh[8];
    uint16_t all_ls[64];
    SignHelper sh;
    IndexHelperIQ3S helper;

    uint32_t block[8];
    __m256i values[8];

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = (const block_iq3_s *)((const char *)vx + (ix + k)*bx);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                dh[k] = x8[k][i].d;
                auto qs = x8[k][i].qs;
                auto qh = x8[k][i].qh;
                auto signs = (const uint16_t *)x8[k][i].signs;
                helper.make2(qs+ 0, qh+0, values+0);
                helper.make2(qs+16, qh+2, values+2);
                sh.sign_4_values(signs+0, values+0);
                helper.make2(qs+32, qh+4, values+4);
                helper.make2(qs+48, qh+6, values+6);
                sh.sign_4_values(signs+8, values+4);
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    all_ls[8*ib32 + k] = (2*((x8[k][i].scales[ib32/2] >> 4*(ib32%2)) & 0xf) + 1);
                    _mm256_storeu_si256((__m256i *)block, values[ib32]);
                    auto q8 = (uint32_t *)y[ib32].qs;
                    for (int l = 0; l < 4; ++l) {
                        q8[8*l + k +  0] = block[l + 0];
                        q8[8*l + k + 32] = block[l + 4];
                    }
                }
            }
            auto vd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)dh));
            for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                auto iscales16 = _mm_loadu_si128((const __m128i *)all_ls + ib32);
                auto iscales32 = _mm256_cvtepi16_epi32(iscales16);
                auto scales = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(iscales32));
                _mm_storeu_si128((__m128i *)y[ib32].d, _mm256_cvtps_ph(scales, _MM_FROUND_TO_NEAREST_INT));
            }
            y += QK_K/32;
        }
    }
}

template <typename Dequantizer> void set_functions(std::array<mul_mat_t, IQK_MAX_NY>& funcs) {
    funcs[0] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 1>;
    funcs[1] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 2>;
    funcs[2] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 3>;
    funcs[3] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 4>;
    funcs[4] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 5>;
    funcs[5] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 6>;
    funcs[6] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 7>;
    funcs[7] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 8>;
}

} // namespace

bool iqk_set_kernels_iquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {

    if (ne00%QK_K != 0) return false;

    if (ggml_type(typeA) == GGML_TYPE_IQ2_XXS) {
        if (ggml_type(typeB) == GGML_TYPE_Q8_2_X4) {
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_2_IQ_N, DequantizerIQ2XXS, kernels);
            func16 = nullptr;
            return true;
        }
        return false;
    }

    if (ggml_type(typeA) == GGML_TYPE_IQ3_XXS) {
        if (ggml_type(typeB) == GGML_TYPE_Q8_2_X4) {
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_2_IQ_N, DequantizerIQ3XXS, kernels);
            func16 = nullptr;
            return true;
        }
        return false;
    }

    if (ggml_type(typeA) == GGML_TYPE_IQ3_S) {
        if (ggml_type(typeB) == GGML_TYPE_Q8_2_X4) {
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_2_IQ_N, DequantizerIQ3S, kernels);
            func16 = nullptr;
            return true;
        }
        return false;
    }

    if (ggml_type(typeB) != GGML_TYPE_Q8_K) {
        return false;
    }

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_IQ2_XXS:
            set_functions<DequantizerIQ2XXS>(kernels);
            break;
        case GGML_TYPE_IQ2_XS:
            set_functions<DequantizerIQ2XS>(kernels);
            break;
        case GGML_TYPE_IQ2_S:
            set_functions<DequantizerIQ2S>(kernels);
            break;
        case GGML_TYPE_IQ3_XXS:
            set_functions<DequantizerIQ3XXS>(kernels);
            break;
        case GGML_TYPE_IQ3_S:
            set_functions<DequantizerIQ3S>(kernels);
            break;
        case GGML_TYPE_IQ2_XXS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_xxs_r4_q8_k, kernels);
            func16 = mul_mat_iq2_xxs_r4_q8_k<16>;
            break;
        case GGML_TYPE_IQ2_XS_R4:
            assert (ne00 % QK_K == 0);
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_xs_r4_q8_k, kernels);
#ifndef HAVE_FANCY_SIMD
            // For some reason Zen4 does not like this particular function
            func16 = mul_mat_iq2_xs_r4_q8_k_16;
#endif
            break;
        case GGML_TYPE_IQ2_S_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_s_r4_q8_k, kernels);
            func16 = mul_mat_iq2_s_r4_q8_k_16;
            break;
        case GGML_TYPE_IQ3_XXS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq3_xxs_r4_q8_k, kernels);
            func16 = mul_mat_iq3_xxs_r4_q8_k<16>;
            break;
        case GGML_TYPE_IQ3_S_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq3_s_r4_q8_k, kernels);
            func16 = mul_mat_iq3_s_r4_q8_k<16>;
            break;
        default:
            return false;
    }

    return true;

}

bool iqk_convert_iquants_q80_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    if (n%QK_K != 0 || nrc_x%8 != 0) return false;
    switch (ggml_type(type)) {
        case GGML_TYPE_IQ2_XXS: iqk_convert_iq2_xxs_q8_0_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ3_XXS: iqk_convert_iq3_xxs_q8_0_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ3_S  : iqk_convert_iq3_s_q8_0_r8  (n, vx, bx, vy, nrc_x); break;
        default: return false;
    }
    return true;
}

#else
// --------------------------------------- __aarch64__ ---------------------------------------------

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

struct SimpleBits {
    uint8x16x4_t b1;
    uint8x16x4_t b2;
};

inline int32x4x2_t prepare_scales_8(const uint32x4_t& v1, const uint32x4_t& v2) {
    int32x4x2_t scales;
    scales.val[0] = vreinterpretq_s32_u32(vorrq_u32(vshlq_n_u32(vshrq_n_u32(v1, 28), 1), vdupq_n_u32(1)));
    scales.val[1] = vreinterpretq_s32_u32(vorrq_u32(vshlq_n_u32(vshrq_n_u32(v2, 28), 1), vdupq_n_u32(1)));
    return scales;
}

inline void apply_signs_2(uint8x16_t * b, const uint64_t * signs, uint32_t sidx) {
    auto s1 = vcombine_s8(vld1_s8((const int8_t *)(signs + ((sidx >> 0) & 127))), vld1_s8((const int8_t *)(signs + ((sidx >> 7) & 127))));
    auto s2 = vcombine_s8(vld1_s8((const int8_t *)(signs + ((sidx >>14) & 127))), vld1_s8((const int8_t *)(signs + ((sidx >>21) & 127))));
    b[0] = vreinterpretq_u8_s8(vmulq_s8(vreinterpretq_s8_u8(b[0]), s1));
    b[1] = vreinterpretq_u8_s8(vmulq_s8(vreinterpretq_s8_u8(b[1]), s2));
}

struct DequantizerIQ2XXS final : public BaseDequantizer<block_iq2_xxs> {
    DequantizerIQ2XXS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& /*q8*/, float32x4_t * /*acc*/) {
        d = 0.125f * GGML_FP16_TO_FP32(x[i].d);

        auto tmp = vld1q_u32_x4((const uint32_t *)x[i].qs);
        data.val[0] = vuzp1q_u32(tmp.val[0], tmp.val[1]);  // codebook indices for blocks 0...3
        data.val[1] = vuzp2q_u32(tmp.val[0], tmp.val[1]);  // scales and signs for blocks 0...3
        data.val[2] = vuzp1q_u32(tmp.val[2], tmp.val[3]);  // codebook indices for blocks 4...7
        data.val[3] = vuzp2q_u32(tmp.val[2], tmp.val[3]);  // scales and signs for blocks 4...7

        return prepare_scales_8(data.val[1], data.val[3]);
    }

    static inline void prepare2(uint8x16_t * b, const uint8_t * idx, const uint64_t * signs, uint32_t sidx) {
        b[0] = vreinterpretq_u8_u64(uint64x2_t{iq2xxs_grid[idx[0]], iq2xxs_grid[idx[1]]});
        b[1] = vreinterpretq_u8_u64(uint64x2_t{iq2xxs_grid[idx[2]], iq2xxs_grid[idx[3]]});
        apply_signs_2(b, signs, sidx);
    }

    inline void prepare(int /*i*/, int j) {
        const uint8_t * idx = (const uint8_t *)(data.val + 2*j);
        const uint32_t * sidx = (const uint32_t *)(data.val + 2*j+1);
        prepare2(bits.b1.val + 0, idx, keven_signs, sidx[0]); idx += 4;
        prepare2(bits.b1.val + 2, idx, keven_signs, sidx[1]); idx += 4;
        prepare2(bits.b2.val + 0, idx, keven_signs, sidx[2]); idx += 4;
        prepare2(bits.b2.val + 2, idx, keven_signs, sidx[3]);
    }

    uint32x4x4_t data;
    SimpleBits bits;

};

inline int32x4x4_t prepare_4bit_scales16(const uint8_t * sc) {
    auto aux = vld1_u8(sc);
    auto scales_l = vand_u8(aux, vdup_n_u8(0xf));
    auto scales_h = vshr_n_u8(aux, 4);
    auto aux1 = vcombine_u8(vzip1_u8(scales_l, scales_h), vzip2_u8(scales_l, scales_h));

    auto scales8 = vreinterpretq_s8_u8(vorrq_u8(vshlq_n_u8(aux1, 1), vdupq_n_u8(1)));
    int16x8x2_t scales16 = { vmovl_s8(vget_low_s8(scales8)), vmovl_s8(vget_high_s8(scales8)) };
    return make_wider(scales16);
}

struct DequantizerIQ2XS final : public BaseDequantizer<block_iq2_xs> {
    DequantizerIQ2XS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& /*q8*/, float32x4_t * /*acc*/) {
        d = 0.125f * GGML_FP16_TO_FP32(x[i].d);
        return prepare_4bit_scales16(x[i].scales);
    }

    inline static uint8x16_t make1(const uint16_t * qs) {
        auto b = vcombine_u8(vld1_u8((const uint8_t *)(iq2xs_grid + (qs[0] & 511))), vld1_u8((const uint8_t *)(iq2xs_grid + (qs[1] & 511))));
        auto s = vcombine_s8(vld1_s8((const int8_t *)(keven_signs + (qs[0] >> 9))), vld1_s8((const int8_t *)(keven_signs + (qs[1] >> 9))));
        return vreinterpretq_u8_s8(vmulq_s8(vreinterpretq_s8_u8(b), s));
    }

    inline static void make4(const uint16_t * qs, uint8x16_t * b) {
        b[0] = make1(qs + 0);
        b[1] = make1(qs + 2);
        b[2] = make1(qs + 4);
        b[3] = make1(qs + 6);
    }

    inline void prepare(int i, int j) {
        make4(x[i].qs + 16*j + 0, bits.b1.val);
        make4(x[i].qs + 16*j + 8, bits.b2.val);
    }

    SimpleBits bits;


};

struct DequantizerIQ2S final : public BaseDequantizer<block_iq2_s> {
    DequantizerIQ2S(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x4_t new_block(int i, const Q8& /*q8*/, float32x4_t * /*acc*/) {
        d = 0.125f * GGML_FP16_TO_FP32(x[i].d);
        return prepare_4bit_scales16(x[i].scales);
    }

    static inline void make4(SignHelper& sh, const uint8x16_t& signs16, const uint8_t * qs, const uint8_t * qh, uint8x16_t * b) {
        uint32_t aux32[2];
        const uint16_t * aux16 = (const uint16_t *)aux32;
        for (int k = 0; k < 2; ++k) {
            aux32[1] = (qh[k] << 4) | (qh[k] << 18);
            aux32[0] = (aux32[1] << 4) & 0x03000300;
            aux32[1] &= 0x03000300;
            b[2*k+0] = vcombine_u8(vld1_u8((const uint8_t *)(iq2s_grid + (qs[4*k+0] | aux16[0]))),
                                   vld1_u8((const uint8_t *)(iq2s_grid + (qs[4*k+1] | aux16[1]))));
            sh.apply_signs_1(b+2*k+0, signs16);

            b[2*k+1] = vcombine_u8(vld1_u8((const uint8_t *)(iq2s_grid + (qs[4*k+2] | aux16[2]))),
                                   vld1_u8((const uint8_t *)(iq2s_grid + (qs[4*k+3] | aux16[3]))));
            sh.apply_signs_1(b+2*k+1, signs16);
        }
    }

    inline void prepare(int i, int j) {

        const auto * qs = x[i].qs + 16*j;
        const auto * qh = x[i].qh + 4*j;
        const auto signs16 = vld1q_u8(qs + QK_K/8);

        sh.init();
        make4(sh, signs16, qs+0, qh+0, bits.b1.val);
        make4(sh, signs16, qs+8, qh+2, bits.b2.val);
    }

    SimpleBits bits;
    SignHelper sh;


};

struct DequantizerIQ3XXS final : public BaseDequantizer<block_iq3_xxs> {
    DequantizerIQ3XXS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& /*q8*/, float32x4_t * /*acc*/) {
        d = 0.25f * GGML_FP16_TO_FP32(x[i].d);
        gas = vld1q_u32_x2((const uint32_t *)(x[i].qs + QK_K/4));
        return prepare_scales_8(gas.val[0], gas.val[1]);
    }

    inline static void make2(const uint8_t * q3, uint32_t sidx, uint8x16_t * b) {
        b[0] = vreinterpretq_u8_u32(uint32x4_t{iq3xxs_grid[q3[0]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[3]]});
        b[1] = vreinterpretq_u8_u32(uint32x4_t{iq3xxs_grid[q3[4]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[7]]});
        apply_signs_2(b, keven_signs, sidx);
    }
    inline void prepare(int i, int j) {
        const auto * q3 = x[i].qs + 32*j;
        const auto * signs = (const uint32_t *)(gas.val + j);
        make2(q3, signs[0], bits.b1.val + 0); q3 += 8;
        make2(q3, signs[1], bits.b1.val + 2); q3 += 8;
        make2(q3, signs[2], bits.b2.val + 0); q3 += 8;
        make2(q3, signs[3], bits.b2.val + 2);
    }

    SimpleBits bits;
    uint32x4x2_t gas;

};

struct DequantizerIQ3S final : public BaseDequantizer<block_iq3_s> {
    DequantizerIQ3S(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& /*q8*/, float32x4_t * /*acc*/) {
        d = GGML_FP16_TO_FP32(x[i].d);
        uint32_t scales32[2];
        std::memcpy(scales32, x[i].scales, 4);
        scales32[1] = (((scales32[0] >> 4) & 0x0f0f0f0f) << 1) | 0x01010101;
        scales32[0] = ((scales32[0] & 0x0f0f0f0f) << 1) | 0x01010101;
        auto scales8 = vld1_u8((const uint8_t *)scales32); // 0, 2, 4, 6, 1, 3, 5, 7
        scales8 = vtbl1_u8(scales8, vreinterpret_u8_u64(vdup_n_u64(0x0703060205010400)));
        auto scales16 = vreinterpretq_s16_u16(vmovl_u8(scales8));
        int32x4x2_t scales;
        scales.val[0] = vmovl_s16(vget_low_s16(scales16));
        scales.val[1] = vmovl_s16(vget_high_s16(scales16));
        return scales;
    }

    static inline void make2(SignHelper& sh, const uint8x16_t& signs16, const uint16x8_t& idx_l, uint8_t qh,
            const int8x16_t& hshift, uint8x16_t * b) {
        auto vindex = vorrq_u16(idx_l, vandq_u16(vshlq_u16(vdupq_n_u16(qh), hshift), vdupq_n_u16(256)));
        const uint16_t * idx = (const uint16_t *)&vindex;
        b[0] = vreinterpretq_u8_u32(uint32x4_t{iq3s_grid[idx[0]], iq3s_grid[idx[1]], iq3s_grid[idx[2]], iq3s_grid[idx[3]]});
        b[1] = vreinterpretq_u8_u32(uint32x4_t{iq3s_grid[idx[4]], iq3s_grid[idx[5]], iq3s_grid[idx[6]], iq3s_grid[idx[7]]});
        sh.apply_signs_1(b+0, signs16);
        sh.apply_signs_1(b+1, signs16);
    }
    static inline void make4(SignHelper& sh, const uint8x16_t& signs16, const uint8_t * qs, const uint8_t * qh,
            const int8x16_t& hshift, uint8x16_t * b) {
        auto idx_l = vld1q_u8(qs);
        make2(sh, signs16, vmovl_u8(vget_low_u8 (idx_l)), qh[0], hshift, b+0);
        make2(sh, signs16, vmovl_u8(vget_high_u8(idx_l)), qh[1], hshift, b+2);
    }

    inline void prepare(int i, int j) {

        static const int16_t k_shift[8] = {8, 7, 6, 5, 4, 3, 2, 1};
        const auto hshift  = vld1q_s16(k_shift);

        const auto * qs = x[i].qs + 32*j;
        const auto * qh = x[i].qh + 4*j;
        const auto signs16 = vld1q_u8(x[i].signs + 16*j);

        sh.init();
        make4(sh, signs16, qs+ 0, qh+0, hshift, bits.b1.val);
        make4(sh, signs16, qs+16, qh+2, hshift, bits.b2.val);
    }

    SimpleBits bits;
    SignHelper sh;
    uint32x4x2_t gas;

};

template <int nrc_y>
static void mul_mat_iq2_xxs_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
    float32x4_t acc[nrc_y] = {};
    int32x4_t   isum[nrc_y] = {};
    int8x16_t   qx[8];
    SignHelper  sh;
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq2 = (const block_iq2_xxs_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq2[ibl].d));
            auto qs = iq2[ibl].qs;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto sas = vld1q_u8(iq2[ibl].sas + 16*ib);
                auto scale_bits = vandq_u8(sas, vdupq_n_u8(1));
                auto scales = ggml_vdotq_s32(vdupq_n_s32(1), scale_bits, vreinterpretq_s8_u32(vdupq_n_u32(0x10080402)));
                auto signs128 = vandq_u8(sas, vdupq_n_u8(254));
                signs128 = veorq_u8(signs128, vshrq_n_u8(signs128, 1));
                sh.init();
                for (int i = 0; i < 8; ++i) {
                    qx[i] = vreinterpretq_s8_u64(uint64x2_t{iq2xxs_grid[qs[2*i+0]], iq2xxs_grid[qs[2*i+1]]});
                    sh.apply_signs_1((uint8x16_t *)qx+i, signs128);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = vld1q_s8_x2(q8.y[iy][ibl].qs + 32*ib);
                    auto sumi1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[0], y.val[0]), qx[1], y.val[1]);
                    auto sumi2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[2], y.val[0]), qx[3], y.val[1]);
                    auto sumi3 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[4], y.val[0]), qx[5], y.val[1]);
                    auto sumi4 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[6], y.val[0]), qx[7], y.val[1]);
                    auto sumi12 = vpaddq_s32(sumi1, sumi2);
                    auto sumi34 = vpaddq_s32(sumi3, sumi4);
                    auto sumi = vpaddq_s32(sumi12, sumi34);
                    isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                }
                qs += 16;
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vfmaq_f32(acc[iy], vmulq_f32(d4, vdupq_n_f32(q8.scale(iy, ibl))), vcvtq_f32_s32(isum[iy]));
                isum[iy] = vdupq_n_s32(0);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, vmulq_f32(vdupq_n_f32(0.125f), acc[iy]));
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

template <int nrc_y>
static void mul_mat_iq2_xs_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
    static const uint8_t k_shuff[16] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
    auto shuff = vld1q_u8(k_shuff);
    float32x4_t acc[nrc_y] = {};
    int32x4_t   isum[2*nrc_y] = {};
    int8x16_t   qx[8];
    uint16x8x4_t scales16;
    SignHelper  sh;
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq2 = (const block_iq2_xs_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq2[ibl].d));
            auto qs = iq2[ibl].qs;
            for (int is = 0; is < 2; ++is) {
                auto scale_bits = vld1q_u8(iq2[ibl].scales + 16*is);
                auto scales1 = vandq_u8(scale_bits, vdupq_n_u8(0xf));
                auto scales2 = vshrq_n_u8(scale_bits, 4);
                scales1 = vorrq_u8(vshlq_n_u8(scales1, 1), vdupq_n_u8(1));
                scales2 = vorrq_u8(vshlq_n_u8(scales2, 1), vdupq_n_u8(1));
                auto s1 = vzip1q_u8(scales1, scales2);
                auto s2 = vzip2q_u8(scales1, scales2);
                scales16.val[0] = vmovl_u8(vget_low_u8 (s1));
                scales16.val[1] = vmovl_u8(vget_high_u8(s1));
                scales16.val[2] = vmovl_u8(vget_low_u8 (s2));
                scales16.val[3] = vmovl_u8(vget_high_u8(s2));
                for (int ib = 0; ib < QK_K/64; ++ib) {
                    auto v = vld1q_u8_x2((const uint8_t *)qs);
                    auto signs128 = vandq_u8(vqtbl2q_u8(v, shuff), vdupq_n_u8(254));
                    signs128 = veorq_u8(signs128, vshrq_n_u8(signs128, 1));
                    sh.init();
                    for (int i = 0; i < 8; ++i) {
                        qx[i] = vreinterpretq_s8_u64(uint64x2_t{iq2xs_grid[qs[2*i+0] & 511], iq2xs_grid[qs[2*i+1] & 511]});
                        sh.apply_signs_1((uint8x16_t *)qx+i, signs128);
                    }
                    auto s32_1 = vmovl_u16(vget_low_u16 (scales16.val[ib]));
                    auto s32_2 = vmovl_u16(vget_high_u16(scales16.val[ib]));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8_x2(q8.y[iy][ibl].qs + 128*is + 32*ib);
                        auto sumi1 = vpaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[0], y.val[0]), ggml_vdotq_s32(vdupq_n_s32(0), qx[1], y.val[1]));
                        auto sumi2 = vpaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[2], y.val[0]), ggml_vdotq_s32(vdupq_n_s32(0), qx[3], y.val[1]));
                        auto sumi3 = vpaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[4], y.val[0]), ggml_vdotq_s32(vdupq_n_s32(0), qx[5], y.val[1]));
                        auto sumi4 = vpaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[6], y.val[0]), ggml_vdotq_s32(vdupq_n_s32(0), qx[7], y.val[1]));
                        auto sumi12 = vpaddq_s32(sumi1, sumi2); // blocks 0,1,2,3 in rows 0,1
                        auto sumi34 = vpaddq_s32(sumi3, sumi4); // blocks 4,5,6,7 in rows 2,3
                        isum[2*iy+0] = vmlaq_s32(isum[2*iy+0], s32_1, sumi12);
                        isum[2*iy+1] = vmlaq_s32(isum[2*iy+1], s32_2, sumi34);
                    }
                    qs += 16;
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto sumi = vpaddq_s32(isum[2*iy+0], isum[2*iy+1]);
                acc[iy] = vfmaq_f32(acc[iy], vmulq_f32(d4, vdupq_n_f32(q8.scale(iy, ibl))), vcvtq_f32_s32(sumi));
                isum[2*iy] = isum[2*iy+1] = vdupq_n_s32(0);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, vmulq_f32(vdupq_n_f32(0.125f), acc[iy]));
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

template <int nrc_y>
static void mul_mat_iq2_s_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
    float32x4_t acc[nrc_y] = {};
    int32x4_t   isum[2*nrc_y] = {};
    int8x16_t   qx[8];
    uint16x8x4_t scales16;
    SignHelper  sh;
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq2 = (const block_iq2_s_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq2[ibl].d));
            auto qs = iq2[ibl].qs;
            auto qh = iq2[ibl].qh;
            for (int is = 0; is < 2; ++is) {
                auto scale_bits = vld1q_u8(iq2[ibl].scales + 16*is);
                auto scales1 = vandq_u8(scale_bits, vdupq_n_u8(0xf));
                auto scales2 = vshrq_n_u8(scale_bits, 4);
                scales1 = vorrq_u8(vshlq_n_u8(scales1, 1), vdupq_n_u8(1));
                scales2 = vorrq_u8(vshlq_n_u8(scales2, 1), vdupq_n_u8(1));
                auto s1 = vzip1q_u8(scales1, scales2);
                auto s2 = vzip2q_u8(scales1, scales2);
                scales16.val[0] = vmovl_u8(vget_low_u8 (s1));
                scales16.val[1] = vmovl_u8(vget_high_u8(s1));
                scales16.val[2] = vmovl_u8(vget_low_u8 (s2));
                scales16.val[3] = vmovl_u8(vget_high_u8(s2));
                for (int ib = 0; ib < QK_K/64; ++ib) {
                    auto signs128 = vld1q_u8(iq2[ibl].signs + 64*is + 16*ib);
                    sh.init();
                    for (int i = 0; i < 4; ++i) {
                        qx[2*i+0] = vreinterpretq_s8_u64(uint64x2_t{iq2s_grid[qs[4*i+0] | ((qh[i] << 8) & 0x300)], iq2s_grid[qs[4*i+1] | ((qh[i] << 6) & 0x300)]});
                        sh.apply_signs_1((uint8x16_t *)qx+2*i+0, signs128);
                        qx[2*i+1] = vreinterpretq_s8_u64(uint64x2_t{iq2s_grid[qs[4*i+2] | ((qh[i] << 4) & 0x300)], iq2s_grid[qs[4*i+3] | ((qh[i] << 2) & 0x300)]});
                        sh.apply_signs_1((uint8x16_t *)qx+2*i+1, signs128);
                    }
                    qs += 16; qh += 4;
                    auto s32_1 = vmovl_u16(vget_low_u16 (scales16.val[ib]));
                    auto s32_2 = vmovl_u16(vget_high_u16(scales16.val[ib]));
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        auto y = vld1q_s8_x2(q8.y[iy][ibl].qs + 128*is + 32*ib);
                        auto sumi1 = vpaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[0], y.val[0]), ggml_vdotq_s32(vdupq_n_s32(0), qx[1], y.val[1]));
                        auto sumi2 = vpaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[2], y.val[0]), ggml_vdotq_s32(vdupq_n_s32(0), qx[3], y.val[1]));
                        auto sumi3 = vpaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[4], y.val[0]), ggml_vdotq_s32(vdupq_n_s32(0), qx[5], y.val[1]));
                        auto sumi4 = vpaddq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[6], y.val[0]), ggml_vdotq_s32(vdupq_n_s32(0), qx[7], y.val[1]));
                        auto sumi12 = vpaddq_s32(sumi1, sumi2); // blocks 0,1,2,3 in rows 0,1
                        auto sumi34 = vpaddq_s32(sumi3, sumi4); // blocks 4,5,6,7 in rows 2,3
                        isum[2*iy+0] = vmlaq_s32(isum[2*iy+0], s32_1, sumi12);
                        isum[2*iy+1] = vmlaq_s32(isum[2*iy+1], s32_2, sumi34);
                    }
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto sumi = vpaddq_s32(isum[2*iy+0], isum[2*iy+1]);
                acc[iy] = vfmaq_f32(acc[iy], vmulq_f32(d4, vdupq_n_f32(q8.scale(iy, ibl))), vcvtq_f32_s32(sumi));
                isum[2*iy] = isum[2*iy+1] = vdupq_n_s32(0);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, vmulq_f32(vdupq_n_f32(0.125f), acc[iy]));
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

template <int nrc_y>
static void mul_mat_iq3_xxs_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
    float32x4_t acc[nrc_y] = {};
    int32x4_t   isum[nrc_y] = {};
    int8x16_t   qx[8];
    SignHelper  sh;
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq3 = (const block_iq3_xxs_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto d4 = vmulq_f32(vdupq_n_f32(0.25f), vcvt_f32_f16(vld1_f16((const float16_t *)iq3[ibl].d)));
            auto qs = iq3[ibl].qs;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto sas = vld1q_u8(iq3[ibl].sas + 16*ib);
                auto scale_bits = vandq_u8(sas, vdupq_n_u8(1));
                auto scales = ggml_vdotq_s32(vdupq_n_s32(1), scale_bits, vreinterpretq_s8_u32(vdupq_n_u32(0x10080402)));
                auto signs128 = vandq_u8(sas, vdupq_n_u8(254));
                signs128 = veorq_u8(signs128, vshrq_n_u8(signs128, 1));
                sh.init();
                for (int i = 0; i < 8; ++i) {
                    qx[i] = vreinterpretq_s8_u32(uint32x4_t{iq3xxs_grid[qs[4*i+0]], iq3xxs_grid[qs[4*i+1]], iq3xxs_grid[qs[4*i+2]], iq3xxs_grid[qs[4*i+3]]});
                    sh.apply_signs_1((uint8x16_t *)qx+i, signs128);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = vld1q_s8_x2(q8.y[iy][ibl].qs + 32*ib);
                    auto sumi1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[0], y.val[0]), qx[1], y.val[1]);
                    auto sumi2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[2], y.val[0]), qx[3], y.val[1]);
                    auto sumi3 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[4], y.val[0]), qx[5], y.val[1]);
                    auto sumi4 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[6], y.val[0]), qx[7], y.val[1]);
                    auto sumi12 = vpaddq_s32(sumi1, sumi2);
                    auto sumi34 = vpaddq_s32(sumi3, sumi4);
                    auto sumi = vpaddq_s32(sumi12, sumi34);
                    isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                }
                qs += 32;
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vfmaq_f32(acc[iy], vmulq_f32(d4, vdupq_n_f32(q8.scale(iy, ibl))), vcvtq_f32_s32(isum[iy]));
                isum[iy] = vdupq_n_s32(0);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

template <int nrc_y>
static void mul_mat_iq3_s_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int nbl = n / QK_K;
    float32x4_t acc[nrc_y] = {};
    int32x4_t   isum[nrc_y] = {};
    int8x16_t   qx[8];
    auto m1 = vdupq_n_u8(1);
    auto shuff = vreinterpretq_u8_u32(uint32x4_t{0xffffff00, 0xffffff01, 0xffffff02, 0xffffff03});
    uint32_t    stored_scales[8];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto iq3 = (const block_iq3_s_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto d4 = vcvt_f32_f16(vld1_f16((const float16_t *)iq3[ibl].d));
            auto qs = iq3[ibl].qs;
            auto qh = iq3[ibl].qh;
            auto scale_bits = vld1q_u8(iq3[ibl].scales);
            uint8x16x2_t scales8 = { vandq_u8(scale_bits, vdupq_n_u8(0xf)), vshrq_n_u8(scale_bits, 4) };
            scales8.val[0] = vorrq_u8(vshlq_n_u8(scales8.val[0], 1), m1);
            scales8.val[1] = vorrq_u8(vshlq_n_u8(scales8.val[1], 1), m1);
            vst1q_u8_x2((uint8_t *)stored_scales, scales8);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                auto signs128 = vld1q_u8(iq3[ibl].signs+16*ib);
                if constexpr (nrc_y == 1) {
                    auto qh32 = (const uint32_t *)qh;
                    auto idx_h = vreinterpretq_u16_u64(vshlq_u64(vreinterpretq_u64_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(qh32[0])))), int64x2_t{8, 4}));
                    union { uint16x8_t vec; uint16_t val[8]; } hidx;
                    for (int i = 0; i < 4; ++i) {
                        auto idx_l = vmovl_u8(vld1_u8(qs));
                        hidx.vec = vorrq_u16(idx_l, vandq_u16(idx_h, vdupq_n_u16(0x100))); idx_h = vshrq_n_u16(idx_h, 1);
                        qx[2*i+0] = vreinterpretq_s8_u32(uint32x4_t{iq3s_grid[hidx.val[0]], iq3s_grid[hidx.val[1]], iq3s_grid[hidx.val[2]], iq3s_grid[hidx.val[3]]});
                        auto signs = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(vandq_u8(signs128, m1), m1), m1));
                        qx[2*i+0] = vmulq_s8(qx[2*i+0], signs);
                        qx[2*i+1] = vreinterpretq_s8_u32(uint32x4_t{iq3s_grid[hidx.val[4]], iq3s_grid[hidx.val[5]], iq3s_grid[hidx.val[6]], iq3s_grid[hidx.val[7]]});
                        signs = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(vandq_u8(vshrq_n_u8(signs128, 4), m1), m1), m1));
                        qx[2*i+1] = vmulq_s8(qx[2*i+1], signs);
                        signs128 = vshrq_n_u8(signs128, 1);
                        qs += 8;
                    }
                } else {
                    for (int i = 0; i < 4; ++i) {
                        qx[2*i+0] = vreinterpretq_s8_u32(uint32x4_t{iq3s_grid[qs[0] | ((qh[0] << (8-i)) & 0x100)], iq3s_grid[qs[1] | ((qh[1] << (8-i)) & 0x100)],
                                                                    iq3s_grid[qs[2] | ((qh[2] << (8-i)) & 0x100)], iq3s_grid[qs[3] | ((qh[3] << (8-i)) & 0x100)]});
                        auto signs = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(vandq_u8(signs128, m1), m1), m1));
                        qx[2*i+0] = vmulq_s8(qx[2*i+0], signs);

                        qx[2*i+1] = vreinterpretq_s8_u32(uint32x4_t{iq3s_grid[qs[4] | ((qh[0] << (4-i)) & 0x100)], iq3s_grid[qs[5] | ((qh[1] << (4-i)) & 0x100)],
                                                                    iq3s_grid[qs[6] | ((qh[2] << (4-i)) & 0x100)], iq3s_grid[qs[7] | ((qh[3] << (4-i)) & 0x100)]});
                        signs = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(vandq_u8(vshrq_n_u8(signs128, 4), m1), m1), m1));
                        qx[2*i+1] = vmulq_s8(qx[2*i+1], signs);

                        qs += 8;
                        signs128 = vshrq_n_u8(signs128, 1);
                    }
                }
                auto scales = vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(stored_scales[ib])), shuff));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = vld1q_s8_x2(q8.y[iy][ibl].qs + 32*ib);
                    auto sumi = interleaved_dotq(qx, y);
                    isum[iy] = vmlaq_s32(isum[iy], scales, sumi);
                }
                qh += 4;
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vfmaq_f32(acc[iy], vmulq_f32(d4, vdupq_n_f32(q8.scale(iy, ibl))), vcvtq_f32_s32(isum[iy]));
                isum[iy] = vdupq_n_s32(0);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

}

bool iqk_set_kernels_iquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {

    if (ne00%QK_K != 0 || ggml_type(typeB) != GGML_TYPE_Q8_K) {
        return false;
    }

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_IQ2_XXS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ2XXS, kernels);
            break;
        case GGML_TYPE_IQ2_XS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ2XS, kernels);
            break;
        case GGML_TYPE_IQ2_S:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ2S, kernels);
            break;
        case GGML_TYPE_IQ3_XXS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ3XXS, kernels);
            break;
        case GGML_TYPE_IQ3_S:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ3S, kernels);
            break;
        case GGML_TYPE_IQ2_XXS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_xxs_r4_q8_k, kernels);
            func16 = mul_mat_iq2_xxs_r4_q8_k<16>;
            break;
        case GGML_TYPE_IQ2_XS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_xs_r4_q8_k, kernels);
            func16 = mul_mat_iq2_xs_r4_q8_k<16>;
            break;
        case GGML_TYPE_IQ2_S_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_s_r4_q8_k, kernels);
            func16 = mul_mat_iq2_s_r4_q8_k<16>;
            break;
        case GGML_TYPE_IQ3_XXS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq3_xxs_r4_q8_k, kernels);
            func16 = mul_mat_iq3_xxs_r4_q8_k<16>;
            break;
        case GGML_TYPE_IQ3_S_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq3_s_r4_q8_k, kernels);
            func16 = mul_mat_iq3_s_r4_q8_k<16>;
            break;
        default:
            return false;
    }

    return true;

}

#endif

#endif
