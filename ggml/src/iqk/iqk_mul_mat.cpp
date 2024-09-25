// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp fenc=utf-8 :vi
//
//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#if defined IQK_IMPLEMENT
#undef IQK_IMPLEMENT
#endif

#if defined __AVX2__ || defined __ARM_FEATURE_DOTPROD
#define IQK_IMPLEMENT
#endif

#include <cstring>
#include <type_traits>

#if defined IQK_IMPLEMENT

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "iqk_mul_mat.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

// clang-format off

// This matrix - vector and matrix - matrix multiplication implementation
// for k-quants, i-quants, and legacy quants, makes prompt processing
// 150-350% faster (depending on quantization type) compared to mainline llama.cpp.
// It is AVX2 and ARM_NEON only for now.
// There are also implementations for fp16/32 x fp16/32 matrix multiplications
// on AVX2 and fp16 x fp16 on ARM_NEON.
//
// Main idea is that unpacking the quants and the block scales to
// be ready for dot products with the corresponding Q8_X quants
// takes time. Hence, if we are performing a QX x Q8_X matrix matrix
// multiplication (as needed for prompt processing), we can get
// a significant speedup by reusing the unpacked QX quants and scales
// for multiplication with several Q8_X columns.
//
// For fp16/fp32 matri multiplications tiling is used to improve
// performance.

#include <utility>
#include <array>

#ifdef _MSC_VER
#define IQK_NOINLINE __declspec(noinline)
#define IQK_ALWAYS_INLINE inline
#else
#define IQK_NOINLINE __attribute__((__noinline__))
#define IQK_ALWAYS_INLINE __attribute__((__always_inline__))
#endif

namespace {

typedef struct {
    int32_t i1;
    int32_t i2;
} mmid_row_mapping;

struct DataInfo {
    float       * s;
    const char  * cy;
    size_t        bs;
    size_t        by;
    int           cur_y = 0;
    int           ne11;
    const mmid_row_mapping * row_mapping = nullptr;
    size_t        bs2 = 0;

    inline const char * src1_row(int iy) const {
        if (!row_mapping) return cy + (cur_y + iy)*by;
        int i11 = row_mapping[cur_y + iy].i1 % ne11;
        int i12 = row_mapping[cur_y + iy].i2;
        return cy + (i11 + i12*ne11)*by;
    }

    inline void store(int ix, int iy, float result) const {
        *(dst_row(iy) + ix) = result;
    }
    inline float * dst_row(int iy) const {
        if (!row_mapping) return s + (cur_y + iy)*bs;
        int i12 = row_mapping[cur_y + iy].i2;
        int i1  = row_mapping[cur_y + iy].i1;
        int i2  = i12;
        return s + i1*bs + i2*bs2;
    }
};

typedef void (*mul_mat_t)(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x);

struct MulMat {
    std::array<mul_mat_t, 8> funcs = {};
    inline void mul_mat_NxM(int n, const void * vx, size_t bx, DataInfo& info, int nrc_x, int nrc_y) {
#ifdef __aarch64__
        constexpr int k_x_step = 64; //8192; // Tiling does not seem to help on my M2 Max (but difference to tiling is small)
#else
        constexpr int k_x_step = 64; // This works best on my Ryzen-7950X (but differences to other tile size are small)
#endif
        int ny = funcs.size();
        while (!funcs[ny-1] && ny > 0) --ny;
        int n_step = (nrc_y - info.cur_y)/ny;
        if (n_step > 0) {
            for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                auto this_info = info;
                this_info.s += ix;
                int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                for (int iy = 0; iy < n_step; ++iy) {
                    funcs[ny-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                    this_info.cur_y += ny;
                }
            }
            info.cur_y += ny * n_step;
        }
        int n_left = nrc_y - info.cur_y;
        if (n_left > 0) {
            funcs[n_left-1](n, vx, bx, info, nrc_x);
        }
    }
    static bool prepare(int typeA, int typeB, int ne00, MulMat& mm, int Ny);
private:
    template <typename Dequantizer> static void set_functions(MulMat& m);
};

}

bool iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth) {

    MulMat mm;
    if (!MulMat::prepare(typeA, typeB, ne00, mm, Ny)) {
        return false;
    }

    auto row_size_qx = strideA*ggml_type_size(ggml_type(typeA));
    auto row_size_qy = strideB*ggml_type_size(ggml_type(typeB));

    auto nrc_x = (Nx + nth - 1)/nth;
    auto first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx) nrc_x = Nx - first_x;

    DataInfo info{C + first_x, (const char *)B, (size_t)stride_C, row_size_qy, 0, 1, nullptr, 0};

    mm.mul_mat_NxM(ne00, (const char *)A + row_size_qx*first_x, row_size_qx, info, nrc_x, Ny);

    return true;
}

bool iqk_mul_mat_moe(long Nx, long Ny, long ne00, int ne11,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long nb1, long nb2, const void * vrow_mapping, int ith, int nth) {
    const mmid_row_mapping * row_mapping = (const mmid_row_mapping *)vrow_mapping;
    assert(row_mapping != nullptr);

    MulMat mm;
    if (!MulMat::prepare(typeA, typeB, ne00, mm, Ny)) {
        return false;
    }
    auto row_size_qx = strideA*ggml_type_size(ggml_type(typeA));
    auto row_size_qy = strideB*ggml_type_size(ggml_type(typeB));
    int nrc_x = (Nx + nth - 1)/nth;
    int first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx) nrc_x = Nx - first_x;
    DataInfo info{C + first_x, (const char *)B, nb1/sizeof(float),
        row_size_qy, 0, ne11, row_mapping, nb2/sizeof(float)};
    mm.mul_mat_NxM(ne00, (const char *)A + row_size_qx*first_x, row_size_qx, info, nrc_x, Ny);
    return true;
}

namespace {

inline void make_q4_scales(const uint8_t * scales8, uint32_t * aux32) {
    const uint16_t * scales = (const uint16_t *)scales8;
    const uint32_t a0 = scales[0] | (scales[1] << 16);
    const uint32_t a1 = scales[2] | (scales[3] << 16);
    const uint32_t a2 = scales[4] | (scales[5] << 16);
    aux32[3] = ((a2 >> 4) & 0x0f0f0f0f) | ((a1 >> 2) & 0x30303030);
    aux32[1] = ((a2 >> 0) & 0x0f0f0f0f) | ((a0 >> 2) & 0x30303030);
    aux32[2] = a1 & 0x3f3f3f3f;
    aux32[0] = a0 & 0x3f3f3f3f;
}

#ifndef HAVE_FANCY_SIMD
const uint64_t keven_signs[128] = {
    0x0101010101010101, 0xff010101010101ff, 0xff0101010101ff01, 0x010101010101ffff,
    0xff01010101ff0101, 0x0101010101ff01ff, 0x0101010101ffff01, 0xff01010101ffffff,
    0xff010101ff010101, 0x01010101ff0101ff, 0x01010101ff01ff01, 0xff010101ff01ffff,
    0x01010101ffff0101, 0xff010101ffff01ff, 0xff010101ffffff01, 0x01010101ffffffff,
    0xff0101ff01010101, 0x010101ff010101ff, 0x010101ff0101ff01, 0xff0101ff0101ffff,
    0x010101ff01ff0101, 0xff0101ff01ff01ff, 0xff0101ff01ffff01, 0x010101ff01ffffff,
    0x010101ffff010101, 0xff0101ffff0101ff, 0xff0101ffff01ff01, 0x010101ffff01ffff,
    0xff0101ffffff0101, 0x010101ffffff01ff, 0x010101ffffffff01, 0xff0101ffffffffff,
    0xff01ff0101010101, 0x0101ff01010101ff, 0x0101ff010101ff01, 0xff01ff010101ffff,
    0x0101ff0101ff0101, 0xff01ff0101ff01ff, 0xff01ff0101ffff01, 0x0101ff0101ffffff,
    0x0101ff01ff010101, 0xff01ff01ff0101ff, 0xff01ff01ff01ff01, 0x0101ff01ff01ffff,
    0xff01ff01ffff0101, 0x0101ff01ffff01ff, 0x0101ff01ffffff01, 0xff01ff01ffffffff,
    0x0101ffff01010101, 0xff01ffff010101ff, 0xff01ffff0101ff01, 0x0101ffff0101ffff,
    0xff01ffff01ff0101, 0x0101ffff01ff01ff, 0x0101ffff01ffff01, 0xff01ffff01ffffff,
    0xff01ffffff010101, 0x0101ffffff0101ff, 0x0101ffffff01ff01, 0xff01ffffff01ffff,
    0x0101ffffffff0101, 0xff01ffffffff01ff, 0xff01ffffffffff01, 0x0101ffffffffffff,
    0xffff010101010101, 0x01ff0101010101ff, 0x01ff01010101ff01, 0xffff01010101ffff,
    0x01ff010101ff0101, 0xffff010101ff01ff, 0xffff010101ffff01, 0x01ff010101ffffff,
    0x01ff0101ff010101, 0xffff0101ff0101ff, 0xffff0101ff01ff01, 0x01ff0101ff01ffff,
    0xffff0101ffff0101, 0x01ff0101ffff01ff, 0x01ff0101ffffff01, 0xffff0101ffffffff,
    0x01ff01ff01010101, 0xffff01ff010101ff, 0xffff01ff0101ff01, 0x01ff01ff0101ffff,
    0xffff01ff01ff0101, 0x01ff01ff01ff01ff, 0x01ff01ff01ffff01, 0xffff01ff01ffffff,
    0xffff01ffff010101, 0x01ff01ffff0101ff, 0x01ff01ffff01ff01, 0xffff01ffff01ffff,
    0x01ff01ffffff0101, 0xffff01ffffff01ff, 0xffff01ffffffff01, 0x01ff01ffffffffff,
    0x01ffff0101010101, 0xffffff01010101ff, 0xffffff010101ff01, 0x01ffff010101ffff,
    0xffffff0101ff0101, 0x01ffff0101ff01ff, 0x01ffff0101ffff01, 0xffffff0101ffffff,
    0xffffff01ff010101, 0x01ffff01ff0101ff, 0x01ffff01ff01ff01, 0xffffff01ff01ffff,
    0x01ffff01ffff0101, 0xffffff01ffff01ff, 0xffffff01ffffff01, 0x01ffff01ffffffff,
    0xffffffff01010101, 0x01ffffff010101ff, 0x01ffffff0101ff01, 0xffffffff0101ffff,
    0x01ffffff01ff0101, 0xffffffff01ff01ff, 0xffffffff01ffff01, 0x01ffffff01ffffff,
    0x01ffffffff010101, 0xffffffffff0101ff, 0xffffffffff01ff01, 0x01ffffffff01ffff,
    0xffffffffffff0101, 0x01ffffffffff01ff, 0x01ffffffffffff01, 0xffffffffffffffff,
};
#endif

}

#if defined __x86_64__

#if defined HAVE_FANCY_SIMD
    #undef HAVE_FANCY_SIMD
#endif
#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
    #define HAVE_FANCY_SIMD
#endif

namespace {

inline float hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
inline float hsum_float_8(__m256 x) {
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}
inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
inline float hmax_float_8(__m256 x) {
    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    max4 = _mm_max_ps( max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4));
    return  _mm_cvtss_f32(max4);
}
IQK_ALWAYS_INLINE __m256 hsum_float_8x8(__m256 * accm) {
    for (int i = 0; i < 4; ++i) {
        accm[i] = _mm256_set_m128(_mm_add_ps(_mm256_castps256_ps128(accm[i+4]), _mm256_extractf128_ps(accm[i+4], 1)),
                                  _mm_add_ps(_mm256_castps256_ps128(accm[i+0]), _mm256_extractf128_ps(accm[i+0], 1)));
    }
    for (int i = 0; i < 2; ++i) accm[i] = _mm256_add_ps(_mm256_unpacklo_ps(accm[i], accm[i+2]), _mm256_unpackhi_ps(accm[i], accm[i+2]));
    return _mm256_add_ps(_mm256_unpacklo_ps(accm[0], accm[1]), _mm256_unpackhi_ps(accm[0], accm[1]));
}
#ifdef HAVE_FANCY_SIMD
IQK_ALWAYS_INLINE void store_8(int ix, __m256 * accm, const DataInfo& info) {
    union { __m256 vec; float val[8]; } h;
    h.vec = hsum_float_8x8(accm);
    for (int iy = 0; iy < 8; ++iy) info.store(ix, iy, h.val[iy]);
}
#else
// Somehow on the AVX2 system that I have available (Ryzen-5975WX), the store_8 version above
// and the commented out store_8 version below are slower than this.
IQK_ALWAYS_INLINE void store_8(int ix, __m256 * accm, const DataInfo& info) {
    for (int iy = 0; iy < 8; ++iy) info.store(ix, iy, hsum_float_8(accm[iy]));
}
//IQK_ALWAYS_INLINE __m128 hsum_float_4x4(__m128 * a) {
//    for (int i = 0; i < 2; ++i) a[i] = _mm_add_ps(_mm_unpacklo_ps(a[i], a[i+2]), _mm_unpackhi_ps(a[i], a[i+2]));
//    return _mm_add_ps(_mm_unpacklo_ps(a[0], a[1]), _mm_unpackhi_ps(a[0], a[1]));
//}
//IQK_ALWAYS_INLINE void store_8(int ix, __m256 * accm, const DataInfo& info) {
//    union { __m128 vec; float val[4]; } h;
//    __m128 a[4];
//    for (int i = 0; i < 4; ++i) a[i] = _mm_add_ps(_mm256_castps256_ps128(accm[i]), _mm256_extractf128_ps(accm[i], 1));
//    h.vec = hsum_float_4x4(a);
//    for (int iy = 0; iy < 4; ++iy) info.store(ix, iy, h.val[iy]);
//    for (int i = 0; i < 4; ++i) a[i] = _mm_add_ps(_mm256_castps256_ps128(accm[i+4]), _mm256_extractf128_ps(accm[i+4], 1));
//    h.vec = hsum_float_4x4(a);
//    for (int iy = 0; iy < 4; ++iy) info.store(ix, iy+4, h.val[iy]);
#endif


#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

template <int nrc, typename block_q8 = block_q8_K> struct Q8 {

    constexpr static int nrc_y = nrc;

    Q8(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8 *)info.src1_row(iy);
    }

#ifdef HAVE_FANCY_SIMD
    inline __m512i load_quants64(int iy, int i, int j) const { return _mm512_loadu_si512((const __m512i*)y[iy][i].qs + j); }
#endif
    inline __m256i load_quants(int iy, int i, int j) const { return _mm256_loadu_si256((const __m256i*)y[iy][i].qs + j); }
    inline __m256i load_bsums(int iy, int i) const { return _mm256_loadu_si256((const __m256i*)y[iy][i].bsums); }
    inline float scale(int iy, int i) const { return y[iy][i].d; }

    const block_q8 * y[nrc_y];
};

struct Scales8KBase {
    template <typename Q8>
    inline void accum_mins(const __m128i& mins128, const Q8& q8, int i, float c, __m256 * accd) const {
        const __m256i mins = MM256_SET_M128I(_mm_shuffle_epi8(mins128, shuffles[1]), _mm_shuffle_epi8(mins128, shuffles[0]));
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i q8s = q8.load_bsums(iy, i);
            const __m256i prod = _mm256_madd_epi16(mins, q8s);
            accd[iy] = _mm256_fmadd_ps(_mm256_set1_ps(c*q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accd[iy]);
        }
    }
    inline __m256i shuffle(__m128i mins) const {
        return MM256_SET_M128I(_mm_shuffle_epi8(mins, shuffles[1]), _mm_shuffle_epi8(mins, shuffles[0]));
    }
    const __m128i shuffles[2] = {_mm_set_epi32(0x07060706, 0x05040504, 0x03020302, 0x01000100),
                                 _mm_set_epi32(0x0f0e0f0e, 0x0d0c0d0c, 0x0b0a0b0a, 0x09080908)};
};

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

template <typename Block>
struct BaseDequantizer {
    BaseDequantizer(const void * vx, size_t bx) : vx(vx), bx(bx) {}
    inline void new_row(int ix) {
        x = (const Block *)((const char *)vx + bx*ix);
    }

    const void *  vx;
    const size_t  bx;
    const Block * x;

    float d;
};

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

template <typename Q8, typename Bits>
inline void multiply_add(const Bits& bits, const __m256i * scales, int j, int i, const Q8& q8, __m256i * sumi) {
    if (j == 0) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            sumi[iy] = _mm256_dpwssd_epi32(_mm256_setzero_si256(), scales[0], _mm256_maddubs_epi16(bits.values[0], q8.load_quants(iy, i, 0)));
            sumi[iy] = _mm256_dpwssd_epi32(sumi[iy], scales[1], _mm256_maddubs_epi16(bits.values[1], q8.load_quants(iy, i, 1)));
            sumi[iy] = _mm256_dpwssd_epi32(sumi[iy], scales[2], _mm256_maddubs_epi16(bits.values[2], q8.load_quants(iy, i, 2)));
            sumi[iy] = _mm256_dpwssd_epi32(sumi[iy], scales[3], _mm256_maddubs_epi16(bits.values[3], q8.load_quants(iy, i, 3)));
        }
#else
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(bits.values[0], q8.load_quants(iy, i, 0)));
            const __m256i p2 = _mm256_madd_epi16(scales[1], _mm256_maddubs_epi16(bits.values[1], q8.load_quants(iy, i, 1)));
            const __m256i p3 = _mm256_madd_epi16(scales[2], _mm256_maddubs_epi16(bits.values[2], q8.load_quants(iy, i, 2)));
            const __m256i p4 = _mm256_madd_epi16(scales[3], _mm256_maddubs_epi16(bits.values[3], q8.load_quants(iy, i, 3)));
            sumi[iy] = _mm256_add_epi32(_mm256_add_epi32(p1, p3), _mm256_add_epi32(p2, p4));
        }
#endif
    } else {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            sumi[iy] = _mm256_dpwssd_epi32(sumi[iy], scales[0], _mm256_maddubs_epi16(bits.values[0], q8.load_quants(iy, i, 4)));
            sumi[iy] = _mm256_dpwssd_epi32(sumi[iy], scales[1], _mm256_maddubs_epi16(bits.values[1], q8.load_quants(iy, i, 5)));
            sumi[iy] = _mm256_dpwssd_epi32(sumi[iy], scales[2], _mm256_maddubs_epi16(bits.values[2], q8.load_quants(iy, i, 6)));
            sumi[iy] = _mm256_dpwssd_epi32(sumi[iy], scales[3], _mm256_maddubs_epi16(bits.values[3], q8.load_quants(iy, i, 7)));
        }
#else
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(bits.values[0], q8.load_quants(iy, i, 4)));
            const __m256i p2 = _mm256_madd_epi16(scales[1], _mm256_maddubs_epi16(bits.values[1], q8.load_quants(iy, i, 5)));
            const __m256i p3 = _mm256_madd_epi16(scales[2], _mm256_maddubs_epi16(bits.values[2], q8.load_quants(iy, i, 6)));
            const __m256i p4 = _mm256_madd_epi16(scales[3], _mm256_maddubs_epi16(bits.values[3], q8.load_quants(iy, i, 7)));
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p1, p3));
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p2, p4));
        }
#endif
    }
}

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
    inline void sign_4_values(const uint16_t * sign_bits, __m256i * values) const {
#ifdef HAVE_FANCY_SIMD
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

struct SimpleBits {
    __m256i values[4];
};

#ifdef HAVE_FANCY_SIMD
//====================================== Zen4 ==================================================

struct BlockPermuter {
    const __m512i permute1 = _mm512_set_epi64(11, 10,  9,  8, 3, 2, 1, 0);
    const __m512i permute2 = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
};

struct Q4Bits {
    inline void prepare(const uint8_t * q4) {
        auto q4bits = _mm512_loadu_si512((const __m512i*)q4 + 0);
        auto tmp1 = _mm512_and_si512(q4bits, ml);
        auto tmp2 = _mm512_and_si512(_mm512_srli_epi16(q4bits, 4), ml);
        values[0] = _mm512_permutex2var_epi64(tmp1, perm.permute1, tmp2);
        values[1] = _mm512_permutex2var_epi64(tmp1, perm.permute2, tmp2);
        q4bits = _mm512_loadu_si512((const __m512i*)q4 + 1);
        tmp1 = _mm512_and_si512(q4bits, ml);
        tmp2 = _mm512_and_si512(_mm512_srli_epi16(q4bits, 4), ml);
        values[2] = _mm512_permutex2var_epi64(tmp1, perm.permute1, tmp2);
        values[3] = _mm512_permutex2var_epi64(tmp1, perm.permute2, tmp2);
    }
    inline void prepare64(const uint8_t * q4) {
        auto q4bits = _mm512_loadu_si512((const __m512i*)q4 + 0);
        values[0] = _mm512_and_si512(q4bits, ml);
        values[1] = _mm512_and_si512(_mm512_srli_epi16(q4bits, 4), ml);
        q4bits = _mm512_loadu_si512((const __m512i*)q4 + 1);
        values[2] = _mm512_and_si512(q4bits, ml);
        values[3] = _mm512_and_si512(_mm512_srli_epi16(q4bits, 4), ml);
    }
    __m512i values[4];
    const __m512i ml = _mm512_set1_epi8(0xf);
    BlockPermuter perm;
};

struct Q2Bits {
    inline void prepare(const uint8_t * q2) {

        auto q2bits = _mm512_loadu_si512((const __m512i*)q2);
        auto tmp = _mm512_srli_epi16(q2bits, 2);

        values[0] = _mm512_permutex2var_epi64(q2bits, perm.permute1, tmp);
        values[2] = _mm512_permutex2var_epi64(q2bits, perm.permute2, tmp);
        values[1] = _mm512_and_si512(_mm512_srli_epi16(values[0], 4), ml);
        values[3] = _mm512_and_si512(_mm512_srli_epi16(values[2], 4), ml);
        values[0] = _mm512_and_si512(values[0], ml);
        values[2] = _mm512_and_si512(values[2], ml);
    }
    __m512i values[4];
    const __m512i ml = _mm512_set1_epi8(0x03);
    BlockPermuter perm;
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

__m512i load_iq4nl_values_512() {
    static const uint8_t kvalues_iq4nl[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};
    auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
    auto val256 = MM256_SET_M128I(val128, val128);
    return _mm512_inserti32x8(_mm512_castsi256_si512(val256), val256, 1);
}


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
        scales[0] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[0]);
        scales[1] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[1]);
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
    Scales8K s8k;
    ScaleIQ4XS siq4;
    const __m512i values;
    const __m512i permute1 = _mm512_set_epi64(11, 10, 3, 2,  9,  8, 1, 0);
    const __m512i permute2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
};

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

struct DequantizerIQ2TN final : public BaseDequantizer<block_iq2_tn> {
    DequantizerIQ2TN(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}
    template <typename Q8>
    inline void new_block(int i, [[maybe_unused]] const Q8& q8, [[maybe_unused]] __m256 * accm, [[maybe_unused]] __m512i * scales) {
        new_block(i);
    }
    inline void new_block(int i) {
        d = GGML_FP16_TO_FP32(x[i].d);
        bits.prepare(x[i].qs);
    }
    Q2Bits bits;
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
        return _mm_add_epi8(_mm_slli_epi16(scl, 1), m15);
    }
    Q2Bits bits;
    const IQXKScales iqxk;

    const __m512i values;
    const __m128i m15 = _mm_set1_epi8(-15);
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

template <typename Q8>
inline void compute_block_iq2tn(int iy, int i, float d, const Q8& q8, const __m512i * values, __m512 * accd) {
    auto sumi_scales = _mm256_madd_epi16(_mm256_set1_epi16(-1), q8.load_bsums(iy, i));
    auto sumi = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_dpbusd_epi32(
                                        _mm512_inserti32x8(_mm512_setzero_si512(), sumi_scales, 0),
                                        values[0], q8.load_quants64(iy, i, 0)), values[1], q8.load_quants64(iy, i, 1)),
                                        values[2], q8.load_quants64(iy, i, 2)), values[3], q8.load_quants64(iy, i, 3));
    accd[iy] = _mm512_fmadd_ps(_mm512_set1_ps(d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi), accd[iy]);
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
                if constexpr (std::is_same_v<Dequantizer, DequantizerIQ2TN>) {
                    auto sumi_scales = _mm256_madd_epi16(_mm256_set1_epi16(-1), q8.load_bsums(iy, i));
                    auto sumi = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_dpbusd_epi32(
                                        _mm512_inserti32x8(_mm512_setzero_si512(), sumi_scales, 0),
                                        deq.bits.values[0], q8.load_quants64(iy, i, 0)), deq.bits.values[1], q8.load_quants64(iy, i, 1)),
                                        deq.bits.values[2], q8.load_quants64(iy, i, 2)), deq.bits.values[3], q8.load_quants64(iy, i, 3));
                    accd[iy] = _mm512_fmadd_ps(_mm512_set1_ps(deq.d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi), accd[iy]);
                } else {
                    const __m512i p1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), deq.bits.values[0], q8.load_quants64(iy, i, 0));
                    const __m512i p2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), deq.bits.values[1], q8.load_quants64(iy, i, 1));
                    const __m512i p3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), deq.bits.values[2], q8.load_quants64(iy, i, 2));
                    const __m512i p4 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), deq.bits.values[3], q8.load_quants64(iy, i, 3));
                    auto sumi = _mm512_dpwssd_epi32(_mm512_setzero_si512(), scales[0], _mm512_packs_epi32(p1, p2));
                    sumi = _mm512_dpwssd_epi32(sumi, scales[1], _mm512_packs_epi32(p3, p4));
                    accd[iy] = _mm512_fmadd_ps(_mm512_set1_ps(deq.d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi), accd[iy]);
                }
            }

        }

        if constexpr (nrc_y == 8) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                accm[iy] = _mm256_add_ps(accm[iy], _mm256_add_ps(_mm512_castps512_ps256(accd[iy]), _mm512_extractf32x8_ps(accd[iy], 1)));
            }
            store_8(ix, accm, info);
        }
        else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto sum256 = _mm256_add_ps(_mm512_castps512_ps256(accd[iy]), _mm512_extractf32x8_ps(accd[iy], 1));
                info.store(ix, iy, hsum_float_8(_mm256_add_ps(accm[iy], sum256)));
            }
        }

    }
}

template <int nrc_y>
static void mul_mat_iq2tn_q8_K_AVX512(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y> q8(info);

    DequantizerIQ2TN deq1(vx, bx), deq2(vx, bx);

    __m512  accd[2*nrc_y];

    for (int ix = 0; ix < nrc_x; ix += 2) {

        for (int iy = 0; iy < 2*nrc_y; ++iy) accd[iy] = _mm512_setzero_ps();

        deq1.new_row(ix+0);
        deq2.new_row(ix+1);

        for (int i = 0; i < nb; ++i) {

            deq1.new_block(i);
            deq2.new_block(i);
            float d = 0.5f*(deq1.d + deq2.d); // The scale is supposed to be per per tensor, so we can use the same scale for both rows

            for (int iy = 0; iy < nrc_y; ++iy) {
                auto sumi_scales_256 = _mm256_madd_epi16(_mm256_set1_epi16(-1), q8.load_bsums(iy, i));
                auto sumi_scales_512 = _mm512_inserti32x8(_mm512_setzero_si512(), sumi_scales_256, 0);
                auto q8q = q8.load_quants64(iy, i, 0);
                auto sumi_1 = _mm512_dpbusd_epi32(sumi_scales_512, deq1.bits.values[0], q8q);
                auto sumi_2 = _mm512_dpbusd_epi32(sumi_scales_512, deq2.bits.values[0], q8q);
                q8q = q8.load_quants64(iy, i, 1);
                sumi_1 = _mm512_dpbusd_epi32(sumi_1, deq1.bits.values[1], q8q);
                sumi_2 = _mm512_dpbusd_epi32(sumi_2, deq2.bits.values[1], q8q);
                q8q = q8.load_quants64(iy, i, 2);
                sumi_1 = _mm512_dpbusd_epi32(sumi_1, deq1.bits.values[2], q8q);
                sumi_2 = _mm512_dpbusd_epi32(sumi_2, deq2.bits.values[2], q8q);
                q8q = q8.load_quants64(iy, i, 3);
                sumi_1 = _mm512_dpbusd_epi32(sumi_1, deq1.bits.values[3], q8q);
                sumi_2 = _mm512_dpbusd_epi32(sumi_2, deq2.bits.values[3], q8q);
                // The scale is supposed to be per per tensor, so we can use the same scale
                auto vd = _mm512_set1_ps(d*q8.scale(iy, i));
                accd[iy+    0] = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(sumi_1), accd[iy+    0]);
                accd[iy+nrc_y] = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(sumi_2), accd[iy+nrc_y]);
                // Leaving this here just in case ternary models start using per row scales
                //accd[2*iy+0] = _mm512_fmadd_ps(_mm512_set1_ps(deq1.d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi_1), accd[2*iy+0]);
                //accd[2*iy+1] = _mm512_fmadd_ps(_mm512_set1_ps(deq2.d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi_2), accd[2*iy+1]);
            }

        }

        if constexpr (nrc_y == 8) {
            __m256 sums[8];
            for (int iy = 0; iy < nrc_y; ++iy) {
                sums[iy] = _mm256_add_ps(_mm512_castps512_ps256(accd[iy]), _mm512_extractf32x8_ps(accd[iy], 1));
            }
            store_8(ix+0, sums, info);
            for (int iy = 0; iy < nrc_y; ++iy) {
                sums[iy] = _mm256_add_ps(_mm512_castps512_ps256(accd[iy+nrc_y]), _mm512_extractf32x8_ps(accd[iy+nrc_y], 1));
            }
            store_8(ix+1, sums, info);
        } else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                info.store(ix+0, iy, _mm512_reduce_add_ps(accd[iy+    0]));
                info.store(ix+1, iy, _mm512_reduce_add_ps(accd[iy+nrc_y]));
            }
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

        if constexpr (nrc_y == 8) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto sum256 = _mm256_add_ps(_mm512_castps512_ps256(accd[iy]), _mm512_extractf32x8_ps(accd[iy], 1));
                accm[iy] = _mm256_add_ps(accm[iy], sum256);
            }
            store_8(ix, accm, info);
        }
        else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto sum256 = _mm256_add_ps(_mm512_castps512_ps256(accd[iy]), _mm512_extractf32x8_ps(accd[iy], 1));
                info.store(ix, iy, hsum_float_8(_mm256_add_ps(accm[iy], sum256)));
            }
        }

    }
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

    __m256 sums[8];

    int ks = 0;
    for (int ix = 0; ix < nrc_x; ++ix) {

        auto accd = _mm512_setzero_ps();
        auto accm = _mm256_setzero_ps();

        for (int kx = 0; kx < k_nx; ++kx) deq[kx]->new_row(ix);

        for (int i = 0; i < nb/k_nx; ++i) {

            for (int kx = 0; kx < k_nx; ++kx) deq[kx]->new_block(k_nx*i+kx, q8, &accm, scales+2*kx);

            if constexpr (std::is_same_v<Dequantizer, DequantizerIQ2TN>) {
                for (int kx = 0; kx < k_nx; ++kx) {
                    compute_block_iq2tn(0, k_nx*i+kx, deq[kx]->d, q8, deq[kx]->bits.values, &accd);
                }
            } else {
                for (int kx = 0; kx < k_nx; ++kx) {
                    compute_block(0, k_nx*i+kx, deq[kx]->d, q8, deq[kx]->bits.values, scales+2*kx, &accd);
                }
            }

        }
        if (2*(nb/2) < nb) {
            int i0 = 2*(nb/2);
            deq[0]->new_block(i0, q8, &accm, scales);
            if constexpr (std::is_same_v<Dequantizer, DequantizerIQ2TN>) {
                compute_block_iq2tn(0, i0, deq[0]->d, q8, deq[0]->bits.values, &accd);
            } else {
                compute_block(0, i0, deq[0]->d, q8, deq[0]->bits.values, scales, &accd);
            }
        }

        if constexpr (std::is_same_v<Dequantizer, DequantizerIQ2TN>) {
            sums[ks++] = _mm256_add_ps(_mm512_castps512_ps256(accd), _mm512_extractf32x8_ps(accd, 1));
            //info.store(ix, 0, _mm512_reduce_add_ps(accd));
        } else {
            auto sum256 = _mm256_add_ps(_mm512_castps512_ps256(accd), _mm512_extractf32x8_ps(accd, 1));
            sums[ks++] = _mm256_add_ps(accm, sum256);
            //info.store(ix, 0, hsum_float_8(_mm256_add_ps(accm, sum256)));
        }
        if (ks == 8) {
            _mm256_storeu_ps(info.dst_row(0) + ix - 7, hsum_float_8x8(sums));
            ks = 0;
        }
    }
    if (ks > 0) {
        for (int ix = 0; ix < ks; ++ix) info.store(ix, 0, hsum_float_8(sums[ix]));
    }
}

#else
// ===================================== Vanilla AVX2 =====================================

struct Q4Bits {
    inline void prepare(const uint8_t * q4, int j) {
        auto q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+0);
        values[0] = _mm256_and_si256(q4bits, ml);
        values[1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
        q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+1);
        values[2] = _mm256_and_si256(q4bits, ml);
        values[3] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
    }
    inline void prepare64(const uint8_t * q4, int j) {
        auto q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+0);
        values[0] = _mm256_and_si256(q4bits, ml);
        values[2] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
        q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+1);
        values[1] = _mm256_and_si256(q4bits, ml);
        values[3] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
    }
    inline void prepare16(const uint8_t * q4, int j) {
        values[0] = dequant16(q4 + 64*j +  0);
        values[1] = dequant16(q4 + 64*j + 16);
        values[2] = dequant16(q4 + 64*j + 32);
        values[3] = dequant16(q4 + 64*j + 48);
    }
    inline __m256i dequant16(const uint8_t * qs) const {
        const __m128i aux128 = _mm_loadu_si128((const __m128i *)qs);
        const __m256i aux256 = MM256_SET_M128I(_mm_srli_epi16(aux128, 4), aux128);
        return _mm256_and_si256(ml, aux256);
    }
    __m256i values[4];
    const __m256i ml = _mm256_set1_epi8(0xf);
};

struct Q2Bits {
    inline void prepare(const uint8_t * q2, int j) {
        auto q2bits = _mm256_loadu_si256((const __m256i *)q2 + j);
        values[0] = _mm256_and_si256(q2bits, ml);
        values[1] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), ml);
        values[2] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), ml);
        values[3] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), ml);
    }
    __m256i values[4];
    const __m256i ml = _mm256_set1_epi8(0x03);
};

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

__m256i load_iq4nl_values() {
    static const uint8_t kvalues_iq4nl[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};
    auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
    return MM256_SET_M128I(val128, val128);
}

struct DequantizerIQ4XS final : public BaseDequantizer<block_iq4_xs> {
    DequantizerIQ4XS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_iq4nl_values()) {}
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

struct IQXKScales {
    IQXKScales(int8_t shift, int8_t min_val) : min(_mm256_set1_epi16(min_val)), eshift(_mm_set1_epi8(shift)) {}
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m128i scales8, const Q8& q8, __m256 * accm, __m256i * scales) const {
        auto scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales8, hshuff));
        process(i, d, extra, scales16, q8, accm, scales);
        //auto extra128 = _mm_set1_epi16(extra);
        //extra128 = _mm_cmpeq_epi8(_mm_and_si128(extra128, emask), emask);
        //extra128 = _mm_and_si128(extra128, eshift);
        //extra128 = _mm_shuffle_epi8(extra128, eshuffle);
        //auto scales_s = _mm256_mullo_epi16(scales16, _mm256_add_epi16(min, _mm256_cvtepi8_epi16(extra128)));
        //for (int iy = 0; iy < Q8::nrc_y; ++iy) {
        //    const __m256i prod  = _mm256_madd_epi16(scales_s, q8.load_bsums(iy, i));
        //    accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accm[iy]);
        //}
        //prepare_scales_16(scales16, scales);
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
        return _mm_add_epi8(_mm_slli_epi16(scl, 1), m15);
    }

    Q2Bits bits;
    const IQXKScales iqxk;
    const __m256i values;
    const __m128i m15      = _mm_set1_epi8(-15);
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

struct DequantizerIQ4K final : public BaseDequantizer<block_iq4_k> {
    DequantizerIQ4K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(4, -128), values(load_iq4nl_values()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        iqxk.process(i, d, x[i].extra, make_scales(x[i].scales_l, (const uint16_t *)x[i].scales_h), q8, accm, scales);
    }
    inline void prepare(int i, int j) {
        bits.prepare16(x[i].qs, j);
        bits.values[0] = _mm256_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values, bits.values[3]);
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

    Q4Bits bits;
    const IQXKScales iqxk;
    const __m256i values;
    const __m128i maskl    = _mm_set1_epi8(0xf);
    const __m128i maskh    = _mm_set1_epi8(0x30);
    const __m128i m32      = _mm_set1_epi8(-32);
};

struct DequantizerIQ5K final : public BaseDequantizer<block_iq5_k> {
    DequantizerIQ5K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(2, -128) { load_values(values); }
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
    const IQXKScales iqxk;
    __m256i hbits;
    __m256i values[2];
    const __m128i maskl    = _mm_set1_epi8(0xf);
    const __m128i maskh    = _mm_set1_epi8(0x30);
    const __m128i m32      = _mm_set1_epi8(-32);
    const __m256i mh       = _mm256_set1_epi8(-128); // to avoid stupid warning about 0x80 overflowing
};

struct DequantizerIQ6K final : public BaseDequantizer<block_iq6_k> {
    DequantizerIQ6K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(1, -128) { load_values(values); }
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
        auto mask1 = _mm256_andnot_si256(_mm256_or_si256(mask4, _mm256_or_si256(mask2, mask3)), _mm256_set1_epi8(0xff));
        return _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(mask1, _mm256_shuffle_epi8(values[0], l)),
                                               _mm256_and_si256(mask2, _mm256_shuffle_epi8(values[1], l))),
                               _mm256_or_si256(_mm256_and_si256(mask3, _mm256_shuffle_epi8(values[2], l)),
                                               _mm256_and_si256(mask4, _mm256_shuffle_epi8(values[3], l))));
    }
    static void load_values(__m256i * values) {
        static const uint8_t kvalues_iq6nl[64] = {
               1,    7,   13,   19,   24,   30,   35,   40,   44,   49,   54,   58,   62,   66,   70,   74,
              77,   81,   84,   88,   91,   94,   97,  100,  103,  106,  109,  112,  115,  117,  120,  123,
             126,  128,  131,  134,  137,  140,  142,  145,  148,  151,  155,  158,  161,  164,  168,  172,
             175,  179,  183,  187,  191,  196,  200,  205,  210,  215,  220,  226,  231,  237,  243,  249,
        };
        for (int k = 0; k < 4; ++k) {
            auto values128 = _mm_loadu_si128((const __m128i *)kvalues_iq6nl + k);
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

template <typename Q8>
inline void process_mins_and_scales_16(const __m128i& scales128, const Q8& q8, int i, float d,
    __m256 * accm, __m256i * scales) {
    const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
    process_mins_16(all_scales, q8, i, d, accm);
    prepare_scales_16(all_scales, scales);
}

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

struct DequantizerIQ2TN final : public BaseDequantizer<block_iq2_tn> {
    DequantizerIQ2TN(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}

    inline void new_block(int i) {
        d = GGML_FP16_TO_FP32(x[i].d);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
    }

    Q2Bits  bits;
};


template <int nrc_y>
IQK_NOINLINE void mul_mat_iq2tn_q8_K(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK_K == 0);
    const int nb = n/QK_K;

    Q8<nrc_y> q8(info);
    DequantizerIQ2TN deq1(vx, bx), deq2(vx, bx);

    __m256  accd[nrc_y];
    const auto m1 = _mm256_set1_epi16(1);

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq1.new_row(ix);
        deq2.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            deq1.new_block(i);

            if  constexpr (nrc_y == 1) {
                deq1.prepare(i, 0);
                auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(deq1.bits.values[0], q8.load_quants(0, i, 0)),
                                              _mm256_maddubs_epi16(deq1.bits.values[1], q8.load_quants(0, i, 1)));
                sumi1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(deq1.bits.values[2], q8.load_quants(0, i, 2)),
                                                          _mm256_maddubs_epi16(deq1.bits.values[3], q8.load_quants(0, i, 3))), sumi1);

                deq2.prepare(i, 1);
                auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(deq2.bits.values[0], q8.load_quants(0, i, 4)),
                                              _mm256_maddubs_epi16(deq2.bits.values[1], q8.load_quants(0, i, 5)));
                sumi2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(deq2.bits.values[2], q8.load_quants(0, i, 6)),
                                                          _mm256_maddubs_epi16(deq2.bits.values[3], q8.load_quants(0, i, 7))), sumi2);
                auto sumi = _mm256_add_epi16(sumi2, _mm256_sub_epi16(sumi1, q8.load_bsums(0, i)));
                auto vd = _mm256_set1_ps(deq1.d*q8.scale(0, i));
                auto sf = _mm256_cvtepi32_ps(_mm256_madd_epi16(m1, sumi));
                accd[0] = i > 0 ? _mm256_fmadd_ps(vd, sf, accd[0]) : _mm256_mul_ps(vd, sf);
            }
            else {

                deq1.prepare(i, 0); deq2.prepare(i, 1);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto vd = _mm256_set1_ps(deq1.d*q8.scale(iy, i));
                    auto sumi = _mm256_add_epi16(_mm256_maddubs_epi16(deq1.bits.values[0], q8.load_quants(iy, i, 0)),
                                                 _mm256_maddubs_epi16(deq1.bits.values[1], q8.load_quants(iy, i, 1)));
                    sumi = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(deq1.bits.values[2], q8.load_quants(iy, i, 2)),
                                                             _mm256_maddubs_epi16(deq1.bits.values[3], q8.load_quants(iy, i, 3))), sumi);
                    sumi = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(deq2.bits.values[0], q8.load_quants(iy, i, 4)),
                                                             _mm256_maddubs_epi16(deq2.bits.values[1], q8.load_quants(iy, i, 5))), sumi);
                    sumi = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(deq2.bits.values[2], q8.load_quants(iy, i, 6)),
                                                             _mm256_maddubs_epi16(deq2.bits.values[3], q8.load_quants(iy, i, 7))), sumi);
                    sumi = _mm256_sub_epi16(sumi, q8.load_bsums(iy, i));

                    //auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(deq1.bits.values[0], q8.load_quants(iy, i, 0)),
                    //                              _mm256_maddubs_epi16(deq1.bits.values[1], q8.load_quants(iy, i, 1)));
                    //auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(deq1.bits.values[2], q8.load_quants(iy, i, 2)),
                    //                              _mm256_maddubs_epi16(deq1.bits.values[3], q8.load_quants(iy, i, 3)));
                    //sumi1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(deq2.bits.values[0], q8.load_quants(iy, i, 4)),
                    //                                          _mm256_maddubs_epi16(deq2.bits.values[1], q8.load_quants(iy, i, 5))), sumi1);
                    //sumi2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(deq2.bits.values[2], q8.load_quants(iy, i, 6)),
                    //                                          _mm256_maddubs_epi16(deq2.bits.values[3], q8.load_quants(iy, i, 7))), sumi2);
                    //auto sumi = _mm256_add_epi16(sumi2, _mm256_sub_epi16(sumi1, q8.load_bsums(iy, i)));
                    auto sf = _mm256_cvtepi32_ps(_mm256_madd_epi16(m1, sumi));
                    accd[iy] = i > 0 ? _mm256_fmadd_ps(vd, sf, accd[iy]) : _mm256_mul_ps(vd, sf);
                }
            }

        }

        if constexpr (nrc_y == 8) {
            store_8(ix, accd, info);
        } else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                info.store(ix, iy, hsum_float_8(accd[iy]));
            }
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

        if constexpr (nrc_y == 8) {
            store_8(ix, accd, info);
        } else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                info.store(ix, iy, hsum_float_8(accd[iy]));
            }
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

                multiply_add(deq.bits, scales, j, i, q8, sumi);

            }

            for (int iy = 0; iy < nrc_y; ++iy) {
                const __m256 vd = _mm256_set1_ps(deq.d*q8.scale(iy, i));
                accd[iy] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[iy]), accd[iy]);
            }

        }

        if constexpr (nrc_y == 8) {
            store_8(ix, accd, info);
        } else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                info.store(ix, iy, hsum_float_8(accd[iy]));
            }
        }

    }
}

#endif  // Zen4 or vanilla AVX2

template <typename Bits>
inline void multiply_add_1(int j, const Bits& bits, const __m256i * scales, const __m256i * q8, __m256i * sumi) {
    if (j == 0) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
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
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
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

inline void set_scales_8_iq(int j, const __m256i& all_scales, __m256i * scales) {
#ifdef HAVE_FANCY_SIMD
    auto shuffle = j == 0 ? _mm256_set_epi64x(0x0302030203020302, 0x0100010001000100, 0x0302030203020302, 0x0100010001000100)
                          : _mm256_set_epi64x(0x0b0a0b0a0b0a0b0a, 0x0908090809080908, 0x0b0a0b0a0b0a0b0a, 0x0908090809080908);
    scales[0] = _mm256_shuffle_epi8(all_scales, shuffle);
    scales[1] = _mm256_shuffle_epi8(all_scales, _mm256_add_epi8(shuffle, _mm256_set1_epi8(4)));
#else
    set_scales_8(all_scales, j, scales);
#endif
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

        if constexpr (nrc_y == 8) {
            store_8(ix, accd, info);
        } else {
            for (int iy = 0; iy < nrc_y; ++iy) {
                info.store(ix, iy, hsum_float_8(accd[iy]));
            }
        }
    }
}

template <int nrc> struct Q8_K64 {

    constexpr static int nrc_y = nrc;

    Q8_K64(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            const float * dptr = (const float *)info.src1_row(iy);
            std::memcpy(d + 8*iy, dptr, 8*sizeof(float));
            y[iy] = (const int8_t *)(dptr + 8);
        }
    }

    inline __m256i load_quants(int iy, int i, int j) const { return _mm256_loadu_si256((const __m256i*)y[iy] + 4*i + j); }
    inline __m128  scale(int iy) const { return _mm_loadu_ps(d + 8*iy); }
    inline __m128  minus(int iy) const { return _mm_loadu_ps(d + 8*iy + 4); }

    float d[8*nrc_y];
    const int8_t * y[nrc_y];
};

struct DequantizerIQ1BN {
    const __m256i m1_8   = _mm256_set1_epi8(1);
    static __m256i load_shuffle(int i) {
        static const uint8_t data[128] = {
            0, 255, 0, 255, 0, 255, 0, 255, 0, 255,  1, 255,  1, 255,  1, 255,  1, 255,  1, 255,  2, 255,  2, 255,  2, 255,  2, 255,  2, 255, 12, 255,
            3, 255, 3, 255, 3, 255, 3, 255, 3, 255,  4, 255,  4, 255,  4, 255,  4, 255,  4, 255,  5, 255,  5, 255,  5, 255,  5, 255,  5, 255, 12, 255,
            6, 255, 6, 255, 6, 255, 6, 255, 6, 255,  7, 255,  7, 255,  7, 255,  7, 255,  7, 255,  8, 255,  8, 255,  8, 255,  8, 255,  8, 255, 12, 255,
            9, 255, 9, 255, 9, 255, 9, 255, 9, 255, 10, 255, 10, 255, 10, 255, 10, 255, 10, 255, 11, 255, 11, 255, 11, 255, 11, 255, 11, 255, 12, 255,
        };
        return _mm256_loadu_si256((const __m256i*)data + i);
    }
    const __m256i shuff[4] = { load_shuffle(0), load_shuffle(1), load_shuffle(2), load_shuffle(3) };
    const __m256i mult[4]  = {
            _mm256_set_epi64x(0x5100010003000900, 0x1b00510001000300, 0x09001b0051000100, 0x030009001b005100),
            _mm256_set_epi64x(0x1b00010003000900, 0x1b00510001000300, 0x09001b0051000100, 0x030009001b005100),
            _mm256_set_epi64x(0x0900010003000900, 0x1b00510001000300, 0x09001b0051000100, 0x030009001b005100),
            _mm256_set_epi64x(0x0300010003000900, 0x1b00510001000300, 0x09001b0051000100, 0x030009001b005100),
    };
    const __m256i m3 = _mm256_set1_epi16(3);
#ifdef HAVE_FANCY_SIMD
    const __m256i bmask = _mm256_set_epi8(62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
#endif

    IQK_ALWAYS_INLINE void prepare_iq1bn_quants(const block_iq1_bn * x, __m256i& v1, __m256i& v2) const {
        auto data128 = _mm_loadu_si128((const __m128i *)x);  // Note: we load 16 instead of 13 bytes!
        auto data = MM256_SET_M128I(data128, data128);
        auto val1 = _mm256_mulhi_epu16(_mm256_mullo_epi16(_mm256_shuffle_epi8(data, shuff[0]), mult[0]), m3);
        auto val2 = _mm256_mulhi_epu16(_mm256_mullo_epi16(_mm256_shuffle_epi8(data, shuff[1]), mult[1]), m3);
        auto val3 = _mm256_mulhi_epu16(_mm256_mullo_epi16(_mm256_shuffle_epi8(data, shuff[2]), mult[2]), m3);
        auto val4 = _mm256_mulhi_epu16(_mm256_mullo_epi16(_mm256_shuffle_epi8(data, shuff[3]), mult[3]), m3);
#ifdef HAVE_FANCY_SIMD
        v1 = _mm256_permutex2var_epi8(val1, bmask, val2);
        v2 = _mm256_permutex2var_epi8(val3, bmask, val4);
#else
        v1 = _mm256_permute4x64_epi64(_mm256_packs_epi16(val1, val2), 216);
        v2 = _mm256_permute4x64_epi64(_mm256_packs_epi16(val3, val4), 216);
#endif
    }

};

template <int nrc_y, bool is_iq1_tn>
IQK_NOINLINE void mul_mat_iq1bn_q8_K64(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const int nb = n / QK_IQ1BN;
    Q8_K64<nrc_y> q8(info);
    DequantizerIQ1BN deq;
    __m256i accd[nrc_y];
    __m256i val[4];

#if !(defined __AVX512VNNI__ && defined __AVX512VL__)
    const auto m1_16  = _mm256_set1_epi16(1);
#endif

    const block_iq1_bn * x;
    const char * cx0 = (const char *)vx;
    float scale;

    for (int ix = 0; ix < nrc_x; ++ix) {

        const char * cx = cx0 + ix*bx;
        if constexpr (is_iq1_tn) {
            scale = GGML_FP16_TO_FP32(*(const ggml_half *)cx);
            cx += sizeof(ggml_half);
        }
        x = (const block_iq1_bn *)cx;

        if constexpr (nrc_y == 1) {
            __m256i acc1 = _mm256_setzero_si256(), acc2 = _mm256_setzero_si256();
            for (int i = 0; i < nb/2; ++i) {
                deq.prepare_iq1bn_quants(x + 2*i + 0, val[0], val[1]);
                deq.prepare_iq1bn_quants(x + 2*i + 1, val[2], val[3]);
#if defined __AVX512VNNI__ && defined __AVX512VL__
                acc1 = _mm256_dpbusd_epi32(_mm256_dpbusd_epi32(acc1, val[0], q8.load_quants(0, i, 0)), val[1], q8.load_quants(0, i, 1));
                acc2 = _mm256_dpbusd_epi32(_mm256_dpbusd_epi32(acc2, val[2], q8.load_quants(0, i, 2)), val[3], q8.load_quants(0, i, 3));
#else
                auto dot1 = _mm256_add_epi16(_mm256_maddubs_epi16(val[0], q8.load_quants(0, i, 0)),
                                             _mm256_maddubs_epi16(val[1], q8.load_quants(0, i, 1)));
                auto dot2 = _mm256_add_epi16(_mm256_maddubs_epi16(val[2], q8.load_quants(0, i, 2)),
                                             _mm256_maddubs_epi16(val[3], q8.load_quants(0, i, 3)));
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(m1_16, dot1));
                acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(m1_16, dot2));
#endif
            }
            accd[0] = _mm256_add_epi32(acc1, acc2);
        }
        else {

            for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_si256();

            for (int i = 0; i < nb/2; ++i) {

                deq.prepare_iq1bn_quants(x + 2*i + 0, val[0], val[1]);
                deq.prepare_iq1bn_quants(x + 2*i + 1, val[2], val[3]);

                for (int iy = 0; iy < nrc_y; ++iy) {
#if defined __AVX512VNNI__ && defined __AVX512VL__
                    accd[iy]  = _mm256_dpbusd_epi32(_mm256_dpbusd_epi32(_mm256_dpbusd_epi32(_mm256_dpbusd_epi32(accd[iy],
                                        val[0], q8.load_quants(iy, i, 0)),
                                        val[1], q8.load_quants(iy, i, 1)),
                                        val[2], q8.load_quants(iy, i, 2)),
                                        val[3], q8.load_quants(iy, i, 3));
#else
                    auto dot1 = _mm256_add_epi16(_mm256_maddubs_epi16(val[0], q8.load_quants(iy, i, 0)),
                                                 _mm256_maddubs_epi16(val[1], q8.load_quants(iy, i, 1)));
                    auto dot2 = _mm256_add_epi16(_mm256_maddubs_epi16(val[2], q8.load_quants(iy, i, 2)),
                                                 _mm256_maddubs_epi16(val[3], q8.load_quants(iy, i, 3)));
                    dot1 = _mm256_madd_epi16(m1_16, _mm256_add_epi16(dot1, dot2));
                    accd[iy] = _mm256_add_epi32(dot1, accd[iy]);
#endif
                }
            }
        }
        int i = 2*(nb/2);
        if (i < nb) {
            deq.prepare_iq1bn_quants(x + i, val[0], val[1]);
            for (int iy = 0; iy < nrc_y; ++iy) {
#if defined __AVX512VNNI__ && defined __AVX512VL__
                accd[iy] = _mm256_dpbusd_epi32(_mm256_dpbusd_epi32(accd[iy],
                            val[0], q8.load_quants(iy, i/2, 0)), val[1], q8.load_quants(iy, i/2, 1));
#else
                auto dot = _mm256_madd_epi16(m1_16, _mm256_add_epi16(_mm256_maddubs_epi16(val[0], q8.load_quants(iy, i/2, 0)),
                                                                     _mm256_maddubs_epi16(val[1], q8.load_quants(iy, i/2, 1))));
                accd[iy] = _mm256_add_epi32(dot, accd[iy]);
#endif
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            auto vd = q8.scale(iy);
            auto sumi = _mm_add_epi32(_mm256_castsi256_si128(accd[iy]), _mm256_extractf128_si256(accd[iy], 1));
            auto sumf = _mm_fmsub_ps(vd, _mm_cvtepi32_ps(sumi), q8.minus(iy));
            if constexpr (is_iq1_tn) {
                info.store(ix, iy, scale*hsum_float_4(sumf));
            } else {
                info.store(ix, iy, hsum_float_4(sumf));
            }
        }

    }
}

struct DequantizeIQ2BN final : public BaseDequantizer<block_iq2_bn> {
    DequantizeIQ2BN(const void * vx, size_t bx) : BaseDequantizer(vx, bx) {}

    IQK_ALWAYS_INLINE void prepare4(int i, __m256i * val) const {
        auto q2bits_1 = _mm256_loadu_si256((const __m256i *)x[2*i].qs);
        auto q2bits_2 = _mm256_srli_epi16(q2bits_1, 2);
        make2(_mm256_permute2x128_si256(q2bits_1, q2bits_2, 0x20), val+0);
        make2(_mm256_permute2x128_si256(q2bits_1, q2bits_2, 0x31), val+2);
    }
    IQK_ALWAYS_INLINE void make2(__m256i q2_1, __m256i * val) const {
        val[0] = _mm256_and_si256(q2_1, mask2);
        val[1] = _mm256_and_si256(_mm256_srli_epi16(q2_1, 4), mask2);
    }
    IQK_ALWAYS_INLINE void prepare2(int i, __m256i * val) const {
        auto q2bits_1 = _mm_loadu_si128((const __m128i *)x[i].qs);
        make2(MM256_SET_M128I(_mm_srli_epi16(q2bits_1, 2), q2bits_1), val);
    }
    const __m256i m1_8   = _mm256_set1_epi8(1);
    const __m256i mf_8   = _mm256_set1_epi8(16);
    const __m256i mask2  = _mm256_set1_epi8(0x03);
    const __m256i mask3  = _mm256_set1_epi8(0x30);
};

template <int nrc_y>
IQK_NOINLINE void mul_mat_iq2bn_q8_K64(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const int nb = n / QK_IQ1BN;
    Q8_K64<nrc_y> q8(info);
    DequantizeIQ2BN deq(vx, bx);
    __m256i  accd[nrc_y];
    __m256i  val[4];

#if !(defined __AVX512VNNI__ && defined __AVX512VL__)
    const auto m1_16  = _mm256_set1_epi16(1);
#endif

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);

        if constexpr (nrc_y == 1) {
            __m256i acc[2] = {};
            for (int i = 0; i < nb/2; ++i) {
                deq.prepare4(i, val);
#if defined __AVX512VNNI__ && defined __AVX512VL__
                acc[0] = _mm256_dpbusd_epi32(_mm256_dpbusd_epi32(acc[0], val[0], q8.load_quants(0, i, 0)),
                                                                         val[1], q8.load_quants(0, i, 1));
                acc[1] = _mm256_dpbusd_epi32(_mm256_dpbusd_epi32(acc[1], val[2], q8.load_quants(0, i, 2)),
                                                                         val[3], q8.load_quants(0, i, 3));
#else
                auto dot1 = _mm256_add_epi16(_mm256_maddubs_epi16(val[0], q8.load_quants(0, i, 0)),
                                             _mm256_maddubs_epi16(val[1], q8.load_quants(0, i, 1)));
                auto dot2 = _mm256_add_epi16(_mm256_maddubs_epi16(val[2], q8.load_quants(0, i, 2)),
                                             _mm256_maddubs_epi16(val[3], q8.load_quants(0, i, 3)));
                acc[0] = _mm256_add_epi32(acc[0], _mm256_madd_epi16(m1_16, dot1));
                acc[1] = _mm256_add_epi32(acc[1], _mm256_madd_epi16(m1_16, dot2));
#endif
            }
            accd[0] = _mm256_add_epi32(acc[0], acc[1]);
        }
        else {

            for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = _mm256_setzero_si256();

            for (int i = 0; i < nb/2; ++i) {
                deq.prepare4(i, val);
                for (int iy = 0; iy < nrc_y; ++iy) {
#if defined __AVX512VNNI__ && defined __AVX512VL__
                    accd[iy] = _mm256_dpbusd_epi32(_mm256_dpbusd_epi32(_mm256_dpbusd_epi32(_mm256_dpbusd_epi32(accd[iy],
                                        val[0], q8.load_quants(iy, i, 0)), val[1], q8.load_quants(iy, i, 1)),
                                        val[2], q8.load_quants(iy, i, 2)), val[3], q8.load_quants(iy, i, 3));
#else
                    auto dot = _mm256_madd_epi16(m1_16, _mm256_add_epi16(
                                _mm256_add_epi16(_mm256_maddubs_epi16(val[0], q8.load_quants(iy, i, 0)),
                                                 _mm256_maddubs_epi16(val[1], q8.load_quants(iy, i, 1))),
                                _mm256_add_epi16(_mm256_maddubs_epi16(val[2], q8.load_quants(iy, i, 2)),
                                                 _mm256_maddubs_epi16(val[3], q8.load_quants(iy, i, 3)))));
                    accd[iy] = _mm256_add_epi32(dot, accd[iy]);
#endif
                }
            }
        }
        int i = 2*(nb/2);
        if (i < nb) {
            deq.prepare2(i, val);
            for (int iy = 0; iy < nrc_y; ++iy) {
#if defined __AVX512VNNI__ && defined __AVX512VL__
                accd[iy] = _mm256_dpbusd_epi32(_mm256_dpbusd_epi32(accd[iy], val[0], q8.load_quants(iy, i/2, 0)),
                                                                             val[1], q8.load_quants(iy, i/2, 1));
#else
                auto dot = _mm256_madd_epi16(m1_16, _mm256_add_epi16(_mm256_maddubs_epi16(val[0], q8.load_quants(iy, i/2, 0)),
                                                                     _mm256_maddubs_epi16(val[1], q8.load_quants(iy, i/2, 0))));
                accd[iy] = _mm256_add_epi32(dot, accd[iy]);
#endif
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            auto vd = q8.scale(iy);
            auto sumi = _mm_add_epi32(_mm256_castsi256_si128(accd[iy]), _mm256_extractf128_si256(accd[iy], 1));
            auto sumf = _mm_fmsub_ps(vd, _mm_cvtepi32_ps(sumi), q8.minus(iy));
            info.store(ix, iy, hsum_float_4(sumf));
        }
    }
}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_K_q8_K_IQ(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    if constexpr (nrc_y == 1) {
        mul_mat_qX_K_q8_K_IQ_1<Dequantizer>(n, vx, bx, info, nrc_x);
    } else {
        mul_mat_qX_K_q8_K_IQ_N<Dequantizer, nrc_y>(n, vx, bx, info, nrc_x);
    }
}

//#ifdef HAVE_FANCY_SIMD
// Strangely enough, the following implementation makes PP ~6% slower and TG ~6% faster
// compared to the vanilla AVX2 version below.
//struct IndexHelperIQ3S {
//    union index_t {
//        __m256i  vec;
//        uint16_t val[16];
//    };
//    inline void make2(const uint8_t * qs, const uint8_t * qh, __m256i * values) const {
//        auto idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs));
//        const __mmask16 * m16 = (const __mmask16 *)qh;
//        index_t idx;
//        idx.vec = _mm256_mask_add_epi16(idx_l, m16[0], idx_l, offset);
//        values[0] = _mm256_set_epi32(iq3s_grid[idx.val[ 7]], iq3s_grid[idx.val[ 6]], iq3s_grid[idx.val[ 5]], iq3s_grid[idx.val[ 4]],
//                                     iq3s_grid[idx.val[ 3]], iq3s_grid[idx.val[ 2]], iq3s_grid[idx.val[ 1]], iq3s_grid[idx.val[ 0]]);
//        values[1] = _mm256_set_epi32(iq3s_grid[idx.val[15]], iq3s_grid[idx.val[14]], iq3s_grid[idx.val[13]], iq3s_grid[idx.val[12]],
//                                     iq3s_grid[idx.val[11]], iq3s_grid[idx.val[10]], iq3s_grid[idx.val[ 9]], iq3s_grid[idx.val[ 8]]);
//    }
//    const __m256i offset = _mm256_set1_epi16(256);
//};
//#else
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
//#endif

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

struct EvenSignHelper {
#ifdef HAVE_FANCY_SIMD
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
#else
    inline void sign_value(uint32_t aux32, __m256i& value) const {
        auto signs = _mm256_set_epi64x(keven_signs[(aux32 >> 21) & 127], keven_signs[(aux32 >> 14) & 127],
                                       keven_signs[(aux32 >>  7) & 127], keven_signs[(aux32 >>  0) & 127]);
        value = _mm256_sign_epi8(value, signs);
    }
#endif
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
#ifdef HAVE_FANCY_SIMD
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
#ifdef HAVE_FANCY_SIMD
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
#ifndef HAVE_FANCY_SIMD
    Helper helper;
#endif
    const __m256i idx_mask  = _mm256_set1_epi16(511);
    const __m256i min_value = _mm256_set1_epi8(minv);

};

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
#ifdef HAVE_FANCY_SIMD
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

//
// ============================== Legacy quants
//

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

template <int nrc_y> struct MinusType0 {
    inline __m256 compute(__m128 d, int) const { return _mm256_set_m128(d, d); }
    inline float compute(float d, int) const { return d; }
    inline float result(__m256 acc, int) const { return hsum_float_8(acc); }
    //inline void store(int ix, __m256 * acc, const DataInfo& info) {
    //    if constexpr (nrc_y == 8) {
    //        store_8(ix, acc, info);
    //    } else {
    //        for (int iy = 0; iy < nrc_y; ++iy) info.store(ix, iy, hsum_float_8(acc[iy]));
    //    }
    //}
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
    //inline void store(int ix, const __m256 * acc, const DataInfo& info) {
    //    for (int iy = 0; iy < nrc_y; ++iy) {
    //        accm[iy] = _mm_add_ps(accm[iy], _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1)));
    //    }
    //    if constexpr (nrc_y >= 4) {
    //        union { __m128 vec; float val[4]; } h;
    //        for (int i = 0; i < nrc_y/4; ++i) {
    //            accm[4*i+0] = _mm_add_ps(_mm_unpacklo_ps(accm[4*i+0], accm[4*i+2]), _mm_unpackhi_ps(accm[4*i+0], accm[4*i+2]));
    //            accm[4*i+1] = _mm_add_ps(_mm_unpacklo_ps(accm[4*i+1], accm[4*i+3]), _mm_unpackhi_ps(accm[4*i+1], accm[4*i+3]));
    //            h.vec = _mm_add_ps(_mm_unpacklo_ps(accm[4*i+0], accm[4*i+1]), _mm_unpackhi_ps(accm[4*i+0], accm[4*i+1]));
    //            for (int j = 0; j < 4; ++j) info.store(ix, 4*i+j, h.val[j]);
    //        }
    //        for (int iy = 4*(nrc_y/4); iy < nrc_y; ++iy) info.store(ix, iy, hsum_float_4(accm[iy]));
    //    } else {
    //        for (int iy = 0; iy < nrc_y; ++iy) info.store(ix, iy, hsum_float_4(accm[iy]));
    //    }
    //}
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
                    const __m256i p0 = sum.dot.compute(qx[0], _mm256_loadu_si256((const __m256i *)y[iy][i].qs));
                    acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p0), acc[iy]);
                }
            }
        }
        //accm.store(ix, acc, info);
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, accm.result(acc[iy], iy));
        }
    }
};

template <int nrc_y, bool is_multiple_of_4>
using AccumType0 = AccumT<MinusType0<nrc_y>, nrc_y, is_multiple_of_4>;

template <int nrc_y, bool is_multiple_of_4>
using AccumType1 = AccumT<MinusType1<nrc_y>, nrc_y, is_multiple_of_4>;

using Sum4Type0 = Sum4<block_q8_0, block_q8_0_x4, SignedDot>;
using Sum4Type1 = Sum4<block_q8_1, block_q8_1_x4, UnsignedDot>;
using Sum4TypeQ80 = Sum4<block_q8_0, block_q8_0_x4, SignedDot, false>;
using Sum4TypeQ81 = Sum4<block_q8_1, block_q8_1_x4, UnsignedDot, false>;

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
    const __m256i values = load_values();
    inline __m256i dequant(const block_iq4_nl * x) const {
        return _mm256_shuffle_epi8(values, b4.dequant(x->qs));
    }
    static __m256i load_values() {
        static const int8_t iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        auto aux = _mm_loadu_si128((const __m128i *)iq4nl_values);
        return MM256_SET_M128I(aux, aux);
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

struct Q8_0_x4_Unpacker {
    using Sum4T = Sum4TypeQ80;
    inline static int block_size() { return QK8_0; }
    Q8_0_x4_Unpacker(const void * vx, size_t bx) : cx_0((const char *)vx), x((const block_q8_0_x4 *)cx_0), bx(bx) {}

    const char * cx_0;
    const block_q8_0_x4 * x;
    size_t       bx;

    __m256i qx[4];

    inline const __m256i* quants() const { return qx; }

    inline void set_row(int ix) { x = (const block_q8_0_x4 *)(cx_0 + ix*bx); }

    inline auto set_block_4(int i) {
        auto scales = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)x[i].d));
        for (int j = 0; j < 4; ++j) {
            qx[j] = _mm256_loadu_si256((const __m256i *)x[i].qs + j);
        }
        return scales;
    }
    inline auto set_block(int i) {
        auto q8 = (const block_q8_0 *)(x + i);
        qx[0] = _mm256_loadu_si256((const __m256i *)q8->qs);
        return GGML_FP16_TO_FP32(q8->d);
    }
};

struct Q8_0_Unpacker final : public Q_Unpacker<block_q8_0, ScaleHelperQ_0, Q8_0_Dequantizer> {
    Q8_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ80;
    inline static int block_size() { return QK8_0; }
};
struct Q8_0_1_Unpacker final : public Q_Unpacker<block_q8_0, ScaleHelperQ_0_1<127>, Q8_0_1_Dequantizer> {
    Q8_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ81;
    inline static int block_size() { return QK8_0; }
};
struct Q4_0_Unpacker final : public Q_Unpacker<block_q4_0, ScaleHelperQ_0, Q4_0_Dequantizer> {
    Q4_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ80;
    inline static int block_size() { return QK4_0; }
};
struct Q4_0_1_Unpacker final : public Q_Unpacker<block_q4_0, ScaleHelperQ_0_1<8>, Q4_0_1_Dequantizer> {
    Q4_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ81;
    inline static int block_size() { return QK4_0; }
};
struct IQ4_NL_Unpacker final : public Q_Unpacker<block_iq4_nl, ScaleHelperQ_0, IQ4_NL_Dequantizer> {
    IQ4_NL_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ80;
    inline static int block_size() { return QK4_NL; }
};
struct Q5_0_Unpacker final : public Q_Unpacker<block_q5_0, ScaleHelperQ_0, Q5_0_Dequantizer> {
    Q5_0_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ80;
    inline static int block_size() { return QK5_0; }
};
struct Q5_0_1_Unpacker final : public Q_Unpacker<block_q5_0, ScaleHelperQ_0_1<16>, Q5_1_Dequantizer<block_q5_0>> {
    Q5_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ81;
    inline static int block_size() { return QK5_0; }
};
struct Q4_1_Unpacker final : public Q_Unpacker<block_q4_1, ScaleHelperQ_1, Q4_1_Dequantizer> {
    Q4_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4Type1;
    inline static int block_size() { return QK4_1; }
};
struct Q5_1_Unpacker final : public Q_Unpacker<block_q5_1, ScaleHelperQ_1, Q5_1_Dequantizer<block_q5_1>> {
    Q5_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4Type1;
    inline static int block_size() { return QK4_1; }
};

// float matrices - we handle f16 and f32, but only to f32 result

struct QFBase {
#ifdef __AVX512F__
    constexpr static int k_step = 16;
    using Data = __m512;
    using Acc  = __m512;
    static inline Data load(const ggml_half * x) { return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)x)); }
    static inline Data load(const float * x) { return _mm512_loadu_ps(x); }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) {
        return _mm512_fmadd_ps(y, x, prev);
    }
    static inline Acc acc_first(const Data& y, const Data& x) {
        return _mm512_mul_ps(y, x);
    }
    static inline float hsum(Acc acc) {
        return _mm512_reduce_add_ps(acc);
    }
    template <typename Float>
    static inline Data load4Floats(const Float * x) {
        return _mm512_insertf32x4(_mm512_setzero_ps(), load128(x), 0);
    }
#else
    constexpr static int k_step = 8;
    using Data = __m256;
    using Acc  = __m256;
    static inline Data load(const ggml_half * x) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)x)); }
    static inline Data load(const float * x) { return _mm256_loadu_ps(x); }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) {
        return _mm256_fmadd_ps(y, x, prev);
    }
    static inline Acc acc_first(const Data& y, const Data& x) {
        return _mm256_mul_ps(y, x);
    }
    static inline float hsum(Acc acc) {
        return hsum_float_8(acc);
    }
    template <typename Float>
    static inline Data load4Floats(const Float * x) {
        return _mm256_insertf128_ps(_mm256_setzero_ps(), load128(x), 0);
    }
#endif
    static inline __m128 load128(const ggml_half * x) { return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)x)); }
    static inline __m128 load128(const float * x) { return _mm_loadu_ps(x); }
};
template <typename Float, int nrc_in> struct QFT final : public QFBase {
    constexpr static int nrc = nrc_in;
    QFT(const DataInfo& info) {
        for (int iy = 0; iy < nrc; ++iy) y[iy] = (const Float *)info.src1_row(iy);
    }
    QFT(const char * cx, size_t bx) {
        for (int iy = 0; iy < nrc; ++iy) y[iy] = (const Float *)(cx + iy*bx);
    }
    IQK_ALWAYS_INLINE Data load1(int iy, int i) const { return load(y[iy] + k_step*i); }
    IQK_ALWAYS_INLINE Data load_tail(int iy, int i) const { return load4Floats(y[iy] + 4*i); }
    const Float * y[nrc];
};

template <typename Qy, typename Qx>
IQK_NOINLINE void mul_mat_Qx_Qy_MxN(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    int nb = n/QFBase::k_step;
    int nb4 = n/4;
    Qy y(info);
    Qx x(cx + ix0*bx, bx);
    QFBase::Data xv[Qx::nrc];
    QFBase::Acc  acc[Qx::nrc*Qy::nrc];
    auto yv = y.load1(0, 0);
    for (int ix = 0; ix < Qx::nrc; ++ix) {
        xv[ix] = x.load1(ix, 0);
        acc[ix] = QFBase::acc_first(yv, xv[ix]);
    }
    for (int iy = 1; iy < Qy::nrc; ++iy) {
        yv = y.load1(iy, 0);
        for (int ix = 0; ix < Qx::nrc; ++ix) acc[Qx::nrc*iy + ix] = QFBase::acc_first(yv, xv[ix]);
    }
    for (int i = 1; i < nb; ++i) {
        yv = y.load1(0, i);
        for (int ix = 0; ix < Qx::nrc; ++ix) {
            xv[ix] = x.load1(ix, i);
            acc[ix] = QFBase::acc(acc[ix], yv, xv[ix]);
        }
        for (int iy = 1; iy < Qy::nrc; ++iy) {
            yv = y.load1(iy, i);
            for (int ix = 0; ix < Qx::nrc; ++ix) acc[Qx::nrc*iy + ix] = QFBase::acc(acc[Qx::nrc*iy + ix], yv, xv[ix]);
        }
    }
    for (int i = (QFBase::k_step/4)*nb; i < nb4; ++i) {
        yv = y.load_tail(0, i);
        for (int ix = 0; ix < Qx::nrc; ++ix) {
            xv[ix] = x.load_tail(ix, i);
            acc[ix] = QFBase::acc(acc[ix], yv, xv[ix]);
        }
        for (int iy = 1; iy < Qy::nrc; ++iy) {
            yv = y.load_tail(iy, i);
            for (int ix = 0; ix < Qx::nrc; ++ix) acc[Qx::nrc*iy + ix] = QFBase::acc(acc[Qx::nrc*iy + ix], yv, xv[ix]);
        }
    }
    for (int iy = 0; iy < Qy::nrc; ++iy) for (int ix = 0; ix < Qx::nrc; ++ix) info.store(ix0+ix, iy, QFBase::hsum(acc[Qx::nrc*iy+ix]));
}

// This will handle any of f16 x f32, f32 x f16, f16 x f16, f32 x f32, with computations done
// in f32 (i.e., f16 is first converted to f32). It is easy to extend to computations done in
// f16, but I don't have a CPU capable of f16 vector arithmetic, so not doing it for now.
template <int nrc_y, typename FloatX, typename FloatY>
void mul_mat_fX_fY_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
#ifdef __AVX512F__
    constexpr int k_nx = 5;
#else
    constexpr int k_nx = 2;
#endif
    const char * cx = (const char *)vx;
    for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
        mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, k_nx>>(n, cx, bx, ix*k_nx, info);
    }
    int last_x = k_nx*(nrc_x/k_nx);
    if (last_x == nrc_x) return;
    int nx = nrc_x - last_x;
    switch (nx) {
        case 1: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 1>>(n, cx, bx, last_x, info); break;
#ifdef __AVX512F__
        case 2: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 2>>(n, cx, bx, last_x, info); break;
        case 3: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 3>>(n, cx, bx, last_x, info); break;
        case 4: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 4>>(n, cx, bx, last_x, info); break;
#endif
    }
}

#ifdef __AVX512BF16__
struct QFBaseBF16 {
    constexpr static int k_step = 32;
    using Data = __m512bh;
    using Acc  = __m512;
    static inline Data load(const ggml_bf16_t * x) { return __m512bh(_mm512_loadu_si512((const __m512i *)x)); }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) {
        return _mm512_dpbf16_ps(prev, y, x);
    }
    static inline Acc acc_first(const Data& y, const Data& x) {
        return _mm512_dpbf16_ps(_mm512_setzero_ps(), y, x);
    }
    static inline float hsum(Acc acc) {
        return _mm512_reduce_add_ps(acc);
    }
};
template <int nrc_in> struct QFTBF16 final : public QFBaseBF16 {
    constexpr static int nrc = nrc_in;
    QFTBF16(const DataInfo& info) {
        for (int iy = 0; iy < nrc; ++iy) y[iy] = (const ggml_bf16_t *)info.src1_row(iy);
    }
    QFTBF16(const char * cx, size_t bx) {
        for (int iy = 0; iy < nrc; ++iy) y[iy] = (const ggml_bf16_t *)(cx + iy*bx);
    }
    IQK_ALWAYS_INLINE Data load1(int iy, int i) const { return load(y[iy] + k_step*i); }
    const ggml_bf16_t * y[nrc];
};

template <int nrc_y, int nrc_x>
IQK_NOINLINE void mul_mat_Qx_Qy_MxN(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    int nb = n/QFBaseBF16::k_step;
    QFTBF16<nrc_y> y(info);
    QFTBF16<nrc_x> x(cx + ix0*bx, bx);
    QFBaseBF16::Data xv[nrc_x];
    QFBaseBF16::Acc  acc[nrc_x*nrc_y];
    auto yv = y.load1(0, 0);
    for (int ix = 0; ix < nrc_x; ++ix) {
        xv[ix] = x.load1(ix, 0);
        acc[ix] = QFBaseBF16::acc_first(yv, xv[ix]);
    }
    for (int iy = 1; iy < nrc_y; ++iy) {
        yv = y.load1(iy, 0);
        for (int ix = 0; ix < nrc_x; ++ix) acc[nrc_x*iy + ix] = QFBaseBF16::acc_first(yv, xv[ix]);
    }
    for (int i = 1; i < nb; ++i) {
        yv = y.load1(0, i);
        for (int ix = 0; ix < nrc_x; ++ix) {
            xv[ix] = x.load1(ix, i);
            acc[ix] = QFBaseBF16::acc(acc[ix], yv, xv[ix]);
        }
        for (int iy = 1; iy < nrc_y; ++iy) {
            yv = y.load1(iy, i);
            for (int ix = 0; ix < nrc_x; ++ix) acc[nrc_x*iy + ix] = QFBaseBF16::acc(acc[nrc_x*iy + ix], yv, xv[ix]);
        }
    }
    for (int iy = 0; iy < nrc_y; ++iy) for (int ix = 0; ix < nrc_x; ++ix) info.store(ix0+ix, iy, QFBaseBF16::hsum(acc[nrc_x*iy+ix]));
}
template <int nrc_y>
void mul_mat_fX_fY_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    constexpr int k_nx = nrc_y <= 2 ? 8 : 5;
    const char * cx = (const char *)vx;
    for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
        mul_mat_Qx_Qy_MxN<nrc_y, k_nx>(n, cx, bx, ix*k_nx, info);
    }
    int last_x = k_nx*(nrc_x/k_nx);
    if (last_x == nrc_x) return;
    int nx = nrc_x - last_x;
    if constexpr (nrc_y <= 2) {
        if (nx >= 4) {
            mul_mat_Qx_Qy_MxN<nrc_y, 4>(n, cx, bx, last_x, info);
            last_x += 4;
            if (last_x == nrc_x) return;
            nx = nrc_x - last_x;
        }
    }
    switch (nx) {
        case 1: mul_mat_Qx_Qy_MxN<nrc_y, 1>(n, cx, bx, last_x, info); break;
        case 2: mul_mat_Qx_Qy_MxN<nrc_y, 2>(n, cx, bx, last_x, info); break;
        case 3: mul_mat_Qx_Qy_MxN<nrc_y, 3>(n, cx, bx, last_x, info); break;
        case 4: mul_mat_Qx_Qy_MxN<nrc_y, 4>(n, cx, bx, last_x, info); break;
    }
}
#endif

//
// Tiled Q8_0 x Q8_0 implementation. Not used as the templated legacy quant implementation
// above is faster. Left behind so we remember we tried.
//
template <int nrc> struct Q80 {
    constexpr static int nrc_y = nrc;
    Q80(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8_0 *)info.src1_row(iy);
    }
    IQK_ALWAYS_INLINE __m256i load1(int iy, int i) const { return _mm256_loadu_si256((const __m256i *)y[iy][i].qs); }
    IQK_ALWAYS_INLINE float scale(int iy, int i) const { return GGML_FP16_TO_FP32(y[iy][i].d); }

   const block_q8_0 * y[nrc_y];
};
inline __m256i mul_q80(__m256i x, __m256i y) {
    auto ux = _mm256_sign_epi8(x, x);
#ifdef HAVE_FANCY_SIMD
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), ux, _mm256_sign_epi8(y, x));
#else
    return _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(ux, _mm256_sign_epi8(y, x)));
#endif
}
template <int nrc_y>
void mul_mat_q80_q80_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n%QK8_0 == 0);
    constexpr int k_nx = 4;
    int nb = n/QK8_0;
    Q80<nrc_y> q8(info);
    const block_q8_0 * x[k_nx];
    float ds[k_nx];
    __m256 acc[k_nx*nrc_y];
    __m256i xv[k_nx];
    for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
        int ix0 = k_nx*ix;
        for (int kx = 0; kx < k_nx; ++kx) {
            x[kx] = (const block_q8_0 *)((const char *)vx + (ix0 + kx)*bx);
            ds[kx] = GGML_FP16_TO_FP32(x[kx][0].d);
            xv[kx] = _mm256_loadu_si256((const __m256i *)x[kx][0].qs);
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto yv = q8.load1(iy, 0);
            float d = q8.scale(iy, 0);
            for (int kx = 0; kx < k_nx; ++kx) {
                auto dot = mul_q80(yv, xv[kx]);
                acc[k_nx*iy + kx] = _mm256_mul_ps(_mm256_set1_ps(ds[kx]*d), _mm256_cvtepi32_ps(dot));
            }
        }
        for (int i = 1; i < nb; ++i) {
            for (int kx = 0; kx < k_nx; ++kx) {
                ds[kx] = GGML_FP16_TO_FP32(x[kx][i].d);
                xv[kx] = _mm256_loadu_si256((const __m256i *)x[kx][i].qs);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto yv = q8.load1(iy, i);
                float d = q8.scale(iy, i);
                for (int kx = 0; kx < k_nx; ++kx) {
                    auto dot = mul_q80(yv, xv[kx]);
                    acc[k_nx*iy + kx] = _mm256_fmadd_ps(_mm256_set1_ps(ds[kx]*d), _mm256_cvtepi32_ps(dot), acc[k_nx*iy + kx]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            for (int kx = 0; kx < k_nx; ++kx) info.store(ix0+kx, iy, hsum_float_8(acc[k_nx*iy+kx]));
        }
    }
    int last_x = k_nx*(nrc_x/k_nx);
    if (last_x == nrc_x) return;
    // TODO: handle remaining rows
}

template <typename Dequantizer> void MulMat::set_functions(MulMat& m) {
        if constexpr (std::is_same_v<Dequantizer, Q4_0_Unpacker> || std::is_same_v<Dequantizer, Q5_0_Unpacker> ||
                      std::is_same_v<Dequantizer, Q8_0_Unpacker> || std::is_same_v<Dequantizer, IQ4_NL_Unpacker>) {
            m.funcs[0] = mul_mat_qX_0_q8_0_T<Dequantizer, 1>;
            m.funcs[1] = mul_mat_qX_0_q8_0_T<Dequantizer, 2>;
            m.funcs[2] = mul_mat_qX_0_q8_0_T<Dequantizer, 3>;
            m.funcs[3] = mul_mat_qX_0_q8_0_T<Dequantizer, 4>;
            m.funcs[4] = mul_mat_qX_0_q8_0_T<Dequantizer, 5>;
            m.funcs[5] = mul_mat_qX_0_q8_0_T<Dequantizer, 6>;
            m.funcs[6] = mul_mat_qX_0_q8_0_T<Dequantizer, 7>;
            m.funcs[7] = mul_mat_qX_0_q8_0_T<Dequantizer, 8>;
        }
        else if constexpr (std::is_same_v<Dequantizer, Q4_1_Unpacker> || std::is_same_v<Dequantizer, Q5_1_Unpacker> ||
                           std::is_same_v<Dequantizer, Q8_0_1_Unpacker> || std::is_same_v<Dequantizer, Q4_0_1_Unpacker> ||
                           std::is_same_v<Dequantizer, Q5_0_1_Unpacker>) {
            m.funcs[0] = mul_mat_qX_1_q8_1_T<Dequantizer, 1>;
            m.funcs[1] = mul_mat_qX_1_q8_1_T<Dequantizer, 2>;
            m.funcs[2] = mul_mat_qX_1_q8_1_T<Dequantizer, 3>;
            m.funcs[3] = mul_mat_qX_1_q8_1_T<Dequantizer, 4>;
            m.funcs[4] = mul_mat_qX_1_q8_1_T<Dequantizer, 5>;
            m.funcs[5] = mul_mat_qX_1_q8_1_T<Dequantizer, 6>;
            m.funcs[6] = mul_mat_qX_1_q8_1_T<Dequantizer, 7>;
            m.funcs[7] = mul_mat_qX_1_q8_1_T<Dequantizer, 8>;
        }
        else if constexpr (std::is_same_v<Dequantizer, DequantizerIQ3S> || std::is_same_v<Dequantizer, DequantizerIQ3XXS> ||
                           std::is_same_v<Dequantizer, DequantizerIQ2S> || std::is_same_v<Dequantizer, DequantizerIQ2XS>  ||
                           std::is_same_v<Dequantizer, DequantizerIQ2XXS>) {
            m.funcs[0] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 1>;
            m.funcs[1] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 2>;
            m.funcs[2] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 3>;
            m.funcs[3] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 4>;
            m.funcs[4] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 5>;
            m.funcs[5] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 6>;
            m.funcs[6] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 7>;
            m.funcs[7] = mul_mat_qX_K_q8_K_IQ<Dequantizer, 8>;
        }
        else {
#ifdef HAVE_FANCY_SIMD
            if constexpr (std::is_same_v<Dequantizer, DequantizerIQ6K> ||
                          std::is_same_v<Dequantizer, DequantizerIQ5K> ||
                          std::is_same_v<Dequantizer, DequantizerIQ4K> ||
                          std::is_same_v<Dequantizer, DequantizerIQ3K>) {
                m.funcs[0] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 1>;
                m.funcs[1] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 2>;
                m.funcs[2] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 3>;
                m.funcs[3] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 4>;
                m.funcs[4] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 5>;
                m.funcs[5] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 6>;
                m.funcs[6] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 7>;
                m.funcs[7] = mul_mat_iqX_k_q8_K_AVX512<Dequantizer, 8>;
            } else {
                m.funcs[0] = mul_mat_qX_K_q8_K_AVX512_1<Dequantizer>;
                m.funcs[1] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 2>;
                m.funcs[2] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 3>;
                m.funcs[3] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 4>;
                m.funcs[4] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 5>;
                m.funcs[5] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 6>;
                m.funcs[6] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 7>;
                m.funcs[7] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 8>;
            }
#else
            if constexpr (std::is_same_v<Dequantizer, DequantizerQ2K> ||
                          std::is_same_v<Dequantizer, DequantizerQ3K> ||
                          std::is_same_v<Dequantizer, DequantizerQ6K> ||
                          std::is_same_v<Dequantizer, DequantizerIQ2K>||
                          std::is_same_v<Dequantizer, DequantizerIQ3K>||
                          std::is_same_v<Dequantizer, DequantizerIQ4K>||
                          std::is_same_v<Dequantizer, DequantizerIQ5K>||
                          std::is_same_v<Dequantizer, DequantizerIQ6K>) {
                m.funcs[0] = mul_mat_qY_K_q8_K_T<Dequantizer, 1>;
                m.funcs[1] = mul_mat_qY_K_q8_K_T<Dequantizer, 2>;
                m.funcs[2] = mul_mat_qY_K_q8_K_T<Dequantizer, 3>;
                m.funcs[3] = mul_mat_qY_K_q8_K_T<Dequantizer, 4>;
                m.funcs[4] = mul_mat_qY_K_q8_K_T<Dequantizer, 5>;
                m.funcs[5] = mul_mat_qY_K_q8_K_T<Dequantizer, 6>;
                m.funcs[6] = mul_mat_qY_K_q8_K_T<Dequantizer, 7>;
                m.funcs[7] = mul_mat_qY_K_q8_K_T<Dequantizer, 8>;
            } else {
                m.funcs[0] = mul_mat_qX_K_q8_K_T<Dequantizer, 1>;
                m.funcs[1] = mul_mat_qX_K_q8_K_T<Dequantizer, 2>;
                m.funcs[2] = mul_mat_qX_K_q8_K_T<Dequantizer, 3>;
                m.funcs[3] = mul_mat_qX_K_q8_K_T<Dequantizer, 4>;
                m.funcs[4] = mul_mat_qX_K_q8_K_T<Dequantizer, 5>;
                m.funcs[5] = mul_mat_qX_K_q8_K_T<Dequantizer, 6>;
                m.funcs[6] = mul_mat_qX_K_q8_K_T<Dequantizer, 7>;
                m.funcs[7] = mul_mat_qX_K_q8_K_T<Dequantizer, 8>;
            }
#endif
        }
}

template <typename FloatX, typename FloatY>
void set_mul_mat_f(MulMat& mm) {
    for (auto& f : mm.funcs) f = nullptr;
    mm.funcs[0] = mul_mat_fX_fY_T<1, FloatX, FloatY>;
    mm.funcs[1] = mul_mat_fX_fY_T<2, FloatX, FloatY>;
    mm.funcs[2] = mul_mat_fX_fY_T<3, FloatX, FloatY>;
    mm.funcs[3] = mul_mat_fX_fY_T<4, FloatX, FloatY>;
    mm.funcs[4] = mul_mat_fX_fY_T<5, FloatX, FloatY>;
#ifndef __AVX512F__
    mm.funcs[5] = mul_mat_fX_fY_T<6, FloatX, FloatY>;
#endif
}

#ifdef __AVX512BF16__
void set_mul_mat_bf16(MulMat& mm) {
    for (auto& f : mm.funcs) f = nullptr;
    mm.funcs[0] = mul_mat_fX_fY_T<1>;
    mm.funcs[1] = mul_mat_fX_fY_T<2>;
    mm.funcs[2] = mul_mat_fX_fY_T<3>;
    mm.funcs[3] = mul_mat_fX_fY_T<4>;
    mm.funcs[4] = mul_mat_fX_fY_T<5>;
}
#endif

bool MulMat::prepare(int typeA, int typeB, int ne00, MulMat& mm, int Ny) {

    (void)Ny;

    if (typeA == GGML_TYPE_BF16) {
        if (ne00 % 32) return false;
        switch (typeB) {
#ifdef __AVX512BF16__
            case GGML_TYPE_BF16: set_mul_mat_bf16(mm); break;
#endif
            default: return false;
        }
        return true;
    }

    if (typeA == GGML_TYPE_F16 || typeA == GGML_TYPE_F32) {
        if (ne00 % 4) return false;
    }
    if (typeA == GGML_TYPE_F16) {
        switch (typeB) {
            case GGML_TYPE_F16: set_mul_mat_f<ggml_half, ggml_half>(mm); break;
            case GGML_TYPE_F32: set_mul_mat_f<ggml_half, float>(mm);     break;
            default: return false;
        }
        return true;
    }
    if (typeA == GGML_TYPE_F32) {
        switch (typeB) {
            case GGML_TYPE_F16: set_mul_mat_f<float, ggml_half>(mm); break;
            case GGML_TYPE_F32: set_mul_mat_f<float, float>(mm);     break;
            default: return false;
        }
        return true;
    }

    auto expected_typeB = GGML_TYPE_Q8_K;

    switch (typeA) {
        case GGML_TYPE_Q2_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerQ2K>(mm);
            break;
        case GGML_TYPE_IQ2_TN:
            assert (ne00 % QK_K == 0);
#ifdef HAVE_FANCY_SIMD
            //MulMat::set_functions<DequantizerIQ2TN>(mm);
            mm.funcs[0] = mul_mat_qX_K_q8_K_AVX512_1<DequantizerIQ2TN>;
            //mm.funcs[0] = mul_mat_iq2tn_q8_K_AVX512<1>;
            mm.funcs[1] = mul_mat_iq2tn_q8_K_AVX512<2>;
            mm.funcs[2] = mul_mat_iq2tn_q8_K_AVX512<3>;
            mm.funcs[3] = mul_mat_iq2tn_q8_K_AVX512<4>;
            mm.funcs[4] = mul_mat_iq2tn_q8_K_AVX512<5>;
            mm.funcs[5] = mul_mat_iq2tn_q8_K_AVX512<6>;
            mm.funcs[6] = mul_mat_iq2tn_q8_K_AVX512<7>;
            mm.funcs[7] = mul_mat_iq2tn_q8_K_AVX512<8>;
#else
            mm.funcs[0] = mul_mat_iq2tn_q8_K<1>;
            mm.funcs[1] = mul_mat_iq2tn_q8_K<2>;
            mm.funcs[2] = mul_mat_iq2tn_q8_K<3>;
            mm.funcs[3] = mul_mat_iq2tn_q8_K<4>;
            mm.funcs[4] = mul_mat_iq2tn_q8_K<5>;
            mm.funcs[5] = mul_mat_iq2tn_q8_K<6>;
            mm.funcs[6] = mul_mat_iq2tn_q8_K<7>;
            mm.funcs[7] = mul_mat_iq2tn_q8_K<8>;
#endif
            break;
        case GGML_TYPE_Q3_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerQ3K>(mm);
            break;
        case GGML_TYPE_Q4_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerQ4K>(mm);
            break;
        case GGML_TYPE_Q5_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerQ5K>(mm);
            break;
        case GGML_TYPE_Q6_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerQ6K>(mm);
            break;
        case GGML_TYPE_IQ4_XS:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ4XS>(mm);
            break;
        case GGML_TYPE_IQ2_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ2K>(mm);
            break;
        case GGML_TYPE_IQ3_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ3K>(mm);
            break;
        case GGML_TYPE_IQ4_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ4K>(mm);
            break;
        case GGML_TYPE_IQ5_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ5K>(mm);
            break;
        case GGML_TYPE_IQ6_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ6K>(mm);
            break;
        case GGML_TYPE_IQ3_S:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ3S>(mm);
            break;
        case GGML_TYPE_IQ3_XXS:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ3XXS>(mm);
            break;
        case GGML_TYPE_IQ2_S:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ2S>(mm);
            break;
        case GGML_TYPE_IQ2_XS:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ2XS>(mm);
            break;
        case GGML_TYPE_IQ2_XXS:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ2XXS>(mm);
            break;
        case GGML_TYPE_IQ1_BN:
            assert (ne00 % QK_IQ1BN == 0);
            mm.funcs[0] = mul_mat_iq1bn_q8_K64<1, false>;
            mm.funcs[1] = mul_mat_iq1bn_q8_K64<2, false>;
            mm.funcs[2] = mul_mat_iq1bn_q8_K64<3, false>;
            mm.funcs[3] = mul_mat_iq1bn_q8_K64<4, false>;
            mm.funcs[4] = mul_mat_iq1bn_q8_K64<5, false>;
            mm.funcs[5] = mul_mat_iq1bn_q8_K64<6, false>;
            mm.funcs[6] = mul_mat_iq1bn_q8_K64<7, false>;
            mm.funcs[7] = mul_mat_iq1bn_q8_K64<8, false>;
            expected_typeB = GGML_TYPE_Q8_K64;
            break;
        case GGML_TYPE_IQ1_TN:
            assert (ne00 % QK_IQ1BN == 0);
            mm.funcs[0] = mul_mat_iq1bn_q8_K64<1, true>;
            mm.funcs[1] = mul_mat_iq1bn_q8_K64<2, true>;
            mm.funcs[2] = mul_mat_iq1bn_q8_K64<3, true>;
            mm.funcs[3] = mul_mat_iq1bn_q8_K64<4, true>;
            mm.funcs[4] = mul_mat_iq1bn_q8_K64<5, true>;
            mm.funcs[5] = mul_mat_iq1bn_q8_K64<6, true>;
            mm.funcs[6] = mul_mat_iq1bn_q8_K64<7, true>;
            mm.funcs[7] = mul_mat_iq1bn_q8_K64<8, true>;
            expected_typeB = GGML_TYPE_Q8_K64;
            break;
        case GGML_TYPE_IQ2_BN:
            assert (ne00 % QK_IQ1BN == 0);
            mm.funcs[0] = mul_mat_iq2bn_q8_K64<1>;
            mm.funcs[1] = mul_mat_iq2bn_q8_K64<2>;
            mm.funcs[2] = mul_mat_iq2bn_q8_K64<3>;
            mm.funcs[3] = mul_mat_iq2bn_q8_K64<4>;
            mm.funcs[4] = mul_mat_iq2bn_q8_K64<5>;
            mm.funcs[5] = mul_mat_iq2bn_q8_K64<6>;
            mm.funcs[6] = mul_mat_iq2bn_q8_K64<7>;
            mm.funcs[7] = mul_mat_iq2bn_q8_K64<8>;
            expected_typeB = GGML_TYPE_Q8_K64;
            break;
        case GGML_TYPE_Q4_0:
            assert (ne00 % QK4_0 == 0);
            //MulMat::set_functions<Q4_0_Unpacker>(mm);
            //expected_typeB = GGML_TYPE_Q8_0;
            MulMat::set_functions<Q4_0_1_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_1;
            break;
        case GGML_TYPE_Q4_1:
            assert (ne00 % QK4_1 == 0);
            MulMat::set_functions<Q4_1_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_1;
            break;
        case GGML_TYPE_Q5_0:
            assert (ne00 % QK5_0 == 0);
            //MulMat::set_functions<Q5_0_Unpacker>(mm);
            //expected_typeB = GGML_TYPE_Q8_0;
            MulMat::set_functions<Q5_0_1_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_1;
            break;
        case GGML_TYPE_Q5_1:
            assert (ne00 % QK5_1 == 0);
            MulMat::set_functions<Q5_1_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_1;
            break;
        case GGML_TYPE_Q8_0:
            assert (ne00 % QK8_0 == 0);
            //MulMat::set_functions<Q8_0_Unpacker>(mm);
            //expected_typeB = GGML_TYPE_Q8_0;
            MulMat::set_functions<Q8_0_1_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_1;
            break;
        case GGML_TYPE_IQ4_NL:
            assert (ne00 % QK4_NL == 0);
            MulMat::set_functions<IQ4_NL_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_0;
            break;

        default:
            return false;
    }

    return ggml_type(typeB) == expected_typeB;
}

} // namespace


#else   // __aarch64__

namespace {

template <int nrc, typename block_q8 = block_q8_K> struct Q8 {

    constexpr static int nrc_y = nrc;

    Q8(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8 *)info.src1_row(iy);
    }

    inline int8x16x2_t load_quants(int iy, int i, int j) const { return vld1q_s8_x2(y[iy][i].qs + 32*j); }
    inline int8x16x4_t load_quants_64(int iy, int i, int j) const { return vld1q_s8_x4(y[iy][i].qs + 64*j); }
    inline int16x8x2_t load_bsums(int iy, int i) const { return vld1q_s16_x2(y[iy][i].bsums); }
    inline int16x8_t load_bsums8(int iy, int i) const {
        auto q8s = vld1q_s16_x2(y[iy][i].bsums);
        return vpaddq_s16(q8s.val[0], q8s.val[1]);
    }
    inline float scale(int iy, int i) const { return y[iy][i].d; }

    const block_q8 * y[nrc_y];
};

template <typename Q8>
inline void compute_8_blocks(const uint8x16x4_t& qx_1, const uint8x16x4_t& qx_2, const Q8& q8,
        const int32x4x2_t& scales, int iy, int i, int j, int32x4_t& sumi) {
    auto mzero = vdupq_n_s32(0);
    auto q8b_1 = q8.load_quants(iy, i, 4*j+0);
    auto p1 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_1.val[0]), q8b_1.val[0]),
            vreinterpretq_s8_u8(qx_1.val[1]), q8b_1.val[1]); // block 1
    auto q8b_2 = q8.load_quants(iy, i, 4*j+1);
    auto p2 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_1.val[2]), q8b_2.val[0]),
            vreinterpretq_s8_u8(qx_1.val[3]), q8b_2.val[1]); // block 2
    auto p12 = vpaddq_s32(p1, p2);

    auto q8b_3 = q8.load_quants(iy, i, 4*j+2);
    auto p3 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_2.val[0]), q8b_3.val[0]),
            vreinterpretq_s8_u8(qx_2.val[1]), q8b_3.val[1]); // block 1
    auto q8b_4 = q8.load_quants(iy, i, 4*j+3);
    auto p4 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_2.val[2]), q8b_4.val[0]),
            vreinterpretq_s8_u8(qx_2.val[3]), q8b_4.val[1]); // block 2
    auto p34 = vpaddq_s32(p3, p4);

    auto pall = vpaddq_s32(p12, p34);
    sumi = vmlaq_s32(sumi, scales.val[j], pall);
}

template <typename Q8>
inline void compute_16_blocks(const uint8x16x4_t& qx_1, const uint8x16x4_t& qx_2, const Q8& q8,
        const int32x4x4_t& scales, int iy, int i, int j, int32x4_t& sumi) {

    auto mzero = vdupq_n_s32(0);
    auto q8b_1 = q8.load_quants(iy, i, 4*j+0);
    auto p1 = vpaddq_s32(ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_1.val[0]), q8b_1.val[0]),
                         ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_1.val[1]), q8b_1.val[1])); // blocks 0, 0, 1, 1,
    auto q8b_2 = q8.load_quants(iy, i, 4*j+1);
    auto p2 = vpaddq_s32(ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_1.val[2]), q8b_2.val[0]),
                         ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_1.val[3]), q8b_2.val[1])); // blocks 3, 3, 4, 4,
    auto p12 = vpaddq_s32(p1, p2); // blocks 0, 1, 2, 3
    sumi = vmlaq_s32(sumi, scales.val[2*j+0], p12);

    auto q8b_3 = q8.load_quants(iy, i, 4*j+2);
    auto p3 = vpaddq_s32(ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_2.val[0]), q8b_3.val[0]),
                         ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_2.val[1]), q8b_3.val[1])); // block 4, 4, 5, 5,
    auto q8b_4 = q8.load_quants(iy, i, 4*j+3);
    auto p4 = vpaddq_s32(ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_2.val[2]), q8b_4.val[0]),
                         ggml_vdotq_s32(mzero, vreinterpretq_s8_u8(qx_2.val[3]), q8b_4.val[1])); // block 6, 6, 7, 7,
    auto p34 = vpaddq_s32(p3, p4); // blocks 4, 5, 6, 7
    sumi = vmlaq_s32(sumi, scales.val[2*j+1], p34);
}

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

struct Q4bits {
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    uint8x16x4_t b1, b2;
    inline void prepare4(uint8x16x4_t& b, const uint8x16_t * val) const {
        b.val[0] = vandq_u8(val[0], m4b);
        b.val[2] = vshrq_n_u8(val[0], 4);
        b.val[1] = vandq_u8(val[1], m4b);
        b.val[3] = vshrq_n_u8(val[1], 4);
    }
    inline void prepare4_16(uint8x16x4_t& b, const uint8x16_t * val) const {
        b.val[0] = vandq_u8(val[0], m4b);
        b.val[1] = vshrq_n_u8(val[0], 4);
        b.val[2] = vandq_u8(val[1], m4b);
        b.val[3] = vshrq_n_u8(val[1], 4);
    }
    inline void prepare(const uint8_t * qs) {
        auto q4bits = vld1q_u8_x2(qs);
        prepare4(b1, q4bits.val);
        q4bits = vld1q_u8_x2(qs+32);
        prepare4(b2, q4bits.val);
    }
    inline void prepare_v2(const uint8_t * qs) {
        auto q4bits = vld1q_u8_x4(qs);
        prepare4(b1, q4bits.val+0);
        prepare4(b2, q4bits.val+2);
    }
    inline void prepare64(const uint8_t * qs) {
        auto q4bits = vld1q_u8_x4(qs);
        b1.val[0] = vandq_u8(q4bits.val[0], m4b);
        b1.val[1] = vandq_u8(q4bits.val[1], m4b);
        b1.val[2] = vandq_u8(q4bits.val[2], m4b);
        b1.val[3] = vandq_u8(q4bits.val[3], m4b);
        b2.val[0] = vshrq_n_u8(q4bits.val[0], 4);
        b2.val[1] = vshrq_n_u8(q4bits.val[1], 4);
        b2.val[2] = vshrq_n_u8(q4bits.val[2], 4);
        b2.val[3] = vshrq_n_u8(q4bits.val[3], 4);
    }
    inline void prepare16(const uint8_t * qs) {
        auto q4bits = vld1q_u8_x2(qs);
        prepare4_16(b1, q4bits.val);
        q4bits = vld1q_u8_x2(qs+32);
        prepare4_16(b2, q4bits.val);
    }
    inline void prepare16_v2(const uint8_t * qs) {
        auto q4bits = vld1q_u8_x4(qs);
        prepare4_16(b1, q4bits.val+0);
        prepare4_16(b2, q4bits.val+2);
    }
};

struct Q2bits {
    const uint8x16_t m4b = vdupq_n_u8(0x03);
    uint8x16x4_t b1, b2;
    inline void prepare(const uint8_t * qs) {
        auto q2bits = vld1q_u8_x2(qs);
        b1.val[0] = vandq_u8(q2bits.val[0], m4b);
        b1.val[1] = vandq_u8(q2bits.val[1], m4b);

        q2bits.val[0] = vshrq_n_u8(q2bits.val[0], 2);
        q2bits.val[1] = vshrq_n_u8(q2bits.val[1], 2);
        b1.val[2] = vandq_u8(q2bits.val[0], m4b);
        b1.val[3] = vandq_u8(q2bits.val[1], m4b);

        q2bits.val[0] = vshrq_n_u8(q2bits.val[0], 2);
        q2bits.val[1] = vshrq_n_u8(q2bits.val[1], 2);
        b2.val[0] = vandq_u8(q2bits.val[0], m4b);
        b2.val[1] = vandq_u8(q2bits.val[1], m4b);

        q2bits.val[0] = vshrq_n_u8(q2bits.val[0], 2);
        q2bits.val[1] = vshrq_n_u8(q2bits.val[1], 2);
        b2.val[2] = vandq_u8(q2bits.val[0], m4b);
        b2.val[3] = vandq_u8(q2bits.val[1], m4b);
    }
};

template <typename block_q>
struct BaseDequantizer {
    BaseDequantizer(const void * vx, size_t bx, int nrc) : vx(vx), x(nullptr), bx(bx), nrc(nrc) {}
    inline void new_row(int ix) { x = (const block_q *)((const char *)vx + ix*bx); }
    const void * vx;
    const block_q * x;
    const size_t bx;
    const int nrc;
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

    float d;
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

    float d;
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

    float d;
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

    float d;
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

    float d;
};

// ============================= i-quants

inline int32x4x4_t make_wider_8(const int8x16_t& scales8) {
    int16x8x2_t scales16{vmovl_s8(vget_low_s8(scales8)), vmovl_s8(vget_high_s8(scales8))};
    return make_wider(scales16);
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

    float d;
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

    float d;
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

    float d;
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
        int8x16_t scales = vaddq_s8(vreinterpretq_s8_u8(vshlq_n_u8(scl8, 1)), vdupq_n_s8(-15));
        return vqtbl1q_s8(scales, hshuff);
    }

    Q2bits bits;
    const int8x16_t values = vreinterpretq_s8_u64(vdupq_n_u64(0x000000001101f3e1));
    const uint8x16_t hshuff = vreinterpretq_u8_u32(uint32x4_t{0x09010800, 0x0b030a02, 0x0d050c04, 0x0f070e06});

    float d;
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

    float d;
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

    float d;
};

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

    float d;
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

    float d;

};

struct SignHelper {

    inline void init() { shuffle = vcombine_u8(vdup_n_u8(0), vdup_n_u8(1)); }

    inline void apply_signs_1(uint8x16_t * b, const uint8x16_t& signs16) {
        auto aux = vqtbl1q_u8(signs16, shuffle);
        auto s = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(vandq_u8(aux, smask), smask), m1));
        b[0] = vreinterpretq_u8_s8(vmulq_s8(vreinterpretq_s8_u8(b[0]), s));
        shuffle = vaddq_u8(shuffle, step);
    }

    const uint8x16_t smask = vreinterpretq_u8_u64(vdupq_n_u64(0x8040201008040201));
    const uint8x16_t m1    = vdupq_n_u8(1);
    const uint8x16_t step  = vdupq_n_u8(2);
    uint8x16_t shuffle;
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

    float d;

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

    float d;

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

    float d;

};

struct DequantizerIQ2TN final : public BaseDequantizer<block_iq2_tn> {
    DequantizerIQ2TN(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc) {}

    constexpr static int num_blocks() { return 16; }
    constexpr static bool should_scale_quants() { return true; }

    //template <typename Q8>
    //inline void process_scales(int i, [[maybe_unused]] const Q8& q8, [[maybe_unused]] float32x4_t * acc) {
    //    d = GGML_FP16_TO_FP32(x[i].d);
    //}

    inline void new_block(int i) {
        d = GGML_FP16_TO_FP32(x[i].d);
    }

    template <typename Q8>
    inline void compute(const Q8& q8, int i, int j, int32x4_t * sumi) {
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
    template <typename Q8>
    inline void compute1(const Q8& q8, int i, int j, int32x4_t * sumi) {
        auto q8b_1 = q8.load_quants(0, i, 4*j+0);
        sumi[0] = ggml_vdotq_s32(ggml_vdotq_s32(sumi[0], vreinterpretq_s8_u8(bits.b1.val[0]), q8b_1.val[0]),
                vreinterpretq_s8_u8(bits.b1.val[1]), q8b_1.val[1]);

        auto q8b_2 = q8.load_quants(0, i, 4*j+1);
        sumi[1] = ggml_vdotq_s32(ggml_vdotq_s32(sumi[1], vreinterpretq_s8_u8(bits.b1.val[2]), q8b_2.val[0]),
                vreinterpretq_s8_u8(bits.b1.val[3]), q8b_2.val[1]);

        q8b_1 = q8.load_quants(0, i, 4*j+2);
        sumi[0] = ggml_vdotq_s32(ggml_vdotq_s32(sumi[0], vreinterpretq_s8_u8(bits.b2.val[0]), q8b_1.val[0]),
                vreinterpretq_s8_u8(bits.b2.val[1]), q8b_1.val[1]);

        q8b_2 = q8.load_quants(0, i, 4*j+3);
        sumi[1] = ggml_vdotq_s32(ggml_vdotq_s32(sumi[1], vreinterpretq_s8_u8(bits.b2.val[2]), q8b_2.val[0]),
                vreinterpretq_s8_u8(bits.b2.val[3]), q8b_2.val[1]);
    }

    IQK_ALWAYS_INLINE void prepare(int i, int j) {
        bits.prepare(x[i].qs+32*j);
        auto m1 = vdupq_n_s8(1);
        for (int k = 0; k < 4; ++k) {
            bits.b1.val[k] = vsubq_s8(bits.b1.val[k], m1);
            bits.b2.val[k] = vsubq_s8(bits.b2.val[k], m1);
        }
    }

    Q2bits bits;

    float d;
};

template <int nrc_y>
void mul_mat_iq2tn_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y, block_q8_K> q8(info);

    DequantizerIQ2TN deq(vx, bx, nrc_y);
    float32x4_t acc[nrc_y];

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            int32x4_t sumi[nrc_y];
            for (int iy = 0; iy < nrc_y; ++iy) sumi[iy] = vdupq_n_s32(0);

            deq.new_block(i);
            deq.prepare(i, 0);
            deq.compute(q8, i, 0, sumi);
            deq.prepare(i, 1);
            deq.compute(q8, i, 1, sumi);

            if (i > 0) {
                for (int iy = 0; iy < nrc_y; ++iy) {
                    acc[iy] = vmlaq_f32(acc[iy], vcvtq_f32_s32(sumi[iy]), vdupq_n_f32(deq.d*q8.scale(iy, i)));
                }
            } else {
                for (int iy = 0; iy < nrc_y; ++iy) {
                    acc[iy] = vmulq_f32(vcvtq_f32_s32(sumi[iy]), vdupq_n_f32(deq.d*q8.scale(iy, i)));
                }
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, vaddvq_f32(acc[iy]));
        }
    }
}
void mul_mat_iq2tn_K_q8_K_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<1, block_q8_K> q8(info);

    DequantizerIQ2TN deq(vx, bx, 1);

    auto m1 = vdup_n_s16(-1);
    float32x4_t acc[2];

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);

        for (int i = 0; i < nb; ++i) {

            int32x4_t sumi[2] = {};
            deq.new_block(i);
            auto bsums = q8.load_bsums(0, i);
            bsums.val[0] = vaddq_s32(bsums.val[0], bsums.val[1]);
            sumi[0] = vmlal_s16(sumi[0], vget_low_s16 (bsums.val[0]), m1);
            sumi[1] = vmlal_s16(sumi[1], vget_high_s16(bsums.val[0]), m1);
            deq.bits.prepare(deq.x[i].qs);
            deq.compute1(q8, i, 0, sumi);
            deq.bits.prepare(deq.x[i].qs+32);
            deq.compute1(q8, i, 1, sumi);

            auto vd = vdupq_n_f32(deq.d*q8.scale(0, i));
            if (i > 0) {
                acc[0] = vmlaq_f32(acc[0], vcvtq_f32_s32(sumi[0]), vd);
                acc[1] = vmlaq_f32(acc[1], vcvtq_f32_s32(sumi[1]), vd);
            } else {
                acc[0] = vmulq_f32(vcvtq_f32_s32(sumi[0]), vd);
                acc[1] = vmulq_f32(vcvtq_f32_s32(sumi[1]), vd);
            }

        }

        acc[0] = vaddq_f32(acc[0], acc[1]);
        info.store(ix, 0, vaddvq_f32(acc[0]));
    }
}


template <int nrc_y, typename Dequantizer>
void mul_mat_qX_K_q8_K_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    assert(n % QK_K == 0);
    const int nb = n / QK_K;

    Q8<nrc_y, block_q8_K> q8(info);

    Dequantizer deq(vx, bx, nrc_y);

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);

        float32x4_t acc[nrc_y];
        for (int iy = 0; iy < nrc_y; ++iy) acc[iy] = vdupq_n_f32(0.f);

        for (int i = 0; i < nb; ++i) {

            int32x4_t sumi[nrc_y];
            for (int iy = 0; iy < nrc_y; ++iy) sumi[iy] = vdupq_n_s32(0);

            if constexpr (nrc_y > 1 && Dequantizer::should_scale_quants()) {
                deq.process_scales(i, q8, acc);
                deq.prepare(i, 0);
                deq.compute(q8, i, 0, sumi);
                deq.prepare(i, 1);
                deq.compute(q8, i, 1, sumi);
            } else {
                if constexpr (Dequantizer::num_blocks() == 8) {
                    auto scales = deq.new_block(i, q8, acc);
                    deq.prepare(i, 0);
                    for (int iy = 0; iy < nrc_y; ++iy) compute_8_blocks(deq.bits.b1, deq.bits.b2, q8, scales, iy, i, 0, sumi[iy]);
                    deq.prepare(i, 1);
                    for (int iy = 0; iy < nrc_y; ++iy) compute_8_blocks(deq.bits.b1, deq.bits.b2, q8, scales, iy, i, 1, sumi[iy]);
                }
                else if constexpr (Dequantizer::num_blocks() == 16) {
                    auto scales = deq.new_block(i, q8, acc);
                    deq.prepare(i, 0);
                    for (int iy = 0; iy < nrc_y; ++iy) compute_16_blocks(deq.bits.b1, deq.bits.b2, q8, scales, iy, i, 0, sumi[iy]);
                    deq.prepare(i, 1);
                    for (int iy = 0; iy < nrc_y; ++iy) compute_16_blocks(deq.bits.b1, deq.bits.b2, q8, scales, iy, i, 1, sumi[iy]);
                }
                else {
                    GGML_ASSERT(false);
                }
            }

            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vmlaq_f32(acc[iy], vcvtq_f32_s32(sumi[iy]), vdupq_n_f32(deq.d*q8.scale(iy, i)));
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, vaddvq_f32(acc[iy]));
        }
    }
}

// =========================================== Legacy quants

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
        float d = GGML_FP16_TO_FP32(deq.x[i].d);
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

    const int8x16_t m8 = vdupq_n_s8(-8);
    //ggml_half aux[4];
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
        //for (int i = 4*(nb/4); i < nb; ++i) {
        //    q8.process_1_block(i, deq, acc);
        //}

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
        if (nrc_x%2 == 0) {
            Dequantizer deq1(vx, bx), deq2(vx, bx);
            mul_mat_qX_Y_q8_Y_IK(n, deq1, deq2, q8, info, nrc_x);
        } else {
            Dequantizer deq(vx, bx);
            mul_mat_qX_Y_q8_Y(n, deq, q8, info, nrc_x);
        }
        //Dequantizer deq(vx, bx);
        //mul_mat_qX_Y_q8_Y(n, deq, q8, info, nrc_x);
    }
}

template <typename Dequantizer, int nrc_y>
static void mul_mat_qX_0_q8_0(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    Q80<nrc_y> q8(info);
    if constexpr (nrc_y == 1) {
        Dequantizer deq1(vx, bx), deq2(vx, bx);
        mul_mat_qX_Y_q8_Y_1(n, deq1, deq2, q8, info, nrc_x);
    } else {
        if (nrc_x%2 == 0) {
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

struct QF16Base {
    constexpr static int k_step = 8;
    using Data = float16x8_t;
    using Acc  = float16x8_t;
    static inline Data load(const __fp16 * x) { return vld1q_f16(x); }
    static inline Data load4(const __fp16 * x) { return vcombine_f16(vld1_f16(x), vdup_n_f16(0)); }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) {
        return vfmaq_f16(prev, y, x);
    }
    static inline Acc acc_first(const Data& y, const Data& x) {
        return vmulq_f16(y, x);
    }
    //constexpr static int k_step = 16;
    //using Data = float16x8x2_t;
    //static inline Data load(const __fp16 * x) { return vld1q_f16_x2(x); }
    //static inline Acc acc(Acc prev, const Data& y, const Data& x) {
    //    return vfmaq_f16(vfmaq_f16(prev, y.val[0], x.val[0]), y.val[1], x.val[1]);
    //}
    //static inline Acc acc_first(const Data& y, const Data& x) {
    //    return vfmaq_f16(vmulq_f16(y.val[0], x.val[0]), y.val[1], x.val[1]);
    //}
    static inline float hsum(Acc acc) {
        float32x4_t sum = vcvt_f32_f16(vadd_f16(vget_low_f16(acc), vget_high_f16(acc)));
        return vaddvq_f32(sum);
    }
};
template <int nrc> struct QF16 final : public QF16Base {
    using Base = QF16Base;
    constexpr static int nrc_y = nrc;
    QF16(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const __fp16 *)info.src1_row(iy);
    }
    QF16(const char * cx, size_t bx) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const __fp16 *)(cx + iy*bx);
    }
    IQK_ALWAYS_INLINE Data load1(int iy, int i) const { return load(y[iy] + k_step*i); }
    IQK_ALWAYS_INLINE Data load_tail(int iy, int i) const { return load4(y[iy] + 4*i); }
    IQK_ALWAYS_INLINE float16x8x4_t loadx(int iy, int i) const { return vld1q_f16_x4(y[iy] + 4*k_step*i); }
    const __fp16 * y[nrc_y];
};

struct QBF16Base {
    constexpr static int k_step = 4;
    using Data = float32x4_t;
    using Acc  = float32x4_t;
    static inline Data load(const uint16_t * x) { return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vld1_u16(x)), 16)); }
    static inline Data load4(const uint16_t * x) { return load(x); }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) {
        return vfmaq_f32(prev, y, x);
    }
    static inline Acc acc_first(const Data& y, const Data& x) {
        return vmulq_f32(y, x);
    }
    static inline float hsum(Acc acc) { return vaddvq_f32(acc); }
};
template <int nrc> struct QBF16 final : public QBF16Base {
    using Base = QBF16Base;
    constexpr static int nrc_y = nrc;
    QBF16(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const uint16_t *)info.src1_row(iy);
    }
    QBF16(const char * cx, size_t bx) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const uint16_t *)(cx + iy*bx);
    }
    IQK_ALWAYS_INLINE Data load1(int iy, int i) const { return load(y[iy] + k_step*i); }
    IQK_ALWAYS_INLINE Data load_tail(int iy, int i) const { return load(y[iy] + 4*i); }
    const uint16_t * y[nrc_y];
};

struct QF32Base {
    constexpr static int k_step = 4;
    using Data = float32x4_t;
    using Acc  = float32x4_t;
    static inline Data load(const float * x) { return vld1q_f32(x); }
    static inline Data load4(const float * x) { return load(x); }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) { return vfmaq_f32(prev, y, x); }
    static inline Acc acc_first(const Data& y, const Data& x) { return vmulq_f32(y, x); }
    static inline float hsum(Acc acc) { return vaddvq_f32(acc); }
};
template <int nrc> struct QF32 final : public QF32Base {
    using Base = QF32Base;
    constexpr static int nrc_y = nrc;
    QF32(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const float *)info.src1_row(iy);
    }
    QF32(const char * cx, size_t bx) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const float *)(cx + iy*bx);
    }
    IQK_ALWAYS_INLINE Data load1(int iy, int i) const { return load(y[iy] + k_step*i); }
    IQK_ALWAYS_INLINE Data load_tail(int iy, int i) const { return load(y[iy] + 4*i); }
    const float * y[nrc_y];
};

template <typename Qy, typename Qx, bool is_multiple_of_k_step>
IQK_NOINLINE void mul_mat_Qx_Qy_NxN(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    GGML_ASSERT(Qx::Base::k_step == Qy::Base::k_step);
    int nb = n/Qx::Base::k_step;
    Qy y(info);
    Qx x(cx + ix0*bx, bx);
    typename Qx::Base::Data xv[Qx::nrc_y];
    typename Qx::Base::Acc  acc[Qx::nrc_y*Qy::nrc_y];
    auto yv = y.load1(0, 0);
    for (int ix = 0; ix < Qx::nrc_y; ++ix) {
        xv[ix] = x.load1(ix, 0);
        acc[ix] = Qx::Base::acc_first(yv, xv[ix]);
    }
    for (int iy = 1; iy < Qy::nrc_y; ++iy) {
        yv = y.load1(iy, 0);
        for (int ix = 0; ix < Qx::nrc_y; ++ix) acc[Qx::nrc_y*iy + ix] = Qx::Base::acc_first(yv, xv[ix]);
    }
    for (int i = 1; i < nb; ++i) {
        yv = y.load1(0, i);
        for (int ix = 0; ix < Qx::nrc_y; ++ix) {
            xv[ix] = x.load1(ix, i);
            acc[ix] = Qx::Base::acc(acc[ix], yv, xv[ix]);
        }
        for (int iy = 1; iy < Qy::nrc_y; ++iy) {
            yv = y.load1(iy, i);
            for (int ix = 0; ix < Qx::nrc_y; ++ix) acc[Qx::nrc_y*iy + ix] = Qx::Base::acc(acc[Qx::nrc_y*iy + ix], yv, xv[ix]);
        }
    }
    if constexpr (Qx::Base::k_step > 4 && !is_multiple_of_k_step) {
        int nb4 = n/4;
        for (int i = (Qx::Base::k_step/4)*nb; i < nb4; ++i) {
            yv = y.load_tail(0, i);
            for (int ix = 0; ix < Qx::nrc_y; ++ix) {
                xv[ix] = x.load_tail(ix, i);
                acc[ix] = Qx::Base::acc(acc[ix], yv, xv[ix]);
            }
            for (int iy = 1; iy < Qy::nrc_y; ++iy) {
                yv = y.load_tail(iy, i);
                for (int ix = 0; ix < Qx::nrc_y; ++ix) acc[Qx::nrc_y*iy + ix] = Qx::Base::acc(acc[Qx::nrc_y*iy + ix], yv, xv[ix]);
            }
        }
    }
    for (int iy = 0; iy < Qy::nrc_y; ++iy) for (int ix = 0; ix < Qx::nrc_y; ++ix) info.store(ix0+ix, iy, Qx::Base::hsum(acc[Qx::nrc_y*iy+ix]));
}

template <int nrc_y, int nrc_x, bool is_multiple_of_k_step>
IQK_NOINLINE void mul_mat_f16_f16_NxN(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    assert(n%QF16Base::k_step == 0);
    int nb = n/QF16Base::k_step;
    QF16<nrc_y> y(info);
    QF16<nrc_x> x(cx + ix0*bx, bx);
    QF16Base::Data xv[nrc_x];
    QF16Base::Acc  acc[nrc_x*nrc_y];
    auto yv = y.load1(0, 0);
    for (int ix = 0; ix < nrc_x; ++ix) {
        xv[ix] = x.load1(ix, 0);
        acc[ix] = QF16Base::acc_first(yv, xv[ix]);
    }
    for (int iy = 1; iy < nrc_y; ++iy) {
        yv = y.load1(iy, 0);
        for (int ix = 0; ix < nrc_x; ++ix) acc[nrc_x*iy + ix] = QF16Base::acc_first(yv, xv[ix]);
    }
    for (int i = 1; i < nb; ++i) {
        yv = y.load1(0, i);
        for (int ix = 0; ix < nrc_x; ++ix) {
            xv[ix] = x.load1(ix, i);
            acc[ix] = QF16Base::acc(acc[ix], yv, xv[ix]);
        }
        for (int iy = 1; iy < nrc_y; ++iy) {
            yv = y.load1(iy, i);
            for (int ix = 0; ix < nrc_x; ++ix) acc[nrc_x*iy + ix] = QF16Base::acc(acc[nrc_x*iy + ix], yv, xv[ix]);
        }
    }
    if constexpr (!is_multiple_of_k_step) {
        int nb4 = n/4;
        for (int i = (QF16Base::k_step/4)*nb; i < nb4; ++i) {
            yv = y.load_tail(0, i);
            for (int ix = 0; ix < nrc_x; ++ix) {
                xv[ix] = x.load_tail(ix, i);
                acc[ix] = QF16Base::acc(acc[ix], yv, xv[ix]);
            }
            for (int iy = 1; iy < nrc_y; ++iy) {
                yv = y.load_tail(iy, i);
                for (int ix = 0; ix < nrc_x; ++ix) acc[nrc_x*iy + ix] = QF16Base::acc(acc[nrc_x*iy + ix], yv, xv[ix]);
            }
        }
    }
    for (int iy = 0; iy < nrc_y; ++iy) for (int ix = 0; ix < nrc_x; ++ix) info.store(ix0+ix, iy, QF16Base::hsum(acc[nrc_x*iy+ix]));
}

template <typename Qy, template<int> typename Qx>
void mul_mat_Qx_Qy_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%4 == 0);
    constexpr int k_nx = 5;
    const char * cx = (const char *)vx;
    if (n%Qx<k_nx>::Base::k_step == 0) {
        for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
            mul_mat_Qx_Qy_NxN<Qy, Qx<k_nx>, true>(n, cx, bx, ix*k_nx, info);
        }
        int last_x = k_nx*(nrc_x/k_nx);
        if (last_x == nrc_x) return;
        int nx = nrc_x - last_x;
        switch (nx) {
            case 1: mul_mat_Qx_Qy_NxN<Qy, Qx<1>, true>(n, cx, bx, last_x, info); break;
            case 2: mul_mat_Qx_Qy_NxN<Qy, Qx<2>, true>(n, cx, bx, last_x, info); break;
            case 3: mul_mat_Qx_Qy_NxN<Qy, Qx<3>, true>(n, cx, bx, last_x, info); break;
            case 4: mul_mat_Qx_Qy_NxN<Qy, Qx<4>, true>(n, cx, bx, last_x, info); break;
        }
    } else {
        for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
            mul_mat_Qx_Qy_NxN<Qy, Qx<k_nx>, false>(n, cx, bx, ix*k_nx, info);
        }
        int last_x = k_nx*(nrc_x/k_nx);
        if (last_x == nrc_x) return;
        int nx = nrc_x - last_x;
        switch (nx) {
            case 1: mul_mat_Qx_Qy_NxN<Qy, Qx<1>, false>(n, cx, bx, last_x, info); break;
            case 2: mul_mat_Qx_Qy_NxN<Qy, Qx<2>, false>(n, cx, bx, last_x, info); break;
            case 3: mul_mat_Qx_Qy_NxN<Qy, Qx<3>, false>(n, cx, bx, last_x, info); break;
            case 4: mul_mat_Qx_Qy_NxN<Qy, Qx<4>, false>(n, cx, bx, last_x, info); break;
        }
    }
}

template <int nrc_y>
void mul_mat_f16_f16_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%4 == 0);
    constexpr int k_nx = 5;
    const char * cx = (const char *)vx;
    if (n%QF16Base::k_step == 0) {
        for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
            mul_mat_f16_f16_NxN<nrc_y, k_nx, true>(n, cx, bx, ix*k_nx, info);
        }
        int last_x = k_nx*(nrc_x/k_nx);
        if (last_x == nrc_x) return;
        int nx = nrc_x - last_x;
        switch (nx) {
            case 1: mul_mat_f16_f16_NxN<nrc_y, 1, true>(n, cx, bx, last_x, info); break;
            case 2: mul_mat_f16_f16_NxN<nrc_y, 2, true>(n, cx, bx, last_x, info); break;
            case 3: mul_mat_f16_f16_NxN<nrc_y, 3, true>(n, cx, bx, last_x, info); break;
            case 4: mul_mat_f16_f16_NxN<nrc_y, 4, true>(n, cx, bx, last_x, info); break;
        }
    } else {
        for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
            mul_mat_f16_f16_NxN<nrc_y, k_nx, false>(n, cx, bx, ix*k_nx, info);
        }
        int last_x = k_nx*(nrc_x/k_nx);
        if (last_x == nrc_x) return;
        int nx = nrc_x - last_x;
        switch (nx) {
            case 1: mul_mat_f16_f16_NxN<nrc_y, 1, false>(n, cx, bx, last_x, info); break;
            case 2: mul_mat_f16_f16_NxN<nrc_y, 2, false>(n, cx, bx, last_x, info); break;
            case 3: mul_mat_f16_f16_NxN<nrc_y, 3, false>(n, cx, bx, last_x, info); break;
            case 4: mul_mat_f16_f16_NxN<nrc_y, 4, false>(n, cx, bx, last_x, info); break;
        }
    }
}

template <int nrc_x, bool is_multiple_of_k_step>
IQK_NOINLINE void mul_mat_f16_f16_Nx1(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    assert(n%QF16Base::k_step == 0);
    int nb = n/QF16Base::k_step;
    QF16<1> y(info);
    QF16<nrc_x> x(cx + ix0*bx, bx);
    QF16Base::Acc  acc[4*nrc_x];
    auto yv = y.loadx(0, 0);
    for (int ix = 0; ix < nrc_x; ++ix) {
        for (int k = 0; k < 4; ++k) {
            auto xv = x.load1(ix, k);
            acc[4*ix+k] = QF16Base::acc_first(yv.val[k], xv);
        }
    }
    for (int i = 1; i < nb/4; ++i) {
        yv = y.loadx(0, i);
        for (int ix = 0; ix < nrc_x; ++ix) {
            for (int k = 0; k < 4; ++k) {
                auto xv = x.load1(ix, 4*i+k);
                acc[4*ix+k] = QF16Base::acc(acc[4*ix+k], yv.val[k], xv);
            }
        }
    }
    for (int i = 4*(nb/4); i < nb; ++i) {
        auto yv1 = y.load1(0, i);
        for (int ix = 0; ix < nrc_x; ++ix) {
            auto xv1 = x.load1(ix, i);
            acc[4*ix] = QF16Base::acc(acc[4*ix], yv1, xv1);
        }
    }
    if constexpr (!is_multiple_of_k_step) {
        int nb4 = n/4;
        for (int i = (QF16Base::k_step/4)*nb; i < nb4; ++i) {
            auto yv1 = y.load_tail(0, i);
            for (int ix = 0; ix < nrc_x; ++ix) {
                auto xv1 = x.load_tail(ix, i);
                acc[4*ix] = QF16Base::acc(acc[4*ix], yv1, xv1);
            }
        }
    }
    for (int ix = 0; ix < nrc_x; ++ix) {
        auto v1 = vaddq_f16(acc[4*ix+0], acc[4*ix+1]);
        auto v2 = vaddq_f16(acc[4*ix+2], acc[4*ix+3]);
        info.store(ix0+ix, 0, QF16Base::hsum(vaddq_f16(v1, v2)));
    }
}

// At least on my M2-Max the version below, which does the multiplication row-by-row, is faster.
// But let's keep this version commented out for now.
//void mul_mat_f16_f16_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
//    GGML_ASSERT(n%4 == 0);
//    constexpr int k_nx = 2;
//    const char * cx = (const char *)vx;
//    if (n%QF16Base::k_step == 0) {
//        for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
//            mul_mat_f16_f16_Nx1<k_nx, true>(n, cx, bx, ix*k_nx, info);
//        }
//        int last_x = k_nx*(nrc_x/k_nx);
//        if (last_x == nrc_x) return;
//        int nx = nrc_x - last_x;
//        switch (nx) {
//            case 1: mul_mat_f16_f16_Nx1<1, true>(n, cx, bx, last_x, info); break;
//            //case 2: mul_mat_f16_f16_Nx1<2, true>(n, cx, bx, last_x, info); break;
//            //case 3: mul_mat_f16_f16_Nx1<3, true>(n, cx, bx, last_x, info); break;
//        }
//    } else {
//        for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
//            mul_mat_f16_f16_Nx1<k_nx, false>(n, cx, bx, ix*k_nx, info);
//        }
//        int last_x = k_nx*(nrc_x/k_nx);
//        if (last_x == nrc_x) return;
//        int nx = nrc_x - last_x;
//        switch (nx) {
//            case 1: mul_mat_f16_f16_Nx1<1, false>(n, cx, bx, last_x, info); break;
//            //case 2: mul_mat_f16_f16_Nx1<2, false>(n, cx, bx, last_x, info); break;
//            //case 3: mul_mat_f16_f16_Nx1<3, false>(n, cx, bx, last_x, info); break;
//        }
//    }
//}

void mul_mat_f16_f16_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%4 == 0);
    const char * cx = (const char *)vx;
    if (n%QF16Base::k_step == 0) {
        for (int ix = 0; ix < nrc_x; ++ix) {
            mul_mat_f16_f16_Nx1<1, true>(n, cx, bx, ix, info);
        }
    } else {
        for (int ix = 0; ix < nrc_x; ++ix) {
            mul_mat_f16_f16_Nx1<1, false>(n, cx, bx, ix, info);
        }
    }
}

template <int nrc> struct Q8_K64 {

    constexpr static int nrc_y = nrc;

    Q8_K64(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto dptr = (const float *)info.src1_row(iy);
            std::memcpy(d + 8*iy, dptr, 8*sizeof(float));
            y[iy] = (const int8_t *)(dptr + 8);
        }
    }

    inline int8x16x4_t load_quants64(int iy, int i, int j) const { return vld1q_s8_x4(y[iy] + 128*i + 64*j); }
    inline int8x16x2_t load_quants(int iy, int i, int j) const { return vld1q_s8_x2(y[iy] + 128*i + 32*j); }
    inline float32x4_t scale(int iy) const { return vld1q_f32(d + 8*iy); }
    inline float32x4_t minus(int iy) const { return vld1q_f32(d + 8*iy + 4); }

    float d[8*nrc_y];
    const int8_t * y[nrc_y];
};

struct DequantizerIQ1BN {
    const uint8x16_t m1 = vdupq_n_u8(1);

    static inline uint8x16x4_t load_shuffles() {
        static const uint8_t data[64] = {0, 0, 0, 0, 0,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2, 12,
                                         3, 3, 3, 3, 3,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5, 12,
                                         6, 6, 6, 6, 6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8, 12,
                                         9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12};
        return vld1q_u8_x4(data);
    }
    static inline uint8x16x4_t load_mult() {
        static const uint8_t data[64] = {81, 27, 9, 3, 1, 81, 27, 9, 3, 1, 81, 27, 9, 3, 1, 81,
                                         81, 27, 9, 3, 1, 81, 27, 9, 3, 1, 81, 27, 9, 3, 1, 27,
                                         81, 27, 9, 3, 1, 81, 27, 9, 3, 1, 81, 27, 9, 3, 1,  9,
                                         81, 27, 9, 3, 1, 81, 27, 9, 3, 1, 81, 27, 9, 3, 1,  3};
        return vld1q_u8_x4(data);
    }
    const uint8x16x4_t shuff = load_shuffles();
    const uint8x16x4_t mult  = load_mult();

    IQK_ALWAYS_INLINE void prepare_iq1bn_quants(const block_iq1_bn * x, int8x16x4_t& v) const {
        auto data = vld1q_u8((const uint8_t *)x);
        for (int k = 0; k < 4; ++k) {
            auto val = vmulq_u8(vqtbl1q_u8(data, shuff.val[k]), mult.val[k]);
            val = vshrq_n_u8(vhaddq_u8(val, vshrq_n_u8(val, 1)), 6);
            v.val[k] = vsubq_s8(vreinterpretq_s8_u8(val), m1);
        }
    }

    IQK_ALWAYS_INLINE void prepare_iq1bn_quants_nosub(const block_iq1_bn * x, int8x16x4_t& v) const {
        auto data = vld1q_u8((const uint8_t *)x);
        for (int k = 0; k < 4; ++k) {
            auto val = vmulq_u8(vqtbl1q_u8(data, shuff.val[k]), mult.val[k]);
            v.val[k] = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(val, vshrq_n_u8(val, 1)), 6));
        }
    }
};

template <int nrc_y, bool is_iq1_tn>
static void mul_mat_iq1bn_q8_K64(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const int nb = n / QK_IQ1BN;

    Q8_K64<nrc_y> q8(info);
    DequantizerIQ1BN deq;

    int32x4_t   accd[nrc_y];
    int8x16x4_t v1, v2;

    float scale;

    for (int ix = 0; ix < nrc_x; ++ix) {

        const char * cx = ((const char *)vx + ix*bx);
        if constexpr (is_iq1_tn) {
            scale = GGML_FP16_TO_FP32(*(const ggml_half *)cx);
            cx += sizeof(ggml_half);
        }

        const block_iq1_bn * x = (const block_iq1_bn *)cx;

        if constexpr (nrc_y == 1) {
            int32x4_t acc[4] = {};
            for (int i = 0; i < nb/2; ++i) {
                deq.prepare_iq1bn_quants_nosub(x+2*i+0, v1);
                auto q = q8.load_quants64(0, i, 0);
                for (int j = 0; j < 4; ++j) acc[j] = ggml_vdotq_s32(acc[j], q.val[j], v1.val[j]);
                deq.prepare_iq1bn_quants_nosub(x+2*i+1, v2);
                q = q8.load_quants64(0, i, 1);
                for (int j = 0; j < 4; ++j) acc[j] = ggml_vdotq_s32(acc[j], q.val[j], v2.val[j]);
            }
            accd[0] = vaddq_s32(vaddq_s32(acc[0], acc[1]), vaddq_s32(acc[2], acc[3]));
        }
        else {

            for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = vdupq_n_s32(0);

            for (int i = 0; i < nb/2; ++i) {

                deq.prepare_iq1bn_quants_nosub(x+2*i+0, v1);
                deq.prepare_iq1bn_quants_nosub(x+2*i+1, v2);

                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto q = q8.load_quants(iy, i, 0);
                    accd[iy] = ggml_vdotq_s32(ggml_vdotq_s32(accd[iy], q.val[0], v1.val[0]), q.val[1], v1.val[1]);
                    q = q8.load_quants(iy, i, 1);
                    accd[iy] = ggml_vdotq_s32(ggml_vdotq_s32(accd[iy], q.val[0], v1.val[2]), q.val[1], v1.val[3]);
                    q = q8.load_quants(iy, i, 2);
                    accd[iy] = ggml_vdotq_s32(ggml_vdotq_s32(accd[iy], q.val[0], v2.val[0]), q.val[1], v2.val[1]);
                    q = q8.load_quants(iy, i, 3);
                    accd[iy] = ggml_vdotq_s32(ggml_vdotq_s32(accd[iy], q.val[0], v2.val[2]), q.val[1], v2.val[3]);
                }
            }
        }
        int i = 2*(nb/2);
        if (i < nb) {
            deq.prepare_iq1bn_quants_nosub(x+i, v1);
            if constexpr (nrc_y == 1) {
                auto q = q8.load_quants(0, i/2, 0);
                for (int j = 0; j < 4; ++j) {
                    accd[0] = ggml_vdotq_s32(accd[0], q.val[j], v1.val[j]);
                }
            } else {
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto q = q8.load_quants(iy, i/2, 0);
                    accd[iy] = ggml_vdotq_s32(ggml_vdotq_s32(accd[iy], q.val[0], v1.val[0]), q.val[1], v1.val[1]);
                    q = q8.load_quants(iy, i/2, 1);
                    accd[iy] = ggml_vdotq_s32(ggml_vdotq_s32(accd[iy], q.val[0], v1.val[2]), q.val[1], v1.val[3]);
                }
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            if constexpr (is_iq1_tn) {
                info.store(ix, iy, -scale * vaddvq_f32(vfmsq_f32(q8.minus(iy), q8.scale(iy), vcvtq_f32_s32(accd[iy]))));
            } else {
                info.store(ix, iy, -vaddvq_f32(vfmsq_f32(q8.minus(iy), q8.scale(iy), vcvtq_f32_s32(accd[iy]))));
            }
        }

    }
}

template <int nrc_y>
static void mul_mat_iq2bn_q8_K64(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const int nb = n / QK_IQ1BN;

    Q8_K64<nrc_y> q8(info);

    int32x4_t   accd[nrc_y];

    const auto m1 = vdupq_n_u8(1);
    const auto mask2  = vdupq_n_s8(3);

    for (int ix = 0; ix < nrc_x; ++ix) {

        const block_iq2_bn * x = (const block_iq2_bn *)((const char *)vx + ix*bx);

        if constexpr (nrc_y == 1) {
            int8x16x4_t v1;
            int32x4_t acc[4] = {};
            for (int i = 0; i < nb/2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    auto q = q8.load_quants64(0, i, j);
                    auto q2bits = vld1q_u8(x[2*i+j].qs);
                    v1.val[0] = vandq_s8(q2bits, mask2);
                    v1.val[1] = vandq_s8(vshrq_n_u8(q2bits, 2), mask2);
                    v1.val[2] = vandq_s8(vshrq_n_u8(q2bits, 4), mask2);
                    v1.val[3] = vshrq_n_u8(q2bits, 6);
                    acc[0] = ggml_vdotq_s32(acc[0], q.val[0], v1.val[0]);
                    acc[1] = ggml_vdotq_s32(acc[1], q.val[1], v1.val[1]);
                    acc[2] = ggml_vdotq_s32(acc[2], q.val[2], v1.val[2]);
                    acc[3] = ggml_vdotq_s32(acc[3], q.val[3], v1.val[3]);
                }
            }
            accd[0] = vaddq_s32(vaddq_s32(acc[0], acc[1]), vaddq_s32(acc[2], acc[3]));
        } else {
            int8x16x4_t v1, v2;
            for (int iy = 0; iy < nrc_y; ++iy) accd[iy] = vdupq_n_s32(0);
            for (int i = 0; i < nb/2; ++i) {
                auto q2bits = vld1q_u8(x[2*i+0].qs);
                v1.val[0] = vandq_s8(q2bits, mask2);
                v1.val[1] = vandq_s8(vshrq_n_u8(q2bits, 2), mask2);
                v1.val[2] = vandq_s8(vshrq_n_u8(q2bits, 4), mask2);
                v1.val[3] = vshrq_n_u8(q2bits, 6);
                q2bits = vld1q_u8(x[2*i+1].qs);
                v2.val[0] = vandq_s8(q2bits, mask2);
                v2.val[1] = vandq_s8(vshrq_n_u8(q2bits, 2), mask2);
                v2.val[2] = vandq_s8(vshrq_n_u8(q2bits, 4), mask2);
                v2.val[3] = vshrq_n_u8(q2bits, 6);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto q = q8.load_quants(iy, i, 0);
                    accd[iy] = ggml_vdotq_s32(ggml_vdotq_s32(accd[iy], q.val[0], v1.val[0]), q.val[1], v1.val[1]);
                    q = q8.load_quants(iy, i, 1);
                    accd[iy] = ggml_vdotq_s32(ggml_vdotq_s32(accd[iy], q.val[0], v1.val[2]), q.val[1], v1.val[3]);
                    q = q8.load_quants(iy, i, 2);
                    accd[iy] = ggml_vdotq_s32(ggml_vdotq_s32(accd[iy], q.val[0], v2.val[0]), q.val[1], v2.val[1]);
                    q = q8.load_quants(iy, i, 3);
                    accd[iy] = ggml_vdotq_s32(ggml_vdotq_s32(accd[iy], q.val[0], v2.val[2]), q.val[1], v2.val[3]);
                }
            }
        }
        int i = 2*(nb/2);
        if (i < nb) {
            auto q2bits = vld1q_u8(x[i].qs);
            int8x16x4_t v1;
            v1.val[0] = vandq_s8(q2bits, mask2);
            v1.val[1] = vandq_s8(vshrq_n_u8(q2bits, 2), mask2);
            v1.val[2] = vandq_s8(vshrq_n_u8(q2bits, 4), mask2);
            v1.val[3] = vshrq_n_u8(q2bits, 6);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto q = q8.load_quants(iy, i/2, 0);
                accd[iy] = ggml_vdotq_s32(ggml_vdotq_s32(accd[iy], q.val[0], v1.val[0]), q.val[1], v1.val[1]);
                q = q8.load_quants(iy, i/2, 1);
                accd[iy] = ggml_vdotq_s32(ggml_vdotq_s32(accd[iy], q.val[0], v1.val[2]), q.val[1], v1.val[3]);
            }
        }

        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, -vaddvq_f32(vfmsq_f32(q8.minus(iy), q8.scale(iy), vcvtq_f32_s32(accd[iy]))));
        }
    }
}

template <typename Dequantizer> void MulMat::set_functions(MulMat& m) {
    if constexpr (std::is_same_v<Dequantizer, DequantizerQ40> || std::is_same_v<Dequantizer, DequantizerQ50> ||
                  std::is_same_v<Dequantizer, DequantizerQ80> || std::is_same_v<Dequantizer, DequantizerIQ4NL>) {
        m.funcs[0] = mul_mat_qX_0_q8_0<Dequantizer, 1>;
        m.funcs[1] = mul_mat_qX_0_q8_0<Dequantizer, 2>;
        m.funcs[2] = mul_mat_qX_0_q8_0<Dequantizer, 3>;
        m.funcs[3] = mul_mat_qX_0_q8_0<Dequantizer, 4>;
        m.funcs[4] = mul_mat_qX_0_q8_0<Dequantizer, 5>;
        m.funcs[5] = mul_mat_qX_0_q8_0<Dequantizer, 6>;
        m.funcs[6] = mul_mat_qX_0_q8_0<Dequantizer, 7>;
        m.funcs[7] = mul_mat_qX_0_q8_0<Dequantizer, 8>;
    }
    else if constexpr (std::is_same_v<Dequantizer, DequantizerQ41> || std::is_same_v<Dequantizer, DequantizerQ51>) {
        m.funcs[0] = mul_mat_qX_1_q8_1<Dequantizer, 1>;
        m.funcs[1] = mul_mat_qX_1_q8_1<Dequantizer, 2>;
        m.funcs[2] = mul_mat_qX_1_q8_1<Dequantizer, 3>;
        m.funcs[3] = mul_mat_qX_1_q8_1<Dequantizer, 4>;
        m.funcs[4] = mul_mat_qX_1_q8_1<Dequantizer, 5>;
        m.funcs[5] = mul_mat_qX_1_q8_1<Dequantizer, 6>;
        m.funcs[6] = mul_mat_qX_1_q8_1<Dequantizer, 7>;
        m.funcs[7] = mul_mat_qX_1_q8_1<Dequantizer, 8>;
    }
    else {
        m.funcs[0] = mul_mat_qX_K_q8_K_T<1, Dequantizer>;
        m.funcs[1] = mul_mat_qX_K_q8_K_T<2, Dequantizer>;
        m.funcs[2] = mul_mat_qX_K_q8_K_T<3, Dequantizer>;
        m.funcs[3] = mul_mat_qX_K_q8_K_T<4, Dequantizer>;
        m.funcs[4] = mul_mat_qX_K_q8_K_T<5, Dequantizer>;
        m.funcs[5] = mul_mat_qX_K_q8_K_T<6, Dequantizer>;
        m.funcs[6] = mul_mat_qX_K_q8_K_T<7, Dequantizer>;
        m.funcs[7] = mul_mat_qX_K_q8_K_T<8, Dequantizer>;
    }
}

bool MulMat::prepare(int typeA, int typeB, int ne00, MulMat& m, int /*Ny*/) {

    if (typeA == GGML_TYPE_F16 && typeB == GGML_TYPE_F16) {
        if (ne00%4) return false;
        for (auto& f : m.funcs) f = nullptr;
        m.funcs[0] = mul_mat_f16_f16_1;
        m.funcs[1] = mul_mat_f16_f16_T<2>;
        m.funcs[2] = mul_mat_f16_f16_T<3>;
        m.funcs[3] = mul_mat_f16_f16_T<4>;
        m.funcs[4] = mul_mat_f16_f16_T<5>;
        return true;
    }

    if (typeA == GGML_TYPE_BF16 && typeB == GGML_TYPE_F32) {
        if (ne00%4) return false;
        for (auto& f : m.funcs) f = nullptr;
        m.funcs[0] = mul_mat_Qx_Qy_T<QF32<1>, QBF16>;
        m.funcs[1] = mul_mat_Qx_Qy_T<QF32<2>, QBF16>;
        m.funcs[2] = mul_mat_Qx_Qy_T<QF32<3>, QBF16>;
        m.funcs[3] = mul_mat_Qx_Qy_T<QF32<4>, QBF16>;
        m.funcs[4] = mul_mat_Qx_Qy_T<QF32<5>, QBF16>;
        return true;
    }

    auto expected_Btype = GGML_TYPE_Q8_K;

    switch (typeA) {
        case GGML_TYPE_Q2_K:
            MulMat::set_functions<DequantizerQ2K>(m);
            break;
        case GGML_TYPE_IQ2_TN:
            //MulMat::set_functions<DequantizerIQ2TN>(m);
            m.funcs[0] = mul_mat_iq2tn_K_q8_K_1;
            m.funcs[1] = mul_mat_iq2tn_K_q8_K_T<2>;
            m.funcs[2] = mul_mat_iq2tn_K_q8_K_T<3>;
            m.funcs[3] = mul_mat_iq2tn_K_q8_K_T<4>;
            m.funcs[4] = mul_mat_iq2tn_K_q8_K_T<5>;
            m.funcs[5] = mul_mat_iq2tn_K_q8_K_T<6>;
            m.funcs[6] = mul_mat_iq2tn_K_q8_K_T<7>;
            m.funcs[7] = mul_mat_iq2tn_K_q8_K_T<8>;
            break;
        case GGML_TYPE_Q3_K:
            MulMat::set_functions<DequantizerQ3K>(m);
            break;
        case GGML_TYPE_Q4_K:
            MulMat::set_functions<DequantizerQ4K>(m);
            break;
        case GGML_TYPE_Q5_K:
            MulMat::set_functions<DequantizerQ5K>(m);
            break;
        case GGML_TYPE_Q6_K:
            MulMat::set_functions<DequantizerQ6K>(m);
            break;
        case GGML_TYPE_IQ4_XS:
            MulMat::set_functions<DequantizerIQ4XS>(m);
            break;
        case GGML_TYPE_IQ4_K:
            MulMat::set_functions<DequantizerIQ4K>(m);
            break;
        case GGML_TYPE_IQ5_K:
            MulMat::set_functions<DequantizerIQ5K>(m);
            break;
        case GGML_TYPE_IQ6_K:
            MulMat::set_functions<DequantizerIQ6K>(m);
            break;
        case GGML_TYPE_IQ2_K:
            MulMat::set_functions<DequantizerIQ2K>(m);
            break;
        case GGML_TYPE_IQ3_K:
            MulMat::set_functions<DequantizerIQ3K>(m);
            break;
        case GGML_TYPE_IQ2_XXS:
            MulMat::set_functions<DequantizerIQ2XXS>(m);
            break;
        case GGML_TYPE_IQ2_XS:
            MulMat::set_functions<DequantizerIQ2XS>(m);
            break;
        case GGML_TYPE_IQ2_S:
            MulMat::set_functions<DequantizerIQ2S>(m);
            break;
        case GGML_TYPE_IQ3_XXS:
            MulMat::set_functions<DequantizerIQ3XXS>(m);
            break;
        case GGML_TYPE_IQ3_S:
            MulMat::set_functions<DequantizerIQ3S>(m);
            break;
        case GGML_TYPE_IQ1_BN:
            m.funcs[0] = mul_mat_iq1bn_q8_K64<1, false>;
            m.funcs[1] = mul_mat_iq1bn_q8_K64<2, false>;
            m.funcs[2] = mul_mat_iq1bn_q8_K64<3, false>;
            m.funcs[3] = mul_mat_iq1bn_q8_K64<4, false>;
            m.funcs[4] = mul_mat_iq1bn_q8_K64<5, false>;
            m.funcs[5] = mul_mat_iq1bn_q8_K64<6, false>;
            m.funcs[6] = mul_mat_iq1bn_q8_K64<7, false>;
            m.funcs[7] = mul_mat_iq1bn_q8_K64<8, false>;
            expected_Btype = GGML_TYPE_Q8_K64;
            break;
        case GGML_TYPE_IQ1_TN:
            m.funcs[0] = mul_mat_iq1bn_q8_K64<1, true>;
            m.funcs[1] = mul_mat_iq1bn_q8_K64<2, true>;
            m.funcs[2] = mul_mat_iq1bn_q8_K64<3, true>;
            m.funcs[3] = mul_mat_iq1bn_q8_K64<4, true>;
            m.funcs[4] = mul_mat_iq1bn_q8_K64<5, true>;
            m.funcs[5] = mul_mat_iq1bn_q8_K64<6, true>;
            m.funcs[6] = mul_mat_iq1bn_q8_K64<7, true>;
            m.funcs[7] = mul_mat_iq1bn_q8_K64<8, true>;
            expected_Btype = GGML_TYPE_Q8_K64;
            break;
        case GGML_TYPE_IQ2_BN:
            m.funcs[0] = mul_mat_iq2bn_q8_K64<1>;
            m.funcs[1] = mul_mat_iq2bn_q8_K64<2>;
            m.funcs[2] = mul_mat_iq2bn_q8_K64<3>;
            m.funcs[3] = mul_mat_iq2bn_q8_K64<4>;
            m.funcs[4] = mul_mat_iq2bn_q8_K64<5>;
            m.funcs[5] = mul_mat_iq2bn_q8_K64<6>;
            m.funcs[6] = mul_mat_iq2bn_q8_K64<7>;
            m.funcs[7] = mul_mat_iq2bn_q8_K64<8>;
            expected_Btype = GGML_TYPE_Q8_K64;
            break;
        case GGML_TYPE_Q4_0:
            MulMat::set_functions<DequantizerQ40>(m);
            expected_Btype = GGML_TYPE_Q8_0;
            break;
        case GGML_TYPE_Q4_1:
            MulMat::set_functions<DequantizerQ41>(m);
            expected_Btype = GGML_TYPE_Q8_1;
            break;
        case GGML_TYPE_Q5_0:
            MulMat::set_functions<DequantizerQ50>(m);
            expected_Btype = GGML_TYPE_Q8_0;
            break;
        case GGML_TYPE_Q5_1:
            MulMat::set_functions<DequantizerQ51>(m);
            expected_Btype = GGML_TYPE_Q8_1;
            break;
        case GGML_TYPE_Q8_0:
            MulMat::set_functions<DequantizerQ80>(m);
            expected_Btype = GGML_TYPE_Q8_0;
            break;
        case GGML_TYPE_IQ4_NL:
            MulMat::set_functions<DequantizerIQ4NL>(m);
            expected_Btype = GGML_TYPE_Q8_0;
            break;
        default:
            return false;
    }

    return typeB == expected_Btype;
}

}

#endif // __aarch64__

namespace {

#if defined(__ARM_NEON) && defined(__aarch64__)
// copy-pasted from Justine Tunney's contribution to llama.cpp
// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline float32x4_t v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}
inline float16x8_t v_expf(float16x8_t x) {
    auto val1 = v_expf(vcvt_f32_f16(vget_low_f16(x)));
    auto val2 = v_expf(vcvt_f32_f16(vget_high_f16(x)));
    return vcombine_f16(vcvt_f16_f32(val1), vcvt_f16_f32(val2));
}
inline float32x4_t v_tanh(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t two_x = vmulq_f32(x, vdupq_n_f32(2.f));
    const float32x4_t exp_two_x = v_expf(two_x);
    const uint32x4_t mask = vcgtq_f32(x, vdupq_n_f32(10.f));
    const float32x4_t res = vdivq_f32(vsubq_f32(exp_two_x, one), vaddq_f32(exp_two_x, one));
    return vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(one), mask), vbicq_u32(vreinterpretq_u32_f32(res), mask)));
    //return vdivq_f32(vsubq_f32(exp_two_x, one), vaddq_f32(exp_two_x, one));
}
inline float32x4_t v_tanh(float16x8_t x) {
    auto val1 = v_tanh(vcvt_f32_f16(vget_low_f16(x)));
    auto val2 = v_tanh(vcvt_f32_f16(vget_high_f16(x)));
    return vcombine_f16(vcvt_f16_f32(val1), vcvt_f16_f32(val2));
}
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)

// copy-pasted from Justine Tunney's contribution to llama.cpp
// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline __m512 v_expf(__m512 x) {
  const __m512 r = _mm512_set1_ps(0x1.8p23f);
  const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
  const __m512 n = _mm512_sub_ps(z, r);
  const __m512 b =
      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                       _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
  const __mmask16 d =
      _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
  const __m512 u = _mm512_mul_ps(b, b);
  const __m512 j = _mm512_fmadd_ps(
      _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                      _mm512_set1_ps(0x1.573e2ep-5f)),
                      u,
                      _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                      _mm512_set1_ps(0x1.fffdb6p-2f))),
      u,
      _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
  const __m512 res = _mm512_scalef_ps(j, n);
  if (_mm512_kortestz(d, d))
    return res;
  const __m512 zero = _mm512_setzero_ps();
  const __m512 alt = _mm512_mask_blend_ps(
      _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
  return _mm512_mask_blend_ps(d, res, alt);
}
inline __m512 v_tanh(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 exp_two_x = v_expf(_mm512_mul_ps(x, _mm512_set1_ps(2.f)));
    const __mmask16 mask = _mm512_cmp_ps_mask(x, _mm512_set1_ps(10.f), _CMP_GT_OQ);
    const __m512 res = _mm512_div_ps(_mm512_sub_ps(exp_two_x, one), _mm512_add_ps(exp_two_x, one));
    return _mm512_mask_blend_ps(mask, res, one);
}
#endif

#if defined(__AVX2__) && defined(__FMA__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline __m256 v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}
inline __m256 v_tanh(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 exp_two_x = v_expf(_mm256_mul_ps(x, _mm256_set1_ps(2.f)));
    const __m256 res = _mm256_div_ps(_mm256_sub_ps(exp_two_x, one), _mm256_add_ps(exp_two_x, one));
    const __m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(10.f), _CMP_GT_OQ);
    return _mm256_or_ps(_mm256_and_ps(mask, one), _mm256_andnot_ps(mask, res));
}

#endif
} // namespace

namespace {

template <int k_step>
struct BaseHelper {
    BaseHelper(const char * data, int stride) : data(data), block(data), stride(stride) {}

    inline void set_block(int k1) { block = data + k1*k_step*stride; }
    inline void reset_block() { block = data; }
    inline void next_block() { block += k_step*stride; }
    inline const char * lblock(int l1) const { return block + l1*stride; }

    const char * data;
    const char * block;
    int stride;

};

struct F16 {
#ifdef HAVE_FANCY_SIMD
    using Data = __m512;
    constexpr static int block_size = 16;
    constexpr static int num_registers = 32;
    constexpr static int q_step = 8;
    static inline Data zero() { return _mm512_setzero_ps(); }
    static inline Data load(const char * ptr, int i) { return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)ptr + i)); }
    static inline Data set1(float val) { return _mm512_set1_ps(val); }
    static inline Data mul(Data v1, Data v2) { return _mm512_mul_ps(v1, v2); }
    static inline Data sub(Data v1, Data v2) { return _mm512_sub_ps(v1, v2); }
    static inline Data load(const float * ptr) { return _mm512_loadu_ps(ptr); }
    static inline void store(float * ptr, Data data) { _mm512_storeu_ps(ptr, data); }
    static inline float reduce_max(Data data) { return _mm512_reduce_max_ps(data); }
    static inline float reduce_add(Data data) { return _mm512_reduce_add_ps(data); }
    static inline Data fmadd(Data prev, Data v1, Data v2) { return _mm512_fmadd_ps(v1, v2, prev); }
    template <int k_step> static inline float reduce_max(const Data * data) {
        return reduce_T<k_step, _mm512_max_ps, _mm512_reduce_max_ps>(data);
    }
    template <int k_step> static inline float reduce_add(const Data * data) {
        return reduce_T<k_step, _mm512_add_ps, _mm512_reduce_add_ps>(data);
    }
#elif defined __AVX2__
    using Data = __m256;
    constexpr static int block_size = 8;
    constexpr static int num_registers = 16;
    constexpr static int q_step = 8;
    static inline Data zero() { return _mm256_setzero_ps(); }
    static inline Data load(const char * ptr, int i) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)ptr + i)); }
    static inline Data set1(float val) { return _mm256_set1_ps(val); }
    static inline Data mul(Data v1, Data v2) { return _mm256_mul_ps(v1, v2); }
    static inline Data load(const float * ptr) { return _mm256_loadu_ps(ptr); }
    static inline Data sub(Data v1, Data v2) { return _mm256_sub_ps(v1, v2); }
    static inline void store(float * ptr, Data data) { _mm256_storeu_ps(ptr, data); }
    static inline Data fmadd(Data prev, Data v1, Data v2) { return _mm256_fmadd_ps(v1, v2, prev); }
    static inline float reduce_max(Data data) { return hmax_float_8(data); }
    static inline float reduce_add(Data data) { return hsum_float_8(data); }
    template <int k_step> static inline float reduce_max(const Data * data) {
        return reduce_T<k_step, _mm256_max_ps, &F16::reduce_max>(data);
    }
    template <int k_step> static inline float reduce_add(const Data * data) {
        return reduce_T<k_step, _mm256_add_ps, &F16::reduce_add>(data);
    }
#else
    using Data = float16x8_t;
    constexpr static int block_size = 8;
    constexpr static int num_registers = 32;
    constexpr static int q_step = 8;
    static inline Data zero() { return vdupq_n_f16(0); }
    static inline Data load(const char * ptr, int i) { return vld1q_f16((const float16_t *)ptr + block_size*i); }
    static inline Data load(const float16_t * ptr, int i) { return vld1q_f16(ptr + block_size*i); }
    static inline Data load(const float16_t * ptr) { return vld1q_f16(ptr); }
    static inline Data load(const float * ptr) {
        auto val1 = vld1q_f32(ptr);
        auto val2 = vld1q_f32(ptr+4);
        return vcombine_f16(vcvt_f16_f32(val1), vcvt_f16_f32(val2));
    }
    static inline Data set1(float val) { return vdupq_n_f16(val); }
    static inline Data mul(Data v1, Data v2) { return vmulq_f16(v1, v2); }
    static inline Data sub(Data v1, Data v2) { return vsubq_f16(v1, v2); }
    static inline void store(float * ptr, Data data) {
        vst1q_f32(ptr+0, vcvt_f32_f16(vget_low_f16(data)));
        vst1q_f32(ptr+4, vcvt_f32_f16(vget_high_f16(data)));
    }
    static inline void store(float16_t * ptr, Data data) { vst1q_f16(ptr, data); }
    static inline void store(float * ptr, float32x4_t data) { vst1q_f32(ptr, data); }
    static inline Data fmadd(Data prev, Data v1, Data v2) { return vfmaq_f16(prev, v1, v2); }
    static inline float reduce_max(Data data) { return vmaxvq_f16(data); }
    static inline float reduce_add(Data data) {
        auto sum = vadd_f16(vget_low_f16(data), vget_high_f16(data));
        return vaddvq_f32(vcvt_f32_f16(sum));
    }
    template <int k_step> static inline float reduce_max(const Data * data) {
        return reduce_T<k_step, vmaxq_f16, &F16::reduce_max>(data);
    }
    template <int k_step> static inline float reduce_add(const Data * data) {
        return reduce_T<k_step, vaddq_f16, &F16::reduce_add>(data);
    }
#endif
    template <int k_step, Data (*Op_combine)(Data, Data), float (*Op)(Data)>
    static float reduce_T(const Data * data) {
        float result;
        if constexpr (k_step/block_size == 1) {
            result = Op(data[0]);
        }
        else if constexpr (k_step/block_size == 2) {
            result = Op(Op_combine(data[0], data[1]));
        }
        else {
            auto vmax = Op_combine(data[0], data[1]);
            for (int l = 2; l < k_step/block_size; ++l) vmax = Op_combine(vmax, data[l]);
            result = Op(vmax);
        }
        return result;
    }
};

template <int D, int step>
struct HelperF16 final : public BaseHelper<step> {
    using Base = BaseHelper<step>;
    HelperF16(const char * data, int stride) : Base(data, stride) {}

    inline void load(int l1, F16::Data * vk) const {
        auto dr = Base::lblock(l1);
        for (int i = 0; i < D/F16::block_size; ++i) vk[i] = F16::load(dr, i);
    }

    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        //auto dr = (const ggml_half *)Base::lblock(l1);
        auto dr = Base::lblock(l1);
        v1 = F16::load(dr, i + 0);
        v2 = F16::load(dr, i + 1);
    }

    inline void load_2(int l1, F16::Data* vk) const {
        load(l1+0, vk+0);
        load(l1+1, vk+D/16);
    }
};

void quantize_row_q8_0(const float * x, block_q8_0 * y, int k) {
    const int nb = k / QK8_0;
    const int nb4 = 4*(nb/4);

#if defined(__aarch64__)
    block_q8_0_x4 * y4 = (block_q8_0_x4 *)y;
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            if (i < nb4) {
                y4[i4].qs[32*ir + 4*j + 0] = vgetq_lane_s32(vi, 0);
                y4[i4].qs[32*ir + 4*j + 1] = vgetq_lane_s32(vi, 1);
                y4[i4].qs[32*ir + 4*j + 2] = vgetq_lane_s32(vi, 2);
                y4[i4].qs[32*ir + 4*j + 3] = vgetq_lane_s32(vi, 3);
            } else {
                y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
                y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
                y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
                y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
            }
        }
    }
#else
    block_q8_0_x4 * y4 = (block_q8_0_x4 *)y;
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        const float d = maxScalar / 127.f;
        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *)y4[i4].qs + ir, i0);
        } else {
            _mm256_storeu_si256((__m256i *)y[i].qs, i0);
        }
    }
#endif
}

void quantize_row_q8_1(const float * x, block_q8_1 * y, int k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    const int nb4 = 4*(nb/4);
    block_q8_1_x4 * y4 = (block_q8_1_x4 *)y;
#if defined(__aarch64__)
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }

        int32x4_t accv = vdupq_n_s32(0);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            if (i < nb4) {
                y4[i4].qs[QK8_1*ir + 4*j + 0] = vgetq_lane_s32(vi, 0);
                y4[i4].qs[QK8_1*ir + 4*j + 1] = vgetq_lane_s32(vi, 1);
                y4[i4].qs[QK8_1*ir + 4*j + 2] = vgetq_lane_s32(vi, 2);
                y4[i4].qs[QK8_1*ir + 4*j + 3] = vgetq_lane_s32(vi, 3);
            } else {
                y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
                y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
                y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
                y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
            }

            accv = vaddq_s32(accv, vi);
        }

        if (i < nb4) {
            y4[i4].d[ir+4] = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
        } else {
            y[i].s = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float max_scalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = max_scalar / 127.f;
        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }
        const float id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Compute the sum of the quants and set y[i].s
        if (i < nb4) {
            y4[i4].d[ir+4] = GGML_FP32_TO_FP16(d * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3))));
        } else {
            y[i].s = GGML_FP32_TO_FP16(d * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3))));
        }

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *)y4[i4].qs + ir, i0);
        } else {
            _mm256_storeu_si256((__m256i *)y[i].qs, i0);
        }
    }
#endif
}

template <int D, int step>
struct HelperQ80 final : public BaseHelper<step> {
    using Base = BaseHelper<step>;
    using block_q8 = block_q8_0;
    HelperQ80(const char * data, int stride) : Base(data, stride) {}

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        int j = F16::block_size*i;
        auto dl = (const block_q8_0_x4 *)Base::lblock(l1) + j/(4*QK8_0);
        int ii = (j/QK8_0)%4;
#ifdef __aarch64__
        const float16_t * d = (const float16_t *)dl->d;
        auto vd = F16::set1(d[ii]);
        auto qs = vld1_s8_x2(dl->qs + 32*ii + j%32);
        v1 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(qs.val[0])));
        v2 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(qs.val[1])));
#else
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d[ii]));
#ifdef HAVE_FANCY_SIMD
        v1 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)dl->qs+2*ii+0))));
        v2 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)dl->qs+2*ii+1))));
#else
        v1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(dl->qs+32*ii+j%32)))));
        v2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(dl->qs+32*ii+j%32+8)))));
#endif
#endif
    }

    static inline void convert(int nq, int stride_q, const float * q, block_q8_0 * y) {
        GGML_ASSERT(nq <= step);
        for (int i = 0; i < nq; ++i) {
            quantize_row_q8_0(q, y, D);
            q += stride_q;
            y += D/QK8_0;
        }
    }

    static inline void convert(int nq, int stride_q, const float * q, block_q8_1 * y) {
        GGML_ASSERT(nq <= step);
        for (int i = 0; i < nq; ++i) {
            quantize_row_q8_1(q, y, D);
            q += stride_q;
            y += D/QK8_1;
        }
    }
};

template <int D, int step>
struct HelperQ40 final : public BaseHelper<step> {
    using Base = BaseHelper<step>;
    using block_q8 = block_q8_0;
    HelperQ40(const char * data, int stride) : Base(data, stride) {}

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        int j = F16::block_size*i;
        auto dl = (const block_q4_0 *)Base::lblock(l1) + j/QK4_0;
#ifdef __aarch64__
        auto vd = F16::set1(*(const float16_t *)&dl->d);
        auto q  = vld1q_u8(dl->qs);
        q = j%QK4_0 ? vshrq_n_u8(q, 4) : vandq_u8(q, mask);
        q = vaddq_s8(q, m8);
        v1 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(vget_low_s8(q))));
        v2 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(vget_high_s8(q))));
#else
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d));
        auto q  = _mm_loadu_si128((const __m128i *)dl->qs);
#ifdef HAVE_FANCY_SIMD
        auto ql = _mm_add_epi8(_mm_and_si128(q, mask), m8);
        auto qh = _mm_add_epi8(_mm_and_si128(_mm_srli_epi16(q, 4), mask), m8);
        v1 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ql)));
        v2 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qh)));
#else
        if (j%QK4_0) q = _mm_srli_epi16(q, 4);
        auto q16 = _mm256_cvtepi8_epi16(_mm_add_epi8(_mm_and_si128(q, mask), m8));
        v1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16))));
        v2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1))));
#endif
#endif
    }

#ifdef __AVX2__
    const __m128i mask = _mm_set1_epi8(0xf);
    const __m128i m8   = _mm_set1_epi8(-8);
#else
    const uint8x16_t mask = vdupq_n_u8(0xf);
    const  int8x16_t m8   = vdupq_n_s8(-8);
#endif
};

template <int D, int step>
struct HelperQ41 final : public BaseHelper<step> {
    using Base = BaseHelper<step>;
    using block_q8 = block_q8_1;
    HelperQ41(const char * data, int stride) : Base(data, stride) {}

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        int j = F16::block_size*i;
        auto dl = (const block_q4_1 *)Base::lblock(l1) + j/QK4_1;
#ifdef __aarch64__
        auto vd = F16::set1(*(const float16_t *)&dl->d);
        auto vm = F16::set1(*(const float16_t *)&dl->m);
        auto q  = vld1q_u8(dl->qs);
        q = (j%QK4_1) ? vshrq_n_u8(q, 4) : vandq_u8(q, mask);
        v1 = vfmaq_f16(vm, vd, vcvtq_f16_u16(vmovl_u8(vget_low_u8(q))));
        v2 = vfmaq_f16(vm, vd, vcvtq_f16_u16(vmovl_u8(vget_high_u8(q))));
#else
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d));
        auto vm = F16::set1(GGML_FP16_TO_FP32(dl->m));
        auto q  = _mm_loadu_si128((const __m128i *)dl->qs);
#ifdef HAVE_FANCY_SIMD
        auto ql = _mm_and_si128(q, mask);
        auto qh = _mm_and_si128(_mm_srli_epi16(q, 4), mask);
        v1 = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ql)), vm);
        v2 = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qh)), vm);
#else
        if (j%QK4_1) q = _mm_srli_epi16(q, 4);
        auto q16 = _mm256_cvtepi8_epi16(_mm_and_si128(q, mask));
        v1 = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16))), vm);
        v2 = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1))), vm);
#endif
#endif
    }

#ifdef __aarch64__
    const uint8x16_t mask = vdupq_n_u8(0xf);
#else
    const __m128i mask = _mm_set1_epi8(0xf);
#endif
};

template <int q_step, int k_step>
struct FlashMS {
// Something goes wrong when storing and manipulating K*Q as fp16.
// It works for some models (e.g., Gemma-2), but not for others (e.g., LLaMA-3.1-8B).
// As I wasn't able to find where we lose precision, let's comment this out
// for now and do the K*Q part in fp32.
//#ifdef __aarch64__
//    using cache_t = float16_t;
//#else
//    using cache_t = float;
//#endif
    using cache_t = float;

    FlashMS(float scale, float softcap) : vscale(F16::set1(scale)), softcap(softcap), h_inf(GGML_FP32_TO_FP16(-INFINITY)) {}

    inline void init_qstep() {
        for (int j = 0; j < q_step; ++j) {
            S[j] = 0; M[j] = -INFINITY;
        }
    }

    inline void update_M(int j, float smax) {
        if (smax == -INFINITY) {
            std::memset(cache + k_step*j, 0, k_step*sizeof(float));
            need_scaling[j] = M[j] == -INFINITY ? 2 : 0;
            return;
        }
        need_scaling[j] = 0;
        if (smax > M[j]) {
            if (M[j] > -INFINITY) {
                float m = expf(M[j] - smax);
                vms[j] = F16::set1(m);
                need_scaling[j] = 1;
                S[j] *= m;
            } else {
                need_scaling[j] = 2;
                S[j] = 0;
            }
            M[j] = smax;
        }
    }

#ifdef __aarch64__
    inline void update_S(int j, float32x4_t * vk) {
        auto vm = vdupq_n_f32(M[j]);
        auto vsum = vdupq_n_f32(0);
        for (int l = 0; l < k_step/4; ++l) {
            vk[l] = v_expf(vsubq_f32(vk[l], vm));
            vsum = vaddq_f32(vsum, vk[l]);
            F16::store(cache + k_step*j + 4*l, vk[l]);
        }
        S[j] += vaddvq_f32(vsum);
    }
#else
    inline void update_S(int j, F16::Data * vk) {
        auto vm = F16::set1(M[j]);
        for (int l = 0; l < k_step/F16::block_size; ++l) {
            vk[l] = v_expf(F16::sub(vk[l], vm));
            F16::store(cache + k_step*j + F16::block_size*l, vk[l]);
        }
        S[j] += F16::reduce_add<k_step>(vk);
    }
#endif

#ifdef __aarch64__
    inline float load_and_scale(int j, float32x4_t * vk) {
        float32x4_t vmax = vdupq_n_f32(-INFINITY);
        // Something goes wrong when storing and manipulating K*Q as fp16.
        // It works for some models (e.g., Gemma-2), but not for others (e.g., LLaMA-3.1-8B).
        // As I wasn't able to find where we lose precision, let's comment this out
        // for now and do the K*Q part in fp32.
        //if (softcap <= 0.0f) {
        //    for (int l = 0; l < k_step/F16::block_size; ++l) {
        //        auto val = F16::mul(vscale, F16::load(cache + k_step*j + F16::block_size*l));
        //        vk[2*l+0] = vcvt_f32_f16(vget_low_f16(val));
        //        vk[2*l+1] = vcvt_f32_f16(vget_high_f16(val));
        //        vmax = vmaxq_f32(vmax, vmaxq_f32(vk[2*l+0], vk[2*l+1]));
        //    }
        //} else {
        //    auto v_softcap = vdupq_n_f32(softcap);
        //    for (int l = 0; l < k_step/F16::block_size; ++l) {
        //        auto val = F16::mul(vscale, F16::load(cache + k_step*j + F16::block_size*l));
        //        vk[2*l+0] = vcvt_f32_f16(vget_low_f16(val));
        //        vk[2*l+1] = vcvt_f32_f16(vget_high_f16(val));
        //        vk[2*l+0] = vmulq_f32(v_softcap, v_tanh(vk[2*l+0]));
        //        vk[2*l+1] = vmulq_f32(v_softcap, v_tanh(vk[2*l+1]));
        //        vmax = vmaxq_f32(vmax, vmaxq_f32(vk[2*l+0], vk[2*l+1]));
        //    }
        //}
        auto vscale32 = vcvt_f32_f16(vget_low_f16(vscale));
        if (softcap <= 0.0f) {
            for (int l = 0; l < k_step/4; ++l) {
                vk[l] = vmulq_f32(vscale32, vld1q_f32(cache + k_step*j + 4*l));
                vmax = vmaxq_f32(vmax, vk[l]);
            }
        } else {
            auto v_softcap = vdupq_n_f32(softcap);
            for (int l = 0; l < k_step/4; ++l) {
                vk[l] = vmulq_f32(vscale32, vld1q_f32(cache + k_step*j + 4*l));
                vk[l] = vmulq_f32(v_softcap, v_tanh(vk[l]));
                vmax = vmaxq_f32(vmax, vk[l]);
            }
        }
        return vmaxvq_f32(vmax);
    }
    inline float load_apply_mask_and_scale(int j, float32x4_t * vk, const char * mask) {
        auto vzero = vdupq_n_f32(0);
        auto vinf  = vdupq_n_f32(-INFINITY);
        for (int l = 0; l < k_step/8; ++l) {
            auto vm = vceqq_f16(vzero, vld1q_f16((const float16_t *)mask + 8*l));
            auto vm1 = vzip1q_u16(vm, vm);
            auto vm2 = vzip2q_u16(vm, vm);
            auto kq  = vld1q_f32_x2(cache + k_step*j + 8*l);
            vk[2*l+0] = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(kq.val[0]), vm1),
                                                        vbicq_u32(vinf, vm1)));
            vk[2*l+1] = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(kq.val[1]), vm2),
                                                        vbicq_u32(vinf, vm2)));
        }
        float32x4_t vmax = vdupq_n_f32(-INFINITY);
        auto vscale32 = vcvt_f32_f16(vget_low_f16(vscale));
        if (softcap <= 0.0f) {
            for (int l = 0; l < k_step/4; ++l) {
                vk[l] = vmulq_f32(vscale32, vk[l]);
                vmax = vmaxq_f32(vmax, vk[l]);
            }
        } else {
            auto v_softcap = vdupq_n_f32(softcap);
            for (int l = 0; l < k_step/4; ++l) {
                vk[l] = vmulq_f32(vscale32, vk[l]);
                vk[l] = vmulq_f32(v_softcap, v_tanh(vk[l]));
                vmax = vmaxq_f32(vmax, vk[l]);
            }
        }
        return vmaxvq_f32(vmax);
    }
#else
    inline float load_and_scale(int j, F16::Data * vk) {
        if (softcap <= 0.0f) {
            for (int l = 0; l < k_step/F16::block_size; ++l) vk[l] = F16::mul(vscale, F16::load(cache + k_step*j + F16::block_size*l));
        } else {
            auto v_softcap = F16::set1(softcap);
            for (int l = 0; l < k_step/F16::block_size; ++l) {
                auto val = F16::load(cache + k_step*j + F16::block_size*l);
                vk[l] = F16::mul(v_softcap, v_tanh(F16::mul(vscale, val)));
            }
        }
        return F16::reduce_max<k_step>(vk);
    }
    inline float load_apply_mask_and_scale(int j, F16::Data * vk, const char * mask) {
#ifdef HAVE_FANCY_SIMD
        auto vzero = _mm256_set1_epi16(0);
        auto vinf  = _mm512_set1_ps(-INFINITY);
        if (softcap <= 0) {
            for (int l = 0; l < k_step/F16::block_size; ++l) {
                auto m16 = _mm256_cmpeq_epi16_mask(_mm256_loadu_si256((const __m256i *)mask + l), vzero);
                vk[l] = _mm512_mask_mul_ps(vinf, m16, vscale, F16::load(cache + k_step*j + F16::block_size*l));
            }
        } else {
            auto v_softcap = F16::set1(softcap);
            for (int l = 0; l < k_step/F16::block_size; ++l) {
                auto m16 = _mm256_cmpeq_epi16_mask(_mm256_loadu_si256((const __m256i *)mask + l), vzero);
                vk[l] = _mm512_mask_mul_ps(vinf, m16, v_softcap, v_tanh(F16::mul(vscale, F16::load(cache + k_step*j + F16::block_size*l))));
            }
        }
#else
        auto vzero = _mm_set1_epi16(0);
        auto vinf  = F16::set1(-INFINITY);
        for (int l = 0; l < k_step/F16::block_size; ++l) {
            auto m128 = _mm_loadu_si128((const __m128i *)mask + l);
            m128 = _mm_cmpeq_epi16(m128, vzero);
            auto m256 = _mm256_cvtepi16_epi32(m128);
            auto mf = _mm256_castsi256_ps(_mm256_or_si256(m256, _mm256_slli_epi32(m256, 16)));
            auto val = _mm256_loadu_ps(cache + k_step*j + F16::block_size*l);
            vk[l] = _mm256_or_ps(_mm256_and_ps(mf, val), _mm256_andnot_ps(mf, vinf));
        }
        if (softcap <= 0) {
            for (int l = 0; l < k_step/F16::block_size; ++l) vk[l] = F16::mul(vscale, vk[l]);
        } else {
            auto v_softcap = F16::set1(softcap);
            for (int l = 0; l < k_step/F16::block_size; ++l) vk[l] = F16::mul(v_softcap, v_tanh(F16::mul(vscale, vk[l])));
        }
#endif
        return F16::reduce_max<k_step>(vk);
    }
#endif

#ifdef __aarch64__
    inline void update_M_S(int j, float32x4_t * vk) {
        float smax = load_and_scale(j, vk);
        update_M(j, smax);
        update_S(j, vk);
    }
    inline void update_M_S(int j, float32x4_t * vk, const char * mask) {
        float smax = load_apply_mask_and_scale(j, vk, mask);
        update_M(j, smax);
        update_S(j, vk);
    }
#else
    inline void update_M_S(int j, F16::Data * vk) {
        float smax = load_and_scale(j, vk);
        update_M(j, smax);
        update_S(j, vk);
    }
    inline void update_M_S(int j, F16::Data * vk, const char * mask) {
        float smax = load_apply_mask_and_scale(j, vk, mask);
        update_M(j, smax);
        update_S(j, vk);
    }
#endif

    cache_t cache[q_step*k_step];
    float S[q_step], M[q_step];
    int need_scaling[q_step];
    F16::Data vms[q_step];
    const F16::Data vscale;
    const float  softcap;
    const ggml_half h_inf;

};

template <int D, int q_step, int k_step>
struct FlashQKV {

#ifdef __aarch64__
    using qkv_cache_t = float16_t;
#else
    using qkv_cache_t = float;
#endif

    // This fails for head sizes of 80 and 112 as D/16 is odd, so we cannot do steps of 2
    // Hence, for now, we will not handle head sizes of 80 and 112
    template <typename VHelper>
    inline void accumulate_qkv(const VHelper& vh, const FlashMS<q_step, k_step>& fms) {
        F16::Data vk[2*q_step];
        for (int i = 0; i < D/F16::block_size; i += 2) {
            for (int j = 0; j < q_step; ++j) {
                if (fms.need_scaling[j] == 2) {
                    vk[2*j+0] = vk[2*j+1] = F16::zero();
                } else {
                    auto R = qkv_cache + D*j;
                    vk[2*j+0] = F16::load(R + F16::block_size*i);
                    vk[2*j+1] = F16::load(R + F16::block_size*(i + 1));
                    if (fms.need_scaling[j] == 1) {
                        vk[2*j+0] = F16::mul(vk[2*j+0], fms.vms[j]);
                        vk[2*j+1] = F16::mul(vk[2*j+1], fms.vms[j]);
                    }
                }
            }
            F16::Data v1, v2;
            for (int l1 = 0; l1 < k_step; ++l1) {
                vh.load(l1, i, v1, v2);
                for (int j = 0; j < q_step; ++j) {
                    auto vs = F16::set1(fms.cache[k_step*j + l1]);
                    vk[2*j+0] = F16::fmadd(vk[2*j+0], v1, vs);
                    vk[2*j+1] = F16::fmadd(vk[2*j+1], v2, vs);
                }
            }
            for (int j = 0; j < q_step; ++j) {
                auto R = qkv_cache + D*j;
                F16::store(R + F16::block_size*(i + 0), vk[2*j+0]);
                F16::store(R + F16::block_size*(i + 1), vk[2*j+1]);
            }
        }
    }

    template <typename VHelper, int Nq = q_step, class = std::enable_if<Nq >= 2>>
    inline void accumulate_qkv(int nq1, const VHelper& vh, const FlashMS<q_step, k_step>& fms) {
        F16::Data vk[2*q_step];
        for (int i = 0; i < D/F16::block_size; i += 2) {
            for (int j = 0; j < nq1; ++j) {
                if (fms.need_scaling[j] == 2) {
                    vk[2*j+0] = vk[2*j+1] = F16::zero();
                } else {
                    auto R = qkv_cache + D*j;
                    vk[2*j+0] = F16::load(R + F16::block_size*i);
                    vk[2*j+1] = F16::load(R + F16::block_size*(i + 1));
                    if (fms.need_scaling[j] == 1) {
                        vk[2*j+0] = F16::mul(vk[2*j+0], fms.vms[j]);
                        vk[2*j+1] = F16::mul(vk[2*j+1], fms.vms[j]);
                    }
                }
            }
            F16::Data v1, v2;
            for (int l1 = 0; l1 < k_step; ++l1) {
                vh.load(l1, i, v1, v2);
                for (int j = 0; j < nq1; ++j) {
                    auto vs = F16::set1(fms.cache[k_step*j + l1]);
                    vk[2*j+0] = F16::fmadd(vk[2*j+0], v1, vs);
                    vk[2*j+1] = F16::fmadd(vk[2*j+1], v2, vs);
                }
            }
            for (int j = 0; j < nq1; ++j) {
                auto R = qkv_cache + D*j;
                F16::store(R + F16::block_size*(i + 0), vk[2*j+0]);
                F16::store(R + F16::block_size*(i + 1), vk[2*j+1]);
            }
        }
    }

    inline void normalize_and_store(const FlashMS<q_step, k_step>& fms, int j, const qkv_cache_t * R, float * qkv) const {
        GGML_ASSERT(fms.S[j] > 0);
        auto norm = F16::set1(1/fms.S[j]);
        for (int i = 0; i < D/F16::block_size; ++i) {
            auto r = F16::load(R + F16::block_size*i);
            F16::store(qkv + F16::block_size*i, F16::mul(norm, r));
        }
    }

    inline void normalize_and_store(const FlashMS<q_step, k_step>& fms, int nq1, int stride_qkv, float * qkv) const {
        auto R = qkv_cache;
        for (int j = 0; j < nq1; ++j) {
            normalize_and_store(fms, j, R, qkv);
            qkv += stride_qkv;
            R   += D;
        }
    }

    inline void normalize_and_store(const FlashMS<q_step, k_step>& fms, int stride_qkv, float * qkv) const {
        auto R = qkv_cache;
        for (int j = 0; j < q_step; ++j) {
            normalize_and_store(fms, j, R, qkv);
            qkv += stride_qkv;
            R   += D;
        }
    }

    qkv_cache_t qkv_cache[D*q_step];
};

template <int D, int q_step, int k_step>
struct FlashQKfp32 {
    static_assert(D%F16::block_size == 0 && D <= 256);
    static_assert(k_step%F16::block_size == 0);
    static_assert(q_step <= 4 || q_step%4 == 0);

#ifdef __aarch64__
    constexpr static bool is_small_head = false;
#else
    constexpr static bool is_small_head = D <= (F16::num_registers/2)*F16::block_size;
#endif

    template <bool small = is_small_head, class = std::enable_if<small>, typename q_float>
    static inline void mult_mask_kq_one(int l1, int m1, int stride_q, int stride_m, const q_float * q, const char * mask,
            F16::Data * qv, F16::Data * vk, FlashMS<q_step, k_step>& fms) {
        // q index is q_step*i1 + m1
        // k index is k_step*k1 + l1
        const ggml_half * mp = (const ggml_half *)(mask + stride_m*m1);
        fms.cache[k_step*m1 + l1 + 0] = fms.cache[k_step*m1 + l1 + 1] = -INFINITY;
        if (mp[l1+0] == fms.h_inf && mp[l1+1] == fms.h_inf) {
            return;
        }
        auto qr = q + m1*stride_q;
        for (int i = 0; i < D/F16::block_size; ++i) qv[i] = F16::load(qr + F16::block_size*i);
        if (mp[l1+0] != fms.h_inf) {
            auto vsum = F16::zero();
            for (int i = 0; i < D/F16::block_size; ++i) vsum = F16::fmadd(vsum, vk[i], qv[i]);
            fms.cache[k_step*m1 + l1 + 0] = F16::reduce_add(vsum);
        }
        if (mp[l1+1] != fms.h_inf) {
            auto vsum = F16::zero();
            for (int i = 0; i < D/F16::block_size; ++i) vsum = F16::fmadd(vsum, vk[i+D/16], qv[i]);
            fms.cache[k_step*m1 + l1 + 1] = F16::reduce_add(vsum);
        }
    }

    template <bool small = is_small_head, class = std::enable_if<!small>, typename q_float>
    static inline void mult_mask_kq_one(int l1, int m1, int stride_q, int stride_m, const q_float * q, const char * mask,
            F16::Data * vk, FlashMS<q_step, k_step>& fms) {
        // q index is q_step*i1 + m1
        // k index is k_step*k1 + l1
        const ggml_half * mp = (const ggml_half *)(mask + stride_m*m1);
        if (mp[l1] == fms.h_inf) {
            fms.cache[k_step*m1 + l1] = -INFINITY;
            return;
        }
        auto qr = q + m1*stride_q;
        auto vsum = F16::zero();
        for (int i = 0; i < D/F16::block_size; ++i) {
            vsum = F16::fmadd(vsum, vk[i], F16::load(qr + F16::block_size*i));
        }
        fms.cache[k_step*m1 + l1] = F16::reduce_add(vsum);
    }

    template <typename KHelper, bool small = is_small_head, class = std::enable_if<small>, typename q_float>
    static inline void mult_mask_kq(const KHelper& kh, int stride_q, int stride_m, const q_float * q, const char * mask,
            FlashMS<q_step, k_step>& fms) {
        F16::Data qv[D/F16::block_size];
        F16::Data vk[D/(F16::block_size/2)];
        for (int l1 = 0; l1 < k_step; l1 += 2) {
            kh.load_2(l1, vk);
            for (int m1 = 0; m1 < q_step; ++m1) {
                mult_mask_kq_one(l1, m1, stride_q, stride_m, q, mask, qv, vk, fms);
            }
        }
    }

    template <typename KHelper, bool small = is_small_head, class = std::enable_if<!small>, typename q_float>
    static inline void mult_mask_kq_l(const KHelper& kh, int stride_q, int stride_m,
            const q_float * q, const char * mask, FlashMS<q_step, k_step>& fms) {
        F16::Data vk[D/F16::block_size];
        for (int l1 = 0; l1 < k_step; ++l1) {
            kh.load(l1, vk);
            for (int m1 = 0; m1 < q_step; ++m1) {
                mult_mask_kq_one(l1, m1, stride_q, stride_m, q, mask, vk, fms);
            }
        }
    }

    template <typename KHelper, bool small = is_small_head, class = std::enable_if<small>, typename q_float>
    static inline void mult_mask_kq(int nq, const KHelper& kh, int stride_q, int stride_m, const q_float * q, const char * mask,
            FlashMS<q_step, k_step>& fms) {
        F16::Data qv[D/F16::block_size];
        F16::Data vk[D/(F16::block_size/2)];
        for (int l1 = 0; l1 < k_step; l1 += 2) {
            kh.load_2(l1, vk);
            for (int m1 = 0; m1 < nq; ++m1) {
                mult_mask_kq_one(l1, m1, stride_q, stride_m, q, mask, qv, vk, fms);
            }
        }
    }

    template <typename KHelper, bool small = is_small_head, class = std::enable_if<!small>, typename q_float>
    static inline void mult_mask_kq_l(int nq, const KHelper& kh, int stride_q, int stride_m,
            const q_float * q, const char * mask, FlashMS<q_step, k_step>& fms) {
        F16::Data vk[D/F16::block_size];
        for (int l1 = 0; l1 < k_step; ++l1) {
            kh.load(l1, vk);
            for (int m1 = 0; m1 < nq; ++m1) {
                mult_mask_kq_one(l1, m1, stride_q, stride_m, q, mask, vk, fms);
            }
        }
    }

    template <typename KHelper, typename q_float>
    static inline void multiply_mask_kq(const KHelper& kh, int stride_q, int stride_m, const q_float * q, const char * mask,
            FlashMS<q_step, k_step>& fms) {
        if constexpr (is_small_head) {
            mult_mask_kq(kh, stride_q, stride_m, q, mask, fms);
        }
        else {
            mult_mask_kq_l(kh, stride_q, stride_m, q, mask, fms);
        }
#ifdef __aarch64__
        float32x4_t vk[k_step/4];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk);
        }
#else
        F16::Data vk[k_step/F16::block_size];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk);
        }
#endif
    }

    template <typename KHelper, typename q_float>
    static inline void multiply_mask_kq(int nq, const KHelper& kh, int stride_q, int stride_m, const q_float * q, const char * mask,
            FlashMS<q_step, k_step>& fms) {
        if constexpr (is_small_head) {
            mult_mask_kq(nq, kh, stride_q, stride_m, q, mask, fms);
        }
        else {
            mult_mask_kq_l(nq, kh, stride_q, stride_m, q, mask, fms);
        }
#ifdef __aarch64__
        float32x4_t vk[k_step/4];
        for (int j = 0; j < nq; ++j) {
            fms.update_M_S(j, vk);
        }
#else
        F16::Data vk[k_step/F16::block_size];
        for (int j = 0; j < nq; ++j) {
            fms.update_M_S(j, vk);
        }
#endif
    }

#ifdef __aarch64__
    static inline void convert(int nq, int stride_q, const float * q, float16_t * q_f16) {
        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < D; j += 8) {
                auto val1_f32 = vld1q_f32(q + j + 0);
                auto val2_f32 = vld1q_f32(q + j + 4);
                auto val_f16  = vcombine_f16(vcvt_f16_f32(val1_f32), vcvt_f16_f32(val2_f32));
                vst1q_f16(q_f16 + j, val_f16);
            }
            q += stride_q;
            q_f16 += D;
        }
    }
#endif

    template <typename KHelper, typename block_q8>
    static inline void mul_mask_kq(const KHelper& kh, int stride_m,
            const block_q8 * q, const char * mask, FlashMS<q_step, k_step>& fms) {
        static_assert(q_step <= 8);
        if constexpr (std::is_same_v<KHelper, HelperQ40<D, k_step>>) {
            DataInfo info{fms.cache, (const char *)q, k_step, (D/QK8_0)*sizeof(block_q8), 0, 1, nullptr};
#ifdef __aarch64__
            mul_mat_qX_0_q8_0<DequantizerQ40, q_step>(D, kh.block, kh.stride, info, k_step);
#else
            mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, q_step>(D, kh.block, kh.stride, info, k_step);
#endif
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ80<D, k_step>>) {
            DataInfo info{fms.cache, (const char *)q, k_step, (D/QK8_0)*sizeof(block_q8), 0, 1, nullptr};
#ifdef __aarch64__
            mul_mat_qX_0_q8_0<DequantizerQ80_x4, q_step>(D, kh.block, kh.stride, info, k_step);
#else
            if constexpr (D >= 128) {
                mul_mat_qX_0_q8_0_T<Q8_0_x4_Unpacker, q_step>(D, kh.block, kh.stride, info, k_step);
            } else {
                mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, q_step>(D, kh.block, kh.stride, info, k_step);
            }
#endif
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ41<D, k_step>>) {
            DataInfo info{fms.cache, (const char *)q, k_step, (D/QK8_1)*sizeof(block_q8), 0, 1, nullptr};
#ifdef __aarch64__
            mul_mat_qX_1_q8_1<DequantizerQ41, q_step>(D, kh.block, kh.stride, info, k_step);
#else
            mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, q_step>(D, kh.block, kh.stride, info, k_step);
#endif
        }
        else {
            GGML_ASSERT(false);
        }
#ifdef __aarch64__
        float32x4_t vk[k_step/4];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
#else
        F16::Data vk[k_step/F16::block_size];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
#endif
    }
    template <typename KHelper, typename block_q8>
    static inline void mul_mask_kq(int nq, const KHelper& kh, int stride_m,
            const block_q8 * q, const char * mask, FlashMS<q_step, k_step>& fms) {
        GGML_ASSERT(nq < 8);
        if constexpr (std::is_same_v<KHelper, HelperQ40<D, k_step>>) {
            DataInfo info{fms.cache, (const char *)q, k_step, (D/QK8_0)*sizeof(block_q8), 0, 1, nullptr};
            switch (nq) {
#ifdef __aarch64__
                case 1: mul_mat_qX_0_q8_0<DequantizerQ40, 1>(D, kh.block, kh.stride, info, k_step); break;
                case 2: mul_mat_qX_0_q8_0<DequantizerQ40, 2>(D, kh.block, kh.stride, info, k_step); break;
                case 3: mul_mat_qX_0_q8_0<DequantizerQ40, 3>(D, kh.block, kh.stride, info, k_step); break;
                case 4: mul_mat_qX_0_q8_0<DequantizerQ40, 4>(D, kh.block, kh.stride, info, k_step); break;
                case 5: mul_mat_qX_0_q8_0<DequantizerQ40, 5>(D, kh.block, kh.stride, info, k_step); break;
                case 6: mul_mat_qX_0_q8_0<DequantizerQ40, 6>(D, kh.block, kh.stride, info, k_step); break;
                case 7: mul_mat_qX_0_q8_0<DequantizerQ40, 7>(D, kh.block, kh.stride, info, k_step); break;
#else
                case 1: mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, 1>(D, kh.block, kh.stride, info, k_step); break;
                case 2: mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, 2>(D, kh.block, kh.stride, info, k_step); break;
                case 3: mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, 3>(D, kh.block, kh.stride, info, k_step); break;
                case 4: mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, 4>(D, kh.block, kh.stride, info, k_step); break;
                case 5: mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, 5>(D, kh.block, kh.stride, info, k_step); break;
                case 6: mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, 6>(D, kh.block, kh.stride, info, k_step); break;
                case 7: mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, 7>(D, kh.block, kh.stride, info, k_step); break;
#endif
            }
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ80<D, k_step>>) {
            DataInfo info{fms.cache, (const char *)q, k_step, (D/QK8_0)*sizeof(block_q8), 0, 1, nullptr};
#ifdef __aarch64__
            switch (nq) {
                case 1: mul_mat_qX_0_q8_0<DequantizerQ80_x4, 1>(D, kh.block, kh.stride, info, k_step); break;
                case 2: mul_mat_qX_0_q8_0<DequantizerQ80_x4, 2>(D, kh.block, kh.stride, info, k_step); break;
                case 3: mul_mat_qX_0_q8_0<DequantizerQ80_x4, 3>(D, kh.block, kh.stride, info, k_step); break;
                case 4: mul_mat_qX_0_q8_0<DequantizerQ80_x4, 4>(D, kh.block, kh.stride, info, k_step); break;
                case 5: mul_mat_qX_0_q8_0<DequantizerQ80_x4, 5>(D, kh.block, kh.stride, info, k_step); break;
                case 6: mul_mat_qX_0_q8_0<DequantizerQ80_x4, 6>(D, kh.block, kh.stride, info, k_step); break;
                case 7: mul_mat_qX_0_q8_0<DequantizerQ80_x4, 7>(D, kh.block, kh.stride, info, k_step); break;
            }
#else
            if constexpr (D >= 128) {
                switch (nq) {
                    case 1: mul_mat_qX_0_q8_0_T<Q8_0_x4_Unpacker, 1>(D, kh.block, kh.stride, info, k_step); break;
                    case 2: mul_mat_qX_0_q8_0_T<Q8_0_x4_Unpacker, 2>(D, kh.block, kh.stride, info, k_step); break;
                    case 3: mul_mat_qX_0_q8_0_T<Q8_0_x4_Unpacker, 3>(D, kh.block, kh.stride, info, k_step); break;
                    case 4: mul_mat_qX_0_q8_0_T<Q8_0_x4_Unpacker, 4>(D, kh.block, kh.stride, info, k_step); break;
                    case 5: mul_mat_qX_0_q8_0_T<Q8_0_x4_Unpacker, 5>(D, kh.block, kh.stride, info, k_step); break;
                    case 6: mul_mat_qX_0_q8_0_T<Q8_0_x4_Unpacker, 6>(D, kh.block, kh.stride, info, k_step); break;
                    case 7: mul_mat_qX_0_q8_0_T<Q8_0_x4_Unpacker, 7>(D, kh.block, kh.stride, info, k_step); break;
                }
            } else {
                switch (nq) {
                    case 1: mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 1>(D, kh.block, kh.stride, info, k_step); break;
                    case 2: mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 2>(D, kh.block, kh.stride, info, k_step); break;
                    case 3: mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 3>(D, kh.block, kh.stride, info, k_step); break;
                    case 4: mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 4>(D, kh.block, kh.stride, info, k_step); break;
                    case 5: mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 5>(D, kh.block, kh.stride, info, k_step); break;
                    case 6: mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 6>(D, kh.block, kh.stride, info, k_step); break;
                    case 7: mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, 7>(D, kh.block, kh.stride, info, k_step); break;
                }
            }
#endif
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ41<D, k_step>>) {
            DataInfo info{fms.cache, (const char *)q, k_step, (D/QK8_1)*sizeof(block_q8), 0, 1, nullptr};
            switch (nq) {
#ifdef __aarch64__
                case 1: mul_mat_qX_1_q8_1<DequantizerQ41, 1>(D, kh.block, kh.stride, info, k_step); break;
                case 2: mul_mat_qX_1_q8_1<DequantizerQ41, 2>(D, kh.block, kh.stride, info, k_step); break;
                case 3: mul_mat_qX_1_q8_1<DequantizerQ41, 3>(D, kh.block, kh.stride, info, k_step); break;
                case 4: mul_mat_qX_1_q8_1<DequantizerQ41, 4>(D, kh.block, kh.stride, info, k_step); break;
                case 5: mul_mat_qX_1_q8_1<DequantizerQ41, 5>(D, kh.block, kh.stride, info, k_step); break;
                case 6: mul_mat_qX_1_q8_1<DequantizerQ41, 6>(D, kh.block, kh.stride, info, k_step); break;
                case 7: mul_mat_qX_1_q8_1<DequantizerQ41, 7>(D, kh.block, kh.stride, info, k_step); break;
#else
                case 1: mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, 1>(D, kh.block, kh.stride, info, k_step); break;
                case 2: mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, 2>(D, kh.block, kh.stride, info, k_step); break;
                case 3: mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, 3>(D, kh.block, kh.stride, info, k_step); break;
                case 4: mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, 4>(D, kh.block, kh.stride, info, k_step); break;
                case 5: mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, 5>(D, kh.block, kh.stride, info, k_step); break;
                case 6: mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, 6>(D, kh.block, kh.stride, info, k_step); break;
                case 7: mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, 7>(D, kh.block, kh.stride, info, k_step); break;
#endif
            }
        }
        else {
            GGML_ASSERT(false);
        }
#ifdef __aarch64__
        float32x4_t vk[k_step/4];
        for (int j = 0; j < nq; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
#else
        F16::Data vk[k_step/F16::block_size];
        for (int j = 0; j < nq; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
#endif
    }
};

template <int D, int q_step, int k_step, typename KHelper, typename VHelper, typename KQHelper>
void compute_helper(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
        FlashMS<q_step, k_step>& fms,
        FlashQKV<D, q_step, k_step>& fqkv,
        const float * q, const char * mask, float * qkv) {
#ifdef __aarch64__
    float16_t q_f16[D*q_step];
#endif
    for (int i1 = 0; i1 < nq1/q_step; ++i1) {
        fms.init_qstep();
        kh.reset_block();
        vh.reset_block();
#ifdef __aarch64__
        KQHelper::convert(q_step, stride_q, q, q_f16);
#endif
        auto mr = mask;
        for (int k1 = 0; k1 < nk1/k_step; ++k1) {
#ifdef __aarch64__
            KQHelper::multiply_mask_kq(kh, D, stride_m, q_f16, mr, fms);
#else
            KQHelper::multiply_mask_kq(kh, stride_q, stride_m, q, mr, fms);
#endif
            fqkv.accumulate_qkv(vh, fms);
            kh.next_block();
            vh.next_block();
            mr += k_step*sizeof(ggml_half);
        }
        fqkv.normalize_and_store(fms, stride_qkv, qkv);

        q    += q_step*stride_q;
        mask += q_step*stride_m;
        qkv  += q_step*stride_qkv;
    }
    int n_left = nq1 - q_step*(nq1/q_step);
    if (n_left > 0) {
        fms.init_qstep();
        kh.reset_block();
        vh.reset_block();
#ifdef __aarch64__
        KQHelper::convert(n_left, stride_q, q, q_f16);
#endif
        auto mr = mask;
        for (int k1 = 0; k1 < nk1/k_step; ++k1) {
#ifdef __aarch64__
            KQHelper::multiply_mask_kq(n_left, kh, D, stride_m, q_f16, mr, fms);
#else
            KQHelper::multiply_mask_kq(n_left, kh, stride_q, stride_m, q, mr, fms);
#endif
            fqkv.accumulate_qkv(n_left, vh, fms);
            kh.next_block();
            vh.next_block();
            mr += k_step*sizeof(ggml_half);
        }
        fqkv.normalize_and_store(fms, n_left, stride_qkv, qkv);
    }
}

template <int D, int q_step, int k_step, typename KHelper, typename VHelper, typename KQHelper>
void compute_helper_q(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
        FlashMS<q_step, k_step>& fms,
        FlashQKV<D, q_step, k_step>& fqkv,
        const float * q, const char * mask, float * qkv) {
    typename KHelper::block_q8 q8[q_step*(D/QK8_0)];
    for (int i1 = 0; i1 < nq1/q_step; ++i1) {
        fms.init_qstep();
        kh.reset_block();
        vh.reset_block();
        HelperQ80<D, QK8_0>::convert(q_step, stride_q, q, q8);
        auto mr = mask;
        for (int k1 = 0; k1 < nk1/k_step; ++k1) {
            KQHelper::mul_mask_kq(kh, stride_m, q8, mr, fms);
            fqkv.accumulate_qkv(vh, fms);
            kh.next_block();
            vh.next_block();
            mr += k_step*sizeof(ggml_half);
        }
        fqkv.normalize_and_store(fms, stride_qkv, qkv);

        q    += q_step*stride_q;
        mask += q_step*stride_m;
        qkv  += q_step*stride_qkv;
    }
    int n_left = nq1 - q_step*(nq1/q_step);
    if (n_left > 0) {
        fms.init_qstep();
        kh.reset_block();
        vh.reset_block();
        HelperQ80<D, QK8_0>::convert(n_left, stride_q, q, q8);
        auto mr = mask;
        for (int k1 = 0; k1 < nk1/k_step; ++k1) {
            KQHelper::mul_mask_kq(n_left, kh, stride_m, q8, mr, fms);
            fqkv.accumulate_qkv(n_left, vh, fms);
            kh.next_block();
            vh.next_block();
            mr += k_step*sizeof(ggml_half);
        }
        fqkv.normalize_and_store(fms, n_left, stride_qkv, qkv);
    }
}

// Some of the methods in FlashAttn have two identical implementations that only differ by
// one version using a loop over the template parameter q_step, while the other using a loop
// over an input parameter nq (these are loops over the rows of q^T). I dislike this a lot,
// but performance drops signficantly if I remove the version with fixed q_step iterations.
// We only instantiate FlashAttn with q_step = 1 and q_step = 4 or 8 (depending on head size D),
// so when we have to process Nq rows, we process q_step*(Nq/q_step) using fixed q_step loops,
// and use the variable nq version (with lower performance) only for the remaining i1...q_step-1
// rows (if Nq is not a multiple of q_step). One could have made the number of q^T rows to
// process template parameter of such functions, but this would result in the compiler generating
// q_step-1 versions of these functions for us, which I though was too much with q_step = 8.
template <int D, int q_step, int k_step>
struct FlashAttn {
    static_assert(D%F16::block_size == 0 && D <= 256);
    static_assert(k_step%F16::block_size == 0);
    static_assert(q_step <= 4 || q_step%4 == 0);

    FlashAttn(float scale, float softcap) : fms(scale, softcap) {}

    template <typename KHelper, typename VHelper>
    void compute(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
            const float * q, const char * mask, float * qkv) {
        if constexpr (std::is_same_v<KHelper, HelperQ40<D, k_step>> || std::is_same_v<KHelper, HelperQ41<D, k_step>> ||
                      std::is_same_v<KHelper, HelperQ80<D, k_step>>) {
            compute_helper_q<D, q_step, k_step, KHelper, VHelper, FlashQKfp32<D, q_step, k_step>>(
                    kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv);
        } else {
            compute_helper<D, q_step, k_step, KHelper, VHelper, FlashQKfp32<D, q_step, k_step>>(
                    kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv);
        }
    }

    FlashMS<q_step, k_step>     fms;
    FlashQKV<D, q_step, k_step> fqkv;

};

#ifdef __AVX512BF16__

template <int D, int step>
struct HelperBF16 final : public BaseHelper<step> {
    using Base = BaseHelper<step>;
    HelperBF16(const char * data, int stride) : Base(data, stride) {}
    inline void load(int l1, __m512bh * vk) const {
        auto dr = Base::lblock(l1);
        for (int i = 0; i < D/32; ++i) vk[i] = __m512bh(_mm512_loadu_si512((const __m512i*)dr + i));
    }

    inline void load(int l1, int i, __m512& v1, __m512& v2) const {
        auto dr = Base::lblock(l1);
        v1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)dr + i + 0)), 16));
        v2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)dr + i + 1)), 16));
    }

    inline void load_2(int l1, __m512bh * vk) const {
        load(l1+0, vk+0);
        load(l1+1, vk+D/32);
    }

    inline void load_4(int l1, __m512bh * vk) const {
        load(l1+0, vk+0);
        load(l1+1, vk+1*D/32);
        load(l1+2, vk+2*D/32);
        load(l1+3, vk+3*D/32);
    }
};

template <int D, int q_step, int k_step>
struct FlashQKbf16 {
    static_assert(D%32 == 0 && D <= 256);
    static_assert(k_step%32 == 0);
    static_assert(q_step <= 4 || q_step%4 == 0);

    static inline void mult_mask_kq_one(int l1, int m1, int stride_q, int stride_m, const float * q, const char * mask,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        // q index is q_step*i1 + m1
        // k index is k_step*k1 + l1
        const ggml_half * mp = (const ggml_half *)(mask + stride_m*m1);
        fms.cache[k_step*m1 + l1 + 0] = fms.cache[k_step*m1 + l1 + 1] = -INFINITY;
        if (mp[l1+0] == fms.h_inf && mp[l1+1] == fms.h_inf) {
            return;
        }
        auto qr = q + m1*stride_q;
        for (int i = 0; i < D/32; ++i) {
            auto val1 = _mm512_loadu_ps(qr + 32*i);
            auto val2 = _mm512_loadu_ps(qr + 32*i + 16);
            qv[i] = _mm512_cvtne2ps_pbh(val2, val1);
        }
        if (mp[l1+0] != fms.h_inf) {
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i], qv[i]);
            fms.cache[k_step*m1 + l1 + 0] = _mm512_reduce_add_ps(vsum);
        }
        if (mp[l1+1] != fms.h_inf) {
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+D/32], qv[i]);
            fms.cache[k_step*m1 + l1 + 1] = _mm512_reduce_add_ps(vsum);
        }
    }

    static inline void mult_mask_kq_one(int l1, int m1, int stride_m, const ggml_bf16_t * q, const char * mask,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        // q index is q_step*i1 + m1
        // k index is k_step*k1 + l1
        const ggml_half * mp = (const ggml_half *)(mask + stride_m*m1);
        fms.cache[k_step*m1 + l1 + 0] = fms.cache[k_step*m1 + l1 + 1] = -INFINITY;
        if (mp[l1+0] == fms.h_inf && mp[l1+1] == fms.h_inf) {
            return;
        }
        auto qr = q + m1*D;
        for (int i = 0; i < D/32; ++i) qv[i] = __m512bh(_mm512_loadu_si512((const __m512i*)qr + i));
        if (mp[l1+0] != fms.h_inf) {
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i], qv[i]);
            fms.cache[k_step*m1 + l1 + 0] = _mm512_reduce_add_ps(vsum);
        }
        if (mp[l1+1] != fms.h_inf) {
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+D/32], qv[i]);
            fms.cache[k_step*m1 + l1 + 1] = _mm512_reduce_add_ps(vsum);
        }
    }

    static inline void mult_mask_kq_4(int l1, int m1, int stride_q, int stride_m, const float * q, const char * mask,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        // q index is q_step*i1 + m1
        // k index is k_step*k1 + l1
        const ggml_half * mp = (const ggml_half *)(mask + stride_m*m1);
        fms.cache[k_step*m1 + l1 + 0] = fms.cache[k_step*m1 + l1 + 1] =
        fms.cache[k_step*m1 + l1 + 2] = fms.cache[k_step*m1 + l1 + 3] = -INFINITY;
        if (mp[l1+0] == fms.h_inf && mp[l1+1] == fms.h_inf && mp[l1+2] == fms.h_inf && mp[l1+3] == fms.h_inf) {
            return;
        }
        auto qr = q + m1*stride_q;
        for (int i = 0; i < D/32; ++i) {
            auto val1 = _mm512_loadu_ps(qr + 32*i);
            auto val2 = _mm512_loadu_ps(qr + 32*i + 16);
            qv[i] = _mm512_cvtne2ps_pbh(val2, val1);
        }
        for (int k = 0; k < 4; ++k) {
            if (mp[l1+k] == fms.h_inf) continue;
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+k*(D/32)], qv[i]);
            fms.cache[k_step*m1 + l1 + k] = _mm512_reduce_add_ps(vsum);
        }
    }

    static inline void mult_mask_kq_4(int l1, int m1, int stride_m, const ggml_bf16_t * q, const char * mask,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        // q index is q_step*i1 + m1
        // k index is k_step*k1 + l1
        const ggml_half * mp = (const ggml_half *)(mask + stride_m*m1);
        fms.cache[k_step*m1 + l1 + 0] = fms.cache[k_step*m1 + l1 + 1] =
        fms.cache[k_step*m1 + l1 + 2] = fms.cache[k_step*m1 + l1 + 3] = -INFINITY;
        if (mp[l1+0] == fms.h_inf && mp[l1+1] == fms.h_inf && mp[l1+2] == fms.h_inf && mp[l1+3] == fms.h_inf) {
            return;
        }
        auto qr = q + m1*D;
        for (int i = 0; i < D/32; ++i) qv[i] = __m512bh(_mm512_loadu_si512((const __m512i *)qr + i));
        for (int k = 0; k < 4; ++k) {
            if (mp[l1+k] == fms.h_inf) continue;
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+k*(D/32)], qv[i]);
            fms.cache[k_step*m1 + l1 + k] = _mm512_reduce_add_ps(vsum);
        }
    }

    template <typename KHelper>
    static inline void multiply_mask_kq(const KHelper& kh, int stride_q, int stride_m, const float * q,
            const char * mask, FlashMS<q_step, k_step>& fms) {
        {
            __m512bh qv[D/32];
            if constexpr (D <= 128) {
                __m512bh vkh[D/8];
                for (int l1 = 0; l1 < k_step; l1 += 4) {
                    kh.load_4(l1, vkh);
                    for (int j = 0; j < q_step; ++j) {
                        mult_mask_kq_4(l1, j, stride_q, stride_m, q, mask, qv, vkh, fms);
                    }
                }
            } else {
                __m512bh vkh[D/16];
                for (int l1 = 0; l1 < k_step; l1 += 2) {
                    kh.load_2(l1, vkh);
                    for (int j = 0; j < q_step; ++j) {
                        mult_mask_kq_one(l1, j, stride_q, stride_m, q, mask, qv, vkh, fms);
                    }
                }
            }
        }
        __m512 vk[k_step/16];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk);
        }
    }

    template <typename KHelper>
    static inline void multiply_mask_kq(const KHelper& kh, int stride_m, const ggml_bf16_t * q,
            const char * mask, FlashMS<q_step, k_step>& fms) {
        {
            __m512bh qv[D/32];
            if constexpr (D <= 128) {
                __m512bh vkh[D/8];
                for (int l1 = 0; l1 < k_step; l1 += 4) {
                    kh.load_4(l1, vkh);
                    for (int j = 0; j < q_step; ++j) {
                        mult_mask_kq_4(l1, j, stride_m, q, mask, qv, vkh, fms);
                    }
                }
            } else {
                __m512bh vkh[D/16];
                for (int l1 = 0; l1 < k_step; l1 += 2) {
                    kh.load_2(l1, vkh);
                    for (int j = 0; j < q_step; ++j) {
                        mult_mask_kq_one(l1, j, stride_m, q, mask, qv, vkh, fms);
                    }
                }
            }
        }
        __m512 vk[k_step/16];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk);
        }
    }

    template <typename KHelper>
    static inline void multiply_mask_kq(int nq, const KHelper& kh, int stride_q, int stride_m, const float * q,
            const char * mask, FlashMS<q_step, k_step>& fms) {
        {
            __m512bh qv[D/32];
            __m512bh vkh[D/16];
            for (int l1 = 0; l1 < k_step; l1 += 2) {
                kh.load_2(l1, vkh);
                for (int m1 = 0; m1 < nq; ++m1) {
                    mult_mask_kq_one(l1, m1, stride_q, stride_m, q, mask, qv, vkh, fms);
                }
            }
        }
        __m512 vk[k_step/16];
        for (int j = 0; j < nq; ++j) {
            fms.update_M_S(j, vk);
        }
    }

    static inline void convert(int stride_q, const float * q, ggml_bf16_t * bf16) {
        auto qr = q;
        for (int j = 0; j < q_step; ++j) {
            for (int i = 0; i < D/32; ++i) {
                auto val1 = _mm512_loadu_ps(qr + 32*i);
                auto val2 = _mm512_loadu_ps(qr + 32*i + 16);
                _mm512_storeu_si512((__m512i *)bf16 + i, (__m512i)_mm512_cvtne2ps_pbh(val2, val1));
            }
            qr   += stride_q;
            bf16 += D;
        }
    }
};

template <int D, int q_step, int k_step>
struct FlashAttnBF16 {
    static_assert(D%32 == 0 && D <= 256);
    static_assert(k_step%32 == 0);
    static_assert(q_step <= 4 || q_step%4 == 0);

    FlashAttnBF16(float scale, float softcap) : fms(scale, softcap) {}

    template <typename KHelper, typename VHelper>
    void compute(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
            const float * q, const char * mask, float * qkv) {
        ggml_bf16_t q_bf16[q_step*D];
        for (int i1 = 0; i1 < nq1/q_step; ++i1) {
            fms.init_qstep();
            kh.reset_block();
            vh.reset_block();
            FlashQKbf16<D, q_step, k_step>::convert(stride_q, q, q_bf16);
            auto mr = mask;
            for (int k1 = 0; k1 < nk1/k_step; ++k1) {
                FlashQKbf16<D, q_step, k_step>::multiply_mask_kq(kh, stride_m, q_bf16, mr, fms);
                fqkv.accumulate_qkv(vh, fms);
                kh.next_block();
                vh.next_block();
                mr += k_step*sizeof(ggml_half);
            }
            fqkv.normalize_and_store(fms, stride_qkv, qkv);

            q    += q_step*stride_q;
            mask += q_step*stride_m;
            qkv  += q_step*stride_qkv;
        }
        int n_left = nq1 - q_step*(nq1/q_step);
        if (n_left > 0) {
            fms.init_qstep();
            kh.reset_block();
            vh.reset_block();
            auto mr = mask;
            for (int k1 = 0; k1 < nk1/k_step; ++k1) {
                FlashQKbf16<D, q_step, k_step>::multiply_mask_kq(n_left, kh, stride_q, stride_m, q, mr, fms);
                fqkv.accumulate_qkv(n_left, vh, fms);
                kh.next_block();
                vh.next_block();
                mr += k_step*sizeof(ggml_half);
            }
            fqkv.normalize_and_store(fms, n_left, stride_qkv, qkv);
        }
    }

    FlashMS<q_step, k_step>     fms;
    FlashQKV<D, q_step, k_step> fqkv;
};
#endif

template <int D, int q_step, int k_step, typename KHelper, typename VHelper>
inline void iqk_flash_helper(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
                        const float * q, const char * mask, float scale, float softcap, float * qkv) {

    if (nq1 >= q_step) {
        FlashAttn<D, q_step, k_step> fa(scale, softcap);
        fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv);
    } else {
        FlashAttn<D, 1, k_step> fa(scale, softcap);
        fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv);
    }
}

#ifdef __AVX512BF16__
template <int D, int q_step, int k_step>
inline void iqk_flash_helper_T(int nq1, int nk1, int stride_q, int stride_k, int stride_v, int stride_m, int stride_qkv,
                        const float * q, const char * k, const char * v, const char * mask,
                        float scale, float softcap, float * qkv) {
    HelperBF16<D, k_step> kh(k, stride_k);
    HelperBF16<D, k_step> vh(v, stride_v);
    if (nq1 >= q_step) {
        FlashAttnBF16<D, q_step, k_step> fa(scale, softcap);
        fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv);
    } else {
        FlashAttnBF16<D, 1, k_step> fa(scale, softcap);
        fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv);
    }
}
#endif

template <int D, int q_step, int k_step, typename KHelper>
inline void iqk_flash_helper_T(KHelper& kh, ggml_type type_v,
                        int nq1, int nk1, int stride_q, int stride_v, int stride_m, int stride_qkv,
                        const float * q, const char * v, const char * mask,
                        float scale, float softcap, float * qkv) {

    switch (type_v) {
        case GGML_TYPE_F16: {
            HelperF16<D, k_step> vh(v, stride_v);
            iqk_flash_helper<D, q_step, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv);
        } break;
        case GGML_TYPE_Q8_0: {
            HelperQ80<D, k_step> vh(v, stride_v);
            iqk_flash_helper<D, q_step, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv);
        } break;
        case GGML_TYPE_Q4_0: {
            HelperQ40<D, k_step> vh(v, stride_v);
            iqk_flash_helper<D, q_step, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv);
        } break;
        case GGML_TYPE_Q4_1: {
            HelperQ41<D, k_step> vh(v, stride_v);
            iqk_flash_helper<D, q_step, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv);
        } break;
        default: break;
    }
}

template <int D, int q_step, int k_step>
inline void iqk_flash_helper_T(ggml_type type_k, ggml_type type_v,
                        int nq1, int nk1, int stride_q, int stride_k, int stride_v, int stride_m, int stride_qkv,
                        const float * q, const char * k, const char * v, const char * mask,
                        float scale, float softcap, float * qkv) {

    switch (type_k) {
        case GGML_TYPE_F16: {
            HelperF16<D, k_step> kh(k, stride_k);
            iqk_flash_helper_T<D, q_step, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv);
        } break;
        case GGML_TYPE_Q8_0: {
            HelperQ80<D, k_step> kh(k, stride_k);
            iqk_flash_helper_T<D, q_step, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv);
        } break;
        case GGML_TYPE_Q4_0: {
            HelperQ40<D, k_step> kh(k, stride_k);
            iqk_flash_helper_T<D, q_step, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv);
        } break;
        case GGML_TYPE_Q4_1: {
            HelperQ41<D, k_step> kh(k, stride_k);
            iqk_flash_helper_T<D, q_step, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv);
        } break;
        default: break;
    }

}

inline bool flash_attn_is_supported(ggml_type type) {
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1) return true;
#ifdef __AVX512BF16__
    if (type == GGML_TYPE_BF16) return true;
#endif
    return false;
}
}

bool iqk_flash_attn_noalibi(int int_type_k,         // type of k
                            int int_type_v,         // type of v
                            int D,                  // head size
                            int nq1,                // number of columns in q
                            int nk1,                // number of rows in k
                            int stride_q,           // distance between q columns in bytes
                            int stride_k,           // distance between k rows in bytes
                            int stride_v,           // distance between v rows in bytes
                            int stride_m,           // distance between mask rows (in bytes
                            int stride_qkv,         // distance between rows in mask (in bytes)
                            const float * q,        // q matrix.
                            const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            float         scale,    // scale applied before softmax
                            float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            float       * qkv) {    // v*softmax(scale*(k*q))

    auto type_k = ggml_type(int_type_k);
    auto type_v = ggml_type(int_type_v);
    if (!flash_attn_is_supported(type_k) || !flash_attn_is_supported(type_v)) return false;
    if (!mask || nk1%32 != 0) return false; // the implementation assumes mask is not null and nk is a multiple of 32

    auto ck = (const char *)k;
    auto cv = (const char *)v;
    auto cm = (const char *)mask;

    stride_q /= sizeof(float); // q stride as float

#ifdef __AVX512BF16__
    if (type_k == GGML_TYPE_BF16 || type_v == GGML_TYPE_BF16) {
        if (type_k != GGML_TYPE_BF16 || type_v != GGML_TYPE_BF16) return false; // we do not support mixing bf16 with other types
        switch (D) {
            case 64:
                iqk_flash_helper_T< 64, 8, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
            case 96:
                iqk_flash_helper_T< 96, 8, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
            case 128:
                iqk_flash_helper_T<128, 8, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
            case 256:
                iqk_flash_helper_T<256, 8, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
            default:
                return false;
        }

        return true;
    }
#endif

    switch (D) {
        case 64:
            iqk_flash_helper_T< 64, F16::q_step, 32>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
        // Disable until we fix accumulate_qkv for odd D/16
        //case 80:
        //    iqk_flash_helper_T< 80, 4, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
        case 96:
            iqk_flash_helper_T< 96, F16::q_step, 32>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
        // Disable until we fix accumulate_qkv for odd D/16
        //case 112:
        //    iqk_flash_helper_T<112, 4, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
        case 128:
            iqk_flash_helper_T<128, F16::q_step, 32>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
        case 256:
            iqk_flash_helper_T<256, F16::q_step, 32>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
        default:
            return false;
    }

    return true;
}

#else  // IQK_IMPLEMENT

bool iqk_mul_mat(int, long, long, long, int, const void *, long, int, const void *, long, float *, long, int, int) {
    return false;
}

bool iqk_mul_mat_moe(long, long, long, int, int, const void *, long, int, const void *, long, float *, long, long,
        const void *, int, int) {
    return false;
}

bool iqk_flash_attn_noalibi([[maybe_unused]] int int_type_k,         // type of k
                            [[maybe_unused]] int int_type_v,         // type of v
                            [[maybe_unused]] int D,                  // head size
                            [[maybe_unused]] int nq,                 // number of columns in q
                            [[maybe_unused]] int nk,                 // number of rows in k
                            [[maybe_unused]] int stride_q,           // distance between q columns in bytes
                            [[maybe_unused]] int stride_k,           // distance between k rows in bytes
                            [[maybe_unused]] int stride_v,           // distance between v rows in bytes
                            [[maybe_unused]] int stride_m,           // distance between mask rows (in bytes
                            [[maybe_unused]] int stride_qkv,         // distance between rows in mask (in bytes)
                            [[maybe_unused]] const float * q,        // q matrix.
                            [[maybe_unused]] const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            [[maybe_unused]] const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            [[maybe_unused]] const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            [[maybe_unused]] float         scale,    // scale applied before softmax
                            [[maybe_unused]] float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            [[maybe_unused]] float       * qkv) {    // v*softmax(scale*(k*q))
    return false;
}

#endif
