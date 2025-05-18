// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp fenc=utf-8 :vi
//
//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "iqk_config.h"

#if defined IQK_IMPLEMENT

#include <cstring>
#include <type_traits>
#include <vector>
#include <cstdint>

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "iqk_mul_mat.h"
#include "iqk_quantize.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#define FA_TIMING 0

#include <utility>
#include <array>
#if FA_TIMING
#include <chrono>
#include <mutex>
struct Perf {
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    std::array<double, 5> times = {};
    std::mutex mutex;
    bool report;
    static auto cur_time() { return std::chrono::high_resolution_clock::now(); }
    inline void accum(int what, const TimePoint& t1) {
        auto t2 = cur_time();
        auto dt = delta(t1, t2);
        std::lock_guard<std::mutex> lock(mutex);
        times[what] += dt;
    }
    inline void accum_nolock(int what, const TimePoint& t1) {
        auto t2 = cur_time();
        auto dt = delta(t1, t2);
        times[what] += dt;
    }
    inline void add(const Perf& other) {
        std::lock_guard<std::mutex> lock(mutex);
        for (int i = 0; i < int(times.size()); ++i) times[i] += other.times[i];
    }
    Perf(bool r) : report(r) {}
    ~Perf() {
        if (report) {
            double tot = 0;
            for (auto& t : times) tot += t;
            if (!tot) return;
            printf("======================= Timing: %g ms in total\n", tot);
            for (int i = 0; i < int(times.size()); ++i) {
                if (times[i]) {
                    printf("%d:  %g ms -> %g%c\n", i, times[i], 100*times[i]/tot, '%');
                }
            }
        }
    }
    static Perf& instance() {
        static Perf p(true);
        return p;
    }
    static double delta(const TimePoint& t1, const TimePoint& t2) {
        return 1e-6*std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
    }
};
#endif

#ifdef __AVX2__
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
#endif

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
#ifdef __AVX__
    inline void store(int ix, int iy, __m128 result) const {
        _mm_storeu_ps(dst_row(iy) + ix, result);
    }
    inline void store(int ix, int iy, __m256 result) const {
        _mm256_storeu_ps(dst_row(iy) + ix, result);
    }
#endif
#ifdef __AVX512F__
    inline void store(int ix, int iy, __m512 result) const {
        _mm512_storeu_ps(dst_row(iy) + ix, result);
    }
#endif
#ifdef __ARM_NEON
    inline void store(int ix, int iy, float32x4_t result) const {
        vst1q_f32(dst_row(iy) + ix, result);
    }
#endif
    inline float * dst_row(int iy) const {
        if (!row_mapping) return s + (cur_y + iy)*bs;
        int i12 = row_mapping[cur_y + iy].i2;
        int i1  = row_mapping[cur_y + iy].i1;
        int i2  = i12;
        return s + i1*bs + i2*bs2;
    }
};

typedef void (*mul_mat_t)(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x);

#define IQK_MAX_NY 8

#define IQK_SET_MUL_MAT_FUNCTIONS_T(kernel, Dequantizer, funcs) \
            funcs[0] = kernel<Dequantizer, 1>;\
            funcs[1] = kernel<Dequantizer, 2>;\
            funcs[2] = kernel<Dequantizer, 3>;\
            funcs[3] = kernel<Dequantizer, 4>;\
            funcs[4] = kernel<Dequantizer, 5>;\
            funcs[5] = kernel<Dequantizer, 6>;\
            funcs[6] = kernel<Dequantizer, 7>;\
            funcs[7] = kernel<Dequantizer, 8>;\

#define IQK_SET_MUL_MAT_FUNCTIONS(kernel, funcs) \
            funcs[0] = kernel<1>;\
            funcs[1] = kernel<2>;\
            funcs[2] = kernel<3>;\
            funcs[3] = kernel<4>;\
            funcs[4] = kernel<5>;\
            funcs[5] = kernel<6>;\
            funcs[6] = kernel<7>;\
            funcs[7] = kernel<8>;\


// ==================================================================================================

static inline void make_q4_scales(const uint8_t * scales8, uint32_t * aux32) {
    const uint16_t * scales = (const uint16_t *)scales8;
    const uint32_t a0 = scales[0] | (scales[1] << 16);
    const uint32_t a1 = scales[2] | (scales[3] << 16);
    const uint32_t a2 = scales[4] | (scales[5] << 16);
    aux32[3] = ((a2 >> 4) & 0x0f0f0f0f) | ((a1 >> 2) & 0x30303030);
    aux32[1] = ((a2 >> 0) & 0x0f0f0f0f) | ((a0 >> 2) & 0x30303030);
    aux32[2] = a1 & 0x3f3f3f3f;
    aux32[0] = a0 & 0x3f3f3f3f;
}

#if !(defined HAVE_FANCY_SIMD && defined __AVX512VPOPCNTDQ__)
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

#ifdef __AVX2__

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

static inline float hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
static inline float hsum_float_8(__m256 x) {
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
static inline float hmax_float_8(__m256 x) {
    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    max4 = _mm_max_ps( max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4));
    return  _mm_cvtss_f32(max4);
}

static inline __m256 hsum_float_8x8(__m256 * accm) {
    for (int i = 0; i < 4; ++i) {
        accm[i] = _mm256_add_ps(_mm256_permute2f128_ps(accm[i], accm[i+4], 0x20), _mm256_permute2f128_ps(accm[i], accm[i+4], 0x31));
        //accm[i] = _mm256_set_m128(_mm_add_ps(_mm256_castps256_ps128(accm[i+4]), _mm256_extractf128_ps(accm[i+4], 1)),
        //                          _mm_add_ps(_mm256_castps256_ps128(accm[i+0]), _mm256_extractf128_ps(accm[i+0], 1)));
    }
    for (int i = 0; i < 2; ++i) accm[i] = _mm256_add_ps(_mm256_unpacklo_ps(accm[i], accm[i+2]), _mm256_unpackhi_ps(accm[i], accm[i+2]));
    return _mm256_add_ps(_mm256_unpacklo_ps(accm[0], accm[1]), _mm256_unpackhi_ps(accm[0], accm[1]));
}

static inline __m128i load_iq4nl_values_128() {
    static const uint8_t kvalues_iq4nl[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};
    return _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
}

static inline __m256i load_iq4nl_values_256() {
    auto val128 = load_iq4nl_values_128();
    return MM256_SET_M128I(val128, val128);
}

#ifdef HAVE_FANCY_SIMD
static inline __m512i load_iq4nl_values_512() {
    auto val256 = load_iq4nl_values_256();
    return _mm512_inserti32x8(_mm512_castsi256_si512(val256), val256, 1);
}
#endif

static inline __m128i load_iq4k_values_128() {
    return _mm_loadu_si128((const __m128i *)iq4k_values);
}

static inline __m256i load_iq4k_values_256() {
    auto val128 = load_iq4k_values_128();
    return MM256_SET_M128I(val128, val128);
}

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

template <int nrc> struct Q8_16 {

    constexpr static int nrc_y = nrc;

    Q8_16(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto ptr = (const float *)info.src1_row(iy);
            std::memcpy(d + 5*iy, ptr, 5*sizeof(float));
            y[iy] = (const int8_t *)(ptr + 5);
        }
    }

#ifdef HAVE_FANCY_SIMD
    inline __m512i load_quants64(int iy, int i) const { return _mm512_loadu_si512((const __m512i*)y[iy] + i); }
#endif
    inline __m256i load_quants(int iy, int i) const { return _mm256_loadu_si256((const __m256i*)y[iy] + i); }
    inline float scale(int iy, int k) const { return d[5*iy+k]; }
    inline float sum_row(int iy) const { return d[5*iy + 4]; }
    inline __m128 scale(int iy) const { return _mm_loadu_ps(d + 5*iy); }

    float d[5*nrc_y];
    const int8_t * y[nrc_y];
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

template <typename Block, bool per_row_scale = false, bool is_f16 = false>
struct BaseDequantizer {
    BaseDequantizer(const void * vx, size_t bx) : vx(vx), bx(bx) {}
    inline void new_row(int ix) {
        if constexpr (per_row_scale) {
            if constexpr (is_f16) {
                const ggml_half * dptr = (const ggml_half *)((const char *)vx + bx*ix);
                d = GGML_FP16_TO_FP32(*dptr);
                x = (const Block *)(dptr + 1);
            } else {
                const float * dptr = (const float *)((const char *)vx + bx*ix);
                d = *dptr;
                x = (const Block *)(dptr + 1);
            }
        } else {
            x = (const Block *)((const char *)vx + bx*ix);
        }
    }

    const void *  vx;
    const size_t  bx;
    const Block * x;

    float d;
};

template <typename Q8, typename Bits>
static inline void multiply_add(const Bits& bits, const __m256i * scales, int j, int i, const Q8& q8, __m256i * sumi) {
    if (j == 0) {
#ifdef HAVE_FANCY_SIMD
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
#ifdef HAVE_FANCY_SIMD
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

template <typename Q8, typename Bits>
static inline void multiply_add_avx2(const Bits& bits, const __m256i * scales, int j, int i, const Q8& q8, __m256i * sumi) {
    __m256i p[4];
    if (j == 0) {
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            for (int k = 0; k < 4; ++k) {
                auto s = _mm256_sign_epi8(bits.values[k], bits.values[k]);
                p[k] = _mm256_madd_epi16(scales[k], _mm256_maddubs_epi16(s, _mm256_sign_epi8(q8.load_quants(iy, i, k), bits.values[k])));
            }
            sumi[iy] = _mm256_add_epi32(_mm256_add_epi32(p[0], p[1]), _mm256_add_epi32(p[2], p[3]));
        }
    } else {
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            for (int k = 0; k < 4; ++k) {
                auto s = _mm256_sign_epi8(bits.values[k], bits.values[k]);
                p[k] = _mm256_madd_epi16(scales[k], _mm256_maddubs_epi16(s, _mm256_sign_epi8(q8.load_quants(iy, i, 4+k), bits.values[k])));
            }
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p[0], p[2]));
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p[1], p[3]));
        }
    }
}

#ifdef HAVE_FANCY_SIMD

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
    inline void prepare64a(const uint8_t * q4) {
        for (int k = 0; k < 4; ++k) {
            auto q4bits = _mm256_loadu_si256((const __m256i*)q4 + k);
            values[k] = _mm512_inserti32x8(_mm512_castsi256_si512(q4bits), _mm256_srli_epi16(q4bits, 4), 1);
            values[k] = _mm512_and_si512(values[k], ml);
        }
    }
    __m512i values[4];
    const __m512i ml = _mm512_set1_epi8(0xf);
    const BlockPermuter perm;
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

#else

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

#endif

#else

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

#endif

#endif
