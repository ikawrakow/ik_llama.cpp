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

#endif

#endif
