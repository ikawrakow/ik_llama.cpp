// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp fenc=utf-8 :vi
//
//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "iqk_config.h"

#if defined IQK_IMPLEMENT

#include <cstring>
#include <type_traits>
#include <vector>

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

#endif
