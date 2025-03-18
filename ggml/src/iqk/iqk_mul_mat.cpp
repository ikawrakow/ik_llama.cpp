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

struct MulMat {
    std::array<mul_mat_t, 8> funcs = {};
    mul_mat_t func16 = nullptr;
    inline void mul_mat_NxM(int n, const void * vx, size_t bx, DataInfo& info, int nrc_x, int nrc_y) {
#ifdef __aarch64__
        constexpr int k_x_step = 64; //8192; // Tiling does not seem to help on my M2 Max (but difference to tiling is small)
#else
        constexpr int k_x_step = 64; // This works best on my Ryzen-7950X (but differences to other tile size are small)
#endif
        if (func16 && nrc_y >= 16) {
            int n_step = (nrc_y - info.cur_y)/16;
            for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                auto this_info = info;
                this_info.s += ix;
                int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                for (int iy = 0; iy < n_step; ++iy) {
                    func16(n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                    this_info.cur_y += 16;
                }
            }
            info.cur_y += 16 * n_step;
            if (info.cur_y == nrc_y) return;
        }
        int ny = funcs.size();
        while (!funcs[ny-1] && ny > 0) --ny;
        int n_left = nrc_y - info.cur_y;
        int n_step = n_left/ny;
        if (n_step > 0) {
            if (n_step*ny != n_left) {
                ++n_step;
                int ny1 = n_left/n_step;
                int ny2 = ny1 + 1;
                int my1 = n_step*ny2 - n_left;
                int my2 = n_step - my1;
                for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                    auto this_info = info;
                    this_info.s += ix;
                    int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                    for (int iy = 0; iy < my1; ++iy) {
                        funcs[ny1-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny1;
                    }
                    for (int iy = 0; iy < my2; ++iy) {
                        funcs[ny2-1](n, (const void *)((const char *)vx + ix*bx), bx, this_info, this_nrc_x);
                        this_info.cur_y += ny2;
                    }
                }
                info.cur_y += n_left;
            }
            else {
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
        }
        n_left = nrc_y - info.cur_y;
        if (n_left > 0) {
            funcs[n_left-1](n, vx, bx, info, nrc_x);
        }
    }
    inline void gelu(int n, const float * src, float * dst);
    inline void relu(int n, const float * src, float * dst);
    inline void silu(int n, const float * src, float * dst);
    inline void activate(ggml_unary_op op, int n, const float * src, float * dst) {
        if      (op == GGML_UNARY_OP_GELU) gelu(n, src, dst);
        else if (op == GGML_UNARY_OP_RELU) relu(n, src, dst);
        else if (op == GGML_UNARY_OP_SILU) silu(n, src, dst);
        else GGML_ABORT("fatal error");
    }
    inline void mul_mat_up_gate_NxM(int n, const void * vx_up, const void * vx_gate, size_t bx, DataInfo& info, int nrc_x, int nrc_y, int unary_op) {
#ifdef __aarch64__
        constexpr int k_x_step = 64; //8192; // Tiling does not seem to help on my M2 Max (but difference to tiling is small)
#else
        constexpr int k_x_step = 64; // This works best on my Ryzen-7950X (but differences to other tile size are small)
#endif
        auto op = ggml_unary_op(unary_op);
        float tmp[k_x_step*16];
        if (func16 && nrc_y >= 16) {
            int n_step = (nrc_y - info.cur_y)/16;
            for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                auto this_info = info;
                this_info.s += ix;
                int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                for (int iy = 0; iy < n_step; ++iy) {
                    func16(n, (const void *)((const char *)vx_gate + ix*bx), bx, this_info, this_nrc_x);
                    for (int ky = 0; ky < 16; ++ky) {
                        activate(op, this_nrc_x, this_info.dst_row(ky), tmp + ky*k_x_step);
                    }
                    func16(n, (const void *)((const char *)vx_up + ix*bx), bx, this_info, this_nrc_x);
                    for (int ky = 0; ky < 16; ++ky) {
                        auto result = this_info.dst_row(ky);
                        for (int j = 0; j < this_nrc_x; ++j) result[j] *= tmp[ky*k_x_step + j];
                    }
                    this_info.cur_y += 16;
                }
            }
            info.cur_y += 16 * n_step;
            if (info.cur_y == nrc_y) return;
        }
        int ny = funcs.size();
        while (!funcs[ny-1] && ny > 0) --ny;
        int n_left = nrc_y - info.cur_y;
        int n_step = n_left/ny;
        if (n_step > 0) {
            if (n_step*ny != n_left) {
                ++n_step;
                int ny1 = n_left/n_step;
                int ny2 = ny1 + 1;
                int my1 = n_step*ny2 - n_left;
                int my2 = n_step - my1;
                for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                    auto this_info = info;
                    this_info.s += ix;
                    int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                    for (int iy = 0; iy < my1; ++iy) {
                        funcs[ny1-1](n, (const void *)((const char *)vx_gate + ix*bx), bx, this_info, this_nrc_x);
                        for (int ky = 0; ky < ny1; ++ky) activate(op, this_nrc_x, this_info.dst_row(ky), tmp + ky*k_x_step);
                        funcs[ny1-1](n, (const void *)((const char *)vx_up + ix*bx), bx, this_info, this_nrc_x);
                        for (int ky = 0; ky < ny1; ++ky) {
                            auto result = this_info.dst_row(ky);
                            for (int j = 0; j < this_nrc_x; ++j) result[j] *= tmp[ky*k_x_step + j];
                        }
                        this_info.cur_y += ny1;
                    }
                    for (int iy = 0; iy < my2; ++iy) {
                        funcs[ny2-1](n, (const void *)((const char *)vx_gate + ix*bx), bx, this_info, this_nrc_x);
                        for (int ky = 0; ky < ny2; ++ky) activate(op, this_nrc_x, this_info.dst_row(ky), tmp + ky*k_x_step);
                        funcs[ny2-1](n, (const void *)((const char *)vx_up + ix*bx), bx, this_info, this_nrc_x);
                        for (int ky = 0; ky < ny2; ++ky) {
                            auto result = this_info.dst_row(ky);
                            for (int j = 0; j < this_nrc_x; ++j) result[j] *= tmp[ky*k_x_step + j];
                        }
                        this_info.cur_y += ny2;
                    }
                }
                info.cur_y += n_left;
            }
            else {
                for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                    auto this_info = info;
                    this_info.s += ix;
                    int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                    for (int iy = 0; iy < n_step; ++iy) {
                        funcs[ny-1](n, (const void *)((const char *)vx_gate + ix*bx), bx, this_info, this_nrc_x);
                        for (int ky = 0; ky < ny; ++ky) activate(op, this_nrc_x, this_info.dst_row(ky), tmp + ky*k_x_step);
                        funcs[ny-1](n, (const void *)((const char *)vx_up + ix*bx), bx, this_info, this_nrc_x);
                        for (int ky = 0; ky < ny; ++ky) {
                            auto result = this_info.dst_row(ky);
                            for (int j = 0; j < this_nrc_x; ++j) result[j] *= tmp[ky*k_x_step + j];
                        }
                        this_info.cur_y += ny;
                    }
                }
                info.cur_y += ny * n_step;
            }
        }
        n_left = nrc_y - info.cur_y;
        if (n_left > 0) {
            for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                auto this_info = info;
                this_info.s += ix;
                int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                funcs[n_left-1](n, (const void *)((const char *)vx_gate + ix*bx), bx, this_info, this_nrc_x);
                for (int ky = 0; ky < n_left; ++ky) activate(op, this_nrc_x, this_info.dst_row(ky), tmp + ky*k_x_step);
                funcs[n_left-1](n, (const void *)((const char *)vx_up + ix*bx), bx, this_info, this_nrc_x);
                for (int ky = 0; ky < n_left; ++ky) {
                    auto result = this_info.dst_row(ky);
                    for (int j = 0; j < this_nrc_x; ++j) result[j] *= tmp[ky*k_x_step + j];
                }
            }
        }
    }
    static bool prepare(int typeA, int typeB, int ne00, MulMat& mm, int Ny);
    static inline int num_rows(ggml_type type) {
#ifdef HAVE_FANCY_SIMD
        switch (type) {
            case GGML_TYPE_Q2_K_R4:
            case GGML_TYPE_Q3_K_R4:
            case GGML_TYPE_Q6_K_R4:
            case GGML_TYPE_IQ2_K_R4:
            case GGML_TYPE_IQ3_K_R4:
            case GGML_TYPE_IQ4_K_R4:
            case GGML_TYPE_IQ5_K_R4:
            case GGML_TYPE_IQ4_KS_R4:
            case GGML_TYPE_IQ2_XXS_R4:
            case GGML_TYPE_IQ2_XS_R4:
            case GGML_TYPE_IQ2_S_R4:
            case GGML_TYPE_IQ3_XXS_R4:
            case GGML_TYPE_IQ1_S_R4:
            case GGML_TYPE_IQ1_M_R4:
            case GGML_TYPE_IQ3_S_R4: return 4;
            case GGML_TYPE_IQ4_NL_R4:
            case GGML_TYPE_Q5_0_R4:
            case GGML_TYPE_Q6_0_R4:
            case GGML_TYPE_IQ2_BN_R4:
            case GGML_TYPE_IQ4_XS_R8:
            case GGML_TYPE_Q4_K_R4:
            case GGML_TYPE_Q5_K_R4:
            case GGML_TYPE_Q8_KV:
            case GGML_TYPE_Q8_KV_R8:
            case GGML_TYPE_Q8_K_R8: return 8;
            case GGML_TYPE_Q4_0_R8:
            case GGML_TYPE_Q8_0_R8:
            case GGML_TYPE_BF16_R16: return 16;
            default: return 1;
        }
#else
        switch (type) {
            case GGML_TYPE_Q2_K_R4:
            case GGML_TYPE_Q3_K_R4:
            case GGML_TYPE_Q4_K_R4:
            case GGML_TYPE_Q5_K_R4:
            case GGML_TYPE_Q6_K_R4:
            case GGML_TYPE_Q5_0_R4:
            case GGML_TYPE_Q6_0_R4:
            case GGML_TYPE_IQ4_NL_R4:
            case GGML_TYPE_IQ2_K_R4:
            case GGML_TYPE_IQ3_K_R4:
            case GGML_TYPE_IQ4_K_R4:
            case GGML_TYPE_IQ5_K_R4:
            case GGML_TYPE_IQ4_KS_R4:
            case GGML_TYPE_IQ2_XXS_R4:
            case GGML_TYPE_IQ2_XS_R4:
            case GGML_TYPE_IQ2_S_R4:
            case GGML_TYPE_IQ3_XXS_R4:
            case GGML_TYPE_IQ3_S_R4:
            case GGML_TYPE_IQ1_S_R4:
            case GGML_TYPE_IQ1_M_R4:
            case GGML_TYPE_IQ2_BN_R4: return 4;
            case GGML_TYPE_IQ4_XS_R8:
            case GGML_TYPE_Q4_0_R8:
            case GGML_TYPE_Q8_0_R8:
            case GGML_TYPE_Q8_KV:
            case GGML_TYPE_Q8_KV_R8:
            case GGML_TYPE_Q8_K_R8: return 8;
            case GGML_TYPE_BF16_R16: return 16;
            default: return 1;
        }
#endif
    }
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

    size_t row_size_qx = strideA; //*ggml_type_size(ggml_type(typeA));
    size_t row_size_qy = strideB; //*ggml_type_size(ggml_type(typeB));
    //if (ith == 0) printf("%s: ne00 = %d, row_size_qx = %d, strideA = %d\n", __func__, int(ne00), int(row_size_qx), int(strideA));

    auto num_rows = MulMat::num_rows(ggml_type(typeA));
    GGML_ASSERT(Nx%num_rows == 0);
    auto nrc_x = (Nx/num_rows + nth - 1)/nth;
    auto first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx/num_rows) nrc_x = Nx/num_rows - first_x;

    DataInfo info{C + first_x*num_rows, (const char *)B, (size_t)stride_C, row_size_qy, 0, 1, nullptr, 0};

    mm.mul_mat_NxM(ne00, (const char *)A + row_size_qx*first_x*num_rows, row_size_qx, info, nrc_x*num_rows, Ny);

    return true;
}

namespace {
inline uint32_t simple_gcd(uint32_t a, uint32_t b) {
    while (a != b) {
        if (a > b) a -= b;
        else b -= a;
    }
    return a;
}
}

bool iqk_mul_mat_4d(long Nx, long Ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        long nb02, long nb03, long nb12, long nb13, long nb2, long nb3,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth) {

    auto r2 = ne12 / ne02;
    auto r3 = ne13 / ne03;

    if (ne13 == 1 && Ny == 1 && r2 > 1) {
        int gcd = simple_gcd(ne02, nth);
        int counter = 0;
        for (int64_t i12 = 0; i12 < ne02; i12++) {
            if ((counter++ % gcd) == (ith%gcd)) {
                if (!iqk_mul_mat(Nx, r2, ne00,
                            typeA, (const char *)A + i12*nb02, strideA,
                            typeB, (const char *)B + i12*r2*nb12, nb12,
                            C + r2*i12*nb2, nb2,
                            ith/gcd, nth/gcd)) return false;
            }
        }
        return true;
    }

    if (ne13 == 1 && ne12 > 1 && ne12 == ne02 && Ny == 1 && nb02 < strideA) {
        //printf("TG attention gemm for %d heads and Nx = %d\n", (int)ne02, (int)Nx);
        MulMat mm;
        if (!MulMat::prepare(typeA, typeB, ne00, mm, Ny)) {
            return false;
        }
        int n_per_thread = (Nx + nth - 1)/nth;
        int first = ith*n_per_thread;
        if (first >= Nx) return true;
        int last = first + n_per_thread <= Nx ? first + n_per_thread : Nx;
        for (int ix = first; ix < last; ++ix) {
            for (int i02 = 0; i02 < ne02; ++i02) {
                DataInfo info{C + ix + i02*nb2, (const char *)B + i02*nb12, (size_t)nb2, (size_t)nb12, 0, 1, nullptr, 0};
                mm.funcs[0](ne00, (const void *)((const char *)A + ix*strideA + i02*nb02), nb02, info, 1);
            }
        }
        return true;
    }

    int gcd = simple_gcd(ne12*ne13, nth);
    int counter = 0;
    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            if ((counter++ % gcd) == (ith%gcd)) {
                if (!iqk_mul_mat(Nx, Ny, ne00,
                            typeA, (const char *)A + i12/r2*nb02 + i13/r3*nb03, strideA,
                            typeB, (const char *)B + i12*nb12 + i13*nb13, strideB,
                            C + i12*nb2 + i13*nb3, stride_C,
                            ith/gcd, nth/gcd)) return false;
            }
        }
    }
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
    size_t row_size_qx = strideA;
    size_t row_size_qy = strideB;
    auto num_rows = MulMat::num_rows(ggml_type(typeA));
    GGML_ASSERT(Nx%num_rows == 0);
    auto nrc_x = (Nx/num_rows + nth - 1)/nth;
    auto first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx/num_rows) nrc_x = Nx/num_rows - first_x;
    first_x *= num_rows;
    nrc_x *= num_rows;
    DataInfo info{C + first_x, (const char *)B, nb1/sizeof(float),
        row_size_qy, 0, ne11, row_mapping, nb2/sizeof(float)};
    mm.mul_mat_NxM(ne00, (const char *)A + row_size_qx*first_x, row_size_qx, info, nrc_x, Ny);
    return true;
}

bool iqk_moe_fused_up_gate(long Nx, long Ny, long ne00, int ne11, int unary_op,
        int typeA, const void * Aup, const void * Agate, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long nb1, long nb2, const void * vrow_mapping, int ith, int nth) {

    const mmid_row_mapping * row_mapping = (const mmid_row_mapping *)vrow_mapping;
    assert(row_mapping != nullptr);

    MulMat mm;
    if (!MulMat::prepare(typeA, typeB, ne00, mm, Ny)) {
        return false;
    }
    size_t row_size_qx = strideA;
    size_t row_size_qy = strideB;
    auto num_rows = MulMat::num_rows(ggml_type(typeA));
    GGML_ASSERT(Nx%num_rows == 0);
    auto nrc_x = (Nx/num_rows + nth - 1)/nth;
    auto first_x = ith*nrc_x;
    if (first_x + nrc_x > Nx/num_rows) nrc_x = Nx/num_rows - first_x;
    first_x *= num_rows;
    nrc_x *= num_rows;
    DataInfo info{C + first_x, (const char *)B, nb1/sizeof(float),
        row_size_qy, 0, ne11, row_mapping, nb2/sizeof(float)};
    mm.mul_mat_up_gate_NxM(ne00, (const char *)Aup + row_size_qx*first_x, (const char *)Agate + row_size_qx*first_x, row_size_qx, info, nrc_x, Ny, unary_op);
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

#ifdef __AVX2__
static const uint64_t iq1s_grid_us[2048] = {
    0x0000000000000000, 0x0000000000000002, 0x0000000000000101, 0x0000000000000200,
    0x0000000000000202, 0x0000000000010001, 0x0000000000010101, 0x0000000000020000,
    0x0000000000020002, 0x0000000000020200, 0x0000000000020202, 0x0000000001000101,
    0x0000000001010001, 0x0000000001010100, 0x0000000001010102, 0x0000000001020101,
    0x0000000002000000, 0x0000000002000002, 0x0000000002000200, 0x0000000002000202,
    0x0000000002010101, 0x0000000002020000, 0x0000000002020002, 0x0000000002020200,
    0x0000000002020202, 0x0000000100000100, 0x0000000100000101, 0x0000000100010001,
    0x0000000100010100, 0x0000000100010102, 0x0000000100010201, 0x0000000100010202,
    0x0000000100020101, 0x0000000101000001, 0x0000000101000102, 0x0000000101000201,
    0x0000000101010002, 0x0000000101010101, 0x0000000101010202, 0x0000000101020001,
    0x0000000101020100, 0x0000000101020102, 0x0000000101020200, 0x0000000102000101,
    0x0000000102010001, 0x0000000102010100, 0x0000000102010102, 0x0000000102020101,
    0x0000000200000000, 0x0000000200000002, 0x0000000200000200, 0x0000000200000202,
    0x0000000200010101, 0x0000000200020000, 0x0000000200020002, 0x0000000200020200,
    0x0000000200020202, 0x0000000201000101, 0x0000000201010001, 0x0000000201010201,
    0x0000000201020100, 0x0000000201020201, 0x0000000202000000, 0x0000000202000002,
    0x0000000202000200, 0x0000000202000202, 0x0000000202010001, 0x0000000202010101,
    0x0000000202010201, 0x0000000202020000, 0x0000000202020002, 0x0000000202020200,
    0x0000000202020202, 0x0000010000010001, 0x0000010000010100, 0x0000010000010102,
    0x0000010000020101, 0x0000010001000001, 0x0000010001000201, 0x0000010001010101,
    0x0000010001010202, 0x0000010001020100, 0x0000010001020101, 0x0000010002010001,
    0x0000010002010201, 0x0000010002020101, 0x0000010100000001, 0x0000010100000100,
    0x0000010100000101, 0x0000010100000102, 0x0000010100010101, 0x0000010100010200,
    0x0000010100010202, 0x0000010100020201, 0x0000010101000000, 0x0000010101000101,
    0x0000010101000202, 0x0000010101010000, 0x0000010101010001, 0x0000010101010100,
    0x0000010101010101, 0x0000010101010102, 0x0000010101010201, 0x0000010101020000,
    0x0000010101020002, 0x0000010101020101, 0x0000010101020200, 0x0000010101020202,
    0x0000010102000001, 0x0000010102010001, 0x0000010102010101, 0x0000010102010200,
    0x0000010102010202, 0x0000010102020001, 0x0000010102020100, 0x0000010102020101,
    0x0000010102020102, 0x0000010102020201, 0x0000010200010100, 0x0000010200010201,
    0x0000010201000001, 0x0000010201000100, 0x0000010201010000, 0x0000010201010002,
    0x0000010201010101, 0x0000010201010200, 0x0000010201020000, 0x0000010201020001,
    0x0000010201020102, 0x0000010201020201, 0x0000010202000101, 0x0000010202010001,
    0x0000010202010100, 0x0000010202010201, 0x0000020000000000, 0x0000020000000002,
    0x0000020000000200, 0x0000020000000202, 0x0000020000010101, 0x0000020000020000,
    0x0000020000020002, 0x0000020000020200, 0x0000020000020202, 0x0000020001000101,
    0x0000020001010001, 0x0000020001010102, 0x0000020001020101, 0x0000020002000000,
    0x0000020002000002, 0x0000020002000200, 0x0000020002000202, 0x0000020002010101,
    0x0000020002020000, 0x0000020002020002, 0x0000020002020200, 0x0000020002020202,
    0x0000020100000101, 0x0000020100010001, 0x0000020100010100, 0x0000020100010201,
    0x0000020100020100, 0x0000020100020101, 0x0000020101000001, 0x0000020101010000,
    0x0000020101010001, 0x0000020101010101, 0x0000020101020001, 0x0000020101020100,
    0x0000020101020201, 0x0000020102010001, 0x0000020102010100, 0x0000020102010102,
    0x0000020102010201, 0x0000020102020101, 0x0000020200000000, 0x0000020200000002,
    0x0000020200000200, 0x0000020200000202, 0x0000020200010101, 0x0000020200020000,
    0x0000020200020002, 0x0000020200020200, 0x0000020200020202, 0x0000020201000101,
    0x0000020201010001, 0x0000020201010201, 0x0000020201020001, 0x0000020201020101,
    0x0000020202000000, 0x0000020202000002, 0x0000020202000101, 0x0000020202000200,
    0x0000020202000202, 0x0000020202010101, 0x0000020202020000, 0x0000020202020002,
    0x0000020202020200, 0x0000020202020202, 0x0001000000010000, 0x0001000000010001,
    0x0001000000010100, 0x0001000000010201, 0x0001000000020100, 0x0001000000020101,
    0x0001000001000001, 0x0001000001000100, 0x0001000001010000, 0x0001000001010101,
    0x0001000001010200, 0x0001000001020001, 0x0001000001020100, 0x0001000001020101,
    0x0001000001020201, 0x0001000002010001, 0x0001000002010100, 0x0001000002010102,
    0x0001000002020001, 0x0001000002020101, 0x0001000100000001, 0x0001000100000100,
    0x0001000100000102, 0x0001000100000201, 0x0001000100010000, 0x0001000100010002,
    0x0001000100010101, 0x0001000100010200, 0x0001000100020001, 0x0001000100020100,
    0x0001000100020201, 0x0001000101000101, 0x0001000101000202, 0x0001000101010000,
    0x0001000101010001, 0x0001000101010002, 0x0001000101010100, 0x0001000101010101,
    0x0001000101010102, 0x0001000101010201, 0x0001000101020000, 0x0001000101020101,
    0x0001000102000100, 0x0001000102010002, 0x0001000102010101, 0x0001000102020001,
    0x0001000102020100, 0x0001000200010001, 0x0001000200010100, 0x0001000200010102,
    0x0001000200020101, 0x0001000201000000, 0x0001000201000102, 0x0001000201000201,
    0x0001000201010002, 0x0001000201010101, 0x0001000201010200, 0x0001000201010202,
    0x0001000201020100, 0x0001000201020102, 0x0001000202000101, 0x0001000202010001,
    0x0001000202010100, 0x0001000202010102, 0x0001000202020101, 0x0001010000000001,
    0x0001010000000102, 0x0001010000000201, 0x0001010000010100, 0x0001010000010101,
    0x0001010000010200, 0x0001010000010201, 0x0001010000020001, 0x0001010000020102,
    0x0001010001000001, 0x0001010001000101, 0x0001010001000102, 0x0001010001000200,
    0x0001010001000202, 0x0001010001010001, 0x0001010001010100, 0x0001010001010101,
    0x0001010001010102, 0x0001010001010201, 0x0001010001020002, 0x0001010001020101,
    0x0001010001020200, 0x0001010002000100, 0x0001010002000201, 0x0001010002010000,
    0x0001010002010100, 0x0001010002010101, 0x0001010002010200, 0x0001010002010201,
    0x0001010002010202, 0x0001010002020001, 0x0001010002020100, 0x0001010002020101,
    0x0001010002020201, 0x0001010100000002, 0x0001010100000101, 0x0001010100000202,
    0x0001010100010001, 0x0001010100010100, 0x0001010100010101, 0x0001010100010102,
    0x0001010100010201, 0x0001010100020000, 0x0001010100020002, 0x0001010100020101,
    0x0001010100020200, 0x0001010100020202, 0x0001010101000001, 0x0001010101000100,
    0x0001010101000101, 0x0001010101000102, 0x0001010101010001, 0x0001010101010002,
    0x0001010101010100, 0x0001010101010101, 0x0001010101010102, 0x0001010101010201,
    0x0001010101010202, 0x0001010101020001, 0x0001010101020100, 0x0001010101020101,
    0x0001010101020102, 0x0001010101020201, 0x0001010102000000, 0x0001010102000002,
    0x0001010102000100, 0x0001010102000101, 0x0001010102000200, 0x0001010102000202,
    0x0001010102010000, 0x0001010102010001, 0x0001010102010100, 0x0001010102010101,
    0x0001010102010102, 0x0001010102010201, 0x0001010102010202, 0x0001010102020000,
    0x0001010102020002, 0x0001010102020101, 0x0001010200000001, 0x0001010200000100,
    0x0001010200000101, 0x0001010200000102, 0x0001010200010101, 0x0001010200010102,
    0x0001010200010200, 0x0001010200010202, 0x0001010200020001, 0x0001010200020102,
    0x0001010201000000, 0x0001010201000002, 0x0001010201000100, 0x0001010201000101,
    0x0001010201000200, 0x0001010201000202, 0x0001010201010001, 0x0001010201010101,
    0x0001010201010102, 0x0001010201010200, 0x0001010201010201, 0x0001010201020001,
    0x0001010201020100, 0x0001010201020101, 0x0001010201020200, 0x0001010201020201,
    0x0001010201020202, 0x0001010202000102, 0x0001010202000202, 0x0001010202010002,
    0x0001010202010101, 0x0001010202020100, 0x0001010202020201, 0x0001020000010001,
    0x0001020000010102, 0x0001020000020101, 0x0001020001000001, 0x0001020001000100,
    0x0001020001000102, 0x0001020001000201, 0x0001020001010000, 0x0001020001010101,
    0x0001020001010200, 0x0001020001010202, 0x0001020001020000, 0x0001020001020001,
    0x0001020001020100, 0x0001020001020102, 0x0001020001020201, 0x0001020002000101,
    0x0001020002010001, 0x0001020002010100, 0x0001020002020101, 0x0001020100010000,
    0x0001020100010002, 0x0001020100010101, 0x0001020100010202, 0x0001020100020001,
    0x0001020100020101, 0x0001020101000002, 0x0001020101000100, 0x0001020101000101,
    0x0001020101000200, 0x0001020101010001, 0x0001020101010100, 0x0001020101010101,
    0x0001020101010102, 0x0001020101010201, 0x0001020101010202, 0x0001020101020000,
    0x0001020101020101, 0x0001020101020202, 0x0001020102000201, 0x0001020102010001,
    0x0001020102010002, 0x0001020102010101, 0x0001020102010200, 0x0001020102020001,
    0x0001020102020102, 0x0001020102020201, 0x0001020200000201, 0x0001020200010102,
    0x0001020200020100, 0x0001020200020102, 0x0001020201000100, 0x0001020201000102,
    0x0001020201000201, 0x0001020201010000, 0x0001020201010002, 0x0001020201010101,
    0x0001020201010200, 0x0001020201020001, 0x0001020201020102, 0x0001020201020201,
    0x0001020202000101, 0x0001020202010001, 0x0001020202010102, 0x0001020202010202,
    0x0002000000000000, 0x0002000000000002, 0x0002000000000200, 0x0002000000000202,
    0x0002000000010101, 0x0002000000020000, 0x0002000000020002, 0x0002000000020101,
    0x0002000000020200, 0x0002000000020202, 0x0002000001000101, 0x0002000001010001,
    0x0002000001010201, 0x0002000001020001, 0x0002000001020101, 0x0002000002000000,
    0x0002000002000002, 0x0002000002000200, 0x0002000002000202, 0x0002000002010101,
    0x0002000002020000, 0x0002000002020002, 0x0002000002020101, 0x0002000002020200,
    0x0002000002020202, 0x0002000100000101, 0x0002000100010001, 0x0002000100010100,
    0x0002000100010201, 0x0002000100020101, 0x0002000101000002, 0x0002000101000100,
    0x0002000101000201, 0x0002000101010101, 0x0002000101010200, 0x0002000101010202,
    0x0002000101020001, 0x0002000101020100, 0x0002000101020101, 0x0002000101020102,
    0x0002000102000101, 0x0002000102010000, 0x0002000102010102, 0x0002000102010201,
    0x0002000102020101, 0x0002000200000001, 0x0002000200000200, 0x0002000200000202,
    0x0002000200010001, 0x0002000200010101, 0x0002000200020000, 0x0002000200020002,
    0x0002000200020200, 0x0002000200020202, 0x0002000201000101, 0x0002000201010001,
    0x0002000201010102, 0x0002000201010201, 0x0002000201020101, 0x0002000202000001,
    0x0002000202000200, 0x0002000202000202, 0x0002000202010001, 0x0002000202010101,
    0x0002000202020000, 0x0002000202020002, 0x0002000202020200, 0x0002000202020202,
    0x0002010000000101, 0x0002010000010100, 0x0002010000010102, 0x0002010000010201,
    0x0002010000020101, 0x0002010001000100, 0x0002010001000101, 0x0002010001000102,
    0x0002010001000201, 0x0002010001010002, 0x0002010001010101, 0x0002010001010200,
    0x0002010001010202, 0x0002010001020102, 0x0002010002000101, 0x0002010002010001,
    0x0002010002010100, 0x0002010002010201, 0x0002010002020001, 0x0002010002020101,
    0x0002010100000201, 0x0002010100010101, 0x0002010100020001, 0x0002010100020201,
    0x0002010101000000, 0x0002010101000101, 0x0002010101000200, 0x0002010101010001,
    0x0002010101010100, 0x0002010101010101, 0x0002010101010201, 0x0002010101020002,
    0x0002010101020101, 0x0002010101020200, 0x0002010102000201, 0x0002010102010000,
    0x0002010102010100, 0x0002010102010101, 0x0002010102010200, 0x0002010102010202,
    0x0002010102020001, 0x0002010102020100, 0x0002010102020102, 0x0002010102020201,
    0x0002010200000101, 0x0002010200010000, 0x0002010200010002, 0x0002010200010201,
    0x0002010200020101, 0x0002010201000001, 0x0002010201000201, 0x0002010201010101,
    0x0002010201020000, 0x0002010201020001, 0x0002010201020201, 0x0002010202000100,
    0x0002010202000102, 0x0002010202010000, 0x0002010202010202, 0x0002020000000000,
    0x0002020000000002, 0x0002020000000200, 0x0002020000000202, 0x0002020000010101,
    0x0002020000020000, 0x0002020000020002, 0x0002020000020200, 0x0002020000020202,
    0x0002020001000101, 0x0002020001010001, 0x0002020001010100, 0x0002020001020101,
    0x0002020002000000, 0x0002020002000002, 0x0002020002000200, 0x0002020002000202,
    0x0002020002020000, 0x0002020002020002, 0x0002020002020200, 0x0002020002020202,
    0x0002020100000201, 0x0002020100010001, 0x0002020100010100, 0x0002020100010201,
    0x0002020100020101, 0x0002020101000102, 0x0002020101000201, 0x0002020101010002,
    0x0002020101010101, 0x0002020101020001, 0x0002020101020100, 0x0002020101020102,
    0x0002020101020201, 0x0002020102000101, 0x0002020102010000, 0x0002020102010102,
    0x0002020102010201, 0x0002020102020100, 0x0002020102020101, 0x0002020200000000,
    0x0002020200000002, 0x0002020200000200, 0x0002020200000202, 0x0002020200020000,
    0x0002020200020002, 0x0002020200020200, 0x0002020200020202, 0x0002020201000101,
    0x0002020201010001, 0x0002020201010102, 0x0002020201010201, 0x0002020201020101,
    0x0002020202000000, 0x0002020202000002, 0x0002020202000200, 0x0002020202000202,
    0x0002020202010101, 0x0002020202020000, 0x0002020202020002, 0x0002020202020200,
    0x0002020202020202, 0x0100000000000101, 0x0100000000010001, 0x0100000000010102,
    0x0100000000020101, 0x0100000001000201, 0x0100000001010002, 0x0100000001010101,
    0x0100000001010200, 0x0100000001010202, 0x0100000001020001, 0x0100000001020100,
    0x0100000001020102, 0x0100000002010100, 0x0100000002010201, 0x0100000002020001,
    0x0100000002020102, 0x0100000100000000, 0x0100000100000001, 0x0100000100000100,
    0x0100000100000102, 0x0100000100000201, 0x0100000100010002, 0x0100000100010101,
    0x0100000100010102, 0x0100000100010200, 0x0100000100010202, 0x0100000100020001,
    0x0100000100020102, 0x0100000100020201, 0x0100000101000101, 0x0100000101000200,
    0x0100000101000202, 0x0100000101010001, 0x0100000101010100, 0x0100000101010101,
    0x0100000101010102, 0x0100000101010201, 0x0100000101010202, 0x0100000101020101,
    0x0100000101020200, 0x0100000101020202, 0x0100000102000001, 0x0100000102000100,
    0x0100000102000102, 0x0100000102010000, 0x0100000102010002, 0x0100000102010101,
    0x0100000102020000, 0x0100000102020001, 0x0100000102020002, 0x0100000200000101,
    0x0100000200010001, 0x0100000200010100, 0x0100000200010102, 0x0100000200020101,
    0x0100000201000001, 0x0100000201010002, 0x0100000201010101, 0x0100000201010202,
    0x0100000201020100, 0x0100000201020201, 0x0100000202000201, 0x0100000202010100,
    0x0100000202020101, 0x0100010000000001, 0x0100010000010101, 0x0100010000010201,
    0x0100010000020201, 0x0100010001000101, 0x0100010001000200, 0x0100010001000202,
    0x0100010001010001, 0x0100010001010100, 0x0100010001010101, 0x0100010001010102,
    0x0100010001020001, 0x0100010001020002, 0x0100010001020101, 0x0100010001020200,
    0x0100010001020202, 0x0100010002000001, 0x0100010002000102, 0x0100010002000201,
    0x0100010002010000, 0x0100010002010002, 0x0100010002010101, 0x0100010002020000,
    0x0100010002020001, 0x0100010002020201, 0x0100010100000001, 0x0100010100000002,
    0x0100010100000101, 0x0100010100000202, 0x0100010100010001, 0x0100010100010100,
    0x0100010100010101, 0x0100010100010102, 0x0100010100010201, 0x0100010100020000,
    0x0100010100020101, 0x0100010100020202, 0x0100010101000001, 0x0100010101000100,
    0x0100010101000101, 0x0100010101000102, 0x0100010101000201, 0x0100010101010000,
    0x0100010101010001, 0x0100010101010100, 0x0100010101010101, 0x0100010101010102,
    0x0100010101010200, 0x0100010101010201, 0x0100010101020001, 0x0100010101020100,
    0x0100010101020101, 0x0100010101020102, 0x0100010101020201, 0x0100010102000002,
    0x0100010102000100, 0x0100010102000101, 0x0100010102000200, 0x0100010102010001,
    0x0100010102010100, 0x0100010102010101, 0x0100010102010102, 0x0100010102010201,
    0x0100010102010202, 0x0100010102020101, 0x0100010102020200, 0x0100010102020202,
    0x0100010200000001, 0x0100010200000101, 0x0100010200000201, 0x0100010200010100,
    0x0100010200010101, 0x0100010200010200, 0x0100010200010202, 0x0100010200020001,
    0x0100010200020100, 0x0100010200020201, 0x0100010201000000, 0x0100010201000002,
    0x0100010201000101, 0x0100010201000200, 0x0100010201010000, 0x0100010201010001,
    0x0100010201010002, 0x0100010201010101, 0x0100010201010102, 0x0100010201010201,
    0x0100010201020002, 0x0100010201020101, 0x0100010201020200, 0x0100010202000001,
    0x0100010202000101, 0x0100010202000202, 0x0100010202010100, 0x0100010202010101,
    0x0100010202020001, 0x0100010202020100, 0x0100010202020102, 0x0100020000000101,
    0x0100020000010001, 0x0100020000010101, 0x0100020000010202, 0x0100020000020101,
    0x0100020001000002, 0x0100020001000201, 0x0100020001010000, 0x0100020001010101,
    0x0100020001010200, 0x0100020001020001, 0x0100020001020100, 0x0100020001020102,
    0x0100020001020201, 0x0100020002000101, 0x0100020002010001, 0x0100020002010100,
    0x0100020002010102, 0x0100020002010201, 0x0100020002020101, 0x0100020100000001,
    0x0100020100000101, 0x0100020100000102, 0x0100020100000202, 0x0100020100010000,
    0x0100020100010100, 0x0100020100010101, 0x0100020100010200, 0x0100020100020001,
    0x0100020100020100, 0x0100020100020102, 0x0100020101000000, 0x0100020101000101,
    0x0100020101000202, 0x0100020101010001, 0x0100020101010002, 0x0100020101010100,
    0x0100020101010101, 0x0100020101010102, 0x0100020101010201, 0x0100020101020000,
    0x0100020101020002, 0x0100020101020101, 0x0100020101020102, 0x0100020101020202,
    0x0100020102000102, 0x0100020102000201, 0x0100020102010002, 0x0100020102010101,
    0x0100020102010102, 0x0100020102010200, 0x0100020102020001, 0x0100020102020100,
    0x0100020102020102, 0x0100020102020201, 0x0100020200010102, 0x0100020201000100,
    0x0100020201000102, 0x0100020201000201, 0x0100020201010101, 0x0100020201010200,
    0x0100020201010202, 0x0100020201020100, 0x0100020201020201, 0x0100020202010100,
    0x0100020202020101, 0x0101000000000001, 0x0101000000000100, 0x0101000000000101,
    0x0101000000000102, 0x0101000000000201, 0x0101000000010002, 0x0101000000010101,
    0x0101000000010202, 0x0101000000020001, 0x0101000000020100, 0x0101000000020201,
    0x0101000001000000, 0x0101000001000101, 0x0101000001000200, 0x0101000001010001,
    0x0101000001010100, 0x0101000001010101, 0x0101000001010102, 0x0101000001010201,
    0x0101000001020101, 0x0101000001020200, 0x0101000002000102, 0x0101000002000201,
    0x0101000002010101, 0x0101000002010200, 0x0101000002020000, 0x0101000002020001,
    0x0101000002020102, 0x0101000002020201, 0x0101000100000101, 0x0101000100000200,
    0x0101000100000201, 0x0101000100000202, 0x0101000100010001, 0x0101000100010100,
    0x0101000100010101, 0x0101000100010102, 0x0101000100010200, 0x0101000100010201,
    0x0101000100020000, 0x0101000100020101, 0x0101000100020102, 0x0101000100020200,
    0x0101000100020202, 0x0101000101000001, 0x0101000101000100, 0x0101000101000101,
    0x0101000101000102, 0x0101000101000201, 0x0101000101010000, 0x0101000101010001,
    0x0101000101010002, 0x0101000101010100, 0x0101000101010101, 0x0101000101010102,
    0x0101000101010200, 0x0101000101010201, 0x0101000101010202, 0x0101000101020001,
    0x0101000101020100, 0x0101000101020101, 0x0101000101020102, 0x0101000101020201,
    0x0101000102000002, 0x0101000102000101, 0x0101000102010001, 0x0101000102010100,
    0x0101000102010101, 0x0101000102010102, 0x0101000102010201, 0x0101000102020000,
    0x0101000102020101, 0x0101000102020202, 0x0101000200000001, 0x0101000200000102,
    0x0101000200010002, 0x0101000200010101, 0x0101000200010202, 0x0101000200020001,
    0x0101000200020100, 0x0101000201000002, 0x0101000201000101, 0x0101000201000202,
    0x0101000201010001, 0x0101000201010100, 0x0101000201010101, 0x0101000201010102,
    0x0101000201010201, 0x0101000201020002, 0x0101000201020101, 0x0101000202000101,
    0x0101000202010000, 0x0101000202010002, 0x0101000202010101, 0x0101000202010201,
    0x0101000202010202, 0x0101000202020100, 0x0101010000000100, 0x0101010000000101,
    0x0101010000010001, 0x0101010000010100, 0x0101010000010101, 0x0101010000010102,
    0x0101010000010200, 0x0101010000010201, 0x0101010000020001, 0x0101010000020101,
    0x0101010000020200, 0x0101010000020202, 0x0101010001000001, 0x0101010001000100,
    0x0101010001000101, 0x0101010001000102, 0x0101010001000201, 0x0101010001000202,
    0x0101010001010000, 0x0101010001010001, 0x0101010001010100, 0x0101010001010101,
    0x0101010001010102, 0x0101010001010200, 0x0101010001010201, 0x0101010001010202,
    0x0101010001020001, 0x0101010001020002, 0x0101010001020100, 0x0101010001020101,
    0x0101010001020102, 0x0101010001020201, 0x0101010002000000, 0x0101010002000200,
    0x0101010002000202, 0x0101010002010001, 0x0101010002010100, 0x0101010002010101,
    0x0101010002010102, 0x0101010002010201, 0x0101010002020001, 0x0101010002020100,
    0x0101010002020101, 0x0101010002020202, 0x0101010100000001, 0x0101010100000002,
    0x0101010100000100, 0x0101010100000101, 0x0101010100000102, 0x0101010100000201,
    0x0101010100010000, 0x0101010100010001, 0x0101010100010002, 0x0101010100010100,
    0x0101010100010101, 0x0101010100010102, 0x0101010100010201, 0x0101010100010202,
    0x0101010100020001, 0x0101010100020100, 0x0101010100020101, 0x0101010100020102,
    0x0101010100020201, 0x0101010101000000, 0x0101010101000001, 0x0101010101000002,
    0x0101010101000100, 0x0101010101000101, 0x0101010101000102, 0x0101010101000200,
    0x0101010101000201, 0x0101010101010000, 0x0101010101010001, 0x0101010101010002,
    0x0101010101010100, 0x0101010101010101, 0x0101010101010102, 0x0101010101010200,
    0x0101010101010201, 0x0101010101010202, 0x0101010101020000, 0x0101010101020001,
    0x0101010101020100, 0x0101010101020101, 0x0101010101020102, 0x0101010101020200,
    0x0101010101020201, 0x0101010101020202, 0x0101010102000001, 0x0101010102000100,
    0x0101010102000101, 0x0101010102000201, 0x0101010102000202, 0x0101010102010000,
    0x0101010102010001, 0x0101010102010100, 0x0101010102010101, 0x0101010102010102,
    0x0101010102010200, 0x0101010102010201, 0x0101010102020001, 0x0101010102020100,
    0x0101010102020101, 0x0101010102020102, 0x0101010102020201, 0x0101010200000000,
    0x0101010200000001, 0x0101010200000002, 0x0101010200000100, 0x0101010200000102,
    0x0101010200000200, 0x0101010200000201, 0x0101010200010001, 0x0101010200010100,
    0x0101010200010101, 0x0101010200010200, 0x0101010200010201, 0x0101010200020000,
    0x0101010200020001, 0x0101010200020002, 0x0101010200020100, 0x0101010200020101,
    0x0101010200020102, 0x0101010200020200, 0x0101010200020201, 0x0101010201000001,
    0x0101010201000101, 0x0101010201000102, 0x0101010201000200, 0x0101010201000201,
    0x0101010201000202, 0x0101010201010000, 0x0101010201010001, 0x0101010201010002,
    0x0101010201010100, 0x0101010201010101, 0x0101010201010102, 0x0101010201010200,
    0x0101010201010201, 0x0101010201010202, 0x0101010201020001, 0x0101010201020100,
    0x0101010201020101, 0x0101010201020201, 0x0101010202000002, 0x0101010202000101,
    0x0101010202000102, 0x0101010202000200, 0x0101010202000201, 0x0101010202000202,
    0x0101010202010001, 0x0101010202010101, 0x0101010202010202, 0x0101010202020002,
    0x0101010202020101, 0x0101010202020102, 0x0101010202020200, 0x0101010202020201,
    0x0101020000000100, 0x0101020000000101, 0x0101020000000102, 0x0101020000000201,
    0x0101020000010000, 0x0101020000010101, 0x0101020000010200, 0x0101020000020001,
    0x0101020000020202, 0x0101020001000101, 0x0101020001000200, 0x0101020001000202,
    0x0101020001010001, 0x0101020001010100, 0x0101020001010101, 0x0101020001010102,
    0x0101020001010200, 0x0101020001010201, 0x0101020001020000, 0x0101020001020002,
    0x0101020001020100, 0x0101020001020101, 0x0101020002000002, 0x0101020002000201,
    0x0101020002010000, 0x0101020002010002, 0x0101020002010101, 0x0101020002010200,
    0x0101020002020001, 0x0101020002020201, 0x0101020100000001, 0x0101020100000002,
    0x0101020100000101, 0x0101020100000202, 0x0101020100010001, 0x0101020100010100,
    0x0101020100010101, 0x0101020100010102, 0x0101020100010201, 0x0101020100020101,
    0x0101020101000001, 0x0101020101000100, 0x0101020101000101, 0x0101020101000102,
    0x0101020101000201, 0x0101020101010000, 0x0101020101010001, 0x0101020101010002,
    0x0101020101010100, 0x0101020101010101, 0x0101020101010102, 0x0101020101010200,
    0x0101020101010201, 0x0101020101010202, 0x0101020101020001, 0x0101020101020100,
    0x0101020101020101, 0x0101020101020102, 0x0101020101020201, 0x0101020102000001,
    0x0101020102000101, 0x0101020102000201, 0x0101020102010001, 0x0101020102010100,
    0x0101020102010101, 0x0101020102010102, 0x0101020102010200, 0x0101020102010201,
    0x0101020102020101, 0x0101020200000100, 0x0101020200000200, 0x0101020200010101,
    0x0101020200010202, 0x0101020200020000, 0x0101020200020101, 0x0101020200020102,
    0x0101020200020201, 0x0101020201000101, 0x0101020201000200, 0x0101020201000201,
    0x0101020201010001, 0x0101020201010101, 0x0101020201010102, 0x0101020201010200,
    0x0101020201010201, 0x0101020201020002, 0x0101020201020101, 0x0101020201020200,
    0x0101020201020202, 0x0101020202000001, 0x0101020202000202, 0x0101020202010002,
    0x0101020202010101, 0x0101020202010102, 0x0101020202010200, 0x0101020202010202,
    0x0101020202020001, 0x0102000000000101, 0x0102000000010100, 0x0102000000010102,
    0x0102000000010201, 0x0102000000020101, 0x0102000001000100, 0x0102000001010000,
    0x0102000001010101, 0x0102000001010102, 0x0102000001010200, 0x0102000001010202,
    0x0102000001020001, 0x0102000001020100, 0x0102000001020102, 0x0102000001020201,
    0x0102000002000001, 0x0102000002010102, 0x0102000002020101, 0x0102000100000001,
    0x0102000100000100, 0x0102000100000102, 0x0102000100000201, 0x0102000100010002,
    0x0102000100010101, 0x0102000100020001, 0x0102000100020002, 0x0102000100020102,
    0x0102000100020201, 0x0102000101000101, 0x0102000101000201, 0x0102000101010001,
    0x0102000101010101, 0x0102000101010102, 0x0102000101010201, 0x0102000101020101,
    0x0102000101020102, 0x0102000101020202, 0x0102000102000100, 0x0102000102000202,
    0x0102000102010002, 0x0102000102010101, 0x0102000102020001, 0x0102000102020102,
    0x0102000102020201, 0x0102000200010001, 0x0102000200010102, 0x0102000200010201,
    0x0102000201000000, 0x0102000201000001, 0x0102000201000102, 0x0102000201010101,
    0x0102000201010102, 0x0102000201010200, 0x0102000201020000, 0x0102000202000101,
    0x0102000202010001, 0x0102000202010102, 0x0102000202020101, 0x0102010000010001,
    0x0102010000010002, 0x0102010000010101, 0x0102010000010102, 0x0102010000010202,
    0x0102010000020001, 0x0102010000020102, 0x0102010000020201, 0x0102010001000000,
    0x0102010001000002, 0x0102010001000101, 0x0102010001000200, 0x0102010001000202,
    0x0102010001010001, 0x0102010001010100, 0x0102010001010101, 0x0102010001010102,
    0x0102010001010201, 0x0102010001010202, 0x0102010001020000, 0x0102010001020002,
    0x0102010001020101, 0x0102010002000100, 0x0102010002000101, 0x0102010002000201,
    0x0102010002010000, 0x0102010002010002, 0x0102010002010100, 0x0102010002010101,
    0x0102010002010102, 0x0102010002010200, 0x0102010002010202, 0x0102010002020001,
    0x0102010002020100, 0x0102010002020201, 0x0102010100000101, 0x0102010100000200,
    0x0102010100000202, 0x0102010100010001, 0x0102010100010101, 0x0102010100010102,
    0x0102010100010201, 0x0102010101000100, 0x0102010101000101, 0x0102010101000102,
    0x0102010101000201, 0x0102010101010000, 0x0102010101010001, 0x0102010101010100,
    0x0102010101010101, 0x0102010101010102, 0x0102010101010201, 0x0102010101020001,
    0x0102010101020100, 0x0102010101020101, 0x0102010101020102, 0x0102010101020201,
    0x0102010102000102, 0x0102010102000201, 0x0102010102000202, 0x0102010102010001,
    0x0102010102010101, 0x0102010102010102, 0x0102010102010201, 0x0102010102010202,
    0x0102010102020002, 0x0102010102020101, 0x0102010102020102, 0x0102010102020200,
    0x0102010200000002, 0x0102010200000201, 0x0102010200010101, 0x0102010200020000,
    0x0102010200020102, 0x0102010200020200, 0x0102010200020201, 0x0102010201000000,
    0x0102010201000101, 0x0102010201000200, 0x0102010201000202, 0x0102010201010001,
    0x0102010201010100, 0x0102010201010101, 0x0102010201010102, 0x0102010201010200,
    0x0102010201010202, 0x0102010201020000, 0x0102010201020101, 0x0102010201020200,
    0x0102010202000000, 0x0102010202000002, 0x0102010202000101, 0x0102010202000202,
    0x0102010202010100, 0x0102010202010102, 0x0102010202010200, 0x0102010202010201,
    0x0102010202020000, 0x0102010202020100, 0x0102010202020102, 0x0102010202020202,
    0x0102020000010102, 0x0102020000010201, 0x0102020000020101, 0x0102020001000001,
    0x0102020001010002, 0x0102020001010101, 0x0102020001010202, 0x0102020001020001,
    0x0102020001020201, 0x0102020002000101, 0x0102020002010001, 0x0102020002010200,
    0x0102020002020102, 0x0102020100000001, 0x0102020100000100, 0x0102020100010000,
    0x0102020100010101, 0x0102020100020001, 0x0102020100020100, 0x0102020100020102,
    0x0102020100020201, 0x0102020101000000, 0x0102020101000001, 0x0102020101000101,
    0x0102020101000102, 0x0102020101000200, 0x0102020101010001, 0x0102020101010100,
    0x0102020101010101, 0x0102020101010102, 0x0102020101010201, 0x0102020101020000,
    0x0102020101020101, 0x0102020101020202, 0x0102020102000002, 0x0102020102000100,
    0x0102020102000202, 0x0102020102010101, 0x0102020102020001, 0x0102020102020100,
    0x0102020102020101, 0x0102020102020201, 0x0102020200010001, 0x0102020200010102,
    0x0102020200010200, 0x0102020201000001, 0x0102020201000100, 0x0102020201000201,
    0x0102020201010000, 0x0102020201010101, 0x0102020201010200, 0x0102020201010202,
    0x0102020201020100, 0x0102020201020101, 0x0102020201020201, 0x0102020202000102,
    0x0102020202010100, 0x0102020202010200, 0x0102020202010202, 0x0102020202020102,
    0x0200000000000000, 0x0200000000000002, 0x0200000000000200, 0x0200000000000202,
    0x0200000000020000, 0x0200000000020002, 0x0200000000020200, 0x0200000000020202,
    0x0200000001000101, 0x0200000001010000, 0x0200000001010001, 0x0200000001010100,
    0x0200000001010102, 0x0200000001010201, 0x0200000001020101, 0x0200000002000000,
    0x0200000002000002, 0x0200000002000200, 0x0200000002000202, 0x0200000002010101,
    0x0200000002020000, 0x0200000002020002, 0x0200000002020200, 0x0200000002020202,
    0x0200000100000101, 0x0200000100010001, 0x0200000100010100, 0x0200000100010102,
    0x0200000100010201, 0x0200000100020101, 0x0200000101000001, 0x0200000101000100,
    0x0200000101000201, 0x0200000101010000, 0x0200000101010002, 0x0200000101010101,
    0x0200000101010102, 0x0200000101010200, 0x0200000101010201, 0x0200000101020100,
    0x0200000101020102, 0x0200000101020201, 0x0200000102000101, 0x0200000102000201,
    0x0200000102010100, 0x0200000102010102, 0x0200000102010201, 0x0200000102020101,
    0x0200000200000000, 0x0200000200000002, 0x0200000200000200, 0x0200000200000202,
    0x0200000200010101, 0x0200000200020000, 0x0200000200020002, 0x0200000200020200,
    0x0200000200020202, 0x0200000201010001, 0x0200000201010100, 0x0200000201010201,
    0x0200000201020101, 0x0200000202000000, 0x0200000202000002, 0x0200000202000200,
    0x0200000202000202, 0x0200000202010101, 0x0200000202020000, 0x0200000202020002,
    0x0200000202020200, 0x0200000202020202, 0x0200010000010100, 0x0200010000010201,
    0x0200010001000001, 0x0200010001000100, 0x0200010001010001, 0x0200010001010101,
    0x0200010001010202, 0x0200010001020001, 0x0200010001020100, 0x0200010001020201,
    0x0200010002010100, 0x0200010002010201, 0x0200010100000001, 0x0200010100000201,
    0x0200010100010002, 0x0200010100010101, 0x0200010100010202, 0x0200010100020102,
    0x0200010100020201, 0x0200010101000000, 0x0200010101000001, 0x0200010101000101,
    0x0200010101000200, 0x0200010101010001, 0x0200010101010100, 0x0200010101010101,
    0x0200010101010102, 0x0200010101010201, 0x0200010101010202, 0x0200010101020101,
    0x0200010101020102, 0x0200010101020200, 0x0200010101020202, 0x0200010102000001,
    0x0200010102000100, 0x0200010102000102, 0x0200010102000201, 0x0200010102010000,
    0x0200010102010002, 0x0200010102010101, 0x0200010102010200, 0x0200010102020102,
    0x0200010200010001, 0x0200010200010102, 0x0200010200010201, 0x0200010200020101,
    0x0200010201000001, 0x0200010201000100, 0x0200010201000201, 0x0200010201000202,
    0x0200010201010000, 0x0200010201010101, 0x0200010201010201, 0x0200010201010202,
    0x0200010201020001, 0x0200010201020102, 0x0200010201020202, 0x0200010202000101,
    0x0200010202010001, 0x0200010202010202, 0x0200010202020100, 0x0200020000000000,
    0x0200020000000002, 0x0200020000000200, 0x0200020000000202, 0x0200020000010101,
    0x0200020000020000, 0x0200020000020002, 0x0200020000020200, 0x0200020000020202,
    0x0200020001000001, 0x0200020001000101, 0x0200020001010001, 0x0200020001010100,
    0x0200020001010201, 0x0200020001020101, 0x0200020001020201, 0x0200020002000000,
    0x0200020002000002, 0x0200020002000200, 0x0200020002000202, 0x0200020002010101,
    0x0200020002020000, 0x0200020002020002, 0x0200020002020200, 0x0200020002020202,
    0x0200020100000101, 0x0200020100000102, 0x0200020100010001, 0x0200020100010100,
    0x0200020100010102, 0x0200020100020101, 0x0200020101000001, 0x0200020101000100,
    0x0200020101000102, 0x0200020101000201, 0x0200020101010000, 0x0200020101010002,
    0x0200020101010101, 0x0200020101010202, 0x0200020101020001, 0x0200020101020100,
    0x0200020102000101, 0x0200020102010102, 0x0200020102010201, 0x0200020102020101,
    0x0200020200000000, 0x0200020200000002, 0x0200020200000200, 0x0200020200000202,
    0x0200020200010101, 0x0200020200020000, 0x0200020200020002, 0x0200020200020200,
    0x0200020200020202, 0x0200020201000101, 0x0200020201010001, 0x0200020201010100,
    0x0200020201010102, 0x0200020202000000, 0x0200020202000002, 0x0200020202000200,
    0x0200020202000202, 0x0200020202010101, 0x0200020202020000, 0x0200020202020002,
    0x0200020202020200, 0x0200020202020202, 0x0201000000000101, 0x0201000000010001,
    0x0201000000010102, 0x0201000000010200, 0x0201000000010201, 0x0201000000020101,
    0x0201000001000001, 0x0201000001000102, 0x0201000001000201, 0x0201000001010101,
    0x0201000001010200, 0x0201000001010202, 0x0201000001020201, 0x0201000001020202,
    0x0201000002000101, 0x0201000002010001, 0x0201000002010100, 0x0201000002010102,
    0x0201000002010201, 0x0201000002020101, 0x0201000100000001, 0x0201000100000100,
    0x0201000100000102, 0x0201000100000201, 0x0201000100010000, 0x0201000100010101,
    0x0201000100010200, 0x0201000100010202, 0x0201000100020001, 0x0201000100020100,
    0x0201000100020102, 0x0201000100020201, 0x0201000101000000, 0x0201000101000101,
    0x0201000101010000, 0x0201000101010001, 0x0201000101010100, 0x0201000101010101,
    0x0201000101010102, 0x0201000101010201, 0x0201000101020002, 0x0201000101020101,
    0x0201000102000100, 0x0201000102000102, 0x0201000102010002, 0x0201000102010101,
    0x0201000102010200, 0x0201000102020001, 0x0201000102020100, 0x0201000102020102,
    0x0201000102020201, 0x0201000200000101, 0x0201000200010001, 0x0201000200010100,
    0x0201000200010201, 0x0201000200020101, 0x0201000201000100, 0x0201000201000102,
    0x0201000201000201, 0x0201000201010000, 0x0201000201010002, 0x0201000201010101,
    0x0201000201010200, 0x0201000201020102, 0x0201000201020201, 0x0201000202000101,
    0x0201000202010100, 0x0201000202010102, 0x0201000202020201, 0x0201010000000001,
    0x0201010000000100, 0x0201010000000102, 0x0201010000010000, 0x0201010000010101,
    0x0201010000010200, 0x0201010000020102, 0x0201010001000000, 0x0201010001000202,
    0x0201010001010001, 0x0201010001010100, 0x0201010001010101, 0x0201010001010102,
    0x0201010001010200, 0x0201010001010201, 0x0201010001020000, 0x0201010001020001,
    0x0201010001020002, 0x0201010001020101, 0x0201010002000100, 0x0201010002000102,
    0x0201010002010002, 0x0201010002010100, 0x0201010002010101, 0x0201010002010200,
    0x0201010002020001, 0x0201010002020201, 0x0201010100000000, 0x0201010100000101,
    0x0201010100000200, 0x0201010100000202, 0x0201010100010000, 0x0201010100010001,
    0x0201010100010100, 0x0201010100010101, 0x0201010100010102, 0x0201010100010201,
    0x0201010100020001, 0x0201010100020101, 0x0201010100020201, 0x0201010100020202,
    0x0201010101000001, 0x0201010101000100, 0x0201010101000101, 0x0201010101000102,
    0x0201010101000201, 0x0201010101010000, 0x0201010101010001, 0x0201010101010002,
    0x0201010101010100, 0x0201010101010101, 0x0201010101010102, 0x0201010101010200,
    0x0201010101010201, 0x0201010101010202, 0x0201010101020001, 0x0201010101020100,
    0x0201010101020101, 0x0201010101020102, 0x0201010101020201, 0x0201010102000001,
    0x0201010102000101, 0x0201010102000200, 0x0201010102010001, 0x0201010102010002,
    0x0201010102010100, 0x0201010102010101, 0x0201010102010102, 0x0201010102010201,
    0x0201010102010202, 0x0201010102020000, 0x0201010102020002, 0x0201010102020101,
    0x0201010102020200, 0x0201010102020202, 0x0201010200000001, 0x0201010200000100,
    0x0201010200010000, 0x0201010200010101, 0x0201010200010201, 0x0201010200020000,
    0x0201010200020102, 0x0201010200020201, 0x0201010201000101, 0x0201010201000200,
    0x0201010201000201, 0x0201010201010001, 0x0201010201010002, 0x0201010201010101,
    0x0201010201010102, 0x0201010201010201, 0x0201010201020101, 0x0201010201020200,
    0x0201010202000002, 0x0201010202000100, 0x0201010202000201, 0x0201010202000202,
    0x0201010202010002, 0x0201010202010100, 0x0201010202010101, 0x0201010202020100,
    0x0201010202020102, 0x0201010202020201, 0x0201020000000101, 0x0201020000010102,
    0x0201020000010201, 0x0201020000020101, 0x0201020001000001, 0x0201020001000102,
    0x0201020001010000, 0x0201020001010002, 0x0201020001010101, 0x0201020001010102,
    0x0201020001010202, 0x0201020001020100, 0x0201020001020101, 0x0201020002000101,
    0x0201020002010001, 0x0201020002010102, 0x0201020002010201, 0x0201020002020101,
    0x0201020100000100, 0x0201020100000102, 0x0201020100000201, 0x0201020100010000,
    0x0201020100010002, 0x0201020100010101, 0x0201020100010200, 0x0201020100010202,
    0x0201020100020000, 0x0201020100020001, 0x0201020100020100, 0x0201020100020102,
    0x0201020101000000, 0x0201020101000002, 0x0201020101000101, 0x0201020101000200,
    0x0201020101000202, 0x0201020101010001, 0x0201020101010100, 0x0201020101010101,
    0x0201020101010102, 0x0201020101010201, 0x0201020101020002, 0x0201020101020101,
    0x0201020101020102, 0x0201020101020202, 0x0201020102000001, 0x0201020102000100,
    0x0201020102010000, 0x0201020102010002, 0x0201020102010101, 0x0201020102010202,
    0x0201020102020001, 0x0201020102020102, 0x0201020200000101, 0x0201020200010101,
    0x0201020200020101, 0x0201020201000100, 0x0201020201000102, 0x0201020201000201,
    0x0201020201010000, 0x0201020201010101, 0x0201020201010200, 0x0201020201020001,
    0x0201020202000101, 0x0201020202010001, 0x0201020202010100, 0x0201020202010101,
    0x0201020202010102, 0x0202000000000000, 0x0202000000000002, 0x0202000000000200,
    0x0202000000000202, 0x0202000000010101, 0x0202000000020000, 0x0202000000020002,
    0x0202000000020200, 0x0202000000020202, 0x0202000001000101, 0x0202000001010001,
    0x0202000001010100, 0x0202000001010102, 0x0202000001010201, 0x0202000002000000,
    0x0202000002000002, 0x0202000002000200, 0x0202000002000202, 0x0202000002010101,
    0x0202000002020000, 0x0202000002020002, 0x0202000002020200, 0x0202000002020202,
    0x0202000100000101, 0x0202000100000201, 0x0202000100010001, 0x0202000100010100,
    0x0202000100010102, 0x0202000100010201, 0x0202000100010202, 0x0202000101000102,
    0x0202000101000201, 0x0202000101010001, 0x0202000101010101, 0x0202000101010200,
    0x0202000101010202, 0x0202000101020001, 0x0202000101020100, 0x0202000102000101,
    0x0202000102010000, 0x0202000102010002, 0x0202000102010102, 0x0202000102010201,
    0x0202000200000002, 0x0202000200000200, 0x0202000200000202, 0x0202000200010000,
    0x0202000200010201, 0x0202000200020002, 0x0202000200020200, 0x0202000200020202,
    0x0202000201000101, 0x0202000201010001, 0x0202000201010102, 0x0202000201010201,
    0x0202000201020101, 0x0202000202000000, 0x0202000202000002, 0x0202000202000200,
    0x0202000202000202, 0x0202000202010101, 0x0202000202020000, 0x0202000202020002,
    0x0202000202020200, 0x0202000202020202, 0x0202010000010201, 0x0202010000020101,
    0x0202010001000001, 0x0202010001000100, 0x0202010001010000, 0x0202010001010100,
    0x0202010001010101, 0x0202010001010200, 0x0202010001010202, 0x0202010001020001,
    0x0202010001020101, 0x0202010001020102, 0x0202010001020200, 0x0202010001020201,
    0x0202010002000101, 0x0202010100000102, 0x0202010100000201, 0x0202010100010000,
    0x0202010100010002, 0x0202010100010101, 0x0202010100010200, 0x0202010100020102,
    0x0202010100020201, 0x0202010101000002, 0x0202010101000101, 0x0202010101010001,
    0x0202010101010100, 0x0202010101010101, 0x0202010101010102, 0x0202010101010201,
    0x0202010101020101, 0x0202010101020202, 0x0202010102000001, 0x0202010102000100,
    0x0202010102000101, 0x0202010102000102, 0x0202010102000201, 0x0202010102010002,
    0x0202010102010101, 0x0202010102010200, 0x0202010200000101, 0x0202010200010001,
    0x0202010200010102, 0x0202010200010202, 0x0202010200020001, 0x0202010200020101,
    0x0202010201000100, 0x0202010201000102, 0x0202010201000202, 0x0202010201010002,
    0x0202010201010101, 0x0202010201010102, 0x0202010201010200, 0x0202010201020000,
    0x0202010201020002, 0x0202010202000102, 0x0202010202010000, 0x0202010202010101,
    0x0202010202010102, 0x0202010202010201, 0x0202010202020001, 0x0202010202020100,
    0x0202010202020102, 0x0202020000000000, 0x0202020000000002, 0x0202020000000200,
    0x0202020000000202, 0x0202020000020000, 0x0202020000020002, 0x0202020000020200,
    0x0202020000020202, 0x0202020001010001, 0x0202020001010100, 0x0202020001010102,
    0x0202020001010201, 0x0202020002000000, 0x0202020002000002, 0x0202020002000200,
    0x0202020002000202, 0x0202020002010101, 0x0202020002020000, 0x0202020002020002,
    0x0202020002020200, 0x0202020002020202, 0x0202020100000101, 0x0202020100010100,
    0x0202020100010201, 0x0202020100020001, 0x0202020100020101, 0x0202020101000001,
    0x0202020101010000, 0x0202020101010101, 0x0202020101010202, 0x0202020101020001,
    0x0202020101020102, 0x0202020101020201, 0x0202020102010000, 0x0202020102010102,
    0x0202020200000000, 0x0202020200000002, 0x0202020200000200, 0x0202020200000202,
    0x0202020200020000, 0x0202020200020002, 0x0202020200020200, 0x0202020200020202,
    0x0202020201010001, 0x0202020201010100, 0x0202020201010102, 0x0202020202000000,
    0x0202020202000002, 0x0202020202000200, 0x0202020202000202, 0x0202020202010101,
    0x0202020202020000, 0x0202020202020002, 0x0202020202020200, 0x0202020202020202,
};
#else
static const uint32_t iq1s_grid_us[2048] = {
    0x00000000, 0x00000002, 0x00000101, 0x00000200, 0x00000202, 0x00010001, 0x00010101, 0x00020000,
    0x00020002, 0x00020200, 0x00020202, 0x01000101, 0x01010001, 0x01010100, 0x01010102, 0x01020101,
    0x02000000, 0x02000002, 0x02000200, 0x02000202, 0x02010101, 0x02020000, 0x02020002, 0x02020200,
    0x02020202, 0x00000110, 0x00000111, 0x00010011, 0x00010110, 0x00010112, 0x00010211, 0x00010212,
    0x00020111, 0x01000011, 0x01000112, 0x01000211, 0x01010012, 0x01010111, 0x01010212, 0x01020011,
    0x01020110, 0x01020112, 0x01020210, 0x02000111, 0x02010011, 0x02010110, 0x02010112, 0x02020111,
    0x00000020, 0x00000022, 0x00000220, 0x00000222, 0x00010121, 0x00020020, 0x00020022, 0x00020220,
    0x00020222, 0x01000121, 0x01010021, 0x01010221, 0x01020120, 0x01020221, 0x02000020, 0x02000022,
    0x02000220, 0x02000222, 0x02010021, 0x02010121, 0x02010221, 0x02020020, 0x02020022, 0x02020220,
    0x02020222, 0x00011001, 0x00011100, 0x00011102, 0x00021101, 0x01001001, 0x01001201, 0x01011101,
    0x01011202, 0x01021100, 0x01021101, 0x02011001, 0x02011201, 0x02021101, 0x00001011, 0x00001110,
    0x00001111, 0x00001112, 0x00011111, 0x00011210, 0x00011212, 0x00021211, 0x01001010, 0x01001111,
    0x01001212, 0x01011010, 0x01011011, 0x01011110, 0x01011111, 0x01011112, 0x01011211, 0x01021010,
    0x01021012, 0x01021111, 0x01021210, 0x01021212, 0x02001011, 0x02011011, 0x02011111, 0x02011210,
    0x02011212, 0x02021011, 0x02021110, 0x02021111, 0x02021112, 0x02021211, 0x00011120, 0x00011221,
    0x01001021, 0x01001120, 0x01011020, 0x01011022, 0x01011121, 0x01011220, 0x01021020, 0x01021021,
    0x01021122, 0x01021221, 0x02001121, 0x02011021, 0x02011120, 0x02011221, 0x00002000, 0x00002002,
    0x00002200, 0x00002202, 0x00012101, 0x00022000, 0x00022002, 0x00022200, 0x00022202, 0x01002101,
    0x01012001, 0x01012102, 0x01022101, 0x02002000, 0x02002002, 0x02002200, 0x02002202, 0x02012101,
    0x02022000, 0x02022002, 0x02022200, 0x02022202, 0x00002111, 0x00012011, 0x00012110, 0x00012211,
    0x00022110, 0x00022111, 0x01002011, 0x01012010, 0x01012011, 0x01012111, 0x01022011, 0x01022110,
    0x01022211, 0x02012011, 0x02012110, 0x02012112, 0x02012211, 0x02022111, 0x00002020, 0x00002022,
    0x00002220, 0x00002222, 0x00012121, 0x00022020, 0x00022022, 0x00022220, 0x00022222, 0x01002121,
    0x01012021, 0x01012221, 0x01022021, 0x01022121, 0x02002020, 0x02002022, 0x02002121, 0x02002220,
    0x02002222, 0x02012121, 0x02022020, 0x02022022, 0x02022220, 0x02022222, 0x00110000, 0x00110001,
    0x00110100, 0x00110201, 0x00120100, 0x00120101, 0x01100001, 0x01100100, 0x01110000, 0x01110101,
    0x01110200, 0x01120001, 0x01120100, 0x01120101, 0x01120201, 0x02110001, 0x02110100, 0x02110102,
    0x02120001, 0x02120101, 0x00100011, 0x00100110, 0x00100112, 0x00100211, 0x00110010, 0x00110012,
    0x00110111, 0x00110210, 0x00120011, 0x00120110, 0x00120211, 0x01100111, 0x01100212, 0x01110010,
    0x01110011, 0x01110012, 0x01110110, 0x01110111, 0x01110112, 0x01110211, 0x01120010, 0x01120111,
    0x02100110, 0x02110012, 0x02110111, 0x02120011, 0x02120110, 0x00110021, 0x00110120, 0x00110122,
    0x00120121, 0x01100020, 0x01100122, 0x01100221, 0x01110022, 0x01110121, 0x01110220, 0x01110222,
    0x01120120, 0x01120122, 0x02100121, 0x02110021, 0x02110120, 0x02110122, 0x02120121, 0x00101001,
    0x00101102, 0x00101201, 0x00111100, 0x00111101, 0x00111200, 0x00111201, 0x00121001, 0x00121102,
    0x01101001, 0x01101101, 0x01101102, 0x01101200, 0x01101202, 0x01111001, 0x01111100, 0x01111101,
    0x01111102, 0x01111201, 0x01121002, 0x01121101, 0x01121200, 0x02101100, 0x02101201, 0x02111000,
    0x02111100, 0x02111101, 0x02111200, 0x02111201, 0x02111202, 0x02121001, 0x02121100, 0x02121101,
    0x02121201, 0x00101012, 0x00101111, 0x00101212, 0x00111011, 0x00111110, 0x00111111, 0x00111112,
    0x00111211, 0x00121010, 0x00121012, 0x00121111, 0x00121210, 0x00121212, 0x01101011, 0x01101110,
    0x01101111, 0x01101112, 0x01111011, 0x01111012, 0x01111110, 0x01111111, 0x01111112, 0x01111211,
    0x01111212, 0x01121011, 0x01121110, 0x01121111, 0x01121112, 0x01121211, 0x02101010, 0x02101012,
    0x02101110, 0x02101111, 0x02101210, 0x02101212, 0x02111010, 0x02111011, 0x02111110, 0x02111111,
    0x02111112, 0x02111211, 0x02111212, 0x02121010, 0x02121012, 0x02121111, 0x00101021, 0x00101120,
    0x00101121, 0x00101122, 0x00111121, 0x00111122, 0x00111220, 0x00111222, 0x00121021, 0x00121122,
    0x01101020, 0x01101022, 0x01101120, 0x01101121, 0x01101220, 0x01101222, 0x01111021, 0x01111121,
    0x01111122, 0x01111220, 0x01111221, 0x01121021, 0x01121120, 0x01121121, 0x01121220, 0x01121221,
    0x01121222, 0x02101122, 0x02101222, 0x02111022, 0x02111121, 0x02121120, 0x02121221, 0x00112001,
    0x00112102, 0x00122101, 0x01102001, 0x01102100, 0x01102102, 0x01102201, 0x01112000, 0x01112101,
    0x01112200, 0x01112202, 0x01122000, 0x01122001, 0x01122100, 0x01122102, 0x01122201, 0x02102101,
    0x02112001, 0x02112100, 0x02122101, 0x00112010, 0x00112012, 0x00112111, 0x00112212, 0x00122011,
    0x00122111, 0x01102012, 0x01102110, 0x01102111, 0x01102210, 0x01112011, 0x01112110, 0x01112111,
    0x01112112, 0x01112211, 0x01112212, 0x01122010, 0x01122111, 0x01122212, 0x02102211, 0x02112011,
    0x02112012, 0x02112111, 0x02112210, 0x02122011, 0x02122112, 0x02122211, 0x00102221, 0x00112122,
    0x00122120, 0x00122122, 0x01102120, 0x01102122, 0x01102221, 0x01112020, 0x01112022, 0x01112121,
    0x01112220, 0x01122021, 0x01122122, 0x01122221, 0x02102121, 0x02112021, 0x02112122, 0x02112222,
    0x00200000, 0x00200002, 0x00200200, 0x00200202, 0x00210101, 0x00220000, 0x00220002, 0x00220101,
    0x00220200, 0x00220202, 0x01200101, 0x01210001, 0x01210201, 0x01220001, 0x01220101, 0x02200000,
    0x02200002, 0x02200200, 0x02200202, 0x02210101, 0x02220000, 0x02220002, 0x02220101, 0x02220200,
    0x02220202, 0x00200111, 0x00210011, 0x00210110, 0x00210211, 0x00220111, 0x01200012, 0x01200110,
    0x01200211, 0x01210111, 0x01210210, 0x01210212, 0x01220011, 0x01220110, 0x01220111, 0x01220112,
    0x02200111, 0x02210010, 0x02210112, 0x02210211, 0x02220111, 0x00200021, 0x00200220, 0x00200222,
    0x00210021, 0x00210121, 0x00220020, 0x00220022, 0x00220220, 0x00220222, 0x01200121, 0x01210021,
    0x01210122, 0x01210221, 0x01220121, 0x02200021, 0x02200220, 0x02200222, 0x02210021, 0x02210121,
    0x02220020, 0x02220022, 0x02220220, 0x02220222, 0x00201101, 0x00211100, 0x00211102, 0x00211201,
    0x00221101, 0x01201100, 0x01201101, 0x01201102, 0x01201201, 0x01211002, 0x01211101, 0x01211200,
    0x01211202, 0x01221102, 0x02201101, 0x02211001, 0x02211100, 0x02211201, 0x02221001, 0x02221101,
    0x00201211, 0x00211111, 0x00221011, 0x00221211, 0x01201010, 0x01201111, 0x01201210, 0x01211011,
    0x01211110, 0x01211111, 0x01211211, 0x01221012, 0x01221111, 0x01221210, 0x02201211, 0x02211010,
    0x02211110, 0x02211111, 0x02211210, 0x02211212, 0x02221011, 0x02221110, 0x02221112, 0x02221211,
    0x00201121, 0x00211020, 0x00211022, 0x00211221, 0x00221121, 0x01201021, 0x01201221, 0x01211121,
    0x01221020, 0x01221021, 0x01221221, 0x02201120, 0x02201122, 0x02211020, 0x02211222, 0x00202000,
    0x00202002, 0x00202200, 0x00202202, 0x00212101, 0x00222000, 0x00222002, 0x00222200, 0x00222202,
    0x01202101, 0x01212001, 0x01212100, 0x01222101, 0x02202000, 0x02202002, 0x02202200, 0x02202202,
    0x02222000, 0x02222002, 0x02222200, 0x02222202, 0x00202211, 0x00212011, 0x00212110, 0x00212211,
    0x00222111, 0x01202112, 0x01202211, 0x01212012, 0x01212111, 0x01222011, 0x01222110, 0x01222112,
    0x01222211, 0x02202111, 0x02212010, 0x02212112, 0x02212211, 0x02222110, 0x02222111, 0x00202020,
    0x00202022, 0x00202220, 0x00202222, 0x00222020, 0x00222022, 0x00222220, 0x00222222, 0x01202121,
    0x01212021, 0x01212122, 0x01212221, 0x01222121, 0x02202020, 0x02202022, 0x02202220, 0x02202222,
    0x02212121, 0x02222020, 0x02222022, 0x02222220, 0x02222222, 0x10000101, 0x10010001, 0x10010102,
    0x10020101, 0x11000201, 0x11010002, 0x11010101, 0x11010200, 0x11010202, 0x11020001, 0x11020100,
    0x11020102, 0x12010100, 0x12010201, 0x12020001, 0x12020102, 0x10000010, 0x10000011, 0x10000110,
    0x10000112, 0x10000211, 0x10010012, 0x10010111, 0x10010112, 0x10010210, 0x10010212, 0x10020011,
    0x10020112, 0x10020211, 0x11000111, 0x11000210, 0x11000212, 0x11010011, 0x11010110, 0x11010111,
    0x11010112, 0x11010211, 0x11010212, 0x11020111, 0x11020210, 0x11020212, 0x12000011, 0x12000110,
    0x12000112, 0x12010010, 0x12010012, 0x12010111, 0x12020010, 0x12020011, 0x12020012, 0x10000121,
    0x10010021, 0x10010120, 0x10010122, 0x10020121, 0x11000021, 0x11010022, 0x11010121, 0x11010222,
    0x11020120, 0x11020221, 0x12000221, 0x12010120, 0x12020121, 0x10001001, 0x10011101, 0x10011201,
    0x10021201, 0x11001101, 0x11001200, 0x11001202, 0x11011001, 0x11011100, 0x11011101, 0x11011102,
    0x11021001, 0x11021002, 0x11021101, 0x11021200, 0x11021202, 0x12001001, 0x12001102, 0x12001201,
    0x12011000, 0x12011002, 0x12011101, 0x12021000, 0x12021001, 0x12021201, 0x10001011, 0x10001012,
    0x10001111, 0x10001212, 0x10011011, 0x10011110, 0x10011111, 0x10011112, 0x10011211, 0x10021010,
    0x10021111, 0x10021212, 0x11001011, 0x11001110, 0x11001111, 0x11001112, 0x11001211, 0x11011010,
    0x11011011, 0x11011110, 0x11011111, 0x11011112, 0x11011210, 0x11011211, 0x11021011, 0x11021110,
    0x11021111, 0x11021112, 0x11021211, 0x12001012, 0x12001110, 0x12001111, 0x12001210, 0x12011011,
    0x12011110, 0x12011111, 0x12011112, 0x12011211, 0x12011212, 0x12021111, 0x12021210, 0x12021212,
    0x10001021, 0x10001121, 0x10001221, 0x10011120, 0x10011121, 0x10011220, 0x10011222, 0x10021021,
    0x10021120, 0x10021221, 0x11001020, 0x11001022, 0x11001121, 0x11001220, 0x11011020, 0x11011021,
    0x11011022, 0x11011121, 0x11011122, 0x11011221, 0x11021022, 0x11021121, 0x11021220, 0x12001021,
    0x12001121, 0x12001222, 0x12011120, 0x12011121, 0x12021021, 0x12021120, 0x12021122, 0x10002101,
    0x10012001, 0x10012101, 0x10012202, 0x10022101, 0x11002002, 0x11002201, 0x11012000, 0x11012101,
    0x11012200, 0x11022001, 0x11022100, 0x11022102, 0x11022201, 0x12002101, 0x12012001, 0x12012100,
    0x12012102, 0x12012201, 0x12022101, 0x10002011, 0x10002111, 0x10002112, 0x10002212, 0x10012010,
    0x10012110, 0x10012111, 0x10012210, 0x10022011, 0x10022110, 0x10022112, 0x11002010, 0x11002111,
    0x11002212, 0x11012011, 0x11012012, 0x11012110, 0x11012111, 0x11012112, 0x11012211, 0x11022010,
    0x11022012, 0x11022111, 0x11022112, 0x11022212, 0x12002112, 0x12002211, 0x12012012, 0x12012111,
    0x12012112, 0x12012210, 0x12022011, 0x12022110, 0x12022112, 0x12022211, 0x10012122, 0x11002120,
    0x11002122, 0x11002221, 0x11012121, 0x11012220, 0x11012222, 0x11022120, 0x11022221, 0x12012120,
    0x12022121, 0x10100001, 0x10100100, 0x10100101, 0x10100102, 0x10100201, 0x10110002, 0x10110101,
    0x10110202, 0x10120001, 0x10120100, 0x10120201, 0x11100000, 0x11100101, 0x11100200, 0x11110001,
    0x11110100, 0x11110101, 0x11110102, 0x11110201, 0x11120101, 0x11120200, 0x12100102, 0x12100201,
    0x12110101, 0x12110200, 0x12120000, 0x12120001, 0x12120102, 0x12120201, 0x10100111, 0x10100210,
    0x10100211, 0x10100212, 0x10110011, 0x10110110, 0x10110111, 0x10110112, 0x10110210, 0x10110211,
    0x10120010, 0x10120111, 0x10120112, 0x10120210, 0x10120212, 0x11100011, 0x11100110, 0x11100111,
    0x11100112, 0x11100211, 0x11110010, 0x11110011, 0x11110012, 0x11110110, 0x11110111, 0x11110112,
    0x11110210, 0x11110211, 0x11110212, 0x11120011, 0x11120110, 0x11120111, 0x11120112, 0x11120211,
    0x12100012, 0x12100111, 0x12110011, 0x12110110, 0x12110111, 0x12110112, 0x12110211, 0x12120010,
    0x12120111, 0x12120212, 0x10100021, 0x10100122, 0x10110022, 0x10110121, 0x10110222, 0x10120021,
    0x10120120, 0x11100022, 0x11100121, 0x11100222, 0x11110021, 0x11110120, 0x11110121, 0x11110122,
    0x11110221, 0x11120022, 0x11120121, 0x12100121, 0x12110020, 0x12110022, 0x12110121, 0x12110221,
    0x12110222, 0x12120120, 0x10101100, 0x10101101, 0x10111001, 0x10111100, 0x10111101, 0x10111102,
    0x10111200, 0x10111201, 0x10121001, 0x10121101, 0x10121200, 0x10121202, 0x11101001, 0x11101100,
    0x11101101, 0x11101102, 0x11101201, 0x11101202, 0x11111000, 0x11111001, 0x11111100, 0x11111101,
    0x11111102, 0x11111200, 0x11111201, 0x11111202, 0x11121001, 0x11121002, 0x11121100, 0x11121101,
    0x11121102, 0x11121201, 0x12101000, 0x12101200, 0x12101202, 0x12111001, 0x12111100, 0x12111101,
    0x12111102, 0x12111201, 0x12121001, 0x12121100, 0x12121101, 0x12121202, 0x10101011, 0x10101012,
    0x10101110, 0x10101111, 0x10101112, 0x10101211, 0x10111010, 0x10111011, 0x10111012, 0x10111110,
    0x10111111, 0x10111112, 0x10111211, 0x10111212, 0x10121011, 0x10121110, 0x10121111, 0x10121112,
    0x10121211, 0x11101010, 0x11101011, 0x11101012, 0x11101110, 0x11101111, 0x11101112, 0x11101210,
    0x11101211, 0x11111010, 0x11111011, 0x11111012, 0x11111110, 0x11111111, 0x11111112, 0x11111210,
    0x11111211, 0x11111212, 0x11121010, 0x11121011, 0x11121110, 0x11121111, 0x11121112, 0x11121210,
    0x11121211, 0x11121212, 0x12101011, 0x12101110, 0x12101111, 0x12101211, 0x12101212, 0x12111010,
    0x12111011, 0x12111110, 0x12111111, 0x12111112, 0x12111210, 0x12111211, 0x12121011, 0x12121110,
    0x12121111, 0x12121112, 0x12121211, 0x10101020, 0x10101021, 0x10101022, 0x10101120, 0x10101122,
    0x10101220, 0x10101221, 0x10111021, 0x10111120, 0x10111121, 0x10111220, 0x10111221, 0x10121020,
    0x10121021, 0x10121022, 0x10121120, 0x10121121, 0x10121122, 0x10121220, 0x10121221, 0x11101021,
    0x11101121, 0x11101122, 0x11101220, 0x11101221, 0x11101222, 0x11111020, 0x11111021, 0x11111022,
    0x11111120, 0x11111121, 0x11111122, 0x11111220, 0x11111221, 0x11111222, 0x11121021, 0x11121120,
    0x11121121, 0x11121221, 0x12101022, 0x12101121, 0x12101122, 0x12101220, 0x12101221, 0x12101222,
    0x12111021, 0x12111121, 0x12111222, 0x12121022, 0x12121121, 0x12121122, 0x12121220, 0x12121221,
    0x10102100, 0x10102101, 0x10102102, 0x10102201, 0x10112000, 0x10112101, 0x10112200, 0x10122001,
    0x10122202, 0x11102101, 0x11102200, 0x11102202, 0x11112001, 0x11112100, 0x11112101, 0x11112102,
    0x11112200, 0x11112201, 0x11122000, 0x11122002, 0x11122100, 0x11122101, 0x12102002, 0x12102201,
    0x12112000, 0x12112002, 0x12112101, 0x12112200, 0x12122001, 0x12122201, 0x10102011, 0x10102012,
    0x10102111, 0x10102212, 0x10112011, 0x10112110, 0x10112111, 0x10112112, 0x10112211, 0x10122111,
    0x11102011, 0x11102110, 0x11102111, 0x11102112, 0x11102211, 0x11112010, 0x11112011, 0x11112012,
    0x11112110, 0x11112111, 0x11112112, 0x11112210, 0x11112211, 0x11112212, 0x11122011, 0x11122110,
    0x11122111, 0x11122112, 0x11122211, 0x12102011, 0x12102111, 0x12102211, 0x12112011, 0x12112110,
    0x12112111, 0x12112112, 0x12112210, 0x12112211, 0x12122111, 0x10102120, 0x10102220, 0x10112121,
    0x10112222, 0x10122020, 0x10122121, 0x10122122, 0x10122221, 0x11102121, 0x11102220, 0x11102221,
    0x11112021, 0x11112121, 0x11112122, 0x11112220, 0x11112221, 0x11122022, 0x11122121, 0x11122220,
    0x11122222, 0x12102021, 0x12102222, 0x12112022, 0x12112121, 0x12112122, 0x12112220, 0x12112222,
    0x12122021, 0x10200101, 0x10210100, 0x10210102, 0x10210201, 0x10220101, 0x11200100, 0x11210000,
    0x11210101, 0x11210102, 0x11210200, 0x11210202, 0x11220001, 0x11220100, 0x11220102, 0x11220201,
    0x12200001, 0x12210102, 0x12220101, 0x10200011, 0x10200110, 0x10200112, 0x10200211, 0x10210012,
    0x10210111, 0x10220011, 0x10220012, 0x10220112, 0x10220211, 0x11200111, 0x11200211, 0x11210011,
    0x11210111, 0x11210112, 0x11210211, 0x11220111, 0x11220112, 0x11220212, 0x12200110, 0x12200212,
    0x12210012, 0x12210111, 0x12220011, 0x12220112, 0x12220211, 0x10210021, 0x10210122, 0x10210221,
    0x11200020, 0x11200021, 0x11200122, 0x11210121, 0x11210122, 0x11210220, 0x11220020, 0x12200121,
    0x12210021, 0x12210122, 0x12220121, 0x10211001, 0x10211002, 0x10211101, 0x10211102, 0x10211202,
    0x10221001, 0x10221102, 0x10221201, 0x11201000, 0x11201002, 0x11201101, 0x11201200, 0x11201202,
    0x11211001, 0x11211100, 0x11211101, 0x11211102, 0x11211201, 0x11211202, 0x11221000, 0x11221002,
    0x11221101, 0x12201100, 0x12201101, 0x12201201, 0x12211000, 0x12211002, 0x12211100, 0x12211101,
    0x12211102, 0x12211200, 0x12211202, 0x12221001, 0x12221100, 0x12221201, 0x10201111, 0x10201210,
    0x10201212, 0x10211011, 0x10211111, 0x10211112, 0x10211211, 0x11201110, 0x11201111, 0x11201112,
    0x11201211, 0x11211010, 0x11211011, 0x11211110, 0x11211111, 0x11211112, 0x11211211, 0x11221011,
    0x11221110, 0x11221111, 0x11221112, 0x11221211, 0x12201112, 0x12201211, 0x12201212, 0x12211011,
    0x12211111, 0x12211112, 0x12211211, 0x12211212, 0x12221012, 0x12221111, 0x12221112, 0x12221210,
    0x10201022, 0x10201221, 0x10211121, 0x10221020, 0x10221122, 0x10221220, 0x10221221, 0x11201020,
    0x11201121, 0x11201220, 0x11201222, 0x11211021, 0x11211120, 0x11211121, 0x11211122, 0x11211220,
    0x11211222, 0x11221020, 0x11221121, 0x11221220, 0x12201020, 0x12201022, 0x12201121, 0x12201222,
    0x12211120, 0x12211122, 0x12211220, 0x12211221, 0x12221020, 0x12221120, 0x12221122, 0x12221222,
    0x10212102, 0x10212201, 0x10222101, 0x11202001, 0x11212002, 0x11212101, 0x11212202, 0x11222001,
    0x11222201, 0x12202101, 0x12212001, 0x12212200, 0x12222102, 0x10202011, 0x10202110, 0x10212010,
    0x10212111, 0x10222011, 0x10222110, 0x10222112, 0x10222211, 0x11202010, 0x11202011, 0x11202111,
    0x11202112, 0x11202210, 0x11212011, 0x11212110, 0x11212111, 0x11212112, 0x11212211, 0x11222010,
    0x11222111, 0x11222212, 0x12202012, 0x12202110, 0x12202212, 0x12212111, 0x12222011, 0x12222110,
    0x12222111, 0x12222211, 0x10212021, 0x10212122, 0x10212220, 0x11202021, 0x11202120, 0x11202221,
    0x11212020, 0x11212121, 0x11212220, 0x11212222, 0x11222120, 0x11222121, 0x11222221, 0x12202122,
    0x12212120, 0x12212220, 0x12212222, 0x12222122, 0x20000000, 0x20000002, 0x20000200, 0x20000202,
    0x20020000, 0x20020002, 0x20020200, 0x20020202, 0x21000101, 0x21010000, 0x21010001, 0x21010100,
    0x21010102, 0x21010201, 0x21020101, 0x22000000, 0x22000002, 0x22000200, 0x22000202, 0x22010101,
    0x22020000, 0x22020002, 0x22020200, 0x22020202, 0x20000111, 0x20010011, 0x20010110, 0x20010112,
    0x20010211, 0x20020111, 0x21000011, 0x21000110, 0x21000211, 0x21010010, 0x21010012, 0x21010111,
    0x21010112, 0x21010210, 0x21010211, 0x21020110, 0x21020112, 0x21020211, 0x22000111, 0x22000211,
    0x22010110, 0x22010112, 0x22010211, 0x22020111, 0x20000020, 0x20000022, 0x20000220, 0x20000222,
    0x20010121, 0x20020020, 0x20020022, 0x20020220, 0x20020222, 0x21010021, 0x21010120, 0x21010221,
    0x21020121, 0x22000020, 0x22000022, 0x22000220, 0x22000222, 0x22010121, 0x22020020, 0x22020022,
    0x22020220, 0x22020222, 0x20011100, 0x20011201, 0x21001001, 0x21001100, 0x21011001, 0x21011101,
    0x21011202, 0x21021001, 0x21021100, 0x21021201, 0x22011100, 0x22011201, 0x20001011, 0x20001211,
    0x20011012, 0x20011111, 0x20011212, 0x20021112, 0x20021211, 0x21001010, 0x21001011, 0x21001111,
    0x21001210, 0x21011011, 0x21011110, 0x21011111, 0x21011112, 0x21011211, 0x21011212, 0x21021111,
    0x21021112, 0x21021210, 0x21021212, 0x22001011, 0x22001110, 0x22001112, 0x22001211, 0x22011010,
    0x22011012, 0x22011111, 0x22011210, 0x22021112, 0x20011021, 0x20011122, 0x20011221, 0x20021121,
    0x21001021, 0x21001120, 0x21001221, 0x21001222, 0x21011020, 0x21011121, 0x21011221, 0x21011222,
    0x21021021, 0x21021122, 0x21021222, 0x22001121, 0x22011021, 0x22011222, 0x22021120, 0x20002000,
    0x20002002, 0x20002200, 0x20002202, 0x20012101, 0x20022000, 0x20022002, 0x20022200, 0x20022202,
    0x21002001, 0x21002101, 0x21012001, 0x21012100, 0x21012201, 0x21022101, 0x21022201, 0x22002000,
    0x22002002, 0x22002200, 0x22002202, 0x22012101, 0x22022000, 0x22022002, 0x22022200, 0x22022202,
    0x20002111, 0x20002112, 0x20012011, 0x20012110, 0x20012112, 0x20022111, 0x21002011, 0x21002110,
    0x21002112, 0x21002211, 0x21012010, 0x21012012, 0x21012111, 0x21012212, 0x21022011, 0x21022110,
    0x22002111, 0x22012112, 0x22012211, 0x22022111, 0x20002020, 0x20002022, 0x20002220, 0x20002222,
    0x20012121, 0x20022020, 0x20022022, 0x20022220, 0x20022222, 0x21002121, 0x21012021, 0x21012120,
    0x21012122, 0x22002020, 0x22002022, 0x22002220, 0x22002222, 0x22012121, 0x22022020, 0x22022022,
    0x22022220, 0x22022222, 0x20100101, 0x20110001, 0x20110102, 0x20110200, 0x20110201, 0x20120101,
    0x21100001, 0x21100102, 0x21100201, 0x21110101, 0x21110200, 0x21110202, 0x21120201, 0x21120202,
    0x22100101, 0x22110001, 0x22110100, 0x22110102, 0x22110201, 0x22120101, 0x20100011, 0x20100110,
    0x20100112, 0x20100211, 0x20110010, 0x20110111, 0x20110210, 0x20110212, 0x20120011, 0x20120110,
    0x20120112, 0x20120211, 0x21100010, 0x21100111, 0x21110010, 0x21110011, 0x21110110, 0x21110111,
    0x21110112, 0x21110211, 0x21120012, 0x21120111, 0x22100110, 0x22100112, 0x22110012, 0x22110111,
    0x22110210, 0x22120011, 0x22120110, 0x22120112, 0x22120211, 0x20100121, 0x20110021, 0x20110120,
    0x20110221, 0x20120121, 0x21100120, 0x21100122, 0x21100221, 0x21110020, 0x21110022, 0x21110121,
    0x21110220, 0x21120122, 0x21120221, 0x22100121, 0x22110120, 0x22110122, 0x22120221, 0x20101001,
    0x20101100, 0x20101102, 0x20111000, 0x20111101, 0x20111200, 0x20121102, 0x21101000, 0x21101202,
    0x21111001, 0x21111100, 0x21111101, 0x21111102, 0x21111200, 0x21111201, 0x21121000, 0x21121001,
    0x21121002, 0x21121101, 0x22101100, 0x22101102, 0x22111002, 0x22111100, 0x22111101, 0x22111200,
    0x22121001, 0x22121201, 0x20101010, 0x20101111, 0x20101210, 0x20101212, 0x20111010, 0x20111011,
    0x20111110, 0x20111111, 0x20111112, 0x20111211, 0x20121011, 0x20121111, 0x20121211, 0x20121212,
    0x21101011, 0x21101110, 0x21101111, 0x21101112, 0x21101211, 0x21111010, 0x21111011, 0x21111012,
    0x21111110, 0x21111111, 0x21111112, 0x21111210, 0x21111211, 0x21111212, 0x21121011, 0x21121110,
    0x21121111, 0x21121112, 0x21121211, 0x22101011, 0x22101111, 0x22101210, 0x22111011, 0x22111012,
    0x22111110, 0x22111111, 0x22111112, 0x22111211, 0x22111212, 0x22121010, 0x22121012, 0x22121111,
    0x22121210, 0x22121212, 0x20101021, 0x20101120, 0x20111020, 0x20111121, 0x20111221, 0x20121020,
    0x20121122, 0x20121221, 0x21101121, 0x21101220, 0x21101221, 0x21111021, 0x21111022, 0x21111121,
    0x21111122, 0x21111221, 0x21121121, 0x21121220, 0x22101022, 0x22101120, 0x22101221, 0x22101222,
    0x22111022, 0x22111120, 0x22111121, 0x22121120, 0x22121122, 0x22121221, 0x20102101, 0x20112102,
    0x20112201, 0x20122101, 0x21102001, 0x21102102, 0x21112000, 0x21112002, 0x21112101, 0x21112102,
    0x21112202, 0x21122100, 0x21122101, 0x22102101, 0x22112001, 0x22112102, 0x22112201, 0x22122101,
    0x20102110, 0x20102112, 0x20102211, 0x20112010, 0x20112012, 0x20112111, 0x20112210, 0x20112212,
    0x20122010, 0x20122011, 0x20122110, 0x20122112, 0x21102010, 0x21102012, 0x21102111, 0x21102210,
    0x21102212, 0x21112011, 0x21112110, 0x21112111, 0x21112112, 0x21112211, 0x21122012, 0x21122111,
    0x21122112, 0x21122212, 0x22102011, 0x22102110, 0x22112010, 0x22112012, 0x22112111, 0x22112212,
    0x22122011, 0x22122112, 0x20102121, 0x20112121, 0x20122121, 0x21102120, 0x21102122, 0x21102221,
    0x21112020, 0x21112121, 0x21112220, 0x21122021, 0x22102121, 0x22112021, 0x22112120, 0x22112121,
    0x22112122, 0x20200000, 0x20200002, 0x20200200, 0x20200202, 0x20210101, 0x20220000, 0x20220002,
    0x20220200, 0x20220202, 0x21200101, 0x21210001, 0x21210100, 0x21210102, 0x21210201, 0x22200000,
    0x22200002, 0x22200200, 0x22200202, 0x22210101, 0x22220000, 0x22220002, 0x22220200, 0x22220202,
    0x20200111, 0x20200211, 0x20210011, 0x20210110, 0x20210112, 0x20210211, 0x20210212, 0x21200112,
    0x21200211, 0x21210011, 0x21210111, 0x21210210, 0x21210212, 0x21220011, 0x21220110, 0x22200111,
    0x22210010, 0x22210012, 0x22210112, 0x22210211, 0x20200022, 0x20200220, 0x20200222, 0x20210020,
    0x20210221, 0x20220022, 0x20220220, 0x20220222, 0x21200121, 0x21210021, 0x21210122, 0x21210221,
    0x21220121, 0x22200020, 0x22200022, 0x22200220, 0x22200222, 0x22210121, 0x22220020, 0x22220022,
    0x22220220, 0x22220222, 0x20211201, 0x20221101, 0x21201001, 0x21201100, 0x21211000, 0x21211100,
    0x21211101, 0x21211200, 0x21211202, 0x21221001, 0x21221101, 0x21221102, 0x21221200, 0x21221201,
    0x22201101, 0x20201112, 0x20201211, 0x20211010, 0x20211012, 0x20211111, 0x20211210, 0x20221112,
    0x20221211, 0x21201012, 0x21201111, 0x21211011, 0x21211110, 0x21211111, 0x21211112, 0x21211211,
    0x21221111, 0x21221212, 0x22201011, 0x22201110, 0x22201111, 0x22201112, 0x22201211, 0x22211012,
    0x22211111, 0x22211210, 0x20201121, 0x20211021, 0x20211122, 0x20211222, 0x20221021, 0x20221121,
    0x21201120, 0x21201122, 0x21201222, 0x21211022, 0x21211121, 0x21211122, 0x21211220, 0x21221020,
    0x21221022, 0x22201122, 0x22211020, 0x22211121, 0x22211122, 0x22211221, 0x22221021, 0x22221120,
    0x22221122, 0x20202000, 0x20202002, 0x20202200, 0x20202202, 0x20222000, 0x20222002, 0x20222200,
    0x20222202, 0x21212001, 0x21212100, 0x21212102, 0x21212201, 0x22202000, 0x22202002, 0x22202200,
    0x22202202, 0x22212101, 0x22222000, 0x22222002, 0x22222200, 0x22222202, 0x20202111, 0x20212110,
    0x20212211, 0x20222011, 0x20222111, 0x21202011, 0x21212010, 0x21212111, 0x21212212, 0x21222011,
    0x21222112, 0x21222211, 0x22212010, 0x22212112, 0x20202020, 0x20202022, 0x20202220, 0x20202222,
    0x20222020, 0x20222022, 0x20222220, 0x20222222, 0x21212021, 0x21212120, 0x21212122, 0x22202020,
    0x22202022, 0x22202220, 0x22202222, 0x22212121, 0x22222020, 0x22222022, 0x22222220, 0x22222222,
};
#endif

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

__m128i inline load_iq4nl_values_128() {
    static const uint8_t kvalues_iq4nl[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};
    return _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
}

__m256i inline load_iq4nl_values_256() {
    auto val128 = load_iq4nl_values_128();
    return MM256_SET_M128I(val128, val128);
}

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

__m512i inline load_iq4nl_values_512() {
    auto val256 = load_iq4nl_values_256();
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
        return _mm_add_epi8(scl, m8);
    }
    Q2Bits bits;
    const IQXKScales iqxk;

    const __m512i values;
    const __m128i m8 = _mm_set1_epi8(-8);
};

struct DequantizerIQ2KS final : public BaseDequantizer<block_iq2_ks, true, true> {
    DequantizerIQ2KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m512i * scales) {
        prepare(x[i].qs);
        auto scales128 = make_scales(x[i].scales, x[i].extra >> 8);
        auto shifts = _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi8(x[i].extra), hmask), hmask), m5);
        auto scales_s = _mm_mullo_epi16(scales128, _mm_cvtepi8_epi16(_mm_add_epi8(m32, shifts)));
        s8k.accum_mins(scales_s, q8, i, d, accm);
        auto scales256 = MM256_SET_M128I(scales128, scales128);
        auto all_scales = _mm512_inserti32x8(_mm512_castsi256_si512(scales256), scales256, 1);
        scales[0] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[0]);
        scales[1] = _mm512_shuffle_epi8(all_scales, s8k.shuffles512[1]);
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
    Scales8K s8k;

    const __m512i values;
    const __m128i m16 = _mm_set1_epi8(-16);
    const __m128i m5  = _mm_set1_epi8(5);
    const __m128i m32 = _mm_set1_epi8(-32);
    const __m128i hmask = _mm_set1_epi64x(0x8040201008040201);
    const __m128i shuffle = _mm_set1_epi64x(0x0703060205010400);
    const __m128i shift = _mm_set_epi32(0, 0, 4, 0);
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

struct DequantizerIQ4K final : public BaseDequantizer<block_iq4_k> {
    DequantizerIQ4K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(4, -128), values(load_iq4nl_values_256()) {}
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
        auto mask1 = _mm256_andnot_si256(_mm256_or_si256(mask4, _mm256_or_si256(mask2, mask3)), _mm256_set1_epi8(-1)); // 0xff;
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

struct DequantizerIQ4KS final : public BaseDequantizer<block_iq4_ks, true> {
    DequantizerIQ4KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_iq4nl_values_256()) {}
    template <typename Q8>
    inline __m256i new_block(int i, const Q8& q8, __m256 * accd) {
        auto scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
        auto shifts = _mm_and_si128(_mm_cmpeq_epi16(_mm_and_si128(scales128, m1), m1), m4);
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, mask), m127);
        auto scales_s = _mm_mullo_epi16(scales128, _mm_add_epi16(m128, shifts));
        s8k.accum_mins(scales_s, q8, i, d, accd);
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
    Scales8KBase s8k;
    const __m256i values;
    const __m128i mask     = _mm_set1_epi16(254);
    const __m128i m127     = _mm_set1_epi16(-127);
    const __m128i m128     = _mm_set1_epi16(-128);
    const __m128i m1       = _mm_set1_epi16(1);
    const __m128i m4       = _mm_set1_epi16(4);
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

#endif  // Zen4 or vanilla AVX2

template <int nrc_y>
static void mul_mat_iq2_bn_r4_q8_k16_avx2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if (nrc_x%4) {
        printf("%s: %d is not a multiple of 4\n", __func__, nrc_x);
        GGML_ABORT("fatal error");
    }
    Q8_16<nrc_y> q8(info);
    auto m3 = _mm256_set1_epi8(0x3);
    auto m1 = _mm256_set1_epi16(1);
    int nb = n / QK_IQ1BN;
    __m256i qx[4];
    if constexpr (nrc_y > 4) {
    __m256i acc[nrc_y] = {};
    __m128  sum4[nrc_y];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const float * dptr = (const float *)((const char *)vx + ix*bx);
        auto dl = _mm_loadu_ps(dptr);
        const uint8_t * iq2l = (const uint8_t *)(dptr + 4);
        for (int ib = 0; ib < nb; ++ib) {
            auto bits = _mm256_loadu_si256((const __m256i *)iq2l + 2*ib+0);
            qx[0] = _mm256_and_si256(bits, m3);
            qx[1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), m3);
            qx[2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m3);
            qx[3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), m3);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = q8.load_quants(iy, 2*ib+0);
                auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                              _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                              _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                acc[iy] = _mm256_add_epi32(acc[iy], _mm256_madd_epi16(m1, _mm256_add_epi16(sumi1, sumi2)));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto dy = q8.scale(iy);
            auto sumf1 = _mm256_cvtepi32_ps(acc[iy]);
            auto s4 = _mm_mul_ps(_mm256_extractf128_ps(sumf1, 0), _mm_mul_ps(dl, _mm_shuffle_ps(dy, dy, 0x00)));
            s4 = _mm_fmadd_ps(_mm256_extractf128_ps(sumf1, 1), _mm_mul_ps(dl, _mm_shuffle_ps(dy, dy, 0x55)), s4);
            sum4[iy] = _mm_fmadd_ps(dl, _mm_set1_ps(-q8.sum_row(iy)), s4);
            acc[iy] = _mm256_setzero_si256();
        }
        for (int ib = 0; ib < nb; ++ib) {
            auto bits = _mm256_loadu_si256((const __m256i *)iq2l + 2*ib+1);
            qx[0] = _mm256_and_si256(bits, m3);
            qx[1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), m3);
            qx[2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m3);
            qx[3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), m3);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = q8.load_quants(iy, 2*ib+1);
                auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                              _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                              _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                acc[iy] = _mm256_add_epi32(acc[iy], _mm256_madd_epi16(m1, _mm256_add_epi16(sumi1, sumi2)));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto dy = q8.scale(iy);
            auto sumf1 = _mm256_cvtepi32_ps(acc[iy]);
            auto s4 = _mm_fmadd_ps(_mm256_extractf128_ps(sumf1, 0), _mm_mul_ps(dl, _mm_shuffle_ps(dy, dy, 0xaa)), sum4[iy]);
            s4 = _mm_fmadd_ps(_mm256_extractf128_ps(sumf1, 1), _mm_mul_ps(dl, _mm_shuffle_ps(dy, dy, 0xff)), s4);
            info.store(ix, iy, s4);
            acc[iy] = _mm256_setzero_si256();
        }
    }
    } else {
    __m256i acc[2*nrc_y] = {};
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const float * dptr = (const float *)((const char *)vx + ix*bx);
        auto dl = _mm_loadu_ps(dptr);
        const uint8_t * iq2l = (const uint8_t *)(dptr + 4);
        for (int ib = 0; ib < nb; ++ib) {
            auto bits = _mm256_loadu_si256((const __m256i *)iq2l + 2*ib+0);
            qx[0] = _mm256_and_si256(bits, m3);
            qx[1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), m3);
            qx[2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m3);
            qx[3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), m3);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = q8.load_quants(iy, 2*ib+0);
                auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                              _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                              _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                acc[2*iy+0] = _mm256_add_epi32(acc[2*iy+0], _mm256_madd_epi16(m1, _mm256_add_epi16(sumi1, sumi2)));
            }
            bits = _mm256_loadu_si256((const __m256i *)iq2l + 2*ib+1);
            qx[0] = _mm256_and_si256(bits, m3);
            qx[1] = _mm256_and_si256(_mm256_srli_epi16(bits, 2), m3);
            qx[2] = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m3);
            qx[3] = _mm256_and_si256(_mm256_srli_epi16(bits, 6), m3);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = q8.load_quants(iy, 2*ib+1);
                auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00)),
                                              _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55)));
                auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa)),
                                              _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff)));
                acc[2*iy+1] = _mm256_add_epi32(acc[2*iy+1], _mm256_madd_epi16(m1, _mm256_add_epi16(sumi1, sumi2)));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto dy = q8.scale(iy);
            auto sumf1 = _mm256_cvtepi32_ps(acc[2*iy+0]);
            auto sumf2 = _mm256_cvtepi32_ps(acc[2*iy+1]);
            auto sum4 = _mm_mul_ps(_mm256_extractf128_ps(sumf1, 0), _mm_mul_ps(dl, _mm_shuffle_ps(dy, dy, 0x00)));
            sum4 = _mm_fmadd_ps(_mm256_extractf128_ps(sumf1, 1), _mm_mul_ps(dl, _mm_shuffle_ps(dy, dy, 0x55)), sum4);
            sum4 = _mm_fmadd_ps(_mm256_extractf128_ps(sumf2, 0), _mm_mul_ps(dl, _mm_shuffle_ps(dy, dy, 0xaa)), sum4);
            sum4 = _mm_fmadd_ps(_mm256_extractf128_ps(sumf2, 1), _mm_mul_ps(dl, _mm_shuffle_ps(dy, dy, 0xff)), sum4);
            sum4 = _mm_fmadd_ps(dl, _mm_set1_ps(-q8.sum_row(iy)), sum4);
            info.store(ix, iy, sum4);
            acc[2*iy+0] = acc[2*iy+1] = _mm256_setzero_si256();
        }
    }
    }
}

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_iq2_bn_r4_q8_k16(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if (nrc_x%4) {
        printf("%s: %d is not a multiple of 4\n", __func__, nrc_x);
        GGML_ABORT("fatal error");
    }
    if constexpr (nrc_y == 1) {
        mul_mat_iq2_bn_r4_q8_k16_avx2<1>(n, vx, bx, info, nrc_x);
    } else {
    Q8_16<nrc_y> q8(info);
    auto m3 = _mm512_set1_epi8(0x3);
    int nb = n / QK_IQ1BN;
    __m512i acc[2*nrc_y] = {};
    __m512i qx[8];
    for (int ix = 0; ix < nrc_x/8; ++ix) {
        const float * dptr1 = (const float *)((const char *)vx + (8*ix+0)*bx);
        const float * dptr2 = (const float *)((const char *)vx + (8*ix+4)*bx);
        auto dl = _mm_loadu_ps(dptr1);
        auto dh = _mm_loadu_ps(dptr2);
        const uint8_t * iq2l = (const uint8_t *)(dptr1 + 4);
        const uint8_t * iq2h = (const uint8_t *)(dptr2 + 4);
        for (int ib = 0; ib < nb; ++ib) {
            auto bits_l = _mm512_loadu_si512((const __m512i *)iq2l + ib);
            auto bits_h = _mm512_loadu_si512((const __m512i *)iq2h + ib);
            qx[0] = _mm512_and_si512(bits_l, m3);
            qx[1] = _mm512_and_si512(bits_h, m3);
            qx[2] = _mm512_and_si512(_mm512_srli_epi16(bits_l, 2), m3);
            qx[3] = _mm512_and_si512(_mm512_srli_epi16(bits_h, 2), m3);
            qx[4] = _mm512_and_si512(_mm512_srli_epi16(bits_l, 4), m3);
            qx[5] = _mm512_and_si512(_mm512_srli_epi16(bits_h, 4), m3);
            qx[6] = _mm512_and_si512(_mm512_srli_epi16(bits_l, 6), m3);
            qx[7] = _mm512_and_si512(_mm512_srli_epi16(bits_h, 6), m3);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = q8.load_quants64(iy, ib);
                auto sy = _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00));
                acc[2*iy+0] = _mm512_dpbusd_epi32(acc[2*iy+0], qx[0], sy);
                acc[2*iy+1] = _mm512_dpbusd_epi32(acc[2*iy+1], qx[1], sy);
                sy = _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55));
                acc[2*iy+0] = _mm512_dpbusd_epi32(acc[2*iy+0], qx[2], sy);
                acc[2*iy+1] = _mm512_dpbusd_epi32(acc[2*iy+1], qx[3], sy);
                sy = _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa));
                acc[2*iy+0] = _mm512_dpbusd_epi32(acc[2*iy+0], qx[4], sy);
                acc[2*iy+1] = _mm512_dpbusd_epi32(acc[2*iy+1], qx[5], sy);
                sy = _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff));
                acc[2*iy+0] = _mm512_dpbusd_epi32(acc[2*iy+0], qx[6], sy);
                acc[2*iy+1] = _mm512_dpbusd_epi32(acc[2*iy+1], qx[7], sy);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto dy = q8.scale(iy);
            __m128 sum4;
            for (int k = 0; k < 2; ++k) {
                const auto& dx = k == 0 ? dl : dh;
                auto sumf = _mm512_cvtepi32_ps(acc[2*iy+k]);
                sum4 = _mm_mul_ps  (_mm512_extractf32x4_ps(sumf, 0), _mm_mul_ps(dx, _mm_shuffle_ps(dy, dy, 0x00)));
                sum4 = _mm_fmadd_ps(_mm512_extractf32x4_ps(sumf, 1), _mm_mul_ps(dx, _mm_shuffle_ps(dy, dy, 0x55)), sum4);
                sum4 = _mm_fmadd_ps(_mm512_extractf32x4_ps(sumf, 2), _mm_mul_ps(dx, _mm_shuffle_ps(dy, dy, 0xaa)), sum4);
                sum4 = _mm_fmadd_ps(_mm512_extractf32x4_ps(sumf, 3), _mm_mul_ps(dx, _mm_shuffle_ps(dy, dy, 0xff)), sum4);
                sum4 = _mm_fmadd_ps(dx, _mm_set1_ps(-q8.sum_row(iy)), sum4);
                info.store(8*ix + 4*k, iy, sum4);
            }
            acc[2*iy+0] = acc[2*iy+1] = _mm512_setzero_si512();
        }
    }
    if (int ix = 8*(nrc_x/8); ix < nrc_x) {
        const float * dptr = (const float *)((const char *)vx + ix*bx);
        auto dl = _mm_loadu_ps(dptr);
        const uint8_t * iq2l = (const uint8_t *)(dptr + 4);
        for (int ib = 0; ib < nb; ++ib) {
            auto bits_l = _mm512_loadu_si512((const __m512i *)iq2l + ib);
            qx[0] = _mm512_and_si512(bits_l, m3);
            qx[1] = _mm512_and_si512(_mm512_srli_epi16(bits_l, 2), m3);
            qx[2] = _mm512_and_si512(_mm512_srli_epi16(bits_l, 4), m3);
            qx[3] = _mm512_and_si512(_mm512_srli_epi16(bits_l, 6), m3);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = q8.load_quants64(iy, ib);
                acc[iy] = _mm512_dpbusd_epi32(acc[iy], qx[0], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
                acc[iy] = _mm512_dpbusd_epi32(acc[iy], qx[1], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
                acc[iy] = _mm512_dpbusd_epi32(acc[iy], qx[2], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
                acc[iy] = _mm512_dpbusd_epi32(acc[iy], qx[3], _mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto dy = q8.scale(iy);
            auto sumf = _mm512_cvtepi32_ps(acc[iy]);
            auto sum4 = _mm_mul_ps(_mm512_extractf32x4_ps(sumf, 0), _mm_mul_ps(dl, _mm_shuffle_ps(dy, dy, 0x00)));
            sum4 = _mm_fmadd_ps(_mm512_extractf32x4_ps(sumf, 1), _mm_mul_ps(dl, _mm_shuffle_ps(dy, dy, 0x55)), sum4);
            sum4 = _mm_fmadd_ps(_mm512_extractf32x4_ps(sumf, 2), _mm_mul_ps(dl, _mm_shuffle_ps(dy, dy, 0xaa)), sum4);
            sum4 = _mm_fmadd_ps(_mm512_extractf32x4_ps(sumf, 3), _mm_mul_ps(dl, _mm_shuffle_ps(dy, dy, 0xff)), sum4);
            sum4 = _mm_fmadd_ps(dl, _mm_set1_ps(-q8.sum_row(iy)), sum4);
            info.store(ix, iy, sum4);
        }
    }
    }
}
#else
template <int nrc_y>
static void mul_mat_iq2_bn_r4_q8_k16(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if (nrc_x%4) {
        printf("%s: %d is not a multiple of 4\n", __func__, nrc_x);
        GGML_ABORT("fatal error");
    }
    mul_mat_iq2_bn_r4_q8_k16_avx2<nrc_y>(n, vx, bx, info, nrc_x);
}
#endif

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_iq4_nl_r4_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_1_x4> q8(info);
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
                _mm256_storeu_ps(d8+8*iy, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d)));
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
                auto dy = _mm512_set1_ps(GGML_FP16_TO_FP32(qy[ib].d));
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(GGML_FP16_TO_FP32(qy[ib].s)), acc[2*iy+1]);
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
static void mul_mat_iq4_nl_r4_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_1_x4> q8(info);
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
                _mm_storeu_ps(d8+4*iy, _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)q8.y[iy][ib4].d)));
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
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_FP16_TO_FP32(qy[ib].d)));
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
    auto sumi = _mm256_add_epi32(_mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_add_epi16(sumi1, sumi2)),
                                 _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_add_epi16(sumi3, sumi4)));
#endif
    return sumi;
}

template <int nrc_y>
static void mul_mat_q4_0_r8_q8_1_avx2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
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
                helper.vec = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8.y[0][ib4].d));
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
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_FP16_TO_FP32(qy[ib].d)));
                acc1 = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc1);
                acc2 = _mm256_fmadd_ps(scales, _mm256_set1_ps(GGML_FP16_TO_FP32(qy[ib].s)), acc2);
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
                    auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d));
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
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_FP16_TO_FP32(qy[ib].d)));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                acc[iy] = _mm256_fmadd_ps(scales_m, _mm256_set1_ps(GGML_FP16_TO_FP32(qy[ib].s)), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
            acc[iy] = _mm256_setzero_ps();
        }
    }
    }
}

template <int nrc_y>
static void mul_mat_iq1_s_r4_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K128> q8(info);
    int nb = n / 32;
    GGML_ASSERT(nb%4 == 0);
    __m256i qx[4];
    __m256  acc[nrc_y] = {};
    auto m1 = _mm256_set1_epi16(1);
    auto ms = _mm_set1_epi16(-32768);
    float d8[4*nrc_y];
    union { __m256i vec; uint16_t val[16]; } helper;
    struct aux_iq1_s_r4 {
        uint8_t  qs[16];
        uint64_t qh;
    };
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto dptr = (const ggml_half *)((const char *)vx + ix*bx);
        auto d1 = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)dptr));
        auto x = (const aux_iq1_s_r4 *)(dptr + 4);
        for (int ib = 0; ib < nb/4; ++ib) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto bsums = _mm_cvtepi16_epi32(_mm_loadl_epi64((const __m128i *)q8.y[iy][ib].bsums));
                _mm_storeu_ps(d8 + 4*iy, _mm_mul_ps(_mm_set1_ps(q8.y[iy][ib].d), _mm_cvtepi32_ps(bsums)));
            }
            for (int k = 0; k < 4; ++k) {
                auto idxh = _mm256_set1_epi64x(x[4*ib+k].qh);
                auto sas = _mm256_castsi256_si128(idxh);
                auto scales4 = _mm_and_si128(_mm_srli_epi16(sas, 12), _mm_set1_epi16(7));
                scales4 = _mm_or_si128(_mm_slli_epi16(scales4, 1), _mm_set1_epi16(1));
                auto signs = _mm_or_si128(_mm_cmpeq_epi16(_mm_and_si128(sas, ms), ms), _mm256_castsi256_si128(m1));
                signs = _mm_add_epi16(_mm_set1_epi16(-8), signs);
                signs = _mm_mullo_epi16(signs, scales4);
                auto delta4 = _mm_mul_ps(_mm_set1_ps(0.0625f), _mm_cvtepi32_ps(_mm_cvtepi16_epi32(signs)));
                auto delta = _mm256_set_m128(delta4, delta4);
                scales4 = _mm_unpacklo_epi16(scales4, scales4); // 0,0, 1,1, 2,2, 3,3
                auto scales = MM256_SET_M128I(scales4, scales4);
                auto idxl = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)x[4*ib+k].qs));
                idxh = _mm256_sllv_epi64(idxh, _mm256_set_epi64x(0, 2, 5, 8));
                idxh = _mm256_srlv_epi64(idxh, _mm256_set_epi64x(1, 0, 0, 0));
                helper.vec = _mm256_or_si256(idxl, _mm256_and_si256(_mm256_set1_epi16(0x0700), idxh));
                qx[0] = _mm256_set_epi64x(iq1s_grid_us[helper.val[ 9]], iq1s_grid_us[helper.val[ 8]],
                                          iq1s_grid_us[helper.val[ 1]], iq1s_grid_us[helper.val[ 0]]);
                qx[1] = _mm256_set_epi64x(iq1s_grid_us[helper.val[13]], iq1s_grid_us[helper.val[12]],
                                          iq1s_grid_us[helper.val[ 5]], iq1s_grid_us[helper.val[ 4]]);
                qx[2] = _mm256_set_epi64x(iq1s_grid_us[helper.val[11]], iq1s_grid_us[helper.val[10]],
                                          iq1s_grid_us[helper.val[ 3]], iq1s_grid_us[helper.val[ 2]]);
                qx[3] = _mm256_set_epi64x(iq1s_grid_us[helper.val[15]], iq1s_grid_us[helper.val[14]],
                                          iq1s_grid_us[helper.val[ 7]], iq1s_grid_us[helper.val[ 6]]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ib].qs + k);
#ifdef HAVE_FANCY_SIMD
                    // 0,0, 1,1, 0,0, 1,1 as int32_t
                    auto sumi1 = _mm256_dpbusd_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(),
                                qx[0], _mm256_shuffle_epi32(y, 0x44)), qx[1], _mm256_shuffle_epi32(y, 0xee));
                    // 2,2, 3,3, 2,2, 3,3 as int32_t
                    auto sumi2 = _mm256_dpbusd_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(),
                                qx[2], _mm256_shuffle_epi32(y, 0x44)), qx[3], _mm256_shuffle_epi32(y, 0xee));
                    auto sumi = _mm256_packs_epi32(sumi1, sumi2);
#else
                    // 4 x row 0, 4 x row 1, 4 x row 0, 4 x row 1
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x44)),
                                                  _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0xee)));
                    // 4 x row 2, 4 x row 3, 4 x row 2, 4 x row 3
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0x44)),
                                                  _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xee)));
                    // 0,0, 1,1, 0,0, 1,1  as int32_t
                    sumi1 = _mm256_madd_epi16(m1, sumi1);
                    // 2,2, 3,3, 2,2, 3,3  as int32_t
                    sumi2 = _mm256_madd_epi16(m1, sumi2);
                    // 0,0, 1,1, 2,2, 3,3, 0,0, 1,1, 2,2, 3,3 as int16_t
                    auto sumi = _mm256_packs_epi32(sumi1, sumi2);
#endif
                    sumi = _mm256_madd_epi16(scales, sumi);
                    acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(q8.y[iy][ib].d), _mm256_cvtepi32_ps(sumi), acc[iy]);
                    acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d8[4*iy+k]), delta, acc[iy]);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sumf = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, _mm_mul_ps(d1, sumf));
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

// sum[ qy_i * ls_k * (qx_i - 1+/-delta_k)]
// = sum[qy_i * qx_i * ls_k] - 1/8*sum[qy_i * ls_k * (8+/-o_k)]
// = 1/8 * ( sum[qy_i * qx_i * 8*ls+k] - sum[qy_i * ls_k * (8+/-o_k)] )

template <int nrc_y>
static void mul_mat_iq1_s_q8_K(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    __m256i qx[8];
    __m256i scales[4];
    __m256  acc[nrc_y] = {};
    auto delta_mask = _mm_set1_epi16(-32768); // to avoid stupid overflow warnings when using 0x8000
    __m256i shuffle0 = _mm256_set_epi64x(0x0302030203020302, 0x0100010001000100, 0x0302030203020302, 0x0100010001000100);
    for (int ix = 0; ix < nrc_x; ++ix) {
        auto iq1s = (const block_iq1_s *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < n/QK_K; ++ibl) {
            float d = GGML_FP16_TO_FP32(iq1s[ibl].d);
            auto qhb = _mm_loadu_si128((const __m128i *)iq1s[ibl].qh);
            auto scales128 = _mm_and_si128(_mm_srli_epi16(qhb, 12), _mm_set1_epi16(7));
            scales128 = _mm_add_epi16(_mm_slli_epi16(scales128, 1), _mm_set1_epi16(1));
#ifdef HAVE_FANCY_SIMD
            auto mask = _mm_cmpeq_epi16_mask(_mm_and_si128(qhb, delta_mask), delta_mask);
            auto deltas128 = _mm_mask_blend_epi16(mask, _mm_set1_epi16(-7), _mm_set1_epi16(-9));
#else
            auto mask = _mm_cmpeq_epi16(_mm_and_si128(qhb, delta_mask), delta_mask);
            auto deltas128 = _mm_or_si128(_mm_and_si128(mask, _mm_set1_epi16(-9)), _mm_andnot_si128(mask, _mm_set1_epi16(-7)));
#endif
            deltas128 = _mm_mullo_epi16(scales128, deltas128);
            scales128 = _mm_slli_epi16(scales128, 3);
            auto deltas_l = _mm_unpacklo_epi16(deltas128, deltas128);
            auto deltas_h = _mm_unpackhi_epi16(deltas128, deltas128);
            auto deltas = MM256_SET_M128I(deltas_h, deltas_l); // blocks 0,0, 1,1, 2,2, ..., 7,7
            auto all_scales = MM256_SET_M128I(scales128, scales128);
            auto shuffle = shuffle0;
            for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
                scales[ib64] = _mm256_shuffle_epi8(all_scales, shuffle);
                shuffle = _mm256_add_epi8(shuffle, _mm256_set1_epi8(4));
            }
            const uint8_t  * qs = iq1s[ibl].qs;
            const uint16_t * qh = iq1s[ibl].qh;
            for (int ib = 0; ib < QK_K/32; ib += 2) {
                qx[ib+0] = _mm256_set_epi64x(iq1s_grid_us[qs[3] | ((qh[ib+0] >> 1) & 0x700)], iq1s_grid_us[qs[2] | ((qh[ib+0] << 2) & 0x700)],
                                             iq1s_grid_us[qs[1] | ((qh[ib+0] << 5) & 0x700)], iq1s_grid_us[qs[0] | ((qh[ib+0] << 8) & 0x700)]);
                qx[ib+1] = _mm256_set_epi64x(iq1s_grid_us[qs[7] | ((qh[ib+1] >> 1) & 0x700)], iq1s_grid_us[qs[6] | ((qh[ib+1] << 2) & 0x700)],
                                             iq1s_grid_us[qs[5] | ((qh[ib+1] << 5) & 0x700)], iq1s_grid_us[qs[4] | ((qh[ib+1] << 8) & 0x700)]);
                qs += 8;
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto bsums = q8.load_bsums(iy, ibl);
                auto sumi = _mm256_setzero_si256();
                for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
                    auto qy1 = q8.load_quants(iy, ibl, 2*ib64+0);
                    auto qy2 = q8.load_quants(iy, ibl, 2*ib64+1);
#ifdef HAVE_FANCY_SIMD
                    auto dot1 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[2*ib64+0], qy1);
                    auto dot2 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), qx[2*ib64+1], qy2);
                    sumi = _mm256_dpwssd_epi32(sumi, scales[ib64], _mm256_packs_epi32(dot1, dot2));
#else
                    auto dot1 = _mm256_maddubs_epi16(qx[2*ib64+0], qy1);
                    auto dot2 = _mm256_maddubs_epi16(qx[2*ib64+1], qy2);
                    auto dot  = _mm256_add_epi16(_mm256_unpacklo_epi64(dot1, dot2), _mm256_unpackhi_epi64(dot1, dot2));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scales[ib64], dot));
#endif
                }
#ifdef HAVE_FANCY_SIMD
                sumi = _mm256_dpwssd_epi32(sumi, bsums, deltas);
#else
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(bsums, deltas));
#endif
                acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d*q8.scale(iy, ibl)), _mm256_cvtepi32_ps(sumi), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, 0.125f*hsum_float_8(acc[iy]));
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

template <int nrc_y>
static void mul_mat_iq1_m_r4_q8_0(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K128> q8(info);
    int nb = n / 32;
    GGML_ASSERT(nb%4 == 0);
    auto shuffle0 = _mm256_set_epi64x(0x0909090909090909, 0x0808080808080808, 0x0101010101010101, 0x0000000000000000);
    auto step = _mm256_set1_epi8(2);
#ifndef HAVE_FANCY_SIMD
    auto m1 = _mm256_set1_epi16(1);
#endif
    __m256i qx[4];
    __m256  acc[nrc_y] = {};
    __m256i isum[nrc_y] = {};
    auto ms = _mm_set1_epi8(0x08);
    union { __m256i vec; uint16_t val[16]; } helper;
    for (int ix= 0; ix < nrc_x; ix += 4) {
        auto dptr = (const ggml_half *)((const char *)vx + ix*bx);
        auto d1 = _mm_mul_ps(_mm_set1_ps(0.125f), _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)dptr)));
        auto x = (const block_iq1_m_r4 *)(dptr + 4);
        for (int ib = 0; ib < nb/4; ++ib) {
            for (int k = 0; k < 4; ++k) {
                auto qh = (const uint32_t *)x[4*ib+k].qh;
                auto idxh = _mm_set_epi32(qh[1] >> 4, qh[1], qh[0] >> 4, qh[0]);
                auto scales4 = _mm_set1_epi32(((const uint32_t *)x[4*ib+k].scales)[0]);
                scales4 = _mm_and_si128(_mm_srlv_epi32(scales4, _mm_set_epi32(4, 0, 4, 0)), _mm_set1_epi8(0xf));
                scales4 = _mm_cvtepu8_epi16(scales4);
                auto scales = MM256_SET_M128I(_mm_unpackhi_epi16(scales4, scales4), _mm_unpacklo_epi16(scales4, scales4));

                auto signs128 = _mm_or_si128(_mm_cmpeq_epi8(_mm_and_si128(idxh, ms), ms), _mm_set1_epi8(1));
                signs128 = _mm_add_epi8(_mm_set1_epi8(-8), signs128);
                auto signs = MM256_SET_M128I(signs128, signs128);
                auto idxl = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)x[4*ib+k].qs));
                idxh = _mm_and_si128(idxh, _mm_set1_epi8(0x07));
                helper.vec = _mm256_or_si256(idxl, _mm256_slli_epi16(_mm256_cvtepu8_epi16(idxh), 8));
                qx[0] = _mm256_set_epi64x(iq1s_grid_us[helper.val[ 9]], iq1s_grid_us[helper.val[ 8]],
                                          iq1s_grid_us[helper.val[ 1]], iq1s_grid_us[helper.val[ 0]]);
                qx[1] = _mm256_set_epi64x(iq1s_grid_us[helper.val[13]], iq1s_grid_us[helper.val[12]],
                                          iq1s_grid_us[helper.val[ 5]], iq1s_grid_us[helper.val[ 4]]);
                qx[2] = _mm256_set_epi64x(iq1s_grid_us[helper.val[11]], iq1s_grid_us[helper.val[10]],
                                          iq1s_grid_us[helper.val[ 3]], iq1s_grid_us[helper.val[ 2]]);
                qx[3] = _mm256_set_epi64x(iq1s_grid_us[helper.val[15]], iq1s_grid_us[helper.val[14]],
                                          iq1s_grid_us[helper.val[ 7]], iq1s_grid_us[helper.val[ 6]]);
                qx[0] = _mm256_add_epi8(_mm256_slli_epi16(qx[0], 3), _mm256_shuffle_epi8(signs, shuffle0));
                auto shuffle = _mm256_add_epi8(shuffle0, step);
                qx[2] = _mm256_add_epi8(_mm256_slli_epi16(qx[2], 3), _mm256_shuffle_epi8(signs, shuffle));
                shuffle = _mm256_add_epi8(shuffle, step);
                qx[1] = _mm256_add_epi8(_mm256_slli_epi16(qx[1], 3), _mm256_shuffle_epi8(signs, shuffle));
                shuffle = _mm256_add_epi8(shuffle, step);
                qx[3] = _mm256_add_epi8(_mm256_slli_epi16(qx[3], 3), _mm256_shuffle_epi8(signs, shuffle));
                auto s0 = _mm256_sign_epi8(qx[0], qx[0]);
                auto s1 = _mm256_sign_epi8(qx[1], qx[1]);
                auto s2 = _mm256_sign_epi8(qx[2], qx[2]);
                auto s3 = _mm256_sign_epi8(qx[3], qx[3]);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i *)q8.y[iy][ib].qs + k);
                    auto y1 = _mm256_shuffle_epi32(y, 0x44);
                    auto y2 = _mm256_shuffle_epi32(y, 0xee);
#ifdef HAVE_FANCY_SIMD
                    // 0,0, 1,1, 0,0, 1,1 as int32_t
                    auto sumi1 = _mm256_dpbusd_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(),
                                s0, _mm256_sign_epi8(y1, qx[0])), s1, _mm256_sign_epi8(y2, qx[1]));
                    // 2,2, 3,3, 2,2, 3,3 as int32_t
                    auto sumi2 = _mm256_dpbusd_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(),
                                s2, _mm256_sign_epi8(y1, qx[2])), s3, _mm256_sign_epi8(y2, qx[3]));
                    auto sumi = _mm256_packs_epi32(sumi1, sumi2);
#else
                    // 4 x row 0, 4 x row 1, 4 x row 0, 4 x row 1
                    auto sumi1 = _mm256_add_epi16(_mm256_maddubs_epi16(s0, _mm256_sign_epi8(y1, qx[0])),
                                                  _mm256_maddubs_epi16(s1, _mm256_sign_epi8(y2, qx[1])));
                    // 4 x row 2, 4 x row 3, 4 x row 2, 4 x row 3
                    auto sumi2 = _mm256_add_epi16(_mm256_maddubs_epi16(s2, _mm256_sign_epi8(y1, qx[2])),
                                                  _mm256_maddubs_epi16(s3, _mm256_sign_epi8(y2, qx[3])));
                    // 0,0, 1,1, 0,0, 1,1  as int32_t
                    sumi1 = _mm256_madd_epi16(m1, sumi1);
                    // 2,2, 3,3, 2,2, 3,3  as int32_t
                    sumi2 = _mm256_madd_epi16(m1, sumi2);
                    // 0,0, 1,1, 2,2, 3,3, 0,0, 1,1, 2,2, 3,3 as int16_t
                    auto sumi = _mm256_packs_epi32(sumi1, sumi2);
#endif
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(scales, sumi));
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(q8.y[iy][ib].d), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
                isum[iy] = _mm256_setzero_si256();
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sumf = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            info.store(ix, iy, _mm_mul_ps(d1, sumf));
            acc[iy] = _mm256_setzero_ps();
        }
    }
}

#ifdef HAVE_FANCY_SIMD
template <int nrc_y>
static void mul_mat_q4_0_r8_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if constexpr (nrc_y == 1) {
        mul_mat_q4_0_r8_q8_1_avx2<1>(n, vx, bx, info, nrc_x);
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
                _mm256_storeu_ps(d8+8*iy, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d)));
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
                auto dy = _mm512_set1_ps(GGML_FP16_TO_FP32(qy[ib].d));
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(GGML_FP16_TO_FP32(qy[ib].s)), acc[2*iy+1]);
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
static void mul_mat_q4_0_r8_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    mul_mat_q4_0_r8_q8_1_avx2<nrc_y>(n, vx, bx, info, nrc_x);
}
#endif

template <int nrc_y>
static void mul_mat_q5_0_r4_q8_1_avx2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_1_x4> q8(info);
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
                auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d));
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
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_FP16_TO_FP32(qy[ib].d)));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                acc[iy] = _mm256_fmadd_ps(scales, _mm256_set1_ps(-8.f*GGML_FP16_TO_FP32(qy[ib].s)), acc[iy]);
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
static void mul_mat_q5_0_r4_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if constexpr (nrc_y == 1) {
        mul_mat_q5_0_r4_q8_1_avx2<1>(n, vx, bx, info, nrc_x);
    } else {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_1_x4> q8(info);
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
                _mm256_storeu_ps(d8+8*iy, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d)));
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
                auto dy = _mm512_set1_ps(GGML_FP16_TO_FP32(qy[ib].d));
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(GGML_FP16_TO_FP32(qy[ib].s)), acc[2*iy+1]);
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
static void mul_mat_q5_0_r4_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    mul_mat_q5_0_r4_q8_1_avx2<nrc_y>(n, vx, bx, info, nrc_x);
}
#endif

template <int nrc_y>
static void mul_mat_q6_0_r4_q8_1_avx2(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_1_x4> q8(info);
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
                auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d));
                _mm256_storeu_ps(d8 + 8*iy, _mm256_mul_ps(scales, mscale));
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
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_FP16_TO_FP32(qy[ib].d)));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
                acc[iy] = _mm256_fmadd_ps(scales, _mm256_set1_ps(-16.f*GGML_FP16_TO_FP32(qy[ib].s)), acc[iy]);
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
static void mul_mat_q6_0_r4_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if constexpr (nrc_y == 1) {
        mul_mat_q6_0_r4_q8_1_avx2<1>(n, vx, bx, info, nrc_x);
    } else {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_1_x4> q8(info);
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
                auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d));
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
                auto dy = _mm512_set1_ps(GGML_FP16_TO_FP32(qy[ib].d));
                acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(GGML_FP16_TO_FP32(qy[ib].s)), acc[2*iy+1]);
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
static void mul_mat_q6_0_r4_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    mul_mat_q6_0_r4_q8_1_avx2<nrc_y>(n, vx, bx, info, nrc_x);
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
    qx[0] = _mm256_loadu_si256((const __m256i *)x+0);
    qx[1] = _mm256_loadu_si256((const __m256i *)x+1);
    qx[2] = _mm256_loadu_si256((const __m256i *)x+2);
    qx[3] = _mm256_loadu_si256((const __m256i *)x+3);
    qx[4] = _mm256_loadu_si256((const __m256i *)x+4);
    qx[5] = _mm256_loadu_si256((const __m256i *)x+5);
    qx[6] = _mm256_loadu_si256((const __m256i *)x+6);
    qx[7] = _mm256_loadu_si256((const __m256i *)x+7);
    return qx_r8_q8_dot_product(qx, y);
}
template <int nrc_y>
static void mul_mat_q8_0_r8_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%16 == 0);
    Q8<nrc_y, block_q8_1_x4> q8(info);
    int nb = n / QK8_0;
    if constexpr (nrc_y == 1) {
        __m256 acc[2] = {};
        __m256i qx[8];
        float d8[8];
        for (int ix = 0; ix < nrc_x; ix += 8) {
            const block_q8_0_r8 * iq8 = (const block_q8_0_r8 *)((const char *)vx + ix*bx);
            for (int ib4 = 0; ib4 < nb/4; ++ib4) {
                _mm256_storeu_ps(d8, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8.y[0][ib4].d)));
                for (int k = 0; k < 4; ++k) {
                    auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[4*ib4+k].d));
                    auto sumi = q8_0_r8_dot_product((const uint8_t *)iq8[4*ib4+k].qs, q8.y[0][ib4].qs+32*k, qx);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(d8[k]));
                    acc[0] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[0]);
                    acc[1] = _mm256_fmadd_ps(scales, _mm256_set1_ps(d8[k+4]), acc[1]);
                }
            }
            if (4*(nb/4) < nb) {
                auto qy = (const block_q8_1 *)q8.y[0];
                for (int ib = 4*(nb/4); ib < nb; ++ib) {
                    auto scales = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)iq8[ib].d));
                    auto sumi = q8_0_r8_dot_product((const uint8_t *)iq8[ib].qs, qy[ib].qs, qx);
                    auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_FP16_TO_FP32(qy[ib].d)));
                    acc[0] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[0]);
                    acc[1] = _mm256_fmadd_ps(scales, _mm256_set1_ps(GGML_FP16_TO_FP32(qy[ib].s)), acc[1]);
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
                    _mm256_storeu_ps(d8+8*iy, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8.y[iy][ib4].d)));
                }
                for (int k = 0; k < 4; ++k) {
                    auto scales1  = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8l[4*ib4+k].d));
                    auto scales2  = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)q8h[4*ib4+k].d));
                    auto scales   = _mm512_insertf32x8(_mm512_castps256_ps512(scales1), scales2, 1);
                    for (int j = 0; j < 8; ++j) {
                        qx[j] = _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)q8l[4*ib4+k].qs+j)),
                                                                          _mm256_loadu_si256((const __m256i *)q8h[4*ib4+k].qs+j), 1);
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
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto qy = (const block_q8_1 *)q8.y[iy];
                    auto sumi = qx_r8_q8_dot_product(qx, qy[ib].qs);
                    auto dy = _mm512_set1_ps(GGML_FP16_TO_FP32(qy[ib].d));
                    acc[2*iy+0] = _mm512_fmadd_ps(_mm512_mul_ps(scales, dy), _mm512_cvtepi32_ps(sumi), acc[2*iy+0]);
                    acc[2*iy+1] = _mm512_fmadd_ps(scales, _mm512_set1_ps(GGML_FP16_TO_FP32(qy[ib].s)), acc[2*iy+1]);
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
static void mul_mat_q8_0_r8_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    Q8<nrc_y, block_q8_1_x4> q8(info);
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
                auto scales = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)q8.y[iy][ib4].d));
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
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(qy[ib].qs);
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_FP16_TO_FP32(qy[ib].d)));
                acc[iy] = _mm256_fmadd_ps(d4d8, _mm256_cvtepi32_ps(sumi), acc[iy]);
            }
            for (int j = 0; j < 4; ++j) {
                qx[j] = _mm256_loadu_si256((const __m256i *)iq8[ib].qs+4+j);
                sx[j] = _mm256_sign_epi8(qx[j], qx[j]);
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_1 *)q8.y[iy];
                auto sumi = dot(qy[ib].qs+16);
                auto d4d8 = _mm256_mul_ps(scales, _mm256_set1_ps(GGML_FP16_TO_FP32(qy[ib].d)));
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

#ifdef HAVE_FANCY_SIMD
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

#ifdef __AVX512BF16__
template <int nrc_y>
static void mul_mat_bf16_r16_bf16(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%16 == 0);
    const ggml_bf16_t * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const ggml_bf16_t *)info.src1_row(iy);
    for (int ix = 0; ix < nrc_x/32; ++ix) {
        __m512  acc[2*nrc_y] = {};
        __m512bh qx[8];
        const ggml_bf16_t * b8_1 = (const ggml_bf16_t *)((const char *)vx + (32*ix+ 0)*bx);
        const ggml_bf16_t * b8_2 = (const ggml_bf16_t *)((const char *)vx + (32*ix+16)*bx);
        for (int ib = 0; ib < n/8; ++ib) {
            qx[0] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_1+4*ib+0);
            qx[1] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_1+4*ib+1);
            qx[2] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_1+4*ib+2);
            qx[3] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_1+4*ib+3);
            qx[4] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_2+4*ib+0);
            qx[5] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_2+4*ib+1);
            qx[6] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_2+4*ib+2);
            qx[7] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_2+4*ib+3);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y128 = _mm_loadu_si128((const __m128i*)y[iy]+ib);
                //auto y = _mm512_broadcast_i32x4(y128);
                auto y256 = MM256_SET_M128I(y128, y128);
                auto y = _mm512_inserti32x8(_mm512_castsi256_si512(y256), y256, 1);
                acc[2*iy+0] = _mm512_dpbf16_ps(acc[2*iy+0], qx[0], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
                acc[2*iy+0] = _mm512_dpbf16_ps(acc[2*iy+0], qx[1], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
                acc[2*iy+0] = _mm512_dpbf16_ps(acc[2*iy+0], qx[2], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
                acc[2*iy+0] = _mm512_dpbf16_ps(acc[2*iy+0], qx[3], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
                acc[2*iy+1] = _mm512_dpbf16_ps(acc[2*iy+1], qx[4], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
                acc[2*iy+1] = _mm512_dpbf16_ps(acc[2*iy+1], qx[5], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
                acc[2*iy+1] = _mm512_dpbf16_ps(acc[2*iy+1], qx[6], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
                acc[2*iy+1] = _mm512_dpbf16_ps(acc[2*iy+1], qx[7], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(32*ix+ 0, iy, acc[2*iy+0]);
            info.store(32*ix+16, iy, acc[2*iy+1]);
        }
    }
    for (int ix = 32*(nrc_x/32); ix < nrc_x; ix += 16) {
        __m512  acc[nrc_y] = {};
        __m512bh qx[4];
        const ggml_bf16_t * b8 = (const ggml_bf16_t *)((const char *)vx + (ix+0)*bx);
        for (int ib = 0; ib < n/8; ++ib) {
            qx[0] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8+4*ib+0);
            qx[1] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8+4*ib+1);
            qx[2] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8+4*ib+2);
            qx[3] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8+4*ib+3);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y128 = _mm_loadu_si128((const __m128i*)y[iy]+ib);
                auto y256 = MM256_SET_M128I(y128, y128);
                auto y = _mm512_inserti32x8(_mm512_castsi256_si512(y256), y256, 1);
                acc[iy] = _mm512_dpbf16_ps(acc[iy], qx[0], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
                acc[iy] = _mm512_dpbf16_ps(acc[iy], qx[1], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
                acc[iy] = _mm512_dpbf16_ps(acc[iy], qx[2], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
                acc[iy] = _mm512_dpbf16_ps(acc[iy], qx[3], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, acc[iy]);
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

#ifdef HAVE_FANCY_SIMD
                auto q5vl = _mm256_shuffle_epi8(values[0], qx[0]);
                auto q5vh = _mm256_shuffle_epi8(values[1], qx[0]);
                qx[0] = _mm256_mask_blend_epi8(_mm256_cmpeq_epi8_mask(_mm256_and_si256(hb, _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01)), q5vl, q5vh);

                q5vl = _mm256_shuffle_epi8(values[0], qx[1]);
                q5vh = _mm256_shuffle_epi8(values[1], qx[1]);
                qx[1] = _mm256_mask_blend_epi8(_mm256_cmpeq_epi8_mask(_mm256_and_si256(hb, _mm256_set1_epi8(0x10)), _mm256_set1_epi8(0x10)), q5vl, q5vh);

                q5vl = _mm256_shuffle_epi8(values[0], qx[2]);
                q5vh = _mm256_shuffle_epi8(values[1], qx[2]);
                qx[2] = _mm256_mask_blend_epi8(_mm256_cmpeq_epi8_mask(_mm256_and_si256(hb, _mm256_set1_epi8(0x02)), _mm256_set1_epi8(0x02)), q5vl, q5vh);

                q5vl = _mm256_shuffle_epi8(values[0], qx[3]);
                q5vh = _mm256_shuffle_epi8(values[1], qx[3]);
                qx[3] = _mm256_mask_blend_epi8(_mm256_cmpeq_epi8_mask(_mm256_and_si256(hb, _mm256_set1_epi8(0x20)), _mm256_set1_epi8(0x20)), q5vl, q5vh);

                if constexpr (nrc_y == 1) {
                    auto shift = _mm256_and_si256(ms, _mm256_slli_epi16(extra, 1)); extra = _mm256_srli_epi16(extra, 1);
                    shift = _mm256_shuffle_epi8(shift, shift_shuffle);
                    qx[0] = _mm256_add_epi8(qx[0], shift);
                    qx[1] = _mm256_add_epi8(qx[1], shift);
                    qx[2] = _mm256_add_epi8(qx[2], shift);
                    qx[3] = _mm256_add_epi8(qx[3], shift);
                }
#else

                auto q5vl = _mm256_shuffle_epi8(values[0], qx[0]);
                auto q5vh = _mm256_shuffle_epi8(values[1], qx[0]);
                qx[0] = _mm256_blendv_epi8(q5vl, q5vh, _mm256_cmpeq_epi8(_mm256_and_si256(hb, _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01)));

                q5vl = _mm256_shuffle_epi8(values[0], qx[1]);
                q5vh = _mm256_shuffle_epi8(values[1], qx[1]);
                qx[1] = _mm256_blendv_epi8(q5vl, q5vh, _mm256_cmpeq_epi8(_mm256_and_si256(hb, _mm256_set1_epi8(0x10)), _mm256_set1_epi8(0x10)));

                q5vl = _mm256_shuffle_epi8(values[0], qx[2]);
                q5vh = _mm256_shuffle_epi8(values[1], qx[2]);
                qx[2] = _mm256_blendv_epi8(q5vl, q5vh, _mm256_cmpeq_epi8(_mm256_and_si256(hb, _mm256_set1_epi8(0x02)), _mm256_set1_epi8(0x02)));

                q5vl = _mm256_shuffle_epi8(values[0], qx[3]);
                q5vh = _mm256_shuffle_epi8(values[1], qx[3]);
                qx[3] = _mm256_blendv_epi8(q5vl, q5vh, _mm256_cmpeq_epi8(_mm256_and_si256(hb, _mm256_set1_epi8(0x20)), _mm256_set1_epi8(0x20)));

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

template <int nrc_y>
IQK_NOINLINE void mul_mat_iq1bn_q8_K64(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const int nb = n / QK_IQ1BN;
    Q8_K64<nrc_y> q8(info);
    DequantizerIQ1BN deq;
    __m256i accd[nrc_y];
    __m256i val[4];

#ifndef HAVE_FANCY_SIMD
    const auto m1_16  = _mm256_set1_epi16(1);
#endif

    const block_iq1_bn * x;
    const char * cx0 = (const char *)vx;
    float scale;
    ggml_half d16;

    for (int ix = 0; ix < nrc_x; ++ix) {

        const char * cx = cx0 + ix*bx;
        std::memcpy(&d16, cx, sizeof(d16));
        scale = GGML_FP16_TO_FP32(d16);
        cx += sizeof(d16);
        x = (const block_iq1_bn *)cx;

        if constexpr (nrc_y == 1) {
            __m256i acc1 = _mm256_setzero_si256(), acc2 = _mm256_setzero_si256();
            for (int i = 0; i < nb/2; ++i) {
                deq.prepare_iq1bn_quants(x + 2*i + 0, val[0], val[1]);
                deq.prepare_iq1bn_quants(x + 2*i + 1, val[2], val[3]);
#ifdef HAVE_FANCY_SIMD
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
#ifdef HAVE_FANCY_SIMD
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
#ifdef HAVE_FANCY_SIMD
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
            info.store(ix, iy, scale*hsum_float_4(sumf));
        }

    }
}

struct DequantizeIQ2BN final : public BaseDequantizer<block_iq2_bn, true> {
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

#ifndef HAVE_FANCY_SIMD
    const auto m1_16  = _mm256_set1_epi16(1);
#endif

    for (int ix = 0; ix < nrc_x; ++ix) {

        deq.new_row(ix);

        if constexpr (nrc_y == 1) {
            __m256i acc[2] = {};
            for (int i = 0; i < nb/2; ++i) {
                deq.prepare4(i, val);
#ifdef HAVE_FANCY_SIMD
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
#ifdef HAVE_FANCY_SIMD
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
#ifdef HAVE_FANCY_SIMD
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
            info.store(ix, iy, deq.d*hsum_float_4(sumf));
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

struct MinusType0 {
    inline __m256 compute(__m128 d, int) const { return _mm256_set_m128(d, d); }
    inline float compute(float d, int) const { return d; }
    inline float result(__m256 acc, int) const { return hsum_float_8(acc); }
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
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, accm.result(acc[iy], iy));
        }
    }
};

template <int nrc_y, bool is_multiple_of_4>
using AccumType0 = AccumT<MinusType0, nrc_y, is_multiple_of_4>;

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
    const __m256i values = load_iq4nl_values_256();
    inline __m256i dequant(const block_iq4_nl * x) const {
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

struct Q8_0_x4_Unpacker_256 {
    using Sum4T = Sum4TypeQ80;
    inline static int block_size() { return QK8_0; }
    Q8_0_x4_Unpacker_256(const void * vx, size_t bx) : cx_0((const char *)vx), x((const block_q8_0_x4 *)cx_0), bx(bx) {}

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

#ifdef HAVE_FANCY_SIMD
struct Q8_0_x4_Unpacker_512 {
    using Sum4T = Sum4TypeQ81;
    inline static int block_size() { return QK8_0; }
    Q8_0_x4_Unpacker_512(const void * vx, size_t bx) : cx_0((const char *)vx), x((const block_q8_0_x4 *)cx_0), bx(bx) {}

    const char * cx_0;
    const block_q8_0_x4 * x;
    size_t       bx;
    const __m128 min = _mm_set1_ps(-128.f);

    __m256i qx[4];

    inline const __m256i* quants() const { return qx; }

    inline void set_row(int ix) { x = (const block_q8_0_x4 *)(cx_0 + ix*bx); }

    inline auto set_block_4(int i) {
        auto scales = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)x[i].d));
        for (int j = 0; j < 4; ++j) {
            qx[j] = _mm256_loadu_si256((const __m256i *)x[i].qs + j);
            qx[j] = _mm256_xor_si256(qx[j], _mm256_set1_epi8(-128));
        }
        return _mm256_set_m128(_mm_mul_ps(scales, min), scales);
    }
    inline auto set_block(int i) {
        auto q8 = (const block_q8_0 *)(x + i);
        qx[0] = _mm256_loadu_si256((const __m256i *)q8->qs);
        qx[0] = _mm256_xor_si256(qx[0], _mm256_set1_epi8(-128));
        float d = GGML_FP16_TO_FP32(q8->d);
        return std::make_pair(d, -128.f*d);
    }
};
using Q8_0_x4_Unpacker = Q8_0_x4_Unpacker_512;
#else
using Q8_0_x4_Unpacker = Q8_0_x4_Unpacker_256;
#endif

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
struct IQ4_NL_Unpacker final : public Q_Unpacker<block_iq4_nl, ScaleHelperQ_0_1<128>, IQ4_NL_Dequantizer> {
    IQ4_NL_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ81;
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
struct Q6_0_1_Unpacker final : public Q_Unpacker<block_q6_0, ScaleHelperQ_0_1<32>, Q6_0_1_Dequantizer> {
    Q6_0_1_Unpacker(const void * vx, size_t bx) : Q_Unpacker(vx, bx) {}
    using Sum4T = Sum4TypeQ81;
    inline static int block_size() { return QK6_0; }
};

// float matrices - we handle f16, bf16 (if native bf16 support is available) and f32, but only to f32 result

struct QFBase {
#ifdef __AVX512F__
    constexpr static int k_step = 16;
    using Data = __m512;
    using Acc  = __m512;
    static inline Data load(const ggml_half * x) { return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)x)); }
    static inline Data load(const float * x) { return _mm512_loadu_ps(x); }
    static inline Data load(const ggml_bf16_t * x) {
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)x)), 16));
    }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) {
        return _mm512_fmadd_ps(y, x, prev);
    }
    static inline Acc acc_first(const Data& y, const Data& x) {
        return _mm512_mul_ps(y, x);
    }
    static inline Acc add(Acc x, Acc y) { return _mm512_add_ps(x, y); }
    static inline float hsum(Acc acc) {
        return _mm512_reduce_add_ps(acc);
    }
    template <typename Float>
    static inline Data load4Floats(const Float * x) {
        return _mm512_insertf32x4(_mm512_setzero_ps(), load128(x), 0);
    }
    static inline Acc acc_r4(Acc acc, const Data * xv, const Data& yv) {
        acc = _mm512_fmadd_ps(xv[0], _mm512_shuffle_ps(yv, yv, 0x00), acc);
        acc = _mm512_fmadd_ps(xv[1], _mm512_shuffle_ps(yv, yv, 0x55), acc);
        acc = _mm512_fmadd_ps(xv[2], _mm512_shuffle_ps(yv, yv, 0xaa), acc);
        acc = _mm512_fmadd_ps(xv[3], _mm512_shuffle_ps(yv, yv, 0xff), acc);
        return acc;
    }
    static inline Acc acc_r4_first(const Data * xv, const Data& yv) {
        auto acc = _mm512_mul_ps(xv[0], _mm512_shuffle_ps(yv, yv, 0x00));
        acc = _mm512_fmadd_ps(xv[1], _mm512_shuffle_ps(yv, yv, 0x55), acc);
        acc = _mm512_fmadd_ps(xv[2], _mm512_shuffle_ps(yv, yv, 0xaa), acc);
        acc = _mm512_fmadd_ps(xv[3], _mm512_shuffle_ps(yv, yv, 0xff), acc);
        return acc;
    }
    static inline __m128 hsum_r4(Acc acc) {
        auto sum1 = _mm_add_ps(_mm512_extractf32x4_ps(acc, 0), _mm512_extractf32x4_ps(acc, 1));
        auto sum2 = _mm_add_ps(_mm512_extractf32x4_ps(acc, 2), _mm512_extractf32x4_ps(acc, 3));
        return _mm_add_ps(sum1, sum2);
    }
#else
    constexpr static int k_step = 8;
    using Data = __m256;
    using Acc  = __m256;
    static inline Data load(const ggml_half * x) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)x)); }
    static inline Data load(const float * x) { return _mm256_loadu_ps(x); }
    static inline Data load(const ggml_bf16_t * x) {
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)x)), 16));
    }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) {
        return _mm256_fmadd_ps(y, x, prev);
    }
    static inline Acc add(Acc x, Acc y) { return _mm256_add_ps(x, y); }
    static inline Acc acc_r4(Acc acc, const Data * xv, const Data& yv) {
        acc = _mm256_fmadd_ps(xv[0], _mm256_shuffle_ps(yv, yv, 0x00), acc);
        acc = _mm256_fmadd_ps(xv[1], _mm256_shuffle_ps(yv, yv, 0x55), acc);
        acc = _mm256_fmadd_ps(xv[2], _mm256_shuffle_ps(yv, yv, 0xaa), acc);
        acc = _mm256_fmadd_ps(xv[3], _mm256_shuffle_ps(yv, yv, 0xff), acc);
        return acc;
    }
    static inline Acc acc_r4_first(const Data * xv, const Data& yv) {
        auto acc = _mm256_mul_ps(xv[0], _mm256_shuffle_ps(yv, yv, 0x00));
        acc = _mm256_fmadd_ps(xv[1], _mm256_shuffle_ps(yv, yv, 0x55), acc);
        acc = _mm256_fmadd_ps(xv[2], _mm256_shuffle_ps(yv, yv, 0xaa), acc);
        acc = _mm256_fmadd_ps(xv[3], _mm256_shuffle_ps(yv, yv, 0xff), acc);
        return acc;
    }
    static inline Acc acc_first(const Data& y, const Data& x) {
        return _mm256_mul_ps(y, x);
    }
    static inline float hsum(Acc acc) {
        return hsum_float_8(acc);
    }
    static inline __m128 hsum_r4(Acc acc) {
        return _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    }
    template <typename Float>
    static inline Data load4Floats(const Float * x) {
        return _mm256_insertf128_ps(_mm256_setzero_ps(), load128(x), 0);
    }
#endif
    static inline __m128 load128(const ggml_half * x) { return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)x)); }
    static inline __m128 load128(const float * x) { return _mm_loadu_ps(x); }
    static inline __m128 load128(const ggml_bf16_t * x) {
        return _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*)x)), 16));
    }
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
    IQK_ALWAYS_INLINE void load_r4(int ix, int i, Data * xv) const {
        xv[0] = load1(ix+0, i);
        xv[1] = load1(ix+1, i);
        xv[2] = load1(ix+2, i);
        xv[3] = load1(ix+3, i);
#ifdef __AVX512F__
        auto t0 = _mm512_unpacklo_ps(xv[0], xv[1]);
        auto t1 = _mm512_unpacklo_ps(xv[2], xv[3]);
        auto t2 = _mm512_unpackhi_ps(xv[0], xv[1]);
        auto t3 = _mm512_unpackhi_ps(xv[2], xv[3]);
        xv[0] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t0), _mm512_castps_pd(t1)));
        xv[1] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t0), _mm512_castps_pd(t1)));
        xv[2] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t2), _mm512_castps_pd(t3)));
        xv[3] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t2), _mm512_castps_pd(t3)));
#else
        auto t0 = _mm256_unpacklo_ps(xv[0], xv[1]);
        auto t1 = _mm256_unpacklo_ps(xv[2], xv[3]);
        auto t2 = _mm256_unpackhi_ps(xv[0], xv[1]);
        auto t3 = _mm256_unpackhi_ps(xv[2], xv[3]);
        xv[0] = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t1)));
        xv[1] = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t1)));
        xv[2] = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t2), _mm256_castps_pd(t3)));
        xv[3] = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t2), _mm256_castps_pd(t3)));
#endif
    }
    const Float * y[nrc];
};

// TBD if we want this
//template <typename Qy, typename Qx>
//IQK_NOINLINE void mul_mat_Qx_Qy_Mx1(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
//    static_assert(Qy::nrc == 1);
//    int nb = n/QFBase::k_step;
//    int nb4 = n/4;
//    Qy y(info);
//    Qx x(cx + ix0*bx, bx);
//    QFBase::Data xv[2*Qx::nrc];
//    QFBase::Acc  acc[2*Qx::nrc];
//    auto yv1 = y.load1(0, 0);
//    auto yv2 = y.load1(0, 1);
//    for (int ix = 0; ix < Qx::nrc; ++ix) {
//        xv[2*ix+0] = x.load1(ix, 0);
//        xv[2*ix+1] = x.load1(ix, 1);
//        acc[2*ix+0] = QFBase::acc_first(yv1, xv[2*ix+0]);
//        acc[2*ix+1] = QFBase::acc_first(yv2, xv[2*ix+1]);
//    }
//    for (int i = 1; i < nb/2; ++i) {
//        yv1 = y.load1(0, 2*i+0);
//        yv2 = y.load1(0, 2*i+1);
//        for (int ix = 0; ix < Qx::nrc; ++ix) {
//            xv[2*ix+0] = x.load1(ix, 2*i+0);
//            xv[2*ix+1] = x.load1(ix, 2*i+1);
//            acc[2*ix+0] = QFBase::acc(acc[2*ix+0], yv1, xv[2*ix+0]);
//            acc[2*ix+1] = QFBase::acc(acc[2*ix+1], yv2, xv[2*ix+1]);
//        }
//    }
//    for (int i = (QFBase::k_step/4)*nb; i < nb4; ++i) {
//        yv1 = y.load_tail(0, i);
//        for (int ix = 0; ix < Qx::nrc; ++ix) {
//            xv[ix] = x.load_tail(ix, i);
//            acc[2*ix+0] = QFBase::acc(acc[2*ix+0], yv1, xv[ix]);
//        }
//    }
//    for (int ix = 0; ix < Qx::nrc; ++ix) info.store(ix0+ix, 0, QFBase::hsum(QFBase::add(acc[2*ix+0], acc[2*ix+1])));
//}

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

template <typename Qy, typename Qx>
inline void mul_mat_Qx_Qy_MxN_fa(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    int nb = n/QFBase::k_step;
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
    for (int iy = 0; iy < Qy::nrc; ++iy) for (int ix = 0; ix < Qx::nrc; ++ix) info.store(ix0+ix, iy, QFBase::hsum(acc[Qx::nrc*iy+ix]));
}

template <typename Qy, typename Qx>
inline void mul_mat_Qx_Qy_MxN_fa4(int D, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    static_assert(Qx::nrc%4 == 0);
    int nb = D/QFBase::k_step;
    Qy y(info);
    Qx x(cx + ix0*bx, bx);
    QFBase::Data xv[Qx::nrc];
    QFBase::Acc  acc[Qx::nrc*Qy::nrc/4] = {};
    for (int i = 0; i < nb; ++i) {
        for (int ix = 0; ix < Qx::nrc/4; ++ix) x.load_r4(4*ix, i, xv + 4*ix);
        for (int iy = 0; iy < Qy::nrc; ++iy) {
            auto yv = y.load1(iy, i);
            for (int ix = 0; ix < Qx::nrc/4; ++ix) acc[ix*Qy::nrc + iy] = QFBase::acc_r4(acc[ix*Qy::nrc + iy], xv + 4*ix, yv);
        }
    }
    for (int iy = 0; iy < Qy::nrc; ++iy) {
        for (int ix = 0; ix < Qx::nrc/4; ++ix) info.store(ix0+4*ix, iy, QFBase::hsum_r4(acc[ix*Qy::nrc + iy]));
    }
}

// This will handle any of f16 x f32, f32 x f16, f16 x f16, f32 x f32, with computations done
// in f32 (i.e., f16 is first converted to f32). It is easy to extend to computations done in
// f16, but I don't have a CPU capable of f16 vector arithmetic, so not doing it for now.
template <int nrc_y, typename FloatX, typename FloatY>
void mul_mat_fX_fY_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const char * cx = (const char *)vx;
    // TBD if we want this
    //if constexpr (nrc_y == 1) {
    //    constexpr int k_nx = 2;
    //    for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
    //        mul_mat_Qx_Qy_Mx1<QFT<FloatY, nrc_y>, QFT<FloatX, k_nx>>(n, cx, bx, ix*k_nx, info);
    //    }
    //    if (int lastx = k_nx*(nrc_x/k_nx); lastx < nrc_x) {
    //        int nx = nrc_x - lastx;
    //        switch (nx) {
    //            case 1: mul_mat_Qx_Qy_Mx1<QFT<FloatY, nrc_y>, QFT<FloatX, 1>>(n, cx, bx, lastx, info); break;
    //            case 2: mul_mat_Qx_Qy_Mx1<QFT<FloatY, nrc_y>, QFT<FloatX, 2>>(n, cx, bx, lastx, info); break;
    //            case 3: mul_mat_Qx_Qy_Mx1<QFT<FloatY, nrc_y>, QFT<FloatX, 3>>(n, cx, bx, lastx, info); break;
    //        }
    //        //mul_mat_Qx_Qy_Mx1<QFT<FloatY, nrc_y>, QFT<FloatX, 1>>(n, cx, bx, lastx, info);
    //    }
    //    return;
    //}
#ifdef __AVX512F__
    constexpr int k_nx = 5;
#else
    constexpr int k_nx = nrc_y == 1 ? 4 : 2;
#endif
    for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
        mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, k_nx>>(n, cx, bx, ix*k_nx, info);
    }
    int last_x = k_nx*(nrc_x/k_nx);
    if (last_x == nrc_x) return;
    int nx = nrc_x - last_x;
#ifdef __AVX512F__
    switch (nx) {
        case 1: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 1>>(n, cx, bx, last_x, info); break;
        case 2: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 2>>(n, cx, bx, last_x, info); break;
        case 3: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 3>>(n, cx, bx, last_x, info); break;
        case 4: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 4>>(n, cx, bx, last_x, info); break;
    }
#else
    if constexpr (nrc_y == 1) {
        switch (nx) {
            case 1: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 1>>(n, cx, bx, last_x, info); break;
            case 2: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 2>>(n, cx, bx, last_x, info); break;
            case 3: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 3>>(n, cx, bx, last_x, info); break;
        }
    } else {
        switch (nx) {
            case 1: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 1>>(n, cx, bx, last_x, info); break;
        }
    }
#endif
}

#ifdef __AVX512BF16__
struct QFBaseBF16 {
    constexpr static int k_step = 32;
    using Data = __m512bh;
    using Acc  = __m512;
    static inline Data load(const ggml_bf16_t * x) { return __m512bh(_mm512_loadu_si512((const __m512i *)x)); }
    //static inline Acc acc(Acc prev, const Data& y, const Data& x) {
    static inline Acc acc(Acc prev, Data y, Data x) {
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
                      std::is_same_v<Dequantizer, Q8_0_Unpacker>) {
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
                           std::is_same_v<Dequantizer, Q5_0_1_Unpacker> || std::is_same_v<Dequantizer, IQ4_NL_Unpacker> ||
                           std::is_same_v<Dequantizer, Q6_0_1_Unpacker>) {
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
                          std::is_same_v<Dequantizer, DequantizerIQ3K> ||
                          std::is_same_v<Dequantizer, DequantizerIQ4XS>||
                          std::is_same_v<Dequantizer, DequantizerIQ4KS>||
                          std::is_same_v<Dequantizer, DequantizerIQ4KSS>) {
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
void set_mul_mat_bf16_r16(MulMat& mm) {
    for (auto& f : mm.funcs) f = nullptr;
    mm.funcs[0] = mul_mat_bf16_r16_bf16<1>;
    mm.funcs[1] = mul_mat_bf16_r16_bf16<2>;
    mm.funcs[2] = mul_mat_bf16_r16_bf16<3>;
    mm.funcs[3] = mul_mat_bf16_r16_bf16<4>;
    mm.funcs[4] = mul_mat_bf16_r16_bf16<5>;
    mm.funcs[5] = mul_mat_bf16_r16_bf16<6>;
    mm.funcs[6] = mul_mat_bf16_r16_bf16<7>;
    mm.funcs[7] = mul_mat_bf16_r16_bf16<8>;
}
#endif

bool MulMat::prepare(int typeA, int typeB, int ne00, MulMat& mm, int Ny) {

    (void)Ny;

    if (typeA == GGML_TYPE_BF16) {
        if (ne00 % 32) return false;
        switch (typeB) {
#ifdef __AVX512BF16__
            case GGML_TYPE_BF16: set_mul_mat_bf16(mm); break;
#else
            case GGML_TYPE_BF16: set_mul_mat_f<ggml_bf16_t, ggml_bf16_t>(mm); break;
            case GGML_TYPE_F32:  set_mul_mat_f<ggml_bf16_t, float>(mm);       break;
#endif
            default: return false;
        }
        return true;
    }

    if (typeA == GGML_TYPE_BF16_R16) {
        if (ne00 % 16) return false;
        switch (typeB) {
#ifdef __AVX512BF16__
            case GGML_TYPE_BF16: set_mul_mat_bf16_r16(mm); break;
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
        case GGML_TYPE_IQ4_KS:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ4KS>(mm);
            break;
        case GGML_TYPE_IQ4_KSS:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ4KSS>(mm);
            break;
        case GGML_TYPE_IQ2_K:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ2K>(mm);
            break;
        case GGML_TYPE_IQ2_KS:
            assert (ne00 % QK_K == 0);
            MulMat::set_functions<DequantizerIQ2KS>(mm);
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
            mm.funcs[0] = mul_mat_iq1bn_q8_K64<1>;
            mm.funcs[1] = mul_mat_iq1bn_q8_K64<2>;
            mm.funcs[2] = mul_mat_iq1bn_q8_K64<3>;
            mm.funcs[3] = mul_mat_iq1bn_q8_K64<4>;
            mm.funcs[4] = mul_mat_iq1bn_q8_K64<5>;
            mm.funcs[5] = mul_mat_iq1bn_q8_K64<6>;
            mm.funcs[6] = mul_mat_iq1bn_q8_K64<7>;
            mm.funcs[7] = mul_mat_iq1bn_q8_K64<8>;
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
        case GGML_TYPE_IQ2_BN_R4:
            assert (ne00 % QK_IQ1BN == 0);
            mm.funcs[0] = mul_mat_iq2_bn_r4_q8_k16<1>;
            mm.funcs[1] = mul_mat_iq2_bn_r4_q8_k16<2>;
            mm.funcs[2] = mul_mat_iq2_bn_r4_q8_k16<3>;
            mm.funcs[3] = mul_mat_iq2_bn_r4_q8_k16<4>;
            mm.funcs[4] = mul_mat_iq2_bn_r4_q8_k16<5>;
            mm.funcs[5] = mul_mat_iq2_bn_r4_q8_k16<6>;
//#ifdef HAVE_FANCY_SIMD
            mm.funcs[6] = mul_mat_iq2_bn_r4_q8_k16<7>;
            mm.funcs[7] = mul_mat_iq2_bn_r4_q8_k16<8>;
//#endif
            expected_typeB = GGML_TYPE_Q8_K16;
            break;
        case GGML_TYPE_Q4_0:
            assert (ne00 % QK4_0 == 0);
            MulMat::set_functions<Q4_0_1_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_Q4_1:
            assert (ne00 % QK4_1 == 0);
            MulMat::set_functions<Q4_1_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_Q5_0:
            assert (ne00 % QK5_0 == 0);
            MulMat::set_functions<Q5_0_1_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_Q5_1:
            assert (ne00 % QK5_1 == 0);
            MulMat::set_functions<Q5_1_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_Q6_0:
            assert (ne00 % QK6_0 == 0);
            MulMat::set_functions<Q6_0_1_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_Q8_0:
            assert (ne00 % QK8_0 == 0);
#ifdef HAVE_FANCY_SIMD
            MulMat::set_functions<Q8_0_1_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_1_X4;
#else
            MulMat::set_functions<Q8_0_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_0_X4;
#endif
            break;
        case GGML_TYPE_IQ4_NL:
            assert (ne00 % QK4_NL == 0);
            MulMat::set_functions<IQ4_NL_Unpacker>(mm);
            expected_typeB = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_IQ4_NL_R4:
            assert (ne00 % QK4_NL == 0);
            mm.funcs[0] = mul_mat_iq4_nl_r4_q8_1<1>;
            mm.funcs[1] = mul_mat_iq4_nl_r4_q8_1<2>;
            mm.funcs[2] = mul_mat_iq4_nl_r4_q8_1<3>;
            mm.funcs[3] = mul_mat_iq4_nl_r4_q8_1<4>;
            mm.funcs[4] = mul_mat_iq4_nl_r4_q8_1<5>;
            mm.funcs[5] = mul_mat_iq4_nl_r4_q8_1<6>;
            mm.funcs[6] = mul_mat_iq4_nl_r4_q8_1<7>;
            mm.funcs[7] = mul_mat_iq4_nl_r4_q8_1<8>;
            expected_typeB = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_IQ4_XS_R8:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_iq4_xs_r8_q8_k<1>;
            mm.funcs[1] = mul_mat_iq4_xs_r8_q8_k<2>;
            mm.funcs[2] = mul_mat_iq4_xs_r8_q8_k<3>;
            mm.funcs[3] = mul_mat_iq4_xs_r8_q8_k<4>;
            mm.funcs[4] = mul_mat_iq4_xs_r8_q8_k<5>;
            mm.funcs[5] = mul_mat_iq4_xs_r8_q8_k<6>;
            mm.funcs[6] = mul_mat_iq4_xs_r8_q8_k<7>;
            mm.funcs[7] = mul_mat_iq4_xs_r8_q8_k<8>;
            expected_typeB = GGML_TYPE_Q8_K32;
            break;
        case GGML_TYPE_IQ4_KS_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_iq4_ks_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_iq4_ks_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_iq4_ks_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_iq4_ks_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_iq4_ks_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_iq4_ks_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_iq4_ks_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_iq4_ks_r4_q8_k<8>;
#ifndef HAVE_FANCY_SIMD
            // For some reason Zen4 does not like this particular function
            mm.func16 = mul_mat_iq4_ks_r4_q8_k<16>;
#endif
            expected_typeB = GGML_TYPE_Q8_K32;
            break;
        case GGML_TYPE_IQ2_XXS_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_iq2_xxs_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_iq2_xxs_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_iq2_xxs_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_iq2_xxs_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_iq2_xxs_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_iq2_xxs_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_iq2_xxs_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_iq2_xxs_r4_q8_k<8>;
            mm.func16 = mul_mat_iq2_xxs_r4_q8_k<16>;
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ2_XS_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_iq2_xs_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_iq2_xs_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_iq2_xs_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_iq2_xs_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_iq2_xs_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_iq2_xs_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_iq2_xs_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_iq2_xs_r4_q8_k<8>;
#ifndef HAVE_FANCY_SIMD
            // For some reason Zen4 does not like this particular function
            mm.func16 = mul_mat_iq2_xs_r4_q8_k_16;
#endif
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ2_S_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_iq2_s_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_iq2_s_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_iq2_s_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_iq2_s_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_iq2_s_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_iq2_s_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_iq2_s_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_iq2_s_r4_q8_k<8>;
            mm.func16 = mul_mat_iq2_s_r4_q8_k_16;
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ3_XXS_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_iq3_xxs_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_iq3_xxs_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_iq3_xxs_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_iq3_xxs_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_iq3_xxs_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_iq3_xxs_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_iq3_xxs_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_iq3_xxs_r4_q8_k<8>;
            mm.func16 = mul_mat_iq3_xxs_r4_q8_k<16>;
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ3_S_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_iq3_s_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_iq3_s_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_iq3_s_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_iq3_s_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_iq3_s_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_iq3_s_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_iq3_s_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_iq3_s_r4_q8_k<8>;
            mm.func16 = mul_mat_iq3_s_r4_q8_k<16>;
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_Q2_K_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_q2_k_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_q2_k_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_q2_k_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_q2_k_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_q2_k_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_q2_k_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_q2_k_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_q2_k_r4_q8_k<8>;
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_Q3_K_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_q3_k_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_q3_k_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_q3_k_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_q3_k_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_q3_k_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_q3_k_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_q3_k_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_q3_k_r4_q8_k<8>;
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_Q4_K_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_q4_k_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_q4_k_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_q4_k_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_q4_k_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_q4_k_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_q4_k_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_q4_k_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_q4_k_r4_q8_k<8>;
            expected_typeB = GGML_TYPE_Q8_K32;
            break;
        case GGML_TYPE_Q5_K_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_q5_k_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_q5_k_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_q5_k_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_q5_k_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_q5_k_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_q5_k_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_q5_k_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_q5_k_r4_q8_k<8>;
            expected_typeB = GGML_TYPE_Q8_K32;
            break;
        case GGML_TYPE_Q6_K_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_q6_k_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_q6_k_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_q6_k_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_q6_k_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_q6_k_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_q6_k_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_q6_k_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_q6_k_r4_q8_k<8>;
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_Q8_K_R8:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_q8_k_r8_q8_k<1>;
            mm.funcs[1] = mul_mat_q8_k_r8_q8_k<2>;
            mm.funcs[2] = mul_mat_q8_k_r8_q8_k<3>;
            mm.funcs[3] = mul_mat_q8_k_r8_q8_k<4>;
            mm.funcs[4] = mul_mat_q8_k_r8_q8_k<5>;
            mm.funcs[5] = mul_mat_q8_k_r8_q8_k<6>;
            mm.funcs[6] = mul_mat_q8_k_r8_q8_k<7>;
            mm.funcs[7] = mul_mat_q8_k_r8_q8_k<8>;
#ifdef HAVE_FANCY_SIMD
            mm.func16 = mul_mat_q8_k_r8_q8_k<16>;
#endif
            expected_typeB = GGML_TYPE_Q8_KR8;
            break;
        case GGML_TYPE_Q8_KV:
            assert (ne00 % 32 == 0);
            mm.funcs[0] = mul_mat_q8_KV_q8_KV_1<1>;
            mm.funcs[1] = mul_mat_q8_KV_q8_KV<2>;
            mm.funcs[2] = mul_mat_q8_KV_q8_KV<3>;
            mm.funcs[3] = mul_mat_q8_KV_q8_KV<4>;
            mm.funcs[4] = mul_mat_q8_KV_q8_KV<5>;
            mm.funcs[5] = mul_mat_q8_KV_q8_KV<6>;
            mm.funcs[6] = mul_mat_q8_KV_q8_KV<7>;
            mm.funcs[7] = mul_mat_q8_KV_q8_KV<8>;
#ifdef HAVE_FANCY_SIMD
            mm.func16 = mul_mat_q8_KV_q8_KV<16>;
#endif
            expected_typeB = GGML_TYPE_Q8_KV;
            break;
        case GGML_TYPE_Q8_KV_R8:
            assert (ne00 % 32 == 0);
            mm.funcs[0] = mul_mat_q8_KV_r8_q8_KV<1>;
            mm.funcs[1] = mul_mat_q8_KV_r8_q8_KV<2>;
            mm.funcs[2] = mul_mat_q8_KV_r8_q8_KV<3>;
            mm.funcs[3] = mul_mat_q8_KV_r8_q8_KV<4>;
            mm.funcs[4] = mul_mat_q8_KV_r8_q8_KV<5>;
            mm.funcs[5] = mul_mat_q8_KV_r8_q8_KV<6>;
            mm.funcs[6] = mul_mat_q8_KV_r8_q8_KV<7>;
            mm.funcs[7] = mul_mat_q8_KV_r8_q8_KV<8>;
            expected_typeB = GGML_TYPE_Q8_KV;
            break;
        case GGML_TYPE_IQ4_K_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_iq4_k_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_iq4_k_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_iq4_k_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_iq4_k_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_iq4_k_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_iq4_k_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_iq4_k_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_iq4_k_r4_q8_k<8>;
            mm.func16  = mul_mat_iq4_k_r4_q8_k<16>;
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ5_K_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_iq5_k_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_iq5_k_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_iq5_k_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_iq5_k_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_iq5_k_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_iq5_k_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_iq5_k_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_iq5_k_r4_q8_k<8>;
            mm.func16 = mul_mat_iq5_k_r4_q8_k<16>;
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ2_K_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_iq2_k_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_iq2_k_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_iq2_k_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_iq2_k_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_iq2_k_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_iq2_k_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_iq2_k_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_iq2_k_r4_q8_k<8>;
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ3_K_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_iq3_k_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_iq3_k_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_iq3_k_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_iq3_k_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_iq3_k_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_iq3_k_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_iq3_k_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_iq3_k_r4_q8_k<8>;
#ifdef HAVE_FANCY_SIMD
            mm.func16 = mul_mat_iq3_k_r4_q8_k<16>;
#endif
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_Q4_0_R8:
            assert (ne00 % QK4_NL == 0);
            mm.funcs[0] = mul_mat_q4_0_r8_q8_1<1>;
            mm.funcs[1] = mul_mat_q4_0_r8_q8_1<2>;
            mm.funcs[2] = mul_mat_q4_0_r8_q8_1<3>;
            mm.funcs[3] = mul_mat_q4_0_r8_q8_1<4>;
            mm.funcs[4] = mul_mat_q4_0_r8_q8_1<5>;
            mm.funcs[5] = mul_mat_q4_0_r8_q8_1<6>;
            mm.funcs[6] = mul_mat_q4_0_r8_q8_1<7>;
            mm.funcs[7] = mul_mat_q4_0_r8_q8_1<8>;
#ifdef HAVE_FANCY_SIMD
            mm.func16 = mul_mat_q4_0_r8_q8_1<16>;
#endif
            expected_typeB = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_Q5_0_R4:
            assert (ne00 % QK4_NL == 0);
            mm.funcs[0] = mul_mat_q5_0_r4_q8_1<1>;
            mm.funcs[1] = mul_mat_q5_0_r4_q8_1<2>;
            mm.funcs[2] = mul_mat_q5_0_r4_q8_1<3>;
            mm.funcs[3] = mul_mat_q5_0_r4_q8_1<4>;
            mm.funcs[4] = mul_mat_q5_0_r4_q8_1<5>;
            mm.funcs[5] = mul_mat_q5_0_r4_q8_1<6>;
            mm.funcs[6] = mul_mat_q5_0_r4_q8_1<7>;
            mm.funcs[7] = mul_mat_q5_0_r4_q8_1<8>;
            expected_typeB = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_Q6_0_R4:
            assert (ne00 % QK4_NL == 0);
            mm.funcs[0] = mul_mat_q6_0_r4_q8_1<1>;
            mm.funcs[1] = mul_mat_q6_0_r4_q8_1<2>;
            mm.funcs[2] = mul_mat_q6_0_r4_q8_1<3>;
            mm.funcs[3] = mul_mat_q6_0_r4_q8_1<4>;
            mm.funcs[4] = mul_mat_q6_0_r4_q8_1<5>;
            mm.funcs[5] = mul_mat_q6_0_r4_q8_1<6>;
            mm.funcs[6] = mul_mat_q6_0_r4_q8_1<7>;
            mm.funcs[7] = mul_mat_q6_0_r4_q8_1<8>;
            expected_typeB = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_Q8_0_R8:
            assert (ne00 % QK4_NL == 0);
            mm.funcs[0] = mul_mat_q8_0_r8_q8_1<1>;
            mm.funcs[1] = mul_mat_q8_0_r8_q8_1<2>;
            mm.funcs[2] = mul_mat_q8_0_r8_q8_1<3>;
            mm.funcs[3] = mul_mat_q8_0_r8_q8_1<4>;
            mm.funcs[4] = mul_mat_q8_0_r8_q8_1<5>;
            mm.funcs[5] = mul_mat_q8_0_r8_q8_1<6>;
            mm.funcs[6] = mul_mat_q8_0_r8_q8_1<7>;
            mm.funcs[7] = mul_mat_q8_0_r8_q8_1<8>;
            expected_typeB = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_IQ1_S:
            mm.funcs[0] = mul_mat_iq1_s_q8_K<1>;
            mm.funcs[1] = mul_mat_iq1_s_q8_K<2>;
            mm.funcs[2] = mul_mat_iq1_s_q8_K<3>;
            mm.funcs[3] = mul_mat_iq1_s_q8_K<4>;
            mm.funcs[4] = mul_mat_iq1_s_q8_K<5>;
            mm.funcs[5] = mul_mat_iq1_s_q8_K<6>;
            mm.funcs[6] = mul_mat_iq1_s_q8_K<7>;
            mm.funcs[7] = mul_mat_iq1_s_q8_K<8>;
#ifdef HAVE_FANCY_SIMD
            mm.func16 = mul_mat_iq1_s_q8_K<16>;
#endif
            expected_typeB = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ1_S_R4:
            assert (ne00 % QK4_NL == 0);
            mm.funcs[0] = mul_mat_iq1_s_r4_q8_1<1>;
            mm.funcs[1] = mul_mat_iq1_s_r4_q8_1<2>;
            mm.funcs[2] = mul_mat_iq1_s_r4_q8_1<3>;
            mm.funcs[3] = mul_mat_iq1_s_r4_q8_1<4>;
            mm.funcs[4] = mul_mat_iq1_s_r4_q8_1<5>;
            mm.funcs[5] = mul_mat_iq1_s_r4_q8_1<6>;
            mm.funcs[6] = mul_mat_iq1_s_r4_q8_1<7>;
            mm.funcs[7] = mul_mat_iq1_s_r4_q8_1<8>;
#ifdef HAVE_FANCY_SIMD
            mm.func16 = mul_mat_iq1_s_r4_q8_1<16>;
#endif
            expected_typeB = GGML_TYPE_Q8_K128;
            break;
        case GGML_TYPE_IQ1_M_R4:
            assert (ne00 % QK4_NL == 0);
            mm.funcs[0] = mul_mat_iq1_m_r4_q8_0<1>;
            mm.funcs[1] = mul_mat_iq1_m_r4_q8_0<2>;
            mm.funcs[2] = mul_mat_iq1_m_r4_q8_0<3>;
            mm.funcs[3] = mul_mat_iq1_m_r4_q8_0<4>;
            mm.funcs[4] = mul_mat_iq1_m_r4_q8_0<5>;
            mm.funcs[5] = mul_mat_iq1_m_r4_q8_0<6>;
            mm.funcs[6] = mul_mat_iq1_m_r4_q8_0<7>;
            mm.funcs[7] = mul_mat_iq1_m_r4_q8_0<8>;
#ifdef HAVE_FANCY_SIMD
            mm.func16 = mul_mat_iq1_m_r4_q8_0<16>;
#endif
            expected_typeB = GGML_TYPE_Q8_K128;
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

template <typename block_q, bool has_row_scale = false, bool scale_is_f16 = false>
struct BaseDequantizer {
    BaseDequantizer(const void * vx, size_t bx, int nrc) : vx(vx), x(nullptr), bx(bx), nrc(nrc) {}
    inline void new_row(int ix) {
        if constexpr (has_row_scale) {
            if constexpr (scale_is_f16) {
                const ggml_half * dptr = (const ggml_half *)((const char *)vx + ix*bx);
                d = GGML_FP16_TO_FP32(*dptr);
                x = (const block_q *)(dptr + 1);
            } else {
                const float * dptr = (const float *)((const char *)vx + ix*bx);
                d = *dptr;
                x = (const block_q *)(dptr + 1);
            }
        } else {
            x = (const block_q *)((const char *)vx + ix*bx);
        }
    }
    const void * vx;
    const block_q * x;
    const size_t bx;
    const int nrc;
    float d;
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
        for (int k = 0; k < 4; ++k) {
            bits.b1.val[k] = vreinterpretq_u8_s8(vqtbl1q_s8(values.val[x[i].scales[4*j+k] & 1], bits.b1.val[k]));
            bits.b2.val[k] = vreinterpretq_u8_s8(vqtbl1q_s8(values.val[x[i].scales[4*j+k] & 1], bits.b2.val[k]));
        }
    }

    Q4bits bits;
    const int8x16x2_t values;
    const uint16x8_t  mask = vdupq_n_u16(254);
    const int16x8_t   m127 = vdupq_n_s16(-127);
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

template <typename Dequantizer, int nrc_y>
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

template <int nrc_y>
static void mul_mat_iq1bn_q8_K64(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const int nb = n / QK_IQ1BN;

    Q8_K64<nrc_y> q8(info);
    DequantizerIQ1BN deq;

    int32x4_t   accd[nrc_y];
    int8x16x4_t v1, v2;

    float scale;
    ggml_half d16;
    char * c16 = (char *)&d16;

    for (int ix = 0; ix < nrc_x; ++ix) {

        const char * cx = ((const char *)vx + ix*bx);
        c16[0] = cx[0]; c16[1] = cx[1];
        //std::memcpy(&d16, cx, sizeof(d16));
        cx += sizeof(d16);
        scale = GGML_FP16_TO_FP32(d16);

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
            info.store(ix, iy, -scale * vaddvq_f32(vfmsq_f32(q8.minus(iy), q8.scale(iy), vcvtq_f32_s32(accd[iy]))));
        }

    }
}

template <int nrc> struct Q8_16 {

    constexpr static int nrc_y = nrc;

    Q8_16(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto ptr = (const float *)info.src1_row(iy);
            std::memcpy(d + 5*iy, ptr, 5*sizeof(float));
            y[iy] = (const int8_t *)(ptr + 5);
        }
    }

    inline int8x16x4_t load_quants(int iy, int i) const { return vld1q_s8_x4(y[iy] + 64*i); }
    inline int8x16x2_t load_quants_32(int iy, int i) const { return vld1q_s8_x2(y[iy] + 32*i); }
    inline float scale(int iy, int k) const { return d[5*iy+k]; }
    inline float sum_row(int iy) const { return d[5*iy + 4]; }
    inline float32x4_t scale(int iy) const { return vld1q_f32(d + 5*iy); }

    float d[5*nrc_y];
    const int8_t * y[nrc_y];
};

template <int nrc_y>
static IQK_NOINLINE void mul_mat_iq2_bn_r4_q8_k16(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    if (nrc_x%4) {
        printf("%s: %d is not a multiple of 4\n", __func__, nrc_x);
        GGML_ABORT("fatal error");
    }
    Q8_16<nrc_y> q8(info);
    auto m3 = vdupq_n_u8(0x3);
    int nb = n / QK_IQ1BN;
    if constexpr (nrc_y == 1) {
        auto mc = vdupq_n_u8(0xc);
        int32x4_t acc[8];
        for (int ix = 0; ix < nrc_x; ix += 4) {
            for (int k = 0; k < 8; ++k) acc[k] = vdupq_n_s32(0);
            const float * dptr = (const float *)((const char *)vx + ix*bx);
            auto dl = vld1q_f32(dptr);
            const uint8_t * iq2 = (const uint8_t *)(dptr + 4);
            for (int ib = 0; ib < nb; ++ib) {
                auto y = q8.load_quants(0, ib);
                for (int j = 0; j < 4; ++j) {
                    auto bits1 = vld1q_u8(iq2 + 64*ib + 16*j);
                    auto bits2 = vshrq_n_u8(bits1, 4);
                    acc[2*j+0] = vdotq_laneq_s32(acc[2*j+0], vandq_u8(bits1, m3), y.val[j], 0);
                    acc[2*j+1] = vdotq_laneq_s32(acc[2*j+1], vandq_u8(bits1, mc), y.val[j], 1);
                    acc[2*j+0] = vdotq_laneq_s32(acc[2*j+0], vandq_u8(bits2, m3), y.val[j], 2);
                    acc[2*j+1] = vdotq_laneq_s32(acc[2*j+1], vandq_u8(bits2, mc), y.val[j], 3);
                }
            }
            auto dy = vmulq_f32(dl, vdupq_n_f32(q8.scale(0, 0)));
            auto sumf1 = vmulq_f32(  vcvtq_f32_s32(acc[0]), dy);
            auto sumf2 = vmulq_f32(  vcvtq_f32_s32(acc[1]), dy);
            dy = vmulq_f32(dl, vdupq_n_f32(q8.scale(0, 1)));
            sumf1 = vfmaq_f32(sumf1, vcvtq_f32_s32(acc[2]), dy);
            sumf2 = vfmaq_f32(sumf2, vcvtq_f32_s32(acc[3]), dy);
            dy = vmulq_f32(dl, vdupq_n_f32(q8.scale(0, 2)));
            sumf1 = vfmaq_f32(sumf1, vcvtq_f32_s32(acc[4]), dy);
            sumf2 = vfmaq_f32(sumf2, vcvtq_f32_s32(acc[5]), dy);
            dy = vmulq_f32(dl, vdupq_n_f32(q8.scale(0, 3)));
            sumf1 = vfmaq_f32(sumf1, vcvtq_f32_s32(acc[6]), dy);
            sumf2 = vfmaq_f32(sumf2, vcvtq_f32_s32(acc[7]), dy);
            auto sumf = vfmaq_f32(sumf1, vdupq_n_f32(0.25f), sumf2);
            sumf = vfmaq_f32(sumf, dl, vdupq_n_f32(-q8.sum_row(0)));
            info.store(ix, 0, sumf);
        }
    } else {
        int32x4_t acc[4*nrc_y] = {};
        uint8x16_t qx[8];
        for (int ix = 0; ix < nrc_x; ix += 4) {
            const float * dptr = (const float *)((const char *)vx + ix*bx);
            auto dl = vld1q_f32(dptr);
            const uint8_t * iq2 = (const uint8_t *)(dptr + 4);
            for (int ib = 0; ib < nb; ++ib) {
                auto bits = vld1q_u8_x2(iq2 + 64*ib);
                qx[0] = vandq_u8(bits.val[0], m3);
                qx[1] = vandq_u8(vshrq_n_u8(bits.val[0], 2), m3);
                qx[2] = vandq_u8(vshrq_n_u8(bits.val[0], 4), m3);
                qx[3] = vshrq_n_u8(bits.val[0], 6);
                qx[4] = vandq_u8(bits.val[1], m3);
                qx[5] = vandq_u8(vshrq_n_u8(bits.val[1], 2), m3);
                qx[6] = vandq_u8(vshrq_n_u8(bits.val[1], 4), m3);
                qx[7] = vshrq_n_u8(bits.val[1], 6);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = q8.load_quants_32(iy, 2*ib+0);
                    acc[4*iy + 0] = vdotq_laneq_s32(acc[4*iy + 0], qx[0], y.val[0], 0);
                    acc[4*iy + 0] = vdotq_laneq_s32(acc[4*iy + 0], qx[1], y.val[0], 1);
                    acc[4*iy + 0] = vdotq_laneq_s32(acc[4*iy + 0], qx[2], y.val[0], 2);
                    acc[4*iy + 0] = vdotq_laneq_s32(acc[4*iy + 0], qx[3], y.val[0], 3);
                    acc[4*iy + 1] = vdotq_laneq_s32(acc[4*iy + 1], qx[4], y.val[1], 0);
                    acc[4*iy + 1] = vdotq_laneq_s32(acc[4*iy + 1], qx[5], y.val[1], 1);
                    acc[4*iy + 1] = vdotq_laneq_s32(acc[4*iy + 1], qx[6], y.val[1], 2);
                    acc[4*iy + 1] = vdotq_laneq_s32(acc[4*iy + 1], qx[7], y.val[1], 3);
                }
                bits = vld1q_u8_x2(iq2 + 64*ib + 32);
                qx[0] = vandq_u8(bits.val[0], m3);
                qx[1] = vandq_u8(vshrq_n_u8(bits.val[0], 2), m3);
                qx[2] = vandq_u8(vshrq_n_u8(bits.val[0], 4), m3);
                qx[3] = vshrq_n_u8(bits.val[0], 6);
                qx[4] = vandq_u8(bits.val[1], m3);
                qx[5] = vandq_u8(vshrq_n_u8(bits.val[1], 2), m3);
                qx[6] = vandq_u8(vshrq_n_u8(bits.val[1], 4), m3);
                qx[7] = vshrq_n_u8(bits.val[1], 6);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = q8.load_quants_32(iy, 2*ib+1);
                    acc[4*iy + 2] = vdotq_laneq_s32(acc[4*iy + 2], qx[0], y.val[0], 0);
                    acc[4*iy + 2] = vdotq_laneq_s32(acc[4*iy + 2], qx[1], y.val[0], 1);
                    acc[4*iy + 2] = vdotq_laneq_s32(acc[4*iy + 2], qx[2], y.val[0], 2);
                    acc[4*iy + 2] = vdotq_laneq_s32(acc[4*iy + 2], qx[3], y.val[0], 3);
                    acc[4*iy + 3] = vdotq_laneq_s32(acc[4*iy + 3], qx[4], y.val[1], 0);
                    acc[4*iy + 3] = vdotq_laneq_s32(acc[4*iy + 3], qx[5], y.val[1], 1);
                    acc[4*iy + 3] = vdotq_laneq_s32(acc[4*iy + 3], qx[6], y.val[1], 2);
                    acc[4*iy + 3] = vdotq_laneq_s32(acc[4*iy + 3], qx[7], y.val[1], 3);
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto dy = q8.scale(iy);
                float32x4_t sumf = vmulq_f32(vcvtq_f32_s32(acc[4*iy+0]), vmulq_laneq_f32(dl, dy, 0));
                sumf = vfmaq_f32(sumf, vcvtq_f32_s32(acc[4*iy+1]), vmulq_laneq_f32(dl, dy, 1));
                sumf = vfmaq_f32(sumf, vcvtq_f32_s32(acc[4*iy+2]), vmulq_laneq_f32(dl, dy, 2));
                sumf = vfmaq_f32(sumf, vcvtq_f32_s32(acc[4*iy+3]), vmulq_laneq_f32(dl, dy, 3));
                sumf = vfmaq_f32(sumf, dl, vdupq_n_f32(-q8.sum_row(iy)));
                info.store(ix, iy, sumf);
                acc[4*iy+0] = acc[4*iy+1] = acc[4*iy+2] = acc[4*iy+3] = vdupq_n_s32(0);
            }
        }
    }
}

template <int nrc_y>
static void mul_mat_iq2bn_q8_K64(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const int nb = n / QK_IQ1BN;

    Q8_K64<nrc_y> q8(info);

    int32x4_t   accd[nrc_y];

    const auto mask2  = vdupq_n_s8(3);

    for (int ix = 0; ix < nrc_x; ++ix) {

        const float * dptr = (const float *)((const char *)vx + ix*bx);
        const float d = *dptr;
        const block_iq2_bn * x = (const block_iq2_bn *)(dptr + 1);

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
            info.store(ix, iy, -d*vaddvq_f32(vfmsq_f32(q8.minus(iy), q8.scale(iy), vcvtq_f32_s32(accd[iy]))));
        }
    }
}

IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
    auto sumi = vdupq_n_s32(0);
    sumi = vdotq_laneq_s32(sumi, qx[0], y.val[0], 0);
    sumi = vdotq_laneq_s32(sumi, qx[1], y.val[1], 0);
    sumi = vdotq_laneq_s32(sumi, qx[2], y.val[0], 1);
    sumi = vdotq_laneq_s32(sumi, qx[3], y.val[1], 1);
    sumi = vdotq_laneq_s32(sumi, qx[4], y.val[0], 2);
    sumi = vdotq_laneq_s32(sumi, qx[5], y.val[1], 2);
    sumi = vdotq_laneq_s32(sumi, qx[6], y.val[0], 3);
    sumi = vdotq_laneq_s32(sumi, qx[7], y.val[1], 3);
    return sumi;
}

IQK_ALWAYS_INLINE int32x4x2_t interleaved_dotq_b16(const int8x16_t * qx, const int8x16x2_t& y) {
    int32x4x2_t sumi = { vdupq_n_s32(0), vdupq_n_s32(0) };
    sumi.val[0] = vdotq_laneq_s32(sumi.val[0], qx[0], y.val[0], 0);
    sumi.val[1] = vdotq_laneq_s32(sumi.val[1], qx[1], y.val[1], 0);
    sumi.val[0] = vdotq_laneq_s32(sumi.val[0], qx[2], y.val[0], 1);
    sumi.val[1] = vdotq_laneq_s32(sumi.val[1], qx[3], y.val[1], 1);
    sumi.val[0] = vdotq_laneq_s32(sumi.val[0], qx[4], y.val[0], 2);
    sumi.val[1] = vdotq_laneq_s32(sumi.val[1], qx[5], y.val[1], 2);
    sumi.val[0] = vdotq_laneq_s32(sumi.val[0], qx[6], y.val[0], 3);
    sumi.val[1] = vdotq_laneq_s32(sumi.val[1], qx[7], y.val[1], 3);
    return sumi;
}

IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16_t& y) {
    auto sumi = vdupq_n_s32(0);
    sumi = vdotq_laneq_s32(sumi, qx[0], y, 0);
    sumi = vdotq_laneq_s32(sumi, qx[1], y, 1);
    sumi = vdotq_laneq_s32(sumi, qx[2], y, 2);
    sumi = vdotq_laneq_s32(sumi, qx[3], y, 3);
    return sumi;
}

IQK_ALWAYS_INLINE void prepare_iq4_nl_quants(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x4_t& bits, int8x16_t * qx) {
    qx[0] = vqtbl1q_s8(values, vandq_u8(bits.val[0], m4));   //  0...3 from the 4 rows
    qx[1] = vqtbl1q_s8(values, vandq_u8(bits.val[1], m4));   // 16..19
    qx[2] = vqtbl1q_s8(values, vandq_u8(bits.val[2], m4));   //  4...7
    qx[3] = vqtbl1q_s8(values, vandq_u8(bits.val[3], m4));   // 20..23
    qx[4] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[0], 4));  //  8..11
    qx[5] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[1], 4));  // 24..27
    qx[6] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[2], 4));  // 12..15
    qx[7] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[3], 4));  // 28..31
}

IQK_ALWAYS_INLINE void prepare_iq4_nl_quants_r8(const int8x16_t& values, const uint8x16_t& m4, const uint8x16x2_t& bits, int8x16_t * qx) {
    qx[0] = vqtbl1q_s8(values, vandq_u8(  bits.val[0], m4));
    qx[1] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[0],  4));
    qx[2] = vqtbl1q_s8(values, vandq_u8(  bits.val[1], m4));
    qx[3] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[1],  4));
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

static void mul_mat_iq1_s_r4_q8_1_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<1, block_q8_K128> q8(info);
    int nb = n / 32;
    GGML_ASSERT(nb%4 == 0);
    int8x16_t qx[8];
    float32x4_t acc[2] = {};
    int32x4_t isum[8];
    auto ms = vdup_n_u16(0x8000);
    for (int ix= 0; ix < nrc_x; ix += 4) {
        auto dptr = (const ggml_half *)((const char *)vx + ix*bx);
        auto d1 = vcvt_f32_f16(vld1_f16((const float16_t *)dptr));
        auto x = (const block_iq1_s_r4 *)(dptr + 4);
        for (int ib = 0; ib < nb/4; ++ib) {
            auto scale_yd = vdupq_n_f32(q8.y[0][ib].d);
            auto scale_ym = vmulq_f32(scale_yd, vcvtq_f32_s32(vmovl_s16(vld1_s16(q8.y[0][ib].bsums))));
            for (int k = 0; k < 4; ++k) {
                auto sas = vld1_u16(x[4*ib+k].qh);
                auto scales4 = vand_u16(vshr_n_u16(sas, 12), vdup_n_u16(7));
                scales4 = vorr_u16(vshl_n_u16(scales4, 1), vdup_n_u16(1));
                auto signs = vreinterpret_s16_u16(vorr_u16(vceq_u16(vand_u16(sas, ms), ms), vdup_n_u16(1)));
                isum[k+4] = vmull_s16(signs, scales4);
                qx[0] = vreinterpretq_s8_u64(uint64x2_t{iq1s_grid[x[4*ib+k].qs[ 0] | ((x[4*ib+k].qh[0] << 8) & 0x0700)],
                                                        iq1s_grid[x[4*ib+k].qs[ 4] | ((x[4*ib+k].qh[0] << 5) & 0x0700)]});
                qx[1] = vreinterpretq_s8_u64(uint64x2_t{iq1s_grid[x[4*ib+k].qs[ 8] | ((x[4*ib+k].qh[0] << 2) & 0x0700)],
                                                        iq1s_grid[x[4*ib+k].qs[12] | ((x[4*ib+k].qh[0] >> 1) & 0x0700)]});
                qx[2] = vreinterpretq_s8_u64(uint64x2_t{iq1s_grid[x[4*ib+k].qs[ 1] | ((x[4*ib+k].qh[1] << 8) & 0x0700)],
                                                        iq1s_grid[x[4*ib+k].qs[ 5] | ((x[4*ib+k].qh[1] << 5) & 0x0700)]});
                qx[3] = vreinterpretq_s8_u64(uint64x2_t{iq1s_grid[x[4*ib+k].qs[ 9] | ((x[4*ib+k].qh[1] << 2) & 0x0700)],
                                                        iq1s_grid[x[4*ib+k].qs[13] | ((x[4*ib+k].qh[1] >> 1) & 0x0700)]});
                qx[4] = vreinterpretq_s8_u64(uint64x2_t{iq1s_grid[x[4*ib+k].qs[ 2] | ((x[4*ib+k].qh[2] << 8) & 0x0700)],
                                                        iq1s_grid[x[4*ib+k].qs[ 6] | ((x[4*ib+k].qh[2] << 5) & 0x0700)]});
                qx[5] = vreinterpretq_s8_u64(uint64x2_t{iq1s_grid[x[4*ib+k].qs[10] | ((x[4*ib+k].qh[2] << 2) & 0x0700)],
                                                        iq1s_grid[x[4*ib+k].qs[14] | ((x[4*ib+k].qh[2] >> 1) & 0x0700)]});
                qx[6] = vreinterpretq_s8_u64(uint64x2_t{iq1s_grid[x[4*ib+k].qs[ 3] | ((x[4*ib+k].qh[3] << 8) & 0x0700)],
                                                        iq1s_grid[x[4*ib+k].qs[ 7] | ((x[4*ib+k].qh[3] << 5) & 0x0700)]});
                qx[7] = vreinterpretq_s8_u64(uint64x2_t{iq1s_grid[x[4*ib+k].qs[11] | ((x[4*ib+k].qh[3] << 2) & 0x0700)],
                                                        iq1s_grid[x[4*ib+k].qs[15] | ((x[4*ib+k].qh[3] >> 1) & 0x0700)]});
                auto scales = vmovl_u16(scales4);
                auto y = vld1q_s8_x2(q8.y[0][ib].qs + 32*k);
                auto sumi1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[0], y.val[0]), qx[1], y.val[1]);
                auto sumi2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[2], y.val[0]), qx[3], y.val[1]);
                auto sumi3 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[4], y.val[0]), qx[5], y.val[1]);
                auto sumi4 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[6], y.val[0]), qx[7], y.val[1]);
                sumi1 = vpaddq_s32(sumi1, sumi2);
                sumi3 = vpaddq_s32(sumi3, sumi4);
                isum[k] = vmulq_s32(scales, vpaddq_s32(sumi1, sumi3));
            }
            acc[0] = vfmaq_laneq_f32(acc[0], vcvtq_f32_s32(isum[0]), scale_yd, 0);
            acc[0] = vfmaq_laneq_f32(acc[0], vcvtq_f32_s32(isum[1]), scale_yd, 1);
            acc[0] = vfmaq_laneq_f32(acc[0], vcvtq_f32_s32(isum[2]), scale_yd, 2);
            acc[0] = vfmaq_laneq_f32(acc[0], vcvtq_f32_s32(isum[3]), scale_yd, 3);
            acc[1] = vfmaq_laneq_f32(acc[1], vcvtq_f32_s32(isum[4]), scale_ym, 0);
            acc[1] = vfmaq_laneq_f32(acc[1], vcvtq_f32_s32(isum[5]), scale_ym, 1);
            acc[1] = vfmaq_laneq_f32(acc[1], vcvtq_f32_s32(isum[6]), scale_ym, 2);
            acc[1] = vfmaq_laneq_f32(acc[1], vcvtq_f32_s32(isum[7]), scale_ym, 3);
        }
        info.store(ix, 0, vmulq_f32(d1, vfmaq_f32(acc[0], acc[1], vdupq_n_f32(IQ1S_DELTA))));
        acc[0] = acc[1] = vdupq_n_f32(0.f);
    }
}

template <int nrc_y>
static void mul_mat_iq1_s_r4_q8_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K128> q8(info);
    int nb = n / 32;
    GGML_ASSERT(nb%4 == 0);
    uint8x16_t qx[8];
    int32x4_t acc[nrc_y] = {};
    auto ms = vdup_n_u16(0x8000);
    auto mask = vdupq_n_s8(0x03);
    float d8[4*nrc_y];
    for (int ix= 0; ix < nrc_x; ix += 4) {
        auto dptr = (const ggml_half *)((const char *)vx + ix*bx);
        auto d1 = vcvt_f32_f16(vld1_f16((const float16_t *)dptr));
        auto x = (const block_iq1_s_r4 *)(dptr + 4);
        for (int ib = 0; ib < nb/4; ++ib) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto scales = vcvtq_f32_s32(vmovl_s16(vld1_s16(q8.y[iy][ib].bsums)));
                vst1q_f32(d8+4*iy, vmulq_f32(vdupq_n_f32(q8.y[iy][ib].d), scales));
            }
            for (int k = 0; k < 4; ++k) {
                auto sas = vld1_u16(x[4*ib+k].qh);
                auto scales4 = vand_u16(vshr_n_u16(sas, 12), vdup_n_u16(7));
                scales4 = vorr_u16(vshl_n_u16(scales4, 1), vdup_n_u16(1));
                auto signs = vreinterpret_s16_u16(vorr_u16(vceq_u16(vand_u16(sas, ms), ms), vdup_n_u16(1)));
                signs = vadd_s16(vdup_n_s16(-8), signs);
                auto delta4 = vmulq_f32(vdupq_n_f32(0.125f), vcvtq_f32_s32(vmull_s16(signs, scales4)));
                qx[0] = vreinterpretq_u8_u32(uint32x4_t{iq1s_grid_us[x[4*ib+k].qs[ 0] | ((x[4*ib+k].qh[0] << 8) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 1] | ((x[4*ib+k].qh[1] << 8) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 2] | ((x[4*ib+k].qh[2] << 8) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 3] | ((x[4*ib+k].qh[3] << 8) & 0x0700)]});
                qx[2] = vreinterpretq_u8_u32(uint32x4_t{iq1s_grid_us[x[4*ib+k].qs[ 4] | ((x[4*ib+k].qh[0] << 5) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 5] | ((x[4*ib+k].qh[1] << 5) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 6] | ((x[4*ib+k].qh[2] << 5) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 7] | ((x[4*ib+k].qh[3] << 5) & 0x0700)]});
                qx[4] = vreinterpretq_u8_u32(uint32x4_t{iq1s_grid_us[x[4*ib+k].qs[ 8] | ((x[4*ib+k].qh[0] << 2) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 9] | ((x[4*ib+k].qh[1] << 2) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[10] | ((x[4*ib+k].qh[2] << 2) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[11] | ((x[4*ib+k].qh[3] << 2) & 0x0700)]});
                qx[6] = vreinterpretq_u8_u32(uint32x4_t{iq1s_grid_us[x[4*ib+k].qs[12] | ((x[4*ib+k].qh[0] >> 1) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[13] | ((x[4*ib+k].qh[1] >> 1) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[14] | ((x[4*ib+k].qh[2] >> 1) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[15] | ((x[4*ib+k].qh[3] >> 1) & 0x0700)]});
                qx[1] = vandq_u8(vshrq_n_u8(qx[0], 4), mask); qx[0] = vandq_u8(qx[0], mask);
                qx[3] = vandq_u8(vshrq_n_u8(qx[2], 4), mask); qx[2] = vandq_u8(qx[2], mask);
                qx[5] = vandq_u8(vshrq_n_u8(qx[4], 4), mask); qx[4] = vandq_u8(qx[4], mask);
                qx[7] = vandq_u8(vshrq_n_u8(qx[6], 4), mask); qx[6] = vandq_u8(qx[6], mask);
                auto scales = vmovl_u16(scales4);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = vld1q_s8_x2(q8.y[iy][ib].qs + 32*k);
                    auto sumi = vdupq_n_s32(0);
                    sumi = vdotq_laneq_s32(sumi, vreinterpretq_s8_u8(qx[0]), y.val[0], 0);
                    sumi = vdotq_laneq_s32(sumi, vreinterpretq_s8_u8(qx[1]), y.val[0], 1);
                    sumi = vdotq_laneq_s32(sumi, vreinterpretq_s8_u8(qx[2]), y.val[0], 2);
                    sumi = vdotq_laneq_s32(sumi, vreinterpretq_s8_u8(qx[3]), y.val[0], 3);
                    sumi = vdotq_laneq_s32(sumi, vreinterpretq_s8_u8(qx[4]), y.val[1], 0);
                    sumi = vdotq_laneq_s32(sumi, vreinterpretq_s8_u8(qx[5]), y.val[1], 1);
                    sumi = vdotq_laneq_s32(sumi, vreinterpretq_s8_u8(qx[6]), y.val[1], 2);
                    sumi = vdotq_laneq_s32(sumi, vreinterpretq_s8_u8(qx[7]), y.val[1], 3);
                    sumi = vmulq_s32(scales, sumi);
                    acc[iy] = vfmaq_f32(acc[iy], vdupq_n_f32(q8.y[iy][ib].d), vcvtq_f32_s32(sumi));
                    acc[iy] = vfmaq_f32(acc[iy], vdupq_n_f32(d8[4*iy+k]), delta4);
                }
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, vmulq_f32(d1, acc[iy]));
            acc[iy] = vdupq_n_f32(0.f);
        }
    }
}

template <int nrc_y>
static void mul_mat_iq1_s_q8_K(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    int8x16_t qx[16];
    int32x4_t scales[2];
    int16x4_t deltas[2];
    float32x4_t acc[nrc_y] = {};
    auto delta_mask = vdupq_n_u16(0x8000);
    for (int ix = 0; ix < nrc_x; ++ix) {
        auto iq1s = (const block_iq1_s *)((const char *)vx + ix*bx);
        for (int ibl = 0; ibl < n/QK_K; ++ibl) {
            float d = GGML_FP16_TO_FP32(iq1s[ibl].d);
            auto qhb = vld1q_u16(iq1s[ibl].qh);
            auto scales128 = vandq_u16(vshrq_n_u16(qhb, 12), vdupq_n_u16(7));
            scales128 = vaddq_u16(vshlq_n_u16(scales128, 1), vdupq_n_u16(1));
            auto mask = vceqq_u16(vandq_u16(qhb, delta_mask), delta_mask);
            // Note: we explicitely assume IQ1S_DELTA = 0.125
            auto deltas128 = vsubq_s16(vbicq_s16(scales128, mask), vandq_s16(scales128, mask));
            //auto deltas128 = vorrq_s16(vandq_s16(vdupq_n_s16(-1), mask), vbicq_s16(vdupq_n_s16(1), mask));
            //deltas128 = vmulq_s16(scales128, deltas128);
            scales128 = vshlq_n_u16(scales128, 3);
            auto qs = iq1s[ibl].qs;
            auto qh = iq1s[ibl].qh;
            for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
                qx[4*ib64+0] = vreinterpretq_s8_u64(uint64x2_t{iq1s_grid[qs[0] | ((qh[2*ib64+0] << 8) & 0x700)], iq1s_grid[qs[1] | ((qh[2*ib64+0] << 5) & 0x700)]});
                qx[4*ib64+1] = vreinterpretq_s8_u64(uint64x2_t{iq1s_grid[qs[2] | ((qh[2*ib64+0] << 2) & 0x700)], iq1s_grid[qs[3] | ((qh[2*ib64+0] >> 1) & 0x700)]});
                qx[4*ib64+2] = vreinterpretq_s8_u64(uint64x2_t{iq1s_grid[qs[4] | ((qh[2*ib64+1] << 8) & 0x700)], iq1s_grid[qs[5] | ((qh[2*ib64+1] << 5) & 0x700)]});
                qx[4*ib64+3] = vreinterpretq_s8_u64(uint64x2_t{iq1s_grid[qs[6] | ((qh[2*ib64+1] << 2) & 0x700)], iq1s_grid[qs[7] | ((qh[2*ib64+1] >> 1) & 0x700)]});
                qs += 8;
            }
            scales[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16 (scales128)));
            scales[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales128)));
            deltas[0] = vget_low_s16 (deltas128);
            deltas[1] = vget_high_s16(deltas128);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto bsums = q8.load_bsums8(iy, ibl);
                auto sumi = vdupq_n_s32(0);
                sumi = vmlal_s16(sumi, deltas[0], vget_low_s16 (bsums));
                sumi = vmlal_s16(sumi, deltas[1], vget_high_s16(bsums));
                for (int k = 0; k < QK_K/128; ++k) {
                    auto qy = q8.load_quants_64(iy, ibl, 2*k+0);
                    auto dot1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[8*k+0], qy.val[0]), qx[8*k+1], qy.val[1]);
                    auto dot2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[8*k+2], qy.val[2]), qx[8*k+3], qy.val[3]);
                    auto dot12 = vpaddq_s32(dot1, dot2);
                    qy = q8.load_quants_64(iy, ibl, 2*k+1);
                    auto dot3 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[8*k+4], qy.val[0]), qx[8*k+5], qy.val[1]);
                    auto dot4 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), qx[8*k+6], qy.val[2]), qx[8*k+7], qy.val[3]);
                    auto dot34 = vpaddq_s32(dot3, dot4);
                    auto dot = vpaddq_s32(dot12, dot34);
                    sumi = vmlaq_s32(sumi, dot, scales[k]);
                }
                acc[iy] = vfmaq_f32(acc[iy], vdupq_n_f32(d*q8.scale(iy, ibl)), vcvtq_f32_s32(sumi));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, 0.125f*vaddvq_f32(acc[iy]));
            acc[iy] = vdupq_n_f32(0);
        }
    }
}

template <int nrc_y>
static void mul_mat_iq1_m_r4_q8_0(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K128> q8(info);
    int nb = n / 32;
    GGML_ASSERT(nb%4 == 0);
    int8x16_t qx[8];
    float32x4_t acc[nrc_y] = {};
    int32x4_t isum[nrc_y] = {};
    auto shuffle0 = uint32x4_t{0x00000000, 0x01010101, 0x02020202, 0x03030303};
    auto step = vdupq_n_u8(4);
    auto ms = vdupq_n_u8(0x08);
    auto mask = vdupq_n_s8(0x18);
    for (int ix= 0; ix < nrc_x; ix += 4) {
        auto dptr = (const ggml_half *)((const char *)vx + ix*bx);
        auto d1 = vmulq_f32(vdupq_n_f32(0.125f), vcvt_f32_f16(vld1_f16((const float16_t *)dptr)));
        auto x = (const block_iq1_m_r4 *)(dptr + 4);
        for (int ib = 0; ib < nb/4; ++ib) {
            for (int k = 0; k < 4; ++k) {
                auto scales4 = vdup_n_u32(((const uint32_t *)x[4*ib+k].scales)[0]);
                scales4 = vand_u8(vshl_u32(scales4, int32x2_t{0, -4}), vdup_n_u8(0xf));
                auto scales16 = vmovl_u8(scales4);
                auto scales1 = vmovl_u16(vget_low_u16(scales16));
                auto scales2 = vmovl_u16(vget_high_u16(scales16));
                auto qh = (const uint32_t *)x[4*ib+k].qh;
                auto idxh = uint32x4_t{qh[0], qh[0] >> 4, qh[1], qh[1] >> 4};
                auto signs = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(vandq_u8(idxh, ms), ms), vdupq_n_u8(1)));
                signs = vaddq_s8(signs, vdupq_n_s8(-8));
                qx[0] = vreinterpretq_s8_u32(uint32x4_t{iq1s_grid_us[x[4*ib+k].qs[ 0] | ((x[4*ib+k].qh[0] << 8) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 1] | ((x[4*ib+k].qh[1] << 8) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 2] | ((x[4*ib+k].qh[2] << 8) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 3] | ((x[4*ib+k].qh[3] << 8) & 0x0700)]});
                qx[2] = vreinterpretq_s8_u32(uint32x4_t{iq1s_grid_us[x[4*ib+k].qs[ 4] | ((x[4*ib+k].qh[0] << 4) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 5] | ((x[4*ib+k].qh[1] << 4) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 6] | ((x[4*ib+k].qh[2] << 4) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 7] | ((x[4*ib+k].qh[3] << 4) & 0x0700)]});
                qx[4] = vreinterpretq_s8_u32(uint32x4_t{iq1s_grid_us[x[4*ib+k].qs[ 8] | ((x[4*ib+k].qh[4] << 8) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[ 9] | ((x[4*ib+k].qh[5] << 8) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[10] | ((x[4*ib+k].qh[6] << 8) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[11] | ((x[4*ib+k].qh[7] << 8) & 0x0700)]});
                qx[6] = vreinterpretq_s8_u32(uint32x4_t{iq1s_grid_us[x[4*ib+k].qs[12] | ((x[4*ib+k].qh[4] << 4) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[13] | ((x[4*ib+k].qh[5] << 4) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[14] | ((x[4*ib+k].qh[6] << 4) & 0x0700)],
                                                        iq1s_grid_us[x[4*ib+k].qs[15] | ((x[4*ib+k].qh[7] << 4) & 0x0700)]});
                auto shuffle = shuffle0;
                for (int j = 0; j < 4; ++j) {
                    auto s = vqtbl1q_s8(signs, shuffle);
                    qx[2*j+1] = vaddq_s8(s, vandq_s8(vshrq_n_s8(qx[2*j+0], 1), mask));
                    qx[2*j+0] = vaddq_s8(s, vandq_s8(vshlq_n_s8(qx[2*j+0], 3), mask));
                    shuffle = vaddq_u8(shuffle, step);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = vld1q_s8_x2(q8.y[iy][ib].qs + 32*k);
                    auto sumi1 = vdupq_n_s32(0);
                    auto sumi2 = vdupq_n_s32(0);
                    sumi1 = vdotq_laneq_s32(sumi1, vreinterpretq_s8_u8(qx[0]), y.val[0], 0);
                    sumi1 = vdotq_laneq_s32(sumi1, vreinterpretq_s8_u8(qx[1]), y.val[0], 1);
                    sumi1 = vdotq_laneq_s32(sumi1, vreinterpretq_s8_u8(qx[2]), y.val[0], 2);
                    sumi1 = vdotq_laneq_s32(sumi1, vreinterpretq_s8_u8(qx[3]), y.val[0], 3);
                    sumi2 = vdotq_laneq_s32(sumi2, vreinterpretq_s8_u8(qx[4]), y.val[1], 0);
                    sumi2 = vdotq_laneq_s32(sumi2, vreinterpretq_s8_u8(qx[5]), y.val[1], 1);
                    sumi2 = vdotq_laneq_s32(sumi2, vreinterpretq_s8_u8(qx[6]), y.val[1], 2);
                    sumi2 = vdotq_laneq_s32(sumi2, vreinterpretq_s8_u8(qx[7]), y.val[1], 3);
                    isum[iy] = vmlaq_s32(vmlaq_s32(isum[iy], sumi1, scales1), sumi2, scales2);
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = vfmaq_f32(acc[iy], vdupq_n_f32(q8.y[iy][ib].d), vcvtq_f32_s32(isum[iy]));
                isum[iy] = vdupq_n_s32(0);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix, iy, vmulq_f32(d1, acc[iy]));
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
            acc[2*iy+0] = acc[2*iy+1] = vdupq_n_f32(0.f);
        }
    }
}

void mul_mat_iq4_nl_r4_q8_0_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<1, block_q8_0_x4> q8(info);
    auto m4 = vdupq_n_u8(0xf);
    auto values = vld1q_s8(iq4k_values);
    int nb = n / QK4_NL;
    GGML_ASSERT(nb%4 == 0);
    int8x16_t qx[8];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        auto acc = vdupq_n_f32(0.f);
        const block_iq4_nl_r4 * iq4 = (const block_iq4_nl_r4 *)((const char *)vx + ix*bx);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            auto y1 = vld1q_s8_x4(q8.y[0][ib4].qs);
            auto y2 = vld1q_s8_x4(q8.y[0][ib4].qs+64);
            for (int k = 0; k < 4; ++k) {
                auto scales = vcvt_f32_f16(vld1_f16((const float16_t *)iq4[4*ib4+k].d));
                auto d4d8 = vmulq_f32(scales, vdupq_n_f32(GGML_FP16_TO_FP32(q8.y[0][ib4].d[k])));
                auto sumi = vdupq_n_s32(0);
                const auto yval = k < 2 ? y1.val + 2*k : y2.val + 2*(k-2);
                auto bits   = vld1q_u8_x4(iq4[4*ib4+k].qs);
                qx[0] = vqtbl1q_s8(values, vandq_u8(bits.val[0], m4));   //  0...3 from the 4 rows
                qx[1] = vqtbl1q_s8(values, vandq_u8(bits.val[1], m4));   // 16..19
                sumi = vdotq_laneq_s32(sumi, qx[0], yval[0], 0);
                sumi = vdotq_laneq_s32(sumi, qx[1], yval[1], 0);
                qx[2] = vqtbl1q_s8(values, vandq_u8(bits.val[2], m4));   //  4...7
                qx[3] = vqtbl1q_s8(values, vandq_u8(bits.val[3], m4));   // 20..23
                sumi = vdotq_laneq_s32(sumi, qx[2], yval[0], 1);
                sumi = vdotq_laneq_s32(sumi, qx[3], yval[1], 1);
                qx[4] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[0], 4));  //  8..11
                qx[5] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[1], 4));  // 24..27
                sumi = vdotq_laneq_s32(sumi, qx[4], yval[0], 2);
                sumi = vdotq_laneq_s32(sumi, qx[5], yval[1], 2);
                qx[6] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[2], 4));  // 12..15
                qx[7] = vqtbl1q_s8(values, vshrq_n_u8(bits.val[3], 4));  // 28..31
                sumi = vdotq_laneq_s32(sumi, qx[6], yval[0], 3);
                sumi = vdotq_laneq_s32(sumi, qx[7], yval[1], 3);
                acc = vfmaq_f32(acc, d4d8, vcvtq_f32_s32(sumi));
            }
        }
        info.store(ix, 0, acc);
    }
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

struct Q4_0_R4_Dequantizer {
    Q4_0_R4_Dequantizer(const void * vx, size_t bx) : cx((const char *)vx), bx(bx) {}
    inline void new_row(int ix) { iq4 = (const block_iq4_nl_r4 *)(cx + ix*bx); }
    inline float32x4_t prepare(int ib4, int k, int8x16_t * qx) const {
        auto scales = vcvt_f32_f16(vld1_f16((const float16_t *)iq4[4*ib4+k].d));
        auto bits   = vld1q_u8_x4(iq4[4*ib4+k].qs);
        for (int j = 0; j < 4; ++j) bits.val[j] = veorq_u8(m88, bits.val[j]);
        qx[0] = vshlq_n_u8(bits.val[0], 4); //  0...3 from the 4 rows
        qx[1] = vshlq_n_u8(bits.val[1], 4); // 16..19
        qx[2] = vshlq_n_u8(bits.val[2], 4); //  4...7
        qx[3] = vshlq_n_u8(bits.val[3], 4); // 20..23
        qx[4] = vandq_u8(bits.val[0], m4);  //  8..11
        qx[5] = vandq_u8(bits.val[1], m4);  // 24..27
        qx[6] = vandq_u8(bits.val[2], m4);  // 12..15
        qx[7] = vandq_u8(bits.val[3], m4);  // 28..31
        return scales;
    }
    inline float32x4_t result(float32x4_t acc) const {
        return vmulq_f32(norm, acc);
    }

    const char * cx;
    const size_t bx;
    const block_iq4_nl_r4 * iq4;
    const uint8x16_t m4 = vdupq_n_u8(0xf0);
    const uint8x16_t m88 = vdupq_n_u8(0x88);
    const float32x4_t norm = vdupq_n_f32(1.f/16);
};

struct Q4_0_R8_Dequantizer {
    Q4_0_R8_Dequantizer(const void * vx, size_t bx) : cx((const char *)vx), bx(bx) {}
    inline void new_row(int ix) { iq4 = (const block_iq4_nl_r8 *)(cx + ix*bx); }
    inline float32x4x2_t prepare(int ib4, int k, int8x16_t * qx) const {
        auto scales16 = vld1q_f16((const float16_t *)iq4[4*ib4+k].d);
        float32x4x2_t scales = { vcvt_f32_f16(vget_low_f16(scales16)), vcvt_f32_f16(vget_high_f16(scales16)) };
        for (int j = 0; j < 4; ++j) {
            auto bits = vld1q_u8_x2(iq4[4*ib4+k].qs + 32*j);
            //bits.val[0] = veorq_u8(m88, bits.val[0]);
            //bits.val[1] = veorq_u8(m88, bits.val[1]);
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

#define SET_MUL_MAT_FUNCTIONS_T(m, func, Dequantizer) \
            m.funcs[0] = func<Dequantizer, 1>;\
            m.funcs[1] = func<Dequantizer, 2>;\
            m.funcs[2] = func<Dequantizer, 3>;\
            m.funcs[3] = func<Dequantizer, 4>;\
            m.funcs[4] = func<Dequantizer, 5>;\
            m.funcs[5] = func<Dequantizer, 6>;\
            m.funcs[6] = func<Dequantizer, 7>;\
            m.funcs[7] = func<Dequantizer, 8>;\

#define SET_MUL_MAT_FUNCTIONS(m, func) \
            m.funcs[0] = func<1>;\
            m.funcs[1] = func<2>;\
            m.funcs[2] = func<3>;\
            m.funcs[3] = func<4>;\
            m.funcs[4] = func<5>;\
            m.funcs[5] = func<6>;\
            m.funcs[6] = func<7>;\
            m.funcs[7] = func<8>;\

template <typename Dequantizer> void MulMat::set_functions(MulMat& m) {
    if constexpr (std::is_same_v<Dequantizer, DequantizerQ40> || std::is_same_v<Dequantizer, DequantizerQ50> ||
                  std::is_same_v<Dequantizer, DequantizerQ80> || std::is_same_v<Dequantizer, DequantizerIQ4NL> ||
                  std::is_same_v<Dequantizer, DequantizerQ60>) {
        SET_MUL_MAT_FUNCTIONS_T(m, mul_mat_qX_0_q8_0, Dequantizer);
    }
    else if constexpr (std::is_same_v<Dequantizer, DequantizerQ41> || std::is_same_v<Dequantizer, DequantizerQ51>) {
        SET_MUL_MAT_FUNCTIONS_T(m, mul_mat_qX_1_q8_1, Dequantizer);
    }
    else {
        SET_MUL_MAT_FUNCTIONS_T(m, mul_mat_qX_K_q8_K_T, Dequantizer);
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
        case GGML_TYPE_IQ4_KS:
            MulMat::set_functions<DequantizerIQ4KS>(m);
            break;
        case GGML_TYPE_IQ4_KSS:
            MulMat::set_functions<DequantizerIQ4KSS>(m);
            break;
        case GGML_TYPE_IQ2_KS:
            MulMat::set_functions<DequantizerIQ2KS>(m);
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
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq1bn_q8_K64);
            expected_Btype = GGML_TYPE_Q8_K64;
            break;
        case GGML_TYPE_IQ2_BN:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq2bn_q8_K64);
            expected_Btype = GGML_TYPE_Q8_K64;
            break;
        case GGML_TYPE_IQ2_BN_R4:
            m.funcs[0] = mul_mat_iq2_bn_r4_q8_k16<1>;
            m.funcs[1] = mul_mat_iq2_bn_r4_q8_k16<2>;
            m.funcs[2] = mul_mat_iq2_bn_r4_q8_k16<3>;
            m.funcs[3] = mul_mat_iq2_bn_r4_q8_k16<4>;
            m.funcs[4] = mul_mat_iq2_bn_r4_q8_k16<5>;
            //m.funcs[5] = mul_mat_iq2_bn_r4_q8_k16<6>;
            //m.funcs[6] = mul_mat_iq2_bn_r4_q8_k16<7>;
            //m.funcs[7] = mul_mat_iq2_bn_r4_q8_k16<8>;
            expected_Btype = GGML_TYPE_Q8_K16;
            break;
        case GGML_TYPE_Q4_0:
            MulMat::set_functions<DequantizerQ40>(m);
            expected_Btype = GGML_TYPE_Q8_0_X4;
            break;
        case GGML_TYPE_Q4_1:
            MulMat::set_functions<DequantizerQ41>(m);
            expected_Btype = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_Q5_0:
            MulMat::set_functions<DequantizerQ50>(m);
            expected_Btype = GGML_TYPE_Q8_0_X4;
            break;
        case GGML_TYPE_Q5_1:
            MulMat::set_functions<DequantizerQ51>(m);
            expected_Btype = GGML_TYPE_Q8_1_X4;
            break;
        case GGML_TYPE_Q6_0:
            MulMat::set_functions<DequantizerQ60>(m);
            expected_Btype = GGML_TYPE_Q8_0_X4;
            break;
        case GGML_TYPE_Q8_0:
            MulMat::set_functions<DequantizerQ80>(m);
            expected_Btype = GGML_TYPE_Q8_0_X4;
            break;
        case GGML_TYPE_IQ4_NL:
            MulMat::set_functions<DequantizerIQ4NL>(m);
            expected_Btype = GGML_TYPE_Q8_0_X4;
            break;
        case GGML_TYPE_IQ4_NL_R4:
            SET_MUL_MAT_FUNCTIONS_T(m, mul_mat_qx_r4_q8_0, IQ4_NL_R4_Dequantizer);
            expected_Btype = GGML_TYPE_Q8_0_X4;
            break;
        case GGML_TYPE_IQ4_XS_R8:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq4_xs_r8_q8_k);
            expected_Btype = GGML_TYPE_Q8_K32;
            break;
        case GGML_TYPE_IQ4_KS_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq4_ks_r4_q8_k);
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ2_XXS_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq2_xxs_r4_q8_k);
            m.func16 = mul_mat_iq2_xxs_r4_q8_k<16>;
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ2_XS_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq2_xs_r4_q8_k);
            m.func16 = mul_mat_iq2_xs_r4_q8_k<16>;
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ2_S_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq2_s_r4_q8_k);
            m.func16 = mul_mat_iq2_s_r4_q8_k<16>;
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ1_S:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq1_s_q8_K);
            m.func16 = mul_mat_iq1_s_q8_K<16>;
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ1_S_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq1_s_r4_q8_1);
            m.funcs[0] = mul_mat_iq1_s_r4_q8_1_1;
            m.func16 = mul_mat_iq1_s_r4_q8_1<16>;
            expected_Btype = GGML_TYPE_Q8_K128;
            break;
        case GGML_TYPE_IQ1_M_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq1_m_r4_q8_0);
            m.func16 = mul_mat_iq1_m_r4_q8_0<16>;
            expected_Btype = GGML_TYPE_Q8_K128;
            break;
        case GGML_TYPE_IQ3_XXS_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq3_xxs_r4_q8_k);
            m.func16 = mul_mat_iq3_xxs_r4_q8_k<16>;
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ3_S_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq3_s_r4_q8_k);
            m.func16 = mul_mat_iq3_s_r4_q8_k<16>;
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_Q2_K_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_q2_k_r4_q8_k);
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_Q3_K_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_q3_k_r4_q8_k);
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_Q4_K_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_q4_k_r4_q8_k);
            expected_Btype = GGML_TYPE_Q8_K32;
            break;
        case GGML_TYPE_Q5_K_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_q5_k_r4_q8_k);
            expected_Btype = GGML_TYPE_Q8_K32;
            break;
        case GGML_TYPE_Q6_K_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_q6_k_r4_q8_k);
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_Q8_K_R8:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_q8_k_r8_q8_k);
            expected_Btype = GGML_TYPE_Q8_KR8;
            break;
        case GGML_TYPE_Q8_KV:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_q8_KV_q8_KV);
            m.funcs[0] = mul_mat_q8_KV_q8_KV_1;
            m.func16 = mul_mat_q8_KV_q8_KV<16>;
            expected_Btype = GGML_TYPE_Q8_KV;
            break;
        case GGML_TYPE_Q8_KV_R8:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_q8_KV_r8_q8_KV);
            expected_Btype = GGML_TYPE_Q8_KV;
            break;
        case GGML_TYPE_IQ2_K_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq2_k_r4_q8_k);
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ3_K_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq3_k_r4_q8_k);
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ4_K_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq4_k_r4_q8_k);
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_IQ5_K_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq5_k_r4_q8_k);
            expected_Btype = GGML_TYPE_Q8_K;
            break;
        case GGML_TYPE_Q4_0_R8:
            SET_MUL_MAT_FUNCTIONS_T(m, mul_mat_qx_r8_q8_0, Q4_0_R8_Dequantizer);
            expected_Btype = GGML_TYPE_Q8_0_X4;
            break;
        case GGML_TYPE_Q5_0_R4:
            SET_MUL_MAT_FUNCTIONS_T(m, mul_mat_qx_r4_q8_0, Q5_0_R4_Dequantizer);
            expected_Btype = GGML_TYPE_Q8_0_X4;
            break;
        case GGML_TYPE_Q6_0_R4:
            SET_MUL_MAT_FUNCTIONS_T(m, mul_mat_qx_r4_q8_0, Q6_0_R4_Dequantizer);
            expected_Btype = GGML_TYPE_Q8_0_X4;
            break;
        case GGML_TYPE_Q8_0_R8:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_q8_0_r8_q8_0);
            expected_Btype = GGML_TYPE_Q8_0_X4;
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
inline float32x4_t v_silu(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t neg_x = vsubq_f32(zero, x);
    const float32x4_t exp_neg_x = v_expf(neg_x);
    const float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
    return vdivq_f32(x, one_plus_exp_neg_x);
}
inline float32x4_t v_gelu(float32x4_t x, float32x4_t c1, float32x4_t c2) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t arg = vfmaq_f32(one, c1, vmulq_f32(x, x));
    arg = vmulq_f32(arg, vmulq_f32(x, c2));
    float32x4_t exp_arg = v_expf(arg);
    float32x4_t gelu = vmulq_f32(x, vdivq_f32(exp_arg, vaddq_f32(exp_arg, one)));
    uint32x4_t mask = vcgtq_f32(x, vdupq_n_f32(10.f));
    return vbslq_f32(mask, x, gelu);
}

void MulMat::gelu(int n, const float * x, float * y) {
    constexpr float GELU_COEF_A = 0.044715f;
    constexpr float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
    int i = 0;
    auto c1 = vdupq_n_f32(GELU_COEF_A);
    auto c2 = vdupq_n_f32(2.f*SQRT_2_OVER_PI);
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, v_gelu(vld1q_f32(x + i), c1, c2));
    }
    for (; i < n; ++i) y[i] = 0.5f*x[i]*(1.0f + tanhf(SQRT_2_OVER_PI*x[i]*(1.0f + GELU_COEF_A*x[i]*x[i])));
}

void MulMat::silu(int n, const float * x, float * y) {
    int i = 0;
    for (; i + 3 < n; i += 4) vst1q_f32(y + i, v_silu(vld1q_f32(x + i)));
    for (; i < n; ++i) y[i] = x[i]/(1.0f + expf(-x[i]));
}

void MulMat::relu(int n, const float * x, float * y) {
    for (int j = 0; j < n; ++j) y[j] = x[j] > 0 ? x[j] : 0;
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
inline __m512 v_gelu(__m512 x, __m512 c1, __m512 c2) {
    const __m512 one = _mm512_set1_ps(1.0f);
    __m512 arg = _mm512_fmadd_ps(x, _mm512_mul_ps(c1, x), one);
    //__m512 arg = _mm512_add_ps(one, _mm512_mul_ps(_mm512_mul_ps(x, x), c1));
    arg = _mm512_mul_ps(arg, _mm512_mul_ps(c2, x));
    const __mmask16 mask = _mm512_cmp_ps_mask(arg, _mm512_set1_ps(30.f), _CMP_GT_OQ);
    const __m512 exp_arg = v_expf(arg);
    const __m512 ratio = _mm512_div_ps(exp_arg, _mm512_add_ps(exp_arg, one));
    return _mm512_mul_ps(x, _mm512_mask_blend_ps(mask, ratio, one));
}
inline static __m512 v_silu(__m512 x) {
    const __m512 one = _mm512_set1_ps(1);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 neg_x = _mm512_sub_ps(zero, x);
    const __m512 exp_neg_x = v_expf(neg_x);
    const __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(x, one_plus_exp_neg_x);
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
inline static __m256 v_gelu(__m256 x, __m256 c1, __m256 c2) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(10.f), _CMP_GT_OQ);
    __m256 arg = _mm256_add_ps(one, _mm256_mul_ps(_mm256_mul_ps(x, x), c1));
    arg = _mm256_mul_ps(arg, _mm256_mul_ps(x, c2));
    __m256 exp_arg = v_expf(arg);
    __m256 gelu = _mm256_mul_ps(x, _mm256_div_ps(exp_arg, _mm256_add_ps(exp_arg, one)));
    return _mm256_or_ps(_mm256_and_ps(mask, x), _mm256_andnot_ps(mask, gelu));
}
inline static __m256 v_silu(__m256 x) {
    const __m256 one = _mm256_set1_ps(1);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 neg_x = _mm256_sub_ps(zero, x);
    const __m256 exp_neg_x = v_expf(neg_x);
    const __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
    return _mm256_div_ps(x, one_plus_exp_neg_x);
}

void MulMat::gelu(int n, const float * x, float * y) {
    constexpr float GELU_COEF_A = 0.044715f;
    constexpr float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
    //GGML_ASSERT(n%8 == 0);
    int i = 0;
#if defined __AVX512F__ && defined __AVX512DQ__
    {
        __m512 c1 = _mm512_set1_ps(GELU_COEF_A);
        __m512 c2 = _mm512_set1_ps(2.f*SQRT_2_OVER_PI);
        for (; i + 15 < n; i += 16) _mm512_storeu_ps(y + i, v_gelu(_mm512_loadu_ps(x + i), c1, c2));
    }
#endif
#if defined __AVX2__ && defined __FMA__
    if (i + 7 < n) {
        __m256 c1 = _mm256_set1_ps(GELU_COEF_A);
        __m256 c2 = _mm256_set1_ps(2.f*SQRT_2_OVER_PI);
        for (; i + 7 < n; i += 8) _mm256_storeu_ps(y + i, v_gelu(_mm256_loadu_ps(x + i), c1, c2));

    }
#endif
    for (; i < n; ++i) y[i] = 0.5f*x[i]*(1.0f + tanhf(SQRT_2_OVER_PI*x[i]*(1.0f + GELU_COEF_A*x[i]*x[i])));
}

void MulMat::silu(int n, const float * x, float * y) {
    int i = 0;
#if defined __AVX512F__ && defined __AVX512DQ__
    for (; i + 15 < n; i += 16) _mm512_storeu_ps(y + i, v_silu(_mm512_loadu_ps(x + i)));
#endif
#if defined __AVX2__ && defined __FMA__
    for (; i + 7 < n; i += 8) _mm256_storeu_ps(y + i, v_silu(_mm256_loadu_ps(x + i)));
#endif
    for (; i < n; ++i) y[i] = x[i]/(1.0f + expf(-x[i]));
}

void MulMat::relu(int n, const float * x, float * y) {
    for (int j = 0; j < n; ++j) y[j] = x[j] > 0 ? x[j] : 0;
}

#endif
} // namespace

namespace {

template <int k_step>
struct BaseHelper {
    BaseHelper(const char * data, int stride) : data(data), block(data), stride(stride) {}

    //inline void set_block(int k1) { block = data + k1*k_step*stride; }
    inline void reset_block() { block = data; }
    inline void next_block() { block += k_step*stride; }
    inline const char * lblock(int l1) const { return block + l1*stride; }

    const char * data;
    const char * block;
    int stride;

};

struct F16 {
#ifdef __AVX512F__
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
    static inline Data fmadd(Data prev, Data v1, Data v2) { return _mm512_fmadd_ps(v1, v2, prev); }
    static inline float reduce_max(Data data) { return _mm512_reduce_max_ps(data); }
    static inline float reduce_add(Data data) { return _mm512_reduce_add_ps(data); }
    static inline Data max(Data v1, Data v2) { return _mm512_max_ps(v1, v2); }
    static inline Data add(Data v1, Data v2) { return _mm512_add_ps(v1, v2); }
    static inline Data set4(const float * ptr) {
        auto v128 = _mm_loadu_ps(ptr);
        auto v256 = _mm256_set_m128(v128, v128);
        return _mm512_insertf32x8(_mm512_castps256_ps512(v256), v256, 1);
    }
    static inline Data fmadd_lane0(Data prev, Data v1, Data v2) { return _mm512_fmadd_ps(v1, _mm512_shuffle_ps(v2, v2, 0x00), prev); }
    static inline Data fmadd_lane1(Data prev, Data v1, Data v2) { return _mm512_fmadd_ps(v1, _mm512_shuffle_ps(v2, v2, 0x55), prev); }
    static inline Data fmadd_lane2(Data prev, Data v1, Data v2) { return _mm512_fmadd_ps(v1, _mm512_shuffle_ps(v2, v2, 0xaa), prev); }
    static inline Data fmadd_lane3(Data prev, Data v1, Data v2) { return _mm512_fmadd_ps(v1, _mm512_shuffle_ps(v2, v2, 0xff), prev); }
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
    static inline Data max(Data v1, Data v2) { return _mm256_max_ps(v1, v2); }
    static inline Data add(Data v1, Data v2) { return _mm256_add_ps(v1, v2); }
    static inline Data set4(const float * ptr) {
        auto v128 = _mm_loadu_ps(ptr);
        return _mm256_set_m128(v128, v128);
    }
    static inline Data fmadd_lane0(Data prev, Data v1, Data v2) { return _mm256_fmadd_ps(v1, _mm256_shuffle_ps(v2, v2, 0x00), prev); }
    static inline Data fmadd_lane1(Data prev, Data v1, Data v2) { return _mm256_fmadd_ps(v1, _mm256_shuffle_ps(v2, v2, 0x55), prev); }
    static inline Data fmadd_lane2(Data prev, Data v1, Data v2) { return _mm256_fmadd_ps(v1, _mm256_shuffle_ps(v2, v2, 0xaa), prev); }
    static inline Data fmadd_lane3(Data prev, Data v1, Data v2) { return _mm256_fmadd_ps(v1, _mm256_shuffle_ps(v2, v2, 0xff), prev); }
#else
    using Data = float16x8_t;
    constexpr static int block_size = 8;
    //constexpr static int num_registers = 32;
    //constexpr static int q_step = 8;
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
    static inline Data max(Data v1, Data v2) { return vmaxq_f16(v1, v2); }
    static inline Data add(Data v1, Data v2) { return vaddq_f16(v1, v2); }
    static inline float16x4_t set4(const float * ptr) {
        auto val32 = vld1q_f32(ptr);
        return vcvt_f16_f32(val32);
    }
    static inline Data fmadd_lane0(Data prev, Data v1, float16x4_t v2) { return vfmaq_lane_f16(prev, v1, v2, 0); }
    static inline Data fmadd_lane1(Data prev, Data v1, float16x4_t v2) { return vfmaq_lane_f16(prev, v1, v2, 1); }
    static inline Data fmadd_lane2(Data prev, Data v1, float16x4_t v2) { return vfmaq_lane_f16(prev, v1, v2, 2); }
    static inline Data fmadd_lane3(Data prev, Data v1, float16x4_t v2) { return vfmaq_lane_f16(prev, v1, v2, 3); }
#endif
    template <int k_step> static inline float reduce_max(const Data * data) {
        return reduce_T<k_step, &F16::max, &F16::reduce_max>(data);
    }
    template <int k_step> static inline float reduce_add(const Data * data) {
        return reduce_T<k_step, &F16::add, &F16::reduce_add>(data);
    }
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

template <int D> struct block_q8_KV {
    float d;
    int   s;
    int8_t qs[D];
};

template <int D, int step>
struct HelperQ8KV final : public BaseHelper<step> {
    using Base = BaseHelper<step>;
    using block_q8 = block_q8_KV<D>;
    constexpr static int block_size_q = D;
    HelperQ8KV(const char * data, int stride) : Base(data, stride) {}

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        auto q8 = (const block_q8_KV<D> *)Base::lblock(l1);
#ifdef __aarch64__
        auto vd = F16::set1(q8->d);
        auto qs = vld1_s8_x2(q8->qs + 8*i);
        v1 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(qs.val[0])));
        v2 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(qs.val[1])));
#else
        auto vd = F16::set1(q8->d);
#ifdef __AVX512F__
        v1 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)q8->qs+i+0))));
        v2 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)q8->qs+i+1))));
#else
        v1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(q8->qs+8*i+0)))));
        v2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(q8->qs+8*i+8)))));
#endif
#endif
    }
};

template <int D, int step>
struct HelperQ80 final : public BaseHelper<step> {
    using Base = BaseHelper<step>;
#ifdef HAVE_FANCY_SIMD
    using block_q8 = block_q8_1;
    constexpr static int block_size_q = QK8_1;
#else
    using block_q8 = block_q8_0;
    constexpr static int block_size_q = QK8_0;
#endif
    HelperQ80(const char * data, int stride) : Base(data, stride) {}

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        int j = F16::block_size*i;
        auto dl = (const block_q8_0 *)Base::lblock(l1) + j/QK8_0;
#ifdef __aarch64__
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d));
        int ii = j%QK8_0;
        auto qs = vld1_s8_x2(dl->qs + ii);
        v1 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(qs.val[0])));
        v2 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(qs.val[1])));
#else
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d));
#ifdef __AVX512F__
        v1 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)dl->qs+0))));
        v2 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)dl->qs+1))));
#else
        int ii = j%QK8_0;
        v1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(dl->qs+ii+0)))));
        v2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(dl->qs+ii+8)))));
#endif
#endif
    }

    static inline void convert(int nq, int stride_q, const float * q, block_q8_0 * y) {
        //GGML_ASSERT(nq <= step); Why did I have this assert?
        for (int i = 0; i < nq; ++i) {
            quantize_row_q8_0_x4(q, y, D);
            q += stride_q;
            y += D/QK8_0;
        }
    }

    static inline void convert(int nq, int stride_q, const float * q, block_q8_1 * y) {
        //GGML_ASSERT(nq <= step); Why did I have this assert?
        for (int i = 0; i < nq; ++i) {
            quantize_row_q8_1_x4(q, y, D);
            q += stride_q;
            y += D/QK8_1;
        }
    }

    static inline void convert(int nq, int stride_q, const float * q, block_q8_KV<D> * y) {
        for (int i = 0; i < nq; ++i) {
            quantize_row_q8_KV(q, y, D);
            q += stride_q;
            ++y;
        }
    }
};

template <int D, int step>
struct HelperQ80R8 : public BaseHelper<step> {
    using Base = BaseHelper<step>;
#ifdef __AVX2__
    constexpr static int block_size_q = QK8_1;
    using block_q8 = block_q8_1;
#else
    constexpr static int block_size_q = QK8_0;
    using block_q8 = block_q8_0;
#endif
    HelperQ80R8(int nk, const HelperQ80<D, step>& q8) : Base(q8.data, q8.stride) {
        r4 = repack(nk, q8);
        Base::data = (const char *)r4.data();
        Base::stride = (D/QK8_0)*sizeof(block_q8_0);
    }

    static std::vector<block_q8_0_r8> repack(int nk, const HelperQ80<D, step>& q8) {
        static_assert(D%QK8_0 == 0);
        GGML_ASSERT(nk%8 == 0);
        constexpr int nblock = D/QK8_0;
        std::vector<block_q8_0_r8> result(nblock * nk/8);
        auto y = result.data();
        const block_q8_0 * x8[8];
#ifdef __ARM_NEON
        int8x16x2_t m0, m1, m2, m3;
#endif
        for (int row = 0; row < nk; row += 8) {
            for (int k = 0; k < 8; ++k) x8[k] = (const block_q8_0 *)(q8.data + (row + k)*q8.stride);
            for (int ib = 0; ib < nblock; ++ib) {
                for (int k = 0; k < 8; ++k) y[ib].d[k] = x8[k][ib].d;
#ifdef __AVX2__
                auto m0 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[4][ib].qs), _mm_loadu_si128((const __m128i *)x8[0][ib].qs));
                auto m1 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[5][ib].qs), _mm_loadu_si128((const __m128i *)x8[1][ib].qs));
                auto m2 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[6][ib].qs), _mm_loadu_si128((const __m128i *)x8[2][ib].qs));
                auto m3 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[7][ib].qs), _mm_loadu_si128((const __m128i *)x8[3][ib].qs));
                auto t0 = _mm256_unpacklo_epi32(m0, m1);
                auto t1 = _mm256_unpacklo_epi32(m2, m3);
                auto t2 = _mm256_unpackhi_epi32(m0, m1);
                auto t3 = _mm256_unpackhi_epi32(m2, m3);
                m0 = _mm256_unpacklo_epi64(t0, t1);
                m1 = _mm256_unpackhi_epi64(t0, t1);
                m2 = _mm256_unpacklo_epi64(t2, t3);
                m3 = _mm256_unpackhi_epi64(t2, t3);
#ifdef HAVE_FANCY_SIMD
                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
#endif
                _mm256_storeu_si256((__m256i *)y[ib].qs + 0, m0);
                _mm256_storeu_si256((__m256i *)y[ib].qs + 1, m1);
                _mm256_storeu_si256((__m256i *)y[ib].qs + 2, m2);
                _mm256_storeu_si256((__m256i *)y[ib].qs + 3, m3);
                m0 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[4][ib].qs+1), _mm_loadu_si128((const __m128i *)x8[0][ib].qs+1));
                m1 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[5][ib].qs+1), _mm_loadu_si128((const __m128i *)x8[1][ib].qs+1));
                m2 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[6][ib].qs+1), _mm_loadu_si128((const __m128i *)x8[2][ib].qs+1));
                m3 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[7][ib].qs+1), _mm_loadu_si128((const __m128i *)x8[3][ib].qs+1));
                t0 = _mm256_unpacklo_epi32(m0, m1);
                t1 = _mm256_unpacklo_epi32(m2, m3);
                t2 = _mm256_unpackhi_epi32(m0, m1);
                t3 = _mm256_unpackhi_epi32(m2, m3);
                m0 = _mm256_unpacklo_epi64(t0, t1);
                m1 = _mm256_unpackhi_epi64(t0, t1);
                m2 = _mm256_unpacklo_epi64(t2, t3);
                m3 = _mm256_unpackhi_epi64(t2, t3);
#ifdef HAVE_FANCY_SIMD
                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
#endif
                _mm256_storeu_si256((__m256i *)y[ib].qs + 4, m0);
                _mm256_storeu_si256((__m256i *)y[ib].qs + 5, m1);
                _mm256_storeu_si256((__m256i *)y[ib].qs + 6, m2);
                _mm256_storeu_si256((__m256i *)y[ib].qs + 7, m3);
#elif defined __ARM_NEON
                for (int l = 0; l < 2; ++l) {
                    m0.val[0] = vld1q_s8(x8[0][ib].qs+16*l); m0.val[1] = vld1q_s8(x8[4][ib].qs+16*l);
                    m1.val[0] = vld1q_s8(x8[1][ib].qs+16*l); m1.val[1] = vld1q_s8(x8[5][ib].qs+16*l);
                    m2.val[0] = vld1q_s8(x8[2][ib].qs+16*l); m2.val[1] = vld1q_s8(x8[6][ib].qs+16*l);
                    m3.val[0] = vld1q_s8(x8[3][ib].qs+16*l); m3.val[1] = vld1q_s8(x8[7][ib].qs+16*l);
                    auto row01 = vtrnq_s32(vreinterpretq_s32_s8(m0.val[0]), vreinterpretq_s32_s8(m1.val[0]));
                    auto row23 = vtrnq_s32(vreinterpretq_s32_s8(m2.val[0]), vreinterpretq_s32_s8(m3.val[0]));
                    m0.val[0] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                    m1.val[0] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                    m2.val[0] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                    m3.val[0] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                    row01 = vtrnq_s32(vreinterpretq_s32_s8(m0.val[1]), vreinterpretq_s32_s8(m1.val[1]));
                    row23 = vtrnq_s32(vreinterpretq_s32_s8(m2.val[1]), vreinterpretq_s32_s8(m3.val[1]));
                    m0.val[1] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                    m1.val[1] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                    m2.val[1] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                    m3.val[1] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                    vst1q_s8_x2(y[ib].qs +  0 + 128*l, m0);
                    vst1q_s8_x2(y[ib].qs + 32 + 128*l, m1);
                    vst1q_s8_x2(y[ib].qs + 64 + 128*l, m2);
                    vst1q_s8_x2(y[ib].qs + 96 + 128*l, m3);
                }
#else
                for (int l = 0; l < 4; ++l) {
                    for (int k = 0; k < 8; ++k) for (int i = 0; i < 4; ++i) {
                        y[ib].qs[32*l+4*k+i+  0] = x8[k][ib].qs[i+4*l+ 0];
                        y[ib].qs[32*l+4*k+i+128] = x8[k][ib].qs[i+4*l+16];
                    }
                }
#endif
            }
            y += nblock;
        }
        return result;
    }

    std::vector<block_q8_0_r8> r4;
};

// TODO: unite this with the above
template <int D, int step>
struct HelperQ8KVR8 : public BaseHelper<step> {
    using Base = BaseHelper<step>;
    constexpr static int block_size_q = D;
    using block_q8 = block_q8_KV<D>;

    struct block_q8_KV_r8 {
        float  d[8];
        int8_t qs[8*D];
    };

    HelperQ8KVR8(int nk, const HelperQ8KV<D, step>& q8) : Base(q8.data, q8.stride) {
        r4 = repack(nk, q8);
        Base::data = (const char *)r4.data();
        Base::stride = sizeof(block_q8_KV_r8)/8;
    }

    static std::vector<block_q8_KV_r8> repack(int nk, const HelperQ8KV<D, step>& q8) {
        static_assert(D%32 == 0);
        GGML_ASSERT(nk%8 == 0);
        std::vector<block_q8_KV_r8> result(nk/8);
        auto y = result.data();
#ifdef __ARM_NEON
        int8x16x2_t m0, m1, m2, m3;
#endif
        const int8_t * x8[8];
        for (int ix = 0; ix < nk/8; ++ix) {
            for (int k = 0; k < 8; ++k) {
                auto dptr = (const float *)(q8.data + (8*ix + k)*q8.stride);
                y[ix].d[k] = dptr[0];
                x8[k] = (const int8_t *)(dptr + 2);
            }
            for (int ib = 0; ib < D/16; ++ib) {
#ifdef __AVX2__
                auto m0 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[4]+ib), _mm_loadu_si128((const __m128i *)x8[0]+ib));
                auto m1 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[5]+ib), _mm_loadu_si128((const __m128i *)x8[1]+ib));
                auto m2 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[6]+ib), _mm_loadu_si128((const __m128i *)x8[2]+ib));
                auto m3 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[7]+ib), _mm_loadu_si128((const __m128i *)x8[3]+ib));
                auto t0 = _mm256_unpacklo_epi32(m0, m1);
                auto t1 = _mm256_unpacklo_epi32(m2, m3);
                auto t2 = _mm256_unpackhi_epi32(m0, m1);
                auto t3 = _mm256_unpackhi_epi32(m2, m3);
                m0 = _mm256_unpacklo_epi64(t0, t1);
                m1 = _mm256_unpackhi_epi64(t0, t1);
                m2 = _mm256_unpacklo_epi64(t2, t3);
                m3 = _mm256_unpackhi_epi64(t2, t3);
#ifdef HAVE_FANCY_SIMD
                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
#endif
                _mm256_storeu_si256((__m256i *)y[ix].qs + 4*ib+0, m0);
                _mm256_storeu_si256((__m256i *)y[ix].qs + 4*ib+1, m1);
                _mm256_storeu_si256((__m256i *)y[ix].qs + 4*ib+2, m2);
                _mm256_storeu_si256((__m256i *)y[ix].qs + 4*ib+3, m3);
#elif defined __ARM_NEON
                // TODO
                m0.val[0] = vld1q_s8(x8[0]+16*ib); m0.val[1] = vld1q_s8(x8[4]+16*ib);
                m1.val[0] = vld1q_s8(x8[1]+16*ib); m1.val[1] = vld1q_s8(x8[5]+16*ib);
                m2.val[0] = vld1q_s8(x8[2]+16*ib); m2.val[1] = vld1q_s8(x8[6]+16*ib);
                m3.val[0] = vld1q_s8(x8[3]+16*ib); m3.val[1] = vld1q_s8(x8[7]+16*ib);
                auto row01 = vtrnq_s32(vreinterpretq_s32_s8(m0.val[0]), vreinterpretq_s32_s8(m1.val[0]));
                auto row23 = vtrnq_s32(vreinterpretq_s32_s8(m2.val[0]), vreinterpretq_s32_s8(m3.val[0]));
                m0.val[0] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                m1.val[0] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                m2.val[0] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                m3.val[0] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                row01 = vtrnq_s32(vreinterpretq_s32_s8(m0.val[1]), vreinterpretq_s32_s8(m1.val[1]));
                row23 = vtrnq_s32(vreinterpretq_s32_s8(m2.val[1]), vreinterpretq_s32_s8(m3.val[1]));
                m0.val[1] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                m1.val[1] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                m2.val[1] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                m3.val[1] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                vst1q_s8_x2(y[ix].qs +  0 + 128*ib, m0);
                vst1q_s8_x2(y[ix].qs + 32 + 128*ib, m1);
                vst1q_s8_x2(y[ix].qs + 64 + 128*ib, m2);
                vst1q_s8_x2(y[ix].qs + 96 + 128*ib, m3);
#else
                // TODO
                for (int l = 0; l < 4; ++l) {
                    for (int k = 0; k < 8; ++k) for (int i = 0; i < 4; ++i) {
                        y[ib].qs[32*l+4*k+i+  0] = x8[k][ib].qs[i+4*l+ 0];
                        y[ib].qs[32*l+4*k+i+128] = x8[k][ib].qs[i+4*l+16];
                    }
                }
#endif
            }
        }
        return result;
    }

    std::vector<block_q8_KV_r8> r4;
};

template <int D, int step>
struct HelperQ40 final : public BaseHelper<step> {
    using Base = BaseHelper<step>;
    using block_q8 = block_q8_0;
    constexpr static int block_size_q = QK8_0;
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
#ifdef __AVX512F__
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
    constexpr static int block_size_q = QK8_1;
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
#ifdef __AVX512F__
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

template <int D, int step>
struct HelperIQ4nl final : public BaseHelper<step> {
    using Base = BaseHelper<step>;
#ifdef __aarch64__
    using block_q8 = block_q8_0;
    HelperIQ4nl(const char * data, int stride) : Base(data, stride), values(vld1q_s8(iq4k_values)) {}
    constexpr static int block_size_q = QK8_0;
#else
    HelperIQ4nl(const char * data, int stride) : Base(data, stride) {}
    using block_q8 = block_q8_1;
    constexpr static int block_size_q = QK8_1;
#endif

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        int j = F16::block_size*i;
        auto dl = (const block_iq4_nl *)Base::lblock(l1) + j/QK4_0;
#ifdef __aarch64__
        auto vd = F16::set1(*(const float16_t *)&dl->d);
        auto q  = vld1q_u8(dl->qs);
        q = j%QK4_0 ? vshrq_n_u8(q, 4) : vandq_u8(q, mask);
        q = vqtbl1q_s8(values, q);
        v1 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(vget_low_s8(q))));
        v2 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(vget_high_s8(q))));
#else
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d));
        auto q  = _mm_loadu_si128((const __m128i *)dl->qs);
#ifdef __AVX512F__
        auto ql = _mm_shuffle_epi8(values, _mm_and_si128(q, mask));
        auto qh = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q, 4), mask));
        v1 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ql)));
        v2 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qh)));
#else
        if (j%QK4_0) q = _mm_srli_epi16(q, 4);
        auto q16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(values, _mm_and_si128(q, mask)));
        v1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16))));
        v2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1))));
#endif
#endif
    }

#ifdef __aarch64__
    const uint8x16_t mask = vdupq_n_u8(0xf);
    const int8x16_t values;
#else
    const __m128i mask = _mm_set1_epi8(0xf);
    const __m128i values = _mm_loadu_si128((const __m128i *)iq4k_values);
#endif
};

template <int D, int step>
struct HelperQ60 final : public BaseHelper<step> {
#ifdef __aarch64__
    using block_q8 = block_q8_0;
    constexpr static int block_size_q = QK8_0;
#else
    using block_q8 = block_q8_1;
    constexpr static int block_size_q = QK8_1;
#endif
    using Base = BaseHelper<step>;
    HelperQ60(const char * data, int stride) : Base(data, stride) {}

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        int j = F16::block_size*i;
        auto dl = (const block_q6_0 *)Base::lblock(l1) + j/QK6_0;
#ifdef __aarch64__
        // TODO
        auto vd = F16::set1(*(const float16_t *)&dl->d);
        auto qh8 = vld1_u8(dl->qh);
        auto qh  = vcombine_u8(vshl_n_u8(qh8, 4), qh8);
        auto qs  = vld1q_u8(dl->qs);
        qs = j%QK4_0 ? vshrq_n_u8(qs, 4) : vandq_u8(qs, mask_l);
        qs = vorrq_u8(qs, vandq_u8(mask_h, j%QK4_0 ? vshrq_n_u8(qh, 2) : qh));
        qs = vaddq_s8(qs, m32);
        v1 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(vget_low_s8(qs))));
        v2 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(vget_high_s8(qs))));
#else
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d));
        auto bl = _mm_loadu_si128((const __m128i *)dl->qs);
        uint64_t aux64; std::memcpy(&aux64, dl->qh, 8);
        auto bh = _mm_set_epi64x(aux64, aux64 << 4);
#ifdef __AVX512F__
        auto ql = _mm_add_epi8(_mm_or_si128(_mm_and_si128(bl, mask_l), _mm_and_si128(bh, mask_h)), m32);
        auto qh = _mm_add_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(bl, 4), mask_l), _mm_and_si128(_mm_srli_epi16(bh, 2), mask_h)), m32);
        v1 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ql)));
        v2 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qh)));
#else
        if (j%QK4_0) {
            bl = _mm_srli_epi16(bl, 4);
            bh = _mm_srli_epi16(bh, 2);
        }
        auto q16 = _mm256_cvtepi8_epi16(_mm_add_epi8(_mm_or_si128(_mm_and_si128(bl, mask_l), _mm_and_si128(bh, mask_h)), m32));
        v1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16))));
        v2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1))));
#endif
#endif
    }

#ifdef __AVX2__
    const __m128i mask_l = _mm_set1_epi8(0x0f);
    const __m128i mask_h = _mm_set1_epi8(0x30);
    const __m128i m32    = _mm_set1_epi8(-32);
#else
    const uint8x16_t mask_l = vdupq_n_u8(0x0f);
    const uint8x16_t mask_h = vdupq_n_u8(0x30);
    const  int8x16_t m32    = vdupq_n_s8(-32);
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
                vms[j] = m;
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
    static inline __m256 apply_mask(int l, const char * mask, __m256 val, __m256 vinf) {
        auto m128 = _mm_loadu_si128((const __m128i *)mask+l);
        m128 = _mm_cmpeq_epi16(m128, _mm_setzero_si128());
        auto m256 = _mm256_cvtepi16_epi32(m128);
        auto mf = _mm256_castsi256_ps(_mm256_or_si256(m256, _mm256_slli_epi32(m256, 16)));
        return _mm256_or_ps(_mm256_and_ps(mf, val), _mm256_andnot_ps(mf, vinf));
    }
#ifdef __AVX512F__
    static inline __m512 apply_mask(int l, const char * mask, __m512 val, __m512 vinf) {
        auto m256 = _mm256_loadu_si256((const __m256i *)mask+l);
        m256 = _mm256_cmpeq_epi16(m256, _mm256_setzero_si256());
        auto m512 = _mm512_cvtepi16_epi32(m256);
        auto mf = _mm512_castsi512_ps(_mm512_or_si512(m512, _mm512_slli_epi32(m512, 16)));
        return _mm512_or_ps(_mm512_and_ps(mf, val), _mm512_andnot_ps(mf, vinf));
    }
#endif
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
        auto vinf  = F16::set1(-INFINITY);
        for (int l = 0; l < k_step/F16::block_size; ++l) {
            vk[l] = apply_mask(l, mask, F16::load(cache + k_step*j + F16::block_size*l), vinf);
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
        if (M[j] > -INFINITY) update_S(j, vk);
    }
    inline void update_M_S(int j, float32x4_t * vk, const char * mask) {
        float smax = load_apply_mask_and_scale(j, vk, mask);
        update_M(j, smax);
        if (M[j] > -INFINITY) update_S(j, vk);
    }
#else
    inline void update_M_S(int j, F16::Data * vk) {
        float smax = load_and_scale(j, vk);
        update_M(j, smax);
        if (M[j] > -INFINITY) update_S(j, vk);
    }
    inline void update_M_S(int j, F16::Data * vk, const char * mask) {
        float smax = load_apply_mask_and_scale(j, vk, mask);
        update_M(j, smax);
        if (M[j] > -INFINITY) update_S(j, vk);
    }
#endif

    cache_t cache[q_step*k_step];
    float S[q_step], M[q_step];
    int need_scaling[q_step];
    float vms[q_step];
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

    template <typename VHelper>
    inline void accumulate_qkv_1(const VHelper& vh, const FlashMS<q_step, k_step>& fms) {
        F16::Data vq[D/F16::block_size];
        if (fms.need_scaling[0] == 2) {
            for (int i = 0; i < D/F16::block_size; ++i) vq[i] = F16::zero();
        } else {
            for (int i = 0; i < D/F16::block_size; ++i) vq[i] = F16::load(qkv_cache + F16::block_size*i);
            if (fms.need_scaling[0] == 1) {
                auto vms = F16::set1(fms.vms[0]);
                for (int i = 0; i < D/F16::block_size; ++i) vq[i] = F16::mul(vms, vq[i]);
            }
        }
        F16::Data v0, v1;
        for (int l = 0; l < k_step; l += 4) {
            auto vs0 = F16::set1(fms.cache[l + 0]);
            auto vs1 = F16::set1(fms.cache[l + 1]);
            auto vs2 = F16::set1(fms.cache[l + 2]);
            auto vs3 = F16::set1(fms.cache[l + 3]);
            for (int i = 0; i < D/F16::block_size; i += 2) {
                vh.load(l+0, i, v0, v1);
                vq[i+0] = F16::fmadd(vq[i+0], v0, vs0);
                vq[i+1] = F16::fmadd(vq[i+1], v1, vs0);
                vh.load(l+1, i, v0, v1);
                vq[i+0] = F16::fmadd(vq[i+0], v0, vs1);
                vq[i+1] = F16::fmadd(vq[i+1], v1, vs1);
                vh.load(l+2, i, v0, v1);
                vq[i+0] = F16::fmadd(vq[i+0], v0, vs2);
                vq[i+1] = F16::fmadd(vq[i+1], v1, vs2);
                vh.load(l+3, i, v0, v1);
                vq[i+0] = F16::fmadd(vq[i+0], v0, vs3);
                vq[i+1] = F16::fmadd(vq[i+1], v1, vs3);
            }
        }
        for (int i = 0; i < D/F16::block_size; ++i) F16::store(qkv_cache + F16::block_size*i, vq[i]);
    }

    // This fails for head sizes of 80 and 112 as D/16 is odd, so we cannot do steps of 2
    // Hence, for now, we will not handle head sizes of 80 and 112
    template <typename VHelper>
    inline void accumulate_qkv(const VHelper& vh, const FlashMS<q_step, k_step>& fms) {
        if constexpr (q_step == 1) {
            accumulate_qkv_1(vh, fms);
            return;
        }
        F16::Data v[8];
        for (int j = 0; j < q_step; ++j) {
            auto R = qkv_cache + D*j;
            if (fms.need_scaling[j] == 2) {
                std::memset(R, 0, D*sizeof(qkv_cache_t));
            }
            else if (fms.need_scaling[j] == 1) {
                auto vms = F16::set1(fms.vms[j]);
                for (int i = 0; i < D/F16::block_size; ++i) {
                    F16::store(R + F16::block_size*i, F16::mul(vms, F16::load(R + F16::block_size*i)));
                }
            }
        }
        for (int i = 0; i < D/F16::block_size; i += 2) {
            for (int l = 0; l < k_step; l += 4) {
                vh.load(l+0, i, v[0], v[4]);
                vh.load(l+1, i, v[1], v[5]);
                vh.load(l+2, i, v[2], v[6]);
                vh.load(l+3, i, v[3], v[7]);
                for (int j = 0; j < q_step; ++j) {
                    auto R = qkv_cache + D*j;
                    auto s1 = F16::load(R + F16::block_size*(i+0));
                    auto s2 = F16::load(R + F16::block_size*(i+1));
                    auto vs = F16::set4(fms.cache + k_step*j + l);
                    s1 = F16::fmadd_lane0(s1, v[0], vs);
                    s2 = F16::fmadd_lane0(s2, v[4], vs);
                    s1 = F16::fmadd_lane1(s1, v[1], vs);
                    s2 = F16::fmadd_lane1(s2, v[5], vs);
                    s1 = F16::fmadd_lane2(s1, v[2], vs);
                    s2 = F16::fmadd_lane2(s2, v[6], vs);
                    s1 = F16::fmadd_lane3(s1, v[3], vs);
                    s2 = F16::fmadd_lane3(s2, v[7], vs);
                    F16::store(R + F16::block_size*(i+0), s1);
                    F16::store(R + F16::block_size*(i+1), s2);
                }
            }
        }
    }

    template <typename VHelper>
    inline void accumulate_qkv(int nq1, const VHelper& vh, const FlashMS<q_step, k_step>& fms) {
        if (nq1 == 1) {
            accumulate_qkv_1(vh, fms);
            return;
        }
        F16::Data v[8];
        for (int j = 0; j < nq1; ++j) {
            auto R = qkv_cache + D*j;
            if (fms.need_scaling[j] == 2) {
                std::memset(R, 0, D*sizeof(qkv_cache_t));
            }
            else if (fms.need_scaling[j] == 1) {
                auto vms = F16::set1(fms.vms[j]);
                for (int i = 0; i < D/F16::block_size; ++i) {
                    F16::store(R + F16::block_size*i, F16::mul(vms, F16::load(R + F16::block_size*i)));
                }
            }
        }
        for (int i = 0; i < D/F16::block_size; i += 2) {
            for (int l = 0; l < k_step; l += 4) {
                vh.load(l+0, i, v[0], v[4]);
                vh.load(l+1, i, v[1], v[5]);
                vh.load(l+2, i, v[2], v[6]);
                vh.load(l+3, i, v[3], v[7]);
                for (int j = 0; j < nq1; ++j) {
                    auto R = qkv_cache + D*j;
                    auto s1 = F16::load(R + F16::block_size*(i+0));
                    auto s2 = F16::load(R + F16::block_size*(i+1));
                    auto vs = F16::set4(fms.cache + k_step*j + l);
                    s1 = F16::fmadd_lane0(s1, v[0], vs);
                    s2 = F16::fmadd_lane0(s2, v[4], vs);
                    s1 = F16::fmadd_lane1(s1, v[1], vs);
                    s2 = F16::fmadd_lane1(s2, v[5], vs);
                    s1 = F16::fmadd_lane2(s1, v[2], vs);
                    s2 = F16::fmadd_lane2(s2, v[6], vs);
                    s1 = F16::fmadd_lane3(s1, v[3], vs);
                    s2 = F16::fmadd_lane3(s2, v[7], vs);
                    F16::store(R + F16::block_size*(i+0), s1);
                    F16::store(R + F16::block_size*(i+1), s2);
                }
            }
        }
    }

    inline void normalize_and_store_1row(const FlashMS<q_step, k_step>& fms, int j, const qkv_cache_t * R, float * qkv) const {
        GGML_ASSERT(fms.S[j] > 0);
        auto norm = F16::set1(1/fms.S[j]);
        //auto norm = F16::set1(fms.S[j] > 0 ? 1/fms.S[j] : 0.f);
        for (int i = 0; i < D/F16::block_size; ++i) {
            auto r = F16::load(R + F16::block_size*i);
            F16::store(qkv + F16::block_size*i, F16::mul(norm, r));
        }
    }

    inline void normalize_and_store(const FlashMS<q_step, k_step>& fms, int nq1, int stride_qkv, float * qkv, float * M, float * S) const {
        if (M && S) {
            std::memcpy(M, fms.M, nq1*sizeof(float));
            std::memcpy(S, fms.S, nq1*sizeof(float));
            auto R = qkv_cache;
            for (int j = 0; j < nq1; ++j) {
                std::memcpy(qkv, R, D*sizeof(float));
                qkv += stride_qkv;
                R   += D;
            }
        } else {
            auto R = qkv_cache;
            for (int j = 0; j < nq1; ++j) {
                normalize_and_store_1row(fms, j, R, qkv);
                qkv += stride_qkv;
                R   += D;
            }
        }
    }

    inline void normalize_and_store(const FlashMS<q_step, k_step>& fms, int stride_qkv, float * qkv, float * M, float * S) const {
        if (M && S) {
            std::memcpy(M, fms.M, q_step*sizeof(float));
            std::memcpy(S, fms.S, q_step*sizeof(float));
            auto R = qkv_cache;
            for (int j = 0; j < q_step; ++j) {
                std::memcpy(qkv, R, D*sizeof(float));
                qkv += stride_qkv;
                R   += D;
            }
        } else {
            auto R = qkv_cache;
            for (int j = 0; j < q_step; ++j) {
                normalize_and_store_1row(fms, j, R, qkv);
                qkv += stride_qkv;
                R   += D;
            }
        }
    }

    // qkv_cache_t qkv_cache[D*q_step];
    // The initializer is not actually required. But the compiler cannot figure out that when qkv_cache is
    // first used for q_step rows, fms.need_scaling[j] is always 2, which zeroes the content of qkv_cache.
    // As a result, we get an infinite stream of warnings about uninitialized variable use (one for each
    // combination of D, q_step, k_step), which is extremely annoying. Hence, I succumb to the trend of
    // constantly being saved by others (the compiler in this case), and add this 100% unnecessary initialization.
    qkv_cache_t qkv_cache[D*q_step] = {};
};

template <int D, int q_step, int k_step>
struct FlashQKfp32 {
    static_assert(D%F16::block_size == 0 && D <= 576);
    static_assert(k_step%F16::block_size == 0);
    static_assert(q_step <= 4 || q_step%4 == 0);

#ifdef __AVX2__
    template <typename KHelper, typename q_float>
    static inline void multiply_mask_kq(const KHelper& kh, int stride_q, int stride_m, const q_float * q, const char * mask,
            FlashMS<q_step, k_step>& fms) {
#ifdef HAVE_FANCY_SIMD
        constexpr int nrc_q = 8;
        constexpr int nrc_k = 8;
#else
        // somewhat surprisingly, nrc_q = 4, nrc_k = 8 is better than nrc_q = 8, nrc_k = 4
        constexpr int nrc_q = 4;
        constexpr int nrc_k = 8;
#endif
        constexpr int qrem = q_step - nrc_q*(q_step/nrc_q);
        constexpr int krem = k_step - nrc_k*(k_step/nrc_k);
        static_assert(krem == 0);
        DataInfo info{fms.cache, (const char *)q, k_step, stride_q*sizeof(q_float), 0, 1, nullptr};
        for (int iq = 0; iq < q_step/nrc_q; ++iq) {
            for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, nrc_q>, QFT<ggml_half, nrc_k>>(D, kh.block, kh.stride, ik*nrc_k, info);
            }
            info.cur_y += nrc_q;
        }
        if constexpr (qrem > 0) {
            for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, qrem>, QFT<ggml_half, nrc_k>>(D, kh.block, kh.stride, ik*nrc_k, info);
            }
        }
        F16::Data vk[k_step/F16::block_size];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
    }
#else
    template <typename KHelper, typename q_float>
    static inline void multiply_mask_kq(const KHelper& kh, int stride_q, int stride_m, const q_float * q, const char * mask,
            FlashMS<q_step, k_step>& fms) {
        constexpr int nrc_q = 4;
        constexpr int nrc_k = 6;
        constexpr int qrem = q_step - nrc_q*(q_step/nrc_q);
        constexpr int krem = k_step - nrc_k*(k_step/nrc_k);
        DataInfo info{fms.cache, (const char *)q, k_step, stride_q*sizeof(q_float), 0, 1, nullptr};
        for (int iq = 0; iq < q_step/nrc_q; ++iq) {
            for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                mul_mat_f16_f16_NxN<nrc_q, nrc_k, true>(D, kh.block, kh.stride, ik*nrc_k, info);
            }
            if constexpr (krem > 0) {
                mul_mat_f16_f16_NxN<nrc_q, krem, true>(D, kh.block, kh.stride, k_step - krem, info);
            }
            info.cur_y += nrc_q;
        }
        if constexpr (qrem > 0) {
            for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                mul_mat_f16_f16_NxN<qrem, nrc_k, true>(D, kh.block, kh.stride, ik*nrc_k, info);
            }
            if constexpr (krem > 0) {
                mul_mat_f16_f16_NxN<qrem, krem, true>(D, kh.block, kh.stride, k_step - krem, info);
            }
        }
        float32x4_t vk[k_step/4];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
    }
#endif

#ifdef __AVX2__
    template <typename KHelper, typename q_float>
    static inline void multiply_mask_kq(int nq, const KHelper& kh, int stride_q, int stride_m, const q_float * q, const char * mask,
            FlashMS<q_step, k_step>& fms) {
#ifdef HAVE_FANCY_SIMD
        constexpr int nrc_q = 8;
        constexpr int nrc_k = 8;
#else
        // somewhat surprisingly, nrc_q = 4, nrc_k = 8 is better than nrc_q = 8, nrc_k = 4
        constexpr int nrc_q = 4;
        constexpr int nrc_k = 8;
#endif
        static_assert(k_step%nrc_k == 0);
        int qrem = nq - nrc_q*(nq/nrc_q);
        DataInfo info{fms.cache, (const char *)q, k_step, stride_q*sizeof(q_float), 0, 1, nullptr};
        for (int iq = 0; iq < nq/nrc_q; ++iq) {
            for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, nrc_q>, QFT<ggml_half, nrc_k>>(D, kh.block, kh.stride, ik*nrc_k, info);
            }
            info.cur_y += nrc_q;
        }
        if (qrem > 0) {
            switch (qrem) {
                case 1: {
                    for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                        mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 1>, QFT<ggml_half, nrc_k>>(D, kh.block, kh.stride, ik*nrc_k, info);
                    }
                } break;
                case 2: {
                    for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                        mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 2>, QFT<ggml_half, nrc_k>>(D, kh.block, kh.stride, ik*nrc_k, info);
                    }
                } break;
                case 3: {
                    for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                        mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 3>, QFT<ggml_half, nrc_k>>(D, kh.block, kh.stride, ik*nrc_k, info);
                    }
                } break;
#ifdef HAVE_FANCY_SIMD
                case 4: {
                    for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                        mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 4>, QFT<ggml_half, nrc_k>>(D, kh.block, kh.stride, ik*nrc_k, info);
                    }
                } break;
                case 5: {
                    for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                        mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 5>, QFT<ggml_half, nrc_k>>(D, kh.block, kh.stride, ik*nrc_k, info);
                    }
                } break;
                case 6: {
                    for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                        mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 6>, QFT<ggml_half, nrc_k>>(D, kh.block, kh.stride, ik*nrc_k, info);
                    }
                } break;
                case 7: {
                    for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                        mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 7>, QFT<ggml_half, nrc_k>>(D, kh.block, kh.stride, ik*nrc_k, info);
                    }
                } break;
#endif
            }
        }
        F16::Data vk[k_step/F16::block_size];
        for (int j = 0; j < nq; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
    }
#else
    template <typename KHelper, typename q_float>
    static inline void multiply_mask_kq(int nq, const KHelper& kh, int stride_q, int stride_m, const q_float * q, const char * mask,
            FlashMS<q_step, k_step>& fms) {
        constexpr int nrc_q = 4;
        constexpr int nrc_k = 6;
        constexpr int krem = k_step - nrc_k*(k_step/nrc_k);
        const int qrem = q_step - nrc_q*(q_step/nrc_q);
        DataInfo info{fms.cache, (const char *)q, k_step, stride_q*sizeof(q_float), 0, 1, nullptr};
        for (int iq = 0; iq < nq/nrc_q; ++iq) {
            for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                mul_mat_f16_f16_NxN<nrc_q, nrc_k, true>(D, kh.block, kh.stride, ik*nrc_k, info);
            }
            if constexpr (krem > 0) {
                mul_mat_f16_f16_NxN<nrc_q, krem, true>(D, kh.block, kh.stride, k_step - krem, info);
            }
            info.cur_y += nrc_q;
        }
        switch (qrem) {
            case 0: break;
            case 1: {
                for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                    mul_mat_f16_f16_NxN<1, nrc_k, true>(D, kh.block, kh.stride, ik*nrc_k, info);
                }
                if constexpr (krem > 0) {
                    mul_mat_f16_f16_NxN<1, krem, true>(D, kh.block, kh.stride, k_step - krem, info);
                }
            } break;
            case 2: {
                for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                    mul_mat_f16_f16_NxN<2, nrc_k, true>(D, kh.block, kh.stride, ik*nrc_k, info);
                }
                if constexpr (krem > 0) {
                    mul_mat_f16_f16_NxN<2, krem, true>(D, kh.block, kh.stride, k_step - krem, info);
                }
            } break;
            case 3: {
                for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                    mul_mat_f16_f16_NxN<3, nrc_k, true>(D, kh.block, kh.stride, ik*nrc_k, info);
                }
                if constexpr (krem > 0) {
                    mul_mat_f16_f16_NxN<3, krem, true>(D, kh.block, kh.stride, k_step - krem, info);
                }
            } break;
        }
        float32x4_t vk[k_step/4];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
    }
#endif

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

    template <typename KHelper>
    static inline std::pair<mul_mat_t, int> mul_mat_kernel(int nq) {
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
        if constexpr (std::is_same_v<KHelper, HelperQ80<D, k_step>>) {
#ifdef __aarch64__
            MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerQ80, nq);
#else
#ifdef HAVE_FANCY_SIMD
            MAKE_FUNCS(mul_mat_qX_1_q8_1_T<Q8_0_1_Unpacker, nq);
#else
            MAKE_FUNCS(mul_mat_qX_0_q8_0_T<Q8_0_Unpacker, nq);
#endif
#endif
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ8KV<D, k_step>>) {
#ifdef __aarch64__
            if (nq%16 == 0) return std::make_pair(mul_mat_q8_KV_q8_KV<16>, 16);
            if (nq == 1) return std::make_pair(mul_mat_q8_KV_q8_KV_1, 1);
            MAKE_FUNCS_ONLY_NRC(mul_mat_q8_KV_q8_KV, nq);
#else
#ifdef HAVE_FANCY_SIMD
            if (nq%16 == 0) return std::make_pair(mul_mat_q8_KV_q8_KV<16>, 16);
#endif
            if (nq == 1) return std::make_pair(mul_mat_q8_KV_q8_KV_1<1>, 1);
            MAKE_FUNCS_ONLY_NRC(mul_mat_q8_KV_q8_KV, nq);
#endif
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ80R8<D, k_step>>) {
#ifdef __aarch64__
            MAKE_FUNCS_ONLY_NRC(mul_mat_q8_0_r8_q8_0, nq);
#else
            MAKE_FUNCS_ONLY_NRC(mul_mat_q8_0_r8_q8_1, nq);
#endif
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ8KVR8<D, k_step>>) {
            MAKE_FUNCS_ONLY_NRC(mul_mat_q8_KV_r8_q8_KV, nq);
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ60<D, k_step>>) {
#ifdef __aarch64__
            MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerQ60, nq);
#else
            MAKE_FUNCS(mul_mat_qX_1_q8_1_T<Q6_0_1_Unpacker, nq);
#endif
        }
#if GGML_IQK_FA_ALL_QUANTS
        else if constexpr (std::is_same_v<KHelper, HelperQ40<D, k_step>>) {
#ifdef __aarch64__
            MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerQ40, nq);
#else
            MAKE_FUNCS(mul_mat_qX_0_q8_0_T<Q4_0_Unpacker, nq);
#endif
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ41<D, k_step>>) {
#ifdef __aarch64__
            MAKE_FUNCS(mul_mat_qX_1_q8_1<DequantizerQ41, nq);
#else
            MAKE_FUNCS(mul_mat_qX_1_q8_1_T<Q4_1_Unpacker, nq);
#endif
        }
        else if constexpr (std::is_same_v<KHelper, HelperIQ4nl<D, k_step>>) {
#ifdef __aarch64__
            MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerIQ4NL, nq);
#else
            MAKE_FUNCS(mul_mat_qX_1_q8_1_T<IQ4_NL_Unpacker, nq);
#endif
        }
#endif
        else {
            GGML_ASSERT(false);
        }
        return std::make_pair<mul_mat_t, int>(nullptr, 0);
    }

    template <typename KHelper, typename block_q8>
    static inline void mul_mask_kq(const KHelper& kh, int stride_m,
            const block_q8 * q, const char * mask, FlashMS<q_step, k_step>& fms) {
        constexpr int kMaxQ = 8;
        static_assert(q_step < kMaxQ || q_step%kMaxQ == 0);
        auto [mul_mat, nrc_q] = mul_mat_kernel<KHelper>(q_step);
        DataInfo info{fms.cache, (const char *)q, k_step, (D/KHelper::block_size_q)*sizeof(block_q8), 0, 1, nullptr};
        for (int iq = 0; iq < q_step/nrc_q; ++iq) {
            mul_mat(D, kh.block, kh.stride, info, k_step);
            info.cur_y += nrc_q;
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
        auto [mul_mat, nrc_q] = mul_mat_kernel<KHelper>(nq);
        DataInfo info{fms.cache, (const char *)q, k_step, (D/KHelper::block_size_q)*sizeof(block_q8), 0, 1, nullptr};
        for (int iq = 0; iq < nq/nrc_q; ++iq) {
            mul_mat(D, kh.block, kh.stride, info, k_step);
            info.cur_y += nrc_q;
        }
        int iq = nrc_q*(nq/nrc_q);
        if (iq < nq) {
            auto [mul_mat1, nrc_q1] = mul_mat_kernel<KHelper>(nq - iq);
            GGML_ASSERT(nrc_q1 == nq - iq);
            mul_mat1(D, kh.block, kh.stride, info, k_step);
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

template <int Dk, int Dv, int q_step, int k_step, typename KHelper, typename VHelper, typename KQHelper>
void compute_helper(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
        FlashMS<q_step, k_step>& fms,
        FlashQKV<Dv, q_step, k_step>& fqkv,
        const float * q, const char * mask, float * qkv,
        float * M, float * S) {
#ifdef __aarch64__
    float16_t q_f16[Dk*q_step];
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
            KQHelper::multiply_mask_kq(kh, Dk, stride_m, q_f16, mr, fms);
#else
            KQHelper::multiply_mask_kq(kh, stride_q, stride_m, q, mr, fms);
#endif
            fqkv.accumulate_qkv(vh, fms);
            kh.next_block();
            vh.next_block();
            mr += k_step*sizeof(ggml_half);
        }
        fqkv.normalize_and_store(fms, stride_qkv, qkv, M, S);

        q    += q_step*stride_q;
        mask += q_step*stride_m;
        qkv  += q_step*stride_qkv;
        if (M && S) { M += q_step; S += q_step; }
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
            KQHelper::multiply_mask_kq(n_left, kh, Dk, stride_m, q_f16, mr, fms);
#else
            KQHelper::multiply_mask_kq(n_left, kh, stride_q, stride_m, q, mr, fms);
#endif
            fqkv.accumulate_qkv(n_left, vh, fms);
            kh.next_block();
            vh.next_block();
            mr += k_step*sizeof(ggml_half);
        }
        fqkv.normalize_and_store(fms, n_left, stride_qkv, qkv, M, S);
    }
}

template <int Dk, int Dv, int q_step, int k_step, typename KHelper, typename VHelper, typename KQHelper>
void compute_helper_q(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
        FlashMS<q_step, k_step>& fms,
        FlashQKV<Dv, q_step, k_step>& fqkv,
        const float * q, const char * mask, float * qkv,
        float * M, float * S) {
    typename KHelper::block_q8 q8[q_step*(Dk/KHelper::block_size_q)];
#if FA_TIMING
    Perf perf(false);
#endif
    for (int i1 = 0; i1 < nq1/q_step; ++i1) {
#if FA_TIMING
        auto t1 = Perf::cur_time();
#endif
        fms.init_qstep();
        kh.reset_block();
        vh.reset_block();
        HelperQ80<Dk, QK8_0>::convert(q_step, stride_q, q, q8);
#if FA_TIMING
        perf.accum_nolock(0, t1);
#endif
        auto mr = mask;
        for (int k1 = 0; k1 < nk1/k_step; ++k1) {
#if FA_TIMING
            t1 = Perf::cur_time();
            KQHelper::mul_mask_kq(kh, stride_m, q8, mr, fms);
            perf.accum_nolock(1, t1);
            t1 = Perf::cur_time();
            fqkv.accumulate_qkv(vh, fms);
            perf.accum_nolock(2, t1);
#else
            KQHelper::mul_mask_kq(kh, stride_m, q8, mr, fms);
            fqkv.accumulate_qkv(vh, fms);
#endif
            kh.next_block();
            vh.next_block();
            mr += k_step*sizeof(ggml_half);
        }
#if FA_TIMING
        t1 = Perf::cur_time();
        fqkv.normalize_and_store(fms, stride_qkv, qkv, M, S);
        perf.accum_nolock(3, t1);
#else
        fqkv.normalize_and_store(fms, stride_qkv, qkv, M, S);
#endif

        q    += q_step*stride_q;
        mask += q_step*stride_m;
        qkv  += q_step*stride_qkv;
        if (M && S) { M += q_step; S += q_step; }
    }
    int n_left = nq1 - q_step*(nq1/q_step);
    if (n_left > 0) {
        fms.init_qstep();
        kh.reset_block();
        vh.reset_block();
        HelperQ80<Dk, QK8_0>::convert(n_left, stride_q, q, q8);
        auto mr = mask;
        for (int k1 = 0; k1 < nk1/k_step; ++k1) {
            KQHelper::mul_mask_kq(n_left, kh, stride_m, q8, mr, fms);
            fqkv.accumulate_qkv(n_left, vh, fms);
            kh.next_block();
            vh.next_block();
            mr += k_step*sizeof(ggml_half);
        }
        fqkv.normalize_and_store(fms, n_left, stride_qkv, qkv, M, S);
    }
#if FA_TIMING
    Perf::instance().add(perf);
#endif
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
template <int Dk, int Dv, int q_step, int k_step>
struct FlashAttn {
    static_assert(Dk%F16::block_size == 0 && Dk <= 576);
    static_assert(Dv%F16::block_size == 0 && Dv <= 512);
    static_assert(k_step%F16::block_size == 0);
    static_assert(q_step <= 4 || q_step%4 == 0);

    FlashAttn(float scale, float softcap) : fms(scale, softcap) {}

    template <typename KHelper, typename VHelper>
    void compute(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
            const float * q, const char * mask, float * qkv, [[maybe_unused]] float * M, [[maybe_unused]] float * S) {
        if constexpr (std::is_same_v<KHelper, HelperQ40<Dk, k_step>> || std::is_same_v<KHelper, HelperQ41<Dk, k_step>> ||
                      std::is_same_v<KHelper, HelperIQ4nl<Dk, k_step>> ||
                      std::is_same_v<KHelper, HelperQ60<Dk, k_step>>) {
            compute_helper_q<Dk, Dv, q_step, k_step, KHelper, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                    kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, M, S);
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ80<Dk, k_step>>) {
            if (nq1 >= 8) {
#if FA_TIMING
                auto t1 = Perf::cur_time();
                HelperQ80R8<Dk, k_step> khr4(nk1, kh);
                Perf::instance().accum(4, t1);
#else
                HelperQ80R8<Dk, k_step> khr4(nk1, kh);
#endif
                compute_helper_q<Dk, Dv, q_step, k_step, HelperQ80R8<Dk, k_step>, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                        khr4, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, M, S);
            } else{
                compute_helper_q<Dk, Dv, q_step, k_step, KHelper, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                        kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, M, S);
            }
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ8KV<Dk, k_step>>) {
            if (nq1 >= 8) {
#if FA_TIMING
                auto t1 = Perf::cur_time();
                HelperQ8KVR8<Dk, k_step> khr4(nk1, kh);
                Perf::instance().accum(4, t1);
#else
                HelperQ8KVR8<Dk, k_step> khr4(nk1, kh);
#endif
                compute_helper_q<Dk, Dv, q_step, k_step, HelperQ8KVR8<Dk, k_step>, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                        khr4, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, M, S);
            } else{
                compute_helper_q<Dk, Dv, q_step, k_step, KHelper, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                        kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, M, S);
            }
        } else {
            compute_helper<Dk, Dv, q_step, k_step, KHelper, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                    kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, M, S);
        }
    }

    FlashMS<q_step, k_step>      fms;
    FlashQKV<Dv, q_step, k_step> fqkv;

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

    inline void load_8(int l1, __m512bh * vk) const {
        for (int k = 0; k < 8; ++k) load(l1 + k, vk + k*D/32);
    }
};

template <int D, int q_step, int k_step>
struct FlashQKbf16 {
    //static_assert(D%32 == 0 && D <= 256);
    static_assert(D%32 == 0 && D <= 576);
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

    static inline __m128 hsum_float_4x4(__m128 * a) {
        for (int i = 0; i < 2; ++i) a[i] = _mm_add_ps(_mm_unpacklo_ps(a[i], a[i+2]), _mm_unpackhi_ps(a[i], a[i+2]));
        return _mm_add_ps(_mm_unpacklo_ps(a[0], a[1]), _mm_unpackhi_ps(a[0], a[1]));
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

    static inline void mult_mask_kq_4(int l1, int m1, const ggml_bf16_t * q,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        auto qr = q + m1*D;
        for (int i = 0; i < D/32; ++i) qv[i] = __m512bh(_mm512_loadu_si512((const __m512i *)qr + i));
        __m128 sum[4];
        for (int k = 0; k < 4; ++k) {
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+k*(D/32)], qv[i]);
            auto aux = _mm256_add_ps(_mm512_castps512_ps256(vsum), _mm512_extractf32x8_ps(vsum, 1));
            sum[k] = _mm_add_ps(_mm256_castps256_ps128(aux), _mm256_extractf128_ps(aux, 1));
        }
        //auto sum4 = _mm_mask_blend_ps(m8, hsum_float_4x4(sum), _mm_set1_ps(-INFINITY));
        //_mm_storeu_ps(fms.cache + k_step*m1 + l1, sum4);
        _mm_storeu_ps(fms.cache + k_step*m1 + l1, hsum_float_4x4(sum));
    }

    static IQK_ALWAYS_INLINE __m256 hsum_float_8x8(__m256 * accm) {
        for (int i = 0; i < 4; ++i) {
            accm[i] = _mm256_add_ps(_mm256_permute2f128_ps(accm[i], accm[i+4], 0x20), _mm256_permute2f128_ps(accm[i], accm[i+4], 0x31));
            //accm[i] = _mm256_set_m128(_mm_add_ps(_mm256_castps256_ps128(accm[i+4]), _mm256_extractf128_ps(accm[i+4], 1)),
            //                          _mm_add_ps(_mm256_castps256_ps128(accm[i+0]), _mm256_extractf128_ps(accm[i+0], 1)));
        }
        for (int i = 0; i < 2; ++i) accm[i] = _mm256_add_ps(_mm256_unpacklo_ps(accm[i], accm[i+2]), _mm256_unpackhi_ps(accm[i], accm[i+2]));
        return _mm256_add_ps(_mm256_unpacklo_ps(accm[0], accm[1]), _mm256_unpackhi_ps(accm[0], accm[1]));
    }

    static inline void mult_mask_kq_8(int l1, int m1, const ggml_bf16_t * q,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        auto qr = q + m1*D;
        for (int i = 0; i < D/32; ++i) qv[i] = __m512bh(_mm512_loadu_si512((const __m512i *)qr + i));
        __m256 sum[8];
        for (int k = 0; k < 8; ++k) {
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+k*(D/32)], qv[i]);
            sum[k] = _mm256_add_ps(_mm512_castps512_ps256(vsum), _mm512_extractf32x8_ps(vsum, 1));
        }
        _mm256_storeu_ps(fms.cache + k_step*m1 + l1, hsum_float_8x8(sum));
    }

    static inline void mult_mask_kq_one(int l1, int m1, const ggml_bf16_t * q,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        auto qr = q + m1*D;
        for (int i = 0; i < D/32; ++i) qv[i] = __m512bh(_mm512_loadu_si512((const __m512i*)qr + i));
        auto vsum = _mm512_setzero_ps();
        for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i], qv[i]);
        fms.cache[k_step*m1 + l1 + 0] = _mm512_reduce_add_ps(vsum);
        vsum = _mm512_setzero_ps();
        for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+D/32], qv[i]);
        fms.cache[k_step*m1 + l1 + 1] = _mm512_reduce_add_ps(vsum);
    }

#if FA_TIMING
    template <typename KHelper>
    static inline void multiply_mask_kq(const KHelper& kh, int stride_m, const ggml_bf16_t * q,
            const char * mask, FlashMS<q_step, k_step>& fms, Perf& perf) {
        auto t1 = Perf::cur_time();
#else
    template <typename KHelper>
    static inline void multiply_mask_kq(const KHelper& kh, int stride_m, const ggml_bf16_t * q,
            const char * mask, FlashMS<q_step, k_step>& fms) {
#endif
        if constexpr (q_step == 1) {
            __m512bh vq[D/32];
            __m512bh vk[D/32];
            __m256   sum[8];
            for (int i = 0; i < D/32; ++i) vq[i] = __m512bh(_mm512_loadu_si512((const __m512i *)q + i));
            for (int l = 0; l < k_step; l += 8) {
                for (int k = 0; k < 8; ++k) {
                    kh.load(l+k, vk);
                    auto vsum = _mm512_setzero_ps();
                    for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vk[i], vq[i]);
                    sum[k] = _mm256_add_ps(_mm512_castps512_ps256(vsum), _mm512_extractf32x8_ps(vsum, 1));
                }
                _mm256_storeu_ps(fms.cache + l, hsum_float_8x8(sum));
            }
        }
        else {
            __m512bh qv[D/32];
            if constexpr (D <= 128) {
                __m512bh vkh[D/4];
                for (int l1 = 0; l1 < k_step; l1 += 8) {
                    kh.load_8(l1, vkh);
                    for (int j = 0; j < q_step; ++j) mult_mask_kq_8(l1, j, q, qv, vkh, fms);
                }
            } else {
                __m512bh vkh[D/16];
                for (int l1 = 0; l1 < k_step; l1 += 2) {
                    kh.load_2(l1, vkh);
                    for (int j = 0; j < q_step; ++j) mult_mask_kq_one(l1, j, q, qv, vkh, fms);
                }
            }
        }
#if FA_TIMING
        perf.accum_nolock(1, t1);
        t1 = Perf::cur_time();
#endif
        F16::Data vk[k_step/16];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
#if FA_TIMING
        perf.accum_nolock(2, t1);
#endif
    }

    template <typename KHelper>
    static inline void multiply_mask_kq(int nq, const KHelper& kh, int stride_m, const ggml_bf16_t * q,
            const char * mask, FlashMS<q_step, k_step>& fms) {
        {
            __m512bh qv[D/32];
            if constexpr (D <= 128) {
                __m512bh vkh[D/8];
                for (int l1 = 0; l1 < k_step; l1 += 4) {
                    kh.load_4(l1, vkh);
                    for (int j = 0; j < nq; ++j) mult_mask_kq_4(l1, j, q, qv, vkh, fms);
                }
            } else {
                __m512bh vkh[D/16];
                for (int l1 = 0; l1 < k_step; l1 += 2) {
                    kh.load_2(l1, vkh);
                    for (int j = 0; j < nq; ++j) mult_mask_kq_one(l1, j, q, qv, vkh, fms);
                }
            }
        }
        F16::Data vk[k_step/16];
        for (int j = 0; j < nq; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
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

    static inline void convert(int nq, int stride_q, const float * q, ggml_bf16_t * bf16) {
        auto qr = q;
        for (int j = 0; j < nq; ++j) {
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

template <int Dk, int Dv, int q_step, int k_step>
struct FlashAttnBF16 {
    //static_assert(Dk%32 == 0 && Dk <= 256);
    //static_assert(Dv%32 == 0 && Dv <= 256);
    static_assert(Dk%32 == 0 && Dk <= 576);
    static_assert(Dv%32 == 0 && Dv <= 512);
    static_assert(k_step%32 == 0);
    static_assert(q_step <= 4 || q_step%4 == 0);

    FlashAttnBF16(float scale, float softcap) : fms(scale, softcap) {}

    template <typename KHelper, typename VHelper>
    void compute(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
            const float * q, const char * mask, float * qkv, [[maybe_unused]] float * M, [[maybe_unused]] float * S) {
        ggml_bf16_t q_bf16[q_step*Dk];
#if FA_TIMING
        Perf perf(false);
#endif
        for (int i1 = 0; i1 < nq1/q_step; ++i1) {
#if FA_TIMING
            auto t1 = Perf::cur_time();
#endif
            fms.init_qstep();
            kh.reset_block();
            vh.reset_block();
            FlashQKbf16<Dk, q_step, k_step>::convert(stride_q, q, q_bf16);
#if FA_TIMING
            perf.accum_nolock(0, t1);
#endif
            auto mr = mask;
            for (int k1 = 0; k1 < nk1/k_step; ++k1) {
#if FA_TIMING
                //t1 = Perf::cur_time();
                FlashQKbf16<Dk, q_step, k_step>::multiply_mask_kq(kh, stride_m, q_bf16, mr, fms, perf);
                //perf.accum_nolock(1, t1);
                t1 = Perf::cur_time();
                fqkv.accumulate_qkv(vh, fms);
                perf.accum_nolock(3, t1);
#else
                FlashQKbf16<Dk, q_step, k_step>::multiply_mask_kq(kh, stride_m, q_bf16, mr, fms);
                fqkv.accumulate_qkv(vh, fms);
#endif
                kh.next_block();
                vh.next_block();
                mr += k_step*sizeof(ggml_half);
            }
#if FA_TIMING
            t1 = Perf::cur_time();
#endif
            fqkv.normalize_and_store(fms, stride_qkv, qkv, M, S);
#if FA_TIMING
            perf.accum_nolock(4, t1);
#endif

            q    += q_step*stride_q;
            mask += q_step*stride_m;
            qkv  += q_step*stride_qkv;
        }
        int n_left = nq1 - q_step*(nq1/q_step);
        if (n_left > 0) {
            fms.init_qstep();
            kh.reset_block();
            vh.reset_block();
            FlashQKbf16<Dk, q_step, k_step>::convert(n_left, stride_q, q, q_bf16);
            auto mr = mask;
            for (int k1 = 0; k1 < nk1/k_step; ++k1) {
                FlashQKbf16<Dk, q_step, k_step>::multiply_mask_kq(n_left, kh, stride_m, q_bf16, mr, fms);
                fqkv.accumulate_qkv(n_left, vh, fms);
                kh.next_block();
                vh.next_block();
                mr += k_step*sizeof(ggml_half);
            }
            fqkv.normalize_and_store(fms, n_left, stride_qkv, qkv, M, S);
        }
#if FA_TIMING
        Perf::instance().add(perf);
#endif
    }

    FlashMS<q_step, k_step>      fms;
    FlashQKV<Dv, q_step, k_step> fqkv;
};
#endif

template <int Dk, int Dv, int k_step, typename KHelper, typename VHelper>
inline void iqk_flash_helper(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
                        const float * q, const char * mask, float scale, float softcap, float * qkv, float * M, float * S) {

    if (nk1 >= 256) { //4096) {
        if (nq1 >= 64) {
            FlashAttn<Dk, Dv, 64, k_step> fa(scale, softcap);
            fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            return;
        }
        if (nq1 >= 32) {
            FlashAttn<Dk, Dv, 32, k_step> fa(scale, softcap);
            fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            return;
        }
        if (nq1 >= 16) {
            FlashAttn<Dk, Dv, 16, k_step> fa(scale, softcap);
            fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            return;
        }
    }
    if (nq1 >= 8) {
        FlashAttn<Dk, Dv, 8, k_step> fa(scale, softcap);
        fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
    }
    else {
        FlashAttn<Dk, Dv, 1, k_step> fa(scale, softcap);
        fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
    }
}

#ifdef __AVX512BF16__
template <int Dk, int Dv, int k_step>
inline void iqk_flash_helper_T(int nq1, int nk1, int stride_q, int stride_k, int stride_v, int stride_m, int stride_qkv,
                        const float * q, const char * k, const char * v, const char * mask,
                        float scale, float softcap, float * qkv, float * M, float * S) {
    HelperBF16<Dk, k_step> kh(k, stride_k);
    HelperBF16<Dv, k_step> vh(v, stride_v);
    if (nk1 >= 4096) {
        if (nq1 >= 64) {
            FlashAttnBF16<Dk, Dv, 64, k_step> fa(scale, softcap);
            fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            return;
        }
        else if (nq1 >= 16) {
            FlashAttnBF16<Dk, Dv, 16, k_step> fa(scale, softcap);
            fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            return;
        }
    }
    if (nq1 >= 8) {
        FlashAttnBF16<Dk, Dv, 8, k_step> fa(scale, softcap);
        fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
    } else {
        FlashAttnBF16<Dk, Dv, 1, k_step> fa(scale, softcap);
        fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
    }
}
#endif

template <int Dk, int Dv, int k_step, typename KHelper>
inline void iqk_flash_helper_T(KHelper& kh, ggml_type type_v,
                        int nq1, int nk1, int stride_q, int stride_v, int stride_m, int stride_qkv,
                        const float * q, const char * v, const char * mask,
                        float scale, float softcap, float * qkv, float * M, float * S) {

    switch (type_v) {
        case GGML_TYPE_F16: {
            HelperF16<Dv, k_step> vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        } break;
#ifdef __AVX512BF16__
        case GGML_TYPE_BF16: {
            HelperBF16<Dv, k_step> vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        } break;
#endif
        case GGML_TYPE_Q8_0: {
            HelperQ80<Dv, k_step> vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        } break;
        case GGML_TYPE_Q8_KV: {
            HelperQ8KV<Dv, k_step> vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        } break;
        case GGML_TYPE_Q6_0: {
            HelperQ60<Dv, k_step> vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        } break;
#if GGML_IQK_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0: {
            HelperQ40<Dv, k_step> vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        } break;
        case GGML_TYPE_Q4_1: {
            HelperQ41<Dv, k_step> vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        } break;
        case GGML_TYPE_IQ4_NL: {
            HelperIQ4nl<Dv, k_step> vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        } break;
#endif
        default: break;
    }
}

template <int Dk, int Dv, int k_step>
inline void iqk_flash_helper_T(ggml_type type_k, ggml_type type_v,
                        int nq1, int nk1, int stride_q, int stride_k, int stride_v, int stride_m, int stride_qkv,
                        const float * q, const char * k, const char * v, const char * mask,
                        float scale, float softcap, float * qkv, float * M, float * S) {

    switch (type_k) {
        case GGML_TYPE_F16: {
            HelperF16<Dk, k_step> kh(k, stride_k);
            iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, M, S);
        } break;
        case GGML_TYPE_Q8_0: {
            HelperQ80<Dk, k_step> kh(k, stride_k);
            iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, M, S);
        } break;
        case GGML_TYPE_Q8_KV: {
            HelperQ8KV<Dk, k_step> kh(k, stride_k);
            iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, M, S);
        } break;
        case GGML_TYPE_Q6_0: {
            HelperQ60<Dk, k_step> kh(k, stride_k);
            iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, M, S);
        } break;
#if GGML_IQK_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0: {
            HelperQ40<Dk, k_step> kh(k, stride_k);
            iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, M, S);
        } break;
        case GGML_TYPE_Q4_1: {
            HelperQ41<Dk, k_step> kh(k, stride_k);
            iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, M, S);
        } break;
        case GGML_TYPE_IQ4_NL: {
            HelperIQ4nl<Dk, k_step> kh(k, stride_k);
            iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, M, S);
        } break;
#endif
        default: break;
    }

}

inline bool flash_attn_is_supported(ggml_type type) {
#ifdef __AVX512BF16__
    if (type == GGML_TYPE_BF16) return true;
#endif
#if GGML_IQK_FA_ALL_QUANTS
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 ||
        type == GGML_TYPE_Q6_0 || type == GGML_TYPE_IQ4_NL) return true;
#else
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q6_0 || type == GGML_TYPE_Q8_KV) return true;
#endif
    return false;
}

template <int step_k, typename KHelper, typename VHelper>
inline void iqk_deepseek_helper(KHelper& kh, VHelper& vh,
                        int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
                        const float * q, const char * mask, float scale, float softcap, float * qkv, float * M, float * S) {
    if (nq1 % 8 == 0) {
        FlashAttn<576, 512, 8, step_k> fa(scale, softcap);
        fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
    } else {
        FlashAttn<576, 512, 1, step_k> fa(scale, softcap);
        fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
    }
}

template <int step_k>
inline bool iqk_deepseek_helper(ggml_type type_k,
                        int nq1, int nk1, int stride_q, int stride_k, int stride_v, int stride_m, int stride_qkv,
                        const float * q, const char * k, const char * v, const char * mask,
                        float scale, float softcap, float * qkv, float * M, float * S) {
    if (type_k == GGML_TYPE_Q8_0) {
        HelperQ80<576, step_k> kh((const char *)k, stride_k);
        HelperQ80<512, step_k> vh((const char *)v, stride_v);
        iqk_deepseek_helper<step_k>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        return true;
    }
    if (type_k == GGML_TYPE_Q6_0) {
        HelperQ60<576, step_k> kh((const char *)k, stride_k);
        HelperQ60<512, step_k> vh((const char *)v, stride_v);
        iqk_deepseek_helper<step_k>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        return true;
    }
    if (type_k == GGML_TYPE_Q8_KV) {
        HelperQ8KV<576, step_k> kh((const char *)k, stride_k);
        HelperQ8KV<512, step_k> vh((const char *)v, stride_v);
        iqk_deepseek_helper<step_k>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        return true;
    }
    if (type_k == GGML_TYPE_F16) {
        HelperF16<576, step_k> kh((const char *)k, stride_k);
        HelperF16<512, step_k> vh((const char *)v, stride_v);
        iqk_deepseek_helper<step_k>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        return true;
    }
#ifdef __AVX512BF16__
    if (type_k == GGML_TYPE_BF16) {
        HelperBF16<576, step_k> kh((const char *)k, stride_k);
        HelperBF16<512, step_k> vh((const char *)v, stride_v);
        if (nq1 % 8 == 0) {
            FlashAttnBF16<576, 512, 8, step_k> fa(scale, softcap);
            fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
        } else {
            FlashAttnBF16<576, 512, 1, step_k> fa(scale, softcap);
            fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
        }
        return true;
    }
#endif
    return false;
}

}

#include "iqk_flash_impl.h"

bool iqk_flash_attn_impl(int int_type_k,         // type of k
                         int int_type_v,         // type of v
                         int Dk,                 // K head size
                         int Dv,                 // V head size
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
                         float       * qkv,      // v*softmax(scale*(k*q))
                         float * M, float * S) {

    if (!mask || nk1%32 != 0) return false; // the implementation assumes mask is not null and nk is a multiple of 32

    auto type_k = ggml_type(int_type_k);
    auto type_v = ggml_type(int_type_v);

    if (Dk == 576 && Dv == 512) {
        GGML_ASSERT(type_k == type_v);
        stride_q /= sizeof(float); // q stride as float
        return iqk_deepseek_helper<32>(type_k, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                        q, (const char *)k, (const char *)v, (const char *)mask, scale, softcap, qkv, M, S);
    }

    if (!flash_attn_is_supported(type_k) || !flash_attn_is_supported(type_v)) return false;
    if (Dk != Dv && Dk != 192 && Dv != 128) return false;
    if (Dv != 64 && Dv != 96 && Dv != 128 && Dv != 256) return false;
    if (Dk != 64 && Dk != 96 && Dk != 128 && Dk != 192 && Dk != 256) return false;

    auto ck = (const char *)k;
    auto cv = (const char *)v;
    auto cm = (const char *)mask;

    stride_q /= sizeof(float); // q stride as float

#ifdef __AVX512BF16__
    if (type_k == GGML_TYPE_BF16) {
        if (nk1%64 == 0) {
            if (type_v != GGML_TYPE_BF16) return false; // we do not support mixing bf16 k-cache with other types
            switch (Dk) {
                case 64:
                    iqk_flash_helper_T< 64, 64, 64>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
                case 96:
                    iqk_flash_helper_T< 96, 96, 64>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
                case 128:
                    iqk_flash_helper_T<128, 128, 64>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
                case 192:
                    iqk_flash_helper_T<192, 128, 64>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
                case 256:
                    iqk_flash_helper_T<256, 256, 64>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
                default:
                    return false;
            }
            return true;
        }
        if (type_v != GGML_TYPE_BF16) return false; // we do not support mixing bf16 k-cache with other types
        switch (Dk) {
            case 64:
                iqk_flash_helper_T< 64, 64, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            case 96:
                iqk_flash_helper_T< 96, 96, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            case 128:
                iqk_flash_helper_T<128, 128, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            case 192:
                iqk_flash_helper_T<192, 128, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            case 256:
                iqk_flash_helper_T<256, 256, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            default:
                return false;
        }

        return true;
    }
#endif

    if (nk1%64 == 0) {
        switch (Dk) {
            case 64:
                iqk_flash_helper_T< 64, 64, 64>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
                // Disable until we fix accumulate_qkv for odd D/16
                //case 80:
                //    iqk_flash_helper_T< 80, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
            case 96:
                iqk_flash_helper_T< 96, 96, 64>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
                // Disable until we fix accumulate_qkv for odd D/16
                //case 112:
                //    iqk_flash_helper_T<112, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
            case 128:
                iqk_flash_helper_T<128, 128, 64>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            case 192:
                iqk_flash_helper_T<192, 128, 64>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            case 256:
                iqk_flash_helper_T<256, 256, 64>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            default:
                return false;
        }
        return true;
    }
    switch (Dk) {
        case 64:
            iqk_flash_helper_T< 64, 64, 32>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
        // Disable until we fix accumulate_qkv for odd D/16
        //case 80:
        //    iqk_flash_helper_T< 80, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
        case 96:
            iqk_flash_helper_T< 96, 96, 32>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
        // Disable until we fix accumulate_qkv for odd D/16
        //case 112:
        //    iqk_flash_helper_T<112, 32>(nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv); break;
        case 128:
            iqk_flash_helper_T<128, 128, 32>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
        case 192:
            iqk_flash_helper_T<192, 128, 32>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
        case 256:
            iqk_flash_helper_T<256, 256, 32>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
        default:
            return false;
    }

    return true;
}

#else  // IQK_IMPLEMENT

bool iqk_mul_mat(int, long, long, long, int, const void *, long, int, const void *, long, float *, long, int, int) {
    return false;
}

bool iqk_mul_mat_4d(long /*Nx*/, long /*Ny*/, long /*ne00*/,
        long /*ne02*/, long /*ne03*/, long /*ne12*/, long /*ne13*/,
        long /*nb02*/, long /*nb03*/, long /*nb12*/, long /*nb13*/, long /*nb2*/, long /*nb3*/,
        int /*typeA*/, const void * /*A*/, long /*strideA*/,
        int /*typeB*/, const void * /*B*/, long /*strideB*/,
        float * /*C*/, long /*stride_C*/, int /*ith*/, int /*nth*/) {
    return false;
}

bool iqk_mul_mat_moe(long, long, long, int, int, const void *, long, int, const void *, long, float *, long, long,
        const void *, int, int) {
    return false;
}

bool iqk_moe_fused_up_gate(long /*Nx*/, long /*Ny*/, long /*ne00*/, int /*ne11*/, int /*unary_op*/,
        int /*typeA*/, const void * /*Aup*/, const void * /*Agate*/, long /*strideA*/,
        int /*typeB*/, const void * /*B*/, long /*strideB*/,
        float * /*C*/, long /*nb1*/, long /*nb2*/, const void * /*vrow_mapping*/, int /*ith*/, int /*nth*/) {
    return false;
}

#endif
