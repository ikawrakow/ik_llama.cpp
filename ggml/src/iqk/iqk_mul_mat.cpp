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
#include "iqk_flash_impl.h"
#include "iqk_gemm_floats.h"
#include "iqk_gemm_kquants.h"
#include "iqk_gemm_iquants.h"
#include "iqk_gemm_iqk_quants.h"
#include "iqk_gemm_1bit.h"
#include "iqk_gemm_legacy_quants.h"

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

namespace {

struct MulMat {
    std::array<mul_mat_t, IQK_MAX_NY> funcs = {};
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
            case GGML_TYPE_IQ5_KS_R4:
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
            case GGML_TYPE_IQ5_KS_R4:
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
};

}

extern "C" IQK_API bool iqk_mul_mat(long Nx, long Ny, long ne00,
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

extern "C" IQK_API bool iqk_mul_mat_4d(long Nx, long Ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        long nb02, long nb03, long nb12, long nb13, long nb2, long nb3,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth) {

    auto r2 = ne12 / ne02;
    auto r3 = ne13 / ne03;

    if (ne13 == 1 && Ny == 1 && r2 > 1) {
        if (Nx >= 256 && Nx%32 == 0) {
            int nx32 = Nx/32;
            int nchunk = nx32*ne02;
            if (r2 <= IQK_MAX_NY) {
                MulMat mm;
                if (!MulMat::prepare(typeA, typeB, ne00, mm, r2)) return false;
                int ny = mm.funcs.size();
                while (ny > 0 && !mm.funcs[ny-1]) --ny;
                if (ny >= r2) {
                    nchunk = nx32*ne02;
                    for (int ichunk = ith; ichunk < nchunk; ichunk += nth) {
                        int i02 = ichunk/nx32;
                        int ix = 32*(ichunk - i02*nx32);
                        DataInfo info{C + ix + r2*i02*nb2, (const char *)B + r2*i02*nb12, (size_t)nb2, (size_t)nb12, 0, 1, nullptr, 0};
                        mm.funcs[r2-1](ne00, (const void *)((const char *)A + ix*strideA + i02*nb02), strideA, info, 32);
                    }
                    return true;
                }
            }
            for (int ichunk = ith; ichunk < nchunk; ichunk += nth) {
                int i02 = ichunk/nx32;
                int ix = ichunk - i02*nx32;
                if (!iqk_mul_mat(32, r2, ne00,
                            typeA, (const char *)A + 32*ix*strideA + i02*nb02, strideA,
                            typeB, (const char *)B + i02*r2*nb12, nb12,
                            C + 32*ix + r2*i02*nb2, nb2, 0, 1)) return false;

            }
            return true;
        }
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

extern "C" IQK_API bool iqk_mul_mat_moe(long Nx, long Ny, long ne00, int ne11,
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

extern "C" IQK_API bool iqk_moe_fused_up_gate(long Nx, long Ny, long ne00, int ne11, int unary_op,
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

#if defined __x86_64__

namespace {

bool MulMat::prepare(int typeA, int typeB, int ne00, MulMat& mm, int Ny) {

    (void)Ny;

    switch (typeA) {
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_BF16_R16:
            return iqk_set_kernels_float(ne00, typeA, typeB, mm.funcs);
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q2_K_R4:
        case GGML_TYPE_Q3_K_R4:
        case GGML_TYPE_Q4_K_R4:
        case GGML_TYPE_Q5_K_R4:
        case GGML_TYPE_Q6_K_R4:
        case GGML_TYPE_IQ4_XS_R8:
        case GGML_TYPE_Q8_K_R8:
        case GGML_TYPE_Q8_KV:
        case GGML_TYPE_Q8_KV_R8:
            return iqk_set_kernels_kquants(ne00, typeA, typeB, mm.funcs, mm.func16);
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_XXS_R4:
        case GGML_TYPE_IQ2_XS_R4:
        case GGML_TYPE_IQ2_S_R4:
        case GGML_TYPE_IQ3_XXS_R4:
        case GGML_TYPE_IQ3_S_R4:
            return ggml_type(typeB) == GGML_TYPE_Q8_K ? iqk_set_kernels_iquants(ne00, typeA, typeB, mm.funcs, mm.func16) : false;
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ2_K_R4:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ5_K_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_KS_R4:
            return iqk_set_kernels_iqk_quants(ne00, typeA, typeB, mm.funcs, mm.func16);
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_0_R8:
        case GGML_TYPE_Q5_0_R4:
        case GGML_TYPE_Q6_0_R4:
        case GGML_TYPE_Q8_0_R8:
        case GGML_TYPE_IQ4_NL_R4:
            return iqk_set_kernels_legacy_quants(ne00, typeA, typeB, mm.funcs, mm.func16);
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:
        case GGML_TYPE_IQ1_BN:
        case GGML_TYPE_IQ2_BN:
        case GGML_TYPE_IQ2_BN_R4:
            return iqk_set_kernels_1bit(ne00, typeA, typeB, mm.funcs, mm.func16);

        default:
            return false;
    }

    return false;
}

} // namespace


#else   // __aarch64__

namespace {

bool MulMat::prepare(int typeA, int typeB, int ne00, MulMat& m, int /*Ny*/) {

    switch (typeA) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F32:
            return iqk_set_kernels_float(ne00, typeA, typeB, m.funcs);
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q2_K_R4:
        case GGML_TYPE_Q3_K_R4:
        case GGML_TYPE_Q4_K_R4:
        case GGML_TYPE_Q5_K_R4:
        case GGML_TYPE_Q6_K_R4:
        case GGML_TYPE_IQ4_XS_R8:
        case GGML_TYPE_Q8_K_R8:
        case GGML_TYPE_Q8_KV:
        case GGML_TYPE_Q8_KV_R8:
            return iqk_set_kernels_kquants(ne00, typeA, typeB, m.funcs, m.func16);
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ2_K_R4:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ5_K_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_KS_R4:
            return iqk_set_kernels_iqk_quants(ne00, typeA, typeB, m.funcs, m.func16);
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_XXS_R4:
        case GGML_TYPE_IQ2_XS_R4:
        case GGML_TYPE_IQ2_S_R4:
        case GGML_TYPE_IQ3_XXS_R4:
        case GGML_TYPE_IQ3_S_R4:
            return iqk_set_kernels_iquants(ne00, typeA, typeB, m.funcs, m.func16);
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_0_R8:
        case GGML_TYPE_Q5_0_R4:
        case GGML_TYPE_Q6_0_R4:
        case GGML_TYPE_Q8_0_R8:
        case GGML_TYPE_IQ4_NL_R4:
            return iqk_set_kernels_legacy_quants(ne00, typeA, typeB, m.funcs, m.func16);
        case GGML_TYPE_IQ1_BN:
        case GGML_TYPE_IQ2_BN:
        case GGML_TYPE_IQ2_BN_R4:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:
            return iqk_set_kernels_1bit(ne00, typeA, typeB, m.funcs, m.func16);
        default:
            return false;
    }

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
//inline float32x4_t v_tanh(float16x8_t x) {
//    auto val1 = v_tanh(vcvt_f32_f16(vget_low_f16(x)));
//    auto val2 = v_tanh(vcvt_f32_f16(vget_high_f16(x)));
//    return vcombine_f16(vcvt_f16_f32(val1), vcvt_f16_f32(val2));
//}
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

#ifdef GGML_IQK_FLASH_ATTENTION
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
    static inline void set4(const float * ptr, Data * vs) {
        auto v = set4(ptr);
        vs[0] = _mm512_shuffle_ps(v, v, 0x00);
        vs[1] = _mm512_shuffle_ps(v, v, 0x55);
        vs[2] = _mm512_shuffle_ps(v, v, 0xaa);
        vs[3] = _mm512_shuffle_ps(v, v, 0xff);
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
    static inline void set4(const float * ptr, Data * vs) {
        auto v = set4(ptr);
        vs[0] = _mm256_shuffle_ps(v, v, 0x00);
        vs[1] = _mm256_shuffle_ps(v, v, 0x55);
        vs[2] = _mm256_shuffle_ps(v, v, 0xaa);
        vs[3] = _mm256_shuffle_ps(v, v, 0xff);
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
    constexpr static ggml_type type = GGML_TYPE_Q8_KV;
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
    constexpr static ggml_type type = GGML_TYPE_Q8_0;
#ifdef HAVE_FANCY_SIMD
    using block_q8 = block_q8_2;
    constexpr static int block_size_q = QK8_2;
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

    static inline void convert(int nq, int stride_q, const float * q, block_q8_2 * y) {
        //GGML_ASSERT(nq <= step); Why did I have this assert?
        for (int i = 0; i < nq; ++i) {
            quantize_row_q8_2_x4(q, y, D);
            q += stride_q;
            y += D/QK8_2;
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
}

void * iqk_repack_k(int int_type_k, int nek0, int nek1, int nek2, int nek3, long nbk1, long nbk2, long nbk3,
        const void * data, void * work, int ith, int nth, int& repacked_type, uint64_t& row_size) {
    repacked_type = int_type_k;
    auto type_k = ggml_type(int_type_k);
    if (type_k != GGML_TYPE_Q8_0 || nek0%QK8_0 != 0) return work;
    int nrows = nek1*nek2*nek3;
    if (nrows%8 != 0) return work;
    repacked_type = int(GGML_TYPE_Q8_0_R8);
    row_size = ggml_row_size(GGML_TYPE_Q8_0, nek0);
    void * result = (char *)work + nrows*row_size;
    int npt = 8*((nrows/8 + nth - 1)/nth);
    int first = npt*ith;
    if (first >= nrows) return result;
    int last = std::min(first + npt, nrows);
    const block_q8_0 * x8[8];
    auto y = (block_q8_0_r8 *)((char *)work + first*row_size);
    int nblock = nek0/QK8_0;
#ifdef __ARM_NEON
    int8x16x2_t m0, m1, m2, m3;
#endif
    for (int row = first; row < last; row += 8) {
        int ik3 = row/(nek1*nek2);
        int ik2 = (row - ik3*nek1*nek2)/nek1;
        int ik1 = row - ik3*nek1*nek2 - ik2*nek1;
        auto this_data = (const char *)data + ik1*nbk1 + ik2*nbk2 + ik3*nbk3;
        for (int k = 0; k < 8; ++k) x8[k] = (const block_q8_0 *)(this_data + k*nbk1);
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
            //#ifdef HAVE_FANCY_SIMD
            //                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
            //                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
            //                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
            //                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
            //#endif
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
            //#ifdef HAVE_FANCY_SIMD
            //                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
            //                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
            //                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
            //                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
            //#endif
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

namespace {
template <int D, int step>
struct HelperQ80R8 : public BaseHelper<step> {
    using Base = BaseHelper<step>;
    constexpr static ggml_type type = GGML_TYPE_Q8_0_R8;
#ifdef __AVX2__
    constexpr static int block_size_q = QK8_2;
    using block_q8 = block_q8_2;
#else
    constexpr static int block_size_q = QK8_0;
    using block_q8 = block_q8_0;
#endif
    HelperQ80R8(const char * data, int stride) : Base(data, stride) {}
    HelperQ80R8(int nk, const HelperQ80<D, step>& q8) : Base(q8.data, q8.stride) {
        r4 = repack(nk, q8);
        Base::data = (const char *)r4.data();
        Base::stride = (D/QK8_0)*sizeof(block_q8_0);
    }

    static void repack(int nk, const char * q8_data, int q8_stride, block_q8_0_r8 * y) {
        constexpr int nblock = D/QK8_0;
        const block_q8_0 * x8[8];
#ifdef __ARM_NEON
        int8x16x2_t m0, m1, m2, m3;
#endif
        for (int row = 0; row < nk; row += 8) {
            for (int k = 0; k < 8; ++k) x8[k] = (const block_q8_0 *)(q8_data + (row + k)*q8_stride);
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
//#ifdef HAVE_FANCY_SIMD
//                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
//                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
//                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
//                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
//#endif
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
//#ifdef HAVE_FANCY_SIMD
//                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
//                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
//                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
//                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
//#endif
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
    }

    static std::vector<block_q8_0_r8> repack(int nk, const HelperQ80<D, step>& q8) {
        static_assert(D%QK8_0 == 0);
        GGML_ASSERT(nk%8 == 0);
        constexpr int nblock = D/QK8_0;
        std::vector<block_q8_0_r8> result(nblock * nk/8);
        auto y = result.data();
        repack(nk, q8.data, q8.stride, y);
        return result;
    }

    std::vector<block_q8_0_r8> r4;
};

// TODO: unite this with the above
template <int D, int step>
struct HelperQ8KVR8 : public BaseHelper<step> {
    using Base = BaseHelper<step>;
    constexpr static ggml_type type = GGML_TYPE_Q8_KV_R8;
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
//#ifdef HAVE_FANCY_SIMD
//                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
//                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
//                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
//                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
//#endif
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
    constexpr static ggml_type type = GGML_TYPE_Q4_0;
#if defined __AVX2__
    using block_q8 = block_q8_2;
    constexpr static int block_size_q = QK8_2;
#else
    using block_q8 = block_q8_0;
    constexpr static int block_size_q = QK8_0;
#endif
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
    using block_q8 = block_q8_2;
    constexpr static ggml_type type = GGML_TYPE_Q4_1;
    constexpr static int block_size_q = QK8_2;
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
    constexpr static ggml_type type = GGML_TYPE_IQ4_NL;
#ifdef __aarch64__
    using block_q8 = block_q8_0;
    HelperIQ4nl(const char * data, int stride) : Base(data, stride), values(vld1q_s8(iq4k_values)) {}
    constexpr static int block_size_q = QK8_0;
#else
    HelperIQ4nl(const char * data, int stride) : Base(data, stride) {}
#ifdef HAVE_FANCY_SIMD
    using block_q8 = block_q8_2;
    constexpr static int block_size_q = QK8_2;
#else
    using block_q8 = block_q8_0;
    constexpr static int block_size_q = QK8_0;
#endif
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
    constexpr static ggml_type type = GGML_TYPE_Q6_0;
#ifdef __aarch64__
    using block_q8 = block_q8_0;
    constexpr static int block_size_q = QK8_0;
#else
    using block_q8 = block_q8_2;
    constexpr static int block_size_q = QK8_2;
#endif
    using Base = BaseHelper<step>;
    HelperQ60(const char * data, int stride) : Base(data, stride) {}

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        int j = F16::block_size*i;
        auto dl = (const block_q6_0 *)Base::lblock(l1) + j/QK6_0;
#ifdef __aarch64__
        // TODO
        const float16_t * d16 = (const float16_t *)&dl->d;
        auto vd = F16::set1(d16[0]);
        //auto vd = F16::set1(*(const float16_t *)&dl->d);
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
        auto vzero = vdupq_n_f16(0);
        auto vinf  = vdupq_n_f32(-INFINITY);
        for (int l = 0; l < k_step/8; ++l) {
            auto vm = vceqq_f16(vzero, vld1q_f16((const float16_t *)mask + 8*l));
            auto vm1 = vzip1q_u16(vm, vm);
            auto vm2 = vzip2q_u16(vm, vm);
            auto kq  = vld1q_f32_x2(cache + k_step*j + 8*l);
            vk[2*l+0] = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(kq.val[0]), vm1),
                                                        vbicq_u32(vreinterpretq_u32_f32(vinf), vm1)));
            vk[2*l+1] = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(kq.val[1]), vm2),
                                                        vbicq_u32(vreinterpretq_u32_f32(vinf), vm2)));
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
    static inline __m256 apply_mask(int l, const char * mask, __m256 val, [[maybe_unused]] __m256 vinf) {
        return _mm256_add_ps(val, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)mask+l)));
        //auto m128 = _mm_loadu_si128((const __m128i *)mask+l);
        //m128 = _mm_cmpeq_epi16(m128, _mm_setzero_si128());
        //auto m256 = _mm256_cvtepi16_epi32(m128);
        //auto mf = _mm256_castsi256_ps(_mm256_or_si256(m256, _mm256_slli_epi32(m256, 16)));
        //return _mm256_or_ps(_mm256_and_ps(mf, val), _mm256_andnot_ps(mf, vinf));
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
#ifdef __AVX512F__
        if constexpr ((D/F16::block_size)%4 == 0) {
            F16::Data v[16];
            F16::Data vs[4];
            for (int i = 0; i < D/F16::block_size; i += 4) {
                for (int l = 0; l < k_step; l += 4) {
                    for (int k = 0; k < 4; ++k) {
                        vh.load(l+k, i+0, v[4*k+0], v[4*k+1]);
                        vh.load(l+k, i+2, v[4*k+2], v[4*k+3]);
                    }
                    for (int j = 0; j < q_step; ++j) {
                        auto R = qkv_cache + D*j;
                        auto s1 = F16::load(R + F16::block_size*(i+0));
                        auto s2 = F16::load(R + F16::block_size*(i+1));
                        auto s3 = F16::load(R + F16::block_size*(i+2));
                        auto s4 = F16::load(R + F16::block_size*(i+3));
                        F16::set4(fms.cache + k_step*j + l, vs);
                        for (int k = 0; k < 4; ++k) {
                            s1 = F16::fmadd(s1, v[4*k+0], vs[k]);
                            s2 = F16::fmadd(s2, v[4*k+1], vs[k]);
                            s3 = F16::fmadd(s3, v[4*k+2], vs[k]);
                            s4 = F16::fmadd(s4, v[4*k+3], vs[k]);
                        }
                        F16::store(R + F16::block_size*(i+0), s1);
                        F16::store(R + F16::block_size*(i+1), s2);
                        F16::store(R + F16::block_size*(i+2), s3);
                        F16::store(R + F16::block_size*(i+3), s4);
                    }
                }
            }
            return;
        }
#endif
        F16::Data v[8];
#ifdef __AVX2__
        F16::Data vs[4];
#endif
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
#ifdef __AVX2__
                    F16::set4(fms.cache + k_step*j + l, vs);
                    for (int k = 0; k < 4; ++k) {
                        s1 = F16::fmadd(s1, v[k+0], vs[k]);
                        s2 = F16::fmadd(s2, v[k+4], vs[k]);
                    }
#else
                    auto vs = F16::set4(fms.cache + k_step*j + l);
                    s1 = F16::fmadd_lane0(s1, v[0], vs);
                    s2 = F16::fmadd_lane0(s2, v[4], vs);
                    s1 = F16::fmadd_lane1(s1, v[1], vs);
                    s2 = F16::fmadd_lane1(s2, v[5], vs);
                    s1 = F16::fmadd_lane2(s1, v[2], vs);
                    s2 = F16::fmadd_lane2(s2, v[6], vs);
                    s1 = F16::fmadd_lane3(s1, v[3], vs);
                    s2 = F16::fmadd_lane3(s2, v[7], vs);
#endif
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
#ifdef __aarch64__
                for (int i = 0; i < D/F16::block_size; ++i) {
                    F16::store(qkv + F16::block_size*i, F16::load(R + F16::block_size*i));
                }
#else
                std::memcpy(qkv, R, D*sizeof(float));
#endif
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
#ifdef __aarch64__
                for (int i = 0; i < D/F16::block_size; ++i) {
                    F16::store(qkv + F16::block_size*i, F16::load(R + F16::block_size*i));
                }
#else
                std::memcpy(qkv, R, D*sizeof(float));
#endif
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
    qkv_cache_t qkv_cache[D*q_step]; // = {};
    //qkv_cache_t * qkv_cache;
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
        constexpr int nrc_k = 8;
#else
        constexpr int nrc_k = 8;
#endif
        static_assert(k_step%nrc_k == 0);
        DataInfo info{fms.cache, (const char *)q, k_step, stride_q*sizeof(q_float), 0, 1, nullptr};
        iqk_gemm_default_floats(D, q_step, kh.block, kh.stride, info, k_step);
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
        constexpr int nrc_k = 8;
        static_assert(k_step%nrc_k == 0);
        DataInfo info{fms.cache, (const char *)q, k_step, stride_q*sizeof(q_float), 0, 1, nullptr};
        iqk_gemm_default_floats(D, nq, kh.block, kh.stride, info, k_step);
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

    template <typename KHelper, typename block_q8>
    static inline void mul_mask_kq(const KHelper& kh, int stride_m,
            const block_q8 * q, const char * mask, FlashMS<q_step, k_step>& fms) {
        constexpr int kMaxQ = 8;
        static_assert(q_step < kMaxQ || q_step%kMaxQ == 0);
        DataInfo info{fms.cache, (const char *)q, k_step, (D/KHelper::block_size_q)*sizeof(block_q8), 0, 1, nullptr};
        if constexpr (std::is_same_v<KHelper, HelperQ8KVR8<D, k_step>> ||
                      std::is_same_v<KHelper, HelperQ8KV<D, k_step>>) {
            iqk_gemm_q8kv_fa(D, q_step, kh.type, kh.block, kh.stride, info, k_step);
        } else {
            iqk_gemm_legacy_fa(D, q_step, kh.type, kh.block, kh.stride, info, k_step);
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
        GGML_ASSERT(nq < q_step);
        DataInfo info{fms.cache, (const char *)q, k_step, (D/KHelper::block_size_q)*sizeof(block_q8), 0, 1, nullptr};
        if constexpr (std::is_same_v<KHelper, HelperQ8KVR8<D, k_step>> ||
                      std::is_same_v<KHelper, HelperQ8KV<D, k_step>>) {
            iqk_gemm_q8kv_fa(D, nq, kh.type, kh.block, kh.stride, info, k_step);
        } else {
            iqk_gemm_legacy_fa(D, nq, kh.type, kh.block, kh.stride, info, k_step);
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
        float * M, float * S, char * qptr) {
    auto q8 = (typename KHelper::block_q8 *)qptr;
    if constexpr (q_step > 1 && std::is_same_v<KHelper, HelperQ80<Dk, k_step>>) {
        if (nq1 == q_step) {
            fms.init_qstep();
            kh.reset_block();
            vh.reset_block();
            block_q8_0_r8 q8r8[Dk/QK8_0 * k_step/8];
            HelperQ80R8<Dk, k_step> khr8((const char *)q8r8, Dk/QK8_0*sizeof(block_q8_0));
            auto q8r = (typename HelperQ80R8<Dk, k_step>::block_q8 *)qptr;
            HelperQ80<Dk, QK8_0>::convert(q_step, stride_q, q, q8r);
            auto mr = mask;
            for (int k1 = 0; k1 < nk1/k_step; ++k1) {
                HelperQ80R8<Dk, k_step>::repack(k_step, kh.block, kh.stride, q8r8);
                KQHelper::mul_mask_kq(khr8, stride_m, q8r, mr, fms);
                fqkv.accumulate_qkv(vh, fms);
                kh.next_block();
                vh.next_block();
                mr += k_step*sizeof(ggml_half);
            }
            fqkv.normalize_and_store(fms, stride_qkv, qkv, M, S);
            return;
        }
    }
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

char * get_q_storage(size_t size) {
    thread_local std::vector<char> q_storage;
    if (q_storage.size() < size) q_storage.resize(size);
    return q_storage.data();
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
        if constexpr (std::is_same_v<KHelper, HelperQ40<Dk, k_step>> ||
                      std::is_same_v<KHelper, HelperQ41<Dk, k_step>> ||
                      std::is_same_v<KHelper, HelperIQ4nl<Dk, k_step>> ||
                      std::is_same_v<KHelper, HelperQ60<Dk, k_step>> ||
                      std::is_same_v<KHelper, HelperQ80R8<Dk, k_step>> ||
                      std::is_same_v<KHelper, HelperQ80<Dk, k_step>> ||
                      std::is_same_v<KHelper, HelperQ8KV<Dk, k_step>> ||
                      std::is_same_v<KHelper, HelperQ8KVR8<Dk, k_step>>) {
            constexpr size_t kMaxOnStackSize = 576;
            //auto q_size = q_step*(Dk/KHelper::block_size_q)*sizeof(typename KHelper::block_q8);
            auto q_size = q_step*(Dk/QK8_2*sizeof(block_q8_2));
            q_size = GGML_PAD(q_size, 64);
            if (q_size > kMaxOnStackSize) {
                auto qptr = get_q_storage(q_size);
                if (false && nq1 >= 8) {
                    if constexpr (std::is_same_v<KHelper, HelperQ80<Dk, k_step>>) {
#if FA_TIMING
                        auto t1 = Perf::cur_time();
                        HelperQ80R8<Dk, k_step> khr4(nk1, kh);
                        Perf::instance().accum(4, t1);
#else
                        HelperQ80R8<Dk, k_step> khr4(nk1, kh);
#endif
                        compute_helper_q<Dk, Dv, q_step, k_step, HelperQ80R8<Dk, k_step>, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                                khr4, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, M, S, qptr);
                        return;

                    }
                    if constexpr (std::is_same_v<KHelper, HelperQ8KV<Dk, k_step>>) {
#if FA_TIMING
                        auto t1 = Perf::cur_time();
                        HelperQ8KVR8<Dk, k_step> khr4(nk1, kh);
                        Perf::instance().accum(4, t1);
#else
                        HelperQ8KVR8<Dk, k_step> khr4(nk1, kh);
#endif
                        compute_helper_q<Dk, Dv, q_step, k_step, HelperQ8KVR8<Dk, k_step>, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                                khr4, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, M, S, qptr);
                        return;
                    }
                }
                compute_helper_q<Dk, Dv, q_step, k_step, KHelper, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                        kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, M, S, qptr);

            }
            else {
                typename KHelper::block_q8 q8[q_step*(Dk/KHelper::block_size_q)];
                compute_helper_q<Dk, Dv, q_step, k_step, KHelper, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                        kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, M, S, (char *)q8);
            }
        }
        else {
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

    auto update = [&nq1, &mask, &q, &qkv, &M, &S, stride_q, stride_m, stride_qkv] (int n) {
        nq1 -= n;
        if (nq1 == 0) return true;
        q    += n*stride_q;
        mask += n*stride_m;
        qkv  += n*stride_qkv;
        if (M && S) { M += n; S += n; }
        return false;
    };
    if (nk1 >= 512) {
        if (nq1 >= 128) {
            int n_step = nq1/128;
            FlashAttn<Dk, Dv, 64, k_step> fa(scale, softcap);
            fa.compute(kh, vh, 128*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            if (update(128*n_step)) return;
        }
        if (nq1 >= 64) {
            int n_step = nq1/64;
            FlashAttn<Dk, Dv, 64, k_step> fa(scale, softcap);
            fa.compute(kh, vh, 64*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            if (update(64*n_step)) return;
        }
        if (nq1 >= 32) {
            int n_step = nq1/32;
            FlashAttn<Dk, Dv, 32, k_step> fa(scale, softcap);
            fa.compute(kh, vh, 32*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            if (update(32*n_step)) return;
        }
        if (nq1 >= 16) {
            int n_step = nq1/16;
            FlashAttn<Dk, Dv, 16, k_step> fa(scale, softcap);
            fa.compute(kh, vh, 16*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            if (update(16*n_step)) return;
        }
    }
    if (nq1 >= 8) {
        int n_step = nq1/8;
        FlashAttn<Dk, Dv, 8, k_step> fa(scale, softcap);
        fa.compute(kh, vh, 8*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
        if (update(8*n_step)) return;
    }
    else if (nq1 >= 4) {
        int n_step = nq1/4;
        FlashAttn<Dk, Dv, 4, k_step> fa(scale, softcap);
        fa.compute(kh, vh, 4*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
        if (update(4*n_step)) return;
    }
    else if (nq1 >= 2) {
        int n_step = nq1/2;
        FlashAttn<Dk, Dv, 2, k_step> fa(scale, softcap);
        fa.compute(kh, vh, 2*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
        if (update(2*n_step)) return;
    }
    FlashAttn<Dk, Dv, 1, k_step> fa(scale, softcap);
    fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
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
        case GGML_TYPE_Q4_0: {
            HelperQ40<Dv, k_step> vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, M, S);
        } break;
#if GGML_IQK_FA_ALL_QUANTS
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
        case GGML_TYPE_Q8_0_R8: {
            HelperQ80R8<Dk, k_step> kh(k, stride_k);
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
        case GGML_TYPE_Q4_0: {
            HelperQ40<Dk, k_step> kh(k, stride_k);
            iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, M, S);
        } break;
#if GGML_IQK_FA_ALL_QUANTS
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
        type == GGML_TYPE_Q6_0 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q8_0_R8) return true;
#else
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q6_0 || type == GGML_TYPE_Q8_KV || type == GGML_TYPE_Q8_0_R8
            || type == GGML_TYPE_Q4_0) return true;
#endif
    return false;
}

template <int step_k, typename KHelper, typename VHelper>
inline void iqk_deepseek_helper(KHelper& kh, VHelper& vh,
                        int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
                        const float * q, const char * mask, float scale, float softcap, float * qkv, float * M, float * S) {
    auto update = [&nq1, &mask, &q, &qkv, &M, &S, stride_q, stride_m, stride_qkv] (int n) {
        nq1 -= n;
        if (nq1 == 0) return true;
        q    += n*stride_q;
        mask += n*stride_m;
        qkv  += n*stride_qkv;
        if (M && S) { M += n; S += n; }
        return false;
    };
    if (nq1 >= 16) {
        int n_step = nq1/16;
        FlashAttn<576, 512, 16, step_k> fa(scale, softcap);
        fa.compute(kh, vh, 16*n_step, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
        if (update(16*n_step)) return;
    }
    if (nq1 >= 8) {
        int n_step = nq1/8;
        FlashAttn<576, 512, 8, step_k> fa(scale, softcap);
        fa.compute(kh, vh, 8*n_step, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
        if (update(8*n_step)) return;
    }
    if (nq1 >= 4) {
        int n_step = nq1/4;
        FlashAttn<576, 512, 4, step_k> fa(scale, softcap);
        fa.compute(kh, vh, 4*n_step, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
        if (update(4*n_step)) return;
    }
    if (nq1 >= 2) {
        int n_step = nq1/2;
        FlashAttn<576, 512, 2, step_k> fa(scale, softcap);
        fa.compute(kh, vh, 2*n_step, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
        if (update(2*n_step)) return;
    }
    FlashAttn<576, 512, 1, step_k> fa(scale, softcap);
    fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
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
    if (type_k == GGML_TYPE_Q8_0_R8) {
        HelperQ80R8<576, step_k> kh((const char *)k, stride_k);
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
        GGML_ASSERT(type_k == type_v || (type_k == GGML_TYPE_Q8_0_R8 && type_v == GGML_TYPE_Q8_0));
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

    if (nk1%128 == 0) {
        switch (Dk) {
            case 64:
                iqk_flash_helper_T< 64, 64, 128>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            case 96:
                iqk_flash_helper_T< 96, 96, 128>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            case 128:
                iqk_flash_helper_T<128, 128, 128>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            case 192:
                iqk_flash_helper_T<192, 128, 128>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            case 256:
                iqk_flash_helper_T<256, 256, 128>(type_k, type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv, q, ck, cv, cm, scale, softcap, qkv, M, S); break;
            default:
                return false;
        }
        return true;
    }
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
#endif

#else  // IQK_IMPLEMENT

extern "C" IQK_API bool iqk_mul_mat(int, long, long, long, int, const void *, long, int, const void *, long, float *, long, int, int) {
    return false;
}

extern "C" IQK_API bool iqk_mul_mat_4d(long /*Nx*/, long /*Ny*/, long /*ne00*/,
        long /*ne02*/, long /*ne03*/, long /*ne12*/, long /*ne13*/,
        long /*nb02*/, long /*nb03*/, long /*nb12*/, long /*nb13*/, long /*nb2*/, long /*nb3*/,
        int /*typeA*/, const void * /*A*/, long /*strideA*/,
        int /*typeB*/, const void * /*B*/, long /*strideB*/,
        float * /*C*/, long /*stride_C*/, int /*ith*/, int /*nth*/) {
    return false;
}

extern "C" IQK_API bool iqk_mul_mat_moe(long, long, long, int, int, const void *, long, int, const void *, long, float *, long, long,
        const void *, int, int) {
    return false;
}

extern "C" IQK_API bool iqk_moe_fused_up_gate(long /*Nx*/, long /*Ny*/, long /*ne00*/, int /*ne11*/, int /*unary_op*/,
        int /*typeA*/, const void * /*Aup*/, const void * /*Agate*/, long /*strideA*/,
        int /*typeB*/, const void * /*B*/, long /*strideB*/,
        float * /*C*/, long /*nb1*/, long /*nb2*/, const void * /*vrow_mapping*/, int /*ith*/, int /*nth*/) {
    return false;
}

#endif
