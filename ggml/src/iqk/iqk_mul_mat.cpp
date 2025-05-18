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
private:
    template <typename Dequantizer> static void set_functions(MulMat& m);
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
    inline void prepare64a(const uint8_t * q4) {
        for (int k = 0; k < 4; ++k) {
            auto q4bits = _mm256_loadu_si256((const __m256i*)q4 + k);
            values[k] = _mm512_inserti32x8(_mm512_castsi256_si512(q4bits), _mm256_srli_epi16(q4bits, 4), 1);
            values[k] = _mm512_and_si512(values[k], ml);
        }
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

template <typename Q8>
inline void process_mins_and_scales_16(const __m128i& scales128, const Q8& q8, int i, float d,
    __m256 * accm, __m256i * scales) {
    const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
    process_mins_16(all_scales, q8, i, d, accm);
    prepare_scales_16(all_scales, scales);
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

template <typename Dequantizer> void MulMat::set_functions(MulMat& m) {
#ifdef HAVE_FANCY_SIMD
    m.funcs[0] = mul_mat_qX_K_q8_K_AVX512_1<Dequantizer>;
    m.funcs[1] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 2>;
    m.funcs[2] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 3>;
    m.funcs[3] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 4>;
    m.funcs[4] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 5>;
    m.funcs[5] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 6>;
    m.funcs[6] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 7>;
    m.funcs[7] = mul_mat_qX_K_q8_K_AVX512<Dequantizer, 8>;
#else
    m.funcs[0] = mul_mat_qX_K_q8_K_T<Dequantizer, 1>;
    m.funcs[1] = mul_mat_qX_K_q8_K_T<Dequantizer, 2>;
    m.funcs[2] = mul_mat_qX_K_q8_K_T<Dequantizer, 3>;
    m.funcs[3] = mul_mat_qX_K_q8_K_T<Dequantizer, 4>;
    m.funcs[4] = mul_mat_qX_K_q8_K_T<Dequantizer, 5>;
    m.funcs[5] = mul_mat_qX_K_q8_K_T<Dequantizer, 6>;
    m.funcs[6] = mul_mat_qX_K_q8_K_T<Dequantizer, 7>;
    m.funcs[7] = mul_mat_qX_K_q8_K_T<Dequantizer, 8>;
#endif
}

bool MulMat::prepare(int typeA, int typeB, int ne00, MulMat& mm, int Ny) {

    (void)Ny;

    if (typeA == GGML_TYPE_F16 || typeA == GGML_TYPE_F32 || typeA == GGML_TYPE_BF16 || typeA == GGML_TYPE_BF16_R16) {
        return iqk_set_kernels_float(ne00, typeA, typeB, mm.funcs);
    }

    auto expected_typeB = GGML_TYPE_Q8_K;

    switch (typeA) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
            return ggml_type(typeB) == GGML_TYPE_Q8_K ? iqk_set_kernels_kquants(ne00, typeA, typeB, mm.funcs) : false;
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            return ggml_type(typeB) == GGML_TYPE_Q8_K ? iqk_set_kernels_iquants(ne00, typeA, typeB, mm.funcs) : false;
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ6_K:
            return ggml_type(typeB) == GGML_TYPE_Q8_K ? iqk_set_kernels_iqk_quants(ne00, typeA, typeB, mm.funcs) : false;
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
        case GGML_TYPE_IQ5_KS_R4:
            assert (ne00 % QK_K == 0);
            mm.funcs[0] = mul_mat_iq5_ks_r4_q8_k<1>;
            mm.funcs[1] = mul_mat_iq5_ks_r4_q8_k<2>;
            mm.funcs[2] = mul_mat_iq5_ks_r4_q8_k<3>;
            mm.funcs[3] = mul_mat_iq5_ks_r4_q8_k<4>;
            mm.funcs[4] = mul_mat_iq5_ks_r4_q8_k<5>;
            mm.funcs[5] = mul_mat_iq5_ks_r4_q8_k<6>;
            mm.funcs[6] = mul_mat_iq5_ks_r4_q8_k<7>;
            mm.funcs[7] = mul_mat_iq5_ks_r4_q8_k<8>;
#ifndef HAVE_FANCY_SIMD
            // For some reason Zen4 does not like this particular function
            mm.func16 = mul_mat_iq5_ks_r4_q8_k<16>;
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
void mul_mat_iq5_ks_r4_q8_k_neon(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
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
    float32x4_t acc[nrc_y] = {};
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
            acc[2*iy+0] = acc[2*iy+1] = vdupq_n_s32(0.f);
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
            bits.val[0] = veorq_u8(m88, bits.val[0]);
            bits.val[1] = veorq_u8(m88, bits.val[1]);
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
        case GGML_TYPE_IQ5_KS:
            MulMat::set_functions<DequantizerIQ5KS>(m);
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
        case GGML_TYPE_IQ5_KS_R4:
            SET_MUL_MAT_FUNCTIONS(m, mul_mat_iq5_ks_r4_q8_k_neon);
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
            if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q8_0_1_Unpacker, 1, k_step>, 1);
            if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q8_0_1_Unpacker, 2, k_step>, 2);
            if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q8_0_1_Unpacker, 4, k_step>, 4);
            MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q8_0_1_Unpacker, nq);
#else
            if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_0_Tx<Q8_0_Unpacker, 1, k_step>, 1);
            if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_0_Tx<Q8_0_Unpacker, 2, k_step>, 2);
            if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_0_Tx<Q8_0_Unpacker, 4, k_step>, 4);
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
            if (nq == 1) return std::make_pair(mul_mat_q8_KV_q8_KV_1<1>, 1);
#ifdef HAVE_FANCY_SIMD
            if constexpr (D%32 == 0 && k_step%8 == 0) {
                if (nq%16 == 0) return std::make_pair(mul_mat_q8_KV_q8_KV_8<16>, 16);
                MAKE_FUNCS_ONLY_NRC(mul_mat_q8_KV_q8_KV_8, nq);
            } else {
                if (nq%16 == 0) return std::make_pair(mul_mat_q8_KV_q8_KV<16>, 16);
            }
#endif
            MAKE_FUNCS_ONLY_NRC(mul_mat_q8_KV_q8_KV, nq);
#endif
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ80R8<D, k_step>>) {
#ifdef __aarch64__
            MAKE_FUNCS_ONLY_NRC(mul_mat_q8_0_r8_q8_0, nq);
#else
            MAKE_FUNCS_ONLY_NRC(mul_mat_q8_0_r8_q8_2, nq);
#endif
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ8KVR8<D, k_step>>) {
            MAKE_FUNCS_ONLY_NRC(mul_mat_q8_KV_r8_q8_KV, nq);
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ60<D, k_step>>) {
#ifdef __aarch64__
            MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerQ60, nq);
#else
            if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q6_0_1_Unpacker, 1, k_step>, 1);
            if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q6_0_1_Unpacker, 2, k_step>, 2);
            if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q6_0_1_Unpacker, 4, k_step>, 4);
            MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q6_0_1_Unpacker, nq);
#endif
        }
        else if constexpr (std::is_same_v<KHelper, HelperQ40<D, k_step>>) {
#ifdef __aarch64__
            MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerQ40, nq);
#else
            if (nq == 1) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q4_0_1_Unpacker, 1, k_step>, 1);
            if (nq == 2) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q4_0_1_Unpacker, 2, k_step>, 2);
            if (nq == 4) return std::make_pair(mul_mat_qX_0_q8_2_Tx<Q4_0_1_Unpacker, 4, k_step>, 4);
            MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q4_0_1_Unpacker, nq);
#endif
        }
#if GGML_IQK_FA_ALL_QUANTS
        else if constexpr (std::is_same_v<KHelper, HelperQ41<D, k_step>>) {
#ifdef __aarch64__
            MAKE_FUNCS(mul_mat_qX_1_q8_1<DequantizerQ41, nq);
#else
            MAKE_FUNCS(mul_mat_qX_1_q8_2_T<Q4_1_Unpacker, nq);
#endif
        }
        else if constexpr (std::is_same_v<KHelper, HelperIQ4nl<D, k_step>>) {
#ifdef __aarch64__
            MAKE_FUNCS(mul_mat_qX_0_q8_0<DequantizerIQ4NL, nq);
#else
#ifdef HAVE_FANCY_SIMD
            MAKE_FUNCS(mul_mat_qX_1_q8_2_T<IQ4_NL_Unpacker, nq);
#else
            MAKE_FUNCS(mul_mat_qX_0_q8_0_T<IQ4_NL_Unpacker, nq);
#endif
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
