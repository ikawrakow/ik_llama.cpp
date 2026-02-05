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
#include <algorithm>

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "iqk_mul_mat.h"
#include "iqk_quantize.h"
#include "iqk_flash_impl.h"
#include "iqk_gemm_floats.h"
#include "iqk_gemm_kquants.h"
#include "iqk_gemm_ktquants.h"
#include "iqk_gemm_iquants.h"
#include "iqk_gemm_iqk_quants.h"
#include "iqk_gemm_1bit.h"
#include "iqk_gemm_legacy_quants.h"
#include "iqk_utils.h"

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
    inline static void gelu(int n, const float * src, float * dst);
    inline static void relu(int n, const float * src, float * dst);
    inline static void silu(int n, const float * src, float * dst);
    inline static void swiglu_oai(int n, const float * src, float * dst);
    inline static void clamp_oai(int n, float *x);
    inline static void activate(ggml_unary_op op, int n, const float * src, float * dst) {
        if      (op == GGML_UNARY_OP_GELU) gelu(n, src, dst);
        else if (op == GGML_UNARY_OP_RELU) relu(n, src, dst);
        else if (op == GGML_UNARY_OP_SILU) silu(n, src, dst);
        else if (op == GGML_UNARY_OP_SWIGLU_OAI) swiglu_oai(n, src, dst);
        else GGML_ABORT("fatal error");
    }
    inline void mul_mat_up_gate_NxM(int n, const void * vx_up, const void * vx_gate, size_t bx,
            const float * up_b, const float * gate_b,
            DataInfo& info, int nrc_x, int nrc_y, int unary_op, float limit) {
#ifdef __aarch64__
        constexpr int k_x_step = 64; //8192; // Tiling does not seem to help on my M2 Max (but difference to tiling is small)
#else
        constexpr int k_x_step = 64; // This works best on my Ryzen-7950X (but differences to other tile size are small)
#endif
        auto op = ggml_unary_op(unary_op);
        float tmp[k_x_step*16];
        auto process = [&tmp, n, op, vx_gate, vx_up, gate_b, up_b, bx, xstep = k_x_step, limit] (mul_mat_t func, const DataInfo& this_info, int ix, int this_nrc_x, int ny) {
            func(n, (const void *)((const char *)vx_gate + ix*bx), bx, this_info, this_nrc_x);
            for (int ky = 0; ky < ny; ++ky) {
                if (gate_b) {
                    auto b = gate_b + ix;
                    auto x = this_info.dst_row(ky);
                    for (int j = 0; j < this_nrc_x; ++j) x[j] += b[j];
                }
                activate(op, this_nrc_x, this_info.dst_row(ky), tmp + ky*xstep);
                if (limit > 1e-6f) {
                    for (int j = 0; j < this_nrc_x; ++j) tmp[ky*xstep + j] = std::min(tmp[ky*xstep + j], limit);
                }
            }
            func(n, (const void *)((const char *)vx_up + ix*bx), bx, this_info, this_nrc_x);
            for (int ky = 0; ky < ny; ++ky) {
                auto result = this_info.dst_row(ky);
                if (up_b) {
                    auto b = up_b + ix;
                    for (int j = 0; j < this_nrc_x; ++j) result[j] += b[j];
                }
                if (op == GGML_UNARY_OP_SWIGLU_OAI) {
                    clamp_oai(this_nrc_x, result);
                } else if (limit > 1e-6f) {
                    for (int j = 0; j < this_nrc_x; ++j) result[j] = std::max(-limit, std::min(limit, result[j]));
                }
                for (int j = 0; j < this_nrc_x; ++j) result[j] *= tmp[ky*xstep + j];
            }
        };
        if (func16 && nrc_y >= 16) {
            int n_step = (nrc_y - info.cur_y)/16;
            for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                auto this_info = info;
                this_info.s += ix;
                int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                for (int iy = 0; iy < n_step; ++iy) {
                    process(func16, this_info, ix, this_nrc_x, 16);
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
                        process(funcs[ny1-1], this_info, ix, this_nrc_x, ny1);
                        this_info.cur_y += ny1;
                    }
                    for (int iy = 0; iy < my2; ++iy) {
                        process(funcs[ny2-1], this_info, ix, this_nrc_x, ny2);
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
                        process(funcs[ny-1], this_info, ix, this_nrc_x, ny);
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
                process(funcs[n_left-1], this_info, ix, this_nrc_x, n_left);
            }
        }
    }
    static bool prepare(int typeA, int typeB, int ne00, MulMat& mm, int Ny);
    static inline ggml_type is_dequant_better(ggml_type type, int nrc_y) {
#ifdef __AVX2__
#ifdef HAVE_FANCY_SIMD
        auto q8_k_type = GGML_TYPE_Q8_K_R16;
#else
        auto q8_k_type = GGML_TYPE_Q8_K_R8;
#endif
        switch (type) {
            case GGML_TYPE_IQ2_XXS: return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ2_XS : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ2_S  : return nrc_y >= 16 ? q8_k_type : type;
            case GGML_TYPE_IQ3_XXS: return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ4_XS : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ3_S  : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ1_S  : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ1_M  : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_Q2_K   : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_Q3_K   : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_Q4_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
            case GGML_TYPE_Q5_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
            case GGML_TYPE_Q6_K   : return nrc_y >= 64 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ2_KS : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ2_K  : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ2_KL : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ3_KS : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ3_K  : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ4_KS : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ4_KSS: return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ4_K  : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ5_KS : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ5_K  : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_IQ6_K  : return nrc_y >= 32 ? q8_k_type : type;
            case GGML_TYPE_Q4_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_Q4_1   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
            case GGML_TYPE_Q5_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_Q5_1   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
            case GGML_TYPE_Q6_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ4_NL : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_MXFP4  : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_Q8_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ1_KT : return nrc_y >= 16 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ2_KT : return nrc_y >= 16 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ3_KT : return nrc_y >= 16 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ4_KT : return nrc_y >= 24 ? GGML_TYPE_Q8_0_R8 : type;
            default: break;
        }
#else
        switch (type) {
            case GGML_TYPE_Q2_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_Q3_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_Q4_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
            case GGML_TYPE_Q5_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
            case GGML_TYPE_Q6_K   : return nrc_y >= 64 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ1_S  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ1_M  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_XXS: return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_XS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_S  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_XXS: return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_S  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_XS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_Q4_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_Q4_1   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
            case GGML_TYPE_Q5_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_Q5_1   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
            case GGML_TYPE_Q6_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_Q8_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ4_NL : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_MXFP4  : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ1_KT : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ2_KT : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ3_KT : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ4_KT : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ2_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_KL : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_KSS: return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ5_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ5_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ6_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            default: break;
        }
#endif
        return type;
    }
    static inline int num_rows([[maybe_unused]] ggml_type type) {
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
            case GGML_TYPE_Q8_1:
            case GGML_TYPE_Q8_K_R8: return 8;
            case GGML_TYPE_Q4_0_R8:
            case GGML_TYPE_Q8_0_R8:
            case GGML_TYPE_Q8_K_R16:
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
            case GGML_TYPE_Q8_1:
            case GGML_TYPE_Q8_K_R8: return 8;
            case GGML_TYPE_Q8_K_R16:
            case GGML_TYPE_BF16_R16: return 16;
            default: return 1;
        }
#endif
    }
};

static std::vector<char> & thread_local_work_buffer() {
    thread_local std::vector<char> f;
    return f;
}

bool iqk_convert_repack(int typeA, int n, const void * vx, size_t bx, void * vy, size_t stride_y, int nrc_x) {

    switch (typeA) {
        //case GGML_TYPE_F16:
        //case GGML_TYPE_F32:
        //case GGML_TYPE_BF16:
        //case GGML_TYPE_BF16_R16:
        //    return iqk_set_kernels_float(ne00, typeA, typeB, mm.funcs);
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
        //case GGML_TYPE_Q2_K_R4:
        //case GGML_TYPE_Q3_K_R4:
        //case GGML_TYPE_Q4_K_R4:
        //case GGML_TYPE_Q5_K_R4:
        //case GGML_TYPE_Q6_K_R4:
        //case GGML_TYPE_IQ4_XS_R8:
        //case GGML_TYPE_Q8_K_R8:
        //case GGML_TYPE_Q8_KV:
        //case GGML_TYPE_Q8_KV_R8:
            return iqk_convert_kquants_q8X_r8(typeA, n, vx, bx, vy, nrc_x);
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
            return iqk_convert_iquants_q80_r8(typeA, n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ6_K:
        //case GGML_TYPE_IQ2_K_R4:
        //case GGML_TYPE_IQ3_K_R4:
        //case GGML_TYPE_IQ4_K_R4:
        //case GGML_TYPE_IQ5_K_R4:
        //case GGML_TYPE_IQ4_KS_R4:
        //case GGML_TYPE_IQ5_KS_R4:
            return iqk_convert_iqk_quants_q80_r8(typeA, n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
            return iqk_dequantize_ktquants(typeA, n, vx, bx, vy, stride_y, nrc_x);
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        //case GGML_TYPE_Q4_0_R8:
        //case GGML_TYPE_Q5_0_R4:
        //case GGML_TYPE_Q6_0_R4:
        //case GGML_TYPE_Q8_0_R8:
        //case GGML_TYPE_IQ4_NL_R4:
            return iqk_convert_legacy_quants_q8_r8(typeA, n, vx, bx, vy, nrc_x);
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        //case GGML_TYPE_IQ1_S_R4:
        //case GGML_TYPE_IQ1_M_R4:
        //case GGML_TYPE_IQ1_BN:
        //case GGML_TYPE_IQ2_BN:
        //case GGML_TYPE_IQ2_BN_R4:
            return iqk_convert_1bit_q80_r8(typeA, n, vx, bx, vy, nrc_x);

        default:
            break;
    }

    return false;
}

}

extern "C" IQK_API int iqk_dequant_type(int type, int Ny) {
    return MulMat::is_dequant_better(ggml_type(type), Ny);
}

extern "C" IQK_API bool iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth) {

    MulMat mm;

    auto etypeA = ggml_type(typeA);
    if (auto dequant_type = MulMat::is_dequant_better(etypeA, Ny);
             dequant_type != etypeA && MulMat::prepare(dequant_type, typeB, ne00, mm, Ny) &&
             Nx%MulMat::num_rows(ggml_type(dequant_type)) == 0) {

        constexpr int k_x_step = 32;

        auto num_rows = MulMat::num_rows(ggml_type(dequant_type));
        GGML_ASSERT(Nx%num_rows == 0);
        auto nrc_x = (Nx/num_rows + nth - 1)/nth;
        auto first_x = ith*nrc_x;
        if (first_x + nrc_x > Nx/num_rows) nrc_x = Nx/num_rows - first_x;
        first_x *= num_rows;
        nrc_x   *= num_rows;

        size_t row_size_qx = ggml_row_size(dequant_type, ne00);
        size_t row_size_qy = strideB;

        //printf("Dequant mul mat %s x %s: ne00 = %d, row_size = %d\n", ggml_type_name(dequant_type), ggml_type_name(ggml_type(typeB)), (int)ne00, (int)row_size_qx);

        DataInfo info{C + first_x, (const char *)B, (size_t)stride_C, row_size_qy, 0, 1, nullptr, 0};

        auto& f = thread_local_work_buffer();

        for (int ix = 0; ix < nrc_x; ix += k_x_step) {
            auto this_info = info;
            this_info.s += ix;
            int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
            if (f.size() < row_size_qx*this_nrc_x) f.resize(row_size_qx*this_nrc_x);
            if (!iqk_convert_repack(typeA, ne00, (const char *)A + (first_x + ix)*strideA, strideA, f.data(), ne00, this_nrc_x)) {
                GGML_ABORT("Fatal error");
            }
            mm.mul_mat_NxM(ne00, f.data(), row_size_qx, this_info, this_nrc_x, Ny);
        }

        return true;

    }

    if (!MulMat::prepare(typeA, typeB, ne00, mm, Ny)) {
        return false;
    }

    size_t row_size_qx = strideA; //*ggml_type_size(ggml_type(typeA));
    size_t row_size_qy = strideB; //*ggml_type_size(ggml_type(typeB));
    //if (ith == 0) printf("%s: ne00 = %d, row_size_qx = %d, strideA = %d\n", __func__, int(ne00), int(row_size_qx), int(strideA));

    auto num_rows = MulMat::num_rows(ggml_type(typeA));
    if (Nx%num_rows) {
        fprintf(stderr, "%s: Nx = %d, Ny = %d, ne00 = %d, num_rows = %d, types = %s, %s\n", __func__, (int)Nx, (int)Ny,
                (int)ne00, num_rows, ggml_type_name(ggml_type(typeA)), ggml_type_name(ggml_type(typeB)));
        GGML_ASSERT(false);
    }
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

    auto etypeA = ggml_type(typeA);
    //auto etypeB = ggml_type(typeB);
    auto dequant_type = MulMat::is_dequant_better(etypeA, Ny);
    //if (etypeB != GGML_TYPE_F32) {
    //    if (ith == 0) printf("%s: typeA = %s, typeB = %s, dequant_type = %s\n", __func__, ggml_type_name(etypeA), ggml_type_name(etypeB), ggml_type_name(dequant_type));
    //}
    if (dequant_type != etypeA) {
        if (!MulMat::prepare(dequant_type, typeB, ne00, mm, Ny)) {
            return false;
        }

        constexpr int k_x_step = 32;

        auto num_rows = MulMat::num_rows(ggml_type(dequant_type));
        GGML_ASSERT(Nx%num_rows == 0);
        auto nrc_x = (Nx/num_rows + nth - 1)/nth;
        auto first_x = ith*nrc_x;
        if (first_x + nrc_x > Nx/num_rows) nrc_x = Nx/num_rows - first_x;
        first_x *= num_rows;
        nrc_x   *= num_rows;

        size_t row_size_qx = ggml_row_size(dequant_type, ne00);
        size_t row_size_qy = strideB;

        DataInfo info{C + first_x, (const char *)B, nb1/sizeof(float), row_size_qy, 0, ne11, row_mapping, nb2/sizeof(float)};

        auto& f = thread_local_work_buffer();

        for (int ix = 0; ix < nrc_x; ix += k_x_step) {
            auto this_info = info;
            this_info.s += ix;
            int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
            if (f.size() < row_size_qx*this_nrc_x) f.resize(row_size_qx*this_nrc_x);
            if (!iqk_convert_repack(typeA, ne00, (const char *)A + (first_x + ix)*strideA, strideA, f.data(), ne00, this_nrc_x)) {
                GGML_ABORT("Fatal error");
            }
            mm.mul_mat_NxM(ne00, f.data(), row_size_qx, this_info, this_nrc_x, Ny);
        }

        return true;

    }

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
        const char * up_b_c, const char * gate_b_c,
        float * C, long nb1, long nb2, const void * vrow_mapping, float limit, int ith, int nth) {

    const mmid_row_mapping * row_mapping = (const mmid_row_mapping *)vrow_mapping;
    //assert(row_mapping != nullptr);

    MulMat mm;

    auto etypeA = ggml_type(typeA);
    if (auto dequant_type = MulMat::is_dequant_better(etypeA, Ny); dequant_type != etypeA) {
        if (MulMat::prepare(dequant_type, typeB, ne00, mm, Ny)) {

            constexpr int k_x_step = 64;

            auto num_rows = MulMat::num_rows(ggml_type(dequant_type));
            GGML_ASSERT(Nx%num_rows == 0);
            auto nrc_x = (Nx/num_rows + nth - 1)/nth;
            auto first_x = ith*nrc_x;
            if (first_x + nrc_x > Nx/num_rows) nrc_x = Nx/num_rows - first_x;
            first_x *= num_rows;
            nrc_x   *= num_rows;

            size_t row_size_qx = ggml_row_size(dequant_type, ne00);
            size_t row_size_qy = strideB;

            DataInfo info{C + first_x, (const char *)B, nb1/sizeof(float), row_size_qy, 0, ne11, row_mapping, nb2/sizeof(float)};

            auto& f = thread_local_work_buffer();

            for (int ix = 0; ix < nrc_x; ix += k_x_step) {
                auto this_info = info;
                this_info.s += ix;
                int this_nrc_x = ix + k_x_step <= nrc_x ? k_x_step : nrc_x - ix;
                if (f.size() < 2*row_size_qx*this_nrc_x) f.resize(2*row_size_qx*this_nrc_x);
                auto Xu = f.data();
                auto Xg = f.data() + row_size_qx*this_nrc_x;
                if (!iqk_convert_repack(typeA, ne00, (const char *)Aup   + (first_x + ix)*strideA, strideA, Xu, ne00, this_nrc_x)) {
                    GGML_ABORT("Fatal error");
                }
                if (!iqk_convert_repack(typeA, ne00, (const char *)Agate + (first_x + ix)*strideA, strideA, Xg, ne00, this_nrc_x)) {
                    GGML_ABORT("Fatal error");
                }
                auto up_b   = up_b_c   ? (const float *)up_b_c + first_x + ix : nullptr;
                auto gate_b = gate_b_c ? (const float *)gate_b_c + first_x + ix : nullptr;
                mm.mul_mat_up_gate_NxM(ne00, Xu, Xg, row_size_qx, up_b, gate_b, this_info, this_nrc_x, Ny, unary_op, limit);
            }

            return true;
        }

    }

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
    auto up_b   = up_b_c   ? (const float *)up_b_c + first_x : nullptr;
    auto gate_b = gate_b_c ? (const float *)gate_b_c + first_x : nullptr;
    mm.mul_mat_up_gate_NxM(ne00, (const char *)Aup + row_size_qx*first_x, (const char *)Agate + row_size_qx*first_x, row_size_qx,
            up_b, gate_b, info, nrc_x, Ny, unary_op, limit);
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
        case GGML_TYPE_Q8_K_R16:
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
            return iqk_set_kernels_iquants(ne00, typeA, typeB, mm.funcs, mm.func16);
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:
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
            return iqk_set_kernels_iqk_quants(ne00, typeA, typeB, mm.funcs, mm.func16);
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
            return iqk_set_kernels_ktquants(ne00, typeA, typeB, mm.funcs, mm.func16);
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_0_R8:
        case GGML_TYPE_Q5_0_R4:
        case GGML_TYPE_Q6_0_R4:
        case GGML_TYPE_Q8_0_R8:
        case GGML_TYPE_IQ4_NL_R4:
        case GGML_TYPE_MXFP4:
            return iqk_set_kernels_legacy_quants(ne00, typeA, typeB, mm.funcs, mm.func16);
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
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
        case GGML_TYPE_Q8_K_R16:
            return iqk_set_kernels_kquants(ne00, typeA, typeB, m.funcs, m.func16);
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:
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
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_IQ4_NL_R4:
        case GGML_TYPE_MXFP4:
            return iqk_set_kernels_legacy_quants(ne00, typeA, typeB, m.funcs, m.func16);
        case GGML_TYPE_IQ1_BN:
        case GGML_TYPE_IQ2_BN:
        case GGML_TYPE_IQ2_BN_R4:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:
            return iqk_set_kernels_1bit(ne00, typeA, typeB, m.funcs, m.func16);
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
            return iqk_set_kernels_ktquants(ne00, typeA, typeB, m.funcs, m.func16);
        default:
            return false;
    }

}

}

#endif // __aarch64__

namespace {

// TODO: these swiglu_oai constants shouldn't be hard coded
constexpr float k_swiglu_oai_alpha = 1.702f;
constexpr float k_swiglu_oai_limit = 7.f;

void MulMat::swiglu_oai(int n, const float * x, float * y) {
//    int i = 0;
//#if defined __AVX512F__ && defined __AVX512DQ__
//    {
//        auto max = _mm512_set1_ps(k_swiglu_oai_limit);
//        auto alpha = _mm512_set1_ps(-k_swiglu_oai_alpha);
//        for (; i + 15 < n; i += 16) {
//            auto xc = v_clamp_max(_mm512_loadu_ps(x + i), max);
//            _mm512_storeu_ps(y + i, v_silu_oai(xc, alpha));
//        }
//    }
//#endif
//#if defined __AVX2__ && defined __FMA__
//    if (i + 7 < n) {
//        auto max = _mm256_set1_ps(k_swiglu_oai_limit);
//        auto alpha = _mm256_set1_ps(-k_swiglu_oai_alpha);
//        for (; i + 7 < n; i += 8) {
//            auto xc = v_clamp_max(_mm256_loadu_ps(x + i), max);
//            _mm256_storeu_ps(y + i, v_silu_oai(xc, alpha));
//        }
//    }
//#endif
//    for (; i < n; ++i) {
//        auto xi = std::min(x[i], k_swiglu_oai_limit);
//        y[i] = xi / (1.0f + expf(-xi * k_swiglu_oai_alpha));
//    }
    for (int i = 0; i < n; ++i) {
        auto xi = std::min(x[i], k_swiglu_oai_limit);
        y[i] = xi / (1.0f + expf(-xi * k_swiglu_oai_alpha));
    }
}

void MulMat::clamp_oai(int n, float * x) {
    for (int i = 0; i < n; ++i) x[i] = 1.f + std::max(std::min(x[i], k_swiglu_oai_limit), -k_swiglu_oai_limit);
}

#if defined(__ARM_NEON) && defined(__aarch64__)
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

#if defined(__AVX2__) && defined(__FMA__)

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

//void MulMat::swiglu_oai(int n, const float * x, float * y) {
//    int i = 0;
//#if defined __AVX512F__ && defined __AVX512DQ__
//    {
//        auto limit = _mm512_set1_ps(k_swiglu_oai_limit);
//        auto alpha = _mm512_set1_ps(k_swiglu_oai_alpha);
//        for (; i + 15 < n; i += 16) {
//            auto xi = _mm512_loadu_ps(x + i);
//            auto mask = _mm512_cmp
//
//        }
//        __m512 c1 = _mm512_set1_ps(GELU_COEF_A);
//        __m512 c2 = _mm512_set1_ps(2.f*SQRT_2_OVER_PI);
//        for (; i + 15 < n; i += 16) _mm512_storeu_ps(y + i, v_gelu(_mm512_loadu_ps(x + i), c1, c2));
//    }
//#endif
//#if defined __AVX2__ && defined __FMA__
//    if (i + 7 < n) {
//        __m256 c1 = _mm256_set1_ps(GELU_COEF_A);
//        __m256 c2 = _mm256_set1_ps(2.f*SQRT_2_OVER_PI);
//        for (; i + 7 < n; i += 8) _mm256_storeu_ps(y + i, v_gelu(_mm256_loadu_ps(x + i), c1, c2));
//
//    }
//#endif
//    for (; i < n; ++i) {
//        auto xi = std::min(x[i], k_swiglu_oai_limit);
//        y[i] = xi / (1.0f + expf(-xi * k_swiglu_oai_alpha));
//    }
//}


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
void iqk_topk_moe(int n_experts, int n_experts_used, const float * logits,
        float * weights, int32_t * ids, void * work) {

    if (work) {
        auto sorted = (std::pair<float, int> *)work;
        for (int j = 0; j < n_experts; ++j) sorted[j] = {logits[j], j};

        std::partial_sort(sorted, sorted + n_experts_used, sorted + n_experts, std::greater<std::pair<float,int>>{});

        float max = sorted[0].first;
        float sum = 0;
        for (int j = 0; j < n_experts; ++j) {
            float p = expf(sorted[j].first - max);
            weights[j] = p;
            ids[j] = sorted[j].second;
            sum += p;
        }
        float norm = 1/sum;
        for (int j = 0; j < n_experts; ++j) weights[j] *= norm;
    } else {
        for (int j = 0; j < n_experts; ++j) ids[j] = j;

        std::partial_sort(ids, ids + n_experts_used, ids + n_experts,
                [logits] (int i1, int i2) {
                    return logits[i1] > logits[i2];
                });

        float max = logits[ids[0]];
        float sum = 0;
        for (int j = 0; j < n_experts_used; ++j) {
            float p = expf(logits[ids[j]] - max);
            weights[j] = p;
            sum += p;
        }
        for (int j = n_experts_used; j < n_experts; ++j) {
            sum += expf(logits[ids[j]] - max);
        }
        float norm = 1/sum;
        for (int j = 0; j < n_experts_used; ++j) weights[j] *= norm;
    }
}
}

void iqk_topk_moe(int n_experts, int n_experts_used, int nrows, const float * logits,
        float * weights, int32_t * ids, int ith, int nth) {

    int npt = (nrows + nth - 1)/nth;
    int first = ith*npt;
    int last  = std::min(nrows, first + npt);
    for (int row = first; row < last; ++row) {
        auto row_logits  = logits  + row*n_experts;
        auto row_weights = weights + row*n_experts_used;
        auto row_ids     = ids     + row*n_experts;
        iqk_topk_moe(n_experts, n_experts_used, row_logits, row_weights, row_ids, nullptr);
    }
}

#ifdef GGML_IQK_FLASH_ATTENTION

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

#include "iqk_flash_impl.h"
#include "fa/iqk_fa_templates.h"

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
                         const float * sinksf,   // mask. If not null, assumed to be fp16. nq x nk elements
                         [[maybe_unused]] int nsinks,
                         float         scale,    // scale applied before softmax
                         float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                         float       * qkv,      // v*softmax(scale*(k*q))
                         float * M, float * S) {

    if (!mask || nk1%32 != 0) return false; // the implementation assumes mask is not null and nk is a multiple of 32

    if (Dk == 576 && Dv == 512) {
        return iqk_fa_576_512(int_type_k, int_type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, k, v, mask, scale, softcap, qkv, sinksf, M, S);
    }

    if (Dk == 192 && Dv == 128) {
        return iqk_fa_192_128(int_type_k, int_type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, k, v, mask, scale, softcap, qkv, sinksf, M, S);
    }

    if (Dk == 192 && Dv == 192) {
        return iqk_fa_192_192(int_type_k, int_type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, k, v, mask, scale, softcap, qkv, sinksf, M, S);
    }

    if (Dk == 256 && Dv == 256) {
        return iqk_fa_256_256(int_type_k, int_type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, k, v, mask, scale, softcap, qkv, sinksf, M, S);
    }

    if (Dk == 128 && Dv == 128) {
        return iqk_fa_128_128(int_type_k, int_type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, k, v, mask, scale, softcap, qkv, sinksf, M, S);
    }

    if (Dk == 96 && Dv == 96) {
        return iqk_fa_96_96(int_type_k, int_type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, k, v, mask, scale, softcap, qkv, sinksf, M, S);
    }

    if (Dk == 64 && Dv == 64) {
        return iqk_fa_64_64(int_type_k, int_type_v, nq1, nk1, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, k, v, mask, scale, softcap, qkv, sinksf, M, S);
    }

    return false;
}
#endif

#else  // IQK_IMPLEMENT

#include "ggml-impl.h"

extern "C" IQK_API bool iqk_mul_mat(int, long, long, long, int, const void *, long, int, const void *, long, float *, long, int, int) {
    GGML_ABORT("Unsupported CPU. You may need to manually set compilation flags\n");
    return false;
}

extern "C" IQK_API bool iqk_mul_mat_4d(long /*Nx*/, long /*Ny*/, long /*ne00*/,
        long /*ne02*/, long /*ne03*/, long /*ne12*/, long /*ne13*/,
        long /*nb02*/, long /*nb03*/, long /*nb12*/, long /*nb13*/, long /*nb2*/, long /*nb3*/,
        int /*typeA*/, const void * /*A*/, long /*strideA*/,
        int /*typeB*/, const void * /*B*/, long /*strideB*/,
        float * /*C*/, long /*stride_C*/, int /*ith*/, int /*nth*/) {
    GGML_ABORT("Unsupported CPU. You may need to manually set compilation flags\n");
    return false;
}

extern "C" IQK_API bool iqk_mul_mat_moe(long, long, long, int, int, const void *, long, int, const void *, long, float *, long, long,
        const void *, int, int) {
    GGML_ABORT("Unsupported CPU. You may need to manually set compilation flags\n");
    return false;
}

extern "C" IQK_API bool iqk_moe_fused_up_gate(long /*Nx*/, long /*Ny*/, long /*ne00*/, int /*ne11*/, int /*unary_op*/,
        int /*typeA*/, const void * /*Aup*/, const void * /*Agate*/, long /*strideA*/,
        int /*typeB*/, const void * /*B*/, long /*strideB*/,
        float * /*C*/, long /*nb1*/, long /*nb2*/, const void * /*vrow_mapping*/, float, int /*ith*/, int /*nth*/) {
    GGML_ABORT("Unsupported CPU. You may need to manually set compilation flags\n");
    return false;
}

#endif
