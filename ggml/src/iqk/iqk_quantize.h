//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include <stdint.h>
#include <stddef.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#ifdef __cplusplus
#define GGML_RESTRICT
extern "C" {
#else
#define GGML_RESTRICT restrict
#endif

void   quantize_row_iq2_k_ref(const float * GGML_RESTRICT x, block_iq2_k  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq2_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq2_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq2_k(const block_iq2_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq2_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq3_k_ref(const float * GGML_RESTRICT x, block_iq3_k  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq3_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq3_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq3_k(const block_iq3_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq3_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq3_ks_ref(const float * GGML_RESTRICT x, block_iq3_ks  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq3_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq3_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq3_ks(const block_iq3_ks  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq3_ks_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq4_k_ref(const float * GGML_RESTRICT x, block_iq4_k  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq4_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq4_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq4_k(const block_iq4_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq4_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq5_k_ref(const float * GGML_RESTRICT x, block_iq5_k  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq5_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq5_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq5_k(const block_iq5_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq5_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq6_k_ref(const float * GGML_RESTRICT x, block_iq6_k  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq6_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq6_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq6_k(const block_iq6_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq6_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq4_ks_ref(const float * GGML_RESTRICT x, block_iq4_ks  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq4_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq4_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq4_ks(const block_iq4_ks  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq4_ks_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq4_kss_ref(const float * GGML_RESTRICT x, block_iq4_kss  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq4_kss(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq4_kss(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq4_kss(const block_iq4_kss  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq4_kss_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq2_ks_ref(const float * GGML_RESTRICT x, block_iq2_ks  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq2_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq2_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq2_ks(const block_iq2_ks  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq2_ks_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq2_kl_ref(const float * GGML_RESTRICT x, block_iq2_kl  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq2_kl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq2_kl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq2_kl(const block_iq2_kl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq2_kl_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq2_kt_ref(const float * GGML_RESTRICT x, block_iq2_kt  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq2_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq2_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq2_kt(const block_iq2_kt  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq2_kt_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq3_kt_ref(const float * GGML_RESTRICT x, block_iq3_kt  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq3_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq3_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq3_kt(const block_iq3_kt  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq3_kt_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq4_kt_ref(const float * GGML_RESTRICT x, block_iq4_kt  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq4_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq4_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq4_kt(const block_iq4_kt  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq4_kt_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq5_ks_ref(const float * GGML_RESTRICT x, block_iq5_ks  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq5_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq5_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq5_ks(const block_iq5_ks  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq5_ks_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq4_nl_r4_ref(const float * GGML_RESTRICT x, block_iq4_nl_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq4_nl_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq4_nl_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq4_nl_r4(const block_iq4_nl_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq4_nl_r4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_q4_0_r8_ref(const float * GGML_RESTRICT x, block_iq4_nl_r8  * GGML_RESTRICT y, int64_t k);
void   quantize_row_q4_0_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q4_0_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_q4_0_r8(const block_iq4_nl_r8  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q4_0_r8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_q8_0_r8_ref(const float * GGML_RESTRICT x, block_q8_0_r8  * GGML_RESTRICT y, int64_t k);
void   quantize_row_q8_0_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q8_0_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_q8_0_r8(const block_q8_0_r8  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q8_0_r8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_q5_0_r4_ref(const float * GGML_RESTRICT x, block_q5_0_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_q5_0_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q5_0_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_q5_0_r4(const block_q5_0_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q5_0_r4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_q6_0_r4_ref(const float * GGML_RESTRICT x, block_q6_0_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_q6_0_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q6_0_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_q6_0_r4(const block_q6_0_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q6_0_r4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq4_xs_r8_ref(const float * GGML_RESTRICT x, block_iq4_xs_r8 * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq4_xs_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq4_xs_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq4_xs_r8(const block_iq4_xs_r8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq4_xs_r8_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq2_bn_ref (const float * GGML_RESTRICT x, block_iq2_bn  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq2_bn (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void   dequantize_row_iq2_bn (const block_iq2_bn  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
size_t quantize_iq2_bn (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   vec_dot_iq2_bn_q8_K64(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq2_bn_r4_ref (const float * GGML_RESTRICT x, block_iq2_bn  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq2_bn_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void   dequantize_row_iq2_bn_r4(const block_iq2_bn * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
size_t quantize_iq2_bn_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   vec_dot_iq2_bn_r4_q8_K64(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_q3_k_r4_ref(const float * GGML_RESTRICT x, block_q3_k_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_q3_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q3_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_q3_k_r4(const block_q3_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q3_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_q2_k_r4_ref(const float * GGML_RESTRICT x, block_q2_k_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_q2_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q2_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_q2_k_r4(const block_q2_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q2_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_q4_k_r4_ref(const float * GGML_RESTRICT x, block_q4_k_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_q4_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q4_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_q4_k_r4(const block_q4_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q4_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_q5_k_r4_ref(const float * GGML_RESTRICT x, block_q5_k_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_q5_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q5_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_q5_k_r4(const block_q5_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q5_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_q6_k_r4_ref(const float * GGML_RESTRICT x, block_q6_k_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_q6_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q6_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_q6_k_r4(const block_q6_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q6_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq5_k_r4_ref(const float * GGML_RESTRICT x, block_iq5_k_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq5_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq5_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq5_k_r4(const block_iq5_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq5_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq4_k_r4_ref(const float * GGML_RESTRICT x, block_iq4_k_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq4_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq4_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq4_k_r4(const block_iq4_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq4_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq3_k_r4_ref(const float * GGML_RESTRICT x, block_iq3_k_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq3_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq3_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq3_k_r4(const block_iq3_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq3_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq2_k_r4_ref(const float * GGML_RESTRICT x, block_iq2_k_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq2_k_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq2_k_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq2_k_r4(const block_iq2_k_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq2_k_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq4_ks_r4_ref(const float * GGML_RESTRICT x, block_iq4_ks_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq4_ks_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq4_ks_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq4_ks_r4(const block_iq4_ks_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq4_ks_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq5_ks_r4_ref(const float * GGML_RESTRICT x, block_iq5_ks_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq5_ks_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq5_ks_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq5_ks_r4(const block_iq5_ks_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq5_ks_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq2_xxs_r4_ref(const float * GGML_RESTRICT x, block_iq2_xxs_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq2_xxs_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq2_xxs_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq2_xxs_r4(const block_iq2_xxs_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq2_xxs_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq2_xs_r4_ref(const float * GGML_RESTRICT x, block_iq2_xs_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq2_xs_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq2_xs_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq2_xs_r4(const block_iq2_xs_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq2_xs_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq2_s_r4_ref(const float * GGML_RESTRICT x, block_iq2_s_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq2_s_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq2_s_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq2_s_r4(const block_iq2_s_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq2_s_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq3_xxs_r4_ref(const float * GGML_RESTRICT x, block_iq3_xxs_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq3_xxs_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq3_xxs_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq3_xxs_r4(const block_iq3_xxs_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq3_xxs_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq3_s_r4_ref(const float * GGML_RESTRICT x, block_iq3_s_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq3_s_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq3_s_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq3_s_r4(const block_iq3_s_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq3_s_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq1_s_r4_ref(const float * GGML_RESTRICT x, block_iq1_s_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq1_s_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq1_s_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq1_s_r4(const block_iq1_s_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq1_s_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq1_m_r4_ref(const float * GGML_RESTRICT x, block_iq1_m_r4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq1_m_r4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq1_m_r4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq1_m_r4(const block_iq1_m_r4  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq1_m_r4_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_q8_k_r8_ref(const float * GGML_RESTRICT x, block_q8_k_r8  * GGML_RESTRICT y, int64_t k);
void   quantize_row_q8_k_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q8_k_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_q8_k_r8(const block_q8_k_r8  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q8_k_r8_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_q8_KV_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void   quantize_row_q8_KV(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q8_KV(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_q8_KV(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q8_KV_q8_KV(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_q8_KV_r8_ref(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void   quantize_row_q8_KV_r8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_q8_KV_r8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_q8_KV_r8(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_q8_KV_r8_q8_KV(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void iqk_quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void quantize_row_q8_K64_ref(const float * GGML_RESTRICT x, block_q8_K64 * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K64(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K128(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K16(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K32(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_KR8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0_x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_1_x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_2_x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void repack_f32_bf16_r16 (const void * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row);
void repack_bf16_bf16_r16(const void * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row);

void iqk_repack_tensor(struct ggml_tensor * tensor);
bool iqk_modify_tensor(struct ggml_tensor * tensor);

int iqk_repacked_type(const struct ggml_tensor * tensor); // int instead of ggml_type so we don't need to include ggml.h
bool iqk_should_modify_tensor(const struct ggml_tensor * tensor);

// So we can re-pack Microsoft's BitNet I2_S quants
void dequantize_row_ms_i2s(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

typedef void (*to_float_t)  (const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
typedef void (*from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int64_t k);
void iqk_quantize_any(int from_type, int to_type,
                      int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                      uint64_t nb0, uint64_t nb1, uint64_t nb2, uint64_t nb3,
                      const void * GGML_RESTRICT x, void * GGML_RESTRICT y, void * work_buffer,
                      to_float_t to_float, from_float_t from_float, int ith, int nth);

#ifdef __cplusplus
}
#endif
