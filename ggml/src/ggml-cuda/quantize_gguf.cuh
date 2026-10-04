//
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "common.cuh"

// Bit-exact CUDA legacy block quants via Joel's ggml_cuda_quantize() entry in kt-encoder.cu.
// Byte-identical to CPU: exact reductions, fixed rounding, RN division/FP16;
// sigma2 + make_qx/make_qkx3 replayed in CPU order.
// Order after KT: Q8_0, Q6_0, Q5_0, Q4_0 (+Q5_1/Q4_1/IQ4_NL/IQ4_XS); Q5_0/Q4_0
// sections removable; Q6_0 OLS kept, Q8_0 ignores imatrix like CPU.
// Chunked host entries (~128 MiB F32/chunk), checked CUDA calls, return bytes
// written or 0 for CPU fallback.
// Q8_0 (plain ref + fudge; imatrix is ignored to match CPU quantize_q8_0)
size_t ggml_cuda_quantize_q8_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q8_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// Q6_0 OLS (make_qx_quants + fudge; weight = x*x without imatrix)
size_t ggml_cuda_quantize_q6_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q6_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// --- Removable Q5_0 section (delete below to drop Q5_0) ---
size_t ggml_cuda_quantize_q5_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q5_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// --- Removable Q4_0 section (delete below to drop Q4_0) ---
size_t ggml_cuda_quantize_q4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q4_0_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// Q5_1 (plain min/max ref, no fudge; imatrix via make_qkx3, no fudge)
size_t ggml_cuda_quantize_q5_1(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q5_1_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// Q4_1 (plain min/max ref, no fudge; imatrix via make_qkx3, no fudge)
size_t ggml_cuda_quantize_q4_1(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_q4_1_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// IQ4_NL (ntry=7, w=x*x plain or qw*sqrt imatrix)
size_t ggml_cuda_quantize_iq4_nl(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_iq4_nl_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
// IQ4_XS (ntry=7, w=x*x plain or qw*sqrt imatrix)
size_t ggml_cuda_quantize_iq4_xs(const float * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_cuda_quantize_iq4_xs_imatrix(const float * src, void * dst, int64_t nrows, int64_t n_per_row,
        const float * imatrix);
