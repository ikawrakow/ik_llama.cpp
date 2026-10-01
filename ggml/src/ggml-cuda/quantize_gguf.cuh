//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2026 Nexesenex
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "common.cuh"

// Bit-exact CUDA quantization of legacy block quants for GGUF, dispatched
// through Joel's single entry ggml_cuda_quantize() (ggml-cuda.h) in
// kt-encoder.cu, which remains the lead plumbing. This header declares the
// per-type helpers (device 0, chunked, return bytes or 0).
// Guaranteed byte-for-byte identical to the CPU quantize_q* GGUF paths because
// every stored byte depends only on:
//   - max/min reductions (exact, order-independent; argmax ties broken by the
//     lowest index, matching the reference's sequential scan),
//   - per-element rounding (fixed rounding mode),
//   - nearest-even FP16 conversion (__float2half_rn) of (fudge * d),
//   - correctly-rounded division (__fdiv_rn), immune to -use_fast_math.
// No cross-element floating point accumulation is ever introduced, except in
// the imatrix / Q6_0-OLS path where the row-level sigma2 sum is
// order-dependent and is therefore computed on the host in the exact CPU
// summation order, and where the per-block make_qx_quants greedy optimizer is
// replayed sequentially by a single thread in the exact CPU order. That path
// additionally requires make_qx_quants / quantize_row_q4_0_impl /
// quantize_row_q5_0_impl / quantize_row_q6_0_impl in ggml-quants.c to be
// compiled with #pragma STDC FP_CONTRACT OFF (as they now are), so a
// /arch:AVX2 build cannot FMA-contract sumlx += w*x*l and drift by ~1 ulp.
//
// Order after Joel's KT: Q8_0, Q6_0, Q5_0, Q4_0. Q5_0 and Q4_0 sections are
// self-contained so they can be removed with minimal work once Ikawrakow
// confirms (delete their kernels + helpers + dispatcher cases).
//
// Q6_0 OLS is KEPT (not reverted): quantize_q6_0 always runs the OLS
// make_qx_quants impl (weight = x*x without imatrix, qw*sqrt(sigma2+x*x)
// with imatrix) with fudge factor, matching HEAD CPU. Q8_0 ignores imatrix
// on the CPU (quantize_q8_0), so the CUDA Q8_0 path also ignores it.
//
// The host entries process the tensor in fixed-size device chunks (~128 MiB
// F32 per chunk) so single large tensors never require a large contiguous
// VRAM allocation, and check every CUDA return code. On any failure they
// print the CUDA error and return 0 (the caller falls back to the CPU).
//
// Returns the number of bytes written to dst (nrows * ggml_row_size(...)),
// or 0 if the quantization could not be executed (e.g. no CUDA device).
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
