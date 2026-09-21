//
// Copyright (C) 2026 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

// Packed KV-cache storage types - CPU quantisers.
//
//   GGML_TYPE_FP4_B16_E4M3   e2m1 values, block 16, e4m3 scale  (compressed KV)
//   GGML_TYPE_FP4_B32_E8M0   e2m1 values, block 32, e8m0 scale  (indexer K and Q)
//   GGML_TYPE_FP8_B32_E8M0   e4m3 values, block 32, e8m0 scale  (window KV)
//
//   s = 2^ceil(log2(amax/6))   e2m1 / e8m0, floor 6 * 2^-126
//   s = e4m3(amax/6)           e2m1 / e4m3, floor 6 * 2^-9
//   s = 2^ceil(log2(amax/448)) e4m3 / e8m0, floor 1e-4
//
// q = clamp(x/s, -max, max), rounded to nearest with ties to even. Codes are
// packed 2 per byte, element 2j in the low nibble. Storage-only: written by
// ggml_set_rows, read back through ggml_get_rows -> F32; never a compute-op
// operand.

#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void quantize_row_fp4_b16_e4m3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void dequantize_row_fp4_b16_e4m3(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_fp4_b32_e8m0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void dequantize_row_fp4_b32_e8m0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_fp8_b32_e8m0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void dequantize_row_fp8_b32_e8m0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

#ifdef __cplusplus
}
#endif