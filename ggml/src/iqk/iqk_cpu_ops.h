//
// Copyright (C) 2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "iqk_config.h"
#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

void iqk_sumrows_div(struct ggml_tensor * div, int ith, int nth);

void iqk_grouped_top_k(struct ggml_tensor * dst, int ith, int nth);

void iqk_argsort(struct ggml_tensor * dst, int ith, int nth);

void iqk_bailingmoev2_experts(struct ggml_tensor * dst, struct ggml_tensor * topk, int ith, int nth);

void iqk_glm45moe_experts(struct ggml_tensor * dst, struct ggml_tensor * topk_view, int ith, int nth);

void iqk_openai_experts(struct ggml_tensor * topk, struct ggml_tensor * softmax, int ith, int nth);

void iqk_mul_multi_add(struct ggml_tensor * dst, int ith, int nth);

void iqk_hadamard(struct ggml_tensor * dst, int ith, int nth);

#ifdef __cplusplus
}
#endif


