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

void iqk_grouped_top_k(struct ggml_tensor * dst, int ith, int nth);

void iqk_argsort(struct ggml_tensor * dst, int ith, int nth);

#ifdef __cplusplus
}
#endif


