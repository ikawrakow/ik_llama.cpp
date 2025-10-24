//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.cuh"

struct mmvq_args;

void iqk_mul_mat_vec_q(ggml_type type, const mmvq_args & args, cudaStream_t stream);

