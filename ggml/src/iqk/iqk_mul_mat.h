//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

bool iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth);

bool iqk_mul_mat_moe(long Nx, long Ny, long ne00, int ne11,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long nb1, long nb2, const void * vrow_mapping, int ith, int nth);

bool iqk_fused_mul_mat_softmax(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C,
        char * work_buffer, long work_size,
        const char * mask, float scale, float slope,
        int ith, int nth);

void iqk_flash_helper(int nq,                 // number of elements in q
                      int nk,                 // number of rows in k
                      int stride_k,           // distance between rows in k (in bytes)
                      const float * q,        // q vector
                      const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                      const void  * mask,     // mask. If not null, assumed to be fp16. nk elements
                      float         scale,
                      float         slope,
                      float       * qk);      // softmax(k*q) - k elements

void iqk_flash_helper_2(int nq,                 // number of elements in q
                       int nk,                 // number of rows in k
                       int stride_k,           // distance between rows in k (in bytes)
                       int stride_v,           // distance between rows in k (in bytes)
                       const float * q,        // q vector
                       const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                       const void  * v,
                       const void  * mask,     // mask. If not null, assumed to be fp16. nk elements
                       float         scale,
                       float         slope,
                       float       * qk,
                       float       * qkv);      // softmax(k*q) - k elements

#ifdef __cplusplus
}
#endif
