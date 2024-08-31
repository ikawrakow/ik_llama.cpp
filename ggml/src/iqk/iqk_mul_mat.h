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

bool iqk_soft_max_noalibi(int nc, int ir0, int ir1, int ne00, int ne01,
        const float * src, long stride_src,
              float * dst, long stride_dst,
        const float * mask, float scale, float * wp);

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

void iqk_flash_helper_2(bool is_alibi,
                       int nq,                 // number of elements in q
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

bool iqk_flash_helper_3(int ne00,
                        int nq,                 // number of elements in q
                        int nk,                 // number of rows in k
                        int stride_q,
                        int stride_k,           // distance between rows in k (in bytes)
                        int stride_v,           // distance between rows in v (in bytes)
                        int stride_m,           // distance between rows in mask (in bytes)
                        int stride_qkv,         // distance between rows in mask (in bytes)
                        const float * q,        // q vector
                        const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                        const void  * v,
                        const void  * mask,     // mask. If not null, assumed to be fp16. nk elements
                        float         scale,
                        float       * qkv);     // v*softmax(k*q)

void iqk_flash_helper_4(int ne00,
                        int nq,                 // number of elements in q
                        int nk,                 // number of rows in k
                        int stride_q,
                        int stride_k,           // distance between rows in k (in bytes)
                        int stride_v,           // distance between rows in v (in bytes)
                        int stride_m,           // distance between rows in mask (in bytes)
                        int stride_qkv,         // distance between rows in mask (in bytes)
                        const float * q,        // q vector
                        const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                        const void  * v,
                        const void  * mask,     // mask. If not null, assumed to be fp16. nk elements
                        float         scale,
                        float       * qk,       // work buffer for storing Q*K
                        float       * qkv);     // v*softmax(k*q)

bool iqk_flash_attention_noalibi_f16(int ith, int nth,
        int neq2, int neq3, int nek2, int nek3, int nev2, int nev3,
        int64_t nbq1, int64_t nbq2, int64_t nbq3,
        int64_t nbk2, int64_t nbk3, int64_t nbv2, int64_t nbv3,
                        int ne00,
                        int nq,                 // number of elements in q
                        int nk,                 // number of rows in k
                        int stride_q,
                        int stride_k,           // distance between rows in k (in bytes)
                        int stride_v,           // distance between rows in v (in bytes)
                        int stride_m,           // distance between rows in mask (in bytes)
                        int stride_qkv,         // distance between rows in mask (in bytes)
                        const float * q,        // q vector
                        const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                        const void  * v,
                        const void  * mask,     // mask. If not null, assumed to be fp16. nk elements
                        float         scale,
                        float       * qkv);     // v*softmax(k*q)

#ifdef __cplusplus
}
#endif
