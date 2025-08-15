//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include <cstdint>

bool iqk_flash_attn_impl(int type_k,             // type of k
                         int type_v,             // type of v
                         int Dk,                 // K head size
                         int Dv,                 // V head size
                         int nq,                 // number of columns in q
                         int nk,                 // number of rows in k
                         int stride_q,           // distance between q columns in bytes
                         int stride_k,           // distance between k rows in bytes
                         int stride_v,           // distance between v rows in bytes
                         int stride_m,           // distance between mask rows (in bytes
                         int stride_qkv,         // distance between rows in mask (in bytes)
                         const float * q,        // q matrix.
                         const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                         const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                         const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                         const float * sinksf,   // attention sinks
                         int           nsinks,   // number of sinks
                         float         scale,    // scale applied before softmax
                         float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                         float       * qkv,      // v*softmax(scale*(k*q))
                         float       * M,
                         float       * S);

void * iqk_repack_k(int type_k, int nek0, int nek1, int nek2, int nek3, long nbk1, long nbk2, long nbk3,
        const void * k, void * work, int ith, int nth, int& repacked_type, uint64_t& row_size);
