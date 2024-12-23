//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#if GGML_USE_IQK_MULMAT
#include "iqk_mul_mat.h"
#endif
#include "ggml-quants.h"
#include "ggml-impl.h"
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "iqk_quantize.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cmath>
#include <array>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <thread>
#include <atomic>
#include <unordered_map>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#include <intrin.h>
#include <ammintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
inline int popcount(uint8_t x) { return __popcnt(x); }
inline int popcount(uint16_t x) { return __popcnt(x); }
inline int popcount(uint32_t x) { return __popcnt(x); }
inline int popcount(uint64_t x) { return _mm_popcnt_u64(x); }
#else
constexpr int popcount(uint8_t x) { return __builtin_popcount(x); }
constexpr int popcount(uint16_t x) { return __builtin_popcount(x); }
constexpr int popcount(uint32_t x) { return __builtin_popcount(x); }
constexpr int popcount(uint64_t x) { return __builtin_popcountll(x); }
#endif

namespace {

inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

float make_qx_quants(int n, int nmax, const float * x, int8_t * L, const float * qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (!amax) { // all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax-1, l));
        L[i] = l + nmax;
        sumlx += qw[i]*x[i]*l;
        suml2 += qw[i]*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax-1, l));
            sumlx += qw[i]*x[i]*l;
            suml2 += qw[i]*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + std::max(-nmax, std::min(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

struct IQ1BNQuantizer {
    int8_t L[QK_IQ1BN];
    void quantize_one_row_1bn(const float * src, block_iq1_bn * y, int n_per_row, const float * imatrix);
    void quantize_one_row_2bn(const float * src, block_iq2_bn * y, int n_per_row, const float * imatrix);
    static inline float row_max(int n_per_row, const float * src) {
        float max_in_row = 0;
        for (int j = 0; j < n_per_row; ++j) {
            float ax = fabsf(src[j]);
            max_in_row = std::max(max_in_row, ax);
        }
        return max_in_row;
    }
    // The Makefile has issues dwaling with this?
    //static constexpr uint8_t k_mult[5] = {81, 27, 9, 3, 1};
    static const uint8_t k_mult[5];
};

const uint8_t IQ1BNQuantizer::k_mult[5] = {81, 27, 9, 3, 1};

void IQ1BNQuantizer::quantize_one_row_1bn(const float * src, block_iq1_bn * y, int n_per_row, const float * imatrix) {

    static const int k_nb[6] = {1, 3, 9, 27, 81, 243};
    (void)imatrix;

    const int nblock = n_per_row/QK_IQ1BN;

    ggml_half * dptr = (ggml_half *)y;
    y = (block_iq1_bn *)(dptr + 1);

    float max = 0;
    for (int j = 0; j < n_per_row; ++j) max = std::max(max, fabsf(src[j]));
    ggml_half d = GGML_FP32_TO_FP16(max);
    std::memcpy(dptr, &d, sizeof(d));

    float thresh = 0.5f*max;

    for (int ib = 0; ib < nblock; ++ib) {
        std::memset(&y[ib], 0, sizeof(block_iq1_bn));
        auto xb = src + ib*QK_IQ1BN;
        int v13 = 0;
        for (int i16 = 0; i16 < QK_IQ1BN/16; ++i16) {
            for (int k = 0; k < 3; ++k) {
                int idx = 0;
                for (int j = 0; j < 5; ++j) {
                    float v = xb[16*i16 + 5*k + j];
                    int q = fabsf(v) < thresh ? 1 : v < 0 ? 0 : 2;
                    idx += k_nb[j]*q;
                }
                idx = (256*idx + k_nb[5] - 1)/k_nb[5];
                y[ib].ql[3*i16 + k] = idx;
            }
            float v = xb[16*i16 + 15];
            int q = fabsf(v) < thresh ? 1 : v < 0 ? 0 : 2;
            v13 += k_nb[i16]*q;
        }
        y[ib].extra = (256*v13 + k_nb[5] - 1)/k_nb[5];
    }
}

void IQ1BNQuantizer::quantize_one_row_2bn(const float * src, block_iq2_bn * y, int n_per_row, const float * imatrix) {

    (void)imatrix;

    const int nblock = n_per_row/QK_IQ1BN;

    constexpr int Nj = QK_IQ1BN/4;

    float max = 0;
    for (int j = 0; j < n_per_row; ++j) max = std::max(max, fabsf(src[j]));

    float * dptr = (float *)y;
    *dptr = max;
    y = (block_iq2_bn *)(dptr + 1);
    float thresh = 0.5f*max;

    for (int ib = 0; ib < nblock; ++ib) {
        auto xb = src + QK_IQ1BN*ib;
        for (int j = 0; j < QK_IQ1BN; ++j) {
            L[j] = fabsf(xb[j]) < thresh ? 1 : xb[j] < 0 ? 0 : 2;
        }
        for (int j = 0; j < Nj; ++j) {
            y[ib].qs[j] = L[j] | (L[j + Nj] << 2) | (L[j + 2*Nj] << 4) | (L[j + 3*Nj] << 6);
        }
    }
}

}

size_t quantize_iq1_bn(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    IQ1BNQuantizer iq1bn;
    auto row_size = ggml_row_size(GGML_TYPE_IQ1_BN, n_per_row);
    auto qrow = (char *)dst;
    for (int row = 0; row < nrows; ++row) {
        iq1bn.quantize_one_row_1bn(src + row*n_per_row, (block_iq1_bn *)qrow, n_per_row, imatrix);
        qrow += row_size;
    }
    return nrows*row_size;
}

void quantize_row_iq1_bn_ref(const float * x, block_iq1_bn * y, int64_t k) {
    quantize_iq1_bn(x, y, 1, k, nullptr);
}

void quantize_row_iq1_bn(const float * x, void * y, int64_t k) {
    quantize_iq1_bn(x, y, 1, k, nullptr);
}

void dequantize_row_iq1_bn(const block_iq1_bn * x, float * y, int64_t k) {
    assert(k%QK_IQ1BN == 0);
    int nblock = k / QK_IQ1BN;

    for (int i = 0; i < nblock; ++i) {
        uint8_t extra = x[i].extra;
        auto ql = x[i].ql;
        for (int i16 = 0; i16 < QK_IQ1BN/16; ++i16) {
            for (int k = 0; k < 3; ++k) {
                for (int j = 0; j < 5; ++j) {
                    uint8_t v = ql[k]*IQ1BNQuantizer::k_mult[j];
                    int8_t vs = ((v + (v >> 1)) >> 7);
                    *y++ = vs - 1;
                }
            }
            ql += 3;
            uint8_t v = extra*IQ1BNQuantizer::k_mult[i16];
            int8_t vs = ((v + (v >> 1)) >> 7);
            *y++ = vs - 1;
        }
    }
}

size_t quantize_iq2_bn(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    IQ1BNQuantizer iq1bn;
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_BN, n_per_row);
    auto qrow = (char *)dst;
    for (int row = 0; row < nrows; ++row) {
        iq1bn.quantize_one_row_2bn(src + row*n_per_row, (block_iq2_bn *)qrow, n_per_row, imatrix);
        qrow += row_size;
    }
    return nrows*row_size;
}

void quantize_row_iq2_bn_ref(const float * x, block_iq2_bn * y, int64_t k) {
    quantize_iq2_bn(x, y, 1, k, nullptr);
}

void quantize_row_iq2_bn(const float * x, void * y, int64_t k) {
    quantize_iq2_bn(x, y, 1, k, nullptr);
}

void dequantize_row_iq2_bn(const block_iq2_bn * x, float * y, int64_t k) {
    assert(k%QK_IQ1BN == 0);
    int nblock = k / QK_IQ1BN;

    auto d1 = 1.f, d2 = 0.25f, d3 = d2*0.25f, d4 = d3*0.25f;
    auto m = -1.f;
    constexpr int Nj = QK_IQ1BN/4;
    for (int i = 0; i < nblock; ++i) {
        for (int j = 0; j < Nj; ++j) {
            y[j+   0] = d1*(x[i].qs[j] & 0x03) + m;
            y[j+1*Nj] = d2*(x[i].qs[j] & 0x0c) + m;
            y[j+2*Nj] = d3*(x[i].qs[j] & 0x30) + m;
            y[j+3*Nj] = d4*(x[i].qs[j] & 0xc0) + m;
        }
        y += QK_IQ1BN;
    }
}

namespace {
inline int8_t iq1bn_dequant(uint8_t q, int i) {
    uint8_t v = IQ1BNQuantizer::k_mult[i]*q;
    //int8_t vs = (v + (v << 1)) >> 8;
    int8_t vs = 3*v >> 8;
    return vs - 1;
}
}

static const int8_t iq1bn_values[1280] = {
    -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  0, -1, -1, -1,  0,  0, -1, -1, -1,  1,  0,
    -1, -1, -1, -1,  1, -1, -1, -1,  0,  1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1,  0, -1, -1,  0, -1,  0, -1, -1,  1, -1,  0, -1,
    -1, -1,  0,  0, -1, -1,  0,  0,  0, -1, -1,  1,  0,  0, -1, -1, -1,  1,  0, -1, -1,  0,  1,  0, -1, -1,  1,  1,  0, -1, -1, -1,
    -1,  1, -1, -1,  0,  0,  0,  0,  0,  0, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,  0,  1, -1, -1,  0,  0,  1, -1, -1,  1,  0,  1,
    -1, -1, -1,  1,  1, -1, -1,  0,  1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  0, -1,  0, -1, -1,  0, -1,  1, -1, -1,  0, -1,
    -1,  0, -1,  0, -1,  0,  0, -1,  0, -1,  1,  0, -1,  0, -1, -1,  1, -1,  0, -1,  0,  1, -1,  0, -1,  1,  1, -1,  0, -1, -1, -1,
     0,  0, -1,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0,  1, -1,  0,  0, -1, -1,  0,  0,  0, -1,  0,  0,  0,  0, -1,  1,  0,  0,  0,
    -1, -1,  1,  0,  0, -1,  0,  1,  0,  0, -1,  1,  1,  0,  0, -1, -1, -1,  1,  0, -1,  0, -1,  1,  0, -1,  1, -1,  1,  0, -1, -1,
     0,  1,  0, -1,  0,  0,  1,  0, -1,  1,  0,  1,  0, -1, -1,  1,  1,  0, -1,  0,  1,  1,  0, -1,  1,  1,  1,  0, -1, -1, -1, -1,
     1, -1,  0, -1, -1,  1, -1,  1, -1, -1,  1, -1,  0,  0,  0,  0,  0, -1,  0, -1,  1, -1,  0,  0, -1,  1, -1,  1,  0, -1,  1, -1,
    -1,  1, -1,  1, -1,  0,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1,  0,  1, -1,  0, -1,  0,  1, -1,  1, -1,  0,  1, -1, -1,  0,
     0,  1, -1,  0,  0,  0,  1, -1,  1,  0,  0,  1, -1, -1,  1,  0,  1, -1,  0,  1,  0,  1, -1,  1,  1,  0,  1, -1, -1, -1,  1,  1,
    -1,  0, -1,  1,  1, -1,  1, -1,  1,  1, -1,  0,  0,  0,  0,  0, -1,  0,  1,  1, -1,  0,  0,  1,  1, -1,  1,  0,  1,  1, -1, -1,
     1,  1,  1, -1,  0,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1,  0,  1, -1, -1, -1,  0, -1,  0, -1,
    -1,  0,  0,  0, -1, -1,  0,  1,  0, -1, -1,  0, -1,  1, -1, -1,  0,  0,  1, -1, -1,  0,  1,  1, -1, -1,  0, -1, -1,  0, -1,  0,
     0, -1,  0, -1,  0,  1, -1,  0, -1,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1,  0,  0, -1,  0, -1,  1,
     0, -1,  0,  0,  1,  0, -1,  0,  1,  1,  0, -1,  0, -1, -1,  1, -1,  0,  0, -1,  1, -1,  0,  1, -1,  1, -1,  0, -1,  0,  1, -1,
     0,  0,  0,  1, -1,  0,  1,  0,  1, -1,  0, -1,  1,  1, -1,  0,  0,  1,  1, -1,  0,  1,  1,  1, -1,  0, -1, -1, -1,  0,  0,  0,
    -1, -1,  0,  0,  1, -1, -1,  0,  0, -1,  0, -1,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  1,  0, -1,  0,  0, -1,  1, -1,
     0,  0,  0,  1, -1,  0,  0,  1,  1, -1,  0,  0, -1, -1,  0,  0,  0,  0, -1,  0,  0,  0,  1, -1,  0,  0,  0, -1,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  1,  0,  0,  0,  0, -1,  1,  0,  0,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0,  0, -1, -1,  1,  0,  0,  0, -1,
     1,  0,  0,  1, -1,  1,  0,  0, -1,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0, -1,  1,  1,  0,
     0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0, -1, -1, -1,  1,  0,  0, -1, -1,  1,  0,  1, -1, -1,  1,  0, -1,  0, -1,  1,  0,  0,
     0, -1,  1,  0,  1,  0, -1,  1,  0, -1,  1, -1,  1,  0,  0,  1, -1,  1,  0,  1,  1, -1,  1,  0, -1, -1,  0,  1,  0,  0, -1,  0,
     1,  0,  1, -1,  0,  1,  0, -1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0, -1,  1,  0,  1,  0,
     0,  1,  0,  1,  0,  1,  1,  0,  1,  0, -1, -1,  1,  1,  0,  0, -1,  1,  1,  0,  1, -1,  1,  1,  0, -1,  0,  1,  1,  0,  0,  0,
     1,  1,  0,  1,  0,  1,  1,  0, -1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0, -1, -1, -1, -1,  1,  0, -1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  0, -1, -1,  1,  0,  0, -1, -1,  1,  1,  0, -1, -1,  1, -1,  1, -1, -1,  1,  0,  0,  0,  0,  0,  0,
     1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  0, -1,  1,  0, -1,  0, -1,  1,  1, -1,  0, -1,  1, -1,  0,  0, -1,  1,  0,  0,  0,
    -1,  1,  1,  0,  0, -1,  1, -1,  1,  0, -1,  1,  0,  1,  0, -1,  1,  1,  1,  0, -1,  1, -1, -1,  1, -1,  1,  0, -1,  1, -1,  1,
     1, -1,  1, -1,  1, -1,  0,  1, -1,  1,  0,  0,  1, -1,  1,  1,  0,  1, -1,  1, -1,  1,  1, -1,  1,  0,  0,  0,  0,  0,  0,  1,
     1, -1,  1,  1,  1,  1, -1,  1, -1, -1, -1,  0,  1,  0, -1, -1,  0,  1,  1, -1, -1,  0,  1, -1,  0, -1,  0,  1,  0,  0, -1,  0,
     1,  1,  0, -1,  0,  1, -1,  1, -1,  0,  1,  0,  1, -1,  0,  1,  1,  1, -1,  0,  1, -1, -1,  0,  0,  1,  0, -1,  0,  0,  1,  1,
    -1,  0,  0,  1, -1,  0,  0,  0,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  1, -1,  1,  0,  0,  1,  0,  1,  0,  0,  1,  0,  0,  0,
     0,  0,  1,  1,  0,  0,  1, -1, -1,  1,  0,  1,  0, -1,  1,  0,  1,  1, -1,  1,  0,  1, -1,  0,  1,  0,  1,  0,  0,  1,  0,  1,
     1,  0,  1,  0,  1, -1,  1,  1,  0,  1,  0,  1,  1,  0,  1,  1,  1,  1,  0,  1, -1, -1, -1,  1,  1,  0, -1, -1,  1,  1,  1, -1,
    -1,  1,  1, -1,  0, -1,  1,  1,  0,  0, -1,  1,  1,  1,  0, -1,  1,  1, -1,  1, -1,  1,  1,  0,  1, -1,  1,  1,  1,  1, -1,  1,
     1,  0,  0,  0,  0,  0, -1, -1,  0,  1,  1,  0, -1,  0,  1,  1,  1, -1,  0,  1,  1, -1,  0,  0,  1,  1,  0,  0,  0,  1,  1,  1,
     0,  0,  1,  1, -1,  1,  0,  1,  1,  0,  1,  0,  1,  1,  1,  1,  0,  1,  1, -1, -1,  1,  1,  1,  0, -1,  1,  1,  1,  1, -1,  1,
     1,  1, -1,  0,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  1,  1,  1, -1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,
};

void ggml_vec_dot_iq1_bn_q8_K64(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    static_assert(QK_IQ1BN == 64, "This dot product implementation for iq1_bn requires a block size of 64");

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ1_BN, vx, 0, GGML_TYPE_Q8_K64, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    const block_iq1_bn * x = (const block_iq1_bn *)vx;

    const float * d8 = (const float *)vy;
    const int8_t * q8 = (const int8_t *)(d8 + 4);
    int nblock = n / QK_IQ1BN;

    int sumi[8] = {};
    int8_t q1[16];

    for (int ii = 0; ii < nblock; ii += 32) {
        int16_t sum16[8] = {};
        int nb = std::min(ii + 32, nblock);
        for (int i = ii; i < nb; ++i) {
            auto ql = x[i].ql;
            const int8_t * extra = iq1bn_values + 5*x[i].extra;
            for (int i16 = 0; i16 < QK_IQ1BN/16; ++i16) {
                for (int k = 0; k < 3; ++k) {
                    uint8_t q = *ql++;
                    const int8_t * vs = iq1bn_values + 5*q;
                    for (int j = 0; j < 5; ++j) q1[5*k+j] = vs[j];
                }
                q1[15] = extra[i16];
                // We collect 8 q8 values per block into each element of sum16
                // => 32 x 8 = 256 values in each loop over i, so this cannot overflow the int16_t range
                //    (q8 is in -127...127, and hence the sum is in -32512...32512
                for (int j = 0; j < 8; ++j) sum16[j] += q8[2*j+0]*q1[2*j+0] + q8[2*j+1]*q1[2*j+1];
                q8 += 16;
            }
        }
        for (int j = 0; j < 8; ++j) sumi[j] += sum16[j];
    }

    *s = d8[0] * (sumi[0] + sumi[1]) + d8[1] * (sumi[2] + sumi[3]) + d8[2] * (sumi[4] + sumi[5]) + d8[3] * (sumi[6] + sumi[7]);
}

void vec_dot_iq2_bn_q8_K64(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    static_assert(QK_IQ1BN == 64, "This dot product implementation for iq2_bn requires a block size of 64");

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_BN, vx, 0, GGML_TYPE_Q8_K64, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    constexpr int Nj = QK_IQ1BN/4;

    const block_iq2_bn * x = (const block_iq2_bn *)vx;
    int nblock = n / QK_IQ1BN;

    const float * d = (const float *)vy;
    const int8_t * q8 = (const int8_t *)(d + 4);

    int sum[16] = { };
    int sum0[4] = { };

    for (int i = 0; i < nblock; ++i) {
        for (int j = 0; j < Nj/4; ++j) {
            for (int l = 0; l < 4; ++l) {
                sum[4*j + 0] += q8[4*j + l +    0] * (x[i].qs[4*j+l] & 0x03);
                sum[4*j + 1] += q8[4*j + l + 1*Nj] * (x[i].qs[4*j+l] & 0x0c);
                sum[4*j + 2] += q8[4*j + l + 2*Nj] * (x[i].qs[4*j+l] & 0x30);
                sum[4*j + 3] += q8[4*j + l + 3*Nj] * (x[i].qs[4*j+l] & 0xc0);
                sum0[j] += q8[4*j + l] + q8[4*j + l + 1*Nj] + q8[4*j + l + 2*Nj] + q8[4*j + l + 3*Nj];
            }
        }
        q8 += QK_IQ1BN;
    }

    float sumf = 0;
    for (int j = 0; j < 4; ++j) {
        sumf += d[j] * (sum[4*j + 0] + 0.25f*sum[4*j + 1] + 0.0625*sum[4*j + 2] + 0.015625*sum[4*j + 3] - sum0[j]);
    }
    *s = sumf;

}

void quantize_row_q8_K64_ref(const float * x, block_q8_K64 * y, int64_t k) {

    GGML_ASSERT(k >= 8*QK_IQ1BN);

    float * dptr = (float *)y;
    auto qs = (int8_t *)(dptr + 8);
#ifdef __ARM_NEON
    static const uint8_t k_shuffle[16] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60};
    auto shuffle = vld1q_u8(k_shuffle);
    float32x4_t max[4] = { };
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            auto val = vld1q_f32(x + j + 4*i);
            val = vabsq_f32(val);
            max[i] = vmaxq_f32(max[i], val);
        }
    }
    float32x4_t vid[4];
    for (int i = 0; i < 4; ++i) {
        dptr[i] = vmaxvq_f32(max[i])/127;
        float id = dptr[i] > 0 ? 1/dptr[i] : 0.f;
        vid[i] = vdupq_n_f32(id);
    }
    int8x16x4_t q;
    int32x4_t qsum = {};
    const int8x16_t m1 = vdupq_n_s8(1);
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            auto val = vld1q_f32(x + j + 4*i);
            val = vmulq_f32(vid[i], val);
            auto ival = vcvtnq_s32_f32(val);
            q.val[i] = vreinterpretq_s8_s32(ival);
        }
        auto qi = vqtbl4q_s8(q, shuffle);
        qsum = ggml_vdotq_s32(qsum, qi, m1);
        vst1q_s8(qs, qi);
        qs += 16;
    }
    auto sumf = vmulq_f32(vld1q_f32(dptr), vcvtq_f32_s32(qsum));
    vst1q_f32(dptr + 4, sumf);
#elif defined __AVX__
    __m128 max[4] = {};
    __m128 sign_bit = _mm_set1_ps(-0.f);
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            auto val = _mm_loadu_ps(x + j + 4*i);
            val = _mm_andnot_ps(sign_bit, val);
            max[i] = _mm_max_ps(max[i], val);
        }
    }
    __m128 vid[4];
    for (int i = 0; i < 4; ++i) {
        max[i] = _mm_max_ps(max[i], _mm_movehl_ps(max[i], max[i]));
        max[i] = _mm_max_ss(max[i], _mm_movehdup_ps(max[i]));
        float maxi = _mm_cvtss_f32(max[i]);
        dptr[i] = maxi/127;
        float id = dptr[i] > 0 ? 1/dptr[i] : 0.f;
        vid[i] = _mm_set1_ps(id);
    }
    __m128i q[4];
    __m128i sums = _mm_setzero_si128();
    __m128i m1_8 = _mm_set1_epi8(1);
    __m128i m1_16 = _mm_set1_epi16(1);
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            auto val = _mm_loadu_ps(x + j + 4*i);
            val = _mm_round_ps(_mm_mul_ps(vid[i], val), _MM_ROUND_NEAREST);
            q[i] = _mm_cvtps_epi32(val);
        }
        auto q1 = _mm_packs_epi32(q[0], q[1]);
        auto q2 = _mm_packs_epi32(q[2], q[3]);
        auto qi = _mm_packs_epi16(q1, q2);
        auto aux = _mm_maddubs_epi16(m1_8, qi);
        sums = _mm_add_epi32(sums, _mm_madd_epi16(m1_16, aux));
        _mm_storeu_si128((__m128i *)qs, qi);
        qs += 16;
    }
    auto minus = _mm_mul_ps(_mm_loadu_ps(dptr), _mm_cvtepi32_ps(sums));
    _mm_storeu_ps(dptr + 4, minus);
#else
    float aux[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            for (int l = 0; l < 4; ++l) {
                float ax = fabsf(x[j+4*i+l]);
                aux[i] = std::max(aux[i], ax);
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        dptr[i] = aux[i]/127;
        aux[i] = dptr[i] > 0 ? 1/dptr[i] : 0.f;
    }
    int32_t sum[4] = {};
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            for (int l = 0; l < 4; ++l) {
                qs[j+4*i+l] = nearest_int(aux[i]*x[j+4*i+l]);
                sum[i] += qs[j+4*i+l];
            }
        }
    }
    for (int i = 0; i < 4; ++i) dptr[4+i] = dptr[i]*sum[i];
#endif
}

void quantize_row_q8_K64(const float * x, void * y, int64_t k) {
    quantize_row_q8_K64_ref(x, (block_q8_K64 *)y, k);
}

#ifdef __AVX2__
namespace {
inline float hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
inline float hsum_float_8(__m256 x) {
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1))); 
}
inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
inline float hmax_f32_8(__m256 x) {
    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    max4 = _mm_max_ps( max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4));
    return  _mm_cvtss_f32(max4);
}
}
#endif

void quantize_row_q8_K16(const float * x, void * vy, int64_t nk) {
    float * dptr = (float *)vy;
    int8_t * qy = (int8_t *)(dptr + 5);
    int n64 = nk / 64;
#ifdef __AVX2__
    __m256 sign_bit = _mm256_set1_ps(-0.f);
    __m256 vmax[4] = {};
    __m256 vsum[4] = {};
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            auto v1 = _mm256_loadu_ps(x + 64*i64 + 16*k + 0);
            auto v2 = _mm256_loadu_ps(x + 64*i64 + 16*k + 8);
            vsum[k] = _mm256_add_ps(vsum[k], _mm256_add_ps(v1, v2));
            v1 = _mm256_andnot_ps(sign_bit, v1);
            v2 = _mm256_andnot_ps(sign_bit, v2);
            vmax[k] = _mm256_max_ps(vmax[k], _mm256_max_ps(v1, v2));
        }
    }
    __m256 sum = _mm256_add_ps(_mm256_add_ps(vsum[0], vsum[1]), _mm256_add_ps(vsum[2], vsum[3]));
    dptr[4] = hsum_float_8(sum);
    for (int k = 0; k < 4; ++k) {
        float max = hmax_f32_8(vmax[k]);
        dptr[k] = max/127;
        vmax[k] = _mm256_set1_ps(dptr[k] > 0 ? 1/dptr[k] : 0.f);
    }
    __m256i ival[8];
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            __m256 v0 = _mm256_mul_ps(vmax[k], _mm256_loadu_ps(x + 64*i64 + 16*k + 0));
            __m256 v1 = _mm256_mul_ps(vmax[k], _mm256_loadu_ps(x + 64*i64 + 16*k + 8));
            v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
            v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
            ival[2*k+0] = _mm256_cvtps_epi32(v0);
            ival[2*k+1] = _mm256_cvtps_epi32(v1);
        }
        for (int k = 0; k < 2; ++k) {
            auto i0 = _mm256_packs_epi32(ival[4*k+0], ival[4*k+1]);
            auto i1 = _mm256_packs_epi32(ival[4*k+2], ival[4*k+3]);
            i0 = _mm256_packs_epi16(i0, i1);
            i0 = _mm256_permutevar8x32_epi32(i0, perm);
            _mm256_storeu_si256((__m256i *)qy, i0);
            qy += 32;
        }
    }
#elif defined __ARM_NEON
    static const uint8_t k_shuffle[16] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60};
    auto shuffle = vld1q_u8(k_shuffle);
    float32x4_t vmax[4] = {};
    float32x4_t vsum[4] = {};
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            auto v = vld1q_f32_x4(x + 64*i64 + 16*k);
            vsum[k] = vaddq_f32(vsum[k], vaddq_f32(v.val[0], v.val[1]));
            vsum[k] = vaddq_f32(vsum[k], vaddq_f32(v.val[2], v.val[3]));
            vmax[k] = vmaxq_f32(vmax[k], vmaxq_f32(vabsq_f32(v.val[0]), vabsq_f32(v.val[1])));
            vmax[k] = vmaxq_f32(vmax[k], vmaxq_f32(vabsq_f32(v.val[2]), vabsq_f32(v.val[3])));
        }
    }
    dptr[4] = vaddvq_f32(vaddq_f32(vaddq_f32(vsum[0], vsum[1]), vaddq_f32(vsum[2], vsum[3])));
    for (int k = 0; k < 4; ++k) {
        float max = vmaxvq_f32(vmax[k]);
        dptr[k] = max/127;
        vmax[k] = vdupq_n_f32(dptr[k] > 0 ? 1/dptr[k] : 0.f);
    }
    int8x16x4_t q;
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            auto v = vld1q_f32_x4(x + 64*i64 + 16*k);
            for (int j = 0; j < 4; ++j) {
                q.val[j] = vreinterpretq_s8_s32(vcvtnq_s32_f32(vmulq_f32(vmax[k], v.val[j])));
            }
            auto qi = vqtbl4q_s8(q, shuffle);
            vst1q_s8(qy, qi);
            qy += 16;
        }
    }
#else
    float amax[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            for (int j = 0; j < 16; ++j) {
                float ax = std::abs(x[64*i64 + 16*k + j]);
                amax[k] = std::max(amax[k], ax);
            }
        }
    }
    for (int k = 0; k < 4; ++k) {
        dptr[k] = amax[k]/127;
        amax[k] = dptr[k] > 0 ? 1/dptr[k] : 0.f;
    }
    double sumf = 0;
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            for (int j = 0; j < 16; ++j) {
                sumf += x[64*i64 + 16*k + j];
                qy[64*i64 + 16*k + j] = nearest_int(amax[k]*x[64*i64 + 16*k + j]);
            }
        }
    }
    dptr[4] = sumf;
#endif
}

//
// ============================================== iq2_K
//

namespace {

inline int best_index_iq2nl(const int8_t * values, float x) {
    int idx = x < values[1] ? 0 : x > values[2] ? 2 : 1;
    return x - values[idx] < values[idx+1] - x ? idx : idx + 1;
}

void quantize_row_iq2_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights) {

    constexpr int kBlockSize = 16;

    block_iq2_k * y = (block_iq2_k *)vy;

    float scales[QK_K/kBlockSize];
    float weight[kBlockSize];
    float sumx[kBlockSize+1], sumw[kBlockSize+1];
    float sw[QK_K/kBlockSize];
    int8_t Ls[QK_K/kBlockSize];

    std::array<std::pair<float,int>, kBlockSize> pairs;

    const int8_t * shifted_values = iq2nl_values + 4;

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            sw[ib] = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                sw[ib] += weight[j];
                pairs[j] = {xb[j], j};
            }
            std::sort(pairs.begin(), pairs.end());
            sumx[0] = sumw[0] = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                int jj = pairs[j].second;
                sumw[j+1] = sumw[j] + weight[jj];
                sumx[j+1] = sumx[j] + weight[jj]*xb[jj];
            }
            float best = 0, d = 0;
            bool is_shifted = false;
            float sumqx, sumq2;
            for (int i1 = 0; i1 < kBlockSize; ++i1) {
                for (int i2 = i1; i2 < kBlockSize; ++i2) {
                    for (int i3 = i2; i3 < kBlockSize; ++i3) {
                        sumqx = (sumx[i1] - sumx[ 0])*iq2nl_values[0] + (sumx[i2] - sumx[i1])*iq2nl_values[1]
                              + (sumx[i3] - sumx[i2])*iq2nl_values[2] + (sumx[kBlockSize] - sumx[i3])*iq2nl_values[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*iq2nl_values[0]*iq2nl_values[0] + (sumw[i2] - sumw[i1])*iq2nl_values[1]*iq2nl_values[1]
                              + (sumw[i3] - sumw[i2])*iq2nl_values[2]*iq2nl_values[2] + (sumw[kBlockSize] - sumw[i3])*iq2nl_values[3]*iq2nl_values[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*shifted_values[0] + (sumx[i2] - sumx[i1])*shifted_values[1]
                              + (sumx[i3] - sumx[i2])*shifted_values[2] + (sumx[kBlockSize] - sumx[i3])*shifted_values[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*shifted_values[0]*shifted_values[0] + (sumw[i2] - sumw[i1])*shifted_values[1]*shifted_values[1]
                              + (sumw[i3] - sumw[i2])*shifted_values[2]*shifted_values[2] + (sumw[kBlockSize] - sumw[i3])*shifted_values[3]*shifted_values[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*iq2nl_values[3] + (sumx[i2] - sumx[i1])*iq2nl_values[2]
                              + (sumx[i3] - sumx[i2])*iq2nl_values[1] + (sumx[kBlockSize] - sumx[i3])*iq2nl_values[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*iq2nl_values[3]*iq2nl_values[3] + (sumw[i2] - sumw[i1])*iq2nl_values[2]*iq2nl_values[2]
                              + (sumw[i3] - sumw[i2])*iq2nl_values[1]*iq2nl_values[1] + (sumw[kBlockSize] - sumw[i3])*iq2nl_values[0]*iq2nl_values[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*shifted_values[3] + (sumx[i2] - sumx[i1])*shifted_values[2]
                              + (sumx[i3] - sumx[i2])*shifted_values[1] + (sumx[kBlockSize] - sumx[i3])*shifted_values[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*shifted_values[3]*shifted_values[3] + (sumw[i2] - sumw[i1])*shifted_values[2]*shifted_values[2]
                              + (sumw[i3] - sumw[i2])*shifted_values[1]*shifted_values[1] + (sumw[kBlockSize] - sumw[i3])*shifted_values[0]*shifted_values[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = std::max(max_abs_scale, abs_scale);
        }

        if (!max_abs_scale) continue;
        float d = make_qx_quants(QK_K/kBlockSize, 8, scales, Ls, sw);
        if (!d) continue;

        //float d = -max_scale/8;
        y[ibl].extra = extra;
        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            int ls = nearest_int(id*scales[ib]);
            ls = std::max(-8, std::min(7, ls));
            y[ibl].scales[ib/2] |= ((ls + 8) << 4*(ib%2));
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
                const float * xb = xbl + kBlockSize*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/4) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq2nl(block_values, al);
                    qs[j] |= (ibest << 2*(ib32%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        y[ibl].d = GGML_FP32_TO_FP16(1.030f*(sumq2 > 0 ? sumqx/sumq2 : d));

    }
}
}

void quantize_row_iq2_k_ref(const float * x, block_iq2_k  * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_k(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq2_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_k * y = (block_iq2_k *)vy;
    quantize_row_iq2_k_ref(x, y, k);
}

size_t quantize_iq2_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq2_k_impl(src, (void *)qrow, n_per_row, imatrix);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq2_k);
    }
    return nrows * nblock * sizeof(block_iq2_k);
}

void dequantize_row_iq2_k(const block_iq2_k  * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            float dl1 = d * ((x[i].scales[ib32] & 0xf) - 8);
            float dl2 = d * ((x[i].scales[ib32] >>  4) - 8);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j+ 0] >> shift) & 3];
                y[j+16] = dl2 * values2[(qs[j+16] >> shift) & 3];
            }
            y += 32;
            shift += 2;
            if (shift == 8) { qs += 32; shift = 0; }
        }

    }

}

void vec_dot_iq2_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_K, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    GGML_ABORT("not implemented");

}

namespace {
void quantize_row_iq2_ks_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_sw, int8_t * all_Ls) {

    constexpr int kBlockSize = 32;
    constexpr int kMax_i1 = 3*kBlockSize/4;
    constexpr int kMin_i3 = kBlockSize/4;
    //constexpr int kNtry = 5;
    //constexpr float kStep = 1.f;

    ggml_half * dptr = (ggml_half *)vy;
    *dptr = GGML_FP32_TO_FP16(0.f);

    block_iq2_ks * y = (block_iq2_ks *)(dptr + 1);

    float weight[kBlockSize];
    float sumx[kBlockSize+1], sumw[kBlockSize+1];

    std::array<std::pair<float,int>, kBlockSize> pairs;

    float val [4] = {float(iq2nl_values[0]), float(iq2nl_values[1]), float(iq2nl_values[2]), float(iq2nl_values[3])};
    float sval[4] = {float(iq2nl_values[4]), float(iq2nl_values[5]), float(iq2nl_values[6]), float(iq2nl_values[7])};

    const int8_t * shifted_values = iq2nl_values + 4;

    const int nblock = n_per_row/QK_K;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_ks));

        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto sw = all_sw + ibl*(QK_K/kBlockSize);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            sw[ib] = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                sw[ib] += weight[j];
                pairs[j] = {xb[j], j};
            }
            //float amax = 0, max = 0;
            //for (int j = 0; j < kBlockSize; ++j) {
            //    float ax = fabsf(xb[j]);
            //    if (ax > amax) {
            //        amax = ax; max = xb[j];
            //    }
            //}
            //if (!amax) {
            //    scales[ib] = 0;
            //    continue;
            //}
            //float d = kNtry > 0 ? -max/iq2nl_values[0] : max/iq2nl_values[0];
            //float id = 1/d;
            //float sumqx_p = 0, sumq2_p = 0;
            //float sumqx_m = 0, sumq2_m = 0;
            //for (int j = 0; j < kBlockSize; ++j) {
            //    float w = weight[j];
            //    float al = id*xb[j];
            //    int l = best_index_iq2nl(iq2nl_values, al);
            //    float q = iq2nl_values[l];
            //    sumqx_p += w*q*xb[j];
            //    sumq2_p += w*q*q;
            //    l = best_index_iq2nl(iq2nl_values, -al);
            //    q = iq2nl_values[l];
            //    sumqx_m += w*q*xb[j];
            //    sumq2_m += w*q*q;
            //}
            //d = sumqx_p/sumq2_p;
            //float best = d*sumqx_p;
            //if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
            //    d = sumqx_m/sumq2_m; best = d*sumqx_m;
            //}
            //bool is_shifted = false;
            //for (int itry = -kNtry; itry <= kNtry; ++itry) {
            //    id = (kStep*itry + iq2nl_values[0])/max;
            //    sumqx_p = sumq2_p = 0;
            //    sumqx_m = sumq2_m = 0;
            //    for (int j = 0; j < kBlockSize; ++j) {
            //        float w = weight[j];
            //        float al = id*xb[j];
            //        int l = best_index_iq2nl(iq2nl_values, al);
            //        float q = iq2nl_values[l];
            //        sumqx_p += w*q*xb[j];
            //        sumq2_p += w*q*q;
            //        l = best_index_iq2nl(iq2nl_values, -al);
            //        q = iq2nl_values[l];
            //        sumqx_m += w*q*xb[j];
            //        sumq2_m += w*q*q;
            //    }
            //    if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
            //        d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
            //    }
            //    if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
            //        d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
            //    }
            //    id = (kStep*itry + shifted_values[0])/max;
            //    sumqx_p = sumq2_p = 0;
            //    sumqx_m = sumq2_m = 0;
            //    for (int j = 0; j < kBlockSize; ++j) {
            //        float w = weight[j];
            //        float al = id*xb[j];
            //        int l = best_index_iq2nl(shifted_values, al);
            //        float q = shifted_values[l];
            //        sumqx_p += w*q*xb[j];
            //        sumq2_p += w*q*q;
            //        l = best_index_iq2nl(shifted_values, -al);
            //        q = shifted_values[l];
            //        sumqx_m += w*q*xb[j];
            //        sumq2_m += w*q*q;
            //    }
            //    if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
            //        d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
            //    }
            //    if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
            //        d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
            //    }
            //}
            std::sort(pairs.begin(), pairs.end());
            sumx[0] = sumw[0] = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                int jj = pairs[j].second;
                sumw[j+1] = sumw[j] + weight[jj];
                sumx[j+1] = sumx[j] + weight[jj]*xb[jj];
            }
            float best = 0, d = 0;
            bool is_shifted = false;
            float sumqx, sumq2;
            for (int i1 = 0; i1 < kMax_i1; ++i1) {
                for (int i2 = i1; i2 < kBlockSize; ++i2) {
                    for (int i3 = std::max(i2, kMin_i3); i3 < kBlockSize; ++i3) {
                        sumqx = (sumx[i1] - sumx[ 0])*val[0] + (sumx[i2] - sumx[i1])*val[1]
                              + (sumx[i3] - sumx[i2])*val[2] + (sumx[kBlockSize] - sumx[i3])*val[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*val[0]*val[0] + (sumw[i2] - sumw[i1])*val[1]*val[1]
                              + (sumw[i3] - sumw[i2])*val[2]*val[2] + (sumw[kBlockSize] - sumw[i3])*val[3]*val[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*sval[0] + (sumx[i2] - sumx[i1])*sval[1]
                              + (sumx[i3] - sumx[i2])*sval[2] + (sumx[kBlockSize] - sumx[i3])*sval[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*sval[0]*sval[0] + (sumw[i2] - sumw[i1])*sval[1]*sval[1]
                              + (sumw[i3] - sumw[i2])*sval[2]*sval[2] + (sumw[kBlockSize] - sumw[i3])*sval[3]*sval[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*val[3] + (sumx[i2        ] - sumx[i1])*val[2]
                              + (sumx[i3] - sumx[i2])*val[1] + (sumx[kBlockSize] - sumx[i3])*val[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*val[3]*val[3] + (sumw[i2        ] - sumw[i1])*val[2]*val[2]
                              + (sumw[i3] - sumw[i2])*val[1]*val[1] + (sumw[kBlockSize] - sumw[i3])*val[0]*val[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*sval[3] + (sumx[i2        ] - sumx[i1])*sval[2]
                              + (sumx[i3] - sumx[i2])*sval[1] + (sumx[kBlockSize] - sumx[i3])*sval[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*sval[3]*sval[3] + (sumw[i2        ] - sumw[i1])*sval[2]*sval[2]
                              + (sumw[i3] - sumw[i2])*sval[1]*sval[1] + (sumw[kBlockSize] - sumw[i3])*sval[0]*sval[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);

        }
        y[ibl].extra = extra;

    }

    float d = make_qx_quants(nblock*(QK_K/kBlockSize), 16, all_scales, all_Ls, all_sw);

    if (!d) return;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;
        auto Ls = all_Ls + ibl*(QK_K/kBlockSize);
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            int ls = Ls[ib];
            y[ibl].scales[ib/2] |= ((ls & 0xf) << 4*(ib%2));
            y[ibl].extra |= ((ls >> 4) << (8 + ib));
            ls -= 16;
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
                const float * xb = xbl + kBlockSize*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                uint8_t * qs = y[ibl].qs + 32*(ib/4);
                for (int j = 0; j < 32; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq2nl(block_values, al);
                    qs[j] |= (ibest << 2*(ib%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
    }
    *dptr = GGML_FP32_TO_FP16(1.030f*(sumq2 > 0 ? sumqx/sumq2 : d));
}
}

void quantize_row_iq2_ks_ref(const float * x, block_iq2_ks * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_ks(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq2_ks(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_ks * y = (block_iq2_ks *)vy;
    quantize_row_iq2_ks_ref(x, y, k);
}

size_t quantize_iq2_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_KS, n_per_row);
    int nblock = n_per_row/QK_K;
    std::vector<float> all_scales(nblock*(QK_K/kBlockSize)), all_sw(nblock*(QK_K/kBlockSize));
    std::vector<int8_t> all_Ls(nblock*(QK_K/kBlockSize));
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq2_ks_impl(src, (void *)qrow, n_per_row, imatrix, all_scales.data(), all_sw.data(), all_Ls.data());
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}

void dequantize_row_iq2_ks(const block_iq2_ks  * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const ggml_half * dptr = (const ggml_half *)x;
    const float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq2_ks *)(dptr + 1);

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            float dl1 = d * (((x[i].scales[ib64] & 0xf) | ((extra >> 4) & 0x10)) - 16);
            float dl2 = d * (((x[i].scales[ib64] >>  4) | ((extra >> 5) & 0x10)) - 16);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            for (int j = 0; j < 32; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j] >> (shift+0)) & 3];
                y[j+32] = dl2 * values2[(qs[j] >> (shift+2)) & 3];
            }
            y += 64;
            shift += 4;
            if (shift == 8) { qs += 32; shift = 0; }
        }

    }

}

void vec_dot_iq2_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_KS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    const ggml_half * dptr = (const ggml_half *)vx;
    const float d = GGML_FP16_TO_FP32(*dptr);
    const block_iq2_ks * x = (const block_iq2_ks *)(dptr + 1);
    const block_q8_K   * y = (const block_q8_K *)vy;

    const int nb = n / QK_K;
    float sumf = 0;
    for (int i = 0; i < nb; i++) {
        const uint8_t * qs = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        uint16_t extra = x[i].extra;
        int sumi = 0;
        for (int ib128 = 0; ib128 < QK_K/128; ++ib128) {
            int d1 = (((x[i].scales[2*ib128+0] & 0xf) | ((extra >> 4) & 0x10)) - 16);
            int d2 = (((x[i].scales[2*ib128+0] >>  4) | ((extra >> 5) & 0x10)) - 16);
            int d3 = (((x[i].scales[2*ib128+1] & 0xf) | ((extra >> 6) & 0x10)) - 16);
            int d4 = (((x[i].scales[2*ib128+1] >>  4) | ((extra >> 7) & 0x10)) - 16);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values3 = extra & 4 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values4 = extra & 8 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 4;
            int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
            for (int j = 0; j < 32; ++j) {
                sumi1 += q8[j+ 0] * values1[(qs[j] >> 0) & 3];
                sumi2 += q8[j+32] * values2[(qs[j] >> 2) & 3];
                sumi3 += q8[j+64] * values3[(qs[j] >> 4) & 3];
                sumi4 += q8[j+96] * values4[(qs[j] >> 6) & 3];
            }
            sumi += d1*sumi1 + d2*sumi2 + d3*sumi3 + d4*sumi4;
            q8 += 128;
            qs +=  32;
        }
        sumf += y[i].d * sumi;
    }

    *s = d * sumf;

}

//
// ============================================== iq3_k
//
namespace {
const int8_t iq3nl_index[111] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  8,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  9,
  9,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 10, 10,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 11, 11,  4,  4,  4,  4,
  4,  4,  4,  4,  4,  4, 12,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 13, 13,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
  6,  6,  6,  6, 14, 14,  7,  7,  7,  7,  7,  7,  7,  7, 7
};
inline int best_index_iq3nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 111) return ix < 0 ? 0 : 7;
    ix = iq3nl_index[ix];
    return ix < 8 ? ix : x - values[ix-8] < values[ix-7] - x ? ix-8 : ix-7;
}

static void quantize_row_iq3_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights) {

    const int ntry = 5;

    block_iq3_k * y = (block_iq3_k *)vy;

    float scales[QK_K/16];
    float weight[16];

    const int8_t * shifted_values = iq3nl_values + 8;

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (!amax) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/iq3nl_values[0] : max/iq3nl_values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(iq3nl_values, al);
                float q = iq3nl_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(iq3nl_values, -al);
                q = iq3nl_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + iq3nl_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(iq3nl_values, al);
                    float q = iq3nl_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(iq3nl_values, -al);
                    q = iq3nl_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (d) {
                const int8_t * block_values = is_shifted ? shifted_values : iq3nl_values;
                float sumqx = 0, sumq2 = 0;
                id = 1/d;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(block_values, al);
                    float q = block_values[l];
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) d = sumqx/sumq2;
            }
            scales[ib] = d;

            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = MAX(max_abs_scale, abs_scale);
        }

        if (!max_abs_scale) continue;

        float d = max_abs_scale/31;
        y[ibl].extra = extra;
        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = nearest_int(0.5f*(id*fabsf(scales[ib])-1));
            ls = MAX(0, MIN(15, ls));
            y[ibl].scales_l[ib/2] |= (ls << 4*(ib%2));
            if (scales[ib] < 0) y[ibl].scales_h |= (1 << ib);
            ls = (2*ls + 1) * (scales[ib] < 0 ? -1 : 1);
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq3nl_values;
                const float * xb = xbl + 16*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*16;
                    for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/4) + offset;
                uint8_t * qh = y[ibl].qh + 32*(ib32/8) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq3nl(block_values, al);
                    qs[j] |= ((ibest &  3) << 2*(ib32%4));
                    qh[j] |= ((ibest >> 2) << (ib32%8));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        y[ibl].d = GGML_FP32_TO_FP16(1.01f*(sumq2 > 0 ? sumqx/sumq2 : d));

    }
}

}

void quantize_row_iq3_k_ref(const float * x, block_iq3_k * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq3_k(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq3_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq3_k * y = (block_iq3_k *)vy;
    quantize_row_iq3_k_ref(x, y, k);
}

size_t quantize_iq3_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq3_k_impl(src, (void *)qrow, n_per_row, imatrix);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq3_k);
    }
    return nrows * nblock * sizeof(block_iq3_k);
}

void dequantize_row_iq3_k(const block_iq3_k * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;

        uint16_t sh = x[i].scales_h;
        uint16_t extra = x[i].extra;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            float dl1 = d * ((2*(x[i].scales_l[ib32] & 0xf) + 1) * ((sh & 1) ? -1 : 1));
            float dl2 = d * ((2*(x[i].scales_l[ib32] >>  4) + 1) * ((sh & 2) ? -1 : 1));
            sh >>= 2;
            const int8_t * values1 = extra & 1 ? iq3nl_values + 8 : iq3nl_values;
            const int8_t * values2 = extra & 2 ? iq3nl_values + 8 : iq3nl_values;
            extra >>= 2;
            int shift_l = 2*(ib32%4);
            int shift_h = ib32%8;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[((qs[j+ 0] >> shift_l) & 3) | (((qh[j+ 0] >> shift_h) & 1) << 2)];
                y[j+16] = dl2 * values2[((qs[j+16] >> shift_l) & 3) | (((qh[j+16] >> shift_h) & 1) << 2)];
            }
            y += 32;
            if (shift_l == 6) qs += 32;
        }

    }
}

void vec_dot_iq3_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_K, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    GGML_ABORT("not implemented");
}

//
// ============================================== iq4_K
//
void dequantize_row_iq4_k(const block_iq4_k * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);

        uint16_t extra = x[i].extra;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const uint8_t sh = x[i].scales_h[ib/2] >> 4*(ib%2);
            const float dl1 = d * (((x[i].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const float dl2 = d * (((x[i].scales_l[ib] >>  4) | ((sh << 2) & 0x30)) - 32);
            const int8_t * values1 = extra & 1 ? iq4k_values + 16 : iq4k_values;
            const int8_t * values2 = extra & 2 ? iq4k_values + 16 : iq4k_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[qs[j] & 0xf];
                y[j+16] = dl2 * values2[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

void vec_dot_iq4_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_K, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    const int nb = n / QK_K;

    const block_iq4_k * x = (const block_iq4_k *)vx;
    const block_q8_K  * y = (const block_q8_K *)vy;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t extra = x[ibl].extra;
        uint32_t h = *((const uint32_t *)x[ibl].scales_h);
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        int32_t sum = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls1 = ((x[ibl].scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
            const int ls2 = ((x[ibl].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;
            h >>= 4;
            const int8_t * values1 = iq4k_values + 16*(extra & 1);
            const int8_t * values2 = iq4k_values +  8*(extra & 2);
            extra >>= 2;
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * values1[qs[j] & 0xf];
                sumi2 += q8[j+16] * values2[qs[j] >>  4];
            }
            sum += ls1*sumi1 + ls2*sumi2;
            qs += 16;
            q8 += 32;
        }
        sumf += d4d8 * sum;
    }
    *s = sumf;

}

namespace {
const int8_t iq4nl_index[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};
inline int best_index_iq4nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = iq4nl_index[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}

static void quantize_row_iq4_k_impl_bs16(const int super_block_size, const int block_size, const float * x,
        block_iq4_k * y,
        float * scales, float * weight, uint8_t * L,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    GGML_ASSERT(super_block_size == 256 && block_size == 16);

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/super_block_size;

    memset(y, 0, sizeof(block_iq4_k));
    y->d = GGML_FP32_TO_FP16(0.f);

    uint16_t * scales_h = (uint16_t *)y->scales_h;

    const int8_t * shifted_values = values + 16;

    float max_scale = 0, amax_scale = 0;
    uint16_t extra = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        if (!amax) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max/values[0] : max/values[0];
        float id = 1/d;
        float sumqx_p = 0, sumq2_p = 0;
        float sumqx_m = 0, sumq2_m = 0;
        for (int j = 0; j < block_size; ++j) {
            float w = weight[j];
            float al = id*xb[j];
            int l = best_index_iq4nl(values, al);
            float q = values[l];
            sumqx_p += w*q*xb[j];
            sumq2_p += w*q*q;
            l = best_index_iq4nl(values, -al);
            q = values[l];
            sumqx_m += w*q*xb[j];
            sumq2_m += w*q*q;
        }
        d = sumqx_p/sumq2_p;
        bool is_shifted = false;
        float best = d*sumqx_p;
        if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
            d = sumqx_m/sumq2_m; best = d*sumqx_m;
        }
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0])/max;
            sumqx_p = sumq2_p = 0;
            sumqx_m = sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
            }
            id = (itry + shifted_values[0])/max;
            sumqx_p = sumq2_p = 0;
            sumqx_m = sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(shifted_values, al);
                float q = shifted_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(shifted_values, -al);
                q = shifted_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
            }
        }
        if (is_shifted) extra |= (1 << ib);
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }
    float d = -max_scale/32;
    y->d = GGML_FP32_TO_FP16(d);
    y->extra = extra;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const int8_t * block_values = extra & (1 << ib) ? shifted_values : values;
        int l = nearest_int(id*scales[ib]);
        l = MAX(-32, MIN(31, l));
        float dl = d * l;
        float idl = dl ? 1/dl : 0.f;
        uint8_t * Lb = L + ib*block_size;
        const float * xb = x + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        for (int j = 0; j < block_size; ++j) {
            Lb[j] = best_index_iq4nl(block_values, idl*xb[j]);
            float w = weight[j];
            float q = block_values[Lb[j]]*l;
            sumqx += w*q*xb[j];
            sumq2 += w*q*q;
        }
        l += 32;
        uint8_t l_l = l & 0xf;
        uint8_t l_h = l >>  4;
        if (ib%2 == 0) y->scales_l[ib/2] = l_l;
        else y->scales_l[ib/2] |= (l_l << 4);
        scales_h[ib/8] |= (l_h << 2*(ib%8));
    }
    if (sumq2 > 0) y->d = GGML_FP32_TO_FP16(sumqx/sumq2);

    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            y->qs[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

}

void quantize_row_iq4_k_ref(const float * x, block_iq4_k * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_k(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq4_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq4_k * y = (block_iq4_k *)vy;
    quantize_row_iq4_k_ref(x, y, k);
}

size_t quantize_iq4_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    uint8_t L[QK_K];
    float weight[16];
    float scales[QK_K/16];
    for (int64_t row = 0; row < nrows; ++row) {
        block_iq4_k * iq4 = (block_iq4_k *)qrow;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw = imatrix ? imatrix + QK_K*ibl : NULL;
            quantize_row_iq4_k_impl_bs16(QK_K, 16, src + QK_K*ibl, iq4 + ibl,
                    scales, weight, L, iq4k_values, qw, 7);
        }
        src += n_per_row;
        qrow += nblock*sizeof(block_iq4_k);
    }
    return nrows * nblock * sizeof(block_iq4_k);
}

//
// ============================================== iq5_K
//
void dequantize_row_iq5_k(const block_iq5_k * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * sl = x[i].scales_l;
        const uint8_t * sh = x[i].scales_h;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {

            float dl1 = d * (((sl[2*ib64+0] & 0xf) | ((sh[ib64] << 4) & 0x30)) - 32);
            float dl2 = d * (((sl[2*ib64+0] >>  4) | ((sh[ib64] << 2) & 0x30)) - 32);
            float dl3 = d * (((sl[2*ib64+1] & 0xf) | ((sh[ib64] >> 0) & 0x30)) - 32);
            float dl4 = d * (((sl[2*ib64+1] >>  4) | ((sh[ib64] >> 2) & 0x30)) - 32);
            const int8_t * values1 = iq5nl_values + ((extra & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((extra & 2) << 4);
            const int8_t * values3 = iq5nl_values + ((extra & 4) << 3);
            const int8_t * values4 = iq5nl_values + ((extra & 8) << 2);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4)];
                y[j+16] = dl2 * values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4)];
                y[j+32] = dl3 * values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3)];
                y[j+48] = dl4 * values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3)];
            }
            y  += 64;
            qs += 32;
            extra >>= 4;
            shift += 2;
            if (shift == 8) { qh += 32; shift = 0; }
        }

    }
}

void vec_dot_iq5_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ5_K, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    const int nb = n / QK_K;

    const block_iq5_k * x = (const block_iq5_k *)vx;
    const block_q8_K  * y = (const block_q8_K  *)vy;

    float sumf = 0;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * sl = x[i].scales_l;
        const uint8_t * sh = x[i].scales_h;
        const int8_t  * q8 = y[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        int sumb  = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {

            int dl1 = (((sl[2*ib64+0] & 0xf) | ((sh[ib64] << 4) & 0x30)) - 32);
            int dl2 = (((sl[2*ib64+0] >>  4) | ((sh[ib64] << 2) & 0x30)) - 32);
            int dl3 = (((sl[2*ib64+1] & 0xf) | ((sh[ib64] >> 0) & 0x30)) - 32);
            int dl4 = (((sl[2*ib64+1] >>  4) | ((sh[ib64] >> 2) & 0x30)) - 32);
            const int8_t * values1 = iq5nl_values + ((extra & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((extra & 2) << 4);
            const int8_t * values3 = iq5nl_values + ((extra & 4) << 3);
            const int8_t * values4 = iq5nl_values + ((extra & 8) << 2);
            int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4)];
                sumi2 += q8[j+16] * values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4)];
                sumi3 += q8[j+32] * values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3)];
                sumi4 += q8[j+48] * values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3)];
            }
            sumb += dl1 * sumi1 + dl2 * sumi2 + dl3 * sumi3 + dl4 * sumi4;
            q8 += 64;
            qs += 32;
            extra >>= 4;
            shift += 2;
        }
        sumf += d * sumb;

    }

    *s = sumf;

}

namespace {
const int8_t iq5nl_index[248] = {
     0,  0,  0,  0,  0,  0, 32,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 33, 33,  2,  2,  2,  2,  2,  2,  2,  2,  2, 34, 34,  3,  3,
     3,  3,  3,  3,  3,  3, 35, 35,  4,  4,  4,  4,  4,  4,  4, 36, 36,  5,  5,  5,  5,  5,  5,  5, 37, 37,  6,  6,  6,  6,  6,  6,
     6, 38,  7,  7,  7,  7,  7,  7, 39, 39,  8,  8,  8,  8,  8, 40, 40,  9,  9,  9,  9,  9, 41, 41, 10, 10, 10, 10, 10, 42, 11, 11,
    11, 11, 11, 43, 12, 12, 12, 12, 12, 44, 13, 13, 13, 13, 13, 45, 14, 14, 14, 14, 14, 46, 15, 15, 15, 15, 47, 47, 16, 16, 16, 16,
    48, 17, 17, 17, 17, 17, 49, 18, 18, 18, 18, 18, 50, 19, 19, 19, 19, 19, 51, 20, 20, 20, 20, 20, 52, 21, 21, 21, 21, 21, 53, 53,
    22, 22, 22, 22, 22, 54, 54, 23, 23, 23, 23, 23, 23, 55, 24, 24, 24, 24, 24, 24, 24, 56, 25, 25, 25, 25, 25, 25, 25, 57, 57, 26,
    26, 26, 26, 26, 26, 26, 58, 58, 27, 27, 27, 27, 27, 27, 27, 27, 59, 28, 28, 28, 28, 28, 28, 28, 28, 28, 60, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 61, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 62, 31, 31, 31, 31, 31, 31
};
inline int best_index_iq5nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 247) return ix < 0 ? 0 : 31;
    ix = iq5nl_index[ix];
    return ix < 32 ? ix : x - values[ix-32] < values[ix-31] - x ? ix-32 : ix-31;
}

void quantize_row_iq5_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights) {
    const int ntry = 5;
    const float step = 1.f;

    block_iq5_k * y = (block_iq5_k *)vy;

    float scales[QK_K/16];
    float weight[16];

    const int8_t * shifted_values = iq5nl_values + 32;

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq5_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 2*sumx2/QK_K;

        float max_scale = 0, max_abs_scale = 0;
        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (!amax) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/iq5nl_values[0] : max/iq5nl_values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq5nl(iq5nl_values, al);
                float q = iq5nl_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq5nl(iq5nl_values, -al);
                q = iq5nl_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry*step + iq5nl_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(iq5nl_values, al);
                    float q = iq5nl_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq5nl(iq5nl_values, -al);
                    q = iq5nl_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry*step + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq5nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (d) {
                const int8_t * block_values = is_shifted ? shifted_values : iq5nl_values;
                float sumqx = 0, sumq2 = 0;
                id = 1/d;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(block_values, al);
                    float q = block_values[l];
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) d = sumqx/sumq2;
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale; max_scale = scales[ib];
            }

        }

        if (!max_abs_scale) continue;
        float d = -max_scale/32;
        y[ibl].d = GGML_FP32_TO_FP16(d);
        y[ibl].extra = extra;

        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = nearest_int(id*scales[ib]);
            ls = MAX(-32, MIN(31, ls));
            int uls = ls + 32;
            y[ibl].scales_l[ib/2] |= ((uls & 0xf) << 4*(ib%2));
            y[ibl].scales_h[ib/4] |= ((uls >>  4) << 2*(ib%4));
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq5nl_values;
                const float * xb = xbl + 16*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*16;
                    for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/2) + offset;
                uint8_t * qh = y[ibl].qh + 32*(ib32/8) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq5nl(block_values, al);
                    qs[j] |= ((ibest & 0xf) << 4*(ib32%2));
                    qh[j] |= ((ibest >>  4) << (ib32%8));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        if (sumq2 > 0) y[ibl].d = GGML_FP32_TO_FP16(sumqx/sumq2);

    }

}

}

void quantize_row_iq5_k_ref(const float * x, block_iq5_k * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq5_k(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq5_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq5_k * y = (block_iq5_k *)vy;
    quantize_row_iq5_k_ref(x, y, k);
}

size_t quantize_iq5_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq5_k_impl(src, (void *)qrow, n_per_row, imatrix);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq5_k);
    }
    return nrows * nblock * sizeof(block_iq5_k);
}

//
// ============================================== iq6_K
//
#define A_IQ6K -127.f
#define B_IQ6K 6.2568f
#define C_IQ6K 0.11218f
#define D_IQ6K 0.0011972f
#define S_IQ6K 1.f

void dequantize_row_iq6_k(const block_iq6_k * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sl = x[i].scales;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {

            float dl1 = d * sl[4*ib64 + 0];
            float dl2 = d * sl[4*ib64 + 1];
            float dl3 = d * sl[4*ib64 + 2];
            float dl4 = d * sl[4*ib64 + 3];
            float m1 = extra & 1 ? S_IQ6K : 0;
            float m2 = extra & 2 ? S_IQ6K : 0;
            float m3 = extra & 4 ? S_IQ6K : 0;
            float m4 = extra & 8 ? S_IQ6K : 0;
            for (int j = 0; j < 16; ++j) {
                float q1 = ((qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 0x03) << 4));
                float q2 = ((qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 0x03) << 4));
                float q3 = ((qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 0x0c) << 2));
                float q4 = ((qs[j+16] >>  4) | (((qh[j+16] >> shift) & 0x0c) << 2));
                y[j+ 0] = dl1 * (A_IQ6K + q1*(B_IQ6K + q1*(-C_IQ6K + q1*D_IQ6K)) + m1);
                y[j+16] = dl2 * (A_IQ6K + q2*(B_IQ6K + q2*(-C_IQ6K + q2*D_IQ6K)) + m2);
                y[j+32] = dl3 * (A_IQ6K + q3*(B_IQ6K + q3*(-C_IQ6K + q3*D_IQ6K)) + m3);
                y[j+48] = dl4 * (A_IQ6K + q4*(B_IQ6K + q4*(-C_IQ6K + q4*D_IQ6K)) + m4);
            }
            y  += 64;
            qs += 32;
            extra >>= 4;
            shift += 4;
            if (shift == 8) { qh += 32; shift = 0; }
        }

    }
}

void vec_dot_iq6_k_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ6_K, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

    GGML_ABORT("not implemented");

    // TODO
    //const int nb = n / QK_K;

    //const block_iq5_k * x = (const block_iq5_k *)vx;
    //const block_q8_K  * y = (const block_q8_K  *)vy;

    //float sumf = 0;

    //for (int i = 0; i < nb; i++) {

    //    const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
    //    const uint8_t * qs = x[i].qs;
    //    const uint8_t * qh = x[i].qh;
    //    const uint8_t * sl = x[i].scales_l;
    //    const uint8_t * sh = x[i].scales_h;
    //    const int8_t  * q8 = y[i].qs;

    //    uint16_t extra = x[i].extra;

    //    int shift = 0;
    //    int sumb  = 0;
    //    for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {

    //        int dl1 = (((sl[2*ib64+0] & 0xf) | ((sh[ib64] << 4) & 0x30)) - 32);
    //        int dl2 = (((sl[2*ib64+0] >>  4) | ((sh[ib64] << 2) & 0x30)) - 32);
    //        int dl3 = (((sl[2*ib64+1] & 0xf) | ((sh[ib64] >> 0) & 0x30)) - 32);
    //        int dl4 = (((sl[2*ib64+1] >>  4) | ((sh[ib64] >> 2) & 0x30)) - 32);
    //        const int8_t * values1 = iq5nl_values + ((extra & 1) << 5);
    //        const int8_t * values2 = iq5nl_values + ((extra & 2) << 4);
    //        const int8_t * values3 = iq5nl_values + ((extra & 4) << 3);
    //        const int8_t * values4 = iq5nl_values + ((extra & 8) << 2);
    //        int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
    //        for (int j = 0; j < 16; ++j) {
    //            sumi1 += q8[j+ 0] * values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4)];
    //            sumi2 += q8[j+16] * values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4)];
    //            sumi3 += q8[j+32] * values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3)];
    //            sumi4 += q8[j+48] * values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3)];
    //        }
    //        sumb += dl1 * sumi1 + dl2 * sumi2 + dl3 * sumi3 + dl4 * sumi4;
    //        q8 += 64;
    //        qs += 32;
    //        extra >>= 4;
    //        shift += 2;
    //    }
    //    sumf += d * sumb;

    //}

    //*s = sumf;

}

namespace {

inline int best_index(int n, const float * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}
uint8_t iq6nl_index[249] = {
   0,   0,   0,  64,   1,   1,   1,   1,   1,  65,   2,   2,   2,   2,   2,  66,   3,   3,   3,   3,  67,  67,   4,   4,   4,   4,  68,   5,   5,   5,   5,  69,
  69,   6,   6,   6,  70,  70,   7,   7,   7,  71,   8,   8,   8,  72,  72,   9,   9,   9,  73,  73,  10,  10,  10,  74,  11,  11,  11,  75,  12,  12,  12,  76,
  13,  13,  13,  77,  14,  14,  14,  78,  15,  15,  79,  79,  16,  16,  80,  17,  17,  81,  81,  18,  18,  82,  19,  19,  83,  83,  20,  84,  84,  21,  85,  85,
  22,  86,  86,  23,  87,  87,  24,  88,  88,  25,  89,  89,  26,  90,  90,  27,  91,  91,  28,  92,  29,  93,  93,  30,  94,  94,  31,  95,  95,  32,  96,  33,
  97,  97,  34,  98,  98,  35,  99,  99,  36, 100, 100,  37, 101,  38, 102, 102,  39, 103, 103,  40, 104, 104,  41,  41, 105,  42,  42, 106, 106,  43, 107, 107,
  44, 108, 108,  45,  45, 109,  46,  46,  46, 110,  47,  47, 111, 111,  48,  48, 112,  49,  49,  49, 113,  50,  50,  50, 114,  51,  51,  51, 115,  52,  52,  52,
 116, 116,  53,  53,  53, 117,  54,  54,  54, 118, 118,  55,  55,  55, 119, 119,  56,  56,  56, 120, 120,  57,  57,  57, 121, 121,  58,  58,  58,  58, 122,  59,
  59,  59,  59, 123, 123,  60,  60,  60,  60, 124,  61,  61,  61,  61,  61, 125,  62,  62,  62,  62,  62, 126,  63,  63, 63,
};
inline int best_index_iq6nl(const float * values, float x) {
    int ix = (int)(x - values[0]);
    if (ix < 0 || ix >= 249) return ix < 0 ? 0 : 63;
    ix = iq6nl_index[ix];
    return ix < 64 ? ix : x - values[ix-64] < values[ix-63] - x ? ix-64 : ix-63;
    //if (x <= val[0]) return 0;
    //if (x >= val[63]) return 63;
    //int index = iq6nl_index[int(x - val[0])];
    //return index < 64 ? index : x - val[index-64] < val[index-63] - x ? index - 64 : index - 63;
}


void quantize_row_iq6_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, const float * values, const float * shifted_values) {
    const int ntry = 5;
    const float step = 1.f;

    block_iq6_k * y = (block_iq6_k *)vy;

    float scales[QK_K/16];
    float weight[16];

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq6_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 2*sumx2/QK_K;

        float max_scale = 0, max_abs_scale = 0;
        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (!amax) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                //int l = best_index(64, values, al);
                int l = best_index_iq6nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                //l = best_index(64, values, -al);
                l = best_index_iq6nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry*step + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    //int l = best_index(64, values, al);
                    int l = best_index_iq6nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    //l = best_index(64, values, -al);
                    l = best_index_iq6nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry*step + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    //int l = best_index(64, shifted_values, al);
                    int l = best_index_iq6nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    //l = best_index(64, shifted_values, -al);
                    l = best_index_iq6nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (d) {
                const float * block_values = is_shifted ? shifted_values : values;
                float sumqx = 0, sumq2 = 0;
                id = 1/d;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    //int l = best_index(64, block_values, al);
                    int l = best_index_iq6nl(block_values, al);
                    float q = block_values[l];
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) d = sumqx/sumq2;
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale; max_scale = scales[ib];
            }

        }

        if (!max_abs_scale) continue;
        float d = -max_scale/127;
        y[ibl].d = GGML_FP32_TO_FP16(d);
        y[ibl].extra = extra;

        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = nearest_int(id*scales[ib]);
            ls = MAX(-127, MIN(127, ls));
            y[ibl].scales[ib] |= ls;
            float dl = d * ls;
            if (dl) {
                const float * block_values = y[ibl].extra & (1 << ib) ? shifted_values : values;
                const float * xb = xbl + 16*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*16;
                    for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/2) + offset;
                uint8_t * qh = y[ibl].qh + 32*(ib32/4) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    //int ibest = best_index(64, block_values, al);
                    int ibest = best_index_iq6nl(block_values, al);
                    qs[j] |= ((ibest & 0xf) << 4*(ib32%2));
                    qh[j] |= ((ibest >>  4) << 2*(ib32%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        if (sumq2 > 0) y[ibl].d = GGML_FP32_TO_FP16(sumqx/sumq2);

    }
}

}

void quantize_row_iq6_k_ref(const float * x, block_iq6_k * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq6_k(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq6_k(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq6_k * y = (block_iq6_k *)vy;
    quantize_row_iq6_k_ref(x, y, k);
}

size_t quantize_iq6_k(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    char * qrow = (char *)dst;
    float values[128];
    for (int i = 0; i < 64; ++i) {
        values[i] = iq6nl_values[i];
        values[i+64] = values[i] + S_IQ6K;
    }
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq6_k_impl(src, (void *)qrow, n_per_row, imatrix, values, values + 64);
        src += n_per_row;
        qrow += nblock*sizeof(block_iq6_k);
    }
    return nrows * nblock * sizeof(block_iq6_k);
}

template <int q8_type>
void iqk_quantize_row_q8_K_T(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    block_q8_K * y = (block_q8_K *)vy;
#ifdef __AVX2__
    const __m256 signBit = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int i = 0; i < nb; i++) {
        const float * xb = x + i*QK_K;
        __m256 maxAbs = _mm256_setzero_ps();
        const float * xx = xb;
        for (int ib = 0; ib < QK_K/8; ++ib) {
            const __m256 v = _mm256_loadu_ps(xx); xx += 8;
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps(signBit, v));
        }
        const float maxScalar = hmax_f32_8(maxAbs);
        const float d = maxScalar / 127.f;
        y[i].d = d;
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );
        xx = xb;
        int8_t * q8 = y[i].qs;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            __m256 v0 = _mm256_mul_ps(mul, _mm256_loadu_ps(xx)); xx += 8;
            __m256 v1 = _mm256_mul_ps(mul, _mm256_loadu_ps(xx)); xx += 8;
            __m256 v2 = _mm256_mul_ps(mul, _mm256_loadu_ps(xx)); xx += 8;
            __m256 v3 = _mm256_mul_ps(mul, _mm256_loadu_ps(xx)); xx += 8;
            v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
            v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
            v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
            v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);
            __m256i i0 = _mm256_cvtps_epi32(v0);
            __m256i i1 = _mm256_cvtps_epi32(v1);
            __m256i i2 = _mm256_cvtps_epi32(v2);
            __m256i i3 = _mm256_cvtps_epi32(v3);
            if constexpr (q8_type > 0) {
                int bsum = hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
                auto bs = (float *)y[i].bsums;
                bs[ib] = d*bsum;
            } else {
                y[i].bsums[2*ib+0] = hsum_i32_8(_mm256_add_epi32(i0, i1));
                y[i].bsums[2*ib+1] = hsum_i32_8(_mm256_add_epi32(i2, i3));
            }
            i0 = _mm256_packs_epi32( i0, i1 );
            i2 = _mm256_packs_epi32( i2, i3 );
            i0 = _mm256_packs_epi16( i0, i2 );
            i0 = _mm256_permutevar8x32_epi32( i0, perm );
            _mm256_storeu_si256((__m256i *)q8, i0);
            q8 += 32;
        }
        if constexpr (q8_type == 2) {
            auto bs = (float *)y[i].bsums;
            float sum = 0;
            for (int ib = 0; ib < QK_K/32; ++ib) sum += bs[ib];
            bs[0] = sum;
        }
    }
#else
    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        if constexpr (q8_type > 0) {
            auto bs = (float *)y[i].bsums;
            float d = 1/iscale;
            float sum = 0;
            for (int j = 0; j < QK_K/32; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 32; ++ii) {
                    sum += y[i].qs[j*32 + ii];
                }
                bs[j] = d*sum;
                sum += bs[j];
            }
            if constexpr (q8_type == 2) {
                bs[0] = sum;
            }
        } else {
            for (int j = 0; j < QK_K/16; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 16; ++ii) {
                    sum += y[i].qs[j*16 + ii];
                }
                y[i].bsums[j] = sum;
            }
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
#endif

}

void iqk_quantize_row_q8_K(const float * x, void * vy, int64_t k) {
    iqk_quantize_row_q8_K_T<0>(x, vy, k);
}

void quantize_row_q8_K32(const float * x, void * vy, int64_t k) {
    iqk_quantize_row_q8_K_T<1>(x, vy, k);
}

void quantize_row_q8_KR8(const float * x, void * vy, int64_t k) {
    iqk_quantize_row_q8_K_T<2>(x, vy, k);
}

namespace {
static void quantize_row_iq4_k_impl_bs128(const int super_block_size, const int block_size,
        int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    //GGML_ASSERT(super_block_size == 256 && block_size == 128);

    float * dptr = (float *)cy;
    block_iq4_ks * y = (block_iq4_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 16;

    float amax_scale = 0;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq4_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (!amax) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            bool is_shifted = false;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq4nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq4nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq4nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq4nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = d;
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    //float mse = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = y[ibl].scales[ib] & 0x01 ? shifted_values : values;
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = std::max(0, std::min(127, l)) << 1;
            //printf("d = %g, id = %g, scales = %g, l = %d, dl = %g\n", d, id, scales[ib], l, d*(l - 127));
            y[ibl].scales[ib] |= l;
            l -= 127;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            auto qs = y[ibl].qs + ib*(block_size/2);
            for (int j = 0; j < block_size/2; ++j) {
                uint8_t i1 = best_index_iq4nl(block_values, idl*xb[j]);
                uint8_t i2 = best_index_iq4nl(block_values, idl*xb[j+block_size/2]);
                qs[j] = i1 | (i2 << 4);
                float w1 = weight[j];
                float w2 = weight[j+block_size/2];
                float q1 = block_values[i1]*l;
                float q2 = block_values[i2]*l;
                sumqx += w1*q1*xb[j] + w2*q2*xb[j+block_size/2];
                sumq2 += w1*q1*q1 + w2*q2*q2;
                //float diff = xb[j] - d*q1; mse += diff*diff;
                //diff = xb[j+block_size/2] - d*q2; mse += diff*diff;
            }
        }
    }
    //printf("rmse = %g\n", sqrt(mse/n_per_row));
    if (sumq2 > 0) *dptr = sumqx/sumq2;
}
}

void quantize_row_iq4_ks_ref(const float * x, block_iq4_ks * y, int64_t k) {
    quantize_iq4_ks(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq4_ks(const float * x, void * y, int64_t k) {
    quantize_iq4_ks(x, (void *)y, 1, k, nullptr);
}

size_t quantize_iq4_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    //printf("============ %s(%d, %d)\n", __func__, int(nrows), int(n_per_row));
    constexpr int kBlockSize = 32; //128;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    char * qrow = (char *)dst;
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq4_k_impl_bs128(QK_K, kBlockSize, n_per_row, src, qrow, all_scales.data(), weight, iq4k_values, imatrix, 7);
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}

void dequantize_row_iq4_ks(const block_iq4_ks * x, float * y, int64_t k) {
    constexpr int kBlockSize = 32; //128;
    GGML_ASSERT(k%QK_K == 0);
    const float * dptr = (const float *)x;
    float d = *dptr;
    x = (const block_iq4_ks *)(dptr + 1);
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qs = x[ibl].qs;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            float dl = d * ((int)(x[ibl].scales[ib] & 254) - 127);
            const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            for (int j = 0; j < kBlockSize/2; ++j) {
                y[j             ] = dl * values[qs[j] & 0xf];
                y[j+kBlockSize/2] = dl * values[qs[j] >>  4];
            }
            y  += kBlockSize;
            qs += kBlockSize/2;
        }
    }
}

void  vec_dot_iq4_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    constexpr int kBlockSize = 32;
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_KS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    const float * dptr = (const float *)vx;
    const float d = *dptr;
    //printf("%s: n = %d, d = %g\n", __func__, n, d);
    const block_iq4_ks * x = (const block_iq4_ks *)(dptr + 1);
    const block_q8_K    * y = (const block_q8_K    *)vy;
    int nblock = n/QK_K;
    float sumf = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        //int sumi = 0;
        auto qy = y[ibl].qs;
        auto qx = x[ibl].qs;
        float db = d * y[ibl].d;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            float dl = db * ((x[ibl].scales[ib] & 254) - 127);
            //int ls = (x[ibl].scales[ib] & 254) - 127;
            const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            int suml = 0;
            for (int j = 0; j < kBlockSize/2; ++j) {
                suml += qy[j               ] * values[qx[j] & 0xf]
                      + qy[j + kBlockSize/2] * values[qx[j] >>  4];
            }
            sumf += dl * suml;
            //sumi += ls * suml;
            qy += kBlockSize;
            qx += kBlockSize/2;
        }
        //sumf += d * y[ibl].d * sumi;
    }
    *s = sumf;
}

namespace {
const uint16_t * scramble_table() {
    static std::mutex mutex;
    static std::vector<uint16_t> table;
    std::lock_guard<std::mutex> lock(mutex);
    if (table.empty()) {
        table.resize(1 << 15);
        for (int i = 0; i < int(table.size()); ++i) {
            uint16_t val = i;
            int non = popcount(val);
            if (non%2) val |= (1 << 15);
            bool found = false;
            for (int j = 0; j < int(table.size()); ++j) {
                if ((j ^ (j << 1)) == val) {
                    table[i] = j; found = true; break;
                }
            }
            if (!found) {
                printf("Oops: did not find for %d %u\n", i, val);
                exit(1);
            }
        }
    }
    return table.data();
}
uint16_t prune_iq4ks(uint16_t v, const int8_t * values, const float * x, const float * w, float dl) {
    if (popcount(v)%2 == 0) return v;
    float best_score = std::numeric_limits<float>::max();
    uint8_t q4[4];
    int jbest = -1;
    uint8_t bestq = 0;
    for (int j = 0; j < 4; ++j) {
        uint8_t q = (v >> 4*j) & 0xf;
        q4[j] = q;
        auto pc = popcount(q);
        float diff0 = dl*iq4k_values[q] - x[j];
        if (q > 0) {
            uint8_t qm = q - 1u;
            int pcm = popcount(qm);
            if (pcm == pc-1 || pcm == pc+1) {
                float diff1 = dl*values[qm] - x[j];
                float score = w[j]*(diff1*diff1 - diff0*diff0);
                if (score < best_score) {
                    best_score = score; jbest = j; bestq = qm;
                }
            }
        }
        if (q < 15) {
            uint8_t qp = q + 1u;
            int pcp = popcount(qp);
            if (pcp == pc-1 || pcp == pc+1) {
                float diff1 = dl*values[qp] - x[j];
                float score = w[j]*(diff1*diff1 - diff0*diff0);
                if (score < best_score) {
                    best_score = score; jbest = j; bestq = qp;
                }
            }
        }
    }
    GGML_ASSERT(jbest >= 0);
    q4[jbest] = bestq;
    return (q4[0] | (q4[1] << 4) | (q4[2] << 8) | (q4[3] << 12));
}
static void quantize_row_iq4_kss_impl(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const uint16_t * table,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    float * dptr = (float *)cy;
    *dptr = 0;
    block_iq4_kss * y = (block_iq4_kss *)(dptr + 1);

    const int8_t * shifted_values = values + 16;

    uint16_t vps[block_size/2], vms[block_size/2], vs[block_size/2];
    float xv[4], wv[4];

    float amax_scale = 0;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq4_kss));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (!amax) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float d = -max/iq4k_values[0];
            std::memset(vs, 0, block_size);
            for (int itry = -ntry; itry <= ntry; ++itry) {
                float id = (itry + values[0])/max;
                float sumqx_p = 0, sumq2_p = 0;
                float sumqx_m = 0, sumq2_m = 0;
                float this_d = 1/id;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (best_index_iq4nl(values,  al) << 4*j);
                        vm |= (best_index_iq4nl(values, -al) << 4*j);
                    }
                    vp = prune_iq4ks(vp, values, xv, wv,  this_d);
                    vm = prune_iq4ks(vm, values, xv, wv,  this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j];
                        sumq2_p += w*q*q;
                        q = values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j];
                        sumq2_m += w*q*q;
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                bool copy_p = false, copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; copy_p = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; copy_m = true;
                }
                if (copy_m) {
                    std::memcpy(vs, vms, block_size);
                } else if (copy_p) {
                    std::memcpy(vs, vps, block_size);
                }

                id = (itry + shifted_values[0])/max;
                this_d = 1/id;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (best_index_iq4nl(shifted_values,  al) << 4*j);
                        vm |= (best_index_iq4nl(shifted_values, -al) << 4*j);
                    }
                    vp = prune_iq4ks(vp, shifted_values, xv, wv,  this_d);
                    vm = prune_iq4ks(vm, shifted_values, xv, wv,  this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = shifted_values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j];
                        sumq2_p += w*q*q;
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j];
                        sumq2_m += w*q*q;
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                copy_p = copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; copy_p = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; copy_m = true;
                }
                if (copy_m) {
                    std::memcpy(vs, vms, block_size);
                } else if (copy_p) {
                    std::memcpy(vs, vps, block_size);
                }
            }
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = d;
    if (!d) return;
    float id = 1/d;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = (std::max(0, std::min(127, l)) << 1) - 127;
            if (l) {
                float dl = d*l;
                float idl = 1/dl;
                float mse_p = 0, mse_m = 0;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = idl*xv[j];
                        vp |= (best_index_iq4nl(        values, al) << 4*j);
                        vm |= (best_index_iq4nl(shifted_values, al) << 4*j);
                    }
                    vp = prune_iq4ks(vp,         values, xv, wv,  dl);
                    vm = prune_iq4ks(vm, shifted_values, xv, wv,  dl);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        mse_p += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        mse_m += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                const uint16_t * v = vps;
                const int8_t * block_values = values;
                if (mse_m < mse_p) {
                    v = vms;
                    block_values = values + 16;
                }
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    for (int j = 0; j < 4; ++j) {
                        float q = block_values[(v[k] >> 4*j) & 0xf] * l;
                        sumqx += wv[j]*q*xv[j];
                        sumq2 += wv[j]*q*q;
                    }
                }
                l += 127;
                if (mse_m < mse_p) l |= 1;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int k = 0; k < block_size/4; ++k) {
                    auto val = table[v[k] & 0x7fff];
                    q16[k] = (val << 1) | ((l >> k) & 1);
                }
            } else {
                l += 127;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int k = 0; k < block_size/4; ++k) {
                    q16[k] = ((l >> k) & 1);
                }
            }
        }
    }
    if (sumq2 > 0) *dptr = sumqx/sumq2;
}

void prune_iq4ks_to_iq4kss(int n_per_row, const uint16_t * table, const char * cx, const float * x, char *cy,
        const float * quant_weights, float * weight, float * all_scales) {
    constexpr int kBlockSize = 32;
    float xv[4], wv[4];
    uint16_t vps[kBlockSize/4];
    const float * dptr_ks = (const float *)cx;
    const float d_ks = *dptr_ks;
    const block_iq4_ks * iq4ks = (const block_iq4_ks *)(dptr_ks + 1);
    float * dptr = (float *)cy;
    *dptr = d_ks;
    block_iq4_kss * y = (block_iq4_kss *)(dptr + 1);
    int nblock = n_per_row/QK_K;
    float max_abs_scale = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        const float * xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/QK_K;
        const uint16_t * q4 = (const uint16_t *)iq4ks[ibl].qs;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float * xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = xb[j]*xb[j];
            }
            const int8_t * values = iq4k_values + ((iq4ks[ibl].scales[ib] & 1) << 4);
            float dl  = d_ks * ((iq4ks[ibl].scales[ib] & 254) - 127);
            float sumqx = 0, sumq2 = 0;
            for (int k = 0; k < kBlockSize/4; ++k) {
                xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+kBlockSize/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+kBlockSize/2];
                wv[0] = weight[2*k+0]; wv[1] = weight[2*k+kBlockSize/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+kBlockSize/2];
                auto vp = prune_iq4ks(q4[k], values, xv, wv, dl);
                vps[k] = table[vp & 0x7fff];
                for (int j = 0; j < 4; ++j) {
                    float q = values[(vp >> 4*j) & 0xf];
                    sumqx += wv[j]*q*xv[j];
                    sumq2 += wv[j]*q*q;
                }
            }
            for (int k = 0; k < kBlockSize/8; ++k) {
                y[ibl].qs[(kBlockSize/8)*ib + k] = vps[2*k+0] | (vps[2*k+1] << 15) | (((iq4ks[ibl].scales[ib] >> 2*k) & 3) << 30);
                //y[ibl].qs[(kBlockSize/8)*ib + k] = vps[2*k+0] | (vps[2*k+1] << 15);
            }
            scales[ib] = sumq2 > 0 ? sumqx/sumq2 : dl;
            max_abs_scale = std::max(max_abs_scale, scales[ib]);
            q4 += kBlockSize/4;
        }
    }
    //if (!max_abs_scale) return;
    //float d = max_abs_scale/127;
    //*dptr = d;
    //float id = 1/d;
    //for (int ibl = 0; ibl < nblock; ++ibl) {
    //    auto scales = all_scales + ibl*(QK_K/kBlockSize);
    //    for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
    //        int l = nearest_int(0.5f*(id*scales[ib]+127.f));
    //        l = std::max(0, std::min(127, l)) << 1;
    //        l |= (iq4ks[ibl].scales[ib] & 1);
    //        for (int k = 0; k < 4; ++k) {
    //            //y[ibl].qs[4*ib+k] &= 0x3fffffff;
    //            y[ibl].qs[4*ib+k] |= (((l >> 2*k) & 3) << 30);
    //        }
    //    }
    //}
}
}

size_t quantize_iq4_kss(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32; //128;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size    = ggml_row_size(GGML_TYPE_IQ4_KSS, n_per_row);
    auto row_size_ks = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    std::vector<char> work(row_size_ks);
    std::vector<float> all_scales(n_per_row/kBlockSize);
    float weight[kBlockSize];
    auto qrow = (char *)dst;
    auto table = scramble_table();
    for (int row = 0; row < nrows; ++row) {
        quantize_row_iq4_kss_impl(n_per_row, src, qrow, all_scales.data(), weight, iq4k_values, imatrix, table, 7);
        src  += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}

void quantize_row_iq4_kss_ref(const float * x, block_iq4_kss * y, int64_t k) {
    quantize_iq4_kss(x, y, 1, k, nullptr);
}

void quantize_row_iq4_kss(const float * x, void * y, int64_t k) {
    quantize_iq4_kss(x, (block_iq4_kss *)y, 1, k, nullptr);
}

void dequantize_row_iq4_kss(const block_iq4_kss * x, float * y, int64_t k) {
    const float * dptr = (const float *)x;
    const float d = *dptr;
    x = (const block_iq4_kss *)(dptr + 1);
    uint16_t aux16[8];
    const uint8_t * aux8 = (const uint8_t *)aux16;
    for (int ibl = 0; ibl < k/QK_K; ++ibl) {
        auto qs = (const uint16_t *)x[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            //uint8_t ls = ((qs[0] >> 30) | ((qs[1] >> 28) & 0x0c) | ((qs[2] >> 26) & 0x30) | ((qs[3] >> 24) & 0xc0));
            //const int8_t * values = iq4k_values + ((ls & 1) << 4);
            //const float dl = d * ((ls & 254) - 127);
            //for (int k = 0; k < 4; ++k) {
            //    uint16_t vl = qs[k] & 0x7fff;
            //    vl ^= (vl << 1);
            //    uint16_t vh = (qs[k] >> 15) & 0x7fff;
            //    vh ^= (vh << 1);
            //    for (int j = 0; j < 4; ++j) {
            //        y[4*k + j +  0] = dl*values[(vl >> 4*j) & 0xf];
            //        y[4*k + j + 16] = dl*values[(vh >> 4*j) & 0xf];
            //    }
            //}
            int16_t ls = 0;
            for (int k = 0; k < 8; ++k) {
                aux16[k] = qs[k] & 0xfffe;
                aux16[k] ^= (aux16[k] >> 1);
                ls |= (qs[k] & 1) << k;
            }
            const int8_t * values = iq4k_values + ((ls & 1) << 4);
            float dl = d * ((ls & 254) - 127);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * values[aux8[j] & 0xf];
                y[j+16] = dl * values[aux8[j] >>  4];
            }
            y  += 32;
            qs += 8;
        }
    }
}

void vec_dot_iq4_kss_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_KSS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= iq4_nl_r4
//
void quantize_row_iq4_nl_r4_ref(const float * x, block_iq4_nl_r4  * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_iq4_nl_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq4_nl_r4(const float * x, void * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_iq4_nl_r4(x, y, 4, k/4, nullptr);
}

static void repack_iq4_nl(int nrows, int n_per_row, const block_iq4_nl * x, block_iq4_nl_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK4_NL == 0);
    int nblock = n_per_row/QK4_NL;
    const block_iq4_nl * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            for (int k = 0; k < 4; ++k) y[ib].d[k] = x4[k][ib].d;
            for (int k = 0; k < 4; ++k) for (int i = 0; i < 4; ++i) {
                y[ib].qs[4*k+i+ 0] = (x4[k][ib].qs[i+0] & 0xf) | ((x4[k][ib].qs[i+ 8] & 0x0f) << 4);  //  0....3 +  8...11 from each row
                y[ib].qs[4*k+i+16] = (x4[k][ib].qs[i+0] >>  4) | ((x4[k][ib].qs[i+ 8] & 0xf0));       // 16...19 + 24...27 from each row
                y[ib].qs[4*k+i+32] = (x4[k][ib].qs[i+4] & 0xf) | ((x4[k][ib].qs[i+12] & 0x0f) << 4);  //  4....7 + 12...15 from each row
                y[ib].qs[4*k+i+48] = (x4[k][ib].qs[i+4] >>  4) | ((x4[k][ib].qs[i+12] & 0xf0));       // 20...23 + 28...31 from each row
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_iq4_nl_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    auto row_size_nl = ggml_row_size(GGML_TYPE_IQ4_NL, n_per_row);
    std::vector<char> qtmp(4*row_size_nl);
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq4_nl(src, qtmp.data(), 4, n_per_row, imatrix);
        repack_iq4_nl(4, n_per_row, (const block_iq4_nl *)qtmp.data(), (block_iq4_nl_r4 *)qrow);
        src += 4*n_per_row;
        qrow += 4*row_size_nl;
    }
    return nrows*row_size_nl;
}

void dequantize_row_iq4_nl_r4(const block_iq4_nl_r4 * x, float * y, int64_t k) {
    // we assume we are called with 4 rows
    int n_per_row = k/4;
    int nb = n_per_row/QK4_NL;
    float * yk[4];
    for (int k = 0; k < 4; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 4; ++k) {
            float scale = GGML_FP16_TO_FP32(x[ib].d[k]);
            for (int i = 0; i < 4; ++i) {
                yk[k][QK4_NL*ib+i+ 0] = scale * iq4k_values[x[ib].qs[4*k+i+ 0] & 0xf];
                yk[k][QK4_NL*ib+i+ 8] = scale * iq4k_values[x[ib].qs[4*k+i+ 0] >>  4];
                yk[k][QK4_NL*ib+i+16] = scale * iq4k_values[x[ib].qs[4*k+i+16] & 0xf];
                yk[k][QK4_NL*ib+i+24] = scale * iq4k_values[x[ib].qs[4*k+i+16] >>  4];
                yk[k][QK4_NL*ib+i+ 4] = scale * iq4k_values[x[ib].qs[4*k+i+32] & 0xf];
                yk[k][QK4_NL*ib+i+12] = scale * iq4k_values[x[ib].qs[4*k+i+32] >>  4];
                yk[k][QK4_NL*ib+i+20] = scale * iq4k_values[x[ib].qs[4*k+i+48] & 0xf];
                yk[k][QK4_NL*ib+i+28] = scale * iq4k_values[x[ib].qs[4*k+i+48] >>  4];
            }
        }
    }
}

void vec_dot_iq4_nl_r4_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_NL_R4, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= q4_0_r4
//
void quantize_row_q4_0_r4_ref(const float * x, block_iq4_nl_r4  * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q4_0_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_q4_0_r4(const float * x, void * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q4_0_r4(x, y, 4, k/4, nullptr);
}

static void repack_q4_0(int nrows, int n_per_row, const block_q4_0 * x, block_iq4_nl_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK4_NL == 0);
    int nblock = n_per_row/QK4_NL;
    const block_q4_0 * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            //for (int k = 0; k < 4; ++k) y[ib].d[k] = x4[k][ib].d;
            //for (int k = 0; k < 4; ++k) for (int i = 0; i < 4; ++i) {
            //    y[ib].qs[4*k+i+ 0] = (x4[k][ib].qs[i+0] & 0xf) | ((x4[k][ib].qs[i+ 8] & 0x0f) << 4);  //  0....3 +  8...11 from each row
            //    y[ib].qs[4*k+i+16] = (x4[k][ib].qs[i+0] >>  4) | ((x4[k][ib].qs[i+ 8] & 0xf0));       // 16...19 + 24...27 from each row
            //    y[ib].qs[4*k+i+32] = (x4[k][ib].qs[i+4] & 0xf) | ((x4[k][ib].qs[i+12] & 0x0f) << 4);  //  4....7 + 12...15 from each row
            //    y[ib].qs[4*k+i+48] = (x4[k][ib].qs[i+4] >>  4) | ((x4[k][ib].qs[i+12] & 0xf0));       // 20...23 + 28...31 from each row
            //}
            for (int k = 0; k < 4; ++k) {
                y[ib].d[k] = x4[k][ib].d;
                for (int l = 0; l < 4; ++l) {
                    // l = 0 -> 0,  8 with shift 0   -> 4*(l/2), 4*(l/2)+8 with shift 4*(l%2)
                    // l = 1 -> 0,  8 with shift 4
                    // l = 2 -> 4, 12 with shift 0
                    // l = 3 -> 4, 12 with shift 4
                    for (int i = 0; i < 4; ++i) {
                        y[ib].qs[4*k+i+16*l] = ((x4[k][ib].qs[i+4*(l/2)] >> 4*(l%2)) & 0xf) | (((x4[k][ib].qs[i+4*(l/2)+8] >> 4*(l%2)) & 0xf) << 4);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_q4_0_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    auto row_size_nl = ggml_row_size(GGML_TYPE_IQ4_NL, n_per_row);
    std::vector<char> qtmp(4*row_size_nl);
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 4) {
        quantize_q4_0(src, qtmp.data(), 4, n_per_row, imatrix);
        repack_iq4_nl(4, n_per_row, (const block_iq4_nl *)qtmp.data(), (block_iq4_nl_r4 *)qrow);
        src += 4*n_per_row;
        qrow += 4*row_size_nl;
    }
    return nrows*row_size_nl;
}

void dequantize_row_q4_0_r4(const block_iq4_nl_r4 * x, float * y, int64_t k) {
    // we assume we are called with 4 rows
    int n_per_row = k/4;
    int nb = n_per_row/QK4_0;
    float * yk[4];
    for (int k = 0; k < 4; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 4; ++k) {
            float scale = GGML_FP16_TO_FP32(x[ib].d[k]);
            for (int l = 0; l < 4; ++l) {
                int ll = 16*(l%2) + 4*(l/2);
                for (int i = 0; i < 4; ++i) {
                    yk[k][QK4_0*ib+i+ll+0] = scale * ((x[ib].qs[4*k+i+16*l] & 0xf) - 8);
                    yk[k][QK4_0*ib+i+ll+8] = scale * ((x[ib].qs[4*k+i+16*l] >>  4) - 8);
                }
            }
        }
    }
}

void vec_dot_q4_0_r4_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q4_0_R4, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}


//
// ========================================= q8_0_r4
//
void quantize_row_q8_0_r4_ref(const float * x, block_q8_0_x4  * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q8_0_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_q8_0_r4(const float * x, void * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q8_0_r4(x, y, 4, k/4, nullptr);
}

static void repack_q8_0(int nrows, int n_per_row, const block_q8_0 * x, block_q8_0_x4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK8_0 == 0);
    int nblock = n_per_row/QK8_0;
    const block_q8_0 * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            for (int k = 0; k < 4; ++k) y[ib].d[k] = x4[k][ib].d;
            for (int l = 0; l < 4; ++l) {
                for (int k = 0; k < 4; ++k) for (int i = 0; i < 4; ++i) {
                    y[ib].qs[32*l+4*k+i+ 0] = x4[k][ib].qs[i+4*l+ 0];
                    y[ib].qs[32*l+4*k+i+16] = x4[k][ib].qs[i+4*l+16];
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_q8_0_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<char> qtmp(4*row_size_0);
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 4) {
        quantize_q8_0(src, qtmp.data(), 4, n_per_row, imatrix);
        repack_q8_0(4, n_per_row, (const block_q8_0 *)qtmp.data(), (block_q8_0_x4 *)qrow);
        src += 4*n_per_row;
        qrow += 4*row_size_0;
    }
    return nrows*row_size_0;
}

void dequantize_row_q8_0_r4(const block_q8_0_x4 * x, float * y, int64_t k) {
    // we assume we are called with 4 rows
    int n_per_row = k/4;
    int nb = n_per_row/QK8_0;
    float * yk[4];
    for (int k = 0; k < 4; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 4; ++k) {
            float scale = GGML_FP16_TO_FP32(x[ib].d[k]);
            for (int l = 0; l < 4; ++l) for (int i = 0; i < 4; ++i) {
                yk[k][QK8_0*ib+4*l+i+ 0] = scale * x[ib].qs[QK8_0*l+4*k+i+ 0];
                yk[k][QK8_0*ib+4*l+i+16] = scale * x[ib].qs[QK8_0*l+4*k+i+16];
            }
        }
    }
}

void vec_dot_q8_0_r4_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q8_0_R4, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= q5_0_r4
//
void quantize_row_q5_0_r4_ref(const float * x, block_q5_0_r4  * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q5_0_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_q5_0_r4(const float * x, void * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q5_0_r4(x, y, 4, k/4, nullptr);
}

static inline void convert_q5_0(const block_q5_0& x, uint8_t * L) {
    uint32_t qh;
    memcpy(&qh, x.qh, sizeof(qh));

    for (int j = 0; j < QK5_0/2; ++j) {
        const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

        L[j          ] = (x.qs[j] & 0x0F) | xh_0;
        L[j + QK4_0/2] = (x.qs[j] >>   4) | xh_1;
    }
}

static void repack_q5_0(int nrows, int n_per_row, const block_q5_0 * x, block_q5_0_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK5_0 == 0);
    int nblock = n_per_row/QK5_0;
    const block_q5_0 * x4[4];
    uint8_t L[QK5_0];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            std::memset(y[ib].qh, 0, QK5_0/2);
            for (int k = 0; k < 4; ++k) {
                y[ib].d[k] = x4[k][ib].d;
                convert_q5_0(x4[k][ib], L);
                for (int l = 0; l < 4; ++l) {
                    int l1 = 4*(l/2) + 16*(l%2), l2 = l1 + 8;
                    for (int i = 0; i < 4; ++i) {
                        y[ib].qs[4*k+i+16*l] = (L[i + l1] & 0xf) | ((L[i + l2] & 0xf) << 4);
                        y[ib].qh[4*k+i] |= ((L[i + l1] >> 4) | ((L[i + l2] >> 4) << 4)) << l;
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_q5_0_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q5_0, n_per_row);
    std::vector<char> qtmp(4*row_size_0);
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 4) {
        quantize_q5_0(src, qtmp.data(), 4, n_per_row, imatrix);
        repack_q5_0(4, n_per_row, (const block_q5_0 *)qtmp.data(), (block_q5_0_r4 *)qrow);
        src += 4*n_per_row;
        qrow += 4*row_size_0;
    }
    return nrows*row_size_0;
}

void dequantize_row_q5_0_r4(const block_q5_0_r4 * x, float * y, int64_t k) {
    // we assume we are called with 4 rows
    int n_per_row = k/4;
    int nb = n_per_row/QK8_0;
    float * yk[4];
    for (int k = 0; k < 4; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 4; ++k) {
            float d = GGML_FP16_TO_FP32(x[ib].d[k]);
            float m = -16*d;
            for (int l = 0; l < 4; ++l) {
                int ll = 16*(l%2) + 4*(l/2);
                for (int i = 0; i < 4; ++i) {
                    yk[k][QK4_0*ib+i+ll+0] = d * ((x[ib].qs[4*k+i+16*l] & 0xf) | (((x[ib].qh[4*k+i] >> (l+0)) & 1) << 4)) + m;
                    yk[k][QK4_0*ib+i+ll+8] = d * ((x[ib].qs[4*k+i+16*l] >>  4) | (((x[ib].qh[4*k+i] >> (l+4)) & 1) << 4)) + m;
                }
            }
        }
    }
}

void vec_dot_q5_0_r4_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q5_0_R4, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= q6_0_r4
//
void quantize_row_q6_0_r4_ref(const float * x, block_q6_0_r4  * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q6_0_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_q6_0_r4(const float * x, void * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q6_0_r4(x, y, 4, k/4, nullptr);
}

static inline void convert_q6_0(const block_q6_0& x, uint8_t * L) {

    for (int j = 0; j < QK6_0/2; ++j) {
        const uint8_t h = x.qh[j%(QK6_0/4)] >> 4*(j/(QK6_0/4));
        L[j          ] = (x.qs[j] & 0x0F) | ((h << 4) & 0x30);
        L[j + QK6_0/2] = (x.qs[j] >>   4) | ((h << 2) & 0x30);
    }
}

static void repack_q6_0(int nrows, int n_per_row, const block_q6_0 * x, block_q6_0_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK5_0 == 0);
    int nblock = n_per_row/QK6_0;
    const block_q6_0 * x4[4];
    uint8_t L[QK6_0];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            std::memset(y[ib].qh, 0, QK6_0);
            for (int k = 0; k < 4; ++k) {
                y[ib].d[k] = x4[k][ib].d;
                convert_q6_0(x4[k][ib], L);
                for (int l = 0; l < 4; ++l) {
                    int l1 = 4*(l/2) + 16*(l%2), l2 = l1 + 8;
                    for (int i = 0; i < 4; ++i) {
                        y[ib].qs[4*k+i+16*l] = (L[i + l1] & 0xf) | ((L[i + l2] & 0xf) << 4);
                        y[ib].qh[4*k+i+16*(l%2)] |= ((L[i + l1] >> 4) | ((L[i + l2] >> 4) << 4)) << 2*(l/2);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_q6_0_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q6_0, n_per_row);
    std::vector<char> qtmp(4*row_size_0);
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 4) {
        quantize_q6_0(src, qtmp.data(), 4, n_per_row, imatrix);
        repack_q6_0(4, n_per_row, (const block_q6_0 *)qtmp.data(), (block_q6_0_r4 *)qrow);
        src += 4*n_per_row;
        qrow += 4*row_size_0;
    }
    return nrows*row_size_0;
}

void dequantize_row_q6_0_r4(const block_q6_0_r4 * x, float * y, int64_t k) {
    // we assume we are called with 4 rows
    int n_per_row = k/4;
    int nb = n_per_row/QK6_0;
    float * yk[4];
    for (int k = 0; k < 4; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 4; ++k) {
            float d = GGML_FP16_TO_FP32(x[ib].d[k]);
            float m = -32*d;
            for (int l = 0; l < 4; ++l) {
                int ll = 16*(l%2) + 4*(l/2);
                for (int i = 0; i < 4; ++i) {
                    yk[k][QK4_0*ib+i+ll+0] = d * ((x[ib].qs[4*k+i+16*l] & 0xf) | (((x[ib].qh[4*k+i+16*(l%2)] >> (2*(l/2)+0)) & 3) << 4)) + m;
                    yk[k][QK4_0*ib+i+ll+8] = d * ((x[ib].qs[4*k+i+16*l] >>  4) | (((x[ib].qh[4*k+i+16*(l%2)] >> (2*(l/2)+4)) & 3) << 4)) + m;
                }
            }
        }
    }
}

void vec_dot_q6_0_r4_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q6_0_R4, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= iq4_xs_r4
//

void quantize_row_iq4_xs_r4_ref(const float * x, block_iq4_xs_r4 * y, int64_t k) {
    quantize_iq4_xs_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq4_xs_r4(const float * x, void * y, int64_t k) {
    quantize_iq4_xs_r4(x, y, 4, k/4, nullptr);
}

static void repack_iq4_xs(int nrows, int n_per_row, const block_iq4_xs * x, block_iq4_xs_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq4_xs * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/16);
            std::memset(y[ibl].scales_h, 0, QK_K/32);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    uint8_t sl = (x4[k][ibl].scales_l[ib/2] >> 4*(ib%2)) & 0xf;
                    uint8_t sh = (x4[k][ibl].scales_h >> 2*ib) & 3;
                    int i = 4*ib + k;
                    y[ibl].scales_l[i%16] |= (sl << 4*(i/16));
                    y[ibl].scales_h[i%8 ] |= (sh << 2*(i/8));
                }
            }
            for (int ib = 0; ib < QK_K/32; ++ib) {
                for (int k = 0; k < 4; ++k) for (int i = 0; i < 4; ++i) {
                    y[ibl].qs[64*ib+4*k+i+ 0] = (x4[k][ibl].qs[16*ib+i+0] & 0xf) | ((x4[k][ibl].qs[16*ib+i+ 8] & 0x0f) << 4);  //  0....3 +  8...11 from each row
                    y[ibl].qs[64*ib+4*k+i+16] = (x4[k][ibl].qs[16*ib+i+0] >>  4) | ((x4[k][ibl].qs[16*ib+i+ 8] & 0xf0));       // 16...19 + 24...27 from each row
                    y[ibl].qs[64*ib+4*k+i+32] = (x4[k][ibl].qs[16*ib+i+4] & 0xf) | ((x4[k][ibl].qs[16*ib+i+12] & 0x0f) << 4);  //  4....7 + 12...15 from each row
                    y[ibl].qs[64*ib+4*k+i+48] = (x4[k][ibl].qs[16*ib+i+4] >>  4) | ((x4[k][ibl].qs[16*ib+i+12] & 0xf0));       // 20...23 + 28...31 from each row
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_iq4_xs_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_XS, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq4_xs(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_iq4_xs(4, n_per_row, (const block_iq4_xs *)qtmp.data(), (block_iq4_xs_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq4_xs_r4(const block_iq4_xs_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 4*ib + k;
                float dl = d * ((((x[ibl].scales_l[is%16] >> 4*(is/16)) & 0xf) | (((x[ibl].scales_h[is%8] >> 2*(is/8)) & 3) << 4)) - 32);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl * iq4k_values[x[ibl].qs[64*ib+4*k+i+ 0] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl * iq4k_values[x[ibl].qs[64*ib+4*k+i+ 0] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl * iq4k_values[x[ibl].qs[64*ib+4*k+i+16] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl * iq4k_values[x[ibl].qs[64*ib+4*k+i+16] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl * iq4k_values[x[ibl].qs[64*ib+4*k+i+32] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl * iq4k_values[x[ibl].qs[64*ib+4*k+i+32] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl * iq4k_values[x[ibl].qs[64*ib+4*k+i+48] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl * iq4k_values[x[ibl].qs[64*ib+4*k+i+48] >>  4];
                }
            }
        }
        //dequantize_row_iq4_xs(x + ib, ytmp, QK_K);
        //for (int k = 0; k < 4; ++k) {
        //    for (int l = 0; l < 16; ++l) {
        //        for (int i = 0; i < 4; ++i) {
        //            //y4[k][ib*kBlockSize + i + 16*(l%4) + 4*(l/4)] = ytmp[16*l + 4*k + i];
        //            y4[k][ib*kBlockSize + i + 8*(l%8) + 4*(l/8)] = ytmp[16*l + 4*k + i];
        //        }
        //    }
        //}
    }
}

void vec_dot_iq4_xs_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_XS_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= iq4_ks_r4
//

void quantize_row_iq4_ks_r4_ref(const float * x, block_iq4_ks_r4 * y, int64_t k) {
    quantize_iq4_ks_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq4_ks_r4(const float * x, void * y, int64_t k) {
    quantize_iq4_ks_r4(x, y, 4, k/4, nullptr);
}

static void repack_iq4_ks(int nrows, int n_per_row, const block_iq4_ks * x, block_iq4_ks_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    int nblock = n_per_row/QK_K;
    char * cy = (char *)y;
    const char * cx = (const char *)x;
    const block_iq4_ks * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        float * dptr = (float *)cy;
        block_iq4_ks_r4 * y = (block_iq4_ks_r4 *)(dptr + 4);
        for (int k = 0; k < 4; ++k) {
            auto dk = (const float *)(cx + k*row_size);
            dptr[k] = dk[0];
            x4[k] = (const block_iq4_ks *)(dk + 1);
        }
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[64*ib+4*k+i+ 0] = (x4[k][ibl].qs[16*ib+i+0] & 0xf) | ((x4[k][ibl].qs[16*ib+i+ 8] & 0x0f) << 4);  //  0....3 +  8...11 from each row
                        y[ibl].qs[64*ib+4*k+i+16] = (x4[k][ibl].qs[16*ib+i+0] >>  4) | ((x4[k][ibl].qs[16*ib+i+ 8] & 0xf0));       // 16...19 + 24...27 from each row
                        y[ibl].qs[64*ib+4*k+i+32] = (x4[k][ibl].qs[16*ib+i+4] & 0xf) | ((x4[k][ibl].qs[16*ib+i+12] & 0x0f) << 4);  //  4....7 + 12...15 from each row
                        y[ibl].qs[64*ib+4*k+i+48] = (x4[k][ibl].qs[16*ib+i+4] >>  4) | ((x4[k][ibl].qs[16*ib+i+12] & 0xf0));       // 20...23 + 28...31 from each row
                    }
                }
            }
        }
        cx += 4*row_size;
        cy += 4*row_size;
    }
}

size_t quantize_iq4_ks_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq4_ks(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_iq4_ks(4, n_per_row, (const block_iq4_ks *)qtmp.data(), (block_iq4_ks_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq4_ks_r4(const block_iq4_ks_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    const float * dptr = (const float *)x;
    x = (const block_iq4_ks_r4 *)(dptr + 4);
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = dptr[k];
            for (int ib = 0; ib < QK_K/32; ++ib) {
                float dl = d * ((x[ibl].scales[4*ib + k] & 254) - 127);
                auto values = iq4k_values + ((x[ibl].scales[4*ib + k] & 1) << 4);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl * values[x[ibl].qs[64*ib+4*k+i+ 0] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl * values[x[ibl].qs[64*ib+4*k+i+ 0] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl * values[x[ibl].qs[64*ib+4*k+i+16] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl * values[x[ibl].qs[64*ib+4*k+i+16] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl * values[x[ibl].qs[64*ib+4*k+i+32] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl * values[x[ibl].qs[64*ib+4*k+i+32] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl * values[x[ibl].qs[64*ib+4*k+i+48] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl * values[x[ibl].qs[64*ib+4*k+i+48] >>  4];
                }
            }
        }
    }
}

void vec_dot_iq4_ks_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_KS_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= iq2_bn_r4
//
void quantize_row_iq2_bn_r4_ref(const float * x, block_iq2_bn  * y, int64_t k) {
    quantize_iq2_bn_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq2_bn_r4(const float * x, void * y, int64_t k) {
    quantize_iq2_bn_r4(x, y, 4, k/4, nullptr);
}

namespace {
void repack_iq2_bn(int nrows, int n_per_row, const char * x, char * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_IQ1BN == 0);
    int nblock = n_per_row/QK_IQ1BN;
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_BN, n_per_row);
    const uint8_t * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        float * dr4 = (float *)(y + 4*row*row_size);
        for (int k = 0; k < 4; ++k) {
            const float * dptr = (const float *)(x + (row + k)*row_size);
            dr4[k] = *dptr;
            x4[k] = (const uint8_t *)(dptr + 1);
        }
        uint8_t * y4 = (uint8_t *)(dr4 + 4);
        //std::memset(y4, 0, n_per_row);
        for (int ib = 0; ib < nblock; ++ib) {
            //  0...3 from rows 0...3 go to 1st 2 bits of  0...15
            // 16..19 from rows 0...3 go to 1st 2 bits of 16...31
            // 32..35 from rows 0...3 go to 1st 2 bits of 32...47
            // 48..51 from rows 0...3 go to 1st 2 bits of 48...63
            //  4...7 from rows 0...3 go to 2nd 2 bits of  0...15
            // 20..23 from rows 0...3 go to 2nd 2 bits of 16...31
            // 36..39 from rows 0...3 go to 2nd 2 bits of 32...47
            // 52..55 from rows 0...3 go to 2nd 2 bits of 48...63
            //  8..11 from rows 0...3 go to 3rd 2 bits of  0...15
            // 24..27 from rows 0...3 go to 3rd 2 bits of 16...31
            // 40..43 from rows 0...3 go to 3rd 2 bits of 32...47
            // 56..59 from rows 0...3 go to 3rd 2 bits of 48...63
            // 12..15 from rows 0...3 go to 4th 2 bits of  0...15
            // 28..31 from rows 0...3 go to 4th 2 bits of 16...31
            // 44..47 from rows 0...3 go to 4th 2 bits of 32...47
            // 60..63 from rows 0...3 go to 4th 2 bits of 48...63
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 4; ++l) for (int i = 0; i < 4; ++i) {
                    y4[64*ib + 4*k + i + 16*l] = (((x4[k][16*ib + i +  0] >> 2*l) & 3) << 0) |
                                                 (((x4[k][16*ib + i +  4] >> 2*l) & 3) << 2) |
                                                 (((x4[k][16*ib + i +  8] >> 2*l) & 3) << 4) |
                                                 (((x4[k][16*ib + i + 12] >> 2*l) & 3) << 6);
                    //y4[64*ib + 4*k + i +  0] |= (x4[k][16*ib + i] >> 0) & 3;
                    //y4[64*ib + 4*k + i + 16] |= (x4[k][16*ib + i] >> 2) & 3;
                    //y4[64*ib + 4*k + i + 32] |= (x4[k][16*ib + i] >> 4) & 3;
                    //y4[64*ib + 4*k + i + 48] |= (x4[k][16*ib + i] >> 6) & 3;
                    //y4[64*ib + 4*k + i +  0] |= ((x4[k][16*ib + i + 4] >> 0) & 3) << 2;
                    //y4[64*ib + 4*k + i + 16] |= ((x4[k][16*ib + i + 4] >> 2) & 3) << 2;
                    //y4[64*ib + 4*k + i + 32] |= ((x4[k][16*ib + i + 4] >> 4) & 3) << 2;
                    //y4[64*ib + 4*k + i + 48] |= ((x4[k][16*ib + i + 4] >> 6) & 3) << 2;
                    //y4[64*ib + 4*k + i +  0] |= ((x4[k][16*ib + i + 8] >> 0) & 3) << 4;
                    //y4[64*ib + 4*k + i + 16] |= ((x4[k][16*ib + i + 8] >> 2) & 3) << 4;
                    //y4[64*ib + 4*k + i + 32] |= ((x4[k][16*ib + i + 8] >> 4) & 3) << 4;
                    //y4[64*ib + 4*k + i + 48] |= ((x4[k][16*ib + i + 8] >> 6) & 3) << 4;
                    //y4[64*ib + 4*k + i +  0] |= ((x4[k][16*ib + i + 12] >> 0) & 3) << 6;
                    //y4[64*ib + 4*k + i + 16] |= ((x4[k][16*ib + i + 12] >> 2) & 3) << 6;
                    //y4[64*ib + 4*k + i + 32] |= ((x4[k][16*ib + i + 12] >> 4) & 3) << 6;
                    //y4[64*ib + 4*k + i + 48] |= ((x4[k][16*ib + i + 12] >> 6) & 3) << 6;
                }
            }
        }
    }
}
}

size_t quantize_iq2_bn_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_IQ1BN == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_BN, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq2_bn(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_iq2_bn(4, n_per_row, qtmp.data(), qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq2_bn_r4(const block_iq2_bn * x, float * y, int64_t k) {
    static_assert(QK_IQ1BN == 64);
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    const float * d4 = (const float *)x;
    const uint8_t * qx = (const uint8_t *)(d4 + 4);
    int nblock = n_per_row/QK_IQ1BN;
    for (int ib = 0; ib < nblock; ++ib) {
        for (int k = 0; k < 4; ++k) {
            for (int l = 0; l < 4; ++l) for (int i = 0; i < 4; ++i) {
                uint8_t q = qx[4*k + i + 16*l];
                y4[k][64*ib + 16*l + i +  0] = d4[k] * (((q >> 0) & 3) - 1);
                y4[k][64*ib + 16*l + i +  4] = d4[k] * (((q >> 2) & 3) - 1);
                y4[k][64*ib + 16*l + i +  8] = d4[k] * (((q >> 4) & 3) - 1);
                y4[k][64*ib + 16*l + i + 12] = d4[k] * (((q >> 6) & 3) - 1);
            }
        }
        qx += 64;
    }
}

void vec_dot_iq2_bn_r4_q8_K64(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_BN_R4, vx, 0, GGML_TYPE_Q8_K64, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= q4_k_r4
//

void quantize_row_q4_k_r4_ref(const float * x, block_q4_k_r4 * y, int64_t k) {
    quantize_q4_k_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_q4_k_r4(const float * x, void * y, int64_t k) {
    quantize_q4_k_r4(x, y, 4, k/4, nullptr);
}

namespace {
inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
inline void convert_q4_k(const block_q4_K& x, uint8_t * L, uint8_t * Ld, uint8_t * Lm) {
    for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
        get_scale_min_k4(2*ib64+0, x.scales, Ld[2*ib64+0], Lm[2*ib64+0]);
        get_scale_min_k4(2*ib64+1, x.scales, Ld[2*ib64+1], Lm[2*ib64+1]);
        for (int j = 0; j < 32; ++j) {
            L[64*ib64+j+ 0] = x.qs[32*ib64+j] & 0xf;
            L[64*ib64+j+32] = x.qs[32*ib64+j] >>  4;
        }
    }
}
}

static void repack_q4_k(int nrows, int n_per_row, const block_q4_K * x, block_q4_k_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q4_K * x4[4];
    uint8_t L[QK_K], Ld[QK_K/32], Lm[QK_K/32];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k+0] = x4[k][ibl].d;
                y[ibl].d[k+4] = x4[k][ibl].dmin;
                convert_q4_k(x4[k][ibl], L, Ld, Lm);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales_l[4*ib+k] = (Ld[ib] & 0xf) | ((Lm[ib] & 0xf) << 4);
                    uint8_t h = (Ld[ib] >> 4) | ((Lm[ib] >> 4) << 2);
                    y[ibl].scales_h[(4*ib+k)%16] |= (h << 4*((4*ib+k)/16));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[64*ib+4*k+i+ 0] = L[32*ib+i+ 0] | (L[32*ib+i+ 8] << 4);
                        y[ibl].qs[64*ib+4*k+i+16] = L[32*ib+i+16] | (L[32*ib+i+24] << 4);
                        y[ibl].qs[64*ib+4*k+i+32] = L[32*ib+i+ 4] | (L[32*ib+i+12] << 4);
                        y[ibl].qs[64*ib+4*k+i+48] = L[32*ib+i+20] | (L[32*ib+i+28] << 4);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_q4_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_Q4_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_q4_K(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_q4_k(4, n_per_row, (const block_q4_K *)qtmp.data(), (block_q4_k_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_q4_k_r4(const block_q4_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k+0]);
            const float m = GGML_FP16_TO_FP32(x[ibl].d[k+4]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 4*ib + k;
                float dl = d * ((x[ibl].scales_l[is] & 0xf) | (((x[ibl].scales_h[is%16] >> 4*(is/16)) & 0x03) << 4));
                float ml = m * ((x[ibl].scales_l[is] >>  4) | (((x[ibl].scales_h[is%16] >> 4*(is/16)) & 0x0c) << 2));
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl * (x[ibl].qs[64*ib+4*k+i+ 0] & 0xf) - ml;
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl * (x[ibl].qs[64*ib+4*k+i+ 0] >>  4) - ml;
                    y4[k][QK_K*ibl+32*ib+i+16] = dl * (x[ibl].qs[64*ib+4*k+i+16] & 0xf) - ml;
                    y4[k][QK_K*ibl+32*ib+i+24] = dl * (x[ibl].qs[64*ib+4*k+i+16] >>  4) - ml;
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl * (x[ibl].qs[64*ib+4*k+i+32] & 0xf) - ml;
                    y4[k][QK_K*ibl+32*ib+i+12] = dl * (x[ibl].qs[64*ib+4*k+i+32] >>  4) - ml;
                    y4[k][QK_K*ibl+32*ib+i+20] = dl * (x[ibl].qs[64*ib+4*k+i+48] & 0xf) - ml;
                    y4[k][QK_K*ibl+32*ib+i+28] = dl * (x[ibl].qs[64*ib+4*k+i+48] >>  4) - ml;
                }
            }
        }
    }
}

void vec_dot_q4_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q4_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= q6_k_r4
//

void quantize_row_q6_k_r4_ref(const float * x, block_q6_k_r4 * y, int64_t k) {
    quantize_q6_k_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_q6_k_r4(const float * x, void * y, int64_t k) {
    quantize_q6_k_r4(x, y, 4, k/4, nullptr);
}

namespace {
inline void convert_q6_k(const block_q6_K& x, uint8_t * L) {
    const uint8_t * ql = x.ql;
    const uint8_t * qh = x.qh;

    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            L[n + l +  0] = (ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            L[n + l + 32] = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            L[n + l + 64] = (ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4);
            L[n + l + 96] = (ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4);
        }
        ql += 64;
        qh += 32;
    }
}
}

static void repack_q6_k(int nrows, int n_per_row, const block_q6_K * x, block_q6_k_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q6_K * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                convert_q6_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales[8*ib+k+0] = x4[k][ibl].scales[2*ib+0];
                    y[ibl].scales[8*ib+k+4] = x4[k][ibl].scales[2*ib+1];
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].ql[64*ib+4*k+i+ 0] = (L[32*ib+i+ 0] & 0xf) | ((L[32*ib+i+ 8] & 0xf) << 4);
                        y[ibl].ql[64*ib+4*k+i+16] = (L[32*ib+i+16] & 0xf) | ((L[32*ib+i+24] & 0xf) << 4);
                        y[ibl].ql[64*ib+4*k+i+32] = (L[32*ib+i+ 4] & 0xf) | ((L[32*ib+i+12] & 0xf) << 4);
                        y[ibl].ql[64*ib+4*k+i+48] = (L[32*ib+i+20] & 0xf) | ((L[32*ib+i+28] & 0xf) << 4);
                        y[ibl].qh[32*ib+4*k+i+ 0] = (L[32*ib+i+ 0] >> 4) | ((L[32*ib+i+ 8] >> 4) << 2) | ((L[32*ib+i+ 4] >> 4) << 4) | ((L[32*ib+i+12] >> 4) << 6);
                        y[ibl].qh[32*ib+4*k+i+16] = (L[32*ib+i+16] >> 4) | ((L[32*ib+i+24] >> 4) << 2) | ((L[32*ib+i+20] >> 4) << 4) | ((L[32*ib+i+28] >> 4) << 6);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_q6_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_Q6_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_q6_K(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_q6_k(4, n_per_row, (const block_q6_K *)qtmp.data(), (block_q6_k_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_q6_k_r4(const block_q6_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            auto ql = x[ibl].ql;
            auto qh = x[ibl].qh;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                float dl1 = d * x[ibl].scales[8*ib+k+0];
                float dl2 = d * x[ibl].scales[8*ib+k+4];
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * (((ql[4*k+i+ 0] & 0xf) | ((qh[4*k+i+ 0] << 4) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * (((ql[4*k+i+ 0] >>  4) | ((qh[4*k+i+ 0] << 2) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * (((ql[4*k+i+16] & 0xf) | ((qh[4*k+i+16] << 4) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * (((ql[4*k+i+16] >>  4) | ((qh[4*k+i+16] << 2) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * (((ql[4*k+i+32] & 0xf) | ((qh[4*k+i+ 0] >> 0) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * (((ql[4*k+i+32] >>  4) | ((qh[4*k+i+ 0] >> 2) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * (((ql[4*k+i+48] & 0xf) | ((qh[4*k+i+16] >> 0) & 0x30)) - 32);
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * (((ql[4*k+i+48] >>  4) | ((qh[4*k+i+16] >> 2) & 0x30)) - 32);
                }
                ql += 64;
                qh += 32;
            }
        }
    }
}

void vec_dot_q6_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q6_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}


//
// ========================================= q5_k_r4
//

void quantize_row_q5_k_r4_ref(const float * x, block_q5_k_r4 * y, int64_t k) {
    quantize_q5_k_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_q5_k_r4(const float * x, void * y, int64_t k) {
    quantize_q5_k_r4(x, y, 4, k/4, nullptr);
}

namespace {
inline void convert_q5_k(const block_q5_K& x, uint8_t * L, uint8_t * Ld, uint8_t * Lm) {
    for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
        get_scale_min_k4(2*ib64+0, x.scales, Ld[2*ib64+0], Lm[2*ib64+0]);
        get_scale_min_k4(2*ib64+1, x.scales, Ld[2*ib64+1], Lm[2*ib64+1]);
        for (int j = 0; j < 32; ++j) {
            L[64*ib64+j+ 0] = (x.qs[32*ib64+j] & 0xf) | (((x.qh[j] >> (2*ib64+0)) & 1) << 4);
            L[64*ib64+j+32] = (x.qs[32*ib64+j] >>  4) | (((x.qh[j] >> (2*ib64+1)) & 1) << 4);
        }
    }
}
}

static void repack_q5_k(int nrows, int n_per_row, const block_q5_K * x, block_q5_k_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q5_K * x4[4];
    uint8_t L[QK_K], Ld[QK_K/32], Lm[QK_K/32];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k+0] = x4[k][ibl].d;
                y[ibl].d[k+4] = x4[k][ibl].dmin;
                convert_q5_k(x4[k][ibl], L, Ld, Lm);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales_l[4*ib+k] = (Ld[ib] & 0xf) | ((Lm[ib] & 0xf) << 4);
                    uint8_t h = (Ld[ib] >> 4) | ((Lm[ib] >> 4) << 2);
                    y[ibl].scales_h[(4*ib+k)%16] |= (h << 4*((4*ib+k)/16));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[64*ib+4*k+i+ 0] = (L[32*ib+i+ 0] & 0xf) | ((L[32*ib+i+ 8] & 0xf) << 4);
                        y[ibl].qs[64*ib+4*k+i+16] = (L[32*ib+i+16] & 0xf) | ((L[32*ib+i+24] & 0xf) << 4);
                        y[ibl].qs[64*ib+4*k+i+32] = (L[32*ib+i+ 4] & 0xf) | ((L[32*ib+i+12] & 0xf) << 4);
                        y[ibl].qs[64*ib+4*k+i+48] = (L[32*ib+i+20] & 0xf) | ((L[32*ib+i+28] & 0xf) << 4);
                        y[ibl].qh[16*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] >> 4) << 0) | ((L[32*ib+i+ 8] >> 4) << 1) | ((L[32*ib+i+ 4] >> 4) << 2) | ((L[32*ib+i+12] >> 4) << 3) |
                                                    ((L[32*ib+i+16] >> 4) << 4) | ((L[32*ib+i+24] >> 4) << 5) | ((L[32*ib+i+20] >> 4) << 6) | ((L[32*ib+i+28] >> 4) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_q5_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_Q5_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_q5_K(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_q5_k(4, n_per_row, (const block_q5_K *)qtmp.data(), (block_q5_k_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_q5_k_r4(const block_q5_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k+0]);
            const float m = GGML_FP16_TO_FP32(x[ibl].d[k+4]);
            auto ql = x[ibl].qs;
            auto qh = x[ibl].qh;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 4*ib + k;
                float dl = d * ((x[ibl].scales_l[is] & 0xf) | (((x[ibl].scales_h[is%16] >> 4*(is/16)) & 0x03) << 4));
                float ml = m * ((x[ibl].scales_l[is] >>  4) | (((x[ibl].scales_h[is%16] >> 4*(is/16)) & 0x0c) << 2));
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl * ((ql[4*k+i+ 0] & 0xf) | ((qh[4*k+i] << 4) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl * ((ql[4*k+i+ 0] >>  4) | ((qh[4*k+i] << 3) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+16] = dl * ((ql[4*k+i+16] & 0xf) | ((qh[4*k+i] >> 0) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+24] = dl * ((ql[4*k+i+16] >>  4) | ((qh[4*k+i] >> 1) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl * ((ql[4*k+i+32] & 0xf) | ((qh[4*k+i] << 2) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+12] = dl * ((ql[4*k+i+32] >>  4) | ((qh[4*k+i] << 1) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+20] = dl * ((ql[4*k+i+48] & 0xf) | ((qh[4*k+i] >> 2) & 0x10)) - ml;
                    y4[k][QK_K*ibl+32*ib+i+28] = dl * ((ql[4*k+i+48] >>  4) | ((qh[4*k+i] >> 3) & 0x10)) - ml;
                }
                ql += 64;
                qh += 16;
            }
        }
    }
}

void vec_dot_q5_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q5_K_R4, vx, 0, GGML_TYPE_Q8_K32, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= q3_k_r4
//

void quantize_row_q3_k_r4_ref(const float * x, block_q3_k_r4 * y, int64_t k) {
    quantize_q3_k_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_q3_k_r4(const float * x, void * y, int64_t k) {
    quantize_q3_k_r4(x, y, 4, k/4, nullptr);
}

namespace {
inline void convert_q3_k(const block_q3_K& x, uint8_t * L, uint8_t * Ld) {
    constexpr uint32_t kmask1 = 0x03030303;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    memcpy(aux, x.scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    std::memcpy(Ld, aux, 16);

    const uint8_t * q = x.qs;
    const uint8_t * hm = x.hmask;
    uint8_t m = 1;
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            for (int l = 0; l < 32; ++l) {
                *L++ = ((q[l] >> shift) & 3) + ((hm[l] & m) ? 4 : 0);
            }
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}
}

static void repack_q3_k(int nrows, int n_per_row, const block_q3_K * x, block_q3_k_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q3_K * x4[4];
    uint8_t L[QK_K], Ld[QK_K/16];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                convert_q3_k(x4[k][ibl], L, Ld);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    int is = 8*ib+k;
                    y[ibl].scales_l[is%32] |= (Ld[2*ib+0] & 0xf) << 4*(is/32);
                    y[ibl].scales_h[is%16] |= (Ld[2*ib+0] >>  4) << 2*(is/16);
                    is += 4;
                    y[ibl].scales_l[is%32] |= (Ld[2*ib+1] & 0xf) << 4*(is/32);
                    y[ibl].scales_h[is%16] |= (Ld[2*ib+1] >>  4) << 2*(is/16);
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] & 0x3) << 0) | ((L[32*ib+i+ 4] & 0x3) << 2) | ((L[32*ib+i+ 8] & 0x3) << 4) | ((L[32*ib+i+12] & 0x3) << 6);
                        y[ibl].qs[32*ib+4*k+i+16] = ((L[32*ib+i+16] & 0x3) << 0) | ((L[32*ib+i+20] & 0x3) << 2) | ((L[32*ib+i+24] & 0x3) << 4) | ((L[32*ib+i+28] & 0x3) << 6);
                        y[ibl].qh[16*ib+4*k+i+ 0] = ((L[32*ib+i+ 0]  >> 2) << 0) | ((L[32*ib+i+ 4]  >> 2) << 1) | ((L[32*ib+i+ 8]  >> 2) << 2) | ((L[32*ib+i+12]  >> 2) << 3)
                                                  | ((L[32*ib+i+16]  >> 2) << 4) | ((L[32*ib+i+20]  >> 2) << 5) | ((L[32*ib+i+24]  >> 2) << 6) | ((L[32*ib+i+28]  >> 2) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_q3_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_Q3_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_q3_K(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_q3_k(4, n_per_row, (const block_q3_K *)qtmp.data(), (block_q3_k_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_q3_k_r4(const block_q3_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            auto ql = x[ibl].qs;
            auto qh = x[ibl].qh;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl1 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 0x03) << 4)) - 32);
                is += 4;
                float dl2 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 0x03) << 4)) - 32);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * ((((ql[4*k+i+ 0] >> 0) & 3) | ((qh[4*k+i] << 2) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * ((((ql[4*k+i+ 0] >> 2) & 3) | ((qh[4*k+i] << 1) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * ((((ql[4*k+i+ 0] >> 4) & 3) | ((qh[4*k+i] << 0) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * ((((ql[4*k+i+ 0] >> 6) & 3) | ((qh[4*k+i] >> 1) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * ((((ql[4*k+i+16] >> 0) & 3) | ((qh[4*k+i] >> 2) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * ((((ql[4*k+i+16] >> 2) & 3) | ((qh[4*k+i] >> 3) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * ((((ql[4*k+i+16] >> 4) & 3) | ((qh[4*k+i] >> 4) & 4)) - 4);
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * ((((ql[4*k+i+16] >> 6) & 3) | ((qh[4*k+i] >> 5) & 4)) - 4);
                }
                ql += 32;
                qh += 16;
            }
        }
    }
}

void vec_dot_q3_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q3_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= q2_k_r4
//

void quantize_row_q2_k_r4_ref(const float * x, block_q2_k_r4 * y, int64_t k) {
    quantize_q3_k_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_q2_k_r4(const float * x, void * y, int64_t k) {
    quantize_q2_k_r4(x, y, 4, k/4, nullptr);
}

namespace {
inline void convert_q2_k(const block_q2_K& x, uint8_t * L) {

    const uint8_t * qs = x.qs;
    for (int n = 0; n < QK_K; n += 128) {
        for (int j = 0; j < 32; ++j) {
            L[n + j +  0] = (qs[j] >> 0) & 0x3;
            L[n + j + 32] = (qs[j] >> 2) & 0x3;
            L[n + j + 64] = (qs[j] >> 4) & 0x3;
            L[n + j + 96] = (qs[j] >> 6) & 0x3;
        }
        qs += 32;
    }
}
}

static void repack_q2_k(int nrows, int n_per_row, const block_q2_K * x, block_q2_k_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q2_K * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k+0] = x4[k][ibl].d;
                y[ibl].d[k+4] = x4[k][ibl].dmin;
                for (int ib = 0; ib < QK_K/16; ++ib) {
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                }
                convert_q2_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] & 0x3) << 0) | ((L[32*ib+i+ 4] & 0x3) << 2) | ((L[32*ib+i+ 8] & 0x3) << 4) | ((L[32*ib+i+12] & 0x3) << 6);
                        y[ibl].qs[32*ib+4*k+i+16] = ((L[32*ib+i+16] & 0x3) << 0) | ((L[32*ib+i+20] & 0x3) << 2) | ((L[32*ib+i+24] & 0x3) << 4) | ((L[32*ib+i+28] & 0x3) << 6);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_q2_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_Q2_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_q2_K(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_q2_k(4, n_per_row, (const block_q2_K *)qtmp.data(), (block_q2_k_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_q2_k_r4(const block_q2_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k+0]);
            const float m = GGML_FP16_TO_FP32(x[ibl].d[k+4]);
            auto ql = x[ibl].qs;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                float dl1 = d * (x[ibl].scales[8*ib + k + 0] & 0xf);
                float ml1 = m * (x[ibl].scales[8*ib + k + 0] >>  4);
                float dl2 = d * (x[ibl].scales[8*ib + k + 4] & 0xf);
                float ml2 = m * (x[ibl].scales[8*ib + k + 4] >>  4);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * ((ql[4*k+i+ 0] >> 0) & 3) - ml1;
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * ((ql[4*k+i+ 0] >> 2) & 3) - ml1;
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * ((ql[4*k+i+ 0] >> 4) & 3) - ml1;
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * ((ql[4*k+i+ 0] >> 6) & 3) - ml1;
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * ((ql[4*k+i+16] >> 0) & 3) - ml2;
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * ((ql[4*k+i+16] >> 2) & 3) - ml2;
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * ((ql[4*k+i+16] >> 4) & 3) - ml2;
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * ((ql[4*k+i+16] >> 6) & 3) - ml2;
                }
                ql += 32;
            }
        }
    }
}

void vec_dot_q2_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q2_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= iq4_k_r4
//

void quantize_row_iq4_k_r4_ref(const float * x, block_iq4_k_r4 * y, int64_t k) {
    quantize_iq4_k_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq4_k_r4(const float * x, void * y, int64_t k) {
    quantize_iq4_k_r4(x, y, 4, k/4, nullptr);
}

static void repack_iq4_k(int nrows, int n_per_row, const block_iq4_k * x, block_iq4_k_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq4_k * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].extra, 0, 8);
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                auto extra = x4[k][ibl].extra;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    if (extra & 1) y[ibl].extra[k+0] |= (1 << ib);
                    if (extra & 2) y[ibl].extra[k+4] |= (1 << ib);
                    extra >>= 2;
                    uint8_t sl1 = x4[k][ibl].scales_l[ib] & 0xf;
                    uint8_t sl2 = x4[k][ibl].scales_l[ib] >>  4;
                    uint8_t sh  = x4[k][ibl].scales_h[ib/2] >> 4*(ib%2);
                    uint8_t sh1 = (sh >> 0) & 3;
                    uint8_t sh2 = (sh >> 2) & 3;
                    int i = 8*ib + k;
                    y[ibl].scales_l[i%32] |= (sl1 << 4*(i/32));
                    y[ibl].scales_h[i%16] |= (sh1 << 2*(i/16));
                    i += 4;
                    y[ibl].scales_l[i%32] |= (sl2 << 4*(i/32));
                    y[ibl].scales_h[i%16] |= (sh2 << 2*(i/16));
                }
            }
            for (int ib = 0; ib < QK_K/32; ++ib) {
                for (int k = 0; k < 4; ++k) for (int i = 0; i < 4; ++i) {
                    y[ibl].qs[64*ib+4*k+i+ 0] = (x4[k][ibl].qs[16*ib+i+0] & 0xf) | ((x4[k][ibl].qs[16*ib+i+ 8] & 0x0f) << 4);  //  0....3 +  8...11 from each row
                    y[ibl].qs[64*ib+4*k+i+16] = (x4[k][ibl].qs[16*ib+i+0] >>  4) | ((x4[k][ibl].qs[16*ib+i+ 8] & 0xf0));       // 16...19 + 24...27 from each row
                    y[ibl].qs[64*ib+4*k+i+32] = (x4[k][ibl].qs[16*ib+i+4] & 0xf) | ((x4[k][ibl].qs[16*ib+i+12] & 0x0f) << 4);  //  4....7 + 12...15 from each row
                    y[ibl].qs[64*ib+4*k+i+48] = (x4[k][ibl].qs[16*ib+i+4] >>  4) | ((x4[k][ibl].qs[16*ib+i+12] & 0xf0));       // 20...23 + 28...31 from each row
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_iq4_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq4_k(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_iq4_k(4, n_per_row, (const block_iq4_k *)qtmp.data(), (block_iq4_k_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq4_k_r4(const block_iq4_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl1 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 3) << 4)) - 32);
                is += 4;
                float dl2 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 3) << 4)) - 32);
                auto values1 = iq4k_values + (x[ibl].extra[k+0] & (1 << ib) ? 16 : 0);
                auto values2 = iq4k_values + (x[ibl].extra[k+4] & (1 << ib) ? 16 : 0);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * values1[x[ibl].qs[64*ib+4*k+i+ 0] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * values1[x[ibl].qs[64*ib+4*k+i+ 0] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * values2[x[ibl].qs[64*ib+4*k+i+16] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * values2[x[ibl].qs[64*ib+4*k+i+16] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * values1[x[ibl].qs[64*ib+4*k+i+32] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * values1[x[ibl].qs[64*ib+4*k+i+32] >>  4];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * values2[x[ibl].qs[64*ib+4*k+i+48] & 0xf];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * values2[x[ibl].qs[64*ib+4*k+i+48] >>  4];
                }
            }
        }
    }
}

void vec_dot_iq4_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= iq5_k_r4
//

void quantize_row_iq5_k_r4_ref(const float * x, block_iq5_k_r4 * y, int64_t k) {
    quantize_iq5_k_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq5_k_r4(const float * x, void * y, int64_t k) {
    quantize_iq5_k_r4(x, y, 4, k/4, nullptr);
}

namespace {
inline void convert_iq5_k(const block_iq5_k& x, uint8_t * L) {
    const uint8_t * qs = x.qs;
    const uint8_t * qh = x.qh;
    int shift = 0;
    for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
        for (int j = 0; j < 16; ++j) {
            L[j+ 0] = (qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4);
            L[j+16] = (qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4);
            L[j+32] = (qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3);
            L[j+48] = (qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3);
        }
        L  += 64;
        qs += 32;
        shift += 2;
        if (shift == 8) { qh += 32; shift = 0; }
    }
}
}

static void repack_iq5_k(int nrows, int n_per_row, const block_iq5_k * x, block_iq5_k_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq5_k * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].extra, 0, 8);
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                auto extra = x4[k][ibl].extra;
                convert_iq5_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    if (extra & 1) y[ibl].extra[k+0] |= (1 << ib);
                    if (extra & 2) y[ibl].extra[k+4] |= (1 << ib);
                    extra >>= 2;
                    uint8_t sl1 = x4[k][ibl].scales_l[ib] & 0xf;
                    uint8_t sl2 = x4[k][ibl].scales_l[ib] >>  4;
                    uint8_t sh  = x4[k][ibl].scales_h[ib/2] >> 4*(ib%2);
                    uint8_t sh1 = (sh >> 0) & 3;
                    uint8_t sh2 = (sh >> 2) & 3;
                    int i = 8*ib + k;
                    y[ibl].scales_l[i%32] |= (sl1 << 4*(i/32));
                    y[ibl].scales_h[i%16] |= (sh1 << 2*(i/16));
                    i += 4;
                    y[ibl].scales_l[i%32] |= (sl2 << 4*(i/32));
                    y[ibl].scales_h[i%16] |= (sh2 << 2*(i/16));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[64*ib+4*k+i+ 0] = (L[32*ib+i+ 0] & 0xf) | ((L[32*ib+i+ 8] & 0xf) << 4);  //  0....3 +  8...11 from each row
                        y[ibl].qs[64*ib+4*k+i+16] = (L[32*ib+i+16] & 0xf) | ((L[32*ib+i+24] & 0xf) << 4);  // 16...19 + 24...27 from each row
                        y[ibl].qs[64*ib+4*k+i+32] = (L[32*ib+i+ 4] & 0xf) | ((L[32*ib+i+12] & 0xf) << 4);  //  4....7 + 12...15 from each row
                        y[ibl].qs[64*ib+4*k+i+48] = (L[32*ib+i+20] & 0xf) | ((L[32*ib+i+28] & 0xf) << 4);  // 20...23 + 28...31 from each row
                        y[ibl].qh[16*ib+4*k+i   ] = ((L[32*ib+i+ 0] >> 4) << 0) | ((L[32*ib+i+ 8] >> 4) << 1) | ((L[32*ib+i+16] >> 4) << 2) | ((L[32*ib+i+24] >> 4) << 3)
                                                  | ((L[32*ib+i+ 4] >> 4) << 4) | ((L[32*ib+i+12] >> 4) << 5) | ((L[32*ib+i+20] >> 4) << 6) | ((L[32*ib+i+28] >> 4) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_iq5_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq5_k(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_iq5_k(4, n_per_row, (const block_iq5_k *)qtmp.data(), (block_iq5_k_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq5_k_r4(const block_iq5_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl1 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 3) << 4)) - 32);
                is += 4;
                float dl2 = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 3) << 4)) - 32);
                auto values1 = iq5nl_values + (x[ibl].extra[k+0] & (1 << ib) ? 32 : 0);
                auto values2 = iq5nl_values + (x[ibl].extra[k+4] & (1 << ib) ? 32 : 0);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * values1[(x[ibl].qs[64*ib+4*k+i+ 0] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 0) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * values1[(x[ibl].qs[64*ib+4*k+i+ 0] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 1) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * values2[(x[ibl].qs[64*ib+4*k+i+16] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 2) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * values2[(x[ibl].qs[64*ib+4*k+i+16] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 3) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * values1[(x[ibl].qs[64*ib+4*k+i+32] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 4) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * values1[(x[ibl].qs[64*ib+4*k+i+32] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 5) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * values2[(x[ibl].qs[64*ib+4*k+i+48] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 6) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * values2[(x[ibl].qs[64*ib+4*k+i+48] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 7) & 1) << 4)];
                }
            }
        }
    }
}

void vec_dot_iq5_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ5_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= q8_k_r8
//

void quantize_row_q8_k_r8_ref(const float * x, block_q8_k_r8 * y, int64_t k) {
    quantize_q8_k_r8(x, (void *)y, 8, k/8, nullptr);
}

void quantize_row_q8_k_r8(const float * x, void * y, int64_t k) {
    quantize_q8_k_r8(x, y, 8, k/8, nullptr);
}

static void repack_q8_k(int nrows, int n_per_row, const block_q8_K * x, block_q8_k_r8 * y) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q8_K * x8[8];
    for (int row = 0; row < nrows; row += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 8; ++k) {
                y[ibl].d[k] = GGML_FP32_TO_FP16(x8[k][ibl].d);
                for (int ib = 0; ib < QK_K/4; ++ib) {
                    for (int i = 0; i < 4; ++i) y[ibl].qs[32*ib + 4*k + i] = x8[k][ibl].qs[4*ib+i];
                }
            }
        }
        x += 8*nblock;
        y += nblock;
    }
}

size_t quantize_q8_k_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row, [[maybe_unused]] const float * imatrix) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q8_K, n_per_row);
    auto row_size_1 = ggml_row_size(GGML_TYPE_Q8_K_R8, n_per_row);
    std::vector<char> qtmp(8*row_size_0);
    for (int row = 0; row < nrows; row += 8) {
        quantize_row_q8_K32(src, (void *)qtmp.data(), 8*n_per_row);
        repack_q8_k(8, n_per_row, (const block_q8_K *)qtmp.data(), (block_q8_k_r8 *)qcur);
        qcur += 8*row_size_1;
        src += 8*n_per_row;
    }
    return nrows*row_size_1;
}

void dequantize_row_q8_k_r8(const block_q8_k_r8 * x, float * y, int64_t k) {
    auto n_per_row = k/8;
    float * y8[8];
    for (int k = 0; k < 8; ++k) y8[k] = y + n_per_row*k;
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 8; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/4; ++ib) {
                for (int i = 0; i < 4; ++i) {
                    y8[k][QK_K*ibl+4*ib+i] = d * x[ibl].qs[32*ib+4*k+i];
                }
            }
        }
    }
}

void vec_dot_q8_k_r8_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q8_K_R8, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= bf16_r4
//
namespace {
inline ggml_bf16_t to_bf16(const float& x) {
    union { float f; uint32_t u; } helper;
    helper.f = x;
    return ggml_bf16_t{(uint16_t)(helper.u >> 16)};
}
inline ggml_bf16_t to_bf16(const ggml_bf16_t& x) { return x; }
template <typename T>
void repack_bf16(int nrows, int n_per_row, const T * x, ggml_bf16_t * y) {
    GGML_ASSERT(nrows%16 == 0);
    GGML_ASSERT(n_per_row%2 == 0);
    for (int row = 0; row < nrows; row += 16) {
        for (int k = 0; k < 16; ++k) {
            auto x8 = x + k*n_per_row;
            for (int ib = 0; ib < n_per_row/2; ++ib) {
                y[32*ib + 2*k + 0] = to_bf16(x8[2*ib+0]);
                y[32*ib + 2*k + 1] = to_bf16(x8[2*ib+1]);
            }
        }
        x += 16*n_per_row;
        y += 16*n_per_row;
    }
}
}

void repack_f32_bf16_r16(const void * src, void * dst, int64_t nrows, int64_t n_per_row) {
    repack_bf16(nrows, n_per_row, (const float *)src, (ggml_bf16_t *)dst);
}

void repack_bf16_bf16_r16(const void * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row) {
    repack_bf16(nrows, n_per_row, (const ggml_bf16_t *)src, (ggml_bf16_t *)dst);
}

//
// ========================================= iq3_k_r4
//

void quantize_row_iq3_k_r4_ref(const float * x, block_iq3_k_r4 * y, int64_t k) {
    quantize_iq3_k_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq3_k_r4(const float * x, void * y, int64_t k) {
    quantize_iq3_k_r4(x, y, 4, k/4, nullptr);
}

namespace {
inline void convert_iq3_k(const block_iq3_k& x, uint8_t * L) {
    const uint8_t * qs = x.qs;
    const uint8_t * qh = x.qh;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        int shift_l = 2*(ib32%4);
        int shift_h = ib32%8;
        for (int j = 0; j < 16; ++j) {
            L[j+ 0] = ((qs[j+ 0] >> shift_l) & 3) | (((qh[j+ 0] >> shift_h) & 1) << 2);
            L[j+16] = ((qs[j+16] >> shift_l) & 3) | (((qh[j+16] >> shift_h) & 1) << 2);
        }
        L += 32;
        if (shift_l == 6) qs += 32;
    }
}
}

static void repack_iq3_k(int nrows, int n_per_row, const block_iq3_k * x, block_iq3_k_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq3_k * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].extra, 0, 8);
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/32);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                auto extra = x4[k][ibl].extra;
                uint16_t sh  = x4[k][ibl].scales_h;
                convert_iq3_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    if (extra & 1) y[ibl].extra[k+0] |= (1 << ib);
                    if (extra & 2) y[ibl].extra[k+4] |= (1 << ib);
                    extra >>= 2;
                    uint8_t sl1 = x4[k][ibl].scales_l[ib] & 0xf;
                    uint8_t sl2 = x4[k][ibl].scales_l[ib] >>  4;
                    uint8_t sh1 = (sh >> 0) & 1;
                    uint8_t sh2 = (sh >> 1) & 1;
                    sh >>= 2;
                    int i = 8*ib + k;
                    y[ibl].scales_l[i%32] |= (sl1 << 4*(i/32));
                    y[ibl].scales_h[i%8 ] |= (sh1 << (i/8));
                    i += 4;
                    y[ibl].scales_l[i%32] |= (sl2 << 4*(i/32));
                    y[ibl].scales_h[i%8 ] |= (sh2 << (i/8));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] & 0x3) << 0) | ((L[32*ib+i+ 4] & 0x3) << 2) | ((L[32*ib+i+ 8] & 0x3) << 4) | ((L[32*ib+i+12] & 0x3) << 6);
                        y[ibl].qs[32*ib+4*k+i+16] = ((L[32*ib+i+16] & 0x3) << 0) | ((L[32*ib+i+20] & 0x3) << 2) | ((L[32*ib+i+24] & 0x3) << 4) | ((L[32*ib+i+28] & 0x3) << 6);
                        y[ibl].qh[16*ib+4*k+i+ 0] = ((L[32*ib+i+ 0]  >> 2) << 0) | ((L[32*ib+i+ 4]  >> 2) << 1) | ((L[32*ib+i+ 8]  >> 2) << 2) | ((L[32*ib+i+12]  >> 2) << 3)
                                                  | ((L[32*ib+i+16]  >> 2) << 4) | ((L[32*ib+i+20]  >> 2) << 5) | ((L[32*ib+i+24]  >> 2) << 6) | ((L[32*ib+i+28]  >> 2) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_iq3_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq3_k(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_iq3_k(4, n_per_row, (const block_iq3_k *)qtmp.data(), (block_iq3_k_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq3_k_r4(const block_iq3_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            auto ql = x[ibl].qs;
            auto qh = x[ibl].qh;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl1 = d * (2*((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) + 1) * ((x[ibl].scales_h[is%8] >> (is/8)) & 1 ? -1 : 1);
                is += 4;
                float dl2 = d * (2*((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) + 1) * ((x[ibl].scales_h[is%8] >> (is/8)) & 1 ? -1 : 1);
                auto values1 = iq3nl_values + (x[ibl].extra[k+0] & (1 << ib) ? 8 : 0);
                auto values2 = iq3nl_values + (x[ibl].extra[k+4] & (1 << ib) ? 8 : 0);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * values1[((ql[4*k+i+ 0] >> 0) & 3) | ((qh[4*k+i] << 2) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * values1[((ql[4*k+i+ 0] >> 2) & 3) | ((qh[4*k+i] << 1) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * values1[((ql[4*k+i+ 0] >> 4) & 3) | ((qh[4*k+i] << 0) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * values1[((ql[4*k+i+ 0] >> 6) & 3) | ((qh[4*k+i] >> 1) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * values2[((ql[4*k+i+16] >> 0) & 3) | ((qh[4*k+i] >> 2) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * values2[((ql[4*k+i+16] >> 2) & 3) | ((qh[4*k+i] >> 3) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * values2[((ql[4*k+i+16] >> 4) & 3) | ((qh[4*k+i] >> 4) & 4)];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * values2[((ql[4*k+i+16] >> 6) & 3) | ((qh[4*k+i] >> 5) & 4)];
                }
                ql += 32;
                qh += 16;
            }
        }
    }
}

void vec_dot_iq3_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= iq2_k_r4
//

void quantize_row_iq2_k_r4_ref(const float * x, block_iq2_k_r4 * y, int64_t k) {
    quantize_iq2_k_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq2_k_r4(const float * x, void * y, int64_t k) {
    quantize_iq2_k_r4(x, y, 4, k/4, nullptr);
}

namespace {
inline void convert_iq2_k(const block_iq2_k& x, uint8_t * L) {
    const uint8_t * qs = x.qs;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        int shift_l = 2*(ib32%4);
        for (int j = 0; j < 16; ++j) {
            L[j+ 0] = ((qs[j+ 0] >> shift_l) & 3);
            L[j+16] = ((qs[j+16] >> shift_l) & 3);
        }
        L += 32;
        if (shift_l == 6) qs += 32;
    }
}
}

static void repack_iq2_k(int nrows, int n_per_row, const block_iq2_k * x, block_iq2_k_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_k * x4[4];
    uint8_t L[QK_K];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].extra, 0, 8);
            std::memset(y[ibl].scales, 0, QK_K/8);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                auto extra = x4[k][ibl].extra;
                convert_iq2_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    if (extra & 1) y[ibl].extra[k+0] |= (1 << ib);
                    if (extra & 2) y[ibl].extra[k+4] |= (1 << ib);
                    extra >>= 2;
                    uint8_t sl1 = x4[k][ibl].scales[ib] & 0xf;
                    uint8_t sl2 = x4[k][ibl].scales[ib] >>  4;
                    int i = 8*ib + k;
                    y[ibl].scales[i%32] |= (sl1 << 4*(i/32));
                    i += 4;
                    y[ibl].scales[i%32] |= (sl2 << 4*(i/32));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+4*k+i+ 0] = ((L[32*ib+i+ 0] & 0x3) << 0) | ((L[32*ib+i+ 4] & 0x3) << 2) | ((L[32*ib+i+ 8] & 0x3) << 4) | ((L[32*ib+i+12] & 0x3) << 6);
                        y[ibl].qs[32*ib+4*k+i+16] = ((L[32*ib+i+16] & 0x3) << 0) | ((L[32*ib+i+20] & 0x3) << 2) | ((L[32*ib+i+24] & 0x3) << 4) | ((L[32*ib+i+28] & 0x3) << 6);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_iq2_k_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_K, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq2_k(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_iq2_k(4, n_per_row, (const block_iq2_k *)qtmp.data(), (block_iq2_k_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq2_k_r4(const block_iq2_k_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            auto ql = x[ibl].qs;
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl1 = d * (((x[ibl].scales[is%32] >> 4*(is/32)) & 0xf) - 8);
                is += 4;
                float dl2 = d * (((x[ibl].scales[is%32] >> 4*(is/32)) & 0xf) - 8);
                auto values1 = iq2nl_values + (x[ibl].extra[k+0] & (1 << ib) ? 4 : 0);
                auto values2 = iq2nl_values + (x[ibl].extra[k+4] & (1 << ib) ? 4 : 0);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl1 * values1[(ql[4*k+i+ 0] >> 0) & 3];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl1 * values1[(ql[4*k+i+ 0] >> 2) & 3];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl1 * values1[(ql[4*k+i+ 0] >> 4) & 3];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl1 * values1[(ql[4*k+i+ 0] >> 6) & 3];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl2 * values2[(ql[4*k+i+16] >> 0) & 3];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl2 * values2[(ql[4*k+i+16] >> 2) & 3];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl2 * values2[(ql[4*k+i+16] >> 4) & 3];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl2 * values2[(ql[4*k+i+16] >> 6) & 3];
                }
                ql += 32;
            }
        }
    }
}

void vec_dot_iq2_k_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_K_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

namespace {
struct Repack {
    using repack_func = void (*) (int nrows, int n_per_row, const char * src, char * dst);
    ggml_type   new_type;
    int         num_rows;
    repack_func repack;
};
}

namespace {
inline uint8_t scrambled_sign(uint8_t s) {
    static const uint8_t k_table[128] = {
        0x00, 0x7f, 0x7e, 0x01, 0x7c, 0x03, 0x02, 0x7d, 0x78, 0x07, 0x06, 0x79, 0x04, 0x7b, 0x7a, 0x05,
        0x70, 0x0f, 0x0e, 0x71, 0x0c, 0x73, 0x72, 0x0d, 0x08, 0x77, 0x76, 0x09, 0x74, 0x0b, 0x0a, 0x75,
        0x60, 0x1f, 0x1e, 0x61, 0x1c, 0x63, 0x62, 0x1d, 0x18, 0x67, 0x66, 0x19, 0x64, 0x1b, 0x1a, 0x65,
        0x10, 0x6f, 0x6e, 0x11, 0x6c, 0x13, 0x12, 0x6d, 0x68, 0x17, 0x16, 0x69, 0x14, 0x6b, 0x6a, 0x15,
        0x40, 0x3f, 0x3e, 0x41, 0x3c, 0x43, 0x42, 0x3d, 0x38, 0x47, 0x46, 0x39, 0x44, 0x3b, 0x3a, 0x45,
        0x30, 0x4f, 0x4e, 0x31, 0x4c, 0x33, 0x32, 0x4d, 0x48, 0x37, 0x36, 0x49, 0x34, 0x4b, 0x4a, 0x35,
        0x20, 0x5f, 0x5e, 0x21, 0x5c, 0x23, 0x22, 0x5d, 0x58, 0x27, 0x26, 0x59, 0x24, 0x5b, 0x5a, 0x25,
        0x50, 0x2f, 0x2e, 0x51, 0x2c, 0x53, 0x52, 0x2d, 0x28, 0x57, 0x56, 0x29, 0x54, 0x2b, 0x2a, 0x55,
    };
    return k_table[s];
}
}

//
// ========================================= iq2_xxs_r4
//

void quantize_row_iq2_xxs_r4_ref(const float * x, block_iq2_xxs_r4 * y, int64_t k) {
    quantize_iq2_xxs_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq2_xxs_r4(const float * x, void * y, int64_t k) {
    quantize_iq2_xxs_r4(x, y, 4, k/4, nullptr);
}

static void repack_iq2_xxs(int nrows, int n_per_row, const block_iq2_xxs * x, block_iq2_xxs_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_xxs * x4[4];
    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            auto ysas = (uint32_t *)y[ibl].sas;
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    std::memcpy(aux32, x4[k][ibl].qs + 4*ib, 2*sizeof(uint32_t));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[16*ib+4*k+i] = aux8[i];
                    }
                    uint8_t scale = aux32[1] >> 28;
                    uint8_t s1 = (scrambled_sign((aux32[1] >>  0) & 127) << 1) | ((scale >> 0) & 1);
                    uint8_t s2 = (scrambled_sign((aux32[1] >>  7) & 127) << 1) | ((scale >> 1) & 1);
                    uint8_t s3 = (scrambled_sign((aux32[1] >> 14) & 127) << 1) | ((scale >> 2) & 1);
                    uint8_t s4 = (scrambled_sign((aux32[1] >> 21) & 127) << 1) | ((scale >> 3) & 1);
                    aux32[1] = uint32_t(s1) | (uint32_t(s2) << 8) | (uint32_t(s3) << 16) | (uint32_t(s4) << 24);
                    ysas[4*ib+k] = aux32[1];
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_iq2_xxs_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_XXS, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq2_xxs(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_iq2_xxs(4, n_per_row, (const block_iq2_xxs *)qtmp.data(), (block_iq2_xxs_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq2_xxs_r4(const block_iq2_xxs_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    uint32_t s32;
    const uint8_t * s8 = (const uint8_t *)&s32;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const uint32_t * sas = (const uint32_t *)x[ibl].sas;
        for (int k = 0; k < 4; ++k) {
            const float d = 0.125f*GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                uint32_t aux32 = sas[4*ib+k];
                s32 = aux32 & 0x01010101;
                uint8_t scale = s8[0] | (s8[1] << 1) | (s8[2] << 2) | (s8[3] << 3);
                float dl = d*(2*scale+1);
                aux32 &= 0xfefefefe;
                aux32 ^= (aux32 >> 1);
                for (int i = 0; i < 4; ++i) {
                    auto val = (const int8_t *)(iq2xxs_grid + x[ibl].qs[16*ib+4*k+i]);
                    for (int j = 0; j < 8; ++j) y4[k][QK_K*ibl+32*ib+8*i+j] = dl * val[j] * (aux32 & (1 << j) ? -1 : 1);
                    aux32 >>= 8;
                }
            }
        }
    }
}

void vec_dot_iq2_xxs_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_XXS_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= iq2_xs_r4
//

void quantize_row_iq2_xs_r4_ref(const float * x, block_iq2_xs_r4 * y, int64_t k) {
    quantize_iq2_xs_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq2_xs_r4(const float * x, void * y, int64_t k) {
    quantize_iq2_xs_r4(x, y, 4, k/4, nullptr);
}

static void repack_iq2_xs(int nrows, int n_per_row, const block_iq2_xs * x, block_iq2_xs_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_xs * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    for (int i = 0; i < 4; ++i) {
                        uint16_t v = x4[k][ibl].qs[4*ib+i];
                        uint8_t s = v >> 9;
                        y[ibl].qs[16*ib+4*k+i] = (v & 511) | (scrambled_sign(s) << 9);
                    }
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_iq2_xs_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_XS, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq2_xs(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_iq2_xs(4, n_per_row, (const block_iq2_xs *)qtmp.data(), (block_iq2_xs_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq2_xs_r4(const block_iq2_xs_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = 0.125f*GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                float dl1 = d * (2*(x[ibl].scales[4*ib+k] & 0xf) + 1);
                float dl2 = d * (2*(x[ibl].scales[4*ib+k] >>  4) + 1);
                for (int i = 0; i < 4; ++i) {
                    auto val = (const int8_t *)(iq2xs_grid + (x[ibl].qs[16*ib+4*k+i] & 511));
                    auto signs = x[ibl].qs[16*ib+4*k+i] >> 9;
                    signs ^= (signs << 1);
                    float dl = i < 2 ? dl1 : dl2;
                    for (int j = 0; j < 8; ++j) y4[k][QK_K*ibl+32*ib+8*i+j] = dl * val[j] * (signs & (1 << j) ? -1 : 1);
                }
            }
        }
    }
}

void vec_dot_iq2_xs_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_XS_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= iq2_s_r4
//

void quantize_row_iq2_s_r4_ref(const float * x, block_iq2_s_r4 * y, int64_t k) {
    quantize_iq2_s_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq2_s_r4(const float * x, void * y, int64_t k) {
    quantize_iq2_s_r4(x, y, 4, k/4, nullptr);
}

static void repack_iq2_s(int nrows, int n_per_row, const block_iq2_s * x, block_iq2_s_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq2_s * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                auto signs = x4[k][ibl].qs + QK_K/8;
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[16*ib+4*k+i] = x4[k][ibl].qs[4*ib+i];
                        y[ibl].signs[16*ib+4*k+i] = signs[4*ib+i];
                    }
                    y[ibl].qh[4*ib+k] = x4[k][ibl].qh[ib];
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_iq2_s_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_S, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq2_s(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_iq2_s(4, n_per_row, (const block_iq2_s *)qtmp.data(), (block_iq2_s_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq2_s_r4(const block_iq2_s_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = 0.125f*GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                float dl1 = d * (2*(x[ibl].scales[4*ib+k] & 0xf) + 1);
                float dl2 = d * (2*(x[ibl].scales[4*ib+k] >>  4) + 1);
                for (int i = 0; i < 4; ++i) {
                    auto val = (const int8_t *)(iq2s_grid + (x[ibl].qs[16*ib+4*k+i] | ((x[ibl].qh[4*ib+k] << (8 - 2*i)) & 0x300)));
                    auto signs = x[ibl].signs[16*ib+4*k+i];
                    float dl = i < 2 ? dl1 : dl2;
                    for (int j = 0; j < 8; ++j) y4[k][QK_K*ibl+32*ib+8*i+j] = dl * val[j] * (signs & (1 << j) ? -1 : 1);
                }
            }
        }
    }
}

void vec_dot_iq2_s_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_S_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

//
// ========================================= iq3_xxs_r4
//

void quantize_row_iq3_xxs_r4_ref(const float * x, block_iq3_xxs_r4 * y, int64_t k) {
    quantize_iq3_xxs_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq3_xxs_r4(const float * x, void * y, int64_t k) {
    quantize_iq3_xxs_r4(x, y, 4, k/4, nullptr);
}

namespace {
}

static void repack_iq3_xxs(int nrows, int n_per_row, const block_iq3_xxs * x, block_iq3_xxs_r4 * y) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq3_xxs * x4[4];
    uint32_t aux32;
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            auto ysas = (uint32_t *)y[ibl].sas;
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                auto xsas = x4[k][ibl].qs + QK_K/4;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    for (int i = 0; i < 8; ++i) {
                        y[ibl].qs[32*ib+8*k+i] = x4[k][ibl].qs[8*ib+i];
                    }
                    std::memcpy(&aux32, xsas + 4*ib, 4);
                    uint8_t scale = aux32 >> 28;
                    uint8_t s1 = (scrambled_sign((aux32 >>  0) & 127) << 1) | ((scale >> 0) & 1);
                    uint8_t s2 = (scrambled_sign((aux32 >>  7) & 127) << 1) | ((scale >> 1) & 1);
                    uint8_t s3 = (scrambled_sign((aux32 >> 14) & 127) << 1) | ((scale >> 2) & 1);
                    uint8_t s4 = (scrambled_sign((aux32 >> 21) & 127) << 1) | ((scale >> 3) & 1);
                    aux32 = uint32_t(s1) | (uint32_t(s2) << 8) | (uint32_t(s3) << 16) | (uint32_t(s4) << 24);
                    ysas[4*ib+k] = aux32;
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_iq3_xxs_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_XXS, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq3_xxs(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_iq3_xxs(4, n_per_row, (const block_iq3_xxs *)qtmp.data(), (block_iq3_xxs_r4 *)qcur);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq3_xxs_r4(const block_iq3_xxs_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    uint32_t s32;
    const uint8_t * s8 = (const uint8_t *)&s32;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const uint32_t * sas = (const uint32_t *)x[ibl].sas;
        for (int k = 0; k < 4; ++k) {
            const float d = 0.25f*GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                uint32_t aux32 = sas[4*ib+k];
                s32 = aux32 & 0x01010101;
                uint8_t scale = s8[0] | (s8[1] << 1) | (s8[2] << 2) | (s8[3] << 3);
                float dl = d*(2*scale+1);
                aux32 &= 0xfefefefe;
                aux32 ^= (aux32 >> 1);
                for (int i = 0; i < 8; ++i) {
                    auto val = (const int8_t *)(iq3xxs_grid + x[ibl].qs[32*ib+8*k+i]);
                    for (int j = 0; j < 4; ++j) y4[k][QK_K*ibl+32*ib+4*i+j] = dl * val[j] * (aux32 & (1 << j) ? -1 : 1);
                    aux32 >>= 4;
                }
            }
        }
    }
}

void vec_dot_iq3_xxs_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_XXS_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

void iqk_repack_tensor(struct ggml_tensor * tensor) {
    constexpr int kChunk = 8;
    if (!tensor) return;
    if (!ggml_is_contiguous(tensor)) return;
    if (strncmp(tensor->name, "token_embd.weight", GGML_MAX_NAME) == 0) return;
    if (tensor->ne[1] % 4 || tensor->ne[2]*tensor->ne[3] > 1) return;
    static const std::unordered_map<ggml_type, Repack> k_map = {
        { GGML_TYPE_IQ2_K,  { GGML_TYPE_IQ2_K_R4,  4,  (Repack::repack_func)repack_iq2_k}   },
        { GGML_TYPE_IQ3_K,  { GGML_TYPE_IQ3_K_R4,  4,  (Repack::repack_func)repack_iq3_k}   },
        { GGML_TYPE_IQ4_K,  { GGML_TYPE_IQ4_K_R4,  4,  (Repack::repack_func)repack_iq4_k}   },
        { GGML_TYPE_IQ5_K,  { GGML_TYPE_IQ5_K_R4,  4,  (Repack::repack_func)repack_iq5_k}   },
        { GGML_TYPE_IQ4_XS, { GGML_TYPE_IQ4_XS_R4, 4,  (Repack::repack_func)repack_iq4_xs}  },
        { GGML_TYPE_IQ4_KS, { GGML_TYPE_IQ4_KS_R4, 4,  (Repack::repack_func)repack_iq4_ks}  },
        { GGML_TYPE_IQ4_NL, { GGML_TYPE_IQ4_NL_R4, 4,  (Repack::repack_func)repack_iq4_nl}  },
        { GGML_TYPE_IQ2_BN, { GGML_TYPE_IQ2_BN_R4, 4,  (Repack::repack_func)repack_iq2_bn}  },
        { GGML_TYPE_Q2_K,   { GGML_TYPE_Q2_K_R4,   4,  (Repack::repack_func)repack_q2_k}    },
        { GGML_TYPE_Q3_K,   { GGML_TYPE_Q3_K_R4,   4,  (Repack::repack_func)repack_q3_k}    },
        { GGML_TYPE_Q4_K,   { GGML_TYPE_Q4_K_R4,   4,  (Repack::repack_func)repack_q4_k}    },
        { GGML_TYPE_Q5_K,   { GGML_TYPE_Q5_K_R4,   4,  (Repack::repack_func)repack_q5_k}    },
        { GGML_TYPE_Q6_K,   { GGML_TYPE_Q6_K_R4,   4,  (Repack::repack_func)repack_q6_k}    },
        { GGML_TYPE_Q4_0,   { GGML_TYPE_Q4_0_R4,   4,  (Repack::repack_func)repack_q4_0}    },
        { GGML_TYPE_Q5_0,   { GGML_TYPE_Q5_0_R4,   4,  (Repack::repack_func)repack_q5_0}    },
        { GGML_TYPE_Q6_0,   { GGML_TYPE_Q6_0_R4,   4,  (Repack::repack_func)repack_q6_0}    },
        { GGML_TYPE_Q8_0,   { GGML_TYPE_Q8_0_R4,   4,  (Repack::repack_func)repack_q8_0}    },
        { GGML_TYPE_Q8_K,   { GGML_TYPE_Q8_K_R8,   8,  (Repack::repack_func)repack_q8_k}    },
#ifdef __AVX512BF16__
        { GGML_TYPE_BF16,   { GGML_TYPE_BF16_R16, 16,  (Repack::repack_func)repack_bf16<ggml_bf16_t>}    },
#endif
    };

    auto it = k_map.find(tensor->type);
    if (it == k_map.end()) return;
    if (tensor->ne[1] % it->second.num_rows) return;

    auto& r = it->second;

    int max_thread = std::max(1, int(std::thread::hardware_concurrency()/2));
    int num_chunks = (tensor->ne[1] + kChunk*r.num_rows - 1)/(kChunk*r.num_rows);
    int nthread = std::min(num_chunks, max_thread);

    //printf("%s(%s): %s -> %s. %d rows, %d chunks, %d threads\n", __func__, tensor->name, ggml_type_name(tensor->type), ggml_type_name(r.new_type),
    //        int(tensor->ne[1]), num_chunks, nthread);

    std::atomic<int> counter(0);;
    auto compute = [&counter, &r, tensor, num_chunks, chunkSize = kChunk] () {
        int nrows = tensor->ne[1];
        int n_per_row = tensor->ne[0];
        auto row_size = ggml_row_size(tensor->type, n_per_row);
        std::vector<char> qtmp(r.num_rows*row_size);
        auto data = (char *)tensor->data;
        while (true) {
            int chunk = counter.fetch_add(1);
            if (chunk >= num_chunks) break;
            int first_row = chunk*chunkSize*r.num_rows;
            int last_row = std::min(first_row + chunkSize*r.num_rows, nrows);
            for (int row = first_row; row < last_row; row += r.num_rows) {
                std::memcpy(qtmp.data(), data + row*row_size, r.num_rows*row_size);
                r.repack(r.num_rows, n_per_row, qtmp.data(), data + row*row_size);
            }
        }
    };
    std::vector<std::thread> workers(nthread-1);
    for (auto& w : workers) w = std::thread(compute);
    compute();
    for (auto& w : workers) w.join();

    tensor->type = r.new_type;
}

