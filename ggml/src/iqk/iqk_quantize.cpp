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
#include <random>
#include <memory>

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

void ggml_vec_dot_iq2_bn_q8_K64(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

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

void quantize_row_iq2_k_ref(const float * GGML_RESTRICT x, block_iq2_k  * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_k(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq2_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
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

void dequantize_row_iq2_k(const block_iq2_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
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

void vec_dot_iq2_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

void quantize_row_iq2_ks_ref(const float * GGML_RESTRICT x, block_iq2_ks * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_ks(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq2_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
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

void dequantize_row_iq2_ks(const block_iq2_ks  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
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

void vec_dot_iq3_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
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

#ifdef __AVX2__
namespace {
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

void iqk_quantize_row_q8_K(const float * x, void * vy, int64_t k) {
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
            y[i].bsums[2*ib+0] = hsum_i32_8(_mm256_add_epi32(i0, i1));
            y[i].bsums[2*ib+1] = hsum_i32_8(_mm256_add_epi32(i2, i3));
            i0 = _mm256_packs_epi32( i0, i1 );
            i2 = _mm256_packs_epi32( i2, i3 );
            i0 = _mm256_packs_epi16( i0, i2 );
            i0 = _mm256_permutevar8x32_epi32( i0, perm );
            _mm256_storeu_si256((__m256i *)q8, i0);
            q8 += 32;
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
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
#endif

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

template <typename T>
static inline int best_index(int n, const T * values, float x) {
    if (x <= values[0]) return 0;
    if (x >= values[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu - ml > 1) {
        int mav = (mu + ml)/2;
        if (x < values[mav]) mu = mav;
        else ml = mav;
    }
    return x - values[mu-1] < values[mu] - x ? mu - 1 : mu;
}

void quantize_row_iq4_kss_impl(int n_per_row, const float * xr, char * cy,
        const float * quant_weights,
        float * weights, int8_t * quants,
        float * scales) {
        //std::vector<float>& all_steps) {

    constexpr int kBlockSize = 64;
    auto values = iq4k_values + 16;

    //float * dptr = (float *)cy;
    //*dptr = 0;
    ggml_half * dh = (ggml_half *)cy;
    dh[0] = GGML_FP32_TO_FP16(0.f);

    block_iq4_kss * y = (block_iq4_kss *)(dh + 1);
    std::memset(y, 0, (n_per_row/QK_K)*sizeof(block_iq4_kss));

    float max_amax = 0;
    for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
        auto xb = xr + ib*kBlockSize;
        float amax = 0;
        for (int j = 0; j < kBlockSize; ++j) amax = std::max(amax, std::abs(xb[j]));
        scales[ib] = amax;
        max_amax = std::max(amax, max_amax);
    }
    if (!max_amax) {
        return;
    }
    float idm = 16/max_amax;
    for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
        int l = nearest_int(idm*scales[ib]);
        l = std::max(1, std::min(16, l));
        scales[ib] = l;
    }

    float max = 0, amax = 0, sigma2 = 0;
    for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
        auto xb = xr + ib*kBlockSize;
        float norm = 1/scales[ib];
        for (int j = 0; j < kBlockSize; ++j) {
            sigma2 += xb[j]*xb[j];
            float xs = norm*xb[j];
            float axs = std::abs(xs);
            if (axs > amax) {
                amax = axs; max = xs;
            }
        }
    }
    sigma2 *= 1.f/n_per_row;
    if (quant_weights) {
        for (int j = 0; j < n_per_row; ++j) weights[j] = quant_weights[j] * sqrt(sigma2 + xr[j]*xr[j]);
    } else {
        for (int j = 0; j < n_per_row; ++j) weights[j] = 0.25f*sigma2 + xr[j]*xr[j];
    }

    float best = 0, d = max/values[0];
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (values[0] + itry)/max;
        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
            auto xb = xr + ib*kBlockSize;
            auto wb = weights + ib*kBlockSize;
            float norm = 1/scales[ib];
            for (int j = 0; j < kBlockSize; ++j) {
                int idx = best_index_iq4nl(values, id*norm*xb[j]);
                float q = values[idx]*scales[ib];
                sumqx += wb[j]*q*xb[j];
                sumq2 += wb[j]*q*q;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }
    dh[0] = GGML_FP32_TO_FP16(d);
    if (!d) return;

    float id = 1/d;
    //float mse = 0;
    for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
        auto xb = xr + ib*kBlockSize;
        float norm = 1/scales[ib];
        for (int j = 0; j < kBlockSize; ++j) {
            int idx = best_index_iq4nl(values, id*norm*xb[j]);
            quants[ib*kBlockSize+j] = idx;
            //float diff = xb[j] - d*scales[ib]*values[idx];
            //mse += diff*diff;
        }
    }
    //printf("rmse = %g, %g\n", sqrt(mse/n_per_row), sqrt(2*mse/n_per_row/sigma2));

    for (int itry = 0; itry < 3; ++itry) {
        id = 1/d;
        int nchanged = 0;
        for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
            auto xb = xr + ib*kBlockSize;
            auto wb = weights + ib*kBlockSize;
            float best_mse = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float q = scales[ib]*values[quants[ib*kBlockSize+j]];
                float diff = xb[j] - d*q;
                best_mse += wb[j]*diff*diff;
            }
            int l = nearest_int(scales[ib]);
            if (l > 1) {
                float try_scale = l-1;
                float norm = 1/try_scale;
                float this_mse = 0;
                for (int j = 0; j < kBlockSize; ++j) {
                    int idx = best_index_iq4nl(values, id*norm*xb[j]);
                    float q = values[idx]*try_scale;
                    float diff = xb[j] - d*q;
                    this_mse += wb[j]*diff*diff;
                }
                if (this_mse < best_mse) {
                    best_mse = this_mse; scales[ib] = try_scale;
                    ++nchanged;
                }
            }
            if (l < 16) {
                float try_scale = l+1;
                float norm = 1/try_scale;
                float this_mse = 0;
                for (int j = 0; j < kBlockSize; ++j) {
                    int idx = best_index_iq4nl(values, id*norm*xb[j]);
                    float q = values[idx]*try_scale;
                    float diff = xb[j] - d*q;
                    this_mse += wb[j]*diff*diff;
                }
                if (this_mse < best_mse) {
                    best_mse = this_mse; scales[ib] = try_scale;
                    ++nchanged;
                }
            }
        }
        if (nchanged == 0) break;
        float sumqx = 0, sumq2 = 0;
        //float mse = 0;
        for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
            auto xb = xr + ib*kBlockSize;
            auto wb = weights + ib*kBlockSize;
            float norm = 1/scales[ib];
            for (int j = 0; j < kBlockSize; ++j) {
                int idx = best_index_iq4nl(values, id*norm*xb[j]);
                quants[ib*kBlockSize+j] = idx;
                float q = values[idx]*scales[ib];
                sumqx += wb[j]*q*xb[j];
                sumq2 += wb[j]*q*q;
                //float diff = xb[j] - d*q;
                //mse += diff*diff;
            }
        }
        d = sumqx/sumq2;
        //printf("itry = %d: %g, %g\n", itry, sqrt(mse/n_per_row), sqrt(2*mse/n_per_row/sigma2));
    }

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        y[ibl].scales = 0;
        auto qs = y[ibl].qs;
        auto qb = quants + ibl*QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            int l = nearest_int(scales[ibl*(QK_K/kBlockSize)+ib]);
            if (l < 1 || l > 16) {
                printf("Oops: scale = %g, l = %d\n", scales[ib], l);
                GGML_ABORT("fatal error");
            }
            y[ibl].scales |= ((l-1) << 4*ib);
        }
        for (int j = 0; j < QK_K/2; ++j) qs[j] = qb[j] | (qb[j+QK_K/2] << 4);
    }
    dh[0] = GGML_FP32_TO_FP16(d);

    //d = GGML_FP16_TO_FP32(dh[0]);
    //float mse = 0;
    //for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
    //    for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
    //        auto xb = xr + ibl*QK_K + ib*kBlockSize;
    //        int l = ((y[ibl].scales >> 4*ib) & 0xf) + 1;
    //        for (int j = 0; j < kBlockSize; ++j) {
    //            float q = values[(y[ibl].qs[64*(ib%2)+j] >> 4*(ib/2)) & 0xf]*l;
    //            //float q = values[quants[ibl*QK_K+ib*kBlockSize+j]]*l;
    //            float diff = xb[j] - d*q;
    //            mse += diff*diff;
    //        }
    //    }
    //}
    ////for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
    ////    auto xb = xr + ib*kBlockSize;
    ////    for (int j = 0; j < kBlockSize; ++j) {
    ////        float q = values[quants[ib*kBlockSize+j]]*scales[ib];
    ////        float diff = xb[j] - d*q;
    ////        mse += diff*diff;
    ////    }
    ////}
    //printf("Final rmse: %g, %g\n", sqrt(mse/n_per_row), sqrt(2*mse/n_per_row/sigma2));

    //constexpr float kMinGamma = 1.625f;
    //int8_t next_values[16];
    //float grad[16];

    //int8_t * int_values = (int8_t *)(dptr + 1);
    //std::memset(int_values, 0, 16);

    //float sigma2 = 0, amax = 0, max = 0;
    //for (int j = 0; j < n_per_row; ++j) {
    //    sigma2 += xr[j]*xr[j];
    //    float ax = std::abs(xr[j]);
    //    if (ax > amax) {
    //        amax = ax; max = xr[j];
    //    }
    //}
    //if (!sigma2) return;

    //float sigma = sqrt(sigma2/n_per_row);
    //float gamma = amax/sigma;
    //float alpha = gamma > kMinGamma ? (gamma/kMinGamma - 1)/gamma : 0.f;
    //float d = -max/(8*sigma*(1 + alpha*gamma));
    //float id = 1/d;
    //for (int j = 0; j < n_per_row; ++j) {
    //    float xs = xr[j]/sigma;
    //    float z = xs/(1 + alpha*std::abs(xs));
    //    int l = nearest_int(id*z);
    //    l = std::max(-8, std::min(7, l));
    //    quants[j] = l;
    //}

    //sigma2 *= 2.f/n_per_row;
    //if (quant_weights) {
    //    for (int j = 0; j < n_per_row; ++j) weights[j] = quant_weights[j] * sqrt(sigma2 + xr[j]*xr[j]);
    //} else {
    //    for (int j = 0; j < n_per_row; ++j) weights[j] = 0.25f*sigma2 + xr[j]*xr[j];
    //}

    //alpha = std::abs(alpha*d);
    //for (int iter = 0; iter < 9; ++iter) {
    //    float sumqx = 0, sumq2 = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        float q = sigma*quants[j]/(1 - alpha*std::abs(quants[j]));
    //        sumqx += weights[j]*q*xr[j];
    //        sumq2 += weights[j]*q*q;
    //    }
    //    if (sumq2 > 0) d = sumqx/sumq2;
    //    int nchanged = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        float xs = xr[j]/(d*sigma);
    //        float z = xs/(1 + alpha*std::abs(xs));
    //        int l = nearest_int(z);
    //        l = std::max(-8, std::min(7, l));
    //        if (l != quants[j]) ++nchanged;
    //        quants[j] = l;
    //    }
    //    if (nchanged == 0) break;
    //}
    //float c = 15.f*(1 - 8*alpha);
    //for (int i = 0; i < 16; ++i) {
    //    int_values[i] = nearest_int(c*(i-8)/(1-alpha*std::abs(i-8)));
    //}
    //float sumqx = 0, sumq2 = 0;
    //for (int j = 0; j < n_per_row; ++j) {
    //    quants[j] += 8;
    //    float q = int_values[quants[j]];
    //    sumqx += weights[j]*q*xr[j];
    //    sumq2 += weights[j]*q*q;
    //}
    //d = sumqx/sumq2;
    //for (int iter = 0; iter < 5; ++iter) {
    //    id = 1/d;
    //    std::memset(grad, 0, 16*sizeof(float));
    //    sumqx = sumq2 = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        int idx = best_index(16, int_values, id*xr[j]);
    //        float q = int_values[idx];
    //        grad[idx] += weights[j]*d*(xr[j] - d*q);
    //        quants[j] = idx;
    //        sumqx += weights[j]*q*xr[j];
    //        sumq2 += weights[j]*q*q;
    //    }
    //    all_steps.clear();
    //    for (int i = 0; i < 16; ++i) {
    //        int l = int_values[i];
    //        if (grad[i] > 0) {
    //            int lmax = std::min(127, l + 5);
    //            if (i < 16) lmax = std::min(lmax, int_values[i+1] - 1);
    //            for (int k = l + 1; k <= lmax; ++k) {
    //                float step = (k - 0.4999f - l)/grad[i];
    //                all_steps.push_back(step);
    //            }
    //        }
    //        else if (grad[i] < 0) {
    //            int lmin = std::max(-128, l - 5);
    //            if (i > 0) lmin = std::max(lmin, int_values[i-1]+1);
    //            for (int k = l-1; k >= lmin; --k) {
    //                float step = (k + 0.499f - l)/grad[i];
    //                all_steps.push_back(step);
    //            }
    //        }
    //    }
    //    float best = sumqx*sumqx/sumq2;
    //    int best_is = -1;
    //    int nstep = std::min(5, int(all_steps.size()));
    //    std::partial_sort(all_steps.begin(), all_steps.begin() + nstep, all_steps.end());
    //    float last_sumqx = sumqx, last_sumq2 = sumq2;
    //    for (int is = 0; is < nstep; ++is) {
    //        for (int i = 0; i < 16; ++i) {
    //            int l = nearest_int(int_values[i] + all_steps[is]*grad[i]);
    //            next_values[i] = std::max(-128, std::min(127, l));
    //        }
    //        sumqx = last_sumqx, sumq2 = last_sumq2;
    //        for (int j = 0; j < n_per_row; ++j) {
    //            int l = quants[j];
    //            int lnew = l;
    //            float dist = std::abs(id*xr[j] - next_values[l]);
    //            if (l > 0) {
    //                float dist1 = std::abs(id*xr[j] - next_values[l-1]);
    //                if (dist1 < dist) { dist = dist1; lnew = l - 1; }
    //            }
    //            if (l < 15) {
    //                float dist1 = std::abs(id*xr[j] - next_values[l+1]);
    //                if (dist1 < dist) { dist = dist1; lnew = l + 1; }
    //            }
    //            if (next_values[lnew] == int_values[l]) continue;
    //            float q = int_values[l];
    //            sumqx -= weights[j]*q*xr[j];
    //            sumq2 -= weights[j]*q*q;
    //            q = next_values[lnew];
    //            sumqx += weights[j]*q*xr[j];
    //            sumq2 += weights[j]*q*q;
    //        }
    //        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
    //            d = sumqx/sumq2; best = d*sumqx; best_is = is;
    //        }
    //    }
    //    if (best_is < 0) break;
    //    for (int i = 0; i < 16; ++i) {
    //        int l = nearest_int(int_values[i] + all_steps[best_is]*grad[i]);
    //        int_values[i] = l;
    //    }
    //}

    //*dptr = d;

    //uint8_t * qs = (uint8_t *)int_values + 16;

    //for (int ib = 0; ib < n_per_row/QK_K; ++ib) {
    //    for (int j = 0; j < QK_K/2; ++j) qs[j] = quants[j] | (quants[j+QK_K/2] << 4);
    //    qs += QK_K/2;
    //    quants += QK_K;
    //}

    ////for (int j = 0; j < n_per_row/2; ++j) {
    ////    qs[j] = quants[j] | (quants[j+n_per_row/2] << 4);
    ////}

}

static void quantize_row_iq4_kss_impl_old(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const uint16_t * table,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    float * dptr = (float *)cy;
    dptr[0] = 0;
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
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KSS, n_per_row);
    std::vector<float> weights(n_per_row);
    std::vector<float> scales(n_per_row/64);
    //std::vector<float> all_steps;
    std::vector<int8_t> quants(n_per_row);
    auto qrow = (char *)dst;
    for (int row = 0; row < nrows; ++row) {
        //quantize_row_iq4_kss_impl(n_per_row, src, qrow, imatrix, weights.data(), quants.data(), all_steps);
        quantize_row_iq4_kss_impl(n_per_row, src, qrow, imatrix, weights.data(), quants.data(), scales.data());
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
    const ggml_half * dh = (const ggml_half *)x;
    const float d = GGML_FP16_TO_FP32(dh[0]);
    x = (const block_iq4_kss *)(dh + 1);
    const int8_t * values = iq4k_values + 16;
    int nblock = k/QK_K;
    float scales[QK_K/64];
    for (int ib = 0; ib < nblock; ++ib) {
        for (int k = 0; k < QK_K/64; ++k) scales[k] = d*(((x[ib].scales >> 4*k) & 0xf) + 1);
        auto qs = x[ib].qs;
        for (int is = 0; is < 2; ++is) {
            for (int j = 0; j < 64; ++j) {
                y[j       ] = scales[is+0] * values[qs[j] & 0xf];
                y[j+QK_K/2] = scales[is+2] * values[qs[j] >>  4];
            }
            y  += 64;
            qs += 64;
        }
        y += QK_K/2;
    }

    //const float * dptr = (const float *)x;
    //const float d = *dptr;
    //const int8_t * int_values = (const int8_t *)(dptr + 1);
    //const uint8_t * qs = (const uint8_t *)int_values + 16;
    //int nblock = k/QK_K;
    //for (int ib = 0; ib < nblock; ++ib) {
    //    for (int j = 0; j < QK_K/2; ++j) {
    //        y[j       ] = d * int_values[qs[j] & 0xf];
    //        y[j+QK_K/2] = d * int_values[qs[j] >>  4];
    //    }
    //    qs += QK_K/2;
    //    y  += QK_K;
    //}
    //for (int j = 0; j < k/2; ++j) {
    //    y[j    ] = d * int_values[qs[j] & 0xf];
    //    y[j+k/2] = d * int_values[qs[j] >>  4];
    //}
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

// ========================================== iq2_kt ====================================================

namespace {
#ifdef __AVX2__
static inline float hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
static inline float hsum_float_8(__m256 x) {
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}
__m128 hsum_float_4x4(__m128 * accm) {
     accm[0] = _mm_add_ps(_mm_unpacklo_ps(accm[0], accm[2]), _mm_unpackhi_ps(accm[0], accm[2]));
     accm[1] = _mm_add_ps(_mm_unpacklo_ps(accm[1], accm[3]), _mm_unpackhi_ps(accm[1], accm[3]));
     return _mm_add_ps(_mm_unpacklo_ps(accm[0], accm[1]), _mm_unpackhi_ps(accm[0], accm[1]));
}
__m256 hsum_float_8x8(__m256 * accm) {
     for (int i = 0; i < 4; ++i) {
         accm[i] = _mm256_set_m128(_mm_add_ps(_mm256_castps256_ps128(accm[i+4]), _mm256_extractf128_ps(accm[i+4], 1)),
                                   _mm_add_ps(_mm256_castps256_ps128(accm[i+0]), _mm256_extractf128_ps(accm[i+0], 1)));
     }
     for (int i = 0; i < 2; ++i) accm[i] = _mm256_add_ps(_mm256_unpacklo_ps(accm[i], accm[i+2]), _mm256_unpackhi_ps(accm[i], accm[i+2]));
     return _mm256_add_ps(_mm256_unpacklo_ps(accm[0], accm[1]), _mm256_unpackhi_ps(accm[0], accm[1]));
}
__m256 hsum_float_4x8(__m256 * accm) {
     for (int i = 0; i < 2; ++i) accm[i] = _mm256_add_ps(_mm256_unpacklo_ps(accm[i], accm[i+2]), _mm256_unpackhi_ps(accm[i], accm[i+2]));
     return _mm256_add_ps(_mm256_unpacklo_ps(accm[0], accm[1]), _mm256_unpackhi_ps(accm[0], accm[1]));
}
#endif
template <int block_size, int group_size, int num_bits, bool is_abs = false>
class QuantizerIQKT {
    static_assert(group_size == 8 || group_size == 4);
    static_assert(block_size >= 8 && block_size%8 == 0);
public:
    constexpr static int kSuperBlockSize = QK_K;
    constexpr static int kBlockSize = block_size;
    constexpr static int kGroupSize = group_size;
    constexpr static int kNg = kBlockSize/kGroupSize;
    constexpr static int kNblock = kSuperBlockSize/kBlockSize;
    constexpr static int kNumVal = 1 << num_bits; // i.e, 16 bits per group of 8
    constexpr static float kScale = 31.75f;
    constexpr static bool kVerbose = false;

    QuantizerIQKT(int num_clusters, int num_neighbours, int offset = 4096);
    const float * values() const { return m_values.data(); }

    inline void find_best_match(float d, const float * xb, const float * weight, int * best_idx) const;
    inline std::pair<float, float> find_best_scale(const float * xb, const float * weight, const int * best_idx) const;
    inline float find_best_inverse_scale(const float * xb, const float * weight, const int * best_idx) const;

    static inline void set_values(uint32_t i, float * result, float scale, int offset = 4096) {
        constexpr uint32_t ka = 89226354;
        constexpr uint32_t kb = 64248484;
        constexpr uint32_t kmask = 0x8fff8fff;
        constexpr uint32_t km32 = 0x3b603b60;
        uint32_t x = i + offset;
        for (int k = 0; k < kGroupSize; ++k) {
            x = ka*x + kb;
            uint32_t s = (x & kmask) ^ km32;
            float val = GGML_FP16_TO_FP32(s & 65535) + GGML_FP16_TO_FP32(s >> 16);
            if constexpr (is_abs) result[k] = scale*std::abs(val);
            else result[k] = scale*val;
        }
        //for (int k = 0; k < kGroupSize; ++k) {
        //    x = ka*x + kb;
        //    uint32_t s = (x & kmask) ^ km32;
        //    float val = GGML_FP16_TO_FP32(s & 65535) + GGML_FP16_TO_FP32(s >> 16);
        //    x = ka*x + kb;
        //    s = (x & kmask) ^ km32;
        //    val += GGML_FP16_TO_FP32(s & 65535) + GGML_FP16_TO_FP32(s >> 16);
        //    if constexpr (is_abs) result[k] = scale*std::abs(0.5f*val);
        //    else result[k] = 0.5f*scale*val;
        //}
    }

    static inline int bin4(float x) {
        if constexpr (is_abs) {
            return x < 16.f ? 0 : x < 32.f ? 1 : x < 64.f ? 2 : 3;
        } else {
            return x < -24.f ? 0 : x < 0.0f ? 1 : x < 24.f ? 2 : 3;
        }
    }
    static inline int bin5(float x) {
        if constexpr (is_abs) {
            return x < 11.2f ? 0 : x < 24.f ? 1 : x < 39.f ? 2 : x < 58.f ? 3 : 4;
        } else {
            return x < -48.f ? 0 : x < -16.f ? 1 : x < 16.f ? 2 : x < 48.f ? 3 : 4;
        }
    }
    inline int bin3(int idim, float x) const { return x < m_mid[2*idim+0] ? 0 : x < m_mid[2*idim+1] ? 1 : 2; }

    static inline void set_weights(float sigma2_scale, int nblock, const float * x, const float * imatrix, float * row_weights) {
        for (int ibl = 0; ibl < nblock; ++ibl) {

            const float * xbl = x + ibl*kSuperBlockSize;
            float * wbl = row_weights + ibl*kSuperBlockSize;

            float sumx2 = 0;
            for (int j = 0; j < kSuperBlockSize; ++j) sumx2 += xbl[j]*xbl[j];
            const float sigma2 = sigma2_scale*sumx2/kSuperBlockSize;

            if (imatrix) {
                const float * qw = imatrix + ibl*kSuperBlockSize;
                for (int j = 0; j < kSuperBlockSize; ++j) wbl[j] = qw[j] * sqrtf(sigma2 + xbl[j]*xbl[j]);
            } else {
                for (int j = 0; j < kSuperBlockSize; ++j) wbl[j] = 0.25f*sigma2 + xbl[j]*xbl[j];
            }
        }
    }
private:
    static std::vector<float> cluster_points(const std::vector<float>& points, int ncluster, int niter, float * mid);
    static std::vector<std::vector<int>> finalize_clusters(int num_neighbours, const std::vector<float>& points, const std::vector<float>& clusters,
            std::vector<std::vector<float>>& c_values);
    std::vector<float> m_values;
    std::vector<float> m_clusters;
    std::vector<std::vector<int>> m_in_cluster;
    std::vector<std::vector<float>> m_c_values;
    float m_mid[4*kGroupSize];
};

template <int block_size, int group_size, int num_bits, bool is_abs>
QuantizerIQKT<block_size, group_size, num_bits, is_abs>::QuantizerIQKT(int num_clusters, int num_neighbours, int offset) {
    m_values.resize(kNumVal*kGroupSize);
    float * data = m_values.data();
    for (int i = 0; i < kNumVal; ++i) {
        set_values(i, data, kScale, offset);
        data += kGroupSize;
    }
    // Make 128 clusters.
    // Note: we get a slightly better result by using 64 clusters
    //       at the expense of almost doubling the quantization time.
    m_clusters = cluster_points(m_values, num_clusters, 200, m_mid);
    GGML_ASSERT(!m_clusters.empty());
    m_in_cluster = finalize_clusters(num_neighbours, m_values, m_clusters, m_c_values);
}

template <int block_size, int group_size, int num_bits, bool is_abs>
std::pair<float, float> QuantizerIQKT<block_size, group_size, num_bits, is_abs>::find_best_scale(
        const float * xb, const float * weight, const int * best_idx) const {
    float sumqx = 0, sumq2 = 0;
#ifdef __AVX2__
    auto vqx = _mm256_setzero_ps();
    auto vq2 = _mm256_setzero_ps();
    for (int l = 0; l < kBlockSize; l += 8) {
        auto vx = _mm256_loadu_ps(xb+l);
        auto vw = _mm256_loadu_ps(weight+l);
        auto vq = kGroupSize == 8 ? _mm256_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize]) :
            _mm256_set_m128(_mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+1]),
                            _mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+0]));
        auto vqw = _mm256_mul_ps(vq, vw);
        vqx = _mm256_fmadd_ps(vqw, vx, vqx);
        vq2 = _mm256_fmadd_ps(vqw, vq, vq2);
    }
    sumqx = hsum_float_8(vqx);
    sumq2 = hsum_float_8(vq2);
#else
    for (int l = 0; l < kNg; ++l) {
        auto xl = xb + kGroupSize*l;
        auto wl = weight + kGroupSize*l;
        auto ql = m_values.data() + kGroupSize*best_idx[l];
        for (int k = 0; k < kGroupSize; ++k) {
            sumqx += wl[k]*ql[k]*xl[k];
            sumq2 += wl[k]*ql[k]*ql[k];
        }
    }
#endif
    return sumq2 > 0 ? std::make_pair(sumqx/sumq2, sumqx*sumqx/sumq2) : std::make_pair(0.f, 0.f);
}

template <int block_size, int group_size, int num_bits, bool is_abs>
float QuantizerIQKT<block_size, group_size, num_bits, is_abs>::find_best_inverse_scale(
        const float * xb, const float * weight, const int * best_idx) const {
    float sumqx = 0, sumx2 = 0;
#ifdef __AVX2__
    auto vqx = _mm256_setzero_ps();
    auto vx2 = _mm256_setzero_ps();
    for (int l = 0; l < kBlockSize; l += 8) {
        auto vx = _mm256_loadu_ps(xb+l);
        auto vw = _mm256_loadu_ps(weight+l);
        auto vq = kGroupSize == 8 ? _mm256_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize]) :
            _mm256_set_m128(_mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+1]),
                            _mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+0]));
        auto vxw = _mm256_mul_ps(vx, vw);
        vx2 = _mm256_fmadd_ps(vxw, vx, vx2);
        vqx = _mm256_fmadd_ps(vxw, vq, vqx);
    }
    sumqx = hsum_float_8(vqx);
    sumx2 = hsum_float_8(vx2);
#else
    for (int l = 0; l < kNg; ++l) {
        auto xl = xb + kGroupSize*l;
        auto wl = weight + kGroupSize*l;
        auto ql = m_values.data() + kGroupSize*best_idx[l];
        for (int k = 0; k < kGroupSize; ++k) {
            sumqx += wl[k]*ql[k]*xl[k];
            sumx2 += wl[k]*xl[k]*xl[k];
        }
    }
#endif
    return sumx2 > 0 ? sumqx/sumx2 : 0.f;
}

template <int block_size, int group_size, int num_bits, bool is_abs>
void QuantizerIQKT<block_size, group_size, num_bits, is_abs>::find_best_match(float d, const float * xb, const float * weight, int * best_idx) const {
    if (!d) {
        std::memset(best_idx, 0, kNg*sizeof(int));
        return;
    }
    int ncluster = m_clusters.size()/kGroupSize;
    float id = 1/d;
#ifdef __AVX2__
    if constexpr (kGroupSize == 8) {
        __m256 sqx[8];
        const __m256i add_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        float sx[8];
        int   index[8];
        auto vid = _mm256_set1_ps(id);
        auto add8 = _mm256_set1_epi32(8);
        for (int l = 0; l < kNg; ++l) {
            auto xl = xb + 8*l;
            auto wl = weight + 8*l;
            auto vx = _mm256_mul_ps(vid, _mm256_loadu_ps(xl));
            auto vw = _mm256_loadu_ps(wl);
            int jbest = -1;
            if (kGroupSize == 8 && (ncluster == 256 || ncluster == 6561)) {
                _mm256_store_ps(sx, vx);
                uint16_t u = 0;
                if (ncluster == 256) {
                    for (int j = 0; j < 8; ++j) if (sx[j] > m_mid[j]) u |= (1 << j);
                } else {
                    int s = 1;
                    for (int j = 0; j < 8; ++j) { u += s*bin3(j, sx[j]); s *= 3; }
                }
                jbest = u;
            } else {
                auto vbest = _mm256_set1_ps(INFINITY);
                auto best_index = _mm256_set1_epi32(-1);
                float best = INFINITY;
                auto idx = add_idx;
                for (int j = 0; j < ncluster; j += 8) {
                    for (int i = 0; i < 8; ++i) {
                        auto vq = _mm256_loadu_ps(m_clusters.data() + kGroupSize*(j+i));
                        auto vdiff = _mm256_sub_ps(vq, vx);
                        sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                    }
                    auto score = hsum_float_8x8(sqx);
                    auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                    best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                            _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                    vbest = _mm256_min_ps(vbest, score);
                    idx = _mm256_add_epi32(idx, add8);
                }
                _mm256_store_ps(sx, vbest);
                _mm256_store_si256((__m256i *)index, best_index);
                for (int i = 0; i < 8; ++i) {
                    if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
                }
            }
            auto& points = m_in_cluster[jbest];
            auto& values = points.empty() ? m_values : m_c_values[jbest];
            int npoint = values.size()/kGroupSize;
            //if (points.empty() || points.size()%8 != 0) printf("Oops: %d points in cluster %d\n", int(points.size()), jbest);
            GGML_ASSERT(npoint > 0 && npoint%8 == 0);
            int jbest_cluster = jbest;
            auto vbest = _mm256_set1_ps(INFINITY);
            auto best_index = _mm256_set1_epi32(-1);
            auto best = INFINITY; jbest = -1;
            auto idx = add_idx;
            for (int j = 0; j < npoint; j += 8) {
                for (int i = 0; i < 8; ++i) {
                    auto vq = _mm256_loadu_ps(values.data() + kGroupSize*(j+i));
                    auto vdiff = _mm256_sub_ps(vq, vx);
                    sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                }
                auto score = hsum_float_8x8(sqx);
                auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                        _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                vbest = _mm256_min_ps(vbest, score);
                idx = _mm256_add_epi32(idx, add8);
            }
            _mm256_store_ps(sx, vbest);
            _mm256_store_si256((__m256i *)index, best_index);
            for (int i = 0; i < 8; ++i) {
                if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
            }
            if (jbest < 0) {
                fprintf(stderr, "Oops: jbest = %d for cluster %d with %d points\n", jbest, jbest_cluster, int(points.size()));
                GGML_ASSERT(false);
            }
            best_idx[l] = points.empty() ? jbest : points[jbest];
        }
    } else {
        __m256 sqx[4];
        const __m256i add_idx = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
        const __m256 sign_bit = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        float sx[8];
        int   index[8];
        auto vid_p = _mm256_set1_ps(id);
        auto add8 = _mm256_set1_epi32(8);
        for (int l = 0; l < kNg; ++l) {
            auto xl = xb + 4*l;
            auto wl = weight + 4*l;
            auto vx4 = _mm_loadu_ps(xl);
            auto vx = _mm256_mul_ps(vid_p, _mm256_set_m128(vx4, vx4));
            auto vw4 = _mm_loadu_ps(wl);
            auto vw = _mm256_set_m128(vw4, vw4);
            int jbest = -1;
            if (ncluster == 256 || ncluster == 625) {
                _mm256_storeu_ps(sx, vx);
                uint16_t u = 0;
                if (ncluster == 256) {
                    for (int k = 0; k < 4; ++k) u |= (bin4(sx[k]) << 2*k);
                } else {
                    int l = 1;
                    for (int k = 0; k < 4; ++k) { u += bin5(sx[k])*l; l *= 5; }
                }
                jbest = u;
            } else {
                auto vbest = _mm256_set1_ps(INFINITY);
                auto best_index = _mm256_set1_epi32(-1);
                float best = INFINITY;
                auto idx = add_idx;
                for (int j = 0; j < ncluster; j += 8) {
                    for (int i = 0; i < 4; ++i) {
                        auto vq = _mm256_loadu_ps(m_clusters.data() + kGroupSize*(j+2*i));
                        auto vdiff = _mm256_sub_ps(vq, vx);
                        //vdiff = _mm256_mul_ps(vdiff, vdiff);
                        //sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                        vdiff = _mm256_and_ps(sign_bit, vdiff);
                        sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, _mm256_mul_ps(vdiff, vdiff)));
                    }
                    auto score = hsum_float_4x8(sqx);
                    auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                    best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                            _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                    vbest = _mm256_min_ps(vbest, score);
                    idx = _mm256_add_epi32(idx, add8);
                }
                _mm256_store_ps(sx, vbest);
                _mm256_store_si256((__m256i *)index, best_index);
                for (int i = 0; i < 8; ++i) {
                    if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
                }
            }
            auto& points = m_in_cluster[jbest];
            auto& values = m_c_values[jbest];
            GGML_ASSERT(!points.empty() && points.size()%8 == 0);
            int jbest_cluster = jbest;
            auto vbest = _mm256_set1_ps(INFINITY);
            auto best_index = _mm256_set1_epi32(-1);
            float best = INFINITY; jbest = -1;
            auto idx = add_idx;
            for (int j = 0; j < int(points.size()); j += 8) {
                for (int i = 0; i < 4; ++i) {
                    auto vq = _mm256_loadu_ps(values.data() + kGroupSize*(j+2*i));
                    auto vdiff = _mm256_sub_ps(vq, vx);
                    //vdiff = _mm256_mul_ps(vdiff, vdiff);
                    sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                    //vdiff = _mm256_and_ps(sign_bit, vdiff);
                    //sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, _mm256_mul_ps(vdiff, vdiff)));
                }
                auto score = hsum_float_4x8(sqx);
                auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                                       _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                vbest = _mm256_min_ps(vbest, score);
                idx = _mm256_add_epi32(idx, add8);
            }
            _mm256_store_ps(sx, vbest);
            _mm256_store_si256((__m256i *)index, best_index);
            for (int i = 0; i < 8; ++i) {
                if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
            }
            if (jbest < 0) {
                fprintf(stderr, "Oops: jbest = %d for cluster %d with %d points\n", jbest, jbest_cluster, int(points.size()));
                GGML_ASSERT(false);
            }
            best_idx[l] = points[jbest];
        }
    }
#else
    // TODO
    std::memset(best_idx, 0, kNg*sizeof(int));
#endif
}

template <int block_size, int group_size, int num_bits, bool is_abs>
std::vector<std::vector<int>> QuantizerIQKT<block_size, group_size, num_bits, is_abs>::finalize_clusters(int num_neighbours,
        const std::vector<float>& values, const std::vector<float>& clusters, std::vector<std::vector<float>>& c_values) {
    int ncluster = clusters.size()/kGroupSize;
    //GGML_ASSERT(ncluster%8 == 0);
    std::vector<std::vector<int>> p_in_cluster(ncluster);
    std::vector<int> which_cluster(num_neighbours*kNumVal);
    std::vector<int> ibest(num_neighbours);
    std::vector<float> best(num_neighbours);
    for (int ip = 0; ip < kNumVal; ++ip) {
        auto vp = values.data() + ip*kGroupSize;
        for (int j = 0; j < num_neighbours; ++j) {
            best[j] = INFINITY; ibest[j] = -1;
        }
        for (int ic = 0; ic < ncluster; ++ic) {
            auto vc = clusters.data() + ic*kGroupSize;
            float dist2 = 0;
            for (int k = 0; k < kGroupSize; ++k) {
                float d = vp[k] - vc[k]; dist2 += d*d;
            }
            for (int j = 0; j < num_neighbours; ++j) {
                if (dist2 < best[j]) {
                    for (int k = num_neighbours-1; k > j; --k) {
                        best[k] = best[k-1]; ibest[k] = ibest[k-1];
                    }
                    best[j] = dist2; ibest[j] = ic;
                    break;
                }
            }
        }
        for (int j = 0; j < num_neighbours; ++j) {
            if (ibest[j] < 0) {
                printf("Oops: ibest[%d] = %d\n", j, ibest[j]);
            }
            GGML_ASSERT(ibest[j] >= 0);
            p_in_cluster[ibest[j]].push_back(ip);
        }
        std::memcpy(which_cluster.data() + num_neighbours*ip, ibest.data(), num_neighbours*sizeof(int));
    }
    std::vector<std::pair<float, int>> extra;
    extra.reserve(kNumVal);
    for (int ic = 0; ic < ncluster; ++ic) {
        auto& points = p_in_cluster[ic];
        if (!points.empty() && points.size()%8 == 0) continue;
        extra.clear();
        auto vc = clusters.data() + ic*kGroupSize;
        for (int ip = 0; ip < kNumVal; ++ip) {
            bool can_add = true;
            for (int j = 0; j < num_neighbours; ++j) {
                if (which_cluster[num_neighbours*ip+j] == ic) { can_add = false; break; }
            }
            if (!can_add) continue;
            auto vp = values.data() + ip*kGroupSize;
            float dist2 = 0;
            for (int k = 0; k < kGroupSize; ++k) {
                float d = vp[k] - vc[k]; dist2 += d*d;
            }
            extra.push_back(std::make_pair(dist2, ip));
        }
        std::sort(extra.begin(), extra.end());
        int nadd = 8*((points.size()+7)/8) - points.size();
        for (int i = 0; i < nadd; ++i) points.push_back(extra[i].second);
        GGML_ASSERT(points.size()%8 == 0);
    }
    auto min = p_in_cluster.front().size(), max = p_in_cluster.front().size();
    for (auto& points : p_in_cluster) {
        min = std::min(min, points.size());
        max = std::max(max, points.size());
    }
    c_values.resize(p_in_cluster.size());
    for (int i = 0; i < int(p_in_cluster.size()); ++i) {
        auto& points = p_in_cluster[i];
        c_values[i].resize(points.size()*kGroupSize);
        auto ptr = c_values[i].data();
        for (auto j : points) {
            std::memcpy(ptr, values.data() + j*kGroupSize, kGroupSize*sizeof(float));
            ptr += kGroupSize;
        }
    }

    if (kVerbose) {
        printf("%s: prepared %d clusters\n", __func__, ncluster);
        printf("    min number of points in a cluster: %d\n", int(min));
        printf("    max number of points in a cluster: %d\n", int(max));
    }
    return p_in_cluster;
}

template <int block_size, int group_size, int num_bits, bool is_abs>
std::vector<float> QuantizerIQKT<block_size, group_size, num_bits, is_abs>::cluster_points(const std::vector<float>& points, int ncluster, int niter, float * mid) {
    constexpr int ndim = kGroupSize;
    GGML_ASSERT(points.size() % ndim == 0);
    int npoint = points.size() / ndim;
    GGML_ASSERT(npoint >= 2*ncluster);
    std::vector<std::pair<float, float>> range(ndim, std::make_pair(INFINITY, -INFINITY));
    double Fo = 0;
    for (int i = 0; i < npoint; ++i) {
        auto v = points.data() + i*ndim;
        for (int k = 0; k < ndim; ++k) {
            Fo += v[k]*v[k];
            range[k].first  = std::min(range[k].first, v[k]);
            range[k].second = std::max(range[k].second, v[k]);
        }
    }
    if (kVerbose) printf("%s (ndim = %d, npoint = %d): Fo = %g\n", __func__, ndim, npoint, Fo/points.size());
    if constexpr (is_abs) {
        std::vector<int> P(npoint);
        for (int idim = 0; idim < ndim; ++idim) {
            for (int ip = 0; ip < npoint; ++ip) P[ip] = points[ip*ndim+idim];
            std::sort(P.begin(), P.end());
            if (ndim == 8 && ncluster == 6561) {
                mid[2*idim + 0] = P[npoint/3];
                mid[2*idim + 1] = P[2*npoint/3];
            } else {
                mid[idim] = npoint%2 == 0 ? 0.5f*(P[npoint/2] + P[npoint/2-1]) : P[npoint/2];
                if (kVerbose) printf("%s: mid[%d] = %g\n", __func__, idim, mid[idim]);
            }
        }
    } else {
        for (int k = 0; k < ndim; ++k) mid[k] = 0.5f*(range[k].first + range[k].second);
    }
    std::vector<float> sump(ncluster*ndim);
    std::vector<int> counts(ncluster);
    std::vector<float> result(ncluster*ndim);
    if (ndim == 8 && (ncluster == 256 || ncluster == 6561)) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            uint16_t u = 0;
            if (ncluster == 256) {
                for (int k = 0; k < ndim; ++k) if (vp[k] > mid[k]) u |= (1 << k);
            } else {
                int s = 1;
                for (int k = 0; k < ndim; ++k) {
                    int bin = vp[k] < mid[2*k+0] ? 0 : vp[k] < mid[2*k+1] ? 1 : 2;
                    u += s*bin; s *= 3;
                }
            }
            ++counts[u];
            for (int k = 0; k < ndim; ++k) sump[ndim*u + k] += vp[k];
        }
        for (int ic = 0; ic < ncluster; ++ic) {
            if (!counts[ic]) {
                printf("%s: Oops. Cluster %d has no points\n", __func__, ic);
                GGML_ABORT("fatal error");
            }
            for (int k = 0; k < ndim; ++k) result[ic*ndim + k] = sump[ic*ndim + k]/counts[ic];
        }
        return result;
    }
    else if (ndim == 4 && (ncluster == 256 || ncluster == 625)) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            uint16_t u = 0;
            if (ncluster == 256) {
                for (int k = 0; k < ndim; ++k) u |= (bin4(vp[k]) << 2*k);
            } else {
                int s = 1;
                for (int k = 0; k < ndim; ++k) { u += s*bin5(vp[k]); s *= 5; }
            }
            if (u >= int(counts.size())) {
                printf("Oops: u = %u, vp = %g, %g, %g, %g\n", u, vp[0], vp[1], vp[2], vp[3]);
                u = 0;
                if (ncluster == 256) {
                    for (int k = 0; k < ndim; ++k) {
                        auto bin = bin4(vp[k]); u |= (bin << 2*k);
                        printf(" bin[%d] = %d, u = %u", k, bin, u);
                    }
                } else {
                    for (int k = 0; k < ndim; ++k) printf(" bin[%d] = %d", k, bin5(vp[k]));
                }
                printf("\n");
                GGML_ABORT("fatal error");
            }
            ++counts[u];
            for (int k = 0; k < ndim; ++k) sump[ndim*u + k] += vp[k];
        }
        int nzero = 0;
        for (int ic = 0; ic < ncluster; ++ic) {
            if (!counts[ic]) {
                ++nzero;
                printf("%s: Oops. Cluster %d has no points: ", __func__, ic);
                for (int k = 0; k < ndim; ++k) {
                    int l = (ic >> 2*k) & 3;
                    printf(" %d", l);
                }
                printf("\n");
                //GGML_ABORT("fatal error");
            } else {
                for (int k = 0; k < ndim; ++k) result[ic*ndim + k] = sump[ic*ndim + k]/counts[ic];
            }
        }
        if (nzero > 0) printf("%s: %d out of %d clusters dir not have any points\n", __func__, nzero, ncluster);
        //counts.resize(ndim*ncluster);
        //auto fcounts = (float *)counts.data();
        //std::memset(fcounts, 0, counts.size()*sizeof(float));
        //for (int ip = 0; ip < npoint; ++ip) {
        //    auto vp = points.data() + ndim*ip;
        //    uint8_t u = 0;
        //    for (int k = 0; k < ndim; ++k) u |= (bin4(vp[k]) << 2*k);
        //    for (int k = 0; k < ndim; ++k) {
        //        float w = std::abs(vp[k]);
        //        sump[ndim*u + k] += w*vp[k];
        //        fcounts[ndim*u + k] += w;
        //    }
        //}
        //for (int ic = 0; ic < ncluster; ++ic) {
        //    for (int k = 0; k < ndim; ++k) result[ic*ndim + k] = fcounts[ic*ndim + k] > 0 ? sump[ic*ndim + k]/fcounts[ic*ndim + k] : 0.f;
        //}
        return result;
    }
    std::mt19937 rndm(1234);
    float scale = 1.f/4294967296.f;
    for (int i = 0; i < ncluster; ++i) {
        auto v = result.data() + i*ndim;
        for (int k = 0; k < ndim; ++k) v[k] = range[k].first + (range[k].second - range[k].first)*scale*rndm();
    }
    std::vector<int> which_cluster(npoint, -1);
    double Flast = Fo;
    for (int iter = 0; iter < niter; ++iter) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        int nchanged = 0;
        double F = 0;
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            float best = INFINITY; int ibest = -1;
            for (int ic = 0; ic < ncluster; ++ic) {
                auto vc = result.data() + ndim*ic;
                float dist2 = 0;
                for (int k = 0; k < ndim; ++k) {
                    float d = vp[k] - vc[k]; dist2 += d*d;
                }
                if (dist2 < best) {
                    best = dist2; ibest = ic;
                }
            }
            if (ibest < 0) {
                printf("Oops(iteration %d) - failed to find cluster for point", iter);
                for (int k = 0; k < ndim; ++k) printf(" %g", vp[k]);
                printf("\nHave %d clusters\n", ncluster);
            }
            GGML_ASSERT(ibest >= 0);
            F += best;
            if (which_cluster[ip] != ibest) ++nchanged;
            which_cluster[ip] = ibest;
            ++counts[ibest];
            auto vc = sump.data() + ndim*ibest;
            for (int k = 0; k < ndim; ++k) vc[k] += vp[k];
        }
        if (nchanged == 0) break;
        for (int ic = 0; ic < ncluster; ++ic) {
            float norm = counts[ic] > 0 ? 1.f/counts[ic] : 0.f;
            auto vc = sump.data() + ndim*ic;
            auto r  = result.data() + ndim*ic;
            for (int k = 0; k < ndim; ++k) r[k] = vc[k]*norm;
        }
        if (kVerbose) printf("%s(iteration %d): F = %g, nchanged = %d\n", __func__, iter+1, F/points.size(), nchanged);
        if (iter > 1 && Flast/F - 1 < 1e-6) break;
        Flast = F;
    }
    int nzero = 0;
    for (int ic = 0; ic < ncluster; ++ic) {
        if (!counts[ic]) ++nzero;
    }
    if (nzero > 0) printf("%s: there are %d empty clusters\n", __func__, nzero);
    return result;
}

using QuantizerIQ2KT = QuantizerIQKT<32, 8, 16>;

const QuantizerIQ2KT& iq2kt_quantizer() {
    static std::mutex mutex;
    static std::unique_ptr<QuantizerIQ2KT> quantizer;
    std::lock_guard<std::mutex> lock(mutex);
    if (!quantizer) quantizer = std::make_unique<QuantizerIQ2KT>(256, 8);
    return *quantizer;
}

void quantize_row_iq2_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights,
        float * qtmp) {

    constexpr float kSigmaScale = 2.0f;
    using Q = QuantizerIQ2KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq2_kt * y = (block_iq2_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq2kt_quantizer();

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights);

    float amax_scale = 0, max_scale = 0;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_kt));

        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            quantizer.find_best_match( amax/96.f, xb, weight, best_idx);
            auto [dp, score_p] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/96.f, xb, weight, best_idx + Q::kNg);
            auto [dm, score_m] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            auto idx = best_idx;
            if (score_p > score_m) scales[ib] = dp;
            else {
                scales[ib] = dm; idx += Q::kNg;
            }
            auto qt = qtmp + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            for (int ig = 0; ig < Q::kNg; ++ig) {
                auto q = quantizer.values() + idx[ig]*Q::kGroupSize;
                for (int j = 0; j < Q::kGroupSize; ++j) qt[j] = q[j];
                qt += Q::kGroupSize;
            }

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    if (!max_scale) {
        *dptr = 0;
        return;
    }

    float d = max_scale/iq4k_values[0];
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (itry + iq4k_values[0])/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * qb = qtmp + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
                float dl = iq4k_values[ls];
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    float q = dl*qb[j];
                    sumqx += wb[j]*xb[j]*q;
                    sumq2 += wb[j]*q*q;
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
                qb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            int ls1 = best_index_iq4nl(iq4k_values, id*scales[ib]);
            int ls2 = best_index_iq4nl(iq4k_values, id*scales[ib + Q::kNblock/2]);
            y[ibl].scales[ib] = ls1 | (ls2 << 4);
        }
    }

    *dptr = d;
    if (!d) return;

    //d *= 1.05f;

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            auto qs = (uint16_t *)y[ibl].ql;
            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                int ls = iq4k_values[(y[ibl].scales[ib%(Q::kNblock/2)] >> 4*(ib/(Q::kNblock/2))) & 0xf];
                float dl = d*ls;
                quantizer.find_best_match(dl, xb, weight, best_idx);

                for (int j = 0; j < Q::kNg; ++j) {
                    qs[j] = best_idx[j];
                    auto xl = xb + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
                qs += Q::kNg;
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d;
            if (!d) return;
        } else {
            break;
        }

        if (false) {
            for (int ibl = 0; ibl < nblock; ++ibl) {
                const float * xbl = x + ibl*Q::kSuperBlockSize;
                auto scales = all_scales + ibl*Q::kNblock;
                auto qs = (uint16_t *)y[ibl].ql;
                for (int ib = 0; ib < Q::kNblock; ++ib) {
                    const float * xb = xbl + Q::kBlockSize*ib;
                    const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                    for (int j = 0; j < Q::kNg; ++j) best_idx[j] = qs[ib*Q::kNg+j];
                    auto pair = quantizer.find_best_scale(xb, weight, best_idx);
                    scales[ib] = pair.first;
                }
            }
            float id = d ? 1/d : 0.f;
            for (int ibl = 0; ibl < nblock; ++ibl) {
                auto scales = all_scales + ibl*Q::kNblock;
                for (int ib = 0; ib < Q::kNblock/2; ++ib) {
                    int ls1 = best_index_iq4nl(iq4k_values, id*scales[ib]);
                    int ls2 = best_index_iq4nl(iq4k_values, id*scales[ib + Q::kNblock/2]);
                    y[ibl].scales[ib] = ls1 | (ls2 << 4);
                }
            }
        }

    }

}
}

void quantize_row_iq2_kt_ref(const float * GGML_RESTRICT x, block_iq2_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_kt(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq2_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_kt * y = (block_iq2_kt *)vy;
    quantize_row_iq2_kt_ref(x, y, k);
}

size_t quantize_iq2_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_KT, n_per_row);
    std::vector<float> scales(n_per_row/QuantizerIQ2KT::kBlockSize);
    std::vector<float> weights(n_per_row);
    std::vector<float> xtmp(n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq2_kt_impl(src, (void *)qrow, n_per_row, imatrix, scales.data(), weights.data(), xtmp.data());
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}

void dequantize_row_iq2_kt(const block_iq2_kt * x, float * y, int64_t k) {
    assert(k % QuantizerIQ2KT::kSuperBlockSize == 0);
    const int nb = k / QuantizerIQ2KT::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = *dptr * QuantizerIQ2KT::kScale;
    x = (const block_iq2_kt *)(dptr + 1);
    auto& deq = iq2kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto yl = y + ibl*QuantizerIQ2KT::kSuperBlockSize;
        auto yh = yl + QuantizerIQ2KT::kSuperBlockSize/2;
        const uint16_t * ql = (const uint16_t *)x[ibl].ql;
        const uint16_t * qh = ql + QuantizerIQ2KT::kNg*QuantizerIQ2KT::kNblock/2;
        for (int ib = 0; ib < QuantizerIQ2KT::kNblock/2; ++ib) {
            float sl = d * iq4k_values[x[ibl].scales[ib] & 0xf];
            float sh = d * iq4k_values[x[ibl].scales[ib] >>  4];
            for (int ig = 0; ig < QuantizerIQ2KT::kNg; ++ig) {
                deq.set_values(ql[ig], yl, sl);
                deq.set_values(qh[ig], yh, sh);
                yl += QuantizerIQ2KT::kGroupSize;
                yh += QuantizerIQ2KT::kGroupSize;
            }
            ql += QuantizerIQ2KT::kNg;
            qh += QuantizerIQ2KT::kNg;
        }
    }
}

void vec_dot_iq2_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_KT, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

}

namespace {

using QuantizerIQ3KT = QuantizerIQKT<32, 8, 16, true>;
const QuantizerIQ3KT& iq3kt_quantizer() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unique_ptr<QuantizerIQ3KT> quantizer;
    if (!quantizer) quantizer = std::make_unique<QuantizerIQ3KT>(256, 8);
    return *quantizer;
}

void quantize_row_iq3_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales,
        float * all_weights, float * qtmp) {

    constexpr float kSigmaScale = 2.0f;
    constexpr float kStep = 8.0f;

    using Q = QuantizerIQ3KT;

    static_assert(Q::kNumVal%8 == 0);

    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;

    float * dptr = (float *)vy;

    block_iq3_kt * y = (block_iq3_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq3kt_quantizer();

    int nblock = n_per_row / Q::kSuperBlockSize;

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) amax_row = std::max(amax_row, std::abs(x[j]));
    if (!amax_row) {
        *dptr = 0.f;
        std::memset(y, 0, nblock*sizeof(block_iq3_kt));
        return;
    }

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights);

    float amax_scale = 0, max_scale = 0;

    float xaux[Q::kBlockSize];

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_kt));

        auto scales = all_scales + ibl*Q::kNblock;
        auto xbl = x + ibl*Q::kSuperBlockSize;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                xaux[j] = ax;
                amax = std::max(amax, ax);
            }
            scales[ib] = 0;
            if (!amax) continue;

            //quantizer.find_best_match(amax/96.f, xaux, weight, best_idx+Q::kNg);
            //scales[ib] = quantizer.find_best_scale(xaux, weight, best_idx+Q::kNg).first;

            float scale_0 = std::max(84.f, 123.f*amax/amax_row);
            //float scale_0 = std::max(64.f, 123.f*amax/amax_row);
            float best = 0;
            for (int itry = -3; itry <= 3; ++itry) {
                quantizer.find_best_match(amax/(scale_0 + kStep*itry), xaux, weight, best_idx);
                auto [d, score] = quantizer.find_best_scale(xaux, weight, best_idx);
                if (score > best) {
                    best = score;
                    scales[ib] = d;
                    std::memcpy(best_idx+Q::kNg, best_idx, Q::kNg*sizeof(int));
                }
            }

            auto xt = qtmp + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            for (int ig = 0; ig < Q::kNg; ++ig) {
                auto q = quantizer.values() + Q::kGroupSize*best_idx[Q::kNg+ig];
                for (int j = 0; j < Q::kGroupSize; ++j) *xt++ = q[j];
            }

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    GGML_ASSERT(max_scale >= 0);
    float d = max_scale/15;
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (itry*0.2f + 15)/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * qb = qtmp + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = nearest_int(id*scales[ib]);
                ls = std::max(0, std::min(15, ls));
                float dl = ls;
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    float q = dl*qb[j];
                    sumqx += wb[j]*std::abs(xb[j])*q;
                    sumq2 += wb[j]*q*q;
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
                qb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            int ls1 = nearest_int(id*scales[ib]);
            int ls2 = nearest_int(id*scales[ib + Q::kNblock/2]);
            ls1 = std::max(0, std::min(15, ls1));
            ls2 = std::max(0, std::min(15, ls2));
            y[ibl].scales[ib] = ls1 | (ls2 << 4);
        }
    }

    *dptr = d;

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            uint16_t * ql = (uint16_t *)y[ibl].ql;

            std::memset(y[ibl].qh, 0, kNumGroups/2);
            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    xaux[j] = std::abs(xb[j]);
                    if (xb[j] < 0) y[ibl].qh[j] |= (1 << ib);
                }
                int ls = (y[ibl].scales[ib%(Q::kNblock/2)] >> 4*(ib/(Q::kNblock/2))) & 0xf;
                float dl = d*ls;
                quantizer.find_best_match(dl, xaux, weight, best_idx);

                for (int j = 0; j < Q::kNg; ++j) {
                    ql[ib*Q::kNg+j] = best_idx[j];
                    auto xl = xaux + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d;
            if (!d) break;
        } else {
            break;
        }
    }
}
}

void quantize_row_iq3_kt_ref(const float * x, block_iq3_kt * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq3_kt(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq3_kt(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq3_kt * y = (block_iq3_kt *)vy;
    quantize_row_iq3_kt_ref(x, y, k);
}

size_t quantize_iq3_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_KT, n_per_row);
    std::vector<float> scales(n_per_row/QuantizerIQ3KT::kBlockSize);
    std::vector<float> weights(n_per_row), xtmp(n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq3_kt_impl(src, (void *)qrow, n_per_row, imatrix, scales.data(), weights.data(), xtmp.data());
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}

void dequantize_row_iq3_kt(const block_iq3_kt * x, float * y, int64_t k) {
    using Q = QuantizerIQ3KT;
    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;
    assert(k % Q::kSuperBlockSize == 0);
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = *dptr * Q::kScale;
    x = (const block_iq3_kt *)(dptr + 1);
    auto& deq = iq3kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto yl = y + ibl*Q::kSuperBlockSize;
        auto yh = yl + Q::kSuperBlockSize/2;
        auto qll = (const uint16_t *)x[ibl].ql;
        auto qlh = qll + kNumGroups/2;
        int jj = 0;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            float sl = d * (x[ibl].scales[ib] & 0xf);
            float sh = d * (x[ibl].scales[ib] >>  4);
            uint8_t l_mask = 1 << ib;
            uint8_t h_mask = l_mask << (Q::kNblock/2);
            for (int ig = 0; ig < Q::kNg; ++ig) {
                deq.set_values(qll[jj], yl, sl);
                deq.set_values(qlh[jj], yh, sh);
                for (int j = 0; j < Q::kGroupSize; ++j) {
                    if (x[ibl].qh[ig*Q::kGroupSize+j] & l_mask) yl[j] = -yl[j];
                    if (x[ibl].qh[ig*Q::kGroupSize+j] & h_mask) yh[j] = -yh[j];
                }
                yl += Q::kGroupSize;
                yh += Q::kGroupSize;
                ++jj;
            }
        }
    }
}

void vec_dot_iq3_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_KT, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

}

// ======================================== iq4_kt

namespace{

using QuantizerIQ4KT = QuantizerIQKT<32, 4, 15>;

const QuantizerIQ4KT& iq4kt_quantizer(bool with_offset = false) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unique_ptr<QuantizerIQ4KT> quantizer1;
    static std::unique_ptr<QuantizerIQ4KT> quantizer2;
    if (with_offset) {
        if (!quantizer2) quantizer2 = std::make_unique<QuantizerIQ4KT>(625, 6, 4096+32768);
        return *quantizer2;
    }
    if (!quantizer1) quantizer1 = std::make_unique<QuantizerIQ4KT>(625, 6, 4096);
    return *quantizer1;
}

void quantize_row_iq4_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights) {

    constexpr float kSigmaScale = 2.0f;
    constexpr int kNtry = 2;
    using Q = QuantizerIQ4KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq4_kt * y = (block_iq4_kt *)(dptr + 2);

    auto& quantizer1 = iq4kt_quantizer();
    auto& quantizer2 = iq4kt_quantizer(true);

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights);

    float amax_row = 0, row_av = 0;
    for (int j = 0; j < n_per_row; ++j) {
        row_av += x[j];
        amax_row = std::max(amax_row, std::abs(x[j]));
    }
    row_av /= n_per_row;
    dptr[1] = row_av;
    if (!amax_row) {
        dptr[0] = 0.f;
        std::memset(y, 0, nblock*sizeof(block_iq4_kt));
        return;
    }

    int   best_idx[2*Q::kNg];
    float xaux[Q::kBlockSize];

    float amax_scale = 0, max_scale = 0;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq4_kt));

        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                xaux[j] = xbl[ib*Q::kBlockSize+j] - row_av;
                float ax = std::abs(xaux[j]);
                amax = std::max(amax, ax);
            }
            if (!amax) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale_0 = std::max(92.f, 127.f*amax/amax_row);
            for (int itry = -kNtry; itry <= kNtry; ++itry) {
                quantizer1.find_best_match( amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dp, score_p] = quantizer1.find_best_scale(xaux, weight, best_idx);
                if (score_p > best) {
                    best = score_p; scales[ib] = dp;
                }
                quantizer1.find_best_match(-amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dm, score_m] = quantizer1.find_best_scale(xaux, weight, best_idx);
                if (score_m > best) {
                    best = score_m; scales[ib] = dm;
                }
            }

            quantizer2.find_best_match(scales[ib], xaux, weight, best_idx);
            auto [d, score] = quantizer2.find_best_scale(xaux, weight, best_idx);
            if (score > best) {
                scales[ib] = d;
                y[ibl].qs[ib] = 1;
            }
            bool with_offset = false;
            for (int itry = -kNtry; itry <= kNtry; ++itry) {
                quantizer2.find_best_match( amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dp, score_p] = quantizer2.find_best_scale(xaux, weight, best_idx);
                if (score_p > best) {
                    best = score_p; scales[ib] = dp; with_offset = true;
                }
                quantizer2.find_best_match(-amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dm, score_m] = quantizer2.find_best_scale(xaux, weight, best_idx);
                if (score_m > best) {
                    best = score_m; scales[ib] = dm; with_offset = true;
                }
            }
            if (with_offset) y[ibl].qs[ib] = 1;

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    float d = -max_scale/64;

    dptr[0] = d;
    if (!d) return;

    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;

    for (int iloop = 0; iloop < 1; ++iloop) {

        const float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            // high 3 bits + scales
            // each block of 32 needs 8 x 3 (high bits) + 1 x 8 (scale) = 32 bits = 1 x uint32_t
            // we have 8 blocks
            auto shb = y[ibl].qs;  // high 3 bits + scales
            auto ql = (uint8_t *)(shb + Q::kNblock);
            auto qh = ql + kNumGroups;
            std::memset(qh, 0, kNumGroups/2);
            const float * xbl = x + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                auto& quantizer = y[ibl].qs[ib] & 1 ? quantizer2 : quantizer1;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                for (int j = 0; j < Q::kBlockSize; ++j) xaux[j] = xbl[ib*Q::kBlockSize+j] - row_av;
                int ls = nearest_int(id*scales[ib]);
                ls = std::min(ls, 63);
                *(uint8_t *)(shb + ib) = ((ls + 64) << 1) | (shb[ib] & 1);
                float dl = d*ls;
                quantizer.find_best_match(dl, xaux, weight, best_idx);

                for (int j = 0; j < Q::kNg; ++j) {
                    shb[ib] |= ((best_idx[j] >> 12) << (8 + 3*j));
                    ql[Q::kNg*ib + j] = best_idx[j] & 255;
                    qh[(Q::kNg*ib + j)%(kNumGroups/2)] |= ((best_idx[j] >> 8) & 0xf) << 4*((Q::kNg*ib + j)/(kNumGroups/2));
                    auto xl = xaux + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + Q::kGroupSize*best_idx[j];
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            dptr[0] = d;
            if (!d) break;
        } else {
            break;
        }
    }
}
}

void quantize_row_iq4_kt_ref(const float * GGML_RESTRICT x, block_iq4_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_kt(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq4_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq4_kt * y = (block_iq4_kt *)vy;
    quantize_row_iq4_kt_ref(x, y, k);
}

size_t quantize_iq4_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KT, n_per_row);
    std::vector<float> scales(n_per_row/QuantizerIQ4KT::kBlockSize);
    std::vector<float> weights(n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq4_kt_impl(src, (void *)qrow, n_per_row, imatrix, scales.data(), weights.data());
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}

void dequantize_row_iq4_kt(const block_iq4_kt * x, float * y, int64_t k) {
    using Q = QuantizerIQ4KT;
    assert(k % Q::kSuperBlockSize == 0);
    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = dptr[0] * Q::kScale;
    const float row_av = dptr[1];
    x = (const block_iq4_kt *)(dptr + 2);
    auto& deq = iq4kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto shb = x[ibl].qs;
        auto ql = (const uint8_t *)(shb + Q::kNblock);
        auto qh = ql + kNumGroups;
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            int offset = shb[ib] & 1 ? 32768 + 4096 : 4096;
            //auto& deq = shb[ib] & 1 ? deq2 : deq1;
            int ls = int((shb[ib] & 0xff) >> 1) - 64;
            float sl = d * ls;
            for (int ig = 0; ig < Q::kNg; ++ig) {
                int jj = ib*Q::kNg+ig;
                uint16_t idx = ql[jj] | ((qh[jj%(kNumGroups/2)] << (8 - 4*(jj/(kNumGroups/2)))) & 0xf00) | (((shb[ib] >> (8 + 3*ig)) & 7) << 12);
                deq.set_values(idx, y, sl, offset);
                for (int j = 0; j < Q::kGroupSize; ++j) y[j] += row_av;
                y += Q::kGroupSize;
            }
        }
    }
}

void vec_dot_iq4_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_KT, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

}
