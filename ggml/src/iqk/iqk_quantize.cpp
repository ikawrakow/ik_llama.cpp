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
#include "iqk_config.h"

#include "iqk_gemm_ktquants.h"

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
#include <thread>
#include <atomic>
#include <unordered_map>
#include <string>
#include <functional>

namespace {

inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

typedef void (*quantize_func_t)(const float * src, void * qdata, int n_per_row, const float * imatrix);

struct QHelper {
    QHelper(const float * imatrix, int n_per_row, int block_size) : m_imatrix(imatrix),
        m_n_per_row(n_per_row), m_block_size(block_size) {
        if (m_imatrix) {
            m_weight.resize(m_n_per_row);
        }
    }
    const float * row_weights(const float * x) {
        constexpr float kEps  = 1e-9f;
        constexpr float kEps2 = kEps*kEps;
        if (!m_imatrix) return m_imatrix;
        int nblock = m_n_per_row / m_block_size;
        for (int ib = 0; ib < nblock; ++ib) {
            auto wb_in = m_imatrix + ib*m_block_size;
            auto xb = x + ib*m_block_size;
            auto wb = m_weight.data() + ib*m_block_size;
            float sumw2 = 0, sumx2 = 0, sumwx = 0;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = wb_in[j];
                sumw2 += wb[j]*wb[j];
                sumx2 += xb[j]*xb[j];
                sumwx += wb[j]*std::abs(xb[j]);
            }
            if (sumw2 > m_block_size*kEps2 && sumx2 > m_block_size*kEps2 && sumwx > m_block_size*kEps2) continue;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = kEps;
            }
        }
        return m_weight.data();
    }
    template <typename Func>
    void quantize(int nrows, const float * src, void * dst, int row_size, const Func& qfunc) {
        auto cdst = (char *)dst;
        for (int row = 0; row < nrows; ++row) {
            auto weights = row_weights(src);
            qfunc(src, cdst, m_n_per_row, weights);
            src  += m_n_per_row;
            cdst += row_size;
        }
    }
private:
    const float * m_imatrix;
    const int     m_n_per_row;
    const int     m_block_size;
    std::vector<float> m_weight;
};

template <int block_size, typename Block, typename Block_repacked, int n_repack, typename Func, typename RepackFunc>
size_t quantize_repack(ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const Func& q_func, const RepackFunc& repack) {
    GGML_ASSERT(nrows%n_repack == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(type, n_per_row);
    std::vector<char> qtmp(n_repack*row_size);
    QHelper helper(imatrix, n_per_row, block_size);
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += n_repack) {
        helper.quantize(n_repack, src, qtmp.data(), row_size, q_func);
        repack(n_repack, n_per_row, (const Block *)qtmp.data(), (Block_repacked *)qrow, false);
        src += n_repack*n_per_row;
        qrow += n_repack*row_size;
    }
    return nrows*row_size;
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

void iqk_quantize_any(int from_type, int to_type,
                      int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                      uint64_t nb0, uint64_t nb1, uint64_t nb2, uint64_t nb3,
                      const void * x, void * y, void * work_buffer,
                      to_float_t to_float, from_float_t from_float, int ith, int nth) {
    auto type_x = ggml_type(from_type);
    GGML_ASSERT(ggml_type_size(type_x) == nb0);
    auto type_y = ggml_type(to_type);
    auto row_size_y = ggml_row_size(type_y, ne0);
    int64_t nrows = ne1*ne2*ne3;
    int64_t nrows_per_thread = (nrows + nth - 1)/nth;
    int64_t first_row = nrows_per_thread*ith;
    if (first_row >= nrows) return;
    int64_t last_row = std::min(first_row + nrows_per_thread, nrows);
    for (int64_t row = first_row; row < last_row; ++row) {
        int64_t i3 = row/(ne1*ne2);
        int64_t i2 = (row - i3*ne1*ne2)/ne1;
        int64_t i1 = row - i3*ne1*ne2 - i2*ne1;
        const char * cx = (const char *)x + i1*nb1 + i2*nb2 + i3*nb3;
        // TODO: special case common types such as f16, q8_0
        //       (although the performance gains may be too small to justify the added complexity)
        to_float((const void *)cx, (float *)work_buffer, ne0);
        auto cy = (char *)y + (i3*ne1*ne2 + i2*ne1 + i1)*row_size_y;
        from_float((const float *)work_buffer, (void *)cy, ne0);
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

void quantize_row_q8_K16(const float * x, void * vy, int64_t nk) {
    float * dptr = (float *)vy;
    int8_t * qy = (int8_t *)(dptr + 5);
    int n64 = nk / 64;
#ifdef z__AVX2__
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
#elif defined z__ARM_NEON
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
    int sumi[4] = {};
    for (int i64 = 0; i64 < n64; ++i64) {
        for (int k = 0; k < 4; ++k) {
            for (int j = 0; j < 16; ++j) {
                int ix = nearest_int(amax[k]*x[64*i64 + 16*k + j]);
                sumi[k] += ix;
                qy[64*i64 + 16*k + j] = ix;
            }
        }
    }
    dptr[4] = dptr[0]*sumi[0] + dptr[1]*sumi[1] + dptr[2]*sumi[2] + dptr[3]*sumi[3];
#endif
}

void quantize_row_q8_0_x4(const float * x, void * vy, int64_t k) {
    const int nb = k / QK8_0;
    const int nb4 = 4*(nb/4);

    block_q8_0    * y  = (block_q8_0    *)vy;
    block_q8_0_x4 * y4 = (block_q8_0_x4 *)vy;
#if defined(__aarch64__)
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            if (i < nb4) {
                y4[i4].qs[32*ir + 4*j + 0] = vgetq_lane_s32(vi, 0);
                y4[i4].qs[32*ir + 4*j + 1] = vgetq_lane_s32(vi, 1);
                y4[i4].qs[32*ir + 4*j + 2] = vgetq_lane_s32(vi, 2);
                y4[i4].qs[32*ir + 4*j + 3] = vgetq_lane_s32(vi, 3);
            } else {
                y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
                y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
                y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
                y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
            }
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        const float d = maxScalar / 127.f;
        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *)y4[i4].qs + ir, i0);
        } else {
            _mm256_storeu_si256((__m256i *)y[i].qs, i0);
        }
    }
#endif
}

namespace {
template <typename Block, typename Block_x4>
void quantize_row_q8_1_x4_T(const float * x, Block * y, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    const int nb4 = 4*(nb/4);
    Block_x4 * y4 = (Block_x4 *)y;
#if defined(__aarch64__)
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        if (i < nb4) {
            y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
        } else {
            y[i].d = GGML_FP32_TO_FP16(d);
        }

        int32x4_t accv = vdupq_n_s32(0);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            if (i < nb4) {
                y4[i4].qs[QK8_1*ir + 4*j + 0] = vgetq_lane_s32(vi, 0);
                y4[i4].qs[QK8_1*ir + 4*j + 1] = vgetq_lane_s32(vi, 1);
                y4[i4].qs[QK8_1*ir + 4*j + 2] = vgetq_lane_s32(vi, 2);
                y4[i4].qs[QK8_1*ir + 4*j + 3] = vgetq_lane_s32(vi, 3);
            } else {
                y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
                y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
                y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
                y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
            }

            accv = vaddq_s32(accv, vi);
        }

        if constexpr (std::is_same_v<Block, block_q8_1>) {
            if (i < nb4) {
                y4[i4].d[ir+4] = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
            } else {
                y[i].s = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
            }
        } else {
            if (i < nb4) {
                y4[i4].d[ir+4] = GGML_FP32_TO_BF16(d * vaddvq_s32(accv)).bits;
            } else {
                y[i].s = GGML_FP32_TO_BF16(d * vaddvq_s32(accv)).bits;
            }
        }
    }
#else
    for (int i = 0; i < nb; i++) {
        int i4 = i/4, ir = i%4;
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float max_scalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        float d = max_scalar / 127.f;
        if constexpr (std::is_same_v<Block, block_q8_1>) {
            if (i < nb4) {
                y4[i4].d[ir] = GGML_FP32_TO_FP16(d);
            } else {
                y[i].d = GGML_FP32_TO_FP16(d);
            }
        } else {
            auto t = GGML_FP32_TO_BF16(d);
            d = ggml_bf16_to_fp32(t);
            if (i < nb4) {
                y4[i4].d[ir] = t.bits;
            } else {
                y[i].d = t.bits;
            }
        }
        const float id = d > 0 ? 1/d : 0.f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

        // Compute the sum of the quants and set y[i].s
        int isum = hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
        if constexpr (std::is_same_v<Block, block_q8_1>) {
            if (i < nb4) {
                y4[i4].d[ir+4] = GGML_FP32_TO_FP16(d * isum);
            } else {
                y[i].s = GGML_FP32_TO_FP16(d * isum);
            }
        } else {
            if (i < nb4) {
                auto i16 = (int16_t *)y4[i4].d;
                i16[ir+4] = isum;
            } else {
                auto i16 = (int16_t *)&y[i].s;
                i16[0] = isum;
            }
        }

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );  // 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );  // 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );  // 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        if (i < nb4) {
            _mm256_storeu_si256((__m256i *)y4[i4].qs + ir, i0);
        } else {
            _mm256_storeu_si256((__m256i *)y[i].qs, i0);
        }
    }
#endif
}
}

void quantize_row_q8_1_x4(const float * x, void * vy, int64_t k) {
    quantize_row_q8_1_x4_T<block_q8_1, block_q8_1_x4>(x, (block_q8_1 *)vy, k);
}

void quantize_row_q8_2_x4(const float * x, void * vy, int64_t k) {
    quantize_row_q8_1_x4_T<block_q8_2, block_q8_2_x4>(x, (block_q8_2 *)vy, k);
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
    QHelper helper(imatrix, n_per_row, 16);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_K, n_per_row);
    helper.quantize(nrows, src, dst, row_size, quantize_row_iq2_k_impl);
    return nrows * row_size;
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
    auto q_func = [&all_scales, &all_sw, &all_Ls] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq2_ks_impl(x, vy, n_per_row, imatrix, all_scales.data(), all_sw.data(), all_Ls.data());
    };
    QHelper helper(imatrix, n_per_row, kBlockSize);
    helper.quantize(nrows, src, dst, row_size, q_func);
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
// ======================================== iq2_kl
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

void quantize_row_iq2_kl_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales) {
    constexpr int kBlockSize = 32;
    constexpr float kSigmaFactor = 2.25f;
    constexpr int ntry = 5;
    static const int k_index[64] = {-1, -2, 0, -3, -4, 1, -5, -6, 2, -7, -8, 3, -9, 4, -10, 5, -11, 6, 7, -12, 8, 9, 10, -13, 11, -14, -15, -16, 12, 13, -17,
        14, -18, -19, 15, 16, 17, 18, 19, -20, -21, 20, 21, 22, 23, 24, -22, -23, 25, -24, 26, -25, 27, -26, 28, 29, -27, -28, 30, -29, -30, 31, -31, -32};
    static const std::vector<std::vector<int>> k_neighbours = {
        { 2, 0, 6, 11, 7, 3, 8, 15,  },
        { 0, 2, 3, 6, 7, 1, 8, 4,  },
        { 0, 1, 3, 4, 8, 7, 9, 6,  },
        { 1, 0, 3, 4, 8, 9, 7, 10,  },
        { 1, 4, 5, 10, 9, 3, 8, 0,  },
        { 5, 1, 4, 10, 9, 14, 8, 3,  },
        { 6, 2, 7, 0, 3, 11, 8, 15,  },
        { 3, 7, 0, 6, 8, 4, 12, 9,  },
        { 3, 4, 8, 9, 1, 7, 12, 10,  },
        { 4, 10, 5, 9, 1, 8, 13, 14,  },
        { 11, 2, 6, 7, 20, 15, 25, 21,  },
        { 8, 7, 3, 12, 9, 16, 17, 13,  },
        { 14, 5, 10, 19, 9, 13, 4, 18,  },
        { 6, 15, 7, 11, 20, 21, 16, 2,  },
        { 15, 7, 16, 6, 21, 12, 17, 22,  },
        { 12, 16, 17, 8, 15, 7, 13, 22,  },
        { 19, 10, 13, 18, 14, 9, 12, 24,  },
        { 11, 20, 25, 6, 15, 2, 21, 7,  },
        { 20, 15, 21, 6, 11, 7, 16, 26,  },
        { 14, 19, 29, 10, 28, 18, 13, 24,  },
        { 25, 11, 20, 21, 15, 6, 26, 30,  },
        { 19, 24, 28, 18, 29, 23, 13, 17,  },
        { 29, 19, 14, 28, 24, 18, 10, 13,  },
        { 20, 26, 21, 25, 30, 15, 22, 16,  },
        { 27, 26, 22, 23, 21, 30, 16, 24,  },
        { 27, 24, 28, 31, 23, 18, 22, 17,  },
        { 25, 30, 20, 26, 21, 11, 15, 22,  },
        { 30, 26, 25, 20, 21, 27, 22, 15,  },
        { 30, 27, 31, 26, 22, 23, 21, 24,  },
        { 31, 27, 30, 26, 28, 23, 22, 24,  },
        { 31, 28, 29, 27, 24, 23, 19, 18,  },
        { 29, 28, 31, 24, 19, 27, 14, 18,  },
    };
    auto values = iq3nl_values;
    std::pair<int8_t, int8_t> grid[32];
    for (int j = 0; j < 64; ++j) {
        if (int i = k_index[j]; i >= 0) {
            int i1 = j/8, i2 = j%8;
            grid[i] = {values[i1], values[i2]};
        }
    }

    ggml_half * dptr = (ggml_half *)vy;
    auto y = (block_iq2_kl *)(dptr + 1);

    float weight[kBlockSize];

    auto index = [&grid, values] (float id, float x1, float x2, float w1, float w2) {
        float sx1 = id*x1;
        float sx2 = id*x2;
        int l1 = best_index_iq3nl(values, sx1);
        int l2 = best_index_iq3nl(values, sx2);
        int i = k_index[8*l1 + l2];
        if (i >= 0) return i;
        auto& neigh = k_neighbours[-i-1];
        float best = std::numeric_limits<float>::max();
        int ibest = -1;
        for (auto& n : neigh) {
            float diff1 = grid[n].first  - sx1;
            float diff2 = grid[n].second - sx2;
            float score = w1*diff1*diff1 + w2*diff2*diff2;
            if (score < best) {
                best = score; ibest = n;
            }
        }
        GGML_ASSERT(ibest >= 0);
        return ibest;
    };

    float max_scale = 0, max_abs_scale = 0;

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        std::memset(&y[ibl], 0, sizeof(block_iq2_kl));
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= kSigmaFactor/QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            auto xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                auto qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]); //xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
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
            for (int j = 0; j < kBlockSize; j += 2) {
                float w1 = weight[j+0];
                float w2 = weight[j+1];
                int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                float q1 = grid[idx].first ;
                float q2 = grid[idx].second;
                sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_p += w1*q1*q1 + w2*q2*q2;
                idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                q1 = grid[idx].first ;
                q2 = grid[idx].second;
                sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_m += w1*q1*q1 + w2*q2*q2;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < kBlockSize; j += 2) {
                    float w1 = weight[j+0];
                    float w2 = weight[j+1];
                    int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                    float q1 = grid[idx].first ;
                    float q2 = grid[idx].second;
                    sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_p += w1*q1*q1 + w2*q2*q2;
                    idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                    q1 = grid[idx].first ;
                    q2 = grid[idx].second;
                    sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_m += w1*q1*q1 + w2*q2*q2;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m;
                }
            }
            scales[ib] = d;
            float ad = std::abs(d);
            if (ad > max_abs_scale) {
                max_abs_scale = ad; max_scale = d;
            }
        }
    }

    if (!max_abs_scale) {
        dptr[0] = GGML_FP32_TO_FP16(0.f);
        return;
    }

    float d = -max_scale/32;
    float id = 1/d;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= kSigmaFactor/QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            auto xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                auto qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]); //xb[j]*xb[j];
            }
            int ls = nearest_int(id*scales[ib]);
            ls = std::max(-32, std::min(31, ls));
            int lsmin = std::max(-32, ls-1);
            int lsmax = std::min( 31, ls+1);
            float best_score = std::numeric_limits<float>::max();
            int best_ls = ls;
            for (int ils = lsmin; ils <= lsmax; ++ils) {
                float dl = d*ils;
                float idl = dl ? 1/dl : 0.f;
                float score = 0;
                for (int j = 0; j < kBlockSize/2; ++j) {
                    float w1 = weight[2*j+0];
                    float w2 = weight[2*j+1];
                    int idx = index(idl, xb[2*j+0], xb[2*j+1], w1, w2);
                    float diff1 = dl*grid[idx].first  - xb[2*j+0];
                    float diff2 = dl*grid[idx].second - xb[2*j+1];
                    score += w1*diff1*diff1 + w2*diff2*diff2;
                }
                if (score < best_score) {
                    best_score = score;
                    best_ls = ils;
                }
            }
            ls = best_ls;
            int uls = ls + 32;
            y[ibl].scales_l[ib%4] |= ((uls & 0xf) << 4*(ib/4));
            y[ibl].scales_h |= ((uls >> 4) << 2*ib);
            if (ls == 0) continue;
            float dl = d*ls;
            float idl = 1/dl;
            for (int j = 0; j < kBlockSize/2; ++j) {
                float w1 = weight[2*j+0];
                float w2 = weight[2*j+1];
                int idx = index(idl, xb[2*j+0], xb[2*j+1], w1, w2);
                y[ibl].qs[16*(ib/2) + j] |= ((idx & 0xf) << 4*(ib%2));
                y[ibl].qh[j] |= ((idx >> 4) << ib);
                float q1 = ls*grid[idx].first ;
                float q2 = ls*grid[idx].second;
                sumqx += w1*q1*xb[2*j] + w2*q2*xb[2*j+1];
                sumq2 += w1*q1*q1 + w2*q2*q2;
            }
        }
    }
    if (sumq2 > 0) d = sumqx/sumq2;

    dptr[0] = GGML_FP32_TO_FP16(1.025f * d);

}
}

void quantize_row_iq2_kl_ref(const float * x, block_iq2_kl * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_kl(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq2_kl(const float * x, void * vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq2_kl * y = (block_iq2_kl *)vy;
    quantize_row_iq2_kl_ref(x, y, k);
}

size_t quantize_iq2_kl(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_KL, n_per_row);
    int nblock = n_per_row/QK_K;
    std::vector<float> all_scales(nblock*(QK_K/kBlockSize));
    auto q_func = [&all_scales] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq2_kl_impl(x, vy, n_per_row, imatrix, all_scales.data());
    };
    QHelper helper(imatrix, n_per_row, kBlockSize);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

void dequantize_row_iq2_kl(const block_iq2_kl  * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const ggml_half * dptr = (const ggml_half *)x;
    const float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq2_kl *)(dptr + 1);

    for (int i = 0; i < nb; i++) {

        auto qs = x[i].qs;
        auto qh = x[i].qh;
        auto scales_h = x[i].scales_h;

        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            float dl1 = d * (int(((x[i].scales_l[(2*ib64+0)%4] >> 4*(ib64/2)) & 0xf) | (((scales_h >> (4*ib64+0)) & 3) << 4)) - 32);
            float dl2 = d * (int(((x[i].scales_l[(2*ib64+1)%4] >> 4*(ib64/2)) & 0xf) | (((scales_h >> (4*ib64+2)) & 3) << 4)) - 32);
            for (int j = 0; j < 16; ++j) {
                const int8_t * val1 = (const int8_t *)(iq2kl_values + ((qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)));
                const int8_t * val2 = (const int8_t *)(iq2kl_values + ((qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)));
                y[2*j+ 0] = dl1 * val1[0];
                y[2*j+ 1] = dl1 * val1[1];
                y[2*j+32] = dl2 * val2[0];
                y[2*j+33] = dl2 * val2[1];
            }
            y  += 64;
            qs += 16;
        }

    }
}

void vec_dot_iq2_kl_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ2_KL, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
}

//
// ============================================== iq3_k
//
namespace {

static void quantize_row_iq3_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights) {

    constexpr int ntry = 3;

    block_iq3_k * y = (block_iq3_k *)vy;

    float scales[QK_K/16];
    float weight[16];
    uint8_t L[16];

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
            if (amax < 1e-9f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/iq3nl_values[0] : max/iq3nl_values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
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
            if (sumq2_p > 0) {
                d = sumqx_p/sumq2_p;
                best = d*sumqx_p;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (2*itry + iq3nl_values[0])/max;
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
                id = (2*itry + shifted_values[0])/max;
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
            if (!d) {
                scales[ib] = 0; continue;
            }

            const int8_t * block_values = is_shifted ? shifted_values : iq3nl_values;
            float sumqx = 0, sumq2 = 0;
            id = 1/d;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(block_values, al);
                L[j] = l;
                float q = block_values[l];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            if (sumq2 > 0) d = sumqx/sumq2;

            float best_d = d;
            for (int iter = 0; iter < 128; ++iter) {
                float gmax = 0;
                int best_j = -1, dir = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float g = d * w * (xb[j] - d*block_values[L[j]]);
                    if (g > 0 && L[j] < 7) {
                        if (g > gmax) {
                            gmax = g; best_j = j; dir = 1;
                        }
                    }
                    else if (g < 0 && L[j] > 0) {
                        if (-g > gmax) {
                            gmax = -g; best_j = j; dir = -1;
                        }
                    }
                }
                if (best_j < 0) break;

                float w = weight[best_j];
                sumqx += w*xb[best_j]*(block_values[L[best_j]+dir] - block_values[L[best_j]]);
                sumq2 += w*(block_values[L[best_j]+dir]*block_values[L[best_j]+dir] - block_values[L[best_j]]*block_values[L[best_j]]);
                L[best_j] += dir;
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    best_d = sumqx/sumq2; best = best_d*sumqx;
                }
                else if (iter > 8) break;

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
    QHelper helper(imatrix, n_per_row, 16);
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_K, n_per_row);
    helper.quantize(nrows, src, dst, row_size, quantize_row_iq3_k_impl);
    return nrows * row_size;
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
// ============================================== iq3_ks
//
namespace {
static void quantize_row_iq3_ks_impl(const int super_block_size, const int block_size,
        int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    ggml_half * dptr = (ggml_half *)cy;
    block_iq3_ks * y = (block_iq3_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 8;

    float amax_scale = 0;
    float max_scale = 0;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq3_ks));
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
            if (amax < 1e-9f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0) {
                d = sumqx_p/sumq2_p;
                best = d*sumqx_p;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(values, -al);
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
            if (is_shifted) y[ibl].extra |= (1 << (8 + ib));
            scales[ib] = d;
            float ascale = std::abs(d);
            if (ascale > amax_scale) {
                amax_scale = ascale; max_scale = d;
            }
        }
    }
    float d = -max_scale/16;
    *dptr = GGML_FP32_TO_FP16(d);
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = (y[ibl].extra >> (8 + ib)) & 0x01 ? shifted_values : values;
            int l = nearest_int(id*scales[ib]);
            l = std::max(-16, std::min(15, l));
            uint8_t ul = l + 16;
            y[ibl].scales[ib%4] |= (ul & 0xf) << 4*(ib/4);
            y[ibl].extra |= (ul >> 4) << ib;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            auto qs = y[ibl].qs + (ib/4)*block_size;
            auto qh = y[ibl].qh + (ib/8)*block_size;
            for (int j = 0; j < block_size; ++j) {
                uint8_t i = best_index_iq3nl(block_values, idl*xb[j]);
                qs[j] |= ((i &  3) << 2*(ib%4));
                qh[j] |= ((i >> 2) << (ib%8));
                float w = weight[j];
                float q = block_values[i]*l;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    if (sumq2 > 0) *dptr = GGML_FP32_TO_FP16(sumqx/sumq2);
}
}

void quantize_row_iq3_ks_ref(const float * x, block_iq3_ks * y, int64_t k) {
    quantize_iq3_ks(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq3_ks(const float * x, void * y, int64_t k) {
    quantize_iq3_ks(x, (void *)y, 1, k, nullptr);
}

size_t quantize_iq3_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_KS, n_per_row);
    QHelper helper(imatrix, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight, block_size = kBlockSize] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq3_ks_impl(QK_K, block_size, n_per_row, x, (char *)vy, all_scales.data(), weight, iq3nl_values, imatrix, 5);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

void dequantize_row_iq3_ks(const block_iq3_ks * x, float * y, int64_t k) {
    constexpr int kBlockSize = 32;
    static_assert(QK_K/kBlockSize == 8);
    GGML_ASSERT(k%QK_K == 0);
    const ggml_half * dptr = (const ggml_half *)x;
    float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq3_ks *)(dptr + 1);
    float dl[8];
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int j = 0; j < 4; ++j) {
            int ls1 = (x[ibl].scales[j] & 0xf) | (((x[ibl].extra >> (j+0)) & 1) << 4);
            int ls2 = (x[ibl].scales[j] >>  4) | (((x[ibl].extra >> (j+4)) & 1) << 4);
            dl[j+0] = d*(ls1 - 16);
            dl[j+4] = d*(ls2 - 16);
        }
        auto qs = x[ibl].qs;
        auto qh = x[ibl].qh;
        for (int i128 = 0; i128 < QK_K/128; ++i128) {
            for (int ib = 0; ib < 4; ++ib) {
                const int8_t * values = iq3nl_values + ((x[ibl].extra >> (8 + (4*i128+ib)) & 1) << 3);
                for (int j = 0; j < kBlockSize; ++j) {
                    y[j] = dl[4*i128 + ib] * values[((qs[j] >> 2*ib) & 3) | (((qh[j] >> (4*i128+ib)) & 1) << 2)];
                }
                y += kBlockSize;
            }
            qs += kBlockSize;
        }
    }
}

void  vec_dot_iq3_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_KS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_ABORT("Not implemented");
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
    uint8_t L[QK_K];
    float weight[16];
    float scales[QK_K/16];
    auto q_func = [&L, &weight, &scales] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        block_iq4_k * iq4 = (block_iq4_k *)vy;
        int nblock = n_per_row/QK_K;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw = imatrix ? imatrix + QK_K*ibl : nullptr;
            quantize_row_iq4_k_impl_bs16(QK_K, 16, x + QK_K*ibl, iq4 + ibl,
                    scales, weight, L, iq4k_values, qw, 7);
        }
    };
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_K, n_per_row);
    QHelper helper(imatrix, n_per_row, 16);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
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
    QHelper helper(imatrix, n_per_row, 16);
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_K, n_per_row);
    helper.quantize(nrows, src, dst, row_size, quantize_row_iq5_k_impl);
    return nrows * row_size;
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
    float values[128];
    for (int i = 0; i < 64; ++i) {
        values[i] = iq6nl_values[i];
        values[i+64] = values[i] + S_IQ6K;
    }
    auto q_func = [values] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq6_k_impl(x, vy, n_per_row, imatrix, values, values + 64);
    };
    auto row_size = ggml_row_size(GGML_TYPE_IQ6_K, n_per_row);
    QHelper helper(imatrix, n_per_row, 16);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

namespace {
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
        int block_sum_i32 = 0;
        float block_sum_f32 = 0;
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
            if constexpr (q8_type == 1) {
                int bsum = hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
                auto bs = (float *)y[i].bsums;
                bs[ib] = d*bsum;
                block_sum_f32 += bs[ib];
            } else {
                y[i].bsums[2*ib+0] = hsum_i32_8(_mm256_add_epi32(i0, i1));
                y[i].bsums[2*ib+1] = hsum_i32_8(_mm256_add_epi32(i2, i3));
                block_sum_i32 += y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1];
            }
            i0 = _mm256_packs_epi32( i0, i1 );
            i2 = _mm256_packs_epi32( i2, i3 );
            i0 = _mm256_packs_epi16( i0, i2 );
            i0 = _mm256_permutevar8x32_epi32( i0, perm );
            _mm256_storeu_si256((__m256i *)q8, i0);
            q8 += 32;
        }
        if constexpr (q8_type == 1) {
            y[i].sum = block_sum_f32;
        } else {
            y[i].sum = d*block_sum_i32;
        }
        //if constexpr (q8_type == 2) {
        //    auto bs = (float *)y[i].bsums;
        //    float sum = 0;
        //    for (int ib = 0; ib < QK_K/32; ++ib) sum += bs[ib];
        //    bs[0] = sum;
        //}
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
        float d = 1/iscale;
        if constexpr (q8_type == 1) {
            auto bs = (float *)y[i].bsums;
            float sum = 0;
            for (int j = 0; j < QK_K/32; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 32; ++ii) {
                    sum += y[i].qs[j*32 + ii];
                }
                bs[j] = d*sum;
                sum += bs[j];
            }
            y[i].sum = sum;
        } else {
            int tot = 0;
            for (int j = 0; j < QK_K/16; ++j) {
                int sum = 0;
                for (int ii = 0; ii < 16; ++ii) {
                    sum += y[i].qs[j*16 + ii];
                }
                y[i].bsums[j] = sum;
                tot += sum;
            }
            y[i].sum = d*tot;
        }
        y[i].d = d;
        x += QK_K;
    }
#endif
}
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
// TODO: merge this with the above template
void iqk_quantize_row_q8_K128(const float * x, void * vy, int64_t k) {
    constexpr int kBlockSize = 128;
    assert(k % kBlockSize == 0);
    const int nb = k / kBlockSize;
    auto y = (block_q8_K128 *)vy;
#ifdef __AVX2__
    const __m256 signBit = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int i = 0; i < nb; i++) {
        const float * xb = x + i*kBlockSize;
        __m256 maxAbs = _mm256_setzero_ps();
        const float * xx = xb;
        for (int ib = 0; ib < kBlockSize/8; ++ib) {
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
        for (int ib = 0; ib < kBlockSize/32; ++ib) {
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
            y[i].bsums[ib] = hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
            i0 = _mm256_packs_epi32( i0, i1 );
            i2 = _mm256_packs_epi32( i2, i3 );
            i0 = _mm256_packs_epi16( i0, i2 );
            i0 = _mm256_permutevar8x32_epi32( i0, perm );
            _mm256_storeu_si256((__m256i *)q8, i0);
            q8 += 32;
        }
    }
#elif defined __ARM_NEON
    int32x4_t ival[8];
    for (int i = 0; i < nb; i++) {
        const float * xb = x + i*kBlockSize;
        auto vmax = vdupq_n_f32(0.f);
        for (int j = 0; j < kBlockSize; j += 4) {
            vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(xb + j)));
        }
        auto smax = vmaxvq_f32(vmax);
        if (!smax) {
            std::memset(&y[i], 0, sizeof(y[i]));
            continue;
        }
        y[i].d = smax/127;
        auto vid = vdupq_n_f32(127/smax);
        for (int ib = 0; ib < kBlockSize/32; ++ib) {
            auto isum = vdupq_n_s32(0);
            for (int k = 0; k < 8; ++k) {
                auto val = vld1q_f32(xb + 32*ib + 4*k);
                ival[k] = vcvtnq_s32_f32(vmulq_f32(val, vid));
                isum = vaddq_s32(isum, ival[k]);
            }
            y[i].bsums[ib] = vaddvq_s32(isum);
            for (int k = 0; k < 4; ++k) {
                auto i16 = vcombine_s16(vmovn_s32(ival[2*k+0]), vmovn_s32(ival[2*k+1]));
                vst1_s8(y[i].qs + 32*ib + 8*k, vmovn_s16(i16));
            }
        }
    }
#else
    for (int i = 0; i < nb; i++) {

        float amax = 0;
        for (int j = 0; j < kBlockSize; ++j) {
            float ax = std::abs(x[j]);
            amax = std::max(amax, ax);
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, kBlockSize);
            memset(y[i].bsums, 0, kBlockSize/32*(sizeof(int16_t)));
            x += kBlockSize;
            continue;
        }
        const float iscale = 127.f/amax;
        for (int j = 0; j < kBlockSize; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = v;
        }
        for (int j = 0; j < kBlockSize/32; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 32; ++ii) {
                sum += y[i].qs[j*32 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += kBlockSize;
    }
#endif
}
// TODO: merge this with the above template
void iqk_quantize_row_q8_KV(const float * x, void * vy, int64_t k) {
    assert(k % 32 == 0);
    auto dptr = (float *)vy;
    auto q8 = (int8_t *)(dptr + 2);
#ifdef __AVX2__
    const __m256 signBit = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    __m256 maxAbs = _mm256_setzero_ps();
    for (int ib = 0; ib < k/8; ++ib) {
        const __m256 v = _mm256_loadu_ps(x + 8*ib);
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps(signBit, v));
    }
    const float maxScalar = hmax_f32_8(maxAbs);
    if (!maxScalar) {
        dptr[0] = dptr[1] = 0;
        std::memset(q8, 0, k*sizeof(int8_t));
        return;
    }
    dptr[0] = maxScalar / 127.f;
    auto mul = _mm256_set1_ps(1/dptr[0]);
    auto isum = _mm256_setzero_si256();
    for (int i = 0; i < k/32; i++) {
        __m256 v0 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i +  0));
        __m256 v1 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i +  8));
        __m256 v2 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i + 16));
        __m256 v3 = _mm256_mul_ps(mul, _mm256_loadu_ps(x + 32*i + 24));
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);
        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);
        isum = _mm256_add_epi32(isum, _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
        i0 = _mm256_packs_epi32( i0, i1 );
        i2 = _mm256_packs_epi32( i2, i3 );
        i0 = _mm256_packs_epi16( i0, i2 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );
        _mm256_storeu_si256((__m256i *)q8, i0);
        q8 += 32;
    }
    auto iptr = (int32_t *)(dptr + 1);
    iptr[0] = hsum_i32_8(isum);
#elif defined __ARM_NEON
    int32x4_t ival[8];
    auto vmax = vdupq_n_f32(0.f);
    for (int j = 0; j < k; j += 4) {
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(x + j)));
    }
    auto smax = vmaxvq_f32(vmax);
    if (!smax) {
        dptr[0] = dptr[1] = 0;
        std::memset(q8, 0, k*sizeof(int8_t));
        return;
    }
    dptr[0] = smax/127;
    auto vid = vdupq_n_f32(1/dptr[0]);
    auto isum = vdupq_n_s32(0);
    for (int ib = 0; ib < k/32; ++ib) {
        auto xb = x + 32*ib;
        for (int k = 0; k < 8; ++k) {
            auto val = vld1q_f32(xb + 4*k);
            ival[k] = vcvtnq_s32_f32(vmulq_f32(val, vid));
            isum = vaddq_s32(isum, ival[k]);
        }
        for (int k = 0; k < 4; ++k) {
            auto i16 = vcombine_s16(vmovn_s32(ival[2*k+0]), vmovn_s32(ival[2*k+1]));
            vst1_s8(q8, vmovn_s16(i16));
            q8 += 8;
        }
    }
    auto iptr = (int32_t *)(dptr + 1);
    iptr[0] = vaddvq_s32(isum);
#else
    float amax = 0;
    for (int j = 0; j < k; ++j) {
        float ax = std::abs(x[j]);
        amax = std::max(amax, ax);
    }
    if (!amax) {
        dptr[0] = dptr[1] = 0;
        std::memset(q8, 0, k*sizeof(int8_t));
        return;
    }
    dptr[0] = amax/127;
    float id = 1/dptr[0];
    int isum = 0;
    for (int i = 0; i < k; i++) {
        q8[i] = nearest_int(id*x[i]);
        isum += q8[i];
    }
    auto iptr = (int32_t *)(dptr + 1);
    iptr[0] = isum;
#endif
}
}

void quantize_row_q8_K128(const float * x, void * vy, int64_t k) {
    iqk_quantize_row_q8_K128(x, vy, k);
}

// ============================== MXFP4

namespace {
inline int best_index_mxfp4(float d, const int8_t * values, float x) {
    float best = std::abs(x - d*values[0]);
    int index = 0;
    for (int j = 1; j < 16; ++j) {
        float diff = std::abs(x - d*values[j]);
        if (diff < best) { best = diff; index = j; }
    }
    return index;
}
static void quantize_row_mxfp4_impl(int n_per_row, const float * x, char * cy,
        [[maybe_unused]] float * weight,
        const int8_t * values,
        [[maybe_unused]] const float * quant_weights,
        [[maybe_unused]] const int ntry) {

    GGML_ASSERT(n_per_row % QK_MXFP4 == 0);
    GGML_UNUSED(quant_weights);

    block_mxfp4 * y = (block_mxfp4 *)cy;

    //int last_ibl = -1;
    //float sigma2 = 0;

    //const uint8_t e = (uint8_t) (floorf(log2f(amax)) - 2 + 127);
    // -> log2f(amax) ~ e - 125 -> amax = 2^(e - 125)
    //const float d = GGML_E8M0_TO_FP32_HALF(e);

    for (int ib = 0; ib < n_per_row/QK_MXFP4; ++ib) {
        memset(&y[ib], 0, sizeof(block_mxfp4));
        const float * xb = x + ib*QK_MXFP4;
        //if (int ibl = ib/(QK_K/QK_MXFP4); ibl != last_ibl) {
        //    int n = std::min(QK_K, n_per_row - ib*QK_MXFP4);
        //    float sumx2 = 0;
        //    for (int j = 0; j < n; ++j) sumx2 += xb[j]*xb[j];
        //    sigma2 = 2.0f*sumx2/n;
        //    last_ibl = ibl;
        //}
        //if (quant_weights) {
        //    const float * qw = quant_weights + ib*QK_MXFP4;
        //    for (int j = 0; j < QK_MXFP4; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        //} else {
        //    for (int j = 0; j < QK_MXFP4; ++j) weight[j] = xb[j]*xb[j];
        //}
        float amax = 0;
        for (int j = 0; j < QK_MXFP4; ++j) {
            float ax = fabsf(xb[j]);
            amax = std::max(amax, ax);
        }
        if (!amax) {
            continue;
        }
        const uint8_t e = (uint8_t) (floorf(log2f(amax)) - 2 + 127);
        const float d = GGML_E8M0_TO_FP32_HALF(e);
        y[ib].e = e;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            uint8_t v0 = best_index_mxfp4(d, values, xb[j]);
            uint8_t v1 = best_index_mxfp4(d, values, xb[j+QK_MXFP4/2]);
            y[ib].qs[j] = v0 | (v1 << 4);
        }
    }
}
}

void quantize_row_mxfp4_ref(const float * x, block_mxfp4 * y, int64_t k) {
    quantize_mxfp4(x, (void *)y, 1, k, nullptr);
}

void quantize_row_mxfp4(const float * x, void * y, int64_t k) {
    quantize_mxfp4(x, (void *)y, 1, k, nullptr);
}

size_t quantize_mxfp4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = QK_MXFP4;
    GGML_ASSERT(n_per_row%kBlockSize == 0);
    auto row_size = ggml_row_size(GGML_TYPE_MXFP4, n_per_row);
    char * qrow = (char *)dst;
    float weight[kBlockSize];
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_mxfp4_impl(n_per_row, src, qrow, weight, kvalues_mxfp4, imatrix, 7);
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}

void dequantize_row_mxfp4(const block_mxfp4 * x, float * y, int64_t k) {
    constexpr int kBlockSize = QK_MXFP4;
    GGML_ASSERT(k%kBlockSize == 0);
    int nblock = k/kBlockSize;
    for (int ib = 0; ib < nblock; ++ib) {
        float d = GGML_E8M0_TO_FP32_HALF(x[ib].e);
        for (int j = 0; j < kBlockSize/2; ++j) {
            y[j             ] = d * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            y[j+kBlockSize/2] = d * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        y  += kBlockSize;
    }
}

void  vec_dot_mxfp4_q8_0_x4(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_MXFP4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_MXFP4 == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    //const block_mxfp4 * x = (const block_mxfp4 *)vx;
    //const block_q8_K  * y = (const block_q8_K    *)vy;
    //int nblock = n/QK_MXFP4;
    //float sumf = 0;
    //for (int ibl = 0; ibl < nblock; ++ibl) {
    //    //int sumi = 0;
    //    auto qy = y[ibl].qs;
    //    auto qx = x[ibl].qs;
    //    float db = d * y[ibl].d;
    //    for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
    //        float dl = db * ((x[ibl].scales[ib] & 254) - 127);
    //        //int ls = (x[ibl].scales[ib] & 254) - 127;
    //        const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
    //        int suml = 0;
    //        for (int j = 0; j < kBlockSize/2; ++j) {
    //            suml += qy[j               ] * values[qx[j] & 0xf]
    //                  + qy[j + kBlockSize/2] * values[qx[j] >>  4];
    //        }
    //        sumf += dl * suml;
    //        //sumi += ls * suml;
    //        qy += kBlockSize;
    //        qx += kBlockSize/2;
    //    }
    //    //sumf += d * y[ibl].d * sumi;
    //}
    //*s = sumf;
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
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    QHelper helper(imatrix, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight, block_size = kBlockSize] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq4_k_impl_bs128(QK_K, block_size, n_per_row, x, (char *)vy, all_scales.data(), weight, iq4k_values, imatrix, 7);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
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
static void quantize_row_iq5_ks_impl(const int super_block_size, const int block_size,
        int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    float * dptr = (float *)cy;
    dptr[0] = 0;
    block_iq5_ks * y = (block_iq5_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 32;

    float amax_scale = 0;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq5_ks));
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
            if (amax < 1e-15f) {
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
                int l = best_index_iq5nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq5nl(values, -al);
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
                    int l = best_index_iq5nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq5nl(values, -al);
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
            for (int j = 0; j < block_size; ++j) {
                uint8_t idx = best_index_iq5nl(block_values, idl*xb[j]);
                y[ibl].qs[block_size*(ib/2) + j] |= ((idx & 0xf) << 4*(ib%2));
                y[ibl].qh[j] |= ((idx >> 4) << ib);
                float w = weight[j];
                float q = block_values[idx]*l;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    if (sumq2 > 0) *dptr = sumqx/sumq2;
}
}

void quantize_row_iq5_ks_ref(const float * x, block_iq5_ks * y, int64_t k) {
    quantize_iq5_ks(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq5_ks(const float * x, void * y, int64_t k) {
    quantize_iq5_ks(x, (void *)y, 1, k, nullptr);
}

size_t quantize_iq5_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    QHelper helper(imatrix, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight, block_size = kBlockSize] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq5_ks_impl(QK_K, block_size, n_per_row, x, (char *)vy, all_scales.data(), weight, iq5nl_values, imatrix, 5);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

void dequantize_row_iq5_ks(const block_iq5_ks * x, float * y, int64_t k) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(k%QK_K == 0);
    const float * dptr = (const float *)x;
    float d = *dptr;
    x = (const block_iq5_ks *)(dptr + 1);
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qs = x[ibl].qs;
        auto qh = x[ibl].qh;
        for (int ib64 = 0; ib64 < QK_K/(2*kBlockSize); ++ib64) {
            float dl1 = d * ((int)(x[ibl].scales[2*ib64+0] & 254) - 127);
            float dl2 = d * ((int)(x[ibl].scales[2*ib64+1] & 254) - 127);
            const int8_t * values1 = iq5nl_values + ((x[ibl].scales[2*ib64+0] & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((x[ibl].scales[2*ib64+1] & 1) << 5);
            for (int j = 0; j < kBlockSize; ++j) {
                y[j           ] = dl1 * values1[(qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)];
                y[j+kBlockSize] = dl2 * values2[(qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)];
            }
            y  += 2*kBlockSize;
            qs += kBlockSize;
        }
    }
}

void  vec_dot_iq5_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    constexpr int kBlockSize = 32;
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ5_KS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
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
    const block_iq5_ks * x = (const block_iq5_ks *)(dptr + 1);
    const block_q8_K   * y = (const block_q8_K    *)vy;
    int nblock = n/QK_K;
    float sumf = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qy = y[ibl].qs;
        auto qs = x[ibl].qs;
        auto qh = x[ibl].qh;
        float db = d * y[ibl].d;
        for (int ib64 = 0; ib64 < QK_K/(2*kBlockSize); ++ib64) {
            float dl1 = db * ((int)(x[ibl].scales[2*ib64+0] & 254) - 127);
            float dl2 = db * ((int)(x[ibl].scales[2*ib64+1] & 254) - 127);
            const int8_t * values1 = iq5nl_values + ((x[ibl].scales[2*ib64+0] & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((x[ibl].scales[2*ib64+1] & 1) << 5);
            int suml1 = 0;
            int suml2 = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                suml1 += qy[j           ] * values1[(qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)];
                suml2 += qy[j+kBlockSize] * values2[(qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)];
            }
            sumf += dl1*suml1 + dl2*suml2;
            y  += 2*kBlockSize;
            qs += kBlockSize;
        }
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
        int qmin = std::max(int(q)-2,  0);
        int qmax = std::min(int(q)+2, 15);
        for (int iq = qmin; iq <= qmax; ++iq) {
            uint8_t qq = iq;
            if (qq == q) continue;
            int pci = popcount(qq);
            if (std::abs(pci - pc)%2) {
                float diff1 = dl*values[qq] - x[j];
                float score = w[j]*(diff1*diff1 - diff0*diff0);
                if (score < best_score) {
                    best_score = score; jbest = j; bestq = qq;
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
    if (sumq2 > 0) *dptr = sumqx/sumq2 * 1.01f;
}
}

size_t quantize_iq4_kss(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size    = ggml_row_size(GGML_TYPE_IQ4_KSS, n_per_row);
    std::vector<float> all_scales(n_per_row/kBlockSize);
    float weight[kBlockSize];
    auto table = scramble_table();
    QHelper helper(imatrix, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight, table] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq4_kss_impl(n_per_row, x, (char *)vy, all_scales.data(), weight, iq4k_values, imatrix, table, 7);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
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

static void repack_iq4_nl(int nrows, int n_per_row, const block_iq4_nl * x, block_iq4_nl_r4 * y, [[maybe_unused]] bool online) {
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
    QHelper helper(imatrix, n_per_row, 32);
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_iq4_nl(x, (char *)vy, 1, n_per_row, imatrix);
    };
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 4) {
        helper.quantize(4, src, qtmp.data(), row_size_nl, q_func);
        repack_iq4_nl(4, n_per_row, (const block_iq4_nl *)qtmp.data(), (block_iq4_nl_r4 *)qrow, false);
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
// ========================================= q4_0_r8
//
void quantize_row_q4_0_r8_ref(const float * x, block_iq4_nl_r8  * y, int64_t k) {
    // we assume we are called with 8 rows
    quantize_q4_0_r8(x, (void *)y, 8, k/8, nullptr);
}

void quantize_row_q4_0_r8(const float * x, void * y, int64_t k) {
    // we assume we are called with 8 rows
    quantize_q4_0_r8(x, y, 8, k/8, nullptr);
}

static void repack_q4_0(int nrows, int n_per_row, const block_q4_0 * x, block_iq4_nl_r8 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK4_0 == 0);
    int nblock = n_per_row/QK4_0;
    const block_q4_0 * x8[8];
    for (int row = 0; row < nrows; row += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            for (int k = 0; k < 8; ++k) {
                y[ib].d[k] = x8[k][ib].d;
                for (int l = 0; l < 4; ++l) {
                    for (int i = 0; i < 4; ++i) {
                        y[ib].qs[32*l+4*k+i] = x8[k][ib].qs[4*l + i];
                    }
                }
            }
#ifdef __ARM_NEON
            if (online) {
                for (int l = 0; l < 8; ++l) {
                    auto v = vld1q_u8(y[ib].qs + 16*l);
                    vst1q_u8(y[ib].qs + 16*l, veorq_u8(v, vdupq_n_u8(0x88)));
                }
            }
#endif
        }
        x += 8*nblock;
        y += nblock;
    }
}
#ifdef __ARM_NEON
static void modify_q4_0_r8(int64_t k, char * cy) {
    auto y = (block_iq4_nl_r8 *)cy;
    int nb = k/(32*8);
    for (int ib = 0; ib < nb; ++ib) {
        auto v1 = vld1q_u8_x4(y[ib].qs);
        auto v2 = vld1q_u8_x4(y[ib].qs+64);
        for (int j = 0; j < 4; ++j) {
            v1.val[j] = veorq_u8(v1.val[j], vdupq_n_u8(0x88));
            v2.val[j] = veorq_u8(v2.val[j], vdupq_n_u8(0x88));
        }
        vst1q_u8_x4(y[ib].qs+ 0, v1);
        vst1q_u8_x4(y[ib].qs+64, v2);
    }
}
#endif

size_t quantize_q4_0_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%8 == 0);
    auto row_size_nl = ggml_row_size(GGML_TYPE_Q4_0, n_per_row);
    std::vector<char> qtmp(8*row_size_nl);
    QHelper helper(imatrix, n_per_row, 32);
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_q4_0(x, (char *)vy, 1, n_per_row, imatrix);
    };
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 8) {
        helper.quantize(8, src, qtmp.data(), row_size_nl, q_func);
        repack_q4_0(8, n_per_row, (const block_q4_0 *)qtmp.data(), (block_iq4_nl_r8 *)qrow, false);
        src += 8*n_per_row;
        qrow += 8*row_size_nl;
    }
    return nrows*row_size_nl;
}

void dequantize_row_q4_0_r8(const block_iq4_nl_r8 * x, float * y, int64_t k) {
    // we assume we are called with 8 rows
    int n_per_row = k/8;
    int nb = n_per_row/QK4_0;
    float * yk[8];
    for (int k = 0; k < 8; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 8; ++k) {
            float scale = GGML_FP16_TO_FP32(x[ib].d[k]);
            for (int l = 0; l < 4; ++l) {
                for (int i = 0; i < 4; ++i) {
                    yk[k][QK4_0*ib+4*l+i+ 0] = scale * ((x[ib].qs[32*l+4*k+i] & 0xf) - 8);
                    yk[k][QK4_0*ib+4*l+i+16] = scale * ((x[ib].qs[32*l+4*k+i] >>  4) - 8);
                }
            }
        }
    }
}

void vec_dot_q4_0_r8_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q4_0_R8, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
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
// ========================================= q8_0_r8
//
void quantize_row_q8_0_r8_ref(const float * x, block_q8_0_r8  * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q8_0_r8(x, (void *)y, 8, k/8, nullptr);
}

void quantize_row_q8_0_r8(const float * x, void * y, int64_t k) {
    // we assume we are called with 4 rows
    quantize_q8_0_r8(x, y, 8, k/8, nullptr);
}

static void repack_q8_0(int nrows, int n_per_row, const block_q8_0 * x, block_q8_0_r8 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK8_0 == 0);
    int nblock = n_per_row/QK8_0;
    const block_q8_0 * x8[8];
    for (int row = 0; row < nrows; row += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = x + nblock*k;
        for (int ib = 0; ib < nblock; ++ib) {
            for (int k = 0; k < 8; ++k) y[ib].d[k] = x8[k][ib].d;
            for (int l = 0; l < 4; ++l) {
                for (int k = 0; k < 8; ++k) for (int i = 0; i < 4; ++i) {
                    y[ib].qs[32*l+4*k+i+  0] = x8[k][ib].qs[i+4*l+ 0];
                    y[ib].qs[32*l+4*k+i+128] = x8[k][ib].qs[i+4*l+16];
                }
            }
#ifdef HAVE_FANCY_SIMD
            if (online) {
                for (int l = 0; l < 4; ++l) {
                    auto v = _mm512_add_epi8(_mm512_loadu_si512((const __m512i *)y[ib].qs + l), _mm512_set1_epi8(127));
                    _mm512_storeu_si512((__m512i *)y[ib].qs + l, v);
                }
            }
#endif
        }
        x += 8*nblock;
        y += nblock;
    }
}

#ifdef HAVE_FANCY_SIMD
static void modify_q8_0_r8(int64_t k, char * cy) {
    auto y = (block_q8_0_r8 *)cy;
    int nb = k/(32*8);
    for (int ib = 0; ib < nb; ++ib) {
        for (int l = 0; l < 4; ++l) {
            auto v = _mm512_add_epi8(_mm512_loadu_si512((const __m512i *)y[ib].qs + l), _mm512_set1_epi8(127));
            _mm512_storeu_si512((__m512i *)y[ib].qs + l, v);
        }
    }
}
#endif

size_t quantize_q8_0_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%8 == 0);
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
    std::vector<char> qtmp(8*row_size_0);
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += 8) {
        quantize_q8_0(src, qtmp.data(), 8, n_per_row, imatrix);
        repack_q8_0(8, n_per_row, (const block_q8_0 *)qtmp.data(), (block_q8_0_r8 *)qrow, false);
        src += 8*n_per_row;
        qrow += 8*row_size_0;
    }
    return nrows*row_size_0;
}

void dequantize_row_q8_0_r8(const block_q8_0_r8 * x, float * y, int64_t k) {
    // we assume we are called with 4 rows
    int n_per_row = k/8;
    int nb = n_per_row/QK8_0;
    float * yk[8];
    for (int k = 0; k < 8; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nb; ++ib) {
        for (int k = 0; k < 8; ++k) {
            float scale = GGML_FP16_TO_FP32(x[ib].d[k]);
            for (int l = 0; l < 4; ++l) for (int i = 0; i < 4; ++i) {
                yk[k][QK8_0*ib+4*l+i+ 0] = scale * x[ib].qs[32*l+4*k+i+  0];
                yk[k][QK8_0*ib+4*l+i+16] = scale * x[ib].qs[32*l+4*k+i+128];
            }
        }
    }
}

void vec_dot_q8_0_r8_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q8_0_R8, vx, 0, GGML_TYPE_Q8_0, vy, 0, s, 0, 0, 1)) {
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

static void repack_q5_0(int nrows, int n_per_row, const block_q5_0 * x, block_q5_0_r4 * y, [[maybe_unused]] bool online) {
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
        repack_q5_0(4, n_per_row, (const block_q5_0 *)qtmp.data(), (block_q5_0_r4 *)qrow, false);
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

static void repack_q6_0(int nrows, int n_per_row, const block_q6_0 * x, block_q6_0_r4 * y, [[maybe_unused]] bool online) {
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
    QHelper helper(imatrix, n_per_row, 32);
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_q6_0(x, (char *)vy, 1, n_per_row, imatrix);
    };
    for (int row = 0; row < nrows; row += 4) {
        helper.quantize(4, src, qtmp.data(), row_size_0, q_func);
        repack_q6_0(4, n_per_row, (const block_q6_0 *)qtmp.data(), (block_q6_0_r4 *)qrow, false);
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
// ========================================= iq4_xs_r8
//

void quantize_row_iq4_xs_r8_ref(const float * x, block_iq4_xs_r8 * y, int64_t k) {
    quantize_iq4_xs_r8(x, (void *)y, 8, k/8, nullptr);
}

void quantize_row_iq4_xs_r8(const float * x, void * y, int64_t k) {
    quantize_iq4_xs_r8(x, y, 8, k/8, nullptr);
}

static void repack_iq4_xs(int nrows, int n_per_row, const block_iq4_xs * x, block_iq4_xs_r8 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq4_xs * x8[8];
    for (int row = 0; row < nrows; row += 8) {
        for (int k = 0; k < 8; ++k) x8[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales_l, 0, QK_K/8);
            std::memset(y[ibl].scales_h, 0, QK_K/16);
            for (int k = 0; k < 8; ++k) {
                y[ibl].d[k] = x8[k][ibl].d;
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    uint8_t sl = (x8[k][ibl].scales_l[ib/2] >> 4*(ib%2)) & 0xf;
                    uint8_t sh = (x8[k][ibl].scales_h >> 2*ib) & 3;
                    int i = 8*ib + k;
                    y[ibl].scales_l[i%32] |= (sl << 4*(i/32));
                    y[ibl].scales_h[i%16] |= (sh << 2*(i/16));
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[128*ib+4*k+i+ 0] = (x8[k][ibl].qs[16*ib+i+0] & 0xf) | ((x8[k][ibl].qs[16*ib+i+ 4] & 0xf) << 4);
                        y[ibl].qs[128*ib+4*k+i+32] = (x8[k][ibl].qs[16*ib+i+8] & 0xf) | ((x8[k][ibl].qs[16*ib+i+12] & 0xf) << 4);
                        y[ibl].qs[128*ib+4*k+i+64] = (x8[k][ibl].qs[16*ib+i+0] >>  4) | ((x8[k][ibl].qs[16*ib+i+ 4] >>  4) << 4);
                        y[ibl].qs[128*ib+4*k+i+96] = (x8[k][ibl].qs[16*ib+i+8] >>  4) | ((x8[k][ibl].qs[16*ib+i+12] >>  4) << 4);
                    }
                }
            }
        }
        x += 8*nblock;
        y += nblock;
    }
}

size_t quantize_iq4_xs_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_iq4_xs(x, (char *)vy, 1, n_per_row, imatrix);
    };
    return quantize_repack<32, block_iq4_xs, block_iq4_xs_r8, 8>(GGML_TYPE_IQ4_XS, src, dst, nrows, n_per_row, imatrix,
            q_func, repack_iq4_xs);
}

void dequantize_row_iq4_xs_r8(const block_iq4_xs_r8 * x, float * y, int64_t k) {
    auto n_per_row = k/8;
    float * y8[8];
    for (int k = 0; k < 8; ++k) y8[k] = y + n_per_row*k;
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 8; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int is = 8*ib + k;
                float dl = d * ((((x[ibl].scales_l[is%32] >> 4*(is/32)) & 0xf) | (((x[ibl].scales_h[is%16] >> 2*(is/16)) & 3) << 4)) - 32);
                for (int l = 0; l < 4; ++l) for (int i = 0; i < 4; ++i) {
                    y8[k][QK_K*ibl+32*ib+8*l+i+0] = dl * iq4k_values[x[ibl].qs[128*ib+4*k+i+32*l] & 0xf];
                    y8[k][QK_K*ibl+32*ib+8*l+i+4] = dl * iq4k_values[x[ibl].qs[128*ib+4*k+i+32*l] >>  4];
                }
            }
        }
    }
}

void vec_dot_iq4_xs_r8_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ4_XS_R8, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
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

static void repack_iq4_ks(int nrows, int n_per_row, const block_iq4_ks * x, block_iq4_ks_r4 * y, [[maybe_unused]] bool online) {
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
        repack_iq4_ks(4, n_per_row, (const block_iq4_ks *)qtmp.data(), (block_iq4_ks_r4 *)qcur, false);
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
void repack_iq2_bn(int nrows, int n_per_row, const char * x, char * y, [[maybe_unused]] bool online) {
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
        repack_iq2_bn(4, n_per_row, qtmp.data(), qcur, false);
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

static void repack_q4_k(int nrows, int n_per_row, const block_q4_K * x, block_q4_k_r4 * y, [[maybe_unused]] bool online) {
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
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_q4_K(x, (char *)vy, 1, n_per_row, imatrix);
    };
    return quantize_repack<32, block_q4_K, block_q4_k_r4, 4>(GGML_TYPE_Q4_K, src, dst, nrows, n_per_row, imatrix,
            q_func, repack_q4_k);
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

static void repack_q6_k(int nrows, int n_per_row, const block_q6_K * x, block_q6_k_r4 * y, [[maybe_unused]] bool online) {
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
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_q6_K(x, (char *)vy, 1, n_per_row, imatrix);
    };
    return quantize_repack<16, block_q6_K, block_q6_k_r4, 4>(GGML_TYPE_Q6_K, src, dst, nrows, n_per_row, imatrix,
            q_func, repack_q6_k);
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

static void repack_q5_k(int nrows, int n_per_row, const block_q5_K * x, block_q5_k_r4 * y, [[maybe_unused]] bool online) {
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
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_q5_K(x, (char *)vy, 1, n_per_row, imatrix);
    };
    return quantize_repack<32, block_q5_K, block_q5_k_r4, 4>(GGML_TYPE_Q5_K, src, dst, nrows, n_per_row, imatrix,
            q_func, repack_q5_k);
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

static void repack_q3_k(int nrows, int n_per_row, const block_q3_K * x, block_q3_k_r4 * y, [[maybe_unused]] bool online) {
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
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_q3_K(x, (char *)vy, 1, n_per_row, imatrix);
    };
    return quantize_repack<16, block_q3_K, block_q3_k_r4, 4>(GGML_TYPE_Q3_K, src, dst, nrows, n_per_row, imatrix,
            q_func, repack_q3_k);
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

static void repack_q2_k(int nrows, int n_per_row, const block_q2_K * x, block_q2_k_r4 * y, [[maybe_unused]] bool online) {
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
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_q2_K(x, (char *)vy, 1, n_per_row, imatrix);
    };
    return quantize_repack<16, block_q2_K, block_q2_k_r4, 4>(GGML_TYPE_Q2_K, src, dst, nrows, n_per_row, imatrix,
            q_func, repack_q2_k);
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

static void repack_iq4_k(int nrows, int n_per_row, const block_iq4_k * x, block_iq4_k_r4 * y, [[maybe_unused]] bool online) {
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
        repack_iq4_k(4, n_per_row, (const block_iq4_k *)qtmp.data(), (block_iq4_k_r4 *)qcur, false);
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
template <typename Block>
inline void convert_iq5_k(const Block& x, uint8_t * L) {
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

static void repack_iq5_k(int nrows, int n_per_row, const block_iq5_k * x, block_iq5_k_r4 * y, [[maybe_unused]] bool online) {
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
        repack_iq5_k(4, n_per_row, (const block_iq5_k *)qtmp.data(), (block_iq5_k_r4 *)qcur, false);
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
// ========================================= iq5_ks_r4
//

void quantize_row_iq5_ks_r4_ref(const float * x, block_iq5_ks_r4 * y, int64_t k) {
    quantize_iq5_ks_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq5_ks_r4(const float * x, void * y, int64_t k) {
    quantize_iq5_ks_r4(x, y, 4, k/4, nullptr);
}

static void repack_iq5_ks(int nrows, int n_per_row, const block_iq5_ks * x, block_iq5_ks_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    int nblock = n_per_row/QK_K;
    const block_iq5_ks * x4[4];
    uint8_t L[QK_K];
    char * cy = (char *)y;
    const char * cx = (const char *)x;
    for (int row = 0; row < nrows; row += 4) {
        float * dptr = (float *)cy;
        block_iq5_ks_r4 * y = (block_iq5_ks_r4 *)(dptr + 4);
        for (int k = 0; k < 4; ++k) {
            auto dk = (const float *)(cx + k*row_size);
            dptr[k] = dk[0];
            x4[k] = (const block_iq5_ks *)(dk + 1);
        }
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                convert_iq5_k(x4[k][ibl], L);
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].scales[4*ib+k] = x4[k][ibl].scales[ib];
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
        cx += 4*row_size;
        cy += 4*row_size;
    }
}

size_t quantize_iq5_ks_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    std::vector<char> qtmp(4*row_size);
    for (int row = 0; row < nrows; row += 4) {
        quantize_iq5_ks(src, (void *)qtmp.data(), 4, n_per_row, imatrix);
        repack_iq5_ks(4, n_per_row, (const block_iq5_ks *)qtmp.data(), (block_iq5_ks_r4 *)qcur, false);
        qcur += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq5_ks_r4(const block_iq5_ks_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    //auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    int nblock = n_per_row/QK_K;
    const float * dptr = (const float *)x;
    x = (const block_iq5_ks_r4 *)(dptr + 4);
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = dptr[k];
            //if (!isfinite(d)) {
            //    printf("Oops: d = %g for ibl = %d, k = %d\n", d, ibl, k); exit(1);
            //}
            for (int ib = 0; ib < QK_K/32; ++ib) {
                uint8_t sc = x[ibl].scales[4*ib+k];
                float dl = d * ((sc & 254) - 127);
                //if (!isfinite(dl)) {
                //    printf("Oops: dl = %g for ibl = %d, k = %d, ib = %d, d = %g, sc = %u\n", dl, ibl, k, ib, d, sc); exit(1);
                //}
                auto values = iq5nl_values + ((sc & 1) << 5);
                for (int i = 0; i < 4; ++i) {
                    y4[k][QK_K*ibl+32*ib+i+ 0] = dl * values[(x[ibl].qs[64*ib+4*k+i+ 0] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 0) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+ 8] = dl * values[(x[ibl].qs[64*ib+4*k+i+ 0] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 1) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+16] = dl * values[(x[ibl].qs[64*ib+4*k+i+16] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 2) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+24] = dl * values[(x[ibl].qs[64*ib+4*k+i+16] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 3) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+ 4] = dl * values[(x[ibl].qs[64*ib+4*k+i+32] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 4) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+12] = dl * values[(x[ibl].qs[64*ib+4*k+i+32] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 5) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+20] = dl * values[(x[ibl].qs[64*ib+4*k+i+48] & 0xf) | (((x[ibl].qh[16*ib+4*k+i] >> 6) & 1) << 4)];
                    y4[k][QK_K*ibl+32*ib+i+28] = dl * values[(x[ibl].qs[64*ib+4*k+i+48] >>  4) | (((x[ibl].qh[16*ib+4*k+i] >> 7) & 1) << 4)];
                }
                //for (int i = 0; i < 32; ++i) {
                //    if (!isfinite(y4[k][QK_K*ibl+32*ib+i])) {
                //        printf("Oops: y4[%d][%d, %d, %d] = %g\n", k, ibl, ib, i, y4[k][QK_K*ibl+32*ib+i]);
                //        printf("d = %g, dl = %g\n", d, dl);
                //        exit(1);
                //    }
                //}
            }
        }
    }
}

void vec_dot_iq5_ks_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ5_KS_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
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

static void repack_q8_k(int nrows, int n_per_row, const block_q8_K * x, block_q8_k_r8 * y, [[maybe_unused]] bool online) {
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
#ifdef HAVE_FANCY_SIMD
            if (online) {
                for (int l = 0; l < 32; ++l) {
                    auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[ibl].qs + l), _mm512_set1_epi8(-128));
                    _mm512_storeu_si512((__m512i *)y[ibl].qs + l, v);
                }
            }
#endif
        }
        x += 8*nblock;
        y += nblock;
    }
}
#ifdef HAVE_FANCY_SIMD
static void modify_q8_k_r8(int64_t k, char * cy) {
    auto y = (block_q8_k_r8 *)cy;
    int nb = k/(256*8);
    for (int ib = 0; ib < nb; ++ib) {
        for (int l = 0; l < 32; ++l) {
            auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[ib].qs + l), _mm512_set1_epi8(-128));
            _mm512_storeu_si512((__m512i *)y[ib].qs + l, v);
        }
    }
}
#endif

size_t quantize_q8_k_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row, [[maybe_unused]] const float * imatrix) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q8_K, n_per_row);
    auto row_size_1 = ggml_row_size(GGML_TYPE_Q8_K_R8, n_per_row);
    std::vector<char> qtmp(8*row_size_0);
    for (int row = 0; row < nrows; row += 8) {
        quantize_row_q8_K32(src, (void *)qtmp.data(), 8*n_per_row);
        repack_q8_k(8, n_per_row, (const block_q8_K *)qtmp.data(), (block_q8_k_r8 *)qcur, false);
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
// ========================================= q8_k_r16
//

void quantize_row_q8_k_r16_ref(const float * x, block_q8_k_r16 * y, int64_t k) {
    quantize_q8_k_r16(x, (void *)y, 16, k/16, nullptr);
}

void quantize_row_q8_k_r16(const float * x, void * y, int64_t k) {
    quantize_q8_k_r16(x, y, 16, k/16, nullptr);
}

static void repack_q16_k(int nrows, int n_per_row, const block_q8_K * x, block_q8_k_r16 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%16 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_q8_K * x16[16];
    for (int row = 0; row < nrows; row += 16) {
        for (int k = 0; k < 16; ++k) x16[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 16; ++k) {
                y[ibl].d[k] = GGML_FP32_TO_FP16(x16[k][ibl].d);
                for (int ib = 0; ib < QK_K/4; ++ib) {
                    for (int i = 0; i < 4; ++i) y[ibl].qs[64*ib + 4*k + i] = x16[k][ibl].qs[4*ib+i];
                }
            }
#ifdef HAVE_FANCY_SIMD
            for (int l = 0; l < 64; ++l) {
                auto v = _mm512_xor_si512(_mm512_loadu_si512((const __m512i *)y[ibl].qs + l), _mm512_set1_epi8(-128));
                _mm512_storeu_si512((__m512i *)y[ibl].qs + l, v);
            }
#endif
        }
        x += 16*nblock;
        y += nblock;
    }
}

size_t quantize_q8_k_r16(const float * src, void * dst, int64_t nrows, int64_t n_per_row, [[maybe_unused]] const float * imatrix) {
    GGML_ASSERT(nrows%16 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    char * qcur = (char *)dst;
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q8_K, n_per_row);
    auto row_size_1 = ggml_row_size(GGML_TYPE_Q8_K_R16, n_per_row);
    std::vector<char> qtmp(16*row_size_0);
    for (int row = 0; row < nrows; row += 16) {
        quantize_row_q8_K32(src, (void *)qtmp.data(), 16*n_per_row);
        repack_q16_k(16, n_per_row, (const block_q8_K *)qtmp.data(), (block_q8_k_r16 *)qcur, false);
        qcur += 16*row_size_1;
        src += 16*n_per_row;
    }
    return nrows*row_size_1;
}

void dequantize_row_q8_k_r16(const block_q8_k_r16 * x, float * y, int64_t k) {
    auto n_per_row = k/16;
    float * y16[16];
    for (int k = 0; k < 16; ++k) y16[k] = y + n_per_row*k;
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qs = (const uint8_t *)x[ibl].qs;
        for (int k = 0; k < 16; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            const float m = -128.f*d;
            for (int ib = 0; ib < QK_K/4; ++ib) {
                for (int i = 0; i < 4; ++i) {
                    y16[k][QK_K*ibl+4*ib+i] = d * qs[64*ib+4*k+i] + m;
                }
            }
        }
    }
}

void vec_dot_q8_k_r16_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q8_K_R16, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
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
// ========================================= q8_KV_r8
//

void quantize_row_q8_KV_r8_ref(const float * x, void * y, int64_t k) {
    quantize_q8_KV_r8(x, y, 8, k/8, nullptr);
}

void quantize_row_q8_KV_r8(const float * x, void * y, int64_t k) {
    quantize_q8_KV_r8(x, y, 8, k/8, nullptr);
}

static void repack_q8_KV(int nrows, int n_per_row, const char * cx, char * cy, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%16 == 0);
    auto row_size_x = ggml_row_size(GGML_TYPE_Q8_KV,    n_per_row);
    auto row_size_y = ggml_row_size(GGML_TYPE_Q8_KV_R8, n_per_row);
    const int8_t * x8[8];
#ifdef __ARM_NEON
    int8x16x2_t m0, m1, m2, m3;
#endif
    for (int row = 0; row < nrows; row += 8) {
        auto dy = (float *)cy;
        auto qy = (int8_t *)(dy + 8);
        for (int k = 0; k < 8; ++k) {
            auto dx = (const float *)(cx + k*row_size_x);
            dy[k] = dx[0];
            x8[k] = (const int8_t *)(dx + 2);
        }
        for (int ib = 0; ib < n_per_row/16; ++ib) {
#ifdef __AVX2__
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
            auto m0 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[4]+ib), _mm_loadu_si128((const __m128i *)x8[0]+ib));
            auto m1 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[5]+ib), _mm_loadu_si128((const __m128i *)x8[1]+ib));
            auto m2 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[6]+ib), _mm_loadu_si128((const __m128i *)x8[2]+ib));
            auto m3 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[7]+ib), _mm_loadu_si128((const __m128i *)x8[3]+ib));
            auto t0 = _mm256_unpacklo_epi32(m0, m1);
            auto t1 = _mm256_unpacklo_epi32(m2, m3);
            auto t2 = _mm256_unpackhi_epi32(m0, m1);
            auto t3 = _mm256_unpackhi_epi32(m2, m3);
            m0 = _mm256_unpacklo_epi64(t0, t1);
            m1 = _mm256_unpackhi_epi64(t0, t1);
            m2 = _mm256_unpacklo_epi64(t2, t3);
            m3 = _mm256_unpackhi_epi64(t2, t3);
#ifdef HAVE_FANCY_SIMD
            if (online) {
                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
            }
#endif
            _mm256_storeu_si256((__m256i *)qy + 4*ib+0, m0);
            _mm256_storeu_si256((__m256i *)qy + 4*ib+1, m1);
            _mm256_storeu_si256((__m256i *)qy + 4*ib+2, m2);
            _mm256_storeu_si256((__m256i *)qy + 4*ib+3, m3);
#elif defined __ARM_NEON
            m0.val[0] = vld1q_s8(x8[0]+16*ib); m0.val[1] = vld1q_s8(x8[4]+16*ib);
            m1.val[0] = vld1q_s8(x8[1]+16*ib); m1.val[1] = vld1q_s8(x8[5]+16*ib);
            m2.val[0] = vld1q_s8(x8[2]+16*ib); m2.val[1] = vld1q_s8(x8[6]+16*ib);
            m3.val[0] = vld1q_s8(x8[3]+16*ib); m3.val[1] = vld1q_s8(x8[7]+16*ib);
            auto row01 = vtrnq_s32(vreinterpretq_s32_s8(m0.val[0]), vreinterpretq_s32_s8(m1.val[0]));
            auto row23 = vtrnq_s32(vreinterpretq_s32_s8(m2.val[0]), vreinterpretq_s32_s8(m3.val[0]));
            m0.val[0] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
            m1.val[0] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
            m2.val[0] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
            m3.val[0] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
            row01 = vtrnq_s32(vreinterpretq_s32_s8(m0.val[1]), vreinterpretq_s32_s8(m1.val[1]));
            row23 = vtrnq_s32(vreinterpretq_s32_s8(m2.val[1]), vreinterpretq_s32_s8(m3.val[1]));
            m0.val[1] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
            m1.val[1] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
            m2.val[1] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
            m3.val[1] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
            vst1q_s8_x2(qy +  0 + 128*ib, m0);
            vst1q_s8_x2(qy + 32 + 128*ib, m1);
            vst1q_s8_x2(qy + 64 + 128*ib, m2);
            vst1q_s8_x2(qy + 96 + 128*ib, m3);
#else
            // TODO
            for (int l = 0; l < 4; ++l) {
                for (int k = 0; k < 8; ++k) for (int i = 0; i < 4; ++i) {
                    y[ib].qs[32*l+4*k+i+  0] = x8[k][ib].qs[i+4*l+ 0];
                    y[ib].qs[32*l+4*k+i+128] = x8[k][ib].qs[i+4*l+16];
                }
            }
#endif

        }
        cx += 8*row_size_x;
        cy += online ? 8*row_size_x : 8*row_size_y;
        //So, if we are run-time-repacking (online = true) we don't want to change the stride, so we just leave some unused space at the end of each row
    }
}
#ifdef HAVE_FANCY_SIMD
static void modify_q8_KV_r8(int64_t k, char * cy) {
    int8_t * q8 = (int8_t *)(cy + 8*sizeof(float));
    for (int j = 0; j < k; ++j) q8[j] += 127;
}
#endif

size_t quantize_q8_KV_r8(const float * src, void * dst, int64_t nrows, int64_t n_per_row, [[maybe_unused]] const float * imatrix) {
    GGML_ASSERT(nrows%8 == 0);
    GGML_ASSERT(n_per_row%16 == 0);
    char * qcur = (char *)dst;
    auto row_size_0 = ggml_row_size(GGML_TYPE_Q8_KV, n_per_row);
    auto row_size_1 = ggml_row_size(GGML_TYPE_Q8_KV_R8, n_per_row);
    std::vector<char> qtmp(8*row_size_0);
    for (int row = 0; row < nrows; row += 8) {
        quantize_q8_KV(src, (void *)qtmp.data(), 8, n_per_row, imatrix);
        repack_q8_KV(8, n_per_row, qtmp.data(), qcur, false);
        qcur += 8*row_size_1;
        src += 8*n_per_row;
    }
    return nrows*row_size_1;
}

void dequantize_row_q8_KV_r8(const void * vx, float * y, int64_t k) {
    auto n_per_row = k/8;
    float * y8[8];
    for (int k = 0; k < 8; ++k) y8[k] = y + n_per_row*k;
    auto dptr = (const float *)vx;
    auto q8 = (const int8_t *)(dptr + 8);
    for (int ib = 0; ib < n_per_row/16; ++ib) {
        for (int k = 0; k < 8; ++k) {
            for (int l = 0; l < 4; ++l) {
                for (int i = 0; i < 4; ++i) y8[k][16*ib + 4*l + i] = dptr[k] * q8[128*ib + 32*l + 4*k + i];
            }
        }
    }
}

void vec_dot_q8_KV_r8_q8_KV(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q8_KV_R8, vx, 0, GGML_TYPE_Q8_KV, vy, 0, s, 0, 0, 1)) {
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
inline ggml_bf16_t to_bf16(const ggml_half& x) { return to_bf16(GGML_FP16_TO_FP32(x)); }
inline ggml_bf16_t to_bf16(const ggml_bf16_t& x) { return x; }
template <typename T>
void repack_bf16(int nrows, int n_per_row, const T * x, ggml_bf16_t * y, [[maybe_unused]] bool online) {
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
    repack_bf16(nrows, n_per_row, (const float *)src, (ggml_bf16_t *)dst, false);
}

void repack_bf16_bf16_r16(const void * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row) {
    repack_bf16(nrows, n_per_row, (const ggml_bf16_t *)src, (ggml_bf16_t *)dst, false);
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

static void repack_iq3_k(int nrows, int n_per_row, const block_iq3_k * x, block_iq3_k_r4 * y, [[maybe_unused]] bool online) {
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
        repack_iq3_k(4, n_per_row, (const block_iq3_k *)qtmp.data(), (block_iq3_k_r4 *)qcur, false);
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

static void repack_iq2_k(int nrows, int n_per_row, const block_iq2_k * x, block_iq2_k_r4 * y, [[maybe_unused]] bool online) {
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
        repack_iq2_k(4, n_per_row, (const block_iq2_k *)qtmp.data(), (block_iq2_k_r4 *)qcur, false);
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

static void repack_iq2_xxs(int nrows, int n_per_row, const block_iq2_xxs * x, block_iq2_xxs_r4 * y, [[maybe_unused]] bool online) {
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
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_iq2_xxs(x, (char *)vy, 1, n_per_row, imatrix);
    };
    return quantize_repack<32, block_iq2_xxs, block_iq2_xxs_r4, 4>(GGML_TYPE_IQ2_XXS, src, dst, nrows, n_per_row, imatrix,
            q_func, repack_iq2_xxs);
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

static void repack_iq2_xs(int nrows, int n_per_row, const block_iq2_xs * x, block_iq2_xs_r4 * y, [[maybe_unused]] bool online) {
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
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_iq2_xs(x, (char *)vy, 1, n_per_row, imatrix);
    };
    return quantize_repack<16, block_iq2_xs, block_iq2_xs_r4, 4>(GGML_TYPE_IQ2_XS, src, dst, nrows, n_per_row, imatrix,
            q_func, repack_iq2_xs);
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

static void repack_iq2_s(int nrows, int n_per_row, const block_iq2_s * x, block_iq2_s_r4 * y, [[maybe_unused]] bool online) {
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
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_iq2_s(x, (char *)vy, 1, n_per_row, imatrix);
    };
    return quantize_repack<16, block_iq2_s, block_iq2_s_r4, 4>(GGML_TYPE_IQ2_S, src, dst, nrows, n_per_row, imatrix,
            q_func, repack_iq2_s);
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

static void repack_iq3_xxs(int nrows, int n_per_row, const block_iq3_xxs * x, block_iq3_xxs_r4 * y, [[maybe_unused]] bool online) {
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
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_iq3_xxs(x, (char *)vy, 1, n_per_row, imatrix);
    };
    return quantize_repack<32, block_iq3_xxs, block_iq3_xxs_r4, 4>(GGML_TYPE_IQ3_XXS, src, dst, nrows, n_per_row, imatrix,
            q_func, repack_iq3_xxs);
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

//
// ========================================= iq3_s_r4
//

void quantize_row_iq3_s_r4_ref(const float * x, block_iq3_s_r4 * y, int64_t k) {
    quantize_iq3_s_r4(x, (void *)y, 4, k/4, nullptr);
}

void quantize_row_iq3_s_r4(const float * x, void * y, int64_t k) {
    quantize_iq3_s_r4(x, y, 4, k/4, nullptr);
}

static void repack_iq3_s(int nrows, int n_per_row, const block_iq3_s * x, block_iq3_s_r4 * y, [[maybe_unused]] bool online) {
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    int nblock = n_per_row/QK_K;
    const block_iq3_s * x4[4];
    for (int row = 0; row < nrows; row += 4) {
        for (int k = 0; k < 4; ++k) x4[k] = x + nblock*k;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            std::memset(y[ibl].scales, 0, QK_K/16);
            std::memset(y[ibl].signs,  0, QK_K/2);
            std::memset(y[ibl].qh,     0, QK_K/8);
            for (int k = 0; k < 4; ++k) {
                y[ibl].d[k] = x4[k][ibl].d;
                for (int ib = 0; ib < QK_K/64; ++ib) {
                    int j = 8*ib + k;
                    y[ibl].scales[(j+0)%16] |= ((x4[k][ibl].scales[ib] & 0xf) << 4*((j+0)/16));
                    y[ibl].scales[(j+4)%16] |= ((x4[k][ibl].scales[ib] >>  4) << 4*((j+4)/16));
                }
                for (int ib = 0; ib < QK_K/32; ++ib) {
                    y[ibl].qh[4*ib+k] = x4[k][ibl].qh[ib]; // leave ot like this?
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[32*ib+k+8*i+0] = x4[k][ibl].qs[8*ib+i+0];
                        y[ibl].qs[32*ib+k+8*i+4] = x4[k][ibl].qs[8*ib+i+4];
                    }
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].signs[16*ib+4*k+i] = (((x4[k][ibl].signs[4*ib+0] >> i) & 1) << 0) | (((x4[k][ibl].signs[4*ib+0] >> (4+i)) & 1) << 1) |
                                                    (((x4[k][ibl].signs[4*ib+1] >> i) & 1) << 2) | (((x4[k][ibl].signs[4*ib+1] >> (4+i)) & 1) << 3) |
                                                    (((x4[k][ibl].signs[4*ib+2] >> i) & 1) << 4) | (((x4[k][ibl].signs[4*ib+2] >> (4+i)) & 1) << 5) |
                                                    (((x4[k][ibl].signs[4*ib+3] >> i) & 1) << 6) | (((x4[k][ibl].signs[4*ib+3] >> (4+i)) & 1) << 7);
                    }
                }
            }
        }
        x += 4*nblock;
        y += nblock;
    }
}

size_t quantize_iq3_s_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    auto q_func = [] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_iq3_s(x, (char *)vy, 1, n_per_row, imatrix);
    };
    return quantize_repack<16, block_iq3_s, block_iq3_s_r4, 4>(GGML_TYPE_IQ3_S, src, dst, nrows, n_per_row, imatrix,
            q_func, repack_iq3_s);
}

void dequantize_row_iq3_s_r4(const block_iq3_s_r4 * x, float * y, int64_t k) {
    auto n_per_row = k/4;
    float * y4[4] = {y, y + n_per_row, y + 2*n_per_row, y + 3*n_per_row};
    int nblock = n_per_row/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int k = 0; k < 4; ++k) {
            const float d = GGML_FP16_TO_FP32(x[ibl].d[k]);
            for (int ib = 0; ib < QK_K/32; ++ib) {
                int l = 4*ib + k;
                float dl = d * (1 + 2*((x[ibl].scales[l%16] >> 4*(l/16)) & 0xf));
                for (int i = 0; i < 4; ++i) {
                    auto grid1 = (const uint8_t *)(iq3s_grid + x[ibl].qs[32*ib+k+8*i+0] + ((x[ibl].qh[4*ib+k] << (8-i)) & 0x100));
                    auto grid2 = (const uint8_t *)(iq3s_grid + x[ibl].qs[32*ib+k+8*i+4] + ((x[ibl].qh[4*ib+k] << (4-i)) & 0x100));
                    for (int j = 0; j < 4; ++j) {
                        y4[k][QK_K*ibl+32*ib+4*i+ 0+j] = dl * grid1[j] * (x[ibl].signs[16*ib+4*k+j] & (1 << (i+0)) ? -1 : 1);
                        y4[k][QK_K*ibl+32*ib+4*i+16+j] = dl * grid2[j] * (x[ibl].signs[16*ib+4*k+j] & (1 << (i+4)) ? -1 : 1);
                    }
                }
            }
        }
    }
}

void vec_dot_iq3_s_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_S_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

void quantize_row_iq1_s_r4_ref(const float * x, block_iq1_s_r4  * y, int64_t k) {
    quantize_iq1_s_r4(x, y, 4, k/4, nullptr);
}

void quantize_row_iq1_s_r4(const float * x, void * y, int64_t k) {
    quantize_iq1_s_r4(x, y, 4, k/4, nullptr);
}

size_t quantize_iq1_s_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%kBlockSize == 0);
    int nblock = n_per_row/kBlockSize;
    float weight[kBlockSize];
    int8_t L[kBlockSize];
    float pairs[2*kBlockSize];
    float sumx[kBlockSize+1], sumw[kBlockSize+1];
    float max[4];
    uint16_t index[4];
    int shift;
    float invd[4];
    std::vector<float> scales(4*nblock);
    auto row_size = ggml_row_size(GGML_TYPE_IQ1_S_R4, n_per_row);
    char * cy = (char *)dst;
    for (int row = 0; row < nrows; row += 4) {
        ggml_half * dptr = (ggml_half *)cy;
        auto y = (block_iq1_s_r4 *)(dptr + 4);
        for (int k = 0; k < 4; ++k) max[k] = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                auto xb = src + k*n_per_row + kBlockSize*ibl;
                float sumx2 = 0;
                for (int j = 0; j < kBlockSize; ++j) sumx2 += xb[j]*xb[j];
                if (sumx2 < 1e-14f) {
                    //printf("Found block with all zeros\n");
                    // all zero
                    int ind = 1029; // this is the grid entry with all zeros
                    scales[4*ibl+k] = 0;
                    uint16_t h = 0;
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[4*i + k] = ind & 255;
                        h |= (ind >> 8) << 3*i;
                    }
                    y[ibl].qh[k] = h;
                    continue;
                }
                float sigma2 = 1.5f*sumx2/kBlockSize;
                bool have_imatrix = false;
                if (imatrix) {
                    have_imatrix = true;
                    float sumwx = 0;
                    for (int j = 0; j < kBlockSize; ++j) {
                        weight[j] = imatrix[kBlockSize*ibl + j]*sqrt(sigma2 + xb[j]*xb[j]);
                        sumwx += weight[j]*std::abs(xb[j]);
                    }
                    if (sumwx < 1e-14f) {
                        printf("Found block with mismatching importance/model weights\n");
                        // Either all weights are zero, or xb is zero where weight is not zero.
                        // In both of these cases it is better to simply ignore the imatrix
                        have_imatrix = false;
                    }
                }
                if (!have_imatrix) {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = sqrt(sigma2 + xb[j]*xb[j]);
                }
                iq1s_process_1block(kBlockSize, xb, weight, L, scales.data() + 4*ibl + k, index, &shift, pairs, sumx, sumw);
                GGML_ASSERT(scales[4*ibl+k] >= 0);
                max[k] = std::max(max[k], scales[4*ibl+k]);
                uint16_t h = 0;
                for (int i = 0; i < 4; ++i) {
                    GGML_ASSERT(index[i] >= 0 && index[i] < 2048);
                    y[ibl].qs[4*i + k] = index[i] & 255;
                    h |= (index[i] >> 8) << 3*i;
                }
                if (shift < 0) h |= 0x8000;
                y[ibl].qh[k] = h;
            }
        }
        for (int k = 0; k < 4; ++k) {
            dptr[k] = GGML_FP32_TO_FP16(1.0625f*max[k]/15);;
            invd[k] = max[k] ? 15/max[k] : 0.f;
        }
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                int ls = nearest_int(0.5f*(scales[4*ibl+k]*invd[k] - 1));
                ls = std::max(0, std::min(7, ls));
                y[ibl].qh[k] |= (ls << 12);
            }
        }
        cy  += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq1_s_r4(const block_iq1_s_r4  * x, float * y, int64_t n) {
    auto dptr = (const ggml_half *)x;
    x = (const block_iq1_s_r4 *)(dptr + 4);
    float d[4];
    for (int k = 0; k < 4; ++k) d[k] = GGML_FP16_TO_FP32(dptr[k]);
    int n_per_row = n/4;
    GGML_ASSERT(n_per_row%32 == 0);
    int nblock = n_per_row/32;
    float * yk[4];
    for (int k = 0; k < 4; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nblock; ++ib) {
        for (int k = 0; k < 4; ++k) {
            float shift = x[ib].qh[k] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
            float dl = d[k]*(2*((x[ib].qh[k] >> 12) & 7) + 1);
            for (int i = 0; i < 4; ++i) {
                auto idx = x[ib].qs[4*i+k] | (((x[ib].qh[k] >> 3*i) & 7) << 8);
                auto grid = (const int8_t *)(iq1s_grid + idx);
                for (int j = 0; j < 8; ++j) yk[k][32*ib + 8*i + j] = dl*(grid[j] + shift);
            }
        }
    }
}

void vec_dot_iq1_s_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ1_S_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

void quantize_row_iq1_m_r4_ref(const float * x, block_iq1_m_r4  * y, int64_t k) {
    quantize_iq1_m_r4(x, y, 4, k/4, nullptr);
}

void quantize_row_iq1_m_r4(const float * x, void * y, int64_t k) {
    quantize_iq1_m_r4(x, y, 4, k/4, nullptr);
}

size_t quantize_iq1_m_r4(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(nrows%4 == 0);
    GGML_ASSERT(n_per_row%kBlockSize == 0);
    int nblock = n_per_row/kBlockSize;
    float weight[kBlockSize];
    int8_t L[kBlockSize];
    float pairs[2*kBlockSize];
    float max[4];
    uint16_t index[4];
    int shift1, shift2;
    float invd[4];
    const uint8_t masks[4] = {0x00, 0x80, 0x08, 0x88};
    std::vector<float> scales(8*nblock);
    auto row_size = ggml_row_size(GGML_TYPE_IQ1_M_R4, n_per_row);
    char * cy = (char *)dst;
    for (int row = 0; row < nrows; row += 4) {
        ggml_half * dptr = (ggml_half *)cy;
        auto y = (block_iq1_m_r4 *)(dptr + 4);
        for (int k = 0; k < 4; ++k) max[k] = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                auto xb = src + k*n_per_row + kBlockSize*ibl;
                float sumx2l = 0, sumx2h = 0;
                for (int j = 0; j < kBlockSize/2; ++j) sumx2l += xb[j]*xb[j];
                for (int j = kBlockSize/2; j < kBlockSize; ++j) sumx2h += xb[j]*xb[j];
                float sumx2 = sumx2l + sumx2h;
                if (sumx2 < 1e-14f) {
                    scales[8*ibl+2*k+0] = scales[8*ibl+2*k+1] = 0;
                    int ind = 1029;
                    for (int i = 0; i < 4; ++i) {
                        y[ibl].qs[4*i + k] = ind & 255;
                    }
                    for (int i = 0; i < 2; ++i) {
                        y[ibl].qh[4*i+k] = (ind >> 8) | ((ind >> 8) << 4);
                    }
                    continue;
                }
                float sigma2 = 1.5f*sumx2/kBlockSize;
                if (imatrix) {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = imatrix[kBlockSize*ibl + j]*sqrt(sigma2 + xb[j]*xb[j]);
                    float sumwx = 0;
                    for (int j = 0; j < kBlockSize/2; ++j) sumwx += weight[j]*std::abs(xb[j]);
                    if (sumwx < 1e-14f) {
                        for (int j = 0; j < kBlockSize/2; ++j) weight[j] = sqrt(sigma2 + xb[j]*xb[j]);
                    }
                    sumwx = 0;
                    for (int j = kBlockSize/2; j < kBlockSize; ++j) sumwx += weight[j]*std::abs(xb[j]);
                    if (sumwx < 1e-14) {
                        for (int j = kBlockSize/2; j < kBlockSize; ++j) weight[j] = sqrt(sigma2 + xb[j]*xb[j]);
                    }
                } else {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = sqrt(sigma2 + xb[j]*xb[j]);
                }
                if (sumx2l > 1e-14f) {
                    iq1m_process_1block(xb+ 0, weight+ 0, L, scales.data() + 8*ibl + 2*k+0, index+0, &shift1, pairs);
                } else {
                    scales[8*ibl+2*k+0] = 0;
                    index[0] = index[1] = 1029;
                }
                if (sumx2h > 1e-14f) {
                    iq1m_process_1block(xb+16, weight+16, L, scales.data() + 8*ibl + 2*k+1, index+2, &shift2, pairs);
                } else {
                    scales[8*ibl+2*k+1] = 0;
                    index[2] = index[3] = 1029;
                }
                max[k] = std::max(max[k], std::max(scales[8*ibl+2*k+0], scales[8*ibl+2*k+1]));
                for (int i = 0; i < 4; ++i) {
                    y[ibl].qs[4*i + k] = index[i] & 255;
                }
                for (int i = 0; i < 2; ++i) {
                    y[ibl].qh[4*i+k] = (index[2*i+0] >> 8) | ((index[2*i+1] >> 8) << 4);
                }
                y[ibl].qh[0+k] |= masks[shift1];
                y[ibl].qh[4+k] |= masks[shift2];
            }
        }
        for (int k = 0; k < 4; ++k) {
            dptr[k] = GGML_FP32_TO_FP16(1.0625f*max[k]/15);;
            invd[k] = max[k] ? 15/max[k] : 0.f;
        }
        for (int ibl = 0; ibl < nblock; ++ibl) {
            for (int k = 0; k < 4; ++k) {
                int ls1 = nearest_int(scales[8*ibl+2*k+0]*invd[k]);
                int ls2 = nearest_int(scales[8*ibl+2*k+1]*invd[k]);
                ls1 = std::max(0, std::min(15, ls1));
                ls2 = std::max(0, std::min(15, ls2));
                y[ibl].scales[k] = ls1 | (ls2 << 4);
            }
        }
        cy  += 4*row_size;
        src += 4*n_per_row;
    }
    return nrows*row_size;
}

void dequantize_row_iq1_m_r4(const block_iq1_m_r4  * x, float * y, int64_t n) {
    auto dptr = (const ggml_half *)x;
    x = (const block_iq1_m_r4 *)(dptr + 4);
    float d[4];
    for (int k = 0; k < 4; ++k) d[k] = GGML_FP16_TO_FP32(dptr[k]);
    int n_per_row = n/4;
    GGML_ASSERT(n_per_row%32 == 0);
    int nblock = n_per_row/32;
    float dl[2];
    float * yk[4];
    for (int k = 0; k < 4; ++k) yk[k] = y + k*n_per_row;
    for (int ib = 0; ib < nblock; ++ib) {
        for (int k = 0; k < 4; ++k) {
            dl[0] = d[k]*(x[ib].scales[k] & 0xf);
            dl[1] = d[k]*(x[ib].scales[k] >>  4);
            for (int i = 0; i < 2; ++i) {
                auto idx1 = x[ib].qs[8*i+k+0] | ((x[ib].qh[4*i+k] & 0x07) << 8);
                auto idx2 = x[ib].qs[8*i+k+4] | ((x[ib].qh[4*i+k] & 0x70) << 4);
                auto grid1 = (const int8_t *)(iq1s_grid + idx1);
                auto grid2 = (const int8_t *)(iq1s_grid + idx2);
                auto delta1 = x[ib].qh[4*i+k] & 0x08 ? -IQ1M_DELTA : IQ1M_DELTA;
                auto delta2 = x[ib].qh[4*i+k] & 0x80 ? -IQ1M_DELTA : IQ1M_DELTA;
                for (int j = 0; j < 8; ++j) yk[k][32*ib + 16*i + j + 0] = dl[i]*(grid1[j] + delta1);
                for (int j = 0; j < 8; ++j) yk[k][32*ib + 16*i + j + 8] = dl[i]*(grid2[j] + delta2);
            }
        }
    }
}

void vec_dot_iq1_m_r4_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ1_M_R4, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}

void quantize_row_q8_KV(const float * x, void * vy, int64_t k) {
    iqk_quantize_row_q8_KV(x, vy, k);
}

void quantize_row_q8_KV_ref(const float * x, void * y, int64_t k) {
    quantize_row_q8_KV(x, y, k);
}

size_t quantize_q8_KV(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    auto row_size = ggml_row_size(GGML_TYPE_Q8_KV, n_per_row);
    auto q = (char *)dst;
    for (int row = 0; row < nrows; ++row) {
        quantize_row_q8_KV(src, q, n_per_row);
        src += n_per_row;
        q += row_size;
    }
    return row_size*nrows;
}

void dequantize_row_q8_KV(const void * x, float * y, int64_t k) {
    auto dptr = (const float *)x;
    float d = dptr[0];
    auto q8 = (const int8_t *)(dptr + 2);
    for (int j = 0; j < k; ++j) y[j] = d * q8[j];
}

void vec_dot_q8_KV_q8_KV(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_Q8_KV, vx, 0, GGML_TYPE_Q8_KV, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK4_NL == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
}


//================================================

namespace {
struct Repack {
    using repack_func = void (*) (int nrows, int n_per_row, const char * src, char * dst, bool online);
    ggml_type   new_type;
    int         num_rows;
    repack_func repack;
};
struct Modify {
    using modify_func_t = void (*)(int64_t k, char * src_dst);
    modify_func_t  mod_func;
    int            nrows;
};
const Modify * get_modify_info(ggml_type type) {
    static const std::unordered_map<ggml_type, Modify> k_mod_map = {
#ifdef __ARM_NEON
        { GGML_TYPE_Q4_0_R8, {modify_q4_0_r8, 8} },
#endif
#ifdef HAVE_FANCY_SIMD
        { GGML_TYPE_Q8_0_R8,  {modify_q8_0_r8,  8} },
        { GGML_TYPE_Q8_K_R8,  {modify_q8_k_r8,  8} },
        { GGML_TYPE_Q8_KV_R8, {modify_q8_KV_r8, 8} },
#endif
    };
    auto it = k_mod_map.find(type);
    return it != k_mod_map.end() ? &it->second : nullptr;
}
bool is_forbidden_tensor(const std::string& name) {
    static const std::string kTokenEmbd{"token_embd.weight"};
    if (name == kTokenEmbd) return true;
    //if (auto pos = name.find("attn_kv_b.weight"); pos != std::string::npos) return true;
    return false;
}
}

bool iqk_should_modify_tensor([[maybe_unused]] const struct ggml_tensor * tensor) {
    return false;
    //if (is_forbidden_tensor(tensor->name)) return false;
    //auto mptr = get_modify_info(tensor->type);
    //return mptr ? true : false;
}

bool iqk_modify_tensor(struct ggml_tensor * tensor) {
    return false;
    auto mptr = get_modify_info(tensor->type);
    if (!mptr) return false;
    if (is_forbidden_tensor(std::string{tensor->name})) return false;

    auto& m = *mptr;
    int nrows = ggml_nrows(tensor);
    int nchunks = nrows/m.nrows;
    int max_thread = std::max(1, int(std::thread::hardware_concurrency()/2));
    int nthread = std::min(nchunks, max_thread);
    auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
    std::atomic<int> counter(0);
    auto compute = [&counter, &m, tensor, row_size, nchunks] () {
        int64_t n_per_call = m.nrows*tensor->ne[0];
        while (true) {
            int row = counter.fetch_add(1);
            if (row >= nchunks) break;
            m.mod_func(n_per_call, (char *)tensor->data + row_size*row*m.nrows);
        }
    };
    std::vector<std::thread> workers(nthread-1);
    for (auto& w : workers) w = std::thread(compute);
    compute();
    for (auto& w : workers) w.join();

    return true;
}

namespace {
const Repack * get_repack_info(ggml_type type) {
    static const std::unordered_map<ggml_type, Repack> k_map = {
        { GGML_TYPE_IQ2_K,  { GGML_TYPE_IQ2_K_R4,  4,  (Repack::repack_func)repack_iq2_k}   },
        { GGML_TYPE_IQ3_K,  { GGML_TYPE_IQ3_K_R4,  4,  (Repack::repack_func)repack_iq3_k}   },
        { GGML_TYPE_IQ4_K,  { GGML_TYPE_IQ4_K_R4,  4,  (Repack::repack_func)repack_iq4_k}   },
        { GGML_TYPE_IQ5_K,  { GGML_TYPE_IQ5_K_R4,  4,  (Repack::repack_func)repack_iq5_k}   },
        { GGML_TYPE_IQ4_XS, { GGML_TYPE_IQ4_XS_R8, 8,  (Repack::repack_func)repack_iq4_xs}  },
        { GGML_TYPE_IQ4_KS, { GGML_TYPE_IQ4_KS_R4, 4,  (Repack::repack_func)repack_iq4_ks}  },
        { GGML_TYPE_IQ5_KS, { GGML_TYPE_IQ5_KS_R4, 4,  (Repack::repack_func)repack_iq5_ks}  },
        { GGML_TYPE_IQ4_NL, { GGML_TYPE_IQ4_NL_R4, 4,  (Repack::repack_func)repack_iq4_nl}  },
        { GGML_TYPE_IQ2_BN, { GGML_TYPE_IQ2_BN_R4, 4,  (Repack::repack_func)repack_iq2_bn}  },
        { GGML_TYPE_IQ2_XXS,{ GGML_TYPE_IQ2_XXS_R4,4,  (Repack::repack_func)repack_iq2_xxs} },
        { GGML_TYPE_IQ2_XS, { GGML_TYPE_IQ2_XS_R4, 4,  (Repack::repack_func)repack_iq2_xs}  },
        { GGML_TYPE_IQ2_S,  { GGML_TYPE_IQ2_S_R4,  4,  (Repack::repack_func)repack_iq2_s}   },
        { GGML_TYPE_IQ3_XXS,{ GGML_TYPE_IQ3_XXS_R4,4,  (Repack::repack_func)repack_iq3_xxs} },
        { GGML_TYPE_IQ3_S,  { GGML_TYPE_IQ3_S_R4,  4,  (Repack::repack_func)repack_iq3_s}   },
        { GGML_TYPE_Q2_K,   { GGML_TYPE_Q2_K_R4,   4,  (Repack::repack_func)repack_q2_k}    },
        { GGML_TYPE_Q3_K,   { GGML_TYPE_Q3_K_R4,   4,  (Repack::repack_func)repack_q3_k}    },
        { GGML_TYPE_Q4_K,   { GGML_TYPE_Q4_K_R4,   4,  (Repack::repack_func)repack_q4_k}    },
        { GGML_TYPE_Q5_K,   { GGML_TYPE_Q5_K_R4,   4,  (Repack::repack_func)repack_q5_k}    },
        { GGML_TYPE_Q6_K,   { GGML_TYPE_Q6_K_R4,   4,  (Repack::repack_func)repack_q6_k}    },
        { GGML_TYPE_Q4_0,   { GGML_TYPE_Q4_0_R8,   8,  (Repack::repack_func)repack_q4_0}    },
        { GGML_TYPE_Q5_0,   { GGML_TYPE_Q5_0_R4,   4,  (Repack::repack_func)repack_q5_0}    },
        { GGML_TYPE_Q6_0,   { GGML_TYPE_Q6_0_R4,   4,  (Repack::repack_func)repack_q6_0}    },
        { GGML_TYPE_Q8_0,   { GGML_TYPE_Q8_0_R8,   8,  (Repack::repack_func)repack_q8_0}    },
        { GGML_TYPE_Q8_K,   { GGML_TYPE_Q8_K_R8,   8,  (Repack::repack_func)repack_q8_k}    },
        { GGML_TYPE_Q8_KV,  { GGML_TYPE_Q8_KV_R8,  8,  (Repack::repack_func)repack_q8_KV}   },
#ifdef __AVX512BF16__
        { GGML_TYPE_BF16,   { GGML_TYPE_BF16_R16, 16,  (Repack::repack_func)repack_bf16<ggml_bf16_t>}},
        { GGML_TYPE_F16,    { GGML_TYPE_BF16_R16, 16,  (Repack::repack_func)repack_bf16<ggml_half>}  },
#endif
    };
    auto it = k_map.find(type);
    return it != k_map.end() ? &it->second : nullptr;
}
}

int iqk_repacked_type(const struct ggml_tensor * tensor) {
    if (!ggml_is_contiguous(tensor)) return (int)tensor->type;
    if (is_forbidden_tensor(tensor->name)) return (int)tensor->type;
    auto rptr = get_repack_info(tensor->type);
    return rptr && tensor->ne[1] % rptr->num_rows == 0 ? (int)rptr->new_type : (int)tensor->type;
}

void iqk_repack_tensor(struct ggml_tensor * tensor) {
    constexpr int kChunk = 8;
    if (!tensor) return;
    if (!ggml_is_contiguous(tensor)) return;
    if (is_forbidden_tensor(tensor->name)) return;
    if (tensor->ne[1] % 4) return;

    auto rptr = get_repack_info(tensor->type);
    if (!rptr) return;
    if (tensor->ne[1] % rptr->num_rows) return;

    auto& r = *rptr;

    auto nrows = ggml_nrows(tensor);

    int max_thread = std::max(1, int(std::thread::hardware_concurrency()/2));
    int num_chunks = (nrows + kChunk*r.num_rows - 1)/(kChunk*r.num_rows);
    int nthread = std::min(num_chunks, max_thread);

    //printf("%s(%s): %s -> %s. %d rows, %d chunks, %d threads\n", __func__, tensor->name, ggml_type_name(tensor->type), ggml_type_name(r.new_type),
    //        int(tensor->ne[1]), num_chunks, nthread);

    std::atomic<int> counter(0);;
    auto compute = [&counter, &r, tensor, num_chunks, chunkSize = kChunk] () {
        int nrows = ggml_nrows(tensor);
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
                //r.repack(r.num_rows, n_per_row, qtmp.data(), data + row*row_size, true);
                r.repack(r.num_rows, n_per_row, qtmp.data(), data + row*row_size, false);
            }
        }
    };
    std::vector<std::thread> workers(nthread-1);
    for (auto& w : workers) w = std::thread(compute);
    compute();
    for (auto& w : workers) w.join();

    tensor->type = r.new_type;
}

void dequantize_row_ms_i2s(const void * vx, float * y, int64_t k) {
    constexpr int kBlockSize = 128;
    constexpr int kGroupSize = kBlockSize/4;
    GGML_ASSERT(k % kBlockSize == 0);
    const uint8_t * x = (const uint8_t *)vx;
    const float * dptr = (const float *)(x + k/4);
    const float d = dptr[0];
    int nb = k/kBlockSize;
    for (int ib = 0; ib < nb; ++ib) {
        for (int ig = 0; ig < kBlockSize/kGroupSize; ++ig) {
            int shift = 6 - 2*ig;
            for (int j = 0; j < kGroupSize; ++j) {
                y[j] = d * (((x[j] >> shift) & 3) - 1);
            }
            y += kGroupSize;
        }
        x += kGroupSize;
    }
}

namespace {
template <int block_size, int group_size, int num_bits, bool is_abs = false, bool is_int = false>
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
    constexpr static float kScale = is_int ? 1.f : 31.75f;
    constexpr static bool kVerbose = false;

    QuantizerIQKT(int num_clusters, int num_neighbours, int offset = 4096);
    const float * values() const { return m_values.data(); }

    inline void find_best_match(float d, const float * xb, const float * weight, int * best_idx) const;
    inline std::pair<float, float> find_best_scale(const float * xb, const float * weight, const int * best_idx) const;
    inline float find_best_inverse_scale(const float * xb, const float * weight, const int * best_idx) const;

    static inline void set_values(uint32_t i, float * result, float scale, int offset = 4096) {
        uint32_t x = i + offset;
        if constexpr (is_int) {
            constexpr uint32_t ka = 0xCBAC1FED;
            uint32_t s;
            auto i8 = (const int8_t *)&s;
            for (int k = 0; k < kGroupSize; ++k) {
                x = ka*x;
                s = x & 0x3f3f3f3f;
                if constexpr (is_abs) {
                    result[k] = scale*std::abs(i8[0] + i8[1] + i8[2] + i8[3] - 126.f);
                } else {
                    result[k] = scale*(i8[0] + i8[1] + i8[2] + i8[3] - 126.f);
                }
            }
        } else {
            constexpr uint32_t ka = 89226354;
            constexpr uint32_t kb = 64248484;
            constexpr uint32_t kmask = 0x8fff8fff;
            constexpr uint32_t km32 = 0x3b603b60;
            for (int k = 0; k < kGroupSize; ++k) {
                x = ka*x + kb;
                uint32_t s = (x & kmask) ^ km32;
                float val = GGML_FP16_TO_FP32(s & 65535) + GGML_FP16_TO_FP32(s >> 16);
                if constexpr (is_abs) result[k] = scale*std::abs(val);
                else result[k] = scale*val;
            }
        }
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
        constexpr float kEps2   = 1e-14f;
        constexpr float kWeight = 1e-4f;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            const float * xbl = x + ibl*kSuperBlockSize;
            float * wbl = row_weights + ibl*kSuperBlockSize;

            float sumx2 = 0;
            for (int j = 0; j < kSuperBlockSize; ++j) sumx2 += xbl[j]*xbl[j];
            if (sumx2 < kEps2*kSuperBlockSize) {
                // all x in th super block are (almost) zero
                for (int j = 0; j < kSuperBlockSize; ++j) wbl[j] = kWeight;
                continue;
            }
            const float sigma2 = sigma2_scale*sumx2/kSuperBlockSize;

            if (imatrix) {
                for (int ib = 0; ib < kSuperBlockSize/kBlockSize; ++ib) {
                    const float * qw = imatrix + ibl*kSuperBlockSize + ib*kBlockSize;
                    const float * xb = xbl + ib*kBlockSize;
                    float * wb = wbl + ib*kBlockSize;
                    float sumwx = 0, sumw2 = 0, sumx2 = 0;
                    for (int j = 0; j < kBlockSize; ++j) {
                        wb[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                        sumwx += wb[j]*std::abs(xb[j]);
                        sumw2 += wb[j]*wb[j];
                        sumx2 += xb[j]*xb[j];
                    }
                    if (sumx2 < kEps2 || sumw2 < kEps2 || sumwx < kEps2) {
                        for (int j = 0; j < kBlockSize; ++j) wb[j] = kWeight;
                    }
                }
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

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::QuantizerIQKT(int num_clusters, int num_neighbours, int offset) {
    m_values.resize(kNumVal*kGroupSize);
    float * data = m_values.data();
    for (int i = 0; i < kNumVal; ++i) {
        set_values(i, data, kScale, offset);
        data += kGroupSize;
    }
    if (num_clusters == 0) return;
    // Make 128 clusters.
    // Note: we get a slightly better result by using 64 clusters
    //       at the expense of almost doubling the quantization time.
    m_clusters = cluster_points(m_values, num_clusters, 200, m_mid);
    GGML_ASSERT(!m_clusters.empty());
    m_in_cluster = finalize_clusters(num_neighbours, m_values, m_clusters, m_c_values);
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
std::pair<float, float> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_scale(
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

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
float QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_inverse_scale(
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

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
void QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float d, const float * xb, const float * weight, int * best_idx) const {
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
                    sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
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

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
std::vector<std::vector<int>> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::finalize_clusters(int num_neighbours,
        const std::vector<float>& values, const std::vector<float>& clusters, std::vector<std::vector<float>>& c_values) {
    int ncluster = clusters.size()/kGroupSize;
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

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
std::vector<float> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::cluster_points(const std::vector<float>& points, int ncluster, int niter, float * mid) {
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
            } else {
                for (int k = 0; k < ndim; ++k) result[ic*ndim + k] = sump[ic*ndim + k]/counts[ic];
            }
        }
        if (nzero > 0) printf("%s: %d out of %d clusters dir not have any points\n", __func__, nzero, ncluster);
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

// ========================================== iq1_kt ====================================================

using QuantizerIQ1KT = QuantizerIQKT<32, 8, 13, false, true>;

const QuantizerIQ1KT& iq1kt_quantizer() {
    static std::mutex mutex;
    static std::unique_ptr<QuantizerIQ1KT> quantizer;
    std::lock_guard<std::mutex> lock(mutex);
    if (!quantizer) quantizer = std::make_unique<QuantizerIQ1KT>(256, 32);
    return *quantizer;
}

void quantize_row_iq1_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights,
        int * all_idx) {

    constexpr float kSigmaScale = 2.0f;
    using Q = QuantizerIQ1KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq1_kt * y = (block_iq1_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq1kt_quantizer();

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }

    float amax_scale = 0, max_scale = 0;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq1_kt));

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
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp, score_p] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm, score_m] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            auto idx = best_idx;
            if (score_p > score_m) scales[ib] = dp;
            else {
                scales[ib] = dm; idx += Q::kNg; score_p = score_m;
            }
            for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];

            scale_0 -= 8;
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp1, score_p1] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm1, score_m1] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            if (score_p1 > score_p || score_m1 > score_p) {
                idx = best_idx;
                if (score_p1 > score_m1) scales[ib] = dp1;
                else {
                    scales[ib] = dm1; idx += Q::kNg;
                }
                for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];
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
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
                float dl = iq4k_values[ls];
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto qb = quantizer.values() + Q::kGroupSize*all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float q = dl*qb[j];
                        sumqx += wb[jj]*xb[jj]*q;
                        sumq2 += wb[jj]*q*q;
                    }
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
            y[ibl].sh[ib] = ls;
        }
    }

    *dptr = d;
    if (!d) return;

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                int ls = iq4k_values[y[ibl].sh[ib] & 0xf];
                float dl = d*ls;
                quantizer.find_best_match(dl, xb, weight, best_idx);

                auto prev_idx = all_idx + (ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize;

                float mse1 = 0, mse2 = 0;
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto q1 = quantizer.values() + Q::kGroupSize*prev_idx[ig];
                    auto q2 = quantizer.values() + Q::kGroupSize*best_idx[ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float diff1 = xb[jj] - dl*q1[j];
                        float diff2 = xb[jj] - dl*q2[j];
                        mse1 += weight[jj]*diff1*diff1;
                        mse2 += weight[jj]*diff2*diff2;
                    }
                }
                if (mse1 < mse2) {
                    for (int ig = 0; ig < Q::kNg; ++ig) best_idx[ig] = prev_idx[ig];
                } else {
                    for (int ig = 0; ig < Q::kNg; ++ig) prev_idx[ig] = best_idx[ig];
                }

                for (int j = 0; j < Q::kNg; ++j) {
                    y[ibl].ql[ib*Q::kNg+j] = best_idx[j] & 0xff;
                    y[ibl].qh[(ib%(Q::kNblock/2))*Q::kNg+j] |= (((best_idx[j] >> 8) & 0xf) << 4*(ib/(Q::kNblock/2)));
                    y[ibl].sh[ib] |= ((best_idx[j] >> 12) << (4+j));
                    auto xl = xb + Q::kGroupSize*j;
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
            *dptr = d * 1.07f;
            if (!d) return;
        } else {
            break;
        }

    }

}
}

void quantize_row_iq1_kt_ref(const float * GGML_RESTRICT x, block_iq1_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq1_kt(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq1_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_iq1_kt * y = (block_iq1_kt *)vy;
    quantize_row_iq1_kt_ref(x, y, k);
}

size_t quantize_iq1_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ1_KT, n_per_row);
    std::vector<float> scales(n_per_row/QuantizerIQ1KT::kBlockSize);
    std::vector<float> weights(n_per_row);
    std::vector<int> idx(n_per_row/QuantizerIQ1KT::kGroupSize);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq1_kt_impl(src, (void *)qrow, n_per_row, imatrix, scales.data(), weights.data(), idx.data());
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}

void dequantize_row_iq1_kt(const block_iq1_kt * x, float * y, int64_t k) {
    assert(k % QuantizerIQ1KT::kSuperBlockSize == 0);
    using Q = QuantizerIQ1KT;
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = *dptr * Q::kScale;
    x = (const block_iq1_kt *)(dptr + 1);
    auto& deq = iq1kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            float sl = d * iq4k_values[x[ibl].sh[ib] & 0xf];
            for (int ig = 0; ig < Q::kNg; ++ig) {
                uint16_t idx = x[ibl].ql[ib*Q::kNg + ig] | ((x[ibl].qh[(ib%(Q::kNblock/2))*Q::kNg + ig] << (8 - 4*(ib/(Q::kNblock/2)))) & 0xf00);
                idx |= (x[ibl].sh[ib] << (8 - ig) & 0x1000);
                deq.set_values(idx, y, sl);
                y += Q::kGroupSize;
            }
        }
    }
}

void vec_dot_iq1_kt_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ1_KT, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif

}

// ========================================== iq2_kt ====================================================

namespace {

using QuantizerIQ2KT = QuantizerIQKT<32, 8, 16, false, true>;

const QuantizerIQ2KT& iq2kt_quantizer() {
    static std::mutex mutex;
    static std::unique_ptr<QuantizerIQ2KT> quantizer;
    std::lock_guard<std::mutex> lock(mutex);
    if (!quantizer) quantizer = std::make_unique<QuantizerIQ2KT>(256, 8);
    return *quantizer;
}

void quantize_row_iq2_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights,
        int * all_idx) {

    constexpr float kSigmaScale = 2.0f;
    using Q = QuantizerIQ2KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq2_kt * y = (block_iq2_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq2kt_quantizer();

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }

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
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp, score_p] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm, score_m] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            auto idx = best_idx;
            if (score_p > score_m) scales[ib] = dp;
            else {
                scales[ib] = dm; idx += Q::kNg;
            }
            for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];

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
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
                float dl = iq4k_values[ls];
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto qb = quantizer.values() + Q::kGroupSize*all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float q = dl*qb[j];
                        sumqx += wb[jj]*xb[jj]*q;
                        sumq2 += wb[jj]*q*q;
                    }
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
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

                auto prev_idx = all_idx + (ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize;

                float mse1 = 0, mse2 = 0;
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto q1 = quantizer.values() + Q::kGroupSize*prev_idx[ig];
                    auto q2 = quantizer.values() + Q::kGroupSize*best_idx[ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float diff1 = xb[jj] - dl*q1[j];
                        float diff2 = xb[jj] - dl*q2[j];
                        mse1 += weight[jj]*diff1*diff1;
                        mse2 += weight[jj]*diff2*diff2;
                    }
                }
                if (mse1 < mse2) {
                    for (int ig = 0; ig < Q::kNg; ++ig) best_idx[ig] = prev_idx[ig];
                } else {
                    for (int ig = 0; ig < Q::kNg; ++ig) prev_idx[ig] = best_idx[ig];
                }

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
    std::vector<int> idx(n_per_row/QuantizerIQ2KT::kGroupSize);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq2_kt_impl(src, (void *)qrow, n_per_row, imatrix, scales.data(), weights.data(), idx.data());
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}

void dequantize_row_iq2_kt(const block_iq2_kt * x, float * y, int64_t k) {
    assert(k % QuantizerIQ2KT::kSuperBlockSize == 0);
#ifdef __AVX2__
    //if (iqk_dequantize_ktquants(GGML_TYPE_IQ2_KT, k, x, 0, y, 0, 1)) return;
#endif
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

using QuantizerIQ3KT = QuantizerIQKT<32, 8, 16, true, true>;
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
            bool found_solution = false;
            for (int itry = -3; itry <= 3; ++itry) {
                quantizer.find_best_match(amax/(scale_0 + kStep*itry), xaux, weight, best_idx);
                auto [d, score] = quantizer.find_best_scale(xaux, weight, best_idx);
                if (score > best) {
                    best = score;
                    found_solution = true;
                    scales[ib] = d;
                    std::memcpy(best_idx+Q::kNg, best_idx, Q::kNg*sizeof(int));
                }
            }
            if (!found_solution) {
                fprintf(stderr, "======================= %s: failed to find solution for a block\n", __func__);
                fprintf(stderr, "Model weights and importances:\n");
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    fprintf(stderr, "%2d  %g  %g\n", j, xaux[j], weight[j]);
                }
                GGML_ASSERT(false);
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
#ifdef __AVX2__
    //if (iqk_dequantize_ktquants(GGML_TYPE_IQ3_KT, k, x, 0, y, 0, 1)) return;
#endif
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

using QuantizerIQ4KT = QuantizerIQKT<32, 4, 15, false, true>;

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

const QuantizerIQ4KT& iq4kt_dequantizer() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unique_ptr<QuantizerIQ4KT> dequantizer;
    if (!dequantizer) dequantizer = std::make_unique<QuantizerIQ4KT>(0, 0, 4096);
    return *dequantizer;
}

void quantize_row_iq4_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights) {

    constexpr float kSigmaScale = 2.0f;
    constexpr int kNtry = 2;
    using Q = QuantizerIQ4KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq4_kt * y = (block_iq4_kt *)(dptr + 1);

    auto& quantizer1 = iq4kt_quantizer();
    auto& quantizer2 = iq4kt_quantizer(true);

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }
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
                xaux[j] = xbl[ib*Q::kBlockSize+j];
                float ax = std::abs(xaux[j]);
                amax = std::max(amax, ax);
            }
            if (!amax) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
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
                for (int j = 0; j < Q::kBlockSize; ++j) xaux[j] = xbl[ib*Q::kBlockSize+j];
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
#ifdef __AVX2__
    //if (iqk_dequantize_ktquants(GGML_TYPE_IQ4_KT, k, x, 0, y, 0, 1)) return;
#endif
    using Q = QuantizerIQ4KT;
    assert(k % Q::kSuperBlockSize == 0);
    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = dptr[0] * Q::kScale;
    x = (const block_iq4_kt *)(dptr + 1);
    auto& deq = iq4kt_dequantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto shb = x[ibl].qs;
        auto ql = (const uint8_t *)(shb + Q::kNblock);
        auto qh = ql + kNumGroups;
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            int offset = shb[ib] & 1 ? 32768 + 4096 : 4096;
            int ls = int((shb[ib] & 0xff) >> 1) - 64;
            float sl = d * ls;
            for (int ig = 0; ig < Q::kNg; ++ig) {
                int jj = ib*Q::kNg+ig;
                uint16_t idx = ql[jj] | ((qh[jj%(kNumGroups/2)] << (8 - 4*(jj/(kNumGroups/2)))) & 0xf00) | (((shb[ib] >> (8 + 3*ig)) & 7) << 12);
                deq.set_values(idx, y, sl, offset);
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

namespace {
template <typename Block>
inline int check_row_for_blocks_256_fp16(int nblock, const Block * x) {
    int nbad = 0;
    for (int ib = 0; ib < nblock; ++ib) {
        float d = GGML_FP16_TO_FP32(x[ib].d);
        if (isnan(d)) ++nbad;
    }
    return nbad;
}
template <typename Block>
bool check_tensor_for_blocks_256_fp16(const ggml_tensor * tensor) {
    int nblock = tensor->ne[0]/QK_K;
    int nbad = 0;
    for (int row = 0; row < ggml_nrows(tensor); ++row) {
        auto x = (const Block *)((const char *)tensor->data + tensor->nb[1]*row);
        nbad += check_row_for_blocks_256_fp16(nblock, x);
    }
    if (nbad > 0) {
        fprintf(stderr, "%s: found %d NaN block scales out of %ld blocks in tensor %s\n", __func__,
                nbad, ggml_nrows(tensor)*nblock, tensor->name);
        if (tensor->ne[2] > 1) {
            int nb = tensor->ne[0]/QK_K;
            for (int64_t i02 = 0; i02 < tensor->ne[2]; ++i02) {
                int nbad_expert = 0;
                auto xex = (const char *)((const char *)tensor->data + i02*tensor->nb[2]);
                for (int64_t i01 = 0; i01 < tensor->ne[1]; ++i01) {
                    auto xr = (const Block *)(xex + i01*tensor->nb[1]);
                    nbad_expert += check_row_for_blocks_256_fp16(nb, xr);
                }
                if (nbad_expert > 0) fprintf(stderr,"    there are %d NaN block scales for expert %ld\n", nbad_expert, i02);
            }
        }
        return false;
    }
    return true;
}
template <typename Block>
inline int check_row_for_blocks_256_fp16(int nblock, const Block * x, int nr) {
    int nbad = 0;
    for (int ib = 0; ib < nblock; ++ib) {
        for (int j = 0; j < nr; ++j) {
            if (!isfinite(GGML_FP16_TO_FP32(x[ib].d[j]))) ++nbad;
        }
    }
    return nbad;
}
template <typename Block, int nr>
bool check_tensor_for_blocks_256_fp16_repacked(const ggml_tensor * tensor) {
    int nblock = tensor->ne[0]/QK_K;
    int nbad = 0;
    for (int row = 0; row < ggml_nrows(tensor); row += nr) {
        auto x = (const Block *)((const char *)tensor->data + tensor->nb[1]*row);
        nbad += check_row_for_blocks_256_fp16(nblock, x, nr);
    }
    if (nbad > 0) {
        fprintf(stderr, "%s: found %d NaN block scales out of %ld blocks in tensor %s\n", __func__,
                nbad, ggml_nrows(tensor)*nblock, tensor->name);
        if (tensor->ne[2] > 1) {
            int nb = tensor->ne[0]/QK_K;
            for (int64_t i02 = 0; i02 < tensor->ne[2]; ++i02) {
                int nbad_expert = 0;
                auto xex = (const char *)((const char *)tensor->data + i02*tensor->nb[2]);
                for (int64_t i01 = 0; i01 < tensor->ne[1]; i01 += nr) {
                    auto xr = (const Block *)(xex + i01*tensor->nb[1]);
                    nbad_expert += check_row_for_blocks_256_fp16(nb, xr, nr);
                }
                if (nbad_expert > 0) fprintf(stderr,"    there are %d NaN block scales for expert %ld\n", nbad_expert, i02);
            }
        }
        return false;
    }
    return true;
}
struct F32Scale {
    static inline int check_row(const char * data) {
        float d = *(const float *)data;
        return isfinite(d) ? 0 : 1;
    }
};
struct F16Scale {
    static inline int check_row(const char * data) {
        float d = GGML_FP16_TO_FP32(*(const ggml_half *)data);
        return isfinite(d) ? 0 : 1;
    }
};
template <int nr>
struct F32ScaleRX {
    static inline int check_row(const char * data) {
        auto d = (const float *)data;
        int nbad = 0;
        for (int i = 0; i < nr; ++i) {
            if (!isfinite(d[i])) ++nbad;
        }
        return nbad;
    }
};
template <int nr>
struct F16ScaleRX {
    static inline int check_row(const char * data) {
        auto d = (const ggml_half *)data;
        int nbad = 0;
        for (int i = 0; i < nr; ++i) {
            if (!isfinite(GGML_FP16_TO_FP32(d[i]))) ++nbad;
        }
        return nbad;
    }
};
template <typename RS>
bool check_tensor_row_scales(const ggml_tensor * tensor) {
    auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
    int num_rows = ggml_nrows(tensor);
    auto data = (const char *)tensor->data;
    int nbad = 0;
    for (int row = 0; row < num_rows; ++row) {
        nbad += RS::check_row(data);
        data += row_size;
    }
    if (nbad > 0) {
        fprintf(stderr, "%s: found %d NaN row scales out of %d rows in tensor %s\n", __func__,
                nbad, num_rows, tensor->name);
        return false;
    }
    return true;
}
}

bool iqk_validate_tensor(const ggml_tensor * tensor) {
    if (!tensor) return true;
    if (!ggml_is_contiguous(tensor)) return true;

    switch (tensor->type) {
        case GGML_TYPE_IQ2_K:      return check_tensor_for_blocks_256_fp16<block_iq2_k>(tensor);
        case GGML_TYPE_IQ3_K:      return check_tensor_for_blocks_256_fp16<block_iq3_k>(tensor);
        case GGML_TYPE_IQ4_K:      return check_tensor_for_blocks_256_fp16<block_iq4_k>(tensor);
        case GGML_TYPE_IQ5_K:      return check_tensor_for_blocks_256_fp16<block_iq5_k>(tensor);
        case GGML_TYPE_IQ6_K:      return check_tensor_for_blocks_256_fp16<block_iq6_k>(tensor);
        case GGML_TYPE_IQ2_XXS:    return check_tensor_for_blocks_256_fp16<block_iq2_xxs>(tensor);
        case GGML_TYPE_IQ2_XS:     return check_tensor_for_blocks_256_fp16<block_iq2_xs>(tensor);
        case GGML_TYPE_IQ2_S:      return check_tensor_for_blocks_256_fp16<block_iq2_s>(tensor);
        case GGML_TYPE_IQ3_XXS:    return check_tensor_for_blocks_256_fp16<block_iq3_xxs>(tensor);
        case GGML_TYPE_IQ3_S:      return check_tensor_for_blocks_256_fp16<block_iq3_s>(tensor);
        case GGML_TYPE_IQ4_XS:     return check_tensor_for_blocks_256_fp16<block_iq4_xs>(tensor);
        case GGML_TYPE_IQ2_K_R4:   return check_tensor_for_blocks_256_fp16_repacked<block_iq2_k_r4, 4>(tensor);
        case GGML_TYPE_IQ3_K_R4:   return check_tensor_for_blocks_256_fp16_repacked<block_iq3_k_r4, 4>(tensor);
        case GGML_TYPE_IQ4_K_R4:   return check_tensor_for_blocks_256_fp16_repacked<block_iq4_k_r4, 4>(tensor);
        case GGML_TYPE_IQ5_K_R4:   return check_tensor_for_blocks_256_fp16_repacked<block_iq5_k_r4, 4>(tensor);
        case GGML_TYPE_IQ2_XXS_R4: return check_tensor_for_blocks_256_fp16_repacked<block_iq2_xxs_r4, 4>(tensor);
        case GGML_TYPE_IQ2_XS_R4:  return check_tensor_for_blocks_256_fp16_repacked<block_iq2_xs_r4, 4>(tensor);
        case GGML_TYPE_IQ2_S_R4:   return check_tensor_for_blocks_256_fp16_repacked<block_iq2_s_r4, 4>(tensor);
        case GGML_TYPE_IQ3_XXS_R4: return check_tensor_for_blocks_256_fp16_repacked<block_iq3_xxs_r4, 4>(tensor);
        case GGML_TYPE_IQ3_S_R4:   return check_tensor_for_blocks_256_fp16_repacked<block_iq3_s_r4, 4>(tensor);
        case GGML_TYPE_IQ4_XS_R8:  return check_tensor_for_blocks_256_fp16_repacked<block_iq4_xs_r8, 8>(tensor);
        case GGML_TYPE_IQ2_BN:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ5_KS:     return check_tensor_row_scales<F32Scale>(tensor);
        case GGML_TYPE_IQ2_BN_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_KS_R4:  return check_tensor_row_scales<F32ScaleRX<4>>(tensor);
        case GGML_TYPE_IQ1_BN:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:     return check_tensor_row_scales<F16Scale>(tensor);
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:   return check_tensor_row_scales<F16ScaleRX<4>>(tensor);

        default: break;
    }
    return true;
}
