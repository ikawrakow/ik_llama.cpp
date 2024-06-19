//
// Copyright 2024 Iwan Kawrakow
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iqk-quantize.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cmath>
#include <array>
#include <algorithm>
#include <cstring>
#include <mutex>

namespace {

inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

struct IQ1BNData {
    IQ1BNData();
    std::vector<std::pair<int16_t, bool>> map;
    std::vector<uint16_t> rmap;
};

const IQ1BNData& get_iq1bn_data() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static IQ1BNData iq1bn;
    return iq1bn;
}

IQ1BNData::IQ1BNData() {
    map.resize(1 << 16, {int16_t(-1), false});
    uint64_t aux64;
    uint8_t * aux8 = (uint8_t *)&aux64;
    std::vector<uint64_t> values;
    values.reserve(6561);
    rmap.reserve(6561);
    for (int i = 0; i < (1 << 16); ++i) {
        bool is_good = true;
        for (int j = 0; j < 8; ++j) {
            aux8[j] = (i >> 2*j) & 3;
            if (aux8[j] == 3u) { is_good = false; break; }
        }
        if (!is_good) continue;
        auto orig = aux64;
        for (int j = 0; j < 8; ++j) aux8[j] = 2 - aux8[j];
        int k = 0;
        for (; k < int(values.size()); ++k) {
            if (values[k] == aux64) break;
        }
        if (k < int(values.size())) {
            map[i] = {k, true};
        } else {
            map[i].first = values.size();
            values.push_back(orig);
            rmap.push_back(i);
        }
    }
    printf("==================== %s: initialized %d grid points\n", __func__, int(rmap.size()));
}

struct IQ1BNQuantizer {
    constexpr static int block_size = QK_IQ1BN;
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
    static uint16_t quantize_one_block_1bn(const IQ1BNData& iq1l, const float * xb, int8_t * L, uint8_t * ql, uint8_t * qh);
};

uint16_t IQ1BNQuantizer::quantize_one_block_1bn(const IQ1BNData& iq1bn, const float * xb, int8_t * L, uint8_t * ql, uint8_t * qh) {
    for (int j = 0; j < QK_IQ1BN; ++j) {
        L[j] = fabsf(xb[j]) < 1e-6f ? 1 : xb[j] < 0 ? 0 : 2;
    }
    uint16_t extra = 0;
    for (int k = 0; k < QK_IQ1BN/8; ++k) {
        auto Lk = L + 8*k;
        uint16_t u = 0;
        for (int j = 0; j < 8; ++j) u |= (Lk[j] << 2*j);
        auto& val = iq1bn.map[u];
        GGML_ASSERT(val.first >= 0);
        ql[k] = val.first & 255;
        qh[k/2] |= (val.first >> 8) << 4*(k%2);
        if (val.second) extra |= (1 << k);
    }
    return extra;
}

void IQ1BNQuantizer::quantize_one_row_1bn(const float * src, block_iq1_bn * y, int n_per_row, const float * imatrix) {

    (void)imatrix;

    const int nblock = n_per_row/QK_IQ1BN;

    const auto& iq1bn = get_iq1bn_data();

    for (int ib = 0; ib < nblock; ++ib) {
        std::memset(&y[ib], 0, sizeof(block_iq1_bn));
        auto xb = src + QK_IQ1BN*ib;
        y[ib].extra = quantize_one_block_1bn(iq1bn, xb, L, y[ib].ql, y[ib].qh);
    }
}

void IQ1BNQuantizer::quantize_one_row_2bn(const float * src, block_iq2_bn * y, int n_per_row, const float * imatrix) {

    (void)imatrix;

    const int nblock = n_per_row/QK_IQ1BN;

    //auto max_in_row = row_max(n_per_row, src);
    //printf("%s: max = %g\n", __func__, max_in_row);

    constexpr int Nj = QK_IQ1BN/4;

    for (int ib = 0; ib < nblock; ++ib) {
        auto xb = src + QK_IQ1BN*ib;
        for (int j = 0; j < QK_IQ1BN; ++j) {
            L[j] = fabsf(xb[j]) < 1e-6f ? 1 : xb[j] < 0 ? 0 : 2;
        }
        for (int j = 0; j < Nj; ++j) {
            y[ib].qs[j] = L[j] | (L[j + Nj] << 2) | (L[j + 2*Nj] << 4) | (L[j + 3*Nj] << 6);
        }
    }
}

}

void iq1bn_init_impl(void) {
    get_iq1bn_data();
}

size_t quantize_iq1_bn(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    IQ1BNQuantizer iq1bn;
    int nblock = n_per_row/QK_IQ1BN;
    block_iq1_bn * y = (block_iq1_bn *)dst;
    for (int row = 0; row < nrows; ++row) {
        iq1bn.quantize_one_row_1bn(src + row*n_per_row, y, n_per_row, imatrix);
        y += nblock;
    }
    return sizeof(block_iq1_bn)*nblock*nrows;
}

void quantize_row_iq1_bn_reference(const float * x, block_iq1_bn * y, int64_t k) {
    quantize_iq1_bn(x, y, 1, k, nullptr);
}

void quantize_row_iq1_bn(const float * x, void * y, int64_t k) {
    quantize_iq1_bn(x, y, 1, k, nullptr);
}

void dequantize_row_iq1_bn(const block_iq1_bn * x, float * y, int64_t k) {
    assert(k%QK_IQ1BN == 0);
    int nblock = k / QK_IQ1BN;

    for (int i = 0; i < nblock; ++i) {
        float d = iq1bn_fp8_to_float(x[i].extra & 0xff);
        uint8_t extra = x[i].extra >> 8;
        auto qh = x[i].qh;
        auto ql = x[i].ql;
        for (int k = 0; k < QK_IQ1BN/8; ++k) {
            uint16_t idx = ql[k] | ((qh[k/2] << (8 - 4*(k%2))) & 0x0f00);
            uint16_t val = iq1bn_grid_u16[idx];
            float dls = extra & (1 << k) ? -d : d;
            for (int j = 0; j < 8; ++j) y[j] = dls * (((val >> 2*j) & 3) - 1);
            y += 8;
        }
    }
}

size_t quantize_iq2_bn(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    IQ1BNQuantizer iq1bn;
    int nblock = n_per_row/QK_IQ1BN;
    block_iq2_bn * y = (block_iq2_bn *)dst;
    for (int row = 0; row < nrows; ++row) {
        iq1bn.quantize_one_row_2bn(src + row*n_per_row, y, n_per_row, imatrix);
        y += nblock;
    }
    return sizeof(block_iq2_bn)*nblock*nrows;
}

void quantize_row_iq2_bn_reference(const float * x, block_iq2_bn * y, int64_t k) {
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

void ggml_vec_dot_iq1_bn_q8_0 (int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    static_assert(QK_IQ1BN == 64, "This dot product implementation for iq1_bn requires a block size of 64");

    const block_iq1_bn * x = (const block_iq1_bn *)vx;
    const block_q8_0   * y = (const block_q8_0 *)vy;
    int nblock = n / QK_IQ1BN;

    float sumf = 0;

    for (int i = 0; i < nblock; ++i) {
        auto qh = x[i].qh;
        auto ql = x[i].ql;
        auto q8 = y[2*i+0].qs;
        int16_t sumi1 = 0;
        for (int k = 0; k < 4; ++k) {
            uint16_t idx = ql[k] | ((qh[k/2] << (8 - 4*(k%2))) & 0x0f00);
            uint16_t val = iq1bn_grid_u16[idx];
            int16_t sl = 0;
            for (int j = 0; j < 8; ++j) sl += q8[j] * (((val >> 2*j) & 3) - 1);
            sumi1 += x[i].extra & (1 << k) ? -sl : sl;
            q8 += 8;
        }
        q8 = y[2*i+1].qs;
        int16_t sumi2 = 0;
        for (int k = 4; k < 8; ++k) {
            uint16_t idx = ql[k] | ((qh[k/2] << (8 - 4*(k%2))) & 0x0f00);
            uint16_t val = iq1bn_grid_u16[idx];
            int16_t sl = 0;
            for (int j = 0; j < 8; ++j) sl += q8[j] * (((val >> 2*j) & 3) - 1);
            sumi2 += x[i].extra & (1 << k) ? -sl : sl;
            q8 += 8;
        }
        sumf += GGML_FP16_TO_FP32(y[2*i+0].d) * sumi1 + GGML_FP16_TO_FP32(y[2*i+1].d) * sumi2;
    }

    *s = sumf;

}

void ggml_vec_dot_iq1_bn_q8_K64(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    static_assert(QK_IQ1BN == 64, "This dot product implementation for iq1_bn requires a block size of 64");

    const block_iq1_bn * x = (const block_iq1_bn *)vx;
    const block_q8_K64 * y = (const block_q8_K64 *)vy;
    int nblock = n / QK_IQ1BN;

    float sumf = 0;

    for (int i = 0; i < nblock; ++i) {
        auto qh = x[i].qh;
        auto ql = x[i].ql;
        auto q8 = y[i].qs;
        int sumi = 0;
        for (int k = 0; k < 4; ++k) {
            uint16_t idx = ql[k] | ((qh[k/2] << (8 - 4*(k%2))) & 0x0f00);
            uint16_t val = iq1bn_grid_u16[idx];
            int16_t sl = 0;
            for (int j = 0; j < 8; ++j) sl += q8[j] * (((val >> 2*j) & 3) - 1);
            sumi += x[i].extra & (1 << k) ? -sl : sl;
            q8 += 8;
        }
        for (int k = 4; k < 8; ++k) {
            uint16_t idx = ql[k] | ((qh[k/2] << (8 - 4*(k%2))) & 0x0f00);
            uint16_t val = iq1bn_grid_u16[idx];
            int16_t sl = 0;
            for (int j = 0; j < 8; ++j) sl += q8[j] * (((val >> 2*j) & 3) - 1);
            sumi += x[i].extra & (1 << k) ? -sl : sl;
            q8 += 8;
        }
        sumf += y[i].d * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_iq2_bn_q8_K64(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    static_assert(QK_IQ1BN == 64, "This dot product implementation for iq2_bn requires a block size of 64");

    constexpr int Nj = QK_IQ1BN/4;

    const block_iq2_bn * x = (const block_iq2_bn *)vx;
    const block_q8_K64 * y = (const block_q8_K64 *)vy;
    int nblock = n / QK_IQ1BN;

    float sumf = 0;

    for (int i = 0; i < nblock; ++i) {
        auto q8 = y[i].qs;
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        for (int j = 0; j < Nj; ++j) {
            s1 += q8[j+   0] * (x[i].qs[j] & 0x03);
            s2 += q8[j+1*Nj] * (x[i].qs[j] & 0x0c);
            s3 += q8[j+2*Nj] * (x[i].qs[j] & 0x30);
            s4 += q8[j+3*Nj] * (x[i].qs[j] & 0xc0);
            s0 += q8[j] + q8[j+1*Nj] + q8[j+2*Nj] + q8[j+3*Nj];
        }
        sumf += y[i].d * (s1 + 0.25f*s2 + 0.0625*s3 + 0.015625*s4 - s0);
    }

    *s = sumf;

}

void quantize_row_q8_K64_reference(const float * x, block_q8_K64 * y, int64_t k) {
    assert(k % 64 == 0);
    const int64_t nb = k / 64;

    // Check if a row-wise scale works. It almost does, PPL is only ~0.02 higher
    //float amax = 0;
    //for (int j = 0; j < k; ++j) {
    //    float ax = fabsf(x[j]);
    //    amax = MAX(ax, amax);
    //}

    //float d = amax/127;
    //float id = d ? 1/d : 0.f;

    //for (int i = 0; i < nb; i++) {
    //    for (int j = 0; j < 64; ++j) y[i].qs[j] = nearest_int(id*x[j]);
    //    y[i].d = d;
    //    x += 64;
    //}

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < 64; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, 64);
            x += 64;
            continue;
        }
        const float iscale = -127.f/max;
        for (int j = 0; j < 64; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        y[i].d = 1/iscale;
        x += 64;
    }
}

void quantize_row_q8_K64(const float * x, void * y, int64_t k) {
    quantize_row_q8_K64_reference(x, (block_q8_K64 *)y, k);
}

