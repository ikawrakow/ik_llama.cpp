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
#include "iqk_mul_mat.h"
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

namespace {

inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
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
    static constexpr uint8_t k_mult[5] = {81, 27, 9, 3, 1};
};

void IQ1BNQuantizer::quantize_one_row_1bn(const float * src, block_iq1_bn * y, int n_per_row, const float * imatrix) {

    static const int k_nb[6] = {1, 3, 9, 27, 81, 243};
    (void)imatrix;

    const int nblock = n_per_row/QK_IQ1BN;

    for (int ib = 0; ib < nblock; ++ib) {
        std::memset(&y[ib], 0, sizeof(block_iq1_bn));
        auto xb = src + ib*QK_IQ1BN;
        int v13 = 0;
        for (int i16 = 0; i16 < QK_IQ1BN/16; ++i16) {
            for (int k = 0; k < 3; ++k) {
                int idx = 0;
                for (int j = 0; j < 5; ++j) {
                    float v = xb[16*i16 + 5*k + j];
                    int q = fabsf(v) < 1e-6f ? 1 : v < 0 ? 0 : 2;
                    idx += k_nb[j]*q;
                }
                idx = (256*idx + k_nb[5] - 1)/k_nb[5];
                y[ib].ql[3*i16 + k] = idx;
            }
            float v = xb[16*i16 + 15];
            int q = fabsf(v) < 1e-6f ? 1 : v < 0 ? 0 : 2;
            v13 += k_nb[i16]*q;
        }
        y[ib].extra = (256*v13 + k_nb[5] - 1)/k_nb[5];
    }
}

void IQ1BNQuantizer::quantize_one_row_2bn(const float * src, block_iq2_bn * y, int n_per_row, const float * imatrix) {

    (void)imatrix;

    const int nblock = n_per_row/QK_IQ1BN;

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

void ggml_vec_dot_iq1_bn_q8_K64(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    static_assert(QK_IQ1BN == 64, "This dot product implementation for iq1_bn requires a block size of 64");

    if (iqk_mul_mat(GGML_TASK_TYPE_COMPUTE, 1, 1, n, GGML_TYPE_IQ1_BN, vx, 0, GGML_TYPE_Q8_K64, vy, 0, s, 0, 0, 1)) {
        return;
    }

    const block_iq1_bn * x = (const block_iq1_bn *)vx;

    const float * d8 = (const float *)vy;
    const int8_t * q8 = (const int8_t *)(d8 + 4);
    int nblock = n / QK_IQ1BN;

    int sumi[8] = {};
    int8_t q1[16];

    for (int i = 0; i < nblock; ++i) {
        auto ql = x[i].ql;
        auto extra = x[i].extra;
        for (int i16 = 0; i16 < QK_IQ1BN/16; ++i16) {
            for (int k = 0; k < 3; ++k) {
                uint8_t q = *ql++;
                for (int j = 0; j < 5; ++j) {
                    uint8_t v = IQ1BNQuantizer::k_mult[j]*q;
                    int8_t vs = 3*v >> 8;
                    q1[5*k+j] = vs - 1;
                }
            }
            uint8_t v = IQ1BNQuantizer::k_mult[i16]*extra;
            int8_t vs = 3*v >> 8;
            q1[15] = vs - 1;
            for (int j = 0; j < 8; ++j) sumi[j] += q8[j]*q1[j];
            q8 += 8;
            for (int j = 0; j < 8; ++j) sumi[j] += q8[j]*q1[8+j];
            q8 += 8;
        }
    }

    *s = d8[0] * (sumi[0] + sumi[4]) + d8[1] * (sumi[1] + sumi[5]) + d8[2] * (sumi[2] + sumi[6]) + d8[3] * (sumi[3] + sumi[7]);
}

void ggml_vec_dot_iq2_bn_q8_K64(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {

    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    static_assert(QK_IQ1BN == 64, "This dot product implementation for iq2_bn requires a block size of 64");

    if (iqk_mul_mat(GGML_TASK_TYPE_COMPUTE, 1, 1, n, GGML_TYPE_IQ2_BN, vx, 0, GGML_TYPE_Q8_K64, vy, 0, s, 0, 0, 1)) {
        return;
    }

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

void quantize_row_q8_K64_reference(const float * x, block_q8_K64 * y, int64_t k) {

    float * dptr = (float *)y;
    auto qs = (int8_t *)(dptr + 4);
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
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            auto val = vld1q_f32(x + j + 4*i);
            val = vmulq_f32(vid[i], val);
            q.val[i] = vreinterpretq_s8_s32(vcvtnq_s32_f32(val));
        }
        auto qi = vqtbl4q_s8(q, shuffle);
        vst1q_s8(qs, qi);
        qs += 16;
    }
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
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            auto val = _mm_loadu_ps(x + j + 4*i);
            val = _mm_round_ps(_mm_mul_ps(vid[i], val), _MM_ROUND_NEAREST);
            q[i] = _mm_cvtps_epi32(val);
        }
        auto q1 = _mm_packs_epi32(q[0], q[1]);
        auto q2 = _mm_packs_epi32(q[2], q[3]);
        auto qi = _mm_packs_epi16(q1, q2);
        _mm_storeu_si128((__m128i *)qs, qi);
        qs += 16;
    }
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
    for (int j = 0; j < k; j += 16) {
        for (int i = 0; i < 4; ++i) {
            for (int l = 0; l < 4; ++l) qs[j+4*i+l] = nearest_int(aux[i]*x[j+4*i+l]);
        }
    }
#endif
}

void quantize_row_q8_K64(const float * x, void * y, int64_t k) {
    quantize_row_q8_K64_reference(x, (block_q8_K64 *)y, k);
}

