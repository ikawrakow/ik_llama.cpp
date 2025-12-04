//
// Copyright (C) 2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#define IQK_IMPLEMENT

#include "iqk_cpu_ops.h"
#include "iqk_utils.h"
#include "iqk_common.h"
#include "ggml.h"

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace {
// Playing around with group scores: use sum of probabilities in the group
inline float group_score(int n_per_group, const float * data) {
    float sum = 0;
    for (int j = 0; j < n_per_group; ++j) sum += data[j];
    return sum;
}
// Playing around with group scores: use max of probabilities in the group
inline float group_score_max(int n_per_group, const float * data) {
    float max = data[0];
    for (int j = 1; j < n_per_group; ++j) max = std::max(max, data[j]);
    return max;
}
// Actual top-nk group score: sum of top-nk probabilities in the group
inline float group_score(int n_per_group, int nk, const float * data, float * aux) {
    for (int j = 0; j < n_per_group; ++j) aux[j] = data[j];
    std::partial_sort(aux, aux + nk, aux + n_per_group, std::greater<float>{});
    float sum = 0;
    for (int j = 0; j < nk; ++j) sum += aux[j];
    return sum;
}
inline std::vector<std::pair<float,int>> & get_work_buffer(size_t size) {
    thread_local std::vector<std::pair<float,int>> buffer;
    if (buffer.size() < size) buffer.resize(size);
    return buffer;

}
#ifdef __ARM_NEON
inline float32x4_t v_sigmoid(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t neg_x = vsubq_f32(zero, x);
    const float32x4_t exp_neg_x = v_expf(neg_x);
    const float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
    return vdivq_f32(one, one_plus_exp_neg_x);
}
#endif
#ifdef __AVX2__
inline __m256 v_sigmoid(__m256 x) {
    const __m256 one = _mm256_set1_ps(1);
    const __m256 zero  = _mm256_setzero_ps();
    const __m256 neg_x = _mm256_sub_ps(zero, x);
    const __m256 exp_neg_x = v_expf(neg_x);
    const __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
    return _mm256_div_ps(one, one_plus_exp_neg_x);
}
#endif
#if defined __AVX512F__ && defined __AVX512DQ__
inline __m512 v_sigmoid(__m512 x) {
    const __m512 one = _mm512_set1_ps(1);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 neg_x = _mm512_sub_ps(zero, x);
    const __m512 exp_neg_x = v_expf(neg_x);
    const __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(one, one_plus_exp_neg_x);
}
#endif
inline void biased_sigmoid(int n, const float * x, const float * bias, float * y, float * z) {
    int i = 0;
#if defined __AVX512F__ && defined __AVX512DQ__
    for (; i + 15 < n; i += 16) {
        auto v = v_sigmoid(_mm512_loadu_ps(x + i));
        _mm512_storeu_ps(y + i, _mm512_add_ps(v, _mm512_loadu_ps(bias + i)));
        _mm512_storeu_ps(z + i, v);
    }
#endif
#if defined __AVX2__ && defined __FMA__
    for (; i + 7 < n; i += 8) {
        auto v = v_sigmoid(_mm256_loadu_ps(x + i));
        _mm256_storeu_ps(y + i, _mm256_add_ps(v, _mm256_loadu_ps(bias + i)));
        _mm256_storeu_ps(z + i, v);
    }
#endif
#ifdef __ARM_NEON
    for (; i + 3 < n; i += 4) {
        auto v = v_sigmoid(vld1q_f32(x + i));
        vst1q_f32(y + i, vaddq_f32(v, vld1q_f32(bias + i)));
        vst1q_f32(z + i, v);
    }
#endif
    for (; i < n; ++i) {
        z[i] = 1/(1 + expf(-x[i]));
        y[i] = y[i] + bias[i];
    }
}
inline void biased_sigmoid(int n, const float * x, const float * bias, float * y) {
    int i = 0;
#if defined __AVX512F__ && defined __AVX512DQ__
    for (; i + 15 < n; i += 16) {
        auto v = v_sigmoid(_mm512_loadu_ps(x + i));
        _mm512_storeu_ps(y + i, _mm512_add_ps(v, _mm512_loadu_ps(bias + i)));
    }
#endif
#if defined __AVX2__ && defined __FMA__
    for (; i + 7 < n; i += 8) {
        auto v = v_sigmoid(_mm256_loadu_ps(x + i));
        _mm256_storeu_ps(y + i, _mm256_add_ps(v, _mm256_loadu_ps(bias + i)));
    }
#endif
#ifdef __ARM_NEON
    for (; i + 3 < n; i += 4) {
        auto v = v_sigmoid(vld1q_f32(x + i));
        vst1q_f32(y + i, vaddq_f32(v, vld1q_f32(bias + i)));
    }
#endif
    for (; i < n; ++i) {
        y[i] = 1/(1 + expf(-x[i])) + bias[i];
    }
}
}

void iqk_sumrows_div(struct ggml_tensor * div, int ith, int nth) {
    auto src = div->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(div->type == GGML_TYPE_F32);

    int ne00  = src->ne[0];
    int nrows = ggml_nrows(src);
    int npt   = (nrows + nth - 1)/nth;
    int first = ith*npt;
    int last  = std::min(first + npt, nrows);
    if (last < first) return;

    for (int ir = first; ir < last; ++ir) {
        auto values = (const float *)((const char *)src->data + ir*src->nb[1]);
        float sum = 0;
        for (int j = 0; j < ne00; ++j) sum += values[j];
        float norm = sum > 0 ? 1/sum : 0.0f;
        auto result = (float *)((char *)div->data + ir*div->nb[1]);
        for (int j = 0; j < ne00; ++j) result[j] = values[j]*norm;
    }
}

void iqk_grouped_top_k(ggml_tensor * dst, int ith, int nth) {
    auto src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nrows(src) == ggml_nrows(dst));

    auto nrows = ggml_nrows(src);
    auto npt   = (nrows + nth - 1)/nth;
    auto first = npt*ith;
    auto last  = std::min(first + npt, nrows);
    if (last <= first) return;

    int n_groups     = dst->op_params[0];
    int n_top_groups = dst->op_params[1];
    int nk           = dst->op_params[2];

    int ne00 = src->ne[0];
    int ne0  = dst->ne[0];
    GGML_ASSERT(ne0 <= ne00);
    GGML_ASSERT(ne00%n_groups == 0);
    int n_per_group = ne00/n_groups;
    GGML_ASSERT(nk <= n_per_group);
    GGML_ASSERT(n_top_groups <= n_groups);

    size_t work_size = n_groups + n_per_group*n_top_groups;
    auto& aux = get_work_buffer(work_size);

    auto groups = aux.data() + n_per_group*n_top_groups;

    for (int ir = first; ir < last; ++ir) {
        auto data = (const float *)((const char *)src->data + ir*src->nb[1]);
        auto result = (int32_t *)((char *)dst->data + ir*dst->nb[1]);
        if (ne0 > n_per_group*n_top_groups) {
            for (int j = 0; j < ne0; ++j) result[j] = j;
            continue;
        }
        if (n_top_groups < n_groups) {
            for (int ig = 0; ig < n_groups; ++ig) {
                //groups[ig] = { group_score(n_per_group, data + ig*n_per_group), ig };
                //groups[ig] = { group_score_max(n_per_group, data + ig*n_per_group), ig };
                groups[ig] = { group_score(n_per_group, nk, data + ig*n_per_group, (float *)aux.data()), ig };
            }
            std::partial_sort(groups, groups + n_top_groups, groups + n_groups, std::greater<std::pair<float,int>>{});

            for (int ig = 0; ig < n_top_groups; ++ig) {
                int i0 = n_per_group * ig;
                int j0 = n_per_group * groups[ig].second;
                for (int j = 0; j < n_per_group; ++j) aux[i0 + j] = { data[j0 + j], j0 + j };
            }
        } else {
            for (int j = 0; j < ne00; ++j) aux[j] = { data[j], j };
        }
        if (ne0 < n_top_groups*n_per_group) {
            std::partial_sort(aux.begin(), aux.begin() + ne0, aux.begin() + n_top_groups*n_per_group, std::greater<std::pair<float,int>>{});
        } else {
            std::sort(aux.begin(), aux.begin() + ne0, std::greater<std::pair<float,int>>{});
        }
        for (int j = 0; j < ne0; ++j) result[j] = aux[j].second;

    }
}

void iqk_argsort(ggml_tensor * dst, int ith, int nth) {

    auto src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);

    auto nrows = ggml_nrows(src);
    auto npt   = (nrows + nth - 1)/nth;
    auto first = npt*ith;
    auto last  = std::min(first + npt, nrows);
    if (last <= first) return;

    auto order = (ggml_sort_order)dst->op_params[0];
    int nk = dst->op_params[1];

    int ne00 = src->ne[0];
    auto& aux = get_work_buffer(ne00);

    for (int ir = first; ir < last; ++ir) {
        auto data = (const float *)((const char *)src->data + ir*src->nb[1]);
        for (int j = 0; j < ne00; ++j) aux[j] = {data[j], j};
        if (nk < ne00) {
            if (order == GGML_SORT_ORDER_DESC) {
                std::partial_sort(aux.begin(), aux.begin() + nk, aux.begin() + ne00, std::greater<std::pair<float,int>>{});
            } else {
                std::partial_sort(aux.begin(), aux.begin() + nk, aux.begin() + ne00);
            }
        } else {
            if (order == GGML_SORT_ORDER_DESC) {
                std::sort(aux.begin(), aux.begin() + ne00, std::greater<std::pair<float,int>>{});
            } else {
                std::sort(aux.begin(), aux.begin() + ne00);
            }
        }
        auto y = (int32_t *)((char *)dst->data + ir*dst->nb[1]);
        for (int j = 0; j < ne00; ++j) y[j] = aux[j].second;
    }

}

void iqk_bailingmoev2_experts(struct ggml_tensor * dst, struct ggml_tensor * topk, int ith, int nth) {
    auto topk_src = topk->src[0];
    auto probs    = topk_src->src[0]->src[0];
    auto t_bias   = topk_src->src[1];

    auto nrows = ggml_nrows(probs);
    auto npt   = (nrows + nth - 1)/nth;
    auto first = npt*ith;
    auto last  = std::min(first + npt, nrows);
    if (last <= first) return;

    int n_groups     = topk->op_params[0];
    int n_top_groups = topk->op_params[1];
    int nk           = topk->op_params[2];

    int ne00 = probs->ne[0];
    int ne0  = topk->ne[0];
    GGML_ASSERT(ggml_is_contiguous(probs));
    GGML_ASSERT(t_bias->ne[1] == 1);
    GGML_ASSERT(t_bias->ne[0] == probs->ne[0]);
    GGML_ASSERT(ne0 == dst->ne[1]);
    GGML_ASSERT(ne0 <= ne00);
    GGML_ASSERT(ne00%n_groups == 0);
    int n_per_group = ne00/n_groups;
    GGML_ASSERT(nk <= n_per_group);
    GGML_ASSERT(n_top_groups <= n_groups);

    size_t work_size = n_groups + n_per_group*n_top_groups + ne00;
    auto& aux = get_work_buffer(work_size);

    auto groups = aux.data() + n_per_group*n_top_groups;
    auto biased_values = (float *)(groups + n_groups);
    auto values = biased_values + ne00;

    auto bias = (const float *)t_bias->data;

    for (int ir = first; ir < last; ++ir) {
        auto data = (const float *)((const char *)probs->data + ir*probs->nb[1]);
        biased_sigmoid(ne00, data, bias, biased_values, values);
        //for (int j = 0; j < ne00; ++j) values[j] = 1/(1 + expf(-data[j])) + bias[j];
        auto weights = (float *)((char *)dst->data + ir*dst->nb[2]);
        auto ids = (int32_t *)((char *)topk->data + ir*topk->nb[1]);
        if (ne0 > n_per_group*n_top_groups) {
            for (int j = 0; j < ne0; ++j) {
                weights[j] = values[j];
                ids[j]     = j;
            }
            continue;
        }
        if (n_top_groups < n_groups) {
            for (int ig = 0; ig < n_groups; ++ig) {
                groups[ig] = { group_score(n_per_group, nk, biased_values + ig*n_per_group, (float *)aux.data()), ig };
            }
            std::partial_sort(groups, groups + n_top_groups, groups + n_groups, std::greater<std::pair<float,int>>{});

            for (int ig = 0; ig < n_top_groups; ++ig) {
                int i0 = n_per_group * ig;
                int j0 = n_per_group * groups[ig].second;
                for (int j = 0; j < n_per_group; ++j) aux[i0 + j] = { biased_values[j0 + j], j0 + j };
            }
        } else {
            for (int j = 0; j < ne00; ++j) aux[j] = { biased_values[j], j };
        }
        std::partial_sort(aux.begin(), aux.begin() + ne0, aux.begin() + n_top_groups*n_per_group, std::greater<std::pair<float,int>>{});
        for (int j = 0; j < ne0; ++j) {
            weights[j] = values[aux[j].second];
            ids[j]     = aux[j].second;
        }

    }
}

void iqk_glm45moe_experts(struct ggml_tensor * dst, struct ggml_tensor * topk_view, int ith, int nth) {
    GGML_ASSERT(topk_view->op == GGML_OP_VIEW);
    auto topk     = topk_view->src[0];
    auto topk_src = topk->src[0];
    auto probs    = topk_src->src[0]->src[0];
    auto t_bias   = topk_src->src[1];

    auto nrows = ggml_nrows(probs);
    auto npt   = (nrows + nth - 1)/nth;
    auto first = npt*ith;
    auto last  = std::min(first + npt, nrows);
    if (last <= first) return;

    int ne00 = probs->ne[0];
    int ne0  = topk_view->ne[0];
    GGML_ASSERT(ggml_is_contiguous(probs));
    GGML_ASSERT(t_bias->ne[1] == 1);
    GGML_ASSERT(t_bias->ne[0] == probs->ne[0]);
    GGML_ASSERT(ne0 == dst->ne[1]);
    GGML_ASSERT(ne0 <= ne00);

    size_t work_size = 2*ne00;
    auto& aux = get_work_buffer(work_size);

    auto biased_values = (float *)(aux.data() + ne00);
    //auto values = biased_values + ne00;

    auto bias = (const float *)t_bias->data;

    for (int ir = first; ir < last; ++ir) {
        auto data = (const float *)((const char *)probs->data + ir*probs->nb[1]);
        //biased_sigmoid(ne00, data, bias, biased_values, values);
        biased_sigmoid(ne00, data, bias, biased_values);
        auto weights = (float *)((char *)dst->data + ir*dst->nb[2]);
        auto ids = (int32_t *)((char *)topk->data + ir*topk->nb[1]);
        for (int j = 0; j < ne00; ++j) aux[j] = { biased_values[j], j };
        if (ne0 < ne00) {
            std::partial_sort(aux.begin(), aux.begin() + ne0, aux.begin() + ne00, std::greater<std::pair<float,int>>{});
        } else {
            std::sort(aux.begin(), aux.begin() + ne00, std::greater<std::pair<float,int>>{});
        }
        for (int j = 0; j < ne0; ++j) {
            weights[j] = 1/(1 + expf(-data[aux[j].second]));
            ids[j]     = aux[j].second;
        }
    }
}

void iqk_openai_experts(struct ggml_tensor * topk, struct ggml_tensor * softmax, int ith, int nth) {

    auto probs    = topk->src[0];

    auto nrows = ggml_nrows(probs);
    auto npt   = (nrows + nth - 1)/nth;
    auto first = npt*ith;
    auto last  = std::min(first + npt, nrows);
    if (last <= first) return;

    int ne00 = probs->ne[0];
    int ne0  = softmax->ne[0];
    GGML_ASSERT(ggml_is_contiguous(probs));
    GGML_ASSERT(ggml_is_contiguous(softmax));
    GGML_ASSERT(ne0 <= ne00);

    size_t work_size = ne00;
    auto& aux = get_work_buffer(work_size);

    for (int ir = first; ir < last; ++ir) {
        auto data = (const float *)((const char *)probs->data + ir*probs->nb[1]);
        for (int j = 0; j < ne00; ++j) aux[j] = { data[j], j };
        if (ne0 < ne00) {
            std::partial_sort(aux.begin(), aux.begin() + ne0, aux.begin() + ne00, std::greater<std::pair<float,int>>{});
        } else {
            std::sort(aux.begin(), aux.begin() + ne00, std::greater<std::pair<float,int>>{});
        }
        auto weights = (float *)((char *)softmax->data + ir*softmax->nb[1]);
        auto ids = (int32_t *)((char *)topk->data + ir*topk->nb[1]);
        float max = aux.front().first;
        float sum = 0;
        for (int j = 0; j < ne0; ++j) {
            weights[j] = expf(aux[j].first - max);
            ids[j]     = aux[j].second;
            sum += weights[j];
        }
        GGML_ASSERT(sum > 0);
        float norm = 1/sum;
        for (int j = 0; j < ne0; ++j) weights[j] *= norm;
    }
}

void iqk_mul_multi_add(struct ggml_tensor * dst, int ith, int nth) {
    auto src0 = dst->src[0];
    auto src1 = dst->src[1];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] ==  dst->ne[0]);
    GGML_ASSERT(src0->ne[2] ==  dst->ne[1]);
    GGML_ASSERT(src0->ne[1] == src1->ne[1]);
    GGML_ASSERT(src0->ne[2] == src1->ne[2]);
    GGML_ASSERT(src0->ne[3] == src1->ne[3]);
    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[0] == 1);

    int nrows = dst->ne[1];
    int npt   = (nrows + nth - 1)/nth;
    int first = ith*npt;
    int last  = std::min(nrows, first + npt);

    int ne01 = src0->ne[1];
    int ne00 = src0->ne[0];

    for (int ir = first; ir < last; ++ir) {
        auto c0 = (const char *)src0->data + ir*src0->nb[2];
        auto c1 = (const char *)src1->data + ir*src1->nb[2];
        auto cy = (      char *) dst->data + ir* dst->nb[1];
        std::memset(cy, 0, ne00*sizeof(float));
        for (int j = 0; j < ne01; ++j) {
            auto x0 = (const float *)c0;
            auto x1 = (const float *)c1;
            auto  y = (      float *)cy;
            for (int k = 0; k < ne00; ++k) y[k] += x0[k] * x1[0];
            c0 += src0->nb[1];
            c1 += src1->nb[1];
        }
    }
}

namespace {
template <typename T>
void fast_ht(int n, T * values) {
    constexpr float ksqrt2 = 0.707106781f;
    float scale = 1;
    for (int h = 1; h < n; h <<= 1) {
        for (int i = 0; i < n; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                T x = values[j], y = values[j + h];
                values[j+0] = x + y;
                values[j+h] = x - y;
            }
        }
        scale *= ksqrt2;
    }
    for (int i = 0; i < n; ++i) values[i] *= scale;
}
}

void iqk_hadamard(struct ggml_tensor * dst, int ith, int nth) {
    auto src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    int nh = dst->op_params[0];
    GGML_ASSERT(nh > 1 && popcount(uint32_t(nh)) == 1);
    GGML_ASSERT(dst->ne[0] % nh == 0);

    int nc = dst->ne[0]/nh;
    int nr = ggml_nrows(dst) * nc;

    int npt = (nr + nth - 1)/nth;
    int first = npt*ith;
    int last  = std::min(first + npt, nr);

    for (int ir = first; ir < last; ++ir) {
        int i3 = ir / (dst->ne[1] * dst->ne[2] * nc);
        int i2 = (ir - i3*dst->ne[1] * dst->ne[2] * nc)/(dst->ne[1] * nc);
        int i1 = (ir - i3*dst->ne[1] * dst->ne[2] * nc - i2*dst->ne[1]*nc)/nc;
        int ic = (ir - i3*dst->ne[1] * dst->ne[2] * nc - i2*dst->ne[1]*nc - i1*nc);

        auto x = (const float *)((const char *)src->data + i3*src->nb[3] + i2*src->nb[2] + i1*src->nb[1]) + ic*nh;
        auto y = (      float *)((      char *)dst->data + i3*dst->nb[3] + i2*dst->nb[2] + i1*dst->nb[1]) + ic*nh;
        std::memcpy(y, x, nh*sizeof(float));
        fast_ht(nh, y);
    }
}
