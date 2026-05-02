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
//#include <thread>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

bool iqk_has_fancy_simd(void) {
#ifdef HAVE_FANCY_SIMD
    return true;
#else
    return false;
#endif
}

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

    auto src2 = dst->src[2];
    auto src3 = dst->src[3];
    if (src2 && src3) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src3->type == GGML_TYPE_I32);
        GGML_ASSERT(src3->ne[0] == src0->ne[1]);

        auto cids = (const char *)src3->data;
        auto scales = (const float *)src2->data;
        for (int ir = first; ir < last; ++ir) {
            auto c0 = (const char *)src0->data + ir*src0->nb[2];
            auto c1 = (const char *)src1->data + ir*src1->nb[2];
            auto cy = (      char *)dst->data + ir* dst->nb[1];
            auto  y = (     float *)cy;
            auto x0 = (const float *)c0;
            auto x1 = (const float *)c1;
            auto ids = (const int *)(cids + ir*src3->nb[1]);
            float s = scales[ids[0]] * x1[0];
            for (int k = 0; k < ne00; ++k) y[k] = x0[k] * s;
            for (int j = 1; j < ne01; ++j) {
                c0 += src0->nb[1];
                c1 += src1->nb[1];
                x0 = (const float *)c0;
                x1 = (const float *)c1;
                s  = x1[0] * scales[ids[j]];
                for (int k = 0; k < ne00; ++k) y[k] += x0[k] * s;
            }
        }

        return;

    }

    for (int ir = first; ir < last; ++ir) {
        auto c0 = (const char *)src0->data + ir*src0->nb[2];
        auto c1 = (const char *)src1->data + ir*src1->nb[2];
        auto cy = (      char *)dst->data + ir* dst->nb[1];
        auto  y = (     float *)cy;
        auto x0 = (const float *)c0;
        auto x1 = (const float *)c1;
        for (int k = 0; k < ne00; ++k) y[k] = x0[k] * x1[0];
        for (int j = 1; j < ne01; ++j) {
            c0 += src0->nb[1];
            c1 += src1->nb[1];
            x0 = (const float *)c0;
            x1 = (const float *)c1;
            for (int k = 0; k < ne00; ++k) y[k] += x0[k] * x1[0];
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

namespace {
float iqk_exp_with_thresh_impl(int n, float * logits, float max, float min) {
    float sum = 0;
#ifdef __AVX2__
    auto vmax = _mm256_set1_ps(max);
    auto vmin = _mm256_set1_ps(min);
    auto vsum = _mm256_setzero_ps();
    for (int j = 0; j < n/8; ++j) {
        auto x = _mm256_loadu_ps(logits);
        auto mask = _mm256_cmp_ps(x, vmin, _CMP_GE_OQ);
        auto exp_x = v_expf(_mm256_sub_ps(x, vmax));
        exp_x = _mm256_and_ps(exp_x, mask);
        vsum = _mm256_add_ps(vsum, exp_x);
        _mm256_storeu_ps(logits, exp_x);
        logits += 8;
    }
    sum = hsum_float_8(vsum);
    for (int j = 0; j < n - 8*(n/8); ++j) {
        float p = logits[j] > min ? expf(logits[j] - max) : 0;
        sum += p;
        logits[j] = p;
    }
#else
    for (int j = 0; j < n; ++j) {
        float p = logits[j] > min ? expf(logits[j] - max) : 0;
        sum += p;
        logits[j] = p;
    }
#endif
    return sum;
}
}

float iqk_exp_with_thresh(int n, float * logits, float max, float min) {
    return iqk_exp_with_thresh_impl(n, logits, max, min);
    //if (n < (1 << 16)) return iqk_exp_with_thresh_impl(n, logits, max, min);
    //std::array<float, 2> result;
    //auto compute = [logits, max, min, &result] (int first, int last, int ith) {
    //    result[ith] = iqk_exp_with_thresh_impl(last - first, logits + first, max, min);
    //};
    //auto t = std::thread(compute, 0, n/2, 0);
    //compute(n/2, n, 1);
    //t.join();
    //return result[0] + result[1];
}

bool iqk_ssm_conv4(int nr, int nc, int nt,
        uint64_t nb01, uint64_t nb10, uint64_t nb11, uint64_t nb21,
        const float * x0_in, const float * s0_in, const float * c_in,
        float * dst, float * dst_silu, int ith, int nth) {
#if defined __AVX2__
    if (nt <= 32 || nc != 4 || nr%16 != 0) {
        return false;
    }
    int nr16 = nr/16;
    int dr16 = (nr16 + nth - 1)/nth;
    int ir0  = ith*dr16;
    int ir1  = std::min(nr16, ir0 + dr16);
    __m256 vs[8], vc[8];
    float aux[64];
    for (int ir = ir0; ir < ir1; ++ir) {
        auto x  = dst_silu == nullptr ? dst + 16*ir : dst_silu + 16*ir;
        auto s  = dst   + 16*ir*nb21/sizeof(float) + nr*nt;
        auto s0 = s0_in + 16*ir*nb01/sizeof(float); // {d_conv - 1, d_inner, n_kv}
        auto x0 = x0_in + 16*ir*nb10/sizeof(float);
        auto c  = c_in  + 16*ir*nb21/sizeof(float);
        for (int ic = 0; ic < 3; ++ic) {
            for (int j = 0; j < 8; ++j) {
                aux[j + 8*ic +  8] = s0[(j+0)*nb01/sizeof(float) + ic];
                aux[j + 8*ic + 40] = s0[(j+8)*nb01/sizeof(float) + ic];
            }
        }
        // Not necessary, but doing it to shut up compiler warnings
        for (int j = 0; j < 8; ++j) {
            aux[j] = aux[j+32] = 0.0f;
        }
        for (int k = 0; k < 8; ++k) vs[k] = _mm256_loadu_ps(aux + 8*k);
        for (int ic = 0; ic < 4; ++ic) {
            for (int j = 0; j < 8; ++j) {
                aux[j + 8*ic     ] = c[(j+0)*nb21/sizeof(float) + ic];
                aux[j + 8*ic + 32] = c[(j+8)*nb21/sizeof(float) + ic];
            }
        }
        for (int k = 0; k < 8; ++k) vc[k] = _mm256_loadu_ps(aux + 8*k);
        int idx = 0;
        for (int it = 0; it < nt; ++it) {
            vs[idx+0] = _mm256_loadu_ps(x0+0);
            vs[idx+4] = _mm256_loadu_ps(x0+8);
            idx = (idx + 1) & 3;
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            for (int k = 0; k < 4; ++k) {
                int ii = (idx + k) & 3;
                sum1 = _mm256_fmadd_ps(vs[ii+0], vc[k+0], sum1);
                sum2 = _mm256_fmadd_ps(vs[ii+4], vc[k+4], sum2);
            }
            if (dst_silu) {
                sum1 = v_silu(sum1);
                sum2 = v_silu(sum2);
            }
            _mm256_storeu_ps(x+0, sum1);
            _mm256_storeu_ps(x+8, sum2);
            x0 += nb11/sizeof(float);
            x  += nr;
        }
        for (int k = 0; k < 4; ++k) {
            int ii = (idx + k) & 3;
            _mm256_storeu_ps(aux + 8*k +  0, vs[ii+0]);
            _mm256_storeu_ps(aux + 8*k + 32, vs[ii+4]);
        }
        for (int j = 0; j < 8; ++j) {
            for (int ic = 0; ic < 4; ++ic) {
                s[(j+0)*nb21/sizeof(float) + ic] = aux[j + 8*ic +  0];
                s[(j+8)*nb21/sizeof(float) + ic] = aux[j + 8*ic + 32];
            }
        }
    }
    return true;
#elif defined __ARM_NEON
    if (nt <= 32 || nc != 4 || nr%16 != 0) {
        return false;
    }
    int nr16 = nr/16;
    int dr16 = (nr16 + nth - 1)/nth;
    int ir0  = ith*dr16;
    int ir1  = std::min(nr16, ir0 + dr16);
    float32x4x2_t vs[8], vc[8];
    float aux[64];
    for (int ir = ir0; ir < ir1; ++ir) {
        auto x  = dst_silu == nullptr ? dst + 16*ir : dst_silu + 16*ir;
        auto s  = dst   + 16*ir*nb21/sizeof(float) + nr*nt;
        auto s0 = s0_in + 16*ir*nb01/sizeof(float); // {d_conv - 1, d_inner, n_kv}
        auto x0 = x0_in + 16*ir*nb10/sizeof(float);
        auto c  = c_in  + 16*ir*nb21/sizeof(float);
        for (int ic = 0; ic < 3; ++ic) {
            for (int j = 0; j < 8; ++j) {
                aux[j + 8*ic +  8] = s0[(j+0)*nb01/sizeof(float) + ic];
                aux[j + 8*ic + 40] = s0[(j+8)*nb01/sizeof(float) + ic];
            }
        }
        // Not necessary, but doing it to shut up compiler warnings
        for (int j = 0; j < 8; ++j) {
            aux[j] = aux[j+32] = 0.0f;
        }
        for (int k = 0; k < 8; ++k) vs[k] = vld1q_f32_x2(aux + 8*k);
        for (int ic = 0; ic < 4; ++ic) {
            for (int j = 0; j < 8; ++j) {
                aux[j + 8*ic     ] = c[(j+0)*nb21/sizeof(float) + ic];
                aux[j + 8*ic + 32] = c[(j+8)*nb21/sizeof(float) + ic];
            }
        }
        for (int k = 0; k < 8; ++k) vc[k] = vld1q_f32_x2(aux + 8*k);
        for (int it4 = 0; it4 < nt/4; ++it4) {
            float32x4x2_t sum1, sum2;
            vs[0] = vld1q_f32_x2(x0+0);
            vs[4] = vld1q_f32_x2(x0+8);
            for (int j = 0; j < 2; ++j) {
                sum1.val[j] = vmulq_f32(             vs[1].val[j], vc[0].val[j]);
                sum1.val[j] = vfmaq_f32(sum1.val[j], vs[2].val[j], vc[1].val[j]);
                sum1.val[j] = vfmaq_f32(sum1.val[j], vs[3].val[j], vc[2].val[j]);
                sum1.val[j] = vfmaq_f32(sum1.val[j], vs[0].val[j], vc[3].val[j]);
                sum2.val[j] = vmulq_f32(             vs[5].val[j], vc[4].val[j]);
                sum2.val[j] = vfmaq_f32(sum2.val[j], vs[6].val[j], vc[5].val[j]);
                sum2.val[j] = vfmaq_f32(sum2.val[j], vs[7].val[j], vc[6].val[j]);
                sum2.val[j] = vfmaq_f32(sum2.val[j], vs[4].val[j], vc[7].val[j]);
                if (dst_silu) {
                    sum1.val[j] = v_silu(sum1.val[j]);
                    sum2.val[j] = v_silu(sum2.val[j]);
                }
            }
            vst1q_f32_x2(x+0, sum1);
            vst1q_f32_x2(x+8, sum2);
            x0 += nb11/sizeof(float);
            x  += nr;
            vs[1] = vld1q_f32_x2(x0+0);
            vs[5] = vld1q_f32_x2(x0+8);
            for (int j = 0; j < 2; ++j) {
                sum1.val[j] = vmulq_f32(             vs[2].val[j], vc[0].val[j]);
                sum1.val[j] = vfmaq_f32(sum1.val[j], vs[3].val[j], vc[1].val[j]);
                sum1.val[j] = vfmaq_f32(sum1.val[j], vs[0].val[j], vc[2].val[j]);
                sum1.val[j] = vfmaq_f32(sum1.val[j], vs[1].val[j], vc[3].val[j]);
                sum2.val[j] = vmulq_f32(             vs[6].val[j], vc[4].val[j]);
                sum2.val[j] = vfmaq_f32(sum2.val[j], vs[7].val[j], vc[5].val[j]);
                sum2.val[j] = vfmaq_f32(sum2.val[j], vs[4].val[j], vc[6].val[j]);
                sum2.val[j] = vfmaq_f32(sum2.val[j], vs[5].val[j], vc[7].val[j]);
                if (dst_silu) {
                    sum1.val[j] = v_silu(sum1.val[j]);
                    sum2.val[j] = v_silu(sum2.val[j]);
                }
            }
            vst1q_f32_x2(x+0, sum1);
            vst1q_f32_x2(x+8, sum2);
            x0 += nb11/sizeof(float);
            x  += nr;
            vs[2] = vld1q_f32_x2(x0+0);
            vs[6] = vld1q_f32_x2(x0+8);
            for (int j = 0; j < 2; ++j) {
                sum1.val[j] = vmulq_f32(             vs[3].val[j], vc[0].val[j]);
                sum1.val[j] = vfmaq_f32(sum1.val[j], vs[0].val[j], vc[1].val[j]);
                sum1.val[j] = vfmaq_f32(sum1.val[j], vs[1].val[j], vc[2].val[j]);
                sum1.val[j] = vfmaq_f32(sum1.val[j], vs[2].val[j], vc[3].val[j]);
                sum2.val[j] = vmulq_f32(             vs[7].val[j], vc[4].val[j]);
                sum2.val[j] = vfmaq_f32(sum2.val[j], vs[4].val[j], vc[5].val[j]);
                sum2.val[j] = vfmaq_f32(sum2.val[j], vs[5].val[j], vc[6].val[j]);
                sum2.val[j] = vfmaq_f32(sum2.val[j], vs[6].val[j], vc[7].val[j]);
                if (dst_silu) {
                    sum1.val[j] = v_silu(sum1.val[j]);
                    sum2.val[j] = v_silu(sum2.val[j]);
                }
            }
            vst1q_f32_x2(x+0, sum1);
            vst1q_f32_x2(x+8, sum2);
            x0 += nb11/sizeof(float);
            x  += nr;
            vs[3] = vld1q_f32_x2(x0+0);
            vs[7] = vld1q_f32_x2(x0+8);
            for (int j = 0; j < 2; ++j) {
                sum1.val[j] = vmulq_f32(             vs[0].val[j], vc[0].val[j]);
                sum1.val[j] = vfmaq_f32(sum1.val[j], vs[1].val[j], vc[1].val[j]);
                sum1.val[j] = vfmaq_f32(sum1.val[j], vs[2].val[j], vc[2].val[j]);
                sum1.val[j] = vfmaq_f32(sum1.val[j], vs[3].val[j], vc[3].val[j]);
                sum2.val[j] = vmulq_f32(             vs[4].val[j], vc[4].val[j]);
                sum2.val[j] = vfmaq_f32(sum2.val[j], vs[5].val[j], vc[5].val[j]);
                sum2.val[j] = vfmaq_f32(sum2.val[j], vs[6].val[j], vc[6].val[j]);
                sum2.val[j] = vfmaq_f32(sum2.val[j], vs[7].val[j], vc[7].val[j]);
                if (dst_silu) {
                    sum1.val[j] = v_silu(sum1.val[j]);
                    sum2.val[j] = v_silu(sum2.val[j]);
                }
            }
            vst1q_f32_x2(x+0, sum1);
            vst1q_f32_x2(x+8, sum2);
            x0 += nb11/sizeof(float);
            x  += nr;
        }
        int idx = 0;
        for (int it = 4*(nt/4); it < nt; ++it) {
            vs[idx+0] = vld1q_f32_x2(x0+0);
            vs[idx+4] = vld1q_f32_x2(x0+8);
            idx = (idx + 1) & 3;
            float32x4x2_t sum1 = {}, sum2 = {};
            for (int k = 0; k < 4; ++k) {
                int ii = (idx + k) & 3;
                for (int j = 0; j < 2; ++j) {
                    sum1.val[j] = vfmaq_f32(sum1.val[j], vs[ii+0].val[j], vc[k+0].val[j]);
                    sum2.val[j] = vfmaq_f32(sum2.val[j], vs[ii+4].val[j], vc[k+4].val[j]);
                }
            }
            if (dst_silu) {
                for (int j = 0; j < 2; ++j) {
                    sum1.val[j] = v_silu(sum1.val[j]);
                    sum2.val[j] = v_silu(sum2.val[j]);
                }
            }
            vst1q_f32_x2(x+0, sum1);
            vst1q_f32_x2(x+8, sum2);
            x0 += nb11/sizeof(float);
            x  += nr;
        }
        for (int k = 0; k < 4; ++k) {
            int ii = (idx + k) & 3;
            vst1q_f32_x2(aux + 8*k +  0, vs[ii+0]);
            vst1q_f32_x2(aux + 8*k + 32, vs[ii+4]);
        }
        for (int j = 0; j < 8; ++j) {
            for (int ic = 0; ic < 4; ++ic) {
                s[(j+0)*nb21/sizeof(float) + ic] = aux[j + 8*ic +  0];
                s[(j+8)*nb21/sizeof(float) + ic] = aux[j + 8*ic + 32];
            }
        }
    }
    return true;
#else
    return false;
#endif
}

namespace {
inline float sum_row_squared(int ncols, const float * x) {
    float sum = 0;
    int i = 0;
#ifdef __AVX2__
    auto vsum = _mm256_setzero_ps();
    for (; i < ncols - 7; i += 8) {
        auto vx = _mm256_loadu_ps(x + i);
        vsum = _mm256_fmadd_ps(vx, vx, vsum);
    }
    sum = hsum_float_8(vsum);
#endif
    for (; i < ncols; ++i) sum += x[i]*x[i];
    //for (int j = 0; j < ncols; ++j) sum += x[j]*x[j];
    return sum;
}
inline float sum_row_squared(int ncols, const ggml_half * x) {
    float sum = 0;
    for (int j = 0; j < ncols; ++j) {
        float v = GGML_FP16_TO_FP32(x[j]);
        sum += v*v;
    }
    return sum;
}
inline float sum_row_squared(int ncols, const ggml_bf16_t * x) {
    float sum = 0;
    for (int j = 0; j < ncols; ++j) {
        float v = GGML_BF16_TO_FP32(x[j]);
        sum += v*v;
    }
    return sum;
}
inline void rms_rms_add(int ncols, float scale1, float scale2, const float * x1, const float * x2, const float * c1, const float * c2, float * dst) {
    int j = 0;
#ifdef __AVX2__
    auto vs1 = _mm256_set1_ps(scale1);
    auto vs2 = _mm256_set1_ps(scale2);
    for (; j < ncols - 7; j += 8) {
        auto vx1 = _mm256_loadu_ps(x1 + j);
        auto vx2 = _mm256_loadu_ps(x2 + j);
        auto vc1 = _mm256_loadu_ps(c1 + j);
        auto vc2 = _mm256_loadu_ps(c2 + j);
        auto vy = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(vs1, vc1), vx1), _mm256_mul_ps(_mm256_mul_ps(vs2, vc2), vx2));
        _mm256_storeu_ps(dst + j, vy);
    }
#endif
    for (; j < ncols; ++j) {
        dst[j] = scale1 * c1[j] * x1[j] + scale2 * c2[j] * x2[j];
    }
}
inline void rms_rms_add(int ncols, float scale1, float scale2, const ggml_half * x1, const ggml_half * x2, const float * c1, const float * c2, float * dst) {
    for (int j = 0; j < ncols; ++j) {
        float v1 = GGML_FP16_TO_FP32(x1[j]);
        float v2 = GGML_FP16_TO_FP32(x2[j]);
        dst[j] = scale1 * c1[j] * v1 + scale2 * c2[j] * v2;
    }
}
inline void rms_rms_add(int ncols, float scale1, float scale2, const ggml_bf16_t * x1, const ggml_bf16_t * x2, const float * c1, const float * c2, float * dst) {
    for (int j = 0; j < ncols; ++j) {
        float v1 = GGML_BF16_TO_FP32(x1[j]);
        float v2 = GGML_BF16_TO_FP32(x2[j]);
        dst[j] = scale1 * c1[j] * v1 + scale2 * c2[j] * v2;
    }
}
}

void iqk_rms_rms_add(struct ggml_tensor * dst, int ith, int nth) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2];
    const struct ggml_tensor * src3 = dst->src[3];

    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src2) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src2, dst));
    GGML_ASSERT(ggml_nrows(src1) == 1 && ggml_nrows(src3) == 1);
    GGML_ASSERT(src0->ne[0] == src1->ne[0] && src2->ne[0] == src3->ne[0]);
    GGML_ASSERT(src0->type == src2->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_ASSERT(eps > 0.0f);

    int nrows = ggml_nrows(dst);
    int nrows_per_thread = (nrows + nth - 1)/nth;
    int first = ith*nrows_per_thread;
    int last  = MIN(nrows, first + nrows_per_thread);

    const float * c1 = (float *) src1->data;
    const float * c2 = (float *) src3->data;

    const int ncols = dst->ne[0];

    for (int ir = first; ir < last; ++ir) {
        float * y = (float *)dst->data + ir*ncols;

        float sum1 = 0, sum2 = 0;
        if (src0->type == GGML_TYPE_F32) {
            sum1 = sum_row_squared(ncols, (const float *)src0->data + ir*ncols);
            sum2 = sum_row_squared(ncols, (const float *)src2->data + ir*ncols);
        }
        else if (src0->type == GGML_TYPE_F16) {
            sum1 = sum_row_squared(ncols, (const ggml_half *)src0->data + ir*ncols);
            sum2 = sum_row_squared(ncols, (const ggml_half *)src2->data + ir*ncols);
        }
        else {
            sum1 = sum_row_squared(ncols, (const ggml_bf16_t *)src0->data + ir*ncols);
            sum2 = sum_row_squared(ncols, (const ggml_bf16_t *)src2->data + ir*ncols);
        }

        const float mean1  = sum1/ncols;
        const float mean2  = sum2/ncols;
        const float scale1 = 1.0f/sqrtf(mean1 + eps);
        const float scale2 = 1.0f/sqrtf(mean2 + eps);
        if (src0->type == GGML_TYPE_F32) {
            rms_rms_add(ncols, scale1, scale2, (const float *)src0->data + ir*ncols, (const float *)src2->data + ir*ncols, c1, c2, y);
        }
        else if (src0->type == GGML_TYPE_F16) {
            rms_rms_add(ncols, scale1, scale2, (const ggml_half *)src0->data + ir*ncols, (const ggml_half *)src2->data + ir*ncols, c1, c2, y);
        }
        else {
            rms_rms_add(ncols, scale1, scale2, (const ggml_bf16_t *)src0->data + ir*ncols, (const ggml_bf16_t *)src2->data + ir*ncols, c1, c2, y);
        }
    }
}
