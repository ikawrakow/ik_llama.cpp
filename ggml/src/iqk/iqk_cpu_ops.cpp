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

    if (src->type == GGML_TYPE_F32) {
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
        return;
    }

    auto traits = ggml_internal_get_type_traits(src->type);
    GGML_ASSERT(traits.to_float != nullptr);
    const size_t blck_size = traits.blck_size;
    const size_t type_size = traits.type_size;
    GGML_ASSERT(blck_size > 0 && (nh % blck_size == 0 || blck_size % nh == 0));

    for (int ir = first; ir < last; ++ir) {
        int i3 = ir / (dst->ne[1] * dst->ne[2] * nc);
        int i2 = (ir - i3*dst->ne[1] * dst->ne[2] * nc)/(dst->ne[1] * nc);
        int i1 = (ir - i3*dst->ne[1] * dst->ne[2] * nc - i2*dst->ne[1]*nc)/nc;
        int ic = (ir - i3*dst->ne[1] * dst->ne[2] * nc - i2*dst->ne[1]*nc - i1*nc);

        const char * x_row  = (const char *)src->data + i3*src->nb[3] + i2*src->nb[2] + i1*src->nb[1];
        const size_t offset = ((size_t)ic * nh / blck_size) * type_size;
        float      * y      = (float *)((char *)dst->data + i3*dst->nb[3] + i2*dst->nb[2] + i1*dst->nb[1]) + ic*nh;
        traits.to_float(x_row + offset, y, nh);
        fast_ht(nh, y);
    }
}

namespace {
// K2 (t16) trellis: tensor_core_perm maps the window order to the row-major
// (in*16 + out) position inside a 16x16 tile. Dumped from exllamav3's
// modules.quant.exl3_lib.quantize.tensor_core_perm (verified bit-exact).
static const uint8_t k2_t16_perm[256] = {
      0,  16, 128, 144,   8,  24, 136, 152,  32,  48, 160, 176,  40,  56, 168, 184,
     64,  80, 192, 208,  72,  88, 200, 216,  96, 112, 224, 240, 104, 120, 232, 248,
      1,  17, 129, 145,   9,  25, 137, 153,  33,  49, 161, 177,  41,  57, 169, 185,
     65,  81, 193, 209,  73,  89, 201, 217,  97, 113, 225, 241, 105, 121, 233, 249,
      2,  18, 130, 146,  10,  26, 138, 154,  34,  50, 162, 178,  42,  58, 170, 186,
     66,  82, 194, 210,  74,  90, 202, 218,  98, 114, 226, 242, 106, 122, 234, 250,
      3,  19, 131, 147,  11,  27, 139, 155,  35,  51, 163, 179,  43,  59, 171, 187,
     67,  83, 195, 211,  75,  91, 203, 219,  99, 115, 227, 243, 107, 123, 235, 251,
      4,  20, 132, 148,  12,  28, 140, 156,  36,  52, 164, 180,  44,  60, 172, 188,
     68,  84, 196, 212,  76,  92, 204, 220, 100, 116, 228, 244, 108, 124, 236, 252,
      5,  21, 133, 149,  13,  29, 141, 157,  37,  53, 165, 181,  45,  61, 173, 189,
     69,  85, 197, 213,  77,  93, 205, 221, 101, 117, 229, 245, 109, 125, 237, 253,
      6,  22, 134, 150,  14,  30, 142, 158,  38,  54, 166, 182,  46,  62, 174, 190,
     70,  86, 198, 214,  78,  94, 206, 222, 102, 118, 230, 246, 110, 126, 238, 254,
      7,  23, 135, 151,  15,  31, 143, 159,  39,  55, 167, 183,  47,  63, 175, 191,
     71,  87, 199, 215,  79,  95, 207, 223, 103, 119, 231, 247, 111, 127, 239, 255,
};

static inline uint32_t k2_t16_fshr(uint32_t lo, uint32_t hi, uint32_t s) {
    s &= 31;
    return s == 0 ? lo : (hi << (32 - s)) | (lo >> s);
}

// mul1: exact EXL3 codebook. The integer sum is reinterpreted as fp16 bits, then
// the affine map is evaluated like k2_codec.mul1_decode (fp32, one rounding to fp16).
static inline float k2_t16_mul1(uint16_t w) {
    const uint32_t x = uint32_t(w) * 0x83DCD12Du;
    const uint32_t s = 0x6400u + ((x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24));
    const float h = GGML_FP16_TO_FP32(uint16_t(s));
    const float v = h * 0.00676727294921875f + (-10.3828125f);
    return GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(v));
}
}

#if defined(__AVX2__) && defined(__F16C__)
// Dense AVX2 path: defined after the tile helpers (see below), dereferenced here.
static bool k2_t16_dense_avx2(struct ggml_tensor * dst, int ith, int nth);
#endif

void iqk_t16_matmul(struct ggml_tensor * dst, int ith, int nth) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_I16);
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(w->ne[3] == 1 && x->ne[2] == 1 && x->ne[3] == 1);

    const int K = int(w->ne[0] / 16);
    const int J = int(w->ne[1]);
    const int I = int(w->ne[2]);
    const int in = 16*I;
    const int out = 16*J;
    const int n = int(x->ne[1]);

    GGML_ASSERT(K >= 2 && K <= 6);
    GGML_ASSERT(x->ne[0] == in && dst->ne[0] == out && dst->ne[1] == n);

    const int nw32 = 8*K;
    const int64_t xs = x->nb[1] / sizeof(float);
    const int64_t ys = dst->nb[1] / sizeof(float);

#if defined(__AVX2__) && defined(__F16C__)
    {
        const char * e = getenv("K2_T16_CPU_SCALAR");
        if (!(e && atoi(e) != 0) && k2_t16_dense_avx2(dst, ith, nth)) return;
    }
#endif

    for (int j = ith; j < J; j += nth) {
        float * ybase = (float *)dst->data + 16*j;
        // the op accumulates; zero the output slab owned by this thread
        for (int c = 0; c < 16; ++c) {
            float * yc = ybase + c;
            for (int t = 0; t < n; ++t) {
                yc[t*ys] = 0.0f;
            }
        }
        for (int i = 0; i < I; ++i) {
            const uint32_t * tile = (const uint32_t *)((const char *)w->data + i*w->nb[2] + j*w->nb[1]);
            uint16_t win[256];
            for (int t = 0; t < 128; ++t) {
                const int b0 = t*2*K + K - 16 + 256*K;
                const int b2 = b0 + K + 16;
                const int i0 = (b0 / 32) % nw32;
                const int i1u = (b2 - 1) / 32;
                const int i1 = i1u % nw32;
                const uint32_t w1 = k2_t16_fshr(tile[i1], tile[i0], uint32_t((i1u + 1)*32 - b2));
                win[2*t]     = uint16_t((w1 >> K) & 0xFFFFu);
                win[2*t + 1] = uint16_t(w1 & 0xFFFFu);
            }
            float m[256];
            for (int t = 0; t < 256; ++t) {
                m[k2_t16_perm[t]] = k2_t16_mul1(win[t]);
            }
            const float * xrow = (const float *)((const char *)x->data + 16*i*sizeof(float));
            for (int r = 0; r < 16; ++r) {
                const float * mr = m + 16*r;
                for (int t = 0; t < n; ++t) {
                    const float xv = xrow[r + t*xs];
                    float * yt = ybase + t*ys;
                    for (int c = 0; c < 16; ++c) {
                        yt[c] += mr[c] * xv;
                    }
                }
            }
        }
    }
}

namespace {
// Scalar reference implementation of the K2 trellis MoE matmul. It is the specification
// of the op and the fallback for builds without AVX2/F16C; iqk_t16_mul_mat_id at the end
// of this block dispatches to the vectorised kernel when it can handle the shape.
void k2_t16_moe_ref(struct ggml_tensor * dst, int ith, int nth) {
    // MoE reference: y[:,u,t] = sv_e * H( w_e^T H(x[:,t] * su_e) ), e = ids[u,t].
    // Mirrors the dense graph (mul -> hadamard -> matmul -> hadamard -> mul), with the
    // transforms in-op because su/sv are per expert. su/sv may be NULL (identity).
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const ggml_tensor * su  = dst->src[3];
    const ggml_tensor * sv  = dst->src[4];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_I16);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);

    const int K = int(w->ne[0] / 16);
    const int J = int(w->ne[1]);
    const int I = int(w->ne[2]);
    const int E = int(w->ne[3]);
    const int in = 16*I, out = 16*J;
    const int n = int(x->ne[1]);
    const int n_used = int(ids->ne[0]);
    const int nw32 = 8*K;
    GGML_ASSERT(K >= 2 && K <= 6);
    GGML_ASSERT(x->ne[0] == in && dst->ne[0] == out && dst->ne[1] == n_used && dst->ne[2] == n);
    GGML_ASSERT(ids->ne[1] == n);
    GGML_ASSERT(in % 128 == 0 && out % 128 == 0);
    if (su) { GGML_ASSERT(su->type == GGML_TYPE_F32 && su->ne[0] == in && su->ne[1] == E); }
    if (sv) { GGML_ASSERT(sv->type == GGML_TYPE_F32 && sv->ne[0] == out && sv->ne[1] == E); }

    std::vector<float> xp((size_t) in);
    std::vector<float> yp((size_t) out);

    const int64_t total = (int64_t) n * n_used;
    for (int64_t tu = ith; tu < total; tu += nth) {
        const int u = int(tu % n_used);
        const int t = int(tu / n_used);
        const int32_t e = *(const int32_t *)((const char *) ids->data + u*ids->nb[0] + t*ids->nb[1]);
        GGML_ASSERT(e >= 0 && e < E);

        const float * xcol = (const float *)((const char *) x->data + t * x->nb[1]);
        // xp = H(x[:,t] * su_e)
        for (int i = 0; i < in; ++i) {
            float v = xcol[i];
            if (su) {
                v *= *(const float *)((const char *) su->data + (int64_t) e * su->nb[1] + (int64_t) i * 4);
            }
            xp[i] = v;
        }
        for (int blk = 0; blk < in/128; ++blk) { fast_ht(128, xp.data() + (size_t) blk * 128); }

        std::fill(yp.begin(), yp.end(), 0.0f);
        const char * wbase = (const char *) w->data + (size_t) e * w->nb[3];
        for (int j = 0; j < J; ++j) {
            for (int i = 0; i < I; ++i) {
                const uint32_t * tile = (const uint32_t *)(wbase + (size_t) i * w->nb[2] + (size_t) j * w->nb[1]);
                uint16_t win[256];
                for (int tt = 0; tt < 128; ++tt) {
                    const int b0 = tt*2*K + K - 16 + 256*K;
                    const int b2 = b0 + K + 16;
                    const int i0 = (b0 / 32) % nw32;
                    const int i1u = (b2 - 1) / 32;
                    const int i1 = i1u % nw32;
                    const uint32_t w1 = k2_t16_fshr(tile[i1], tile[i0], uint32_t((i1u + 1)*32 - b2));
                    win[2*tt]     = uint16_t((w1 >> K) & 0xFFFFu);
                    win[2*tt + 1] = uint16_t(w1 & 0xFFFFu);
                }
                float m[256];
                for (int tt = 0; tt < 256; ++tt) {
                    m[k2_t16_perm[tt]] = k2_t16_mul1(win[tt]);
                }
                for (int rr = 0; rr < 16; ++rr) {
                    const float xv = xp[(size_t) 16*i + rr];
                    const float * mr = m + 16*rr;
                    float * yo = yp.data() + (size_t) 16*j;
                    for (int c = 0; c < 16; ++c) {
                        yo[c] += mr[c] * xv;
                    }
                }
            }
        }
        // yp = sv_e * H(yp)
        for (int blk = 0; blk < out/128; ++blk) { fast_ht(128, yp.data() + (size_t) blk * 128); }
        float * ycol = (float *)((char *) dst->data + u*dst->nb[1] + t*dst->nb[2]);
        for (int o = 0; o < out; ++o) {
            float v = yp[o];
            if (sv) {
                v *= *(const float *)((const char *) sv->data + (int64_t) e * sv->nb[1] + (int64_t) o * 4);
            }
            ycol[o] = v;
        }
    }
}
} // namespace k2_t16_moe_ref (scalar reference)

#if defined __AVX2__ && defined __F16C__

// The scalar reference accumulates with a separate multiply and add, and so does the
// auto-vectorised form GCC emits for it. GCC's default (-ffp-contract=fast) would fuse the
// multiply and add of the kernel below into an FMA, which rounds once instead of twice and
// changes the last bits of the output; disable contraction for this block so both paths
// are bit-identical (see tests/test-k2-moe-cpu.cpp, which reports 0 error).
#if defined __GNUC__ && !defined __clang__
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif

namespace {

// ---- vectorised K2 trellis (t16) MoE matmul --------------------------------
//
// Weight layout: one (i,j) tile is 16K uint16 == 8K uint32 words holding a 16x16 block of
// trellis weights. With tpos = inv_perm[q] (q = 16*r + c, exllamav3's tensor_core_perm) the
// weight of output row r / column c of the tile is
//     fshr(tile[hi], tile[lo], sh) & 0xffff
//     b = tpos*K + K - 16 + 256*K,  lo = (b+15)/32,  hi = b/32,  sh = (lo+1)*32 - (b+16)
// with lo/hi taken mod 8K (all four expressions come from the scalar loop above, which
// applies them to two windows at a time; this is the same expression evaluated per output
// element, which was verified to give bit-identical windows for K = 2..6).
//
// The two useful properties for SIMD:
//  * for a fixed r the 16 columns split into two groups of 8 in which `sh` is constant and
//    lo == base + K*c, so the 8 windows of a group are 8 words at a fixed stride plus one
//    uniform shift, and the decoded tile comes out already in (r,c) order (no 16x16
//    permutation needed - which is why there is no transpose anywhere below);
//  * the decode of a tile does not depend on the token, so a tile decoded for one expert
//    is reused by every token routed to that expert (NT below).

struct K2_t16_tab {
    int32_t  lo[16][2][8];   // word index of the bits that are shifted down
    int32_t  hi[16][2][8];   // word index of the bits that are shifted in
    uint32_t sh[16][2];      // right shift, constant per (r, half)
    bool     same[16][2];    // hi == lo for all 8 lanes -> the second load set is skipped
    bool     ok;             // false: unexpected layout, use the scalar reference
};

K2_t16_tab k2_t16_make_tab(int K) {
    K2_t16_tab t{};
    t.ok = true;
    int tpos_of[256];
    for (int tt = 0; tt < 256; ++tt) tpos_of[k2_t16_perm[tt]] = tt;
    const int nw = 8*K;
    for (int r = 0; r < 16; ++r) {
        for (int g = 0; g < 2; ++g) {
            uint32_t sh = 0;
            bool same = true;
            for (int c = 0; c < 8; ++c) {
                const int tpos = tpos_of[16*r + 8*g + c];
                const int b  = tpos*K + K - 16 + 256*K;
                const int ih = b >> 5;
                const int il = (b + 15) >> 5;
                const uint32_t s = uint32_t((il + 1)*32 - (b + 16));
                if (c == 0) {
                    sh = s;
                } else if (s != sh) {
                    t.ok = false;   // would need per-lane shifts (does not happen)
                }
                t.lo[r][g][c] = il % nw;
                t.hi[r][g][c] = ih % nw;
                if (t.lo[r][g][c] != (t.lo[r][g][0] + K*c) % nw) t.ok = false;
                if (t.hi[r][g][c] != t.lo[r][g][c]) same = false;
            }
            t.sh[r][g]   = sh & 31u;
            t.same[r][g] = same;
        }
    }
    return t;
}

const K2_t16_tab * k2_t16_tab_for(int K) {
    static const K2_t16_tab tab[7] = {
        K2_t16_tab{}, K2_t16_tab{},
        k2_t16_make_tab(2), k2_t16_make_tab(3), k2_t16_make_tab(4),
        k2_t16_make_tab(5), k2_t16_make_tab(6),
    };
    if (K < 2 || K > 6) return nullptr;
    return tab[K].ok ? &tab[K] : nullptr;
}

// stages h = 1, 2, 4 of a 128-point FWHT inside one 8-float register
static inline __m256 k2_ht8(__m256 v) {
    {
        const __m256 p = _mm256_permute_ps(v, 0xB1);
        v = _mm256_blend_ps(_mm256_add_ps(v, p), _mm256_sub_ps(p, v), 0xAA);
    }
    {
        const __m256 p = _mm256_permute_ps(v, 0x4E);
        v = _mm256_blend_ps(_mm256_add_ps(v, p), _mm256_sub_ps(p, v), 0xCC);
    }
    {
        const __m256 p = _mm256_permute2f128_ps(v, v, 0x01);
        v = _mm256_blend_ps(_mm256_add_ps(v, p), _mm256_sub_ps(p, v), 0xF0);
    }
    return v;
}

// fast_ht(128, x) with AVX2: same butterflies in the same order, then the same
// accumulated 1/sqrt(2)^7 scale, so the result is bit-identical to the scalar version.
static inline void k2_ht128(float * x) {
    __m256 r[16];
    for (int i = 0; i < 16; ++i) r[i] = k2_ht8(_mm256_loadu_ps(x + 8*i));
    for (int h = 1; h < 16; h <<= 1) {
        for (int i = 0; i < 16; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const __m256 a = r[j], b = r[j + h];
                r[j]     = _mm256_add_ps(a, b);
                r[j + h] = _mm256_sub_ps(a, b);
            }
        }
    }
    float scale = 1.0f;
    for (int h = 1; h < 128; h <<= 1) scale *= 0.707106781f;
    const __m256 sc = _mm256_set1_ps(scale);
    for (int i = 0; i < 16; ++i) _mm256_storeu_ps(x + 8*i, _mm256_mul_ps(r[i], sc));
}

// k2_t16_mul1 for 8 windows at once. The reference computes
//     s = 0x6400 + bytesum(w * 0x83DCD12D);  h = fp16(s);
//     v = h*0.00676727294921875 - 10.3828125;  result = fp32(fp16(v))
// The fp16 pattern 0x6400 + s (s <= 1020) is the normal half 1024 + s exactly, so h can be
// built in fp32 without a half conversion; the final rounding uses the same hardware
// (round-to-nearest-even) instruction that GGML_FP32_TO_FP16 expands to on F16C builds.
static inline __m256 k2_t16_mul1_8(__m256i w) {
    const __m256i x  = _mm256_mullo_epi32(_mm256_and_si256(w, _mm256_set1_epi32(0x0000FFFF)),
                                          _mm256_set1_epi32((int) 0x83DCD12Du));
    const __m256i bs = _mm256_madd_epi16(_mm256_maddubs_epi16(x, _mm256_set1_epi8(1)),
                                         _mm256_set1_epi16(1));   // 4-byte sum, 0..1020
    const __m256 h = _mm256_add_ps(_mm256_cvtepi32_ps(bs), _mm256_set1_ps(1024.0f));
    const __m256 v = _mm256_sub_ps(_mm256_mul_ps(h, _mm256_set1_ps(0.00676727294921875f)),
                                   _mm256_set1_ps(10.3828125f));
    return _mm256_cvtph_ps(_mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
}

// Decode one 16x16 tile and accumulate it into `acc` for NT tokens.
// `xr[nt]` points at the transformed input row of token nt; element base+r is the input
// row of the tile. Accumulation order per output element is (i, r) - identical to the
// reference, and there is no FMA, so every add rounds exactly as the scalar loop does.
//
// The 8 windows of a (r, half) group are 8 separate words, so they are assembled with 8
// scalar loads + vpinsrd. That is deliberately not a gather: on this class of CPU
// (Coffee Lake) vpgatherdd costs ~29 cycles per 8 elements here and made the whole kernel
// 3.3x slower than these plain loads (measurements in qat_tools/K2_MOE_CPU_KERNEL.md).
template <int NT>
static inline void k2_t16_tile_accum(const uint32_t * tile, const K2_t16_tab & tab,
                                     const float * const * xr, int base, __m256 acc[NT][2]) {
    for (int r = 0; r < 16; ++r) {
        for (int g = 0; g < 2; ++g) {
            const int * L = tab.lo[r][g];
            const int * H = tab.hi[r][g];
            const int s = (int) tab.sh[r][g];
            const __m256i lo = _mm256_setr_epi32((int) tile[L[0]], (int) tile[L[1]], (int) tile[L[2]], (int) tile[L[3]],
                                                (int) tile[L[4]], (int) tile[L[5]], (int) tile[L[6]], (int) tile[L[7]]);
            __m256i hi = lo;
            if (!tab.same[r][g]) {
                hi = _mm256_setr_epi32((int) tile[H[0]], (int) tile[H[1]], (int) tile[H[2]], (int) tile[H[3]],
                                       (int) tile[H[4]], (int) tile[H[5]], (int) tile[H[6]], (int) tile[H[7]]);
            }
            const __m256i w1 = s == 0 ? lo
                : _mm256_or_si256(_mm256_srlv_epi32(lo, _mm256_set1_epi32(s)),
                                  _mm256_sllv_epi32(hi, _mm256_set1_epi32(32 - s)));
            const __m256 m = k2_t16_mul1_8(w1);
            for (int nt = 0; nt < NT; ++nt) {
                const __m256 xv = _mm256_broadcast_ss(xr[nt] + base + r);
                acc[nt][g] = _mm256_add_ps(acc[nt][g], _mm256_mul_ps(m, xv));
            }
        }
    }
}

struct K2_moe_args {
    const K2_t16_tab * tab;
    const char * w;  size_t w_nb1, w_nb2, w_nb3;
    const float * x; int64_t x_nb1;
    const float * su, * sv; int64_t su_nb1, sv_nb1;
    char * y; int64_t y_nb1, y_nb2;
    int I, J, n_used;
    float * xp;     // NT*in scratch
    float * yacc;   // NT*128 scratch
};

// One expert x NT tokens: transform the inputs, run the tile sweep in 128-output blocks
// (the granularity of the output Hadamard), then transform and store each token's block.
template <int NT>
static void k2_t16_moe_chunk(const K2_moe_args & a, int e, const int32_t * pairs) {
    const int in = 16*a.I;
    const float * xr[NT];
    int tk[NT], uk[NT];
    for (int nt = 0; nt < NT; ++nt) {
        const int t = pairs[nt] / a.n_used;
        const int u = pairs[nt] % a.n_used;
        tk[nt] = t;
        uk[nt] = u;
        float * xd = a.xp + (size_t) nt * in;
        const float * xcol = (const float *)((const char *) a.x + (int64_t) t * a.x_nb1);
        if (a.su) {
            const float * surow = (const float *)((const char *) a.su + (int64_t) e * a.su_nb1);
            for (int k = 0; k < in; ++k) xd[k] = xcol[k] * surow[k];
        } else {
            std::memcpy(xd, xcol, (size_t) in * sizeof(float));
        }
        for (int blk = 0; blk < in/128; ++blk) k2_ht128(xd + 128*blk);
        xr[nt] = xd;
    }
    const char * wbase = a.w + (size_t) e * a.w_nb3;
    for (int jb = 0; jb < a.J/8; ++jb) {
        for (int jt = 0; jt < 8; ++jt) {
            const int j = 8*jb + jt;
            __m256 acc[NT][2];
            for (int nt = 0; nt < NT; ++nt) {
                acc[nt][0] = _mm256_setzero_ps();
                acc[nt][1] = _mm256_setzero_ps();
            }
            for (int i = 0; i < a.I; ++i) {
                const uint32_t * tile = (const uint32_t *)(wbase + (size_t) i * a.w_nb2 + (size_t) j * a.w_nb1);
                k2_t16_tile_accum<NT>(tile, *a.tab, xr, 16*i, acc);
            }
            for (int nt = 0; nt < NT; ++nt) {
                _mm256_storeu_ps(a.yacc + (size_t) nt * 128 + 16*jt,     acc[nt][0]);
                _mm256_storeu_ps(a.yacc + (size_t) nt * 128 + 16*jt + 8, acc[nt][1]);
            }
        }
        const int o = 128*jb;
        const float * svrow = a.sv
            ? (const float *)((const char *) a.sv + (int64_t) e * a.sv_nb1) + o : nullptr;
        for (int nt = 0; nt < NT; ++nt) {
            float * yv = a.yacc + (size_t) nt * 128;
            k2_ht128(yv);
            float * ycol = (float *)(a.y + (int64_t) uk[nt] * a.y_nb1 + (int64_t) tk[nt] * a.y_nb2) + o;
            if (svrow) {
                for (int k = 0; k < 128; ++k) ycol[k] = yv[k] * svrow[k];
            } else {
                std::memcpy(ycol, yv, 128 * sizeof(float));
            }
        }
    }
}

struct K2_moe_scratch {
    std::vector<float>   xp;
    std::vector<float>   yacc;
    std::vector<int32_t> cnt, pos, fill, order;
};

static K2_moe_scratch & k2_moe_scratch_get() {
    thread_local K2_moe_scratch s;
    return s;
}

// Returns false when the shape is outside what this kernel handles (the caller then runs
// the scalar reference, which also carries the asserts).
static bool k2_t16_moe_avx2(struct ggml_tensor * dst, int ith, int nth) {
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const ggml_tensor * su  = dst->src[3];
    const ggml_tensor * sv  = dst->src[4];

    if (dst->type != GGML_TYPE_F32 || w->type != GGML_TYPE_I16) return false;
    if (x->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) return false;
    if (w->ne[0] % 16 != 0 || w->ne[3] < 1) return false;
    const int K = int(w->ne[0] / 16);
    if (K < 2 || K > 6) return false;
    const int J = int(w->ne[1]), I = int(w->ne[2]), E = int(w->ne[3]);
    const int in = 16*I, out = 16*J;
    const int n = int(x->ne[1]), n_used = int(ids->ne[0]);
    if (x->ne[0] != in || dst->ne[0] != out || dst->ne[1] != n_used || dst->ne[2] != n) return false;
    if (ids->ne[1] != n || n < 1 || n_used < 1) return false;
    if (in % 128 != 0 || out % 128 != 0) return false;
    if (su && (su->type != GGML_TYPE_F32 || su->ne[0] != in || su->ne[1] != E)) return false;
    if (sv && (sv->type != GGML_TYPE_F32 || sv->ne[0] != out || sv->ne[1] != E)) return false;

    const K2_t16_tab * tab = k2_t16_tab_for(K);
    if (!tab) return false;

    const int total = n * n_used;
    auto & sc = k2_moe_scratch_get();

    // counting sort of the (t,u) pairs by expert: tokens sharing an expert share the
    // decoded tiles, which is what makes the batched path cheaper than the reference
    sc.cnt.assign((size_t) E, 0);
    for (int t = 0; t < n; ++t) {
        const char * idrow = (const char *) ids->data + (size_t) t * ids->nb[1];
        for (int u = 0; u < n_used; ++u) {
            const int32_t e = *(const int32_t *)(idrow + (size_t) u * ids->nb[0]);
            if (e < 0 || e >= E) return false;
            ++sc.cnt[e];
        }
    }
    sc.pos.assign((size_t) E + 1, 0);
    int ngrp = 0;
    for (int e = 0; e < E; ++e) {
        sc.pos[e] = ngrp;
        ngrp += sc.cnt[e];
    }
    sc.pos[E] = ngrp;
    sc.order.assign((size_t) total, 0);
    sc.fill.assign(sc.pos.begin(), sc.pos.begin() + E);
    for (int t = 0; t < n; ++t) {
        const char * idrow = (const char *) ids->data + (size_t) t * ids->nb[1];
        for (int u = 0; u < n_used; ++u) {
            const int32_t e = *(const int32_t *)(idrow + (size_t) u * ids->nb[0]);
            sc.order[sc.fill[e]++] = t * n_used + u;
        }
    }

    if (sc.xp.size() < (size_t) 4 * in) sc.xp.resize((size_t) 4 * in);
    if (sc.yacc.size() < 4 * 128)       sc.yacc.resize(4 * 128);

    K2_moe_args a;
    a.tab    = tab;
    a.w      = (const char *) w->data;
    a.w_nb1  = w->nb[1];
    a.w_nb2  = w->nb[2];
    a.w_nb3  = w->nb[3];
    a.x      = (const float *) x->data;
    a.x_nb1  = x->nb[1];
    a.su     = su ? (const float *) su->data : nullptr;
    a.su_nb1 = su ? (int64_t) su->nb[1] : 0;
    a.sv     = sv ? (const float *) sv->data : nullptr;
    a.sv_nb1 = sv ? (int64_t) sv->nb[1] : 0;
    a.y      = (char *) dst->data;
    a.y_nb1  = dst->nb[1];
    a.y_nb2  = dst->nb[2];
    a.I      = I;
    a.J      = J;
    a.n_used = n_used;
    a.xp     = sc.xp.data();
    a.yacc   = sc.yacc.data();

    // work items are (expert, up to 4 tokens) chunks, spread over the threads round-robin
    int work = 0;
    for (int e = 0; e < E; ++e) {
        const int gs = sc.cnt[e];
        if (gs == 0) continue;
        const int32_t * pairs = sc.order.data() + sc.pos[e];
        for (int c0 = 0; c0 < gs; ) {
            const int rem = gs - c0;
            const bool mine = (work++ % nth) == ith;
            if (mine) {
                if (rem >= 4)      { k2_t16_moe_chunk<4>(a, e, pairs + c0); c0 += 4; }
                else if (rem >= 2) { k2_t16_moe_chunk<2>(a, e, pairs + c0); c0 += 2; }
                else               { k2_t16_moe_chunk<1>(a, e, pairs + c0); c0 += 1; }
            } else {
                c0 += rem >= 4 ? 4 : (rem >= 2 ? 2 : 1);
            }
        }
    }
    return true;
}

} // namespace

#if defined __GNUC__ && !defined __clang__
#pragma GCC pop_options
#endif

#endif // __AVX2__ && __F16C__

#if defined(__AVX2__) && defined(__F16C__)
// One dense (non-MoE) K2 matmul, vectorised by reusing k2_t16_tile_accum, which decodes 8
// windows per register with the same accumulation order (i, r) and no FMA as the scalar
// reference, so the results are bit-identical. The scalar reference above stays the
// specification and the fallback; K2_T16_CPU_SCALAR=1 forces it.
template <int NT>
static void k2_t16_dense_nt(struct ggml_tensor * dst, int ith, int nth) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    const int K = int(w->ne[0] / 16), J = int(w->ne[1]), I = int(w->ne[2]);
    const int n = int(x->ne[1]);
    const int64_t ys = dst->nb[1] / sizeof(float);
    const K2_t16_tab * tab = k2_t16_tab_for(K);
    if (!tab) return;
    for (int j = ith; j < J; j += nth) {
        for (int t0 = 0; t0 < n; t0 += NT) {
            const int ntl = (n - t0 < NT) ? (n - t0) : NT;
            float * yt[NT]; const float * xr[NT];
            __m256 acc[NT][2];
            for (int t = 0; t < NT; ++t) {
                const int tt = (t < ntl) ? t : 0;
                yt[t] = (float *) dst->data + 16*j + (t0 + tt)*ys;
                xr[t] = (const float *) ((const char *) x->data + (t0 + tt)*x->nb[1]);
                acc[t][0] = _mm256_setzero_ps();
                acc[t][1] = _mm256_setzero_ps();
            }
            for (int i = 0; i < I; ++i) {
                const uint32_t * tile = (const uint32_t *) ((const char *) w->data + i*w->nb[2] + j*w->nb[1]);
                k2_t16_tile_accum<NT>(tile, *tab, xr, 16*i, acc);
            }
            for (int t = 0; t < ntl; ++t) {
                _mm256_storeu_ps(yt[t],     acc[t][0]);
                _mm256_storeu_ps(yt[t] + 8, acc[t][1]);
            }
        }
    }
}

static bool k2_t16_dense_avx2(struct ggml_tensor * dst, int ith, int nth) {
    const int n = int(dst->src[1]->ne[1]);
    if (n == 1) { k2_t16_dense_nt<1>(dst, ith, nth); return true; }
    if (n == 2) { k2_t16_dense_nt<2>(dst, ith, nth); return true; }
    k2_t16_dense_nt<4>(dst, ith, nth); return true;   // NT=4 covers tails
}
#endif

void iqk_t16_mul_mat_id(struct ggml_tensor * dst, int ith, int nth) {
#if defined __AVX2__ && defined __F16C__
    if (k2_t16_moe_avx2(dst, ith, nth)) return;
#endif
    k2_t16_moe_ref(dst, ith, nth);
}

void iqk_t16_embed_get_rows(struct ggml_tensor * dst, int ith, int nth) {
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * sg  = dst->src[1];
    const ggml_tensor * ids = dst->src[2];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_I16 && sg->type == GGML_TYPE_F16 && ids->type == GGML_TYPE_I32);

    const int K      = dst->op_params[0];
    const int groups = dst->op_params[1];
    const int hidden = dst->op_params[2];
    const int W      = int(w->ne[0] / groups);
    const int nw     = W - 1;
    const int n      = int(ids->ne[0]);
    GGML_ASSERT(dst->ne[0] == hidden && dst->ne[1] == n);
    GGML_ASSERT(K >= 1 && K <= 8 && nw == 16*K);

    const uint16_t * signs = (const uint16_t *) sg->data;
    const int32_t  * tok   = (const int32_t *) ids->data;
    const char * data = (const char *) w->data;
    const int64_t row_bytes = (int64_t) w->ne[0] * sizeof(uint16_t);
    const uint32_t mask = (1u << K) - 1u;
    const int reps = (15 + K) / K;
    const int64_t vocab = w->ne[1];

    for (int t = ith; t < n; t += nth) {
        const int64_t id = tok[t];
        GGML_ASSERT(id >= 0 && id < vocab);
        const uint16_t * row = (const uint16_t *)(data + id * row_bytes);
        float * out = (float *) dst->data + (int64_t) t * hidden;
        for (int g = 0; g < groups; ++g) {
            const uint16_t * rg = row + (int64_t) g * W;
            const float scale = GGML_FP16_TO_FP32(rg[0]);
            const uint16_t * sw = rg + 1;
            uint16_t sym[256];
            for (int i = 0; i < 256; ++i) {
                const int bit = i * K;
                const int wi  = bit >> 4;
                const int off = bit & 15;
                const uint32_t win = (uint32_t) sw[wi] | ((uint32_t) sw[(wi + 1) % nw] << 16);
                sym[i] = (uint16_t)((win >> off) & mask);
            }
            float vals[256];
            for (int i = 0; i < 256; ++i) {
                uint32_t st = 0;
                for (int j = 0; j < reps; ++j) {
                    st |= (uint32_t) sym[(i - j) & 255] << (j * K);
                }
                vals[i] = k2_t16_mul1((uint16_t)(st & 0xFFFFu)) * scale;
            }
            fast_ht(256, vals);
            for (int j = 0; j < 256; ++j) {
                const int o = g * 256 + j;
                if (o >= hidden) break;
                out[o] = vals[j] * GGML_FP16_TO_FP32(signs[g * 256 + j]);
            }
        }
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

namespace {
template <typename Data, typename Idx>
inline void iqk_blend_row(int n, int nidx, const Data * x, const Idx * idx, Data * y, float c) {
    Data b;
    if constexpr (std::is_same_v<Data, float>) {
        b = c;
    }
    else if constexpr (std::is_same_v<Data, ggml_half>) {
        b = GGML_FP32_TO_FP16(c);
    }
    else {
        b = GGML_FP32_TO_BF16(c);
    }
    if (y != x) {
        for (int j = 0; j < n; ++j) y[j] = x[j];
    }
    for (int j = 0; j < nidx; ++j) y[idx[j]] = b;
}
}

void iqk_blend(struct ggml_tensor * dst, int ith, int nth) {
    auto src0 = dst->src[0];
    auto src1 = dst->src[1];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32 || src1->type == GGML_TYPE_I64);
    GGML_ASSERT(src1->ne[0] <= src0->ne[0]);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src0->ne[0] == dst->ne[0]);
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        GGML_ASSERT(src0->ne[dim] == src1->ne[dim]);
        GGML_ASSERT(src0->ne[dim] == dst->ne[dim]);
    }
    float c;
    std::memcpy(&c, dst->op_params, sizeof(c));
    int nrows = ggml_nrows(src0);
    int npt   = (nrows + nth - 1)/nth;
    int first = ith*npt;
    int last  = std::min(nrows, first + npt);
    for (int ir = first; ir < last; ++ir) {
        int ii = ir;
        int i3 = ii/(src0->ne[1]*src0->ne[2]); ii -= i3*src0->ne[1]*src0->ne[2];
        int i2 = ii/(src0->ne[1]);             ii -= i2*src0->ne[1];
        int i1 = ii;
        auto x   = (const char *)src0->data + i1*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
        auto y   = (      char *) dst->data + i1* dst->nb[1] + i2* dst->nb[2] + i3* dst->nb[3];
        auto idx = (const char *)src1->data + i1*src1->nb[1] + i2*src1->nb[2] + i3*src1->nb[3];
        if (src1->type == GGML_TYPE_I32) {
            if (src0->type == GGML_TYPE_F32) {
                iqk_blend_row(src0->ne[0], src1->ne[0], (const float *)x, (const int32_t *)idx, (float *)y, c);
            }
            else if (src0->type == GGML_TYPE_F16) {
                iqk_blend_row(src0->ne[0], src1->ne[0], (const ggml_half *)x, (const int32_t *)idx, (ggml_half *)y, c);
            }
            else {
                iqk_blend_row(src0->ne[0], src1->ne[0], (const ggml_bf16_t *)x, (const int32_t *)idx, (ggml_bf16_t *)y, c);
            }
        } else {
            if (src0->type == GGML_TYPE_F32) {
                iqk_blend_row(src0->ne[0], src1->ne[0], (const float *)x, (const int64_t *)idx, (float *)y, c);
            }
            else if (src0->type == GGML_TYPE_F16) {
                iqk_blend_row(src0->ne[0], src1->ne[0], (const ggml_half *)x, (const int64_t *)idx, (ggml_half *)y, c);
            }
            else {
                iqk_blend_row(src0->ne[0], src1->ne[0], (const ggml_bf16_t *)x, (const int64_t *)idx, (ggml_bf16_t *)y, c);
            }
        }
    }
}

namespace {
inline void iqk_add_f16(int n, const ggml_half * x, ggml_half * y) {
    int i = 0;
#if defined __AVX2__ && defined __F16C__
    for ( ; i + 7 < n; i += 8) {
        auto vx32 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x + i)));
        auto vy32 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(y + i)));
        vy32 = _mm256_add_ps(vy32, vx32);
        _mm_storeu_si128((__m128i *)(y + i), _mm256_cvtps_ph(vy32, _MM_FROUND_TO_NEAREST_INT));
    }
#endif
    for (; i < n; ++i) {
        float y32 = GGML_FP16_TO_FP32(y[i]) + GGML_FP16_TO_FP32(x[i]);
        y[i] = GGML_FP32_TO_FP16(y32);
    }
}
}

void iqk_mask_topk(struct ggml_tensor * dst, int ith, int nth) {
    auto mask = dst->src[0];
    auto topk = dst->src[1];
    GGML_ASSERT(mask->ne[0] >= topk->ne[0]);
    GGML_ASSERT(mask->ne[1] >= topk->ne[1] && mask->ne[2] == topk->ne[2] && mask->ne[3] == topk->ne[3]);
    GGML_ASSERT(ggml_are_same_shape(mask, dst));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(topk->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == dst->type);

    int nrows = ggml_nrows(dst);
    int npt = (nrows + nth - 1)/nth;
    int first = ith*npt;
    int last  = std::min(first + npt, nrows);

    auto hinf  = GGML_FP32_TO_FP16(-INFINITY);
    auto hzero = GGML_FP32_TO_FP16(0.0f);

    int n = dst->ne[0];
    int nidx = topk->ne[0];

    for (int ir = first; ir < last; ++ir) {
        int i3 = ir/(dst->ne[1]*dst->ne[2]);
        int i2 = (ir - i3*dst->ne[1]*dst->ne[2])/dst->ne[1];
        int i1 = ir - i3*dst->ne[1]*dst->ne[2] - i2*dst->ne[1];
        auto idx = (const int *)((const char *)topk->data + topk->nb[1]*i1 + topk->nb[2]*i2 + topk->nb[3]*i3);
        if (dst->type == GGML_TYPE_F16) {
            auto x = (const ggml_half *)((const char *)mask->data + mask->nb[1]*i1 + mask->nb[2]*i2 + mask->nb[3]*i3);
            auto y = (ggml_half *)((char *)dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
            if (i1 < topk->ne[1]) {
                for (int j = 0; j < n; ++j) y[j] = hinf;
                for (int j = 0; j < nidx; ++j) y[idx[j]] = hzero;
                iqk_add_f16(n, x, y);
            } else {
                for (int j = 0; j < n; ++j) y[j] = x[j];
            }
        } else {
            auto x = (const float *)((const char *)mask->data + mask->nb[1]*i1 + mask->nb[2]*i2 + mask->nb[3]*i3);
            auto y = (float *)((char *)dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
            if (i1 < topk->ne[1]) {
                for (int j = 0; j < n; ++j) y[j] = -INFINITY;
                for (int j = 0; j < nidx; ++j) y[idx[j]] = 0.0f;
                for (int j = 0; j < n; ++j) y[j] += x[j];
            } else {
                for (int j = 0; j < n; ++j) y[j] = x[j];
            }
        }
    }
}
