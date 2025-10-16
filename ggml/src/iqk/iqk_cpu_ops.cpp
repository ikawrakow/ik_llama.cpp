#include "iqk_cpu_ops.h"
#include "ggml.h"

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>

namespace {
inline float group_score(int n_per_group, const float * data) {
    float sum = 0;
    for (int j = 0; j < n_per_group; ++j) sum += data[j];
    return sum;
}
inline float group_score(int n_per_group, int nk, const float * data, float * aux) {
    for (int j = 0; j < n_per_group; ++j) aux[j] = data[j];
    std::partial_sort(aux, aux + nk, aux + n_per_group, std::greater<float>{});
    float sum = 0;
    for (int j = 0; j < nk; ++j) sum += aux[j];
    return sum;
}
inline float group_score_max(int n_per_group, const float * data) {
    float max = data[0];
    for (int j = 1; j < n_per_group; ++j) max = std::max(max, data[j]);
    return max;
}
}

void iqk_grouped_top_k([[maybe_unused]] ggml_tensor * dst, [[maybe_unused]] int ith, [[maybe_unused]] int nth) {
    auto src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src, dst));

    auto nrows = ggml_nrows(src);
    auto npt   = (nrows + nth - 1)/nth;
    auto first = npt*ith;
    auto last  = std::min(first + npt, nrows);
    if (last <= first) return;

    int n_groups     = dst->op_params[0];
    int n_top_groups = dst->op_params[1];
    int nk           = dst->op_params[2];

    //if (ith == 0) printf("%s: ne00 = %ld, n_groups = %d, n_top_groups = %d, nk = %d\n", __func__, src->ne[0], n_groups, n_top_groups, nk);

    int ne00 = src->ne[0];
    GGML_ASSERT(ne00%n_groups == 0);
    int n_per_group = ne00/n_groups;
    GGML_ASSERT(nk <= n_per_group);

    thread_local std::vector<std::pair<float,int>> aux;
    if ((int)aux.size() < n_per_group + n_groups) aux.resize(n_per_group + n_groups);

    auto groups = aux.data() + n_per_group;

    for (int ir = first; ir < last; ++ir) {
        auto data = (const float *)((const char *)src->data + ir*src->nb[1]);
        auto result = (float *)((char *)dst->data + ir*dst->nb[1]);
        for (int ig = 0; ig < n_groups; ++ig) {
            //groups[ig] = { group_score(n_per_group, data + ig*n_per_group), ig };
            groups[ig] = { group_score(n_per_group, nk, data + ig*n_per_group, (float *)aux.data()), ig };
            //groups[ig] = { group_score_max(n_per_group, data + ig*n_per_group), ig };
        }
        std::partial_sort(groups, groups + n_top_groups, groups + n_groups, std::greater<std::pair<float,int>>{});

        for (int ig = 0; ig < n_top_groups; ++ig) {
            int jg = groups[ig].second;
            for (int j = 0; j < n_per_group; ++j) result[jg*n_per_group + j] = data[jg*n_per_group + j];
        }
        for (int ig = n_top_groups; ig < n_groups; ++ig) {
            int jg = groups[ig].second;
            for (int j = 0; j < n_per_group; ++j) result[jg*n_per_group + j] = -INFINITY;
        }

    }

}

void iqk_grouped_top_k_orig([[maybe_unused]] ggml_tensor * dst, [[maybe_unused]] int ith, [[maybe_unused]] int nth) {
    auto src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src, dst));

    auto nrows = ggml_nrows(src);
    auto npt   = (nrows + nth - 1)/nth;
    auto first = npt*ith;
    auto last  = std::min(first + npt, nrows);
    if (last <= first) return;

    int n_groups     = dst->op_params[0];
    int n_top_groups = dst->op_params[1];
    int nk           = dst->op_params[2];

    //if (ith == 0) printf("%s: ne00 = %ld, n_groups = %d, n_top_groups = %d, nk = %d\n", __func__, src->ne[0], n_groups, n_top_groups, nk);

    int ne00 = src->ne[0];
    GGML_ASSERT(ne00%n_groups == 0);
    int n_per_group = ne00/n_groups;
    GGML_ASSERT(nk <= n_per_group);

    thread_local std::vector<std::pair<float,int>> aux;
    if ((int)aux.size() < n_per_group + n_groups) aux.resize(n_per_group + n_groups);

    auto groups = aux.data() + n_per_group;

    for (int ir = first; ir < last; ++ir) {
        auto data = (const float *)((const char *)src->data + ir*src->nb[1]);
        auto result = (float *)((char *)dst->data + ir*dst->nb[1]);
        for (int j = 0; j < ne00; ++j) result[j] = -INFINITY;
        for (int ig = 0; ig < n_groups; ++ig) {
            for (int j = 0; j < n_per_group; ++j) {
                int jj = ig*n_per_group + j;
                aux[j] = { data[jj], jj };
            }
            std::partial_sort(aux.begin(), aux.begin() + nk, aux.end(), std::greater<std::pair<float,int>>{});
            for (int j = 0; j < nk; ++j) result[aux[j].second] = data[aux[j].second];
            //float sum = 0;
            //for (int j = 0; j < nk; ++j) sum += aux[j].first;
            //groups[ig] = { sum, ig };
        }
        //std::partial_sort(groups, groups + n_top_groups, groups + n_groups, std::greater<std::pair<float,int>>{});

        //for (int ig = 0; ig < n_top_groups; ++ig) {
        //    int jg = groups[ig].second;
        //    for (int j = 0; j < n_per_group; ++j) result[jg*n_per_group + j] = data[jg*n_per_group + j];
        //}
        //for (int ig = n_top_groups; ig < n_groups; ++ig) {
        //    int jg = groups[ig].second;
        //    for (int j = 0; j < n_per_group; ++j) result[jg*n_per_group + j] = -INFINITY;
        //}

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
    thread_local std::vector<std::pair<float,int>> aux;
    if ((int)aux.size() < ne00) aux.resize(ne00);
    //std::vector<std::pair<float,int>> aux(ne00);

    for (int ir = first; ir < last; ++ir) {
        auto data = (const float *)((const char *)src->data + ir*src->nb[1]);
        for (int j = 0; j < ne00; ++j) aux[j] = {data[j], j};
        if (nk < ne00) {
            if (order == GGML_SORT_ORDER_DESC) {
                std::partial_sort(aux.begin(), aux.begin() + nk, aux.end(), std::greater<std::pair<float,int>>{});
            } else {
                std::partial_sort(aux.begin(), aux.begin() + nk, aux.end());
            }
        } else {
            if (order == GGML_SORT_ORDER_DESC) {
                std::sort(aux.begin(), aux.end(), std::greater<std::pair<float,int>>{});
            } else {
                std::sort(aux.begin(), aux.end());
            }
        }
        auto y = (int32_t *)((char *)dst->data + ir*dst->nb[1]);
        for (int j = 0; j < ne00; ++j) y[j] = aux[j].second;
    }

}

