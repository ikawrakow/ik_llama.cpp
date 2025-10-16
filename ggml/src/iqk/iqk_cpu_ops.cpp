#include "iqk_cpu_ops.h"
#include "ggml.h"

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>

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

    size_t work_size = n_per_group + n_groups;
    work_size = std::max(work_size, size_t(n_per_group*n_top_groups));
    auto& aux = get_work_buffer(n_per_group + n_groups);

    auto groups = aux.data() + n_per_group;

    for (int ir = first; ir < last; ++ir) {
        auto data = (const float *)((const char *)src->data + ir*src->nb[1]);
        auto result = (int32_t *)((char *)dst->data + ir*dst->nb[1]);
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
        std::partial_sort(aux.begin(), aux.begin() + ne0, aux.end(), std::greater<std::pair<float,int>>{});
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

