//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
#include "argsort.cuh"
#include "sumrows.cuh"

template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

struct store_ser {
    constexpr static bool has_thresh = true;
    int   min_experts;
    float thresh_experts;
    store_ser(int min, float thresh) : min_experts(min), thresh_experts(thresh) {}
};

struct store {
    constexpr static bool has_thresh = false;
};

template<ggml_sort_order order, typename Store, typename dst_t>
static __global__ void k_argsort_f32_T(const float * x, dst_t * dst, const int ncols, int ncols_pad, int ntop, Store s) {
//        int min_experts, float thresh_experts) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.y;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }

    if constexpr (Store::has_thresh) {
        __syncthreads();
        float max_val = x_row[dst_row[0]];
        if (col < ntop) {
            if constexpr (std::is_same_v<dst_t, int>) {
                dst[row * ntop + col] = col < s.min_experts || x_row[dst_row[col]] >= s.thresh_experts*max_val ? dst_row[col] : -1;
            } else {
                dst[row * ntop + col] = col < s.min_experts || x_row[dst_row[col]] >= s.thresh_experts*max_val ? x_row[dst_row[col]] : 0.f;
            }
        }
    } else {
        if (col < ntop) {
            if constexpr (std::is_same_v<dst_t, int>) {
                dst[row * ntop + col] = dst_row[col];
            } else {
                dst[row * ntop + col] = x_row[dst_row[col]];
            }
        }
    }
}

#if 0
// Somehow this is not working. Someone sees the bug?
template<ggml_sort_order order>
static __global__ void k_topk_sum(const float * x, float * dst, const int ncols, int ncols_pad, int n_top_k) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.y;

    if (col >= ncols_pad) {
        return;
    }

    const float * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }

    float val = col < n_top_k ? x[dst_row[col]] : 0;
    val = warp_reduce_sum(val);
    if (blockDim.x > WARP_SIZE) {
        __syncthreads();
        auto s_sum = dst_row;
        const int        warp_id = threadIdx.x / WARP_SIZE;
        const int        lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = val;
        }
        __syncthreads();
        val = 0.0f;
        if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
            val = s_sum[lane_id];
        }
        val = warp_reduce_sum(val);
    }

    if (col == 0) {
        dst[row] = val;
    }
}
#endif

static __global__ void k_apply_mask(float * dst, const int * groups,
        const int n_top_groups, const int n_per_group, const int ncols) {
    int row = blockIdx.y;
    for (int col = threadIdx.x; col < n_top_groups*n_per_group; col += blockDim.x) {
        int ig = groups[row*n_top_groups + col / n_per_group];
        int ic = col % n_per_group;
        dst[row*ncols + ig*n_per_group + ic] = -INFINITY;
    }
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

template <typename dst_t>
static void argsort_f32_T_cuda(const float * x, dst_t * dst, const int ncols, const int nrows, int ntop,
        ggml_sort_order order, int min_experts, float thresh_experts, cudaStream_t stream) {
    // bitonic sort requires ncols to be power of 2
    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(1, nrows, 1);
    const size_t shared_mem = ncols_pad * sizeof(int);

    // FIXME: this limit could be raised by ~2-4x on Ampere or newer
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    if (order == GGML_SORT_ORDER_ASC) {
        if (min_experts >= 0 && min_experts < ncols && thresh_experts > 0) {
            k_argsort_f32_T<GGML_SORT_ORDER_ASC, store_ser><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad,
                    ntop, {min_experts, thresh_experts});
        } else {
            k_argsort_f32_T<GGML_SORT_ORDER_ASC, store><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, ntop, {});
        }
    } else if (order == GGML_SORT_ORDER_DESC) {
        if (min_experts >= 0 && min_experts < ncols && thresh_experts > 0) {
            k_argsort_f32_T<GGML_SORT_ORDER_DESC, store_ser><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad,
                    ntop, {min_experts, thresh_experts});
        } else {
            k_argsort_f32_T<GGML_SORT_ORDER_DESC, store><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, ntop, {});
        }
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

    argsort_f32_T_cuda(src0_d, (int *)dst_d, ncols, nrows, ncols, order, -1, 0.f, stream);
}

void ggml_cuda_op_argsort_thresh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    int min_experts = dst->op_params[0];
    float thresh;
    memcpy(&thresh, dst->op_params + 1, sizeof(float));

    argsort_f32_T_cuda(src0_d, (int *)dst_d, ncols, nrows, ncols, GGML_SORT_ORDER_DESC, min_experts, thresh, stream);
}

#if 0
static void ggml_cuda_op_topk_sum(ggml_backend_cuda_context & ctx, const float * src, float * dst, int ncols, int nrows, int n_top_k) {

    GGML_ASSERT(n_top_k <= ncols);

    const int ncols_pad = next_power_of_2(ncols);

    const dim3 block_dims(ncols_pad, 1, 1);
    const dim3 block_nums(1, nrows, 1);
    const size_t shared_mem = std::max(ncols_pad, WARP_SIZE) * sizeof(int);
    GGML_ASSERT(shared_mem <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    k_topk_sum<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, ctx.stream()>>>(src, dst, ncols, ncols_pad, n_top_k);
}
#endif

void ggml_cuda_op_grouped_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nrows(src) == ggml_nrows(dst));

    auto nrows = ggml_nrows(src);

    int n_groups     = dst->op_params[0];
    int n_top_groups = dst->op_params[1];
    int nk           = dst->op_params[2];

    int ne00 = src->ne[0];
    int ne0  = dst->ne[0];
    GGML_ASSERT(ne0 <= ne00);
    GGML_ASSERT(ne00%n_groups == 0);
    int n_per_group = ne00/n_groups;
    GGML_ASSERT(nk <= n_per_group);
    GGML_ASSERT(n_top_groups < n_groups);
    int n_discarded_groups = n_groups - n_top_groups;

    ggml_cuda_pool_alloc<float> sorted_group_scores(ctx.pool(), nk*nrows*n_groups);
    argsort_f32_T_cuda((const float *)src->data, sorted_group_scores.get(), n_per_group, nrows*n_groups, nk,
            GGML_SORT_ORDER_DESC, -1, 0.0f, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_pool_alloc<float> group_scores(ctx.pool(), nrows*n_groups);
    sum_rows_f32_cuda((const float *)sorted_group_scores.get(), group_scores.get(), nk, nrows*n_groups, ctx.stream());
    CUDA_CHECK(cudaGetLastError());

    // This is not working for some reason, so we resort to the slightly less efficient implementation above
    //ggml_cuda_pool_alloc<float> group_scores(ctx.pool(), nrows*n_groups);
    //ggml_cuda_op_topk_sum(ctx, (const float *)src->data, group_scores.get(), n_per_group, nrows*n_groups, nk);
    ////sum_rows_f32_cuda((const float *)src->data, group_scores.get(), n_per_group, nrows*n_groups, ctx.stream());
    //CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<int> discarded_groups(ctx.pool(), nrows*n_discarded_groups);
    argsort_f32_T_cuda(group_scores.get(), discarded_groups.get(), n_groups, nrows, n_discarded_groups, GGML_SORT_ORDER_ASC, -1, 0.0f, ctx.stream());
    CUDA_CHECK(cudaGetLastError());

    {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const dim3 block_nums(1, nrows, 1);
        cudaStream_t stream = ctx.stream();
        k_apply_mask<<<block_nums, block_dims, 0, ctx.stream()>>>((float *)src->data, discarded_groups.get(), n_discarded_groups, n_per_group, ne00);
        CUDA_CHECK(cudaGetLastError());
    }

    argsort_f32_T_cuda((const float *)src->data, (int *)dst->data, ne00, nrows, ne0, GGML_SORT_ORDER_DESC, -1, 0.0f, ctx.stream());

}
