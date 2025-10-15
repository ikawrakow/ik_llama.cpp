//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
#include "argsort.cuh"

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

template<ggml_sort_order order, typename Store>
static __global__ void k_argsort_f32_i32(const float * x, int * dst, const int ncols, int ncols_pad, Store s) {
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
        if (col < ncols) {
            dst[row * ncols + col] = col < s.min_experts || x_row[dst_row[col]] >= s.thresh_experts*max_val ? dst_row[col] : -1;
        }
    } else {
        if (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
    //if (min_experts >= 0 && min_experts < ncols && thresh_experts > 0) {
    //    __syncthreads();
    //    float max_val = x_row[dst_row[0]];
    //    if (col < ncols) {
    //        dst[row * ncols + col] = col < min_experts || x_row[dst_row[col]] >= thresh_experts*max_val ? dst_row[col] : -1;
    //    }
    //}
    //else {
    //    // copy the result to dst without the padding
    //    if (col < ncols) {
    //        dst[row * ncols + col] = dst_row[col];
    //    }
    //}
}

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static void argsort_f32_i32_cuda(const float * x, int * dst, const int ncols, const int nrows,
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
            k_argsort_f32_i32<GGML_SORT_ORDER_ASC, store_ser><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad,
                    {min_experts, thresh_experts});
        } else {
            k_argsort_f32_i32<GGML_SORT_ORDER_ASC, store><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, {});
        }
        //k_argsort_f32_i32<GGML_SORT_ORDER_ASC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, min_experts, thresh_experts);
    } else if (order == GGML_SORT_ORDER_DESC) {
        if (min_experts >= 0 && min_experts < ncols && thresh_experts > 0) {
            k_argsort_f32_i32<GGML_SORT_ORDER_DESC, store_ser><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad,
                    {min_experts, thresh_experts});
        } else {
            k_argsort_f32_i32<GGML_SORT_ORDER_DESC, store><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, {});
        }
        //k_argsort_f32_i32<GGML_SORT_ORDER_DESC><<<block_nums, block_dims, shared_mem, stream>>>(x, dst, ncols, ncols_pad, min_experts, thresh_experts);
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

    argsort_f32_i32_cuda(src0_d, (int *)dst_d, ncols, nrows, order, -1, 0.f, stream);
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

    argsort_f32_i32_cuda(src0_d, (int *)dst_d, ncols, nrows, GGML_SORT_ORDER_DESC, min_experts, thresh, stream);
}
