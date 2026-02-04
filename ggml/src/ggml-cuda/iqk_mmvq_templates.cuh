//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once
#include "iqk_mmvq.cuh"
#include "iqk_cuda_common.h"
#include "mmvq-args.h"

typedef void (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float *);

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_M_R4> {
    static constexpr int qk = 32;
    static constexpr int qr = 2;
    static constexpr int qi = 4;
};

template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y, int n_interleaved = 1>
static __device__ void iqk_mul_mat_vec_q_kerne(
    const void * __restrict__ vx, const void * __restrict__ vy,
    const float * bias, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst, const int64_t row_size) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int nwarps              = 1;
    constexpr int rows_per_cuda_block = n_interleaved;
#else
    constexpr int nwarps              = n_interleaved == 1 ? ncols_y <= 4 ? 4 : 2 : 1;
    constexpr int rows_per_cuda_block = n_interleaved == 1 ? ncols_y == 1 ? 1 : 2 : n_interleaved;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

// partial sum for each thread
    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
            if constexpr (n_interleaved == 1) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    vec_dot_q_cuda((const void *)((const char *)vx + (row0 + i)*row_size),
                            &y[j*blocks_per_col_y + kby], kbx, kqs, &tmp[j][i]);
                }
            } else {
                vec_dot_q_cuda((const void *)((const char *)vx + row0*row_size),
                    &y[j*blocks_per_col_y + kby], kbx, kqs, tmp[j]);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < nrows_dst)) {
            dst[j*nrows_dst + row0 + threadIdx.x] = bias ? tmp[j][threadIdx.x] + bias[row0 + threadIdx.x] : tmp[j][threadIdx.x];
        }
    }
}

template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y, int n_interleaved = 1>
static __device__ void iqk_fused_mul_mat_vec_q_kernel(
    const void * __restrict__ vup, const void * __restrict__ vgate, const void * __restrict__ vy, float * __restrict__ dst,
    const float * __restrict__ bias_u, const float * __restrict__ bias_g,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst, const int64_t row_size,
    ggml_unary_op unary_op, float limit) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int nwarps              = 1;
    constexpr int rows_per_cuda_block = n_interleaved;
#else
    constexpr int nwarps              = n_interleaved == 1 ? ncols_y <= 4 ? 4 : 2 : 1;
    constexpr int rows_per_cuda_block = n_interleaved == 1 ? ncols_y == 1 ? 1 : 2 : n_interleaved;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

// partial sum for each thread
    float tmp_u[ncols_y][rows_per_cuda_block] = {0.0f};
    float tmp_g[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
            if constexpr (n_interleaved == 1) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    vec_dot_q_cuda((const void *)((const char *)vup + (row0 + i)*row_size),
                            &y[j*blocks_per_col_y + kby], kbx, kqs, &tmp_u[j][i]);
                    vec_dot_q_cuda((const void *)((const char *)vgate + (row0 + i)*row_size),
                            &y[j*blocks_per_col_y + kby], kbx, kqs, &tmp_g[j][i]);
                }
            } else {
                vec_dot_q_cuda((const void *)((const char *)vup + row0*row_size),
                    &y[j*blocks_per_col_y + kby], kbx, kqs, tmp_u[j]);
                vec_dot_q_cuda((const void *)((const char *)vgate + row0*row_size),
                    &y[j*blocks_per_col_y + kby], kbx, kqs, tmp_g[j]);
            }
        }
    }

    __shared__ float tmp_shared_u[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    __shared__ float tmp_shared_g[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared_u[threadIdx.y-1][j][i][threadIdx.x] = tmp_u[j][i];
                tmp_shared_g[threadIdx.y-1][j][i][threadIdx.x] = tmp_g[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp_u[j][i] += tmp_shared_u[l][j][i][threadIdx.x];
                tmp_g[j][i] += tmp_shared_g[l][j][i][threadIdx.x];
            }
            tmp_u[j][i] = warp_reduce_sum(tmp_u[j][i]);
            tmp_g[j][i] = warp_reduce_sum(tmp_g[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < nrows_dst)) {
            float u = tmp_u[j][threadIdx.x];
            float g = tmp_g[j][threadIdx.x];
            float r;
            switch (unary_op) {
                case GGML_UNARY_OP_SILU:
                    {
                        g = g/(1 + expf(-g));
                        g = min(g, limit);
                        r = max(-limit, min(limit, u))*g;
                    } break;
                case GGML_UNARY_OP_RELU: r = fmaxf(g, 0.0f) * u; break;
                case GGML_UNARY_OP_GELU: {
                    constexpr float GELU_COEF_A    = 0.044715f;
                    constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
                    r = 0.5f*g*u*(1.0f + tanhf(SQRT_2_OVER_PI*g*(1.0f + GELU_COEF_A*g*g)));
                } break;
                // we assume that the supported ops have been checked by the caller
                default: {
                    constexpr float alpha = 1.702f;
                    constexpr float limit = 7.0f;
                    g += bias_g[row0 + threadIdx.x];
                    u += bias_u[row0 + threadIdx.x];
                    g = fminf(g, limit);
                    u = fmaxf(fminf(u, limit), -limit);
                    r = g / (1.0f + expf(-g * alpha)) * (1.0f + u);
                } break;
            }
            dst[j*nrows_dst + row0 + threadIdx.x] = r;
        }
    }
}

template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y, int n_interleaved = 1>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__((ncols_y <= 4 ? 4 : 2)*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void iqk_mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const char * __restrict__ ids_data, const void * __restrict__ bias,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst, const int64_t row_size,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, const int64_t bias_nb1) {
    int i2 = blockIdx.y;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) return;
    const char * cx = (const char *)vx + i02*nb02;
    const char * cy = (const char *)vy + i2*nb12;
    char * cdst = (char *)dst + i2*nb2;
    const float * b = (const float *)(bias ? ids_data ? (const char *)bias + i02*bias_nb1 : bias : nullptr);
    iqk_mul_mat_vec_q_kerne<type, vdr, vec_dot_q_cuda, ncols_y, n_interleaved>(cx, cy, b, (float *)cdst, ncols_x, nrows_x, nrows_y, nrows_dst, row_size);
}

template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y, int n_interleaved = 1>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__((ncols_y <= 4 ? 4 : 2)*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void iqk_fused_mul_mat_vec_q(
    const void * __restrict__ vx_u, const void * __restrict__ vx_g, const void * __restrict__ vy, float * __restrict__ dst,
    const char * __restrict__ ids_data, const void * __restrict__ bias_u, const void * __restrict__ bias_g, const uint64_t bias_nb1,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst, const int64_t row_size,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, ggml_unary_op unary_op, float limit) {

    int i2 = blockIdx.y;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) return;
    const char * cx_u = (const char *)vx_u + i02*nb02;
    const char * cx_g = (const char *)vx_g + i02*nb02;
    const char * cy = (const char *)vy + i2*nb12;
    const float * cx_u_b = bias_u ? (const float *)((const char *)bias_u + i02*bias_nb1) : nullptr;
    const float * cx_g_b = bias_g ? (const float *)((const char *)bias_g + i02*bias_nb1) : nullptr;
    char * cdst = (char *)dst + i2*nb2;
    iqk_fused_mul_mat_vec_q_kernel<type, vdr, vec_dot_q_cuda, ncols_y, n_interleaved>(
            cx_u, cx_g, cy, (float *)cdst, cx_u_b, cx_g_b,
            ncols_x, nrows_x, nrows_y, nrows_dst, row_size, unary_op, limit);
}

template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int n_interleaved = 1>
static void iqk_mul_mat_vec_q_cuda(const mmvq_args & args, cudaStream_t stream) {

    GGML_ASSERT(args.ncols_x % ggml_blck_size(type) == 0);
    //GGML_ASSERT(ncols_y <= MMVQ_MAX_BATCH_SIZE);

    int id = ggml_cuda_get_device();

    int64_t nwarps = 1;
    int64_t rows_per_cuda_block = n_interleaved;

    if (ggml_cuda_info().devices[id].cc < CC_RDNA2) { // NVIDIA and AMD older than RDNA2
        switch(args.ncols_y) {
            case 1:
                nwarps = n_interleaved == 1 ? 4 : 1;
                rows_per_cuda_block = n_interleaved == 1 ? 1 : n_interleaved;
                break;
            case 2:
            case 3:
            case 4:
                nwarps = n_interleaved == 1 ? 4 : 1;
                rows_per_cuda_block = n_interleaved == 1 ? 2 : n_interleaved;
                break;
            case 5:
            case 6:
            case 7:
            case 8:
                nwarps = n_interleaved == 1 ? 2 : 1;
                rows_per_cuda_block = n_interleaved == 1 ? 2 : n_interleaved;
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
    }
    const int64_t nblocks = (args.nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, args.ne2, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    const int64_t row_size = ggml_row_size(type, args.ncols_x);

    if (args.vx_u && args.vx_g && args.unary_op != GGML_UNARY_OP_COUNT) {
    switch (args.ncols_y) {
        case 1:
            iqk_fused_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 1, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vx_g, args.vy, args.dst,
                    args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size,
                    args.nb02, args.nb12, args.nb2, args.ids_nb0, args.unary_op, args.limit);
            break;
        case 2:
            iqk_fused_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 2, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vx_g, args.vy, args.dst,
                    args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size,
                    args.nb02, args.nb12, args.nb2, args.ids_nb0, args.unary_op, args.limit);
            break;
        case 3:
            iqk_fused_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 3, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vx_g, args.vy, args.dst,
                    args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size,
                    args.nb02, args.nb12, args.nb2, args.ids_nb0, args.unary_op, args.limit);
            break;
        case 4:
            iqk_fused_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 4, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vx_g, args.vy, args.dst,
                    args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size,
                    args.nb02, args.nb12, args.nb2, args.ids_nb0, args.unary_op, args.limit);
            break;
        case 5:
            iqk_fused_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 5, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vx_g, args.vy, args.dst,
                    args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size,
                    args.nb02, args.nb12, args.nb2, args.ids_nb0, args.unary_op, args.limit);
            break;
        case 6:
            iqk_fused_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 6, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vx_g, args.vy, args.dst,
                    args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size,
                    args.nb02, args.nb12, args.nb2, args.ids_nb0, args.unary_op, args.limit);
            break;
        case 7:
            iqk_fused_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 7, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vx_g, args.vy, args.dst,
                    args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size,
                    args.nb02, args.nb12, args.nb2, args.ids_nb0, args.unary_op, args.limit);
            break;
        case 8:
            iqk_fused_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 8, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vx_g, args.vy, args.dst,
                    args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size,
                    args.nb02, args.nb12, args.nb2, args.ids_nb0, args.unary_op, args.limit);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }
    } else {
    switch (args.ncols_y) {
        case 1:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 1, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst,
                    row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 2:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 2, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst,
                    row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 3:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 3, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst,
                    row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 4:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 4, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst,
                    row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 5:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 5, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst,
                    row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 6:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 6, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst,
                    row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 7:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 7, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst,
                    row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 8:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 8, n_interleaved><<<block_nums, block_dims, 0, stream>>>(
                    args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst,
                    row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }
    }
}

static __device__ __forceinline__ void get_int_from_table_16_shift(const uint32_t & q4, uint16_t shift, const uint8_t * all_values,
        int & val1, int & val2) {

    uint32_t aux32; const uint8_t * q8 = (const uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    const uint8_t * values = all_values + 16*(shift & 1);
    uint16_t v1 = values[q8[0]] | (values[q8[1]] << 8);
    uint16_t v2 = values[q8[2]] | (values[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    values = all_values + 8*(shift & 2);
    v1 = values[q8[0]] | (values[q8[1]] << 8);
    v2 = values[q8[2]] | (values[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * values) {
#if defined(__CUDA_ARCH__)
    uint32_t v1, v2, v3, v4, mask;
    const uint32_t * values32 = (const uint32_t *)values;

    mask = (0x32103210 | ((q4 & 0x88888888) >> 1));
    // Perform lookups in the lower half of the table (indices 0-7).
    v1 = __byte_perm(values32[0], values32[1], q4);
    // Perform lookups in the upper half of the table (indices 8-15).
    v2 = __byte_perm(values32[2], values32[3], q4);
    // Select between the low and high results based on the MSB of each index nibble.
    v3 = __byte_perm(v1, v2, mask);
    // Same for the upper part of q4.
    v1 = __byte_perm(values32[0], values32[1], q4 >> 16);
    v2 = __byte_perm(values32[2], values32[3], q4 >> 16);
    v4 = __byte_perm(v1, v2, mask >> 16);

    // Mix the results to get the final int2.
    return make_int2(__byte_perm(v3, v4, 0x6420), __byte_perm(v3, v4, 0x7531));
#else
    const int      q0_32  = (q4 >> 0) & 0x0F0F0F0F;
    const int8_t * q0_8   = (const int8_t *) &q0_32;
    const char4    val0_8 = make_char4(values[q0_8[0]], values[q0_8[1]], values[q0_8[2]], values[q0_8[3]]);

    const int      q1_32  = (q4 >> 4) & 0x0F0F0F0F;
    const int8_t * q1_8   = (const int8_t *) &q1_32;
    const char4    val1_8 = make_char4(values[q1_8[0]], values[q1_8[1]], values[q1_8[2]], values[q1_8[3]]);

    return make_int2(*((const int *) &val0_8), *((const int *) &val1_8));
#endif
}

static __device__ __forceinline__ int int_from_table(const uint8_t * a8, const uint8_t * values) {
    uint16_t v1 = values[a8[0]] | (values[a8[1]] << 8);
    uint16_t v2 = values[a8[2]] | (values[a8[3]] << 8);
    return v1 | (v2 << 16);
}

#define VDR_IQ4_K_Q8_1_MMVQ 4
#define VDR_IQ4_K_Q8_1_MMQ  4

#define VDR_IQ4_KS_Q8_1_MMVQ 4
#define VDR_IQ4_KS_Q8_1_MMQ  4

#define VDR_IQ4_KSS_Q8_1_MMVQ 4
#define VDR_IQ4_KSS_Q8_1_MMQ  4

#define VDR_IQ5_K_Q8_1_MMVQ 4
#define VDR_IQ5_K_Q8_1_MMQ  4

#define VDR_IQ6_K_Q8_1_MMVQ 4
#define VDR_IQ6_K_Q8_1_MMQ  4

#define VDR_IQ2_K_Q8_1_MMVQ 4
#define VDR_IQ2_K_Q8_1_MMQ  4

#define VDR_IQ2_KS_Q8_1_MMVQ 4
#define VDR_IQ2_KS_Q8_1_MMQ  4

#define VDR_IQ3_K_Q8_1_MMVQ 4
#define VDR_IQ3_K_Q8_1_MMQ  4

extern void mul_mat_vec_iq2_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq3_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq4_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq5_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq6_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2_kl_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq3_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq4_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq5_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq1_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq3_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq4_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq3_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq4_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq5_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq1_s_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq1_m_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq4_ks_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq5_ks_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq1_bn_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2_bn_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq4_kss_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
