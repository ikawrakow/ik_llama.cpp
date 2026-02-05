//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once
#include "mmvq.cuh"
#include "iqk_mmvq.cuh"
#include "vecdotq.cuh"
#include "mmvq-args.h"

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0   : return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1   : return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0   : return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1   : return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q6_0   : return vec_dot_q6_0_q8_1;
        case GGML_TYPE_Q8_0   : return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q2_K   : return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K   : return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K   : return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K   : return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K   : return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS : return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S  : return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S  : return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M  : return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL : return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_MXFP4  : return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_IQ4_XS : return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S  : return vec_dot_iq3_s_q8_1;
        default               : return nullptr;
    }
}

static constexpr __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0    : return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1    : return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0    : return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1    : return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q6_0    : return VDR_Q6_0_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0    : return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K    : return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K    : return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K    : return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K    : return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K    : return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS : return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS  : return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S   : return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS : return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S   : return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL  : return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4   : return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS  : return VDR_IQ4_XS_Q8_1_MMVQ;
        default                : return 1;
    }
}

template <ggml_type type, int ncols_y, int nwarps>
static __device__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy,
    const float * bias, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    //int64_t rows_per_cuda_block = ggml_cuda_info().devices[id].cc < CC_RDNA2 ?
    //    ncols_y < 4 ? 1 : 2 : 1;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int rows_per_cuda_block = 1;
#else
    constexpr int rows_per_cuda_block = ncols_y < 4 ? 1 : 2;
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
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(vx, &y[j*blocks_per_col_y + kby], (row0 + i)*blocks_per_row_x + kbx, kqs);
            }
        }
    }

    float local_bias[rows_per_cuda_block]    = { 0.0f };
    if (bias && threadIdx.y == 0 && threadIdx.x < rows_per_cuda_block && row0 + threadIdx.x < nrows_dst) {
        local_bias[threadIdx.x] = bias[row0 + threadIdx.x];
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
            dst[j*nrows_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x] + local_bias[threadIdx.x];
        }
    }
}

template <ggml_type type, int ncols_y, int nwarps>
static __device__ void fused_mul_mat_vec_q(
    const void * __restrict__ vup, const void * __restrict__ vgate,
    const float * __restrict__ bias_u, const float * __restrict__ bias_g,
    const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst, ggml_unary_op unary_op, float limit) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    //int64_t rows_per_cuda_block = ggml_cuda_info().devices[id].cc < CC_RDNA2 ?
    //    ncols_y < 4 ? 1 : 2 : 1;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int rows_per_cuda_block = 1;
#else
    constexpr int rows_per_cuda_block = ncols_y < 4 ? 1 : 2;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

// partial sum for each thread
    float tmp_u[ncols_y][rows_per_cuda_block] = {0.0f};
    float tmp_g[ncols_y][rows_per_cuda_block] = {0.0f};
    float local_bias_u[rows_per_cuda_block]    = { 0.0f };
    float local_bias_g[rows_per_cuda_block]    = { 0.0f };
    if (bias_u && threadIdx.y == 0 && threadIdx.x < rows_per_cuda_block && row0 + threadIdx.x < nrows_dst) {
        local_bias_u[threadIdx.x] = bias_u[row0 + threadIdx.x];
    }
    if (bias_g && threadIdx.y == 0 && threadIdx.x < rows_per_cuda_block && row0 + threadIdx.x < nrows_dst) {
        local_bias_g[threadIdx.x] = bias_g[row0 + threadIdx.x];
    }

    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_u[j][i] += vec_dot_q_cuda(vup  , &y[j*blocks_per_col_y + kby], (row0 + i)*blocks_per_row_x + kbx, kqs);
                tmp_g[j][i] += vec_dot_q_cuda(vgate, &y[j*blocks_per_col_y + kby], (row0 + i)*blocks_per_row_x + kbx, kqs);
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
                    }break;
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
                  g += local_bias_g[threadIdx.x];
                  u += local_bias_u[threadIdx.x];
                  g = fminf(g, limit);
                  u = fmaxf(fminf(u, limit), -limit);
                  r = g / (1.0f + expf(-g * alpha)) * (1.0f + u);
                } break;
            }
            dst[j*nrows_dst + row0 + threadIdx.x] = r;
        }
    }
}

template <ggml_type type, int ncols_y, int nwarps>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const char * __restrict__ ids_data, const void * __restrict__ bias,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, const int64_t bias_nb1) {

    int i2 = blockIdx.y;
    char * cdst = (char *)dst + i2*nb2;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) {
        return;
    }
    const char * cx = (const char *)vx + i02*nb02;
    const char * cy = (const char *)vy + i2*nb12;
    const float * b = (const float *)(bias ? ids_data ? (const char *)bias + i02*bias_nb1 : bias : nullptr);
    mul_mat_vec_q<type, ncols_y, nwarps>(cx, cy, b, (float *)cdst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

template <ggml_type type, int ncols_y, int nwarps>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void fused_mul_mat_vec_q(
    const void * __restrict__ vup, const void * __restrict__ vgate,
    const void * __restrict__ vy, float * __restrict__ dst, const char * __restrict__ ids_data,
    const void * __restrict__ bias_u, const void * __restrict__ bias_g, const uint64_t bias_nb1,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, ggml_unary_op unary_op, float limit) {

    int i2 = blockIdx.y;
    char * cdst = (char *)dst + i2*nb2;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) {
        return;
    }
    const char * cx_u = (const char *)vup   + i02*nb02;
    const char * cx_g = (const char *)vgate + i02*nb02;
    const float * cx_u_b = bias_u ? (const float *)((const char *)bias_u + i02*bias_nb1) : nullptr;
    const float * cx_g_b = bias_g ? (const float *)((const char *)bias_g + i02*bias_nb1) : nullptr;
    const char * cy = (const char *)vy + i2*nb12;
    fused_mul_mat_vec_q<type, ncols_y, nwarps>(cx_u, cx_g, cx_u_b, cx_g_b, cy, (float *)cdst, ncols_x, nrows_x, nrows_y, nrows_dst,
            unary_op, limit);
}

template <ggml_type type, int nwarps>
static void mul_mat_vec_q_cuda_T(const mmvq_args & args, cudaStream_t stream) {

    GGML_ASSERT(args.ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(args.ncols_y <= MMVQ_MAX_BATCH_SIZE);

    int id = ggml_cuda_get_device();

    int64_t rows_per_cuda_block = ggml_cuda_info().devices[id].cc < CC_RDNA2 ?
        args.ncols_y < 4 ? 1 : 2 : 1;

    const int64_t nblocks = (args.nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, args.ne2, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    if (args.vx_u && args.vx_g && args.unary_op != GGML_UNARY_OP_COUNT) {
    switch (args.ncols_y) {
        case 1:
            fused_mul_mat_vec_q<type, 1, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 2:
            fused_mul_mat_vec_q<type, 2, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 3:
            fused_mul_mat_vec_q<type, 3, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 4:
            fused_mul_mat_vec_q<type, 4, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 5:
            fused_mul_mat_vec_q<type, 5, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 6:
            fused_mul_mat_vec_q<type, 6, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 7:
            fused_mul_mat_vec_q<type, 7, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        case 8:
            fused_mul_mat_vec_q<type, 8, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vx_g, args.vy,
                    args.dst, args.ids_data, args.bias_u, args.bias_g, args.bias_nb1,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0,
                    args.unary_op, args.limit);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
    } else {
    switch (args.ncols_y) {
        case 1:
            mul_mat_vec_q<type, 1, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 2:
            mul_mat_vec_q<type, 2, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 3:
            mul_mat_vec_q<type, 3, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 4:
            mul_mat_vec_q<type, 4, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 5:
            mul_mat_vec_q<type, 5, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 6:
            mul_mat_vec_q<type, 6, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 7:
            mul_mat_vec_q<type, 7, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        case 8:
            mul_mat_vec_q<type, 8, nwarps><<<block_nums, block_dims, 0, stream>>>(args.vx_u, args.vy, args.dst, args.ids_data, args.bias_u,
                    args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, args.nb02, args.nb12, args.nb2, args.ids_nb0, args.bias_nb1);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
    }
}

template <ggml_type type>
static void mul_mat_vec_q_cuda(const mmvq_args & args, cudaStream_t stream) {
    int nwarps = 1;
    int id = ggml_cuda_get_device();
    if (args.ne2 < 2 && ggml_cuda_info().devices[id].cc < CC_RDNA2) { // NVIDIA and AMD older than RDNA2
        nwarps = args.ncols_y <= 4 ? 4 : 2;
    }
    switch (nwarps) {
        case 1:
            mul_mat_vec_q_cuda_T<type, 1>(args, stream);
            break;
        case 2:
            mul_mat_vec_q_cuda_T<type, 2>(args, stream);
            break;
        default:
            mul_mat_vec_q_cuda_T<type, 4>(args, stream);
    }
}

extern void mul_mat_vec_q4_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q4_1_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q5_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q5_1_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q6_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q8_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q2_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q3_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q4_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q5_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q6_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2_xxs_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2_xs_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq2_s_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq3_xxs_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq1_s_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq1_m_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq4_nl_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_mxfp4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq4_xs_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_iq3_s_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);

