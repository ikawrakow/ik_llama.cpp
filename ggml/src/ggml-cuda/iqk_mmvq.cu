//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

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

//  Reminder:
//    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
//    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
//    constexpr int vdr = get_vdr_mmvq(type);

// QI4_XS = 256/(4*2) = 32
// vdr = 4, qi = 32 -> qi/vdr = 8, kqs = 4*(tid%8),  blocks_per_iter = 4*1*32/32 = 4
// vdr = 2, qi = 32 -> qi/vdr =16, kqs = 2*(tid%16), blocks_per_iter = 2*1*32/32 = 2
namespace {
template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y, int n_interleaved = 1>
__device__ void iqk_mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
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
            dst[j*nrows_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y, int n_interleaved = 1>
__device__ void iqk_fused_mul_mat_vec_q(
    const void * __restrict__ vup, const void * __restrict__ vgate, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst, const int64_t row_size,
    ggml_unary_op unary_op) {

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
                    &y[j*blocks_per_col_y + kby], kbx, kqs, tmp_u[j]);
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
                case GGML_UNARY_OP_SILU: r = u*g/(1 + expf(-g)); break;
                case GGML_UNARY_OP_RELU: r = fmaxf(g, 0.0f) * u; break;
                // we assume that the supported ops have been checked by the caller
                default: {
                    constexpr float GELU_COEF_A    = 0.044715f;
                    constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
                    r = 0.5f*g*u*(1.0f + tanhf(SQRT_2_OVER_PI*g*(1.0f + GELU_COEF_A*g*g)));
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
__global__ void iqk_mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst, const char * __restrict__ ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst, const int64_t row_size,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0) {
    int i2 = blockIdx.y;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) return;
    const char * cx = (const char *)vx + i02*nb02;
    const char * cy = (const char *)vy + i2*nb12;
    char * cdst = (char *)dst + i2*nb2;
    iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, ncols_y, n_interleaved>(cx, cy, (float *)cdst, ncols_x, nrows_x, nrows_y, nrows_dst, row_size);
}

template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int n_interleaved = 1>
void iqk_mul_mat_vec_q_cuda(const mmvq_args & args, cudaStream_t stream) {

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

    switch (args.ncols_y) {
        case 1:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 1, n_interleaved><<<block_nums, block_dims, 0, stream>>>(args.vx, args.vy, args.dst, args.ids_data, args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0);
            break;
        case 2:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 2, n_interleaved><<<block_nums, block_dims, 0, stream>>>(args.vx, args.vy, args.dst, args.ids_data, args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0);
            break;
        case 3:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 3, n_interleaved><<<block_nums, block_dims, 0, stream>>>(args.vx, args.vy, args.dst, args.ids_data, args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0);
            break;
        case 4:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 4, n_interleaved><<<block_nums, block_dims, 0, stream>>>(args.vx, args.vy, args.dst, args.ids_data, args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0);
            break;
        case 5:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 5, n_interleaved><<<block_nums, block_dims, 0, stream>>>(args.vx, args.vy, args.dst, args.ids_data, args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0);
            break;
        case 6:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 6, n_interleaved><<<block_nums, block_dims, 0, stream>>>(args.vx, args.vy, args.dst, args.ids_data, args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0);
            break;
        case 7:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 7, n_interleaved><<<block_nums, block_dims, 0, stream>>>(args.vx, args.vy, args.dst, args.ids_data, args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0);
            break;
        case 8:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 8, n_interleaved><<<block_nums, block_dims, 0, stream>>>(args.vx, args.vy, args.dst, args.ids_data, args.ncols_x, args.nrows_x, args.nrows_y, args.nrows_dst, row_size, args.nb02, args.nb12, args.nb2, args.ids_nb0);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }
}

__device__ __forceinline__ void get_int_from_table_16_shift(const uint32_t & q4, uint16_t shift, const uint8_t * all_values,
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

#define VDR_IQ4_K_Q8_1_MMVQ 4
#define VDR_IQ4_K_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq4_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq4_k * bq4 = (const block_iq4_k *) vbq + kbx;
    const uint8_t * all_values = (const uint8_t *)iq4k_values;

    // iqs is 0...28
    const int ib32 = iqs/4;
    // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint16_t * q4 = (const uint16_t *)bq4->qs + 8*ib32;
    const uint16_t extra = bq4->extra >> 2*ib32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        const uint32_t aux32 = q4[2*j+0] | (q4[2*j+1] << 16);
        get_int_from_table_16_shift(aux32, extra, all_values, v1, v2);
        sumi1 = ggml_cuda_dp4a(v1, q8[j+0], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8[j+4], sumi2);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1[ib32].ds);
    const uint8_t sh = bq4->scales_h[ib32/2] >> 4*(ib32%2);
    const int ls1 = ((bq4->scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32;
    const int ls2 = ((bq4->scales_l[ib32] >>  4) | ((sh << 2) & 0x30)) - 32;
    *result += d * (sumi1 * ls1 + sumi2 * ls2);
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

__device__ __forceinline__ void vec_dot_iq4_k_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq4_k_r4 * bq4 = (const block_iq4_k_r4 *)vbq + kbx;

    // iqs is 0...28 in steps of 2
    const int ib16 = iqs/2;
    const float d8 = __low2float(bq8_1[ib16/2].ds);
    const int32_t  * q8 = (const int *)bq8_1[ib16/2].qs + 4*(ib16%2);

    int ib32 = ib16/2;
    int is   = ib16%2;
    int scales;
    const uint32_t * scales_l = (const uint32_t *)bq4->scales_l;
    const uint32_t * scales_h = (const uint32_t *)bq4->scales_h;
    scales = __vsub4(((scales_l[2*(ib32%4)+is] >> 4*(ib32/4)) & 0x0f0f0f0f) | (((scales_h[2*(ib32%2)+is] >> 2*(ib32/2)) & 0x03030303) << 4), 0x20202020);
    const int8_t * s8 = (const int8_t *)&scales;
    int2 val1;
    const int * q4 = (const int *)bq4->qs + 16*ib32;
    for (int i = 0; i < 4; ++i) {
        auto values1 = iq4k_values + (((bq4->extra[i+4*is] >> ib32) & 1) << 4);
        int sumi1 = 0;
        val1  = get_int_from_table_16(q4[i+4*is+0], values1);
        sumi1 = ggml_cuda_dp4a(val1.x, q8[0], ggml_cuda_dp4a(val1.y, q8[2], sumi1));
        val1  = get_int_from_table_16(q4[i+4*is+8], values1);
        sumi1 = ggml_cuda_dp4a(val1.x, q8[1], ggml_cuda_dp4a(val1.y, q8[3], sumi1));
        const float d = __half2float(bq4->d[i]) * d8;
        result[i] += d * sumi1 * s8[i];
    }
}

__device__ __forceinline__ void vec_dot_iq4_ks_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const float * dptr = (const float *)vbq;
    const block_iq4_ks_r4 * bq4 = (const block_iq4_ks_r4 *)(dptr + 4) + kbx;

    // iqs is 0...28 in steps of 2
    const int ib16 = iqs/2;
    const float d8 = __low2float(bq8_1[ib16/2].ds);
    const int32_t  * q8 = (const int *)bq8_1[ib16/2].qs + 4*(ib16%2);

    int ib32 = ib16/2;
    int is   = ib16%2;
    const uint32_t * scales32 = (const uint32_t *)bq4->scales;
    int scales = __vsub4(scales32[ib32] & 0xfefefefe, 0x7f7f7f7f);
    const int8_t * s8 = (const int8_t *)&scales;
    int2 val;
    const int * q4 = (const int *)bq4->qs + 16*ib32;
    for (int i = 0; i < 4; ++i) {
        auto values = iq4k_values + ((bq4->scales[4*ib32+i] & 1) << 4);
        int sumi = 0;
        val  = get_int_from_table_16(q4[i+4*is+0], values);
        sumi = ggml_cuda_dp4a(val.x, q8[0], ggml_cuda_dp4a(val.y, q8[2], sumi));
        val  = get_int_from_table_16(q4[i+4*is+8], values);
        sumi = ggml_cuda_dp4a(val.x, q8[1], ggml_cuda_dp4a(val.y, q8[3], sumi));
        const float d = dptr[i] * d8;
        result[i] += d * sumi * s8[i];
    }
}

__device__ __forceinline__ void vec_dot_iq1_s_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const half * dptr = (const half *)vbq;
    const block_iq1_s_r4 * bq1 = (const block_iq1_s_r4 *)(dptr + 4) + kbx;

    // iqs is 0 or 2
    const float d8 = __low2float(bq8_1->ds);
    const int32_t  * q8 = (const int *)bq8_1->qs;

    int32_t grid32[2];
    const int * igrid = (const int *)grid32;

    int minus = 0;
    for (int k = 0; k < 4; ++k) minus = ggml_cuda_dp4a(0x01010101, q8[4*(iqs/2)+k], minus);

    for (int i = 0; i < 4; ++i) {
        float dl = __half2float(dptr[i])*(2*((bq1->qh[i] >> 12) & 7) + 1) * d8;
        float ml = dl * (bq1->qh[i] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA);
        grid32[0] = iq1s_grid_gpu[bq1->qs[4*iqs+i] | (((bq1->qh[i] >> 3*iqs) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        int sumi = ggml_cuda_dp4a(igrid[0], q8[4*(iqs/2)+0], ggml_cuda_dp4a(igrid[1], q8[4*(iqs/2)+1], 0));
        grid32[0] = iq1s_grid_gpu[bq1->qs[4*iqs+i+4] | (((bq1->qh[i] >> (3*iqs+3)) & 7) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        sumi = ggml_cuda_dp4a(igrid[0], q8[4*(iqs/2)+2], ggml_cuda_dp4a(igrid[1], q8[4*(iqs/2)+3], sumi));
        result[i] += dl * sumi + ml * minus;
    }
}

__device__ __forceinline__ void vec_dot_iq1_m_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const half * dptr = (const half *)vbq;
    const block_iq1_m_r4 * bq1 = (const block_iq1_m_r4 *)(dptr + 4) + kbx;

    // iqs is 0 or 2
    const float d8 = __low2float(bq8_1->ds);
    const int32_t  * q8 = (const int *)bq8_1->qs;

    int32_t grid32[2];
    const int * igrid = (const int *)grid32;

    int minus1 = ggml_cuda_dp4a(0x01010101, q8[4*(iqs/2)+0], ggml_cuda_dp4a(0x01010101, q8[4*(iqs/2)+1], 0));
    int minus2 = ggml_cuda_dp4a(0x01010101, q8[4*(iqs/2)+2], ggml_cuda_dp4a(0x01010101, q8[4*(iqs/2)+3], 0));

    for (int i = 0; i < 4; ++i) {
        float dl = __half2float(dptr[i])*((bq1->scales[i] >> 4*(iqs/2)) & 0xf) * d8;
        float ml1 = dl * (bq1->qh[4*(iqs/2)+i] & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA);
        float ml2 = dl * (bq1->qh[4*(iqs/2)+i] & 0x80 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA);
        grid32[0] = iq1s_grid_gpu[bq1->qs[4*iqs+i] | ((bq1->qh[4*(iqs/2)+i] & 0x07) << 8)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        int sumi = ggml_cuda_dp4a(igrid[0], q8[4*(iqs/2)+0], ggml_cuda_dp4a(igrid[1], q8[4*(iqs/2)+1], 0));
        grid32[0] = iq1s_grid_gpu[bq1->qs[4*iqs+i+4] | ((bq1->qh[4*(iqs/2)+i] & 0x70) << 4)];
        grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
        grid32[0] &= 0x0f0f0f0f;
        sumi = ggml_cuda_dp4a(igrid[0], q8[4*(iqs/2)+2], ggml_cuda_dp4a(igrid[1], q8[4*(iqs/2)+3], sumi));
        result[i] += dl * sumi + ml1 * minus1 + ml2*minus2;
    }
}

#define VDR_IQ4_KS_Q8_1_MMVQ 4
#define VDR_IQ4_KS_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq4_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq4_ks * bq4 = (const block_iq4_ks *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4; // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint32_t * q4 = (const uint32_t *)bq4->qs + 4*ib32;
    const float dl = scale * ((bq4->scales[ib32] & 254) - 127);
    auto values = iq4k_values + ((bq4->scales[ib32] & 1) << 4);
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        auto v = get_int_from_table_16(q4[j], values);
        sumi = ggml_cuda_dp4a(v.x, q8[j+0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[j+4], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

__device__ __forceinline__ void vec_dot_iq4_kt_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

    float scale = *(const float *)vbq;
    const block_iq4_kt * bq4 = (const block_iq4_kt *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4; // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    //const int8_t  * q8 = bq8_1[ib32].qs;
    const int ls = (bq4->qs[ib32] & 0xff) >> 1;
    const float dl = scale * (ls - 64);
    const uint32_t idx0 = ((bq4->qs[ib32] & 1) << 15) + 4096;
    auto ql = (const uint8_t *)(bq4->qs + 8);
    auto qh = ql + 64;
    ql += 8*ib32;
    qh += 8*(ib32%4);
    const int shift1 = 8 - 4*(ib32/4);
    int sumi = 0;
    for (int j = 0; j < 8; ++j) {
        const uint32_t sh = bq4->qs[ib32] >> (8 + 3*j);
        uint32_t val = ql[j] + ((qh[j] << shift1) & 0xf00) + ((sh & 7) << 12) + idx0;
        int v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            //int s = val & km;
            //sumi += q8[4*j+k] * ggml_cuda_dp4a(s, 0x01010101, -126);
            v4 |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        sumi = ggml_cuda_dp4a(v4, q8[j], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

__device__ __forceinline__ void vec_dot_iq1_kt_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

    float scale = *(const float *)vbq;
    const block_iq1_kt * bq1 = (const block_iq1_kt *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4;
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const int ls = iq4k_values[bq1->sh[ib32] & 0xf];
    const float dl = scale * ls;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t val = bq1->ql[4*ib32+j] + 4096 + ((bq1->qh[4*(ib32%4)+j] << (8 - 4*(ib32/4))) & 0xf00) + ((bq1->sh[ib32] << (8 - j)) & 0x1000);
        int v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v4 |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        sumi = ggml_cuda_dp4a(v4, q8[2*j+0], sumi);
        v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v4 |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        sumi = ggml_cuda_dp4a(v4, q8[2*j+1], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

__device__ __forceinline__ void vec_dot_iq2_kt_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

    float scale = *(const float *)vbq;
    const block_iq2_kt * bq2 = (const block_iq2_kt *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4;
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const int ls = iq4k_values[(bq2->scales[ib32%4] >> 4*(ib32/4)) & 0xf];
    const float dl = scale * ls * 1.05f;
    auto ql = (const uint16_t *)bq2->ql;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t val = ql[4*ib32+j] + 4096;
        int v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v4 |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        sumi = ggml_cuda_dp4a(v4, q8[2*j+0], sumi);
        v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v4 |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        sumi = ggml_cuda_dp4a(v4, q8[2*j+1], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

__device__ __forceinline__ void vec_dot_iq3_kt_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

    float scale = *(const float *)vbq;
    const block_iq3_kt * bq3 = (const block_iq3_kt *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4;
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const int ls = (bq3->scales[ib32%4] >> 4*(ib32/4)) & 0xf;
    const float dl = scale * ls * 1.015f;
    auto ql = (const uint16_t *)bq3->ql;
    uint32_t mask = 0x01010101 << ib32;
    const uint32_t * qh = (const uint32_t *)bq3->qh;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t val = ql[4*ib32+j] + 4096;
        int v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            int8_t q = std::abs(ggml_cuda_dp4a(val & km, 0x01010101, -126));
            v4 |= q << 8*k;
        }
        uint32_t signs = __vcmpne4(qh[2*j+0] & mask, 0);
        v4 = __vsub4(v4 ^ signs, signs);
        sumi = ggml_cuda_dp4a(v4, q8[2*j+0], sumi);
        v4 = 0;
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            int8_t q = std::abs(ggml_cuda_dp4a(val & km, 0x01010101, -126));
            v4 |= q << 8*k;
        }
        signs = __vcmpne4(qh[2*j+1] & mask, 0);
        v4 = __vsub4(v4 ^ signs, signs);
        sumi = ggml_cuda_dp4a(v4, q8[2*j+1], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

#define VDR_IQ4_KSS_Q8_1_MMVQ 4
#define VDR_IQ4_KSS_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq4_kss_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq4_kss * bq4 = (const block_iq4_kss *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0...28
    const int ib32 = iqs/4; // Why iqs/4 ?
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint32_t * q4 = (const uint32_t *)bq4->qs + 4*ib32;
    uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
    uint8_t ls = (s32 | (s32 >> 15)) & 0xff;
    const float dl = scale * ((ls & 254) - 127);
    auto values = iq4k_values + ((ls & 1) << 4);
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t aux32 = q4[j] & 0xfffefffe;
        aux32 ^= (aux32 >> 1);
        auto v = get_int_from_table_16(aux32, values);
        sumi = ggml_cuda_dp4a(v.x, q8[j+0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[j+4], sumi);
    }
    *result += dl * __low2float(bq8_1[ib32].ds) * sumi;
}

#define VDR_IQ5_K_Q8_1_MMVQ 4
#define VDR_IQ5_K_Q8_1_MMQ  4

__device__ __forceinline__ int int_from_table(const uint8_t * a8, const uint8_t * values) {
    uint16_t v1 = values[a8[0]] | (values[a8[1]] << 8);
    uint16_t v2 = values[a8[2]] | (values[a8[3]] << 8);
    return v1 | (v2 << 16);
}

__device__ __forceinline__ void vec_dot_iq5_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq5_k * bq5 = (const block_iq5_k *) vbq + kbx;
    const uint8_t * all_values = (const uint8_t *)iq5nl_values;

    int i4 = iqs/4;  // 0...7.  Blocks of 16 index is 4*(i4/2) + (i4%2) + (0 and 2)

    const int32_t  * q8_1 = (const int *)bq8_1[2*(i4/2)+0].qs + 4*(i4%2);
    const int32_t  * q8_2 = (const int *)bq8_1[2*(i4/2)+1].qs + 4*(i4%2);
    const uint32_t * q4 = (const uint32_t *)bq5->qs + 8*(i4/2) + 4*(i4%2);
    const uint32_t * qh = (const uint32_t *)bq5->qh + 4*(i4%2);
    const uint16_t extra = bq5->extra >> (4*(i4/2) + (i4%2));
    const uint8_t * values1 = all_values + 32*(extra & 1);
    const uint8_t * values2 = all_values +  8*(extra & 4);
    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t h = qh[j] >> 2*(i4/2);
        aux32[0] = ((q4[j] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x10101010);
        aux32[1] = ((q4[j] >> 4) & 0x0f0f0f0f) | ((h << 3) & 0x10101010);
        v1 = int_from_table(a8+0, values1);
        v2 = int_from_table(a8+4, values2);
        sumi1 = ggml_cuda_dp4a(v1, q8_1[j], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8_2[j], sumi2);
    }
    const float d5 = __half2float(bq5->d);
    const uint8_t sh = bq5->scales_h[i4/2] >> 2*(i4%2);
    const int ls1 = (((bq5->scales_l[2*(i4/2)+0] >> 4*(i4%2)) & 0xf) | ((sh << 4) & 0x30)) - 32;
    const int ls2 = (((bq5->scales_l[2*(i4/2)+1] >> 4*(i4%2)) & 0xf) | ((sh << 0) & 0x30)) - 32;
    *result += d5 * (__low2float(bq8_1[2*(i4/2)+0].ds) * sumi1 * ls1 + __low2float(bq8_1[2*(i4/2)+1].ds) * sumi2 * ls2);
}

__device__ __forceinline__ void vec_dot_iq5_k_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq5_k_r4 * bq5 = (const block_iq5_k_r4 *)vbq + kbx;

    // iqs is 0...28 in steps of 2
    const int ib16 = iqs/2;
    const float d8 = __low2float(bq8_1[ib16/2].ds);
    const int32_t  * q8 = (const int *)bq8_1[ib16/2].qs + 4*(ib16%2);

    int ib32 = ib16/2;
    int is   = ib16%2;
    int scales;
    const uint32_t * scales_l = (const uint32_t *)bq5->scales_l;
    const uint32_t * scales_h = (const uint32_t *)bq5->scales_h;
    scales = __vsub4(((scales_l[2*(ib32%4)+is] >> 4*(ib32/4)) & 0x0f0f0f0f) | (((scales_h[2*(ib32%2)+is] >> 2*(ib32/2)) & 0x03030303) << 4), 0x20202020);
    const int8_t * s8 = (const int8_t *)&scales;
    int2 val1;
    const int * q4 = (const int *)bq5->qs + 16*ib32;
    const int * qh = (const int *)bq5->qh +  4*ib32;
    int aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
    for (int i = 0; i < 4; ++i) {
        auto values1 = iq5nl_values + (((bq5->extra[i+4*is] >> ib32) & 1) << 5);
        int sumi1 = 0;
        aux32[0] = ((q4[i+4*is+0] >> 0) & 0x0f0f0f0f) | (((qh[i] >> (2*is+0)) & 0x01010101) << 4);
        aux32[1] = ((q4[i+4*is+0] >> 4) & 0x0f0f0f0f) | (((qh[i] >> (2*is+1)) & 0x01010101) << 4);
        val1.x  = int_from_table(aux8+0, (const uint8_t *)values1);
        val1.y  = int_from_table(aux8+4, (const uint8_t *)values1);
        sumi1 = ggml_cuda_dp4a(val1.x, q8[0], ggml_cuda_dp4a(val1.y, q8[2], sumi1));
        aux32[0] = ((q4[i+4*is+8] >> 0) & 0x0f0f0f0f) | (((qh[i] >> (2*is+4)) & 0x01010101) << 4);
        aux32[1] = ((q4[i+4*is+8] >> 4) & 0x0f0f0f0f) | (((qh[i] >> (2*is+5)) & 0x01010101) << 4);
        val1.x  = int_from_table(aux8+0, (const uint8_t *)values1);
        val1.y  = int_from_table(aux8+4, (const uint8_t *)values1);
        sumi1 = ggml_cuda_dp4a(val1.x, q8[1], ggml_cuda_dp4a(val1.y, q8[3], sumi1));
        const float d = __half2float(bq5->d[i]) * d8;
        result[i] += d * sumi1 * s8[i];
    }
}

__device__ __forceinline__ void vec_dot_iq5_ks_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const float * dptr = (const float *)vbq;
    const block_iq5_ks_r4 * bq5 = (const block_iq5_ks_r4 *)(dptr + 4) + kbx;

    // iqs is 0...28 in steps of 2
    const int ib16 = iqs/2;
    const float d8 = __low2float(bq8_1[ib16/2].ds);
    const int32_t  * q8 = (const int *)bq8_1[ib16/2].qs + 4*(ib16%2);

    int ib32 = ib16/2;
    int is   = ib16%2;
    const uint32_t * scales32 = (const uint32_t *)bq5->scales;
    int scales = __vsub4(scales32[ib32] & 0xfefefefe, 0x7f7f7f7f);
    const int8_t * s8 = (const int8_t *)&scales;
    int2 val;
    const int * q4 = (const int *)bq5->qs + 16*ib32;
    const int * qh = (const int *)bq5->qh +  4*ib32;
    int aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
    for (int i = 0; i < 4; ++i) {
        auto values = iq5nl_values + ((bq5->scales[4*ib32+i] & 1) << 5);
        int sumi = 0;
        aux32[0] = ((q4[i+4*is+0] >> 0) & 0x0f0f0f0f) | (((qh[i] >> (2*is+0)) & 0x01010101) << 4);
        aux32[1] = ((q4[i+4*is+0] >> 4) & 0x0f0f0f0f) | (((qh[i] >> (2*is+1)) & 0x01010101) << 4);
        val.x  = int_from_table(aux8+0, (const uint8_t *)values);
        val.y  = int_from_table(aux8+4, (const uint8_t *)values);
        sumi = ggml_cuda_dp4a(val.x, q8[0], ggml_cuda_dp4a(val.y, q8[2], sumi));
        aux32[0] = ((q4[i+4*is+8] >> 0) & 0x0f0f0f0f) | (((qh[i] >> (2*is+4)) & 0x01010101) << 4);
        aux32[1] = ((q4[i+4*is+8] >> 4) & 0x0f0f0f0f) | (((qh[i] >> (2*is+5)) & 0x01010101) << 4);
        val.x  = int_from_table(aux8+0, (const uint8_t *)values);
        val.y  = int_from_table(aux8+4, (const uint8_t *)values);
        sumi = ggml_cuda_dp4a(val.x, q8[1], ggml_cuda_dp4a(val.y, q8[3], sumi));
        result[i] += dptr[i] * d8 * sumi * s8[i];
    }
}

__device__ __forceinline__ void vec_dot_iq5_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq5_ks * bq5 = (const block_iq5_ks *)((const char *)vbq + sizeof(float)) + kbx;
    const uint8_t * all_values = (const uint8_t *)iq5nl_values;

    int i4 = iqs/4;  // 0...7.  Blocks of 16 index is 4*(i4/2) + (i4%2) + (0 and 2)

    const int32_t  * q8_1 = (const int *)bq8_1[2*(i4/2)+0].qs + 4*(i4%2);
    const int32_t  * q8_2 = (const int *)bq8_1[2*(i4/2)+1].qs + 4*(i4%2);
    const uint32_t * q4 = (const uint32_t *)bq5->qs + 8*(i4/2) + 4*(i4%2);
    const uint32_t * qh = (const uint32_t *)bq5->qh + 4*(i4%2);
    const uint8_t * values1 = all_values + ((bq5->scales[2*(i4/2)+0] & 1) << 5);
    const uint8_t * values2 = all_values + ((bq5->scales[2*(i4/2)+1] & 1) << 5);
    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t h = qh[j] >> 2*(i4/2);
        aux32[0] = ((q4[j] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x10101010);
        aux32[1] = ((q4[j] >> 4) & 0x0f0f0f0f) | ((h << 3) & 0x10101010);
        v1 = int_from_table(a8+0, values1);
        v2 = int_from_table(a8+4, values2);
        sumi1 = ggml_cuda_dp4a(v1, q8_1[j], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8_2[j], sumi2);
    }
    const int ls1 = (bq5->scales[2*(i4/2)+0] & 254) - 127;
    const int ls2 = (bq5->scales[2*(i4/2)+1] & 254) - 127;
    *result += scale * (__low2float(bq8_1[2*(i4/2)+0].ds) * sumi1 * ls1 + __low2float(bq8_1[2*(i4/2)+1].ds) * sumi2 * ls2);
}

__device__ __forceinline__ void vec_dot_iq3_k_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq3_k_r4 * bq3 = (const block_iq3_k_r4 *)vbq + kbx;

    // iqs is 0...30 in steps of 2
    const int ib16 = iqs/2;
    const float d8 = __low2float(bq8_1[ib16/2].ds);
    const int32_t  * q8 = (const int *)bq8_1[ib16/2].qs + 4*(ib16%2);

    int ib32 = ib16/2;
    int is   = ib16%2;
    int scales[2];
    const uint32_t * scales_l = (const uint32_t *)bq3->scales_l;
    const uint32_t * scales_h = (const uint32_t *)bq3->scales_h;

    scales[0] = (((scales_l[2*(ib32%4)+is] >> 4*(ib32/4)) & 0x0f0f0f0f) << 1) | 0x01010101;
    scales[1] = (scales_h[is] >> ib32) & 0x01010101;
    // This is not faster. Why?
    //scales[1] = __vcmpeq4((scales_h[is] >> ib32) & 0x01010101, 0x01010101);
    //scales[0] = __vsub4(scales[0] ^ scales[1], scales[1]);
    const int8_t * s8 = (const int8_t *)scales;
    const uint32_t * q2 = (const uint32_t *)bq3->qs + 8*ib32 + 4*is;
    const uint32_t * qh = (const uint32_t *)bq3->qh + 4*ib32;
    for (int i = 0; i < 4; ++i) {
        uint32_t extra32 = uint32_t((bq3->extra[i+4*is] >> ib32) & 1) * 0x88888888;

        int sumi1 = 0;
        uint32_t h = qh[i] >> 4*is;
        uint32_t val1 = ((q2[i] >> 0) & 0x33333333) | extra32 | ((h << 2) & 0x04040404) | ((h << 4) & 0x40404040);
        uint32_t val2 = ((q2[i] >> 2) & 0x33333333) | extra32 | ((h << 1) & 0x04040404) | ((h << 3) & 0x40404040);
        int2 v1 = get_int_from_table_16(val1, iq3nl_values);
        int2 v2 = get_int_from_table_16(val2, iq3nl_values);
        sumi1 = ggml_cuda_dp4a(v1.x, q8[0], ggml_cuda_dp4a(v2.x, q8[1], sumi1));
        sumi1 = ggml_cuda_dp4a(v1.y, q8[2], ggml_cuda_dp4a(v2.y, q8[3], sumi1));
        const float d = __half2float(bq3->d[i]) * d8;
        result[i] += d * sumi1 * s8[i] * (s8[i+4] ? -1 : 1);
    }
}

#define VDR_IQ6_K_Q8_1_MMVQ 4
#define VDR_IQ6_K_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq6_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq6_k * bq6 = (const block_iq6_k *) vbq + kbx;
    const uint8_t * all_values = (const uint8_t *)iq6nl_values;

    int i4 = iqs/4;  // 0...7.  Blocks of 16 index is 4*(i4/2) + (i4%2) + (0 and 2)
                     //         Blocks of 32 index is 2*(i4/2) + 0 or 1

    const int32_t  * q8_1 = (const int *)bq8_1[2*(i4/2)+0].qs + 4*(i4%2);
    const int32_t  * q8_2 = (const int *)bq8_1[2*(i4/2)+1].qs + 4*(i4%2);
    const uint32_t * q4 = (const uint32_t *)bq6->qs + 8*(i4/2) + 4*(i4%2);
    const uint32_t * qh = (const uint32_t *)bq6->qh + 8*(i4/4) + 4*(i4%2);
    const uint16_t extra = bq6->extra >> (4*(i4/2) + (i4%2));
    const uint8_t * values1 = all_values + 64*(extra & 1);
    const uint8_t * values2 = all_values + 16*(extra & 4);
    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        uint32_t h = qh[j] >> 4*((i4/2)%2);
        aux32[0] = ((q4[j] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x30303030);
        aux32[1] = ((q4[j] >> 4) & 0x0f0f0f0f) | ((h << 2) & 0x30303030);
        v1 = int_from_table(a8+0, values1);
        v2 = int_from_table(a8+4, values2);
        sumi1 = ggml_cuda_dp4a(v1, q8_1[j], sumi1);
        sumi2 = ggml_cuda_dp4a(v2, q8_2[j], sumi2);
    }
    const float d6 = __half2float(bq6->d);
    *result += d6 * (__low2float(bq8_1[2*(i4/2)+0].ds) * sumi1 * bq6->scales[4*(i4/2)+(i4%2)] + __low2float(bq8_1[2*(i4/2)+1].ds) * sumi2 * bq6->scales[4*(i4/2)+(i4%2)+2]);
}

#define VDR_IQ2_K_Q8_1_MMVQ 4
#define VDR_IQ2_K_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq2_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    // iqs is 0, 4, 8, 12, 16, 20, 24, 28
    // we have 16 packed quants (when cast to int)

    int i4 = iqs/4;  // 0...7. We will process q8 blocks 4*(i4/4), 4*(i4/4)+1, 4*(i4/4)+2, 4*(i4/4)+3
    const int32_t  * q8_1 = (const int *)bq8_1[4*(i4/4)+0].qs + 2*(i4%4);
    const int32_t  * q8_2 = (const int *)bq8_1[4*(i4/4)+1].qs + 2*(i4%4);
    const int32_t  * q8_3 = (const int *)bq8_1[4*(i4/4)+2].qs + 2*(i4%4);
    const int32_t  * q8_4 = (const int *)bq8_1[4*(i4/4)+3].qs + 2*(i4%4);

    const block_iq2_k * bq2 = (const block_iq2_k *) vbq + kbx;
    const uint32_t * q2 = (const uint32_t *)bq2->qs + 8*(i4/4) + 2*(i4%4);
    const uint16_t extra = bq2->extra >> (8*(i4/4) + (i4%4)/2);

    const uint32_t * scales = (const uint32_t *)bq2->scales;
    uint32_t s32 = __vsub4((scales[i4/4] >> 4*(((i4%4)/2)%2)) & 0x0f0f0f0f, 0x08080808);
    const int8_t * s8 = (const int8_t *)&s32;

    // Block of 16: (32*(4*(i4/4)+k)+8*(i4%4))/16 = 8*(i4/4) + 2*k + (i4%4)/2
    // -> scales_l[4*(i4/4) + k] >> 4*(((i4%4)/2)%2)

#ifdef __CUDA_ARCH__
    uint32_t extra32 = uint32_t(extra & 0xff) * 0x01010101;
    uint32_t extra32_1 = (extra32 << 2) & 0x44444444;
    uint32_t extra32_2 = (extra32 << 0) & 0x44444444;

    uint32_t val1, val2;

    val1 = ((q2[0] >> 0) & 0x33333333) | extra32_1; val2 = ((q2[1] >> 0) & 0x33333333) | extra32_1;
    int2 v1 = get_int_from_table_8(val1, iq2nl_values);
    int2 v2 = get_int_from_table_8(val2, iq2nl_values);
    int sumi1 = ggml_cuda_dp4a(v2.x, q8_1[1], ggml_cuda_dp4a(v1.x, q8_1[0], 0)) * s8[0];
    int sumi3 = ggml_cuda_dp4a(v2.y, q8_3[1], ggml_cuda_dp4a(v1.y, q8_3[0], 0)) * s8[2];

    val1 = ((q2[0] >> 2) & 0x33333333) | extra32_2; val2 = ((q2[1] >> 2) & 0x33333333) | extra32_2;
    v1 = get_int_from_table_8(val1, iq2nl_values);
    v2 = get_int_from_table_8(val2, iq2nl_values);
    int sumi2 = ggml_cuda_dp4a(v2.x, q8_2[1], ggml_cuda_dp4a(v1.x, q8_2[0], 0)) * s8[1];
    int sumi4 = ggml_cuda_dp4a(v2.y, q8_4[1], ggml_cuda_dp4a(v1.y, q8_4[0], 0)) * s8[3];

#else

    const int * all_values = (const int *)iq2k_table;
    const int * values;

    uint32_t val1 = q2[0], val2 = q2[1];

    uint32_t aux32[2];
    int v1, v2;

    aux32[0] = ((val1 >> 0) & 0x03030303); aux32[1] = ((val2 >> 0) & 0x03030303); values = all_values + ((extra & 0x01) << 8);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi1 = ggml_cuda_dp4a(v2, q8_1[1], ggml_cuda_dp4a(v1, q8_1[0], 0)) * s8[0];

    aux32[0] = ((val1 >> 2) & 0x03030303); aux32[1] = ((val2 >> 2) & 0x03030303); values = all_values + ((extra & 0x04) << 6);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi2 = ggml_cuda_dp4a(v2, q8_2[1], ggml_cuda_dp4a(v1, q8_2[0], 0)) * s8[1];

    aux32[0] = ((val1 >> 4) & 0x03030303); aux32[1] = ((val2 >> 4) & 0x03030303); values = all_values + ((extra & 0x10) << 4);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi3 = ggml_cuda_dp4a(v2, q8_3[1], ggml_cuda_dp4a(v1, q8_3[0], 0)) * s8[2];

    aux32[0] = ((val1 >> 6) & 0x03030303); aux32[1] = ((val2 >> 6) & 0x03030303); values = all_values + ((extra & 0x40) << 2);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi4 = ggml_cuda_dp4a(v2, q8_4[1], ggml_cuda_dp4a(v1, q8_4[0], 0)) * s8[3];
#endif

    *result += __half2float(bq2->d) * (__low2float(bq8_1[4*(i4/4)+0].ds) * sumi1
                                    +  __low2float(bq8_1[4*(i4/4)+1].ds) * sumi2
                                    +  __low2float(bq8_1[4*(i4/4)+2].ds) * sumi3
                                    +  __low2float(bq8_1[4*(i4/4)+3].ds) * sumi4);
}

#define VDR_IQ2_KS_Q8_1_MMVQ 4
#define VDR_IQ2_KS_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq2_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const half *)vbq;
    const block_iq2_ks * bq2 = (const block_iq2_ks *)((const char *)vbq + sizeof(half)) + kbx;

    int i4 = iqs/4;  // 0...7. We will process q8 blocks 4*(i4/4), 4*(i4/4)+1, 4*(i4/4)+2, 4*(i4/4)+3
    const int32_t  * q8_1 = (const int *)bq8_1[4*(i4/4)+0].qs + 2*(i4%4);
    const int32_t  * q8_2 = (const int *)bq8_1[4*(i4/4)+1].qs + 2*(i4%4);
    const int32_t  * q8_3 = (const int *)bq8_1[4*(i4/4)+2].qs + 2*(i4%4);
    const int32_t  * q8_4 = (const int *)bq8_1[4*(i4/4)+3].qs + 2*(i4%4);

    const uint16_t * q2 = (const uint16_t *)bq2->qs + 16*(i4/4) + 4*(i4%4);
    const uint16_t extra = bq2->extra >> 4*(i4/4);

    uint32_t val1 = q2[0] | (q2[1] << 16), val2 = q2[2] | (q2[3] << 16);

    int32_t scales32;
    const uint16_t * scales16 = (const uint16_t *)bq2->scales;
    scales32 = __vsub4((scales16[i4/4] | (scales16[i4/4] << 12)) & 0x0f0f0f0f, 0x10101010);
    int8_t * s8 = (int8_t *)&scales32;
    s8[0] += ((extra >> 4) & 0x10);
    s8[1] += ((extra >> 6) & 0x10);
    s8[2] += ((extra >> 5) & 0x10);
    s8[3] += ((extra >> 7) & 0x10);

#ifdef __CUDA_ARCH__

    uint32_t extra32 = uint32_t(extra & 0xf) * 0x01010101;

    uint32_t this_extra = ((extra32 << 2) & 0x04040404) | ((extra32 << 4) & 0x40404040);
    uint32_t idx1 = ((val1 >> 0) & 0x33333333) | this_extra;
    uint32_t idx2 = ((val2 >> 0) & 0x33333333) | this_extra;
    int2 v1 = get_int_from_table_8(idx1, iq2nl_values);
    int2 v2 = get_int_from_table_8(idx2, iq2nl_values);

    int sumi1 = ggml_cuda_dp4a(v2.x, q8_1[1], ggml_cuda_dp4a(v1.x, q8_1[0], 0)) * s8[0];
    int sumi3 = ggml_cuda_dp4a(v2.y, q8_3[1], ggml_cuda_dp4a(v1.y, q8_3[0], 0)) * s8[1];

    this_extra = ((extra32 << 1) & 0x04040404) | ((extra32 << 3) & 0x40404040);
    idx1 = ((val1 >> 2) & 0x33333333) | this_extra;
    idx2 = ((val2 >> 2) & 0x33333333) | this_extra;
    v1 = get_int_from_table_8(idx1, iq2nl_values);
    v2 = get_int_from_table_8(idx2, iq2nl_values);

    int sumi2 = ggml_cuda_dp4a(v2.x, q8_2[1], ggml_cuda_dp4a(v1.x, q8_2[0], 0)) * s8[2];
    int sumi4 = ggml_cuda_dp4a(v2.y, q8_4[1], ggml_cuda_dp4a(v1.y, q8_4[0], 0)) * s8[3];

#else

    uint32_t aux32[2];
    int v1, v2;
    const int * all_values = (const int *)iq2k_table;
    const int * values;

    aux32[0] = ((val1 >> 0) & 0x03030303); aux32[1] = ((val2 >> 0) & 0x03030303); values = all_values + ((extra & 0x01) << 8);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi1 = ggml_cuda_dp4a(v2, q8_1[1], ggml_cuda_dp4a(v1, q8_1[0], 0)) * s8[0];

    aux32[0] = ((val1 >> 2) & 0x03030303); aux32[1] = ((val2 >> 2) & 0x03030303); values = all_values + ((extra & 0x02) << 7);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi2 = ggml_cuda_dp4a(v2, q8_2[1], ggml_cuda_dp4a(v1, q8_2[0], 0)) * s8[2];

    aux32[0] = ((val1 >> 4) & 0x03030303); aux32[1] = ((val2 >> 4) & 0x03030303); values = all_values + ((extra & 0x04) << 6);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi3 = ggml_cuda_dp4a(v2, q8_3[1], ggml_cuda_dp4a(v1, q8_3[0], 0)) * s8[1];

    aux32[0] = ((val1 >> 6) & 0x03030303); aux32[1] = ((val2 >> 6) & 0x03030303); values = all_values + ((extra & 0x08) << 5);
    v1 = int_from_table_4(aux32[0], values);
    v2 = int_from_table_4(aux32[1], values);
    int sumi4 = ggml_cuda_dp4a(v2, q8_4[1], ggml_cuda_dp4a(v1, q8_4[0], 0)) * s8[3];
#endif

    *result += scale * (__low2float(bq8_1[4*(i4/4)+0].ds) * sumi1
                     +  __low2float(bq8_1[4*(i4/4)+1].ds) * sumi2
                     +  __low2float(bq8_1[4*(i4/4)+2].ds) * sumi3
                     +  __low2float(bq8_1[4*(i4/4)+3].ds) * sumi4);
}

__device__ __forceinline__ void vec_dot_iq2_k_r4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const block_iq2_k_r4 * bq2 = (const block_iq2_k_r4 *)vbq + kbx;

    // iqs is 0...30 in steps of 2
    const int ib16 = iqs/2;
    const float d8 = __low2float(bq8_1[ib16/2].ds);
    const int32_t  * q8 = (const int *)bq8_1[ib16/2].qs + 4*(ib16%2);

    int ib32 = ib16/2;
    int is   = ib16%2;
    const int * scales_l = (const int *)bq2->scales;

    int scales = __vsub4(((scales_l[2*(ib32%4)+is] >> 4*(ib32/4)) & 0x0f0f0f0f), 0x08080808);
    const int8_t * s8 = (const int8_t *)&scales;

    const int * q2 = (const int *)bq2->qs + 8*ib32 + 4*is;

#ifdef __CUDA_ARCH__

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t extra32 = uint32_t((bq2->extra[i+4*is] >> ib32) & 1) * 0x04040404;
        extra32 |= (extra32 << 4);
        uint32_t val1 = ((q2[i] >> 0) & 0x33333333) | extra32;
        uint32_t val2 = ((q2[i] >> 2) & 0x33333333) | extra32;
        int2 v1 = get_int_from_table_8(val1, iq2nl_values);
        int2 v2 = get_int_from_table_8(val2, iq2nl_values);
        int sumi = 0;
        sumi = ggml_cuda_dp4a(v1.x, q8[0], ggml_cuda_dp4a(v2.x, q8[1], sumi));
        sumi = ggml_cuda_dp4a(v1.y, q8[2], ggml_cuda_dp4a(v2.y, q8[3], sumi));
        const float d = __half2float(bq2->d[i]) * d8;
        result[i] += d * sumi * s8[i];
    }

#else
    const int * all_values = (const int *)iq2k_table;
    int2 val1;
    int aux32[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        auto values1 = all_values + (((bq2->extra[i+4*is] >> ib32) & 1) << 8);
        int sumi1 = 0;
        aux32[0] = ((q2[i] >> 0) & 0x03030303);
        aux32[1] = ((q2[i] >> 2) & 0x03030303);
        val1.x  = int_from_table_4(aux32[0], values1);
        val1.y  = int_from_table_4(aux32[1], values1);
        sumi1 = ggml_cuda_dp4a(val1.x, q8[0], ggml_cuda_dp4a(val1.y, q8[1], sumi1));
        aux32[0] = ((q2[i] >> 4) & 0x03030303);
        aux32[1] = ((q2[i] >> 6) & 0x03030303);
        val1.x  = int_from_table_4(aux32[0], values1);
        val1.y  = int_from_table_4(aux32[1], values1);
        sumi1 = ggml_cuda_dp4a(val1.x, q8[2], ggml_cuda_dp4a(val1.y, q8[3], sumi1));
        const float d = __half2float(bq2->d[i]) * d8;
        result[i] += d * sumi1 * s8[i];
    }
#endif
}

#define VDR_IQ3_K_Q8_1_MMVQ 4
#define VDR_IQ3_K_Q8_1_MMQ  4

__device__ __forceinline__ void vec_dot_iq3_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs, float * result) {
    const block_iq3_k * bq3 = (const block_iq3_k *) vbq + kbx;

    int iqs = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1. 0 works on quants 0...127, 1 on quants 128...255
                              // Each thread processes 8 quants in each of the 4 32-blocks
    const int il8   = iqs%4;  // 0...3. 0 works on quants 0...7, 1 on quants 8...15, 2 on 16...23, 3 on 24...31
    const int shift = 4*(il8/2);

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    uint32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    const int hshift = 4*(1-ib128);
    const uint16_t sh = bq3->scales_h >> (8*ib128 + il8/2);

    const uint8_t extra = bq3->extra >> (8*ib128 + il8/2);
    uint32_t extra32 = uint32_t(extra) * 0x01010101;
    uint32_t extra32_1 = ((extra32 << 3) & 0x08080808) | ((extra32 << 5) & 0x80808080);
    uint32_t extra32_2 = ((extra32 << 2) & 0x08080808) | ((extra32 << 4) & 0x80808080);

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
    for (int i = 0; i < 2; ++i) {
        uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
        uint32_t vh = ((qh[2*i+0] | (qh[2*i+1] << 16)) << hshift);

        uint32_t val1 = ((vl >> 0) & 0x33333333) | extra32_1 | ((vh >> 2) & 0x04040404) | ((vh >> 0) & 0x40404040);
        uint32_t val2 = ((vl >> 2) & 0x33333333) | extra32_2 | ((vh >> 3) & 0x04040404) | ((vh >> 1) & 0x40404040);
        int2 v1 = get_int_from_table_16(val1, iq3nl_values);
        int2 v2 = get_int_from_table_16(val2, iq3nl_values);

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        sumi[0] = ggml_cuda_dp4a(v1.x, q8[i], sumi[0]);

        q8 += sizeof(block_q8_1)/4;
        sumi[1] = ggml_cuda_dp4a(v2.x, q8[i], sumi[1]);

        q8 += sizeof(block_q8_1)/4;
        sumi[2] = ggml_cuda_dp4a(v1.y, q8[i], sumi[2]);

        q8 += sizeof(block_q8_1)/4;
        sumi[3] = ggml_cuda_dp4a(v2.y, q8[i], sumi[3]);
    }
    const float d = __half2float(bq3->d);
    const uint16_t * sl16 = (const uint16_t *)bq3->scales_l + 2*ib128;
    aux32 = ((((sl16[0] | (sl16[1] << 16)) >> shift) & 0x0f0f0f0f) << 1) | 0x01010101;
    *result += d * (__low2float(bq8_1[4*ib128+0].ds) * aux8[0] * (sh & 0x01 ? -1 : 1) * sumi[0] +
                    __low2float(bq8_1[4*ib128+1].ds) * aux8[1] * (sh & 0x04 ? -1 : 1) * sumi[1] +
                    __low2float(bq8_1[4*ib128+2].ds) * aux8[2] * (sh & 0x10 ? -1 : 1) * sumi[2] +
                    __low2float(bq8_1[4*ib128+3].ds) * aux8[3] * (sh & 0x40 ? -1 : 1) * sumi[3]);

}

// TODO
__device__ __forceinline__ void vec_dot_iq2_kl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs, float * result) {

    float d = __half2float(*(const half *)vbq);
    const block_iq2_kl * bq2 = (const block_iq2_kl *)((const char *)vbq + sizeof(half)) + kbx;

    int iqs = iiqs/4;
    const int ib64 = iqs/2;  // 0...3. 0 works on quants 0...63, 1 on quants 64...127, etc.
                             // Each thread processes 16 quants in each of the 2 32-blocks
    const int il16 = iqs%2;  // 0...3. 0 works on quants 0...7, 1 on quants 8...15, 2 on 16...23, 3 on 24...31

    const uint16_t * ql = (const uint16_t *)bq2->qs + 8*ib64 + 4*il16;
    const uint16_t * qh = (const uint16_t *)bq2->qh + 4*il16;

    int32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    const int * q8l = (const int *)bq8_1[2*ib64+0].qs + 4*il16;
    const int * q8h = (const int *)bq8_1[2*ib64+1].qs + 4*il16;

    int sumi1 = 0, sumi2 = 0;
    int v1, v2;
    for (int i = 0; i < 2; ++i) {
        uint32_t vl =  ql[2*i+0] | (ql[2*i+1] << 16);
        uint32_t vh = (qh[2*i+0] | (qh[2*i+1] << 16)) >> 2*ib64;

        aux32 = (vl & 0x0f0f0f0f) | ((vh << 4) & 0x10101010);
        v1 = iq2kl_values[aux8[0]] | (iq2kl_values[aux8[1]] << 16);
        v2 = iq2kl_values[aux8[2]] | (iq2kl_values[aux8[3]] << 16);
        sumi1 = ggml_cuda_dp4a(v1, q8l[2*i+0], ggml_cuda_dp4a(v2, q8l[2*i+1], sumi1));

        aux32 = ((vl >> 4) & 0x0f0f0f0f) | ((vh << 3) & 0x10101010);
        v1 = iq2kl_values[aux8[0]] | (iq2kl_values[aux8[1]] << 16);
        v2 = iq2kl_values[aux8[2]] | (iq2kl_values[aux8[3]] << 16);
        sumi2 = ggml_cuda_dp4a(v1, q8h[2*i+0], ggml_cuda_dp4a(v2, q8h[2*i+1], sumi2));
    }

    auto sh = bq2->scales_h >> 4*ib64;
    int ls1 = int(((bq2->scales_l[(2*ib64+0)%4] >> 4*(ib64/2)) & 0xf) | ((sh << 4) & 0x30)) - 32;
    int ls2 = int(((bq2->scales_l[(2*ib64+1)%4] >> 4*(ib64/2)) & 0xf) | ((sh << 2) & 0x30)) - 32;

    *result += d * (__low2float(bq8_1[2*ib64+0].ds) * ls1 * sumi1 + __low2float(bq8_1[2*ib64+1].ds) * ls2 * sumi2);

}

__device__ __forceinline__ void vec_dot_iq3_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs, float * result) {

    float d = __half2float(*(const half *)vbq);
    const block_iq3_ks * bq3 = (const block_iq3_ks *)((const char *)vbq + sizeof(half)) + kbx;

    int iqs = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1. 0 works on quants 0...127, 1 on quants 128...255
                              // Each thread processes 8 quants in each of the 4 32-blocks
    const int il8   = iqs%4;  // 0...3. 0 works on quants 0...7, 1 on quants 8...15, 2 on 16...23, 3 on 24...31

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    uint16_t extra = bq3->extra >> 4*ib128;
    uint32_t extra_v = uint32_t(extra >> 8) * 0x01010101;

    uint32_t extra32_1 = ((extra_v << 3) & 0x08080808) | ((extra_v << 5) & 0x80808080);
    uint32_t extra32_2 = ((extra_v << 2) & 0x08080808) | ((extra_v << 4) & 0x80808080);

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
    for (int i = 0; i < 2; ++i) {
        uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
        uint32_t vh = ((qh[2*i+0] | (qh[2*i+1] << 16)) >> 4*ib128);

        uint32_t val1 = ((vl >> 0) & 0x33333333) | extra32_1 | ((vh << 2) & 0x04040404) | ((vh << 4) & 0x40404040);
        uint32_t val2 = ((vl >> 2) & 0x33333333) | extra32_2 | ((vh << 1) & 0x04040404) | ((vh << 3) & 0x40404040);
        int2 v1 = get_int_from_table_16(val1, iq3nl_values);
        int2 v2 = get_int_from_table_16(val2, iq3nl_values);

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        sumi[0] = ggml_cuda_dp4a(v1.x, q8[i], sumi[0]);

        q8 += sizeof(block_q8_1)/4;
        sumi[1] = ggml_cuda_dp4a(v2.x, q8[i], sumi[1]);

        q8 += sizeof(block_q8_1)/4;
        sumi[2] = ggml_cuda_dp4a(v1.y, q8[i], sumi[2]);

        q8 += sizeof(block_q8_1)/4;
        sumi[3] = ggml_cuda_dp4a(v2.y, q8[i], sumi[3]);
    }
    const uint16_t * sl16 = (const uint16_t *)bq3->scales;
    int32_t aux32 = __vsub4(((sl16[0] | (sl16[1] << 16)) >> 4*ib128) & 0x0f0f0f0f, 0x10101010);
    const int8_t * a8 = (const int8_t *)&aux32;
    *result += d * (__low2float(bq8_1[4*ib128+0].ds) * (a8[0] + ((extra << 4) & 0x10)) * sumi[0] +
                    __low2float(bq8_1[4*ib128+1].ds) * (a8[1] + ((extra << 3) & 0x10)) * sumi[1] +
                    __low2float(bq8_1[4*ib128+2].ds) * (a8[2] + ((extra << 2) & 0x10)) * sumi[2] +
                    __low2float(bq8_1[4*ib128+3].ds) * (a8[3] + ((extra << 1) & 0x10)) * sumi[3]);

}

__device__ __forceinline__ void vec_dot_iq1_bn_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    half d16; memcpy(&d16, vbq, sizeof(d16));
    float scale = d16;
    const block_iq1_bn * bq1 = (const block_iq1_bn *)((const char *)vbq + sizeof(d16)) + kbx;

    // iqs is 0 or 1

    int sumi = 0;
#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    uint16_t mult[2];
    mult[1] = iqs == 0 ? 27 : 3;
    mult[0] = mult[1] + (mult[1] << 1);
    const int * q8 = (const int *)bq8_1[iqs].qs;
    int val[4];
    for (int l = 0; l < 2; ++l) {
        int8_t * a = (int8_t *)val;
        const int i16 = 2*iqs + l;
        for (int k = 0; k < 3; ++k) {
            uint16_t q = bq1->ql[3*i16+k];
            for (int j = 4; j >= 0; --j) {
                uint16_t v = q & 0xff;
                v += v << 1;
                a[j] = v >> 8;
                q += q << 1;
            }
            a += 5;
        }
        uint16_t v = (mult[l]*bq1->extra) & 0xff;
        v += v << 1;
        *a = v >> 8;
        sumi = __dp4a(val[0], q8[4*l+0], __dp4a(val[1], q8[4*l+1], __dp4a(val[2], q8[4*l+2], __dp4a(val[3], q8[4*l+3], sumi))));
    }
    float2 d8 = __half22float2(bq8_1[iqs].ds);
    *result += scale * (d8.x * sumi - d8.y);
#else
    static const uint16_t k_mult[5] = {81, 27, 9, 3, 1};
    const int8_t * q8 = bq8_1[iqs].qs;
    for (int l = 0; l < 2; ++l) {
        const int i16 = 2*iqs + l;
        for (int k = 0; k < 3; ++k) {
            uint8_t q = bq1->ql[3*i16+k];
            for (int j = 0; j < 5; ++j) {
                uint8_t v = k_mult[j]*q;
                int8_t vs = (v + (v >> 1)) >> 7;
                sumi += q8[j]*(vs - 1);
            }
            q8 += 5;
        }
        uint8_t v = k_mult[i16]*bq1->extra;
        int8_t vs = (v + (v >> 1)) >> 7;
        sumi += q8[0]*(vs - 1);
        q8++;
    }
    *result += scale * __low2float(bq8_1[iqs].ds) * sumi;
#endif
}

__device__ __forceinline__ void vec_dot_iq2_bn_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    float scale = *(const float *)vbq;
    const block_iq2_bn * bq2 = (const block_iq2_bn *)((const char *)vbq + sizeof(float)) + kbx;

    // iqs is 0 or 1

#if __CUDA_ARCH__ >= MIN_CC_DP4A // lowest compute capability for integer intrinsics
    auto qs  = (const int *)bq2->qs + 2*iqs;
    auto q8l = (const int *)bq8_1[0].qs + 2*iqs;
    auto q8h = (const int *)bq8_1[1].qs + 2*iqs;
    int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
    for (int j = 0; j < 2; ++j) {
        int vl = qs[j];
        int vh = qs[j] >> 4;
        sumi1 = __dp4a(vl & 0x03030303, q8l[j+0], sumi1);
        sumi2 = __dp4a(vl & 0x0c0c0c0c, q8l[j+4], sumi2);
        sumi3 = __dp4a(vh & 0x03030303, q8h[j+0], sumi3);
        sumi4 = __dp4a(vh & 0x0c0c0c0c, q8h[j+4], sumi4);
    }
    auto d8l = __half22float2(bq8_1[0].ds);
    auto d8h = __half22float2(bq8_1[1].ds);
    *result += scale * (d8l.x * (sumi1 + 0.25f*sumi2) + d8h.x * (sumi3 + 0.25f * sumi4) - 0.5f*d8l.y - 0.5f*d8h.y);
#else
    int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
    auto q8l = bq8_1[0].qs + 8*iqs;
    auto q8h = bq8_1[1].qs + 8*iqs;
    auto qs  = bq2->qs + 8*iqs;
    for (int j = 0; j < 8; ++j) {
        sumi1 += q8l[j+ 0] * (qs[j] & 0x03);
        sumi2 += q8l[j+16] * (qs[j] & 0x0c);
        sumi3 += q8h[j+ 0] * (qs[j] & 0x30);
        sumi4 += q8h[j+16] * (qs[j] & 0xc0);
    }
    auto d8l = __half22float2(bq8_1[0].ds);
    auto d8h = __half22float2(bq8_1[1].ds);
    *result += scale * (d8l.x * (sumi1 + 0.25f*sumi2) + 0.0625f * d8h.x*(sumi3 + 0.25f*sumi4) - 0.5f*d8l.y - 0.5f*d8h.y);
#endif
}

} // namespace

static void mul_mat_vec_iq2_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_K, VDR_IQ2_K_Q8_1_MMVQ, vec_dot_iq2_k_q8_1>(args, stream);
}

static void mul_mat_vec_iq3_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_K, VDR_IQ3_K_Q8_1_MMVQ, vec_dot_iq3_k_q8_1>(args, stream);
}

static void mul_mat_vec_iq4_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_K, VDR_IQ4_K_Q8_1_MMVQ, vec_dot_iq4_k_q8_1>(args, stream);
}

static void mul_mat_vec_iq4_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_K_R4, 2, vec_dot_iq4_k_r4_q8_1, 4>(args, stream);
}

static void mul_mat_vec_iq4_ks_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KS_R4, 2, vec_dot_iq4_ks_r4_q8_1, 4>(args, stream);
}

static void mul_mat_vec_iq1_s_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ1_S_R4, 2, vec_dot_iq1_s_r4_q8_1, 4>(args, stream);
}

static void mul_mat_vec_iq1_m_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ1_M_R4, 2, vec_dot_iq1_m_r4_q8_1, 4>(args, stream);
}

static void mul_mat_vec_iq5_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ5_K_R4, 2, vec_dot_iq5_k_r4_q8_1, 4>(args, stream);
}

static void mul_mat_vec_iq5_ks_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ5_KS_R4, 2, vec_dot_iq5_ks_r4_q8_1, 4>(args, stream);
}

static void mul_mat_vec_iq2_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_K_R4, 2, vec_dot_iq2_k_r4_q8_1, 4>(args, stream);
}

static void mul_mat_vec_iq3_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_K_R4, 2, vec_dot_iq3_k_r4_q8_1, 4>(args, stream);
}

static void mul_mat_vec_iq4_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KS, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq4_ks_q8_1>(args, stream);
}

static void mul_mat_vec_iq2_kl_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_KL, VDR_IQ3_K_Q8_1_MMVQ, vec_dot_iq2_kl_q8_1>(args, stream);
}

static void mul_mat_vec_iq3_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_KS, VDR_IQ3_K_Q8_1_MMVQ, vec_dot_iq3_ks_q8_1>(args, stream);
}

static void mul_mat_vec_iq4_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq4_kt_q8_1>(args, stream);
}

static void mul_mat_vec_iq1_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ1_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq1_kt_q8_1>(args, stream);
}

static void mul_mat_vec_iq2_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq2_kt_q8_1>(args, stream);
}

static void mul_mat_vec_iq3_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq3_kt_q8_1>(args, stream);
}

static void mul_mat_vec_iq4_kss_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KSS, VDR_IQ4_KSS_Q8_1_MMVQ, vec_dot_iq4_kss_q8_1>(args, stream);
}

static void mul_mat_vec_iq2_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_KS, VDR_IQ2_KS_Q8_1_MMVQ, vec_dot_iq2_ks_q8_1>(args, stream);
}

static void mul_mat_vec_iq5_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ5_K, VDR_IQ5_K_Q8_1_MMVQ, vec_dot_iq5_k_q8_1>(args, stream);
}

static void mul_mat_vec_iq5_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ5_KS, VDR_IQ5_K_Q8_1_MMVQ, vec_dot_iq5_ks_q8_1>(args, stream);
}

static void mul_mat_vec_iq6_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ6_K, VDR_IQ6_K_Q8_1_MMVQ, vec_dot_iq6_k_q8_1>(args, stream);
}

static void mul_mat_vec_iq1_bn_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ1_BN, 1, vec_dot_iq1_bn_q8_1>(args, stream);
}

static void mul_mat_vec_iq2_bn_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_BN, 1, vec_dot_iq2_bn_q8_1>(args, stream);
}

void iqk_mul_mat_vec_q(ggml_type type, const mmvq_args & args, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_IQ1_BN:
            mul_mat_vec_iq1_bn_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_BN:
            mul_mat_vec_iq2_bn_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_K:
            mul_mat_vec_iq2_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_K:
            mul_mat_vec_iq3_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_KL:
            mul_mat_vec_iq2_kl_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_KS:
            mul_mat_vec_iq3_ks_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_K:
            mul_mat_vec_iq4_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_KS:
            mul_mat_vec_iq4_ks_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_KSS:
            mul_mat_vec_iq4_kss_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ1_KT:
            mul_mat_vec_iq1_kt_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_KT:
            mul_mat_vec_iq2_kt_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_KT:
            mul_mat_vec_iq3_kt_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_KT:
            mul_mat_vec_iq4_kt_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_KS:
            mul_mat_vec_iq2_ks_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ5_K:
            mul_mat_vec_iq5_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ5_KS:
            mul_mat_vec_iq5_ks_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ6_K:
            mul_mat_vec_iq6_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_K_R4:
            mul_mat_vec_iq2_k_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_K_R4:
            mul_mat_vec_iq3_k_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_K_R4:
            mul_mat_vec_iq4_k_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_KS_R4:
            mul_mat_vec_iq4_ks_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ5_K_R4:
            mul_mat_vec_iq5_k_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ5_KS_R4:
            mul_mat_vec_iq5_ks_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ1_S_R4:
            mul_mat_vec_iq1_s_r4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ1_M_R4:
            mul_mat_vec_iq1_m_r4_q8_1_cuda(args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
