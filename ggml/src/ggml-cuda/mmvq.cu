//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "mmvq.cuh"
#include "iqk_mmvq.cuh"
#include "vecdotq.cuh"

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
        case GGML_TYPE_IQ4_XS  : return VDR_IQ4_XS_Q8_1_MMVQ;
        default                : return 1;
    }
}

template <ggml_type type, int ncols_y, int nwarps>
static __device__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
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

template <ggml_type type, int ncols_y, int nwarps>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst, const char * __restrict__ ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst,
    const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0) {
    int i2 = blockIdx.y;
    char * cdst = (char *)dst + i2*nb2;
    int i02 = ids_data ? *(const int *)(ids_data + i2*ids_nb0) : i2;
    if (i02 < 0) {
        // We clear the buffer via cudaMemset instead
//#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
//        constexpr int rows_per_cuda_block = 1;
//#else
//        constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;
//#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)
//        const int row0 = rows_per_cuda_block*blockIdx.x;
//        if (threadIdx.y == 0) {
//            dst = (float *)cdst;
//            for (int j = 0; j < ncols_y; ++j) {
//                if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || row0 + threadIdx.x < nrows_dst)) {
//                    dst[j*nrows_dst + row0 + threadIdx.x] = 0;
//                }
//            }
//        }
        return;
    }
    const char * cx = (const char *)vx + i02*nb02;
    const char * cy = (const char *)vy + i2*nb12;
    mul_mat_vec_q<type, ncols_y, nwarps>(cx, cy, (float *)cdst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

template <ggml_type type, int nwarps>
static void mul_mat_vec_q_cuda_T(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_y <= MMVQ_MAX_BATCH_SIZE);

    int id = ggml_cuda_get_device();

    int64_t rows_per_cuda_block = ggml_cuda_info().devices[id].cc < CC_RDNA2 ?
        ncols_y < 4 ? 1 : 2 : 1;

    //if (ggml_cuda_info().devices[id].cc < CC_RDNA2) { // NVIDIA and AMD older than RDNA2
    //    switch(ncols_y) {
    //        case 1:
    //            nwarps = 4;
    //            rows_per_cuda_block = 1;
    //            break;
    //        case 2:
    //        case 3:
    //        case 4:
    //            nwarps = 4;
    //            rows_per_cuda_block = 2;
    //            break;
    //        case 5:
    //        case 6:
    //        case 7:
    //        case 8:
    //            nwarps = 2;
    //            rows_per_cuda_block = 2;
    //            break;
    //        default:
    //            GGML_ABORT("fatal error");
    //            break;
    //    }
    //}
    const int64_t nblocks = (nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, ne2, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    switch (ncols_y) {
        case 1:
            mul_mat_vec_q<type, 1, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 2:
            mul_mat_vec_q<type, 2, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 3:
            mul_mat_vec_q<type, 3, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 4:
            mul_mat_vec_q<type, 4, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 5:
            mul_mat_vec_q<type, 5, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 6:
            mul_mat_vec_q<type, 6, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 7:
            mul_mat_vec_q<type, 7, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        case 8:
            mul_mat_vec_q<type, 8, nwarps><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, nrows_dst, nb02, nb12, nb2, ids_nb0);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template <ggml_type type>
static void mul_mat_vec_q_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0, cudaStream_t stream) {
    int nwarps = 1;
    int id = ggml_cuda_get_device();
    if (ne2 < 2 && ggml_cuda_info().devices[id].cc < CC_RDNA2) { // NVIDIA and AMD older than RDNA2
        nwarps = ncols_y <= 4 ? 4 : 2;
    }
    switch (nwarps) {
        case 1:
            mul_mat_vec_q_cuda_T<type, 1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst,
                    ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case 2:
            mul_mat_vec_q_cuda_T<type, 2>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst,
                    ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        default:
            mul_mat_vec_q_cuda_T<type, 4>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst,
                    ne2, nb02, nb12, nb2, ids_nb0, stream);
    }
}

static void mul_mat_vec_q4_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q4_0>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q4_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q4_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q5_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q5_0>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q5_1_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q5_1>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q6_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q6_0>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q8_0_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q8_0>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q2_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q2_K>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q3_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q3_K>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q4_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q4_K>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q5_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q5_K>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_q6_K_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_Q6_K>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq2_xxs_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ2_XXS>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq2_xs_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ2_XS>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq2_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ2_S>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq3_xxs_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ3_XXS>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq1_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ1_S>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq1_m_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ1_M>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq4_nl_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ4_NL>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq4_xs_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ4_XS>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void mul_mat_vec_iq3_s_q8_1_cuda(
    const void * vx, const void * vy, float * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, int64_t ids_nb0, cudaStream_t stream) {

    mul_mat_vec_q_cuda<GGML_TYPE_IQ3_S>(vx, vy, dst, ids_data, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
}

static void ggml_cuda_op_mul_mat_vec_q_impl(ggml_backend_cuda_context & ctx, ggml_type type,
        const int64_t ne00, const int64_t ne0, const int64_t ne2,
        const int64_t nb02, const int64_t nb12, const int64_t nb2, const int64_t ids_nb0,
        const char * src0_dd_i, const char * src1_ddq_i, float * dst_dd_i, const char * ids_data,
        const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
        const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    switch (type) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q4_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q5_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q5_1_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q6_0:
            mul_mat_vec_q6_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q2_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q3_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q4_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q5_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q6_K_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_iq2_xxs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_iq2_xs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_iq2_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_iq3_xxs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_iq1_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_iq1_m_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ1_BN:
            mul_mat_vec_iq1_bn_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_BN:
            mul_mat_vec_iq2_bn_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_iq4_nl_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_iq4_xs_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_K:
            mul_mat_vec_iq2_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_K:
            mul_mat_vec_iq3_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_KL:
            mul_mat_vec_iq2_kl_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_KS:
            mul_mat_vec_iq3_ks_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_K:
            mul_mat_vec_iq4_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_KS:
            mul_mat_vec_iq4_ks_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_KSS:
            mul_mat_vec_iq4_kss_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ1_KT:
            mul_mat_vec_iq1_kt_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_KT:
            mul_mat_vec_iq2_kt_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_KT:
            mul_mat_vec_iq3_kt_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_KT:
            mul_mat_vec_iq4_kt_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_KS:
            mul_mat_vec_iq2_ks_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,  ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ5_K:
            mul_mat_vec_iq5_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ5_KS:
            mul_mat_vec_iq5_ks_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ6_K:
            mul_mat_vec_iq6_k_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_iq3_s_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ2_K_R4:
            mul_mat_vec_iq2_k_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_K_R4:
            mul_mat_vec_iq3_k_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_K_R4:
            mul_mat_vec_iq4_k_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_KS_R4:
            mul_mat_vec_iq4_ks_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ5_K_R4:
            mul_mat_vec_iq5_k_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ5_KS_R4:
            mul_mat_vec_iq5_ks_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ1_S_R4:
            mul_mat_vec_iq1_s_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ1_M_R4:
            mul_mat_vec_iq1_m_r4_q8_1_cuda(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst,   ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

}

void ggml_cuda_op_mul_mat_vec_q_3D(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);
    GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src0->ne[2] == src1->ne[2] && src0->ne[2] == dst->ne[2]);

    const int64_t ne0 = dst->ne[0];

    const int64_t src1_row_size = ggml_row_size(GGML_TYPE_Q8_1, src1_padded_row_size);

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, dst->ne[2],
        src0->nb[2], src1_row_size, dst->nb[2], 0,
        src0_dd_i, src1_ddq_i, dst_dd_i, nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, stream);

    GGML_UNUSED(src1_ddf_i);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, 1, 0, 0, 0, 0,
        src0_dd_i, src1_ddq_i, dst_dd_i, nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, stream);

    GGML_UNUSED(src1_ddf_i);
}

void ggml_cuda_op_mul_mat_vec_q_id(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);
    GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src1->ne[1] == 1 && src1->ne[2] == 1);
    GGML_ASSERT(ids->ne[0] == dst->ne[2]);

    const int64_t ne0 = dst->ne[0];

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, dst->ne[2],
        src0->nb[2], src1->nb[2], dst->nb[2], ids->nb[0],
        src0_dd_i, src1_ddq_i, dst_dd_i, (const char *)ids->data,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, stream);

    GGML_UNUSED(src1_ddf_i);
}

bool ggml_cuda_mmvq_type_supported(ggml_type src0_type) {
    switch (src0_type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_BN:
        case GGML_TYPE_IQ2_BN:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_K_R4:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_K_R4:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
            return true;
        default:
            return false;
    }
}
