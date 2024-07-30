#include "iqk_mmvq.cuh"

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

namespace {
template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y>
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__((ncols_y <= 4 ? 4 : 2)*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__global__ void iqk_mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int nwarps              = 1;
    constexpr int rows_per_cuda_block = 1;
#else
    constexpr int nwarps              = ncols_y <= 4 ? 4 : 2;
    constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;
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

template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
void iqk_mul_mat_vec_q_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    //GGML_ASSERT(ncols_y <= MMVQ_MAX_BATCH_SIZE);

    int id = ggml_cuda_get_device();

    int64_t nwarps = 1;
    int64_t rows_per_cuda_block = 1;

    if (ggml_cuda_info().devices[id].cc < CC_RDNA2) { // NVIDIA and AMD older than RDNA2
        switch(ncols_y) {
            case 1:
                nwarps = 4;
                rows_per_cuda_block = 1;
                break;
            case 2:
            case 3:
            case 4:
                nwarps = 4;
                rows_per_cuda_block = 2;
                break;
            case 5:
            case 6:
            case 7:
            case 8:
                nwarps = 2;
                rows_per_cuda_block = 2;
                break;
            default:
                GGML_ASSERT(false);
                break;
        }
    }
    const int64_t nblocks = (nrows_x + rows_per_cuda_block - 1) / rows_per_cuda_block;
    const dim3 block_nums(nblocks, 1, 1);
    const dim3 block_dims(WARP_SIZE, nwarps, 1);

    switch (ncols_y) {
        case 1:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 1><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 2:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 2><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 3:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 3><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 4:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 4><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 5:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 5><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 6:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 6><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 7:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 7><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
            break;
        case 8:
            iqk_mul_mat_vec_q<type, vdr, vec_dot_q_cuda, 8><<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
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

__device__ __forceinline__ float vec_dot_iq4_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

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
    return d * (sumi1 * ls1 + sumi2 * ls2);
}

#define VDR_IQ5_K_Q8_1_MMVQ 4
#define VDR_IQ5_K_Q8_1_MMQ  4

__device__ __forceinline__ int int_from_table(const uint8_t * a8, const uint8_t * values) {
    uint16_t v1 = values[a8[0]] | (values[a8[1]] << 8);
    uint16_t v2 = values[a8[2]] | (values[a8[3]] << 8);
    return v1 | (v2 << 16);
}

__device__ __forceinline__ float vec_dot_iq5_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {


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
    return d5 * (__low2float(bq8_1[2*(i4/2)+0].ds) * sumi1 * ls1 + __low2float(bq8_1[2*(i4/2)+1].ds) * sumi2 * ls2);
}

#define VDR_IQ2_K_Q8_1_MMVQ 4
#define VDR_IQ2_K_Q8_1_MMQ  4

// TODO
__device__ __forceinline__ float vec_dot_iq2_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    return 0;
//
//    const block_iq2_k * bq4 = (const block_iq2_k *) vbq + kbx;
//    const uint8_t * all_values = (const uint8_t *)iq4k_values;
//
//    // iqs is 0...28
//    const int ib32 = iqs/4;
//    // Why iqs/4 ?
//    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
//    const uint16_t * q4 = (const uint16_t *)bq4->qs + 8*ib32;
//    const uint16_t extra = bq4->extra >> 2*ib32;
//    int v1, v2;
//    int sumi1 = 0, sumi2 = 0;
//    for (int j = 0; j < 4; ++j) {
//        const uint32_t aux32 = q4[2*j+0] | (q4[2*j+1] << 16);
//        get_int_from_table_16_shift(aux32, extra, all_values, v1, v2);
//        sumi1 = ggml_cuda_dp4a(v1, q8[j+0], sumi1);
//        sumi2 = ggml_cuda_dp4a(v2, q8[j+4], sumi2);
//    }
//    const float d = __half2float(bq4->d) * __low2float(bq8_1[ib32].ds);
//    const uint8_t sh = bq4->scales_h[ib32/2] >> 4*(ib32%2);
//    const int ls1 = ((bq4->scales_l[ib32] & 0xf) | ((sh << 4) & 0x30)) - 32;
//    const int ls2 = ((bq4->scales_l[ib32] >>  4) | ((sh << 2) & 0x30)) - 32;
//    return d * (sumi1 * ls1 + sumi2 * ls2);
}

}

void mul_mat_vec_iq2_k_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_K, VDR_IQ2_K_Q8_1_MMVQ, vec_dot_iq2_k_q8_1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

void mul_mat_vec_iq4_k_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_K, VDR_IQ4_K_Q8_1_MMVQ, vec_dot_iq4_k_q8_1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

void mul_mat_vec_iq5_k_q8_1_cuda(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream) {

    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ5_K, VDR_IQ5_K_Q8_1_MMVQ, vec_dot_iq5_k_q8_1>(vx, vy, dst, ncols_x, nrows_x, nrows_y, ncols_y, nrows_dst, stream);
}

