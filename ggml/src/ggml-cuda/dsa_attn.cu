#include "dsa_attn.cuh"

static inline bool v_is_k_view(const ggml_tensor * K, const ggml_tensor * V) {
    if (!V || !V->data) return false;
    auto k_data = (const char *)K->data;
    auto v_data = (const char *)V->data;
    auto k_row_size = ggml_row_size(K->type, K->ne[0]);
    auto v_row_size = ggml_row_size(V->type, V->ne[0]);
    return v_data >= k_data && v_data + v_row_size <= k_data + k_row_size;
}

static __global__ void k_prepare_mask(int nidx, const int * __restrict__ idx, const half * __restrict__ m_in,
        half * __restrict__ m_out, size_t stride_idx, size_t stride_m) {
    int row = blockIdx.x;
    int col = blockIdx.y*blockDim.x + threadIdx.x;
    idx += row*stride_idx;
    m_out[row*nidx + col] = m_in[row*stride_m + idx[col]];
}

static __global__ void k_prepare_one_batch_kv(int nk, int ncol, const int * idx, const char * k_in,
        half * k_out, size_t stride_k, size_t stride_idx) {
    int row = blockIdx.y;
    int col = blockIdx.x;
    int i = idx[row*stride_idx + col];
    auto k_row = (const half *)(k_in + stride_k * i);
    k_out += (row*ncol + col)*nk;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        k_out[j] = k_row[j];
    }
}

static __global__ void k_prepare_one_batch_q(int ne0, int ne1, size_t nb1, size_t nb2,
        const float * q_in, half * q_out) {
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }
    int i1 = blockIdx.y;
    int i2 = blockIdx.z;
    q_out[i0 + (i2 + i1*ne1)*ne0] = __float2half(q_in[i0 + i1*nb1 + i2*nb2]);
}

static __global__ void k_copy_dst(int nelem, const half * kqv16, float * dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    dst[i] = __half2float(kqv16[i]);
}

template <int ncols_template, int block_size_template>
static __global__ void soft_max_f16_simple(half * x, const half * mask, const int ncols_par, const int nrows_y, const float scale) {
    const int ncols = ncols_template == 0 ? ncols_par : ncols_template;

    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;
    const int rowy = rowx / nrows_y; // broadcast the mask in the row dimension

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    extern __shared__ float data_soft_max_f32[];
    float * buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float * vals = buf_iw + WARP_SIZE;

    float max_val = -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const int64_t ix = (int64_t)rowx*ncols + col;
        const int64_t iy = (int64_t)rowy*ncols + col;

        const float val = scale*__half2float(x[ix]) + __half2float(mask[iy]);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        const int64_t ix = (int64_t)rowx*ncols + col;
        x[ix] = __float2half(vals[col] * inv_sum);
    }
}

#define CUDA_SOFT_MAX_BLOCK_SIZE 1024

static void soft_max_f16_cuda_simple(half * x, const half * mask, const int ncols_x, const int nrows_x,
        const int nrows_y, const float scale, cudaStream_t stream) {
    int nth = WARP_SIZE;
    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth,     1, 1);
    const dim3 block_nums(nrows_x, 1, 1);
    const size_t shmem = (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE)*sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");

    GGML_ASSERT(shmem < ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    switch (ncols_x) {
        case 32:
            soft_max_f16_simple<32, 32><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 64:
            soft_max_f16_simple<64, 64><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 128:
            soft_max_f16_simple<128, 128><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 256:
            soft_max_f16_simple<256, 256><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 512:
            soft_max_f16_simple<512, 512><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 1024:
            soft_max_f16_simple<1024, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 2048:
            soft_max_f16_simple<2048, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 4096:
            soft_max_f16_simple<4096, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        default:
            soft_max_f16_simple<0, 0><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
    }
}

bool ggml_cuda_dsa_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!dst) return false;

    constexpr int k_max_rows = 32;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sink = dst->src[4];
    const ggml_tensor * indexer = dst->src[5];

    if (sink) return false; // We do not support sinks at this point
    if (!Q || !K || !V || !mask || !indexer) return false;

    if (indexer->ne[0] % 256 != 0) return false; // lazyness to add checks and handle tailes in case of not multiple of 256
                                                 // But are there DSA variants where top_k is not a multiple of 256?
    if (K->ne[1] < 4*indexer->ne[0]) return false; // for efficiency
    if (K->ne[2] > 1 || K->ne[3] > 1 || mask->ne[2] > 1 || mask->ne[3] > 1 || Q->ne[3] > 1) return false;
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 || Q->type != GGML_TYPE_F32) return false;
    if (K->ne[0] != Q->ne[0]) return false;

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const half alpha = 1.0f;
    const half beta  = 0.0f;

    int max_rows = std::min<int>(Q->ne[1], k_max_rows);
    bool is_k_view = v_is_k_view(K, V);
    auto mask_size = indexer->ne[0]*Q->ne[1]; // mask is relatively small, so we can do it once for the whole calculation
    auto k_cache_size = indexer->ne[0]*K->ne[0]*max_rows;
    auto v_cache_size = indexer->ne[0]*V->ne[0]*max_rows;
    auto q_size = Q->ne[0]*Q->ne[2]*max_rows;
    auto kq_size = indexer->ne[0]*Q->ne[2]*max_rows;
    auto kqv_size = V->ne[0]*Q->ne[2]*max_rows;
    ggml_cuda_pool_alloc<half> q16(ctx.pool(), q_size);
    ggml_cuda_pool_alloc<half> kq16(ctx.pool(), kq_size);
    ggml_cuda_pool_alloc<half> kqv16(ctx.pool(), kqv_size);
    ggml_cuda_pool_alloc<half> mask16(ctx.pool(), mask_size);
    ggml_cuda_pool_alloc<half> k16(ctx.pool(), k_cache_size);
    ggml_cuda_pool_alloc<half> v16(ctx.pool());
    size_t v_offset = 0;
    if (is_k_view) {
        v_offset = (const half *)V->data - (const half *)K->data;
    } else {
        v16.alloc(v_cache_size);
    }
    auto stride_idx = indexer->nb[1]/sizeof(int);
    {
        dim3 grid(Q->ne[1], indexer->ne[0]/256, 1);
        k_prepare_mask<<<grid, 256, 0, ctx.stream()>>>(indexer->ne[0], (const int * )indexer->data,
                (const half *)mask->data, mask16.get(), stride_idx, mask->nb[1]/sizeof(half));
    }

    int nstep = (Q->ne[1] + max_rows - 1)/max_rows;

    for (int istep = 0; istep < nstep; ++istep) {
        int first = istep*max_rows;
        int last  = std::min<int>(first + max_rows, Q->ne[1]);
        int nrows = last - first;
        {
            dim3 grid(indexer->ne[0], nrows, 1);
            k_prepare_one_batch_kv<<<grid, 256, 0, ctx.stream()>>>(K->ne[0], indexer->ne[0],
                    (const int *)indexer->data + stride_idx*first,
                    (const char *)K->data, k16.get(), K->nb[1], stride_idx);
            if (!is_k_view) {
                k_prepare_one_batch_kv<<<grid, 256, 0, ctx.stream()>>>(V->ne[0], indexer->ne[0],
                        (const int *)indexer->data + stride_idx*first,
                        (const char *)V->data, v16.get(), V->nb[1], stride_idx);
            }
        }
        {
            int nblock = (Q->ne[0] + 255)/256;
            dim3 grid(nblock, nrows, Q->ne[2]);
            k_prepare_one_batch_q<<<grid, 256, 0, ctx.stream()>>>(Q->ne[0], Q->ne[2],
                    Q->nb[1]/sizeof(float), Q->nb[2]/sizeof(float),
                    (const float *)((const char *)Q->data + first*Q->nb[1]), q16.get());
        }

        CUBLAS_CHECK(cublasHgemmStridedBatched(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                    indexer->ne[0], Q->ne[2], Q->ne[0],
                    &alpha, k16.get(), K->ne[0], K->ne[0]*indexer->ne[0],
                    q16.get(), Q->ne[0], Q->ne[0]*Q->ne[2],
                    &beta, kq16.get(), indexer->ne[0], indexer->ne[0]*Q->ne[2], nrows));

        soft_max_f16_cuda_simple(kq16.get(), mask16.get() + first*indexer->ne[0], indexer->ne[0], Q->ne[2]*nrows,
                Q->ne[2], scale, ctx.stream());
        CUDA_CHECK(cudaGetLastError());

        if (is_k_view) {
            CUBLAS_CHECK(cublasHgemmStridedBatched(ctx.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                        V->ne[0], Q->ne[2], indexer->ne[0],
                        &alpha, k16.get() + v_offset, K->ne[0], K->ne[0]*indexer->ne[0],
                        kq16.get(), indexer->ne[0], indexer->ne[0]*Q->ne[2],
                        &beta, kqv16.get(), V->ne[0], V->ne[0]*Q->ne[2], nrows));
        } else {
            CUBLAS_CHECK(cublasHgemmStridedBatched(ctx.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                        V->ne[0], Q->ne[2], indexer->ne[0],
                        &alpha, v16.get(), V->ne[0], V->ne[0]*indexer->ne[0],
                        kq16.get(), indexer->ne[0], indexer->ne[0]*Q->ne[2],
                        &beta, kqv16.get(), V->ne[0], V->ne[0]*Q->ne[2], nrows));
        }

        {
            int nelem = V->ne[0]*Q->ne[2]*nrows;
            int nblock = (nelem + 255)/256;
            k_copy_dst<<<nblock, 256, 0, ctx.stream()>>>(nelem, kqv16.get(),
                    (float *)((char *)dst->data + dst->nb[2]*first));

        }

    }

    return true;
}
