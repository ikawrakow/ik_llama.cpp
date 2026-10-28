#include "indexer_topk.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "convert.cuh"
#include "argsort.cuh"
#include "mma_new.cuh"

template <typename mask_t>
static __device__ __forceinline__ float indexer_mask_value(const mask_t * m, int col) {
    return (float)m[col];
}

template <>
__device__ __forceinline__ float indexer_mask_value(const int * m, int col) {
    return col < *m ? 0.0f : -INFINITY;
}

template <typename kq_t, typename mask_t>
static __global__ void k_fused_relu_mul_sum_rows(const kq_t * __restrict__ kq, const float * __restrict__ w, const mask_t * __restrict__ m, float * __restrict__ dst, const int ncols, const int nhead, size_t nbm) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    int64_t step = ncols*nhead;
    auto this_w  = w + blockIdx.x*nhead;
    auto this_m  = (const mask_t *)((const char *)m + nbm*row);

    for (int i = col; i < ncols; i += blockDim.x) {
        float sum = indexer_mask_value(this_m, i);
        auto this_kq = kq + blockIdx.x * step;
        for (int head = 0; head < nhead; ++head) {
            float relu = (float)this_kq[i];
            relu = relu > 0.0f ? relu : 0.0f;
            sum += relu * this_w[head];
            this_kq += ncols;
        }
        dst[ncols*row + i] = sum;
    }
}

template <typename kq_t, typename mask_t>
static __global__ void k_fused_relu_mul_sum_rows_2(const kq_t * __restrict__ kq, const float * __restrict__ w, const mask_t * __restrict__ m, float * __restrict__ dst, const int ncols, const int nhead, size_t nbm) {
    const int row = blockIdx.x;
    const int col = blockIdx.y*blockDim.x + threadIdx.x;
    if (col >= ncols) {
        return;
    }

    int64_t step = ncols*nhead;
    auto this_w  = w + blockIdx.x*nhead;
    auto this_m  = (const mask_t *)((const char *)m + nbm*row);

    float sum = indexer_mask_value(this_m, col);
    auto this_kq = kq + row * step;
    for (int head = 0; head < nhead; ++head) {
        float relu = (float)this_kq[col];
        relu = relu > 0.0f ? relu : 0.0f;
        sum += relu * this_w[head];
        this_kq += ncols;
    }
    dst[ncols*row + col] = sum;
}

template <typename kq_t, typename mask_t>
static __global__ void k_fused_relu_mul_sum_rows_3(const kq_t * __restrict__ kq, const float * __restrict__ w, const mask_t * __restrict__ m,
        const int * __restrict__ cell_idx, float * __restrict__ dst, const int ncols, int ncols_kq, const int nhead, size_t nbm) {
    const int row = blockIdx.x;
    const int col = blockIdx.y*blockDim.x + threadIdx.x;
    if (col >= ncols) {
        return;
    }

    int64_t step = ncols_kq*nhead;
    auto this_w  = w + blockIdx.x*nhead;
    auto this_m  = (const mask_t *)((const char *)m + nbm*row);

    int col_kq = cell_idx[col];

    float sum = indexer_mask_value(this_m, col);
    auto this_kq = kq + row * step;
    for (int head = 0; head < nhead; ++head) {
        float relu = (float)this_kq[col_kq];
        relu = relu > 0.0f ? relu : 0.0f;
        sum += relu * this_w[head];
        this_kq += ncols_kq;
    }
    dst[ncols*row + col] = sum;
}

template <bool k_q8_0, typename mask_t>
static __global__ void __launch_bounds__(8*WARP_SIZE, 2) k_indexer_score(const half2 * __restrict__ q, const char * __restrict__ k,
        const float * __restrict__ w, const mask_t * __restrict__ m, float * __restrict__ dst,
        const size_t nbk1, const size_t nbm, const int ncols, const int nrows, const int nhead, const int rows_per_block) {
#ifdef INT8_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    typedef tile<16, 8, half2> tile_A;
    typedef tile< 8, 8, half2> tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int nwarps   = 8;
    constexpr int D2       = 64;
    constexpr int D2_pad   = D2 + 4;
    constexpr int nc_warp  = WARP_SIZE;
    constexpr int nc_pad   = nc_warp + 8;
    constexpr int max_head = 64;

    __shared__ half2 q_s[max_head][D2_pad];
    __shared__ float kq_s[nwarps][tile_A::I][nc_pad];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*WARP_SIZE + lane;
    const int col0 = blockIdx.x*nwarps*nc_warp + warp*nc_warp;

    for (int i = nhead*D2_pad + tid; i < max_head*D2_pad; i += nwarps*WARP_SIZE) {
        ((half2 *)q_s)[i] = make_half2(0.0f, 0.0f);
    }

    tile_B B[nc_warp/tile_B::I][D2/tile_B::J];
#pragma unroll
    for (int n = 0; n < nc_warp/tile_B::I; ++n) {
        const int col = col0 + n*tile_B::I + tile_B::get_i(0);
        if (col >= ncols) {
            continue;
        }
        const char * k_row = k + col*nbk1;
#pragma unroll
        for (int ks = 0; ks < D2/tile_B::J; ++ks) {
#pragma unroll
            for (int l = 0; l < tile_B::ne; ++l) {
                const int i2 = ks*tile_B::J + tile_B::get_j(l);
                if constexpr (k_q8_0) {
                    const block_q8_0 * b = (const block_q8_0 *)k_row + i2/(QK8_0/2);
                    const char2 qs = ((const char2 *)b->qs)[i2 % (QK8_0/2)];
                    B[n][ks].x[l] = __hmul2(make_half2(qs.x, qs.y), __half2half2(b->d));
                } else {
                    B[n][ks].x[l] = ((const half2 *)k_row)[i2];
                }
            }
        }
    }

    const int col = col0 + lane;
    const int row_end = min(nrows, (blockIdx.y + 1)*rows_per_block);
    for (int row = blockIdx.y*rows_per_block; row < row_end; ++row) {
        __syncthreads();
        const int4 * q_row = (const int4 *)(q + (size_t)row*nhead*D2);
        for (int i = tid; i < nhead*D2/4; i += nwarps*WARP_SIZE) {
            *(int4 *)&q_s[i/(D2/4)][4*(i%(D2/4))] = q_row[i];
        }
        __syncthreads();

        auto this_m = (const mask_t *)((const char *)m + nbm*row);
        float sum = col < ncols ? indexer_mask_value(this_m, col) : 0.0f;
        if constexpr (std::is_same_v<mask_t, int>) {
            if (col0 >= *this_m) {
                if (col < ncols) {
                    dst[(size_t)ncols*row + col] = sum;
                }
                continue;
            }
        }
        const float * this_w = w + nhead*row;
        for (int h0 = 0; h0 < nhead; h0 += tile_A::I) {
            tile_C C[nc_warp/tile_B::I];
#pragma unroll
            for (int ks = 0; ks < D2/tile_A::J; ++ks) {
                tile_A A;
                load_ldmatrix(A, &q_s[h0][ks*tile_A::J], D2_pad);
#pragma unroll
                for (int n = 0; n < nc_warp/tile_B::I; ++n) {
                    mma(C[n], A, B[n][ks]);
                }
            }
#pragma unroll
            for (int n = 0; n < nc_warp/tile_B::I; ++n) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; l += 2) {
                    *(float2 *)&kq_s[warp][tile_C::get_i(l)][n*tile_B::I + tile_C::get_j(l)] = make_float2(C[n].x[l], C[n].x[l + 1]);
                }
            }
            __syncwarp();
            const int nh = min(tile_A::I, nhead - h0);
            for (int h = 0; h < nh; ++h) {
                float relu = kq_s[warp][h][lane];
                relu = relu > 0.0f ? relu : 0.0f;
                sum += relu * this_w[h0 + h];
            }
            __syncwarp();
        }
        if (col < ncols) {
            dst[(size_t)ncols*row + col] = sum;
        }
    }
#else
    GGML_UNUSED(q); GGML_UNUSED(k); GGML_UNUSED(w); GGML_UNUSED(m); GGML_UNUSED(dst);
    GGML_UNUSED(nbk1); GGML_UNUSED(nbm); GGML_UNUSED(ncols); GGML_UNUSED(nrows); GGML_UNUSED(nhead); GGML_UNUSED(rows_per_block);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template <typename mask_t>
static void indexer_score_cuda(const half2 * q, const ggml_tensor * k, const float * w, const mask_t * m,
        float * dst, size_t nbm, int ncols, int nrows, int nhead, cudaStream_t stream) {
    constexpr int nwarps = 8;
    constexpr int rows_per_block = 32;
    const dim3 grid((ncols + nwarps*WARP_SIZE - 1)/(nwarps*WARP_SIZE), (nrows + rows_per_block - 1)/rows_per_block, 1);
    const dim3 block(WARP_SIZE, nwarps, 1);
    if (k->type == GGML_TYPE_Q8_0) {
        k_indexer_score<true><<<grid, block, 0, stream>>>(q, (const char *)k->data, w, m, dst, k->nb[1], nbm, ncols, nrows, nhead, rows_per_block);
    } else {
        k_indexer_score<false><<<grid, block, 0, stream>>>(q, (const char *)k->data, w, m, dst, k->nb[1], nbm, ncols, nrows, nhead, rows_per_block);
    }
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void k_copy_topk(const int * __restrict__ sorted, int * dst, const int ncols, const int n_top_k) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    sorted += int64_t(ncols)*row;
    dst    += int64_t(n_top_k)*row;
    for (int i = col; i < n_top_k; i += blockDim.x) {
        dst[i] = sorted[i];
    }
}

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)

static __device__ __forceinline__ uint32_t topk_float_to_key(float x) {
    const uint32_t b = __float_as_uint(x) == 0x80000000u ? 0u : __float_as_uint(x);
    return b ^ ((uint32_t)((int32_t)b >> 31) | 0x80000000u);
}

template <int block_size>
static __device__ __forceinline__ int topk_block_exclusive_sum(int v, int * scratch, int & total) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    int sum = v;
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        const int t = __shfl_up_sync(0xffffffff, sum, offset, WARP_SIZE);
        if (lane >= offset) sum += t;
    }
    if (lane == WARP_SIZE - 1) scratch[warp] = sum;
    __syncthreads();
    int before = 0;
    total = 0;
#pragma unroll
    for (int w = 0; w < block_size/WARP_SIZE; ++w) {
        const int s = scratch[w];
        before += w < warp ? s : 0;
        total  += s;
    }
    __syncthreads();
    return before + sum - v;
}

// radix select of the k largest values per row, sorted by value (-0 equal to +0), lower column first on ties
template <int block_size>
static __global__ void k_topk_select(const float * __restrict__ x, int * __restrict__ dst, const int ncols, const int k, const int P) {
    constexpr int nbins = 256;
    constexpr int unroll = 4;
    extern __shared__ uint64_t sel[];

    const int tid = threadIdx.x;
    x   += int64_t(ncols)*blockIdx.x;
    dst += int64_t(k)*blockIdx.x;

    __shared__ int hist[nbins];
    __shared__ int scratch[block_size/WARP_SIZE];
    __shared__ uint32_t s_prefix;
    __shared__ int s_rank;

    uint32_t prefix = 0, prefix_mask = 0;
    int rank = k;
    for (int shift = 32 - 8; shift >= 0; shift -= 8) {
        for (int i = tid; i < nbins; i += block_size) hist[i] = 0;
        __syncthreads();
        for (int i0 = 0; i0 < ncols; i0 += unroll*block_size) {
            uint32_t key[unroll];
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                const int i = i0 + u*block_size + tid;
                key[u] = i < ncols ? topk_float_to_key(x[i]) : 0;
            }
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                const int i = i0 + u*block_size + tid;
                if (i < ncols && (key[u] & prefix_mask) == prefix) atomicAdd(&hist[(key[u] >> shift) & (nbins - 1)], 1);
            }
        }
        __syncthreads();
        if (tid < WARP_SIZE) {
            int local[8], sum = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) { local[j] = hist[nbins - 1 - 8*tid - j]; sum += local[j]; }
            int incl = sum;
#pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                const int t = __shfl_up_sync(0xffffffff, incl, offset, WARP_SIZE);
                if (tid >= offset) incl += t;
            }
            int owner = incl >= rank ? tid : WARP_SIZE - 1;
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                owner = min(owner, __shfl_xor_sync(0xffffffff, owner, offset, WARP_SIZE));
            }
            if (tid == owner) {
                int above = incl - sum, j = 0;
                for (; j < 7 && above + local[j] < rank; ++j) above += local[j];
                s_prefix = prefix | ((uint32_t)(nbins - 1 - 8*tid - j) << shift);
                s_rank   = rank - above;
            }
        }
        __syncthreads();
        prefix = s_prefix;
        rank   = s_rank;
        prefix_mask |= (uint32_t)(nbins - 1) << shift;
    }

    // prefix is now the key of the smallest selected value, and rank values equal to it are taken
    for (int i = k + tid; i < P; i += block_size) sel[i] = 0;
    int n_eq = 0, n_out = 0;
    for (int i0 = 0; i0 < ncols; i0 += unroll*block_size) {
        uint32_t key[unroll];
#pragma unroll
        for (int u = 0; u < unroll; ++u) {
            const int i = i0 + u*block_size + tid;
            key[u] = i < ncols ? topk_float_to_key(x[i]) : 0;
        }
#pragma unroll
        for (int u = 0; u < unroll; ++u) {
            const int i = i0 + u*block_size + tid;
            const bool gt = i < ncols && key[u] >  prefix;
            const bool eq = i < ncols && key[u] == prefix;
            int total;
            const int before = topk_block_exclusive_sum<block_size>((int(gt) << 16) | int(eq), scratch, total);
            const int eq_left = max(rank - n_eq, 0);
            if (gt || (eq && (before & 0xffff) < eq_left)) {
                sel[n_out + (before >> 16) + min(before & 0xffff, eq_left)] = (uint64_t(key[u]) << 32) | uint32_t(~i);
            }
            n_out += (total >> 16) + min(total & 0xffff, eq_left);
            n_eq  += total & 0xffff;
        }
    }

    for (int size = 2; size <= P; size *= 2) {
        for (int stride = size/2; stride > 0; stride /= 2) {
            __syncthreads();
            for (int i = tid; i < P/2; i += block_size) {
                const int lo = 2*i - (i & (stride - 1));
                const int hi = lo + stride;
                const uint64_t a = sel[lo], b = sel[hi];
                if ((a < b) == ((lo & size) == 0)) { sel[lo] = b; sel[hi] = a; }
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < k; i += block_size) dst[i] = ~uint32_t(sel[i]);
}

#endif

static void indexer_select_top_k(ggml_backend_cuda_context & ctx, const float * score, int * sorted, int * dst,
        int ncols, int nrows, int n_top_k) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
    constexpr int max_sorted = 4096; // P*8 bytes of shared memory
    int P = 1;
    while (P < n_top_k) P *= 2;
    // a single row is faster on the CUB radix sort
    if (nrows > 1 && P <= max_sorted) {
        k_topk_select<512><<<nrows, 512, P*sizeof(uint64_t), ctx.stream()>>>(score, dst, ncols, n_top_k, P);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif
    argsort_f32_i32_cuda_cub(ctx.pool(), score, sorted, ncols, nrows, GGML_SORT_ORDER_DESC, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    k_copy_topk<<<nrows, 256, 0, ctx.stream()>>>(sorted, dst, ncols, n_top_k);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_indexer_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto op = ggml_unary_op(dst->op_params[0]);
    GGML_ASSERT(op == GGML_UNARY_OP_RELU);
    auto k = dst->src[0];
    auto q = dst->src[1];
    auto w = dst->src[2];
    auto m = dst->src[3];
    auto c = dst->src[4];
    int n_top_k = dst->ne[0];
    int n_kv    = m->type != GGML_TYPE_I32 ? m->ne[0] : c ? c->ne[0] : k->ne[1];
    int n_head  = q->ne[1];
    //if (k->type != GGML_TYPE_F16 && !ggml_is_quantized(k->type)) printf("%s: K is %s?\n", __func__, ggml_type_name(k->type));
    GGML_ASSERT(k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_F32 || ggml_is_quantized(k->type));
    GGML_ASSERT(k->ne[2] == 1 || k->ne[3] == 1);
    GGML_ASSERT(k->ne[0] == q->ne[0]);
    if (c) {
        GGML_ASSERT(c->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nrows(c) == 1);
        GGML_ASSERT(c->ne[0] == n_kv);
        //printf("Using cell_idx: k = %ld x %ld, c = %ld, n_kv = %d\n", k->ne[0], k->ne[1], c->ne[0], n_kv);
    } else {
        GGML_ASSERT(k->ne[1] > n_top_k);
        GGML_ASSERT(k->ne[1] == n_kv);
    }
    GGML_ASSERT(q->ne[2] == m->ne[1]);
    GGML_ASSERT(q->ne[1] == w->ne[0]);
    GGML_ASSERT(q->ne[2] == w->ne[1]);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(m->type == GGML_TYPE_F32 || m->type == GGML_TYPE_F16 || m->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(w));

    constexpr int k_block_size = 256;

    if ((k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) && q->type == GGML_TYPE_F32) {
        const bool use_mma = new_mma_available(ggml_cuda_info().devices[ctx.device].cc) && q->ne[0] == 128 && q->ne[1] <= 64 && !c;
        constexpr size_t k_max_buf_size = 1 << 28;
        size_t per_row = size_t(n_kv)*((use_mma ? 0 : q->ne[1]*sizeof(half)) + sizeof(int) + sizeof(float)) + q->ne[0]*q->ne[1]*sizeof(half);
        int max_rows = (k_max_buf_size + per_row - 1)/per_row;
        max_rows = std::min<int>(max_rows, q->ne[2]);
        int nstep = (q->ne[2] + max_rows - 1)/max_rows;

        ggml_cuda_pool_alloc<float> kq(ctx.pool());
        if (!use_mma) {
            kq.alloc(k->ne[1]*q->ne[1]*max_rows);
        }
        ggml_cuda_pool_alloc<float> score(ctx.pool(), int64_t(n_kv)*max_rows);
        ggml_cuda_pool_alloc<int>   sorted(ctx.pool(), int64_t(n_kv)*max_rows);
        ggml_cuda_pool_alloc<half>  q_f16(ctx.pool(), q->ne[0]*q->ne[1]*max_rows);
        ggml_cuda_pool_alloc<half>  k_f16(ctx.pool());

        auto k_data = (const half *)k->data;
        if (k->type == GGML_TYPE_Q8_0 && !use_mma) {
            k_f16.alloc(k->ne[0]*k->ne[1]);
            auto to_fp16_cuda = ggml_get_to_fp16_cuda(k->type);
            GGML_ASSERT(to_fp16_cuda);
            to_fp16_cuda(k->data, k_f16.get(), k->ne[0]*k->ne[1], 1, ctx.stream());
            CUDA_CHECK(cudaGetLastError());
            k_data = k_f16.get();
        }

        auto to_fp16_cuda = ggml_get_to_fp16_cuda(q->type);
        GGML_ASSERT(to_fp16_cuda);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(ctx.device), ctx.stream()));

        for (int istep = 0; istep < nstep; ++istep) {
            int first_row = max_rows*istep;
            if (first_row >= int(q->ne[2])) break;
            int last_row  = std::min(first_row + max_rows, int(q->ne[2]));
            int nrows     = last_row - first_row;

            to_fp16_cuda((const float *)q->data + q->ne[0]*q->ne[1]*first_row, q_f16.get(), q->ne[0]*q->ne[1]*nrows, 1, ctx.stream());
            CUDA_CHECK(cudaGetLastError());

            if (use_mma) {
                auto q_data = (const half2 *)q_f16.get();
                auto w_data = (const float *)w->data + first_row*q->ne[1];
                auto m_data = (const char *)m->data + first_row*m->nb[1];
                if (m->type == GGML_TYPE_F32) {
                    indexer_score_cuda(q_data, k, w_data, (const float *)m_data, score.get(), m->nb[1], n_kv, nrows, q->ne[1], ctx.stream());
                } else if (m->type == GGML_TYPE_I32) {
                    indexer_score_cuda(q_data, k, w_data, (const int   *)m_data, score.get(), m->nb[1], n_kv, nrows, q->ne[1], ctx.stream());
                } else {
                    indexer_score_cuda(q_data, k, w_data, (const half  *)m_data, score.get(), m->nb[1], n_kv, nrows, q->ne[1], ctx.stream());
                }
                indexer_select_top_k(ctx, score.get(), sorted.get(), (int *)((char *)dst->data + first_row*dst->nb[1]),
                        n_kv, nrows, dst->ne[0]);
                continue;
            }

            CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(ctx.device), CUBLAS_OP_T, CUBLAS_OP_N,
                    k->ne[1], q->ne[1]*nrows, q->ne[0],
                    &alpha, k_data,       CUDA_R_16F, k->ne[0],
                            q_f16.get(),  CUDA_R_16F, q->ne[0],
                    &beta,  kq.get(),     CUDA_R_32F, k->ne[1],
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            if (c) {
                int nblocks = (n_kv + k_block_size - 1)/k_block_size;
                dim3 grid(nrows, nblocks, 1);
                if (m->type == GGML_TYPE_F32) {
                    k_fused_relu_mul_sum_rows_3<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                            (const float *)w->data + first_row*q->ne[1],
                            (const float *)((const char *)m->data + first_row*m->nb[1]),
                            (const int *)c->data,
                            score.get(), n_kv, k->ne[1], q->ne[1], m->nb[1]);
                } else if (m->type == GGML_TYPE_I32) {
                    k_fused_relu_mul_sum_rows_3<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                            (const float *)w->data + first_row*q->ne[1],
                            (const int   *)((const char *)m->data + first_row*m->nb[1]),
                            (const int *)c->data,
                            score.get(), n_kv, k->ne[1], q->ne[1], m->nb[1]);
                } else {
                    k_fused_relu_mul_sum_rows_3<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                            (const float *)w->data + first_row*q->ne[1],
                            (const half  *)((const char *)m->data + first_row*m->nb[1]),
                            (const int *)c->data,
                            score.get(), n_kv, k->ne[1], q->ne[1], m->nb[1]);
                }
                CUDA_CHECK(cudaGetLastError());
            } else {
                int nblocks = (k->ne[1] + k_block_size - 1)/k_block_size;
                dim3 grid(nrows, nblocks, 1);
                if (m->type == GGML_TYPE_F32) {
                    k_fused_relu_mul_sum_rows_2<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                            (const float *)w->data + first_row*q->ne[1],
                            (const float *)((const char *)m->data + first_row*m->nb[1]),
                            score.get(), k->ne[1], q->ne[1], m->nb[1]);
                } else if (m->type == GGML_TYPE_I32) {
                    k_fused_relu_mul_sum_rows_2<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                            (const float *)w->data + first_row*q->ne[1],
                            (const int   *)((const char *)m->data + first_row*m->nb[1]),
                            score.get(), k->ne[1], q->ne[1], m->nb[1]);
                } else {
                    k_fused_relu_mul_sum_rows_2<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(),
                            (const float *)w->data + first_row*q->ne[1],
                            (const half  *)((const char *)m->data + first_row*m->nb[1]),
                            score.get(), k->ne[1], q->ne[1], m->nb[1]);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            indexer_select_top_k(ctx, score.get(), sorted.get(), (int *)((char *)dst->data + first_row*dst->nb[1]),
                    n_kv, nrows, dst->ne[0]);
        }

        return;

    }

    constexpr int64_t k_max_work_buffer_elements = 1 << 26;

    int max_rows = k_max_work_buffer_elements / n_kv / n_head;
    if (max_rows < 1) max_rows = 1;
    if (max_rows > q->ne[2]) max_rows = q->ne[2];

    int nstep = (q->ne[2] + max_rows - 1)/max_rows;

    ggml_cuda_pool_alloc<float> kq(ctx.pool(), int64_t(n_kv)*max_rows*n_head);
    ggml_cuda_pool_alloc<float> score(ctx.pool(), int64_t(n_kv)*max_rows);
    ggml_cuda_pool_alloc<int>   sorted(ctx.pool(), int64_t(n_kv)*max_rows);
    ggml_cuda_pool_alloc<float> k_f32(ctx.pool());
    ggml_cuda_pool_alloc<char>  q_converted(ctx.pool());
    const float * k_data = nullptr;
    int k_ld = k->ne[0];
    auto q_padded = GGML_PAD(q->ne[0], MATRIX_ROW_PADDING);
    if (ggml_is_quantized(k->type)) {
        auto nbytes_q = (size_t)(q_padded/QK8_1) * ((size_t) q->ne[1] * max_rows) * sizeof(block_q8_1);
        nbytes_q += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
        q_converted.alloc(nbytes_q);
    } else if (k->type == GGML_TYPE_F32) {
        k_data = (const float *) k->data;
        k_ld   = k->nb[1]/sizeof(float);
    } else {
        k_f32.alloc(k->ne[0]*k->ne[1]);
        ggml_get_to_fp32_cuda(k->type)(k->data, k_f32.get(), k->ne[1]*k->ne[0], 1, ctx.stream());
        CUDA_CHECK(cudaGetLastError());
        k_data = k_f32.get();
    }

    for (int istep = 0; istep < nstep; ++istep) {
        int first = istep*max_rows;
        int last  = std::min(first + max_rows, int(q->ne[2]));
        int nrows = last - first;
        auto q_data = (const char *)q->data + istep*max_rows*q->nb[2];
        auto m_data = (const char *)m->data + istep*max_rows*m->nb[1];
        auto w_data = (const float *)w->data + first*q->ne[1];
        if (ggml_is_quantized(k->type)) {
            quantize_mmq_q8_1_cuda((const float *)q_data, q_converted.get(), q->ne[0], q->ne[1]*nrows, 1, q_padded, k->type, ctx.stream());
            CUDA_CHECK(cudaGetLastError());
            mmq_args args{(const char *)k->data, q_converted.get(), kq.get(),
                k->ne[0], k->ne[1], int64_t(k->nb[1]),
                q_padded, q->ne[1]*nrows, q->ne[1]*nrows, k->ne[1]};
            ggml_cuda_op_mul_mat_q(ctx, k->type, args);
            CUDA_CHECK(cudaGetLastError());
        } else {
            // I wonder if it makes sense to use CUBLAS. If we did simple dot products we could fuse the
            // relu, mul, sum_rows all in one kernel, avoiding the k*q intermediate result.
            const float alpha = 1.0f;
            const float beta = 0.0f;
            CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(ctx.device), ctx.stream()));
            CUBLAS_CHECK(cublasSgemm(ctx.cublas_handle(ctx.device), CUBLAS_OP_T, CUBLAS_OP_N,
                    k->ne[1], q->ne[1]*nrows, q->ne[0],
                    &alpha,     k_data,                k_ld,
                       (const float *)q_data, q->ne[0],
                    &beta,      kq.get(),     k->ne[1]));
        }
        if (m->type == GGML_TYPE_F32) {
            k_fused_relu_mul_sum_rows<<<nrows, k_block_size, 0, ctx.stream()>>>(kq.get(), w_data, (const float *)m_data,
                    score.get(), k->ne[1], q->ne[1], m->nb[1]);
        } else if (m->type == GGML_TYPE_I32) {
            k_fused_relu_mul_sum_rows<<<nrows, k_block_size, 0, ctx.stream()>>>(kq.get(), w_data, (const int   *)m_data,
                    score.get(), k->ne[1], q->ne[1], m->nb[1]);
        } else {
            k_fused_relu_mul_sum_rows<<<nrows, k_block_size, 0, ctx.stream()>>>(kq.get(), w_data, (const half  *)m_data,
                    score.get(), k->ne[1], q->ne[1], m->nb[1]);
        }
        CUDA_CHECK(cudaGetLastError());

        indexer_select_top_k(ctx, score.get(), sorted.get(), (int *)((char *)dst->data + first*dst->nb[1]),
                k->ne[1], nrows, dst->ne[0]);
    }

}

template <typename mask_t>
static __global__ void k_indexer_mask(int ne0, int ne1, int ne2, int ntopk, int ne11,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb1,  size_t nb2,  size_t nb3,
        const mask_t * mask, const int * idx, mask_t * dst) {
    int i1 = blockIdx.x;
    int i3 = i1 / (ne1*ne2); i1 -= i3*ne1*ne2;
    int i2 = i1 / (ne1);     i1 -= i2*ne1;

    auto m = (const mask_t *)((const char *)mask + i1*nb01 + i2*nb02 + i3*nb03);
    auto i = (const int *)((const char *)idx + i1*nb11 + i2*nb12 + i3*nb13);
    auto d = (mask_t *)((char *)dst + i1*nb1 + i2*nb2 + i3*nb3);

    mask_t inf, zero;
    if constexpr (std::is_same_v<mask_t, half>) {
        inf  = __float2half(-INFINITY);
        zero = __float2half(0.0f);
    } else {
        inf  = -INFINITY;
        zero = 0.0f;
    }

    if (i1 < ne11) {
        for (int j = threadIdx.x; j < ne0;   j += blockDim.x) d[j] = inf;
        __syncthreads();
        for (int j = threadIdx.x; j < ntopk; j += blockDim.x) d[i[j]] = zero;
        __syncthreads();
        for (int j = threadIdx.x; j < ne0;   j += blockDim.x) d[j] += m[j];
    } else {
        for (int j = threadIdx.x; j < ne0;   j += blockDim.x) d[j] = m[j];
    }

}

template <typename mask_t>
static __global__ void k_visible_mask(int ne0, const int * n_visible, mask_t * dst) {
    const int n = n_visible[blockIdx.x];
    dst += int64_t(ne0)*blockIdx.x;
    for (int j = threadIdx.x; j < ne0; j += blockDim.x) dst[j] = j < n ? 0.0f : -INFINITY;
}

void ggml_cuda_op_indexer_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto mask = dst->src[0];
    auto topk = dst->src[1];
    if (mask->type == GGML_TYPE_I32) {
        if (dst->type == GGML_TYPE_F16) {
            k_visible_mask<<<ggml_nrows(dst), 256, 0, ctx.stream()>>>(dst->ne[0], (const int *)mask->data, (half *)dst->data);
        } else {
            k_visible_mask<<<ggml_nrows(dst), 256, 0, ctx.stream()>>>(dst->ne[0], (const int *)mask->data, (float *)dst->data);
        }
        return;
    }
    GGML_ASSERT(mask->ne[0] >= topk->ne[0]);
    GGML_ASSERT(mask->ne[1] >= topk->ne[1] && mask->ne[2] == topk->ne[2] && mask->ne[3] == topk->ne[3]);
    GGML_ASSERT(ggml_are_same_shape(mask, dst));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(topk->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == dst->type);

    int nrows = ggml_nrows(dst);

    if (dst->type == GGML_TYPE_F16) {
        k_indexer_mask<<<nrows, 256, 0, ctx.stream()>>>(dst->ne[0], dst->ne[1], dst->ne[2], topk->ne[0], topk->ne[1],
                mask->nb[1], mask->nb[2], mask->nb[3],
                topk->nb[1], topk->nb[2], topk->nb[3],
                dst->nb[1],  dst->nb[2],  dst->nb[3],
                (const half *)mask->data, (const int *)topk->data, (half *)dst->data);
    } else {
        k_indexer_mask<<<nrows, 256, 0, ctx.stream()>>>(dst->ne[0], dst->ne[1], dst->ne[2], topk->ne[0], topk->ne[1],
                mask->nb[1], mask->nb[2], mask->nb[3],
                topk->nb[1], topk->nb[2], topk->nb[3],
                dst->nb[1],  dst->nb[2],  dst->nb[3],
                (const float *)mask->data, (const int *)topk->data, (float *)dst->data);
    }

}

template <typename mask_t>
static __global__ void k_mask_to_index(int ne00, [[maybe_unused]] int ne0,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb1,  size_t nb2,  size_t nb3,
        const mask_t * __restrict__ mask, int * __restrict__ idx) {
    int i1 = blockIdx.x;
    int i2 = blockIdx.y;
    int i3 = blockIdx.z;

    mask_t zero;
    if constexpr (std::is_same_v<mask_t, half>) {
        zero = __float2half(0.0f);
    } else {
        zero = 0.0f;
    }
    __shared__ int counts[WARP_SIZE];
    const mask_t * mask_r = (const mask_t *)((const char *)mask + i1*nb01 + i2*nb02 + i3*nb03);
    int * idx_r = (int *)((char *)idx + i1*nb1 + i2*nb2 + i3*nb3);

    for (int j = threadIdx.x; j < ne0; j += WARP_SIZE) {
        idx_r[j] = -1;
    }

    int nOn = 0;
    for (int j = threadIdx.x; j < ne00; j += WARP_SIZE) {
        nOn += (mask_r[j] == zero ? 1 : 0);
    }
    counts[threadIdx.x] = nOn;
    __syncthreads();
    int cum[WARP_SIZE];
    int start = 0;
    for (int i = 0; i < WARP_SIZE; ++i) {
        cum[i] = start;
        start += counts[i];
    }
    start = cum[threadIdx.x];
    for (int j = threadIdx.x; j < ne00; j += WARP_SIZE) {
        if (mask_r[j] == zero) idx_r[start++] = j;
    }
}

// the list of k_mask_to_index for the mask concat(mask, visible top-k columns), in its order: column % WARP_SIZE, then column
template <typename mask_t>
static __global__ void k_mask_topk_to_index(int ne00, int ne0, int ntopk, int ntok, int P,
        size_t nb01, size_t nb11, size_t nb21, size_t nb1,
        const mask_t * __restrict__ mask, const int * __restrict__ topk, const char * __restrict__ n_visible, int * __restrict__ idx) {
    extern __shared__ uint32_t keys[];
    __shared__ int n_on;

    const int row = blockIdx.x;
    const mask_t * mask_r = (const mask_t *)((const char *)mask + row*nb01);
    int * idx_r = (int *)((char *)idx + row*nb1);

    mask_t zero;
    if constexpr (std::is_same_v<mask_t, half>) {
        zero = __float2half(0.0f);
    } else {
        zero = 0.0f;
    }

    if (threadIdx.x == 0) n_on = 0;
    for (int i = threadIdx.x; i < P; i += blockDim.x) keys[i] = 0xffffffff;
    __syncthreads();

    for (int j = threadIdx.x; j < ne00; j += blockDim.x) {
        if (mask_r[j] == zero) {
            const int s = atomicAdd(&n_on, 1);
            if (s < P) keys[s] = (uint32_t(j % WARP_SIZE) << 26) | uint32_t(j / WARP_SIZE);
        }
    }
    if (row < ntok) {
        const int * t = (const int *)((const char *)topk + row*nb11);
        const int n = *(const int *)(n_visible + row*nb21);
        for (int j = threadIdx.x; j < ntopk; j += blockDim.x) {
            if (t[j] < n) {
                const int c = ne00 + t[j];
                const int s = atomicAdd(&n_on, 1);
                if (s < P) keys[s] = (uint32_t(c % WARP_SIZE) << 26) | uint32_t(c / WARP_SIZE);
            }
        }
    }

    for (int size = 2; size <= P; size *= 2) {
        for (int stride = size/2; stride > 0; stride /= 2) {
            __syncthreads();
            for (int i = threadIdx.x; i < P/2; i += blockDim.x) {
                const int lo = 2*i - (i & (stride - 1));
                const int hi = lo + stride;
                const uint32_t a = keys[lo], b = keys[hi];
                if ((a > b) == ((lo & size) == 0)) { keys[lo] = b; keys[hi] = a; }
            }
        }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < ne0; j += blockDim.x) idx_r[j] = keys[j] == 0xffffffff ? -1 : int((keys[j] & 0x3ffffff)*WARP_SIZE + (keys[j] >> 26));
}

void ggml_cuda_op_mask_to_index(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    if (auto topk = dst->src[1]) {
        auto vis = dst->src[2];
        int P = 1;
        while (P < dst->ne[0]) P *= 2;
        if (src->type == GGML_TYPE_F16) {
            k_mask_topk_to_index<<<dst->ne[1], 256, P*sizeof(uint32_t), ctx.stream()>>>(src->ne[0], dst->ne[0], topk->ne[0], topk->ne[1], P,
                    src->nb[1], topk->nb[1], vis->nb[1], dst->nb[1],
                    (const half *)src->data, (const int *)topk->data, (const char *)vis->data, (int *)dst->data);
        } else {
            k_mask_topk_to_index<<<dst->ne[1], 256, P*sizeof(uint32_t), ctx.stream()>>>(src->ne[0], dst->ne[0], topk->ne[0], topk->ne[1], P,
                    src->nb[1], topk->nb[1], vis->nb[1], dst->nb[1],
                    (const float *)src->data, (const int *)topk->data, (const char *)vis->data, (int *)dst->data);
        }
        return;
    }
    GGML_ASSERT(src->ne[1] == dst->ne[1] && src->ne[2] == dst->ne[2] && src->ne[3] == dst->ne[3]);
    GGML_ASSERT(src->ne[0] >= dst->ne[0]);

    dim3 grid(dst->ne[1], dst->ne[2], dst->ne[3]);
    if (src->type == GGML_TYPE_F16) {
        k_mask_to_index<<<grid, WARP_SIZE, 0, ctx.stream()>>>(src->ne[0], dst->ne[0],
                src->nb[1], src->nb[2], src->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3],
                (const half *)src->data, (int *)dst->data);
    } else {
        k_mask_to_index<<<grid, WARP_SIZE, 0, ctx.stream()>>>(src->ne[0], dst->ne[0],
                src->nb[1], src->nb[2], src->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3],
                (const float *)src->data, (int *)dst->data);
    }

}
