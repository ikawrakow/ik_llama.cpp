#include "indexer_topk.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "convert.cuh"
#include "argsort.cuh"

template <typename kq_t, typename mask_t>
static __global__ void k_fused_relu_mul_sum_rows(const kq_t * __restrict__ kq, const float * __restrict__ w, const mask_t * __restrict__ m, float * __restrict__ dst, const int ncols, const int nhead, size_t nbm) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    int64_t step = ncols*nhead;
    auto this_w  = w + blockIdx.x*nhead;
    auto this_m  = (const mask_t *)((const char *)m + nbm*row);

    for (int i = col; i < ncols; i += blockDim.x) {
        float sum = (float)this_m[i];
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

    float sum = (float)this_m[col];
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

    float sum = (float)this_m[col];
    auto this_kq = kq + row * step;
    for (int head = 0; head < nhead; ++head) {
        float relu = (float)this_kq[col_kq];
        relu = relu > 0.0f ? relu : 0.0f;
        sum += relu * this_w[head];
        this_kq += ncols_kq;
    }
    dst[ncols*row + col] = sum;
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
    int n_kv    = m->ne[0];
    int n_head  = q->ne[1];
    //if (k->type != GGML_TYPE_F16 && !ggml_is_quantized(k->type)) printf("%s: K is %s?\n", __func__, ggml_type_name(k->type));
    GGML_ASSERT(k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_F32 || ggml_is_quantized(k->type));
    GGML_ASSERT(k->ne[2] == 1 || k->ne[3] == 1);
    GGML_ASSERT(k->ne[0] == q->ne[0]);
    if (c) {
        GGML_ASSERT(c->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nrows(c) == 1);
        GGML_ASSERT(c->ne[0] == m->ne[0]);
        //printf("Using cell_idx: k = %ld x %ld, c = %ld, n_kv = %d\n", k->ne[0], k->ne[1], c->ne[0], n_kv);
    } else {
        GGML_ASSERT(k->ne[1] > n_top_k);
        GGML_ASSERT(k->ne[1] == m->ne[0]);
    }
    GGML_ASSERT(q->ne[2] == m->ne[1]);
    GGML_ASSERT(q->ne[1] == w->ne[0]);
    GGML_ASSERT(q->ne[2] == w->ne[1]);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(m->type == GGML_TYPE_F32 || m->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(w));

    constexpr int k_block_size = 256;

    if ((k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_Q8_0) && q->type == GGML_TYPE_F32) {
        constexpr size_t k_max_buf_size = 1 << 28;
        size_t per_row = size_t(n_kv)*(q->ne[1]*sizeof(half) + sizeof(int) + sizeof(float)) + q->ne[0]*q->ne[1]*sizeof(half);
        int max_rows = (k_max_buf_size + per_row - 1)/per_row;
        max_rows = std::min<int>(max_rows, q->ne[2]);
        int nstep = (q->ne[2] + max_rows - 1)/max_rows;

        ggml_cuda_pool_alloc<float> kq(ctx.pool(), k->ne[1]*q->ne[1]*max_rows);
        ggml_cuda_pool_alloc<float> score(ctx.pool(), int64_t(n_kv)*max_rows);
        ggml_cuda_pool_alloc<int>   sorted(ctx.pool(), int64_t(n_kv)*max_rows);
        ggml_cuda_pool_alloc<half>  q_f16(ctx.pool(), q->ne[0]*q->ne[1]*max_rows);
        ggml_cuda_pool_alloc<half>  k_f16(ctx.pool());

        auto k_data = (const half *)k->data;
        if (k->type == GGML_TYPE_Q8_0) {
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
        const int ncols_sort = c ? n_kv : k->ne[1];
        if (c) {
            int nblocks = (ncols_sort + k_block_size - 1)/k_block_size;
            dim3 grid(nrows, nblocks, 1);
            if (m->type == GGML_TYPE_F32) {
                k_fused_relu_mul_sum_rows_3<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(), w_data, (const float *)m_data,
                        (const int *)c->data, score.get(), ncols_sort, k->ne[1], q->ne[1], m->nb[1]);
            } else {
                k_fused_relu_mul_sum_rows_3<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(), w_data, (const half  *)m_data,
                        (const int *)c->data, score.get(), ncols_sort, k->ne[1], q->ne[1], m->nb[1]);
            }
        } else if (m->type == GGML_TYPE_F32) {
            k_fused_relu_mul_sum_rows<<<nrows, k_block_size, 0, ctx.stream()>>>(kq.get(), w_data, (const float *)m_data,
                    score.get(), k->ne[1], q->ne[1], m->nb[1]);
        } else {
            k_fused_relu_mul_sum_rows<<<nrows, k_block_size, 0, ctx.stream()>>>(kq.get(), w_data, (const half  *)m_data,
                    score.get(), k->ne[1], q->ne[1], m->nb[1]);
        }
        CUDA_CHECK(cudaGetLastError());

        indexer_select_top_k(ctx, score.get(), sorted.get(), (int *)((char *)dst->data + first*dst->nb[1]),
                ncols_sort, nrows, dst->ne[0]);
        CUDA_CHECK(cudaGetLastError());
    }

}

template <typename mask_t>
static __global__ void k_indexer_mask(int ne0, int ne1, int ne2, int ntopk, int ne11,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb1,  size_t nb2,  size_t nb3,
        const mask_t * mask, const int * idx, const float * __restrict__ force, mask_t * dst) {
    int i1 = blockIdx.x;
    int i3 = i1 / (ne1*ne2); i1 -= i3*ne1*ne2;
    int i2 = i1 / (ne1);     i1 -= i2*ne1;

    auto m = (const mask_t *)((const char *)mask + i1*nb01 + i2*nb02 + i3*nb03);
    auto i = (const int *)((const char *)idx + i1*nb11 + i2*nb12 + i3*nb13);
    auto fv = force ? force + (size_t(i3)*ne1*ne2 + i2*ne1 + i1)*ne0 : nullptr;
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
        if (fv) {
            for (int j = threadIdx.x; j < ne0; j += blockDim.x) if (fv[j] > 0.5f) d[j] = zero;
        }
        __syncthreads();
        for (int j = threadIdx.x; j < ne0;   j += blockDim.x) d[j] += m[j];
    } else {
        for (int j = threadIdx.x; j < ne0;   j += blockDim.x) d[j] = m[j];
    }

}

void ggml_cuda_op_indexer_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto mask = dst->src[0];
    auto topk = dst->src[1];
    GGML_ASSERT(mask->ne[0] >= topk->ne[0]);
    GGML_ASSERT(mask->ne[1] >= topk->ne[1] && mask->ne[2] == topk->ne[2] && mask->ne[3] == topk->ne[3]);
    GGML_ASSERT(ggml_are_same_shape(mask, dst));
    GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
    GGML_ASSERT(topk->type == GGML_TYPE_I32);
    GGML_ASSERT(mask->type == dst->type);

    int nrows = ggml_nrows(dst);

    const float * force = nullptr;
    if (dst->src[2]) {
        GGML_ASSERT(dst->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(dst->src[2]));
        force = (const float *) dst->src[2]->data;
    }

    if (dst->type == GGML_TYPE_F16) {
        k_indexer_mask<<<nrows, 256, 0, ctx.stream()>>>(dst->ne[0], dst->ne[1], dst->ne[2], topk->ne[0], topk->ne[1],
                mask->nb[1], mask->nb[2], mask->nb[3],
                topk->nb[1], topk->nb[2], topk->nb[3],
                dst->nb[1],  dst->nb[2],  dst->nb[3],
                (const half *)mask->data, (const int *)topk->data, force, (half *)dst->data);
    } else {
        k_indexer_mask<<<nrows, 256, 0, ctx.stream()>>>(dst->ne[0], dst->ne[1], dst->ne[2], topk->ne[0], topk->ne[1],
                mask->nb[1], mask->nb[2], mask->nb[3],
                topk->nb[1], topk->nb[2], topk->nb[3],
                dst->nb[1],  dst->nb[2],  dst->nb[3],
                (const float *)mask->data, (const int *)topk->data, force, (float *)dst->data);
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

void ggml_cuda_op_mask_to_index(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto src = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
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
