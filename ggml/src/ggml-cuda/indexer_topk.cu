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

static __global__ void k_copy_topk(const int * __restrict__ sorted, int * dst, const int ncols, const int n_top_k) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    sorted += int64_t(ncols)*row;
    dst    += int64_t(n_top_k)*row;
    for (int i = col; i < n_top_k; i += blockDim.x) {
        dst[i] = sorted[i];
    }
}

template <typename mask_t>
static __global__ void k_fused_gemm_relu_mul_sum_rows(const half * __restrict__ k, const float * __restrict__ q,
        const float * __restrict__ w, const mask_t * __restrict__ m, float * __restrict__ dst,
        int head_size, int n_kv, int n_head, size_t nbm) {
    const int row_k = blockIdx.x;
    const int row_q = blockIdx.y;
    const int col = threadIdx.x;

    k += size_t(row_k) * head_size;
    q += size_t(row_q) * head_size * n_head;
    w += size_t(row_q) * n_head;
    auto this_m = (const mask_t *)((const char *)m + nbm*row_q);

    auto k2 = (const half2  *)k;
    auto q2 = (const float2 *)q;
    float score = (float)this_m[row_k];
    for (int head = 0; head < n_head; ++head) {
        half2 sum = {};
        for (int i = col; i < head_size/2; i += blockDim.x) {
            auto qh = __float22half2_rn(q2[i]);
            sum += k2[i] * qh;
        }
        float sumf = (float)(sum.x + sum.y);
        sumf = warp_reduce_sum(sumf);
        sumf = sumf > 0.0f ? sumf : 0.0f;
        score += sumf * w[head];
        q2 += head_size/2;
    }
    if (col == 0) {
        dst[size_t(n_kv)*row_q + row_k] = score;
    }
}

//template <typename mask_t>
//static __global__ void k_fused_gemm_relu_mul_sum_rows(const half * __restrict__ k, const float * __restrict__ q,
//        const float * __restrict__ w, const mask_t * __restrict__ m, float * __restrict__ dst,
//        int head_size, int n_kv, int n_head, size_t nbm) {
//    const int row_k = blockIdx.x;
//    const int row_q = blockIdx.y;
//    const int col  = threadIdx.x % WARP_SIZE;
//    const int head = threadIdx.x / WARP_SIZE;
//
//    k += size_t(row_k) * head_size;
//    q += size_t(row_q) * head_size * n_head + head * head_size;
//    w += size_t(row_q) * n_head;
//    auto this_m = (const mask_t *)((const char *)m + nbm*row_q);
//
//    __shared__ float s[32];
//
//    auto k2 = (const half2  *)k;
//    auto q2 = (const float2 *)q;
//    half2 sum = {};
//    for (int i = col; i < head_size/2; i += WARP_SIZE) {
//        auto qh = __float22half2_rn(q2[i]);
//        sum += k2[i] * qh;
//    }
//    float sumf = (float)(sum.x + sum.y);
//    sumf = warp_reduce_sum(sumf);
//    sumf = sumf > 0.0f ? sumf * w[head] : 0.0f;
//    if (col == 0) {
//        s[head] = sumf;
//    }
//    __syncthreads();
//    if (head == 0) {
//        sumf = s[col];
//        sumf = warp_reduce_sum(sumf);
//        if (col == 0) {
//            dst[size_t(n_kv)*row_q + row_k] = sumf + (float)this_m[row_k];
//        }
//    }
//}

void ggml_cuda_op_indexer_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto op = ggml_unary_op(dst->op_params[0]);
    GGML_ASSERT(op == GGML_UNARY_OP_RELU);
    auto k = dst->src[0];
    auto q = dst->src[1];
    auto w = dst->src[2];
    auto m = dst->src[3];
    int n_top_k = dst->ne[0];
    int n_kv    = k->ne[1];
    int n_head  = q->ne[1];
    GGML_ASSERT(k->type == GGML_TYPE_F16 || ggml_is_quantized(k->type));
    GGML_ASSERT(k->ne[2] == 1 || k->ne[3] == 1);
    GGML_ASSERT(k->ne[1] > n_top_k);
    GGML_ASSERT(k->ne[1] == m->ne[0]);
    GGML_ASSERT(k->ne[0] == q->ne[0]);
    GGML_ASSERT(q->ne[2] == m->ne[1]);
    GGML_ASSERT(q->ne[1] == w->ne[0]);
    GGML_ASSERT(q->ne[2] == w->ne[1]);
    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(m->type == GGML_TYPE_F32 || m->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(w));

    constexpr int k_block_size = 256;

    if (false && k->type == GGML_TYPE_F16 && q->type == GGML_TYPE_F32) {
        printf("%s: using alternative\n", __func__);
        ggml_cuda_pool_alloc<float> score (ctx.pool(), q->ne[2] * n_kv);
        ggml_cuda_pool_alloc<int>   sorted(ctx.pool(), q->ne[2] * n_kv);
        dim3 grid(n_kv, q->ne[2], 1);
        if (m->type == GGML_TYPE_F16) {
            k_fused_gemm_relu_mul_sum_rows<<<grid, WARP_SIZE, 0, ctx.stream()>>>((const half *)k->data, (const float *)q->data,
                    (const float *)w->data, (const half *)m->data, score.get(), k->ne[0], k->ne[1], q->ne[1], m->nb[1]);
        } else {
            k_fused_gemm_relu_mul_sum_rows<<<grid, WARP_SIZE, 0, ctx.stream()>>>((const half *)k->data, (const float *)q->data,
                    (const float *)w->data, (const float *)m->data, score.get(), k->ne[0], k->ne[1], q->ne[1], m->nb[1]);
        }
        CUDA_CHECK(cudaGetLastError());

        argsort_f32_i32_cuda_cub(ctx.pool(), score.get(), sorted.get(), k->ne[1], q->ne[2], GGML_SORT_ORDER_DESC, ctx.stream());
        CUDA_CHECK(cudaGetLastError());

        k_copy_topk<<<q->ne[2], k_block_size, 0, ctx.stream()>>>(sorted.get(), (int *)dst->data, k->ne[1], dst->ne[0]);
        CUDA_CHECK(cudaGetLastError());

        return;
    }

    if (k->type == GGML_TYPE_F16 && q->type == GGML_TYPE_F32 && q->ne[2] <= 16) {
        ggml_cuda_pool_alloc<half>  kq(ctx.pool(), int64_t(n_kv)*q->ne[2]*q->ne[1]);
        ggml_cuda_pool_alloc<float> score(ctx.pool(), int64_t(n_kv)*q->ne[2]);
        ggml_cuda_pool_alloc<int>   sorted(ctx.pool(), int64_t(n_kv)*q->ne[2]);
        ggml_cuda_pool_alloc<half>  q_f16(ctx.pool(), q->ne[0]*q->ne[1]*q->ne[2]);

        auto to_fp16_cuda = ggml_get_to_fp16_cuda(q->type);
        to_fp16_cuda((const float *)q->data, q_f16.get(), q->ne[0]*q->ne[1]*q->ne[2], 1, ctx.stream());
        CUDA_CHECK(cudaGetLastError());

        const half alpha = 1.0f;
        const half beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(ctx.device), ctx.stream()));
        CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(ctx.device), CUBLAS_OP_T, CUBLAS_OP_N,
                    k->ne[1], q->ne[1]*q->ne[2], q->ne[0],
                    &alpha, (const half *)k->data,       CUDA_R_16F, k->ne[0],
                             q_f16.get(),       CUDA_R_16F, q->ne[0],
                    &beta,   kq.get(), CUDA_R_16F, k->ne[1],
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        int nblocks = (k->ne[1] + k_block_size - 1)/k_block_size;
        dim3 grid(q->ne[2], nblocks, 1);
        if (m->type == GGML_TYPE_F32) {
            k_fused_relu_mul_sum_rows_2<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(), (const float *)w->data, (const float *)m->data,
                    score.get(), k->ne[1], q->ne[1], m->nb[1]);
        } else {
            k_fused_relu_mul_sum_rows_2<<<grid, k_block_size, 0, ctx.stream()>>>(kq.get(), (const float *)w->data, (const half  *)m->data,
                    score.get(), k->ne[1], q->ne[1], m->nb[1]);
        }
        //if (m->type == GGML_TYPE_F32) {
        //    k_fused_relu_mul_sum_rows<<<q->ne[2], k_block_size, 0, ctx.stream()>>>(kq.get(), (const float *)w->data, (const float *)m->data, score.get(), k->ne[1], q->ne[1], m->nb[1]);
        //} else {
        //    k_fused_relu_mul_sum_rows<<<q->ne[2], k_block_size, 0, ctx.stream()>>>(kq.get(), (const float *)w->data, (const half  *)m->data, score.get(), k->ne[1], q->ne[1], m->nb[1]);
        //}
        CUDA_CHECK(cudaGetLastError());

        argsort_f32_i32_cuda_cub(ctx.pool(), score.get(), sorted.get(), k->ne[1], q->ne[2], GGML_SORT_ORDER_DESC, ctx.stream());
        CUDA_CHECK(cudaGetLastError());

        k_copy_topk<<<q->ne[2], k_block_size, 0, ctx.stream()>>>(sorted.get(), (int *)dst->data, k->ne[1], dst->ne[0]);
        CUDA_CHECK(cudaGetLastError());

        return;

    }

    constexpr int64_t k_max_work_buffer_elements = 1 << 28;

    int max_rows = k_max_work_buffer_elements / n_kv / n_head;
    if (max_rows < 1) max_rows = 1;
    if (max_rows > q->ne[2]) max_rows = q->ne[2];

    int nstep = (q->ne[2] + max_rows - 1)/max_rows;

    ggml_cuda_pool_alloc<float> kq(ctx.pool(), int64_t(n_kv)*max_rows*n_head);
    ggml_cuda_pool_alloc<float> score(ctx.pool(), int64_t(n_kv)*max_rows);
    ggml_cuda_pool_alloc<int>   sorted(ctx.pool(), int64_t(n_kv)*max_rows);
    ggml_cuda_pool_alloc<float> k_f32(ctx.pool());
    ggml_cuda_pool_alloc<char>  q_converted(ctx.pool());
    auto q_padded = GGML_PAD(q->ne[0], MATRIX_ROW_PADDING);
    if (ggml_is_quantized(k->type)) {
        auto nbytes_q = q->ne[1] * max_rows * sizeof(block_q8_1)/QK8_1;
        nbytes_q += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
        q_converted.alloc(nbytes_q);
    } else {
        k_f32.alloc(k->ne[0]*k->ne[1]);
        auto to_fp32_cuda = ggml_get_to_fp32_cuda(k->type);
        to_fp32_cuda(k->data, k_f32.get(), k->ne[1]*k->ne[0], 1, ctx.stream());
        CUDA_CHECK(cudaGetLastError());
    }

    for (int istep = 0; istep < nstep; ++istep) {
        int first = istep*max_rows;
        int last  = std::min(first + max_rows, int(q->ne[2]));
        int nrows = last - first;
        auto q_data = (const char *)q->data + istep*max_rows*q->nb[2];
        auto m_data = (const char *)m->data + istep*max_rows*m->nb[1];
        if (ggml_is_quantized(k->type)) {
            quantize_mmq_q8_1_cuda((const float *)q_data, q_converted.get(), q->ne[0], nrows, 1, q_padded, k->type, ctx.stream());
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
                    &alpha,     k_f32.get(),  k->ne[0],
                       (const float *)q_data, q->ne[0],
                    &beta,      kq.get(),     k->ne[1]));
        }
        // sum_row1(relu(kq)*w)
        if (m->type == GGML_TYPE_F32) {
            k_fused_relu_mul_sum_rows<<<nrows, k_block_size, 0, ctx.stream()>>>(kq.get(), (const float *)w->data, (const float *)m_data, score.get(), k->ne[1], q->ne[1], m->nb[1]);
        } else {
            k_fused_relu_mul_sum_rows<<<nrows, k_block_size, 0, ctx.stream()>>>(kq.get(), (const float *)w->data, (const half  *)m_data, score.get(), k->ne[1], q->ne[1], m->nb[1]);
        }
        CUDA_CHECK(cudaGetLastError());

        argsort_f32_i32_cuda_cub(ctx.pool(), score.get(), sorted.get(), k->ne[1], nrows, GGML_SORT_ORDER_DESC, ctx.stream());
        CUDA_CHECK(cudaGetLastError());

        k_copy_topk<<<nrows, k_block_size, 0, ctx.stream()>>>(sorted.get(), (int *)((char *)dst->data + first*dst->nb[1]), k->ne[1], dst->ne[0]);
        CUDA_CHECK(cudaGetLastError());
    }

}
