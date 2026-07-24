#include "common.cuh"
#include "convert.cuh"
#include "latent_attn.cuh"

#include <climits>

// Latent attention over a packed K/V cache with an independently-visible K/V prefix
// (ggml_latent_attn_prefix_ext, dense mode 0). CUDA path for F32, F16, and Q8_0 caches
// (Q8_0 is dequantized once to a contiguous F16 buffer, then takes the F16 path).
//
// K is shared across every query (MLA: the latent cache does not depend on the head),
// so the whole score matrix is two plain GEMMs, not a per-head batch:
//   scores[P+N, QT] = scale * [ prefix_k^T ; cache^T ] @ Q      (QT = T*H flattened)
// then a fused mask+softmax down each column, then two value GEMMs:
//   out[Dv, QT] = cacheV @ W_cache + prefix_v^T @ W_prefix
// where cacheV is the value slice cache[dv_off .. dv_off+Dv, :] read in place (no
// transpose, no cont). Query columns are tiled to bound the [P+N, cw] score buffer.

static __device__ float latent_block_reduce_max(float value, float * buf) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    value = warp_reduce_max(value);
    if (blockDim.x > WARP_SIZE) {
        __syncthreads();
        if (warp == 0) {
            buf[lane] = -INFINITY;
        }
        __syncthreads();
        if (lane == 0) {
            buf[warp] = value;
        }
        __syncthreads();
        value = warp_reduce_max(buf[lane]);
    }
    return value;
}

static __device__ float latent_block_reduce_sum(float value, float * buf) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    value = warp_reduce_sum(value);
    if (blockDim.x > WARP_SIZE) {
        __syncthreads();
        if (warp == 0) {
            buf[lane] = 0.0f;
        }
        __syncthreads();
        if (lane == 0) {
            buf[warp] = value;
        }
        __syncthreads();
        value = warp_reduce_sum(buf[lane]);
    }
    return value;
}

// Fused mask-add + column softmax. One block per score column. scores is column-major
// [PN, cw] (a column's PN logits are contiguous). The cache segment (rows >= P) gets the
// additive mask for that column's token t; the prefix segment (rows < P) is always visible.
// Writes normalized weights into wout (float or half) with the same layout. For the half
// path, restore[col] reverses the power-of-two query scaling before mask and softmax.
template <typename Tout, bool Restore>
static __global__ void k_latent_mask_softmax(
        const float * __restrict__ scores, Tout * __restrict__ wout,
        const float * __restrict__ restore,
        const float * __restrict__ mask, int PN, int P, int T, int c0,
        int64_t mask_nb1_f /* row stride of mask in floats */) {
    const int col   = blockIdx.x;
    const int tid   = threadIdx.x;
    const int nth   = blockDim.x;
    const float * s = scores + (size_t) col*PN;
    Tout        * w = wout   + (size_t) col*PN;
    const float r = Restore ? restore[col] : 1.0f;
    const bool do_restore = Restore && r != 1.0f;
    const int t = (c0 + col) % T;
    const float * mrow = mask ? mask + (size_t) t*mask_nb1_f : nullptr;

    extern __shared__ float shbuf[]; // WARP_SIZE floats

    // pass 1: max of (logit + mask)
    float local_max = -INFINITY;
    for (int k = tid; k < PN; k += nth) {
        float v = s[k];
        if (do_restore) v *= r;
        if (mrow && k >= P) v += mrow[k - P];
        local_max = fmaxf(local_max, v);
    }
    const float mx = latent_block_reduce_max(local_max, shbuf);

    // pass 2: sum of exp
    float local_sum = 0.0f;
    for (int k = tid; k < PN; k += nth) {
        float v = s[k];
        if (do_restore) v *= r;
        if (mrow && k >= P) v += mrow[k - P];
        local_sum += expf(v - mx);
    }
    const float sum = latent_block_reduce_sum(local_sum, shbuf);
    const float inv = sum > 0.0f ? 1.0f/sum : 0.0f;

    // pass 3: normalized weights
    for (int k = tid; k < PN; k += nth) {
        float v = s[k];
        if (do_restore) v *= r;
        if (mrow && k >= P) v += mrow[k - P];
        w[k] = (Tout) (expf(v - mx) * inv);
    }
}

// Mode 1 packs query columns token-major ([Dk, H*rows]) so a gathered cache tile for
// one token can be paired with all H heads in one strided-batched GEMM.
template <typename Tout>
static __global__ void k_latent_pack_q_indexed(
        const float * __restrict__ q, Tout * __restrict__ qout,
        int Dk, int H, int first, int rows, int64_t q_nb1_f, int64_t q_nb2_f) {
    const int col = blockIdx.x;
    if (col >= rows*H) {
        return;
    }
    const int tr = col/H;
    const int h  = col%H;
    const float * qrow = q + (size_t) (first + tr)*q_nb1_f + (size_t) h*q_nb2_f;
    Tout * dst = qout + (size_t) col*Dk;
    for (int d = threadIdx.x; d < Dk; d += blockDim.x) {
        dst[d] = (Tout) qrow[d];
    }
}

// Range-safe F32 -> F16 query pack. Each query column is divided by the smallest
// power of two that puts its largest finite magnitude in the F16 domain. The score
// softmax kernel multiplies the resulting logits by restore[col] before adding masks.
static __global__ void k_latent_pack_q_scaled(
        const float * __restrict__ q, half * __restrict__ qout,
        float * __restrict__ restore, int Dk, int H, int first, int rows,
        int64_t q_nb1_f, int64_t q_nb2_f, bool indexed) {
    const int col = blockIdx.x;
    const int tid = threadIdx.x;
    if (col >= rows*H) {
        return;
    }

    const int tr = indexed ? col/H : col;
    const int h  = indexed ? col%H : 0;
    const float * qrow = q + (size_t) (first + tr)*q_nb1_f + (size_t) h*q_nb2_f;

    // Dk <= 4*nth covers the in-tree MLA shapes (OpenPangu/GLM use Dk=576).
    // Keeping those values in registers avoids rereading q after the reduction;
    // larger generic shapes retain the bounded two-read fallback.
    constexpr int kValsPerThread = 4;
    const bool cache_values = Dk <= blockDim.x*kValsPerThread;
    float qvals[kValsPerThread];
    int nvals = 0;
    float amax = 0.0f;
    for (int d = tid; d < Dk; d += blockDim.x) {
        const float v = qrow[d];
        if (cache_values) qvals[nvals++] = v;
        amax = fmaxf(amax, fabsf(v));
    }

    extern __shared__ float shbuf[];
    amax = latent_block_reduce_max(amax, shbuf);
    __syncthreads(); // reduce slots must be quiesced before shbuf[0] is reused below
    if (tid == 0) {
        float r = 1.0f;
        if (isfinite(amax)) {
            while (amax/r > 65504.0f) {
                r *= 2.0f;
            }
        }
        restore[col] = r;
        shbuf[0] = r;
    }
    __syncthreads();

    const float r = shbuf[0];
    half * dst = qout + (size_t) col*Dk;
    nvals = 0;
    for (int d = tid; d < Dk; d += blockDim.x) {
        const float v = cache_values ? qvals[nvals++] : qrow[d];
        dst[d] = (half) (v/r);
    }
}

// F32/F16 row gather. Each selected cache row is copied whole; values are later read
// from the block-aligned [dv_off, dv_off+dv) slice of this full-row tile.
template <typename T>
static __global__ void k_latent_gather_rows_indexed(
        const char * __restrict__ cache, int64_t cache_nb1,
        const int32_t * __restrict__ indices, int64_t indices_nb1_i,
        T * __restrict__ gathered, int Dk, int topk, int first, int rows) {
    const int row = blockIdx.x;
    if (row >= rows*topk) {
        return;
    }
    const int tr = row/topk;
    const int k  = row%topk;
    const int32_t idx = indices[(size_t) (first + tr)*indices_nb1_i + k];
    const T * src = (const T *) (cache + (size_t) idx*cache_nb1);
    T * dst = gathered + (size_t) row*Dk;
    for (int d = threadIdx.x; d < Dk; d += blockDim.x) {
        dst[d] = src[d];
    }
}

// Q8_0 is decoded only for selected rows. The complete packed row is the source and the
// complete dequantized row is the destination; no narrowed quantized view is formed.
static __global__ void k_latent_gather_rows_q8_0_indexed(
        const char * __restrict__ cache, int64_t cache_nb1,
        const int32_t * __restrict__ indices, int64_t indices_nb1_i,
        half * __restrict__ gathered, int Dk, int topk, int first, int rows) {
    const int row = blockIdx.x;
    if (row >= rows*topk) {
        return;
    }
    const int tr = row/topk;
    const int k  = row%topk;
    const int32_t idx = indices[(size_t) (first + tr)*indices_nb1_i + k];
    const block_q8_0 * src = (const block_q8_0 *) (cache + (size_t) idx*cache_nb1);
    half * dst = gathered + (size_t) row*Dk;
    for (int d = threadIdx.x; d < Dk; d += blockDim.x) {
        const block_q8_0 & block = src[d/QK8_0];
        dst[d] = (half) ((float) block.d * block.qs[d%QK8_0]);
    }
}

// Indexed fused mask-add + softmax. Score columns are token-major, so col/H selects
// the token and col%H selects the head. Only gathered cache logits consult the mask.
template <typename Tout, bool Restore>
static __global__ void k_latent_mask_softmax_indexed(
        const float * __restrict__ scores, Tout * __restrict__ wout,
        const float * __restrict__ restore,
        const float * __restrict__ mask, const int32_t * __restrict__ indices,
        int M, int P, int H, int first, int64_t mask_nb1_f, int64_t indices_nb1_i) {
    const int col   = blockIdx.x;
    const int tid   = threadIdx.x;
    const int nth   = blockDim.x;
    const float * s = scores + (size_t) col*M;
    Tout        * w = wout   + (size_t) col*M;
    const float r = Restore ? restore[col] : 1.0f;
    const bool do_restore = Restore && r != 1.0f;
    const int t = first + col/H;
    const float * mrow = mask ? mask + (size_t) t*mask_nb1_f : nullptr;
    const int32_t * idx = indices + (size_t) t*indices_nb1_i;

    extern __shared__ float shbuf[];

    float local_max = -INFINITY;
    for (int k = tid; k < M; k += nth) {
        float v = s[k];
        if (do_restore) v *= r;
        if (mrow && k >= P) v += mrow[idx[k - P]];
        local_max = fmaxf(local_max, v);
    }
    const float mx = latent_block_reduce_max(local_max, shbuf);

    float local_sum = 0.0f;
    for (int k = tid; k < M; k += nth) {
        float v = s[k];
        if (do_restore) v *= r;
        if (mrow && k >= P) v += mrow[idx[k - P]];
        local_sum += expf(v - mx);
    }
    const float sum = latent_block_reduce_sum(local_sum, shbuf);
    const float inv = sum > 0.0f ? 1.0f/sum : 0.0f;

    for (int k = tid; k < M; k += nth) {
        float v = s[k];
        if (do_restore) v *= r;
        if (mrow && k >= P) v += mrow[idx[k - P]];
        w[k] = (Tout) (expf(v - mx) * inv);
    }
}

// Convert token-major tile output [Dv, H*rows] back to ggml's [Dv, T, H] strides.
static __global__ void k_latent_copy_out_indexed(
        const float * __restrict__ src, float * __restrict__ dst,
        int Dv, int H, int first, int rows, int64_t dst_nb1_f, int64_t dst_nb2_f) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t ne = (int64_t) Dv*H*rows;
    if (i >= ne) {
        return;
    }
    const int d   = i%Dv;
    const int col = i/Dv;
    const int tr  = col/H;
    const int h   = col%H;
    dst[(size_t) (first + tr)*dst_nb1_f + (size_t) h*dst_nb2_f + d] = src[i];
}

// ---- host-side small helpers -------------------------------------------------------

bool ggml_cuda_latent_attn_is_supported(const ggml_tensor * op) {
    const ggml_tensor * cache   = op->src[1];
    if (cache == nullptr) {
        return false;
    }

    const int mode = op->op_params[4];
    if (cache->type != GGML_TYPE_F32 && cache->type != GGML_TYPE_F16 && cache->type != GGML_TYPE_Q8_0) {
        return false;
    }

    // Both dense (cuBLAS) and indexed (row-gather) readers index each cache row as a packed
    // element array. The dense path additionally requires an int-sized cuBLAS leading dimension.
    if (cache->type == GGML_TYPE_F32 || cache->type == GGML_TYPE_F16) {
        const size_t element_size = ggml_type_size(cache->type);
        if (cache->nb[0] != element_size) {
            return false;
        }
        if (mode == 0) {
            if (cache->nb[1] % element_size != 0) {
                return false;
            }
            const size_t lda = cache->nb[1] / element_size;
            if (lda < (size_t) cache->ne[0] || lda > (size_t) INT_MAX) {
                return false;
            }
        }
    }
    if (ggml_is_quantized(cache->type) &&
            cache->nb[1] != ggml_row_size(cache->type, cache->ne[0])) {
        return false;
    }
    return true;
}

// f32 GEMM path: C[m,n] = alpha * op(A) @ op(B) + beta * C, all float.
static void sgemm(ggml_backend_cuda_context & ctx, cublasOperation_t ta, cublasOperation_t tb,
        int m, int n, int k, float alpha, const float * A, int lda,
        const float * B, int ldb, float beta, float * C, int ldc) {
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(ctx.device), ctx.stream()));
    CUBLAS_CHECK(cublasSgemm(ctx.cublas_handle(ctx.device), ta, tb, m, n, k,
            &alpha, A, lda, B, ldb, &beta, C, ldc));
}

// f16-input GEMM with f32 accumulate into an f32 C.
static void hgemm_f32acc(ggml_backend_cuda_context & ctx, cublasOperation_t ta, cublasOperation_t tb,
        int m, int n, int k, float alpha, const half * A, int lda,
        const half * B, int ldb, float beta, float * C, int ldc) {
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(ctx.device), ctx.stream()));
    CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(ctx.device), ta, tb, m, n, k,
            &alpha, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb,
            &beta,  C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void sgemm_strided_batched(ggml_backend_cuda_context & ctx, cublasOperation_t ta, cublasOperation_t tb,
        int m, int n, int k, float alpha, const float * A, int lda, int64_t stride_a,
        const float * B, int ldb, int64_t stride_b, float beta, float * C, int ldc,
        int64_t stride_c, int batch_count) {
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(ctx.device), ctx.stream()));
    CUBLAS_CHECK(cublasSgemmStridedBatched(ctx.cublas_handle(ctx.device), ta, tb, m, n, k,
            &alpha, A, lda, stride_a, B, ldb, stride_b, &beta, C, ldc, stride_c, batch_count));
}

static void hgemm_f32acc_strided_batched(
        ggml_backend_cuda_context & ctx, cublasOperation_t ta, cublasOperation_t tb,
        int m, int n, int k, float alpha, const half * A, int lda, int64_t stride_a,
        const half * B, int ldb, int64_t stride_b, float beta, float * C, int ldc,
        int64_t stride_c, int batch_count) {
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(ctx.device), ctx.stream()));
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(ctx.cublas_handle(ctx.device), ta, tb, m, n, k,
            &alpha, A, CUDA_R_16F, lda, stride_a, B, CUDA_R_16F, ldb, stride_b,
            &beta, C, CUDA_R_32F, ldc, stride_c, batch_count,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void ggml_cuda_op_latent_attn_indexed(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q       = dst->src[0];
    const ggml_tensor * cache   = dst->src[1];
    const ggml_tensor * pk      = dst->src[2];
    const ggml_tensor * pv      = dst->src[3];
    const ggml_tensor * mask    = dst->src[4];
    const ggml_tensor * indices = dst->src[5];

    const int Dk   = q->ne[0];
    const int T    = q->ne[1];
    const int H    = q->ne[2];
    const int P    = pk ? pk->ne[1] : 0;
    const int topk = indices->ne[0];
    const int M    = P + topk;

    float scale;
    memcpy(&scale, &dst->op_params[0], sizeof(float));
    const int dv     = dst->op_params[2];
    const int dv_off = dst->op_params[3];

    const bool f16 = cache->type != GGML_TYPE_F32;
    const int64_t q_nb1_f       = q->nb[1]/sizeof(float);
    const int64_t q_nb2_f       = q->nb[2]/sizeof(float);
    const int64_t mask_nb1_f    = mask ? mask->nb[1]/sizeof(float) : 0;
    const int64_t indices_nb1_i = indices->nb[1]/sizeof(int32_t);
    const int64_t dst_nb1_f     = dst->nb[1]/sizeof(float);
    const int64_t dst_nb2_f     = dst->nb[2]/sizeof(float);
    cudaStream_t stream = ctx.stream();

    // Score scratch is [P+topk, H*rows] f32. Cap it at 16M floats and retain the
    // DSA max_rows=32 knob: rows=min(T, 32, floor(16M/((P+topk)*H))), clamped to 1.
    constexpr int kMaxRows = 32;
    constexpr int64_t kMaxScoreElems = 16*1024*1024;
    const int64_t score_elems_per_row = (int64_t) M*H;
    int max_rows = std::max<int64_t>(1, kMaxScoreElems/score_elems_per_row);
    max_rows = std::min(max_rows, kMaxRows);
    max_rows = std::min(max_rows, T);

    // Prefix tensors are converted once to the cache compute type. F32 cache stays on
    // SGEMM; F16 and gathered Q8_0 use half inputs with f32 accumulation.
    ggml_cuda_pool_alloc<float> pk_f32(ctx.pool()), pv_f32(ctx.pool());
    ggml_cuda_pool_alloc<half>  pk_f16(ctx.pool()), pv_f16(ctx.pool());
    const float * pk_f = nullptr;
    const float * pv_f = nullptr;
    const half  * pk_h = nullptr;
    const half  * pv_h = nullptr;
    if (P > 0) {
        if (!f16) {
            if (pk->type == GGML_TYPE_F32) {
                pk_f = (const float *) pk->data;
            } else {
                pk_f32.alloc((int64_t) Dk*P);
                ggml_get_to_fp32_cuda(pk->type)(pk->data, pk_f32.get(), (int64_t) Dk*P, 1, stream);
                pk_f = pk_f32.get();
            }
            if (pv->type == GGML_TYPE_F32) {
                pv_f = (const float *) pv->data;
            } else {
                pv_f32.alloc((int64_t) P*dv);
                ggml_get_to_fp32_cuda(pv->type)(pv->data, pv_f32.get(), (int64_t) P*dv, 1, stream);
                pv_f = pv_f32.get();
            }
        } else {
            if (pk->type == GGML_TYPE_F16) {
                pk_h = (const half *) pk->data;
            } else {
                pk_f16.alloc((int64_t) Dk*P);
                ggml_get_to_fp16_cuda(pk->type)(pk->data, pk_f16.get(), (int64_t) Dk*P, 1, stream);
                pk_h = pk_f16.get();
            }
            if (pv->type == GGML_TYPE_F16) {
                pv_h = (const half *) pv->data;
            } else {
                pv_f16.alloc((int64_t) P*dv);
                ggml_get_to_fp16_cuda(pv->type)(pv->data, pv_f16.get(), (int64_t) P*dv, 1, stream);
                pv_h = pv_f16.get();
            }
        }
    }

    const int64_t tile_cols = (int64_t) H*max_rows;
    ggml_cuda_pool_alloc<float> scores(ctx.pool(), (int64_t) M*tile_cols);
    ggml_cuda_pool_alloc<float> outbuf(ctx.pool(), (int64_t) dv*tile_cols);
    ggml_cuda_pool_alloc<float> q_f32(ctx.pool()), gathered_f32(ctx.pool()), w_f32(ctx.pool());
    ggml_cuda_pool_alloc<half>  q_f16(ctx.pool()), gathered_f16(ctx.pool()), w_f16(ctx.pool());
    ggml_cuda_pool_alloc<float> q_restore(ctx.pool());
    if (!f16) {
        q_f32.alloc((int64_t) Dk*tile_cols);
        gathered_f32.alloc((int64_t) Dk*topk*max_rows);
        w_f32.alloc((int64_t) M*tile_cols);
    } else {
        q_f16.alloc((int64_t) Dk*tile_cols);
        gathered_f16.alloc((int64_t) Dk*topk*max_rows);
        w_f16.alloc((int64_t) M*tile_cols);
        q_restore.alloc(tile_cols);
    }

    constexpr int nth = 256;
    const size_t softmax_shmem = WARP_SIZE*sizeof(float);
    for (int first = 0; first < T; first += max_rows) {
        const int rows = std::min(max_rows, T - first);
        const int cols = rows*H;

        if (!f16) {
            k_latent_pack_q_indexed<float><<<cols, nth, 0, stream>>>(
                    (const float *) q->data, q_f32.get(), Dk, H, first, rows, q_nb1_f, q_nb2_f);
            k_latent_gather_rows_indexed<float><<<rows*topk, nth, 0, stream>>>(
                    (const char *) cache->data, cache->nb[1], (const int32_t *) indices->data,
                    indices_nb1_i, gathered_f32.get(), Dk, topk, first, rows);
            CUDA_CHECK(cudaGetLastError());

            if (P > 0) {
                sgemm(ctx, CUBLAS_OP_T, CUBLAS_OP_N, P, cols, Dk, scale,
                        pk_f, Dk, q_f32.get(), Dk, 0.0f, scores.get(), M);
            }
            sgemm_strided_batched(ctx, CUBLAS_OP_T, CUBLAS_OP_N, topk, H, Dk, scale,
                    gathered_f32.get(), Dk, (int64_t) Dk*topk,
                    q_f32.get(), Dk, (int64_t) Dk*H, 0.0f, scores.get() + P, M,
                    (int64_t) M*H, rows);
        } else {
            k_latent_pack_q_scaled<<<cols, nth, softmax_shmem, stream>>>(
                    (const float *) q->data, q_f16.get(), q_restore.get(), Dk, H, first, rows,
                    q_nb1_f, q_nb2_f, true);
            if (cache->type == GGML_TYPE_F16) {
                k_latent_gather_rows_indexed<half><<<rows*topk, nth, 0, stream>>>(
                        (const char *) cache->data, cache->nb[1], (const int32_t *) indices->data,
                        indices_nb1_i, gathered_f16.get(), Dk, topk, first, rows);
            } else {
                k_latent_gather_rows_q8_0_indexed<<<rows*topk, nth, 0, stream>>>(
                        (const char *) cache->data, cache->nb[1], (const int32_t *) indices->data,
                        indices_nb1_i, gathered_f16.get(), Dk, topk, first, rows);
            }
            CUDA_CHECK(cudaGetLastError());

            if (P > 0) {
                hgemm_f32acc(ctx, CUBLAS_OP_T, CUBLAS_OP_N, P, cols, Dk, scale,
                        pk_h, Dk, q_f16.get(), Dk, 0.0f, scores.get(), M);
            }
            hgemm_f32acc_strided_batched(ctx, CUBLAS_OP_T, CUBLAS_OP_N, topk, H, Dk, scale,
                    gathered_f16.get(), Dk, (int64_t) Dk*topk,
                    q_f16.get(), Dk, (int64_t) Dk*H, 0.0f, scores.get() + P, M,
                    (int64_t) M*H, rows);
        }

        if (!f16) {
            k_latent_mask_softmax_indexed<float, false><<<cols, nth, softmax_shmem, stream>>>(
                    scores.get(), w_f32.get(), nullptr, mask ? (const float *) mask->data : nullptr,
                    (const int32_t *) indices->data, M, P, H, first, mask_nb1_f, indices_nb1_i);
        } else {
            k_latent_mask_softmax_indexed<half, true><<<cols, nth, softmax_shmem, stream>>>(
                    scores.get(), w_f16.get(), q_restore.get(), mask ? (const float *) mask->data : nullptr,
                    (const int32_t *) indices->data, M, P, H, first, mask_nb1_f, indices_nb1_i);
        }
        CUDA_CHECK(cudaGetLastError());

        if (!f16) {
            sgemm_strided_batched(ctx, CUBLAS_OP_N, CUBLAS_OP_N, dv, H, topk, 1.0f,
                    gathered_f32.get() + dv_off, Dk, (int64_t) Dk*topk,
                    w_f32.get() + P, M, (int64_t) M*H, 0.0f, outbuf.get(), dv,
                    (int64_t) dv*H, rows);
            if (P > 0) {
                sgemm(ctx, CUBLAS_OP_T, CUBLAS_OP_N, dv, cols, P, 1.0f,
                        pv_f, P, w_f32.get(), M, 1.0f, outbuf.get(), dv);
            }
        } else {
            hgemm_f32acc_strided_batched(ctx, CUBLAS_OP_N, CUBLAS_OP_N, dv, H, topk, 1.0f,
                    gathered_f16.get() + dv_off, Dk, (int64_t) Dk*topk,
                    w_f16.get() + P, M, (int64_t) M*H, 0.0f, outbuf.get(), dv,
                    (int64_t) dv*H, rows);
            if (P > 0) {
                hgemm_f32acc(ctx, CUBLAS_OP_T, CUBLAS_OP_N, dv, cols, P, 1.0f,
                        pv_h, P, w_f16.get(), M, 1.0f, outbuf.get(), dv);
            }
        }

        const int64_t ne = (int64_t) dv*cols;
        k_latent_copy_out_indexed<<<(ne + nth - 1)/nth, nth, 0, stream>>>(
                outbuf.get(), (float *) dst->data, dv, H, first, rows, dst_nb1_f, dst_nb2_f);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ggml_cuda_op_latent_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_latent_attn_is_supported(dst));
    if (dst->op_params[4] == 1) {
        ggml_cuda_op_latent_attn_indexed(ctx, dst);
        return;
    }

    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * cache = dst->src[1];
    const ggml_tensor * pk    = dst->src[2];
    const ggml_tensor * pv    = dst->src[3];
    const ggml_tensor * mask  = dst->src[4];

    const int64_t Dk = q->ne[0];
    const int64_t T  = q->ne[1];
    const int64_t H  = q->ne[2];
    const int64_t N  = cache->ne[1];
    const int64_t P  = pk ? pk->ne[1] : 0;
    const int64_t QT = T*H;
    const int64_t PN = P + N;

    float scale;
    memcpy(&scale, &dst->op_params[0], sizeof(float));
    const int dv     = dst->op_params[2];
    const int dv_off = dst->op_params[3];

    // Quantized caches dequantize to a contiguous F16 buffer once, then take the F16 path.
    const bool cache_quant = ggml_is_quantized(cache->type);
    const bool f16 = cache_quant || cache->type == GGML_TYPE_F16;

    const int64_t mask_nb1_f = mask ? mask->nb[1] / sizeof(float) : 0;

    cudaStream_t stream = ctx.stream();

    // resolve the cache into a compute-type pointer + element stride
    ggml_cuda_pool_alloc<half> cache_deq(ctx.pool());
    const half  * cache_h = nullptr;
    const float * cache_f = nullptr;
    int64_t cache_stride  = 0;
    if (cache->type == GGML_TYPE_F32) {
        cache_f = (const float *) cache->data;
        cache_stride = cache->nb[1] / sizeof(float);
    } else if (cache->type == GGML_TYPE_F16) {
        cache_h = (const half *) cache->data;
        cache_stride = cache->nb[1] / sizeof(half);
    } else {
        cache_deq.alloc(Dk*N);
        ggml_get_to_fp16_cuda(cache->type)(cache->data, cache_deq.get(), N, Dk, stream);
        CUDA_CHECK(cudaGetLastError());
        cache_h = cache_deq.get();
        cache_stride = Dk;
    }

    // column tile: bound the [PN, cw] score buffer to ~16M floats
    const int64_t kMaxScoreElems = 16*1024*1024;
    int64_t cw = PN > 0 ? kMaxScoreElems / PN : QT;
    if (cw < 1)  cw = 1;
    if (cw > QT) cw = QT;

    // ---- prefix K/V converted once to the compute type (if needed) ----
    ggml_cuda_pool_alloc<float> pk_f32(ctx.pool()), pv_f32(ctx.pool());
    ggml_cuda_pool_alloc<half>  pk_f16(ctx.pool()), pv_f16(ctx.pool());
    const float * pk_f = nullptr; const float * pv_f = nullptr;
    const half  * pk_h = nullptr; const half  * pv_h = nullptr;
    if (P > 0) {
        if (!f16) {
            if (pk->type == GGML_TYPE_F32) { pk_f = (const float *) pk->data; }
            else { pk_f32.alloc(Dk*P); ggml_get_to_fp32_cuda(pk->type)(pk->data, pk_f32.get(), Dk*P, 1, stream); pk_f = pk_f32.get(); }
            if (pv->type == GGML_TYPE_F32) { pv_f = (const float *) pv->data; }
            else { pv_f32.alloc(P*dv); ggml_get_to_fp32_cuda(pv->type)(pv->data, pv_f32.get(), P*dv, 1, stream); pv_f = pv_f32.get(); }
        } else {
            if (pk->type == GGML_TYPE_F16) { pk_h = (const half *) pk->data; }
            else { pk_f16.alloc(Dk*P); ggml_get_to_fp16_cuda(pk->type)(pk->data, pk_f16.get(), Dk*P, 1, stream); pk_h = pk_f16.get(); }
            if (pv->type == GGML_TYPE_F16) { pv_h = (const half *) pv->data; }
            else { pv_f16.alloc(P*dv); ggml_get_to_fp16_cuda(pv->type)(pv->data, pv_f16.get(), P*dv, 1, stream); pv_h = pv_f16.get(); }
        }
    }

    // ---- per-tile buffers ----
    ggml_cuda_pool_alloc<float> scores(ctx.pool(), PN*cw);
    ggml_cuda_pool_alloc<float> outbuf(ctx.pool(), (int64_t) dv*cw);
    ggml_cuda_pool_alloc<half>  q_f16(ctx.pool());   // f16 path: range-scaled q tile
    ggml_cuda_pool_alloc<half>  w_f16(ctx.pool());   // f16 path: normalized weights
    ggml_cuda_pool_alloc<float> w_f32(ctx.pool());   // f32 path: weights (== softmax out)
    ggml_cuda_pool_alloc<float> q_restore(ctx.pool());
    if (f16) {
        q_f16.alloc(Dk*cw);
        w_f16.alloc(PN*cw);
        q_restore.alloc(cw);
    } else {
        w_f32.alloc(PN*cw);
    }

    const int sm_nth = 256;
    const size_t sm_shmem = WARP_SIZE*sizeof(float);

    for (int64_t c0 = 0; c0 < QT; c0 += cw) {
        const int64_t cwn = std::min<int64_t>(cw, QT - c0);

        // ---- scores = scale * K^T @ Q  (prefix rows [0,P), cache rows [P,PN)) ----
        const float * qtile = (const float *) q->data + c0*Dk; // contiguous [Dk, cwn]
        if (!f16) {
            if (P > 0) sgemm(ctx, CUBLAS_OP_T, CUBLAS_OP_N, P, cwn, Dk, scale,
                             pk_f, Dk, qtile, Dk, 0.0f, scores.get(), PN);
            sgemm(ctx, CUBLAS_OP_T, CUBLAS_OP_N, N, cwn, Dk, scale,
                  cache_f, cache_stride, qtile, Dk, 0.0f, scores.get() + P, PN);
        } else {
            k_latent_pack_q_scaled<<<cwn, sm_nth, sm_shmem, stream>>>(
                    qtile, q_f16.get(), q_restore.get(), Dk, 1, 0, cwn, Dk, 0, false);
            CUDA_CHECK(cudaGetLastError());
            if (P > 0) hgemm_f32acc(ctx, CUBLAS_OP_T, CUBLAS_OP_N, P, cwn, Dk, scale,
                                    pk_h, Dk, q_f16.get(), Dk, 0.0f, scores.get(), PN);
            hgemm_f32acc(ctx, CUBLAS_OP_T, CUBLAS_OP_N, N, cwn, Dk, scale,
                         cache_h, cache_stride, q_f16.get(), Dk, 0.0f, scores.get() + P, PN);
        }

        // ---- fused range restoration + mask + column softmax -> weights ----
        if (f16) {
            k_latent_mask_softmax<half, true><<<cwn, sm_nth, sm_shmem, stream>>>(
                    scores.get(), w_f16.get(), q_restore.get(),
                    mask ? (const float *) mask->data : nullptr, PN, P, T, c0, mask_nb1_f);
        } else {
            k_latent_mask_softmax<float, false><<<cwn, sm_nth, sm_shmem, stream>>>(
                    scores.get(), w_f32.get(), nullptr,
                    mask ? (const float *) mask->data : nullptr, PN, P, T, c0, mask_nb1_f);
        }
        CUDA_CHECK(cudaGetLastError());

        // ---- values: out = cacheV @ W_cache (+ prefix_v^T @ W_prefix) ----
        if (!f16) {
            const float * cacheV = cache_f + dv_off; // row offset within packed row
            sgemm(ctx, CUBLAS_OP_N, CUBLAS_OP_N, dv, cwn, N, 1.0f,
                  cacheV, cache_stride, w_f32.get() + P, PN, 0.0f, outbuf.get(), dv);
            if (P > 0) sgemm(ctx, CUBLAS_OP_T, CUBLAS_OP_N, dv, cwn, P, 1.0f,
                             pv_f, P, w_f32.get(), PN, 1.0f, outbuf.get(), dv);
        } else {
            const half * cacheV = cache_h + dv_off;
            hgemm_f32acc(ctx, CUBLAS_OP_N, CUBLAS_OP_N, dv, cwn, N, 1.0f,
                         cacheV, cache_stride, w_f16.get() + P, PN, 0.0f, outbuf.get(), dv);
            if (P > 0) hgemm_f32acc(ctx, CUBLAS_OP_T, CUBLAS_OP_N, dv, cwn, P, 1.0f,
                                    pv_h, P, w_f16.get(), PN, 1.0f, outbuf.get(), dv);
        }

        // ---- copy tile -> dst (both column-major [Dv, .] contiguous) ----
        CUDA_CHECK(cudaMemcpyAsync((float *) dst->data + c0*dv, outbuf.get(),
                (size_t) dv*cwn*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
}
