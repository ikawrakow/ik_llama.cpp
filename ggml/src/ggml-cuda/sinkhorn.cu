#include "common.cuh"
#include "sinkhorn.cuh"

// Sinkhorn normalization of T independent S x S matrices (S <= 8, so a matrix is
// at most 64 floats). One thread per token: the whole matrix lives in a thread-local
// array and the 6-node-per-iteration graph chain collapses into this single kernel.
// Semantics match the reference: softmax over columns, column normalization,
// then (iters - 1) rounds of row + column normalization (ends on columns).
// Input is the flat [S*S, T] row-major tensor (column index fastest); output is
// [S, S, T] with ne0 = row, i.e. transposed on write.

template <int S>
static __global__ void k_sinkhorn(const float * __restrict__ x, float * __restrict__ dst,
                                  const int64_t T, const int iters, const float eps,
                                  const int transposed, const int64_t nb1) {
    const int64_t t = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) {
        return;
    }

    const float * xt = (const float *)((const char *) x + t*nb1);
    float m[S*S];

    #pragma unroll
    for (int r = 0; r < S; ++r) {
        float mx = xt[r*S];
        for (int c = 1; c < S; ++c) mx = fmaxf(mx, xt[r*S + c]);
        float sum = 0.0f;
        for (int c = 0; c < S; ++c) { m[r*S + c] = expf(xt[r*S + c] - mx); sum += m[r*S + c]; }
        for (int c = 0; c < S; ++c) m[r*S + c] = m[r*S + c]/sum + eps;
    }
    #pragma unroll
    for (int c = 0; c < S; ++c) {
        float sum = eps;
        for (int r = 0; r < S; ++r) sum += m[r*S + c];
        for (int r = 0; r < S; ++r) m[r*S + c] /= sum;
    }
    for (int i = 0; i < iters - 1; ++i) {
        #pragma unroll
        for (int r = 0; r < S; ++r) {
            float sum = eps;
            for (int c = 0; c < S; ++c) sum += m[r*S + c];
            for (int c = 0; c < S; ++c) m[r*S + c] /= sum;
        }
        #pragma unroll
        for (int c = 0; c < S; ++c) {
            float sum = eps;
            for (int r = 0; r < S; ++r) sum += m[r*S + c];
            for (int r = 0; r < S; ++r) m[r*S + c] /= sum;
        }
    }

    float * yt = dst + t*S*S;
    if (transposed) {
        #pragma unroll
        for (int c = 0; c < S; ++c) {
            for (int r = 0; r < S; ++r) yt[c*S + r] = m[r*S + c];
        }
    } else {
        #pragma unroll
        for (int k = 0; k < S*S; ++k) yt[k] = m[k];
    }
}

template <int S>
static __global__ void k_hc_pre(const float * __restrict__ x, const float * __restrict__ scale,
        const float * __restrict__ bias, float * __restrict__ dst,
        const int64_t T, const int iters, const float eps, const int64_t nb1) {
    const int64_t t = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= T) {
        return;
    }

    float m[S*S];

    const float * x_pre  = (const float *)((const char *) x + t*nb1);
    const float * x_post = x_pre + S;
    const float * x_comb = x_pre + 2*S;

    float * y_pre  = dst + S*t;
    float * y_post = dst + S*(T + t);
    float * y_comb = dst + 2*S*T + S*S*t;

    #pragma unroll
    for (int i = 0; i < S; ++i) {
        float val = x_pre[i] * scale[0] + bias[i];
        y_pre[i]  = 1.f / (1.f + expf(-val)) + eps;
    }

    #pragma unroll
    for (int i = 0; i < S; ++i) {
        float val = x_post[i] * scale[1] + bias[S + i];
        y_post[i]  = 2.f / (1.f + expf(-val));
    }

    #pragma unroll
    for (int i = 0; i < S*S; ++i) {
        m[i] = x_comb[i] * scale[2] + bias[2*S + i];
    }

    #pragma unroll
    for (int r = 0; r < S; ++r) {
        float mx = m[r*S];
        for (int c = 1; c < S; ++c) mx = fmaxf(mx, m[r*S + c]);
        float sum = 0.0f;
        for (int c = 0; c < S; ++c) { m[r*S + c] = expf(m[r*S + c] - mx); sum += m[r*S + c]; }
        for (int c = 0; c < S; ++c) m[r*S + c] = m[r*S + c]/sum + eps;
    }
    #pragma unroll
    for (int c = 0; c < S; ++c) {
        float sum = eps;
        for (int r = 0; r < S; ++r) sum += m[r*S + c];
        for (int r = 0; r < S; ++r) m[r*S + c] /= sum;
    }
    for (int i = 0; i < iters - 1; ++i) {
        #pragma unroll
        for (int r = 0; r < S; ++r) {
            float sum = eps;
            for (int c = 0; c < S; ++c) sum += m[r*S + c];
            for (int c = 0; c < S; ++c) m[r*S + c] /= sum;
        }
        #pragma unroll
        for (int c = 0; c < S; ++c) {
            float sum = eps;
            for (int r = 0; r < S; ++r) sum += m[r*S + c];
            for (int r = 0; r < S; ++r) m[r*S + c] /= sum;
        }
    }

    #pragma unroll
    for (int i = 0; i < S*S; ++i) y_comb[i] = m[i];
}

void ggml_cuda_op_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const int S     = dst->op_params[0];
    const int iters = dst->op_params[1];
    float eps;
    memcpy(&eps, &dst->op_params[2], sizeof(float));
    const int transposed = dst->op_params[3];
    const int64_t T = src0->ne[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(S >= 1 && S <= 8);
    GGML_ASSERT(src0->ne[0] == (int64_t) S * S);
    GGML_ASSERT(ggml_is_contiguous(dst));

    if (T == 0) {
        return;
    }

    const int block = 256;
    const int64_t grid = (T + block - 1)/block;
    cudaStream_t stream = ctx.stream();

    const float * x = (const float *) src0->data;
    float * y = (float *) dst->data;

    switch (S) {
        case 1: k_sinkhorn<1><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 2: k_sinkhorn<2><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 3: k_sinkhorn<3><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 4: k_sinkhorn<4><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 5: k_sinkhorn<5><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 6: k_sinkhorn<6><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 7: k_sinkhorn<7><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        case 8: k_sinkhorn<8><<<grid, block, 0, stream>>>(x, y, T, iters, eps, transposed, src0->nb[1]); break;
        default: GGML_ABORT("sinkhorn: unsupported S");
    }
}

void ggml_cuda_op_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const int S     = dst->op_params[0];
    const int iters = dst->op_params[1];
    float eps;
    memcpy(&eps, &dst->op_params[2], sizeof(float));
    const int64_t T = src0->ne[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(S >= 1 && S <= 8);
    GGML_ASSERT(src0->ne[0] == (int64_t) S * S + 2 * S);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src1->ne[0] >= 3 && ggml_nrows(src1) == 1);
    GGML_ASSERT(src2->ne[0] >= S*(S+2) && ggml_nrows(src2) == 1);

    if (T == 0) {
        return;
    }

    const int block = 256;
    const int64_t grid = (T + block - 1)/block;
    cudaStream_t stream = ctx.stream();

    const float * x = (const float *) src0->data;
    const float * s = (const float *) src1->data;
    const float * b = (const float *) src2->data;
    float * y = (float *) dst->data;

    switch (S) {
        case 1: k_hc_pre<1><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 2: k_hc_pre<2><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 3: k_hc_pre<3><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 4: k_hc_pre<4><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 5: k_hc_pre<5><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 6: k_hc_pre<6><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 7: k_hc_pre<7><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        case 8: k_hc_pre<8><<<grid, block, 0, stream>>>(x, s, b, y, T, iters, eps, src0->nb[1]); break;
        default: GGML_ABORT("sinkhorn: unsupported S");
    }
}
