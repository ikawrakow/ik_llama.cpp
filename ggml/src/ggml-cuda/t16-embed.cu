#include "t16-embed.cuh"

// K2E token embedding lookup: one CUDA block (256 threads) per token.
// Per 256-wide group: unpack the row ring, mul1, scale, Hadamard 256, signs.

static __device__ __forceinline__ float k2e_mul1(uint16_t w) {
    const uint32_t x = uint32_t(w) * 0x83DCD12Du;
    const uint32_t s = 0x6400u + ((x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24));
    const __half h = __ushort_as_half((unsigned short)(s & 0xFFFFu));
    const __half v = __hfma(h, __ushort_as_half((unsigned short)0x1EEEu), __ushort_as_half((unsigned short)0xC931u));
    return __half2float(v);
}

static __global__ void k2e_get_rows_kernel(
        const uint16_t * __restrict__ w,
        const uint16_t * __restrict__ signs,
        const int32_t  * __restrict__ ids,
        float          * __restrict__ out,
        int K, int groups, int hidden, int row_words, int n) {
    const int t = blockIdx.x;
    if (t >= n) return;
    const int tid = threadIdx.x; // 0..255
    const int64_t id = ids[t];
    const uint16_t * row = w + id * (int64_t) row_words;
    float * o = out + (int64_t) t * hidden;

    __shared__ uint16_t sym[256];
    __shared__ float xs[256];

    const int W = row_words / groups;
    const int nw = W - 1;
    const uint32_t mask = (1u << K) - 1u;
    const int reps = (15 + K) / K;

    for (int g = 0; g < groups; ++g) {
        const uint16_t * rg = row + g * W;
        const float scale = __half2float(__ushort_as_half(rg[0]));
        const uint16_t * sw = rg + 1;

        const int bit = tid * K;
        const int wi = bit >> 4;
        const int off = bit & 15;
        const uint32_t win = (uint32_t) sw[wi] | ((uint32_t) sw[(wi + 1) % nw] << 16);
        sym[tid] = (uint16_t)((win >> off) & mask);
        __syncthreads();

        uint32_t st = 0;
        for (int j = 0; j < reps; ++j) {
            st |= (uint32_t) sym[(tid - j) & 255] << (j * K);
        }
        xs[tid] = k2e_mul1((uint16_t)(st & 0xFFFFu)) * scale;
        __syncthreads();

        for (int h = 1; h < 256; h <<= 1) {
            const int j = tid & (2*h - 1);
            if (j < h) {
                const float a = xs[tid], b = xs[tid + h];
                xs[tid] = a + b;
                xs[tid + h] = a - b;
            }
            __syncthreads();
        }

        const int oi = g*256 + tid;
        if (oi < hidden) {
            o[oi] = xs[tid] * (1.0f/16.0f) * __half2float(__ushort_as_half(signs[g*256 + tid]));
        }
        __syncthreads();
    }
}

void ggml_cuda_op_t16_embed_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w    = dst->src[0];
    const ggml_tensor * signs= dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(w->type == GGML_TYPE_I16 && signs->type == GGML_TYPE_F16 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(w) && ggml_is_contiguous(signs) && ggml_is_contiguous(ids));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));

    const int K      = (int) dst->op_params[0];
    const int groups = (int) dst->op_params[1];
    const int hidden = (int) dst->op_params[2];
    const int row_words = (int) w->ne[0];
    const int n      = (int) ids->ne[0];
    GGML_ASSERT((int) signs->ne[1] == groups && (int) signs->ne[0] == 256);
    GGML_ASSERT(dst->ne[0] == hidden && dst->ne[1] == n);
    GGML_ASSERT(row_words % groups == 0 && (row_words / groups - 1) % 16 == 0);

    cudaStream_t stream = ctx.stream();
    k2e_get_rows_kernel<<<n, 256, 0, stream>>>(
            (const uint16_t *) w->data, (const uint16_t *) signs->data,
            (const int32_t *) ids->data, (float *) dst->data,
            K, groups, hidden, row_words, n);
    CUDA_CHECK(cudaGetLastError());
}
