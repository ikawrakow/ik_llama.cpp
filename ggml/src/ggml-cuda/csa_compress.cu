// Step-5 CSA region compress: per-(region, dim) softmax(z)-weighted mean of the proxy keys
// over each region's tokens; padded slots get a -1e30 pre-softmax bias. See ggml.h.

#include "csa_compress.cuh"

#include <cuda_fp8.h>

__device__ __forceinline__ float csa_round_bf16_e4m3(float x) {
    // F32 -> BF16 (RNE) -> E4M3FN (RN, satfinite: clamps at +-448, no NaN byte) -> F32.
    const __nv_bfloat16 bf = __float2bfloat16_rn(x);
    const __nv_fp8_storage_t fp8 = __nv_cvt_bfloat16raw_to_fp8(
            __nv_bfloat16_raw(__bfloat16_as_ushort(bf)), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8_e4m3 q;
    q.__x = fp8;
    return (float) q;
}

template <typename z_t>
static __global__ void k_csa_compress(const z_t * __restrict__ z,
        const float * __restrict__ k, const float * __restrict__ vis, float * __restrict__ dst,
        const int B, const int head_size, const int round_e4m3) {
    const int b = blockIdx.x;      // region
    const int d = threadIdx.x;     // dim (head_size threads, one per proxy dim)
    if (d >= head_size) {
        return;
    }

    const int base = b*B;
    float mx = -INFINITY;
    for (int t = 0; t < B; ++t) {
        const float bias = vis[base + t]*1e30f - 1e30f;
        mx = fmaxf(mx, (float) z[d + (base + t)*head_size] + bias);
    }
    float num = 0.0f, denom = 0.0f;
    for (int t = 0; t < B; ++t) {
        const float bias = vis[base + t]*1e30f - 1e30f;
        const float w = expf((float) z[d + (base + t)*head_size] + bias - mx);
        denom += w;
        num += w * k[d + (base + t)*head_size];
    }
    const float value = num/denom;
    dst[d + b*head_size] = round_e4m3 ? csa_round_bf16_e4m3(value) : value;
}

void ggml_cuda_op_csa_compress(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * z   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * vis = dst->src[2];
    GGML_ASSERT(z->type == GGML_TYPE_F16 || z->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32 && vis->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(z->ne[0] == k->ne[0] && z->ne[1] == k->ne[1]);
    GGML_ASSERT(vis->ne[0] == k->ne[1]);
    GGML_ASSERT(ggml_is_contiguous(z) && ggml_is_contiguous(k) && ggml_is_contiguous(vis));

    const int B          = dst->op_params[0];
    const int round_e4m3 = dst->op_params[1];
    const int head_size  = dst->ne[0];
    const int n_blocks   = dst->ne[1];
    GGML_ASSERT(n_blocks == k->ne[1]/B);

    // head_size == proxy_dim (256 in practice); generic thread count.
    const int threads = ((head_size + 31)/32)*32;
    GGML_ASSERT(threads <= 1024);

    if (z->type == GGML_TYPE_F16) {
        k_csa_compress<half><<<n_blocks, threads, 0, ctx.stream()>>>(
                (const half *) z->data, (const float *) k->data,
                (const float *) vis->data, (float *) dst->data, B, head_size, round_e4m3);
    } else {
        k_csa_compress<float><<<n_blocks, threads, 0, ctx.stream()>>>(
                (const float *) z->data, (const float *) k->data,
                (const float *) vis->data, (float *) dst->data, B, head_size, round_e4m3);
    }
    CUDA_CHECK(cudaGetLastError());
}
