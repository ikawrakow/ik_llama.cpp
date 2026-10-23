// Step-5 CSA region compress, fused: per-(region, dim) softmax(z)-weighted mean of the
// proxy keys over each region's tokens, with the official BF16->E4M3 double round in the
// store epilogue. Official semantics: _official/inference/kernel.py:_csa_compress_regions_kernel
// (one program per (region, head) there; one thread per (region, dim) here -- the softmax and
// the sum run over the region's tokens only, so no cross-thread reduction is needed at all).
//
// Masking follows the graph semantics of the decomposed path, not the reference prefill's
// ragged-tail skip: padded slots get z - 1e30 pre-softmax (their weight is exp(-1e30 - max)
// == 0) and partial blocks still summarize. The masked z is computed as z + (vis*1e30 - 1e30)
// -- the add AFTER the scale-bias -- exactly like the graph's ggml_add(ggml_scale_bias(vis)).
//
// round_e4m3: 0 = plain F32 summary, 1 = kernel.py:345 double round
// F32 -> BF16 (RNE) -> E4M3 (RN, satfinite) widened back to F32. Widening the E4M3 result to
// F32 is byte-equivalent to the reference widening E4M3 -> bf16 before the score matmul
// (kernel.py:2136), since BF16 exactly represents every E4M3 value.
//
// Normalization mirrors ggml's softmax: w = exp(z - max) * (1/sum) -- inv-sum multiply, not
// divide (ggml-cuda/softmax.cu). Residual ~1 ulp divergence from the decomposed graph is
// possible (its block-reduce sums in a different order); validation is the E4M3-byte
// replay and behavioral tests, not byte-identity with the decomposed chain.

#include "csa_compress.cuh"

#include <cuda_fp8.h>

__device__ __forceinline__ float csa_round_bf16_e4m3(float x) {
    // 1. F32 -> BF16, RNE (matches Triton/PyTorch .to(bfloat16) and ggml's
    //    ggml_compute_fp32_to_bf16; the intermediate round can move a value onto an E4M3
    //    midpoint -- kernel.py:341).
    const __nv_bfloat16 bf = __float2bfloat16_rn(x);
    // 2. BF16 -> E4M3FN, RN + satfinite (matches .to(float8e4nv) == cvt.rn.satfinite.e4m3x2:
    //    clamps at +-448 instead of producing the NaN byte).
    const __nv_fp8_storage_t fp8 = __nv_cvt_bfloat16raw_to_fp8(
            __nv_bfloat16_raw(__bfloat16_as_ushort(bf)), __NV_SATFINITE, __NV_E4M3);
    // 3. Dequantize E4M3 -> F32 for the downstream F32 score GEMM.
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
    float denom = 0.0f;
    for (int t = 0; t < B; ++t) {
        const float bias = vis[base + t]*1e30f - 1e30f;
        denom += expf((float) z[d + (base + t)*head_size] + bias - mx);
    }
    const float inv_sum = 1.0f / denom;
    float num = 0.0f;
    for (int t = 0; t < B; ++t) {
        const float bias = vis[base + t]*1e30f - 1e30f;
        const float w = expf((float) z[d + (base + t)*head_size] + bias - mx) * inv_sum;
        num += w * k[d + (base + t)*head_size];
    }
    dst[d + b*head_size] = round_e4m3 ? csa_round_bf16_e4m3(num) : num;
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

    // head_size == proxy_dim == 256 in practice; generic thread count, still one wave.
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
