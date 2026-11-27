#include "add-exact-cols.cuh"

template <typename vt> static __device__ __forceinline__ float aec_to_float(vt v);
template <> __device__ __forceinline__ float aec_to_float<float>(float v) { return v; }
template <> __device__ __forceinline__ float aec_to_float<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }

template <typename vt>
static __global__ void k_add_exact_cols(float * __restrict__ dst, const char * __restrict__ x, const int32_t * __restrict__ cols,
        const char * __restrict__ vals, const char * __restrict__ ids, int64_t nr, int k, int64_t x_ne1,
        size_t x_nb1, size_t x_nb2, size_t d_nb1, size_t d_nb2, size_t ids_nb0, size_t ids_nb1, size_t vals_nb1, size_t vals_nb2) {
    const int64_t r = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= nr) return;
    const int i1 = blockIdx.y, i2 = blockIdx.z;
    const int m = ids ? *(const int32_t *)(ids + i1*ids_nb0 + i2*ids_nb1) : 0;
    const float * xr = (const float *)(x + (ids ? i1 % x_ne1 : i1)*x_nb1 + i2*x_nb2);
    const int32_t * cm = cols + m*k;
    const char * vm = vals + m*vals_nb2;
    float s = 0.0f;
    for (int j = 0; j < k; ++j) s += aec_to_float(((const vt *)(vm + j*vals_nb1))[r]) * xr[cm[j]];
    float * d = (float *)((char *)dst + i1*d_nb1 + i2*d_nb2);
    d[r] += s;
}

void ggml_cuda_op_add_exact_cols(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x    = dst->src[1];
    const ggml_tensor * cols = dst->src[2];
    const ggml_tensor * vals = dst->src[3];
    const ggml_tensor * ids  = dst->src[4];
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && cols->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->ne[3] == 1 && x->ne[3] == 1 && dst->ne[1] <= 65535 && dst->ne[2] <= 65535);
    const int64_t nr = dst->ne[0];
    const int k = (int)cols->ne[0];
    const dim3 block(128, 1, 1), grid((nr + 127)/128, dst->ne[1], dst->ne[2]);
    cudaStream_t stream = ctx.stream();
    const char * idsd = ids ? (const char *)ids->data : nullptr;
    const size_t ids_nb0 = ids ? ids->nb[0] : 0, ids_nb1 = ids ? ids->nb[1] : 0;
    if (vals->type == GGML_TYPE_BF16) {
        k_add_exact_cols<nv_bfloat16><<<grid, block, 0, stream>>>((float *)dst->data, (const char *)x->data, (const int32_t *)cols->data,
                (const char *)vals->data, idsd, nr, k, x->ne[1], x->nb[1], x->nb[2], dst->nb[1], dst->nb[2], ids_nb0, ids_nb1, vals->nb[1], vals->nb[2]);
    } else {
        GGML_ASSERT(vals->type == GGML_TYPE_F32);
        k_add_exact_cols<float><<<grid, block, 0, stream>>>((float *)dst->data, (const char *)x->data, (const int32_t *)cols->data,
                (const char *)vals->data, idsd, nr, k, x->ne[1], x->nb[1], x->nb[2], dst->nb[1], dst->nb[2], ids_nb0, ids_nb1, vals->nb[1], vals->nb[2]);
    }
}
