#include "convert.cuh"
#include "fma.cuh"

template <bool b_is_const, typename T>
static __global__ void fma_kernel(const int ne0, T * __restrict__ dst,
                                  const T * __restrict__ c,
                                  const T * __restrict__ a,
                                  const T * __restrict__ b,
                                  const int64_t ne1, const int64_t ne2, const int64_t ne3,
                                  const int64_t ne11, const int64_t ne12, const int64_t ne13,
                                  const int64_t ne21, const int64_t ne22, const int64_t ne23,
                                  const int64_t nb1, const int64_t nb2, const int64_t nb3,
                                  const int64_t nb01, const int64_t nb02, const int64_t nb03,
                                  const int64_t nb11, const int64_t nb12, const int64_t nb13,
                                  const int64_t nb21, const int64_t nb22, const int64_t nb23,
                                  const int64_t total_elements) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) {
        return;
    }

    const int64_t i0 = idx % ne0;
    const int64_t i1 = (idx / ne0) % ne1;
    const int64_t i2 = (idx / (ne0 * ne1)) % ne2;
    const int64_t i3 = idx / (ne0 * ne1 * ne2);

    const int64_t i11 = i1 % ne11;
    const int64_t i12 = i2 % ne12;
    const int64_t i13 = i3 % ne13;

    const int64_t i21 = i1 % ne21;
    const int64_t i22 = i2 % ne22;
    const int64_t i23 = i3 % ne23;

    auto dc = c + i1*nb01 + i2*nb02 + i3*nb03;
    auto da = a + i11*nb11 + i12*nb12 + i13*nb13;
    auto db = b + i21*nb21 + i22*nb22 + i23*nb23;

    auto dr = dst + i1*nb1 + i2*nb2 + i3*nb3;

    if constexpr (b_is_const) {
        dr[i0] = dc[i0] + da[i0] * db[0];
    } else {
        dr[i0] = dc[i0] + da[i0] * db[i0];
    }
}

template <bool b_is_const, typename T>
static __global__ void fma_kernel2(const int ne0, T * __restrict__ dst,
                                  const T * __restrict__ c,
                                  const T * __restrict__ a,
                                  const T * __restrict__ b,
                                  const int ne11, const int ne12,
                                  const int ne21, const int ne22,
                                  const int64_t nb1,  const int64_t nb2,
                                  const int64_t nb01, const int64_t nb02,
                                  const int64_t nb11, const int64_t nb12,
                                  const int64_t nb21, const int64_t nb22) {

    int i0 = blockIdx.z*blockDim.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }
    int i2 = blockIdx.x;
    int i1 = blockIdx.y;

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;

    const int i21 = i1 % ne21;
    const int i22 = i2 % ne22;

    auto dc = c + i1*nb01 + i2*nb02;
    auto da = a + i11*nb11 + i12*nb12;
    auto db = b + i21*nb21 + i22*nb22;

    auto dr = dst + i1*nb1 + i2*nb2;

    if constexpr (b_is_const) {
        dr[i0] = dc[i0] + da[i0] * db[0];
    } else {
        dr[i0] = dc[i0] + da[i0] * db[i0];
    }
}

void ggml_cuda_op_fma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    constexpr int k_block_size = 256;

    auto c = dst->src[0];
    auto a = dst->src[1];
    auto b = dst->src[2];

    GGML_ASSERT(c->type == a->type && c->type == b->type && c->type == dst->type);
    GGML_ASSERT(c->type == GGML_TYPE_F32 || c->type == GGML_TYPE_F16 || c->type == GGML_TYPE_BF16);
    GGML_ASSERT(ggml_are_same_shape(c, dst));
    GGML_ASSERT(c->ne[0] == a->ne[0]);
    GGML_ASSERT(c->ne[0] == b->ne[0] || b->ne[0] == 1);

    auto d = dst;

    if (c->ne[3] == 1) {
        int nblocks = (c->ne[0] + k_block_size - 1)/k_block_size;
        const dim3 grid(c->ne[2], c->ne[1], nblocks);
        if (b->ne[0] == 1 && b->ne[0] != c->ne[0]) {
            if (c->type == GGML_TYPE_F32) {
                auto tsize = sizeof(float);
                GGML_ASSERT(c->nb[0] == tsize && a->nb[0] == tsize && b->nb[0] == tsize && dst->nb[0] == tsize);
                fma_kernel2<true><<<grid, k_block_size, 0, ctx.stream()>>>(c->ne[0], (float *)d->data,
                        (const float *)c->data, (const float *)a->data, (const float *)b->data,
                        a->ne[1], a->ne[2], b->ne[1], b->ne[2], d->nb[1]/tsize, d->nb[2]/tsize,
                        c->nb[1]/tsize, c->nb[2]/tsize, a->nb[1]/tsize, a->nb[2]/tsize, b->nb[1]/tsize, b->nb[2]/tsize);
            }
            else if (c->type == GGML_TYPE_F16) {
                auto tsize = sizeof(half);
                GGML_ASSERT(c->nb[0] == tsize && a->nb[0] == tsize && b->nb[0] == tsize && dst->nb[0] == tsize);
                fma_kernel2<true><<<grid, k_block_size, 0, ctx.stream()>>>(c->ne[0], (half *)d->data,
                        (const half *)c->data, (const half *)a->data, (const half *)b->data,
                        a->ne[1], a->ne[2], b->ne[1], b->ne[2], d->nb[1]/tsize, d->nb[2]/tsize,
                        c->nb[1]/tsize, c->nb[2]/tsize, a->nb[1]/tsize, a->nb[2]/tsize, b->nb[1]/tsize, b->nb[2]/tsize);
            }
            else {
                auto tsize = sizeof(nv_bfloat16);
                GGML_ASSERT(c->nb[0] == tsize && a->nb[0] == tsize && b->nb[0] == tsize && dst->nb[0] == tsize);
                fma_kernel2<true><<<grid, k_block_size, 0, ctx.stream()>>>(c->ne[0], (nv_bfloat16 *)d->data,
                        (const nv_bfloat16 *)c->data, (const nv_bfloat16 *)a->data, (const nv_bfloat16 *)b->data,
                        a->ne[1], a->ne[2], b->ne[1], b->ne[2], d->nb[1]/tsize, d->nb[2]/tsize,
                        c->nb[1]/tsize, c->nb[2]/tsize, a->nb[1]/tsize, a->nb[2]/tsize, b->nb[1]/tsize, b->nb[2]/tsize);
            }
        } else {
            if (c->type == GGML_TYPE_F32) {
                auto tsize = sizeof(float);
                GGML_ASSERT(c->nb[0] == tsize && a->nb[0] == tsize && b->nb[0] == tsize && dst->nb[0] == tsize);
                fma_kernel2<false><<<grid, k_block_size, 0, ctx.stream()>>>(c->ne[0], (float *)d->data,
                        (const float *)c->data, (const float *)a->data, (const float *)b->data,
                        a->ne[1], a->ne[2], b->ne[1], b->ne[2], d->nb[1]/tsize, d->nb[2]/tsize,
                        c->nb[1]/tsize, c->nb[2]/tsize, a->nb[1]/tsize, a->nb[2]/tsize, b->nb[1]/tsize, b->nb[2]/tsize);
            }
            else if (c->type == GGML_TYPE_F16) {
                auto tsize = sizeof(half);
                GGML_ASSERT(c->nb[0] == tsize && a->nb[0] == tsize && b->nb[0] == tsize && dst->nb[0] == tsize);
                fma_kernel2<false><<<grid, k_block_size, 0, ctx.stream()>>>(c->ne[0], (half *)d->data,
                        (const half *)c->data, (const half *)a->data, (const half *)b->data,
                        a->ne[1], a->ne[2], b->ne[1], b->ne[2], d->nb[1]/tsize, d->nb[2]/tsize,
                        c->nb[1]/tsize, c->nb[2]/tsize, a->nb[1]/tsize, a->nb[2]/tsize, b->nb[1]/tsize, b->nb[2]/tsize);
            }
            else {
                auto tsize = sizeof(nv_bfloat16);
                GGML_ASSERT(c->nb[0] == tsize && a->nb[0] == tsize && b->nb[0] == tsize && dst->nb[0] == tsize);
                fma_kernel2<false><<<grid, k_block_size, 0, ctx.stream()>>>(c->ne[0], (nv_bfloat16 *)d->data,
                        (const nv_bfloat16 *)c->data, (const nv_bfloat16 *)a->data, (const nv_bfloat16 *)b->data,
                        a->ne[1], a->ne[2], b->ne[1], b->ne[2], d->nb[1]/tsize, d->nb[2]/tsize,
                        c->nb[1]/tsize, c->nb[2]/tsize, a->nb[1]/tsize, a->nb[2]/tsize, b->nb[1]/tsize, b->nb[2]/tsize);
            }
        }
        return;
    }

    int64_t nelem = ggml_nelements(c);
    int n_blocks = (nelem + k_block_size - 1)/k_block_size;

    if (b->ne[0] == 1 && b->ne[0] != c->ne[0]) {
        if (c->type == GGML_TYPE_F32) {
            auto tsize = sizeof(float);
            GGML_ASSERT(c->nb[0] == tsize && a->nb[0] == tsize && b->nb[0] == tsize && dst->nb[0] == tsize);
            fma_kernel<true><<<n_blocks, k_block_size, 0, ctx.stream()>>>(c->ne[0], (float *)dst->data,
                    (const float *)c->data, (const float *)a->data, (const float *)b->data,
                    c->ne[1], c->ne[2], c->ne[3], a->ne[1], a->ne[2], a->ne[3], b->ne[1], b->ne[2], b->ne[3],
                    d->nb[1]/tsize, d->nb[2]/tsize, d->nb[3]/tsize,
                    c->nb[1]/tsize, c->nb[2]/tsize, c->nb[3]/tsize,
                    a->nb[1]/tsize, a->nb[2]/tsize, a->nb[3]/tsize,
                    b->nb[1]/tsize, b->nb[2]/tsize, b->nb[3]/tsize, nelem);
        }
        else if (c->type == GGML_TYPE_F16) {
            auto tsize = sizeof(half);
            GGML_ASSERT(c->nb[0] == tsize && a->nb[0] == tsize && b->nb[0] == tsize && dst->nb[0] == tsize);
            fma_kernel<true><<<n_blocks, k_block_size, 0, ctx.stream()>>>(c->ne[0], (half *)dst->data,
                    (const half *)c->data, (const half *)a->data, (const half *)b->data,
                    c->ne[1], c->ne[2], c->ne[3], a->ne[1], a->ne[2], a->ne[3], b->ne[1], b->ne[2], b->ne[3],
                    d->nb[1]/tsize, d->nb[2]/tsize, d->nb[3]/tsize,
                    c->nb[1]/tsize, c->nb[2]/tsize, c->nb[3]/tsize,
                    a->nb[1]/tsize, a->nb[2]/tsize, a->nb[3]/tsize,
                    b->nb[1]/tsize, b->nb[2]/tsize, b->nb[3]/tsize, nelem);
        } else {
            auto tsize = sizeof(nv_bfloat16);
            GGML_ASSERT(c->nb[0] == tsize && a->nb[0] == tsize && b->nb[0] == tsize && dst->nb[0] == tsize);
            fma_kernel<true><<<n_blocks, k_block_size, 0, ctx.stream()>>>(c->ne[0], (nv_bfloat16 *)dst->data,
                    (const nv_bfloat16 *)c->data, (const nv_bfloat16 *)a->data, (const nv_bfloat16 *)b->data,
                    c->ne[1], c->ne[2], c->ne[3], a->ne[1], a->ne[2], a->ne[3], b->ne[1], b->ne[2], b->ne[3],
                    d->nb[1]/tsize, d->nb[2]/tsize, d->nb[3]/tsize,
                    c->nb[1]/tsize, c->nb[2]/tsize, c->nb[3]/tsize,
                    a->nb[1]/tsize, a->nb[2]/tsize, a->nb[3]/tsize,
                    b->nb[1]/tsize, b->nb[2]/tsize, b->nb[3]/tsize, nelem);
        }
    } else {
        if (c->type == GGML_TYPE_F32) {
            auto tsize = sizeof(float);
            GGML_ASSERT(c->nb[0] == tsize && a->nb[0] == tsize && b->nb[0] == tsize && dst->nb[0] == tsize);
            fma_kernel<false><<<n_blocks, k_block_size, 0, ctx.stream()>>>(c->ne[0], (float *)dst->data,
                    (const float *)c->data, (const float *)a->data, (const float *)b->data,
                    c->ne[1], c->ne[2], c->ne[3], a->ne[1], a->ne[2], a->ne[3], b->ne[1], b->ne[2], b->ne[3],
                    d->nb[1]/tsize, d->nb[2]/tsize, d->nb[3]/tsize,
                    c->nb[1]/tsize, c->nb[2]/tsize, c->nb[3]/tsize,
                    a->nb[1]/tsize, a->nb[2]/tsize, a->nb[3]/tsize,
                    b->nb[1]/tsize, b->nb[2]/tsize, b->nb[3]/tsize, nelem);
        }
        else if (c->type == GGML_TYPE_F16) {
            auto tsize = sizeof(half);
            GGML_ASSERT(c->nb[0] == tsize && a->nb[0] == tsize && b->nb[0] == tsize && dst->nb[0] == tsize);
            fma_kernel<false><<<n_blocks, k_block_size, 0, ctx.stream()>>>(c->ne[0], (half *)dst->data,
                    (const half *)c->data, (const half *)a->data, (const half *)b->data,
                    c->ne[1], c->ne[2], c->ne[3], a->ne[1], a->ne[2], a->ne[3], b->ne[1], b->ne[2], b->ne[3],
                    d->nb[1]/tsize, d->nb[2]/tsize, d->nb[3]/tsize,
                    c->nb[1]/tsize, c->nb[2]/tsize, c->nb[3]/tsize,
                    a->nb[1]/tsize, a->nb[2]/tsize, a->nb[3]/tsize,
                    b->nb[1]/tsize, b->nb[2]/tsize, b->nb[3]/tsize, nelem);
        } else {
            auto tsize = sizeof(nv_bfloat16);
            GGML_ASSERT(c->nb[0] == tsize && a->nb[0] == tsize && b->nb[0] == tsize && dst->nb[0] == tsize);
            fma_kernel<false><<<n_blocks, k_block_size, 0, ctx.stream()>>>(c->ne[0], (nv_bfloat16 *)dst->data,
                    (const nv_bfloat16 *)c->data, (const nv_bfloat16 *)a->data, (const nv_bfloat16 *)b->data,
                    c->ne[1], c->ne[2], c->ne[3], a->ne[1], a->ne[2], a->ne[3], b->ne[1], b->ne[2], b->ne[3],
                    d->nb[1]/tsize, d->nb[2]/tsize, d->nb[3]/tsize,
                    c->nb[1]/tsize, c->nb[2]/tsize, c->nb[3]/tsize,
                    a->nb[1]/tsize, a->nb[2]/tsize, a->nb[3]/tsize,
                    b->nb[1]/tsize, b->nb[2]/tsize, b->nb[3]/tsize, nelem);
        }
    }
}
