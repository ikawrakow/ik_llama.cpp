#include "blend.cuh"

#define CUDA_BLEND_BLOCK_SIZE 256

template <typename Data, typename Idx>
static __global__ void kernel_blend(int n, int nidx, const Data * x, const Idx * idx, Data * y, float c,
        int ne1, int ne2,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb1,  size_t nb2,  size_t nb3) {
    Data b;
    if constexpr (std::is_same_v<Data, nv_bfloat16>) {
        b = __float2bfloat16(c);
    } else {
        b = (Data)c;
    }
    int ii = blockIdx.x;
    int i3 = ii / (ne1*ne2); ii -= i3*ne1*ne2;
    int i2 = ii / ne1;
    int i1 = ii - i2*ne1;
    auto x_row = x + i1*nb01 + i2*nb02 + i3*nb03;
    auto y_row = y + i1*nb1  + i2*nb2  + i3*nb3;
    auto idx_row = idx + i1*nb11 + i2*nb12 + i3*nb13;

    if (x_row != y_row) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            y_row[i] = x_row[i];
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < nidx; i += blockDim.x) {
        y_row[idx_row[i]] = b;
    }
}

void ggml_cuda_op_blend(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src1->type == GGML_TYPE_I32 || src1->type == GGML_TYPE_I64);
    GGML_ASSERT(src1->ne[1] == src0->ne[1] && src1->ne[2] == src0->ne[2] && src1->ne[3] == src0->ne[3]);
    GGML_ASSERT(src1->ne[0] <= src0->ne[0]);

    float c;
    memcpy(&c, dst->op_params, sizeof(c));

    auto nrows = ggml_nrows(dst);
    dim3 grid_dims(nrows, 1, 1);
    dim3 block_size(CUDA_BLEND_BLOCK_SIZE, 1, 1);
    if (src1->type == GGML_TYPE_I32) {
        auto idx = (const int32_t *)src1->data;
        if (src0->type == GGML_TYPE_F32) {
            kernel_blend<<<grid_dims, block_size, 0, ctx.stream()>>>(src0->ne[0], src1->ne[0],
                    (const float *)src0->data, idx, (float *)dst->data, c, src0->ne[1], src0->ne[2],
                    src0->nb[1]/sizeof(float), src0->nb[2]/sizeof(float), src0->nb[3]/sizeof(float),
                    src1->nb[1]/sizeof(int32_t), src1->nb[2]/sizeof(int32_t), src1->nb[3]/sizeof(int32_t),
                    dst->nb[1]/sizeof(float), dst->nb[2]/sizeof(float), dst->nb[3]/sizeof(float));
        }
        else if (src0->type == GGML_TYPE_F16) {
            kernel_blend<<<grid_dims, block_size, 0, ctx.stream()>>>(src0->ne[0], src1->ne[0],
                    (const half *)src0->data, idx, (half *)dst->data, c, src0->ne[1], src0->ne[2],
                    src0->nb[1]/sizeof(half), src0->nb[2]/sizeof(half), src0->nb[3]/sizeof(half),
                    src1->nb[1]/sizeof(int32_t), src1->nb[2]/sizeof(int32_t), src1->nb[3]/sizeof(int32_t),
                    dst->nb[1]/sizeof(half), dst->nb[2]/sizeof(half), dst->nb[3]/sizeof(half));
        }
        else {
            kernel_blend<<<grid_dims, block_size, 0, ctx.stream()>>>(src0->ne[0], src1->ne[0],
                    (const nv_bfloat16 *)src0->data, idx, (nv_bfloat16 *)dst->data, c, src0->ne[1], src0->ne[2],
                    src0->nb[1]/sizeof(nv_bfloat16), src0->nb[2]/sizeof(nv_bfloat16), src0->nb[3]/sizeof(nv_bfloat16),
                    src1->nb[1]/sizeof(int32_t), src1->nb[2]/sizeof(int32_t), src1->nb[3]/sizeof(int32_t),
                    dst->nb[1]/sizeof(nv_bfloat16), dst->nb[2]/sizeof(nv_bfloat16), dst->nb[3]/sizeof(nv_bfloat16));
        }
    } else {
        auto idx = (const int64_t *)src1->data;
        if (src0->type == GGML_TYPE_F32) {
            kernel_blend<<<grid_dims, block_size, 0, ctx.stream()>>>(src0->ne[0], src1->ne[0],
                    (const float *)src0->data, idx, (float *)dst->data, c, src0->ne[1], src0->ne[2],
                    src0->nb[1]/sizeof(float), src0->nb[2]/sizeof(float), src0->nb[3]/sizeof(float),
                    src1->nb[1]/sizeof(int64_t), src1->nb[2]/sizeof(int64_t), src1->nb[3]/sizeof(int64_t),
                    dst->nb[1]/sizeof(float), dst->nb[2]/sizeof(float), dst->nb[3]/sizeof(float));
        }
        else if (src0->type == GGML_TYPE_F16) {
            kernel_blend<<<grid_dims, block_size, 0, ctx.stream()>>>(src0->ne[0], src1->ne[0],
                    (const half *)src0->data, idx, (half *)dst->data, c, src0->ne[1], src0->ne[2],
                    src0->nb[1]/sizeof(half), src0->nb[2]/sizeof(half), src0->nb[3]/sizeof(half),
                    src1->nb[1]/sizeof(int64_t), src1->nb[2]/sizeof(int64_t), src1->nb[3]/sizeof(int64_t),
                    dst->nb[1]/sizeof(half), dst->nb[2]/sizeof(half), dst->nb[3]/sizeof(half));
        }
        else {
            kernel_blend<<<grid_dims, block_size, 0, ctx.stream()>>>(src0->ne[0], src1->ne[0],
                    (const nv_bfloat16 *)src0->data, idx, (nv_bfloat16 *)dst->data, c, src0->ne[1], src0->ne[2],
                    src0->nb[1]/sizeof(nv_bfloat16), src0->nb[2]/sizeof(nv_bfloat16), src0->nb[3]/sizeof(nv_bfloat16),
                    src1->nb[1]/sizeof(int64_t), src1->nb[2]/sizeof(int64_t), src1->nb[3]/sizeof(int64_t),
                    dst->nb[1]/sizeof(nv_bfloat16), dst->nb[2]/sizeof(nv_bfloat16), dst->nb[3]/sizeof(nv_bfloat16));
        }
    }

}
