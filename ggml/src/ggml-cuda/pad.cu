#include "pad.cuh"

static __global__ void pad_f32(const float * x, float * dst, const int ne0, const int ne00, const int ne01, const int ne02, const int ne03) {
    // blockIdx.z: idx of ne2*ne3, aka ne02*ne03
    // blockIdx.y: idx of ne1
    // blockIDx.x: idx of ne0 / BLOCK_SIZE
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    // operation
    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;
    if (nidx < ne00 && blockIdx.y < ne01 && blockIdx.z < ne02*ne03) {
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        dst[offset_dst] = 0.0f;
    }
}

static void pad_f32_cuda(const float * x, float * dst,
    const int ne00, const int ne01, const int ne02, const int ne03,
    const int ne0, const int ne1, const int ne2, const int ne3, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2*ne3);
    pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(x, dst, ne0, ne00, ne01, ne02, ne03);
}

template <int dim>
static __global__ void pad_f32_nc(const char * cx, float * dst, int nelem,
        int ne0, int ne1, int ne2, int ne3, int ne00, int ne01, int ne02, int ne03,
        size_t nb00, size_t nb01, size_t nb02, size_t nb03) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    int ii = i;
    int i3 = ii/(ne0*ne1*ne2); ii -= i3*ne0*ne1*ne2;
    int i2 = ii/(ne0*ne1    ); ii -= i2*ne0*ne1;
    int i1 = ii/(ne0        );
    int i0 = ii - i1*ne0;

    if constexpr (dim == 0) {
        dst[i] = i0 < ne00 ? *(const float *)(cx + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03) : 0.0f;
    }
    else if constexpr (dim == 1) {
        dst[i] = i1 < ne01 ? *(const float *)(cx + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03) : 0.0f;
    }
    else if constexpr (dim == 2) {
        dst[i] = i2 < ne02 ? *(const float *)(cx + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03) : 0.0f;
    }
    else if constexpr (dim == 3) {
        dst[i] = i3 < ne03 ? *(const float *)(cx + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03) : 0.0f;
    }
    else if constexpr (dim == 4) {
        dst[i] = *(const float *)(cx + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    }
    else {
        dst[i] = i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03 ? *(const float *)(cx + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03) : 0.0f;
    }
}

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    if (ggml_is_contiguous(src0)) {
        GGML_ASSERT(src0->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors

        pad_f32_cuda(src0_d, dst_d,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
        return;
    }

    int npad = 0; int pad_dim = -1;
    for (int i = 0; i < 4; ++i) {
        if (dst->ne[i] > src0->ne[i]) {
            ++npad; pad_dim = i;
        }
    }
    //if (npad == 0) {
    //    printf("Oops: npad = 0: %ld vs %ld, %ld vx %ld, %ld vs %ld, %ld vs %ld\n", dst->ne[0], src0->ne[0], dst->ne[1], src0->ne[1], dst->ne[2], src0->ne[2], dst->ne[3], src0->ne[3]);
    //}
    //GGML_ASSERT(npad > 0);

    constexpr int kBlockSize = 256;
    int nelem = ggml_nelements(dst);
    int nblock = (nelem + kBlockSize - 1)/kBlockSize;

    if (npad == 0) {
        //printf("%s: %ld x %ld x %ld x %ld; %zu x %zu x %zu x %zu\n", src0->name, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
        pad_f32_nc<4><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    } else if (npad == 1) {
        if (pad_dim == 0) {
            pad_f32_nc<0><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
        } else if (pad_dim == 1) {
            pad_f32_nc<1><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
        } else if (pad_dim == 2) {
            pad_f32_nc<2><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
        } else if (pad_dim == 3) {
            pad_f32_nc<3><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
        } else {
            GGML_ABORT("Fatal error");
        }
    } else {
        pad_f32_nc<-1><<<nblock, kBlockSize, 0, ctx.stream()>>>((const char *)src0->data, (float *)dst->data, nelem,
                    dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    }
}
