#include "cumsum.cuh"

#define CUDA_CUMSUM_BLOCK_SIZE 256

static __global__ void cumsum_f32_kernel(
        const float * src, float * dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s00,  const int64_t s01,  const int64_t s02,  const int64_t s03,
        const int64_t d0,   const int64_t d1,   const int64_t d2,   const int64_t d3) {
    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y;
    const int64_t i3 = blockIdx.z;

    if (i1 >= ne01 || i2 >= ne02 || i3 >= ne03) {
        return;
    }

    const float * src_row = src + i1 * s01 + i2 * s02 + i3 * s03;
    float * dst_row = dst + i1 * d1 + i2 * d2 + i3 * d3;

    extern __shared__ float s_scan[];

    float carry = 0.0f;
    for (int64_t start = 0; start < ne00; start += blockDim.x) {
        const int tile_n = (int) ((ne00 - start) < (int64_t) blockDim.x ? (ne00 - start) : (int64_t) blockDim.x);

        float value = 0.0f;
        if (threadIdx.x < tile_n) {
            value = src_row[(start + threadIdx.x) * s00];
        }
        s_scan[threadIdx.x] = value;
        __syncthreads();

        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            float add = 0.0f;
            if (threadIdx.x >= offset) {
                add = s_scan[threadIdx.x - offset];
            }
            __syncthreads();
            if (threadIdx.x >= offset) {
                s_scan[threadIdx.x] += add;
            }
            __syncthreads();
        }

        if (threadIdx.x < tile_n) {
            dst_row[(start + threadIdx.x) * d0] = s_scan[threadIdx.x] + carry;
        }

        __syncthreads();
        if (threadIdx.x == tile_n - 1) {
            carry += s_scan[threadIdx.x];
        }
        __syncthreads();
    }
}

void ggml_cuda_op_cumsum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int block_size = WARP_SIZE;
    while (block_size < src0->ne[0] && block_size < CUDA_CUMSUM_BLOCK_SIZE) {
        block_size <<= 1;
    }

    dim3 grid_dims(src0->ne[1], src0->ne[2], src0->ne[3]);
    cumsum_f32_kernel<<<grid_dims, block_size, block_size * sizeof(float), ctx.stream()>>>(
        (const float *) src0->data,
        (float *) dst->data,
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        src0->nb[0] / sizeof(float), src0->nb[1] / sizeof(float), src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
        dst->nb[0] / sizeof(float), dst->nb[1] / sizeof(float), dst->nb[2] / sizeof(float), dst->nb[3] / sizeof(float));
}
