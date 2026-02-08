#include "fill.cuh"
#include "convert.cuh"

#define CUDA_FILL_BLOCK_SIZE 256

template <typename T>
static __global__ void fill_kernel(T * dst, const int64_t k, const T value) {
    const int64_t i = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = value;
}

void ggml_cuda_op_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    float value = 0.0f;
    memcpy(&value, dst->op_params, sizeof(float));

    const int64_t k = ggml_nelements(dst);
    const int64_t num_blocks = (k + CUDA_FILL_BLOCK_SIZE - 1) / CUDA_FILL_BLOCK_SIZE;

    switch (dst->type) {
        case GGML_TYPE_F32:
            fill_kernel<<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, ctx.stream()>>>((float *) dst->data, k, value);
            break;
        case GGML_TYPE_F16:
            fill_kernel<<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, ctx.stream()>>>((half *) dst->data, k, ggml_cuda_cast<half>(value));
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
