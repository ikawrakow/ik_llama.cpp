#include "batched_mix.cuh"
#include "../ggml-batched-mix.h"

#include <climits>

static __global__ void batched_mix_f32(
        const char * __restrict__ r,
        const char * __restrict__ mix,
        float * __restrict__ dst,
        int64_t total,
        int64_t D,
        int64_t J,
        int64_t O,
        size_t r_nb1,
        size_t r_nb2,
        size_t mix_nb1,
        size_t mix_nb2) {
    const int64_t index = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    const int64_t d = index % D;
    const int64_t q = index / D;
    const int64_t o = q % O;
    const int64_t t = q / O;
    float sum = 0.0f;
    for (int64_t j = 0; j < J; ++j) {
        const float rv = *(const float *)(r +
                (size_t) d*sizeof(float) + (size_t) j*r_nb1 + (size_t) t*r_nb2);
        const float mv = *(const float *)(mix +
                (size_t) j*sizeof(float) + (size_t) o*mix_nb1 + (size_t) t*mix_nb2);
        sum += rv*mv;
    }
    dst[index] = sum;
}

bool ggml_cuda_batched_mix_is_supported(const ggml_tensor * op) {
    const ggml_tensor * r   = op->src[0];
    const ggml_tensor * mix = op->src[1];
    if (r == nullptr || mix == nullptr) {
        return false;
    }
    for (int i = 2; i < GGML_MAX_SRC; ++i) {
        if (op->src[i] != nullptr) {
            return false;
        }
    }
    for (size_t i = 0; i < GGML_MAX_OP_PARAMS/sizeof(op->op_params[0]); ++i) {
        if (op->op_params[i] != 0) {
            return false;
        }
    }
    if (!ggml_batched_mix_f32_layout_is_valid(r) ||
            !ggml_batched_mix_f32_layout_is_valid(mix) ||
            !ggml_batched_mix_f32_layout_is_valid(op)) {
        return false;
    }
    if (r->ne[0] <= 0 || r->ne[1] < 1 || r->ne[1] > 8 || r->ne[2] <= 0 ||
            r->ne[3] != 1 || mix->ne[0] != r->ne[1] || mix->ne[1] <= 0 ||
            mix->ne[2] != r->ne[2] || mix->ne[3] != 1) {
        return false;
    }
    if (op->ne[0] != r->ne[0] || op->ne[1] != mix->ne[1] ||
            op->ne[2] != r->ne[2] || op->ne[3] != 1 || !ggml_is_contiguous(op)) {
        return false;
    }
    constexpr int64_t block_size = 256;
    return ggml_nelements(op) <= (int64_t) INT_MAX*block_size;
}

void ggml_cuda_op_batched_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * r   = dst->src[0];
    const ggml_tensor * mix = dst->src[1];
    GGML_ASSERT(r != nullptr && mix != nullptr);
    for (int i = 2; i < GGML_MAX_SRC; ++i) {
        GGML_ASSERT(dst->src[i] == nullptr);
    }
    GGML_ASSERT(dst->op_params[0] == 0 && dst->op_params[1] == 0);
    for (size_t i = 2; i < GGML_MAX_OP_PARAMS/sizeof(dst->op_params[0]); ++i) {
        GGML_ASSERT(dst->op_params[i] == 0);
    }
    GGML_ASSERT(ggml_batched_mix_f32_layout_is_valid(r));
    GGML_ASSERT(ggml_batched_mix_f32_layout_is_valid(mix));
    GGML_ASSERT(ggml_batched_mix_f32_layout_is_valid(dst));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t D = r->ne[0];
    const int64_t J = r->ne[1];
    const int64_t O = mix->ne[1];
    const int64_t T = r->ne[2];
    GGML_ASSERT(D > 0 && J >= 1 && J <= 8 && O > 0 && T > 0);
    GGML_ASSERT(r->ne[3] == 1 && mix->ne[0] == J && mix->ne[2] == T && mix->ne[3] == 1);
    GGML_ASSERT(dst->ne[0] == D && dst->ne[1] == O && dst->ne[2] == T && dst->ne[3] == 1);

    constexpr int block_size = 256;
    const int64_t total = ggml_nelements(dst);
    const int64_t grid_size = (total + block_size - 1)/block_size;
    GGML_ASSERT(grid_size <= INT_MAX);
    batched_mix_f32<<<(unsigned int) grid_size, block_size, 0, ctx.stream()>>>(
            (const char *) r->data, (const char *) mix->data, (float *) dst->data,
            total, D, J, O, r->nb[1], r->nb[2], mix->nb[1], mix->nb[2]);
}
