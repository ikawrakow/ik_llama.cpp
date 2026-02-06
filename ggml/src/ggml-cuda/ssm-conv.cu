#include "ssm-conv.cuh"

#define CUDA_SSM_CONV_BLOCK_SIZE 256

static __global__ void ssm_conv_init_states_f32_nc4(
        const float * src0,
        float * state,
        int nr,
        int n_kv) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq = blockIdx.y;

    if (row >= nr || seq >= n_kv) {
        return;
    }

    const float * src_row = src0 + (size_t) seq * nr * 3 + (size_t) row * 3;
    float * state_row = state + (size_t) seq * nr * 4 + (size_t) row * 4;

    state_row[1] = src_row[0];
    state_row[2] = src_row[1];
    state_row[3] = src_row[2];
}

static __global__ void ssm_conv_init_states_f32(
        const float * src0,
        float * state,
        int nc,
        int nr,
        int n_kv) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int seq = blockIdx.y;

    if (row >= nr || seq >= n_kv) {
        return;
    }

    const float * src_row = src0 + (size_t) seq * nr * (nc - 1) + (size_t) row * (nc - 1);
    float * state_row = state + (size_t) seq * nr * nc + (size_t) row * nc;

    for (int i0 = 0; i0 < nc - 1; ++i0) {
        state_row[1 + i0] = src_row[i0];
    }
}

static __global__ void ssm_conv_f32_kernel(
        const float * src0,
        const float * src1,
        const float * src2,
        const int32_t * src3,
        float * dst_x,
        float * dst_state,
        int nc,
        int nr,
        int n_t,
        int n_kv,
        int src1_nb1,
        int src3_nb1) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nr) {
        return;
    }

    const float * c_row = src2 + (size_t) row * nc;

    for (int t = 0; t < n_t; ++t) {
        const int32_t * sq = src3 + (size_t) t * src3_nb1;
        const int seq0 = sq[0];

        if (seq0 < 0 || seq0 >= n_kv) {
            continue;
        }

        float * state_row = dst_state + (size_t) seq0 * nr * nc + (size_t) row * nc;
        const float * src_state_row;
        if (t == 0) {
            src_state_row = src0 + (size_t) seq0 * nr * (nc - 1) + (size_t) row * (nc - 1);
        } else {
            src_state_row = state_row + 1;
        }

        for (int i0 = 0; i0 < nc - 1; ++i0) {
            state_row[i0] = src_state_row[i0];
        }
        state_row[nc - 1] = src1[row + (size_t) t * src1_nb1];

        for (int i3 = 1; i3 < n_kv; ++i3) {
            const int seq = sq[i3];
            if (seq < 0 || seq >= n_kv) {
                break;
            }

            float * state_row_copy = dst_state + (size_t) seq * nr * nc + (size_t) row * nc;
            for (int i0 = 0; i0 < nc; ++i0) {
                state_row_copy[i0] = state_row[i0];
            }
        }

        float sumf = 0.0f;
        for (int i0 = 0; i0 < nc; ++i0) {
            sumf += state_row[i0] * c_row[i0];
        }
        dst_x[row + (size_t) t * nr] = sumf;
    }
}

template <bool has_multi_seq>
static __global__ void ssm_conv_f32_kernel_nc4(
        const float * src0,
        const float * src1,
        const float * src2,
        const int32_t * src3,
        float * dst_x,
        float * dst_state,
        int nr,
        int n_t,
        int n_kv,
        int src1_nb1,
        int src3_nb1) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nr) {
        return;
    }

    const float * c_row = src2 + (size_t) row * 4;
    const float c0 = c_row[0];
    const float c1 = c_row[1];
    const float c2 = c_row[2];
    const float c3 = c_row[3];

    for (int t = 0; t < n_t; ++t) {
        const int32_t * sq = src3 + (size_t) t * src3_nb1;
        const int seq0 = sq[0];

        if (seq0 < 0 || seq0 >= n_kv) {
            continue;
        }

        float * state_row = dst_state + (size_t) seq0 * nr * 4 + (size_t) row * 4;

        const float * src_state_row;
        if (t == 0) {
            src_state_row = src0 + (size_t) seq0 * nr * 3 + (size_t) row * 3;
        } else {
            src_state_row = state_row + 1;
        }

        const float s0 = src_state_row[0];
        const float s1 = src_state_row[1];
        const float s2 = src_state_row[2];
        const float x = src1[row + (size_t) t * src1_nb1];

        state_row[0] = s0;
        state_row[1] = s1;
        state_row[2] = s2;
        state_row[3] = x;

        if constexpr (has_multi_seq) {
            for (int i3 = 1; i3 < n_kv; ++i3) {
                const int seq = sq[i3];
                if (seq < 0 || seq >= n_kv) {
                    break;
                }

                float * state_row_copy = dst_state + (size_t) seq * nr * 4 + (size_t) row * 4;
                state_row_copy[0] = s0;
                state_row_copy[1] = s1;
                state_row_copy[2] = s2;
                state_row_copy[3] = x;
            }
        }

        dst_x[row + (size_t) t * nr] = s0 * c0 + s1 * c1 + s2 * c2 + x * c3;
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // conv_state: [d_conv - 1, d_inner, n_kv]
    const ggml_tensor * src1 = dst->src[1]; // x: [d_inner, n_tokens]
    const ggml_tensor * src2 = dst->src[2]; // conv1d.weight: [d_conv, d_inner]
    const ggml_tensor * src3 = dst->src[3]; // state_seq: [n_kv, n_tokens]

    const int nc   = src2->ne[0];
    const int nr   = src0->ne[1];
    const int n_t  = src1->ne[1];
    const int n_kv = src0->ne[2];

    GGML_ASSERT((int64_t) nr * n_t + (int64_t) nc * nr * n_kv == ggml_nelements(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT(src3->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(int32_t));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));
    GGML_ASSERT(src2->nb[1] == src2->ne[0] * sizeof(float));
    GGML_ASSERT(src2->nb[2] == src2->ne[1] * src2->ne[0] * sizeof(float));

    GGML_ASSERT(src2->ne[0] == src0->ne[0] + 1);
    GGML_ASSERT(src2->ne[1] == src0->ne[1]);
    GGML_ASSERT(src1->ne[0] == src0->ne[1]);
    GGML_ASSERT(src3->ne[0] == src0->ne[2]);
    GGML_ASSERT(src3->ne[1] == src1->ne[1]);

    float * dst_data = (float *) dst->data;
    float * dst_x = dst_data;
    float * dst_state = dst_data + (size_t) nr * n_t;

    const dim3 block_dims(CUDA_SSM_CONV_BLOCK_SIZE, 1, 1);
    const dim3 row_grid((nr + CUDA_SSM_CONV_BLOCK_SIZE - 1) / CUDA_SSM_CONV_BLOCK_SIZE, 1, 1);

    if (n_kv > 1) {
        const dim3 init_grid(row_grid.x, n_kv, 1);
        if (nc == 4) {
            ssm_conv_init_states_f32_nc4<<<init_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                dst_state,
                nr, n_kv);
        } else {
            ssm_conv_init_states_f32<<<init_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                dst_state,
                nc, nr, n_kv);
        }
    }

    if (nc == 4) {
        if (n_kv > 1) {
            ssm_conv_f32_kernel_nc4<true><<<row_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                (const float *) src1->data,
                (const float *) src2->data,
                (const int32_t *) src3->data,
                dst_x,
                dst_state,
                nr, n_t, n_kv,
                src1->nb[1] / sizeof(float),
                src3->nb[1] / sizeof(int32_t));
        } else {
            ssm_conv_f32_kernel_nc4<false><<<row_grid, block_dims, 0, ctx.stream()>>>(
                (const float *) src0->data,
                (const float *) src1->data,
                (const float *) src2->data,
                (const int32_t *) src3->data,
                dst_x,
                dst_state,
                nr, n_t, n_kv,
                src1->nb[1] / sizeof(float),
                src3->nb[1] / sizeof(int32_t));
        }
    } else {
        ssm_conv_f32_kernel<<<row_grid, block_dims, 0, ctx.stream()>>>(
            (const float *) src0->data,
            (const float *) src1->data,
            (const float *) src2->data,
            (const int32_t *) src3->data,
            dst_x,
            dst_state,
            nc, nr, n_t, n_kv,
            src1->nb[1] / sizeof(float),
            src3->nb[1] / sizeof(int32_t));
    }
}
