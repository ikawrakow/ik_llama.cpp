#include "ds4_comp.cuh"

static __global__ void k_ds4_comp(int ne0, int nblock, int ratio, int nidx,
        size_t state_stride, size_t score_stride,
        const float * __restrict__ state, const float * __restrict__ score, const int * __restrict__ idx, float * dst) {

    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int ib = ii / ne0;
    if (ib >= nblock) {
        return;
    }
    int i0 = ii % ne0;

    idx += ratio*ib;
    int row_p = idx[0];
    int row_c = idx[nidx];
    float vp = score[row_p*score_stride + i0];
    float vc = score[row_c*score_stride + i0 + ne0];
    float max_v = max(vp, vc);
    for (int ir = 1; ir < ratio; ++ir) {
        row_p = idx[ir];
        row_c = idx[ir+nidx];
        vp = score[row_p*score_stride + i0];
        vc = score[row_c*score_stride + i0 + ne0];
        max_v = max(max_v, max(vp, vc));
    }
    float sum_num = 0.0f, sum_den = 0.0f;
    for (int ir = 0; ir < ratio; ++ir) {
        row_p = idx[ir];
        row_c = idx[ir+nidx];
        vp = score[row_p*score_stride + i0];
        vc = score[row_c*score_stride + i0 + ne0];
        float sp = state[row_p*state_stride + i0];
        float sc = state[row_c*state_stride + i0 + ne0];
        float wp = expf(vp - max_v);
        float wc = expf(vc - max_v);
        sum_den += wp + wc;
        sum_num += wp*sp + wc*sc;
    }
    dst[ib*ne0 + i0] = sum_num / sum_den;
}

static __global__ void k_ds4_comp_4(int ne0, int nblock, int nidx,
        size_t state_stride, size_t score_stride,
        const float * __restrict__ state, const float * __restrict__ score, const int * __restrict__ idx, float * dst) {

    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int ib = ii / ne0;
    if (ib >= nblock) {
        return;
    }
    int i0 = ii % ne0;

    idx += 4*ib;

    float values[16];
    #pragma unroll
    for (int ir = 0; ir < 4; ++ir) {
        int row_p = idx[ir];
        int row_c = idx[ir+nidx];
        values[2*ir+0] = score[row_p*score_stride + i0];
        values[2*ir+1] = score[row_c*score_stride + i0 + ne0];
        values[2*ir+8] = state[row_p*state_stride + i0];
        values[2*ir+9] = state[row_c*state_stride + i0 + ne0];
    }
    float max_v = values[0];
    #pragma unroll
    for (int ir = 1; ir < 8; ++ir) max_v = max(max_v, values[ir]);

    float sum_num = 0.0f, sum_den = 0.0f;
    #pragma unroll
    for (int ir = 0; ir < 8; ++ir) {
        float w = expf(values[ir] - max_v);
        sum_den += w;
        sum_num += w*values[ir+8];
    }
    dst[ib*ne0 + i0] = sum_num / sum_den;
}

static __global__ void k_ds4_comp_type1(int ne0, int nblock, int ratio,
        size_t state_stride, size_t score_stride,
        const float * __restrict__ state, const float * __restrict__ score, const int * __restrict__ idx, float * dst) {

    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int ib = ii / ne0;
    if (ib >= nblock) {
        return;
    }
    int i0 = ii % ne0;

    idx += ratio*ib;
    int row = idx[0];
    float max_v = score[row*score_stride + i0];
    for (int ir = 1; ir < ratio; ++ir) {
        row = idx[ir];
        float v = score[row*score_stride + i0];
        max_v = max(max_v, v);
    }
    float sum_num = 0.0f, sum_den = 0.0f;
    for (int ir = 0; ir < ratio; ++ir) {
        row = idx[ir];
        float v = score[row*score_stride + i0];
        float s = state[row*state_stride + i0];
        float w = expf(v - max_v);
        sum_den += w;
        sum_num += w*s;
    }
    dst[ib*ne0 + i0] = sum_num / sum_den;
}

void ggml_cuda_op_ds4_comp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int k_block_size = 128;
    auto * state = dst->src[0];
    auto * score = dst->src[1];
    auto *   idx = dst->src[2];
    GGML_ASSERT(state->type == GGML_TYPE_F32);
    GGML_ASSERT(score->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(score, state));
    GGML_ASSERT(state->ne[2] == 1 && state->ne[3] == 1);
    GGML_ASSERT(  idx->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_nrows(idx) == 1);

    int type = dst->op_params[0];
    int nblock = dst->ne[1];

    if (type == 0) {
        GGML_ASSERT(state->ne[0] % 64 == 0);
        GGML_ASSERT(dst->ne[0] == state->ne[0]/2);

        int ratio  = idx->ne[0] / (2*nblock);

        GGML_ASSERT(idx->ne[0] % (2*ratio) == 0);

        int ne0 = dst->ne[0];
        int nelem = ne0 * nblock;
        int nb = (nelem + k_block_size - 1)/k_block_size;

        if (ratio == 4) {
            k_ds4_comp_4<<<nb, k_block_size, 0, ctx.stream()>>>(ne0, nblock, idx->ne[0]/2,
                    state->nb[1]/sizeof(float), score->nb[1]/sizeof(float),
                    (const float *)state->data, (const float *)score->data, (const int *)idx->data, (float *)dst->data);
        } else {
            k_ds4_comp<<<nb, k_block_size, 0, ctx.stream()>>>(ne0, nblock, ratio, idx->ne[0]/2,
                    state->nb[1]/sizeof(float), score->nb[1]/sizeof(float),
                    (const float *)state->data, (const float *)score->data, (const int *)idx->data, (float *)dst->data);
        }
    } else {
        GGML_ASSERT(state->ne[0] % 32 == 0);
        GGML_ASSERT(dst->ne[0] == state->ne[0]);

        int ratio  = idx->ne[0] / nblock;

        GGML_ASSERT(idx->ne[0] % ratio == 0);

        int ne0 = dst->ne[0];
        int nelem = ne0 * nblock;
        int nb = (nelem + k_block_size - 1)/k_block_size;

        k_ds4_comp_type1<<<nb, k_block_size, 0, ctx.stream()>>>(ne0, nblock, ratio,
                state->nb[1]/sizeof(float), score->nb[1]/sizeof(float),
                (const float *)state->data, (const float *)score->data, (const int *)idx->data, (float *)dst->data);
    }

}
