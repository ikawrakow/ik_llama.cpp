#include "sumrows.cuh"

static __global__ void k_sum_rows_f32(const float * x, float * dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float sum = 0.0f;
    for (int i = col; i < ncols; i += blockDim.x) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum);

    if (col == 0) {
        dst[row] = sum;
    }
}

static __global__ void k_sum_rows_nc_f32(const char * x, char * y, const int ncols,
        size_t nb00, size_t nb01, size_t nb02, size_t nb03, size_t nb1, size_t nb2, size_t nb3) {

    const char * src = x + nb03*blockIdx.z + nb02*blockIdx.y + nb01*blockIdx.x;
    float * dst = (float *)(y + nb3*blockIdx.z  + nb2*blockIdx.y  + nb1*blockIdx.x);
    const int col = threadIdx.x;

    float sum = 0.0f;
    for (int i = col; i < ncols; i += blockDim.x) {
        sum += *(const float *)(src + i*nb00);
    }

    sum = warp_reduce_sum(sum);

    if (col == 0) {
        dst[0] = sum;
    }
}

static __global__ void k_sum_rows_div_f32(const float * __restrict__ x, float * __restrict__ dst, const int ncols, float s, float b) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float sum = 0.0f;
    for (int i = col; i < ncols; i += blockDim.x) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum);

    //sum = s*sum + b;

    float norm = sum > 0 ? 1/sum : 0.0f;
    for (int i = col; i < ncols; i += blockDim.x) {
        dst[row * ncols + i] = x[row * ncols + i] * norm;
    }
    //for (int i = col; i < ncols; i += blockDim.x) {
    //    dst[row * ncols + i] = x[row * ncols + i] / sum;
    //}
}

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    k_sum_rows_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
}
static void sum_rows_f32_cuda_nc(const char * x, char * dst, int ne0, int ne1, int ne2, int ne3,
        size_t nb00, size_t nb01, size_t nb02, size_t nb03, size_t nb1, size_t nb2, size_t nb3, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(ne1, ne2, ne3);
    k_sum_rows_nc_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ne0, nb00, nb01, nb02, nb03, nb1, nb2, nb3);
}

static void sum_rows_div_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, float s, float b, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    k_sum_rows_div_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, s, b);
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    sum_rows_f32_cuda(src0_d, dst_d, ncols, nrows, stream);

}

void ggml_cuda_op_sum_rows_nc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0]->src[0];
    GGML_ASSERT(src->op == GGML_OP_TRANSPOSE);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && src->type == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();

    sum_rows_f32_cuda_nc((const char *)src->data, (char *)dst->data, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
                src->nb[0], src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3], stream);
}

void ggml_cuda_op_sum_rows_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    float s = 1, b = 0;
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[1]->op == GGML_OP_SUM_ROWS || dst->src[1]->op == GGML_OP_SCALE);
    if (dst->src[1]->op == GGML_OP_SCALE) {
        GGML_ASSERT(dst->src[1]->src[0]->op == GGML_OP_SUM_ROWS);
        auto params = (const float *)dst->src[1]->op_params;
        s = params[0];
        b = params[1];
    }

    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    sum_rows_div_f32_cuda(src0_d, dst_d, ncols, nrows, s, b, stream);
}
