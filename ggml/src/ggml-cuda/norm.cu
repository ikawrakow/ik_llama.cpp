#include "norm.cuh"

template <int block_size>
static __global__ void norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var);
    if (block_size > WARP_SIZE) {
        __shared__ float2 s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        mean_var = s_sum[lane_id];
        mean_var = warp_reduce_sum(mean_var);
    }

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = (x[row*ncols + col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    int start = blockIdx.x * group_size;
    int end = start + group_size;

    start += threadIdx.x;

    if (end >= ne_elements) {
        end = ne_elements;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    float variance = tmp / group_size;
    float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size>
static __global__ void rms_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = scale * x[row*ncols + col];
    }
}

template <int block_size>
static __global__ void fused_rms_norm_f32(const float * x, const float * y, const float * z, float * dst, const int ncols,
        const int64_t ne0[4], const int64_t ne1[4], const int64_t ne2[4], const size_t nb1[4], const size_t nb2[4], const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    int64_t i03 = row/ne0[3];
    int64_t i02 = (row - i03*ne0[3])/ne0[2];
    int64_t i01 = (row - i03*ne0[3] - i02*ne0[2])/ne0[1];

    if (y && z) {
        int64_t i13 = i03 % ne1[3];
        int64_t i12 = i02 % ne1[2];
        int64_t i11 = i01 % ne1[1];
        int64_t i23 = i03 % ne2[3];
        int64_t i22 = i02 % ne2[2];
        int64_t i21 = i01 % ne2[1];
        const float * yr = (const float *)((const char *)x + i13*nb1[3] + i12*nb1[2] + i11*nb1[11]);
        const float * zr = (const float *)((const char *)z + i23*nb2[3] + i22*nb2[2] + i21*nb1[11]);
        for (int col = tid; col < ncols; col += block_size) {
            int64_t i01 = col % ne1[0];
            int64_t i02 = col % ne2[0];
            dst[row*ncols + col] = scale * yr[i01] * x[row*ncols + col] + zr[i02];
        }
    }
    else if (y) {
        int64_t i13 = i03 % ne1[3];
        int64_t i12 = i02 % ne1[2];
        int64_t i11 = i01 % ne1[1];
        const float * yr = (const float *)((const char *)x + i13*nb1[3] + i12*nb1[2] + i11*nb1[11]);
        for (int col = tid; col < ncols; col += block_size) {
            int64_t i01 = col % ne1[0];
            dst[row*ncols + col] = scale * yr[i01] * x[row*ncols + col];
        }
    }
    else {
        int64_t i23 = i03 % ne2[3];
        int64_t i22 = i02 % ne2[2];
        int64_t i21 = i01 % ne2[1];
        const float * zr = (const float *)((const char *)z + i23*nb2[3] + i22*nb2[2] + i21*nb1[11]);
        for (int col = tid; col < ncols; col += block_size) {
            int64_t i02 = col % ne2[0];
            dst[row*ncols + col] = scale * x[row*ncols + col] + zr[i02];
        }
    }
}

static void norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    }
}

static void group_norm_f32_cuda(const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

static void rms_norm_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    }
}

    //fused_rms_norm_f32_cuda(src0_d, src1_d, src2_d, dst_d, ne00, nrows, eps, ne0, ne1, ne2, nb1, nb2, stream);
static void fused_rms_norm_f32_cuda(const float * x, const float * y, const float * z, float * dst,
        const int ncols, const int nrows, const float eps, const int64_t ne0[4], const int64_t ne1[4], const int64_t ne2[4],
        const size_t nb1[4], const size_t nb2[4], cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        fused_rms_norm_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(x, y, z, dst, ncols, ne0, ne1, ne2, nb1, nb2, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        fused_rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, y, z, dst, ncols, ne0, ne1, ne2, nb1, nb2, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    norm_f32_cuda(src0_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    rms_norm_f32_cuda(src0_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_fused_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!dst->src[1] && !dst->src[2]) {
        ggml_cuda_op_rms_norm(ctx, dst);
        return;
    }
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!dst->src[1] || dst->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(!dst->src[2] || dst->src[2]->type == GGML_TYPE_F32);
    if (dst->src[1] && dst->src[2]) {
        GGML_ASSERT(dst->src[1]->ne[0] == dst->src[2]->ne[0]);
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const float * src1_d = dst->src[1] ? (const float *)dst->src[1]->data : nullptr;
    const float * src2_d = dst->src[2] ? (const float *)dst->src[2]->data : nullptr;

    auto ne0 = src0->ne;
    auto ne1 = dst->src[1] ? dst->src[1]->ne : nullptr;
    auto ne2 = dst->src[2] ? dst->src[2]->ne : nullptr;
    auto nb1 = dst->src[1] ? dst->src[1]->nb : nullptr;
    auto nb2 = dst->src[2] ? dst->src[2]->nb : nullptr;

    fused_rms_norm_f32_cuda(src0_d, src1_d, src2_d, dst_d, ne00, nrows, eps, ne0, ne1, ne2, nb1, nb2, stream);
}
