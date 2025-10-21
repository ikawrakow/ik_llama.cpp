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
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = scale * x[row*ncols + col];
    }
}

template <int block_size>
static __global__ void rms_norm_f32_nc(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
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
        dst[col] = scale * x[col];
    }
}

template <int block_size>
static __global__ void fused_rms_norm_f32(const float * x, const float * y, float * dst, const int ncols, const float eps) {
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
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = scale * y[col] * x[row*ncols + col];
    }
}

template <int block_size>
static __global__ void fused_rms_norm_f32_nc(
        const float * x, const float * y, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    //const int channel   = blockIdx.y * blockDim.y + threadIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        //if constexpr (block_size == 1024) {
        //    tmp = s_sum[lane_id];
        //} else {
        //    tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        //}
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale * y[col] * x[col];
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
    constexpr int kBlockSize = 256;
    if (ncols < 1024) {
        const dim3 block_dims(kBlockSize, 1, 1);
        rms_norm_f32<kBlockSize><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, dst, ncols, eps);
    }
}

static void rms_norm_f32_nc_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_f32_nc<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f32_nc<1024><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void fused_rms_norm_f32_cuda(const float * x, const float * y, float * dst,
        const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < kBlockSize) {
        switch (ncols) {
            case  32: fused_rms_norm_f32< 32><<<nrows,  32, 0, stream>>>(x, y, dst, ncols, eps); break;
            case  64: fused_rms_norm_f32< 64><<<nrows,  64, 0, stream>>>(x, y, dst, ncols, eps); break;
            case  96: fused_rms_norm_f32< 96><<<nrows,  96, 0, stream>>>(x, y, dst, ncols, eps); break;
            case 128: fused_rms_norm_f32<128><<<nrows, 128, 0, stream>>>(x, y, dst, ncols, eps); break;
            case 160: fused_rms_norm_f32<160><<<nrows, 160, 0, stream>>>(x, y, dst, ncols, eps); break;
            case 192: fused_rms_norm_f32<192><<<nrows, 192, 0, stream>>>(x, y, dst, ncols, eps); break;
            default : fused_rms_norm_f32<224><<<nrows, 224, 0, stream>>>(x, y, dst, ncols, eps); break;
        }
    }
    else if (ncols < 1024) {
        const dim3 block_dims(kBlockSize, 1, 1);
        fused_rms_norm_f32<kBlockSize><<<nrows, block_dims, 0, stream>>>(x, y, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        fused_rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(x, y, dst, ncols, eps);
    }
}

static void fused_rms_norm_f32_nc_cuda(
        const float * x, const float * y, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        fused_rms_norm_f32_nc<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, y, dst, ncols, stride_row, stride_channel, stride_sample, eps);
        //constexpr int kBlockSize = 256;

        //if (nchannels%4 == 0) {
        //    const dim3 blocks_num(nrows, nchannels/4, nsamples);
        //    const dim3 block_dims(kBlockSize, 4, 1);
        //    fused_rms_norm_f32_nc<kBlockSize><<<blocks_num, block_dims, 0, stream>>>(x, y, dst, ncols, stride_row, stride_channel, stride_sample, eps);
        //} else {
        //    const dim3 block_dims(kBlockSize, 1, 1);
        //    fused_rms_norm_f32_nc<kBlockSize><<<blocks_num, block_dims, 0, stream>>>(x, y, dst, ncols, stride_row, stride_channel, stride_sample, eps);
        //}
    } else {
        const dim3 block_dims(1024, 1, 1);
        fused_rms_norm_f32_nc<1024><<<blocks_num, block_dims, 0, stream>>>(x, y, dst, ncols, stride_row, stride_channel, stride_sample, eps);
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

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];
    if (ggml_is_contiguous(src0)) {
        const int64_t nrows = ggml_nrows(src0);
        rms_norm_f32_cuda(src0_d, dst_d, ne00, nrows, eps, stream);
    } else {
        auto ts0 = ggml_type_size(src0->type);
        GGML_ASSERT(src0->nb[0] == ts0);
        auto s01 = src0->nb[1] / ts0;
        auto s02 = src0->nb[2] / ts0;
        auto s03 = src0->nb[3] / ts0;
        rms_norm_f32_nc_cuda(src0_d, dst_d, ne00, src0->ne[1], src0->ne[2], src0->ne[3], s01, s02, s03, eps, stream);
    }
}

void ggml_cuda_op_fused_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!dst->src[1]) {
        ggml_cuda_op_rms_norm(ctx, dst);
        return;
    }
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(ggml_nrows(src1) == 1);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];

    if (ggml_is_contiguous(src0)) {
        const int64_t nrows = ggml_nrows(src0);
        fused_rms_norm_f32_cuda(src0_d, src1_d, dst_d, ne00, nrows, eps, stream);
    } else {
        auto ts0 = ggml_type_size(src0->type);
        GGML_ASSERT(src0->nb[0] == ts0);
        auto s01 = src0->nb[1] / ts0;
        auto s02 = src0->nb[2] / ts0;
        auto s03 = src0->nb[3] / ts0;
        fused_rms_norm_f32_nc_cuda(src0_d, src1_d, dst_d, ne00, src0->ne[1], src0->ne[2], src0->ne[3], s01, s02, s03, eps, stream);
    }
}

template <int block_size>
static __global__ void fused_add_rms_norm_f32(const float * a, const float * b, const float * c,
        float * dst_add, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = a[row*ncols + col] + b[row*ncols + col];
        tmp += xi * xi;
        dst_add[row*ncols + col] = xi;
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
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = scale * c[col] * dst_add[row*ncols + col];
    }
}

template <int block_size>
static __global__ void fused_add_add_rms_norm_f32(const float * a1, const float * a2, const float * b, const float * c,
        float * dst_add, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = a1[row*ncols + col] + a2[row*ncols + col] + b[row*ncols + col];
        tmp += xi * xi;
        dst_add[row*ncols + col] = xi;
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
        tmp = lane_id < block_size/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row*ncols + col] = scale * c[col] * dst_add[row*ncols + col];
    }
}

static void fused_add_rms_norm_f32_cuda(const float * a, const float * b, const float * c, float * dst_add, float * dst,
        const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        fused_add_rms_norm_f32<256><<<nrows, block_dims, 0, stream>>>(a, b, c, dst_add, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        fused_add_rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(a, b, c, dst_add, dst, ncols, eps);
    }
}

void ggml_cuda_op_fused_add_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * add, ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    //const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(add->data == src0->data);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(add->src[0]));
    GGML_ASSERT(ggml_is_contiguous(add->src[1]));
    GGML_ASSERT(ggml_are_same_shape(add->src[0], add->src[1]));
    GGML_ASSERT(ggml_are_same_shape(add->src[0], src0));
    GGML_ASSERT(add->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(add->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(ggml_nrows(src1) == 1);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];

    const int64_t nrows = ggml_nrows(src0);
    fused_add_rms_norm_f32_cuda((const float *)add->src[0]->data, (const float *)add->src[1]->data,
            src1_d, (float *)add->data, dst_d, ne00, nrows, eps, stream);
}

static void fused_add_add_rms_norm_f32_cuda(const float * a1, const float * a2, const float * b, const float * c, float * dst_add, float * dst,
        const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        fused_add_add_rms_norm_f32<256><<<nrows, block_dims, 0, stream>>>(a1, a2, b, c, dst_add, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        fused_add_add_rms_norm_f32<1024><<<nrows, block_dims, 0, stream>>>(a1, a2, b, c, dst_add, dst, ncols, eps);
    }
}

void ggml_cuda_op_fused_add_add_rms_norm(ggml_backend_cuda_context & ctx,
        ggml_tensor * add1, ggml_tensor * add2, ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    //const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(add1->data == add2->src[0]->data);
    GGML_ASSERT(add2->data == src0->data);
    GGML_ASSERT(ggml_is_contiguous(src0));
    //GGML_ASSERT(ggml_is_contiguous(add->src[0]));
    //GGML_ASSERT(ggml_is_contiguous(add->src[1]));
    //GGML_ASSERT(ggml_are_same_shape(add->src[0], add->src[1]));
    //GGML_ASSERT(ggml_are_same_shape(add->src[0], src0));
    //GGML_ASSERT(add->src[0]->type == GGML_TYPE_F32);
    //GGML_ASSERT(add->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);
    GGML_ASSERT(ggml_nrows(src1) == 1);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t ne00 = src0->ne[0];

    const int64_t nrows = ggml_nrows(src0);
    fused_add_add_rms_norm_f32_cuda((const float *)add1->src[0]->data, (const float *)add1->src[1]->data, (const float *)add2->src[1]->data,
            src1_d, (float *)add2->data, dst_d, ne00, nrows, eps, stream);
}
