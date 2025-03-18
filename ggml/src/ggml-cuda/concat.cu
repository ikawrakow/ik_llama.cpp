#include "concat.cuh"

// contiguous kernels
static __global__ void concat_f32_dim0(const float * x, const float * y, float * dst, const int64_t ne0, const int64_t ne00) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

// contiguous kernels
static __global__ void concat_f32_dim0(const float * x, const float * y, float * dst, const int64_t ne0, const int64_t ne00,
        int64_t nb02, int64_t nb12, int64_t nb2) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * nb2;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * nb02;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * nb12;
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim1(const float * x, const float * y, float * dst, const int64_t ne0, const int64_t ne01) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.y < ne01) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            (blockIdx.y - ne01) * ne0 +
            blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim2(const float * x, const float * y, float * dst, const int64_t ne0, const int64_t ne02) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.z < ne02) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 *  gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f32_cuda(const float * x, const float * y, float * dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    if (dim == 0 && ne1 >= 65536) {
        int64_t nstep = (ne1 + 32767)/32768;
        for (int64_t istep = 0; istep < nstep; ++istep) {
            int64_t i1 = 32768*istep;
            int64_t n1 = i1 + 32768 <= ne1 ? 32768 : ne1 - i1;
            dim3 gridDim(num_blocks, n1, ne2);
            const float * xi = x + i1*ne00;
            const float * yi = y + i1*(ne0 - ne00);
            float * dst_i = dst + i1*ne0;
            concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(xi, yi, dst_i, ne0, ne00, ne00*ne01, (ne0-ne00)*ne01, ne0*ne1);
        }
        return;
    }
    dim3 gridDim(num_blocks, ne1, ne2);
    if (dim == 0) {
        concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00);
        //concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00, ne00*ne01, (ne0-ne00)*ne01, ne0*ne1);
        return;
    }
    if (dim == 1) {
        concat_f32_dim1<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne01);
        return;
    }
    concat_f32_dim2<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne02);
}

// non-contiguous kernel (slow)
static __global__ void concat_f32_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3,
          int32_t   dim) {
    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

    const float * x;

    for (int i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            x = (const float *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
        }

        float * y = (float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type && src0->type == dst->type);

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
        (dim == 3 || (dim == 2 && dst->ne[3] == 1) || (dim == 1 && dst->ne[2]*dst->ne[3] == 1))) {
        const size_t size0 = ggml_nbytes(src0);
        const size_t size1 = ggml_nbytes(src1);
        CUDA_CHECK(cudaMemcpyAsync((char *)dst->data,         src0->data, size0, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *)dst->data + size0, src1->data, size1, cudaMemcpyDeviceToDevice, stream));
        return;
    }

    if (dim == 0 && src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == ggml_type_size(src1->type) &&
            src0->nb[1] % sizeof(float) == 0 && src1->nb[1] % sizeof(float) == 0) {
        auto bs = ggml_blck_size(dst->type);
        auto ts = ggml_type_size(dst->type);
        auto ne00_eff = (src0->ne[0]/bs)*ts/sizeof(float);
        auto ne0_eff  = (dst->ne[0]/bs)*ts/sizeof(float);
        if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
            //if (dst->ne[1] >= 65536 || dst->ne[2] >= 65536) {
            //    fprintf(stderr, "%s: ne1 = %ld, ne2 = %ld exceed max. blocks when computing %s\n", __func__, dst->ne[1], dst->ne[2], dst->name);
            //    GGML_ABORT("fatal error");
            //}
            const float * src0_d = (const float *)src0->data;
            const float * src1_d = (const float *)src1->data;
            float * dst_d = (float *)dst->data;
            //printf("%s(%s, %s): %ld %zu %zu  %ld %zu %zu\n", __func__, src0->name, src1->name, src0->ne[0], src0->nb[0], src0->nb[1], dst->ne[0], dst->nb[0], dst->nb[1]);
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_f32_cuda(
                        src0_d + i3 * (src0->nb[3] / 4),
                        src1_d + i3 * (src1->nb[3] / 4),
                        dst_d + i3 * ( dst->nb[3] / 4),
                        ne00_eff, src0->ne[1], src0->ne[2],
                        ne0_eff, dst->ne[1], dst->ne[2], dim, stream);
                        //src0->nb[1]/sizeof(float), src0->ne[1], src0->ne[2],
                        //dst->nb[1]/sizeof(float), dst->ne[1], dst->ne[2], dim, stream);
                        //src0->ne[0]*src0->nb[0]/sizeof(float), src0->ne[1], src0->ne[2],
                        //dst->ne[0]*dst->nb[0]/sizeof(float),  dst->ne[1],  dst->ne[2], dim, stream);
            }
        } else {
            //printf("%s(not contiguous): %s(%s) and %s(%s)\n", __func__, src0->name, ggml_type_name(src0->type), src1->name, ggml_type_name(src1->type));
            auto ne10_eff = (src1->ne[0]/bs)*ts/sizeof(float);
            dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
            concat_f32_non_cont<<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                    (const char *)src0->data,
                    (const char *)src1->data,
                    (      char *)dst->data,
                    ne00_eff, src0->ne[1], src0->ne[2], src0->ne[3],
                    //src0->ne[0]*src0->nb[0]/sizeof(float), src0->ne[1], src0->ne[2], src0->ne[3],
                    sizeof(float), src0->nb[1], src0->nb[2], src0->nb[3],
                    ne10_eff, src1->ne[1], src1->ne[2], src1->ne[3],
                    //src1->ne[0]*src1->nb[0]/sizeof(float), src1->ne[1], src1->ne[2], src1->ne[3],
                    sizeof(float), src1->nb[1], src1->nb[2], src1->nb[3],
                    ne0_eff,  dst->ne[1],  dst->ne[2],  dst->ne[3],
                    //dst->ne[0]*dst->nb[0]/sizeof(float),  dst->ne[1],  dst->ne[2],  dst->ne[3],
                    sizeof(float),  dst->nb[1],  dst->nb[2],  dst->nb[3], dim);
        }
        return;
    }

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        //if (dst->ne[1] >= 65536 || dst->ne[2] >= 65536) {
        //    fprintf(stderr, "%s: ne1 = %ld, ne2 = %ld exceed max. blocks when computing %s\n", __func__, dst->ne[1], dst->ne[2], dst->name);
        //    GGML_ABORT("fatal error");
        //}
        const float * src0_d = (const float *)src0->data;
        const float * src1_d = (const float *)src1->data;

        float * dst_d = (float *)dst->data;

        for (int i3 = 0; i3 < dst->ne[3]; i3++) {
            concat_f32_cuda(
                    src0_d + i3 * (src0->nb[3] / 4),
                    src1_d + i3 * (src1->nb[3] / 4),
                    dst_d + i3 * ( dst->nb[3] / 4),
                    src0->ne[0], src0->ne[1], src0->ne[2],
                    dst->ne[0],  dst->ne[1],  dst->ne[2], dim, stream);
        }
    } else {
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        concat_f32_non_cont<<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *)src0->data,
                (const char *)src1->data,
                (      char *)dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0],  dst->ne[1],  dst->ne[2],  dst->ne[3],
                dst->nb[0],  dst->nb[1],  dst->nb[2],  dst->nb[3], dim);
    }
}
