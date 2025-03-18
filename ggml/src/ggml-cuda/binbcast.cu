#include "binbcast.cuh"

static __device__ __forceinline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

static __device__ __forceinline__ float op_add(const float a, const float b) {
    return a + b;
}

static __device__ __forceinline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __device__ __forceinline__ float op_div(const float a, const float b) {
    return a / b;
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_bin_bcast(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        /*int s00,*/ int s01, int s02, int s03,
        /*int s10,*/ int s11, int s12, int s13) {
    const int i0s = blockDim.x*blockIdx.x + threadIdx.x;
    const int i1 = (blockDim.y*blockIdx.y + threadIdx.y);
    const int i2 = (blockDim.z*blockIdx.z + threadIdx.z) / ne3;
    const int i3 = (blockDim.z*blockIdx.z + threadIdx.z) % ne3;

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    for (int i0 = i0s; i0 < ne0; i0 += blockDim.x*gridDim.x) {
        const int i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
    }
}

template<float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t>
static __global__ void k_bin_bcast_unravel(const src0_t * src0, const src1_t * src1, dst_t * dst,
        int ne0, int ne1, int ne2, int ne3,
        int ne10, int ne11, int ne12, int ne13,
        /*int s0, */ int s1,  int s2,  int s3,
        /*int s00,*/ int s01, int s02, int s03,
        /*int s10,*/ int s11, int s12, int s13) {

    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    const int i3 = i/(ne2*ne1*ne0);
    const int i2 = (i/(ne1*ne0)) % ne2;
    const int i1 = (i/ne0) % ne1;
    const int i0 = i % ne0;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int i11 = i1 % ne11;
    const int i12 = i2 % ne12;
    const int i13 = i3 % ne13;

    const size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    const size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    const size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    const src0_t * src0_row = src0 + i_src0;
    const src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    const int i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (float)src0_row[i0] : 0.0f, (float)src1_row[i10]);
}

template<float (*bin_op)(const float, const float)>
struct bin_bcast_cuda {
    template<typename src0_t, typename src1_t, typename dst_t>
    void operator()(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst,
            const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
            cudaStream_t stream) {

        GGML_TENSOR_BINARY_OP_LOCALS

        int nr0 = ne10/ne0;
        int nr1 = ne11/ne1;
        int nr2 = ne12/ne2;
        int nr3 = ne13/ne3;

        int nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};

        size_t cnb[] = {nb0, nb1, nb2, nb3};
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};

        auto collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        auto collapse_nb = [](size_t cnb[], const int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
            for (int i = 0; i < 4; i++) {
                if (nr[i] != 1) {
                    break;
                }
                if (i > 0) {
                    collapse_nb(cnb, cne);
                    collapse_nb(cnb0, cne0);
                    collapse_nb(cnb1, cne1);
                    collapse(cne);
                    collapse(cne0);
                    collapse(cne1);
                }
            }
        }

        {
            int64_t ne0 = cne[0];
            int64_t ne1 = cne[1];
            int64_t ne2 = cne[2];
            int64_t ne3 = cne[3];

            //int64_t ne00 = cne0[0]; GGML_UNUSED(ne00);
            //int64_t ne01 = cne0[1]; GGML_UNUSED(ne01);
            //int64_t ne02 = cne0[2]; GGML_UNUSED(ne02);
            //int64_t ne03 = cne0[3]; GGML_UNUSED(ne03);

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb[0];
            size_t nb1 = cnb[1];
            size_t nb2 = cnb[2];
            size_t nb3 = cnb[3];

            size_t nb00 = cnb0[0];
            size_t nb01 = cnb0[1];
            size_t nb02 = cnb0[2];
            size_t nb03 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / sizeof(dst_t);
            size_t s2 = nb2 / sizeof(dst_t);
            size_t s3 = nb3 / sizeof(dst_t);

            size_t s10 = nb10 / sizeof(src1_t);
            size_t s11 = nb11 / sizeof(src1_t);
            size_t s12 = nb12 / sizeof(src1_t);
            size_t s13 = nb13 / sizeof(src1_t);

            size_t s00 = nb00 / sizeof(src0_t);
            size_t s01 = nb01 / sizeof(src0_t);
            size_t s02 = nb02 / sizeof(src0_t);
            size_t s03 = nb03 / sizeof(src0_t);

            GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
            GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

            GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
            GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

            GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
            GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

            GGML_ASSERT(s0 == 1);
            GGML_ASSERT(s00 == 1);
            GGML_ASSERT(s10 == 1);

            const int block_size = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            dim3 block_dims;
            block_dims.x = std::min<unsigned int>(hne0, block_size);
            block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
            block_dims.z = std::min(std::min<unsigned int>(ne2*ne3, block_size / block_dims.x / block_dims.y), 64U);

            dim3 block_nums(
                (hne0 + block_dims.x - 1) / block_dims.x,
                (ne1 + block_dims.y - 1) / block_dims.y,
                (ne2*ne3 + block_dims.z - 1) / block_dims.z
            );

            if (block_nums.z > 65535) {
                // this is the maximum number of blocks in z dimension, fallback to 1D grid kernel
                int block_num = (ne0*ne1*ne2*ne3 + block_size - 1) / block_size;
                k_bin_bcast_unravel<bin_op><<<block_num, block_size, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd,
                    ne0, ne1, ne2, ne3,
                    ne10, ne11, ne12, ne13,
                    /* s0, */ s1, s2, s3,
                    /* s00, */ s01, s02, s03,
                    /* s10, */ s11, s12, s13);
            } else {
                k_bin_bcast<bin_op><<<block_nums, block_dims, 0, stream>>>(
                    src0_dd, src1_dd, dst_dd,
                    ne0, ne1, ne2, ne3,
                    ne10, ne11, ne12, ne13,
                    /* s0, */ s1, s2, s3,
                    /* s00, */ s01, s02, s03,
                    /* s10, */ s11, s12, s13);
            }
        }
    }
};

template<class op>
static void ggml_cuda_op_bin_bcast(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const void * src0_dd, const void * src1_dd, void * dst_dd, cudaStream_t stream) {

    //GGML_ASSERT(src1->type == GGML_TYPE_F32);

    if (src1->type == GGML_TYPE_F32) {
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            op()(src0, src1, dst, (const float *)src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (half *) dst_dd, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
        } else {
            fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
                    ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
            GGML_ABORT("fatal error");
        }
    }
    else if (src1->type == GGML_TYPE_F16) {
        if (src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            op()(src0, src1, dst, (const float *)src0_dd, (const half *)src1_dd, (float *)dst_dd, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            op()(src0, src1, dst, (const half *) src0_dd, (const half *)src1_dd, (half *) dst_dd, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
            op()(src0, src1, dst, (const half *) src0_dd, (const half *)src1_dd, (float *)dst_dd, stream);
        } else {
            fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
                    ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
            GGML_ABORT("fatal error");
        }
    }
    else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == dst->src[0]->type);
    if (dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16) {
        ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat>>(dst, dst->src[0], dst, nullptr, dst->src[0]->data, dst->data, ctx.stream());
        return;
    }
    auto src = dst->src[0];
    auto bs = ggml_blck_size(src->type);
    auto ts = ggml_type_size(src->type);
    if (src->nb[0] != ts || ts*(src->ne[0]/bs) % 2 != 0) {
        fprintf(stderr, "%s: unsupported case type = %s, nb[0] = %zu, type_size = %zu\n", __func__, ggml_type_name(src->type), src->nb[0], ts);
        GGML_ABORT("fatal error");
    }
    auto aux_src = *src;
    aux_src.type = GGML_TYPE_F16;
    aux_src.ne[0] = ts*(src->ne[0]/bs)/2;
    aux_src.nb[0] = 2;
    auto aux_dst = *dst;
    aux_dst.type = GGML_TYPE_F16;
    aux_dst.ne[0] = ts*(dst->ne[0]/bs)/2;
    aux_dst.nb[0] = 2;
    aux_dst.src[0] = &aux_src;
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat>>(&aux_dst, &aux_src, &aux_dst, nullptr, dst->src[0]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_add>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

static __global__ void scale_f32_l(const float * x, float * dst, const void * data, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float * scale = (const float *)data;
    dst[i] = scale[0] * x[i];
}

static void scale_f32_cuda_l(const float * x, float * dst, const void * data, const int k, cudaStream_t stream) {
    constexpr int CUDA_SCALE_BLOCK_SIZE = 512; //256;
    const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f32_l<<<num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, data, k);
}

static void ggml_cuda_op_scale_tensor(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    float scale;
    memcpy(&scale, dst->src[1]->data, sizeof(float));

    scale_f32_cuda_l(src0_d, dst_d, dst->src[1]->data, ggml_nelements(src0), stream);
}

void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_nelements(dst->src[1]) == 1 && dst->src[1]->type == GGML_TYPE_F32 && dst->src[0]->type == GGML_TYPE_F32) {
        ggml_cuda_op_scale_tensor(ctx, dst);
        return;
    }
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_mul>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_div>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}
