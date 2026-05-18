//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "getrows.cuh"
#include "dequantize.cuh"

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
            const void * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01, /*int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12/*, size_t s13*/) {

    const int i00 = (blockIdx.x*blockDim.x + threadIdx.x)*2;
    const int i10 = blockDim.y*blockIdx.y + threadIdx.y;
    const int i11 = (blockIdx.z*blockDim.z + threadIdx.z)/ne12;
    const int i12 = (blockIdx.z*blockDim.z + threadIdx.z)%ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const void * src0_row = (const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    const int ib = i00/qk; // block index
    const int iqs = (i00%qk)/qr; // quant index
    const int iybs = i00 - i00%qk; // dst block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    if (i01 >= 0 && i01 < ne01) {
        dequantize_kernel(src0_row, ib, iqs, v);
    } else {
        v.x = v.y = 0;
    }

    dst_row[iybs + iqs + 0]        = v.x;
    dst_row[iybs + iqs + y_offset] = v.y;
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
            const src0_t * src0, const int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne01, /*int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12/*, size_t s13*/) {

    const int i00 = blockIdx.x*blockDim.x + threadIdx.x;
    const int i10 = blockDim.y*blockIdx.y + threadIdx.y;
    const int i11 = (blockIdx.z*blockDim.z + threadIdx.z)/ne12;
    const int i12 = (blockIdx.z*blockDim.z + threadIdx.z)%ne12;

    if (i00 >= ne00) {
        return;
    }

    const int i01 = src1[i10*s10 + i11*s11 + i12*s12];
    dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
    const src0_t * src0_row = (const src0_t *)((const char *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_row[i00] = i01 >= 0 && i01 < ne01 ? dst_t(src0_row[i00]) : dst_t(0);
}

template<int qk, int qr, dequantize_kernel_t dq>
static void get_rows_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                            const void * src0_dd, const int32_t * src1_dd, float * dst_dd, cudaStream_t stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(block_num_x, ne10, ne11*ne12);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
            src0_dd, src1_dd, dst_dd,
            ne00, ne01, /*ne02, ne03,*/
            /*ne10, ne11,*/ ne12, /*ne13,*/
            /* s0,*/ s1, s2, s3,
            /* nb00,*/ nb01, nb02, nb03,
            s10, s11, s12/*, s13*/);

    GGML_UNUSED(dst);
}

template<typename src0_t>
static void get_rows_cuda_float(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                const src0_t * src0_dd, const int32_t * src1_dd, float * dst_dd, cudaStream_t stream) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, ne10, ne11*ne12);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    const size_t s1 = nb1 / ggml_element_size(dst);
    const size_t s2 = nb2 / ggml_element_size(dst);
    const size_t s3 = nb3 / ggml_element_size(dst);

    const size_t s10 = nb10 / ggml_element_size(src1);
    const size_t s11 = nb11 / ggml_element_size(src1);
    const size_t s12 = nb12 / ggml_element_size(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    k_get_rows_float<<<block_nums, block_dims, 0, stream>>>(
            src0_dd, src1_dd, dst_dd,
            ne00, ne01, /*ne02, ne03,*/
            /*ne10, ne11,*/ ne12, /*ne13,*/
            /* s0,*/ s1, s2, s3,
            /* nb00,*/ nb01, nb02, nb03,
            s10, s11, s12/*, s13*/);

    GGML_UNUSED(dst);
}

// Helper for k-quant scale extraction (matches convert.cu implementation)
static inline __device__ void get_scale_min_k4_gr(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Gather + dequantize Q4_K rows. 32 threads, blockIdx.x = block-in-row, blockIdx.y = token.
template<typename dst_t>
static __global__ void k_get_rows_q4_K(
        const void * src0, const int32_t * src1, dst_t * dst,
        int64_t nb01, int64_t s1) {
    const int64_t i_block  = blockIdx.x;
    const int64_t i_token  = blockIdx.y;
    const int32_t row_idx  = src1[i_token];
    const block_q4_K * x   = (const block_q4_K *)((const char *)src0 + row_idx * nb01) + i_block;
    dst_t * y               = dst + i_token * s1 + i_block * QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid / 8;
    const int64_t ir  = tid % 8;
    const int64_t is  = 2 * il;

    y += 64 * il + 4 * ir;

    const float dall = __low2half(x->dm);
    const float dmin = __high2half(x->dm);
    const uint8_t * q = x->qs + 32 * il + 4 * ir;

    uint8_t sc, m;
    get_scale_min_k4_gr(is + 0, x->scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    get_scale_min_k4_gr(is + 1, x->scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;
    for (int l = 0; l < 4; ++l) {
        y[l +  0] = d1 * (q[l] & 0xF) - m1;
        y[l + 32] = d2 * (q[l] >>  4) - m2;
    }
}

// Gather + dequantize Q5_K rows. 64 threads, blockIdx.x = block-in-row, blockIdx.y = token.
template<typename dst_t>
static __global__ void k_get_rows_q5_K(
        const void * src0, const int32_t * src1, dst_t * dst,
        int64_t nb01, int64_t s1) {
    const int64_t i_block  = blockIdx.x;
    const int64_t i_token  = blockIdx.y;
    const int32_t row_idx  = src1[i_token];
    const block_q5_K * x   = (const block_q5_K *)((const char *)src0 + row_idx * nb01) + i_block;
    dst_t * y               = dst + i_token * s1 + i_block * QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid / 16;
    const int64_t ir  = tid % 16;
    const int64_t is  = 2 * il;

    y += 64 * il + 2 * ir;

    const float dall = __low2half(x->dm);
    const float dmin = __high2half(x->dm);
    const uint8_t * ql = x->qs + 32 * il + 2 * ir;
    const uint8_t * qh = x->qh + 2 * ir;

    uint8_t sc, m;
    get_scale_min_k4_gr(is + 0, x->scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    get_scale_min_k4_gr(is + 1, x->scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;

    uint8_t hm = 1 << (2 * il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
}

// Gather + dequantize Q6_K rows. 64 threads, blockIdx.x = block-in-row, blockIdx.y = token.
template<typename dst_t>
static __global__ void k_get_rows_q6_K(
        const void * src0, const int32_t * src1, dst_t * dst,
        int64_t nb01, int64_t s1) {
    const int64_t i_block  = blockIdx.x;
    const int64_t i_token  = blockIdx.y;
    const int32_t row_idx  = src1[i_token];
    const block_q6_K * x   = (const block_q6_K *)((const char *)src0 + row_idx * nb01) + i_block;
    dst_t * y               = dst + i_token * s1 + i_block * QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid / 32;
    const int64_t il  = tid % 32;
    const int64_t is  = 8 * ip + il / 16;

    y += 128 * ip + il;

    const float d = x->d;
    const uint8_t * ql = x->ql + 64 * ip + il;
    const uint8_t   qh = x->qh[32 * ip + il];
    const int8_t  * sc = x->scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

static void get_rows_q4_K_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src0_d, const int32_t * src1_d, float * dst_d, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    const int64_t s1_dst = nb1 / ggml_element_size(dst);
    const dim3 block_nums(ne00 / QK_K, ne10, ne11 * ne12);
    k_get_rows_q4_K<float><<<block_nums, 32, 0, stream>>>(src0_d, src1_d, dst_d, nb01, s1_dst);
    GGML_UNUSED(src1); GGML_UNUSED(dst);
}

static void get_rows_q5_K_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src0_d, const int32_t * src1_d, float * dst_d, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    const int64_t s1_dst = nb1 / ggml_element_size(dst);
    const dim3 block_nums(ne00 / QK_K, ne10, ne11 * ne12);
    k_get_rows_q5_K<float><<<block_nums, 64, 0, stream>>>(src0_d, src1_d, dst_d, nb01, s1_dst);
    GGML_UNUSED(src1); GGML_UNUSED(dst);
}

static void get_rows_q6_K_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const void * src0_d, const int32_t * src1_d, float * dst_d, cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    const int64_t s1_dst = nb1 / ggml_element_size(dst);
    const dim3 block_nums(ne00 / QK_K, ne10, ne11 * ne12);
    k_get_rows_q6_K<float><<<block_nums, 64, 0, stream>>>(src0_d, src1_d, dst_d, nb01, s1_dst);
    GGML_UNUSED(src1); GGML_UNUSED(dst);
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();


    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    const int32_t * src1_i32 = (const int32_t *) src1_d;

    switch (src0->type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float(src0, src1, dst, (const half *)src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda<QK4_0, QR4_0, dequantize_q4_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda<QK4_1, QR4_1, dequantize_q4_1>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda<QK5_0, QR5_0, dequantize_q5_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda<QK5_1, QR5_1, dequantize_q5_1>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda<QK8_0, QR8_0, dequantize_q8_0>(src0, src1, dst, src0_d, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q4_K:
            get_rows_q4_K_cuda(src0, src1, dst, src0->data, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q5_K:
            get_rows_q5_K_cuda(src0, src1, dst, src0->data, src1_i32, dst_d, stream);
            break;
        case GGML_TYPE_Q6_K:
            get_rows_q6_K_cuda(src0, src1, dst, src0->data, src1_i32, dst_d, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported type: %s\n", __func__, ggml_type_name(src0->type));
            break;
    }
}
