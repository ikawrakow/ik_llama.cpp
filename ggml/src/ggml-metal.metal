//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#define GGML_COMMON_DECL_METAL
#define GGML_COMMON_IMPL_METAL
#include "ggml-common.h"

#include <metal_stdlib>

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

enum ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

// general-purpose kernel for addition, multiplication and division of two tensors
// pros: works for non-contiguous tensors, supports broadcast across all dims
// cons: not very efficient
kernel void kernel_add(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        constant  int64_t & offs,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01 + offs;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1  + offs;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) + *((device float *)(src1_ptr + i10*nb10));
    }
}

kernel void kernel_mul(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) * *((device float *)(src1_ptr + i10*nb10));
    }
}

kernel void kernel_div(
        device const char * src0,
        device const char * src1,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig.z;
    const int64_t i02 = tgpig.y;
    const int64_t i01 = tgpig.x;

    const int64_t i13 = i03 % ne13;
    const int64_t i12 = i02 % ne12;
    const int64_t i11 = i01 % ne11;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    device const char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    device       char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i10 = i0 % ne10;
        *((device float *)(dst_ptr + i0*nb0)) = *((device float *)(src0_ptr + i0*nb00)) / *((device float *)(src1_ptr + i10*nb10));
    }
}

template<typename T>
kernel void kernel_repeat(
        device const char * src0,
        device       char * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3 % ne03;
    const int64_t i02 = i2 % ne02;
    const int64_t i01 = i1 % ne01;

    device const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    device       char * dst_ptr  = dst  +  i3*nb3  +  i2*nb2  +  i1*nb1 ;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int i00 = i0 % ne00;
        *((device T *)(dst_ptr + i0*nb0)) = *((device T *)(src0_ptr + i00*nb00));
    }
}

typedef decltype(kernel_repeat<float>) kernel_repeat_t;

template [[host_name("kernel_repeat_f32")]] kernel kernel_repeat_t kernel_repeat<float>;
template [[host_name("kernel_repeat_f16")]] kernel kernel_repeat_t kernel_repeat<half>;
template [[host_name("kernel_repeat_i32")]] kernel kernel_repeat_t kernel_repeat<int>;
template [[host_name("kernel_repeat_i16")]] kernel kernel_repeat_t kernel_repeat<short>;

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_add_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig % nb];
}
kernel void kernel_add_4(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] + src1[tpig];
}

kernel void kernel_mul_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb  [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig % nb];
}

kernel void kernel_mul_4(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src1[tpig];
}

kernel void kernel_div_row(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        constant   uint64_t & nb  [[buffer(28)]],
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] / src1[tpig % nb];
}
kernel void kernel_div_4(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] / src1[tpig];
}

kernel void kernel_scale(
        device const float * src0,
        device       float * dst,
        constant     float & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_scale_4(
        device const float4 * src0,
        device       float4 * dst,
        constant     float  & scale,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * scale;
}

kernel void kernel_softcap(
        device const float * src0,
        device       float * dst,
        constant     float & s_before,
        constant     float & s_after,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = s_after * precise::tanh(src0[tpig] * s_before);
}

kernel void kernel_softcap_4(
        device const float4 * src0,
        device       float4 * dst,
        constant     float  & s_before,
        constant     float  & s_after,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = s_after * precise::tanh(src0[tpig] * s_before);
}

kernel void kernel_clamp(
        device const float * src0,
        device       float * dst,
        constant     float & min,
        constant     float & max,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] < min ? min : (src0[tpig] > max ? max : src0[tpig]);
}

kernel void kernel_relu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]);
}

kernel void kernel_mul_relu(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = max(0.0f, src0[tpig]) * src1[tpig];
}

kernel void kernel_sigmoid(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = 1.0f / (1.0f + exp(-src0[tpig]));
}

kernel void kernel_tanh(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    dst[tpig] = precise::tanh(x);
}

constant float GELU_COEF_A     = 0.044715f;
constant float GELU_QUICK_COEF = -1.702f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

kernel void kernel_gelu(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];

    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_4(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    // BEWARE !!!
    // Simply using "tanh" instead of "precise::tanh" will sometimes results in NaNs!
    // This was observed with Falcon 7B and 40B models
    //
    dst[tpig] = 0.5f*x*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_mul_gelu(
    device const float * src0,
    device const float * src1,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];

    dst[tpig] = 0.5f*x*src1[tpig]*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_mul_gelu_4(
    device const float4 * src0,
    device const float4 * src1,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    // BEWARE !!!
    // Simply using "tanh" instead of "precise::tanh" will sometimes results in NaNs!
    // This was observed with Falcon 7B and 40B models
    //
    dst[tpig] = 0.5f*x*src1[tpig]*(1.0f + precise::tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_quick(
    device const float * src0,
    device       float * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];

    dst[tpig] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}

kernel void kernel_gelu_quick_4(
    device const float4 * src0,
    device       float4 * dst,
    uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];

    dst[tpig] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}

kernel void kernel_silu(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_silu_4(
        device const float4 * src0,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];
    dst[tpig] = x / (1.0f + exp(-x));
}

kernel void kernel_mul_silu(
        device const float * src0,
        device const float * src1,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float & x = src0[tpig];
    dst[tpig] = x * src1[tpig] / (1.0f + exp(-x));
}

kernel void kernel_mul_silu_4(
        device const float4 * src0,
        device const float4 * src1,
        device       float4 * dst,
        uint tpig[[thread_position_in_grid]]) {
    device const float4 & x = src0[tpig];
    dst[tpig] = x * src1[tpig] / (1.0f + exp(-x));
}

kernel void kernel_swiglu(
        device const float * src0,
        device       float * dst,
        constant      uint & ne0,
        constant      uint & stride,
        uint tpig[[thread_position_in_grid]]) {
    const uint row = tpig/ne0;
    const uint idx = tpig%ne0;
    const uint j   = row*stride + idx;
    dst[tpig] = src0[j] * src0[j + ne0] / (1.0f + exp(-src0[j]));
}

kernel void kernel_swiglu_4(
        device const float4 * src0,
        device       float4 * dst,
        constant       uint & ne0,
        constant       uint & stride,
        uint tpig[[thread_position_in_grid]]) {
    const uint row = tpig/ne0;
    const uint idx = tpig%ne0;
    const uint j   = row*stride + idx;
    dst[tpig] = src0[j] * src0[j + ne0] / (1.0f + exp(-src0[j]));
}

kernel void kernel_sqr(
        device const float * src0,
        device       float * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] * src0[tpig];
}

kernel void kernel_multi_add_4(
        device const float4 * src0,
        device       float4 * dst,
        constant int64_t    & ne0,
        constant int64_t    & ne1,
        constant int64_t    & nb1,
        constant int64_t    & nb01,
        constant int        & n_expert,
        uint tpig[[thread_position_in_grid]]) {

    int64_t i0 = tpig % (ne0/4);
    int64_t i1 = tpig / (ne0/4);
    device float4 * dst_ptr = dst + i1*(nb1/16) + i0;
    device const float4 * src_ptr = src0 + i1*(nb01/16) + i0;
    float4 sum = src_ptr[0] + src_ptr[ne0/4];
    for (int i = 2; i < n_expert; ++i) sum += src_ptr[i*ne0/4];
    dst_ptr[0] = sum;
}

kernel void kernel_multi_add(
        device const float  * src0,
        device       float  * dst,
        constant int64_t    & ne0,
        constant int64_t    & ne1,
        constant int64_t    & nb1,
        constant int64_t    & nb01,
        constant int        & n_expert,
        uint tpig[[thread_position_in_grid]]) {

    int64_t i0 = tpig % ne0;
    int64_t i1 = tpig / ne0;
    device float * dst_ptr = dst + i1*nb1/4 + i0;
    device const float * src_ptr = src0 + i1*nb01/4 + i0;
    float sum = src_ptr[0] + src_ptr[ne0];
    for (int i = 2; i < n_expert; ++i) sum += src_ptr[i*ne0];
    dst_ptr[0] = sum;
}

kernel void kernel_sum_rows(
        device const float * src0,
        device       float * dst,
        constant  int64_t & ne00,
        constant  int64_t & ne01,
        constant  int64_t & ne02,
        constant  int64_t & ne03,
        constant uint64_t & nb00,
        constant uint64_t & nb01,
        constant uint64_t & nb02,
        constant uint64_t & nb03,
        constant  int64_t & ne10,
        constant  int64_t & ne11,
        constant  int64_t & ne12,
        constant  int64_t & ne13,
        constant uint64_t & nb10,
        constant uint64_t & nb11,
        constant uint64_t & nb12,
        constant uint64_t & nb13,
        constant  int64_t & ne0,
        constant  int64_t & ne1,
        constant  int64_t & ne2,
        constant  int64_t & ne3,
        constant uint64_t & nb0,
        constant uint64_t & nb1,
        constant uint64_t & nb2,
        constant uint64_t & nb3,
        uint3 tpig[[thread_position_in_grid]]) {
    int64_t i3 = tpig.z;
    int64_t i2 = tpig.y;
    int64_t i1 = tpig.x;

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    device const float * src_row = (device const float *) ((device const char *) src0 + i1*nb01 + i2*nb02 + i3*nb03);
    device       float * dst_row = (device       float *) ((device       char *) dst  + i1*nb1  + i2*nb2  + i3*nb3);

    float row_sum = 0;

    for (int64_t i0 = 0; i0 < ne00; i0++) {
        row_sum += src_row[i0];
    }

    dst_row[0] = row_sum;
}

template<typename T>
kernel void kernel_soft_max(
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float * psrc0 = (device const float *) src0 + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    device const     T * pmask = src1 != src0 ? (device const    T *) src1         + i01*ne00 : nullptr;
    device       float * pdst  = (device       float *) dst  + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        const int64_t h = i02;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = -INFINITY;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        lmax = MAX(lmax, psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        const float exp_psrc0 = exp((psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_max_4(
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant  uint32_t & n_head_log2,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float4 * psrc4 = (device const float4 *) src0 + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00)/4;
    device const      T * pmask = src1 != src0 ? (device const     T *) src1         + i01*ne00/4 : nullptr;
    device       float4 * pdst4 = (device       float4 *) dst  + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00)/4;

    float slope = 1.0f;

    if (max_bias > 0.0f) {
        const int64_t h = i02;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = -INFINITY;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        pdst4[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_cap_max(
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant     float & s_before,
        constant     float & s_after,
        constant  uint32_t & n_head_log2,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float * psrc0 = (device const float *) src0 + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    device const     T * pmask = src1 != src0 ? (device const    T *) src1         + i01*ne00 : nullptr;
    device       float * pdst  = (device       float *) dst  + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        const int64_t h = i02;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = -INFINITY;

    const float tot_scale = scale * s_after;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        lmax = MAX(lmax, precise::tanh(s_before*psrc0[i00])*tot_scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        const float exp_psrc0 = exp((psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_cap_max_4(
        device const  char * src0,
        device const  char * src1,
        device        char * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant     float & scale,
        constant     float & max_bias,
        constant     float & m0,
        constant     float & m1,
        constant     float & s_before,
        constant     float & s_after,
        constant  uint32_t & n_head_log2,
        threadgroup  float * buf [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = (tgpig) / (ne02*ne01);
    const int64_t i02 = (tgpig - i03*ne02*ne01) / ne01;
    const int64_t i01 = (tgpig - i03*ne02*ne01 - i02*ne01);

    device const float4 * psrc4 = (device const float4 *) src0 + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00)/4;
    device const      T * pmask = src1 != src0 ? (device const     T *) src1         + i01*ne00/4 : nullptr;
    device       float4 * pdst4 = (device       float4 *) dst  + (i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00)/4;

    float slope = 1.0f;

    if (max_bias > 0.0f) {
        const int64_t h = i02;

        const float base = h < n_head_log2 ? m0 : m1;
        const int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    const float tot_scale = scale * s_after;

    // parallel max
    float4 lmax4 = -INFINITY;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        lmax4 = fmax(lmax4, precise::tanh(s_before*psrc4[i00])*tot_scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggerganov/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

template [[host_name("kernel_soft_max_f16")]]   kernel kernel_soft_max_t   kernel_soft_max<half>;
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f16_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<half4>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;

typedef decltype(kernel_soft_cap_max<float>)    kernel_soft_cap_max_t;
typedef decltype(kernel_soft_cap_max_4<float4>) kernel_soft_cap_max_4_t;

template [[host_name("kernel_soft_cap_max_f16")]]   kernel kernel_soft_cap_max_t   kernel_soft_cap_max<half>;
template [[host_name("kernel_soft_cap_max_f32")]]   kernel kernel_soft_cap_max_t   kernel_soft_cap_max<float>;
template [[host_name("kernel_soft_cap_max_f16_4")]] kernel kernel_soft_cap_max_4_t kernel_soft_cap_max_4<half4>;
template [[host_name("kernel_soft_cap_max_f32_4")]] kernel kernel_soft_cap_max_4_t kernel_soft_cap_max_4<float4>;

kernel void kernel_diag_mask_inf(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant       int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {
    const int64_t i02 = tpig[2];
    const int64_t i01 = tpig[1];
    const int64_t i00 = tpig[0];

    if (i00 > n_past + i01) {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;
    } else {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];
    }
}

kernel void kernel_diag_mask_inf_8(
        device const float4 * src0,
        device       float4 * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne01,
        constant        int & n_past,
        uint3 tpig[[thread_position_in_grid]]) {

    const int64_t i = 2*tpig[0];

    dst[i+0] = src0[i+0];
    dst[i+1] = src0[i+1];
    int64_t i4 = 4*i;
    const int64_t i02 = i4/(ne00*ne01); i4 -= i02*ne00*ne01;
    const int64_t i01 = i4/(ne00);      i4 -= i01*ne00;
    const int64_t i00 = i4;
    for (int k = 3; k >= 0; --k) {
        if (i00 + 4 + k <= n_past + i01) {
            break;
        }
        dst[i+1][k] = -INFINITY;
        if (i00 + k > n_past + i01) {
            dst[i][k] = -INFINITY;
        }
    }
}

kernel void kernel_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * sum [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float * x = (device const float *) ((device const char *) src0 + tgpig*nb01);
    // MEAN
    // parallel sum
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        sum[tpitg] += x[i00];
    }
    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean  = sum[0] / ne00;

    // recenter and VARIANCE
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float * y = dst + tgpig*ne00;
    sum[tpitg] = 0.0f;
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = x[i00] - mean;
        sum[tpitg] += y[i00] * y[i00];
    }

    // reduce
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = ntg/2; i > 0; i /= 2) {
        if (tpitg < i) {
            sum[tpitg] += sum[tpitg + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float variance = sum[0] / ne00;

    const float scale = 1.0f/sqrt(variance + eps);
    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {
        y[i00] = y[i00] * scale;
    }
}

kernel void kernel_rms_norm(
        device const  void * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float4 * x = (device const float4 *) ((device const char *) src0 + tgpig*nb01);

    float4 sumf = 0;
    float all_sum = 0;

    // parallel sum
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        sumf += x[i00] * x[i00];
    }
    all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    all_sum = simd_sum(all_sum);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = all_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        all_sum = buf[tiisg];
        all_sum = simd_sum(all_sum);
    }

    const float mean  = all_sum/ne00;
    const float scale = 1.0f/sqrt(mean + eps);

    device float4 * y = (device float4 *) (dst + tgpig*ne00);
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        y[i00] = x[i00] * scale;
    }
}

kernel void kernel_fused_rms_norm(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant     float & eps,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    device const float4 * x = (device const float4 *) ((device const char *) src0 + tgpig*nb01);

    float4 sumf = 0;
    float all_sum = 0;

    // parallel sum
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        sumf += x[i00] * x[i00];
    }
    all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];
    all_sum = simd_sum(all_sum);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = all_sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        all_sum = buf[tiisg];
        all_sum = simd_sum(all_sum);
    }

    const float mean  = all_sum/ne00;
    const float scale = 1.0f/sqrt(mean + eps);

    device float4 * y = (device float4 *) (dst + tgpig*ne00);
    device float4 * z = (device float4 *)src1;
    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {
        y[i00] = x[i00] * z[i00] * scale;
    }
}

kernel void kernel_group_norm(
        device const float * src0,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int32_t & n_groups,
        constant     float & eps,
        threadgroup float  * buf [[threadgroup(0)]],
        uint tgpig[[threadgroup_position_in_grid]],
        uint tpitg[[thread_position_in_threadgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint   ntg[[threads_per_threadgroup]]) {
    const int64_t ne = ne00*ne01*ne02;
    const int64_t gs = ne00*ne01*((ne02 + n_groups - 1) / n_groups);

    int start = tgpig * gs;
    int end   = start + gs;

    start += tpitg;

    if (end >= ne) {
        end = ne;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += ntg) {
        tmp += src0[j];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float mean = tmp / gs;
    tmp = 0.0f;

    for (int j = start; j < end; j += ntg) {
        float xi = src0[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = simd_sum(tmp);
    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = tmp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        tmp = buf[tiisg];
        tmp = simd_sum(tmp);
    }

    const float variance = tmp / gs;
    const float scale = 1.0f/sqrt(variance + eps);
    for (int j = start; j < end; j += ntg) {
        dst[j] *= scale;
    }
}

// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float2 acc = 0.f;

    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (sumy * -8.f + acc[0] + acc[1]);
}

// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float2 acc = 0.f;

    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 2 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)
                + yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)
                + yl[i + 9] * (qs[i / 2] & 0xF000);
    }
    return d * (acc[0] + acc[1]) + sumy * m;
}

// function for calculate inner product between half a q5_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float2 acc = 0.f;

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010))
                + yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[1] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100))
                + yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }
    return d * (sumy * -16.f + acc[0] + acc[1]);
}

// function for calculate inner product between half a q5_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_1/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float2 acc = 0.f;

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010))
                + yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[1] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100))
                + yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }    
    return d * (acc[0] + acc[1]) + sumy * m; 
}

// function for calculate inner product between half a q6_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q6 quants begin (0 or QK6_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q6_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float2 acc = 0.f;

    device const uint16_t * qh = (device const uint16_t *)qb_curr->qh;
    device const uint16_t * qs = (device const uint16_t *)qb_curr->qs + il/2;

    const int shift = 4*(il/8);
    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * ((qs[i/2] & 0x000F) | ((qh[i/2] << (4-shift)) & 0x0030))
                + yl[i + 1] * ((qs[i/2] & 0x0F00) | ((qh[i/2] << (4-shift)) & 0x3000));
        acc[1] += yl[i + 8] * ((qs[i/2] & 0x00F0) | ((qh[i/2] << (6-shift)) & 0x0300))
                + yl[i + 9] * ((qs[i/2] & 0xF000) | (((uint32_t)qh[i/2] << (6-shift)) & 0x30000));
    }
    return d * (sumy * -32.f + acc[0] + acc[1]);
}

// putting them in the kernel cause a significant performance penalty
#define N_DST 4        // each SIMD group works on 4 rows
#define N_SIMDGROUP 2  // number of SIMD groups in a thread group
//Note: This is a template, but strictly speaking it only applies to
//      quantizations where the block size is 32. It also does not
//      guard against the number of rows not being divisible by
//      N_DST, so this is another explicit assumption of the implementation.
template<typename block_q_type, int nr, int nsg, int nw>
void mul_vec_q_n_f32_impl(
        device const void  * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3 tgpig, uint tiisg, uint sgitg) {
    const int nb = ne00/QK4_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q_type * x = (device const block_q_type *) src0 + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16]; // src1 vector cache
    float sumf[nr] = {0.f};

    const int ix = (tiisg/2);
    const int il = (tiisg%2)*8;

    device const float * yb = y + ix * QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    for (int ib = ix; ib < nb; ib += nw/2) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i+1];
            yl[i+0] = yb[i+ 0];
            yl[i+1] = yb[i+ 1]/256.f;

            sumy += yb[i+16] + yb[i+17];
            yl[i+8] = yb[i+16]/16.f;
            yl[i+9] = yb[i+17]/4096.f;
        }

        for (int row = 0; row < nr; row++) {
            sumf[row] += block_q_n_dot_y(x+ib+row*nb, sumy, yl, il);
        }

        yb += QK4_0 * 16;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[im*ne0*ne1 + r1*ne0 + first_row + row] = tot;
        }
    }
}

kernel void kernel_mul_mv_q4_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q4_1_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {
     mul_vec_q_n_f32_impl<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q5_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q5_1_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

kernel void kernel_mul_mv_q6_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q6_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

#define NB_Q8_0 8

void kernel_mul_mv_q8_0_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {
    const int nr  = N_DST;
    const int nsg = N_SIMDGROUP;
    const int nw  = N_SIMDWIDTH;

    const int nb = ne00/QK8_0;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * nsg + sgitg) * nr;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q8_0 * x = (device const block_q8_0 *) src0 + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[NB_Q8_0];
    float sumf[nr]={0.f};

    const int ix = tiisg/4;
    const int il = tiisg%4;

    device const float * yb = y + ix * QK8_0 + NB_Q8_0*il;

    // each thread in a SIMD group deals with NB_Q8_0 quants at a time
    for (int ib = ix; ib < nb; ib += nw/4) {
        for (int i = 0; i < NB_Q8_0; ++i) {
            yl[i] = yb[i];
        }

        device const block_q8_0 * xr = x + ib;

        for (int row = 0; row < nr; row++) {
            //device const int8_t * qs = x[ib+row*nb].qs + NB_Q8_0*il;
            device const int8_t * qs = xr->qs + NB_Q8_0*il;
            float sumq = 0.f;
            for (int iq = 0; iq < NB_Q8_0; ++iq) {
                sumq += qs[iq] * yl[iq];
            }
            //sumf[row] += sumq*x[ib+row*nb].d;
            sumf[row] += sumq*xr->d;
            xr += nb;
        }

        yb += NB_Q8_0 * nw;
    }

    for (int row = 0; row < nr; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl(src0,src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,nullptr,tgpig,tiisg,sgitg);
}

#define N_MV_T_T 4

template<typename T0, typename T04, typename T1, typename T14>
void kernel_mul_mv_impl(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb00,
                  uint64_t   nb01,
                  uint64_t   nb02,
                   int64_t   ne10,
                   int64_t   ne11,
                   int64_t   ne12,
                  uint64_t   nb10,
                  uint64_t   nb11,
                  uint64_t   nb12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
                   uint3     tgpig,
                   uint      tiisg) {
    const int64_t r0 = tgpig.x;
    const int64_t rb = tgpig.y*N_MV_T_T;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb02*ne02;

    device const T0 * x = (device const T0 *) (src0 + offset0);

    if (ne00 < 128) {
        for (int row = 0; row < N_MV_T_T; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            device const T1 * y = (device const T1 *) (src1 + r1*nb11 + im*nb12);

            float sumf = 0;
            for (int i = tiisg; i < ne00; i += 32) {
                sumf += (T0) x[i] * (T1) y[i];
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        device const T04 * x4 = (device const T04 *) x;
        for (int row = 0; row < N_MV_T_T; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            device const T1  * y  = (device const T1  *) (src1 + r1*nb11 + im*nb12);
            device const T14 * y4 = (device const T14 *) y;

            float sumf = 0;
            for (int i = tiisg; i < ne00/4; i += 32) {
                for (int k = 0; k < 4; ++k) sumf += (float) (x4[i][k] * y4[i][k]);
            }

            float all_sum = simd_sum(sumf);
            if (tiisg == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) all_sum += (float) (x[i] * y[i]);
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}

template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_impl<T0, T04, T1, T14>(
        src0,
        src1,
        dst,
        ne00,
        ne01,
        ne02,
        nb00,
        nb01,
        nb02,
        ne10,
        ne11,
        ne12,
        nb10,
        nb11,
        nb12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg);
}

template<typename T1>
void kernel_mul_mv_bf16_impl(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb00,
                  uint64_t   nb01,
                  uint64_t   nb02,
                   int64_t   ne10,
                   int64_t   ne11,
                   int64_t   ne12,
                  uint64_t   nb10,
                  uint64_t   nb11,
                  uint64_t   nb12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
                   uint3     tgpig,
                   uint      tiisg) {
    const int64_t r0 = tgpig.x;
    const int64_t rb = tgpig.y*N_MV_T_T;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb02*ne02;

    device const uint16_t * x = (device const uint16_t *) (src0 + offset0);

    typedef union { uint32_t u[4]; float f[4]; } aux_t;
    aux_t aux;

    for (int row = 0; row < N_MV_T_T; ++row) {
        int r1 = rb + row;
        if (r1 >= ne11) {
            break;
        }

        device const T1 * y = (device const T1 *) (src1 + r1*nb11 + im*nb12);

        float sumf = 0;
        for (int i = tiisg; i < ne00/4; i += 32) {
            aux.u[0] = x[4*i+0] << 16;
            aux.u[1] = x[4*i+1] << 16;
            aux.u[2] = x[4*i+2] << 16;
            aux.u[3] = x[4*i+3] << 16;
            sumf += aux.f[0] * (float)y[4*i+0] + aux.f[1] * (float)y[4*i+1] + aux.f[2] * (float)y[4*i+2] + aux.f[3] * (float)y[4*i+3];
        }

        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

template<typename T1>
kernel void kernel_mul_mv_bf16(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_bf16_impl<T1>(
        src0,
        src1,
        dst,
        ne00,
        ne01,
        ne02,
        nb00,
        nb01,
        nb02,
        ne10,
        ne11,
        ne12,
        nb10,
        nb11,
        nb12,
        ne0,
        ne1,
        r2,
        r3,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv<half, half4, half, half4>) mul_mv_t;

template [[host_name("kernel_mul_mv_f32_f32")]]   kernel mul_mv_t kernel_mul_mv<float,  float4,  float,  float4>;
template [[host_name("kernel_mul_mv_f16_f32")]]   kernel mul_mv_t kernel_mul_mv<half,   half4,   float,  float4>;
template [[host_name("kernel_mul_mv_f16_f16")]]   kernel mul_mv_t kernel_mul_mv<half,   half4,   half,   half4>;
template [[host_name("kernel_mul_mv_bf16_f16")]]  kernel mul_mv_t kernel_mul_mv_bf16<half>;
template [[host_name("kernel_mul_mv_bf16_f32")]]  kernel mul_mv_t kernel_mul_mv_bf16<float>;

template<typename T, typename T4>
kernel void kernel_mul_mv_1row(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]]) {

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb02*ne02;

    device const T     * x = (device const T     *) (src0 + offset0);
    device const float * y = (device const float *) (src1 + r1*nb11 + im*nb12);

    float sumf = 0;
    if (ne00 < 128) {
        for (int i = tiisg; i < ne00; i += 32) {
            sumf += (float) x[i] * (float) y[i];
        }
        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    } else {
        device const T4     * x4 = (device const T4     *) x;
        device const float4 * y4 = (device const float4 *) y;

        for (int i = tiisg; i < ne00/4; i += 32) {
            for (int k = 0; k < 4; ++k) sumf += (float) (x4[i][k] * y4[i][k]);
        }

        float all_sum = simd_sum(sumf);

        if (tiisg == 0) {
            for (int i = 4*(ne00/4); i < ne00; ++i) all_sum += (float) (x[i] * y[i]);
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

typedef decltype(kernel_mul_mv_1row<half, half4>) mul_mv_1row_t;

template [[host_name("kernel_mul_mv_f16_f32_1row")]]  kernel mul_mv_1row_t kernel_mul_mv_1row<half,   half4>;

// Assumes row size (ne00) is a multiple of 4
template<typename T, typename T4>
kernel void kernel_mul_mv_l4(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]]) {

    const int nrows = ne11;
    const int64_t r0 = tgpig.x;
    const int64_t im = tgpig.z;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb02*ne02;

    device const T4 * x4 = (device const T4 *) (src0 + offset0);

    for (int r1 = 0; r1 < nrows; ++r1) {
        device const float4 * y4 = (device const float4 *) (src1 + r1*nb11 + im*nb12);

        float sumf = 0;
        for (int i = tiisg; i < ne00/4; i += 32) {
            for (int k = 0; k < 4; ++k) sumf += (float) (x4[i][k] * y4[i][k]);
        }

        float all_sum = simd_sum(sumf);
        if (tiisg == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

typedef decltype(kernel_mul_mv_l4<half, half4>) mul_mv_l4_t;

template [[host_name("kernel_mul_mv_f16_f32_l4")]]  kernel mul_mv_l4_t kernel_mul_mv_l4<half, half4>;

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static inline void rope_yarn(
    float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
    thread float * cos_theta, thread float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
    // start and end correction dims
    dims[0] = max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

template<typename T>
kernel void kernel_rope_norm(
        device const    void * src0,
        device const int32_t * src1,
        device const   float * src2,
        device         float * dst,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant     int64_t & ne03,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant    uint64_t & nb03,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant     int64_t & ne2,
        constant     int64_t & ne3,
        constant    uint64_t & nb0,
        constant    uint64_t & nb1,
        constant    uint64_t & nb2,
        constant    uint64_t & nb3,
        constant         int & n_past,
        constant         int & n_dims,
        constant         int & n_ctx_orig,
        constant       float & freq_base,
        constant       float & freq_scale,
        constant       float & ext_factor,
        constant       float & attn_factor,
        constant       float & beta_fast,
        constant       float & beta_slow,
        uint  tiitg[[thread_index_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]],
        uint3 tgpig[[threadgroup_position_in_grid]]) {
    const int64_t i3 = tgpig[2];
    const int64_t i2 = tgpig[1];
    const int64_t i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    device const int32_t * pos = src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/n_dims;

    float cos_theta;
    float sin_theta;

    for (int64_t i0 = 2*tiitg; i0 < ne0; i0 += 2*tptg.x) {
        if (i0 < n_dims) {
            const int64_t ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0*cos_theta - x1*sin_theta;
            dst_data[1] = x0*sin_theta + x1*cos_theta;
        } else {
            device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

template<typename T>
kernel void kernel_rope_neox(
        device const    void * src0,
        device const int32_t * src1,
        device const   float * src2,
        device         float * dst,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant     int64_t & ne03,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant    uint64_t & nb03,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant     int64_t & ne2,
        constant     int64_t & ne3,
        constant    uint64_t & nb0,
        constant    uint64_t & nb1,
        constant    uint64_t & nb2,
        constant    uint64_t & nb3,
        constant         int & n_past,
        constant         int & n_dims,
        constant         int & n_ctx_orig,
        constant       float & freq_base,
        constant       float & freq_scale,
        constant       float & ext_factor,
        constant       float & attn_factor,
        constant       float & beta_fast,
        constant       float & beta_slow,
        uint  tiitg[[thread_index_in_threadgroup]],
        uint3 tptg[[threads_per_threadgroup]],
        uint3 tgpig[[threadgroup_position_in_grid]]) {
    const int64_t i3 = tgpig[2];
    const int64_t i2 = tgpig[1];
    const int64_t i1 = tgpig[0];

    float corr_dims[2];
    rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    device const int32_t * pos = src1;

    const float theta_base = (float) pos[i2];
    const float inv_ndims = -1.f/n_dims;

    float theta = theta_base * pow(freq_base, 2*tiitg*inv_ndims);
    const float theta_multiplier = pow(freq_base, 2*tptg.x*inv_ndims);

    float cos_theta;
    float sin_theta;

    int64_t i0 = 2*tiitg;
    for ( ; i0 < n_dims; i0 += 2*tptg.x) {
        const int64_t ic = i0/2;

        const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

        rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

        device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
        device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

        const float x0 = src[0];
        const float x1 = src[n_dims/2];

        dst_data[0]        = x0*cos_theta - x1*sin_theta;
        dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;

        theta *= theta_multiplier;
    }
    for ( ; i0 < ne0; i0 += 2*tptg.x) {
        device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

        dst_data[0] = src[0];
        dst_data[1] = src[1];
    }

    // Original version
    //for (int64_t i0 = 2*tiitg; i0 < ne0; i0 += 2*tptg.x) {
    //    if (i0 < n_dims) {
    //        const int64_t ic = i0/2;

    //        // Who thought that having a pow() evaluation in a loop is a good idea?
    //        //const float theta = theta_base * pow(freq_base, inv_ndims*i0);

    //        const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

    //        rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

    //        device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
    //        device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

    //        const float x0 = src[0];
    //        const float x1 = src[n_dims/2];

    //        dst_data[0]        = x0*cos_theta - x1*sin_theta;
    //        dst_data[n_dims/2] = x0*sin_theta + x1*cos_theta;

    //        theta *= theta_multiplier;
    //    } else {
    //        device const T * const src = (device T *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
    //        device       T * dst_data  = (device T *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

    //        dst_data[0] = src[0];
    //        dst_data[1] = src[1];
    //    }
    //}
}

typedef decltype(kernel_rope_norm<float>) kernel_rope_norm_t;
typedef decltype(kernel_rope_neox<float>) kernel_rope_neox_t;

template [[host_name("kernel_rope_norm_f32")]] kernel kernel_rope_norm_t kernel_rope_norm<float>;
template [[host_name("kernel_rope_norm_f16")]] kernel kernel_rope_norm_t kernel_rope_norm<half>;

template [[host_name("kernel_rope_neox_f32")]] kernel kernel_rope_neox_t kernel_rope_neox<float>;
template [[host_name("kernel_rope_neox_f16")]] kernel kernel_rope_neox_t kernel_rope_neox<half>;

typedef void (im2col_t)(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]);

template <typename T>
kernel void kernel_im2col(
        device const float * x,
        device        char * dst,
        constant   int32_t & ofs0,
        constant   int32_t & ofs1,
        constant   int32_t & IW,
        constant   int32_t & IH,
        constant   int32_t & CHW,
        constant   int32_t & s0,
        constant   int32_t & s1,
        constant   int32_t & p0,
        constant   int32_t & p1,
        constant   int32_t & d0,
        constant   int32_t & d1,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t iiw = tgpig[2] * s0 + tpitg[2] * d0 - p0;
    const int32_t iih = tgpig[1] * s1 + tpitg[1] * d1 - p1;

    const int32_t offset_dst =
        (tpitg[0] * tgpg[1] * tgpg[2] + tgpig[1] * tgpg[2] + tgpig[2]) * CHW +
        (tgpig[0] * (ntg[1] * ntg[2]) + tpitg[1] * ntg[2] + tpitg[2]);

    device T * pdst = (device T *) (dst);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        pdst[offset_dst] = 0.0f;
    } else {
        const int32_t offset_src = tpitg[0] * ofs0 + tgpig[0] * ofs1;
        pdst[offset_dst] = x[offset_src + iih * IW + iiw];
    }
}

template [[host_name("kernel_im2col_f32")]] kernel im2col_t kernel_im2col<float>;
template [[host_name("kernel_im2col_f16")]] kernel im2col_t kernel_im2col<half>;

kernel void kernel_upscale_f32(
    device  const char * src0,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    constant     float & sf0,
    constant     float & sf1,
    constant     float & sf2,
    constant     float & sf3,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3/sf3;
    const int64_t i02 = i2/sf2;
    const int64_t i01 = i1/sf1;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        const int64_t i00 = i0/sf0;

        device const float * src0_ptr = (device const float *) (src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        device       float * dst_ptr  = (device       float *) (dst  +  i3*nb3  +  i2*nb2  +  i1*nb1  +  i0*nb0);

        dst_ptr[0] = src0_ptr[0];
    }
}

kernel void kernel_pad_f32(
    device  const char * src0,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*nb3  +  i2*nb2  +  i1*nb1);

    if (i1 < ne01 && i2 < ne02 && i3 < ne03) {
        for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
            if (i0 < ne00) {
                dst_ptr[i0] = src0_ptr[i0];
            } else {
                dst_ptr[i0] = 0.0f;
            }
        }

        return;
    }

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        dst_ptr[i0] = 0.0f;
    }
}

kernel void kernel_arange_f32(
    device        char * dst,
    constant   int64_t & ne0,
    constant   float   & start,
    constant   float   & step,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        dst_ptr[i0] = start + step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    device  const char * src0,
    device        char * dst,
    constant  uint64_t & nb1,
    constant  int      & dim,
    constant  int      & max_period,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst +  i*nb1);

    int half_ = dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (dim % 2 != 0 && tpitg.x == 0) {
        embed_data[dim] = 0.f;
    }
}

// bitonic sort implementation following the CUDA kernels as reference
typedef void (argsort_t)(
        device const float  * x,
        device     int32_t  * dst,
        constant   int64_t  & ncols,
        constant   int64_t  & ncols_pad,
        threadgroup int32_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]);

template<ggml_sort_order order>
kernel void kernel_argsort_f32_i32(
        device const float   * x,
        device       int32_t * dst,
        constant     int64_t & ncols,
        constant     int64_t & ncols_pad,
        threadgroup int32_t  * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    // bitonic sort
    int col = tpitg[0];
    int row = tgpig[1];

    if (col >= ncols_pad) return;

    device const float   * x_row   = x + row * ncols;
    threadgroup int32_t  * dst_row = shared_values;

    // initialize indices
    dst_row[col] = col;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == GGML_SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        SWAP(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

template [[host_name("kernel_argsort_f32_i32_asc")]]  kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_ASC>;
template [[host_name("kernel_argsort_f32_i32_desc")]] kernel argsort_t kernel_argsort_f32_i32<GGML_SORT_ORDER_DESC>;

kernel void kernel_leaky_relu_f32(
        device const float * src0,
        device       float * dst,
        constant     float & slope,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = src0[tpig] > 0.0f ? src0[tpig] : src0[tpig] * slope;
}

//==========================================================================================
// NOTE: this is not dequantizing - we are simply fitting the template
template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_f16_t4(device const half4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    if (is_same_v<type4x4, half4x4>) {
        const half d = xb->d;
        for (int i = 0; i < 16; i++) {
            reg[i/4][i%4] = (half)qs[i + 16*il] * d;
        }
    } else {
        const float d = xb->d;
        for (int i = 0; i < 16; i++) {
            reg[i/4][i%4] = qs[i + 16*il] * d;
        }
    }
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const int8_t * qs = ((device const int8_t *)xb->qs);
    const float d = xb->d;
    for (int i = 0; i < 4; i++) {
        reg[i] = (qs[4*(il%4) + i + 16*(il/4)] * d);
    }
}

typedef struct {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2; // assume K and V are same shape
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    uint64_t nb31;
    int32_t  ne1;
    int32_t  ne2;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    uint16_t n_head_log2;
    float    logit_softcap;
} ggml_metal_kargs_flash_attn_ext;

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    typename q_t,     // query types in shared memory
    typename q4_t,
    typename q8x8_t,
    typename k_t,     // key types in shared memory
    typename k4x4_t,
    typename k8x8_t,
    typename v_t,     // value types in shared memory
    typename v4x4_t,
    typename v8x8_t,
    typename qk_t,    // Q*K types
    typename qk8x8_t,
    typename s_t,     // soft-max types
    typename s8x8_t,
    typename o_t,     // attention accumulation types
    typename o4_t,
    typename o8x8_t,
    typename kd4x4_t, // key type in device memory
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread k4x4_t &),
    typename vd4x4_t, // key type in device memory
    short nl_v,
    void (*deq_v)(device const vd4x4_t *, short, thread v4x4_t &),
    short DK,        // K head size
    short DV,        // V head size
    short Q  = 8,    // queries per threadgroup
    short KV = 8,    // key/value processed per each simdgroup
    short C  = 32>   // cache items per threadgroup
kernel void kernel_flash_attn_ext(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3   ntg[[threads_per_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short nsg = ntg.y; // number of simdgroups

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*Q;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;
    constexpr short DV4  = DV/4;
    constexpr short DV8  = DV/8;
    constexpr short DV16 = DV/16;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short SH  = (2*C + Q); // shared memory per simdgroup (s_t == float) default: 72

    const short TS = nsg*SH;   // shared memory size per query in (s_t == float) = 288 for nsg = 4
    const short T  = DK + 2*TS; // shared memory size per query in (half)        = 1152 for nsg = 4 and DK = 576
                                // => Q*T is 9216 for nsg = 4 and DK = 576 => 18432 bytes => overflows the 16384 bytes predicted as shmem in ggml-metal.m

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 +              0*DK); // holds the query data
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 +              0*DK); // same as above but in q4_t
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 +              0*DK); // reuse query data for accumulation
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 +              0*DK); // same as above but in o4_t
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + 2*sgitg*SH + Q*DK); // scratch buffer for attention, mask and diagonal matrix

    threadgroup k_t    * sk    = (threadgroup k_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T); // scratch buffer to load K in shared memory
    threadgroup k4x4_t * sk4x4 = (threadgroup k4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T); // same as above but in k4x4_t

    threadgroup v_t    * sv    = (threadgroup v_t    *) (shmem_f16 + sgitg*(4*16*KV) + Q*T); // scratch buffer to load V in shared memory
    threadgroup v4x4_t * sv4x4 = (threadgroup v4x4_t *) (shmem_f16 + sgitg*(4*16*KV) + Q*T); // same as above but in v4x4_t

    // store the result for all queries in local memory in 8x8 matrices (the O matrix from the paper)
    o8x8_t lo[DV8]; // For DV = 512 we have DV8 = 64 => 4096 entries per thread. Do we even have so much 

    // load heads from Q to shared memory
    for (short j = sgitg; j < Q; j += nsg) {
        device const float4 * q4 = (device const float4 *) ((device const char *) q + ((iq1 + j)*args.nb01 + iq2*args.nb02 + iq3*args.nb03));

        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < args.ne01) {
                sq4[j*DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j*DK4 + i] = (q4_t) 0.0f;
            }
        }
    }

    // zero out lo
    for (short i = 0; i < DV8; ++i) {
        lo[i] = make_filled_simdgroup_matrix<o_t, 8>((o_t) 0.0f);
    }

    // zero out shared memory SH
    for (short j = 0; j < Q; ++j) {
        for (short i = tiisg; i < SH; i += NW) {
            ss[j*TS + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S[Q] = { [0 ... Q-1] = 0.0f };
        float M[Q] = { [0 ... Q-1] = -__FLT16_MAX__/2 };

        // thread indices inside the simdgroup
        // TODO: see if we can utilize quad-group functions for better performance
        //       https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (6.9.3)
        const short tx = tiisg%4;
        const short ty = tiisg/4;

        // broadcast kv
        //const short rk2 = args.ne02/args.ne12;
        //const short rk3 = args.ne03/args.ne13;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        const bool has_mask = mask != q;

        float slope = 1.0f;

        // ALiBi
        if (args.max_bias > 0.0f) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = 0; ic0 < args.ne11; ic0 += C*nsg) {
            const int ic = ic0 + C*sgitg;
            if (ic >= args.ne11) {
                break;
            }

            if (has_mask) {
                // used to detect blocks full of -INF
                float smax = -INFINITY;

                // load the mask in shared memory
                #pragma unroll(Q)
                for (short j = 0; j < Q; ++j) {
                    device const half * pm = (device const half *) ((device const char *) mask + (iq1 + j)*args.nb31);

                    const float m = pm[ic + tiisg];

                    ss[j*TS + C + tiisg] = m;
                    smax = max(smax, m);
                }

                smax = simd_max(smax);

                if (smax == -INFINITY) {
                    continue;
                }
            }

            // Q*K^T
            {
                for (short cc = 0; cc < C/8; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    // this is compile-time check, so it does not have runtime overhead
                    if (is_same<kd4x4_t, k4x4_t>::value) {
                        // we can read directly from global memory
                        device const k_t * pk = (device const k_t *) ((device const char *) k + ((ic + 8*cc)*args.nb11 + ikv2*args.nb12 + ikv3*args.nb13));

                        #pragma unroll(DK8)
                        for (short i = 0; i < DK8; ++i) {
                            k8x8_t mk;
                            simdgroup_load(mk, pk + i*8, args.nb11/sizeof(k_t), 0, true); // transpose // TODO: use ne10

                            q8x8_t mq;
                            simdgroup_load(mq, sq + i*8, DK);
                            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                        }
                    } else {
                        for (short ii = 0; ii < DK16; ii += 4) {
                            device const kd4x4_t * pk4x4 = (device const kd4x4_t *) ((device const char *) k + ((ic + 8*cc + ty)*args.nb11 + ikv2*args.nb12 + ikv3*args.nb13));

                            if (DK16%4 == 0) {
                                // the head is evenly divisible by 4*16 = 64, so no need for bound checks
                                {
                                    k4x4_t tmp;
                                    deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                    sk4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                #pragma unroll(4)
                                for (short k = 0; k < 4; ++k) {
                                    k8x8_t mk;
                                    q8x8_t mq;

                                    simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                    simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                    simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                    simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                    simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                    simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                                }
                            } else {
                                if (ii + tx < DK16) {
                                    k4x4_t tmp;
                                    deq_k(pk4x4 + (ii + tx)/nl_k, (ii + tx)%nl_k, tmp);
                                    sk4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < DK16; ++k) {
                                    k8x8_t mk;
                                    q8x8_t mq;

                                    simdgroup_load(mk, sk + 16*k + 0*8, 4*16, 0, true); // transpose
                                    simdgroup_load(mq, sq + (2*(ii + k) + 0)*8, DK);
                                    simdgroup_multiply_accumulate(mqk, mq, mk, mqk);

                                    simdgroup_load(mk, sk + 16*k + 1*8, 4*16, 0, true); // transpose
                                    simdgroup_load(mq, sq + (2*(ii + k) + 1)*8, DK);
                                    simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
                                }
                            }
                        }
                    }

                    // cast qk_t -> s_t
                    //s8x8_t mqks(1.0f);
                    //simdgroup_multiply(mqks, mqk, mqks);
                    //simdgroup_store(mqks, ss + 8*cc, TS, 0, false);

                    simdgroup_store(mqk, ss + 8*cc, TS, 0, false);
                }
            }

            // online softmax
            {
                for (ushort j = 0; j < Q; ++j) {
                    const float m = M[j];

                    // scale and apply the logitcap / mask
                    float s = ss[j*TS + tiisg]*args.scale;

                    if (args.logit_softcap != 0.0f) {
                        s = args.logit_softcap*precise::tanh(s);
                    }

                    // mqk = mqk + mask*slope
                    s += slope*ss[j*TS + C + tiisg];

                    M[j] = simd_max(max(M[j], s));

                    const float ms = exp(m - M[j]);
                    const float vs = exp(s - M[j]);

                    S[j] = S[j]*ms + simd_sum(vs);

                    // the P matrix from the paper (Q rows, C columns)
                    ss[j*TS + tiisg] = vs;

                    // create a QxQ diagonal matrix for rescaling the output
                    if (tiisg == j) {
                        ss[j*TS + 2*C + j] = ms;
                    }
                }
            }

            // O = diag(ms)*O
            {
                s8x8_t mm;
                simdgroup_load(mm, ss + 2*C, TS, 0, false);

                #pragma unroll(DV8)
                for (short i = 0; i < DV8; ++i) {
                    simdgroup_multiply(lo[i], mm, lo[i]);
                }
            }

            // O = O + (Q*K^T)*V
            {
                for (short cc = 0; cc < C/8; ++cc) {
                    s8x8_t ms;
                    simdgroup_load(ms, ss + 8*cc, TS, 0, false);

                    if (is_same<vd4x4_t, v4x4_t>::value) {
                        // we can read directly from global memory
                        device const v_t * pv = (device const v_t *) ((device const char *) v + ((ic + 8*cc)*args.nb21 + ikv2*args.nb22 + ikv3*args.nb23));

                        #pragma unroll(DV8)
                        for (short i = 0; i < DV8; ++i) {
                            v8x8_t mv;
                            simdgroup_load(mv, pv + i*8, args.nb21/sizeof(v_t), 0, false); // TODO: use ne20

                            simdgroup_multiply_accumulate(lo[i], ms, mv, lo[i]);
                        }
                    } else {
                        for (short ii = 0; ii < DV16; ii += 4) {
                            device const vd4x4_t * pv4x4 = (device const vd4x4_t *) ((device const char *) v + ((ic + 8*cc + ty)*args.nb21 + ikv2*args.nb22 + ikv3*args.nb23));

                            if (DV16%4 == 0) {
                                // no need for bound checks
                                {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                #pragma unroll(4)
                                for (short k = 0; k < 4; ++k) {
                                    v8x8_t mv;

                                    simdgroup_load(mv, sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_multiply_accumulate(lo[2*(ii + k) + 0], ms, mv, lo[2*(ii + k) + 0]);

                                    simdgroup_load(mv, sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_multiply_accumulate(lo[2*(ii + k) + 1], ms, mv, lo[2*(ii + k) + 1]);
                                }
                            } else {
                                if (ii + tx < DV16) {
                                    v4x4_t tmp;
                                    deq_v(pv4x4 + (ii + tx)/nl_v, (ii + tx)%nl_v, tmp);
                                    sv4x4[4*ty + tx] = tmp;
                                }

                                simdgroup_barrier(mem_flags::mem_threadgroup);

                                for (short k = 0; k < 4 && ii + k < DV16; ++k) {
                                    v8x8_t mv;

                                    simdgroup_load(mv, sv + 16*k + 0*8, 4*16, 0, false);
                                    simdgroup_multiply_accumulate(lo[2*(ii + k) + 0], ms, mv, lo[2*(ii + k) + 0]);

                                    simdgroup_load(mv, sv + 16*k + 1*8, 4*16, 0, false);
                                    simdgroup_multiply_accumulate(lo[2*(ii + k) + 1], ms, mv, lo[2*(ii + k) + 1]);
                                }
                            }
                        }
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        for (short j = 0; j < Q; ++j) {
            if (tiisg == 0) {
                ss[j*TS + 0] = S[j];
                ss[j*TS + 1] = M[j];
            }
        }
    }

    // reduce the warps sequentially
    for (ushort sg = 1; sg < nsg; ++sg) {
        float S = { 0.0f };
        float M = { -__FLT16_MAX__/2 };

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // each simdgroup stores its output to shared memory, reusing sq
        if (sgitg == sg) {
            for (short i = 0; i < DV8; ++i) {
                simdgroup_store(lo[i], so + i*8, DV, 0, false);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // the first simdgroup accumulates the results from the other simdgroups
        if (sgitg == 0) {
            for (short j = 0; j < Q; ++j) {
                const float S0 = ss[j*TS +         0];
                const float S1 = ss[j*TS + sg*SH + 0];

                const float M0 = ss[j*TS +         1];
                const float M1 = ss[j*TS + sg*SH + 1];

                M = max(M0, M1);

                const float ms0 = exp(M0 - M);
                const float ms1 = exp(M1 - M);

                S = S0*ms0 + S1*ms1;

                if (tiisg == 0) {
                    ss[j*TS + 0] = S;
                    ss[j*TS + 1] = M;

                    ss[j*TS + 2*C + j        ] = ms0;
                    ss[j*TS + 2*C + j + sg*SH] = ms1;
                }
            }

            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
            {
                s8x8_t ms0;
                s8x8_t ms1;

                simdgroup_load(ms0, ss + 2*C,         TS, 0, false);
                simdgroup_load(ms1, ss + 2*C + sg*SH, TS, 0, false);

                #pragma unroll(DV8)
                for (short i = 0; i < DV8; ++i) {
                    o8x8_t t;

                    simdgroup_load    (t, so + i*8, DV, 0, false);
                    simdgroup_multiply(t, ms1, t);

                    simdgroup_multiply_accumulate(lo[i], ms0, lo[i], t);
                }
            }
        }
    }

    // store result to shared memory (reuse sq)
    if (sgitg == 0) {
        for (short i = 0; i < DV8; ++i) {
            simdgroup_store(lo[i], so + i*8, DV, 0, false);
        }
    }

    device float4 * dst4 = (device float4 *) dst;

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        for (short j = 0; j < Q && iq1 + j < args.ne01; ++j) {
            const float S = ss[j*TS + 0];

            for (short i = tiisg; i < DV4; i += NW) {
                dst4[((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)(iq1 + j)*args.ne1)*DV4 + i] = (float4) so4[j*DV4 + i]/S;
            }
        }
    }
}

// TODO: this is quite ugly. in the future these types will be hardcoded in the kernel, but for now keep them as
//       template to be able to explore different combinations
//
#define FA_TYPES \
    half,  half4,   simdgroup_half8x8,  \
    half,  half4x4, simdgroup_half8x8,  \
    half,  half4x4, simdgroup_half8x8,  \
    float,          simdgroup_float8x8, \
    float,          simdgroup_float8x8, \
    half,  half4,   simdgroup_half8x8

typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;

template [[host_name("kernel_flash_attn_ext_f16_h64" )]]         kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  64,  64>;
template [[host_name("kernel_flash_attn_ext_f16_h80" )]]         kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  80,  80>;
template [[host_name("kernel_flash_attn_ext_f16_h96" )]]         kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  96,  96>;
template [[host_name("kernel_flash_attn_ext_f16_h112")]]         kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  112, 112>;
template [[host_name("kernel_flash_attn_ext_f16_h128")]]         kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  128, 128>;
template [[host_name("kernel_flash_attn_ext_f16_h256")]]         kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  256, 256>;
template [[host_name("kernel_flash_attn_ext_f16_hk192_hv128")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  192, 128>;
template [[host_name("kernel_flash_attn_ext_f16_hk576_hv512")]]  kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, half4x4,    1, dequantize_f16,  half4x4,    1, dequantize_f16,  576, 512>;

template [[host_name("kernel_flash_attn_ext_q8_0_h64" )]]        kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q8_0_h80" )]]        kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q8_0_h96" )]]        kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q8_0_h112")]]        kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q8_0_h128")]]        kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q8_0_h192")]]        kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q8_0_h256")]]        kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q8_0_hk192_hv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q8_0_hk576_hv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_q8_0, 2, dequantize_q8_0, block_q8_0, 2, dequantize_q8_0, 576, 512>;

#undef FA_TYPES

template<
    typename q4_t,  // query types in shared memory
    typename k4_t,  // key types in shared memory
    typename v4_t,  // value types in shared memory
    typename qk_t,  // Q*K types
    typename s_t,   // soft-max types
    typename s4_t,
    typename o4_t,  // attention accumulation types
    typename kd4_t, // key type in device memory
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t, // key type in device memory
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,       // K head size
    short DV,       // V head size
    short NE = 4,   // head elements per thread
    short Q  = 1,   // queries per threadgroup
    short C  = 32>  // cache items per threadgroup
kernel void kernel_flash_attn_ext_vec(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3   ntg[[threads_per_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short nsg = ntg.y; // number of simdgroups

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE; // note: this can be adjusted to support different head sizes and simdgroup work loads
    constexpr short SH  = 4*C;   // shared memory per simdgroup

    const short T = DK + nsg*SH; // shared memory size per query in (half)

  //threadgroup q_t   * sq  = (threadgroup q_t   *) (shmem_f16 +                  0*DK); // holds the query data
    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                  0*DK); // same as above but in q4_t
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 + sgitg*SH       + Q*DK); // scratch buffer for attention
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 + sgitg*SH       + Q*DK); // same as above but in s4_t
    threadgroup float * sm  = (threadgroup float *) (shmem_f16 + sgitg*SH + 2*C + Q*DK); // scratch buffer for mask
    threadgroup o4_t  * sr4 = (threadgroup o4_t  *) (shmem_f16 + sgitg*DV       + Q*T);  // scratch buffer for the results

    // store the result for all queries in local memory (the O matrix from the paper)
    o4_t lo[DV4/NL];

    // load heads from Q to shared memory
    device const float4 * q4 = (device const float4 *) ((device const char *) q + (iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03));

    for (short i = tiisg; i < DK4; i += NW) {
        if (iq1 < args.ne01) {
            sq4[i] = (q4_t) q4[i];
        } else {
            sq4[i] = (q4_t) 0.0f;
        }
    }

    // zero out lo
    for (short i = 0; i < DV4/NL; ++i) {
        lo[i] = (o4_t) 0.0f;
    }

    // zero out shared memory SH
    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = 0.0f;
        float M = -__FLT16_MAX__/2;

        // thread indices inside the simdgroup
        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        // broadcast kv
        //const short rk2 = args.ne02/args.ne12;
        //const short rk3 = args.ne03/args.ne13;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        const bool has_mask = mask != q;

        // pointer to the mask
        device const half * pm = (device const half *) (mask + iq1*args.nb31);

        float slope = 1.0f;

        // ALiBi
        if (args.max_bias > 0.0f) {
            const short h = iq2;

            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

            slope = pow(base, exph);
        }

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = 0; ic0 < args.ne11; ic0 += C*nsg) {
            const int ic = ic0 + C*sgitg;
            if (ic >= args.ne11) {
                break;
            }

            if (has_mask) {
                sm[tiisg] = pm[ic + tiisg];
            }

            // Q*K^T
            {
                // each simdgroup processes 1 query and NE (NW/NL) head elements
                for (short cc = 0; cc < C/NE; ++cc) {
                    qk_t mqk = 0.0f;

                    device const kd4_t * pk = (device const kd4_t *) ((device const char *) k + ((ic + NE*cc + ty)*args.nb11 + ikv2*args.nb12 + ikv3*args.nb13));

                    #pragma unroll(DK4/NL)
                    for (short ii = 0; ii < DK4; ii += NL) {
                        const short i = ii + tx;

                        k4_t mk;
                        deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                        // note: this is less precise than the version below
                        //mqka[0] += dot(mq[0], mk[0]);
                        //mqka[1] += dot(mq[1], mk[1]);
                        //mqka[2] += dot(mq[2], mk[2]);
                        //mqka[3] += dot(mq[3], mk[3]);

                        //q4x4_t mq = sq4x4[i];
                        //mqka[0] += dot((float4) mq[0], (float4) mk[0]);
                        //mqka[1] += dot((float4) mq[1], (float4) mk[1]);
                        //mqka[2] += dot((float4) mq[2], (float4) mk[2]);
                        //mqka[3] += dot((float4) mq[3], (float4) mk[3]);

                        mqk += dot((float4) mk, (float4) sq4[i]);
                    }

                    static_assert(NE > 1, "NE must be > 1"); // note: not sure why NE == 1 fails

                    // simdgroup reduce (NE = 4)
                    // [ 0 ..  7] -> [ 0]
                    // [ 8 .. 15] -> [ 8]
                    // [16 .. 23] -> [16]
                    // [24 .. 31] -> [24]
                    if (NE <= 1) {
                        mqk += simd_shuffle_down(mqk, 16);
                    }
                    if (NE <= 2) {
                        mqk += simd_shuffle_down(mqk,  8);
                    }
                    if (NE <= 4) {
                        mqk += simd_shuffle_down(mqk,  4);
                    }
                    if (NE <= 8) {
                        mqk += simd_shuffle_down(mqk,  2);
                    }
                    if (NE <= 16) {
                        mqk += simd_shuffle_down(mqk,  1);
                    }

                    // mqk = mqk*scale + mask*slope
                    if (tx == 0) {
                        mqk *= args.scale;

                        if (args.logit_softcap != 0.0f) {
                            mqk = args.logit_softcap*precise::tanh(mqk);
                        }

                        mqk += sm[NE*cc + ty]*slope;

                        ss[NE*cc + ty] = mqk;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            {
                const float m = M;
                const float s = ss[tiisg];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                // the P matrix from the paper (Q rows, C columns)
                ss[tiisg] = vs;

                // O = diag(ms)*O
                #pragma unroll(DV4/NL)
                for (short ii = 0; ii < DV4; ii += NL) {
                    lo[ii/NL] *= ms;
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                //#pragma unroll(C/NE)
                for (short cc = 0; cc < C/NE; ++cc) {
                    device const vd4_t * pv4 = (device const vd4_t *) ((device const char *) v + ((ic + NE*cc + ty)*args.nb21 + ikv2*args.nb22 + ikv3*args.nb23));

                    const s4_t ms(ss[NE*cc + ty]);

                    #pragma unroll(DV4/NL)
                    for (short ii = 0; ii < DV4; ii += NL) {
                        const short i = ii + tx;

                        v4_t mv;
                        deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                        lo[ii/NL] += o4_t(float4(mv)*float4(ms));
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    // simdgroup reduce (NE = 4)
    // [ 0,  8, 16, 24] -> [ 0]
    // [ 1,  9, 17, 25] -> [ 1]
    // [ 2, 10, 18, 26] -> [ 2]
    // [ 3, 11, 19, 27] -> [ 3]
    // [ 4, 12, 20, 28] -> [ 4]
    // [ 5, 13, 21, 29] -> [ 5]
    // [ 6, 14, 22, 30] -> [ 6]
    // [ 7, 15, 23, 31] -> [ 7]
    for (short ii = 0; ii < DV4; ii += NL) {
        if (NE > 1) {
            lo[ii/NL][0] += simd_shuffle_down(lo[ii/NL][0], 16);
            lo[ii/NL][1] += simd_shuffle_down(lo[ii/NL][1], 16);
            lo[ii/NL][2] += simd_shuffle_down(lo[ii/NL][2], 16);
            lo[ii/NL][3] += simd_shuffle_down(lo[ii/NL][3], 16);
        }

        if (NE > 2) {
            lo[ii/NL][0] += simd_shuffle_down(lo[ii/NL][0],  8);
            lo[ii/NL][1] += simd_shuffle_down(lo[ii/NL][1],  8);
            lo[ii/NL][2] += simd_shuffle_down(lo[ii/NL][2],  8);
            lo[ii/NL][3] += simd_shuffle_down(lo[ii/NL][3],  8);
        }

        if (NE > 4) {
            lo[ii/NL][0] += simd_shuffle_down(lo[ii/NL][0],  4);
            lo[ii/NL][1] += simd_shuffle_down(lo[ii/NL][1],  4);
            lo[ii/NL][2] += simd_shuffle_down(lo[ii/NL][2],  4);
            lo[ii/NL][3] += simd_shuffle_down(lo[ii/NL][3],  4);
        }

        if (NE > 8) {
            lo[ii/NL][0] += simd_shuffle_down(lo[ii/NL][0],  2);
            lo[ii/NL][1] += simd_shuffle_down(lo[ii/NL][1],  2);
            lo[ii/NL][2] += simd_shuffle_down(lo[ii/NL][2],  2);
            lo[ii/NL][3] += simd_shuffle_down(lo[ii/NL][3],  2);
        }

        if (NE > 16) {
            lo[ii/NL][0] += simd_shuffle_down(lo[ii/NL][0],  1);
            lo[ii/NL][1] += simd_shuffle_down(lo[ii/NL][1],  1);
            lo[ii/NL][2] += simd_shuffle_down(lo[ii/NL][2],  1);
            lo[ii/NL][3] += simd_shuffle_down(lo[ii/NL][3],  1);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // store results to shared memory
    for (short i = tiisg; i < DV4; i += NL) {
        sr4[i] = lo[i/NL];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce
    for (short r = nsg/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float S0 = ss[           0];
            const float S1 = ss[r*(SH/2) + 0];

            const float M0 = ss[           1];
            const float M1 = ss[r*(SH/2) + 1];

            const float M = max(M0, M1);

            const float ms0 = exp(M0 - M);
            const float ms1 = exp(M1 - M);

            const float S = S0*ms0 + S1*ms1;

            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }

            // O_0 = diag(ms0)*O_0 + diag(ms1)*O_1
            for (short i = tiisg; i < DV4; i += NW) {
                sr4[i] = sr4[i]*ms0 + sr4[i + r*DV4]*ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float4 * dst4 = (device float4 *) dst;

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        const float S = ss[0];

        for (short i = tiisg; i < DV4; i += NW) {
            dst4[((uint64_t)iq3*args.ne2*args.ne1 + iq2 + (uint64_t)iq1*args.ne1)*DV4 + i] = (float4) sr4[i]/S;
        }
    }
}

// note: I think the s_t can be half instead of float, because the Q*K scaling is done before storing to shared mem
//       in the other (non-vec) kernel, we need s_t to also be float because we scale during the soft_max
//
#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           half4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

template [[host_name("kernel_flash_attn_ext_vec_f16_h64")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,             1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  64, 64, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_h64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0,        8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f16_h80")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,             1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  80, 80, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_h80")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0,        8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 80, 80, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f16_h96")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,             1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_h96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0,        8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f16_h112")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,             1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  112, 112, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_h112")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0,        8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 112, 112, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f16_h128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,             1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  128, 128, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_h128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0,        8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f16_h256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,             1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_h256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0,        8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f16_hk192_hv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_hk192_hv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4>;

template [[host_name("kernel_flash_attn_ext_vec_f16_hk576_hv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_hk576_hv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4>;

#undef FA_TYPES

template<typename T0, typename T1>
kernel void kernel_cpy(
        device  const void * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    device T1 * dst_data = (device T1 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {
        device const T0 * src = (device T0 *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = (T1) src[0];
    }
}

typedef decltype(kernel_cpy<float, float>) kernel_cpy_t;

template [[host_name("kernel_cpy_f32_f32")]]  kernel kernel_cpy_t kernel_cpy<float,  float>;
template [[host_name("kernel_cpy_f32_f16")]]  kernel kernel_cpy_t kernel_cpy<float,  half>;
template [[host_name("kernel_cpy_f16_f16")]]  kernel kernel_cpy_t kernel_cpy<half,   half>;
template [[host_name("kernel_cpy_f16_f32")]]  kernel kernel_cpy_t kernel_cpy<half,   float>;

kernel void kernel_cpy_f32_q8_0(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK8_0;

    device block_q8_0 * dst_data = (device block_q8_0 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK8_0; i00 < ne00; i00 += ntg.x*QK8_0) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = src[j];
            amax = MAX(amax, fabs(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK8_0].d = d;

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = src[j]*id;

            dst_data[i00/QK8_0].qs[j] = round(x0);
        }
    }
}

kernel void kernel_cpy_f32_q4_0(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK4_0;

    device block_q4_0 * dst_data = (device block_q4_0 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK4_0; i00 < ne00; i00 += ntg.x*QK4_0) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK4_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK4_0].d = d;

        for (int j = 0; j < QK4_0/2; ++j) {
            const float x0 = src[0       + j]*id;
            const float x1 = src[QK4_0/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            dst_data[i00/QK4_0].qs[j]  = xi0;
            dst_data[i00/QK4_0].qs[j] |= xi1 << 4;
        }
    }
}

kernel void kernel_cpy_f32_q4_1(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK4_1;

    device block_q4_1 * dst_data = (device block_q4_1 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK4_1; i00 < ne00; i00 += ntg.x*QK4_1) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int j = 0; j < QK4_1; j++) {
            const float v = src[j];
            if (min > v) min = v;
            if (max < v) max = v;
        }

        const float d = (max - min) / ((1 << 4) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK4_1].d = d;
        dst_data[i00/QK4_1].m = min;

        for (int j = 0; j < QK4_1/2; ++j) {
            const float x0 = (src[0       + j] - min)*id;
            const float x1 = (src[QK4_1/2 + j] - min)*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

            dst_data[i00/QK4_1].qs[j]  = xi0;
            dst_data[i00/QK4_1].qs[j] |= xi1 << 4;
        }
    }
}

kernel void kernel_cpy_f32_q5_0(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK5_0;

    device block_q5_0 * dst_data = (device block_q5_0 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK5_0; i00 < ne00; i00 += ntg.x*QK5_0) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK5_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / -16;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK5_0].d = d;

        uint32_t qh = 0;
        for (int j = 0; j < QK5_0/2; ++j) {
            const float x0 = src[0       + j]*id;
            const float x1 = src[QK5_0/2 + j]*id;

            const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
            const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

            dst_data[i00/QK5_0].qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
        }
        thread const uint8_t * qh8 = (thread const uint8_t *)&qh;
        for (int j = 0; j < 4; ++j) {
            dst_data[i00/QK5_0].qh[j] = qh8[j];
        }
    }
}

kernel void kernel_cpy_f32_q5_1(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK5_1;

    device block_q5_1 * dst_data = (device block_q5_1 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK5_1; i00 < ne00; i00 += ntg.x*QK5_1) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float max = src[0];
        float min = src[0];

        for (int j = 1; j < QK5_1; j++) {
            const float v = src[j];
            min = v < min ? v : min;
            max = v > max ? v : max;
        }

        const float d = (max - min) / 31;
        const float id = d ? 1.0f/d : 0.0f;

        dst_data[i00/QK5_1].d = d;
        dst_data[i00/QK5_1].m = min;

        uint32_t qh = 0;
        for (int j = 0; j < QK5_1/2; ++j) {
            const float x0 = (src[0       + j] - min)*id;
            const float x1 = (src[QK5_1/2 + j] - min)*id;

            const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
            const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

            dst_data[i00/QK5_1].qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
            qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
            qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
        }
        thread const uint8_t * qh8 = (thread const uint8_t *)&qh;
        for (int j = 0; j < 4; ++j) {
            dst_data[i00/QK5_1].qh[j] = qh8[j];
        }
    }
}

kernel void kernel_cpy_f32_q6_0(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK6_0;

    device block_q6_0 * dst_data = (device block_q6_0 *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK6_0; i00 < ne00; i00 += ntg.x*QK6_0) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK6_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / -32;
        const float id = d ? 1.0f/d : 0.0f;

        device block_q6_0 & b6 = dst_data[i00/QK6_0];
        b6.d = d;
        device uint16_t * aux16 = (device uint16_t *)b6.qh;
        aux16[0] = aux16[1] = aux16[2] = aux16[3] = 0;

        for (int j = 0; j < QK6_0/2; ++j) {
            const float x0 = src[0       + j]*id;
            const float x1 = src[QK6_0/2 + j]*id;

            const uint8_t xi0 = MIN(63, (int8_t)(x0 + 32.5f));
            const uint8_t xi1 = MIN(63, (int8_t)(x1 + 32.5f));

            b6.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
            const uint8_t h = (xi0 >> 4) | ((xi1 >> 4) << 2);
            b6.qh[j%(QK6_0/4)] |= (h << 4*(j/(QK6_0/4)));
        }
    }
}

static inline int best_index_int8(int n, constant float * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant static float kvalues_iq4k_f[32] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f,
    -123.f, -100.f, -79.f, -61.f, -45.f, -31.f, -18.f,  -6.f, 5.f, 17.f, 29.f, 42.f, 57.f, 73.f, 93.f, 117.f,
};

constexpr constant static float kvalues_iq5k_f[64] = {
    -126.f, -114.f, -103.f, -92.f, -83.f, -74.f, -65.f, -57.f, -50.f, -43.f, -36.f, -30.f, -24.f, -18.f, -12.f, -6.f, -1.f, 5.f, 11.f, 17.f, 23.f, 29.f, 36.f, 43.f, 51.f, 59.f, 68.f, 77.f, 87.f, 97.f, 109.f, 121.f,
    -124.f, -112.f, -101.f, -90.f, -81.f, -72.f, -63.f, -55.f, -48.f, -41.f, -34.f, -28.f, -22.f, -16.f, -10.f, -4.f,  1.f, 7.f, 13.f, 19.f, 25.f, 31.f, 38.f, 45.f, 53.f, 61.f, 70.f, 79.f, 89.f, 99.f, 111.f, 123.f,
};

constexpr constant static float kvalues_iq6k_f[128] = {
    -127.f, -121.f, -115.f, -109.f, -104.f,  -98.f,  -93.f,  -88.f,  -84.f,  -79.f,  -74.f,  -70.f,  -66.f,  -62.f,  -58.f,  -54.f,
     -51.f,  -47.f,  -44.f,  -40.f,  -37.f,  -34.f,  -31.f,  -28.f,  -25.f,  -22.f,  -19.f,  -16.f,  -13.f,  -11.f,   -8.f,   -5.f,
      -2.f,    0.f,    3.f,    6.f,    9.f,   12.f,   14.f,   17.f,   20.f,   23.f,   27.f,   30.f,   33.f,   36.f,   40.f,   44.f,
      47.f,   51.f,   55.f,   59.f,   63.f,   68.f,   72.f,   77.f,   82.f,   87.f,   92.f,   98.f,  103.f,  109.f,  115.f,  121.f,
    -126.f, -120.f, -114.f, -108.f, -103.f,  -97.f,  -92.f,  -87.f,  -83.f,  -78.f,  -73.f,  -69.f,  -65.f,  -61.f,  -57.f,  -53.f,
     -50.f,  -46.f,  -43.f,  -39.f,  -36.f,  -33.f,  -30.f,  -27.f,  -24.f,  -21.f,  -18.f,  -15.f,  -12.f,  -10.f,   -7.f,   -4.f,
      -1.f,    1.f,    4.f,    7.f,   10.f,   13.f,   15.f,   18.f,   21.f,   24.f,   28.f,   31.f,   34.f,   37.f,   41.f,   45.f,
      48.f,   52.f,   56.f,   60.f,   64.f,   69.f,   73.f,   78.f,   83.f,   88.f,   93.f,   99.f,  104.f,  110.f,  116.f,  122.f,
};

constexpr constant static float kvalues_iq2k_f[8] = { -31.f, -13.f, 1.f, 17.f, -26.f, -8.f, 6.f, 22.f };
constexpr constant static half  kvalues_iq2k_h[8] = { -31.h, -13.h, 1.h, 17.h, -26.h, -8.h, 6.h, 22.h };

constexpr constant static float kvalues_iq3k_f[16] = { -63.f, -40.f, -23.f, -10.f, 1.f, 13.f, 28.f,  47.f, -59.f, -36.f, -19.f,  -6.f, 5.f, 17.f, 32.f,  51.f };
constexpr constant static half  kvalues_iq3k_h[16] = { -63.h, -40.h, -23.h, -10.h, 1.h, 13.h, 28.h,  47.h, -59.h, -36.h, -19.h,  -6.h, 5.h, 17.h, 32.h,  51.h };

kernel void kernel_cpy_f32_iq4_nl(
        device const float * src0,
        device        void * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant   int64_t & ne03,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant  uint64_t & nb03,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   int64_t & ne2,
        constant   int64_t & ne3,
        constant  uint64_t & nb0,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        constant  uint64_t & nb3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int64_t i03 = tgpig[2];
    const int64_t i02 = tgpig[1];
    const int64_t i01 = tgpig[0];

    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    const int64_t i3 = n / (ne2*ne1*ne0);
    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0)/QK4_NL;

    device block_iq4_nl * dst_data = (device block_iq4_nl *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int64_t i00 = tpitg.x*QK4_NL; i00 < ne00; i00 += ntg.x*QK4_NL) {
        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < QK4_0; j++) {
            const float v = src[j];
            if (amax < fabs(v)) {
                amax = fabs(v);
                max  = v;
            }
        }

        const float d = max / kvalues_iq4nl_f[0];
        const float id = d ? 1.0f/d : 0.0f;

        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            const float x0 = src[0        + j]*id;
            const float x1 = src[QK4_NL/2 + j]*id;

            const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
            const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

            dst_data[i00/QK4_NL].qs[j] = xi0 | (xi1 << 4);

            const float v0 = kvalues_iq4nl_f[xi0];
            const float v1 = kvalues_iq4nl_f[xi1];
            const float w0 = src[0        + j]*src[0        + j];
            const float w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
            sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
            sumq2 += w0*v0*v0 + w1*v1*v1;

        }

        dst_data[i00/QK4_NL].d = sumq2 > 0 ? sumqx/sumq2 : d;

    }
}

template <typename src_t>
static inline void concat_impl(
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne10,
    constant   int64_t & ne11,
    constant   int64_t & ne12,
    constant   int64_t & ne13,
    constant  uint64_t & nb10,
    constant  uint64_t & nb11,
    constant  uint64_t & nb12,
    constant  uint64_t & nb13,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    constant   int32_t & dim,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

    device const src_t * x;

    for (int i0 = tpitg.x; i0 < ne0; i0 += ntg.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (device const src_t *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            x = (device const src_t *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
        }

        device src_t * y = (device src_t *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

kernel void kernel_concat_f32(
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne10,
    constant   int64_t & ne11,
    constant   int64_t & ne12,
    constant   int64_t & ne13,
    constant  uint64_t & nb10,
    constant  uint64_t & nb11,
    constant  uint64_t & nb12,
    constant  uint64_t & nb13,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    constant   int32_t & dim,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    concat_impl<float>(src0, src1, dst, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3, dim, tgpig, tpitg, ntg);
}

kernel void kernel_concat_f16(
    device  const char * src0,
    device  const char * src1,
    device        char * dst,
    constant   int64_t & ne00,
    constant   int64_t & ne01,
    constant   int64_t & ne02,
    constant   int64_t & ne03,
    constant  uint64_t & nb00,
    constant  uint64_t & nb01,
    constant  uint64_t & nb02,
    constant  uint64_t & nb03,
    constant   int64_t & ne10,
    constant   int64_t & ne11,
    constant   int64_t & ne12,
    constant   int64_t & ne13,
    constant  uint64_t & nb10,
    constant  uint64_t & nb11,
    constant  uint64_t & nb12,
    constant  uint64_t & nb13,
    constant   int64_t & ne0,
    constant   int64_t & ne1,
    constant   int64_t & ne2,
    constant   int64_t & ne3,
    constant  uint64_t & nb0,
    constant  uint64_t & nb1,
    constant  uint64_t & nb2,
    constant  uint64_t & nb3,
    constant   int32_t & dim,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    concat_impl<half>(src0, src1, dst, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3, dim, tgpig, tpitg, ntg);
}



void kernel_mul_mv_q2_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q2_K * x = (device const block_q2_K *) src0 + ib_row + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int step = sizeof(block_q2_K) * nb;

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3
    const int is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += step/2;
            sc += step;
            dh += step/2;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_q2_K_f32")]]
kernel void kernel_mul_mv_q2_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q3_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int64_t im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q3_K * x = (device const block_q3_K *) src0 + first_row*nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];

    //const uint16_t kmask1 = 0x3030;
    //const uint16_t kmask2 = 0x0f0f;

    const int tid = tiisg/4;
    const int ix  = tiisg%4;
    const int ip  = tid/4;          // 0 or 1
    const int il  = 2*((tid%4)/2);  // 0 or 2
    const int ir  = tid%2;
    const int n   = 8;
    const int l0  = n*ir;

    // One would think that the Metal compiler would figure out that ip and il can only have
    // 4 possible states, and optimize accordingly. Well, no. It needs help, and we do it
    // with these two tales.
    //
    // Possible masks for the high bit
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip = 0, il = 0
                           {0x0004, 0x0400, 0x0008, 0x0800},  // ip = 0, il = 2
                           {0x0010, 0x1000, 0x0020, 0x2000},  // ip = 1, il = 0
                           {0x0040, 0x4000, 0x0080, 0x8000}}; // ip = 1, il = 2

    // Possible masks for the low 2 bits
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2*ip + il/2];

    const int shift = 2*il;
    const float    v1 = il == 0 ? 4.f : 64.f;
    const float    v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const int q_offset = 32*ip + l0;
    const int y_offset = 128*ip + 32*il + l0;

    const int step = sizeof(block_q3_K) * nb / 2;

    device const float * y1 = yy + ix*QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[2] = {0.f};
    float sumf2[2] = {0.f};
    for (int i = ix; i < nb; i += 4) {

        for (int l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (int row = 0; row < 2; ++row) {

            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (int l = 0; l < n; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (int l = 0; l < n; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += step;
            h  += step;
            a  += step;
            dh += step;

        }

        y1 += 4 * QK_K;

    }

    for (int row = 0; row < 2; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }
    if (tiisg == 0) {
        for (int row = 0; row < 2; ++row) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = sumf1[row];
        }
    }
}

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q4_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    //const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int first_row = r0 * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row + offset0;
    device const float      * y = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16];
    float yh[16];
    float sumf[N_DST]={0.f}, all_sum;

    const int step = sizeof(block_q4_K) * nb / 2;

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {

        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+0] * (q1[i/2] & 0x000F);
                acc1[1] += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1[2] += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1[3] += yl[i+9] * (q1[i/2] & 0xF000);
                acc2[0] += yh[i+0] * (q2[i/2] & 0x000F);
                acc2[1] += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2[2] += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2[3] += yh[i+9] * (q2[i/2] & 0xF000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                 (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                 (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += step;
            sc += step;
            dh += step;
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q5_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q5_K * x = (device const block_q5_K *) src0 + first_row*nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float sumf[2]={0.f};

    const int step = sizeof(block_q5_K) * nb;

    float yl[16], yh[16];

    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = tiisg/4;
    const int ix  = tiisg%4;
    const int iq  = tid/4;
    const int ir  = tid%4;
    const int n   = 8;

    const int l0 = n*ir;
    const int q_offset = 32*iq + l0;
    const int y_offset = 64*iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {

        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (int row = 0; row < 2; ++row) {

            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            for (int l = 0; l < n; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }
            const float dall = dh[0];
            const float dmin = dh[1];
            sumf[row] += dall * (sc8[0] * (acc1[0] +  16.f*acc2[0]) +
                                 sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                 sc8[4] * (acc1[2] +  16.f*acc2[2]) +
                                 sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += step;
            qh += step;
            dh += step/2;
            a  += step/2;

        }

        y1 += 4 * QK_K;

    }

    for (int row = 0; row < 2; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_q6_K_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const int nb = ne00/QK_K;

    const int64_t r0 = tgpig.x;
    const int64_t r1 = tgpig.y;
    const int     im = tgpig.z;

    const int row = 2 * r0 + sgitg;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_q6_K * x = (device const block_q6_K *) src0 + row * nb + offset0;
    device const float     * yy = (device const float      *) src1 + r1*ne10 + im*ne00*ne1;

    float sumf = 0;

    const int tid  = tiisg/2;
    const int ix   = tiisg%2;
    const int ip   = tid/8;         // 0 or 1
    const int il   = tid%8;
    const int n    = 4;
    const int l0   = n*il;
    const int is   = 8*ip + l0/16;

    const int y_offset = 128*ip + l0;
    const int q_offset_l = 64*ip + l0;
    const int q_offset_h = 32*ip + l0;

    for (int i = ix; i < nb; i += 2) {

        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;

        device const float * y = yy + i * QK_K + y_offset;

        const float dall = x[i].d;

        float4 sums = {0.f, 0.f, 0.f, 0.f};
        for (int l = 0; l < n; ++l) {
            sums[0] += y[l+ 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            sums[1] += y[l+32] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            sums[2] += y[l+64] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
            sums[3] += y[l+96] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
        }

        sumf += dall * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

    }

    const float tot = simd_sum(sumf);
    if (tiisg == 0) {
        dst[r1*ne0 + im*ne0*ne1 + row] = tot;
    }
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

// ======================= "True" 2-bit

void kernel_mul_mv_iq2_xxs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) src0 + ib_row + offset0;
    device const float         * y = (device const float         *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * values = (threadgroup uint64_t *)shared_values;
    threadgroup uint8_t  * shared_signs = (threadgroup uint8_t *)(values + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) shared_signs[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(values + aux8[l]);
                const uint8_t signs = shared_signs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += nb*sizeof(block_iq2_xxs)/2;
            q2 += nb*sizeof(block_iq2_xxs)/2;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xxs_f32")]]
kernel void kernel_mul_mv_iq2_xxs_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xxs_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq2_xs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq2_xs * x = (device const block_iq2_xs *) src0 + ib_row + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * values = (threadgroup uint64_t *)shared_values;
    threadgroup uint8_t  * shared_signs = (threadgroup uint8_t *)(values + 512);
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq2xs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) shared_signs[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (int l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(values + (q2[l] & 511));
                const uint8_t signs = shared_signs[(q2[l] >> 9)];
                for (int j = 0; j < 8; ++j) {
                    sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (int l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(values + (q2[l] & 511));
                const uint8_t signs = shared_signs[(q2[l] >> 9)];
                for (int j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += nb*sizeof(block_iq2_xs)/2;
            q2 += nb*sizeof(block_iq2_xs)/2;
            sc += nb*sizeof(block_iq2_xs);
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xs_f32")]]
kernel void kernel_mul_mv_iq2_xs_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq3_xxs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq3_xxs * x = (device const block_iq3_xxs *) src0 + ib_row + offset0;
    device const float         * y = (device const float         *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * values = (threadgroup uint32_t *)shared_values;
    threadgroup uint8_t  * shared_signs = (threadgroup uint8_t *)(values + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq3xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) shared_signs[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(values + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(values + q3[2*l+1]);
                const uint8_t signs = shared_signs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += nb*sizeof(block_iq3_xxs)/2;
            q3  += nb*sizeof(block_iq3_xxs);
            gas += nb*sizeof(block_iq3_xxs)/2;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.5f;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq3_s_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq3_s * x = (device const block_iq3_s *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * values = (threadgroup uint32_t *)shared_values;
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) values[pos + i] = iq3s_grid[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

            float2 sum = {0};
            for (int l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? values + 256 : values;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? values + 256 : values;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                for (int j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += nb*sizeof(block_iq3_s)/2;
            qs  += nb*sizeof(block_iq3_s);
            qh  += nb*sizeof(block_iq3_s);
            sc  += nb*sizeof(block_iq3_s);
            signs += nb*sizeof(block_iq3_s);
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_s_f32")]]
kernel void kernel_mul_mv_iq3_s_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq2_s_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq2_s * x = (device const block_iq2_s *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    //threadgroup uint64_t * values = (threadgroup uint64_t *)shared_values;
    //{
    //    int nval = 32;
    //    int pos  = (32*sgitg + tiisg)*nval;
    //    for (int i = 0; i < nval; ++i) values[pos + i] = iq2s_grid[pos + i];
    //    threadgroup_barrier(mem_flags::mem_threadgroup);
    //}

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + ib;
        device const uint8_t * signs = qs + QK_K/8;
        device const half * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >>  4));

            float2 sum = {0};
            for (int l = 0; l < 2; ++l) {
                //const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(values + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                //const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(values + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sum[0] += yl[8*l + j +  0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);
                    sum[1] += yl[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];

            dh  += nb*sizeof(block_iq2_s)/2;
            qs  += nb*sizeof(block_iq2_s);
            qh  += nb*sizeof(block_iq2_s);
            sc  += nb*sizeof(block_iq2_s);
            signs += nb*sizeof(block_iq2_s);
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_s_f32")]]
kernel void kernel_mul_mv_iq2_s_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq1_s_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_value,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq1_s * x = (device const block_iq1_s *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        float sumy = 0;
        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
            sumy += yl[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_s * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint16_t * qh = xr->qh + ib;
        device const half     * dh = &xr->d;

        for (int row = 0; row < N_DST; row++) {

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 5) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[0] << 2) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[0] >> 1) & 0x700)));

            float sum = 0;
            for (int j = 0; j < 4; ++j) {
                sum += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                     + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4)
                     + yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                     + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);

            dh += nb*sizeof(block_iq1_s)/2;
            qs += nb*sizeof(block_iq1_s);
            qh += nb*sizeof(block_iq1_s)/2;
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

void kernel_mul_mv_iq1_m_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_value,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq1_m * x = (device const block_iq1_m *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f}, all_sum;

    const int nb32 = nb * (QK_K / 32);

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    iq1m_scale_t scale;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {

        float4 sumy = {0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_m * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint8_t  * qh = xr->qh + 2 * ib;
        device const uint16_t * sc = (device const uint16_t *)xr->scales;

        for (int row = 0; row < N_DST; row++) {
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

            float2 sum = {0.f};
            for (int j = 0; j < 4; ++j) {
                sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                        + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);
                sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                        + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

            sumf[row] += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +
                                             (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));

            sc += nb*sizeof(block_iq1_m)/2;
            qs += nb*sizeof(block_iq1_m);
            qh += nb*sizeof(block_iq1_m);
        }

        y4 += 32 * 32;
    }

    for (int row = 0; row < N_DST; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

static inline float iq1bn_fp8_to_float(uint8_t fp8) {
    typedef union { float f; uint32_t i; } scale_t;
    scale_t s; s.i = (((fp8 >> 5) + 116) << 23) | ((fp8 & 0x1f) << 18);
    return s.f;
}

void kernel_mul_mv_iq1_bn_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_value,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_IQ1BN;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int row_size = nb*sizeof(block_iq1_bn) + 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = ((i12/r2)*ne01 + (i13/r3)*ne01*ne02)*row_size;
    device const uint8_t * cx = (device const uint8_t *) src0 + first_row*row_size + offset0;
    device const float   *  y = (device const float   *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[16];
    float sumf[N_DST]={0.f};
    float scale[N_DST];

    const int nb32 = nb * (QK_IQ1BN / 32);

    const int ix = tiisg/2;
    const int ir = tiisg%2;

    for (int row = 0; row < N_DST; ++row) {
        half d16;
        thread uint8_t * aux = (thread uint8_t *)&d16;
        device const uint8_t * cr = cx + row*row_size;
        aux[0] = cr[0]; aux[1] = cr[1];
        scale[row] = d16;
    }
    device const block_iq1_bn * x = (device const block_iq1_bn *)(cx + 2);

    device const float * y4 = (device const float *)y + 32 * ix + 16 * ir;

    constexpr uint16_t k_mult[5] = {81, 27, 9, 3, 1};

    const int ib  = ix % (QK_IQ1BN / 32);
    const int i16 = 2*ib + ir;

    float sumy = 0;
    for (int ib32 = ix; ib32 < nb32; ib32 += 16) {

        for (int j = 0; j < 16; ++j) { yl[j] = y4[j]; sumy += y4[j]; }

        const int ibl = ib32 / (QK_IQ1BN / 32);
        device const block_iq1_bn * xr = x + ibl;
        device const uint8_t * ql = xr->ql + 3*i16;
        device const uint8_t * extra = (device const uint8_t *)&xr->extra;

        for (int row = 0; row < N_DST; row++) {

            float acc = 0;
            thread const float * yy = yl;
            for (int k = 0; k < 3; ++k) {
                uint16_t q = ql[k];
                for (int j = 4; j >= 0; --j) {
                    uint16_t v = q & 0xff;
                    v += v << 1;
                    acc += yy[j] * (v & 0xff00);
                    q += q << 1;
                }
                yy += 5;
            }
            uint16_t v = (k_mult[i16]*extra[0]) & 0xff;
            v += v << 1;
            acc += yl[15] * (v & 0xff00);

            sumf[row] += acc;

            extra += row_size;
            ql    += row_size;
        }

        y4 += 32 * 16;
    }

    for (int row = 0; row < N_DST; row += 2) {
        float2 r = {0.00390625f * sumf[row] - sumy, 0.00390625 * sumf[row+1] - sumy};
        r = simd_sum(r);
        if (tiisg < 2) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row + tiisg] = r[tiisg] * scale[row + tiisg];
        }
    }
}

void kernel_mul_mv_iq2_bn_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_value,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_IQ1BN;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int row_size = nb*sizeof(block_iq2_bn) + sizeof(float);

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = ((i12/r2)*ne01 + (i13/r3)*ne01*ne02)*row_size;

    device const uint8_t * cx = (device const uint8_t *) src0 + first_row*row_size + offset0;
    device const float   *  y = (device const float   *) src1 + r1*ne10 + im*ne00*ne1;

    float4 yl[4];
    float  sumf[N_DST]={0.f};
    float  scale[N_DST];

    for (int row = 0; row < N_DST; ++row) {
        scale[row] = *((device const float *)(cx + row*row_size));
    }

    const int ix = tiisg/4; // 0...7
    const int ir = tiisg%4; // 0...3

    device const float4  * y4  = (device const float4 *)(y + QK_IQ1BN * ix + 4 * ir);
    device const uint8_t * qs0 = cx + sizeof(float) + (QK_IQ1BN/4)*ix + 4*ir;

    for (int ib = ix; ib < nb; ib += 8) {

        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[8]; yl[3] = y4[12];
        float4 tmp = yl[0] + yl[1] + yl[2] + yl[3];
        const float sumy = tmp[0] + tmp[1] + tmp[2] + tmp[3];

        device const uint8_t * qs = qs0;

        for (int row = 0; row < N_DST; row++) {

            float4 acc = {0.f};

            for (int j = 0; j < 4; ++j) {
                acc[0] += yl[0][j] * (qs[j] & 0x03);
                acc[1] += yl[1][j] * (qs[j] & 0x0c);
                acc[2] += yl[2][j] * (qs[j] & 0x30);
                acc[3] += yl[3][j] * (qs[j] & 0xc0);
            }

            sumf[row] += acc[0] + 0.25f*acc[1] + 0.0625*acc[2] + 0.015625f*acc[3] - sumy;

            qs += row_size;
        }

        y4  += QK_IQ1BN * 2;
        qs0 += QK_IQ1BN * 2;
    }

    for (int row = 0; row < N_DST; row += 2) {
        float2 r = {sumf[row], sumf[row+1]};
        r = simd_sum(r);
        if (tiisg < 2) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row+tiisg] = r[tiisg] * scale[row+tiisg];
        }
    }
}

void kernel_mul_mv_iq4_nl_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values_i8,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    threadgroup float * shared_values = (threadgroup float *)shared_values_i8;
    const int nb = ne00/QK4_NL;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq4_nl * x = (device const block_iq4_nl *) src0 + ib_row + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    const int ix = tiisg/2;  // 0...15
    const int it = tiisg%2;  // 0 or 1

    shared_values[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2]={0.f}, all_sum;

    device const float * yb = y + ix * QK4_NL + it * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ib = ix; ib < nb; ib += 16) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];

        for (int row = 0; row < 2 && first_row + row < ne01; ++row) {

            device const block_iq4_nl & xb = x[row*nb + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);

        }

        yb += 16 * QK4_NL;
    }

    for (int row = 0; row < 2 && first_row + row < ne01; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

void kernel_mul_mv_iq4_xs_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values_i8,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    threadgroup float * shared_values = (threadgroup float *)shared_values_i8;
    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq4_xs * x = (device const block_iq4_xs *) src0 + ib_row + offset0;
    device const float        * y = (device const float        *) src1 + r1*ne10 + im*ne00*ne1;

    const int ix = tiisg/16;  // 0 or 1
    const int it = tiisg%16;  // 0...15
    const int ib = it/2;
    const int il = it%2;

    shared_values[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2]={0.f}, all_sum;

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];

        for (int row = 0; row < 2; ++row) {

            device const block_iq4_xs & xb = x[row*nb + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[1] & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shared_values[q8[0]], shared_values[q8[1]], shared_values[q8[2]], shared_values[q8[3]]};
            qf2 = {shared_values[q8[4]], shared_values[q8[5]], shared_values[q8[6]], shared_values[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);

        }

        yb += 2 * QK_K;
    }

    for (int row = 0; row < 2; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

void kernel_mul_mv_iq4_ks_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values_i8,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    threadgroup float * shared_values = (threadgroup float *)shared_values_i8;
    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint row_size = 4 + nb*sizeof(block_iq4_ks);
    const uint offset0 = (i12/r2)*ne01 + (i13/r3)*(ne01*ne02);
    device const char  * cx = (device const char  *)src0 + (first_row + offset0)*row_size;
    device const float *  y = (device const float *)src1 + r1*ne10 + im*ne00*ne1;

    const int ix = tiisg/16;  // 0 or 1
    const int it = tiisg%16;  // 0...15
    const int ib = it/2;
    const int il = it%2;

    shared_values[tiisg] = kvalues_iq4k_f[tiisg];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float2 sumf = 0.f;
    float  d[2];

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    device const float * dptr = (device const float *)cx;
    d[0] = *dptr;
    device const block_iq4_ks * x = (device const block_iq4_ks *)(dptr + 1) + ix;
    dptr += row_size/4;
    d[1] = *dptr;

    for (int ibl = ix; ibl < nb; ibl += 2) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];

        device const uint8_t * scales = x->scales;

        for (int row = 0; row < 2; ++row) {

            threadgroup const float * block_values = shared_values + ((scales[ib] & 1) << 4);
            const float ls = ((scales[ib] & 254) - 127);

            device const uint32_t * q4 = (device const uint32_t *)scales + QK_K/128 + 4*ib + 2*il;

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {block_values[q8[0]], block_values[q8[1]], block_values[q8[2]], block_values[q8[3]]};
            qf2 = {block_values[q8[4]], block_values[q8[5]], block_values[q8[6]], block_values[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[1] & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {block_values[q8[0]], block_values[q8[1]], block_values[q8[2]], block_values[q8[3]]};
            qf2 = {block_values[q8[4]], block_values[q8[5]], block_values[q8[6]], block_values[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += d[row] * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);

            scales += row_size;

        }

        yb += 2 * QK_K;
        x  += 2;
    }

    sumf = simd_sum(sumf);
    if (tiisg < 2) {
        dst[r1*ne0 + im*ne0*ne1 + first_row + tiisg] = sumf[tiisg];
    }
}

// TODO
void kernel_mul_mv_iq5_ks_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values_i8,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    threadgroup float * shared_values = (threadgroup float *)shared_values_i8;
    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint row_size = 4 + nb*sizeof(block_iq5_ks);
    const uint offset0 = (i12/r2)*ne01 + (i13/r3)*(ne01*ne02);
    device const char  * cx = (device const char  *)src0 + (first_row + offset0)*row_size;
    device const float *  y = (device const float *)src1 + r1*ne10 + im*ne00*ne1;

    const int ix = tiisg/16;  // 0 or 1
    const int it = tiisg%16;  // 0...15
    const int ib64 = it/4;    // 0...3
    const int il64 = it%4;    // 0...3

    shared_values[2*tiisg+0] = kvalues_iq5k_f[2*tiisg+0];
    shared_values[2*tiisg+1] = kvalues_iq5k_f[2*tiisg+1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float2 sumf = 0.f;
    float  d[2];

    device const float * yb = y + ix * QK_K + ib64 * 64 + il64 * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    device const float * dptr = (device const float *)cx;
    d[0] = *dptr;
    device const block_iq5_ks * x = (device const block_iq5_ks *)(dptr + 1) + ix;
    dptr += row_size/4;
    d[1] = *dptr;

    for (int ibl = ix; ibl < nb; ibl += 2) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[8]; yl[2] = y4[1]; yl[3] = y4[9];

        device const uint8_t * scales = x->scales;

        for (int row = 0; row < 2; ++row) {

            threadgroup const float * values1 = shared_values + ((scales[2*ib64+0] & 1) << 5);
            threadgroup const float * values2 = shared_values + ((scales[2*ib64+1] & 1) << 5);
            const float ls1 = ((scales[2*ib64+0] & 254) - 127);
            const float ls2 = ((scales[2*ib64+1] & 254) - 127);

            device const uint32_t * q4 = (device const uint32_t *)scales + QK_K/128 + 8*ib64 + 2*il64;
            device const uint32_t * qh = (device const uint32_t *)scales + QK_K/128 + QK_K/8 + 2*il64;

            float4 acc1 = {0.f}, acc2 = {0.f};

            uint32_t h = qh[0] >> 2*ib64;
            aux32[0] = ((q4[0] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x10101010);
            aux32[1] = ((q4[0] >> 4) & 0x0f0f0f0f) | ((h << 3) & 0x10101010);
            qf1 = {values1[q8[0]], values1[q8[1]], values1[q8[2]], values1[q8[3]]};
            qf2 = {values2[q8[4]], values2[q8[5]], values2[q8[6]], values2[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            h = qh[1] >> 2*ib64;
            aux32[0] = ((q4[1] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x10101010);
            aux32[1] = ((q4[1] >> 4) & 0x0f0f0f0f) | ((h << 3) & 0x10101010);
            qf1 = {values1[q8[0]], values1[q8[1]], values1[q8[2]], values1[q8[3]]};
            qf2 = {values2[q8[4]], values2[q8[5]], values2[q8[6]], values2[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            sumf[row] += ls1 * (acc1[0] + acc1[1] + acc1[2] + acc1[3]) + ls2 * (acc2[0] + acc2[1] + acc2[2] + acc2[3]);

            scales += row_size;

        }

        yb += 2 * QK_K;
        x  += 2;
    }

    sumf = simd_sum(sumf);
    if (tiisg < 2) {
        dst[r1*ne0 + im*ne0*ne1 + first_row + tiisg] = sumf[tiisg] * d[tiisg];
    }
}

void kernel_mul_mv_iq4_kss_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values_i8,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    threadgroup float * shared_values = (threadgroup float *)shared_values_i8;
    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint row_size = 4 + nb*sizeof(block_iq4_kss);
    const uint offset0 = (i12/r2)*ne01 + (i13/r3)*(ne01*ne02);
    device const char  * cx = (device const char  *)src0 + (first_row + offset0)*row_size;
    device const float *  y = (device const float *)src1 + r1*ne10 + im*ne00*ne1;

    const int ix = tiisg/16;  // 0 or 1
    const int it = tiisg%16;  // 0...15
    const int ib = it/2;
    const int il = it%2;

    shared_values[tiisg] = kvalues_iq4k_f[tiisg];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float2 sumf = 0.f;
    float d[2];

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;

    float4 qf1, qf2;

    device const float * dptr = (device const float *)cx;
    d[0] = *dptr;
    device const uint32_t * qptr = (device const uint32_t *)(dptr + 1) + ix*(QK_K/8) + 4*ib;
    dptr += row_size/4;
    d[1] = *dptr;

    for (int ibl = ix; ibl < nb; ibl += 2) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];

        device const uint32_t * q4 = qptr;

        for (int row = 0; row < 2; ++row) {

            uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
            int16_t ls = (s32 | (s32 >> 15)) & 0xff;

            threadgroup const float * block_values = shared_values + ((ls & 1) << 4);
            const float scale = ((ls & 254) - 127);

            float4 acc1 = {0.f}, acc2 = {0.f};

            uint32_t v32 = q4[2*il+0] & 0xfffefffe;
            v32 ^= (v32 >> 1);
            aux32 = v32 & 0x0f0f0f0f;
            qf1 = {block_values[q8[0]], block_values[q8[1]], block_values[q8[2]], block_values[q8[3]]};
            acc1 += yl[0] * qf1;
            aux32 = (v32 >> 4) & 0x0f0f0f0f;
            qf2 = {block_values[q8[0]], block_values[q8[1]], block_values[q8[2]], block_values[q8[3]]};
            acc2 += yl[1] * qf2;

            v32 = q4[2*il+1] & 0xfffefffe;
            v32 ^= (v32 >> 1);
            aux32 = v32 & 0x0f0f0f0f;
            qf1 = {block_values[q8[0]], block_values[q8[1]], block_values[q8[2]], block_values[q8[3]]};
            acc1 += yl[2] * qf1;
            aux32 = (v32 >> 4) & 0x0f0f0f0f;
            qf2 = {block_values[q8[0]], block_values[q8[1]], block_values[q8[2]], block_values[q8[3]]};
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += d[row] * scale * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);

            q4 += row_size/4;

        }

        yb += 2 * QK_K;
        qptr += 2 * (QK_K/8);
    }

    sumf = simd_sum(sumf);
    if (tiisg < 2) {
        dst[r1*ne0 + im*ne0*ne1 + first_row + tiisg] = sumf[tiisg];
    }
}

void kernel_mul_mv_iq2_k_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq2_k * x = (device const block_iq2_k *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f};

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3
    const int is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    threadgroup float * all_values = (threadgroup float *)shared_values + 8*sgitg;
    {
        if (tiisg < 8) all_values[tiisg] = kvalues_iq2k_f[tiisg];
        simdgroup_barrier(mem_flags::mem_none);
    }

    uint32_t aux32[2];
    thread const uint8_t * aux8 = (thread const uint8_t *)aux32;

    for (int ib = ix; ib < nb; ib += 4) {

        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0];
            yl[i+ 8] = y4[i+32];
            yl[i+16] = y4[i+64];
            yl[i+24] = y4[i+96];
        }

        for (int row = 0; row < N_DST; row++) {

            device const block_iq2_k & xb = x[row*nb + ib];
            device const uint32_t * q32 = (device const uint32_t *)xb.qs + 8*iq + 2*ir;
            device const uint32_t * sc  = (device const uint32_t *)xb.scales;

            const uint32_t scales32 = (sc[iq] >> 4*is) & 0x0f0f0f0f;
            thread const int8_t * s8 = (thread const int8_t *)&scales32;
            uint16_t extra = (xb.extra >> (8*iq + is)) << 2;

            float4 acc = {0.f};
            for (int l = 0; l < 4; ++l) {
                threadgroup const float * values = all_values + (extra & 4);
                aux32[0] = (q32[0] >> 2*l) & 0x03030303;
                aux32[1] = (q32[1] >> 2*l) & 0x03030303;
                for (int j = 0; j < 8; ++j) acc[l] += yl[8*l+j] * values[aux8[j]];
                extra >>= 2;
            }

            sumf[row] += (float)xb.d * (acc[0] * s8[0] + acc[1] * s8[1] + acc[2] * s8[2] + acc[3] * s8[3] - 8.f*(acc[0] + acc[1] + acc[2] + acc[3]));

        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; row += 2) {
        float2 tmp{sumf[row], sumf[row+1]};
        tmp = simd_sum(tmp);
        if (tiisg < 2) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row + tiisg] = tmp[tiisg];
        }
    }
}

struct Trellis3 {
    constexpr constant static uint32_t kmask = 0x3f3f3f3f;
    constexpr constant static uint32_t ka  = 0xCBAC1FED;
    constexpr constant static uint32_t ka1 = ka*ka;
    constexpr constant static uint32_t ka2 = ka1*ka;
    constexpr constant static uint32_t ka3 = ka2*ka;
    static inline char4 gen4(uint32_t val) {
        thread uint32_t aux[4] = {(ka*val) & kmask, (ka1*val) & kmask, (ka2*val) & kmask, (ka3*val) & kmask};
        thread const int8_t * a8 = (thread const int8_t *)aux;
        char4 result;
        for (int i = 0; i < 4; ++i) result[i] = -126 + a8[4*i+0] + a8[4*i+1] + a8[4*i+2] + a8[4*i+3];
        return result;
    }
    template <typename T4>
    static inline void gen8(uint32_t val, thread T4& v1, thread T4& v2) {
        thread uint32_t aux[4] = {ka*val, ka1*val, ka2*val, ka3*val};
        uint32_t aux32[2];
        thread const int8_t * a8 = (thread const int8_t *)aux32;
        //thread const char4 * a8 = (thread const char4 *)aux32;
        for (int i = 0; i < 4; ++i) {
            aux32[0] = aux[i] & kmask;
            aux32[1] = (ka3*aux[i]) & kmask;
            v1[i] = -126 + a8[0] + a8[1] + a8[2] + a8[3];
            v2[i] = -126 + a8[4] + a8[5] + a8[6] + a8[7];
            // Much slower:
            //v1[i] = -126 + a8[0][0] + a8[0][1] + a8[0][2] + a8[0][3];
            //v2[i] = -126 + a8[1][0] + a8[1][1] + a8[1][2] + a8[1][3];
        }
    }
};

struct Trellis {
    constexpr constant static uint32_t kmask1 = 0x8fff8fff;
    constexpr constant static uint32_t kmask2 = 0x3b603b60;
    constexpr constant static uint32_t ka  = 89226354;
    constexpr constant static uint32_t kb  = 64248484;
    constexpr constant static uint32_t ka1 = ka*ka;
    constexpr constant static uint32_t kb1 = kb*ka+kb;
    constexpr constant static uint32_t ka2 = ka1*ka;
    constexpr constant static uint32_t kb2 = kb1*ka+kb;
    constexpr constant static uint32_t ka3 = ka2*ka;
    constexpr constant static uint32_t kb3 = kb2*ka+kb;
    constexpr constant static uint32_t ka4 = ka3*ka;
    constexpr constant static uint32_t kb4 = kb3*ka+kb;
    constexpr constant static uint32_t ka5 = ka4*ka;
    constexpr constant static uint32_t kb5 = kb4*ka+kb;
    constexpr constant static uint32_t ka6 = ka5*ka;
    constexpr constant static uint32_t kb6 = kb5*ka+kb;
    constexpr constant static uint32_t ka7 = ka6*ka;
    constexpr constant static uint32_t kb7 = kb6*ka+kb;

    static inline half4 gen4(uint32_t val) {
        thread uint32_t aux[4] = {((ka *val + kb ) & kmask1) ^ kmask2,
                                  ((ka1*val + kb1) & kmask1) ^ kmask2,
                                  ((ka2*val + kb2) & kmask1) ^ kmask2,
                                  ((ka3*val + kb3) & kmask1) ^ kmask2};
        const thread half * h = (const thread half *)aux;
        return { h[0]+h[1], h[2]+h[3], h[4]+h[5], h[6]+h[7] };
    }
    template <typename T4>
    static inline void gen8(uint32_t val, thread T4& v1, thread T4& v2) {
        thread uint32_t aux[8] = {((ka *val + kb ) & kmask1) ^ kmask2,
                                  ((ka1*val + kb1) & kmask1) ^ kmask2,
                                  ((ka2*val + kb2) & kmask1) ^ kmask2,
                                  ((ka3*val + kb3) & kmask1) ^ kmask2,
                                  ((ka4*val + kb4) & kmask1) ^ kmask2,
                                  ((ka5*val + kb5) & kmask1) ^ kmask2,
                                  ((ka6*val + kb6) & kmask1) ^ kmask2,
                                  ((ka7*val + kb7) & kmask1) ^ kmask2};
        const thread half * h = (const thread half *)aux;
        if constexpr (is_same_v<T4, half4>) {
            v1 = { h[0]+h[1], h[2]+h[3], h[4]+h[5], h[6]+h[7] };
            v2 = { h[8]+h[9], h[10]+h[11], h[12]+h[13], h[14]+h[15] };
        } else {
            v1 = { (float)(h[0]+h[1]), (float)(h[ 2]+h[ 3]), (float)(h[ 4]+h[ 5]), (float)(h[ 6]+h[ 7]) };
            v2 = { (float)(h[8]+h[9]), (float)(h[10]+h[11]), (float)(h[12]+h[13]), (float)(h[14]+h[15]) };
        }
    }
};

void kernel_mul_mv_iq2_kt_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const uint row_size = sizeof(float) + nb*sizeof(block_iq2_kt);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(ne01) + (i13/r3)*(ne01*ne02);

    device const char  * cx = (device const char  *) src0 + (first_row + offset0)*row_size;
    device const float *  y = (device const float *) src1 + r1*ne10 + im*ne00*ne1;

    float4  sumf={0.f};

    const int ix = tiisg/16;  // 0...1
    const int it = tiisg%16;  // 0...15

    device const float4 * y4 = (device const float4 *)y + ix * (QK_K/4) + 4 * it;

    float4 v1, v2;

    float drow[N_DST];
    for (int row = 0; row < N_DST; ++row) {
        device const float * dptr = (device const float *)(cx + row*row_size);
        drow[row] = dptr[0] * 1.05f;
    }

    device const block_iq2_kt * x = (device const block_iq2_kt *)(cx + sizeof(float));

    for (int ib = ix; ib < nb; ib += 2) {

        device const uint8_t * sc  = (device const uint8_t *)x[ib].scales;

        for (int row = 0; row < N_DST; row++) {

            device const uint16_t * q2 = (device const uint16_t *)(sc + 4);

            const float ls = drow[row] * iq4k_values[(sc[(it/2)%4] >> 4*(it/8)) & 0xf];

            Trellis3::gen8(q2[2*it+0]+4096, v1, v2);
            auto sum = v1*y4[0] + v2*y4[1];

            Trellis3::gen8(q2[2*it+1]+4096, v1, v2);
            sum += v1*y4[2] + v2*y4[3];

            sum *= ls;

            sumf[row] += sum[0] + sum[1] + sum[2] + sum[3];

            sc  += row_size;

        }

        y4 += QK_K/2;
    }

    sumf = simd_sum(sumf);
    if (tiisg < 4) {
        dst[r1*ne0 + im*ne0*ne1 + first_row + tiisg] = sumf[tiisg];
    }

}

[[host_name("kernel_mul_mv_iq2_kt_f32")]]
kernel void kernel_mul_mv_iq2_kt_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_kt_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq3_kt_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const uint row_size = sizeof(float) + nb*sizeof(block_iq3_kt);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(ne01) + (i13/r3)*(ne01*ne02);

    device const char  * cx = (device const char  *) src0 + (first_row + offset0)*row_size;
    device const float *  y = (device const float *) src1 + r1*ne10 + im*ne00*ne1;

    float4  sumf={0.f};

    const int ix = tiisg/16;  // 0...1
    const int it = tiisg%16;  // 0...15

    device const float4 * y4 = (device const float4 *)y + ix * (QK_K/4) + 4 * it;

    float4 v[2];
    thread uint32_t * u32 = (thread uint32_t *)v;

    float drow[N_DST];
    for (int row = 0; row < N_DST; ++row) {
        device const float * dptr = (device const float *)(cx + row*row_size);
        drow[row] = dptr[0] * 1.01f;
    }

    device const block_iq3_kt * x = (device const block_iq3_kt *)(cx + sizeof(float));

    for (int ib = ix; ib < nb; ib += 2) {

        device const uint8_t * sc  = (device const uint8_t *)x[ib].scales;

        for (int row = 0; row < N_DST; row++) {

            device const uint16_t * q2 = (device const uint16_t *)(sc + 4);
            device const uint8_t  * qh = (device const uint8_t  *)(q2 + QK_K/8) + 16*(it%2);

            const float ls = drow[row] * ((sc[(it/2)%4] >> 4*(it/8)) & 0xf);
            const uint8_t mask = 1 << (it/2);

            Trellis3::gen8(q2[2*it+0]+4096, v[0], v[1]);
            for (int j = 0; j < 8; ++j) {
                u32[j] &= 0x7fffffff;
                u32[j] |= qh[j+0] & mask ? 0x80000000 : 0;
            }

            auto sum = v[0]*y4[0] + v[1]*y4[1];

            Trellis3::gen8(q2[2*it+1]+4096, v[0], v[1]);
            for (int j = 0; j < 8; ++j) {
                u32[j] &= 0x7fffffff;
                u32[j] |= qh[j+8] & mask ? 0x80000000 : 0;
            }

            sum += v[0]*y4[2] + v[1]*y4[3];

            sum *= ls;

            sumf[row] += sum[0] + sum[1] + sum[2] + sum[3];

            sc  += row_size;

        }

        y4 += QK_K/2;
    }

    sumf = simd_sum(sumf);
    if (tiisg < 4) {
        dst[r1*ne0 + im*ne0*ne1 + first_row + tiisg] = sumf[tiisg];
    }

}

[[host_name("kernel_mul_mv_iq3_kt_f32")]]
kernel void kernel_mul_mv_iq3_kt_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_kt_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

//TODO
void kernel_mul_mv_iq4_kt_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const uint row_size = 2*sizeof(float) + nb*sizeof(block_iq4_kt);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(ne01) + (i13/r3)*(ne01*ne02);

    device const char  * cx = (device const char  *) src0 + (first_row + offset0)*row_size;
    device const float *  y = (device const float *) src1 + r1*ne10 + im*ne00*ne1;

    float4  sumf={0.f};

    const int ix = tiisg/16;  // 0...1
    const int it = tiisg%16;  // 0...15

    device const float4 * y4 = (device const float4 *)y + ix * (QK_K/4) + 4 * it;

    float4 v[2];
    thread uint32_t * u32 = (thread uint32_t *)v;

    //float drow[2*N_DST];
    //for (int row = 0; row < N_DST; ++row) {
    //    device const float * dptr = (device const float *)(cx + row*row_size);
    //    drow[2*row+0] = dptr[0] * 31.75f * 1.01f;
    //    drow[2*row+1] = dptr[1];
    //}
    float drow[N_DST];
    for (int row = 0; row < N_DST; ++row) {
        device const float * dptr = (device const float *)(cx + row*row_size);
        drow[row] = dptr[0] * 31.75f * 1.01f;
    }

    device const block_iq4_kt * x = (device const block_iq4_kt *)(cx + 2*sizeof(float));

    for (int ib = ix; ib < nb; ib += 2) {

        //auto sumy = y4[0] + y4[1] + y4[2] + y4[3];

        device const uint32_t * shb  = x[ib].qs;

        for (int row = 0; row < N_DST; row++) {

            device const uint8_t * ql = (device const uint8_t *)(shb + 8);
            device const uint8_t * qh = ql + 64;

            const float ls = drow[row] * (((shb[it/2] & 0xff) >> 1) - 64);

            const int jj = 8*(it/2) + 4*(it%2);
            ql += jj;
            qh += jj%32;

            const uint32_t offset = 4096 + ((shb[it/2] & 1) << 15);
            const int shift = 8 - 4*(jj/32);
            uint32_t sh = (shb[it/2] >> (8 + 12*(it%2))) << 12;

            float4 sum = {0.f};
            for (int j = 0; j < 4; ++j) {
                uint32_t idx = ql[j] + ((qh[j] << shift) & 0xf00) + ((sh >> 3*j) & 0x7000) + offset; 
                auto v = Trellis::gen4(idx);
                sum += y4[j] * (float4)v;
            }
            sum *= ls;

            //sumf[row] += sum[0] + sum[1] + sum[2] + sum[3] + drow[2*row+1]*(sumy[0] + sumy[1] + sumy[2] + sumy[3]);
            sumf[row] += sum[0] + sum[1] + sum[2] + sum[3];

            shb += row_size/4;

        }

        y4 += QK_K/2;
    }

    sumf = simd_sum(sumf);
    if (tiisg < 4) {
        dst[r1*ne0 + im*ne0*ne1 + first_row + tiisg] = sumf[tiisg];
    }

}

[[host_name("kernel_mul_mv_iq4_kt_f32")]]
kernel void kernel_mul_mv_iq4_kt_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_kt_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}


[[host_name("kernel_mul_mv_iq2_k_f32")]]
kernel void kernel_mul_mv_iq2_k_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_k_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq2_ks_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const uint row_size = 2 + nb*sizeof(block_iq2_ks);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(ne01) + (i13/r3)*(ne01*ne02);

    device const char  * cx = (device const char  *) src0 + (first_row + offset0)*row_size;
    device const float *  y = (device const float *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f};

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    threadgroup float * all_values = (threadgroup float *)shared_values + 32*sgitg;
    {
        //int row = tiisg%N_DST;
        //device const half * dptr = (device const half *)(cx + row*row_size);
        //const float d = *dptr;
        //all_values[8*row + tiisg/N_DST] = d*iq2nl_values[tiisg/N_DST];
        //threadgroup_barrier(mem_flags::mem_threadgroup);
        int row = tiisg/8;
        int pos = tiisg%8;
        device const half * dptr = (device const half *)(cx + row*row_size);
        const float d = *dptr;
        all_values[8*row + pos] = d*kvalues_iq2k_f[pos];
        simdgroup_barrier(mem_flags::mem_none);
        //threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    cx += sizeof(half);

    uint32_t q32[2];
    uint32_t aux32[2];
    thread const uint8_t * aux8 = (thread const uint8_t *)aux32;

    for (int ib = ix; ib < nb; ib += 4) {

        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0];
            yl[i+ 8] = y4[i+32];
            yl[i+16] = y4[i+64];
            yl[i+24] = y4[i+96];
        }

        device const block_iq2_ks * x = (device const block_iq2_ks *)cx + ib;
        device const uint16_t * q16 = (device const uint16_t *)x->qs + 16*iq + 4*ir;
        device const uint16_t * sc  = (device const uint16_t *)x->scales;
        device const uint16_t * ex  = (device const uint16_t *)&x->extra;

        for (int row = 0; row < N_DST; row++) {

            threadgroup const float * row_values = all_values + 8*row;

            uint32_t sc32 = (sc[iq] | (sc[iq] << 12)) & 0x0f0f0f0f;
            thread const int8_t * s8 = (thread const int8_t *)&sc32;

            q32[0] = q16[0] | (q16[1] << 16);
            q32[1] = q16[2] | (q16[3] << 16);

            uint8_t extra = ex[0] << 4*(1-iq);

            float4 acc = {0.f};
            for (int l = 0; l < 4; ++l) {
                threadgroup const float * values = row_values + ((extra >> (2 + l)) & 4);
                aux32[0] = (q32[0] >> 2*l) & 0x03030303;
                aux32[1] = (q32[1] >> 2*l) & 0x03030303;
                for (int j = 0; j < 8; ++j) acc[l] += yl[8*l+j] * values[aux8[j]];
            }
            extra = ex[0] >> (8 + 4*iq);
            sumf[row] += acc[0] * (s8[0] - (extra & 1 ? 0 : 16)) + acc[1] * (s8[2] - (extra & 2 ? 0 : 16))
                       + acc[2] * (s8[1] - (extra & 4 ? 0 : 16)) + acc[3] * (s8[3] - (extra & 8 ? 0 : 16));

            q16 += row_size/2;
            sc  += row_size/2;
            ex  += row_size/2;

        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; row += 2) {
        float2 tmp = {sumf[row], sumf[row+1]};
        tmp = simd_sum(tmp);
        if (tiisg < 2) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row + tiisg] = tmp[tiisg];
        }
    }
}

[[host_name("kernel_mul_mv_iq2_ks_f32")]]
kernel void kernel_mul_mv_iq2_ks_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_ks_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq2_kl_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const uint row_size = 2 + nb*sizeof(block_iq2_kl);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(ne01) + (i13/r3)*(ne01*ne02);

    device const char  * cx0 = (device const char  *) src0 + (first_row + offset0)*row_size;
    device const float *  y  = (device const float *) src1 + r1*ne10 + im*ne00*ne1;

    float yl[32];
    float sumf[N_DST]={0.f};
    float drow[N_DST];

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/2;     // 0...3
    const int ir = it%2;     // 0 or 1

    device const float * y4 = y + ix * QK_K + 64 * iq + 16 * ir;

    uint16_t aux16[2];
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux16;

    device const char * cx = cx0;
    for (int row = 0; row < N_DST; row++) {
        device const half * dptr = (device const half *)cx;
        drow[row] = dptr[0];
        cx += row_size;
    }

    cx0 += sizeof(half);

    for (int ib = ix; ib < nb; ib += 4) {

        for (int i = 0; i < 16; ++i) {
            yl[i+ 0] = y4[i+ 0];
            yl[i+16] = y4[i+32];
        }

        //device const block_iq2_kl * x = (device const block_iq2_kl *)cx + ib;
        //device const uint16_t * ql = (device const uint16_t *)x->qs + 8*iq + 4*ir;
        //device const uint16_t * qh = (device const uint16_t *)x->qh + 4*ir;
        //device const uint8_t  * sl = x->scales_l;
        //device const uint16_t * sh = &x->scales_h;

        device const char * cx = cx0;

        for (int row = 0; row < N_DST; row++) {

            device const block_iq2_kl * x = (device const block_iq2_kl *)cx + ib;

            //threadgroup const float * row_values = all_values + 8*row;

            int8_t ls1 = int8_t(((x->scales_l[(2*iq+0)%4] >> 4*((2*iq+0)/4)) & 0xf) | (((x->scales_h >> (4*iq+0)) & 0x03) << 4)) - 32;
            int8_t ls2 = int8_t(((x->scales_l[(2*iq+1)%4] >> 4*((2*iq+1)/4)) & 0xf) | (((x->scales_h >> (4*iq+2)) & 0x03) << 4)) - 32;

            device const uint16_t * ql = (device const uint16_t *)x->qs + 8*iq + 4*ir;
            device const uint16_t * qh = (device const uint16_t *)x->qh + 4*ir;

            float2 acc = {0.f};
            for (int l = 0; l < 4; ++l) {
                uint16_t h = qh[l] >> 2*iq;
                aux16[0] = ((ql[l] >> 0) & 0x0f0f) | ((h & 0x0101) << 4);
                aux16[1] = ((ql[l] >> 4) & 0x0f0f) | ((h & 0x0202) << 3);
                for (int j = 0; j < 2; ++j) {
                    constant const int8_t * val1 = (constant const int8_t *)(iq2kl_values + aux8[j+0]);
                    constant const int8_t * val2 = (constant const int8_t *)(iq2kl_values + aux8[j+2]);
                    acc[0] += yl[4*l+2*j+ 0] * val1[0] + yl[4*l+2*j+ 1] * val1[1];
                    acc[1] += yl[4*l+2*j+16] * val2[0] + yl[4*l+2*j+17] * val2[1];
                }

            }
            sumf[row] += drow[row] * (acc[0] * ls1 + acc[1] * ls2);

            cx += row_size;

            //ql += row_size/2;
            //qh += row_size/2;
            //sl += row_size;
            //sh += row_size/2;

        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; row += 2) {
        float2 tmp = {sumf[row], sumf[row+1]};
        tmp = simd_sum(tmp);
        if (tiisg < 2) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row + tiisg] = tmp[tiisg];
        }
    }
}

[[host_name("kernel_mul_mv_iq2_kl_f32")]]
kernel void kernel_mul_mv_iq2_kl_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_kl_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq3_k_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    device const block_iq3_k * x = (device const block_iq3_k *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    threadgroup float * all_values = (threadgroup float *)shared_values + 16*sgitg;
    {
        if (tiisg < 16) all_values[tiisg] = kvalues_iq3k_f[tiisg];
        simdgroup_barrier(mem_flags::mem_none);
    }

    float yl[32];
    float sumf[N_DST]={0.f};

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3
    const int is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    uint32_t vl[2], vh[2];
    uint32_t aux32[2];
    thread const uint8_t * aux8 = (thread const uint8_t *)aux32;

    for (int ib = ix; ib < nb; ib += 4) {

        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0];
            yl[i+ 8] = y4[i+32];
            yl[i+16] = y4[i+64];
            yl[i+24] = y4[i+96];
        }

        for (int row = 0; row < N_DST; row++) {

            device const block_iq3_k & xb = x[row*nb + ib];
            device const uint16_t * ql16 = (device const uint16_t *)xb.qs + 16*iq + 4*ir;
            device const uint16_t * qh16 = (device const uint16_t *)xb.qh + 4*ir;
            device const uint16_t * sc16 = (device const uint16_t *)xb.scales_l;

            uint32_t scales32 = sc16[2*iq+0] | (sc16[2*iq+1] << 16);
            scales32 = ((scales32 >> 4*is) & 0x0f0f0f0f) << 1;
            thread const int8_t * s8 = (thread const int8_t *)&scales32;
            uint16_t extra = (xb.extra >> (8*iq + is)) << 3;
            uint16_t signs = xb.scales_h >> (8*iq + is);

            vl[0] = ql16[0] | ql16[1] << 16;
            vl[1] = ql16[2] | ql16[3] << 16;
            vh[0] = ((qh16[0] | (qh16[1] << 16)) << 4*(1-iq)) >> 2;
            vh[1] = ((qh16[2] | (qh16[3] << 16)) << 4*(1-iq)) >> 2;

            float4 acc = {0.f};
            for (int l = 0; l < 4; ++l) {
                threadgroup const float * values = all_values + (extra & 8);
                //constant float * values = kvalues_iq3k_f + (extra & 8);
                aux32[0] = (vl[0] & 0x03030303) | (vh[0] & 0x04040404);
                aux32[1] = (vl[1] & 0x03030303) | (vh[1] & 0x04040404);
                for (int j = 0; j < 8; ++j) acc[l] += yl[8*l+j] * values[aux8[j]];
                vl[0] >>= 2; vl[1] >>= 2;
                vh[0] >>= 1; vh[1] >>= 1;
                extra >>= 2;
            }

            sumf[row] += (float)xb.d * (acc[0] * (signs & 0x01 ? -s8[0] : s8[0]) + acc[1] * (signs & 0x04 ? -s8[1] : s8[1]) +
                                        acc[2] * (signs & 0x10 ? -s8[2] : s8[2]) + acc[3] * (signs & 0x40 ? -s8[3] : s8[3]));

        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; row += 2) {
        float2 tmp{sumf[row], sumf[row+1]};
        tmp = simd_sum(tmp);
        if (tiisg < 2) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row + tiisg] = tmp[tiisg];
        }
    }
}

[[host_name("kernel_mul_mv_iq3_k_f32")]]
kernel void kernel_mul_mv_iq3_k_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_k_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

// TODO
void kernel_mul_mv_iq3_ks_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    const uint row_size = sizeof(half) + nb*sizeof(block_iq3_ks);
    device const char  * cx = (device const char  *)src0 + (first_row + offset0)*row_size;
    device const float * y  = (device const float *)src1 + r1*ne10 + im*ne00*ne1;

    threadgroup float * all_values = (threadgroup float *)shared_values + 16*sgitg;
    {
        if (tiisg < 16) all_values[tiisg] = kvalues_iq3k_f[tiisg];
        simdgroup_barrier(mem_flags::mem_none);
    }

    float yl[32];
    float sumf[N_DST]={0.f};
    float d[N_DST];

    const int ix = tiisg/8;  // 0...3
    const int it = tiisg%8;  // 0...7
    const int iq = it/4;     // 0 or 1
    const int ir = it%4;     // 0...3

    device const half * dptr = (device const half *)cx;
    d[0] = (float)dptr[0];
    for (int i = 1; i < N_DST; ++i) {
        dptr += row_size/2;
        d[i] = (float)dptr[0];
    }

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    uint32_t vl[2], vh[2];
    uint32_t aux32[2];
    thread const uint8_t * aux8 = (thread const uint8_t *)aux32;

    for (int ib = ix; ib < nb; ib += 4) {

        for (int i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0];
            yl[i+ 8] = y4[i+32];
            yl[i+16] = y4[i+64];
            yl[i+24] = y4[i+96];
        }

        for (int row = 0; row < N_DST; row++) {

            device const block_iq3_ks * x = (device const block_iq3_ks *)(cx + row_size*row + sizeof(half));
            device const block_iq3_ks & xb = x[ib];
            device const uint16_t * ql16 = (device const uint16_t *)xb.qs + 16*iq + 4*ir;
            device const uint16_t * qh16 = (device const uint16_t *)xb.qh + 4*ir;
            device const uint16_t * sc16 = (device const uint16_t *)xb.scales;

            uint8_t extra_s = (xb.extra & 0xff) >> 4*iq;
            uint8_t extra_v = xb.extra >> (8 + 4*iq);

            uint32_t scales32 = sc16[0] | (sc16[1] << 16);
            scales32 = (scales32 >> 4*iq) & 0x0f0f0f0f;
            thread int8_t * s8 = (thread int8_t *)&scales32;
            s8[0] += ((extra_s << 4) & 0x10) - 16;
            s8[1] += ((extra_s << 3) & 0x10) - 16;
            s8[2] += ((extra_s << 2) & 0x10) - 16;
            s8[3] += ((extra_s << 1) & 0x10) - 16;

            vl[0] = ql16[0] | ql16[1] << 16;
            vl[1] = ql16[2] | ql16[3] << 16;
            vh[0] = ((qh16[0] | (qh16[1] << 16)) << 4*(1-iq)) >> 2;
            vh[1] = ((qh16[2] | (qh16[3] << 16)) << 4*(1-iq)) >> 2;

            float4 acc = {0.f};
            for (int l = 0; l < 4; ++l) {
                threadgroup const float * values = all_values + ((extra_v & 1) << 3);
                aux32[0] = (vl[0] & 0x03030303) | (vh[0] & 0x04040404);
                aux32[1] = (vl[1] & 0x03030303) | (vh[1] & 0x04040404);
                for (int j = 0; j < 8; ++j) acc[l] += yl[8*l+j] * values[aux8[j]];
                vl[0] >>= 2; vl[1] >>= 2;
                vh[0] >>= 1; vh[1] >>= 1;
                extra_v >>= 1;
            }

            sumf[row] += d[row] * (acc[0] * s8[0] + acc[1] * s8[1] + acc[2] * s8[2] + acc[3] * s8[3]);

        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < N_DST; row += 2) {
        float2 tmp{sumf[row], sumf[row+1]};
        tmp = simd_sum(tmp);
        if (tiisg < 2) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row + tiisg] = tmp[tiisg];
        }
    }
}

[[host_name("kernel_mul_mv_iq3_ks_f32")]]
kernel void kernel_mul_mv_iq3_ks_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_ks_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

void kernel_mul_mv_iq4_k_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values_i8,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    threadgroup float * shared_values = (threadgroup float *)shared_values_i8;
    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq4_k * x = (device const block_iq4_k *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    const int ix = tiisg/16;  // 0 or 1
    const int it = tiisg%16;  // 0...15
    const int ib = it/2;
    const int il = it%2;

    shared_values[tiisg] = kvalues_iq4k_f[tiisg];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2]={0.f}, all_sum;

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];
        //float2 sumy;
        //sumy[0] = -4.f*(yl[0][0] + yl[0][1] + yl[0][2] + yl[0][3] + yl[2][0] + yl[2][1] + yl[2][2] + yl[2][3]);
        //sumy[1] = -4.f*(yl[1][0] + yl[1][1] + yl[1][2] + yl[1][3] + yl[3][0] + yl[3][1] + yl[3][2] + yl[3][3]);

        for (int row = 0; row < 2; ++row) {

            device const block_iq4_k & xb = x[row*nb + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            uint16_t extra = xb.extra >> 2*ib;
            threadgroup const float * values1 = shared_values + 16*(extra & 1);
            threadgroup const float * values2 = shared_values +  8*(extra & 2);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {values1[q8[0]], values1[q8[1]], values1[q8[2]], values1[q8[3]]};
            qf2 = {values2[q8[4]], values2[q8[5]], values2[q8[6]], values2[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[1] & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {values1[q8[0]], values1[q8[1]], values1[q8[2]], values1[q8[3]]};
            qf2 = {values2[q8[4]], values2[q8[5]], values2[q8[6]], values2[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            const uint8_t h = xb.scales_h[ib/2] >> 4*(ib%2);
            const int ls1 = ((xb.scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
            const int ls2 = ((xb.scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;
            sumf[row] += (float)xb.d * (ls1 * (acc1[0] + acc1[1] + acc1[2] + acc1[3]) + ls2 * (acc2[0] + acc2[1] + acc2[2] + acc2[3]));
            //uint16_t extra = xb.extra >> 2*ib;
            //sumf[row] += (float)xb.d * (ls1 * (acc1[0] + acc1[1] + acc1[2] + acc1[3] + (extra & 1 ? sumy[0] : 0)) +
            //                            ls2 * (acc2[0] + acc2[1] + acc2[2] + acc2[3] + (extra & 2 ? sumy[1] : 0)));

        }

        yb += 2 * QK_K;
    }

    for (int row = 0; row < 2; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

void kernel_mul_mv_iq5_k_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values_i8,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    threadgroup float * shared_values = (threadgroup float *)shared_values_i8;
    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq5_k * x = (device const block_iq5_k *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    const int ix = tiisg/16;  // 0 or 1
    const int it = tiisg%16;  // 0...15
    const int ib64 = it/4;
    const int il64 = it%4;

    shared_values[2*tiisg+0] = kvalues_iq5k_f[2*tiisg+0];
    shared_values[2*tiisg+1] = kvalues_iq5k_f[2*tiisg+1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2]={0.f}, all_sum;

    device const float * yb = y + ix * QK_K + ib64 * 64 + il64 * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[8]; yl[2] = y4[1]; yl[3] = y4[9];

        for (int row = 0; row < 2; ++row) {

            device const block_iq5_k & xb = x[row*nb + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 32*ib64 + 8*il64);
            device const uint32_t * qh = (device const uint32_t *)(xb.qh + 8*il64);

            uint16_t extra = xb.extra >> (4*ib64 + il64/2);
            threadgroup const float * values1 = shared_values + 32*(extra & 1);
            threadgroup const float * values2 = shared_values +  8*(extra & 4);

            float4 acc1 = {0.f}, acc2 = {0.f};

            uint32_t h = qh[0] >> 2*ib64;
            aux32[0] = ((q4[0] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x10101010);
            aux32[1] = ((q4[0] >> 4) & 0x0f0f0f0f) | ((h << 3) & 0x10101010);
            qf1 = {values1[q8[0]], values1[q8[1]], values1[q8[2]], values1[q8[3]]};
            qf2 = {values2[q8[4]], values2[q8[5]], values2[q8[6]], values2[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            h = qh[1] >> 2*ib64;
            aux32[0] = ((q4[1] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x10101010);
            aux32[1] = ((q4[1] >> 4) & 0x0f0f0f0f) | ((h << 3) & 0x10101010);
            qf1 = {values1[q8[0]], values1[q8[1]], values1[q8[2]], values1[q8[3]]};
            qf2 = {values2[q8[4]], values2[q8[5]], values2[q8[6]], values2[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            const uint8_t sh = xb.scales_h[ib64] >> 2*(il64/2);
            const int ls1 = (((xb.scales_l[2*ib64 + 0 + il64/2] >> 4*(il64/2)) & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int ls2 = (((xb.scales_l[2*ib64 + 1 + il64/2] >> 4*(il64/2)) & 0xf) | ((sh << 0) & 0x30)) - 32;
            sumf[row] += (float)xb.d * (ls1 * (acc1[0] + acc1[1] + acc1[2] + acc1[3]) + ls2 * (acc2[0] + acc2[1] + acc2[2] + acc2[3]));

        }

        yb += 2 * QK_K;
    }

    for (int row = 0; row < 2; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

void kernel_mul_mv_iq6_k_f32_impl(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values_i8,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg) {

    threadgroup float * shared_values = (threadgroup float *)shared_values_i8;
    const int nb = ne00/QK_K;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * 2 + sgitg) * 2;
    const int ib_row = first_row * nb;

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    const uint offset0 = (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    device const block_iq6_k * x = (device const block_iq6_k *) src0 + ib_row + offset0;
    device const float       * y = (device const float       *) src1 + r1*ne10 + im*ne00*ne1;

    const int ix = tiisg/16;  // 0 or 1
    const int it = tiisg%16;  // 0...15
    const int ib64 = it/4;    // 0...3
    const int il64 = it%4;    // 0...3

    shared_values[4*tiisg+0] = kvalues_iq6k_f[4*tiisg+0];
    shared_values[4*tiisg+1] = kvalues_iq6k_f[4*tiisg+1];
    shared_values[4*tiisg+2] = kvalues_iq6k_f[4*tiisg+2];
    shared_values[4*tiisg+3] = kvalues_iq6k_f[4*tiisg+3];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[2]={0.f}, all_sum;

    device const float * yb = y + ix * QK_K + ib64 * 64 + il64 * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    for (int ibl = ix; ibl < nb; ibl += 2) {

        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[8]; yl[2] = y4[1]; yl[3] = y4[9];

        for (int row = 0; row < 2; ++row) {

            device const block_iq6_k & xb = x[row*nb + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 32*ib64 + 8*il64);
            device const uint32_t * qh = (device const uint32_t *)(xb.qh + 32*(ib64/2) + 8*il64);

            uint16_t extra = xb.extra >> (4*ib64 + il64/2);
            threadgroup const float * values1 = shared_values + 64*(extra & 1);
            threadgroup const float * values2 = shared_values + 16*(extra & 4);

            float4 acc1 = {0.f}, acc2 = {0.f};

            uint32_t h = qh[0] >> 4*(ib64%2);
            aux32[0] = ((q4[0] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x30303030);
            aux32[1] = ((q4[0] >> 4) & 0x0f0f0f0f) | ((h << 2) & 0x30303030);
            qf1 = {values1[q8[0]], values1[q8[1]], values1[q8[2]], values1[q8[3]]};
            qf2 = {values2[q8[4]], values2[q8[5]], values2[q8[6]], values2[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            h = qh[1] >> 4*(ib64%2);
            aux32[0] = ((q4[1] >> 0) & 0x0f0f0f0f) | ((h << 4) & 0x30303030);
            aux32[1] = ((q4[1] >> 4) & 0x0f0f0f0f) | ((h << 2) & 0x30303030);
            qf1 = {values1[q8[0]], values1[q8[1]], values1[q8[2]], values1[q8[3]]};
            qf2 = {values2[q8[4]], values2[q8[5]], values2[q8[6]], values2[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            const int ls1 = xb.scales[4*ib64 + 0 + il64/2];
            const int ls2 = xb.scales[4*ib64 + 2 + il64/2];
            sumf[row] += (float)xb.d * (ls1 * (acc1[0] + acc1[1] + acc1[2] + acc1[3]) + ls2 * (acc2[0] + acc2[1] + acc2[2] + acc2[3]));

        }

        yb += 2 * QK_K;
    }

    for (int row = 0; row < 2; ++row) {
        all_sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = all_sum;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_bn_f32")]]
kernel void kernel_mul_mv_iq1_bn_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_bn_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_bn_f32")]]
kernel void kernel_mul_mv_iq2_bn_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_bn_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_ks_f32")]]
kernel void kernel_mul_mv_iq4_ks_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_ks_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq5_ks_f32")]]
kernel void kernel_mul_mv_iq5_ks_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq5_ks_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_kss_f32")]]
kernel void kernel_mul_mv_iq4_kss_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_kss_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_k_f32")]]
kernel void kernel_mul_mv_iq4_k_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_k_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq5_k_f32")]]
kernel void kernel_mul_mv_iq5_k_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq5_k_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq6_k_f32")]]
kernel void kernel_mul_mv_iq6_k_f32(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant   int64_t & ne01,
        constant   int64_t & ne02,
        constant  uint64_t & nb00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant   int64_t & ne11,
        constant   int64_t & ne12,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb12,
        constant   int64_t & ne0,
        constant   int64_t & ne1,
        constant   uint    & r2,
        constant   uint    & r3,
        threadgroup int8_t * shared_values [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint tiisg[[thread_index_in_simdgroup]],
        uint sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq6_k_f32_impl(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3, shared_values, tgpig, tiisg, sgitg);
}

//============================= templates and their specializations =============================

// NOTE: this is not dequantizing - we are simply fitting the template

template <typename type4x4>
void dequantize_bf16(device const half4x4 * src, short il, thread type4x4 & reg) {
    device const uint16_t * src_u16 = (device const uint16_t *)src;
    typedef union { uint32_t u; float f; } aux_t;
    aux_t aux;
    for (int i = 0; i < 16; ++i) {
        aux.u = (uint32_t)src_u16[i] << 16;
        reg[i/4][i%4] = aux.f;
    }
}

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i=0;i<8;i++) {
        reg[i/2][2*(i%2)+0] = d1 * (qs[i] & mask0) + md;
        reg[i/2][2*(i%2)+1] = d2 * (qs[i] & mask1) + md;
    }
}

template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i=0;i<8;i++) {
        reg[i/2][2*(i%2)+0] = ((qs[i] & mask0) * d1) + m;
        reg[i/2][2*(i%2)+1] = ((qs[i] & mask1) * d2) + m;
    }
}

template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[i/2][2*(i%2)+0] = d * x0 + md;
        reg[i/2][2*(i%2)+1] = d * x1 + md;
    }
}

template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 *xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[i/2][2*(i%2)+0] = d * x0 + m;
        reg[i/2][2*(i%2)+1] = d * x1 + m;
    }
}

template <typename type4x4>
void dequantize_q6_0(device const block_q6_0 *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float m = -32.h * xb->d;
    device const uint8_t * qh = xb->qh;
    device const uint8_t * qs = qh + 8;

    for (int i = 0; i < 8; i++) {
        reg[i/4][i%4] = d * (((qs[i] >> 4*il) & 0xf) | (((qh[i] >> 2*il) << 4) & 0x30)) + m;
    }
    for (int i = 0; i < 8; i++) {
        reg[2+i/4][i%4] = d * (((qs[i+8] >> 4*il) & 0xf) | ((qh[i] >> 2*il) & 0x30)) + m;
    }
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K * xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    q = q + 32*(il/8) + 16*(il&1);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;

    il = (il/2) & 3;
    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K *xb, short il, thread type4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il<2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * ql = (device const uint8_t *)xb->ql;
    device const uint8_t * qh = (device const uint8_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 64*(il/8) + 32*((il/2)&1) + 16*(il&1);
    qh = qh + 32*(il/8) + 16*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint16_t  kmask1 = il>1 ? (il>2 ? 192 : 48) : (il>0 ? 12 : 3);
    const uint16_t  kmask2 = il>1 ? 0xF0              : 0x0F;
    const float       coef = il>1 ? 1.f/16.f          : 1.f;
    const float ml = d_all * sc * 32.f;
    const float dl = d_all * sc * coef;
    for (int i = 0; i < 16; ++i) {
        const half q = il&1 ? ((ql[i] & kmask2) | ((qh[i] & kmask1) << 2))
                            : ((ql[i] & kmask2) | ((qh[i] & kmask1) << 4));
        reg[i/4][i%4] = dl * q - ml;
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    // each block of 32 needs 2 uint32_t's for the quants & scale, so 4 uint16_t's.
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq2_xs(device const block_iq2_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * q3 = xb->qs + 8*ib32;
    device const uint16_t * gas = (device const uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[1][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f);
        reg[3][i] = dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f);
    }
}

template <typename type4x4>
void dequantize_iq3_s(device const block_iq3_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 8*ib32;
    device const uint8_t * signs = xb->signs + 4*ib32 + 2*il;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]);
        reg[1][i] = dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]);
    }
    grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
    grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]);
        reg[3][i] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]);
    }
}

template <typename type4x4>
void dequantize_iq2_s(device const block_iq2_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * signs = qs + QK_K/8;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[0] | ((qh << 8) & 0x300)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[1] | ((qh << 6) & 0x300)));
    for (int i = 0; i < 8; ++i) {
        reg[i/4+0][i%4] = dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i]);
        reg[i/4+2][i%4] = dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i]);
    }
}

template <typename type4x4>
void dequantize_iq1_s(device const block_iq1_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    const float d = xb->d;
    device const uint8_t  * qs = xb->qs + 4*ib32 + 2*il;
    device const uint16_t * qh = xb->qh;
    const float dl = d * (2*((qh[ib32] >> 12) & 7) + 1);
    const float ml = dl * (qh[ib32] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);
    const uint16_t h = qh[ib32] >> 6*il;
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((h << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((h << 5) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml;
        reg[1][i] = dl * (grid1[i] >>  4) + ml;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml;
        reg[3][i] = dl * (grid2[i] >>  4) + ml;
    }
}

template <typename type4x4>
void dequantize_iq1_m(device const block_iq1_m * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * sc = (device const uint16_t *)xb->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = scale.f16;

    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * qh = xb->qh + 2*ib32 + il;

    const float dl  = d * (2*((sc[ib32/2] >> (6*(ib32%2)+3*il)) & 7) + 1);
    const float ml1 = dl * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    const float ml2 = dl * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = dl * (grid1[i] & 0xf) + ml1;
        reg[1][i] = dl * (grid1[i] >>  4) + ml1;
        reg[2][i] = dl * (grid2[i] & 0xf) + ml2;
        reg[3][i] = dl * (grid2[i] >>  4) + ml2;
    }
}


template <typename type4x4>
void dequantize_iq1_bn(half d, device const block_iq1_bn * xb, short il, thread type4x4 & reg) {
    // il is in 0...3

    constexpr uint16_t k_mult[5] = {81, 27, 9, 3, 1};
    const half k_values[3] = {-d, 0.h, d};

    for (int k = 0; k < 3; ++k) {
        uint16_t q = xb->ql[3*il + k];
        int i = 5*k + 4;
        for (int j = 4; j >= 0; --j) {
            uint16_t v = q & 0xff;
            v += v << 1;
            reg[i/4][i%4] = k_values[v >> 8];
            q += q << 1;
            --i;
        }
    }
    uint16_t v = (k_mult[il]*xb->extra) & 0xff;
    v += v << 1;
    reg[3][3] = k_values[v >> 8];
}

template <typename type4x4>
void dequantize_iq2_bn(half d, device const block_iq2_bn * xb, short il, thread type4x4 & reg) {
    // il is in 0...3
    constexpr half k_scale[4] = {1.h, 0.25h, 0.0625h, 0.015625h};
    const half db = d * k_scale[il];
    const uint32_t mask = 0x03030303 << 2*il;

    device const uint32_t * qs = (device const uint32_t *)xb->qs;
    uint32_t aux32;
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32;

    for (int j = 0; j < 4; ++j) {
        aux32 = qs[j] & mask;
        reg[j][0] = db * aux8[0] - d;
        reg[j][1] = db * aux8[1] - d;
        reg[j][2] = db * aux8[2] - d;
        reg[j][3] = db * aux8[3] - d;
    }
}

template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = d * kvalues_iq4nl_f[q8[0]];
        reg[i][1] = d * kvalues_iq4nl_f[q8[1]];
        reg[i][2] = d * kvalues_iq4nl_f[q8[2]];
        reg[i][3] = d * kvalues_iq4nl_f[q8[3]];
    }
}

template <typename type4x4>
void dequantize_iq4_ks(device const block_iq4_ks * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    const float ls = (xb->scales[ib32] & 254) - 127;
    constant float * values = kvalues_iq4k_f + ((xb->scales[ib32] & 1) << 4);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = ls * values[q8[0]];
        reg[i][1] = ls * values[q8[1]];
        reg[i][2] = ls * values[q8[2]];
        reg[i][3] = ls * values[q8[3]];
    }
}

template <typename type4x4>
void dequantize_iq5_ks(device const block_iq5_ks * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 8*(ib32/2) + 4*(il%2);
    device const uint32_t * qh = (device const uint32_t *)xb->qh + 4*(il%2);
    const float ls = (xb->scales[ib32] & 254) - 127;
    constant float * values = kvalues_iq5k_f + ((xb->scales[ib32] & 1) << 5);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[i] >> 4*(ib32%2)) & 0x0f0f0f0f) | (((qh[i] >> ib32) & 0x01010101) << 4);
        reg[i][0] = ls * values[q8[0]];
        reg[i][1] = ls * values[q8[1]];
        reg[i][2] = ls * values[q8[2]];
        reg[i][3] = ls * values[q8[3]];
    }
}

template <typename type4x4>
void dequantize_iq4_kss(device const block_iq4_kss * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
    uint8_t ls = (s32 | (s32 >> 15)) & 0xff;
    const half scale = (ls & 254) - 127;
    constant float * values = kvalues_iq4k_f + ((ls & 1) << 4);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = q4[i] & 0xfffefffe;
        aux32 ^= (aux32 >> 1);
        aux32 = (aux32 >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = scale * values[q8[0]];
        reg[i][1] = scale * values[q8[1]];
        reg[i][2] = scale * values[q8[2]];
        reg[i][3] = scale * values[q8[3]];
    }
}

template <typename type4x4>
void dequantize_iq2_kt(device const block_iq2_kt * x, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256
    int ib32 = il/2;
    half scale = iq4k_values[((x->scales[ib32%4] >> 4*(ib32/4)) & 0xf)] * 1.05h;
    device const uint16_t * q2 = (device const uint16_t *)x->ql + 4*ib32 + 2*(il%2);

    char4 v1, v2;
    for (int i = 0; i < 2; ++i) {
        Trellis3::gen8(q2[i]+4096, v1, v2);
        if constexpr (is_same_v<type4x4, half4x4>) {
            reg[2*i+0] = {scale*(half)v1[0], scale*(half)v1[1], scale*(half)v1[2], scale*(half)v1[3]};
            reg[2*i+1] = {scale*(half)v2[0], scale*(half)v2[1], scale*(half)v2[2], scale*(half)v2[3]};
        } else {
            reg[2*i+0] = {scale*(float)v1[0], scale*(float)v1[1], scale*(float)v1[2], scale*(float)v1[3]};
            reg[2*i+1] = {scale*(float)v2[0], scale*(float)v2[1], scale*(float)v2[2], scale*(float)v2[3]};
        }
    }
}

template <typename type4x4>
void dequantize_iq3_kt(device const block_iq3_kt * x, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256
    int ib32 = il/2;
    half scale = (half)((x->scales[ib32%4] >> 4*(ib32/4)) & 0xf) * 1.01h;
    device const uint16_t * q2 = (device const uint16_t *)x->ql + 4*ib32 + 2*(il%2);
    device const uint8_t  * qh = x->qh + 16*(il%2);
    const uint8_t mask = 1 << ib32;

    half4 v1, v2;
    for (int i = 0; i < 2; ++i) {
        Trellis3::gen8(q2[i]+4096, v1, v2);
        v1 = abs(v1)*scale; v2 = abs(v2)*scale;
        for (int j = 0; j < 4; ++j) reg[2*i+0][j] = qh[8*i+0+j] & mask ? -v1[j] : v1[j];
        for (int j = 0; j < 4; ++j) reg[2*i+1][j] = qh[8*i+4+j] & mask ? -v2[j] : v2[j];
    }
}

void dequantize_iq4_kt(device const block_iq4_kt * x, short il, float d, thread float4x4 & reg) {
    // il is 0...15 for QK_K = 256
    int ib32 = il/2;
    device const uint32_t * shb = x->qs;
    device const uint8_t * ql = (device const uint8_t *)(shb + 8);
    device const uint8_t * qh = ql + 64;
    const int ls = (shb[ib32] & 0xff) >> 1;
    const float scale = d * (ls - 64);
    const uint32_t offset = 4096 + ((shb[ib32] & 1) << 15);

    ql += 8*ib32;
    qh += 8*(ib32%4);

    uint32_t sh = (shb[ib32] >> (8 + 12*(il%2))) << 12;
    const int shift = 8 - 4*(ib32/4);

    for (int i = 0; i < 4; ++i) {
        uint32_t idx = ql[i] + ((qh[i] << shift) & 0xf00) + ((sh >> 3*i) & 0x7000) + offset; 
        auto c4 = Trellis3::gen4(idx);
        reg[i] = {scale*c4[0], scale*c4[1], scale*c4[2], scale*c4[3]};
    }
}

template <typename type4x4>
void dequantize_iq2_k(device const block_iq2_k * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256
    device const uint32_t * q32 = (device const uint32_t *)xb->qs + 8*(il/8) + 4*(il&1);
    half d = xb->d * (((xb->scales[il/2] >> 4*(il&1)) & 0xf) - 8);

    constant half4 * half_values = (constant half4 *)kvalues_iq2k_h;
    half4 values = half_values[(xb->extra >> il) & 1] * d;

    const int shift = 2*((il%8)/2);
    uint32_t aux32;
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q32[i] >> shift) & 0x03030303;
        for (int j = 0; j < 4; ++j) reg[i][j] = values[aux8[j]];
    }
}

template <typename type4x4>
void dequantize_iq2_ks(device const block_iq2_ks * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256
    device const uint16_t * q16 = (device const uint16_t *)xb->qs + 16*(il/8) + 8*(il&1);
    const short ib32 = il/2;
    half d = (((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf) - ((xb->extra >> (8 + ib32)) & 1 ? 0 : 16));

    constant half4 * half_values = (constant half4 *)kvalues_iq2k_h;
    half4 values = half_values[(xb->extra >> ib32) & 1] * d;

    const int shift = 2*((il%8)/2);
    thread uint16_t aux16[2];
    thread const uint8_t * aux8 = (thread const uint8_t *)aux16;
    for (int i = 0; i < 4; ++i) {
        aux16[0] = (q16[2*i+0] >> shift) & 0x0303;
        aux16[1] = (q16[2*i+1] >> shift) & 0x0303;
        for (int j = 0; j < 4; ++j) reg[i][j] = values[aux8[j]];
    }
}

template <typename type4x4>
void dequantize_iq2_kl(device const block_iq2_kl * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256
    const short ib32 = il/2;
    device const uint16_t * ql = (device const uint16_t * )xb->qs + 8*(ib32/2) + 4*(il%2);
    device const uint16_t * qh = (device const uint16_t * )xb->qh + 4*(il%2);

    half d = (int16_t(((xb->scales_l[ib32%4] >> 4*(ib32/4)) & 0xf) | (((xb->scales_h >> 2*ib32) & 0x3) << 4)) - 32);

    thread uint16_t aux16;
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux16;
    for (int i = 0; i < 4; ++i) {
        aux16 = ((ql[i] >> 4*(ib32%2)) & 0x0f0f) | (((qh[i] >> ib32) & 0x0101) << 4);
        constant const int8_t * val1 = (constant const int8_t *)(iq2kl_values + aux8[0]);
        constant const int8_t * val2 = (constant const int8_t *)(iq2kl_values + aux8[1]);
        reg[i][0] = d * val1[0];
        reg[i][1] = d * val1[1];
        reg[i][2] = d * val2[0];
        reg[i][3] = d * val2[1];
    }
}

template <typename type4x4>
void dequantize_iq3_k(device const block_iq3_k * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256
    device const uint16_t * q16l = (device const uint16_t *)xb->qs + 16*(il/8) + 8*(il&1);
    device const uint16_t * q16h = (device const uint16_t *)xb->qh + 8*(il&1);
    half d = xb->d * (2*((xb->scales_l[il/2] >> 4*(il&1)) & 0xf) + 1) * (xb->scales_h & (1 << il) ? -1 : 1);

    constant half * values = kvalues_iq3k_h + 8*((xb->extra >> il) & 1);

    const int shift = 2*((il%8)/2);
    uint32_t aux32;
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        uint32_t vl = q16l[2*i+0] | (q16l[2*i+1] << 16);
        uint32_t vh = q16h[2*i+0] | (q16h[2*i+1] << 16);
        aux32 = ((vl >> shift) & 0x03030303) | (((vh >> ((il/2)%8)) << 2) & 0x04040404);
        for (int j = 0; j < 4; ++j) reg[i][j] = d * values[aux8[j]];
    }
}

template <typename type4x4>
void dequantize_iq3_ks(device const block_iq3_ks * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256
    int ib32 = il/2;
    device const uint16_t * q16l = (device const uint16_t *)xb->qs + 16*(il/8) + 8*(il&1);
    device const uint16_t * q16h = (device const uint16_t *)xb->qh + 8*(il&1);

    int8_t ls = int8_t(((xb->scales[ib32%4] >> 4*(ib32/4)) & 0xf) | (((xb->extra >> ib32) & 1) << 4)) - 16;
    half d = ls;

    constant half * values = kvalues_iq3k_h + 8*((xb->extra >> (8+ib32)) & 1);

    const int shift = 2*((il%8)/2);
    uint32_t aux32;
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        uint32_t vl = q16l[2*i+0] | (q16l[2*i+1] << 16);
        uint32_t vh = q16h[2*i+0] | (q16h[2*i+1] << 16);
        aux32 = ((vl >> shift) & 0x03030303) | (((vh >> ((il/2)%8)) << 2) & 0x04040404);
        for (int j = 0; j < 4; ++j) reg[i][j] = d * values[aux8[j]];
    }
}

template <typename type4x4>
void dequantize_iq4_k(device const block_iq4_k * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    const int l = il%2;
    // l = 0 or 1. l = 0 processes the first 16 quants in a block of 32, l = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    const int ls = ((xb->scales_l[ib32] >> 4*l) & 0xf) | (((xb->scales_h[il/4] >> 2*(il%4)) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    constant float * values = kvalues_iq4k_f + 16*((xb->extra >> il) & 1);
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*l) & 0x0f0f0f0f;
        reg[i][0] = d * values[q8[0]];
        reg[i][1] = d * values[q8[1]];
        reg[i][2] = d * values[q8[2]];
        reg[i][3] = d * values[q8[3]];
    }
}

template <typename type4x4>
void dequantize_iq5_k(device const block_iq5_k * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    const int l = il%2;
    // l = 0 or 1. l = 0 processes the first 16 quants in a block of 32, l = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 8*(ib32/2) + 4*l;
    device const uint32_t * qh = (device const uint32_t *)xb->qh + 4*l;
    const int ls = ((xb->scales_l[ib32] >> 4*l) & 0xf) | (((xb->scales_h[il/4] >> 2*(il%4)) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    constant float * values = kvalues_iq5k_f + 32*((xb->extra >> il) & 1);
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[i] >> 4*(ib32%2)) & 0x0f0f0f0f) | (((qh[i] >> ib32) & 0x01010101) << 4);
        reg[i][0] = d * values[q8[0]];
        reg[i][1] = d * values[q8[1]];
        reg[i][2] = d * values[q8[2]];
        reg[i][3] = d * values[q8[3]];
    }
}

template <typename type4x4>
void dequantize_iq6_k(device const block_iq6_k * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    const int l = il%2;
    // l = 0 or 1. l = 0 processes the first 16 quants in a block of 32, l = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 8*(ib32/2) + 4*l;
    device const uint32_t * qh = (device const uint32_t *)xb->qh + 8*(ib32/4) + 4*l;
    const float d = (float)xb->d * xb->scales[2*ib32+l];
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    constant float * values = kvalues_iq6k_f + 64*((xb->extra >> il) & 1);
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[i] >> 4*(ib32%2)) & 0x0f0f0f0f) | (((qh[i] >> 2*(ib32%4)) & 0x03030303) << 4);
        reg[i][0] = d * values[q8[0]];
        reg[i][1] = d * values[q8[1]];
        reg[i][2] = d * values[q8[2]];
        reg[i][3] = d * values[q8[3]];
    }
}

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_get_rows_q(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int64_t ind = tiitg; ind < ne00/16; ind += tptg.x) {
        float4x4 temp;
        dequantize_func(((device const block_q *) ((const device char *) src0 + r*nb01 + i02*nb02)) + ind/nl, ind%nl, temp);
        *(((device float4x4 *) ((device char *) dst + i11*nb2 + i10*nb1)) + ind) = temp;
    }
}

template<typename Dequantizer>
kernel void kernel_get_rows_q2(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    Dequantizer deq((const device char *) src0 + r*nb01 + i02*nb02);

    for (int64_t ind = tiitg; ind < ne00/16; ind += tptg.x) {
        float4x4 temp;
        deq.convert(ind, temp);
        *(((device float4x4 *) ((device char *) dst + i11*nb2 + i10*nb1)) + ind) = temp;
    }
}

template<typename T>
kernel void kernel_get_rows_f(
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < ne00; ind += tptg.x) {
        ((      device float *) ((      device char *)  dst + i11*nb2  + i10*nb1))[ind] =
        ((const device T     *) ((const device char *) src0 + i02*nb02 +  r*nb01))[ind];
    }
}

kernel void kernel_get_rows_i32(
        device const  void * src0,
        device const  void * src1,
        device     int32_t * dst,
        constant   int64_t & ne00,
        constant  uint64_t & nb01,
        constant  uint64_t & nb02,
        constant   int64_t & ne10,
        constant  uint64_t & nb10,
        constant  uint64_t & nb11,
        constant  uint64_t & nb1,
        constant  uint64_t & nb2,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int64_t i10 = tgpig.x;
    const int64_t i11 = tgpig.y;

    const int64_t r = ((const device int32_t *) ((const device char *) src1 + i11*nb11 + i10*nb10))[0];

    const int64_t i02 = i11;

    for (int ind = tiitg; ind < ne00; ind += tptg.x) {
        ((      device int32_t *) ((      device char *) dst  + i11*nb2 + i10*nb1))[ind] =
        ((const device int32_t *) ((const device char *) src0 + i02*nb02 + r*nb01))[ind];
    }
}


#define BLOCK_SIZE_M 64 // 8 simdgroup matrices from matrix A
#define BLOCK_SIZE_N 32 // 4 simdgroup matrices from matrix B
#define BLOCK_SIZE_K 32
#define THREAD_MAT_M 4 // each thread take 4 simdgroup matrices from matrix A
#define THREAD_MAT_N 2 // each thread take 2 simdgroup matrices from matrix B
#define THREAD_PER_BLOCK 128
#define THREAD_PER_ROW 2 // 2 thread for each row in matrix A to load numbers
#define THREAD_PER_COL 4 // 4 thread for each row in matrix B to load numbers
#define SG_MAT_SIZE 64 // simdgroup matrix is of shape 8x8
#define SG_MAT_ROW 8

template <typename T4x4, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
struct DefaultDequantizer {
    using type4x4 = T4x4;
    using Block   = block_q;
    DefaultDequantizer(device const char * cx, short il) : x((device const block_q *)cx + il/nl), il(il) {}
    inline void convert(thread T4x4& t) const { dequantize_func(x, il, t); }
    inline void next() {
        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
    }
    device const block_q * x;
    short il;
};

template <typename T4x4, typename Block, typename Scale, int nl, void (*dequantize)(device const Block *, short, thread T4x4&), bool may_not_be_aligned = false>
struct DequantizerRS {
    using type4x4 = T4x4;
    DequantizerRS(device const char * cx, short il = 0) : il(il) {
        if (may_not_be_aligned) {
            thread char * aux = (thread char *)&d;
            for (int i = 0; i < int(sizeof(d)); ++i) aux[i] = cx[i];
        } else {
            d = *(device const Scale *)cx;
        }
        x = (device const Block *)(cx + sizeof(Scale));
    }
    inline void convert(thread T4x4& t) const {
        dequantize(x, il, t);
        t *= d;
    }
    inline void convert(int64_t ind, thread T4x4& t) {
        dequantize(x + ind/nl, ind%nl, t);
        t *= d;
    }
    inline void next() {
        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
    }
    device const Block * x;
    short il;
    Scale d;
};

template <typename T4x4, typename Block, typename Scale, int nl, void (*dequantize)(device const Block *, short, thread T4x4&)>
struct DequantizerRST4 {
    using type4x4 = T4x4;
    DequantizerRST4(device const char * cx, short il = 0) : il(il) {
        device const Scale * dptr = (device const Scale *)cx;
        d[0] = dptr[0] * Scale(31.75f * 1.01f);
        d[1] = dptr[1];
        x = (device const Block *)(dptr + 2);
    }
    inline void convert(thread T4x4& t) const {
        dequantize(x, il, t);
        for (int i = 0; i < 4; ++i) t[i] = t[i]*d[0] + d[1];
    }
    inline void convert(int64_t ind, thread T4x4& t) {
        dequantize(x + ind/nl, ind%nl, t);
        for (int i = 0; i < 4; ++i) t[i] = t[i]*d[0] + d[1];
    }
    inline void next() {
        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
    }
    device const Block * x;
    short il;
    Scale d[2];
};

template <typename T4x4, int nl>
struct DequantizerKT4 {
    using Block = block_iq4_kt;
    using type4x4 = T4x4;
    DequantizerKT4(device const char * cx, short il = 0) : il(il) {
        device const float * dptr = (device const float *)cx;
        d = dptr[0] * 1.01f;
        x = (device const Block *)(dptr + 1);
    }
    inline void convert(thread T4x4& t) const {
        float4x4 tmp;
        dequantize_iq4_kt(x, il, d, tmp);
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) t[i][j] = tmp[i][j];
    }
    inline void convert(int64_t ind, thread T4x4& t) {
        float4x4 tmp;
        dequantize_iq4_kt(x + ind/nl, ind%nl, d, tmp);
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) t[i][j] = tmp[i][j];
    }
    inline void next() {
        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
    }
    device const Block * x;
    short il;
    float d;
};

template <typename T4x4, typename Block, typename Scale, int nl, void (*dequantize)(half d, device const Block *, short, thread T4x4&), bool may_not_be_aligned = false>
struct DequantizerRSBN {
    using type4x4 = T4x4;
    DequantizerRSBN(device const char * cx, short il = 0) : il(il) {
        if (may_not_be_aligned) {
            thread char * aux = (thread char *)&d;
            for (int i = 0; i < int(sizeof(d)); ++i) aux[i] = cx[i];
        } else {
            d = *(device const Scale *)cx;
        }
        x = (device const Block *)(cx + sizeof(Scale));
    }
    inline void convert(thread T4x4& t) const {
        dequantize(d, x, il, t);
    }
    inline void convert(int64_t ind, thread T4x4& t) {
        dequantize(d, x + ind/nl, ind%nl, t);
    }
    inline void next() {
        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2+nl-1)/nl : x;
    }
    device const Block * x;
    short il;
    Scale d;
};

// each block_q contains 16*nl weights
template<typename T, typename simdgroup_T8x8, typename Dequantizer, typename src1_t>
kernel void kernel_mul_mm(device const  uchar * src0,
                          device const  uchar * src1,
                          device        float * dst,
                          constant    int64_t & ne00,
                          constant    int64_t & ne02,
                          constant   uint64_t & nb01,
                          constant   uint64_t & nb02,
                          constant    int64_t & ne12,
                          constant   uint64_t & nb10,
                          constant   uint64_t & nb11,
                          constant   uint64_t & nb12,
                          constant    int64_t & ne0,
                          constant    int64_t & ne1,
                          constant       uint & r2,
                          constant       uint & r3,
                          threadgroup   uchar * shared_memory [[threadgroup(0)]],
                          uint3                 tgpig[[threadgroup_position_in_grid]],
                          uint                  tiitg[[thread_index_in_threadgroup]],
                          uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup T     * sa = (threadgroup T     *)(shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;
    const uint im = tgpig.z;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_T8x8     ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 c_res[8];
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    short il = (tiitg % THREAD_PER_ROW);

    const uint i12 = im%ne12;
    const uint i13 = im/ne12;

    uint   offset0 = (i12/r2)*nb02 + (i13/r3)*(nb02*ne02);

    device const char   * cx = (device const char  *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01 + offset0);
    device const src1_t * y = (device const src1_t *)(src1
        + nb12 * im
        + nb11 * (r1 * BLOCK_SIZE_N + thread_col)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    Dequantizer deq(cx, il);

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        typename Dequantizer::type4x4 temp_a;
        deq.convert(temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll(16)
        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        if (is_same_v<src1_t, float>) {
            *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *)y);
        } else {
            half2x4 h = *((device half2x4 *)y);
            *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = (float2x4)h;
        }

        deq.next();
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup T     * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        #pragma unroll(4)
        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            #pragma unroll(4)
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i],lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll(2)
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i],lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            #pragma unroll(8)
            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }
        }
    }

    if ((r0 + 1) * BLOCK_SIZE_M <= ne0 && (r1 + 1) * BLOCK_SIZE_N <= ne1) {
        device float * C = dst + (BLOCK_SIZE_M * r0 + 32 * (sgitg &  1)) \
                               + (BLOCK_SIZE_N * r1 + 16 * (sgitg >> 1)) * ne0 + im*ne1*ne0;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], C + 8 * (i%4) + 8 * ne0 * (i/4), ne0);
        }
    } else {
        // block is smaller than 64x32, we should avoid writing data outside of the matrix
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * C = dst + (BLOCK_SIZE_M * r0) + (BLOCK_SIZE_N * r1) * ne0 + im*ne1*ne0;
        if (sgitg == 0) {
            for (int i = 0; i < n_rows; i++) {
                for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                    *(C + i + j * ne0) = *(temp_str + i + j * BLOCK_SIZE_M);
                }
            }
        }
    }
}

// same as kernel_mul_mm_impl, but src1 and dst are accessed via indices stored in rowids
template<typename Dequantizer>
void kernel_mul_mm_id_impl(
        device const  uchar * src0,
        device const  uchar * src1,
        threadgroup ushort2 * rowids,
        device        float * dst,
        constant    int64_t & ne00,
        constant    int64_t & ne02,
        constant   uint64_t & nb01,
        constant   uint64_t & nb02,
        constant    int64_t & ne11,
        constant    int64_t & ne12,
        constant   uint64_t & nb10,
        constant   uint64_t & nb11,
        constant   uint64_t & nb12,
        constant    int64_t & ne0,
                    int64_t   ne1,
                    int64_t   ne0ne1,
        threadgroup   uchar * shared_memory,
        uint3                 tgpig[[threadgroup_position_in_grid]],
        uint                  tiitg[[thread_index_in_threadgroup]],
        uint                  sgitg[[simdgroup_index_in_threadgroup]]) {

    threadgroup half  * sa = (threadgroup half  *)(shared_memory);
    threadgroup float * sb = (threadgroup float *)(shared_memory + 4096);

    const uint r0 = tgpig.y;
    const uint r1 = tgpig.x;

    if (r1 * BLOCK_SIZE_N >= ne1) return;

    // if this block is of 64x32 shape or smaller
    short n_rows = (ne0 - r0 * BLOCK_SIZE_M < BLOCK_SIZE_M) ? (ne0 - r0 * BLOCK_SIZE_M) : BLOCK_SIZE_M;
    short n_cols = (ne1 - r1 * BLOCK_SIZE_N < BLOCK_SIZE_N) ? (ne1 - r1 * BLOCK_SIZE_N) : BLOCK_SIZE_N;

    // a thread shouldn't load data outside of the matrix
    short thread_row = ((short)tiitg/THREAD_PER_ROW) < n_rows ? ((short)tiitg/THREAD_PER_ROW) : n_rows - 1;
    short thread_col = ((short)tiitg/THREAD_PER_COL) < n_cols ? ((short)tiitg/THREAD_PER_COL) : n_cols - 1;

    simdgroup_half8x8  ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 c_res[8];
    for (int i = 0; i < 8; i++){
        c_res[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }
    short il = (tiitg % THREAD_PER_ROW);

    threadgroup const auto & id = rowids[r1 * BLOCK_SIZE_N + thread_col];

    device const char  * cx = (device const char  *)(src0 + (r0 * BLOCK_SIZE_M + thread_row) * nb01);
    device const float *  y = (device const float *)(src1
        + nb12 * id[1]
        + nb11 * (id[0] % ne11)
        + nb10 * (BLOCK_SIZE_K / THREAD_PER_COL * (tiitg % THREAD_PER_COL)));

    Dequantizer deq(cx, il);

    for (int loop_k = 0; loop_k < ne00; loop_k += BLOCK_SIZE_K) {
        // load data and store to threadgroup memory
        half4x4 temp_a;
        deq.convert(temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < 16; i++) {
            *(sa + SG_MAT_SIZE * ((tiitg / THREAD_PER_ROW / 8) \
            +                     (tiitg % THREAD_PER_ROW) * 16 + (i / 8) * 8) \
            +                     (tiitg / THREAD_PER_ROW) % 8  + (i & 7) * 8) = temp_a[i/4][i%4];
        }

        *(threadgroup float2x4 *)(sb + (tiitg % THREAD_PER_COL) * 8 * 32 + 8 * (tiitg / THREAD_PER_COL)) = *((device float2x4 *)y);

        deq.next();
        y += BLOCK_SIZE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // load matrices from threadgroup memory and conduct outer products
        threadgroup half  * lsma = (sa + THREAD_MAT_M * SG_MAT_SIZE * (sgitg % 2));
        threadgroup float * lsmb = (sb + THREAD_MAT_N * SG_MAT_SIZE * (sgitg / 2));

        for (int ik = 0; ik < BLOCK_SIZE_K / 8; ik++) {
            for (int i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + SG_MAT_SIZE * i);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (int i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + SG_MAT_SIZE * i);
            }

            lsma += BLOCK_SIZE_M / SG_MAT_ROW * SG_MAT_SIZE;
            lsmb += BLOCK_SIZE_N / SG_MAT_ROW * SG_MAT_SIZE;

            for (int i = 0; i < 8; i++){
                simdgroup_multiply_accumulate(c_res[i], mb[i/4], ma[i%4], c_res[i]);
            }
        }
    }

    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shared_memory) \
                                      + 32 * (sgitg&1) + (16 * (sgitg>>1)) * BLOCK_SIZE_M;
        for (int i = 0; i < 8; i++) {
            simdgroup_store(c_res[i], temp_str + 8 * (i%4) + 8 * BLOCK_SIZE_M * (i/4), BLOCK_SIZE_M);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float * C = dst + (BLOCK_SIZE_M * r0);
        if (sgitg == 0) {
            for (int j = tiitg; j < n_cols; j += BLOCK_SIZE_N) {
                threadgroup const auto & jid = rowids[r1 * BLOCK_SIZE_N + j];
                int joff =  jid[0] * ne0 + jid[1] * ne0ne1;
                for (int i = 0; i < n_rows; i++) {
                    *(C + i + joff) = *(temp_str + i + j * BLOCK_SIZE_M);
                }
            }
        }
    }
}

template<typename Dequantizer>
kernel void kernel_mul_mm_id(
        device const   uchar * src0s,
        device const   uchar * src1,
        device         float * dst,
        device const   uchar * ids,
        constant     int64_t & nei0,
        constant     int64_t & nei1,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne02,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        threadgroup    uchar * shared_memory [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint3                  ntg3[[threads_per_threadgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {

    const int32_t i02 = tgpig.z;
    tgpig.z = 0;

    device const uchar * src0 = src0s + i02*nb02;

    uint ntg = ntg3.x * ntg3.y * ntg3.z;
    uint n   = nei0*nei1;

    uint nhave = 0;
    for (uint i = tiitg; i < n; i += ntg) {
        uint ii0 = i % nei0;
        uint ii1 = i / nei0;
        int32_t id = ((device int32_t *) (ids + ii1*nbi1))[ii0];
        if (id == i02) ++nhave;
    }
    threadgroup uint * nums = (threadgroup uint *)shared_memory;
    nums[tiitg] = nhave;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint stride = 1;
    while (stride <= ntg/2) {
        uint index = (tiitg+1)*stride*2 - 1; // index - stride = 2*tiitg*stride + stride - 1;
        if (index < ntg) nums[index] += nums[index-stride];
        stride <<= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    stride = ntg/2;
    while (stride > 0) {
        uint index = (tiitg+1)*stride*2 - 1;
        if (index+stride < ntg) nums[index+stride] += nums[index];
        stride >>= 1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint _ne1 = nums[ntg-1];
    if (!_ne1) return;

    uint nprev = tiitg > 0 ? nums[tiitg-1] : 0;

    threadgroup ushort2 * rowids = (threadgroup ushort2 *)(shared_memory + 8192);
    for (uint i = tiitg; i < n; i += ntg) {
        uint ii0 = i % nei0;
        uint ii1 = i / nei0;
        int32_t id = ((device int32_t *) (ids + ii1*nbi1))[ii0];
        if (id == i02) rowids[nprev++] = ushort2(ii0, ii1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint nstep = (_ne1 + BLOCK_SIZE_N - 1)/BLOCK_SIZE_N;

    for (uint istep = 0; istep < nstep; ++istep) {

        uint first = BLOCK_SIZE_N*istep; 
        uint last  = first + BLOCK_SIZE_N < _ne1 ? first + BLOCK_SIZE_N : _ne1;
        int64_t this_ne1 = last - first;
        threadgroup ushort2 * this_rowids = rowids + istep*BLOCK_SIZE_N;

        kernel_mul_mm_id_impl<Dequantizer>(
            src0,
            src1,
            this_rowids,
            dst,
            ne00,
            ne02,
            nb01,
            nb02,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            this_ne1,
            ne0*ne1,
            shared_memory,
            tgpig,
            tiitg,
            sgitg);
    }
}

#define QK_NL 16

//
// get rows
//

typedef decltype(kernel_get_rows_f<float>) get_rows_f_t;

template [[host_name("kernel_get_rows_f32")]]  kernel get_rows_f_t kernel_get_rows_f<float>;
template [[host_name("kernel_get_rows_f16")]]  kernel get_rows_f_t kernel_get_rows_f<half>;

typedef decltype(kernel_get_rows_q<block_q4_0, 2, dequantize_q4_0>) get_rows_q_t;

template [[host_name("kernel_get_rows_bf16")]]    kernel get_rows_q_t kernel_get_rows_q<half4x4,       1, dequantize_bf16>;
template [[host_name("kernel_get_rows_q4_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_0,    2, dequantize_q4_0>;
template [[host_name("kernel_get_rows_q4_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_1,    2, dequantize_q4_1>;
template [[host_name("kernel_get_rows_q5_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_0,    2, dequantize_q5_0>;
template [[host_name("kernel_get_rows_q5_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_1,    2, dequantize_q5_1>;
template [[host_name("kernel_get_rows_q6_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q6_0,    2, dequantize_q6_0>;
template [[host_name("kernel_get_rows_q8_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q8_0,    2, dequantize_q8_0>;
template [[host_name("kernel_get_rows_q2_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_get_rows_q3_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_get_rows_q4_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_get_rows_q5_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_get_rows_q6_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_get_rows_iq2_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_get_rows_iq2_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_get_rows_iq3_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_get_rows_iq3_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_get_rows_iq2_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_get_rows_iq1_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_get_rows_iq1_m")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_get_rows_iq4_nl")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_get_rows_iq4_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_xs,  QK_NL, dequantize_iq4_xs>;
template [[host_name("kernel_get_rows_iq2_k")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq2_k,   QK_NL, dequantize_iq2_k>;
template [[host_name("kernel_get_rows_iq3_k")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq3_k,   QK_NL, dequantize_iq3_k>;
template [[host_name("kernel_get_rows_iq4_k")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq4_k,   QK_NL, dequantize_iq4_k>;
template [[host_name("kernel_get_rows_iq5_k")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq5_k,   QK_NL, dequantize_iq5_k>;
template [[host_name("kernel_get_rows_iq6_k")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq6_k,   QK_NL, dequantize_iq6_k>;
template [[host_name("kernel_get_rows_iq1_bn")]]  kernel get_rows_q_t kernel_get_rows_q2<DequantizerRSBN<float4x4, block_iq1_bn,  half,  4, dequantize_iq1_bn, true>>;
template [[host_name("kernel_get_rows_iq2_bn")]]  kernel get_rows_q_t kernel_get_rows_q2<DequantizerRSBN<float4x4, block_iq2_bn, float,  4, dequantize_iq2_bn>>;
template [[host_name("kernel_get_rows_iq3_ks")]]  kernel get_rows_q_t kernel_get_rows_q2<DequantizerRS<float4x4, block_iq3_ks,  half, 16, dequantize_iq3_ks>>;
template [[host_name("kernel_get_rows_iq4_ks")]]  kernel get_rows_q_t kernel_get_rows_q2<DequantizerRS<float4x4, block_iq4_ks, float, 16, dequantize_iq4_ks>>;
template [[host_name("kernel_get_rows_iq5_ks")]]  kernel get_rows_q_t kernel_get_rows_q2<DequantizerRS<float4x4, block_iq5_ks, float, 16, dequantize_iq5_ks>>;
template [[host_name("kernel_get_rows_iq4_kss")]] kernel get_rows_q_t kernel_get_rows_q2<DequantizerRS<float4x4, block_iq4_kss,float, 16, dequantize_iq4_kss>>;
template [[host_name("kernel_get_rows_iq2_ks")]]  kernel get_rows_q_t kernel_get_rows_q2<DequantizerRS<float4x4, block_iq2_ks,  half, 16, dequantize_iq2_ks>>;
template [[host_name("kernel_get_rows_iq2_kl")]]  kernel get_rows_q_t kernel_get_rows_q2<DequantizerRS<float4x4, block_iq2_kl,  half, 16, dequantize_iq2_kl>>;
template [[host_name("kernel_get_rows_iq2_kt")]]  kernel get_rows_q_t kernel_get_rows_q2<DequantizerRS<float4x4, block_iq2_kt, float, 16, dequantize_iq2_kt>>;
template [[host_name("kernel_get_rows_iq3_kt")]]  kernel get_rows_q_t kernel_get_rows_q2<DequantizerRS<float4x4, block_iq3_kt, float, 16, dequantize_iq3_kt>>;
template [[host_name("kernel_get_rows_iq4_kt")]]  kernel get_rows_q_t kernel_get_rows_q2<DequantizerKT4<float4x4, 16>>;

//
// matrix-matrix multiplication
//

template <typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
using DD = DefaultDequantizer<half4x4, block_q, nl, dequantize_func>;

typedef decltype(kernel_mul_mm<half, simdgroup_half8x8, DD<float4x4, 1, dequantize_f32>, float>) mat_mm_t;

template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<float4x4,      1,     dequantize_f32>, float>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<half4x4,       1,     dequantize_f16>, float>;
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<half4x4,       1,     dequantize_bf16>, float>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q4_0,    2,     dequantize_q4_0>, float>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q4_1,    2,     dequantize_q4_1>, float>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q5_0,    2,     dequantize_q5_0>, float>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q5_1,    2,     dequantize_q5_1>, float>;
template [[host_name("kernel_mul_mm_q6_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q6_0,    2,     dequantize_q6_0>, float>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q8_0,    2,     dequantize_q8_0>, float>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q2_K,    QK_NL, dequantize_q2_K>, float>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q3_K,    QK_NL, dequantize_q3_K>, float>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q4_K,    QK_NL, dequantize_q4_K>, float>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q5_K,    QK_NL, dequantize_q5_K>, float>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q6_K,    QK_NL, dequantize_q6_K>, float>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>, float>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq2_xs,  QK_NL, dequantize_iq2_xs>, float>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>, float>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq3_s,   QK_NL, dequantize_iq3_s>, float>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq2_s,   QK_NL, dequantize_iq2_s>, float>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq1_s,   QK_NL, dequantize_iq1_s>, float>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq1_m,   QK_NL, dequantize_iq1_m>, float>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq4_nl,  2,     dequantize_iq4_nl>, float>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq4_xs,  QK_NL, dequantize_iq4_xs>, float>;
template [[host_name("kernel_mul_mm_iq2_k_f32")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq2_k,   QK_NL, dequantize_iq2_k>, float>;
template [[host_name("kernel_mul_mm_iq3_k_f32")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq3_k,   QK_NL, dequantize_iq3_k>, float>;
template [[host_name("kernel_mul_mm_iq4_k_f32")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq4_k,   QK_NL, dequantize_iq4_k>, float>;
template [[host_name("kernel_mul_mm_iq5_k_f32")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq5_k,   QK_NL, dequantize_iq5_k>, float>;
template [[host_name("kernel_mul_mm_iq6_k_f32")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq6_k,   QK_NL, dequantize_iq6_k>, float>;
template [[host_name("kernel_mul_mm_iq1_bn_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRSBN<half4x4, block_iq1_bn,  half,  4, dequantize_iq1_bn, true>, float>;
template [[host_name("kernel_mul_mm_iq2_bn_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRSBN<half4x4, block_iq2_bn, float,  4, dequantize_iq2_bn>, float>;
template [[host_name("kernel_mul_mm_iq3_ks_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq3_ks,  half, 16, dequantize_iq3_ks>, float>;
template [[host_name("kernel_mul_mm_iq4_ks_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq4_ks, float, 16, dequantize_iq4_ks>, float>;
template [[host_name("kernel_mul_mm_iq5_ks_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq5_ks, float, 16, dequantize_iq5_ks>, float>;
template [[host_name("kernel_mul_mm_iq4_kss_f32")]] kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq4_kss,float, 16, dequantize_iq4_kss>, float>;
template [[host_name("kernel_mul_mm_iq2_ks_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq2_ks,  half, 16, dequantize_iq2_ks>, float>;
template [[host_name("kernel_mul_mm_iq2_kl_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq2_kl,  half, 16, dequantize_iq2_kl>, float>;
template [[host_name("kernel_mul_mm_iq2_kt_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq2_kt, float, 16, dequantize_iq2_kt>, float>;
template [[host_name("kernel_mul_mm_iq3_kt_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq3_kt, float, 16, dequantize_iq3_kt>, float>;
template [[host_name("kernel_mul_mm_iq4_kt_f32")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerKT4<half4x4, 16>, float>;

template [[host_name("kernel_mul_mm_f32_f16")]]     kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<float4x4,      1,     dequantize_f32>, half>;
template [[host_name("kernel_mul_mm_f16_f16")]]     kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<half4x4,       1,     dequantize_f16>, half>;
template [[host_name("kernel_mul_mm_bf16_f16")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<half4x4,       1,     dequantize_bf16>, half>;
template [[host_name("kernel_mul_mm_q4_0_f16")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q4_0,    2,     dequantize_q4_0>, half>;
template [[host_name("kernel_mul_mm_q4_1_f16")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q4_1,    2,     dequantize_q4_1>, half>;
template [[host_name("kernel_mul_mm_q5_0_f16")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q5_0,    2,     dequantize_q5_0>, half>;
template [[host_name("kernel_mul_mm_q5_1_f16")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q5_1,    2,     dequantize_q5_1>, half>;
template [[host_name("kernel_mul_mm_q6_0_f16")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q6_0,    2,     dequantize_q6_0>, half>;
template [[host_name("kernel_mul_mm_q8_0_f16")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q8_0,    2,     dequantize_q8_0>, half>;
template [[host_name("kernel_mul_mm_q2_K_f16")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q2_K,    QK_NL, dequantize_q2_K>, half>;
template [[host_name("kernel_mul_mm_q3_K_f16")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q3_K,    QK_NL, dequantize_q3_K>, half>;
template [[host_name("kernel_mul_mm_q4_K_f16")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q4_K,    QK_NL, dequantize_q4_K>, half>;
template [[host_name("kernel_mul_mm_q5_K_f16")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q5_K,    QK_NL, dequantize_q5_K>, half>;
template [[host_name("kernel_mul_mm_q6_K_f16")]]    kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_q6_K,    QK_NL, dequantize_q6_K>, half>;
template [[host_name("kernel_mul_mm_iq2_xxs_f16")]] kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>, half>;
template [[host_name("kernel_mul_mm_iq2_xs_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq2_xs,  QK_NL, dequantize_iq2_xs>, half>;
template [[host_name("kernel_mul_mm_iq3_xxs_f16")]] kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>, half>;
template [[host_name("kernel_mul_mm_iq3_s_f16")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq3_s,   QK_NL, dequantize_iq3_s>, half>;
template [[host_name("kernel_mul_mm_iq2_s_f16")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq2_s,   QK_NL, dequantize_iq2_s>, half>;
template [[host_name("kernel_mul_mm_iq1_s_f16")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq1_s,   QK_NL, dequantize_iq1_s>, half>;
template [[host_name("kernel_mul_mm_iq1_m_f16")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq1_m,   QK_NL, dequantize_iq1_m>, half>;
template [[host_name("kernel_mul_mm_iq4_nl_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq4_nl,  2,     dequantize_iq4_nl>, half>;
template [[host_name("kernel_mul_mm_iq4_xs_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq4_xs,  QK_NL, dequantize_iq4_xs>, half>;
template [[host_name("kernel_mul_mm_iq2_k_f16")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq2_k,   QK_NL, dequantize_iq2_k>, half>;
template [[host_name("kernel_mul_mm_iq3_k_f16")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq3_k,   QK_NL, dequantize_iq3_k>, half>;
template [[host_name("kernel_mul_mm_iq4_k_f16")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq4_k,   QK_NL, dequantize_iq4_k>, half>;
template [[host_name("kernel_mul_mm_iq5_k_f16")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq5_k,   QK_NL, dequantize_iq5_k>, half>;
template [[host_name("kernel_mul_mm_iq6_k_f16")]]   kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DD<block_iq6_k,   QK_NL, dequantize_iq6_k>, half>;
template [[host_name("kernel_mul_mm_iq1_bn_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRSBN<half4x4, block_iq1_bn,  half,  4, dequantize_iq1_bn, true>, half>;
template [[host_name("kernel_mul_mm_iq2_bn_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRSBN<half4x4, block_iq2_bn, float,  4, dequantize_iq2_bn>, half>;
template [[host_name("kernel_mul_mm_iq3_ks_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq3_ks,  half, 16, dequantize_iq3_ks>, half>;
template [[host_name("kernel_mul_mm_iq4_ks_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq4_ks, float, 16, dequantize_iq4_ks>, half>;
template [[host_name("kernel_mul_mm_iq5_ks_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq5_ks, float, 16, dequantize_iq5_ks>, half>;
template [[host_name("kernel_mul_mm_iq4_kss_f16")]] kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq4_kss,float, 16, dequantize_iq4_kss>, half>;
template [[host_name("kernel_mul_mm_iq2_ks_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq2_ks,  half, 16, dequantize_iq2_ks>, half>;
template [[host_name("kernel_mul_mm_iq2_kl_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq2_kl,  half, 16, dequantize_iq2_kl>, half>;
template [[host_name("kernel_mul_mm_iq2_kt_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq2_kt, float, 16, dequantize_iq2_kt>, half>;
template [[host_name("kernel_mul_mm_iq3_kt_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerRS<half4x4, block_iq3_kt, float, 16, dequantize_iq3_kt>, half>;
template [[host_name("kernel_mul_mm_iq4_kt_f16")]]  kernel mat_mm_t kernel_mul_mm<half, simdgroup_half8x8, DequantizerKT4<half4x4, 16>, half>;


//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<DD<float4x4, 1, dequantize_f32>>) mat_mm_id_t;

template [[host_name("kernel_mul_mm_id_f32_f32")]]     kernel mat_mm_id_t kernel_mul_mm_id<DD<float4x4,      1,     dequantize_f32>>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]     kernel mat_mm_id_t kernel_mul_mm_id<DD<half4x4,       1,     dequantize_f16>>;
template [[host_name("kernel_mul_mm_id_bf16_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<DD<half4x4,       1,     dequantize_bf16>>;
template [[host_name("kernel_mul_mm_id_q4_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<DD<block_q4_0,    2,     dequantize_q4_0>>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<DD<block_q4_1,    2,     dequantize_q4_1>>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<DD<block_q5_0,    2,     dequantize_q5_0>>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<DD<block_q5_1,    2,     dequantize_q5_1>>;
template [[host_name("kernel_mul_mm_id_q6_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<DD<block_q6_0,    2,     dequantize_q6_0>>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<DD<block_q8_0,    2,     dequantize_q8_0>>;
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<DD<block_q2_K,    QK_NL, dequantize_q2_K>>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<DD<block_q3_K,    QK_NL, dequantize_q3_K>>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<DD<block_q4_K,    QK_NL, dequantize_q4_K>>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<DD<block_q5_K,    QK_NL, dequantize_q5_K>>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mat_mm_id_t kernel_mul_mm_id<DD<block_q6_K,    QK_NL, dequantize_q6_K>>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq2_xs,  QK_NL, dequantize_iq2_xs>>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq3_s,   QK_NL, dequantize_iq3_s>>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq2_s,   QK_NL, dequantize_iq2_s>>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq1_s,   QK_NL, dequantize_iq1_s>>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq1_m,   QK_NL, dequantize_iq1_m>>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq4_nl,  2,     dequantize_iq4_nl>>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq4_xs,  QK_NL, dequantize_iq4_xs>>;
template [[host_name("kernel_mul_mm_id_iq2_k_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq2_k,   QK_NL, dequantize_iq2_k>>;
template [[host_name("kernel_mul_mm_id_iq3_k_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq3_k,   QK_NL, dequantize_iq3_k>>;
template [[host_name("kernel_mul_mm_id_iq4_k_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq4_k,   QK_NL, dequantize_iq4_k>>;
template [[host_name("kernel_mul_mm_id_iq5_k_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq5_k,   QK_NL, dequantize_iq5_k>>;
template [[host_name("kernel_mul_mm_id_iq6_k_f32")]]   kernel mat_mm_id_t kernel_mul_mm_id<DD<block_iq6_k,   QK_NL, dequantize_iq6_k>>;
template [[host_name("kernel_mul_mm_id_iq1_bn_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DequantizerRSBN<half4x4, block_iq1_bn,  half,  4, dequantize_iq1_bn, true>>;
template [[host_name("kernel_mul_mm_id_iq2_bn_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DequantizerRSBN<half4x4, block_iq2_bn, float,  4, dequantize_iq2_bn>>;
template [[host_name("kernel_mul_mm_id_iq3_ks_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DequantizerRS<half4x4, block_iq3_ks,  half, 16, dequantize_iq3_ks>>;
template [[host_name("kernel_mul_mm_id_iq4_ks_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DequantizerRS<half4x4, block_iq4_ks, float, 16, dequantize_iq4_ks>>;
template [[host_name("kernel_mul_mm_id_iq5_ks_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DequantizerRS<half4x4, block_iq5_ks, float, 16, dequantize_iq5_ks>>;
template [[host_name("kernel_mul_mm_id_iq4_kss_f32")]] kernel mat_mm_id_t kernel_mul_mm_id<DequantizerRS<half4x4, block_iq4_kss,float, 16, dequantize_iq4_kss>>;
template [[host_name("kernel_mul_mm_id_iq2_ks_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DequantizerRS<half4x4, block_iq2_ks,  half, 16, dequantize_iq2_ks>>;
template [[host_name("kernel_mul_mm_id_iq2_kl_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DequantizerRS<half4x4, block_iq2_kl,  half, 16, dequantize_iq2_kl>>;
template [[host_name("kernel_mul_mm_id_iq2_kt_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DequantizerRS<half4x4, block_iq2_kt, float, 16, dequantize_iq2_kt>>;
template [[host_name("kernel_mul_mm_id_iq3_kt_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DequantizerRS<half4x4, block_iq3_kt, float, 16, dequantize_iq3_kt>>;
template [[host_name("kernel_mul_mm_id_iq4_kt_f32")]]  kernel mat_mm_id_t kernel_mul_mm_id<DequantizerKT4<half4x4, 16>>;

//
// matrix-vector multiplication
//

typedef void (kernel_mul_mv_impl_t)(
        device const  char * src0,
        device const  char * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                  uint64_t   nb00,
                  uint64_t   nb01,
                  uint64_t   nb02,
                   int64_t   ne10,
                   int64_t   ne11,
                   int64_t   ne12,
                  uint64_t   nb10,
                  uint64_t   nb11,
                  uint64_t   nb12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
                   uint3     tgpig,
                   uint      tiisg);

typedef void (kernel_mul_mv2_impl_t)(
        device const  void * src0,
        device const float * src1,
        device       float * dst,
                   int64_t   ne00,
                   int64_t   ne01,
                   int64_t   ne02,
                   int64_t   ne10,
                   int64_t   ne12,
                   int64_t   ne0,
                   int64_t   ne1,
                   uint      r2,
                   uint      r3,
        threadgroup int8_t * shared_values,
                   uint3     tgpig,
                   uint      tiisg,
                   uint      sgitg);

template<kernel_mul_mv_impl_t impl_fn>
void mmv_fn(
        device const    char * src0,
        device const    char * src1,
        device         float * dst,
                     int64_t   ne00,
                     int64_t   ne01,
                     int64_t   ne02,
                    uint64_t   nb00,
                    uint64_t   nb01,
                    uint64_t   nb02,
                     int64_t   ne10,
                     int64_t   ne11,
                     int64_t   ne12,
                     int64_t   ne13,
                    uint64_t   nb10,
                    uint64_t   nb11,
                    uint64_t   nb12,
                     int64_t   ne0,
                     int64_t   ne1,
                    uint64_t   nb1,
                        uint   r2,
                        uint   r3,
        threadgroup int8_t   * shared_values,
        uint3                  tgpig,
        uint                   tiitg,
        uint                   tiisg,
        uint                   sgitg) {
    impl_fn(src0,src1,dst,ne00,ne01,ne02,nb00,nb01,nb02,ne10,ne11,ne12,nb10,nb11,nb12,ne0,ne1,r2,r3,tgpig,tiisg);
}

template<kernel_mul_mv2_impl_t impl_fn>
void mmv_fn(
        device const    char * src0,
        device const    char * src1,
        device         float * dst,
                     int64_t   ne00,
                     int64_t   ne01,
                     int64_t   ne02,
                    uint64_t   nb00,
                    uint64_t   nb01,
                    uint64_t   nb02,
                     int64_t   ne10,
                     int64_t   ne11,
                     int64_t   ne12,
                     int64_t   ne13,
                    uint64_t   nb10,
                    uint64_t   nb11,
                    uint64_t   nb12,
                     int64_t   ne0,
                     int64_t   ne1,
                    uint64_t   nb1,
                        uint   r2,
                        uint   r3,
        threadgroup int8_t   * shared_values,
        uint3                  tgpig,
        uint                   tiitg,
        uint                   tiisg,
        uint                   sgitg) {
    impl_fn(src0,(const device float *)src1,dst,ne00,ne01,ne02,ne10,ne12,ne0,ne1,r2,r3,shared_values,tgpig,tiisg,sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_impl<half, half4, half, half4>>) mul_mv_impl_fn_t;

template<mul_mv_impl_fn_t impl_fn>
kernel void kernel_mul_mv_id(
        device const    char * src0s,
        device const    char * src1,
        device         float * dst,
        device const    char * ids,
        constant     int64_t & nei0,
        constant     int64_t & nei1,
        constant    uint64_t & nbi1,
        constant     int64_t & ne00,
        constant     int64_t & ne01,
        constant     int64_t & ne02,
        constant    uint64_t & nb00,
        constant    uint64_t & nb01,
        constant    uint64_t & nb02,
        constant     int64_t & ne10,
        constant     int64_t & ne11,
        constant     int64_t & ne12,
        constant     int64_t & ne13,
        constant    uint64_t & nb10,
        constant    uint64_t & nb11,
        constant    uint64_t & nb12,
        constant     int64_t & ne0,
        constant     int64_t & ne1,
        constant    uint64_t & nb1,
        threadgroup int8_t   * shared_values [[threadgroup(0)]],
        uint3                  tgpig[[threadgroup_position_in_grid]],
        uint                   tiitg[[thread_index_in_threadgroup]],
        uint                   tiisg[[thread_index_in_simdgroup]],
        uint                   sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/nei0;
    const int idx = tgpig.z%nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*nbi1))[idx];

    const int64_t i11 = idx % ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src0_cur = src0s + i02*nb02;
    device const char * src1_cur = src1 + i11*nb11 + i12*nb12;
    device      float * dst_cur  = dst + i1*ne0 + i2*ne1*ne0;

    impl_fn(
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        /* ne00 */ ne00,
        /* ne01 */ ne01,
        /* ne02 */ 1,//ne02,
        /* nb00 */ nb00,
        /* nb01 */ nb01,
        /* nb02 */ nb02,
        /* ne10 */ ne10,
        /* ne11 */ 1,//ne11,
        /* ne12 */ 1,//ne12,
        /* ne13 */ 1,//ne13,
        /* nb10 */ nb10,
        /* nb11 */ nb11,
        /* nb12 */ nb12,
        /* ne0  */ ne0,
        /* ne1  */ 1,//ne1,
        /* nb1  */ nb1,
        /* r2   */ 1,
        /* r3   */ 1,
        shared_values,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<float, float4, float, float4>>>) kernel_mul_mv_id_t;

template [[host_name("kernel_mul_mv_id_f32_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<float, float4, float, float4>>>;
template [[host_name("kernel_mul_mv_id_f16_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_impl<half, half4, float, float4>>>;
template [[host_name("kernel_mul_mv_id_bf16_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_bf16_impl<float>>>;
template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q4_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q4_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q5_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q5_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q6_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q6_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>>>;
template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q3_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q3_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q5_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_q6_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq1_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_s_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq1_m_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_m_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq1_bn_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_bn_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_bn_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_bn_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq3_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_xxs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq3_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_s_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_s_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq4_nl_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_nl_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq4_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_xs_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq3_ks_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_ks_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq4_ks_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_ks_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq5_ks_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq5_ks_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq4_kss_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_kss_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_k_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_k_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_ks_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_ks_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_kl_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_kl_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq2_kt_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_kt_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq3_kt_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_kt_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq4_kt_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_kt_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq3_k_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_k_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq4_k_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_k_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq5_k_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq5_k_f32_impl>>;
template [[host_name("kernel_mul_mv_id_iq6_k_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq6_k_f32_impl>>;
