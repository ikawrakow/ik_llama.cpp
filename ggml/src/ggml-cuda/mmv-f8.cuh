#pragma once

#include "f8.cuh"
#include "unary.cuh"

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif

static __device__ __forceinline__ float2 ggml_cuda_f8_e4m3x2_to_float2(uint16_t x) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080 && __CUDA_ARCH__ >= 890
    return __half22float2(__nv_cvt_fp8x2_to_halfraw2(x, __NV_E4M3));
#else
    return make_float2(ggml_cuda_f8_e4m3_to_float({uint8_t(x)}),
                      ggml_cuda_f8_e4m3_to_float({uint8_t(x >> 8)}));
#endif
}

template <typename layout>
static __device__ __forceinline__ float dot_f8_e4m3(int n, const void * row_data, const float * y) {
    const auto * x = layout::codes(row_data);
    float sum = 0.0f;
    if (n % 2 == 0 && uintptr_t(x) % alignof(uint16_t) == 0 && uintptr_t(y) % alignof(float2) == 0) {
        const auto * y2 = (const float2 *)y;
        float2 sum2 = {};
        for (int j = threadIdx.x; j < n/2; j += 256) {
            const float2 w = ggml_cuda_f8_e4m3x2_to_float2(layout::pair(row_data, 2*j));
            const float scale = layout::dot_scale(row_data, 2*j);
            const float2 a = y2[j];
            sum2.x = fmaf(w.x * scale, a.x, sum2.x);
            sum2.y = fmaf(w.y * scale, a.y, sum2.y);
        }
        sum = sum2.x + sum2.y;
    } else {
        for (int j = threadIdx.x; j < n; j += 256) {
            sum = fmaf(ggml_cuda_f8_e4m3_to_float(layout::code(row_data, j)) * layout::dot_scale(row_data, j), y[j], sum);
        }
    }
    sum = warp_reduce_sum(sum);
    __shared__ float tmp[256/WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) tmp[warp] = sum;
    __syncthreads();
    sum = lane < 256/WARP_SIZE ? tmp[lane] : 0.0f;
    sum = warp_reduce_sum(sum);
    return sum * layout::scale(row_data);
}

template <typename layout>
static __global__ void mul_mat_vec_f8_e4m3(int n, int ldc, const ggml_fp8_e4m3_t * x,
        const float * y, float * z, size_t nb11) {
    const void * row = (const char *)x + size_t(blockIdx.x)*layout::row_size(n);
    const float sum = dot_f8_e4m3<layout>(n, row, y + size_t(blockIdx.y)*nb11);
    if (threadIdx.x == 0) z[size_t(blockIdx.y)*ldc + blockIdx.x] = sum;
}

template <int width, bool aligned = false, bool warp = false>
static __device__ __forceinline__ float dot_f8_e4m3_wide(int n, const void * row_data, const float * y) {
    const auto * x = ggml_cuda_f8_row::codes(row_data);
    const int peel = aligned ? 0 : (-uintptr_t(x)) % width;
    const int end = aligned ? n : peel + (n - peel)/width*width;
    const int thread = warp ? threadIdx.x % 32 : threadIdx.x;
    const int step = warp ? 32 : blockDim.x;
    float sums[width] = {};
    for (int j = peel + width*thread; j < end; j += width*step) {
        uint32_t words[width/4];
        if constexpr (width == 16) {
            const uint4 codes = *(const uint4 *)(x + j);
            words[0] = codes.x; words[1] = codes.y; words[2] = codes.z; words[3] = codes.w;
        } else {
            const uint2 codes = *(const uint2 *)(x + j);
            words[0] = codes.x; words[1] = codes.y;
        }
#pragma unroll
        for (int k = 0; k < width/4; ++k) {
            const float2 w0 = ggml_cuda_f8_e4m3x2_to_float2(words[k]);
            const float2 w1 = ggml_cuda_f8_e4m3x2_to_float2(words[k] >> 16);
            const float4 a = *(const float4 *)(y + j + 4*k);
            sums[4*k+0] = fmaf(w0.x, a.x, sums[4*k+0]);
            sums[4*k+1] = fmaf(w0.y, a.y, sums[4*k+1]);
            sums[4*k+2] = fmaf(w1.x, a.z, sums[4*k+2]);
            sums[4*k+3] = fmaf(w1.y, a.w, sums[4*k+3]);
        }
    }
    float sum = 0;
#pragma unroll
    for (int k = 0; k < width; ++k) sum += sums[k];
    if constexpr (!aligned) {
        if (threadIdx.x < peel) sum += ggml_cuda_f8_e4m3_to_float(x[threadIdx.x]) * y[threadIdx.x];
        const int tail = end + threadIdx.x;
        if (tail < n) sum += ggml_cuda_f8_e4m3_to_float(x[tail]) * y[tail];
    }
    sum = warp_reduce_sum(sum);
    if constexpr (!warp) {
        __shared__ float tmp[8];
        const int lane = threadIdx.x % 32;
        if (lane == 0) tmp[threadIdx.x/32] = sum;
        __syncthreads();
        sum = lane < blockDim.x/32 ? tmp[lane] : 0;
        sum = warp_reduce_sum(sum);
    }
    return sum * ggml_cuda_f8_row::scale(row_data);
}

template <int width, bool aligned = false>
static __global__ void mul_mat_vec_f8_e4m3_wide(int n, int ldc, const void * vx,
        const float * y, float * z, size_t nb11) {
    const int row = blockIdx.x;
    const void * row_data = (const char *)vx + size_t(row)*ggml_cuda_f8_row::row_size(n);
    const float sum = dot_f8_e4m3_wide<width, aligned>(n, row_data, y + size_t(blockIdx.y)*nb11);
    if (threadIdx.x == 0) z[size_t(blockIdx.y)*ldc + row] = sum;
}

template <bool wide = false, bool warp = false>
static __device__ __forceinline__ float dot_f8_e4m3_b128(int n, const void * row, const float * y) {
    const int lane = threadIdx.x % 32;
    const int first = warp ? 0 : threadIdx.x / 32;
    const int step = warp ? 1 : blockDim.x / 32;
    float sum = 0;
    if constexpr (wide) {
        const int skip = lane/8 + 1;
        for (int p = first; p < n/512; p += step) {
            const uint4 * x = (const uint4 *)row + 33*p + lane;
            const uint4 lo = x[0], hi = x[1];
            const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            const float scale = __shfl_sync(0xffffffff, __uint_as_float(words[skip - 1]), lane & ~7, 32);
            const float4 * a = (const float4 *)(y + 512*p + 16*lane);
            float partial[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 w0 = ggml_cuda_f8_e4m3x2_to_float2(words[skip + k]);
                const float2 w1 = ggml_cuda_f8_e4m3x2_to_float2(words[skip + k] >> 16);
                partial[k] = w0.x*a[k].x;
                partial[k] = fmaf(w0.y, a[k].y, partial[k]);
                partial[k] = fmaf(w1.x, a[k].z, partial[k]);
                partial[k] = fmaf(w1.y, a[k].w, partial[k]);
            }
            sum = fmaf(partial[0] + partial[1] + partial[2] + partial[3], scale, sum);
        }
    } else for (int b = first; b < n/128; b += step) {
        const void * block = (const char *)row + b*132;
        const auto * x = ggml_cuda_f8_block::codes(block);
        uint32_t codes;
        if (uintptr_t(x) % alignof(uint32_t) == 0) codes = *(const uint32_t *)(x + 4*lane);
        else memcpy(&codes, x + 4*lane, sizeof(codes));
        const float2 w0 = ggml_cuda_f8_e4m3x2_to_float2(codes);
        const float2 w1 = ggml_cuda_f8_e4m3x2_to_float2(codes >> 16);
        const float * a = y + b*128 + 4*lane;
        float partial = w0.x*a[0];
        partial = fmaf(w0.y, a[1], partial);
        partial = fmaf(w1.x, a[2], partial);
        partial = fmaf(w1.y, a[3], partial);
        sum = fmaf(partial, ggml_cuda_f8_row::scale(block), sum);
    }
    sum = warp_reduce_sum(sum);
    if constexpr (warp) return sum;
    __shared__ float tmp[8];
    if (lane == 0) tmp[first] = sum;
    __syncthreads();
    sum = lane < blockDim.x/32 ? tmp[lane] : 0;
    return warp_reduce_sum(sum);
}

template <bool wide>
static __global__ void mul_mat_vec_f8_e4m3_b128(int n, int ldc, const void * x,
        const float * y, float * z, size_t nb11) {
    const void * row = (const char *)x + size_t(blockIdx.x)*ggml_cuda_f8_block::row_size(n);
    const float sum = dot_f8_e4m3_b128<wide>(n, row, y + size_t(blockIdx.y)*nb11);
    if (threadIdx.x == 0) z[size_t(blockIdx.y)*ldc + blockIdx.x] = sum;
}

template <bool blocked, bool aligned = false>
static __global__ void mul_mat_vec_f8_e4m3_id(int n, int channels,
        const char * x, const char * y, const char * ids, char * z,
        size_t nb01, size_t nb02, size_t nb11, size_t nb12, size_t nbi0, size_t nbi1, size_t nb1, size_t nb2, int rows = 0) {
    constexpr bool warp = blocked || aligned;
    const int r = warp ? blockIdx.x*(blockDim.x/32) + threadIdx.x/32 : blockIdx.x;
    if constexpr (warp) {
        if (r >= rows) return;
    }
    const int slot = blockIdx.y;
    const int token = blockIdx.z;
    const int expert = *(const int *)(ids + size_t(token)*nbi1 + size_t(slot)*nbi0);
    const void * row = x + size_t(expert)*nb02 + size_t(r)*nb01;
    const auto * a = (const float *)(y + size_t(slot%channels)*nb11 + size_t(token)*nb12);
    float sum;
    if constexpr (blocked) sum = dot_f8_e4m3_b128<aligned, true>(n, row, a);
    else if constexpr (aligned) sum = dot_f8_e4m3_wide<16, true, true>(n, row, a);
    else sum = dot_f8_e4m3<ggml_cuda_f8_row>(n, row, a);
    if (warp ? threadIdx.x % 32 == 0 : threadIdx.x == 0) *(float *)(z + size_t(token)*nb2 + size_t(slot)*nb1 + r*sizeof(float)) = sum;
}

template <bool blocked, bool aligned = false>
static __global__ void mul_mat_vec_f8_e4m3_up_gate_id(int n, int rows, int channels,
        const char * up, const char * gate, const char * y, const char * ids,
        const char * bias_up, const char * bias_gate, float * dst, ggml_unary_op unary, float limit, float beta, float beta_linear,
        size_t nbu1, size_t nbu2, size_t nbg1, size_t nbg2, size_t nb11, size_t nb12,
        size_t nbi0, size_t nbi1, size_t nbbu1, size_t nbbg1) {
    const int slot = blockIdx.y;
    const int token = blockIdx.z;
    const int row = aligned ? blockIdx.x*(blockDim.x/32) + threadIdx.x/32 : blockIdx.x;
    if constexpr (aligned) {
        if (row >= rows) return;
    }
    const int expert = *(const int *)(ids + size_t(token)*nbi1 + size_t(slot)*nbi0);
    const size_t out = (size_t(token)*gridDim.y + slot)*rows + row;
    const auto * a = (const float *)(y + size_t(slot%channels)*nb11 + size_t(token)*nb12);
    const void * ur = up + size_t(expert)*nbu2 + size_t(row)*nbu1;
    const void * gr = gate + size_t(expert)*nbg2 + size_t(row)*nbg1;
    float su, sg;
    if constexpr (blocked) su = dot_f8_e4m3_b128<aligned, aligned>(n, ur, a);
    else if constexpr (aligned) su = dot_f8_e4m3_wide<16, true, true>(n, ur, a);
    else su = dot_f8_e4m3<ggml_cuda_f8_row>(n, ur, a);
    if constexpr (!aligned) __syncthreads();
    if constexpr (blocked) sg = dot_f8_e4m3_b128<aligned, aligned>(n, gr, a);
    else if constexpr (aligned) sg = dot_f8_e4m3_wide<16, true, true>(n, gr, a);
    else sg = dot_f8_e4m3<ggml_cuda_f8_row>(n, gr, a);
    if (aligned ? threadIdx.x % 32 == 0 : threadIdx.x == 0) {
        if (bias_up) su += ((const float *)(bias_up + size_t(expert)*nbbu1))[row];
        if (bias_gate) sg += ((const float *)(bias_gate + size_t(expert)*nbbg1))[row];
        switch (unary) {
            case GGML_UNARY_OP_SILU:
                if (limit < 1e-6f) dst[out] = sg * su / (1.0f + expf(-sg));
                else dst[out] = min(sg / (1.0f + expf(-sg)), limit) * max(-limit, min(limit, su));
                break;
            case GGML_UNARY_OP_RELU:
                dst[out] = fmaxf(sg, 0) * su;
                break;
            case GGML_UNARY_OP_GELU:
                dst[out] = 0.5f*sg*su*(1.0f + tanhf(0.79788456080286535587989211986876f*sg*(1.0f + 0.044715f*sg*sg)));
                break;
            case GGML_UNARY_OP_SIGMOID:
                dst[out] = su / (1.0f + expf(-sg));
                break;
            case GGML_UNARY_OP_SITU_GLU:
                dst[out] = situ_glu(sg, su, beta, beta_linear);
                break;
            case GGML_UNARY_OP_SWIGLU_OAI:
                sg = fminf(sg, 7.0f);
                su = fmaxf(fminf(su, 7.0f), -7.0f);
                dst[out] = (sg / (1.0f + expf(-sg * 1.702f))) * (1.0f + su);
                break;
            default: assert(false);
        }
    }
}
