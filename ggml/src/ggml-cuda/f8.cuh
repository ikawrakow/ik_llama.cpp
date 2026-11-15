#pragma once

#include "common.cuh"
#include "quantize.cuh"

static __device__ __forceinline__ float ggml_cuda_f8_e4m3_to_float(ggml_fp8_e4m3_t x) {
    const uint32_t sign = (uint32_t)(x.bits & 0x80) << 24;
    const uint32_t mag = x.bits & 0x7f;
    if (mag == 0x7f) {
        return __uint_as_float(sign | 0x7fc00000);
    }
    if (mag < 8) {
        return __uint_as_float(sign | __float_as_uint(mag * 0.001953125f));
    }
    return __uint_as_float(sign | ((mag / 8 + 120) << 23) | ((mag & 7) << 20));
}

struct ggml_cuda_f8_row {
    static constexpr size_t row_meta_size = 16;
    static __device__ __forceinline__ float scale(const void * row, int = 0) {
        float value;
        memcpy(&value, row, sizeof(value));
        return value;
    }
    static __device__ __forceinline__ const ggml_fp8_e4m3_t * codes(const void * row) {
        return (const ggml_fp8_e4m3_t *)((const char *)row + row_meta_size);
    }
    static __device__ __forceinline__ ggml_fp8_e4m3_t code(const void * row, int col) { return codes(row)[col]; }
    static __device__ __forceinline__ uint16_t pair(const void * row, int col) {
        return *(const uint16_t *)(codes(row) + col);
    }
    static __device__ __forceinline__ float dot_scale(const void *, int) { return 1.0f; }
    static __host__ __device__ size_t row_size(int n) { return row_meta_size + n; }
};

struct ggml_cuda_f8_block {
    static __device__ __forceinline__ const ggml_fp8_e4m3_t * codes(const void * block) {
        return (const ggml_fp8_e4m3_t *)((const char *)block + sizeof(float));
    }
    static __device__ __forceinline__ float scale(const void * row, int col) {
        return ggml_cuda_f8_row::scale((const char *)row + col/128*132);
    }
    static __device__ __forceinline__ ggml_fp8_e4m3_t code(const void * row, int col) {
        return codes((const char *)row + col/128*132)[col%128];
    }
    static __host__ __device__ size_t row_size(int n) { return size_t(n/128)*132; }
};

template <typename layout>
static __device__ __forceinline__ float ggml_cuda_f8_value(const void * row, int col) {
    return ggml_cuda_f8_e4m3_to_float(layout::code(row, col)) * layout::scale(row, col);
}

template <typename layout>
static __global__ void convert_f8_rows(const void * vx, float * y, int64_t n, int ne0) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const void * row = (const char *)vx + (i/ne0)*layout::row_size(ne0);
    y[i] = ggml_cuda_f8_e4m3_to_float(layout::code(row, i % ne0)) * layout::scale(row, i % ne0);
}
