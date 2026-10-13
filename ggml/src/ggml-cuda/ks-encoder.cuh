#pragma once

#include "common.cuh"

size_t ggml_cuda_quantize_ks(int device, ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int64_t nslice,
        const float * imatrix);

inline int64_t ggml_cuda_quantize_chunk(int64_t total_rows, size_t per_row) {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return std::max<int64_t>(1, std::min<int64_t>(total_rows, std::min<size_t>(free_mem/2, size_t(1) << 30)/per_row));
}

template <typename T>
inline T * ggml_cuda_quantize_upload(const T * src, size_t n) {
    T * dst;
    CUDA_CHECK(cudaMalloc(&dst, n*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dst, src, n*sizeof(T), cudaMemcpyHostToDevice));
    return dst;
}
