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

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
template <ggml_type type> struct ks_traits {
    static constexpr int nval    = type == GGML_TYPE_IQ5_KS ?  32 : type == GGML_TYPE_IQ4_KS ?  16 :   8;
    static constexpr int nindex  = type == GGML_TYPE_IQ5_KS ? 247 : type == GGML_TYPE_IQ4_KS ? 241 : 111;
    static constexpr int ntry    = type == GGML_TYPE_IQ4_KS ?   7 :   5;
    static constexpr int nmap    = type == GGML_TYPE_IQ3_KS ? 127 : 240;
    static constexpr int nsimple = type == GGML_TYPE_IQ3_KS ? 128 : 256;
};

// (int)x as cvttss2si computes it: INT_MIN for NaN and for values outside the int range
static __device__ __forceinline__ int ks_trunc(float x) {
    return x >= -2147483648.f && x < 2147483648.f ? __float2int_rz(x) : INT_MIN;
}

template <ggml_type type>
static __device__ __forceinline__ int ks_best_index(const int8_t * values, const int8_t * index, float x) {
    constexpr int nval = ks_traits<type>::nval;
    int ix = ks_trunc(x) - values[0];
    if (ix < 0 || ix >= ks_traits<type>::nindex) {
        return ix < 0 ? 0 : nval - 1;
    }
    ix = index[ix];
    return ix < nval ? ix : __fsub_rn(x, values[ix - nval]) < __fsub_rn(values[ix - nval + 1], x) ? ix - nval : ix - nval + 1;
}

const int8_t * ggml_cuda_ks_index(int device, ggml_type type, bool simple);
#endif
