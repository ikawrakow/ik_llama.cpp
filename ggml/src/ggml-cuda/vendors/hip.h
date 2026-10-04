#pragma once

// ROCm 6 and later provide __shfl_sync() and friends as templates that static_assert on the
// width of the mask, so the shims below have to replace them rather than shadow them
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#ifndef __HIP_BF16_H__
#include <hip/hip_bf16.h>
#endif

// Map NVIDIA bfloat16 types to ROCm equivalents for gfx900 compatibility
#if defined(__HIP_PLATFORM_AMD__)
#define __HIP_NO_HALF_CONVERSIONS__ 1
typedef __hip_bfloat16 nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;

// Implicit float conversion helpers missing from ROCm __hip_bfloat16
__device__ __forceinline__ float __bfloat162float_compat(const __hip_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ __hip_bfloat16 __float2bfloat16_compat(const float x) {
    return __float2bfloat16(x);
}
// HIP warp sync primitives
#define __shfl_sync(mask, var, srcLane, width) __shfl(var, srcLane, width)
#define __shfl_up_sync(mask, var, delta, width) __shfl_up(var, delta, width)
#define __shfl_down_sync(mask, var, delta, width) __shfl_down(var, delta, width)
#define __any_sync(mask, predicate) __any(predicate)
#define __reduce_add_sync(mask, val) (val)
#define __all_sync(mask, predicate) __all(predicate)
#define __syncwarp(mask) __builtin_amdgcn_wave_barrier()

template<typename T>
__device__ __forceinline__ float ggml_hip_to_float(const T x) { return (float)x; }
template<>
__device__ __forceinline__ float ggml_hip_to_float(const __hip_bfloat16 x) { return __bfloat162float(x); }
static inline hipError_t cudaStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags = 0) { return hipStreamWaitEvent(stream, event, flags); }
#endif // __HIP_PLATFORM_AMD__
#ifdef __HIP_PLATFORM_AMD__
// for rocblas_initialize()
#include "rocblas/rocblas.h"
#endif // __HIP_PLATFORM_AMD__
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUDA_R_16F  HIPBLAS_R_16F
#define CUDA_R_16BF HIPBLAS_R_16B
#define CUDA_R_32F  HIPBLAS_R_32F
#define CUBLAS_SIDE_RIGHT HIPBLAS_SIDE_RIGHT
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define __shfl_sync(mask, var, laneMask, width) __shfl(var, laneMask, width)
#define __shfl_up_sync(mask, var, laneMask, width) __shfl_up(var, laneMask, width)
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#define __all_sync(mask, var) __all(var)
#define __any_sync(mask, var) __any(var)
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasGemmEx hipblasGemmEx
#define cublasGemmBatchedEx hipblasGemmBatchedEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasHandle_t hipblasHandle_t
#define cublasOperation_t hipblasOperation_t
#define cublasStrsmBatched hipblasStrsmBatched
// hipBLAS has no equivalent of the cuBLAS math mode, TF32 is not a thing here
#define cublasMath_t int
#define CUBLAS_DEFAULT_MATH 0
#define cublasGetMathMode(handle, mode) (*(mode) = CUBLAS_DEFAULT_MATH, CUBLAS_STATUS_SUCCESS)
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS
#define cublasSetStream hipblasSetStream
#define cublasSgemm hipblasSgemm
#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasStatus_t hipblasStatus_t
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventDestroy hipEventDestroy
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
// cudaFuncSetAttribute() is a template over the kernel type, hipFuncSetAttribute() takes a const void *
#define cudaFuncSetAttribute(func, attr, value) hipFuncSetAttribute((const void *)(func), attr, value)
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#define cudaHostUnregister hipHostUnregister
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc hipMalloc
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define cudaSetDevice hipSetDevice
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamFireAndForget hipStreamFireAndForget
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamPerThread hipStreamPerThread
#define cudaStreamSynchronize hipStreamSynchronize
<<<<<<< HEAD
#define cudaStreamWaitEvent hipStreamWaitEvent
=======
>>>>>>> 1ba11cf6 (Add gfx900 (Vega10/MI25) HIP compatibility for ROCm builds)
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define __trap() do { abort(); __builtin_unreachable(); } while(0)
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED

// hipblasDatatype_t was deprecated in ROCm 6.5 and the GEMM entry points now take the
// dedicated compute and data types instead
#if HIP_VERSION >= 60500000
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#define cublasComputeType_t hipblasComputeType_t
#define cudaDataType_t hipDataType
#else
#define CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#define cublasComputeType_t hipblasDatatype_t
#define cudaDataType_t hipblasDatatype_t
#endif // HIP_VERSION >= 60500000

#define __CUDA_ARCH__ 1300
#ifndef GGML_USE_HIP
#define GGML_USE_HIP
#endif

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define RDNA3
#endif

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#define RDNA2
#endif

#if defined(__gfx1010__) || defined(__gfx1012__)
#define RDNA1
#endif

#ifndef __has_builtin
    #define __has_builtin(x) 0
#endif

typedef __hip_bfloat16  nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;

// hipBLAS spells a half as an unsigned short, so the pointers need a cast the cuBLAS call does not
static inline hipblasStatus_t ggml_cuda_hgemm_strided_batched(
        hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k,
        const half * alpha, const half * AP, int lda, long long strideA,
        const half * BP, int ldb, long long strideB,
        const half * beta, half * CP, int ldc, long long strideC, int batchCount) {
    return hipblasHgemmStridedBatched(handle, transA, transB, m, n, k,
            (const hipblasHalf *)alpha, (const hipblasHalf *)AP, lda, strideA,
            (const hipblasHalf *)BP, ldb, strideB,
            (const hipblasHalf *)beta, (hipblasHalf *)CP, ldc, strideC, batchCount);
}
#define cublasHgemmStridedBatched ggml_cuda_hgemm_strided_batched

typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));
static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(c);
#else
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        if(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        if(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    return reinterpret_cast<int &>(c);
#endif // __has_builtin(__builtin_elementwise_sub_sat)
}

static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    return __vsubss4(a, b);
}

static __device__ __forceinline__ unsigned int __vcmpeq4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    return c;
}

static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
    }
    return c;
}

#if defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 50600000
// __shfl_xor() for half2 was added in ROCm 5.6
static __device__ __forceinline__ half2 __shfl_xor(half2 var, int laneMask, int width) {
    typedef union half2_b32 {
        half2 val;
        int   b32;
    } half2_b32_t;
    half2_b32_t tmp;
    tmp.val = var;
    tmp.b32 = __shfl_xor(tmp.b32, laneMask, width);
    return tmp.val;
}
#endif // defined(__HIP_PLATFORM_AMD__) && HIP_VERSION < 50600000
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define CUDA_R_16BF HIPBLAS_R_16B
__device__ __forceinline__ int __dp4a(int a, int b, int c) {
    // gfx900 fallback - no dot product hardware
    const int8_t* a8 = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b8 = reinterpret_cast<const int8_t*>(&b);
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
}
