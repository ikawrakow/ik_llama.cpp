#include "atsinfer-cuda.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

#if defined(GGML_USE_CUDA)
#include <cuda_runtime.h>
#endif

ATSInferCudaManager::ATSInferCudaManager()
    : device(0), compute_stream(nullptr), transfer_stream(nullptr), initialized(false) {
}

ATSInferCudaManager::~ATSInferCudaManager() {
    cleanup();
}

bool ATSInferCudaManager::init(int device_id) {
    if (initialized) return true;
    device = device_id;

#if defined(GGML_USE_CUDA)
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return false;

    cudaStream_t c_stream, t_stream;
    err = cudaStreamCreateWithFlags(&c_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) return false;

    err = cudaStreamCreateWithFlags(&t_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        cudaStreamDestroy(c_stream);
        return false;
    }

    compute_stream = (void*)c_stream;
    transfer_stream = (void*)t_stream;
    initialized = true;
    return true;
#else
    initialized = true;
    return true;
#endif
}

void ATSInferCudaManager::cleanup() {
    if (!initialized) return;

#if defined(GGML_USE_CUDA)
    if (compute_stream) {
        cudaStreamDestroy((cudaStream_t)compute_stream);
        compute_stream = nullptr;
    }
    if (transfer_stream) {
        cudaStreamDestroy((cudaStream_t)transfer_stream);
        transfer_stream = nullptr;
    }
#endif

    initialized = false;
}

void * ATSInferCudaManager::alloc_pinned_host(size_t size_bytes) {
#if defined(GGML_USE_CUDA)
    void * ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, size_bytes, cudaHostAllocDefault);
    if (err == cudaSuccess) return ptr;
#endif
    return malloc(size_bytes);
}

void ATSInferCudaManager::free_pinned_host(void * ptr) {
    if (!ptr) return;
#if defined(GGML_USE_CUDA)
    cudaError_t err = cudaFreeHost(ptr);
    if (err == cudaSuccess) return;
#endif
    free(ptr);
}

void * ATSInferCudaManager::alloc_device(size_t size_bytes) {
#if defined(GGML_USE_CUDA)
    void * ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size_bytes);
    if (err == cudaSuccess) return ptr;
#endif
    return nullptr;
}

void ATSInferCudaManager::free_device(void * ptr) {
    if (!ptr) return;
#if defined(GGML_USE_CUDA)
    cudaFree(ptr);
#endif
}

void * ATSInferCudaManager::create_event() {
#if defined(GGML_USE_CUDA)
    cudaEvent_t ev;
    cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    if (err == cudaSuccess) return (void*)ev;
#endif
    return nullptr;
}

void ATSInferCudaManager::destroy_event(void * event_handle) {
    if (!event_handle) return;
#if defined(GGML_USE_CUDA)
    cudaEventDestroy((cudaEvent_t)event_handle);
#endif
}

bool ATSInferCudaManager::migrate_h2d_async(
    void * device_dst,
    const void * host_src,
    size_t size_bytes,
    void * event_handle) {

    if (!device_dst || !host_src || size_bytes == 0) return false;

#if defined(GGML_USE_CUDA)
    cudaStream_t t_stream = transfer_stream ? (cudaStream_t)transfer_stream : 0;
    cudaError_t err = cudaMemcpyAsync(device_dst, host_src, size_bytes, cudaMemcpyHostToDevice, t_stream);
    if (err != cudaSuccess) return false;

    if (event_handle) {
        err = cudaEventRecord((cudaEvent_t)event_handle, t_stream);
        if (err != cudaSuccess) return false;
    }
    return true;
#else
    memcpy(device_dst, host_src, size_bytes);
    return true;
#endif
}

bool ATSInferCudaManager::wait_for_transfer_event(void * event_handle) {
    if (!event_handle) return true;

#if defined(GGML_USE_CUDA)
    cudaStream_t c_stream = compute_stream ? (cudaStream_t)compute_stream : 0;
    cudaError_t err = cudaStreamWaitEvent(c_stream, (cudaEvent_t)event_handle, 0);
    return (err == cudaSuccess);
#else
    return true;
#endif
}

void * ATSInferCudaManager::get_compute_stream() const {
    return compute_stream;
}

void * ATSInferCudaManager::get_transfer_stream() const {
    return transfer_stream;
}

bool ATSInferCudaManager::is_initialized() const {
    return initialized;
}
