#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, const void * params, const void * model);

GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_API GGML_CALL void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);

GGML_API GGML_CALL void ggml_backend_cuda_invalidate_graphs(const void * model);

// Phase 4 expert cache promotion (M3b): a dedicated-copy-stream async HtoD
// engine bound to one CUDA backend/device. Copies are ordered after the
// compute-stream point captured by sync_compute() at queue time, so a slot can
// be overwritten while later compute is already in flight (the slot must be
// excluded from reads until the copy fence completes — enforced by the
// caller). Fence ids are monotonically increasing; polling a completed fence
// retires it and all earlier fences (one copy stream => in-order completion).
typedef struct ggml_cuda_copy_engine * ggml_cuda_copy_engine_t;

GGML_API GGML_CALL ggml_cuda_copy_engine_t ggml_backend_cuda_copy_engine_new(ggml_backend_t backend);
GGML_API GGML_CALL void                      ggml_backend_cuda_copy_engine_free(ggml_cuda_copy_engine_t engine);
// order subsequently enqueued copies after all work submitted so far on the
// backend's compute stream; call at queue time, before the job's h2d calls
GGML_API GGML_CALL void                      ggml_backend_cuda_copy_engine_sync_compute(ggml_cuda_copy_engine_t engine);
// enqueue one HtoD copy on the copy stream; dst = device pointer on the engine's device
GGML_API GGML_CALL void                      ggml_backend_cuda_copy_engine_h2d(ggml_cuda_copy_engine_t engine, void * dst, const void * src, size_t size);
// enable a pinned staging ring for h2d: the producer thread memcpy's pageable
// sources into pinned chunks first, so the copies stay true async DMA instead
// of degenerating to driver-staged synchronous copies that serialize with
// other streams' HtoD (expert-cache promotions vs input copies). Each chunk's
// memcpy is split across n_threads (producer + n_threads-1 helpers). Single
// producer thread only; size 0 keeps the legacy direct path. Call once,
// before the first h2d.
GGML_API GGML_CALL void                      ggml_backend_cuda_copy_engine_set_staging(ggml_cuda_copy_engine_t engine, size_t size, int n_threads);
// record a fence after all copies enqueued so far on the copy stream
GGML_API GGML_CALL uint64_t                  ggml_backend_cuda_copy_engine_fence_copy(ggml_cuda_copy_engine_t engine);
// true when the fence's work has completed (an already-retired id counts as complete)
GGML_API GGML_CALL bool                      ggml_backend_cuda_copy_engine_poll(ggml_cuda_copy_engine_t engine, uint64_t fence);
#ifdef  __cplusplus
}
#endif
