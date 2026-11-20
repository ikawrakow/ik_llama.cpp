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

// Stage 2a NUMA expert band placement: deferred pinning of CUDA_Host buffers.
// While ggml_host_pin_defer(true) is set, CUDA host allocations come up as
// plain anonymous mmap (unpinned, unfaulted) so the caller can mbind page
// ranges to NUMA nodes before any page exists. Call ggml_host_pin_commit()
// once placement is done (before any transient CUDA_Host allocation runs -
// e.g. before loading weights): it prefaults every page through the band
// mbind policy and registers (pins) the regions in place - pinned speed with
// node-local pages. Weight bytes then write into the already-pinned pages.
// Regions freed inside the defer window are discarded and never registered.
// Returns the number of bytes successfully registered. Linux only; no-ops
// elsewhere. GGML_CUDA_NO_NUMA_PIN=1 or GGML_NUMA_NO_DEFER_PIN=1 restores
// the stock allocate-and-pin-immediately behavior.
GGML_API void   ggml_host_pin_defer(bool enable);
GGML_API size_t ggml_host_pin_commit(void);

GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
GGML_API GGML_CALL bool ggml_backend_cuda_is_integrated(int device);

GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_API GGML_CALL void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);

GGML_API GGML_CALL void ggml_backend_cuda_invalidate_graphs(const void * model);

GGML_API GGML_CALL size_t ggml_cuda_quantize(int device, enum ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int64_t nslice, const float * imatrix);

GGML_CALL bool ggml_backend_cuda_copy_all_inputs_to_cpu(ggml_backend_sched_t sched, ggml_backend_t backend, int n_inputs,
        ggml_tensor ** inputs, ggml_tensor ** input_copies, const ggml_tensor * node0, bool has_lru_cache);

#ifdef  __cplusplus
}
#endif
