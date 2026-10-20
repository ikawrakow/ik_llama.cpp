#pragma once

#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define GGML_MOE_PROVIDER_MAX_BATCH 32
#define GGML_MOE_PROVIDER_MAX_ROWS 1024

#ifdef __cplusplus
extern "C" {
#endif

// Borrowed until scheduler synchronization, successful begin() is paired with end().
// Rows are token-major and failed dispatch/collection must quiesce writes.
struct ggml_moe_provider {
    void * user_data;
    bool (*begin)(void * user_data, const struct ggml_tensor * op, void ** request);
    bool (*plan)(void * user_data, void * request, const struct ggml_tensor * op, uint8_t * gpu_rows, size_t n_rows);
    bool (*dispatch)(void * user_data, void * request, struct ggml_tensor * op, const uint8_t * gpu_rows, size_t n_rows);
    bool (*collect)(void * user_data, void * request, struct ggml_tensor * op, const uint8_t * gpu_rows, size_t n_rows);
    void (*end)(void * user_data, void * request);
};

enum ggml_status ggml_graph_compute_with_moe_provider(
        struct ggml_cgraph * cgraph,
        struct ggml_cplan * cplan,
        const struct ggml_moe_provider * provider);

GGML_API void ggml_backend_sched_set_moe_provider(
        ggml_backend_sched_t sched,
        const struct ggml_moe_provider * provider);

#ifdef GGML_USE_CUDA
GGML_API GGML_CALL bool ggml_cuda_moe_cache_try_tensor_set_async(ggml_backend_t backend,
        struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
GGML_API GGML_CALL bool ggml_cuda_moe_cache_try_synchronize(ggml_backend_t backend);
#endif

#ifdef __cplusplus
}

#ifdef GGML_USE_CUDA
GGML_API bool ggml_cuda_moe_cache_type_supported(enum ggml_type type);
GGML_API bool ggml_cuda_moe_cache_dispatch(
        ggml_backend_t backend, enum ggml_type type,
        const void * expert_bank, const void * host_input, void * device_input,
        void * device_q8, float * device_output, float * host_output,
        size_t input_bytes, size_t output_bytes,
        int64_t k, int64_t m, size_t expert_bytes, int32_t n_hits,
        int32_t n_activations, int64_t padded_k,
        const void * expert_gate, enum ggml_unary_op unary_op, float limit);
#endif
#endif
