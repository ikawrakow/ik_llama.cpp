#pragma once

#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The scheduler borrows this provider until synchronized and replaced or destroyed.
// A successful begin is paired with exactly one end. A failed begin owns no session.
// gpu_rows is indexed by token * selected_experts + selection and is zeroed before plan.
// Failed dispatch/collection must quiesce writes before returning false. Successful collect makes GPU rows visible.
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

#ifdef __cplusplus
}

#ifdef GGML_USE_CUDA
// Enqueues one packed cache input copy, direct expert matvec, and actual-row output copy.
GGML_API bool ggml_cuda_moe_cache_type_supported(enum ggml_type type);
GGML_API bool ggml_cuda_moe_cache_dispatch(
        ggml_backend_t backend, enum ggml_type type,
        const void * expert_bank, const void * host_input, void * device_input,
        void * device_q8, float * device_output, float * host_output,
        size_t input_bytes, size_t output_bytes,
        int64_t k, int64_t m, size_t expert_bytes, int32_t n_hits,
        int32_t n_activations, int64_t padded_k);

GGML_API bool ggml_cuda_moe_cache_dispatch_fused(
        ggml_backend_t backend, enum ggml_type type,
        const void * expert_up, const void * expert_gate,
        const void * host_input, void * device_input, void * device_q8,
        float * device_output, float * host_output,
        size_t input_bytes, size_t output_bytes,
        int64_t k, int64_t m, size_t expert_bytes, int32_t n_hits,
        int32_t n_activations, int64_t padded_k, enum ggml_unary_op unary_op, float limit);
#endif
#endif
