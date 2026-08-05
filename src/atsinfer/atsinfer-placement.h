#ifndef ATSINFER_PLACEMENT_H
#define ATSINFER_PLACEMENT_H

#include "atsinfer-profiler.h"
#include "ggml-backend.h"
#include <vector>
#include <string>
#include <unordered_map>

enum class ATSInferBackend {
    CPU,
    GPU
};

struct atsinfer_placement_decision {
    std::unordered_map<std::string, ATSInferBackend> placement;
    // must be initialized here: atsinfer_compute_static_placement() default-initializes
    // its local `result` and then accumulates into these with +=, which reads an
    // indeterminate value (UB). Observed as "VRAM used: 7677182227.62 GiB" in the loader log.
    size_t total_vram_used_bytes   = 0;
    float  expected_total_latency_ms = 0.0f;
};

// Algoritmo 1: Static Tensor Placement Solver (Dense & MoE)
atsinfer_placement_decision atsinfer_compute_static_placement(
    const std::vector<atsinfer_tensor_profile> & tensor_profiles,
    size_t vram_budget_bytes,
    bool is_moe_model
);

// Map static placement decision to backend buffer type per tensor
std::unordered_map<std::string, ggml_backend_buffer_type_t> atsinfer_map_placement_to_buft(
    const atsinfer_placement_decision & decision,
    ggml_backend_buffer_type_t cpu_buft,
    ggml_backend_buffer_type_t gpu_buft
);

#endif // ATSINFER_PLACEMENT_H
