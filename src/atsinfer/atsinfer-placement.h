#ifndef ATSINFER_PLACEMENT_H
#define ATSINFER_PLACEMENT_H

#include "atsinfer-profiler.h"
#include "ggml-backend.h"
#include <vector>
#include <string>
#include <unordered_map>

// Device index a tensor is placed on: -1 = CPU (pinned host memory), >= 0 = GPU device
// index into the budgets / bufts arrays passed to the solver.
constexpr int ATSINFER_DEVICE_CPU = -1;

struct atsinfer_placement_decision {
    // tensor name -> device index (ATSINFER_DEVICE_CPU or a GPU device index)
    std::unordered_map<std::string, int> placement;
    // bytes placed per GPU device, index-aligned with the solver's budget vector
    std::vector<size_t> vram_used_per_device;
    // must be initialized here: atsinfer_compute_static_placement() default-initializes
    // its local `result` and then accumulates into these with +=, which reads an
    // indeterminate value (UB). Observed as "VRAM used: 7677182227.62 GiB" in the loader log.
    size_t total_vram_used_bytes   = 0;
    float  expected_total_latency_ms = 0.0f;
};

// Algoritmo 1: Static Tensor Placement Solver (Dense & MoE), multi-GPU.
// One knapsack budget per GPU device. Non-expert tensors keep GPU priority and expert
// groups (a layer's ffn_up/gate/down_exps) are always placed whole on one device, so a
// single-device budget vector reduces exactly to the original single-GPU solver.
//
// Devices are filled sequentially: the knapsack for device d+1 runs over what device d
// left on the CPU, so switching costs between tensors placed on different devices are not
// modelled and expected_total_latency_ms is a sum of per-device DP scores -- an
// approximation used only as the bootstrap placement for the dynamic scheduler.
atsinfer_placement_decision atsinfer_compute_static_placement_multi(
    const std::vector<atsinfer_tensor_profile> & tensor_profiles,
    const std::vector<size_t> & vram_budget_bytes_per_device,
    bool is_moe_model
);

// Single-GPU convenience wrapper over atsinfer_compute_static_placement_multi().
atsinfer_placement_decision atsinfer_compute_static_placement(
    const std::vector<atsinfer_tensor_profile> & tensor_profiles,
    size_t vram_budget_bytes,
    bool is_moe_model
);

// Map static placement decision to backend buffer type per tensor.
// gpu_bufts[i] is the buffer type of GPU device i (index-aligned with the solver's
// budgets). Tensors placed on a device without a buft fall back to cpu_buft.
std::unordered_map<std::string, ggml_backend_buffer_type_t> atsinfer_map_placement_to_buft(
    const atsinfer_placement_decision & decision,
    ggml_backend_buffer_type_t cpu_buft,
    const std::vector<ggml_backend_buffer_type_t> & gpu_bufts
);

#endif // ATSINFER_PLACEMENT_H
