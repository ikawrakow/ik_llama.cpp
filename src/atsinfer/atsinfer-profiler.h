#ifndef ATSINFER_PROFILER_H
#define ATSINFER_PROFILER_H

#include "ggml.h"
#include <string>
#include <vector>
#include <unordered_map>

// Per-backend-flip penalty charged by the static placement DP (Algoritmo 1). It models the
// graph-split overhead at a backend boundary (kernel launch + synchronization), NOT the
// weight transfer time. A transfer is a one-time load cost and in decode the host-resident
// weights are re-read every round regardless of flips, so charging the full size/B_pcie here
// made r_i < c_i for every large tensor and the DP degenerated to "everything on CPU" -- on
// a 122B MoE that left 26/31 GiB of VRAM empty and all expert layers in system RAM.
inline constexpr float ATSINFER_SPLIT_OVERHEAD_MS = 0.02f; // ~20 us kernel launch + sync

struct atsinfer_tensor_profile {
    std::string tensor_name;
    size_t size_bytes         = 0;     // s_i
    float exec_time_cpu_ms    = 0.0f;  // t_i^c
    float exec_time_gpu_ms    = 0.0f;  // t_i^g
    float latency_reduction   = 0.0f;  // r_i = t_i^c - t_i^g
    float switching_cost_ms   = 0.0f;  // c_i = graph-split overhead per backend flip
    float performance_density = 0.0f;  // k_i^b = t_i^b / s_i

    // Classification metadata
    int layer_id = -1;         // -1 if global, >= 0 for transformer layer
    bool is_moe_expert = false;
    bool is_attn = false;
    bool is_ffn = false;
};

struct atsinfer_hardware_profile {
    float pcie_bandwidth_mbps;     // B_pcie Host-to-Device in MB/s
    float pcie_d2h_bandwidth_mbps; // Device-to-Host in MB/s
    size_t gpu_vram_budget;        // M in bytes
    bool is_measured = false;      // True if dynamic CUDA profiling was executed
};

// Per-layer expert measurements collected from ggml_backend_sched_get_split_timings()
// during a profiling decode round.  One entry per MoE layer whose expert group timings
// were observed; the DP groups expert tensors by layer, so per-layer timing is the
// natural granularity for replacing heuristics with real data.
struct atsinfer_expert_measurement {
    int   layer_id = -1;
    float t_cpu_ms = 0.0f;
    float t_gpu_ms = 0.0f;
};

// Profile Host-to-Device transfer speed with pinned memory if CUDA is enabled
atsinfer_hardware_profile atsinfer_profile_hardware(size_t vram_budget_bytes);

// Profile operator latencies for model tensors
std::unordered_map<std::string, atsinfer_tensor_profile> atsinfer_profile_tensors(
    const std::vector<struct ggml_tensor *> & tensors,
    float pcie_bandwidth_mbps
);

// Cache serialization / deserialization.
// V3 adds optional per-layer expert measurements (EXPERT lines) collected from
// ggml_backend_sched_get_split_timings() via atsinfer_dt_collect().  When present,
// the caller should apply them with atsinfer_apply_expert_measurements() before
// running the DP solver.
bool atsinfer_save_profile_cache(
    const std::string & filename,
    const atsinfer_hardware_profile & hw,
    const std::unordered_map<std::string, atsinfer_tensor_profile> & profiles,
    const std::vector<atsinfer_expert_measurement> & measurements = {}
);

// Total footprint of a profile set, used as part of the cache's model fingerprint
size_t atsinfer_profile_total_bytes(const std::unordered_map<std::string, atsinfer_tensor_profile> & profiles);

// Returns false when the cache is missing, was written by an older version, or was written for a
// different model. Pass expect_n_tensors = 0 to skip the model check (tests only) -- in the loader
// it must be supplied, otherwise a profile from another model is applied silently.
//
// When the cache contains V3 EXPERT measurement lines they populate 'measurements' (if non-null).
bool atsinfer_load_profile_cache(
    const std::string & filename,
    atsinfer_hardware_profile & hw,
    std::unordered_map<std::string, atsinfer_tensor_profile> & profiles,
    size_t expect_n_tensors = 0,
    size_t expect_total_bytes = 0,
    std::vector<atsinfer_expert_measurement> * measurements = nullptr
);

// Apply per-layer expert measurements to the individual expert tensor profiles.
// Each expert tensor in a layer shares the measured t_cpu / t_gpu proportionally
// to its size within the layer's expert group.  Call before the DP solver when
// a V3 cache with measurements was loaded.
void atsinfer_apply_expert_measurements(
    std::unordered_map<std::string, atsinfer_tensor_profile> & profiles,
    const std::vector<atsinfer_expert_measurement> & measurements,
    float pcie_bandwidth_mbps
);

#endif // ATSINFER_PROFILER_H
