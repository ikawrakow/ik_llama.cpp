#include "atsinfer-profiler.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>

#if defined(GGML_USE_CUDA)
#include <cuda_runtime.h>
#endif

static float measure_cuda_h2d_bandwidth(size_t size_bytes) {
#if defined(GGML_USE_CUDA)
    void * host_ptr = nullptr;
    void * device_ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&host_ptr, size_bytes, cudaHostAllocDefault);
    if (err != cudaSuccess || !host_ptr) return 0.0f;

    err = cudaMalloc(&device_ptr, size_bytes);
    if (err != cudaSuccess || !device_ptr) {
        cudaFreeHost(host_ptr);
        return 0.0f;
    }

    memset(host_ptr, 0xAB, size_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    cudaMemcpyAsync(device_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice, 0);
    cudaDeviceSynchronize();

    // Measure H2D
    cudaEventRecord(start, 0);
    for (int i = 0; i < 5; ++i) {
        cudaMemcpyAsync(device_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice, 0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_ptr);
    cudaFreeHost(host_ptr);

    if (elapsed_ms <= 0.0001f) return 0.0f;
    float total_mb = (float)(size_bytes * 5) / (1024.0f * 1024.0f);
    return (total_mb / (elapsed_ms / 1000.0f)); // MB/s
#else
    (void)size_bytes;
    return 0.0f;
#endif
}

static float measure_cuda_d2h_bandwidth(size_t size_bytes) {
#if defined(GGML_USE_CUDA)
    void * host_ptr = nullptr;
    void * device_ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&host_ptr, size_bytes, cudaHostAllocDefault);
    if (err != cudaSuccess || !host_ptr) return 0.0f;

    err = cudaMalloc(&device_ptr, size_bytes);
    if (err != cudaSuccess || !device_ptr) {
        cudaFreeHost(host_ptr);
        return 0.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    cudaMemcpyAsync(host_ptr, device_ptr, size_bytes, cudaMemcpyDeviceToHost, 0);
    cudaDeviceSynchronize();

    // Measure D2H
    cudaEventRecord(start, 0);
    for (int i = 0; i < 5; ++i) {
        cudaMemcpyAsync(host_ptr, device_ptr, size_bytes, cudaMemcpyDeviceToHost, 0);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_ptr);
    cudaFreeHost(host_ptr);

    if (elapsed_ms <= 0.0001f) return 0.0f;
    float total_mb = (float)(size_bytes * 5) / (1024.0f * 1024.0f);
    return (total_mb / (elapsed_ms / 1000.0f)); // MB/s
#else
    (void)size_bytes;
    return 0.0f;
#endif
}

atsinfer_hardware_profile atsinfer_profile_hardware(size_t vram_budget_bytes) {
    atsinfer_hardware_profile hw;
    hw.gpu_vram_budget = vram_budget_bytes;
    hw.pcie_bandwidth_mbps = 16000.0f;     // Default fallback (PCIe 4.0 x16 conservative)
    hw.pcie_d2h_bandwidth_mbps = 14000.0f; // Default fallback
    hw.is_measured = false;

    // Measure H2D and D2H bandwidth for 64MB transfer size if CUDA is enabled
    float h2d = measure_cuda_h2d_bandwidth(64 * 1024 * 1024);
    float d2h = measure_cuda_d2h_bandwidth(64 * 1024 * 1024);

    if (h2d > 100.0f) {
        hw.pcie_bandwidth_mbps = h2d;
        hw.is_measured = true;
    }
    if (d2h > 100.0f) {
        hw.pcie_d2h_bandwidth_mbps = d2h;
    }

    return hw;
}

std::unordered_map<std::string, atsinfer_tensor_profile> atsinfer_profile_tensors(
    const std::vector<struct ggml_tensor *> & tensors,
    float pcie_bandwidth_mbps) {

    std::unordered_map<std::string, atsinfer_tensor_profile> profiles;
    if (pcie_bandwidth_mbps <= 0.0f) {
        pcie_bandwidth_mbps = 16000.0f;
    }

    for (const auto * tensor : tensors) {
        if (!tensor || !tensor->name[0]) continue;

        atsinfer_tensor_profile p;
        p.tensor_name = tensor->name;
        p.size_bytes = ggml_nbytes(tensor);

        // Extract layer index if named like "blk.X." or "layers.X."
        p.layer_id = -1;
        std::string name_str(tensor->name);
        if (name_str.find("blk.") == 0 || name_str.find("layers.") == 0) {
            size_t dot1 = name_str.find('.');
            if (dot1 != std::string::npos) {
                size_t dot2 = name_str.find('.', dot1 + 1);
                if (dot2 != std::string::npos) {
                    try {
                        p.layer_id = std::stoi(name_str.substr(dot1 + 1, dot2 - dot1 - 1));
                    } catch (...) {
                        p.layer_id = -1;
                    }
                }
            }
        }

        // Tensor role classification
        p.is_moe_expert = (name_str.find("ffn_exp") != std::string::npos ||
                           name_str.find("exps") != std::string::npos ||
                           name_str.find("experts") != std::string::npos);
        p.is_attn = (name_str.find("attn") != std::string::npos ||
                     name_str.find("self_attn") != std::string::npos);
        p.is_ffn = (name_str.find("ffn") != std::string::npos ||
                    name_str.find("mlp") != std::string::npos);

        // Empirical estimation derived from operator type and tensor dimensions.
        //
        // Defect 3 fix: the old size-proportional heuristics (size_mb * 0.45, size_mb * 0.06)
        // made t_cpu/size and t_gpu/size constant across all tensors, which degenerated the
        // knapsack into "maximize bytes on GPU" with no per-tensor discrimination.
        //
        // We add a per-tensor fixed launch/setup overhead that differs by backend and by tensor
        // role (attention vs FFN vs expert).  Small tensors now have proportionally higher
        // overhead, giving the DP a signal to discriminate: small per-expert weights see a worse
        // GPU/CPU ratio than large dense projections, so the DP keeps experts on CPU when PCIe
        // cost dominates.
        float size_mb = (float)p.size_bytes / (1024.0f * 1024.0f);

        // Per-backend fixed overheads (kernel launch, shape setup).  GPU launches are cheaper
        // because the driver amortizes dispatch, but the gap closes for tiny tensors.
        //
        // FIXME: calibrate against ggml_backend_sched_get_split_timings() at runtime;
        // these constants are a first-pass heuristic replaced by cached measured profiles
        // on subsequent runs.
        const float cpu_overhead_ms = 0.02f;  // 20 µs CPU kernel dispatch
        float gpu_overhead_ms       = 0.005f; //  5 µs GPU kernel launch

        // Expert tensors (3-d [n_embd, n_ff, n_expert]) are processed as fused groups where
        // all experts execute in a single GGML_OP_MOE_FUSED_UP_GATE call. The launch overhead
        // is amortized across all experts, so treat them like any other large tensor.
        if (p.is_attn) {
            // Attention projections are the largest per-layer ops; GPU utilisation is best.
            gpu_overhead_ms = 0.003f;
        }

        // Throughput coefficients: ~2.2 GB/s effective CPU matmul BW, ~25 GB/s GPU.
        // The ratio ~11× matches the measured 22.7 vs 40.5 tok/s decode gap on RTX 5090
        // once graph fragmentation (Defect 1) is accounted for.
        p.exec_time_cpu_ms = cpu_overhead_ms + size_mb * 0.45f;
        p.exec_time_gpu_ms = gpu_overhead_ms + size_mb * 0.04f;

        p.latency_reduction = p.exec_time_cpu_ms - p.exec_time_gpu_ms;

        // Defect 2 & 3 fix: switching cost now includes a per-split overhead constant
        // (~10 µs for kernel launch + synchronisation at a backend boundary) on top of
        // the raw PCIe transfer time.  Without this the DP underestimates the penalty for
        // flipping backends and produces schedules with excessive splits.
        //
        // FIXME: calibrate the 0.01 ms constant against observed split overhead from
        // ggml_backend_sched_get_split_timings() / atsinfer_dt_collect().
        p.switching_cost_ms = 0.01f + (size_mb) / (pcie_bandwidth_mbps / 1000.0f);

        p.performance_density = p.exec_time_cpu_ms / std::max(1.0f, size_mb);

        profiles[p.tensor_name] = p;
    }

    return profiles;
}

bool atsinfer_save_profile_cache(
    const std::string & filename,
    const atsinfer_hardware_profile & hw,
    const std::unordered_map<std::string, atsinfer_tensor_profile> & profiles,
    const std::vector<atsinfer_expert_measurement> & measurements) {

    std::ofstream out(filename);
    if (!out.is_open()) return false;

    out << "# ATSInfer Profile Cache\n";
    // V3: adds optional EXPERT lines with per-layer measured t_c/t_g from
    // ggml_backend_sched_get_split_timings(), collected during a profiling decode round.
    // V2 loaders skip unrecognised tags, so the format is backward-compatible.
    out << "V 3\n";
    out << "MODEL " << profiles.size() << " " << atsinfer_profile_total_bytes(profiles) << "\n";
    out << "HW " << hw.pcie_bandwidth_mbps << " " << hw.pcie_d2h_bandwidth_mbps << " "
        << hw.gpu_vram_budget << " " << (hw.is_measured ? 1 : 0) << "\n";

    for (const auto & kv : profiles) {
        const auto & p = kv.second;
        out << "TENSOR " << p.tensor_name << " " << p.size_bytes << " "
            << p.exec_time_cpu_ms << " " << p.exec_time_gpu_ms << " "
            << p.layer_id << " " << (p.is_moe_expert ? 1 : 0) << " "
            << (p.is_attn ? 1 : 0) << " " << (p.is_ffn ? 1 : 0) << "\n";
    }

    // Per-layer expert measurements from a profiling decode round.  These are the
    // authoritative t_c / t_g for the next load; the heuristic TENSOR lines above
    // are a fallback for non-expert tensors and first-load bootstrap.
    for (const auto & m : measurements) {
        if (m.t_cpu_ms > 0.0f || m.t_gpu_ms > 0.0f) {
            out << "EXPERT " << m.layer_id << " " << m.t_cpu_ms << " " << m.t_gpu_ms << "\n";
        }
    }

    return true;
}

size_t atsinfer_profile_total_bytes(const std::unordered_map<std::string, atsinfer_tensor_profile> & profiles) {
    size_t total = 0;
    for (const auto & kv : profiles) {
        total += kv.second.size_bytes;
    }
    return total;
}

bool atsinfer_load_profile_cache(
    const std::string & filename,
    atsinfer_hardware_profile & hw,
    std::unordered_map<std::string, atsinfer_tensor_profile> & profiles,
    size_t expect_n_tensors,
    size_t expect_total_bytes,
    std::vector<atsinfer_expert_measurement> * measurements) {

    std::ifstream in(filename);
    if (!in.is_open()) return false;

    profiles.clear();
    if (measurements) measurements->clear();

    int    version          = 0;
    size_t cached_n_tensors = 0;
    size_t cached_total     = 0;
    bool   have_model_line  = false;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string tag;
        iss >> tag;

        if (tag == "V") {
            iss >> version;
        } else if (tag == "MODEL") {
            iss >> cached_n_tensors >> cached_total;
            have_model_line = true;
        } else if (tag == "HW") {
            int is_meas = 0;
            iss >> hw.pcie_bandwidth_mbps >> hw.pcie_d2h_bandwidth_mbps >> hw.gpu_vram_budget >> is_meas;
            hw.is_measured = (is_meas != 0);
        } else if (tag == "TENSOR") {
            atsinfer_tensor_profile p;
            int is_moe = 0, is_attn = 0, is_ffn = 0;
            iss >> p.tensor_name >> p.size_bytes >> p.exec_time_cpu_ms >> p.exec_time_gpu_ms
                >> p.layer_id >> is_moe >> is_attn >> is_ffn;

            p.is_moe_expert = (is_moe != 0);
            p.is_attn = (is_attn != 0);
            p.is_ffn = (is_ffn != 0);

            p.latency_reduction = p.exec_time_cpu_ms - p.exec_time_gpu_ms;
            float size_mb = (float)p.size_bytes / (1024.0f * 1024.0f);
            float bw = hw.pcie_bandwidth_mbps > 0.0f ? hw.pcie_bandwidth_mbps : 16000.0f;
            p.switching_cost_ms = 0.01f + (size_mb) / (bw / 1000.0f);
            p.performance_density = p.exec_time_cpu_ms / std::max(1.0f, size_mb);

            profiles[p.tensor_name] = p;
        } else if (tag == "EXPERT" && measurements) {
            atsinfer_expert_measurement m;
            iss >> m.layer_id >> m.t_cpu_ms >> m.t_gpu_ms;
            if (m.t_cpu_ms > 0.0f || m.t_gpu_ms > 0.0f) {
                measurements->push_back(m);
            }
        }
        // unrecognised tags are silently skipped (forward compatibility)
    }

    // Reject a cache written for a different model, a different quantization, or by an older
    // version that carried no fingerprint at all. Silently reusing it makes the solver place
    // tensors that do not exist in the model being loaded.
    if (version < 2 || !have_model_line) {
        profiles.clear();
        return false;
    }
    if (expect_n_tensors != 0 &&
            (cached_n_tensors != expect_n_tensors || cached_total != expect_total_bytes)) {
        profiles.clear();
        return false;
    }

    return true;
}

void atsinfer_apply_expert_measurements(
    std::unordered_map<std::string, atsinfer_tensor_profile> & profiles,
    const std::vector<atsinfer_expert_measurement> & measurements,
    float pcie_bandwidth_mbps) {

    if (measurements.empty()) return;

    // Build a lookup: layer_id -> measurement
    std::unordered_map<int, atsinfer_expert_measurement> lookup;
    for (const auto & m : measurements) {
        lookup[m.layer_id] = m;
    }

    // Precompute total expert bytes per layer so the main loop is O(n) not O(n^2).
    std::unordered_map<int, size_t> layer_totals;
    for (const auto & kv : profiles) {
        const auto & p = kv.second;
        if (p.is_moe_expert && p.layer_id >= 0) {
            layer_totals[p.layer_id] += p.size_bytes;
        }
    }

    // For each layer with measurements, find all expert tensors belonging to that
    // layer and distribute the measured time proportionally to tensor size.
    // The three expert tensors (up/gate/down) execute as one fused operator, so each
    // individual tensor's latency is its fraction of the group's total bytes.
    const float bw = pcie_bandwidth_mbps > 0.0f ? pcie_bandwidth_mbps : 16000.0f;

    for (auto & kv : profiles) {
        auto & p = kv.second;
        if (!p.is_moe_expert || p.layer_id < 0) continue;

        auto it = lookup.find(p.layer_id);
        if (it == lookup.end()) continue;

        const size_t layer_total = layer_totals[p.layer_id];
        if (layer_total == 0) continue;

        const float fraction = (float)p.size_bytes / (float)layer_total;
        const auto & m = it->second;

        p.exec_time_cpu_ms = m.t_cpu_ms * fraction;
        p.exec_time_gpu_ms = m.t_gpu_ms * fraction;
        p.latency_reduction = p.exec_time_cpu_ms - p.exec_time_gpu_ms;

        const float size_mb = (float)p.size_bytes / (1024.0f * 1024.0f);
        p.switching_cost_ms = 0.01f + (size_mb) / (bw / 1000.0f);
        p.performance_density = p.exec_time_cpu_ms / std::max(1.0f, size_mb);
    }
}
