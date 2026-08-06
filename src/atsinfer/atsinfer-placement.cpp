#include "atsinfer-placement.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <limits>

// Result of one device's knapsack: which input items were selected for the GPU.
struct atsinfer_knapsack_result {
    std::vector<char> on_gpu;           // per input item: 1 = GPU, 0 = CPU
    size_t vram_used_bytes    = 0;
    float  expected_latency_ms = 0.0f;
};

static atsinfer_knapsack_result solve_knapsack_dp(
    const std::vector<atsinfer_tensor_profile> & tensors,
    size_t vram_budget_bytes) {

    atsinfer_knapsack_result result;
    result.on_gpu.assign(tensors.size(), 0);

    size_t n = tensors.size();
    if (n == 0) return result;

    // Discretize budget to MB to keep DP memory and time complexity O(n M)
    constexpr size_t MB = 1024 * 1024;
    size_t budget_mb = vram_budget_bytes / MB;

    // State dp[i][w][last_backend]: Max reduction in latency
    // last_backend: 0 = CPU, 1 = GPU
    std::vector<std::vector<std::vector<float>>> dp(
        n + 1, std::vector<std::vector<float>>(budget_mb + 1, std::vector<float>(2, -1e9f)));

    std::vector<std::vector<std::vector<int>>> parent_w(
        n + 1, std::vector<std::vector<int>>(budget_mb + 1, std::vector<int>(2, 0)));
    std::vector<std::vector<std::vector<int>>> parent_b(
        n + 1, std::vector<std::vector<int>>(budget_mb + 1, std::vector<int>(2, 0)));

    // Base case: 0 items
    dp[0][0][0] = 0.0f; // Start on CPU
    dp[0][0][1] = 0.0f; // Start on GPU

    for (size_t i = 1; i <= n; ++i) {
        const auto & t = tensors[i - 1];
        size_t size_mb = (t.size_bytes + MB - 1) / MB;
        float r_i = t.latency_reduction;
        float c_i = t.switching_cost_ms;

        for (size_t w = 0; w <= budget_mb; ++w) {
            for (int prev_b = 0; prev_b < 2; ++prev_b) {
                if (dp[i - 1][w][prev_b] < -1e8f) continue;

                // Option 1: Place on CPU (curr_b = 0)
                float penalty_cpu = (prev_b == 1) ? c_i : 0.0f;
                float score_cpu = dp[i - 1][w][prev_b] - penalty_cpu;
                if (score_cpu > dp[i][w][0]) {
                    dp[i][w][0] = score_cpu;
                    parent_w[i][w][0] = (int)w;
                    parent_b[i][w][0] = prev_b;
                }

                // Option 2: Place on GPU (curr_b = 1)
                if (w + size_mb <= budget_mb) {
                    float penalty_gpu = (prev_b == 0) ? c_i : 0.0f;
                    float score_gpu = dp[i - 1][w][prev_b] + r_i - penalty_gpu;
                    size_t next_w = w + size_mb;
                    if (score_gpu > dp[i][next_w][1]) {
                        dp[i][next_w][1] = score_gpu;
                        parent_w[i][next_w][1] = (int)w;
                        parent_b[i][next_w][1] = prev_b;
                    }
                }
            }
        }
    }

    // Find best final state
    float best_score = -1e9f;
    size_t best_w = 0;
    int best_b = 0;

    for (size_t w = 0; w <= budget_mb; ++w) {
        for (int b = 0; b < 2; ++b) {
            if (dp[n][w][b] > best_score) {
                best_score = dp[n][w][b];
                best_w = w;
                best_b = b;
            }
        }
    }

    // Backtrack to recover decisions
    size_t curr_w = best_w;
    int curr_b = best_b;
    for (size_t i = n; i >= 1; --i) {
        const auto & t = tensors[i - 1];
        result.on_gpu[i - 1] = (curr_b == 1) ? 1 : 0;
        if (curr_b == 1) {
            result.vram_used_bytes += t.size_bytes;
        }

        int prev_w = parent_w[i][curr_w][curr_b];
        int prev_b = parent_b[i][curr_w][curr_b];
        curr_w = (size_t)prev_w;
        curr_b = prev_b;
    }

    result.expected_latency_ms = best_score;
    return result;
}

static bool atsinfer_is_expert_tensor(const atsinfer_tensor_profile & p) {
    // Prefer the profiler's classification: split-expert GGUFs use names such as
    // ffn_up_exp.0.weight, while merged tensors use ffn_up_exps.weight.
    return p.is_moe_expert ||
           p.tensor_name.find("ffn_exp") != std::string::npos ||
           p.tensor_name.find("exps") != std::string::npos ||
           p.tensor_name.find("expert") != std::string::npos ||
           p.tensor_name.find(".exp.") != std::string::npos;
}

// Group expert tensors (ffn_up_exps, ffn_gate_exps, ffn_down_exps) by layer so the
// DP cannot split a fused-MoE operator across backends (Defect 1). Each layer's triple
// becomes a single knapsack item with combined size, combined latency reduction, and
// combined switching cost.
static std::vector<atsinfer_tensor_profile> atsinfer_group_expert_tensors(
    const std::vector<atsinfer_tensor_profile> & t_exp) {

    // Collect expert tensors by layer_id. Layer -1 (unparseable) tensors stay individual.
    std::unordered_map<int, std::vector<atsinfer_tensor_profile>> groups;
    std::vector<atsinfer_tensor_profile> ungrouped;

    for (const auto & p : t_exp) {
        if (p.layer_id >= 0) {
            groups[p.layer_id].push_back(p);
        } else {
            ungrouped.push_back(p);
        }
    }

    // Iterate groups in ascending layer order (= graph execution order). unordered_map
    // iteration order is hash-bucket order, which for > bucket_count layers (53 on libstdc++)
    // scrambles the sequence and makes the DP's switching penalties apply over a random
    // order -- a source of non-deterministic, scattered placements on larger MoE models.
    std::vector<int> layers;
    layers.reserve(groups.size());
    for (const auto & kv : groups) {
        layers.push_back(kv.first);
    }
    std::sort(layers.begin(), layers.end());

    std::vector<atsinfer_tensor_profile> result;

    for (int layer : layers) {
        auto & members = groups[layer];
        if (members.size() <= 1) {
            // Single expert tensor for this layer (e.g., gate-only models): no grouping needed
            for (auto & p : members) {
                result.push_back(std::move(p));
            }
            continue;
        }

        // Create a combined profile for the layer's expert group.
        // MUST value-initialise: size_bytes, exec_time_cpu_ms, latency_reduction etc.
        // have no default values in the struct and would otherwise be stack garbage.
        atsinfer_tensor_profile combined{};
        combined.layer_id = layer;
        combined.is_moe_expert = true;
        combined.is_ffn = true;

        for (const auto & p : members) {
            combined.size_bytes       += p.size_bytes;
            combined.latency_reduction += p.latency_reduction;
            combined.switching_cost_ms += p.switching_cost_ms;
            combined.exec_time_cpu_ms  += p.exec_time_cpu_ms;
            combined.exec_time_gpu_ms  += p.exec_time_gpu_ms;
        }

        // The per-split overhead (ATSINFER_SPLIT_OVERHEAD_MS in atsinfer_profile_tensors)
        // is paid once per backend switch, not once per tensor. Summing N individual
        // switching costs double-counts it (N-1) times. Remove the excess.
        if (members.size() > 1) {
            combined.switching_cost_ms -= ATSINFER_SPLIT_OVERHEAD_MS * (float)(members.size() - 1);
        }

        // Name the group after the layer so placement lookup can fan out to members
        combined.tensor_name = std::string("blk.") + std::to_string(layer) + ".ffn_exps_group";

        result.push_back(std::move(combined));
    }

    for (auto & p : ungrouped) {
        result.push_back(std::move(p));
    }

    return result;
}

// Expand a grouped expert decision ("blk.N.ffn_exps_group" -> device) to the individual
// expert tensors of the layer. Ungrouped tensors are placed directly.
static void atsinfer_apply_group_decision(
    const atsinfer_tensor_profile & group,
    int device,
    const std::vector<atsinfer_tensor_profile> & t_exp,
    atsinfer_placement_decision & result) {

    if (group.tensor_name.find("ffn_exps_group") != std::string::npos) {
        // Grouping is by layer_id, not by a particular GGUF naming prefix. This also
        // handles models using "layers.N." instead of "blk.N." tensor names.
        for (const auto & p : t_exp) {
            if (p.layer_id == group.layer_id) {
                result.placement[p.tensor_name] = device;
            }
        }
    } else {
        result.placement[group.tensor_name] = device;
    }
}

atsinfer_placement_decision atsinfer_compute_static_placement_multi(
    const std::vector<atsinfer_tensor_profile> & tensor_profiles,
    const std::vector<size_t> & vram_budget_bytes_per_device,
    bool is_moe_model) {

    const size_t n_dev = vram_budget_bytes_per_device.size();

    atsinfer_placement_decision result;
    result.vram_used_per_device.assign(n_dev, 0);
    result.total_vram_used_bytes = 0;
    result.expected_total_latency_ms = 0.0f;

    // Everything starts on the CPU; the devices overwrite the entries they win.
    for (const auto & p : tensor_profiles) {
        result.placement[p.tensor_name] = ATSINFER_DEVICE_CPU;
    }

    size_t total_budget = 0;
    for (size_t b : vram_budget_bytes_per_device) {
        total_budget += b;
    }

    // When the complete profile fits in the aggregate budget, cap each solver bucket
    // to its proportional share of the model. Without this, sequential per-device
    // knapsacks put the whole model on device 0 whenever one GPU can hold it, leaving
    // the other GPU with only a few leftover tensors despite an equal budget.
    std::vector<size_t> solver_budgets = vram_budget_bytes_per_device;
    size_t total_profile_bytes = 0;
    for (const auto & p : tensor_profiles) {
        total_profile_bytes += p.size_bytes;
    }
    if (total_profile_bytes < total_budget && total_budget > 0) {
        size_t assigned = 0;
        for (size_t d = 0; d < solver_budgets.size(); ++d) {
            solver_budgets[d] = (size_t) ((long double) total_profile_bytes * vram_budget_bytes_per_device[d] / total_budget);
            assigned += solver_budgets[d];
        }
        for (size_t d = 0; assigned < total_profile_bytes && d < solver_budgets.size(); ++d) {
            if (solver_budgets[d] < vram_budget_bytes_per_device[d]) {
                ++solver_budgets[d];
                ++assigned;
            }
        }
    }

    if (!is_moe_model) {
        // Dense: one knapsack per device over the tensors the previous devices did not take.
        std::vector<char> taken(tensor_profiles.size(), 0);
        for (size_t d = 0; d < n_dev; ++d) {
            std::vector<atsinfer_tensor_profile> pool;
            std::vector<size_t> pool_idx;
            pool.reserve(tensor_profiles.size());
            pool_idx.reserve(tensor_profiles.size());
            for (size_t i = 0; i < tensor_profiles.size(); ++i) {
                if (!taken[i]) {
                    pool.push_back(tensor_profiles[i]);
                    pool_idx.push_back(i);
                }
            }
            auto k = solve_knapsack_dp(pool, solver_budgets[d]);
            for (size_t j = 0; j < pool.size(); ++j) {
                if (k.on_gpu[j]) {
                    taken[pool_idx[j]] = 1;
                    result.placement[tensor_profiles[pool_idx[j]].tensor_name] = (int)d;
                }
            }
            result.vram_used_per_device[d]   += k.vram_used_bytes;
            result.total_vram_used_bytes     += k.vram_used_bytes;
            result.expected_total_latency_ms += k.expected_latency_ms;
        }
        return result;
    }

    // MoE specific partitioning: T_nonexp vs T_exp
    std::vector<atsinfer_tensor_profile> t_nonexp;
    std::vector<atsinfer_tensor_profile> t_exp;
    size_t nonexp_size = 0;

    for (const auto & p : tensor_profiles) {
        if (atsinfer_is_expert_tensor(p)) {
            t_exp.push_back(p);
        } else {
            t_nonexp.push_back(p);
            nonexp_size += p.size_bytes;
        }
    }

    // Group expert triples by layer before running the DP (Defect 1).
    // Without grouping, the DP can place ffn_up_exps on GPU and ffn_gate_exps / ffn_down_exps
    // on CPU for the same layer, which breaks GGML_OP_MOE_FUSED_UP_GATE and inflates graph
    // splits from 44 to 64 on MoE models. Grouping also guarantees a layer's expert triple
    // lands on one device under multi-GPU.
    std::vector<atsinfer_tensor_profile> t_exp_grouped = atsinfer_group_expert_tensors(t_exp);

    // Expert groups are placed after non-experts, but any capacity left by a partial
    // non-expert fit is still useful. The old all-or-nothing branch kept every expert on
    // the CPU whenever nonexp_size exceeded the total budget, which made a large budget
    // look like it only placed attention tensors.
    auto place_experts = [&](const std::vector<size_t> & remaining) {
        std::vector<char> exp_taken(t_exp_grouped.size(), 0);
        for (size_t d = 0; d < n_dev; ++d) {
            std::vector<atsinfer_tensor_profile> pool;
            std::vector<size_t> pool_idx;
            for (size_t gi = 0; gi < t_exp_grouped.size(); ++gi) {
                if (!exp_taken[gi]) {
                    pool.push_back(t_exp_grouped[gi]);
                    pool_idx.push_back(gi);
                }
            }
            auto k = solve_knapsack_dp(pool, remaining[d]);
            for (size_t j = 0; j < pool.size(); ++j) {
                if (k.on_gpu[j]) {
                    const size_t gi = pool_idx[j];
                    exp_taken[gi] = 1;
                    atsinfer_apply_group_decision(t_exp_grouped[gi], (int)d, t_exp, result);
                }
            }
            result.vram_used_per_device[d]   += k.vram_used_bytes;
            result.total_vram_used_bytes     += k.vram_used_bytes;
            result.expected_total_latency_ms += k.expected_latency_ms;
        }
    };

    if (total_budget >= nonexp_size) {
        // Non-expert tensors fit across the devices: give them GPU priority (the same rule
        // as the single-GPU solver), placing each on the device with the most remaining
        // budget that fits. That spreads them and leaves the largest holes for the expert
        // DP. A tensor bigger than any single device stays on the CPU.
        std::vector<size_t> remaining = solver_budgets;
        for (const auto & p : t_nonexp) {
            size_t best_d = n_dev, best_rem = 0;
            for (size_t d = 0; d < n_dev; ++d) {
                if (remaining[d] >= p.size_bytes && remaining[d] > best_rem) {
                    best_d = d;
                    best_rem = remaining[d];
                }
            }
            if (best_d == n_dev) {
                continue; // oversized for every device; stays on CPU
            }
            result.placement[p.tensor_name] = (int)best_d;
            remaining[best_d] -= p.size_bytes;
            result.vram_used_per_device[best_d] += p.size_bytes;
            result.total_vram_used_bytes += p.size_bytes;
        }
        place_experts(remaining);
    } else {
        // Non-experts have priority, but do not discard residual capacity: after the
        // per-device knapsacks, place as many complete expert groups as fit in the holes.
        std::vector<char> taken(t_nonexp.size(), 0);
        std::vector<size_t> remaining = solver_budgets;
        for (size_t d = 0; d < n_dev; ++d) {
            std::vector<atsinfer_tensor_profile> pool;
            std::vector<size_t> pool_idx;
            for (size_t i = 0; i < t_nonexp.size(); ++i) {
                if (!taken[i]) {
                    pool.push_back(t_nonexp[i]);
                    pool_idx.push_back(i);
                }
            }
            auto k = solve_knapsack_dp(pool, solver_budgets[d]);
            for (size_t j = 0; j < pool.size(); ++j) {
                if (k.on_gpu[j]) {
                    taken[pool_idx[j]] = 1;
                    result.placement[t_nonexp[pool_idx[j]].tensor_name] = (int)d;
                }
            }
            result.vram_used_per_device[d]   += k.vram_used_bytes;
            result.total_vram_used_bytes     += k.vram_used_bytes;
            remaining[d] = solver_budgets[d] - k.vram_used_bytes;
            result.expected_total_latency_ms += k.expected_latency_ms;
        }
        place_experts(remaining);
    }

    return result;
}

atsinfer_placement_decision atsinfer_compute_static_placement(
    const std::vector<atsinfer_tensor_profile> & tensor_profiles,
    size_t vram_budget_bytes,
    bool is_moe_model) {

    return atsinfer_compute_static_placement_multi(
        tensor_profiles, std::vector<size_t>{ vram_budget_bytes }, is_moe_model);
}

std::unordered_map<std::string, ggml_backend_buffer_type_t> atsinfer_map_placement_to_buft(
    const atsinfer_placement_decision & decision,
    ggml_backend_buffer_type_t cpu_buft,
    const std::vector<ggml_backend_buffer_type_t> & gpu_bufts) {

    std::unordered_map<std::string, ggml_backend_buffer_type_t> result;
    for (const auto & kv : decision.placement) {
        const int dev = kv.second;
        if (dev >= 0 && (size_t)dev < gpu_bufts.size() && gpu_bufts[dev]) {
            result[kv.first] = gpu_bufts[dev];
        } else {
            result[kv.first] = cpu_buft;
        }
    }
    return result;
}
