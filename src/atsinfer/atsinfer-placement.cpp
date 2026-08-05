#include "atsinfer-placement.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <limits>

static atsinfer_placement_decision solve_knapsack_dp(
    const std::vector<atsinfer_tensor_profile> & tensors,
    size_t vram_budget_bytes) {

    atsinfer_placement_decision result;
    result.total_vram_used_bytes = 0;
    result.expected_total_latency_ms = 0.0f;

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
        ATSInferBackend backend = (curr_b == 1) ? ATSInferBackend::GPU : ATSInferBackend::CPU;
        result.placement[t.tensor_name] = backend;
        if (backend == ATSInferBackend::GPU) {
            result.total_vram_used_bytes += t.size_bytes;
        }

        int prev_w = parent_w[i][curr_w][curr_b];
        int prev_b = parent_b[i][curr_w][curr_b];
        curr_w = (size_t)prev_w;
        curr_b = prev_b;
    }

    result.expected_total_latency_ms = best_score;
    return result;
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

    std::vector<atsinfer_tensor_profile> result;

    for (auto & kv : groups) {
        auto & members = kv.second;
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
        combined.layer_id = kv.first;
        combined.is_moe_expert = true;
        combined.is_ffn = true;

        for (const auto & p : members) {
            combined.size_bytes       += p.size_bytes;
            combined.latency_reduction += p.latency_reduction;
            combined.switching_cost_ms += p.switching_cost_ms;
            combined.exec_time_cpu_ms  += p.exec_time_cpu_ms;
            combined.exec_time_gpu_ms  += p.exec_time_gpu_ms;
        }

        // The per-split overhead (0.01ms in atsinfer_profile_tensors) is paid once per
        // backend switch, not once per tensor. Summing N individual switching costs
        // double-counts it (N-1) times. Remove the excess.
        if (members.size() > 1) {
            combined.switching_cost_ms -= 0.01f * (float)(members.size() - 1);
        }

        // Name the group after the layer so placement lookup can fan out to members
        combined.tensor_name = std::string("blk.") + std::to_string(kv.first) + ".ffn_exps_group";

        result.push_back(std::move(combined));
    }

    for (auto & p : ungrouped) {
        result.push_back(std::move(p));
    }

    return result;
}

atsinfer_placement_decision atsinfer_compute_static_placement(
    const std::vector<atsinfer_tensor_profile> & tensor_profiles,
    size_t vram_budget_bytes,
    bool is_moe_model) {

    if (!is_moe_model) {
        return solve_knapsack_dp(tensor_profiles, vram_budget_bytes);
    }

    // MoE specific partitioning: T_nonexp vs T_exp
    std::vector<atsinfer_tensor_profile> t_nonexp;
    std::vector<atsinfer_tensor_profile> t_exp;
    size_t nonexp_size = 0;

    for (const auto & p : tensor_profiles) {
        if (p.tensor_name.find("exps") != std::string::npos ||
            p.tensor_name.find("expert") != std::string::npos) {
            t_exp.push_back(p);
        } else {
            t_nonexp.push_back(p);
            nonexp_size += p.size_bytes;
        }
    }

    // Group expert triples by layer before running the DP (Defect 1).
    // Without grouping, the DP can place ffn_up_exps on GPU and ffn_gate_exps / ffn_down_exps
    // on CPU for the same layer, which breaks GGML_OP_MOE_FUSED_UP_GATE and inflates graph
    // splits from 44 to 64 on MoE models.
    std::vector<atsinfer_tensor_profile> t_exp_grouped = atsinfer_group_expert_tensors(t_exp);

    atsinfer_placement_decision result;
    if (vram_budget_bytes >= nonexp_size) {
        // Non-expert tensors fit in VRAM, assign them to GPU and solve DP for experts
        for (const auto & p : t_nonexp) {
            result.placement[p.tensor_name] = ATSInferBackend::GPU;
        }
        result.total_vram_used_bytes += nonexp_size;

        size_t remaining_budget = vram_budget_bytes - nonexp_size;
        auto exp_decision = solve_knapsack_dp(t_exp_grouped, remaining_budget);

        // Fan out group decisions to individual expert tensors.
        // Group names follow the pattern "blk.N.ffn_exps_group".
        for (const auto & kv : exp_decision.placement) {
            const std::string & group_name = kv.first;
            ATSInferBackend backend = kv.second;

            // Check if this is a group key (contains "ffn_exps_group")
            if (group_name.find("ffn_exps_group") != std::string::npos) {
                // Extract layer prefix, e.g. "blk.0." from "blk.0.ffn_exps_group"
                std::string prefix = group_name;
                size_t pos = prefix.find(".ffn_exps_group");
                if (pos != std::string::npos) {
                    prefix = prefix.substr(0, pos + 1); // "blk.N."
                }
                // Fan out to all expert tensors whose name starts with this prefix.
                // Also verify the tensor actually contains an expert marker to avoid
                // accidentally matching a non-expert tensor from the same layer.
                for (const auto & p : t_exp) {
                    if (p.tensor_name.find(prefix) == 0 &&
                        (p.tensor_name.find("exps") != std::string::npos ||
                         p.tensor_name.find("expert") != std::string::npos)) {
                        result.placement[p.tensor_name] = backend;
                    }
                }
            } else {
                // Ungrouped expert tensor, place directly
                result.placement[group_name] = backend;
            }
        }
        result.total_vram_used_bytes += exp_decision.total_vram_used_bytes;
    } else {
        // Budget limited: keep experts on CPU, use DP for non-experts
        for (const auto & p : t_exp) {
            result.placement[p.tensor_name] = ATSInferBackend::CPU;
        }
        auto nonexp_decision = solve_knapsack_dp(t_nonexp, vram_budget_bytes);
        for (const auto & kv : nonexp_decision.placement) {
            result.placement[kv.first] = kv.second;
        }
        result.total_vram_used_bytes += nonexp_decision.total_vram_used_bytes;
    }

    return result;
}

std::unordered_map<std::string, ggml_backend_buffer_type_t> atsinfer_map_placement_to_buft(
    const atsinfer_placement_decision & decision,
    ggml_backend_buffer_type_t cpu_buft,
    ggml_backend_buffer_type_t gpu_buft) {

    std::unordered_map<std::string, ggml_backend_buffer_type_t> result;
    for (const auto & kv : decision.placement) {
        if (kv.second == ATSInferBackend::GPU) {
            result[kv.first] = gpu_buft ? gpu_buft : cpu_buft;
        } else {
            result[kv.first] = cpu_buft;
        }
    }
    return result;
}
