// Synthetic two-device validation for the ATSInfer multi-GPU static solver.
//
// Runs the REAL atsinfer_compute_static_placement_multi() over the tensor profiles of a
// real model (loaded from an ATSInfer profile cache, e.g. the one the loader writes to
// CWD) with two simulated per-device budgets -- a stand-in for a 2-GPU box, since the
// solver's only hardware inputs are the budgets and the profiles. Verifies the placement
// invariants the runtime depends on and prints a per-device distribution report.
//
// Usage: test-atsinfer-multigpu [cache_path] [budget_mib ...]
//   any number of per-device budgets (defaults: 12288 16896, i.e. 12 GiB + 16.5 GiB)

#include "atsinfer/atsinfer-profiler.h"
#include "atsinfer/atsinfer-placement.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr size_t MiB = 1024ULL * 1024ULL;
static constexpr size_t GiB = 1024ULL * MiB;

static const char * dev_label(int dev) {
    if (dev < 0) return "CPU";
    static char buf[16];
    snprintf(buf, sizeof(buf), "dev%d", dev);
    return buf;
}

int main(int argc, char ** argv) {
    const char * cache_path = argc > 1 ? argv[1] : "atsinfer_profile.cache";
    std::vector<size_t> budget_mib;
    if (argc > 2) {
        for (int i = 2; i < argc; ++i) {
            budget_mib.push_back((size_t) strtoull(argv[i], nullptr, 10));
        }
    } else {
        budget_mib = { 12288, 16896 }; // 12 GiB + 16.5 GiB
    }

    atsinfer_hardware_profile hw;
    std::unordered_map<std::string, atsinfer_tensor_profile> profiles;
    std::vector<atsinfer_expert_measurement> measurements;
    if (!atsinfer_load_profile_cache(cache_path, hw, profiles, 0, 0, &measurements)) {
        printf("FAIL: could not load profile cache '%s'\n", cache_path);
        return 1;
    }

    // Mirror the loader's pipeline (llama-load-tensors.cpp): replace the cached heuristic
    // expert timings with the measured per-layer values captured from a real decode round.
    // Without this the harness solves with different inputs than production.
    if (!measurements.empty()) {
        atsinfer_apply_expert_measurements(profiles, measurements, hw.pcie_bandwidth_mbps);
    }

    // Deterministic input order (the loader iterates an unordered_map; order does not
    // change correctness, but a sorted vector keeps this harness reproducible).
    std::vector<atsinfer_tensor_profile> tensor_profiles;
    tensor_profiles.reserve(profiles.size());
    for (const auto & kv : profiles) {
        tensor_profiles.push_back(kv.second);
    }
    std::sort(tensor_profiles.begin(), tensor_profiles.end(),
            [](const atsinfer_tensor_profile & a, const atsinfer_tensor_profile & b) {
                return a.tensor_name < b.tensor_name;
            });

    size_t total_bytes = 0;
    bool is_moe = false;
    for (const auto & p : tensor_profiles) {
        total_bytes += p.size_bytes;
        if (p.is_moe_expert || p.tensor_name.find("exps") != std::string::npos ||
                p.tensor_name.find("expert") != std::string::npos) {
            is_moe = true;
        }
    }

    printf("ATSInfer multi-GPU synthetic validation\n");
    printf("  profiles : %zu tensors, %.2f GiB total%s\n", tensor_profiles.size(),
            (double) total_bytes / (double) GiB, is_moe ? " (MoE)" : " (dense)");
    printf("  cache    : %s (pcie %.0f MB/s)\n", cache_path, hw.pcie_bandwidth_mbps);
    if (!measurements.empty()) {
        printf("  expert measurements : %zu per-layer timings applied from cache\n",
                measurements.size());
    }
    printf("  budgets  :");
    for (size_t i = 0; i < budget_mib.size(); ++i) {
        printf(" dev%zu = %zu MiB (%.2f GiB)", i, budget_mib[i],
                (double)(budget_mib[i] * MiB) / (double) GiB);
    }
    printf("\n\n");

    std::vector<size_t> budgets;
    budgets.reserve(budget_mib.size());
    for (size_t b : budget_mib) budgets.push_back(b * MiB);
    auto decision = atsinfer_compute_static_placement_multi(tensor_profiles, budgets, is_moe);

    int failures = 0;

    // 1. Every tensor has a placement.
    if (decision.placement.size() == tensor_profiles.size()) {
        printf("  [OK] all %zu tensors have a placement\n", tensor_profiles.size());
    } else {
        printf("  [FAIL] %zu placements for %zu tensors\n",
                decision.placement.size(), tensor_profiles.size());
        ++failures;
    }

    // 2. Independent per-device accounting (recompute from the placement map).
    std::vector<size_t> sum_by_dev(budgets.size(), 0);
    std::vector<size_t> n_by_dev(budgets.size(), 0);
    size_t cpu_bytes = 0, n_cpu = 0;
    bool range_ok = true;
    for (const auto & p : tensor_profiles) {
        const int dev = decision.placement[p.tensor_name];
        if (dev < 0) {
            cpu_bytes += p.size_bytes;
            ++n_cpu;
        } else if ((size_t) dev < budgets.size()) {
            sum_by_dev[dev] += p.size_bytes;
            ++n_by_dev[dev];
        } else {
            printf("  [FAIL] %s placed on out-of-range device %d\n", p.tensor_name.c_str(), dev);
            range_ok = false;
        }
    }
    if (range_ok) {
        printf("  [OK] no out-of-range device indices\n");
    } else {
        ++failures;
    }

    for (size_t d = 0; d < budgets.size(); ++d) {
        if (sum_by_dev[d] <= budgets[d]) {
            printf("  [OK] device %zu: %.2f GiB used <= %.2f GiB budget\n", d,
                    (double) sum_by_dev[d] / (double) GiB, (double) budgets[d] / (double) GiB);
        } else {
            printf("  [FAIL] device %zu over budget (%.2f > %.2f GiB)\n", d,
                    (double) sum_by_dev[d] / (double) GiB, (double) budgets[d] / (double) GiB);
            ++failures;
        }
        if (decision.vram_used_per_device[d] == sum_by_dev[d]) {
            printf("  [OK] device %zu: decision accounting matches placement map\n", d);
        } else {
            printf("  [FAIL] device %zu accounting mismatch (decision %zu vs map %zu bytes)\n",
                    d, decision.vram_used_per_device[d], sum_by_dev[d]);
            ++failures;
        }
    }

    size_t total_used = 0;
    for (size_t s : sum_by_dev) total_used += s;
    if (decision.total_vram_used_bytes == total_used) {
        printf("  [OK] total VRAM used is consistent\n");
    } else {
        printf("  [FAIL] total VRAM used mismatch (decision %zu vs sum %zu)\n",
                decision.total_vram_used_bytes, total_used);
        ++failures;
    }
    if (total_used + cpu_bytes == total_bytes) {
        printf("  [OK] GPU + CPU == model total (%.2f GiB)\n", (double) total_bytes / (double) GiB);
    } else {
        printf("  [FAIL] GPU + CPU (%zu) != model total (%zu)\n", total_used + cpu_bytes, total_bytes);
        ++failures;
    }

    // 3. Expert group integrity: a layer's expert triple must be whole on one device.
    std::map<int, std::vector<std::string>> exp_by_layer;
    for (const auto & p : tensor_profiles) {
        if ((p.is_moe_expert || p.tensor_name.find("exps") != std::string::npos) &&
                p.layer_id >= 0) {
            exp_by_layer[p.layer_id].push_back(p.tensor_name);
        }
    }
    std::map<int, int> group_dev; // layer -> device
    size_t n_groups = 0;
    bool groups_ok = true;
    for (const auto & kv : exp_by_layer) {
        if (kv.second.size() < 2) continue; // single-tensor "group": trivially whole
        ++n_groups;
        const int dev = decision.placement[kv.second[0]];
        group_dev[kv.first] = dev;
        for (const auto & name : kv.second) {
            if (decision.placement[name] != dev) {
                printf("  [FAIL] expert group layer %d split: %s on %s vs %s\n", kv.first,
                        name.c_str(), dev_label(dev), dev_label(decision.placement[name]));
                groups_ok = false;
            }
        }
    }
    if (groups_ok && n_groups > 0) {
        printf("  [OK] %zu expert groups: each whole on one device\n", n_groups);
    } else if (n_groups == 0) {
        printf("  [WARN] no multi-tensor expert groups in this profile\n");
    } else {
        ++failures;
    }

    // 4. All simulated devices actually get used.
    bool used_ok = true;
    for (size_t d = 0; d < budgets.size(); ++d) {
        if (n_by_dev[d] == 0) {
            printf("  [FAIL] device %zu received no tensors\n", d);
            used_ok = false;
        }
    }
    if (used_ok) {
        printf("  [OK] all %zu devices used\n", budgets.size());
    } else {
        ++failures;
    }

    // 5. Single-GPU wrapper equivalence on this real profile set: multi with one budget
    //    (the sum of all device budgets) must equal the wrapper.
    {
        size_t sum_budget = 0;
        for (size_t b : budgets) sum_budget += b;
        const auto multi_single =
            atsinfer_compute_static_placement_multi(tensor_profiles, { sum_budget }, is_moe);
        const auto wrapper =
            atsinfer_compute_static_placement(tensor_profiles, sum_budget, is_moe);
        bool same = multi_single.placement.size() == wrapper.placement.size();
        if (same) {
            for (const auto & kv : multi_single.placement) {
                const auto it = wrapper.placement.find(kv.first);
                if (it == wrapper.placement.end() || it->second != kv.second) {
                    same = false;
                    break;
                }
            }
        }
        if (same) {
            printf("  [OK] single-GPU wrapper == multi({sum}) on this profile set\n");
        } else {
            printf("  [FAIL] single-GPU wrapper diverges from multi({sum})\n");
            ++failures;
        }
    }

    // ---- distribution report ----
    printf("\n== distribution ==\n");
    for (size_t d = 0; d < budgets.size(); ++d) {
        printf("  device %zu : %4zu tensors, %8.2f GiB  (budget %7.2f GiB, %5.1f%% used)\n", d,
                n_by_dev[d], (double) sum_by_dev[d] / (double) GiB,
                (double) budgets[d] / (double) GiB,
                budgets[d] ? 100.0 * (double) sum_by_dev[d] / (double) budgets[d] : 0.0);
    }
    printf("  cpu      : %4zu tensors, %8.2f GiB\n", n_cpu, (double) cpu_bytes / (double) GiB);
    printf("  total VRAM used : %.2f GiB\n\n", (double) total_used / (double) GiB);

    if (is_moe) {
        std::map<int, int> dev_counts;
        for (const auto & g : group_dev) ++dev_counts[g.second];
        printf("  expert groups : %zu total ->", n_groups);
        for (const auto & c : dev_counts) {
            printf(" %s %d", dev_label(c.first), c.second);
        }
        printf("\n  per-layer expert device:");
        for (const auto & g : group_dev) {
            printf(" %d=%s", g.first, dev_label(g.second));
        }
        printf("\n");
    }

    printf("\n%s (%d failure%s)\n", failures == 0 ? "VALIDATION PASSED" : "VALIDATION FAILED",
            failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
