#include "atsinfer/atsinfer-profiler.h"
#include "atsinfer/atsinfer-placement.h"
#include "atsinfer/atsinfer-scheduler.h"
#include "atsinfer/atsinfer-cache.h"
#include "atsinfer/atsinfer-cuda.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_tensor_cache() {
    std::cout << "[TEST] Running Tensor Cache & Eviction Test..." << std::endl;
    size_t budget = 1000 * 1024 * 1024; // 1000 MB VRAM budget
    ATSInferTensorCache cache(budget);

    cache.register_tensor("layer.0.w", 400 * 1024 * 1024, 0, false, false, ATSInferResidency::CPU_AND_GPU);
    cache.register_tensor("layer.1.w", 400 * 1024 * 1024, 1, false, false, ATSInferResidency::CPU_AND_GPU);
    cache.register_tensor("layer.2.w", 400 * 1024 * 1024, 2, false, false, ATSInferResidency::CPU_ONLY);

    cache.update_usage("layer.0.w", 100);
    cache.update_usage("layer.1.w", 200);

    std::vector<std::string> evicted;
    bool reserved = cache.reserve_gpu_space("layer.2.w", 400 * 1024 * 1024, 2, evicted);
    assert(reserved);
    assert(evicted.size() == 1);
    assert(evicted[0] == "layer.0.w"); // LRU candidate evicted

    auto * state0 = cache.get_tensor_state("layer.0.w");
    assert(state0->residency == ATSInferResidency::CPU_ONLY);
    auto * state2 = cache.get_tensor_state("layer.2.w");
    assert(state2->residency == ATSInferResidency::CPU_AND_GPU);

    std::cout << " -> Tensor Cache & Eviction Test PASSED!" << std::endl;
}

void test_cuda_manager() {
    std::cout << "[TEST] Running CUDA Manager Test..." << std::endl;
    ATSInferCudaManager cuda_mgr;
    bool inited = cuda_mgr.init(0);
    assert(inited);

    void * host_ptr = cuda_mgr.alloc_pinned_host(1024 * 1024);
    assert(host_ptr != nullptr);

    void * ev = cuda_mgr.create_event();

    bool sync_ok = cuda_mgr.wait_for_transfer_event(ev);
    assert(sync_ok);

    cuda_mgr.destroy_event(ev);
    cuda_mgr.free_pinned_host(host_ptr);
    cuda_mgr.cleanup();

    std::cout << " -> CUDA Manager Test PASSED!" << std::endl;
}

void test_profiler() {
    std::cout << "[TEST] Running Profiler Test..." << std::endl;
    auto hw = atsinfer_profile_hardware(6ULL * 1024 * 1024 * 1024); // 6GB VRAM
    assert(hw.gpu_vram_budget == 6ULL * 1024 * 1024 * 1024);
    assert(hw.pcie_bandwidth_mbps > 0.0f);

    struct ggml_tensor dummy_tensor;
    snprintf(dummy_tensor.name, sizeof(dummy_tensor.name), "blk.0.attn_q.weight");
    dummy_tensor.ne[0] = 4096;
    dummy_tensor.ne[1] = 4096;
    dummy_tensor.ne[2] = 1;
    dummy_tensor.ne[3] = 1;
    dummy_tensor.type = GGML_TYPE_F16;

    std::vector<struct ggml_tensor *> tensors = { &dummy_tensor };
    auto profiles = atsinfer_profile_tensors(tensors, hw.pcie_bandwidth_mbps);

    assert(profiles.find("blk.0.attn_q.weight") != profiles.end());
    const auto & p = profiles["blk.0.attn_q.weight"];
    assert(p.latency_reduction > 0.0f);
    assert(p.switching_cost_ms >= 0.0f);
    std::cout << " -> Profiler Test PASSED!" << std::endl;
}

void test_static_placement_dense() {
    std::cout << "[TEST] Running Static Placement Dense Test..." << std::endl;
    std::vector<atsinfer_tensor_profile> profiles;

    for (int i = 0; i < 5; ++i) {
        atsinfer_tensor_profile p;
        p.tensor_name = "layer." + std::to_string(i) + ".weight";
        p.size_bytes = 500 * 1024 * 1024; // 500 MB
        p.exec_time_cpu_ms = 20.0f;
        p.exec_time_gpu_ms = 2.0f;
        p.latency_reduction = 18.0f;
        p.switching_cost_ms = 1.0f;
        profiles.push_back(p);
    }

    size_t budget = 1200 * 1024 * 1024; // 1.2 GB VRAM (Fits 2 tensors of 500 MB)
    auto decision = atsinfer_compute_static_placement(profiles, budget, false);

    size_t gpu_count = 0;
    for (const auto & kv : decision.placement) {
        if (kv.second >= 0) { // device index >= 0 means a GPU
            gpu_count++;
        }
    }

    assert(gpu_count <= 2);
    assert(decision.total_vram_used_bytes <= budget);
    std::cout << " -> Static Placement Dense Test PASSED! (Allocated " << gpu_count << " tensors on GPU)" << std::endl;
}

void test_static_placement_moe() {
    std::cout << "[TEST] Running Static Placement MoE Test..." << std::endl;
    std::vector<atsinfer_tensor_profile> profiles;

    // Non-expert tensor
    atsinfer_tensor_profile p_attn;
    p_attn.tensor_name = "blk.0.attn_q.weight";
    p_attn.size_bytes = 200 * 1024 * 1024;
    p_attn.exec_time_cpu_ms = 15.0f;
    p_attn.exec_time_gpu_ms = 1.0f;
    p_attn.latency_reduction = 14.0f;
    p_attn.switching_cost_ms = 0.5f;
    profiles.push_back(p_attn);

    // Expert tensor
    atsinfer_tensor_profile p_exp;
    p_exp.tensor_name = "blk.0.exps.0.weight";
    p_exp.size_bytes = 400 * 1024 * 1024;
    p_exp.exec_time_cpu_ms = 30.0f;
    p_exp.exec_time_gpu_ms = 3.0f;
    p_exp.latency_reduction = 27.0f;
    p_exp.switching_cost_ms = 1.0f;
    profiles.push_back(p_exp);

    size_t budget = 500 * 1024 * 1024; // 500 MB
    auto decision = atsinfer_compute_static_placement(profiles, budget, true);

    assert(decision.placement["blk.0.attn_q.weight"] >= 0); // non-experts get GPU priority

    // Budget too small for the non-experts (100 < 200): the priority rule keeps ALL experts
    // on the CPU and spends the budget on the non-experts (which do not fit either here).
    {
        auto small = atsinfer_compute_static_placement(profiles, 100 * 1024 * 1024, true);
        assert(small.placement["blk.0.exps.0.weight"] == ATSINFER_DEVICE_CPU);
        assert(small.placement["blk.0.attn_q.weight"] == ATSINFER_DEVICE_CPU);
        assert(small.total_vram_used_bytes <= 100 * 1024 * 1024);
    }

    std::cout << " -> Static Placement MoE Test PASSED!" << std::endl;
}

void test_static_placement_multi_device() {
    std::cout << "[TEST] Running Static Placement Multi-GPU Test..." << std::endl;
    constexpr size_t MB = 1024ULL * 1024ULL;

    std::vector<atsinfer_tensor_profile> profiles;

    // 4 MoE layers. Each layer: an expert triple (150+150+100 = 400 MB group) and a
    // 100 MB attention tensor (non-expert) => nonexp_size = 400 MB total.
    const char * exp_types[3] = { "ffn_up_exps", "ffn_gate_exps", "ffn_down_exps" };
    const size_t exp_sizes[3] = { 150 * MB, 150 * MB, 100 * MB };
    for (int l = 0; l < 4; ++l) {
        for (int e = 0; e < 3; ++e) {
            atsinfer_tensor_profile p;
            p.tensor_name = "blk." + std::to_string(l) + "." + exp_types[e] + ".weight";
            p.size_bytes = exp_sizes[e];
            p.exec_time_cpu_ms = 30.0f;
            p.exec_time_gpu_ms = 3.0f;
            p.latency_reduction = 27.0f;
            p.switching_cost_ms = 0.5f;
            p.layer_id = l;
            p.is_moe_expert = true;
            profiles.push_back(p);
        }
        atsinfer_tensor_profile a;
        a.tensor_name = "blk." + std::to_string(l) + ".attn_q.weight";
        a.size_bytes = 100 * MB;
        a.exec_time_cpu_ms = 15.0f;
        a.exec_time_gpu_ms = 1.0f;
        a.latency_reduction = 14.0f;
        a.switching_cost_ms = 0.5f;
        a.layer_id = l;
        profiles.push_back(a);
    }

    // Two 1000 MB devices. Total 2000 >= nonexp 400 -> all non-experts on GPU, expert
    // groups placed by per-device knapsack (2 groups fit on each device).
    std::vector<size_t> budgets = { 1000 * MB, 1000 * MB };
    auto decision = atsinfer_compute_static_placement_multi(profiles, budgets, true);

    // Every tensor is accounted for.
    assert(decision.placement.size() == profiles.size());

    // An expert triple is never split across devices.
    for (int l = 0; l < 4; ++l) {
        const int dev = decision.placement["blk." + std::to_string(l) + "." + exp_types[0] + ".weight"];
        for (int e = 1; e < 3; ++e) {
            assert(decision.placement["blk." + std::to_string(l) + "." + exp_types[e] + ".weight"] == dev);
        }
    }

    // Per-device budgets are respected.
    assert(decision.vram_used_per_device.size() == 2);
    assert(decision.vram_used_per_device[0] <= budgets[0]);
    assert(decision.vram_used_per_device[1] <= budgets[1]);
    assert(decision.total_vram_used_bytes <= budgets[0] + budgets[1]);

    // Both GPUs actually get expert groups.
    int on_dev0 = 0, on_dev1 = 0;
    for (int l = 0; l < 4; ++l) {
        const int dev = decision.placement["blk." + std::to_string(l) + ".ffn_up_exps.weight"];
        if (dev == 0) ++on_dev0;
        if (dev == 1) ++on_dev1;
    }
    assert(on_dev0 >= 1 && on_dev1 >= 1);

    // Scenario 2: budgets too small for the non-experts (total 200 < 400) -> ALL experts
    // stay on the CPU and only non-experts get the GPU (single-GPU priority rule).
    std::vector<size_t> small_budgets = { 100 * MB, 100 * MB };
    auto small = atsinfer_compute_static_placement_multi(profiles, small_budgets, true);
    for (int l = 0; l < 4; ++l) {
        for (int e = 0; e < 3; ++e) {
            assert(small.placement["blk." + std::to_string(l) + "." + exp_types[e] + ".weight"] == ATSINFER_DEVICE_CPU);
        }
    }
    size_t attn_on_gpu = 0;
    for (int l = 0; l < 4; ++l) {
        if (small.placement["blk." + std::to_string(l) + ".attn_q.weight"] >= 0) ++attn_on_gpu;
    }
    assert(attn_on_gpu >= 1);
    assert(attn_on_gpu <= 2); // one 100 MB budget per device: at most one non-expert each

    // Scenario 3: when non-experts do not all fit, residual capacity must still accept
    // complete expert groups. The old all-or-nothing branch left every expert on CPU.
    {
        std::vector<atsinfer_tensor_profile> partial;
        for (int i = 0; i < 4; ++i) {
            atsinfer_tensor_profile a;
            a.tensor_name = "blk." + std::to_string(i) + ".attn_q.weight";
            a.size_bytes = 120 * MB;
            a.exec_time_cpu_ms = 10.0f;
            a.exec_time_gpu_ms = 1.0f;
            a.latency_reduction = 9.0f;
            a.switching_cost_ms = 0.5f;
            a.layer_id = i;
            partial.push_back(a);
        }
        atsinfer_tensor_profile e;
        e.tensor_name = "layers.0.ffn_up_exp.0.weight";
        e.size_bytes = 60 * MB;
        e.exec_time_cpu_ms = 10.0f;
        e.exec_time_gpu_ms = 1.0f;
        e.latency_reduction = 9.0f;
        e.switching_cost_ms = 0.5f;
        e.layer_id = 0;
        // Exercise name-based classification as a cache/import fallback.
        e.is_moe_expert = false;
        partial.push_back(e);

        // 4*120 MB of non-experts > 2*200 MB total, but each device has 80 MB
        // left after taking one attention tensor, enough for the 60 MB expert.
        auto partial_dec = atsinfer_compute_static_placement_multi(
                partial, { 200 * MB, 200 * MB }, true);
        assert(partial_dec.placement[e.tensor_name] >= 0);
        assert(partial_dec.total_vram_used_bytes <= 400 * MB);
    }

    // Scenario 4: dense model across two devices.
    std::vector<atsinfer_tensor_profile> dense;
    for (int i = 0; i < 5; ++i) {
        atsinfer_tensor_profile p;
        p.tensor_name = "layer." + std::to_string(i) + ".weight";
        p.size_bytes = 500 * MB;
        p.exec_time_cpu_ms = 20.0f;
        p.exec_time_gpu_ms = 2.0f;
        p.latency_reduction = 18.0f;
        p.switching_cost_ms = 1.0f;
        dense.push_back(p);
    }
    auto dense_dec = atsinfer_compute_static_placement_multi(dense, { 600 * MB, 1200 * MB }, false);
    assert(dense_dec.placement.size() == 5);
    size_t d0 = 0, d1 = 0;
    for (const auto & kv : dense_dec.placement) {
        if (kv.second == 0) ++d0;
        if (kv.second == 1) ++d1;
    }
    assert(d0 == 1 && d1 == 2); // 500 fits 600; 2x500 fits 1200; the rest stays on CPU

    std::cout << " -> Static Placement Multi-GPU Test PASSED!" << std::endl;
}

static atsinfer_round_unit mk_unit(int layer, bool static_gpu, float t_cpu, float t_gpu, float c, float w) {
    atsinfer_round_unit u;
    u.layer      = layer;
    u.static_gpu = static_gpu;
    u.t_cpu_ms   = t_cpu;
    u.t_gpu_ms   = t_gpu;
    u.c_ms       = c;
    u.w_ms       = w;
    return u;
}

void test_dynamic_transfer_scheduler() {
    std::cout << "[TEST] Running Dynamic Transfer Scheduler Test (Algorithm 2)..." << std::endl;

    // GPU-resident units are forced to the GPU regardless of their timings.
    {
        std::vector<atsinfer_round_unit> units = {
            mk_unit(0, true, 25.0f, 2.0f, 1.0f, 0.0f),
            mk_unit(1, true, 25.0f, 2.0f, 1.0f, 0.0f),
        };
        auto plan = atsinfer_schedule_round(units);
        assert(plan.run_on_gpu.size() == 2);
        assert(plan.run_on_gpu[0] == 1 && plan.run_on_gpu[1] == 1);
        assert(plan.n_promoted == 0);
    }

    // A lone CPU unit whose weight transfer costs more than the CPU/GPU gap stays put.
    // Nothing precedes it, so the whole transfer is exposed: 40 + 2 > 25.
    {
        std::vector<atsinfer_round_unit> units = { mk_unit(0, false, 25.0f, 2.0f, 1.0f, 40.0f) };
        auto plan = atsinfer_schedule_round(units);
        assert(plan.run_on_gpu[0] == 0);
        assert(plan.n_promoted == 0);
    }

    // Same unit, but cheap to transfer: promoting is now the better option.
    {
        std::vector<atsinfer_round_unit> units = { mk_unit(0, false, 25.0f, 2.0f, 1.0f, 3.0f) };
        auto plan = atsinfer_schedule_round(units);
        assert(plan.run_on_gpu[0] == 1);
        assert(plan.n_promoted == 1);
    }

    // The overlap window is what makes promotion pay off. Unit 2's transfer (20 ms) is
    // expensive on its own, but it can be hidden behind unit 1's 30 ms of CPU work, so the
    // exposed cost is zero and unit 2 should be promoted while unit 1 stays on the CPU.
    {
        std::vector<atsinfer_round_unit> units = {
            mk_unit(0, false, 30.0f, 3.0f, 1.0f, 100.0f), // too expensive to move
            mk_unit(1, false, 30.0f, 3.0f, 1.0f,  20.0f), // hidden behind unit 0
        };
        auto plan = atsinfer_schedule_round(units);
        assert(plan.run_on_gpu[0] == 0);
        assert(plan.run_on_gpu[1] == 1);
        assert(plan.n_promoted == 1);
        // 30 (cpu unit 0) + max(20, 0 overlap window between j=0 and i=1) + 1 (switch) + 3 (gpu)
        assert(plan.estimated_latency_ms > 0.0f);
    }

    // Degenerate input must not crash.
    {
        auto plan = atsinfer_schedule_round({});
        assert(plan.run_on_gpu.empty());
        assert(plan.n_promoted == 0);
    }

    std::cout << " -> Dynamic Transfer Scheduler Test PASSED!" << std::endl;
}

void test_load_aware_rescheduler() {
    std::cout << "[TEST] Running Load-Aware Rescheduler Test..." << std::endl;
    ATSInferRescheduler rescheduler(0.15f, 5);

    // Initial state: no plan exists yet, so schedule once unconditionally
    assert(rescheduler.should_reschedule(0.0f, 40.0f, 40.0f) == true);
    rescheduler.record_reschedule_event();

    // Small deviation (5%) - should NOT reschedule
    assert(rescheduler.should_reschedule(40.0f, 42.0f, 40.0f) == false);

    // Large deviation (25%) but not enough time elapsed (< 5 * TPOT)
    assert(rescheduler.should_reschedule(40.0f, 50.0f, 40.0f) == false);

    // Advance time beyond 5 * TPOT
    rescheduler.should_reschedule(40.0f, 40.0f, 40.0f);
    rescheduler.should_reschedule(40.0f, 40.0f, 40.0f);
    rescheduler.should_reschedule(40.0f, 40.0f, 40.0f);
    rescheduler.should_reschedule(40.0f, 40.0f, 40.0f);

    // Large deviation now - SHOULD reschedule
    assert(rescheduler.should_reschedule(40.0f, 52.0f, 40.0f) == true);
    std::cout << " -> Load-Aware Rescheduler Test PASSED!" << std::endl;
}

void test_promotion_device_selection() {
    std::cout << "[TEST] Running Promotion Device Selection Test..." << std::endl;
    constexpr size_t GiB = 1024ULL * 1024ULL * 1024ULL;
    constexpr size_t MB  = 1024ULL * 1024ULL;

    // No candidates -> nothing to promote to.
    assert(atsinfer_select_promotion_device({}, -1) == -1);

    // A single GPU with room.
    {
        std::vector<atsinfer_device_candidate> c = { { 8*GiB, 2*GiB, 0 } };
        assert(atsinfer_select_promotion_device(c, -1) == 0);
    }

    // No GPU has room for the expert group -> refuse (stay on CPU rather than OOM).
    {
        std::vector<atsinfer_device_candidate> c = {
            { 1*GiB, 2*GiB, 0 },
            { 512*MB, 2*GiB, 2 },
        };
        assert(atsinfer_select_promotion_device(c, -1) == -1);
    }

    // Picks the GPU with the most headroom after the promotion.
    {
        std::vector<atsinfer_device_candidate> c = {
            { 3*GiB,  2*GiB, 1 },  // 1 GiB headroom
            { 8*GiB,  2*GiB, 0 },  // 6 GiB headroom -> wins
        };
        assert(atsinfer_select_promotion_device(c, -1) == 1);
    }

    // Stability: keeps the previous device when it still has room, even if another
    // device now has more headroom (avoids a graph rebuild from device churn).
    {
        std::vector<atsinfer_device_candidate> c = {
            { 3*GiB, 2*GiB, 1 },
            { 8*GiB, 2*GiB, 0 },
        };
        assert(atsinfer_select_promotion_device(c, 0) == 0);
    }

    // ...but not when the previous device no longer fits.
    {
        std::vector<atsinfer_device_candidate> c = {
            { 1*GiB, 2*GiB, 1 },
            { 8*GiB, 2*GiB, 0 },
        };
        assert(atsinfer_select_promotion_device(c, 0) == 1);
    }

    // Tie-break: equal headroom -> the less-loaded device wins.
    {
        std::vector<atsinfer_device_candidate> c = {
            { 4*GiB, 1*GiB, 5 },
            { 4*GiB, 1*GiB, 1 },  // fewer layers already promoted here -> wins
        };
        assert(atsinfer_select_promotion_device(c, -1) == 1);
    }

    // Tie-break: equal headroom and equal load -> lowest index (deterministic).
    {
        std::vector<atsinfer_device_candidate> c = {
            { 4*GiB, 1*GiB, 0 },
            { 4*GiB, 1*GiB, 0 },
        };
        assert(atsinfer_select_promotion_device(c, -1) == 0);
    }

    std::cout << " -> Promotion Device Selection Test PASSED!" << std::endl;
}

void test_profile_serialization() {
    std::cout << "[TEST] Running Profile Serialization Test..." << std::endl;
    atsinfer_hardware_profile hw_orig;
    hw_orig.pcie_bandwidth_mbps = 24000.0f;
    hw_orig.pcie_d2h_bandwidth_mbps = 22000.0f;
    hw_orig.gpu_vram_budget = 8192ULL * 1024 * 1024;
    hw_orig.is_measured = true;

    std::unordered_map<std::string, atsinfer_tensor_profile> profiles_orig;
    atsinfer_tensor_profile p;
    p.tensor_name = "blk.3.attn_q.weight";
    p.size_bytes = 100 * 1024 * 1024;
    p.exec_time_cpu_ms = 12.5f;
    p.exec_time_gpu_ms = 1.2f;
    p.layer_id = 3;
    p.is_attn = true;
    profiles_orig[p.tensor_name] = p;

    std::string cache_path = "test_atsinfer_cache.txt";
    bool saved = atsinfer_save_profile_cache(cache_path, hw_orig, profiles_orig);
    assert(saved);

    atsinfer_hardware_profile hw_loaded;
    std::unordered_map<std::string, atsinfer_tensor_profile> profiles_loaded;
    bool loaded = atsinfer_load_profile_cache(cache_path, hw_loaded, profiles_loaded);
    assert(loaded);

    assert(hw_loaded.pcie_bandwidth_mbps == hw_orig.pcie_bandwidth_mbps);
    assert(hw_loaded.gpu_vram_budget == hw_orig.gpu_vram_budget);
    assert(profiles_loaded.find("blk.3.attn_q.weight") != profiles_loaded.end());
    assert(profiles_loaded["blk.3.attn_q.weight"].layer_id == 3);
    assert(profiles_loaded["blk.3.attn_q.weight"].is_attn == true);

    // A cache written for a different model must be rejected, not silently applied. This
    // actually happened: a 40-layer MoE profile was loaded for a 65-layer dense model and the
    // solver placed tensors that did not exist in it.
    {
        std::unordered_map<std::string, atsinfer_tensor_profile> wrong_model;
        atsinfer_hardware_profile hw_tmp;
        const size_t right_bytes = atsinfer_profile_total_bytes(profiles_orig);

        assert(atsinfer_load_profile_cache(cache_path, hw_tmp, wrong_model,
                    profiles_orig.size(), right_bytes) == true);

        assert(atsinfer_load_profile_cache(cache_path, hw_tmp, wrong_model,
                    profiles_orig.size() + 1, right_bytes) == false);
        assert(wrong_model.empty());

        assert(atsinfer_load_profile_cache(cache_path, hw_tmp, wrong_model,
                    profiles_orig.size(), right_bytes + 1) == false);
        assert(wrong_model.empty());
    }

    std::remove(cache_path.c_str());
    std::cout << " -> Profile Serialization Test PASSED!" << std::endl;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "    ATSInfer Unit Test Suite Execution    " << std::endl;
    std::cout << "==========================================" << std::endl;

    test_profiler();
    test_static_placement_dense();
    test_static_placement_moe();
    test_static_placement_multi_device();
    test_dynamic_transfer_scheduler();
    test_load_aware_rescheduler();
    test_profile_serialization();
    test_tensor_cache();
    test_cuda_manager();
    test_promotion_device_selection();

    std::cout << "==========================================" << std::endl;
    std::cout << "   ALL ATSINFER UNIT TESTS PASSED (10/10)  " << std::endl;
    std::cout << "==========================================" << std::endl;
    return 0;
}
