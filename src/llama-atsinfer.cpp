// Load-Aware Dynamic Transfer runtime glue, section 4.4 of arXiv 2607.10183v2.
//
// Algorithm 2 (the per-round DP) and Algorithm 3 (the rate-limited re-scheduling gate) live in
// src/atsinfer/atsinfer-scheduler.cpp as pure functions. This file is the part that has to know
// about llama internals: it derives the static placement b from the loaded model, turns measured
// per-split timings into the DP's inputs, and applies the resulting rb to the next graph.

#include "llama-atsinfer.h"

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "atsinfer/atsinfer-profiler.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

// Nodes emitted by build_moe_ffn() that consume a layer's expert weights. The graph-build
// callback appends "-<il>", so a split's first node identifies both the operator and the layer.
static bool atsinfer_is_moe_expert_node(const char * name, int * layer_out) {
    static const char * prefixes[] = {
        "ffn_moe_up", "ffn_moe_gate_par", "ffn_moe_gate", "ffn_moe_down",
    };

    const char * dash = strrchr(name, '-');
    if (!dash || dash[1] == '\0') {
        return false;
    }

    const size_t base_len = (size_t) (dash - name);
    for (const char * p : prefixes) {
        const size_t plen = strlen(p);
        if (base_len == plen && strncmp(name, p, plen) == 0) {
            char * end = nullptr;
            const long il = strtol(dash + 1, &end, 10);
            if (end && *end == '\0' && il >= 0) {
                if (layer_out) *layer_out = (int) il;
                return true;
            }
        }
    }
    return false;
}

bool atsinfer_dt_init(llama_context & lctx) {
    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    lctx.atsinfer_units.clear();
    lctx.atsinfer_run_on_gpu.clear();
    lctx.atsinfer_unit_of_layer.assign(hparams.n_layer, -1);
    lctx.atsinfer_dt_active = false;

    if (hparams.n_expert == 0) {
        LLAMA_LOG_INFO("%s: ATSInfer dynamic transfer needs an MoE model; disabled\n", __func__);
        return false;
    }

    // B_pcie, measured with CUDA events during load; fall back to a PCIe 4.0 x16 estimate.
    const float bw_mbps = lctx.atsinfer_h2d_mbps > 0.0f ? lctx.atsinfer_h2d_mbps : 12000.0f;

    const int n_expert_used = hparams.n_expert_used > 0 ? hparams.n_expert_used : 1;

    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * ws[3] = { layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps };

        size_t total_bytes = 0;
        bool   any         = false;
        bool   all_host    = true;
        for (ggml_tensor * w : ws) {
            if (!w || !w->buffer) {
                continue;
            }
            any = true;
            total_bytes += ggml_nbytes(w);
            if (!ggml_backend_buffer_is_host(w->buffer)) {
                all_host = false;
            }
        }
        if (!any) {
            continue;
        }

        atsinfer_round_unit u;
        u.layer = (int) il;
        // b_i: a unit is CPU-resident only when the whole expert group lives on the host. A
        // partially resident group cannot be promoted as one, so treat it as already on the GPU.
        u.static_gpu = !all_host;

        // w_i: only the routed experts move in a decode round (section 4.4.1), not all of them.
        const double moved_bytes = (double) total_bytes * (double) n_expert_used / (double) hparams.n_expert;
        u.w_ms = (float) (moved_bytes / (1024.0 * 1024.0) / bw_mbps * 1000.0);

        lctx.atsinfer_unit_of_layer[il] = (int) lctx.atsinfer_units.size();
        lctx.atsinfer_units.push_back(u);
    }

    size_t n_cpu_units = 0;
    for (const auto & u : lctx.atsinfer_units) {
        if (!u.static_gpu) ++n_cpu_units;
    }

    if (lctx.atsinfer_units.empty() || n_cpu_units == 0) {
        LLAMA_LOG_INFO("%s: ATSInfer dynamic transfer found no host-resident expert layers; disabled\n", __func__);
        return false;
    }

    lctx.atsinfer_run_on_gpu.assign(lctx.atsinfer_units.size(), 0);
    for (size_t i = 0; i < lctx.atsinfer_units.size(); ++i) {
        lctx.atsinfer_run_on_gpu[i] = lctx.atsinfer_units[i].static_gpu ? 1 : 0;
    }

    lctx.atsinfer_dt_active          = true;
    lctx.atsinfer_pending_reschedule = false;
    // the first round is measured, so the DP starts from real timings instead of guesses;
    // its latency is not representative and must not become the reference
    lctx.atsinfer_want_profile        = true;
    lctx.atsinfer_prev_round_profiled = true;

    LLAMA_LOG_INFO("%s: ATSInfer dynamic transfer active: %zu MoE layers, %zu host-resident, "
                   "H2D %.0f MB/s\n", __func__, lctx.atsinfer_units.size(), n_cpu_units, bw_mbps);
    return true;
}

void atsinfer_dt_collect(llama_context & lctx) {
    if (!lctx.atsinfer_dt_active) {
        return;
    }

    std::vector<ggml_backend_split_timing> timings(512);
    const int n = ggml_backend_sched_get_split_timings(lctx.sched, timings.data(), (int) timings.size());
    if (n <= 0) {
        return;
    }

    const int cpu_backend_id = ggml_backend_sched_backend_index(lctx.sched, lctx.backend_cpu);

    // A layer's expert group can span several splits; accumulate per layer and per backend.
    std::vector<float> cpu_ms(lctx.atsinfer_units.size(), 0.0f);
    std::vector<float> gpu_ms(lctx.atsinfer_units.size(), 0.0f);

    // Round-level totals. These are what decide whether the asynchronous coordination work of
    // section 4.2 is worth doing: copy_total is the transfer time currently on the critical
    // path, i.e. the absolute ceiling on what overlapping transfer with compute could recover.
    double copy_total_us = 0.0;
    double exec_cpu_us   = 0.0;
    double exec_gpu_us   = 0.0;

    for (int i = 0; i < n; ++i) {
        const bool on_cpu = timings[i].backend_id == cpu_backend_id;

        copy_total_us += timings[i].copy_us;
        (on_cpu ? exec_cpu_us : exec_gpu_us) += timings[i].us;

        // Attribute execution time to a layer when the split is unambiguously that layer's
        // expert work, i.e. it both starts and ends inside it. A split that merely contains
        // expert ops among many others cannot be attributed without over-counting: on the GPU
        // side consecutive layers are merged into large splits, so this leaves t_g unobserved
        // there and it gets filled from sibling layers below.
        int il_first = -1, il_last = -1;
        const bool first_is_expert = atsinfer_is_moe_expert_node(timings[i].name, &il_first);
        const bool last_is_expert  = atsinfer_is_moe_expert_node(timings[i].last_name, &il_last);
        if (!first_is_expert || !last_is_expert || il_first != il_last) {
            continue;
        }
        if (il_first < 0 || il_first >= (int) lctx.atsinfer_unit_of_layer.size()) {
            continue;
        }
        const int idx = lctx.atsinfer_unit_of_layer[il_first];
        if (idx < 0) {
            continue;
        }
        (on_cpu ? cpu_ms : gpu_ms)[idx] += (float) (timings[i].us / 1000.0);
    }

    lctx.atsinfer_copy_total_ms = (float) (copy_total_us / 1000.0);
    lctx.atsinfer_exec_cpu_ms   = (float) (exec_cpu_us / 1000.0);
    lctx.atsinfer_exec_gpu_ms   = (float) (exec_gpu_us / 1000.0);

    // Only overwrite what was actually observed this round. A unit that ran on the CPU yields no
    // GPU sample and vice versa, so previous estimates are kept rather than zeroed -- otherwise
    // the DP would see t_gpu = 0 and promote everything unconditionally.
    for (size_t i = 0; i < lctx.atsinfer_units.size(); ++i) {
        auto & u = lctx.atsinfer_units[i];
        if (cpu_ms[i] > 0.0f) u.t_cpu_ms = cpu_ms[i];
        if (gpu_ms[i] > 0.0f) u.t_gpu_ms = gpu_ms[i];
    }

    // A unit only ever runs on the backend its placement assigned it to, so one of t_c / t_g
    // stays unobserved for every unit. Waiting for both would mean profiling forever, and a
    // profiled round is serialized -- measured, profiling every round costs ~40% of decode.
    //
    // MoE layers are structurally identical here, so the missing side is estimated from the
    // median of the units that did run on that backend. With --n-cpu-moe both backends are
    // exercised in the same round, which makes the estimate a same-round measurement rather
    // than a guess.
    auto median_of = [](std::vector<float> v) -> float {
        if (v.empty()) return 0.0f;
        const size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        return v[mid];
    };

    std::vector<float> cpu_samples, gpu_samples;
    for (const auto & u : lctx.atsinfer_units) {
        if (u.t_cpu_ms > 0.0f) cpu_samples.push_back(u.t_cpu_ms);
        if (u.t_gpu_ms > 0.0f) gpu_samples.push_back(u.t_gpu_ms);
    }

    const float cpu_med = median_of(cpu_samples);
    const float gpu_med = median_of(gpu_samples);

    for (auto & u : lctx.atsinfer_units) {
        if (u.t_cpu_ms <= 0.0f) u.t_cpu_ms = cpu_med;
        if (u.t_gpu_ms <= 0.0f) u.t_gpu_ms = gpu_med;
    }

    const float round_total = lctx.atsinfer_exec_cpu_ms + lctx.atsinfer_exec_gpu_ms + lctx.atsinfer_copy_total_ms;
    LLAMA_LOG_INFO("%s: ATSInfer round profile: exec CPU %.2f ms, exec GPU %.2f ms, "
                   "input copy %.2f ms (%.1f%% of %.2f ms)\n",
                   __func__, lctx.atsinfer_exec_cpu_ms, lctx.atsinfer_exec_gpu_ms,
                   lctx.atsinfer_copy_total_ms,
                   round_total > 0.0f ? 100.0f * lctx.atsinfer_copy_total_ms / round_total : 0.0f,
                   round_total);
    LLAMA_LOG_INFO("%s: ATSInfer expert layers measured: %zu with t_c, %zu with t_g; "
                   "median t_c %.3f ms, t_g %.3f ms\n",
                   __func__, cpu_samples.size(), gpu_samples.size(), cpu_med, gpu_med);

    // Persist measured per-layer timings back to the profile cache so the next
    // model load can use real t_c / t_g instead of heuristics (Defects 2 & 3 fix).
    // We only write after both CPU and GPU measurements are available; a one-sided
    // measurement is the median of the other side and not worth persisting.
    // The guard ensures we flush at most once per load cycle — subsequent profiling
    // rounds (triggered by Algorithm 3 rescheduling) carry identical measurements.
    if (!lctx.atsinfer_measurements_flushed &&
            cpu_samples.size() >= 1 && gpu_samples.size() >= 1) {
        std::vector<atsinfer_expert_measurement> measurements;
        for (const auto & u : lctx.atsinfer_units) {
            if (u.t_cpu_ms > 0.0f && u.t_gpu_ms > 0.0f) {
                atsinfer_expert_measurement m;
                m.layer_id = u.layer;
                m.t_cpu_ms = u.t_cpu_ms;
                m.t_gpu_ms = u.t_gpu_ms;
                measurements.push_back(m);
            }
        }

        if (!measurements.empty()) {
            const std::string cache_path = "atsinfer_profile.cache";
            atsinfer_hardware_profile hw;
            std::unordered_map<std::string, atsinfer_tensor_profile> profiles;
            if (atsinfer_load_profile_cache(cache_path, hw, profiles)) {
                atsinfer_save_profile_cache(cache_path, hw, profiles, measurements);
                lctx.atsinfer_measurements_flushed = true;
                LLAMA_LOG_INFO("%s: ATSInfer updated profile cache with %zu measured "
                        "expert layer timings for next load\n", __func__, measurements.size());
            } else {
                LLAMA_LOG_WARN("%s: ATSInfer could not load profile cache to update "
                        "measurements; measured data will not persist\n", __func__);
            }
        }
    }
}

bool atsinfer_dt_plan_round(llama_context & lctx, float last_round_ms) {
    if (!lctx.atsinfer_dt_active || !lctx.atsinfer_sched) {
        return false;
    }

    // A profiled round synchronizes after every split, so its latency is not representative.
    // Feeding it to Algorithm 3 as a reference would make every following normal round look
    // like a >15% improvement and trigger an endless profile/re-schedule oscillation.
    if (lctx.atsinfer_prev_round_profiled) {
        lctx.atsinfer_prev_round_profiled = false;
        return false;
    }

    // t_c falls out of a normal profiled round: a host-resident expert group sits between GPU
    // work on both sides, so it forms a split of its own and is attributable. t_g does not --
    // GPU-resident layers are merged into large multi-layer splits whose time cannot be split
    // back apart without over-counting.
    //
    // So calibrate t_g using the mechanism itself: promote exactly one host-resident layer for
    // one round. Surrounded by CPU work, it becomes a pure expert split on the GPU and the same
    // attribution measures it exactly. One sample is enough because the layers are homogeneous,
    // and atsinfer_dt_collect() propagates it to the rest by median.
    bool have_t_c = false, have_t_g = false;
    for (const auto & u : lctx.atsinfer_units) {
        if (u.t_cpu_ms > 0.0f) have_t_c = true;
        if (u.t_gpu_ms > 0.0f) have_t_g = true;
    }

    if (!have_t_c) {
        // plain measured round, no placement change
        lctx.atsinfer_want_profile        = true;
        lctx.atsinfer_prev_round_profiled = true;
        return false;
    }

    if (!have_t_g) {
        if (lctx.atsinfer_calib_unit < 0) {
            for (size_t i = 0; i < lctx.atsinfer_units.size(); ++i) {
                if (!lctx.atsinfer_units[i].static_gpu) {
                    lctx.atsinfer_calib_unit    = (int) i;
                    lctx.atsinfer_run_on_gpu[i] = 1;
                    LLAMA_LOG_INFO("%s: ATSInfer calibrating t_g by promoting layer %d for one round\n",
                            __func__, lctx.atsinfer_units[i].layer);
                    break;
                }
            }
        }
        if (lctx.atsinfer_calib_unit >= 0) {
            lctx.atsinfer_want_profile        = true;
            lctx.atsinfer_prev_round_profiled = true;
            lctx.atsinfer_pending_reschedule  = true; // the promotion changes the graph
            return true;
        }
        return false;
    }

    // calibration done: put the probe layer back where the static placement had it, and let the
    // DP decide from here on
    if (lctx.atsinfer_calib_unit >= 0) {
        lctx.atsinfer_run_on_gpu[lctx.atsinfer_calib_unit] = 0;
        lctx.atsinfer_calib_unit = -1;
    }

    // Establish the reference latency from the first representative round before arming the gate.
    if (lctx.atsinfer_ref_latency_ms <= 0.0f) {
        lctx.atsinfer_ref_latency_ms = last_round_ms;
    }

    // Algorithm 3: only re-run the DP when the load actually moved and enough time has passed.
    if (!lctx.atsinfer_sched->should_reschedule(lctx.atsinfer_ref_latency_ms, last_round_ms, last_round_ms)) {
        return false;
    }

    const auto plan = atsinfer_schedule_round(lctx.atsinfer_units);
    if (plan.run_on_gpu.size() != lctx.atsinfer_units.size()) {
        return false;
    }

    const bool changed = plan.run_on_gpu != lctx.atsinfer_run_on_gpu;

    lctx.atsinfer_run_on_gpu   = plan.run_on_gpu;
    lctx.atsinfer_ref_latency_ms = last_round_ms;
    lctx.atsinfer_sched->record_reschedule_event();
    ++lctx.atsinfer_n_reschedules;

    if (changed) {
        // the node -> backend assignment changed, so the cached graph is stale
        lctx.atsinfer_pending_reschedule = true;
        LLAMA_LOG_INFO("%s: ATSInfer re-scheduled (#%d): %d/%zu expert layers promoted to GPU, "
                       "estimated %.2f ms\n", __func__, lctx.atsinfer_n_reschedules,
                       plan.n_promoted, lctx.atsinfer_units.size(), plan.estimated_latency_ms);
    }

    return changed;
}

void atsinfer_dt_apply(llama_context & lctx, ggml_tensor * cur, const char * name, int il) {
    if (!lctx.atsinfer_dt_active || il < 0) {
        return;
    }
    if (il >= (int) lctx.atsinfer_unit_of_layer.size()) {
        return;
    }
    const int idx = lctx.atsinfer_unit_of_layer[il];
    if (idx < 0 || idx >= (int) lctx.atsinfer_run_on_gpu.size()) {
        return;
    }

    // only the expert operators are steered; the router and the weighted sum stay where the
    // scheduler put them
    static const char * steered[] = { "ffn_moe_up", "ffn_moe_gate", "ffn_moe_gate_par", "ffn_moe_down" };
    bool match = false;
    for (const char * s : steered) {
        if (strcmp(name, s) == 0) {
            match = true;
            break;
        }
    }
    if (!match) {
        return;
    }

    const bool want_gpu = lctx.atsinfer_run_on_gpu[idx] != 0;
    if (want_gpu == lctx.atsinfer_units[idx].static_gpu) {
        // nothing to override: the static placement already puts it where we want it
        return;
    }

    ggml_backend_t target = nullptr;
    if (want_gpu) {
        for (auto * backend : lctx.backends) {
            if (backend == lctx.backend_cpu) {
                continue;
            }
            if (ggml_backend_supports_op(backend, cur) || ggml_backend_offload_op(backend, cur)) {
                target = backend;
                break;
            }
        }
    } else {
        target = lctx.backend_cpu;
    }

    if (target) {
        ggml_backend_sched_set_tensor_backend(lctx.sched, cur, target);
    }
}
