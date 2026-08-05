#include "atsinfer-scheduler.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

// ---------------------------------------------------------------------------
// Algorithm 3: load-aware re-scheduling
// ---------------------------------------------------------------------------

ATSInferRescheduler::ATSInferRescheduler(float deviation_threshold, int min_tpot_multiplier)
    : threshold(deviation_threshold)
    , tpot_multiplier(min_tpot_multiplier)
    , last_reschedule_time_ms(0.0f)
    , accumulated_time_ms(0.0f)
    , ever_scheduled(false) {}

bool ATSInferRescheduler::should_reschedule(
    float reference_latency_ms,
    float current_latency_ms,
    float current_tpot_ms) {

    // one call == one round; the caller must not poll this
    accumulated_time_ms += current_tpot_ms > 0.0f ? current_tpot_ms : 0.0f;

    // no plan yet: schedule once before applying the interval gate
    if (!ever_scheduled) {
        return true;
    }

    if (reference_latency_ms <= 0.0f) {
        return false;
    }

    const float deviation    = std::fabs(current_latency_ms - reference_latency_ms) / reference_latency_ms;
    const float min_interval = current_tpot_ms * (float) tpot_multiplier;

    return deviation >= threshold &&
           (accumulated_time_ms - last_reschedule_time_ms) >= min_interval;
}

void ATSInferRescheduler::record_reschedule_event() {
    last_reschedule_time_ms = accumulated_time_ms;
    ever_scheduled          = true;
}

// ---------------------------------------------------------------------------
// Algorithm 2: dynamic transfer scheduling
// ---------------------------------------------------------------------------

namespace {

constexpr float k_inf = std::numeric_limits<float>::infinity();

enum : int { ST_CPU = 0, ST_GPU = 1 };

// How a dp state was reached, so the plan can be reconstructed.
struct back_ref {
    int  prev_state = -1;  // state at i-1 that we extended
    int  promo_from = -1;  // >= 0 when this state is a promotion starting its transfer at j
};

} // namespace

atsinfer_round_plan atsinfer_schedule_round(const std::vector<atsinfer_round_unit> & units) {
    atsinfer_round_plan plan;

    const int n = (int) units.size();
    if (n == 0) {
        return plan;
    }

    auto t_default = [&](int i) {
        return units[i].static_gpu ? units[i].t_gpu_ms : units[i].t_cpu_ms;
    };

    // Default-path cost prefix, so that
    //   seg(j,i) = sum_{k=j+1}^{i-1} [ t_{b_k}(k) + c_k*1{b_{k-1} != b_k} ]
    //            = pref[i] - pref[j+1]
    std::vector<float> pref(n + 1, 0.0f);
    for (int k = 0; k < n; ++k) {
        float d = t_default(k);
        if (k > 0 && units[k].static_gpu != units[k - 1].static_gpu) {
            d += units[k].c_ms;
        }
        pref[k + 1] = pref[k] + d;
    }
    auto seg = [&](int j, int i) {
        // work available to hide unit i's weight transfer, i.e. the default path over (j, i)
        return std::max(0.0f, pref[i] - pref[j + 1]);
    };

    std::vector<std::array<float, 2>>    dp(n, {k_inf, k_inf});
    std::vector<std::array<back_ref, 2>> bt(n);

    // ---- i = 0 -------------------------------------------------------------
    if (units[0].static_gpu) {
        dp[0][ST_GPU] = units[0].t_gpu_ms;
    } else {
        dp[0][ST_CPU] = units[0].t_cpu_ms;
        // promoting the very first unit: there is no preceding CPU endpoint, so nothing to
        // overlap against and the whole weight transfer is exposed
        dp[0][ST_GPU]            = units[0].w_ms + units[0].t_gpu_ms;
        bt[0][ST_GPU].promo_from = -1;
    }

    // ---- i >= 1 ------------------------------------------------------------
    for (int i = 1; i < n; ++i) {
        const auto & u = units[i];

        // NOTE: the paper's Algorithm 2 line 17 writes the switch term as c_i*1{b_{i-1}=GPU},
        // which charges an activation transfer exactly when no backend change occurs. We follow
        // the objective function in section 4.4.1 instead -- c_i*1{rb_{i-1} != rb_i} -- which is
        // unambiguous and is what the surrounding text describes.
        auto extend = [&](int to_state) {
            const float t_exec = to_state == ST_GPU ? u.t_gpu_ms : u.t_cpu_ms;
            float best  = k_inf;
            int   bestp = -1;
            for (int from_state : {ST_CPU, ST_GPU}) {
                if (dp[i - 1][from_state] == k_inf) {
                    continue;
                }
                const float cand = dp[i - 1][from_state] + (from_state != to_state ? u.c_ms : 0.0f);
                if (cand < best) {
                    best  = cand;
                    bestp = from_state;
                }
            }
            if (best == k_inf) {
                return;
            }
            dp[i][to_state]              = best + t_exec;
            bt[i][to_state].prev_state   = bestp;
            bt[i][to_state].promo_from   = -1;
        };

        if (u.static_gpu) {
            // GPU-resident by default: it always runs on the GPU, only the switch cost varies
            extend(ST_GPU);
            dp[i][ST_CPU] = k_inf;
            continue;
        }

        // CPU-resident by default: keeping it on the CPU is always legal
        extend(ST_CPU);

        // ...or promote it, starting the weight transfer at a previous CPU-side endpoint j.
        // The units in (j, i) then run along the default path, which is the overlap window.
        float best  = k_inf;
        int   bestj = -1;
        for (int j = 0; j < i; ++j) {
            if (units[j].static_gpu || dp[j][ST_CPU] == k_inf) {
                continue;
            }
            const float cand = dp[j][ST_CPU] + std::max(u.w_ms, seg(j, i));
            if (cand < best) {
                best  = cand;
                bestj = j;
            }
        }
        if (best != k_inf) {
            // crossing into the GPU costs one activation transfer, since the default path
            // arriving at i ends on the CPU by construction (b_j = CPU, and (j,i) is default)
            const bool prev_on_gpu = units[i - 1].static_gpu;
            const float promoted   = best + (prev_on_gpu ? 0.0f : u.c_ms) + u.t_gpu_ms;
            if (promoted < dp[i][ST_GPU]) {
                dp[i][ST_GPU]            = promoted;
                bt[i][ST_GPU].prev_state = -1;
                bt[i][ST_GPU].promo_from = bestj;
            }
        }
    }

    // ---- pick the best terminal state and backtrack ------------------------
    int final_state = dp[n - 1][ST_CPU] <= dp[n - 1][ST_GPU] ? ST_CPU : ST_GPU;
    if (dp[n - 1][final_state] == k_inf) {
        // no feasible schedule was found; fall back to the static placement
        plan.run_on_gpu.resize(n);
        for (int i = 0; i < n; ++i) {
            plan.run_on_gpu[i] = units[i].static_gpu ? 1 : 0;
        }
        plan.estimated_latency_ms = pref[n];
        return plan;
    }

    plan.estimated_latency_ms = dp[n - 1][final_state];
    plan.run_on_gpu.assign(n, 0);

    int i     = n - 1;
    int state = final_state;
    while (i >= 0) {
        plan.run_on_gpu[i] = state == ST_GPU ? 1 : 0;

        const back_ref & ref = bt[i][state];
        if (ref.promo_from >= 0 || (state == ST_GPU && ref.prev_state == -1 && !units[i].static_gpu)) {
            // unit i was promoted; the units in (j, i) ran along the default path
            const int j = ref.promo_from;
            for (int k = i - 1; k > j; --k) {
                plan.run_on_gpu[k] = units[k].static_gpu ? 1 : 0;
            }
            if (j < 0) {
                break;
            }
            i     = j;
            state = ST_CPU;
            continue;
        }

        if (ref.prev_state < 0) {
            break;
        }
        state = ref.prev_state;
        --i;
    }

    for (int k = 0; k < n; ++k) {
        if (!units[k].static_gpu && plan.run_on_gpu[k]) {
            ++plan.n_promoted;
        }
    }

    return plan;
}
