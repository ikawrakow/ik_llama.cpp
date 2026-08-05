#ifndef ATSINFER_SCHEDULER_H
#define ATSINFER_SCHEDULER_H

#include "atsinfer-placement.h"
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>

// Load-Aware Dynamic Transfer, section 4.4 of arXiv 2607.10183v2.
//
// One schedulable unit in execution order. Here a unit is one layer's group of expert
// operators (up/gate/down): they share a routing decision, and splitting them across
// backends breaks GGML_OP_MOE_FUSED_UP_GATE.
struct atsinfer_round_unit {
    int   layer      = -1;
    bool  static_gpu = false;  // b_i: true if static placement made this GPU-resident
    float t_cpu_ms   = 0.0f;   // t_i^c, measured
    float t_gpu_ms   = 0.0f;   // t_i^g, measured
    float c_ms       = 0.0f;   // c_i, activation transfer cost at a backend boundary
    float w_ms       = 0.0f;   // w_i, weight transfer time of the ACTIVATED experts only
    // Full size of the layer's expert group in bytes. A promoted op copies these weights
    // into the target GPU, so this is the VRAM a promotion must fit into. The required
    // headroom check in atsinfer_dt_apply() uses it to refuse promotions that would OOM.
    size_t weight_bytes = 0;
};

struct atsinfer_round_plan {
    std::vector<uint8_t> run_on_gpu;                // rb_i, 1 = execute on GPU this round
    float                estimated_latency_ms = 0.0f;
    int                  n_promoted           = 0;  // units with b_i=CPU but rb_i=GPU
};

// Candidate GPU backend for promoting a unit, one per backend that can run the op.
struct atsinfer_device_candidate {
    size_t free_bytes   = 0; // free VRAM on the device right now
    size_t weight_bytes = 0; // VRAM the promotion needs (the unit's expert group)
    size_t n_promoted   = 0; // layers already promoted here (load signal)
};

// Pick which GPU a promoted unit should run on. Returns the candidate index, or -1 when
// none has room for weight_bytes -- the unit then stays on the CPU rather than risking
// an OOM that would take the whole context down mid-decode.
//
// Keeps the unit on the device it already used (preferred_index) when that still has room,
// so a layer does not bounce between GPUs round to round (each move changes the node's
// backend and forces a graph rebuild). Otherwise picks the candidate with the most
// headroom after the promotion, ties broken toward the least-loaded device (fewest
// already-promoted layers) and then the lowest index for determinism.
int atsinfer_select_promotion_device(
    const std::vector<atsinfer_device_candidate> & candidates,
    int preferred_index);

// Algorithm 2: minimize round latency
//   T = sum t_rb_i + sum c_i*1{rb_{i-1} != rb_i} + sum_{i in G} delta_i
// where G is the set of promoted units and delta_i = max(w_i - seg(j,i), 0) is the transfer
// time left exposed after overlapping with the work between the previous CPU-side endpoint
// j and unit i.
atsinfer_round_plan atsinfer_schedule_round(const std::vector<atsinfer_round_unit> & units);

// Algorithm 3: threshold + rate limited re-scheduling.
// Paper settings (section 4.4.3): epsilon = 15%, minimum interval = 5x recent TPOT.
class ATSInferRescheduler {
public:
    ATSInferRescheduler(float deviation_threshold = 0.15f, int min_tpot_multiplier = 5);

    // Advance the clock by one round. Call exactly once per round: it accumulates elapsed
    // time internally. Returns true when a new schedule should be computed.
    bool should_reschedule(float reference_latency_ms, float current_latency_ms, float current_tpot_ms);

    // Call after a re-schedule actually happened, to arm the minimum-interval gate.
    void record_reschedule_event();

    float elapsed_ms() const { return accumulated_time_ms; }

private:
    float threshold;
    int   tpot_multiplier;
    float last_reschedule_time_ms;
    float accumulated_time_ms;
    bool  ever_scheduled;
};

#endif // ATSINFER_SCHEDULER_H
