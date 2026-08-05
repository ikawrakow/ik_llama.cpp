#pragma once

// Load-Aware Dynamic Transfer, section 4.4 of arXiv 2607.10183v2.
// Implementation notes are in src/llama-atsinfer.cpp.

#include "ggml.h"

struct llama_context;

// Derive the static placement b from the loaded model and build the per-round unit list.
// Returns false (and leaves dynamic transfer inactive) for dense models or when no expert
// layer is host-resident, in which case there is nothing to promote.
bool atsinfer_dt_init(llama_context & lctx);

// Fold the scheduler's per-split timings from the round that just ran into t_c / t_g.
// Only call after a round that was executed with profiling enabled.
void atsinfer_dt_collect(llama_context & lctx);

// Run Algorithm 3, and Algorithm 2 when it fires. Returns true when the assignment changed,
// meaning the cached graph must be rebuilt.
bool atsinfer_dt_plan_round(llama_context & lctx, float last_round_ms);

// Steer one graph node according to the current round's assignment. Called from the graph-build
// callback, which supplies the unsuffixed node name and the layer index.
void atsinfer_dt_apply(llama_context & lctx, ggml_tensor * cur, const char * name, int il);
