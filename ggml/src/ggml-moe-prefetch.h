#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// mapping registry; a registered range marks tensors as mmap-backed and
// eligible for prefetch. Nothing is opened or retained beyond (base, size).
void ggml_moe_prefetch_register_mapping  (const void * base, size_t size);
void ggml_moe_prefetch_unregister_mapping(const void * base);

// worker pool control; n_threads > 0 (re)creates the pool, <= 0 shuts it down
void ggml_moe_prefetch_set_n_threads(int n_threads);
bool ggml_moe_prefetch_enabled(void);

// the epoch is bumped once per scheduler pass so per-tensor enqueues are
// idempotent within one graph execution but re-issued on the next.
void ggml_moe_prefetch_new_epoch(void);

// selective enqueue for one MoE node (GGML_OP_MUL_MAT_ID or
// GGML_OP_MOE_FUSED_UP_GATE); reads the ids tensor bytes (must be final and
// host-visible) and enqueues the selected expert ranges of src weights.
void ggml_moe_prefetch_node(const struct ggml_tensor * node);

// full-tensor lookahead enqueue (low priority), e.g. for prompt processing
// where the next layer's experts are needed in bulk.
void ggml_moe_prefetch_tensor(const struct ggml_tensor * w);

// block until all pending prefetch jobs for tensor w are complete.
// Returns immediately when nothing is pending.
void ggml_moe_prefetch_wait(const struct ggml_tensor * w);

// MADV_COLD the pages the streamer had to read from storage for tensor w,
// so prompt-processing streaming traffic is reclaimed ahead of the decode
// working set. Pages already resident before the sweep are left alone.
// Only applies to tensors enqueued via ggml_moe_prefetch_tensor.
void ggml_moe_prefetch_cold(const struct ggml_tensor * w);

// kernel-entry hook, called at MoE matmul start (gated by
// cplan->moe_expert_prefetch). Thread 0 fire-and-forget enqueues when the
// scheduler hook did not run for this epoch (pure-CPU graphs); falls back to
// madvise(MADV_WILLNEED) when the engine is off.
void ggml_moe_prefetch_kernel_hook(const struct ggml_tensor * node, int ith);

#ifdef __cplusplus
}
#endif
