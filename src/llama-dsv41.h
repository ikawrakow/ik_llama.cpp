#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>
struct llama_batch;
struct llama_context;

bool llama_prepare_dsv41_graph_inputs(llama_context & lctx, const llama_batch & batch, bool set_tensors, bool reserve_plan);
void llama_reset_dsv41_state(llama_context * ctx, int32_t seq_id = -1);

// test-only debug access: the gathered engram rows (F32, [n_cols*key_len, n_tokens] of the
// last prepared ubatch) that were set into the dsv41_engram_rows0 graph input
const std::vector<float> & llama_dsv41_debug_engram_gather(const llama_context * ctx);

// test-only debug access: number of positions pushed into the engram n-gram cache
// (0 right after llama_reset_dsv41_state)
uint32_t llama_dsv41_debug_engram_size(const llama_context * ctx);

// test-only debug access to the ratio-2 comp plan of the last prepared ubatch
const std::vector<int32_t> & llama_dsv41_debug_r2_write_pos(const llama_context * ctx);
const std::vector<int64_t> & llama_dsv41_debug_r2_write_idxs(const llama_context * ctx);

// test-only debug access: per-token compressed visibility of the last prepared
// ubatch's r2/r1 plans (drives the causal compressed masks)
const std::vector<int32_t> & llama_dsv41_debug_r2_n_visible(const llama_context * ctx);
const std::vector<int32_t> & llama_dsv41_debug_r1_n_visible(const llama_context * ctx);

// test-only debug access: total bytes of the dsv41 compressed-cache buffers
size_t llama_dsv41_debug_cache_bytes(const llama_context * ctx);
