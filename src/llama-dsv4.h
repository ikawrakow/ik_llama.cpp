#pragma once

#include <cstdint>
struct llama_batch;
struct llama_context;

bool llama_prepare_dsv4_graph_inputs(llama_context & lctx, const llama_batch & batch, bool set_tensors, bool reserve_plan);
void llama_reset_dsv4_state(llama_context * ctx, int32_t seq_id = -1);
