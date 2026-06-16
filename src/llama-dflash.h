#pragma once

#include <cstdint>

struct llama_context;

bool llama_prepare_dflash_graph_inputs(llama_context & lctx, uint32_t n_tokens);
void llama_sync_dflash_workspace_if_pending(llama_context & lctx);
