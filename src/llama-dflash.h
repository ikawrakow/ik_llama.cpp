#pragma once

#include <cstdint>

struct llama_context;

bool llama_prepare_dflash_graph_inputs(llama_context & lctx, uint32_t n_tokens);
