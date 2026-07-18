#pragma once

#include "common.cuh"

void ggml_cuda_op_pack_cache_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_pack_cache_rows_supports(const ggml_tensor * op, int max_grid_x);
