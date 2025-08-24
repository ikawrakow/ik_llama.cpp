#pragma once

#include "common.cuh"

void ggml_cuda_mul_mat_q_id(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, char * ids_data, char * src1_quantized_data);
