#pragma once

#include "common.cuh"

void ggml_cuda_mul_mat_q_id(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst, char * ids_data, char * src1_quantized_data);

void compute_row_ids(const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int64_t ne02, int64_t ne12, int64_t n_expert_used, int64_t ne11, int64_t nb11, int64_t nb12, int64_t nb21, cudaStream_t stream);

bool ggml_cuda_can_use_mmq_id(enum ggml_type type, int cc, int64_t ne11);
