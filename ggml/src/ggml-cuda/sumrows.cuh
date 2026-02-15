#include "common.cuh"

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream);

void ggml_cuda_op_sum_rows_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_sum_rows_nc(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
