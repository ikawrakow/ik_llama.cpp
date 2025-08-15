#include "common.cuh"

void ggml_cuda_op_add_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_add_id(const float * src0, const float * src1, const int32_t * src2, float * dst,
        int64_t ne00, int64_t ne01, int64_t ne02,
        int64_t ne0, int64_t ne1, size_t nb01, size_t nb02, size_t nb11, size_t nb21, cudaStream_t stream);

