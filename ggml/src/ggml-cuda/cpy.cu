#include "cpy.cuh"
#include "dequantize.cuh"
#include "graph.cuh"
#include "cpy-utils.cuh"
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
#include "ggml-musa/mudnn.cuh"
#endif // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY

typedef void (*cpy_kernel_t)(const char * cx, char * cdst);

template <cpy_kernel_t cpy_1>
static __global__ void cpy_flt(const char * cx, char * cdst_direct, const int ne,
                               const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                               const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                               const int nb12, const int nb13, char ** cdst_indirect, int graph_cpynode_index) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    char * cdst = (cdst_indirect != nullptr) ? cdst_indirect[graph_cpynode_index]: cdst_direct;

    // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}

template <typename src_t, typename dst_t>
static __global__ void cpy_flt_contiguous(const char * cx, char * cdst_direct, const int ne,
                               char ** cdst_indirect, int graph_cpynode_index) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    auto dst = (cdst_indirect != nullptr) ? (dst_t *)cdst_indirect[graph_cpynode_index] : (dst_t *)cdst_direct;
    auto src = (const src_t *)cx;

    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        dst[i] = __float2bfloat16(src[i]);
    } else {
        dst[i] = (dst_t)src[i];
    }
}

static __device__ void cpy_blck_q8_0_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *)(cdsti);

#pragma unroll
    for (int j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j) = dq.x;
        *(cdstf + j + 1) = dq.y;
    }
}

static __device__ void cpy_blck_q8_0_f16(const char * cxi, char * cdsti) {
    half * dsth = (half *)(cdsti);

#pragma unroll
    for (int j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(dsth + j + 0) = __float2half(dq.x);
        *(dsth + j + 1) = __float2half(dq.y);
    }
}

template<dequantize_kernel_t dequant, int qk>
static __device__ void cpy_blck_q_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *)(cdsti);

#pragma unroll
    for (int j = 0; j < qk/2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j) = dq.x;
        *(cdstf + j + qk/2) = dq.y;
    }
}

template<dequantize_kernel_t dequant, int qk>
static __device__ void cpy_blck_q_f16(const char * cxi, char * cdsti) {
    half * dsth = (half *)(cdsti);

#pragma unroll
    for (int j = 0; j < qk/2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(dsth + j + 0) = __float2half(dq.x);
        *(dsth + j + qk/2) = __float2half(dq.y);
    }
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_f32_q(const char * cx, char * cdst_direct, const int ne,
                                 const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                                 const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                 const int nb12, const int nb13, char ** cdst_indirect, int graph_cpynode_index) {
    const int i = (blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    char * cdst = (cdst_indirect != nullptr) ? cdst_indirect[graph_cpynode_index]: cdst_direct;

    const int i03 = i/(ne00 * ne01 * ne02);
    const int i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int i13 = i/(ne10 * ne11 * ne12);
    const int i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int dst_offset = (i10/qk)*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_q_f32(const char * cx, char * cdst_direct, const int ne,
                                 const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
                                 const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11,
                                 const int nb12, const int nb13, char ** cdst_indirect, int graph_cpynode_index) {
    const int i = (blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    char * cdst = (cdst_indirect != nullptr) ? cdst_indirect[graph_cpynode_index]: cdst_direct;

    const int i03 = i/(ne00 * ne01 * ne02);
    const int i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int i13 = i/(ne10 * ne11 * ne12);
    const int i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

// Copy destination pointers to GPU to be available when pointer indirection is in use

void ggml_cuda_cpy_dest_ptrs_copy(ggml_cuda_graph * cuda_graph, char ** host_dest_ptrs, const int host_dest_ptrs_size, cudaStream_t stream) {
#if defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS) || defined(GGML_MUSA_GRAPHS)
    if (cuda_graph->dest_ptrs_size < host_dest_ptrs_size) { // (re-)allocate GPU memory for destination pointers
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (cuda_graph->dest_ptrs_d != nullptr) {
            CUDA_CHECK(cudaFree(cuda_graph->dest_ptrs_d));
        }
        CUDA_CHECK(cudaMalloc(&cuda_graph->dest_ptrs_d, host_dest_ptrs_size*sizeof(char *)));
        cuda_graph->dest_ptrs_size = host_dest_ptrs_size;
    }
    // copy destination pointers to GPU
    CUDA_CHECK(cudaMemcpyAsync(cuda_graph->dest_ptrs_d, host_dest_ptrs, host_dest_ptrs_size*sizeof(char *), cudaMemcpyHostToDevice, stream));
    cuda_graph->graph_cpynode_index = 0; // reset index
#else
    GGML_UNUSED(cuda_graph); GGML_UNUSED(host_dest_ptrs);
    GGML_UNUSED(host_dest_ptrs_size); GGML_UNUSED(stream);
#endif
}

template<typename src_t, typename dst_t>
static void ggml_cpy_flt_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_flt<cpy_1_flt<src_t, dst_t>><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

template<typename src_t, typename dst_t>
static void ggml_cpy_flt_contiguous_cuda(
    const char * cx, char * cdst, const int ne,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_flt_contiguous<src_t, dst_t><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_f32_q8_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {

    GGML_ASSERT(ne % QK8_0 == 0);
    const int num_blocks = ne / QK8_0;
    cpy_f32_q<cpy_blck_f32_q8_0, QK8_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_q8_0_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {

    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q8_0_f32, QK8_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_q8_0_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {

    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q8_0_f16, QK8_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_f32_q4_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {

    GGML_ASSERT(ne % QK4_0 == 0);
    const int num_blocks = ne / QK4_0;
    cpy_f32_q<cpy_blck_f32_q4_0, QK4_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_q4_0_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {
    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_q4_0_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {
    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q_f16<dequantize_q4_0, QK4_0>, QK4_0><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_f32_q4_1_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {

    GGML_ASSERT(ne % QK4_1 == 0);
    const int num_blocks = ne / QK4_1;
    cpy_f32_q<cpy_blck_f32_q4_1, QK4_1><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_q4_1_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {
    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_q4_1_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {
    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q_f16<dequantize_q4_1, QK4_1>, QK4_1><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_iq4_nl_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {
    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q_f32<dequantize_iq4_nl, QK4_NL>, QK4_NL><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_iq4_nl_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {
    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q_f16<dequantize_iq4_nl, QK4_NL>, QK4_NL><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_f32_q5_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {

    GGML_ASSERT(ne % QK5_0 == 0);
    const int num_blocks = ne / QK5_0;
    cpy_f32_q<cpy_blck_f32_q5_0, QK5_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_q5_0_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {
    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_q5_0_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {
    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q_f16<dequantize_q5_0, QK5_0>, QK5_0><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_f32_q5_1_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {

    GGML_ASSERT(ne % QK5_1 == 0);
    const int num_blocks = ne / QK5_1;
    cpy_f32_q<cpy_blck_f32_q5_1, QK5_1><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_q5_1_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {
    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_q5_1_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {
    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q_f16<dequantize_q5_1, QK5_1>, QK5_1><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_f32_iq4_nl_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {

    GGML_ASSERT(ne % QK4_NL == 0);
    const int num_blocks = ne / QK4_NL;
    cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_f32_q6_0_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02, const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12, const int nb10, const int nb11, const int nb12, const int nb13, cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {

    GGML_ASSERT(ne % QK6_0 == 0);
    const int num_blocks = ne / QK6_0;
    cpy_f32_q<cpy_blck_f32_q6_0, QK6_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_q6_0_f32_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {
    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q_f32<dequantize_q6_0, QK6_0>, QK6_0><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static void ggml_cpy_q6_0_f16_cuda(
    const char * cx, char * cdst, const int ne,
    const int ne00, const int ne01, const int ne02,
    const int nb00, const int nb01, const int nb02,
    const int nb03, const int ne10, const int ne11, const int ne12,
    const int nb10, const int nb11, const int nb12, const int nb13,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {
    const int num_blocks = ne;
    cpy_q_f32<cpy_blck_q_f16<dequantize_q6_0, QK6_0>, QK6_0><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13, cdst_indirect, graph_cpynode_index++);
}

static __global__ void k_transpose_q8_0(const char * cx, char * cdst,
                                   const int ne10, const int ne11, const int ne12,
                                   const int nb01, const int nb02, const int nb03,
                                   const int nb11, const int nb12, const int nb13) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;

    //const int64_t ne00 = ne11;
    //const int64_t ne01 = ne10;
    //const int64_t ne02 = ne12;
    const int64_t i03 = i13;
    const int64_t i02 = i12;
    const int64_t i01 = i10; //(i - i03*ne00*ne01*ne02 - i02*ne00*ne01) / ne00;
    const int64_t i00 = i11; //i - i03*ne00*ne01*ne02 - i02*ne00*ne01 - i01*ne00;

    const block_q8_0 * q8 = (const block_q8_0 *)(cx + i01*nb01 + i02*nb02 + i03*nb03);
    const int ib0 = i00/QK8_0;
    const int iq0 = i00%QK8_0;

    float xi = __half2float(q8[ib0].d)*q8[ib0].qs[iq0];
    float amax = fabsf(xi);
    amax = warp_reduce_max(amax);

    float d = amax/127;
    int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    block_q8_0 * dst = (block_q8_0 *)(cdst + i11*nb11 + i12*nb12 + i13*nb13);
    dst[i10 / QK8_0].qs[i10 % QK8_0] = q;

    if (threadIdx.x == 0) {
        dst[i10 / QK8_0].d = __float2half(d);
    }
}

static void transpose_q8_0(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    auto stream = ctx.stream();
    auto num_blocks = ggml_nelements(dst)/QK8_0;
    k_transpose_q8_0<<<num_blocks, QK8_0, 0, stream>>>(
            (const char *)src->data, (char *)dst->data,
            dst->ne[0], dst->ne[1], dst->ne[2], src->nb[0], src->nb[2], src->nb[3],
            dst->nb[1], dst->nb[2], dst->nb[3]);
}


void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1, bool disable_indirection_for_this_node) {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    //GGML_ASSERT(src0->ne[3] == 1);

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    //GGML_ASSERT(src1->ne[3] == 1);

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    cudaStream_t main_stream = ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;

    bool fast_cpy = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_are_same_shape(src0, src1);

    char ** dest_ptrs_d = nullptr;
    int graph_cpynode_index = -1;
#if defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS) || defined(GGML_MUSA_GRAPHS)
    if(!disable_indirection_for_this_node && ctx.cuda_graph && ctx.cuda_graph->use_cpy_indirection) {
        dest_ptrs_d = ctx.cuda_graph->dest_ptrs_d;
        graph_cpynode_index = ctx.cuda_graph->graph_cpynode_index;
    }
#else
    GGML_UNUSED(disable_indirection_for_this_node);
#endif
    if (src0->type == src1->type && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        GGML_ASSERT(ggml_nbytes(src0) == ggml_nbytes(src1));
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
        if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) {
            CUDA_CHECK(mudnnMemcpyAsync(ctx, src1, src0));
        } else
#endif // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY
        {
            if (src0->type == GGML_TYPE_F32) {
                ggml_cpy_flt_cuda<float, float> (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(src1_ddc, src0_ddc, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, main_stream));
            }
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        if (fast_cpy) {
            ggml_cpy_flt_contiguous_cuda<float, float>(src0_ddc, src1_ddc, ne, main_stream, dest_ptrs_d, graph_cpynode_index);
        } else {
            ggml_cpy_flt_cuda<float, float> (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        if (fast_cpy) {
            ggml_cpy_flt_contiguous_cuda<float, nv_bfloat16>(src0_ddc, src1_ddc, ne, main_stream, dest_ptrs_d, graph_cpynode_index);
        } else {
            ggml_cpy_flt_cuda<float, nv_bfloat16> (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        if (fast_cpy) {
            ggml_cpy_flt_contiguous_cuda<float, half>(src0_ddc, src1_ddc, ne, main_stream, dest_ptrs_d, graph_cpynode_index);
        } else {
            ggml_cpy_flt_cuda<float, half> (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_q8_0_f16_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
            nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_q4_0_f16_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
            nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_q4_1_f16_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
            nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_q5_0_f16_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_F32) {
        ggml_cpy_iq4_nl_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
            nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_F16) {
        ggml_cpy_iq4_nl_f16_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_q5_1_f16_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q6_0) {
        ggml_cpy_f32_q6_0_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_Q6_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q6_0_f32_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_Q6_0 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_q6_0_f16_cuda(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_flt_cuda<half, half> (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_flt_cuda<half, nv_bfloat16> (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_flt_cuda<half, float> (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_flt_cuda<nv_bfloat16, nv_bfloat16> (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_flt_cuda<nv_bfloat16, half> (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_flt_cuda<nv_bfloat16, float> (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_flt_cuda<float, int32_t> (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_flt_cuda<int32_t, float> (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream, dest_ptrs_d, graph_cpynode_index);
    } else if (ggml_are_same_shape(src0, src1) && src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        // This is needed for MLA with mla=2 when using q8_0 cache.
        transpose_q8_0(ctx, src0, src1);
    } else {
        GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
#if defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS) || defined(GGML_MUSA_GRAPHS)
    if(!disable_indirection_for_this_node && ctx.cuda_graph && ctx.cuda_graph->use_cpy_indirection) {
        ctx.cuda_graph->graph_cpynode_index = graph_cpynode_index;
    }
#else
    GGML_UNUSED(disable_indirection_for_this_node);
#endif

}

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    bool disable_indirection = true;
    ggml_cuda_cpy(ctx, src0, dst, disable_indirection);
}

void* ggml_cuda_cpy_fn(const ggml_tensor * src0, ggml_tensor * src1) {
    bool fast_cpy = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_are_same_shape(src0, src1);
    if (src0->type == src1->type && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        // Prioritize CUDA graph compatibility over direct memory copy optimization.
        // Using copy kernels here maintains graph indirection support, preventing performance regression from disabled CUDA graphs.
        if (src0->type == GGML_TYPE_F32) {
            return (void*) cpy_flt<cpy_1_flt<float, float>>;
        } else {
            return nullptr;
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        return fast_cpy ? (void *)cpy_flt_contiguous<float, float> : (void*) cpy_flt<cpy_1_flt<float, float>>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        return fast_cpy ? (void *)cpy_flt_contiguous<float, nv_bfloat16> : (void*) cpy_flt<cpy_1_flt<float, nv_bfloat16>>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        return fast_cpy ? (void *)cpy_flt_contiguous<float, half> : (void*) cpy_flt<cpy_1_flt<float, half>>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        return (void*) cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>;
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>;
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F16) {
        return (void*) cpy_q_f32<cpy_blck_q8_0_f16, QK8_0>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        return (void*) cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>;
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>;
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F16) {
        return (void*) cpy_q_f32<cpy_blck_q_f16<dequantize_q4_0, QK4_0>, QK4_0>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        return (void*) cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>;
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>;
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F16) {
        return (void*) cpy_q_f32<cpy_blck_q_f16<dequantize_q4_1, QK4_1>, QK4_1>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        return (void*) cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>;
    } else if (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_q_f32<cpy_blck_q_f32<dequantize_iq4_nl, QK4_NL>, QK4_NL>;
    } else if (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_F16) {
        return (void*) cpy_q_f32<cpy_blck_q_f16<dequantize_iq4_nl, QK4_NL>, QK4_NL>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        return (void*) cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>;
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>;
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F16) {
        return (void*) cpy_q_f32<cpy_blck_q_f16<dequantize_q5_0, QK5_0>, QK5_0>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        return (void*) cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>;
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>;
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F16) {
        return (void*) cpy_q_f32<cpy_blck_q_f16<dequantize_q5_1, QK5_1>, QK5_1>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q6_0) {
        return (void*) cpy_f32_q<cpy_blck_f32_q6_0, QK6_0>;
    } else if (src0->type == GGML_TYPE_Q6_0 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_q_f32<cpy_blck_q_f32<dequantize_q6_0, QK6_0>, QK6_0>;
    } else if (src0->type == GGML_TYPE_Q6_0 && src1->type == GGML_TYPE_F16) {
        return (void*) cpy_q_f32<cpy_blck_q_f16<dequantize_q6_0, QK6_0>, QK6_0>;
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        return (void*) cpy_flt<cpy_1_flt<half, half>>;
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        return (void*) cpy_flt<cpy_1_flt<half, nv_bfloat16>>;
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_flt<cpy_1_flt<half, float>>;
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        return (void*) cpy_flt<cpy_1_flt<nv_bfloat16, half>>;
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        return (void*) cpy_flt<cpy_1_flt<nv_bfloat16, nv_bfloat16>>;
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_flt<cpy_1_flt<nv_bfloat16, float>>;
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        return (void*) cpy_flt<cpy_1_flt<float, int32_t>>;
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        return (void*) cpy_flt<cpy_1_flt<int32_t, float>>;
    } else if (ggml_are_same_shape(src0, src1) && src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        return (void *)transpose_q8_0;
    } else {
        GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

template <typename src_t, typename dst_t>
static __global__ void cpy_flt_contiguous(const int ne, const char * cx1, const char * cx2, char * cdst_direct1, char * cdst_direct2,
                               char ** cdst_indirect, int graph_cpynode_index) {
    const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    auto dst1 = (cdst_indirect != nullptr) ? (dst_t *)cdst_indirect[graph_cpynode_index+0] : (dst_t *)cdst_direct1;
    auto dst2 = (cdst_indirect != nullptr) ? (dst_t *)cdst_indirect[graph_cpynode_index+1] : (dst_t *)cdst_direct2;
    auto src1 = (const src_t *)cx1;
    auto src2 = (const src_t *)cx2;

    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        dst1[i] = __float2bfloat16(src1[i]);
        dst2[i] = __float2bfloat16(src2[i]);
    } else {
        dst1[i] = (dst_t)src1[i];
        dst2[i] = (dst_t)src2[i];
    }
}

template<typename src_t, typename dst_t>
static void ggml_cpy_flt_contiguous_cuda_2(
    const char * cx1, const char * cx2, char * cdst1, char * cdst2, const int ne,
    cudaStream_t stream, char ** cdst_indirect, int & graph_cpynode_index) {

    const int num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    cpy_flt_contiguous<src_t, dst_t><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (ne, cx1, cx2, cdst1, cdst2, cdst_indirect, graph_cpynode_index);
    graph_cpynode_index += 2;
}

bool ggml_cuda_cpy_2(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, const ggml_tensor * src2,
        ggml_tensor * dst1, ggml_tensor * dst2, bool disable_indirection) {
    if (src1->type != GGML_TYPE_F32 || src2->type != GGML_TYPE_F32) return false;
    if (dst1->type != GGML_TYPE_F16 && dst1->type != GGML_TYPE_BF16) return false;
    if (dst2->type != GGML_TYPE_F16 && dst2->type != GGML_TYPE_BF16) return false;
    bool fast_cpy_1 = ggml_is_contiguous(src1) && ggml_is_contiguous(dst1) && ggml_are_same_shape(src1, dst1);
    bool fast_cpy_2 = ggml_is_contiguous(src2) && ggml_is_contiguous(dst2) && ggml_are_same_shape(src2, dst2);
    if (!fast_cpy_1 || !fast_cpy_2) return false;
    auto nelem = ggml_nelements(dst1);
    if (ggml_nelements(dst2) != nelem) return false;

    char ** dest_ptrs = nullptr;
    int graph_cpynode_index = -1;
#if defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS) || defined(GGML_MUSA_GRAPHS)
    if(ctx.cuda_graph->use_cpy_indirection && !disable_indirection) {
        dest_ptrs = ctx.cuda_graph->dest_ptrs_d;
        graph_cpynode_index = ctx.cuda_graph->graph_cpynode_index;
    }
#else
    GGML_UNUSED(disable_indirection);
#endif

    if (dst1->type == GGML_TYPE_F16) {
        ggml_cpy_flt_contiguous_cuda_2<float, half>((const char *)src1->data, (const char *)src2->data,
                (char *)dst1->data, (char *)dst2->data, nelem, ctx.stream(), dest_ptrs, graph_cpynode_index);
    } else {
        ggml_cpy_flt_contiguous_cuda_2<float, nv_bfloat16>((const char *)src1->data, (const char *)src2->data,
                (char *)dst1->data, (char *)dst2->data, nelem, ctx.stream(), dest_ptrs, graph_cpynode_index);
    }

#if defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS) || defined(GGML_MUSA_GRAPHS)
    if(ctx.cuda_graph->use_cpy_indirection && !disable_indirection) {
        ctx.cuda_graph->graph_cpynode_index = graph_cpynode_index;
    }
#endif
    return true;
}

template <typename src_t, typename dst_t>
static __global__ void concat_cpy(const char * csrc1, const char * csrc2, char * cdst, int ne1, int ne,
        char ** dest_ptrs, int copy_index) {

    auto dst = (dst_t *)(dest_ptrs ? dest_ptrs[copy_index] : cdst);
    auto src1 = (const src_t *)csrc1;
    auto src2 = (const src_t *)csrc2;

    for (int i = threadIdx.x; i < ne; i += blockDim.x) {
        if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
            dst[i] = __float2bfloat16(i < ne1 ? src1[i] : src2[i - ne1]);
        } else {
            dst[i] = (dst_t)(i < ne1 ? src1[i] : src2[i - ne1]);
        }
    }
}

template <typename src_t, typename dst_t>
static void ggml_concat_cpy_cuda(const char * src1, const char * src2, char * dst, int ne1, int ne, cudaStream_t stream,
        char ** dest_ptrs, int& copy_index) {

    int block_dim = std::min(ne, 768);
    concat_cpy<src_t, dst_t><<<1, block_dim, 0, stream>>>(src1, src2, dst, ne1, ne, dest_ptrs, copy_index);
    ++copy_index;
}

bool ggml_cuda_concat_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, const ggml_tensor * dst,
        [[maybe_unused]] bool disable_indirection) {

    if (dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_BF16) return false;
    //if (ggml_nrows(dst) > 1) return false;
    if (dst->src[0] != concat) return false;
    if (ggml_nrows(concat->src[0]) != 1 || ggml_nrows(concat->src[1]) != 1) return false;
    if (concat->src[0]->type != GGML_TYPE_F32 || concat->src[1]->type != GGML_TYPE_F32) return false;
    if (dst->ne[0] != concat->src[0]->ne[0] + concat->src[1]->ne[0]) return false;

    char ** dest_ptrs = nullptr;
    int graph_cpynode_index = -1;
#if defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS) || defined(GGML_MUSA_GRAPHS)
    if(ctx.cuda_graph->use_cpy_indirection && !disable_indirection) {
        dest_ptrs = ctx.cuda_graph->dest_ptrs_d;
        graph_cpynode_index = ctx.cuda_graph->graph_cpynode_index;
    }
#endif

    if (dst->type == GGML_TYPE_F16) {
        ggml_concat_cpy_cuda<float, half>((const char *)concat->src[0]->data, (const char *)concat->src[1]->data,
                (char *)dst->data, concat->src[0]->ne[0], dst->ne[0], ctx.stream(), dest_ptrs, graph_cpynode_index);
    } else {
        ggml_concat_cpy_cuda<float, nv_bfloat16>((const char *)concat->src[0]->data, (const char *)concat->src[1]->data,
                (char *)dst->data, concat->src[0]->ne[0], dst->ne[0], ctx.stream(), dest_ptrs, graph_cpynode_index);
    }

#if defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS) || defined(GGML_MUSA_GRAPHS)
    if(ctx.cuda_graph->use_cpy_indirection && !disable_indirection) {
        ctx.cuda_graph->graph_cpynode_index = graph_cpynode_index;
    }
#endif
    return true;

}
