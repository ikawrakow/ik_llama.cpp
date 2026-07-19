#include "pack-cache-rows.cuh"

#include "cpy-utils.cuh"
#include "graph.cuh"
#include "../ggml-pack-cache-rows.h"

#include <cstdint>
#include <limits>

static constexpr int CUDA_PACK_CACHE_ROWS_BLOCK_SIZE = 256;
static_assert(alignof(block_q8_0) == alignof(ggml_fp16_t),
        "Q8_0 pack alignment must match its fp16 scale");

template <typename dst_t>
static __global__ void pack_cache_rows_f32(
        const float * __restrict__ a,
        const float * __restrict__ b,
        char * dst_direct,
        int64_t d0,
        int64_t d1,
        int64_t rows,
        size_t a_nb1,
        size_t b_nb1,
        size_t dst_nb1,
        char ** dst_indirect,
        int write_index) {
    const int64_t d = d0 + d1;
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= d*rows) {
        return;
    }

    const int64_t row = i/d;
    const int64_t col = i - row*d;
    const float value = col < d0
            ? *(const float *) ((const char *) a + (size_t) row*a_nb1 + (size_t) col*sizeof(float))
            : *(const float *) ((const char *) b + (size_t) row*b_nb1 + (size_t) (col - d0)*sizeof(float));
    char * dst_base = dst_indirect != nullptr ? dst_indirect[write_index] : dst_direct;
    ((dst_t *) (dst_base + (size_t) row*dst_nb1))[col] = (dst_t) value;
}

static __global__ void pack_cache_rows_q8_0(
        const float * __restrict__ a,
        const float * __restrict__ b,
        char * dst_direct,
        int64_t d0,
        int64_t d1,
        int64_t rows,
        size_t a_nb1,
        size_t b_nb1,
        size_t dst_nb1,
        char ** dst_indirect,
        int write_index) {
    const int64_t blocks0 = d0/QK8_0;
    const int64_t blocks1 = d1/QK8_0;
    const int64_t blocks_per_row = blocks0 + blocks1;
    const int64_t i = (int64_t) blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= blocks_per_row*rows) {
        return;
    }

    const int64_t row = i/blocks_per_row;
    const int64_t block = i - row*blocks_per_row;
    const float * src = block < blocks0
            ? (const float *) ((const char *) a + (size_t) row*a_nb1) + block*QK8_0
            : (const float *) ((const char *) b + (size_t) row*b_nb1) + (block - blocks0)*QK8_0;
    char * dst_base = dst_indirect != nullptr ? dst_indirect[write_index] : dst_direct;
    block_q8_0 * dst = (block_q8_0 *) (dst_base + (size_t) row*dst_nb1);
    quantize_f32_q8_0_block(src, dst + block);
}

static bool pack_cache_rows_launch_fits(const ggml_tensor * op, int max_grid_x) {
    const int64_t d = op->ne[0];
    const int64_t rows = op->ne[1];
    if (max_grid_x <= 0 || d <= 0 || rows <= 0 ||
        d > std::numeric_limits<int64_t>::max()/rows) {
        return false;
    }
    const int64_t work_items = op->type == GGML_TYPE_Q8_0 ? (d/QK8_0)*rows : d*rows;
    const uint64_t blocks = ((uint64_t) work_items + CUDA_PACK_CACHE_ROWS_BLOCK_SIZE - 1)/
            CUDA_PACK_CACHE_ROWS_BLOCK_SIZE;
    return blocks > 0 && blocks <= (uint64_t) max_grid_x;
}

bool ggml_cuda_pack_cache_rows_supports(const ggml_tensor * op, int max_grid_x) {
    if (!ggml_pack_cache_rows_op_is_valid(op)) {
        return false;
    }
    return pack_cache_rows_launch_fits(op, max_grid_x);
}

void ggml_cuda_op_pack_cache_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int max_grid_x = ggml_cuda_info().devices[ctx.device].max_grid_x;
    GGML_ASSERT(ggml_cuda_pack_cache_rows_supports(dst, max_grid_x));
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    GGML_ASSERT(a->data != nullptr && b->data != nullptr && dst->data != nullptr);

    char ** write_dest_ptrs_d = nullptr;
    int write_index = -1;
#if defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS) || defined(GGML_MUSA_GRAPHS)
    if (ctx.cur_graph != nullptr && ctx.cur_graph->use_cpy_indirection) {
        write_dest_ptrs_d = ctx.cur_graph->dest_ptrs_d;
        write_index = ctx.cur_graph->graph_cpynode_index;
    }
#endif

    const int64_t work_items = dst->type == GGML_TYPE_Q8_0
            ? (dst->ne[0]/QK8_0)*dst->ne[1]
            : dst->ne[0]*dst->ne[1];
    const uint32_t blocks = (uint32_t) (((uint64_t) work_items + CUDA_PACK_CACHE_ROWS_BLOCK_SIZE - 1)/
            CUDA_PACK_CACHE_ROWS_BLOCK_SIZE);
    cudaStream_t stream = ctx.stream();
    switch (dst->type) {
        case GGML_TYPE_F32:
            pack_cache_rows_f32<float><<<blocks, CUDA_PACK_CACHE_ROWS_BLOCK_SIZE, 0, stream>>>(
                    (const float *) a->data, (const float *) b->data, (char *) dst->data,
                    a->ne[0], b->ne[0], a->ne[1], a->nb[1], b->nb[1], dst->nb[1],
                    write_dest_ptrs_d, write_index);
            break;
        case GGML_TYPE_F16:
            pack_cache_rows_f32<half><<<blocks, CUDA_PACK_CACHE_ROWS_BLOCK_SIZE, 0, stream>>>(
                    (const float *) a->data, (const float *) b->data, (char *) dst->data,
                    a->ne[0], b->ne[0], a->ne[1], a->nb[1], b->nb[1], dst->nb[1],
                    write_dest_ptrs_d, write_index);
            break;
        case GGML_TYPE_Q8_0:
            pack_cache_rows_q8_0<<<blocks, CUDA_PACK_CACHE_ROWS_BLOCK_SIZE, 0, stream>>>(
                    (const float *) a->data, (const float *) b->data, (char *) dst->data,
                    a->ne[0], b->ne[0], a->ne[1], a->nb[1], b->nb[1], dst->nb[1],
                    write_dest_ptrs_d, write_index);
            break;
        default:
            GGML_ABORT("unsupported ggml_pack_cache_rows destination type");
    }

#if defined(GGML_CUDA_USE_GRAPHS) || defined(GGML_HIP_GRAPHS) || defined(GGML_MUSA_GRAPHS)
    if (ctx.cur_graph != nullptr && ctx.cur_graph->use_cpy_indirection) {
        ctx.cur_graph->graph_cpynode_index = write_index + 1;
    }
#endif
}
