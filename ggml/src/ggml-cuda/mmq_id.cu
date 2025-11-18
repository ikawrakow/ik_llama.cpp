#include "mmq_id_common.cuh"
#include "mmq_id.cuh"
#include "quantize_id.cuh"
#include "quantize.cuh"

#include <vector>
#include <climits>
#include <cstdint>

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mmq_ids_helper_store {
    uint32_t data;

    __device__ mmq_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mmq_ids_helper_store) == 4, "unexpected size for mmq_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mmq_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mmq_ids_helper[];
    mmq_ids_helper_store * store = (mmq_ids_helper_store *) data_mmq_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mmq_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= offset) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mmq_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mmq_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_src1[nex_prev + itc] = it*sis1          + iex_used % nchannels_y;
        ids_dst [nex_prev + itc] = it*n_expert_used + iex_used;
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < gridDim.x - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mmq_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mmq_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mmq_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_get_physical_warp_size_host(); //ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mmq_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mmq_ids_helper_store);
    mmq_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1);
}

static void ggml_cuda_mul_mat_q_switch_type_id(ggml_backend_cuda_context & ctx, const mmq_args_id & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case_id<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case_id<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case_id<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case_id<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_0:
            mul_mat_q_case_id<GGML_TYPE_Q6_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case_id<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case_id<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case_id<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case_id<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case_id<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case_id<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case_id<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case_id<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case_id<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case_id<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case_id<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case_id<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case_id<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S_R4:
            mul_mat_q_case_id<GGML_TYPE_IQ1_S_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case_id<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case_id<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_KS:
            mul_mat_q_case_id<GGML_TYPE_IQ2_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_KL:
            mul_mat_q_case_id<GGML_TYPE_IQ2_KL>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_K:
            mul_mat_q_case_id<GGML_TYPE_IQ2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_K_R4:
            mul_mat_q_case_id<GGML_TYPE_IQ2_K_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_K:
            mul_mat_q_case_id<GGML_TYPE_IQ3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_K_R4:
            mul_mat_q_case_id<GGML_TYPE_IQ3_K_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_KS:
            mul_mat_q_case_id<GGML_TYPE_IQ3_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KSS:
            mul_mat_q_case_id<GGML_TYPE_IQ4_KSS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KS:
            mul_mat_q_case_id<GGML_TYPE_IQ4_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KS_R4:
            mul_mat_q_case_id<GGML_TYPE_IQ4_KS_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_K:
            mul_mat_q_case_id<GGML_TYPE_IQ4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_K_R4:
            mul_mat_q_case_id<GGML_TYPE_IQ4_K_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_KS:
            mul_mat_q_case_id<GGML_TYPE_IQ5_KS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_KS_R4:
            mul_mat_q_case_id<GGML_TYPE_IQ5_KS_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_K:
            mul_mat_q_case_id<GGML_TYPE_IQ5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ5_K_R4:
            mul_mat_q_case_id<GGML_TYPE_IQ5_K_R4>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ6_K:
            mul_mat_q_case_id<GGML_TYPE_IQ6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_KT:
            mul_mat_q_case_id<GGML_TYPE_IQ1_KT>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_KT:
            mul_mat_q_case_id<GGML_TYPE_IQ2_KT>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_KT:
            mul_mat_q_case_id<GGML_TYPE_IQ3_KT>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_KT:
            mul_mat_q_case_id<GGML_TYPE_IQ4_KT>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void compute_row_ids(const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int64_t ne02, int64_t ne12, int64_t n_expert_used, int64_t ne11, int64_t nb11, int64_t nb12, int64_t nb21,
        cudaStream_t stream) {

    const int si1  = nb21 / sizeof(int);
    const int sis1 = nb12 / nb11;

    switch (n_expert_used) {
        case  2:
            launch_mmq_ids_helper< 2> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
        case  4:
            launch_mmq_ids_helper< 4> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
        case  6:
            launch_mmq_ids_helper< 6> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
        case  8:
            launch_mmq_ids_helper< 8> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
        case 16:
            launch_mmq_ids_helper<16> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
        case 32:
            launch_mmq_ids_helper<32> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
        default:
            launch_mmq_ids_helper< 0> (ids, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_q_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids_tensor, ggml_tensor * dst, char * ids_data, char * src1_quantized_data) {
    GGML_ASSERT(       src1->type == GGML_TYPE_F32);
    GGML_ASSERT(       dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids_tensor || ids_tensor->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    //const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    //GGML_ASSERT(       nb00       == ts_src0);
    GGML_ASSERT(       nb10       == ts_src1);
    GGML_ASSERT(       nb0        == ts_dst);
    GGML_ASSERT(!ids_tensor || ids_tensor->nb[0] == ggml_type_size(ids_tensor->type));

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1];// / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2];// / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3];// / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc);

    if (!ids_tensor) {

        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
        if (!src1_quantized_data) {
            const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1
                                          + get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
            src1_q8_1.alloc(nbytes_src1_q8_1);
            quantize_mmq_q8_1_cuda(src1_d, src1_q8_1.get(), ne10, ne11, 1, ne10_padded, src0->type, stream);
            CUDA_CHECK(cudaGetLastError());
            src1_quantized_data = src1_q8_1.get();
        }

        const int64_t s12 = ne11*ne10_padded * sizeof(block_q8_1)/(QK8_1*sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args_id args = {
            src0_d, src0->type, (const int *)src1_quantized_data, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k, ne1};

        ggml_cuda_mul_mat_q_switch_type_id(ctx, args, stream);
        return;
    }

    const int64_t n_expert_used = ids_tensor->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1_local(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ids_dst_local(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> expert_bounds_local(ctx.pool());

    int32_t * ids_src1, *ids_dst, *expert_bounds;
    if (ids_data) {
        ids_src1 = (int32_t *)ids_data;
        ids_dst  = ids_src1 + ne_get_rows;
        expert_bounds = ids_dst + ne_get_rows;
    }
    else {
        GGML_ASSERT(ids_tensor->nb[0] == ggml_element_size(ids_tensor));

        ids_src1_local.alloc(ne_get_rows);
        ids_dst_local.alloc(ne_get_rows);
        expert_bounds_local.alloc(ne02 + 1);

        ids_src1 = ids_src1_local.get();
        ids_dst  = ids_dst_local.get();
        expert_bounds = expert_bounds_local.get();

        const int si1  = ids_tensor->nb[1] / ggml_element_size(ids_tensor);
        const int sis1 = nb12 / nb11;

        switch (n_expert_used) {
            case  2:
                launch_mmq_ids_helper< 2> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
            case  4:
                launch_mmq_ids_helper< 4> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
            case  6:
                launch_mmq_ids_helper< 6> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
            case  8:
                launch_mmq_ids_helper< 8> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
            case 16:
                launch_mmq_ids_helper<16> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
            case 32:
                launch_mmq_ids_helper<32> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
            default:
                launch_mmq_ids_helper< 0> ((const int32_t *) ids_tensor->data, ids_src1, ids_dst, expert_bounds,
                    ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
                break;
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    const size_t nbytes_src1_q8_1 = ne11_flat*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);

    ggml_cuda_pool_alloc<char> src1_q8_1_local(ctx.pool());

    char * src1_q8_1;

    if (src1_quantized_data) {
        src1_q8_1 = src1_quantized_data;
    } else {

        src1_q8_1_local.alloc(nbytes_src1_q8_1);
        src1_q8_1 = src1_q8_1_local.get();

        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_mmq_q8_1_cuda_id(src1_d, ids_src1, src1_q8_1, src0->type,
            ne10, s11, s12, s13, ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12 = ne11*ne10_padded * sizeof(block_q8_1)/(QK8_1*sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args_id args = {
        src0_d, src0->type, (const int *) src1_q8_1, ids_dst, expert_bounds, dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ne12};

    //printf("ne00 = %ld, ne01 = %ld, ne_get_rows = %ld, s01 = %ld, s1 = %ld\n", ne00, ne01, ne_get_rows, s01, s1);
    //printf("ne02 = %ld, s02 = %ld, s12 = %ld, s2 = %ld\n", ne02, s02, s12, s2);
    //printf("ne03 = %ld, s03 = %ld, s13 = %ld, s3 = %ld\n", ne03, s03, s13, s3);

    ggml_cuda_mul_mat_q_switch_type_id(ctx, args, stream);
}

bool ggml_cuda_can_use_mmq_id(enum ggml_type type, int cc, int64_t ne11) {
    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_K_R4:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ5_K_R4:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

   if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (ne11 <= 128 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0
                        || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q6_0) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    return (!GGML_CUDA_CC_IS_RDNA4(cc) && !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;

}
