#include "common.cuh"
#include "kda.cuh"

#include <cstdint>

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <int block_size>
__device__ __forceinline__ float reduce_sum(float x, float * s) {
    x = warp_reduce_sum(x);
    if constexpr (block_size > WARP_SIZE) {
        //__shared__ float s[block_size/WARP_SIZE];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s[warp_id] = x;
        }
        __syncthreads();
        x = lane_id < block_size/WARP_SIZE ? s[lane_id] : 0.0f;
        x = warp_reduce_sum(x);
    }
    return x;
}

template <int HEAD_DIM, int block_size>
__global__ void kda_recurrent_f32(
    const float * __restrict__ q,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ k,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ v,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ g,
    const float * __restrict__ beta_in,   // [1, n_tokens, n_heads, n_seqs]
    const float * state_in,               // [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs], aliases state_out when fused
    float * __restrict__ dst,             // output
    float * state_out,                    // new state
    float * __restrict__ saved_states,
    const int32_t n_heads,
    const int32_t gqa_ratio,
    const int32_t repeat_type,
    const int32_t n_tokens,
    const int32_t n_seqs,
    size_t vnb1, size_t vnb2, size_t vnb3,
    size_t gnb0, size_t gnb1, size_t gnb2, size_t gnb3,
    size_t bnb1, size_t bnb2, size_t bnb3) {
    constexpr int32_t warps_per_head = HEAD_DIM/WARP_SIZE;
    const int32_t batch_idx = blockIdx.x / (warps_per_head*n_heads);
    const int32_t sub_head_idx = blockIdx.x % (warps_per_head*n_heads);
    const int32_t head_idx = sub_head_idx / warps_per_head;
    const int32_t sub_idx = sub_head_idx % warps_per_head;
    const int32_t head_idx_kq = repeat_type == 0 ? head_idx / gqa_ratio : head_idx % (n_heads/gqa_ratio);
    const int32_t tid = threadIdx.x;

    // Strides for input tensors (column-major)
    // Q/K/V: [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const int32_t qkv_stride_token = HEAD_DIM;
    const int32_t qkv_stride_head = HEAD_DIM * n_tokens;
    const int32_t qkv_stride_batch = HEAD_DIM * n_tokens * n_heads;
    const int32_t qkv_stride_batch_kq = qkv_stride_batch / gqa_ratio;

    // State: [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs]
    // For head h: columns h*HEAD_DIM to (h+1)*HEAD_DIM
    // state[row, col] for head h = state[row, h*HEAD_DIM + col]
    // Linear index: row + (h*HEAD_DIM + col) * HEAD_DIM = row + h*HEAD_DIM^2 + col*HEAD_DIM
    const int32_t state_head_offset = head_idx * HEAD_DIM * HEAD_DIM;
    const int32_t state_batch_stride = HEAD_DIM * HEAD_DIM * n_heads;

    // State step stride for save_all_states: HEAD_DIM^2 * n_heads * n_seqs
    const int32_t state_step_stride = HEAD_DIM * HEAD_DIM * n_heads * n_seqs;

    // Pointers for this batch/head
    const float * q_ptr = q + batch_idx * qkv_stride_batch_kq + head_idx_kq * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch_kq + head_idx_kq * qkv_stride_head;
    const float * v_ptr = v + batch_idx * vnb3 + head_idx * vnb2;
    const float * g_ptr = g + batch_idx * gnb3 + head_idx * gnb2;
    const float * beta_ptr = beta_in + batch_idx * bnb3 + head_idx * bnb2;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;

    // Output layout: [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]
    // For [dim, head, token, batch]: index = dim + head*S_v + token*S_v*H_v + batch*S_v*H_v*n_tokens
    float * out_base = dst + batch_idx * (HEAD_DIM * n_heads * n_tokens) + head_idx * HEAD_DIM;
    const int32_t out_token_stride = HEAD_DIM * n_heads;  // stride between tokens
    float * state_dst = state_out + batch_idx * state_batch_stride + state_head_offset;

    // Shared memory for current token's Q, K, V (normalized), and intermediate results
    extern __shared__ float smem[];
    float * sQ = smem;                          // HEAD_DIM
    float * sK = sQ + HEAD_DIM;                 // HEAD_DIM

    const float scale = rsqrtf((float)HEAD_DIM);

    __shared__ float sum_helper[block_size/WARP_SIZE];

    constexpr int32_t num_warps = block_size/WARP_SIZE;
    const int32_t row = tid % WARP_SIZE;
    const int32_t col_idx_0 = tid / WARP_SIZE;
    const int32_t row_out = row + sub_idx * WARP_SIZE;

    // Keep the state in registers, copy the final state to its destination at the end
    float state_local[HEAD_DIM/num_warps];
    for (int32_t i = 0; i < HEAD_DIM/num_warps; ++i) {
        int32_t col = num_warps*i + col_idx_0;
        state_local[i] = state_src[col*HEAD_DIM + row_out];
    }

    constexpr int32_t WARP_SIZE_S = WARP_SIZE + 1;
    constexpr int32_t num_stored_rows = block_size/WARP_SIZE;
    __shared__ float all_sum[2*WARP_SIZE_S*num_stored_rows];
    auto all_sum1 = all_sum;
    auto all_sum2 = all_sum1 + WARP_SIZE_S*num_stored_rows;

    for (int32_t t = 0; t < n_tokens; t++) {
        float sum_kq = 0.0f;
        for (int32_t i = tid; i < HEAD_DIM; i += block_size) {
            sQ[i] = q_ptr[t * qkv_stride_token + i] * scale;
            sK[i] = k_ptr[t * qkv_stride_token + i];
            sum_kq += sK[i] * sQ[i];
        }

        float attn_score = reduce_sum<block_size>(sum_kq, sum_helper);

        float beta_val = sigmoid_f(beta_ptr[t*bnb1]);

        float sum1 = 0, sum2 = 0;
#pragma unroll
        for (int32_t i = 0; i < HEAD_DIM/num_warps; ++i) {
            int32_t col = num_warps*i + col_idx_0;
            float decay = row == 0 ? expf(fminf(g_ptr[t*gnb0 + col*gnb1], 50.0f)) : 0.0f;
            decay = __shfl_sync(0xFFFFFFFF, decay, 0, WARP_SIZE);
            state_local[i] *= decay;
            sum1 += state_local[i] * sK[col];
            sum2 += state_local[i] * sQ[col];
        }
        all_sum1[col_idx_0*WARP_SIZE_S + row] = sum1;
        all_sum2[col_idx_0*WARP_SIZE_S + row] = sum2;
        __syncthreads();

        sum1 = sum2 = 0;
#pragma unroll
        for (int32_t i = 0; i < block_size/WARP_SIZE; ++i) {
            sum1 += all_sum1[i*WARP_SIZE_S + row];
            sum2 += all_sum2[i*WARP_SIZE_S + row];
        }

        float sv_new = beta_val * (v_ptr[t * vnb1 + row_out] - sum1);
        if (col_idx_0 == 0) {
            out_base[t * out_token_stride + row_out] = sum2 + sv_new * attn_score;
        }

        for (int32_t i = 0; i < HEAD_DIM/num_warps; ++i) {
            int32_t col = num_warps*i + col_idx_0;
            float new_state_val = state_local[i] + sv_new * sK[col];
            new_state_val = fminf(fmaxf(new_state_val, -1e6f), 1e6f);
            state_local[i] = new_state_val;
        }

        // Save per-step state if requested
        if (saved_states && t < n_tokens - 1) {
            float * state_step_dst = saved_states + batch_idx * state_batch_stride + state_head_offset + t * state_step_stride;
            for (int32_t i = 0; i < HEAD_DIM/num_warps; ++i) {
                int32_t col = num_warps*i + col_idx_0;
                state_step_dst[col*HEAD_DIM + row_out] = state_local[i];
            }
        }

        // Barrier required: (a) sK reads in the state update above must complete
        // before next iteration overwrites sK at the top of the loop, and (b) this
        // single barrier also orders all_sum1/all_sum2 reads above vs. the next
        // iteration's writes — subsuming the prior barriers after the cross-warp
        // reduction and after the loop exit.
        __syncthreads();
    }
    // Copy the final state to its destination
    for (int32_t i = 0; i < HEAD_DIM/num_warps; ++i) {
        int32_t col = num_warps*i + col_idx_0;
        state_dst[col*HEAD_DIM + row_out] = state_local[i];
    }
}

static void kda_f32_cuda(
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    const float * state_in,
    float * dst,
    float * state_out,
    float * saved_states,
    const int32_t head_dim,
    const int32_t n_tokens,
    const int32_t n_heads,
    const int32_t gqa_ratio,
    const int32_t repeat_type,
    const int32_t n_seqs,
    size_t vnb1, size_t vnb2, size_t vnb3,
    size_t gnb0, size_t gnb1, size_t gnb2, size_t gnb3,
    size_t bnb1, size_t bnb2, size_t bnb3,
    cudaStream_t stream) {
    if (head_dim != 64 && head_dim != 128) {
        GGML_ABORT("Unsupported KDA head size");
    }

    const int32_t num_blocks = n_seqs * n_heads * (head_dim/WARP_SIZE);
    const size_t smem_size = 2 * head_dim * sizeof(float);

    if (n_tokens <= 8) {
        constexpr int32_t threads_per_block = 256;
        if (head_dim == 64) {
            kda_recurrent_f32<64, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, state_out, saved_states, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs,
                    vnb1, vnb2, vnb3, gnb0, gnb1, gnb2, gnb3, bnb1, bnb2, bnb3);
        } else {
            kda_recurrent_f32<128, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, state_out, saved_states, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs,
                    vnb1, vnb2, vnb3, gnb0, gnb1, gnb2, gnb3, bnb1, bnb2, bnb3);
        }
    } else {
        constexpr int32_t threads_per_block = 128;
        if (head_dim == 64) {
            kda_recurrent_f32<64, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, state_out, saved_states, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs,
                    vnb1, vnb2, vnb3, gnb0, gnb1, gnb2, gnb3, bnb1, bnb2, bnb3);
        } else {
            kda_recurrent_f32<128, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, state_out, saved_states, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs,
                    vnb1, vnb2, vnb3, gnb0, gnb1, gnb2, gnb3, bnb1, bnb2, bnb3);
        }
    }

    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_kda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    const ggml_tensor * src3 = dst->src[3];
    const ggml_tensor * src4 = dst->src[4];
    const ggml_tensor * src5 = dst->src[5];
    const ggml_tensor * src6 = dst->src[6];
    const ggml_tensor * src7 = dst->src[7];

    const int32_t head_dim = (int32_t) src0->ne[0];
    const int32_t n_tokens = (int32_t) src0->ne[1];
    const int32_t n_heads = (int32_t) src2->ne[2];
    const int32_t n_heads_kq = (int32_t) src0->ne[2];
    const int32_t n_seqs = (int32_t) src0->ne[3];
    const int32_t gqa_ratio = n_heads / n_heads_kq;
    const int32_t repeat_type = dst->op_params[0];
    const size_t output_size = (size_t) head_dim * n_tokens * n_heads * n_seqs;

    kda_f32_cuda(
        (const float *)src0->data,
        (const float *)src1->data,
        (const float *)src2->data,
        (const float *)src3->data,
        (const float *)src4->data,
        (const float *)src5->data,
        (float *)dst->data,
        src7 ? (float *)src7->data : (float *)dst->data + output_size,
        src6 ? (float *)src6->data : nullptr,
        head_dim, n_tokens, n_heads, gqa_ratio, repeat_type, n_seqs,
        src2->nb[1]/sizeof(float), src2->nb[2]/sizeof(float), src2->nb[3]/sizeof(float),
        src3->nb[0]/sizeof(float), src3->nb[1]/sizeof(float), src3->nb[2]/sizeof(float), src3->nb[3]/sizeof(float),
        src4->nb[1]/sizeof(float), src4->nb[2]/sizeof(float), src4->nb[3]/sizeof(float),
        ctx.stream());
}
