#include "common.cuh"
#include "delta-net.cuh"
#include <cstdlib>
#include <cstring>

// Delta Net Linear Attention Kernel for Qwen3-Next (HEAD_DIM=128)
// State layout: [S_v, S_v*H_v, 1, n_seqs] (column-major)

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
__global__ void delta_net_recurrent_f32(
    const float * __restrict__ q,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ k,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ v,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ g,         // [n_tokens, 1, n_heads, n_seqs]
    const float * __restrict__ beta_in,   // [1, n_tokens, n_heads, n_seqs]
    const float * __restrict__ state_in,  // [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs]
    float * __restrict__ dst,             // output + new_state concatenated
    const int64_t n_heads,
    const int64_t n_tokens,
    const int64_t n_seqs,
    const int64_t output_offset,          // offset where state starts in output
    const float eps) {
    constexpr int warps_per_head = HEAD_DIM/WARP_SIZE;
    const int batch_idx = blockIdx.x / (warps_per_head*n_heads);
    const int sub_head_idx  = blockIdx.x % (warps_per_head*n_heads);
    const int head_idx = sub_head_idx / warps_per_head;
    const int sub_idx  = sub_head_idx % warps_per_head;
    const int tid = threadIdx.x;

    // Strides for input tensors (column-major)
    // Q/K/V: [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const int64_t qkv_stride_token = HEAD_DIM;
    const int64_t qkv_stride_head = HEAD_DIM * n_tokens;
    const int64_t qkv_stride_batch = HEAD_DIM * n_tokens * n_heads;

    // G/Beta: [n_tokens, 1, n_heads, n_seqs] / [1, n_tokens, n_heads, n_seqs]
    const int64_t g_stride_head = n_tokens;
    const int64_t g_stride_batch = n_tokens * n_heads;

    // State: [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs]
    // For head h: columns h*HEAD_DIM to (h+1)*HEAD_DIM
    // state[row, col] for head h = state[row, h*HEAD_DIM + col]
    // Linear index: row + (h*HEAD_DIM + col) * HEAD_DIM = row + h*HEAD_DIM^2 + col*HEAD_DIM
    const int64_t state_head_offset = head_idx * HEAD_DIM * HEAD_DIM;
    const int64_t state_batch_stride = HEAD_DIM * HEAD_DIM * n_heads;

    // Pointers for this batch/head
    const float * q_ptr = q + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * v_ptr = v + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * g_ptr = g + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * beta_ptr = beta_in + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;

    // Output layout: [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]
    // For [dim, head, token, batch]: index = dim + head*S_v + token*S_v*H_v + batch*S_v*H_v*n_tokens
    float * out_base = dst + batch_idx * (HEAD_DIM * n_heads * n_tokens) + head_idx * HEAD_DIM;
    const int64_t out_token_stride = HEAD_DIM * n_heads;  // stride between tokens
    float * state_dst = dst + output_offset + batch_idx * state_batch_stride + state_head_offset;

    // Shared memory for current token's Q, K, V (normalized), and intermediate results
    extern __shared__ float smem[];
    float * sQ = smem;                          // HEAD_DIM
    float * sK = sQ + HEAD_DIM;                 // HEAD_DIM

    const float scale = rsqrtf((float)HEAD_DIM);

    __shared__ float sum_helper[block_size/WARP_SIZE];

    constexpr int num_warps = block_size/WARP_SIZE;
    const int row = tid % WARP_SIZE;
    const int col_idx_0 = tid / WARP_SIZE;
    const int row_out = row + sub_idx * WARP_SIZE;

    // Keep the state in registers, copy the final state to its destination at the end
    float state_local[HEAD_DIM/num_warps];
    for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
        int col = num_warps*i + col_idx_0;
        state_local[i] = state_src[col*HEAD_DIM + row_out];
    }

    constexpr int WARP_SIZE_S = WARP_SIZE + 1;
    constexpr int num_stored_rows = block_size/WARP_SIZE;
    __shared__ float all_sum[2*WARP_SIZE_S*num_stored_rows];
    auto all_sum1 = all_sum;
    auto all_sum2 = all_sum1 + WARP_SIZE_S*num_stored_rows;

    for (int64_t t = 0; t < n_tokens; t++) {
        float sum_kq = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += block_size) {
            sQ[i] = q_ptr[t * qkv_stride_token + i] * scale;
            sK[i] = k_ptr[t * qkv_stride_token + i];
            sum_kq += sK[i] * sQ[i];
        }

        float attn_score = reduce_sum<block_size>(sum_kq, sum_helper);

        float beta_val = sigmoid_f(beta_ptr[t]);
        float decay    = expf(fminf(g_ptr[t], 50.0f));

        float sum1 = 0, sum2 = 0;
#pragma unroll
        for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
            int col = num_warps*i + col_idx_0;
            sum1 += state_local[i] * sK[col];
            sum2 += state_local[i] * sQ[col];
        }
        all_sum1[col_idx_0*WARP_SIZE_S + row] = sum1;
        all_sum2[col_idx_0*WARP_SIZE_S + row] = sum2;

        __syncthreads();

        sum1 = sum2 = 0;
#pragma unroll
        for (int i = 0; i < block_size/WARP_SIZE; ++i) {
            sum1 += all_sum1[i*WARP_SIZE_S + row];
            sum2 += all_sum2[i*WARP_SIZE_S + row];
        }
        // To be honest, I don't understand why we need this sync. But without it I observe results varying from run to run
        __syncthreads();

        float sv_new = beta_val * (v_ptr[t * qkv_stride_token + row_out] - sum1 * decay);
        if (col_idx_0 == 0) {
            out_base[t * out_token_stride + row_out] = sum2 * decay + sv_new * attn_score;
        }

        for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
            int col = num_warps*i + col_idx_0;
            float new_state_val = decay * state_local[i] + sv_new * sK[col];
            new_state_val = fminf(fmaxf(new_state_val, -1e6f), 1e6f);
            state_local[i] = new_state_val;
        }

    }
    // Copy the final state to its destination
    for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
        int col = num_warps*i + col_idx_0;
        state_dst[col*HEAD_DIM + row_out] = state_local[i];
    }
}

static void delta_net_f32_cuda(
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    const float * state_in,
    float * dst,
    const int64_t head_dim,
    const int64_t n_tokens,
    const int64_t n_heads,
    const int64_t n_seqs,
    const float eps,
    const int device_id,
    const int cc,  // compute capability (e.g., 890 for SM 8.9, 1200 for SM 12.0)
    cudaStream_t stream) {
    GGML_UNUSED(device_id);
    GGML_UNUSED(cc);

    const int64_t output_offset = head_dim * n_tokens * n_heads * n_seqs;

    if (head_dim != 64 && head_dim != 128) {
        GGML_ABORT("Unsupported delta net head size");
    }

    GGML_ASSERT(head_dim % WARP_SIZE == 0);
    const int num_blocks = n_seqs * n_heads * (head_dim/WARP_SIZE);
    const size_t smem_size = 2 * head_dim * sizeof(float);

    if (n_tokens <= 8) {
        constexpr int threads_per_block = 256;
        if (head_dim == 64) {
            delta_net_recurrent_f32<64, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, n_heads, n_tokens, n_seqs, output_offset, eps);
        } else {
            delta_net_recurrent_f32<128, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, n_heads, n_tokens, n_seqs, output_offset, eps);
        }
    } else {
        constexpr int threads_per_block = 128;
        if (head_dim == 64) {
            delta_net_recurrent_f32<64, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, n_heads, n_tokens, n_seqs, output_offset, eps);
        } else {
            delta_net_recurrent_f32<128, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, n_heads, n_tokens, n_seqs, output_offset, eps);
        }
    }

    CUDA_CHECK(cudaGetLastError());

}

void ggml_cuda_op_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // q
    const ggml_tensor * src1 = dst->src[1];  // k
    const ggml_tensor * src2 = dst->src[2];  // v
    const ggml_tensor * src3 = dst->src[3];  // g
    const ggml_tensor * src4 = dst->src[4];  // beta
    const ggml_tensor * src5 = dst->src[5];  // state

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t head_dim = src0->ne[0];
    const int64_t n_tokens = src0->ne[1];
    const int64_t n_heads = src0->ne[2];
    const int64_t n_seqs = src0->ne[3];

    // Dimension validation
    // Q/K: [head_dim, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src1->ne[0] == head_dim && src1->ne[1] == n_tokens && src1->ne[2] == n_heads && src1->ne[3] == n_seqs);
    // V: [head_dim, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src2->ne[0] == head_dim && src2->ne[1] == n_tokens && src2->ne[2] == n_heads && src2->ne[3] == n_seqs);
    // G: [n_tokens, 1, n_heads, n_seqs]
    GGML_ASSERT(src3->ne[0] == n_tokens && src3->ne[1] == 1 && src3->ne[2] == n_heads && src3->ne[3] == n_seqs);
    // Beta: [1, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src4->ne[0] == 1 && src4->ne[1] == n_tokens && src4->ne[2] == n_heads && src4->ne[3] == n_seqs);
    // State: [head_dim, head_dim*n_heads, 1, n_seqs]
    GGML_ASSERT(src5->ne[0] == head_dim && src5->ne[1] == head_dim * n_heads && src5->ne[2] == 1 && src5->ne[3] == n_seqs);

    // Verify output tensor size
    const int64_t output_size = head_dim * n_tokens * n_heads * n_seqs;
    const int64_t state_size = head_dim * head_dim * n_heads * n_seqs;
    GGML_ASSERT(ggml_nelements(dst) == output_size + state_size);

    const float eps = 1e-6f;

    GGML_ASSERT(head_dim <= 256);  // Reasonable limit for shared memory

    // Get device info from ctx (avoids calling CUDA runtime APIs inside dispatch)
    const int device_id = ctx.device;
    const int cc = ggml_cuda_info().devices[device_id].cc;

    delta_net_f32_cuda(
        (const float *)src0->data,
        (const float *)src1->data,
        (const float *)src2->data,
        (const float *)src3->data,
        (const float *)src4->data,
        (const float *)src5->data,
        (float *)dst->data,
        head_dim, n_tokens, n_heads, n_seqs, eps,
        device_id, cc,
        ctx.stream());

}
