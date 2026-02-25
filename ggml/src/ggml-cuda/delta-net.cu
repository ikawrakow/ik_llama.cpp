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
    const int batch_idx = blockIdx.x / n_heads;
    const int head_idx  = blockIdx.x % n_heads;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;  // 0-7 for 256 threads
    const int lane_id = tid % WARP_SIZE;  // 0-31
    constexpr int NUM_WARPS = block_size/WARP_SIZE;

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
    float * sV = sK + HEAD_DIM;                 // HEAD_DIM
    float * sVNew = sV + HEAD_DIM;              // HEAD_DIM

    const float scale = rsqrtf((float)HEAD_DIM);

    __shared__ float sum_helper[block_size/WARP_SIZE];

    // Copy initial state to output buffer (will be updated in place)
    for (int i = tid; i < HEAD_DIM * HEAD_DIM; i += block_size) {
        state_dst[i] = state_src[i];
    }

    __shared__ float all_sum[2*HEAD_DIM*NUM_WARPS];
    auto all_sum1 = all_sum;
    auto all_sum2 = all_sum1 + HEAD_DIM*NUM_WARPS;

    // Process each token sequentially
    for (int64_t t = 0; t < n_tokens; t++) {

        float sum_kq = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += block_size) {
            sQ[i] = q_ptr[t * qkv_stride_token + i];
            sK[i] = k_ptr[t * qkv_stride_token + i];
            sV[i] = v_ptr[t * qkv_stride_token + i];
            sum_kq += sK[i] * sQ[i];
        }

        sum_kq = reduce_sum<block_size>(sum_kq, sum_helper);

        float beta_val = sigmoid_f(beta_ptr[t]);
        float decay    = expf(fminf(g_ptr[t], 50.0f));

        float attn_score = sum_kq * scale;

        for (int row_out = lane_id; row_out < HEAD_DIM; row_out += WARP_SIZE) {
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            #pragma unroll
            for (int col = warp_id; col < HEAD_DIM; col += NUM_WARPS) {
                float sval = state_dst[row_out + col * HEAD_DIM];
                sum1 += sval * sK[col];
                sum2 += sval * sQ[col];
            }
            all_sum1[warp_id*HEAD_DIM + row_out] = sum1;
            all_sum2[warp_id*HEAD_DIM + row_out] = sum2;
        }
        __syncthreads();

        for (int row_out = tid; row_out < HEAD_DIM; row_out += block_size) {
            float sum1 = all_sum1[row_out];
            float sum2 = all_sum2[row_out];
            for (int i = 1; i < NUM_WARPS; ++i) {
                sum1 += all_sum1[row_out + i*HEAD_DIM];
                sum2 += all_sum2[row_out + i*HEAD_DIM];
            }
            sum1 *= beta_val * decay;
            sum2 *= scale * decay;
            sVNew[row_out] = sV[row_out] * beta_val - sum1;
            float v_attn = sVNew[row_out] * attn_score;
            out_base[t * out_token_stride + row_out] = sum2 + v_attn;
        }
        __syncthreads();

        //for (int row_out = warp_id; row_out < HEAD_DIM; row_out += NUM_WARPS) {
        //    float sum1 = 0.0f;
        //    float sum2 = 0.0f;
        //    #pragma unroll
        //    for (int col = lane_id; col < HEAD_DIM; col += WARP_SIZE) {
        //        float sval = state_dst[row_out + col * HEAD_DIM];
        //        sum1 += sval * sK[col];
        //        sum2 += sval * sQ[col];
        //    }
        //    sum1 = warp_reduce_sum(sum1) * beta_val * decay;
        //    sum2 = warp_reduce_sum(sum2) * scale * decay;
        //    if (lane_id == 0) {
        //        sVNew[row_out] = sV[row_out] * beta_val - sum1;
        //        float v_attn = sVNew[row_out] * attn_score;
        //        out_base[t * out_token_stride + row_out] = sum2 + v_attn;
        //    }
        //}
        //__syncthreads();

        for (int out_dim = warp_id; out_dim < HEAD_DIM; out_dim += NUM_WARPS) {
            #pragma unroll
            for (int row = lane_id; row < HEAD_DIM; row += WARP_SIZE) {
                float state_val = state_dst[row + out_dim * HEAD_DIM];
                float new_state_val = decay * state_val + sVNew[row] * sK[out_dim];
                new_state_val = fminf(fmaxf(new_state_val, -1e6f), 1e6f);
                state_dst[row + out_dim * HEAD_DIM] = new_state_val;
            }
        }
        //if (t < n_tokens - 1) {
        //    __syncthreads();
        //}

    }
}

// Generic kernel that handles any HEAD_DIM at runtime (slower but flexible)
__global__ void delta_net_recurrent_generic_f32(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ g,
    const float * __restrict__ beta_in,
    const float * __restrict__ state_in,
    float * __restrict__ dst,
    const int64_t head_dim,
    const int64_t n_tokens,
    const int64_t n_heads,
    const int64_t n_seqs,
    const int64_t output_offset,
    const float eps) {
    const int batch_idx = blockIdx.x / n_heads;
    const int head_idx = blockIdx.x % n_heads;
    const int tid = threadIdx.x;

    // Strides (column-major)
    const int64_t qkv_stride_token = head_dim;
    const int64_t qkv_stride_head = head_dim * n_tokens;
    const int64_t qkv_stride_batch = head_dim * n_tokens * n_heads;

    const int64_t g_stride_head = n_tokens;
    const int64_t g_stride_batch = n_tokens * n_heads;

    const int64_t state_head_offset = head_idx * head_dim * head_dim;
    const int64_t state_batch_stride = head_dim * head_dim * n_heads;

    // Pointers
    const float * q_ptr = q + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * v_ptr = v + batch_idx * qkv_stride_batch + head_idx * qkv_stride_head;
    const float * g_ptr = g + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * beta_ptr = beta_in + batch_idx * g_stride_batch + head_idx * g_stride_head;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;

    // Output layout: [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]
    float * out_base = dst + batch_idx * (head_dim * n_heads * n_tokens) + head_idx * head_dim;
    const int64_t out_token_stride = head_dim * n_heads;
    float * state_dst = dst + output_offset + batch_idx * state_batch_stride + state_head_offset;

    // Shared memory for scalars (outside loop)
    __shared__ float shared_g_val, shared_beta_val, shared_decay, shared_attn_score;

    // Dynamic shared memory
    extern __shared__ float smem[];
    float * sQ = smem;
    float * sK = sQ + head_dim;
    float * sV = sK + head_dim;
    float * sKBeta = sV + head_dim;             // plain k for state update
    float * sVBeta = sKBeta + head_dim;         // v * sigmoid(beta)
    float * sOut = sVBeta + head_dim;
    float * sKCumdecay = sOut + head_dim;       // k * sigmoid(beta) * exp(g)
    float * sVPrime = sKCumdecay + head_dim;    // state @ k_cumdecay
    float * sVNew = sVPrime + head_dim;         // v_beta - v_prime
    float * sNorm = sVNew + head_dim;

    const float scale = rsqrtf((float)head_dim);

    // Copy initial state to output buffer
    for (int i = tid; i < head_dim * head_dim; i += blockDim.x) {
        int col = i / head_dim;
        int row = i % head_dim;
        state_dst[row + col * head_dim] = state_src[row + col * head_dim];
    }
    __syncthreads();

    // Process each token
    for (int64_t t = 0; t < n_tokens; t++) {
        if (tid < 2) sNorm[tid] = 0.0f;
        __syncthreads();

        // Load Q, K, V
        for (int i = tid; i < head_dim; i += blockDim.x) {
            sQ[i] = q_ptr[t * qkv_stride_token + i];
            sK[i] = k_ptr[t * qkv_stride_token + i];
            sV[i] = v_ptr[t * qkv_stride_token + i];
        }
        __syncthreads();

        // L2 normalize Q and K
        float q_sq = 0.0f, k_sq = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            q_sq += sQ[i] * sQ[i];
            k_sq += sK[i] * sK[i];
        }

        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            q_sq += __shfl_xor_sync(0xffffffff, q_sq, offset);
            k_sq += __shfl_xor_sync(0xffffffff, k_sq, offset);
        }

        if (tid % WARP_SIZE == 0) {
            atomicAdd(&sNorm[0], q_sq);
            atomicAdd(&sNorm[1], k_sq);
        }
        __syncthreads();

        float q_norm = rsqrtf(sNorm[0] + eps);
        float k_norm = rsqrtf(sNorm[1] + eps);

        for (int i = tid; i < head_dim; i += blockDim.x) {
            sQ[i] *= q_norm * scale;
            sK[i] *= k_norm;
        }
        __syncthreads();

        // Load g and beta, compute decay
        if (tid == 0) {
            shared_g_val = g_ptr[t];
            shared_beta_val = sigmoid_f(beta_ptr[t]);
            shared_decay = expf(fminf(shared_g_val, 50.0f));
        }
        __syncthreads();

        float beta_val = shared_beta_val;
        float decay = shared_decay;

        // Compute k_beta, v_beta, k_cumdecay
        for (int i = tid; i < head_dim; i += blockDim.x) {
            sKBeta[i] = sK[i];
            sVBeta[i] = sV[i] * beta_val;
            sKCumdecay[i] = sK[i] * beta_val * decay;
        }
        __syncthreads();

        // Compute v_prime = state @ k_cumdecay
        for (int row_out = tid; row_out < head_dim; row_out += blockDim.x) {
            float v_prime_val = 0.0f;
            for (int col = 0; col < head_dim; col++) {
                // Access state[row_out, col] = state_dst[row_out + col * head_dim] for state @ k
                v_prime_val += state_dst[row_out + col * head_dim] * sKCumdecay[col];
            }
            sVPrime[row_out] = v_prime_val;
        }
        __syncthreads();

        // Compute v_new = v_beta - v_prime (the value residual)
        for (int i = tid; i < head_dim; i += blockDim.x) {
            sVNew[i] = sVBeta[i] - sVPrime[i];
        }
        __syncthreads();

        // Compute attn_score = dot(k, q) (L2 normalized vectors)
        if (tid == 0) {
            float dot_sum = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                dot_sum += sK[i] * sQ[i];
            }
            shared_attn_score = dot_sum;
        }
        __syncthreads();

        // Compute output: o[t] = attn_inter + v_attn
        // attn_inter = state @ (q * exp(g)) = sum_col(state[row_out, col] * q[col] * exp(g))
        // The decomposed path uses: attn_inter = ggml_mul_mat(state_t, q_g_exp)
        // Since ggml_mul_mat(A,B) = A^T @ B, attn_inter = state_t^T @ q_g_exp = state @ (q * exp(g))
        for (int row_out = tid; row_out < head_dim; row_out += blockDim.x) {
            float attn_inter = 0.0f;

            for (int col = 0; col < head_dim; col++) {
                // Access state[row_out, col] = state_dst[row_out + col * head_dim] for state @ q
                float state_val = state_dst[row_out + col * head_dim];
                attn_inter += sQ[col] * decay * state_val;
            }

            // v_attn = v_new * attn_score
            float v_attn = sVNew[row_out] * shared_attn_score;

            // Output = attn_inter + v_attn (correct DeltaNet formula)
            sOut[row_out] = attn_inter + v_attn;
        }
        __syncthreads();

        // Update state: state_new = decay * state + outer(v_new, k)
        // Fixed: outer product orientation matches decomposed: state[v_idx, k_idx] += v_new[v_idx] * k[k_idx]
        // Uses transposed indexing: state_dst[row + out_dim * head_dim] = state[row][out_dim]
        // Only protect against NaN/Inf - do NOT clamp decay value
        float safe_decay = decay;
        if (isnan(safe_decay) || isinf(safe_decay)) {
            safe_decay = 1.0f;
        }

        for (int out_dim = tid; out_dim < head_dim; out_dim += blockDim.x) {
            for (int row = 0; row < head_dim; row++) {
                float state_val = state_dst[row + out_dim * head_dim];

                // state_new[row][out_dim] = decay * state[row][out_dim] + v_new[row] * k[out_dim]
                // Fix: outer product matches decomposed path: state[v_idx, k_idx] += v_new[v_idx] * k[k_idx]
                float new_state_val = safe_decay * state_val + sVNew[row] * sKBeta[out_dim];

                // Clamp state to prevent overflow
                new_state_val = fminf(fmaxf(new_state_val, -1e6f), 1e6f);
                state_dst[row + out_dim * head_dim] = new_state_val;
            }
        }
        __syncthreads();

        // Write output
        for (int i = tid; i < head_dim; i += blockDim.x) {
            out_base[t * out_token_stride + i] = sOut[i];
        }
        __syncthreads();
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

    // One block per (batch, head) pair
    const int num_blocks = n_seqs * n_heads;
    constexpr int threads_per_block = 512; //256;

    // Shared memory: 9 * head_dim (for Q, K, V, KBeta, VBeta, Out, KCumdecay, VPrime, VNew)
    // Plus 6 floats for Norm[2], g_val, beta_val, decay, attn_score
    //const size_t smem_size = (9 * head_dim + 6) * sizeof(float);
    //const size_t smem_size = (4 * head_dim + 2 * n_tokens) * sizeof(float);
    const size_t smem_size = 4 * head_dim * sizeof(float);

    // Use templated kernel for common head dimensions, generic for others
    if (head_dim == 64) {
        delta_net_recurrent_f32<64, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
            q, k, v, g, beta, state_in, dst, n_heads, n_tokens, n_seqs, output_offset, eps);
    } else if (head_dim == 128) {
        GGML_ASSERT(num_blocks % 8 == 0);
        delta_net_recurrent_f32<128, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, n_heads, n_tokens, n_seqs, output_offset, eps);
    } else {
        delta_net_recurrent_generic_f32<<<num_blocks, threads_per_block, smem_size, stream>>>(
            q, k, v, g, beta, state_in, dst, head_dim, n_tokens, n_heads, n_seqs, output_offset, eps);
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
