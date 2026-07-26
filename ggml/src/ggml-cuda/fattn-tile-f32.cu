//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tile-f32.cuh"

#define FATTN_KQ_STRIDE_TILE_F32 32

// S3 exact-retile (sm_60/P100). Same fp32 arithmetic as the original all-fp32 tile kernel -- fp32 Q
// (scaled in fp32), exact fp16->fp32 K/V conversion, fp32 products, fp32 scores and online softmax,
// fp32 post-exp probability store, fp32 P.V accumulate -- restructured to run a leaner inner loop:
//   * K/V tiles are staged in shared memory as raw half2 exactly as they arrive from the fp16 KV
//     cache and converted to fp32 in registers at the point of use. fp16->fp32 conversion is exact,
//     so every FFMA sees bit-identical inputs to the original fp32-staged tiles. This halves the K/V
//     tile (16512 B -> 8320 B), roughly halving the QK loop's shared-memory loads.
//   * Raw KQ scores never round-trip through shared memory: the (threadIdx.x, threadIdx.y) ->
//     (i_KQ, j_KQ) mapping of the score pass is identical to the softmax pass, so the scores stay
//     in registers and only the post-exp fp32 probabilities are stored (unchanged values, one
//     tile-sized smem round trip AND one __syncthreads() fewer per KV tile).
// The measured +4-9% vs the un-retiled fp32 tile kernel on P100 is a COMPOUND of two effects the half2
// staging produces together, and we did not isolate their individual shares: (1) an occupancy gain -- the
// smem drop 36992 -> 28800 B/block lets 2 blocks fit per SM where the un-retiled kernel fits only 1
// (2 x 28800 = 57600 <= 65536; 80 regs would allow 3, so smem is the ceiling); and (2) a leaner inner
// loop (fewer smem loads, the dropped score round-trip, one fewer barrier). __launch_bounds__(..., 2)
// only PINS the target the smem reduction already reaches (a (..., 1) bound compiles bit-identically), so
// the directive is not itself a lever; and a 3rd block does not help (the branch cost-curve and an
// implemented Q_f-to-registers attempt that pushed regs 80 -> 124 both confirm smem caps occupancy at 2).
// Scope: this kernel is only the explicit GGML_PREC_F32 tile path. On P100 that is where the fp16
// fast path is routed off; standalone it also affects sm_61 consumer-Pascal prefill, not benchmarked.
// The ONLY numerical difference vs the original kernel: the QK dot product accumulates its D terms
// in natural order 0,1,...,D-1, while the original accumulated them even/odd-deinterleaved within
// each 64-element chunk (an artifact of its fp32 staging layout). Both are all-fp32 sums of the same
// exact products; individual scores may differ by ~1 ulp. The P.V accumulation order is bit-identical.

template<int D, int ncols, int nwarps, int parallel_blocks, bool use_softcap> // D == head size
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 2)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_tile_ext_f32(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const float softcap,
        const uint32_t n_head_log2,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {

    // Skip unused kernel variants for faster compilation:
    if (use_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = (blockIdx.x / parallel_blocks) * ncols; // Index of the Q/QKV column to work on.
    const int ip  =  blockIdx.x % parallel_blocks; // Index in group of blocks running for the same column in parallel.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float2 * Q_f2  = (const float2 *) (Q    + nb02* blockIdx.y              + nb01*ic0);
    const half2  * K_h2  = (const half2  *) (K    + nb12*(blockIdx.y / gqa_ratio));
    const half2  * V_h2  = (const half2  *) (V    + nb12*(blockIdx.y / gqa_ratio)); // K and V have same shape
    // Mask row stride must come from the mask tensor (nb31), not ne11 (= K->ne[1]). For SWA models the
    // fattn.cu n_swa windowing re-points K/V/mask to the last nton tokens (ne11 = nton) while the mask keeps
    // its original row stride, so indexing the mask by ne11 reads garbage and yields NaN on the tile kernels.
    const int    stride_mask = nb31 / sizeof(half);
    const half   * maskh  = (const half   *)  mask + stride_mask*ic0;
    const float  * sinksf = (const float  *)  sinks;

    const int stride_KV2 = nb11 / sizeof(half2);

    const float slope = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);
    static_assert(D % (2 * WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");

    __shared__ float KQ[ncols*FATTN_KQ_STRIDE_TILE_F32]; // post-exp fp32 probabilities only

    __shared__ half2 KV_tmp[FATTN_KQ_STRIDE_TILE_F32][D/2 + 1]; // Raw fp16 K/V tile; pad to avoid memory bank conflicts.

    float kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float kqsum[ncols/nwarps] = {0.0f};

    float2 VKQ[ncols/nwarps][(D/2)/WARP_SIZE] = {{{0.0f, 0.0f}}};

    // Store Q in shared memory, fp32 and pre-scaled (same values as the original kernel; float2-typed
    // natural-order layout so the QK loop below pairs it directly with half2 K):
    __shared__ float2 Q_f[ncols][D/2];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            float2 tmp = ic0 + j < ne01 ? Q_f2[j*(nb01/sizeof(float2)) + i] : make_float2(0.0f, 0.0f);
            tmp.x *= scale;
            tmp.y *= scale;
            Q_f[j][i] = tmp;
        }
    }

    __syncthreads();

    const int k_start = parallel_blocks == 1 ? 0 : ip*FATTN_KQ_STRIDE_TILE_F32;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*FATTN_KQ_STRIDE_TILE_F32) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        float kqmax_new[ncols/nwarps];
#pragma unroll
        for (int j = 0; j < ncols/nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

        // Stage the K tile as raw half2 (no conversion here; converted in registers at use below):
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
                const int k_KQ = k_KQ_0 + threadIdx.x;

                KV_tmp[i_KQ][k_KQ] = K_h2[(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ];
            }
        }

        __syncthreads();

        float sum[FATTN_KQ_STRIDE_TILE_F32/WARP_SIZE][ncols/nwarps] = {{0.0f}};

#pragma unroll
        for (int k_KQ = 0; k_KQ < D/2; ++k_KQ) {
            float2 K_k[FATTN_KQ_STRIDE_TILE_F32/WARP_SIZE];
            float2 Q_k[ncols/nwarps];

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += WARP_SIZE) {
                const int i_KQ = i_KQ_0 + threadIdx.x;

                K_k[i_KQ_0/WARP_SIZE] = __half22float2(KV_tmp[i_KQ][k_KQ]); // exact fp16->fp32
            }
#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                Q_k[j_KQ_0/nwarps] = Q_f[j_KQ][k_KQ];
            }

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += WARP_SIZE) {
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                    // Two ordered fp32 FMAs == two scalar iterations of the original kernel.
                    sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += K_k[i_KQ_0/WARP_SIZE].x * Q_k[j_KQ_0/nwarps].x;
                    sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += K_k[i_KQ_0/WARP_SIZE].y * Q_k[j_KQ_0/nwarps].y;
                }
            }
        }

        // Softcap/mask/running-max on the raw scores. The scores stay in REGISTERS: this pass and the
        // softmax pass below use the same (threadIdx.x, threadIdx.y) -> (i_KQ, j_KQ) ownership as the
        // QK loop above, so no shared-memory round trip (and no __syncthreads()) is needed in between.
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F32; i_KQ_0 += WARP_SIZE) {
            const int i_KQ = i_KQ_0 + threadIdx.x;

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                if (use_softcap) {
                    sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] = softcap * tanhf(sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                }

                sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += mask ? slope*__half2float(maskh[j_KQ*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;

                kqmax_new[j_KQ_0/nwarps] = fmaxf(kqmax_new[j_KQ_0/nwarps], sum[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
            }
        }

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            kqmax_new[j0/nwarps] = warp_reduce_max(kqmax_new[j0/nwarps]);
            const float KQ_max_scale = expf(kqmax[j0/nwarps] - kqmax_new[j0/nwarps]);
            kqmax[j0/nwarps] = kqmax_new[j0/nwarps];

            float kqsum_add = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < FATTN_KQ_STRIDE_TILE_F32; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float diff = sum[i0/WARP_SIZE][j0/nwarps] - kqmax[j0/nwarps];
                const float val = expf(diff);
                kqsum_add += val;
                KQ[j*FATTN_KQ_STRIDE_TILE_F32 + i] = val; // fp32 post-exp probability store (exact path)
            }
            kqsum[j0/nwarps] = kqsum[j0/nwarps]*KQ_max_scale + kqsum_add;

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                VKQ[j0/nwarps][i0/WARP_SIZE].x *= KQ_max_scale;
                VKQ[j0/nwarps][i0/WARP_SIZE].y *= KQ_max_scale;
            }
        }

        __syncthreads();

        // Stage the V tile as raw half2. Reusing KV_tmp is safe: the __syncthreads() above orders these
        // writes after every thread's last K read (program order puts each thread's K reads before its
        // softmax pass, and the barrier puts all softmax passes before any V write).
#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F32; k0 += nwarps) {
            const int k = k0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                KV_tmp[k][i] = V_h2[(k_VKQ_0 + k)*stride_KV2 + i];
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < FATTN_KQ_STRIDE_TILE_F32; ++k) {
            float2 V_k[(D/2)/WARP_SIZE];
            float  KQ_k[ncols/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                V_k[i0/WARP_SIZE] = __half22float2(KV_tmp[k][i]); // exact fp16->fp32
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                const int j = j0 + threadIdx.y;

                KQ_k[j0/nwarps] = KQ[j*FATTN_KQ_STRIDE_TILE_F32 + k];
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                    VKQ[j0/nwarps][i0/WARP_SIZE].x += V_k[i0/WARP_SIZE].x*KQ_k[j0/nwarps];
                    VKQ[j0/nwarps][i0/WARP_SIZE].y += V_k[i0/WARP_SIZE].y*KQ_k[j0/nwarps];
                }
            }
        }

        __syncthreads();
    }

    // Apply attention sinks (e.g. gpt-oss): the sink is a per-head extra softmax logit whose value
    // contribution is zero, so it only joins the running max and rescales the denominator/value
    // accumulator. Only ip==0 adds the sink term so it is counted exactly once across the
    // parallel_blocks KV split. Ported from fattn-vec-f16.cuh. kqmax is warp-uniform here (already
    // warp_reduce_max'd in the KV loop); kqsum holds per-lane partials reduced in the epilogue, so
    // the sink term is added on a single lane.
    if (sinksf && ip == 0) {
        const float sink = sinksf[blockIdx.y];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const float kqmax_new_j = fmaxf(kqmax[j0/nwarps], sink);
            const float KQ_max_scale = expf(kqmax[j0/nwarps] - kqmax_new_j);
            kqmax[j0/nwarps] = kqmax_new_j;

            kqsum[j0/nwarps] = kqsum[j0/nwarps]*KQ_max_scale;
            if (threadIdx.x == 0) {
                kqsum[j0/nwarps] += expf(sink - kqmax[j0/nwarps]);
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                VKQ[j0/nwarps][i0/WARP_SIZE].x *= KQ_max_scale;
                VKQ[j0/nwarps][i0/WARP_SIZE].y *= KQ_max_scale;
            }
        }
    }

#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < ncols; j_VKQ_0 += nwarps) {
        const int j_VKQ = j_VKQ_0 + threadIdx.y;

        if (ic0 + j_VKQ >= ne01) {
            return;
        }

        float kqsum_j = kqsum[j_VKQ_0/nwarps];
        kqsum_j = warp_reduce_sum(kqsum_j);

#pragma unroll
        for (int i00 = 0; i00 < D; i00 += 2*WARP_SIZE) {
            const int i0 = i00 + 2*threadIdx.x;

            float2 dst_val = VKQ[j_VKQ_0/nwarps][i0/(2*WARP_SIZE)];
            if (parallel_blocks == 1) {
                dst_val.x /= kqsum_j;
                dst_val.y /= kqsum_j;
            }
            const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;
            dst[j_dst*D*gridDim.y + D*blockIdx.y + i0 + 0] = dst_val.x;
            dst[j_dst*D*gridDim.y + D*blockIdx.y + i0 + 1] = dst_val.y;
        }

        if (parallel_blocks != 1 && threadIdx.x == 0) {
            dst_meta[(ic0 + j_VKQ)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = make_float2(kqmax[j_VKQ_0/nwarps], kqsum_j);
        }
    }
}

template <int cols_per_block, int parallel_blocks, bool use_softcap>
void launch_fattn_tile_f32_64_128(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (Q->ne[0]) {
        case  64: {
            constexpr int      D = 64;
            constexpr int nwarps = 8;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f32<D, cols_per_block, nwarps, parallel_blocks, use_softcap>;
            launch_fattn<D, D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block, true, true);
        } break;
        case 128: {
            constexpr int      D = 128;
            constexpr int nwarps = 8;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f32<D, cols_per_block, nwarps, parallel_blocks, use_softcap>;
            launch_fattn<D, D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block, true, true);
        } break;
        default: {
            GGML_ABORT("FlashAttention without tensor cores only supports head sizes 64 and 128.");
        } break;
    }
}

void ggml_cuda_flash_attn_ext_tile_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    float softcap;
    memcpy(&softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (Q->ne[1] <= 16) {
        constexpr int cols_per_block = 16;
        constexpr int parallel_blocks = 4;
        if (softcap == 0.0f) {
            launch_fattn_tile_f32_64_128<cols_per_block, parallel_blocks, false>(ctx, dst);
        } else {
            launch_fattn_tile_f32_64_128<cols_per_block, parallel_blocks, true>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 32) {
        constexpr int cols_per_block = 32;
        constexpr int parallel_blocks = 4;
        if (softcap == 0.0f) {
            launch_fattn_tile_f32_64_128<cols_per_block, parallel_blocks, false>(ctx, dst);
        } else {
            launch_fattn_tile_f32_64_128<cols_per_block, parallel_blocks, true>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 32;
    constexpr int parallel_blocks = 1;
    if (softcap == 0.0f) {
        launch_fattn_tile_f32_64_128<cols_per_block, parallel_blocks, false>(ctx, dst);
    } else {
        launch_fattn_tile_f32_64_128<cols_per_block, parallel_blocks, true>(ctx, dst);
    }
}

bool ggml_cuda_fattn_tile_f32_is_supported([[maybe_unused]] ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    auto K = dst->src[1];
    auto V = dst->src[2];
    if (K->ne[0] != V->ne[0]) return false;
    return K->ne[0] == 64 || K->ne[0] == 128;
}
