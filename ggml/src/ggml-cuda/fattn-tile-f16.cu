//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tile-f16.cuh"

#define FATTN_KQ_STRIDE_TILE_F16 64

// FABsum P.V flush granularity, in KEYS. MUST evenly divide FATTN_KQ_STRIDE_TILE_F16 (=64) and be even
// (the P.V loop advances k0 by 2), i.e. one of {64, 32, 16}. 64 == one flush per tile at the natural
// per-tile online-softmax seam (near-free). 32 == two flushes per tile (higher accuracy, one extra
// fold+zero per tile). Error bound of the two-level accumulator is ~(FABSUM_PV_BLOCK+1)*u_fp16 and is
// INDEPENDENT of context length (the fp16 partial never accumulates more than FABSUM_PV_BLOCK keys).
// Composition operating point: PV=16 pairs with QK_FLUSH_BLOCK=8 (below) for the best measured
// accuracy-per-speed FABsum point on mellum2-12B: same-top 98.81% at 744 tok/s prefill (vs the
// PV=64/QK=D/2 base 98.19% @ 752, and vs the exact-QK reference 98.83% @ 695). Override to 64 for
// the max-speed base point, or 32 for an intermediate.
#ifndef FABSUM_PV_BLOCK
#define FABSUM_PV_BLOCK 16
#endif
// The flush-vs-rescale seam is only safe when FABSUM_PV_BLOCK evenly divides the tile stride and is even
// (the P.V loop advances k0 by 2); an odd or non-dividing override reintroduces the O(n) fp16 accumulation
// across a tile boundary -> silent degradation. Enforce the contract at compile time.
static_assert(FABSUM_PV_BLOCK % 2 == 0 && FATTN_KQ_STRIDE_TILE_F16 % FABSUM_PV_BLOCK == 0,
              "FABSUM_PV_BLOCK must be even and evenly divide FATTN_KQ_STRIDE_TILE_F16 (64): one of {64,32,16}");

// FABsum-for-QK flush granularity, in K-STEPS of the K.Q dot product's inner loop (one k-step = one
// half2 = 2 head-dim elements; the chain is D/2 = 64 steps at D=128, 32 at D=64). The half2 lane
// accumulator sum2 runs at full HFMA2 rate and, every QK_FLUSH_BLOCK k-steps, is folded into the
// persistent fp32 score accumulator KQ_acc (exact widening promotion + fp32 add) and zeroed -- the
// P.V FABSUM_PV_BLOCK mechanism applied to the kernel's SECOND fp16 accumulation chain. The fp16 QK
// chain then never exceeds QK_FLUSH_BLOCK fused-FMA roundings, independent of D; the fp32 seam adds
// only (D/2)/QK_FLUSH_BLOCK exact promotions per score. An explicit value MUST evenly divide D/2 for
// EVERY instantiated head size (D=64 and D=128 are both compiled, so it must divide 32: one of
// {8, 16, 32}; enforced by a static_assert inside the kernel). The default is 8, the composition
// operating point (pairs with FABSUM_PV_BLOCK=16). Setting it to (D/2) collapses the K.Q flush to a
// single fold at the end of the chain, recovering the flush-less FABsum base. NOTE: that base is NOT
// bit-identical to the upstream stock tile_f16 kernel -- the FABsum rewrite already keeps the softmax
// max/sum and the P.V/K.Q reductions in float (float adds vs the stock fp16 half-adds), so "flush off"
// restores the FABsum base, not the current upstream kernel.
#ifndef QK_FLUSH_BLOCK
#define QK_FLUSH_BLOCK 8
#endif

template<int D, int ncols, int nwarps, int parallel_blocks, bool use_softcap> // D == head size
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(nwarps*WARP_SIZE, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_tile_ext_f16(
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
#ifdef FP16_AVAILABLE
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

    const float slopef = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");

    // QK_FLUSH_BLOCK contract (see the #define above): it must evenly divide the K.Q chain length D/2,
    // or the final flush misses the end of the chain and the k-steps after the last flush are silently
    // DROPPED from the score (the reduction below reads only KQ_acc when the flush is active). Checked
    // here, per instantiation, because D is a template parameter: an explicit -DQK_FLUSH_BLOCK must
    // divide both compiled chain lengths (32 at D=64, 64 at D=128), i.e. be one of {8, 16, 32}.
    static_assert((QK_FLUSH_BLOCK) >= 1 && (D/2) % (QK_FLUSH_BLOCK) == 0,
                  "QK_FLUSH_BLOCK must be >= 1 and evenly divide D/2 for every compiled head size "
                  "(D=64 and D=128): one of {8,16,32}, or leave unset for the flush-once default");

    __shared__ half KQ[ncols*FATTN_KQ_STRIDE_TILE_F16];
    half2 * KQ2 = (half2 *) KQ;

    __shared__ half2 KV_tmp[FATTN_KQ_STRIDE_TILE_F16][D/2 + 1]; // Pad D to avoid memory bank conflicts.

    // KQ scores and the online-softmax statistics (max/sum) are kept in float (upstream precision split).
    // Level-3 additionally promotes the VKQ (P.V) accumulator to float2 (see below).
    float kqmax[ncols/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        kqmax[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float kqsum[ncols/nwarps] = {0.0f};

    // FABsum two-level P.V accumulator (P100). VKQ is the PERSISTENT cross-block accumulator and stays
    // float2 -- it is what kills the O(n_keys) growth and it is the ONLY accumulator the online-softmax
    // KQ_max_scale rescale, the sinks term, and the epilogue ever touch. VKQ_h2 is a per-tile HALF2 PARTIAL
    // that accumulates the P.V products at full HFMA2 rate (identical to the stock kernel's inner loop) and
    // is FLUSHED into VKQ and zeroed every FABSUM_PV_BLOCK keys. Within a block |acc|/|term| <= block size
    // << 2^11, so the fp16 partial never swamps; the float2 sum across blocks removes the context-length
    // growth. VKQ_h2 is guaranteed zero at every rescale seam (the last flush of each tile lands at the
    // tile boundary), so the rescale never sees un-flushed, mis-scaled fp16 mass.
    float2 VKQ   [ncols/nwarps][(D/2)/WARP_SIZE] = {{{0.0f, 0.0f}}};
    half2  VKQ_h2[ncols/nwarps][(D/2)/WARP_SIZE] = {{{0.0f, 0.0f}}};

    // Convert Q to half2 and store in registers:
    __shared__ half2 Q_h2[ncols][D/2];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;

            const float2 tmp = ic0 + j < ne01 ? Q_f2[j*(nb01/sizeof(float2)) + i] : make_float2(0.0f, 0.0f);
            // Full-speed QK revert: Q is stored at full scale (stock path). The K.Q dot product accumulates in
            // half2 (below) at full HFMA2 rate; stock ships this scale with no overflow, so the C' 0.25x / x4
            // guard is dropped. (If any model NaNs, restore the guard: use scale*0.25f here and multiply the
            // reduced score by 4.0f at the reduction below -- same speed, 4x fp16 headroom, one extra float mul.)
            Q_h2[j][i] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
        }
    }

    __syncthreads();

    const int k_start = parallel_blocks == 1 ? 0 : ip*FATTN_KQ_STRIDE_TILE_F16;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*FATTN_KQ_STRIDE_TILE_F16) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        float kqmax_new[ncols/nwarps];
#pragma unroll
        for (int j = 0; j < ncols/nwarps; ++j) {
            kqmax_new[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += WARP_SIZE) {
                const int k_KQ = k_KQ_0 + threadIdx.x;

                KV_tmp[i_KQ][k_KQ] = K_h2[(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ];
            }
        }

        __syncthreads();

        // Full-speed QK: accumulate the K.Q dot product in a HALF2 register at full HFMA2 rate (stock path).
        // sum2 spans only D/2 (<=64) terms within THIS tile and never carries across tiles, so the fp16
        // accumulation is bounded and context-length-independent -- the proven O(n) error was the P.V
        // accumulator, which FABsum still fixes with its two-level half2->float2 flush. KQ_acc keeps the
        // FLOAT reduced score (written in the reduction block below) so the online-softmax stats stay float.
        //
        // FABsum-for-QK (QK_FLUSH_BLOCK < D/2): KQ_acc doubles as the PERSISTENT fp32 half of a two-level
        // QK accumulator. Zero-initialized here, it absorbs sum2 every QK_FLUSH_BLOCK k-steps (flush block
        // in the k-loop below), so the fp16 fused-FMA chain never exceeds QK_FLUSH_BLOCK roundings. No new
        // variable exists: the same registers later hold the post-softcap/mask score, exactly as before.
        // At the flush-once default (QK_FLUSH_BLOCK == D/2) the flush compiles out, the reduction below
        // selects the stock lane-reduction expression, and this initializer is a dead store.
        half2 sum2[FATTN_KQ_STRIDE_TILE_F16/WARP_SIZE][ncols/nwarps] = {{{0.0f, 0.0f}}};
        float KQ_acc[FATTN_KQ_STRIDE_TILE_F16/WARP_SIZE][ncols/nwarps] = {{0.0f}};

#pragma unroll
        for (int k_KQ = 0; k_KQ < D/2; ++k_KQ) {
            half2 K_k[FATTN_KQ_STRIDE_TILE_F16/WARP_SIZE];
            half2 Q_k[ncols/nwarps];

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
                const int i_KQ = i_KQ_0 + threadIdx.x;

                K_k[i_KQ_0/WARP_SIZE] = KV_tmp[i_KQ][k_KQ];
            }
#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                Q_k[j_KQ_0/nwarps] = Q_h2[j_KQ][k_KQ];
            }

#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                    // Full-speed QK: fused half2 multiply-add (stock path). Issues as HFMA2 at full fp16 rate;
                    // the two lanes hold the partial dot product and are summed in float at the reduction below.
                    sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += K_k[i_KQ_0/WARP_SIZE]*Q_k[j_KQ_0/nwarps];
                }
            }

            // FABsum-for-QK flush: every QK_FLUSH_BLOCK k-steps, fold the half2 dot-product partial into
            // the persistent fp32 KQ_acc (half->float promotion is exact; the adds are fp32) and zero it --
            // the P.V flush below, applied to the QK chain. k_KQ is fully unrolled and QK_FLUSH_BLOCK is a
            // compile-time constant, so the guard folds away and the flush body is emitted ONLY at block
            // boundaries (e.g. k_KQ = 15,31,47,63 for -DQK_FLUSH_BLOCK=16 at D=128). All flush state is
            // thread-private registers: no sync, no smem traffic, no cross-thread coordination. The
            // static_assert above guarantees the last flush lands exactly at k_KQ == D/2-1, so sum2 is zero
            // when the loop exits and the reduction below reads KQ_acc alone -- a tail cannot exist (the
            // QK mirror of the P.V tile-boundary invariant). Overflow headroom strictly improves: the fp16
            // partial now spans <= QK_FLUSH_BLOCK of the same full-scale terms stock already ships over the
            // whole D/2 chain. At the default QK_FLUSH_BLOCK == D/2 the guard is compile-time false and NO
            // flush code is emitted -- the k-loop body is the current kernel's, unchanged.
            if ((QK_FLUSH_BLOCK) < D/2 && (k_KQ + 1) % (QK_FLUSH_BLOCK) == 0) {
#pragma unroll
                for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
#pragma unroll
                    for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                        KQ_acc[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] += __low2float(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps])
                                                                 + __high2float(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                        sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] = make_half2(0.0f, 0.0f);
                    }
                }
            }
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE_TILE_F16; i_KQ_0 += WARP_SIZE) {
            const int i_KQ = i_KQ_0 + threadIdx.x;

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < ncols; j_KQ_0 += nwarps) {
                const int j_KQ = j_KQ_0 + threadIdx.y;

                // Reduce the two half2 lanes of the K.Q dot product in FLOAT (stock reduces in half; we keep it
                // float to feed the float online-softmax). Q is at full scale, so no x4 rescale is applied.
                // With FABsum-for-QK active (QK_FLUSH_BLOCK < D/2) the k-loop's final flush has already folded
                // the whole chain into KQ_acc and zeroed sum2, so the score is read from KQ_acc; at the
                // flush-once default this selects the stock lane-reduction expression, unchanged. The
                // condition is compile-time constant -- exactly one side is ever emitted.
                float sum = (QK_FLUSH_BLOCK) < D/2
                    ? KQ_acc[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]
                    : __low2float(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]) + __high2float(sum2[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps]);
                if (use_softcap) {
                    sum = softcap * tanhf(sum);
                }
                sum += mask ? slopef*__half2float(maskh[j_KQ*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;

                kqmax_new[j_KQ_0/nwarps] = fmaxf(kqmax_new[j_KQ_0/nwarps], sum + FATTN_KQ_MAX_OFFSET);

                // Keep the score in registers; the raw (pre-exp) value is no longer stored to shared KQ.
                KQ_acc[i_KQ_0/WARP_SIZE][j_KQ_0/nwarps] = sum;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            kqmax_new[j0/nwarps] = warp_reduce_max(kqmax_new[j0/nwarps]);
            const float KQ_max_scale = expf(kqmax[j0/nwarps] - kqmax_new[j0/nwarps]);
            kqmax[j0/nwarps] = kqmax_new[j0/nwarps];

            float kqsum_add = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < FATTN_KQ_STRIDE_TILE_F16; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float val = expf(KQ_acc[i0/WARP_SIZE][j0/nwarps] - kqmax[j0/nwarps]);
                kqsum_add += val;
                // Store the post-exp probability as half (not float) to keep shared usage at 2 blocks/SM on P100.
                // Ownership (i = i0 + threadIdx.x) matches the V-loop read of KQ below.
                KQ[j*FATTN_KQ_STRIDE_TILE_F16 + i] = __float2half(val);
            }
            kqsum[j0/nwarps] = kqsum[j0/nwarps]*KQ_max_scale + kqsum_add;

            // Rescale the running float2 VKQ accumulator by the (float) online-softmax correction.
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                VKQ[j0/nwarps][i0/WARP_SIZE].x *= KQ_max_scale;
                VKQ[j0/nwarps][i0/WARP_SIZE].y *= KQ_max_scale;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F16; k0 += nwarps) {
            const int k = k0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                KV_tmp[k][i] = V_h2[(k_VKQ_0 + k)*stride_KV2 + i];
            }
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < FATTN_KQ_STRIDE_TILE_F16; k0 += 2) {
            half2  V_k[(D/2)/WARP_SIZE][2];
            half2 KQ_k[ncols/nwarps];

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                V_k[i0/WARP_SIZE][0] = KV_tmp[k0 + 0][i];
                V_k[i0/WARP_SIZE][1] = KV_tmp[k0 + 1][i];
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                const int j = j0 + threadIdx.y;

                KQ_k[j0/nwarps] = KQ2[j*(FATTN_KQ_STRIDE_TILE_F16/2) + k0/2];
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
#pragma unroll
                for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                    // FABsum inner: accumulate the two P.V products in the HALF2 partial at full HFMA2 rate.
                    // This is bit-for-bit the stock kernel's inner loop (no per-key float conversion), so it
                    // issues as HFMA2, not FFMA. __low2half2/__high2half2 splat p(k0)/p(k0+1) across both
                    // head-dim lanes carried in each V_k half2.
                    VKQ_h2[j0/nwarps][i0/WARP_SIZE] += V_k[i0/WARP_SIZE][0]* __low2half2(KQ_k[j0/nwarps]);
                    VKQ_h2[j0/nwarps][i0/WARP_SIZE] += V_k[i0/WARP_SIZE][1]*__high2half2(KQ_k[j0/nwarps]);
                }
            }

            // FABsum flush: every FABSUM_PV_BLOCK keys, fold the half2 partial into the persistent float2
            // accumulator and zero it. k0 advances by 2, so (k0+2) keys of this tile have been processed
            // after this iteration. FABSUM_PV_BLOCK divides 64 and k0 is fully unrolled, so (k0+2) %
            // FABSUM_PV_BLOCK is a compile-time constant and the flush body is emitted ONLY at block
            // boundaries (e.g. once, at k0==62, for the K=64 default). All keys in a tile share a single
            // kqmax (the online rescale ran before this loop), so there is no mid-tile rescale and the
            // partial is at one consistent scale between flushes. Because 64 % FABSUM_PV_BLOCK == 0 the final
            // block boundary always coincides with the tile boundary, so VKQ_h2 is zero on exit -> the next
            // tile's KQ_max_scale rescale (and the sinks term and the epilogue) see only the flushed VKQ.
            if ((k0 + 2) % FABSUM_PV_BLOCK == 0) {
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
#pragma unroll
                    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
                        const float2 fh = __half22float2(VKQ_h2[j0/nwarps][i0/WARP_SIZE]);
                        VKQ[j0/nwarps][i0/WARP_SIZE].x += fh.x;
                        VKQ[j0/nwarps][i0/WARP_SIZE].y += fh.y;
                        VKQ_h2[j0/nwarps][i0/WARP_SIZE] = make_half2(0.0f, 0.0f);
                    }
                }
            }
        }

        __syncthreads();
    }

    // Apply attention sinks (e.g. gpt-oss): the sink is a per-head extra softmax logit whose value
    // contribution is zero, so it only joins the running max and rescales the denominator/value
    // accumulator. Only ip==0 adds the sink term so it is counted exactly once across the
    // parallel_blocks KV split. Ported from fattn-vec-f16.cuh. kqmax is warp-uniform here (already
    // warp_reduce_max'd in the KV loop), so no shared-mem reduction is needed.
    if (sinksf && ip == 0) {
        const float sink = sinksf[blockIdx.y];

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const float kqmax_new_j = fmaxf(kqmax[j0/nwarps], sink);
            const float KQ_max_scale = expf(kqmax[j0/nwarps] - kqmax_new_j);
            kqmax[j0/nwarps] = kqmax_new_j;

            kqsum[j0/nwarps] = kqsum[j0/nwarps]*KQ_max_scale;
            if (threadIdx.x == 0) {
                // kqsum holds per-lane partials reduced in the epilogue, so add the sink term on a single lane.
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
#else
   NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int cols_per_block, int parallel_blocks, bool use_softcap>
void launch_fattn_tile_f16_64_128(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (Q->ne[0]) {
        case  64: {
            constexpr int      D = 64;
            constexpr int nwarps = 8;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f16<D, cols_per_block, nwarps, parallel_blocks, use_softcap>;
            launch_fattn<D, D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block, true, true);
        } break;
        case 128: {
            constexpr int      D = 128;
            constexpr int nwarps = 8;
            fattn_kernel_t fattn_kernel = flash_attn_tile_ext_f16<D, cols_per_block, nwarps, parallel_blocks, use_softcap>;
            launch_fattn<D, D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block, true, true);
        } break;
        default: {
            GGML_ABORT("FlashAttention without tensor cores only supports head sizes 64 and 128.");
        } break;
    }
}

void ggml_cuda_flash_attn_ext_tile_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    const int32_t precision = KQV->op_params[3];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

    float softcap;
    memcpy(&softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (Q->ne[1] <= 16) {
        constexpr int cols_per_block = 16;
        constexpr int parallel_blocks = 4;
        if (softcap == 0.0f) {
            launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, false>(ctx, dst);
        } else {
            launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, true>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 32) {
        constexpr int cols_per_block = 32;
        constexpr int parallel_blocks = 4;
        if (softcap == 0.0f) {
            launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, false>(ctx, dst);
        } else {
            launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, true>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 32;
    constexpr int parallel_blocks = 1;
    if (softcap == 0.0f) {
        launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, false>(ctx, dst);
    } else {
        launch_fattn_tile_f16_64_128<cols_per_block, parallel_blocks, true>(ctx, dst);
    }
}

bool ggml_cuda_fattn_tile_f16_is_supported([[maybe_unused]] ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    auto K = dst->src[1];
    auto V = dst->src[2];
    if (K->ne[0] != V->ne[0]) return false;
    return K->ne[0] == 64 || K->ne[0] == 128;
}
