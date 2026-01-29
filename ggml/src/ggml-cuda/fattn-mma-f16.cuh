#include "common.cuh"
#include "cp-async.cuh"
#include "mma_new.cuh"
#include "fattn-common.cuh"

using namespace ggml_cuda_mma;

typedef tile<16,  8, half2> tile_A;
typedef tile< 8,  8, half2> tile_B;
typedef tile<16,  8, half2> tile_B_16;
typedef tile<16,  8, float> tile_C_KQ;
typedef tile<16, 16, float> tile_C_KQ_16;
typedef tile<16,  4, half2> tile_C_VKQ;
typedef tile<16,  8, half2> tile_C_VKQ_16;

typedef void (* fattn_kernel_mma_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int2 * __restrict__ bounds,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const float softcap,
        const float fa_offset,
        const uint32_t n_head_log2,
        const int ne00, const int ne01, const int ne02, const int ne03,
        const int ne10, const int ne11, const int ne12, const int ne13,
        const int ne31, const int nb31,
        const int nb01, const int nb02, const int nb03,
        const int nb11, const int nb12, const int nb13,
        const int nb21, const int nb22, const int nb23,
        const int ne0, const int ne1, const int ne2, const int ne3);

template<int D, int nwarps, int KQ_per_iter>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int stride_KV) {
    constexpr int D2_padded = D/2 + 4; // Size of D in half2, padded to avoid shared memory bank conflicts.

    // If cp.async is available, load up to the highest power of 2 in D asynchronously:
#ifdef CP_ASYNC_AVAILABLE
    static_assert(D >= 64 && D < 512, "bad D");
    constexpr int k0_sync_start = D/2 < 64 ? 32 : (D/2 < 128 ? 64 : 128);

    const unsigned int tile_KV_32 = __cvta_generic_to_shared(tile_KV);

    constexpr int preload = 64;
    constexpr int h2_per_chunk = 16/sizeof(half2);
    constexpr int chunks_per_row = k0_sync_start / h2_per_chunk;
    constexpr int stride_i = WARP_SIZE / chunks_per_row;
#pragma unroll
    for (int i0 = 0; i0 < KQ_per_iter; i0 += nwarps*stride_i) {
        const int i = i0 + threadIdx.y*stride_i + (chunks_per_row == WARP_SIZE ? 0 : threadIdx.x / chunks_per_row);
        const int k = (chunks_per_row == WARP_SIZE ? threadIdx.x : threadIdx.x % chunks_per_row)*h2_per_chunk;

        cp_async_cg_16<preload>(tile_KV_32 + (i*D2_padded + k)*sizeof(half2), KV + i*stride_KV + k);
    }
#else
    constexpr int k0_sync_start = 0;
#endif // CP_ASYNC_AVAILABLE
    static_assert(k0_sync_start % WARP_SIZE == 0, "bad k0_sync_start");

    // If D is not a power of 2, the rest is loaded synchronously.
    // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
    static_assert(KQ_per_iter % (4*nwarps) == 0, "out of bounds");
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start = stride_k == WARP_SIZE ? k0_sync_start : D/2 - (D/2) % (2*stride_k);
        const int k0_stop  =                                         D/2 - (D/2) % (1*stride_k);
        const int stride_i = WARP_SIZE / stride_k;

        if (k0_start == k0_stop || k0_stop <= k0_sync_start) {
            continue;
        }

#pragma unroll
        for (int i0 = 0; i0 < KQ_per_iter; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

#pragma unroll
            for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                tile_KV[i*D2_padded + k] = KV[i*stride_KV + k];
            }
        }
    }
}

template<int ncols1, int nwarps, int KQ_per_iter>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(
        const half2 * const __restrict__ mask_h2, half2 * const __restrict__ tile_mask, const int stride_mask) {
    static_assert(KQ_per_iter == 2*WARP_SIZE || KQ_per_iter == WARP_SIZE, "bad KQ_per_iter");
#ifdef CP_ASYNC_AVAILABLE
    constexpr int preload = KQ_per_iter * sizeof(half);
    constexpr int cols_per_warp = 8*WARP_SIZE/KQ_per_iter;
    constexpr int stride_j = nwarps * cols_per_warp;

    const unsigned int tile_mask_32 = __cvta_generic_to_shared(tile_mask);

#pragma unroll
    for (int j0 = 0; j0 < ncols1; j0 += stride_j) {
        const int j = j0 + threadIdx.y*cols_per_warp +
            (KQ_per_iter == 2*WARP_SIZE ? threadIdx.x / (WARP_SIZE/4) : threadIdx.x / (WARP_SIZE/8));

        if (j0 + stride_j > ncols1 && j >= ncols1) {
            break;
        }

        const int i = 4 * (KQ_per_iter == 2*WARP_SIZE ? threadIdx.x % (WARP_SIZE/4) : threadIdx.x % (WARP_SIZE/8));

        cp_async_cg_16<preload>(tile_mask_32 + j*(KQ_per_iter*sizeof(half) + 16) + i*sizeof(half2), mask_h2 + j*stride_mask + i);
    }
#else
    constexpr int cols_per_warp = 2*WARP_SIZE/KQ_per_iter;
    constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
    for (int j0 = 0; j0 < ncols1; j0 += stride_j) {
        const int j = j0 + threadIdx.y*cols_per_warp + (KQ_per_iter == 2*WARP_SIZE ? 0 : threadIdx.x / (WARP_SIZE/2));

        if (j0 + stride_j > ncols1 && j >= ncols1) {
            break;
        }

        const int i = KQ_per_iter == 2*WARP_SIZE ? threadIdx.x : threadIdx.x % (WARP_SIZE/2);

        tile_mask[j*(KQ_per_iter/2 + 4) + i] = mask_h2[j*stride_mask + i];
    }
#endif // CP_ASYNC_AVAILABLE
}

template<int D, int ncols1, int ncols2, int nwarps, int KQ_per_iter, int ntiles, bool use_logit_softcap, bool needs_fixup, bool is_fixup, bool last_iter>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half2  * const __restrict__ mask_h2,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const float fa_offset,
        const int ne01,
        const int ne02,
        const int stride_KV,
        const int stride_mask,
        const int jt,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half2        * const __restrict__ tile_mask,
        const tile_B * const __restrict__ Q_B,
        tile_C_VKQ   * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int kb0) {
#ifdef INT8_MMA_AVAILABLE
    constexpr int cols_per_warp   = ntiles * tile_B::I;
    constexpr int cols_per_thread = ntiles == 1 ? 2 : ntiles;
    constexpr int np              = nwarps * (cols_per_warp/ncols2) / ncols1; // Number of parallel CUDA warps per Q column.
    constexpr int D2_padded       = D/2 + 4; // Size of D in half2, padded to avoid shared memory bank conflicts.

    const int k_VKQ_0 = kb0 * KQ_per_iter;
    tile_C_KQ KQ_C[KQ_per_iter/(np*tile_C_KQ::I) * ntiles];

    // Use wide variants of tiles if ntiles >= 2.
    tile_B_16     * Q_B_16   = (tile_B_16     *) Q_B;
    tile_C_VKQ_16 * VKQ_C_16 = (tile_C_VKQ_16 *) VKQ_C;
    tile_C_KQ_16  * KQ_C_16  = (tile_C_KQ_16  *) KQ_C;

#ifdef CP_ASYNC_AVAILABLE
    cp_async_wait_all();
    __syncthreads();
    flash_attn_ext_f16_load_tile<D, nwarps, KQ_per_iter>(V_h2 + k_VKQ_0*stride_KV, tile_V, stride_KV);
#else
    if (ncols2 > 1 || mask_h2) {
        flash_attn_ext_f16_load_mask<ncols1, nwarps, KQ_per_iter>(mask_h2 + k_VKQ_0/2, tile_mask, stride_mask);
    }
    flash_attn_ext_f16_load_tile<D, nwarps, KQ_per_iter>(K_h2 + k_VKQ_0*stride_KV, tile_K, stride_KV);
    __syncthreads();
#endif // CP_ASYNC_AVAILABLE

    // Calculate tile of KQ:
#pragma unroll
    for (int i_KQ_00 = 0; i_KQ_00 < KQ_per_iter; i_KQ_00 += np*tile_A::I) {
        const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*tile_A::I;
#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += tile_A::J) {
            tile_A K_A;
            load_ldmatrix(K_A, tile_K + i_KQ_0*D2_padded + k_KQ_0, D2_padded);
            if (ntiles == 1) {
                mma(KQ_C[i_KQ_00/(np*tile_A::I)], K_A, Q_B[k_KQ_0/tile_A::J]);
            } else {
#pragma unroll
                for (int t = 0; t < ntiles/2; ++t) {
                    // Wide version of KQ_C is column-major => swap A and B.
                    mma(KQ_C_16[i_KQ_00/(np*tile_A::I) * ntiles/2 + t], Q_B_16[k_KQ_0/tile_A::J * ntiles/2 + t], K_A);
                }
            }
        }
    }

#ifndef CP_ASYNC_AVAILABLE
    __syncthreads(); // Only needed if tile_K == tile_V.
#endif // CP_ASYNC_AVAILABLE

    if (use_logit_softcap) {
        static_assert(KQ_per_iter % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < KQ_per_iter/(np*tile_C_KQ::I) * ntiles; ++i) {
#pragma unroll
            for (int l = 0; l < tile_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if (ntiles == 1) {
        if (ncols2 > 1 || mask_h2) {
#pragma unroll
            for (int i00 = 0; i00 < KQ_per_iter; i00 += np*tile_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*tile_C_KQ::I;
#pragma unroll
                for (int l = 0; l < tile_C_KQ::ne; ++l) {
                    const int i = i0 + tile_C_KQ::get_i(l);
                    const int j = ((threadIdx.y / np)*tile_C_KQ::J + tile_C_KQ::get_j(l)) / ncols2;

                    KQ_C[i00/(np*tile_C_KQ::I)].x[l] += slope *
                        __half2float(((const half *) tile_mask)[j*(KQ_per_iter + 8) + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(KQ_per_iter % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < KQ_per_iter/(np*tile_C_KQ::I); ++k) {
#pragma unroll
            for (int l = 0; l < tile_C_KQ::ne; ++l) {
                KQ_max_new[l % 2] = fmaxf(KQ_max_new[l % 2], KQ_C[k].x[l] + fa_offset);
            }
        }

        // Values per KQ column are spread across 8 threads, does not need full warp reduce:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }

        static_assert(KQ_per_iter % (np*tile_C_KQ::I) == 0, "bad loop size");

#pragma unroll
        for (int k = 0; k < KQ_per_iter/(np*tile_C_KQ::I); ++k) {
#pragma unroll
            for (int l = 0; l < tile_C_KQ::ne; ++l) {
                KQ_C[k].x[l] = expf(KQ_C[k].x[l] - KQ_max_new[l % 2]);

                KQ_rowsum_add[l % 2] += KQ_C[k].x[l];
            }
        }
    } else { // ntiles > 1
        if (ncols2 > 1 || mask_h2) {
#pragma unroll
            for (int i00 = 0; i00 < KQ_per_iter; i00 += np*tile_C_KQ_16::J) {
                const int i0 = i00 + (threadIdx.y % np)*tile_C_KQ_16::J;
#pragma unroll
                for (int t = 0; t < ntiles/2; ++t) {
#pragma unroll
                    for (int l0 = 0; l0 < tile_C_KQ_16::ne; l0 += 2) {
                        const int i = (i0 + tile_C_KQ_16::get_j(l0)) / 2;
                        const int j = ((threadIdx.y / np)*cols_per_warp + t*tile_C_KQ_16::I + tile_C_KQ_16::get_i(l0)) / ncols2;

                        const float2 tmp = __half22float2(tile_mask[j*(KQ_per_iter/2 + 4) + i]);
                        const int KQ_index = i00/(np*tile_C_KQ_16::J) * ntiles/2 + t;
                        KQ_C_16[KQ_index].x[l0 + 0] += slope*tmp.x;
                        KQ_C_16[KQ_index].x[l0 + 1] += slope*tmp.y;
                    }
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(KQ_per_iter % (np*tile_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < KQ_per_iter/(np*tile_C_KQ_16::J); ++k) {
#pragma unroll
            for (int t = 0; t < ntiles/2; ++t) {
#pragma unroll
                for (int l = 0; l < tile_C_KQ_16::ne; ++l) {
                    const int KQ_index = 2*t + (l/2) % 2;
                    KQ_max_new[KQ_index] = fmaxf(KQ_max_new[KQ_index], KQ_C_16[k*ntiles/2 + t].x[l] + fa_offset);
                }
            }
        }

        // Values per KQ column are spread across 4 threads, does not need full warp reduce:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 2; offset >= 1; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }

        static_assert(KQ_per_iter % (np*tile_C_KQ_16::J) == 0, "bad loop size");
#pragma unroll
        for (int k = 0; k < KQ_per_iter/(np*tile_C_KQ_16::J); ++k) {
#pragma unroll
            for (int t = 0; t < ntiles/2; ++t) {
#pragma unroll
                for (int l = 0; l < tile_C_KQ_16::ne; ++l) {
                    const int KQ_index = 2*t + (l/2) % 2;

                    KQ_C_16[k*ntiles/2 + t].x[l] = expf(KQ_C_16[k*ntiles/2 + t].x[l] - KQ_max_new[KQ_index]);

                    KQ_rowsum_add[KQ_index] += KQ_C_16[k*ntiles/2 + t].x[l];
                }
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            KQ_max_scale[col] = expf(KQ_max[col] - KQ_max_new[col]);
            KQ_max[col] = KQ_max_new[col];

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

        if (ntiles == 1) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < D/tile_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < tile_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < D/tile_C_VKQ_16::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < tile_C_VKQ_16::ne; l0 += 2) {
                        VKQ_C_16[i*ntiles/2 + col/2].x[l0 + col % 2] *= KQ_max_scale_h2;
                    }
                }
            }
        }
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    tile_B B[KQ_per_iter/(np*2*tile_B::J) * ntiles];
    tile_B_16 * B_16 = (tile_B_16 *) B;
    static_assert(KQ_per_iter % (np*2*tile_B::J) == 0, "bad loop size");
    if (ntiles == 1) {
#pragma unroll
        for (int k = 0; k < KQ_per_iter/(np*2*tile_B::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < KQ_per_iter/(np*2*tile_B_16::J); ++k) {
#pragma unroll
            for (int t = 0; t < ntiles/2; ++t) {
                B_16[k*ntiles/2 + t] = get_half2(KQ_C_16[k*ntiles/2 + t]);
            }
        }
    }

#ifdef CP_ASYNC_AVAILABLE
    // Preload K tile for next iteration:
    cp_async_wait_all();
    __syncthreads();
    if (!last_iter) {
        if (ncols2 > 1 || mask_h2) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, KQ_per_iter>(mask_h2 + (k_VKQ_0 + KQ_per_iter)/2, tile_mask, stride_mask);
        }
        flash_attn_ext_f16_load_tile<D, nwarps, KQ_per_iter>(K_h2 + (k_VKQ_0 + KQ_per_iter)*stride_KV, tile_K, stride_KV);
    }
#else
    flash_attn_ext_f16_load_tile<D, nwarps, KQ_per_iter>(V_h2 + k_VKQ_0*stride_KV, tile_V, stride_KV);
    __syncthreads();
#endif // CP_ASYNC_AVAILABLE

    // Calculate VKQ tile:
#pragma unroll
    for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += tile_C_VKQ::I) {
        static_assert((KQ_per_iter/2) % (np*tile_A::J) == 0, "bad loop size");
#pragma unroll
        for (int k00 = 0; k00 < KQ_per_iter/2; k00 += np*tile_A::J) {
            const int k0 = k00 + (threadIdx.y % np)*tile_A::J;

            tile_A A;
            load_ldmatrix_trans(A, tile_V + 2*k0*D2_padded + i_VKQ_0/2, D2_padded);
            if (ntiles == 1) {
                mma(VKQ_C[i_VKQ_0/tile_C_VKQ::I], A, B[k00/(np*tile_A::J)]);
            } else {
#pragma unroll
                for (int t = 0; t < ntiles/2; ++t) {
                    // Wide version of VKQ_C is column-major => swap A and B.
                    mma(VKQ_C_16[i_VKQ_0/tile_C_VKQ::I * ntiles/2 + t], B_16[k00/(np*tile_A::J) * ntiles/2 + t], A);
                }
            }
        }
    }

#ifndef CP_ASYNC_AVAILABLE
    __syncthreads(); // Only needed if tile_K == tile_V.
#endif // CP_ASYNC_AVAILABLE

#else
    GGML_UNUSED(Q_f2); GGML_UNUSED(K_h2); GGML_UNUSED(V_h2);
    GGML_UNUSED(mask_h2); GGML_UNUSED(dstk); GGML_UNUSED(dstk_fixup);
    GGML_UNUSED(scale); GGML_UNUSED(slope); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(stride_KV);
    GGML_UNUSED(stride_mask); GGML_UNUSED(jt); GGML_UNUSED(tile_K);
    GGML_UNUSED(stride_mask); GGML_UNUSED(jt); GGML_UNUSED(tile_K);
    GGML_UNUSED(tile_V); GGML_UNUSED(tile_mask); GGML_UNUSED(Q_B);
    GGML_UNUSED(VKQ_C); GGML_UNUSED(KQ_max); GGML_UNUSED(KQ_rowsum);
    GGML_UNUSED(kb0);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template<int D, int ncols1, int ncols2, int nwarps, int KQ_per_iter, int ntiles, bool use_logit_softcap, bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half2  * const __restrict__ mask_h2,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const float fa_offset,
        const int ne01,
        const int ne02,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_KV,
        const int stride_mask,
        const int jt,
        const int kb0_start,
        const int kb0_stop) {
#ifdef INT8_MMA_AVAILABLE
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int ncols           = ncols1 * ncols2;
    constexpr int cols_per_warp   = ntiles * tile_B::I;
    constexpr int cols_per_thread = ntiles == 1 ? 2 : ntiles;
    constexpr int np              = nwarps * (cols_per_warp/ncols2) / ncols1; // Number of parallel CUDA warps per Q column.

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    static_assert(D           % nwarps == 0, "bad D");
    static_assert(KQ_per_iter % nwarps == 0, "bad KQ_per_iter");

    constexpr int D2_padded = D/2 + 4; // Size of D in half2, padded to avoid shared memory bank conflicts.

    // Temporary shared buffer for loading K/V data with KQ_per_iter*D logical elements:
    extern __shared__ half2 tile_K[];
#ifdef CP_ASYNC_AVAILABLE
    half2 * tile_V    = tile_K + KQ_per_iter*D2_padded;
#else
    half2 * tile_V    = tile_K;
#endif // CP_ASYNC_AVAILABLE
    half2 * tile_mask = tile_V + KQ_per_iter*D2_padded;

    tile_B       Q_B[D/(2*tile_B::J) * ntiles];
    tile_C_VKQ VKQ_C[D/tile_C_VKQ::I * ntiles];

    tile_B_16     * Q_B_16   = (tile_B_16     *) Q_B;
    tile_C_VKQ_16 * VKQ_C_16 = (tile_C_VKQ_16 *) VKQ_C;

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Temporarily load Q data into tile_K, will be loaded into registers afterwards.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
        const int k0_start  = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
        const int k0_stop   =                             D/2 - (D/2) % (1*stride_k);
        const int stride_jc = WARP_SIZE / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if (jt*ncols1 + j < ne01) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_K[jc*D2_padded + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_K[jc*D2_padded + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < D/2; k0 += tile_B::J) {
            if (ntiles == 1) {
                load_ldmatrix(Q_B[k0/tile_B::J], tile_K + j0*D2_padded + k0, D2_padded);
            } else {
#pragma unroll
                for (int t = 0; t < ntiles/2; ++t) {
                    load_ldmatrix(Q_B_16[k0/tile_B_16::J * ntiles/2 + t],
                        tile_K + (j0 + t*tile_B_16::I)*D2_padded + k0, D2_padded);
                }
            }
        }
    }

    __syncthreads();

    // Preload mask and K data for first iteration when using cp_async:
#ifdef CP_ASYNC_AVAILABLE
    if (ncols2 > 1 || mask_h2) {
        flash_attn_ext_f16_load_mask<ncols1, nwarps, KQ_per_iter>(mask_h2 + kb0_start*KQ_per_iter/2, tile_mask, stride_mask);
    }
    flash_attn_ext_f16_load_tile<D, nwarps, KQ_per_iter>(K_h2 + kb0_start*KQ_per_iter*stride_KV, tile_K, stride_KV);
#endif // CP_ASYNC_AVAILABLE

    // Iterate over ne11 == previous tokens:
    for (int kb0 = kb0_start; kb0 < kb0_stop-1; ++kb0) {
        constexpr bool last_iter = false;
        flash_attn_ext_f16_iter<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup, last_iter>
            (Q_f2, K_h2, V_h2, mask_h2, dstk, dstk_fixup, scale, slope, logit_softcap, fa_offset,
             ne01, ne02, stride_KV, stride_mask, jt, tile_K, tile_V, tile_mask, Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
    }
    { // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
        constexpr bool last_iter = true;
        flash_attn_ext_f16_iter<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup, last_iter>
            (Q_f2, K_h2, V_h2, mask_h2, dstk, dstk_fixup, scale, slope, logit_softcap, fa_offset,
             ne01, ne02, stride_KV, stride_mask, jt, tile_K, tile_V, tile_mask, Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0_stop-1);
    }

    // With cp_async there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
#ifdef CP_ASYNC_AVAILABLE
    if (nwarps*cols_per_warp > KQ_per_iter) {
        __syncthreads();
    }
#endif // CP_ASYNC_AVAILABLE

    // Finally, sum up partial KQ rowsums.
    // The partial sums are spread across 8/4 threads each, does not need full reduce.
    {
        constexpr int offset_first = ntiles == 1 ? 16 : 2;
        constexpr int offset_last  = ntiles == 1 ?  4 : 1;
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, WARP_SIZE);
            }
        }
    }

    // If attention sinks are used, potentially re-scale if KQ_max is small.
    // Also add the sink as a value to KQ_rowsum, this is done after synchonization of KQ_rowsum
    //     so it's being done unconditionally for every thread.
    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            static_assert(ntiles == 1 || ntiles == 2, "ntiles > 2 not implemented");
            const int jc = ntiles == 1 ? 2*tile_C_VKQ::get_j(col/2) + col % 2 : tile_C_VKQ_16::get_i(col);
            const float sink = sinks_f[jc % ncols2];

            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_max_add;
        }

        if (ntiles == 1) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < D/tile_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < tile_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < D/tile_C_VKQ_16::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < tile_C_VKQ_16::ne; l0 += 2) {
                        VKQ_C_16[i*ntiles/2 + col/2].x[l0 + col % 2] *= KQ_max_scale_h2;
                    }
                }
            }
        }
    }

    // Write VKQ accumulators to shared memory in column-major format.
    // It's faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // Also for np > 1 the combination is done via these values in shared memory.
    if (ntiles == 1) {
        const int jc_cwd = threadIdx.y*tile_B::I + tile_B::get_i(-1); // jc combine write data
#pragma unroll
        for (int k0 = 0; k0 < D/2; k0 += tile_B::J) {
            const tile_B B = get_transposed(VKQ_C[k0/tile_B::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
            for (int l = 0; l < tile_B::ne; ++l) {
                const int k = k0 + tile_B::get_j(l);

                tile_K[jc_cwd*D2_padded + k] = B.x[l];
            }
        }
    } else {
#pragma unroll
        for (int t = 0; t < ntiles/2; ++t) {
            const int j0 = threadIdx.y*cols_per_warp + t*tile_C_VKQ_16::I;
#pragma unroll
            for (int k0 = 0; k0 < D/2; k0 += tile_C_VKQ_16::J) {
#pragma unroll
                for (int l = 0; l < tile_C_VKQ_16::ne; ++l) {
                    const int j = j0 + tile_C_VKQ_16::get_i(l);
                    const int k = k0 + tile_C_VKQ_16::get_j(l);

                    tile_K[j*D2_padded + k] = VKQ_C_16[k0/tile_C_VKQ_16::J * ntiles/2 + t].x[l];
                }
            }
        }
    }

    if constexpr (ntiles == 1) {
        const int jc_cwmo = (threadIdx.x % (2*tile_C_VKQ::J)) / tile_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = threadIdx.y*(2*tile_C_VKQ::J) + 2*tile_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*tile_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_K)[jc_cwm*(D2_padded/2) + D/4] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && threadIdx.x < tile_B::I) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < tile_B::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    } else {
        static_assert(ntiles == 2 || ntiles == 4, "bad ntiles");
        const int jc_cwm = threadIdx.y*cols_per_warp // jc combine write meta
            + (ntiles == 4 ? ((threadIdx.x % 4) / 2) * tile_C_VKQ_16::I : 0)
            + tile_C_VKQ_16::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]); // KQ combine max rowsum

        if (((!needs_fixup && !is_fixup) || np > 1) && (ntiles == 4 || threadIdx.x % 4 < cols_per_thread)) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_K)[jc_cwm*(D2_padded/2) + D/4] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && (ntiles == 4 || threadIdx.x % 4 < ntiles)) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && (ntiles == 4 || threadIdx.x % 4 < ntiles)) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    static_assert(np == 1 || ntiles == 1 || ntiles == 2, "bad ntiles");
    if (np > 1 && threadIdx.y % np == 0) {
        // Combine the meta data for parallel warps via shared memory.
        // Warps with threadIdx.y % np != 0 must NOT return early.
        // All threads must return simultaneously to avoid race conditions with work on the next tile.

        constexpr int nmeta = np*cols_per_warp >= WARP_SIZE ? np*cols_per_warp/WARP_SIZE : 1;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < WARP_SIZE ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_K) + jc_meta*(D2_padded/2) + D/4;
        float2 meta[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            meta[imeta] = meta_ptr[imeta * WARP_SIZE * D2_padded/2];
        }

        float KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset >= WARP_SIZE) {
                continue;
            }
            KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, WARP_SIZE));
        }

        float KQ_cms[nmeta]; // KQ combine max scale per warp.
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
        }

        float KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_crs += KQ_cms[imeta]*meta[imeta].y;
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset >= WARP_SIZE) {
                continue;
            }
            KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, WARP_SIZE);
        }

        // Write back combined meta data:
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            if (np*cols_per_warp >= WARP_SIZE || threadIdx.x < np*cols_per_warp) {
                // Combined KQ max scale + rowsum.
                meta_ptr[imeta * WARP_SIZE * D2_padded/2] = make_float2(KQ_cms[imeta], KQ_crs);
            }
        }

        // Combined KQ max + rowsum.
        static_assert(cols_per_warp <= WARP_SIZE);
        if (needs_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        if (is_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
    }

    if (np > 1) {
        __syncthreads();
    }

    if (np == 1 || threadIdx.y % np == 0) {
        // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
        // The values after that are for the partial results of the individual blocks.
        float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(D/2));

#pragma unroll
        for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
            const int k0_start  = stride_k == WARP_SIZE ? 0 : D/2 - (D/2) % (2*stride_k);
            const int k0_stop   =                             D/2 - (D/2) % (1*stride_k);
            const int stride_jc = WARP_SIZE / stride_k;

            if (k0_start == k0_stop) {
                continue;
            }

#pragma unroll
            for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                    break;
                }

                const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                const int j_dst = jc_dst / ncols2;
                const int c_dst = jc_dst % ncols2;

                if (!is_fixup && jt*ncols1 + j_dst >= ne01) {
                    continue;
                }

                const float * meta_j = (const float *) tile_K + jc_tile_K*D2_padded + D/2;
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                    for (int ip = 0; ip < np; ++ip) {
                        const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * D2_padded + 0];
                        const float2 dstk_val_add = __half22float2(tile_K[(jc_tile_K + ip*cols_per_warp) * D2_padded + k]);
                        dstk_val.x += dstk_val_add.x*KQ_crs;
                        dstk_val.y += dstk_val_add.y*KQ_crs;
                    }

                    if (!needs_fixup && !is_fixup) {
                        const float KQ_rowsum_j = meta_j[1];
                        dstk_val.x /= KQ_rowsum_j;
                        dstk_val.y /= KQ_rowsum_j;
                    }

                    if (is_fixup) {
                        dstk_fixup_data[jc_dst*(D/2) + k] = dstk_val;
                    } else {
                        dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(D/2) + k] = dstk_val;
                    }
                }
            }
        }
    }

    if (np > 1) {
        __syncthreads();
    }
#else
    GGML_UNUSED(Q_f2); GGML_UNUSED(K_h2); GGML_UNUSED(V_h2);
    GGML_UNUSED(mask_h2); GGML_UNUSED(dstk); GGML_UNUSED(dstk_fixup);
    GGML_UNUSED(scale); GGML_UNUSED(slope); GGML_UNUSED(logit_softcap);
    GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(stride_Q1);
    GGML_UNUSED(stride_Q2); GGML_UNUSED(stride_KV); GGML_UNUSED(stride_mask);
    GGML_UNUSED(jt); GGML_UNUSED(kb0_start); GGML_UNUSED(kb0_stop);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}

template<int D, int ncols1, int ncols2, int nwarps, int KQ_per_iter, int ntiles, bool use_logit_softcap>
__launch_bounds__(nwarps*WARP_SIZE, 2)
static __global__ void flash_attn_mma_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int2 * __restrict__ bounds,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const float logit_softcap,
        const float fa_offset,
        const uint32_t n_head_log2,
        const int ne00, const int ne01, const int ne02, const int ne03,
        const int ne10, const int ne11, const int ne12, const int ne13,
        const int ne31, const int nb31,
        const int nb01, const int nb02, const int nb03,
        const int nb11, const int nb12, const int nb13,
        const int nb21, const int nb22, const int nb23,
        const int ne0, const int ne1, const int ne2, const int ne3) {
#if defined(INT8_MMA_AVAILABLE)

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(FATTN_KQ_STRIDE % KQ_per_iter == 0, "bad KQ_per_iter");

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_KV   = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half2);

    const int iter_k = ne11 / FATTN_KQ_STRIDE;
    const int iter_j = (ne01 + (ncols1 - 1)) / ncols1;

    constexpr int kb_niter = FATTN_KQ_STRIDE / KQ_per_iter; // Number of kernel iterations per assigned KQ slice.

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = (blockIdx.x + 0)*iter_k*iter_j*(ne02/ncols2) / gridDim.x;
    const int kbc_stop = (blockIdx.x + 1)*iter_k*iter_j*(ne02/ncols2) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);
    while (kbc < kbc_stop && kb0_stop == iter_k) {
        const int channel = kbc / (iter_k*iter_j);
        const int jt      = (kbc - channel*iter_k*iter_j) / iter_k; // j index of current tile.

        const float2 * Q_f2    = (const float2 *) (Q + nb02* channel*ncols2);
        const half2  * K_h2    = (const half2  *) (K + nb12*(channel*ncols2 / gqa_ratio));
        const half2  * V_h2    = (const half2  *) (V + nb12*(channel*ncols2 / gqa_ratio)); // K and V have same shape
        const half2  * mask_h2 = ncols2 > 1 || mask ? (const half2  *) mask + (nb31/sizeof(half2))*jt*ncols1 : nullptr;
        float2       * dstk    = ((float2 *) dst) + channel*(ncols2 * D/2);
        const float  * sinks_f = sinks ? (const float *) sinks + channel * ncols2 : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, channel, n_head_log2, m0, m1) : 1.0f;

        int kb0_start_kernel = kb0_start * kb_niter;
        int kb0_stop_kernel  = kb0_stop  * kb_niter;

        if (bounds) {
            kb0_start_kernel = max(kb0_start_kernel, bounds[jt].x / KQ_per_iter);
            kb0_stop_kernel  = min(kb0_stop_kernel,  bounds[jt].y / KQ_per_iter);
        }

        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h2, sinks_f, dstk, dst_meta, scale, slope, logit_softcap, fa_offset,
                 ne01, ne02, stride_Q1, stride_Q2, stride_KV, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is working on the beginning of a tile.
            flash_attn_ext_f16_process_tile<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup>
                (Q_f2, K_h2, V_h2, mask_h2, sinks_f, dstk, dst_meta, scale, slope, logit_softcap, fa_offset,
                 ne01, ne02, stride_Q1, stride_Q2, stride_KV, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int channel = kbc / (iter_k*iter_j);
    const int jt      = (kbc - channel*iter_k*iter_j) / iter_k; // j index of current tile.

    const float2 * Q_f2    = (const float2 *) (Q + nb02* channel*ncols2);
    const half2  * K_h2    = (const half2  *) (K + nb12*(channel*ncols2 / gqa_ratio));
    const half2  * V_h2    = (const half2  *) (V + nb12*(channel*ncols2 / gqa_ratio)); // K and V have same shape
    const half2  * mask_h2 = ncols2 > 1 || mask ? (const half2  *) mask + (nb31/sizeof(half2))*jt*ncols1 : nullptr;
    float2       * dstk    = ((float2 *) dst) + channel*(ncols2 * D/2);
    const float  * sinks_f = sinks ? (const float *) sinks + channel*ncols2 : nullptr;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, channel, n_head_log2, m0, m1) : 1.0f;

    int kb0_start_kernel = kb0_start * kb_niter;
    int kb0_stop_kernel  = kb0_stop  * kb_niter;
    if (bounds) {
        kb0_start_kernel = max(kb0_start_kernel, bounds[jt].x / KQ_per_iter);
        kb0_stop_kernel  = min(kb0_stop_kernel,  bounds[jt].y / KQ_per_iter);
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap, needs_fixup, is_fixup>
        (Q_f2, K_h2, V_h2, mask_h2, sinks_f, dstk, dst_meta, scale, slope, logit_softcap, fa_offset,
         ne01, ne02, stride_Q1, stride_Q2, stride_KV, stride_mask, jt, kb0_start_kernel, kb0_stop_kernel);
#else
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(mask); GGML_UNUSED(sinks);
    GGML_UNUSED(dst); GGML_UNUSED(dst_meta); GGML_UNUSED(scale);
    GGML_UNUSED(max_bias); GGML_UNUSED(m0); GGML_UNUSED(m1);
    GGML_UNUSED(n_head_log2); GGML_UNUSED(logit_softcap); GGML_UNUSED(ne00);
    GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03); GGML_UNUSED(ne10);
    GGML_UNUSED(ne11); GGML_UNUSED(ne12); GGML_UNUSED(ne13); GGML_UNUSED(ne31);
    GGML_UNUSED(nb31); GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03);
    GGML_UNUSED(nb11); GGML_UNUSED(nb12); GGML_UNUSED(nb13); GGML_UNUSED(nb21);
    GGML_UNUSED(nb22); GGML_UNUSED(nb23); GGML_UNUSED(ne0); GGML_UNUSED(ne1);
    GGML_UNUSED(ne2); GGML_UNUSED(ne3);
    NO_DEVICE_CODE;
#endif // defined(INT8_MMA_AVAILABLE)
}

template<int D, int ncols1, int ncols2, int KQ_stride> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_mma_stream_k_fixup(
        float * __restrict__ dst, const float2 * __restrict__ dst_fixup, const int ne01, const int ne02, const int ne11) {
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int iter_k = ne11 / FATTN_KQ_STRIDE;
    const int iter_j = (ne01 + (ncols1 - 1)) / ncols1;

    const int kbc0      = (bidx0 + 0)*iter_k*iter_j*(ne02/ncols2) / gridDim.x;
    const int kbc0_stop = (bidx0 + 1)*iter_k*iter_j*(ne02/ncols2) / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % iter_k == 0;
    const bool did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    const int channel = kbc0 / (iter_k*iter_j);
    const int jt      = (kbc0 - channel*iter_k*iter_j) / iter_k;

    if (jt*ncols1 + j >= ne01) {
        return;
    }

    dst += jt*ne02*(ncols1*D) + channel*(ncols2*D) + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }


    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = bidx*iter_k*iter_j*(ne02/ncols2) / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
#if !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_mma_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst,
        const int parallel_blocks) {
    VKQ_parts += parallel_blocks*D * gridDim.z*blockIdx.x;
    VKQ_meta  += parallel_blocks   * gridDim.z*blockIdx.x;
    dst       +=                 D * gridDim.z*blockIdx.x;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    if (tid < 2*parallel_blocks) {
        ((float *) meta)[threadIdx.x] = ((const float *)VKQ_meta) [blockIdx.z*(2*parallel_blocks) + tid];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float diff = meta[l].x - kqmax;
        float KQ_max_scale = expf(diff);
        const uint32_t ftz_mask = 0xFFFFFFFF * (diff > SOFTMAX_FTZ_THRESHOLD);
        *((uint32_t *) &KQ_max_scale) &= ftz_mask;

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*gridDim.z*D + blockIdx.z*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[blockIdx.z*D + tid] = VKQ_numerator / VKQ_denominator;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ int warp_reduce_all(int x) {
    if constexpr (width == WARP_SIZE) { //ggml_cuda_get_physical_warp_size()) {
        return __all_sync(0xffffffff, x);
    } else {
#pragma unroll
        for (int offset = width/2; offset > 0; offset >>= 1) {
            x = __shfl_xor_sync(0xffffffff, x, offset, width) && x;
        }
        return x;
    }
}

template <int ncols1, bool is_swa>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_min_max(
        const half2 * __restrict__ mask, int2 * __restrict__ KV_min_max, const int ne30, const int s31, const int s33) {
    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    if constexpr (!is_swa) {
        if (threadIdx.x == 0) {
            KV_min_max[sequence*ne31 + jt] = {0, KV_max_sj + FATTN_KQ_STRIDE};
        }
        return;
    }

    if (threadIdx.x == 0) {
        KV_min_max[sequence*ne31 + jt].y = KV_max_sj + FATTN_KQ_STRIDE;
    }

    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    __syncthreads();

    int KV_min_sj = 0;
    for (; KV_min_sj < KV_max_sj; KV_min_sj += FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_min_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    if (threadIdx.x == 0) {
        KV_min_max[sequence*ne31 + jt].x = KV_min_sj;
    }
}


template <int D, int ncols1, int ncols2, int KQ_stride>
void launch_fattn_mma(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_mma_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int KQ_row_granularity, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(!mask || mask->ne[1] >= GGML_PAD(Q->ne[1], 16) &&
                                "the Flash-Attention CUDA kernel requires the mask to be padded to 16 and at least n_queries big");

    GGML_ASSERT(K->ne[1] % FATTN_KQ_STRIDE == 0 && "Incorrect KV cache padding.");

    GGML_ASSERT(Q->ne[3] == 1);

    int n_swa;
    memcpy(&n_swa, (const int *) KQV->op_params + 4, sizeof(int));

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<int2>   KV_min_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        K_f16.alloc(ggml_nelements(K));
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
        to_fp16(K_data, K_f16.ptr, 1, ggml_nelements(K), main_stream);
        K_data = (char *) K_f16.ptr;

        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        nb11 = nb11*bs*sizeof(half)/ts;
        nb12 = nb12*bs*sizeof(half)/ts;
        nb13 = nb13*bs*sizeof(half)/ts;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        V_f16.alloc(ggml_nelements(V));
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
        to_fp16(V_data, V_f16.ptr, 1, ggml_nelements(V), main_stream);
        V_data = (char *) V_f16.ptr;

        const size_t bs = ggml_blck_size(V->type);
        const size_t ts = ggml_type_size(V->type);

        nb21 = nb21*bs*sizeof(half)/ts;
        nb22 = nb22*bs*sizeof(half)/ts;
        nb23 = nb23*bs*sizeof(half)/ts;
    }

    int parallel_blocks = 1;

    const int ntiles_x = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int ntiles_total = ntiles_x * (Q->ne[2] / ncols2) * Q->ne[3];

    if (mask && (Q->ne[1] >= 1024 || (n_swa > 0 && K->ne[1] >= FATTN_KQ_STRIDE + n_swa))) {
        const int s31 = mask->nb[1] / sizeof(half2);
        const int s33 = mask->nb[3] / sizeof(half2);
        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);
        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;
        KV_min_max.alloc(ne_KV_max);
        if (n_swa > 0) {
            flash_attn_mask_to_KV_min_max<ncols1, true><<<blocks_num_KV_max, block_dim_KV_max, 0, main_stream>>>
                ((const half2 *) mask->data, KV_min_max.ptr, iter_k, s31, s33);
        } else {
            flash_attn_mask_to_KV_min_max<ncols1, false><<<blocks_num_KV_max, block_dim_KV_max, 0, main_stream>>>
                ((const half2 *) mask->data, KV_min_max.ptr, iter_k, s31, s33);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = Q->ne[1] > 1 ? 2*nsm : nsm;
        const int tiles_nwaves = (ntiles_total + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_total / (max_blocks*tiles_nwaves);

        const int nblocks_stream_k = max_blocks;

        //const bool use_stream_k = cc >= CC_ADA_LOVELACE || tiles_efficiency_percent < 75;
        //  On my RTX-4080 the above is slightly slower for PP. It would be useful to try and see what happens on Blackwell
        const bool use_stream_k = tiles_efficiency_percent < 75 || Q->ne[1] > 2048;

        blocks_num.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_num.y = 1;
        blocks_num.z = 1;

        dst_tmp_meta.alloc(blocks_num.x*ncols * (2*2 + D) * sizeof(float));
    } else {
        GGML_ASSERT(K->ne[1] % KQ_row_granularity == 0);
        const int ntiles_KQ = K->ne[1] / KQ_row_granularity; // Max. number of parallel blocks limited by tensor size.

        int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));

        // parallel_blocks should be at least large enough to achieve max. occupancy for a single wave:
        parallel_blocks = std::max((nsm * max_blocks_per_sm) / ntiles_total, 1);

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_total * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 90 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = Q->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }
    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    GGML_ASSERT(block_dim.x % warp_size == 0);
    fattn_kernel<<<blocks_num, block_dim, nbytes_shared, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *)sinks->data) : nullptr,
        KV_min_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, logit_softcap, ctx.fa_offset, n_head_log2,
        Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3],
        mask ? mask->ne[1] : 0, mask ?  mask->nb[1] : 0,
        Q->nb[1], Q->nb[2], Q->nb[3],
        nb11, nb12, nb13,
        nb21, nb22, nb23,
        KQV->ne[0], KQV->ne[1], KQV->ne[2], KQV->ne[3]
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if (ntiles_total % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            const dim3 block_dim_combine(D, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            flash_attn_mma_stream_k_fixup<D, ncols1, ncols2, KQ_stride>
                <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
                ((float *) KQV->data, dst_tmp_meta.ptr, Q->ne[1], Q->ne[2], K->ne[1]);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(D, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], 1, blocks_num.z);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        flash_attn_mma_combine_results<D>
            <<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream>>>
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <int D, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int ncols         = ncols1 * ncols2;
    constexpr int KQ_per_iter   = D <= 128 && ncols1 <= 64 ? 64 : 32;
    constexpr int nwarps        = (KQ_per_iter == 32 && ncols <= 16) ? 2 : 4;
    constexpr int ntiles        = ncols <= 8 ? 1 : (ncols <= 64 ? 2 : 4);
    constexpr int cols_per_warp = ntiles * tile_B::I;

    static_assert(D     %    tile_B::J  == 0, "bad D");
    static_assert(ncols % cols_per_warp == 0, "bad ncols");

    const ggml_tensor * KQV = dst;
    const int id    = ggml_cuda_get_device();
    const int cc    = ggml_cuda_info().devices[id].cc;

    const int KQ_shared_rows = cp_async_available(cc) ? 2*KQ_per_iter : KQ_per_iter;

    const size_t nbytes_shared_KV      = KQ_shared_rows       * (D           + 8) * sizeof(half);
    const size_t nbytes_shared_mask    = ncols1               * (KQ_per_iter + 8) * sizeof(half);
    const size_t nbytes_shared_combine = nwarps*cols_per_warp * (D           + 8) * sizeof(half);

    const size_t nbytes_shared_total = std::max(nbytes_shared_KV + nbytes_shared_mask, nbytes_shared_combine);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    fattn_kernel_mma_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_mma_ext_f16<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap>;
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_mma_ext_f16<D, ncols1, ncols2, nwarps, KQ_per_iter, ntiles, use_logit_softcap>;
    }

    launch_fattn_mma<D, ncols1, ncols2, KQ_per_iter>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, FATTN_KQ_STRIDE, true, true, true);
}


#define DECL_FATTN_MMA_F16_CASE(D, ncols1, ncols2)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                     \
    <D, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(D, ncols) \
    extern DECL_FATTN_MMA_F16_CASE(D, (ncols)/1, 1); \
    extern DECL_FATTN_MMA_F16_CASE(D, (ncols)/2, 2); \
    extern DECL_FATTN_MMA_F16_CASE(D, (ncols)/4, 4); \
    extern DECL_FATTN_MMA_F16_CASE(D, (ncols)/8, 8); \

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256,   8)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256,  16)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256,  32)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256,  64)

// Kernels with ncols == 128 are only 4% faster due to register pressure.
// DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64, 128)
// DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80, 128)
// DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96, 128)
// DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 128)
// DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128)
// DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 128) // Needs too much shared memory.
