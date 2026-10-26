// Adapted from https://github.com/ggml-org/llama.cpp/blob/fa72aeccb23947074c12b5fec25f5b6ced28cbfe/ggml/src/ggml-cuda/fattn-tile.cuh
//
// Copyright (C) 2025 The ggml authors
// Copyright (C) 2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-tile-f32.cuh"

// Only the fp32 path and head size 512 are ported. On sm_60 the fp16 path moved Gemma-4 PPL by 5% at n_ctx 8192.
// K and V pass through shared memory nbatch_K columns at a time and only the Q tile grows with the head size,
// so 16 columns of head size 512 fit into 48 kiB.

// nbatch_fa == number of KQ rows to process per iteration
// nbatch_K  == number of K columns to load in parallel for KQ calculation

// The ROCm compiler cannot handle templating in __launch_bounds__.
// As a workaround, define a macro to package the kernel parameters as uint32_t:
#define GGML_CUDA_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                          \
        static_assert((nthreads)          <= 512, "bad nthreads");                                    \
        static_assert((occupancy)         <=   8, "bad occupancy");                                   \
        static_assert((nbatch_fa)         <= 256, "bad nbatch_fa");                                   \
        static_assert((nbatch_K)          <= 256, "bad nbatch_K");                                    \
        return ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);    \
    }                                                                                                 \

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  4, 128, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512,  8, 256, 2,  32,  64)
    GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2,  32,  64)

    return 0;
}

static constexpr __host__ __device__ int ggml_cuda_fattn_tile_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols) >> 0) & ((1 << 10) - 1);
}

static constexpr __host__ __device__ int ggml_cuda_fattn_tile_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols) >> 10) & ((1 << 4) - 1);
}

static constexpr __host__ __device__ int ggml_cuda_fattn_tile_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols) >> 14) & ((1 << 9) - 1);
}

static constexpr __host__ __device__ int ggml_cuda_fattn_tile_get_nbatch_K(const int DKQ, const int DV, const int ncols) {
    return (ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols) >> 23) & ((1 << 9) - 1);
}

// Mainline keeps these helpers in common.cuh.

static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
#if __CUDA_ARCH__ >= CC_VOLTA
    return 16;
#else
    return 8;
#endif // __CUDA_ARCH__ >= CC_VOLTA
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes, especially on AMD.
// Important: do not use this function if dst and src both point at registers.
template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(void * __restrict__ dst, const void * __restrict__ src) {
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((int2 *) dst)[i] = ((const int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}

static __device__ __forceinline__ void ggml_cuda_mad(float & acc, const float v, const float u) {
    acc += v*u;
}

template <int n>
struct ggml_cuda_unroll {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_cuda_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_cuda_unroll<1> {
    template <typename Func, typename... Args>
    __device__ void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

template<int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_load_tile(
        const half2 * const __restrict__ KV, float * const __restrict__ tile_KV, const int stride_KV, const int i_sup) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    auto load = [&] __device__ (const int n) {
        const int stride_j = warp_size >> n;

        if (stride_j == 0) {
            return;
        }

        const int j0_start = stride_j == warp_size ? 0 : (J/cpy_ne) - (J/cpy_ne) % (2*stride_j);
        const int j0_stop  =                             (J/cpy_ne) - (J/cpy_ne) % (1*stride_j);
        const int stride_i = warp_size / stride_j;

        if (j0_start == j0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < I; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_j == warp_size ? 0 : threadIdx.x / stride_j);

            if (i0 + nwarps*stride_i <= I || i < I) {
#pragma unroll
                for (int j0 = j0_start; j0 < j0_stop; j0 += stride_j) {
                    const int j = j0*(cpy_ne/2) + (stride_j == warp_size ? threadIdx.x : threadIdx.x % stride_j)*(cpy_ne/2);

                    const half2 zero[cpy_ne/2] = {{0.0f, 0.0f}};
                    __align__(16) half2 tmp_h2[cpy_ne/2];
                    ggml_cuda_memcpy_1<sizeof(tmp_h2)>(
                        tmp_h2, !oob_check || i < i_sup ? KV + i*stride_KV + j : zero);

                    __align__(16) float2 tmp_f2[cpy_ne/2];
#pragma unroll
                    for (int l = 0; l < cpy_ne/2; ++l) {
                        tmp_f2[l] = __half22float2(tmp_h2[l]);
                    }
                    ggml_cuda_memcpy_1<sizeof(tmp_f2)>(tile_KV + i*(J + J_padding) + 2*j, tmp_f2);
                }
            }
        }
    };
    // 1: max 32*16=512 bytes, 128 float
    // 2: max 16*16=256 bytes,  64 float
    // 3: max  8*16=128 bytes,  32 float
    // 4: max  4*16= 64 bytes,  16 float
    // 5: max  2*16= 32 bytes,   8 float
    static_assert(J % 8 == 0, "bad J");
    static_assert(J % cpy_ne == 0, "bad J");
    ggml_cuda_unroll<5>{}(load);
}

// Function that performs a single iteration in for the KQ matrix multiplication:
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_iter_KQ(
        float       * const Q_tmp,
        const half2 * const __restrict__ K_h2,
        float       * const KV_tmp,
        const int stride_K2,
        const int k_VKQ_0,
        const int k_VKQ_sup,
        const int k_KQ_0,
        float * KQ_acc) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    flash_attn_tile_load_tile<warp_size, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    __syncthreads();

    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __align__(16) float K_k[nbatch_fa/(np*warp_size)][cpy_ne];
        __align__(16) float Q_k[cpw][cpy_ne];

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;

            ggml_cuda_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K + cpy_ne) + k_KQ_1]);
        }
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (threadIdx.y / np)*cpw;

            ggml_cuda_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*DKQ + k_KQ_0 + k_KQ_1]);
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                for (int k = 0; k < cpy_ne; ++k) {
                    ggml_cuda_mad(KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0], K_k[i_KQ_0/(np*warp_size)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    if (k_KQ_0 + nbatch_K < DKQ) {
        __syncthreads(); // Sync not needed on last iteration.
    }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int DV, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_iter(
        float * const Q_tmp,
        const half2 * const __restrict__ K_h2,
        const half2 * const __restrict__ V_h2,
        const half  * const __restrict__ mask,
        const int ne01,
        const float logit_softcap,
        const float slope,
        float * const KQ,
        float * const KV_tmp,
        const int stride_K2,
        const int stride_V2,
        const int stride_mask,
        float * const KQ_max,
        float * const KQ_sum,
        float2 * const VKQ,
        const int k_VKQ_0,
        const int k_VKQ_max,
        const int col_Q_0) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

    constexpr int DVp = (DV + 2*warp_size - 1) & ~(2*warp_size - 1); // DV padded to multiple of 2*warp_size.

    // KQ_cs == KQ chunk size, number of KQ values in j direction to store as one contiguous chunk in memory.
    // KQ is originally 2D but uses a Z-shaped 3D memory pattern like KQ[ncols/KQ_cs][DVp][KQ_cs].
    constexpr int KQ_cs = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;
    static_assert(cpw % KQ_cs == 0, "bad KQ_cs");
    const int k_VKQ_sup = k_VKQ_max - k_VKQ_0; // k supremum, only smaller k values have valid KV data

    float KQ_max_new[cpw];
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_max_new[jc0] = KQ_max[jc0];
    }

    float KQ_acc[nbatch_fa/(np*warp_size) * cpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

    // KQ = K @ Q matrix multiplication:
    constexpr int nbatch_K_last = DKQ % nbatch_K;
#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < DKQ - nbatch_K_last; k_KQ_0 += nbatch_K) {
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }
    if (nbatch_K_last > 0) {
        constexpr int k_KQ_0 = DKQ - nbatch_K_last;
        flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K_last, use_logit_softcap, oob_check>(
            Q_tmp, K_h2, KV_tmp, stride_K2, k_VKQ_0, k_VKQ_sup, k_KQ_0, KQ_acc);
    }

    // Apply logit softcap + mask, update KQ_max:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int j = (col_Q_0 + (jc0 + (threadIdx.y / np)*cpw)/ncols2) % ne01;

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;

            if (use_logit_softcap) {
                KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] = logit_softcap * tanhf(KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0]);
            }

            if (!oob_check || i_KQ < k_VKQ_sup) {
                KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] += (ncols2 > 1 || mask) ?
                    slope*__half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]) : 0.0f;

                KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[(i_KQ_0/(np*warp_size))*cpw + jc0] + FATTN_KQ_MAX_OFFSET);
            }
        }

        KQ_max_new[jc0] = warp_reduce_max(KQ_max_new[jc0]);
    }

    __syncthreads();

    // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; jc0 += KQ_cs) {
        __align__(16) float tmp[nbatch_fa/(np*warp_size)][KQ_cs];

#pragma unroll
        for (int jc1 = 0; jc1 < KQ_cs; ++jc1) {
            const int jc = jc0 + jc1;

            const float KQ_max_scale = expf(KQ_max[jc] - KQ_max_new[jc]);
            KQ_max[jc] = KQ_max_new[jc];

            float KQ_sum_add = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
                const float val = !oob_check || i0 + (threadIdx.y % np)*warp_size + threadIdx.x < static_cast<uint32_t>(k_VKQ_sup) ?
                    expf(KQ_acc[(i0/(np*warp_size))*cpw + jc] - KQ_max[jc]) : 0.0f;
                KQ_sum_add += val;
                tmp[i0/(np*warp_size)][jc1] = val;
            }
            KQ_sum[jc] = KQ_sum[jc]*KQ_max_scale + KQ_sum_add;

#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].x *= KQ_max_scale;
                VKQ[jc*((DVp/2)/warp_size) + i0/warp_size].y *= KQ_max_scale;
            }
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += np*warp_size) {
            const int i = i0 + (threadIdx.y % np)*warp_size + threadIdx.x;

            ggml_cuda_memcpy_1<sizeof(tmp[0])>(
                KQ + (jc0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs))*(nbatch_fa*KQ_cs) + i*KQ_cs,
                tmp[i0/(np*warp_size)]);
        }
    }

    // VKQ = V @ KQ matrix multiplication:
    static_assert(DV <= DKQ, "bad DV");
    static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K*2/3) == 0), "bad nbatch_K");
    constexpr int nbatch_V = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K*2/3) * nbatch_fa / DV; // Number of V columns that fit in SRAM for K.
    static_assert(nbatch_fa % nbatch_V == 0, "bad nbatch_V");
    static_assert(nbatch_V % np == 0, "bad nbatch_V");
#pragma unroll
    for (int k0 = 0; k0 < nbatch_fa; k0 += nbatch_V) {
        flash_attn_tile_load_tile<warp_size, nwarps, nbatch_V, DV, 0, oob_check>
            (V_h2 + int64_t(k_VKQ_0 + k0)*stride_V2, KV_tmp, stride_V2, k_VKQ_sup - k0);
        __syncthreads();

#pragma unroll
        for (int k1 = 0; k1 < nbatch_V; k1 += np) {
            __align__(16) float2 V_k[(DVp/2)/warp_size];
            __align__(16) float  KQ_k[cpw];

            constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
            for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
                ggml_cuda_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_size)], &KV_tmp[(k1 + threadIdx.y % np)*DV + i0 + threadIdx.x*cpy_ne_D]);
            }
#pragma unroll
            for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; jc_VKQ_0 += KQ_cs) {
                const int jc_KQ = jc_VKQ_0/KQ_cs + (threadIdx.y / np)*(cpw/KQ_cs);

                ggml_cuda_memcpy_1<KQ_cs*sizeof(float)>(
                    &KQ_k[jc_VKQ_0], KQ + jc_KQ*(nbatch_fa*KQ_cs) + (k0 + k1 + threadIdx.y % np)*KQ_cs);
            }

#pragma unroll
            for (int i0 = 0; i0 < DVp/2; i0 += warp_size) {
#pragma unroll
                for (int jc_VKQ_0 = 0; jc_VKQ_0 < cpw; ++jc_VKQ_0) {
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].x += V_k[i0/warp_size].x*KQ_k[jc_VKQ_0];
                    VKQ[jc_VKQ_0*((DVp/2)/warp_size) + i0/warp_size].y += V_k[i0/warp_size].y*KQ_k[jc_VKQ_0];
                }
            }
        }

        __syncthreads();
    }
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_tile_get_occupancy(DKQ, DV, ncols1*ncols2))
static __global__ void flash_attn_tile_new(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const float logit_softcap,
        const uint32_t n_head_log2,
        const int ne01, const int ne02, const int ne11, const int ne12,
        const int nb01, const int nb02, const int nb03,
        const int nb11, const int nb12, const int64_t nb13,
        const int nb21, const int nb22, const int64_t nb23,
        const int nb31) {
#ifdef FP16_AVAILABLE
    static_assert(ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");

    constexpr int ncols     = ncols1*ncols2;
    constexpr int warp_size = 32;
    constexpr int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, ncols1*ncols2) / warp_size;
    constexpr int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1*ncols2);
    constexpr int nbatch_K  = ggml_cuda_fattn_tile_get_nbatch_K (DKQ, DV, ncols1*ncols2);

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int col_Q_0 = blockIdx.x * ncols1; // Index of the first Q column for this CUDA block to work on.

    const int sequence = blockIdx.z / (ne02/ncols2);
    const int head0 = blockIdx.z*ncols2 - sequence*ne02; // == blockIdx.z % (ne02/ncols2)
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f  = (const float *) (Q + nb03*sequence + nb02* head0);
    const half2 * K_h2 = (const half2 *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
    const half2 * V_h2 = (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio)); // K and V have same shape

    const half * maskh = (const half *) mask;

    const int stride_K2   = nb11 / sizeof(half2);
    const int stride_V2   = nb21 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int cpw = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp.
    constexpr int np  = nwarps > ncols ? nwarps/ncols : 1; // Number of parallel warps per Q column.
    static_assert(np == 1, "bad np");
    static_assert(nbatch_fa % (np*warp_size) == 0, "nbatch_fa % (np*warp_size) != 0");

    constexpr int DKQp = (DKQ + 2*warp_size - 1) & ~(2*warp_size - 1); // DKQ padded to multiple of 2*warp_size.
    constexpr int DVp  = (DV  + 2*warp_size - 1) & ~(2*warp_size - 1); // DV  padded to multiple of 2*warp_size.

    // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
    // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
    //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
    // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
    // VKQ == Accumulators in registers for the final VKQ result.
    __shared__ float Q_tmp[ncols * DKQ];
    __shared__ float KV_tmp[nbatch_fa * (nbatch_K + cpy_ne) + DVp-DV];
    __shared__ float KQ[ncols * nbatch_fa];
    __align__(16) float2 VKQ[cpw * ((DVp/2)/warp_size)] = {{0.0f, 0.0f}};

    float KQ_max[cpw];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float KQ_sum[cpw] = {0.0f};

#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y / np)*cpw;

        const int j = jc / ncols2;
        const int c = jc % ncols2;

        constexpr int cpy_ne_D = cpy_ne < DKQp/warp_size ? cpy_ne : DKQp/warp_size;

#pragma unroll
        for (int i0 = 0; i0 < DKQp; i0 += np*warp_size*cpy_ne_D) {
            if (i0 + np*warp_size*cpy_ne_D <= DKQ || i0 + (threadIdx.y % np)*(warp_size*cpy_ne_D) + threadIdx.x*cpy_ne_D < DKQ) {
                __align__(16) float tmp_f[cpy_ne_D] = {0.0f};
                ggml_cuda_memcpy_1<sizeof(tmp_f)>
                    (tmp_f, &Q_f[c*(nb02/sizeof(float)) + (col_Q_0 + j) % ne01*(nb01/sizeof(float))
                                 + i0 + (threadIdx.y % np)*(warp_size*cpy_ne_D) + threadIdx.x*cpy_ne_D]);

#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                    tmp_f[i1] *= scale;
                }

                ggml_cuda_memcpy_1<sizeof(tmp_f)>(
                    &Q_tmp[jc* DKQ    + i0   + (threadIdx.y % np)*(warp_size*cpy_ne_D)   + threadIdx.x* cpy_ne_D],
                    tmp_f);
            }
        }
    }

    __syncthreads();

    // Main loop over KV cache:
    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
    for (int k_VKQ_0 = blockIdx.y*nbatch_fa; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*nbatch_fa) {
        constexpr bool oob_check = false;
        flash_attn_tile_iter<warp_size, nwarps, ncols1, ncols2, DKQ, DV, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>
            (Q_tmp, K_h2, V_h2, maskh, ne01, logit_softcap, slope, KQ, KV_tmp,
            stride_K2, stride_V2, stride_mask, KQ_max, KQ_sum, VKQ, k_VKQ_0, k_VKQ_max, col_Q_0);
    }

#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        KQ_sum[jc0] = warp_reduce_sum(KQ_sum[jc0]);
    }

    // Write back results:
#pragma unroll
    for (int jc0 = 0; jc0 < cpw; ++jc0) {
        const int jc = jc0 + (threadIdx.y/np)*cpw;

        const int j = jc / ncols2;
        const int c = jc % ncols2;

        if (ncols1 > 1 && col_Q_0 + j >= ne01) {
            return;
        }

        const float scale = gridDim.y == 1 ? 1.0f/KQ_sum[jc0] : 1.0f;

        const int j_dst_unrolled = ((sequence*ne01 + col_Q_0 + j)*ne02 + head0 + c)*gridDim.y + blockIdx.y;

        constexpr int cpy_ne_D = cpy_ne < DVp/warp_size ? cpy_ne : DVp/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < DVp; i0 += warp_size*cpy_ne_D) {
            if (i0 + warp_size*cpy_ne_D <= DV || i0 + threadIdx.x*cpy_ne_D < DV) {
#pragma unroll
                for (int i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].x *= scale;
                    VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size) + i1].y *= scale;
                }
                ggml_cuda_memcpy_1<cpy_ne_D*4>(
                    &dst[j_dst_unrolled*DV + i0 + threadIdx.x*cpy_ne_D],
                    &VKQ[jc0*((DVp/2)/warp_size) + i0/(2*warp_size)]);
            }
        }

        if (gridDim.y != 1 && threadIdx.x == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[jc0], KQ_sum[jc0]);
        }
    }
#else
    NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

// Scan the mask to find the last KV block that is not fully masked out, so that the main loop can stop early.
template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_tile_new_mask_to_KV_max(
        const half2 * __restrict__ mask, int * __restrict__ KV_max, const int ne30, const int s31, const int s33) {
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

        all_inf = __all_sync(0xffffffff, all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = __all_sync(0xffffffff, all_inf);

        if (!all_inf) {
            break;
        }
    }

    if (threadIdx.x == 0) {
        KV_max[sequence*ne31 + jt] = KV_max_sj + FATTN_KQ_STRIDE;
    }
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_tile_new_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float        * __restrict__ dst,
        const int parallel_blocks) {
    const int j_dst_unrolled = (blockIdx.z*gridDim.x + blockIdx.x)*gridDim.y + blockIdx.y;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *) VKQ_meta)[i];
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

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap>
static void launch_fattn_tile_new(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);
    GGML_ASSERT(mask && mask->type == GGML_TYPE_F16);
    GGML_ASSERT(mask->ne[1] >= GGML_PAD(Q->ne[1], 16) &&
        "the Flash-Attention CUDA kernel requires the mask to be padded to 16 and at least n_queries big");
    GGML_ASSERT(K->ne[1] % FATTN_KQ_STRIDE == 0 && "Incorrect KV cache padding.");
    GGML_ASSERT(Q->ne[3] == 1);

    constexpr int ncols     = ncols1*ncols2;
    constexpr int nwarps    = ggml_cuda_fattn_tile_get_nthreads (DKQ, DV, ncols) / WARP_SIZE;
    constexpr int nbatch_fa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, ncols);

    const auto fattn_kernel = flash_attn_tile_new<DKQ, DV, ncols1, ncols2, use_logit_softcap>;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<int>    KV_max(pool);
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

    if (K->type != GGML_TYPE_F16) {
        K_f16.alloc(ggml_nelements(K));
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
        to_fp16(K_data, K_f16.ptr, 1, ggml_nelements(K), main_stream);
        K_data = (const char *) K_f16.ptr;

        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        nb11 = nb11*bs*sizeof(half)/ts;
        nb12 = nb12*bs*sizeof(half)/ts;
        nb13 = nb13*bs*sizeof(half)/ts;
    }

    if (V->type != GGML_TYPE_F16) {
        V_f16.alloc(ggml_nelements(V));
        to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
        to_fp16(V_data, V_f16.ptr, 1, ggml_nelements(V), main_stream);
        V_data = (const char *) V_f16.ptr;

        const size_t bs = ggml_blck_size(V->type);
        const size_t ts = ggml_type_size(V->type);

        nb21 = nb21*bs*sizeof(half)/ts;
        nb22 = nb22*bs*sizeof(half)/ts;
        nb23 = nb23*bs*sizeof(half)/ts;
    }

    const int ntiles_x     = (Q->ne[1] + ncols1 - 1) / ncols1;
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z     = (gqa_ratio + ncols2 - 1) / ncols2;
    const int ntiles_total = ntiles_x * ntiles_z * K->ne[2];
    const int ntiles_KV    = (K->ne[1] + nbatch_fa - 1) / nbatch_fa;

    // Only worth the overhead if there is at least one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped.
    if (Q->ne[1] >= 1024) {
        const int s31 = mask->nb[1] / sizeof(half2);
        const int s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, 1, 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        KV_max.alloc(ntiles_x);
        flash_attn_tile_new_mask_to_KV_max<ncols1><<<blocks_num_KV_max, block_dim_KV_max, 0, main_stream>>>
            ((const half2 *) mask->data, KV_max.ptr, K->ne[1]/FATTN_KQ_STRIDE, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(WARP_SIZE, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x*block_dim.y, 0));

    // parallel_blocks must not be larger than what the tensor size allows:
    int parallel_blocks = std::min(std::max((nsm*max_blocks_per_sm) / ntiles_total, 1), ntiles_KV);

    // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
    // Test whether parallel_blocks can be set to a higher value for better efficiency.
    const int blocks_per_wave = nsm * max_blocks_per_sm;
    int nwaves_best = 0;
    int efficiency_percent_best = 0;
    for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
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

    const dim3 blocks_num(ntiles_x, parallel_blocks, ntiles_z*K->ne[2]);

    if (parallel_blocks > 1) {
        dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
        dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
    }

    float scale = 1.0f;
    memcpy(&scale, (const float *) KQV->op_params + 0, sizeof(float));

    fattn_kernel<<<blocks_num, block_dim, 0, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        (const char *) mask->data,
        KV_max.get(),
        parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, 0.0f, 1.0f, 1.0f, 0.0f, 1u,
        Q->ne[1], Q->ne[2], K->ne[1], K->ne[2],
        Q->nb[1], Q->nb[2], Q->nb[3],
        nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask->nb[1]
    );
    CUDA_CHECK(cudaGetLastError());

    if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        flash_attn_tile_new_combine_results<DV>
            <<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream>>>
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <int DKQ, int DV, int ncols2, bool use_logit_softcap>
static void launch_fattn_tile_new_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    if (Q->ne[1] > 8/ncols2) {
        launch_fattn_tile_new<DKQ, DV, 16/ncols2, ncols2, use_logit_softcap>(ctx, dst);
        return;
    }

    if constexpr (ncols2 == 4) {
        if (Q->ne[1] > 1) {
            launch_fattn_tile_new<DKQ, DV, 2, ncols2, use_logit_softcap>(ctx, dst);
            return;
        }
    }

    launch_fattn_tile_new<DKQ, DV, 1, ncols2, use_logit_softcap>(ctx, dst);
}

void ggml_cuda_flash_attn_ext_tile_new(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];

    constexpr bool use_logit_softcap = false;
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio % 8 == 0) {
        launch_fattn_tile_new_switch_ncols1<512, 512, 8, use_logit_softcap>(ctx, dst);
    } else {
        launch_fattn_tile_new_switch_ncols1<512, 512, 4, use_logit_softcap>(ctx, dst);
    }
}

bool ggml_cuda_fattn_tile_new_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // Only the variants that pack 4 or 8 Q heads per K/V head are compiled. Everything else stays on the CPU.
    if (K->ne[0] != 512 || V->ne[0] != 512 || !mask || sinks || max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (K->ne[1] % FATTN_KQ_STRIDE != 0 || Q->ne[3] != 1 || Q->ne[2] % K->ne[2] != 0 || (Q->ne[2] / K->ne[2]) % 4 != 0) {
        return false;
    }

    // A quantized cache is converted to f16 first. Q8_KV has no converter.
    if ((K->type != GGML_TYPE_F16 && !ggml_get_to_fp16_cuda(K->type)) ||
        (V->type != GGML_TYPE_F16 && !ggml_get_to_fp16_cuda(V->type))) {
        return false;
    }

    // The conversion reads ggml_nelements() as one contiguous run, so it cannot handle a strided view.
    if ((K->type != GGML_TYPE_F16 && !ggml_is_contiguously_allocated(K)) ||
        (V->type != GGML_TYPE_F16 && !ggml_is_contiguously_allocated(V))) {
        return false;
    }

    return true;
}
