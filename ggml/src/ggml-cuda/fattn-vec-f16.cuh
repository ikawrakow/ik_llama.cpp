//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.cuh"
#include "fattn-vec-common.cuh"

template<int Dk, int Dv, int ncols, ggml_type type_K, ggml_type type_V, bool use_logit_softcap> // Dk, Dv == K, V head sizes
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(Dk, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_vec_ext_f16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int2 * __restrict__ KV_min_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FP16_AVAILABLE)

    // Skip unused kernel variants for faster compilation:
    if constexpr (Dk == Dv || (Dk == 192 && Dv == 128) || (Dk == 576 && Dv == 512)) {
    if (use_logit_softcap && !(Dk == 128 || Dk == 256)) {
        NO_DEVICE_CODE;
        return;
    }
    }
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (ncols > 1) {
        NO_DEVICE_CODE;
        return;
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr vec_dot_KQ_f16_t vec_dot_KQ = get_vec_dot_KQ_f16<Dk>(type_K);
    constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16;
    constexpr dequantize_1_f16_t dequantize_1_v = get_dequantize_1_f16(type_V);

    const int ic0 = blockIdx.x * ncols; // Index of the Q/QKV column to work on.

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    Q += nb03*sequence + nb02* head              + nb01*ic0;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);

    const half  * maskh  = (const half  *) (mask + nb33*(sequence % ne33) + nb31*ic0);
    const float * sinksf = (const float *) (sinks);

    const float slopef = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);

    static_assert(Dk % (2*WARP_SIZE) == 0, "Dk not divisible by 2*WARP_SIZE == 64.");
    static_assert(Dv % (2*WARP_SIZE) == 0, "Dv not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = Dk / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < Dk);

    __shared__ half KQ[ncols*Dk];
    half2 * KQ2 = (half2 *) KQ;

    half kqmax[ncols];
    half kqsum[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -HALF_MAX_HALF;
        kqsum[j] = 0.0f;
    }

    __shared__ half kqmax_shared[ncols][WARP_SIZE];
    __shared__ half kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -HALF_MAX_HALF;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }

    __shared__ half maskh_shared[ncols*Dk];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        maskh_shared[j*Dk + tid] = 0.0f;
    }

    __syncthreads();

    half2 VKQ[ncols] = {{0.0f, 0.0f}};

    const int k_VKQ_max = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].y : ne11;
    const int first_y = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].x : 0;
    if (first_y + blockIdx.y*Dk < k_VKQ_max) {

    // Convert Q to half2 (f16 K) or q8_1 (quantized K) and store in registers:
    half2  Q_h2[ncols][Dk/(2*WARP_SIZE)];
    int   Q_i32[ncols][Dk/(sizeof(int)*QK8_1) == 0 ? 1 : Dk/(sizeof(int)*QK8_1)];
    half2  Q_ds[ncols][Dk/QK8_1 == 0 ? 1 : Dk/QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            if (j0 + nwarps > ncols && j >= ncols) {
                break;
            }

            // Reuse KQ as temporary storage for converting Q to q8_1:
            int   * tmp_q_i32 = (int   *) &KQ[j*Dk];
            half2 * tmp_q_ds  = (half2 *) (tmp_q_i32 + Dk/sizeof(int));

            // Set memory to zero if out of bounds:
            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < Dk/sizeof(int); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i32[i] = 0;
                }
                if (threadIdx.x < Dk/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_half2(0.0f, 0.0f);
                }
                continue;
            }

            const float * Q_f = (const float *) (Q + j*nb01);
#pragma unroll
            for (int i0 = 0; i0 < Dk/sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<half2>(Q_f + 4*i0, scale, tmp_q_i32, tmp_q_ds);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int   * tmp_q_i32 = (int   *) &KQ[j*Dk];
            half2 * tmp_q_ds  = (half2 *) (tmp_q_i32 + Dk/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < Dk/sizeof(int); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i32[j][i0/WARP_SIZE] = tmp_q_i32[i];
                Q_ds[j][i0/WARP_SIZE]  = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 * Q_f2_j = (const float2 *) (Q + j*nb01);

#pragma unroll
            for (int i0 = 0; i0 < Dk/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                const float2 tmp = ncols <= 2 || ic0 + j < ne01 ? Q_f2_j[i] : make_float2(0.0f, 0.0f);
                Q_h2[j][i0/WARP_SIZE] = make_half2(scale, scale) * make_half2(tmp.x, tmp.y);
            }
        }
    }


#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*Dk + tid] = -HALF_MAX_HALF;
    }
    __syncthreads();

    const int k_VKQ_max = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].y : ne11;
    const int first_y = KV_min_max ? KV_min_max[sequence*gridDim.x + blockIdx.x].x : 0;
    //if (KV_min_max && threadIdx.x == 0 && head == 0) {
    //    printf("gridDims = %u, %u, %u, ncols = %d, head = %d, blockIdx.x = %d, blockIdx.y = %d, bounds = %d, %d, ne11 = %d, nb11 = %d, blockIdx.y*Dk = %d\n", gridDim.x, gridDim.y, gridDim.z, ncols, head, blockIdx.x, blockIdx.y, KV_min_max[sequence*gridDim.x + blockIdx.x].x, KV_min_max[sequence*gridDim.x + blockIdx.x].y, ne11, nb11, blockIdx.y*Dk);
    //}
    K     += (first_y + blockIdx.y*Dk) * nb11;
    V     += (first_y + blockIdx.y*Dv) * nb21;
    maskh += (first_y + blockIdx.y*Dk);
    for (int k_VKQ_0 = first_y + blockIdx.y*Dk; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += gridDim.y*Dk,
             // Increment pointers after each loop:
             K += gridDim.y*Dk*nb11, V += gridDim.y*Dv*nb21, maskh += gridDim.y*Dk) {

        // For unknown reasons using a half array of size 1 for kqmax_new causes a performance regression,
        // see https://github.com/ggerganov/llama.cpp/pull/7061 .
        // Therefore this variable is defined twice but only used once (so that the compiler can optimize out the unused variable).
        half kqmax_new = kqmax[0];
        half kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

        if (mask) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                maskh_shared[j*Dk + tid] = slopeh*maskh[j*ne11 + tid];
            }
            __syncthreads();
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < Dk; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > Dk && i_KQ >= Dk) || (FATTN_KQ_STRIDE % Dk != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                half sum = vec_dot_KQ(K + i_KQ*nb11, Q_h2[j], Q_i32[j], Q_ds[j]);
                sum = warp_reduce_sum((float)sum);

                if (use_logit_softcap) {
                    sum = logit_softcap*tanhf(sum);
                }

                sum += maskh_shared[j*Dk + i_KQ];

                if (ncols == 1) {
                    kqmax_new        = ggml_cuda_hmax(kqmax_new,        sum);
                } else {
                    kqmax_new_arr[j] = ggml_cuda_hmax(kqmax_new_arr[j], sum);
                }

                if (threadIdx.x == 0) {
                    KQ[j*Dk + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = ncols == 1 ? kqmax_new : kqmax_new_arr[j];

            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const half KQ_max_scale = hexp(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const half val = hexp(KQ[j*Dk + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*Dk + tid] = val;

            VKQ[j] *= __half2half2(KQ_max_scale);
        }

        __syncthreads();

#pragma unroll
        for (int k0 = 0; k0 < Dv; k0 += 2) {
            if (FATTN_KQ_STRIDE % Dv != 0 && k_VKQ_0 + k0 >= ne11) {
                break;
            }

            half2 V_k;
            reinterpret_cast<half&>(V_k.x) = dequantize_1_v(V + (k0 + 0)*nb21, tid);
            reinterpret_cast<half&>(V_k.y) = dequantize_1_v(V + (k0 + 1)*nb21, tid);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_k*KQ2[j*(Dk/2) + k0/2];
            }
        }

        __syncthreads();
    }
    }

    if (sinksf && blockIdx.y == 0) {
        const half sink = __float2half(sinksf[head]);

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = fmaxf(kqmax[j], sink);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            half kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const half KQ_max_scale = hexp(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const half val = hexp(sink - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale;

            if (tid == 0) {
                kqsum[j] += val;
            }

            VKQ[j] *= __half2half2(KQ_max_scale);
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = warp_reduce_sum((float)kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum((float)kqsum[j_VKQ]);

        half dst_val = (__low2half(VKQ[j_VKQ]) + __high2half(VKQ[j_VKQ]));
        if (gridDim.y == 1) {
            dst_val /= kqsum[j_VKQ];
        }
        dst[(((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*gridDim.y + blockIdx.y)*Dv + tid] = dst_val;
    }

    if (gridDim.y != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[((sequence*ne01 + ic0 + tid)*ne02 + head)*gridDim.y + blockIdx.y] = make_float2(kqmax[tid], kqsum[tid]);
    }
#else
   NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
}

template <int Dk, int Dv, int cols_per_block, ggml_type type_K, ggml_type type_V, bool use_logit_softcap>
void ggml_cuda_flash_attn_ext_vec_f16_case_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps = Dk/WARP_SIZE;
    fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f16<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>;
    constexpr bool need_f16_K = Dk != 128 && Dk != 192;
    constexpr bool need_f16_V = Dv != 128 && Dv != 64;
    constexpr size_t nbytes_shared = 0;
    launch_fattn<Dv, cols_per_block, 1>(ctx, dst, fattn_kernel, nwarps, nbytes_shared, Dv, need_f16_K, need_f16_V);
}

template <int Dk, int Dv, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_vec_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];

    const int32_t precision = KQV->op_params[3];
    GGML_ASSERT(precision == GGML_PREC_DEFAULT);

    GGML_ASSERT(K->type == type_K);
    GGML_ASSERT(V->type == type_V);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (Q->ne[1] == 1 || GGML_CUDA_CC_IS_NVIDIA(cc)) {
        constexpr int cols_per_block = 1;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] == 2) {
        constexpr int cols_per_block = 2;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    if (Q->ne[1] <= 4) {
        constexpr int cols_per_block = 4;
        if (logit_softcap == 0.0f) {
            constexpr bool use_logit_softcap = false;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        } else {
            constexpr bool use_logit_softcap = true;
            ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
        }
        return;
    }

    constexpr int cols_per_block = 8;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_flash_attn_ext_vec_f16_case_impl<Dk, Dv, cols_per_block, type_K, type_V, use_logit_softcap>(ctx, dst);
    }
}

#define DECL_FATTN_VEC_F16_CASE(D, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_f16_case                     \
    <D, D, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_VEC_F16_CASE_DKDV(Dk, Dv, type_K, type_V)                          \
    template void ggml_cuda_flash_attn_ext_vec_f16_case                     \
    <Dk, Dv, type_K, type_V>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0);

//extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16,  GGML_TYPE_IQ4_NL);
//extern DECL_FATTN_VEC_F16_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL);
//extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_IQ4_NL);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL);
//extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_F16,  GGML_TYPE_IQ4_NL);
//extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0);

extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16);

extern DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16);

extern DECL_FATTN_VEC_F16_CASE_DKDV(192, 128, GGML_TYPE_F16, GGML_TYPE_F16);
extern DECL_FATTN_VEC_F16_CASE_DKDV(192, 128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0);
