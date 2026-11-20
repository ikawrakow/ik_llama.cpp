#pragma once

#include "mmv-f8.cuh"
#include "mma_new.cuh"
#include "quantize.cuh"

template <int bytes>
static __device__ __forceinline__ void f8_pp_copy(void * dst, const void * src, int count) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
    const uint32_t address = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::
        "r"(address), "l"(src), "n"(bytes), "r"(count));
#else
    if (count) memcpy(dst, src, count);
    if (count < bytes) memset((char *)dst + count, 0, bytes-count);
#endif
}

struct f8_moe_routing {
    const int32_t * rows;
    const int32_t * bounds;
    const int32_t * tiles;
    int experts;
    size_t row_stride;
    size_t expert_stride;
    const char * bias;
    size_t bias_stride;
    const float * up;
    uint8_t * codes;
    float * out_scales;
    bool sorted;
};

static constexpr int f8_activation_group = 128;

template <int G>
using f8_activation = std::conditional_t<G == 0, half, uint8_t>;

static bool f8_mma_available(int cc) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= CC_ADA_LOVELACE;
#else
    GGML_UNUSED(cc);
    return false;
#endif
}

static bool f8_w8a8(const ggml_tensor * weight, int cc) {
    return (weight->type == GGML_TYPE_F8_E4M3_KS || weight->type == GGML_TYPE_F8_E4M3_B128) &&
        ggml_get_input_scale(weight) != 0.0f && f8_mma_available(cc);
}

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080
static __device__ __forceinline__ uint8_t f8_encode(float value, float scale) {
    return scale == 0 ? 0 : __nv_cvt_float_to_fp8(__fdiv_rn(value, scale), __NV_SATFINITE, __NV_E4M3);
}
#endif

template <int G>
static __device__ void f8_quantize_group(const float * input, uint8_t * output, float * scales, int n, int group,
        const float * gate = nullptr) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080 && __CUDA_ARCH__ >= 890
    const int lane = threadIdx.x%32;
    float values[G/32], amax = 0;
#pragma unroll
    for (int j = 0; j < G/32; ++j) {
        const int k = group*G + lane + 32*j;
        values[j] = k < n ? (gate ? gate[k] * input[k] / (1.0f + expf(-gate[k])) : input[k]) : 0;
        amax = fmaxf(amax, fabsf(values[j]));
    }
    amax = warp_reduce_max(amax);
    const float scale = __fdiv_rn(amax, 448.0f);
    if (lane == 0) scales[group] = scale == 0 ? 1.0f : scale;
#pragma unroll
    for (int j = 0; j < G/32; ++j) {
        const int k = group*G + lane + 32*j;
        if (k < n) output[k] = f8_encode(values[j], scale);
    }
#else
    GGML_UNUSED(input); GGML_UNUSED(output); GGML_UNUSED(scales); GGML_UNUSED(n); GGML_UNUSED(group); GGML_UNUSED(gate);
    NO_DEVICE_CODE;
#endif
}

template <int G>
static __global__ void f8_quantize(const float * input, uint8_t * output, float * scales, int n, int stride) {
    const int group = blockIdx.x*4 + threadIdx.x/32;
    if (group < (n+G-1)/G) {
        f8_quantize_group<G>(input + blockIdx.y*stride, output + size_t(blockIdx.y)*n,
                scales + size_t(blockIdx.y)*((n+G-1)/G), n, group);
    }
}

template <int G>
static __global__ void f8_silu_quantize(const float * gate, const float * up, uint8_t * output, float * scales, int n, int stride) {
    const int group = blockIdx.x*4 + threadIdx.x/32;
    if (group < (n+G-1)/G) {
        f8_quantize_group<G>(up + blockIdx.y*stride, output + size_t(blockIdx.y)*n,
                scales + size_t(blockIdx.y)*((n+G-1)/G), n, group, gate + blockIdx.y*stride);
    }
}

template <int G>
static __global__ void f8_norm_quantize(const float * x, const float * w, uint8_t * output, float * scales,
        int k, int stride, float eps, float fixed) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080 && __CUDA_ARCH__ >= 890
    __shared__ float shared[G ? 512 : 32];
    const float * xr = x + size_t(blockIdx.x)*stride;
    const int lane = threadIdx.x%32, warp = threadIdx.x/32;
    float values[16];
    float tmp = 0;
    int count = 0;
    for (int col = threadIdx.x; col < k; col += 1024) {
        const float xi = xr[col];
        values[count++] = xi;
        tmp += xi * xi;
    }
    tmp = warp_reduce_sum(tmp);
    if (lane == 0) shared[warp] = tmp;
    __syncthreads();
    tmp = shared[lane];
    tmp = warp_reduce_sum(tmp);
    const float mean = tmp / k;
    const float scale = rsqrtf(mean + eps);
    if constexpr (G == 0) {
        __shared__ float qscale;
        float amax = 0;
        count = 0;
        for (int col = threadIdx.x; col < k; col += 1024) {
            const float v = scale * w[col] * values[count];
            values[count++] = v;
            amax = fmaxf(amax, fabsf(v));
        }
        __syncthreads();
        amax = block_reduce_max(amax, shared);
        if (threadIdx.x == 0) {
            qscale = fixed > 0 ? fixed : __fdiv_rn(amax, 448.0f);
            scales[blockIdx.x] = qscale == 0 ? 1.0f : qscale;
        }
        __syncthreads();
        count = 0;
        for (int col = threadIdx.x; col < k; col += 1024) {
            output[size_t(blockIdx.x)*k+col] = f8_encode(values[count++], qscale);
        }
    } else {
        __syncthreads();
        for (int i = 0; 1024*i < k; ++i) {
            const int col = threadIdx.x + 1024*i;
            values[i] = col < k ? scale * w[col] * values[i] : 0;
            const float amax = warp_reduce_max(fabsf(values[i]));
            if (lane == 0) shared[32*i + warp] = amax;
        }
        __syncthreads();
        for (int i = 0, col = threadIdx.x; col < k; ++i, col += 1024) {
            float amax = 0;
            for (int j = 0; j < G/32; ++j) amax = fmaxf(amax, shared[32*i + warp/(G/32)*(G/32) + j]);
            const float qscale = __fdiv_rn(amax, 448.0f);
            if (threadIdx.x%G == 0) scales[size_t(blockIdx.x)*((k+G-1)/G) + col/G] = qscale == 0 ? 1.0f : qscale;
            output[size_t(blockIdx.x)*k + col] = f8_encode(values[i], qscale);
        }
    }
#else
    GGML_UNUSED(x); GGML_UNUSED(w); GGML_UNUSED(output); GGML_UNUSED(scales); GGML_UNUSED(k); GGML_UNUSED(stride); GGML_UNUSED(eps);
    NO_DEVICE_CODE;
#endif
}

template <typename layout, int BN, int NW, int stages, bool tail = true, int BK = 64, bool indexed = false, int G = 0>
static __global__ void __launch_bounds__(NW*32, 1) mul_mat_f8_pp(
        int n, int rows, int cols, int ldc, const char * __restrict__ x, const f8_activation<G> * __restrict__ y, float * __restrict__ z,
        f8_moe_routing routing = {}, const float * activation_scales = nullptr) {
#ifdef INT8_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    constexpr bool blocked = std::is_same_v<layout, ggml_cuda_f8_block>;
    constexpr int BM = 128, stride = (G ? BK/4 : BK/2)+4, code_stride = BK/4+4, WC = NW/4, NJ = BN/WC/8;
    extern __shared__ __align__(16) uint32_t storage[];
    uint32_t * codes = storage;
    using packed = std::conditional_t<G == 0, half2, uint32_t>;
    packed * b = (packed *)(codes + stages*BM*code_stride);
    float * scales = (float *)(b + stages*BN*stride);
    float * activation_tile_scales = scales + stages*BM;
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int tile_index = blockIdx.x + blockIdx.y*gridDim.x;
    const int group = tile_index/(8*gridDim.y);
    const int group_rows = min(8, int(gridDim.x)-8*group);
    int first_row = (8*group + (tile_index%(8*gridDim.y))%group_rows)*BM;
    int first_col = ((tile_index%(8*gridDim.y))/group_rows)*BN;
    int expert = 0;
    if constexpr (indexed) {
        if (blockIdx.y >= routing.tiles[routing.experts]) return;
        int hi = routing.experts;
        while (expert + 1 < hi) {
            const int mid = (expert + hi)/2;
            if (routing.tiles[mid] <= int(blockIdx.y)) expert = mid;
            else hi = mid;
        }
        first_row = blockIdx.x*BM;
        first_col = routing.bounds[expert] + (blockIdx.y-routing.tiles[expert])*BN;
        cols = routing.bounds[expert+1];
        x += size_t(expert)*routing.expert_stride;
    }
    const int warp_row = (threadIdx.y/WC)*32, warp_col = (threadIdx.y%WC)*(BN/WC);
    const size_t row_bytes = indexed ? routing.row_stride : layout::row_size(G && blocked ? (n+127)/128*128 : n);
    auto load = [&](int k0, int s) {
        if (k0 < n) {
            if constexpr (!tail) {
                constexpr int chunks = BK/16 + blocked;
                for (int i = tid; i < BM*chunks; i += NW*32) {
                    const int r = i/chunks, j = i%chunks;
                    const char * row = x + size_t(first_row+r)*row_bytes;
                    const char * begin;
                    if constexpr (blocked) begin = row + k0/128*132 + 4 + k0%128;
                    else begin = row + ggml_cuda_f8_row::row_meta_size + k0;
                    const char * src = (const char *)(uintptr_t(begin) & ~uintptr_t(15)) + 16*j;
                    f8_pp_copy<16>(codes + (s*BM+r)*code_stride + 4*j, src, min(16, int(begin+BK-src)));
                }
            } else {
                for (int i = tid; i < BM*BK/4; i += NW*32) {
                    const int r = i/(BK/4), k = k0 + 4*(i%(BK/4));
                    uint32_t * dst = codes + (s*BM+r)*code_stride + i%(BK/4);
                    const bool valid = !tail || (first_row + r < rows && k + 3 < n);
                    const char * row = x + size_t(first_row + r)*row_bytes;
                    const char * src = x;
                    if (valid) {
                        if constexpr (blocked) src = row + k/128*132 + 4 + k%128;
                        else src = row + ggml_cuda_f8_row::row_meta_size + k;
                    }
                    if (tail && valid && uintptr_t(src)%4 != 0) {
                        uint32_t v;
                        memcpy(&v, src, 4);
                        *dst = v;
                    } else if (tail && first_row + r < rows && k < n && !valid) {
                        uint32_t v = 0;
#pragma unroll
                        for (int j = 0; j < 4; ++j) if (k+j < n) v |= uint32_t(layout::code(row, k+j).bits) << (8*j);
                        *dst = v;
                    } else {
                        f8_pp_copy<4>(dst, src, valid ? 4 : 0);
                    }
                }
            }
            if constexpr (G != 0) {
                for (int i = tid; i < BN*((BK+G-1)/G); i += NW*32) {
                    const int c = i/((BK+G-1)/G), g = k0/G+i%((BK+G-1)/G);
                    const bool valid = first_col+c < cols && g < (n+G-1)/G;
                    const float * src = valid ? activation_scales + size_t(first_col+c)*((n+G-1)/G)+g : activation_scales;
                    f8_pp_copy<4>(activation_tile_scales+s*BN*((BK+G-1)/G)+i, src, valid ? 4 : 0);
                }
                for (int i = tid; i < BN*BK/16; i += NW*32) {
                    const int c = i/(BK/16), k = k0 + 16*(i%(BK/16));
                    auto * dst = (uint8_t *)(b + (s*BN+c)*stride + (k-k0)/4);
                    const bool valid = first_col+c < cols && k+15 < n;
                    const auto * src = valid ? y + size_t(first_col+c)*n + k : y;
                    if (valid && uintptr_t(src)%16 == 0) {
                        f8_pp_copy<16>(dst, src, 16);
                    } else {
#pragma unroll
                        for (int j = 0; j < 16; ++j) dst[j] = first_col+c < cols && k+j < n ? y[size_t(first_col+c)*n+k+j] : 0;
                    }
                }
            } else {
                for (int i = tid; i < BN*BK/8; i += NW*32) {
                    const int c = i/(BK/8), k = k0 + 8*(i%(BK/8));
                    half * dst = (half *)(b + (s*BN+c)*stride + (k-k0)/2);
                    const bool valid = (!tail && !indexed) || (first_col+c < cols && k+7 < n);
                    const half * src = valid ? y + size_t(!tail && !indexed ? min(first_col+c, cols-1) : first_col+c)*n + k : y;
                    if (valid && (!tail || uintptr_t(src)%16 == 0)) {
                        f8_pp_copy<16>(dst, src, 16);
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j) dst[j] = first_col+c < cols && k+j < n ? y[size_t(first_col+c)*n+k+j] : __float2half(0);
                    }
                }
            }
            if constexpr (blocked && G != 0) {
                for (int r = tid; r < BM; r += NW*32) {
                    const bool valid = !tail || first_row+r < rows;
                    const char * src = valid ? x + size_t(first_row+r)*row_bytes + k0/128*132 : x;
                    f8_pp_copy<4>(scales+s*BM+r, src, valid ? 4 : 0);
                }
            } else if constexpr (blocked) {
                for (int r = tid; r < BM; r += NW*32) {
                    scales[s*BM+r] = (!tail || first_row+r < rows) ? layout::scale(x + size_t(first_row+r)*row_bytes, k0) : 0;
                }
            }
        }
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;");
#endif
    };
#pragma unroll
    for (int s = 0; s < stages-1; ++s) load(s*BK, s);
    tile<16, 8, float> total[2][NJ], partial[G && blocked ? 1 : 2][G && blocked ? 1 : NJ];
    for (int k0 = 0, step = 0; k0 < n; k0 += BK, ++step) {
        const int s = step%stages;
        load(k0 + (stages-1)*BK, (step+stages-1)%stages);
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.wait_group %0;" :: "n"(stages-1));
#endif
        __syncthreads();
        if constexpr (G != 0) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 890
            int shifts[2][2] = {};
            if constexpr (blocked && !tail) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int l = 0; l < 2; ++l) {
                        const int row = warp_row+16*i+threadIdx.x/4+8*l;
                        const char * ptr = x + size_t(first_row+row)*row_bytes + k0/128*132 + 4 + k0%128;
                        shifts[i][l] = (uintptr_t(ptr)%16)/4;
                    }
                }
            }
            if constexpr (blocked) {
#pragma unroll
                for (int g = 0; g < BK; g += G) {
                    uint32_t at[2][G/32][4];
#pragma unroll
                    for (int k = 0; k < G; k += 32) {
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
#pragma unroll
                            for (int l = 0; l < 4; ++l) {
                                const int row = warp_row+16*i+threadIdx.x/4+8*(l%2);
                                const int shift = shifts[i][l%2];
                                at[i][k/32][l] = codes[(s*BM+row)*code_stride+shift+(g+k)/4+threadIdx.x%4+4*(l/2)];
                            }
                        }
                    }
#pragma unroll
                    for (int j = 0; j < NJ; ++j) {
                        const int c = warp_col+8*j+threadIdx.x/4;
                        uint32_t bt[G/32][2];
#pragma unroll
                        for (int k = 0; k < G; k += 32) {
                            bt[k/32][0] = b[(s*BN+c)*stride+(g+k)/4+threadIdx.x%4];
                            bt[k/32][1] = b[(s*BN+c)*stride+(g+k)/4+threadIdx.x%4+4];
                        }
                        const int sc = warp_col+8*j+2*(threadIdx.x%4);
                        const float column_scale[2] = {
                            activation_tile_scales[(s*BN+sc)*((BK+G-1)/G)+g/G],
                            activation_tile_scales[(s*BN+sc+1)*((BK+G-1)/G)+g/G]};
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            tile<16, 8, float> acc;
#pragma unroll
                            for (int k = 0; k < G/32; ++k) {
                                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                    : "+f"(acc.x[0]), "+f"(acc.x[1]), "+f"(acc.x[2]), "+f"(acc.x[3])
                                    : "r"(at[i][k][0]), "r"(at[i][k][1]), "r"(at[i][k][2]), "r"(at[i][k][3]), "r"(bt[k][0]), "r"(bt[k][1]));
                            }
                            float row_scale[2];
                            if constexpr (blocked) {
                                row_scale[0] = scales[s*BM+warp_row+16*i+threadIdx.x/4];
                                row_scale[1] = scales[s*BM+warp_row+16*i+threadIdx.x/4+8];
                            }
#pragma unroll
                            for (int l = 0; l < total[i][j].ne; ++l) {
                                const float scale = __fmul_rn(row_scale[l/2], column_scale[l%2]);
                                total[i][j].x[l] = fmaf(acc.x[l], scale, total[i][j].x[l]);
                            }
                        }
                    }
                }
            } else {
#pragma unroll
                for (int k = 0; k < BK && k0+k < n; k += 32) {
                    uint32_t at[2][4];
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
#pragma unroll
                        for (int l = 0; l < 4; ++l) {
                            const int row = warp_row+16*i+threadIdx.x/4+8*(l%2);
                            at[i][l] = codes[(s*BM+row)*code_stride+k/4+threadIdx.x%4+4*(l/2)];
                        }
                    }
#pragma unroll
                    for (int j = 0; j < NJ; ++j) {
                        const int c = warp_col+8*j+threadIdx.x/4;
                        const uint32_t b0 = b[(s*BN+c)*stride+k/4+threadIdx.x%4];
                        const uint32_t b1 = b[(s*BN+c)*stride+k/4+threadIdx.x%4+4];
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
                            auto & acc = partial[i][j];
                            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                : "+f"(acc.x[0]), "+f"(acc.x[1]), "+f"(acc.x[2]), "+f"(acc.x[3])
                                : "r"(at[i][0]), "r"(at[i][1]), "r"(at[i][2]), "r"(at[i][3]), "r"(b0), "r"(b1));
                        }
                    }
                    if ((k0+k+32)%G == 0 || k0+k+32 >= n) {
#pragma unroll
                        for (int i = 0; i < 2; ++i) {
#pragma unroll
                            for (int j = 0; j < NJ; ++j) {
                                const int c = warp_col+8*j+2*(threadIdx.x%4);
                                const float column_scale[2] = {
                                    activation_tile_scales[(s*BN+c)*((BK+G-1)/G)+k/G],
                                    activation_tile_scales[(s*BN+c+1)*((BK+G-1)/G)+k/G]};
                                float row_scale[2];
                                if constexpr (blocked) {
                                    row_scale[0] = scales[s*BM+warp_row+16*i+threadIdx.x/4];
                                    row_scale[1] = scales[s*BM+warp_row+16*i+threadIdx.x/4+8];
                                }
#pragma unroll
                                for (int l = 0; l < total[i][j].ne; ++l) {
                                    float scale = column_scale[l%2];
                                    if constexpr (blocked) scale *= row_scale[l/2];
                                    total[i][j].x[l] = fmaf(partial[i][j].x[l], scale, total[i][j].x[l]);
                                    partial[i][j].x[l] = 0;
                                }
                            }
                        }
                    }
                }
            }
#else
            NO_DEVICE_CODE;
#endif
        } else {
#pragma unroll
            for (int k = 0; k < BK/2; k += 8) {
                tile<16, 8, half2> at[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int l = 0; l < at[i].ne; ++l) {
                        const int row = warp_row+16*i+at[i].get_i(l);
                        int shift = 0;
                        if constexpr (blocked && !tail) {
                            const char * ptr = x + size_t(first_row+row)*row_bytes + k0/128*132 + 4 + k0%128;
                            shift = (uintptr_t(ptr)%16)/2;
                        }
                        const uint16_t v = ((const uint16_t *)(codes + (s*BM+row)*code_stride))[shift+k+at[i].get_j(l)];
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11080 && __CUDA_ARCH__ >= 890
                        at[i].x[l] = __nv_cvt_fp8x2_to_halfraw2(v, __NV_E4M3);
#else
                        const float2 f = ggml_cuda_f8_e4m3x2_to_float2(v);
                        at[i].x[l] = make_half2(f.x, f.y);
#endif
                    }
                }
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    tile<8, 8, half2> bt;
                    load_ldmatrix(bt, b + (s*BN+warp_col+8*j)*stride+k, stride);
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        if constexpr (blocked) mma(partial[i][j], at[i], bt);
                        else mma(total[i][j], at[i], bt);
                    }
                }
            }
        }
        if constexpr (blocked && G == 0) {
            if (step%(128/BK) == 128/BK-1 || k0+BK >= n) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
#pragma unroll
                    for (int j = 0; j < NJ; ++j) {
#pragma unroll
                        for (int l = 0; l < total[i][j].ne; ++l) {
                            total[i][j].x[l] = fmaf(partial[i][j].x[l], scales[s*BM+warp_row+16*i+total[i][j].get_i(l)], total[i][j].x[l]);
                            partial[i][j].x[l] = 0;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
#pragma unroll
            for (int l = 0; l < total[i][j].ne; ++l) {
                const int row = first_row + warp_row + 16*i + total[i][j].get_i(l);
                const int col = first_col + warp_col + 8*j + total[i][j].get_j(l);
                if (row < rows && col < cols) {
                    float value = total[i][j].x[l];
                    if constexpr (!blocked) value *= layout::scale(x + size_t(row)*row_bytes);
                    if constexpr (indexed) {
                        if (routing.bias) value += ((const float *)(routing.bias + size_t(expert)*routing.bias_stride))[row];
                        if (G != 0 && routing.codes) {
                            const float u = routing.up[size_t(col)*ldc+row];
                            ((float *)storage)[(col-first_col)*BM+row-first_row] = value * u / (1.0f + expf(-value));
                        } else {
                            z[size_t(routing.sorted ? col : routing.rows[col])*ldc+row] = value;
                        }
                    } else {
                        z[size_t(col)*ldc+row] = value;
                    }
                }
            }
        }
    }
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;");
#endif
    if constexpr (indexed && G != 0) {
        if (routing.codes) {
            __syncthreads();
            const int group = first_row/BM;
            for (int c = threadIdx.y; c < BN && first_col+c < cols; c += NW) {
                f8_quantize_group<128>((const float *)storage + c*BM - group*128, routing.codes + size_t(first_col+c)*rows,
                        routing.out_scales + size_t(first_col+c)*((rows+127)/128), rows, group);
            }
        }
    }
#else
    GGML_UNUSED(n); GGML_UNUSED(rows); GGML_UNUSED(cols); GGML_UNUSED(ldc);
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(z); GGML_UNUSED(routing); GGML_UNUSED(activation_scales);
    NO_DEVICE_CODE;
#endif
}

template <int BN, int stages, int BK = 64, int G = 0>
static constexpr int f8_pp_shared() {
    return 4*stages*(128*(BK/4+4)+BN*((G ? BK/4 : BK/2)+4)+128+(G ? BN*((BK+G-1)/G) : 0));
}

template <typename layout, int BN, int NW, int stages, int BK, int G = 0>
static void launch_f8_pp(int n, int rows, int cols, int ldc, const char * x, const f8_activation<G> * y, float * z, cudaStream_t stream, const float * activation_scales = nullptr) {
    constexpr int bytes = f8_pp_shared<BN, stages, BK, G>();
    const bool tail = n%BK != 0 || rows%128 != 0 || uintptr_t(x)%16 != 0;
    auto kernel = tail ? mul_mat_f8_pp<layout, BN, NW, stages, true, BK, false, G> : mul_mat_f8_pp<layout, BN, NW, stages, false, BK, false, G>;
    if (tail) {
        CUDA_SET_SHARED_MEMORY_LIMIT(kernel, bytes);
    } else {
        CUDA_SET_SHARED_MEMORY_LIMIT(kernel, bytes);
    }
    kernel<<<dim3((rows+127)/128, (cols+BN-1)/BN), dim3(32, NW), bytes, stream>>>(n, rows, cols, ldc, x, y, z, {}, activation_scales);
}

template <typename layout>
static void launch_f8_pp(int cc, int n, int rows, int cols, int ldc, const char * x, const half * y, float * z, cudaStream_t stream) {
    constexpr bool blocked = std::is_same_v<layout, ggml_cuda_f8_block>;
    if (cc >= CC_ADA_LOVELACE && n%128 == 0) {
        if (cols <= 64) {
            return launch_f8_pp<layout, 32, 4, blocked ? 1 : 2, 128>(n, rows, cols, ldc, x, y, z, stream);
        }
        if (!blocked || cc < 1200) {
            return launch_f8_pp<layout, 64, 4, 1, 128>(n, rows, cols, ldc, x, y, z, stream);
        }
    }
    if (cc >= 1200) {
        if constexpr (blocked) {
            return launch_f8_pp<layout, 128, 8, 1, 128>(n, rows, cols, ldc, x, y, z, stream);
        }
        return launch_f8_pp<layout, 128, 8, 2, 64>(n, rows, cols, ldc, x, y, z, stream);
    }
    return launch_f8_pp<layout, 64, 4, 2, 64>(n, rows, cols, ldc, x, y, z, stream);
}

static __global__ void f8_moe_histogram(const char * ids, int32_t * counts, int32_t * ranks,
        int slots, int tokens, int experts, size_t nb0, size_t nb1) {
    __shared__ int32_t local_ids[256];
    const int tid = threadIdx.x, col = 256*blockIdx.x + tid;
    const int id = col < slots*tokens ? *(const int32_t *)(ids + size_t(col/slots)*nb1 + size_t(col%slots)*nb0) : -1;
    local_ids[tid] = id;
    for (int e = tid; e < experts; e += blockDim.x) counts[blockIdx.x*experts+e] = 0;
    __syncthreads();
    if (id < 0) return;
    int rank = 0;
    for (int j = 0; j < tid; ++j) rank += local_ids[j] == id;
    ranks[col] = rank;
    atomicAdd(counts + blockIdx.x*experts+id, 1);
}

static __global__ void f8_moe_prefix(int32_t * counts, int32_t * bounds, int32_t * tiles, int experts, int chunks, int w0, int w1, int w2) {
    extern __shared__ int32_t size[];
    const int width[3] = {w0, w1, w2};
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        int sum = 0;
        for (int c = 0; c < chunks; ++c) {
            const int count = counts[c*experts+e];
            counts[c*experts+e] = sum;
            sum += count;
        }
        size[e] = sum;
        for (int i = 0; i < 3 && width[i]; ++i) size[(i+1)*experts+e] = (sum+width[i]-1)/width[i];
    }
    __syncthreads();
    for (int e = threadIdx.x; e <= experts; e += blockDim.x) {
        int rows = 0, count[3] = {0, 0, 0};
        for (int j = 0; j < e; ++j) {
            rows += size[j];
            for (int i = 0; i < 3 && width[i]; ++i) count[i] += size[(i+1)*experts+j];
        }
        bounds[e] = rows;
        for (int i = 0; i < 3 && width[i]; ++i) tiles[i*(experts+1)+e] = count[i];
    }
}

static __global__ void f8_moe_scatter(const char * ids, const int32_t * counts, const int32_t * ranks,
        const int32_t * bounds, int32_t * rows, int slots, int tokens, int experts, size_t nb0, size_t nb1) {
    const int col = 256*blockIdx.x + threadIdx.x;
    if (col >= slots*tokens) return;
    const int id = *(const int32_t *)(ids + size_t(col/slots)*nb1 + size_t(col%slots)*nb0);
    rows[bounds[id]+counts[blockIdx.x*experts+id]+ranks[col]] = col;
}

template <int G = 0>
static __global__ void f8_moe_convert(const char * src, f8_activation<G> * dst, const int32_t * rows,
        int n, int slots, int channels, size_t nb1, size_t nb2, float * activation_scales = nullptr) {
    const int col = blockIdx.x;
    const int row = rows[col];
    const float * input = (const float *)(src + size_t(row/slots)*nb2 + size_t(row%slots%channels)*nb1);
    if constexpr (G != 0) {
        for (int g = threadIdx.x/32; g < (n+G-1)/G; g += blockDim.x/32) {
            f8_quantize_group<G>(input, dst + size_t(col)*n, activation_scales + size_t(col)*((n+G-1)/G), n, g);
        }
    } else {
        for (int j = threadIdx.x; j < n; j += blockDim.x) dst[size_t(col)*n+j] = __float2half(input[j]);
    }
}

static __global__ void f8_moe_sum(int used, int m, int n, const nv_bfloat16 * __restrict__ x, const float * __restrict__ w,
        const float * residual, float * dst) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const int col = i/m;
    x += col*used*m + i%m;
    w += col*used;
    float sum = 0;
    for (int j = 0; j < used; ++j) sum += __bfloat162float(x[j*m]) * w[j];
    dst[i] = residual ? sum + residual[i] : sum;
}

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM) && CUDART_VERSION >= 12000 && !defined(_WIN32)
#define F8_TMA_AVAILABLE
#include <cuda.h>

static bool f8_tma_available(int cc) {
    return cc >= 1200 && ggml_cuda_highest_compiled_arch(cc) >= 900;
}

static bool f8_tma_map(CUtensorMap & map, CUtensorMapDataType type, const void * data, uint64_t d0, uint64_t d1, uint64_t stride,
        uint32_t b0, uint32_t b1) {
    const cuuint64_t dim[2] = {d0, d1}, strides[1] = {stride};
    const cuuint32_t box[2] = {b0, b1}, step[2] = {1, 1};
    return cuTensorMapEncodeTiled(&map, type, 2, const_cast<void *>(data), dim, strides, box, step, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

template <int RN, bool UG, bool BLK = false>
static constexpr int f8_tma_shared(int groups) {
    return std::max(2*((UG ? 2 : 1)*(BLK ? 144 : 128)+16*RN)*128 + 16*RN*groups*4 + 24, UG ? 16*RN*128*4 : 0);
}

template <int RN, bool UG, bool BLK = false, bool ADD = false>
static __global__ void __launch_bounds__(256, 1) mul_mat_f8_tma(int K, int M, int expert_rows,
        const char * __restrict__ x, const char * __restrict__ xg, float * __restrict__ z, nv_bfloat16 * __restrict__ zb,
        uint8_t * __restrict__ codes, float * __restrict__ out_scales, int out_stride, f8_moe_routing routing, const char * bias_gate,
        const __grid_constant__ CUtensorMap ma, const __grid_constant__ CUtensorMap mg, const __grid_constant__ CUtensorMap mb,
        const __grid_constant__ CUtensorMap ms, int SG, int CG, int cols, const float * other) {
#if __CUDA_ARCH__ >= 900
    constexpr int RM = 2, WMB = 4, WNB = 2, BM = WMB*16*RM, BN = WNB*8*RN, BK = 128, NA = UG ? 2 : 1, NW = WMB*WNB, PA = BLK ? 144 : BK;
    const int J = BLK ? (routing.row_stride%16 == 0 ? 1 : routing.row_stride%8 == 0 ? 2 : 4) : 1, JR = BM/J;
    GGML_UNUSED(J);
    GGML_UNUSED(JR);
    int bx = blockIdx.x, by = blockIdx.y;
    if constexpr (BLK) {
        if (CG > 0) {
            bx = blockIdx.y%(M/BM);
            by = blockIdx.y/(M/BM)*CG + blockIdx.x;
        }
    } else {
        GGML_UNUSED(CG);
    }
    int expert = 0, first_col = by*BN, ncols = cols-first_col;
    if (routing.tiles) {
        if (by >= routing.tiles[routing.experts]) return;
        int hi = routing.experts;
        while (expert + 1 < hi) {
            const int mid = (expert + hi)/2;
            if (routing.tiles[mid] <= by) expert = mid;
            else hi = mid;
        }
        first_col = routing.bounds[expert] + (by-routing.tiles[expert])*BN;
        ncols = routing.bounds[expert+1]-first_col;
    } else if (ncols <= 0) return;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t * sA = smem, * sB = smem + 2*NA*BM*PA;
    float * sS = (float *)(sB + 2*BN*BK);
    const unsigned bar0 = (unsigned)__cvta_generic_to_shared(sS + BN*SG);
    const int row0 = bx*BM, tid = threadIdx.x, warp = tid/32, lane = tid%32;
    const int wm = warp%WMB, wn = warp/WMB, g = lane>>2, t4 = lane&3;
    if (tid == 0) {
#pragma unroll
        for (int i = 0; i < 3; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar0+8*i));
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    auto tile = [&](const CUtensorMap * map, void * dst, unsigned bar, int c0, int c1) {
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
            :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
    };
    auto load = [&](int k0, int s) {
        if (k0 < K && tid == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar0+8*s), "r"(NA*BM*PA+BN*BK));
            if constexpr (BLK) {
                for (int j = 0; j < J; ++j) {
                    const int origin = j*int(routing.row_stride) + k0/BK*132;
                    tile(&ma, sA + (s*NA*BM + j*JR)*PA, bar0+8*s, origin - origin%16, (expert*expert_rows+row0)/J);
                    if constexpr (UG) tile(&mg, sA + ((s*NA+1)*BM + j*JR)*PA, bar0+8*s, origin - origin%16, (expert*expert_rows+row0)/J);
                }
            } else {
                tile(&ma, sA + s*NA*BM*PA, bar0+8*s, k0, expert*expert_rows+row0);
                if constexpr (UG) tile(&mg, sA + (s*NA+1)*BM*PA, bar0+8*s, k0, expert*expert_rows+row0);
            }
            tile(&mb, sB + s*BN*BK, bar0+8*s, k0, first_col);
        }
    };
    auto wait = [&](int s, int parity) {
        unsigned done = 0;
        while (!done) {
            asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n }"
                : "=r"(done) : "r"(bar0+8*s), "r"(parity) : "memory");
        }
    };
    if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar0+16), "r"(BN*SG*4));
        tile(&ms, sS, bar0+16, 0, first_col);
    }
    load(0, 0);
    float res[ADD ? RM : 1][ADD ? RN : 1][4];
    GGML_UNUSED(res);
    if constexpr (ADD) {
#pragma unroll
        for (int i = 0; i < RM; ++i) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int rl = wm*16*RM + 16*i + g + 8*h, rp = BLK ? rl%JR*J + rl/JR : rl;
#pragma unroll
                for (int j = 0; j < RN; ++j) {
#pragma unroll
                    for (int l = 0; l < 2; ++l) {
                        const int c = wn*8*RN + 8*j + 2*t4 + l;
                        res[i][j][2*h+l] = c < ncols ? other[size_t(routing.rows ? routing.rows[first_col+c] : first_col+c)*M + row0 + rp] : 0.0f;
                    }
                }
            }
        }
    }
    wait(2, 0);
    float total[NA][RM][RN][4];
#pragma unroll
    for (int t = 0; t < NA; ++t) {
#pragma unroll
        for (int i = 0; i < RM; ++i) {
#pragma unroll
            for (int j = 0; j < RN; ++j) {
#pragma unroll
                for (int l = 0; l < 4; ++l) total[t][i][j][l] = 0;
            }
        }
    }
    for (int k0 = 0, s = 0; k0 < K; k0 += BK, s ^= 1) {
        load(k0 + BK, s^1);
        wait(s, (k0/BK/2)&1);
        const uint8_t * pb = sB + (s*BN + wn*8*RN)*BK + 32*t4;
        uint4 a[NA][RM][2][2];
        float bw[NA][RM][2];
        GGML_UNUSED(bw);
#pragma unroll
        for (int t = 0; t < NA; ++t) {
            if constexpr (BLK) {
                const uint8_t * pa = sA + ((s*NA+t)*BM + wm*16*RM)*PA;
                const int ph = (wm*16*RM/JR*int(routing.row_stride) + k0/BK*132)%16/4;
#pragma unroll
                for (int i = 0; i < RM; ++i) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const uint8_t * row = pa + (16*i+g+8*h)*PA;
                        const uint4 * r = (const uint4 *)row + 2*t4 + (ph == 3);
                        const uint4 v0 = r[0], v1 = r[1], v2 = r[ph == 3 ? 1 : 2];
                        bw[t][i][h] = *(const float *)(row + 4*ph);
                        if (ph == 0) {
                            a[t][i][h][0] = make_uint4(v0.y, v0.z, v0.w, v1.x);
                            a[t][i][h][1] = make_uint4(v1.y, v1.z, v1.w, v2.x);
                        } else if (ph == 1) {
                            a[t][i][h][0] = make_uint4(v0.z, v0.w, v1.x, v1.y);
                            a[t][i][h][1] = make_uint4(v1.z, v1.w, v2.x, v2.y);
                        } else if (ph == 2) {
                            a[t][i][h][0] = make_uint4(v0.w, v1.x, v1.y, v1.z);
                            a[t][i][h][1] = make_uint4(v1.w, v2.x, v2.y, v2.z);
                        } else {
                            a[t][i][h][0] = v0;
                            a[t][i][h][1] = v1;
                        }
                    }
                }
            } else {
                const uint8_t * pa = sA + ((s*NA+t)*BM + wm*16*RM)*BK + 32*t4;
#pragma unroll
                for (int i = 0; i < RM; ++i) {
                    const uint4 * r0 = (const uint4 *)(pa + (16*i+g)*BK), * r1 = (const uint4 *)(pa + (16*i+g+8)*BK);
                    a[t][i][0][0] = r0[0]; a[t][i][0][1] = r0[1]; a[t][i][1][0] = r1[0]; a[t][i][1][1] = r1[1];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < RN; ++j) {
            if (wn*8*RN + 8*j >= ncols) continue;
            const uint4 * c = (const uint4 *)(pb + (8*j+g)*BK);
            const uint4 b[2] = {c[0], c[1]};
            const float e0 = sS[(wn*8*RN + 8*j + 2*t4)*SG + k0/BK], e1 = sS[(wn*8*RN + 8*j + 2*t4 + 1)*SG + k0/BK];
#pragma unroll
            for (int t = 0; t < NA; ++t) {
#pragma unroll
                for (int i = 0; i < RM; ++i) {
                    float acc[4] = {0, 0, 0, 0};
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const uint4 bv = b[q/2], x0 = a[t][i][0][q/2], x1 = a[t][i][1][q/2];
                        const uint32_t b0 = q%2 ? bv.z : bv.x, b1 = q%2 ? bv.w : bv.y;
                        const uint32_t a0 = q%2 ? x0.z : x0.x, a2 = q%2 ? x0.w : x0.y, a1 = q%2 ? x1.z : x1.x, a3 = q%2 ? x1.w : x1.y;
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                    }
                    if constexpr (BLK) {
                        total[t][i][j][0] = fmaf(acc[0], e0*bw[t][i][0], total[t][i][j][0]);
                        total[t][i][j][1] = fmaf(acc[1], e1*bw[t][i][0], total[t][i][j][1]);
                        total[t][i][j][2] = fmaf(acc[2], e0*bw[t][i][1], total[t][i][j][2]);
                        total[t][i][j][3] = fmaf(acc[3], e1*bw[t][i][1], total[t][i][j][3]);
                    } else {
                        total[t][i][j][0] = fmaf(acc[0], e0, total[t][i][j][0]);
                        total[t][i][j][1] = fmaf(acc[1], e1, total[t][i][j][1]);
                        total[t][i][j][2] = fmaf(acc[2], e0, total[t][i][j][2]);
                        total[t][i][j][3] = fmaf(acc[3], e1, total[t][i][j][3]);
                    }
                }
            }
        }
        __syncthreads();
    }
    float * hs = (float *)smem;
#pragma unroll
    for (int i = 0; i < RM; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rl = wm*16*RM + 16*i + g + 8*h, rp = BLK ? rl%JR*J + rl/JR : rl, r = row0 + rp;
            const size_t offset = size_t(expert)*routing.expert_stride + size_t(r)*routing.row_stride;
            const float scale = BLK ? 1.0f : ggml_cuda_f8_row::scale(x + offset);
            const float bias = routing.bias ? ((const float *)(routing.bias + size_t(expert)*routing.bias_stride))[r] : 0.0f;
            float scale_g = 0, bias_g = 0;
            if constexpr (UG) {
                scale_g = BLK ? 1.0f : ggml_cuda_f8_row::scale(xg + offset);
                if (bias_gate) bias_g = ((const float *)(bias_gate + size_t(expert)*routing.bias_stride))[r];
            }
#pragma unroll
            for (int j = 0; j < RN; ++j) {
#pragma unroll
                for (int l = 0; l < 2; ++l) {
                    const int c = wn*8*RN + 8*j + 2*t4 + l;
                    if (c < ncols) {
                        float value = total[0][i][j][2*h+l]*scale;
                        if (routing.bias) value += bias;
                        if constexpr (UG) {
                            float gate = total[NA-1][i][j][2*h+l]*scale_g;
                            if (bias_gate) gate += bias_g;
                            hs[c*BM + rp] = gate * value / (1.0f + expf(-gate));
                        } else {
                            const size_t o = size_t(routing.rows ? routing.rows[first_col+c] : first_col+c)*M + r;
                            if constexpr (ADD) value = __fadd_rn(value, res[i][j][2*h+l]);
                            if (zb) zb[o] = __float2bfloat16(value);
                            else z[o] = value;
                        }
                    }
                }
            }
        }
    }
    if constexpr (UG) {
        __syncthreads();
        const int group = bx;
        for (int c = warp; c < BN && c < ncols; c += NW) {
            f8_quantize_group<128>(hs + c*BM - group*128, codes + size_t(first_col+c)*M,
                    out_scales + size_t(first_col+c)*out_stride, M, group);
        }
    }
#else
    GGML_UNUSED(K); GGML_UNUSED(M); GGML_UNUSED(expert_rows); GGML_UNUSED(x); GGML_UNUSED(xg); GGML_UNUSED(z); GGML_UNUSED(zb); GGML_UNUSED(codes);
    GGML_UNUSED(out_scales); GGML_UNUSED(out_stride); GGML_UNUSED(routing); GGML_UNUSED(bias_gate); GGML_UNUSED(SG); GGML_UNUSED(CG); GGML_UNUSED(cols); GGML_UNUSED(other);
    NO_DEVICE_CODE;
#endif
}

template <int RN, bool UG, bool BLK = false, bool ADD = false>
static bool launch_f8_tma(int K, int M, int cols, const char * x, const char * xg, const uint8_t * y, const float * scales, int SG,
        float * z, nv_bfloat16 * zb, uint8_t * codes, float * out_scales, int out_stride, const f8_moe_routing & routing,
        const char * bias_gate, cudaStream_t stream, int cg = 0, const float * other = nullptr) {
    constexpr int BN = 16*RN;
    const int expert_rows = routing.expert_stride/routing.row_stride;
    alignas(64) CUtensorMap ma, mg{}, mb, ms;
    const int J = BLK ? (routing.row_stride%16 == 0 ? 1 : routing.row_stride%8 == 0 ? 2 : 4) : 1;
    auto weights = [&](CUtensorMap & map, const char * w) {
        const uint64_t rows = uint64_t(routing.experts)*expert_rows/J;
        return BLK ? f8_tma_map(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, w, J*routing.row_stride, rows, J*routing.row_stride, 144, 128/J) :
            f8_tma_map(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, w + ggml_cuda_f8_row::row_meta_size, K, rows, routing.row_stride, 128, 128);
    };
    if (!weights(ma, x) || (UG && !weights(mg, xg)) ||
        !f8_tma_map(mb, CU_TENSOR_MAP_DATA_TYPE_UINT8, y, K, cols, K, 128, BN) ||
        !f8_tma_map(ms, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, scales, SG, cols, SG*4, SG, BN)) return false;
    const int bytes = f8_tma_shared<RN, UG, BLK>(SG);
    static int limit[GGML_CUDA_MAX_DEVICES] = {};
    const int id = ggml_cuda_get_device();
    if (bytes > limit[id]) {
        CUDA_CHECK(cudaFuncSetAttribute((mul_mat_f8_tma<RN, UG, BLK, ADD>), cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        limit[id] = bytes;
    }
    mul_mat_f8_tma<RN, UG, BLK, ADD><<<cg > 0 ? dim3(cg, (M/128)*(((cols+BN-1)/BN+cg-1)/cg)) : dim3(M/128, (cols+BN-1)/BN+routing.experts), 256, bytes, stream>>>(K, M, expert_rows, x, xg, z, zb,
            codes, out_scales, out_stride, routing, bias_gate, ma, mg, mb, ms, SG, cg, cols, other);
    return true;
}
#endif

template <typename layout, int BN, int NW, int stages, int BK, int G = 0>
static void launch_f8_moe(int n, int rows, int cols, const char * x, const f8_activation<G> * y, float * z,
        const f8_moe_routing & routing, cudaStream_t stream, const float * activation_scales = nullptr) {
    constexpr int limit = std::max(f8_pp_shared<BN, stages, BK, G>(), G != 0 ? BN*128*4 : 0);
    const int bytes = routing.codes ? limit : f8_pp_shared<BN, stages, BK, G>();
    const bool tail = n%BK != 0 || rows%128 != 0 || uintptr_t(x)%16 != 0 || routing.expert_stride%16 != 0;
    auto kernel = tail ? mul_mat_f8_pp<layout, BN, NW, stages, true, BK, true, G> : mul_mat_f8_pp<layout, BN, NW, stages, false, BK, true, G>;
    if (tail) {
        CUDA_SET_SHARED_MEMORY_LIMIT(kernel, limit);
    } else {
        CUDA_SET_SHARED_MEMORY_LIMIT(kernel, limit);
    }
    kernel<<<dim3((rows+127)/128, (cols+BN-1)/BN+routing.experts), dim3(32,NW), bytes, stream>>>(
            n, rows, cols, rows, x, y, z, routing, activation_scales);
}
