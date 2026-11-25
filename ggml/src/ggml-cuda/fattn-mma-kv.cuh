#pragma once

#include "cp-async.cuh"
#include "mma_new.cuh"

typedef ggml_cuda_mma::tile<16, 8, half2> tile_A;

template<int D, ggml_type type_KV>
constexpr int fattn_mma_raw_row_bytes = type_KV == GGML_TYPE_Q8_0 ? D/QK8_0 * sizeof(block_q8_0) :
                                        type_KV == GGML_TYPE_Q4_0 ? D/QK4_0 * sizeof(block_q4_0) : 0;

#ifdef CP_ASYNC_AVAILABLE
static __device__ __forceinline__ uint4 fattn_mma_expand_q4_0(const char * row, int k) {
    uint4 dst;
    half2 * h = (half2 *) &dst;
    const block_q4_0 & b = ((const block_q4_0 *) row)[k/QK4_0];
    const float d = __half2float(b.d);
    const float dm = -8*d;
    const int shift = 4*((k % QK4_0)/(QK4_0/2));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint16_t q = ((const uint16_t *) b.qs)[(k % (QK4_0/2))/2 + j] >> shift;
        h[j] = make_half2(d*(q & 15) + dm, d*((q >> 8) & 15) + dm);
    }
    return dst;
}
#endif // CP_ASYNC_AVAILABLE

template<int D, ggml_type type_KV>
static __device__ __forceinline__ void flash_attn_ext_f16_stage_inplace(
        const void * kv, int row0, int stride, half2 * tile) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr int rows = tile_A::I;
    constexpr int row_bytes = fattn_mma_raw_row_bytes<D, type_KV>;
    constexpr int tile_stride = D/2 + 4;
    half2 * warp_tile = tile + threadIdx.y*rows*tile_stride;
    char * raw = (char *)(warp_tile + rows*tile_stride) - rows*row_bytes;
    const char * src = (const char *)kv + int64_t(row0 + threadIdx.y*rows)*stride;
    const unsigned int dst = __cvta_generic_to_shared(raw);
    if constexpr (row_bytes % 16 != 0 && row_bytes % 8 == 0) {
        constexpr int units = row_bytes/8;
#pragma unroll
        for (int i = threadIdx.x; i < rows*units; i += WARP_SIZE) {
            cp_async_ca_8(dst + 8*i, src + int64_t(i/units)*stride + 8*(i%units));
        }
    } else if constexpr (row_bytes % 16 != 0) {
#pragma unroll
        for (int i = threadIdx.x; i < rows*row_bytes/16; i += WARP_SIZE) {
            cp_async_cg_16<0>(dst + 16*i, src + 16*i);
        }
    } else {
        constexpr int units = row_bytes/16;
#pragma unroll
        for (int i = threadIdx.x; i < rows*units; i += WARP_SIZE) {
            cp_async_cg_16<0>(dst + 16*i, src + int64_t(i/units)*stride + 16*(i%units));
        }
    }
#else
    GGML_UNUSED(kv); GGML_UNUSED(row0); GGML_UNUSED(stride); GGML_UNUSED(tile);
    NO_DEVICE_CODE;
#endif
}

template<int D, ggml_type type_KV>
static __device__ __forceinline__ void flash_attn_ext_f16_expand_inplace(
        half2 * tile) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr int rows = tile_A::I;
    constexpr int row_bytes = fattn_mma_raw_row_bytes<D, type_KV>;
    constexpr int tile_stride = D/2 + 4;
    half2 * warp_tile = tile + threadIdx.y*rows*tile_stride;
    char * raw = (char *)(warp_tile + rows*tile_stride) - rows*row_bytes;
    constexpr int chunks = (D+127)/128;
#pragma unroll 1
    for (int i = 0; i < rows; i += 2) {
        const int r = i + threadIdx.x/16;
        const char * row = raw + r*row_bytes;
        uint4 expanded[chunks];
#pragma unroll
        for (int t = 0; t < chunks; ++t) {
            const int k = t*128 + 8*(threadIdx.x%16);
            if (k >= D) continue;
            if constexpr (type_KV == GGML_TYPE_Q8_0) {
                const block_q8_0 * blocks = (const block_q8_0 *) row;
                const int lane = (k%64)/8;
                const uint32_t * words = (const uint32_t *)(blocks + 2*(k/64));
                const uint32_t mid = words[2*lane+1];
                const uint32_t end = words[2*lane+2];
                const uint32_t lo = (lane < 4 ? __funnelshift_r(words[2*lane], mid, 16) : mid) ^ 0x80808080u;
                const uint32_t hi = (lane < 4 ? __funnelshift_r(mid, end, 16) : end) ^ 0x80808080u;
                const half2 d = __half2half2(blocks[k/QK8_0].d);
                half2 * h = (half2 *)&expanded[t];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t v = __byte_perm(j < 2 ? lo : hi, 0x64646464u, j%2 ? 0x4342 : 0x4140);
                    h[j] = __hmul2(__hsub2(*(const half2 *)&v, make_half2(1152, 1152)), d);
                }
            } else {
                expanded[t] = fattn_mma_expand_q4_0(row, k);
            }
        }
        __syncwarp();
#pragma unroll
        for (int t = 0; t < chunks; ++t) {
            const int k = t*128 + 8*(threadIdx.x%16);
            if (k < D) *(uint4 *)(warp_tile + r*tile_stride + k/2) = expanded[t];
        }
    }
    __syncwarp();
#else
    GGML_UNUSED(tile);
    NO_DEVICE_CODE;
#endif
}

template<int D, ggml_type type_KV>
static __device__ __forceinline__ void flash_attn_ext_f16_load_inplace(
        const void * kv, int row0, int stride, half2 * tile) {
    flash_attn_ext_f16_stage_inplace<D, type_KV>(kv, row0, stride, tile);
    cp_async_wait_all();
    ggml_cuda_syncwarp();
    flash_attn_ext_f16_expand_inplace<D, type_KV>(tile);
}
