#include "../mmq_id_common.cuh"

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x%16;

#ifdef __CUDA_ARCH__
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 2*nwarps) {
        int i = i0 + 2*threadIdx.y + threadIdx.x/16;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_ks * bxi = (const block_iq2_ks *)(x + i*stride + sizeof(half)) + kbx0;

        uint16_t extra = bxi->extra >> 4*(kqsx/8);
        int q2 = get_int_b2(bxi->qs, kqsx);

        uint32_t extra32 = uint32_t(extra & 0xf) * 0x01010101;
        uint32_t val1 = ((q2 >> 0) & 0x33333333) | ((extra32 << 2) & 0x04040404) | ((extra32 << 4) & 0x40404040);
        uint32_t val2 = ((q2 >> 2) & 0x33333333) | ((extra32 << 1) & 0x04040404) | ((extra32 << 3) & 0x40404040);
        int2 v1 = get_int_from_table_8(val1, iq2nl_values);
        int2 v2 = get_int_from_table_8(val2, iq2nl_values);

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) +  0] = v1.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) +  8] = v2.x;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) + 16] = v1.y;
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) + 24] = v2.y;
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) +  0] = v1.x;
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) +  8] = v2.x;
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) + 16] = v1.y;
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) + 24] = v2.y;
#endif // INT8_MMA_AVAILABLE
    }

#else // __CUDA_ARCH__


    const int * all_values = (const int *)iq2k_table;
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 2*nwarps) {
        int i = i0 + 2*threadIdx.y + threadIdx.x/16;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq2_ks * bxi = (const block_iq2_ks *)(x + i*stride + sizeof(half)) + kbx0;

        uint16_t extra = bxi->extra >> 4*(kqsx/8);
        int q2 = get_int_b2(bxi->qs, kqsx);

#ifdef INT8_MMA_AVAILABLE
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) +  0] = int_from_table_4((q2 >> 0) & 0x03030303, all_values + ((extra & 1) << 8));
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) +  8] = int_from_table_4((q2 >> 2) & 0x03030303, all_values + ((extra & 2) << 7));
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) + 16] = int_from_table_4((q2 >> 4) & 0x03030303, all_values + ((extra & 4) << 6));
        x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx%8 + 32*(kqsx/8) + 24] = int_from_table_4((q2 >> 6) & 0x03030303, all_values + ((extra & 8) << 5));
#else
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) +  0] = int_from_table_4((q2 >> 0) & 0x03030303, all_values + ((extra & 1) << 8));
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) +  8] = int_from_table_4((q2 >> 2) & 0x03030303, all_values + ((extra & 2) << 7));
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) + 16] = int_from_table_4((q2 >> 4) & 0x03030303, all_values + ((extra & 4) << 6));
        x_qs[i*(2*WARP_SIZE + 1)     + kqsx%8 + 32*(kqsx/8) + 24] = int_from_table_4((q2 >> 6) & 0x03030303, all_values + ((extra & 8) << 5));
#endif // INT8_MMA_AVAILABLE
    }
#endif // __CUDA_ARCH__

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = i0 + threadIdx.y * 8 + threadIdx.x / 4;

        if (need_check) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *)(x + i*stride);
        const float d = dptr[0];
        const block_iq2_ks * bxi = (const block_iq2_ks *)(dptr + 1) + kbx0;
        const int ls1 = ((bxi->scales[threadIdx.x % 4] >> 0) & 0xf) | ((bxi->extra >> (4 + 2*(threadIdx.x % 4))) & 0x10);
        const int ls2 = ((bxi->scales[threadIdx.x % 4] >> 4) & 0xf) | ((bxi->extra >> (5 + 2*(threadIdx.x % 4))) & 0x10);

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + 2*(threadIdx.x % 4) + 0] = d * (ls1 - 16);
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + 2*(threadIdx.x % 4) + 1] = d * (ls2 - 16);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + 2*(threadIdx.x % 4) + 0] = d * (ls1 - 16);
        x_df[i*(WARP_SIZE/4) + i/4   + 2*(threadIdx.x % 4) + 1] = d * (ls2 - 16);
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits_id<mmq_x, mmq_y, need_check, GGML_TYPE_IQ2_KS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_ks<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

DECL_MMQ_CASE(GGML_TYPE_IQ2_KS);
