#include "../mmq.cuh"

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_kss(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x / 4;

    uint32_t aux32[2];
    auto a8 = (const uint8_t *)aux32;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const block_iq4_kss * bxi = (const block_iq4_kss *)(dptr + 1) + kbx0;
        const uint32_t * q4 = bxi->qs + 4*kqsx;
        uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
        uint8_t ls = (s32 | (s32 >> 15)) & 0xff;

        auto values = iq4k_table + ((ls & 1) << 8);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t val = q4[j] & 0xfffefffe;
            val = val ^ (val >> 1);
            aux32[0] = (val >> 0) & 0x0f0f0f0f;
            aux32[1] = (val >> 4) & 0x0f0f0f0f;
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + j + 0] = int_from_table_x(a8+0, values);
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + j + 4] = int_from_table_x(a8+4, values);
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + j + 0] = int_from_table_x(a8+0, values);
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + j + 4] = int_from_table_x(a8+4, values);
#endif // INT8_MMA_AVAILABLE
        }
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = dptr[0] * ((ls & 254) - 127);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = dptr[0] * ((ls & 254) - 127);
#endif // INT8_MMA_AVAILABLE
    }

}


template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ4_KSS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_kss<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

DECL_MMQ_CASE(GGML_TYPE_IQ4_KSS);

