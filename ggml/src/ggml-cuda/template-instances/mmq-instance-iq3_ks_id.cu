#include "../mmq_id_common.cuh"

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *)(x + i*stride);
        const float d = __half2float(dptr[0]);
        const block_iq3_ks * bxi = (const block_iq3_ks *)(dptr + 1) + kbx0;

        //uint16_t extra = bxi->extra >> 8;
        int qh = get_int_b2(bxi->qh, kqsx);

        uint32_t extra32 = uint32_t(bxi->extra >> 8) * 0x01010101;

        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            const int ql = get_int_b2(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((qh << 2) & 0x04040404) | ((extra32 << 3) & 0x08080808)
                                                     | ((qh << 4) & 0x40404040) | ((extra32 << 5) & 0x80808080);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((qh << 1) & 0x04040404) | ((extra32 << 2) & 0x08080808)
                                                     | ((qh << 3) & 0x40404040) | ((extra32 << 4) & 0x80808080);
            int2 v1 = get_int_from_table_16(val1, iq3nl_values);
            int2 v2 = get_int_from_table_16(val2, iq3nl_values);

            extra32 >>= 4;
            qh      >>= 4;

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32*l +  0] = v1.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32*l +  8] = v2.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32*l + 16] = v1.y;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 24] = v2.y;
#endif // INT8_MMA_AVAILABLE
        }

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = d * (int(((bxi->scales[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->extra >> kqsx) & 1) << 4)) - 16);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = d * (int(((bxi->scales[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->extra >> kqsx) & 1) << 4)) - 16);
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits_id<mmq_x, mmq_y, need_check, GGML_TYPE_IQ3_KS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_ks<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

DECL_MMQ_CASE(GGML_TYPE_IQ3_KS);
