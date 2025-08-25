#include "../mmq_id_common.cuh"

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_k(
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

        const block_iq3_k * bxi = (const block_iq3_k *)(x + i*stride) + kbx0;

        const float d = bxi->d;

        uint16_t extra = bxi->extra >> (kqsx/4);
        uint32_t extra32[2] = { uint32_t(extra & 0xff) * 0x01010101, uint32_t(extra >> 8) * 0x01010101 };
        int qh = get_int_b2(bxi->qh, kqsx);

    #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            //extra << 3, extra << 1, extra >> 1, extra >> 3
            const int ql = get_int_b2(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((extra32[l] << 3) & 0x88888888)
                          | ((qh << 2) & 0x04040404) | ((qh << 4) & 0x40404040);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((extra32[l] << 1) & 0x88888888)
                          | ((qh << 1) & 0x04040404) | ((qh << 3) & 0x40404040);
            int2 v1 = get_int_from_table_16(val1, iq3nl_values);
            int2 v2 = get_int_from_table_16(val2, iq3nl_values);

            qh >>= 4;

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  0] = v1.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l +  8] = v2.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 16] = v1.y;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32*l + 24] = v2.y;
#endif // INT8_MMA_AVAILABLE
        }

        uint16_t sh = bxi->scales_h >> 2*kqsx;
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = d * ((2*(bxi->scales_l[kqsx] & 0xf) + 1) * (sh & 1 ? -1 : 1));
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = d * ((2*(bxi->scales_l[kqsx] >>  4) + 1) * (sh & 2 ? -1 : 1));
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ((2*(bxi->scales_l[kqsx] & 0xf) + 1) * (sh & 1 ? -1 : 1));
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ((2*(bxi->scales_l[kqsx] >>  4) + 1) * (sh & 2 ? -1 : 1));
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_k_r4(
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

    const int kqsx = threadIdx.x/4;  // 0...7 -> block of 32

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (need_check) {
            i = min(i, i_max);
        }
        int i4 = i/4;
        int ir = i%4;

        const block_iq3_k_r4 * bxi = (const block_iq3_k_r4 *)(x + 4*i4*stride) + kbx0;

        const float d = __half2float(bxi->d[ir]);

        int qh = get_int_b4(bxi->qh, 4*kqsx+ir);

    #pragma unroll
        for (int l = 0; l < 2; ++l) {

            //auto values_l = iq3k_table + (((bxi->extra[ir+4*l] >> kqsx) & 1) << 6);
            uint32_t extra32 = uint32_t((bxi->extra[ir+4*l] >> kqsx) & 1) * 0x88888888;

            const int ql = get_int_b4(bxi->qs, 8*kqsx + ir + 4*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | extra32 | ((qh << 2) & 0x04040404) | ((qh << 4) & 0x40404040);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | extra32 | ((qh << 1) & 0x04040404) | ((qh << 3) & 0x40404040);
            int2 v1 = get_int_from_table_16(val1, iq3nl_values);
            int2 v2 = get_int_from_table_16(val2, iq3nl_values);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + 4*l + 0] = v1.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + 4*l + 1] = v2.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + 4*l + 2] = v1.y;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + 8*kqsx + 4*l + 3] = v2.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + 4*l + 0] = v1.x;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + 4*l + 1] = v2.x;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + 4*l + 2] = v1.y;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + 4*l + 3] = v2.y;
#endif // INT8_MMA_AVAILABLE

            qh >>= 4;
        }

        int is = 8*kqsx + ir;
        float dl1 = d * (2*((bxi->scales_l[is%32] >> 4*(is/32)) & 0xf) + 1) * ((bxi->scales_h[is%8] >> (is/8)) & 1 ? -1 : 1);
        is += 4;
        float dl2 = d * (2*((bxi->scales_l[is%32] >> 4*(is/32)) & 0xf) + 1) * ((bxi->scales_h[is%8] >> (is/8)) & 1 ? -1 : 1);

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = dl1;
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = dl2;
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = dl1;
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = dl2;
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits_id<mmq_x, mmq_y, need_check, GGML_TYPE_IQ3_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_k<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits_id<mmq_x, mmq_y, need_check, GGML_TYPE_IQ3_K_R4> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq3_k_r4<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y>;
};


DECL_MMQ_CASE(GGML_TYPE_IQ3_K);
DECL_MMQ_CASE(GGML_TYPE_IQ3_K_R4);
