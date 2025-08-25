#include "../mmq_id_common.cuh"

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq6_k(
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

    auto values = iq6nl_values;
    int qh[2];

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_iq6_k * bxi = (const block_iq6_k *)(x + i*stride) + kbx0;

        const float d = bxi->d;
        uint16_t extra = bxi->extra >> (kqsx/4);

        qh[0] = get_int_b4(bxi->qh, kqsx+0);
        qh[1] = get_int_b4(bxi->qh, kqsx+8);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh[l/2] & 0x03030303) << 4) | ((extra & 1) * 0x40404040);
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh[l/2] & 0x0c0c0c0c) << 2) | ((extra & 4) * 0x10101010);
            qh[l/2] >>= 4;
            extra   >>= 4;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q3_K + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16*l + 8] = *(const int *)&val1;
#endif // INT8_MMA_AVAILABLE
        }


#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = d * bxi->scales[2*kqsx+0];
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = d * bxi->scales[2*kqsx+1];
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * bxi->scales[2*kqsx+0];
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * bxi->scales[2*kqsx+1];
#endif // INT8_MMA_AVAILABLE
    }
}

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits_id<mmq_x, mmq_y, need_check, GGML_TYPE_IQ6_K> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq6_k<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y>;
};

DECL_MMQ_CASE(GGML_TYPE_IQ6_K);
