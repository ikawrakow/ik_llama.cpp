#include "../mmq.cuh"

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_kl(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x/4;

    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (need_check) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *)(x + i*stride);
        const float d = *dptr;
        const block_iq2_kl * bxi = (const block_iq2_kl *)(dptr + 1) + kbx0;

        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            auto ql = get_int_b2(bxi->qs, 4*(kqsx/2) + 2*(kqsx%2) + j);
            auto qh = get_int_b2(bxi->qh, 2*(kqsx%2) + j) >> 2*(kqsx/2);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh << 4) & 0x10101010);
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh << 3) & 0x10101010);
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int val1 = iq2kl_values[a8[2*l+0]] | (iq2kl_values[a8[2*l+1]] << 16);
                int val2 = iq2kl_values[a8[2*l+4]] | (iq2kl_values[a8[2*l+5]] << 16);
#ifdef INT8_MMA_AVAILABLE
                x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 16*(kqsx/2) + 4*(kqsx%2) + 2*j + l + 0] = val1;
                x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 16*(kqsx/2) + 4*(kqsx%2) + 2*j + l + 8] = val2;
#else
                x_qs[i*(2*WARP_SIZE + 1)     + 16*(kqsx/2) + 4*(kqsx%2) + 2*j + l + 0] = val1;
                x_qs[i*(2*WARP_SIZE + 1)     + 16*(kqsx/2) + 4*(kqsx%2) + 2*j + l + 8] = val2;
#endif
            }
        }

        int ls = int(((bxi->scales_l[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->scales_h >> 2*kqsx) & 3) << 4)) - 32;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = d * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = d * ls;
#endif
    }

}

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ2_KL> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq2_kl<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

DECL_MMQ_CASE(GGML_TYPE_IQ2_KL);
