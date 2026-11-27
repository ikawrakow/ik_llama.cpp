#include "../mmq.cuh"
#include "../nt-common.cuh"

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_nt(
    const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride, int blocks_per_ne00) {

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif

    const int kqsx = threadIdx.x / 4;
    const int hdr = nt_hdr(128*blocks_per_ne00);
    const int chunk = kqsx/4, w0 = 4*(kqsx%4);

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (need_check) {
            i = min(i, i_max);
        }

        const char * cx = x + i*stride;
        const uint32_t * q32 = (const uint32_t *)(cx + hdr + 64*kbx0) + 16*chunk;
        uint32_t prev = q32[w0 - 1];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t cur = q32[w0 + m];
            uint32_t lo, hi;
            nt_labels8(prev, cur, lo, hi);
            prev = cur;
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 32*chunk + w0 + m +  0] = (int)lo;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 32*chunk + w0 + m + 16] = (int)hi;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 32*chunk + w0 + m +  0] = (int)lo;
            x_qs[i*(2*WARP_SIZE + 1)     + 32*chunk + w0 + m + 16] = (int)hi;
#endif
        }
        const float s = nt_block_scale(*(const float *)cx, ((const uint8_t *)cx)[4 + kbx0 + chunk]);
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = s;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = s;
#endif
    }
}

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, GGML_TYPE_IQ4_NT> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_nt<mmq_y, nwarps, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y, nwarps>;
};

DECL_MMQ_CASE(GGML_TYPE_IQ4_NT);
