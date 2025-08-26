#include "../mmq_id_common.cuh"

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_kss(
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

    const int kqsx = threadIdx.x / 4;

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

        auto values = iq4k_values + ((ls & 1) << 4);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t val = q4[j] & 0xfffefffe;
            val = val ^ (val >> 1);
            auto v = get_int_from_table_16(val, values);
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + j + 0] = v.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + j + 4] = v.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + j + 0] = v.x;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + j + 4] = v.y;
#endif // INT8_MMA_AVAILABLE
        }
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = dptr[0] * ((ls & 254) - 127);
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = dptr[0] * ((ls & 254) - 127);
#endif // INT8_MMA_AVAILABLE
    }

}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_ks(
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

    const int kqsx = threadIdx.x / 4;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const block_iq4_ks * bxi = (const block_iq4_ks *)(dptr + 1) + kbx0;
        const int ls = (bxi->scales[kqsx] & 254) - 127;

        auto values = iq4k_values + ((bxi->scales[kqsx] & 1) << 4);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q4 = get_int_b4(bxi->qs, 4*kqsx+j);
            const int2 v = get_int_from_table_16(q4, values);
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + j + 0] = v.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*kqsx + j + 4] = v.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + j + 0] = v.x;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*kqsx + j + 4] = v.y;
#endif // INT8_MMA_AVAILABLE
        }
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = dptr[0] * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = dptr[0] * ls;
#endif // INT8_MMA_AVAILABLE
    }

}

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_ks_r4(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {

    constexpr int nwarps = mmq_get_nwarps_device();

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_KS_R4, mmq_y);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

    const int kqsx = threadIdx.x/4;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (need_check) {
            i = min(i, i_max);
        }
        int i4 = i/4;
        int ir = i%4;

        const float * dptr = (const float *)(x + 4*i4*stride);
        const block_iq4_ks_r4 * bxi = (const block_iq4_ks_r4 *)(dptr + 4) + kbx0;

        const int ls = (bxi->scales[4*kqsx + ir] & 254) - 127;
        auto values = iq4k_values + ((bxi->scales[4*kqsx+ir] & 1) << 4);

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q4 = get_int_b4(bxi->qs, 16*kqsx+4*j+ir);
            const int2 v = get_int_from_table_16(q4, values);
            const int k0 = 8*kqsx + 4*(j%2) + j/2;
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 0] = v.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + k0 + 2] = v.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + k0 + 0] = v.x;
            x_qs[i*(2*WARP_SIZE + 1)     + k0 + 2] = v.y;
#endif // INT8_MMA_AVAILABLE
        }
#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = dptr[ir] * ls;
#else
        x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = dptr[ir] * ls;
#endif // INT8_MMA_AVAILABLE

    }

}

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits_id<mmq_x, mmq_y, need_check, GGML_TYPE_IQ4_KSS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_kss<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits_id<mmq_x, mmq_y, need_check, GGML_TYPE_IQ4_KS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_ks<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits_id<mmq_x, mmq_y, need_check, GGML_TYPE_IQ4_KS_R4> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_iq4_ks_r4<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_D4>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};

DECL_MMQ_CASE(GGML_TYPE_IQ4_KSS);
DECL_MMQ_CASE(GGML_TYPE_IQ4_KS);
DECL_MMQ_CASE(GGML_TYPE_IQ4_KS_R4);

