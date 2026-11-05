#include "../mmq_id_common.cuh"

template <int mmq_y, bool need_check> static __device__ __forceinline__ void load_tiles_nvfp4_ks(
    const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride, [[maybe_unused]] int blocks_per_ne00) {

    constexpr int nwarps = mmq_get_nwarps_device();

#ifdef INT8_MMA_AVAILABLE
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
    constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (need_check) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const block_nvfp4 * bxi = (const block_nvfp4 *)(dptr + 1) + kbx0 + kqsx/2;
        const int ib16 = 2*(kqsx%2);

    #pragma unroll
        for (int l = 0; l < 2; ++l) {
            const int2 v1 = get_int_from_table_16(get_int_b4(bxi->qs, 2*(ib16 + l) + 0), kvalues_mxfp4);
            const int2 v2 = get_int_from_table_16(get_int_b4(bxi->qs, 2*(ib16 + l) + 1), kvalues_mxfp4);
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
#endif
        }

        const float d = dptr[0]*0.5f;

#ifdef INT8_MMA_AVAILABLE
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+0] = d * ggml_cuda_ue4m3_to_fp32(bxi->d[ib16+0]);
        x_df[i*MMQ_MMA_TILE_X_K_Q3_K               + 2*kqsx+1] = d * ggml_cuda_ue4m3_to_fp32(bxi->d[ib16+1]);
#else
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ggml_cuda_ue4m3_to_fp32(bxi->d[ib16+0]);
        x_df[i*(2*WARP_SIZE*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ggml_cuda_ue4m3_to_fp32(bxi->d[ib16+1]);
#endif
    }
}

template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits_id<mmq_x, mmq_y, need_check, GGML_TYPE_NVFP4_KS> {
    static constexpr load_tiles_mmq_t load_tiles   = load_tiles_nvfp4_ks<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma  = vec_dot_q8_0_16_q8_1_mma<mmq_x, mmq_y>;
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_16_q8_1_dp4a<mmq_x, mmq_y>;
};

DECL_MMQ_CASE(GGML_TYPE_NVFP4_KS);
