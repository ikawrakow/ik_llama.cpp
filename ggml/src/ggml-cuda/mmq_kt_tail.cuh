#pragma once

template <ggml_type type> struct mmq_kt_tail { static constexpr bool value = false; };

template <> struct mmq_kt_tail<GGML_TYPE_IQ3_KT> {
    static constexpr bool value = true;
    template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load(
        const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride, const int & nt) {

        constexpr uint32_t ka = 0xCBAC1FED;
        constexpr uint32_t km = 0x3f3f3f3f;

#ifdef INT8_MMA_AVAILABLE
        int   * x_qs = (int   *)  x_tile;
        float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
        constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
        int   * x_qs = (int   *)  x_tile;
        float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

        const int kqsx = threadIdx.x;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
            int i = i0 + threadIdx.y;

            if (need_check) {
                i = min(i, i_max);
            }

            const uint8_t * tail = (const uint8_t *)((const block_iq3_kt *)(x + i*stride + sizeof(float)) + kbx0);

            int ib32 = kqsx/4;
            int j    = kqsx%4;
            int2 v = {0, 0};
            if (ib32 < nt) {
                const uint8_t * ql = tail + 2*(4*ib32 + j);
                const uint32_t sb = tail[8*nt + 4*ib32 + j];
                uint32_t val = (ql[0] | (ql[1] << 8)) + 4096;
                for (int k = 0; k < 4; ++k) {
                    val *= ka;
                    v.x |= std::abs(ggml_cuda_dp4a(val & km, 0x01010101, -126)) << 8*k;
                }
                auto signs = __vcmpne4(((sb & 0x0f) * 0x00204081) & 0x01010101, 0);
                v.x = __vsub4(v.x ^ signs, signs);
                for (int k = 0; k < 4; ++k) {
                    val *= ka;
                    v.y |= std::abs(ggml_cuda_dp4a(val & km, 0x01010101, -126)) << 8*k;
                }
                signs = __vcmpne4(((sb >> 4) * 0x00204081) & 0x01010101, 0);
                v.y = __vsub4(v.y ^ signs, signs);
            }
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 0] = v.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 1] = v.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 0] = v.x;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 1] = v.y;
#endif // INT8_MMA_AVAILABLE
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
            int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

            if (need_check) {
                i = min(i, i_max);
            }

            const float * dptr = (const float *)(x + i*stride);
            const float d = dptr[0] * 1.01f;
            const uint8_t * scales = (const uint8_t *)((const block_iq3_kt *)(dptr + 1) + kbx0) + 12*nt;
            int ib32 = threadIdx.x % 8;
            const int ls = ib32 < nt ? (scales[ib32/2] >> 4*(ib32&1)) & 0xf : 0;

#ifdef INT8_MMA_AVAILABLE
            x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = d * ls;
#else
            x_df[i*(WARP_SIZE/4) + i/4   + threadIdx.x % 8] = d * ls;
#endif // INT8_MMA_AVAILABLE
        }
    }
};

template <> struct mmq_kt_tail<GGML_TYPE_IQ4_KT> {
    static constexpr bool value = true;
    template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load(
        const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride, const int & nt) {

        constexpr uint32_t ka = 0xCBAC1FED;
        constexpr uint32_t km = 0x3f3f3f3f;

#ifdef INT8_MMA_AVAILABLE
        int   * x_qs = (int   *)  x_tile;
        float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
        constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, mmq_y);
        int   * x_qs = (int   *)  x_tile;
        float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

        const int kqsx = threadIdx.x;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
            int i = i0 + threadIdx.y;

            if (need_check) {
                i = min(i, i_max);
            }

            const uint32_t * shb = (const uint32_t *)((const block_iq4_kt *)(x + i*stride + sizeof(float)) + kbx0);
            const uint8_t * tail = (const uint8_t *)shb;

            int ib32 = kqsx/4;
            int j    = kqsx%4;
            int2 v = {0, 0};
            if (ib32 < nt) {
                const uint8_t * ql = tail + 16*ib32 + 4 + 2*j;
                const uint32_t qh  = tail[16*ib32 + 12 + j];
                const uint32_t sh = shb[4*ib32] >> (8 + 6*j);
                uint32_t offset = 4096 + ((shb[4*ib32] & 1) << 15);
                uint32_t val1 = offset + ql[0] + ((qh & 0x0f) << 8) + ((sh & 7) << 12);
                uint32_t val2 = offset + ql[1] + ((qh & 0xf0) << 4) + ((sh & 56) << 9);
                for (int k = 0; k < 4; ++k) {
                    val1 *= ka;
                    val2 *= ka;
                    v.x |= (ggml_cuda_dp4a(val1 & km, 0x01010101, -126) & 0xff) << 8*k;
                    v.y |= (ggml_cuda_dp4a(val2 & km, 0x01010101, -126) & 0xff) << 8*k;
                }
            }
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 0] = v.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 1] = v.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 0] = v.x;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 1] = v.y;
#endif // INT8_MMA_AVAILABLE
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
            int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

            if (need_check) {
                i = min(i, i_max);
            }

            const float * dptr = (const float *)(x + i*stride);
            const uint32_t * shb = (const uint32_t *)((const block_iq4_kt *)(dptr + 1) + kbx0);
            int ib32 = threadIdx.x % 8;
            const int ls = ib32 < nt ? (shb[4*ib32] & 0xff) >> 1 : 64;

#ifdef INT8_MMA_AVAILABLE
            x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = dptr[0] * (ls - 64);
#else
            x_df[i*(WARP_SIZE/4) + i/4   + threadIdx.x % 8] = dptr[0] * (ls - 64);
#endif // INT8_MMA_AVAILABLE
        }
    }
};

template <> struct mmq_kt_tail<GGML_TYPE_IQ3KS_R16> {
    static constexpr bool value = true;
    template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load(
        const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride, const int & nt) {

#ifdef INT8_MMA_AVAILABLE
        int   * x_qs = (int   *)  x_tile;
        float * x_df = (float *) (x_qs + WARP_SIZE*2);
#else
        constexpr tile_x_sizes txs = MMQ_DP4A_TXS_Q8_0_16;
        int   * x_qs = (int   *)  x_tile;
        float * x_df = (float *) (x_qs + txs.qs);
#endif // INT8_MMA_AVAILABLE

        const int kqsx = threadIdx.x;

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
            int i = i0 + threadIdx.y;

            if (need_check) {
                i = min(i, i_max);
            }

            const int ir = i % 16;
            const char * band = x + (i - ir)*stride;

            const int ib32 = kqsx/4;
            const int j    = kqsx%4;
            int2 v = {0, 0};
            if (ib32 < nt) {
                const block_iq3_ks_r16 * bxi = (const block_iq3_ks_r16 *)(band + 64) + kbx0*8 + ib32;
                const uint32_t q0 = *(const uint32_t *)(bxi->qs + ir*4);
                const uint32_t q1 = *(const uint32_t *)(bxi->qs + 64 + ir*4);
                const uint32_t h  = *(const uint32_t *)(bxi->qh + ir*4);
                const int page = (bxi->extra >> (16 + ir)) & 1;
                const uint32_t * vtab = (const uint32_t *)iq3nl_values;
                const uint32_t Tl = vtab[2*page + 0];
                const uint32_t Th = vtab[2*page + 1];
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const int mm = 2*j + m;
                    const uint32_t A = (mm < 4 ? ((q0 >> (2*mm)) & 0x03030303u) : ((q1 >> (2*(mm-4))) & 0x03030303u))
                                     | (((h >> mm) & 0x01010101u) << 2);
                    const uint32_t t = (A | (A >> 4)) & 0x00FF00FFu;
                    const uint32_t s = (t | (t >> 8)) & 0x0000FFFFu;
                    if (m == 0) v.x = __byte_perm(Tl, Th, s); else v.y = __byte_perm(Tl, Th, s);
                }
            }
#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 0] = v.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + 8*ib32 + 2*j + 1] = v.y;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 0] = v.x;
            x_qs[i*(2*WARP_SIZE + 1)     + 8*ib32 + 2*j + 1] = v.y;
#endif // INT8_MMA_AVAILABLE
        }

#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
            int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

            if (need_check) {
                i = min(i, i_max);
            }

            const int ir = i % 16;
            const char * band = x + (i - ir)*stride;
            const float d = ((const float *)band)[ir];
            const block_iq3_ks_r16 * bxi = (const block_iq3_ks_r16 *)(band + 64) + kbx0*8;
            const int ib32 = threadIdx.x % 8;
            // only read the block when it is part of the tail (the others would
            // be past the row's block area); padding rows get ul = 16 -> df = 0.
            const int ul = ib32 < nt
                ? (((bxi[ib32].scales[ir & 7] >> (4*(ir >> 3))) & 0xf) | (((bxi[ib32].extra >> ir) & 1) << 4))
                : 16;

#ifdef INT8_MMA_AVAILABLE
            x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + threadIdx.x % 8] = d * (ul - 16);
#else
            x_df[i*(WARP_SIZE/4) + i/4   + threadIdx.x % 8] = d * (ul - 16);
#endif // INT8_MMA_AVAILABLE
        }
    }
};
