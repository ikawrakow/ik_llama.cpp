#pragma once

template <ggml_type type> struct mmq_kt_tail { static constexpr bool value = false; };

// IQ3KS_R16 (type 46): rows are 128-weight granular; ne00 % 256 is either 0 or
// 128, so the tail is always exactly one 128-weight superblock (nt == 4). The
// second half of the 256-col iteration is zero-filled so the shared
// vec_dot_q8_0_q8_1_{dp4a,mma} kernels contribute nothing for it.
template <> struct mmq_kt_tail<GGML_TYPE_IQ3KS_R16> {
    static constexpr bool value = true;
    template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load(
        const char * __restrict__ x, int * __restrict__ x_tile, const int & kbx0, const int & i_max, const int & stride, const int & nt) {

        GGML_UNUSED(nt); // always 4 for this type (ne00 % 256 == 128 when there is a tail)

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

            const int ir = i % 16;
            const char * band = x + (i - ir)*stride;
            const float d = __half2float(((const half *)band)[ir]);

            // the tail superblock = first SB of the (partial) 256-col block
            const block_q3ks_g128 * sb = (const block_q3ks_g128 *)(band + 32 + (2*kbx0)*16*sizeof(block_q3ks_g128)) + ir;

            // byte-wise read: odd row offsets (51 B) misalign get_int_b2
            const int ql = get_int_b1(sb->qs, kqsx);

            const uint32_t h0 = sb->qh[2*kqsx + 0];
            const uint32_t h1 = sb->qh[2*kqsx + 1];
            // byte m of qhk = weight (4*kqsx+m)'s code: see the byte-comment in mmq.cuh
        const uint32_t qhk = (h0 & 0xf) | ((h0 & 0xf0) << 4) | ((h1 & 0x0f) << 16) | ((h1 & 0xf0) << 20);

            const uint32_t cb32 = uint32_t((sb->extra >> 4) & 0xf) * 0x01010101;

            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((qhk << 2) & 0x04040404) | ((cb32 << 3) & 0x08080808)
                                                     | ((qhk << 4) & 0x40404040) | ((cb32 << 5) & 0x80808080);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((qhk << 1) & 0x04040404) | ((cb32 << 2) & 0x08080808)
                                                     | ((qhk << 3) & 0x40404040) | ((cb32 << 4) & 0x80808080);
            int2 v1 = get_int_from_table_16(val1, iq3nl_values);
            int2 v2 = get_int_from_table_16(val2, iq3nl_values);

#ifdef INT8_MMA_AVAILABLE
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx +  0] = v1.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx +  8] = v2.x;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 16] = v1.y;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 24] = v2.y;
            // no second superblock in the tail block
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 32] = 0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 40] = 0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 48] = 0;
            x_qs[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx + 56] = 0;

            x_df[i*MMQ_MMA_TILE_X_K_Q8_0 + kqsx] = kqsx < 4 ? d * (int(((sb->scales[kqsx/2] >> 4*(kqsx%2)) & 0xf) | (((sb->extra >> kqsx) & 1) << 4)) - 16) : 0.f;
#else
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx +  0] = v1.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx +  8] = v2.x;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 16] = v1.y;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 24] = v2.y;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 32] = 0;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 40] = 0;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 48] = 0;
            x_qs[i*(2*WARP_SIZE + 1)     + kqsx + 56] = 0;

            x_df[i*(WARP_SIZE/4) + i/4   + kqsx] = kqsx < 4 ? d * (int(((sb->scales[kqsx/2] >> 4*(kqsx%2)) & 0xf) | (((sb->extra >> kqsx) & 1) << 4)) - 16) : 0.f;
#endif // INT8_MMA_AVAILABLE
        }
    }
};

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
