//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "mmvq-templates.cuh"

// IQ3KS_R16 (type 46) W3A16: band addressing — d in the band header at
// 2*(row%16), the row's superblocks at band + 32 + tile*816 + (row%16)*51.
static __global__ void mul_mat_vec_iq3ks_r16_f16_kernel(const mmvq_args args) {

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= args.nrows_x) return;

    const int i2 = blockIdx.z;

    const int nslices = args.ncols_x / 8;
    const size_t band_size = 32 + (args.ncols_x / QK3KS_G128) * 16 * sizeof(block_q3ks_g128);
    const char * xr = (const char *)args.vx_u + i2*args.nb02 + (size_t)(row >> 4)*band_size;
    const int ir = row & 15;
    const float d = __half2float(__ushort_as_half(
        (unsigned short)(((uint16_t)(uint8_t)xr[2*ir]) | ((uint16_t)(uint8_t)xr[2*ir + 1] << 8))));
    const char * xrow = xr + 32 + (size_t)ir*sizeof(block_q3ks_g128);
    const float * y0 = (const float *)((const char *)args.vy + i2*args.nb12)
                     + (int64_t)blockIdx.y * args.ncols_x;

    const uint8_t * vtab = (const uint8_t *)iq3nl_values;
    const uint32_t T0 = (uint32_t)vtab[0] | ((uint32_t)vtab[1] << 8)
                      | ((uint32_t)vtab[2] << 16) | ((uint32_t)vtab[3] << 24);
    const uint32_t T1 = (uint32_t)vtab[4] | ((uint32_t)vtab[5] << 8)
                      | ((uint32_t)vtab[6] << 16) | ((uint32_t)vtab[7] << 24);
    const uint32_t T2 = (uint32_t)vtab[8] | ((uint32_t)vtab[9] << 8)
                      | ((uint32_t)vtab[10] << 16) | ((uint32_t)vtab[11] << 24);
    const uint32_t T3 = (uint32_t)vtab[12] | ((uint32_t)vtab[13] << 8)
                      | ((uint32_t)vtab[14] << 16) | ((uint32_t)vtab[15] << 24);

    float tmp = 0.0f;

    for (int i = threadIdx.x; i < nslices; i += 32) {
        const int ibl = i >> 4;          // superblock index within row
        const int s  = i & 15;           // 8-weight slice within superblock
        const int g  = s >> 2;           // codebook group 0..3
        const int j0 = (s & 3) * 8;      // within-group start
        const block_q3ks_g128 * b = (const block_q3ks_g128 *)(xrow + (size_t)ibl*16*sizeof(block_q3ks_g128));
        const uint8_t ex = b->extra;
        const uint8_t s0 = b->scales[0], s1 = b->scales[1];
        const int ul = (g == 0) ? ((s0 & 0xf) | ((ex & 1) << 4)) :
                       (g == 1) ? ((s0 >> 4)  | ((ex & 2) << 3)) :
                       (g == 2) ? ((s1 & 0xf) | ((ex & 4) << 2)) :
                                  ((s1 >> 4)  | ((ex & 8) << 1));
        const float dl = d * (ul - 16);
        const uint32_t Tl = (((ex >> (4 + g)) & 1) == 0) ? T0 : T2;
        const uint32_t Th = (((ex >> (4 + g)) & 1) == 0) ? T1 : T3;
        const uint8_t * qs = b->qs + j0;
        const uint8_t * qh = b->qh + (j0 >> 1);
        const float * ys = y0 + ibl*QK3KS_G128 + g*32 + j0;
        const float4 yv0 = *((const float4 *)(ys + 0));
        const float4 yv1 = *((const float4 *)(ys + 4));
        const int sh = 2*g;
        const int i0 = ((qs[0] >> sh) & 3) | (((qh[0] >> (g + 0)) & 1) << 2);
        const int i1 = ((qs[1] >> sh) & 3) | (((qh[0] >> (g + 4)) & 1) << 2);
        const int i2 = ((qs[2] >> sh) & 3) | (((qh[1] >> (g + 0)) & 1) << 2);
        const int i3 = ((qs[3] >> sh) & 3) | (((qh[1] >> (g + 4)) & 1) << 2);
        const int i4 = ((qs[4] >> sh) & 3) | (((qh[2] >> (g + 0)) & 1) << 2);
        const int i5 = ((qs[5] >> sh) & 3) | (((qh[2] >> (g + 4)) & 1) << 2);
        const int i6 = ((qs[6] >> sh) & 3) | (((qh[3] >> (g + 0)) & 1) << 2);
        const int i7 = ((qs[7] >> sh) & 3) | (((qh[3] >> (g + 4)) & 1) << 2);
        const int32_t p0 = __byte_perm(Tl, Th, (i0 | (i1 << 4) | (i2 << 8) | (i3 << 12)));
        const int32_t p1 = __byte_perm(Tl, Th, (i4 | (i5 << 4) | (i6 << 8) | (i7 << 12)));
        const float acc = (float)((int8_t)(p0 >>  0))*yv0.x + (float)((int8_t)(p0 >>  8))*yv0.y
                        + (float)((int8_t)(p0 >> 16))*yv0.z + (float)((int8_t)(p0 >> 24))*yv0.w
                        + (float)((int8_t)(p1 >>  0))*yv1.x + (float)((int8_t)(p1 >>  8))*yv1.y
                        + (float)((int8_t)(p1 >> 16))*yv1.z + (float)((int8_t)(p1 >> 24))*yv1.w;
        tmp += dl * acc;
    }

    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        float result = tmp;
        if (args.bias_u) {
            result += ((const float *)args.bias_u)[row];
        }
        float * dst = (float *)((char *)args.dst + i2*args.nb2);
        dst[blockIdx.y*args.nrows_dst + row] = result;
    }
}

void mul_mat_vec_iq3ks_r16_f16_cuda(const mmvq_args & args, cudaStream_t stream) {
    GGML_ASSERT(args.ncols_x % QK3KS_G128 == 0);
    GGML_ASSERT(args.nrows_x % 16 == 0);
    constexpr int rows_per_block = 4;
    const dim3 block(32, rows_per_block, 1);
    const dim3 grid((args.nrows_x + rows_per_block - 1)/rows_per_block, args.ncols_y, args.ne2);
    mul_mat_vec_iq3ks_r16_f16_kernel<<<grid, block, 0, stream>>>(args);
}
// IQ3KS_R16 (type 46) dp4a MMVQ: same band addressing as the f16 kernel.
// Multi-warp: the block's warps split the (tile,group) work items of the
// 4-row slice; the per-warp partial sums combine via shared memory.
static __global__ void mul_mat_vec_iq3ks_r16_band_q8_1_kernel(const mmvq_args args) {

    // band-cooperative, 4-row slices: one block owns a 4-row slice of a
    // band; lanes cover (tile,group) pairs strided by 32*nwarps; the q8_1 y
    // ints are loaded once per pair and reused across the 4 rows (4x fewer
    // y loads than row-per-warp) while keeping block-level parallelism at
    // nrows/4. Decode is the packed prmt/dp4a version (bit-exact vs the
    // byte-wise decode: standalone probes 0/16 mismatches vs both the CPU
    // reference and the previous kernel).

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int nwarps = blockDim.y;
    const int band = blockIdx.x >> 2;            // 16-row band index
    const int r0   = (blockIdx.x & 3) * 4;       // first row within the band
    const int i2   = blockIdx.z;

    const int ntiles = args.ncols_x / QK3KS_G128;
    const size_t band_size = 32 + (size_t)ntiles * 16 * sizeof(block_q3ks_g128);
    const char * xr = (const char *)args.vx_u + i2*args.nb02 + (size_t)band*band_size;

    const uint8_t * vtab = (const uint8_t *)iq3nl_values;
    const uint32_t T0 = (uint32_t)vtab[0] | ((uint32_t)vtab[1] << 8)
                      | ((uint32_t)vtab[2] << 16) | ((uint32_t)vtab[3] << 24);
    const uint32_t T1 = (uint32_t)vtab[4] | ((uint32_t)vtab[5] << 8)
                      | ((uint32_t)vtab[6] << 16) | ((uint32_t)vtab[7] << 24);
    const uint32_t T2 = (uint32_t)vtab[8] | ((uint32_t)vtab[9] << 8)
                      | ((uint32_t)vtab[10] << 16) | ((uint32_t)vtab[11] << 24);
    const uint32_t T3 = (uint32_t)vtab[12] | ((uint32_t)vtab[13] << 8)
                      | ((uint32_t)vtab[14] << 16) | ((uint32_t)vtab[15] << 24);

    float acc[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) acc[r] = 0.f;

    const int blocks_per_col_y = args.nrows_y / QK8_1;

    for (int ig = lane + 32*warp; ig < ntiles*4; ig += 32*nwarps) {
        const int ibl = ig >> 2;
        const int g   = ig & 3;

        // y-side: one q8_1 block per (tile,group); ints loaded once here,
        // reused by the 4 rows of this block's slice.
        const block_q8_1 * yb = (const block_q8_1 *)((const char *)args.vy + i2*args.nb12)
                                + blockIdx.y*blocks_per_col_y + ibl*4 + g;
        const float yd = __half2float(yb->data.d);
        int ya[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) memcpy(&ya[i], yb->qs + 4*i, 4);

        const int sh = 2*g;

        for (int r = 0; r < 4; ++r) {
            const uint8_t * braw = (const uint8_t *)(xr + 32 + (size_t)ibl*16*sizeof(block_q3ks_g128)
                                                     + (size_t)(r0 + r)*sizeof(block_q3ks_g128));
            const uint32_t sb = ((uint32_t)(uintptr_t)braw) & 3;   // braw % 4
            const uint32_t * C = (const uint32_t *)(braw - sb);    // 4-aligned window
            const uint8_t ex = braw[50];
            const uint32_t aux = (uint32_t)braw[48] | ((uint32_t)braw[49] << 8);
            const int ul = ((aux >> (4*g)) & 0xf) | (((ex >> g) & 1) << 4);
            const float d = __half2float(__ushort_as_half(
                (uint16_t)(uint8_t)xr[2*(r0 + r)] | ((uint16_t)(uint8_t)xr[2*(r0 + r)+1] << 8)));
            const float dl = d * (ul - 16);
            const uint32_t Tl = (((ex >> (4+g)) & 1) == 0) ? T0 : T2;
            const uint32_t Th = (((ex >> (4+g)) & 1) == 0) ? T1 : T3;

            int isumi = 0;
            #pragma unroll
            for (int s4 = 0; s4 < 4; ++s4) {
                // aligned u32 loads + funnel realignment (13 loads / 12 shifts per 32
                // weights, vs 48 byte loads + address math): C[2s4], C[2s4+1] cover
                // qs bytes 8s4..8s4+8; C[8+s4], C[9+s4] cover qh bytes 4s4..4s4+4.
                const uint32_t QA = __funnelshift_r(C[2*s4+0], C[2*s4+1], 8*sb);
                const uint32_t QB = __funnelshift_r(C[2*s4+1], C[2*s4+2], 8*sb);
                const uint32_t H  = __funnelshift_r(C[8+s4+0], C[8+s4+1], 8*sb);
                const uint32_t A = (QA >> sh) & 0x03030303u;
                const uint32_t B = (QB >> sh) & 0x03030303u;
                const uint32_t Wq = __byte_perm(A | (A >> 4), B | (B >> 4), 0x6420);
                const uint32_t W = Wq | (((H >> g) & 0x11111111u) << 2);
                const int vlo = __byte_perm(Tl, Th, W);
                const int vhi = __byte_perm(Tl, Th, W >> 16);
                isumi += __dp4a(vlo, ya[2*s4+0], 0) + __dp4a(vhi, ya[2*s4+1], 0);
            }
            acc[r] += dl * yd * (float)isumi;
        }
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) acc[r] = warp_reduce_sum(acc[r]);

    __shared__ float red[8][4];
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) red[warp][r] = acc[r];
    }
    __syncthreads();

    if (warp == 0 && lane < 4) {
        float result = 0.f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (w < nwarps) result += red[w][lane];
        }
        const int row = (int)blockIdx.x*4 + lane;
        if (args.bias_u) {
            result += ((const float *)args.bias_u)[row];
        }
        float * dst = (float *)((char *)args.dst + i2*args.nb2);
        dst[blockIdx.y*args.nrows_dst + row] = result;
    }
}

void mul_mat_vec_iq3ks_r16_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    GGML_ASSERT(args.ncols_x % QK3KS_G128 == 0);
    GGML_ASSERT(args.nrows_x % 16 == 0);
    const int npairs = (args.ncols_x / QK3KS_G128) * 4;
    // warps split the (tile,group) work items: 1 warp covers <=32 items;
    // long-K layers (hidden 2048/8192: 16-64 tiles) get 2-4 warps so each
    // lane iterates ~1-3 chunks instead of serially walking all of K.
    const int nwarps = npairs >= 128 ? 4 : (npairs >= 64 ? 2 : 1);
    const dim3 block(32, nwarps, 1);
    const dim3 grid(args.nrows_x/4, args.ncols_y, args.ne2);
    mul_mat_vec_iq3ks_r16_band_q8_1_kernel<<<grid, block, 0, stream>>>(args);
}
static void ggml_cuda_op_mul_mat_vec_q_impl(ggml_backend_cuda_context & ctx, ggml_type type,
        const int64_t ne00, const int64_t ne0, const int64_t ne2,
        const int64_t nb02, const int64_t nb12, const int64_t nb2, const int64_t ids_nb0, const int64_t bias_nb1,
        const char * src0_dd_u, const char * src0_dd_g, const char * src1_ddq_i, float * dst_dd_i, const char * ids_data,
        const void * bias_u, const void * bias_g,
        const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
        const int64_t src1_padded_row_size, ggml_unary_op unary_op, float limit, cudaStream_t stream) {

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    mmvq_args args{/* vx_u     */ src0_dd_u,
                   /* vx_g     */ src0_dd_g,
                   /* bias_u   */ bias_u,
                   /* bias_g   */ bias_g,
                   /* vy       */ src1_ddq_i,
                   /* dst      */ dst_dd_i,
                   /* ids_data */ ids_data,
                   /* ncols_x  */ int(ne00),
                   /* nrows_x  */ int(row_diff),
                   /* nrows_y  */ int(src1_padded_row_size),
                   /* ncols_y  */ int(src1_ncols),
                   /* nrows_dst*/ int(nrows_dst),
                   /* ne2      */ int(ne2),
                   /* nb02     */ uint64_t(nb02),
                   /* nb12     */ uint64_t(nb12),
                   /* nb2      */ uint64_t(nb2),
                   /* ids_nb0  */ uint64_t(ids_nb0),
                   /* bias_nb1 */ uint64_t(bias_nb1),
                   /* unary_op */ unary_op,
                   /* limit    */ limit > 1e-6f ? limit : INFINITY
    };

    switch (type) {
        case GGML_TYPE_IQ3KS_R16:
            mul_mat_vec_iq3ks_r16_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q4_1_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q5_0_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q5_1_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_Q6_0:
            mul_mat_vec_q6_0_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q2_K_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q3_K_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q4_K_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q5_K_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q6_K_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_iq2_xxs_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_iq2_xs_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_iq2_s_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_iq3_xxs_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_iq3_s_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_iq1_s_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_iq1_m_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_iq4_nl_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_mxfp4_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_iq4_xs_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ1_BN:
        case GGML_TYPE_IQ2_BN:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ2_K_R4:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_K_R4:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:
            iqk_mul_mat_vec_q(type, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

}

void ggml_cuda_op_mul_mat_vec_q_3D(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);
    GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src0->ne[2] == src1->ne[2] && src0->ne[2] == dst->ne[2]);

    const int64_t ne0 = dst->ne[0];

    const int64_t src1_row_size = ggml_row_size(GGML_TYPE_Q8_1, src1_padded_row_size);

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, dst->ne[2],
        src0->nb[2], src1_row_size, dst->nb[2], 0, 0,
        src0_dd_i, nullptr, src1_ddq_i, dst_dd_i, nullptr, nullptr, nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, GGML_UNARY_OP_COUNT, 0.0f, stream);

    GGML_UNUSED(src1_ddf_i);
}

void ggml_cuda_op_mul_mat_vec_q_biased(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const ggml_tensor * bias,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    if (bias) {
        if (bias->ne[0] != ne0) {
            printf("Oops: bias %s is %ld x %ld x %ld x %ld, dst %s is %ld x %ld x %ld x %ld\n",
                    bias->name, bias->ne[0], bias->ne[1], bias->ne[2], bias->ne[3],
                    dst->name, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        }
        GGML_ASSERT(bias->ne[0] == ne0);
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        if (ggml_nrows(bias) != 1) {
            printf("Oops: bias %s is %ld x %ld x %ld x %ld\n", bias->name, bias->ne[0], bias->ne[1], bias->ne[2], bias->ne[3]);
        }
        GGML_ASSERT(ggml_nrows(bias) == 1);
    }

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, 1, 0, 0, 0, 0, 0,
        src0_dd_i, nullptr, src1_ddq_i, dst_dd_i, nullptr, bias ? bias->data : nullptr, nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, GGML_UNARY_OP_COUNT, 0.0f, stream);

    GGML_UNUSED(src1_ddf_i);
}
void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {
    ggml_cuda_op_mul_mat_vec_q_biased(ctx, src0, src1, dst, nullptr, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i, row_low, row_high, src1_ncols,
            src1_padded_row_size, stream);
}

void ggml_cuda_op_mul_mat_vec_q_id(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_tensor * bias,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);
    GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src1->ne[1] <= MMVQ_MAX_BATCH_SIZE && src1->ne[2] == 1);
    GGML_ASSERT(ids->ne[0] == dst->ne[2]);

    const int64_t ne0 = dst->ne[0];

    if (bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(bias->ne[0] == ne0);
        if (ids) {
            //GGML_ASSERT(bias->ne[1] == src0->ne[2]);
            GGML_ASSERT(bias->ne[2] == 1 && bias->ne[3] == 1);
        } else {
            GGML_ASSERT(ggml_nrows(bias) == 1);
        }
    }

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, dst->ne[2],
        src0->nb[2], src1->nb[2], dst->nb[2], ids->nb[0], bias ? bias->nb[1] : 0,
        src0_dd_i, nullptr, src1_ddq_i, dst_dd_i, (const char *)ids->data, bias ? bias->data : nullptr, nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, GGML_UNARY_OP_COUNT, 0.0f, stream);

    GGML_UNUSED(src1_ddf_i);
}

void ggml_cuda_op_fused_mul_mat_vec_q_id(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_tensor * bias_u, const ggml_tensor * bias_g,
    const char * src0_dd_u, const char * src0_dd_g, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, ggml_unary_op unary_op, float limit, cudaStream_t stream) {

    if (!bias_u && !bias_g) {
        GGML_ASSERT(unary_op == GGML_UNARY_OP_SILU ||
                    unary_op == GGML_UNARY_OP_RELU ||
                    unary_op == GGML_UNARY_OP_GELU ||
                    unary_op == GGML_UNARY_OP_SWIGLU_OAI);
    } else {
        GGML_ASSERT(unary_op == GGML_UNARY_OP_SWIGLU_OAI);
        GGML_ASSERT(bias_u && bias_g);
        GGML_ASSERT(bias_u->data && bias_g->data);
        GGML_ASSERT(bias_u->nb[1] == bias_g->nb[1]);
        GGML_ASSERT(bias_u->ne[0] == dst->ne[0]);
        GGML_ASSERT(bias_g->ne[0] == dst->ne[0]);
    }
    GGML_ASSERT(src0_dd_u && src0_dd_g);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);
    GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src1->ne[1] == 1 && src1->ne[2] == 1);
    //if (ids && ids->ne[0] != dst->ne[2]) {
    //    printf("%s(%s->%s): unexpected situation\n", __func__, src0->name, dst->name);
    //    printf("  src0 = %ld x %ld x %ld x %ld\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //    printf("  src1 = %ld x %ld x %ld x %ld\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //    printf("   ids = %ld x %ld x %ld x %ld\n", ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3]);
    //    printf("   dst = %ld x %ld x %ld x %ld\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    //    GGML_ABORT("Fatal error");
    //}

    const int64_t ne0 = dst->ne[0];

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, dst->ne[2],
        src0->nb[2], src1->nb[2], dst->nb[2], ids ? ids->nb[0] : 0, bias_u ? bias_u->nb[1] : 0,
        src0_dd_u, src0_dd_g, src1_ddq_i, dst_dd_i, ids ? (const char *)ids->data : nullptr,
        bias_u ? bias_u->data : nullptr, bias_g ? bias_g->data : nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, unary_op, limit, stream);

    GGML_UNUSED(src1_ddf_i);
}


bool ggml_cuda_mmvq_type_supported(ggml_type src0_type) {
    switch (src0_type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_BN:
        case GGML_TYPE_IQ2_BN:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_K_R4:
        case GGML_TYPE_IQ3_K_R4:
        case GGML_TYPE_IQ4_K_R4:
        case GGML_TYPE_IQ4_KS_R4:
        case GGML_TYPE_IQ5_K_R4:
        case GGML_TYPE_IQ5_KS_R4:
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
            return true;
        case GGML_TYPE_IQ3KS_R16:
            return true;
        default:
            return false;
    }
}
