#include "../iqk_mmvq_templates.cuh"

// IQ3KS_R16 (type 354): band = 16 fp16 row scales + ntiles*(16 rows x 51 B).
// One 128-col tile holds the 2-bit pairs of all four groups (group g at bits
// 2g of every qs byte) with the high index bits nibble-packed in qh (column
// 2n in the low nibble of qh[n]). w = d_row*(ul-16)*iq3nl[cb][idx].

__device__ __forceinline__ void vec_dot_iq3ks_r16_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const int ib32 = iqs/4;        // 0...7: (tile, group) unit of the 256-col chunk
    const int tile = ib32/4;
    const int g    = ib32%4;

    const uint8_t * band = (const uint8_t *)vbq;
    const uint8_t * braw = band + 32 + (size_t)(2*kbx + tile)*816;

    const auto * vtab = (const uint32_t *)iq3nl_values;

    const float yd = __low2float(bq8_1[ib32].ds);
    const int * ya = (const int *)bq8_1[ib32].qs;

    const int sh = 2*g;

    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const uint8_t * xr = braw + (size_t)r*51;
        const float d = __half2float(((const __half *)band)[r]);
        const uint8_t ex = xr[50];
        const uint32_t aux = (uint32_t)xr[48] | ((uint32_t)xr[49] << 8);
        const int ul = ((aux >> (4*g)) & 0xf) | (((ex >> g) & 1) << 4);
        const float dl = d * (ul - 16);
        const int it = ((ex >> (4+g)) & 1) << 1;
        const uint32_t Tl = vtab[it+0];
        const uint32_t Th = vtab[it+1];

        const uint32_t sb = ((uint32_t)(uintptr_t)xr) & 3;
        const uint32_t * C = (const uint32_t *)(xr - sb);

        int isumi1 = 0, isumi2 = 0;
        #pragma unroll
        for (int o = 0; o < 4; ++o) {
            const uint32_t QA = __funnelshift_r(C[2*o+0], C[2*o+1], 8*sb);
            const uint32_t QB = __funnelshift_r(C[2*o+1], C[2*o+2], 8*sb);
            const uint32_t H  = __funnelshift_r(C[8+o+0], C[8+o+1], 8*sb);
            const uint32_t A = (QA >> sh) & 0x03030303u;
            const uint32_t B = (QB >> sh) & 0x03030303u;
            const uint32_t W2 = __byte_perm(A | (A >> 4), B | (B >> 4), 0x6420);
            const uint32_t W = W2 | (((H >> g) & 0x11111111u) << 2);
            const int vlo = __byte_perm(Tl, Th, W);
            const int vhi = __byte_perm(Tl, Th, W >> 16);
            isumi1 = ggml_cuda_dp4a(vlo, ya[2*o+0], isumi1);
            isumi2 = ggml_cuda_dp4a(vhi, ya[2*o+1], isumi2);
        }
        result[r] += dl * yd * (float)(isumi1 + isumi2);
    }
}

__device__ __forceinline__ void vec_dot_iq3ks_r16_q8_1_tail(
    const void * __restrict__ vbq, const void * __restrict__ bq8_1_v, const int & kbx, const int & iqs, const int & nt, float * result) {

    if (nt != 4) return;   // a 128-col tail: 4 groups x 32 columns

    const block_q8_1 * bq8_1 = (const block_q8_1 *)bq8_1_v;

    const int ib32 = iqs/4;        // 0...7
    const int g    = ib32%4;
    const int r0   = 8*(ib32/4);   // first row of the lane's half of the band

    const uint8_t * band = (const uint8_t *)vbq;
    const uint8_t * braw = band + 32 + (size_t)(2*kbx)*816;

    const auto * vtab = (const uint32_t *)iq3nl_values;

    const float yd = __low2float(bq8_1[g].ds);
    const int * ya = (const int *)bq8_1[g].qs;

    const int sh = 2*g;

    #pragma unroll
    for (int ir = 0; ir < 8; ++ir) {
        const int r = r0 + ir;
        const uint8_t * xr = braw + (size_t)r*51;
        const float d = __half2float(((const __half *)band)[r]);
        const uint8_t ex = xr[50];
        const uint32_t aux = (uint32_t)xr[48] | ((uint32_t)xr[49] << 8);
        const int ul = ((aux >> (4*g)) & 0xf) | (((ex >> g) & 1) << 4);
        const float dl = d * (ul - 16);
        const int it = ((ex >> (4+g)) & 1) << 1;
        const uint32_t Tl = vtab[it+0];
        const uint32_t Th = vtab[it+1];

        const uint32_t sb = ((uint32_t)(uintptr_t)xr) & 3;
        const uint32_t * C = (const uint32_t *)(xr - sb);

        int isumi1 = 0, isumi2 = 0;
        #pragma unroll
        for (int o = 0; o < 4; ++o) {
            const uint32_t QA = __funnelshift_r(C[2*o+0], C[2*o+1], 8*sb);
            const uint32_t QB = __funnelshift_r(C[2*o+1], C[2*o+2], 8*sb);
            const uint32_t H  = __funnelshift_r(C[8+o+0], C[8+o+1], 8*sb);
            const uint32_t A = (QA >> sh) & 0x03030303u;
            const uint32_t B = (QB >> sh) & 0x03030303u;
            const uint32_t W2 = __byte_perm(A | (A >> 4), B | (B >> 4), 0x6420);
            const uint32_t W = W2 | (((H >> g) & 0x11111111u) << 2);
            const int vlo = __byte_perm(Tl, Th, W);
            const int vhi = __byte_perm(Tl, Th, W >> 16);
            isumi1 = ggml_cuda_dp4a(vlo, ya[2*o+0], isumi1);
            isumi2 = ggml_cuda_dp4a(vhi, ya[2*o+1], isumi2);
        }
        result[r] += dl * yd * (float)(isumi1 + isumi2);
    }
}

void mul_mat_vec_iq3ks_r16_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    if (args.ncols_x % QK_K == 0) {
        iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3KS_R16, 4, vec_dot_iq3ks_r16_q8_1, 16>(args, stream);
    } else {
        iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3KS_R16, 4, vec_dot_iq3ks_r16_q8_1, 16, vec_dot_iq3ks_r16_q8_1_tail>(args, stream);
    }
}
