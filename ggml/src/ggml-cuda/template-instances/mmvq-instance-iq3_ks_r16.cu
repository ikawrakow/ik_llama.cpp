#include "../iqk_mmvq_templates.cuh"

// The ggml_cuda_type_traits<GGML_TYPE_IQ3_KS_R16> specialization lives in
// common.cuh (qk = 32, qr = 2, qi = 8) so that both the MMVQ and the MMQ
// translation units see it.

// IQ3KS_R16 (type 354), 4-row-slice shape (Iwan's IQ4_KS_R16 structure):
// n_interleaved = 4 -> grid = nrows/4; the template passes the band base via
// ggml_cuda_actual_row0<type>; the slice is recovered from blockIdx.x.
// One lane per 32-column block: 4 rows x 32 columns per lane.
//
// __byte_perm consumes its selector as NIBBLES, so the per-byte index word is
// compacted with two SWAR steps before the lookup.
__device__ __forceinline__ void vec_dot_iq3ks_r16_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const int ir = 4*(blockIdx.x%4);          // first row of this block's slice
    GGML_UNUSED(iqs);
    const float * dptr = (const float *)vbq;
    const block_iq3_ks_r16 * b = (const block_iq3_ks_r16 *)(dptr + 16) + kbx;

    const float yd = __low2float(bq8_1->ds);
    const int * ya = (const int *)bq8_1->qs;

    const auto * vtab = (const uint32_t *)iq3nl_values;

    #pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int r = ir + rr;
        const float d = dptr[r];
        const int ul = ((b->scales[r & 7] >> (4*(r >> 3))) & 0xf) | (((b->extra >> r) & 1) << 4);
        const float dl = d * (ul - 16);
        const int it = ((b->extra >> (16 + r)) & 1) << 1;
        const uint32_t Tl = vtab[it+0];
        const uint32_t Th = vtab[it+1];

        const uint32_t q0 = *(const uint32_t *)(b->qs + r*4);
        const uint32_t q1 = *(const uint32_t *)(b->qs + 64 + r*4);
        const uint32_t h  = *(const uint32_t *)(b->qh + r*4);

        int isumi1 = 0, isumi2 = 0;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t A = ((q0 >> (2*p)) & 0x03030303u) | (((h >> p) & 0x01010101u) << 2);
            const uint32_t B = ((q1 >> (2*p)) & 0x03030303u) | (((h >> (p+4)) & 0x01010101u) << 2);
            const uint32_t tA = (A | (A >> 4)) & 0x00FF00FFu;
            const uint32_t tB = (B | (B >> 4)) & 0x00FF00FFu;
            const uint32_t sA = (tA | (tA >> 8)) & 0x0000FFFFu;
            const uint32_t sB = (tB | (tB >> 8)) & 0x0000FFFFu;
            const int vlo = __byte_perm(Tl, Th, sA);
            const int vhi = __byte_perm(Tl, Th, sB);
            isumi1 = ggml_cuda_dp4a(vlo, ya[p], isumi1);
            isumi2 = ggml_cuda_dp4a(vhi, ya[4+p], isumi2);
        }
        result[rr] += dl * yd * (float)(isumi1 + isumi2);
    }
}

void mul_mat_vec_iq3ks_r16_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_KS_R16, 8, vec_dot_iq3ks_r16_q8_1, 4>(args, stream);
}
