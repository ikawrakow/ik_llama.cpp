#include "t16-matmul.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>

// K2 (t16): exact EXL3 trellis decode.
//
//   window = 16 circular bits at stream position tpos (K bits per step)
//   x = window * 0x83DCD12D (u32); s = 0x6400 + byte_sum(x)
//   h = reinterpret_f16(low 16 bits of s); val = hfma(h, 0x1EEE, 0xC931)
//
// The trellis tile (i, j) is 16K little-endian uint16 words at
//   data + j*nb1 + i*nb2,
// and the decoded 16x16 tile is row-major (in, out): position 16*r + c holds the
// window inv_perm[16*r + c]. One block decodes one tile for up to K2_T16_TOK_BLK
// token columns; threads are (c = tid/16, r = tid%16) so the 16-lane reduction over
// r is a width-16 shuffle.

#define K2_T16_TOK_BLK 32

// inv_perm[q]: window stream position for the row-major tile position q = 16*r + c
// (inverse of exllamav3's tensor_core_perm, verified against k2_codec).
static __constant__ uint8_t c_k2_t16_inv_perm[256] = {
      0,  32,  64,  96, 128, 160, 192, 224,   4,  36,  68, 100, 132, 164, 196, 228,
      1,  33,  65,  97, 129, 161, 193, 225,   5,  37,  69, 101, 133, 165, 197, 229,
      8,  40,  72, 104, 136, 168, 200, 232,  12,  44,  76, 108, 140, 172, 204, 236,
      9,  41,  73, 105, 137, 169, 201, 233,  13,  45,  77, 109, 141, 173, 205, 237,
     16,  48,  80, 112, 144, 176, 208, 240,  20,  52,  84, 116, 148, 180, 212, 244,
     17,  49,  81, 113, 145, 177, 209, 241,  21,  53,  85, 117, 149, 181, 213, 245,
     24,  56,  88, 120, 152, 184, 216, 248,  28,  60,  92, 124, 156, 188, 220, 252,
     25,  57,  89, 121, 153, 185, 217, 249,  29,  61,  93, 125, 157, 189, 221, 253,
      2,  34,  66,  98, 130, 162, 194, 226,   6,  38,  70, 102, 134, 166, 198, 230,
      3,  35,  67,  99, 131, 163, 195, 227,   7,  39,  71, 103, 135, 167, 199, 231,
     10,  42,  74, 106, 138, 170, 202, 234,  14,  46,  78, 110, 142, 174, 206, 238,
     11,  43,  75, 107, 139, 171, 203, 235,  15,  47,  79, 111, 143, 175, 207, 239,
     18,  50,  82, 114, 146, 178, 210, 242,  22,  54,  86, 118, 150, 182, 214, 246,
     19,  51,  83, 115, 147, 179, 211, 243,  23,  55,  87, 119, 151, 183, 215, 247,
     26,  58,  90, 122, 154, 186, 218, 250,  30,  62,  94, 126, 158, 190, 222, 254,
     27,  59,  91, 123, 155, 187, 219, 251,  31,  63,  95, 127, 159, 191, 223, 255,
};

static __device__ __forceinline__ uint32_t k2_t16_fshr(uint32_t lo, uint32_t hi, uint32_t s) {
    s &= 31;
    return s == 0 ? lo : ((hi << (32 - s)) | (lo >> s));
}

static __device__ __forceinline__ float k2_t16_mul1(uint16_t w) {
    const uint32_t x = uint32_t(w) * 0x83DCD12Du;
    // 0x6400 + unsigned byte sum; the low 16 bits are fp16 *bits*.
    // Measured faster than __dp4a on sm_86 in this kernel (69.6 vs 83.6 t/s tg).
    const uint32_t s = 0x6400u + ((x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24));
    const __half h = __ushort_as_half((unsigned short)(s & 0xFFFFu));
    const __half v = __hfma(h, __ushort_as_half((unsigned short)0x1EEEu), __ushort_as_half((unsigned short)0xC931u));
    return __half2float(v);
}

static __global__ void k2_t16_matmul_kernel(
        const uint16_t * __restrict__ trellis,
        const float    * __restrict__ x,
        float          * __restrict__ y,
        int K, int I, int J, int n,
        size_t nb1, size_t nb2) {
    const int j    = blockIdx.x;
    const int t0   = blockIdx.y * K2_T16_TOK_BLK;
    const int tid  = threadIdx.x;
    const int c    = tid >> 4;
    const int r    = tid & 15;
    const int q    = 16*r + c;
    const int tpos = c_k2_t16_inv_perm[q];
    const int b    = tpos*K + K - 16 + 256*K;
    const int ih   = b >> 5;
    const int il   = (b + 15) >> 5;
    const int sh   = (il + 1)*32 - (b + 16);
    const int nw   = 8*K;

    float acc[K2_T16_TOK_BLK];
#pragma unroll
    for (int t = 0; t < K2_T16_TOK_BLK; ++t) {
        acc[t] = 0.0f;
    }

    const int ntok = min(K2_T16_TOK_BLK, n - t0);
    const char * tile0 = (const char *) trellis + j*nb1;
    for (int i = 0; i < I; ++i) {
        const uint32_t * p = (const uint32_t *)(tile0 + i*nb2);
        const uint32_t w1 = k2_t16_fshr(p[il % nw], p[ih % nw], (uint32_t) sh);
        const float v = k2_t16_mul1((uint16_t)(w1 & 0xFFFFu));
        const float * xr = x + (16*i + r);
        for (int t = 0; t < ntok; ++t) {
            acc[t] += v * xr[(t0 + t)*(size_t)(16*I)];
        }
    }

#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
#pragma unroll
        for (int t = 0; t < K2_T16_TOK_BLK; ++t) {
            acc[t] += __shfl_down_sync(0xFFFFFFFFu, acc[t], off, 16);
        }
    }

    if (r == 0) {
        float * yc = y + (16*j + c);
        for (int t = 0; t < ntok; ++t) {
            yc[(t0 + t)*(size_t)(16*J)] = acc[t];
        }
    }
}

// ---------------------------------------------------------------------------
// Fast decode GEMV (design §1/§2): one warp per k-split, 2 out-tiles per block,
// quad window extraction + half2 decode + HFMA2 dot, fp32 fold every 4 k-steps.
// Correctness oracle = the simple kernel above; enable with K2_T16_FAST=1.
// K <= 5 only (the single-funnel quad form is exact there; K=6 uses the simple path).
// ---------------------------------------------------------------------------
struct k2q { uint32_t i0, i2, s2; };

static __device__ __forceinline__ k2q k2_quad_c(int K, int lane, int g) {
    const int nw = 8 * K;
    const int n0 = 8 * lane + 4 * g;
    const int b0 = n0 * K + K - 16 + 256 * K;
    const int b2 = b0 + 3 * K + 16;
    const int i0u = b0 >> 5, i2u = (b2 - 1) >> 5;
    k2q q;
    q.i0 = (uint32_t)(i0u % nw);
    // the funnel window is 32 bits from b0, so its high word is ALWAYS i0+1; using
    // (b2-1)>>5 skips a word when the quad span exceeds 32 bits (K=6: 34 bits) -> wrong digits
    q.i2 = (uint32_t)(i2u % nw);
    q.s2 = (uint32_t)((i2u + 1) * 32 - b2);
    return q;
}

static __device__ __forceinline__ uint32_t k2_sum1(uint32_t u) {
    // unsigned dp4a: 0x6400 + unsigned byte sum (exact mul1)
    return __dp4a(u * 0x83DCD12Du, 0x01010101u, 0x6400u);
}

static __device__ __forceinline__ half2 k2_pair(uint32_t u0, uint32_t u1) {
    const half2 h = __halves2half2(__ushort_as_half((unsigned short)k2_sum1(u0)),
                                   __ushort_as_half((unsigned short)k2_sum1(u1)));
    return __hfma2(h, __halves2half2(__ushort_as_half(0x1EEEu), __ushort_as_half(0x1EEEu)),
                      __halves2half2(__ushort_as_half(0xC931u), __ushort_as_half(0xC931u)));
}

static __device__ __forceinline__ void k2_dec8(int K, uint32_t w, k2q qa, k2q qb, half2 f[4]) {
    uint32_t o0 = __shfl_sync(0xFFFFFFFFu, w, qa.i0);
    uint32_t o2 = __shfl_sync(0xFFFFFFFFu, w, qa.i2);
    const uint32_t fa = __funnelshift_r(o2, o0, qa.s2);
    o0 = __shfl_sync(0xFFFFFFFFu, w, qb.i0);
    o2 = __shfl_sync(0xFFFFFFFFu, w, qb.i2);
    const uint32_t fb = __funnelshift_r(o2, o0, qb.s2);
    const uint32_t m = 0xffffu;
    // funnel places the LAST window of the quad in the low bits (k2_codec.unpack_words:
    // out[2t+1] = w1 & 0xffff, out[2t] = w1 >> K), so digits run from high to low.
    const uint32_t u0 = (fa >> (3 * K)) & m, u1 = (fa >> (2 * K)) & m, u2 = (fa >> K) & m, u3 = fa & m;
    const uint32_t u4 = (fb >> (3 * K)) & m, u5 = (fb >> (2 * K)) & m, u6 = (fb >> K) & m, u7 = fb & m;
    f[0] = k2_pair(u0, u1);   // out m,     in k, k+1
    f[1] = k2_pair(u2, u3);   // out m,     in k+8, k+9
    f[2] = k2_pair(u4, u5);   // out m+8,   in k, k+1
    f[3] = k2_pair(u6, u7);   // out m+8,   in k+8, k+9
}

// wide variant (K > 4): a tile has nw = 8K > 32 words, so each lane holds two words
// (lane and lane+32) and the quad's source word selects the register.
static __device__ __forceinline__ uint32_t k2_shfl2(uint32_t w0, uint32_t w1, uint32_t idx) {
    const uint32_t a = __shfl_sync(0xFFFFFFFFu, w0, (int)(idx & 31u));
    const uint32_t b = __shfl_sync(0xFFFFFFFFu, w1, (int)(idx & 31u));
    return idx < 32u ? a : b;
}

static __device__ __forceinline__ void k2_dec8_w(int K, uint32_t w0, uint32_t w1,
                                                 k2q qa, k2q qb, half2 f[4]) {
    uint32_t o0 = k2_shfl2(w0, w1, qa.i0);
    uint32_t o2 = k2_shfl2(w0, w1, qa.i2);
    const uint32_t fa = __funnelshift_r(o2, o0, qa.s2);
    o0 = k2_shfl2(w0, w1, qb.i0);
    o2 = k2_shfl2(w0, w1, qb.i2);
    const uint32_t fb = __funnelshift_r(o2, o0, qb.s2);
    const uint32_t m = 0xffffu;
    const uint32_t u0 = (fa >> (3 * K)) & m, u1 = (fa >> (2 * K)) & m, u2 = (fa >> K) & m, u3 = fa & m;
    const uint32_t u4 = (fb >> (3 * K)) & m, u5 = (fb >> (2 * K)) & m, u6 = (fb >> K) & m, u7 = fb & m;
    f[0] = k2_pair(u0, u1); f[1] = k2_pair(u2, u3); f[2] = k2_pair(u4, u5); f[3] = k2_pair(u6, u7);
}

// pair-based decode (K >= 6): four per-pair funnels per dec8, each using the reference's own
// (i0, i1, s1) from unpack_words. Validated offline: 256/256 windows for K=2..8.
struct k2pf { uint32_t i0, i1, s1; };

static __device__ __forceinline__ k2pf k2_pf_c(int K, int lane, int j) {
    const int nw = 8 * K;
    const int p  = 4 * lane + j;
    const int b0 = p * 2 * K + K - 16 + 256 * K;
    const int b2 = b0 + K + 16;
    const int i0u = b0 >> 5;
    const int i1u = (b2 - 1) >> 5;
    k2pf q;
    q.i0 = (uint32_t)(i0u % nw);
    q.i1 = (uint32_t)(i1u % nw);
    q.s1 = (uint32_t)(((i1u + 1) * 32 - b2) & 31);
    return q;
}

static __device__ __forceinline__ void k2_dec8_p(int K, uint32_t w0, uint32_t w1,
                                                 k2pf p0, k2pf p1, k2pf p2, k2pf p3, half2 f[4]) {
    const uint32_t m = 0xffffu;
    const k2pf ps[4] = {p0, p1, p2, p3};
    uint32_t d[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t lo = k2_shfl2(w0, w1, ps[j].i1);
        const uint32_t hi = k2_shfl2(w0, w1, ps[j].i0);
        const uint32_t v = ps[j].s1 ? (((hi << (32 - ps[j].s1)) | (lo >> ps[j].s1)) & 0xFFFFFFFFu) : lo;
        d[2 * j]     = (v >> K) & m;
        d[2 * j + 1] = v & m;
    }
    f[0] = k2_pair(d[0], d[1]);
    f[1] = k2_pair(d[2], d[3]);
    f[2] = k2_pair(d[4], d[5]);
    f[3] = k2_pair(d[6], d[7]);
}

template <int K, int NW, bool WIDE = false>
static __global__ __launch_bounds__(32 * NW, 1024 / (32 * NW))
void k2_t16_gemv_kernel(const uint16_t * __restrict__ tr,
                        const float    * __restrict__ x,   // [in] (n == 1)
                        float          * __restrict__ y,   // [out]
                        int I, int J, size_t nb1, size_t nb2) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int p = lane & 3;
    const int mcol = lane >> 2;                 // out column within a tile (0..7)
    const int j0 = 2 * blockIdx.x;
    // k ranges by integer division: works for any I >= 1 (warps may be empty)
    const int ia = (warp * I) / NW, ib = ((warp + 1) * I) / NW;
    const k2q qa = k2_quad_c(K, lane, 0), qb = k2_quad_c(K, lane, 1);
    const k2pf fq0 = k2_pf_c(K, lane, 0), fq1 = k2_pf_c(K, lane, 1);
    const k2pf fq2 = k2_pf_c(K, lane, 2), fq3 = k2_pf_c(K, lane, 3);
    const int nw = 8 * K;

    half2 acc[2][2];
    float af[2][2];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            acc[t][c] = __halves2half2(__float2half(0.f), __float2half(0.f));
            af[t][c] = 0.f;
        }

    // software pipeline: current iteration's 2 tile words live in registers,
    // the next iteration's loads issue before the decode of the current one.
    const char * p0 = (const char *)tr + (size_t)ia * nb2 + (size_t)j0 * nb1;
    const char * p1 = p0 + nb1;
    uint32_t wa = 0u, wb = 0u, wa1 = 0u, wb1 = 0u;
    if (ia < ib) {
        wa = lane < nw ? ((const uint32_t *)p0)[lane] : 0u;
        wb = lane < nw ? ((const uint32_t *)p1)[lane] : 0u;
        if (WIDE) {
            wa1 = (lane + 32) < nw ? ((const uint32_t *)p0)[lane + 32] : 0u;
            wb1 = (lane + 32) < nw ? ((const uint32_t *)p1)[lane + 32] : 0u;
        }
    }

    int cnt = 0;
    for (int i = ia; i < ib; ++i, ++cnt) {
        const uint32_t ca = wa, cb = wb, ca1 = wa1, cb1 = wb1;
        p0 += nb2; p1 += nb2;
        if (i + 1 < ib) {
            wa = lane < nw ? ((const uint32_t *)p0)[lane] : 0u;
            wb = lane < nw ? ((const uint32_t *)p1)[lane] : 0u;
            if (WIDE) {
                wa1 = (lane + 32) < nw ? ((const uint32_t *)p0)[lane + 32] : 0u;
                wb1 = (lane + 32) < nw ? ((const uint32_t *)p1)[lane + 32] : 0u;
            }
        }
        const int k0 = 16 * i;
        const half2 xa = __floats2half2_rn(x[k0 + 2 * p], x[k0 + 2 * p + 1]);
        const half2 xb = __floats2half2_rn(x[k0 + 8 + 2 * p], x[k0 + 9 + 2 * p]);
        half2 f0[4], f1[4];
        if constexpr (K >= 6) {
            k2_dec8_p(K, ca, ca1, fq0, fq1, fq2, fq3, f0);
            k2_dec8_p(K, cb, cb1, fq0, fq1, fq2, fq3, f1);
        } else if (WIDE) {
            k2_dec8_w(K, ca, ca1, qa, qb, f0);
            k2_dec8_w(K, cb, cb1, qa, qb, f1);
        } else {
            k2_dec8(K, ca, qa, qb, f0);
            k2_dec8(K, cb, qa, qb, f1);
        }
        acc[0][0] = __hfma2(f0[0], xa, __hfma2(f0[1], xb, acc[0][0]));
        acc[0][1] = __hfma2(f0[2], xa, __hfma2(f0[3], xb, acc[0][1]));
        acc[1][0] = __hfma2(f1[0], xa, __hfma2(f1[1], xb, acc[1][0]));
        acc[1][1] = __hfma2(f1[2], xa, __hfma2(f1[3], xb, acc[1][1]));
        if ((cnt & 3) == 3) {
#pragma unroll
            for (int t = 0; t < 2; ++t)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    af[t][c] += __low2float(acc[t][c]) + __high2float(acc[t][c]);
                    acc[t][c] = __halves2half2(__float2half(0.f), __float2half(0.f));
                }
        }
    }
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c) af[t][c] += __low2float(acc[t][c]) + __high2float(acc[t][c]);

    // sum the 4 lanes of the quad (same out column, different in rows)
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int c = 0; c < 2; ++c) af[t][c] += __shfl_xor_sync(0xFFFFFFFFu, af[t][c], off);

    __shared__ float red[NW][32];
    if (p == 0) {
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            red[warp][t * 16 + mcol]     = af[t][0];
            red[warp][t * 16 + mcol + 8] = af[t][1];
        }
    }
    __syncthreads();
    if (warp == 0) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) s += red[w][lane];
        const int jout = 16 * (j0 + (lane >> 4)) + (lane & 15);
        y[jout] = s;
    }
}

// Chunked variant: process PF k-steps at a time with the next chunk's tile words
// loaded up front (PF x 2 LDGs in flight per warp). Same math as the kernel above.
template <int K, int NW, int PF>
static __global__ __launch_bounds__(32 * NW, 1024 / (32 * NW))
void k2_t16_gemv4_kernel(const uint16_t * __restrict__ tr,
                         const float    * __restrict__ x,
                         float          * __restrict__ y,
                         int I, int J, size_t nb1, size_t nb2) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int p = lane & 3;
    const int mcol = lane >> 2;
    const int j0 = 2 * blockIdx.x;
    const int ia = (warp * I) / NW, ib = ((warp + 1) * I) / NW;
    const k2q qa = k2_quad_c(K, lane, 0), qb = k2_quad_c(K, lane, 1);
    const int nw = 8 * K;

    half2 acc[2][2];
    float af[2][2];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            acc[t][c] = __halves2half2(__float2half(0.f), __float2half(0.f));
            af[t][c] = 0.f;
        }

    const char * base = (const char *)tr + (size_t)j0 * nb1;
    uint32_t ca[PF], cb[PF];
#pragma unroll
    for (int u = 0; u < PF; ++u) {
        const int i = ia + u;
        ca[u] = cb[u] = 0u;
        if (i < ib && lane < nw) {
            const uint32_t * p0 = (const uint32_t *)(base + (size_t)i * nb2);
            ca[u] = p0[lane];
            cb[u] = ((const uint32_t *)((const char *)p0 + nb1))[lane];
        }
    }

    for (int i0 = ia; i0 < ib; i0 += PF) {
        uint32_t na[PF], nb2v[PF];
#pragma unroll
        for (int u = 0; u < PF; ++u) {
            const int i = i0 + PF + u;
            na[u] = nb2v[u] = 0u;
            if (i < ib && lane < nw) {
                const uint32_t * p0 = (const uint32_t *)(base + (size_t)i * nb2);
                na[u] = p0[lane];
                nb2v[u] = ((const uint32_t *)((const char *)p0 + nb1))[lane];
            }
        }
#pragma unroll
        for (int u = 0; u < PF; ++u) {
            const int i = i0 + u;
            if (i < ib) {
                const int k0 = 16 * i;
                const half2 xa = __floats2half2_rn(x[k0 + 2 * p], x[k0 + 2 * p + 1]);
                const half2 xb = __floats2half2_rn(x[k0 + 8 + 2 * p], x[k0 + 9 + 2 * p]);
                half2 f0[4], f1[4];
                k2_dec8(K, ca[u], qa, qb, f0);
                k2_dec8(K, cb[u], qa, qb, f1);
                acc[0][0] = __hfma2(f0[0], xa, __hfma2(f0[1], xb, acc[0][0]));
                acc[0][1] = __hfma2(f0[2], xa, __hfma2(f0[3], xb, acc[0][1]));
                acc[1][0] = __hfma2(f1[0], xa, __hfma2(f1[1], xb, acc[1][0]));
                acc[1][1] = __hfma2(f1[2], xa, __hfma2(f1[3], xb, acc[1][1]));
            }
        }
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                af[t][c] += __low2float(acc[t][c]) + __high2float(acc[t][c]);
                acc[t][c] = __halves2half2(__float2half(0.f), __float2half(0.f));
            }
#pragma unroll
        for (int u = 0; u < PF; ++u) { ca[u] = na[u]; cb[u] = nb2v[u]; }
    }
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c) af[t][c] += __low2float(acc[t][c]) + __high2float(acc[t][c]);

#pragma unroll
    for (int off = 1; off <= 2; off <<= 1)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int c = 0; c < 2; ++c) af[t][c] += __shfl_xor_sync(0xFFFFFFFFu, af[t][c], off);

    __shared__ float red[NW][32];
    if (p == 0) {
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            red[warp][t * 16 + mcol]     = af[t][0];
            red[warp][t * 16 + mcol + 8] = af[t][1];
        }
    }
    __syncthreads();
    if (warp == 0) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) s += red[w][lane];
        const int jout = 16 * (j0 + (lane >> 4)) + (lane & 15);
        y[jout] = s;
    }
}

// direct-load decode: identical math to k2_dec8/k2_dec8_p, but each lane reads the words it
// needs straight from the tile (L1-resident, 128-192 B) instead of pulling them from other
// lanes' registers with shuffles. Removes the warp-wide shuffle chain from the critical path.
static __device__ __forceinline__ void k2_dec8_d(int K, const uint32_t * __restrict__ tp,
                                                 k2q qa, k2q qb, half2 f[4]) {
    const uint32_t fa = k2_t16_fshr(tp[qa.i2], tp[qa.i0], qa.s2);
    const uint32_t fb = k2_t16_fshr(tp[qb.i2], tp[qb.i0], qb.s2);
    const uint32_t m = 0xffffu;
    const uint32_t u0 = (fa >> (3 * K)) & m, u1 = (fa >> (2 * K)) & m, u2 = (fa >> K) & m, u3 = fa & m;
    const uint32_t u4 = (fb >> (3 * K)) & m, u5 = (fb >> (2 * K)) & m, u6 = (fb >> K) & m, u7 = fb & m;
    f[0] = k2_pair(u0, u1); f[1] = k2_pair(u2, u3);
    f[2] = k2_pair(u4, u5); f[3] = k2_pair(u6, u7);
}

static __device__ __forceinline__ void k2_dec8_pd(int K, const uint32_t * __restrict__ tp,
                                                  k2pf p0, k2pf p1, k2pf p2, k2pf p3, half2 f[4]) {
    const uint32_t m = 0xffffu;
    const k2pf ps[4] = {p0, p1, p2, p3};
    uint32_t d[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t v = ps[j].s1 ? k2_t16_fshr(tp[ps[j].i1], tp[ps[j].i0], ps[j].s1) : tp[ps[j].i1];
        d[2 * j]     = (v >> K) & m;
        d[2 * j + 1] = v & m;
    }
    f[0] = k2_pair(d[0], d[1]); f[1] = k2_pair(d[2], d[3]);
    f[2] = k2_pair(d[4], d[5]); f[3] = k2_pair(d[6], d[7]);
}

// --- baked defaults -------------------------------------------------------------------------
// Validated 2026-10-26 on Qwen2.5-1.5B at 3.34 bpw (llama-bench, RTX 3060): tg128 175.5 +/- 0.4
// t/s with the fast path, 36.7 +/- 0.0 without; perplexity identical to four decimals (8.7898 on
// a 4-chunk probe). During development these were six environment knobs; the validated values are
// constants now. A single hook, K2_T16_SLOW=1, forces the reference (non-fast) path so the two
// paths can still be A/B-tested for exactness.
static bool k2_t16_slow() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("K2_T16_SLOW");
        v = (e && atoi(e) != 0) ? 1 : 0;
    }
    return v != 0;
}

static int k2_t16_direct() {
    return 1;   // direct-load decode was the measured winner (2026-10-23)
}

// multi-tile variant (R output tiles per block): the activation vector is loaded once per
// k-tile and shared by all R tiles, and R independent decode chains fill the latency of the
// shuffle/funnel pipeline. Same math, same layout, no format change.
template <int K, int NW, int R, bool WIDE = false, bool DIRECT = false>
static __global__ __launch_bounds__(32 * NW, 1024 / (32 * NW))
void k2_t16_gemv_r_kernel(const uint16_t * __restrict__ tr,
                          const float    * __restrict__ x,
                          float          * __restrict__ y,
                          int I, int J, size_t nb1, size_t nb2) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int p = lane & 3;
    const int mcol = lane >> 2;
    const int j0 = R * blockIdx.x;
    const int ia = (warp * I) / NW, ib = ((warp + 1) * I) / NW;
    const int nw = 8 * K;
    const k2q qa = k2_quad_c(K, lane, 0), qb = k2_quad_c(K, lane, 1);
    const k2pf fq0 = k2_pf_c(K, lane, 0), fq1 = k2_pf_c(K, lane, 1);
    const k2pf fq2 = k2_pf_c(K, lane, 2), fq3 = k2_pf_c(K, lane, 3);

    half2 acc[R][2];
    float af[R][2];
#pragma unroll
    for (int t = 0; t < R; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            acc[t][c] = __halves2half2(__float2half(0.f), __float2half(0.f));
            af[t][c] = 0.f;
        }

    const char * base = (const char *)tr + (size_t)j0 * nb1;
    uint32_t cw[R][2];
    const bool live = ia < ib;
#pragma unroll
    for (int t = 0; t < R; ++t) {
        cw[t][0] = cw[t][1] = 0u;
        if (live && lane < nw) {
            const uint32_t * tp = (const uint32_t *)(base + (size_t)(j0 + t < J ? t : 0) * nb1 +
                                                     (size_t)ia * nb2);
            cw[t][0] = tp[lane];
            if (WIDE && lane + 32 < nw) cw[t][1] = tp[lane + 32];
        }
    }

    int cnt = 0;
    for (int i = ia; i < ib; ++i, ++cnt) {
        uint32_t cur[R][2];
#pragma unroll
        for (int t = 0; t < R; ++t) { cur[t][0] = cw[t][0]; cur[t][1] = cw[t][1]; }
        if (i + 1 < ib) {
#pragma unroll
            for (int t = 0; t < R; ++t) {
                cw[t][0] = cw[t][1] = 0u;
                if (lane < nw) {
                    const uint32_t * tp = (const uint32_t *)(base +
                        (size_t)(j0 + t < J ? t : 0) * nb1 + (size_t)(i + 1) * nb2);
                    cw[t][0] = tp[lane];
                    if (WIDE && lane + 32 < nw) cw[t][1] = tp[lane + 32];
                }
            }
        }
        const int k0 = 16 * i;
        const half2 xa = __floats2half2_rn(x[k0 + 2 * p], x[k0 + 2 * p + 1]);
        const half2 xb = __floats2half2_rn(x[k0 + 8 + 2 * p], x[k0 + 9 + 2 * p]);
#pragma unroll
        for (int t = 0; t < R; ++t) {
            half2 f[4];
            const uint32_t * tp = (const uint32_t *)(base + (size_t)(j0 + t < J ? t : 0) * nb1 +
                                                     (size_t)i * nb2);
            if constexpr (DIRECT) {
                (void) tp;  // silence unused warnings when DIRECT is off
                if constexpr (K >= 6) k2_dec8_pd(K, tp, fq0, fq1, fq2, fq3, f);
                else                  k2_dec8_d(K, tp, qa, qb, f);
            } else if constexpr (K >= 6) {
                k2_dec8_p(K, cur[t][0], cur[t][1], fq0, fq1, fq2, fq3, f);
            } else if (WIDE) {
                k2_dec8_w(K, cur[t][0], cur[t][1], qa, qb, f);
            } else {
                k2_dec8(K, cur[t][0], qa, qb, f);
            }
            acc[t][0] = __hfma2(f[0], xa, __hfma2(f[1], xb, acc[t][0]));
            acc[t][1] = __hfma2(f[2], xa, __hfma2(f[3], xb, acc[t][1]));
        }
        if ((cnt & 3) == 3) {
#pragma unroll
            for (int t = 0; t < R; ++t)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    af[t][c] += __low2float(acc[t][c]) + __high2float(acc[t][c]);
                    acc[t][c] = __halves2half2(__float2half(0.f), __float2half(0.f));
                }
        }
    }
#pragma unroll
    for (int t = 0; t < R; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c) af[t][c] += __low2float(acc[t][c]) + __high2float(acc[t][c]);

#pragma unroll
    for (int off = 1; off <= 2; off <<= 1)
#pragma unroll
        for (int t = 0; t < R; ++t)
#pragma unroll
            for (int c = 0; c < 2; ++c) af[t][c] += __shfl_xor_sync(0xFFFFFFFFu, af[t][c], off);

    __shared__ float red[NW][16 * R];
    if (p == 0) {
#pragma unroll
        for (int t = 0; t < R; ++t) {
            red[warp][t * 16 + mcol]     = af[t][0];
            red[warp][t * 16 + mcol + 8] = af[t][1];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int u = 0; u < (R + 1) / 2; ++u) {
            const int idx = 32 * u + lane;
            if (idx >= 16 * R) break;            // odd R: the upper lanes have no output
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < NW; ++w) s += red[w][idx];
            if (j0 + idx / 16 < J) y[16 * j0 + idx] = s;
        }
    }
}

static int k2_t16_rows() {
    return 4;   // 4-row multi-tile: 324.1 t/s vs 8-row 287.2 and 2-row 314.3 (2026-10-23)
}

static bool k2_t16_fast_enabled() {
    return !k2_t16_slow();
}

static int k2_t16_nw() {
    return 16;  // 16 warps: 175.5 t/s vs 159.1 at 8 (2026-10-26)
}
static int k2_t16_pf() {
    return 1;   // prefetch depth 1
}

template <int K, int NW>
static void k2_t16_launch(int pf, dim3 grid, dim3 block, cudaStream_t stream,
                          const uint16_t * tw, const float * xf, float * yf,
                          int I, int J, size_t nb1, size_t nb2) {
    const int rows = k2_t16_rows();
    const bool dir = k2_t16_direct() != 0;
    if (rows >= 8 && J >= 128 && J % 8 == 0) {  // rows=8 only where the grid stays large
        const dim3 g(J / 8);
        if (dir) k2_t16_gemv_r_kernel<K, NW, 8, false, true><<<g, block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
        else     k2_t16_gemv_r_kernel<K, NW, 8, false, false><<<g, block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
        return;
    }
    if (rows >= 4 && J >= 64 && J % 4 == 0) {
        const dim3 g(J / 4);
        if (dir) k2_t16_gemv_r_kernel<K, NW, 4, false, true><<<g, block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
        else     k2_t16_gemv_r_kernel<K, NW, 4, false, false><<<g, block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
        return;
    }
    if (pf == 2) {
        k2_t16_gemv4_kernel<K, NW, 2><<<grid, block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
    } else if (pf == 4) {
        k2_t16_gemv4_kernel<K, NW, 4><<<grid, block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
    } else {
        k2_t16_gemv_kernel<K, NW><<<grid, block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
    }
}

template <int K, int NW>
static void k2_t16_launch_wide(dim3 grid, dim3 block, cudaStream_t stream,
                               const uint16_t * tw, const float * xf, float * yf,
                               int I, int J, size_t nb1, size_t nb2) {
    const int rows_req = k2_t16_rows();
    const int rows = (rows_req >= 4 && J >= 64) ? 4 : (J < 16 ? 1 : 2);  // rows=8 worse everywhere
    const bool dir = k2_t16_direct() != 0;
    if (rows >= 8 && J % 8 == 0) {
        if (dir) k2_t16_gemv_r_kernel<K, NW, 8, true, true><<<dim3(J / 8), block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
        else     k2_t16_gemv_r_kernel<K, NW, 8, true, false><<<dim3(J / 8), block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
        return;
    }
    if (rows == 1) {
        const dim3 g1(J);
        if (dir) k2_t16_gemv_r_kernel<K, NW, 1, true, true><<<g1, block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
        else     k2_t16_gemv_r_kernel<K, NW, 1, true, false><<<g1, block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
        return;
    }
    if (rows == 2 && dir) {
        const dim3 g2(J / 2);
        k2_t16_gemv_r_kernel<K, NW, 2, true, true><<<g2, block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
        return;
    }
    if (rows >= 4 && J % 4 == 0) {
        if (dir) k2_t16_gemv_r_kernel<K, NW, 4, true, true><<<dim3(J / 4), block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
        else     k2_t16_gemv_r_kernel<K, NW, 4, true, false><<<dim3(J / 4), block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
        return;
    }
    k2_t16_gemv_kernel<K, NW, true><<<grid, block, 0, stream>>>(tw, xf, yf, I, J, nb1, nb2);
}

template <int K>
static void k2_t16_dispatch(int nw, int pf, dim3 grid, dim3 block, cudaStream_t stream,
                            const uint16_t * tw, const float * xf, float * yf,
                            int I, int J, size_t nb1, size_t nb2) {
    if (nw == 8) {
        k2_t16_launch<K, 8>(pf, grid, block, stream, tw, xf, yf, I, J, nb1, nb2);
    } else if (nw == 16) {
        k2_t16_launch<K, 16>(pf, grid, block, stream, tw, xf, yf, I, J, nb1, nb2);
    } else {
        k2_t16_launch<K, 32>(pf, grid, block, stream, tw, xf, yf, I, J, nb1, nb2);
    }
}

// ---------------------------------------------------------------------------
// Staged variant for K=5/6 (8K = 40/48 words > 32 lanes): tiles go through smem,
// windows through the codec's pair form (2 windows per funnel; exact for any K).
// Pair q of a lane covers windows 8*lane+2q (high half) and +2q+1 (low half).
// ---------------------------------------------------------------------------
struct k2pc { uint32_t i0, i2, s2; };

static __device__ __forceinline__ k2pc k2_pair_c(int K, int lane, int q) {
    const int nw = 8 * K;
    const int t = 4 * lane + q;
    const int b0 = t * 2 * K + K - 16 + 256 * K;
    const int b2 = b0 + K + 16;
    const int i0u = b0 >> 5, i2u = (b2 - 1) >> 5;
    k2pc r;
    r.i0 = (uint32_t)(i0u % nw);
    r.i2 = (uint32_t)(i2u % nw);
    r.s2 = (uint32_t)((i2u + 1) * 32 - b2);
    return r;
}

static __device__ __forceinline__ void k2_dec8_staged(int K, const uint32_t * __restrict__ s,
                                                      const k2pc pc[4], half2 f[4]) {
    const uint32_t m = 0xffffu;
    uint32_t w[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) w[q] = __funnelshift_r(s[pc[q].i2], s[pc[q].i0], pc[q].s2);
#pragma unroll
    for (int q = 0; q < 4; ++q) f[q] = k2_pair((w[q] >> K) & m, w[q] & m);
}

template <int K, int NW>
static __global__ __launch_bounds__(32 * NW, 1024 / (32 * NW))
void k2_t16_gemv_staged_kernel(const uint16_t * __restrict__ tr,
                               const float    * __restrict__ x,
                               float          * __restrict__ y,
                               int I, int J, size_t nb1, size_t nb2) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int p = lane & 3;
    const int mcol = lane >> 2;
    const int j0 = 2 * blockIdx.x;
    const int ia = (warp * I) / NW, ib = ((warp + 1) * I) / NW;
    const int nw = 8 * K;
    k2pc pc[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) pc[q] = k2_pair_c(K, lane, q);

    __shared__ uint32_t sm[NW][2][48];   // per warp: 2 tiles, 8K words max
    uint32_t * s0 = sm[warp][0];
    uint32_t * s1 = sm[warp][1];

    half2 acc[2][2];
    float af[2][2];
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            acc[t][c] = __halves2half2(__float2half(0.f), __float2half(0.f));
            af[t][c] = 0.f;
        }

    // prime the buffer with iteration ia
    if (ia < ib) {
        const char * q0 = (const char *)tr + (size_t)ia * nb2 + (size_t)j0 * nb1;
        if (lane < nw) {
            s0[lane] = ((const uint32_t *)q0)[lane];
            s1[lane] = ((const uint32_t *)(q0 + nb1))[lane];
        }
        if (32 + lane < nw) {
            s0[32 + lane] = ((const uint32_t *)q0)[32 + lane];
            s1[32 + lane] = ((const uint32_t *)(q0 + nb1))[32 + lane];
        }
    }
    __syncwarp();

    for (int i = ia; i < ib; ++i) {
        // issue the next iteration's loads first
        uint32_t v0 = 0u, v1 = 0u, v0b = 0u, v1b = 0u;
        const bool have_next = (i + 1 < ib);
        if (have_next) {
            const char * q0 = (const char *)tr + (size_t)(i + 1) * nb2 + (size_t)j0 * nb1;
            if (lane < nw) {
                v0 = ((const uint32_t *)q0)[lane];
                v1 = ((const uint32_t *)(q0 + nb1))[lane];
            }
            if (32 + lane < nw) {
                v0b = ((const uint32_t *)q0)[32 + lane];
                v1b = ((const uint32_t *)(q0 + nb1))[32 + lane];
            }
        }
        const int k0 = 16 * i;
        const half2 xa = __floats2half2_rn(x[k0 + 2 * p], x[k0 + 2 * p + 1]);
        const half2 xb = __floats2half2_rn(x[k0 + 8 + 2 * p], x[k0 + 9 + 2 * p]);
        half2 f0[4], f1[4];
        k2_dec8_staged(K, s0, pc, f0);
        k2_dec8_staged(K, s1, pc, f1);
        acc[0][0] = __hfma2(f0[0], xa, __hfma2(f0[1], xb, acc[0][0]));
        acc[0][1] = __hfma2(f0[2], xa, __hfma2(f0[3], xb, acc[0][1]));
        acc[1][0] = __hfma2(f1[0], xa, __hfma2(f1[1], xb, acc[1][0]));
        acc[1][1] = __hfma2(f1[2], xa, __hfma2(f1[3], xb, acc[1][1]));
        __syncwarp();
        if (have_next) {
            if (lane < nw) { s0[lane] = v0; s1[lane] = v1; }
            if (32 + lane < nw) { s0[32 + lane] = v0b; s1[32 + lane] = v1b; }
        }
        __syncwarp();
        if (((i - ia) & 3) == 3) {
#pragma unroll
            for (int t = 0; t < 2; ++t)
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    af[t][c] += __low2float(acc[t][c]) + __high2float(acc[t][c]);
                    acc[t][c] = __halves2half2(__float2half(0.f), __float2half(0.f));
                }
        }
    }
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c) af[t][c] += __low2float(acc[t][c]) + __high2float(acc[t][c]);

#pragma unroll
    for (int off = 1; off <= 2; off <<= 1)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int c = 0; c < 2; ++c) af[t][c] += __shfl_xor_sync(0xFFFFFFFFu, af[t][c], off);

    __shared__ float red[NW][32];
    if (p == 0) {
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            red[warp][t * 16 + mcol]     = af[t][0];
            red[warp][t * 16 + mcol + 8] = af[t][1];
        }
    }
    __syncthreads();
    if (warp == 0) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) s += red[w][lane];
        const int jout = 16 * (j0 + (lane >> 4)) + (lane & 15);
        y[jout] = s;
    }
}

// --- MoE helpers -----------------------------------------------------------------
// 128-point Walsh-Hadamard (same as the graph's hadamard: FWHT * 1/sqrt(128)),
// one warp per 128 values, 4 values per lane.
static __device__ __forceinline__ void k2_fwht128(float v[4], int lane) {
#pragma unroll
    for (int h = 1; h <= 2; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 4; i += 2 * h) {
            const float a = v[i], b = v[i + h];
            v[i] = a + b;
            v[i + h] = a - b;
        }
    }
#pragma unroll
    for (int h = 4; h < 128; h <<= 1) {
        const int lp = h >> 2;
        const bool up = ((lane & lp) == 0);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float o = __shfl_xor_sync(0xFFFFFFFFu, v[k], lp);
            v[k] = up ? (v[k] + o) : (o - v[k]);
        }
    }
    const float sc = 0.08838834764831845f;   // 1/sqrt(128)
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] *= sc;
}

static __global__ void k2_moe_prep_kernel(const float * __restrict__ x,
                                          const float * __restrict__ su,      // may be null (f32)
                                          const int32_t * __restrict__ ids,
                                          int64_t ids_nb1,
                                          float * __restrict__ xp,
                                          int in, int n_used, int nwarp,
                                          int64_t x_nb1, int64_t su_nb1) {
    const int gwarp = int((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (gwarp >= nwarp) return;
    const int lane = int(threadIdx.x & 31);
    const int blocks = in >> 7;
    const int blk = gwarp % blocks;
    const int tu = gwarp / blocks;
    const int u = tu % n_used, t = tu / n_used;
    const int32_t e = *(const int32_t *)((const char *) ids + (int64_t) t * ids_nb1 + (int64_t) u * 4);
    const int base = (blk << 7) + (lane << 2);
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = base + k;
        float xv = x[t * x_nb1 + i];
        if (su) {
            xv *= *(const float *)((const char *) su + (int64_t) e * su_nb1 + (int64_t) i * 4);
        }
        v[k] = xv;
    }
    k2_fwht128(v, lane);
#pragma unroll
    for (int k = 0; k < 4; ++k) xp[(int64_t) tu * in + base + k] = v[k];
}

static __global__ void k2_t16_moe_kernel(const uint16_t * __restrict__ tr,
                                         const float    * __restrict__ xp,    // [in, n_used * n]
                                         const int32_t  * __restrict__ ids,
                                         float          * __restrict__ y,
                                         int K, int I, int J, int n_used,
                                         int64_t ids_nb1, size_t nb1, size_t nb2, size_t nb3,
                                         int64_t y_nb1, int64_t y_nb2) {
    const int j    = blockIdx.x;
    const int u    = blockIdx.y;
    const int t    = blockIdx.z;
    const int tid  = threadIdx.x;
    const int c    = tid >> 4;
    const int r    = tid & 15;
    const int q    = 16 * r + c;
    const int tpos = c_k2_t16_inv_perm[q];
    const int b    = tpos * K + K - 16 + 256 * K;
    const int ih   = b >> 5, il = (b + 15) >> 5;
    const int sh   = (il + 1) * 32 - (b + 16);
    const int nw   = 8 * K;

    const int32_t e = *(const int32_t *)((const char *) ids + (int64_t) t * ids_nb1 + (int64_t) u * 4);
    const char * tile0 = (const char *) tr + (size_t) e * nb3 + (size_t) j * nb1;
    const float * xcol = xp + (int64_t)(t * n_used + u) * (16 * I);

    float acc = 0.0f;
    for (int i = 0; i < I; ++i) {
        const uint32_t * p = (const uint32_t *)(tile0 + (size_t) i * nb2);
        const uint32_t w1  = k2_t16_fshr(p[il % nw], p[ih % nw], (uint32_t) sh);
        acc += k2_t16_mul1((uint16_t)(w1 & 0xFFFFu)) * xcol[16 * i + r];
    }
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off, 16);
    }
    if (r == 0) {
        float * ycol = (float *)((char *) y + (int64_t) u * y_nb1 + (int64_t) t * y_nb2);
        ycol[16 * j + c] = acc;
    }
}

static __global__ void k2_moe_post_kernel(float * __restrict__ y,
                                          const float * __restrict__ sv,      // may be null (f32)
                                          const int32_t * __restrict__ ids,
                                          int64_t ids_nb1,
                                          int out, int n_used, int nwarp,
                                          int64_t y_nb1, int64_t y_nb2, int64_t sv_nb1) {
    const int gwarp = int((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (gwarp >= nwarp) return;
    const int lane = int(threadIdx.x & 31);
    const int blocks = out >> 7;
    const int blk = gwarp % blocks;
    const int tu = gwarp / blocks;
    const int u = tu % n_used, t = tu / n_used;
    const int32_t e = *(const int32_t *)((const char *) ids + (int64_t) t * ids_nb1 + (int64_t) u * 4);
    const int base = (blk << 7) + (lane << 2);
    float * ycol = (float *)((char *) y + (int64_t) u * y_nb1 + (int64_t) t * y_nb2);
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = ycol[base + k];
    k2_fwht128(v, lane);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float val = v[k];
        if (sv) {
            val *= *(const float *)((const char *) sv + (int64_t) e * sv_nb1 + (int64_t) (base + k) * 4);
        }
        ycol[base + k] = val;
    }
}

void ggml_cuda_op_t16_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w   = dst->src[0];
    const ggml_tensor * x   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    const ggml_tensor * su  = dst->src[3];
    const ggml_tensor * sv  = dst->src[4];

    GGML_ASSERT(w->type == GGML_TYPE_I16 && ggml_is_contiguous(w));
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ggml_is_contiguous(x));
    GGML_ASSERT(ids->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!su || su->type == GGML_TYPE_F32);
    GGML_ASSERT(!sv || sv->type == GGML_TYPE_F32);

    const int K = (int) (w->ne[0] / 16);
    const int J = (int) w->ne[1];
    const int I = (int) w->ne[2];
    const int E = (int) w->ne[3];
    const int in = 16 * I, out = 16 * J;
    const int n = (int) x->ne[1];
    const int n_used = (int) ids->ne[0];
    GGML_ASSERT(K >= 2 && K <= 6);
    GGML_ASSERT(x->ne[0] == in && dst->ne[0] == out && dst->ne[1] == n_used && dst->ne[2] == n);
    GGML_ASSERT(in % 128 == 0 && out % 128 == 0);

    cudaStream_t stream = ctx.stream();
    const int threads = 256;

    // xp = H(x[:,t] * su_e) per (t,u)
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> xp(pool, (size_t) n * n_used * in);
    const int warps_in = n * n_used * (in / 128);
    k2_moe_prep_kernel<<<(warps_in * 32 + threads - 1) / threads, threads, 0, stream>>>(
            (const float *) x->data,
            su ? (const float *) su->data : nullptr,
            (const int32_t *) ids->data, ids->nb[1],
            xp.get(), in, n_used, warps_in,
            x->nb[1] / sizeof(float), su ? su->nb[1] / 4 : 0);

    // y = w_e^T xp (one device launch; reads ids on the GPU)
    (void) E;
    k2_t16_moe_kernel<<<dim3(J, n_used, n), 256, 0, stream>>>(
            (const uint16_t *) w->data, xp.get(), (const int32_t *) ids->data, (float *) dst->data,
            K, I, J, n_used, ids->nb[1], w->nb[1], w->nb[2], w->nb[3], dst->nb[1], dst->nb[2]);

    // y = sv_e * H(y), in place
    const int warps_out = n * n_used * (out / 128);
    k2_moe_post_kernel<<<(warps_out * 32 + threads - 1) / threads, threads, 0, stream>>>(
            (float *) dst->data,
            sv ? (const float *) sv->data : nullptr,
            (const int32_t *) ids->data, ids->nb[1],
            out, n_used, warps_out,
            dst->nb[1], dst->nb[2], sv ? sv->nb[1] / 4 : 0);

    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_t16_matmul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];

    GGML_ASSERT(w->type == GGML_TYPE_I16 && w->ne[3] == 1);
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ggml_is_contiguous(x));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));

    const int K = (int) (w->ne[0] / 16);
    const int J = (int) w->ne[1];
    const int I = (int) w->ne[2];
    const int n = (int) x->ne[1];
    GGML_ASSERT(K >= 2 && K <= 6);
    GGML_ASSERT((int) w->ne[0] == 16*K);
    GGML_ASSERT(x->ne[0] == 16*I && dst->ne[0] == 16*J && dst->ne[1] == n);

    cudaStream_t stream = ctx.stream();
    if (n == 1 && K >= 2 && K <= 6 && J % 2 == 0 && k2_t16_fast_enabled()) {
        const int nw = k2_t16_nw();
        const int pf = k2_t16_pf();
        const dim3 grid(J / 2);
        const dim3 block(32, nw);
        const uint16_t * tw = (const uint16_t *) w->data;
        const float    * xf = (const float *) x->data;
        float          * yf = (float *) dst->data;
        if (K == 2) {
            k2_t16_dispatch<2>(nw, pf, grid, block, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
        } else if (K == 3) {
            k2_t16_dispatch<3>(nw, pf, grid, block, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
        } else if (K == 4) {
            k2_t16_dispatch<4>(nw, pf, grid, block, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
        } else {
            // wide path (K=5/6): two words per lane; multi-tile when K2_T16_ROWS > 2;
            // K2_T16_NW now applies here too (it used to be hardcoded to 16)
            const int nww = k2_t16_nw();
            const dim3 blkW(32, nww);
            if (K == 5) {
                if (nww == 8)       k2_t16_launch_wide<5, 8>(grid, blkW, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
                else if (nww == 32) k2_t16_launch_wide<5, 32>(grid, blkW, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
                else                k2_t16_launch_wide<5, 16>(grid, blkW, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
            } else {
                if (nww == 8)       k2_t16_launch_wide<6, 8>(grid, blkW, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
                else if (nww == 32) k2_t16_launch_wide<6, 32>(grid, blkW, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
                else                k2_t16_launch_wide<6, 16>(grid, blkW, stream, tw, xf, yf, I, J, w->nb[1], w->nb[2]);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    dim3 grid(J, (n + K2_T16_TOK_BLK - 1) / K2_T16_TOK_BLK);
    k2_t16_matmul_kernel<<<grid, 256, 0, stream>>>(
            (const uint16_t *) w->data, (const float *) x->data, (float *) dst->data,
            K, I, J, n, w->nb[1], w->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}
