// t16-gemv.cuh — the K2 trellis GEMV kernels, templates only (no instantiations).
// Instances live in template-instances/mmvq-instance-t16-k*.cu, one per K, using the fork's
// file layout for this kind of work (a shared template header plus per-K files picked up by the
// existing mmvq-instance*.cu glob, so no CMakeLists change).  The instance body differs from
// iqk_mmvq_templates.cuh's: there an instance supplies a device vec_dot_* that the shared
// templates turn into a kernel, here it is a host launcher that fixes K for k2_t16_dispatch,
// because this is a standalone GEMV rather than an MMVQ specialization.
#pragma once

#include "common.cuh"

// Shared device primitives (moved here so every instance TU sees them).
struct k2q { uint32_t i0, i2, s2; };

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

