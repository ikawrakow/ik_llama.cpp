#include "common.cuh"
#include "ks-encoder.cuh"
#include "../iqk/iqk_quantize.h"

#include <cstring>
#include <mutex>
#include <vector>

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)

// same result as IQ4_KT/IQ3_KT/IQ2_KT/IQ1_KT quantization on the CPU; the _rn intrinsics mirror each rounding and FMA of the GCC -O3 build

constexpr int   kKtBlock  = 32;
constexpr float kKtWeight = 1e-4f;
constexpr float kKtEps2   = 1e-14f;
constexpr int   kKtWarps  = 8;
constexpr int   kKtTries  = 19;

struct kt_bank {
    const int      * offsets;
    const uint16_t * points;
    const float    * values;
    float            mid[8];
    uint32_t         offset;
    int              num_val;
};

struct kt_codebook {
    kt_bank bank[2];
};

static __device__ __forceinline__ int kt_bin5(float x) {
    return x < -48.f ? 0 : x < -16.f ? 1 : x < 16.f ? 2 : x < 48.f ? 3 : 4;
}

// GCC turns nearest_int(a*b) into one FMA with the 12582912 constant
static __device__ __forceinline__ int kt_nearest_int(float a, float b) {
    const int i = __float_as_int(__fmaf_rn(a, b, 12582912.f));
    return (i & 0x007fffff) - 0x00400000;
}

static __device__ __forceinline__ unsigned long long kt_warp_min(unsigned long long best) {
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const unsigned long long other = __shfl_xor_sync(0xffffffff, best, offset);
        best = other < best ? other : best;
    }
    return best;
}

static __device__ int kt_find_best_match_g4(const kt_bank & b, float d, float x, float w, int * error) {
    if (d == 0.f) {
        return 0;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const float xs = __fmul_rn(__fdiv_rn(1.f, d), x);
    const int bin = kt_bin5(xs);
    int u = 0;
    for (int k = 3; k >= 0; --k) {
        u = 5*u + __shfl_sync(0xffffffff, bin, (lane & ~3) + k);
    }
    int result = 0;
    for (int g = 0; g < kKtBlock/4; ++g) {
        const int   c  = __shfl_sync(0xffffffff, u,  4*g);
        const float x0 = __shfl_sync(0xffffffff, xs, 4*g+0);
        const float x1 = __shfl_sync(0xffffffff, xs, 4*g+1);
        const float x2 = __shfl_sync(0xffffffff, xs, 4*g+2);
        const float x3 = __shfl_sync(0xffffffff, xs, 4*g+3);
        const float w0 = __shfl_sync(0xffffffff, w,  4*g+0);
        const float w1 = __shfl_sync(0xffffffff, w,  4*g+1);
        const float w2 = __shfl_sync(0xffffffff, w,  4*g+2);
        const float w3 = __shfl_sync(0xffffffff, w,  4*g+3);
        const int first = b.offsets[c];
        const int n = b.offsets[c+1] - first;
        float best_s = INFINITY;
        int best_p = n;
        for (int p = lane; p < n; p += WARP_SIZE) {
            // recompute the values from the index; loading b.values is L2-bound
            uint32_t v = b.points[first + p] + b.offset;
            float q[4];
            for (int k = 0; k < 4; ++k) {
                v *= 0xCBAC1FED;
                q[k] = ggml_cuda_dp4a(v & 0x3f3f3f3f, 0x01010101, -126);
            }
            const float d0 = __fsub_rn(q[0], x0);
            const float d1 = __fsub_rn(q[1], x1);
            const float d2 = __fsub_rn(q[2], x2);
            const float d3 = __fsub_rn(q[3], x3);
            const float t0 = __fmul_rn(w0, __fmul_rn(d0, d0));
            const float t1 = __fmul_rn(w1, __fmul_rn(d1, d1));
            const float t2 = __fmul_rn(w2, __fmul_rn(d2, d2));
            const float t3 = __fmul_rn(w3, __fmul_rn(d3, d3));
            // summation order of hsum_float_4x8
            const float s = __fadd_rn(__fadd_rn(t0, t2), __fadd_rn(t1, t3));
            if (s < best_s) {
                best_s = s;
                best_p = p;
            }
        }
        // AVX2 tie order: score, then lane 4*(p&1) + ((p&7)>>1), then p/8
        const int o = best_p & 7;
        const unsigned long long best = kt_warp_min(best_p < n ?
                ((unsigned long long)__float_as_uint(best_s) << 32) | ((unsigned)(4*(o & 1) + (o >> 1)) << 24) | (unsigned)(best_p >> 3) : ~0ull);
        if (best == ~0ull) {
            *error = 2;
            continue;
        }
        const int l = (best >> 24) & 0xff;
        const int p = 8*int(best & 0xffffff) + 2*(l & 3) + (l >> 2);
        if (lane/4 == g) {
            result = b.points[first + p];
        }
    }
    return result;
}

template <bool kAbs>
static __device__ int kt_find_best_match_g8(const kt_bank & b, float d, float x, float w, int * error) {
    if (d == 0.f) {
        return 0;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const float xs = __fmul_rn(__fdiv_rn(1.f, d), x);
    const uint32_t route = __ballot_sync(0xffffffff, xs > b.mid[lane & 7]);
    int result = 0;
    for (int g = 0; g < kKtBlock/8; ++g) {
        const int c = (route >> 8*g) & 0xff;
        float xg[8], wg[8];
        for (int k = 0; k < 8; ++k) {
            xg[k] = __shfl_sync(0xffffffff, xs, 8*g + k);
            wg[k] = __shfl_sync(0xffffffff, w,  8*g + k);
        }
        const int first = b.offsets[c];
        int n = b.offsets[c+1] - first;
        // the CPU searches the whole codebook for an empty cluster
        const bool full = n == 0;
        if (full) {
            n = b.num_val;
        }
        float best_s = INFINITY;
        int best_p = n;
        for (int p = lane; p < n; p += WARP_SIZE) {
            uint32_t v = (full ? p : b.points[first + p]) + b.offset;
            float t[8];
            for (int k = 0; k < 8; ++k) {
                v *= 0xCBAC1FED;
                const int vk = ggml_cuda_dp4a(v & 0x3f3f3f3f, 0x01010101, -126);
                const float dk = __fsub_rn(float(kAbs ? abs(vk) : vk), xg[k]);
                t[k] = __fmul_rn(wg[k], __fmul_rn(dk, dk));
            }
            // hsum_float_8x8 order; ties: score, p%8, p/8
            const float s = __fadd_rn(__fadd_rn(__fadd_rn(t[0], t[4]), __fadd_rn(t[2], t[6])),
                                      __fadd_rn(__fadd_rn(t[1], t[5]), __fadd_rn(t[3], t[7])));
            if (s < best_s) {
                best_s = s;
                best_p = p;
            }
        }
        const unsigned long long best = kt_warp_min(best_p < n ?
                ((unsigned long long)__float_as_uint(best_s) << 32) | ((unsigned)(best_p & 7) << 24) | (unsigned)(best_p >> 3) : ~0ull);
        if (best == ~0ull) {
            *error = 2;
            continue;
        }
        const int p = 8*int(best & 0xffffff) + int((best >> 24) & 0xff);
        if (lane/8 == g) {
            result = full ? p : b.points[first + p];
        }
    }
    return result;
}

static __device__ float2 kt_find_best_scale(float x, float w, float q) {
    const int l = threadIdx.x & 7;
    const float qw = __fmul_rn(q, w);
    float ax = 0.f, a2 = 0.f;
    for (int r = 0; r < 4; ++r) {
        const float qwr = __shfl_sync(0xffffffff, qw, l + 8*r);
        const float xr  = __shfl_sync(0xffffffff, x,  l + 8*r);
        const float qr  = __shfl_sync(0xffffffff, q,  l + 8*r);
        ax = __fmaf_rn(qwr, xr, ax);
        a2 = __fmaf_rn(qwr, qr, a2);
    }
    const float hx = __fadd_rn(ax, __shfl_sync(0xffffffff, ax, (l & 3) + 4));
    const float h2 = __fadd_rn(a2, __shfl_sync(0xffffffff, a2, (l & 3) + 4));
    const float sumqx = __fadd_rn(__fadd_rn(__shfl_sync(0xffffffff, hx, 0), __shfl_sync(0xffffffff, hx, 2)),
                                  __fadd_rn(__shfl_sync(0xffffffff, hx, 1), __shfl_sync(0xffffffff, hx, 3)));
    const float sumq2 = __fadd_rn(__fadd_rn(__shfl_sync(0xffffffff, h2, 0), __shfl_sync(0xffffffff, h2, 2)),
                                  __fadd_rn(__shfl_sync(0xffffffff, h2, 1), __shfl_sync(0xffffffff, h2, 3)));
    return sumq2 > 0.f ? make_float2(__fdiv_rn(sumqx, sumq2), __fdiv_rn(__fmul_rn(sumqx, sumqx), sumq2)) : make_float2(0.f, 0.f);
}

static __device__ __forceinline__ float kt_row_amax(const float * sb_amax, int nb) {
    const int lane = threadIdx.x % WARP_SIZE;
    float amax = 0.f;
    for (int i = lane; i < nb; i += WARP_SIZE) amax = fmaxf(amax, sb_amax[i]);
    return warp_reduce_max(amax);
}

static __global__ void k_kt_weights(const float * __restrict__ x, const float * __restrict__ imatrix, float * __restrict__ w,
        float * __restrict__ sb_amax, int64_t nsuper, int n_per_row, int np, int64_t row0, int64_t nrows) {
    const int64_t i = blockIdx.x*int64_t(blockDim.x) + threadIdx.x;
    if (i >= nsuper) {
        return;
    }
    const int nb = np/QK_K;
    const int ibl = i % nb;
    // one imatrix slice per expert, nrows rows each
    if (imatrix) {
        imatrix += (row0 + i/nb)/nrows*np;
    }
    const float * xb = x + i*QK_K;
    float * wb = w + i*QK_K;
    const int n = min(QK_K, n_per_row - ibl*QK_K);
    for (int j = n; j < QK_K; ++j) wb[j] = 0.f;
    float sumx2 = 0.f, amax = 0.f;
    for (int j = 0; j < n; ++j) {
        sumx2 = __fadd_rn(sumx2, __fmul_rn(xb[j], xb[j]));
        amax = fmaxf(amax, fabsf(xb[j]));
    }
    sb_amax[i] = amax;
    if (sumx2 < __fmul_rn(kKtEps2, float(n))) {
        for (int j = 0; j < n; ++j) wb[j] = kKtWeight;
        return;
    }
    const float sigma2 = __fdiv_rn(__fadd_rn(sumx2, sumx2), float(n));
    if (imatrix) {
        for (int ib = 0; ib < n/kKtBlock; ++ib) {
            const float * qw = imatrix + ibl*QK_K + ib*kKtBlock;
            const float * xv = xb + ib*kKtBlock;
            float * wv = wb + ib*kKtBlock;
            float sumwx = 0.f, sumw2 = 0.f, sumx2b = 0.f;
            for (int j = 0; j < kKtBlock; ++j) {
                wv[j] = __fmul_rn(qw[j], __fsqrt_rn(__fmaf_rn(xv[j], xv[j], sigma2)));
                sumwx  = __fmaf_rn(fabsf(xv[j]), wv[j], sumwx);
                sumw2  = __fmaf_rn(wv[j], wv[j], sumw2);
                sumx2b = __fmaf_rn(xv[j], xv[j], sumx2b);
            }
            if (sumx2b < kKtEps2 || sumw2 < kKtEps2 || sumwx < kKtEps2) {
                for (int j = 0; j < kKtBlock; ++j) wv[j] = kKtWeight;
            }
        }
    } else {
        const float s = __fmul_rn(0.25f, sigma2);
        for (int j = 0; j < n; ++j) wb[j] = __fmaf_rn(xb[j], xb[j], s);
    }
}

static __global__ void k_iq4kt_first_pass(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ sb_amax,
        float * __restrict__ scales, uint8_t * __restrict__ banks, int * __restrict__ error, int64_t nblocks, int np, const kt_codebook cb) {
    const int64_t ib = (blockIdx.x*int64_t(blockDim.x) + threadIdx.x)/WARP_SIZE;
    if (ib >= nblocks) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const float xv = x[ib*kKtBlock + lane];
    const float wv = w[ib*kKtBlock + lane];
    const float amax = warp_reduce_max(fabsf(xv));
    if (amax < 1e-16f) {
        if (lane == 0) {
            scales[ib] = 0.f;
            banks[ib] = 0;
        }
        return;
    }
    const int nb = np/QK_K;
    const float amax_row = kt_row_amax(sb_amax + ib/(np/kKtBlock)*nb, nb);
    const float scale_0 = fmaxf(90.f, __fdiv_rn(__fmul_rn(124.f, amax), amax_row));
    float best = 0.f, scale = 0.f;
    int bank = 0;
    for (int b = 0; b < 2; ++b) {
        if (b == 1) {
            const int idx = kt_find_best_match_g4(cb.bank[1], scale, xv, wv, error);
            const float2 r = kt_find_best_scale(xv, wv, cb.bank[1].values[4*idx + (lane & 3)]);
            if (r.y > best) { best = r.y; scale = r.x; bank = 1; }
        }
        for (int itry = -2; itry <= 2; ++itry) {
            const float d = __fdiv_rn(amax, __fadd_rn(8.f*itry, scale_0));
            for (int sign = 0; sign < 2; ++sign) {
                const int idx = kt_find_best_match_g4(cb.bank[b], sign ? -d : d, xv, wv, error);
                const float2 r = kt_find_best_scale(xv, wv, cb.bank[b].values[4*idx + (lane & 3)]);
                if (r.y > best) { best = r.y; scale = r.x; bank = b; }
            }
        }
    }
    if (lane == 0) {
        scales[ib] = scale;
        banks[ib] = bank;
    }
}

static __global__ void k_iq4kt_row_scale(const float * __restrict__ scales, const float * __restrict__ sb_amax, float * __restrict__ row_d,
        char * __restrict__ y, int64_t nrows, int np, size_t row_bytes, float fudge) {
    const int64_t row = blockIdx.x*int64_t(blockDim.x) + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    const int nb = np/QK_K;
    float amax_row = 0.f;
    for (int i = 0; i < nb; ++i) amax_row = fmaxf(amax_row, sb_amax[row*nb + i]);
    float * dptr = (float *)(y + row*row_bytes);
    if (amax_row == 0.f) {
        *dptr = 0.f;
        row_d[row] = 0.f;
        return;
    }
    const int nb32 = np/kKtBlock;
    float amax_scale = 0.f, max_scale = 0.f;
    for (int i = 0; i < nb32; ++i) {
        const float s = scales[row*nb32 + i];
        if (fabsf(s) > amax_scale) {
            amax_scale = fabsf(s);
            max_scale = s;
        }
    }
    const float d = __fmul_rn(max_scale, -0.015625f);
    *dptr = __fmul_rn(d, fudge);
    row_d[row] = d;
}

static __global__ void k_iq4kt_final_pass(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ scales,
        const uint8_t * __restrict__ banks, const float * __restrict__ row_d, float * __restrict__ qout, char * __restrict__ y,
        int * __restrict__ error, int64_t nsuper, int np, size_t row_bytes, const kt_codebook cb) {
    __shared__ int sidx[kKtWarps][QK_K/4];
    const int64_t isb = (blockIdx.x*int64_t(blockDim.x) + threadIdx.x)/WARP_SIZE;
    if (isb >= nsuper) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    const int nb = np/QK_K;
    const int64_t row = isb/nb;
    const int ibl = isb % nb;
    const float d = row_d[row];
    int my_ls = 0, my_bank = 0;
    for (int ib = 0; ib < QK_K/kKtBlock; ++ib) {
        const int64_t i32 = isb*(QK_K/kKtBlock) + ib;
        const int bank = banks[i32];
        int ls = 0, idx = 0;
        if (d != 0.f) {
            ls = min(kt_nearest_int(__fdiv_rn(1.f, d), scales[i32]), 63);
            const float dl = __fmul_rn(float(ls), d);
            idx = kt_find_best_match_g4(cb.bank[bank], dl, x[i32*kKtBlock + lane], w[i32*kKtBlock + lane], error);
            qout[i32*kKtBlock + lane] = __fmul_rn(cb.bank[bank].values[4*idx + (lane & 3)], float(ls));
        }
        if ((lane & 3) == 0) sidx[wid][8*ib + lane/4] = idx;
        if (lane == ib) {
            my_ls = ls;
            my_bank = bank;
        }
    }
    __syncwarp();
    const int * gi = sidx[wid];
    uint32_t word = 0;
    if (lane < 8) {
        if (d == 0.f) {
            word = my_bank;
        } else {
            word = uint8_t(((my_ls + 64) << 1) | my_bank);
            for (int j = 0; j < 8; ++j) word |= uint32_t(gi[8*lane + j] >> 12) << (8 + 3*j);
        }
    } else if (d != 0.f && lane < 24) {
        const int i = 4*(lane - 8);
        for (int k = 0; k < 4; ++k) word |= uint32_t(gi[i + k] & 255) << 8*k;
    } else if (d != 0.f) {
        const int i = 4*(lane - 24);
        for (int k = 0; k < 4; ++k) word |= uint32_t(((gi[i + k] >> 8) & 0xf) | (((gi[i + k + 32] >> 8) & 0xf) << 4)) << 8*k;
    }
    uint32_t * yb = (uint32_t *)(y + row*row_bytes + sizeof(float) + ibl*sizeof(block_iq4_kt));
    yb[lane] = word;
}

static __global__ void k_iq3kt_first_pass(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ sb_amax,
        float * __restrict__ scales, float * __restrict__ qtmp, int * __restrict__ error, int64_t nblocks, int np, const kt_bank b) {
    const int64_t ib = (blockIdx.x*int64_t(blockDim.x) + threadIdx.x)/WARP_SIZE;
    if (ib >= nblocks) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const float xv = fabsf(x[ib*kKtBlock + lane]);
    const float wv = w[ib*kKtBlock + lane];
    const float amax = warp_reduce_max(xv);
    if (amax < 1e-16f) {
        if (lane == 0) scales[ib] = 0.f;
        qtmp[ib*kKtBlock + lane] = 0.f;
        return;
    }
    const int nb = np/QK_K;
    const float amax_row = kt_row_amax(sb_amax + ib/(np/kKtBlock)*nb, nb);
    const float scale_0 = fmaxf(84.f, __fdiv_rn(__fmul_rn(123.f, amax), amax_row));
    float best = 0.f, scale = 0.f, qbest = 0.f;
    for (int itry = -3; itry <= 3; ++itry) {
        const int idx = kt_find_best_match_g8<true>(b, __fdiv_rn(amax, __fadd_rn(scale_0, 8.f*itry)), xv, wv, error);
        const float q = b.values[8*idx + (lane & 7)];
        const float2 r = kt_find_best_scale(xv, wv, q);
        if (r.y > best) { best = r.y; scale = r.x; qbest = q; }
    }
    if (lane == 0) {
        if (best == 0.f) *error = 1;
        scales[ib] = scale;
    }
    qtmp[ib*kKtBlock + lane] = qbest;
}

static __device__ __forceinline__ float kt_max_scale(const float * scales, int nb32) {
    float amax_scale = 0.f, max_scale = 0.f;
    for (int i = 0; i < nb32; ++i) {
        if (fabsf(scales[i]) > amax_scale) {
            amax_scale = fabsf(scales[i]);
            max_scale = scales[i];
        }
    }
    return max_scale;
}

static __global__ void k_iq3kt_row_try(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ scales,
        const float * __restrict__ qtmp, float2 * __restrict__ sums, int64_t nrows, int np) {
    const int64_t i = blockIdx.x*int64_t(blockDim.x) + threadIdx.x;
    if (i >= nrows*kKtTries) {
        return;
    }
    const int64_t row = i/kKtTries;
    const int itry = int(i % kKtTries) - kKtTries/2;
    const int nb32 = np/kKtBlock;
    const float max_scale = kt_max_scale(scales + row*nb32, nb32);
    float sumqx = 0.f, sumq2 = 0.f;
    if (max_scale > 0.f) {
        const float id = __fdiv_rn(__fmaf_rn(float(itry), 0.2f, 15.f), max_scale);
        for (int ib = 0; ib < nb32; ++ib) {
            const float dl = float(max(0, min(15, kt_nearest_int(id, scales[row*nb32 + ib]))));
            const int64_t j0 = row*np + ib*kKtBlock;
            // separate multiply and add here, as in the AVX2 CPU loop
            for (int j = 0; j < kKtBlock; ++j) {
                const float q = __fmul_rn(dl, qtmp[j0 + j]);
                sumqx = __fadd_rn(sumqx, __fmul_rn(__fmul_rn(w[j0 + j], fabsf(x[j0 + j])), q));
                sumq2 = __fadd_rn(sumq2, __fmul_rn(__fmul_rn(w[j0 + j], q), q));
            }
        }
    }
    sums[i] = make_float2(sumqx, sumq2);
}

static __global__ void k_iq3kt_row_scale(const float * __restrict__ scales, const float * __restrict__ sb_amax, const float2 * __restrict__ sums,
        float * __restrict__ row_d, uint32_t * __restrict__ packed, char * __restrict__ y, int64_t nrows, int np, size_t row_bytes, float fudge) {
    const int64_t row = blockIdx.x*int64_t(blockDim.x) + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    const int nb = np/QK_K;
    const int nb32 = np/kKtBlock;
    float amax_row = 0.f;
    for (int i = 0; i < nb; ++i) amax_row = fmaxf(amax_row, sb_amax[row*nb + i]);
    float * dptr = (float *)(y + row*row_bytes);
    if (amax_row == 0.f) {
        *dptr = 0.f;
        row_d[row] = 0.f;
        return;
    }
    const float * rs = scales + row*nb32;
    float d = __fdiv_rn(kt_max_scale(rs, nb32), 15.f), best = 0.f;
    for (int t = 0; t < kKtTries; ++t) {
        const float2 s = sums[row*kKtTries + t];
        if (s.y > 0.f && __fmul_rn(s.x, s.x) > __fmul_rn(best, s.y)) {
            d = __fdiv_rn(s.x, s.y);
            best = __fmul_rn(d, s.x);
        }
    }
    const float id = d != 0.f ? __fdiv_rn(1.f, d) : 0.f;
    for (int ibl = 0; ibl < nb; ++ibl) {
        uint32_t p = 0;
        for (int ib = 0; ib < 4; ++ib) {
            const int ls1 = max(0, min(15, kt_nearest_int(id, rs[8*ibl + ib])));
            const int ls2 = max(0, min(15, kt_nearest_int(id, rs[8*ibl + ib + 4])));
            p |= uint32_t(ls1 | (ls2 << 4)) << 8*ib;
        }
        packed[row*nb + ibl] = p;
    }
    *dptr = __fmul_rn(d, fudge);
    row_d[row] = d;
}

static __global__ void k_iq3kt_final_pass(const float * __restrict__ x, const float * __restrict__ w, const uint32_t * __restrict__ packed,
        const float * __restrict__ sb_amax, const float * __restrict__ row_d, float * __restrict__ qout, char * __restrict__ y,
        int * __restrict__ error, int64_t nsuper, int np, size_t row_bytes, const kt_bank b) {
    __shared__ int sidx[kKtWarps][QK_K/8];
    const int64_t isb = (blockIdx.x*int64_t(blockDim.x) + threadIdx.x)/WARP_SIZE;
    if (isb >= nsuper) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    const int nb = np/QK_K;
    const int64_t row = isb/nb;
    const int ibl = isb % nb;
    uint32_t * yb = (uint32_t *)(y + row*row_bytes + sizeof(float) + ibl*sizeof(block_iq3_kt));
    constexpr int kWords = sizeof(block_iq3_kt)/4;
    if (kt_row_amax(sb_amax + row*nb, nb) == 0.f) {
        if (lane < kWords) yb[lane] = 0;
        return;
    }
    const float d = row_d[row];
    const uint32_t p = packed[isb];
    uint32_t signs = 0;
    for (int ib = 0; ib < QK_K/kKtBlock; ++ib) {
        const int64_t e = isb*QK_K + ib*kKtBlock + lane;
        const float xv = x[e];
        if (xv < 0.f) signs |= 1u << ib;
        const int ls = (p >> (8*(ib%4) + 4*(ib/4))) & 0xf;
        const int idx = kt_find_best_match_g8<true>(b, __fmul_rn(d, float(ls)), fabsf(xv), w[e], error);
        qout[e] = __fmul_rn(b.values[8*idx + (lane & 7)], float(ls));
        if ((lane & 7) == 0) sidx[wid][4*ib + lane/8] = idx;
    }
    __syncwarp();
    uint32_t qh = 0;
    for (int k = 0; k < 4; ++k) qh |= (__shfl_sync(0xffffffff, signs, (4*(lane - 17) + k) & 31) & 0xff) << 8*k;
    const int * gi = sidx[wid];
    if (lane == 0) {
        yb[0] = p;
    } else if (lane <= 16) {
        yb[lane] = uint32_t(gi[2*(lane - 1)]) | (uint32_t(gi[2*(lane - 1) + 1]) << 16);
    } else if (lane < kWords) {
        yb[lane] = qh;
    }
}

template <ggml_type type>
static __global__ void k_iq2kt_first_pass(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ sb_amax,
        float * __restrict__ scales, int * __restrict__ idx, float * __restrict__ qtmp, int * __restrict__ error, int np,
        const kt_bank b) {
    const int64_t ib = (blockIdx.x*int64_t(blockDim.x) + threadIdx.x)/WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const float xv = x[ib*kKtBlock + lane];
    const float wv = w[ib*kKtBlock + lane];
    const float amax = warp_reduce_max(fabsf(xv));
    float scale = 0.f;
    int best = 0;
    float q = b.values[lane & 7];
    if (amax >= 1e-16f) {
        const int nb = np/QK_K;
        const float amax_row = kt_row_amax(sb_amax + ib/(np/kKtBlock)*nb, nb);
        float scale_0 = fmaxf(90.f, __fdiv_rn(__fmul_rn(124.f, amax), amax_row));
        float score = 0.f;
        for (int itry = 0; itry < (type == GGML_TYPE_IQ1_KT ? 2 : 1); ++itry) {
            if (itry) {
                scale_0 = __fsub_rn(scale_0, 8.f);
            }
            const float d = __fdiv_rn(amax, scale_0);
            const int ip = kt_find_best_match_g8<false>(b, d, xv, wv, error);
            const float qp = b.values[8*ip + (lane & 7)];
            const float2 rp = kt_find_best_scale(xv, wv, qp);
            const int im = kt_find_best_match_g8<false>(b, -d, xv, wv, error);
            const float qm = b.values[8*im + (lane & 7)];
            const float2 rm = kt_find_best_scale(xv, wv, qm);
            if (itry && !(rp.y > score || rm.y > score)) {
                continue;
            }
            const bool plus = rp.y > rm.y;
            scale = plus ? rp.x : rm.x;
            score = plus ? rp.y : rm.y;
            best  = plus ? ip : im;
            q     = plus ? qp : qm;
        }
    }
    if (lane == 0) {
        scales[ib] = scale;
    }
    if ((lane & 7) == 0) {
        idx[ib*(kKtBlock/8) + lane/8] = best;
    }
    qtmp[ib*kKtBlock + lane] = q;
}

static __global__ void k_iq2kt_row_try(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ scales,
        const float * __restrict__ qtmp, const int8_t * __restrict__ index, float2 * __restrict__ sums, int64_t nrows, int np) {
    const int64_t i = blockIdx.x*int64_t(blockDim.x) + threadIdx.x;
    if (i >= nrows*kKtTries) {
        return;
    }
    const int64_t row = i/kKtTries;
    const int itry = int(i % kKtTries) - kKtTries/2;
    const int nb32 = np/kKtBlock;
    const float max_scale = kt_max_scale(scales + row*nb32, nb32);
    float sumqx = 0.f, sumq2 = 0.f;
    if (max_scale != 0.f) {
        const float id = __fdiv_rn(float(itry + iq4k_values[0]), max_scale);
        for (int ib = 0; ib < nb32; ++ib) {
            const float dl = iq4k_values[ks_best_index<GGML_TYPE_IQ4_KS>(iq4k_values, index, __fmul_rn(id, scales[row*nb32 + ib]))];
            const int64_t j0 = row*np + ib*kKtBlock;
            for (int j = 0; j < kKtBlock; ++j) {
                const float q = __fmul_rn(dl, qtmp[j0 + j]);
                sumqx = __fmaf_rn(__fmul_rn(w[j0 + j], x[j0 + j]), q, sumqx);
                sumq2 = __fmaf_rn(__fmul_rn(w[j0 + j], q), q, sumq2);
            }
        }
    }
    sums[i] = make_float2(sumqx, sumq2);
}

static __global__ void k_iq2kt_row_scale(const float * __restrict__ scales, const float2 * __restrict__ sums, const int8_t * __restrict__ index,
        float * __restrict__ row_d, uint32_t * __restrict__ packed, char * __restrict__ y, int64_t nrows, int np, size_t row_bytes, float fudge) {
    const int64_t row = blockIdx.x*int64_t(blockDim.x) + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    const int nb = np/QK_K;
    const int nb32 = np/kKtBlock;
    const float * rs = scales + row*nb32;
    const float max_scale = kt_max_scale(rs, nb32);
    float * dptr = (float *)(y + row*row_bytes);
    if (max_scale == 0.f) {
        for (int ibl = 0; ibl < nb; ++ibl) packed[row*nb + ibl] = 0;
        *dptr = 0.f;
        row_d[row] = 0.f;
        return;
    }
    float d = __fdiv_rn(max_scale, float(iq4k_values[0])), best = 0.f;
    for (int t = 0; t < kKtTries; ++t) {
        const float2 s = sums[row*kKtTries + t];
        if (s.y > 0.f && __fmul_rn(s.x, s.x) > __fmul_rn(best, s.y)) {
            d = __fdiv_rn(s.x, s.y);
            best = __fmul_rn(d, s.x);
        }
    }
    const float id = d != 0.f ? __fdiv_rn(1.f, d) : 0.f;
    for (int ibl = 0; ibl < nb; ++ibl) {
        uint32_t p = 0;
        for (int ib = 0; ib < 4; ++ib) {
            const int ls1 = ks_best_index<GGML_TYPE_IQ4_KS>(iq4k_values, index, __fmul_rn(id, rs[8*ibl + ib]));
            const int ls2 = ks_best_index<GGML_TYPE_IQ4_KS>(iq4k_values, index, __fmul_rn(id, rs[8*ibl + ib + 4]));
            p |= uint32_t(ls1 | (ls2 << 4)) << 8*ib;
        }
        packed[row*nb + ibl] = p;
    }
    *dptr = __fmul_rn(d, fudge);
    row_d[row] = d;
}

template <ggml_type type>
static __global__ void k_iq2kt_final_pass(const float * __restrict__ x, const float * __restrict__ w, const int * __restrict__ idx,
        const float * qtmp, const uint32_t * __restrict__ packed, const float * __restrict__ row_d, float * qout,
        char * __restrict__ y, int * __restrict__ error, int64_t nsuper, int np, size_t row_bytes, const kt_bank b) {
    __shared__ int sidx[kKtWarps][QK_K/8];
    const int64_t isb = (blockIdx.x*int64_t(blockDim.x) + threadIdx.x)/WARP_SIZE;
    if (isb >= nsuper) {
        return;
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    const int nb = np/QK_K;
    const int64_t row = isb/nb;
    const int ibl = isb % nb;
    const float d = row_d[row];
    const uint32_t p = packed[isb];
    for (int ib = 0; ib < QK_K/kKtBlock; ++ib) {
        const int64_t e = isb*QK_K + ib*kKtBlock + lane;
        int iout = 0;
        float qo = 0.f;
        if (d != 0.f) {
            const int ls = iq4k_values[(p >> (8*(ib%4) + 4*(ib/4))) & 0xf];
            const float dl = __fmul_rn(d, float(ls));
            const float xv = x[e], wv = w[e];
            const int inew = kt_find_best_match_g8<false>(b, dl, xv, wv, error);
            const float q1 = qtmp[e];
            const float q2 = b.values[8*inew + (lane & 7)];
            const float diff1 = __fmaf_rn(-dl, q1, xv);
            const float diff2 = __fmaf_rn(-dl, q2, xv);
            const float t1 = __fmul_rn(wv, diff1), t2 = __fmul_rn(wv, diff2);
            float mse1 = 0.f, mse2 = 0.f;
            for (int k = 0; k < kKtBlock; ++k) {
                mse1 = __fmaf_rn(__shfl_sync(0xffffffff, t1, k), __shfl_sync(0xffffffff, diff1, k), mse1);
                mse2 = __fmaf_rn(__shfl_sync(0xffffffff, t2, k), __shfl_sync(0xffffffff, diff2, k), mse2);
            }
            const bool prev = mse1 < mse2;
            iout = prev ? idx[(isb*(QK_K/kKtBlock) + ib)*(kKtBlock/8) + lane/8] : inew;
            qo = __fmul_rn(prev ? q1 : q2, float(ls));
        }
        qout[e] = qo;
        if ((lane & 7) == 0) sidx[wid][4*ib + lane/8] = iout;
    }
    __syncwarp();
    const int * gi = sidx[wid];
    uint32_t * yb = (uint32_t *)(y + row*row_bytes + sizeof(float));
    if (type == GGML_TYPE_IQ2_KT) {
        yb += ibl*(sizeof(block_iq2_kt)/4);
        if (lane == 0) {
            yb[0] = p;
        } else if (lane <= 16) {
            yb[lane] = uint32_t(gi[2*(lane - 1)]) | (uint32_t(gi[2*(lane - 1) + 1]) << 16);
        }
    } else {
        yb += ibl*(sizeof(block_iq1_kt)/4);
        uint32_t word = 0;
        for (int k = 0; k < 4; ++k) {
            uint32_t v = 0;
            if (lane < 2) {
                const int ib = 4*lane + k;
                v = (p >> (8*(ib%4) + 4*(ib/4))) & 0xf;
                for (int j = 0; j < 4; ++j) v |= uint32_t(gi[4*ib + j] >> 12) << (4 + j);
            } else if (lane < 10) {
                v = gi[4*(lane - 2) + k] & 0xff;
            } else if (lane < 14) {
                const int m = 4*(lane - 10) + k;
                v = ((gi[m] >> 8) & 0xf) | (((gi[m + 16] >> 8) & 0xf) << 4);
            }
            word |= v << 8*k;
        }
        if (lane < 14) {
            yb[lane] = word;
        }
    }
}

template <bool kAbs>
static __global__ void k_kt_refit(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ q,
        const float * __restrict__ row_d, char * __restrict__ y, int64_t nrows, int np, size_t row_bytes, float fudge) {
    const int64_t row = blockIdx.x*int64_t(blockDim.x) + threadIdx.x;
    if (row >= nrows || row_d[row] == 0.f) {
        return;
    }
    const float * xr = x + row*np;
    const float * wr = w + row*np;
    const float * qr = q + row*np;
    float sumqx = 0.f, sumq2 = 0.f;
    for (int j = 0; j < np; ++j) {
        sumqx = __fmaf_rn(__fmul_rn(wr[j], kAbs ? fabsf(xr[j]) : xr[j]), qr[j], sumqx);
        sumq2 = __fmaf_rn(__fmul_rn(wr[j], qr[j]), qr[j], sumq2);
    }
    if (sumq2 > 0.f) {
        *(float *)(y + row*row_bytes) = __fmul_rn(__fdiv_rn(sumqx, sumq2), fudge);
    }
}

static kt_codebook kt_get_codebook(int device, ggml_type type) {
    static std::mutex mutex;
    static kt_codebook books[GGML_CUDA_MAX_DEVICES][4];
    static bool ready[GGML_CUDA_MAX_DEVICES][4] = {};
    const int it = type == GGML_TYPE_IQ4_KT ? 0 : type == GGML_TYPE_IQ3_KT ? 1 : type == GGML_TYPE_IQ2_KT ? 2 : 3;
    std::lock_guard<std::mutex> lock(mutex);
    if (!ready[device][it]) {
        const int nbank = it ? 1 : 2;
        const int group_size = it ? 8 : 4;
        const int num_val = it == 0 ? 1 << 15 : it == 3 ? 1 << 13 : 1 << 16;
        for (int ib = 0; ib < nbank; ++ib) {
            const int * offsets, * points;
            const float * values, * mid;
            const int nc = iqk_kt_codebook(type, ib, &offsets, &points, &values, &mid);
            const std::vector<uint16_t> points16(points, points + offsets[nc]);
            kt_bank & b = books[device][it].bank[ib];
            b.offsets = ggml_cuda_quantize_upload(offsets, nc + 1);
            b.points = ggml_cuda_quantize_upload(points16.data(), points16.size());
            b.values = ggml_cuda_quantize_upload(values, size_t(group_size)*num_val);
            std::memcpy(b.mid, mid, sizeof(b.mid));
            b.offset = ib ? 4096 + 32768 : 4096;
            b.num_val = num_val;
        }
        ready[device][it] = true;
    }
    return books[device][it];
}

#endif

GGML_CALL size_t ggml_cuda_quantize(int device, enum ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int64_t nslice,
        const float * imatrix) {
    const bool ks = type == GGML_TYPE_IQ3_KS || type == GGML_TYPE_IQ4_KS || type == GGML_TYPE_IQ5_KS || type == GGML_TYPE_IQ4_KS_R4 ||
        type == GGML_TYPE_IQ4_KS_R16 || type == GGML_TYPE_IQ5_KS_R4;
    if (type != GGML_TYPE_IQ4_KT && type != GGML_TYPE_IQ3_KT && type != GGML_TYPE_IQ2_KT && type != GGML_TYPE_IQ1_KT && !ks) {
        return 0;
    }
#if defined(GGML_USE_HIPBLAS) || defined(GGML_USE_MUSA)
    const bool available = false;
#else
    const bool available = device < ggml_backend_cuda_get_device_count();
#endif
    if (!available) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "%s: device %d cannot run the CUDA encoder, tensors use the CPU encoder\n", __func__, device);
            warned = true;
        }
        return 0;
    }
#if defined(GGML_USE_HIPBLAS) || defined(GGML_USE_MUSA)
    GGML_UNUSED(src); GGML_UNUSED(dst); GGML_UNUSED(nrows); GGML_UNUSED(n_per_row); GGML_UNUSED(nslice); GGML_UNUSED(imatrix);
    return 0;
#else
    const bool iq3 = type == GGML_TYPE_IQ3_KT;
    const bool iq21 = type == GGML_TYPE_IQ2_KT || type == GGML_TYPE_IQ1_KT;
    ggml_cuda_set_device(device);
    if (ks) {
        return ggml_cuda_quantize_ks(device, type, src, dst, nrows, n_per_row, nslice, imatrix);
    }
    const kt_codebook cb = kt_get_codebook(device, type);

    const int np = GGML_PAD(n_per_row, QK_K);
    const int nb = np/QK_K;
    const int nb32 = np/kKtBlock;
    const int nt = (n_per_row % QK_K)/kKtBlock;
    const size_t row_bytes = ggml_row_size(type, np);
    const size_t out_row = ggml_row_size(type, n_per_row);
    const float fudge = ggml_get_quantize_fudge_factor(type);
    const int64_t total_rows = nrows*nslice;
    const bool replicate = iq21 && nt;
    const int8_t * d_index = iq21 ? ggml_cuda_ks_index(device, GGML_TYPE_IQ4_KS, false) : nullptr;

    const size_t per_row = 3*size_t(np)*sizeof(float) + nb32*(sizeof(float) + 1 + 4*sizeof(int)) + nb*(sizeof(float) + sizeof(uint32_t)) +
        sizeof(float) + kKtTries*sizeof(float2) + row_bytes;
    const int64_t chunk = ggml_cuda_quantize_chunk(total_rows, per_row);

    float * d_x, * d_w, * d_q, * d_scales, * d_sb_amax, * d_row_d, * d_imatrix = nullptr;
    float2 * d_sums;
    uint32_t * d_packed;
    uint8_t * d_banks;
    int * d_idx, * d_error;
    char * d_y;
    CUDA_CHECK(cudaMalloc(&d_x, chunk*np*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, chunk*np*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q, chunk*np*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, chunk*nb32*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_banks, chunk*nb32));
    CUDA_CHECK(cudaMalloc(&d_idx, chunk*nb32*4*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sb_amax, chunk*nb*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_packed, chunk*nb*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_row_d, chunk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, chunk*kKtTries*sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&d_error, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y, chunk*row_bytes));
    CUDA_CHECK(cudaMemset(d_error, 0, sizeof(int)));
    if (imatrix) {
        std::vector<float> padded(nslice*np, 0.f);
        for (int64_t is = 0; is < nslice; ++is) {
            if (replicate) {
                iqk_kt_pad_row(imatrix + is*n_per_row, padded.data() + is*np, n_per_row);
            } else {
                std::memcpy(padded.data() + is*np, imatrix + is*n_per_row, n_per_row*sizeof(float));
            }
        }
        d_imatrix = ggml_cuda_quantize_upload(padded.data(), padded.size());
    }

    std::vector<float> xpad(nt ? chunk*np : 0, 0.f);
    std::vector<char> ypad(nt ? chunk*row_bytes : 0);
    for (int64_t r0 = 0; r0 < total_rows; r0 += chunk) {
        const int64_t nr = std::min(chunk, total_rows - r0);
        // rows with a tail are padded to np (IQ2_KT and IQ1_KT: copies, others: zeros); iqk_kt_finish_row repacks the last block into the tail format
        if (nt) {
            for (int64_t r = 0; r < nr; ++r) {
                if (replicate) {
                    iqk_kt_pad_row(src + (r0 + r)*n_per_row, xpad.data() + r*np, n_per_row);
                } else {
                    std::memcpy(xpad.data() + r*np, src + (r0 + r)*n_per_row, n_per_row*sizeof(float));
                }
            }
            CUDA_CHECK(cudaMemcpy(d_x, xpad.data(), nr*np*sizeof(float), cudaMemcpyHostToDevice));
        } else {
            CUDA_CHECK(cudaMemcpy(d_x, src + r0*n_per_row, nr*np*sizeof(float), cudaMemcpyHostToDevice));
        }
        const int64_t nsuper = nr*nb;
        const int64_t nblocks = nr*nb32;
        const int warp_grid_blocks = (nblocks + kKtWarps - 1)/kKtWarps;
        const int warp_grid_super = (nsuper + kKtWarps - 1)/kKtWarps;
        k_kt_weights<<<(nsuper + 127)/128, 128>>>(d_x, d_imatrix, d_w, d_sb_amax, nsuper, replicate ? np : n_per_row, np, r0, nrows);
        if (iq21) {
            if (type == GGML_TYPE_IQ2_KT) {
                k_iq2kt_first_pass<GGML_TYPE_IQ2_KT><<<warp_grid_blocks, kKtWarps*WARP_SIZE>>>(d_x, d_w, d_sb_amax, d_scales, d_idx, d_q, d_error,
                        np, cb.bank[0]);
            } else {
                k_iq2kt_first_pass<GGML_TYPE_IQ1_KT><<<warp_grid_blocks, kKtWarps*WARP_SIZE>>>(d_x, d_w, d_sb_amax, d_scales, d_idx, d_q, d_error,
                        np, cb.bank[0]);
            }
            k_iq2kt_row_try<<<(nr*kKtTries + 127)/128, 128>>>(d_x, d_w, d_scales, d_q, d_index, d_sums, nr, np);
            k_iq2kt_row_scale<<<(nr + 127)/128, 128>>>(d_scales, d_sums, d_index, d_row_d, d_packed, d_y, nr, np, row_bytes, fudge);
            if (type == GGML_TYPE_IQ2_KT) {
                k_iq2kt_final_pass<GGML_TYPE_IQ2_KT><<<warp_grid_super, kKtWarps*WARP_SIZE>>>(d_x, d_w, d_idx, d_q, d_packed, d_row_d, d_q, d_y,
                        d_error, nsuper, np, row_bytes, cb.bank[0]);
            } else {
                k_iq2kt_final_pass<GGML_TYPE_IQ1_KT><<<warp_grid_super, kKtWarps*WARP_SIZE>>>(d_x, d_w, d_idx, d_q, d_packed, d_row_d, d_q, d_y,
                        d_error, nsuper, np, row_bytes, cb.bank[0]);
            }
            k_kt_refit<false><<<(nr + 127)/128, 128>>>(d_x, d_w, d_q, d_row_d, d_y, nr, np, row_bytes, fudge);
        } else if (iq3) {
            k_iq3kt_first_pass<<<warp_grid_blocks, kKtWarps*WARP_SIZE>>>(d_x, d_w, d_sb_amax, d_scales, d_q, d_error, nblocks, np, cb.bank[0]);
            k_iq3kt_row_try<<<(nr*kKtTries + 127)/128, 128>>>(d_x, d_w, d_scales, d_q, d_sums, nr, np);
            k_iq3kt_row_scale<<<(nr + 127)/128, 128>>>(d_scales, d_sb_amax, d_sums, d_row_d, d_packed, d_y, nr, np, row_bytes, fudge);
            k_iq3kt_final_pass<<<warp_grid_super, kKtWarps*WARP_SIZE>>>(d_x, d_w, d_packed, d_sb_amax, d_row_d, d_q, d_y, d_error, nsuper, np, row_bytes, cb.bank[0]);
            k_kt_refit<true><<<(nr + 127)/128, 128>>>(d_x, d_w, d_q, d_row_d, d_y, nr, np, row_bytes, fudge);
        } else {
            k_iq4kt_first_pass<<<warp_grid_blocks, kKtWarps*WARP_SIZE>>>(d_x, d_w, d_sb_amax, d_scales, d_banks, d_error, nblocks, np, cb);
            k_iq4kt_row_scale<<<(nr + 127)/128, 128>>>(d_scales, d_sb_amax, d_row_d, d_y, nr, np, row_bytes, fudge);
            k_iq4kt_final_pass<<<warp_grid_super, kKtWarps*WARP_SIZE>>>(d_x, d_w, d_scales, d_banks, d_row_d, d_q, d_y, d_error, nsuper, np, row_bytes, cb);
            k_kt_refit<false><<<(nr + 127)/128, 128>>>(d_x, d_w, d_q, d_row_d, d_y, nr, np, row_bytes, fudge);
        }
        CUDA_CHECK(cudaGetLastError());
        char * out = (char *)dst + r0*out_row;
        if (!nt) {
            CUDA_CHECK(cudaMemcpy(out, d_y, nr*row_bytes, cudaMemcpyDeviceToHost));
        } else {
            CUDA_CHECK(cudaMemcpy(ypad.data(), d_y, nr*row_bytes, cudaMemcpyDeviceToHost));
            for (int64_t r = 0; r < nr; ++r) {
                iqk_kt_finish_row(type, ypad.data() + r*row_bytes, out + r*out_row, n_per_row);
            }
        }
    }
    int error = 0;
    CUDA_CHECK(cudaMemcpy(&error, d_error, sizeof(int), cudaMemcpyDeviceToHost));
    if (error == 1) {
        GGML_ABORT("%s: failed to find solution for an IQ3_KT block", __func__);
    }
    if (error) {
        GGML_ABORT("%s: no %s codebook point has a finite score", __func__, ggml_type_name(type));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_banks));
    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_sb_amax));
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_row_d));
    CUDA_CHECK(cudaFree(d_sums));
    CUDA_CHECK(cudaFree(d_error));
    CUDA_CHECK(cudaFree(d_y));
    if (d_imatrix) {
        CUDA_CHECK(cudaFree(d_imatrix));
    }
    return total_rows*out_row;
#endif
}
