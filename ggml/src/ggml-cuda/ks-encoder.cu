#include "ks-encoder.cuh"
#include "../iqk/iqk_quantize.h"

#include <mutex>
#include <vector>

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)

// same result as IQ3_KS/IQ4_KS/IQ5_KS quantization on the CPU; the _rn intrinsics mirror each rounding and FMA of the GCC -O3 -march=native build

constexpr int kKsBlock   = 32;
constexpr int kKsThreads = 128;

template <ggml_type type> struct ks_traits {
    static constexpr int nval    = type == GGML_TYPE_IQ5_KS ?  32 : type == GGML_TYPE_IQ4_KS ?  16 :   8;
    static constexpr int nindex  = type == GGML_TYPE_IQ5_KS ? 247 : type == GGML_TYPE_IQ4_KS ? 241 : 111;
    static constexpr int ntry    = type == GGML_TYPE_IQ4_KS ?   7 :   5;
    static constexpr int nmap    = type == GGML_TYPE_IQ3_KS ? 127 : 240;
    static constexpr int nsimple = type == GGML_TYPE_IQ3_KS ? 128 : 256;
};

// (int)x as cvttss2si computes it: INT_MIN for NaN and for values outside the int range
static __device__ __forceinline__ int ks_trunc(float x) {
    return x >= -2147483648.f && x < 2147483648.f ? __float2int_rz(x) : INT_MIN;
}

static __device__ __forceinline__ int ks_nearest_int(float v) {
    // the x86 default NaN gives 0 here; CUDA arithmetic returns a NaN with other bits
    return isnan(v) ? 0 : (__float_as_int(v) & 0x007fffff) - 0x00400000;
}

template <ggml_type type>
static __device__ __forceinline__ int ks_best_index(const int8_t * values, const int8_t * index, float x) {
    constexpr int nval = ks_traits<type>::nval;
    int ix = ks_trunc(x) - values[0];
    if (ix < 0 || ix >= ks_traits<type>::nindex) {
        return ix < 0 ? 0 : nval - 1;
    }
    ix = index[ix];
    return ix < nval ? ix : __fsub_rn(x, values[ix - nval]) < __fsub_rn(values[ix - nval + 1], x) ? ix - nval : ix - nval + 1;
}

template <ggml_type type>
static __device__ __forceinline__ void ks_load_tables(int8_t * vals, int8_t * index, const int8_t * index_g, int nindex) {
    const int8_t * values = type == GGML_TYPE_IQ5_KS ? iq5nl_values : type == GGML_TYPE_IQ4_KS ? iq4k_values : iq3nl_values;
    for (int i = threadIdx.x; i < 2*ks_traits<type>::nval; i += blockDim.x) vals[i] = values[i];
    for (int i = threadIdx.x; i < nindex; i += blockDim.x) index[i] = index_g[i];
}

// the value that best_index_avx2() selects
template <ggml_type type>
static __device__ __forceinline__ float ks_simple_value(const int8_t * values, const int8_t * index, float x) {
    const int t = index[min(max(ks_trunc(x) - values[0], 0), ks_traits<type>::nmap)];
    const float c0 = values[t], c1 = values[t + 1];
    return __fsub_rn(x, c0) < __fsub_rn(c1, x) ? c0 : c1;
}

// hsum_float_8()
static __device__ __forceinline__ float ks_hsum8(const float * s) {
    return __fadd_rn(__fadd_rn(__fadd_rn(s[0], s[4]), __fadd_rn(s[2], s[6])), __fadd_rn(__fadd_rn(s[1], s[5]), __fadd_rn(s[3], s[7])));
}

// evaluate_block_avx2(): 8 partial sums, element j goes to sum j%8
template <ggml_type type>
static __device__ __forceinline__ void ks_sums(const float * xb, const float * wb, const int8_t * values, const int8_t * index, float id,
        float & sumqx_p, float & sumq2_p, float & sumqx_m, float & sumq2_m) {
    float sqx_p[8] = {}, sq2_p[8] = {}, sqx_m[8] = {}, sq2_m[8] = {};
    for (int j = 0; j < kKsBlock; ++j) {
        const float al = __fmul_rn(id, xb[j]);
        const float qp = ks_simple_value<type>(values, index, al);
        const float qm = ks_simple_value<type>(values, index, -al);
        const float wp = __fmul_rn(wb[j], qp);
        const float wm = __fmul_rn(wb[j], qm);
        sqx_p[j%8] = __fmaf_rn(wp, xb[j], sqx_p[j%8]);
        sq2_p[j%8] = __fmaf_rn(wp, qp, sq2_p[j%8]);
        sqx_m[j%8] = __fmaf_rn(wm, xb[j], sqx_m[j%8]);
        sq2_m[j%8] = __fmaf_rn(wm, qm, sq2_m[j%8]);
    }
    sumqx_p = ks_hsum8(sqx_p);
    sumq2_p = ks_hsum8(sq2_p);
    sumqx_m = ks_hsum8(sqx_m);
    sumq2_m = ks_hsum8(sq2_m);
}

template <ggml_type type>
static __global__ void k_ks_search(const float * __restrict__ x, const float * __restrict__ imatrix, const int8_t * __restrict__ index_g,
        float * __restrict__ w, float * __restrict__ scales, uint8_t * __restrict__ shifted, int n_per_row, int64_t row0, int64_t nrows,
        float eps) {
    using tr = ks_traits<type>;
    constexpr int npass = 1 + 2*(2*tr::ntry + 1);
    __shared__ float xs[QK_K];
    __shared__ float ws[QK_K];
    __shared__ int8_t vals[2*tr::nval];
    __shared__ int8_t index[tr::nsimple];
    ks_load_tables<type>(vals, index, index_g, tr::nsimple);
    const int64_t i = int64_t(blockIdx.x)*QK_K + threadIdx.x;
    const int64_t ib = i/kKsBlock;
    const int j = threadIdx.x % kKsBlock;
    xs[threadIdx.x] = x[i];
    __syncthreads();
    const float * xb = xs + threadIdx.x - j;
    float * wb = ws + threadIdx.x - j;

    if (imatrix) {
        float sigma2 = 0.f;
        // every lane runs the CPU loops below and gets the same result
        for (int k = 0; k < QK_K; ++k) sigma2 = __fadd_rn(sigma2, __fmul_rn(xs[k], xs[k]));
        sigma2 = __fmul_rn(sigma2, 2.f/QK_K);
        const int nb32 = n_per_row/kKsBlock;
        // one imatrix slice per expert, nrows rows each
        const float * qw = imatrix + (row0 + ib/nb32)/nrows*n_per_row + (ib % nb32)*kKsBlock;
        float sumw2 = 0.f, sumx2 = 0.f, sumwx = 0.f;
        for (int k = 0; k < kKsBlock; ++k) {
            sumw2 = __fadd_rn(sumw2, __fmul_rn(qw[k], qw[k]));
            sumx2 = __fadd_rn(sumx2, __fmul_rn(xb[k], xb[k]));
            sumwx = __fadd_rn(sumwx, __fmul_rn(qw[k], fabsf(xb[k])));
        }
        const bool keep = sumw2 > eps && sumx2 > eps && sumwx > eps;
        wb[j] = __fmul_rn(keep ? qw[j] : 1e-9f, __fsqrt_rn(__fmaf_rn(xb[j], xb[j], sigma2)));
    } else {
        wb[j] = __fmul_rn(xb[j], xb[j]);
    }
    w[i] = wb[j];
    __syncwarp();

    float amax = 0.f, max = 0.f;
    for (int k = 0; k < kKsBlock; ++k) {
        const float ax = fabsf(xb[k]);
        if (ax > amax) {
            amax = ax; max = xb[k];
        }
    }
    if (amax < 1e-16f) {
        if (j == 0) {
            scales[ib] = 0.f;
            shifted[ib] = 0;
        }
        return;
    }
    const int8_t * values = vals;
    const int8_t * shifted_values = vals + tr::nval;
    float d = __fdiv_rn(-max, float(values[0]));
    float sumqx_p = 0.f, sumq2_p = 0.f, sumqx_m = 0.f, sumq2_m = 0.f;
    // lane 0 runs the first pass, and lanes 2k+1 and 2k+2 run try k with values and with shifted_values; the loop below compares them in CPU order
    if (j < npass) {
        const int8_t * v = j > 0 && j % 2 == 0 ? shifted_values : values;
        const float id = j ? __fdiv_rn(float((j - 1)/2 - tr::ntry + v[0]), max) : __fdiv_rn(1.f, d);
        ks_sums<type>(xb, wb, v, index, id, sumqx_p, sumq2_p, sumqx_m, sumq2_m);
    }
    float best = 0.f;
    bool is_shifted = false;
    for (int p = 0; p < npass; ++p) {
        const float qxp = __shfl_sync(0xffffffff, sumqx_p, p);
        const float q2p = __shfl_sync(0xffffffff, sumq2_p, p);
        const float qxm = __shfl_sync(0xffffffff, sumqx_m, p);
        const float q2m = __shfl_sync(0xffffffff, sumq2_m, p);
        if (q2p > 0 && __fmul_rn(qxp, qxp) > __fmul_rn(best, q2p)) {
            d = __fdiv_rn(qxp, q2p); best = __fmul_rn(d, qxp); is_shifted = p > 0 && p % 2 == 0;
        }
        if (q2m > 0 && __fmul_rn(qxm, qxm) > __fmul_rn(best, q2m)) {
            d = __fdiv_rn(qxm, q2m); best = __fmul_rn(d, qxm); is_shifted = p > 0 && p % 2 == 0;
        }
    }
    if (j == 0) {
        scales[ib] = d;
        shifted[ib] = is_shifted;
    }
}

template <ggml_type type>
static __global__ void k_ks_finish(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ scales,
        const uint8_t * __restrict__ shifted, const int8_t * __restrict__ index_g, char * __restrict__ y, int64_t nr, int n_per_row,
        size_t row_size, float fudge) {
    using tr = ks_traits<type>;
    using block_t = std::conditional_t<type == GGML_TYPE_IQ4_KS, block_iq4_ks, block_iq5_ks>;
    __shared__ int8_t vals[2*tr::nval];
    __shared__ int8_t index[tr::nindex];
    ks_load_tables<type>(vals, index, index_g, tr::nindex);
    __syncthreads();
    const int64_t row = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (row >= nr) {
        return;
    }
    const int nb32 = n_per_row/kKsBlock;
    const float * xr = x + row*n_per_row;
    const float * wr = w + row*n_per_row;
    const float * sr = scales + row*nb32;
    const uint8_t * fr = shifted + row*nb32;
    char * yr = y + row*row_size;
    for (size_t k = 0; k < row_size; ++k) yr[k] = 0;
    float * dptr = (float *)yr;
    block_t * yb = (block_t *)(dptr + 1);

    float amax_scale = 0.f;
    for (int ib = 0; ib < nb32; ++ib) {
        if (fr[ib]) yb[ib/8].scales[ib%8] = 0x01;
        const float a = fabsf(sr[ib]);
        if (amax_scale < a) amax_scale = a;
    }
    const float d = __fdiv_rn(amax_scale, 127.f);
    *dptr = __fmul_rn(fudge, d);
    if (!d) {
        return;
    }
    const float id = __fdiv_rn(1.f, d);
    float sumqx = 0.f, sumq2 = 0.f;
    for (int ib = 0; ib < nb32; ++ib) {
        block_t & blk = yb[ib/8];
        const int8_t * bv = fr[ib] ? vals + tr::nval : vals;
        // nearest_int(0.5f*(id*scales[ib] + 127.f)) as GCC fuses it
        int l = ks_nearest_int(__fmaf_rn(__fmaf_rn(id, sr[ib], 127.f), 0.5f, 12582912.f));
        l = max(0, min(127, l)) << 1;
        blk.scales[ib%8] |= l;
        l -= 127;
        const float dl = __fmul_rn(d, float(l));
        const float idl = dl ? __fdiv_rn(1.f, dl) : 0.f;
        const float * xb = xr + ib*kKsBlock;
        const float * wb = wr + ib*kKsBlock;
        if constexpr (type == GGML_TYPE_IQ4_KS) {
            uint8_t * qs = blk.qs + (ib%8)*(kKsBlock/2);
            for (int j = 0; j < kKsBlock/2; ++j) {
                const int i1 = ks_best_index<type>(bv, index, __fmul_rn(idl, xb[j]));
                const int i2 = ks_best_index<type>(bv, index, __fmul_rn(idl, xb[j + kKsBlock/2]));
                qs[j] = i1 | (i2 << 4);
                const float q1 = float(bv[i1]*l);
                const float q2 = float(bv[i2]*l);
                const float a = __fmul_rn(q1, wb[j]);
                const float b = __fmul_rn(q2, wb[j + kKsBlock/2]);
                // the CPU loop adds two products per step; GCC fuses them this way
                sumqx = __fadd_rn(sumqx, __fmaf_rn(a, xb[j], __fmul_rn(b, xb[j + kKsBlock/2])));
                sumq2 = __fadd_rn(sumq2, __fmaf_rn(q1, a, __fmul_rn(b, q2)));
            }
        } else {
            uint8_t * qs = blk.qs + (ib%8/2)*kKsBlock;
            for (int j = 0; j < kKsBlock; ++j) {
                const int i = ks_best_index<type>(bv, index, __fmul_rn(idl, xb[j]));
                qs[j] |= (i & 0xf) << 4*(ib%2);
                blk.qh[j] |= (i >> 4) << ib%8;
                const float q = float(bv[i]*l);
                const float wq = __fmul_rn(wb[j], q);
                sumqx = __fmaf_rn(wq, xb[j], sumqx);
                sumq2 = __fmaf_rn(wq, q, sumq2);
            }
        }
    }
    if (sumq2 > 0) *dptr = __fdiv_rn(__fmul_rn(fudge, sumqx), sumq2);
}

static __global__ void k_iq3ks_finish(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ scales,
        const uint8_t * __restrict__ shifted, const int8_t * __restrict__ index_g, char * __restrict__ y, int64_t nr, int n_per_row,
        size_t row_size, float fudge) {
    __shared__ int8_t vals[16];
    __shared__ int8_t index[111];
    ks_load_tables<GGML_TYPE_IQ3_KS>(vals, index, index_g, 111);
    __syncthreads();
    const int64_t row = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (row >= nr) {
        return;
    }
    const int nb32 = n_per_row/kKsBlock;
    const float * xr = x + row*n_per_row;
    const float * wr = w + row*n_per_row;
    const float * sr = scales + row*nb32;
    const uint8_t * fr = shifted + row*nb32;
    char * yr = y + row*row_size;
    for (size_t k = 0; k < row_size; ++k) yr[k] = 0;
    half * dptr = (half *)yr;
    block_iq3_ks * yb = (block_iq3_ks *)(dptr + 1);

    float amax_scale = 0.f, max_scale = 0.f;
    for (int ib = 0; ib < nb32; ++ib) {
        if (fr[ib]) yb[ib/8].extra |= 1 << (8 + ib%8);
        const float a = fabsf(sr[ib]);
        if (a > amax_scale) {
            amax_scale = a; max_scale = sr[ib];
        }
    }
    const float d = __fdiv_rn(-max_scale, 16.f);
    *dptr = __float2half_rn(d);
    if (!d) {
        return;
    }
    const float id = __fdiv_rn(1.f, d);
    float sumqx = 0.f, sumq2 = 0.f;
    for (int ib = 0; ib < nb32; ++ib) {
        block_iq3_ks & blk = yb[ib/8];
        const int is = ib%8;
        const int8_t * bv = fr[ib] ? vals + 8 : vals;
        int l = ks_nearest_int(__fmaf_rn(id, sr[ib], 12582912.f));
        l = max(-16, min(15, l));
        const int ul = l + 16;
        blk.scales[is%4] |= (ul & 0xf) << 4*(is/4);
        blk.extra |= (ul >> 4) << is;
        const float dl = __fmul_rn(d, float(l));
        const float idl = dl ? __fdiv_rn(1.f, dl) : 0.f;
        const float * xb = xr + ib*kKsBlock;
        const float * wb = wr + ib*kKsBlock;
        uint8_t * qs = blk.qs + (is/4)*kKsBlock;
        for (int j = 0; j < kKsBlock; ++j) {
            const int i = ks_best_index<GGML_TYPE_IQ3_KS>(bv, index, __fmul_rn(idl, xb[j]));
            qs[j] |= (i & 3) << 2*(is%4);
            blk.qh[j] |= (i >> 2) << is;
            const float q = float(bv[i]*l);
            const float wq = __fmul_rn(wb[j], q);
            sumqx = __fmaf_rn(wq, xb[j], sumqx);
            sumq2 = __fmaf_rn(wq, q, sumq2);
        }
    }
    if (sumq2 > 0) *dptr = __float2half_rn(__fdiv_rn(__fmul_rn(fudge, sumqx), sumq2));
}

static const int8_t * ks_get_index(int device, ggml_type type, bool simple) {
    static std::mutex mutex;
    static int8_t * tables[GGML_CUDA_MAX_DEVICES][6] = {};
    const int it = (type == GGML_TYPE_IQ4_KS ? 0 : type == GGML_TYPE_IQ3_KS ? 1 : 2) + 3*simple;
    std::lock_guard<std::mutex> lock(mutex);
    if (!tables[device][it]) {
        const int8_t * index;
        const int n = simple ? iqk_ks_simple_index(type, &index) : iqk_ks_index(type, &index);
        tables[device][it] = ggml_cuda_quantize_upload(index, n);
    }
    return tables[device][it];
}

size_t ggml_cuda_quantize_ks(int device, ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int64_t nslice,
        const float * imatrix) {
    if (n_per_row % QK_K) {
        return 0;
    }
    if (type == GGML_TYPE_IQ4_KS_R4 || type == GGML_TYPE_IQ4_KS_R16 || type == GGML_TYPE_IQ5_KS_R4) {
        const ggml_type base = type == GGML_TYPE_IQ5_KS_R4 ? GGML_TYPE_IQ5_KS : GGML_TYPE_IQ4_KS;
        std::vector<char> tmp(nrows*nslice*ggml_row_size(base, n_per_row));
        const size_t size = ggml_cuda_quantize_ks(device, base, src, tmp.data(), nrows, n_per_row, nslice, imatrix);
        if (type == GGML_TYPE_IQ4_KS_R4) {
            repack_iq4_ks(nrows*nslice, n_per_row, (const block_iq4_ks *)tmp.data(), (block_iq4_ks_r4 *)dst, false);
        } else if (type == GGML_TYPE_IQ4_KS_R16) {
            repack_iq4_ks_r16(nrows*nslice, n_per_row, (const block_iq4_ks *)tmp.data(), (block_iq4_ks_r16 *)dst, false, n_per_row);
        } else {
            repack_iq5_ks(nrows*nslice, n_per_row, (const block_iq5_ks *)tmp.data(), (block_iq5_ks_r4 *)dst, false);
        }
        return size;
    }
    const int8_t * d_index = ks_get_index(device, type, false);
    const int8_t * d_simple = ks_get_index(device, type, true);
    const int nb32 = n_per_row/kKsBlock;
    const size_t row_size = ggml_row_size(type, n_per_row);
    const float fudge = ggml_get_quantize_fudge_factor(type);
    constexpr float kEps = 1e-9f;
    constexpr float kEps2 = kEps*kEps;
    const float eps = kKsBlock*kEps2;
    const int64_t total_rows = nrows*nslice;

    const size_t per_row = 2*size_t(n_per_row)*sizeof(float) + nb32*(sizeof(float) + 1) + row_size;
    const int64_t chunk = ggml_cuda_quantize_chunk(total_rows, per_row);

    float * d_x, * d_w, * d_scales, * d_imatrix = nullptr;
    uint8_t * d_shifted;
    char * d_y;
    CUDA_CHECK(cudaMalloc(&d_x, chunk*n_per_row*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, chunk*n_per_row*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, chunk*nb32*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_shifted, chunk*nb32));
    CUDA_CHECK(cudaMalloc(&d_y, chunk*row_size));
    if (imatrix) {
        d_imatrix = ggml_cuda_quantize_upload(imatrix, nslice*n_per_row);
    }

    for (int64_t r0 = 0; r0 < total_rows; r0 += chunk) {
        const int64_t nr = std::min(chunk, total_rows - r0);
        CUDA_CHECK(cudaMemcpy(d_x, src + r0*n_per_row, nr*n_per_row*sizeof(float), cudaMemcpyHostToDevice));
        const int grid_search = nr*n_per_row/QK_K;
        const int grid_rows = (nr + kKsThreads - 1)/kKsThreads;
        if (type == GGML_TYPE_IQ4_KS) {
            k_ks_search<GGML_TYPE_IQ4_KS><<<grid_search, QK_K>>>(d_x, d_imatrix, d_simple, d_w, d_scales, d_shifted, n_per_row, r0, nrows, eps);
            k_ks_finish<GGML_TYPE_IQ4_KS><<<grid_rows, kKsThreads>>>(d_x, d_w, d_scales, d_shifted, d_index, d_y, nr, n_per_row, row_size, fudge);
        } else if (type == GGML_TYPE_IQ5_KS) {
            k_ks_search<GGML_TYPE_IQ5_KS><<<grid_search, QK_K>>>(d_x, d_imatrix, d_simple, d_w, d_scales, d_shifted, n_per_row, r0, nrows, eps);
            k_ks_finish<GGML_TYPE_IQ5_KS><<<grid_rows, kKsThreads>>>(d_x, d_w, d_scales, d_shifted, d_index, d_y, nr, n_per_row, row_size, fudge);
        } else {
            k_ks_search<GGML_TYPE_IQ3_KS><<<grid_search, QK_K>>>(d_x, d_imatrix, d_simple, d_w, d_scales, d_shifted, n_per_row, r0, nrows, eps);
            k_iq3ks_finish<<<grid_rows, kKsThreads>>>(d_x, d_w, d_scales, d_shifted, d_index, d_y, nr, n_per_row, row_size, fudge);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy((char *)dst + r0*row_size, d_y, nr*row_size, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_shifted));
    CUDA_CHECK(cudaFree(d_y));
    if (d_imatrix) {
        CUDA_CHECK(cudaFree(d_imatrix));
    }
    return total_rows*row_size;
}

#endif
