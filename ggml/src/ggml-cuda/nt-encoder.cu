#include "nt-encoder.cuh"
#include "../iqk/iqk_quantize.h"

#include <cstring>
#include <thread>
#include <vector>

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)


constexpr int kNtStates = 4096, kNtThreads = 256;

static __global__ void nt_forward(const float * __restrict__ xn, const float * __restrict__ ww, int n,
                                  unsigned long long * __restrict__ bp, float * __restrict__ final_cost, const float * __restrict__ pq) {
    __shared__ float cost[2][kNtStates];
    __shared__ unsigned int smin;
    const int row = blockIdx.x, r = threadIdx.x;
    const float * x = xn + (size_t)row*n, * w = ww + (size_t)row*n;
    unsigned long long * bpr = bp + (size_t)row*n*kNtThreads;

    float P[16], Q[16];
#pragma unroll
    for (int u = 0; u < 16; ++u) { P[u] = pq[r*32 + u]; Q[u] = pq[r*32 + 16 + u]; }

#pragma unroll
    for (int n0 = 0; n0 < 16; ++n0) cost[0][n0*256 + r] = 1e30f;
    if (r == 0) cost[0][0] = 0.f;
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < n; ++t) {
        const float xv = x[t], wv = w[t];
        float old[16];
#pragma unroll
        for (int z = 0; z < 16; ++z) old[z] = cost[cur][r*16 + z];
        unsigned long long word = 0;
        float mine = 3.4e38f;
#pragma unroll
        for (int n0 = 0; n0 < 16; ++n0) {
            float best = 3e38f; unsigned int arg = 0;
#pragma unroll
            for (int z = 0; z < 16; ++z) {
                const float d = __fsub_rn(xv, __fadd_rn(P[n0], Q[(n0 + z) & 15]));
                const float e = __fmaf_rn(__fmul_rn(d, d), wv, old[z]);
                if (e < best) { best = e; arg = z; }
            }
            cost[cur ^ 1][n0*256 + r] = best;
            word |= (unsigned long long)arg << (4*n0);
            mine = fminf(mine, best);
        }
        bpr[(size_t)t*kNtThreads + r] = word;
        cur ^= 1;
        if ((t & 31) == 31) {
            if (r == 0) smin = 0x7f800000u;
            __syncthreads();
            atomicMin(&smin, __float_as_uint(mine));
            __syncthreads();
            const float m = __uint_as_float(smin);
#pragma unroll
            for (int n0 = 0; n0 < 16; ++n0) cost[cur][n0*256 + r] = __fsub_rn(cost[cur][n0*256 + r], m);
        }
        __syncthreads();
    }
    float * fc = final_cost + (size_t)row*kNtStates;
#pragma unroll
    for (int n0 = 0; n0 < 16; ++n0) fc[n0*256 + r] = cost[cur][n0*256 + r];
}

static __global__ void nt_backtrack(const unsigned long long * __restrict__ bp, const int * __restrict__ start, int n, int rows,
                                    uint8_t * __restrict__ nib) {
    const int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const unsigned long long * bpr = bp + (size_t)row*n*kNtThreads;
    uint8_t * out = nib + (size_t)row*n;
    int s = start[row];
    for (int t = n - 1; t >= 0; --t) {
        const int r = s & 255, n0 = s >> 8;
        const int z = (int)((bpr[(size_t)t*kNtThreads + r] >> (4*n0)) & 15);
        out[t] = (uint8_t)n0;
        s = r*16 + z;
    }
}

static bool nt_pq(std::vector<float> & pq) {
    const int8_t * lab = iqk_nt_label_table();
    pq.assign(256*32, 0.f);
    for (int r = 0; r < 256; ++r) {
        const int n1 = r >> 4, n2 = r & 15;
        auto L = [&](int n0, int n3) { return (int)lab[((n0*16 + n1)*16 + n2)*16 + n3]; };
        int P[16], Q[16];
        for (int u = 0; u < 16; ++u) Q[u] = L(0, u);
        for (int n0 = 0; n0 < 16; ++n0) P[n0] = L(n0, 0) - Q[n0];
        for (int n0 = 0; n0 < 16; ++n0) for (int z = 0; z < 16; ++z) if (L(n0, z) != P[n0] + Q[(n0 + z) & 15]) return false;
        for (int u = 0; u < 16; ++u) { pq[r*32 + u] = (float)P[u]; pq[r*32 + 16 + u] = (float)Q[u]; }
    }
    return true;
}

struct NtSearch {
    int max_rows = 0, n = 0;
    unsigned long long * bp = nullptr;
    float * xn = nullptr, * ww = nullptr, * fc = nullptr, * pq = nullptr;
    int * start = nullptr;
    uint8_t * nib = nullptr;
    std::vector<float> h_fc;
    std::vector<int> h_start;
    bool ok = true;
    ~NtSearch() {
        cudaFree(bp); cudaFree(xn); cudaFree(ww); cudaFree(fc); cudaFree(pq); cudaFree(start); cudaFree(nib);
    }
    bool alloc(int rows, int n_) {
        max_rows = rows; n = n_;
        const size_t nx = (size_t)rows*n;
        std::vector<float> t;
        if (!nt_pq(t)) return false;
        return cudaMalloc(&bp, nx*kNtThreads*sizeof(unsigned long long)) == cudaSuccess && cudaMalloc(&xn, nx*4) == cudaSuccess &&
               cudaMalloc(&ww, nx*4) == cudaSuccess && cudaMalloc(&nib, nx) == cudaSuccess &&
               cudaMalloc(&fc, (size_t)rows*kNtStates*4) == cudaSuccess && cudaMalloc(&start, (size_t)rows*4) == cudaSuccess &&
               cudaMalloc(&pq, t.size()*4) == cudaSuccess && cudaMemcpy(pq, t.data(), t.size()*4, cudaMemcpyHostToDevice) == cudaSuccess;
    }
};

static int nt_search(void * vctx, const float * xn, const float * ww, int rows, int n, uint8_t * nib) {
    NtSearch & S = *(NtSearch *)vctx;
    if (rows > S.max_rows || n != S.n) return 1;
    const size_t nx = (size_t)rows*n;
    if (cudaMemcpy(S.xn, xn, nx*4, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    if (cudaMemcpy(S.ww, ww, nx*4, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    nt_forward<<<rows, kNtThreads>>>(S.xn, S.ww, n, S.bp, S.fc, S.pq);
    if (cudaGetLastError() != cudaSuccess) return 1;
    S.h_fc.resize((size_t)rows*kNtStates);
    if (cudaMemcpy(S.h_fc.data(), S.fc, S.h_fc.size()*4, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    S.h_start.resize(rows);
    for (int r = 0; r < rows; ++r) {
        const float * N = S.h_fc.data() + (size_t)r*kNtStates;
        int s = 0; float mn = N[0];
        for (int i = 1; i < kNtStates; ++i) if (N[i] < mn) { mn = N[i]; s = i; }
        S.h_start[r] = s;
    }
    if (cudaMemcpy(S.start, S.h_start.data(), (size_t)rows*4, cudaMemcpyHostToDevice) != cudaSuccess) return 1;
    nt_backtrack<<<(rows + 127)/128, 128>>>(S.bp, S.start, n, rows, S.nib);
    if (cudaGetLastError() != cudaSuccess) return 1;
    if (cudaMemcpy(nib, S.nib, nx, cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    return 0;
}

size_t ggml_cuda_quantize_nt(int device, const float * src, void * dst, int64_t nrows, int64_t n_per_row, int64_t nslice,
        const float * imatrix) {
    if (n_per_row % 128 || n_per_row > 65536 || nrows < 1) return 0;
    ggml_cuda_set_device(device);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    const size_t reserve = 1024ull << 20;
    const size_t per_row = (size_t)n_per_row*(kNtThreads*sizeof(unsigned long long) + 9) + kNtStates*4 + 4;
    const int64_t max_rows = free_b > reserve ? (int64_t)((free_b - reserve)/per_row) : 0;
    if (max_rows < 1) return 0;
    const int rows = (int)std::min(std::min(max_rows, (int64_t)1024), nrows);
    NtSearch S;
    if (!S.alloc(rows, (int)n_per_row)) {
        (void)cudaGetLastError();
        return 0;
    }
    const int nthread = std::max(1, (int)std::thread::hardware_concurrency());
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ4_NT, n_per_row);
    size_t total = 0;
    for (int64_t is = 0; is < nslice; ++is) {
        const size_t got = iqk_quantize_iq4_nt_batched(src + is*nrows*n_per_row, (char *)dst + is*nrows*row_size, nrows, n_per_row,
                imatrix ? imatrix + is*n_per_row : nullptr, nt_search, &S, rows, nthread);
        if (got == 0) return 0;
        total += got;
    }
    return total;
}

#else

size_t ggml_cuda_quantize_nt(int, const float *, void *, int64_t, int64_t, int64_t, const float *) {
    return 0;
}

#endif
