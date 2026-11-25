#pragma once

#include "fattn-rel-bias-col.cuh"

struct fattn_rel_bias_tile_col {
    const char * row;
    int q;
    float tail;
};

template <int ncols>
struct fattn_rel_bias_block {
    fattn_rel_bias_tile_col col[ncols];
    int2 q_range;
};

template <int n>
static __device__ __forceinline__ int2 fattn_rel_bias_range(const char * pos, int stride, int count = n) {
    int lo = INT_MAX, hi = INT_MIN;
#pragma unroll
    for (int i = 0; i < n; i += WARP_SIZE) {
        if (i + threadIdx.x < count) {
            const int p = *(const int *) (pos + (i + threadIdx.x)*stride);
            lo = min(lo, p);
            hi = max(hi, p);
        }
    }
#pragma unroll
    for (int s = WARP_SIZE/2; s; s >>= 1) {
        lo = min(lo, __shfl_xor_sync(0xffffffff, lo, s, WARP_SIZE));
        hi = max(hi, __shfl_xor_sync(0xffffffff, hi, s, WARP_SIZE));
    }
    return make_int2(lo, hi);
}

template <int nq, int nh>
static __device__ __forceinline__ void fattn_rel_bias_init_block(const fattn_rel_bias & rb,
        fattn_rel_bias_block<nq*nh> & b, int q0, int h0, int ne01) {
    const int j = threadIdx.y*WARP_SIZE + threadIdx.x;
    for (int c = j; c < nq*nh; c += blockDim.y*WARP_SIZE) {
        const int iq = min(q0 + c/nh, ne01 - 1), head = h0 + c%nh;
        const auto col = fattn_rel_bias_col(rb, iq, head);
        b.col[c] = {rb.rs + head*rb.nb1 + iq*rb.nb2, col.q, col.tail};
    }
    if (threadIdx.y == 0) {
        const int2 range = fattn_rel_bias_range<nq>(rb.pos_q + q0*rb.nbq, rb.nbq, min(nq, ne01 - q0));
        if (threadIdx.x == 0) {
            b.q_range = range;
        }
    }
}

struct fattn_rel_bias_slice {
    int first;
    int count;
};

template <int kd, int ncols, int nkeys>
static __device__ __forceinline__ void fattn_rel_bias_load_slice(const fattn_rel_bias & rb,
        const fattn_rel_bias_block<ncols> & b, int k0, int first, int count, float * scratch) {
    const int iw = threadIdx.y*WARP_SIZE + threadIdx.x;
    const int nth = blockDim.y*WARP_SIZE;
    const int width = kd ? kd : rb.kd;
    for (int i = iw; i < nkeys; i += nth) {
        ((int *)scratch)[i] = *(const int *)(rb.pos_k + (k0 + i)*rb.nbk);
    }
    float * rows = scratch + nkeys;
    float * weights = rows + ncols*width;
    if constexpr (kd > 0 && kd % 4 == 0) {
        if (fattn_rel_bias_can_load_float4(rb, uintptr_t(rb.rs) | rb.nb1 | rb.nb2)) {
            constexpr int width4 = kd/4;
            for (int i = iw; i < ncols*width4; i += nth) {
                ((float4 *) rows)[i] = ((const float4 *) b.col[i/width4].row)[i%width4];
            }
            for (int i = iw; i < count*width4; i += nth) {
                const char * w = rb.w + size_t(first + i/width4)*rb.nbw1;
                ((float4 *) weights)[i] = ((const float4 *) w)[i%width4];
            }
            return;
        }
    }
    for (int i = iw; i < ncols*width; i += nth) {
        rows[i] = *(const float *)(b.col[i/width].row + (i%width)*rb.nb0);
    }
    for (int i = iw; i < count*width; i += nth) {
        weights[i] = *(const float *)(rb.w + size_t(first + i/width)*rb.nbw1 + (i%width)*rb.nbw0);
    }
}

template <int nkeys, int capacity, bool sync_before_write, int ncols>
static __device__ __forceinline__ fattn_rel_bias_slice fattn_rel_bias_stage(const fattn_rel_bias & rb,
        const fattn_rel_bias_block<ncols> & b, int k0, float * scratch) {
    const int2 k = fattn_rel_bias_range<nkeys>(rb.pos_k + k0*rb.nbk, rb.nbk);
    const int2 q = b.q_range;
    const unsigned first = q.x > k.y ? unsigned(q.x) - unsigned(k.y) : 0;
    if (first >= rb.last || q.y < k.x || rb.kd == 0) {
        return {0, 0};
    }
    const unsigned end = min(rb.last - 1, unsigned(q.y) - unsigned(k.x));
    const unsigned count = end - first + 1;
    const unsigned rows = (capacity - nkeys)/rb.kd;
    if (count > rows || unsigned(ncols) > rows - count) {
        return {0, -1};
    }
    if constexpr (sync_before_write) {
        __syncthreads();
    }
    if (rb.kd == 20) {
        fattn_rel_bias_load_slice<20, ncols, nkeys>(rb, b, k0, first, count, scratch);
    } else {
        fattn_rel_bias_load_slice<0, ncols, nkeys>(rb, b, k0, first, count, scratch);
    }
    __syncthreads();
    return {int(first), int(count)};
}

template <int kd, int ncols, int nkeys>
static __device__ __forceinline__ float fattn_rel_bias_slice_get(const fattn_rel_bias & rb,
        const fattn_rel_bias_tile_col & col, const float * scratch, int first, int c, int i) {
    const int k = ((const int *)scratch)[i];
    const unsigned delta = unsigned(col.q) - unsigned(k);
    if (col.q < k || delta >= rb.last) {
        return col.tail;
    }
    const int width = kd ? kd : rb.kd;
    const float * row = scratch + nkeys + c*width;
    const float * w = scratch + nkeys + (ncols + delta - first)*width;
    float sum = 0.0f;
    for (int d = 0; d < width; ++d) {
        sum += row[d]*w[d];
    }
    return sum;
}

template <int nkeys, int capacity, bool sync_before_write, int ncols, typename Visit>
static __device__ __forceinline__ void fattn_rel_bias_add_tile(const fattn_rel_bias & rb,
        const fattn_rel_bias_block<ncols> & b, int k0, float * scratch, Visit visit) {
    const auto s = fattn_rel_bias_stage<nkeys, capacity, sync_before_write>(rb, b, k0, scratch);
    if (s.count == 0) {
        visit([&](int c, int) { return b.col[c].tail; });
    } else if (s.count < 0) {
        visit([&](int c, int i) {
            const auto col = b.col[c];
            const fattn_rel_bias_column desc = {col.row, col.q, col.tail};
            return fattn_rel_bias_get(rb, desc, k0 + i);
        });
    } else if (rb.kd == 20) {
        visit([&](int c, int i) { return fattn_rel_bias_slice_get<20, ncols, nkeys>(rb, b.col[c], scratch, s.first, c, i); });
    } else {
        visit([&](int c, int i) { return fattn_rel_bias_slice_get<0, ncols, nkeys>(rb, b.col[c], scratch, s.first, c, i); });
    }
    if constexpr (!sync_before_write) {
        if (s.count > 0) {
            __syncthreads();
        }
    }
}
