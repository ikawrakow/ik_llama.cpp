#pragma once

#include "fattn-rel-bias.cuh"

struct fattn_rel_bias_column {
    const char * row;
    int32_t q;
    float tail;
    float4 rs[8];
};

static __device__ __forceinline__ float fattn_rel_bias_dot(const fattn_rel_bias & rb, const fattn_rel_bias_column & c, int distance) {
    if (c.row) {
        return rb.dot(c.row, distance);
    }
    const char * w = rb.w + size_t(distance)*rb.nbw1;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (4*i < rb.kd) {
            const float4 v = ((const float4 *) w)[i];
            sum += c.rs[i].x*v.x;
            sum += c.rs[i].y*v.y;
            sum += c.rs[i].z*v.z;
            sum += c.rs[i].w*v.w;
        }
    }
    return sum;
}

static __device__ __forceinline__ bool fattn_rel_bias_can_load_float4(const fattn_rel_bias & rb, uintptr_t rows) {
    return rb.kd % 4 == 0 && rb.nb0 == sizeof(float) && rb.nbw0 == int(sizeof(float)) &&
            ((rows | uintptr_t(rb.w) | rb.nbw1) & 15) == 0;
}

static __device__ __forceinline__ fattn_rel_bias_column fattn_rel_bias_col(const fattn_rel_bias & rb, int iq, int head, bool need_tail = true) {
    const char * row = rb.rs + head*rb.nb1 + iq*rb.nb2;
    fattn_rel_bias_column c = {row, *(const int32_t *)(rb.pos_q + iq*rb.nbq)};
    if (rb.kd <= 32 && fattn_rel_bias_can_load_float4(rb, uintptr_t(row))) {
        c.row = nullptr;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (4*i < rb.kd) {
                c.rs[i] = ((const float4 *) row)[i];
            }
        }
    }
    if (need_tail) {
        c.tail = fattn_rel_bias_dot(rb, c, rb.last);
    }
    return c;
}

template <int n>
static __device__ __forceinline__ void fattn_rel_bias_cols(const fattn_rel_bias & rb, fattn_rel_bias_column (&cols)[n],
        int iq, int stride, int ne01, int head) {
#pragma unroll
    for (int j = 0; j < n; ++j) {
        cols[j] = fattn_rel_bias_col(rb, min(iq + j*stride, ne01 - 1), head);
    }
}

static __device__ __forceinline__ bool fattn_rel_bias_is_far(const fattn_rel_bias & rb, int32_t q_min, int32_t k_max) {
    return q_min >= k_max && uint32_t(q_min) - uint32_t(k_max) >= rb.last;
}

template <int nkeys>
static __device__ __forceinline__ bool fattn_rel_bias_tail_tile(const fattn_rel_bias & rb, int32_t q_min, int k0) {
    int32_t k_max = *(const int32_t *)(rb.pos_k + (k0 + threadIdx.x)*rb.nbk);
#pragma unroll
    for (int ik = threadIdx.x + WARP_SIZE; ik < nkeys; ik += WARP_SIZE) {
        k_max = max(k_max, *(const int32_t *)(rb.pos_k + (k0 + ik)*rb.nbk));
    }
    return __all_sync(0xffffffff, fattn_rel_bias_is_far(rb, q_min, k_max));
}

static __device__ __forceinline__ float fattn_rel_bias_at_pos(const fattn_rel_bias & rb, const fattn_rel_bias_column & c, int32_t k) {
    const uint32_t delta = uint32_t(c.q) - uint32_t(k);
    return c.q >= k && delta < rb.last ? fattn_rel_bias_dot(rb, c, delta) : c.tail;
}

static __device__ __forceinline__ float fattn_rel_bias_get(const fattn_rel_bias & rb, const fattn_rel_bias_column & c, int ik) {
    return fattn_rel_bias_at_pos(rb, c, *(const int32_t *)(rb.pos_k + ik*rb.nbk));
}

template <int ncols>
static __device__ __forceinline__ void fattn_rel_bias_fill_tile(const fattn_rel_bias & rb,
        const fattn_rel_bias_column (&cols)[ncols], int k0, int nkeys, float * dst, int stride, int tid, int nth) {
    for (int i = tid; i < nkeys; i += nth) {
        const int32_t k = *(const int32_t *)(rb.pos_k + (k0 + i)*rb.nbk);
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            dst[j*stride + i] = fattn_rel_bias_at_pos(rb, cols[j], k);
        }
    }
}

struct fattn_rel_bias_head {
    fattn_rel_bias_column col;
    int head, key0, key_step;
};

template <int ncols, int nkeys>
static __device__ __forceinline__ void fattn_rel_bias_fill_heads(const fattn_rel_bias & rb,
        const fattn_rel_bias_head & h, int k0, float * dst) {
    for (int i = h.key0; i < nkeys; i += h.key_step) {
        dst[i*ncols + h.head] = fattn_rel_bias_get(rb, h.col, k0 + i);
    }
}
