#pragma once

#include "fattn-rel-bias.cuh"

struct fattn_rel_bias_column {
    const char * row;
    int32_t q;
    float tail;
};

static __device__ __forceinline__ fattn_rel_bias_column fattn_rel_bias_col(const fattn_rel_bias & rb, int iq, int head) {
    const char * row = rb.table + head*rb.nb1 + iq*rb.nb2;
    return {row, *(const int32_t *)(rb.pos_q + iq*rb.nbq), *(const float *)(row + rb.last*rb.nb0)};
}

template <int nkeys>
static __device__ __forceinline__ bool fattn_rel_bias_tail_tile(const fattn_rel_bias & rb, int32_t q_min, int k0) {
    int32_t k_max = *(const int32_t *)(rb.pos_k + (k0 + threadIdx.x)*rb.nbk);
#pragma unroll
    for (int ik = threadIdx.x + WARP_SIZE; ik < nkeys; ik += WARP_SIZE) {
        k_max = max(k_max, *(const int32_t *)(rb.pos_k + (k0 + ik)*rb.nbk));
    }
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        k_max = max(k_max, __shfl_xor_sync(0xFFFFFFFF, k_max, offset, WARP_SIZE));
    }
    return q_min >= k_max && uint32_t(q_min) - uint32_t(k_max) >= rb.last;
}

static __device__ __forceinline__ float fattn_rel_bias_get(const fattn_rel_bias & rb, const fattn_rel_bias_column & c, int ik) {
    const int32_t k = *(const int32_t *)(rb.pos_k + ik*rb.nbk);
    const uint32_t delta = uint32_t(c.q) - uint32_t(k);
    return c.q >= k && delta < rb.last ? *(const float *)(c.row + delta*rb.nb0) : c.tail;
}
