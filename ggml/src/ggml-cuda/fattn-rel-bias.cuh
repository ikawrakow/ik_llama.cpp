#pragma once

#include "common.cuh"

struct fattn_rel_bias {
    const char * rs;
    const char * w;
    const char * pos_q;
    const char * pos_k;
    uint32_t last;
    int kd;
    size_t nb0, nb1, nb2, nbq, nbk;
    int nbw0, nbw1;

    __device__ __forceinline__ float dot(const char * row, uint32_t distance) const {
        const char * x = w + size_t(distance)*nbw1;
        float sum = 0.0f;
        for (int d = 0; d < kd; ++d) {
            sum += *(const float *)(row + d*nb0) * *(const float *)(x + d*nbw0);
        }
        return sum;
    }

    __device__ __forceinline__ float get(int iq, int head, int ik) const {
        const int32_t q = *(const int32_t *)(pos_q + iq*nbq);
        const int32_t k = *(const int32_t *)(pos_k + ik*nbk);
        const uint32_t delta = q >= k ? min(uint32_t(q) - uint32_t(k), last) : last;
        return dot(rs + head*nb1 + iq*nb2, delta);
    }
};

static inline fattn_rel_bias get_fattn_rel_bias(const ggml_tensor * dst) {
    const ggml_tensor * rs = dst->src[6];
    if (!rs) return {};
    const ggml_tensor * q = dst->src[7];
    const ggml_tensor * k = dst->src[8];
    const ggml_tensor * w = dst->src[9];
    return {(const char *)rs->data, (const char *)w->data, (const char *)q->data, (const char *)k->data,
            uint32_t(w->ne[1] - 1), int(rs->ne[0]), rs->nb[0], rs->nb[1], rs->nb[2], q->nb[0], k->nb[0],
            int(w->nb[0]), int(w->nb[1])};
}
