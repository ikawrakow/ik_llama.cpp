#pragma once

#include "common.cuh"

struct fattn_rel_bias {
    const char * table;
    const char * pos_q;
    const char * pos_k;
    uint32_t last;
    size_t nb0, nb1, nb2, nbq, nbk;

    __device__ __forceinline__ float get(int iq, int head, int ik) const {
        const int32_t q = *(const int32_t *)(pos_q + iq*nbq);
        const int32_t k = *(const int32_t *)(pos_k + ik*nbk);
        const uint32_t delta = q >= k ? min(uint32_t(q) - uint32_t(k), last) : last;
        return *(const float *)(table + delta*nb0 + head*nb1 + iq*nb2);
    }
};

static inline fattn_rel_bias get_fattn_rel_bias(const ggml_tensor * dst) {
    const ggml_tensor * t = dst->src[6];
    if (!t) return {};
    const ggml_tensor * q = dst->src[7];
    const ggml_tensor * k = dst->src[8];
    return {(const char *)t->data, (const char *)q->data, (const char *)k->data, uint32_t(t->ne[0] - 1),
            t->nb[0], t->nb[1], t->nb[2], q->nb[0], k->nb[0]};
}
