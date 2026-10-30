//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
#pragma once
//
// TQ2_T / TQK: a 128-weight block is 32 trellis steps of 4 weights. Step t has a 16-bit state s (the
// codebook index uses its low TQ_STATE_BITS bits) and its weights are
//   d * lut[x >> 21].{lo,hi}, d * lut[(x >> 10) & 2047].{lo,hi},  x = (s & TQ_STATE_MASK) * 0x9e3779b1
// where lut is the 2048-point fp16 codebook tq2t_lut_f16 read as u32 pairs (lo | hi << 16).
//
// TQP: a 128-weight block is 64 steps of 2 weights, d * lut[x >> 21].{lo,hi} (the first pair
// of the same hash), with 15-bit states read from a circular 64*K-bit stream.
//

#include "common.cuh"
#include "tq-lut-i8.cuh"

#define TQ_LUT_POINTS   2048
#define TQ_LANES        8   // lanes per 128-weight block
#define TQ_STEPS_LANE   4   // ints of 4 int8 weights per lane (= TQ/TQK steps per lane)
#define TQ_PAIRS_LANE   8   // codebook pairs per lane
#define VDR_TQ_Q8_1_MMVQ TQ_STEPS_LANE

static constexpr __host__ __device__ bool ggml_cuda_type_is_tqp(ggml_type type) {
    return type == GGML_TYPE_TQP5 || type == GGML_TYPE_TQP6 || type == GGML_TYPE_TQP7;
}

// Global (L1/L2-cached) view of the codebook as u32 pairs; the table is 16-byte aligned.
static __device__ __forceinline__ const uint32_t * tq_lut_global() {
    return (const uint32_t *) tq2t_lut_f16;
}

// Copy the codebook into shared memory (nthreads threads with linear id tid) and sync.
static __device__ __forceinline__ void tq_lut_load_shared(uint32_t * lut_s, const int tid, const int nthreads) {
    const uint32_t * lut_g = tq_lut_global();
    for (int i = tid; i < TQ_LUT_POINTS; i += nthreads) {
        lut_s[i] = lut_g[i];
    }
    __syncthreads();
}

static __device__ __forceinline__ float2 tq_unpack_pair(const uint32_t v) {
    return make_float2(__half2float(__ushort_as_half((unsigned short) (v & 0xFFFFu))),
                       __half2float(__ushort_as_half((unsigned short) (v >> 16))));
}

template <ggml_type type> struct tq_block_info;
template <> struct tq_block_info<GGML_TYPE_TQ2_T> { typedef block_tq2_t block_t; static constexpr int K = 8; };
template <> struct tq_block_info<GGML_TYPE_TQK6>  { typedef block_tqk6  block_t; static constexpr int K = 6; };
template <> struct tq_block_info<GGML_TYPE_TQK7>  { typedef block_tqk7  block_t; static constexpr int K = 7; };
template <> struct tq_block_info<GGML_TYPE_TQP5>  { typedef block_tqp5  block_t; static constexpr int K = 5; };
template <> struct tq_block_info<GGML_TYPE_TQP6>  { typedef block_tqp6  block_t; static constexpr int K = 6; };
template <> struct tq_block_info<GGML_TYPE_TQP7>  { typedef block_tqp7  block_t; static constexpr int K = 7; };
static_assert(QK_TQP == QK_TQK, "trellis block sizes differ");

// States of steps 4l..4l+3 of one block. qs is the block's qs array, 2-byte aligned
// (every trellis block has an even size and qs sits right after the fp16 scale).
template <ggml_type type>
static __device__ __forceinline__ void tq_states4(const uint8_t * qs, const int l, uint32_t s[TQ_STEPS_LANE]) {
    const uint16_t * qs16 = (const uint16_t *) qs;
    if constexpr (type == GGML_TYPE_TQ2_T) {
        // s_t = qs[(t + 31) % 32] << 8 | qs[t]
        const uint32_t w01  = qs16[2*l + 0];
        const uint32_t w23  = qs16[2*l + 1];
        const uint32_t prev = qs[(4*l + 31) & 31];
        const uint32_t b0 = w01 & 0xFF, b1 = w01 >> 8, b2 = w23 & 0xFF, b3 = w23 >> 8;
        s[0] = (prev << 8) | b0;
        s[1] = (b0   << 8) | b1;
        s[2] = (b1   << 8) | b2;
        s[3] = (b2   << 8) | b3;
    } else {
        // Step t's state is the 16 stream bits from bit (31 - t)*K (circular over 32*K bits,
        // i.e. 2*K u16 words). Steps 4l..4l+3 all lie in the four words from
        // wb = off3 >> 4, off3 = (28 - 4l)*K, since (off3 & 15) + 3*K + 16 <= 64.
        constexpr int K      = tq_block_info<type>::K;
        constexpr int nwords = 2*K;
        const int off3 = (28 - 4*l)*K;
        const int wb   = off3 >> 4;
        const int r    = off3 & 15;
        const int wi1  = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
        const int wi2  = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
        const int wi3  = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
        const uint64_t v = (uint64_t) qs16[wb] | ((uint64_t) qs16[wi1] << 16) |
                           ((uint64_t) qs16[wi2] << 32) | ((uint64_t) qs16[wi3] << 48);
#pragma unroll
        for (int i = 0; i < TQ_STEPS_LANE; ++i) {
            s[i] = (uint32_t) (v >> (r + (3 - i)*K)) & 0xFFFFu;
        }
    }
}

// TQP: the 15-bit states of steps 8l..8l+7 (s[j] = step 8l + j). Step t's window starts at stream
// bit (63 - t)*K, so the lane's steps lie in the bits from S = (56 - 8l)*K: step 8l + j at S + (7 - j)*K.
// S = 8*(7 - l)*K, so r = S & 15 is 0 or 8 and r + 7*K + 15 <= 72: five u16 words (wrapped modulo the
// 4*K words of the stream) cover it, the fifth only when r + 7*K + 15 > 64 (never for K = 5).
template <ggml_type type>
static __device__ __forceinline__ void tqp_states8(const uint8_t * qs, const int l, uint32_t s[TQ_PAIRS_LANE]) {
    constexpr int K      = tq_block_info<type>::K;
    constexpr int nwords = 4*K;
    const uint16_t * qs16 = (const uint16_t *) qs;
    const int S   = (56 - 8*l)*K;
    const int wb  = S >> 4;
    const int r   = S & 15;
    const int wi1 = wb + 1 >= nwords ? wb + 1 - nwords : wb + 1;
    const int wi2 = wb + 2 >= nwords ? wb + 2 - nwords : wb + 2;
    const int wi3 = wb + 3 >= nwords ? wb + 3 - nwords : wb + 3;
    uint64_t v = (uint64_t) qs16[wb] | ((uint64_t) qs16[wi1] << 16) |
                 ((uint64_t) qs16[wi2] << 32) | ((uint64_t) qs16[wi3] << 48);
    v >>= r;
    if constexpr (8 + 7*K + TQ_STATE_BITS > 64) {
        const int wi4 = wb + 4 >= nwords ? wb + 4 - nwords : wb + 4;
        // r is 0 or 8; for r = 0 the fifth word is not needed (and a shift by 64 is undefined)
        v |= r ? (uint64_t) qs16[wi4] << (64 - r) : 0;
    }
#pragma unroll
    for (int j = 0; j < TQ_PAIRS_LANE; ++j) {
        s[j] = (uint32_t) (v >> ((7 - j)*K)) & TQ_STATE_MASK;
    }
}

// The 8 codebook indices of lane l in weight-pair order: pair p = weights 16l + 2p, 16l + 2p + 1.
template <ggml_type type>
static __device__ __forceinline__ void tq_lane_idx(const uint8_t * qs, const int l, uint32_t idx[TQ_PAIRS_LANE]) {
    if constexpr (ggml_cuda_type_is_tqp(type)) {
        uint32_t s[TQ_PAIRS_LANE];
        tqp_states8<type>(qs, l, s);
#pragma unroll
        for (int j = 0; j < TQ_PAIRS_LANE; ++j) {
            idx[j] = (s[j] * 0x9e3779b1u) >> 21;
        }
    } else {
        uint32_t s[TQ_STEPS_LANE];
        tq_states4<type>(qs, l, s);
#pragma unroll
        for (int i = 0; i < TQ_STEPS_LANE; ++i) {
            const uint32_t x = (s[i] & TQ_STATE_MASK) * 0x9e3779b1u;
            idx[2*i + 0] = x >> 21;
            idx[2*i + 1] = (x >> 10) & 2047u;
        }
    }
}

// Dequantize lane l (16 weights) of block x into out[0..15].
template <ggml_type type>
static __device__ __forceinline__ void tq_dequant_lane(const void * x, const int l, const uint32_t * lut, float out[16]) {
    typedef typename tq_block_info<type>::block_t block_t;
    const block_t * b = (const block_t *) x;
    const float d = __half2float(b->d);
    uint32_t idx[TQ_PAIRS_LANE];
    tq_lane_idx<type>(b->qs, l, idx);
#pragma unroll
    for (int p = 0; p < TQ_PAIRS_LANE; ++p) {
        const float2 w = tq_unpack_pair(lut[idx[p]]);
        out[2*p + 0] = d*w.x;
        out[2*p + 1] = d*w.y;
    }
}

// mmvq dot product of one lane of block kbx against q8_1 values. bq8_1 points at the
// q8_1 block aligned with the start of block kbx (4 q8_1 blocks per trellis block);
// iqs = VDR_TQ_Q8_1_MMVQ * lane.
template <ggml_type type>
static __device__ __forceinline__ float vec_dot_tq_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    typedef typename tq_block_info<type>::block_t block_t;
    const block_t * b = (const block_t *) vbq + kbx;
    const int l = iqs / VDR_TQ_Q8_1_MMVQ;

    uint32_t idx[TQ_PAIRS_LANE];
    tq_lane_idx<type>(b->qs, l, idx);

    const block_q8_1 * b8 = bq8_1 + l/2;
    const int * q8 = (const int *) b8->qs + 4*(l & 1);

    // int8 codebook (one global scale) + dp4a: two codebook pairs (2 bytes each) packed into
    // one int give 4 weights, i.e. one integer dot product with 4 q8_1 values.
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < TQ_STEPS_LANE; ++i) {
        const uint32_t w = (uint32_t) tq_lut_i8[idx[2*i]] | ((uint32_t) tq_lut_i8[idx[2*i + 1]] << 16);
        sumi = ggml_cuda_dp4a((int) w, q8[i], sumi);
    }
    return __half2float(b->d) * (__low2float(b8->ds) * TQ_LUT_I8_SCALE) * (float) sumi;
}

// MMQ (batched) path: lane l of block x as 16 int8 values (4 packed ints, weight order) with
// one float scale per lane, i.e. weights ~= scale * q. The lane's ints are the int8 codebook
// entries themselves (Q8_0_16 tile layout, as requantized by IQ2_XS / Q3_K) and the scale is
// d * TQ_LUT_I8_SCALE: the codebook's int8 error (0.85% of its std) stays far below the
// trellis quantization error.
template <ggml_type type>
static __device__ __forceinline__ float tq_lane_to_q8(const void * x, const int l, int q[TQ_STEPS_LANE]) {
    typedef typename tq_block_info<type>::block_t block_t;
    const block_t * b = (const block_t *) x;

    uint32_t idx[TQ_PAIRS_LANE];
    tq_lane_idx<type>(b->qs, l, idx);

#pragma unroll
    for (int i = 0; i < TQ_STEPS_LANE; ++i) {
        q[i] = (int) ((uint32_t) tq_lut_i8[idx[2*i]] | ((uint32_t) tq_lut_i8[idx[2*i + 1]] << 16));
    }
    return __half2float(b->d) * TQ_LUT_I8_SCALE;
}
