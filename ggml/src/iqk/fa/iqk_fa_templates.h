// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp fenc=utf-8 :vi
//
//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include "iqk/iqk_config.h"

#if defined IQK_IMPLEMENT && defined GGML_IQK_FLASH_ATTENTION

#include <cstring>
#include <type_traits>
#include <vector>

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "iqk/iqk_quantize.h"
#include "iqk/iqk_gemm_floats.h"
#include "iqk/iqk_gemm_kquants.h"
#include "iqk/iqk_gemm_legacy_quants.h"
#include "iqk/iqk_utils.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

// clang-format off

namespace {

struct BaseHelper {
    BaseHelper(const char * data, int stride) : data(data), block(data), stride(stride) {}

    //inline void set_block(int k1) { block = data + k1*k_step*stride; }
    inline void reset_block() { block = data; }
    inline void next_block(int step) { block += step*stride; }
    inline const char * lblock(int l1) const { return block + l1*stride; }

    const char * data;
    const char * block;
    int stride;

};

struct F16 {
#ifdef __AVX512F__
    using Data = __m512;
    constexpr static int block_size = 16;
    constexpr static int num_registers = 32;
    constexpr static int q_step = 8;
    static inline Data zero() { return _mm512_setzero_ps(); }
    static inline Data load(const char * ptr, int i) { return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)ptr + i)); }
    static inline Data set1(float val) { return _mm512_set1_ps(val); }
    static inline Data mul(Data v1, Data v2) { return _mm512_mul_ps(v1, v2); }
    static inline Data sub(Data v1, Data v2) { return _mm512_sub_ps(v1, v2); }
    static inline Data load(const float * ptr) { return _mm512_loadu_ps(ptr); }
    static inline void store(float * ptr, Data data) { _mm512_storeu_ps(ptr, data); }
    static inline Data fmadd(Data prev, Data v1, Data v2) { return _mm512_fmadd_ps(v1, v2, prev); }
    static inline float reduce_max(Data data) { return _mm512_reduce_max_ps(data); }
    static inline float reduce_add(Data data) { return _mm512_reduce_add_ps(data); }
    static inline Data max(Data v1, Data v2) { return _mm512_max_ps(v1, v2); }
    static inline Data add(Data v1, Data v2) { return _mm512_add_ps(v1, v2); }
    static inline Data set4(const float * ptr) {
        auto v128 = _mm_loadu_ps(ptr);
        auto v256 = _mm256_set_m128(v128, v128);
        return _mm512_insertf32x8(_mm512_castps256_ps512(v256), v256, 1);
    }
    static inline void set4(const float * ptr, Data * vs) {
        auto v = set4(ptr);
        vs[0] = _mm512_shuffle_ps(v, v, 0x00);
        vs[1] = _mm512_shuffle_ps(v, v, 0x55);
        vs[2] = _mm512_shuffle_ps(v, v, 0xaa);
        vs[3] = _mm512_shuffle_ps(v, v, 0xff);
    }
    static inline Data fmadd_lane0(Data prev, Data v1, Data v2) { return _mm512_fmadd_ps(v1, _mm512_shuffle_ps(v2, v2, 0x00), prev); }
    static inline Data fmadd_lane1(Data prev, Data v1, Data v2) { return _mm512_fmadd_ps(v1, _mm512_shuffle_ps(v2, v2, 0x55), prev); }
    static inline Data fmadd_lane2(Data prev, Data v1, Data v2) { return _mm512_fmadd_ps(v1, _mm512_shuffle_ps(v2, v2, 0xaa), prev); }
    static inline Data fmadd_lane3(Data prev, Data v1, Data v2) { return _mm512_fmadd_ps(v1, _mm512_shuffle_ps(v2, v2, 0xff), prev); }
#elif defined __AVX2__
    using Data = __m256;
    constexpr static int block_size = 8;
    constexpr static int num_registers = 16;
    constexpr static int q_step = 8;
    static inline Data zero() { return _mm256_setzero_ps(); }
    static inline Data load(const char * ptr, int i) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)ptr + i)); }
    static inline Data set1(float val) { return _mm256_set1_ps(val); }
    static inline Data mul(Data v1, Data v2) { return _mm256_mul_ps(v1, v2); }
    static inline Data load(const float * ptr) { return _mm256_loadu_ps(ptr); }
    static inline Data sub(Data v1, Data v2) { return _mm256_sub_ps(v1, v2); }
    static inline void store(float * ptr, Data data) { _mm256_storeu_ps(ptr, data); }
    static inline Data fmadd(Data prev, Data v1, Data v2) { return _mm256_fmadd_ps(v1, v2, prev); }
    static inline float reduce_max(Data data) { return hmax_float_8(data); }
    static inline float reduce_add(Data data) { return hsum_float_8(data); }
    static inline Data max(Data v1, Data v2) { return _mm256_max_ps(v1, v2); }
    static inline Data add(Data v1, Data v2) { return _mm256_add_ps(v1, v2); }
    static inline Data set4(const float * ptr) {
        auto v128 = _mm_loadu_ps(ptr);
        return _mm256_set_m128(v128, v128);
    }
    static inline void set4(const float * ptr, Data * vs) {
        auto v = set4(ptr);
        vs[0] = _mm256_shuffle_ps(v, v, 0x00);
        vs[1] = _mm256_shuffle_ps(v, v, 0x55);
        vs[2] = _mm256_shuffle_ps(v, v, 0xaa);
        vs[3] = _mm256_shuffle_ps(v, v, 0xff);
    }
    static inline Data fmadd_lane0(Data prev, Data v1, Data v2) { return _mm256_fmadd_ps(v1, _mm256_shuffle_ps(v2, v2, 0x00), prev); }
    static inline Data fmadd_lane1(Data prev, Data v1, Data v2) { return _mm256_fmadd_ps(v1, _mm256_shuffle_ps(v2, v2, 0x55), prev); }
    static inline Data fmadd_lane2(Data prev, Data v1, Data v2) { return _mm256_fmadd_ps(v1, _mm256_shuffle_ps(v2, v2, 0xaa), prev); }
    static inline Data fmadd_lane3(Data prev, Data v1, Data v2) { return _mm256_fmadd_ps(v1, _mm256_shuffle_ps(v2, v2, 0xff), prev); }
#else
    using Data = float16x8_t;
    constexpr static int block_size = 8;
    //constexpr static int num_registers = 32;
    //constexpr static int q_step = 8;
    static inline Data zero() { return vdupq_n_f16(0); }
    static inline Data load(const char * ptr, int i) { return vld1q_f16((const float16_t *)ptr + block_size*i); }
    static inline Data load(const float16_t * ptr, int i) { return vld1q_f16(ptr + block_size*i); }
    static inline Data load(const float16_t * ptr) { return vld1q_f16(ptr); }
    static inline Data load(const float * ptr) {
        auto val1 = vld1q_f32(ptr);
        auto val2 = vld1q_f32(ptr+4);
        return vcombine_f16(vcvt_f16_f32(val1), vcvt_f16_f32(val2));
    }
    static inline Data set1(float val) { return vdupq_n_f16(val); }
    static inline Data mul(Data v1, Data v2) { return vmulq_f16(v1, v2); }
    static inline Data sub(Data v1, Data v2) { return vsubq_f16(v1, v2); }
    static inline void store(float * ptr, Data data) {
        vst1q_f32(ptr+0, vcvt_f32_f16(vget_low_f16(data)));
        vst1q_f32(ptr+4, vcvt_f32_f16(vget_high_f16(data)));
    }
    static inline void store(float16_t * ptr, Data data) { vst1q_f16(ptr, data); }
    static inline void store(float * ptr, float32x4_t data) { vst1q_f32(ptr, data); }
    static inline Data fmadd(Data prev, Data v1, Data v2) { return vfmaq_f16(prev, v1, v2); }
    static inline float reduce_max(Data data) { return vmaxvq_f16(data); }
    static inline float reduce_add(Data data) {
        auto sum = vadd_f16(vget_low_f16(data), vget_high_f16(data));
        return vaddvq_f32(vcvt_f32_f16(sum));
    }
    static inline Data max(Data v1, Data v2) { return vmaxq_f16(v1, v2); }
    static inline Data add(Data v1, Data v2) { return vaddq_f16(v1, v2); }
    static inline float16x4_t set4(const float * ptr) {
        auto val32 = vld1q_f32(ptr);
        return vcvt_f16_f32(val32);
    }
    static inline Data fmadd_lane0(Data prev, Data v1, float16x4_t v2) { return vfmaq_lane_f16(prev, v1, v2, 0); }
    static inline Data fmadd_lane1(Data prev, Data v1, float16x4_t v2) { return vfmaq_lane_f16(prev, v1, v2, 1); }
    static inline Data fmadd_lane2(Data prev, Data v1, float16x4_t v2) { return vfmaq_lane_f16(prev, v1, v2, 2); }
    static inline Data fmadd_lane3(Data prev, Data v1, float16x4_t v2) { return vfmaq_lane_f16(prev, v1, v2, 3); }
#endif
    template <int k_step> static inline float reduce_max(const Data * data) {
        return reduce_T<k_step, &F16::max, &F16::reduce_max>(data);
    }
    template <int k_step> static inline float reduce_add(const Data * data) {
        return reduce_T<k_step, &F16::add, &F16::reduce_add>(data);
    }
    template <int k_step, Data (*Op_combine)(Data, Data), float (*Op)(Data)>
    static float reduce_T(const Data * data) {
        float result;
        if constexpr (k_step/block_size == 1) {
            result = Op(data[0]);
        }
        else if constexpr (k_step/block_size == 2) {
            result = Op(Op_combine(data[0], data[1]));
        }
        else {
            auto vmax = Op_combine(data[0], data[1]);
            for (int l = 2; l < k_step/block_size; ++l) vmax = Op_combine(vmax, data[l]);
            result = Op(vmax);
        }
        return result;
    }
};

struct HelperF16 final : public BaseHelper {
    using Base = BaseHelper;
    HelperF16(const char * data, int stride) : Base(data, stride) {}

    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        //auto dr = (const ggml_half *)Base::lblock(l1);
        auto dr = Base::lblock(l1);
        v1 = F16::load(dr, i + 0);
        v2 = F16::load(dr, i + 1);
    }
};

template <int D> struct block_q8_KV {
    float d;
    int   s;
    int8_t qs[D];
};

template <int D>
struct HelperQ8KV final : public BaseHelper {
    using Base = BaseHelper;
    using block_q8 = block_q8_KV<D>;
    constexpr static ggml_type type = GGML_TYPE_Q8_KV;
    constexpr static int block_size_q = D;
    HelperQ8KV(const char * data, int stride) : Base(data, stride) {}

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        auto q8 = (const block_q8_KV<D> *)Base::lblock(l1);
#ifdef __aarch64__
        auto vd = F16::set1(q8->d);
        auto qs = vld1_s8_x2(q8->qs + 8*i);
        v1 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(qs.val[0])));
        v2 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(qs.val[1])));
#else
        auto vd = F16::set1(q8->d);
#ifdef __AVX512F__
        v1 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)q8->qs+i+0))));
        v2 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)q8->qs+i+1))));
#else
        v1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(q8->qs+8*i+0)))));
        v2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(q8->qs+8*i+8)))));
#endif
#endif
    }
};

struct HelperQ80 final : public BaseHelper {
    using Base = BaseHelper;
    constexpr static ggml_type type = GGML_TYPE_Q8_0;
//#ifdef HAVE_FANCY_SIMD
#ifdef __AVX2__
    using block_q8 = block_q8_2;
    constexpr static int block_size_q = QK8_2;
#else
    using block_q8 = block_q8_0;
    constexpr static int block_size_q = QK8_0;
#endif
    HelperQ80(const char * data, int stride) : Base(data, stride) {}

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        int j = F16::block_size*i;
        auto dl = (const block_q8_0 *)Base::lblock(l1) + j/QK8_0;
#ifdef __aarch64__
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d));
        int ii = j%QK8_0;
        auto qs = vld1_s8_x2(dl->qs + ii);
        v1 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(qs.val[0])));
        v2 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(qs.val[1])));
#else
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d));
#ifdef __AVX512F__
        v1 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)dl->qs+0))));
        v2 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)dl->qs+1))));
#else
        int ii = j%QK8_0;
        v1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(dl->qs+ii+0)))));
        v2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(dl->qs+ii+8)))));
#endif
#endif
    }

    template <int D>
    static inline void convert(int nq, int stride_q, const float * q, block_q8_0 * y) {
        for (int i = 0; i < nq; ++i) {
            quantize_row_q8_0_x4(q, y, D);
            q += stride_q;
            y += D/QK8_0;
        }
    }

    template <int D>
    static inline void convert(int nq, int stride_q, const float * q, block_q8_1 * y) {
        for (int i = 0; i < nq; ++i) {
            quantize_row_q8_1_x4(q, y, D);
            q += stride_q;
            y += D/QK8_1;
        }
    }

    template <int D>
    static inline void convert(int nq, int stride_q, const float * q, block_q8_2 * y) {
        for (int i = 0; i < nq; ++i) {
            quantize_row_q8_2_x4(q, y, D);
            q += stride_q;
            y += D/QK8_2;
        }
    }

    template <int D>
    static inline void convert(int nq, int stride_q, const float * q, block_q8_KV<D> * y) {
        for (int i = 0; i < nq; ++i) {
            quantize_row_q8_KV(q, y, D);
            q += stride_q;
            ++y;
        }
    }
};

template <int D>
struct HelperQ80R8 : public BaseHelper {
    using Base = BaseHelper;
    constexpr static ggml_type type = GGML_TYPE_Q8_0_R8;
#ifdef __AVX2__
    constexpr static int block_size_q = QK8_2;
    using block_q8 = block_q8_2;
#else
    constexpr static int block_size_q = QK8_0;
    using block_q8 = block_q8_0;
#endif
    HelperQ80R8(const char * data, int stride) : Base(data, stride) {}
    HelperQ80R8(int nk, const HelperQ80& q8) : Base(q8.data, q8.stride) {
        r4 = repack(nk, q8);
        Base::data = (const char *)r4.data();
        Base::stride = (D/QK8_0)*sizeof(block_q8_0);
    }

    static void repack(int nk, const char * q8_data, int q8_stride, block_q8_0_r8 * y) {
        constexpr int nblock = D/QK8_0;
        const block_q8_0 * x8[8];
#ifdef __ARM_NEON
        int8x16x2_t m0, m1, m2, m3;
#endif
        for (int row = 0; row < nk; row += 8) {
            for (int k = 0; k < 8; ++k) x8[k] = (const block_q8_0 *)(q8_data + (row + k)*q8_stride);
            for (int ib = 0; ib < nblock; ++ib) {
                for (int k = 0; k < 8; ++k) y[ib].d[k] = x8[k][ib].d;
#ifdef __AVX2__
                auto m0 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[4][ib].qs), _mm_loadu_si128((const __m128i *)x8[0][ib].qs));
                auto m1 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[5][ib].qs), _mm_loadu_si128((const __m128i *)x8[1][ib].qs));
                auto m2 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[6][ib].qs), _mm_loadu_si128((const __m128i *)x8[2][ib].qs));
                auto m3 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[7][ib].qs), _mm_loadu_si128((const __m128i *)x8[3][ib].qs));
                auto t0 = _mm256_unpacklo_epi32(m0, m1);
                auto t1 = _mm256_unpacklo_epi32(m2, m3);
                auto t2 = _mm256_unpackhi_epi32(m0, m1);
                auto t3 = _mm256_unpackhi_epi32(m2, m3);
                m0 = _mm256_unpacklo_epi64(t0, t1);
                m1 = _mm256_unpackhi_epi64(t0, t1);
                m2 = _mm256_unpacklo_epi64(t2, t3);
                m3 = _mm256_unpackhi_epi64(t2, t3);
//#ifdef HAVE_FANCY_SIMD
//                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
//                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
//                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
//                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
//#endif
                _mm256_storeu_si256((__m256i *)y[ib].qs + 0, m0);
                _mm256_storeu_si256((__m256i *)y[ib].qs + 1, m1);
                _mm256_storeu_si256((__m256i *)y[ib].qs + 2, m2);
                _mm256_storeu_si256((__m256i *)y[ib].qs + 3, m3);
                m0 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[4][ib].qs+1), _mm_loadu_si128((const __m128i *)x8[0][ib].qs+1));
                m1 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[5][ib].qs+1), _mm_loadu_si128((const __m128i *)x8[1][ib].qs+1));
                m2 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[6][ib].qs+1), _mm_loadu_si128((const __m128i *)x8[2][ib].qs+1));
                m3 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[7][ib].qs+1), _mm_loadu_si128((const __m128i *)x8[3][ib].qs+1));
                t0 = _mm256_unpacklo_epi32(m0, m1);
                t1 = _mm256_unpacklo_epi32(m2, m3);
                t2 = _mm256_unpackhi_epi32(m0, m1);
                t3 = _mm256_unpackhi_epi32(m2, m3);
                m0 = _mm256_unpacklo_epi64(t0, t1);
                m1 = _mm256_unpackhi_epi64(t0, t1);
                m2 = _mm256_unpacklo_epi64(t2, t3);
                m3 = _mm256_unpackhi_epi64(t2, t3);
//#ifdef HAVE_FANCY_SIMD
//                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
//                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
//                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
//                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
//#endif
                _mm256_storeu_si256((__m256i *)y[ib].qs + 4, m0);
                _mm256_storeu_si256((__m256i *)y[ib].qs + 5, m1);
                _mm256_storeu_si256((__m256i *)y[ib].qs + 6, m2);
                _mm256_storeu_si256((__m256i *)y[ib].qs + 7, m3);
#elif defined __ARM_NEON
                for (int l = 0; l < 2; ++l) {
                    m0.val[0] = vld1q_s8(x8[0][ib].qs+16*l); m0.val[1] = vld1q_s8(x8[4][ib].qs+16*l);
                    m1.val[0] = vld1q_s8(x8[1][ib].qs+16*l); m1.val[1] = vld1q_s8(x8[5][ib].qs+16*l);
                    m2.val[0] = vld1q_s8(x8[2][ib].qs+16*l); m2.val[1] = vld1q_s8(x8[6][ib].qs+16*l);
                    m3.val[0] = vld1q_s8(x8[3][ib].qs+16*l); m3.val[1] = vld1q_s8(x8[7][ib].qs+16*l);
                    auto row01 = vtrnq_s32(vreinterpretq_s32_s8(m0.val[0]), vreinterpretq_s32_s8(m1.val[0]));
                    auto row23 = vtrnq_s32(vreinterpretq_s32_s8(m2.val[0]), vreinterpretq_s32_s8(m3.val[0]));
                    m0.val[0] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                    m1.val[0] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                    m2.val[0] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                    m3.val[0] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                    row01 = vtrnq_s32(vreinterpretq_s32_s8(m0.val[1]), vreinterpretq_s32_s8(m1.val[1]));
                    row23 = vtrnq_s32(vreinterpretq_s32_s8(m2.val[1]), vreinterpretq_s32_s8(m3.val[1]));
                    m0.val[1] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                    m1.val[1] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                    m2.val[1] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                    m3.val[1] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                    vst1q_s8_x2(y[ib].qs +  0 + 128*l, m0);
                    vst1q_s8_x2(y[ib].qs + 32 + 128*l, m1);
                    vst1q_s8_x2(y[ib].qs + 64 + 128*l, m2);
                    vst1q_s8_x2(y[ib].qs + 96 + 128*l, m3);
                }
#else
                for (int l = 0; l < 4; ++l) {
                    for (int k = 0; k < 8; ++k) for (int i = 0; i < 4; ++i) {
                        y[ib].qs[32*l+4*k+i+  0] = x8[k][ib].qs[i+4*l+ 0];
                        y[ib].qs[32*l+4*k+i+128] = x8[k][ib].qs[i+4*l+16];
                    }
                }
#endif
            }
            y += nblock;
        }
    }

    static std::vector<block_q8_0_r8> repack(int nk, const HelperQ80& q8) {
        static_assert(D%QK8_0 == 0);
        GGML_ASSERT(nk%8 == 0);
        constexpr int nblock = D/QK8_0;
        std::vector<block_q8_0_r8> result(nblock * nk/8);
        auto y = result.data();
        repack(nk, q8.data, q8.stride, y);
        return result;
    }

    std::vector<block_q8_0_r8> r4;
};

// TODO: unite this with the above
template <int D>
struct HelperQ8KVR8 : public BaseHelper {
    using Base = BaseHelper;
    constexpr static ggml_type type = GGML_TYPE_Q8_KV_R8;
    constexpr static int block_size_q = D;
    using block_q8 = block_q8_KV<D>;

    struct block_q8_KV_r8 {
        float  d[8];
        int8_t qs[8*D];
    };

    HelperQ8KVR8(int nk, const HelperQ8KV<D>& q8) : Base(q8.data, q8.stride) {
        r4 = repack(nk, q8);
        Base::data = (const char *)r4.data();
        Base::stride = sizeof(block_q8_KV_r8)/8;
    }

    static std::vector<block_q8_KV_r8> repack(int nk, const HelperQ8KV<D>& q8) {
        static_assert(D%32 == 0);
        GGML_ASSERT(nk%8 == 0);
        std::vector<block_q8_KV_r8> result(nk/8);
        auto y = result.data();
#ifdef __ARM_NEON
        int8x16x2_t m0, m1, m2, m3;
#endif
        const int8_t * x8[8];
        for (int ix = 0; ix < nk/8; ++ix) {
            for (int k = 0; k < 8; ++k) {
                auto dptr = (const float *)(q8.data + (8*ix + k)*q8.stride);
                y[ix].d[k] = dptr[0];
                x8[k] = (const int8_t *)(dptr + 2);
            }
            for (int ib = 0; ib < D/16; ++ib) {
#ifdef __AVX2__
                auto m0 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[4]+ib), _mm_loadu_si128((const __m128i *)x8[0]+ib));
                auto m1 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[5]+ib), _mm_loadu_si128((const __m128i *)x8[1]+ib));
                auto m2 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[6]+ib), _mm_loadu_si128((const __m128i *)x8[2]+ib));
                auto m3 = MM256_SET_M128I(_mm_loadu_si128((const __m128i *)x8[7]+ib), _mm_loadu_si128((const __m128i *)x8[3]+ib));
                auto t0 = _mm256_unpacklo_epi32(m0, m1);
                auto t1 = _mm256_unpacklo_epi32(m2, m3);
                auto t2 = _mm256_unpackhi_epi32(m0, m1);
                auto t3 = _mm256_unpackhi_epi32(m2, m3);
                m0 = _mm256_unpacklo_epi64(t0, t1);
                m1 = _mm256_unpackhi_epi64(t0, t1);
                m2 = _mm256_unpacklo_epi64(t2, t3);
                m3 = _mm256_unpackhi_epi64(t2, t3);
//#ifdef HAVE_FANCY_SIMD
//                m0 = _mm256_add_epi8(m0, _mm256_set1_epi8(127));
//                m1 = _mm256_add_epi8(m1, _mm256_set1_epi8(127));
//                m2 = _mm256_add_epi8(m2, _mm256_set1_epi8(127));
//                m3 = _mm256_add_epi8(m3, _mm256_set1_epi8(127));
//#endif
                _mm256_storeu_si256((__m256i *)y[ix].qs + 4*ib+0, m0);
                _mm256_storeu_si256((__m256i *)y[ix].qs + 4*ib+1, m1);
                _mm256_storeu_si256((__m256i *)y[ix].qs + 4*ib+2, m2);
                _mm256_storeu_si256((__m256i *)y[ix].qs + 4*ib+3, m3);
#elif defined __ARM_NEON
                // TODO
                m0.val[0] = vld1q_s8(x8[0]+16*ib); m0.val[1] = vld1q_s8(x8[4]+16*ib);
                m1.val[0] = vld1q_s8(x8[1]+16*ib); m1.val[1] = vld1q_s8(x8[5]+16*ib);
                m2.val[0] = vld1q_s8(x8[2]+16*ib); m2.val[1] = vld1q_s8(x8[6]+16*ib);
                m3.val[0] = vld1q_s8(x8[3]+16*ib); m3.val[1] = vld1q_s8(x8[7]+16*ib);
                auto row01 = vtrnq_s32(vreinterpretq_s32_s8(m0.val[0]), vreinterpretq_s32_s8(m1.val[0]));
                auto row23 = vtrnq_s32(vreinterpretq_s32_s8(m2.val[0]), vreinterpretq_s32_s8(m3.val[0]));
                m0.val[0] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                m1.val[0] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                m2.val[0] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                m3.val[0] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                row01 = vtrnq_s32(vreinterpretq_s32_s8(m0.val[1]), vreinterpretq_s32_s8(m1.val[1]));
                row23 = vtrnq_s32(vreinterpretq_s32_s8(m2.val[1]), vreinterpretq_s32_s8(m3.val[1]));
                m0.val[1] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                m1.val[1] = vreinterpretq_s8_s64(vtrn1q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                m2.val[1] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[0]), vreinterpretq_s64_s32(row23.val[0])));
                m3.val[1] = vreinterpretq_s8_s64(vtrn2q_s64(vreinterpretq_s64_s32(row01.val[1]), vreinterpretq_s64_s32(row23.val[1])));
                vst1q_s8_x2(y[ix].qs +  0 + 128*ib, m0);
                vst1q_s8_x2(y[ix].qs + 32 + 128*ib, m1);
                vst1q_s8_x2(y[ix].qs + 64 + 128*ib, m2);
                vst1q_s8_x2(y[ix].qs + 96 + 128*ib, m3);
#else
                // TODO
                for (int l = 0; l < 4; ++l) {
                    for (int k = 0; k < 8; ++k) for (int i = 0; i < 4; ++i) {
                        y[ib].qs[32*l+4*k+i+  0] = x8[k][ib].qs[i+4*l+ 0];
                        y[ib].qs[32*l+4*k+i+128] = x8[k][ib].qs[i+4*l+16];
                    }
                }
#endif
            }
        }
        return result;
    }

    std::vector<block_q8_KV_r8> r4;
};

struct HelperQ40 final : public BaseHelper {
    using Base = BaseHelper;
    constexpr static ggml_type type = GGML_TYPE_Q4_0;
#if defined __AVX2__
    using block_q8 = block_q8_2;
    constexpr static int block_size_q = QK8_2;
#else
    using block_q8 = block_q8_0;
    constexpr static int block_size_q = QK8_0;
#endif
    HelperQ40(const char * data, int stride) : Base(data, stride) {}

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        int j = F16::block_size*i;
        auto dl = (const block_q4_0 *)Base::lblock(l1) + j/QK4_0;
#ifdef __aarch64__
        auto vd = F16::set1(*(const float16_t *)&dl->d);
        auto q  = vld1q_u8(dl->qs);
        q = j%QK4_0 ? vshrq_n_u8(q, 4) : vandq_u8(q, mask);
        q = vaddq_s8(q, m8);
        v1 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(vget_low_s8(q))));
        v2 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(vget_high_s8(q))));
#else
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d));
        auto q  = _mm_loadu_si128((const __m128i *)dl->qs);
#ifdef __AVX512F__
        auto ql = _mm_add_epi8(_mm_and_si128(q, mask), m8);
        auto qh = _mm_add_epi8(_mm_and_si128(_mm_srli_epi16(q, 4), mask), m8);
        v1 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ql)));
        v2 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qh)));
#else
        if (j%QK4_0) q = _mm_srli_epi16(q, 4);
        auto q16 = _mm256_cvtepi8_epi16(_mm_add_epi8(_mm_and_si128(q, mask), m8));
        v1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16))));
        v2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1))));
#endif
#endif
    }

#ifdef __AVX2__
    const __m128i mask = _mm_set1_epi8(0xf);
    const __m128i m8   = _mm_set1_epi8(-8);
#else
    const uint8x16_t mask = vdupq_n_u8(0xf);
    const  int8x16_t m8   = vdupq_n_s8(-8);
#endif
};

struct HelperQ41 final : public BaseHelper {
    using Base = BaseHelper;
    using block_q8 = block_q8_2;
    constexpr static ggml_type type = GGML_TYPE_Q4_1;
    constexpr static int block_size_q = QK8_2;
    HelperQ41(const char * data, int stride) : Base(data, stride) {}

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        int j = F16::block_size*i;
        auto dl = (const block_q4_1 *)Base::lblock(l1) + j/QK4_1;
#ifdef __aarch64__
        auto vd = F16::set1(*(const float16_t *)&dl->d);
        auto vm = F16::set1(*(const float16_t *)&dl->m);
        auto q  = vld1q_u8(dl->qs);
        q = (j%QK4_1) ? vshrq_n_u8(q, 4) : vandq_u8(q, mask);
        v1 = vfmaq_f16(vm, vd, vcvtq_f16_u16(vmovl_u8(vget_low_u8(q))));
        v2 = vfmaq_f16(vm, vd, vcvtq_f16_u16(vmovl_u8(vget_high_u8(q))));
#else
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d));
        auto vm = F16::set1(GGML_FP16_TO_FP32(dl->m));
        auto q  = _mm_loadu_si128((const __m128i *)dl->qs);
#ifdef __AVX512F__
        auto ql = _mm_and_si128(q, mask);
        auto qh = _mm_and_si128(_mm_srli_epi16(q, 4), mask);
        v1 = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ql)), vm);
        v2 = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qh)), vm);
#else
        if (j%QK4_1) q = _mm_srli_epi16(q, 4);
        auto q16 = _mm256_cvtepi8_epi16(_mm_and_si128(q, mask));
        v1 = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16))), vm);
        v2 = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1))), vm);
#endif
#endif
    }

#ifdef __aarch64__
    const uint8x16_t mask = vdupq_n_u8(0xf);
#else
    const __m128i mask = _mm_set1_epi8(0xf);
#endif
};

struct HelperIQ4nl final : public BaseHelper {
    using Base = BaseHelper;
    constexpr static ggml_type type = GGML_TYPE_IQ4_NL;
#ifdef __aarch64__
    using block_q8 = block_q8_0;
    HelperIQ4nl(const char * data, int stride) : Base(data, stride), values(vld1q_s8(iq4k_values)) {}
    constexpr static int block_size_q = QK8_0;
#else
    HelperIQ4nl(const char * data, int stride) : Base(data, stride) {}
    using block_q8 = block_q8_2;
    constexpr static int block_size_q = QK8_2;
#endif

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        int j = F16::block_size*i;
        auto dl = (const block_iq4_nl *)Base::lblock(l1) + j/QK4_0;
#ifdef __aarch64__
        auto vd = F16::set1(*(const float16_t *)&dl->d);
        auto q  = vld1q_u8(dl->qs);
        q = j%QK4_0 ? vshrq_n_u8(q, 4) : vandq_u8(q, mask);
        q = vqtbl1q_s8(values, q);
        v1 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(vget_low_s8(q))));
        v2 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(vget_high_s8(q))));
#else
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d));
        auto q  = _mm_loadu_si128((const __m128i *)dl->qs);
#ifdef __AVX512F__
        auto ql = _mm_shuffle_epi8(values, _mm_and_si128(q, mask));
        auto qh = _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q, 4), mask));
        v1 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ql)));
        v2 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qh)));
#else
        if (j%QK4_0) q = _mm_srli_epi16(q, 4);
        auto q16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(values, _mm_and_si128(q, mask)));
        v1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16))));
        v2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1))));
#endif
#endif
    }

#ifdef __aarch64__
    const uint8x16_t mask = vdupq_n_u8(0xf);
    const int8x16_t values;
#else
    const __m128i mask = _mm_set1_epi8(0xf);
    const __m128i values = _mm_loadu_si128((const __m128i *)iq4k_values);
#endif
};

struct HelperQ60 final : public BaseHelper {
    constexpr static ggml_type type = GGML_TYPE_Q6_0;
#ifdef __aarch64__
    using block_q8 = block_q8_0;
    constexpr static int block_size_q = QK8_0;
#else
    using block_q8 = block_q8_2;
    constexpr static int block_size_q = QK8_2;
#endif
    using Base = BaseHelper;
    HelperQ60(const char * data, int stride) : Base(data, stride) {}

    // Needed for v * softmax(k * q)
    inline void load(int l1, int i, F16::Data& v1, F16::Data& v2) const {
        int j = F16::block_size*i;
        auto dl = (const block_q6_0 *)Base::lblock(l1) + j/QK6_0;
#ifdef __aarch64__
        // TODO
        const float16_t * d16 = (const float16_t *)&dl->d;
        auto vd = F16::set1(d16[0]);
        //auto vd = F16::set1(*(const float16_t *)&dl->d);
        auto qh8 = vld1_u8(dl->qh);
        auto qh  = vcombine_u8(vshl_n_u8(qh8, 4), qh8);
        auto qs  = vld1q_u8(dl->qs);
        qs = j%QK4_0 ? vshrq_n_u8(qs, 4) : vandq_u8(qs, mask_l);
        qs = vorrq_u8(qs, vandq_u8(mask_h, j%QK4_0 ? vshrq_n_u8(qh, 2) : qh));
        qs = vaddq_s8(qs, m32);
        v1 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(vget_low_s8(qs))));
        v2 = vmulq_f16(vd, vcvtq_f16_s16(vmovl_s8(vget_high_s8(qs))));
#else
        auto vd = F16::set1(GGML_FP16_TO_FP32(dl->d));
        auto bl = _mm_loadu_si128((const __m128i *)dl->qs);
        uint64_t aux64; std::memcpy(&aux64, dl->qh, 8);
        auto bh = _mm_set_epi64x(aux64, aux64 << 4);
#ifdef __AVX512F__
        auto ql = _mm_add_epi8(_mm_or_si128(_mm_and_si128(bl, mask_l), _mm_and_si128(bh, mask_h)), m32);
        auto qh = _mm_add_epi8(_mm_or_si128(_mm_and_si128(_mm_srli_epi16(bl, 4), mask_l), _mm_and_si128(_mm_srli_epi16(bh, 2), mask_h)), m32);
        v1 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(ql)));
        v2 = _mm512_mul_ps(vd, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qh)));
#else
        if (j%QK4_0) {
            bl = _mm_srli_epi16(bl, 4);
            bh = _mm_srli_epi16(bh, 2);
        }
        auto q16 = _mm256_cvtepi8_epi16(_mm_add_epi8(_mm_or_si128(_mm_and_si128(bl, mask_l), _mm_and_si128(bh, mask_h)), m32));
        v1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16))));
        v2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1))));
#endif
#endif
    }

#ifdef __AVX2__
    const __m128i mask_l = _mm_set1_epi8(0x0f);
    const __m128i mask_h = _mm_set1_epi8(0x30);
    const __m128i m32    = _mm_set1_epi8(-32);
#else
    const uint8x16_t mask_l = vdupq_n_u8(0x0f);
    const uint8x16_t mask_h = vdupq_n_u8(0x30);
    const  int8x16_t m32    = vdupq_n_s8(-32);
#endif
};

template <int q_step_in, int k_step_in>
struct FlashMS {
    constexpr static int q_step = q_step_in;
    constexpr static int k_step = k_step_in;
// Something goes wrong when storing and manipulating K*Q as fp16.
// It works for some models (e.g., Gemma-2), but not for others (e.g., LLaMA-3.1-8B).
// As I wasn't able to find where we lose precision, let's comment this out
// for now and do the K*Q part in fp32.
//#ifdef __aarch64__
//    using cache_t = float16_t;
//#else
//    using cache_t = float;
//#endif
    using cache_t = float;

    FlashMS(float scale, float softcap) : vscale(F16::set1(scale)), softcap(softcap), h_inf(GGML_FP32_TO_FP16(-INFINITY)) {}

    inline void init_qstep() {
        for (int j = 0; j < q_step; ++j) {
            S[j] = 0; M[j] = -INFINITY;
        }
    }

    inline void update_M(int j, float smax) {
        if (smax == -INFINITY) {
            std::memset(cache + k_step*j, 0, k_step*sizeof(float));
            need_scaling[j] = M[j] == -INFINITY ? 2 : 0;
            return;
        }
        need_scaling[j] = 0;
        if (smax > M[j]) {
            if (M[j] > -INFINITY) {
                float m = expf(M[j] - smax);
                vms[j] = m;
                need_scaling[j] = 1;
                S[j] *= m;
            } else {
                need_scaling[j] = 2;
                S[j] = 0;
            }
            M[j] = smax;
        }
    }

#ifdef __aarch64__
    inline void update_S(int j, float32x4_t * vk) {
        auto vm = vdupq_n_f32(M[j]);
        auto vsum = vdupq_n_f32(0);
        for (int l = 0; l < k_step/4; ++l) {
            vk[l] = v_expf(vsubq_f32(vk[l], vm));
            vsum = vaddq_f32(vsum, vk[l]);
            F16::store(cache + k_step*j + 4*l, vk[l]);
        }
        S[j] += vaddvq_f32(vsum);
    }
#else
    inline void update_S(int j, F16::Data * vk) {
        auto vm = F16::set1(M[j]);
        for (int l = 0; l < k_step/F16::block_size; ++l) {
            vk[l] = v_expf(F16::sub(vk[l], vm));
            F16::store(cache + k_step*j + F16::block_size*l, vk[l]);
        }
        S[j] += F16::reduce_add<k_step>(vk);
    }
#endif

#ifdef __aarch64__
    inline float load_and_scale(int j, float32x4_t * vk) {
        float32x4_t vmax = vdupq_n_f32(-INFINITY);
        // Something goes wrong when storing and manipulating K*Q as fp16.
        // It works for some models (e.g., Gemma-2), but not for others (e.g., LLaMA-3.1-8B).
        // As I wasn't able to find where we lose precision, let's comment this out
        // for now and do the K*Q part in fp32.
        //if (softcap <= 0.0f) {
        //    for (int l = 0; l < k_step/F16::block_size; ++l) {
        //        auto val = F16::mul(vscale, F16::load(cache + k_step*j + F16::block_size*l));
        //        vk[2*l+0] = vcvt_f32_f16(vget_low_f16(val));
        //        vk[2*l+1] = vcvt_f32_f16(vget_high_f16(val));
        //        vmax = vmaxq_f32(vmax, vmaxq_f32(vk[2*l+0], vk[2*l+1]));
        //    }
        //} else {
        //    auto v_softcap = vdupq_n_f32(softcap);
        //    for (int l = 0; l < k_step/F16::block_size; ++l) {
        //        auto val = F16::mul(vscale, F16::load(cache + k_step*j + F16::block_size*l));
        //        vk[2*l+0] = vcvt_f32_f16(vget_low_f16(val));
        //        vk[2*l+1] = vcvt_f32_f16(vget_high_f16(val));
        //        vk[2*l+0] = vmulq_f32(v_softcap, v_tanh(vk[2*l+0]));
        //        vk[2*l+1] = vmulq_f32(v_softcap, v_tanh(vk[2*l+1]));
        //        vmax = vmaxq_f32(vmax, vmaxq_f32(vk[2*l+0], vk[2*l+1]));
        //    }
        //}
        auto vscale32 = vcvt_f32_f16(vget_low_f16(vscale));
        if (softcap <= 0.0f) {
            for (int l = 0; l < k_step/4; ++l) {
                vk[l] = vmulq_f32(vscale32, vld1q_f32(cache + k_step*j + 4*l));
                vmax = vmaxq_f32(vmax, vk[l]);
            }
        } else {
            auto v_softcap = vdupq_n_f32(softcap);
            for (int l = 0; l < k_step/4; ++l) {
                vk[l] = vmulq_f32(vscale32, vld1q_f32(cache + k_step*j + 4*l));
                vk[l] = vmulq_f32(v_softcap, v_tanh(vk[l]));
                vmax = vmaxq_f32(vmax, vk[l]);
            }
        }
        return vmaxvq_f32(vmax);
    }
    inline float load_apply_mask_and_scale(int j, float32x4_t * vk, const char * mask) {
        auto vzero = vdupq_n_f16(0);
        auto vinf  = vdupq_n_f32(-INFINITY);
        for (int l = 0; l < k_step/8; ++l) {
            auto vm = vceqq_f16(vzero, vld1q_f16((const float16_t *)mask + 8*l));
            auto vm1 = vzip1q_u16(vm, vm);
            auto vm2 = vzip2q_u16(vm, vm);
            auto kq  = vld1q_f32_x2(cache + k_step*j + 8*l);
            vk[2*l+0] = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(kq.val[0]), vm1),
                                                        vbicq_u32(vreinterpretq_u32_f32(vinf), vm1)));
            vk[2*l+1] = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(kq.val[1]), vm2),
                                                        vbicq_u32(vreinterpretq_u32_f32(vinf), vm2)));
        }
        float32x4_t vmax = vdupq_n_f32(-INFINITY);
        auto vscale32 = vcvt_f32_f16(vget_low_f16(vscale));
        if (softcap <= 0.0f) {
            for (int l = 0; l < k_step/4; ++l) {
                vk[l] = vmulq_f32(vscale32, vk[l]);
                vmax = vmaxq_f32(vmax, vk[l]);
            }
        } else {
            auto v_softcap = vdupq_n_f32(softcap);
            for (int l = 0; l < k_step/4; ++l) {
                vk[l] = vmulq_f32(vscale32, vk[l]);
                vk[l] = vmulq_f32(v_softcap, v_tanh(vk[l]));
                vmax = vmaxq_f32(vmax, vk[l]);
            }
        }
        return vmaxvq_f32(vmax);
    }
#else
    inline float load_and_scale(int j, F16::Data * vk) {
        if (softcap <= 0.0f) {
            for (int l = 0; l < k_step/F16::block_size; ++l) vk[l] = F16::mul(vscale, F16::load(cache + k_step*j + F16::block_size*l));
        } else {
            auto v_softcap = F16::set1(softcap);
            for (int l = 0; l < k_step/F16::block_size; ++l) {
                auto val = F16::load(cache + k_step*j + F16::block_size*l);
                vk[l] = F16::mul(v_softcap, v_tanh(F16::mul(vscale, val)));
            }
        }
        return F16::reduce_max<k_step>(vk);
    }
    static inline __m256 apply_mask(int l, const char * mask, __m256 val, [[maybe_unused]] __m256 vinf) {
        return _mm256_add_ps(val, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)mask+l)));
        //auto m128 = _mm_loadu_si128((const __m128i *)mask+l);
        //m128 = _mm_cmpeq_epi16(m128, _mm_setzero_si128());
        //auto m256 = _mm256_cvtepi16_epi32(m128);
        //auto mf = _mm256_castsi256_ps(_mm256_or_si256(m256, _mm256_slli_epi32(m256, 16)));
        //return _mm256_or_ps(_mm256_and_ps(mf, val), _mm256_andnot_ps(mf, vinf));
    }
#ifdef __AVX512F__
    static inline __m512 apply_mask(int l, const char * mask, __m512 val, __m512 vinf) {
        auto m256 = _mm256_loadu_si256((const __m256i *)mask+l);
        m256 = _mm256_cmpeq_epi16(m256, _mm256_setzero_si256());
        auto m512 = _mm512_cvtepi16_epi32(m256);
        auto mf = _mm512_castsi512_ps(_mm512_or_si512(m512, _mm512_slli_epi32(m512, 16)));
        return _mm512_or_ps(_mm512_and_ps(mf, val), _mm512_andnot_ps(mf, vinf));
    }
#endif
    inline float load_apply_mask_and_scale(int j, F16::Data * vk, const char * mask) {
#ifdef HAVE_FANCY_SIMD
        auto vzero = _mm256_set1_epi16(0);
        auto vinf  = _mm512_set1_ps(-INFINITY);
        if (softcap <= 0) {
            for (int l = 0; l < k_step/F16::block_size; ++l) {
                auto m16 = _mm256_cmpeq_epi16_mask(_mm256_loadu_si256((const __m256i *)mask + l), vzero);
                vk[l] = _mm512_mask_mul_ps(vinf, m16, vscale, F16::load(cache + k_step*j + F16::block_size*l));
            }
        } else {
            auto v_softcap = F16::set1(softcap);
            for (int l = 0; l < k_step/F16::block_size; ++l) {
                auto m16 = _mm256_cmpeq_epi16_mask(_mm256_loadu_si256((const __m256i *)mask + l), vzero);
                vk[l] = _mm512_mask_mul_ps(vinf, m16, v_softcap, v_tanh(F16::mul(vscale, F16::load(cache + k_step*j + F16::block_size*l))));
            }
        }
#else
        auto vinf  = F16::set1(-INFINITY);
        for (int l = 0; l < k_step/F16::block_size; ++l) {
            vk[l] = apply_mask(l, mask, F16::load(cache + k_step*j + F16::block_size*l), vinf);
        }
        if (softcap <= 0) {
            for (int l = 0; l < k_step/F16::block_size; ++l) vk[l] = F16::mul(vscale, vk[l]);
        } else {
            auto v_softcap = F16::set1(softcap);
            for (int l = 0; l < k_step/F16::block_size; ++l) vk[l] = F16::mul(v_softcap, v_tanh(F16::mul(vscale, vk[l])));
        }
#endif
        return F16::reduce_max<k_step>(vk);
    }
#endif

#ifdef __aarch64__
    inline void update_M_S(int j, float32x4_t * vk) {
        float smax = load_and_scale(j, vk);
        update_M(j, smax);
        if (M[j] > -INFINITY) update_S(j, vk);
    }
    inline void update_M_S(int j, float32x4_t * vk, const char * mask) {
        float smax = load_apply_mask_and_scale(j, vk, mask);
        update_M(j, smax);
        if (M[j] > -INFINITY) update_S(j, vk);
    }
#else
    inline void update_M_S(int j, F16::Data * vk) {
        float smax = load_and_scale(j, vk);
        update_M(j, smax);
        if (M[j] > -INFINITY) update_S(j, vk);
    }
    inline void update_M_S(int j, F16::Data * vk, const char * mask) {
        float smax = load_apply_mask_and_scale(j, vk, mask);
        update_M(j, smax);
        if (M[j] > -INFINITY) update_S(j, vk);
    }
#endif

    cache_t cache[q_step*k_step];
    float S[q_step], M[q_step];
    int need_scaling[q_step];
    float vms[q_step];
    const F16::Data vscale;
    const float  softcap;
    const ggml_half h_inf;

};

template <int D, int q_step, int k_step>
struct FlashQKV {

#ifdef __aarch64__
    using qkv_cache_t = float16_t;
#else
    using qkv_cache_t = float;
#endif

    template <typename VHelper, typename FMS>
    inline void accumulate_qkv_1(const VHelper& vh, const FMS& fms) {
        static_assert(q_step == FMS::q_step);
        F16::Data vq[D/F16::block_size];
        if (fms.need_scaling[0] == 2) {
            for (int i = 0; i < D/F16::block_size; ++i) vq[i] = F16::zero();
        } else {
            for (int i = 0; i < D/F16::block_size; ++i) vq[i] = F16::load(qkv_cache + F16::block_size*i);
            if (fms.need_scaling[0] == 1) {
                auto vms = F16::set1(fms.vms[0]);
                for (int i = 0; i < D/F16::block_size; ++i) vq[i] = F16::mul(vms, vq[i]);
            }
        }
        F16::Data v0, v1;
        for (int l = 0; l < k_step; l += 4) {
            auto vs0 = F16::set1(fms.cache[l + 0]);
            auto vs1 = F16::set1(fms.cache[l + 1]);
            auto vs2 = F16::set1(fms.cache[l + 2]);
            auto vs3 = F16::set1(fms.cache[l + 3]);
            for (int i = 0; i < D/F16::block_size; i += 2) {
                vh.load(l+0, i, v0, v1);
                vq[i+0] = F16::fmadd(vq[i+0], v0, vs0);
                vq[i+1] = F16::fmadd(vq[i+1], v1, vs0);
                vh.load(l+1, i, v0, v1);
                vq[i+0] = F16::fmadd(vq[i+0], v0, vs1);
                vq[i+1] = F16::fmadd(vq[i+1], v1, vs1);
                vh.load(l+2, i, v0, v1);
                vq[i+0] = F16::fmadd(vq[i+0], v0, vs2);
                vq[i+1] = F16::fmadd(vq[i+1], v1, vs2);
                vh.load(l+3, i, v0, v1);
                vq[i+0] = F16::fmadd(vq[i+0], v0, vs3);
                vq[i+1] = F16::fmadd(vq[i+1], v1, vs3);
            }
        }
        for (int i = 0; i < D/F16::block_size; ++i) F16::store(qkv_cache + F16::block_size*i, vq[i]);
    }

    // This fails for head sizes of 80 and 112 as D/16 is odd, so we cannot do steps of 2
    // Hence, for now, we will not handle head sizes of 80 and 112
    template <typename VHelper, typename FMS>
    inline void accumulate_qkv(const VHelper& vh, const FMS& fms) {
        static_assert(q_step == FMS::q_step);
        if constexpr (q_step == 1) {
            accumulate_qkv_1(vh, fms);
            return;
        }
        for (int j = 0; j < q_step; ++j) {
            auto R = qkv_cache + D*j;
            if (fms.need_scaling[j] == 2) {
                std::memset(R, 0, D*sizeof(qkv_cache_t));
            }
            else if (fms.need_scaling[j] == 1) {
                auto vms = F16::set1(fms.vms[j]);
                for (int i = 0; i < D/F16::block_size; ++i) {
                    F16::store(R + F16::block_size*i, F16::mul(vms, F16::load(R + F16::block_size*i)));
                }
            }
        }
#ifdef __AVX512F__
        if constexpr ((D/F16::block_size)%4 == 0) {
            F16::Data v[16];
            F16::Data vs[4];
            for (int i = 0; i < D/F16::block_size; i += 4) {
                for (int l = 0; l < k_step; l += 4) {
                    for (int k = 0; k < 4; ++k) {
                        vh.load(l+k, i+0, v[4*k+0], v[4*k+1]);
                        vh.load(l+k, i+2, v[4*k+2], v[4*k+3]);
                    }
                    for (int j = 0; j < q_step; ++j) {
                        auto R = qkv_cache + D*j;
                        auto s1 = F16::load(R + F16::block_size*(i+0));
                        auto s2 = F16::load(R + F16::block_size*(i+1));
                        auto s3 = F16::load(R + F16::block_size*(i+2));
                        auto s4 = F16::load(R + F16::block_size*(i+3));
                        F16::set4(fms.cache + k_step*j + l, vs);
                        for (int k = 0; k < 4; ++k) {
                            s1 = F16::fmadd(s1, v[4*k+0], vs[k]);
                            s2 = F16::fmadd(s2, v[4*k+1], vs[k]);
                            s3 = F16::fmadd(s3, v[4*k+2], vs[k]);
                            s4 = F16::fmadd(s4, v[4*k+3], vs[k]);
                        }
                        F16::store(R + F16::block_size*(i+0), s1);
                        F16::store(R + F16::block_size*(i+1), s2);
                        F16::store(R + F16::block_size*(i+2), s3);
                        F16::store(R + F16::block_size*(i+3), s4);
                    }
                }
            }
            return;
        }
#endif
        F16::Data v[8];
#ifdef __AVX2__
        F16::Data vs[4];
#endif
        for (int i = 0; i < D/F16::block_size; i += 2) {
            for (int l = 0; l < k_step; l += 4) {
                vh.load(l+0, i, v[0], v[4]);
                vh.load(l+1, i, v[1], v[5]);
                vh.load(l+2, i, v[2], v[6]);
                vh.load(l+3, i, v[3], v[7]);
                for (int j = 0; j < q_step; ++j) {
                    auto R = qkv_cache + D*j;
                    auto s1 = F16::load(R + F16::block_size*(i+0));
                    auto s2 = F16::load(R + F16::block_size*(i+1));
#ifdef __AVX2__
                    F16::set4(fms.cache + k_step*j + l, vs);
                    for (int k = 0; k < 4; ++k) {
                        s1 = F16::fmadd(s1, v[k+0], vs[k]);
                        s2 = F16::fmadd(s2, v[k+4], vs[k]);
                    }
#else
                    auto vs = F16::set4(fms.cache + k_step*j + l);
                    s1 = F16::fmadd_lane0(s1, v[0], vs);
                    s2 = F16::fmadd_lane0(s2, v[4], vs);
                    s1 = F16::fmadd_lane1(s1, v[1], vs);
                    s2 = F16::fmadd_lane1(s2, v[5], vs);
                    s1 = F16::fmadd_lane2(s1, v[2], vs);
                    s2 = F16::fmadd_lane2(s2, v[6], vs);
                    s1 = F16::fmadd_lane3(s1, v[3], vs);
                    s2 = F16::fmadd_lane3(s2, v[7], vs);
#endif
                    F16::store(R + F16::block_size*(i+0), s1);
                    F16::store(R + F16::block_size*(i+1), s2);
                }
            }
        }
    }

    template <typename VHelper, typename FMS>
    inline void accumulate_qkv(int nq1, const VHelper& vh, const FMS& fms) {
        static_assert(q_step == FMS::q_step);
        if (nq1 == 1) {
            accumulate_qkv_1(vh, fms);
            return;
        }
        F16::Data v[8];
        for (int j = 0; j < nq1; ++j) {
            auto R = qkv_cache + D*j;
            if (fms.need_scaling[j] == 2) {
                std::memset(R, 0, D*sizeof(qkv_cache_t));
            }
            else if (fms.need_scaling[j] == 1) {
                auto vms = F16::set1(fms.vms[j]);
                for (int i = 0; i < D/F16::block_size; ++i) {
                    F16::store(R + F16::block_size*i, F16::mul(vms, F16::load(R + F16::block_size*i)));
                }
            }
        }
        for (int i = 0; i < D/F16::block_size; i += 2) {
            for (int l = 0; l < k_step; l += 4) {
                vh.load(l+0, i, v[0], v[4]);
                vh.load(l+1, i, v[1], v[5]);
                vh.load(l+2, i, v[2], v[6]);
                vh.load(l+3, i, v[3], v[7]);
                for (int j = 0; j < nq1; ++j) {
                    auto R = qkv_cache + D*j;
                    auto s1 = F16::load(R + F16::block_size*(i+0));
                    auto s2 = F16::load(R + F16::block_size*(i+1));
                    auto vs = F16::set4(fms.cache + k_step*j + l);
                    s1 = F16::fmadd_lane0(s1, v[0], vs);
                    s2 = F16::fmadd_lane0(s2, v[4], vs);
                    s1 = F16::fmadd_lane1(s1, v[1], vs);
                    s2 = F16::fmadd_lane1(s2, v[5], vs);
                    s1 = F16::fmadd_lane2(s1, v[2], vs);
                    s2 = F16::fmadd_lane2(s2, v[6], vs);
                    s1 = F16::fmadd_lane3(s1, v[3], vs);
                    s2 = F16::fmadd_lane3(s2, v[7], vs);
                    F16::store(R + F16::block_size*(i+0), s1);
                    F16::store(R + F16::block_size*(i+1), s2);
                }
            }
        }
    }

    template <typename FMS>
    inline void normalize_and_store_1row(const FMS& fms, int j, qkv_cache_t * R, float * qkv, const float * sinkf) const {
        static_assert(q_step == FMS::q_step);
        float S = fms.S[j];
        if (sinkf) {
            float s = *sinkf;
            if (s > fms.M[j]) {
                float m = expf(fms.M[j] - s);
                auto vm = F16::set1(m);
                for (int i = 0; i < D/F16::block_size; ++i) {
                    auto Ri = R + F16::block_size*i;
                    F16::store(Ri, F16::mul(vm, F16::load(Ri)));
                }
                S = S*m + 1;
            } else {
                S += expf(s - fms.M[j]);
            }
        }
        GGML_ASSERT(S > 0);
        auto norm = F16::set1(1/S);
        //auto norm = F16::set1(fms.S[j] > 0 ? 1/fms.S[j] : 0.f);
        for (int i = 0; i < D/F16::block_size; ++i) {
            auto r = F16::load(R + F16::block_size*i);
            F16::store(qkv + F16::block_size*i, F16::mul(norm, r));
        }
    }

    template <typename FMS>
    inline void normalize_and_store(const FMS& fms, int nq1, int stride_qkv, float * qkv, const float * sinkf, float * M, float * S) {
        static_assert(q_step == FMS::q_step);
        if (M && S) {
            std::memcpy(M, fms.M, nq1*sizeof(float));
            std::memcpy(S, fms.S, nq1*sizeof(float));
            auto R = qkv_cache;
            for (int j = 0; j < nq1; ++j) {
#ifdef __aarch64__
                for (int i = 0; i < D/F16::block_size; ++i) {
                    F16::store(qkv + F16::block_size*i, F16::load(R + F16::block_size*i));
                }
#else
                std::memcpy(qkv, R, D*sizeof(float));
#endif
                qkv += stride_qkv;
                R   += D;
            }
        } else {
            auto R = qkv_cache;
            for (int j = 0; j < nq1; ++j) {
                normalize_and_store_1row(fms, j, R, qkv, sinkf);
                qkv += stride_qkv;
                R   += D;
            }
        }
    }

    template <typename FMS>
    inline void normalize_and_store(const FMS& fms, int stride_qkv, float * qkv, const float * sinkf, float * M, float * S) {
        static_assert(q_step == FMS::q_step);
        if (M && S) {
            std::memcpy(M, fms.M, q_step*sizeof(float));
            std::memcpy(S, fms.S, q_step*sizeof(float));
            auto R = qkv_cache;
            for (int j = 0; j < q_step; ++j) {
#ifdef __aarch64__
                for (int i = 0; i < D/F16::block_size; ++i) {
                    F16::store(qkv + F16::block_size*i, F16::load(R + F16::block_size*i));
                }
#else
                std::memcpy(qkv, R, D*sizeof(float));
#endif
                qkv += stride_qkv;
                R   += D;
            }
        } else {
            auto R = qkv_cache;
            for (int j = 0; j < q_step; ++j) {
                normalize_and_store_1row(fms, j, R, qkv, sinkf);
                qkv += stride_qkv;
                R   += D;
            }
        }
    }

    // qkv_cache_t qkv_cache[D*q_step];
    // The initializer is not actually required. But the compiler cannot figure out that when qkv_cache is
    // first used for q_step rows, fms.need_scaling[j] is always 2, which zeroes the content of qkv_cache.
    // As a result, we get an infinite stream of warnings about uninitialized variable use (one for each
    // combination of D, q_step, k_step), which is extremely annoying. Hence, I succumb to the trend of
    // constantly being saved by others (the compiler in this case), and add this 100% unnecessary initialization.
    qkv_cache_t qkv_cache[D*q_step]; // = {};
    //qkv_cache_t * qkv_cache;
};

template <int D, int q_step, int k_step>
struct FlashQKfp32 {
    static_assert(D%F16::block_size == 0 && D <= 576);
    static_assert(k_step%F16::block_size == 0);
    static_assert(q_step <= 4 || q_step%4 == 0);

    template <typename KHelper, typename q_float>
    static inline void multiply_mask_kq(const KHelper& kh, int stride_q, int stride_m, const q_float * q, const char * mask,
            FlashMS<q_step, k_step>& fms) {
#ifdef __AVX2__
        constexpr int nrc_k = 8;
        static_assert(k_step%nrc_k == 0);
#endif
        DataInfo info{fms.cache, (const char *)q, k_step, stride_q*sizeof(q_float), 0, 1, nullptr};
        iqk_gemm_default_floats(D, q_step, kh.block, kh.stride, info, k_step);
#ifdef __AVX2__
        F16::Data vk[k_step/F16::block_size];
#else
        float32x4_t vk[k_step/4];
#endif
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
    }

    template <typename KHelper, typename q_float>
    static inline void multiply_mask_kq(int nq, const KHelper& kh, int stride_q, int stride_m, const q_float * q, const char * mask,
            FlashMS<q_step, k_step>& fms) {
#ifdef __AVX2__
        constexpr int nrc_k = 8;
        static_assert(k_step%nrc_k == 0);
#endif
        DataInfo info{fms.cache, (const char *)q, k_step, stride_q*sizeof(q_float), 0, 1, nullptr};
        iqk_gemm_default_floats(D, nq, kh.block, kh.stride, info, k_step);
#ifdef __AVX2__
        F16::Data vk[k_step/F16::block_size];
#else
        float32x4_t vk[k_step/4];
#endif
        for (int j = 0; j < nq; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
    }

#ifdef __aarch64__
    static inline void convert(int nq, int stride_q, const float * q, float16_t * q_f16) {
        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < D; j += 8) {
                auto val1_f32 = vld1q_f32(q + j + 0);
                auto val2_f32 = vld1q_f32(q + j + 4);
                auto val_f16  = vcombine_f16(vcvt_f16_f32(val1_f32), vcvt_f16_f32(val2_f32));
                vst1q_f16(q_f16 + j, val_f16);
            }
            q += stride_q;
            q_f16 += D;
        }
    }
#endif

    template <typename KHelper, typename block_q8>
    static inline void mul_mask_kq(const KHelper& kh, int stride_m,
            const block_q8 * q, const char * mask, FlashMS<q_step, k_step>& fms) {
        // As far as I can tell, this static assert is a remnant of the times where the matrix multiplications were done inline
        // here with bespoke kernels instead of just using the regular mat mul kernels. But, just in case, leaving it in place
        // but commneted out.
        //constexpr int kMaxQ = 8;
        //static_assert(q_step < kMaxQ || q_step%kMaxQ == 0);
        DataInfo info{fms.cache, (const char *)q, k_step, (D/KHelper::block_size_q)*sizeof(block_q8), 0, 1, nullptr};
        if constexpr (std::is_same_v<KHelper, HelperQ8KVR8<D>> ||
                      std::is_same_v<KHelper, HelperQ8KV<D>>) {
            iqk_gemm_q8kv_fa(D, q_step, kh.type, kh.block, kh.stride, info, k_step);
        } else {
            iqk_gemm_legacy_fa(D, q_step, kh.type, kh.block, kh.stride, info, k_step);
        }
#ifdef __aarch64__
        float32x4_t vk[k_step/4];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
#else
        F16::Data vk[k_step/F16::block_size];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
#endif
    }

    template <typename KHelper, typename block_q8>
    static inline void mul_mask_kq(int nq, const KHelper& kh, int stride_m,
            const block_q8 * q, const char * mask, FlashMS<q_step, k_step>& fms) {
        GGML_ASSERT(nq < q_step);
        DataInfo info{fms.cache, (const char *)q, k_step, (D/KHelper::block_size_q)*sizeof(block_q8), 0, 1, nullptr};
        if constexpr (std::is_same_v<KHelper, HelperQ8KVR8<D>> ||
                      std::is_same_v<KHelper, HelperQ8KV<D>>) {
            iqk_gemm_q8kv_fa(D, nq, kh.type, kh.block, kh.stride, info, k_step);
        } else {
            iqk_gemm_legacy_fa(D, nq, kh.type, kh.block, kh.stride, info, k_step);
        }
#ifdef __aarch64__
        float32x4_t vk[k_step/4];
        for (int j = 0; j < nq; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
#else
        F16::Data vk[k_step/F16::block_size];
        for (int j = 0; j < nq; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
#endif
    }
};

template <int Dk, int Dv, int q_step, int k_step, typename KHelper, typename VHelper, typename KQHelper>
void compute_helper(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
        FlashMS<q_step, k_step>& fms,
        FlashQKV<Dv, q_step, k_step>& fqkv,
        const float * q, const char * mask, float * qkv,
        const float * sinkf, float * M, float * S) {
#ifdef __aarch64__
    float16_t q_f16[Dk*q_step];
#endif

    for (int i1 = 0; i1 < nq1/q_step; ++i1) {
        fms.init_qstep();
        kh.reset_block();
        vh.reset_block();
#ifdef __aarch64__
        KQHelper::convert(q_step, stride_q, q, q_f16);
#endif
        auto mr = mask;
        auto Mc = (const uint16_t *)(mr + (q_step - 1)*stride_m);
        int ik = nk1 - k_step;
        for (; ik >=0 && Mc[ik] != 0; ik -= k_step);
        ik += k_step;
        for (int k1 = 0; k1 < ik/k_step; ++k1) {
#ifdef __aarch64__
            KQHelper::multiply_mask_kq(kh, Dk, stride_m, q_f16, mr, fms);
#else
            KQHelper::multiply_mask_kq(kh, stride_q, stride_m, q, mr, fms);
#endif
            fqkv.accumulate_qkv(vh, fms);
            kh.next_block(k_step);
            vh.next_block(k_step);
            mr += k_step*sizeof(ggml_half);
        }
        fqkv.normalize_and_store(fms, stride_qkv, qkv, sinkf, M, S);

        q    += q_step*stride_q;
        mask += q_step*stride_m;
        qkv  += q_step*stride_qkv;
        if (M && S) { M += q_step; S += q_step; }
    }
    int n_left = nq1 - q_step*(nq1/q_step);
    if (n_left > 0) {
        fms.init_qstep();
        kh.reset_block();
        vh.reset_block();
#ifdef __aarch64__
        KQHelper::convert(n_left, stride_q, q, q_f16);
#endif
        auto mr = mask;
        for (int k1 = 0; k1 < nk1/k_step; ++k1) {
#ifdef __aarch64__
            KQHelper::multiply_mask_kq(n_left, kh, Dk, stride_m, q_f16, mr, fms);
#else
            KQHelper::multiply_mask_kq(n_left, kh, stride_q, stride_m, q, mr, fms);
#endif
            fqkv.accumulate_qkv(n_left, vh, fms);
            kh.next_block(k_step);
            vh.next_block(k_step);
            mr += k_step*sizeof(ggml_half);
        }
        fqkv.normalize_and_store(fms, n_left, stride_qkv, qkv, sinkf, M, S);
    }
}

template <int Dk, int Dv, int q_step, int k_step, typename KHelper, typename VHelper, typename KQHelper>
void compute_helper_q(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
        FlashMS<q_step, k_step>& fms,
        FlashQKV<Dv, q_step, k_step>& fqkv,
        const float * q, const char * mask, float * qkv,
        const float * sinkf, float * M, float * S, char * qptr) {
    auto q8 = (typename KHelper::block_q8 *)qptr;
    // This optimization fails under certain conditions (see https://github.com/ikawrakow/ik_llama.cpp/issues/1205)
    // => disabling until I figure out what goes wrong
    if constexpr (q_step >= 4 && std::is_same_v<KHelper, HelperQ80>) {
        if (nq1 == q_step) {
            fms.init_qstep();
            kh.reset_block();
            vh.reset_block();
            block_q8_0_r8 q8r8[Dk/QK8_0 * k_step/8];
            HelperQ80R8<Dk> khr8((const char *)q8r8, Dk/QK8_0*sizeof(block_q8_0));
            auto q8r = (typename HelperQ80R8<Dk>::block_q8 *)qptr;
            HelperQ80::convert<Dk>(q_step, stride_q, q, q8r);
            auto mr = mask;
            auto Mc = (const uint16_t *)(mr + (q_step - 1)*stride_m);
            int ik = nk1 - k_step;
            for (; ik >=0 && Mc[ik] != 0; ik -= k_step);
            ik += k_step;
            for (int k1 = 0; k1 < ik/k_step; ++k1) {
                HelperQ80R8<Dk>::repack(k_step, kh.block, kh.stride, q8r8);
                KQHelper::mul_mask_kq(khr8, stride_m, q8r, mr, fms);
                fqkv.accumulate_qkv(vh, fms);
                kh.next_block(k_step);
                vh.next_block(k_step);
                mr += k_step*sizeof(ggml_half);
            }
            fqkv.normalize_and_store(fms, stride_qkv, qkv, sinkf, M, S);
            return;
        }
    }
#if FA_TIMING
    Perf perf(false);
#endif
    for (int i1 = 0; i1 < nq1/q_step; ++i1) {
#if FA_TIMING
        auto t1 = Perf::cur_time();
#endif
        fms.init_qstep();
        kh.reset_block();
        vh.reset_block();
        HelperQ80::convert<Dk>(q_step, stride_q, q, q8);
#if FA_TIMING
        perf.accum_nolock(0, t1);
#endif
        auto mr = mask;
        auto Mc = (const uint16_t *)(mr + (q_step - 1)*stride_m);
        int ik = nk1 - k_step;
        for (; ik >=0 && Mc[ik] != 0; ik -= k_step);
        ik += k_step;
        for (int k1 = 0; k1 < ik/k_step; ++k1) {
#if FA_TIMING
            t1 = Perf::cur_time();
            KQHelper::mul_mask_kq(kh, stride_m, q8, mr, fms);
            perf.accum_nolock(1, t1);
            t1 = Perf::cur_time();
            fqkv.accumulate_qkv(vh, fms);
            perf.accum_nolock(2, t1);
#else
            KQHelper::mul_mask_kq(kh, stride_m, q8, mr, fms);
            fqkv.accumulate_qkv(vh, fms);
#endif
            kh.next_block(k_step);
            vh.next_block(k_step);
            mr += k_step*sizeof(ggml_half);
        }
#if FA_TIMING
        t1 = Perf::cur_time();
        fqkv.normalize_and_store(fms, stride_qkv, qkv, sinkf, M, S);
        perf.accum_nolock(3, t1);
#else
        fqkv.normalize_and_store(fms, stride_qkv, qkv, sinkf, M, S);
#endif

        q    += q_step*stride_q;
        mask += q_step*stride_m;
        qkv  += q_step*stride_qkv;
        if (M && S) { M += q_step; S += q_step; }
    }
    int n_left = nq1 - q_step*(nq1/q_step);
    if (n_left > 0) {
        fms.init_qstep();
        kh.reset_block();
        vh.reset_block();
        HelperQ80::convert<Dk>(n_left, stride_q, q, q8);
        auto mr = mask;
        for (int k1 = 0; k1 < nk1/k_step; ++k1) {
            KQHelper::mul_mask_kq(n_left, kh, stride_m, q8, mr, fms);
            fqkv.accumulate_qkv(n_left, vh, fms);
            kh.next_block(k_step);
            vh.next_block(k_step);
            mr += k_step*sizeof(ggml_half);
        }
        fqkv.normalize_and_store(fms, n_left, stride_qkv, qkv, sinkf, M, S);
    }
#if FA_TIMING
    Perf::instance().add(perf);
#endif
}

char * get_q_storage(size_t size) {
    thread_local std::vector<char> q_storage;
    if (q_storage.size() < size) q_storage.resize(size);
    return q_storage.data();
}

// Some of the methods in FlashAttn have two identical implementations that only differ by
// one version using a loop over the template parameter q_step, while the other using a loop
// over an input parameter nq (these are loops over the rows of q^T). I dislike this a lot,
// but performance drops signficantly if I remove the version with fixed q_step iterations.
// We only instantiate FlashAttn with q_step = 1 and q_step = 4 or 8 (depending on head size D),
// so when we have to process Nq rows, we process q_step*(Nq/q_step) using fixed q_step loops,
// and use the variable nq version (with lower performance) only for the remaining i1...q_step-1
// rows (if Nq is not a multiple of q_step). One could have made the number of q^T rows to
// process template parameter of such functions, but this would result in the compiler generating
// q_step-1 versions of these functions for us, which I though was too much with q_step = 8.
template <int Dk, int Dv, int q_step, int k_step>
struct FlashAttn {
    static_assert(Dk%F16::block_size == 0 && Dk <= 576);
    static_assert(Dv%F16::block_size == 0 && Dv <= 512);
    static_assert(k_step%F16::block_size == 0);
    static_assert(q_step <= 4 || q_step%4 == 0);

    FlashAttn(float scale, float softcap, const float * sinkf) : fms(scale, softcap), sinkf(sinkf) {}

    template <typename KHelper, typename VHelper>
    void compute(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
            const float * q, const char * mask, float * qkv, [[maybe_unused]] float * M, [[maybe_unused]] float * S) {
        if constexpr (std::is_same_v<KHelper, HelperQ40> ||
                      std::is_same_v<KHelper, HelperQ41> ||
                      std::is_same_v<KHelper, HelperIQ4nl> ||
                      std::is_same_v<KHelper, HelperQ60> ||
                      std::is_same_v<KHelper, HelperQ80R8<Dk>> ||
                      std::is_same_v<KHelper, HelperQ80> ||
                      std::is_same_v<KHelper, HelperQ8KV<Dk>> ||
                      std::is_same_v<KHelper, HelperQ8KVR8<Dk>>) {
            constexpr size_t kMaxOnStackSize = 576;
            //auto q_size = q_step*(Dk/KHelper::block_size_q)*sizeof(typename KHelper::block_q8);
            auto q_size = q_step*(Dk/QK8_2*sizeof(block_q8_2));
            q_size = GGML_PAD(q_size, 64);
            if (q_size > kMaxOnStackSize) {
                auto qptr = get_q_storage(q_size);
                if (false && nq1 >= 8) {
                    if constexpr (std::is_same_v<KHelper, HelperQ80>) {
#if FA_TIMING
                        auto t1 = Perf::cur_time();
                        HelperQ80R8<Dk, k_step> khr4(nk1, kh);
                        Perf::instance().accum(4, t1);
#else
                        HelperQ80R8<Dk> khr4(nk1, kh);
#endif
                        compute_helper_q<Dk, Dv, q_step, k_step, HelperQ80R8<Dk>, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                                khr4, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, sinkf, M, S, qptr);
                        return;

                    }
#if GGML_IQK_FA_ALL_QUANTS
                    if constexpr (std::is_same_v<KHelper, HelperQ8KV<Dk>>) {
#if FA_TIMING
                        auto t1 = Perf::cur_time();
                        HelperQ8KVR8<Dk, k_step> khr4(nk1, kh);
                        Perf::instance().accum(4, t1);
#else
                        HelperQ8KVR8<Dk> khr4(nk1, kh);
#endif
                        compute_helper_q<Dk, Dv, q_step, k_step, HelperQ8KVR8<Dk>, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                                khr4, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, sinkf, M, S, qptr);
                        return;
                    }
#endif
                }
                compute_helper_q<Dk, Dv, q_step, k_step, KHelper, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                        kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, sinkf, M, S, qptr);

            }
            else {
                typename KHelper::block_q8 q8[q_step*(Dk/KHelper::block_size_q)];
                compute_helper_q<Dk, Dv, q_step, k_step, KHelper, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                        kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, sinkf, M, S, (char *)q8);
            }
        }
        else {
            compute_helper<Dk, Dv, q_step, k_step, KHelper, VHelper, FlashQKfp32<Dk, q_step, k_step>>(
                    kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, fms, fqkv, q, mask, qkv, sinkf, M, S);
        }
    }

    FlashMS<q_step, k_step>      fms;
    FlashQKV<Dv, q_step, k_step> fqkv;
    const float *                sinkf;

};

#ifdef __AVX512BF16__

template <int D, int step>
struct HelperBF16 final : public BaseHelper {
    using Base = BaseHelper;
    HelperBF16(const char * data, int stride) : Base(data, stride) {}
    inline void load(int l1, __m512bh * vk) const {
        auto dr = Base::lblock(l1);
        for (int i = 0; i < D/32; ++i) vk[i] = __m512bh(_mm512_loadu_si512((const __m512i*)dr + i));
    }

    inline void load(int l1, int i, __m512& v1, __m512& v2) const {
        auto dr = Base::lblock(l1);
        v1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)dr + i + 0)), 16));
        v2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)dr + i + 1)), 16));
    }

    inline void load_2(int l1, __m512bh * vk) const {
        load(l1+0, vk+0);
        load(l1+1, vk+D/32);
    }

    inline void load_4(int l1, __m512bh * vk) const {
        load(l1+0, vk+0);
        load(l1+1, vk+1*D/32);
        load(l1+2, vk+2*D/32);
        load(l1+3, vk+3*D/32);
    }

    inline void load_8(int l1, __m512bh * vk) const {
        for (int k = 0; k < 8; ++k) load(l1 + k, vk + k*D/32);
    }
};

template <int D, int q_step, int k_step>
struct FlashQKbf16 {
    //static_assert(D%32 == 0 && D <= 256);
    static_assert(D%32 == 0 && D <= 576);
    static_assert(k_step%32 == 0);
    static_assert(q_step <= 4 || q_step%4 == 0);

    static inline void mult_mask_kq_one(int l1, int m1, int stride_q, int stride_m, const float * q, const char * mask,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        // q index is q_step*i1 + m1
        // k index is k_step*k1 + l1
        const ggml_half * mp = (const ggml_half *)(mask + stride_m*m1);
        fms.cache[k_step*m1 + l1 + 0] = fms.cache[k_step*m1 + l1 + 1] = -INFINITY;
        if (mp[l1+0] == fms.h_inf && mp[l1+1] == fms.h_inf) {
            return;
        }
        auto qr = q + m1*stride_q;
        for (int i = 0; i < D/32; ++i) {
            auto val1 = _mm512_loadu_ps(qr + 32*i);
            auto val2 = _mm512_loadu_ps(qr + 32*i + 16);
            qv[i] = _mm512_cvtne2ps_pbh(val2, val1);
        }
        if (mp[l1+0] != fms.h_inf) {
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i], qv[i]);
            fms.cache[k_step*m1 + l1 + 0] = _mm512_reduce_add_ps(vsum);
        }
        if (mp[l1+1] != fms.h_inf) {
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+D/32], qv[i]);
            fms.cache[k_step*m1 + l1 + 1] = _mm512_reduce_add_ps(vsum);
        }
    }

    static inline void mult_mask_kq_one(int l1, int m1, int stride_m, const ggml_bf16_t * q, const char * mask,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        // q index is q_step*i1 + m1
        // k index is k_step*k1 + l1
        const ggml_half * mp = (const ggml_half *)(mask + stride_m*m1);
        fms.cache[k_step*m1 + l1 + 0] = fms.cache[k_step*m1 + l1 + 1] = -INFINITY;
        if (mp[l1+0] == fms.h_inf && mp[l1+1] == fms.h_inf) {
            return;
        }
        auto qr = q + m1*D;
        for (int i = 0; i < D/32; ++i) qv[i] = __m512bh(_mm512_loadu_si512((const __m512i*)qr + i));
        if (mp[l1+0] != fms.h_inf) {
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i], qv[i]);
            fms.cache[k_step*m1 + l1 + 0] = _mm512_reduce_add_ps(vsum);
        }
        if (mp[l1+1] != fms.h_inf) {
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+D/32], qv[i]);
            fms.cache[k_step*m1 + l1 + 1] = _mm512_reduce_add_ps(vsum);
        }
    }

    static inline void mult_mask_kq_4(int l1, int m1, int stride_q, int stride_m, const float * q, const char * mask,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        // q index is q_step*i1 + m1
        // k index is k_step*k1 + l1
        const ggml_half * mp = (const ggml_half *)(mask + stride_m*m1);
        fms.cache[k_step*m1 + l1 + 0] = fms.cache[k_step*m1 + l1 + 1] =
        fms.cache[k_step*m1 + l1 + 2] = fms.cache[k_step*m1 + l1 + 3] = -INFINITY;
        if (mp[l1+0] == fms.h_inf && mp[l1+1] == fms.h_inf && mp[l1+2] == fms.h_inf && mp[l1+3] == fms.h_inf) {
            return;
        }
        auto qr = q + m1*stride_q;
        for (int i = 0; i < D/32; ++i) {
            auto val1 = _mm512_loadu_ps(qr + 32*i);
            auto val2 = _mm512_loadu_ps(qr + 32*i + 16);
            qv[i] = _mm512_cvtne2ps_pbh(val2, val1);
        }
        for (int k = 0; k < 4; ++k) {
            if (mp[l1+k] == fms.h_inf) continue;
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+k*(D/32)], qv[i]);
            fms.cache[k_step*m1 + l1 + k] = _mm512_reduce_add_ps(vsum);
        }
    }

    static inline void mult_mask_kq_4(int l1, int m1, int stride_m, const ggml_bf16_t * q, const char * mask,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        // q index is q_step*i1 + m1
        // k index is k_step*k1 + l1
        const ggml_half * mp = (const ggml_half *)(mask + stride_m*m1);
        fms.cache[k_step*m1 + l1 + 0] = fms.cache[k_step*m1 + l1 + 1] =
        fms.cache[k_step*m1 + l1 + 2] = fms.cache[k_step*m1 + l1 + 3] = -INFINITY;
        if (mp[l1+0] == fms.h_inf && mp[l1+1] == fms.h_inf && mp[l1+2] == fms.h_inf && mp[l1+3] == fms.h_inf) {
            return;
        }
        auto qr = q + m1*D;
        for (int i = 0; i < D/32; ++i) qv[i] = __m512bh(_mm512_loadu_si512((const __m512i *)qr + i));
        for (int k = 0; k < 4; ++k) {
            if (mp[l1+k] == fms.h_inf) continue;
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+k*(D/32)], qv[i]);
            fms.cache[k_step*m1 + l1 + k] = _mm512_reduce_add_ps(vsum);
        }
    }

    static inline __m128 hsum_float_4x4(__m128 * a) {
        for (int i = 0; i < 2; ++i) a[i] = _mm_add_ps(_mm_unpacklo_ps(a[i], a[i+2]), _mm_unpackhi_ps(a[i], a[i+2]));
        return _mm_add_ps(_mm_unpacklo_ps(a[0], a[1]), _mm_unpackhi_ps(a[0], a[1]));
    }

    template <typename KHelper>
    static inline void multiply_mask_kq(const KHelper& kh, int stride_q, int stride_m, const float * q,
            const char * mask, FlashMS<q_step, k_step>& fms) {
        {
            __m512bh qv[D/32];
            if constexpr (D <= 128) {
                __m512bh vkh[D/8];
                for (int l1 = 0; l1 < k_step; l1 += 4) {
                    kh.load_4(l1, vkh);
                    for (int j = 0; j < q_step; ++j) {
                        mult_mask_kq_4(l1, j, stride_q, stride_m, q, mask, qv, vkh, fms);
                    }
                }
            } else {
                __m512bh vkh[D/16];
                for (int l1 = 0; l1 < k_step; l1 += 2) {
                    kh.load_2(l1, vkh);
                    for (int j = 0; j < q_step; ++j) {
                        mult_mask_kq_one(l1, j, stride_q, stride_m, q, mask, qv, vkh, fms);
                    }
                }
            }
        }
        __m512 vk[k_step/16];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk);
        }
    }

    static inline void mult_mask_kq_4(int l1, int m1, const ggml_bf16_t * q,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        auto qr = q + m1*D;
        for (int i = 0; i < D/32; ++i) qv[i] = __m512bh(_mm512_loadu_si512((const __m512i *)qr + i));
        __m128 sum[4];
        for (int k = 0; k < 4; ++k) {
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+k*(D/32)], qv[i]);
            auto aux = _mm256_add_ps(_mm512_castps512_ps256(vsum), _mm512_extractf32x8_ps(vsum, 1));
            sum[k] = _mm_add_ps(_mm256_castps256_ps128(aux), _mm256_extractf128_ps(aux, 1));
        }
        //auto sum4 = _mm_mask_blend_ps(m8, hsum_float_4x4(sum), _mm_set1_ps(-INFINITY));
        //_mm_storeu_ps(fms.cache + k_step*m1 + l1, sum4);
        _mm_storeu_ps(fms.cache + k_step*m1 + l1, hsum_float_4x4(sum));
    }

    static IQK_ALWAYS_INLINE __m256 hsum_float_8x8(__m256 * accm) {
        for (int i = 0; i < 4; ++i) {
            accm[i] = _mm256_add_ps(_mm256_permute2f128_ps(accm[i], accm[i+4], 0x20), _mm256_permute2f128_ps(accm[i], accm[i+4], 0x31));
            //accm[i] = _mm256_set_m128(_mm_add_ps(_mm256_castps256_ps128(accm[i+4]), _mm256_extractf128_ps(accm[i+4], 1)),
            //                          _mm_add_ps(_mm256_castps256_ps128(accm[i+0]), _mm256_extractf128_ps(accm[i+0], 1)));
        }
        for (int i = 0; i < 2; ++i) accm[i] = _mm256_add_ps(_mm256_unpacklo_ps(accm[i], accm[i+2]), _mm256_unpackhi_ps(accm[i], accm[i+2]));
        return _mm256_add_ps(_mm256_unpacklo_ps(accm[0], accm[1]), _mm256_unpackhi_ps(accm[0], accm[1]));
    }

    static inline void mult_mask_kq_8(int l1, int m1, const ggml_bf16_t * q,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        auto qr = q + m1*D;
        for (int i = 0; i < D/32; ++i) qv[i] = __m512bh(_mm512_loadu_si512((const __m512i *)qr + i));
        __m256 sum[8];
        for (int k = 0; k < 8; ++k) {
            auto vsum = _mm512_setzero_ps();
            for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+k*(D/32)], qv[i]);
            sum[k] = _mm256_add_ps(_mm512_castps512_ps256(vsum), _mm512_extractf32x8_ps(vsum, 1));
        }
        _mm256_storeu_ps(fms.cache + k_step*m1 + l1, hsum_float_8x8(sum));
    }

    static inline void mult_mask_kq_one(int l1, int m1, const ggml_bf16_t * q,
            __m512bh * qv, const __m512bh * vkh, FlashMS<q_step, k_step>& fms) {
        auto qr = q + m1*D;
        for (int i = 0; i < D/32; ++i) qv[i] = __m512bh(_mm512_loadu_si512((const __m512i*)qr + i));
        auto vsum = _mm512_setzero_ps();
        for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i], qv[i]);
        fms.cache[k_step*m1 + l1 + 0] = _mm512_reduce_add_ps(vsum);
        vsum = _mm512_setzero_ps();
        for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vkh[i+D/32], qv[i]);
        fms.cache[k_step*m1 + l1 + 1] = _mm512_reduce_add_ps(vsum);
    }

#if FA_TIMING
    template <typename KHelper>
    static inline void multiply_mask_kq(const KHelper& kh, int stride_m, const ggml_bf16_t * q,
            const char * mask, FlashMS<q_step, k_step>& fms, Perf& perf) {
        auto t1 = Perf::cur_time();
#else
    template <typename KHelper>
    static inline void multiply_mask_kq(const KHelper& kh, int stride_m, const ggml_bf16_t * q,
            const char * mask, FlashMS<q_step, k_step>& fms) {
#endif
        if constexpr (q_step == 1) {
            __m512bh vq[D/32];
            __m512bh vk[D/32];
            __m256   sum[8];
            for (int i = 0; i < D/32; ++i) vq[i] = __m512bh(_mm512_loadu_si512((const __m512i *)q + i));
            for (int l = 0; l < k_step; l += 8) {
                for (int k = 0; k < 8; ++k) {
                    kh.load(l+k, vk);
                    auto vsum = _mm512_setzero_ps();
                    for (int i = 0; i < D/32; ++i) vsum = _mm512_dpbf16_ps(vsum, vk[i], vq[i]);
                    sum[k] = _mm256_add_ps(_mm512_castps512_ps256(vsum), _mm512_extractf32x8_ps(vsum, 1));
                }
                _mm256_storeu_ps(fms.cache + l, hsum_float_8x8(sum));
            }
        }
        else {
            __m512bh qv[D/32];
            if constexpr (D <= 128) {
                __m512bh vkh[D/4];
                for (int l1 = 0; l1 < k_step; l1 += 8) {
                    kh.load_8(l1, vkh);
                    for (int j = 0; j < q_step; ++j) mult_mask_kq_8(l1, j, q, qv, vkh, fms);
                }
            } else {
                __m512bh vkh[D/16];
                for (int l1 = 0; l1 < k_step; l1 += 2) {
                    kh.load_2(l1, vkh);
                    for (int j = 0; j < q_step; ++j) mult_mask_kq_one(l1, j, q, qv, vkh, fms);
                }
            }
        }
#if FA_TIMING
        perf.accum_nolock(1, t1);
        t1 = Perf::cur_time();
#endif
        F16::Data vk[k_step/16];
        for (int j = 0; j < q_step; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
#if FA_TIMING
        perf.accum_nolock(2, t1);
#endif
    }

    template <typename KHelper>
    static inline void multiply_mask_kq(int nq, const KHelper& kh, int stride_m, const ggml_bf16_t * q,
            const char * mask, FlashMS<q_step, k_step>& fms) {
        {
            __m512bh qv[D/32];
            if constexpr (D <= 128) {
                __m512bh vkh[D/8];
                for (int l1 = 0; l1 < k_step; l1 += 4) {
                    kh.load_4(l1, vkh);
                    for (int j = 0; j < nq; ++j) mult_mask_kq_4(l1, j, q, qv, vkh, fms);
                }
            } else {
                __m512bh vkh[D/16];
                for (int l1 = 0; l1 < k_step; l1 += 2) {
                    kh.load_2(l1, vkh);
                    for (int j = 0; j < nq; ++j) mult_mask_kq_one(l1, j, q, qv, vkh, fms);
                }
            }
        }
        F16::Data vk[k_step/16];
        for (int j = 0; j < nq; ++j) {
            fms.update_M_S(j, vk, mask + stride_m*j);
        }
    }

    template <typename KHelper>
    static inline void multiply_mask_kq(int nq, const KHelper& kh, int stride_q, int stride_m, const float * q,
            const char * mask, FlashMS<q_step, k_step>& fms) {
        {
            __m512bh qv[D/32];
            __m512bh vkh[D/16];
            for (int l1 = 0; l1 < k_step; l1 += 2) {
                kh.load_2(l1, vkh);
                for (int m1 = 0; m1 < nq; ++m1) {
                    mult_mask_kq_one(l1, m1, stride_q, stride_m, q, mask, qv, vkh, fms);
                }
            }
        }
        __m512 vk[k_step/16];
        for (int j = 0; j < nq; ++j) {
            fms.update_M_S(j, vk);
        }
    }

    static inline void convert(int stride_q, const float * q, ggml_bf16_t * bf16) {
        auto qr = q;
        for (int j = 0; j < q_step; ++j) {
            for (int i = 0; i < D/32; ++i) {
                auto val1 = _mm512_loadu_ps(qr + 32*i);
                auto val2 = _mm512_loadu_ps(qr + 32*i + 16);
                _mm512_storeu_si512((__m512i *)bf16 + i, (__m512i)_mm512_cvtne2ps_pbh(val2, val1));
            }
            qr   += stride_q;
            bf16 += D;
        }
    }

    static inline void convert(int nq, int stride_q, const float * q, ggml_bf16_t * bf16) {
        auto qr = q;
        for (int j = 0; j < nq; ++j) {
            for (int i = 0; i < D/32; ++i) {
                auto val1 = _mm512_loadu_ps(qr + 32*i);
                auto val2 = _mm512_loadu_ps(qr + 32*i + 16);
                _mm512_storeu_si512((__m512i *)bf16 + i, (__m512i)_mm512_cvtne2ps_pbh(val2, val1));
            }
            qr   += stride_q;
            bf16 += D;
        }
    }
};

template <int Dk, int Dv, int q_step, int k_step>
struct FlashAttnBF16 {
    //static_assert(Dk%32 == 0 && Dk <= 256);
    //static_assert(Dv%32 == 0 && Dv <= 256);
    static_assert(Dk%32 == 0 && Dk <= 576);
    static_assert(Dv%32 == 0 && Dv <= 512);
    static_assert(k_step%32 == 0);
    static_assert(q_step <= 4 || q_step%4 == 0);

    FlashAttnBF16(float scale, float softcap, const float * sinkf) : fms(scale, softcap), sinkf(sinkf) {}

    template <typename KHelper, typename VHelper>
    void compute(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
            const float * q, const char * mask, float * qkv, [[maybe_unused]] float * M, [[maybe_unused]] float * S) {
        ggml_bf16_t q_bf16[q_step*Dk];
#if FA_TIMING
        Perf perf(false);
#endif
        for (int i1 = 0; i1 < nq1/q_step; ++i1) {
#if FA_TIMING
            auto t1 = Perf::cur_time();
#endif
            fms.init_qstep();
            kh.reset_block();
            vh.reset_block();
            FlashQKbf16<Dk, q_step, k_step>::convert(stride_q, q, q_bf16);
#if FA_TIMING
            perf.accum_nolock(0, t1);
#endif
            auto mr = mask;
            auto Mc = (const uint16_t *)(mr + (q_step - 1)*stride_m);
            int ik = nk1 - k_step;
            for (; ik >=0 && Mc[ik] != 0; ik -= k_step);
            ik += k_step;
            for (int k1 = 0; k1 < ik/k_step; ++k1) {
#if FA_TIMING
                //t1 = Perf::cur_time();
                FlashQKbf16<Dk, q_step, k_step>::multiply_mask_kq(kh, stride_m, q_bf16, mr, fms, perf);
                //perf.accum_nolock(1, t1);
                t1 = Perf::cur_time();
                fqkv.accumulate_qkv(vh, fms);
                perf.accum_nolock(3, t1);
#else
                FlashQKbf16<Dk, q_step, k_step>::multiply_mask_kq(kh, stride_m, q_bf16, mr, fms);
                fqkv.accumulate_qkv(vh, fms);
#endif
                kh.next_block(k_step);
                vh.next_block(k_step);
                mr += k_step*sizeof(ggml_half);
            }
#if FA_TIMING
            t1 = Perf::cur_time();
#endif
            fqkv.normalize_and_store(fms, stride_qkv, qkv, sinkf, M, S);
#if FA_TIMING
            perf.accum_nolock(4, t1);
#endif

            q    += q_step*stride_q;
            mask += q_step*stride_m;
            qkv  += q_step*stride_qkv;
        }
        int n_left = nq1 - q_step*(nq1/q_step);
        if (n_left > 0) {
            fms.init_qstep();
            kh.reset_block();
            vh.reset_block();
            FlashQKbf16<Dk, q_step, k_step>::convert(n_left, stride_q, q, q_bf16);
            auto mr = mask;
            for (int k1 = 0; k1 < nk1/k_step; ++k1) {
                FlashQKbf16<Dk, q_step, k_step>::multiply_mask_kq(n_left, kh, stride_m, q_bf16, mr, fms);
                fqkv.accumulate_qkv(n_left, vh, fms);
                kh.next_block(k_step);
                vh.next_block(k_step);
                mr += k_step*sizeof(ggml_half);
            }
            fqkv.normalize_and_store(fms, n_left, stride_qkv, qkv, sinkf, M, S);
        }
#if FA_TIMING
        Perf::instance().add(perf);
#endif
    }

    FlashMS<q_step, k_step>      fms;
    FlashQKV<Dv, q_step, k_step> fqkv;
    const float *                sinkf;
};
#endif

template <int Dk, int Dv, int k_step, typename KHelper, typename VHelper>
inline void iqk_flash_helper(KHelper& kh, VHelper& vh, int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
                        const float * q, const char * mask, float scale, float softcap, float * qkv,
                        const float * sinkf, float * M, float * S) {

    auto update = [&nq1, &mask, &q, &qkv, &M, &S, stride_q, stride_m, stride_qkv] (int n) {
        nq1 -= n;
        if (nq1 == 0) return true;
        q    += n*stride_q;
        mask += n*stride_m;
        qkv  += n*stride_qkv;
        if (M && S) { M += n; S += n; }
        return false;
    };
    if (nk1 >= 512) {
        if (nq1 >= 128) {
            int n_step = nq1/128;
            FlashAttn<Dk, Dv, 64, k_step> fa(scale, softcap, sinkf);
            fa.compute(kh, vh, 128*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            if (update(128*n_step)) return;
        }
        if (nq1 >= 64) {
            int n_step = nq1/64;
            FlashAttn<Dk, Dv, 64, k_step> fa(scale, softcap, sinkf);
            fa.compute(kh, vh, 64*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            if (update(64*n_step)) return;
        }
        if (nq1 >= 32) {
            int n_step = nq1/32;
            FlashAttn<Dk, Dv, 32, k_step> fa(scale, softcap, sinkf);
            fa.compute(kh, vh, 32*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            if (update(32*n_step)) return;
        }
        if (nq1 >= 16) {
            int n_step = nq1/16;
            FlashAttn<Dk, Dv, 16, k_step> fa(scale, softcap, sinkf);
            fa.compute(kh, vh, 16*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            if (update(16*n_step)) return;
        }
    }
    if (nq1 == 12) {
        // Special case: TG for GLM-4.5/4.6
        FlashAttn<Dk, Dv, 12, k_step> fa(scale, softcap, sinkf);
        fa.compute(kh, vh, 12, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
        return;
    }
    if (nq1 >= 8) {
        int n_step = nq1/8;
        FlashAttn<Dk, Dv, 8, k_step> fa(scale, softcap, sinkf);
        fa.compute(kh, vh, 8*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
        if (update(8*n_step)) return;
    }
    else if (nq1 >= 4) {
        int n_step = nq1/4;
        FlashAttn<Dk, Dv, 4, k_step> fa(scale, softcap, sinkf);
        fa.compute(kh, vh, 4*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
        if (update(4*n_step)) return;
    }
    else if (nq1 >= 2) {
        int n_step = nq1/2;
        FlashAttn<Dk, Dv, 2, k_step> fa(scale, softcap, sinkf);
        fa.compute(kh, vh, 2*n_step, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
        if (update(2*n_step)) return;
    }
    FlashAttn<Dk, Dv, 1, k_step> fa(scale, softcap, sinkf);
    fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
}

#ifdef __AVX512BF16__
template <int Dk, int Dv, int k_step>
inline void iqk_flash_helper_T(int nq1, int nk1, int stride_q, int stride_k, int stride_v, int stride_m, int stride_qkv,
                        const float * q, const char * k, const char * v, const char * mask,
                        float scale, float softcap, float * qkv, const float * sinkf, float * M, float * S) {
    HelperBF16<Dk, k_step> kh(k, stride_k);
    HelperBF16<Dv, k_step> vh(v, stride_v);
    if (nk1 >= 4096) {
        if (nq1 >= 64) {
            FlashAttnBF16<Dk, Dv, 64, k_step> fa(scale, softcap, sinkf);
            fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            return;
        }
        else if (nq1 >= 16) {
            FlashAttnBF16<Dk, Dv, 16, k_step> fa(scale, softcap, sinkf);
            fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
            return;
        }
    }
    if (nq1 >= 8) {
        FlashAttnBF16<Dk, Dv, 8, k_step> fa(scale, softcap, sinkf);
        fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
    } else {
        FlashAttnBF16<Dk, Dv, 1, k_step> fa(scale, softcap, sinkf);
        fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
    }
}
#endif

template <int Dk, int Dv, int k_step, typename KHelper>
inline bool iqk_flash_helper_T(KHelper& kh, ggml_type type_v,
                        int nq1, int nk1, int stride_q, int stride_v, int stride_m, int stride_qkv,
                        const float * q, const char * v, const char * mask,
                        float scale, float softcap, float * qkv, const float * sinkf, float * M, float * S) {

    switch (type_v) {
        case GGML_TYPE_F16: {
            HelperF16 vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
#ifdef __AVX512BF16__
        case GGML_TYPE_BF16: {
            HelperBF16<Dv, k_step> vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
#endif
        case GGML_TYPE_Q8_0: {
            HelperQ80 vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
        case GGML_TYPE_Q8_KV: {
            HelperQ8KV<Dv> vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
        case GGML_TYPE_Q6_0: {
            HelperQ60 vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
#if GGML_IQK_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0: {
            HelperQ40 vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
        case GGML_TYPE_Q4_1: {
            HelperQ41 vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
        case GGML_TYPE_IQ4_NL: {
            HelperIQ4nl vh(v, stride_v);
            iqk_flash_helper<Dk, Dv, k_step>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
#endif
        default: return false;
    }
    return true;
}

template <int Dk, int Dv, int k_step>
inline bool iqk_flash_helper_T(ggml_type type_k, ggml_type type_v,
                        int nq1, int nk1, int stride_q, int stride_k, int stride_v, int stride_m, int stride_qkv,
                        const float * q, const char * k, const char * v, const char * mask,
                        float scale, float softcap, float * qkv, const float * sinkf, float * M, float * S) {

    bool result = false;
    switch (type_k) {
        case GGML_TYPE_F16: {
            HelperF16 kh(k, stride_k);
            result = iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
        case GGML_TYPE_Q8_0: {
            HelperQ80 kh(k, stride_k);
            result = iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
        case GGML_TYPE_Q8_0_R8: {
            HelperQ80R8<Dk> kh(k, stride_k);
            result = iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
        case GGML_TYPE_Q6_0: {
            HelperQ60 kh(k, stride_k);
            result = iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
#if GGML_IQK_FA_ALL_QUANTS
        case GGML_TYPE_Q8_KV: {
            HelperQ8KV<Dk> kh(k, stride_k);
            result = iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
        case GGML_TYPE_Q4_0: {
            HelperQ40 kh(k, stride_k);
            result = iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
        case GGML_TYPE_Q4_1: {
            HelperQ41 kh(k, stride_k);
            result = iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
        case GGML_TYPE_IQ4_NL: {
            HelperIQ4nl kh(k, stride_k);
            result = iqk_flash_helper_T<Dk, Dv, k_step>(kh, type_v, nq1, nk1, stride_q, stride_v, stride_m, stride_qkv, q, v, mask, scale, softcap, qkv, sinkf, M, S);
        } break;
#endif
        default: break;
    }

    return result;
}

}

#define IQK_FA_CASE(name) bool name(int int_type_k, int int_type_v,int nq,int nk,\
                         int stride_q, int stride_k, int stride_v, int stride_m, int stride_qkv,\
                         const float * q, const void * k, const void * v, const void * mask,\
                         float scale, float softcap,\
                         float       * qkv, const float * sinkf, float * M, float * S)

IQK_FA_CASE(iqk_fa_576_512);
IQK_FA_CASE(iqk_fa_192_128);
IQK_FA_CASE(iqk_fa_192_192);
IQK_FA_CASE(iqk_fa_256_256);
IQK_FA_CASE(iqk_fa_128_128);
IQK_FA_CASE(iqk_fa_96_96);
IQK_FA_CASE(iqk_fa_64_64);

#endif

