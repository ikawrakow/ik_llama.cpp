// quillama.cpp: Intel AMX BF16 GEMM for CPU prompt processing. See quill-amx.h.
//
// Layout
//   A (weights):     a thread-local panel of MR weight rows dequantized to BF16, row-major [MR][Kp].
//                    An A tile is 16 rows x 32 BF16 (64 bytes) at stride Kp*2.
//   B (activations): packed once per matmul (shared by all threads) in the AMX "VNNI" layout:
//                    [T/16 groups][Kp/2 k-pairs][16 tokens][2 BF16]. A B tile is 16 consecutive
//                    k-pairs (= 32 K values) of one 16-token group, stride 64 bytes.
//   C:               C[m][t] (16 weight rows x 16 tokens, FP32) accumulates over the full K in the
//                    tile registers, then is transposed in AVX-512 registers and written to dst[t][m].
//
// The micro-kernel is the usual 2x2 tile block: 2 A tiles x 2 B tiles -> 4 C tiles (32 rows x 32
// tokens), 8 tiles in total. The FFN up/gate variant uses 16 rows x 32 tokens for both matrices.

#include "quill-amx.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <atomic>
#include <algorithm>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && defined(__linux__)
#define QUILL_AMX_BUILD 1
#else
#define QUILL_AMX_BUILD 0
#endif

#if QUILL_AMX_BUILD

// Everything below is compiled for AVX-512 + AMX and must only run after quill_amx_available()
// returned true. CPU detection and the eligibility checks, which run on every CPU, live in
// quill-amx-dispatch.cpp, compiled with the default flags.
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512dq,avx512bf16,f16c,fma,amx-tile,amx-bf16")
#include <immintrin.h>

namespace {

constexpr int64_t kTileK   = 32;   // BF16 elements per tile row (64 bytes)
constexpr int64_t kGroup   = 16;   // tokens per packed group / rows per tile

inline int64_t round_up(int64_t x, int64_t n) { return (x + n - 1)/n*n; }

struct alignas(64) tile_config {
    uint8_t  palette_id = 1;
    uint8_t  start_row  = 0;
    uint8_t  reserved[14] = {};
    uint16_t colsb[16] = {};
    uint8_t  rows[16]  = {};
};

void ensure_tile_config() {
    thread_local bool configured = false;
    if (configured) return;
    tile_config cfg;
    for (int i = 0; i < 8; ++i) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
    _tile_loadconfig(&cfg);
    configured = true;
}

template <typename T>
struct aligned_buffer {
    T *    ptr  = nullptr;
    size_t size = 0;
    ~aligned_buffer() { free(ptr); }
    T * get(size_t n) {
        if (n > size) {
            free(ptr);
            size = 0;
            ptr  = (T *) aligned_alloc(64, round_up(n*sizeof(T), 64));
            if (ptr == nullptr) {
                GGML_ABORT("quill-amx: out of memory allocating a %zu-byte scratch buffer (QUILL_AMX=0 disables AMX)", n*sizeof(T));
            }
            size = n;
        }
        return ptr;
    }
};

// 16x16 transpose of 32-bit elements held in 16 zmm registers
inline void transpose16x16(__m512i r[16]) {
    __m512i t[16];
    for (int i = 0; i < 8; ++i) {
        t[2*i+0] = _mm512_unpacklo_epi32(r[2*i], r[2*i+1]);
        t[2*i+1] = _mm512_unpackhi_epi32(r[2*i], r[2*i+1]);
    }
    for (int i = 0; i < 4; ++i) {
        r[4*i+0] = _mm512_unpacklo_epi64(t[4*i+0], t[4*i+2]);
        r[4*i+1] = _mm512_unpackhi_epi64(t[4*i+0], t[4*i+2]);
        r[4*i+2] = _mm512_unpacklo_epi64(t[4*i+1], t[4*i+3]);
        r[4*i+3] = _mm512_unpackhi_epi64(t[4*i+1], t[4*i+3]);
    }
    // now r[4i+q], 128-bit lane j = column 4j+q of rows 4i..4i+3: transpose the 4x4 grid of lanes
    for (int q = 0; q < 4; ++q) {
        __m512i a = r[q], b = r[4+q], c = r[8+q], d = r[12+q];
        __m512i u0 = _mm512_shuffle_i32x4(a, b, 0x44);
        __m512i u1 = _mm512_shuffle_i32x4(a, b, 0xEE);
        __m512i u2 = _mm512_shuffle_i32x4(c, d, 0x44);
        __m512i u3 = _mm512_shuffle_i32x4(c, d, 0xEE);
        t[ 0+q] = _mm512_shuffle_i32x4(u0, u2, 0x88);
        t[ 4+q] = _mm512_shuffle_i32x4(u0, u2, 0xDD);
        t[ 8+q] = _mm512_shuffle_i32x4(u1, u3, 0x88);
        t[12+q] = _mm512_shuffle_i32x4(u1, u3, 0xDD);
    }
    for (int i = 0; i < 16; ++i) r[i] = t[i];
}

inline __mmask16 tail_mask(int64_t n) {
    return n >= 16 ? (__mmask16)0xffff : n <= 0 ? (__mmask16)0 : (__mmask16)((1u << n) - 1);
}

// f32 row (K values) -> BF16 row padded with zeros to Kp
inline void f32_to_bf16_row(const float * x, int64_t K, int64_t Kp, ggml_bf16_t * y) {
    for (int64_t k = 0; k < Kp; k += 32) {
        const __m512 lo = _mm512_maskz_loadu_ps(tail_mask(K - k),      x + std::min(k, K));
        const __m512 hi = _mm512_maskz_loadu_ps(tail_mask(K - k - 16), x + std::min(k + 16, K));
        _mm512_store_si512((__m512i *)(y + k), (__m512i)_mm512_cvtne2ps_pbh(hi, lo));
    }
}

inline void f16_to_bf16_row(const ggml_fp16_t * x, int64_t K, int64_t Kp, ggml_bf16_t * y) {
    for (int64_t k = 0; k < Kp; k += 32) {
        const __m512 lo = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tail_mask(K - k),      x + std::min(k, K)));
        const __m512 hi = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(tail_mask(K - k - 16), x + std::min(k + 16, K)));
        _mm512_store_si512((__m512i *)(y + k), (__m512i)_mm512_cvtne2ps_pbh(hi, lo));
    }
}

inline void bf16_copy_row(const ggml_bf16_t * x, int64_t K, int64_t Kp, ggml_bf16_t * y) {
    for (int64_t k = 0; k < Kp; k += 32) {
        const int64_t n = std::max<int64_t>(0, std::min<int64_t>(32, K - k));
        const __mmask32 m = n >= 32 ? (__mmask32)0xffffffffu : (__mmask32)((1u << n) - 1);
        _mm512_store_si512((__m512i *)(y + k), _mm512_maskz_loadu_epi16(m, x + std::min(k, K)));
    }
}

// ---------------------------------------------------------------------------------------------
// Fast AVX-512 dequantization of the common GGUF weight types (raw block layouts, see
// ggml-common.h). ggml's to_float() is scalar for most types and would dominate the cost when an
// expert sees only a few dozen tokens. Each function writes K floats.
// ---------------------------------------------------------------------------------------------

inline float h2f(const uint8_t * p) { uint16_t h; memcpy(&h, p, 2); return _cvtsh_ss(h); }

// y[0..16) = a*u + b for 16 unsigned bytes u
inline void fma_u8(float * y, __m128i u, __m512 a, __m512 b) {
    _mm512_storeu_ps(y, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u)), a, b));
}
// y[0..16) = a*s for 16 signed bytes s
inline void mul_s8(float * y, __m128i v, __m512 a) {
    _mm512_storeu_ps(y, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v)), a));
}

void dq_q4_0(const uint8_t * x, int64_t K, float * y) {
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (int64_t i = 0; i < K/32; ++i, x += 18, y += 32) {
        const float d = h2f(x);
        const __m512 a = _mm512_set1_ps(d), b = _mm512_set1_ps(-8.0f*d);
        const __m128i q = _mm_loadu_si128((const __m128i *)(x + 2));
        fma_u8(y,      _mm_and_si128(q, m4), a, b);
        fma_u8(y + 16, _mm_and_si128(_mm_srli_epi16(q, 4), m4), a, b);
    }
}

void dq_q8_0(const uint8_t * x, int64_t K, float * y) {
    for (int64_t i = 0; i < K/32; ++i, x += 34, y += 32) {
        const __m512 a = _mm512_set1_ps(h2f(x));
        mul_s8(y,      _mm_loadu_si128((const __m128i *)(x + 2)),  a);
        mul_s8(y + 16, _mm_loadu_si128((const __m128i *)(x + 18)), a);
    }
}

static const float k_iq4nl_f[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline void lut4(float * y, __m128i u, __m512 lut, __m512 scale) {
    _mm512_storeu_ps(y, _mm512_mul_ps(_mm512_permutexvar_ps(_mm512_cvtepu8_epi32(u), lut), scale));
}

void dq_iq4_nl(const uint8_t * x, int64_t K, float * y) {
    const __m512 lut = _mm512_loadu_ps(k_iq4nl_f);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (int64_t i = 0; i < K/32; ++i, x += 18, y += 32) {
        const __m512 d = _mm512_set1_ps(h2f(x));
        const __m128i q = _mm_loadu_si128((const __m128i *)(x + 2));
        lut4(y,      _mm_and_si128(q, m4), lut, d);
        lut4(y + 16, _mm_and_si128(_mm_srli_epi16(q, 4), m4), lut, d);
    }
}

void dq_iq4_xs(const uint8_t * x, int64_t K, float * y) {
    // { half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128] } = 136 bytes
    const __m512 lut = _mm512_loadu_ps(k_iq4nl_f);
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (int64_t i = 0; i < K/256; ++i, x += 136) {
        const float d = h2f(x);
        uint16_t sh; memcpy(&sh, x + 2, 2);
        const uint8_t * sl = x + 4;
        const uint8_t * qs = x + 8;
        for (int ib = 0; ib < 8; ++ib, qs += 16, y += 32) {
            const int ls = ((sl[ib/2] >> 4*(ib%2)) & 0xf) | (((sh >> 2*ib) & 3) << 4);
            const __m512 dl = _mm512_set1_ps(d*(ls - 32));
            const __m128i q = _mm_loadu_si128((const __m128i *)qs);
            lut4(y,      _mm_and_si128(q, m4), lut, dl);
            lut4(y + 16, _mm_and_si128(_mm_srli_epi16(q, 4), m4), lut, dl);
        }
    }
}

inline void k4_scale_min(int j, const uint8_t * q, uint8_t & sc, uint8_t & m) {
    if (j < 4) {
        sc = q[j] & 63; m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

void dq_q4_k(const uint8_t * x, int64_t K, float * y) {
    // { half d; half dmin; uint8 scales[12]; uint8 qs[128] } = 144 bytes
    const __m256i m4 = _mm256_set1_epi8(0x0f);
    for (int64_t i = 0; i < K/256; ++i, x += 144) {
        const float d = h2f(x), dmin = h2f(x + 2);
        const uint8_t * sc = x + 4;
        const uint8_t * qs = x + 16;
        for (int j = 0; j < 4; ++j, qs += 32, y += 64) {
            uint8_t s1, m1, s2, m2;
            k4_scale_min(2*j + 0, sc, s1, m1);
            k4_scale_min(2*j + 1, sc, s2, m2);
            const __m256i q  = _mm256_loadu_si256((const __m256i *)qs);
            const __m256i lo = _mm256_and_si256(q, m4);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            const __m512 a1 = _mm512_set1_ps(d*s1), b1 = _mm512_set1_ps(-dmin*m1);
            const __m512 a2 = _mm512_set1_ps(d*s2), b2 = _mm512_set1_ps(-dmin*m2);
            fma_u8(y +  0, _mm256_castsi256_si128(lo),      a1, b1);
            fma_u8(y + 16, _mm256_extracti128_si256(lo, 1), a1, b1);
            fma_u8(y + 32, _mm256_castsi256_si128(hi),      a2, b2);
            fma_u8(y + 48, _mm256_extracti128_si256(hi, 1), a2, b2);
        }
    }
}

void dq_q5_k(const uint8_t * x, int64_t K, float * y) {
    // { half d; half dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128] } = 176 bytes
    const __m256i m4 = _mm256_set1_epi8(0x0f), one = _mm256_set1_epi8(1);
    for (int64_t i = 0; i < K/256; ++i, x += 176) {
        const float d = h2f(x), dmin = h2f(x + 2);
        const uint8_t * sc = x + 4;
        const __m256i qh = _mm256_loadu_si256((const __m256i *)(x + 16));
        const uint8_t * qs = x + 48;
        for (int j = 0; j < 4; ++j, qs += 32, y += 64) {
            uint8_t s1, m1, s2, m2;
            k4_scale_min(2*j + 0, sc, s1, m1);
            k4_scale_min(2*j + 1, sc, s2, m2);
            const __m256i q  = _mm256_loadu_si256((const __m256i *)qs);
            const __m256i h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2*j + 0), one), 4);
            const __m256i h2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2*j + 1), one), 4);
            const __m256i lo = _mm256_or_si256(_mm256_and_si256(q, m4), h1);
            const __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q, 4), m4), h2);
            const __m512 a1 = _mm512_set1_ps(d*s1), b1 = _mm512_set1_ps(-dmin*m1);
            const __m512 a2 = _mm512_set1_ps(d*s2), b2 = _mm512_set1_ps(-dmin*m2);
            fma_u8(y +  0, _mm256_castsi256_si128(lo),      a1, b1);
            fma_u8(y + 16, _mm256_extracti128_si256(lo, 1), a1, b1);
            fma_u8(y + 32, _mm256_castsi256_si128(hi),      a2, b2);
            fma_u8(y + 48, _mm256_extracti128_si256(hi, 1), a2, b2);
        }
    }
}

void dq_q6_k(const uint8_t * x, int64_t K, float * y) {
    // { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d } = 210 bytes
    const __m256i m4 = _mm256_set1_epi8(0x0f), m3 = _mm256_set1_epi8(3), m32 = _mm256_set1_epi8(32);
    for (int64_t i = 0; i < K/256; ++i, x += 210) {
        const float d = h2f(x + 208);
        const int8_t * sc = (const int8_t *)(x + 192);
        for (int n = 0; n < 2; ++n, y += 128) {
            const uint8_t * ql = x + 64*n;
            const uint8_t * qh = x + 128 + 32*n;
            const int8_t  * s  = sc + 8*n;
            const __m256i l0 = _mm256_loadu_si256((const __m256i *)ql);
            const __m256i l1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            const __m256i h  = _mm256_loadu_si256((const __m256i *)qh);
            __m256i q[4];
            q[0] = _mm256_or_si256(_mm256_and_si256(l0, m4),                         _mm256_slli_epi16(_mm256_and_si256(h, m3), 4));
            q[1] = _mm256_or_si256(_mm256_and_si256(l1, m4),                         _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), m3), 4));
            q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4),   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 4), m3), 4));
            q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4),   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 6), m3), 4));
            for (int k = 0; k < 4; ++k) {
                const __m256i v = _mm256_sub_epi8(q[k], m32);
                mul_s8(y + 32*k,      _mm256_castsi256_si128(v),      _mm512_set1_ps(d*s[2*k + 0]));
                mul_s8(y + 32*k + 16, _mm256_extracti128_si256(v, 1), _mm512_set1_ps(d*s[2*k + 1]));
            }
        }
    }
}

// --- 4-row interleaved repacks (ik's run-time repack, -rtr): write 4 consecutive rows of K floats ---
//
// Within each 32-value sub-block, the 64 quant bytes hold four 16-byte groups X0..X3; dword k of each
// group belongs to row k. With v0 = lo(X0), v1 = hi(X0), v2 = lo(X1), v3 = hi(X1), v4 = lo(X2),
// v5 = hi(X2), v6 = lo(X3), v7 = hi(X3), row k is the dwords [v0 v4 v1 v5 | v2 v6 v3 v7] (k-th each).
inline void transpose4x4_epi32(__m128i & a, __m128i & b, __m128i & c, __m128i & d) {
    __m128 fa = _mm_castsi128_ps(a), fb = _mm_castsi128_ps(b), fc = _mm_castsi128_ps(c), fd = _mm_castsi128_ps(d);
    _MM_TRANSPOSE4_PS(fa, fb, fc, fd);
    a = _mm_castps_si128(fa); b = _mm_castps_si128(fb); c = _mm_castps_si128(fc); d = _mm_castps_si128(fd);
}

void dq_q4_k_r4(const uint8_t * x, int64_t K, float * y) {
    // { half d[8] (d for rows 0-3, dmin for rows 0-3); uint8 scales_h[16]; uint8 scales_l[32]; uint8 qs[512] } = 576 bytes
    const __m128i m4 = _mm_set1_epi8(0x0f);
    for (int64_t ibl = 0; ibl < K/256; ++ibl, x += 576) {
        float d[4], m[4];
        for (int k = 0; k < 4; ++k) { d[k] = h2f(x + 2*k); m[k] = h2f(x + 8 + 2*k); }
        const uint8_t * sh = x + 16;
        const uint8_t * sl = x + 32;
        const uint8_t * qs = x + 64;
        for (int ib = 0; ib < 8; ++ib, qs += 64) {
            __m128i X[4];
            for (int g = 0; g < 4; ++g) X[g] = _mm_loadu_si128((const __m128i *)(qs + 16*g));
            __m128i v[8];
            for (int g = 0; g < 4; ++g) {
                v[2*g+0] = _mm_and_si128(X[g], m4);
                v[2*g+1] = _mm_and_si128(_mm_srli_epi16(X[g], 4), m4);
            }
            __m128i a0 = v[0], a1 = v[4], a2 = v[1], a3 = v[5];
            __m128i b0 = v[2], b1 = v[6], b2 = v[3], b3 = v[7];
            transpose4x4_epi32(a0, a1, a2, a3);
            transpose4x4_epi32(b0, b1, b2, b3);
            const __m128i lo[4] = {a0, a1, a2, a3}, hi[4] = {b0, b1, b2, b3};
            for (int k = 0; k < 4; ++k) {
                const int is = 4*ib + k;
                const int s6 = (sh[is%16] >> 4*(is/16));
                const float dl = d[k]*((sl[is] & 0xf) | ((s6 & 0x03) << 4));
                const float ml = m[k]*((sl[is] >>  4) | ((s6 & 0x0c) << 2));
                const __m512 a = _mm512_set1_ps(dl), b = _mm512_set1_ps(-ml);
                float * yk = y + k*K + 256*ibl + 32*ib;
                fma_u8(yk,      lo[k], a, b);
                fma_u8(yk + 16, hi[k], a, b);
            }
        }
    }
}

void dq_q6_k_r4(const uint8_t * x, int64_t K, float * y) {
    // { half d[4]; int8 scales[64]; uint8 qh[256]; uint8 ql[512] } = 840 bytes
    const __m128i m4 = _mm_set1_epi8(0x0f), m30 = _mm_set1_epi8(0x30), m32 = _mm_set1_epi8(32);
    for (int64_t ibl = 0; ibl < K/256; ++ibl, x += 840) {
        float d[4];
        for (int k = 0; k < 4; ++k) d[k] = h2f(x + 2*k);
        const int8_t  * sc = (const int8_t *)(x + 8);
        const uint8_t * qh = x + 72;
        const uint8_t * ql = x + 328;
        for (int ib = 0; ib < 8; ++ib, ql += 64, qh += 32) {
            const __m128i X0 = _mm_loadu_si128((const __m128i *)(ql +  0));
            const __m128i X1 = _mm_loadu_si128((const __m128i *)(ql + 16));
            const __m128i X2 = _mm_loadu_si128((const __m128i *)(ql + 32));
            const __m128i X3 = _mm_loadu_si128((const __m128i *)(ql + 48));
            const __m128i H0 = _mm_loadu_si128((const __m128i *)(qh +  0));
            const __m128i H1 = _mm_loadu_si128((const __m128i *)(qh + 16));
            auto lo = [&](__m128i q) { return _mm_and_si128(q, m4); };
            auto hi = [&](__m128i q) { return _mm_and_si128(_mm_srli_epi16(q, 4), m4); };
            __m128i v[8];
            v[0] = _mm_or_si128(lo(X0), _mm_and_si128(_mm_slli_epi16(H0, 4), m30));
            v[1] = _mm_or_si128(hi(X0), _mm_and_si128(_mm_slli_epi16(H0, 2), m30));
            v[2] = _mm_or_si128(lo(X1), _mm_and_si128(_mm_slli_epi16(H1, 4), m30));
            v[3] = _mm_or_si128(hi(X1), _mm_and_si128(_mm_slli_epi16(H1, 2), m30));
            v[4] = _mm_or_si128(lo(X2), _mm_and_si128(H0, m30));
            v[5] = _mm_or_si128(hi(X2), _mm_and_si128(_mm_srli_epi16(H0, 2), m30));
            v[6] = _mm_or_si128(lo(X3), _mm_and_si128(H1, m30));
            v[7] = _mm_or_si128(hi(X3), _mm_and_si128(_mm_srli_epi16(H1, 2), m30));
            __m128i a0 = v[0], a1 = v[4], a2 = v[1], a3 = v[5];
            __m128i b0 = v[2], b1 = v[6], b2 = v[3], b3 = v[7];
            transpose4x4_epi32(a0, a1, a2, a3);
            transpose4x4_epi32(b0, b1, b2, b3);
            const __m128i first[4] = {a0, a1, a2, a3}, second[4] = {b0, b1, b2, b3};
            for (int k = 0; k < 4; ++k) {
                float * yk = y + k*K + 256*ibl + 32*ib;
                mul_s8(yk,      _mm_sub_epi8(first[k],  m32), _mm512_set1_ps(d[k]*sc[8*ib + k + 0]));
                mul_s8(yk + 16, _mm_sub_epi8(second[k], m32), _mm512_set1_ps(d[k]*sc[8*ib + k + 4]));
            }
        }
    }
}

typedef void (*fast_dq_t)(const uint8_t *, int64_t, float *);

fast_dq_t fast_dequantizer(ggml_type type, int64_t K) {
    if (getenv("QUILL_AMX_SLOW_DQ")) return nullptr;
    switch (type) {
        case GGML_TYPE_Q4_0:   return K % 32  == 0 ? dq_q4_0   : nullptr;
        case GGML_TYPE_Q8_0:   return K % 32  == 0 ? dq_q8_0   : nullptr;
        case GGML_TYPE_IQ4_NL: return K % 32  == 0 ? dq_iq4_nl : nullptr;
        case GGML_TYPE_IQ4_XS: return K % 256 == 0 ? dq_iq4_xs : nullptr;
        case GGML_TYPE_Q4_K:   return K % 256 == 0 ? dq_q4_k   : nullptr;
        case GGML_TYPE_Q5_K:   return K % 256 == 0 ? dq_q5_k   : nullptr;
        case GGML_TYPE_Q6_K:   return K % 256 == 0 ? dq_q6_k   : nullptr;
        case GGML_TYPE_Q4_K_R4: return K % 256 == 0 ? dq_q4_k_r4 : nullptr;
        case GGML_TYPE_Q6_K_R4: return K % 256 == 0 ? dq_q6_k_r4 : nullptr;
        default:               return nullptr;
    }
}

struct dequantizer {
    ggml_type       type;
    ggml_to_float_t to_float;
    int64_t         K, Kp;
    fast_dq_t       fast = nullptr;
    int64_t         ileave = 1;   // rows interleaved per group (*_R4, *_R8, *_R16 repacks)

    void row(const char * src, ggml_bf16_t * dst, float * tmp) const {
        switch (type) {
            case GGML_TYPE_F32:  f32_to_bf16_row((const float *)src, K, Kp, dst); break;
            case GGML_TYPE_F16:  f16_to_bf16_row((const ggml_fp16_t *)src, K, Kp, dst); break;
            case GGML_TYPE_BF16: bf16_copy_row((const ggml_bf16_t *)src, K, Kp, dst); break;
            default:
                if (fast) fast((const uint8_t *)src, K, tmp);
                else      to_float(src, tmp, K);
                f32_to_bf16_row(tmp, K, Kp, dst);
                break;
        }
    }
};

// dequantize rows [m0, m0+mr) of W into a [mr_alloc][Kp] BF16 panel; rows >= M are zero
void dequantize_panel(const dequantizer & dq, const char * w, size_t w_stride, int64_t M,
                      int64_t m0, int64_t mr, ggml_bf16_t * panel, float * tmp) {
    if (dq.ileave > 1) {
        // row-interleaved repack: to_float() of a group of R rows yields the R rows one after another
        const int64_t R = dq.ileave;
        for (int64_t i = 0; i < mr; i += R) {
            if (m0 + i < M) {
                if (dq.fast) dq.fast((const uint8_t *)(w + (m0 + i)*w_stride), dq.K, tmp);
                else         dq.to_float(w + (m0 + i)*w_stride, tmp, R*dq.K);
                for (int64_t r = 0; r < R; ++r) f32_to_bf16_row(tmp + r*dq.K, dq.K, dq.Kp, panel + (i + r)*dq.Kp);
            } else {
                memset(panel + i*dq.Kp, 0, R*dq.Kp*sizeof(ggml_bf16_t));
            }
        }
        return;
    }
    for (int64_t i = 0; i < mr; ++i) {
        ggml_bf16_t * dst = panel + i*dq.Kp;
        if (m0 + i < M) {
            dq.row(w + (m0 + i)*w_stride, dst, tmp);
        } else {
            memset(dst, 0, dq.Kp*sizeof(ggml_bf16_t));
        }
    }
}

// store a 16x16 C block (rows = weight rows m0.., cols = tokens t0..) transposed into dst
template <typename Combine>
inline void store_block(const float * c, int64_t m0, int64_t M, int64_t t0, int64_t T,
                        float * dst, size_t dst_stride, float * const * dst_rows, Combine && combine) {
    __m512i r[16];
    for (int i = 0; i < 16; ++i) r[i] = _mm512_load_si512((const __m512i *)(c + 16*i));
    transpose16x16(r);
    const __mmask16 mask = tail_mask(M - m0);
    const int64_t nt = std::min<int64_t>(16, T - t0);
    for (int64_t j = 0; j < nt; ++j) {
        float * row = dst_rows ? dst_rows[t0 + j] : (float *)((char *)dst + (t0 + j)*dst_stride);
        _mm512_mask_storeu_ps(row + m0, mask, combine((__m512)r[j], j));
    }
}

struct work_split {
    int64_t mr;          // rows per panel (multiple of 32)
    int64_t n_panels;
    int64_t tchunk;      // tokens per work item (multiple of 32)
    int64_t n_tchunks;
    int64_t n_items() const { return n_panels*n_tchunks; }
};

work_split plan_split(int64_t M, int64_t T, int64_t Kp, int nth, int n_mats) {
    work_split s;
    // keep the BF16 panel(s) within ~1 MiB so they stay in L2
    int64_t mr = (int64_t(1) << 20) / (Kp*2*n_mats);
    mr = std::max<int64_t>(32, std::min<int64_t>(256, mr/32*32));
    while (mr > 32 && (M + mr - 1)/mr < 2*nth) mr -= 32;
    s.mr = mr;
    s.n_panels = (M + mr - 1)/mr;
    const int64_t tp = round_up(T, 32);
    s.n_tchunks = 1;
    if (s.n_panels < nth) {
        s.n_tchunks = std::min<int64_t>(tp/32, (nth + s.n_panels - 1)/s.n_panels);
    }
    s.tchunk = round_up((tp + s.n_tchunks - 1)/s.n_tchunks, 32);
    s.n_tchunks = (tp + s.tchunk - 1)/s.tchunk;
    return s;
}

inline const ggml_bf16_t * b_tile(const ggml_bf16_t * packed, int64_t Kp, int64_t g, int64_t k) {
    // group g, K offset k (multiple of 32): k-pair row k/2, each k-pair row is 16 tokens x 2 BF16
    return packed + (g*(Kp/2) + k/2)*32;
}

} // namespace

extern "C" void quill_amx_pack_group(const void * x, size_t x_stride, const int32_t * row_ids, int64_t K, int64_t T,
                                     void * packed_v, int64_t g) {
    const int64_t Kp = round_up(K, kTileK);
    ggml_bf16_t * packed = (ggml_bf16_t *)(((uintptr_t)packed_v + 63) & ~(uintptr_t)63);
    const float * rows[16];
    for (int j = 0; j < 16; ++j) {
        const int64_t t = g*kGroup + j;
        rows[j] = t < T ? (const float *)((const char *)x + (row_ids ? row_ids[t] : t)*x_stride) : nullptr;
    }
    for (int64_t k = 0; k < Kp; k += 32) {
        __m512i r[16];
        const __mmask16 mlo = tail_mask(K - k), mhi = tail_mask(K - k - 16);
        for (int j = 0; j < 16; ++j) {
            if (rows[j]) {
                const __m512 lo = _mm512_maskz_loadu_ps(mlo, rows[j] + std::min(k, K));
                const __m512 hi = _mm512_maskz_loadu_ps(mhi, rows[j] + std::min(k + 16, K));
                r[j] = (__m512i)_mm512_cvtne2ps_pbh(hi, lo);   // dword p = (x[k+2p], x[k+2p+1])
            } else {
                r[j] = _mm512_setzero_si512();
            }
        }
        transpose16x16(r);                                    // r[p] = k-pair p of the 16 tokens
        ggml_bf16_t * out = packed + (g*(Kp/2) + k/2)*32;
        for (int p = 0; p < 16; ++p) _mm512_store_si512((__m512i *)(out + 32*p), r[p]);
    }
}

extern "C" void quill_amx_pack(const void * x, size_t x_stride, const int32_t * row_ids, int64_t K, int64_t T,
                               void * packed_v, int ith, int nth) {
    const int64_t n_groups = round_up(T, 32)/kGroup;
    for (int64_t g = ith; g < n_groups; g += nth) {
        quill_amx_pack_group(x, x_stride, row_ids, K, T, packed_v, g);
    }
}

extern "C" void quill_amx_gemm(enum ggml_type type, const void * w, size_t w_stride,
                               int64_t K, int64_t M, int64_t T, const void * packed_v,
                               float * dst, size_t dst_stride, float * const * dst_rows,
                               int ith, int nth) {
    const int64_t Kp = round_up(K, kTileK);
    const ggml_bf16_t * packed = (const ggml_bf16_t *)(((uintptr_t)packed_v + 63) & ~(uintptr_t)63);
    const work_split s = plan_split(M, T, Kp, nth, 1);
    if (ith >= s.n_items()) return;

    ensure_tile_config();
    thread_local aligned_buffer<ggml_bf16_t> panel_buf;
    thread_local aligned_buffer<float>       tmp_buf;
    alignas(64) float c[4][256];

    const dequantizer dq{type, ggml_internal_get_type_traits(type).to_float, K, Kp, fast_dequantizer(type, K), quill_amx_interleave(type)};
    ggml_bf16_t * panel = panel_buf.get(s.mr*Kp);
    float *       tmp   = tmp_buf.get(dq.ileave*K + 32);
    const int64_t a_stride = Kp*sizeof(ggml_bf16_t);
    struct { __m512 operator()(__m512 v, int64_t) const { return v; } } identity;

    int64_t loaded_panel = -1;
    for (int64_t item = ith; item < s.n_items(); item += nth) {
        const int64_t ip = item / s.n_tchunks;
        const int64_t it = item % s.n_tchunks;
        const int64_t m0 = ip*s.mr;
        const int64_t mr = std::min<int64_t>(s.mr, round_up(M - m0, 32));
        if (ip != loaded_panel) {
            dequantize_panel(dq, (const char *)w, w_stride, M, m0, mr, panel, tmp);
            loaded_panel = ip;
        }
        const int64_t t_begin = it*s.tchunk;
        const int64_t t_end   = std::min<int64_t>(round_up(T, 32), t_begin + s.tchunk);
        for (int64_t t0 = t_begin; t0 < t_end; t0 += 32) {
            const int64_t g0 = t0/kGroup;
            for (int64_t r0 = 0; r0 < mr; r0 += 32) {
                const ggml_bf16_t * a0 = panel + r0*Kp;
                const ggml_bf16_t * a1 = a0 + 16*Kp;
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                for (int64_t k = 0; k < Kp; k += kTileK) {
                    _tile_loadd(4, a0 + k, a_stride);
                    _tile_loadd(5, a1 + k, a_stride);
                    _tile_loadd(6, b_tile(packed, Kp, g0,     k), 64);
                    _tile_loadd(7, b_tile(packed, Kp, g0 + 1, k), 64);
                    _tile_dpbf16ps(0, 4, 6);
                    _tile_dpbf16ps(1, 4, 7);
                    _tile_dpbf16ps(2, 5, 6);
                    _tile_dpbf16ps(3, 5, 7);
                }
                _tile_stored(0, c[0], 64);
                _tile_stored(1, c[1], 64);
                _tile_stored(2, c[2], 64);
                _tile_stored(3, c[3], 64);
                const int64_t mrow = m0 + r0;
                if (mrow < M) {
                    if (t0      < T) store_block(c[0], mrow, M, t0,      T, dst, dst_stride, dst_rows, identity);
                    if (t0 + 16 < T) store_block(c[1], mrow, M, t0 + 16, T, dst, dst_stride, dst_rows, identity);
                }
                if (mrow + 16 < M) {
                    if (t0      < T) store_block(c[2], mrow + 16, M, t0,      T, dst, dst_stride, dst_rows, identity);
                    if (t0 + 16 < T) store_block(c[3], mrow + 16, M, t0 + 16, T, dst, dst_stride, dst_rows, identity);
                }
            }
        }
    }
}

namespace {

inline __m512 v_expf(__m512 x) {
    // exp(x) = 2^(x*log2e), Cody-Waite style range reduction + degree-5 polynomial
    const __m512 log2e = _mm512_set1_ps(1.44269504089f);
    x = _mm512_max_ps(_mm512_min_ps(x, _mm512_set1_ps(88.0f)), _mm512_set1_ps(-88.0f));
    __m512 t = _mm512_mul_ps(x, log2e);
    __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 f = _mm512_sub_ps(t, n);                       // f in [-0.5, 0.5]
    __m512 p = _mm512_set1_ps(1.3333558146e-3f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.6181291076e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.5504108665e-2f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.4022650696e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.9314718056e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}

inline __m512 v_silu(__m512 x) {
    return _mm512_div_ps(x, _mm512_add_ps(_mm512_set1_ps(1.0f), v_expf(_mm512_sub_ps(_mm512_setzero_ps(), x))));
}

inline __m512 v_gelu(__m512 x) {
    // ggml's tanh approximation; tanh(y) = 1 - 2/(exp(2y)+1)
    const __m512 c0 = _mm512_set1_ps(0.7978845608f), c1 = _mm512_set1_ps(0.044715f);
    __m512 x3 = _mm512_mul_ps(_mm512_mul_ps(x, x), x);
    __m512 y  = _mm512_mul_ps(c0, _mm512_fmadd_ps(c1, x3, x));
    __m512 e  = v_expf(_mm512_add_ps(y, y));
    __m512 th = _mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_div_ps(_mm512_set1_ps(2.0f), _mm512_add_ps(e, _mm512_set1_ps(1.0f))));
    return _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5f), x), _mm512_add_ps(_mm512_set1_ps(1.0f), th));
}

} // namespace

extern "C" bool quill_amx_gemm_up_gate(enum ggml_type type, const void * w_up, const void * w_gate, size_t w_stride,
                                       int64_t K, int64_t M, int64_t T, const void * packed_v,
                                       float * dst, size_t dst_stride, float * const * dst_rows,
                                       int unary_op, float limit, int ith, int nth) {
    const ggml_unary_op op = (ggml_unary_op)unary_op;
    if (op != GGML_UNARY_OP_SILU && op != GGML_UNARY_OP_RELU && op != GGML_UNARY_OP_GELU) return false;

    const int64_t Kp = round_up(K, kTileK);
    const ggml_bf16_t * packed = (const ggml_bf16_t *)(((uintptr_t)packed_v + 63) & ~(uintptr_t)63);
    const work_split s = plan_split(M, T, Kp, nth, 2);
    if (ith >= s.n_items()) return true;

    ensure_tile_config();
    thread_local aligned_buffer<ggml_bf16_t> panel_buf;
    thread_local aligned_buffer<float>       tmp_buf;
    alignas(64) float c[4][256];   // up(t0), up(t0+16), gate(t0), gate(t0+16)

    const dequantizer dq{type, ggml_internal_get_type_traits(type).to_float, K, Kp, fast_dequantizer(type, K), quill_amx_interleave(type)};
    ggml_bf16_t * panel_up   = panel_buf.get(2*s.mr*Kp);
    ggml_bf16_t * panel_gate = panel_up + s.mr*Kp;
    float *       tmp        = tmp_buf.get(dq.ileave*K + 32);
    const int64_t a_stride   = Kp*sizeof(ggml_bf16_t);
    const bool    clamp      = limit > 1e-6f;
    const __m512  vlim       = _mm512_set1_ps(limit);

    int64_t loaded_panel = -1;
    for (int64_t item = ith; item < s.n_items(); item += nth) {
        const int64_t ip = item / s.n_tchunks;
        const int64_t it = item % s.n_tchunks;
        const int64_t m0 = ip*s.mr;
        const int64_t mr = std::min<int64_t>(s.mr, round_up(M - m0, 32));
        if (ip != loaded_panel) {
            dequantize_panel(dq, (const char *)w_up,   w_stride, M, m0, mr, panel_up,   tmp);
            dequantize_panel(dq, (const char *)w_gate, w_stride, M, m0, mr, panel_gate, tmp);
            loaded_panel = ip;
        }
        const int64_t t_begin = it*s.tchunk;
        const int64_t t_end   = std::min<int64_t>(round_up(T, 32), t_begin + s.tchunk);
        for (int64_t t0 = t_begin; t0 < t_end; t0 += 32) {
            const int64_t g0 = t0/kGroup;
            for (int64_t r0 = 0; r0 < mr && m0 + r0 < M; r0 += 16) {
                const ggml_bf16_t * au = panel_up   + r0*Kp;
                const ggml_bf16_t * ag = panel_gate + r0*Kp;
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                for (int64_t k = 0; k < Kp; k += kTileK) {
                    _tile_loadd(4, au + k, a_stride);
                    _tile_loadd(5, ag + k, a_stride);
                    _tile_loadd(6, b_tile(packed, Kp, g0,     k), 64);
                    _tile_loadd(7, b_tile(packed, Kp, g0 + 1, k), 64);
                    _tile_dpbf16ps(0, 4, 6);
                    _tile_dpbf16ps(1, 4, 7);
                    _tile_dpbf16ps(2, 5, 6);
                    _tile_dpbf16ps(3, 5, 7);
                }
                _tile_stored(0, c[0], 64);
                _tile_stored(1, c[1], 64);
                _tile_stored(2, c[2], 64);
                _tile_stored(3, c[3], 64);
                for (int h = 0; h < 2; ++h) {
                    const int64_t tt = t0 + 16*h;
                    if (tt >= T) break;
                    // transpose the gate block once, then combine while storing the up block
                    __m512i g[16];
                    for (int i = 0; i < 16; ++i) g[i] = _mm512_load_si512((const __m512i *)(c[2 + h] + 16*i));
                    transpose16x16(g);
                    auto combine = [&](__m512 up, int64_t j) {
                        __m512 x = (__m512)g[j];
                        __m512 a = op == GGML_UNARY_OP_SILU ? v_silu(x)
                                 : op == GGML_UNARY_OP_GELU ? v_gelu(x)
                                 : _mm512_max_ps(x, _mm512_setzero_ps());
                        if (clamp) {
                            a  = _mm512_min_ps(a, vlim);
                            up = _mm512_max_ps(_mm512_sub_ps(_mm512_setzero_ps(), vlim), _mm512_min_ps(up, vlim));
                        }
                        return _mm512_mul_ps(up, a);
                    };
                    store_block(c[h], m0 + r0, M, tt, T, dst, dst_stride, dst_rows, combine);
                }
            }
        }
    }
    return true;
}


// =============================================================================================
// Flash attention (prompt processing): S = Q.K^T and O += P.V on AMX tiles, online softmax in
// AVX-512. A work unit is (sequence, KV head, query range); the query heads that share a KV head
// (GQA) reuse each packed K/V chunk.
// =============================================================================================
namespace {

constexpr int64_t kFaChunk = 256;   // keys per K/V chunk

// one K/V row (D values) as floats
inline void kv_row_to_f32(ggml_type type, const char * src, int64_t D, float * y) {
    switch (type) {
        case GGML_TYPE_F32: memcpy(y, src, D*sizeof(float)); break;
        case GGML_TYPE_F16:
            for (int64_t i = 0; i < D; i += 16) _mm512_storeu_ps(y + i, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(src + 2*i))));
            break;
        case GGML_TYPE_BF16:
            for (int64_t i = 0; i < D; i += 16) {
                const __m256i b = _mm256_loadu_si256((const __m256i *)(src + 2*i));
                _mm512_storeu_ps(y + i, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(b), 16)));
            }
            break;
        case GGML_TYPE_Q8_0: dq_q8_0((const uint8_t *)src, D, y); break;
        default: GGML_ABORT("quill AMX FA: unsupported K/V type");
    }
}

inline __m512 v_tanh(__m512 x) {
    const __m512 e = v_expf(_mm512_add_ps(x, x));
    return _mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_div_ps(_mm512_set1_ps(2.0f), _mm512_add_ps(e, _mm512_set1_ps(1.0f))));
}

struct fa_args {
    const char * q; const char * k; const char * v; const char * mask;
    size_t nbq1, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3, nbm1, nbm2, nbm3;
    int64_t D, N, n_kv, H, H_kv, B, Bk, Bv, mask_ne2, mask_ne3;
    ggml_type kt, vt;
    float scale, softcap;
    char * dst; size_t nb1; int64_t ne1, ne2;
};

void fa_unit(const fa_args & a, int64_t b, int64_t hk, int64_t q0, int64_t q1) {
    const int64_t D   = a.D;
    const int64_t rk  = a.H/a.H_kv;
    const int64_t nq  = q1 - q0;
    const int64_t nqp = round_up(nq, 32);
    const int64_t bk  = b % a.Bk, bv = b % a.Bv;

    thread_local aligned_buffer<ggml_bf16_t> qbuf_b, kbuf_b, vbuf_b, pbuf_b;
    thread_local aligned_buffer<float>       obuf_b, sbuf_b, ml_b, row_b;
    ggml_bf16_t * qbuf = qbuf_b.get(rk*nqp*D);               // [rk][nqp][D]   (scaled Q)
    ggml_bf16_t * kbuf = kbuf_b.get(kFaChunk*D);              // VNNI [16-key groups][D/2][16][2]
    ggml_bf16_t * vbuf = vbuf_b.get(kFaChunk*D);              // VNNI [D/16][keys/2][16][2]
    ggml_bf16_t * pbuf = pbuf_b.get(32*kFaChunk);             // [32][kFaChunk]
    float * obuf = obuf_b.get(rk*nqp*D);                      // [rk][nqp][D]
    float * sbuf = sbuf_b.get(32*kFaChunk);                   // [32][kFaChunk]
    float * mrow = ml_b.get(2*rk*nqp);                        // running max, then running sum
    float * lrow = mrow + rk*nqp;
    float * row  = row_b.get(D + 32);

    // scaled Q -> BF16
    for (int64_t j = 0; j < rk; ++j) {
        const int64_t h = hk*rk + j;
        for (int64_t i = 0; i < nqp; ++i) {
            ggml_bf16_t * dstq = qbuf + (j*nqp + i)*D;
            if (i < nq) {
                const float * src = (const float *)(a.q + (q0 + i)*a.nbq1 + h*a.nbq2 + b*a.nbq3);
                const __m512 sc = _mm512_set1_ps(a.scale);
                for (int64_t d = 0; d < D; d += 16) _mm512_storeu_ps(row + d, _mm512_mul_ps(_mm512_loadu_ps(src + d), sc));
                f32_to_bf16_row(row, D, D, dstq);
            } else {
                memset(dstq, 0, D*sizeof(ggml_bf16_t));
            }
        }
    }
    memset(obuf, 0, rk*nqp*D*sizeof(float));
    for (int64_t i = 0; i < rk*nqp; ++i) { mrow[i] = -INFINITY; lrow[i] = 0.0f; }

    const char * kbase = a.k + hk*a.nbk2 + bk*a.nbk3;
    const char * vbase = a.v + hk*a.nbv2 + bv*a.nbv3;
    const __m512 vninf = _mm512_set1_ps(-INFINITY);

    for (int64_t c0 = 0; c0 < a.n_kv; c0 += kFaChunk) {
        const int64_t kc  = std::min<int64_t>(kFaChunk, a.n_kv - c0);
        const int64_t kcp = round_up(kc, 32);

        // pack K: keys play the role of tokens, D the role of K
        for (int64_t g = 0; g < kcp/16; ++g) {
            __m512i r[16];
            alignas(64) float rows[16][16];
            for (int64_t d0 = 0; d0 < D; d0 += 32) {
                for (int jj = 0; jj < 16; ++jj) {
                    const int64_t key = g*16 + jj;
                    if (key < kc) {
                        if (d0 == 0) {
                            // dequantize the full row once into the per-key scratch of this group
                            kv_row_to_f32(a.kt, kbase + (c0 + key)*a.nbk1, D, sbuf + jj*D);
                        }
                        const float * x = sbuf + jj*D + d0;
                        r[jj] = (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + 16), _mm512_loadu_ps(x));
                    } else {
                        r[jj] = _mm512_setzero_si512();
                    }
                }
                transpose16x16(r);
                ggml_bf16_t * out = kbuf + (g*(D/2) + d0/2)*32;
                for (int p = 0; p < 16; ++p) _mm512_store_si512((__m512i *)(out + 32*p), r[p]);
            }
            (void) rows;
        }
        // pack V: B tile rows are key pairs, 16 head dims x (key 2p, key 2p+1)
        for (int64_t p = 0; p < kcp/2; ++p) {
            float * v0 = row;
            float * v1 = sbuf;   // scratch (S is not live here)
            if (2*p     < kc) kv_row_to_f32(a.vt, vbase + (c0 + 2*p    )*a.nbv1, D, v0); else memset(v0, 0, D*sizeof(float));
            if (2*p + 1 < kc) kv_row_to_f32(a.vt, vbase + (c0 + 2*p + 1)*a.nbv1, D, v1); else memset(v1, 0, D*sizeof(float));
            for (int64_t db = 0; db < D/16; ++db) {
                const __m256i b0 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(v0 + 16*db));
                const __m256i b1 = (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(v1 + 16*db));
                // interleave 16-bit lanes: (v0[d], v1[d]) for d = 0..15
                const __m512i w0 = _mm512_cvtepu16_epi32(b0);
                const __m512i w1 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(b1), 16);
                _mm512_store_si512((__m512i *)(vbuf + (db*(kFaChunk/2) + p)*32), _mm512_or_si512(w0, w1));
            }
        }

        for (int64_t j = 0; j < rk; ++j) {
            const int64_t h = hk*rk + j;
            for (int64_t i0 = 0; i0 < nqp; i0 += 32) {
                const int64_t nr = std::min<int64_t>(32, nq - i0);
                if (nr <= 0) break;
                const ggml_fp16_t * mrows[32];
                for (int r = 0; r < 32; ++r) {
                    mrows[r] = a.mask && r < nr
                        ? (const ggml_fp16_t *)(a.mask + (q0 + i0 + r)*a.nbm1 + (h % a.mask_ne2)*a.nbm2 + (b % a.mask_ne3)*a.nbm3) + c0
                        : nullptr;
                }
                // which 32-key blocks are fully masked for these 32 queries (causal attention)
                bool active[kFaChunk/32];
                bool any = false;
                for (int64_t kb = 0; kb < kcp; kb += 32) {
                    bool act = !a.mask;
                    if (a.mask) {
                        const int64_t nk = std::min<int64_t>(32, kc - kb);
                        for (int r = 0; r < nr && !act; ++r) {
                            for (int64_t t = 0; t < nk; ++t) {
                                if ((mrows[r][kb + t] & 0x7fff) != 0x7c00) { act = true; break; }   // not +-inf
                            }
                        }
                    }
                    active[kb/32] = act;
                    any |= act;
                }
                if (!any) continue;

                // S = Q.K^T
                const ggml_bf16_t * qa = qbuf + (j*nqp + i0)*D;
                for (int64_t kb = 0; kb < kcp; kb += 32) {
                    if (!active[kb/32]) continue;
                    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                    for (int64_t d0 = 0; d0 < D; d0 += 32) {
                        _tile_loadd(4, qa + d0,        D*sizeof(ggml_bf16_t));
                        _tile_loadd(5, qa + 16*D + d0, D*sizeof(ggml_bf16_t));
                        _tile_loadd(6, kbuf + ((kb/16    )*(D/2) + d0/2)*32, 64);
                        _tile_loadd(7, kbuf + ((kb/16 + 1)*(D/2) + d0/2)*32, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_dpbf16ps(3, 5, 7);
                    }
                    _tile_stored(0, sbuf + kb,                 kFaChunk*sizeof(float));
                    _tile_stored(1, sbuf + kb + 16,            kFaChunk*sizeof(float));
                    _tile_stored(2, sbuf + 16*kFaChunk + kb,      kFaChunk*sizeof(float));
                    _tile_stored(3, sbuf + 16*kFaChunk + kb + 16, kFaChunk*sizeof(float));
                }

                // online softmax over this chunk
                for (int r = 0; r < 32; ++r) {
                    float * srow = sbuf + r*kFaChunk;
                    ggml_bf16_t * prow = pbuf + r*kFaChunk;
                    if (r >= nr) {
                        for (int64_t kb = 0; kb < kcp; kb += 32) if (active[kb/32]) memset(prow + kb, 0, 64);
                        continue;
                    }
                    // logits: softcap, mask, padding
                    __m512 vmax = vninf;
                    for (int64_t kb = 0; kb < kcp; kb += 16) {
                        if (!active[kb/32]) continue;
                        __m512 sv = _mm512_loadu_ps(srow + kb);
                        if (a.softcap != 0.0f) sv = _mm512_mul_ps(_mm512_set1_ps(a.softcap), v_tanh(sv));
                        const __mmask16 valid = tail_mask(kc - kb);
                        if (mrows[r]) {
                            const __m256i mh = _mm256_maskz_loadu_epi16(valid, mrows[r] + kb);
                            sv = _mm512_add_ps(sv, _mm512_cvtph_ps(mh));
                        }
                        sv = _mm512_mask_blend_ps(valid, vninf, sv);
                        _mm512_storeu_ps(srow + kb, sv);
                        vmax = _mm512_max_ps(vmax, sv);
                    }
                    float * m = mrow + j*nqp + i0 + r;
                    float * l = lrow + j*nqp + i0 + r;
                    const float m_new = std::max(*m, _mm512_reduce_max_ps(vmax));
                    if (m_new == -INFINITY) {
                        for (int64_t kb = 0; kb < kcp; kb += 32) if (active[kb/32]) memset(prow + kb, 0, 64);
                        continue;
                    }
                    const float corr = *m == -INFINITY ? 0.0f : expf(*m - m_new);
                    const __m512 vm = _mm512_set1_ps(m_new);
                    __m512 vsum = _mm512_setzero_ps();
                    for (int64_t kb = 0; kb < kcp; kb += 32) {
                        if (!active[kb/32]) continue;
                        const __m512 e0 = v_expf(_mm512_sub_ps(_mm512_loadu_ps(srow + kb),      vm));
                        const __m512 e1 = v_expf(_mm512_sub_ps(_mm512_loadu_ps(srow + kb + 16), vm));
                        vsum = _mm512_add_ps(vsum, _mm512_add_ps(e0, e1));
                        _mm512_storeu_si512((__m512i *)(prow + kb), (__m512i)_mm512_cvtne2ps_pbh(e1, e0));
                    }
                    *l = *l*corr + _mm512_reduce_add_ps(vsum);
                    *m = m_new;
                    if (corr != 1.0f) {
                        float * o = obuf + (j*nqp + i0 + r)*D;
                        const __m512 vc = _mm512_set1_ps(corr);
                        for (int64_t d = 0; d < D; d += 16) _mm512_storeu_ps(o + d, _mm512_mul_ps(_mm512_loadu_ps(o + d), vc));
                    }
                }

                // O += P.V
                float * o = obuf + (j*nqp + i0)*D;
                for (int64_t d0 = 0; d0 < D; d0 += 32) {
                    _tile_loadd(0, o + d0,             D*sizeof(float));
                    _tile_loadd(1, o + d0 + 16,        D*sizeof(float));
                    _tile_loadd(2, o + 16*D + d0,      D*sizeof(float));
                    _tile_loadd(3, o + 16*D + d0 + 16, D*sizeof(float));
                    for (int64_t kb = 0; kb < kcp; kb += 32) {
                        if (!active[kb/32]) continue;
                        _tile_loadd(4, pbuf + kb,                 kFaChunk*sizeof(ggml_bf16_t));
                        _tile_loadd(5, pbuf + 16*kFaChunk + kb,   kFaChunk*sizeof(ggml_bf16_t));
                        _tile_loadd(6, vbuf + ((d0/16    )*(kFaChunk/2) + kb/2)*32, 64);
                        _tile_loadd(7, vbuf + ((d0/16 + 1)*(kFaChunk/2) + kb/2)*32, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_dpbf16ps(3, 5, 7);
                    }
                    _tile_stored(0, o + d0,             D*sizeof(float));
                    _tile_stored(1, o + d0 + 16,        D*sizeof(float));
                    _tile_stored(2, o + 16*D + d0,      D*sizeof(float));
                    _tile_stored(3, o + 16*D + d0 + 16, D*sizeof(float));
                }
            }
        }
    }

    // normalize and write: dst row for (query i, head h) is at ((b*ne2 + i)*ne1 + h)*nb1
    for (int64_t j = 0; j < rk; ++j) {
        const int64_t h = hk*rk + j;
        for (int64_t i = 0; i < nq; ++i) {
            const float l = lrow[j*nqp + i];
            const __m512 vinv = _mm512_set1_ps(l > 0.0f ? 1.0f/l : 0.0f);
            const float * o = obuf + (j*nqp + i)*D;
            float * y = (float *)(a.dst + ((b*a.ne2 + q0 + i)*a.ne1 + h)*a.nb1);
            for (int64_t d = 0; d < D; d += 16) _mm512_storeu_ps(y + d, _mm512_mul_ps(_mm512_loadu_ps(o + d), vinv));
        }
    }
}

} // namespace

extern "C" void quill_amx_flash_attn(struct ggml_tensor * dst, int ith, int nth) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    fa_args a;
    a.q = (const char *)q->data; a.k = (const char *)k->data; a.v = (const char *)v->data;
    a.mask = mask ? (const char *)mask->data : nullptr;
    a.nbq1 = q->nb[1]; a.nbq2 = q->nb[2]; a.nbq3 = q->nb[3];
    a.nbk1 = k->nb[1]; a.nbk2 = k->nb[2]; a.nbk3 = k->nb[3];
    a.nbv1 = v->nb[1]; a.nbv2 = v->nb[2]; a.nbv3 = v->nb[3];
    a.nbm1 = mask ? mask->nb[1] : 0; a.nbm2 = mask ? mask->nb[2] : 0; a.nbm3 = mask ? mask->nb[3] : 0;
    a.mask_ne2 = mask ? mask->ne[2] : 1; a.mask_ne3 = mask ? mask->ne[3] : 1;
    a.D = q->ne[0]; a.N = q->ne[1]; a.n_kv = k->ne[1]; a.H = q->ne[2]; a.H_kv = k->ne[2];
    a.B = q->ne[3]; a.Bk = k->ne[3]; a.Bv = v->ne[3];
    a.kt = k->type; a.vt = v->type;
    memcpy(&a.scale,   (const float *)dst->op_params + 0, sizeof(float));
    memcpy(&a.softcap, (const float *)dst->op_params + 2, sizeof(float));
    if (a.softcap != 0.0f) a.scale /= a.softcap;
    a.dst = (char *)dst->data; a.nb1 = dst->nb[1]; a.ne1 = dst->ne[1]; a.ne2 = dst->ne[2];

    ensure_tile_config();
    const int64_t n_groups = a.B*a.H_kv;
    const int64_t nqb      = (a.N + 31)/32;
    const int64_t qsplit   = std::max<int64_t>(1, std::min<int64_t>(nqb, (nth + n_groups - 1)/n_groups));
    const int64_t qstep    = ((nqb + qsplit - 1)/qsplit)*32;
    const int64_t n_units  = n_groups*qsplit;
    for (int64_t u = ith; u < n_units; u += nth) {
        const int64_t g  = u / qsplit, qs = u % qsplit;
        const int64_t b  = g / a.H_kv, hk = g % a.H_kv;
        const int64_t q0 = qs*qstep, q1 = std::min<int64_t>(a.N, q0 + qstep);
        if (q0 < q1) fa_unit(a, b, hk, q0, q1);
    }
}

#else // !QUILL_AMX_BUILD

extern "C" void   quill_amx_pack(const void *, size_t, const int32_t *, int64_t, int64_t, void *, int, int) {}
extern "C" void   quill_amx_pack_group(const void *, size_t, const int32_t *, int64_t, int64_t, void *, int64_t) {}
extern "C" void   quill_amx_gemm(enum ggml_type, const void *, size_t, int64_t, int64_t, int64_t, const void *,
                                 float *, size_t, float * const *, int, int) {}
extern "C" void   quill_amx_flash_attn(struct ggml_tensor *, int, int) {}
extern "C" bool   quill_amx_gemm_up_gate(enum ggml_type, const void *, const void *, size_t, int64_t, int64_t, int64_t,
                                         const void *, float *, size_t, float * const *, int, float, int, int) { return false; }

#endif
