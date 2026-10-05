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

#pragma GCC target("avx512f,avx512bw,avx512vl,avx512dq,avx512bf16,f16c,fma,amx-tile,amx-bf16")
#include <immintrin.h>
#include <cpuid.h>
#include <unistd.h>
#include <sys/syscall.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

namespace {

constexpr int64_t kTileK   = 32;   // BF16 elements per tile row (64 bytes)
constexpr int64_t kGroup   = 16;   // tokens per packed group / rows per tile

inline int64_t round_up(int64_t x, int64_t n) { return (x + n - 1)/n*n; }

bool cpu_has_amx_bf16() {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    const bool amx_bf16 = edx & (1u << 22);
    const bool amx_tile = edx & (1u << 24);
    const bool avx512f  = ebx & (1u << 16);
    const bool avx512bw = ebx & (1u << 30);
    if (!(amx_bf16 && amx_tile && avx512f && avx512bw)) return false;
    if (!__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) return false;
    const bool avx512_bf16 = eax & (1u << 5);
    return avx512_bf16;
}

bool request_tile_permission() {
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
}

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
            ptr  = (T *) aligned_alloc(64, round_up(n*sizeof(T), 64));
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
        default:               return nullptr;
    }
}

struct dequantizer {
    ggml_type       type;
    ggml_to_float_t to_float;
    int64_t         K, Kp;
    fast_dq_t       fast = nullptr;

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

static int g_amx_state = -1;   // -1 unknown, 0 off, 1 on

extern "C" bool quill_amx_available(void) {
    if (g_amx_state < 0) {
        const char * env = getenv("QUILL_AMX");
        bool on = !(env && (strcmp(env, "0") == 0 || strcmp(env, "off") == 0));
        on = on && cpu_has_amx_bf16() && request_tile_permission();
        g_amx_state = on ? 1 : 0;
    }
    return g_amx_state == 1;
}

extern "C" int quill_amx_min_batch(void) {
    static int n = [] {
        const char * env = getenv("QUILL_AMX_MIN_BATCH");
        int v = env ? atoi(env) : 32;
        return v > 0 ? v : 32;
    }();
    return n;
}

extern "C" bool quill_amx_type_supported(enum ggml_type type) {
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) return true;
    if (!ggml_is_quantized(type)) return false;
    const auto traits = ggml_internal_get_type_traits(type);
    if (!traits.to_float) return false;
    // row-interleaved repacks (*_R4, *_R8, *_R16) do not dequantize row by row
    const char * name = ggml_type_name(type);
    if (!name || strstr(name, "_r") || strstr(name, "_R")) return false;
    // row-wise meta data (e.g. IQ4_KS) is fine: we always dequantize whole rows
    return true;
}

extern "C" size_t quill_amx_packed_size(int64_t K, int64_t T) {
    return (size_t)round_up(T, 32) * (size_t)round_up(K, kTileK) * sizeof(ggml_bf16_t) + 64;
}

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

    const dequantizer dq{type, ggml_internal_get_type_traits(type).to_float, K, Kp, fast_dequantizer(type, K)};
    ggml_bf16_t * panel = panel_buf.get(s.mr*Kp);
    float *       tmp   = tmp_buf.get(K + 32);
    const int64_t a_stride = Kp*sizeof(ggml_bf16_t);
    auto identity = [](__m512 v, int64_t) { return v; };

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

    const dequantizer dq{type, ggml_internal_get_type_traits(type).to_float, K, Kp, fast_dequantizer(type, K)};
    ggml_bf16_t * panel_up   = panel_buf.get(2*s.mr*Kp);
    ggml_bf16_t * panel_gate = panel_up + s.mr*Kp;
    float *       tmp        = tmp_buf.get(K + 32);
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

#else // !QUILL_AMX_BUILD

extern "C" bool   quill_amx_available(void) { return false; }
extern "C" bool   quill_amx_type_supported(enum ggml_type) { return false; }
extern "C" int    quill_amx_min_batch(void) { return 1 << 30; }
extern "C" size_t quill_amx_packed_size(int64_t, int64_t) { return 0; }
extern "C" void   quill_amx_pack(const void *, size_t, const int32_t *, int64_t, int64_t, void *, int, int) {}
extern "C" void   quill_amx_pack_group(const void *, size_t, const int32_t *, int64_t, int64_t, void *, int64_t) {}
extern "C" void   quill_amx_gemm(enum ggml_type, const void *, size_t, int64_t, int64_t, int64_t, const void *,
                                 float *, size_t, float * const *, int, int) {}
extern "C" bool   quill_amx_gemm_up_gate(enum ggml_type, const void *, const void *, size_t, int64_t, int64_t, int64_t,
                                         const void *, float *, size_t, float * const *, int, float, int, int) { return false; }

#endif
