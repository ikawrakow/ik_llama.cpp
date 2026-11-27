//
// IQ4_NT: 4-bit nibble trellis. Shared by the quantizer, the reference decode and the kernels.
//
// One nibble per weight. The int8 label of a weight is a function of its own nibble n0 and the three nibbles n1, n2, n3 before it
// in the stream (n1 the most recent):
//     v = T0[(n0+n2)&7] + T1[(n0+n3)&7] + T2[((n0+n1+n3)&15)>>1] + T3[((n0+n2+n3)&15)>>1],   |v| <= 124,
// with four 8-entry tables.
// Row layout for K weights (K a multiple of 128):
//     float    d                    row scale
//     uint8_t  q[K/128]             block scales: s_b = d * (q_b/255), one per 128 weights (= one nibble chunk)
//     zero bytes                    at least 2, so that the header is a multiple of 4 bytes
//     uint8_t  qs[K/2]              nibbles
// Nibble order: 64 bytes hold 128 weights; the low nibbles of bytes 0...63 are columns 0...63 of that chunk, the high nibbles
// columns 64...127. The stream runs lo(0), hi(0), lo(1), hi(1), ... through the row and the context before the first weight is
// zero, which the zero bytes in front of qs provide to a decoder that reads qs-1 and qs-2.
// Weight of column j in block b: s_b * v_j.
// The kNx most important columns of a tensor (highest imatrix value) are coded with zero weight in the search and are made exact
// outside the rows: llama-quantize stores their indices and per-row values (x - s_b*v) in two companion tensors, and the graph
// adds V * x[cols] to the matmul result (GGML_OP_ADD_EXACT_COLS).
//
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

namespace iqk_nt {

constexpr int kNx     = 4;
constexpr int kBlock  = 128;
constexpr int kChunk  = 128;

static const int8_t kTables[4][8] = {
    {  2,   7,  -2, -13,  -7, -31,  31,  13 },
    { 31,  -7,   2, -13,   7,  -2, -31,  13 },
    {  2,  -7, -13, -31,  31,  -2,   7,  13 },
    { 13,   7,   2,  31,  -2,  -7, -13, -31 },
};
constexpr int kTableMax = 31;

static inline int label(int n0, int n1, int n2, int n3) {
    return kTables[0][(n0 + n2) & 7] + kTables[1][(n0 + n3) & 7] + kTables[2][((n0 + n1 + n3) & 15) >> 1] + kTables[3][((n0 + n2 + n3) & 15) >> 1];
}

static inline int8_t table16(int k, int i) { return k < 2 ? kTables[k][i & 7] : kTables[k][i >> 1]; }

static inline size_t header_size(int64_t K) { return ((4 + K/kBlock + 2 + 3)/4)*4; }
static inline size_t row_size(int64_t K)    { return header_size(K) + K/2; }

static inline int stream_column(int t) {
    const int ch = t/kChunk, p = t%kChunk;
    return ch*kChunk + ((p & 1) ? kChunk/2 + p/2 : p/2);
}

static inline float block_scale(float d, int q) { return d*(float)((double)q/255); }

struct RowView {
    float d;
    const uint8_t * q;
    const uint8_t * qs;
};
static inline RowView row_view(const void * row, int64_t K) {
    const uint8_t * p = (const uint8_t *)row;
    RowView r; std::memcpy(&r.d, p, 4);
    r.q = p + 4; r.qs = p + header_size(K);
    return r;
}

static inline void decode_row(const void * row, float * y, int64_t K) {
    const RowView r = row_view(row, K);
    int n1 = 0, n2 = 0, n3 = 0;
    for (int64_t t = 0; t < K; ++t) {
        const int ch = (int)(t/kChunk), p = (int)(t%kChunk);
        const uint8_t byte = r.qs[ch*(kChunk/2) + p/2];
        const int n0 = (p & 1) ? byte >> 4 : byte & 15;
        const float s = block_scale(r.d, r.q[t/kBlock]);
        y[ch*kChunk + ((p & 1) ? kChunk/2 + p/2 : p/2)] = s*(float)label(n0, n1, n2, n3);
        n3 = n2; n2 = n1; n1 = n0;
    }
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
struct LabelsAvx512 {
    __m512i t[4], m4;
    explicit LabelsAvx512(int off) {
        for (int k = 0; k < 4; ++k) {
            int8_t tb[16];
            for (int i = 0; i < 16; ++i) tb[i] = (int8_t)(table16(k, i) + off);
            t[k] = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)tb));
        }
        m4 = _mm512_set1_epi8(0x0f);
    }
    inline __m512i sum4(__m512i i0, __m512i i1, __m512i i2, __m512i i3) const {
        return _mm512_add_epi8(_mm512_add_epi8(_mm512_shuffle_epi8(t[0], i0), _mm512_shuffle_epi8(t[1], i1)),
                               _mm512_add_epi8(_mm512_shuffle_epi8(t[2], i2), _mm512_shuffle_epi8(t[3], i3)));
    }
    inline void labels(const uint8_t * qs, __m512i & ve, __m512i & vo) const {
        const __m512i B0 = _mm512_loadu_si512((const __m512i *)qs);
        const __m512i B1 = _mm512_loadu_si512((const __m512i *)(qs - 1));
        const __m512i B2 = _mm512_loadu_si512((const __m512i *)(qs - 2));
        const __m512i E0 = _mm512_and_si512(B0, m4);
        const __m512i O0 = _mm512_and_si512(_mm512_srli_epi16(B0, 4), m4);
        const __m512i E1 = _mm512_and_si512(B1, m4);
        const __m512i O1 = _mm512_and_si512(_mm512_srli_epi16(B1, 4), m4);
        const __m512i O2 = _mm512_and_si512(_mm512_srli_epi16(B2, 4), m4);
        const __m512i a1 = _mm512_add_epi8(E0, O2);
        ve = sum4(_mm512_add_epi8(E0, E1), a1, _mm512_add_epi8(a1, O1), _mm512_add_epi8(a1, E1));
        const __m512i a2 = _mm512_add_epi8(O0, E1);
        vo = sum4(_mm512_add_epi8(O0, O1), a2, _mm512_add_epi8(a2, E0), _mm512_add_epi8(a2, O1));
    }
};
#endif

struct Labels {
    std::vector<float> nat, vit;
    float sig = 1;
    Labels() {
        nat.resize(65536); vit.resize(65536);
        for (int i = 0; i < 65536; ++i) nat[i] = (float)label(i >> 12, (i >> 8) & 15, (i >> 4) & 15, i & 15);
        for (size_t i = 0; i < 65536; ++i) {
            const size_t hi = i >> 8, y = (i >> 4) & 15, z = i & 15;
            vit[(hi << 8) | (z << 4) | y] = nat[i];
        }
        double s2 = 0; for (float v : nat) s2 += double(v)*v;
        sig = float(std::sqrt(s2/65536));
    }
};
static inline const Labels & labels() { static const Labels L; return L; }

struct Work {
    std::vector<float> N, T, xn, ww, q;
    std::vector<uint8_t> bp;
    std::vector<uint32_t> idx;
};

constexpr size_t kStates = 4096;

#ifdef __AVX512F__
static inline void transpose16(const float * in, float * out) {
    __m512 r[16], t[16];
    for (int i = 0; i < 16; ++i) r[i] = _mm512_loadu_ps(in + 16*i);
    for (int i = 0; i < 16; i += 2) { t[i] = _mm512_unpacklo_ps(r[i], r[i+1]); t[i+1] = _mm512_unpackhi_ps(r[i], r[i+1]); }
    for (int i = 0; i < 16; i += 4) {
        r[i+0] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i+0]), _mm512_castps_pd(t[i+2])));
        r[i+1] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[i+0]), _mm512_castps_pd(t[i+2])));
        r[i+2] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t[i+1]), _mm512_castps_pd(t[i+3])));
        r[i+3] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t[i+1]), _mm512_castps_pd(t[i+3])));
    }
    for (int k = 0; k < 4; ++k) {
        t[k]    = _mm512_shuffle_f32x4(r[k],   r[4+k],  0x88);
        t[k+4]  = _mm512_shuffle_f32x4(r[k],   r[4+k],  0xdd);
        t[k+8]  = _mm512_shuffle_f32x4(r[8+k], r[12+k], 0x88);
        t[k+12] = _mm512_shuffle_f32x4(r[8+k], r[12+k], 0xdd);
    }
    for (int k = 0; k < 4; ++k) {
        _mm512_storeu_ps(out + 16*(k),      _mm512_shuffle_f32x4(t[k],   t[k+8],  0x88));
        _mm512_storeu_ps(out + 16*(8+k),    _mm512_shuffle_f32x4(t[k],   t[k+8],  0xdd));
        _mm512_storeu_ps(out + 16*(4+k),    _mm512_shuffle_f32x4(t[k+4], t[k+12], 0x88));
        _mm512_storeu_ps(out + 16*(12+k),   _mm512_shuffle_f32x4(t[k+4], t[k+12], 0xdd));
    }
}

static inline double viterbi(const Labels & L, const float * xn, const float * ww, int n, uint32_t * idx, Work & wk) {
    const size_t S = kStates, P = S/16, nT = S/256;
    wk.N.resize(S); wk.T.resize(S); wk.bp.resize(size_t(n)*S);
    float * N = wk.N.data(), * T = wk.T.data();
    std::fill(N, N + S, 1e30f); N[0] = 0;
    for (size_t b = 0; b < nT; ++b) transpose16(N + 256*b, T + 256*b);
    __m512i zi[16]; for (int z = 0; z < 16; ++z) zi[z] = _mm512_set1_epi32(z);
    const float * Lv = L.vit.data();
    double offset = 0;
    for (int t = 0; t < n; ++t) {
        const __m512 xv = _mm512_set1_ps(xn[t]), wv = _mm512_set1_ps(ww[t]);
        uint8_t * bp = wk.bp.data() + size_t(t)*S;
        for (size_t p = 0; p < P; ++p) {
            const float * cT = T + (p % nT)*256, * lv = Lv + p*256;
            __m512 best = _mm512_set1_ps(3e38f); __m512i arg = _mm512_setzero_si512();
            for (int z = 0; z < 16; ++z) {
                const __m512 d = _mm512_sub_ps(xv, _mm512_loadu_ps(lv + 16*z));
                const __m512 e = _mm512_fmadd_ps(_mm512_mul_ps(d, d), wv, _mm512_loadu_ps(cT + 16*z));
                const __mmask16 m = _mm512_cmp_ps_mask(e, best, _CMP_LT_OQ);
                best = _mm512_mask_mov_ps(best, m, e);
                arg  = _mm512_mask_mov_epi32(arg, m, zi[z]);
            }
            _mm512_storeu_ps(N + 16*p, best);
            _mm_storeu_si128((__m128i *)(bp + 16*p), _mm512_cvtepi32_epi8(arg));
        }
        if ((t & 31) == 31) {
            __m512 mn = _mm512_loadu_ps(N);
            for (size_t i = 16; i < S; i += 16) mn = _mm512_min_ps(mn, _mm512_loadu_ps(N + i));
            const float m = _mm512_reduce_min_ps(mn); const __m512 mv = _mm512_set1_ps(m);
            for (size_t i = 0; i < S; i += 16) _mm512_storeu_ps(N + i, _mm512_sub_ps(_mm512_loadu_ps(N + i), mv));
            offset += m;
        }
        for (size_t b = 0; b < nT; ++b) transpose16(N + 256*b, T + 256*b);
    }
    size_t s = 0; float mn = N[0];
    for (size_t i = 1; i < S; ++i) if (N[i] < mn) { mn = N[i]; s = i; }
    for (int t = n - 1; t >= 0; --t) {
        const uint8_t z = wk.bp[size_t(t)*S + s];
        idx[t] = uint32_t(s*16 + z);
        s = (s % P)*16 + z;
    }
    return offset + mn;
}
#else
static inline double viterbi(const Labels & L, const float * xn, const float * ww, int n, uint32_t * idx, Work & wk) {
    const size_t S = kStates, P = S/16;
    wk.N.assign(S, 1e30f); wk.T.resize(S); wk.bp.resize(size_t(n)*S);
    float * cost = wk.N.data(), * nw = wk.T.data(); cost[0] = 0;
    double offset = 0;
    for (int t = 0; t < n; ++t) {
        uint8_t * bp = wk.bp.data() + size_t(t)*S;
        for (size_t sp = 0; sp < S; ++sp) {
            const size_t h = sp % P; float best = 3e38f; int arg = 0;
            for (int z = 0; z < 16; ++z) {
                const float d = xn[t] - L.nat[sp*16 + z];
                const float e = d*d*ww[t] + cost[h*16 + z];
                if (e < best) { best = e; arg = z; }
            }
            nw[sp] = best; bp[sp] = uint8_t(arg);
        }
        std::swap(cost, nw);
        if ((t & 31) == 31) {
            float m = cost[0]; for (size_t i = 1; i < S; ++i) m = std::min(m, cost[i]);
            for (size_t i = 0; i < S; ++i) cost[i] -= m;
            offset += m;
        }
    }
    size_t s = 0; for (size_t i = 1; i < S; ++i) if (cost[i] < cost[s]) s = i;
    const double tot = offset + cost[s];
    for (int t = n - 1; t >= 0; --t) { const uint8_t z = wk.bp[size_t(t)*S + s]; idx[t] = uint32_t(s*16 + z); s = (s % P)*16 + z; }
    return tot;
}
#endif

static inline void decode_stream(const Labels & L, const uint8_t * nib, int n, float * v) {
    size_t win = 0;
    for (int t = 0; t < n; ++t) { win = (win >> 4) | (size_t(nib[t]) << 12); v[t] = L.nat[win]; }
}

static inline void quantize_scales(float * s, const double * num, const double * den, int nb, float & d_out, uint8_t * q_out) {
    double d = 0; for (int b = 0; b < nb; ++b) d = std::max(d, (double)s[b]);
    if (!(d > 0)) {
        d_out = 0; for (int b = 0; b < nb; ++b) { q_out[b] = 0; s[b] = 0; }
        return;
    }
    std::vector<int> q(nb);
    for (int it = 0; it < 2; ++it) {
        double a = 0, c = 0;
        for (int b = 0; b < nb; ++b) {
            const double r = s[b]/d;
            int qb = 0;
            if (r > 0) {
                qb = (int)std::nearbyint(std::min(1.0, std::max(0.0, r))*255); int best = qb;
                for (int cand = qb - 1; cand <= qb + 1; ++cand)
                    if (cand >= 0 && cand <= 255 && std::fabs(double(cand)/255 - r) < std::fabs(double(best)/255 - r)) best = cand;
                qb = best;
            }
            q[b] = qb; const double g = qb <= 0 ? 0 : double(qb)/255; a += num[b]*g; c += den[b]*g*g;
        }
        if (a > 0 && c > 0) d = a/c;
    }
    d_out = (float)d;
    for (int b = 0; b < nb; ++b) { q_out[b] = (uint8_t)q[b]; s[b] = block_scale(d_out, q[b]); }
}

struct EncState {
    std::vector<float>   s;
    std::vector<float>   q;
    std::vector<double>  vnum, vden;
    std::vector<uint8_t> qs8;
    std::vector<uint8_t> nib_best, q_best;
    float  d_best = 0;
    double best = -1;
};

static inline void enc_init(const Labels & L, const float * x, const float * w, int n, EncState & st) {
    const int sb = kBlock, nb = n/sb;
    st.s.resize(nb); st.q.resize(n); st.vnum.resize(nb); st.vden.resize(nb); st.qs8.resize(nb); st.q_best.resize(nb); st.nib_best.resize(n);
    st.best = -1; st.d_best = 0;
    for (int b = 0; b < nb; ++b) {
        double sw = 0, sx = 0;
        for (int j = b*sb; j < (b+1)*sb; ++j) { sw += w[j]; sx += double(w[j])*x[j]*x[j]; }
        st.s[b] = (sw > 0 && sx > 0) ? float(std::sqrt(sx/sw)/L.sig) : 1e-12f;
    }
}

static inline void enc_targets(const EncState & st, const float * x, const float * w, int n, float * xn, float * ww) {
    for (int j = 0; j < n; ++j) { const float sc = st.s[j/kBlock]; xn[j] = x[j]/sc; ww[j] = w[j]*sc*sc; }
}

static inline void enc_finish(const Labels & L, const float * x, const float * w, int n, const uint8_t * nib, EncState & st) {
    const int sb = kBlock, nb = n/sb;
    decode_stream(L, nib, n, st.q.data());
    for (int b = 0; b < nb; ++b) {
        double num = 0, den = 0;
        const int j0 = b*sb, j1 = (b+1)*sb;
        for (int j = j0; j < j1; ++j) { num += double(w[j])*x[j]*st.q[j]; den += double(w[j])*st.q[j]*st.q[j]; }
        if (den > 0 && num != 0) st.s[b] = float(num/den);
        st.vnum[b] = num; st.vden[b] = den;
    }
    float d; quantize_scales(st.s.data(), st.vnum.data(), st.vden.data(), nb, d, st.qs8.data());
    double E = 0;
    for (int b = 0; b < nb; ++b) {
        const int j0 = b*sb, j1 = (b+1)*sb;
        for (int j = j0; j < j1; ++j) { const double e = x[j] - double(st.s[b])*st.q[j]; E += w[j]*e*e; }
    }
    if (st.best < 0 || E < st.best) {
        st.best = E; st.d_best = d;
        std::memcpy(st.q_best.data(), st.qs8.data(), nb);
        std::memcpy(st.nib_best.data(), nib, n);
    }
}

static inline double encode_stream(const Labels & L, const float * x, const float * w, int n, int passes,
                                   uint8_t * nib, float & d_out, uint8_t * q_out, Work & wk) {
    EncState st;
    enc_init(L, x, w, n, st);
    wk.xn.resize(n); wk.ww.resize(n); wk.idx.resize(n);
    std::vector<uint8_t> nb4(n);
    for (int pass = 0; pass < passes; ++pass) {
        enc_targets(st, x, w, n, wk.xn.data(), wk.ww.data());
        viterbi(L, wk.xn.data(), wk.ww.data(), n, wk.idx.data(), wk);
        for (int j = 0; j < n; ++j) nb4[j] = uint8_t(wk.idx[j] >> 12);
        enc_finish(L, x, w, n, nb4.data(), st);
    }
    d_out = st.d_best;
    std::memcpy(q_out, st.q_best.data(), n/kBlock);
    std::memcpy(nib, st.nib_best.data(), n);
    return st.best;
}

}
