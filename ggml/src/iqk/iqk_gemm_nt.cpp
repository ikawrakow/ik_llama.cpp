#include "iqk_common.h"
#include "iqk_gemm_nt.h"
#include "ggml.h"

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "iqk_nt_codec.h"

#include <vector>

#ifdef HAVE_FANCY_SIMD

namespace {

template <int nrc_y>
void mul_mat_iq4_nt_q8_k128(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%iqk_nt::kBlock == 0);
    constexpr int kOff = iqk_nt::kTableMax;
    const int nb = n/iqk_nt::kBlock;

    const block_q8_K128 * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8_K128 *)info.src1_row(iy);

    thread_local std::vector<float> buffer;
    if ((int)buffer.size() < 9*nb + 16) buffer.resize(9*nb + 16);
    float * dys = buffer.data(), * sblk = dys + 8*nb;
    for (int ib = 0; ib < nb; ++ib) {
        for (int iy = 0; iy < 8; ++iy) dys[8*ib + iy] = iy < nrc_y ? y[iy][ib].d : 0.f;
    }

    const iqk_nt::LabelsAvx512 L(kOff);

    float f[8];

    for (int ix = 0; ix < nrc_x; ++ix) {
        const auto r = iqk_nt::row_view((const char *)vx + ix*bx, n);

        const __m512 vd = _mm512_set1_ps(r.d*(1.f/255));
        for (int ib = 0; ib < nb; ib += 16) {
            const __mmask16 km = nb - ib >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << (nb - ib)) - 1);
            _mm512_storeu_ps(sblk + ib, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(km, r.q + ib))), vd));
        }

        __m512 accf[nrc_y];
        for (int iy = 0; iy < nrc_y; ++iy) accf[iy] = _mm512_setzero_ps();
        const uint8_t * qs = r.qs;
        for (int ib = 0; ib < nb; ++ib) {
            __m512i ve, vo;
            L.labels(qs, ve, vo); qs += 64;
            const __m256 s8 = _mm256_broadcast_ss(sblk + ib);
            _mm256_storeu_ps(f, _mm256_mul_ps(s8, _mm256_loadu_ps(dys + 8*ib)));
            for (int iy = 0; iy < nrc_y; ++iy) {
                const int8_t * yq = y[iy][ib].qs;
                __m512i acc = _mm512_dpbusd_epi32(_mm512_zextsi128_si512(_mm_cvtsi32_si128(-(4*kOff)*y[iy][ib].s)), ve, _mm512_loadu_si512((const __m512i *)yq));
                acc = _mm512_dpbusd_epi32(acc, vo, _mm512_loadu_si512((const __m512i *)(yq + 64)));
                accf[iy] = _mm512_fmadd_ps(_mm512_set1_ps(f[iy]), _mm512_cvtepi32_ps(acc), accf[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) info.store(ix, iy, _mm512_reduce_add_ps(accf[iy]));
    }
}


inline void transpose_xor128(const __m256i * stage, int8_t * qs) {
    const auto ia0 = _mm512_setr_epi32(0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30);
    const auto ia1 = _mm512_setr_epi32(1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31);
    const auto ib0 = _mm512_setr_epi32(0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29);
    const auto ib1 = _mm512_setr_epi32(2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31);
    const auto ic0 = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 16, 24, 17, 25, 18, 26, 19, 27);
    const auto ic1 = _mm512_setr_epi32(4, 12, 5, 13, 6, 14, 7, 15, 20, 28, 21, 29, 22, 30, 23, 31);
    const auto flip = _mm512_set1_epi8(-128);
    for (int ib32 = 0; ib32 < 4; ++ib32) {
        auto in = (const __m512i *)(stage + 16*ib32);
        __m512i z[8];
        for (int j = 0; j < 8; ++j) z[j] = _mm512_loadu_si512(in + j);
        __m512i a[8];
        a[0] = _mm512_permutex2var_epi32(z[0], ia0, z[1]);
        a[1] = _mm512_permutex2var_epi32(z[0], ia1, z[1]);
        a[2] = _mm512_permutex2var_epi32(z[2], ia0, z[3]);
        a[3] = _mm512_permutex2var_epi32(z[2], ia1, z[3]);
        a[4] = _mm512_permutex2var_epi32(z[4], ia0, z[5]);
        a[5] = _mm512_permutex2var_epi32(z[4], ia1, z[5]);
        a[6] = _mm512_permutex2var_epi32(z[6], ia0, z[7]);
        a[7] = _mm512_permutex2var_epi32(z[6], ia1, z[7]);
        __m512i b[8];
        b[0] = _mm512_permutex2var_epi32(a[0], ib0, a[2]);
        b[1] = _mm512_permutex2var_epi32(a[1], ib0, a[3]);
        b[2] = _mm512_permutex2var_epi32(a[0], ib1, a[2]);
        b[3] = _mm512_permutex2var_epi32(a[1], ib1, a[3]);
        b[4] = _mm512_permutex2var_epi32(a[4], ib0, a[6]);
        b[5] = _mm512_permutex2var_epi32(a[5], ib0, a[7]);
        b[6] = _mm512_permutex2var_epi32(a[4], ib1, a[6]);
        b[7] = _mm512_permutex2var_epi32(a[5], ib1, a[7]);
        auto out = (__m512i *)qs + 8*ib32;
        _mm512_storeu_si512(out + 0, _mm512_xor_si512(_mm512_permutex2var_epi32(b[0], ic0, b[4]), flip));
        _mm512_storeu_si512(out + 1, _mm512_xor_si512(_mm512_permutex2var_epi32(b[2], ic0, b[6]), flip));
        _mm512_storeu_si512(out + 2, _mm512_xor_si512(_mm512_permutex2var_epi32(b[0], ic1, b[4]), flip));
        _mm512_storeu_si512(out + 3, _mm512_xor_si512(_mm512_permutex2var_epi32(b[2], ic1, b[6]), flip));
        _mm512_storeu_si512(out + 4, _mm512_xor_si512(_mm512_permutex2var_epi32(b[1], ic0, b[5]), flip));
        _mm512_storeu_si512(out + 5, _mm512_xor_si512(_mm512_permutex2var_epi32(b[3], ic0, b[7]), flip));
        _mm512_storeu_si512(out + 6, _mm512_xor_si512(_mm512_permutex2var_epi32(b[1], ic1, b[5]), flip));
        _mm512_storeu_si512(out + 7, _mm512_xor_si512(_mm512_permutex2var_epi32(b[3], ic1, b[7]), flip));
    }
}

template <int nrc_y>
void mul_mat_q8_k128_r16_q8_k128(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%16 == 0);
    GGML_ASSERT(n%128 == 0);
    const block_q8_K128 * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8_K128 *)info.src1_row(iy);
    const int nbl = n/128;
    int ix = 0;
    for (; ix + 32 <= nrc_x; ix += 32) {
        const block_q8_k128_r16 * iqa = (const block_q8_k128_r16 *)((const char *)vx + (ix +  0)*bx);
        const block_q8_k128_r16 * iqb = (const block_q8_k128_r16 *)((const char *)vx + (ix + 16)*bx);
        __m512  acca[nrc_y] = {};
        __m512  accb[nrc_y] = {};
        __m512i suma[nrc_y], sumb[nrc_y];
        __m512i qa[4], qb[4];
        for (int ibl = 0; ibl < nbl; ++ibl) {
            for (int iy = 0; iy < nrc_y; ++iy) suma[iy] = sumb[iy] = _mm512_set1_epi32(-128*y[iy][ibl].s);
            for (int ib = 0; ib < 8; ++ib) {
                for (int k = 0; k < 4; ++k) {
                    qa[k] = _mm512_loadu_si512((const __m512i *)iqa[ibl].qs + 4*ib + k);
                    qb[k] = _mm512_loadu_si512((const __m512i *)iqb[ibl].qs + 4*ib + k);
                }
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const int8_t * yb = y[iy][ibl].qs + 16*ib;
                    for (int k = 0; k < 4; ++k) {
                        auto v = _mm512_set1_epi32(*(const int32_t *)(yb + 4*k));
                        suma[iy] = _mm512_dpbusd_epi32(suma[iy], qa[k], v);
                        sumb[iy] = _mm512_dpbusd_epi32(sumb[iy], qb[k], v);
                    }
                }
            }
            auto da = _mm512_loadu_ps(iqa[ibl].d);
            auto db = _mm512_loadu_ps(iqb[ibl].d);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto dy = _mm512_set1_ps(y[iy][ibl].d);
                acca[iy] = _mm512_fmadd_ps(_mm512_mul_ps(da, dy), _mm512_cvtepi32_ps(suma[iy]), acca[iy]);
                accb[iy] = _mm512_fmadd_ps(_mm512_mul_ps(db, dy), _mm512_cvtepi32_ps(sumb[iy]), accb[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info.store(ix +  0, iy, acca[iy]);
            info.store(ix + 16, iy, accb[iy]);
        }
    }
    for (; ix < nrc_x; ix += 16) {
        const block_q8_k128_r16 * iq = (const block_q8_k128_r16 *)((const char *)vx + ix*bx);
        __m512  acc[nrc_y] = {};
        __m512i sum[nrc_y];
        __m512i qx[4];
        for (int ibl = 0; ibl < nbl; ++ibl) {
            for (int iy = 0; iy < nrc_y; ++iy) sum[iy] = _mm512_set1_epi32(-128*y[iy][ibl].s);
            for (int ib = 0; ib < 8; ++ib) {
                for (int k = 0; k < 4; ++k) qx[k] = _mm512_loadu_si512((const __m512i *)iq[ibl].qs + 4*ib + k);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    const int8_t * yb = y[iy][ibl].qs + 16*ib;
                    for (int k = 0; k < 4; ++k) sum[iy] = _mm512_dpbusd_epi32(sum[iy], qx[k], _mm512_set1_epi32(*(const int32_t *)(yb + 4*k)));
                }
            }
            auto d = _mm512_loadu_ps(iq[ibl].d);
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm512_fmadd_ps(_mm512_mul_ps(d, _mm512_set1_ps(y[iy][ibl].d)), _mm512_cvtepi32_ps(sum[iy]), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) info.store(ix, iy, acc[iy]);
    }
}

}

bool iqk_set_kernels_nt(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {
    func16 = nullptr;
    if (ne00%iqk_nt::kBlock != 0 || ggml_type(typeB) != GGML_TYPE_Q8_K128) return false;
    if (ggml_type(typeA) == GGML_TYPE_IQ4_NT) {
        IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_nt_q8_k128, kernels);
        return true;
    }
    if (ggml_type(typeA) == GGML_TYPE_Q8_K128_R16) {
        IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_k128_r16_q8_k128, kernels);
        return true;
    }
    return false;
}

bool iqk_convert_nt_q8_k128_r16(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%iqk_nt::kBlock == 0);
    GGML_ASSERT(nrc_x%16 == 0);
    const int nb = n/iqk_nt::kBlock;
    block_q8_k128_r16 * y = (block_q8_k128_r16 *)vy;
    const iqk_nt::LabelsAvx512 L(0);
    __m256i stage[64];
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    iqk_nt::RowView r[16];
    for (int ix = 0; ix < nrc_x; ix += 16) {
        for (int k = 0; k < 16; ++k) r[k] = iqk_nt::row_view((const char *)vx + (ix + k)*bx, n);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 16; ++k) {
                __m512i ve, vo;
                L.labels(r[k].qs + 64*i, ve, vo);
                stage[16*0 + k] = _mm256_permutevar8x32_epi32(_mm512_castsi512_si256(ve), perm);
                stage[16*1 + k] = _mm256_permutevar8x32_epi32(_mm512_extracti64x4_epi64(ve, 1), perm);
                stage[16*2 + k] = _mm256_permutevar8x32_epi32(_mm512_castsi512_si256(vo), perm);
                stage[16*3 + k] = _mm256_permutevar8x32_epi32(_mm512_extracti64x4_epi64(vo, 1), perm);
                y[i].d[k] = iqk_nt::block_scale(r[k].d, r[k].q[i]);
            }
            transpose_xor128(stage, y[i].qs);
        }
        y += nb;
    }
    return true;
}

#elif defined __AVX2__

namespace {

template <int nrc_y>
void mul_mat_iq4_nt_q8_k128(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%iqk_nt::kBlock == 0);
    const int nb = n/iqk_nt::kBlock;

    const block_q8_K128 * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8_K128 *)info.src1_row(iy);

    const iqk_nt::LabelsAvx2 L;
    const __m256i ones = _mm256_set1_epi16(1);

    for (int ix = 0; ix < nrc_x; ++ix) {
        const auto r = iqk_nt::row_view((const char *)vx + ix*bx, n);
        const float dr = r.d*(1.f/255);
        __m256 accf[nrc_y];
        for (int iy = 0; iy < nrc_y; ++iy) accf[iy] = _mm256_setzero_ps();
        const uint8_t * qs = r.qs;
        for (int ib = 0; ib < nb; ++ib) {
            __m256i v[4], a[4];
            L.labels(qs, v); qs += 64;
            for (int k = 0; k < 4; ++k) a[k] = _mm256_abs_epi8(v[k]);
            const float sb = (float)r.q[ib]*dr;
            for (int iy = 0; iy < nrc_y; ++iy) {
                const int8_t * yq = y[iy][ib].qs;
                __m256i sum = _mm256_setzero_si256();
                for (int k = 0; k < 4; ++k) {
                    const __m256i yk = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(yq + 32*k)), v[k]);
                    sum = _mm256_add_epi32(sum, _mm256_madd_epi16(ones, _mm256_maddubs_epi16(a[k], yk)));
                }
                accf[iy] = _mm256_fmadd_ps(_mm256_set1_ps(sb*y[iy][ib].d), _mm256_cvtepi32_ps(sum), accf[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) info.store(ix, iy, hsum_float_8(accf[iy]));
    }
}

template <int nrc_y>
void mul_mat_q8_k128_r16_q8_k128(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%16 == 0);
    GGML_ASSERT(n%128 == 0);
    const block_q8_K128 * y[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const block_q8_K128 *)info.src1_row(iy);
    const int nbl = n/128;
    const __m256i flip = _mm256_set1_epi8(-128);
    const __m256i ones = _mm256_set1_epi16(1);
    for (int ix = 0; ix < nrc_x; ix += 16) {
        const block_q8_k128_r16 * iq = (const block_q8_k128_r16 *)((const char *)vx + ix*bx);
        for (int h = 0; h < 2; ++h) {
            __m256 acc[nrc_y];
            for (int iy = 0; iy < nrc_y; ++iy) acc[iy] = _mm256_setzero_ps();
            for (int ibl = 0; ibl < nbl; ++ibl) {
                const int8_t * qs = iq[ibl].qs + 32*h;
                __m256i sum[nrc_y];
                for (int iy = 0; iy < nrc_y; ++iy) sum[iy] = _mm256_setzero_si256();
                for (int j = 0; j < 32; ++j) {
                    const __m256i q = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(qs + 64*j)), flip);
                    const __m256i a = _mm256_abs_epi8(q);
                    for (int iy = 0; iy < nrc_y; ++iy) {
                        const __m256i yj = _mm256_set1_epi32(*(const int32_t *)(y[iy][ibl].qs + 4*j));
                        sum[iy] = _mm256_add_epi32(sum[iy], _mm256_madd_epi16(ones, _mm256_maddubs_epi16(a, _mm256_sign_epi8(yj, q))));
                    }
                }
                const __m256 d = _mm256_loadu_ps(iq[ibl].d + 8*h);
                for (int iy = 0; iy < nrc_y; ++iy) {
                    acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d, _mm256_set1_ps(y[iy][ibl].d)), _mm256_cvtepi32_ps(sum[iy]), acc[iy]);
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) info.store(ix + 8*h, iy, acc[iy]);
        }
    }
}

inline void transpose_8x8_xor128(const __m256i * r, int8_t * out) {
    const __m256i flip = _mm256_set1_epi8(-128);
    __m256i t[8], u[8];
    for (int k = 0; k < 4; ++k) {
        t[2*k+0] = _mm256_unpacklo_epi32(r[2*k], r[2*k+1]);
        t[2*k+1] = _mm256_unpackhi_epi32(r[2*k], r[2*k+1]);
    }
    for (int k = 0; k < 2; ++k) {
        u[4*k+0] = _mm256_unpacklo_epi64(t[4*k+0], t[4*k+2]);
        u[4*k+1] = _mm256_unpackhi_epi64(t[4*k+0], t[4*k+2]);
        u[4*k+2] = _mm256_unpacklo_epi64(t[4*k+1], t[4*k+3]);
        u[4*k+3] = _mm256_unpackhi_epi64(t[4*k+1], t[4*k+3]);
    }
    for (int m = 0; m < 4; ++m) {
        _mm256_storeu_si256((__m256i *)(out + 64*(m + 0)), _mm256_xor_si256(_mm256_permute2x128_si256(u[m], u[m+4], 0x20), flip));
        _mm256_storeu_si256((__m256i *)(out + 64*(m + 4)), _mm256_xor_si256(_mm256_permute2x128_si256(u[m], u[m+4], 0x31), flip));
    }
}

}

bool iqk_set_kernels_nt(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {
    func16 = nullptr;
    if (ne00%iqk_nt::kBlock != 0 || ggml_type(typeB) != GGML_TYPE_Q8_K128) return false;
    if (ggml_type(typeA) == GGML_TYPE_IQ4_NT) {
        IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_nt_q8_k128, kernels);
        return true;
    }
    if (ggml_type(typeA) == GGML_TYPE_Q8_K128_R16) {
        IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_q8_k128_r16_q8_k128, kernels);
        return true;
    }
    return false;
}

bool iqk_convert_nt_q8_k128_r16(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%iqk_nt::kBlock == 0);
    GGML_ASSERT(nrc_x%16 == 0);
    const int nb = n/iqk_nt::kBlock;
    block_q8_k128_r16 * y = (block_q8_k128_r16 *)vy;
    const iqk_nt::LabelsAvx2 L;
    __m256i v[16][4], r[8];
    iqk_nt::RowView rv[16];
    for (int ix = 0; ix < nrc_x; ix += 16) {
        for (int k = 0; k < 16; ++k) rv[k] = iqk_nt::row_view((const char *)vx + (ix + k)*bx, n);
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 16; ++k) {
                L.labels(rv[k].qs + 64*i, v[k]);
                y[i].d[k] = iqk_nt::block_scale(rv[k].d, rv[k].q[i]);
            }
            for (int h = 0; h < 2; ++h) {
                for (int c = 0; c < 4; ++c) {
                    for (int k = 0; k < 8; ++k) r[k] = v[8*h + k][c];
                    transpose_8x8_xor128(r, y[i].qs + 64*8*c + 32*h);
                }
            }
        }
        y += nb;
    }
    return true;
}

#else

bool iqk_set_kernels_nt(int, int, int, std::array<mul_mat_t, IQK_MAX_NY>&, mul_mat_t& func16) {
    func16 = nullptr;
    return false;
}

bool iqk_convert_nt_q8_k128_r16(int, const void *, size_t, void *, int) {
    return false;
}


#endif

#endif
