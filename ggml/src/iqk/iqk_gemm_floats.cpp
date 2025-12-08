#include "iqk_gemm_floats.h"

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#ifdef __x86_64__

namespace {

// float matrices - we handle f16, bf16 (if native bf16 support is available) and f32, but only to f32 result

struct QFBase {
#ifdef __AVX512F__
    constexpr static int k_step = 16;
    using Data = __m512;
    using Acc  = __m512;
    static inline Data load(const ggml_half * x) { return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)x)); }
    static inline Data load(const float * x) { return _mm512_loadu_ps(x); }
    static inline Data load(const ggml_bf16_t * x) {
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)x)), 16));
    }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) {
        return _mm512_fmadd_ps(y, x, prev);
    }
    static inline Acc acc_first(const Data& y, const Data& x) {
        return _mm512_mul_ps(y, x);
    }
    static inline Acc add(Acc x, Acc y) { return _mm512_add_ps(x, y); }
    static inline float hsum(Acc acc) {
        return _mm512_reduce_add_ps(acc);
    }
    template <typename Float>
    static inline Data load4Floats(const Float * x) {
        return _mm512_insertf32x4(_mm512_setzero_ps(), load128(x), 0);
    }
    static inline Acc acc_r4(Acc acc, const Data * xv, const Data& yv) {
        acc = _mm512_fmadd_ps(xv[0], _mm512_shuffle_ps(yv, yv, 0x00), acc);
        acc = _mm512_fmadd_ps(xv[1], _mm512_shuffle_ps(yv, yv, 0x55), acc);
        acc = _mm512_fmadd_ps(xv[2], _mm512_shuffle_ps(yv, yv, 0xaa), acc);
        acc = _mm512_fmadd_ps(xv[3], _mm512_shuffle_ps(yv, yv, 0xff), acc);
        return acc;
    }
    static inline Acc acc_r4_first(const Data * xv, const Data& yv) {
        auto acc = _mm512_mul_ps(xv[0], _mm512_shuffle_ps(yv, yv, 0x00));
        acc = _mm512_fmadd_ps(xv[1], _mm512_shuffle_ps(yv, yv, 0x55), acc);
        acc = _mm512_fmadd_ps(xv[2], _mm512_shuffle_ps(yv, yv, 0xaa), acc);
        acc = _mm512_fmadd_ps(xv[3], _mm512_shuffle_ps(yv, yv, 0xff), acc);
        return acc;
    }
    static inline __m128 hsum_r4(Acc acc) {
        auto sum1 = _mm_add_ps(_mm512_extractf32x4_ps(acc, 0), _mm512_extractf32x4_ps(acc, 1));
        auto sum2 = _mm_add_ps(_mm512_extractf32x4_ps(acc, 2), _mm512_extractf32x4_ps(acc, 3));
        return _mm_add_ps(sum1, sum2);
    }
#else
    constexpr static int k_step = 8;
    using Data = __m256;
    using Acc  = __m256;
    static inline Data load(const ggml_half * x) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)x)); }
    static inline Data load(const float * x) { return _mm256_loadu_ps(x); }
    static inline Data load(const ggml_bf16_t * x) {
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)x)), 16));
    }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) {
        return _mm256_fmadd_ps(y, x, prev);
    }
    static inline Acc add(Acc x, Acc y) { return _mm256_add_ps(x, y); }
    static inline Acc acc_r4(Acc acc, const Data * xv, const Data& yv) {
        acc = _mm256_fmadd_ps(xv[0], _mm256_shuffle_ps(yv, yv, 0x00), acc);
        acc = _mm256_fmadd_ps(xv[1], _mm256_shuffle_ps(yv, yv, 0x55), acc);
        acc = _mm256_fmadd_ps(xv[2], _mm256_shuffle_ps(yv, yv, 0xaa), acc);
        acc = _mm256_fmadd_ps(xv[3], _mm256_shuffle_ps(yv, yv, 0xff), acc);
        return acc;
    }
    static inline Acc acc_r4_first(const Data * xv, const Data& yv) {
        auto acc = _mm256_mul_ps(xv[0], _mm256_shuffle_ps(yv, yv, 0x00));
        acc = _mm256_fmadd_ps(xv[1], _mm256_shuffle_ps(yv, yv, 0x55), acc);
        acc = _mm256_fmadd_ps(xv[2], _mm256_shuffle_ps(yv, yv, 0xaa), acc);
        acc = _mm256_fmadd_ps(xv[3], _mm256_shuffle_ps(yv, yv, 0xff), acc);
        return acc;
    }
    static inline Acc acc_first(const Data& y, const Data& x) {
        return _mm256_mul_ps(y, x);
    }
    static inline float hsum(Acc acc) {
        return hsum_float_8(acc);
    }
    static inline __m128 hsum_r4(Acc acc) {
        return _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    }
    template <typename Float>
    static inline Data load4Floats(const Float * x) {
        return _mm256_insertf128_ps(_mm256_setzero_ps(), load128(x), 0);
    }
#endif
    static inline __m128 load128(const ggml_half * x) { return _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)x)); }
    static inline __m128 load128(const float * x) { return _mm_loadu_ps(x); }
    static inline __m128 load128(const ggml_bf16_t * x) {
        return _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*)x)), 16));
    }
};

template <typename Float, int nrc_in> struct QFT final : public QFBase {
    constexpr static int nrc = nrc_in;
    QFT(const DataInfo& info) {
        for (int iy = 0; iy < nrc; ++iy) y[iy] = (const Float *)info.src1_row(iy);
    }
    QFT(const char * cx, size_t bx) {
        for (int iy = 0; iy < nrc; ++iy) y[iy] = (const Float *)(cx + iy*bx);
    }
    IQK_ALWAYS_INLINE Data load1(int iy, int i) const { return load(y[iy] + k_step*i); }
    IQK_ALWAYS_INLINE Data load_tail(int iy, int i) const { return load4Floats(y[iy] + 4*i); }
    IQK_ALWAYS_INLINE void load_r4(int ix, int i, Data * xv) const {
        xv[0] = load1(ix+0, i);
        xv[1] = load1(ix+1, i);
        xv[2] = load1(ix+2, i);
        xv[3] = load1(ix+3, i);
#ifdef __AVX512F__
        auto t0 = _mm512_unpacklo_ps(xv[0], xv[1]);
        auto t1 = _mm512_unpacklo_ps(xv[2], xv[3]);
        auto t2 = _mm512_unpackhi_ps(xv[0], xv[1]);
        auto t3 = _mm512_unpackhi_ps(xv[2], xv[3]);
        xv[0] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t0), _mm512_castps_pd(t1)));
        xv[1] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t0), _mm512_castps_pd(t1)));
        xv[2] = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t2), _mm512_castps_pd(t3)));
        xv[3] = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t2), _mm512_castps_pd(t3)));
#else
        auto t0 = _mm256_unpacklo_ps(xv[0], xv[1]);
        auto t1 = _mm256_unpacklo_ps(xv[2], xv[3]);
        auto t2 = _mm256_unpackhi_ps(xv[0], xv[1]);
        auto t3 = _mm256_unpackhi_ps(xv[2], xv[3]);
        xv[0] = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t1)));
        xv[1] = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t1)));
        xv[2] = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t2), _mm256_castps_pd(t3)));
        xv[3] = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t2), _mm256_castps_pd(t3)));
#endif
    }
    const Float * y[nrc];
};

// TBD if we want this
//template <typename Qy, typename Qx>
//IQK_NOINLINE void mul_mat_Qx_Qy_Mx1(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
//    static_assert(Qy::nrc == 1);
//    int nb = n/QFBase::k_step;
//    int nb4 = n/4;
//    Qy y(info);
//    Qx x(cx + ix0*bx, bx);
//    QFBase::Data xv[2*Qx::nrc];
//    QFBase::Acc  acc[2*Qx::nrc];
//    auto yv1 = y.load1(0, 0);
//    auto yv2 = y.load1(0, 1);
//    for (int ix = 0; ix < Qx::nrc; ++ix) {
//        xv[2*ix+0] = x.load1(ix, 0);
//        xv[2*ix+1] = x.load1(ix, 1);
//        acc[2*ix+0] = QFBase::acc_first(yv1, xv[2*ix+0]);
//        acc[2*ix+1] = QFBase::acc_first(yv2, xv[2*ix+1]);
//    }
//    for (int i = 1; i < nb/2; ++i) {
//        yv1 = y.load1(0, 2*i+0);
//        yv2 = y.load1(0, 2*i+1);
//        for (int ix = 0; ix < Qx::nrc; ++ix) {
//            xv[2*ix+0] = x.load1(ix, 2*i+0);
//            xv[2*ix+1] = x.load1(ix, 2*i+1);
//            acc[2*ix+0] = QFBase::acc(acc[2*ix+0], yv1, xv[2*ix+0]);
//            acc[2*ix+1] = QFBase::acc(acc[2*ix+1], yv2, xv[2*ix+1]);
//        }
//    }
//    for (int i = (QFBase::k_step/4)*nb; i < nb4; ++i) {
//        yv1 = y.load_tail(0, i);
//        for (int ix = 0; ix < Qx::nrc; ++ix) {
//            xv[ix] = x.load_tail(ix, i);
//            acc[2*ix+0] = QFBase::acc(acc[2*ix+0], yv1, xv[ix]);
//        }
//    }
//    for (int ix = 0; ix < Qx::nrc; ++ix) info.store(ix0+ix, 0, QFBase::hsum(QFBase::add(acc[2*ix+0], acc[2*ix+1])));
//}

template <typename Qy, typename Qx>
IQK_NOINLINE void mul_mat_Qx_Qy_MxN(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    int nb = n/QFBase::k_step;
    int nb4 = n/4;
    Qy y(info);
    Qx x(cx + ix0*bx, bx);
    QFBase::Data xv[Qx::nrc];
    QFBase::Acc  acc[Qx::nrc*Qy::nrc];
    auto yv = y.load1(0, 0);
    for (int ix = 0; ix < Qx::nrc; ++ix) {
        xv[ix] = x.load1(ix, 0);
        acc[ix] = QFBase::acc_first(yv, xv[ix]);
    }
    for (int iy = 1; iy < Qy::nrc; ++iy) {
        yv = y.load1(iy, 0);
        for (int ix = 0; ix < Qx::nrc; ++ix) acc[Qx::nrc*iy + ix] = QFBase::acc_first(yv, xv[ix]);
    }
    for (int i = 1; i < nb; ++i) {
        yv = y.load1(0, i);
        for (int ix = 0; ix < Qx::nrc; ++ix) {
            xv[ix] = x.load1(ix, i);
            acc[ix] = QFBase::acc(acc[ix], yv, xv[ix]);
        }
        for (int iy = 1; iy < Qy::nrc; ++iy) {
            yv = y.load1(iy, i);
            for (int ix = 0; ix < Qx::nrc; ++ix) acc[Qx::nrc*iy + ix] = QFBase::acc(acc[Qx::nrc*iy + ix], yv, xv[ix]);
        }
    }
    for (int i = (QFBase::k_step/4)*nb; i < nb4; ++i) {
        yv = y.load_tail(0, i);
        for (int ix = 0; ix < Qx::nrc; ++ix) {
            xv[ix] = x.load_tail(ix, i);
            acc[ix] = QFBase::acc(acc[ix], yv, xv[ix]);
        }
        for (int iy = 1; iy < Qy::nrc; ++iy) {
            yv = y.load_tail(iy, i);
            for (int ix = 0; ix < Qx::nrc; ++ix) acc[Qx::nrc*iy + ix] = QFBase::acc(acc[Qx::nrc*iy + ix], yv, xv[ix]);
        }
    }
    for (int iy = 0; iy < Qy::nrc; ++iy) for (int ix = 0; ix < Qx::nrc; ++ix) info.store(ix0+ix, iy, QFBase::hsum(acc[Qx::nrc*iy+ix]));
}

template <typename Qy, typename Qx>
inline void mul_mat_Qx_Qy_MxN_fa(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    int nb = n/QFBase::k_step;
    Qy y(info);
    Qx x(cx + ix0*bx, bx);
    QFBase::Data xv[Qx::nrc];
    QFBase::Acc  acc[Qx::nrc*Qy::nrc];
    auto yv = y.load1(0, 0);
    for (int ix = 0; ix < Qx::nrc; ++ix) {
        xv[ix] = x.load1(ix, 0);
        acc[ix] = QFBase::acc_first(yv, xv[ix]);
    }
    for (int iy = 1; iy < Qy::nrc; ++iy) {
        yv = y.load1(iy, 0);
        for (int ix = 0; ix < Qx::nrc; ++ix) acc[Qx::nrc*iy + ix] = QFBase::acc_first(yv, xv[ix]);
    }
    for (int i = 1; i < nb; ++i) {
        yv = y.load1(0, i);
        for (int ix = 0; ix < Qx::nrc; ++ix) {
            xv[ix] = x.load1(ix, i);
            acc[ix] = QFBase::acc(acc[ix], yv, xv[ix]);
        }
        for (int iy = 1; iy < Qy::nrc; ++iy) {
            yv = y.load1(iy, i);
            for (int ix = 0; ix < Qx::nrc; ++ix) acc[Qx::nrc*iy + ix] = QFBase::acc(acc[Qx::nrc*iy + ix], yv, xv[ix]);
        }
    }
    for (int iy = 0; iy < Qy::nrc; ++iy) for (int ix = 0; ix < Qx::nrc; ++ix) info.store(ix0+ix, iy, QFBase::hsum(acc[Qx::nrc*iy+ix]));
}

template <typename Qy, typename Qx>
inline void mul_mat_Qx_Qy_MxN_fa4(int D, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    static_assert(Qx::nrc%4 == 0);
    int nb = D/QFBase::k_step;
    Qy y(info);
    Qx x(cx + ix0*bx, bx);
    QFBase::Data xv[Qx::nrc];
    QFBase::Acc  acc[Qx::nrc*Qy::nrc/4] = {};
    for (int i = 0; i < nb; ++i) {
        for (int ix = 0; ix < Qx::nrc/4; ++ix) x.load_r4(4*ix, i, xv + 4*ix);
        for (int iy = 0; iy < Qy::nrc; ++iy) {
            auto yv = y.load1(iy, i);
            for (int ix = 0; ix < Qx::nrc/4; ++ix) acc[ix*Qy::nrc + iy] = QFBase::acc_r4(acc[ix*Qy::nrc + iy], xv + 4*ix, yv);
        }
    }
    for (int iy = 0; iy < Qy::nrc; ++iy) {
        for (int ix = 0; ix < Qx::nrc/4; ++ix) info.store(ix0+4*ix, iy, QFBase::hsum_r4(acc[ix*Qy::nrc + iy]));
    }
}

// This will handle any of f16 x f32, f32 x f16, f16 x f16, f32 x f32, with computations done
// in f32 (i.e., f16 is first converted to f32). It is easy to extend to computations done in
// f16, but I don't have a CPU capable of f16 vector arithmetic, so not doing it for now.
template <int nrc_y, typename FloatX, typename FloatY>
void mul_mat_fX_fY_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    const char * cx = (const char *)vx;
    // TBD if we want this
    //if constexpr (nrc_y == 1) {
    //    constexpr int k_nx = 2;
    //    for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
    //        mul_mat_Qx_Qy_Mx1<QFT<FloatY, nrc_y>, QFT<FloatX, k_nx>>(n, cx, bx, ix*k_nx, info);
    //    }
    //    if (int lastx = k_nx*(nrc_x/k_nx); lastx < nrc_x) {
    //        int nx = nrc_x - lastx;
    //        switch (nx) {
    //            case 1: mul_mat_Qx_Qy_Mx1<QFT<FloatY, nrc_y>, QFT<FloatX, 1>>(n, cx, bx, lastx, info); break;
    //            case 2: mul_mat_Qx_Qy_Mx1<QFT<FloatY, nrc_y>, QFT<FloatX, 2>>(n, cx, bx, lastx, info); break;
    //            case 3: mul_mat_Qx_Qy_Mx1<QFT<FloatY, nrc_y>, QFT<FloatX, 3>>(n, cx, bx, lastx, info); break;
    //        }
    //        //mul_mat_Qx_Qy_Mx1<QFT<FloatY, nrc_y>, QFT<FloatX, 1>>(n, cx, bx, lastx, info);
    //    }
    //    return;
    //}
#ifdef __AVX512F__
    constexpr int k_nx = 5;
#else
    constexpr int k_nx = nrc_y == 1 ? 4 : 2;
#endif
    for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
        mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, k_nx>>(n, cx, bx, ix*k_nx, info);
    }
    int last_x = k_nx*(nrc_x/k_nx);
    if (last_x == nrc_x) return;
    int nx = nrc_x - last_x;
#ifdef __AVX512F__
    switch (nx) {
        case 1: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 1>>(n, cx, bx, last_x, info); break;
        case 2: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 2>>(n, cx, bx, last_x, info); break;
        case 3: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 3>>(n, cx, bx, last_x, info); break;
        case 4: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 4>>(n, cx, bx, last_x, info); break;
    }
#else
    if constexpr (nrc_y == 1) {
        switch (nx) {
            case 1: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 1>>(n, cx, bx, last_x, info); break;
            case 2: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 2>>(n, cx, bx, last_x, info); break;
            case 3: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 3>>(n, cx, bx, last_x, info); break;
        }
    } else {
        switch (nx) {
            case 1: mul_mat_Qx_Qy_MxN<QFT<FloatY, nrc_y>, QFT<FloatX, 1>>(n, cx, bx, last_x, info); break;
        }
    }
#endif
}

#ifdef __AVX512BF16__
template <int nrc_y>
static void mul_mat_bf16_r16_bf16(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%16 == 0);
    const ggml_bf16_t * y[nrc_y];
    static_for<nrc_y>([&](const int iy) { y[iy] = (const ggml_bf16_t *)info.src1_row(iy); });

    for (int ix = 0; ix < nrc_x/32; ++ix) {
        __m512  acc[2*nrc_y] = {};
        __m512bh qx[8];
        const ggml_bf16_t * b8_1 = (const ggml_bf16_t *)((const char *)vx + (32*ix+ 0)*bx);
        const ggml_bf16_t * b8_2 = (const ggml_bf16_t *)((const char *)vx + (32*ix+16)*bx);
        for (int ib = 0; ib < n/8; ++ib) {
            qx[0] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_1+4*ib+0);
            qx[1] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_1+4*ib+1);
            qx[2] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_1+4*ib+2);
            qx[3] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_1+4*ib+3);
            qx[4] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_2+4*ib+0);
            qx[5] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_2+4*ib+1);
            qx[6] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_2+4*ib+2);
            qx[7] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8_2+4*ib+3);
            static_for<nrc_y>([&](const int iy) {
                auto y128 = _mm_loadu_si128((const __m128i*)y[iy]+ib);
                //auto y = _mm512_broadcast_i32x4(y128);
                auto y256 = MM256_SET_M128I(y128, y128);
                auto y = _mm512_inserti32x8(_mm512_castsi256_si512(y256), y256, 1);
                acc[2*iy+0] = _mm512_dpbf16_ps(acc[2*iy+0], qx[0], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
                acc[2*iy+0] = _mm512_dpbf16_ps(acc[2*iy+0], qx[1], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
                acc[2*iy+0] = _mm512_dpbf16_ps(acc[2*iy+0], qx[2], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
                acc[2*iy+0] = _mm512_dpbf16_ps(acc[2*iy+0], qx[3], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
                acc[2*iy+1] = _mm512_dpbf16_ps(acc[2*iy+1], qx[4], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
                acc[2*iy+1] = _mm512_dpbf16_ps(acc[2*iy+1], qx[5], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
                acc[2*iy+1] = _mm512_dpbf16_ps(acc[2*iy+1], qx[6], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
                acc[2*iy+1] = _mm512_dpbf16_ps(acc[2*iy+1], qx[7], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
            });
        }
        static_for<nrc_y>([&](const int iy) {
            info.store(32*ix+ 0, iy, acc[2*iy+0]);
            info.store(32*ix+16, iy, acc[2*iy+1]);
        });
    }
    for (int ix = 32*(nrc_x/32); ix < nrc_x; ix += 16) {
        __m512  acc[nrc_y] = {};
        __m512bh qx[4];
        const ggml_bf16_t * b8 = (const ggml_bf16_t *)((const char *)vx + (ix+0)*bx);
        for (int ib = 0; ib < n/8; ++ib) {
            qx[0] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8+4*ib+0);
            qx[1] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8+4*ib+1);
            qx[2] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8+4*ib+2);
            qx[3] = (__m512bh)_mm512_loadu_si512((const __m512i *)b8+4*ib+3);
            static_for<nrc_y>([&](const int iy) {
                auto y128 = _mm_loadu_si128((const __m128i*)y[iy]+ib);
                auto y256 = MM256_SET_M128I(y128, y128);
                auto y = _mm512_inserti32x8(_mm512_castsi256_si512(y256), y256, 1);
                acc[iy] = _mm512_dpbf16_ps(acc[iy], qx[0], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x00)));
                acc[iy] = _mm512_dpbf16_ps(acc[iy], qx[1], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0x55)));
                acc[iy] = _mm512_dpbf16_ps(acc[iy], qx[2], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xaa)));
                acc[iy] = _mm512_dpbf16_ps(acc[iy], qx[3], (__m512bh)_mm512_shuffle_epi32(y, _MM_PERM_ENUM(0xff)));
            });
        }
        static_for<nrc_y>([&](const int iy) {
            info.store(ix, iy, acc[iy]);
        });
    }
}

struct QFBaseBF16 {
    constexpr static int k_step = 32;
    using Data = __m512bh;
    using Acc  = __m512;
    static inline Data load(const ggml_bf16_t * x) { return __m512bh(_mm512_loadu_si512((const __m512i *)x)); }
    static inline Acc acc(Acc prev, Data y, Data x) {
        return _mm512_dpbf16_ps(prev, y, x);
    }
    static inline Acc acc_first(const Data& y, const Data& x) {
        return _mm512_dpbf16_ps(_mm512_setzero_ps(), y, x);
    }
    static inline float hsum(Acc acc) {
        return _mm512_reduce_add_ps(acc);
    }
};
template <int nrc_in> struct QFTBF16 final : public QFBaseBF16 {
    constexpr static int nrc = nrc_in;
    QFTBF16(const DataInfo& info) {
        for (int iy = 0; iy < nrc; ++iy) y[iy] = (const ggml_bf16_t *)info.src1_row(iy);
    }
    QFTBF16(const char * cx, size_t bx) {
        for (int iy = 0; iy < nrc; ++iy) y[iy] = (const ggml_bf16_t *)(cx + iy*bx);
    }
    IQK_ALWAYS_INLINE Data load1(int iy, int i) const { return load(y[iy] + k_step*i); }
    const ggml_bf16_t * y[nrc];
};

template <int nrc_y, int nrc_x>
IQK_NOINLINE void mul_mat_Qx_Qy_MxN(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    int nb = n/QFBaseBF16::k_step;
    QFTBF16<nrc_y> y(info);
    QFTBF16<nrc_x> x(cx + ix0*bx, bx);
    QFBaseBF16::Data xv[nrc_x];
    QFBaseBF16::Acc  acc[nrc_x*nrc_y];
    auto yv = y.load1(0, 0);
    for (int ix = 0; ix < nrc_x; ++ix) {
        xv[ix] = x.load1(ix, 0);
        acc[ix] = QFBaseBF16::acc_first(yv, xv[ix]);
    }
    for (int iy = 1; iy < nrc_y; ++iy) {
        yv = y.load1(iy, 0);
        for (int ix = 0; ix < nrc_x; ++ix) acc[nrc_x*iy + ix] = QFBaseBF16::acc_first(yv, xv[ix]);
    }
    for (int i = 1; i < nb; ++i) {
        yv = y.load1(0, i);
        for (int ix = 0; ix < nrc_x; ++ix) {
            xv[ix] = x.load1(ix, i);
            acc[ix] = QFBaseBF16::acc(acc[ix], yv, xv[ix]);
        }
        for (int iy = 1; iy < nrc_y; ++iy) {
            yv = y.load1(iy, i);
            for (int ix = 0; ix < nrc_x; ++ix) acc[nrc_x*iy + ix] = QFBaseBF16::acc(acc[nrc_x*iy + ix], yv, xv[ix]);
        }
    }
    for (int iy = 0; iy < nrc_y; ++iy) for (int ix = 0; ix < nrc_x; ++ix) info.store(ix0+ix, iy, QFBaseBF16::hsum(acc[nrc_x*iy+ix]));
}

template <int nrc_y>
void mul_mat_fX_fY_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    constexpr int k_nx = nrc_y <= 2 ? 8 : 5;
    const char * cx = (const char *)vx;
    for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
        mul_mat_Qx_Qy_MxN<nrc_y, k_nx>(n, cx, bx, ix*k_nx, info);
    }
    int last_x = k_nx*(nrc_x/k_nx);
    if (last_x == nrc_x) return;
    int nx = nrc_x - last_x;
    if constexpr (nrc_y <= 2) {
        if (nx >= 4) {
            mul_mat_Qx_Qy_MxN<nrc_y, 4>(n, cx, bx, last_x, info);
            last_x += 4;
            if (last_x == nrc_x) return;
            nx = nrc_x - last_x;
        }
    }
    switch (nx) {
        case 1: mul_mat_Qx_Qy_MxN<nrc_y, 1>(n, cx, bx, last_x, info); break;
        case 2: mul_mat_Qx_Qy_MxN<nrc_y, 2>(n, cx, bx, last_x, info); break;
        case 3: mul_mat_Qx_Qy_MxN<nrc_y, 3>(n, cx, bx, last_x, info); break;
        case 4: mul_mat_Qx_Qy_MxN<nrc_y, 4>(n, cx, bx, last_x, info); break;
    }
}
#endif


template <typename FloatX, typename FloatY>
void set_mul_mat_f(std::array<mul_mat_t, IQK_MAX_NY>& funcs) {
    for (auto& f : funcs) f = nullptr;
    funcs[0] = mul_mat_fX_fY_T<1, FloatX, FloatY>;
    funcs[1] = mul_mat_fX_fY_T<2, FloatX, FloatY>;
    funcs[2] = mul_mat_fX_fY_T<3, FloatX, FloatY>;
    funcs[3] = mul_mat_fX_fY_T<4, FloatX, FloatY>;
    funcs[4] = mul_mat_fX_fY_T<5, FloatX, FloatY>;
#ifndef __AVX512F__
    funcs[5] = mul_mat_fX_fY_T<6, FloatX, FloatY>;
#endif
}

#ifdef __AVX512BF16__
void set_mul_mat_bf16(std::array<mul_mat_t, IQK_MAX_NY>& funcs) {
    for (auto& f : funcs) f = nullptr;
    funcs[0] = mul_mat_fX_fY_T<1>;
    funcs[1] = mul_mat_fX_fY_T<2>;
    funcs[2] = mul_mat_fX_fY_T<3>;
    funcs[3] = mul_mat_fX_fY_T<4>;
    funcs[4] = mul_mat_fX_fY_T<5>;
}
void set_mul_mat_bf16_r16(std::array<mul_mat_t, IQK_MAX_NY>& funcs) {
    for (auto& f : funcs) f = nullptr;
    funcs[0] = mul_mat_bf16_r16_bf16<1>;
    funcs[1] = mul_mat_bf16_r16_bf16<2>;
    funcs[2] = mul_mat_bf16_r16_bf16<3>;
    funcs[3] = mul_mat_bf16_r16_bf16<4>;
    funcs[4] = mul_mat_bf16_r16_bf16<5>;
    funcs[5] = mul_mat_bf16_r16_bf16<6>;
    funcs[6] = mul_mat_bf16_r16_bf16<7>;
    funcs[7] = mul_mat_bf16_r16_bf16<8>;
}
#endif

} // namespace

bool iqk_set_kernels_float(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels) {

    if (typeA == GGML_TYPE_BF16) {
        if (ne00 % 32) return false;
        switch (typeB) {
#ifdef __AVX512BF16__
            case GGML_TYPE_BF16: set_mul_mat_bf16(kernels); break;
#else
            case GGML_TYPE_BF16: set_mul_mat_f<ggml_bf16_t, ggml_bf16_t>(kernels); break;
            case GGML_TYPE_F32:  set_mul_mat_f<ggml_bf16_t, float>(kernels);       break;
#endif
            default: return false;
        }
        return true;
    }

    if (typeA == GGML_TYPE_BF16_R16) {
        if (ne00 % 16) return false;
        switch (typeB) {
#ifdef __AVX512BF16__
            case GGML_TYPE_BF16: set_mul_mat_bf16_r16(kernels); break;
#endif
            default: return false;
        }
        return true;
    }

    if (typeA == GGML_TYPE_F16 || typeA == GGML_TYPE_F32) {
        if (ne00 % 4) return false;
    }
    if (typeA == GGML_TYPE_F16) {
        switch (typeB) {
            case GGML_TYPE_F16: set_mul_mat_f<ggml_half, ggml_half>(kernels); break;
            case GGML_TYPE_F32: set_mul_mat_f<ggml_half, float>(kernels);     break;
            default: return false;
        }
        return true;
    }
    if (typeA == GGML_TYPE_F32) {
        switch (typeB) {
            case GGML_TYPE_F16: set_mul_mat_f<float, ggml_half>(kernels); break;
            case GGML_TYPE_F32: set_mul_mat_f<float, float>(kernels);     break;
            default: return false;
        }
        return true;
    }

    return false;

}

void iqk_gemm_default_floats(int D, int nq, const char * cx, size_t bx, DataInfo& info, int k_step) {
    using q_float = float;
#ifdef HAVE_FANCY_SIMD
    constexpr int nrc_q = 8;
    constexpr int nrc_k = 8;
#else
    // somewhat surprisingly, nrc_q = 4, nrc_k = 8 is better than nrc_q = 8, nrc_k = 4
    constexpr int nrc_q = 4;
    constexpr int nrc_k = 8;
#endif
    GGML_ASSERT(k_step%nrc_k == 0);
    int qrem = nq - nrc_q*(nq/nrc_q);
    for (int iq = 0; iq < nq/nrc_q; ++iq) {
        for (int ik = 0; ik < k_step/nrc_k; ++ik) {
            mul_mat_Qx_Qy_MxN_fa4<QFT<float, nrc_q>, QFT<ggml_half, nrc_k>>(D, cx, bx, ik*nrc_k, info);
        }
        info.cur_y += nrc_q;
    }
    if (qrem > 0) {
        switch (qrem) {
            case 1: {
                for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                    mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 1>, QFT<ggml_half, nrc_k>>(D, cx, bx, ik*nrc_k, info);
                }
            } break;
            case 2: {
                for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                    mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 2>, QFT<ggml_half, nrc_k>>(D, cx, bx, ik*nrc_k, info);
                }
            } break;
            case 3: {
                for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                    mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 3>, QFT<ggml_half, nrc_k>>(D, cx, bx, ik*nrc_k, info);
                }
            } break;
#ifdef HAVE_FANCY_SIMD
            case 4: {
                for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                    mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 4>, QFT<ggml_half, nrc_k>>(D, cx, bx, ik*nrc_k, info);
                }
            } break;
            case 5: {
                for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                    mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 5>, QFT<ggml_half, nrc_k>>(D, cx, bx, ik*nrc_k, info);
                }
            } break;
            case 6: {
                for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                    mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 6>, QFT<ggml_half, nrc_k>>(D, cx, bx, ik*nrc_k, info);
                }
            } break;
            case 7: {
                for (int ik = 0; ik < k_step/nrc_k; ++ik) {
                    mul_mat_Qx_Qy_MxN_fa4<QFT<q_float, 7>, QFT<ggml_half, nrc_k>>(D, cx, bx, ik*nrc_k, info);
                }
            } break;
#endif
        }
    }
}

#else
// ----------------------------------- __aarch64__ -----------------------------------------------

namespace {

struct QF16Base {
    constexpr static int k_step = 8;
    using Data = float16x8_t;
    using Acc  = float16x8_t;
    static inline Data load(const __fp16 * x) { return vld1q_f16(x); }
    static inline Data load4(const __fp16 * x) { return vcombine_f16(vld1_f16(x), vdup_n_f16(0)); }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) {
        return vfmaq_f16(prev, y, x);
    }
    static inline Acc acc_first(const Data& y, const Data& x) {
        return vmulq_f16(y, x);
    }
    //constexpr static int k_step = 16;
    //using Data = float16x8x2_t;
    //static inline Data load(const __fp16 * x) { return vld1q_f16_x2(x); }
    //static inline Acc acc(Acc prev, const Data& y, const Data& x) {
    //    return vfmaq_f16(vfmaq_f16(prev, y.val[0], x.val[0]), y.val[1], x.val[1]);
    //}
    //static inline Acc acc_first(const Data& y, const Data& x) {
    //    return vfmaq_f16(vmulq_f16(y.val[0], x.val[0]), y.val[1], x.val[1]);
    //}
    static inline float hsum(Acc acc) {
        float32x4_t sum = vcvt_f32_f16(vadd_f16(vget_low_f16(acc), vget_high_f16(acc)));
        return vaddvq_f32(sum);
    }
};
template <int nrc> struct QF16 final : public QF16Base {
    using Base = QF16Base;
    constexpr static int nrc_y = nrc;
    QF16(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const __fp16 *)info.src1_row(iy);
    }
    QF16(const char * cx, size_t bx) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const __fp16 *)(cx + iy*bx);
    }
    IQK_ALWAYS_INLINE Data load1(int iy, int i) const { return load(y[iy] + k_step*i); }
    IQK_ALWAYS_INLINE Data load_tail(int iy, int i) const { return load4(y[iy] + 4*i); }
    IQK_ALWAYS_INLINE float16x8x4_t loadx(int iy, int i) const { return vld1q_f16_x4(y[iy] + 4*k_step*i); }
    const __fp16 * y[nrc_y];
};

struct QBF16Base {
    constexpr static int k_step = 4;
    using Data = float32x4_t;
    using Acc  = float32x4_t;
    static inline Data load(const uint16_t * x) { return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vld1_u16(x)), 16)); }
    static inline Data load4(const uint16_t * x) { return load(x); }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) {
        return vfmaq_f32(prev, y, x);
    }
    static inline Acc acc_first(const Data& y, const Data& x) {
        return vmulq_f32(y, x);
    }
    static inline float hsum(Acc acc) { return vaddvq_f32(acc); }
};
template <int nrc> struct QBF16 final : public QBF16Base {
    using Base = QBF16Base;
    constexpr static int nrc_y = nrc;
    QBF16(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const uint16_t *)info.src1_row(iy);
    }
    QBF16(const char * cx, size_t bx) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const uint16_t *)(cx + iy*bx);
    }
    IQK_ALWAYS_INLINE Data load1(int iy, int i) const { return load(y[iy] + k_step*i); }
    IQK_ALWAYS_INLINE Data load_tail(int iy, int i) const { return load(y[iy] + 4*i); }
    const uint16_t * y[nrc_y];
};

struct QF32Base {
    constexpr static int k_step = 4;
    using Data = float32x4_t;
    using Acc  = float32x4_t;
    static inline Data load(const float * x) { return vld1q_f32(x); }
    static inline Data load4(const float * x) { return load(x); }
    static inline Acc acc(Acc prev, const Data& y, const Data& x) { return vfmaq_f32(prev, y, x); }
    static inline Acc acc_first(const Data& y, const Data& x) { return vmulq_f32(y, x); }
    static inline float hsum(Acc acc) { return vaddvq_f32(acc); }
};
template <int nrc> struct QF32 final : public QF32Base {
    using Base = QF32Base;
    constexpr static int nrc_y = nrc;
    QF32(const DataInfo& info) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const float *)info.src1_row(iy);
    }
    QF32(const char * cx, size_t bx) {
        for (int iy = 0; iy < nrc_y; ++iy) y[iy] = (const float *)(cx + iy*bx);
    }
    IQK_ALWAYS_INLINE Data load1(int iy, int i) const { return load(y[iy] + k_step*i); }
    IQK_ALWAYS_INLINE Data load_tail(int iy, int i) const { return load(y[iy] + 4*i); }
    const float * y[nrc_y];
};

template <typename Qy, typename Qx, bool is_multiple_of_k_step>
IQK_NOINLINE void mul_mat_Qx_Qy_NxN(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    GGML_ASSERT(Qx::Base::k_step == Qy::Base::k_step);
    int nb = n/Qx::Base::k_step;
    Qy y(info);
    Qx x(cx + ix0*bx, bx);
    typename Qx::Base::Data xv[Qx::nrc_y];
    typename Qx::Base::Acc  acc[Qx::nrc_y*Qy::nrc_y];
    auto yv = y.load1(0, 0);
    for (int ix = 0; ix < Qx::nrc_y; ++ix) {
        xv[ix] = x.load1(ix, 0);
        acc[ix] = Qx::Base::acc_first(yv, xv[ix]);
    }
    for (int iy = 1; iy < Qy::nrc_y; ++iy) {
        yv = y.load1(iy, 0);
        for (int ix = 0; ix < Qx::nrc_y; ++ix) acc[Qx::nrc_y*iy + ix] = Qx::Base::acc_first(yv, xv[ix]);
    }
    for (int i = 1; i < nb; ++i) {
        yv = y.load1(0, i);
        for (int ix = 0; ix < Qx::nrc_y; ++ix) {
            xv[ix] = x.load1(ix, i);
            acc[ix] = Qx::Base::acc(acc[ix], yv, xv[ix]);
        }
        for (int iy = 1; iy < Qy::nrc_y; ++iy) {
            yv = y.load1(iy, i);
            for (int ix = 0; ix < Qx::nrc_y; ++ix) acc[Qx::nrc_y*iy + ix] = Qx::Base::acc(acc[Qx::nrc_y*iy + ix], yv, xv[ix]);
        }
    }
    if constexpr (Qx::Base::k_step > 4 && !is_multiple_of_k_step) {
        int nb4 = n/4;
        for (int i = (Qx::Base::k_step/4)*nb; i < nb4; ++i) {
            yv = y.load_tail(0, i);
            for (int ix = 0; ix < Qx::nrc_y; ++ix) {
                xv[ix] = x.load_tail(ix, i);
                acc[ix] = Qx::Base::acc(acc[ix], yv, xv[ix]);
            }
            for (int iy = 1; iy < Qy::nrc_y; ++iy) {
                yv = y.load_tail(iy, i);
                for (int ix = 0; ix < Qx::nrc_y; ++ix) acc[Qx::nrc_y*iy + ix] = Qx::Base::acc(acc[Qx::nrc_y*iy + ix], yv, xv[ix]);
            }
        }
    }
    for (int iy = 0; iy < Qy::nrc_y; ++iy) for (int ix = 0; ix < Qx::nrc_y; ++ix) info.store(ix0+ix, iy, Qx::Base::hsum(acc[Qx::nrc_y*iy+ix]));
}

template <int nrc_y, int nrc_x, bool is_multiple_of_k_step>
IQK_NOINLINE void mul_mat_f16_f16_NxN(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    assert(n%QF16Base::k_step == 0);
    int nb = n/QF16Base::k_step;
    QF16<nrc_y> y(info);
    QF16<nrc_x> x(cx + ix0*bx, bx);
    QF16Base::Data xv[nrc_x];
    QF16Base::Acc  acc[nrc_x*nrc_y];
    auto yv = y.load1(0, 0);
    for (int ix = 0; ix < nrc_x; ++ix) {
        xv[ix] = x.load1(ix, 0);
        acc[ix] = QF16Base::acc_first(yv, xv[ix]);
    }
    for (int iy = 1; iy < nrc_y; ++iy) {
        yv = y.load1(iy, 0);
        for (int ix = 0; ix < nrc_x; ++ix) acc[nrc_x*iy + ix] = QF16Base::acc_first(yv, xv[ix]);
    }
    for (int i = 1; i < nb; ++i) {
        yv = y.load1(0, i);
        for (int ix = 0; ix < nrc_x; ++ix) {
            xv[ix] = x.load1(ix, i);
            acc[ix] = QF16Base::acc(acc[ix], yv, xv[ix]);
        }
        for (int iy = 1; iy < nrc_y; ++iy) {
            yv = y.load1(iy, i);
            for (int ix = 0; ix < nrc_x; ++ix) acc[nrc_x*iy + ix] = QF16Base::acc(acc[nrc_x*iy + ix], yv, xv[ix]);
        }
    }
    if constexpr (!is_multiple_of_k_step) {
        int nb4 = n/4;
        for (int i = (QF16Base::k_step/4)*nb; i < nb4; ++i) {
            yv = y.load_tail(0, i);
            for (int ix = 0; ix < nrc_x; ++ix) {
                xv[ix] = x.load_tail(ix, i);
                acc[ix] = QF16Base::acc(acc[ix], yv, xv[ix]);
            }
            for (int iy = 1; iy < nrc_y; ++iy) {
                yv = y.load_tail(iy, i);
                for (int ix = 0; ix < nrc_x; ++ix) acc[nrc_x*iy + ix] = QF16Base::acc(acc[nrc_x*iy + ix], yv, xv[ix]);
            }
        }
    }
    for (int iy = 0; iy < nrc_y; ++iy) for (int ix = 0; ix < nrc_x; ++ix) info.store(ix0+ix, iy, QF16Base::hsum(acc[nrc_x*iy+ix]));
}

template <typename Qy, template<int> typename Qx>
void mul_mat_Qx_Qy_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%4 == 0);
    constexpr int k_nx = 5;
    const char * cx = (const char *)vx;
    if (n%Qx<k_nx>::Base::k_step == 0) {
        for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
            mul_mat_Qx_Qy_NxN<Qy, Qx<k_nx>, true>(n, cx, bx, ix*k_nx, info);
        }
        int last_x = k_nx*(nrc_x/k_nx);
        if (last_x == nrc_x) return;
        int nx = nrc_x - last_x;
        switch (nx) {
            case 1: mul_mat_Qx_Qy_NxN<Qy, Qx<1>, true>(n, cx, bx, last_x, info); break;
            case 2: mul_mat_Qx_Qy_NxN<Qy, Qx<2>, true>(n, cx, bx, last_x, info); break;
            case 3: mul_mat_Qx_Qy_NxN<Qy, Qx<3>, true>(n, cx, bx, last_x, info); break;
            case 4: mul_mat_Qx_Qy_NxN<Qy, Qx<4>, true>(n, cx, bx, last_x, info); break;
        }
    } else {
        for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
            mul_mat_Qx_Qy_NxN<Qy, Qx<k_nx>, false>(n, cx, bx, ix*k_nx, info);
        }
        int last_x = k_nx*(nrc_x/k_nx);
        if (last_x == nrc_x) return;
        int nx = nrc_x - last_x;
        switch (nx) {
            case 1: mul_mat_Qx_Qy_NxN<Qy, Qx<1>, false>(n, cx, bx, last_x, info); break;
            case 2: mul_mat_Qx_Qy_NxN<Qy, Qx<2>, false>(n, cx, bx, last_x, info); break;
            case 3: mul_mat_Qx_Qy_NxN<Qy, Qx<3>, false>(n, cx, bx, last_x, info); break;
            case 4: mul_mat_Qx_Qy_NxN<Qy, Qx<4>, false>(n, cx, bx, last_x, info); break;
        }
    }
}

template <int nrc_y>
void mul_mat_f16_f16_T(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%4 == 0);
    constexpr int k_nx = 5;
    const char * cx = (const char *)vx;
    if (n%QF16Base::k_step == 0) {
        for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
            mul_mat_f16_f16_NxN<nrc_y, k_nx, true>(n, cx, bx, ix*k_nx, info);
        }
        int last_x = k_nx*(nrc_x/k_nx);
        if (last_x == nrc_x) return;
        int nx = nrc_x - last_x;
        switch (nx) {
            case 1: mul_mat_f16_f16_NxN<nrc_y, 1, true>(n, cx, bx, last_x, info); break;
            case 2: mul_mat_f16_f16_NxN<nrc_y, 2, true>(n, cx, bx, last_x, info); break;
            case 3: mul_mat_f16_f16_NxN<nrc_y, 3, true>(n, cx, bx, last_x, info); break;
            case 4: mul_mat_f16_f16_NxN<nrc_y, 4, true>(n, cx, bx, last_x, info); break;
        }
    } else {
        for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
            mul_mat_f16_f16_NxN<nrc_y, k_nx, false>(n, cx, bx, ix*k_nx, info);
        }
        int last_x = k_nx*(nrc_x/k_nx);
        if (last_x == nrc_x) return;
        int nx = nrc_x - last_x;
        switch (nx) {
            case 1: mul_mat_f16_f16_NxN<nrc_y, 1, false>(n, cx, bx, last_x, info); break;
            case 2: mul_mat_f16_f16_NxN<nrc_y, 2, false>(n, cx, bx, last_x, info); break;
            case 3: mul_mat_f16_f16_NxN<nrc_y, 3, false>(n, cx, bx, last_x, info); break;
            case 4: mul_mat_f16_f16_NxN<nrc_y, 4, false>(n, cx, bx, last_x, info); break;
        }
    }
}

template <int nrc_x, bool is_multiple_of_k_step>
IQK_NOINLINE void mul_mat_f16_f16_Nx1(int n, const char * cx, size_t bx, int ix0, const DataInfo& info) {
    assert(n%QF16Base::k_step == 0);
    int nb = n/QF16Base::k_step;
    QF16<1> y(info);
    QF16<nrc_x> x(cx + ix0*bx, bx);
    QF16Base::Acc  acc[4*nrc_x];
    auto yv = y.loadx(0, 0);
    for (int ix = 0; ix < nrc_x; ++ix) {
        for (int k = 0; k < 4; ++k) {
            auto xv = x.load1(ix, k);
            acc[4*ix+k] = QF16Base::acc_first(yv.val[k], xv);
        }
    }
    for (int i = 1; i < nb/4; ++i) {
        yv = y.loadx(0, i);
        for (int ix = 0; ix < nrc_x; ++ix) {
            for (int k = 0; k < 4; ++k) {
                auto xv = x.load1(ix, 4*i+k);
                acc[4*ix+k] = QF16Base::acc(acc[4*ix+k], yv.val[k], xv);
            }
        }
    }
    for (int i = 4*(nb/4); i < nb; ++i) {
        auto yv1 = y.load1(0, i);
        for (int ix = 0; ix < nrc_x; ++ix) {
            auto xv1 = x.load1(ix, i);
            acc[4*ix] = QF16Base::acc(acc[4*ix], yv1, xv1);
        }
    }
    if constexpr (!is_multiple_of_k_step) {
        int nb4 = n/4;
        for (int i = (QF16Base::k_step/4)*nb; i < nb4; ++i) {
            auto yv1 = y.load_tail(0, i);
            for (int ix = 0; ix < nrc_x; ++ix) {
                auto xv1 = x.load_tail(ix, i);
                acc[4*ix] = QF16Base::acc(acc[4*ix], yv1, xv1);
            }
        }
    }
    for (int ix = 0; ix < nrc_x; ++ix) {
        auto v1 = vaddq_f16(acc[4*ix+0], acc[4*ix+1]);
        auto v2 = vaddq_f16(acc[4*ix+2], acc[4*ix+3]);
        info.store(ix0+ix, 0, QF16Base::hsum(vaddq_f16(v1, v2)));
    }
}

// At least on my M2-Max the version below, which does the multiplication row-by-row, is faster.
// But let's keep this version commented out for now.
//void mul_mat_f16_f16_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
//    GGML_ASSERT(n%4 == 0);
//    constexpr int k_nx = 2;
//    const char * cx = (const char *)vx;
//    if (n%QF16Base::k_step == 0) {
//        for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
//            mul_mat_f16_f16_Nx1<k_nx, true>(n, cx, bx, ix*k_nx, info);
//        }
//        int last_x = k_nx*(nrc_x/k_nx);
//        if (last_x == nrc_x) return;
//        int nx = nrc_x - last_x;
//        switch (nx) {
//            case 1: mul_mat_f16_f16_Nx1<1, true>(n, cx, bx, last_x, info); break;
//            //case 2: mul_mat_f16_f16_Nx1<2, true>(n, cx, bx, last_x, info); break;
//            //case 3: mul_mat_f16_f16_Nx1<3, true>(n, cx, bx, last_x, info); break;
//        }
//    } else {
//        for (int ix = 0; ix < nrc_x/k_nx; ++ix) {
//            mul_mat_f16_f16_Nx1<k_nx, false>(n, cx, bx, ix*k_nx, info);
//        }
//        int last_x = k_nx*(nrc_x/k_nx);
//        if (last_x == nrc_x) return;
//        int nx = nrc_x - last_x;
//        switch (nx) {
//            case 1: mul_mat_f16_f16_Nx1<1, false>(n, cx, bx, last_x, info); break;
//            //case 2: mul_mat_f16_f16_Nx1<2, false>(n, cx, bx, last_x, info); break;
//            //case 3: mul_mat_f16_f16_Nx1<3, false>(n, cx, bx, last_x, info); break;
//        }
//    }
//}

void mul_mat_f16_f16_1(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(n%4 == 0);
    const char * cx = (const char *)vx;
    if (n%QF16Base::k_step == 0) {
        for (int ix = 0; ix < nrc_x; ++ix) {
            mul_mat_f16_f16_Nx1<1, true>(n, cx, bx, ix, info);
        }
    } else {
        for (int ix = 0; ix < nrc_x; ++ix) {
            mul_mat_f16_f16_Nx1<1, false>(n, cx, bx, ix, info);
        }
    }
}

}

bool iqk_set_kernels_float(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels) {

    if (ne00%4 == 0) {

        if (typeA == GGML_TYPE_F16 && typeB == GGML_TYPE_F16) {
            for (auto& f : kernels) f = nullptr;
            kernels[0] = mul_mat_f16_f16_1;
            kernels[1] = mul_mat_f16_f16_T<2>;
            kernels[2] = mul_mat_f16_f16_T<3>;
            kernels[3] = mul_mat_f16_f16_T<4>;
            kernels[4] = mul_mat_f16_f16_T<5>;
            return true;
        }
        else if (typeA == GGML_TYPE_BF16 && typeB == GGML_TYPE_F32) {
            for (auto& f : kernels) f = nullptr;
            kernels[0] = mul_mat_Qx_Qy_T<QF32<1>, QBF16>;
            kernels[1] = mul_mat_Qx_Qy_T<QF32<2>, QBF16>;
            kernels[2] = mul_mat_Qx_Qy_T<QF32<3>, QBF16>;
            kernels[3] = mul_mat_Qx_Qy_T<QF32<4>, QBF16>;
            kernels[4] = mul_mat_Qx_Qy_T<QF32<5>, QBF16>;
            return true;
        }

    }

    return false;

}

namespace {
template <int nrc_q>
inline void mm_helper(int D, int nq, const char * cx, size_t bx, DataInfo& info, int k_step) {
    constexpr int nrc_k = 6;
    int krem = k_step - nrc_k*(k_step/nrc_k);
    for (int iq = 0; iq < nq/nrc_q; ++iq) {
        for (int ik = 0; ik < k_step/nrc_k; ++ik) {
            mul_mat_f16_f16_NxN<nrc_q, nrc_k, true>(D, cx, bx, ik*nrc_k, info);
        }
        if (krem > 0) {
            switch (krem) {
                case  1: mul_mat_f16_f16_NxN<nrc_q, 1, true>(D, cx, bx, k_step - krem, info); break;
                case  2: mul_mat_f16_f16_NxN<nrc_q, 2, true>(D, cx, bx, k_step - krem, info); break;
                case  3: mul_mat_f16_f16_NxN<nrc_q, 3, true>(D, cx, bx, k_step - krem, info); break;
                case  4: mul_mat_f16_f16_NxN<nrc_q, 4, true>(D, cx, bx, k_step - krem, info); break;
                default: mul_mat_f16_f16_NxN<nrc_q, 5, true>(D, cx, bx, k_step - krem, info); break;
            }
        }
        info.cur_y += nrc_q;
    }
}
}

void iqk_gemm_default_floats(int D, int nq, const char * cx, size_t bx, DataInfo& info, int k_step) {
    constexpr int nrc_q = 4;
    mm_helper<nrc_q>(D, nq, cx, bx, info, k_step);
    if (int qrem = nq - nrc_q*(nq/nrc_q); qrem > 0) {
        switch (qrem) {
            case  1: mm_helper<1>(D, nq, cx, bx, info, k_step);
            case  2: mm_helper<2>(D, nq, cx, bx, info, k_step);
            default: mm_helper<3>(D, nq, cx, bx, info, k_step);
        }
    }
}

#endif

#endif
