### 🔀 [#57](https://github.com/ikawrakow/ik_llama.cpp/pull/57) - AVX2/Zen4 horizontal sums 

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2024-09-17 |
| **Updated** | 2024-09-17 |

---

#### Description

It is really strange that there is no instruction to horizontally sum the elements of a SIMD vector in `AVX/AVX2/AVX512` as this is needed all the time. In `AVX512` there is `_mm512_reduce_add_ps(x)`, but this expands to multiple instructions. E.g., from GCC-12 `immintrin.h`:
```
#undef __MM512_REDUCE_OP
#define __MM512_REDUCE_OP(op) \
  __m256 __T1 = (__m256) _mm512_extractf64x4_pd ((__m512d) __A, 1); \
  __m256 __T2 = (__m256) _mm512_extractf64x4_pd ((__m512d) __A, 0); \
  __m256 __T3 = __T1 op __T2;                       \
  __m128 __T4 = _mm256_extractf128_ps (__T3, 1);            \
  __m128 __T5 = _mm256_extractf128_ps (__T3, 0);            \
  __m128 __T6 = __T4 op __T5;                       \
  __m128 __T7 = __builtin_shuffle (__T6, (__v4si) { 2, 3, 0, 1 });  \
  __m128 __T8 = __T6 op __T7;                       \
  return __T8[0] op __T8[1]

extern __inline float 
__attribute__ ((__gnu_inline__, __always_inline__, __artificial__))
_mm512_reduce_add_ps (__m512 __A)
{
  __MM512_REDUCE_OP (+);
}
```
On `AVX2` I have been using
```
 inline float hsum_float_4(__m128 x) { 
    x = _mm_add_ps(x, _mm_movehl_ps(x, x)); 
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
inline float hsum_float_8(__m256 x) { 
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}
```
i.e., 8 instructions to sum 8 float elements. I have been wondering to what extend this affects the performance of matrix-matrix and matrix-vector multiplications as it needs to get done for every element of the resulting matrix/vector.

In `iqk_mul_mat` most matrix-matrix multiplications are done by simultaneously processing 8 columns of the right matrix, so we end up with 8 SIMD vectors containing the dot products of a row from the left matrix with the 8 columns. In this case it is possible to have a more efficient implementation where we end up with a single SIMD vector containing the horizontal sums of the 8 SIMD vectors like this
```
inline __m256 hsum_float_8x8(__m256 * accm) {
    for (int i = 0; i < 4; ++i) {
        accm[i] = _mm256_set_m128(_mm_add_ps(_mm256_castps256_ps128(accm[i+4]), _mm256_extractf128_ps(accm[i+4], 1)),
                                  _mm_add_ps(_mm256_castps256_ps128(accm[i+0]), _mm256_extractf128_ps(accm[i+0], 1)));
    }
    for (int i = 0; i < 2; ++i) accm[i] = _mm256_add_ps(_mm256_unpacklo_ps(accm[i], accm[i+2]), _mm256_unpackhi_ps(accm[i], accm[i+2]));
    return _mm256_add_ps(_mm256_unpacklo_ps(accm[0], accm[1]), _mm256_unpackhi_ps(accm[0], accm[1]));
}
```
I count 29 instructions, so less than 4 instructions per horizontal sum.

Plugging this into `iqk_mul_mat` results in 1-2% performance improvements for basically all quantized matrix-matrix multiplications (float multiplications are done with 5x5 tiles on Zen4, so this idea is not directly or easily transferable). Strangely enough, on a pure `AVX2` system (Ryzen-5975WX), I observe 1-2% reduced performance, hence in this PR the 8x8 sum is only used on Zen4.

One can also apply the idea to matrix-vector multiplications by simply gathering 8 dot products and then using the 8x8 horizontal sum. This is relevant for TG. TG is severly memory-bound on `x86_64` systems, so there is no benefit when using the number of threads that results in peak performance. But with just 1 thread I observe up to 10% speedup on Zen4.