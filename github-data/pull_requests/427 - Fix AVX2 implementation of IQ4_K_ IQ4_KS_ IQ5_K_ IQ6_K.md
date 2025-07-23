### 🐛 [#427](https://github.com/ikawrakow/ik_llama.cpp/pull/427) - Fix AVX2 implementation of IQ4_K, IQ4_KS, IQ5_K, IQ6_K

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-16 |
| **Updated** | 2025-05-16 |

---

#### Description

I have made the exact same mistake a number of times.

On `AVX2` the instruction to perform dot products of `int8_t` vectors (as needed in quantized matrix multiplications) is `_mm256_maddubs_epi8(x, y)`, where `x` must be unsigned and `y` signed, and the result is a SIMD vector of signed `int16_t` values $z_i = x_{2i} y_{2i} + x_{2i+1} y_{2i+1}$. The quant values `x` and quantized activations `y` are signed, so one way to deal with the the strangeness of this instruction is to add a suitable constant value `c` to `x` so that it becomes unsigned, use `_mm256_maddubs_epi8(c+x, y)` to accumulate the dot product, and at the end subtract $c \cdot b$, where $b = \sum y_i$ has been pre-computed when quantizing the activations `y`. The issue arises when the `x` values span the full `int8_t` range as it is the case with the non-linear quants `IQ4_NL, IQ4_XS, IQ4_K, IQ4_KS, IQ5_K, IQ5_KS, IQ6_K`. In that case `c = 128`, the `c+x` values span the full `uint8_t` range, and hence it is possible to overflow the signed `int16_t` range.

I had though that I had fixed this mistake, but while working on the `IQ5_KS` type added in PR #422 I noticed that the issue still exists `IQ4_K, IQ4_KS, IQ5_K, IQ6_K` and was only fixed for the corresponding repacked variants.

The PR corrects the problem. There will be a slight (a few percent) PP performance degradation on `AVX2` for these quantization types.