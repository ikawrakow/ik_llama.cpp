### 🔀 [#515](https://github.com/ikawrakow/ik_llama.cpp/pull/515) - IQ2_XXS: much faster CPU prompt processing

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-11 |
| **Updated** | 2025-06-11 |

---

#### Description

While experimenting with the trellis quants in PRs #505 and #511, I realized that CPU matrix multiplications (GEMM) for quants that are slow to unpack and make ready for `int8_t` dot products (as the trellis quants are) are much faster if one unpacks a given number of rows to, e.g., `Q8_0_R8`, and then uses the `Q8_0_R8 x Q8_2_X4` GEMM to perform the multiplication with **all columns** of the right matrix.

This PR applies the approach of #505/#511 to `IQ2_XXS` (`AVX2/Zen4`  only). We get nearly 3X improvement in PP performance compared to `IQ2_XXS` on the main branch, and 2X compared to `IQ2_XXS_R4`!

The same approach can be used out-of-the-box for `IQ3_XXS` (left for a follow up PR).

`IQ2_XS, IQ2_S` and `IQ3_S` use blocks of 16, so one would need a new row-interleaved 8-bit type with blocks of 16 for those.