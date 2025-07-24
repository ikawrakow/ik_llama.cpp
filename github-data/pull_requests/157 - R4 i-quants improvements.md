### 🔀 [#157](https://github.com/ikawrakow/ik_llama.cpp/pull/157) - R4 i-quants improvements

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-22 |
| **Updated** | 2024-12-22 |

---

#### Description

Unpacking k- and i-quants is computationally expensive. Because of this, it is useful to re-use the unpacked quants for multiplication with as many columns in the right matrix as possible. At the same time one also needs to restrict the number of columns being used to some maximum number so that accumulated results can remain in vector registers, so in `iqk_mul_mat` up to 8 columns are used. But  unpacking `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS` is computationally so expensive that is cheaper to load/unload accumulated results to/from vector registers so that unpacked quants can be reused more than 8 times.

This PR adds this change using 16 columns. We get non-negligible performance gains for `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, and even gain somewhat for `IQ3_K`, `IQ4_K`, `IQ4_KS`, and `IQ5_K`.

The table shows PP-512 performance comparisons between the main branch and this PR for LLaMA-3.1-8B and the affected quants on `ARM_NEON` (M2-Max), `Zen4` (Ryzen-7950X) and `AVX2` (Ryzen-5075WX). When a given quantization/platform combination is missing in the table, the change did not improve performance, so it was not enabled for the given combination.

| Quantization | Platform | Threads | t/s (main) | t/s (PR) | Speedup |
| ---: | ---: | ---: | ---: | ---: | --- |
| IQ2_XXS_R4 | ARM_NEON | 8 | 76.34 ± 0.58 | 85.33 ± 1.59 | 1.118 |
|                        | Zen4             | 16 | 151.08 ± 0.22 | 162.72 ± 0.49 | 1.077 |
|                        | AVX2             | 32 | 195.72 ± 0.20 | 221.85 ± 0.38  | 1.134 |
| IQ2_XS_R4   | ARM_NEON  | 8 | 54.13 ± 0.19  | 67.99 ± 0.22 | 1.256 |
|                        | AVX2            | 32 | 192.60 ± 0.37 | 220.56 ± 0.48 | 1.145 |
| IQ2_M_R4      | ARM_NEON | 8 | 50.40 ± 0.18 | 62.29 ± 0.21 | 1.236 |
|                         | Zen4            | 16 | 148.51 ± 0.51 | 169.49 ± 0.53  | 1.141 |
|                         | AVX2           | 32 | 176.76 ± 0.27 | 203.35 ± 0.46 | 1.150 |
| IQ3_XXS_R4 | ARM_NEON | 8 | 67.45 ± 0.78 | 73.56 ± 1.26 | 1.091 |
|                        | Zen4             | 16 | 141.62 ± 0.30  | 149.41 ± 0.49  | 1.055 |
|                        | AVX2            | 32 | 184.42 ± 0.26 | 206.96 ± 0.44  | 1.122 |
| IQ3_K_R4     | Zen4             | 16 | 230.33 ± 0.13  | 243.34 ± 0.50 | 1.056 |
| IQ4_KS_R4  | AVX2             | 32 | 245.37 ± 0.52 | 250.76 ± 0.50 | 1.022 |
| IQ4_K_R4    | AVX2             | 32 | 249.11 ± 0.38  | 264.23 ± 0.41 | 1.061 |
| IQ5_K_R4    | Zen4             | 16 | 230.23 ± 0.23 | 240.65 ± 0.58 | 1.045 |
|                      | AVX2             | 32 | 231.50 ± 0.43 | 245.98 ± 0.37 | 1.063 |