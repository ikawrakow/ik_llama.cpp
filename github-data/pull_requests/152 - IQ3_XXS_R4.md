### 🔀 [#152](https://github.com/ikawrakow/ik_llama.cpp/pull/152) - IQ3_XXS_R4

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-20 |
| **Updated** | 2024-12-20 |

---

#### Description

Sub-4 bpw i-quants have a terrible CPU performance, so I was curious to see if we can improve by interleaving rows.

This PR adds `IQ3_XXS_R4`, a 4-row interleaved version of `IQ3_XXS`.

We get decent performance gains, but still remain much slower than k- or legacy quants. I think there is still potential for optimization, but I was getting constantly confused about shuffling signs and scales, so at the end gave up with this result. 

Here is `PP-512` for LLaMA-3.1-8B on `Zen4` (Ryzen-7950X), `ARM_NEON` (M2-Max) and `AVX2` (Ryzen-5975WX)

| Platform |  Threads | IQ3_XXS | IQ3_XXS_R4 | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| ARM_NEON |  8 |  48.18 ± 0.69  | 67.45 ± 0.78 | 1.400 |
| Zen4            | 16 | 107.42 ± 0.33 | 141.62 ± 0.30   | 1.318 |
| AVX2           | 32 | 142.38 ± 0.48 |  184.42 ± 0.26  | 1.295 |

We get decent performance gains for TG as well, especially on `AVX2`.
Here results for TG-128 on LLaMA-3.1-8B with different numbers of threads:

| Platform |  Threads | IQ4_KS | IQ4_KS_R4 | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| ARM_NEON | 2 |  3.46 ± 0.02 | 4.79 ± 0.00 | 1.384 |
|                      | 4 | 6.65 ± 0.01 | 8.78 ± 0.04 | 1.320 |
|                      | 8 | 10.83 ± 0.18 | 15.95 ± 0.25  | 1.473 |
| Zen4            | 2 |  5.18 ± 0.00  | 6.53 ± 0.00  |  1.261 |
|                      | 4 |  9.70 ± 0.0 | 12.15 ± 0.00   |  1.253 |
|                      | 8 |  17.19 ± 0.18  | 17.93 ± 0.00  |  1.044 |
| AVX2           | 2 | 2.04 ± 0.0  | 4.07 ± 0.00 | 1.995 |
|                     | 4 | 4.04 ± 0.00  |  7.94 ± 0.00 | 1.965 |
|                     | 8 |  7.40 ± 0.01  | 14.16 ± 0.06  | 1.914 |
|                     | 16 |  13.64 ± 0.00  |  17.92 ± 0.01  | 1.314 |

We now manage to saturate the available memory bandwidth on the Ryzen CPUs at 8 (Ryzen-7950X) or 16 (Ryzen-5975WX) threads, but are far from being memory bound on the M2-Max.