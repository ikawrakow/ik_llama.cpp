### 🔀 [#154](https://github.com/ikawrakow/ik_llama.cpp/pull/154) - IQ2_XXS_R4

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-20 |
| **Updated** | 2024-12-20 |

---

#### Description

Sub-4 bpw i-quants have a terrible CPU performance, so I was curious to see if we can improve by interleaving rows.

This PR adds `IQ2_XXS_R4`, a 4-row interleaved version of `IQ2_XXS`.

We get decent performance gains, but still remain much slower than k- or legacy quants. I think there is still potential for optimization, but I was getting constantly confused about shuffling signs and scales, so at the end gave up with this result. 

Here is `PP-512` for LLaMA-3.1-8B on `Zen4` (Ryzen-7950X), `ARM_NEON` (M2-Max) and `AVX2` (Ryzen-5975WX)

| Platform |  Threads | IQ2_XXS | IQ2_XXS_R4 | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| ARM_NEON |  8 |  56.40 ± 0.99  | 76.34 ± 0.58 | 1.354 |
| Zen4            | 16 | 134.68 ± 0.31 | 153.60 ± 0.23   | 1.140 |
| AVX2           | 32 | 155.48 ± 0.17 |  195.72 ± 0.20  | 1.259 |

We get very decent performance gains for TG as well, especially on `AVX2`.
Here results for TG-128 on LLaMA-3.1-8B with different numbers of threads:

| Platform |  Threads | IQ2_XXS | IQ2_XXS_R4 | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| ARM_NEON | 2 |  4.40 ± 0.03  | 6.65 ± 0.00  | 1.511 |
|                      | 4 | 8.61 ± 0.01  | 12.20 ± 0.02 | 1.417 |
|                      | 8 | 15.84 ± 0.34 | 21.76 ± 0.31  | 1.374 |
| Zen4            | 2 |  6.59 ± 0.00  | 8.66 ± 0.00  |  1.314 |
|                      | 4 |  11.62 ± 0.81 | 15.49 ± 0.36  |  1.333 |
|                      | 8 |  20.40 ± 0.70  | 23.37 ± 0.03  |  1.146 |
| AVX2           | 2 | 2.62 ± 0.00  | 5.54 ± 0.00 | 2.115 |
|                     | 4 | 5.17 ± 0.00  |  10.81 ± 0.00 | 2.091 |
|                     | 8 |  9.49 ± 0.02  | 18.93 ± 0.08   | 1.995 |
|                     | 16 |  16.97 ± 0.00  |  25.70 ± 0.01  | 1.514 |

We now manage to saturate the available memory bandwidth on the Ryzen CPUs at 8 (Ryzen-7950X) or 16 (Ryzen-5975WX) threads, but are far from being memory bound on the M2-Max.