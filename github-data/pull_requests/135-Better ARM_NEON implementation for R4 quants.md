### 🔀 [#135](https://github.com/ikawrakow/ik_llama.cpp/pull/135) - Better ARM_NEON implementation for R4 quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-11 |
| **Updated** | 2024-12-11 |

---

#### Description

We get improved performance for `IQ4_XS_R4`, `Q4_K_R4`, `Q5_K_R4`, `Q6_K_R4`. The trick was to accumulate super-blocks in `int32_t`, thus avoiding expensive `int -> float` conversions.

Here performance comparisons for LLaMA-3.1-8B on M2-Max between the previous implementation and this PR

| Quant |  Task | Threads | t/s (main) | t/s (PR) | Speedup | 
| ---: | ---: | ---: | ---: | ---: | ---: | 
| IQ4_XS_R4 | pp512 | 8 | 115.43 ± 0.57 | 131.28 ± 0.51 | 1.137 |
|                      | tg128 | 2 | 12.71 ± 0.01 | 13.44 ± 0.01 | 1.057 |
|                      | tg128 | 4 | 22.35 ± 0.17 | 22.98 ± 0.05  | 1.028 |
| Q4_K_R4    | pp512 | 8 | 110.02 ± 1.31 | 122.12 ± 1.28 | 1.110 |
|                      | tg128 | 2 | 12.17 ± 0.01 | 13.72 ± 0.01 | 1.127 |
|                      | tg128 | 4 | 21.56 ± 0.06  | 22.46 ± 0.20 | 1.042 |
| Q5_K_R4.    | pp512 | 8 | 96.90 ± 0.79 | 108.66 ± 0.27 | 1.121 |
|                      | tg128 | 2 | 8.22 ± 0.01 | 8.66 ± 0.01 | 1.054 |
|                      | tg128 | 4 | 15.54 ± 0.09 | 16.13 ± 0.05 | 1.038 |
| Q6_K_R4     | pp512 | 8 | 83.25 ± 0.81 | 104.19 ± 1.96 | 1.252 |
|                      | tg128 | 2 | 7.35 ± 0.01 | 8.05 ± 0.00 | 1.095 |
|                      | tg128 | 4 | 13.80 ± 0.01 | 14.92 ± 0.03 | 1.081 |
 
TG results only up to 4 threads because at 8 threads the result is 100% memory bound, so the same within noise.