### 🔀 [#552](https://github.com/ikawrakow/ik_llama.cpp/pull/552) - Much faster prompt processing for k-quants (ARM_NEON)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-24 |
| **Updated** | 2025-06-24 |

---

#### Description

It is time to give some attention to the `ARM_NEON` back-end, which has fallen behind quite a bit.

This PR corresponds to PRs #531, #533, #534, #546, #549, #550, and applies the on-the-fly repacking technique to k-quants (`Q2_K,  Q3_K, Q4_K, Q5_K, Q6_K`) and to `IQ4_XS` for the `ARM_NEON` implementation.

Here is a PP-512 performance comparison between the main branch and this PR for LlaMA-3.1-8B-Instruct on M2-Max

| type |  t/s (main) | t/s (PR) | Speedup |
| ---: | ---: | ---: | ---: |
| Q2_K | 85.74 | 168.07 | 1.960 |
| Q3_K | 45.68 | 170.83 | 3.740 |
| Q4_K | 58.24 | 114.78 | 1.971 |
| Q5_K | 54.88 | 114.92 | 2.094 |
| Q6_K | 47.67 | 123.98 | 2.601 |
| IQ4_XS | 71.19 | 167.84 | 2.358 |

`Q2_K, Q3_K` and `IQ4_XS` join the top-tier group in terms of prompt processing speed.

`Q4_K` and `Q5_K` get repacked to `Q8_1`, and this ends up being slower than `Q4_K_R4/Q5_K_R4`, so it may have been better to simply repack to the corresponding row-interleaved variant. This is left for a future PR.