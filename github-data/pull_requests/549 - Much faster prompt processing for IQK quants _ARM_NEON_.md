### 🔀 [#549](https://github.com/ikawrakow/ik_llama.cpp/pull/549) - Much faster prompt processing for IQK quants (ARM_NEON)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-23 |
| **Updated** | 2025-06-23 |

---

#### Description

It is time to give some attention to the `ARM_NEON` back-end, which has fallen behind quite a bit.

This PR corresponds to PRs #531, #533, #534, #546 and applies the on-the-fly repacking technique to `IQK` quants (`IQ2_KS, IQ2_K, IQ3_K, IQ4_KS, IQ4_K, IQ5_KS, IQ5_K, IQ6_K`) for the `ARM_NEON` implementation.

Here is a PP-512 performance comparison between the main branch and this PR for LlaMA-3.1-8B-Instruct on M2-Max

| type |  t/s (main) | t/s (PR) | Speedup |
| ---: | ---: | ---: | ---: |
| IQ2_KS | 75.66 | 166.10 | 2.195 |
| IQ2_K   | 47.40 | 166.94 | 3.522 |
| IQ3_K   | 47.28 | 166.48 | 3.521 |
| IQ4_KS | 70.03 | 167.32 | 2.389 |
| IQ4_K   | 46.41 | 167.19 | 3.602 |
| IQ5_KS | 63.76 | 166.01 | 2.604 |
| IQ5_K   | 45.80 | 167.57 | 3.569 |
| IQ6_K   | 43.92 | 164.29 | 3.741 | 

At this point `IQK` quants are the top tier quants for prompt processing speed on `ARM_NEON`.