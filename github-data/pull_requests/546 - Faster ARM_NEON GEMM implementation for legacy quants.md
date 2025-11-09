## 🔀 [Pull Request #546](https://github.com/ikawrakow/ik_llama.cpp/pull/546) - Faster ARM_NEON GEMM implementation for legacy quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/gemm_neon_legacy` |
| **Target Branch** | `main` |
| **Created** | 2025-06-21 |
| **Updated** | 2025-06-22 |
| **Merged** | 2025-06-21 |

---

## 📄 Description

It is time to give some attention to the `ARM_NEON` back-end, which has fallen behind quite a bit.

This PR corresponds to PRs [#531](https://github.com/ikawrakow/ik_llama.cpp/issues/531), [#533](https://github.com/ikawrakow/ik_llama.cpp/issues/533), [#534](https://github.com/ikawrakow/ik_llama.cpp/issues/534) and applies the on-the-fly repacking technique to `Q4_0, Q4_1, Q5_0, Q5_1, Q6_0, Q8_0, IQ4_NL` for the `ARM_NEON` implementation.

Here is a PP-512 performance comparison between the main branch and this PR for LlaMA-3.1-8B-Instruct on M2-Max

| type |  t/s (main) | t/s (PR) | Speedup |
| ---: | ---: | ---: | ---: |
| Q4_0 | 83.58 | 128.41 | 1.536 |
| Q5_0 | 74.20 |  128.57 | 1.733 |
| Q6_0 | 74.25 | 128.79 | 1.735 |
| Q8_0 | 84.45 | 128.63 | 1.523 |
| IQ4_NL | 84.47 | 128.09 | 1.516 |
| Q4_1 | 74.44 | 115.36 | 1.550 |
| Q5_1 | 64.16 | 114.89 | 1.791 |