### 🔀 [#553](https://github.com/ikawrakow/ik_llama.cpp/pull/553) - Much faster prompt processing for IQ1_S and IQ1_M on ARM_NEON

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-24 |
| **Updated** | 2025-06-24 |

---

#### Description

This PR corresponds to PRs #531, #533, #534, #546, #549, #550, #552, and applies the on-the-fly repacking technique to
the 1-bit quants `IQ1_S` and `IQ1_M` on `ARM_NEON`. 

Here is a PP-512 performance comparison between the main branch and this PR for LlaMA-3.1-8B-Instruct on M2-Max

| type |  t/s (main) | t/s (PR) | Speedup |
| ---: | ---: | ---: | ---: |
| IQ1_S | 66.3 | 168.8 | 2.546 |
| IQ1_M | 19.0 | 163.9 | 8.626 |

`IQ1_M` did not have a faster `IQK` implementation, so the 19 t/s is what one has within the standard `ggml` GEMM framework.