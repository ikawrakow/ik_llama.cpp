### 🔀 [#544](https://github.com/ikawrakow/ik_llama.cpp/pull/544) - New integer trellis on ARM_NEON 

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-20 |
| **Updated** | 2025-06-20 |

---

#### Description

This PR adapts the ARM_NEON trellis implementation to the new integer trellis.

Test done on an M2-Max CPU using LlaMA-3.1-8B-Instruct.

Very respectable PP performance:

 | model            |       size |          test |              t/s |
| ---------------- | ---------: | ------------: | ---------------: |
| llama 8B IQ2_KT  |   2.77 GiB |         pp512 |    129.19 ± 0.22 |
| llama 8B IQ3_KT  |   3.58 GiB |         pp512 |    127.66 ± 0.38 |
| llama 8B IQ4_KT  |   4.30 GiB |         pp512 |    125.23 ± 0.44 |

Still very low TG performance:

| model            |       size |          test |              t/s |
| ---------------- | ---------: | ------------: | ---------------: |
| llama 8B IQ2_KT  |   2.77 GiB |         tg128 |     12.59 ± 0.15 |
| llama 8B IQ3_KT  |   3.58 GiB |         tg128 |      9.92 ± 0.02 |
| llama 8B IQ4_KT  |   4.30 GiB |         tg128 |      9.73 ± 0.05 |

Don't ask Apple Silicon to do too much work with a piece of data fetched from memory.

Nevertheless, compared to PR #471 we observe ~13% speedup for `IQ2_KT`, ~30% speedup for `IQ3_KT`, and nearly 70% speedup for `Q4_KT`.