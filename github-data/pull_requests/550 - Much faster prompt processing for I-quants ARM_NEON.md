## 🔀 [Pull Request #550](https://github.com/ikawrakow/ik_llama.cpp/pull/550) - Much faster prompt processing for I-quants (ARM_NEON)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/gemm_neon_iquants` |
| **Target Branch** | `main` |
| **Created** | 2025-06-23 |
| **Updated** | 2025-06-23 |
| **Merged** | 2025-06-23 |

---

## 📄 Description

It is time to give some attention to the `ARM_NEON` back-end, which has fallen behind quite a bit.

This PR corresponds to PRs [#531](https://github.com/ikawrakow/ik_llama.cpp/issues/531), [#533](https://github.com/ikawrakow/ik_llama.cpp/issues/533), [#534](https://github.com/ikawrakow/ik_llama.cpp/issues/534), [#546](https://github.com/ikawrakow/ik_llama.cpp/issues/546), [#549](https://github.com/ikawrakow/ik_llama.cpp/issues/549), and applies the on-the-fly repacking technique to i-quants (`IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S`) for the `ARM_NEON` implementation.

Here is a PP-512 performance comparison between the main branch and this PR for LlaMA-3.1-8B-Instruct on M2-Max

| type |  t/s (main) | t/s (PR) | Speedup |
| ---: | ---: | ---: | ---: |
| IQ2_XXS | 55.79 | 167.55 | 3.003 |
| IQ2_XS   | 46.40 | 166.65 | 3.592 |
| IQ2_S     | 42.75 | 166.83 | 3.903 |
| IQ3_XXS | 51.84 | 165.56 | 3.194 |
| IQ3_S   | 46.02 | 162.03 | 3.521 |

At this point i- and `IQK` quants are the top tier quants for prompt processing speed on `ARM_NEON`.