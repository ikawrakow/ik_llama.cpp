### 🔀 [#125](https://github.com/ikawrakow/ik_llama.cpp/pull/125) - R4 improvements on ARM_NEON

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-08 |
| **Updated** | 2024-12-08 |

---

#### Description

This PR accomplishes two things:
* Reduces bloat by using a template for the `ARM_NEON` matrix multiplication implementation of interleaved rows quants `Q4_0_R4, Q5_0_R4, Q6_0_R4, IQ4_NL_X4, IQ4_XS_R4, Q8_0_R4` (and I should do the same for `AVX2/Zen4`)
* Achieves a ~7% PP speedup for all `R4` quants except `IQ4_XS_R4`. With this
  - `Q4_0_R4` now outperforms the hand-written assembly in mainline `llama.cpp` by a small margin (125 t/s vs 123 t/s)
  - `Q8_0_R4` becomes the fastest type for prompt processing on `ARM_NEON` (PP-512 = 128 t/s for LLaMA-3.1-8B on M2-Max).
  - All `R4` quants achieve PP-512 > 100 t/s for LLaMA-3.1-8B on M2-Max