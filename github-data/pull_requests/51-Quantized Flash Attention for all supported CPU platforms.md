### 🔀 [#51](https://github.com/ikawrakow/ik_llama.cpp/pull/51) - Quantized Flash Attention for all supported CPU platforms

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-12 |
| **Updated** | 2024-09-12 |

---

#### Description

This PR adds two features:
* All supported CPU platforms (`Zen4, AVX2, ARM_NEON`) now have implementations for quantized kv-cache. `Q4_0, Q4_1`, and `Q8_0` can be used
* When the cache is quantized, a quantized matrix multiplication is used for `K*Q`. 

The second bullet leads to performance improvements that increase with context length. The following graph shows an example of prompt processing speed for `Q4_K_S`-quantized LLaMA-3.1-8B as a function of prompt length. The orange curve is the new implementation in this PR of cache quantized with `Q8_0`. Results are on a Ryzen-7950X CPU (`Zen4`). At 32k tokens we now have 91.4 t/s vs 64.4 t.s without FA, so a 42% improvement in the quest to [improve CPU performance for large contexts](https://github.com/ikawrakow/ik_llama.cpp/discussions/25). I did not have the patience to wait for mainline `llama.cpp` to complete processing 32k tokens, but at the longest context of 8k tokens where my patience was not exhausted, we are now 2.2X faster compared to no-FA, and 3X faster compared to FA. 

![fa_q](https://github.com/user-attachments/assets/6a26d1ce-5fd2-4f54-87eb-3b8a5007f0bf)