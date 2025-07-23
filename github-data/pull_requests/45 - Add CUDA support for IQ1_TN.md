### 🔀 [#45](https://github.com/ikawrakow/ik_llama.cpp/pull/45) - Add CUDA support for IQ1_TN

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-09 |
| **Updated** | 2024-09-09 |

---

#### Description

Just reuse the `IQ1_BN` implementation. The only twist is that we now have the row scale stored at the beginning of the row, so we need a small modification of the dot product template to have a pointer to the beginning of the row passed to the dot product implementation.

It is slightly slower than `IQ2_TN` (305 t/s vs 320 t/s for the 4B TriLM model on RTX-4080), but this is to be expected given the bit twiddling we need to unpack the ternary bits.