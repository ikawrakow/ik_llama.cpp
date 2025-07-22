### 🐛 [#269](https://github.com/ikawrakow/ik_llama.cpp/pull/269) - Fix ggml_compute_forward_dup_q

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-19 |
| **Updated** | 2025-03-19 |

---

#### Description

I broke it with PR #265. I was testing with a model where the wk_b and wk_v tensors were present, so didn't need to be computed, so didn't notice that the change I made to ggml_compute_forward_dup_q breaks that computation.