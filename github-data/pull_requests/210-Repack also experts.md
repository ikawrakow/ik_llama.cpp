### 🔀 [#210](https://github.com/ikawrakow/ik_llama.cpp/pull/210) - Repack also experts

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-19 |
| **Updated** | 2025-02-19 |

---

#### Description

When I implemented run time repacking, I required the tensor to be 2D to be eligible for repacking, I guess to simplify the code. But I forgot about MoE models, where expert weights are in 3D tensors.

This PR fixes that. This leads to very significant performance gains. E.g., for DeepSeek-Lite quantized with `IQ4_XS`, we get `PP-512 = 545` t/s on the main branch, and `PP-512 = 677 t/s` with this PR when using run time repacking.