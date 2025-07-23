### 🔀 [#488](https://github.com/ikawrakow/ik_llama.cpp/pull/488) - Faster CPU prompt processing for Trellis quants and MoE models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-03 |
| **Updated** | 2025-06-05 |

---

#### Description

This PR is a follow up to #482, and applies the same dequantizing GEMM for MoE matrix multiplications.

For a DeepSeek-Lite model where only the `ffn_up` and `ffn_gate` tensors are quantized with `IQ2_KT` I observe a ~35% improvement in PP performance compared to te main branch.