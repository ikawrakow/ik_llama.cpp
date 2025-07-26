### [Pull Request #111](https://github.com/ikawrakow/ik_llama.cpp/pull/111) - Use fused mul - unary op also for MoE models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Created** | 2024-10-26 |
| **Updated** | 2024-10-26 |
| **Merged** | 2024-10-26 |

---

#### Description

This gives us a ~1% speedup for MoE models on CUDA and Metal.