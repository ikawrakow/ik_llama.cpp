### 🔀 [#70](https://github.com/ikawrakow/ik_llama.cpp/pull/70) - Fused unary(x)*y

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-30 |
| **Updated** | 2024-10-02 |

---

#### Description

This is useful for parallel FFNs. `unary` can be `silu, gelu` or `relu`.

Implemented for CPU, CUDA and Metal.

Speedup is disappointingly small (1-3% for PP, depending on platform and model).

Let me think some more if I want to merge it.