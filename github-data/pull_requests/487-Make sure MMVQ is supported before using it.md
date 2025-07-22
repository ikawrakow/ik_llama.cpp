### 🔀 [#487](https://github.com/ikawrakow/ik_llama.cpp/pull/487) - Make sure MMVQ is supported before using it

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-06-03 |
| **Updated** | 2025-06-03 |

---

#### Description

The new trellis quants do not support quantized matrix-vector multiplications (a.k.a., MMVQ), but the fused ffn_up+ffn_gate implementation does not check for that, which leads to an assert when the MMVQ is called for a trellis quant.

This PR attempts to fix it.