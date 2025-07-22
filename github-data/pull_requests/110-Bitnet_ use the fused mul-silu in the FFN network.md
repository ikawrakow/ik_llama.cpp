### 🔀 [#110](https://github.com/ikawrakow/ik_llama.cpp/pull/110) - Bitnet: use the fused mul-silu in the FFN network

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-26 |
| **Updated** | 2024-10-26 |

---

#### Description

I had forgotten that `build_bitnet()` does not use the standerd `llm_build_ffn` function, so the fused mul-silu didn't get used automatically for Bitnet when I added it to llm_build_ffn.

This gives us another ~1% speedup for TG-128 on Metal and CUDA.