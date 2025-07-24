### 🔀 [#330](https://github.com/ikawrakow/ik_llama.cpp/pull/330) - Allow q8_0 KV cache for head size 256

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-15 |
| **Updated** | 2025-04-15 |

---

#### Description

Gemma models have a head size of 256. For whatever reason, the inherited CUDA FA code only allows `fp16` KV cache for this head size. This PR adds the ability to also use `Q8_0` KV cache with FA.