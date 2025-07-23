### 🔀 [#265](https://github.com/ikawrakow/ik_llama.cpp/pull/265) - Allow q8_0 cache on the CPU for FlashMLA-2

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-18 |
| **Updated** | 2025-03-18 |

---

#### Description

Somehow I had the concept that `Q8_0` KV cache is working for CPU-only inference with FlashMLA-2. Indeed it is for prompt processing, but not for TG (two different paths are taken). Clearly too many options as I'm getting confused myself. Anyhow, this PR adds the missing `Q8_0 -> Q8_0` contiguous transpose operation, so now we can use `Q8_0` KV cache with FlashMLA-2 also on the CPU.