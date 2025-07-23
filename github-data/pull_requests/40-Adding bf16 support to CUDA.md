### 🔀 [#40](https://github.com/ikawrakow/ik_llama.cpp/pull/40) - Adding bf16 support to CUDA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-05 |
| **Updated** | 2024-09-14 |

---

#### Description

Haha, `llama.cpp` seems to not support `bf16` on CUDA?

This PR adds it. It works fine on my `RTX-4080`, but I have no idea if it will work on older GPUs (if I understood correctly it should, with reduced performance), ROCm, etc.

Performance is the same as `f16` for TG (TG-128 = 41.2 t/s for LLaMA-3.1-8B for both).

PP is lower but quite decent for prompt processing (PP-512(`bf16`) = 5250 t/s vs PP-512(`f16`) = 7250 t/s  for LLaMA-3.1-8B). In any case, much better than running on the CPU for `bf16` models.