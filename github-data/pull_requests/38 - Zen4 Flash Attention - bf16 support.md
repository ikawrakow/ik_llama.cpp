### 🔀 [#38](https://github.com/ikawrakow/ik_llama.cpp/pull/38) - Zen4 Flash Attention - bf16 support

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-04 |
| **Updated** | 2024-09-05 |

---

#### Description

This PR adds support for using `bf16` for the kv-cache.

As Zen4 has native support for `bf16` fused-multiply-add, I was hoping that this might give better performance than `fp16`. But with this implementation it is basically the same as `fp16`. We get a tiny improvement for Gemma2-2b at 4k and 8k tokens as shown in this graph (there is no `bf16` support for kv-cache in `llama.cpp`, so no comparison in the graph).

 
![fa_gemma2b](https://github.com/user-attachments/assets/8f104aeb-563d-46c8-a661-18ddd93ffe28)

Given this outcome, I have only enabled support for K- and V-cache both as `bf16` (i.e., one cannot mix `bf16` with other types as it is possible with `fp16`, `Q4_0`, `Q4_1` and `Q8_0`.