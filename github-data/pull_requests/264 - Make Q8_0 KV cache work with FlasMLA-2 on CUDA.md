### 🔀 [#264](https://github.com/ikawrakow/ik_llama.cpp/pull/264) - Make Q8_0 KV cache work with FlasMLA-2 on CUDA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-18 |
| **Updated** | 2025-03-18 |

---

#### Description

For DeepSeek-V3/R1 this reduces KV cache size by ~2 GiB for a context of 65k tokens.

Using
```
-amb 512 -mla 2 -fa -ctk q8_0
```
one should now be able to use 65k context with a single 24 GB GPU processing all attention calculations and all non-MoE expert tensors offloaded to it. See PR #260 for meaning and effect of the `-amb` command line option. 

There is still an issue with one or more  of the `GGML_OP_REPEAT, GGML_OP_CONCAT, GGML_OP_CPY` operations on CUDA, which are required to implement the entire attention computation using quantized tensors, so this PR takes the pragmatic approach of computing the attention operations with `fp16` on CUDA. The downside is that `fp16` will be used  also on the CPU if the code was built with CUDA enabled (and this is slower than using `Q8_0` directly, wit the gap in performance increasing with context length).