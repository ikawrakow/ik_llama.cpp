### 🔀 [#584](https://github.com/ikawrakow/ik_llama.cpp/pull/584) - Vulkan: flash attention for DeepSeek models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-04 |
| **Updated** | 2025-07-05 |

---

#### Description

This PR is a cherry-pick of [PR 14509](https://github.com/ggml-org/llama.cpp/pull/14509) in mainline `llama.cpp` with minor adaptations, and adds FA for the DeepSeek models to the Vulkan back-end.

### Caveats

* The batch size cannot be greater than the maximum context length. Under normal usage this is never the case, but if one runs `perplexity` with default parameters where context is set to 512 tokens while batch size is 2048 tokens, one gets NaNs after the first context chunk. I have spent the better part of of the day trying to understand the reason, and just don't see it. Almost prepared to give a bounty to the person who finds the bug. 
* For now KV cache can only be `fp16` as I have not implemented the various additions required to make quantized cache work with DeepSeek models in the Vulkan back-end (quantized KV cache can of course be used with models that do not use MLA)

I have tested with DeepSeek-V2-Lite on an RTX-4080 GPU with coopmat2 enabled. We are starting to see more significant performances gains compared to mainline `llama.cpp` as illustrated in the following two graphs. The first graph shows PP-2048 performance as a function of the number of tokens in the KV cache `N_KV`. Surprisingly, we don't see significant performance gains from `mla = 3` compared to `mla = 1` as we do with CUDA (see below). Nevertheless, at 32k tokens `ik_llama.cpp` is about 40% faster than `llama.cpp`.

![vulkan_dsl2_pp](https://github.com/user-attachments/assets/08952afa-6872-47a6-b7be-8c949cd7acc9)

The next graph compares TG performance as a function of `N_KV`. Here performance gains compared to mainline are even greater, with `ik_llama.cpp` nearly 2X faster than `llama.cpp` for a context of 32 tokens.

![vulkan_dsl2_tg](https://github.com/user-attachments/assets/375bc61b-9e44-4bda-8ccc-8f58f960c6a2)

Before you get too excited about these results, a reminder that the Vulkan back-end does not yet implement the fused MoE `ffn_up+ffn_gate` op, so it is still far behind CUDA. The next two graphs compare PP and TG performance as a function of `N_KV` on **the same RTX-4080 GPU**.

![vulkan_dsl2_vs_cuda_pp](https://github.com/user-attachments/assets/7a0f101c-eabc-45de-8d13-940c94ba1a84)
![vulkan_dsl2_vs_cuda_tg](https://github.com/user-attachments/assets/708df9d1-5ee2-436a-965f-3017c5c0db8c)