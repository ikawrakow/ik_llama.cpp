### 🔀 [#581](https://github.com/ikawrakow/ik_llama.cpp/pull/581) - Vulkan: Disable multi-add for now

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-03 |
| **Updated** | 2025-07-03 |

---

#### Description

...until we implement it for Vulkan, else it will run on the CPU and performance of MoE models will be terrible.

Also the Vulkan back-end has the very strange restriction that the number of experts times the number of tokens must be `<= 4096` for indirect matrix multiplications (as needed in MoE models). Haven't looked into why this restriction is imposed (as I'm not familiar with the Vulkan back-end  at all), so for now just using a very recent PR in mainline to split the indirect matrix multiplication into chunks, where each chunks satisfies the restriction.

But this basically means a horrible performance for MoE models. Case in point, with DeepSeek-V2-Lite I'm getting in the range of 1600 t/s PP speed (here and in mainline) vs ~9000 with the `ik_llama.cpp` CUDA back-end on an RTX-4080.  

Curious if someone is using the Vullkan back-end in `llama.cpp` to run DeepSeek-V3/R1 and/or Qwen3-235B-A22B and/or LlaMA-4, etc.