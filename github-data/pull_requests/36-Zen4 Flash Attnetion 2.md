### 🔀 [#36](https://github.com/ikawrakow/ik_llama.cpp/pull/36) - Zen4 Flash Attnetion 2

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-03 |
| **Updated** | 2024-09-04 |

---

#### Description

This PR is a follow up on #32 and adds the ability to use quantized K- and V-cache in the flash attention (FA) kernel. `Q4_0`, `Q4_1` and `Q8_0` are supported as cache quantization types. It is trivial to add additional types, but the implementation is templated, so number of template instantiations grows quadraticly with the number of supported quantization types, so I decided to settle for these 3 types for now.

Performance is slightly lower than `fp16` cache (see graph below), so main use case is KV-cache size reduction for very large context lengths. Still, unlike mainline `llama.cpp`, performance remains strictly above no-FA.

The graph below shows PP performance as a function of context length (logarithmic scale) for Gemma-2-2b quantized with `Q4_K_S` on a Ryzen-7950X CPU.

![fa_gemma2b_q](https://github.com/user-attachments/assets/8e42d3eb-74f5-45ba-9d63-92d661363e60)