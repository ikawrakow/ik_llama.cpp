### 🔀 [#32](https://github.com/ikawrakow/ik_llama.cpp/pull/32) - Zen4 Flash Attention

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-01 |
| **Updated** | 2024-09-01 |

---

#### Description

### TL;DR

This PR adds a flash attention (FA) implementation optimized for the Zen4 architecture as part of the quest to improve CPU inference for long contexts (#25, #26).

### Limitations

* It is Zen4-only for now. Strictly speaking, a much smaller subset of the AVX512 specification is required in the implementation (just `AVX512F` and `AVX512DQ`) compared to what Zen4 provides, but I didn't want to have too many variants, so decided to enable for Zen4 only.
* It is not implemented for ALiBi or unmasked attention. It is trivial to add these but I didn't want to clutter the implementation with branches that are mostly irrelevant. 

### Performance comparisons

The following graph compares the prompt processing (PP) performance of mainline `llama.cpp` (build: a47667cf - 3650) without (green symbols) and with (blue symbols) FA to PP performance in this repository for `Q4_K_S`-quantized LLaMA-3.1-8B running on a Ryzen-7950X CPU where
* Black symbols are without FA
* Brown symbols are with FA inherited from `llama.cpp`
* Magenta symbols are with the new FA implementation in this PR

![fa](https://github.com/user-attachments/assets/57078b91-cdcf-45b8-ba41-eee97774bc56)

We observe that the original FA implementation results in a significant performance degradation in mainline `llama.cpp` and also here. The effect is much stronger for the version here. This is due to the `K*Q` and `V*(softmax(K*Q)` matrix multiplications being much faster in this repository thanks to `iqk_mul_mat`, so performance hit is larger when they are replaced with the original `llama.cpp` FA CPU kernel. The new FA implementation improves performance. The improvement increases with context length, reaching about 24% at 32k tokens.

The next graph shows results for `Q4_K_S`-quantized Gemma-2-2b. Symbol colors are the same as above.

![fa_gemma2b](https://github.com/user-attachments/assets/8206ee28-02a0-43b6-be67-f9ea03378eb3)

In this case the original FA kernel improves performance in mainline `llama.cpp`. The difference in behavior compared to LLaMA-3.1-8B is easily explained by the fact that the Gemma-2 series of models use "soft-caping" in their attention layers, where `softcap(x) = c * tanh(x/c)` (`c` is a model-defined constant). This is implemented as 3 different operations in `llama.cpp`. When FA is enabled, these 3 operations, along with `softmax` are fused into a single kernel, and this results in am improvement of mainline `llama.cpp` performance even for short contexts. But when the original FA kernel is used in our version, where "soft-caping" is already handled by a dedicated fused operation, we get a massive drop in performance just like in the LLaMA-3.1-8B case above. The new implementation in this PR is much better and performance improves again, reaching 11% at 8k tokens, which is the maximum training context length of Gemma-2-2b.