### 🔀 [#23](https://github.com/ikawrakow/ik_llama.cpp/pull/23) - iq4_k tweak

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-08-20 |
| **Updated** | 2024-08-20 |

---

#### Description

Use `iq5_k` for `attn_v` also when `n_gqa = 2`. 
This improves size vs quality tradeoff for Gemma-2 models.

This graph shows quantization error `PPL(Q)/PPL(f16)-1` for Gemma-2-27b-it and the various quantization types after this change. Other Gemma-2 models have similar quantization errors. The `IQ6_K` point is not visible because quantization error is zero (and the y-axis is logarithmic). Based on the peculiar legacy quants behavior, where `Q4_1` has a lower quantization error than `Q5_0`, one can hypothesize that there is a significant asymmetry in the model weights of Gemma-2-27b. This is also visible (but to a much lesser extent) for k-quants, where `Q3_K` and `Q6_K` (both of type `weight = a * q`, so assuming symmetric weights) are somewhat higher than what one would expect from `Q2_K, Q4_K, Q5_K` (of type `weight = a * q + b`, so taking into account possible model weight asymmetry). The new `IQX_K` quants are much better at 4+ bits-per-weight (bpw), but even at 2- and 3-bpw there is a non-negligible improvement compared to the similarly sized `IQ2_S` and `IQ3_S`. 
 
![g27](https://github.com/user-attachments/assets/ed84b8ea-662c-45e9-b0f5-48b15993c521)