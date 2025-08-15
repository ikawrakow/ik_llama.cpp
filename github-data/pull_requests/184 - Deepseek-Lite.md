### 🔀 [#184](https://github.com/ikawrakow/ik_llama.cpp/pull/184) - Deepseek-Lite

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-30 |
| **Updated** | 2025-01-30 |

---

#### Description

I was playing with Deepseek-Lite and noticed that
* Quantization mixes are inadequate, so added a few quick changes to that
* As some of the tensors row sizes are not divisible by 256, we get quite a few tensors quantized with `IQ4_NL`, so I noticed that after repacking to `IQ4_NL_R4` it does not work for row sizes that are not a multiple of 128 (4 blocks). So, I fixed that (AVX2 and Zen4)
* Once at it, also fixed `Q5_0_R4` and `Q6_0_R4`

Quantization error as measured by PPL is surprisingly low for the low-bit quants, even `IQ1_S` is kind of semi-usable. It is not a "true" `IQ1_S` quantization as quite a few tensors get quantized to `IQ4_NL`, and I changed the attention tensors, which represent a tiny fraction of the overall model sizes, to be quantized with much higher bpw.  We end up using 2.525 bpw for the repeating layers, and `PPL(IQ1_S)/PPL(fp16) - 1 = 49.4%`. But I now understand the hype around the Internet when the other day somebody was pretending to have invented 1-bit quantization and quantization mixes by using `IQ1_S` in `llama.cpp` for Deepseek-R1.