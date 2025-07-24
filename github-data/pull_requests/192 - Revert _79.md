### 🔀 [#192](https://github.com/ikawrakow/ik_llama.cpp/pull/192) - Revert [#79](https://github.com/ikawrakow/ik_llama.cpp/issues/79)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-07 |
| **Updated** | 2025-02-08 |

---

#### Description

While testing potential improvements of `IQ1_S_R4` quantization, I ran into NaNs while running a DeepSeek-Lite perplexity calculation. I did a `grep -r` on a folder with many big files while running the calculation and suddenly I got a NaN PPL. I repeated the calculation without doing anything else at the same time and the NaN did not happen. I then ran with 32 threads on a 16-core system and was able to reliably get a NaN at some random chunk.

This means there is a race.
   
The race was most likely introduced in #79 (avoid repeating already done quantizations of activations). I honestly do not understand why there could be a race, or even less do I understand why it would only happen for DeepSeek-Lite quantized with `IQ1_S_R4`. I have done countless runs since #79 and never observed anything suspicious.

Either way, this PR reverts #79. After doing so, there aren't any NaNs no matter how busy I make the system while running DeepSeek-Lite inference.  Hopefully this will also fix the NaNs @saood06 gets with `IQ1_S_R4` quantized DeepSeek-R1 (see discussion in #185).