### 🔀 [#212](https://github.com/ikawrakow/ik_llama.cpp/pull/212) - Optimized GEMM/GEMV for IQ1_S

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-20 |
| **Updated** | 2025-02-20 |

---

#### Description

Apparently there are many people who would prefer to just run Unsloth's `IQ1_S` DeepSeek-R1 model as is instead of quantizing to `IQ1_S_R4` and taking advantage of the better model quality and improved inference speed.

So, here is a `iqk_mul_mat.cpp` implementation for `IQ1_S`.

I don't have the ability to run DeepSeek-R1, so using DeepSeek-Lite as a surrogate to test performance as it has the same architecture. The downside is that we don't test "pure" `IQ1_S` performance as various tensors that would have been quantized to `IQ1_S` get quantized to `IQ4_NL` due to their row sizes not being divisible by 256 (the `IQ1_S` block size). Performance tests are run on Ryzen-7950X (`Zen4`), Ryzen-5975WX (`AVX2`) and M2-Max CPU (`NEON`)

| model               | backend    | threads |     test |      t/s (main)  |   t/s (PR)       |   Speedup |
| ------------------- | ---------- | ------: | -------: | ---------------: | ---------------: | --------: |
| deepseek2 16B IQ1_S | AVX2       |      32 |    pp512 |    209.49 ± 0.61 |    484.99 ± 4.61 |  2.315    |
| deepseek2 16B IQ1_S |            |       2 |    tg128 |     12.13 ± 0.01 |     15.74 ± 0.01 |  1.298    |
| deepseek2 16B IQ1_S |            |       4 |    tg128 |     21.26 ± 0.01 |     26.29 ± 0.05 |  1.237    |
| deepseek2 16B IQ1_S |            |       8 |    tg128 |     30.85 ± 0.07 |     36.24 ± 0.13 |  1.175    |
| deepseek2 16B IQ1_S |            |      16 |    tg128 |     40.04 ± 0.01 |     42.00 ± 0.01 |  1.049    |
| deepseek2 16B IQ1_S | Zen4       |      16 |    pp512 |    142.33 ± 1.06 |    496.32 ± 1.75 |  3.487    |
| deepseek2 16B IQ1_S |            |       2 |    tg128 |     14.15 ± 0.02 |     19.08 ± 0.01 |  1.348    |
| deepseek2 16B IQ1_S |            |       4 |    tg128 |     24.34 ± 0.01 |     31.31 ± 0.08 |  1.286    |
| deepseek2 16B IQ1_S |            |       8 |    tg128 |     35.64 ± 0.01 |     42.48 ± 0.02 |  1.192    |
| deepseek2 16B IQ1_S |            |      16 |    tg128 |     44.37 ± 0.08 |     47.84 ± 0.18 |  1.078    |
| deepseek2 16B IQ1_S | NEON       |       8 |    pp512 |     88.77 ± 0.30 |    229.23 ± 1.53 |  2.582    |
| deepseek2 16B IQ1_S |            |       2 |    tg128 |     17.80 ± 0.01 |     22.72 ± 0.00 |  1.276    |
| deepseek2 16B IQ1_S |            |       4 |    tg128 |     29.80 ± 0.13 |     37.27 ± 0.24 |  1.251    |
| deepseek2 16B IQ1_S |            |       8 |    tg128 |     49.28 ± 0.07 |     59.28 ± 0.27 |  1.203    |

I think one can do better by interleaving 4 rows on the fly, but I leave this for another day.

---

#### 💬 Conversation

👤 **godrosev** commented the **2025-02-20** at **13:15:29**:<br>

ikawrakow, thank you so much. This helped me a lot!
Also, it's not that I'm reluctant to use it IQ1_S_R4。Instead, I need a smaller file size and memory (you said he would reduce it by a few GB), it's just that my current work requires running ready-made Unsloth's DeepSeek-R1.
As soon as I'm done with the job, I'll start doing my own quantification of the IQ1_S_R4 using your suggestion, and my device will test the R1 of the 671B very well and I'll tell you the results! I am 100% convinced that this new way(IQ1_S_R4) of quantizing will have better quality and speed!!
Thanks,again!