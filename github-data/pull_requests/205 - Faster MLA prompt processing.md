### 🔀 [#205](https://github.com/ikawrakow/ik_llama.cpp/pull/205) - Faster MLA prompt processing

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-12 |
| **Updated** | 2025-02-13 |

---

#### Description

This PR speeds up prompt processing (PP) when MLA is enabled. It is still slower than no-MLA, so I'm making this a draft for now to try some more. Still it would be great if somebody else tested to confirm that a) I did not introduce bugs and b) It is indeed faster on their systems.

The PR also adds the changes suggested by @saood06 in the review of #188 

Speedup is achieved by concatenating the no- and rotational position encoding parts of `K` and `Q` (this also eliminates the `k_r` cache), which allows us to combine the former `kq_nope` and `kq_pe` matrix multiplications into a single matrix multiplication. This also eliminates the fairly expensive addition of  `kq_nope` and `kq_pe`. 

Here is a comparison between PP performance on the main branch and this PR for DeepSeek-Lite quantized with `IQ4_XS` and running on a Ryzen-7950X using `Q8_0` for K-cache

| model                |          test |    t/s (main)    |    t/s (PR)      |  Speedup |
| -------------------- | ------------: | ---------------: | ---------------: | -------: |
| deepseek2 16B IQ4_XS |         pp512 |    478.58 ± 5.14 |    489.40 ± 1.08 |  1.023   |
| deepseek2 16B IQ4_XS |        pp1024 |    438.56 ± 0.75 |    458.37 ± 1.51 |  1.045   |
| deepseek2 16B IQ4_XS |        pp2048 |    378.95 ± 1.40 |    407.83 ± 2.07 |  1.076   |
| deepseek2 16B IQ4_XS |        pp4096 |    294.71 ± 2.86 |    327.88 ± 0.18 |  1.113   |
| deepseek2 16B IQ4_XS |        pp8192 |    204.52 ± 0.27 |    234.17 ± 0.37 |  1.145   |
| deepseek2 16B IQ4_XS |       pp16384 |    126.31 ± 0.13 |    148.35 ± 0.38 |  1.174   |

TG performance (the whole point of MLA) is not sacrificed. Here the results of `llama-bench -gp -Np,64` for different prompt lengths `Np`

| model                 |          test |     t/s (main)   |  t/s (PR)        |   Speedup |
| --------------------- | ------------: | ---------------: | ---------------: | --------: |
| deepseek2 16B IQ4_XS  |    tg64@pp128 |     33.58 ± 0.06 |     33.80 ± 0.00 |  1.007    |
| deepseek2 16B IQ4_XS  |    tg64@pp256 |     32.67 ± 0.00 |     32.76 ± 0.01 |  1.003    |
| deepseek2 16B IQ4_XS  |    tg64@pp512 |     32.38 ± 0.08 |     32.68 ± 0.05 |  1.009    |
| deepseek2 16B IQ4_XS  |   tg64@pp1024 |     31.50 ± 0.02 |     32.02 ± 0.00 |  1.017    |
| deepseek2 16B IQ4_XS  |   tg64@pp2048 |     30.01 ± 0.01 |     30.31 ± 0.03 |  1.010    |
| deepseek2 16B IQ4_XS  |   tg64@pp4096 |     27.08 ± 0.03 |     27.54 ± 0.10 |  1.017    |
| deepseek2 16B IQ4_XS  |   tg64@pp8192 |     22.82 ± 0.00 |     23.12 ± 0.01 |  1.013    |
| deepseek2 16B IQ4_XS  |  tg64@pp16384 |     17.24 ± 0.00 |     18.74 ± 0.09 |  1.087    |

Not sure if the ~9% improvement at 16k tokens is real. It may be just due to less thermal trottling because of the prompt processing part finishing quicker.

---

#### 💬 Conversation

👤 **saood06** submitted a review the **2025-02-12** at **20:10:21**: 💬 `COMMENTED`

---

👤 **ikawrakow** submitted a review the **2025-02-13** at **08:57:48**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-02-13** at **08:57:48** on `src/llama.cpp`:<br>

Thanks. Added a hopefully visible warning.

---

👤 **ikawrakow** commented the **2025-02-13** at **09:04:18**:<br>

The PR also adds a compile time option to disable the transposed KV cache when using MLA (simple look for `MLA_USE_TRANSPOSED_CACHE` and set it to 0). This cuts KV cache size in nearly half at the expense of a lower TG performance with long contexts. PP performance stays about the same. Here is a comparison between MLA with and without transposed cache

 | model                |          test |  t/s (with c^T)  |  t/s (without c^T)|
| -------------------- | ------------: | ---------------: | ----------------: |
| deepseek2 16B IQ4_XS |    tg64@pp128 |     33.58 ± 0.06 |      33.05 ± 0.05 |
| deepseek2 16B IQ4_XS |    tg64@pp256 |     32.67 ± 0.00 |      31.54 ± 0.07 |
| deepseek2 16B IQ4_XS |    tg64@pp512 |     32.38 ± 0.08 |      30.26 ± 0.33 |
| deepseek2 16B IQ4_XS |   tg64@pp1024 |     31.50 ± 0.02 |      28.50 ± 0.01 |
| deepseek2 16B IQ4_XS |   tg64@pp2048 |     30.01 ± 0.01 |      24.75 ± 0.01 |
| deepseek2 16B IQ4_XS |   tg64@pp4096 |     27.08 ± 0.03 |      20.67 ± 0.09 |
| deepseek2 16B IQ4_XS |   tg64@pp8192 |     22.82 ± 0.00 |      14.89 ± 0.01 |