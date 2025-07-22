### 🔀 [#243](https://github.com/ikawrakow/ik_llama.cpp/pull/243) - Better FlashMLA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-06 |
| **Updated** | 2025-03-07 |

---

#### Description

This PR improves FlashMLA performance on the CPU for token generation (TG) with long contexts. The same strategy should also improve FA performance of GQA models, but something is not quite right there, so I have enabled only for MLA for now.

Here is a performance comparison between the main branch and this PR for DeepSeek-Lite on a Ryzen-7950X CPU

| model                |          test |    t/s (main)    |      t/s (PR)    |  Speedup |
| ---------------------| ------------: | ---------------: | ---------------: | -------: |
| deepseek2 16B IQ4_NL |    tg64@pp128 |     32.41 ± 0.04 |     32.22 ± 0.02 |  0.994   |
| deepseek2 16B IQ4_NL |    tg64@pp256 |     32.16 ± 0.02 |     31.96 ± 0.03 |  0.994   |
| deepseek2 16B IQ4_NL |    tg64@pp512 |     31.80 ± 0.00 |     31.85 ± 0.05 |  1.002   |
| deepseek2 16B IQ4_NL |   tg64@pp1024 |     31.30 ± 0.03 |     31.51 ± 0.00 |  1.007   |
| deepseek2 16B IQ4_NL |   tg64@pp2048 |     30.44 ± 0.01 |     30.93 ± 0.02 |  1.016   |
| deepseek2 16B IQ4_NL |   tg64@pp4096 |     28.50 ± 0.01 |     29.69 ± 0.08 |  1.042   |
| deepseek2 16B IQ4_NL |   tg64@pp8192 |     25.31 ± 0.14 |     27.19 ± 0.11 |  1.074   |
| deepseek2 16B IQ4_NL |  tg64@pp16384 |     20.40 ± 0.10 |     22.31 ± 0.03 |  1.094   |

For TG the `V*softmax(K*Q)` is parallelized along the heads, so given enough threads, the `K*Q` operation computed by each thread becomes a GEMV, which is notoriously memory bound. In this PR parallelization is done along the K-cache entries, with the `K*Q` portions computed by each thread being GEMM, which is faster. But this requires one additional thread synchronization before combining the results of the threads. My guess is that this extra barrier leads to the observed slightly lower performance for short contexts (where with the main branch implementation `K*Q` is fast despite being GEMV).

To put the above table into perspective, TG speed with a context of 16k tokens is around 10 t/s without MLA and FA for this model on this CPU.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-03-07** at **07:46:44**:<br>

The above table is for `Q8_KV` KV cache. Here is a comparison between the main branch and this PR for `fp16` KV cache:

| model                |          test |      t/s (main)  |      t/s (PR)    |  Speedup |
| ---------------------| ------------: | ---------------: | ---------------: | -------: |
| deepseek2 16B IQ4_NL |    tg64@pp128 |     31.54 ± 0.06 |     32.24 ± 0.01 |  1.022   |
| deepseek2 16B IQ4_NL |    tg64@pp256 |     30.79 ± 0.08 |     31.86 ± 0.05 |  1.035   |
| deepseek2 16B IQ4_NL |    tg64@pp512 |     29.83 ± 0.02 |     31.90 ± 0.01 |  1.069   |
| deepseek2 16B IQ4_NL |   tg64@pp1024 |     28.48 ± 0.02 |     31.48 ± 0.03 |  1.105   |
| deepseek2 16B IQ4_NL |   tg64@pp2048 |     26.05 ± 0.01 |     30.69 ± 0.00 |  1.178   |
| deepseek2 16B IQ4_NL |   tg64@pp4096 |     22.12 ± 0.04 |     29.45 ± 0.05 |  1.331   |
| deepseek2 16B IQ4_NL |   tg64@pp8192 |     17.25 ± 0.16 |     27.37 ± 0.14 |  1.587   |
| deepseek2 16B IQ4_NL |  tg64@pp16384 |     11.78 ± 0.03 |     23.13 ± 0.64 |  1.963   |

I.e., the PR is a massive upgrade in this case (but it also tells us that the original `fp16` FA kernel was far from optimal).