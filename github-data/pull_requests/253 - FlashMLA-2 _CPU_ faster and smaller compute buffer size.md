### 🔀 [#253](https://github.com/ikawrakow/ik_llama.cpp/pull/253) - FlashMLA-2 (CPU): faster and smaller compute buffer size

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-12 |
| **Updated** | 2025-03-13 |

---

#### Description

This PR improves the CPU implementation of FlashMLA in 3 ways:
* Faster prompt processing - about 13% improvement for a context of 16k tokens
* Smaller compute buffer size - about  60% reduction for a context of 128k tokens

To recall, FlashMLA-2 is enabled via `-mla 2 -fa`, and is the variant that works on the CPU and on CUDA.

The improvement is achieved by adding implementations for
* `ggml_mul_mat` where the second operand is not `fp32`
* `ggml_concat` where the operands are quantized
* `ggml_repeat` where the operand is not `fp32`

This allows us to avoid conversions to `fp32` that can become quite costly when operating on a very large context.

Here is a PP performance comparison for DeepSeek-Lite running on a Ryzen-7950X CPU between the main branch and this PR

| model                |          test |    t/s (main)    |   t/s (PR)       |  Speedup |
| ---------------------| ------------: | ---------------: | ---------------: | -------: |
| deepseek2 16B IQ4_NL |         pp512 |    668.46 ± 1.74 |   680.74 ± 21.47 |  1.018   |
| deepseek2 16B IQ4_NL |        pp1024 |    646.86 ± 0.94 |    668.65 ± 0.44 |  1.034   |
| deepseek2 16B IQ4_NL |        pp2048 |    596.56 ± 1.70 |    628.99 ± 1.72 |  1.054   |
| deepseek2 16B IQ4_NL |        pp4096 |    513.16 ± 1.42 |    552.36 ± 4.61 |  1.076   |
| deepseek2 16B IQ4_NL |        pp8192 |    398.45 ± 3.51 |    442.89 ± 3.96 |  1.112   |
| deepseek2 16B IQ4_NL |       pp16384 |    272.58 ± 7.06 |    308.21 ± 5.91 |  1.131   |

And here is a comparison between compute buffer sizes along with KV cache size for `fp16` cache

| context |  KV cache size (MiB)  | compute buffer (MiB, PR)  | compute buffer (MiB, main) |
| ----: | ---: | ---: | ---: |
|   2048 |  60.75 | 204.00 |  204.00 |
|  4096  | 121.50 | 204.00 | 204.00 |
| 8192 | 243.00 | 220.01 | 358.01 |
| 16384 | 486.00 | 452.01 | 712.01 |
|  32768 | 972.00 | 884.01 | 1404.02 |
| 65536 | 1944.00 | 1748.01 | 2788.02 |
| 131072 | 3888.00 | 3476.02 | 5556.02 |

I did a quick attempt to also implement on CUDA, but something wasn't working, so left it for a future PR. This also implies that the new way of preparing the compute graph will only be used if the code was built without support for additional back-ends (even if zero layers are uploaded to them, to avoid fighting with the back-end scheduler).

---

#### 💬 Conversation

👤 **davidsyoung** commented the **2025-03-12** at **14:53:09**:<br>

Nice! The compute buffer on CUDA makes it hard to balance model layers with the compute buffer, so when you manage to get CUDA implementation working it'll be amazing. Thank you for your work on this