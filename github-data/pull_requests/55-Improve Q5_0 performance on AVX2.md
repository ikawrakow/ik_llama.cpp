### 🔀 [#55](https://github.com/ikawrakow/ik_llama.cpp/pull/55) - Improve Q5_0 performance on AVX2

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-14 |
| **Updated** | 2024-09-14 |

---

#### Description

The main purpose of the [previous PR](https://github.com/ikawrakow/ik_llama.cpp/pull/54) was to try to improve `K*Q` matrix multiplications for flash attention with `Q8_0` quantized k-cache. Sadly, the performance improvement that we got for `Q8_0` did not translate into better FA performance. It is a rainy Saturday, so need something to brighten my day. The last PR is very easily applied to `Q5_0`, so here we are.

The table shows performance comparison to mainline `llama.cpp` for LLaMA-3.1-8B ona Ryzen-7950X

| model         | backend    | threads |          test |     t/s (llama.cpp)  |    t/s (PR)      |  Speedup |
| --------------| ---------- | ------: | ------------: | -------------------: | ---------------: | -------: |
| llama 8B Q5_0 | CPU        |      16 |         pp512 |         55.72 ± 0.25 |    152.10 ± 0.74 |  2.793   |   
| llama 8B Q5_0 | CPU        |       2 |         tg128 |          5.22 ± 0.01 |      8.88 ± 0.01 |  1.701   |   
| llama 8B Q5_0 | CPU        |       4 |         tg128 |          9.24 ± 0.01 |     11.57 ± 0.00 |  1.252   |