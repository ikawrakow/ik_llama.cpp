### 🔀 [#175](https://github.com/ikawrakow/ik_llama.cpp/pull/175) - Better BF16 support on AVX2

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-22 |
| **Updated** | 2025-01-22 |

---

#### Description

On the main branch `bf16` models are computed via `ggml`, which results in a horrible performance. This PR adds much better `GEMM` an `GEMV` for `bf16 x fp32`. The table shows a performance comparison between the main branch and this PR for LLaMA-3.1-8B-Instruct on a Ryzen-5975WX CPU

 | model         |       size |     params | threads |      test |   t/s (main)     |  t/s (PR)     |  Speedup |
| ------------- | ---------: | ---------: | ------: | --------: | ---------------: | ------------: | -------: |
| llama 8B BF16 |  14.96 GiB |     8.03 B |      32 |     pp512 |     47.17 ± 0.04 | 152.80 ± 0.12 |  3.239   |   
| llama 8B BF16 |  14.96 GiB |     8.03 B |       1 |     tg128 |      1.37 ± 0.00 |   2.06 ± 0.00 |  1.504   |
| llama 8B BF16 |  14.96 GiB |     8.03 B |       2 |     tg128 |      2.53 ± 0.00 |   3.21 ± 0.00 |  1.269   |
| llama 8B BF16 |  14.96 GiB |     8.03 B |       4 |     tg128 |      3.19 ± 0.00 |   3.64 ± 0.00 |  1.141   |
| llama 8B BF16 |  14.96 GiB |     8.03 B |       8 |     tg128 |      3.39 ± 0.00 |   3.64 ± 0.00 |  1.074   |