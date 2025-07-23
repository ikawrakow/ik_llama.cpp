### 🔀 [#307](https://github.com/ikawrakow/ik_llama.cpp/pull/307) - Metal: much faster MoE prompt processing

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-02 |
| **Updated** | 2025-04-03 |

---

#### Description

The prompt processing (PP) performance on Metal for MoE models with many experts (such as DeepSeek) is pathetic. Here, and also in mainline before the very recent [PR 12612](https://github.com/ggml-org/llama.cpp/pull/12612). This mainline PR brings PP performance to a more acceptable level by effectively using GEMV for matrix multiplications involving MoE tensors.

This PR does much better than that. On my M2-Max (30-core GPU) PP performance for DeepSeek-Lite is now 1.75X faster than mainline (`build: a6f32f0b3 (5018)`), and 5X compared to the main branch.

Also, on mainline I observe a very peculiar performance behavior as a function of `u_batch`:

| model                          |       size |    backend    | n_ubatch |          test |                  t/s |
| --------------------- | ---------- | ------: | -------: | ------------: | -------------------: |
| deepseek2 16B Q8_0  |  15.55 GiB |   Metal      |       128 |         pp512 |        254.43 ± 2.02 |
| deepseek2 16B Q8_0  |  15.55 GiB |   Metal      |       256 |         pp512 |        142.42 ± 0.24 |
| deepseek2 16B Q8_0  |  15.55 GiB |   Metal      |       512 |         pp512 |        417.56 ± 0.18 |

Interesting, right? For `u_batch = 512` (where performance is maximized) the matrix multiplication is done using GEMV. For `u_batch = 128, 256`, it is done using GEMM, but in an extremely inefficient way, where the inefficiency increases with `u_batch` size, so performance degrades.

Here is what we get with this PR:

| model               |       size | backend    | n_ubatch |          test |              t/s |
| ------------------- | ---------: | ---------- | -------: | ------------: | ---------------: |
| deepseek2 16B Q8_0  |  15.55 GiB | Metal      |      128 |         pp512 |    585.19 ± 1.07 |
| deepseek2 16B Q8_0  |  15.55 GiB | Metal      |      256 |         pp512 |    685.58 ± 3.39 |
| deepseek2 16B Q8_0  |  15.55 GiB | Metal      |      512 |         pp512 |    726.94 ± 2.35 |

The PR became much bigger than it should have been. But as TG performance is now slightly lower than mainline, and the only change that seemed promising to explain the difference was [PR 9698](https://github.com/ggml-org/llama.cpp/pull/9698), I decided to add that change. It made zero difference, but resulted in 2k lines of code moved around.