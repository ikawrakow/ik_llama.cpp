### 🔀 [#142](https://github.com/ikawrakow/ik_llama.cpp/pull/142) - BF16_R16 - 16 interleaved bf16 rows  

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-14 |
| **Updated** | 2024-12-15 |

---

#### Description

After breaking the world record for 8-bit quantized matrix multiplications with `Q8_K_R8` in PR #141, I got excited to try to speed up `bf16` CPU inference. This PR is the somewhat disappointing result. I tried interleaving 4, 8, and 16 rows, 16 is fastest (but only very slightly faster than 8). It is disappointing because we only gain about 11% in prompt processing speed compared to the `bf16` implementation in `iqk_mul_mat` (but that one is already ~3X faster compared to mainline `llama.cpp`). On the bright side we do get TG speedup - 3.12 t/s vs 2.5 t/s for LLaMA-3.1-8B with 1 thread on a Ryzen-7950X, 4.25 t/s vs 3.9 t/s with 2 threads (and 2 threads fully saturate the memory bandwidth when using `BF16_R16`).   

Anyway, here a table with the `BF16_R16` PP-512 and TG-128 speeds on Ryzen-7950X

| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| llama 8B BF16_R16               |  14.96 GiB |     8.03 B | CPU        |      16 |         pp512 |    263.15 ± 0.19 |
| llama 8B BF16_R16               |  14.96 GiB |     8.03 B | CPU        |       1 |         tg128 |      3.12 ± 0.00 |
| llama 8B BF16_R16               |  14.96 GiB |     8.03 B | CPU        |       2 |         tg128 |      4.25 ± 0.00 |
| llama 8B BF16_R16               |  14.96 GiB |     8.03 B | CPU        |       4 |         tg128 |      4.14 ± 0.00 |