### 🔀 [#207](https://github.com/ikawrakow/ik_llama.cpp/pull/207) - Faster CPU TG for GQA models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-15 |
| **Updated** | 2025-02-15 |

---

#### Description

This PR
* Absorbs the `iqk` matrix multiplication logic in `ggml` into a new `iqk` function `iqk_mul_mat_4d`. The change to `ggml` to incorporate the `iqk`-added functionality is now much less intusive
* Adds to `iqk_mul_mat_4d` special handling of the TG case with GQA. In this case the `K` and `V` tensors have a shape `N x M x Lkv` (`N` is the head size, `Lkv` is the number of KV heads), and they multiply a tensor (`Q` or `K*Q`) with shape `N x 1 x L` (`L` is the number of heads, `L > Lkv`). If we rearrange `Q` as `N x L/Lkv x Lkv`, we now have GEMM instead of GEMV, and this is significantly faster.

This better approach only gives noticeable TG speedup for long context (large KV cache), as without that the fraction of time spent on the `K*Q` and `V*softmax(K*Q)` is small. So, here is a table comparing TG performance on main and with this PR for LLaMA-3.1-8B for different prompt lengths. Model is quantized with `IQ4_XS` and is running on a Ryzen-7950X (Zen4) or M2-Max CPU

 | model            | backend    | threads |          test |     t/s (main)   |   t/s (PR)       |  Speedup |
| ---------------- | ---------- | ------: | ------------: | ---------------: | ---------------: | -------: |
| llama 8B IQ4_XS  | Zen4       |       8 |    tg64@pp128 |     13.85 ± 0.01 |     13.88 ± 0.00 |  1.002   |
| llama 8B IQ4_XS  | Zen4       |       8 |    tg64@pp256 |     13.72 ± 0.01 |     13.80 ± 0.00 |  1.006   |
| llama 8B IQ4_XS  | Zen4       |       8 |    tg64@pp512 |     13.48 ± 0.02 |     13.63 ± 0.02 |  1.011   |
| llama 8B IQ4_XS  | Zen4       |       8 |   tg64@pp1024 |     13.05 ± 0.02 |     13.33 ± 0.00 |  1.021   |
| llama 8B IQ4_XS  | Zen4       |       8 |   tg64@pp2048 |     12.21 ± 0.01 |     12.77 ± 0.00 |  1.046   |
| llama 8B IQ4_XS  | Zen4       |       8 |   tg64@pp4096 |     10.72 ± 0.00 |     11.82 ± 0.00 |  1.103   |
| llama 8B IQ4_XS  | Zen4       |       8 |   tg64@pp8192 |      8.60 ± 0.00 |     10.26 ± 0.01 |  1.193   |
| llama 8B IQ4_XS  | M2-Max     |       8 |    tg64@pp128 |     26.82 ± 0.07 |     28.01 ± 0.06 |  1.044    |
| llama 8B IQ4_XS  | M2-Max     |       8 |    tg64@pp256 |     26.49 ± 0.04 |     27.90 ± 0.01 |  1.053    |
| llama 8B IQ4_XS  | M2-Max     |       8 |    tg64@pp512 |     25.94 ± 0.00 |     27.47 ± 0.00 |  1.059    |
| llama 8B IQ4_XS  | M2-Max     |       8 |   tg64@pp1024 |     24.80 ± 0.00 |     26.28 ± 0.40 |  1.060    |
| llama 8B IQ4_XS  | M2-Max     |       8 |   tg64@pp2048 |     22.66 ± 0.01 |     25.17 ± 0.00 |  1.111    |
| llama 8B IQ4_XS  | M2-Max     |       8 |   tg64@pp4096 |     18.99 ± 0.01 |     23.12 ± 0.02 |  1.217    |
| llama 8B IQ4_XS  | M2-Max     |       8 |   tg64@pp8192 |     14.07 ± 0.00 |     19.66 ± 0.02 |  1.397    |

On the M2-Max, which has a higher memory bandwidth (so better TG performance) but lower computing power than the Ryzen-7950X, the speedup is significantly higher.