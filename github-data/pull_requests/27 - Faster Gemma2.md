### [Pull Request #27](https://github.com/ikawrakow/ik_llama.cpp/pull/27) - Faster Gemma2

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Created** | 2024-08-27 |
| **Updated** | 2024-08-27 |
| **Merged** | 2024-08-27 |

---

#### Description

In a [previous PR](https://github.com/ikawrakow/ik_llama.cpp/pull/9) I has fused `scale - tanh - scale` used for "soft-capping" activations into a `GGML_OP_SOFTCAP` operation. This PR further fuses `GGML_OP_SOFTCAP` with `GGML_OP_SOFT_MAX` into a new `GGML_OP_SOFT_CAP_MAX` operation. This is useful for, e.g., self-attention in the Gemma-2 series of models, and leads to a significant performance increase.

In addition, "soft-capping" is added to flash attention. I see this has also been done in mainline `llama.cpp` in  PR-8542 and PR-9159.

Here some performance comparisons to `llama.cpp` (build 3631) for Gemma-2-2b on `CUDA` (RTX-4080), `Metal` (30-core M2-Max GPU),  `AVX2` (Ryzen-7950X) and `ARM_NEON` (M2-Max CPU). The model is quantized with `Q4_K_S` (the performance gap between this repo and mainline `llama.cpp` is smaller for this quantization type compared to most other quants).

### No Flash attention

| backend    | ngl | threads |          test |   t/s (llama.cpp) |  t/s (PR)        |   Speedup |
| ---------- | --: | ------: | ------------: | ----------------: | ---------------: | --------: |
| CUDA       | 100 |       1 |         tg128 |    239.20 ± 0.27  |  244.47 ± 0.42   |  1.022    |   
|            | 100 |       1 |         pp512 | 18413.90 ± 566    | 18824.91 ± 480   |  1.022    |   
|            | 100 |       1 |        pp2048 | 17827.18 ± 106    | 18307.66 ± 77    |  1.027    |   
|            | 100 |       1 |        pp8192 |   8814.67 ± 7.27  | 11673.96 ± 8.07  |  1.324    |   
|            | 100 |       1 |       pp32768 |  2827.13 ± 12.12  | 4634.12 ± 4.84   |  1.639    |   
| AVX2       |   0 |       4 |         tg128 |     32.68 ± 0.08  |     35.26 ± 0.05 |  1.079    |   
|            |   0 |      16 |         pp512 |    278.34 ± 1.04  |    620.40 ± 3.24 |  2.229    |   
|            |   0 |      16 |        pp2048 |    217.57 ± 0.70  |    562.58 ± 2.31 |  2.586    |   
|            |   0 |      16 |        pp8192 |    111.29 ± 0.15  |    414.44 ± 0.83 |  3.724    |   
|            |   0 |      16 |       pp32768 |     35.78 ± 0.00  |    199.58 ± 0.00 |  5.578    |   
| Metal      | 100 |       8 |         tg128 |     88.82 ± 0.19  |     91.06 ± 0.18 |  1.025    |
|            | 100 |       8 |         pp512 |   1427.74 ± 1.44  |   1512.66 ± 0.59 |  1.059    |
|            | 100 |       8 |        pp2048 |   1363.51 ± 0.62  |   1456.12 ± 0.73 |  1.068    |
|            | 100 |       8 |        pp8192 |   1093.02 ± 0.86  |   1224.56 ± 0.52 |  1.120    |
|            | 100 |       8 |       pp32768 |    572.65 ± 1.13  |    728.75 ± 5.56 |  1.272    |
| ARN_NEON   |   0 |       8 |         tg128 |     54.06 ± 0.15  |     62.49 ± 0.18 |  1.156    |   
|            |   0 |       8 |         pp512 |    148.92 ± 0.15  |    243.09 ± 0.06 |  1.632    |
|            |   0 |       8 |        pp2048 |    130.66 ± 1.84  |    226.46 ± 5.41 |  1.733    |
|            |   0 |       8 |        pp8192 |     97.95 ± 3.57  |    189.65 ± 4.30 |  1.936    |

For very large prompts (pp32768) the performance difference is striking, reaching 5.5X for `AVX2`!

### Flash attention

Flash attention is only useful on CUDA (on the 3 other platforms I have available performance is lower with flash attention), so here only CUDA results:

| backend    | ngl | threads | fa |          test |   t/s (llama.cpp) |   t/s (PR)       |    Speedup  |
| ---------- | --: | ------: | -: | ------------: | ----------------: | ---------------: | ----------: |
| CUDA       | 100 |       1 |  1 |         tg128 |    251.86 ± 0.56  |    256.15 ± 0.76 |  1.017      |
| CUDA       | 100 |       1 |  1 |         pp512 | 19127.14 ± 529.58 | 19712.11 ± 167.06|  1.031      |
| CUDA       | 100 |       1 |  1 |        pp2048 | 18641.99 ± 72.13  | 19823.18 ± 91.26 |  1.063      |
| CUDA       | 100 |       1 |  1 |        pp8192 | 13566.85 ± 111.75 | 16108.68 ± 30.32 |  1.187      |
| CUDA       | 100 |       1 |  1 |       pp32768 |   6472.16 ± 4.43  |   9053.46 ± 9.68 |  1.399      |

40% faster for 32k tokens is quite nice.