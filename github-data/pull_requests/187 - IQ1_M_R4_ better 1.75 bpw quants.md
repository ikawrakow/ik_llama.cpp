### 🔀 [#187](https://github.com/ikawrakow/ik_llama.cpp/pull/187) - IQ1_M_R4: better 1.75 bpw quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-06 |
| **Updated** | 2025-02-06 |

---

#### Description

Following in the foot steps of #185, this PR adds `IQ1_M_R4`, a 4-row interleaved version of `IQ1_M`. 

* I have removed the `f16` super-block scale (replaced with a `f16` per row scale) and have changed the 3-bit `IQ1_M` block scales with 4 bit. Hence, we end up using the same 1.75 bpw as `IQ1_M`.
* The above change allows to implement `IQ1_M_R4` with a block size of 32. I wanted to have this because DeepSeek-Lite, the model I'm testing with, has a lot of tensors with row sizes not divisible by 256, so a significant fraction of tensors gets quantized to `IQ4_NL` when using `IQ1_M`
*  Quantization mixes for MoE models are adjusted. Today's mainline `llama.cpp` arrives at a context-512 perplexity (`PPL(512)` in what follows) of 20.75 for DeepSeek-Lite using 2.74 bpw with `IQ1_M`. The `IQ1_M_R4` quantization in this PR gets `PPL-512 = 8.85` with 1.966 bpw for the repeating layers.
* `IQ1_M_R4` is **much faster** on the CPU compared to `IQ1_M` (see tables below). I never implemented iqk-style GEMM for `IQ1_S/IQ1_M`, so these quantization types run at the snail speed of mainline `llama.cpp`.
* Caveat: it is CPU only for now.

The following table compares prompt processing (pp512) and token generation (tg128) speed for LLaMA-3.1-8B on `AVX2` (Ryzen-5975WX), `Zen4` (Ryzen-7950X) and `ARM_NEON` (M2-Max CPU). I didn't use DeepSeek-Lite for this comparison to avoid the difference in quantization types one ends up with due to not all tensors having row sizes that are multiple of 256.

| platform   | threads |          test |     t/s (IQ1_M)  |   t/s (IQ1_M_R4) |  Speedup |
| ---------- | ------: | ------------: | ---------------: | ---------------: | -------: |
| AVX2       |      32 |         pp512 |     43.98 ± 0.09 |    187.94 ± 0.21 |  4.273   |
| Zen4       |      16 |         pp512 |     26.70 ± 0.03 |    149.57 ± 0.31 |  5.602   |
| NEON       |       8 |         pp512 |     17.61 ± 0.03 |     95.04 ± 0.16 |  5.397   |
| AVX2       |       2 |         tg128 |      2.66 ± 0.00 |      3.96 ± 0.00 |  1.489   |
|            |       4 |         tg128 |      5.25 ± 0.00 |      7.76 ± 0.00 |  1.478   |
|            |       8 |         tg128 |      9.93 ± 0.16 |     13.71 ± 0.01 |  1.381   |
|            |      16 |         tg128 |     17.14 ± 0.00 |     22.60 ± 0.01 |  1.319   |
|            |      32 |         tg128 |     23.91 ± 0.01 |     25.39 ± 0.02 |  1.062   |
| Zen4       |       2 |         tg128 |      3.39 ± 0.00 |      5.29 ± 0.00 |  1.560   |
|            |       4 |         tg128 |      6.50 ± 0.00 |     10.19 ± 0.00 |  1.568   |
|            |       8 |         tg128 |     11.68 ± 0.01 |     17.54 ± 0.01 |  1.502   |
|            |      16 |         tg128 |     19.13 ± 0.05 |     25.91 ± 0.43 |  1.354   |
| NEON       |       2 |         tg128 |      4.16 ± 0.00 |      5.27 ± 0.01 |  1.267   |
|            |       4 |         tg128 |      7.88 ± 0.00 |      9.99 ± 0.01 |  1.268   |
|            |       8 |         tg128 |     14.74 ± 0.26 |     19.19 ± 0.01 |  1.302   |