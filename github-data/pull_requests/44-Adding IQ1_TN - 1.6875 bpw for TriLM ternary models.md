### 🔀 [#44](https://github.com/ikawrakow/ik_llama.cpp/pull/44) - Adding IQ1_TN - 1.6875 bpw for TriLM ternary models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-09 |
| **Updated** | 2024-09-09 |

---

#### Description

For the Bitnt-1.58b ternary models I had added `IQ1_BN` (1.625 bpw) and `IQ2_BN` (2.0 bpw) quants. But for TriLM I only added `IQ2_TN` (2.0625 bpw). This PR fills the gap adding the corresponding 1.6875 bpw quantization type `IQ1_TN`.

The matrix multiplication implementation simply reuses the existing `IQ1_BN` implementation. We just need to add the multiplication with the row scale at the end of a vector dot product between a row in the left matrix and a column in the right matrix (in `IQ1_BN` there are no scales in the quantized data, and the scale is applied separately via a `ggml_scale` operation).

While adding `IQ1_TN` to the `IQ1_BN` implementation, I noticed an optimization opportunity. As a result, this PR also improves `IQ1_BN` performance and `IQ2_BN` performance.

As [PR-8151](https://github.com/ggerganov/llama.cpp/pull/8151) has now been merged in mainline `llama.cpp` I was curious to compare `IQ1_TN` with the corresponding `TQ1_0` and `IQ2_TN` with the corresponding `TQ2_0` in `llama.cpp`. 

The CPU's used in the comparisons below are Ryzen-7950X (Zen4), Ryzen-5975WX (AVX2) and M2-Max (NEON).

### IQ1_TN vs TQ1_0, 4B TriLM model

| backend    | threads |       test |   t/s (TQ1_0)   |   t/s (IQ1_TN)   |  Speedup |
| ---------- | ------: | ---------: | --------------: | ---------------: | -------: |
| CPU (Zen4) |      16 |      pp512 |   157.50 ± 0.40 |    485.83 ± 2.23 |  3.085   |
|            |       8 |      tg128 |    51.71 ± 0.05 |     54.31 ± 0.13 |  1.050   |
| CPU (AVX2) |      32 |      pp512 |   231.71 ± 0.41 |    530.97 ± 1.29 |  2.292   |
|            |      16 |      tg128 |    55.93 ± 0.01 |     51.07 ± 0.04 |  0.913   |
| CPU (NEON) |       8 |      pp512 |    75.66 ± 0.02 |    201.25 ± 0.06 |  2.660   |
|            |       8 |      tg128 |    55.63 ± 0.02 |     58.92 ± 0.19 |  1.059   |

### IQ2_TN vs TQ2_0, 4B TriLM model

| backend    | threads |       test |   t/s (TQ1_0)   |   t/s (IQ1_TN)   |  Speedup |
| ---------- | ------: | ---------: | --------------: | ---------------: | -------: |
| CPU (Zen4) |      16 |      pp512 |   274.65 ± 0.75 |    445.31 ± 0.77 |  1.621   |
|            |       4 |      tg128 |    46.72 ± 0.10 |     48.88 ± 0.06 |  1.050   |
| CPU (AVX2) |      32 |      pp512 |   437.11 ± 0.55 |    494.08 ± 0.79 |  1.130   |
|            |       8 |      tg128 |    35.88 ± 0.04 |     43.34 ± 0.01 |  1.208   |
| CPU (NEON) |       8 |      pp512 |   117.55 ± 0.09 |    209.86 ± 0.12 |  1.785   |
|            |       8 |      tg128 |    69.33 ± 0.06 |     78.93 ± 0.26 |  1.138   |

As `IQ2_BN` PP performance is better than `IQ1_BN`, these tables indicate that my `IQ2_TN` implementation on `Zen4/AVX2` is likely not optimal. There also seem to be a bottleneck somewhere for TG with more than 8 threads than I need to look into.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2024-09-09** at **11:56:12**:<br>

For the record, here is how this PR improves `IQ1/2_BN` performance for PP

| model             | backend    | threads |          test |     t/s (main)   |    TS (PR)       |  Speedup |
| ----------------- | ---------- | ------: | ------------: | ---------------: | ---------------: | -------: |
| bitnet 3B IQ2_BN  | Zen4       |      16 |         pp512 |    515.59 ± 2.05 |    606.56 ± 6.29 | 1.176    |
| bitnet 3B IQ1_BN  | Zen4       |      16 |         pp512 |    411.92 ± 0.30 |    571.68 ± 2.42 | 1.388    |
| bitnet 3B IQ2_BN  | AVX2       |      32 |         pp512 |    637.75 ± 0.92 |    772.61 ± 1.27 | 1.211    |
| bitnet 3B IQ1_BN  | AVX2       |      32 |         pp512 |    517.17 ± 0.54 |    650.72 ± 6.02 | 1.258    |
| bitnet 3B IQ2_BN  | NEON       |       8 |         pp512 |    242.97 ± 0.60 |    247.82 ± 0.68 | 1.020    |
| bitnet 3B IQ1_BN  | NEON       |       8 |         pp512 |    207.05 ± 0.48 |    211.21 ± 0.65 | 1.020    |