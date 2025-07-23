### 🔀 [#124](https://github.com/ikawrakow/ik_llama.cpp/pull/124) - iq2_bn_r4: fastest Bitnet CPU implementation on the planet

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-06 |
| **Updated** | 2024-12-06 |

---

#### Description

In the footsteps of #118, #119, #120, #121, #122, #123, this PR adds `IQ2_BN_R4`, a 4-rows interleaved packing of the 2-bit Bitnet quantization type `IQ2_BN`.

Here is `PP-512` for Bitner-1.58b-3B on `Zen4` (Ryzen-7950X), `ARM_NEON` (M2-Max) and `AVX2` (Ryzen-5975WX)

| Platform |  Threads | IQ2_BN | IQ2_BN_R4 | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| ARM_NEON |  8 |  246.57 ± 1.66 | 304.68 ± 0.77  | 1.236 |
| Zen4            | 16 | 631.27 ± 2.81  | 834.46 ± 2.77  | 1.322 |
| AVX2           | 32 | 694.17 ± 0.60  | 704.62 ± 0.60 | 1.0125 |

There aren't enough vector registers on AVX2 for all necessary accumulators when processing 8 right matrix columns at once. Hence, one needs two passes per left matrix interleaved row, so the gain on AVX2 is very minor. But on Zen4 we now achieve 834 t/s! In comparison, [T-MAC](https://github.com/microsoft/T-MAC), a repository with currently 607 stars making bold claims about being the fastest Bitnet CPU implementation achieves 300 t/s on the same Ryzen-7950X system. 

TG is of course memory bound, but for small number of threads I also observe a speedup. The table shows measurements for TG-128 on the above 3 platforms (table only shows results up to the number of threads that achieves maximum performance):

| Platform |  Threads | IQ2_BN | IQ2_BN_R4 | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| ARM_NEON | 1 | 21.01 ± 0.08 | 24.75 ± 0.08 | 1.178 |
|                      | 2 | 39.15 ± 0.02 | 45.48 ± 0.08 | 1.162 |
|                      | 4 | 64.39 ± 0.17 | 71.82 ± 1.84 | 1.115 |
|                      | 8 |  99.60  ± 0.53 | 100.74 ± 1.13 | 1.011 |
| Zen4            | 1 | 25.91 ± 0.12 | 30.35 ± 0.15 | 1.171 |
|                      | 2 | 45.03 ± 0.22 | 50.93 ± 0.18 | 1.131 |
|                      | 4 | 57.42 ± 0.08 | 57.40 ± 0.06 | 1.000 |
| AVX2            | 1 | 16.39 ± 0.00 | 18.42 ± 0.11 | 1.124 |
|                      | 2 | 29.94 ± 0.03 | 31.56 ± 0.01 | 1.054 |
|                      | 4 | 44.09 ± 0.02 | 45.26 ± 0.03 | 1.027 |
|                      | 8 | 47.28 ± 0.04  | 49.25 ± 0.02 | 1.042 |