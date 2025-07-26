### [Pull Request #172](https://github.com/ikawrakow/ik_llama.cpp/pull/172) - CPU Flash Attention improvements

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fattn_bf16` |
| **Target Branch** | `main` |
| **Created** | 2025-01-15 |
| **Updated** | 2025-01-15 |
| **Merged** | 2025-01-15 |

---

#### Description

This PR
* Improves FA CPU performance for long contexts
* Fixes K-cache quantized to `Q8_0` when not using FA. This was broken because online `Q8_0` quantization packed quants into blocks of 128 (`block_q8_0_x4`), so `K*Q` became garbage when using `Q8_0` quantized K-cache without FA.

FA performance improvements are for `AVX2/Zen4`. The following table shows `PP-512` comparison between the main branch and this PR with FA using `bf16` or `Q8_0` for KV cache. Model is LLaMA-3.1-8B quantized to `IQ4_XS` and run-time-repacked to `IQ4_XS_R4`. The CPU is Ryzen 7950X. When the quoted uncertainty in the table is zero, I have run just a single repetition in `llama-bench` (it takes quite a while to process 16k or even 32k tokens)

   | type_k | type_v | fa | rtr |          test |    t/s (main)    |    t/s (pr)      | Speedup |
| -----: | -----: | -: | --: | ------------: | ---------------: | ---------------: | ------: |
|   bf16 |   bf16 |  1 |   1 |         pp128 |    275.27 ± 1.63 |    278.40 ± 1.60 | 1.011   |   
|   bf16 |   bf16 |  1 |   1 |         pp256 |    276.16 ± 3.46 |    283.51 ± 1.22 | 1.027   |   
|   bf16 |   bf16 |  1 |   1 |         pp512 |    274.71 ± 0.51 |    276.83 ± 0.36 | 1.008   |   
|   bf16 |   bf16 |  1 |   1 |        pp1024 |    265.81 ± 1.65 |    270.05 ± 0.41 | 1.016   |   
|   bf16 |   bf16 |  1 |   1 |        pp2048 |    256.95 ± 0.39 |    260.11 ± 0.14 | 1.012   |   
|   bf16 |   bf16 |  1 |   1 |        pp4096 |    237.97 ± 0.37 |    242.29 ± 0.75 | 1.018   |   
|   bf16 |   bf16 |  1 |   1 |        pp8192 |    206.34 ± 1.25 |    213.98 ± 0.35 | 1.037   |   
|   bf16 |   bf16 |  1 |   1 |       pp16384 |    156.40 ± 0.00 |    173.44 ± 0.00 | 1.109   |   
|   bf16 |   bf16 |  1 |   1 |       pp32768 |     82.97 ± 0.00 |    122.47 ± 0.00 | 1.476   |   
|   q8_0 |   q8_0 |  1 |   1 |         pp128 |    273.44 ± 1.04 |    279.27 ± 1.43 | 1.021   |   
|   q8_0 |   q8_0 |  1 |   1 |         pp256 |    278.57 ± 1.03 |    283.00 ± 0.63 | 1.016   |   
|   q8_0 |   q8_0 |  1 |   1 |         pp512 |    271.56 ± 0.05 |    275.97 ± 0.79 | 1.016   |   
|   q8_0 |   q8_0 |  1 |   1 |        pp1024 |    264.31 ± 0.89 |    269.35 ± 0.33 | 1.019   |   
|   q8_0 |   q8_0 |  1 |   1 |        pp2048 |    253.70 ± 0.24 |    258.22 ± 0.36 | 1.018   |   
|   q8_0 |   q8_0 |  1 |   1 |        pp4096 |    232.07 ± 0.88 |    236.83 ± 1.38 | 1.021   |   
|   q8_0 |   q8_0 |  1 |   1 |        pp8192 |    199.90 ± 1.37 |    204.74 ± 0.34 | 1.024   |   
|   q8_0 |   q8_0 |  1 |   1 |       pp16384 |    153.62 ± 0.00 |    164.50 ± 0.00 | 1.071   |   
|   q8_0 |   q8_0 |  1 |   1 |       pp32768 |    103.48 ± 0.00 |    113.35 ± 0.00 | 1.095   |