### 🔀 [#482](https://github.com/ikawrakow/ik_llama.cpp/pull/482) - Trellis quants: faster CPU prompt processing

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-01 |
| **Updated** | 2025-06-01 |

---

#### Description

The trellis quants `IQ2_KT, IQ3_KT, IQ4_KT` are very slow on the CPU. On the main branch using BLAS results in a better prompt processing performance. But BLAS is slower for basically all other data types, so that's not a good idea.

This PR improves prompt processing speed of the trellis quants by adding "dequantizing GEMM". Basically, blocks of trelis quantized weights are converted to `fp32` (AVX2 )or `fp16` (ARM) on-the-fly, and then the `fp32/fp16` GEMM kernels are used to multiply the bock with the entire right matrix. This amortizes the very high dequantization cost much better than the standard kernel templates that allow up to 8 right matrix columns.

On my `Zen4/AVX2` CPUs this results in a better PP performance than using BLAS (or Intel MKL). On the M2-Max PP performance is about 80% of BLAS (which tells me that my `ARM_NEON` GEMM kernel for `fp16` is not optimal).

TG performance is not affected by the PR and is still very low.

Here is a PP-512 performance comparison between the main branch (without BLAS) and this PR for LlaMA-3.1-8B on a Ryzen-7950X CPU

| quant | PP-512 (main) | PP-512 (PR) | Speedup |
| ---: | ---: | ---: | ---: |
| IQ2_KT | 57.98 |  132.47 |  2.28 |
| IQ3_KT | 47.44 |  127.80 |  2.69 |
| IQ4_KT | 40.09 |  126.31 |  3.15 |