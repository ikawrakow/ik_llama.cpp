### 🔀 [#127](https://github.com/ikawrakow/ik_llama.cpp/pull/127) - Q4_0_R4 on CUDA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2024-12-08 |
| **Updated** | 2025-01-09 |

---

#### Description

With the massive improvements in prompt processing speed on the CPU achieved via interleaving 4 tensor rows (see #118, #119, #120, #121, #122, #123, #124), I was curious to see if one can get a good implementation for the `X_R4` quants on CUDA. This PR is a POC that implements CUDA dequantization and matrix x vector multiplication for `Q4_0_R4`. It achieves the same TG speed as `Q4_0`. It was disappointing to not get a speedup via row interleaving, but at least there is no performance regression. To make it a full PR I should also implement quantized matrix x matrix multiplication for `Q4_0_R4` (here it is done via dequantize to `f16` and cuBLAS, so it is slower than `Q4_0` MMQ).