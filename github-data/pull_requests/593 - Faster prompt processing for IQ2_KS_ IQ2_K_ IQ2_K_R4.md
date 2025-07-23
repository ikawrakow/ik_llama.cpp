### 🔀 [#593](https://github.com/ikawrakow/ik_llama.cpp/pull/593) - Faster prompt processing for IQ2_KS, IQ2_K, IQ2_K_R4

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-08 |
| **Updated** | 2025-07-08 |

---

#### Description

Here a comparison to the main branch for LlaMA-3.1-8B on RTX-4080

| model               |          test |    t/s (main)    |   t/s (PR)       |  Speedup |
| ------------------- | ------------: | ---------------: | ---------------: | -------: |
| llama 8B IQ2_KS     |         pp512 | 7834.83 ± 158.78 | 8613.55 ± 159.26 |  1.099   |   
| llama 8B IQ2_K      |         pp512 | 6781.98 ± 115.12 | 7165.57 ± 133.82 |  1.056   |   
| llama 8B IQ2_K_R4   |         pp512 | 6587.47 ± 136.21 | 7344.46 ± 139.87 |  1.115   |   

I have adjusted the threshold at which dequantize+cuBLAS kicks in for `IQ2_K` and `IQ2_K_R4` to 2048 tokens as MMQ is now faster on my GPU for u-batches up to about 2k tokens.

`IQ2_KS` is now the second fastest quant for prompt processing after `IQ2_KT`.

The trick is to lookup 4 values at once, which is feasible for the 2-bit quants as there are only 256 possibilities. In one of the commits there is also an alternative version that does not use lookup at all, which is faster than the main branch but slower than the 4-value lookup.