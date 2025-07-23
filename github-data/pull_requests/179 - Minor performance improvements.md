### 🔀 [#179](https://github.com/ikawrakow/ik_llama.cpp/pull/179) - Minor performance improvements

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-27 |
| **Updated** | 2025-01-27 |

---

#### Description

This PR does two things
1. It changes `Q4_0_R4` to 8 interleaved rows
1. It adds the ability to apply platform specific transformations of the tensor data while repacking

Examples for the usage of 2.:
* On `ARM_NEON` it is useful to apply a `XOR` operation with a mask `0x88` to `Q4_0` quants. In this way one does not need to subtract `8` during run time. This tweak improves `Q4_0` PP performance by nearly 5% on my M2-Max CPU. This is absolutely not useful on `AVX2/Zen4`, so this becomes a platform specific transformation when run-time-repacking on an `ARM_NEON` CPU. 
* On `Zen4` one can add `128` to the signed `Q8` quants to make them unsigned (so they can be used directly in `_mmXXX_dpbusd_epi32()`. This improves `Q8_0` and `Q8_K_R8` performance by about 3%. The transformation is not useful on `ARM_NEON` (one needs signed `int8_t`'s) or vanilla `AVX2` (the `_mm256_maddubs_epi16` dot product may overflow), so it only gets applied when repacking on `Zen4`.

The table shows some comparisons for `PP-512` LlaMA-3.1-8B for the affected quantization types using Flash Attention and `Q8_0` KV-cache.

| model            | backend    |          test |      t/s (main)  |   t/s  (PR)    |  Speedup |
| ---------------- | ---------- | ------------: | ---------------: | -------------: | -------: |
| llama 8B Q4_0    | NEON       |         pp512 |    130.92 ± 0.10 |  137.39 ± 0.32 |  1.049   |   
| llama 8B Q8_K_R8 | Zen4       |         pp512 |    380.75 ± 1.52 |  390.40 ± 0.88 |  1.025   |   
| llama 8B Q8_0    | Zen4       |         pp512 |    295.62 ± 0.80 |  307.80 ± 0.34 |  1.041   |   
| llama 8B Q4_0    | Zen4       |         pp512 |    281.38 ± 0.73 |  294.43 ± 0.68 |  1.046   |   
| llama 8B Q4_0    | AVX2       |         pp512 |    302.61 ± 0.29 |  316.23 ± 0.31 |  1.045   | 

I really wanted to hit 400 t/s for `Q8_K_R8`, but it will be on another day.