### 🔀 [#9](https://github.com/ikawrakow/ik_llama.cpp/pull/9) - Fused soft cap and SIMD-ified GeLU 

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-08-02 |
| **Updated** | 2024-08-20 |

---

#### Description

Some models use a so called "soft cap" in their attention portions, some may use a "soft cap" also for the final output. This is currently implemented as
```
x = ggml_scale(x, 1/softcap_parameter)
x = ggml_tanh(x)
x = ggml_scale(x, softcap_parameter)
```
By fusing these 3 operations into a single kernel, we gain about 1% on all tested backends (`AVX2, NEON, CUDA, Metal`).

Also added a SIMD-ified implementation of GeLU (`AVX512, AVX2, NEON`). This gives another ~1% performance gain on `AVX512/AVX2`. The `ggml` GeLU lookup table is faster on my M2-Max CPU, so using that on `NEON`.

The above is based on just checking the `PP-512` and `TG-128` performance. But soft cap is used in the attention portion of Gemma-2 models, so let's look at a large context where self-attention plays a more significant role. I'll use Gemma-2-9b and a context of 8192 tokens, but instead of comparing to the main branch in this repository I'll compare against the current mainline `llama.cpp` version. The following table compares `PP-8192` performance for `AVX2` (Ryzen-7950X), `CUDA` (RTX-4080), `ARM_NEON` (M2-Max CPU), and `Metal` (30-core M2-Max GPU). To keep the table small, results are given just for `Q4_K_S` quantization

| backend    |          test | t/s (llama.cpp)  |  t/s (this PR) |  Speedup |
| ---------- | ------------: | ---------------: | -------------: | -------: |
| AVX2       |        pp8192 |     32.90 ± 0.00 | 103.16 ± 0.00  | 3.136    |
| CUDA       |        pp8192 |   2495.19 ± 1.20 | 3068.44 ± 0.68 | 1.230    |
| NEON       |        pp8192 |     26.44 ± 0.00 |  48.30 ± 0.00  | 1.827    |
| Metal      |        pp8192 |    294.33 ± 0.40 | 325.78 ± 1.94  | 1.107    |

As I have not changed much in the `CUDA` and `Metal` back-ends, the 23% (`CUDA`) or 10% (`Metal`) performance difference comes from this one fused operation! On `AVX2` the performance gap has grown to 3.136X up from the 1.874X we had from the improved matrix multiplications (see 1st table on the main page). On `ARM_NEON` this implementation is now 1.827X faster, up from 1.639X. I think that the much larger increase in relative performance on the Ryzen-7950X can be explained with its less capable memory subsystem: for a context of 8192 tokens the `K*Q` tensor on which the soft-cap is applied no longer fits in the cache, so the `ggml_scale + ggml_tanh + ggml_scale` implementation in `llama.cpp` requires it to be loaded from / stored to main memory 3 times instead of just once when these 3 operations are fused into a single op.