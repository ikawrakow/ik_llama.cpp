### 🔀 [#206](https://github.com/ikawrakow/ik_llama.cpp/pull/206) - MLA: allow Q8_0 K-cache for MLA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-13 |
| **Updated** | 2025-02-13 |

---

#### Description

After PR #205 we have two KV caches left when using MLA:
* `kv_l` - contiguous, not transposed
* `kvt_l` - a transposed version of `kv_l`

`kv_l` can be quantized, and this PR adds the necessary changes.
`kvl_t`, being a transposed version of `kv_l`, cannot be quantized. It can be eliminated by setting `MLA_USE_TRANSPOSED_CACHE` to 0 in `llama.cpp` (but then `kv_l` cannot be quantized as making a contiguous transposed tensor out of a quantized tensor as needed during inference does not work at this point).

Apart from reducing required KV cache memory, a quantized `kv_l` cache can also slightly improve TG performance after a long prompt. Here is a comparison between the main branch and this PR for `tg64@ppN` for different prompt lengths `N`. Model is `IQ4_XS` quantized DeepSeek-Lite. The results for the main branch are for `fp16` `kv_l` and `kvt_l` cache, the PR used `Q8_0` for `kv_l` and `bf16` for `kvt_l` (using `bf16` only makes sense for a CPU with native support, such as the Ryzen-7950X used to run the benchmark)

  | model                |          test |    t/s (main)    |     t/s (PR)     |  Speedup |
| -------------------- | ------------: | ---------------: | ---------------: | -------: |
| deepseek2 16B IQ4_XS |    tg64@pp128 |     33.80 ± 0.00 |     33.67 ± 0.01 |  0.996   |
| deepseek2 16B IQ4_XS |    tg64@pp256 |     32.76 ± 0.01 |     33.55 ± 0.01 |  1.024   |
| deepseek2 16B IQ4_XS |    tg64@pp512 |     32.68 ± 0.05 |     32.31 ± 0.00 |  0.989   |
| deepseek2 16B IQ4_XS |   tg64@pp1024 |     32.02 ± 0.00 |     32.07 ± 0.00 |  1.002   |
| deepseek2 16B IQ4_XS |   tg64@pp2048 |     30.31 ± 0.03 |     30.93 ± 0.00 |  1.020   |
| deepseek2 16B IQ4_XS |   tg64@pp4096 |     27.54 ± 0.10 |     28.79 ± 0.07 |  1.045   |
| deepseek2 16B IQ4_XS |   tg64@pp8192 |     23.12 ± 0.01 |     25.21 ± 0.02 |  1.090   |
| deepseek2 16B IQ4_XS |  tg64@pp16384 |     18.74 ± 0.09 |     19.81 ± 0.05 |  1.057   |