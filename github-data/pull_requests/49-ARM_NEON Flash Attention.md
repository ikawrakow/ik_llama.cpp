### 🔀 [#49](https://github.com/ikawrakow/ik_llama.cpp/pull/49) - ARM_NEON Flash Attention

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-11 |
| **Updated** | 2024-09-11 |

---

#### Description

This PR adds Flash Attention for `ARM_NEON`. The `Zen4/AVX2` implementation is reused with a few platform specific additions for `ARM_NEON`. As with `AVX2`, it is just for `fp16` kv-cache for now.

On `ARM_NEON` `fp16` arithmetic is used to compute `K*Q` (unlike `Zen4/AVX2`, which use `fp32`). Initially I was also using `fp16` to operate on the `K*Q` product (the `soft_max` related stuff), and that worked fine for the models I was using for testing (Gemma2-2b, TriLM-4B). But `fp16` fails for LLaMA-3.1-8B, so I had to change for `fp32`<sup>1</sup>. 

Performance gains are not as good as `Zen4/AVX2`. My guess is that due to the significantly higher memory bandwidth of the M2 Max used for testing the `ARM_NEON` implementation (compared to the `Zen4/AVX2` systems I have available), the penalty of not having intermediate results in the cache when computing `KQV` is less. Nevertheless, for LLaMA-3.1-8B at a context of 2k tokens, using FA is about 4% faster than not using FA on the M2 Max. In contrast, the mainline `llama.cpp` FA implementation is ~17% slower than no-FA.    

<sup>1</sup> I must admit I don't really understand why because `expf` (and `tanh` when soft-capping is involved) are computed in `fp32` even when `K*Q` is `fp16`, so possibly there was a bug that I was not able to find in the `fp32 <-> fp16` conversions rather than a loss of precision.