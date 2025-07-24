### 🔀 [#28](https://github.com/ikawrakow/ik_llama.cpp/pull/28) - Binary KQ mask

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2024-08-28 |

---

#### Description

This PR is another attempt to improve performance for large contexts, see #25 

Basically, when we want to process a very long context, the KQ mask, which is stored as `f32` (or `f16`, if using flash attention), becomes quite significant in size. If running on the GPU, the cost for copying the KQ mask to the GPU (the mask is created on the host CPU) becomes non-negligible. If running on a CPU that has limited memory bandwidth (basically all `x86` or `x86_64`), the KQ mask may not fit in the cache, or if it does fit it reduces the cache available for other data by a significant amount, which results in a measurable impact on the performance of the `SOFT_MAX` (or the new fused `SOFT_CAP_MAX`) operation. Hence, it will be desirable to reduce the size of the KQ mask.

If not using ALiBi  (basically almost always these days), the KQ mask stored 2 values: `0,  -INFINITY`. It can therefore be represented as a binary mask, thus reducing its size by a factor of 32.

This PR adds an option to use a binary KQ mask. It is off by default as not all platforms are implemented, but can be turned on using `-bkq` or `--binary-kq` on the command line. This will have no effect if flash attention is used (KQ mask remains `f16` as before). If turned on but not supported by the back-end (non-`AVX512` CPUs), the program will assert and terminate.

I see 3-5% performance gains on CUDA and a Ryzen-7950X CPU for a context of 32k tokens, and about 2-3% on Metal for a context of 16k. So, nothing earth-shattering. and hence not quite convinced to merge it.