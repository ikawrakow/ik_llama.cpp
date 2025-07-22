### 🔀 [#73](https://github.com/ikawrakow/ik_llama.cpp/pull/73) - CUDA: faster float -> iq4_nl conversion

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-01 |
| **Updated** | 2024-10-01 |

---

#### Description

I had forgotten that `IQ4_NL` can be used for kv-cache on CUDA. It can be, but it is slower than `fp16, q4_0, ...`.

This PR speeds up the CUDA `IQ4_NL` quantization. The following table shows a performance comparison between the main branch and this PR for LLaMA-3.1-8B with FA enabled and `IQ4_NL` cache running on RTX-4080

| model           |  type_k | type_v |          test |    t/s (main)    |   t/s (PR)      |  Speedup |
| --------------- |  -----: | -----: | ------------: | ---------------: | --------------: | -------: |
| llama 8B Q4_K_S | iq4_nl | iq4_nl |         pp512 |  6933.65 ± 14.39 | 7274.27 ± 13.54 |  1.049   |
| llama 8B Q4_K_S | iq4_nl | iq4_nl |        pp8192 |   5557.13 ± 1.59 | 5771.27 ± 6.53  |  1.039   |
| llama 8B Q4_K_S | iq4_nl | iq4_nl |       pp32768 |   3300.51 ± 3.99 | 3372.49 ± 4.25  |  1.022   |

In comparison, `PP(512, Q4_0) = 7389.61` and `PP(32768, Q4_0) = 3409.85`, so `IQ4_NL` is 1.6% / 1.1% slower after the PR, which I think is an acceptable tradeoff given the improved accuracy:
```
PPL(Q4_0)   = 6.7648
PPL(IQ4_NL) = 6.6992
```
The `IQ4_NL` result is comparable to `Q4_1` kv-cache, which is 11% larger.

Note that the CUDA `IQ4_NL` quantization method is not the same as the one used when quantizing models. It must be fast else the performance penalty would be too large. Thus, kv-cache `IQ4_NL` quantization quality is not as good as when quantizing model weights, and hence we can only get to `Q4_1`quantization quality.