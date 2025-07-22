### 🔀 [#4](https://github.com/ikawrakow/ik_llama.cpp/pull/4) - Simdify and multi-thread tanh

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-07-27 |
| **Updated** | 2024-07-27 |

---

#### Description

It seemed Gemma-2 performance is lower than expected for its size. Looking at the architecture, I noticed that `tanh` is used in each layer, and then at the end for soft-caping the final output. `ggml` had `tanh` set to be computed with a single thread. Combined with `tanh(x)` being a pretty expensive operation, this resulted in a significant fraction of the time being spent in the `tanh` operation.

After multi-threading `ggml_vec_soft_max_f32` and simd-ifying the `tanh` computation, I observe a 33% gain in prompt processing speed for Gemma-2-9b (!!!) TG is of course memory bound, but despite this, we still get a ~2% boost at 4 threads (which gives max TG performance on my Ryzen-7950X).

Simd-ifying:
We have
```
   tanh(x) = (exp(2*x) - 1)/(exp(2*x) + 1)
```
so we can just use Justine Tunney's SIMD implementation for the exponential function.