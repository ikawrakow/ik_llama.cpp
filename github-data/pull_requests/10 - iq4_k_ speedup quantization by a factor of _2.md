### 🔀 [#10](https://github.com/ikawrakow/ik_llama.cpp/pull/10) - iq4_k: speedup quantization by a factor of ~2

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-08-03 |
| **Updated** | 2024-08-03 |

---

#### Description

It is interesting to observe that `clang` produces code that is ~6X faster than the `GCC` result on a simple benchmark that measures the speed of the `best_index_iq4n` function (which is the bottleneck during `IQ4_K` quantization). But when this is used in practice in  `quantize_row_iq4_k_impl_bs16`, the `clang` executable is actually worse than the `GCC` executable. Either way, both compilers need a hand, so this PR gives it to them. This gives us a ~2X speedup in the `IQ4_K` quantization.