### 🔀 [#429](https://github.com/ikawrakow/ik_llama.cpp/pull/429) - Option to enable or disable the CPU FA kernels

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-17 |
| **Updated** | 2025-05-17 |

---

#### Description

The compilation of `iqk_mul_mat.cpp` takes extremely long - currently 2m22s on my Ryzen-7950X CPU, with some users reporting times in the range of 30 minutes on an Antroid phone using Termux . This is to a large extent due to the Flash Attention (FA) kernels. Hence, this PR adds a `cmake` option to enable or disable the CPU FA kernels. It is set on by default, and can be changed using
```
cmake -DGGML_IQK_FLASH_ATTENTION=OFF ...
```
Setting it to off reduces compilation time of `iqk_mul_mat.cpp` to 25 seconds on the Ryzen-7950 CPU, so a speedup of 5.7X. Hopefully this will make it easier to build `ik_llama.cpp` on an Android phone.

If `GGML_IQK_FLASH_ATTENTION` is set to `OFF`, FA is still available but will be computed using the `ggml` implementation, which is very slow on any CPU I have tried.