### 🔀 [#197](https://github.com/ikawrakow/ik_llama.cpp/pull/197) - FA: Add option to build all FA kernels

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-09 |
| **Updated** | 2025-02-09 |

---

#### Description

Similar to the CUDA situation.
It is OFF by default.
If OFF, only `F16, Q8_0, Q6_0`, and, if the CPU provides native `BF16` support, `BF16` CPU FA kernels will be included.
To enable all,
```
cmake -DGGML_IQK_FA_ALL_QUANTS=1 ...
```

This cuts compilation time for `iqk_mul_mat.cpp` by almost half (45 seconds vs 81 seconds on my Ryzen-7950X).
This is poor men's solution of the long build time until #183 is tackled.