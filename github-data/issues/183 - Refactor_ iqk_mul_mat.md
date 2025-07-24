### 📝 [#183](https://github.com/ikawrakow/ik_llama.cpp/issues/183) - Refactor: iqk_mul_mat

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-30 |
| **Updated** | 2025-05-22 |

---

#### Description

### Background Description

`iqk_mul_mat.cpp` compilation time has become unacceptably long. If I keep going that way soon it will rival CUDA build times.

As an experiment at some point I factored out the Flash Attention (FA) part from the matrix multiplication code. This resulted in a FA build time of ~45 seconds and GEMM/GEMV build time of ~30 seconds, so better than the ~75 seconds I observe for `iqk_mul_mat.cpp` on my Ryzen-7950X, but still far from really useful, so I did not commit.  

### Possible Refactor Approaches

_No response_