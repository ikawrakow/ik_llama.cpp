### 📝 [#249](https://github.com/ikawrakow/ik_llama.cpp/issues/249) - CUDA: results for MoE models are not reproducible

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-10 |
| **Updated** | 2025-03-25 |

---

#### Description

### What happened?

Running `llama-perplexity` with the same MoE model (observed with DeepSeek-Lite) produces different PPL values in each run.

The non-reproducibility is  not observed for TG when using the same random seed.

### Name and Version

All versions. The issue is also present in mainline `llama.cpp` (tested with latest as of today (`build: 4858 (1e2f78a0)`), so it is not due to a change I made. I think the non-reproducibility is due to [this kernel](https://github.com/ikawrakow/ik_llama.cpp/blob/b096a5de7a9bdf516bb20729d5d0a3b2a12cba2f/ggml/src/ggml-cuda.cu#L2039), where the order in which the rows of the `src1` tensor are copied to contiguous memory depends on how the stars have fallen today.


### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell

```