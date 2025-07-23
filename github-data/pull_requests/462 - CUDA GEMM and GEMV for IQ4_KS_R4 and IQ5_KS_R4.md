### 🔀 [#462](https://github.com/ikawrakow/ik_llama.cpp/pull/462) - CUDA GEMM and GEMV for IQ4_KS_R4 and IQ5_KS_R4

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-26 |
| **Updated** | 2025-05-27 |

---

#### Description

This PR is a follow up to PR #461 and adds CUDA implementation for `IQ4_KS_R4` and `IQ5_KS_R4`

Note: because GEMM is implemented via dequantize+cuBLAS, if you want to use a IQX_K_R4 DeepSeek-V3/R1 model on the GPU, you may need to build with -DGGML_CUDA_IQK_FORCE_BF16=1 to force bf16 arithmetic with cuBLAS as fp16 has been noted to lead to numerical instabilities and garbled output. I did not enable GGML_CUDA_IQK_FORCE_BF16 by default as it reduces prompt processing performance while, as far as I can tell, bf16 is only required for DeepSeek.