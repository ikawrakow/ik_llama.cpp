## 🔀 [Pull Request #261](https://github.com/ikawrakow/ik_llama.cpp/pull/261) - Compile time option to use bf16 for quants without MMQ kernels

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/use_bf16_when_no_mmq` |
| **Target Branch** | `main` |
| **Created** | 2025-03-17 |
| **Updated** | 2025-03-18 |
| **Merged** | 2025-03-18 |

---

## 📄 Description

The `IQ2_KS, IQ2_K, ..., IQ6_K` quantization types do not have MMQ kernels, so matrix multiplications for model weights quantized with these types are done via dequantization to `fp16` and `cublasGemmEx` GEMM using `fp16` precision. For the DeepSeek series of MoE models this leads to NaNs.

Ideally I should add MMQ kernels for these quantization types. But for now, the PR provides a quick fix: dequantize to `bf16` and use `bf16` cuBLAS GEMM. This is added as a compile time option enabled via
```
cmake -DGGML_CUDA_IQK_FORCE_BF16 $other_cmake_options
```
(or, if you like me prefer using `ccmake`, after pulling the PR, `cmake .. && ccmake .`, and then set the `GGML_CUDA_IQK_FORCE_BF16` to `ON`). 

I have tested with DeepSeek-Lite quantized with `IQ4_KSS` and `IQ4_K`. In both cases I get NaNs when running `perplexity` on the main branch. Turning on the `GGML_CUDA_IQK_FORCE_BF16` option provided by this PR results in meaningful PPL values.

@davidsyoung This should solve the issues with the `IQ4_KSS` DeepSeek-R1 model you created.

---

## 💬 Conversation

👤 **davidsyoung** commented on **2025-03-17** at **23:38:28**

Awesome! Will re-quant over night and test tomorrow!

---

👤 **saood06** commented on **2025-03-17** at **23:43:23**

> Awesome! Will re-quant over night and test tomorrow!

In case you still have the old quants, you can just use those with the new code you don't have to make new quants.

---

👤 **davidsyoung** commented on **2025-03-17** at **23:45:25**

Unfortunately I don’t! My cache drive is limited so I tend to delete pretty soon.