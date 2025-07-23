### 🔀 [#589](https://github.com/ikawrakow/ik_llama.cpp/pull/589) - CUDA: small PP performance improvement for MoE models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-06 |
| **Updated** | 2025-07-07 |

---

#### Description

This PR brings a small (2-3%) prompt processing performance improvement on CUDA for quantized MoE models (when  `-fmoe` is used).

Instead of first copying activations to contiguous memory and the quantizing, quantization is done directly using the row mapping IDs, thus saving the associated kernel launch overhead. 

Here is a performance comparison for `Q4_0` quantized DeepSeek-Lite on RTX-4080 using `-mla 3 -fa -fmoe -b 4096 -ub 4096`

### Main branch

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    0.480 |  8532.52 |    5.640 |   181.55 |
|  4096 |   1024 |   4096 |    0.566 |  7240.62 |    5.904 |   173.43 |
|  4096 |   1024 |   8192 |    0.674 |  6073.99 |    6.143 |   166.68 |
|  4096 |   1024 |  12288 |    0.789 |  5189.61 |    6.421 |   159.47 |

### PR

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    0.469 |  8738.41 |    5.638 |   181.61 |
|  4096 |   1024 |   4096 |    0.554 |  7388.85 |    5.909 |   173.29 |
|  4096 |   1024 |   8192 |    0.670 |  6117.30 |    6.148 |   166.57 |
|  4096 |   1024 |  12288 |    0.779 |  5256.86 |    6.435 |   159.14 |