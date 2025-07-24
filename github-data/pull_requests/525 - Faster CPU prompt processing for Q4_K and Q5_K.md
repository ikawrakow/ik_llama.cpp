### 🔀 [#525](https://github.com/ikawrakow/ik_llama.cpp/pull/525) - Faster CPU prompt processing for Q4_K and Q5_K

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-12 |
| **Updated** | 2025-06-13 |

---

#### Description

These two quantization types are quite popular, so I thought it makes sense to improve their performance. The repacked variants `Q4_K_R4` and `Q5_K_R4`  do not have a CUDA implementation, so repacking is not useful in a hybrid CPU/GPU setup where it may be better to offload tensors stored in RAM to the GPU when processing large batched.

The PR uses the same trick as #515, #516, #517, #518. When processing batches `>= 32` tokens, `Q4_K` or `Q5_K` quantized tensors are repacked on-the-fly to `Q8_1_R8`. 

Here some sweep-bench results for LLaMA-3.1-8B-Instruct on a Ryzen-7950X CPU

### Q4_K, main branch

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.853 |   179.49 |    9.792 |    13.07 |
|   512 |    128 |    512 |    2.745 |   186.52 |   10.119 |    12.65 |
|   512 |    128 |   1024 |    2.806 |   182.49 |   10.118 |    12.65 |
|   512 |    128 |   1536 |    2.905 |   176.22 |   10.273 |    12.46 |
|   512 |    128 |   2048 |    3.434 |   149.08 |   10.492 |    12.20 |

### Q4_K_R4

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.015 |   254.10 |    9.808 |    13.05 |
|   512 |    128 |    512 |    2.051 |   249.65 |    9.992 |    12.81 |
|   512 |    128 |   1024 |    2.131 |   240.28 |   10.145 |    12.62 |
|   512 |    128 |   1536 |    2.247 |   227.84 |   10.297 |    12.43 |
|   512 |    128 |   2048 |    2.338 |   219.02 |   10.478 |    12.22 |

### Q4_K, PR

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.903 |   269.00 |    9.719 |    13.17 |
|   512 |    128 |    512 |    1.974 |   259.37 |    9.975 |    12.83 |
|   512 |    128 |   1024 |    2.004 |   255.47 |   10.024 |    12.77 |
|   512 |    128 |   1536 |    2.351 |   217.73 |   10.033 |    12.76 |
|   512 |    128 |   2048 |    2.114 |   242.19 |   10.150 |    12.61 |

### Q5_K, main branch

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.894 |   176.89 |   11.650 |    10.99 |
|   512 |    128 |    512 |    3.461 |   147.93 |   11.760 |    10.88 |
|   512 |    128 |   1024 |    2.986 |   171.44 |   11.818 |    10.83 |
|   512 |    128 |   1536 |    3.026 |   169.22 |   11.875 |    10.78 |
|   512 |    128 |   2048 |    3.172 |   161.39 |   11.967 |    10.70 |

### Q5_K_R4

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.149 |   238.30 |   11.712 |    10.93 |
|   512 |    128 |    512 |    2.189 |   233.89 |   11.899 |    10.76 |
|   512 |    128 |   1024 |    2.269 |   225.62 |   11.953 |    10.71 |
|   512 |    128 |   1536 |    2.328 |   219.90 |   12.044 |    10.63 |
|   512 |    128 |   2048 |    2.343 |   218.54 |   12.050 |    10.62 |

### Q5_K, PR

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.929 |   265.41 |   11.599 |    11.04 |
|   512 |    128 |    512 |    2.042 |   250.69 |   11.810 |    10.84 |
|   512 |    128 |   1024 |    2.051 |   249.64 |   11.888 |    10.77 |
|   512 |    128 |   1536 |    2.350 |   217.91 |   11.888 |    10.77 |
|   512 |    128 |   2048 |    2.133 |   240.00 |   11.998 |    10.67 |

Here performance gains are not as large as in #514, #515, #516, #518 as k-quants are much faster than sub-4 bpw i-quants. Nevertheless, we see a nearly 50% PP performance improvement compared to the non-interleaved variants, and 5-10% improvement compared to the `_R4` variants.