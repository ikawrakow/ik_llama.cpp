## 🔀 [Pull Request #517](https://github.com/ikawrakow/ik_llama.cpp/pull/517) - IQ1_S: much faster CPU prompt processing

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/iq1_s_gemm` |
| **Target Branch** | `main` |
| **Created** | 2025-06-11 |
| **Updated** | 2025-06-11 |
| **Merged** | 2025-06-11 |

---

## 📄 Description

This PR is a follow up of [#515](https://github.com/ikawrakow/ik_llama.cpp/issues/515) and [#516](https://github.com/ikawrakow/ik_llama.cpp/issues/516), and applies the same technique to `IQ1_S`. We see nearly 2X increase in prompt processing speed compared to `IQ1_S` and `IQ1_S_R4.

Sweep-bench for `IQ1_S` quantization of LlaMA-3.1-8B on a Ryzen-7950X CPU:

### IQ1_S, main branch

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.272 |   156.47 |    4.605 |    27.79 |
|   512 |    128 |    512 |    3.351 |   152.77 |    5.092 |    25.14 |
|   512 |    128 |   1024 |    3.402 |   150.52 |    5.084 |    25.18 |
|   512 |    128 |   1536 |    3.677 |   139.25 |    5.201 |    24.61 |
|   512 |    128 |   2048 |    3.586 |   142.79 |    5.515 |    23.21 |

### IQ1_S_R4, main branch

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.101 |   165.10 |    4.543 |    28.18 |
|   512 |    128 |    512 |    3.166 |   161.74 |    4.836 |    26.47 |
|   512 |    128 |   1024 |    3.309 |   154.75 |    5.282 |    24.23 |
|   512 |    128 |   1536 |    3.348 |   152.92 |    5.093 |    25.13 |
|   512 |    128 |   2048 |    3.447 |   148.55 |    5.265 |    24.31 |


### IQ1_S, PR

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.855 |   275.94 |    4.643 |    27.57 |
|   512 |    128 |    512 |    1.940 |   263.87 |    5.056 |    25.32 |
|   512 |    128 |   1024 |    2.188 |   234.05 |    5.099 |    25.10 |
|   512 |    128 |   1536 |    2.097 |   244.20 |    5.112 |    25.04 |
|   512 |    128 |   2048 |    2.184 |   234.42 |    5.368 |    23.85 |