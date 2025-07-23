### 🔀 [#453](https://github.com/ikawrakow/ik_llama.cpp/pull/453) - Faster IQ3_KT and IQ4_KT

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-24 |
| **Updated** | 2025-05-24 |

---

#### Description

The PR improves `AVX2` performance for the trellis quants `IQ3_KT`  and `IQ4_KT` recently added in PR #441.
The results below are for LLaMA-3.1-8B on a Ryzen-5975WX CPU.

### IQ3_KT

|   N_KV | S_PP t/s (main) | S_PP t/s (PR) | PP speedup | S_TG t/s (main) | S_TG t/s (PR) | TG speedup |
|--------|----------|----------|------------|----------|----------|------------|
|      0 |    61.98 |    71.59 |  1.155     |    11.17 |    13.30 |  1.191     |
|    512 |    61.27 |    70.79 |  1.155     |    11.10 |    13.19 |  1.188     |   
|   1024 |    60.48 |    69.93 |  1.156     |    11.04 |    13.10 |  1.187     |   
|   1536 |    59.94 |    69.15 |  1.154     |    10.95 |    12.96 |  1.184     |   
|   2048 |    59.48 |    68.55 |  1.152     |    10.87 |    12.85 |  1.182     |   

### IQ4_KT

|   N_KV | S_PP t/s (main) | S_PP t/s (PR) | PP speedup | S_TG t/s (main) | S_TG t/s (PR) | TG speedup |
|--------|----------|----------|------------|----------|----------|------------|
|      0 |    44.32 |    64.91 |  1.465     |     9.36 |    11.69 |  1.249     |
|    512 |    43.90 |    64.12 |  1.461     |     9.26 |    11.56 |  1.248     |
|   1024 |    43.60 |    63.39 |  1.454     |     9.19 |    11.47 |  1.248     |
|   1536 |    43.32 |    62.86 |  1.451     |     9.12 |    11.37 |  1.247     |
|   2048 |    43.07 |    62.37 |  1.448     |     9.06 |    11.28 |  1.245     |

CPU performance is still much lower than other quantization types. But memory bandwidth is far from saturated, so PP and TG will be better on a faster CPU with more cores.