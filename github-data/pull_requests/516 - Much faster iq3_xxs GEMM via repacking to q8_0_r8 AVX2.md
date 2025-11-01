## 🔀 [Pull Request #516](https://github.com/ikawrakow/ik_llama.cpp/pull/516) - Much faster iq3_xxs GEMM via repacking to q8_0_r8 (AVX2)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/iq3_xxs_gemm` |
| **Target Branch** | `main` |
| **Created** | 2025-06-11 |
| **Updated** | 2025-06-11 |
| **Merged** | 2025-06-11 |

---

## 📄 Description

This PR is a follow up of [#515](https://github.com/ikawrakow/ik_llama.cpp/issues/515), and applies the same technique to `IQ3_XXS`. We see nearly 3X increase in prompt processing speed compared to `IQ3_XXS`, and over 2X compared to `IQ3_XXS_R4`.

Sweep-bench for pure `IQ3_XXS` quantization of LlaMA-3.1-8B on a Ryzen-7950X CPU:

### IQ3_XXS, main branch

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|  
|   512 |    128 |      0 |    5.023 |   101.94 |    7.365 |    17.38 |   
|   512 |    128 |    512 |    5.281 |    96.96 |    8.088 |    15.83 |   
|   512 |    128 |   1024 |    5.170 |    99.03 |    7.977 |    16.05 |   
|   512 |    128 |   1536 |    5.324 |    96.16 |    7.942 |    16.12 |   
|   512 |    128 |   2048 |    5.389 |    95.02 |    8.043 |    15.91 |

### IQ3_XXS_R4, main branch

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.836 |   133.47 |    7.675 |    16.68 |
|   512 |    128 |    512 |    3.687 |   138.87 |    8.279 |    15.46 |
|   512 |    128 |   1024 |    3.805 |   134.57 |    8.245 |    15.53 |
|   512 |    128 |   1536 |    3.906 |   131.08 |    8.252 |    15.51 |
|   512 |    128 |   2048 |    4.076 |   125.61 |    8.545 |    14.98 |

### IQ3_XXS, PR

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.730 |   296.01 |    7.641 |    16.75 |
|   512 |    128 |    512 |    1.807 |   283.30 |    8.333 |    15.36 |
|   512 |    128 |   1024 |    1.896 |   269.98 |    8.070 |    15.86 |
|   512 |    128 |   1536 |    1.978 |   258.78 |    8.481 |    15.09 |
|   512 |    128 |   2048 |    2.062 |   248.32 |    8.514 |    15.03 |