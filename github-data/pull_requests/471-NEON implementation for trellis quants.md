### 🔀 [#471](https://github.com/ikawrakow/ik_llama.cpp/pull/471) - NEON implementation for trellis quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-29 |
| **Updated** | 2025-05-29 |

---

#### Description

Alternative to #460 

One wouldn't really want to use this on a NEON CPU as it is much too slow. But for the sake of completeness, here it is.

Sweep bench results for LLaMA-3.1-8B-Instruct **with BLAS** on M2-Max CPU (PP performance is much lower without BLAS)

### IQ2_KT

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    5.364 |    95.44 |   11.527 |    11.10 |
|   512 |    128 |    512 |    4.644 |   110.25 |   11.739 |    10.90 |
|   512 |    128 |   1024 |    4.870 |   105.14 |   12.270 |    10.43 |
|   512 |    128 |   1536 |    5.055 |   101.29 |   12.644 |    10.12 |
|   512 |    128 |   2048 |    5.289 |    96.81 |   12.732 |    10.05 |

### IQ3_KT

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    7.470 |    68.54 |   16.866 |     7.59 |
|   512 |    128 |    512 |    6.764 |    75.70 |   16.985 |     7.54 |
|   512 |    128 |   1024 |    6.987 |    73.28 |   17.157 |     7.46 |
|   512 |    128 |   1536 |    7.180 |    71.31 |   17.459 |     7.33 |
|   512 |    128 |   2048 |    7.401 |    69.18 |   17.453 |     7.33 |

### IQ4_KT

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    5.443 |    94.07 |   22.327 |     5.73 |
|   512 |    128 |    512 |    4.658 |   109.91 |   22.432 |     5.71 |
|   512 |    128 |   1024 |    4.889 |   104.73 |   22.937 |     5.58 |
|   512 |    128 |   1536 |    5.069 |   101.01 |   22.843 |     5.60 |
|   512 |    128 |   2048 |    5.295 |    96.70 |   22.816 |     5.61 |

This is nevertheless quite a bit faster than #460, so I'll go with this PR.

**Of note:** I couldn't make `IQ4_KT` work with `fp16` arithmetic for some reason. Not sure if there really is `fp16` range overflow, or if I just have a bug in the `fp16` implementation that I simply cannot see.