## 🔀 [Pull Request #515](https://github.com/ikawrakow/ik_llama.cpp/pull/515) - IQ2_XXS: much faster CPU prompt processing

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/iq2_xxs_gemm` |
| **Target Branch** | `main` |
| **Created** | 2025-06-11 |
| **Updated** | 2025-06-11 |
| **Merged** | 2025-06-11 |

---

## 📄 Description

While experimenting with the trellis quants in PRs [#505](https://github.com/ikawrakow/ik_llama.cpp/issues/505) and [#511](https://github.com/ikawrakow/ik_llama.cpp/issues/511), I realized that CPU matrix multiplications (GEMM) for quants that are slow to unpack and make ready for `int8_t` dot products (as the trellis quants are) are much faster if one unpacks a given number of rows to, e.g., `Q8_0_R8`, and then uses the `Q8_0_R8 x Q8_2_X4` GEMM to perform the multiplication with **all columns** of the right matrix.

This PR applies the approach of [#505](https://github.com/ikawrakow/ik_llama.cpp/issues/505)/[#511](https://github.com/ikawrakow/ik_llama.cpp/issues/511) to `IQ2_XXS` (`AVX2/Zen4`  only). We get nearly 3X improvement in PP performance compared to `IQ2_XXS` on the main branch, and 2X compared to `IQ2_XXS_R4`!

The same approach can be used out-of-the-box for `IQ3_XXS` (left for a follow up PR).

`IQ2_XS, IQ2_S` and `IQ3_S` use blocks of 16, so one would need a new row-interleaved 8-bit type with blocks of 16 for those.

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-06-11** at **07:59:26**

Here some sweep-bench tables on a Ryzen-7950X CPU for LlaMA-3.1-8B

### IQ2_XXS, main branch

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.793 |   106.82 |    5.452 |    23.48 |
|   512 |    128 |    512 |    4.984 |   102.73 |    6.375 |    20.08 |
|   512 |    128 |   1024 |    5.357 |    95.58 |    6.191 |    20.68 |
|   512 |    128 |   1536 |    5.062 |   101.15 |    6.290 |    20.35 |
|   512 |    128 |   2048 |    5.168 |    99.07 |    6.559 |    19.51 |

### IQ2_XXS_R4

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.467 |   147.69 |    5.508 |    23.24 |
|   512 |    128 |    512 |    3.764 |   136.03 |    5.964 |    21.46 |
|   512 |    128 |   1024 |    3.573 |   143.31 |    6.292 |    20.34 |
|   512 |    128 |   1536 |    3.660 |   139.88 |    6.341 |    20.19 |
|   512 |    128 |   2048 |    3.729 |   137.29 |    6.620 |    19.33 |

### IQ2_XXS, PR
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.778 |   288.03 |    5.484 |    23.34 |
|   512 |    128 |    512 |    1.860 |   275.28 |    5.685 |    22.52 |
|   512 |    128 |   1024 |    1.948 |   262.82 |    5.848 |    21.89 |
|   512 |    128 |   1536 |    2.040 |   250.93 |    6.158 |    20.78 |
|   512 |    128 |   2048 |    2.131 |   240.32 |    6.322 |    20.25 |