### 🔀 [#163](https://github.com/ikawrakow/ik_llama.cpp/pull/163) - q4_0_r4: Use AVX2 version for matrix x vector

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-23 |
| **Updated** | 2024-12-23 |

---

#### Description

Performance is better. Packing quants into 512-bit registers is costly and when we have just 1 column to multiply, using the `AVX512` version becomes slower. I had already done this for most (all?) other quants, but somehow missed `Q4_0`.