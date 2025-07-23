### 🔀 [#144](https://github.com/ikawrakow/ik_llama.cpp/pull/144) - Slightly faster IQ4_K_R4 on AVX2/Zen4

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-16 |
| **Updated** | 2024-12-16 |

---

#### Description

We get PP-512(LLaMA-3.1-8B) = 251 t/s (Ryzen-7950X) or 249 t/s (Ryzen-5975WX), up from 232/227 t/s.