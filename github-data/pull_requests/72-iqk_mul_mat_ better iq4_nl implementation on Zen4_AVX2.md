### 🔀 [#72](https://github.com/ikawrakow/ik_llama.cpp/pull/72) - iqk_mul_mat: better iq4_nl implementation on Zen4/AVX2

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-01 |
| **Updated** | 2024-10-01 |

---

#### Description

PP-512 performance for LLaMA-3.1-8B goes to 162.6 t/s up from 133.2 t/s (22% speedup).

This is mostly as preparation for investigating `IQ4_NL` usage for KV-cache, but still quite useful if someone is using it.