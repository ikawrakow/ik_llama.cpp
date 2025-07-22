### 🐛 [#448](https://github.com/ikawrakow/ik_llama.cpp/pull/448) - Fix MSVC compilation

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-23 |
| **Updated** | 2025-05-23 |

---

#### Description

MSVC does not like `^` with SIMD vectors.

Closes #447