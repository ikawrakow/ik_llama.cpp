### 🐛 [#275](https://github.com/ikawrakow/ik_llama.cpp/pull/275) - Fix bug: missing parentheses in logical expression

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-21 |
| **Updated** | 2025-03-21 |

---

#### Description

This results in GGGGGGGGGGGGG when generating with mla = 3, fa = 0.

Likely also affects @saood06's benchmarking [here](https://github.com/ikawrakow/ik_llama.cpp/pull/273#issuecomment-2743023599)