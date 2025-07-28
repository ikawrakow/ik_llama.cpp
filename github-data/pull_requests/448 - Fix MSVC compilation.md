## 🔀 [Pull Request #448](https://github.com/ikawrakow/ik_llama.cpp/pull/448) - Fix MSVC compilation

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_447` |
| **Target Branch** | `main` |
| **Created** | 2025-05-23 |
| **Updated** | 2025-05-23 |
| **Merged** | 2025-05-23 |

---

## 📄 Description

MSVC does not like `^` with SIMD vectors.

Closes [#447](https://github.com/ikawrakow/ik_llama.cpp/issues/447)