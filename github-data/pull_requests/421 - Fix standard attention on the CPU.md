### 🐛 [#421](https://github.com/ikawrakow/ik_llama.cpp/pull/421) - Fix standard attention on the CPU

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-15 |
| **Updated** | 2025-05-15 |

---

#### Description

I have focusing on FA, MLA, FlashMLA lately, and at some point I have broken the standard self attention CPU implementation. This PR fixes it and closes #420.