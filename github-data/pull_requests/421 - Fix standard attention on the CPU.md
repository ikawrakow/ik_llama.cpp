## 🔀 [Pull Request #421](https://github.com/ikawrakow/ik_llama.cpp/pull/421) - Fix standard attention on the CPU

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_standard_attention_cpu` |
| **Target Branch** | `main` |
| **Created** | 2025-05-15 |
| **Updated** | 2025-05-15 |
| **Merged** | 2025-05-15 |

---

## 📄 Description

I have focusing on FA, MLA, FlashMLA lately, and at some point I have broken the standard self attention CPU implementation. This PR fixes it and closes [#420](https://github.com/ikawrakow/ik_llama.cpp/issues/420).