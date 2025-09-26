## 🔀 [Pull Request #174](https://github.com/ikawrakow/ik_llama.cpp/pull/174) - On Zen4 repack fp16 models to bf16_r16

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/zen4_repack_f16` |
| **Target Branch** | `main` |
| **Created** | 2025-01-21 |
| **Updated** | 2025-01-21 |
| **Merged** | 2025-01-21 |

---

## 📄 Description

...when run-time-repacking is requested via `-rtr`

This massively improves performance. As this is opt-in, we do not worry about possible precision loss in the `f16 -> bf16` conversion.