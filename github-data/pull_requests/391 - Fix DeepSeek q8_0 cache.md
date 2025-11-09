## 🔀 [Pull Request #391](https://github.com/ikawrakow/ik_llama.cpp/pull/391) - Fix DeepSeek q8_0 cache

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_deepseek_q80_cache` |
| **Target Branch** | `main` |
| **Created** | 2025-05-07 |
| **Updated** | 2025-05-07 |
| **Merged** | 2025-05-07 |

---

## 📄 Description

Nobody has used `ik_llama.cpp` with a DeepSeek model and `Q8_0` KV cache since PR [#351](https://github.com/ikawrakow/ik_llama.cpp/issues/351)?

This PR fixes the assert one gets when one tries to use a DeepSeek model on the CPU using `Q8_0` KV cache.

Also, it seems the optimization I added in [#351](https://github.com/ikawrakow/ik_llama.cpp/issues/351) to repack the `K` cache to `Q8_0_R8` seems to lower TG performance for DeepSeek models, so disabling it.