## 🔀 [Pull Request #69](https://github.com/ikawrakow/ik_llama.cpp/pull/69) - Allow bf16 kv-cache

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/bf16_kv_cache` |
| **Target Branch** | `main` |
| **Created** | 2024-09-29 |
| **Updated** | 2024-09-29 |
| **Merged** | 2024-09-29 |

---

## 📄 Description

On the CPU I get the exact same PPL with and without FA using `bf16` for kv-cache. But on CUDA the `bf16` kv-cache result is about the same as the `fp16` kv-cache CPU result, so I'm missing some conversion somewhere. Either way, we can now run on all platforms supported here with `bf16` kv-cache.