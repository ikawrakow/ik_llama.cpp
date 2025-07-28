## 🔀 [Pull Request #48](https://github.com/ikawrakow/ik_llama.cpp/pull/48) - AVX2 Flash Attention

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/avx2_flash_attn` |
| **Target Branch** | `main` |
| **Created** | 2024-09-10 |
| **Updated** | 2024-09-10 |
| **Merged** | 2024-09-10 |

---

## 📄 Description

We don't gain as much as on a Zen4 system as there aren't as many vector registers, so we need to load/store data much more often. Still, we do get a small gain in performance.

For now it supports only `fp16` kv-cache. Quantized kv-cache will be added later.