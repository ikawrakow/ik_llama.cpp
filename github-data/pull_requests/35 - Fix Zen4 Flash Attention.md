## 🔀 [Pull Request #35](https://github.com/ikawrakow/ik_llama.cpp/pull/35) - Fix Zen4 Flash Attention

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_flash_attn` |
| **Target Branch** | `main` |
| **Created** | 2024-09-02 |
| **Updated** | 2024-09-02 |
| **Merged** | 2024-09-02 |

---

## 📄 Description

Closes [#34](https://github.com/ikawrakow/ik_llama.cpp/issues/34) 

Funny enough, the bug was not in the FA implementation but in the way I was calling `iqk_flash_attn_noalibi` from `ggml`.