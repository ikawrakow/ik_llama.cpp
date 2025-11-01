## 🔀 [Pull Request #58](https://github.com/ikawrakow/ik_llama.cpp/pull/58) - Fix compiler warnings

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_ggml_common` |
| **Target Branch** | `main` |
| **Created** | 2024-09-17 |
| **Updated** | 2024-09-17 |
| **Merged** | 2024-09-17 |

---

## 📄 Description

I got tired of the "ISO C++ forbids anonymous structures" warnings that are due to the way the quants scales are defined in `ggml-common.h`, so fixing it with this PR. 

Once at it
* Also added `-Wno-c99-extensions` when building on APPLE to avoid the gazillion warnings I'm getting due to `arm_neon.h`.
* Fixed the warnings in `iqk_quantize.cpp` and added `GGML_ABORT` when an implementation is missing.