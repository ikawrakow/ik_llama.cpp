## 🔀 [Pull Request #422](https://github.com/ikawrakow/ik_llama.cpp/pull/422) - Adding IQ5_KS - 5.25 bpw quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/iq5_ks` |
| **Target Branch** | `main` |
| **Created** | 2025-05-15 |
| **Updated** | 2025-05-18 |
| **Merged** | 2025-05-15 |

---

## 📄 Description

For motivation, see the CUDA performance graphs in [#417](https://github.com/ikawrakow/ik_llama.cpp/issues/417) and [#418](https://github.com/ikawrakow/ik_llama.cpp/issues/418).

Implementation for `AVX2, Zen4, ARM_NEON, CUDA, Metal`.

The `AVX2` implementation suffers from `int16_t` overflow, and so do the `IQ4_K, IQ5_K, IQ6_K` and `IQ4_KS`, so I will have to fix all of these in a follow up PR.

I also want to add interleaved variant `IQ5_KS_R4` before giving more performance and accuracy details.

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-05-18** at **21:18:35**

Just did some testing of a mixed `IQ5_KS` / `IQ4_KS` quant of Qwen3-14B dense showing some Perplexity and Speed comparisons for full CUDA offload in this [new quant cookers guide](https://github.com/ikawrakow/ik_llama.cpp/discussions/434) (just scroll to bottom, can't link anchors in gh discussions...)

Thanks for adding, the quality looks really good for the size!