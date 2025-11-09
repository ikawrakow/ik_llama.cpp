## 🔀 [Pull Request #42](https://github.com/ikawrakow/ik_llama.cpp/pull/42) - Adding fused rms_norm

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fused_rms_norm` |
| **Target Branch** | `main` |
| **Created** | 2024-09-08 |
| **Updated** | 2024-09-08 |
| **Merged** | 2024-09-08 |

---

## 📄 Description

Many models have one or more of `rms_norm` followed by multiplication with a normalization tensor that is (almost) always just a single row. Fusing these two operations into a single op reduces thread synchronization cost and thus has the potential to improve performance, especially for relatively small models.

This PR adds this fused operation with implementations for the CPU, CUDA and Metal. We get about 1% speedup for PP and TG for Gemma2-2b on all implemented platforms. If we look at a tiny model such as the 99M parameter ternary TriLM, performance improvement is in the range of 5-7%.