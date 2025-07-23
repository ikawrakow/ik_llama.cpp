### 🐛 [#87](https://github.com/ikawrakow/ik_llama.cpp/pull/87) - iq3_k: fix and optimize Metal dot product

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-14 |
| **Updated** | 2024-10-14 |

---

#### Description

I was accessing the scales as 4-byte aligned, but `IQ3_K` is not 4-byte aligned. Instead of throwing an error (as it happens
on CUDA when one makes a mistake such as this), Metal silently accepts and we get garbage. But we don't get garbage right away so one can easily notice, no we get garbage after some tokens have been generated.

PR also makes a minor optimization of the Metal dot product (~2.5% speedup).