## 🔀 [Pull Request #442](https://github.com/ikawrakow/ik_llama.cpp/pull/442) - CUDA call tracer

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Source Branch** | `ik/cuda_tracer` |
| **Target Branch** | `main` |
| **Created** | 2025-05-21 |
| **Updated** | 2025-05-23 |

---

## 📄 Description

This PR adds a CUDA call tracer. The main purpose of the tracer is to hopefully help debug the illegal memory access crashes reported in [#398](https://github.com/ikawrakow/ik_llama.cpp/issues/398) and [#425](https://github.com/ikawrakow/ik_llama.cpp/issues/425). If there is a crash, the last 32 invocations of `CUDA_CHECK` will be printed to `stderr` before aborting. In my testing the overhead added by the tracer has negligible impact on performance.

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-05-23** at **15:26:16**

I can close this one now.