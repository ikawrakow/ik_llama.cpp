## 🔀 [Pull Request #238](https://github.com/ikawrakow/ik_llama.cpp/pull/238) - A better way to measure the cost of ggml_barrier

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/measure_barriers` |
| **Target Branch** | `main` |
| **Created** | 2025-03-01 |
| **Updated** | 2025-03-01 |
| **Merged** | 2025-03-01 |

---

## 📄 Description

Trying to measure it on each `ggml_barrier` invocation is too imprecise as the best time resolution we have in `ggml` is 1 us. Hence, measure the total graph execution time and and the sum of the node execution times. The difference is then the cost of thread synchronization via `ggml_barrier`.

Using this on TG runs with DeepSeek-Lite I'm finding that `ggml_barrier` costs about 7% of the graph evaluation time when running on the CPU.

---

## 💬 Conversation

👤 **davidsyoung** commented on **2025-03-01** at **09:51:17**

@ikawrakow you are seriously cooking!

---

👤 **ikawrakow** commented on **2025-03-01** at **15:12:54**

> @ikawrakow you are seriously cooking!

I like cooking. Well, at least this kind of cooking. Real cooking I tend to avoid by going to restaurants.