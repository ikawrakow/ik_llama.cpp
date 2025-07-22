### 🐛 [#349](https://github.com/ikawrakow/ik_llama.cpp/pull/349) - Fix division by zero bug

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-26 |
| **Updated** | 2025-04-26 |

---

#### Description

The bug was in the calculation of number of work items to use when computing FA on the CPU. In my case (maximum of 32 threads) it triggered with the GLM-4 model that has an unusually small number of KV heads (just 2). But I guess it can also trigger with a larger number of threads for more common numbers of KV heads.

Fixed by just using `max(1, nk)`. This will result in a far from optimal number of compute chunks, but at least it works.

I'm working on a better strategy for dividing the work between the threads on [this branch](https://github.com/ikawrakow/ik_llama.cpp/tree/ik/fattn_work_buffer),  but not quite ready for a PR yet.