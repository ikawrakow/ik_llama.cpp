### 🗣️ [#564](https://github.com/ikawrakow/ik_llama.cpp/discussions/564) - Maybe an interesting CUDA PR here.

| **Author** | `Nexesenex` |
| :--- | :--- |
| **Created** | 2025-06-29 |
| **Updated** | 2025-07-01 |

---

#### Description

Title : Overlap CUDA graph building and processing to minimize GPU idle time and improve tokens per seconds performance.
#11867
Link : https://github.com/ggml-org/llama.cpp/pull/11867
Author : @Aendk
Use : a few % boost on Cuda PP and TG?

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-07-01** at **13:56:23**:<br>

Yes, I saw this PR. But to quote Diego's statement in the PR discussion

> I still think that this change adds a significant amount of complexity, to code that is already too fragile and complex to reasonably maintain. 

I fully agree with that. The back-end is really fragile, so performance gains must be way more than 2-3% to warrant a change such as that one.