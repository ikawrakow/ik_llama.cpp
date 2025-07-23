### 🔀 [#320](https://github.com/ikawrakow/ik_llama.cpp/pull/320) - Guard against attempts to use MLA for non-MLA models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-08 |
| **Updated** | 2025-04-08 |

---

#### Description

So we don't crash when someone uses `-mla` with non-MLA models.