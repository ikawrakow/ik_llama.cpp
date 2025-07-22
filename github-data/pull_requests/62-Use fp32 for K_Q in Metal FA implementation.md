### 🔀 [#62](https://github.com/ikawrakow/ik_llama.cpp/pull/62) - Use fp32 for K*Q in Metal FA implementation

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-25 |
| **Updated** | 2024-09-25 |

---

#### Description

Else some models (e.g., Qwen2-7B-Instruct) produce garbage. Borrowed from PR-9595 in mainline `llama.cpp`.

Strangely enough, `K*Q` is done using `fp16` in my `ARM_NEON` FA implementation, and it works just fine there.