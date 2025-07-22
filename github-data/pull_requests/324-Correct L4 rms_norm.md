### 🔀 [#324](https://github.com/ikawrakow/ik_llama.cpp/pull/324) - Correct L4 rms_norm

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-11 |
| **Updated** | 2025-04-11 |

---

#### Description

I was wondering about the hard-coded `1e-6` when porting the mainline PR, but left it the way it is. Mainline has now [corrected it](https://github.com/ggml-org/llama.cpp/pull/12882), so let's do that here as well.