### 🔀 [#343](https://github.com/ikawrakow/ik_llama.cpp/pull/343) - cuda: use switch in constexpr funcs

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-24 |
| **Updated** | 2025-04-24 |

---

#### Description

Based on [PR 13095](https://github.com/ggml-org/llama.cpp/pull/13095) in mainline. Did not measure, but had the impression that CUDA compile time is reduced.