## 🔀 [Pull Request #582](https://github.com/ikawrakow/ik_llama.cpp/pull/582) - Vulkan: adding GGML_OP_MULTI_ADD implementation

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/vulkan_multi_add` |
| **Target Branch** | `main` |
| **Created** | 2025-07-04 |
| **Updated** | 2025-07-04 |
| **Merged** | 2025-07-04 |

---

## 📄 Description

This is relevant for MoE models. The performance improvement is surprisingly small. Somewhere it was mentioned that Vulkan kernel launch overhead is significantly larger than CUDA, so I would have expected a more significant performance benefit. For DeepSeek-Lite, the number of graph nodes in `ik_llama.cpp` with this PR is 1420 vs 1871 in mainline `llama.cpp`.

But, if nothing else, this removes  the last Vulkan special-casing when building the compute graph.