## 🔀 [Pull Request #577](https://github.com/ikawrakow/ik_llama.cpp/pull/577) - Vulkan: fused rms norm

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/vulkan_fused_rms` |
| **Target Branch** | `main` |
| **Created** | 2025-07-03 |
| **Updated** | 2025-07-03 |
| **Merged** | 2025-07-03 |

---

## 📄 Description

I see zero performance benefit, but at least we don't need to cpecial-case Vulkan when creating the graph.