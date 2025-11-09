## 🔀 [Pull Request #569](https://github.com/ikawrakow/ik_llama.cpp/pull/569) - Conditionally disable fused ops when building with Vulkan enabled

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/vulkan_disable_fused_ops` |
| **Target Branch** | `main` |
| **Created** | 2025-07-02 |
| **Updated** | 2025-07-02 |
| **Merged** | 2025-07-02 |

---

## 📄 Description

Last PR just disabled them, here we disable them only if building with Vulkan support.

This is temporary until the fused ops are implemented in the Vulkan backend.