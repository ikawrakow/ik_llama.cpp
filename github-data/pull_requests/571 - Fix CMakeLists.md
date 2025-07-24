### 🐛 [#571](https://github.com/ikawrakow/ik_llama.cpp/pull/571) - Fix CMakeLists

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-02 |
| **Updated** | 2025-07-02 |

---

#### Description

The Vulkan stuff had ended up outside the `if (GGML_VULKAN)` condition, which prevents building any configuration unless having Vulkan installed.

This PR fixes it.

Oh, it shows as 130 lines changed because I retabed (don't like having tabs in source code). The change is much smaller in reality.