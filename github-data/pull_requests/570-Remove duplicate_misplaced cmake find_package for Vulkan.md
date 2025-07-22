### 🔀 [#570](https://github.com/ikawrakow/ik_llama.cpp/pull/570) - Remove duplicate/misplaced cmake find_package for Vulkan

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-02 |
| **Updated** | 2025-07-02 |

---

#### Description

This line `find_package(Vulkan COMPONENTS glslc REQUIRED)` prevented to build anything on MSVS 2022 if the package was not present on the system, this even if Vulkan was not selected.

It's already present in the Vulkan conditionality.

```
if (GGML_VULKAN)
find_package(Vulkan COMPONENTS glslc REQUIRED)
```

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High