## 🔀 [Pull Request #570](https://github.com/ikawrakow/ik_llama.cpp/pull/570) - Remove duplicate/misplaced cmake find_package for Vulkan

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Source Branch** | `fix_novulkan_cmake` |
| **Target Branch** | `main` |
| **Created** | 2025-07-02 |
| **Updated** | 2025-07-02 |

---

## 📄 Description

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

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-07-02** at **14:13:30**

I just merged [#571](https://github.com/ikawrakow/ik_llama.cpp/issues/571) that should fix it. Thanks for reporting and making a PR.

I preferred [#571](https://github.com/ikawrakow/ik_llama.cpp/issues/571) because also the function testing Vulkan features needed to go inside the Vulkan block.

---

👤 **Nexesenex** commented on **2025-07-02** at **14:16:27**

Of course, np!

I wondered about reformating the Vulkan block in the cmakelist also, but you did it all already!