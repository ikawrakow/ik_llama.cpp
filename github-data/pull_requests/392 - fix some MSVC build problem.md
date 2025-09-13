## 🔀 [Pull Request #392](https://github.com/ikawrakow/ik_llama.cpp/pull/392) - fix some MSVC build problem.

| **Author** | `Gaolingx` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `main` |
| **Target Branch** | `main` |
| **Created** | 2025-05-07 |
| **Updated** | 2025-05-07 |
| **Merged** | 2025-05-07 |

---

## 📄 Description

fix some MSVC build problem.
From PR :
1. [Commit 4dd34ff](https://github.com/ggml-org/llama.cpp/commit/4dd34ff83165a483ebff7bd43621b28490fa1fd6)
2. [Commit f35726c](https://github.com/ggml-org/llama.cpp/commit/f35726c2fb0a824246e004ab4bedcde37f3f0dd0)

Build Result:
![1db9f898-c116-4268-b545-14211f895cf9](https://github.com/user-attachments/assets/1ce36d3f-abc9-4c69-80fb-81d178f56614)

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

## 💬 Conversation

👤 **ikawrakow** started a conversation on `CMakeLists.txt` on **2025-05-07** at **12:31:01**

Why are you deleting these? As a `vim` user they are essential for my CUDA editing experience.

> 👤 **Gaolingx** replied on **2025-05-07** at **12:37:49**
> 
> sorry, I don't know what happened deleting these, I forget revert these after build.

---

👤 **ikawrakow** approved this pull request ✅ on **2025-05-07** at **12:47:42**