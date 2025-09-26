## 🔀 [Pull Request #98](https://github.com/ikawrakow/ik_llama.cpp/pull/98) - Avoid rebuild of GGML graph for each token

| **Author** | `agray3` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ag_avoid_ggml_graph_rebuild` |
| **Target Branch** | `main` |
| **Created** | 2024-10-19 |
| **Updated** | 2024-10-20 |
| **Merged** | 2024-10-20 |

---

## 📄 Description

Introduces caching of GGML graph to avoid unnecessary full rebuild between each token. KV cache parameters, which change with each token, are updated directly in cached GGML graph. Can be disabled with GGML_DISABLE_GRAPH_CACHING environment variable.



- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [ ] Medium
  - [ ] High

---

## 💬 Conversation

👤 **agray3** commented on **2024-10-19** at **19:19:21**

See https://github.com/ikawrakow/ik_llama.cpp/pull/94

---

👤 **ikawrakow** approved this pull request ✅ on **2024-10-20** at **06:35:58**