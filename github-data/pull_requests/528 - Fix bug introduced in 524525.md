## 🔀 [Pull Request #528](https://github.com/ikawrakow/ik_llama.cpp/pull/528) - Fix bug introduced in [#524](https://github.com/ikawrakow/ik_llama.cpp/issues/524)/[#525](https://github.com/ikawrakow/ik_llama.cpp/issues/525)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_bug_481` |
| **Target Branch** | `main` |
| **Created** | 2025-06-14 |
| **Updated** | 2025-06-14 |
| **Merged** | 2025-06-14 |

---

## 📄 Description

When adding the faster GEMM in [#524](https://github.com/ikawrakow/ik_llama.cpp/issues/524) / [#525](https://github.com/ikawrakow/ik_llama.cpp/issues/525) I forgot to add the call to `iqk_convert_repack` also in the MoE matrix multiplication functions, which causes a crash (see [#527](https://github.com/ikawrakow/ik_llama.cpp/issues/527)). This PR fixes it.

---

## 💬 Conversation

👤 **ycat3** commented on **2025-06-14** at **10:30:08**

Thanks.
It works fine.
[#527](https://github.com/ikawrakow/ik_llama.cpp/issues/527)