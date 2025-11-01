## 🔀 [Pull Request #648](https://github.com/ikawrakow/ik_llama.cpp/pull/648) - Fix missing token per second for webui after function call update

| **Author** | `firecoperana` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `fcp/missing_token_ps` |
| **Target Branch** | `main` |
| **Created** | 2025-07-25 |
| **Updated** | 2025-07-27 |
| **Merged** | 2025-07-27 |
| **Assignees** | `firecoperana` |

---

## 📄 Description

1. Moves Preset to top of settings window for easier navigation
2. Send timings in streaming_chunks otherwise webui won't show token per second

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

## 💬 Conversation

👤 **saood06** approved this pull request ✅ on **2025-07-27** at **01:51:35**

LGTM.

Tested and see the t/s measurements which was missing when testing main.

---

👤 **saood06** commented on **2025-07-27** at **02:31:21**

@firecoperana 

Just to let you know the [CONTRIBUTING.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/CONTRIBUTING.md) says to "Squash-merge PRs", you created a merge commit, which is different. I know that document was created from mainline and hasn't been touched here, but it does seem that squash merge commits are the norm here as well.

---

👤 **firecoperana** commented on **2025-07-27** at **03:43:59**

Thanks for the heads up. Will do that next time.