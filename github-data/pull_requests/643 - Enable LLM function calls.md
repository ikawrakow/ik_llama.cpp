## 🔀 [Pull Request #643](https://github.com/ikawrakow/ik_llama.cpp/pull/643) - Enable LLM function calls

| **Author** | `iSevenDays` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `qwen3-function-calls` |
| **Target Branch** | `main` |
| **Created** | 2025-07-24 |
| **Updated** | 2025-07-24 |
| **Merged** | 2025-07-24 |

---

## 📄 Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

The PR fixes the logic when LLM responds with text and includes tool calls, but responds with "stop" instead of "tool_calls".

The PR enables LLM to work with Claude Code proxies in streaming mode.

---

## 💬 Conversation

👤 **ikawrakow** approved this pull request ✅ on **2025-07-24** at **18:24:00**