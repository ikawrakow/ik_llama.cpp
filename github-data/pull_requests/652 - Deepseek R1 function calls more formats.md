## 🔀 [Pull Request #652](https://github.com/ikawrakow/ik_llama.cpp/pull/652) - Deepseek R1 function calls (more formats)

| **Author** | `iSevenDays` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `deepseek-r1-parsing` |
| **Target Branch** | `main` |
| **Created** | 2025-07-26 |
| **Updated** | 2025-07-26 |

---

## 📄 Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [x] Medium
  - [ ] High


Implemented more DeepSeek R1 supported function tool calls formats.
The diff of `examples/server/function_calls.md` shows what formats are supported. 

I was testing DeepSeek R1 and I found out that it often uses different formats with Claude Code, so I decided to support them as well. It can be useful when next version of DeepSeek is released, so we will have better support than even original llama.cpp