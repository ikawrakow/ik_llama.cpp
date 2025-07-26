### [Pull Request #654](https://github.com/ikawrakow/ik_llama.cpp/pull/654) - Fix text generation endpoint

| **Author** | `iSevenDays` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `patch-2` |
| **Target Branch** | `main` |
| **Created** | 2025-07-26 |
| **Updated** | 2025-07-26 |

---

#### Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High
  - [ ] 
The recent function call implementation changed streaming responses to always send empty content with diffs, which broke text completion streaming endpoints (like those used by mikupad) that need actual token content in each streaming chunk. This fix differentiates between OpenAI-compatible chat completion (which uses diffs) and text completion endpoints (which need actual content) using the existing slot.oaicompat flag.