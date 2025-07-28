## 🔀 [Pull Request #654](https://github.com/ikawrakow/ik_llama.cpp/pull/654) - Fix text generation endpoint

| **Author** | `iSevenDays` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `patch-2` |
| **Target Branch** | `main` |
| **Created** | 2025-07-26 |
| **Updated** | 2025-07-27 |
| **Merged** | 2025-07-27 |

---

## 📄 Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High
  - [ ] 
The recent function call implementation changed streaming responses to always send empty content with diffs, which broke text completion streaming endpoints (like those used by mikupad) that need actual token content in each streaming chunk. This fix differentiates between OpenAI-compatible chat completion (which uses diffs) and text completion endpoints (which need actual content) using the existing slot.oaicompat flag.

---

## 💬 Conversation

👤 **iSevenDays** commented on **2025-07-26** at **18:38:01**

The fix has been also verified by another person here https://github.com/ikawrakow/ik_llama.cpp/pull/628#issuecomment-3122219232

---

👤 **saood06** commented on **2025-07-27** at **00:15:39**

> The fix has been also verified by another person here [[#628](https://github.com/ikawrakow/ik_llama.cpp/issues/628) (comment)](https://github.com/ikawrakow/ik_llama.cpp/pull/628#issuecomment-3122219232)

And by me.

---

👤 **saood06** approved this pull request ✅ on **2025-07-27** at **00:17:02**

Tested.

This restored functionality to the `/completion` endpoint

---

👤 **saood06** commented on **2025-07-27** at **00:32:21**

@ikawrakow 

I've been very intentional in not pushing code into branches that are not mine (including main) without your approval as this is your repo, but I am making an exception in this case as this is a very minor change, that fixes a rather serious bug and you are out on vacation.