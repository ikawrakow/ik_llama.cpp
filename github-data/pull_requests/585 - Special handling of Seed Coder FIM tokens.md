### [Pull Request #585](https://github.com/ikawrakow/ik_llama.cpp/pull/585) - Special handling of Seed Coder FIM tokens

| **Author** | `fizzAI` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `main` |
| **Target Branch** | `main` |
| **Created** | 2025-07-04 |
| **Updated** | 2025-07-06 |
| **Merged** | 2025-07-06 |

---

#### Description

Needed this for some quants and realized it didn't support it already, so figured I'd just PR upstream  
Seems a bit odd to need to figure out model families by vocab size? But I'm not sure of a better way to do it, so left it as-is for now

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 🔀 Conversation

👤 **fizzAI** commented on **2025-07-04** at **21:23:47**

Actually need to merge some tokenizer support from regular lcpp too, please hold lol

---

👤 **fizzAI** commented on **2025-07-04** at **22:43:32**

Appears to work, now