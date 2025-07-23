### 🔀 [#585](https://github.com/ikawrakow/ik_llama.cpp/pull/585) - Special handling of Seed Coder FIM tokens

| **Author** | `fizzAI` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-04 |
| **Updated** | 2025-07-06 |

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

#### 💬 Conversation

👤 **fizzAI** commented the **2025-07-04** at **21:23:47**:<br>

Actually need to merge some tokenizer support from regular lcpp too, please hold lol

---

👤 **fizzAI** commented the **2025-07-04** at **22:43:32**:<br>

Appears to work, now

---

👤 **ikawrakow** submitted a review the **2025-07-05** at **09:29:56**: 💬 `COMMENTED`

---

👤 **ikawrakow** commented during a code review the **2025-07-05** at **09:29:56** on `convert_hf_to_gguf.py`:<br>

It is the only model that has a vocabulary of 155,136 tokens?

---

👤 **ikawrakow** commented during a code review the **2025-07-05** at **09:30:24** on `include/llama.h`:<br>

Pleas format the same way as the surrounding code.

---

👤 **ikawrakow** commented during a code review the **2025-07-05** at **09:30:33** on `src/llama.cpp`:<br>

Pleas format the same way as the surrounding code.

---

👤 **ikawrakow** submitted a review the **2025-07-05** at **09:30:54**: ✅ `APPROVED`

---

👤 **fizzAI** submitted a review the **2025-07-05** at **19:35:38**: 💬 `COMMENTED`

---

👤 **fizzAI** submitted a review the **2025-07-05** at **19:35:56**: 💬 `COMMENTED`

---

👤 **fizzAI** commented during a code review the **2025-07-05** at **19:35:56** on `include/llama.h`:<br>

D: damn my editor