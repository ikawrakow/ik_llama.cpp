### 🐛 [#604](https://github.com/ikawrakow/ik_llama.cpp/pull/604) - Fix attn_v conditionality when quantizing.

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-12 |
| **Updated** | 2025-07-13 |

---

#### Description

To retain compatibility with : https://github.com/ikawrakow/ik_llama.cpp/pull/91 We need "else if" and not "if", otherwise the MOE and 70b condition takes precedence over the specified quant in the CLI.

I can also expand this legacy custom quant to the IQ1 and IQ2 types quant strategies tree, and add the shexp tensor to it, if that's all right.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-07-13** at **09:24:27**: ✅ `APPROVED`<br>

This is OK, but I think you should really start using `--custom-q`. That way you can make the mixes any way you like without relying on the logic in this function.