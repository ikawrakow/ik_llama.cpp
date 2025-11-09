## 🔀 [Pull Request #604](https://github.com/ikawrakow/ik_llama.cpp/pull/604) - Fix attn_v conditionality when quantizing.

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `little_fix_attn_v` |
| **Target Branch** | `main` |
| **Created** | 2025-07-12 |
| **Updated** | 2025-07-13 |
| **Merged** | 2025-07-13 |

---

## 📄 Description

To retain compatibility with : https://github.com/ikawrakow/ik_llama.cpp/pull/91 We need "else if" and not "if", otherwise the MOE and 70b condition takes precedence over the specified quant in the CLI.

I can also expand this legacy custom quant to the IQ1 and IQ2 types quant strategies tree, and add the shexp tensor to it, if that's all right.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

## 💬 Conversation

👤 **ikawrakow** approved this pull request ✅ on **2025-07-13** at **09:24:27**

This is OK, but I think you should really start using `--custom-q`. That way you can make the mixes any way you like without relying on the logic in this function.

---

👤 **Nexesenex** commented on **2025-07-13** at **15:00:01**

Well, you're right.
I used your and ubergarm's recipes to make my first custom-q and it works for me too.
I'll switch on the custom-q method from now on.