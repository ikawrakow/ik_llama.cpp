## 🔀 [Pull Request #431](https://github.com/ikawrakow/ik_llama.cpp/pull/431) - Forgotten MMQ ref and typo

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `fix_mmq` |
| **Target Branch** | `main` |
| **Created** | 2025-05-18 |
| **Updated** | 2025-05-22 |
| **Merged** | 2025-05-18 |

---

## 📄 Description

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

## 💬 Conversation

👤 **ikawrakow** approved this pull request ✅ on **2025-05-18** at **14:36:30**

Hey, you are back!

---

👤 **Nexesenex** commented on **2025-05-18** at **14:48:44**

Hey!
Yeah, you sounded the horn with those MMQ Kernels for the IQ_K quants, I waited for them for a long time. I merged in Croco your IQ quants (included the KS ones with success last year, before the rev 14 of the GGUF format broke compatibility with them, possibly due to the template change introduced in https://github.com/ikawrakow/ik_llama.cpp/pull/45 )
Meanwhile, I was amusing myself merging models, among other nerdy delights.
Congrats for all the amazing developments you made, even if it's hard for me to swing between mainline and IK_Llama to feed my Croco.
Also, Turboderp switched on QTIP based quants for Exllamav3.
Things are getting exciting!