## 🔀 [Pull Request #457](https://github.com/ikawrakow/ik_llama.cpp/pull/457) - Remove GGML_IQK_MUL_MAT option

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `ik/remove_iqk_option` |
| **Target Branch** | `main` |
| **Created** | 2025-05-25 |
| **Updated** | 2025-05-25 |

---

## 📄 Description

There is no point in using `ik_llama.cpp` without `GGML_IQK_MUL_MAT`.

Closes [#456](https://github.com/ikawrakow/ik_llama.cpp/issues/456)

---

## 💬 Conversation

👤 **Nexesenex** commented on **2025-05-25** at **12:34:51**

There is actually a point to leave this as a legacy marking for your new quants, because it helps a lot with merging them on my fork, including the potential future ones, which are still compatible with only a few formatting adaptation with the mainline ggml framework, even if the ops are not.

I'm really good at shooting in my own foot! :D

---

👤 **ikawrakow** commented on **2025-05-25** at **15:10:55**

> as a legacy marking

Legacy marking in what sense?

---

👤 **Nexesenex** commented on **2025-05-25** at **18:31:43**

In the sense that even if the option is not used anymore to compile, it's pretty handy for your average enthusiast such as myself to still have the distinction in the code between IKL multmat dependent code and the rest of your  code to help the merging with a mainline fork, at least on anything with a compatible part.

For now, everything is still quite clear, but as time will pass, the divergence between IKL and mainline will increase, and having at least the point of reference of what works (theoretically) and what doesn't with mainline of August 2024 is an invaluable help.