### 🔀 [#457](https://github.com/ikawrakow/ik_llama.cpp/pull/457) - Remove GGML_IQK_MUL_MAT option

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-05-25 |
| **Updated** | 2025-05-25 |

---

#### Description

There is no point in using `ik_llama.cpp` without `GGML_IQK_MUL_MAT`.

Closes #456

---

#### 💬 Conversation

👤 **Nexesenex** commented the **2025-05-25** at **12:34:51**:<br>

There is actually a point to leave this as a legacy marking for the quants, because it helps a lot with merging your quants, including the potential future ones, which are still compatible with only a few formatting adaptation with the mainline ggml framework, even if the ops are not.

I'm really good at shooting in my own foot! :D

---

👤 **ikawrakow** commented the **2025-05-25** at **15:10:55**:<br>

> as a legacy marking

Legacy marking in what sense?