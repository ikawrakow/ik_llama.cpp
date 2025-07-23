### 🔀 [#369](https://github.com/ikawrakow/ik_llama.cpp/pull/369) - cmake: force MSVC compiler charset to utf-8

| **Author** | `Gaolingx` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-03 |
| **Updated** | 2025-05-03 |

---

#### Description

This commit is to prevent `tests\test-grammar-integration.cpp(483,13): error C2001: newline in constant` showing up in non-UTF8 windows system while using MSVC.

![image](https://github.com/user-attachments/assets/9d769ba8-94dc-4eef-943c-ad4b8a41793c)

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-05-03** at **12:26:22**: ✅ `APPROVED`<br>

LGTM, but I cannot test. It would be useful if at least one other person tested before we merge.

---

👤 **ikawrakow** submitted a review the **2025-05-03** at **12:26:22**: ✅ `APPROVED`<br>

LGTM, but I cannot test.

---

👤 **Gaolingx** commented the **2025-05-03** at **12:54:45**:<br>

> LGTM, but I cannot test. It would be useful if at least one other person tested before we merge.

At first, it couldn't be compiled while using MSVC, then I found the solution [https://github.com/ggml-org/llama.cpp/pull/9989](https://github.com/ggml-org/llama.cpp/pull/9989) , Well, it worked.