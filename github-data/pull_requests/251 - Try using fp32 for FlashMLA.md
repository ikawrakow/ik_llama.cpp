### [Pull Request #251](https://github.com/ikawrakow/ik_llama.cpp/pull/251) - Try using fp32 for FlashMLA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Source Branch** | `ik/flash_precision` |
| **Target Branch** | `main` |
| **Created** | 2025-03-10 |
| **Updated** | 2025-03-12 |

---

#### Description

_No description provided._

---

#### 🔀 Conversation

👤 **ikawrakow** commented on **2025-03-12** at **07:51:20**

Closing this as the numerical issues were caused by `fp16` experts matrix multiplications.