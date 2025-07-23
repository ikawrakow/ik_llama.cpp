### 🔀 [#465](https://github.com/ikawrakow/ik_llama.cpp/pull/465) - Set cache_prompt default to true

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-28 |
| **Updated** | 2025-05-28 |

---

#### Description

There is very little reason to not enable cache_prompt, so it makes more sense for it to be enabled since it benefits those who either don't know about this or use tools that do not set this, and the option to turn it off is still allowed in the very niche situations where this behavior is not desired.

Closes #455

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-05-28** at **05:18:19**: ✅ `APPROVED`