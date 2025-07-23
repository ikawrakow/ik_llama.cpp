### 🔀 [#470](https://github.com/ikawrakow/ik_llama.cpp/pull/470) - Send [DONE] for OAI compatibility

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-29 |
| **Updated** | 2025-06-17 |

---

#### Description

See #467

The PR adds a command line parameter `--send-done`, which makes the server send a `data: [DONE]\n\n` message when a stop token is encountered.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-06-17** at **07:33:28**:<br>

Closes #467