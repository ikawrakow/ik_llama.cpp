### ✨ [#614](https://github.com/ikawrakow/ik_llama.cpp/issues/614) - Feature Request: port no-mmproj-offload

| **Author** | `erazortt` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-15 |
| **Updated** | 2025-07-16 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Please port over the flag no-mmproj-offload.

### Motivation

This helps saving VRAM and since I use the vision model quite seldom, I can wait a little longer when I do use it.

### Possible Implementation

_No response_

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-07-16** at **09:19:35**:<br>

There is no vision support at all in `ik_llama.cpp`, see my response in #615