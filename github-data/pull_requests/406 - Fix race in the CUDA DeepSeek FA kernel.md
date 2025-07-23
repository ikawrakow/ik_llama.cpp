### 🐛 [#406](https://github.com/ikawrakow/ik_llama.cpp/pull/406) - Fix race in the CUDA DeepSeek FA kernel

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-11 |
| **Updated** | 2025-05-13 |

---

#### Description

Reference: https://github.com/ggml-org/llama.cpp/pull/13438

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-05-12** at **15:59:39**:<br>

Just saw what looks like a small patch in mainline's [earlier ggml-org/llama.cpp#13438 just updated in #13469 (linked here)](https://github.com/ggml-org/llama.cpp/pull/13469)

Could be related to my issue with `DDDD` showing up for longer contexts which I attributed to `-ser` [as we were discussing here](https://github.com/ikawrakow/ik_llama.cpp/pull/386#issuecomment-2869078136)?