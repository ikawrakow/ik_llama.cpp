### 🔀 [#64](https://github.com/ikawrakow/ik_llama.cpp/pull/64) - Better sub-3-bit quantization mixes with a qkv tensor

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-28 |
| **Updated** | 2024-09-28 |

---

#### Description

Phi3.5-mini uses a combined `QKV` tensor. As a result, the quantization mix strategies used for sub-3-bit quants fail. This PR fixes it, and here is what we get as quantization error using wiki text perplexity

![iphi3 5_ppl](https://github.com/user-attachments/assets/8b9f08d2-e79c-447c-b9d0-929377f254d0)