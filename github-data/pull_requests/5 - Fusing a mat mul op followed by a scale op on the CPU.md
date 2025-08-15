### 🔀 [#5](https://github.com/ikawrakow/ik_llama.cpp/pull/5) - Fusing a mat mul op followed by a scale op on the CPU

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-07-27 |
| **Updated** | 2025-02-08 |

---

#### Description

This is useful for Bitnet here we have almost all matrix multiplications be followed by scale operations.
As a result, we get a ~2% boost in Bitnet PP performance.

Implementation is easy when the matrix multiplication is done by `iqk_mul_mat`. But if `iqk_mul_mat` is not implemented for the quant type/architecture, we need to add the scaling to llamafile sgemm and to `ggml` itself, which is way more messy, so I didn't do it yet.

Given that Bitnet is just a niche thing for now, I'll just leave it on a draft PR for now.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-02-08** at **14:27:07**:<br>

I don't think I'll ever merge this.