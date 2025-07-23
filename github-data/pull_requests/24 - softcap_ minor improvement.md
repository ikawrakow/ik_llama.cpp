### 🔀 [#24](https://github.com/ikawrakow/ik_llama.cpp/pull/24) - softcap: minor improvement

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-08-21 |
| **Updated** | 2024-08-21 |

---

#### Description

With this change we get 104 t/s for Gemma-2-9b with a context of 8192 tokens on a Ryzen-7950X.

For this model and context size, about 10% of the time is spent in `softcap` (5.8%) and `soft_max` (4.2%) when running on the Ryzen-7950X CPU. I wonder if it wouldn't be better to merge `softcap` and `soft_max` into a single op (for Gemma-2, `softcap` in the attention layer is immediately followed by `soft_max`)