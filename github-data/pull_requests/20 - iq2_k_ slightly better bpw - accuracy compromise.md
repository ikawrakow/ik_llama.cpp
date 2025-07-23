### 🔀 [#20](https://github.com/ikawrakow/ik_llama.cpp/pull/20) - iq2_k: slightly better bpw - accuracy compromise

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-08-19 |
| **Updated** | 2024-08-19 |

---

#### Description

For LLaMA-3.1 models:
* It is better to quantize all of attn_v with iq3_k instead of half of attn_v with iq4_k
* Quantizing attn_output with iq3_k results in a larger PPL decrease compared to what one expects from the added bpw.