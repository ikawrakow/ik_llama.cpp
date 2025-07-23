### 🔀 [#91](https://github.com/ikawrakow/ik_llama.cpp/pull/91) - CLI - Specify GGML_TYPE to quantize for the main tensors.

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-17 |
| **Updated** | 2024-10-18 |

---

#### Description

To complement the cli based custom quantization of token_embd.weight and output.weight, the ggml_type of the following tensors can now be specified :

attn_v.weight
attn_k.weight.
attn_q_weight
attn_output.weight
attn_qkv.weight
ffn_gate
ffn_down
ffn_up

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2024-10-17** at **06:32:51**: ✅ `APPROVED`<br>

This looks fine. I'm traveling today. Will do some testing and merge it tomorrow.