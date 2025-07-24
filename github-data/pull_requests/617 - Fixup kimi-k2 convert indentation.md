### 🐛 [#617](https://github.com/ikawrakow/ik_llama.cpp/pull/617) - Fixup kimi-k2 convert indentation

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-16 |
| **Updated** | 2025-07-16 |

---

#### Description

Fixup a copy-paste python indent bug on the convert_hf_to_gguf.py script for kimi-k2-instruct. Thanks @anikifoss for testing and if you have success let me know here to confirm this patch is good.

https://github.com/ikawrakow/ik_llama.cpp/pull/612#issuecomment-3076684820

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-07-16** at **13:24:15**: ✅ `APPROVED`

---

👤 **ubergarm** commented the **2025-07-16** at **13:30:08**:<br>

> Still running, 8 hours later at 50%. There is `attn_kv_b` in the output GGUF.
> 
> Why do you need `attn_kv_b` anyway?

@anikifoss 

Thanks for running this long job and testing!

Check here for some more info: https://github.com/ikawrakow/ik_llama.cpp/issues/601#issuecomment-3070185792

Based on that discussion I've changed my recipes a bit for Kimi and future deepseek models.

---

👤 **ikawrakow** commented the **2025-07-16** at **14:43:46**:<br>

> I hope this is somewhat accurate

It is. Basically, you don't need to have the `attn_kv_b` tensors to create imatrix data and a good quantized model for `ik_llama.cpp`. The only potential benefit from having `attn_kv_b` in the GGUF is that then these tensors becomes part of the contiguously allocated (or mmap'ed) tensor data storage, while if they are not present in the GGUF, memory is allocated separately for them (but still on the same device that stores the corresponding `attn_k` and `attn_v` tensors). Considering how sensitive the big NUMA systems are to the way the tensors are stored in RAM, this may have some performance implications. But nobody has studied this effect in detail yet, so we don't really know.