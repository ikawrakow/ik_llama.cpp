### 🐛 [#402](https://github.com/ikawrakow/ik_llama.cpp/pull/402) - Fix missing rope_freqs with convert_hf_to_gguf

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-09 |
| **Updated** | 2025-05-09 |

---

#### Description

This ports https://github.com/ggml-org/llama.cpp/pull/9396 and https://github.com/ggml-org/llama.cpp/pull/9117 (I don't think I needed this as the changes in here are basically reverted in 9396).

The issue was that the convert script used generate_extra_tensors for those tensors but there was no code that called that function.

I tested with [Llama-3_1-Nemotron-51B-Instruct](https://huggingface.co/nvidia/Llama-3_1-Nemotron-51B-Instruct) and it now generates the rope_freqs.weight which was missing previously.

Look at #377 for more information.

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-05-09** at **14:16:12**: ✅ `APPROVED`