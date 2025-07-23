### 🔀 [#195](https://github.com/ikawrakow/ik_llama.cpp/pull/195) -  Deepseek MLA Optimizations V2

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-08 |
| **Updated** | 2025-02-09 |

---

#### Description

@ikawrakow 

This PR contains the following things
- A fairydreaming commit that is supposed to increase PP
- Avoid allocating the MHA KV cache in MLA mode
- Adds a change I originally missed that is used for gguf-py.

I will follow up with:
- Having all the MoE experts load during warmup, that can be placed in this PR if you want, or a separate one. It is a very large QoL feature for large MoE. Without it the model is slowly loaded in on use, with it, the model is loaded immediately and at a faster rate.
- The mmap based KV cache buffer, it is functional but I have yet to make it a CLI option.

---

#### 💬 Conversation

👤 **ikawrakow** submitted a review the **2025-02-09** at **07:36:43**: ✅ `APPROVED`<br>

Looks good. I added a minor change to check if `wk_b` and `wv_b` are available before turning on MLA (so we don't crash if someone is using an old model and asked for MLA).

PP-4096 for `Q8_0_R8` quantized DeepSeek-Lite with `-mla` goes up to 292 t/s from 275 t/s with this change.