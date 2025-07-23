### 🔀 [#99](https://github.com/ikawrakow/ik_llama.cpp/pull/99) - Enable IQ4_NL for KV-cache in token generation using Flash Attention 

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-20 |
| **Updated** | 2024-10-21 |

---

#### Description

Only added for head size = 128 for now, we can add other head sizes if needed.

For me `-ctk q8_0 -ctv iq4_nl` is the most useful combination in terms of the compromise between generation quality and KV-cache size.

**Update**

Based on @Nexesenex comment in #92, added `IQ4_NL + IQ4_NL` as a possible KV-cache combination for head size of 128. Hopefully this is a better alternative than `Q4_0 + Q4_0` for the VRAM poor.

---

#### 💬 Conversation

👤 **saood06** commented the **2024-10-20** at **18:48:37**:<br>

Since you're enabling q8_0/iq4_nl by default you should update the on_no_fattn_vec_case function in fattn-common.cuh to mention it.

---

👤 **ikawrakow** commented the **2024-10-21** at **08:10:33**:<br>

> Since you're enabling q8_0/iq4_nl by default you should update the on_no_fattn_vec_case function in fattn-common.cuh to mention it.

Thanks for pointing out. It is now updated to reflect the possible quantized cache combinations.

---

👤 **Nexesenex** commented the **2024-10-21** at **09:47:46**:<br>

It works. In the name of the VRAM poor that I do so well represent, thanks! xD