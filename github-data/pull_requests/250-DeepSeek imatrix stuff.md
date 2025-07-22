### 🔀 [#250](https://github.com/ikawrakow/ik_llama.cpp/pull/250) - DeepSeek imatrix stuff

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-10 |
| **Updated** | 2025-03-10 |

---

#### Description

In DeepSeek models there are two additional tensors, `*attn_k_b.weight` and `*attn_v_b.weight` required for MLA. When MLA is enabled, these will get used for attention computation. When standard attention is used, then the `*attn_kv_b.weight` tensors are used instead. Hence, when one has used standard attention to compute the imatrix, there will be no data for `*attn_k_b.weight` and `*attn_v_b.weight`; if one uses MLA, then there will be no data for `*attn_kv_b.weight`. As the `*attn_v_b.weight` tensors are simply the lower half of `*attn_kv_b.weight` (i.e., the second half of rows), they "see" the exact same activations as the `*attn_kv_b.weight` tensors. This PR takes advantage of this and enables the usage of `*attn_kv_b.weight` imatrix data for `*attn_v_b.weight` and vice versa.

The situation with `*attn_k_b.weight`  is more tricky and will require a much bigger change to be fixed. `*attn_k_b.weight`  is the transposed upper half of `*attn_kv_b.weight`. The `*attn_kv_b.weight` tensors have a shape of `512 x 4096`, so the upper half is `512 x 2048`. At run time it multiplies activations `X` to produce a `2048 x n_token` tensor, which is then viewed as `128 x n_token x 16` for further processing by the 16 attention heads. On the other hand, `*attn_k_b.weight` is stored as `128 x 8192` and is then viewed as `128 x 512 x 16` for multiplication with the query `Q`, so the imatrix data collection functions sees a matrix with just 128 columns, so quite useless to actually guide the quantization process. To make this actually useful, a modification in the `imatrix` tool is required to collect data for `128 x 16` columns, along with a modification in the quantization function to make use of imatrix data with `128 x 16` columns. This is left for a future PR, so for now there will be no imatrix data for `*attn_k_b.weight` even if the imatrix was computed with MLA enabled.

---

#### 💬 Conversation

👤 **davidsyoung** commented the **2025-03-10** at **14:24:47**:<br>

This is great, for lack of better understanding, if I am using an imatrix file that I assume was computed with standard attention, and I re-compute now, I should see better performance due to the `attn_v_b.weight` tensor now having imatrix data?

It's still of course lacking the imatrix data for `attn_k_b.weight` tensor. It would be interesting to understand what difference these changes will make to perplexity.

---

👤 **ikawrakow** commented the **2025-03-10** at **15:08:27**:<br>

If you are quantizing the attention tensors to `q8_0` you will not see a difference. The imatrix helps a lot for 1-, 2-, and 3-bit quantization, has a more modest impact at 4 bits, has almost no impact at 5 bits, and has basically no impact at 6+ bits.

---

👤 **davidsyoung** commented the **2025-03-10** at **15:21:47**:<br>

> If you are quantizing the attention tensors to `q8_0` you will not see a difference. The imatrix helps a lot for 1-, 2-, and 3-bit quantization, has a more modest impact at 4 bits, has almost no impact at 5 bits, and has basically no impact at 6+ bits.

Great to know, thank you!