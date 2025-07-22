### 🔀 [#310](https://github.com/ikawrakow/ik_llama.cpp/pull/310) - Metal: FA and FlashMLA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-03 |
| **Updated** | 2025-04-03 |

---

#### Description

Performance is not great, but it works with standard attentions and all 3 MLA options.

"Works" as:
* `f16` KV cache works for all combinations of `fa` and `mla`
* I have allowed only `Q8_0` quantized cache
* Quantized cache only works with standard attention (`-mla 0`) without FA
* With FA quantized cache kind of works, but we get messages such as `ggml_metal_get_buffer: error: tensor 'v-26' buffer is nil`. Not sure why. PPL is slightly higher than without FA