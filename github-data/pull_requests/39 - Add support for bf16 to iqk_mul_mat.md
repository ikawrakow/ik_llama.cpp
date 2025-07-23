### 🔀 [#39](https://github.com/ikawrakow/ik_llama.cpp/pull/39) - Add support for bf16 to iqk_mul_mat

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-04 |
| **Updated** | 2024-09-05 |

---

#### Description

Only when natively supported (e.g., Zen4), else left to `ggml` to handle.

For LLaMA-3.1-8B we get `PP512 = 205` t/s vs `74 t/s` in `llama.cpp` on my Ryzen-7950X CPU.

I get `204` t/s with [llamafile](https://github.com/Mozilla-Ocho/llamafile), so I guess Justine Tunney has not contributed the more recent `tinyBLAS` improvements to `llama.cpp`.