### 🔀 [#143](https://github.com/ikawrakow/ik_llama.cpp/pull/143) - Slightly faster IQ4_XS_R4 on AVX2

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-16 |
| **Updated** | 2024-12-16 |

---

#### Description

PPL-512(LLaMA-3.1-8B) on Ryzen-5975WX goes to 262.2 t/s up from 248.2 t/s.

On AVX2/Zen4 it is much better to interleave 8 rows - see [this branch](https://github.com/ikawrakow/ik_llama.cpp/tree/ik/iq4_xs_r8). We get 284 t/s on Zen4 and 275 t/s on AVX2. But the `ARM_NEON` implementation becomes extremely messy, and we get ~1-2% lower performance. Hence sticking with 4 interleaved rows for now.