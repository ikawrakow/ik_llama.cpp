### 🔀 [#90](https://github.com/ikawrakow/ik_llama.cpp/pull/90) - iq4_ks: faster dot product on Metal

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-16 |
| **Updated** | 2024-10-16 |

---

#### Description

Haha, I keep forgetting that the Metal compiler often needs a hand to produce fast code.
In this particular instance, we gain almost 8.5% token generation (TG) speedup for `IQ4_KS`:
TG-128(LLaMA-3.1-8B) goes to 52.5 t/s up from 48.4 t/s on my M2-Max 30-core GPU.
The actual computation did not change in any way, we just helped the compiler fetch data ore effectively.