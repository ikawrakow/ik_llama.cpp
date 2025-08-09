## 🔀 [Pull Request #603](https://github.com/ikawrakow/ik_llama.cpp/pull/603) - Check if MMQ should be used before using it

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_596` |
| **Target Branch** | `main` |
| **Created** | 2025-07-12 |
| **Updated** | 2025-07-13 |
| **Merged** | 2025-07-13 |

---

## 📄 Description

In [#589](https://github.com/ikawrakow/ik_llama.cpp/issues/589) I added an optimization of the fused ffn_up/gate op to not repeat the quantization of the activations when `ffn_up` and `ffn_gate` are quantized with the same type. But the check to use the direct route did not consider the possibility that some quantization types do not have MMQ implementation (e.g., `IQ1_M`), which then results in an assert.

This PR adds the missing check, which should fix [#596](https://github.com/ikawrakow/ik_llama.cpp/issues/596)