## 🔀 [Pull Request #630](https://github.com/ikawrakow/ik_llama.cpp/pull/630) - GEMM for IQ1_M

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/iq1m_gemm` |
| **Target Branch** | `main` |
| **Created** | 2025-07-18 |
| **Updated** | 2025-07-18 |
| **Merged** | 2025-07-18 |

---

## 📄 Description

Closes [#626](https://github.com/ikawrakow/ik_llama.cpp/issues/626) 

Hopefully the collective knowledge on Reddit and elsewhere that one cannot use `-fmoe` because of the missing `IQ1_M` GEMM has not already been perpetuated for all eternity...

After this PR, you can use `-fmoe` for any model.

Oh, no `ARM_NEON` for now, this will come later.