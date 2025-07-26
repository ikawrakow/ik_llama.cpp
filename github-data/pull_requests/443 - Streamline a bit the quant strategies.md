### [Pull Request #443](https://github.com/ikawrakow/ik_llama.cpp/pull/443) - Streamline a bit the quant strategies

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `QS_streamline` |
| **Target Branch** | `main` |
| **Created** | 2025-05-22 |
| **Updated** | 2025-05-22 |
| **Merged** | 2025-05-22 |

---

#### Description

Unlike last time..

No change over the existing patterns, except for the bump for attn_k and attn_v for the models with 4 and 6 experts (several frankensteins seen on HF, and which also use GQA).
The rest is applying the existing patterns to the new IQ_K quants.
Also, a Q8_0 for attn_q slipped into the MOEs 8 experts rule, I removed it, because that tensor is much bigger than attn_k or attn_v.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [x] Low
  - [ ] Medium
  - [ ] High