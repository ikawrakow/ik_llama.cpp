## 🔀 [Pull Request #75](https://github.com/ikawrakow/ik_llama.cpp/pull/75) - Fix Q5_0 flash attention

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_q5_0_fa` |
| **Target Branch** | `main` |
| **Created** | 2024-10-01 |
| **Updated** | 2024-10-01 |
| **Merged** | 2024-10-01 |

---

## 📄 Description

When I changed `iqk_mul_mat` to use type-1 dot products for type-0 legacy quants, I forgot to also change the `vec_dot_type` when the dot product is done via ggml as in flash attention. This PR fixes it.