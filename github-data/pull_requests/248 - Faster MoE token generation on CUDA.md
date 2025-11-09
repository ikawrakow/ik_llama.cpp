## 🔀 [Pull Request #248](https://github.com/ikawrakow/ik_llama.cpp/pull/248) - Faster MoE token generation on CUDA

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/cuda_faster_moe_tg` |
| **Target Branch** | `main` |
| **Created** | 2025-03-09 |
| **Updated** | 2025-03-10 |
| **Merged** | 2025-03-10 |

---

## 📄 Description

This PR adds special purpose matrix-vector multiplications for MoE models.

For DeepSeek-Lite this results in a ~25% speedup for token generation.

For now only implemented ~with the `-fmoe` option and only~ for quantized experts.