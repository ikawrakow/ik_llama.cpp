### 🔀 [#610](https://github.com/ikawrakow/ik_llama.cpp/pull/610) - q8_k_r8: experimental AVX512 version

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-14 |
| **Updated** | 2025-07-18 |

---

#### Description

@ubergarm This is specifically for your 9950X CPU.

On my 7950X this is ~10% slower than what we have on the main branch. The 7950X supports `AVX512`, but 512-bit instructions get executed as two 256-bit instructions. Hence, I'm expecting (hoping?) this `Q8_K_R8` GEMM version to be significantly faster on a CPU with "real" 512-bit instructions such as the 9950X.

Please benchmark it so I can decide if it is worth adding this to the main branch.