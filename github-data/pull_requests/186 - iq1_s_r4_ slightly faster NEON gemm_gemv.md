### [Pull Request #186](https://github.com/ikawrakow/ik_llama.cpp/pull/186) - iq1_s_r4: slightly faster NEON gemm/gemv

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/iq1_s_r4_neon` |
| **Target Branch** | `main` |
| **Created** | 2025-02-05 |
| **Updated** | 2025-02-05 |
| **Merged** | 2025-02-05 |

---

#### Description

DeepSeek-Lite on M2-Max CPU:

| model                  | threads |     test |   t/s (main)     |     t/s (PR)     |  Speedup |
| ---------------------- | ------: | -------: | ---------------: | ---------------: | -------: |
| deepseek2 16B IQ1_S_R4 |       2 |    tg128 |     22.76 ± 0.15 |     24.07 ± 0.19 |  1.058   |
| deepseek2 16B IQ1_S_R4 |       4 |    tg128 |     37.83 ± 0.00 |     39.58 ± 0.02 |  1.046   |
| deepseek2 16B IQ1_S_R4 |       8 |    tg128 |     62.01 ± 0.02 |     65.26 ± 0.82 |  1.052   |
| deepseek2 16B IQ1_S_R4 |       8 |    pp512 |    251.97 ± 0.09 |    283.20 ± 0.54 |  1.124   |