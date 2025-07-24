### 🔀 [#54](https://github.com/ikawrakow/ik_llama.cpp/pull/54) - Improve Q4_0 and Q8_0 performance on AVX2/Zen4

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-14 |
| **Updated** | 2024-09-14 |

---

#### Description

This PR improves `Q4_0` and `Q8_0` performance on `AVX2` and `Zen4`. The table shows comparisons to `llama.cpp` for LLaMA-3.1-8B on a Ryzen-7950X (Zen4) and a Ryzen-5975WX (AVX2) CPU.

| model         | backend    | threads |          test |     t/s (llama.cpp)  |     t/s (PR)      |   Speedup |
| --------------| ---------- | ------: | ------------: | -------------------: | ----------------: | --------: |
| llama 8B Q4_0 | Zen4       |      16 |         pp512 |        123.46 ± 0.09 |     165.26 ± 0.54 |  1.339    |   
| llama 8B Q8_0 | Zen4       |      16 |         pp512 |        141.30 ± 0.86 |     169.26 ± 0.57 |  1.200    |   
| llama 8B Q4_0 | Zen4       |       4 |         tg128 |         11.25 ± 0.02 |      13.88 ± 0.01 |  1.234    |   
| llama 8B Q8_0 | Zen4       |       4 |         tg128 |          7.56 ± 0.01 |       7.79 ± 0.02 |  1.030    |   
| llama 8B Q4_0 | AVX2       |      32 |         pp512 |        139.09 ± 0.62 |     212.70 ± 0.82 |  1.529    |   
| llama 8B Q8_0 | AVX2       |      32 |         pp512 |        162.21 ± 0.42 |     217.14 ± 0.65 |  1.339    |   
| llama 8B Q4_0 | AVX2       |       8 |         tg128 |         11.90 ± 0.00 |      11.99 ± 0.00 |  1.008    |
| llama 8B Q8_0 | AVX2       |       8 |         tg128 |          8.13 ± 0.00 |       8.21 ± 0.00 |  1.010    |