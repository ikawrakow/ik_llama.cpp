### 🔀 [#182](https://github.com/ikawrakow/ik_llama.cpp/pull/182) - Faster Q4_K_R4 and Q5_K_R4 on AVX2/Zen4

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-30 |
| **Updated** | 2025-01-30 |

---

#### Description

TG is about the same. PP-512 comparison between main and this PR for LLaMA-3.1-8B on a Ryzen-5975WX (`AVX2`) and a Ryzen-7950X (`Zen4`)

| model            | backend    | threads |    test |   t/s (main)     |   t/s (PR)    |  Speedup |
| ---------------- | ---------- | ------: | ------: | ---------------: | ------------: | -------: |
| llama 8B Q4_K_S  | AVX2       |      32 |   pp512 |    291.90 ± 0.64 | 327.98 ± 0.51 |  1.124   |   
| llama 8B Q5_K_S  | AVX2       |      32 |   pp512 |    273.59 ± 0.37 | 302.13 ± 0.61 |  1.104   |   
| llama 8B Q4_K_S  | Zen4       |      16 |   pp512 |    258.78 ± 1.05 | 267.69 ± 0.31 |  1.034   |   
| llama 8B Q5_K_S  | Zen4       |      16 |   pp512 |    246.19 ± 0.65 | 249.12 ± 0.42 |  1.012   |

The improvement on `Zen4` is very minor. The benefit there is bloat reduction as I'm now reusing the same implementation as `AVX2`.