### 🔀 [#428](https://github.com/ikawrakow/ik_llama.cpp/pull/428) - Zen4: Faster PP for IQ2_KS, IQ4_KS, IQ5_KS

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-17 |
| **Updated** | 2025-05-17 |

---

#### Description

| model            |       size | threads |          test |     t/s (main)   |      t/s (PR) |  Speedup |
| ---------------- | ---------: | ------: | ------------: | ---------------: | ------------: | -------: |
| llama 8B IQ2_KS  |   2.46 GiB |      16 |         pp512 |    179.51 ± 1.13 | 196.20 ± 1.59 |  1.093   |   
| llama 8B IQ4_KS  |   4.14 GiB |      16 |         pp512 |    172.36 ± 1.28 | 198.57 ± 1.74 |  1.152   |   
| llama 8B IQ5_KS  |   4.95 GiB |      16 |         pp512 |    150.93 ± 1.61 | 196.20 ± 1.59 |  1.300   |