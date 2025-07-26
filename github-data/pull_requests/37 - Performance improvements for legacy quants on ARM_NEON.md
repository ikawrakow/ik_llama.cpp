### [Pull Request #37](https://github.com/ikawrakow/ik_llama.cpp/pull/37) - Performance improvements for legacy quants on ARM_NEON

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/neon_improve_legacy_quants` |
| **Target Branch** | `main` |
| **Created** | 2024-09-03 |
| **Updated** | 2024-09-04 |
| **Merged** | 2024-09-04 |

---

#### Description

If we process 2 rows in the left matrix at a time we get in the range of 20% performance boost for PP-512 (except for `Q8_0`, where performance was already higher than the other quants). The table summarizes the results or LLaMA-3.1-8B on an M2-Max CPU. As I like keeping track of how we perform relative to mainline `llama.cpp`, the table includes results for the current `llama.cpp` build (`69a480a (3660)`). tinyBLAS is enabled in `llama.cpp`, so the 33% (`Q4_0`) or 16.6% (`Q8_0`) improvement is compared to tinyBLAS, which does not provide implementation for `Q4_1`, `Q5_0` and `Q5_1` (and correspondingly the performance gap is much larger).

| Quants  |    t/s (llama.cpp)   |      t/s (main)  |       t/s (PR)   |   Speedup vs main |  Speedup vs llama.cpp |
| ------- | -------------------: | ---------------: | ---------------: | ----------------: | --------------------: |
| Q4_0    |         65.45 ± 0.01 |     72.88 ± 0.61 |     87.22 ± 0.85 |         1.197     |     1.333             |
| Q4_1    |         35.18 ± 0.51 |     59.95 ± 1.26 |     73.87 ± 0.47 |         1.232     |     2.100             |
| Q5_0    |         26.69 ± 0.35 |     62.63 ± 1.47 |     74.32 ± 0.13 |         1.187     |     2.785             |
| Q5_1    |         23.33 ± 0.06 |     52.83 ± 1.32 |     60.79 ± 0.19 |         1.151     |     2.606             |
| Q8_0    |         75.44 ± 1.84 |     85.08 ± 1.74 |     88.01 ± 0.11 |         1.034     |     1.166             |