### 🔀 [#158](https://github.com/ikawrakow/ik_llama.cpp/pull/158) - Faster R4 legacy quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-22 |
| **Updated** | 2024-12-22 |

---

#### Description

It seems converting `fp16` to `fp32` is extremely slow on the Ryzen-5975WX CPU (or `ggml`'s `GGML_FP16_TO_FP32` is inadequate), so it is better to convert the `fp16` `Q8_1_x4` block scales using `AVX2` intrinsics, store the result, and then use the converted `fp32` scales when performing the dot product. This PR does that on `AVX2` for `Q4_0_R4, Q5_0_R4, Q6_0_R4` and `Q8_0_R4`.  There was no benefit on the Ryzen-7950X (`Zen4`), so not implemented there.

The table shows PP-512 comparison between the main branch and this PR for LLaMA-3.1-8B on the Ryzen-5975WX

| Quant | t/s (main) | t/s (PR) | Speedup |
| ---: | ---: | ---: | ---: |
| Q4_0_R4 | 251.00 ± 0.51 | 283.61 ± 0.50 | 1.130 |
| Q5_0_R4 | 236.33 ± 0.56 | 271.57 ± 0.52 | 1.149 |
| Q6_0_R4 | 231.53 ± 0.60  | 260.22 ± 0.53 | 1.124 |
| Q8_0_R4 | 234.40 ± 0.60  | 246.11 ± 0.54 | 1.050 |