### 🔀 [#14](https://github.com/ikawrakow/ik_llama.cpp/pull/14) - Adding IQ6_K

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-08-09 |
| **Updated** | 2024-08-09 |

---

#### Description

This PR

* Adds `IQ6_K` - see #8 for motivation
* Fixes the Zen4 implementation of `IQ3_K`, `IQ4_K` and `IQ5_K`

### New IQ6_K

The graph below is a copy of the graph in #8 with the quantization error of the new `IQ6_K` non-linear quantization type added (cyan circle near 6.6 bpw).  We observe a significant improvement compared to `Q6_K` (0.4% vs 0.65%). LLaMA-3.1-8B quantization error is better too (0.15% vs 0.26%), so I think this is a worthwhile addition.

![l31_70B](https://github.com/user-attachments/assets/e8b4447c-cbf3-4bb8-9185-793f06510e3f)

### Fixing the Zen4 implementation of `IQ3_K`, `IQ4_K` and `IQ5_K`

While working on `IQ6_K`, I have noticed that there is a problem with the Zen4 implementation of the `IQ3,4,5_K` quants. I was using the standard k-quants matrix multiplication template (`mul_mat_qX_K_q8_K_AVX512`). On Zen4, this template uses the `_mm512_dpbusd_epi32` instruction to perform the dot product between the quants of the left matrix and the `Q8_K` quants of the right matrix, which produces a SIMD vector containing 32-bit integer results. But for k-quants these 32-bit integers fall within `int16_t` range, so they get packed to 16-bit and are then multiplied with the block scales. But for the 3+ bit non-linear quants, the `_mm512_dpbusd_epi32` may go outside of the `int16_t` range, which then leads to truncation and a wrong result. I have now corrected the implementation. This results in a small performance regression. The table below shows a performance comparison for LLaMA-3.1-8B between the original Zen4 implementation and the corrected Zen4 implementation for `IQ3_K` on a Ryzen-7950X  (using 16 threads for PP-512 and 4 threads for TG-128)

|   | t/s (PP-512) | t/s (TG-128) |
| ---: | ----: | ----: |
| Before fix | 180.77 ± 0.62 | 16.10 ± 0.16 |
| After fix  | 167.69 ± 0.69 | 15.84 ± 0.33 |
| Ratio      | 0.940 | 0.984 |