### 🔀 [#331](https://github.com/ikawrakow/ik_llama.cpp/pull/331) - Better gemm/gemv on AVX2 fr q4_0_r8

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-15 |
| **Updated** | 2025-04-15 |

---

#### Description

I constantly get confused how many `int16_t` dot products (`_mm256_maddubs_epi16()` results) I can sum up as `int16_t` before overflowing. In the case of `Q4_0` I was adding too few, and was having one unnecessary `_mm256_madd_epi16` because of that.  This PR fixes this. The result is a ~10% gain in performance when tested with Geema-3-12B-Instruct.