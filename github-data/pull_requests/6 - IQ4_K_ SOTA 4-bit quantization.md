### 🔀 [#6](https://github.com/ikawrakow/ik_llama.cpp/pull/6) - IQ4_K: SOTA 4-bit quantization

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-07-28 |
| **Updated** | 2024-07-28 |

---

#### Description

* Same 4.5 bpw as `Q4_K`.
* Significantly reduces quantization error of LLaMA-3.1 (and also 3.0). E.g., 1.77% vs 2.9% for `Q4_K_S` for LLaMA-3.1-8B (with quantization error defined as `PPL(Q)/PPL(fp16)-1`)
* Non-linear quantization similar to `IQ4_XS` and `IQ4_NL` with the following differences
  - Blocks of 16 instead of blocks of 32
  - Non-linear values in each block of 16 can be on the original non-linear grid, or can be on a shifted grid. This is indicated by one bit, so we need 16 extra bits per block of 256
  - So, we need `256 * 4` bits for the quants, `16 * 6` bits for the 6-bit block scales, 16 bits for the super-block float scale, and 16 bits for the shift bits, ending up with exactly 4.5 bpw
 * Performance is on par with `Q4_K` on `AVX2` and `CUDA`, and slightly lower on `ARM_NEON` and `Metal`