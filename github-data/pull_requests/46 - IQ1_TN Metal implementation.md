### 🔀 [#46](https://github.com/ikawrakow/ik_llama.cpp/pull/46) - IQ1_TN Metal implementation

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-10 |
| **Updated** | 2024-09-10 |

---

#### Description

`IQ1_BN` stores a scale at the beginning of each row, followed by `IQ1_BN` packing of the ternary quants. The existing Metal implementation does not allow for that sort of thing, so some changes were necessary (apart from adding the necessary additions in `ggml-metal.m`): 
* We modify the `kernel_mul_mm` and `kernel_mul_mm_id_impl` templates to have a dequantizer type as a template parameter (instead of a dequantization function)
* We provide a default dequantizer that does what the existing implementation does. This is used for all existing quants
* We add a dequantizer for `IQ1_BN`. It simply gets the scale from the first two bytes of a row, uses the existing `IQ1_BN` implementation to convert the ternary bits to `float4x4` or `half4x4`, and then multiplies the result with the row scale before returning it to the caller.
* We also add a dequantization kernel that takes a dequantizer as a template parameter (heeded for `get_rows`)

With this, the `IQ1_TN` implementation is complete for all supported platforms (`Zen4`, `AVX2`, `ARM_NEON`, `CUDA`, `Metal`).