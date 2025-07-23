### 🔀 [#41](https://github.com/ikawrakow/ik_llama.cpp/pull/41) - iqk_mul_mat(ARM_NEON): adding bf16 support

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-05 |
| **Updated** | 2024-09-16 |

---

#### Description

It looks like ArmV8 ISA has support for `bf16`, but my M2 Max does not have it, so resorting to `bf16 -> f32` conversion and computations in `f32`. This is 2X slower than `f16`, but 8X better compared to what I get if I try to run a `bf16` model on the M2 (`NEON` and `Metal`).