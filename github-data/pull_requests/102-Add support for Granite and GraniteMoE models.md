### 🔀 [#102](https://github.com/ikawrakow/ik_llama.cpp/pull/102) - Add support for Granite and GraniteMoE models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-22 |
| **Updated** | 2024-10-22 |

---

#### Description

On CUDA GraniteMoE-1b suffers from precision issues in the attention portion, so I became curious to see why. One way to avoid the NaNs is to set the precision of the `K*Q` matrix multiplication to `F32`. What also fixes it is to apply the attention scale on `Q` before the `K*Q` multiplication (the solution I went with in this PR). One can apply the scale before or after RoPE. It works in both cases, so this really narrows it down to the `K*Q` multiplication suffering from precision issues when done in `f16`.  Strange how these models were trained in the first place.