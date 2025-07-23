### 🔀 [#126](https://github.com/ikawrakow/ik_llama.cpp/pull/126) - Rename iq4_nl_x4 to iq4_nl_r4

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-12-08 |
| **Updated** | 2024-12-08 |

---

#### Description

To be consistent with the other quants interleaving 4 rows.

I started the interleaved rows experiment with `IQ4_NL` and named the packing `IQ4_NL_X4`. But then I thought that `_X4` is actually ambiguous. 4 times of what? We already have quants where 4 consecutive blocks are packed together into a larger "X4" block. Because of that I named all following interleaved rows quants using "_R4" (as in 4 rows). To be consistent with this naming convention this PR renames `IQ4_NL_X4` to `IQ4_NL_R4`.