### 🔀 [#219](https://github.com/ikawrakow/ik_llama.cpp/pull/219) - Fuse MoE up and gate matrix multiplications

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-22 |
| **Updated** | 2025-02-22 |

---

#### Description

No new op, instead the fusing is done during graph compute in the CPU back end (same could be also done for the other back ends).
 
The advantage of fusing the `ffn_up` and `ffn_gate` matrix multiplication is that a) there is one less thread synchronization; b) half the threads evaluate `ffn_up` and the other half `ffn_gata` in parallel.

This leads to a small but measurable performance gain (1-2%) for PP and TG.

As for MoE models the `ffn_up` and `ffn_gate` matrix multiplications are always followed by element wise  multiplication of `result1 * op(result2)` (where `op` is `SILU` or `GELU`), one could go one step further and add a new operation that does all of this together. This would a) further reduce thread synchronization cost and b) reduce memory writes/loads by removing the need for the intermediate results. But this is a bigger change that requires implementation of the new op on CUDA and Metal, so left for another day.