### 🔀 [#107](https://github.com/ikawrakow/ik_llama.cpp/pull/107) - Faster IQ1_BN Metal implementation

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-26 |
| **Updated** | 2024-10-26 |

---

#### Description

On my 30-core M2-Max TG-128 for Bitnet-1.58b-3.3B improves from 82 t/s to 94.7 t/s.
PP-512 goes from 686 t/s to 702 t/s.

Integer multiplications are expensive, so the trick used is to replace them with shifts and additions.

There is also a minor `IQ2_BN` PP-512 improvement (710 -> 714 t/s).