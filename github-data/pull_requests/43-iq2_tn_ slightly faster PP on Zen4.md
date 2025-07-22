### 🔀 [#43](https://github.com/ikawrakow/ik_llama.cpp/pull/43) - iq2_tn: slightly faster PP on Zen4

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-08 |
| **Updated** | 2024-09-08 |

---

#### Description

With this change we get `PP512 = 494 t/s` (using flash attention), up from `468 t/s` (~5% improvement) running on a Ryzen-7950X CPU.

Compared to the initial `IQ2_TN` PR #13 the cumulative improvement is 15%.

Compared to `TQ2_0` in `llama.cpp`, which has now been merged, we are now 80% faster.