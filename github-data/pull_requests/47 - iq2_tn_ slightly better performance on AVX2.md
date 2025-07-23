### 🔀 [#47](https://github.com/ikawrakow/ik_llama.cpp/pull/47) - iq2_tn: slightly better performance on AVX2

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-10 |
| **Updated** | 2024-09-10 |

---

#### Description

We get `PP-512 = 545` t/s for the 4B TriLM model compared to `PP-512 = 498` t/s on the main branch (on a Ryzen-5975WX). TG is not affected.

It is possible to increase `PP-512` performance to 600 t/s by representing `IQ2_TN` as a row scale + `IQ1_BN` packed quants, and reusing the `IQ2_BN` implementation, see the [iq2_tn_as_iq2_bn branch](https://github.com/ikawrakow/ik_llama.cpp/tree/ik/iq2_tn_as_iq2_bn). The issue with the `iq2_tn_as_iq2_bn` implementation is that TG performance on the Ryzen-5975WX saturates at about 38 t/s, while here we have 50.5 t/s. So, preferring this change for now, perhaps I can sort out where the TG bottleneck is in `iq2_tn_as_iq2_bn` later.