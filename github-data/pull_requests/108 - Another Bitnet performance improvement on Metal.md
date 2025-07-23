### 🔀 [#108](https://github.com/ikawrakow/ik_llama.cpp/pull/108) - Another Bitnet performance improvement on Metal

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-26 |
| **Updated** | 2024-10-26 |

---

#### Description

This time just the dequantize function. 

For Bitnet-1.58b-3B on 30-core M2-Max GPU 
* `IQ1_BN` goes from 702 t/s to 716 t/s
* `IQ2_BN` goes from 714 t/s to 743 t/s