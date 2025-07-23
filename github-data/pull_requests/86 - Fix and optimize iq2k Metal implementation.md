### 🐛 [#86](https://github.com/ikawrakow/ik_llama.cpp/pull/86) - Fix and optimize iq2k Metal implementation

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-13 |
| **Updated** | 2024-10-13 |

---

#### Description

I completely forgot to change the `IQ2_K` Metal implementation after changing the `IQ2_K` block scales in the last PR. This PR fixes it. It also improves the performance of the `IQ2_K` Metal dot product - TG-128 for LLaMA-3.1-8B goes to 46.2 t/s up from 42.6 t./s.