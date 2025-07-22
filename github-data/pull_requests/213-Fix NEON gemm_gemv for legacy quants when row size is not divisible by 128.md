### 🐛 [#213](https://github.com/ikawrakow/ik_llama.cpp/pull/213) - Fix NEON gemm/gemv for legacy quants when row size is not divisible by 128

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-20 |
| **Updated** | 2025-02-20 |

---

#### Description

I have broken it quite a while ago when I changed the NEON implementation to do two rows at a time. I haven't noticed as all models I typically use have row sizes that are multiple of 128. But as I was working on the `IQ1_S` NEON implementation for PR #212, I was testing with DeepSeek-Lite (where K cache row size is 576, so not divisible by 128), using `Q8_0` for K cache (but no FA, where it works), and was getting NaNs or gibberish. I lost so much time until I finally realized that the issue is with the K cache `Q8_0` matrix multiplication rather than my `IQ1_S` implementation.

This PR fixes this.