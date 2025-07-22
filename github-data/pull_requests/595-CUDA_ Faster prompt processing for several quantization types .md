### 🔀 [#595](https://github.com/ikawrakow/ik_llama.cpp/pull/595) - CUDA: Faster prompt processing for several quantization types 

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-09 |
| **Updated** | 2025-07-10 |

---

#### Description

This PR slightly improves prompt processing speed for `IQ3_K, IQ3_K_R4, IQ4_KS, IQ4_KS_R4, IQ4_K, IQ4_K_R4` and `IQ4_XS`.

Here some PP-512 results for LlaMA-3.1-8B on RTX-4080

 | model              |          test |    t/s (main)    |    t/s (PR)      |  Speedup |
| ------------------ | ------------: | ---------------: | ---------------: | -------: |
| llama 8B IQ3_K     |         pp512 |  6467.57 ± 18.48 |  6628.75 ± 14.24 |  1.025   |   
| llama 8B IQ3_K_R4  |         pp512 |  6102.36 ± 14.63 |  6464.58 ± 10.89 |  1.059   |   
| llama 8B IQ4_K     |         pp512 |  6442.38 ± 17.97 |  6625.94 ± 22.90 |  1.028   |   
| llama 8B IQ4_K_R4  |         pp512 |  6391.48 ± 16.77 |  6450.58 ± 11.54 |  1.009   |   
| llama 8B IQ4_KS    |         pp512 |  7732.35 ± 26.04 |  8074.07 ± 16.37 |  1.044   |   
| llama 8B IQ4_KS_R  |         pp512 |  7912.27 ± 21.10 |  8178.74 ± 28.14 |  1.034   |   
| llama 8B IQ4_XS    |         pp512 |  7748.68 ± 20.75 |  8149.86 ± 28.13 |  1.051   |