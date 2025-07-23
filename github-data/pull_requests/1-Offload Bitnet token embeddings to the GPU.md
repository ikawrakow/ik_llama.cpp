### 🔀 [#1](https://github.com/ikawrakow/ik_llama.cpp/pull/1) - Offload Bitnet token embeddings to the GPU

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-07-26 |
| **Updated** | 2024-07-26 |

---

#### Description

This PR puts the `token_embedding` tensor on the GPU for the Bitnet-1.58b model. This results in a significantly improved performance on CUDA/Metal as can be seen in the table. `CUDA` is for RTX-4080, `Metal` is for a 30-code M2-Max GPU, the host CPU is a Ryzen-7950X for `CUDA`.

| model  | backend    |    test | t/s (PR)      |  t/s (main)   | Speedup |
| ------ | ---------: | ------: | ------------: | ------------: | ------: |
| IQ2_BN | CUDA       |   tg128 | 322.10 ± 0.07 | 241.34 ± 0.27 | 1.325   |   
| IQ1_BN | CUDA       |   tg128 | 301.44 ± 0.12 | 229.21 ± 0.89 | 1.315   |   
| IQ2_BN | CUDA       |   pp512 | 10780 ± 164   | 9811 ± 25     | 1.099   |   
| IQ1_BN | CUDA       |   pp512 | 10661 ± 172   | 9655 ± 21     | 1.104   |   
| IQ2_BN | Metal      |   pp512 | 723.19 ± 0.53 | 722.66 ± 0.47 | 1.001   |   
| IQ1_BN | Metal      |   pp512 | 698.25 ± 1.91 | 697.59 ± 2.12 | 1.000   |   
| IQ2_BN | Metal      |   tg128 | 110.39 ± 0.13 | 95.22 ± 0.55  | 1.159   |   
| IQ1_BN | Metal      |   tg128 |  76.70 ± 0.05 | 69.33 ± 0.07  | 1.106   |   

Bitnet uses the same tensor for token embeddings and for output. When the token embedding tensor is specified to be on the CPU, as done in mainline `llama.cpp` and here before this PR, this leads to the final matrix multiplication with the output tensor to be performed on the CPU even when using a GPU backend, and this results in a significant drop in performance (the larger the performance differential between the GPU and the host CPU, the larger the effect). As this might affect other models as well (e.g., Gemma), it would be useful to find a more general solution, but I'm finding the back-end stuff in `llama.cpp` to be opaque and hard to understand, so solved in a hacky way just for Bitnet for now.