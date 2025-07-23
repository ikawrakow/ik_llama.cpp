### 🔀 [#218](https://github.com/ikawrakow/ik_llama.cpp/pull/218) - Better strategy for attention matrix multiplications when generating tokens 

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-21 |
| **Updated** | 2025-02-22 |

---

#### Description

The `K*Q` and `V*softmax(K*Q)` matrix multiplications have the shape

$$\left(K x N_t x N_k\right) \times \left(K x N_b x N_h\right)$$ 

where $K$ is the head size, $N_t$ is the number of tokens in the cache, $N_b$ is the number of tokens in the current batch, $N_k$ is the `K` or `V` number of heads, and $N_h$ is the total number of heads. In `llama.cpp` this tensor multiplication has been traditionally performed as $N_h$ consecutive matrix multiplications, each being of shape

$$\left(K x N_t\right) \times \left(K x N_b\right)$$
 
The issue with this is that for token generation (TG) we have $N_b = 1$, so we are dealing with $N_h$ matrix-vector multiplications, which are notoriously memory bound, and hence limit performance for large cache size (long contexts). To add insult to injury, the stride between consecutive rows in the left matrix is not just the row size $R$, but rather $N_k R$, so fetching data from memory is associated with big jumps and sub-optimal cache use, which is not exactly ideal in a memory bound situation.

When $N_h > N_k$ (GQA, in that case $N_h$ is divisible by $N_k$), PR #207 changed the multiplication strategy to perform $N_k$ matrix multiplications, each with shape $\left(K x N_t\right) \times \left(K x N_h/N_k\right)$, thus turning many matrix-vector multiplications into fewer matrix-matrix multiplications. This leads to non negligible performance gains for long contexts.

But when $N_h = N_k$ (e.g., DeepSeek attention architecture), the above does not work. What we could do instead is to perform $N_t x N_h$ dot products, where the inner loop is over $N_h$ and the outer loop is over $N_t$. When multi-threaded, each thread performs $N_t/M x N_h$ dot products (where $M$ is the number of threads). The advantage of doing this is that memory is accessed consecutively, resulting in better throughput and cache utilization. This is being done with this PR.

To access performance impact, I use DeepSeek-Lite quantized with `IQ1_S`. This minimizes the model size, thus allowing to achieve higher tokens per second and hence the size of the KV cache has a stronger impact. Calculations are on a Ryzen-7950X (Zen4), Ryzen-5975WX (AVX2) and M2-Max CPU (NEON).  Calculations are without FA so the change in tensor multiplication strategy is invoked. As performance is also influenced by cache size and quantization type (if the cache is quantized), we examine `fp16, Q8_0, Q8_KV` and, on Zen4, `bf16` for the K-cache (without FA the V cache cannot be quantized).

### AVX2

| model                | threads | type_k |          test |      t/s (main)  |     t/s (PR)     |   Speedup |
| -------------------- | ------: | -----: | ------------: | ---------------: | ---------------: | --------: |
| deepseek2 16B IQ1_S  |      16 |   fp16 |   tg128@pp128 |     40.39 ± 0.03 |     42.76 ± 0.03 |   1.059   |   
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp256 |     37.51 ± 0.00 |     41.37 ± 0.03 |   1.103   |   
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp512 |     32.31 ± 0.01 |     38.63 ± 0.01 |   1.196   |   
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp1024 |     26.64 ± 0.01 |     34.28 ± 0.02 |   1.289   |   
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp2048 |     19.82 ± 0.00 |     27.81 ± 0.01 |   1.403   |   
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp4096 |     13.60 ± 0.01 |     20.57 ± 0.00 |   1.512   |   
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp8192 |      8.38 ± 0.00 |     13.71 ± 0.00 |   1.636   |   
| deepseek2 16B IQ1_S  |      16 |        | tg128@pp16384 |      4.77 ± 0.00 |      8.20 ± 0.00 |   1.719   |   
| deepseek2 16B IQ1_S  |      16 |  q8_KV |   tg128@pp128 |     42.11 ± 0.00 |     42.74 ± 0.02 |   1.015   |   
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp256 |     40.26 ± 0.02 |     41.66 ± 0.02 |   1.035   |   
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp512 |     37.32 ± 0.01 |     39.94 ± 0.01 |   1.070   |   
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp1024 |     32.04 ± 0.00 |     36.32 ± 0.02 |   1.133   |   
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp2048 |     26.42 ± 0.01 |     31.48 ± 0.01 |   1.192   |   
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp4096 |     19.04 ± 0.01 |     24.04 ± 0.01 |   1.263   |   
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp8192 |     12.44 ± 0.00 |     16.25 ± 0.01 |   1.306   |   
| deepseek2 16B IQ1_S  |      16 |        | tg128@pp16384 |      6.88 ± 0.00 |     10.23 ± 0.00 |   1.487   |   
| deepseek2 16B IQ1_S  |      16 |   q8_0 |   tg128@pp128 |     42.77 ± 0.01 |     43.70 ± 0.01 |   1.022   |
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp256 |     41.07 ± 0.00 |     42.23 ± 0.00 |   1.028   |
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp512 |     38.53 ± 0.01 |     40.34 ± 0.00 |   1.047   |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp1024 |     33.90 ± 0.01 |     37.18 ± 0.02 |   1.097   |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp2048 |     27.15 ± 0.02 |     31.71 ± 0.00 |   1.168   |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp4096 |     19.88 ± 0.00 |     24.76 ± 0.00 |   1.245   |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp8192 |     13.03 ± 0.01 |     16.89 ± 0.01 |   1.296   |
| deepseek2 16B IQ1_S  |      16 |        | tg128@pp16384 |      8.03 ± 0.00 |     10.12 ± 0.00 |   1.260   |

### NEON (M2-Max CPU)

| model                | threads | type_k |          test |   t/s (main)     |     t/s (PR)     |   Speedup |
| -------------------  | ------: | -----: | ------------: | ---------------: | ---------------: | --------: |
| deepseek2 16B IQ1_S  |      8  | fp16   |   tg128@pp128 |     56.84 ± 0.05 |     58.21 ± 0.05 |  1.024    |   
| deepseek2 16B IQ1_S  |      8  |        |   tg128@pp256 |     54.55 ± 0.01 |     57.45 ± 0.07 |  1.053    |   
| deepseek2 16B IQ1_S  |      8  |        |   tg128@pp512 |     50.99 ± 0.04 |     55.47 ± 0.11 |  1.088    |   
| deepseek2 16B IQ1_S  |      8  |        |  tg128@pp1024 |     44.53 ± 0.48 |     51.93 ± 0.01 |  1.166    |   
| deepseek2 16B IQ1_S  |      8  |        |  tg128@pp2048 |     35.92 ± 0.02 |     45.80 ± 0.02 |  1.275    |   
| deepseek2 16B IQ1_S  |      8  |        |  tg128@pp4096 |     25.96 ± 0.01 |     37.36 ± 0.00 |  1.439    |   
| deepseek2 16B IQ1_S  |      8  |        |  tg128@pp4096 |     16.38 ± 0.11 |     27.21 ± 0.03 |  1.661    |   
| deepseek2 16B IQ1_S  |      8  |  q8_KV |   tg128@pp128 |     57.73 ± 0.28 |     58.10 ± 0.65 |  1.006    |   
| deepseek2 16B IQ1_S  |    8    |        |   tg128@pp256 |     56.40 ± 0.22 |     57.27 ± 0.02 |  1.015    |   
| deepseek2 16B IQ1_S  |    8    |        |   tg128@pp512 |     53.61 ± 0.41 |     55.95 ± 0.31 |  1.044    |   
| deepseek2 16B IQ1_S  |    8    |        |  tg128@pp1024 |     49.15 ± 0.12 |     54.00 ± 0.03 |  1.099    |   
| deepseek2 16B IQ1_S  |      8  |        |  tg128@pp2048 |     41.54 ± 0.12 |     48.59 ± 0.14 |  1.170    |   
| deepseek2 16B IQ1_S  |      8  |        |  tg128@pp4096 |     31.24 ± 0.00 |     41.31 ± 0.03 |  1.322    |   
| deepseek2 16B IQ1_S  |      8  |        |  tg128@pp8192 |     21.75 ± 0.01 |     31.66 ± 0.01 |  1.456    |   

### Zen4 (Ryzen-7950X)

| model                | threads | type_k |          test |     t/s (main)   |     t/s (PR)     |   Speedup |
| -------------------- | ------: | -----: | ------------: | ---------------: | ---------------: | --------: |
| deepseek2 16B IQ1_S  |      16 |   bf16 |   tg128@pp128 |     48.84 ± 0.08 |     49.32 ± 0.31 |  1.010    |
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp256 |     46.17 ± 0.27 |     47.52 ± 0.60 |  1.029    |
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp512 |     41.76 ± 0.17 |     44.86 ± 0.14 |  1.074    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp1024 |     36.58 ± 0.38 |     38.99 ± 0.13 |  1.066    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp2048 |     29.55 ± 0.03 |     33.11 ± 0.15 |  1.120    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp4096 |     20.95 ± 0.17 |     24.87 ± 0.25 |  1.187    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp8192 |     14.55 ± 0.48 |     16.72 ± 0.13 |  1.149    |
| deepseek2 16B IQ1_S  |      16 |        | tg128@pp16384 |      9.11 ± 0.00 |     10.14 ± 0.00 |  1.113    |
| deepseek2 16B IQ1_S  |      16 |   fp16 |   tg128@pp128 |     48.25 ± 0.42 |     49.61 ± 0.41 |  1.028    |
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp256 |     45.62 ± 0.04 |     47.76 ± 1.06 |  1.047    |
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp512 |     42.08 ± 0.22 |     45.34 ± 0.05 |  1.077    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp1024 |     37.14 ± 0.20 |     39.65 ± 0.00 |  1.068    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp2048 |     29.74 ± 0.23 |     33.98 ± 0.05 |  1.142    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp4096 |     21.98 ± 0.03 |     25.09 ± 0.05 |  1.141    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp8192 |     14.59 ± 0.07 |     16.92 ± 0.03 |  1.160    |
| deepseek2 16B IQ1_S  |      16 |        | tg128@pp16384 |      9.52 ± 0.00 |     10.10 ± 0.00 |  1.061    |
| deepseek2 16B IQ1_S  |      16 |  q8_KV |   tg128@pp128 |     49.87 ± 0.10 |     50.47 ± 0.21 |  1.012    |
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp256 |     46.89 ± 0.53 |     49.02 ± 0.16 |  1.045    |
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp512 |     44.08 ± 0.41 |     46.57 ± 0.25 |  1.056    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp1024 |     40.59 ± 0.09 |     42.50 ± 0.02 |  1.047    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp2048 |     34.32 ± 0.04 |     37.55 ± 0.18 |  1.094    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp4096 |     26.09 ± 0.99 |     29.50 ± 0.06 |  1.131    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp8192 |     19.43 ± 0.35 |     20.64 ± 0.04 |  1.062    |
| deepseek2 16B IQ1_S  |      16 |        | tg128@pp16384 |     11.48 ± 0.00 |     13.03 ± 0.00 |  1.135    |
| deepseek2 16B IQ1_S  |      16 |   q8_0 |   tg128@pp128 |     50.69 ± 0.17 |     50.70 ± 0.02 |  1.000    |
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp256 |     48.54 ± 0.15 |     49.55 ± 0.12 |  1.021    |
| deepseek2 16B IQ1_S  |      16 |        |   tg128@pp512 |     45.99 ± 0.11 |     46.98 ± 0.03 |  1.022    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp1024 |     42.85 ± 0.06 |     42.35 ± 0.05 |  0.988    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp2048 |     37.02 ± 0.11 |     37.57 ± 0.03 |  1.015    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp4096 |     29.10 ± 0.07 |     29.63 ± 0.00 |  1.018    |
| deepseek2 16B IQ1_S  |      16 |        |  tg128@pp8192 |     20.55 ± 0.09 |     20.71 ± 0.12 |  1.008    |
| deepseek2 16B IQ1_S  |      16 |        | tg128@pp16384 |     12.91 ± 0.00 |     13.06 ± 0.00 |  1.012    |