### 🗣️ [#164](https://github.com/ikawrakow/ik_llama.cpp/discussions/164) - Latest CPU performance comparison with llama.cpp

| **Author** | `ikawrakow` |
| :--- | :--- |
| **Created** | 2024-12-24 |
| **Updated** | 2025-04-28 |

---

#### Description

There has been quite a bit of development here and in mainline `llama.cpp` since the performance results on the front page were generated, so I decided to make a new CPU performance comparison.

* Using `llama.cpp` build `14b699ec (4384)` (latest as of December 23 2024)
* Quantization is performed with mainline `llama.cpp`
* Performance is evaluated using the `llama-bench` tool for `PP-512` and `TG-128`
* For the results of `ik_llama.cpp` the command-line option `-rtr 1` is used when running `llama-bench`. This causes all model weights to be repacked into row-interleaved format (if available)
* `AVX2/Zen4` performance is on a Ryzen-7950X, `ARM` is on `M2-Max`
* LLaMA-3.1-8B-Instruct is used in all cases
* For not quantized variants the respective native 16-bit floats are used (`fp16` on M2-Max, `bf16` on the Ryzen-7950X)

### AVX2

| model                    |       size | threads |          test |   t/s (llama.cpp)    | t/s (ik_llama.cpp) |  Speedup |
| ------------------------ | ---------: | ------: | ------------: | -------------------: | -----------------: | -------: |
| 8B BF16                  |  14.96 GiB |      16 |         pp512 |         78.58 ± 0.10 |      256.90 ± 0.36 |   3.269  |
| 8B BF16                  |  14.96 GiB |       2 |         tg128 |          4.05 ± 0.00 |        4.27 ± 0.00 |   1.054  |
| 8B Q8_0                  |   7.95 GiB |      16 |         pp512 |        147.92 ± 0.52 |      268.19 ± 0.19 |   1.813  |
| 8B Q8_0                  |   7.95 GiB |       2 |         tg128 |          4.95 ± 0.01 |        7.63 ± 0.00 |   1.541  |
| 8B Q5_0                  |   5.22 GiB |      16 |         pp512 |        111.68 ± 0.36 |      251.21 ± 0.41 |   2.249  |
| 8B Q5_0                  |   5.22 GiB |       2 |         tg128 |          5.30 ± 0.00 |       11.14 ± 0.00 |   2.102  |
| 8B Q4_0                  |   4.35 GiB |      16 |         pp512 |        153.52 ± 0.21 |      273.54 ± 0.33 |   1.782  |
| 8B Q4_0                  |   4.35 GiB |       2 |         tg128 |         11.23 ± 0.01 |       12.92 ± 0.00 |   1.150  |
| 8B Q2_K - Small          |   2.78 GiB |      16 |         pp512 |        122.37 ± 0.31 |      269.96 ± 0.29 |   2.206  |
| 8B Q2_K - Small          |   2.78 GiB |       2 |         tg128 |         11.33 ± 0.00 |       17.10 ± 0.01 |   1.509  |
| 8B Q3_K - Small          |   3.41 GiB |      16 |         pp512 |         85.19 ± 0.32 |      255.30 ± 0.24 |   2.997  |
| 8B Q3_K - Small          |   3.41 GiB |       2 |         tg128 |          8.80 ± 0.00 |       12.99 ± 0.01 |   1.476  |
| 8B Q4_K - Small          |   4.36 GiB |      16 |         pp512 |        108.40 ± 0.25 |      269.60 ± 0.27 |   2.487  |
| 8B Q4_K - Small          |   4.36 GiB |       2 |         tg128 |          9.57 ± 0.00 |       13.48 ± 0.00 |   1.409  |
| 8B Q5_K - Small          |   5.21 GiB |      16 |         pp512 |         75.52 ± 0.19 |      254.68 ± 0.36 |   3.372  |
| 8B Q5_K - Small          |   5.21 GiB |       2 |         tg128 |          7.51 ± 0.00 |       11.41 ± 0.00 |   1.519  |
| 8B Q6_K                  |   6.14 GiB |      16 |         pp512 |         82.56 ± 0.28 |      259.21 ± 0.37 |   3.140  |
| 8B Q6_K                  |   6.14 GiB |       2 |         tg128 |          7.62 ± 0.00 |       10.05 ± 0.00 |   1.319  |
| 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |      16 |         pp512 |        123.36 ± 0.27 |      265.88 ± 0.52 |   2.155  |
| 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |       2 |         tg128 |          5.96 ± 0.01 |        9.30 ± 0.00 |   1.560  |
| 8B IQ4_XS - 4.25 bpw     |   4.13 GiB |      16 |         pp512 |         74.39 ± 0.18 |      269.91 ± 0.37 |   3.628  |
| 8B IQ4_XS - 4.25 bpw     |   4.13 GiB |       2 |         tg128 |          8.15 ± 0.00 |       13.58 ± 0.00 |   1.666  |
| 8B IQ2_XXS - 2.0625 bpw  |   2.23 GiB |      16 |         pp512 |         45.78 ± 0.09 |      164.37 ± 0.48 |   3.590  |
| 8B IQ2_XXS - 2.0625 bpw  |   2.23 GiB |       2 |         tg128 |          5.47 ± 0.00 |        8.74 ± 0.01 |   1.598  |
| 8B IQ2_XS - 2.3125 bpw   |   2.42 GiB |      16 |         pp512 |         49.72 ± 0.06 |      156.50 ± 0.26 |   3.148  |
| 8B IQ2_XS - 2.3125 bpw   |   2.42 GiB |       2 |         tg128 |          5.87 ± 0.00 |        6.87 ± 0.00 |   1.170  |
| 8B IQ2_M - 2.7 bpw       |   2.74 GiB |      16 |         pp512 |         43.80 ± 0.09 |      181.64 ± 0.62 |   4.147  |           
| 8B IQ2_M - 2.7 bpw       |   2.74 GiB |       2 |         tg128 |          5.24 ± 0.00 |        5.57 ± 0.00 |   1.063  |
| 8B IQ3_XXS - 3.0625 bpw  |   3.04 GiB |      16 |         pp512 |         34.17 ± 0.06 |      149.68 ± 0.14 |   4.380  |
| 8B IQ3_XXS - 3.0625 bpw  |   3.04 GiB |       2 |         tg128 |          4.18 ± 0.01 |        6.23 ± 0.00 |   1.490  |
| 8B IQ3_S - 3.4375 bpw    |   3.42 GiB |      16 |         pp512 |         30.20 ± 0.05 |      156.47 ± 0.34 |   5.181  |
| 8B IQ3_S - 3.4375 bpw    |   3.42 GiB |       2 |         tg128 |          3.71 ± 0.00 |        4.47 ± 0.00 |   1.205  |

### ARM_NEON

| model                    |       size | threads |          test |   t/s (llama.cpp)    | t/s (ik_llama.cpp) |  Speedup |
| ------------------------ | ---------: | ------: | ------------: | -------------------: | -----------------: | -------: |
| 8B F16                   |  14.96 GiB |       8 |         pp512 |         28.96 ± 0.27 |       91.24 ± 0.24 |   3.151  | 
| 8B F16                   |  14.96 GiB |       4 |         tg128 |          7.89 ± 0.02 |        7.89 ± 0.02 |   1.000  |  
| 8B Q8_0                  |   7.95 GiB |       8 |         pp512 |         54.54 ± 1.35 |      129.70 ± 1.33 |   2.378  |
| 8B Q8_0                  |   7.95 GiB |       3 |         tg128 |         14.04 ± 0.02 |       14.29 ± 0.05 |   1.017  | 
| 8B Q5_0                  |   5.22 GiB |       8 |         pp512 |         25.15 ± 0.92 |      103.94 ± 0.62 |   4.133  |
| 8B Q5_0                  |   5.22 GiB |       4 |         tg128 |         12.20 ± 0.01 |       16.63 ± 0.04 |   1.363  | 
| 8B Q4_0                  |   4.35 GiB |       8 |         pp512 |        114.63 ± 2.08 |      122.52 ± 0.15 |   1.069  |
| 8B Q4_0                  |   4.35 GiB |       4 |         tg128 |         23.89 ± 0.13 |       23.43 ± 0.22 |   0.981  | 
| 8B Q2_K - Small          |   2.78 GiB |       8 |         pp512 |         33.02 ± 0.05 |      108.98 ± 0.24 |   3.300  |
| 8B Q2_K - Small          |   2.78 GiB |       4 |         tg128 |         13.91 ± 0.01 |       23.49 ± 0.12 |   1.689  | 
| 8B Q3_K - Small          |   3.41 GiB |       8 |         pp512 |         24.95 ± 0.02 |      107.16 ± 0.64 |   4.295  |
| 8B Q3_K - Small          |   3.41 GiB |       4 |         tg128 |         11.10 ± 0.00 |       15.29 ± 0.04 |   1.377  | 
| 8B Q4_K - Small          |   4.36 GiB |       8 |         pp512 |         43.30 ± 0.57 |      126.53 ± 0.45 |   2.922  |
| 8B Q4_K - Small          |   4.36 GiB |       4 |         tg128 |         17.55 ± 0.01 |       22.49 ± 0.07 |   1.281  | 
| 8B Q5_K - Small          |   5.21 GiB |       8 |         pp512 |         27.82 ± 0.52 |      108.44 ± 0.19 |   3.898  |
| 8B Q5_K - Small          |   5.21 GiB |       4 |         tg128 |         12.26 ± 0.01 |       16.15 ± 0.05 |   1.317  | 
| 8B Q6_K                  |   6.14 GiB |       8 |         pp512 |         26.73 ± 0.46 |      106.15 ± 1.22 |   3.971  |
| 8B Q6_K                  |   6.14 GiB |       4 |         tg128 |         11.62 ± 0.01 |       14.86 ± 0.05 |   1.279  | 
| 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |       8 |         pp512 |         92.64 ± 2.46 |      121.59 ± 1.41 |   1.313  |
| 8B IQ4_NL - 4.5 bpw      |   4.35 GiB |       4 |         tg128 |         23.45 ± 0.06 |       22.97 ± 0.01 |   0.980  | 
| 8B IQ4_XS - 4.25 bpw     |   4.13 GiB |       8 |         pp512 |         37.90 ± 0.59 |      134.02 ± 0.66 |   3.536  |
| 8B IQ4_XS - 4.25 bpw     |   4.13 GiB |       4 |         tg128 |         16.03 ± 0.02 |       23.36 ± 0.18 |   1.457  | 
| 8B IQ2_XXS - 2.0625 bpw  |   2.23 GiB |       8 |         pp512 |         18.50 ± 0.53 |       87.89 ± 0.76 |   4.751  | 
| 8B IQ2_XXS - 2.0625 bpw  |   2.23 GiB |       4 |         tg128 |          8.67 ± 0.02 |       12.28 ± 0.10 |   1.416  | 
| 8B IQ2_XS - 2.3125 bpw   |   2.42 GiB |       8 |         pp512 |         20.40 ± 0.37 |       70.09 ± 0.12 |   3.436  | 
| 8B IQ2_XS - 2.3125 bpw   |   2.42 GiB |       4 |         tg128 |          9.49 ± 0.01 |       11.12 ± 0.09 |   1.172  | 
| 8B IQ2_M - 2.7 bpw       |   2.74 GiB |       8 |         pp512 |         14.61 ± 0.02 |       67.56 ± 0.41 |   4.624  | 
| 8B IQ2_M - 2.7 bpw       |   2.74 GiB |       4 |         tg128 |          6.77 ± 0.01 |        8.90 ± 0.02 |   1.315  |  
| 8B IQ3_XXS - 3.0625 bpw  |   3.04 GiB |       8 |         pp512 |         13.42 ± 0.14 |       78.29 ± 0.33 |   5.833  | 
| 8B IQ3_XXS - 3.0625 bpw  |   3.04 GiB |       4 |         tg128 |          6.26 ± 0.01 |        8.54 ± 0.07 |   1.364  |  
| 8B IQ3_S - 3.4375 bpw    |   3.42 GiB |       8 |         pp512 |         11.49 ± 0.01 |       80.89 ± 0.25 |   7.040  | 
| 8B IQ3_S - 3.4375 bpw    |   3.42 GiB |       4 |         tg128 |          5.34 ± 0.01 |        6.61 ± 0.02 |   1.238  |  

* We see that the CPU performance gap has widened significantly since July when I made the comparison on the front page.
*  Only `llama.cpp's` low-quality 4-bit quantization `Q4_0` on `ARM_NEON` (which gets repacked to a 4-row interleaved format, formerly known as `Q4_0_4_4`) is competitive.
* The performance gap grown is taken by `IQ3_S` (7X faster on the M2-Max, 5.2X faster on the Ryzen-7950X).
* Even mainstream k-quants are now very significantly faster here
* On the Ryzen-7950X the slowest quantization type in `ik_llama.cpp` is faster than the fastest type in `llama.cpp` for prompt processing
* On the M2-Max the slowest `ik_llama.cpp` type outperforms all `llama.cpp` types except `Q4_0` and `IQ4_NL`.
  
### Prompt processing (prefill) champion

The fastest way to do prompt processing with `ik_llama.cpp` is the new 8-bit, 8-row interleaved `Q8_K_R8` type. Getting 370 t/s for LLaMA-3.1-8B (~7.5 billion parameters excluding token embeddings) corresponds to ~5.5 TFLOPS!

| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| llama 8B Q8_K_R8               |   7.56 GiB |     8.03 B | Zen4        |      16 |         pp512 |    370.11 ± 0.58 |
 llama 8B Q8_K_R8               |   7.56 GiB |     8.03 B | ARM_NEON      |   8 |         pp512 |    170.68 ± 0.56 |

---

#### 🗣️ Discussion

👤 **saood06** replied the **2025-01-10** at **23:34:54**:<br>

I ran some benchmarks on an AVX2 machine (Xeon E5-2683 v4, 32 core, quad channel broadwell) on an IQ4_XS of Midnight Miqu 70B v1.5 via batched bench ( with arguments -pps -fa -t 32 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32 -c 32768 [context only needed to be set for llama.cpp as otherwise it would skip some tests but ik_llama.cpp defaulted to 32768] ), build 4404 for llama.cpp. No runtime repacking for ik_llama.cpp.
I was curious about batch performance since there is inference software like arrows or loom which would definitely benefit from it.

| PP      | TG       | B      | N_KV     | T_TG s (llama.cpp)    | S_TG t/s (llama.cpp)  | T_TG s (ik_llama.cpp)    | S_TG t/s (ik_llama.cpp)  | Speedup |
|---------|----------|--------|----------|-----------------------|-----------------------|--------------------------|--------------------------|---------|
| 128     | 128      | 1      | 256      | 92.1                  | 1.39                  | 90.247                   | 1.42                     | 1.02    |
| 128     | 128      | 2      | 384      | 115.871               | 2.21                  | 93.563                   | 2.74                     | 1.24    |
| 128     | 128      | 4      | 640      | 209.851               | 2.44                  | 111.702                  | 4.58                     | 1.88    |
| 128     | 128      | 8      | 1152     | 399.978               | 2.56                  | 209.249                  | 4.89                     | 1.91    |
| 128     | 128      | 16     | 2176     | 783.003               | 2.62                  | 427.421                  | 4.79                     | 1.83    |
| 128     | 128      | 32     | 4224     | 1556.121              | 2.63                  | 896.142                  | 4.57                     | 1.74    |
| 128     | 256      | 1      | 384      | 184.753               | 1.39                  | 181.031                  | 1.41                     | 1.02    |
| 128     | 256      | 2      | 640      | 233.044               | 2.2                   | 185.192                  | 2.76                     | 1.26    |
| 128     | 256      | 4      | 1152     | 423.01                | 2.42                  | 227.289                  | 4.51                     | 1.86    |
| 128     | 256      | 8      | 2176     | 807.7                 | 2.54                  | 434.213                  | 4.72                     | 1.86    |
| 128     | 256      | 16     | 4224     | 1578.773              | 2.59                  | 908.93                   | 4.51                     | 1.74    |
| 128     | 256      | 32     | 8320     | 3143.512              | 2.61                  | 2024.429                 | 4.05                     | 1.55    |
| 256     | 128      | 1      | 384      | 92.622                | 1.38                  | 90.92                    | 1.41                     | 1.02    |
| 256     | 128      | 2      | 512      | 118.038               | 2.17                  | 92.551                   | 2.77                     | 1.28    |
| 256     | 128      | 4      | 768      | 212.751               | 2.41                  | 113.572                  | 4.51                     | 1.87    |
| 256     | 128      | 8      | 1280     | 404.917               | 2.53                  | 211.062                  | 4.85                     | 1.92    |
| 256     | 128      | 16     | 2304     | 789.767               | 2.59                  | 428.125                  | 4.78                     | 1.84    |
| 256     | 128      | 32     | 4352     | 1569.485              | 2.61                  | 899.613                  | 4.55                     | 1.74    |
| 256     | 256      | 1      | 512      | 186.991               | 1.37                  | 181.844                  | 1.41                     | 1.03    |
| 256     | 256      | 2      | 768      | 237.34                | 2.16                  | 186.438                  | 2.75                     | 1.27    |
| 256     | 256      | 4      | 1280     | 428.1                 | 2.39                  | 229.219                  | 4.47                     | 1.87    |
| 256     | 256      | 8      | 2304     | 815.064               | 2.51                  | 437.482                  | 4.68                     | 1.86    |
| 256     | 256      | 16     | 4352     | 1591.762              | 2.57                  | 911.641                  | 4.49                     | 1.75    |
| 256     | 256      | 32     | 8448     | 3170.023              | 2.58                  | 2058.671                 | 3.98                     | 1.54    |
| 512     | 128      | 1      | 640      | 93.876                | 1.36                  | 92.345                   | 1.39                     | 1.02    |
| 512     | 128      | 2      | 768      | 118.683               | 2.16                  | 93.867                   | 2.73                     | 1.26    |
| 512     | 128      | 4      | 1024     | 215.082               | 2.38                  | 114.616                  | 4.47                     | 1.88    |
| 512     | 128      | 8      | 1536     | 411.704               | 2.49                  | 215.892                  | 4.74                     | 1.91    |
| 512     | 128      | 16     | 2560     | 803.455               | 2.55                  | 439.992                  | 4.65                     | 1.83    |
| 512     | 128      | 32     | 4608     | 1595.727              | 2.57                  | 928.049                  | 4.41                     | 1.72    |
| 512     | 256      | 1      | 768      | 188.209               | 1.36                  | 183.237                  | 1.4                      | 1.03    |
| 512     | 256      | 2      | 1024     | 238.668               | 2.15                  | 191.19                   | 2.68                     | 1.25    |
| 512     | 256      | 4      | 1536     | 435.484               | 2.35                  | 233.338                  | 4.39                     | 1.87    |
| 512     | 256      | 8      | 2560     | 828.696               | 2.47                  | 443.92                   | 4.61                     | 1.87    |
| 512     | 256      | 16     | 4608     | 1618.7                | 2.53                  | 927.963                  | 4.41                     | 1.74    |
| 512     | 256      | 32     | 8704     | 3222.905              | 2.54                  | 2082.961                 | 3.93                     | 1.55    |

The table does not have PP results as they did not vary much between tests since the prompt is shared as that is more aligned with my usecase, but even then ik_llama.cpp was faster (~5.05 t/s vs ~2.70 t/s).

 I manually repacked it from the IQ4_XS and tested the R4 version of the quant on ik_llama.cpp more thoroughly results below.

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |    128 |    1 |    256 |   19.497 |     6.56 |   92.423 |     1.38 |  111.921 |     2.29 |
|   128 |    128 |    2 |    384 |   19.332 |     6.62 |   92.578 |     2.77 |  111.910 |     3.43 |
|   128 |    128 |    3 |    512 |   19.325 |     6.62 |   94.344 |     4.07 |  113.669 |     4.50 |
|   128 |    128 |    4 |    640 |   19.342 |     6.62 |   96.776 |     5.29 |  116.119 |     5.51 |
|   128 |    128 |    5 |    768 |   19.345 |     6.62 |  106.289 |     6.02 |  125.634 |     6.11 |
|   128 |    128 |    6 |    896 |   19.358 |     6.61 |  124.053 |     6.19 |  143.412 |     6.25 |
|   128 |    128 |    7 |   1024 |   19.344 |     6.62 |  145.853 |     6.14 |  165.197 |     6.20 |
|   128 |    128 |    8 |   1152 |   19.374 |     6.61 |  169.257 |     6.05 |  188.631 |     6.11 |
|   128 |    128 |    9 |   1280 |   19.340 |     6.62 |  188.213 |     6.12 |  207.553 |     6.17 |
|   128 |    128 |   10 |   1408 |   19.354 |     6.61 |  210.678 |     6.08 |  230.033 |     6.12 |
|   128 |    128 |   11 |   1536 |   19.349 |     6.62 |  219.492 |     6.41 |  238.841 |     6.43 |
|   128 |    128 |   12 |   1664 |   19.341 |     6.62 |  251.357 |     6.11 |  270.697 |     6.15 |
|   128 |    128 |   13 |   1792 |   19.341 |     6.62 |  258.946 |     6.43 |  278.287 |     6.44 |
|   128 |    128 |   14 |   1920 |   19.355 |     6.61 |  299.999 |     5.97 |  319.354 |     6.01 |
|   128 |    128 |   15 |   2048 |   19.345 |     6.62 |  302.160 |     6.35 |  321.505 |     6.37 |
|   128 |    128 |   16 |   2176 |   19.362 |     6.61 |  339.064 |     6.04 |  358.426 |     6.07 |
|   128 |    256 |    1 |    384 |   19.365 |     6.61 |  180.876 |     1.42 |  200.241 |     1.92 |
|   128 |    256 |    2 |    640 |   19.382 |     6.60 |  189.188 |     2.71 |  208.570 |     3.07 |
|   128 |    256 |    3 |    896 |   19.359 |     6.61 |  191.263 |     4.02 |  210.621 |     4.25 |
|   128 |    256 |    4 |   1152 |   19.372 |     6.61 |  197.427 |     5.19 |  216.798 |     5.31 |
|   128 |    256 |    5 |   1408 |   19.373 |     6.61 |  219.152 |     5.84 |  238.525 |     5.90 |
|   128 |    256 |    6 |   1664 |   19.370 |     6.61 |  258.357 |     5.95 |  277.727 |     5.99 |
|   128 |    256 |    7 |   1920 |   19.370 |     6.61 |  303.584 |     5.90 |  322.954 |     5.95 |
|   128 |    256 |    8 |   2176 |   19.372 |     6.61 |  349.893 |     5.85 |  369.265 |     5.89 |
|   128 |    256 |    9 |   2432 |   19.327 |     6.62 |  386.352 |     5.96 |  405.680 |     5.99 |
|   128 |    256 |   10 |   2688 |   19.337 |     6.62 |  444.917 |     5.75 |  464.255 |     5.79 |
|   128 |    256 |   11 |   2944 |   19.341 |     6.62 |  451.427 |     6.24 |  470.768 |     6.25 |
|   128 |    256 |   12 |   3200 |   19.345 |     6.62 |  528.326 |     5.81 |  547.671 |     5.84 |
|   128 |    256 |   13 |   3456 |   19.546 |     6.55 |  532.030 |     6.26 |  551.576 |     6.27 |
|   128 |    256 |   14 |   3712 |   19.333 |     6.62 |  646.512 |     5.54 |  665.845 |     5.57 |
|   128 |    256 |   15 |   3968 |   19.335 |     6.62 |  619.687 |     6.20 |  639.021 |     6.21 |
|   128 |    256 |   16 |   4224 |   19.328 |     6.62 |  732.538 |     5.59 |  751.866 |     5.62 |
|   256 |    128 |    1 |    384 |   38.431 |     6.66 |   92.778 |     1.38 |  131.209 |     2.93 |
|   256 |    128 |    2 |    512 |   38.513 |     6.65 |   93.080 |     2.75 |  131.592 |     3.89 |
|   256 |    128 |    3 |    640 |   38.412 |     6.66 |   95.364 |     4.03 |  133.776 |     4.78 |
|   256 |    128 |    4 |    768 |   38.417 |     6.66 |   98.235 |     5.21 |  136.652 |     5.62 |
|   256 |    128 |    5 |    896 |   38.448 |     6.66 |  107.889 |     5.93 |  146.337 |     6.12 |
|   256 |    128 |    6 |   1024 |   38.443 |     6.66 |  125.778 |     6.11 |  164.221 |     6.24 |
|   256 |    128 |    7 |   1152 |   38.437 |     6.66 |  149.730 |     5.98 |  188.167 |     6.12 |
|   256 |    128 |    8 |   1280 |   38.462 |     6.66 |  170.487 |     6.01 |  208.949 |     6.13 |
|   256 |    128 |    9 |   1408 |   38.433 |     6.66 |  189.718 |     6.07 |  228.151 |     6.17 |
|   256 |    128 |   10 |   1536 |   38.438 |     6.66 |  213.574 |     5.99 |  252.011 |     6.09 |
|   256 |    128 |   11 |   1664 |   38.455 |     6.66 |  222.606 |     6.33 |  261.061 |     6.37 |
|   256 |    128 |   12 |   1792 |   38.445 |     6.66 |  252.863 |     6.07 |  291.308 |     6.15 |
|   256 |    128 |   13 |   1920 |   38.443 |     6.66 |  260.814 |     6.38 |  299.257 |     6.42 |
|   256 |    128 |   14 |   2048 |   38.438 |     6.66 |  305.763 |     5.86 |  344.202 |     5.95 |
|   256 |    128 |   15 |   2176 |   38.475 |     6.65 |  303.104 |     6.33 |  341.579 |     6.37 |
|   256 |    128 |   16 |   2304 |   38.469 |     6.65 |  342.793 |     5.97 |  381.262 |     6.04 |
|   256 |    256 |    1 |    512 |   38.455 |     6.66 |  183.865 |     1.39 |  222.320 |     2.30 |
|   256 |    256 |    2 |    768 |   38.479 |     6.65 |  187.584 |     2.73 |  226.063 |     3.40 |
|   256 |    256 |    3 |   1024 |   38.463 |     6.66 |  192.895 |     3.98 |  231.358 |     4.43 |
|   256 |    256 |    4 |   1280 |   38.399 |     6.67 |  199.713 |     5.13 |  238.111 |     5.38 |
|   256 |    256 |    5 |   1536 |   38.439 |     6.66 |  223.437 |     5.73 |  261.875 |     5.87 |
|   256 |    256 |    6 |   1792 |   38.427 |     6.66 |  260.056 |     5.91 |  298.482 |     6.00 |
|   256 |    256 |    7 |   2048 |   38.398 |     6.67 |  307.312 |     5.83 |  345.710 |     5.92 |
|   256 |    256 |    8 |   2304 |   38.415 |     6.66 |  355.564 |     5.76 |  393.979 |     5.85 |
|   256 |    256 |    9 |   2560 |   38.497 |     6.65 |  387.482 |     5.95 |  425.979 |     6.01 |
|   256 |    256 |   10 |   2816 |   38.498 |     6.65 |  451.367 |     5.67 |  489.865 |     5.75 |
|   256 |    256 |   11 |   3072 |   38.493 |     6.65 |  452.656 |     6.22 |  491.149 |     6.25 |
|   256 |    256 |   12 |   3328 |   38.669 |     6.62 |  534.248 |     5.75 |  572.917 |     5.81 |
|   256 |    256 |   13 |   3584 |   38.485 |     6.65 |  534.845 |     6.22 |  573.330 |     6.25 |
|   256 |    256 |   14 |   3840 |   38.486 |     6.65 |  649.772 |     5.52 |  688.257 |     5.58 |
|   256 |    256 |   15 |   4096 |   39.294 |     6.51 |  624.510 |     6.15 |  663.804 |     6.17 |
|   256 |    256 |   16 |   4352 |   38.648 |     6.62 |  745.863 |     5.49 |  784.511 |     5.55 |
|   512 |    128 |    1 |    640 |   77.207 |     6.63 |   91.468 |     1.40 |  168.674 |     3.79 |
|   512 |    128 |    2 |    768 |   76.844 |     6.66 |   94.375 |     2.71 |  171.219 |     4.49 |
|   512 |    128 |    3 |    896 |   77.835 |     6.58 |   97.286 |     3.95 |  175.120 |     5.12 |
|   512 |    128 |    4 |   1024 |   76.964 |     6.65 |  100.195 |     5.11 |  177.159 |     5.78 |
|   512 |    128 |    5 |   1152 |   76.998 |     6.65 |  110.516 |     5.79 |  187.514 |     6.14 |
|   512 |    128 |    6 |   1280 |   77.134 |     6.64 |  128.599 |     5.97 |  205.733 |     6.22 |
|   512 |    128 |    7 |   1408 |   77.085 |     6.64 |  153.659 |     5.83 |  230.744 |     6.10 |
|   512 |    128 |    8 |   1536 |   77.157 |     6.64 |  174.060 |     5.88 |  251.217 |     6.11 |
|   512 |    128 |    9 |   1664 |   77.074 |     6.64 |  192.851 |     5.97 |  269.925 |     6.16 |
|   512 |    128 |   10 |   1792 |   77.079 |     6.64 |  219.608 |     5.83 |  296.688 |     6.04 |
|   512 |    128 |   11 |   1920 |   78.024 |     6.56 |  224.332 |     6.28 |  302.356 |     6.35 |
|   512 |    128 |   12 |   2048 |   77.056 |     6.64 |  258.370 |     5.94 |  335.426 |     6.11 |
|   512 |    128 |   13 |   2176 |   76.931 |     6.66 |  264.692 |     6.29 |  341.624 |     6.37 |
|   512 |    128 |   14 |   2304 |   77.061 |     6.64 |  310.472 |     5.77 |  387.533 |     5.95 |
|   512 |    128 |   15 |   2432 |   77.067 |     6.64 |  305.914 |     6.28 |  382.981 |     6.35 |
|   512 |    128 |   16 |   2560 |   77.067 |     6.64 |  352.858 |     5.80 |  429.925 |     5.95 |
|   512 |    256 |    1 |    768 |   77.023 |     6.65 |  183.489 |     1.40 |  260.512 |     2.95 |
|   512 |    256 |    2 |   1024 |   77.015 |     6.65 |  190.038 |     2.69 |  267.052 |     3.83 |
|   512 |    256 |    3 |   1280 |   77.911 |     6.57 |  196.900 |     3.90 |  274.811 |     4.66 |
|   512 |    256 |    4 |   1536 |   76.980 |     6.65 |  204.269 |     5.01 |  281.249 |     5.46 |
|   512 |    256 |    5 |   1792 |   76.875 |     6.66 |  226.576 |     5.65 |  303.451 |     5.91 |
|   512 |    256 |    6 |   2048 |   77.435 |     6.61 |  267.788 |     5.74 |  345.223 |     5.93 |
|   512 |    256 |    7 |   2304 |   76.984 |     6.65 |  315.387 |     5.68 |  392.370 |     5.87 |
|   512 |    256 |    8 |   2560 |   76.968 |     6.65 |  362.447 |     5.65 |  439.416 |     5.83 |
|   512 |    256 |    9 |   2816 |   76.947 |     6.65 |  393.626 |     5.85 |  470.573 |     5.98 |
|   512 |    256 |   10 |   3072 |   76.959 |     6.65 |  463.783 |     5.52 |  540.742 |     5.68 |
|   512 |    256 |   11 |   3328 |   76.890 |     6.66 |  458.811 |     6.14 |  535.701 |     6.21 |
|   512 |    256 |   12 |   3584 |   77.875 |     6.57 |  544.833 |     5.64 |  622.708 |     5.76 |
|   512 |    256 |   13 |   3840 |   77.002 |     6.65 |  542.172 |     6.14 |  619.174 |     6.20 |
|   512 |    256 |   14 |   4096 |   77.088 |     6.64 |  668.595 |     5.36 |  745.683 |     5.49 |
|   512 |    256 |   15 |   4352 |   77.021 |     6.65 |  629.146 |     6.10 |  706.168 |     6.16 |
|   512 |    256 |   16 |   4608 |   78.044 |     6.56 |  758.943 |     5.40 |  836.987 |     5.51 |

Performance is good, but I don't understand why odd batch sizes seem to perform better. Also is converting from IQ4_XS to IQ4_XS_R4 via the quantize command not reccomended? I did it just for the test above and it went from:
type  f32:  161 tensors
type q5_K:   80 tensors
type q6_K:    1 tensors
type iq4_xs:  481 tensors

And after conversion:
type  f32:  161 tensors
type q5_K:   10 tensors
type q6_K:    1 tensors
type iq4_xs:    1 tensors
type iq5_k:   80 tensors
type iq4_xs_r4:  470 tensors

I only ask because I'm not sure if the 80 tensors going from q5_K to iq5_k is lossy.

---

👤 **ikawrakow** replied the **2025-01-11** at **07:28:46**:<br>

@saood06 Thanks for testing.

> Performance is good, but I don't understand why odd batch sizes seem to perform better. 

Neither do I. I'll have to look into it.

> Also is converting from IQ4_XS to IQ4_XS_R4 via the quantize command not reccomended? I did it just for the test above and it went from:

Sorry, the goal was to make the `_R4` quants use the same quantization mixes, but apparently I have not quite succeeded. The function where the quantization type is selected is quite messy. But instead of re-quantizing to `*_R4`, you can use the `-rtr` command line option, which will make your model use the exact same mix of quantization types (but those where an `_R4` variant is available will be repacked to that).

>  I only ask because I'm not sure if the 80 tensors going from q5_K to iq5_k is lossy.

`IQ5_K` is normally quite a bit better than `Q5_K`, so most of the time I would expect this to perform better.

> 👤 **saood06** replied the **2025-01-11** at **09:59:16**:<br>
> >Sorry, the goal was to make the _R4 quants use the same quantization mixes, but apparently I have not quite succeeded. The function where the quantization type is selected is quite messy. But instead of re-quantizing to *_R4, you can use the -rtr command line option, which will make your model use the exact same mix of quantization types (but those where an _R4 variant is available will be repacked to that).
> 
> No worries, I only made the quant to test (for actual use, I'd make an IQK quant) and I didn't realize batched-bench supported rtr. It also didn't matter for this machine and test, but I also wasn't sure how runtime repacking and NUMA would behave, if the runtime repacking would interfere with the benefits from POSIX_MADV_RANDOM.
> 
> >IQ5_K is normally quite a bit better than Q5_K, so most of the time I would expect this to perform better.
> 
> Yes, but if the tensor was originally Q5_K converting it can't recover accuracy, it can only maintain it or lose more.
> 
> On another note, I also got Deepseek V3 working with ik_llama.cpp. I don't have direct comparisons to llama.cpp ( and I don't know if I will, making a quant takes 4 hours) but running IQ4_K ( on different hardware then the Midnight Miqu test above, this one is a dual socket Xeon E5-2690 v3). Indirectly comparing to what people were posting on reddit with either machine's that were far better than mine, or quants that were smaller the performance I have seems a lot better. The only thing is this model based on both my experience and some issues made on llama.cpp takes a LOT of tokens to get fully faulted into RAM, which might be why people were posting such low performance numbers.
> 
> Once almost all the model is in system cache, it did Prompt processing at 11.5 t/s, and token generation at 2.75 t/s. I still couldn't get it to fully fault, but it did basically stop paging, and performance stopped improving, once it hit those numbers.
> 
> I couldn't get it to run with an _R4 quant it hit the GGML_ASSERT(nrc_x%4 == 0), but even without that I'm still happy with the performance of it.
> 
> 👤 **ikawrakow** replied the **2025-01-11** at **10:38:23**:<br>
> > I couldn't get it to run with an _R4 quant it hit the GGML_ASSERT(nrc_x%4 == 0), but even without that I'm still happy with the performance of it.
> 
> Can you post the assert you see? I was hoping to have covered all places where one needs to check for divisibility by 4 before using `_R4` quants, but apparently I'm still missing checks somewhere. What are the tensor dimensions of this model?
> 
> 👤 **saood06** replied the **2025-01-11** at **11:03:54**:<br>
> >Can you post the assert you see?
> 
> Here's the full error output I got when trying to run it. I put it in a detail's thing as it is long.
> 
> <details>
> 
> ```
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: 
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: 
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> 
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> 
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> 
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242: GGML_ASSERT(nrc_x%4 == 0) failed
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.ptrace: Operation not permitted.ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> 
> ptrace: Operation not permitted.ptrace: Operation not permitted.ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.ptrace: Operation not permitted.ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.ptrace: Operation not permitted.ptrace: Operation not permitted.
> 
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.ptrace: Operation not permitted.ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> warning: process 2173336 is already traced by process 2173436
> 
> ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.
> ptrace: Operation not permitted.
> ptrace: Operation not permitted.
> 
> 
> ptrace: Operation not permitted.
> 
> ptrace: Operation not permitted.
> 
> 
> ptrace: Operation not permitted.
> 
> ptrace: Operation not permitted.
> 
> 
> No stack.No stack.ptrace: Operation not permitted.
> 
> 
> 
> ptrace: Operation not permitted.ptrace: Operation not permitted.ptrace: Operation not permitted.
> 
> 
> ptrace: Operation not permitted.ptrace: Operation not permitted.
> No stack.No stack.ptrace: Operation not permitted.
> 
> 
> 
> No stack.No stack.No stack.No stack.
> 
> No stack.No stack.No stack.No stack.
> No stack.No stack.No stack.No stack.
> 
> 
> No stack.The program is not being run.
> No stack.
> No stack.
> No stack.
> 
> No stack.No stack.No stack.No stack.No stack.
> The program is not being run.
> 
> 
> 
> No stack.
> 
> 
> No stack.
> No stack.
> 
> 
> No stack.
> No stack.
> 
> No stack.
> No stack.
> 
> 
> No stack.
> 
> 
> 
> 
> 
> No stack.The program is not being run.No stack.No stack.No stack.
> No stack.
> The program is not being run.No stack.
> The program is not being run.The program is not being run.The program is not being run.
> The program is not being run.The program is not being run.
> The program is not being run.
> The program is not being run.The program is not being run.The program is not being run.The program is not being run.
> The program is not being run.The program is not being run.The program is not being run.The program is not being run.The program is not being run.
> 
> The program is not being run.
> The program is not being run.The program is not being run.The program is not being run.The program is not being run.
> The program is not being run.The program is not being run.
> 
> 
> The program is not being run.
> 
> 
> The program is not being run.The program is not being run.
> 
> The program is not being run.
> 
> The program is not being run.
> 
> 
> The program is not being run.
> The program is not being run.
> 
> 
> 
> The program is not being run.
> 
> 
> 
> 
> The program is not being run.
> 
> The program is not being run.
> 
> 
> The program is not being run.The program is not being run.
> 
> 
> 
> 
> 
> 
> 
> 
> warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.
> warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> 
> warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.
> warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.
> ptrace: Operation not permitted.warning: process 2173336 is already traced by process 2173436
> 
> 
> warning: process 2173336 is already traced by process 2173436
> ptrace: Operation not permitted.No stack.No stack.
> 
> 
> ptrace: Operation not permitted.
> 
> No stack.
> No stack.The program is not being run.The program is not being run.
> 
> 
> The program is not being run.No stack.No stack.
> 
> The program is not being run.
> The program is not being run.
> 
> No stack.The program is not being run.No stack.
> The program is not being run.
> 
> 
> The program is not being run.
> The program is not being run.
> [New LWP 2173387]
> [New LWP 2173386]
> [New LWP 2173385]
> [New LWP 2173384]
> [New LWP 2173383]
> [New LWP 2173382]
> [New LWP 2173381]
> [New LWP 2173380]
> [New LWP 2173379]
> [New LWP 2173378]
> [New LWP 2173377]
> [New LWP 2173376]
> [New LWP 2173375]
> [New LWP 2173374]
> [New LWP 2173373]
> [New LWP 2173372]
> [New LWP 2173371]
> [New LWP 2173370]
> [New LWP 2173369]
> [New LWP 2173368]
> [New LWP 2173367]
> [New LWP 2173366]
> [New LWP 2173365]
> [New LWP 2173364]
> [New LWP 2173363]
> [New LWP 2173362]
> [New LWP 2173361]
> [New LWP 2173360]
> [New LWP 2173359]
> [New LWP 2173358]
> [New LWP 2173357]
> [New LWP 2173356]
> [New LWP 2173355]
> [New LWP 2173354]
> [New LWP 2173353]
> [New LWP 2173352]
> [New LWP 2173351]
> [New LWP 2173350]
> [New LWP 2173349]
> [New LWP 2173348]
> [New LWP 2173347]
> [New LWP 2173346]
> [New LWP 2173345]
> [New LWP 2173344]
> [New LWP 2173343]
> [New LWP 2173342]
> [New LWP 2173341]
> [Thread debugging using libthread_db enabled]
> Using host libthread_db library "/usr/lib64/libthread_db.so.1".
> 0x000055770a10e177 in __GI___wait4 () at ../sysdeps/unix/sysv/linux/wait4.c:30
> warning: 30     ../sysdeps/unix/sysv/linux/wait4.c: No such file or directory
> #0  0x000055770a10e177 in __GI___wait4 () at ../sysdeps/unix/sysv/linux/wait4.c:30
> 30      in ../sysdeps/unix/sysv/linux/wait4.c
> #1  0x000055770a817f7a in ggml_print_backtrace () at /home/saood06/ik_llama.cpp/ggml/src/ggml.c:241
> 241             waitpid(pid, &wstatus, 0);
> #2  0x000055770a840bc8 in ggml_abort (file=0x55770abb91f0 "/home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp", line=5242, fmt=0x55770abb4051 "GGML_ASSERT(%s) failed") at /home/saood06/ik_llama.cpp/ggml/src/ggml.c:268
> 268         ggml_print_backtrace();
> #3  0x000055770aa0814a in (anonymous namespace)::mul_mat_iq4_k_r4_q8_k<1> (n=<optimized out>, vx=<optimized out>, bx=<optimized out>, info=..., nrc_x=<optimized out>) at /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:5242
> 5242        GGML_ASSERT(nrc_x%4 == 0);
> #4  0x000055770ab7454c in (anonymous namespace)::MulMat::mul_mat_NxM (this=0x7ffe16539de0, n=7168, vx=0x551fe175a500, bx=<optimized out>, info=..., nrc_x=<optimized out>, nrc_y=7168) at /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:183
> 183                 funcs[n_left-1](n, vx, bx, info, nrc_x);
> #5  (anonymous namespace)::MulMat::mul_mat_NxM (this=0x7ffe16539de0, n=7168, vx=0x551fe175a500, bx=<optimized out>, info=..., nrc_x=6, nrc_y=7168) at /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:123
> 123         inline void mul_mat_NxM(int n, const void * vx, size_t bx, DataInfo& info, int nrc_x, int nrc_y) {
> #6  iqk_mul_mat_moe (Nx=Nx@entry=2048, Ny=Ny@entry=1, ne00=ne00@entry=7168, ne11=ne11@entry=1, typeA=<optimized out>, A=A@entry=0x551fe175a500, strideA=<optimized out>, typeB=15, B=0x55770ff8ef60, strideB=8176, C=0x551d8392b820, nb1=8192, nb2=655            36, vrow_mapping=0x55770ff937e0, ith=0, nth=48) at /home/saood06/ik_llama.cpp/ggml/src/iqk/iqk_mul_mat.cpp:265
> 265         mm.mul_mat_NxM(ne00, (const char *)A + row_size_qx*first_x, row_size_qx, info, nrc_x, Ny);
> #7  0x000055770a82e9a5 in ggml_compute_forward_mul_mat_id (params=<optimized out>, dst=0x557709930930) at /home/saood06/ik_llama.cpp/ggml/src/ggml.c:14281
> 14281              if (!iqk_mul_mat_moe(nr0, nr1, ne00, ne11,
> #8  0x000055770a85c1e7 in ggml_graph_compute_thread (data=data@entry=0x7ffe1653a150) at /home/saood06/ik_llama.cpp/ggml/src/ggml.c:21029
> 21029           ggml_compute_forward(&params, node);
> #9  0x000055770a85c335 in ggml_graph_compute._omp_fn.0 () at /home/saood06/ik_llama.cpp/ggml/src/ggml.c:21080
> 21080               ggml_graph_compute_thread(&worker);
> #10 0x000055770a3b7dc6 in GOMP_parallel () from /usr/lib64/libgomp.so.1
> #11 0x000055770a85f984 in ggml_graph_compute (cgraph=cgraph@entry=0x55770fdda578, cplan=cplan@entry=0x7ffe1653a230) at /home/saood06/ik_llama.cpp/ggml/src/ggml.c:21066
> 21066           #pragma omp parallel num_threads(n_threads)
> #12 0x000055770a86f272 in ggml_backend_cpu_graph_compute (backend=<optimized out>, cgraph=0x55770fdda578) at /home/saood06/ik_llama.cpp/ggml/src/ggml-backend.c:815
> 815         return ggml_graph_compute(cgraph, &cplan);
> #13 0x000055770a872f7a in ggml_backend_graph_compute_async (backend=0x5577104efd20, cgraph=0x55770fdda578) at /home/saood06/ik_llama.cpp/ggml/src/ggml-backend.c:282
> 282         return backend->iface.graph_compute(backend, cgraph);
> #14 ggml_backend_sched_compute_splits (sched=0x55770ff4a860) at /home/saood06/ik_llama.cpp/ggml/src/ggml-backend.c:1795
> 1795                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
> #15 0x000055770ad9d036 in llama_graph_compute (lctx=..., gf=0x5577098df030, n_threads=48) at /home/saood06/ik_llama.cpp/src/llama.cpp:14917
> 14917       ggml_backend_sched_graph_compute_async(lctx.sched, gf);
> #16 llama_decode_internal (batch_all=..., lctx=...) at /home/saood06/ik_llama.cpp/src/llama.cpp:15133
> 15133           llama_graph_compute(lctx, gf, n_threads);
> #17 llama_decode (ctx=0x55770fde9e00, batch=...) at /home/saood06/ik_llama.cpp/src/llama.cpp:19318
> 19318       const int ret = llama_decode_internal(*ctx, batch);
> #18 0x000055770ae99991 in llama_init_from_gpt_params (params=...) at /home/saood06/ik_llama.cpp/common/common.cpp:2179
> 2179                llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch), 0, 0));
> #19 0x000055770ae6bbac in main (argc=<optimized out>, argv=<optimized out>) at /home/saood06/ik_llama.cpp/examples/main/main.cpp:210
> 210         llama_init_result llama_init = llama_init_from_gpt_params(params);
> Aborted (core dumped)
> [Inferior 1 (process 2173336) detached]
> 
> ```
> </details>
> 
> >What are the tensor dimensions of this model?
> 
> https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q2_K_L?show_file_info=DeepSeek-V3-Q2_K_L%2FDeepSeek-V3-Q2_K_L-00001-of-00005.gguf
> 
> That link should list them in a relatively nice format. You'll have to click through to view all 5 parts though.
> 
> 👤 **ikawrakow** replied the **2025-01-11** at **11:17:30**:<br>
> Thanks! This explains it. It is a MoE model, so I must have forgotten to make sure the number of rows is a multiple of 4 when splitting work between threads in the MoE matrix multiplication implementation. I'll try to fix it.
> 
> 👤 **saood06** replied the **2025-01-12** at **18:08:54**:<br>
> >Thanks! This explains it.
> 
> I'm glad you were able to figure out the issue.
> 
> >I'll try to fix it.
> 
> I see you did with #170, now the _R4 works for Deepseek V3 but performance is different from what I was expecting. I am pleasantly surprised by token generation going from 2.75 t/s to 3.10 t/s. Prompt processing on the other hand dropped from 11.5 t/s to 9.8 t/s.
> 
> Either way thanks for the quick fix. The bump in TG speeds is nice, even if PP speed went down for me.
> 
> 👤 **ikawrakow** replied the **2025-01-13** at **05:54:15**:<br>
> > Prompt processing on the other hand dropped from 11.5 t/s to 9.8 t/s.
> 
> This is strange. In my testing with Mixtral8x7B, after the fix `IQ4_XS_R4` is about 30% faster than `IQ4_XS` for prompt processing. Deepseek V3 is beyond my compute capabilities, so not able to investigate.
> 
> 👤 **saood06** replied the **2025-01-19** at **13:00:33**:<br>
> >after the fix IQ4_XS_R4 is about 30% faster than IQ4_XS for prompt processing
> 
> I've been testing IQ4_K_R4 vs IQ4_K. but I will test both IQ4_XS some for Mixtral-8x22B as I plan to test that, and I'll give some numbers against llama.cpp.
> 
> >Deepseek V3 is beyond my compute capabilities, so not able to investigate.
> 
> I understand, it is a large model and why I have yet to test IQ4_XS, to compare against both in ppl, and also against llama.cpp. But even if you can't test the implementation, I got permission from the author of the Deepseek PR to create a PR here, would you accept it.

---

👤 **ikawrakow** replied the **2025-01-11** at **07:58:35**:<br>

> > Performance is good, but I don't understand why odd batch sizes seem to perform better. 

> Neither do I. I'll have to look into it.

It is related to flash attention (FA). Here is a graph that shows t/s as a function of batch size with and without FA (LLaMA-3.1-8B-Instruct, Ryzen-7950X CPU)
![batches](https://github.com/user-attachments/assets/2c2e6020-4bea-41f9-9b56-f51bcfd3c61a)

Clearly I'm doing something there that works better for odd number of queries. I'll need to investigate.

---

👤 **saood06** replied the **2025-01-19** at **13:33:06**:<br>

>We see that the CPU performance gap has widened significantly since July when I made the comparison on the front page.

Do you plan to update the README.md with these numbers? The R4 quants are very impressive.

> 👤 **ikawrakow** replied the **2025-01-19** at **15:30:36**:<br>
> I should, I know. It is just that I prefer to solve problems rather that write about how I solved the problem and what came out.
> 
> 👤 **saood06** replied the **2025-04-27** at **09:33:26**:<br>
> You made a good list of things [here](https://github.com/ikawrakow/ik_llama.cpp/discussions/256#discussioncomment-12496828), the "Why?" section can be updated with newer models like the official bitnet release, Deepseek, Llama-4. Updating the benchmarks though I know is a lot.
> 
> 👤 **ikawrakow** replied the **2025-04-28** at **14:29:33**:<br>
> Something like PR #352 ?

---

👤 **bartowski1182** replied the **2025-01-23** at **02:58:19**:<br>

Out of curiousity, do you intend to maintain this fork as an alternative to llama.cpp perpetually? or is it more of a testing grounds before upstreaming?

wondering if it's worth recommending people run this specifically for better performance or if it's more of a "bleeding edge" kind of project that people should just wait to get later when it's more ready

> 👤 **ikawrakow** replied the **2025-01-23** at **08:18:58**:<br>
> > Out of curiousity, do you intend to maintain this fork as an alternative to llama.cpp perpetually? or is it more of a testing grounds before upstreaming?
> 
> Nothing is perpetual in this world :smiley: 
> 
> But no, I have no intention to be upstreaming to `llama.cpp`. 
> 
> It is also a bit of a chicken and egg game: I'll only get a more significant number of users if people know (or at least expect) that I'm seriously committed to his project and the project gets advertised around social networks, but I can only know if I want to seriously commit to maintaining this project long term for a significant number of users if I already have many users and have dealt with the associated bug reports and feature requests :smiley:
> 
> As it stands, this project is only useful for technical users who are not scared to build the project themself (no docker images and pre-build binaries), and are using one of the platforms I develop/test on (Linux and macOS, `AVX2` or `ARM_NEON` CPUs, newer Nvidia GPUs). It may or may not work on Windows/Android/etc, old Nvidia or AMD GPUs, etc. I absolutely don't have the bandwidth (or desire) to be supporting every operating system and computing platform under the sun, including 10+ year old CPUs and GPUs, and obscure platforms used by exactly 3 people in the worlds, as `llama.cpp` does.
> 
> 👤 **bartowski1182** replied the **2025-01-23** at **15:12:49**:<br>
> yeah that makes sense! would be cool to see someone attempt to upstream some improvements but I understand your lack of desire considering it's probably quite the headache
> 
> Good to know though you intend to keep this going for at least awhile

---

👤 **saood06** replied the **2025-01-30** at **22:48:57**:<br>

I was curious due to Deepseek's design to test the MHA 35B c4ai-command-r-v01.Q8_0 on my Xeon E5-2683 v4. Ran as much context as I had RAM for. TG is set 5 not 32 as it was slow.

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|
|   128 |      5 |    1 |    133 |   20.344 |     6.29 |    5.500 |     0.91 |   25.843 |     5.15 |
|   256 |      5 |    1 |    261 |   34.275 |     7.47 |   30.895 |     0.16 |   65.170 |     4.00 |
|   512 |      5 |    1 |    517 |   56.097 |     9.13 |   31.850 |     0.16 |   87.947 |     5.88 |
|  1024 |      5 |    1 |   1029 |  112.460 |     9.11 |   21.224 |     0.24 |  133.684 |     7.70 |
|  2048 |      5 |    1 |   2053 |  218.188 |     9.39 |   32.941 |     0.15 |  251.130 |     8.18 |
|  4096 |      5 |    1 |   4101 |  448.955 |     9.12 |   31.231 |     0.16 |  480.186 |     8.54 |
|  8192 |      5 |    1 |   8197 |  977.908 |     8.38 |   42.563 |     0.12 | 1020.471 |     8.03 |
| 16384 |      5 |    1 |  16389 | 2339.461 |     7.00 |   39.989 |     0.13 | 2379.450 |     6.89 |
| 22000 |      5 |    1 |  22005 | 3484.923 |     6.31 |   44.705 |     0.11 | 3529.628 |     6.23 |