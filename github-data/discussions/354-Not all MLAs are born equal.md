### 🗣️ [#354](https://github.com/ikawrakow/ik_llama.cpp/discussions/354) - Not all MLAs are born equal

| **Author** | `ikawrakow` |
| :--- | :--- |
| **Created** | 2025-04-29 |
| **Updated** | 2025-05-13 |

---

#### Description

## Intro

After several attempts, they have added MLA for DeepSeek models in mainline `llama.cpp` via [this PR](https://github.com/ggml-org/llama.cpp/pull/12801), and I was curious to see how it performs.  They have of course made it maximally painful - one needs to re-download and re-convert the model to be able to take advantage of the MLA feature. Fortunately for me, on my hardware I can only run DeepSeek-Lite, i.e., a 32 GB download, so not too bad (but in comparison, `ik_llama.cpp` allows usage of MLA with an original DeepSeek GGUF as the tensors necessary for MLA get created on-the-fly). Anyway, I'm on a 300 Mb/s connection, so 15 minutes later I'm up and running.

What is the TL;DR? As the title already said - not all MLAs are born equal.

## Setup

I'll be using a `Q4_0` quantized DeepSeek-Lite model for all comparison. `Q4_0` is the fastest quantization type in mainline due to the extraordinary amount of attention it receives. GPU performance measurements are done on an RTX-4080 GPU. CPU performance is measured on a Ryzen-7950X CPU (and the RTX-4080 is in the Ryzen-7950X rig). 

## CUDA performance

I was most curious about CUDA performance. Why? Because [in this PR](https://github.com/ggml-org/llama.cpp/pull/13014) @JohannesGaessler has completely independently, without [ever looking at ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp/pull/283/files/7f6980fa5166d029ad04cef395d2993ddc8da307#r2029830357), discovered [this optimization](https://github.com/ikawrakow/ik_llama.cpp/pull/248) in `ik_llama.cpp`, so I wanted to know how the two implementations compare. Mainline does not support Flash Attention (FA) for DeepSeek on CUDA (due to K- and V-head sizes being different). `ik_llama.cpp` uses FlashMLA-2.  

This graph shows CUDA TG performance as a function of `N_KV`, the number of tokens in the KV cache. For `N_KV = 0`, mainline is now about 15% faster than `ik_llama.cpp`. This can be due to the fact that @JohannesGaessler  is a much better GPU programmer than I'm, so has achieved a more optimized implementation. However, looking at the comments and performance measurements in [the PR](https://github.com/ggml-org/llama.cpp/pull/13014), a more likely explanation is the enabling of CUDA graphs for TG with MoE models in [this PR](https://github.com/ggml-org/llama.cpp/pull/12970) (CUDA graphs are disabled in `ik_llama.cpp` for MoE models). But as soon as there are some tokens in the KV cache (the normal use case scenario), `ik_llama.cpp` becomes faster. The performance gap grows with increasing KV cache size and reaches 1.8X at 32k tokens.

![dsl2_cuda_tg](https://github.com/user-attachments/assets/49af1fbc-4cad-4929-9147-5faf18aa65ce)

The next graph compares CUDA PP performance as a function of `N_KV` for `u_batch` size of 1024 tokens. The performance optimizations in `ik_llama.cpp` have not been independently discovered yet, so here performance gap is 1.85X for small `N_KV`, increasing to 2.5X at 32k tokens. 

![dsl2_cuda_pp](https://github.com/user-attachments/assets/5ceffcaa-c2dc-4e9a-8833-9405d5c34a00)

<details>
<summary> llama.cpp CUDA performance data</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    0.316 |  3243.40 |    1.216 |   210.47 |
|  1024 |    256 |   1024 |    0.270 |  3798.75 |    1.651 |   155.05 |
|  1024 |    256 |   2048 |    0.296 |  3464.06 |    1.843 |   138.94 |
|  1024 |    256 |   3072 |    0.325 |  3150.91 |    2.050 |   124.88 |
|  1024 |    256 |   4096 |    0.356 |  2877.39 |    2.231 |   114.76 |
|  1024 |    256 |   5120 |    0.389 |  2630.72 |    2.444 |   104.75 |
|  1024 |    256 |   6144 |    0.417 |  2457.48 |    2.641 |    96.93 |
|  1024 |    256 |   7168 |    0.449 |  2278.58 |    2.850 |    89.84 |
|  1024 |    256 |   8192 |    0.489 |  2096.06 |    3.063 |    83.59 |
|  1024 |    256 |   9216 |    0.531 |  1927.90 |    3.272 |    78.23 |
|  1024 |    256 |  10240 |    0.553 |  1852.72 |    3.498 |    73.18 |
|  1024 |    256 |  11264 |    0.593 |  1725.85 |    3.703 |    69.13 |
|  1024 |    256 |  12288 |    0.614 |  1667.04 |    3.930 |    65.14 |
|  1024 |    256 |  13312 |    0.635 |  1611.74 |    4.145 |    61.76 |
|  1024 |    256 |  14336 |    0.678 |  1509.69 |    4.372 |    58.55 |
|  1024 |    256 |  15360 |    0.696 |  1470.41 |    4.586 |    55.83 |
|  1024 |    256 |  16384 |    0.740 |  1382.99 |    4.807 |    53.26 |
|  1024 |    256 |  17408 |    0.762 |  1343.59 |    5.029 |    50.91 |
|  1024 |    256 |  18432 |    0.787 |  1301.07 |    5.242 |    48.83 |
|  1024 |    256 |  19456 |    0.823 |  1244.17 |    5.463 |    46.86 |
|  1024 |    256 |  20480 |    0.846 |  1210.20 |    5.669 |    45.16 |
|  1024 |    256 |  21504 |    0.892 |  1148.57 |    5.911 |    43.31 |
|  1024 |    256 |  22528 |    0.915 |  1119.55 |    6.113 |    41.88 |
|  1024 |    256 |  23552 |    0.955 |  1071.99 |    6.345 |    40.35 |
|  1024 |    256 |  24576 |    0.979 |  1045.94 |    6.538 |    39.15 |
|  1024 |    256 |  25600 |    1.002 |  1021.85 |    6.779 |    37.76 |
|  1024 |    256 |  26624 |    1.045 |   980.14 |    6.967 |    36.74 |
|  1024 |    256 |  27648 |    1.065 |   961.08 |    7.211 |    35.50 |
|  1024 |    256 |  28672 |    1.105 |   926.56 |    7.398 |    34.60 |
|  1024 |    256 |  29696 |    1.132 |   904.44 |    7.654 |    33.45 |
|  1024 |    256 |  30720 |    1.167 |   877.39 |    7.846 |    32.63 |
|  1024 |    256 |  31744 |    1.185 |   864.19 |    8.107 |    31.58 |

</details>

<details>
<summary> ik_llama.cpp CUDA performance data</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    0.152 |  6756.76 |    1.411 |   181.44 |
|  1024 |    256 |   1024 |    0.146 |  7030.26 |    1.500 |   170.61 |
|  1024 |    256 |   2048 |    0.153 |  6676.49 |    1.600 |   160.02 |
|  1024 |    256 |   3072 |    0.166 |  6175.71 |    1.666 |   153.67 |
|  1024 |    256 |   4096 |    0.178 |  5762.29 |    1.776 |   144.18 |
|  1024 |    256 |   5120 |    0.188 |  5444.81 |    1.873 |   136.67 |
|  1024 |    256 |   6144 |    0.197 |  5202.70 |    1.959 |   130.66 |
|  1024 |    256 |   7168 |    0.206 |  4962.35 |    2.063 |   124.09 |
|  1024 |    256 |   8192 |    0.218 |  4696.99 |    2.136 |   119.83 |
|  1024 |    256 |   9216 |    0.229 |  4468.32 |    2.251 |   113.72 |
|  1024 |    256 |  10240 |    0.241 |  4240.46 |    2.344 |   109.20 |
|  1024 |    256 |  11264 |    0.254 |  4036.79 |    2.426 |   105.54 |
|  1024 |    256 |  12288 |    0.265 |  3861.63 |    2.518 |   101.68 |
|  1024 |    256 |  13312 |    0.276 |  3704.23 |    2.610 |    98.09 |
|  1024 |    256 |  14336 |    0.289 |  3547.76 |    2.718 |    94.19 |
|  1024 |    256 |  15360 |    0.299 |  3419.88 |    2.796 |    91.55 |
|  1024 |    256 |  16384 |    0.310 |  3305.62 |    2.897 |    88.38 |
|  1024 |    256 |  17408 |    0.321 |  3189.96 |    2.976 |    86.02 |
|  1024 |    256 |  18432 |    0.332 |  3084.30 |    3.075 |    83.24 |
|  1024 |    256 |  19456 |    0.342 |  2993.22 |    3.179 |    80.53 |
|  1024 |    256 |  20480 |    0.352 |  2908.33 |    3.273 |    78.22 |
|  1024 |    256 |  21504 |    0.363 |  2823.02 |    3.360 |    76.19 |
|  1024 |    256 |  22528 |    0.373 |  2744.26 |    3.455 |    74.09 |
|  1024 |    256 |  23552 |    0.384 |  2665.50 |    3.543 |    72.26 |
|  1024 |    256 |  24576 |    0.395 |  2590.50 |    3.664 |    69.88 |
|  1024 |    256 |  25600 |    0.408 |  2506.74 |    3.768 |    67.94 |
|  1024 |    256 |  26624 |    0.419 |  2446.47 |    3.884 |    65.90 |
|  1024 |    256 |  27648 |    0.429 |  2384.76 |    4.016 |    63.74 |
|  1024 |    256 |  28672 |    0.439 |  2331.18 |    4.171 |    61.38 |
|  1024 |    256 |  29696 |    0.452 |  2264.41 |    4.282 |    59.78 |
|  1024 |    256 |  30720 |    0.462 |  2214.40 |    4.441 |    57.65 |
|  1024 |    256 |  31744 |    0.472 |  2168.74 |    4.562 |    56.11 |

</details>

Perhaps also of interest is the extra VRAM required. For DeepSeek-Lite at 32k tokens mainline KV-cache size 1836 MiB, along with a CUDA compute buffer size of 2280 MiB, for a total of 4116 MiB. In comparison, `ik_llama.cpp` uses 972 MiV of K-cache (there is no V-cache required as it gets computed from the K-cache at the expense of some performance reduction) plus 936 MiB of CUDA compute buffer for a total of 1908 MiB, so 2.15X times less.   

## CPU performance

Mainline does support FA on the CPU, but performance is quite bad, so I'm including mainline results with and without FA enabled. When FA is enabled, the KV cache is quantized with `Q8_0`. `ik_llama.cpp` calculations are with FlashMLA-3, which is the best option for CPU inference. 

The following graph shows CPU TG  performance as a function of `N_KV`. Here mainline FA is faster by about 3% when the KV cache is empty. This is an artifact of the way FA is implemented: the minimum size of the u-batch created is 256 tokens. When there is no actual context in the KV cache almost all tokens are masked away. Mainline's FA implementation checks for that and skips the `K*Q` dot product for such tokens. I have not bothered adding this optimization to `ik_llama.cpp` as it never is useful in actual usage (when the KV cache is not empty). With any context `ik_llama.cpp` is faster. The performance gap increases with increasing number of tokens in the KV cache and reaches 39% (no FA) or 70% (FA) at 16k tokens.
   
![dsl2_cpu_tg](https://github.com/user-attachments/assets/eb8a1793-d8ba-4157-a327-283c4b7629cf)

The next graph shows PP performance as a function of `N_KV`. Here the performance gap to mainline without FA is 2.87X for zero context, increasing to 4.5X at 16k tokens. When FA is enabled in mainline, it is 10X slower at 16k tokens.

![dsl2_cpu_pp](https://github.com/user-attachments/assets/d68ba66b-c3bf-4fae-adc8-e8dd8cb59b04)

<details>
<summary> llama.cpp CPU performance data (FA disabled)</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.938 |   264.21 |    3.802 |    33.67 |
|   512 |    128 |    512 |    2.207 |   231.96 |    3.936 |    32.52 |
|   512 |    128 |   1024 |    2.523 |   202.97 |    4.091 |    31.29 |
|   512 |    128 |   1536 |    2.883 |   177.61 |    4.273 |    29.96 |
|   512 |    128 |   2048 |    3.175 |   161.26 |    4.405 |    29.06 |
|   512 |    128 |   2560 |    3.502 |   146.20 |    4.466 |    28.66 |
|   512 |    128 |   3072 |    3.818 |   134.09 |    4.634 |    27.62 |
|   512 |    128 |   3584 |    4.134 |   123.84 |    4.685 |    27.32 |
|   512 |    128 |   4096 |    4.460 |   114.79 |    4.838 |    26.46 |
|   512 |    128 |   4608 |    4.783 |   107.04 |    4.967 |    25.77 |
|   512 |    128 |   5120 |    5.102 |   100.36 |    5.105 |    25.07 |
|   512 |    128 |   5632 |    5.398 |    94.84 |    5.246 |    24.40 |
|   512 |    128 |   6144 |    5.737 |    89.25 |    5.396 |    23.72 |
|   512 |    128 |   6656 |    6.067 |    84.40 |    5.529 |    23.15 |
|   512 |    128 |   7168 |    6.372 |    80.35 |    5.663 |    22.60 |
|   512 |    128 |   7680 |    6.682 |    76.63 |    5.781 |    22.14 |
|   512 |    128 |   8192 |    7.010 |    73.03 |    5.909 |    21.66 |
|   512 |    128 |   8704 |    7.335 |    69.81 |    6.020 |    21.26 |
|   512 |    128 |   9216 |    7.643 |    66.99 |    6.125 |    20.90 |
|   512 |    128 |   9728 |    7.928 |    64.58 |    6.233 |    20.53 |
|   512 |    128 |  10240 |    8.282 |    61.82 |    6.358 |    20.13 |
|   512 |    128 |  10752 |    8.601 |    59.53 |    6.487 |    19.73 |
|   512 |    128 |  11264 |    8.912 |    57.45 |    6.625 |    19.32 |
|   512 |    128 |  11776 |    9.194 |    55.69 |    6.760 |    18.94 |
|   512 |    128 |  12288 |    9.549 |    53.62 |    6.898 |    18.56 |
|   512 |    128 |  12800 |    9.872 |    51.86 |    7.028 |    18.21 |
|   512 |    128 |  13312 |   10.186 |    50.27 |    7.161 |    17.87 |
|   512 |    128 |  13824 |   10.465 |    48.92 |    7.281 |    17.58 |
|   512 |    128 |  14336 |   10.824 |    47.30 |    7.398 |    17.30 |
|   512 |    128 |  14848 |   11.142 |    45.95 |    7.508 |    17.05 |
|   512 |    128 |  15360 |   11.462 |    44.67 |    7.620 |    16.80 |
|   512 |    128 |  15872 |   11.733 |    43.64 |    7.721 |    16.58 |

</details>

<details>
<summary> llama.cpp CPU performance data (FA enabled)</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.912 |   267.73 |    3.695 |    34.64 |
|   512 |    128 |    512 |    2.618 |   195.55 |    3.846 |    33.28 |
|   512 |    128 |   1024 |    3.394 |   150.85 |    4.028 |    31.78 |
|   512 |    128 |   1536 |    4.184 |   122.38 |    4.211 |    30.40 |
|   512 |    128 |   2048 |    4.958 |   103.27 |    4.416 |    28.98 |
|   512 |    128 |   2560 |    5.711 |    89.65 |    4.582 |    27.94 |
|   512 |    128 |   3072 |    6.545 |    78.22 |    4.767 |    26.85 |
|   512 |    128 |   3584 |    7.257 |    70.55 |    4.958 |    25.81 |
|   512 |    128 |   4096 |    8.079 |    63.37 |    5.143 |    24.89 |
|   512 |    128 |   4608 |    8.981 |    57.01 |    5.336 |    23.99 |
|   512 |    128 |   5120 |    9.600 |    53.33 |    5.468 |    23.41 |
|   512 |    128 |   5632 |   10.373 |    49.36 |    5.660 |    22.62 |
|   512 |    128 |   6144 |   11.271 |    45.43 |    5.850 |    21.88 |
|   512 |    128 |   6656 |   11.922 |    42.95 |    6.058 |    21.13 |
|   512 |    128 |   7168 |   12.692 |    40.34 |    6.247 |    20.49 |
|   512 |    128 |   7680 |   13.498 |    37.93 |    6.435 |    19.89 |
|   512 |    128 |   8192 |   14.237 |    35.96 |    6.563 |    19.50 |
|   512 |    128 |   8704 |   15.004 |    34.12 |    6.755 |    18.95 |
|   512 |    128 |   9216 |   15.794 |    32.42 |    6.942 |    18.44 |
|   512 |    128 |   9728 |   16.552 |    30.93 |    7.131 |    17.95 |
|   512 |    128 |  10240 |   17.326 |    29.55 |    7.321 |    17.48 |
|   512 |    128 |  10752 |   18.126 |    28.25 |    7.520 |    17.02 |
|   512 |    128 |  11264 |   18.846 |    27.17 |    7.713 |    16.60 |
|   512 |    128 |  11776 |   19.618 |    26.10 |    7.902 |    16.20 |
|   512 |    128 |  12288 |   20.404 |    25.09 |    8.096 |    15.81 |
|   512 |    128 |  12800 |   21.219 |    24.13 |    8.286 |    15.45 |
|   512 |    128 |  13312 |   21.950 |    23.33 |    8.543 |    14.98 |
|   512 |    128 |  13824 |   22.765 |    22.49 |    8.735 |    14.65 |
|   512 |    128 |  14336 |   23.532 |    21.76 |    8.933 |    14.33 |
|   512 |    128 |  14848 |   24.284 |    21.08 |    9.119 |    14.04 |
|   512 |    128 |  15360 |   25.070 |    20.42 |    9.316 |    13.74 |
|   512 |    128 |  15872 |   25.856 |    19.80 |    9.510 |    13.46 |

</details>

<details>
<summary>ik_llama.cpp CPU performance data</summary>
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.739 |   693.23 |    3.836 |    33.37 |
|   512 |    128 |    512 |    0.769 |   665.76 |    3.931 |    32.56 |
|   512 |    128 |   1024 |    0.817 |   626.90 |    3.958 |    32.34 |
|   512 |    128 |   1536 |    0.869 |   589.09 |    3.991 |    32.07 |
|   512 |    128 |   2048 |    0.912 |   561.30 |    4.037 |    31.71 |
|   512 |    128 |   2560 |    0.967 |   529.68 |    4.087 |    31.32 |
|   512 |    128 |   3072 |    1.020 |   502.07 |    4.146 |    30.87 |
|   512 |    128 |   3584 |    1.087 |   470.96 |    4.182 |    30.61 |
|   512 |    128 |   4096 |    1.132 |   452.35 |    4.235 |    30.22 |
|   512 |    128 |   4608 |    1.189 |   430.73 |    4.290 |    29.84 |
|   512 |    128 |   5120 |    1.247 |   410.52 |    4.351 |    29.42 |
|   512 |    128 |   5632 |    1.304 |   392.59 |    4.426 |    28.92 |
|   512 |    128 |   6144 |    1.363 |   375.64 |    4.508 |    28.39 |
|   512 |    128 |   6656 |    1.420 |   360.52 |    4.584 |    27.92 |
|   512 |    128 |   7168 |    1.485 |   344.78 |    4.665 |    27.44 |
|   512 |    128 |   7680 |    1.542 |   332.04 |    4.751 |    26.94 |
|   512 |    128 |   8192 |    1.605 |   318.99 |    4.821 |    26.55 |
|   512 |    128 |   8704 |    1.669 |   306.76 |    4.736 |    27.02 |
|   512 |    128 |   9216 |    1.736 |   294.93 |    4.773 |    26.82 |
|   512 |    128 |   9728 |    1.802 |   284.05 |    4.832 |    26.49 |
|   512 |    128 |  10240 |    1.865 |   274.57 |    4.889 |    26.18 |
|   512 |    128 |  10752 |    1.927 |   265.65 |    4.949 |    25.87 |
|   512 |    128 |  11264 |    1.994 |   256.77 |    5.015 |    25.53 |
|   512 |    128 |  11776 |    2.063 |   248.24 |    5.074 |    25.23 |
|   512 |    128 |  12288 |    2.127 |   240.67 |    5.139 |    24.91 |
|   512 |    128 |  12800 |    2.194 |   233.39 |    5.207 |    24.58 |
|   512 |    128 |  13312 |    2.262 |   226.33 |    5.272 |    24.28 |
|   512 |    128 |  13824 |    2.326 |   220.10 |    5.342 |    23.96 |
|   512 |    128 |  14336 |    2.389 |   214.35 |    5.399 |    23.71 |
|   512 |    128 |  14848 |    2.456 |   208.43 |    5.461 |    23.44 |
|   512 |    128 |  15360 |    2.522 |   203.02 |    5.511 |    23.23 |
|   512 |    128 |  15872 |    2.590 |   197.72 |    5.573 |    22.97 |

</details>

---

#### 🗣️ Discussion

👤 **JohannesGaessler** replied the **2025-04-29** at **07:29:26**:<br>

Since you are tagging me: I did look at the more general implementation for mapping MoE to regular matrix multiplications in the PR where I commented but I did not look at any MoE-specific CUDA code for matrix vector multiplication, nor was I aware that this repository had such an optimization. It's just the natural way of writing a fused kernel.

> 👤 **ikawrakow** replied the **2025-04-29** at **14:39:31**:<br>
> >  It's just the natural way of writing a fused kernel.
> 
> Sure, a kernel that did not get written for a very long time, despite the well known fact that `llama.cpp` CUDA performance for MoE models is really bad. Which indicates that the understanding how badly the fused kernel was needed was missing. It is not very often that one has a PR that [improves performance up to 4X](https://github.com/ggml-org/llama.cpp/pull/13014#issuecomment-2816637977).
> 
> But if it is so as you say, then sorry.
> 
> 👤 **JohannesGaessler** replied the **2025-04-29** at **15:33:40**:<br>
> Apology accepted. My top priority was and still is good performance for dense GEMM/GEMV because that is the most fundamental operation. MoE optimizations have now simply reached the front of the priority queue.

---

👤 **cmoncure** replied the **2025-05-06** at **15:50:00**:<br>

I read this and the warning on the README.md about incompatible GGUFs is quite unfortunate. I don't mind spending the time to create my own quants for this fork in the pursuit of maximum performance. I am a total noob to creating quants, however.

I am building an EPYC box with 768 GB RAM and 96 GB VRAM (2x48). Will I be able to use scripts to conveniently convert such releases as DeepSeek V3/R1 or the curious tngtech/DeepSeek-R1T-Chimera model from safetensors?

Do you plan to support the incompatible mainline GGUF files? Can I assume that GGUFs created before mid-April or so will be compatible? (Downloading these larger models represents a considerable cost.)

Thank you for creating this work and making it available. You are a true wizard.

> 👤 **ikawrakow** replied the **2025-05-06** at **16:16:34**:<br>
> > Can I assume that GGUFs created before mid-April or so will be compatible? (Downloading these larger models represents a considerable cost.)
> 
> I think so. But to make sure, if you are downloading from HF, you can check the content of the GGUF. To be compatible, it needs to have tensors ` blk.X.attn_kv_b.weight` (where `X` is the layer index, so 0,1,...). If it does, it will work with this fork. If instead it has separate tensors `blk.X.attn_k_b.weight` and `blk.X.attn_v_b.weight`, it is most likely not compatible. 
> 
> > Do you plan to support the incompatible mainline GGUF files? 
> 
> No, not really. There are implications beyond compatibility. The change impacts quantization of the attention tensors, and I think there are now some reports from users about reduced model quality after the change was made and the quantized models compatible with that change started coming out.
> 
> 👤 **saood06** replied the **2025-05-06** at **20:24:09**:<br>
> > I think so. But to make sure, if you are downloading from HF, you can check the content of the GGUF. To be compatible, it needs to have tensors ` blk.X.attn_kv_b.weight` (where `X` is the layer index, so 0,1,...). If it does, it will work with this fork. If instead it has separate tensors `blk.X.attn_k_b.weight` and `blk.X.attn_v_b.weight`, it is most likely not compatible.
> 
> Just to be more clear after looking at one converted with the compatible version of MLA that works [here](https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF/tree/main/DeepSeek-V3-0324-IQ2_K_R4?show_file_info=DeepSeek-V3-0324-IQ2_K_R4%2FDeepSeek-V3-0324-IQ2_K_R4-00001-of-00005.gguf) , it has `attn_k_b.weight`, `attn_v_b.weight` and `attn_kv_b.weight`.
> 
> Looking at one converted with the incompatible version of MLA that does not work [here](https://huggingface.co/bullerwins/DeepSeek-R1T-Chimera-GGUF/tree/main/DeepSeek-R1T-Chimera-Q4_K_M?show_file_info=DeepSeek-R1T-Chimera-Q4_K_M%2FDeepSeek-R1T-Chimera-Q4_K_M-00001-of-00010.gguf) it is missing `attn_kv_b.weight` but has `attn_k_b.weight` and `attn_v_b.weight`.
> 
> Looking at one converted from before MLA support which will work here by generating the MLA tensors on the fly [here](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q2_K_L?show_file_info=DeepSeek-V3-Q2_K_L%2FDeepSeek-V3-Q2_K_L-00001-of-00005.gguf) it has `attn_kv_b.weight` but not  `attn_k_b.weight`, `attn_v_b.weight`.
> 
> So in conclusion if the model has all three `attn_k_b.weight`, `attn_v_b.weight` and `attn_kv_b.weight` or just `attn_kv_b.weight` it will work here, but if it has `attn_k_b.weight` and `attn_v_b.weight` but no `attn_kv_b.weight` it will not work here.
> 
> Edit: The above is outdated, see #394 and #409
> 
> 👤 **ubergarm** replied the **2025-05-12** at **15:39:39**:<br>
> Sorry for late reply @cmoncure , I have a rough outline of the process of going from fp8 to GGUF for ik's fork [buried in a fold in my quickstart guide](https://github.com/ikawrakow/ik_llama.cpp/discussions/258) under the "Custom Quants" section.
> 
> Its a bit dated already, but the basic procedures are described there. I'd suggest making your own imatrix and take [this new PR411 into consideration ](https://github.com/ikawrakow/ik_llama.cpp/pull/411) for that step as well.
> 
> 👤 **saood06** replied the **2025-05-13** at **00:23:49**:<br>
> > Sorry for late reply @cmoncure , I have a rough outline of the process of going from fp8 to GGUF for ik's fork [buried in a fold in my quickstart guide](https://github.com/ikawrakow/ik_llama.cpp/discussions/258) under the "Custom Quants" section.
> > 
> > Its a bit dated already, but the basic procedures are described there. I'd suggest making your own imatrix and take [this new PR411 into consideration ](https://github.com/ikawrakow/ik_llama.cpp/pull/411) for that step as well.
> 
> The dequant method in your guide (that I had recommended) may need more precise instructions to work now. For more info see [this](https://github.com/ikawrakow/ik_llama.cpp/issues/383#issuecomment-2865306085) and the following comments.
> 
> 👤 **ubergarm** replied the **2025-05-13** at **20:13:04**:<br>
> Thanks @saood06 , I managed to `git apply saood06.patch` copy/pasting your comment and that fixes up building `triton-cpu`. I tested with `uv venv ./venv --python 3.12 --python-preference=only-managed` for my venv and updated a couple lines of the quick start guide.
> 
> Hopefully enough bread crumbs our future selves can figure it out.
> 
> 👤 **saood06** replied the **2025-05-13** at **21:09:54**:<br>
> > Thanks @saood06 , I managed to `git apply saood06.patch` copy/pasting your comment and that fixes up building `triton-cpu`. 
> 
> Mind telling me the exact version/commit hash of `triton-cpu` you built?
> 
> I noticed mine is 3.2.0 and they seem to be on 3.3.0 (and thus I hoped the bug would be fixed upstream)
> 
> 👤 **ubergarm** replied the **2025-05-13** at **21:21:58**:<br>
> > > Thanks @saood06 , I managed to `git apply saood06.patch` copy/pasting your comment and that fixes up building `triton-cpu`.
> > 
> > Mind telling me the exact version/commit hash of `triton-cpu` you built?
> > 
> > I noticed mine is 3.2.0 and they seem to be on 3.3.0 (and thus I hoped the bug would be fixed upstream)
> 
> I added your patch to `main@0625715c` `Artlesbol` `[MathToVecLib] Add support for setting bit-widths for AVX512...` `Apr 26 12:24:21 2025 +0800`
> 
> I originally tried to use the same git sha I used the first time, but it doesn't exist anymore, so I guess they force pushed main or something somewhere along the way between now and March 13, 2025 maybe?
> 
> 👤 **saood06** replied the **2025-05-13** at **21:45:22**:<br>
> > I originally tried to use the same git sha I used the first time, but it doesn't exist anymore, so I guess they force pushed main or something somewhere along the way between now and March 13, 2025 maybe?
> 
> I noticed similar things when trying to look into the history of the repo. Whatever they are doing it makes tracing down the source of changes in their repo very tedious and annoying.
> 
> Thanks for confirming the issue still exists in their latest commit, I don't currently plan on creating a better fix for them so I made an issue https://github.com/triton-lang/triton-cpu/issues/237 and hopefully they fix it.
> 
> 👤 **saood06** replied the **2025-05-13** at **22:33:34**:<br>
> @ubergarm if you still have the build errors that my patch solves do you mind sharing them in the issue I made. I don't have them, and they are requesting them in the issue I opened.
> 
> 👤 **ubergarm** replied the **2025-05-13** at **23:10:18**:<br>
> > @ubergarm if you still have the build errors that my patch solves do you mind sharing them in the issue I made. I don't have them, and they are requesting them in the issue I opened.
> 
> Its a goofy browser ssh client for this specific rig, i tried to scroll my tmux back but its gone... 
> 
> I see the issue and will just delete my `venv` and try to repro and paste it in there: https://github.com/triton-lang/triton-cpu/issues/237