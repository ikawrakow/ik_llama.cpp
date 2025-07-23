### 🗣️ [#357](https://github.com/ikawrakow/ik_llama.cpp/discussions/357) - Qwen3 - early performance comparisons

| **Author** | `ikawrakow` |
| :--- | :--- |
| **Created** | 2025-04-29 |
| **Updated** | 2025-05-19 |

---

#### Description

The Qwen3 models were [officially released](https://qwenlm.github.io/blog/qwen3/), and support was added in `ik_llama.cpp` in PR #355, so I was curious to run some performance benchmarks. As much as I would like to try the flagship model, I don't have enough horse power for that, so I experimented with [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B), the 30B total, 3B active parameter MoE model.

This time I'm using a custom quantization where all experts are quantized with `IQ4_XS`, all attention tensors with `Q5_K`, and the output tensor is `Q6_K`. PPL for this model is only 1.25% above the PPL of the `bf16` model, so it is a pretty decent quality quantization. Benchmarks are run on a Ryzen-7950X system with an RTX-4080 GPU.  Compared are the latest `ik_kllama.cpp` and `llama.cpp` versions as of this morning (April 29 2025).

## CPU-only performance

The command line for `ik_llama.cpp` is
```
./bin/llama-sweep-bench -m $model -c 16384 -t 16 -fa -rtr -fmoe -ctk q8_0 -ctv q8_0
```
`llama.cpp` is similar, except that there is no `-rtr -fmoe`. I'm also including mainline results without Flash Attention (FA). In this case the K-cache is quantized with `Q8_0` and the V-cache is `fp16`.

The following graph shows TG performance as a function of `N_KV`, the number of tokens in the KV cache. Performance is pretty close for empty KV cache, with a performance gap increasing with `N_KV`. At 16k tokens `ik_llama.cpp` is 44% faster than mainline without FA, and 3.3 times faster than mainline with FA enabled. 

  
![qwen3_cpu_tg](https://github.com/user-attachments/assets/1d088f6a-6f73-4eba-8e88-76729170269b)

The next graph shows prompt processing (PP) speed as a function of `N_KV`. As usual for CPU only inference, `ik_llama.cpp` is much faster than mainline for PP - 3.3X for small `N_KV`, increasing to 3.9X at 16k tokens. This is compared to mainline without FA. Compared to `llama.cpp` with FA enabled, `ik_llama.cpp` is 11.2X faster. 

![qwen3_cpu_pp](https://github.com/user-attachments/assets/39b2695f-93f6-4f9b-9975-61c62bb650eb)

<details>
<summary>llama.cpp CPU-only performance data without FA</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.610 |   141.84 |    5.067 |    25.26 |
|   512 |    128 |    512 |    3.902 |   131.20 |    5.354 |    23.91 |
|   512 |    128 |   1024 |    4.228 |   121.09 |    5.344 |    23.95 |
|   512 |    128 |   1536 |    4.582 |   111.74 |    5.528 |    23.16 |
|   512 |    128 |   2048 |    4.837 |   105.84 |    5.713 |    22.40 |
|   512 |    128 |   2560 |    5.188 |    98.69 |    5.745 |    22.28 |
|   512 |    128 |   3072 |    5.484 |    93.37 |    5.917 |    21.63 |
|   512 |    128 |   3584 |    5.793 |    88.38 |    6.035 |    21.21 |
|   512 |    128 |   4096 |    6.039 |    84.78 |    6.256 |    20.46 |
|   512 |    128 |   4608 |    6.433 |    79.59 |    6.449 |    19.85 |
|   512 |    128 |   5120 |    6.685 |    76.59 |    6.630 |    19.31 |
|   512 |    128 |   5632 |    7.013 |    73.00 |    6.852 |    18.68 |
|   512 |    128 |   6144 |    7.278 |    70.35 |    7.075 |    18.09 |
|   512 |    128 |   6656 |    7.689 |    66.59 |    7.259 |    17.63 |
|   512 |    128 |   7168 |    7.869 |    65.07 |    7.428 |    17.23 |
|   512 |    128 |   7680 |    8.337 |    61.41 |    7.604 |    16.83 |
|   512 |    128 |   8192 |    8.488 |    60.32 |    7.788 |    16.44 |
|   512 |    128 |   8704 |    8.958 |    57.15 |    7.925 |    16.15 |
|   512 |    128 |   9216 |    9.084 |    56.36 |    8.080 |    15.84 |
|   512 |    128 |   9728 |    9.557 |    53.57 |    8.226 |    15.56 |
|   512 |    128 |  10240 |    9.725 |    52.65 |    8.466 |    15.12 |
|   512 |    128 |  10752 |   10.470 |    48.90 |    8.575 |    14.93 |
|   512 |    128 |  11264 |   10.334 |    49.55 |    8.774 |    14.59 |
|   512 |    128 |  11776 |   10.861 |    47.14 |    8.940 |    14.32 |
|   512 |    128 |  12288 |   10.974 |    46.65 |    9.121 |    14.03 |
|   512 |    128 |  12800 |   11.494 |    44.55 |    9.321 |    13.73 |
|   512 |    128 |  13312 |   11.575 |    44.23 |    9.494 |    13.48 |
|   512 |    128 |  13824 |   12.063 |    42.44 |    9.665 |    13.24 |
|   512 |    128 |  14336 |   12.267 |    41.74 |    9.854 |    12.99 |
|   512 |    128 |  14848 |   12.737 |    40.20 |    9.970 |    12.84 |
|   512 |    128 |  15360 |   13.034 |    39.28 |   10.103 |    12.67 |
|   512 |    128 |  15872 |   13.427 |    38.13 |   10.231 |    12.51 |

</details>

<details>
<summary>llama.cpp CPU-only performance data with FA enabled</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.677 |   139.25 |    5.061 |    25.29 |
|   512 |    128 |    512 |    4.714 |   108.62 |    5.427 |    23.59 |
|   512 |    128 |   1024 |    5.922 |    86.46 |    5.987 |    21.38 |
|   512 |    128 |   1536 |    6.963 |    73.53 |    6.495 |    19.71 |
|   512 |    128 |   2048 |    8.207 |    62.39 |    7.086 |    18.06 |
|   512 |    128 |   2560 |    9.405 |    54.44 |    7.753 |    16.51 |
|   512 |    128 |   3072 |   10.370 |    49.37 |    8.375 |    15.28 |
|   512 |    128 |   3584 |   11.482 |    44.59 |    8.908 |    14.37 |
|   512 |    128 |   4096 |   12.604 |    40.62 |    9.487 |    13.49 |
|   512 |    128 |   4608 |   13.798 |    37.11 |    9.951 |    12.86 |
|   512 |    128 |   5120 |   15.149 |    33.80 |   10.504 |    12.19 |
|   512 |    128 |   5632 |   16.055 |    31.89 |   11.201 |    11.43 |
|   512 |    128 |   6144 |   17.214 |    29.74 |   11.740 |    10.90 |
|   512 |    128 |   6656 |   18.347 |    27.91 |   12.409 |    10.31 |
|   512 |    128 |   7168 |   19.478 |    26.29 |   12.842 |     9.97 |
|   512 |    128 |   7680 |   20.593 |    24.86 |   13.410 |     9.55 |
|   512 |    128 |   8192 |   21.726 |    23.57 |   14.082 |     9.09 |
|   512 |    128 |   8704 |   22.886 |    22.37 |   14.582 |     8.78 |
|   512 |    128 |   9216 |   23.937 |    21.39 |   15.117 |     8.47 |
|   512 |    128 |   9728 |   25.038 |    20.45 |   15.800 |     8.10 |
|   512 |    128 |  10240 |   26.188 |    19.55 |   16.390 |     7.81 |
|   512 |    128 |  10752 |   27.328 |    18.74 |   16.962 |     7.55 |
|   512 |    128 |  11264 |   28.434 |    18.01 |   17.550 |     7.29 |
|   512 |    128 |  11776 |   29.491 |    17.36 |   18.265 |     7.01 |
|   512 |    128 |  12288 |   30.663 |    16.70 |   18.898 |     6.77 |
|   512 |    128 |  12800 |   31.799 |    16.10 |   19.649 |     6.51 |
|   512 |    128 |  13312 |   32.887 |    15.57 |   20.277 |     6.31 |
|   512 |    128 |  13824 |   34.042 |    15.04 |   20.914 |     6.12 |
|   512 |    128 |  14336 |   35.152 |    14.57 |   21.562 |     5.94 |
|   512 |    128 |  14848 |   36.281 |    14.11 |   22.194 |     5.77 |
|   512 |    128 |  15360 |   37.400 |    13.69 |   22.754 |     5.63 |
|   512 |    128 |  15872 |   38.559 |    13.28 |   23.348 |     5.48 |

</details>

<details>
<summary>ik_llama.cpp CPU-only performance data</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.079 |   474.34 |    4.858 |    26.35 |
|   512 |    128 |    512 |    1.118 |   458.04 |    5.140 |    24.90 |
|   512 |    128 |   1024 |    1.194 |   428.88 |    5.059 |    25.30 |
|   512 |    128 |   1536 |    1.273 |   402.21 |    5.138 |    24.91 |
|   512 |    128 |   2048 |    1.353 |   378.31 |    5.241 |    24.42 |
|   512 |    128 |   2560 |    1.421 |   360.38 |    5.318 |    24.07 |
|   512 |    128 |   3072 |    1.501 |   341.07 |    5.397 |    23.72 |
|   512 |    128 |   3584 |    1.580 |   324.10 |    5.443 |    23.52 |
|   512 |    128 |   4096 |    1.654 |   309.50 |    5.522 |    23.18 |
|   512 |    128 |   4608 |    1.731 |   295.70 |    5.557 |    23.03 |
|   512 |    128 |   5120 |    1.809 |   283.11 |    5.622 |    22.77 |
|   512 |    128 |   5632 |    1.879 |   272.50 |    5.688 |    22.51 |
|   512 |    128 |   6144 |    1.963 |   260.87 |    5.750 |    22.26 |
|   512 |    128 |   6656 |    2.040 |   250.94 |    5.820 |    21.99 |
|   512 |    128 |   7168 |    2.122 |   241.24 |    5.893 |    21.72 |
|   512 |    128 |   7680 |    2.193 |   233.47 |    5.966 |    21.45 |
|   512 |    128 |   8192 |    2.281 |   224.44 |    6.039 |    21.19 |
|   512 |    128 |   8704 |    2.353 |   217.56 |    6.109 |    20.95 |
|   512 |    128 |   9216 |    2.436 |   210.21 |    6.176 |    20.73 |
|   512 |    128 |   9728 |    2.504 |   204.46 |    6.245 |    20.50 |
|   512 |    128 |  10240 |    2.596 |   197.19 |    6.317 |    20.26 |
|   512 |    128 |  10752 |    2.670 |   191.76 |    6.386 |    20.04 |
|   512 |    128 |  11264 |    2.756 |   185.79 |    6.459 |    19.82 |
|   512 |    128 |  11776 |    2.822 |   181.46 |    6.528 |    19.61 |
|   512 |    128 |  12288 |    2.917 |   175.54 |    6.596 |    19.41 |
|   512 |    128 |  12800 |    2.987 |   171.41 |    6.671 |    19.19 |
|   512 |    128 |  13312 |    3.073 |   166.62 |    6.740 |    18.99 |
|   512 |    128 |  13824 |    3.121 |   164.03 |    6.819 |    18.77 |
|   512 |    128 |  14336 |    3.230 |   158.50 |    6.888 |    18.58 |
|   512 |    128 |  14848 |    3.288 |   155.73 |    6.961 |    18.39 |
|   512 |    128 |  15360 |    3.389 |   151.07 |    7.037 |    18.19 |
|   512 |    128 |  15872 |    3.444 |   148.68 |    7.109 |    18.00 |

</details>

## Hybrid inference

The custom `IQ4_XS` model is 15.4 GiB, so cannot be fully loaded on my 16 GB RTX-4080 GPU. This gives me the opportunity to try hybrid GPU+CPU inference via tensor overrides on both systems. The command line used in this case is
```
./bin/llama-sweep-bench -m $model -c 32768 -t 16 -ngl 100 -fa -ot "blk\.3[4-9]\.ffn=CPU,blk\.4[0-9]\.ffn=CPU"
```
I.e., everything is offloaded to the GPU except for the last 14 layers of the experts tensors. This leaves enough free VRAM to go up to a context of 32k tokens. In the case of `ik_llama.cpp` run-time-repacking (for the experts left on the CPU) and fused MoE `(ffn_up*X)*silu(ffn_gate*X)` is enabled via `-rtr -fmoe`.

The next graph shows TG performance as a function of `N_KV`.  [Compared to DeepSeek](https://github.com/ikawrakow/ik_llama.cpp/discussions/354), Here the performance advantage of `ik_llama.cpp` is smaller and decreases with increasing `N_KV`. As there is no MLA involved, and we are dealing just with a standard attention mechanism, the CUDA FA improvements [in this mainline PR](https://github.com/ggml-org/llama.cpp/pull/12014) that I have not (yet) ported over to `ik_llama.cpp` counteract the performance gains from the fused MoE operations in `ik_llama.cpp`, so we end up with a relatively close TG performance. 

![qwen3_hybrid_tg](https://github.com/user-attachments/assets/a8172134-c9b8-47d8-83e2-7bda514703f0)

The next graph shows PP performance as a function of `N_KV`. Also here the performance gap decreases with `N_KV`, from about 60% for small `N_KV`, to about 18% at 32k tokens.

![qwen3_hybrid_pp](https://github.com/user-attachments/assets/d8154ca0-7512-4eab-9292-83d2c7de910a)

<details>
<summary>llama.cpp hybrid GPU+CPU performance data</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.587 |   871.89 |    1.864 |    68.68 |
|   512 |    128 |    512 |    0.499 |  1025.61 |    1.893 |    67.63 |
|   512 |    128 |   1024 |    0.505 |  1013.85 |    1.924 |    66.53 |
|   512 |    128 |   1536 |    0.504 |  1015.33 |    1.936 |    66.11 |
|   512 |    128 |   2048 |    0.519 |   987.22 |    1.959 |    65.33 |
|   512 |    128 |   2560 |    0.508 |  1008.35 |    1.978 |    64.71 |
|   512 |    128 |   3072 |    0.512 |   999.60 |    1.991 |    64.30 |
|   512 |    128 |   3584 |    0.508 |  1008.64 |    2.020 |    63.37 |
|   512 |    128 |   4096 |    0.516 |   992.09 |    2.027 |    63.15 |
|   512 |    128 |   4608 |    0.517 |   989.86 |    2.055 |    62.28 |
|   512 |    128 |   5120 |    0.520 |   983.77 |    2.065 |    61.97 |
|   512 |    128 |   5632 |    0.518 |   987.91 |    2.085 |    61.40 |
|   512 |    128 |   6144 |    0.522 |   980.59 |    2.110 |    60.66 |
|   512 |    128 |   6656 |    0.525 |   975.45 |    2.117 |    60.45 |
|   512 |    128 |   7168 |    0.532 |   962.98 |    2.147 |    59.62 |
|   512 |    128 |   7680 |    0.530 |   966.27 |    2.157 |    59.34 |
|   512 |    128 |   8192 |    0.539 |   950.13 |    2.181 |    58.68 |
|   512 |    128 |   8704 |    0.534 |   958.91 |    2.191 |    58.43 |
|   512 |    128 |   9216 |    0.538 |   952.23 |    2.216 |    57.76 |
|   512 |    128 |   9728 |    0.541 |   946.25 |    2.239 |    57.17 |
|   512 |    128 |  10240 |    0.538 |   951.61 |    2.259 |    56.66 |
|   512 |    128 |  10752 |    0.550 |   930.85 |    2.258 |    56.70 |
|   512 |    128 |  11264 |    0.547 |   935.91 |    2.272 |    56.33 |
|   512 |    128 |  11776 |    0.550 |   930.19 |    2.291 |    55.87 |
|   512 |    128 |  12288 |    0.550 |   931.21 |    2.307 |    55.49 |
|   512 |    128 |  12800 |    0.555 |   923.16 |    2.330 |    54.95 |
|   512 |    128 |  13312 |    0.556 |   921.17 |    2.355 |    54.36 |
|   512 |    128 |  13824 |    0.558 |   917.56 |    2.366 |    54.10 |
|   512 |    128 |  14336 |    0.557 |   918.53 |    2.388 |    53.60 |
|   512 |    128 |  14848 |    0.563 |   908.69 |    2.400 |    53.33 |
|   512 |    128 |  15360 |    0.565 |   905.61 |    2.425 |    52.79 |
|   512 |    128 |  15872 |    0.570 |   897.66 |    2.435 |    52.57 |
|   512 |    128 |  16384 |    0.570 |   897.53 |    2.447 |    52.30 |
|   512 |    128 |  16896 |    0.573 |   893.67 |    2.472 |    51.77 |
|   512 |    128 |  17408 |    0.578 |   885.91 |    2.484 |    51.54 |
|   512 |    128 |  17920 |    0.579 |   884.78 |    2.508 |    51.04 |
|   512 |    128 |  18432 |    0.585 |   875.25 |    2.523 |    50.72 |
|   512 |    128 |  18944 |    0.582 |   879.31 |    2.556 |    50.07 |
|   512 |    128 |  19456 |    0.590 |   868.21 |    2.585 |    49.52 |
|   512 |    128 |  19968 |    0.592 |   865.23 |    2.612 |    49.01 |
|   512 |    128 |  20480 |    0.585 |   875.09 |    2.637 |    48.53 |
|   512 |    128 |  20992 |    0.590 |   867.98 |    2.655 |    48.21 |
|   512 |    128 |  21504 |    0.596 |   858.70 |    2.671 |    47.92 |
|   512 |    128 |  22016 |    0.597 |   858.04 |    2.692 |    47.55 |
|   512 |    128 |  22528 |    0.602 |   849.98 |    2.713 |    47.17 |
|   512 |    128 |  23040 |    0.604 |   847.68 |    2.733 |    46.83 |
|   512 |    128 |  23552 |    0.604 |   847.62 |    2.759 |    46.40 |
|   512 |    128 |  24064 |    0.607 |   844.15 |    2.785 |    45.96 |
|   512 |    128 |  24576 |    0.609 |   840.08 |    2.804 |    45.65 |
|   512 |    128 |  25088 |    0.610 |   839.13 |    2.830 |    45.23 |
|   512 |    128 |  25600 |    0.609 |   840.04 |    2.841 |    45.06 |
|   512 |    128 |  26112 |    0.613 |   835.24 |    2.866 |    44.66 |
|   512 |    128 |  26624 |    0.617 |   829.66 |    2.878 |    44.47 |
|   512 |    128 |  27136 |    0.620 |   825.17 |    2.907 |    44.03 |
|   512 |    128 |  27648 |    0.622 |   823.54 |    2.932 |    43.65 |
|   512 |    128 |  28160 |    0.628 |   815.24 |    2.957 |    43.28 |
|   512 |    128 |  28672 |    0.635 |   806.54 |    3.022 |    42.35 |
|   512 |    128 |  29184 |    0.635 |   806.74 |    3.029 |    42.26 |
|   512 |    128 |  29696 |    0.635 |   805.74 |    3.054 |    41.91 |
|   512 |    128 |  30208 |    0.635 |   806.01 |    3.066 |    41.74 |
|   512 |    128 |  30720 |    0.641 |   799.08 |    3.094 |    41.37 |
|   512 |    128 |  31232 |    0.641 |   798.16 |    3.119 |    41.04 |
|   512 |    128 |  31744 |    0.642 |   797.16 |    3.134 |    40.85 |
|   512 |    128 |  32256 |    0.647 |   791.04 |    3.155 |    40.57 |

</details>

<details>
<summary>ik_llama.cpp hybrid GPU+CPU performance data</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.354 |  1445.39 |    1.668 |    76.74 |
|   512 |    128 |    512 |    0.305 |  1676.45 |    1.678 |    76.27 |
|   512 |    128 |   1024 |    0.311 |  1644.31 |    1.708 |    74.95 |
|   512 |    128 |   1536 |    0.309 |  1656.71 |    1.724 |    74.23 |
|   512 |    128 |   2048 |    0.322 |  1588.27 |    1.759 |    72.77 |
|   512 |    128 |   2560 |    0.318 |  1609.63 |    1.771 |    72.29 |
|   512 |    128 |   3072 |    0.326 |  1568.33 |    1.798 |    71.19 |
|   512 |    128 |   3584 |    0.324 |  1578.28 |    1.817 |    70.43 |
|   512 |    128 |   4096 |    0.331 |  1545.52 |    1.830 |    69.93 |
|   512 |    128 |   4608 |    0.336 |  1524.39 |    1.864 |    68.66 |
|   512 |    128 |   5120 |    0.338 |  1512.69 |    1.876 |    68.24 |
|   512 |    128 |   5632 |    0.341 |  1503.24 |    1.915 |    66.84 |
|   512 |    128 |   6144 |    0.345 |  1483.42 |    1.920 |    66.65 |
|   512 |    128 |   6656 |    0.350 |  1464.58 |    1.933 |    66.22 |
|   512 |    128 |   7168 |    0.356 |  1439.26 |    1.969 |    65.02 |
|   512 |    128 |   7680 |    0.358 |  1432.11 |    1.983 |    64.54 |
|   512 |    128 |   8192 |    0.365 |  1401.85 |    2.008 |    63.75 |
|   512 |    128 |   8704 |    0.364 |  1406.00 |    2.030 |    63.05 |
|   512 |    128 |   9216 |    0.370 |  1384.70 |    2.048 |    62.49 |
|   512 |    128 |   9728 |    0.374 |  1370.08 |    2.074 |    61.72 |
|   512 |    128 |  10240 |    0.375 |  1366.56 |    2.085 |    61.39 |
|   512 |    128 |  10752 |    0.384 |  1334.85 |    2.118 |    60.44 |
|   512 |    128 |  11264 |    0.384 |  1333.89 |    2.134 |    59.98 |
|   512 |    128 |  11776 |    0.389 |  1316.69 |    2.146 |    59.63 |
|   512 |    128 |  12288 |    0.391 |  1309.81 |    2.177 |    58.80 |
|   512 |    128 |  12800 |    0.396 |  1293.36 |    2.190 |    58.45 |
|   512 |    128 |  13312 |    0.399 |  1282.92 |    2.223 |    57.57 |
|   512 |    128 |  13824 |    0.403 |  1271.01 |    2.240 |    57.15 |
|   512 |    128 |  14336 |    0.405 |  1263.29 |    2.254 |    56.78 |
|   512 |    128 |  14848 |    0.412 |  1242.83 |    2.285 |    56.01 |
|   512 |    128 |  15360 |    0.416 |  1231.56 |    2.302 |    55.60 |
|   512 |    128 |  15872 |    0.419 |  1221.90 |    2.332 |    54.90 |
|   512 |    128 |  16384 |    0.422 |  1212.98 |    2.326 |    55.04 |
|   512 |    128 |  16896 |    0.427 |  1200.46 |    2.347 |    54.54 |
|   512 |    128 |  17408 |    0.431 |  1186.63 |    2.381 |    53.77 |
|   512 |    128 |  17920 |    0.434 |  1178.56 |    2.393 |    53.50 |
|   512 |    128 |  18432 |    0.475 |  1078.71 |    2.432 |    52.63 |
|   512 |    128 |  18944 |    0.476 |  1074.59 |    2.435 |    52.56 |
|   512 |    128 |  19456 |    0.483 |  1059.64 |    2.466 |    51.91 |
|   512 |    128 |  19968 |    0.488 |  1049.40 |    2.485 |    51.51 |
|   512 |    128 |  20480 |    0.488 |  1049.01 |    2.502 |    51.15 |
|   512 |    128 |  20992 |    0.494 |  1036.95 |    2.542 |    50.35 |
|   512 |    128 |  21504 |    0.500 |  1024.56 |    2.535 |    50.49 |
|   512 |    128 |  22016 |    0.503 |  1017.51 |    2.560 |    50.00 |
|   512 |    128 |  22528 |    0.509 |  1006.46 |    2.570 |    49.81 |
|   512 |    128 |  23040 |    0.524 |   976.26 |    2.596 |    49.31 |
|   512 |    128 |  23552 |    0.517 |   990.80 |    2.617 |    48.91 |
|   512 |    128 |  24064 |    0.523 |   979.07 |    2.628 |    48.71 |
|   512 |    128 |  24576 |    0.486 |  1053.92 |    2.664 |    48.05 |
|   512 |    128 |  25088 |    0.489 |  1046.91 |    2.684 |    47.70 |
|   512 |    128 |  25600 |    0.520 |   984.47 |    2.704 |    47.34 |
|   512 |    128 |  26112 |    0.498 |  1027.80 |    2.747 |    46.59 |
|   512 |    128 |  26624 |    0.503 |  1017.92 |    2.762 |    46.34 |
|   512 |    128 |  27136 |    0.509 |  1006.38 |    2.794 |    45.81 |
|   512 |    128 |  27648 |    0.514 |   995.15 |    2.814 |    45.49 |
|   512 |    128 |  28160 |    0.518 |   987.73 |    2.837 |    45.12 |
|   512 |    128 |  28672 |    0.528 |   970.19 |    2.853 |    44.87 |
|   512 |    128 |  29184 |    0.531 |   965.04 |    2.871 |    44.58 |
|   512 |    128 |  29696 |    0.535 |   957.76 |    2.900 |    44.13 |
|   512 |    128 |  30208 |    0.533 |   961.28 |    2.910 |    43.99 |
|   512 |    128 |  30720 |    0.540 |   948.50 |    2.944 |    43.47 |
|   512 |    128 |  31232 |    0.541 |   946.85 |    2.956 |    43.30 |
|   512 |    128 |  31744 |    0.542 |   943.99 |    2.987 |    42.85 |
|   512 |    128 |  32256 |    0.550 |   930.73 |    3.007 |    42.56 |

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-04-29** at **13:57:33**:<br>

Anyone who has the horse power to run Qwen3-235B-A22B, please feel free to add your results to this discussion.

> 👤 **ubergarm** replied the **2025-04-29** at **16:30:10**:<br>
> I'm away from home but frantically trying to remote into a server I just got access too again and cook up a good Qwen3-235B-A22B mix for my home 3090TI 24GB VRAM + 96GB RAM system which is about the limit of common AM5 gaming rigs (with the faster and more supported 2x DIMM configuration).
> 
> Any particular reason you chose `IQ4_XS` for the experts over `IQ4_K` (possibly GPU inference speed?).
> 
> I haven't finished yet but my very rough WIP custom quantize script so far is:
> <details>
> 
> <summary>Very rough ik_llama.cpp custom quantize script</summary>
> 
> ```bash
> #!/usr/bin/env bash
> 
> custom="
> #token_embd.weight - [ 4096, 151936,     1,     1], type =   bf16, Using custom type q8_0 for tensor token_embd.weight
> #blk.1.ffn_gate_inp.weight - [ 4096,   128,     1,     1], type =    f32, size =    2.000 MB
> #blk.1.attn_k_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
> #blk.1.attn_q_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
> #blk.1.attn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
> #blk.1.ffn_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
> 
> #blk.1.attn_k.weight - [ 4096,   512,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.1.attn_k.weight
> #blk.1.attn_q.weight - [ 4096,  8192,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.1.attn_q.weight
> #blk.1.attn_v.weight - [ 4096,   512,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.1.attn_v.weight
> #blk.1.attn_output.weight - [ 8192,  4096,     1,     1], type =   bf16, Using custom type q8_0 for tensor blk.1.attn_output.weight
> 
> #blk.1.ffn_down_exps.weight - [ 1536,  4096,   128,     1], type =   bf16, Using custom type q8_0 for tensor blk.1.ffn_down_exps.weight
> #blk.1.ffn_gate_exps.weight - [ 4096,  1536,   128,     1], type =   bf16, Using custom type q8_0 for tensor blk.1.ffn_gate_exps.weight
> #blk.1.ffn_up_exps.weight - [ 4096,  1536,   128,     1], type =   bf16, Using custom type q8_0 for tensor blk.1.ffn_up_exps.weight
> 
> #output_norm.weight - [ 4096,     1,     1,     1], type =    f32, size =    0.016 MB
> 
> # Token embedding
> token_embd\.weight=q8_0
> 
> # Attention
> blk\..*\.attn_k.*=iq6_k
> blk\..*\.attn_q.*=iq4_k
> blk\..*\.attn_v.*=iq6_k
> blk\..*\.attn_output.*=iq4_k
> 
> # Experts
> blk\..*\.ffn_down_exps\.weight=iq4_k
> blk\..*\.ffn_(gate|up)_exps\.weight=iq3_k
> "
> 
> custom=$(
>   echo "$custom" | grep -v '^#' | \
>   sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
> )
> 
>     #--token-embedding-type q8_0 \
>     #--output-tensor-type q8_0 \
> ./build/bin/llama-quantize \
>     --custom-q "$custom" \
>     --imatrix /mnt/raid/models/ubergarm/Qwen3-235B-A22B-GGUF/imatrix-Qwen3-235B-A22B.dat \
>     /mnt/raid/models/Qwen/Qwen3-235B-A22B/Qwen3-235B-A22B-BF16-00001-of-00011.gguf \
>     /mnt/raid/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K.gguf \
>     IQ3_K \
>     24
> ```
> 
> </details>
> 
> Did you bother to make an imatrix for your quant, and if so, were you able to activate enough experts with your imatrix corpus text? Thanks again, exciting times with Qwen3 MoE out and wondering if R2 is around the corner haha...
> 
> 👤 **ikawrakow** replied the **2025-04-29** at **16:34:39**:<br>
> > Any particular reason you chose IQ4_XS for the experts over IQ4_K (possibly GPU inference speed?).
> 
> I wanted to have a quantized model that I can run with `ik_llama.cpp` and with `llama.cpp` so we have a fair performance comparison.
> 
> I'm playing with some quantization recipes for Qwen3-30B-A3B. I'll post the results tomorrow, maybe that can be useful to you for "cooking" the  Qwen3-235B-A22B quants.
> 
> 👤 **Gaolingx** replied the **2025-05-06** at **13:15:17**:<br>
> I run Qwen3-235B-A22B on my pc(#385 ), but the performance not better, might the memory performance of RAM is too slow...

---

👤 **ubergarm** replied the **2025-04-30** at **04:45:24**:<br>

Just "cooked" my first `ik_llama.cpp` exclusive experimental quant and uploaded to [huggingface ubergarm/Qwen3-235B-A22B-GGUF](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF). Just tried a benchmark on my local gaming rig as it just finished downloading. Hybrid GPU+CPU inferencing with about 12 ffn layers on GPU and the rest repacked on CPU. *Barely* fits in VRAM+RAM (had to close my browser haha).

![qwen3-moe-troll-rig](https://github.com/user-attachments/assets/c67e4e62-c645-4e01-8b72-0b98180b994c)

Looks pretty good! Only other somewhat comparable benchmark I've seen is from latest [ktransformers v0.3 on a rig with better GPU and more RAM](https://www.reddit.com/r/LocalLLaMA/comments/1ka94qx/qwen_3_ktransformers_03_amx_ai_workstationpc/).

<details>

<summary>👈 Logs</summary>

```
model=/mnt/ai/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf

CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
  --model "$model" \
  -fa \
  -ctk q8_0 -ctv q8_0 \
  -c 32768 \
  -fmoe \
  -amb 512 \
  -rtr \
  -ot blk\.1[2-9]\.ffn.*=CPU \
  -ot blk\.[2-8][0-9]\.ffn.*=CPU \
  -ot blk\.9[0-3]\.ffn.*=CPU \
  -ngl 99 \
  --threads 16

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from /mnt/ai/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 139
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  37:                                   split.no u16              = 0
llama_model_loader: - kv  38:                                split.count u16              = 3
llama_model_loader: - kv  39:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type iq3_k:  188 tensors
llama_model_loader: - type iq4_k:   94 tensors
llama_model_loader: - type iq6_k:  376 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 106.830 GiB (3.903 BPW)
llm_load_print_meta: repeating layers = 105.598 GiB (3.879 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
Tensor blk.12.ffn_norm.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_norm.weight buffer type overriden to CPU
.
.
.
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_norm.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 89709.28 MiB
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size = 19053.73 MiB
....................................................................................................
============ Repacked 246 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3196.05 MiB
llama_new_context_with_model: KV self size  = 3196.00 MiB, K (q8_0): 1598.00 MiB, V (q8_0): 1598.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   312.75 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   128.01 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 330

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 16, n_threads_batch = 16
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.165 |   122.94 |   11.925 |    10.73 |
|   512 |    128 |    512 |    3.848 |   133.06 |   12.031 |    10.64 |
|   512 |    128 |   1024 |    3.581 |   142.97 |   12.163 |    10.52 |
|   512 |    128 |   1536 |    3.631 |   140.99 |   12.343 |    10.37 |
|   512 |    128 |   2048 |    3.622 |   141.36 |   12.491 |    10.25 |
|   512 |    128 |   2560 |    3.631 |   140.99 |   12.677 |    10.10 |
|   512 |    128 |   3072 |    3.657 |   140.02 |   12.859 |     9.95 |
|   512 |    128 |   3584 |    3.667 |   139.63 |   13.039 |     9.82 |
|   512 |    128 |   4096 |    3.694 |   138.61 |   13.226 |     9.68 |
|   512 |    128 |   4608 |    3.710 |   138.00 |   13.399 |     9.55 |
|   512 |    128 |   5120 |    3.719 |   137.67 |   13.587 |     9.42 |
|   512 |    128 |   5632 |    3.773 |   135.69 |   13.767 |     9.30 |
|   512 |    128 |   6144 |    3.756 |   136.32 |   13.936 |     9.18 |
|   512 |    128 |   6656 |    3.776 |   135.59 |   14.103 |     9.08 |
|   512 |    128 |   7168 |    3.796 |   134.88 |   14.277 |     8.97 |
|   512 |    128 |   7680 |    3.804 |   134.60 |   14.473 |     8.84 |
|   512 |    128 |   8192 |    3.879 |   132.00 |   14.638 |     8.74 |
|   512 |    128 |   8704 |    3.849 |   133.02 |   14.847 |     8.62 |
|   512 |    128 |   9216 |    3.929 |   130.31 |   15.027 |     8.52 |
|   512 |    128 |   9728 |    3.943 |   129.84 |   15.216 |     8.41 |
|   512 |    128 |  10240 |    3.908 |   131.02 |   15.385 |     8.32 |
|   512 |    128 |  10752 |    3.923 |   130.51 |   15.560 |     8.23 |
|   512 |    128 |  11264 |    3.935 |   130.12 |   15.741 |     8.13 |
|   512 |    128 |  11776 |    3.982 |   128.59 |   15.695 |     8.16 |
|   512 |    128 |  12288 |    3.971 |   128.94 |   15.602 |     8.20 |
|   512 |    128 |  12800 |    3.982 |   128.58 |   15.740 |     8.13 |
|   512 |    128 |  13312 |    3.993 |   128.22 |   15.901 |     8.05 |
|   512 |    128 |  13824 |    4.019 |   127.40 |   16.079 |     7.96 |
|   512 |    128 |  14336 |    4.044 |   126.62 |   16.265 |     7.87 |
|   512 |    128 |  14848 |    4.056 |   126.23 |   16.399 |     7.81 |
|   512 |    128 |  15360 |    4.070 |   125.80 |   16.582 |     7.72 |
|   512 |    128 |  15872 |    4.114 |   124.46 |   16.754 |     7.64 |
|   512 |    128 |  16384 |    4.101 |   124.86 |   16.899 |     7.57 |
|   512 |    128 |  16896 |    4.120 |   124.26 |   17.061 |     7.50 |
|   512 |    128 |  17408 |    4.148 |   123.43 |   17.219 |     7.43 |
|   512 |    128 |  17920 |    4.170 |   122.79 |   17.386 |     7.36 |
|   512 |    128 |  18432 |    4.183 |   122.41 |   17.559 |     7.29 |
|   512 |    128 |  18944 |    4.212 |   121.55 |   17.744 |     7.21 |
|   512 |    128 |  19456 |    4.222 |   121.26 |   17.925 |     7.14 |
|   512 |    128 |  19968 |    4.250 |   120.48 |   18.072 |     7.08 |
|   512 |    128 |  20480 |    4.253 |   120.38 |   18.233 |     7.02 |
|   512 |    128 |  20992 |    4.318 |   118.57 |   18.365 |     6.97 |
|   512 |    128 |  21504 |    4.289 |   119.38 |   18.574 |     6.89 |
|   512 |    128 |  22016 |    4.310 |   118.79 |   18.722 |     6.84 |
|   512 |    128 |  22528 |    4.337 |   118.05 |   18.884 |     6.78 |
|   512 |    128 |  23040 |    4.349 |   117.72 |   19.071 |     6.71 |
|   512 |    128 |  23552 |    4.361 |   117.40 |   19.233 |     6.66 |
|   512 |    128 |  24064 |    4.459 |   114.83 |   19.375 |     6.61 |
|   512 |    128 |  24576 |    4.396 |   116.47 |   19.506 |     6.56 |
|   512 |    128 |  25088 |    4.418 |   115.90 |   19.668 |     6.51 |
|   512 |    128 |  25600 |    4.432 |   115.53 |   19.840 |     6.45 |
|   512 |    128 |  26112 |    4.450 |   115.06 |   20.016 |     6.39 |
|   512 |    128 |  26624 |    4.464 |   114.70 |   20.157 |     6.35 |
|   512 |    128 |  27136 |    4.484 |   114.17 |   20.332 |     6.30 |
|   512 |    128 |  27648 |    4.502 |   113.72 |   20.479 |     6.25 |
|   512 |    128 |  28160 |    4.532 |   112.96 |   20.657 |     6.20 |
|   512 |    128 |  28672 |    4.534 |   112.92 |   20.814 |     6.15 |
|   512 |    128 |  29184 |    4.561 |   112.26 |   20.982 |     6.10 |
|   512 |    128 |  29696 |    4.565 |   112.16 |   21.138 |     6.06 |
|   512 |    128 |  30208 |    4.579 |   111.82 |   21.284 |     6.01 |
|   512 |    128 |  30720 |    4.614 |   110.97 |   21.457 |     5.97 |
|   512 |    128 |  31232 |    4.628 |   110.64 |   21.709 |     5.90 |
|   512 |    128 |  31744 |    4.647 |   110.17 |   21.866 |     5.85 |
|   512 |    128 |  32256 |    4.669 |   109.66 |   21.961 |     5.83 |

</details>


---

In the mean time, I ran a quick comparison of the Q8_0 on the remote threadripper pro 24 core using a single RTX A6000 48GB VRAM GPU and offloading the rest to CPU for a somewhat similar "hybrid inference" test.

Note that for some reason `ik_llama.cpp` could offload one additional `ffn` layer than mainline `llama.cpp` in this test. I didn't go back and re-run the test by reducing the layers by one on ik so it isn't technically *exactly* the same configuration but close enough for tonight!

![qwen3-moe](https://github.com/user-attachments/assets/30ea3559-d86f-4167-95f9-6ed52d0c4435)

<details>

<summary>👈 Logs</summary>

# ik_llama.cpp
```
model=/mnt/raid/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q8_0.gguf

# Offload 48GB onto single RTX A6000 VRAM
CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
  --model "$model" \
  -fa \
  -ctk f16 -ctv f16 \
  -c 32768 \
  -fmoe \
  -amb 512 \
  -rtr \
  -ot blk\.1[4-9]\.ffn.*=CPU \
  -ot blk\.[2-8][0-9]\.ffn.*=CPU \
  -ot blk\.9[0-3]\.ffn.*=CPU \
  -ngl 99 \
  --threads 24

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
llama_model_loader: loaded meta data with 33 key-value pairs and 1131 tensors from /mnt/raid/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q8_0
.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 7
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:  660 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 232.769 GiB (8.505 BPW)
llm_load_print_meta: repeating layers = 231.538 GiB (8.505 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
Tensor blk.14.ffn_norm.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_norm.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_norm.weight buffer type overriden to CPU
.
.
.
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 196001.25 MiB
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size = 41723.89 MiB
....................................................................................................
============ Repacked 240 tensors
llama_new_context_with_model: n_ctx      = 32768
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  6016.00 MiB
llama_new_context_with_model: KV self size  = 6016.00 MiB, K (f16): 3008.00 MiB, V (f16): 3008.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   312.75 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   128.01 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 322

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 24, n_threads_batch = 24

$ numastat -mp $(pidof llama-sweep-bench)
                          Node 0           Total                                                                                       00:10:39 [2/27]
                 --------------- ---------------
                 --------------- ---------------
MemTotal               257213.74       257213.74
MemFree                  1088.58         1088.58
MemUsed                256125.16       256125.16
SwapCached                 27.07           27.07
Active                  70427.99        70427.99
Inactive               181810.73       181810.73
Active(anon)            70360.04        70360.04
Inactive(anon)         126793.92       126793.92
Active(file)               67.95           67.95
Inactive(file)          55016.81        55016.81
Unevictable                 6.03            6.03
Mlocked                     0.02            0.02
Dirty                       0.18            0.18
Writeback                   0.00            0.00
FilePages               55889.19        55889.19
Mapped                   1024.76         1024.76
AnonPages              196380.88       196380.88
Shmem                     776.73          776.73
KernelStack                16.69           16.69
PageTables                407.07          407.07
SecPageTables             632.02          632.02
NFS_Unstable                0.00            0.00
Bounce                      0.00            0.00
WritebackTmp                0.00            0.00
Slab                     1134.24         1134.24
SReclaimable              633.10          633.10
SUnreclaim                501.14          501.14
AnonHugePages               0.00            0.00
ShmemHugePages              0.00            0.00
ShmemPmdMapped              0.00            0.00
FileHugePages               0.00            0.00
FilePmdMapped               0.00            0.00
HugePages_Total             0.00            0.00
HugePages_Free              0.00            0.00
HugePages_Surp              0.00            0.00
KReclaimable              633.10          633.10
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.115 |   164.36 |   11.715 |    10.93 |
|   512 |    128 |    512 |    3.086 |   165.89 |   11.837 |    10.81 |
|   512 |    128 |   1024 |    3.130 |   163.59 |   12.006 |    10.66 |
|   512 |    128 |   1536 |    3.106 |   164.85 |   12.066 |    10.61 |
|   512 |    128 |   2048 |    3.306 |   154.88 |   12.210 |    10.48 |
|   512 |    128 |   2560 |    3.346 |   153.03 |   12.307 |    10.40 |
|   512 |    128 |   3072 |    3.272 |   156.46 |   12.439 |    10.29 |
|   512 |    128 |   3584 |    3.170 |   161.52 |   12.523 |    10.22 |
|   512 |    128 |   4096 |    3.215 |   159.23 |   12.683 |    10.09 |
|   512 |    128 |   4608 |    3.222 |   158.91 |   12.732 |    10.05 |
|   512 |    128 |   5120 |    3.391 |   150.98 |   12.896 |     9.93 |
|   512 |    128 |   5632 |    3.343 |   153.18 |   12.943 |     9.89 |
|   512 |    128 |   6144 |    3.275 |   156.34 |   13.115 |     9.76 |
|   512 |    128 |   6656 |    3.280 |   156.09 |   13.241 |     9.67 |
|   512 |    128 |   7168 |    3.305 |   154.90 |   13.354 |     9.59 |
|   512 |    128 |   7680 |    3.328 |   153.83 |   13.450 |     9.52 |
|   512 |    128 |   8192 |    3.341 |   153.24 |   13.589 |     9.42 |
|   512 |    128 |   8704 |    3.365 |   152.16 |   13.692 |     9.35 |
|   512 |    128 |   9216 |    3.382 |   151.37 |   13.821 |     9.26 |
|   512 |    128 |   9728 |    3.395 |   150.80 |   13.924 |     9.19 |
|   512 |    128 |  10240 |    3.417 |   149.82 |   14.069 |     9.10 |
|   512 |    128 |  10752 |    3.491 |   146.64 |   14.153 |     9.04 |
|   512 |    128 |  11264 |    3.460 |   147.96 |   14.279 |     8.96 |
|   512 |    128 |  11776 |    3.478 |   147.21 |   14.367 |     8.91 |
|   512 |    128 |  12288 |    3.501 |   146.23 |   14.506 |     8.82 |
|   512 |    128 |  12800 |    3.729 |   137.29 |   14.588 |     8.77 |
|   512 |    128 |  13312 |    3.532 |   144.94 |   14.600 |     8.77 |
|   512 |    128 |  13824 |    3.555 |   144.03 |   14.732 |     8.69 |
|   512 |    128 |  14336 |    3.574 |   143.25 |   14.809 |     8.64 |
|   512 |    128 |  14848 |    3.596 |   142.39 |   14.981 |     8.54 |
|   512 |    128 |  15360 |    3.613 |   141.72 |   15.042 |     8.51 |
|   512 |    128 |  15872 |    3.634 |   140.91 |   15.220 |     8.41 |
|   512 |    128 |  16384 |    3.765 |   135.98 |   15.266 |     8.38 |
|   512 |    128 |  16896 |    3.671 |   139.47 |   15.390 |     8.32 |
|   512 |    128 |  17408 |    3.687 |   138.86 |   15.519 |     8.25 |
|   512 |    128 |  17920 |    3.703 |   138.25 |   15.617 |     8.20 |
|   512 |    128 |  18432 |    3.732 |   137.19 |   15.891 |     8.05 |
|   512 |    128 |  18944 |    3.810 |   134.40 |   15.866 |     8.07 |
|   512 |    128 |  19456 |    3.805 |   134.57 |   15.952 |     8.02 |
|   512 |    128 |  19968 |    3.812 |   134.33 |   16.093 |     7.95 |
|   512 |    128 |  20480 |    3.808 |   134.44 |   16.192 |     7.90 |
|   512 |    128 |  20992 |    3.824 |   133.89 |   16.340 |     7.83 |
|   512 |    128 |  21504 |    3.992 |   128.26 |   16.427 |     7.79 |
|   512 |    128 |  22016 |    3.870 |   132.29 |   16.546 |     7.74 |
|   512 |    128 |  22528 |    3.890 |   131.62 |   16.680 |     7.67 |
|   512 |    128 |  23040 |    4.018 |   127.41 |   16.809 |     7.62 |
|   512 |    128 |  23552 |    3.928 |   130.34 |   16.909 |     7.57 |
|   512 |    128 |  24064 |    3.955 |   129.47 |   17.031 |     7.52 |
|   512 |    128 |  24576 |    3.976 |   128.77 |   17.144 |     7.47 |
|   512 |    128 |  25088 |    3.993 |   128.23 |   17.331 |     7.39 |
|   512 |    128 |  25600 |    4.004 |   127.88 |   17.475 |     7.32 |
|   512 |    128 |  26112 |    4.026 |   127.17 |   17.515 |     7.31 |
|   512 |    128 |  26624 |    4.049 |   126.44 |   17.693 |     7.23 |
|   512 |    128 |  27136 |    4.074 |   125.68 |   17.808 |     7.19 |
|   512 |    128 |  27648 |    4.132 |   123.92 |   17.931 |     7.14 |
|   512 |    128 |  28160 |    4.098 |   124.94 |   18.083 |     7.08 |
|   512 |    128 |  28672 |    4.116 |   124.40 |   18.200 |     7.03 |
|   512 |    128 |  29184 |    4.137 |   123.75 |   18.314 |     6.99 |
|   512 |    128 |  29696 |    4.155 |   123.21 |   18.461 |     6.93 |
|   512 |    128 |  30208 |    4.304 |   118.95 |   18.597 |     6.88 |
|   512 |    128 |  30720 |    4.233 |   120.95 |   18.717 |     6.84 |
|   512 |    128 |  31232 |    4.306 |   118.91 |   18.847 |     6.79 |
|   512 |    128 |  31744 |    4.232 |   120.97 |   18.987 |     6.74 |
|   512 |    128 |  32256 |    4.288 |   119.39 |   19.105 |     6.70 |

## llama.cpp
```
model=/mnt/raid/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q8_0.gguf

CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
  --no-mmap \
  --model "$model" \
  -fa \
  -ctk f16 -ctv f16 \
  -c 32768 \
  -ot blk\.1[3-9]\.ffn.*=CPU \
  -ot blk\.[2-8][0-9]\.ffn.*=CPU \
  -ot blk\.9[0-3]\.ffn.*=CPU \
  -ngl 99 \
  --threads 24

ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no                                                                                          23:49:33 [92/1809]
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
build: 5192 (e59a5f1e) with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA RTX A6000) - 48267 MiB free
llama_model_loader: loaded meta data with 33 key-value pairs and 1131 tensors from /mnt/raid/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q8_0
.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 7
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:  660 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q8_0
print_info: file size   = 232.77 GiB (8.51 BPW)
load: special tokens cache size = 26
load: token to piece cache size = 0.9311 MB
print_info: arch             = qwen3moe
print_info: vocab_only       = 0
print_info: n_ctx_train      = 40960
print_info: n_embd           = 4096
print_info: n_layer          = 94
print_info: n_head           = 64
print_info: n_head_kv        = 4
print_info: n_rot            = 128
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 16
print_info: n_embd_k_gqa     = 512
print_info: n_embd_v_gqa     = 512
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 12288
print_info: n_expert         = 128
print_info: n_expert_used    = 8
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 1000000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 40960
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 235B.A22B
print_info: model params     = 235.09 B
print_info: general.name     = Qwen3 235B A22B
print_info: n_ff_exp         = 1536
print_info: vocab type       = BPE
print_info: n_vocab          = 151936
print_info: n_merges         = 151387
print_info: BOS token        = 151643 '<|endoftext|>'
print_info: EOS token        = 151645 '<|im_end|>'
print_info: EOT token        = 151645 '<|im_end|>'
print_info: PAD token        = 151643 '<|endoftext|>'
print_info: LF token         = 198 'Ċ'
print_info: FIM PRE token    = 151659 '<|fim_prefix|>'
print_info: FIM SUF token    = 151661 '<|fim_suffix|>'
print_info: FIM MID token    = 151660 '<|fim_middle|>'
print_info: FIM PAD token    = 151662 '<|fim_pad|>'
print_info: FIM REP token    = 151663 '<|repo_name|>'
print_info: FIM SEP token    = 151664 '<|file_sep|>'
print_info: EOG token        = 151643 '<|endoftext|>'
print_info: EOG token        = 151645 '<|im_end|>'
print_info: EOG token        = 151662 '<|fim_pad|>'
print_info: EOG token        = 151663 '<|repo_name|>'
print_info: EOG token        = 151664 '<|file_sep|>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = false)
load_tensors: offloading 94 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 95/95 layers to GPU
load_tensors:    CUDA_Host model buffer size =   630.59 MiB
load_tensors:        CUDA0 model buffer size = 39273.87 MiB
load_tensors:          CPU model buffer size = 198451.27 MiB
....................................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 32768
llama_context: n_ctx_per_seq = 32768
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 1
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (32768) < n_ctx_train (40960) -- the full capacity of the model will not be utilized
llama_context:  CUDA_Host  output buffer size =     0.58 MiB
init: kv_size = 32768, offload = 1, type_k = 'f16', type_v = 'f16', n_layer = 94, can_shift = 1
init:      CUDA0 KV buffer size =  6016.00 MiB
llama_context: KV self size  = 6016.00 MiB, K (f16): 3008.00 MiB, V (f16): 3008.00 MiB
llama_context:      CUDA0 compute buffer size =  1024.00 MiB
llama_context:  CUDA_Host compute buffer size =    72.01 MiB
llama_context: graph nodes  = 5741
llama_context: graph splits = 407 (with bs=512), 164 (with bs=1)
common_init_from_params: setting dry_penalty_last_n to ctx_size = 32768
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)

system_info: n_threads = 24 (n_threads_batch = 24) / 48 | CUDA : ARCHS = 860 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : SSE3 = 1 | SSSE3 = 1
 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | LLAMAFILE = 1 | OPENMP = 1
| AARCH64_REPACK = 1 |

main: n_kv_max = 32768, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 24, n_threads_batch = 24

$ numastat -mp $(pidof llama-sweep-bench)
                          Node 0           Total                                                                                       00:10:39 [2/27]
                 --------------- ---------------
MemTotal               257213.74       257213.74
MemFree                  3319.93         3319.93
MemUsed                253893.81       253893.81
SwapCached                 27.97           27.97
Active                  73301.44        73301.44
Inactive               176693.16       176693.16
Active(anon)            73109.92        73109.92
Inactive(anon)         126449.67       126449.67
Active(file)              191.52          191.52
Inactive(file)          50243.50        50243.50
Unevictable                 6.03            6.03
Mlocked                     0.02            0.02
Dirty                       0.17            0.17
Writeback                   0.00            0.00
FilePages               51183.76        51183.76
Mapped                    972.10          972.10
AnonPages              198841.57       198841.57
Shmem                     720.74          720.74
KernelStack                16.81           16.81
PageTables                411.71          411.71
SecPageTables             632.02          632.02
NFS_Unstable                0.00            0.00
Bounce                      0.00            0.00
WritebackTmp                0.00            0.00
Slab                     1134.53         1134.53
SReclaimable              633.56          633.56
SUnreclaim                500.96          500.96
AnonHugePages               0.00            0.00
ShmemHugePages              0.00            0.00
ShmemPmdMapped              0.00            0.00
FileHugePages               0.00            0.00
FilePmdMapped               0.00            0.00
HugePages_Total             0.00            0.00
HugePages_Free              0.00            0.00
HugePages_Surp              0.00            0.00
KReclaimable              633.56          633.56
```

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    9.319 |    54.94 |   12.265 |    10.44 |
|   512 |    128 |    512 |    9.242 |    55.40 |   12.236 |    10.46 |
|   512 |    128 |   1024 |    9.257 |    55.31 |   12.202 |    10.49 |
|   512 |    128 |   1536 |    9.277 |    55.19 |   12.141 |    10.54 |
|   512 |    128 |   2048 |    9.296 |    55.08 |   12.236 |    10.46 |
|   512 |    128 |   2560 |    9.284 |    55.15 |   12.161 |    10.53 |
|   512 |    128 |   3072 |    9.305 |    55.02 |   12.266 |    10.44 |
|   512 |    128 |   3584 |    9.303 |    55.04 |   12.309 |    10.40 |
|   512 |    128 |   4096 |    9.334 |    54.85 |   12.233 |    10.46 |
|   512 |    128 |   4608 |    9.324 |    54.91 |   12.263 |    10.44 |
|   512 |    128 |   5120 |    9.350 |    54.76 |   12.256 |    10.44 |
|   512 |    128 |   5632 |    9.339 |    54.83 |   12.357 |    10.36 |
|   512 |    128 |   6144 |    9.364 |    54.68 |   12.363 |    10.35 |
|   512 |    128 |   6656 |    9.364 |    54.68 |   12.471 |    10.26 |
|   512 |    128 |   7168 |    9.393 |    54.51 |   12.375 |    10.34 |
|   512 |    128 |   7680 |    9.390 |    54.53 |   12.451 |    10.28 |
|   512 |    128 |   8192 |    9.406 |    54.44 |   12.435 |    10.29 |
|   512 |    128 |   8704 |    9.413 |    54.39 |   12.409 |    10.31 |
|   512 |    128 |   9216 |    9.424 |    54.33 |   12.417 |    10.31 |
|   512 |    128 |   9728 |    9.430 |    54.29 |   12.528 |    10.22 |
|   512 |    128 |  10240 |    9.440 |    54.24 |   12.564 |    10.19 |
|   512 |    128 |  10752 |    9.461 |    54.12 |   12.872 |     9.94 |
|   512 |    128 |  11264 |    9.448 |    54.19 |   12.627 |    10.14 |
|   512 |    128 |  11776 |    9.474 |    54.04 |   12.575 |    10.18 |
|   512 |    128 |  12288 |    9.478 |    54.02 |   12.578 |    10.18 |
|   512 |    128 |  12800 |    9.484 |    53.99 |   12.630 |    10.13 |
|   512 |    128 |  13312 |    9.475 |    54.04 |   12.623 |    10.14 |
|   512 |    128 |  13824 |    9.498 |    53.91 |   12.609 |    10.15 |
|   512 |    128 |  14336 |    9.501 |    53.89 |   12.627 |    10.14 |
|   512 |    128 |  14848 |    9.513 |    53.82 |   12.640 |    10.13 |
|   512 |    128 |  15360 |    9.520 |    53.78 |   12.698 |    10.08 |
|   512 |    128 |  15872 |    9.534 |    53.70 |   12.695 |    10.08 |
|   512 |    128 |  16384 |    9.542 |    53.66 |   12.827 |     9.98 |
|   512 |    128 |  16896 |    9.544 |    53.64 |   12.812 |     9.99 |
|   512 |    128 |  17408 |    9.567 |    53.52 |   12.850 |     9.96 |
|   512 |    128 |  17920 |    9.570 |    53.50 |   12.933 |     9.90 |
|   512 |    128 |  18432 |    9.579 |    53.45 |   12.841 |     9.97 |
|   512 |    128 |  18944 |    9.579 |    53.45 |   12.829 |     9.98 |
|   512 |    128 |  19456 |    9.606 |    53.30 |   12.846 |     9.96 |
|   512 |    128 |  19968 |    9.620 |    53.22 |   12.846 |     9.96 |
|   512 |    128 |  20480 |    9.600 |    53.33 |   12.864 |     9.95 |
|   512 |    128 |  20992 |    9.605 |    53.30 |   12.878 |     9.94 |
|   512 |    128 |  21504 |    9.629 |    53.17 |   12.979 |     9.86 |
|   512 |    128 |  22016 |    9.644 |    53.09 |   13.079 |     9.79 |
|   512 |    128 |  22528 |    9.656 |    53.03 |   12.995 |     9.85 |
|   512 |    128 |  23040 |    9.653 |    53.04 |   13.008 |     9.84 |
|   512 |    128 |  23552 |    9.663 |    52.98 |   13.057 |     9.80 |
|   512 |    128 |  24064 |    9.685 |    52.87 |   13.084 |     9.78 |
|   512 |    128 |  24576 |    9.690 |    52.84 |   13.778 |     9.29 |
|   512 |    128 |  25088 |    9.702 |    52.77 |   13.490 |     9.49 |
|   512 |    128 |  25600 |    9.692 |    52.83 |   13.059 |     9.80 |
|   512 |    128 |  26112 |    9.717 |    52.69 |   13.050 |     9.81 |
|   512 |    128 |  26624 |    9.731 |    52.61 |   13.111 |     9.76 |
|   512 |    128 |  27136 |    9.737 |    52.58 |   13.187 |     9.71 |
|   512 |    128 |  27648 |    9.751 |    52.51 |   13.208 |     9.69 |
|   512 |    128 |  28160 |    9.751 |    52.51 |   13.233 |     9.67 |
|   512 |    128 |  28672 |    9.766 |    52.43 |   13.234 |     9.67 |
|   512 |    128 |  29184 |    9.785 |    52.32 |   13.183 |     9.71 |
|   512 |    128 |  29696 |    9.786 |    52.32 |   13.204 |     9.69 |
|   512 |    128 |  30208 |    9.787 |    52.32 |   13.274 |     9.64 |
|   512 |    128 |  30720 |    9.794 |    52.28 |   13.268 |     9.65 |
|   512 |    128 |  31232 |    9.811 |    52.19 |   13.290 |     9.63 |
|   512 |    128 |  31744 |    9.814 |    52.17 |   13.309 |     9.62 |
|   512 |    128 |  32256 |    9.841 |    52.03 |   13.433 |     9.53 |

</details>

Interestingly I could hear my fans spin up and down periodically every 15 seconds or so as the CPU ramped up and the GPU dropped down a bit. I noticed this more on the Q8_0 test visually with `btop` as the CPU would drop to almost 0 and the GPU would ramp up and oscillate slowly back and forth.

> 👤 **ikawrakow** replied the **2025-04-30** at **06:07:34**:<br>
> > Note that for some reason ik_llama.cpp could offload one additional ffn layer than mainline llama.cpp in this test
> 
> This is because the `ik_llama.cpp` CUDA compute buffer is smaller. This is most likely due to the fused `ffn_up+ffn_gate` op that you get with `-fmoe`. In any case, having 80 instead of 81 MoE experts competed on the CPU will not make a significant difference in performance.
> 
> 👤 **ubergarm** replied the **2025-04-30** at **17:46:53**:<br>
> I don't have access to enough RAM+VRAM currently to run the full `bf16`, so I'm using the `Q8_0` as the baseline for my imatrix data and PPL/KLD.
> 
> <details>
> 
> <summary>👈 PPL and KLD comparisons on two test corpi</summary>
> 
> * `Qwen/Qwen3-235B-A22B/Qwen3-235B-A22B-BF16-00001-of-00011.gguf`
>   - 438GiB
>   - TODO
> * `ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q8_0`
>   - 233GiB
>   - Final estimate: PPL = 5.3141 +/- 0.03321 `wiki.test.raw`
>   - Final estimate: PPL = 11.7194 +/- 0.07212 `ubergarm-kld-test-corpus.txt`
> * [ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K.gguf](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF?show_file_info=Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf)
>   - 107GiB
>   - Final estimate: PPL = 5.4403 +/- 0.03421 `wiki.test.raw`
>   - Mean PPL(Q)                   :  11.788282 ±   0.072648 `ubergarm-kld-test-corpus.txt`
>   - ====== KL divergence statistics ======
>   - Mean    KLD:   0.014594 ±   0.000064
>   - Maximum KLD:   2.906263
>   - 99.9%   KLD:   0.296680
>   - 99.0%   KLD:   0.098368
>   - ====== Token probability statistics ======
>   - Mean    Δp: -0.049 ± 0.006 %
>   - Maximum Δp: 63.764%
>   - 99.9%   Δp: 17.122%
>   - 99.0%   Δp:  8.257%
>   - 95.0%   Δp:  4.175%
>   - 90.0%   Δp:  2.504%
> * [unsloth/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-UD-Q3_K_XL](https://huggingface.co/unsloth/Qwen3-235B-A22B-128K-GGUF?show_file_info=UD-Q3_K_XL%2FQwen3-235B-A22B-UD-Q3_K_XL-00001-of-00003.gguf)
>   - 97GiB
>   - Final estimate: PPL = 5.5695 +/- 0.03524 `wiki-test.raw`
>   - Mean PPL(Q):  11.855173 ±   0.073300 `ubergarm-kld-test-corpus.txt`
>   - ====== KL divergence statistics ======
>   - Mean    KLD:   0.029122 ±   0.000123
>   - Maximum KLD:   5.471307
>   - 99.9%   KLD:   0.543533
>   - 99.0%   KLD:   0.180988
>   - ====== Token probability statistics ======
>   - Mean    Δp: -0.059 ± 0.009 %
>   - Maximum Δp: 64.130%
>   - 99.9%   Δp: 22.421%
>   - 99.0%   Δp: 11.713%
>   - 95.0%   Δp:  5.976%
>   - 90.0%   Δp:  3.649%
> * [lmstudio-community/Qwen_Qwen3-235B-A22B-GGUF](https://huggingface.co/lmstudio-community/Qwen_Qwen3-235B-A22B-GGUF)
>   - *NOTE*: bartowski releases these models quickly for lm studio without imatrix as per their preference
>   - 104GiB
>   - Final estimate: PPL = 5.6582 +/- 0.03584 `wiki-test.raw`
>   - Mean PPL(Q)                   :  11.904309 ±   0.073302 `ubergarm-kld-test-corpus.txt`
>   - ====== KL divergence statistics ======
>   - Mean    KLD:   0.036266 ±   0.000140
>   - Maximum KLD:   8.358958
>   - 99.9%   KLD:   0.628216
>   - 99.0%   KLD:   0.219563
>   - ====== Token probability statistics ======
>   - Mean    Δp: -0.284 ± 0.010 %
>   - Maximum Δp: 77.349%
>   - 99.9%   Δp: 24.126%
>   - 99.0%   Δp: 12.470%
>   - 95.0%   Δp:  6.267%
>   - 90.0%   Δp:  3.742%
> * [bartowski/Qwen_Qwen3-235B-A22B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-235B-A22B-GGUF)
>   - TODO, waiting for bartowski to finish the small imatrix quants before releasing them all
> 
> </details>
> 
> <details>
> 
> <summary>👈 ubergarm-kld-test-corpus.txt</summary>
> 
> I created a ~1.6MiB plain text test corpus using `whisper-large-v3` audio transcripts of newer episodes of [Rick Archer's *Buddha at the Gas Pump* Podcast and Youtube Channel](https://batgap.com/interviews/interview-content-search/) for which I maintain a searchable full text index [among other channels with similar content](https://search.emptyduck.com).
> 
> The formatting is a little odd as there are no paragraphs and i used `fmt` for line breaks. The thought was that at least this text *probably* hasn't *yet* been used in training or fine-tuning ai models and is different from the corpus I use to generate imatrix data.
> 
> I'd rather not release it publicly easily accessible in full for various reasons, but contact me if you are doing some research or want exact comparisons with my quants (or I could possibly run your quant if I have time). Here is a snippet so you can see what it looks like:
> 
> ```
> $ head ubergarm-kld-test-corpus.txt
> ## The Telepathy Tapes - Dr. Diane Hennacy Powell - Buddha at the Gas Pump Interview
> 
> Another thing that we have anecdotes about is the precognition. So
> we have, for example, a girl on the podcast who had a dream that her
> father slipped on ice.  And they live in Arizona where there's no
> ice. And it happened three weeks later when he was on a business trip.
> He slipped on the ice. And she... She also knew that he'd end up in the
> hospital with a broken hip as a result, which was the case.  So that's
> a really fascinating anecdote and I've heard many like that over the
> years. But once again, to say that you have evidence for precognition
> ```
> 
> </details>

---

👤 **ikawrakow** replied the **2025-04-30** at **05:57:12**:<br>

@ubergarm Can you try the attached `sweep_bench.cpp` adaptation for `llama.cpp` instead of your adaptation? Thanks! 

[sweep-bench.cpp.gz](https://github.com/user-attachments/files/19971777/sweep-bench.cpp.gz)

> 👤 **ubergarm** replied the **2025-04-30** at **17:21:50**:<br>
> I compared your `sweep-bench.cpp` adaptation to mainline llama.cpp with [my adaptation](https://github.com/ubergarm/llama.cpp/blob/ug/port-sweep-bench/examples/sweep-bench/sweep-bench.cpp) of @saood06 's code. A couple quick results suggest they are pretty similar for two benchmarks I had run:
> 
> ## bartowski/THUDM_GLM-Z1-32B-0414-IQ4_XS.gguf GQA FA
> 
> ![thud-sweep-mine-vs-iks-adaptation](https://github.com/user-attachments/assets/2326af57-c779-4ea2-afd4-5f401357cca6)
> 
> Running the same and comparing against [this previous data](https://github.com/ikawrakow/ik_llama.cpp/pull/344#issuecomment-2832581799).
> 
> ## Qwen3-235B-A22B-Q8_0 GQA FA
> 
> ![qwen3-moe-ik-vs-ug-sweep-adaptation](https://github.com/user-attachments/assets/4dd766e7-85d9-4615-8a38-2d994528e21a)
> ^ title is wrong, this big model was on the thread ripper pro with RTX A6000 oops
> 
> Running the same and comparing against the above chart.
> 
> ## Conclusion
> 
> The general trends seem to hold, but your implementation seems a bit more consistent without the occasional dips unless that was just some noise or me doing something else on the machine. I'll use your adaptation going forward just to keep it as similar as possible with your comparisons. Thanks!

---

👤 **ikawrakow** replied the **2025-04-30** at **14:04:50**:<br>

OK, after thinking more about this, I can see why mainline has a better large context TG performance on CUDA for Qwen3-235B-A22B (and previously noted for LLaMA-4): these models have a quite large GQA factor, and I'm still using the old CUDA FA implementation that did not take advantage of that. Improved GQA FA performance was added in [this mainline PR](https://github.com/ggml-org/llama.cpp/pull/12014).

`ik_llama.cpp` does take advantage of GQA in the CPU FA implementation. Given the above results, it is clear that it is time to do the same for CUDA. I have two options:
* Pickup the mainline PR (but heavy adaptation will be required as things have diverged a lot, and mainline FA does not support different K and V head sizes as required for DeepSeek models)
* Finally sit down and write my own CUDA FA implementation

> 👤 **ubergarm** replied the **2025-04-30** at **15:02:22**:<br>
> Interesting, yes, I first noticed this with GLM-4 (which uses GQA) in the [CUDA + Flash Attention case](https://github.com/ikawrakow/ik_llama.cpp/pull/344#issuecomment-2832581799) benchmark.
> 
> I still have the dream of converting an existing GQA architecture model to MLA but the additional fine-tuning required even with a fraction of the original training data seems daunting:
> 
> > The expressiveness of MLA is greater than that of GQA when both have the same size of KV cache.
> > -[TransMLA: Multi-head Latent Attention Is All You Need](https://arxiv.org/html/2502.07864v1)
> 
> But until MLA catches on more across other models, it might make sense to revisit the CUDA FA implementation for GQA, if that is something that interests you. Of course as soon as R2 comes around, this fickle world will jump on the next hype train lmao...
> 
> In the mean-time I'll re-run a couple `llama-sweep-bench` comparisons with your mainline `sweep-bench.cpp` adaptation to confirm or reject my prior benchmarks!
> 
> Thanks!
> 
> 👤 **ikawrakow** replied the **2025-04-30** at **16:14:03**:<br>
> > I still have the dream of converting an existing GQA architecture model to MLA but the additional fine-tuning required even with a fraction of the original training data seems daunting:
> 
> But MLA is not all roses either. It took quite a bit of experimentation to arrive at a meaningful compromise between TG and PP performance. Mainline has a long way to go there (see #354). And then we have this much smaller KV cache, but then we need giant compute buffers to get meaningful performance, so we need to compute self attention in chunks to keep compute memory usage at a reasonable level, so suddenly the compute graph building becomes this huge pile of complications instead of being just a few tens of lines of simple code as ii is for the other models. And then seeing the massive drop in performance with large contexts in your DeepSeek-V3/R1 benchmarks, my guess is that it is still far from optimum.

---

👤 **AesSedai** replied the **2025-05-03** at **00:37:46**:<br>

Hello, @artus-dev and @ubergarm asked me to run some sweeps for Qwen3-235B-A22B. My homelab has a substantial server with a VM in it that has the following allocation:
```
56 threads of a AMD EPYC 9355 (64t total)
512GB of 12 channel DDR5 6000 ECC RAM (768GB total)
2x 24GB Nvidia 3090
```

I've run four sweeps as follows:
```
ik_llama.cpp CPU only
ik_llama.cpp one GPU
llama.cpp CPU only
llama.cpp one GPU
```
Both ik_llama.cpp and llama.cpp were compiled with CUDA and OpenBLAS support.

The sweeps were run with the following quants:
```
ik_llama.cpp: https://huggingface.co/ArtusDev/Qwen3-235B-A22B-GGUF (IQ6_K, ~212GB)
llama.cpp: https://huggingface.co/unsloth/Qwen3-235B-A22B-128K-GGUF (Q6_K, ~193GB)
```

The llama.cpp tests were conducted with the `sweep-bench.cpp` included in https://github.com/ikawrakow/ik_llama.cpp/discussions/357#discussioncomment-12988686

For the GPU tests, I kept the layer offloads identical between the two. This means that were was slightly less GPU VRAM utilization for the llama.cpp test because the model is smaller, but I felt that was the best way to keep the tests as comparable as I could manage:
```
-ot "blk\.(0|1|2|3|4)\.ffn.*=CUDA0"
```

Logs for the runs are as follows:
<details>
<summary>ik_llama.cpp CPU logs</summary>

```
./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf -c 16384 -t 48 -fa -rtr -fmoe -ctk q8_0 -ctv q8_0
llama_model_loader: additional 4 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 39 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Models
llama_model_loader: - kv   3:                         general.size_label str              = 128x10B
llama_model_loader: - kv   4:                            general.license str              = apache-2.0
llama_model_loader: - kv   5:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   6:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                          general.file_type u32              = 142
llama_model_loader: - kv  19:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  20:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  28:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                      quantize.imatrix.file str              = /workspace/ubergarm/imatrix-Qwen3-235...
llama_model_loader: - kv  33:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  34:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  35:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  36:                                   split.no u16              = 0
llama_model_loader: - kv  37:                                split.count u16              = 5
llama_model_loader: - kv  38:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:   96 tensors
llama_model_loader: - type iq6_k:  564 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ6_K - 6.6 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 198.259 GiB (7.244 BPW) 
llm_load_print_meta: repeating layers = 197.028 GiB (7.237 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Models
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.50 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/95 layers to GPU
llm_load_tensors:  CUDA_Host buffer size = 203017.61 MiB
....................................................................................................
============ Repacked 95 tensors
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:  CUDA_Host KV buffer size =  1598.00 MiB
llama_new_context_with_model: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =    88.00 MiB
llama_new_context_with_model:        CPU compute buffer size =   304.75 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   129.01 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 1225

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 48, n_threads_batch = 48

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   32.900 |    15.56 |    8.978 |    14.26 |
|   512 |    128 |    512 |   33.034 |    15.50 |    8.744 |    14.64 |
|   512 |    128 |   1024 |   33.181 |    15.43 |    9.281 |    13.79 |
|   512 |    128 |   1536 |   33.299 |    15.38 |    9.199 |    13.91 |
|   512 |    128 |   2048 |   33.424 |    15.32 |    9.158 |    13.98 |
|   512 |    128 |   2560 |   33.579 |    15.25 |    9.624 |    13.30 |
|   512 |    128 |   3072 |   33.646 |    15.22 |    9.632 |    13.29 |
|   512 |    128 |   3584 |   33.863 |    15.12 |    9.266 |    13.81 |
|   512 |    128 |   4096 |   34.018 |    15.05 |    9.630 |    13.29 |
|   512 |    128 |   4608 |   34.192 |    14.97 |   10.042 |    12.75 |
|   512 |    128 |   5120 |   34.280 |    14.94 |    9.658 |    13.25 |
|   512 |    128 |   5632 |   34.481 |    14.85 |   11.059 |    11.57 |
|   512 |    128 |   6144 |   34.654 |    14.77 |   11.382 |    11.25 |
|   512 |    128 |   6656 |   34.813 |    14.71 |   10.431 |    12.27 |
|   512 |    128 |   7168 |   35.101 |    14.59 |   12.036 |    10.63 |
|   512 |    128 |   7680 |   35.158 |    14.56 |   13.169 |     9.72 |
|   512 |    128 |   8192 |   35.381 |    14.47 |   13.049 |     9.81 |
|   512 |    128 |   8704 |   35.544 |    14.40 |   14.775 |     8.66 |
|   512 |    128 |   9216 |   35.633 |    14.37 |   15.850 |     8.08 |
|   512 |    128 |   9728 |   35.774 |    14.31 |   15.061 |     8.50 |
|   512 |    128 |  10240 |   35.845 |    14.28 |   16.518 |     7.75 |
|   512 |    128 |  10752 |   36.028 |    14.21 |   16.483 |     7.77 |
|   512 |    128 |  11264 |   36.193 |    14.15 |   15.264 |     8.39 |
|   512 |    128 |  11776 |   36.357 |    14.08 |   16.721 |     7.66 |
|   512 |    128 |  12288 |   36.393 |    14.07 |   16.834 |     7.60 |
|   512 |    128 |  12800 |   36.579 |    14.00 |   15.609 |     8.20 |
|   512 |    128 |  13312 |   36.701 |    13.95 |   16.984 |     7.54 |
|   512 |    128 |  13824 |   36.927 |    13.87 |   17.220 |     7.43 |
|   512 |    128 |  14336 |   37.027 |    13.83 |   15.938 |     8.03 |
|   512 |    128 |  14848 |   37.247 |    13.75 |   17.507 |     7.31 |
|   512 |    128 |  15360 |   37.359 |    13.70 |   17.540 |     7.30 |
|   512 |    128 |  15872 |   37.496 |    13.65 |   16.480 |     7.77 |
```

</details>

<details>
<summary>ik_llama.cpp GPU logs</summary>

```
CUDA_VISIBLE_DEVICES="0" ./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf -c 16384 -t 48 -fa -rtr -fmoe -ctk q8_0 -ctv q8_0 -ngl 99 -ot "blk\.(0|1|2|3|4)\.ffn.*=CUDA0" -ot "blk\.(5|6|7|8|9)\.ffn.*=CPU" -ot "blk\.1[0-9]\.ffn.*=CPU" -ot "blk\.[2-9][0-9]\.ffn.*=CPU"
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
llama_model_loader: additional 4 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 39 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Models
llama_model_loader: - kv   3:                         general.size_label str              = 128x10B
llama_model_loader: - kv   4:                            general.license str              = apache-2.0
llama_model_loader: - kv   5:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   6:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                          general.file_type u32              = 142
llama_model_loader: - kv  19:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  20:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  28:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                      quantize.imatrix.file str              = /workspace/ubergarm/imatrix-Qwen3-235...
llama_model_loader: - kv  33:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  34:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  35:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  36:                                   split.no u16              = 0
llama_model_loader: - kv  37:                                split.count u16              = 5
llama_model_loader: - kv  38:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:   96 tensors
llama_model_loader: - type iq6_k:  564 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ6_K - 6.6 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 198.259 GiB (7.244 BPW) 
llm_load_print_meta: repeating layers = 197.028 GiB (7.237 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Models
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_norm.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_norm.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_norm.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_norm.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_norm.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_norm.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_norm.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_norm.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_norm.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_norm.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_norm.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_norm.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_norm.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_norm.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_norm.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_norm.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_norm.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_norm.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_norm.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_norm.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_norm.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_norm.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_norm.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_norm.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_norm.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_norm.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_norm.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_norm.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_norm.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_norm.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_norm.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_norm.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_norm.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_norm.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_norm.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_norm.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_norm.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_norm.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_norm.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_norm.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_norm.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_norm.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_norm.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_norm.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_norm.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_norm.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_norm.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_norm.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_norm.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_norm.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_norm.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_norm.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_norm.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_norm.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_norm.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_norm.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_norm.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_norm.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_norm.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_norm.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_norm.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_norm.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_norm.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_norm.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_norm.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_norm.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_norm.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_norm.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_norm.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_norm.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_norm.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_norm.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_norm.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_norm.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_norm.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_norm.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_norm.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_norm.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_norm.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_norm.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_norm.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_norm.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_norm.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_norm.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_norm.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_norm.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_norm.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 186011.39 MiB
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size = 16375.62 MiB
....................................................................................................
============ Repacked 89 tensors
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  1598.05 MiB
llama_new_context_with_model: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   312.75 MiB
llama_new_context_with_model:        CPU compute buffer size =     8.25 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   120.01 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 358

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 48, n_threads_batch = 48

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.089 |   125.21 |    7.414 |    17.26 |
|   512 |    128 |    512 |    4.120 |   124.26 |    7.778 |    16.46 |
|   512 |    128 |   1024 |    4.077 |   125.58 |    8.250 |    15.51 |
|   512 |    128 |   1536 |    4.123 |   124.17 |   10.897 |    11.75 |
|   512 |    128 |   2048 |    4.181 |   122.45 |   11.487 |    11.14 |
|   512 |    128 |   2560 |    4.193 |   122.12 |   11.506 |    11.12 |
|   512 |    128 |   3072 |    4.197 |   122.01 |   11.770 |    10.88 |
|   512 |    128 |   3584 |    4.249 |   120.50 |   12.058 |    10.62 |
|   512 |    128 |   4096 |    4.316 |   118.64 |   12.234 |    10.46 |
|   512 |    128 |   4608 |    4.337 |   118.06 |   12.299 |    10.41 |
|   512 |    128 |   5120 |    4.331 |   118.23 |   12.540 |    10.21 |
|   512 |    128 |   5632 |    4.380 |   116.91 |   12.850 |     9.96 |
|   512 |    128 |   6144 |    4.413 |   116.03 |   13.086 |     9.78 |
|   512 |    128 |   6656 |    4.416 |   115.93 |   13.052 |     9.81 |
|   512 |    128 |   7168 |    4.462 |   114.75 |   13.409 |     9.55 |
|   512 |    128 |   7680 |    4.477 |   114.36 |   13.776 |     9.29 |
|   512 |    128 |   8192 |    4.505 |   113.66 |   13.847 |     9.24 |
|   512 |    128 |   8704 |    4.499 |   113.81 |   13.971 |     9.16 |
|   512 |    128 |   9216 |    4.494 |   113.93 |   14.251 |     8.98 |
|   512 |    128 |   9728 |    4.489 |   114.06 |   14.196 |     9.02 |
|   512 |    128 |  10240 |    4.470 |   114.53 |   14.242 |     8.99 |
|   512 |    128 |  10752 |    4.491 |   114.01 |   14.250 |     8.98 |
|   512 |    128 |  11264 |    4.521 |   113.25 |   14.597 |     8.77 |
|   512 |    128 |  11776 |    4.568 |   112.08 |   14.801 |     8.65 |
|   512 |    128 |  12288 |    4.562 |   112.23 |   14.969 |     8.55 |
|   512 |    128 |  12800 |    4.581 |   111.78 |   15.320 |     8.36 |
|   512 |    128 |  13312 |    4.582 |   111.73 |   15.368 |     8.33 |
|   512 |    128 |  13824 |    4.598 |   111.35 |   15.639 |     8.18 |
|   512 |    128 |  14336 |    4.619 |   110.84 |   15.904 |     8.05 |
|   512 |    128 |  14848 |    4.639 |   110.38 |   15.952 |     8.02 |
|   512 |    128 |  15360 |    4.649 |   110.14 |   16.225 |     7.89 |
|   512 |    128 |  15872 |    4.663 |   109.79 |   16.326 |     7.84 |
```

</details>


<details>
<summary>llama.cpp CPU logs</summary>

```
./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-128K-GGUF/Q6_K/Qwen3-235B-A22B-128K-Q6_K-00001-of-00004.gguf -c 16384 -t 48 -fa -ctk q8_0 -ctv q8_0
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
build: 5269 (1d36b367) with cc (GCC) 14.2.1 20250110 (Red Hat 14.2.1-7) for x86_64-redhat-linux
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 3090) - 23871 MiB free
llama_model_load_from_file_impl: using device CUDA1 (NVIDIA GeForce RTX 3090) - 23871 MiB free
llama_model_loader: additional 3 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 47 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-128K-GGUF/Q6_K/Qwen3-235B-A22B-128K-Q6_K-00001-of-00004.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-235B-A22B-128K
llama_model_loader: - kv   3:                           general.finetune str              = 128k
llama_model_loader: - kv   4:                           general.basename str              = Qwen3-235B-A22B-128K
llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   6:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   7:                            general.license str              = apache-2.0
llama_model_loader: - kv   8:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   9:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv  10:                   general.base_model.count u32              = 1
llama_model_loader: - kv  11:                  general.base_model.0.name str              = Qwen3 235B A22B
llama_model_loader: - kv  12:          general.base_model.0.organization str              = Qwen
llama_model_loader: - kv  13:              general.base_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv  14:                               general.tags arr[str,2]       = ["unsloth", "text-generation"]
llama_model_loader: - kv  15:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv  16:                    qwen3moe.context_length u32              = 131072
llama_model_loader: - kv  17:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  18:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  19:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  20:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  21:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  22:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  23:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  24:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  25:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  26:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  27:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  28:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  29:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  30:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  31:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  32:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  33:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  34:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  35:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  36:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  37:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  38:               general.quantization_version u32              = 2
llama_model_loader: - kv  39:                          general.file_type u32              = 18
llama_model_loader: - kv  40:                      quantize.imatrix.file str              = Qwen3-235B-A22B-128K-GGUF/imatrix_uns...
llama_model_loader: - kv  41:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-235B-A22B-1...
llama_model_loader: - kv  42:             quantize.imatrix.entries_count i32              = 752
llama_model_loader: - kv  43:              quantize.imatrix.chunks_count i32              = 46
llama_model_loader: - kv  44:                                   split.no u16              = 0
llama_model_loader: - kv  45:                        split.tensors.count i32              = 1131
llama_model_loader: - kv  46:                                split.count u16              = 4
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q6_K:  660 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q6_K
print_info: file size   = 179.75 GiB (6.57 BPW) 
load: special tokens cache size = 26
load: token to piece cache size = 0.9311 MB
print_info: arch             = qwen3moe
print_info: vocab_only       = 0
print_info: n_ctx_train      = 131072
print_info: n_embd           = 4096
print_info: n_layer          = 94
print_info: n_head           = 64
print_info: n_head_kv        = 4
print_info: n_rot            = 128
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 16
print_info: n_embd_k_gqa     = 512
print_info: n_embd_v_gqa     = 512
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 12288
print_info: n_expert         = 128
print_info: n_expert_used    = 8
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 1000000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 131072
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 235B.A22B
print_info: model params     = 235.09 B
print_info: general.name     = Qwen3-235B-A22B-128K
print_info: n_ff_exp         = 1536
print_info: vocab type       = BPE
print_info: n_vocab          = 151936
print_info: n_merges         = 151387
print_info: BOS token        = 151643 '<|endoftext|>'
print_info: EOS token        = 151645 '<|im_end|>'
print_info: EOT token        = 151645 '<|im_end|>'
print_info: PAD token        = 151643 '<|endoftext|>'
print_info: LF token         = 198 'Ċ'
print_info: FIM PRE token    = 151659 '<|fim_prefix|>'
print_info: FIM SUF token    = 151661 '<|fim_suffix|>'
print_info: FIM MID token    = 151660 '<|fim_middle|>'
print_info: FIM PAD token    = 151662 '<|fim_pad|>'
print_info: FIM REP token    = 151663 '<|repo_name|>'
print_info: FIM SEP token    = 151664 '<|file_sep|>'
print_info: EOG token        = 151643 '<|endoftext|>'
print_info: EOG token        = 151645 '<|im_end|>'
print_info: EOG token        = 151662 '<|fim_pad|>'
print_info: EOG token        = 151663 '<|repo_name|>'
print_info: EOG token        = 151664 '<|file_sep|>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 0 repeating layers to GPU
load_tensors: offloaded 0/95 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 47091.25 MiB
load_tensors:   CPU_Mapped model buffer size = 47433.32 MiB
load_tensors:   CPU_Mapped model buffer size = 47377.52 MiB
load_tensors:   CPU_Mapped model buffer size = 42166.10 MiB
....................................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 16384
llama_context: n_ctx_per_seq = 16384
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 1
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (16384) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_context:        CPU  output buffer size =     0.58 MiB
llama_kv_cache_unified: kv_size = 16384, type_k = 'q8_0', type_v = 'q8_0', n_layer = 94, can_shift = 1, padding = 256
llama_kv_cache_unified:        CPU KV buffer size =  1598.00 MiB
llama_kv_cache_unified: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
llama_context:      CUDA0 compute buffer size =   742.00 MiB
llama_context:        CPU compute buffer size =   304.75 MiB
llama_context:  CUDA_Host compute buffer size =    65.01 MiB
llama_context: graph nodes  = 5741
llama_context: graph splits = 1602 (with bs=512), 189 (with bs=1)

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 48, n_threads_batch = 48

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   42.357 |    12.09 |    9.060 |    14.13 |
|   512 |    128 |    512 |   51.197 |    10.00 |   12.240 |    10.46 |
|   512 |    128 |   1024 |   60.284 |     8.49 |   14.398 |     8.89 |
|   512 |    128 |   1536 |   68.365 |     7.49 |   17.737 |     7.22 |
|   512 |    128 |   2048 |   76.989 |     6.65 |   24.649 |     5.19 |
|   512 |    128 |   2560 |   87.495 |     5.85 |   29.569 |     4.33 |
|   512 |    128 |   3072 |   99.493 |     5.15 |   33.176 |     3.86 |
|   512 |    128 |   3584 |  104.887 |     4.88 |   35.395 |     3.62 |
|   512 |    128 |   4096 |  110.847 |     4.62 |   37.481 |     3.42 |
|   512 |    128 |   4608 |  118.194 |     4.33 |   46.298 |     2.76 |
|   512 |    128 |   5120 |  126.544 |     4.05 |   43.575 |     2.94 |
|   512 |    128 |   5632 |  132.354 |     3.87 |   51.306 |     2.49 |
|   512 |    128 |   6144 |  141.580 |     3.62 |   53.846 |     2.38 |
|   512 |    128 |   6656 |  147.841 |     3.46 |   51.455 |     2.49 |
|   512 |    128 |   7168 |  155.069 |     3.30 |   52.843 |     2.42 |
|   512 |    128 |   7680 |  166.590 |     3.07 |   61.982 |     2.07 |
|   512 |    128 |   8192 |  174.021 |     2.94 |   62.082 |     2.06 |
|   512 |    128 |   8704 |  180.649 |     2.83 |   68.306 |     1.87 |
|   512 |    128 |   9216 |  191.221 |     2.68 |   71.603 |     1.79 |
|   512 |    128 |   9728 |  197.848 |     2.59 |   78.050 |     1.64 |
|   512 |    128 |  10240 |  205.342 |     2.49 |   85.140 |     1.50 |
|   512 |    128 |  10752 |  209.842 |     2.44 |   82.100 |     1.56 |
|   512 |    128 |  11264 |  218.246 |     2.35 |   78.315 |     1.63 |
|   512 |    128 |  11776 |  229.003 |     2.24 |   81.961 |     1.56 |
|   512 |    128 |  12288 |  241.294 |     2.12 |   81.073 |     1.58 |
|   512 |    128 |  12800 |  247.041 |     2.07 |   92.054 |     1.39 |
|   512 |    128 |  13312 |  246.231 |     2.08 |   90.119 |     1.42 |
|   512 |    128 |  13824 |  267.642 |     1.91 |   91.823 |     1.39 |
|   512 |    128 |  14336 |  262.708 |     1.95 |   92.070 |     1.39 |
|   512 |    128 |  14848 |  276.199 |     1.85 |   93.608 |     1.37 |
|   512 |    128 |  15360 |  286.268 |     1.79 |   97.714 |     1.31 |
|   512 |    128 |  15872 |  293.752 |     1.74 |   97.181 |     1.32 |
```

</details>

<details>
<summary>llama.cpp GPU logs</summary>

```
CUDA_VISIBLE_DEVICES="0" ./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-128K-GGUF/Q6_K/Qwen3-235B-A22B-128K-Q6_K-00001-of-00004.gguf -c 16384 -t 48 -fa -ctk q8_0 -ctv q8_0 -ngl 99 -ot "blk\.(0|1|2|3|4)\.ffn.*=CUDA0" -ot "blk\.(5|6|7|8|9)\.ffn.*=CPU" -ot "blk\.1[0-9]\.ffn.*=CPU" -ot "blk\.[2-9][0-9]\.ffn.*=CPU"
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
build: 5269 (1d36b367) with cc (GCC) 14.2.1 20250110 (Red Hat 14.2.1-7) for x86_64-redhat-linux
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 3090) - 23871 MiB free
llama_model_loader: additional 3 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 47 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-128K-GGUF/Q6_K/Qwen3-235B-A22B-128K-Q6_K-00001-of-00004.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-235B-A22B-128K
llama_model_loader: - kv   3:                           general.finetune str              = 128k
llama_model_loader: - kv   4:                           general.basename str              = Qwen3-235B-A22B-128K
llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   6:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   7:                            general.license str              = apache-2.0
llama_model_loader: - kv   8:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   9:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv  10:                   general.base_model.count u32              = 1
llama_model_loader: - kv  11:                  general.base_model.0.name str              = Qwen3 235B A22B
llama_model_loader: - kv  12:          general.base_model.0.organization str              = Qwen
llama_model_loader: - kv  13:              general.base_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv  14:                               general.tags arr[str,2]       = ["unsloth", "text-generation"]
llama_model_loader: - kv  15:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv  16:                    qwen3moe.context_length u32              = 131072
llama_model_loader: - kv  17:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  18:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  19:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  20:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  21:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  22:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  23:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  24:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  25:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  26:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  27:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  28:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  29:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  30:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  31:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  32:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  33:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  34:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  35:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  36:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  37:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  38:               general.quantization_version u32              = 2
llama_model_loader: - kv  39:                          general.file_type u32              = 18
llama_model_loader: - kv  40:                      quantize.imatrix.file str              = Qwen3-235B-A22B-128K-GGUF/imatrix_uns...
llama_model_loader: - kv  41:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-235B-A22B-1...
llama_model_loader: - kv  42:             quantize.imatrix.entries_count i32              = 752
llama_model_loader: - kv  43:              quantize.imatrix.chunks_count i32              = 46
llama_model_loader: - kv  44:                                   split.no u16              = 0
llama_model_loader: - kv  45:                        split.tensors.count i32              = 1131
llama_model_loader: - kv  46:                                split.count u16              = 4
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q6_K:  660 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q6_K
print_info: file size   = 179.75 GiB (6.57 BPW) 
load: special tokens cache size = 26
load: token to piece cache size = 0.9311 MB
print_info: arch             = qwen3moe
print_info: vocab_only       = 0
print_info: n_ctx_train      = 131072
print_info: n_embd           = 4096
print_info: n_layer          = 94
print_info: n_head           = 64
print_info: n_head_kv        = 4
print_info: n_rot            = 128
print_info: n_swa            = 0
print_info: n_swa_pattern    = 1
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 16
print_info: n_embd_k_gqa     = 512
print_info: n_embd_v_gqa     = 512
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 12288
print_info: n_expert         = 128
print_info: n_expert_used    = 8
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 1000000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 131072
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 235B.A22B
print_info: model params     = 235.09 B
print_info: general.name     = Qwen3-235B-A22B-128K
print_info: n_ff_exp         = 1536
print_info: vocab type       = BPE
print_info: n_vocab          = 151936
print_info: n_merges         = 151387
print_info: BOS token        = 151643 '<|endoftext|>'
print_info: EOS token        = 151645 '<|im_end|>'
print_info: EOT token        = 151645 '<|im_end|>'
print_info: PAD token        = 151643 '<|endoftext|>'
print_info: LF token         = 198 'Ċ'
print_info: FIM PRE token    = 151659 '<|fim_prefix|>'
print_info: FIM SUF token    = 151661 '<|fim_suffix|>'
print_info: FIM MID token    = 151660 '<|fim_middle|>'
print_info: FIM PAD token    = 151662 '<|fim_pad|>'
print_info: FIM REP token    = 151663 '<|repo_name|>'
print_info: FIM SEP token    = 151664 '<|file_sep|>'
print_info: EOG token        = 151643 '<|endoftext|>'
print_info: EOG token        = 151645 '<|im_end|>'
print_info: EOG token        = 151662 '<|fim_pad|>'
print_info: EOG token        = 151663 '<|repo_name|>'
print_info: EOG token        = 151664 '<|file_sep|>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 94 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 95/95 layers to GPU
load_tensors:        CUDA0 model buffer size = 15191.95 MiB
load_tensors:   CPU_Mapped model buffer size = 46604.38 MiB
load_tensors:   CPU_Mapped model buffer size = 47377.52 MiB
load_tensors:   CPU_Mapped model buffer size = 47377.52 MiB
load_tensors:   CPU_Mapped model buffer size = 42166.10 MiB
....................................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 16384
llama_context: n_ctx_per_seq = 16384
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 1
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (16384) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_context:  CUDA_Host  output buffer size =     0.58 MiB
llama_kv_cache_unified: kv_size = 16384, type_k = 'q8_0', type_v = 'q8_0', n_layer = 94, can_shift = 1, padding = 256
llama_kv_cache_unified:      CUDA0 KV buffer size =  1598.00 MiB
llama_kv_cache_unified: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
llama_context:      CUDA0 compute buffer size =   774.00 MiB
llama_context:        CPU compute buffer size =     8.25 MiB
llama_context:  CUDA_Host compute buffer size =    40.01 MiB
llama_context: graph nodes  = 5741
llama_context: graph splits = 536 (with bs=512), 180 (with bs=1)

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 48, n_threads_batch = 48

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    8.145 |    62.86 |   10.488 |    12.20 |
|   512 |    128 |    512 |    8.082 |    63.35 |   10.974 |    11.66 |
|   512 |    128 |   1024 |    8.101 |    63.20 |   10.899 |    11.74 |
|   512 |    128 |   1536 |    8.120 |    63.05 |   10.961 |    11.68 |
|   512 |    128 |   2048 |    8.133 |    62.96 |   11.266 |    11.36 |
|   512 |    128 |   2560 |    8.137 |    62.92 |   11.590 |    11.04 |
|   512 |    128 |   3072 |    8.155 |    62.78 |   11.656 |    10.98 |
|   512 |    128 |   3584 |    8.150 |    62.82 |   11.651 |    10.99 |
|   512 |    128 |   4096 |    8.178 |    62.61 |   11.773 |    10.87 |
|   512 |    128 |   4608 |    8.174 |    62.64 |   11.889 |    10.77 |
|   512 |    128 |   5120 |    8.200 |    62.44 |   12.031 |    10.64 |
|   512 |    128 |   5632 |    8.204 |    62.41 |   12.040 |    10.63 |
|   512 |    128 |   6144 |    8.215 |    62.32 |   12.113 |    10.57 |
|   512 |    128 |   6656 |    8.224 |    62.26 |   12.227 |    10.47 |
|   512 |    128 |   7168 |    8.235 |    62.17 |   12.386 |    10.33 |
|   512 |    128 |   7680 |    8.246 |    62.09 |   12.543 |    10.20 |
|   512 |    128 |   8192 |    8.268 |    61.93 |   12.871 |     9.94 |
|   512 |    128 |   8704 |    8.264 |    61.95 |   12.922 |     9.91 |
|   512 |    128 |   9216 |    8.278 |    61.85 |   13.009 |     9.84 |
|   512 |    128 |   9728 |    8.312 |    61.60 |   13.256 |     9.66 |
|   512 |    128 |  10240 |    8.313 |    61.59 |   13.236 |     9.67 |
|   512 |    128 |  10752 |    8.316 |    61.57 |   13.518 |     9.47 |
|   512 |    128 |  11264 |    8.323 |    61.52 |   13.594 |     9.42 |
|   512 |    128 |  11776 |    8.337 |    61.41 |   13.412 |     9.54 |
|   512 |    128 |  12288 |    8.376 |    61.13 |   13.554 |     9.44 |
|   512 |    128 |  12800 |    8.379 |    61.10 |   13.561 |     9.44 |
|   512 |    128 |  13312 |    8.367 |    61.19 |   13.692 |     9.35 |
|   512 |    128 |  13824 |    8.386 |    61.05 |   13.817 |     9.26 |
|   512 |    128 |  14336 |    8.402 |    60.94 |   13.954 |     9.17 |
|   512 |    128 |  14848 |    8.408 |    60.89 |   14.156 |     9.04 |
|   512 |    128 |  15360 |    8.416 |    60.84 |   14.256 |     8.98 |
|   512 |    128 |  15872 |    8.439 |    60.67 |   14.597 |     8.77 |
```

</details>

I used the `sweep-bench-plot.py` to generate the following charts. The series are disambiguated by filename that includes `{llama | ik-llama}-{cpu | gpu}`.

CPU performance PP comparison:
![performance_comparison_pp_cpu](https://github.com/user-attachments/assets/681eeba8-f426-4e93-992d-c67707df49d8)

GPU performance PP comparison:
![performance_comparison_pp_gpu](https://github.com/user-attachments/assets/cdd6236c-73fc-4c7c-a84b-ad4acf3bc2f7)

CPU performance TG comparison:
![performance_comparison_tg_cpu](https://github.com/user-attachments/assets/a06ef3f3-93ea-4e6e-94b2-1ba0d14aa0d3)

GPU performance TG comparison:
![performance_comparison_tg_gpu](https://github.com/user-attachments/assets/293e3761-88f2-4cb9-8754-169aa9d6b153)

> 👤 **AesSedai** replied the **2025-05-03** at **05:29:15**:<br>
> One more test, I disabled pipeline parallelism (setting it to 1) and re-built ik_llama.cpp:
> ```
> cmake -DBLAS_INCLUDE_DIRS=/usr/include/openblas -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_SCHED_MAX_COPIES=1
> ```
> 
> This let me use my second 3090 and offload a little more.
> 
> <details>
> 
> <summary>ik_llama.cpp 2x GPU logs</summary>
> 
> ```
> ./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf -c 16384 -t 48 -fa -rtr -fmoe -ctk q8_0 -ctv q8_0 -ngl 99 -ot "blk\.(0|1|2|3|4|5|6)\.ffn.*=CUDA0" -ot "blk\.(7|8|9|10|11|12|13)\.ffn.*=CUDA1" -ot "blk\.1[4-9]\.ffn.*=CPU" -ot "blk\.[2-9][0-9]\.ffn.*=CPU"
> ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
> ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
> ggml_cuda_init: found 2 CUDA devices:
>   Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
> llama_model_loader: additional 4 GGUFs metadata loaded.
> llama_model_loader: loaded meta data with 39 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf (version GGUF V3 (latest))
> llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
> llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
> llama_model_loader: - kv   1:                               general.type str              = model
> llama_model_loader: - kv   2:                               general.name str              = Models
> llama_model_loader: - kv   3:                         general.size_label str              = 128x10B
> llama_model_loader: - kv   4:                            general.license str              = apache-2.0
> llama_model_loader: - kv   5:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
> llama_model_loader: - kv   6:                               general.tags arr[str,1]       = ["text-generation"]
> llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 94
> llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
> llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 4096
> llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 12288
> llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 64
> llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
> llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
> llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
> llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
> llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
> llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
> llama_model_loader: - kv  18:                          general.file_type u32              = 142
> llama_model_loader: - kv  19:                      qwen3moe.expert_count u32              = 128
> llama_model_loader: - kv  20:        qwen3moe.expert_feed_forward_length u32              = 1536
> llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
> llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = qwen2
> llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
> llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
> llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
> llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 151645
> llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 151643
> llama_model_loader: - kv  28:                tokenizer.ggml.bos_token_id u32              = 151643
> llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = false
> llama_model_loader: - kv  30:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
> llama_model_loader: - kv  31:               general.quantization_version u32              = 2
> llama_model_loader: - kv  32:                      quantize.imatrix.file str              = /workspace/ubergarm/imatrix-Qwen3-235...
> llama_model_loader: - kv  33:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
> llama_model_loader: - kv  34:             quantize.imatrix.entries_count i32              = 753
> llama_model_loader: - kv  35:              quantize.imatrix.chunks_count i32              = 225
> llama_model_loader: - kv  36:                                   split.no u16              = 0
> llama_model_loader: - kv  37:                                split.count u16              = 5
> llama_model_loader: - kv  38:                        split.tensors.count i32              = 1131
> llama_model_loader: - type  f32:  471 tensors
> llama_model_loader: - type q8_0:   96 tensors
> llama_model_loader: - type iq6_k:  564 tensors
> llm_load_vocab: special tokens cache size = 26
> llm_load_vocab: token to piece cache size = 0.9311 MB
> llm_load_print_meta: format           = GGUF V3 (latest)
> llm_load_print_meta: arch             = qwen3moe
> llm_load_print_meta: vocab type       = BPE
> llm_load_print_meta: n_vocab          = 151936
> llm_load_print_meta: n_merges         = 151387
> llm_load_print_meta: vocab_only       = 0
> llm_load_print_meta: n_ctx_train      = 40960
> llm_load_print_meta: n_embd           = 4096
> llm_load_print_meta: n_layer          = 94
> llm_load_print_meta: n_head           = 64
> llm_load_print_meta: n_head_kv        = 4
> llm_load_print_meta: n_rot            = 128
> llm_load_print_meta: n_swa            = 0
> llm_load_print_meta: n_swa_pattern    = 1
> llm_load_print_meta: n_embd_head_k    = 128
> llm_load_print_meta: n_embd_head_v    = 128
> llm_load_print_meta: n_gqa            = 16
> llm_load_print_meta: n_embd_k_gqa     = 512
> llm_load_print_meta: n_embd_v_gqa     = 512
> llm_load_print_meta: f_norm_eps       = 0.0e+00
> llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
> llm_load_print_meta: f_clamp_kqv      = 0.0e+00
> llm_load_print_meta: f_max_alibi_bias = 0.0e+00
> llm_load_print_meta: f_logit_scale    = 0.0e+00
> llm_load_print_meta: n_ff             = 12288
> llm_load_print_meta: n_expert         = 128
> llm_load_print_meta: n_expert_used    = 8
> llm_load_print_meta: causal attn      = 1
> llm_load_print_meta: pooling type     = 0
> llm_load_print_meta: rope type        = 2
> llm_load_print_meta: rope scaling     = linear
> llm_load_print_meta: freq_base_train  = 1000000.0
> llm_load_print_meta: freq_scale_train = 1
> llm_load_print_meta: n_ctx_orig_yarn  = 40960
> llm_load_print_meta: rope_finetuned   = unknown
> llm_load_print_meta: ssm_d_conv       = 0
> llm_load_print_meta: ssm_d_inner      = 0
> llm_load_print_meta: ssm_d_state      = 0
> llm_load_print_meta: ssm_dt_rank      = 0
> llm_load_print_meta: model type       = ?B
> llm_load_print_meta: model ftype      = IQ6_K - 6.6 bpw
> llm_load_print_meta: model params     = 235.094 B
> llm_load_print_meta: model size       = 198.259 GiB (7.244 BPW) 
> llm_load_print_meta: repeating layers = 197.028 GiB (7.237 BPW, 233.849 B parameters)
> llm_load_print_meta: general.name     = Models
> llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
> llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
> llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
> llm_load_print_meta: LF token         = 148848 'ÄĬ'
> llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
> llm_load_print_meta: max token length = 256
> llm_load_print_meta: n_ff_exp         = 1536
> llm_load_tensors: ggml ctx size =    1.49 MiB
> Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
> Tensor blk.0.ffn_gate_inp.weight buffer type overriden to CUDA0
> Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CUDA0
> Tensor blk.0.ffn_down_exps.weight buffer type overriden to CUDA0
> Tensor blk.0.ffn_up_exps.weight buffer type overriden to CUDA0
> Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
> Tensor blk.1.ffn_gate_inp.weight buffer type overriden to CUDA0
> Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CUDA0
> Tensor blk.1.ffn_down_exps.weight buffer type overriden to CUDA0
> Tensor blk.1.ffn_up_exps.weight buffer type overriden to CUDA0
> Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
> Tensor blk.2.ffn_gate_inp.weight buffer type overriden to CUDA0
> Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CUDA0
> Tensor blk.2.ffn_down_exps.weight buffer type overriden to CUDA0
> Tensor blk.2.ffn_up_exps.weight buffer type overriden to CUDA0
> Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
> Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
> Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
> Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
> Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
> Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
> Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
> Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
> Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
> Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
> Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
> Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
> Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
> Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
> Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
> Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
> Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
> Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA0
> Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA0
> Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA0
> Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA1
> Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA1
> Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA1
> Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA1
> Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA1
> Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA1
> Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA1
> Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA1
> Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA1
> Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA1
> Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA1
> Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA1
> Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA1
> Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA1
> Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA1
> Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA1
> Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA1
> Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA1
> Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA1
> Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA1
> Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA1
> Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA1
> Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA1
> Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA1
> Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA1
> Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA1
> Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA1
> Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA1
> Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA1
> Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA1
> Tensor blk.13.ffn_norm.weight buffer type overriden to CUDA1
> Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CUDA1
> Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA1
> Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA1
> Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA1
> Tensor blk.14.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.15.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.16.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.17.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.18.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.19.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.20.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.21.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.22.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.23.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.24.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.25.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.26.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.27.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.28.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.29.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.30.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.31.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.32.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.33.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.34.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.35.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.36.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.36.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.37.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.37.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.38.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.38.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.39.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.39.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.40.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.40.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.41.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.41.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.42.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.42.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.43.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.43.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.44.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.44.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.45.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.45.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.46.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.46.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.47.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.47.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.48.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.48.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.49.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.49.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.50.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.50.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.51.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.51.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.52.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.52.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.53.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.53.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.54.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.54.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.55.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.55.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.56.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.56.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.57.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.57.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.58.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.58.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.59.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.59.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.60.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.60.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.61.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.61.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.62.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.62.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.63.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.63.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.64.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.64.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.65.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.65.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.66.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.66.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.67.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.67.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.68.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.68.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.69.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.69.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.70.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.70.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.71.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.71.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.72.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.72.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.73.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.73.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.74.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.74.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.75.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.75.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.76.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.76.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.77.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.77.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.78.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.78.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.79.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.79.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.80.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.80.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.81.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.81.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.82.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.82.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.83.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.83.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.84.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.84.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.85.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.85.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.86.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.86.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.87.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.87.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.88.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.88.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.89.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.89.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.90.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.90.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.91.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.91.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.92.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.92.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
> Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
> Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
> Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
> Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
> Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
> llm_load_tensors: offloading 94 repeating layers to GPU
> llm_load_tensors: offloading non-repeating layers to GPU
> llm_load_tensors: offloaded 95/95 layers to GPU
> llm_load_tensors:        CPU buffer size = 167201.25 MiB
> llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
> llm_load_tensors:      CUDA0 buffer size = 17333.91 MiB
> llm_load_tensors:      CUDA1 buffer size = 17851.86 MiB
> ....................................................................................................
> ============ Repacked 80 tensors
> llama_new_context_with_model: n_ctx      = 16384
> llama_new_context_with_model: n_batch    = 2048
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 0
> llama_new_context_with_model: attn_max_b = 0
> llama_new_context_with_model: fused_moe  = 1
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 1000000.0
> llama_new_context_with_model: freq_scale = 1
> llama_kv_cache_init:      CUDA0 KV buffer size =   816.02 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =   782.02 MiB
> llama_new_context_with_model: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
> llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size =   144.00 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size =   312.75 MiB
> llama_new_context_with_model:        CPU compute buffer size =     8.25 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =   120.01 MiB
> llama_new_context_with_model: graph nodes  = 3672
> llama_new_context_with_model: graph splits = 336
> 
> main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 48, n_threads_batch = 48
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |   512 |    128 |      0 |    3.957 |   129.39 |    7.209 |    17.75 |
> |   512 |    128 |    512 |    3.910 |   130.94 |    7.676 |    16.67 |
> |   512 |    128 |   1024 |    3.963 |   129.18 |    7.769 |    16.48 |
> |   512 |    128 |   1536 |    3.961 |   129.27 |    7.837 |    16.33 |
> |   512 |    128 |   2048 |    4.033 |   126.94 |    8.236 |    15.54 |
> |   512 |    128 |   2560 |    4.054 |   126.28 |    8.429 |    15.19 |
> |   512 |    128 |   3072 |    4.072 |   125.73 |   10.839 |    11.81 |
> |   512 |    128 |   3584 |    4.098 |   124.94 |   11.515 |    11.12 |
> |   512 |    128 |   4096 |    4.177 |   122.56 |   11.817 |    10.83 |
> |   512 |    128 |   4608 |    4.182 |   122.44 |   12.003 |    10.66 |
> |   512 |    128 |   5120 |    4.215 |   121.48 |   12.178 |    10.51 |
> |   512 |    128 |   5632 |    4.213 |   121.54 |   12.464 |    10.27 |
> |   512 |    128 |   6144 |    4.275 |   119.76 |   12.475 |    10.26 |
> |   512 |    128 |   6656 |    4.200 |   121.89 |   12.690 |    10.09 |
> |   512 |    128 |   7168 |    4.220 |   121.32 |   12.896 |     9.93 |
> |   512 |    128 |   7680 |    4.251 |   120.45 |   13.109 |     9.76 |
> |   512 |    128 |   8192 |    4.279 |   119.66 |   13.253 |     9.66 |
> |   512 |    128 |   8704 |    4.293 |   119.26 |   13.550 |     9.45 |
> |   512 |    128 |   9216 |    4.291 |   119.31 |   13.668 |     9.37 |
> |   512 |    128 |   9728 |    4.301 |   119.04 |   13.804 |     9.27 |
> |   512 |    128 |  10240 |    4.306 |   118.90 |   14.200 |     9.01 |
> |   512 |    128 |  10752 |    4.338 |   118.02 |   14.255 |     8.98 |
> |   512 |    128 |  11264 |    4.330 |   118.25 |   14.403 |     8.89 |
> |   512 |    128 |  11776 |    4.375 |   117.03 |   14.506 |     8.82 |
> |   512 |    128 |  12288 |    4.413 |   116.03 |   14.864 |     8.61 |
> |   512 |    128 |  12800 |    4.414 |   116.00 |   14.960 |     8.56 |
> |   512 |    128 |  13312 |    4.419 |   115.86 |   15.197 |     8.42 |
> |   512 |    128 |  13824 |    4.440 |   115.32 |   15.448 |     8.29 |
> |   512 |    128 |  14336 |    4.463 |   114.72 |   15.592 |     8.21 |
> |   512 |    128 |  14848 |    4.473 |   114.46 |   15.740 |     8.13 |
> |   512 |    128 |  15360 |    4.507 |   113.61 |   15.883 |     8.06 |
> |   512 |    128 |  15872 |    4.514 |   113.43 |   16.207 |     7.90 |
> ```
> 
> </details>
> 
> 
> <details>
> 
> <summary>llama.cpp 2x GPU logs</summary>
> 
> ```
> ./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-128K-GGUF/Q6_K/Qwen3-235B-A22B-128K-Q6_K-00001-of-00004.gguf -c 16384 -t 48 -fa -ctk q8_0 -ctv q8_0 -ngl 99 -ot "blk\.(0|1|2|3|4|5|6)\.ffn.*=CUDA0" -ot "blk\.(7|8|9|10|11|12|13)\.ffn.*=CUDA1" -ot "blk\.1[4-9]\.ffn.*=CPU" -ot "blk\.[2-9][0-9]\.ffn.*=CPU"
> ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
> ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
> ggml_cuda_init: found 2 CUDA devices:
>   Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
> build: 5269 (1d36b367) with cc (GCC) 14.2.1 20250110 (Red Hat 14.2.1-7) for x86_64-redhat-linux
> llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 3090) - 23871 MiB free
> llama_model_load_from_file_impl: using device CUDA1 (NVIDIA GeForce RTX 3090) - 23871 MiB free
> llama_model_loader: additional 3 GGUFs metadata loaded.
> llama_model_loader: loaded meta data with 47 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-128K-GGUF/Q6_K/Qwen3-235B-A22B-128K-Q6_K-00001-of-00004.gguf (version GGUF V3 (latest))
> llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
> llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
> llama_model_loader: - kv   1:                               general.type str              = model
> llama_model_loader: - kv   2:                               general.name str              = Qwen3-235B-A22B-128K
> llama_model_loader: - kv   3:                           general.finetune str              = 128k
> llama_model_loader: - kv   4:                           general.basename str              = Qwen3-235B-A22B-128K
> llama_model_loader: - kv   5:                       general.quantized_by str              = Unsloth
> llama_model_loader: - kv   6:                         general.size_label str              = 235B-A22B
> llama_model_loader: - kv   7:                            general.license str              = apache-2.0
> llama_model_loader: - kv   8:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
> llama_model_loader: - kv   9:                           general.repo_url str              = https://huggingface.co/unsloth
> llama_model_loader: - kv  10:                   general.base_model.count u32              = 1
> llama_model_loader: - kv  11:                  general.base_model.0.name str              = Qwen3 235B A22B
> llama_model_loader: - kv  12:          general.base_model.0.organization str              = Qwen
> llama_model_loader: - kv  13:              general.base_model.0.repo_url str              = https://huggingface.co/Qwen/Qwen3-235...
> llama_model_loader: - kv  14:                               general.tags arr[str,2]       = ["unsloth", "text-generation"]
> llama_model_loader: - kv  15:                       qwen3moe.block_count u32              = 94
> llama_model_loader: - kv  16:                    qwen3moe.context_length u32              = 131072
> llama_model_loader: - kv  17:                  qwen3moe.embedding_length u32              = 4096
> llama_model_loader: - kv  18:               qwen3moe.feed_forward_length u32              = 12288
> llama_model_loader: - kv  19:              qwen3moe.attention.head_count u32              = 64
> llama_model_loader: - kv  20:           qwen3moe.attention.head_count_kv u32              = 4
> llama_model_loader: - kv  21:                    qwen3moe.rope.freq_base f32              = 1000000.000000
> llama_model_loader: - kv  22:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
> llama_model_loader: - kv  23:                 qwen3moe.expert_used_count u32              = 8
> llama_model_loader: - kv  24:              qwen3moe.attention.key_length u32              = 128
> llama_model_loader: - kv  25:            qwen3moe.attention.value_length u32              = 128
> llama_model_loader: - kv  26:                      qwen3moe.expert_count u32              = 128
> llama_model_loader: - kv  27:        qwen3moe.expert_feed_forward_length u32              = 1536
> llama_model_loader: - kv  28:                       tokenizer.ggml.model str              = gpt2
> llama_model_loader: - kv  29:                         tokenizer.ggml.pre str              = qwen2
> llama_model_loader: - kv  30:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
> llama_model_loader: - kv  31:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
> llama_model_loader: - kv  32:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
> llama_model_loader: - kv  33:                tokenizer.ggml.eos_token_id u32              = 151645
> llama_model_loader: - kv  34:            tokenizer.ggml.padding_token_id u32              = 151643
> llama_model_loader: - kv  35:                tokenizer.ggml.bos_token_id u32              = 151643
> llama_model_loader: - kv  36:               tokenizer.ggml.add_bos_token bool             = false
> llama_model_loader: - kv  37:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
> llama_model_loader: - kv  38:               general.quantization_version u32              = 2
> llama_model_loader: - kv  39:                          general.file_type u32              = 18
> llama_model_loader: - kv  40:                      quantize.imatrix.file str              = Qwen3-235B-A22B-128K-GGUF/imatrix_uns...
> llama_model_loader: - kv  41:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-235B-A22B-1...
> llama_model_loader: - kv  42:             quantize.imatrix.entries_count i32              = 752
> llama_model_loader: - kv  43:              quantize.imatrix.chunks_count i32              = 46
> llama_model_loader: - kv  44:                                   split.no u16              = 0
> llama_model_loader: - kv  45:                        split.tensors.count i32              = 1131
> llama_model_loader: - kv  46:                                split.count u16              = 4
> llama_model_loader: - type  f32:  471 tensors
> llama_model_loader: - type q6_K:  660 tensors
> print_info: file format = GGUF V3 (latest)
> print_info: file type   = Q6_K
> print_info: file size   = 179.75 GiB (6.57 BPW) 
> load: special tokens cache size = 26
> load: token to piece cache size = 0.9311 MB
> print_info: arch             = qwen3moe
> print_info: vocab_only       = 0
> print_info: n_ctx_train      = 131072
> print_info: n_embd           = 4096
> print_info: n_layer          = 94
> print_info: n_head           = 64
> print_info: n_head_kv        = 4
> print_info: n_rot            = 128
> print_info: n_swa            = 0
> print_info: n_swa_pattern    = 1
> print_info: n_embd_head_k    = 128
> print_info: n_embd_head_v    = 128
> print_info: n_gqa            = 16
> print_info: n_embd_k_gqa     = 512
> print_info: n_embd_v_gqa     = 512
> print_info: f_norm_eps       = 0.0e+00
> print_info: f_norm_rms_eps   = 1.0e-06
> print_info: f_clamp_kqv      = 0.0e+00
> print_info: f_max_alibi_bias = 0.0e+00
> print_info: f_logit_scale    = 0.0e+00
> print_info: f_attn_scale     = 0.0e+00
> print_info: n_ff             = 12288
> print_info: n_expert         = 128
> print_info: n_expert_used    = 8
> print_info: causal attn      = 1
> print_info: pooling type     = 0
> print_info: rope type        = 2
> print_info: rope scaling     = linear
> print_info: freq_base_train  = 1000000.0
> print_info: freq_scale_train = 1
> print_info: n_ctx_orig_yarn  = 131072
> print_info: rope_finetuned   = unknown
> print_info: ssm_d_conv       = 0
> print_info: ssm_d_inner      = 0
> print_info: ssm_d_state      = 0
> print_info: ssm_dt_rank      = 0
> print_info: ssm_dt_b_c_rms   = 0
> print_info: model type       = 235B.A22B
> print_info: model params     = 235.09 B
> print_info: general.name     = Qwen3-235B-A22B-128K
> print_info: n_ff_exp         = 1536
> print_info: vocab type       = BPE
> print_info: n_vocab          = 151936
> print_info: n_merges         = 151387
> print_info: BOS token        = 151643 '<|endoftext|>'
> print_info: EOS token        = 151645 '<|im_end|>'
> print_info: EOT token        = 151645 '<|im_end|>'
> print_info: PAD token        = 151643 '<|endoftext|>'
> print_info: LF token         = 198 'Ċ'
> print_info: FIM PRE token    = 151659 '<|fim_prefix|>'
> print_info: FIM SUF token    = 151661 '<|fim_suffix|>'
> print_info: FIM MID token    = 151660 '<|fim_middle|>'
> print_info: FIM PAD token    = 151662 '<|fim_pad|>'
> print_info: FIM REP token    = 151663 '<|repo_name|>'
> print_info: FIM SEP token    = 151664 '<|file_sep|>'
> print_info: EOG token        = 151643 '<|endoftext|>'
> print_info: EOG token        = 151645 '<|im_end|>'
> print_info: EOG token        = 151662 '<|fim_pad|>'
> print_info: EOG token        = 151663 '<|repo_name|>'
> print_info: EOG token        = 151664 '<|file_sep|>'
> print_info: max token length = 256
> load_tensors: loading model tensors, this can take a while... (mmap = true)
> load_tensors: offloading 94 repeating layers to GPU
> load_tensors: offloading output layer to GPU
> load_tensors: offloaded 95/95 layers to GPU
> load_tensors:        CUDA0 model buffer size = 15922.41 MiB
> load_tensors:        CUDA1 model buffer size = 16297.68 MiB
> load_tensors:   CPU_Mapped model buffer size = 46604.38 MiB
> load_tensors:   CPU_Mapped model buffer size = 47377.52 MiB
> load_tensors:   CPU_Mapped model buffer size = 47377.52 MiB
> load_tensors:   CPU_Mapped model buffer size = 42166.10 MiB
> ....................................................................................................
> llama_context: constructing llama_context
> llama_context: n_seq_max     = 1
> llama_context: n_ctx         = 16384
> llama_context: n_ctx_per_seq = 16384
> llama_context: n_batch       = 2048
> llama_context: n_ubatch      = 512
> llama_context: causal_attn   = 1
> llama_context: flash_attn    = 1
> llama_context: freq_base     = 1000000.0
> llama_context: freq_scale    = 1
> llama_context: n_ctx_per_seq (16384) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
> llama_context:  CUDA_Host  output buffer size =     0.58 MiB
> llama_kv_cache_unified: kv_size = 16384, type_k = 'q8_0', type_v = 'q8_0', n_layer = 94, can_shift = 1, padding = 256
> llama_kv_cache_unified:      CUDA0 KV buffer size =   816.00 MiB
> llama_kv_cache_unified:      CUDA1 KV buffer size =   782.00 MiB
> llama_kv_cache_unified: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
> llama_context:      CUDA0 compute buffer size =   774.00 MiB
> llama_context:      CUDA1 compute buffer size =   304.75 MiB
> llama_context:        CPU compute buffer size =     8.25 MiB
> llama_context:  CUDA_Host compute buffer size =    40.01 MiB
> llama_context: graph nodes  = 5741
> llama_context: graph splits = 543 (with bs=512), 176 (with bs=1)
> 
> main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 48, n_threads_batch = 48
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |   512 |    128 |      0 |    7.567 |    67.67 |    7.486 |    17.10 |
> |   512 |    128 |    512 |    7.474 |    68.51 |    8.910 |    14.37 |
> |   512 |    128 |   1024 |    7.488 |    68.38 |   10.591 |    12.09 |
> |   512 |    128 |   1536 |    7.515 |    68.13 |   10.581 |    12.10 |
> |   512 |    128 |   2048 |    7.535 |    67.95 |   10.588 |    12.09 |
> |   512 |    128 |   2560 |    7.537 |    67.93 |   11.006 |    11.63 |
> |   512 |    128 |   3072 |    7.552 |    67.80 |   11.114 |    11.52 |
> |   512 |    128 |   3584 |    7.563 |    67.70 |   11.234 |    11.39 |
> |   512 |    128 |   4096 |    7.578 |    67.56 |   11.320 |    11.31 |
> |   512 |    128 |   4608 |    7.600 |    67.37 |   11.389 |    11.24 |
> |   512 |    128 |   5120 |    7.594 |    67.42 |   11.932 |    10.73 |
> |   512 |    128 |   5632 |    7.595 |    67.41 |   11.827 |    10.82 |
> |   512 |    128 |   6144 |    7.612 |    67.26 |   11.759 |    10.89 |
> |   512 |    128 |   6656 |    7.626 |    67.14 |   11.961 |    10.70 |
> |   512 |    128 |   7168 |    7.656 |    66.88 |   12.073 |    10.60 |
> |   512 |    128 |   7680 |    7.660 |    66.84 |   12.190 |    10.50 |
> |   512 |    128 |   8192 |    7.672 |    66.74 |   12.343 |    10.37 |
> |   512 |    128 |   8704 |    7.682 |    66.65 |   12.790 |    10.01 |
> |   512 |    128 |   9216 |    7.693 |    66.56 |   12.578 |    10.18 |
> |   512 |    128 |   9728 |    7.712 |    66.39 |   12.825 |     9.98 |
> |   512 |    128 |  10240 |    7.725 |    66.28 |   13.087 |     9.78 |
> |   512 |    128 |  10752 |    7.736 |    66.18 |   13.017 |     9.83 |
> |   512 |    128 |  11264 |    7.750 |    66.07 |   13.143 |     9.74 |
> |   512 |    128 |  11776 |    7.750 |    66.07 |   13.249 |     9.66 |
> |   512 |    128 |  12288 |    7.769 |    65.90 |   13.393 |     9.56 |
> |   512 |    128 |  12800 |    7.773 |    65.87 |   13.537 |     9.46 |
> |   512 |    128 |  13312 |    7.787 |    65.75 |   13.620 |     9.40 |
> |   512 |    128 |  13824 |    7.805 |    65.60 |   13.697 |     9.34 |
> |   512 |    128 |  14336 |    7.823 |    65.44 |   13.976 |     9.16 |
> |   512 |    128 |  14848 |    7.825 |    65.43 |   14.067 |     9.10 |
> |   512 |    128 |  15360 |    7.824 |    65.44 |   14.361 |     8.91 |
> |   512 |    128 |  15872 |    7.838 |    65.32 |   14.393 |     8.89 |
> ```
> 
> </details>
> 
> and these are the updated GPU graphs:
> 
> ![performance_comparison_pp_gpu](https://github.com/user-attachments/assets/31612aea-90d7-446c-acc1-30f064185fc4)
> 
> ![performance_comparison_tg_gpu](https://github.com/user-attachments/assets/ecec5ab4-f1c9-42f5-b998-3b84a1986747)

---

👤 **ikawrakow** replied the **2025-05-03** at **05:47:58**:<br>

Thank you for these results!

I think it would be better to disable BLAS for both. CPU Prompt processing with `ik_llama.cpp` is likely faster without BLAS, but also it is more interesting to measure how well matrix multiplications are implemented in the two tool kits instead of measuring how well they call somebody else's GEMM implementation. 

Prompt processing speed on CUDA will also benefit from larger u-batches (e.g., `-ub 2048`, in case VRAM permits).

The CUDA TG results are somewhat surprising (sharp performance drop with context length for `ik_llama.cpp`, performance basically the same as CPU-only for long context, performance decreasing with more layers offloaded to a second GPU).

> 👤 **AesSedai** replied the **2025-05-03** at **06:07:58**:<br>
> I just re-ran the above with 2x GPU for llama.cpp as well and edited the comment / graph. I was already re-running ik_llama w/o BLAS, I'll have the results of that shortly.
> 
> 👤 **AesSedai** replied the **2025-05-03** at **06:19:29**:<br>
> Posted!

---

👤 **AesSedai** replied the **2025-05-03** at **06:18:41**:<br>

Some more data, this time compiled w/ no BLAS:
```
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1
```

Note: the ik_llama.cpp CPU NO BLAS did hit a CUDA error on the very last iteration.

<details>

<summary>ik_llama.cpp CPU NO BLAS logs</summary>

```
./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf -c 16384 -t 48 -fa -rtr -fmoe -ctk q8_0 -ctv q8_0
llama_model_loader: additional 4 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 39 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Models
llama_model_loader: - kv   3:                         general.size_label str              = 128x10B
llama_model_loader: - kv   4:                            general.license str              = apache-2.0
llama_model_loader: - kv   5:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   6:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                          general.file_type u32              = 142
llama_model_loader: - kv  19:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  20:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  28:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                      quantize.imatrix.file str              = /workspace/ubergarm/imatrix-Qwen3-235...
llama_model_loader: - kv  33:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  34:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  35:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  36:                                   split.no u16              = 0
llama_model_loader: - kv  37:                                split.count u16              = 5
llama_model_loader: - kv  38:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:   96 tensors
llama_model_loader: - type iq6_k:  564 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ6_K - 6.6 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 198.259 GiB (7.244 BPW) 
llm_load_print_meta: repeating layers = 197.028 GiB (7.237 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Models
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.50 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/95 layers to GPU
llm_load_tensors:  CUDA_Host buffer size = 203017.61 MiB
....................................................................................................
============ Repacked 95 tensors
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:  CUDA_Host KV buffer size =  1598.00 MiB
llama_new_context_with_model: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   161.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   304.75 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 1319

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 48, n_threads_batch = 48

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.591 |   111.51 |   14.650 |     8.74 |
|   512 |    128 |    512 |    4.285 |   119.49 |   14.375 |     8.90 |
|   512 |    128 |   1024 |    4.340 |   117.99 |   15.437 |     8.29 |
|   512 |    128 |   1536 |    4.327 |   118.32 |   15.208 |     8.42 |
|   512 |    128 |   2048 |    4.380 |   116.90 |   14.995 |     8.54 |
|   512 |    128 |   2560 |    4.413 |   116.03 |   15.521 |     8.25 |
|   512 |    128 |   3072 |    4.424 |   115.72 |   16.063 |     7.97 |
|   512 |    128 |   3584 |    4.401 |   116.34 |   15.785 |     8.11 |
|   512 |    128 |   4096 |    4.481 |   114.25 |   16.065 |     7.97 |
|   512 |    128 |   4608 |    4.519 |   113.29 |   16.395 |     7.81 |
|   512 |    128 |   5120 |    4.475 |   114.41 |   15.560 |     8.23 |
|   512 |    128 |   5632 |    4.553 |   112.45 |   16.124 |     7.94 |
|   512 |    128 |   6144 |    4.552 |   112.47 |   16.174 |     7.91 |
|   512 |    128 |   6656 |    4.564 |   112.19 |   15.575 |     8.22 |
|   512 |    128 |   7168 |    4.611 |   111.05 |   16.761 |     7.64 |
|   512 |    128 |   7680 |    4.630 |   110.57 |   16.769 |     7.63 |
|   512 |    128 |   8192 |    4.610 |   111.06 |   16.260 |     7.87 |
|   512 |    128 |   8704 |    4.677 |   109.47 |   17.069 |     7.50 |
|   512 |    128 |   9216 |    4.675 |   109.52 |   17.445 |     7.34 |
|   512 |    128 |   9728 |    4.706 |   108.80 |   16.528 |     7.74 |
|   512 |    128 |  10240 |    4.735 |   108.12 |   17.745 |     7.21 |
|   512 |    128 |  10752 |    4.746 |   107.89 |   17.733 |     7.22 |
|   512 |    128 |  11264 |    4.790 |   106.90 |   16.780 |     7.63 |
|   512 |    128 |  11776 |    4.822 |   106.18 |   17.795 |     7.19 |
|   512 |    128 |  12288 |    4.883 |   104.85 |   18.182 |     7.04 |
|   512 |    128 |  12800 |    4.875 |   105.02 |   17.035 |     7.51 |
|   512 |    128 |  13312 |    4.886 |   104.80 |   18.353 |     6.97 |
|   512 |    128 |  13824 |    4.932 |   103.80 |   18.488 |     6.92 |
|   512 |    128 |  14336 |    4.960 |   103.23 |   17.388 |     7.36 |
|   512 |    128 |  14848 |    4.999 |   102.42 |   18.743 |     6.83 |
|   512 |    128 |  15360 |    4.931 |   103.83 |   18.383 |     6.96 |
CUDA error: invalid argument
  current device: 0, in function ggml_backend_cuda_buffer_set_tensor at /home/jarvis/development/ik_llama.cpp/ggml/src/ggml-cuda.cu:507
  cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, ((cudaStream_t)0x2))
/home/jarvis/development/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
```

</details>


<details>

<summary>ik_llama.cpp 2x GPU NO BLAS logs</summary>

```
./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf -c 16384 -t 48 -fa -rtr -fmoe -ctk q8_0 -ctv q8_0 -ngl 99 -ot "blk\.(0|1|2|3|4|5|6)\.ffn.*=CUDA0" -ot "blk\.(7|8|9|10|11|12|13)\.ffn.*=CUDA1" -ot "blk\.1[4-9]\.ffn.*=CPU" -ot "blk\.[2-9][0-9]\.ffn.*=CPU"
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
llama_model_loader: additional 4 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 39 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Models
llama_model_loader: - kv   3:                         general.size_label str              = 128x10B
llama_model_loader: - kv   4:                            general.license str              = apache-2.0
llama_model_loader: - kv   5:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   6:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                          general.file_type u32              = 142
llama_model_loader: - kv  19:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  20:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  28:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                      quantize.imatrix.file str              = /workspace/ubergarm/imatrix-Qwen3-235...
llama_model_loader: - kv  33:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  34:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  35:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  36:                                   split.no u16              = 0
llama_model_loader: - kv  37:                                split.count u16              = 5
llama_model_loader: - kv  38:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:   96 tensors
llama_model_loader: - type iq6_k:  564 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ6_K - 6.6 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 198.259 GiB (7.244 BPW) 
llm_load_print_meta: repeating layers = 197.028 GiB (7.237 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Models
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    1.49 MiB
Tensor blk.0.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.14.ffn_norm.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_norm.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_norm.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_norm.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_norm.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_norm.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_norm.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_norm.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_norm.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_norm.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_norm.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_norm.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_norm.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_norm.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_norm.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_norm.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_norm.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_norm.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_norm.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_norm.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_norm.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_norm.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_norm.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_norm.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_norm.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_norm.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_norm.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_norm.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_norm.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_norm.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_norm.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_norm.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_norm.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_norm.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_norm.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_norm.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_norm.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_norm.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_norm.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_norm.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_norm.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_norm.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_norm.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_norm.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_norm.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_norm.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_norm.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_norm.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_norm.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_norm.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_norm.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_norm.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_norm.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_norm.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_norm.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_norm.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_norm.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_norm.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_norm.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_norm.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_norm.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_norm.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_norm.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_norm.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_norm.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_norm.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_norm.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_norm.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_norm.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_norm.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_norm.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_norm.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_norm.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_norm.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_norm.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_norm.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_norm.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_norm.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_norm.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_norm.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_inp.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 167201.25 MiB
llm_load_tensors:  CUDA_Host buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size = 17221.25 MiB
llm_load_tensors:      CUDA1 buffer size = 17964.52 MiB
....................................................................................................
============ Repacked 80 tensors
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =   782.02 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   816.02 MiB
llama_new_context_with_model: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =   144.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   312.75 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   120.01 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 336

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 48, n_threads_batch = 48

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.738 |   136.96 |   10.651 |    12.02 |
|   512 |    128 |    512 |    3.657 |   140.02 |   10.862 |    11.78 |
|   512 |    128 |   1024 |    3.710 |   138.00 |   11.029 |    11.61 |
|   512 |    128 |   1536 |    3.727 |   137.38 |   10.993 |    11.64 |
|   512 |    128 |   2048 |    3.727 |   137.39 |   11.148 |    11.48 |
|   512 |    128 |   2560 |    3.788 |   135.15 |   11.437 |    11.19 |
|   512 |    128 |   3072 |    3.798 |   134.81 |   11.629 |    11.01 |
|   512 |    128 |   3584 |    3.761 |   136.14 |   12.042 |    10.63 |
|   512 |    128 |   4096 |    3.822 |   133.96 |   11.777 |    10.87 |
|   512 |    128 |   4608 |    3.838 |   133.41 |   11.934 |    10.73 |
|   512 |    128 |   5120 |    3.878 |   132.04 |   12.227 |    10.47 |
|   512 |    128 |   5632 |    3.911 |   130.93 |   12.452 |    10.28 |
|   512 |    128 |   6144 |    3.902 |   131.23 |   12.550 |    10.20 |
|   512 |    128 |   6656 |    3.929 |   130.30 |   12.660 |    10.11 |
|   512 |    128 |   7168 |    3.968 |   129.02 |   12.961 |     9.88 |
|   512 |    128 |   7680 |    4.016 |   127.49 |   13.059 |     9.80 |
|   512 |    128 |   8192 |    3.993 |   128.21 |   13.391 |     9.56 |
|   512 |    128 |   8704 |    4.052 |   126.37 |   13.501 |     9.48 |
|   512 |    128 |   9216 |    4.061 |   126.06 |   13.644 |     9.38 |
|   512 |    128 |   9728 |    4.093 |   125.10 |   13.916 |     9.20 |
|   512 |    128 |  10240 |    4.140 |   123.69 |   14.154 |     9.04 |
|   512 |    128 |  10752 |    4.190 |   122.20 |   14.478 |     8.84 |
|   512 |    128 |  11264 |    4.198 |   121.97 |   14.657 |     8.73 |
|   512 |    128 |  11776 |    4.224 |   121.21 |   14.880 |     8.60 |
|   512 |    128 |  12288 |    4.242 |   120.70 |   15.020 |     8.52 |
|   512 |    128 |  12800 |    4.244 |   120.63 |   15.034 |     8.51 |
|   512 |    128 |  13312 |    4.258 |   120.25 |   15.194 |     8.42 |
|   512 |    128 |  13824 |    4.238 |   120.81 |   15.376 |     8.32 |
|   512 |    128 |  14336 |    4.268 |   119.96 |   15.554 |     8.23 |
|   512 |    128 |  14848 |    4.260 |   120.20 |   15.667 |     8.17 |
|   512 |    128 |  15360 |    4.285 |   119.49 |   15.961 |     8.02 |
|   512 |    128 |  15872 |    4.315 |   118.65 |   16.185 |     7.91 |
```

</details>

ik_llama.cpp BLAS vs NO BLAS PP comparison:
![performance_comparison_pp_gpu](https://github.com/user-attachments/assets/c6319126-8867-48be-aca4-e9360dedb5d8)

ik_llama.cpp BLAS vs NO BLAS TG comparison:
![performance_comparison_tg_gpu](https://github.com/user-attachments/assets/06035299-c021-4279-907e-cb1d6a2f9f74)

> 👤 **ikawrakow** replied the **2025-05-03** at **06:24:36**:<br>
> Oh, for CPU-only inference you want to build **without CUDA**. The almighty `ggml` back-end scheduler that is very difficult to work around takes all sorts of funny decisions where to run stuff when one has more than one back-end enabled.
> 
> 👤 **AesSedai** replied the **2025-05-03** at **06:25:03**:<br>
> D'oh, okay. I can redo it :)

---

👤 **AesSedai** replied the **2025-05-03** at **07:04:12**:<br>

ik_llama.cpp, no cuda, no blas:
```
cmake -B build -DGGML_RPC=ON -DGGML_CUDA=OFF -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1
```

<details>

<summary>ik_llama.cpp CPU NO CUDA NO BLAS logs</summary>

```
./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf -c 16384 -t 48 -fa -rtr -fmoe -ctk q8_0 -ctv q8_0                        
llama_model_loader: additional 4 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 39 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Models
llama_model_loader: - kv   3:                         general.size_label str              = 128x10B
llama_model_loader: - kv   4:                            general.license str              = apache-2.0
llama_model_loader: - kv   5:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   6:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                          general.file_type u32              = 142
llama_model_loader: - kv  19:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  20:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  28:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                      quantize.imatrix.file str              = /workspace/ubergarm/imatrix-Qwen3-235...
llama_model_loader: - kv  33:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  34:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  35:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  36:                                   split.no u16              = 0
llama_model_loader: - kv  37:                                split.count u16              = 5
llama_model_loader: - kv  38:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:   96 tensors
llama_model_loader: - type iq6_k:  564 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ6_K - 6.6 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 198.259 GiB (7.244 BPW) 
llm_load_print_meta: repeating layers = 197.028 GiB (7.237 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Models
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.50 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/95 layers to GPU
llm_load_tensors:        CPU buffer size = 203017.61 MiB
....................................................................................................
============ Repacked 95 tensors
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =  1598.00 MiB
llama_new_context_with_model: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.58 MiB
llama_new_context_with_model:        CPU compute buffer size =   304.75 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 1

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 48, n_threads_batch = 48

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    5.810 |    88.13 |    8.608 |    14.87 |
|   512 |    128 |    512 |    5.915 |    86.57 |    8.561 |    14.95 |
|   512 |    128 |   1024 |    6.033 |    84.86 |    9.789 |    13.08 |
|   512 |    128 |   1536 |    6.295 |    81.34 |   13.626 |     9.39 |
|   512 |    128 |   2048 |    6.432 |    79.60 |   13.771 |     9.29 |
|   512 |    128 |   2560 |    6.527 |    78.44 |   14.327 |     8.93 |
|   512 |    128 |   3072 |    6.730 |    76.08 |   14.390 |     8.90 |
|   512 |    128 |   3584 |    6.829 |    74.98 |   13.901 |     9.21 |
|   512 |    128 |   4096 |    7.007 |    73.07 |   14.830 |     8.63 |
|   512 |    128 |   4608 |    7.170 |    71.41 |   14.868 |     8.61 |
|   512 |    128 |   5120 |    7.288 |    70.26 |   14.373 |     8.91 |
|   512 |    128 |   5632 |    7.450 |    68.73 |   15.489 |     8.26 |
|   512 |    128 |   6144 |    7.642 |    67.00 |   15.442 |     8.29 |
|   512 |    128 |   6656 |    7.808 |    65.57 |   14.509 |     8.82 |
|   512 |    128 |   7168 |    7.956 |    64.35 |   15.828 |     8.09 |
|   512 |    128 |   7680 |    8.136 |    62.93 |   15.789 |     8.11 |
|   512 |    128 |   8192 |    8.227 |    62.23 |   14.904 |     8.59 |
|   512 |    128 |   8704 |    8.396 |    60.98 |   15.952 |     8.02 |
|   512 |    128 |   9216 |    8.511 |    60.16 |   16.488 |     7.76 |
|   512 |    128 |   9728 |    8.664 |    59.10 |   15.098 |     8.48 |
|   512 |    128 |  10240 |    8.850 |    57.86 |   16.699 |     7.66 |
|   512 |    128 |  10752 |    8.979 |    57.02 |   16.578 |     7.72 |
|   512 |    128 |  11264 |    9.150 |    55.96 |   15.277 |     8.38 |
|   512 |    128 |  11776 |    9.339 |    54.82 |   16.711 |     7.66 |
|   512 |    128 |  12288 |    9.455 |    54.15 |   16.698 |     7.67 |
|   512 |    128 |  12800 |    9.627 |    53.18 |   15.516 |     8.25 |
|   512 |    128 |  13312 |    9.767 |    52.42 |   17.165 |     7.46 |
|   512 |    128 |  13824 |    9.944 |    51.49 |   17.486 |     7.32 |
|   512 |    128 |  14336 |   10.074 |    50.82 |   16.163 |     7.92 |
|   512 |    128 |  14848 |   10.183 |    50.28 |   17.452 |     7.33 |
|   512 |    128 |  15360 |   10.347 |    49.48 |   17.787 |     7.20 |
|   512 |    128 |  15872 |   10.553 |    48.52 |   16.355 |     7.83 |
```

</details>

![performance_comparison_pp](https://github.com/user-attachments/assets/22a6981f-9852-454e-a12c-69b5b8ba6b88)

![performance_comparison_tg](https://github.com/user-attachments/assets/4676ce0d-80e7-4940-9166-e4ac786d0dc2)

---

👤 **ikawrakow** replied the **2025-05-03** at **07:21:24**:<br>

Thanks!

So, CPU PP is much better now and more inline with what I would have expected. Looking at the TG graph, it is clear that I still need to work on improving how the work is divided between the threads. The Qwen3 MoE models have a high GQA factor, so one should be able to achieve ~70-80% of zero-context performance at 16k tokens.

But I see that the Epyc 9355 has 32 cores, so we are using hyper-threading?

> 👤 **AesSedai** replied the **2025-05-03** at **07:23:30**:<br>
> That's good news!
> 
> Yes, this is with hyperthreading. Out of the 64 threads on the system, 56 are passed through to the virtual machine and I have it configured to use 48 of those during the sweep.
> 
> Is there a particular `-t` count (or thread passthrough count) you would like me to try?
> 
> 👤 **ikawrakow** replied the **2025-05-03** at **07:27:14**:<br>
> On bare metal one achieves the best performance by setting the number of threads to the physical core count. But I have no idea how a VM will behave. You can try `-t 32`, but that would be only better if you get 32 cores involved, and not e.g. 16 cores with 2 threads per core.
> 
> 👤 **AesSedai** replied the **2025-05-03** at **07:58:15**:<br>
> Yes, I think it's about a ~10% performance loss because it's in a VM. The system is a hypervisor though and used for other homelab things, so I'm fine taking that loss. I was able to run `likwid-bench` inside the VM before and achieve ~500GB/s memory bandwidth for reference, theoretical maximum is ~576GB/s.
> 
> For completeness sake, I've disabled SMT on the host:
> ```
> echo off > /sys/devices/system/cpu/smt/control
> ```
> and verified the core count with `btop` shows 32. Re-launched the inference VM with all 32 cores set to the VM.
> 
> I also turned off the other three VMs on the system, so this is the "max performance" configuration that I can achieve without moving everything from the VM to the hypervisor host directly.
> 
> <details>
> 
> <summary>ik_llama.cpp CPU NO CUDA NO BLAS NO SMTlogs</summary>
> 
> ```
> ./build/bin/llama-sweep-bench -m /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf -c 16384 -t 32 -fa -rtr -fmoe -ctk q8_0 -ctv q8_0
> llama_model_loader: additional 4 GGUFs metadata loaded.
> llama_model_loader: loaded meta data with 39 key-value pairs and 1131 tensors from /mnt/srv/slush/gguf/Qwen3-235B-A22B-GGUF-ik-llama/Qwen3-235B-A22B-mix-IQ6_K-00001-of-00005.gguf (version GGUF V3 (latest))
> llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
> llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
> llama_model_loader: - kv   1:                               general.type str              = model
> llama_model_loader: - kv   2:                               general.name str              = Models
> llama_model_loader: - kv   3:                         general.size_label str              = 128x10B
> llama_model_loader: - kv   4:                            general.license str              = apache-2.0
> llama_model_loader: - kv   5:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
> llama_model_loader: - kv   6:                               general.tags arr[str,1]       = ["text-generation"]
> llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 94
> llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
> llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 4096
> llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 12288
> llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 64
> llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
> llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
> llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
> llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
> llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
> llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
> llama_model_loader: - kv  18:                          general.file_type u32              = 142
> llama_model_loader: - kv  19:                      qwen3moe.expert_count u32              = 128
> llama_model_loader: - kv  20:        qwen3moe.expert_feed_forward_length u32              = 1536
> llama_model_loader: - kv  21:                       tokenizer.ggml.model str              = gpt2
> llama_model_loader: - kv  22:                         tokenizer.ggml.pre str              = qwen2
> llama_model_loader: - kv  23:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
> llama_model_loader: - kv  24:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
> llama_model_loader: - kv  25:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
> llama_model_loader: - kv  26:                tokenizer.ggml.eos_token_id u32              = 151645
> llama_model_loader: - kv  27:            tokenizer.ggml.padding_token_id u32              = 151643
> llama_model_loader: - kv  28:                tokenizer.ggml.bos_token_id u32              = 151643
> llama_model_loader: - kv  29:               tokenizer.ggml.add_bos_token bool             = false
> llama_model_loader: - kv  30:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
> llama_model_loader: - kv  31:               general.quantization_version u32              = 2
> llama_model_loader: - kv  32:                      quantize.imatrix.file str              = /workspace/ubergarm/imatrix-Qwen3-235...
> llama_model_loader: - kv  33:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
> llama_model_loader: - kv  34:             quantize.imatrix.entries_count i32              = 753
> llama_model_loader: - kv  35:              quantize.imatrix.chunks_count i32              = 225
> llama_model_loader: - kv  36:                                   split.no u16              = 0
> llama_model_loader: - kv  37:                                split.count u16              = 5
> llama_model_loader: - kv  38:                        split.tensors.count i32              = 1131
> llama_model_loader: - type  f32:  471 tensors
> llama_model_loader: - type q8_0:   96 tensors
> llama_model_loader: - type iq6_k:  564 tensors
> llm_load_vocab: special tokens cache size = 26
> llm_load_vocab: token to piece cache size = 0.9311 MB
> llm_load_print_meta: format           = GGUF V3 (latest)
> llm_load_print_meta: arch             = qwen3moe
> llm_load_print_meta: vocab type       = BPE
> llm_load_print_meta: n_vocab          = 151936
> llm_load_print_meta: n_merges         = 151387
> llm_load_print_meta: vocab_only       = 0
> llm_load_print_meta: n_ctx_train      = 40960
> llm_load_print_meta: n_embd           = 4096
> llm_load_print_meta: n_layer          = 94
> llm_load_print_meta: n_head           = 64
> llm_load_print_meta: n_head_kv        = 4
> llm_load_print_meta: n_rot            = 128
> llm_load_print_meta: n_swa            = 0
> llm_load_print_meta: n_swa_pattern    = 1
> llm_load_print_meta: n_embd_head_k    = 128
> llm_load_print_meta: n_embd_head_v    = 128
> llm_load_print_meta: n_gqa            = 16
> llm_load_print_meta: n_embd_k_gqa     = 512
> llm_load_print_meta: n_embd_v_gqa     = 512
> llm_load_print_meta: f_norm_eps       = 0.0e+00
> llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
> llm_load_print_meta: f_clamp_kqv      = 0.0e+00
> llm_load_print_meta: f_max_alibi_bias = 0.0e+00
> llm_load_print_meta: f_logit_scale    = 0.0e+00
> llm_load_print_meta: n_ff             = 12288
> llm_load_print_meta: n_expert         = 128
> llm_load_print_meta: n_expert_used    = 8
> llm_load_print_meta: causal attn      = 1
> llm_load_print_meta: pooling type     = 0
> llm_load_print_meta: rope type        = 2
> llm_load_print_meta: rope scaling     = linear
> llm_load_print_meta: freq_base_train  = 1000000.0
> llm_load_print_meta: freq_scale_train = 1
> llm_load_print_meta: n_ctx_orig_yarn  = 40960
> llm_load_print_meta: rope_finetuned   = unknown
> llm_load_print_meta: ssm_d_conv       = 0
> llm_load_print_meta: ssm_d_inner      = 0
> llm_load_print_meta: ssm_d_state      = 0
> llm_load_print_meta: ssm_dt_rank      = 0
> llm_load_print_meta: model type       = ?B
> llm_load_print_meta: model ftype      = IQ6_K - 6.6 bpw
> llm_load_print_meta: model params     = 235.094 B
> llm_load_print_meta: model size       = 198.259 GiB (7.244 BPW) 
> llm_load_print_meta: repeating layers = 197.028 GiB (7.237 BPW, 233.849 B parameters)
> llm_load_print_meta: general.name     = Models
> llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
> llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
> llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
> llm_load_print_meta: LF token         = 148848 'ÄĬ'
> llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
> llm_load_print_meta: max token length = 256
> llm_load_print_meta: n_ff_exp         = 1536
> llm_load_tensors: ggml ctx size =    0.50 MiB
> llm_load_tensors: offloading 0 repeating layers to GPU
> llm_load_tensors: offloaded 0/95 layers to GPU
> llm_load_tensors:        CPU buffer size = 203017.61 MiB
> ....................................................................................................
> ============ Repacked 95 tensors
> llama_new_context_with_model: n_ctx      = 16384
> llama_new_context_with_model: n_batch    = 2048
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 0
> llama_new_context_with_model: attn_max_b = 0
> llama_new_context_with_model: fused_moe  = 1
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 1000000.0
> llama_new_context_with_model: freq_scale = 1
> llama_kv_cache_init:        CPU KV buffer size =  1598.00 MiB
> llama_new_context_with_model: KV self size  = 1598.00 MiB, K (q8_0):  799.00 MiB, V (q8_0):  799.00 MiB
> llama_new_context_with_model:        CPU  output buffer size =     0.58 MiB
> llama_new_context_with_model:        CPU compute buffer size =   304.75 MiB
> llama_new_context_with_model: graph nodes  = 3672
> llama_new_context_with_model: graph splits = 1
> 
> main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = -1, n_threads = 32, n_threads_batch = 32
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |   512 |    128 |      0 |    4.359 |   117.46 |    7.881 |    16.24 |
> |   512 |    128 |    512 |    4.373 |   117.09 |    9.647 |    13.27 |
> |   512 |    128 |   1024 |    4.610 |   111.06 |    8.834 |    14.49 |
> |   512 |    128 |   1536 |    4.641 |   110.33 |   10.037 |    12.75 |
> |   512 |    128 |   2048 |    4.876 |   105.01 |   13.032 |     9.82 |
> |   512 |    128 |   2560 |    4.976 |   102.89 |   14.779 |     8.66 |
> |   512 |    128 |   3072 |    5.062 |   101.14 |   13.366 |     9.58 |
> |   512 |    128 |   3584 |    5.151 |    99.40 |   13.447 |     9.52 |
> |   512 |    128 |   4096 |    5.239 |    97.74 |   15.546 |     8.23 |
> |   512 |    128 |   4608 |    5.356 |    95.60 |   13.975 |     9.16 |
> |   512 |    128 |   5120 |    5.455 |    93.87 |   13.837 |     9.25 |
> |   512 |    128 |   5632 |    5.543 |    92.37 |   15.700 |     8.15 |
> |   512 |    128 |   6144 |    5.766 |    88.80 |   14.063 |     9.10 |
> |   512 |    128 |   6656 |    5.768 |    88.77 |   14.064 |     9.10 |
> |   512 |    128 |   7168 |    5.923 |    86.44 |   16.084 |     7.96 |
> |   512 |    128 |   7680 |    6.006 |    85.25 |   14.581 |     8.78 |
> |   512 |    128 |   8192 |    6.145 |    83.32 |   14.257 |     8.98 |
> |   512 |    128 |   8704 |    6.153 |    83.22 |   16.262 |     7.87 |
> |   512 |    128 |   9216 |    6.258 |    81.82 |   15.010 |     8.53 |
> |   512 |    128 |   9728 |    6.395 |    80.06 |   14.768 |     8.67 |
> |   512 |    128 |  10240 |    6.575 |    77.87 |   16.422 |     7.79 |
> |   512 |    128 |  10752 |    6.695 |    76.48 |   14.911 |     8.58 |
> |   512 |    128 |  11264 |    6.817 |    75.11 |   14.985 |     8.54 |
> |   512 |    128 |  11776 |    6.784 |    75.48 |   16.958 |     7.55 |
> |   512 |    128 |  12288 |    6.937 |    73.81 |   15.634 |     8.19 |
> |   512 |    128 |  12800 |    7.936 |    64.52 |   15.454 |     8.28 |
> |   512 |    128 |  13312 |    7.044 |    72.69 |   15.884 |     8.06 |
> |   512 |    128 |  13824 |    7.244 |    70.68 |   15.820 |     8.09 |
> |   512 |    128 |  14336 |    7.365 |    69.52 |   17.121 |     7.48 |
> |   512 |    128 |  14848 |    7.635 |    67.06 |   16.042 |     7.98 |
> |   512 |    128 |  15360 |    7.601 |    67.36 |   15.867 |     8.07 |
> |   512 |    128 |  15872 |    7.696 |    66.53 |   17.578 |     7.28 |
> ```
> 
> </details>
> 
> ![performance_comparison_pp](https://github.com/user-attachments/assets/e37399af-4851-46f1-a73f-2611a51038da)
> 
> ![performance_comparison_tg](https://github.com/user-attachments/assets/6540f4da-1469-422b-a5cf-d39a45818016)
> 
> 👤 **ikawrakow** replied the **2025-05-03** at **08:08:26**:<br>
> So, ~30% better for PP, but not much difference for TG. I need to understand the cause of the sharp drop in TG performance for the first ~2k tokens. I'll investigate.
> 
> Thanks a lot for these benchmarks!
> 
> 👤 **AesSedai** replied the **2025-05-03** at **08:10:21**:<br>
> You're welcome, let me know if you want me to re-run any of these benchmarks at some point in the future and I can pull / rebuild / re-test. Excited to see what shakes out!
> 
> 👤 **VinnyG9** replied the **2025-05-19** at **14:08:09**:<br>
> > That's good news!
> > 
> > Yes, this is with hyperthreading. Out of the 64 threads on the system, 56 are passed through to the virtual machine and I have it configured to use 48 of those during the sweep.
> > 
> > Is there a particular `-t` count (or thread passthrough count) you would like me to try?
> 
> hey, no idea what hypervisor you're running but unless it's ESXi big chances it can't handle AMD multi CCD chips or numa systems in general
> 
> unless you pin ONE CCD to your VM performance is likely taking a hit