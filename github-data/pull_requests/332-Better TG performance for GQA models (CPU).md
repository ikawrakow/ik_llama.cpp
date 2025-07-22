### 🔀 [#332](https://github.com/ikawrakow/ik_llama.cpp/pull/332) - Better TG performance for GQA models (CPU)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-16 |
| **Updated** | 2025-04-17 |

---

#### Description

This PR adds improved TG performance on the CPU for GQA models (LLaMA-2+, Gemma, etc.).
We see performance gains with and without FA. The gains without FA are fairly minor and come from a different way of distributing the work between the threads for the `K*Q` and `V*softmax(K*Q)` matrix multiplications. The performance gains with FA enabled are very significant, and FA now outperforms no-FA also for TG.

Here is an example for LLaMA-3.1-8B-Instruct. Model is quantized with `Q4_0`, KV cache is `Q8_0` (V-cache is `f16` when FA is not enabled). Results are for a R$yzen-5975WX CPU (vanilla `AVX2`). Also included for comparison are mainline `llama.cpp` results (build 5139) with FA enabled shown with orange symbols. Results are obtained with `llama-sweet-bench` using
```
./bin/llama-sweep-bench -m $model -c 10240 -ctk q8_0 -ctv q8_0 -t 32 -fa
```
The x-axis is `N_KV`, the number of tokens in the KV cache. 

![l3_sweep](https://github.com/user-attachments/assets/5db5d3e8-1615-43b8-a483-177ac851a131)

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-04-16** at **16:04:38**:<br>

Here another comparison to mainline, this time for Gemma3-12B-Instruct. Only runs with FA enabled, `Q8_0` KV-cache, `Q4_0` quantized model, Risen-5975WX CPU. I have rerun the mainline benchmark multiple times, dropping caches or not between runs, and the peculiar sudden drop in performance for the first 1024 tokens in the KV cache remained unchanged.  Here mainline does significantly better relative to `ik_llama.cpp` compared to LLaMA-3.1-8B in the above graph. I suspect this is due to the fact that the benefit from the improvement this PR adds is less. Gemma3 has 16 attention heads in total and 8 KV heads. This results in the `K*Q` and `V*softmax(K*Q)` GEMM's for TG to be done with matrices with just 2 rows (compared to 4 rows for LLaMA-3), so the gain from using GEMM instead of GEMV is less. It is also possible that there is something in mainline that makes it perform better with the Gemma3 head size of 256 (vs 128 for LLaMA-3). The mainline CPU code has changed a lot since I left the project, so I cannot say I know very well what happens there.  
 
![g3_sweep](https://github.com/user-attachments/assets/ec50809b-3838-42a3-855d-8ff244b976ce)

---

👤 **saood06** commented the **2025-04-17** at **00:32:59**:<br>

>and FA now outperforms no-FA also for TG.

Nice.

>Results are obtained with `llama-sweet-bench` using
> 
> ```
> ./bin/llama-sweep-bench -m $model -c 10240 -ctk q8_0 -ctv q8_0 -t 32 -fa
> ```
> 

Do you still have the raw markdown results? I know PP wasn't affected by this PR but I'm curious where it stands vs mainline.

>Here mainline does significantly better relative to ik_llama.cpp compared to LLaMA-3.1-8B in the above graph.

I wonder if they cross over at higher contexts the gap does seem to be closing here.

---

👤 **ikawrakow** commented the **2025-04-17** at **05:54:21**:<br>

> Do you still have the raw markdown results? I know PP wasn't affected by this PR but I'm curious where it stands vs mainline.

Mainline PP performance with FA is embarrassing. I also picked the fastest mainline quant that receives an extraordinary amount of attention (`Q4_0`). I had not kept the logs, so reran `sweep-bench` this morning up to a context of 16k. This particular computer is quite sensitive to dropping caches between runs. It seems also that results are somewhat sensitive to the amount of KV cache allocated, so slightly different from yesterday.

### Gemma3-12B-Instruct

At 16k tokens mainline TG performance is indeed slightly better than `ik_llama.cpp`. But mainline PP performance drops from 55.5% at zero context to 42.4% at 16k tokens.

* Mainline

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.669 |   109.67 |   12.164 |    10.52 |
|   512 |    128 |    512 |    4.811 |   106.42 |   13.061 |     9.80 |
|   512 |    128 |   1024 |    5.049 |   101.40 |   13.818 |     9.26 |
|   512 |    128 |   1536 |    5.164 |    99.15 |   13.960 |     9.17 |
|   512 |    128 |   2048 |    5.280 |    96.97 |   14.107 |     9.07 |
|   512 |    128 |   2560 |    5.423 |    94.40 |   14.248 |     8.98 |
|   512 |    128 |   3072 |    5.619 |    91.11 |   14.395 |     8.89 |
|   512 |    128 |   3584 |    5.823 |    87.92 |   14.535 |     8.81 |
|   512 |    128 |   4096 |    6.070 |    84.35 |   14.677 |     8.72 |
|   512 |    128 |   4608 |    6.306 |    81.19 |   14.825 |     8.63 |
|   512 |    128 |   5120 |    6.547 |    78.20 |   14.969 |     8.55 |
|   512 |    128 |   5632 |    6.890 |    74.31 |   15.131 |     8.46 |
|   512 |    128 |   6144 |    7.227 |    70.85 |   15.281 |     8.38 |
|   512 |    128 |   6656 |    7.513 |    68.15 |   15.394 |     8.32 |
|   512 |    128 |   7168 |    7.918 |    64.67 |   15.537 |     8.24 |
|   512 |    128 |   7680 |    8.334 |    61.43 |   15.680 |     8.16 |
|   512 |    128 |   8192 |    8.800 |    58.18 |   15.830 |     8.09 |
|   512 |    128 |   8704 |    9.200 |    55.65 |   15.971 |     8.01 |
|   512 |    128 |   9216 |    9.523 |    53.76 |   16.101 |     7.95 |
|   512 |    128 |   9728 |   10.048 |    50.95 |   16.242 |     7.88 |
|   512 |    128 |  10240 |   10.495 |    48.78 |   16.371 |     7.82 |
|   512 |    128 |  10752 |   10.955 |    46.73 |   16.507 |     7.75 |
|   512 |    128 |  11264 |   11.375 |    45.01 |   16.662 |     7.68 |
|   512 |    128 |  11776 |   11.837 |    43.26 |   16.798 |     7.62 |
|   512 |    128 |  12288 |   12.320 |    41.56 |   16.949 |     7.55 |
|   512 |    128 |  12800 |   12.613 |    40.59 |   17.085 |     7.49 |
|   512 |    128 |  13312 |   12.815 |    39.95 |   17.208 |     7.44 |
|   512 |    128 |  13824 |   13.100 |    39.08 |   17.364 |     7.37 |
|   512 |    128 |  14336 |   13.466 |    38.02 |   17.518 |     7.31 |
|   512 |    128 |  14848 |   13.669 |    37.46 |   17.655 |     7.25 |
|   512 |    128 |  15360 |   13.789 |    37.13 |   17.797 |     7.19 |
|   512 |    128 |  15872 |   13.874 |    36.90 |   17.937 |     7.14 |

* `ik_llama.cpp`

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.593 |   197.46 |   12.301 |    10.41 |
|   512 |    128 |    512 |    2.662 |   192.34 |   12.501 |    10.24 |
|   512 |    128 |   1024 |    2.756 |   185.77 |   12.703 |    10.08 |
|   512 |    128 |   1536 |    2.854 |   179.42 |   12.946 |     9.89 |
|   512 |    128 |   2048 |    2.946 |   173.78 |   13.143 |     9.74 |
|   512 |    128 |   2560 |    3.040 |   168.42 |   13.331 |     9.60 |
|   512 |    128 |   3072 |    3.136 |   163.26 |   13.507 |     9.48 |
|   512 |    128 |   3584 |    3.235 |   158.25 |   13.711 |     9.34 |
|   512 |    128 |   4096 |    3.336 |   153.48 |   13.907 |     9.20 |
|   512 |    128 |   4608 |    3.432 |   149.20 |   14.088 |     9.09 |
|   512 |    128 |   5120 |    3.530 |   145.05 |   14.290 |     8.96 |
|   512 |    128 |   5632 |    3.632 |   140.99 |   14.483 |     8.84 |
|   512 |    128 |   6144 |    3.729 |   137.31 |   14.673 |     8.72 |
|   512 |    128 |   6656 |    3.834 |   133.53 |   14.879 |     8.60 |
|   512 |    128 |   7168 |    3.934 |   130.14 |   15.074 |     8.49 |
|   512 |    128 |   7680 |    4.046 |   126.55 |   15.266 |     8.38 |
|   512 |    128 |   8192 |    4.140 |   123.67 |   15.443 |     8.29 |
|   512 |    128 |   8704 |    4.243 |   120.66 |   15.616 |     8.20 |
|   512 |    128 |   9216 |    4.342 |   117.91 |   15.838 |     8.08 |
|   512 |    128 |   9728 |    4.450 |   115.06 |   16.008 |     8.00 |
|   512 |    128 |  10240 |    4.552 |   112.48 |   16.197 |     7.90 |
|   512 |    128 |  10752 |    4.721 |   108.46 |   16.429 |     7.79 |
|   512 |    128 |  11264 |    4.762 |   107.51 |   16.622 |     7.70 |
|   512 |    128 |  11776 |    4.869 |   105.16 |   16.823 |     7.61 |
|   512 |    128 |  12288 |    4.973 |   102.96 |   16.982 |     7.54 |
|   512 |    128 |  12800 |    5.077 |   100.84 |   17.208 |     7.44 |
|   512 |    128 |  13312 |    5.175 |    98.93 |   17.419 |     7.35 |
|   512 |    128 |  13824 |    5.278 |    97.02 |   17.603 |     7.27 |
|   512 |    128 |  14336 |    5.461 |    93.75 |   17.798 |     7.19 |
|   512 |    128 |  14848 |    5.560 |    92.08 |   19.126 |     7.12 |
|   512 |    128 |  15360 |    5.717 |    89.55 |   19.383 |     7.06 |
|   512 |    128 |  15872 |    5.891 |    86.91 |   19.640 |     7.00 |

### LLaMA-3.1-8B-Instruct

Here mainline does not do well for PP or TG. Mainline TG is 55.5% of `ik_llama.cpp` at 16k tokens. Mainline PP is totally embarrassing. It starts at about 60% of `ik_llama.cpp` for zero context, and finishes at 7.2% at 16k (14X slower). So, whatever was done to optimize performance for a head size of 256, it is a killer for a head size of 128 (the most common head size). Here the data:

* Mainline

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    2.737 |   187.04 |    7.548 |    16.96 |
|   512 |    128 |    512 |    3.185 |   160.76 |    7.953 |    16.09 |
|   512 |    128 |   1024 |    3.721 |   137.60 |    8.409 |    15.22 |
|   512 |    128 |   1536 |    4.219 |   121.35 |    8.826 |    14.50 |
|   512 |    128 |   2048 |    4.711 |   108.68 |    9.199 |    13.91 |
|   512 |    128 |   2560 |    5.206 |    98.34 |    9.592 |    13.34 |
|   512 |    128 |   3072 |    5.704 |    89.76 |    9.980 |    12.83 |
|   512 |    128 |   3584 |    6.252 |    81.89 |   10.370 |    12.34 |
|   512 |    128 |   4096 |    6.867 |    74.55 |   10.765 |    11.89 |
|   512 |    128 |   4608 |    7.507 |    68.20 |   11.157 |    11.47 |
|   512 |    128 |   5120 |    8.231 |    62.21 |   11.552 |    11.08 |
|   512 |    128 |   5632 |    9.214 |    55.57 |   11.941 |    10.72 |
|   512 |    128 |   6144 |   10.467 |    48.91 |   12.330 |    10.38 |
|   512 |    128 |   6656 |   11.646 |    43.96 |   12.713 |    10.07 |
|   512 |    128 |   7168 |   13.104 |    39.07 |   13.109 |     9.76 |
|   512 |    128 |   7680 |   14.813 |    34.56 |   13.500 |     9.48 |
|   512 |    128 |   8192 |   16.570 |    30.90 |   13.885 |     9.22 |
|   512 |    128 |   8704 |   18.246 |    28.06 |   14.277 |     8.97 |
|   512 |    128 |   9216 |   20.142 |    25.42 |   14.675 |     8.72 |
|   512 |    128 |   9728 |   21.729 |    23.56 |   15.072 |     8.49 |
|   512 |    128 |  10240 |   23.615 |    21.68 |   15.454 |     8.28 |
|   512 |    128 |  10752 |   25.406 |    20.15 |   15.840 |     8.08 |
|   512 |    128 |  11264 |   27.299 |    18.76 |   16.236 |     7.88 |
|   512 |    128 |  11776 |   29.122 |    17.58 |   16.625 |     7.70 |
|   512 |    128 |  12288 |   31.079 |    16.47 |   17.012 |     7.52 |
|   512 |    128 |  12800 |   33.052 |    15.49 |   17.407 |     7.35 |
|   512 |    128 |  13312 |   34.958 |    14.65 |   17.796 |     7.19 |
|   512 |    128 |  13824 |   37.170 |    13.77 |   18.188 |     7.04 |
|   512 |    128 |  14336 |   39.425 |    12.99 |   18.570 |     6.89 |
|   512 |    128 |  14848 |   41.661 |    12.29 |   18.959 |     6.75 |
|   512 |    128 |  15360 |   43.766 |    11.70 |   19.350 |     6.62 |
|   512 |    128 |  15872 |   46.129 |    11.10 |   19.730 |     6.49 |

* `ik_llama.cpp`

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.638 |   312.56 |    7.739 |    16.54 |
|   512 |    128 |    512 |    1.661 |   308.28 |    7.852 |    16.30 |
|   512 |    128 |   1024 |    1.705 |   300.35 |    7.961 |    16.08 |
|   512 |    128 |   1536 |    1.766 |   289.90 |    8.075 |    15.85 |
|   512 |    128 |   2048 |    1.806 |   283.52 |    8.170 |    15.67 |
|   512 |    128 |   2560 |    1.860 |   275.34 |    8.261 |    15.50 |
|   512 |    128 |   3072 |    1.914 |   267.51 |    8.363 |    15.31 |
|   512 |    128 |   3584 |    1.981 |   258.45 |    8.468 |    15.11 |
|   512 |    128 |   4096 |    2.022 |   253.22 |    8.592 |    14.90 |
|   512 |    128 |   4608 |    2.076 |   246.61 |    8.706 |    14.70 |
|   512 |    128 |   5120 |    2.132 |   240.12 |    8.800 |    14.55 |
|   512 |    128 |   5632 |    2.189 |   233.92 |    8.902 |    14.38 |
|   512 |    128 |   6144 |    2.240 |   228.58 |    8.998 |    14.23 |
|   512 |    128 |   6656 |    2.298 |   222.81 |    9.093 |    14.08 |
|   512 |    128 |   7168 |    2.352 |   217.66 |    9.191 |    13.93 |
|   512 |    128 |   7680 |    2.407 |   212.69 |    9.297 |    13.77 |
|   512 |    128 |   8192 |    2.462 |   207.92 |    9.409 |    13.60 |
|   512 |    128 |   8704 |    2.519 |   203.22 |    9.514 |    13.45 |
|   512 |    128 |   9216 |    2.573 |   199.02 |    9.619 |    13.31 |
|   512 |    128 |   9728 |    2.630 |   194.71 |    9.702 |    13.19 |
|   512 |    128 |  10240 |    2.683 |   190.82 |    9.796 |    13.07 |
|   512 |    128 |  10752 |    2.739 |   186.91 |    9.904 |    12.92 |
|   512 |    128 |  11264 |    2.795 |   183.19 |   10.018 |    12.78 |
|   512 |    128 |  11776 |    2.851 |   179.62 |   10.124 |    12.64 |
|   512 |    128 |  12288 |    2.905 |   176.24 |   10.228 |    12.51 |
|   512 |    128 |  12800 |    2.963 |   172.78 |   10.321 |    12.40 |
|   512 |    128 |  13312 |    3.018 |   169.64 |   10.413 |    12.29 |
|   512 |    128 |  13824 |    3.078 |   166.34 |   10.538 |    12.15 |
|   512 |    128 |  14336 |    3.133 |   163.43 |   10.632 |    12.04 |
|   512 |    128 |  14848 |    3.192 |   160.40 |   10.738 |    11.92 |
|   512 |    128 |  15360 |    3.249 |   157.61 |   10.838 |    11.81 |
|   512 |    128 |  15872 |    3.305 |   154.91 |   10.942 |    11.70 |

Btw, my surprise at the 6X drop in PP performance for DeepSeek-V3/R1 that I expressed elsewhere was based on results such as these. `ik_llama.cpp` PP performance at 16k tokens is 2X lower for LLaMA-3.1, and 2.3X lower for Gemma3.

---

👤 **saood06** commented the **2025-04-17** at **07:45:00**:<br>

> Mainline PP performance with FA is embarrassing. 

It is really nice being able to use FA here and benefit.

>I also picked the fastest mainline quant that receives an extraordinary amount of attention (`Q4_0`). 

For gemma this also makes the most sense as they released QAT versions of Q4_0 ([this](https://huggingface.co/Dampfinchen/google-gemma-3-12b-it-qat-q4_0-gguf-small-fix) being the best version for 12B, some measurements [here](https://huggingface.co/Dampfinchen/google-gemma-3-12b-it-qat-q4_0-gguf-small-fix)).

>I had not kept the logs, so reran `sweep-bench` this morning up to a context of 16k. 

Thanks for doing that.

>It seems also that results are somewhat sensitive to the amount of KV cache allocated, so slightly different from yesterday.

Ya surprisingly the newer run with higher KV performed better looking at both.


> At 16k tokens mainline TG performance is indeed slightly better than `ik_llama.cpp`. 

Here's the visual generated with the python script in the sweep-bench example folder, in order to see the crossover point.

![performance_comparison_tg](https://github.com/user-attachments/assets/5358c7b9-5301-40b1-b665-fa9efa4acfa7)


>But mainline PP performance drops from 55.5% at zero context to 42.4% at 16k tokens.

Yes both model the PP graphs just show ik_llama clearly above mainline.

> ### LLaMA-3.1-8B-Instruct
> 
> Here mainline does not do well for PP or TG. Mainline TG is 55.5% of `ik_llama.cpp` at 16k tokens. Mainline PP is totally embarrassing. It starts at about 60% of `ik_llama.cpp` for zero context, and finishes at 7.2% at 16k (14X slower). So, whatever was done to optimize performance for a head size of 256, it is a killer for a head size of 128 (the most common head size). Here the data:

PP graph again not very interesting but TG is interesting showing the different curves.

![image](https://github.com/user-attachments/assets/87d6baa6-82ff-41f1-bac2-5e3ea16154d1)

> Btw, my surprise at the 6X drop in PP performance for DeepSeek-V3/R1 that I expressed elsewhere was based on results such as these. `ik_llama.cpp` PP performance at 16k tokens is 2X lower for LLaMA-3.1, and 2.3X lower for Gemma3.

Ya that architecture's performance surprises me too like when I saw peak batched TG performance for Deepseek being higher than PP performance instead of just approaching it like I normally observe.