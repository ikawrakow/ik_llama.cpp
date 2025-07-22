### 🔀 [#374](https://github.com/ikawrakow/ik_llama.cpp/pull/374) - CUDA: MMQ for IQ4_KS

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-04 |
| **Updated** | 2025-05-07 |

---

#### Description

`IQX_K` quants offer better quantization quality for the same amount of bits spent compared to k- and i-quants. But on CUDA they are slower for prompt processing (PP) because matrix multiplications are done via dequantize->cuBLAS, so I thought it is time to fix this.

This PR adds quantized matrix multiplications, also known as MMQ, for `IQ4_KS`.

The following graph shows PP performance as a function of the number of tokens in the KV cache `N_KV` for the main branch (black) and the PR (red). Model is LLaMA-3.1-8B-Instruct, GPU is RTX-4080. We see a very nice performance improvement in the range of 25%.

![z4](https://github.com/user-attachments/assets/807ce486-4398-431c-a98e-536a3eb546dd)

<details>
<summary>Main branch</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.128 |  3994.38 |    0.995 |   128.62 |
|   512 |    128 |    512 |    0.091 |  5635.54 |    1.003 |   127.59 |
|   512 |    128 |   1024 |    0.093 |  5526.71 |    1.016 |   126.03 |
|   512 |    128 |   1536 |    0.095 |  5405.29 |    1.030 |   124.31 |
|   512 |    128 |   2048 |    0.096 |  5308.45 |    1.046 |   122.40 |
|   512 |    128 |   2560 |    0.098 |  5237.80 |    1.061 |   120.63 |
|   512 |    128 |   3072 |    0.101 |  5079.26 |    1.079 |   118.59 |
|   512 |    128 |   3584 |    0.101 |  5052.15 |    1.095 |   116.86 |
|   512 |    128 |   4096 |    0.103 |  4965.28 |    1.113 |   114.97 |
|   512 |    128 |   4608 |    0.105 |  4883.49 |    1.128 |   113.47 |
|   512 |    128 |   5120 |    0.107 |  4783.71 |    1.152 |   111.10 |
|   512 |    128 |   5632 |    0.109 |  4713.94 |    1.158 |   110.56 |
|   512 |    128 |   6144 |    0.110 |  4644.54 |    1.171 |   109.30 |
|   512 |    128 |   6656 |    0.112 |  4573.92 |    1.184 |   108.10 |
|   512 |    128 |   7168 |    0.114 |  4498.61 |    1.198 |   106.88 |
|   512 |    128 |   7680 |    0.116 |  4421.23 |    1.211 |   105.68 |
|   512 |    128 |   8192 |    0.118 |  4345.69 |    1.225 |   104.46 |
|   512 |    128 |   8704 |    0.120 |  4279.68 |    1.239 |   103.34 |
|   512 |    128 |   9216 |    0.121 |  4220.63 |    1.253 |   102.17 |
|   512 |    128 |   9728 |    0.123 |  4151.40 |    1.281 |    99.89 |
|   512 |    128 |  10240 |    0.125 |  4088.80 |    1.293 |    98.99 |
|   512 |    128 |  10752 |    0.127 |  4034.39 |    1.297 |    98.72 |
|   512 |    128 |  11264 |    0.129 |  3963.86 |    1.308 |    97.83 |
|   512 |    128 |  11776 |    0.130 |  3927.22 |    1.321 |    96.90 |
|   512 |    128 |  12288 |    0.132 |  3864.65 |    1.334 |    95.93 |
|   512 |    128 |  12800 |    0.135 |  3803.55 |    1.350 |    94.83 |
|   512 |    128 |  13312 |    0.136 |  3753.64 |    1.363 |    93.89 |
|   512 |    128 |  13824 |    0.138 |  3698.46 |    1.379 |    92.80 |
|   512 |    128 |  14336 |    0.140 |  3649.74 |    1.392 |    91.93 |
|   512 |    128 |  14848 |    0.142 |  3600.23 |    1.418 |    90.24 |
|   512 |    128 |  15360 |    0.145 |  3531.69 |    1.429 |    89.60 |
|   512 |    128 |  15872 |    0.146 |  3496.17 |    1.442 |    88.79 |

</details>

<details>
<summary>PR</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.107 |  4778.97 |    0.995 |   128.59 |
|   512 |    128 |    512 |    0.068 |  7487.24 |    1.003 |   127.58 |
|   512 |    128 |   1024 |    0.070 |  7337.56 |    1.015 |   126.16 |
|   512 |    128 |   1536 |    0.072 |  7143.26 |    1.030 |   124.23 |
|   512 |    128 |   2048 |    0.073 |  6976.14 |    1.046 |   122.32 |
|   512 |    128 |   2560 |    0.074 |  6896.64 |    1.064 |   120.30 |
|   512 |    128 |   3072 |    0.077 |  6618.49 |    1.079 |   118.68 |
|   512 |    128 |   3584 |    0.079 |  6496.14 |    1.093 |   117.06 |
|   512 |    128 |   4096 |    0.080 |  6367.76 |    1.112 |   115.14 |
|   512 |    128 |   4608 |    0.082 |  6212.61 |    1.127 |   113.61 |
|   512 |    128 |   5120 |    0.083 |  6179.25 |    1.151 |   111.17 |
|   512 |    128 |   5632 |    0.085 |  6045.51 |    1.158 |   110.55 |
|   512 |    128 |   6144 |    0.087 |  5889.32 |    1.170 |   109.43 |
|   512 |    128 |   6656 |    0.088 |  5815.14 |    1.183 |   108.18 |
|   512 |    128 |   7168 |    0.092 |  5592.88 |    1.196 |   106.99 |
|   512 |    128 |   7680 |    0.094 |  5473.71 |    1.210 |   105.76 |
|   512 |    128 |   8192 |    0.095 |  5367.61 |    1.225 |   104.51 |
|   512 |    128 |   8704 |    0.097 |  5286.96 |    1.237 |   103.50 |
|   512 |    128 |   9216 |    0.099 |  5192.65 |    1.251 |   102.35 |
|   512 |    128 |   9728 |    0.101 |  5050.26 |    1.279 |   100.07 |
|   512 |    128 |  10240 |    0.102 |  4997.66 |    1.290 |    99.19 |
|   512 |    128 |  10752 |    0.104 |  4906.99 |    1.294 |    98.90 |
|   512 |    128 |  11264 |    0.106 |  4850.78 |    1.306 |    97.98 |
|   512 |    128 |  11776 |    0.108 |  4745.57 |    1.320 |    96.97 |
|   512 |    128 |  12288 |    0.110 |  4664.34 |    1.332 |    96.09 |
|   512 |    128 |  12800 |    0.112 |  4582.72 |    1.347 |    95.00 |
|   512 |    128 |  13312 |    0.113 |  4522.89 |    1.360 |    94.09 |
|   512 |    128 |  13824 |    0.114 |  4485.80 |    1.376 |    93.02 |
|   512 |    128 |  14336 |    0.117 |  4386.19 |    1.389 |    92.13 |
|   512 |    128 |  14848 |    0.119 |  4311.14 |    1.417 |    90.32 |
|   512 |    128 |  15360 |    0.120 |  4249.60 |    1.426 |    89.74 |
|   512 |    128 |  15872 |    0.124 |  4143.10 |    1.439 |    88.94 |

</details>


Are you wondering why PP performance for `N_KV = 0` is significantly lower? I did as well, so I checked `llama-sweep-bench`, the tool with which the data for this graph is generated.  Warm-up is done via a single TG run. I checked that if I add another warn-up run with `n_ubatch` tokens, performance for `N_KV = 0` becomes higher than `N_KV = 512` as expected. I guess, I will submit a separate PR for that.

TG performance is not affected at all by the PR, so no graph for that.

---

#### 💬 Conversation

👤 **saood06** commented the **2025-05-04** at **07:33:54**:<br>

> I checked that if I add another warn-up run with n_ubatch tokens, performance for N_KV = 0 becomes higher than N_KV = 512 as expected. I guess, I will submit a separate PR for that.

Interesting, I've always dealt with it by either comparing the second row (as it is generally more stable between runs anyways) or just running a very low context sweep-bench as a warmup

---

👤 **ikawrakow** commented the **2025-05-04** at **07:41:21**:<br>

> Interesting, I've always dealt with it by either comparing the second row (as it is generally more stable between runs anyways) or just running a very low context sweep-bench as a warmup

It does not affect CPU performance. But on CUDA the time it takes to find and load the pre-compiled kernels is not negligible when compared to the time for computing a batch (well, at least for the 8B model I used here). I had noticed this peculiar behavior, but as I have been testing mostly MoE models lately I thought it was somehow related to that (we know MoE models do better with larger u-batches).

I'll make the PP warm-up pass optional via a command line argument as for very large models on the CPU it does take some time to process a batch of 512 tokens.

---

👤 **saood06** commented the **2025-05-04** at **07:52:57**:<br>

>It does not affect CPU performance.

I just looked back at my notes/logs, it is the first TG for CPU that does vary, and the cause is different as there is corresponding disk activity that is almost certainly to blame (very little but still some, and even a single HDD seek can sometime be seen from the numbers in my experience). I have done GPU speed testing but I generally don't look at the PP results especially not at low contexts so I never reran to see it go away.

>I'll make the PP warm-up pass optional via a command line argument as for very large models on the CPU it does take some time to process a batch of 512 tokens.

I was going to suggest that, as that is very true for some of my testing.

---

👤 **ubergarm** commented the **2025-05-07** at **22:02:48**:<br>

I'm working on some benchmarks for various Qwen3-30B-A3B quants and ran some llama-sweep-benches and this PR is looking good for your `IQ4_KS`. Used the `--warmup-batch` PR as well.

## ik_llama.cpp
![Qwen3-30B-A3B-ik-ggufs](https://github.com/user-attachments/assets/5529cd92-f733-4a00-a482-ab6672a3ba58)

## mainline
![Qwen3-30B-A3B-mainline-gguf-roundup](https://github.com/user-attachments/assets/0d855616-455e-4ba1-875c-f6b4570f394d)