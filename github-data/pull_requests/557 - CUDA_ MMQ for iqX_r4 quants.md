### 🔀 [#557](https://github.com/ikawrakow/ik_llama.cpp/pull/557) - CUDA: MMQ for iqX_r4 quants 

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-25 |
| **Updated** | 2025-06-26 |

---

#### Description

CUDA matrix multiplications for `IQ2_K_R4, ..., IQ5_K_R4` quants on the main branch are implemented via deqantize to `fp16` (or `bf16`) + cuBLAS. As a result, there is a constant overhead for the dequantization step, which leads to relatively low performance when the number of tokens being processed is small. This is often the case for MoE models with many experts where each expert "sees" a small fraction of the tokens. For instance, for DeepSeek-R1/V3, for a batch size of 4096 tokens, experts will process on average just 128 tokens.

This PR addresses the issue by adding quantized matrix multiplication kernels (a.k.a., MMQ) for `IQ2_K_R4, IQ3_K_R4, IQ4_K_R4, IQ5_K_R4`. 

The benefit is illustrated with the following graph, which shows prompt processing performance as a function of prompt length for LlaMA-3.1-8B-Instruct using pure `IQ2_K_R4` quantization. GPU is RTX-4080. Black circles are for the main branch, red circles for this PR. While working on the PR I made the interesting observation that for these quants (all have block size of 16 weights, so use the much less efficient MMQ kernel template), dequantize+cuBLAS becomes faster than MMQ for batch sizes greater than 1000 tokens or so. Hence in the PR MMQ gets used for batches of fewer than 1024 tokens. The blue circles show MMQ-only. At 128 tokens, the new MMQ implementation is two times faster than dequantize+cuBLAS, so I expect to see a positive impact on prompt processing speed for @ubergarm's `*_R4` DeepSeek models.

![iqk](https://github.com/user-attachments/assets/0068aab7-fcc9-498f-b93c-c2a9759abd19)

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-06-25** at **15:39:08**:<br>

Ran one test of my `IQ2_K_R4` on the Thread Ripper Pro 24x core offloading some layers onto 2x RTX A6000 GPUs showing some uplift for PP with this PR. I didn't try larger batch sizes at it sounds like this mostly benefits smaller batch sizes. Also I could have offloaded a couple more layers at least which would likely help given this boosts the CUDA code path speeds.

<details>

<summary>👈 sweep-bench command, data, and screen-shot of nvtop</summary>

I had some VRAM left so could proably have taken another layer or two each GPU.

![pr557-nvtop-screenshot](https://github.com/user-attachments/assets/a2568709-abb2-4e03-acfe-7d59920b2dfe)

```bash
model=DeepSeek-R1-0528-IQ2_K_R4-00001-of-00005.gguf
./build/bin/llama-sweep-bench \
    --model "$model" \
    --no-mmap \
    --ctx-size 8704 \
    -ctk q8_0 \
    -mla 3 -fa \
    -fmoe \
    -amb 512 \
    -ngl 99 \
    -ot "blk\.(3|4|5|6|7|8|9|10|11|12)\.ffn_.*=CUDA0" \
    -ot "blk\.(13|14|15|16|17|18|19|20|21|22)\.ffn_.*=CUDA1" \
    -ot exps=CPU \
    --warmup-batch \
    --threads 24
```

## PR557@b3417c93
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    3.891 |   131.57 |    8.242 |    15.53 |
|   512 |    128 |    512 |    4.628 |   110.62 |    8.311 |    15.40 |
|   512 |    128 |   1024 |    4.355 |   117.56 |    8.197 |    15.62 |
|   512 |    128 |   1536 |    4.240 |   120.76 |    8.299 |    15.42 |
|   512 |    128 |   2048 |    4.268 |   119.97 |    8.253 |    15.51 |
|   512 |    128 |   2560 |    4.660 |   109.88 |    8.490 |    15.08 |
|   512 |    128 |   3072 |    4.418 |   115.89 |    8.573 |    14.93 |
|   512 |    128 |   3584 |    4.550 |   112.52 |    8.517 |    15.03 |
|   512 |    128 |   4096 |    5.525 |    92.67 |    8.552 |    14.97 |
|   512 |    128 |   4608 |    4.770 |   107.33 |    8.485 |    15.09 |
|   512 |    128 |   5120 |    4.931 |   103.84 |    8.585 |    14.91 |
|   512 |    128 |   5632 |    4.901 |   104.47 |    8.975 |    14.26 |
|   512 |    128 |   6144 |    5.039 |   101.61 |    8.812 |    14.53 |
|   512 |    128 |   6656 |    5.124 |    99.93 |    8.901 |    14.38 |
|   512 |    128 |   7168 |    5.119 |   100.02 |    8.961 |    14.28 |
|   512 |    128 |   7680 |    5.200 |    98.46 |    8.836 |    14.49 |
|   512 |    128 |   8192 |    5.363 |    95.46 |    9.309 |    13.75 |

## main@b5f2f001
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.348 |   117.76 |    8.091 |    15.82 |
|   512 |    128 |    512 |    4.418 |   115.89 |    8.195 |    15.62 |
|   512 |    128 |   1024 |    4.520 |   113.27 |    8.200 |    15.61 |
|   512 |    128 |   1536 |    4.695 |   109.06 |    8.220 |    15.57 |
|   512 |    128 |   2048 |    4.787 |   106.96 |    8.258 |    15.50 |
|   512 |    128 |   2560 |    4.871 |   105.11 |    8.389 |    15.26 |
|   512 |    128 |   3072 |    4.960 |   103.23 |    8.453 |    15.14 |
|   512 |    128 |   3584 |    5.034 |   101.71 |    8.466 |    15.12 |
|   512 |    128 |   4096 |    5.152 |    99.37 |    8.448 |    15.15 |
|   512 |    128 |   4608 |    5.352 |    95.66 |    8.502 |    15.06 |
|   512 |    128 |   5120 |    5.423 |    94.41 |    8.523 |    15.02 |
|   512 |    128 |   5632 |    5.505 |    93.01 |    8.732 |    14.66 |
|   512 |    128 |   6144 |    5.490 |    93.27 |    8.706 |    14.70 |
|   512 |    128 |   6656 |    5.479 |    93.45 |    8.826 |    14.50 |
|   512 |    128 |   7168 |    5.595 |    91.51 |    8.783 |    14.57 |
|   512 |    128 |   7680 |    5.656 |    90.52 |    8.835 |    14.49 |
|   512 |    128 |   8192 |    5.800 |    88.28 |    8.985 |    14.25 |

</details>

![sweep-bench-pr557](https://github.com/user-attachments/assets/052420c8-caf9-412a-aa36-b636183334e7)