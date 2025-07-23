### 🔀 [#574](https://github.com/ikawrakow/ik_llama.cpp/pull/574) - Change KQ mask padding to 64

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-03 |
| **Updated** | 2025-07-03 |

---

#### Description

This is needed by the Vulkan back-end when coopmat2 is enabled.

It is 64 in mainline too.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-07-03** at **08:42:47**:<br>

So, I updated the Nvidia driver on one of my two remote machines to 575, which enables Vulkan coopmat2. This triggers an assert in the Vulkan back-end, which is the reason for this PR fixing it. But I was more interested in the performance implications as I saw a factor of 3 lower Vulkan performance with coopmat1 compared to CUDA. As per [this comment](https://github.com/ikawrakow/ik_llama.cpp/discussions/562#discussioncomment-13630937), the difference between the CUDA and Vulkan back-ends on the same Nvidia GPU should be in the range of 20-25% when coopmat2 is enabled. Sadly, this is not the case on my RTX-4080. Coopmat2 is better, but PP is still a factor of 2 lower compared to CUDA. Here is a sweep bench for `Q4_0`-quantized LlaMA-3.1-8B-Instruct for u-batch of 1024 and FA enabled:

### Vulkan

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    0.248 |  4128.32 |    2.700 |    94.80 |
|  1024 |    256 |   1024 |    0.263 |  3887.37 |    2.684 |    95.37 |
|  1024 |    256 |   2048 |    0.272 |  3769.07 |    2.752 |    93.03 |
|  1024 |    256 |   3072 |    0.281 |  3639.22 |    2.807 |    91.21 |
|  1024 |    256 |   4096 |    0.288 |  3560.62 |    2.865 |    89.37 |
|  1024 |    256 |   5120 |    0.303 |  3380.02 |    2.932 |    87.30 |
|  1024 |    256 |   6144 |    0.324 |  3158.54 |    2.993 |    85.53 |
|  1024 |    256 |   7168 |    0.333 |  3074.87 |    3.026 |    84.59 |
|  1024 |    256 |   8192 |    0.344 |  2977.47 |    3.100 |    82.59 |
|  1024 |    256 |   9216 |    0.351 |  2920.00 |    3.156 |    81.11 |
|  1024 |    256 |  10240 |    0.356 |  2876.61 |    3.221 |    79.47 |
|  1024 |    256 |  11264 |    0.376 |  2725.05 |    3.270 |    78.30 |
|  1024 |    256 |  12288 |    0.386 |  2651.13 |    3.319 |    77.13 |
|  1024 |    256 |  13312 |    0.399 |  2564.51 |    3.388 |    75.56 |
|  1024 |    256 |  14336 |    0.415 |  2470.40 |    3.443 |    74.36 |
|  1024 |    256 |  15360 |    0.427 |  2400.04 |    3.499 |    73.17 |

### CUDA

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    0.122 |  8379.71 |    2.054 |   124.65 |
|  1024 |    256 |   1024 |    0.125 |  8170.82 |    2.092 |   122.39 |
|  1024 |    256 |   2048 |    0.134 |  7615.59 |    2.154 |   118.84 |
|  1024 |    256 |   3072 |    0.141 |  7277.02 |    2.221 |   115.26 |
|  1024 |    256 |   4096 |    0.149 |  6857.34 |    2.290 |   111.77 |
|  1024 |    256 |   5120 |    0.156 |  6555.32 |    2.371 |   107.97 |
|  1024 |    256 |   6144 |    0.163 |  6273.82 |    2.412 |   106.14 |
|  1024 |    256 |   7168 |    0.171 |  6000.02 |    2.467 |   103.77 |
|  1024 |    256 |   8192 |    0.182 |  5627.80 |    2.527 |   101.32 |
|  1024 |    256 |   9216 |    0.188 |  5440.44 |    2.580 |    99.23 |
|  1024 |    256 |  10240 |    0.190 |  5400.07 |    2.665 |    96.04 |
|  1024 |    256 |  11264 |    0.200 |  5130.03 |    2.700 |    94.83 |
|  1024 |    256 |  12288 |    0.206 |  4970.97 |    2.751 |    93.06 |
|  1024 |    256 |  13312 |    0.215 |  4769.69 |    2.810 |    91.10 |
|  1024 |    256 |  14336 |    0.226 |  4538.54 |    2.865 |    89.34 |
|  1024 |    256 |  15360 |    0.230 |  4459.33 |    2.936 |    87.18 |