### 🔀 [#410](https://github.com/ikawrakow/ik_llama.cpp/pull/410) - Better CPU FA performance for DeepSeek-Lite

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-12 |
| **Updated** | 2025-05-20 |

---

#### Description

This FA tweak improves DeepSeek-Lite CPU TG performance with `Q8_0` KV cache.

Not sure if it will have a positive impact for the large DeepSeek models. To optimize the FA strategy for those I need to be able to test, which I cannot atm.

The graph shows a comparison between the main branch and this PR for a `Q4_0` quantized DeepSeek-Lite model. The CPU is Ryzen-7950X. The x-axis is `N_KV/1000`, where `N__KV` is the number of tokens in the K cache, which is quantized with `Q8_0`. The `sweep-bench` command was
```
./bin/llama-sweep-bench -m $model -c 16384 -ub 1024 -t 16 -mla 3 -fmoe -fa -rtr
```
  
![z12](https://github.com/user-attachments/assets/fcdcf29d-2802-4562-84ab-f535d09a4c73)

<details>
<summary>Main branch</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    1.488 |   688.02 |    7.112 |    35.99 |
|  1024 |    256 |   1024 |    1.674 |   611.73 |    7.361 |    34.78 |
|  1024 |    256 |   2048 |    1.788 |   572.75 |    7.524 |    34.02 |
|  1024 |    256 |   3072 |    1.951 |   524.97 |    7.728 |    33.13 |
|  1024 |    256 |   4096 |    2.104 |   486.65 |    7.927 |    32.29 |
|  1024 |    256 |   5120 |    2.276 |   449.93 |    8.152 |    31.40 |
|  1024 |    256 |   6144 |    2.483 |   412.40 |    8.441 |    30.33 |
|  1024 |    256 |   7168 |    2.841 |   360.45 |    8.795 |    29.11 |
|  1024 |    256 |   8192 |    2.794 |   366.55 |    9.294 |    27.54 |
|  1024 |    256 |   9216 |    2.974 |   344.36 |    9.142 |    28.00 |
|  1024 |    256 |  10240 |    3.130 |   327.15 |    9.404 |    27.22 |
|  1024 |    256 |  11264 |    3.328 |   307.69 |    9.654 |    26.52 |
|  1024 |    256 |  12288 |    3.499 |   292.67 |   10.078 |    25.40 |
|  1024 |    256 |  13312 |    3.840 |   266.70 |   10.536 |    24.30 |
|  1024 |    256 |  14336 |    3.886 |   263.53 |   10.969 |    23.34 |
|  1024 |    256 |  15360 |    4.055 |   252.52 |   11.430 |    22.40 |

</details>

<details>
<summary>PR</summary>

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    1.469 |   696.86 |    7.126 |    35.93 |
|  1024 |    256 |   1024 |    1.601 |   639.65 |    7.322 |    34.96 |
|  1024 |    256 |   2048 |    1.759 |   582.03 |    7.446 |    34.38 |
|  1024 |    256 |   3072 |    1.920 |   533.47 |    7.673 |    33.36 |
|  1024 |    256 |   4096 |    2.081 |   491.98 |    7.728 |    33.13 |
|  1024 |    256 |   5120 |    2.282 |   448.64 |    7.852 |    32.60 |
|  1024 |    256 |   6144 |    2.413 |   424.33 |    7.991 |    32.04 |
|  1024 |    256 |   7168 |    2.626 |   389.95 |    8.122 |    31.52 |
|  1024 |    256 |   8192 |    2.753 |   372.02 |    8.238 |    31.08 |
|  1024 |    256 |   9216 |    2.934 |   348.97 |    8.394 |    30.50 |
|  1024 |    256 |  10240 |    3.159 |   324.17 |    8.538 |    29.98 |
|  1024 |    256 |  11264 |    3.299 |   310.44 |    8.668 |    29.53 |
|  1024 |    256 |  12288 |    3.501 |   292.47 |    8.818 |    29.03 |
|  1024 |    256 |  13312 |    3.684 |   277.98 |    8.969 |    28.54 |
|  1024 |    256 |  14336 |    4.074 |   251.37 |    9.089 |    28.16 |
|  1024 |    256 |  15360 |    4.086 |   250.63 |    9.167 |    27.93 |

</details>

---

#### 💬 Conversation

👤 **saood06** commented the **2025-05-20** at **08:19:37**:<br>

I did end up doing a fresh build, drop cache and server launch and have used it up to 32K tokens (double where I normally test sweep-bench), and my informal results are that it is about the same, maybe a little better. I don't see the same large improvement that seems to scale with context size that you do.

I may run a full sweep-bench later to get a better comparison, I only ran it at very low amounts just to validate the model was warmed up and running at normal speeds ( I usually do this before launching server) and it performed about the same.

---

👤 **saood06** commented the **2025-05-20** at **09:19:47**:<br>

> > I don't see the same large improvement that seems to scale with context size that you do.
> 
>So, I guess, it is somehow related to NUMA, so it is bottle-necked on that when computing self-attention. If so, yes, you probably will not see (significant) performance improvement.

I'm not sure because it has good local hitrate on TG see this: https://github.com/ikawrakow/ik_llama.cpp/discussions/201#discussioncomment-13203928

---

👤 **ikawrakow** commented the **2025-05-20** at **09:44:56**:<br>

> I'm not sure because it has good local hitrate on TG see this: https://github.com/ikawrakow/ik_llama.cpp/discussions/201#discussioncomment-13203928

The high local TG hit rate is measured at what context?