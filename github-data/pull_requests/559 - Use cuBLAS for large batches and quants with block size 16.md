### 🔀 [#559](https://github.com/ikawrakow/ik_llama.cpp/pull/559) - Use cuBLAS for large batches and quants with block size 16

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-26 |
| **Updated** | 2025-07-02 |

---

#### Description

While working on #557 I noticed that dequantize+cuBLAS is faster than MMQ for the `iqX_k_r4` quants when the batch size is larger than some threshold.

The same applies to all quantization types with block size of 16: `Q2_K, Q3_K, Q6_K, IQ2_XS, IQ2_S, IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K`. Hence, this PR changes the `ggml_cuda_should_use_mmq()` function to return `false` if the batch size (number of rows in the right matrix) is greater than some quantization type specific threshold.

This graph illustrates the PP performance improvement achieved this way for k-quants. Model is LlaMA-3.1-8B-Instruct, GPU is RTX-4080, and in all cases pure quantization is used. `Q2_K` appears to have a particularly bad MMQ implementation (I need to look into that more closely), so there we benefit from switching to dequantize+cuBLAS already at 384 tokens, and achieve a solid 30-35% improvement for batch sizes above 1000 tokens. The MMQ implementation for the other quants (also those not shown) is better, so performance gains are in the range of 10% at a batch size of 4k tokens. For quants with a block size of 32 (all others not listed above) MMQ is always better than dequantize+cuBLAS up to a batch size of 4096 tokens, so they are left unchanged by the PR.   

![k_quants](https://github.com/user-attachments/assets/477588a9-9566-4a2c-9473-bd6d3bd783bf)

---

#### 💬 Conversation

👤 **ewhacc** commented the **2025-06-26** at **20:12:34**:<br>

I tried this "build = 3773 (3dbc8437)"  on ubergam's  DeepSeek-R1-0528-GGUF  IQ2_K_R4  with  -b 4096 -ub 4096.
Getting no difference on PP speed, compared to "build = 3762 (1843ed22)".

Both are about the same:
prompt eval time =   25328.73 ms /  6889 tokens (    3.68 ms per token,   271.98 tokens per second)

Did I something wrong?

My rig is Epyc Genoa + 6000 ada.

Built with
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1

---

👤 **ubergarm** commented the **2025-06-26** at **20:19:59**:<br>

@ewhacc 
@ewhacc 

Yeah, the speed boosts specific to IQ2_K_R4 and IQ3_K_R4 quantizations (in the quan you mention) were *already* added in PR557. This PR is doing a similar thing for some *other* quant types like Q2_K etc.

I just did another test for PR557 using this git sha, which is a bit confusing as I'm not actually testing all the new quants added here. But you can see the speed up is pretty good relative to just *before* PR557 was merged as shown below:

![sweep-bench-PR557-revisit](https://github.com/user-attachments/assets/bda70fa0-94a1-4e08-85b6-2850f0fd1815)

<details>

<summary>👈</summary>

```bash
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1 -DGGML_CUDA_F16=ON
cmake --build ./build --config Release -j $(nproc)

model=DeepSeek-R1-0528-IQ2_K_R4-00001-of-00005.gguf
./build/bin/llama-sweep-bench \
    --model "$model" \
    --no-mmap \
    --ctx-size 12288 \
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

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q5_0:   61 tensors
llama_model_loader: - type iq4_ks:  116 tensors
llama_model_loader: - type iq5_ks:  435 tensors
llama_model_loader: - type iq2_k_r4:  116 tensors
llama_model_loader: - type iq3_k_r4:   58 tensors
```

## PR559@3dbc8437 -ub 512 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.153 |   123.28 |    8.016 |    15.97 |
|   512 |    128 |    512 |    3.844 |   133.18 |    8.126 |    15.75 |
|   512 |    128 |   1024 |    3.932 |   130.22 |    8.246 |    15.52 |
|   512 |    128 |   1536 |    4.104 |   124.74 |    8.179 |    15.65 |
|   512 |    128 |   2048 |    4.185 |   122.35 |    8.188 |    15.63 |
|   512 |    128 |   2560 |    4.265 |   120.04 |    8.452 |    15.14 |
|   512 |    128 |   3072 |    4.576 |   111.89 |    8.376 |    15.28 |
|   512 |    128 |   3584 |    5.258 |    97.37 |    8.491 |    15.07 |
|   512 |    128 |   4096 |    4.538 |   112.83 |    8.456 |    15.14 |
|   512 |    128 |   4608 |    4.625 |   110.69 |    8.483 |    15.09 |
|   512 |    128 |   5120 |    4.717 |   108.55 |    8.609 |    14.87 |
|   512 |    128 |   5632 |    4.796 |   106.76 |    8.704 |    14.71 |
|   512 |    128 |   6144 |    4.950 |   103.42 |    8.862 |    14.44 |
|   512 |    128 |   6656 |    4.939 |   103.66 |    8.781 |    14.58 |
|   512 |    128 |   7168 |    5.195 |    98.55 |    8.722 |    14.68 |
|   512 |    128 |   7680 |    5.062 |   101.14 |    8.778 |    14.58 |
|   512 |    128 |   8192 |    5.199 |    98.49 |    8.962 |    14.28 |

## PR559@3dbc8437 -ub 2048 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    9.450 |   216.73 |   32.442 |    15.78 |
|  2048 |    512 |   2048 |    9.884 |   207.20 |   32.834 |    15.59 |
|  2048 |    512 |   4096 |   10.350 |   197.87 |   33.770 |    15.16 |
|  2048 |    512 |   6144 |   10.742 |   190.65 |   34.733 |    14.74 |
|  2048 |    512 |   8192 |   11.167 |   183.39 |   36.017 |    14.22 |


## PR559@3dbc8437 -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   12.824 |   319.40 |   65.575 |    15.62 |
|  4096 |   1024 |   4096 |   14.822 |   276.35 |   68.417 |    14.97 |
|  4096 |   1024 |   8192 |   17.282 |   237.01 |   72.403 |    14.14 |

## main@8e5106b2 -ub 512 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.339 |   117.99 |    8.091 |    15.82 |
|   512 |    128 |    512 |    4.411 |   116.08 |    8.349 |    15.33 |
|   512 |    128 |   1024 |    4.516 |   113.38 |    8.158 |    15.69 |
|   512 |    128 |   1536 |    4.873 |   105.07 |    8.190 |    15.63 |
|   512 |    128 |   2048 |    4.667 |   109.71 |    8.288 |    15.44 |
|   512 |    128 |   2560 |    4.763 |   107.49 |    8.379 |    15.28 |
|   512 |    128 |   3072 |    4.854 |   105.48 |    8.572 |    14.93 |
|   512 |    128 |   3584 |    4.932 |   103.82 |    8.421 |    15.20 |
|   512 |    128 |   4096 |    5.477 |    93.48 |    8.420 |    15.20 |
|   512 |    128 |   4608 |    5.125 |    99.90 |    8.553 |    14.97 |
|   512 |    128 |   5120 |    5.283 |    96.92 |    8.611 |    14.87 |
|   512 |    128 |   5632 |    5.393 |    94.94 |    8.668 |    14.77 |
|   512 |    128 |   6144 |    5.853 |    87.48 |    8.709 |    14.70 |
|   512 |    128 |   6656 |    5.466 |    93.66 |    8.837 |    14.48 |
|   512 |    128 |   7168 |    5.547 |    92.29 |    8.730 |    14.66 |
|   512 |    128 |   7680 |    5.648 |    90.64 |    8.885 |    14.41 |
|   512 |    128 |   8192 |    5.796 |    88.34 |    8.954 |    14.29 |

## main@8e5106b2 -ub 2048 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   11.483 |   178.34 |   32.442 |    15.78 |
|  2048 |    512 |   2048 |   11.937 |   171.56 |   33.131 |    15.45 |
|  2048 |    512 |   4096 |   12.262 |   167.02 |   33.925 |    15.09 |
|  2048 |    512 |   6144 |   12.714 |   161.08 |   34.877 |    14.68 |
|  2048 |    512 |   8192 |   13.044 |   157.01 |   36.298 |    14.11 |


## main@8e5106b2 -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   14.738 |   277.93 |   65.731 |    15.58 |
|  4096 |   1024 |   4096 |   16.671 |   245.70 |   68.219 |    15.01 |
|  4096 |   1024 |   8192 |   19.206 |   213.26 |   72.408 |    14.14 |


</details>

---

👤 **ikawrakow** commented the **2025-06-27** at **06:40:41**:<br>

> Noob question and sorry to ask here, but does this PR apply to sub k quants? Like q2_k_s, q3_k_m, q4_k_l, q5_k_xl, etc

I know this is confusing. Users specify the quantization with a llama type (`Q2_K_S, Q2_K_M, Q3_K_S, Q3_K_M`, etc). This gets translated into actual quantization types. For instance, the llama type `Q4_K_M` results in a quantization where most tensors are quantized with `Q4_K`, a few with `Q5_K`, and a few with `Q6_K`, where now `Q4_K`, etc., refers to the internal quantization type. Not being a marketing genius as Unsloth, I called the llama types "quantization mixes" instead of "dynamic quants". In the early days, where there were just a few viable open weight models, this approach made sense as it made it very easy for the user to create quantized models with varying sizes, where the choice of on which tensors to spend more bits was done internally (and the choice was based on a careful evaluation of which tensors have the largest impact on quantization quality). But today we have so many models that behave in subtly different ways, that it is easier and better to just explicitly specify quantization types via `--custom-q`. This is all Unsloth do btw. They just have their own quantization mixes of existing quantization types, which they call "dynamic quants", a concept that has existed in `llama.cpp` since the initial [k-quants PR](https://github.com/ggml-org/llama.cpp/pull/1684).

---

👤 **ikawrakow** commented the **2025-06-27** at **07:02:46**:<br>

Performance impact is easier to test with a dense model. For a MoE model such as DeepSeek-R1/V3, even at a batch size of 4096 tokens, experts process on average just 128 tokens, so still far away from the point where the transition to dequantize+cuBLAS occurs. Most of the self attention computations are within the FA implementation, which does not use the regular matrix multiplications, so there are just a few matrix multiplications left that get affected, but they usually take a small fraction of the overall calculation, so impact is negligible (and, as pointed out by @ubergarm, the test done by @ewhacc is not affected by this PR).

But if you are running a dense model with partial offload, you will want to have larger batches/u-batches to minimize the time spent on copying tensors from RAM to VRAM relative to the time spent on actual calculations. In that case you ought to see a measurable impact on PP performance, provided the model contains quantization types affected by this PR.

---

👤 **ikawrakow** commented the **2025-06-27** at **07:26:28**:<br>

Here is an example illustrating my previous post. Running LlaMA-3.1-70B quantized with `Q2_K_S` on my paltry RTX-4080 with 16 GB VRAM:

| model            | n_ubatch |     test |    t/s (main)    |    t/s (PR)      |  Speedup |
| ---------------- | -------: | -------: | ---------------: | ---------------: | -------: |
| llama 70B Q2_K   |      512 |   pp4096 |    302.11 ± 0.38 |    328.51 ± 1.02 |  1.087   |
| llama 70B Q2_K   |     1024 |   pp4096 |    397.43 ± 0.36 |    488.37 ± 0.27 |  1.229   |
| llama 70B Q2_K   |     2048 |   pp4096 |    468.44 ± 0.02 |    626.39 ± 0.30 |  1.338   |
| llama 70B Q2_K   |     4096 |   pp4096 |    509.45 ± 0.19 |    722.58 ± 0.40 |  1.418   |

I have uploaded only 30 out of 80 layers to the GPU so I can run with the larger u-batch. If instead I use the default u-batch of 512, I can upload 50 layers to the GPU. With that I get `pp4096 = 372 t/s` on the main branch, so a pretty good speedup with this PR and `u-batch = 4096` with almost double the performance.

---

👤 **ubergarm** commented the **2025-06-27** at **14:03:24**:<br>

Okay, I made a few Qwen3-14B dense "pure" quants (q4_K token_embd, q6_K output "head") and seeing roughly 1.4x speedup on PP with this PR over main for `-ub 4096 -b 4096` batch sizes.

This is great and really changes things given `iq4_k` and `iq5_k` are now *faster* than the `ks` counterparts as shown in the graph:

![sweep-bench-pr541-qwen3-14b](https://github.com/user-attachments/assets/87382c5f-840e-4798-926c-05bb638c17f8)

<details>

<summary>👈 sweep-bench command and results</summary>

```bash
CUDA_VISIBLE_DEVICES="0" \
    ./build/bin/llama-sweep-bench \
        --model "$model" \
        --ctx-size 20480 \
        -ctk f16 -ctv f16 \
        -fa \
        -ngl 99 \
        -ub 4096 -b 4096 \
        --no-mmap \
        --warmup-batch \
        --threads 1
```

## IQ4_K PR559@3dbc8437 -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    1.397 |  2931.10 |   16.804 |    60.94 |
|  4096 |   1024 |   4096 |    1.664 |  2461.65 |   19.088 |    53.65 |
|  4096 |   1024 |   8192 |    1.931 |  2121.11 |   21.343 |    47.98 |
|  4096 |   1024 |  12288 |    2.195 |  1865.99 |   23.547 |    43.49 |
|  4096 |   1024 |  16384 |    2.462 |  1663.59 |   25.710 |    39.83 |

## IQ4_KS PR559@3dbc8437 -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    1.687 |  2427.29 |   15.177 |    67.47 |
|  4096 |   1024 |   4096 |    1.957 |  2092.94 |   17.336 |    59.07 |
|  4096 |   1024 |   8192 |    2.224 |  1841.42 |   19.477 |    52.57 |
|  4096 |   1024 |  12288 |    2.485 |  1648.45 |   21.591 |    47.43 |
|  4096 |   1024 |  16384 |    2.747 |  1491.03 |   23.672 |    43.26 |

## IQ5_K PR559@3dbc8437 -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    1.425 |  2873.91 |   18.492 |    55.37 |
|  4096 |   1024 |   4096 |    1.691 |  2422.55 |   20.701 |    49.47 |
|  4096 |   1024 |   8192 |    1.949 |  2101.61 |   22.837 |    44.84 |
|  4096 |   1024 |  12288 |    2.207 |  1856.22 |   24.911 |    41.11 |
|  4096 |   1024 |  16384 |    2.476 |  1654.56 |   26.981 |    37.95 |

## IQ5_KS PR559@3dbc8437 -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    1.773 |  2309.95 |   18.037 |    56.77 |
|  4096 |   1024 |   4096 |    2.041 |  2007.28 |   20.177 |    50.75 |
|  4096 |   1024 |   8192 |    2.302 |  1779.68 |   22.225 |    46.07 |
|  4096 |   1024 |  12288 |    2.573 |  1591.83 |   24.321 |    42.10 |
|  4096 |   1024 |  16384 |    2.832 |  1446.44 |   26.453 |    38.71 |

## IQ4_K main@5236c98b -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    1.959 |  2090.59 |   17.302 |    59.19 |
|  4096 |   1024 |   4096 |    2.225 |  1840.67 |   19.540 |    52.41 |
|  4096 |   1024 |   8192 |    2.490 |  1645.04 |   21.677 |    47.24 |
|  4096 |   1024 |  12288 |    2.749 |  1490.13 |   23.767 |    43.09 |
|  4096 |   1024 |  16384 |    3.011 |  1360.19 |   25.834 |    39.64 |

## IQ5_K main@5236c98b -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    1.959 |  2091.01 |   18.450 |    55.50 |
|  4096 |   1024 |   4096 |    2.237 |  1830.95 |   20.664 |    49.56 |
|  4096 |   1024 |   8192 |    2.512 |  1630.39 |   22.848 |    44.82 |
|  4096 |   1024 |  12288 |    2.779 |  1473.69 |   24.981 |    40.99 |
|  4096 |   1024 |  16384 |    3.042 |  1346.62 |   27.103 |    37.78 |

</details>

---

👤 **ikawrakow** commented the **2025-06-27** at **14:17:17**:<br>

Before you throw these quants away, try `-b 2048 -ub 512` and `b 2048 -ub 1024`.

---

👤 **ubergarm** commented the **2025-06-27** at **14:22:59**:<br>

Sure thing.

Also it is interesting now that q6_K is a little faster PP than q4_K at 4096 ub/b

---

👤 **ubergarm** commented the **2025-06-27** at **14:39:13**:<br>

![sweep-bench-pr559-IQ4_K](https://github.com/user-attachments/assets/d2ad35d4-1a0a-49e8-ac09-87d93bdb5f6f)

---

👤 **ikawrakow** commented the **2025-06-27** at **15:34:33**:<br>

So, the A6000 has more memory bandwidth than the 4080. This shifts things in favor of dequantize+cuBLAS because the dequantize step is memory bound, so it is quicker on the A6000. I guess this is why with `-ub 4096` `IQ4_K` outperforms `IQ4_KS`. I guess, I should look into making the thresholds at which the transitions between MMQ and dequantize+cuBLAS happens configurable. But I'll leave this for another PR.

---

👤 **ikawrakow** commented the **2025-06-27** at **15:43:44**:<br>

Based on @ubergarm's and my own testing this PR looks like a winner, so merging.

---

👤 **ikawrakow** commented the **2025-06-29** at **16:28:37**:<br>

These performance results look pretty good to me. Has anyone ever reported a better result for hybrid GPU/CPU DeepSeek-R1/V3 inference?

---

👤 **Panchovix** commented the **2025-06-30** at **20:58:35**:<br>

Haven't managed to test much as I accidentaly wiped my Fedora installation from Windows lol. But I was testing with llama sweep bench and got one error, but can't remember exactly the error, and/or is related to this PR.

I have just saved at how I run the model, which is

```
./llama-server -m '/models_llm/DeepSeek-V3-0324-UD-Q3_K_XL-merged.gguf' -c 32768 --no-mmap -ngl 999 \
-ot "blk.(0|1|2|3|4|5|6|7).ffn.=CUDA0" \
-ot "blk.(8|9|10|11).ffn.=CUDA1" \
-ot "blk.(12|13|14|15).ffn.=CUDA2" \
-ot "blk.(16|17|18|19|20).ffn.=CUDA3" \
-ot "blk.(21|22|23).ffn.=CUDA4" \
-ot "blk.(24|25|26).ffn.=CUDA5" \
-ot "blk.(27|28|29|30|31|32|33|34).ffn.=CUDA6" \
-ot "blk.35.ffn_(norm|gate_inp|gate_shexp|down_shexp|up_shexp).weight=CUDA4" \
-ot "blk.35.ffn_gate_exps.weight=CUDA4" \
-ot "blk.36.ffn_(norm|gate_inp|gate_shexp|down_shexp|up_shexp).weight=CUDA5" \
-ot "blk.36.ffn_gate_exps.weight=CUDA5" \
-ot "ffn.*=CPU" \
-fa -mg 0 -ub 2048 -mla 1
```

I managed to see 200 t/s PP and 8.73 t/s TG, but then got a error. Again I will try to update when I get Linux installed again, as offloading + multigpu is just not worth it on Windows.

---

👤 **Panchovix** commented the **2025-07-01** at **15:43:44**:<br>

Okay finally installed Fedora yesterday, testing remotely now so it is a bit slower (I'm using software encoding and it uses 2-3 threads)

```
ggml_cuda_init: found 7 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 2: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 3: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 4: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 5: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 6: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
...
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |   11.682 |   175.31 |   72.116 |     7.10 |
|  2048 |    512 |   2048 |   12.111 |   169.10 |   72.112 |     7.10 |
|  2048 |    512 |   4096 |   12.881 |   158.99 |   72.678 |     7.04 |
|  2048 |    512 |   6144 |   13.611 |   150.47 |   73.289 |     6.99 |
CUDA error: an illegal memory access was encountered
  current device: 1, in function prepare_row_mappigs at /run/media/pancho/60A2FCEDA2FCC894/ChatIAs/ik_llama.cpp/ggml/src/ggml-cuda.cu:2222
  cudaMemcpyAsync(ids_host.data(), ids_dev, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream)
/run/media/pancho/60A2FCEDA2FCC894/ChatIAs/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
```

WIth the same command as above. Sometimes it also crashes with another cuda error but still have to get it again. Again, not sure what is related to.