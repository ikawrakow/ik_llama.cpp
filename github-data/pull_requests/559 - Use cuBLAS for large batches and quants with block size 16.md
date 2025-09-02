## 🔀 [Pull Request #559](https://github.com/ikawrakow/ik_llama.cpp/pull/559) - Use cuBLAS for large batches and quants with block size 16

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/mmq_to_cublas` |
| **Target Branch** | `main` |
| **Created** | 2025-06-26 |
| **Updated** | 2025-07-02 |
| **Merged** | 2025-06-27 |

---

## 📄 Description

While working on [#557](https://github.com/ikawrakow/ik_llama.cpp/issues/557) I noticed that dequantize+cuBLAS is faster than MMQ for the `iqX_k_r4` quants when the batch size is larger than some threshold.

The same applies to all quantization types with block size of 16: `Q2_K, Q3_K, Q6_K, IQ2_XS, IQ2_S, IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K`. Hence, this PR changes the `ggml_cuda_should_use_mmq()` function to return `false` if the batch size (number of rows in the right matrix) is greater than some quantization type specific threshold.

This graph illustrates the PP performance improvement achieved this way for k-quants. Model is LlaMA-3.1-8B-Instruct, GPU is RTX-4080, and in all cases pure quantization is used. `Q2_K` appears to have a particularly bad MMQ implementation (I need to look into that more closely), so there we benefit from switching to dequantize+cuBLAS already at 384 tokens, and achieve a solid 30-35% improvement for batch sizes above 1000 tokens. The MMQ implementation for the other quants (also those not shown) is better, so performance gains are in the range of 10% at a batch size of 4k tokens. For quants with a block size of 32 (all others not listed above) MMQ is always better than dequantize+cuBLAS up to a batch size of 4096 tokens, so they are left unchanged by the PR.   

![k_quants](https://github.com/user-attachments/assets/477588a9-9566-4a2c-9473-bd6d3bd783bf)

---

## 💬 Conversation

👤 **ewhacc** commented on **2025-06-26** at **20:12:34**

I tried this "build = 3773 (3dbc8437)"  on ubergam's  DeepSeek-R1-0528-GGUF  IQ2_K_R4  with  -b 4096 -ub 4096.
Getting no difference on PP speed, compared to "build = 3762 (1843ed22)".

Both are about the same:
prompt eval time =   25328.73 ms /  6889 tokens (    3.68 ms per token,   271.98 tokens per second)

Did I something wrong?

My rig is Epyc Genoa + 6000 ada.

Built with
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1

---

👤 **ubergarm** commented on **2025-06-26** at **20:19:59**

@ewhacc 

*EDIT* wait, your old test was on 1843ed2 which was *before* PR557 was merged?? huh, i would imagine you would see some speed boost. compare against the commands i'm using below to see if something else is up?

Yeah, the speed boosts specific to IQ2_K_R4 and IQ3_K_R4 quantizations (in the quan you mention) were *already* added in PR557. This PR is doing a similar thing for some *other* quant types like Q2_K etc.

I just did another test for PR557 using this git sha, which is a bit confusing as I'm not actually testing all the new quants added here. But you can see the speed up is pretty good relative to just *before* PR557 was merged as shown below:

![sweep-bench-PR557-revisit](https://github.com/user-attachments/assets/bda70fa0-94a1-4e08-85b6-2850f0fd1815)

<details>

<summary>👈compile, llama-sweep-bench, data</summary>

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

👤 **Panchovix** commented on **2025-06-26** at **22:14:16**

Noob question and sorry to ask here, but does this PR apply to sub k quants? Like q2_k_s, q3_k_m, q4_k_l, q5_k_xl, etc

---

👤 **ubergarm** commented on **2025-06-27** at **00:55:18**

@ewhacc 

I thought about it some more, and both this PR559 and PR557 only apply when the mentioned quantized tensors are running on CUDA. So for my quant that you mention, the `IQ2_K_R4` only the `ffn_(gate|down|up)_exps` tensors are quantized at one of those involved in these PRs. 

So to see the speed boost you have to use offload more of those specific layers onto CUDA e.g. `-ot "blk\.(3|4|5|6|7|8|9|10|11|12)\.ffn_.*=CUDA0"`. If you're not offloading more of those layers, then you would see the same speeds.

This kinda ties into @Panchovix great question, and I'd love to do a video called "What is in a quant?" to explain better, because it is pretty confusing until you dig into it with either `./gguf-py/scripts/gguf_dump.py` or more simply looking at the hugging face side-bar e.g. here for a specific example: [bartowski's DeepSeek-R1-0528-Q3_K_M](https://huggingface.co/bartowski/deepseek-ai_DeepSeek-R1-0528-GGUF?show_file_info=deepseek-ai_DeepSeek-R1-0528-Q3_K_M%2Fdeepseek-ai_DeepSeek-R1-0528-Q3_K_M-00001-of-00008.gguf)

You see it has the filename `Q3_K_M`, but when you scroll down and look at the tensors, not *every* tensor is quantized at Q3_K_M.  Also for unsloth you'll *never* see a tensor quantized with `UD-Q4_K_XL` as that is not even a real thing.

> Like q2_k_s, q3_k_m, q4_k_l, q5_k_xl, etc

So things like `Q2_K_S` are *both*  of an actual quantization type and also a pre-defined recipe according to [llama-quantize](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) code. Things like `XL` or even my `-mix-` prefix are kind of naming conventions for recipes that suggest which tensors might be a little bigger or smaller but "mostly" around that size. I like to joke about the mythical 8.5 BPW `IQ1_S_XXXL` for example which is just an absurd extension of this basic naming conventions.

Personally I don't follow the conventions established in llama-quantize and pretty much always override everything with whatever I want to use. So when you start my `IQ2_K_R4` llama-server will print out:

```
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q5_0:   61 tensors - attn_k_b
llama_model_loader: - type iq4_ks:  116 tensors - ffn_(gate|up)_shexp
llama_model_loader: - type iq5_ks:  435 tensors token_embd,output,ffn_down_shexp,first 3 ffn_(down|gate|up),remaining attn_
llama_model_loader: - type iq2_k_r4:  116 tensors ffn_(gate|up)_exps
llama_model_loader: - type iq3_k_r4:   58 tensors ffn_down_exps
```

So there is a lot more going on under the hood than the name belies. My personal convention is to name the quant "recipe" filename after whatever the main `ffn_(gate|up)` tensors are quantized.

To keep it relevant to this PR, you need to look inside your gguf and see if any of the mentioned quantizations types apply to tensors which you are running on CUDA.

Cheers!

---

👤 **ikawrakow** commented on **2025-06-27** at **06:40:41**

> Noob question and sorry to ask here, but does this PR apply to sub k quants? Like q2_k_s, q3_k_m, q4_k_l, q5_k_xl, etc

I know this is confusing. Users specify the quantization with a llama type (`Q2_K_S, Q2_K_M, Q3_K_S, Q3_K_M`, etc). This gets translated into actual quantization types. For instance, the llama type `Q4_K_M` results in a quantization where most tensors are quantized with `Q4_K`, a few with `Q5_K`, and a few with `Q6_K`, where now `Q4_K`, etc., refers to the internal quantization type. Not being a marketing genius as Unsloth, I called the llama types "quantization mixes" instead of "dynamic quants". In the early days, where there were just a few viable open weight models, this approach made sense as it made it very easy for the user to create quantized models with varying sizes, where the choice of on which tensors to spend more bits was done internally (and the choice was based on a careful evaluation of which tensors have the largest impact on quantization quality). But today we have so many models that behave in subtly different ways, that it is easier and better to just explicitly specify quantization types via `--custom-q`. This is all Unsloth do btw. They just have their own quantization mixes of existing quantization types, which they call "dynamic quants", a concept that has existed in `llama.cpp` since the initial [k-quants PR](https://github.com/ggml-org/llama.cpp/pull/1684).

---

👤 **ikawrakow** commented on **2025-06-27** at **07:02:46**

Performance impact is easier to test with a dense model. For a MoE model such as DeepSeek-R1/V3, even at a batch size of 4096 tokens, experts process on average just 128 tokens, so still far away from the point where the transition to dequantize+cuBLAS occurs. Most of the self attention computations are within the FA implementation, which does not use the regular matrix multiplications, so there are just a few matrix multiplications left that get affected, but they usually take a small fraction of the overall calculation, so impact is negligible (and, as pointed out by @ubergarm, the test done by @ewhacc is not affected by this PR).

But if you are running a dense model with partial offload, you will want to have larger batches/u-batches to minimize the time spent on copying tensors from RAM to VRAM relative to the time spent on actual calculations. In that case you ought to see a measurable impact on PP performance, provided the model contains quantization types affected by this PR.

---

👤 **ikawrakow** commented on **2025-06-27** at **07:26:28**

Here is an example illustrating my previous post. Running LlaMA-3.1-70B quantized with `Q2_K_S` on my paltry RTX-4080 with 16 GB VRAM:

| model            | n_ubatch |     test |    t/s (main)    |    t/s (PR)      |  Speedup |
| ---------------- | -------: | -------: | ---------------: | ---------------: | -------: |
| llama 70B Q2_K   |      512 |   pp4096 |    302.11 ± 0.38 |    328.51 ± 1.02 |  1.087   |
| llama 70B Q2_K   |     1024 |   pp4096 |    397.43 ± 0.36 |    488.37 ± 0.27 |  1.229   |
| llama 70B Q2_K   |     2048 |   pp4096 |    468.44 ± 0.02 |    626.39 ± 0.30 |  1.338   |
| llama 70B Q2_K   |     4096 |   pp4096 |    509.45 ± 0.19 |    722.58 ± 0.40 |  1.418   |

I have uploaded only 30 out of 80 layers to the GPU so I can run with the larger u-batch. If instead I use the default u-batch of 512, I can upload 50 layers to the GPU. With that I get `pp4096 = 372 t/s` on the main branch, so a pretty good speedup with this PR and `u-batch = 4096` with almost double the performance.

---

👤 **ubergarm** commented on **2025-06-27** at **14:03:24**

Okay, I made a few Qwen3-14B dense "pure" quants (q4_K token_embd, q6_K output "head") and seeing roughly 1.4x speedup on PP with this PR over main for `-ub 4096 -b 4096` batch sizes.

This is great and really changes things given `iq4_k` and `iq5_k` are now *faster* than the `ks` counterparts as shown in the graph:

![sweep-bench-pr559](https://github.com/user-attachments/assets/b48f121a-1ab6-4348-a16f-4a5d7db8e5ca)

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

I didn't check the other remaining quantization types with block size of 16.

---

👤 **ikawrakow** commented on **2025-06-27** at **14:17:17**

Before you throw these quants away, try `-b 2048 -ub 512` and `b 2048 -ub 1024`.

---

👤 **ubergarm** commented on **2025-06-27** at **14:22:59**

Sure thing.

Also it is interesting now that q6_K is a little faster PP than q4_K at 4096 ub/b

---

👤 **ubergarm** commented on **2025-06-27** at **14:28:42**

![sweep-bench-Q6_K-ik-vs-mainline](https://github.com/user-attachments/assets/cfc20343-2118-4277-aaa5-6cdfc4d6c53e)

---

👤 **ubergarm** commented on **2025-06-27** at **14:39:13**

So for IQ4_K the sweet spot is closer to -ub 2048 -b 2048
*NOTE*: TITLE is wrong on this, leftover from before, this is *only* ik fork:
![sweep-bench-pr559-IQ4_K](https://github.com/user-attachments/assets/d2ad35d4-1a0a-49e8-ac09-87d93bdb5f6f)

<details>

<summary>data</summary>

## IQ4_K PR559@3dbc8437 -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    1.397 |  2931.10 |   16.804 |    60.94 |
|  4096 |   1024 |   4096 |    1.664 |  2461.65 |   19.088 |    53.65 |
|  4096 |   1024 |   8192 |    1.931 |  2121.11 |   21.343 |    47.98 |
|  4096 |   1024 |  12288 |    2.195 |  1865.99 |   23.547 |    43.49 |
|  4096 |   1024 |  16384 |    2.462 |  1663.59 |   25.710 |    39.83 |

## IQ4_K PR559@3dbc8437 -ub 2048 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    0.656 |  3121.27 |    8.250 |    62.06 |
|  2048 |    512 |   2048 |    0.717 |  2855.02 |    8.806 |    58.14 |
|  2048 |    512 |   4096 |    0.782 |  2617.76 |    9.391 |    54.52 |
|  2048 |    512 |   6144 |    0.853 |  2400.71 |    9.962 |    51.40 |
|  2048 |    512 |   8192 |    0.922 |  2221.92 |   10.529 |    48.63 |
|  2048 |    512 |  10240 |    0.994 |  2059.88 |   11.085 |    46.19 |
|  2048 |    512 |  12288 |    1.059 |  1934.63 |   11.654 |    43.93 |
|  2048 |    512 |  14336 |    1.122 |  1825.66 |   12.197 |    41.98 |
|  2048 |    512 |  16384 |    1.188 |  1723.89 |   12.727 |    40.23 |

## IQ4_K PR559@3dbc8437 -ub 1024 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    0.349 |  2933.38 |    4.174 |    61.33 |
|  1024 |    256 |   1024 |    0.363 |  2817.92 |    4.326 |    59.18 |
|  1024 |    256 |   2048 |    0.379 |  2701.71 |    4.466 |    57.32 |
|  1024 |    256 |   3072 |    0.395 |  2592.14 |    4.614 |    55.48 |
|  1024 |    256 |   4096 |    0.409 |  2503.88 |    4.753 |    53.86 |
|  1024 |    256 |   5120 |    0.423 |  2418.81 |    4.890 |    52.35 |
|  1024 |    256 |   6144 |    0.440 |  2325.79 |    5.044 |    50.76 |
|  1024 |    256 |   7168 |    0.455 |  2251.27 |    5.180 |    49.42 |
|  1024 |    256 |   8192 |    0.470 |  2179.03 |    5.318 |    48.14 |
|  1024 |    256 |   9216 |    0.486 |  2107.66 |    5.455 |    46.93 |
|  1024 |    256 |  10240 |    0.502 |  2041.52 |    5.588 |    45.81 |
|  1024 |    256 |  11264 |    0.519 |  1973.34 |    5.729 |    44.68 |
|  1024 |    256 |  12288 |    0.537 |  1908.08 |    5.866 |    43.64 |
|  1024 |    256 |  13312 |    0.551 |  1859.53 |    5.998 |    42.68 |
|  1024 |    256 |  14336 |    0.568 |  1804.21 |    6.129 |    41.77 |
|  1024 |    256 |  15360 |    0.584 |  1753.24 |    6.264 |    40.87 |
|  1024 |    256 |  16384 |    0.602 |  1701.41 |    6.397 |    40.02 |

## IQ4_K PR559@3dbc8437 -ub 512 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.214 |  2391.80 |    2.070 |    61.83 |
|   512 |    128 |    512 |    0.216 |  2365.51 |    2.096 |    61.08 |
|   512 |    128 |   1024 |    0.221 |  2318.91 |    2.133 |    60.02 |
|   512 |    128 |   1536 |    0.224 |  2280.78 |    2.166 |    59.09 |
|   512 |    128 |   2048 |    0.230 |  2229.61 |    2.211 |    57.88 |
|   512 |    128 |   2560 |    0.233 |  2199.67 |    2.246 |    57.00 |
|   512 |    128 |   3072 |    0.237 |  2159.36 |    2.284 |    56.04 |
|   512 |    128 |   3584 |    0.242 |  2112.85 |    2.315 |    55.30 |
|   512 |    128 |   4096 |    0.245 |  2087.07 |    2.357 |    54.30 |
|   512 |    128 |   4608 |    0.249 |  2053.19 |    2.390 |    53.55 |
|   512 |    128 |   5120 |    0.253 |  2022.21 |    2.427 |    52.74 |
|   512 |    128 |   5632 |    0.258 |  1983.98 |    2.460 |    52.02 |
|   512 |    128 |   6144 |    0.262 |  1951.78 |    2.498 |    51.23 |
|   512 |    128 |   6656 |    0.265 |  1930.62 |    2.536 |    50.48 |
|   512 |    128 |   7168 |    0.269 |  1903.37 |    2.571 |    49.79 |
|   512 |    128 |   7680 |    0.274 |  1868.29 |    2.607 |    49.10 |
|   512 |    128 |   8192 |    0.277 |  1845.98 |    2.639 |    48.50 |
|   512 |    128 |   8704 |    0.281 |  1821.41 |    2.678 |    47.79 |
|   512 |    128 |   9216 |    0.285 |  1799.54 |    2.715 |    47.15 |
|   512 |    128 |   9728 |    0.289 |  1773.15 |    2.747 |    46.60 |
|   512 |    128 |  10240 |    0.292 |  1750.98 |    2.784 |    45.97 |
|   512 |    128 |  10752 |    0.297 |  1726.16 |    2.820 |    45.38 |
|   512 |    128 |  11264 |    0.301 |  1699.55 |    2.858 |    44.79 |
|   512 |    128 |  11776 |    0.305 |  1678.82 |    2.890 |    44.29 |
|   512 |    128 |  12288 |    0.308 |  1662.74 |    2.924 |    43.78 |
|   512 |    128 |  12800 |    0.314 |  1629.31 |    2.959 |    43.26 |
|   512 |    128 |  13312 |    0.316 |  1620.90 |    2.992 |    42.78 |
|   512 |    128 |  13824 |    0.321 |  1594.61 |    3.026 |    42.30 |
|   512 |    128 |  14336 |    0.323 |  1582.96 |    3.058 |    41.86 |
|   512 |    128 |  14848 |    0.327 |  1563.48 |    3.092 |    41.39 |
|   512 |    128 |  15360 |    0.333 |  1537.87 |    3.125 |    40.96 |
|   512 |    128 |  15872 |    0.336 |  1523.45 |    3.160 |    40.51 |
|   512 |    128 |  16384 |    0.340 |  1505.35 |    3.194 |    40.08 |


</details>

Again for IQ5_K I'm seeing a peak closer to -ub 2048.

*NOTE* title is wrong on this next graph, forgot to update my script, this is *only* ik fork:
![sweep-bench-pr559-IQ5_K](https://github.com/user-attachments/assets/0e0654bb-9c3c-4850-9b44-4ab2918607e8)

<details>

<summary>IQ5_K data</summary>

## IQ5_K PR559@3dbc8437 -ub 4096 -b 4096
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    1.425 |  2873.91 |   18.492 |    55.37 |
|  4096 |   1024 |   4096 |    1.691 |  2422.55 |   20.701 |    49.47 |
|  4096 |   1024 |   8192 |    1.949 |  2101.61 |   22.837 |    44.84 |
|  4096 |   1024 |  12288 |    2.207 |  1856.22 |   24.911 |    41.11 |
|  4096 |   1024 |  16384 |    2.476 |  1654.56 |   26.981 |    37.95 |

## IQ5_K PR559@3dbc8437 -ub 2048 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    0.659 |  3108.06 |    9.025 |    56.73 |
|  2048 |    512 |   2048 |    0.717 |  2858.24 |    9.576 |    53.46 |
|  2048 |    512 |   4096 |    0.782 |  2617.54 |   10.101 |    50.69 |
|  2048 |    512 |   6144 |    0.851 |  2407.13 |   10.648 |    48.08 |
|  2048 |    512 |   8192 |    0.924 |  2216.49 |   11.196 |    45.73 |
|  2048 |    512 |  10240 |    0.994 |  2060.10 |   11.739 |    43.62 |
|  2048 |    512 |  12288 |    1.060 |  1932.34 |   12.290 |    41.66 |
|  2048 |    512 |  14336 |    1.128 |  1815.37 |   12.877 |    39.76 |
|  2048 |    512 |  16384 |    1.193 |  1716.14 |   13.405 |    38.19 |

## IQ5_K PR559@3dbc8437 -ub 1024 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    0.355 |  2881.92 |    4.574 |    55.97 |
|  1024 |    256 |   1024 |    0.370 |  2769.15 |    4.713 |    54.32 |
|  1024 |    256 |   2048 |    0.385 |  2657.51 |    4.851 |    52.77 |
|  1024 |    256 |   3072 |    0.399 |  2565.03 |    4.985 |    51.35 |
|  1024 |    256 |   4096 |    0.415 |  2469.14 |    5.121 |    49.99 |
|  1024 |    256 |   5120 |    0.429 |  2384.88 |    5.268 |    48.60 |
|  1024 |    256 |   6144 |    0.446 |  2298.44 |    5.410 |    47.32 |
|  1024 |    256 |   7168 |    0.460 |  2225.92 |    5.532 |    46.28 |
|  1024 |    256 |   8192 |    0.475 |  2155.30 |    5.658 |    45.25 |
|  1024 |    256 |   9216 |    0.491 |  2083.92 |    5.793 |    44.19 |
|  1024 |    256 |  10240 |    0.507 |  2021.02 |    5.929 |    43.18 |
|  1024 |    256 |  11264 |    0.523 |  1957.58 |    6.059 |    42.25 |
|  1024 |    256 |  12288 |    0.541 |  1893.88 |    6.187 |    41.38 |
|  1024 |    256 |  13312 |    0.555 |  1845.18 |    6.320 |    40.50 |
|  1024 |    256 |  14336 |    0.572 |  1790.66 |    6.480 |    39.50 |
|  1024 |    256 |  15360 |    0.588 |  1740.57 |    6.618 |    38.68 |
|  1024 |    256 |  16384 |    0.606 |  1690.16 |    6.745 |    37.96 |

## IQ5_K PR559@3dbc8437 -ub 512 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.223 |  2298.96 |    2.285 |    56.01 |
|   512 |    128 |    512 |    0.226 |  2261.09 |    2.314 |    55.31 |
|   512 |    128 |   1024 |    0.230 |  2226.23 |    2.350 |    54.47 |
|   512 |    128 |   1536 |    0.233 |  2194.14 |    2.390 |    53.56 |
|   512 |    128 |   2048 |    0.237 |  2158.63 |    2.422 |    52.84 |
|   512 |    128 |   2560 |    0.242 |  2118.92 |    2.455 |    52.15 |
|   512 |    128 |   3072 |    0.245 |  2088.94 |    2.489 |    51.43 |
|   512 |    128 |   3584 |    0.250 |  2044.38 |    2.523 |    50.73 |
|   512 |    128 |   4096 |    0.252 |  2028.45 |    2.562 |    49.96 |
|   512 |    128 |   4608 |    0.258 |  1983.71 |    2.596 |    49.30 |
|   512 |    128 |   5120 |    0.261 |  1958.57 |    2.627 |    48.73 |
|   512 |    128 |   5632 |    0.265 |  1932.14 |    2.659 |    48.14 |
|   512 |    128 |   6144 |    0.271 |  1890.02 |    2.692 |    47.54 |
|   512 |    128 |   6656 |    0.273 |  1875.44 |    2.724 |    46.98 |
|   512 |    128 |   7168 |    0.276 |  1853.02 |    2.762 |    46.34 |
|   512 |    128 |   7680 |    0.281 |  1822.79 |    2.795 |    45.80 |
|   512 |    128 |   8192 |    0.285 |  1797.94 |    2.826 |    45.29 |
|   512 |    128 |   8704 |    0.289 |  1773.17 |    2.860 |    44.75 |
|   512 |    128 |   9216 |    0.292 |  1750.49 |    2.903 |    44.09 |
|   512 |    128 |   9728 |    0.297 |  1722.68 |    2.938 |    43.56 |
|   512 |    128 |  10240 |    0.300 |  1708.24 |    2.971 |    43.08 |
|   512 |    128 |  10752 |    0.303 |  1691.26 |    3.007 |    42.56 |
|   512 |    128 |  11264 |    0.308 |  1663.54 |    3.039 |    42.12 |
|   512 |    128 |  11776 |    0.312 |  1641.40 |    3.073 |    41.66 |
|   512 |    128 |  12288 |    0.315 |  1625.32 |    3.101 |    41.27 |
|   512 |    128 |  12800 |    0.320 |  1601.04 |    3.142 |    40.73 |
|   512 |    128 |  13312 |    0.323 |  1586.34 |    3.177 |    40.29 |
|   512 |    128 |  13824 |    0.326 |  1569.61 |    3.208 |    39.90 |
|   512 |    128 |  14336 |    0.331 |  1545.56 |    3.241 |    39.49 |
|   512 |    128 |  14848 |    0.334 |  1531.33 |    3.274 |    39.10 |
|   512 |    128 |  15360 |    0.338 |  1514.19 |    3.310 |    38.67 |
|   512 |    128 |  15872 |    0.343 |  1493.92 |    3.342 |    38.30 |
|   512 |    128 |  16384 |    0.347 |  1475.48 |    3.372 |    37.96 |

</details>

---

👤 **ikawrakow** commented on **2025-06-27** at **14:48:26**

> So for IQ4_K the sweet spot is closer to -ub 2048 -b 2048

I guess you have a higher end GPU. On my RTX-4080 for fully offloaded dense model the peak is somewhere between `-ub 512` and `-ub 1024`. 

But the `Q6_K` comparison with mainline is interesting. It means Johannes has improved the block-of-16 kernel, which here has an unreasonably low performance. I need to look into that. Can you try another block-of-16 quant that also works in mainline? (`Q2_K`, `Q3_K`, `IQ2_XS`, `IQ2_S` all have blocks of 16).

---

👤 **ubergarm** commented on **2025-06-27** at **15:17:24**

> Can you try another block-of-16 quant that also works in mainline? (Q2_K, Q3_K, IQ2_XS, IQ2_S all have blocks of 16).

Here is the IQ2_XS on both forks with `-ub 2048 -b 2048` as well as default values of `-ub 512 -b 2048`. The llama.cpp sha1 references my own fork with llama-sweep-bench rebased on top of recent `llama.cpp@8d94219a`

![sweep-bench-pr559-IQ2_XS-comparison](https://github.com/user-attachments/assets/1f278c6e-cbc6-4741-8df3-5881c667617e)

<details>

<summary>data</summary>

## IQ2_XS ik_llama.cpp@3dbc8437 -ub 2048 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    0.674 |  3038.23 |    6.037 |    84.81 |
|  2048 |    512 |   2048 |    0.729 |  2811.15 |    6.517 |    78.56 |
|  2048 |    512 |   4096 |    0.792 |  2587.22 |    7.054 |    72.58 |
|  2048 |    512 |   6144 |    0.855 |  2396.34 |    7.575 |    67.59 |
|  2048 |    512 |   8192 |    0.927 |  2209.61 |    8.126 |    63.01 |
|  2048 |    512 |  10240 |    0.999 |  2049.15 |    8.692 |    58.91 |
|  2048 |    512 |  12288 |    1.067 |  1919.42 |    9.249 |    55.36 |
|  2048 |    512 |  14336 |    1.133 |  1808.12 |    9.809 |    52.20 |
|  2048 |    512 |  16384 |    1.202 |  1703.61 |   10.355 |    49.45 |

## IQ2_XS ik_llama.cpp@3dbc8437 -ub 512 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.205 |  2494.23 |    1.528 |    83.78 |
|   512 |    128 |    512 |    0.209 |  2453.36 |    1.561 |    81.98 |
|   512 |    128 |   1024 |    0.213 |  2400.31 |    1.597 |    80.13 |
|   512 |    128 |   1536 |    0.216 |  2366.51 |    1.632 |    78.43 |
|   512 |    128 |   2048 |    0.220 |  2329.54 |    1.669 |    76.67 |
|   512 |    128 |   2560 |    0.224 |  2282.28 |    1.703 |    75.16 |
|   512 |    128 |   3072 |    0.228 |  2241.91 |    1.736 |    73.75 |
|   512 |    128 |   3584 |    0.231 |  2213.82 |    1.770 |    72.30 |
|   512 |    128 |   4096 |    0.235 |  2175.32 |    1.806 |    70.89 |
|   512 |    128 |   4608 |    0.240 |  2134.20 |    1.842 |    69.48 |
|   512 |    128 |   5120 |    0.244 |  2099.82 |    1.876 |    68.23 |
|   512 |    128 |   5632 |    0.247 |  2069.75 |    1.910 |    67.03 |
|   512 |    128 |   6144 |    0.251 |  2037.92 |    1.943 |    65.88 |
|   512 |    128 |   6656 |    0.255 |  2010.66 |    1.979 |    64.67 |
|   512 |    128 |   7168 |    0.258 |  1982.53 |    2.011 |    63.65 |
|   512 |    128 |   7680 |    0.262 |  1952.23 |    2.049 |    62.46 |
|   512 |    128 |   8192 |    0.266 |  1924.80 |    2.081 |    61.52 |
|   512 |    128 |   8704 |    0.271 |  1887.20 |    2.115 |    60.51 |
|   512 |    128 |   9216 |    0.274 |  1866.07 |    2.149 |    59.57 |
|   512 |    128 |   9728 |    0.279 |  1834.82 |    2.184 |    58.62 |
|   512 |    128 |  10240 |    0.283 |  1811.73 |    2.217 |    57.73 |
|   512 |    128 |  10752 |    0.286 |  1787.92 |    2.251 |    56.86 |
|   512 |    128 |  11264 |    0.290 |  1766.16 |    2.285 |    56.03 |
|   512 |    128 |  11776 |    0.295 |  1737.30 |    2.319 |    55.20 |
|   512 |    128 |  12288 |    0.299 |  1715.08 |    2.352 |    54.43 |
|   512 |    128 |  12800 |    0.304 |  1684.40 |    2.386 |    53.64 |
|   512 |    128 |  13312 |    0.307 |  1667.90 |    2.417 |    52.96 |
|   512 |    128 |  13824 |    0.310 |  1653.42 |    2.451 |    52.21 |
|   512 |    128 |  14336 |    0.314 |  1630.45 |    2.485 |    51.51 |
|   512 |    128 |  14848 |    0.319 |  1604.84 |    2.522 |    50.75 |
|   512 |    128 |  15360 |    0.324 |  1582.47 |    2.553 |    50.13 |
|   512 |    128 |  15872 |    0.327 |  1565.88 |    2.588 |    49.47 |
|   512 |    128 |  16384 |    0.330 |  1549.55 |    2.622 |    48.81 |

## IQ2_XS llama.cpp@6c510f3b -ub 2048 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    0.880 |  2326.71 |    6.355 |    80.57 |
|  2048 |    512 |   2048 |    0.942 |  2173.10 |    6.991 |    73.23 |
|  2048 |    512 |   4096 |    1.012 |  2024.59 |    7.567 |    67.66 |
|  2048 |    512 |   6144 |    1.084 |  1890.13 |    8.183 |    62.57 |
|  2048 |    512 |   8192 |    1.157 |  1770.18 |    8.748 |    58.53 |
|  2048 |    512 |  10240 |    1.228 |  1667.69 |    9.314 |    54.97 |
|  2048 |    512 |  12288 |    1.297 |  1578.65 |    9.906 |    51.69 |
|  2048 |    512 |  14336 |    1.363 |  1502.27 |   10.476 |    48.87 |
|  2048 |    512 |  16384 |    1.432 |  1430.39 |   11.050 |    46.34 |

## IQ2_XS llama.cpp@6c510f3b -ub 512 -b 2048
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.218 |  2344.71 |    1.566 |    81.73 |
|   512 |    128 |    512 |    0.220 |  2322.95 |    1.609 |    79.56 |
|   512 |    128 |   1024 |    0.222 |  2301.97 |    1.643 |    77.91 |
|   512 |    128 |   1536 |    0.227 |  2259.38 |    1.675 |    76.40 |
|   512 |    128 |   2048 |    0.230 |  2226.16 |    1.702 |    75.22 |
|   512 |    128 |   2560 |    0.235 |  2181.74 |    1.744 |    73.40 |
|   512 |    128 |   3072 |    0.238 |  2149.97 |    1.782 |    71.82 |
|   512 |    128 |   3584 |    0.242 |  2116.66 |    1.810 |    70.73 |
|   512 |    128 |   4096 |    0.245 |  2088.01 |    1.853 |    69.08 |
|   512 |    128 |   4608 |    0.249 |  2058.57 |    1.890 |    67.72 |
|   512 |    128 |   5120 |    0.253 |  2026.99 |    1.911 |    66.99 |
|   512 |    128 |   5632 |    0.256 |  1998.94 |    1.956 |    65.44 |
|   512 |    128 |   6144 |    0.261 |  1960.68 |    1.994 |    64.20 |
|   512 |    128 |   6656 |    0.264 |  1936.94 |    2.027 |    63.15 |
|   512 |    128 |   7168 |    0.270 |  1898.54 |    2.068 |    61.88 |
|   512 |    128 |   7680 |    0.273 |  1877.47 |    2.105 |    60.79 |
|   512 |    128 |   8192 |    0.276 |  1852.21 |    2.135 |    59.94 |
|   512 |    128 |   8704 |    0.280 |  1825.56 |    2.177 |    58.81 |
|   512 |    128 |   9216 |    0.286 |  1792.70 |    2.214 |    57.82 |
|   512 |    128 |   9728 |    0.289 |  1772.26 |    2.247 |    56.97 |
|   512 |    128 |  10240 |    0.293 |  1747.48 |    2.291 |    55.87 |
|   512 |    128 |  10752 |    0.298 |  1719.77 |    2.327 |    55.00 |
|   512 |    128 |  11264 |    0.302 |  1694.59 |    2.356 |    54.33 |
|   512 |    128 |  11776 |    0.305 |  1678.35 |    2.401 |    53.32 |
|   512 |    128 |  12288 |    0.310 |  1653.56 |    2.436 |    52.55 |
|   512 |    128 |  12800 |    0.316 |  1620.14 |    2.466 |    51.91 |
|   512 |    128 |  13312 |    0.318 |  1608.49 |    2.513 |    50.94 |
|   512 |    128 |  13824 |    0.323 |  1587.40 |    2.552 |    50.16 |
|   512 |    128 |  14336 |    0.326 |  1568.80 |    2.577 |    49.67 |
|   512 |    128 |  14848 |    0.332 |  1539.96 |    2.624 |    48.78 |
|   512 |    128 |  15360 |    0.336 |  1523.84 |    2.663 |    48.06 |
|   512 |    128 |  15872 |    0.339 |  1510.49 |    2.693 |    47.54 |
|   512 |    128 |  16384 |    0.344 |  1490.30 |    2.732 |    46.85 |


</details>

Interestingly mainline llama.cpp is slightly *slower* when increasing ubatch size over default.

---

👤 **ikawrakow** commented on **2025-06-27** at **15:24:19**

Oops, sorry, I misread your `Q6_K` graph, thinking `llama.cpp` has somehow become faster. So, nothing new under the sun.

---

👤 **ikawrakow** commented on **2025-06-27** at **15:34:33**

So, the A6000 has more memory bandwidth than the 4080. This shifts things in favor of dequantize+cuBLAS because the dequantize step is memory bound, so it is quicker on the A6000. I guess this is why with `-ub 4096` `IQ4_K` outperforms `IQ4_KS`. I guess, I should look into making the thresholds at which the transitions between MMQ and dequantize+cuBLAS happens configurable. But I'll leave this for another PR.

---

👤 **ikawrakow** commented on **2025-06-27** at **15:43:44**

Based on @ubergarm's and my own testing this PR looks like a winner, so merging.

---

👤 **ewhacc** commented on **2025-06-27** at **17:55:39**

@ubergarm 

Hi, this is u/smflx in reddit.  Thanks a lot for detailed reply.   :)

Yes, the old test was on 1843ed2 which is little before PR557.   This, PR557, PR559 are all the same PP speed of 272 t/s.   Yes, it was boosted recently.  If the boost specific to IQ2_K_R4 is already added, it's understandable. 

In your graph showing the boost on IQ2_K_R4, main is before PR557.  Right?   My PP speed of 272 t/s is similar to your S_PP  276.35 t/s.  So, it seems OK.   I will check llama-sweep-bench later. Thanks a lot for the the guide.

My setup is about the same except :  -DGGML_CUDA_F16=ON , -ctk q16_0

```
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1

CUDA_VISIBLE_DEVICES="0" \
ik_llama.cpp/build/bin/llama-server --model $model_path \
    --ctx-size 98304 \
    -mla 3 -fa -amb 512 -fmoe \
    -b 4096 -ub 4096 \
    --n-gpu-layers 63 \
    -ot "blk\.(3|4|5|6|7)\.ffn_.*=CUDA0" \
    --override-tensor exps=CPU \
    --parallel 2 --threads 32

CUDA_VISIBLE_DEVICES="0,1" \
ik_llama.cpp/build/bin/llama-server --model $model_path \
    --ctx-size 98304 \
    -mla 3 -fa -amb 512 -fmoe \
    -b 4096 -ub 4096 \
    --n-gpu-layers 63 \
    -ot "blk\.(3|4|5|6|7|8|9)\.ffn_.*=CUDA0" \
    -ot "blk\.1(0|1|2|3|4|5|6)\.ffn_.*=CUDA1" \
    --override-tensor exps=CPU \
    --parallel 2 --threads 32
```
I'm using 6000ada, but I think the speed will be the same to a6000.  GPUs are not fully utilized.  I guess PCIe speed is bottleneck.

---

👤 **ewhacc** commented on **2025-06-29** at **09:07:29**

@ubergarm 

I have tested with the same llama-sweep-bench setup you provided on my rig. 
Epyc 9534 + 2x 6000ada

I just changed the thread count to '--threads 32', which is optimal for 9534. 
Also tested with ' -ctk f16', which I use.  The speed is the same.  (But, 2x KV in VRAM)

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   10.274 |   398.66 |   48.584 |    21.08 |
|  4096 |   1024 |   4096 |   11.003 |   372.26 |   50.596 |    20.24 |
|  4096 |   1024 |   8192 |   11.893 |   344.41 |   51.931 |    19.72 |

---

👤 **ikawrakow** commented on **2025-06-29** at **16:28:37**

These performance results look pretty good to me. Has anyone ever reported a better result for hybrid GPU/CPU DeepSeek-R1/V3 inference?

---

👤 **Panchovix** commented on **2025-06-30** at **20:58:35**

Haven't managed to test much as I accidentaly wiped my Fedora installation from Windows lol. But I was testing with llama sweep bench and got one error, but can't remember exactly the error, and/or is related to this PR.

I have just saved at how I run the model, which is

```
./llama-sweep-bench -m '/models_llm/DeepSeek-V3-0324-UD-Q3_K_XL-merged.gguf' -c 32768 --no-mmap -ngl 999 \
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

I managed to see 200 t/s PP and 8.73 t/s TG, but then got a error. Again I will try to update when I get Linux installed again, as offloading + multigpu is just not worth it on Windows, speeds are way worse.

---

👤 **Panchovix** commented on **2025-07-01** at **15:43:44**

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

---

👤 **Panchovix** commented on **2025-07-02** at **20:12:03**

Okay finally got the other error.

```
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    9.166 |   223.43 |   56.876 |     9.00 |
|  2048 |    512 |   2048 |    9.549 |   214.48 |   57.088 |     8.97 |
|  2048 |    512 |   4096 |   10.041 |   203.96 |   57.929 |     8.84 |
|  2048 |    512 |   6144 |   10.534 |   194.42 |   58.584 |     8.74 |
Oops(ggml_compute_forward_sum_rows_f32, ffn_moe_weights_sum-60): found nan for i1 = 0, i2 = 0, i3 = 0. ne00 = 8
```
Sorry for the spam, gonna raise an issue, but I still don't know how to replicate it always.