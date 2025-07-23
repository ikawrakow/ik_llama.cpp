### 🗣️ [#385](https://github.com/ikawrakow/ik_llama.cpp/discussions/385) - Qwen3 235B performance on Intel Xeon Scalable processor

| **Author** | `Gaolingx` |
| :--- | :--- |
| **Created** | 2025-05-06 |
| **Updated** | 2025-05-27 |

---

#### Description

## Introduction

The Qwen3 models were officially released on 29th, April, 2025. This is a mixture-of-experts (MoE) models which 235B in total and 22B activated, here are the following features.

- Type: Causal Language Models
- Training Stage: Pretraining & Post-training
- Number of Parameters: 235B in total and 22B activated
- Number of Paramaters (Non-Embedding): 234B
- Number of Layers: 94
- Number of Attention Heads (GQA): 64 for Q and 4 for KV
- Number of Experts: 128
- Number of Activated Experts: 8
- Context Length: 32,768 natively and 131,072 tokens with YaRN.

The qwen3moe had supported in in PR #355, I tried to run the biggest model [Qwen3-235B-A22B-128K-GGUF](https://hf-mirror.com/unsloth/Qwen3-235B-A22B-128K-GGUF) with ik_llama.cpp on my Workstation, I need better generation quality, an my system has sufficient memory(Total 512G RAM), so I chose the relatively higher quality quantization `Q8_0`.

## System Info

Here are my SystemInfo(include hardware and software)

- Hardware
  - CPU: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz(20c, 40t) x2
  - RAM: RDIMM DDR4 2666 2Rx4 32G x16(12 Channels total)
  - Motherboard: Supermicro X11DPi-N
  - SSD: ZHITAI TiPlus7100 1TB
- Software
  - OS: Microsoft Windows 10 Pro
  - BIOS: Hyper-Threading-Enable, SNC-Disable
  - Model: Qwen3-235B-A22B-128K-Q8_0(unsloth/Qwen3-235B-A22B-128K-GGUF)
  - ik_llama.cpp:
  ```text
  INFO [                    main] build info | tid="61372" timestamp=1746525421 build=3667 commit="e3fec173"
  INFO [                    main] system info | tid="61372" timestamp=1746525421 n_threads=16 n_threads_batch=-1 total_threads=40 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
  ```

## Memory Performance

![cachemem2](https://github.com/user-attachments/assets/264caeef-bc57-4d42-9d8a-21b835fc9219)

## CPU-backend performance

The command line for is `ik_llama.cpp`

llama-sweep-bench:

```text
./llama-sweep-bench -m "%MODEL_PATH%" -c 16384 -t 20 -ngl 0 -fa
```

### ik_llama.cpp CPU-only performance data(Qwen3-235B-A22B-128K-Q8_0)

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 0, n_threads = 20, n_threads_batch = 20

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   67.198 |     7.62 |   53.220 |     2.41 |
|   512 |    128 |    512 |   65.739 |     7.79 |   51.455 |     2.49 |
|   512 |    128 |   1024 |   67.660 |     7.57 |   51.890 |     2.47 |
|   512 |    128 |   1536 |   68.719 |     7.45 |   52.238 |     2.45 |
|   512 |    128 |   2048 |   70.073 |     7.31 |   53.222 |     2.41 |
|   512 |    128 |   2560 |   71.726 |     7.14 |   53.961 |     2.37 |
|   512 |    128 |   3072 |   73.097 |     7.00 |   54.397 |     2.35 |
|   512 |    128 |   3584 |   74.688 |     6.86 |   54.247 |     2.36 |
|   512 |    128 |   4096 |   76.166 |     6.72 |   56.074 |     2.28 |
|   512 |    128 |   4608 |   78.441 |     6.53 |   55.985 |     2.29 |
|   512 |    128 |   5120 |   85.400 |     6.00 |   56.714 |     2.26 |
|   512 |    128 |   5632 |   80.910 |     6.33 |   58.679 |     2.18 |
|   512 |    128 |   6144 |   82.747 |     6.19 |   56.730 |     2.26 |
|   512 |    128 |   6656 |   83.653 |     6.12 |   57.644 |     2.22 |
|   512 |    128 |   7168 |   85.044 |     6.02 |   57.860 |     2.21 |
|   512 |    128 |   7680 |   86.687 |     5.91 |   59.510 |     2.15 |
|   512 |    128 |   8192 |   88.306 |     5.80 |   59.983 |     2.13 |
|   512 |    128 |   8704 |   95.135 |     5.38 |   58.736 |     2.18 |
|   512 |    128 |   9216 |   91.348 |     5.60 |   60.733 |     2.11 |
|   512 |    128 |   9728 |   97.391 |     5.26 |   60.376 |     2.12 |
|   512 |    128 |  10240 |   95.785 |     5.35 |   64.163 |     1.99 |
|   512 |    128 |  10752 |   98.549 |     5.20 |   63.393 |     2.02 |
|   512 |    128 |  11264 |   98.616 |     5.19 |   61.447 |     2.08 |
|   512 |    128 |  11776 |  105.775 |     4.84 |   65.116 |     1.97 |
|   512 |    128 |  12288 |  102.959 |     4.97 |   67.291 |     1.90 |
|   512 |    128 |  12800 |  105.210 |     4.87 |   65.661 |     1.95 |
|   512 |    128 |  13312 |  107.702 |     4.75 |   66.114 |     1.94 |
|   512 |    128 |  13824 |  109.233 |     4.69 |   64.225 |     1.99 |
|   512 |    128 |  14336 |  111.032 |     4.61 |   67.671 |     1.89 |
|   512 |    128 |  14848 |  114.479 |     4.47 |   66.681 |     1.92 |
|   512 |    128 |  15360 |  117.857 |     4.34 |   73.044 |     1.75 |
|   512 |    128 |  15872 |  120.052 |     4.26 |   71.046 |     1.80 |

---

![02](https://github.com/user-attachments/assets/9bbdc4f2-0222-4e68-bfa8-145cabe97691)

## ik_llama.cpp CPU-only performance data(Qwen3-30B-A3B-128K-GGUF)

I also experimented with `Qwen3-30B-A3B-128K-Q8_0(unsloth/Qwen3-235B-A22B-128K-GGUF)`, Here are the results, well, the performance is much better than I though.

main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 0, n_threads = 20, n_threads_batch = 20

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    8.519 |    60.10 |    9.924 |    12.90 |
|   512 |    128 |    512 |    8.950 |    57.21 |   10.045 |    12.74 |
|   512 |    128 |   1024 |    9.279 |    55.18 |   10.204 |    12.54 |
|   512 |    128 |   1536 |    9.648 |    53.07 |   10.613 |    12.06 |
|   512 |    128 |   2048 |   10.097 |    50.71 |   10.722 |    11.94 |
|   512 |    128 |   2560 |   10.486 |    48.83 |   11.015 |    11.62 |
|   512 |    128 |   3072 |   10.999 |    46.55 |   11.164 |    11.47 |
|   512 |    128 |   3584 |   11.336 |    45.17 |   11.139 |    11.49 |
|   512 |    128 |   4096 |   12.480 |    41.03 |   11.718 |    10.92 |
|   512 |    128 |   4608 |   12.244 |    41.82 |   11.725 |    10.92 |
|   512 |    128 |   5120 |   12.551 |    40.79 |   12.213 |    10.48 |
|   512 |    128 |   5632 |   13.537 |    37.82 |   12.453 |    10.28 |
|   512 |    128 |   6144 |   13.356 |    38.34 |   12.584 |    10.17 |
|   512 |    128 |   6656 |   13.847 |    36.98 |   12.603 |    10.16 |
|   512 |    128 |   7168 |   14.128 |    36.24 |   12.656 |    10.11 |
|   512 |    128 |   7680 |   14.631 |    34.99 |   13.198 |     9.70 |
|   512 |    128 |   8192 |   15.002 |    34.13 |   13.520 |     9.47 |
|   512 |    128 |   8704 |   15.356 |    33.34 |   13.095 |     9.77 |
|   512 |    128 |   9216 |   16.050 |    31.90 |   13.614 |     9.40 |
|   512 |    128 |   9728 |   16.395 |    31.23 |   13.093 |     9.78 |
|   512 |    128 |  10240 |   16.790 |    30.49 |   14.537 |     8.80 |
|   512 |    128 |  10752 |   17.052 |    30.03 |   14.793 |     8.65 |
|   512 |    128 |  11264 |   17.668 |    28.98 |   13.957 |     9.17 |
|   512 |    128 |  11776 |   18.276 |    28.02 |   15.028 |     8.52 |
|   512 |    128 |  12288 |   18.335 |    27.92 |   15.267 |     8.38 |
|   512 |    128 |  12800 |   19.061 |    26.86 |   15.272 |     8.38 |
|   512 |    128 |  13312 |   19.379 |    26.42 |   15.310 |     8.36 |
|   512 |    128 |  13824 |   19.764 |    25.91 |   15.000 |     8.53 |
|   512 |    128 |  14336 |   20.432 |    25.06 |   15.612 |     8.20 |
|   512 |    128 |  14848 |   21.632 |    23.67 |   15.587 |     8.21 |
|   512 |    128 |  15360 |   22.311 |    22.95 |   17.303 |     7.40 |
|   512 |    128 |  15872 |   21.767 |    23.52 |   16.894 |     7.58 |

---

![03](https://github.com/user-attachments/assets/3f4f1148-85dc-471d-85ee-0a4afa13db07)

## Profiler Data

I also use `Intel VTune Profiler 2025.0.1` capture some interesting data when running llama-server with `Qwen3-30B-A3B-128K-Q8_0`, I will show them as well.

![2025-05-04T15_17_00](https://github.com/user-attachments/assets/8ed1d864-4cb5-483b-9df9-a72bbbfc426b)

![2025-05-04T15_51_53](https://github.com/user-attachments/assets/152044c8-9a54-4992-8afb-501a791260c6)

![2025-05-04T15_52_19](https://github.com/user-attachments/assets/5af8f7da-8b6d-4686-a4c9-68c7ffeb2925)

---

#### 🗣️ Discussion

👤 **ikawrakow** replied the **2025-05-06** at **13:11:51**:<br>

Thank you for these results. Quite amazing that it works reasonably well on an almost 8 years old CPU!

I'm curious if you might get better performance by repacking the model (unlikely for TG, very likely for PP). You can repack either on the fly by adding `-rtr` to the command line, or offline like this
```
./bin/llama-quantize --repack $model $repacked_model q8_0_r8
```
This shouldn't take very long, even for the 235B model.

Another note: at least on the CPUs that I have available, one gets better performance using `q8_0` KV cache (add `-ctk q8_0 -ctv q8_0` to the command line). Not so much for short contexts, but quite noticeable for long contexts.

> 👤 **saood06** replied the **2025-05-06** at **20:29:54**:<br>
> > Another note: at least on the CPUs that I have available, one gets better performance using `q8_0` KV cache (add `-ctk q8_0 -ctv q8_0` to the command line). Not so much for short contexts, but quite noticeable for long contexts.
> 
> I have seen this https://www.reddit.com/r/LocalLLaMA/comments/1kewkno/qwen_30b_a3b_performance_degradation_with_kv/ where they report using `q8_0` KV cache causes the model to not able to solve a problem with a comment saying:
> ```
> KV cache q8_0: 0/5
> KV cache f16: 2/2
> ```
> 
> 👤 **Gaolingx** replied the **2025-05-07** at **07:16:13**:<br>
> Ok, Thanks for the info. I found that the memory bandwidth was not filled when I use vtune profiler analysis the memory access, Maybe numa system works in Linux better, I will try to use `numactl` changes the memory policy ([https://github.com/ggml-org/llama.cpp/issues/1437](https://github.com/ggml-org/llama.cpp/issues/1437)), and repack the model with `q8_0_r8`. I will see if I can do better yet however.

---

👤 **Gaolingx** replied the **2025-05-07** at **18:42:39**:<br>

Note: when I run llama-server with `-fa` and `-rtr` parameter, the speed is a little faster than only use `-fa`, the prefill and decode are increased, That is a good beginning!

`-c 8192 -t 16 -fa`:
```text
INFO [           print_timings] prompt eval time     =  197624.81 ms /  1266 tokens (  156.10 ms per token,     6.41 tokens per second) | tid="46204" timestamp=1746371113 id_slot=0 id_task=4917 t_prompt_processing=197624.812 n_prompt_tokens_processed=1266 t_token=156.10174723538705 n_tokens_second=6.406078200342577
INFO [           print_timings] generation eval time =  372468.51 ms /   861 runs   (  432.60 ms per token,     2.31 tokens per second) | tid="46204" timestamp=1746371113 id_slot=0 id_task=4917 t_token_generation=372468.513 n_decoded=861 t_token=432.5998989547038 n_tokens_second=2.3116047932889296
INFO [           print_timings]           total time =  570093.32 ms | tid="46204" timestamp=1746371113 id_slot=0 id_task=4917 t_prompt_processing=197624.812 t_token_generation=372468.513 t_total=570093.325
```

`-c 8192 -t 16 -fa -rtr`:
```text
INFO [           print_timings] prompt eval time     =    9707.99 ms /   168 tokens (   57.79 ms per token,    17.31 tokens per second) | tid="46820" timestamp=1746855833 id_slot=0 id_task=9260 t_prompt_processing=9707.992 n_prompt_tokens_processed=168 t_token=57.78566666666667 n_tokens_second=17.30532946463079
INFO [           print_timings] generation eval time =   26156.20 ms /    76 runs   (  344.16 ms per token,     2.91 tokens per second) | tid="46820" timestamp=1746855833 id_slot=0 id_task=9260 t_token_generation=26156.196 n_decoded=76 t_token=344.1604736842105 n_tokens_second=2.905621291414088
INFO [           print_timings]           total time =   35864.19 ms | tid="46820" timestamp=1746855833 id_slot=0 id_task=9260 t_prompt_processing=9707.992 t_token_generation=26156.196 t_total=35864.188
```

---

👤 **ikawrakow** replied the **2025-05-08** at **12:59:17**:<br>

@saood06 

> I have seen this https://www.reddit.com/r/LocalLLaMA/comments/1kewkno/qwen_30b_a3b_performance_degradation_with_kv/ where they report using q8_0 KV cache causes the model to not able to solve a problem with a comment saying:

This grabbed my attention as I have never seen any significant difference between `f16` and `q8_0` KV cache (if anything, I would be more suspect towards `f16` because it can overflow, and I think there have been reports about that). So, being someone who does not take thinks for granted, I tried it myself.

### Attempt 1

* I saw Redditor is using a `Q4_K_M` model, so try a stock `Q4_K_M` quantization
* `f16` and `Q8_0` KV cache both fail in all 3 attempts
* `f16` and `q8_0` both at some point arrive at the correct conclusion that two characters in the encoded text correspond to a single letter, but both abandon the idea after some unsuccessful attempts
* `f16` and `q8_0` both enter into seemingly infinite loop of trying the same ideas again and again. Sometimes they stop and give an incorrect answer, sometimes they keep going until they run out of tokens (I gave a limit of 20k tokens)

### Attempt 2
* Quantize to stock `IQ4_K`
* 3 attempts with `f16` and 3 attempts with `q8_0`. Each attempt uses the same seed for `q8_0` and for `f16`, but there are 3 different seeds for the 3 attempts
* `f16`: 2 out of 3 correct. The failed attempt runs out of tokens. Correct, Correct, Incorrect
* `q8_0`: 2 out of 3 correct. The failed attempt comes back with an incorrect result after about 12k tokens. Correct, Incorrect, Correct
* Each run consumes a different amount of thinking tokens

Hence, I think that the outcome is largely determined by the quality of the quantized model and by some luck. We know that in a random process (as we have here) slight differences in the computed token probabilities can make the model go on a very different path, even if the same seed was used.

> 👤 **saood06** replied the **2025-05-08** at **22:40:13**:<br>
> >So, being someone who does not take thinks for granted, I tried it myself.
> 
> Thank you. Do you mind saying what sampler settings you used?
> 
> > Hence, I think that the outcome is largely determined by the quality of the quantized model and by some luck. We know that in a random process (as we have here) slight differences in the computed token probabilities can make the model go on a very different path, even if the same seed was used.
> 
> The "luck" factor can be at least somewhat lessened based on how you sample (and why I like manually sampling and exploring many branches, and often injecting in tokens that would others never be sampled [since min_p would have removed it as it would be too low]). In my experience there are places where the "luck" of a single token selected by sane sampler settings does have an outsized impact on the internal world state, but often it doesn't with the model using different words or changing trivial things but otherwise staying on the same track. Either way for entire responses yes, there are often large variations between seeds and sampling parameters.
> 
> There are other ways that are being researched to try and improve outcomes such as using majority voting, incorporating scoring models or reward models and other highly compute intensive ways of trying to eek out more performance and consistency from models but for me manually sampling works well (and I also find it interesting and enjoyable trying to create a mental model of the AI's mental model).
> 
> >This grabbed my attention as I have never seen any significant difference between f16 and q8_0 KV cache (if anything, I would be more suspect towards f16 because it can overflow, and I think there have been reports about that).
> 
> For me, with Deepseek based models I tend to use f16 as I don't see the need to save the space and speed is very close between them, but with other models I do quantize the KV cache, so I was also really surprised by the thread I linked. One last thing I saw in there that I forgot to mention was him stating "I know but as a side test I tried also Roo Code that I could not get to use all the tools with KV cache Q8 and worked fine with F16." so I'm not sure why his experience shows such stark differences that I also have never really experienced.

---

👤 **Gaolingx** replied the **2025-05-13** at **00:52:27**:<br>

Note: qwen3moe uses 8 experts by default. I found that we can speed up token generation(2.7 token/s->3.2 token/s) by reducing some experts used (from Top-8 to Top-6), without a significant drop in quality.

parameter:
`.\llama-server --model "%MODEL%" --host %HOST% --port %PORT% --threads 16 --n-gpu-layers 0 --ctx-size 8192 --flash-attn --run-time-repack --override-kv qwen3moe.expert_used_count=int:6`

```text
INFO [           print_timings] prompt eval time     =   10360.09 ms /   153 tokens (   67.71 ms per token,    14.77 tokens per second) | tid="71476" timestamp=1747096864 id_slot=0 id_task=9696 t_prompt_processing=10360.092 n_prompt_tokens_processed=153 t_token=67.71301960784314 n_tokens_second=14.768208622085595
INFO [           print_timings] generation eval time =   15317.10 ms /    50 runs   (  306.34 ms per token,     3.26 tokens per second) | tid="71476" timestamp=1747096864 id_slot=0 id_task=9696 t_token_generation=15317.103 n_decoded=50 t_token=306.34206 n_tokens_second=3.2643248530743705
INFO [           print_timings]           total time =   25677.19 ms | tid="71476" timestamp=1747096864 id_slot=0 id_task=9696 t_prompt_processing=10360.092 t_token_generation=15317.103 t_total=25677.195
```

> 👤 **saood06** replied the **2025-05-13** at **01:03:32**:<br>
> > Note: qwen3moe uses 8 experts by default. I found that we can speed up token generation(2.7 token/s->3.2 token/s) by reducing some experts used (from Top-8 to Top-6), without a significant drop in quality.
> 
> There is this feature: https://github.com/ikawrakow/ik_llama.cpp/pull/239 I personally haven't had much success using it (for Deepseek V3/R1) , but it may work for you on Qwen.
> 
> 👤 **Gaolingx** replied the **2025-05-13** at **01:45:22**:<br>
> > > Note: qwen3moe uses 8 experts by default. I found that we can speed up token generation(2.7 token/s->3.2 token/s) by reducing some experts used (from Top-8 to Top-6), without a significant drop in quality.
> > 
> > There is this feature: #239 I personally haven't had much success using it (for Deepseek V3/R1) , but it may work for you on Qwen.
> 
> All right, it seems that `--smart-expert-reduction` not works well on qwen3moe, there are a lot of garbled characters appeared and continuous output appeared.
> 
> `--flash-attn --run-time-repack --smart-expert-reduction 6,1`
> ![批注 2025-05-13 093200](https://github.com/user-attachments/assets/3320649a-ae4f-466e-a2f6-dcc949ca4919)
> 
> `--flash-attn --run-time-repack --smart-expert-reduction 7,1`
> ![批注 2025-05-13 094242](https://github.com/user-attachments/assets/370fb493-a9c7-42c7-a380-90935df8f23e)
> 
> 👤 **ikawrakow** replied the **2025-05-13** at **12:35:23**:<br>
> Can you both try PR #415 and let me know if it now works? Thanks!
> 
> 👤 **Gaolingx** replied the **2025-05-14** at **01:42:24**:<br>
> > Can you both try PR #415 and let me know if it now works? Thanks!
> 
> yes, I pulled PR(#415 ), The smart expert reduction works very well on cpu backend, thank you fix it.
> ![批注 2025-05-14 093324](https://github.com/user-attachments/assets/88e0af59-555c-4375-b5f8-78e0fd7789e7)
> 
> `--flash-attn --run-time-repack --smart-expert-reduction 6,1`
> 
> ```text
> INFO [           print_timings] prompt eval time     =    8951.82 ms /   165 tokens (   54.25 ms per token,    18.43 tokens per second) | tid="52244" timestamp=1747186657 id_slot=0 id_task=491 t_prompt_processing=8951.82 n_prompt_tokens_processed=165 t_token=54.253454545454545 n_tokens_second=18.432006005482684
> INFO [           print_timings] generation eval time =   24997.27 ms /    86 runs   (  290.67 ms per token,     3.44 tokens per second) | tid="52244" timestamp=1747186657 id_slot=0 id_task=491 t_token_generation=24997.269 n_decoded=86 t_token=290.66591860465115 n_tokens_second=3.4403758266553037
> INFO [           print_timings]           total time =   33949.09 ms | tid="52244" timestamp=1747186657 id_slot=0 id_task=491 t_prompt_processing=8951.82 t_token_generation=24997.269 t_total=33949.089
> ```

---

👤 **VinnyG9** replied the **2025-05-19** at **15:30:30**:<br>

you forgot to set -nkvo?
what snoop mode you're using for numa?
 are you using one node?
here's some numbers on the xeon v4 @Q2KL

| model                             |      size |   params | backend | ngl | threads | fa | amb | ser | rtr | fmoe |  test |           t/s |
| ----------------------------------- | ----------: | ---------: | --------- | ----: | --------: | ---: | ----: | ----: | ----: | -----: | ------: | --------------: |
| ============ Repacked 659 tensors |           |          |         |     |         |    |     |     |     |      |       |               |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |   0 |      31 |  1 | 512 | 4,1 |   1 |    1 |  pp32 | 34.41 ± 2.53 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |   0 |      31 |  1 | 512 | 4,1 |   1 |    1 |  pp64 | 44.84 ± 1.45 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |   0 |      31 |  1 | 512 | 4,1 |   1 |    1 | pp128 | 54.11 ± 0.49 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |   0 |      31 |  1 | 512 | 4,1 |   1 |    1 | pp256 | 55.99 ± 2.86 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |   0 |      31 |  1 | 512 | 4,1 |   1 |    1 |  tg32 |  6.73 ± 0.14 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |   0 |      31 |  1 | 512 | 4,1 |   1 |    1 |  tg64 |  7.28 ± 0.38 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |   0 |      31 |  1 | 512 | 4,1 |   1 |    1 | tg128 |  8.29 ± 0.25 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |   0 |      31 |  1 | 512 | 4,1 |   1 |    1 | tg256 |  8.65 ± 0.20 |

---

👤 **ikawrakow** replied the **2025-05-19** at **15:38:58**:<br>

You cannot  compare `Q2_K` to `Q8_0` for TG, there is going to be a factor in the range of 3X difference. Her PP is for a short prompt, and we don't know if it was a single prompt of 165 tokens or 10 prompts with 16 tokens each.

> 👤 **VinnyG9** replied the **2025-05-19** at **15:48:34**:<br>
> > You cannot compare `Q2_K` to `Q8_0` for TG, there is going to be a factor in the range of 3X difference. Her PP is for a short prompt, and we don't know if it was a single prompt of 165 tokens or 10 prompts with 16 tokens each.
> 
> or 2.5x going by model size :)
> i didn't mean to compare apples to apples just want to see more CPU benchmarks on the big MoEs, and point out OP is on a multi node system with HT On but limiting it to 25% of total threads(the MoEs will scale w/ all threads)
> no --numa flag, no info on snoop mode which makes the biggest difference I've seen in my tests 
> 
>  multi socket is way more complicated but can be worth it

---

👤 **Gaolingx** replied the **2025-05-27** at **13:06:54**:<br>

Well, I use `-ser 4,1` parameter to improve token generation(TG) performance, now we can get ~4.1 token/s TG(< 4k context size), and the 
quality not declined too much. all right, I admit this is just my opinion. Others can offer their own opinions on this point...We don't know what will happen in complex tasks...

`.\llama-server --model "%MODEL%" --host %HOST% --port %PORT% --threads 16 --n-gpu-layers 0 --ctx-size 8192 --flash-attn --run-time-repack --fused-moe --smart-expert-reduction 4,1`

```text
INFO [           print_timings] prompt eval time     =    3343.34 ms /    66 tokens (   50.66 ms per token,    19.74 tokens per second) | tid="12196" timestamp=1748316424 id_slot=0 id_task=5716 t_prompt_processing=3343.336 n_prompt_tokens_processed=66 t_token=50.65660606060606 n_tokens_second=19.740761921625587
INFO [           print_timings] generation eval time =  177876.86 ms /   731 runs   (  243.33 ms per token,     4.11 tokens per second) | tid="12196" timestamp=1748316424 id_slot=0 id_task=5716 t_token_generation=177876.858 n_decoded=731 t_token=243.3335950752394 n_tokens_second=4.109584620614335
INFO [           print_timings]           total time =  181220.19 ms | tid="12196" timestamp=1748316424 id_slot=0 id_task=5716 t_prompt_processing=3343.336 t_token_generation=177876.858 t_total=181220.19400000002
```
---
![image](https://github.com/user-attachments/assets/7ba9179c-a661-466d-bba8-518ea755d082)