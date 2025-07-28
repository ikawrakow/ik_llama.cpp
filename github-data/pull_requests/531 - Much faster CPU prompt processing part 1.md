## 🔀 [Pull Request #531](https://github.com/ikawrakow/ik_llama.cpp/pull/531) - Much faster CPU prompt processing (part 1)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/q6_k_gemm` |
| **Target Branch** | `main` |
| **Created** | 2025-06-16 |
| **Updated** | 2025-06-17 |
| **Merged** | 2025-06-17 |

---

## 📄 Description

This PR is a continuation of [#515](https://github.com/ikawrakow/ik_llama.cpp/issues/515), [#516](https://github.com/ikawrakow/ik_llama.cpp/issues/516), [#517](https://github.com/ikawrakow/ik_llama.cpp/issues/517), [#518](https://github.com/ikawrakow/ik_llama.cpp/issues/518) with the following differences
* Quants are repacked to `Q8_K_R8` instead of `Q8_0_R8`. `Q8_K_R8` is the fastest quant known to human kind (see [#141](https://github.com/ikawrakow/ik_llama.cpp/issues/141)), and that helps achieve significant performance gains when batch size is greater than 32 tokens or so
* The technique of on-the-fly repacking before matrix multiplications is extended to a larger set of quants: `IQ1_M, IQ2_XS, IQ2_S, Q3_K` in addition to `IQ1_S, IQ2_XXS, IQ3_XXS, IQ3_S` already improved in the quoted PRs 
* There is also `Q6_K` added, but in this case repacking is to `Q8_0_R8` as `Q6_K` cannot be losslessly repacked to `Q8_K`, and I was worried that there could be a non-negligible accuracy loss due to that. 

The following table shows a PP-512 performance comparison between the main branch and this PR. Model is LlaMA-3.1-8B-Instruct. Quantization is always "pure" (i.e., all tensors except the output tensor and the token embedding tensor are quantized with the selected quantization type). CPU is Ryzen-7950X

| model            |       size |          test |              t/s |              t/s |  Speedup |
| -----------------| ---------: | ------------: | ---------------: | ---------------: | -------: |
| llama 8B IQ1_S   |   2.07 GiB |         pp512 |    264.36 ± 0.32 |    308.67 ± 3.45 |  1.168   |   
| llama 8B IQ1_M   |   2.21 GiB |         pp512 |     25.12 ± 0.15 |    309.81 ± 2.78 | 12.333   |   
| llama 8B IQ2_XXS |   2.35 GiB |         pp512 |    284.22 ± 2.46 |    344.02 ± 4.27 |  1.210   |   
| llama 8B IQ2_XS  |   2.56 GiB |         pp512 |    108.77 ± 2.32 |    346.11 ± 2.26 |  3.182   |   
| llama 8B IQ2_S   |   2.76 GiB |         pp512 |    101.43 ± 1.13 |    341.02 ± 1.60 |  3.362   |   
| llama 8B IQ3_XXS |   3.17 GiB |         pp512 |    280.56 ± 3.15 |    341.95 ± 3.33 |  1.219   |   
| llama 8B Q3_K    |   3.41 GiB |         pp512 |    178.56 ± 2.99 |    344.45 ± 4.15 |  1.929   |   
| llama 8B IQ3_S   |   3.47 GiB |         pp512 |    283.86 ± 2.62 |    340.68 ± 2.87 |  1.200   |   
| llama 8B Q6_K    |   6.14 GiB |         pp512 |    178.49 ± 1.78 |    271.50 ± 2.96 |  1.521   |   

A few notes:
* Gains for the quants that already had repacking to `Q8_0_R8` (`IQ1_S, IQ2_XXS, IQ3_XXS, IQ3_S`) are in the range of 15-20%
* `IQ1_M` stands out because it did not have a fast `iqk` GEMM implementation at all, so we gain a factor of 12X!
* The PR changes the status of i-quants from being slow for CPU inference to being among the fastest (well, at least at this point before I apply this technique to `IQX_K` quants).

I have the impression that most people use `ik_llama.cpp` for MoE models. MoE models are quite different compared to dense models such as LLaMA-3.1-8B because each routed expert "sees" a small fraction of the tokens in a batch, so effective batch size is much smaller compared to a dense model. Hence, PP performance gains for MoE models will be more modest. It is instructive to look as PP performance as a function of batch size. The following graph shows the result for `Q3_K`, which has a reasonably efficient `iqk` GEMM implementation. The repacking strategy kicks in at 32 tokens, so up to that point performance is the same. The relative performance gain from this PR then slowly grows to about 1.9X at 256 tokens, and remains (nearly) the same from there on.  

![z2](https://github.com/user-attachments/assets/34c92f90-ff68-427d-8232-720bcaddec30)

Based on this we can expect lower performance gains for a MoE model. For instance, DeepSeek-R1/V3 have 256 total experts but only 8 active experts, so effectively this strategy will not become active (or will have a very small impact) up to u-batch sizes of 1024 tokens. I cannot run DeepSeek-R1/V3, but I can run Qwen3-30B-A3B, and the next graphs shows performance for this model quantized with `Q3_K`. As expected, performance gains are smaller, about 1.4X at the peak, and poerformance improvement is not significant before 64 tokens.

  
![z3](https://github.com/user-attachments/assets/6370ace4-3ae6-4e3e-a5d0-a5846f4ed63a)

---

## 💬 Conversation

👤 **saood06** commented on **2025-06-16** at **10:26:55**

Does this also improve the behavior at higher contexts? For me running Deepseek at higher contexts PP and TG both approach ~1 t/s.

---

👤 **ikawrakow** commented on **2025-06-16** at **10:31:53**

> For me running Deepseek at higher contexts PP and TG both approach ~1 t/s.

This indicates that your computer spends the entire time computing self attention for long enough context. If so, this PR will have zero impact on your long context performance.

---

👤 **saood06** commented on **2025-06-16** at **12:25:14**

> This indicates that your computer spends the entire time computing self attention for long enough context. 

I'm trying to understand but that explanation (at least to me) doesn't explain why at low context PP uses a lot more power than TG (as it is compute bound), but at higher context the power usage looks a lot closer to TG (which is memory/QPI bandwidth bound).

I don't have actual numbers (as I don't think the exact numbers matter) but the difference is stark enough for me to notice based on the CPU temperatures.

---

👤 **ikawrakow** commented on **2025-06-16** at **12:53:47**

> but at higher context the power usage looks a lot closer to TG (which is memory/QPI bandwidth bound).

Or is it rather the other way around (TG looks a lot closer to PP)? If you buy my explanation that for a large context all the time is spent in the self attention calculation, then there isn't that much of a difference between TG and PP: for DeepSeek each row in the KV  cache multiples 128 rows of activations (`K*Q` and `V*softmax(K*Q)`), so the matrix multiplications in TG and PP have very similar characteristics (there isn't much of a difference between multiplying 128 rows and 128 x n_ubatch rows), and it is compute bound, not memory bound.

---

👤 **saood06** commented on **2025-06-16** at **13:54:42**

>If you buy my explanation

I do, I was just trying to understand it.

> Or is it rather the other way around (TG looks a lot closer to PP)? that for a large context all the time is spent in the self attention calculation, then there isn't that much of a difference between TG and PP: for DeepSeek each row in the KV cache multiples 128 rows of activations (`K*Q` and `V*softmax(K*Q)`), so the matrix multiplications in TG and PP have very similar characteristics (there isn't much of a difference between multiplying 128 rows and 128 x n_ubatch rows), and it is compute bound, not memory bound.

That makes sense. 

I did attempt to look at the [PCM](https://github.com/intel/pcm) data I had from earlier and just generated, and looked at CPU power usage and IPC but I'm not sure if the numbers are actually useful since I found during TG that it was causing paging (there really isn't much spare RAM on my system during inference).

---

👤 **ubergarm** commented on **2025-06-16** at **23:06:48**

Not a comprehensive test, but this `PR531` does indeed speed-up PP as compared to `main` on my DeepSeek-R1-0528-IQ1_S.

So while not as dramatic given only 58 `ffn_down_exps@iq1_m` on this MoE, the `iq1_s` speed-ups are already merged into main so overall much faster than before.

The `IQ1_S_R4` still benches faster for this specific configuration at least and seems to be the same speed on both this PR and main as I would expect.

Note, to keep it simple, I did *not* use `-rtr` to repack the attn/shexp tensors; so actual CPU-only scenario would likely be faster still.

## DeepSeek-R1-0528-IQ1_S
- type    f32:  361 tensors
- type   q4_0:   61 tensors `attn_k_b`
- type  iq1_s:  116 tensors `ffn_(gate|up)_exps`
- type  iq1_m:   58 tensors `ffn_down_exps`
- type iq4_ks:  551 tensors `everything else`

## DeepSeek-R1-0528-IQ1_S_R4
- type      f32:  361 tensors
- type     q4_0:   61 tensors `attn_k_b`
- type iq1_s_r4:  116 tensors `ffn_(gate|up)_exps`
- type iq1_m_r4:   58 tensors `ffn_down_exps`
- type   iq4_ks:  551 tensors `everything else`

Importantly, `llama-perplexity` runs clean on PR531@72fd9faa so the new `iq1_m` implementation seems solid. Here's the values using `-ctk f16`:

* `IQ1_S`: `Final estimate: PPL = 4.8910 +/- 0.02856`
* `IQ1_S_R4`: `Final estimate: PPL = 4.8805 +/- 0.02876` (computed back on PR494)

<details>

<summary>👈 sweep-bench data</summary>

```bash
model=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ1_S_R4/DeepSeek-R1-0528-IQ1_S_R4-00001-of-00003.gguf
#model=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ1_S/DeepSeek-R1-0528-IQ1_S-00001-of-00003.gguf

numactl -N 0 -m 0 \
./build/bin/llama-sweep-bench \
    --model "$model" \
    -c 8704 \
    -ctk q8_0 \
    -mla 3 -fa \
    -fmoe \
    --no-mmap \
    --threads 80 \
    --threads-batch 128 \
    --numa numactl \
    --warmup-batch
```

## DeepSeek-R1-0528-IQ1_S_R4
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    4.423 |   115.77 |   17.351 |     7.38 |
|   512 |    128 |    512 |    4.687 |   109.23 |   19.213 |     6.66 |
|   512 |    128 |   1024 |    5.096 |   100.46 |   19.777 |     6.47 |
|   512 |    128 |   1536 |    5.244 |    97.63 |   23.691 |     5.40 |
|   512 |    128 |   2048 |    6.130 |    83.52 |   23.180 |     5.52 |
|   512 |    128 |   2560 |    5.937 |    86.24 |   23.369 |     5.48 |
|   512 |    128 |   3072 |    6.240 |    82.05 |   23.431 |     5.46 |
|   512 |    128 |   3584 |    7.088 |    72.23 |   20.811 |     6.15 |
|   512 |    128 |   4096 |    7.450 |    68.72 |   23.252 |     5.50 |
|   512 |    128 |   4608 |    7.118 |    71.93 |   21.718 |     5.89 |
|   512 |    128 |   5120 |    7.433 |    68.88 |   21.636 |     5.92 |
|   512 |    128 |   5632 |    7.707 |    66.44 |   22.484 |     5.69 |
|   512 |    128 |   6144 |    8.019 |    63.85 |   22.216 |     5.76 |
|   512 |    128 |   6656 |    8.271 |    61.91 |   22.708 |     5.64 |
|   512 |    128 |   7168 |    8.604 |    59.51 |   24.151 |     5.30 |
|   512 |    128 |   7680 |    8.840 |    57.92 |   23.185 |     5.52 |
|   512 |    128 |   8192 |    9.295 |    55.08 |   22.992 |     5.57 |

## PR531@72fd9faa DeepSeek-R1-0528-IQ1_S
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    6.139 |    83.40 |   17.278 |     7.41 |
|   512 |    128 |    512 |    6.244 |    82.00 |   18.809 |     6.81 |
|   512 |    128 |   1024 |    6.436 |    79.55 |   21.856 |     5.86 |
|   512 |    128 |   1536 |    6.754 |    75.81 |   22.630 |     5.66 |
|   512 |    128 |   2048 |    7.189 |    71.22 |   23.058 |     5.55 |
|   512 |    128 |   2560 |    8.803 |    58.16 |   22.779 |     5.62 |
|   512 |    128 |   3072 |    9.001 |    56.88 |   22.750 |     5.63 |
|   512 |    128 |   3584 |    8.404 |    60.92 |   24.276 |     5.27 |
|   512 |    128 |   4096 |    9.322 |    54.93 |   23.410 |     5.47 |
|   512 |    128 |   4608 |    9.230 |    55.47 |   23.225 |     5.51 |
|   512 |    128 |   5120 |    9.237 |    55.43 |   23.691 |     5.40 |
|   512 |    128 |   5632 |    9.139 |    56.02 |   24.198 |     5.29 |
|   512 |    128 |   6144 |   10.114 |    50.62 |   26.936 |     4.75 |
|   512 |    128 |   6656 |   10.054 |    50.93 |   23.654 |     5.41 |
|   512 |    128 |   7168 |    9.958 |    51.41 |   24.267 |     5.27 |
|   512 |    128 |   7680 |   11.029 |    46.42 |   24.723 |     5.18 |
|   512 |    128 |   8192 |   10.682 |    47.93 |   24.311 |     5.27 |

## main@6fc5bbb6 DeepSeek-R1-0528-IQ1_S
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    8.530 |    60.02 |   17.123 |     7.48 |
|   512 |    128 |    512 |    8.767 |    58.40 |   20.432 |     6.26 |
|   512 |    128 |   1024 |    8.826 |    58.01 |   20.463 |     6.26 |
|   512 |    128 |   1536 |    8.964 |    57.12 |   22.866 |     5.60 |
|   512 |    128 |   2048 |    9.520 |    53.78 |   23.782 |     5.38 |
|   512 |    128 |   2560 |   10.572 |    48.43 |   22.904 |     5.59 |
|   512 |    128 |   3072 |   10.952 |    46.75 |   23.303 |     5.49 |
|   512 |    128 |   3584 |   10.747 |    47.64 |   23.772 |     5.38 |
|   512 |    128 |   4096 |   10.734 |    47.70 |   23.223 |     5.51 |
|   512 |    128 |   4608 |   11.519 |    44.45 |   23.582 |     5.43 |
|   512 |    128 |   5120 |   12.040 |    42.53 |   24.150 |     5.30 |
|   512 |    128 |   5632 |   12.694 |    40.33 |   23.282 |     5.50 |
|   512 |    128 |   6144 |   11.878 |    43.11 |   26.545 |     4.82 |
|   512 |    128 |   6656 |   12.168 |    42.08 |   24.220 |     5.28 |
|   512 |    128 |   7168 |   12.605 |    40.62 |   24.069 |     5.32 |
|   512 |    128 |   7680 |   12.843 |    39.87 |   24.390 |     5.25 |
|   512 |    128 |   8192 |   13.228 |    38.71 |   23.570 |     5.43 |


</details>

![sweep-bench-PR31](https://github.com/user-attachments/assets/98b1266a-cbfe-4794-950d-9bee98983280)

---

👤 **ikawrakow** commented on **2025-06-17** at **10:32:11**

> The IQ1_S_R4 still benches faster for this specific configuration at least and seems to be the same speed on both this PR and main as I would expect.

This is because of the extremely high total_experts/active_experts=32 ratio in DeeSeek-V3. For u_batch size of 512 we are still far away from the regime where this new repacking scheme pays large dividends. Perhaps the gains will be bigger for `u_batch = 1024` or even `u_batch = 2048`?

But yes, I see that this PR may not have the huge impact that it should because people have somehow decided that `ik_llama.cpp` is only good for very large MoE models, so they keep using `llama.cpp` for everything else, missing out big times on performance for CPU-only inference (and it isn't so that CPU performance is not discussed in the `llama.cpp` repository on a regular basis).

---

👤 **ubergarm** commented on **2025-06-17** at **16:35:26**

>  Perhaps the gains will be bigger for u_batch = 1024 or even u_batch = 2048?

Yes, looks like even with the high ratio of deepseek MoE, this new repacking scheme begins to outstrip the `_r4` variants at high enough batch sizes on this CPU only test using same xeon 6980P as above:

  ## PR531@72fd9faa DeepSeek-R1-0528-IQ1_S_R4 -b 4096 -ub 4096
  |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
  |-------|--------|--------|----------|----------|----------|----------|
  |  4096 |   1024 |      0 |   40.982 |    99.95 |  150.696 |     6.80 |
  |  4096 |   1024 |   4096 |   52.413 |    78.15 |  189.641 |     5.40 |

  ## PR531@72fd9faa DeepSeek-R1-0528-IQ1_S -b 4096 -ub 4096
  |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
  |-------|--------|--------|----------|----------|----------|----------|
  |  4096 |   1024 |      0 |   34.827 |   117.61 |  149.490 |     6.85 |
  |  4096 |   1024 |   4096 |   49.865 |    82.14 |  180.852 |     5.66 |

> missing out big times on performance for CPU-only inference

I might try quanting this qwen2.5-72b finetune [moonshotai/Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B) today. your recent improvements (and reading commit logs for `ik/iqk_gemm` on improving iq4/5_ks *even more*) will make 72B dense models much more usable for hybrid inferencing...

honestly, the biggest hurdle to general adoption of this fork, imo, is the lack of a pre-compiled distributible binary e.g. [appimage](https://appimage.org/) format etc... my guess is the *majority* of possible end-users don't know how to `apt-get install cuda-toolkit`... i've been noodling on that challenge some at least for linux users, not sure on windows/macos...

---

👤 **saood06** commented on **2025-06-17** at **16:39:44**

> > Perhaps the gains will be bigger for u_batch = 1024 or even u_batch = 2048?
> 
> Yes, looks like even with the high ratio of deepseek MoE, this new repacking scheme begins to outstrip the `_r4` variants at high enough batch sizes on this CPU only test using same xeon 6980P

I would be curious to the cutoff point. With something like `./bin/llama-bench [...] -p 32,64,128,256,512,1024,2048,3072,4096`

---

👤 **ikawrakow** commented on **2025-06-17** at **16:46:32**

> I would be curious to the cutoff point. With something like ./bin/llama-bench [...] -p 32,64,128,256,512,1024,2048,3072,4096

It is model and quantization type dependent. But I'm not removing the `_R4/_R8` quants, so everyone is free to do their performance evaluation and decide if to use this or go with the row-interleaved variant. For sure this is a big gain for people who don't want to get involved with repacking and all that stuff, but just want to run a mainline `llama.cpp` model they downloaded from HF or elsewhere. This also removes the need to worry about if the row-interleaved variant is supported on CUDA or not in case of hybrid inference.

---

👤 **saood06** commented on **2025-06-17** at **20:56:40**

>For me running Deepseek at higher contexts PP and TG both approach ~1 t/s.

I had been so used to V3 where I never enabled high batch sizes with amb because I rarely requested over the default batch size of 512. But with R1 that is not in the case (due to thought tokens removal which results in reprocessing context).

I ran an experiment at high context, processing 4096 tokens (33640 to 37736) and this went from 2950 to 1619 seconds, and even a reduction in compute buffer (`15387.76 MiB` vs `9404.80 MiB`).