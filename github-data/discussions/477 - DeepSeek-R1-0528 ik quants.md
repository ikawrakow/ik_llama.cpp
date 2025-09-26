## 🗣️ [Discussion #477](https://github.com/ikawrakow/ik_llama.cpp/discussions/477) - DeepSeek-R1-0528 ik quants!

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-05-30 |
| **Updated** | 2025-07-26 |

---

## 📄 Description

## What
Starting this "show and tell" discussion about the updated DeepSeek-R1-0528 model and various quants beginning to emerge. 

## Info

1. I just cooked up a couple `ik_llama.cpp` exclusive quants released at [ubergarm/DeepSeek-R1-0528-GGUF](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF). I am curious what other sizes might be of interest to folks e.g. a larger one for big RAM systems or maybe a very small one sacrificing quality to fit in lower RAM/VRAM systems perhaps? 
2. I'm running some benchmarks to measure the effects of quantizing attn/shexp layers while holding exps constant given the recent MLA fixes here in [PR411](https://github.com/ikawrakow/ik_llama.cpp/pull/411#issuecomment-2923060127).  Seems like mainline llama.cpp might have an issue still so folks are keeping `attn_k_b` and `attn_v_b` at `Q8_0` for those tensors.
3. Folks might have questions about offloading extra layers and multi-gpu systems which hopefully will go smoother now with [PR461](https://github.com/ikawrakow/ik_llama.cpp/pull/461) allowing repacked `_R4` quants to run on CUDA (but requires explicitly setting `-DGGML_CUDA_IQK_FORCE_BF16=1` compilation for this model).

*EDIT* Check out this [youtube video by fahdmirzac showing some examples of installing and running ik_llama.cpp with these quants here](https://www.youtube.com/watch?v=DiMZqWC7-04). Thanks Fahd!

## Benchmarks
#### Perplexity 
So far the perplexity values I've measured are as follows:

* `DeepSeek-R1-0528-Q8_0` 666GiB
  - `Final estimate: PPL = 3.2130 +/- 0.01698`
* `DeepSeek-R1-0528-IQ3_K_R4`  301GiB
  - `Final estimate: PPL = 3.2730 +/- 0.01738`
  - Fits 32k context in under 24GiB VRAM
* `DeepSeek-R1-0528-IQ2_K_R4`  220GiB
  - `Final estimate: PPL = 3.5069 +/- 0.01893`
  - Fits 32k context in under 16GiB VRAM

Compare to my previous recipes for V3-0324:

* `DeepSeek-V3-0324-Q8_0` 666GiB
  - `Final estimate: PPL = 3.2454 +/- 0.01773` 
* `DeepSeek-V3-0324-IQ4_K_R4`  387GiB
  - `Final estimate: PPL = 3.2596 +/- 0.01786`
* `DeepSeek-V3-0324-IQ2_K_R4` 227GiB
  - `Final estimate: PPL = 3.5614 +/- 0.02001`
  - Fits 32k context in under 24GiB VRAM

#### Speed
With time I hope to grab some `llama-sweep-bench` on these quants too.

## Conclusion
Thanks and let me know if you try these out or have questions or comments. Feel free to use the imatrix I uploaded as well to make your own quants. Cheers!

---

## 💬 Discussion

👤 **randoentity** commented on **2025-05-31** at **05:56:18**

Thanks for these quants and the rest of your work you publish. Could you do one that fits in 128GB RAM and 72GB VRAM with 32K context? I tried the unsloth IQ1_S and got about 2.7 t/s generation on mainline and 2.15 t/s on ik. It was coherent and delivered surprisingly good responses to real world coding tasks. Oh but the R4 variants don't support Q1 yet, right?

> 👤 **ubergarm** replied on **2025-06-01** at **17:54:28**
> 
> Yeah getting that small becomes tricky. I've been noodling on it and want to try out some experiments.. the iq2_kt quants might be interesting but will take a long time to quantize. they will get us down to 2.125 BPW but likely not performant given a lot of CPU inferencing.
> 
> I could look into the IQ1 stuff but haven't ever messed with those really... but yes there are no `_r4` repacked versions of the smaller sub ~4bpw guys yet.
> 
> If you have a good PCIe Gen5 NVMe e.g. the T705 or similar you might actually get faster going with my `IQ2_KS` which is 220GiB and using the default mmap() to let some of it "hang off" into page cache. Hoping to try that soon and expect 3-5 tok/sec on my gaming rig (96GB RAM +24GB VRAM) but it does heat up the SSD (though no write level wear as it is read only).

> 👤 **ubergarm** replied on **2025-06-02** at **04:43:27**
> 
> @randoentity 
> 
> So I'm about to upload a `IQ1_S_R4` 1.664 BPW (131GiB) that might actually fit in 128GB RAM + 24GB VRAM and has lower perplexity than Qwen3-235B-A22B-Q8_0 haha... Not sure if it is "better" though, but kind of surprising.
> 
> If you have enough RAM+VRAM to fully fit a larger model I'd recommend that over this tiny one, and you probably won't be able to run the these repacked quants on CUDA yet to take advantage of offloading extra layers. Though you can up your `-b 4096 -ub 4096` or possibly higher and use the full 160k context with all your extra VRAM.
> 
> It should be finished uploading by monday morning NYC Eastern Time.

> 👤 **randoentity** replied on **2025-06-02** at **17:18:21**
> 
> I'm only getting 0.05 TG, probably because it isn't running on CUDA. Higher batch did improve TG on mainline.

> 👤 **ubergarm** replied on **2025-06-02** at **19:45:52**
> 
> @randoentity 
> 
> > I'm only getting 0.05 TG, probably because it isn't running on CUDA.
> 
> What are you trying to do? Test out the IQ1_S_R4 quant? Provide your full command here and we can workshop it as 0.05 tok/sec TG (assuming that is what you mean?) sounds low for a 128GB RAM + 72GB VRAM system. Also provide what mix of GPUs you have e.g. a 2x 3090s and whatever.

> 👤 **ThomasBaruzier** replied on **2025-06-02** at **22:20:58**
> 
> https://github.com/ikawrakow/ik_llama.cpp/discussions/242#discussioncomment-12452986
> @randoentity I have the same setup as you and managed 7tok/s TG and 40 tok/s PP
> 
> Edit: the setup described in the link probably needs updating with all the new PRs, like mla3, but I haven't tested yet

> 👤 **randoentity** replied on **2025-06-03** at **19:00:26**
> 
> @ThomasBaruzier thanks! Unfortunately your example didn't help me. I had already tried that and other combinations.

---

👤 **Ph0rk0z** commented on **2025-06-01** at **13:19:59**

Will -rtr fix the R4 quants so they don't have to use the BF16 path?

I downloaded IQ1_S from unsloth and got 90t/s PP but same and slightly lower 10.x t/s output. So IQ2_XXS from previous V3 is not much 
different in that regard. Granted, I can use full 32k context now and maintain speeds.

Smaller AMB than 512 often lets you fit a couple more pieces due to the reduced buffer. Every little bit on GPU helps when CPU/Memory isn't that strong.

> 👤 **ubergarm** replied on **2025-06-01** at **17:57:01**
> 
> > Will -rtr fix the R4 quants so they don't have to use the BF16 path?
> 
> `-rtr` will try to make non `_r4` quants into `_r4` quants so I believe the answer is no. Though some folks are reporting `-DGGML_CUDA_IQK_FORCE_BF16=1` is giving them a slight speed *boost* probably depending on what model GPU you have.

---

👤 **ubergarm** commented on **2025-06-01** at **15:20:15**

I had an [interesting report from huggingface.co/ciprianv](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/discussions/2#683b68b8df33990a5ac0a1f7) that compiling with `-DGGML_CUDA_IQK_FORCE_BF16=1` was giving a speed *boost* on these quants which is not what I expected. 

I tried it out myself and confirmed with `llama-sweep` bench. This also is showing some small speed-ups by offloading additional layers onto GPU. I didn't have the patience to finish running one of them but you get the jist.

Interestingly it does suggest that for some hardware configurations it may be beneficial to PP to compile with `-DGGML_CUDA_IQK_FORCE_BF16=1` which surprised me given discussion in [PR[#461](https://github.com/ikawrakow/ik_llama.cpp/issues/461)](https://github.com/ikawrakow/ik_llama.cpp/pull/461#issue-3091345746)

![sweep-bench-r1-0528-bf16](https://github.com/user-attachments/assets/fb7fdd7f-f4a6-4e30-9a02-a987297fb9bb)

<details>

<summary>👈 Methodology and Data logs</summary>

Compilation flags with and without FORCE_BF16.
```bash
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1
cmake --build ./build --config Release -j $(nproc)

cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=0
```
llama-sweep-bench test
```bash
CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
  --model "$model" \
  -mla 3 -fa \
  -amb 512 \
  -fmoe \
  -ctk f16 \
  -c 16384 \
  -ngl 99 \
  -ot "blk\.(3|4|5|6|7|8|9)\.ffn_.*=CUDA0" \ # <--- with or without extra layers offloaded to GPU
  -ot exps=CPU \
  --warmup-batch \
  --no-mmap \
  --threads 24

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q5_0:   61 tensors
llama_model_loader: - type iq4_ks:  116 tensors
llama_model_loader: - type iq5_ks:  435 tensors
llama_model_loader: - type iq2_k_r4:  116 tensors
llama_model_loader: - type iq3_k_r4:   58 tensors
```

## `-DGGML_CUDA_IQK_FORCE_BF16=1 -ot "blk\.(3|4|5|6|7|8|9)\.ffn_.*=CUDA0" -ot exps=CPU`
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   10.448 |    49.00 |    8.496 |    15.07 |
|   512 |    128 |    512 |   10.626 |    48.18 |    8.548 |    14.97 |
|   512 |    128 |   1024 |   10.704 |    47.83 |    8.601 |    14.88 |
|   512 |    128 |   1536 |   10.803 |    47.39 |    9.029 |    14.18 |
|   512 |    128 |   2048 |   10.938 |    46.81 |    8.645 |    14.81 |
|   512 |    128 |   2560 |   10.983 |    46.62 |    8.789 |    14.56 |
|   512 |    128 |   3072 |   11.132 |    46.00 |    8.824 |    14.51 |
|   512 |    128 |   3584 |   11.152 |    45.91 |    8.845 |    14.47 |
|   512 |    128 |   4096 |   11.285 |    45.37 |    9.060 |    14.13 |
|   512 |    128 |   4608 |   11.432 |    44.79 |    8.842 |    14.48 |
|   512 |    128 |   5120 |   11.415 |    44.85 |    8.893 |    14.39 |
|   512 |    128 |   5632 |   11.542 |    44.36 |    9.071 |    14.11 |
|   512 |    128 |   6144 |   11.605 |    44.12 |    9.085 |    14.09 |
|   512 |    128 |   6656 |   11.719 |    43.69 |    9.258 |    13.83 |
|   512 |    128 |   7168 |   11.851 |    43.20 |    9.104 |    14.06 |
|   512 |    128 |   7680 |   11.884 |    43.08 |    9.115 |    14.04 |
|   512 |    128 |   8192 |   12.052 |    42.48 |    9.434 |    13.57 |

## `-DGGML_CUDA_IQK_FORCE_BF16=1 -ot exps=CPU`
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   11.488 |    44.57 |    8.968 |    14.27 |
|   512 |    128 |    512 |   11.665 |    43.89 |    8.923 |    14.34 |
|   512 |    128 |   1024 |   11.746 |    43.59 |    8.912 |    14.36 |
|   512 |    128 |   1536 |   11.841 |    43.24 |    9.110 |    14.05 |
|   512 |    128 |   2048 |   11.981 |    42.73 |    8.966 |    14.28 |
|   512 |    128 |   2560 |   12.023 |    42.58 |    9.144 |    14.00 |
|   512 |    128 |   3072 |   12.112 |    42.27 |    9.216 |    13.89 |
|   512 |    128 |   3584 |   12.257 |    41.77 |    9.215 |    13.89 |
|   512 |    128 |   4096 |   12.323 |    41.55 |    9.224 |    13.88 |
|   512 |    128 |   4608 |   12.452 |    41.12 |    9.191 |    13.93 |
|   512 |    128 |   5120 |   12.512 |    40.92 |    9.220 |    13.88 |
|   512 |    128 |   5632 |   12.555 |    40.78 |    9.378 |    13.65 |
|   512 |    128 |   6144 |   12.695 |    40.33 |    9.354 |    13.68 |
|   512 |    128 |   6656 |   12.822 |    39.93 |    9.480 |    13.50 |
|   512 |    128 |   7168 |   12.829 |    39.91 |    9.454 |    13.54 |
|   512 |    128 |   7680 |   12.937 |    39.58 |    9.502 |    13.47 |
|   512 |    128 |   8192 |   13.148 |    38.94 |    9.604 |    13.33 |
|   512 |    128 |   8704 |   13.142 |    38.96 |    9.626 |    13.30 |
|   512 |    128 |   9216 |   13.268 |    38.59 |    9.758 |    13.12 |
|   512 |    128 |   9728 |   13.410 |    38.18 |    9.604 |    13.33 |
|   512 |    128 |  10240 |   13.429 |    38.13 |    9.613 |    13.32 |
|   512 |    128 |  10752 |   13.522 |    37.87 |    9.856 |    12.99 |
|   512 |    128 |  11264 |   13.653 |    37.50 |    9.790 |    13.08 |
|   512 |    128 |  11776 |   13.780 |    37.15 |    9.779 |    13.09 |
|   512 |    128 |  12288 |   13.772 |    37.18 |    9.825 |    13.03 |
|   512 |    128 |  12800 |   13.886 |    36.87 |   10.041 |    12.75 |
|   512 |    128 |  13312 |   14.037 |    36.47 |    9.906 |    12.92 |
|   512 |    128 |  13824 |   14.078 |    36.37 |   10.013 |    12.78 |
|   512 |    128 |  14336 |   14.178 |    36.11 |   10.172 |    12.58 |
|   512 |    128 |  14848 |   14.289 |    35.83 |   10.043 |    12.74 |
|   512 |    128 |  15360 |   14.406 |    35.54 |    9.980 |    12.83 |
|   512 |    128 |  15872 |   14.414 |    35.52 |   10.023 |    12.77 |

## `-DGGML_CUDA_IQK_FORCE_BF16=0 -ot exps=CPU`
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   12.572 |    40.73 |    8.800 |    14.55 |
|   512 |    128 |    512 |   12.639 |    40.51 |    8.911 |    14.36 |
|   512 |    128 |   1024 |   12.810 |    39.97 |    9.140 |    14.00 |
|   512 |    128 |   1536 |   12.985 |    39.43 |    8.942 |    14.31 |
|   512 |    128 |   2048 |   12.998 |    39.39 |    9.217 |    13.89 |
|   512 |    128 |   2560 |   13.119 |    39.03 |    9.378 |    13.65 |
|   512 |    128 |   3072 |   13.247 |    38.65 |    9.137 |    14.01 |
|   512 |    128 |   3584 |   13.293 |    38.52 |    9.186 |    13.93 |
|   512 |    128 |   4096 |   13.488 |    37.96 |    9.341 |    13.70 |
|   512 |    128 |   4608 |   13.496 |    37.94 |    9.235 |    13.86 |
|   512 |    128 |   5120 |   13.522 |    37.86 |    9.405 |    13.61 |
|   512 |    128 |   5632 |   13.695 |    37.39 |    9.388 |    13.63 |
|   512 |    128 |   6144 |   13.716 |    37.33 |    9.352 |    13.69 |
|   512 |    128 |   6656 |   13.905 |    36.82 |    9.530 |    13.43 |
|   512 |    128 |   7168 |   13.911 |    36.80 |    9.413 |    13.60 |
|   512 |    128 |   7680 |   14.024 |    36.51 |    9.630 |    13.29 |
|   512 |    128 |   8192 |   14.210 |    36.03 |    9.601 |    13.33 |
|   512 |    128 |   8704 |   14.277 |    35.86 |    9.595 |    13.34 |
|   512 |    128 |   9216 |   14.361 |    35.65 |    9.571 |    13.37 |
|   512 |    128 |   9728 |   14.438 |    35.46 |    9.798 |    13.06 |
|   512 |    128 |  10240 |   14.577 |    35.12 |    9.717 |    13.17 |
|   512 |    128 |  10752 |   14.605 |    35.06 |    9.887 |    12.95 |
|   512 |    128 |  11264 |   14.683 |    34.87 |   10.044 |    12.74 |
|   512 |    128 |  11776 |   14.881 |    34.41 |    9.796 |    13.07 |
|   512 |    128 |  12288 |   14.909 |    34.34 |    9.840 |    13.01 |
|   512 |    128 |  12800 |   14.982 |    34.18 |    9.832 |    13.02 |
|   512 |    128 |  13312 |   15.094 |    33.92 |   10.101 |    12.67 |
|   512 |    128 |  13824 |   15.219 |    33.64 |   10.060 |    12.72 |
|   512 |    128 |  14336 |   15.265 |    33.54 |   10.282 |    12.45 |
|   512 |    128 |  14848 |   15.333 |    33.39 |   10.172 |    12.58 |
|   512 |    128 |  15360 |   15.493 |    33.05 |    9.979 |    12.83 |
|   512 |    128 |  15872 |   15.553 |    32.92 |    9.987 |    12.82 

</details>

> 👤 **ikawrakow** replied on **2025-06-01** at **15:30:25**
> 
> Ha, this is interesting. On my RTX-4080 `bf16` is ~10-20% slower than `fp16`.

> 👤 **ikawrakow** replied on **2025-06-01** at **15:40:55**
> 
> Btw, if you have space VRAM, try `-b 4096 -ub 4096`. This should give you a very significant boost in PP performance.

> 👤 **ubergarm** replied on **2025-06-01** at **16:27:29**
> 
> Holy Ravioli, Batman!
> 
> ![sweep-bench-r1-0528-bf16-ubatch](https://github.com/user-attachments/assets/a7471361-8803-411e-9850-70facdad469c)

> 👤 **ciprianveg** replied on **2025-06-01** at **17:06:40**
> 
> Exactly,  you can go to 6144, if vram permits,  an even further bump in pp speed..

> 👤 **Ph0rk0z** replied on **2025-06-01** at **17:53:51**
> 
> >-b 4096 -ub 4096
> 
> This gives me a bump from 90 to 127 but the buffer sizes mean I have to offload less layers. Offloading the wrong things can cause PCIE related gpu bottleneck too.

> 👤 **RodriMora** replied on **2025-06-02** at **09:15:30**
> 
> results with and without -b 4096 -ub 4096
> 
> ![image](https://github.com/user-attachments/assets/73561ba8-f858-4502-8f5b-bccb7d64b07f)
> 
> I can offload a few more layers without -b 4096 -ub 4096 giving a bit better TG
> 
> ![image](https://github.com/user-attachments/assets/dd4c34b6-e654-427d-aaed-58fa19585e00)
> 
> <details>
> <summary>llama-sweep-bench command with defaults -b and -ub and a bit more layers</summary>
> 
> 
> ```
> CUDA_VISIBLE_DEVICES="2,4,0,1,3,5" \
>                                              ./build/bin/llama-sweep-bench \
>                                                --model /home/ubuntuai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ2_K_R4/DeepSeek-R1-0528-IQ2_K_R4-00001-of-00005.gguf \
>                                                --alias ubergarm/DeepSeek-R1-0528-IQ2_K_R4 -mla 3 -fa \
>                                                -amb 512 \
>                                                -fmoe \
>                                                -ctk f16 \
>                                                -c 16384 \
>                                                -ngl 99 \
>                                                -ot "blk\.(3|4|5|6|7|8|9)\.ffn_.*=CUDA0" \
>                                                -ot "blk\.(10|11|12|13|14|15|16)\.ffn_.*=CUDA1" \
>                                                -ot "blk\.(17|18|19|20|21)\.ffn_.*=CUDA2" \
>                                                -ot "blk\.(22|23|24|25|26)\.ffn_.*=CUDA3" \
>                                                -ot "blk\.(27|28|29|30|31)\.ffn_.*=CUDA4" \
>                                                -ot "blk\.(32|33|34|35|36)\.ffn_.*=CUDA5" \
>                                                -ot exps=CPU \
>                                                --warmup-batch \
>                                                --no-mmap \
>                                                --threads 24
> ```
> 
> 
> </details>
> 
> <details>
> <summary>llama-sweep-bench command with -b 4096 -ub 4096 but less layers into vram</summary>
> 
> 
> ```
> CUDA_VISIBLE_DEVICES="2,4,0,1,3,5" \
>                                              ./build/bin/llama-sweep-bench \
>                                                --model /home/ubuntuai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ2_K_R4/DeepSeek-R1-0528-IQ2_K_R4-00001-of-00005.gguf \
>                                                --alias ubergarm/DeepSeek-R1-0528-IQ2_K_R4 -mla 3 -fa \
>                                                -amb 512 \
>                                                -fmoe \
>                                                -ctk f16 \
>                                                -c 16384 \
>                                                -ngl 99 \
>                                                -ot "blk\.(3|4|5|6|7|8)\.ffn_.*=CUDA0" \
>                                                -ot "blk\.(9|10|11|12|13|14)\.ffn_.*=CUDA1" \
>                                                -ot "blk\.(15|16|17|18)\.ffn_.*=CUDA2" \
>                                                -ot "blk\.(20|21|22|23)\.ffn_.*=CUDA3" \
>                                                -ot "blk\.(25|26|27|28)\.ffn_.*=CUDA4" \
>                                                -ot "blk\.(30|31|32|33)\.ffn_.*=CUDA5" \
>                                                -ot exps=CPU \
>                                                -b 4096 -ub 4096 \
>                                                --warmup-batch \
>                                                --no-mmap \
>                                                --threads 24
> ```
> 
> 
> </details>
> 
> 
> compiled with:
> pulled this commit 7a8abe29f745cff95896095bf19cf247bdf2c661
> ```
> rm -rf build
> cmake -B build -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1
> cmake --build build --config Release -j$(nproc)
> ```

> 👤 **cmoncure** replied on **2025-06-02** at **14:05:02**
> 
> > Offloading the wrong things can cause PCIE related gpu bottleneck too.
> 
> Tell me more.  Isn't -ot just a static offload of tensors, and if you put too many, the process blows up when it runs out of VRAM? Where does PCI-E come into play?

> 👤 **Ph0rk0z** replied on **2025-06-02** at **15:23:34**
> 
> If you split a layer across cards you can have a situation where GPU usage is high and they transfer a lot of data back and forth. Like place a gate on one and down on another. The CPU usage then craters to half or less and your overall speed is cooked. Especially evident for RTR. Remember a forward pass goes through these weights and I think passes states along.

> 👤 **ubergarm** replied on **2025-06-03** at **20:20:29**
> 
> @RodriMora 
> 
> Thanks for the graphs. I thought I recognized that combination of GPUs from reddit lmao... Cheers at stitching together a sweet vibe coding rig haha

---

👤 **anikifoss** commented on **2025-06-01** at **16:12:44**

I uploaded the custom quant I use for coding [here](https://huggingface.co/anikifoss/DeepSeek-R1-0528-DQ4_K_R4) with some of the infromation how I arrived there and relevant benchmarks. I added some teasers on command line arguments to experiment with, as this branch is moving quickly and small performance improvements can add up over time.

> 👤 **ubergarm** replied on **2025-06-04** at **21:08:29**
> 
> Thanks again for your quant, pretty sure it is the biggest boi of them all so a great choice for anyone with a big rig that wants the more BPW than my quants!

---

👤 **ubergarm** commented on **2025-06-01** at **18:28:15**

Qantization Effects of `attn`/`shexp` on Perplexity
===

## Motivation
> I would be curious to see how much degradation in quality there is from using 6- or 5-bit quants for the attention tensors and shared experts. @ikawrakow

This research grew out of [PR[#411](https://github.com/ikawrakow/ik_llama.cpp/issues/411) discussions](https://github.com/ikawrakow/ik_llama.cpp/pull/411#issuecomment-2922464774). I've expanded on ik's example bash script to create 10 test quants each about \~355GiB in size. All the quants hold constant `q4_0` for `ffn.*` and `token_embd` while varying `attn.*` and `shexp` using all quants between 4~6bpw.

If anyone wants to publish this, just hit me up and just cite myself and the project here appropriately.

*EDIT* Added the new `iq4_kt` trellis quant to graph and data!
## Results
![trellis-iq2_kt-ppl-r1-0528](https://github.com/user-attachments/assets/59761cb2-e057-4d55-920f-e7400b6539b0)

## Methodology and Data

I chose the Y-Axis scale based on [some discussion here](https://github.com/ikawrakow/ik_llama.cpp/pull/461#issuecomment-2927455318). The actual reported Final PPL values are in the annotation and for scale perspective the "worst case" q4_0 is only about 1.5% higher PPL than the q8_0.

You can check the scripts below for exact quantization strategies, and do note that I left `attn_k_b` the closest sized `qN_0` quant due to size restrictions preventing using `iqN_k` etc.

<details>

<summary>👈 Scripts and Logs</summary>

#### quantization script
```bash
#!/usr/bin/env bash

model=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-256x21B-0528-BF16-00001-of-00030.gguf
imatrix=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/imatrix-DeepSeek-R1-0528.dat
outdir=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF
basename=DeepSeek-R1-0528
base_q=q4_0

# iterate over list of tuples as attn_k_b shape requires qN_0 types
for q in q8_0,q8_0 q6_0,q6_K q6_0,iq6_k q5_0,q5_K q5_0,iq5_k q5_0,iq5_ks q4_0,q4_K q4_0,iq4_k q4_0,iq4_ks q4_0,q4_0
do
    # unpack tuples into $1,$2
    IFS=","
    set -- $q

    # quantize using $1 for attn_k_b and $2 for rest of attn and base_q for all else
    numactl --interleave=all \
    ./build/bin/llama-quantize \
        --imatrix $imatrix \
        --custom-q attn_k_b=$1 \
        --custom-q attn=$2 \
        --custom-q shexp=$2 \
        --custom-q exps=$base_q \
        $model \
        $outdir/$basename-$base_q-attn-shexp-$2.gguf \
        $base_q \
        2>&1 | tee -a logs/quantize-$basename-$base_q-attn-shexp-$2.log
done
```

#### resultant test quants
```
$ du -h /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/*q4_0*
353G    /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-q4_0-attn-shexp-iq4_k.gguf
353G    /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-q4_0-attn-shexp-iq4_ks.gguf
355G    /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-q4_0-attn-shexp-iq5_k.gguf
355G    /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-q4_0-attn-shexp-iq5_ks.gguf
357G    /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-q4_0-attn-shexp-iq6_k.gguf
353G    /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-q4_0-attn-shexp-q4_0.gguf
353G    /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-q4_0-attn-shexp-q4_K.gguf
355G    /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-q4_0-attn-shexp-q5_K.gguf
357G    /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-q4_0-attn-shexp-q6_K.gguf
360G    /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-q4_0-attn-shexp-q8_0.gguf
```

#### perplexity test script
```
#!/usr/bin/env bash

for model in $(ls /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/*q4_0*.gguf); do
    logfile=logs/perplexity-$(basename "${model%.*}").log

    numactl -N 0,1,2 --interleave=0,1,2 \
    ./build/bin/llama-perplexity \
        --model "$model" \
        -mla 3 -fa \
        -amb 512 \
        -rtr \
        -fmoe \
        -f wiki.test.raw \
        --seed 1337 \
        --threads 128 \
        --numa numactl \
        2>&1 | tee -a $logfile
done
```

## raw data in JSON format
```json
[
  {
    "name": "q4_0",
    "ppl": "3.2895 +/- 0.01755",
    "size": 352.656,
    "bpw": 4.508,
    "legend": "test"
  },
  {
    "name": "q4_K",
    "ppl": "3.2688 +/- 0.01739",
    "size": 352.656,
    "bpw": 4.508,
    "legend": "test"
  },
  {
    "name": "iq4_k",
    "ppl": "3.2713 +/- 0.01742",
    "size": 352.656,
    "bpw": 4.508,
    "legend": "test"
  },
  {
    "name": "iq4_ks",
    "ppl": "3.2676 +/- 0.01736",
    "size": 352.255,
    "bpw": 4.502,
    "legend": "test"
  },
  {
    "name": "iq4_kt",
    "ppl": "3.2832 +/- 0.01749",
    "size": 351.855,
    "bpw": 4.497,
    "legend": "test"
  },
  {
    "name": "q5_K",
    "ppl": "3.2565 +/- 0.01729",
    "size": 354.401,
    "bpw": 4.530,
    "legend": "test"
  },
  {
    "name": "iq5_k",
    "ppl": "3.2555 +/- 0.01729",
    "size": 354.401,
    "bpw": 4.530,
    "legend": "test"
  },
  {
    "name": "iq5_ks",
    "ppl": "3.2541 +/- 0.01726",
    "size": 354.001,
    "bpw": 4.525,
    "legend": "test"
  },
  {
    "name": "q6_K",
    "ppl": "3.2553 +/- 0.01732",
    "size": 356.251,
    "bpw": 4.553,
    "legend": "test"
  },
  {
    "name": "iq6_k",
    "ppl": "3.2577 +/- 0.01729",
    "size": 356.357,
    "bpw": 4.555,
    "legend": "test"
  },
  {
    "name": "q8_0",
    "ppl": "3.2485 +/- 0.01722",
    "size": 359.636,
    "bpw": 4.597,
    "legend": "test"
  }
]
```

#### python script for plotting
I vibe coded this using my R1-0528-IQ2_K_R4 and it loads the JSON I manually created as a file. Hopefully it didn't hallucinate anything haha...

```python
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from adjustText import adjust_text
import numpy as np
from matplotlib.lines import Line2D

# Read JSON data from file
with open('ppl-r1-0528.json', 'r') as f:
    data = json.load(f)

# Filter out incomplete entries and extract mean perplexity and error
filtered_data = []
for entry in data:
    if 'ppl' in entry and 'size' in entry and 'bpw' in entry and 'legend' in entry:
        # Parse perplexity string to get mean and error
        ppl_parts = entry['ppl'].split()
        mean_ppl = float(ppl_parts[0])
        error = float(ppl_parts[2])  # The value after "+/-"

        filtered_data.append({
            'name': entry['name'],
            'mean_ppl': mean_ppl,
            'error': error,
            'size': float(entry['size']),
            'bpw': float(entry['bpw']),
            'legend': entry['legend']
        })

# Sort by size (smallest to largest)
sorted_data = sorted(filtered_data, key=lambda x: x['size'])

# Prepare plot data
names = [d['name'] for d in sorted_data]
sizes = [d['size'] for d in sorted_data]
ppls = [d['mean_ppl'] for d in sorted_data]
errors = [d['error'] for d in sorted_data]
bpws = [d['bpw'] for d in sorted_data]
legends = [d['legend'] for d in sorted_data]

# Find minimum perplexity (best model)
min_ppl = min(ppls)

# Calculate ln(PPL/min_ppl) for each point
ln_ratios = [np.log(p / min_ppl) for p in ppls]
# Calculate error for ln ratio: d(ln(p)) = dp/p
ln_ratio_errors = [e / p for e, p in zip(errors, ppls)]

# Create annotation labels (show original perplexity values)
labels = [
    f"{name}\nppl: {ppl:.4f}\nbpw: {bpw:.3f}"
    for name, ppl, bpw in zip(names, ppls, bpws)
]

# Apply solarized style
plt.style.use('Solarize_Light2')

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Set Y-axis limits for ln ratio
ax.set_ylim(0, 0.015)  # Adjusted for ln(PPL/min) scale
ax.set_xlim(min(sizes)-0.5, max(sizes)+0.5)

# Set labels
ax.set_xlabel('Model Size (GiB)', fontsize=12)
ax.set_ylabel('ln(PPL / min(PPL)) wiki.test.raw', fontsize=12)  # Updated Y-axis label

# Set title and subtitle with increased padding
main_title = "DeepSeek-R1-0528 ik_llama.cpp"
subtitle = "Varying attn/shexp with fixed Q4_0 exps/token_embd"

ax.set_title(main_title, fontsize=16, pad=40)
ax.text(0.5, 1.05, subtitle, transform=ax.transAxes,
        ha='center', fontsize=13, style='italic', color='#586e75')

# Add grid
ax.grid(True, linestyle='--', alpha=0.7)

# Plot dotted connecting line
ax.plot(sizes, ln_ratios, ':', color='#586e75', linewidth=1.5, zorder=1)

# Define unique markers and color map for legend groups
markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', 'X', 'd', 'P', '>']
unique_legends = sorted(set(legends))  # Sort for consistent ordering
colors = plt.cm.Set2(np.linspace(0, 1, len(unique_legends)))

# Create mapping from legend to color and marker
legend_color_map = {legend: colors[i] for i, legend in enumerate(unique_legends)}
legend_marker_map = {legend: markers[i % len(markers)] for i, legend in enumerate(unique_legends)}

# Plot each point with error bars, using group-specific color and marker
for i, (size, ln_ratio, ln_error, legend) in enumerate(zip(sizes, ln_ratios, ln_ratio_errors, legends)):
    # Get color and marker for this legend group
    color = legend_color_map[legend]
    marker = legend_marker_map[legend]

    # Add error bar
    ax.errorbar(
        size,
        ln_ratio,
        yerr=ln_error,
        fmt='none',  # Don't plot main line
        ecolor=color,
        elinewidth=1.5,
        capsize=4,
        alpha=0.7,
        zorder=2
    )

    # Add scatter point with marker based on legend
    ax.scatter(
        size,
        ln_ratio,
        marker=marker,
        color=color,
        s=100,
        edgecolor='#586e75',  # Solarized base01 for outline
        linewidth=0.8,
        zorder=3
    )

# Create text annotations without boxes
texts = []
for size, ln_ratio, label in zip(sizes, ln_ratios, labels):
    texts.append(
        plt.text(
            size,
            ln_ratio,
            label,
            fontsize=9,
            ha='center',
            va='bottom',
            zorder=4
        )
    )

# Adjust text positions to avoid overlaps
adjust_text(
    texts,
    x=sizes,
    y=ln_ratios,
    arrowprops=dict(
        arrowstyle='->',
        color='#586e75',  # Solarized base01
        alpha=0.7,
        linewidth=1.0
    ),
    expand=(1.2, 1.8),
    ensure_inside_axes=True,
    min_arrow_len=0.15,
    prevent_crossings=False,
    only_move={'points': 'xy', 'text': 'xy'},
    max_move=150
)

# Add horizontal line at 0 for reference (ln(1)=0)
ax.axhline(y=0, color='#93a1a1', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)

# Create custom legend for legend groups with group-specific colors
legend_handles = [
    Line2D([0], [0],
           marker=legend_marker_map[legend],
           color=legend_color_map[legend],
           markersize=10,
           label=legend,
           linewidth=0,
           markeredgecolor='gray')
    for legend in unique_legends
]

# Add legend to plot
ax.legend(
    handles=legend_handles,
    title='Legend Groups',
    loc='upper right',
    framealpha=0.9
)

# Save figure
out_filename = 'ppl-r1-0528.png'
plt.tight_layout()
plt.savefig(out_filename, dpi=150, bbox_inches='tight')
print(f"Plot saved to {out_filename}")
print(f"Reference: Minimum perplexity = {min_ppl:.4f} (q8_0 model)")
```

</details>

## Conclusion

My personal observations and thoughts are:
1. Even the Q4_0 is only about 1.5% worse than full Q8_0 attn/shexp. So not sacrificing a ton for likely faster TG speeds.
2. I was surprised that the iq6_k was slightly "worse" than the q6_K
3. The 32 block size [_ks](https://github.com/ikawrakow/ik_llama.cpp/pull/83#issue-2575352790) quants are looking really strong here especially given recent CUDA speed-ups. I'm eyeing that `iq5_ks` for future recipes and glad I already used them my released `IQ2_K_R4`
4. The error bars crack me up.

> 👤 **ubergarm** replied on **2025-06-02** at **04:46:51**
> 
> ![perplexity](https://github.com/user-attachments/assets/55a55312-b41b-49c5-86cb-922d82b62190)
> 
> Just ran some perplexity numbers for all of the quants I've released to huggingface. Running a few KLD on a very short "novel" test corpus also mainly to compare against quants from other cookers using different imatrix test corpus and methodologies and confirm if the PPL compares between us all okay or what.
> 
> Interestingly the small `IQ1_S_R4` has a perplexity lower than `Qwen3-235B-A22B-Q8_0`=`Final estimate: PPL = 5.3141 +/- 0.03321` 232.769 GiB though that doesn't necessarily mean it is "better" but possibly more trained against wiki.test.raw?

> 👤 **ikawrakow** replied on **2025-06-02** at **05:36:13**
> 
> So, `iq5_ks` looks like the winning option for attention tensors.
> 
> Concerning `IQ1_S` lower PPL: these are two different models, so PPL cannot be used to compare. PPL is useful for measuring quality degradation with different quantization types applied to the **same model**. My guess is that the PPL difference between `f16` (or `Q8_0`) Qwen3-235B-A22B and DeepSeek-R1 is quite large.

> 👤 **ubergarm** replied on **2025-06-02** at **14:14:22**
> 
> > So, iq5_ks looks like the winning option for attention tensors.
> 
> Yes, just for fun I ran a very short kld test corpus against them as well. The graph is kind of gnarly but is attempting to show `RMS Δp`, `99.0%   Δp`, and `Maximum Δp` percentage for each of the experimental attn/shexp quants. Seems to still point towards `iq5_ks` as it has a surprisingly tight Δp relative the to pure q8_0 everything ~666GiB baseline.
> 
> *EDIT* Added the new iq4_kt trellis quant to graph and data!
> 
> ![trellis-iq2_kt-kld-r1-0528](https://github.com/user-attachments/assets/8b4a863b-1084-49d6-8b25-a0fa700323ce)
> 
> Each experimental quant has 3x data points plotted in a vertical line. It isn't super clear but here is the JSON data if anyone wants to slice and dice it further.
> 
> <details>
> 
> <summary>👈 JSON datafile</summary>
> 
> ```json
> [
>   {
>     "name": "q4_0",
>     "ppl": "3.2895 +/- 0.01755",
>     "size": 352.656,
>     "bpw": 4.508,
>     "legend": "baseline",
>     "dp_max": 31.887,
>     "dp_99": 10.354,
>     "dp_rms": "3.775 +/- 0.062"
>   },
>   {
>     "name": "q4_K",
>     "ppl": "3.2688 +/- 0.01739",
>     "size": 352.656,
>     "bpw": 4.508,
>     "legend": "test",
>     "dp_max": 29.435,
>     "dp_99": 9.642,
>     "dp_rms": "3.347 +/- 0.062"
>   },
>   {
>     "name": "iq4_k",
>     "ppl": "3.2713 +/- 0.01742",
>     "size": 352.656,
>     "bpw": 4.508,
>     "legend": "test",
>     "dp_max": 24.338,
>     "dp_99": 9.274,
>     "dp_rms": "3.067 +/- 0.051"
>   },
>   {
>     "name": "iq4_ks",
>     "ppl": "3.2676 +/- 0.01736",
>     "size": 352.255,
>     "bpw": 4.502,
>     "legend": "test",
>     "dp_max": 41.175,
>     "dp_99": 9.538,
>     "dp_rms": "3.259 +/- 0.061"
>   },
>   {
>     "name": "iq4_kt",
>     "ppl": "3.2832 +/- 0.01749",
>     "size": 351.855,
>     "bpw": 4.497,
>     "legend": "test",
>     "dp_max": 46.908,
>     "dp_99": 9.005,
>     "dp_rms": "3.221 +/- 0.073"
>   },
>   {
>     "name": "q5_K",
>     "ppl": "3.2565 +/- 0.01729",
>     "size": 354.401,
>     "bpw": 4.530,
>     "legend": "test",
>     "dp_max": 25.725,
>     "dp_99": 8.523,
>     "dp_rms": "2.859 +/- 0.051"
>   },
>   {
>     "name": "iq5_k",
>     "ppl": "3.2555 +/- 0.01729",
>     "size": 354.401,
>     "bpw": 4.530,
>     "legend": "test",
>     "dp_max": 28.849,
>     "dp_99": 8.484,
>     "dp_rms": "2.772 +/- 0.055"
>   },
>   {
>     "name": "iq5_ks",
>     "ppl": "3.2541 +/- 0.01726",
>     "size": 354.001,
>     "bpw": 4.525,
>     "legend": "test",
>     "dp_max": 22.856,
>     "dp_99": 8.026,
>     "dp_rms": "2.780 +/- 0.052"
>   },
>   {
>     "name": "q6_K",
>     "ppl": "3.2553 +/- 0.01732",
>     "size": 356.251,
>     "bpw": 4.553,
>     "legend": "test",
>     "dp_max": 42.780,
>     "dp_99": 8.358,
>     "dp_rms": "2.707 +/- 0.060"
>   },
>   {
>     "name": "iq6_k",
>     "ppl": "3.2577 +/- 0.01729",
>     "size": 356.357,
>     "bpw": 4.555,
>     "legend": "test",
>     "dp_max": 31.809,
>     "dp_99": 8.842,
>     "dp_rms": "2.854 +/- 0.055"
>   },
>   {
>     "name": "q8_0",
>     "ppl": "3.2485 +/- 0.01722",
>     "size": 359.636,
>     "bpw": 4.597,
>     "legend": "test",
>     "dp_max": 26.032,
>     "dp_99": 6.632,
>     "dp_rms": "2.236 +/- 0.053"
>   }]
> ```
> 
> </details>
> 
> > PPL is useful for measuring quality degradation with different quantization types applied to the same model. 
> 
> Thanks, that makes sense. I'm wondering if it is okay to use PPL to measure relative quality of the same model quantized with different imatrix corpus / methodologies? I don't know how much stock to put into my PPL comparisons of R1-0528 quants done by myself, unsloth, bartowski, given somewhat varying imatrix methodologies.

> 👤 **saood06** replied on **2025-06-04** at **04:32:52**
> 
> > Yes, just for fun I ran a very short kld test corpus against them as well. The graph is kind of gnarly but is attempting to show `RMS Δp`, `99.0% Δp`, and `Maximum Δp` percentage for each of the experimental attn/shexp quants. Seems to still point towards `iq5_ks` as it has a surprisingly tight Δp relative the to pure q8_0 everything ~666GiB baseline.
> 
> If you find it fun/interesting can you see what quants you have pass the maze test. As mentioned here https://github.com/ikawrakow/ik_llama.cpp/issues/383#issuecomment-2882600098, I found it quite interesting the difference in pass rate between IQ4_K_R4 and IQ4_KS_R4.
> 
> If you don't find it fun/interesting then don't bother.

> 👤 **randoentity** replied on **2025-06-04** at **20:24:31**
> 
> I tried one pass and the IQ1_S succeeded, but it took 19 minutes of thinking (at 4.7 t/s).
> 
> Edit: 3/3 so far, quasi-random maze (I skipped ones that required fewer than 3 steps).

---

👤 **Ph0rk0z** commented on **2025-06-02** at **11:12:38**

So here is a new surprise, since I'm eying that IQ1 quant you're publishing. On a lark I turned off the -rtr switch and in unsloth's quant, it was cutting my prompt processing by half. It did buff textgen to over 11t/s though. The mind wobbles. Will try reloading the larger quant of V3 to check results. On Qwens it sped things up 100%

On another note, I tried to test mainline llama and that sweep bench segfaults with deepseek and does not recognize the -FA parameter. I was able to load on llama-server and get a blazing fast 6t/s PP, 6t/s TG. So much for that.

> 👤 **ubergarm** replied on **2025-06-04** at **21:11:14**
> 
> Check out this [PR492](https://github.com/ikawrakow/ik_llama.cpp/pull/492), given one cannot simply repack IQ1_S to IQ1_S_R4 is possibly related to the mind wobbles. haha..

---

👤 **cmoncure** commented on **2025-06-03** at **00:58:43**

Still struggling to understand some things.

✔ All tensors on CPU
✔ `exps=CPU, -ngl 99 -ot attn=GPU0 -sm none`
✔ `exps=CPU, -ngl 99 attn=GPU0, blk.3.ffn_.*=GPU0 -sm none`
✔ `exps=CPU, -ngl 8 -sm layer`
✘ `exps=CPU, blk.3.ffn_.*=GPU0, blk.4.ffn_.*=GPU1 -sm none` illegal memory access
✘ `exps=CPU, blk.3.ffn_.*=GPU0, blk.4.ffn_.*=GPU1 -sm layer` tries to allocate 1.5 TB of VRAM
✘ `--run-time-repack -sm layer` OOM killed??

With Q4_K_M `-rtr -sm none -ot attn=GPU0` I get 80-90 PP and 14-16 TG.  CUDA0 ~25% utilization during PP, 43% during TG.

With Q4_K_M `-ngl 8 -sm layer -b 4096` it's 180-200 PP but less ideal 6-8 TG.  CUDA0 100% utilization and CUDA1 <10% utilization with just a tiny blip of activity every batch. I guess the contribution of CUDA1 here is nominal.

(IQ4_K_R4 `-ngl 8 -sm layer -b 4096` performance is not "tokens per second" but "seconds per token")

Either way I have a whole GPU worth of compute just sitting idle.  There has to be a way to utilize it. Can I not have the `-ngl 8 -sm layer` approach during PP on CUDA0, and then the `-rtr -sm none` approach during TG on CUDA1? Can I produce a quant that gets me the best of both worlds?

> 👤 **Ph0rk0z** replied on **2025-06-03** at **02:39:39**
> 
> Trial and error :( Helps to print the sizes on mainline and then see what you can fit. Generally on deepseek, only EXP layers help. All the little small ones don't do much.

> 👤 **cmoncure** replied on **2025-06-03** at **15:24:54**
> 
> Why can I seemingly split any combination of tensors between CPU and GPU0, but as soon as I try putting one tensor on to GPU1 this is suddenly impossible?

> 👤 **ubergarm** replied on **2025-06-03** at **16:19:17**
> 
> Its hard for me to understand what you're doing without a full command. A few quick thoughts:
> 1. Order matters, always put `-ot exps=CPU` *last* and any kind of offload to CUDA0 *before* it.
> 2. What is `GPU0`? Does that work? I've only used `CUDA0` but maybe you have non nvidia? i dunno...
> 3. Only ever use `-ot` with `-ngl 99` (or any big number >= number of layers). I have never ever used `-sm` just leave it default.
> 
> While there are a ton of knobs, no need to go wild with all of them. Look at the example commands people are showing e.g. on [this model card](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF#ik_llamacpp-api-server-for-multigpucpu) and start small and work your way up. To be fair I'm still confused about some things e.g. other people are not using `-ts 24,24` and seem to be fine. I think maybe I shouldn't use `-ts` as it seems to become unbalanced combining `-ts` and `-ot` with multi-gpu...
> 
> Anyway, you'll get there! Start off using only a *single* GPU and get that dialed in okay, then add a second. Also make sure to compile with `-DGGML_SCHED_MAX_COPIES=1` for multi GPU as it becomes less confusing.
> 
> I usually use this now when running R1-0528 and f16 for the new iq4_kt type trellis quants which are not quite prime time yet.
> ```
> cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_CUDA_IQK_FORCE_BF16=1 -DGGML_CUDA_F16=ON -DGGML_SCHED_MAX_COPIES=1
> cmake --build build --config Release -j $(nproc)
> ```
> 
> Have fun!

> 👤 **Ph0rk0z** replied on **2025-06-03** at **17:22:42**
> 
> SM row and layer is the pseudo tensor parallel switch, mainly for GPU inference only. If we had real TP I bet our t/s go up by a third. Does TS even do anything here when you curate what layers to offload? 
> 
> I could put NGL 3 (maybe not that low, it segfaults) and just OT the layers I want to GPU. NGL only seems to stuff some unmentioned piddly layers on there and determine if pipeline parallel enables or not which affects the buffer size.
> 
> Having high GPU utilization among multiple GPUs is actually *bad*. Means lots of transfers are happening. You really can bottleneck yourself. All it takes is nvtop and the sweep bench to see.
> 
> Super easy to get started, you just rack ffn or ffn_exp onto each GPU until it reaches a point where it doesn't OOM after the buffer is added. Can lower the buffer with AMB or smaller batch/ubatch. Ideally you have 4096, 2048, 1024 batches for context and then lower that to gain more t/s. It really is a balance of what you want.
> 
> Likely with Q4KM the layers are large too. Going to have to pick and choose. Sincerely hope that only 2 layers aren't fitting because that's nothing.

> 👤 **randoentity** replied on **2025-06-03** at **18:58:13**
> 
> I've tried the example commands and a ton of combinations, but I can't get the IQ1_ik generate faster than the unsloth IQ1_S. The fastest I can get is about 2.8 t/s and that's with **only** `--override-tensor exps=CPU,attn_kv_b=CPU`. As soon as I add more ffn layers (as per example) to CUDA (4@16x) it slows down. I've played with batch sizes, fa+ctv, bf16 enabled or not (it is a bit faster with it on!), and also the unsloth -ot examples. I (again) must have missed something obvious, like ik_llama.cpp requiring AVX512 or more than 6 cores.

> 👤 **Thireus** replied on **2025-06-03** at **19:05:09**
> 
> > I've tried the example commands and a ton of combinations, but I can't get the IQ1_ik generate faster than the unsloth IQ1_S. The fastest I can get is about 2.8 t/s and that's with **only** `--override-tensor exps=CPU,attn_kv_b=CPU`. As soon as I add more ffn layers (as per example) to CUDA (4@16x) it slows down. I've played with batch sizes, fa+ctv, bf16 enabled or not (it is a bit faster with it on!), and also the unsloth -ot examples. I (again) must have missed something obvious, like ik_llama.cpp requiring AVX512 or more than 6 cores.
> 
> I'm observing the same behaviour and I'm suspecting it has to do with memory/pcie bandwidth being saturated. Which CPU are you using?

> 👤 **ubergarm** replied on **2025-06-03** at **20:04:14**
> 
> Heya all, I have another thread going to help people specifically related to my smol boi 131GiB `IQ1_S_R4` ik_llama.cpp quant with some more example commands and discussion here: https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/discussions/6#683e4c6ede3f6dd9c43ad4ad
> 
> If you want some help always give your CPU, RAM size, and list GPUs with VRAM each/total as well as the *full* current command you're trying. That will help me diagnose and optimize your command.
> 
> If you have only 128GB RAM, its still not clear to me that people will be able to fit the whole 131GiB weights into RAM+VRAM without a really tight squeeze, possibly only headless and definitely offloading some layers to GPU.
> 
> So you have to make sure that the *entire* model is loaded and none is let mmap()'ing off of your disk drive. You can check with `btop` while inferencing and should not see a constant 1GiB/s (or more) disk i/o.
> 
> @randoentity 
> > --override-tensor exps=CPU,attn_kv_b=CPU
> 
> I'm not sure why you'd ever override kv_b to CPU as it is very small and best left on GPU? So not sure where you found that.
> 
> Also be sure to override the layers to CUDA0 and CUDA1 etc *before* you put that final `-ot exps=CPU` as order matters.
> 
> @Thireus 
> 
> Oh hello from reddit! I'm voidalchemy haha... Hopefully we can get a good command ironed out as folks start learning ik_llama.cpp which can be a little different than mainline llama.cpp.
> 
> So yeah post your current command and system specs and hopefully can get you a few more tok/sec.

> 👤 **randoentity** replied on **2025-06-03** at **20:27:15**
> 
> ```sh
>      ./build_bf16/bin/llama-server \
>         --model /mnt/x/models/ubergarm/DeepSeek-R1-0528-     IQ1_S_R4-00001-of-00003.gguf \
>         -ctk q8_0 \
>         --flash-attn \                                                    --ubatch-size 2048 --batch-size 2048 \
>         --alias reason  \
>         -mla 3 \                                                          -amb 512 \
>         -ot ".ffn_(up|down)_exps.=CPU" \
>         --parallel 1 \
>         --temp 0.6 \
>         --top_p 0.95 \                                                    --min_p 0.01 \
>         --n-gpu-layers 99 \                                               -c 2048
> ```
> 
> A ton of variations of that one. Also with -fmoe. Unsloth (172GiB) runs at 3.8 t/s, this one runs at 1.2 t/s.
> 
> Isn't the problem just that IQ1_R4 isn't implemented (https://github.com/ikawrakow/ik_llama.cpp/pull/461)? Because the more I offload to CUDA the slower it gets. I.e. `-ot exps=CPU` alone is faster than adding more ffn blocks to CUDA (also tested single or multiple devices; same result).
> 
> The kv_b example I got from https://huggingface.co/anikifoss/DeepSeek-R1-0528-DQ4_K_R4 (see above). I just added it to show that I've tried a ton of things.
> 
> I do use a headless system and I don't have any swap allocated. The 172GiB one fits just fine and I can run it with --no-mmap.

> 👤 **Thireus** replied on **2025-06-03** at **20:33:12**
> 
> 👋 @ubergarm - thank you for all your posts, I've been digging them all and tried various combinations with ik_llama.cpp on Windows.
> 
> I kept note of my progress (but not of everything I've tried) here: https://thireus.com/GITHUB/ik_llama_Thireus_bench_01.txt (Firefox: View -> Repair Text Encoding), please let me know if you have suggestions that might help.
> 
> I am running out of ideas and I suspect my CPU/RAM is the limiting factor here. I've also seen your graph on https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/146 and wish I had the same results with some full layers loaded to the GPU, but sadly it doesn't improve my results instead it makes things much worse as @randoentity pointed out.
> 
> Hardware:
> 
> > i9-7980XE - 4.2Ghz on all cores <-- As someone else pointed out on Reddit, 85 GB/s is my max memory bandwidth
> > 256GB DDR4 F4-3200C14Q2-256GTRS - XMP enabled
> > 1x 5090 (x16)
> > 1x 3090 (x16)
> > 1x 3090 (x8)
> > Prime-X299-A-II
> 
> The windows build I'm using: https://github.com/Thireus/ik_llama.cpp/releases
> 
> Using CUDA 12.8 (and Blackwell compatible) + -DGGML_AVX512=ON -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1
> See https://github.com/Thireus/ik_llama.cpp/blob/main/.github/workflows/release.yml#L448-L450

> 👤 **Thireus** replied on **2025-06-03** at **21:06:54**
> 
> @cmoncure 
> 
> > Why can I seemingly split any combination of tensors between CPU and GPU0, but as soon as I try putting one tensor on to GPU1 this is suddenly impossible?
> 
> That happened to me when not using `--flash-attn` or `-mla 3`.

> 👤 **anikifoss** replied on **2025-06-03** at **22:51:59**
> 
> The `attn_kv_b=CPU` flag can save up to 1GB VRAM without losing any speed, which is huge when you're trying to squeeze more context out of a 24GB card!

> 👤 **ubergarm** replied on **2025-06-03** at **22:53:33**
> 
> @randoentity 
> 
> > Isn't the problem just that IQ1_R4 isn't implemented (https://github.com/ikawrakow/ik_llama.cpp/pull/461)? Because the more I offload to CUDA the slower it gets. I.e. -ot exps=CPU alone is faster than adding more ffn blocks to CUDA (also tested single or multiple devices; same result).
> 
> Oof you are correct! I totally forgot despite writing that on the model card haha... So I suppose possible options could be:
> 1. I could roll a "non repacked" version of this quant so folks could `-rtr` or manually `llama-quantize --repack --repack-pattern ...` for the exact number of GPU offload tensors.
> 2. Hope ik eventually releases a patch to support IQ1_R4 on CUDA.
> 
> > The kv_b example I got from https://huggingface.co/anikifoss/DeepSeek-R1-0528-DQ4_K_R4 (see above). 
> 
> Oh yeah, sure enough, I see the description from @anikifoss on the [model card here](https://huggingface.co/anikifoss/DeepSeek-R1-0528-DQ4_K_R4#quantization-approach). I knew there were differences in how mainline llama.cpp implemented MLA after ik had already done it, but wasn't clear on the exact implementation differences. That tensor is an odd shape an most quants don't work with it so I keep it a similar `qN_0` size usually hah..
> 
> > I do use a headless system and I don't have any swap allocated. The 172GiB one fits just fine and I can run it with --no-mmap.
> 
> Okay headless with no swap is ineed the way to go. On my home rig I usually would run the smallest ~2bpw quant I'd made as it was faster than some 1bpw quants too even though it was mmap()'ing and pulling 5~6GiB/s off the NVMe drive. 
> 
> @Thireus 
> 
> Wow thanks for the detailed notes, this helps!
> 
> You have 256GB RAM + 72GB VRAM = 328 GB. Why are you running the IQ1_S_R4 given you can fit a larger model that will likely run faster?  You might consider the [IQ2_K_R4 2.799 BPW (220GiB) ](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF#iq2_k_r4-2799-bpw-220gib) which is what I personally use for vibe coding given it is about the fastest on my remote setup. It's repacked quant types are supported for GPU offload so you can actually take advantage of all your VRAM unlike the IQ1_S_R4 as pointed out above.
> 
> >  I had the same results with some full layers loaded to the GPU
> 
> Correct, offloading more layers of the IQ1_S_R4 will not improve speed as the RAM acts like "expensive memory" as ik said once haha... Hence I recommend moving up a size and it will be much faster which is counter-intuitive I know.
> 
> I'll look at your commands and come up with an example one to run the larger IQ2_K_R4 and reply later.
> 
> Seems like I should roll an unpacked version as 128GB RAM does not seem like enough without using GPU offload and GPU offload doesn't speed anything up so not great. Got it!

> 👤 **ubergarm** replied on **2025-06-03** at **23:13:54**
> 
> @Thireus 
> 
> #### Assumptions
> 
> * Device 0: 1x 5090 32GB VRAM
> * Device 1: 1x 3090 24GB VRAM
> * Device 2: 1x 3090 24GB VRAM
> 
> #### Option 1
> So the fastest way to run the existing IQ1_S is probably to only use your single fastest GPU for all attn/shexp as designed and given you have enough RAM the repacked exps will fit and run on RAM no problem.
> 
> ```bash
> CUDA_DEVICE_ORDER=PCI_BUS_ID \
> CUDA_VISIBLE_DEVICES=0 \
> ~/ik_llama-main-b3746-f26fe36-bin-win-cuda-12.8-x64/llama-sweep-bench \
>     -m DeepSeek-R1-0528-IQ1_S_R4-00001-of-00003.gguf \
>     -c 8192 \
>     -mla 3 -f \
>     -amb 512 \
>     -fmoe \
>     -ngl 99 \
>     --ot exps=CPU \
>     --warmup-batch \
>     --threads 18 # <--- psure u have 18 physical cores on i9-7980XE (not SMT/hyperthreads)
> ```
> The only other thing might be to try to add `-ub 1024 -b 1024` and then `-ub 2028 -b 2048` and might get some PP boost. It is counter intuitive to only use a single GPU and not offload more layers but that is a limitation of the repacked iq1_s_r4 quant type at the moment. I switched to llama-sweep-bench as it is easier to read and gives more useful information and has the same syntax as llama-server so much easier than llama-bench which has its own argument style.
> 
> I changed a few things just for style and not for any specific reason. When you run the same command with `llama-server` just increase the context as much as you'd like and remove warmup batch.
> 
> #### Option 2
> Here is how I'd run the recommended the one size up IQ2_K_R4. This will be faster than Option 1 and more suited to your rig.
> ```bash
> CUDA_DEVICE_ORDER=PCI_BUS_ID \
> CUDA_VISIBLE_DEVICES=0,2,1 \
> ~/ik_llama-main-b3746-f26fe36-bin-win-cuda-12.8-x64/llama-sweep-bench \
>     -m DeepSeek-R1-0528-IQ2_K_R4-00001-of-00005.gguf \
>     -c 8192 \
>     -mla 3 -f \
>     -amb 512 \
>     -fmoe \
>     -ngl 99 \
>     --main-gpu 0 \
>     -ot "blk\.(3|4|5).ffn_.*=CUDA0" \
>     -ot "blk\.(6|7|8).ffn_.*=CUDA1" \
>     -ot "blk\.(9|10|11).ffn_.*=CUDA2" \
>     --ot exps=CPU \
>     --warmup-batch \
>     --threads 18 # <--- psure u have 18 physical cores on i9-7980XE (not SMT/hyperthreads)
> ```
> 
> Once you get it to at least run right then go about increasing the actual number of (3|4|5) layers to squeeze as much onto GPUs as you can after deciding how much context with which you'd like to run.  Take a look at the commands in the details folds by [@RodriMora](https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13341279) where they tuned up batch and achieved your wish:
> 
> > wish I had the same results with some full layers loaded to the GPU
> 
> Okay, hope that helps! Thanks for helping me figure out why folks are having issue with the IQ1_S_R4 which cannot run any additional layers on GPU!

> 👤 **ubergarm** replied on **2025-06-04** at **04:19:07**
> 
> Okay, uploading the `IQ1_S` now that supports offloading more layers onto GPU. Ideally you would run it with `-rtr` too which takes a little time but should now fit in 128GiB RAM + 24GB VRAM rigs in my testing. Updating model card with two working examples.

> 👤 **Thireus** replied on **2025-06-04** at **07:16:06**
> 
> @ubergarm, thank you for the tips, I'm downloading IQ2_K_R4 and IQ1_S. Will report back.
> 
> I believe `-f` meant `-fa` from your commands, and `--ot` should be `-ot`.
> 
> On Intel, matching the number of threads to the number of CPU threads gives it a 25% boost. Unfortunately I'm still capped at PP 21t/s no matter the -b -ub combination... See results: https://thireus.com/GITHUB/ik_llama_Thireus_bench_02.txt (Firefox: View -> Repair Text Encoding)

> 👤 **Thireus** replied on **2025-06-04** at **08:31:00**
> 
> @ubergarm, I need to do more testing but happy days! `IQ1_S` gives me 246t/s PP 🏎️💨
> The trick was indeed NOT TO USE `IQ1_S_R4` for now until support is added for CUDA - https://github.com/ikawrakow/ik_llama.cpp/pull/461
> 
> Single GPU (5090) with VRAM far from being maxed out:
> ```
> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ~/ik_llama-main-b3746-f26fe36-bin-win-cuda-12.8-x64/llama-sweep-bench -m DeepSeek-R1-0528-IQ1_S-00001-of-00003.gguf  -mla 3 -fa \
>   -amb 512 \
>   -fmoe \
>   -ctk f16 \
>   -c 16384 \
>   -ngl 99 \
>   -ot "blk\.(3|4|5|6|7)\.ffn_.*=CUDA0" \
>   -ot exps=CPU \
>   -b 4096 -ub 4096 \
>   --warmup-batch \
>   --no-mmap \
>   --threads 36
> ...
> main: n_kv_max = 16384, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 99, n_threads = 36, n_threads_batch = 36
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |   21.983 |   186.33 |  301.159 |     3.40 |
> |  4096 |   1024 |   4096 |   23.136 |   177.04 |  303.922 |     3.37 |
> |  4096 |   1024 |   8192 |   24.425 |   167.69 |  305.637 |     3.35 |
> |  4096 |   1024 |  12288 |   25.620 |   159.88 |  306.497 |     3.34 |
> ```
> 
> Multi-GPU (5090 + 2x3090) with maxed out VRAM on all GPUs:
> ```
> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 ~/ik_llama-main-b3746-f26fe36-bin-win-cuda-12.8-x64/llama-sweep-bench -m DeepSeek-R1-0528-IQ1_S-00001-of-00003.gguf  -mla 3 -fa \
>   -amb 512 \
>   -fmoe \
>   -ctk f16 \
>   -c 16384 \
>   -ngl 99 \
>   -ot "blk\.(3|4|5|6|7|8|9|10|11|12)\.ffn_.*=CUDA0" -ot "blk\.(13|14|15|16|17|18|19)\.ffn_.*=CUDA1" -ot "blk\.(20|21|22|23|24|25|26)\.ffn_.*=CUDA2" \
>   -ot exps=CPU \
>   -b 4096 -ub 4096 \
>   --warmup-batch \
>   --no-mmap \
>   --threads 36 \
>   --main-gpu 0
> ...
> main: n_kv_max = 16384, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 99, n_threads = 36, n_threads_batch = 36
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |   16.613 |   246.56 |  177.385 |     5.77 |
> |  4096 |   1024 |   4096 |   17.240 |   237.59 |  176.868 |     5.79 |
> ```
> 
> Loading more layers onto GPU VRAMs finally gets me higher speeds with `IQ1_S`!

> 👤 **randoentity** replied on **2025-06-04** at **10:48:48**
> 
> Happy day! It works and I get above TG 4 t/s.
> @Thireus what is CUDA_DEVICE_ORDER=PCI_BUS_ID for? More consistency when rearranging devices with CUDA_VISIBLE_DEVICES as you don't rely on the heuristics which could change between CUDA versions and potentially hardware conditions?

> 👤 **Thireus** replied on **2025-06-04** at **10:51:46**
> 
> @randoentity yep exactly this, it ensures to directly rely on the PCIe order, so I know exactly which card is which.

> 👤 **randoentity** replied on **2025-06-04** at **10:59:44**
> 
> Ohh and does anyone know if the --main-gpu setting uses the cuda ordering? So if I do CUDA_VISIBLE_DEVICES=2,0,1 will doing -mg=0 select the first device in aforementioned list (I.e. the one that appears as device 2 in nvtop/nvidia-smi)? I've tried playing with this but empiricism ran away from me at some point.

> 👤 **RodriMora** replied on **2025-06-04** at **11:04:07**
> 
> > Ohh and does anyone know if the --main-gpu setting uses the cuda ordering? So if I do CUDA_VISIBLE_DEVICES=2,0,1 will doing -mg=0 select the first device in aforementioned list (I.e. the one that appears as device 2 in nvtop/nvidia-smi)? I've tried playing with this but empiricism ran away from me at some point.
> 
> I believe when you do CUDA_VISIBLE_DEVICES=2,0,1, for ik_llama.cpp now cuda0 is the real cuda2

> 👤 **randoentity** replied on **2025-06-04** at **12:00:53**
> 
> Same command as Thireus but with 7 layers in CUDA0 and only 6 cores, which seems to massively cripple PP, but it could be something else. I'll run some more tests, but that this runs and is not outputting gibberish is absolutely astonishing!
> 
> ```
> main: n_kv_max = 16384, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 99, n_threads = 6, n_threads_batch = 6
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |  125.382 |    32.67 |  208.638 |     4.91 |
> |  4096 |   1024 |   4096 |  125.511 |    32.63 |  213.956 |     4.79 |
> |  4096 |   1024 |   8192 |  127.407 |    32.15 |  218.763 |     4.68 |
> |  4096 |   1024 |  12288 |  129.336 |    31.67 |  221.664 |     4.62 |
> ```
> 
> **Edit:** S_PP t/s in the 160  range with `--threads-batch = 12`!

> 👤 **Thireus** replied on **2025-06-04** at **12:38:47**
> 
> Nice! I haven't played with --threads-batch yet, but will do.
> 
> I've cranked the b and ub values to `-b 16384 -ub 8192`, which give much higher PP speeds now. But doesn't leave much room for context size.
> 
> ```
> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 ~/ik_llama-main-b3746-f26fe36-bin-win-cuda-12.8-x64/llama-sweep-bench -m DeepSeek-R1-0528-IQ1_S-00001-of-00003.gguf  -mla 3 -fa \
>   -amb 1024 \
>   -fmoe \
>   -ctk f16 \
>   -c 16384 \
>   -ngl 99 \
>   -ot "blk\.(3|4|5|6|7|8|9|10)\.ffn_.*=CUDA0" -ot "blk\.(11|12|13|14|15)\.ffn_.*=CUDA1" -ot "blk\.(16|17|18|19|20)\.ffn_.*=CUDA2" \
>   -ot exps=CPU \
>   -b 16384 -ub 8192 \
>   --warmup-batch \
>   --no-mmap \
>   --threads 36 \
>   --main-gpu 0
> ---
> main: n_kv_max = 16384, n_batch = 16384, n_ubatch = 8192, flash_attn = 1, n_gpu_layers = 99, n_threads = 36, n_threads_batch = 36
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  8192 |   2048 |      0 |   24.003 |   341.30 |  397.380 |     5.15 |
> |  8192 |   2048 |   8192 |   31.843 |   257.26 |  404.438 |     5.06 |
> ---
> ```

> 👤 **Ph0rk0z** replied on **2025-06-04** at **16:14:17**
> 
> Heh.. from the tests I run yesterday/today.. it seems pointless to download other people's R4 quants unless you have the exact same configuration as they do else you get massive speed hits. https://github.com/ikawrakow/ik_llama.cpp/discussions/491
> 
> If I didn't do something wrong, it's more ideal to just use RTR if you want higher tg at the expense of prompt processing. There is a sweet spot for the tradeoff, imo. My CPU is xeon scalable without vnni.. perhaps another codepath or single CPU doesn't have the problem.

> 👤 **ubergarm** replied on **2025-06-04** at **21:12:39**
> 
> @Thireus @randoentity and all,
> 
> More good news, ik took a crack at getting `IQ1_S_R4` CUDA implementation going with [PR492](https://github.com/ikawrakow/ik_llama.cpp/pull/492). Feel free to build that branch and compare speeds as it will likely increase your TG numbers.

> 👤 **randoentity** replied on **2025-06-05** at **04:27:36**
> 
> Thanks @ubergarm . It looks like a 10% speedup in TG, but slower PP as a tradeoff. However, more space for context might be nice, especially for those with only 24GB VRAM. I'll do some more of those maze tests if you decide to release a pure IQ1_S_R4 (as you mention in the PR, the IQ1_S_R4 you uploaded on HF doesn't work). It might be worth it to make another post on LocalLlama for that.

> 👤 **ubergarm** replied on **2025-06-05** at **15:04:07**
> 
> Yeah I did make and test that `IQ1_S_R4-smol` i call it with iq5_ks for all attn/shexp/token_embd then IQ1_S_R4 for all ffn_up/down/gate_exps but as ik mentioned it is indeed a little bit more dumb despite being just a little bit smaller. 
> `Final estimate: PPL = 5.0048 +/- 0.02978`
> 
> I decided to not be so brash and just wait a little bit as sounds like ik is interested in also adding `IQ1_M_R4` cuda support in which case that first model I released would be good to go. Oh yes I'll go test [PR494](https://github.com/ikawrakow/ik_llama.cpp/pull/494) now!

> 👤 **randoentity** replied on **2025-06-05** at **21:12:37**
> 
> About 20% faster TG and PP didn't take a hit! I think I could even squeeze in another layer or two. Now let's see if this smolly can solve mazes.
> ```sh
> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,0,1 ./
> build_bf16/bin/llama-sweep-bench \
> --no-mmap \
> --attention-max-batch 64 \
> --batch-size 4096 --ubatch-size 4096 \
> --cache-type-k f16 \
> --ctx-size 32768 \
> --flash-attn \
> --fused-moe \
> --main-gpu 0 \
> --min_p 0.01 \
> --mla-use 3 \
> --model /mnt/models/ubergarm/dsr1-0528-iq1-s4/DeepSeek-R1-0528-
> IQ1_S_R4-00001-of-00003.gguf \
> --n-gpu-layers 99 \
> --override-tensor "blk\.(3|4|5|6|7|8|9)\.ffn_.*=CUDA0" \
> --override-tensor "blk\.(10|11|12|13|14|15|16)\.ffn_.*=CUDA2" \
> --override-tensor "blk\.(17|18|19|20|21|22|23)\.ffn_.*=CUDA1" \
> --override-tensor exps=CPU,attn_kv_b=CUDA1 \
> --temp 0.6 \
> --threads 6 \
> --threads-batch 12 \
> --top_p 0.95 \
> --warmup-batch
> ```
> ```
> main: n_kv_max = 32768, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 99, n_threads = 6, n_threads_batch = 12
> 
> |    PP |     TG |   N KV |   T PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |   23.449 |   174.68 |  180.280 |     5.68 |
> |  4096 |   1024 |   4096 |   27.103 |   151.13 |  183.508 |     5.58 |   
> |  4096 |   1024 |   8192 |   31.199 |   131.29 |  187.610 |     5.46 |
> |  4096 |   1024 |  12288 |   35.090 |   116.73 |  190.219 |     5.38 |     
> ```

> 👤 **Thireus** replied on **2025-06-05** at **21:34:55**
> 
> Sorry if this is a silly question, but aren't unsloth's quants supported on ik_llama? I can see they load but fatal error occurs on inference.

> 👤 **randoentity** replied on **2025-06-05** at **22:01:02**
> 
> @Thireus ah yeah, try disabling fmoe.

> 👤 **ikawrakow** replied on **2025-06-06** at **06:09:51**
> 
> Does [#495](https://github.com/ikawrakow/ik_llama.cpp/issues/495) solve the `-fmoe` issue with Unsloth's model?

> 👤 **randoentity** replied on **2025-06-06** at **12:53:56**
> 
> For those with multi-GPU having uneven bandwidth (i.e. different  number of lanes or PCIe generation): try playing with `--tensor-split`. I got from 175 PP 5.6 TG to 200 PP 6.0 TG by setting it to 1,0,0. Having fewer full layers on the fastest GPU, but more tensors overall seems to give a modest boost.
> 
> I also found that `-amb` doesn't do much for speeds, so setting it to 64 frees up some memory (lower doesn't work).
> 
> Finally, the bf16 compilation option prevents use of ctk q8_0, and I have to double check this still, but the speed boost doesn't seem significant on the R4 quant.

> 👤 **ikawrakow** replied on **2025-06-06** at **13:30:09**
> 
> > Finally, the bf16 compilation option prevents use of ctk q8_0
> 
> This would be news to me. 
> 
> > I also found that -amb doesn't do much for speeds, so setting it to 64 frees up some memory (lower doesn't work).
> 
> For your specific system, with the specific model you are using. The `-amb` option was added in PR [#260](https://github.com/ikawrakow/ik_llama.cpp/issues/260), which has an explanation what it does. Please don't recommend `-amb 64` as a general truth to others.

> 👤 **randoentity** replied on **2025-06-06** at **14:51:22**
> 
> I've created [#499](https://github.com/ikawrakow/ik_llama.cpp/issues/499) for the error.
> 
> Thanks for the link to the explanation for `-amb`! I didn't mean to spread misinformation, sorry. It was meant in the context of multi-GPU, this model, and this quant.

> 👤 **Ph0rk0z** replied on **2025-06-06** at **15:32:34**
> 
> I have set BF16 and almost always use Q8 cache with different AMB, including 64. It shrinks the compute buffer so you can fit another piece of a layer or layer itself. For me it also didn't do much for speeds on it's own. Best to benchmark. Has worked both with deepseek and qwen including the IQ1 unsloth.

---

👤 **cmoncure** commented on **2025-06-03** at **21:25:22**

Can anyone explain to me in simple terms.  When considering tensor offload configurations, what exactly is the nature of the stickiness or entanglement between tensors? What tensors MUST go together as an indivisible unit?

✔ All tensors on CPU
✔ All tensors on GPU
✔ attn=CUDA0 exps=CPU
✔ blk.(3|4|5|6).ffn_*=CUDA0 exps=CPU

FACT: attention and exps can be separated between CPU and GPU. 
FACT: Entire layers can be offloaded from CPU to GPU.  

But, you want to do something like this?

✘ attn=CUDA0, blk.3.*=CUDA1 exps=CPU
✘ blk.3.ffn_.*=CUDA0, blk.4.ffn_.*=CUDA1 exps=CPU
✘ R4 quant layers with -sm none => CUDA0; K quant layers with -sm layer => CUDA1

Are these **impossible** for REASONS or just "not supported" i.e. go learn the domain and write the code myself?

> 👤 **Thireus** replied on **2025-06-03** at **21:32:54**
> 
> I'm reading this answer - https://chatgpt.com/share/683f69cc-bff8-800f-8610-55aa4de145ed

> 👤 **ubergarm** replied on **2025-06-03** at **23:25:38**
> 
> @cmoncure 
> 
> Zero offense intended, and just being a mirror, but for some reason I have a hard time understanding your writing for some reason. Perhaps you're just asking broad questions beyond my level of understanding as my brain is usually in the weeds ignoring the forest to mix my metaphores haha... Are you maybe copy pasting ai generated stuff as I never type unicode checks and x's. Anyway, just working on my communication, thanks.
> 
> Let me try to answer what makes sense to me:
> 
> > What tensors MUST go together as an indivisible unit?
> 
> 1. If you are using `-fmoe` which I believe you should be then check out [PR229](https://github.com/ikawrakow/ik_llama.cpp/pull/229) where `ffn_(up|gate)` computation was optimized in such a way that I'd recommend not putting those on different devices for a given layer.
> 
> In general you want to avoid sending data between different devices as it incurs some time to copy it from say the GPU to the CPU or from CUDA0 via the PCIe bus to CUDA1 etc. Most folks here don't have magic RDMA GPUs nor P2P drivers nor NVLinks which can help with that.
> 
> > Are these impossible for REASONS or just "not supported" i.e. go learn the domain and write the code myself?
> 
> mu

> 👤 **cmoncure** replied on **2025-06-04** at **00:12:27**
> 
> "go learn the domain and write the code yourself" then, got it.

> 👤 **cmoncure** replied on **2025-06-04** at **00:23:01**
> 
> > attn=CUDA0, blk.3=CUDA1, exps=CPU
> 
> > If “blk.3” means “all of layer 3 (attention + feed‑forward)” goes to CUDA:1, but you also try to put “attention” itself (the subcomponent of layer 3) on CUDA:0, you’ve overlapped. The “attention” sub‐block lives partly on CUDA:0 (its matmuls → exps) and partly on CUDA:1 (the rest of the layer 3). As soon as you compute context = softmax(scores) @ V, you need Q/K/V and the output projection to be together. If some of those weights/activations are on CUDA:1 and some on CUDA:0, you’d have to copy intermediates back and forth in the middle of that attention forward. In practice, no mainstream codebase will (a) know how to break attention in exactly two devices at the same time, or (b) optimize all of those back‑and‑forth copies.
> 
> Well, let's look at this helpful and reasonable explanation from ChatGPT.  All is well and good here! No codebase can handle this scenario where the whole of layer 3 (attention + feed forward) goes to CUDA1, but attention remains on CUDA0, because the activations get split between CUDA0 and CUDA1.  Totally makes sense.
> 
> Okay well, how then does this work when I do `-ot attn=CUDA0 exps=CPU`?  Now attention is on CUDA0 and feed forward is on CPU... they are split!  IMPOSSIBLE! ... impossible, right ChatGPT? :face_exhaling:

> 👤 **Ph0rk0z** replied on **2025-06-04** at **11:33:47**
> 
> >ffn_(up|gate) computation was optimized in such a way that I'd recommend not putting those on different devices for a given layer.
> 
> So that explains why that causes being GPU bound. It seems I can put individual ups or gates on GPU vs CPU but I can't put one up or gate from the same layer on different GPUs. Both up/gate on the same GPU speeds things up though.

---

👤 **cmoncure** commented on **2025-06-06** at **15:04:26**

Day 4 of chasing performance with bespoke repacking and the delicate and mercurial (i.e. broken) configuration args.  I'm ready to give up. I tried so many blends of tensor offload parameters and statically repacking my head is spinning.  Nothing I tried can reach the high water marks of:
16 TG t/s with `--rtr -ot attn=CUDA0` _(but bad PP)_
200 PP t/s with no repacking and `-sm layer -ngl 8` _(but bad TG)_

I made a repacked quant that converts only the exps tensors running on CPU to _r4 (exps 11...60) and run everything else on CUDA0 and CUDA1 with --sm layer.  It should be the best of both worlds, but it's the worst of both worlds: PP 71 and TG 9.

The domain may seem like black magic but at the end of the day all we're doing here is matrix multiplication. My instinct is screaming at me that there's huge amounts of performance left on the table.  The wild and frankly shocking comment that "high gpu utilization is actually a bad thing" notwithstanding, the goal is to get the most math done per unit time as possible. It's very telling that seemingly no one can give an explanation that holds water of what operations must be tied to one another on a compute device, or why the tensors can be split in one way between CPU and CUDA0 but as soon as you extend the split to involve CUDA1 the performance bombs.  We want to run big models on commodity hardware and that means finding the way of distributing the computation among multiple relatively-low-capacity compute units that maximizes the contribution of all the units.

> 👤 **Thireus** replied on **2025-06-06** at **15:08:15**
> 
> Don't give up so soon! I'm in the same boat and I need motivation. 😂
> 
> Which model/quant and ik_llama build are you using?

> 👤 **cmoncure** replied on **2025-06-06** at **15:48:32**
> 
> version: 3722 (7a8abe29)
> 
> bartowski/deepseek-ai_DeepSeek-V3-0324-GGUF and my various repackings of it.
> `./ik_llama.cpp/build/bin/llama-quantize --repack --repack-pattern "blk.(11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|50|51|52|53|54|55|56|57|58|59|60).ffn_gate_exps","blk.(11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|50|51|52|53|54|55|56|57|58|59|60).ffn_down_exps","blk.(11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|50|51|52|53|54|55|56|57|58|59|60).ffn_up_exps" ~/AIModels/textgen/deepseek-ai_DeepSeek-V3-0324-Q4_K_M-V2-00001-of-00011.gguf ~/AIModels/textgen/repacked5.gguf COPY
> `

> 👤 **VinnyG9** replied on **2025-06-11** at **03:31:14**
> 
> > Day 4 of chasing performance with bespoke repacking and the delicate and mercurial (i.e. broken) configuration args. I'm ready to give up. I tried so many blends of tensor offload parameters and statically repacking my head is spinning. Nothing I tried can reach the high water marks of: 16 TG t/s with `--rtr -ot attn=CUDA0` _(but bad PP)_ 200 PP t/s with no repacking and `-sm layer -ngl 8` _(but bad TG)_
> > 
> > I made a repacked quant that converts only the exps tensors running on CPU to _r4 (exps 11...60) and run everything else on CUDA0 and CUDA1 with --sm layer. It should be the best of both worlds, but it's the worst of both worlds: PP 71 and TG 9.
> > 
> > The domain may seem like black magic but at the end of the day all we're doing here is matrix multiplication. My instinct is screaming at me that there's huge amounts of performance left on the table. The wild and frankly shocking comment that "high gpu utilization is actually a bad thing" notwithstanding, the goal is to get the most math done per unit time as possible. It's very telling that seemingly no one can give an explanation that holds water of what operations must be tied to one another on a compute device, or why the tensors can be split in one way between CPU and CUDA0 but as soon as you extend the split to involve CUDA1 the performance bombs. We want to run big models on commodity hardware and that means finding the way of distributing the computation among multiple relatively-low-capacity compute units that maximizes the contribution of all the units.
> 
> here fellow OCD, see if [this](https://www.reddit.com/r/LocalLLaMA/comments/1kpe33n/comment/msxzv0s/) helps

> 👤 **cmoncure** replied on **2025-06-11** at **19:21:38**
> 
> I can't use this approach at all because as soon as I try to involve CUDA1 with `-sm none` and `-mg` the code attempts to allocate 1.5 trillion bytes of memory on the GPU (four times the size of the entire model tensors)

> 👤 **saood06** replied on **2025-06-12** at **01:38:20**
> 
> @cmoncure 
> 
> Are you building with `-DGGML_SCHED_MAX_COPIES=1`?
> 
> That may be needed for now to avoid that issue, see https://github.com/ikawrakow/ik_llama.cpp/issues/437#issuecomment-2954768207

> 👤 **VinnyG9** replied on **2025-06-13** at **18:06:09**
> 
> > I can't use this approach at all because as soon as I try to involve CUDA1 with `-sm none` and `-mg` the code attempts to allocate 1.5 trillion bytes of memory on the GPU (four times the size of the entire model tensors)
> 
> set ngl to all minus 1 layer

---

👤 **Gaolingx** commented on **2025-06-06** at **18:13:10**

I am running on 1x epyc 9334qs + 12x ddr5 6400mhz(works on 4800mhz) 48g + 3070 16g, **~10.3t/s TG, ~78t/s PP**, it works well, but the VRAM has used about 12GB, I am not sure how large a context window(`--ctx-size`) I can open.

model: unsloth/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-Q4_K_M-00001-of-00009.gguf
parameter: 
```text
./llama-server --model "$MODEL_PATH" \
    --host :: \
    --port 21434 \
    --threads 24 \
    --n-gpu-layers 63 \
    --ctx-size 8192 \
    --mla-use 3 \
    --flash-attn \
    --cache-type-k f16 \
    --run-time-repack \
    --fused-moe \
    --override-tensor exps=CPU
 ```
---
![493c7dc6-09ee-4c0d-b161-7460df01df1a](https://github.com/user-attachments/assets/bf76a18c-0ca5-4213-acf5-df827c5447d7)
![7a111fa3-9e55-496e-a4e6-45c94f83da32](https://github.com/user-attachments/assets/7e3f7f0d-5b06-409c-9f74-affd7b2568bb)
![b1a2b995-aa68-48c9-a096-6287a6f147eb](https://github.com/user-attachments/assets/e67cbfe6-3abe-41ab-9aae-dab8d28a45a9)
![20f42fad-1929-4dbb-950f-f1dd30fe47e1](https://github.com/user-attachments/assets/d09ffc9c-d088-4689-b78a-fabc8baf850e)
![46da9012-a4c8-4392-b4c2-d79dd66a9371](https://github.com/user-attachments/assets/287327ce-7d2b-4e0d-b3a4-bdec64d2dbc6)

---

👤 **ciprianveg** commented on **2025-06-06** at **18:18:44**

Add -b 4096 -ub 4096 and you will have 3x your pp speed

> 👤 **zts9989** replied on **2025-06-26** at **01:36:14**
> 
> https://github.com/ggml-org/llama.cpp/issues/14325
> Thanks.

---

👤 **saood06** commented on **2025-06-11** at **15:05:50**

So I finally cooked a quant after sitting on the BF16 for so long. 

I ended up going with @ubergarm's imatrix with:
`--custom-q "token_embd\.weight=q4_K,attn_k_b.weight=q5_0,attn_*=iq4_ks_r4,output\.weight=q6_K,.*=iq4_k_r4"`

Running sweep right now but early impressions are good enough that I may end up using this for a while before attempting some more mixes. (PP seems a bit better, TG seems about the same)

(As a reminder the quant I end up settling on for V3-0324 was a very simple `--custom-q "token_embd.weight=iq4_k,.*=iq4_k_r4"`)

---

👤 **zts9989** commented on **2025-06-26** at **01:44:18**

Thank you for the discussion. Sharing my experimental results for your reference.

![p1](https://github.com/user-attachments/assets/ff88b36f-ec69-4956-9694-56b2142d554e)
![p5](https://github.com/user-attachments/assets/ac731f45-b798-472b-879b-d5400c865787)

https://github.com/ggml-org/llama.cpp/issues/14325

> 👤 **saood06** replied on **2025-06-26** at **01:58:27**
> 
> You said in the linked post:
> 
> >I tested ik llamacpp and found some performance improvements, but the stability was insufficient (there also seem to be other issues with usability and stability)
> 
> Can you make issues for the usability and stability problems you mentioned.

> 👤 **zts9989** replied on **2025-06-26** at **02:03:56**
> 
> Absolutely. I can provide that shortly. Please excuse the informal nature of my issue description—it's based more on observational feel than quantitative metrics or official specifications. Much of the feedback I provide within the llama.cpp community tends to reflect practical usage experiences rather than technical documentation standards.

> 👤 **saood06** replied on **2025-06-26** at **02:09:12**
> 
> > Absolutely. I can provide that shortly.
> 
> Thanks.
> 
> >Please excuse the informal nature of my issue description—it's based more on observational feel than quantitative metrics or official specifications. Much of the feedback I provide within the llama.cpp community tends to reflect practical usage experiences rather than technical documentation standards.
> 
> No worries, I've seen your feedback to llama.cpp (especially your NUMA stuff) and in my view it is very useful.

> 👤 **zts9989** replied on **2025-06-26** at **03:38:50**
> 
> My sincere apologies, I retract what I said (Please forgive me for trying to use ik llama.cpp the same way I use the standard llama.cpp, which led to unexpected results. For example, with llama-cli, I didn't add the -cnv switch, so the model went off the rails and generated output I didn't expect).
> 
> ik llama.cpp does offer a performance improvement over standard llama.cpp. Speed increased from 17.4 t/s (llama.cpp) to 18.xx t/s (ik).
> 
> **Apologies again. (I'm really sorry.)**

> 👤 **ikawrakow** replied on **2025-06-26** at **06:48:24**
> 
> The recommended batch/u-batch size for `ik_llama.cpp` **with MoE models** is 4096 tokens (if you have enough RAM/VRAM; derfault u-batch is perfectly fine for dense models). Performance gains beyond 4096 are quite minor and do not justify the massive increase of compute buffer sizes. Some users go up to 6144. A batch/u-batch size of 16384 is really pushing it.
> 
> You are reporting a few percent performance benefit for TG with `ik_llama.cpp` vs `llama.cpp`. The difference in PP should be quite a bit larger, no? Interesting you are not looking at that, considering that the whole thread is about batch/u-batch size, which only matters for PP. 
> 
> Having to add `-cnv` in `ik_llama.cpp` is my personal preference. This is how `llama.cpp` used to behave as well, and I'm annoyed each time I want to use `llama-cli` in `llama.cpp` for a quick performance/coherence check when it starts in conversation mode rather than completing my prompt. And because I don't use mainline very often, each time I need to go and check if it was `--no-conv` or `-no-conv` to disable the conversation mode. Extremely annoying.

> 👤 **zts9989** replied on **2025-06-26** at **08:17:43**
> 
> PP (Prompt Processing) speed in ik_llama.cpp is significantly faster than in standard llama.cpp.
> At a batch size of 8192, llama.cpp achieves 170 tokens/s while ik_llama.cpp reaches 200 tokens/s (I will provide screenshot evidence later).
> At a batch size of 16384, llama.cpp achieves 270 tokens/s, but ik_llama.cpp enters an infinite loop and generates irrelevant outputs. This prevented further performance testing (my screenshot evidence here is insufficient since terminating the process via Ctrl+C doesn’t log PP/TG metrics).
> 
> The biggest challenge in offline DeepSeek deployment is PP performance. Compared to enterprise-grade Prefill/Decode (PD)-separated architectures that deliver robust PP and TG performance, single-machine deployments (for individuals/small teams) struggle with long-context (>10K token) processing due to suboptimal PP efficiency.
> 
> From my perspective: If GPU VRAM can double PP performance, it’s maximizing resource utilization. Using VRAM to host sparsely activated expert weights (at only ~4% utilization rate) seems wasteful.
> 
> The 270 tokens/s at 16384 batch size represents the peak PP performance I achieved after exhaustive tuning of configurations, CPU/GPU combinations, and offline DeepSeek deployment setups.
> I still strongly advocate for official support of 16384 batch size.
> 
> I sincerely apologize again for my earlier statements.
> Looking forward to future updates—I wish both llama.cpp and ik_llama.cpp continued success. Thank you (and apologies for my previous remarks) for your efforts and open-source work, which enable offline LLM usage.
> 
> Screenshot evidence will be attached as noted.

---

👤 **zts9989** commented on **2025-06-26** at **08:21:32**

PP (Prompt Processing) speed in ik_llama.cpp is significantly faster than in standard llama.cpp.
At a batch size of 8192, llama.cpp achieves 170 tokens/s while ik_llama.cpp reaches 200 tokens/s (I will provide screenshot evidence later).
At a batch size of 16384, llama.cpp achieves 270 tokens/s, but ik_llama.cpp enters an infinite loop and generates irrelevant outputs. This prevented further performance testing (my screenshot evidence here is insufficient since terminating the process via Ctrl+C doesn’t log PP/TG metrics).

The biggest challenge in offline DeepSeek deployment is PP performance. Compared to enterprise-grade Prefill/Decode (PD)-separated architectures that deliver robust PP and TG performance, single-machine deployments (for individuals/small teams) struggle with long-context (>10K token) processing due to suboptimal PP efficiency.

From my perspective: If GPU VRAM can double PP performance, it’s maximizing resource utilization. Using VRAM to host sparsely activated expert weights (at only ~4% utilization rate) seems wasteful.

The 270 tokens/s at 16384 batch size represents the peak PP performance I achieved after exhaustive tuning of configurations, CPU/GPU combinations, and offline DeepSeek deployment setups.
I still strongly advocate for official support of 16384 batch size.

I sincerely apologize again for my earlier statements.
Looking forward to future updates—I wish both llama.cpp and ik_llama.cpp continued success. Thank you (and apologies for my previous remarks) for your efforts and open-source work, which enable offline LLM usage.

llama.cpp
batsize 4096 pp 133t/s
batsize 8192 pp 170t/s  (up to 160k with DeepSeek's solution   ggml_cuda_cpy)
batsize 16384 pp 270t/s  (up to 80k with DeepSeek's solution   ggml_cuda_cpy)

ik llama.cpp
4096 pp 148.7t/s
8192 pp 200t/s
16384 pp na

ik llama.cpp -mla 3 -fmoe -amb 512/1024
4096 177  
8192  281  
16384 347   36k input 
16384 na     50k input

Screenshot evidence will be attached as noted.

![Screenshot_2025-06-26_15-21-42](https://github.com/user-attachments/assets/38f9bf03-6121-4548-88d8-6e3e43dd12aa)
![Screenshot_2025-06-26_15-29-56](https://github.com/user-attachments/assets/fa1c28d4-8060-4f73-ae76-8f7d60da89ce)

> 👤 **ikawrakow** replied on **2025-06-26** at **09:04:13**
> 
> I suggest you try `-mla 3 -fmoe`. If you run out of VRAM, add `-amb 512`. For the 36k tokens you are processing you should get a very significant performance boost in PP performance.

> 👤 **Thireus** replied on **2025-06-26** at **09:14:12**
> 
> @zts9989 - Yep, similar observations here https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13367713 ;)

> 👤 **zts9989** replied on **2025-06-26** at **09:17:36**
> 
> > I suggest you try `-mla 3 -fmoe`. If you run out of VRAM, add `-amb 512`. For the 36k tokens you are processing you should get a very significant performance boost in PP performance.
> 
> ![Screenshot_2025-06-26_17-15-45](https://github.com/user-attachments/assets/5a41852e-89d9-46ab-a4fb-d785523c805a)
> 
> ![Screenshot_2025-06-26_17-28-04](https://github.com/user-attachments/assets/4a001025-848f-4ee3-85fc-702330c0ac3a)
> 
> ![Screenshot_2025-06-26_17-33-48](https://github.com/user-attachments/assets/dbaefbef-005b-4cca-9e7e-8c2b6dfed301)
> ![Screenshot_2025-06-26_17-38-03](https://github.com/user-attachments/assets/cd480b00-dc89-4dc6-9fb7-3e168a189d26)
> ![Screenshot_2025-06-26_17-40-59](https://github.com/user-attachments/assets/2795ffbf-0ec8-43b9-a385-ffc52917881c)

---

👤 **zts9989** commented on **2025-06-26** at **09:56:07**

Turns out I was using ik llama.cpp incorrectly all along.
Coming full circle, I'm back to square one:
Please optimize the ggml_cuda_cpy function to support copying tensors larger than 2GB.
Thanks!
(DeepSeek's solution can fully utilize 163,840 context length under -ub 8192 -b 8192 configuration.)"

---

👤 **ikawrakow** commented on **2025-06-26** at **10:38:51**

> Please optimize the ggml_cuda_cpy function to support copying tensors larger than 2GB.

I can see what I can do, but I don't feel particularly motivated to engage in hunting down integer overflows and CUDA maximum block size exceeded issues in code that I didn't write myself or at least modified at some point. There are still some performance optimizations left that would be more interesting to work on.

But based on your performance numbers, I estimate you have a 30 GB/s PCI-E, so it takes about 13 seconds to upload all experts stored in RAM to the GPU(s). For u-batch size of 16k tokens you are getting 347 t/s, so the u-batch takes about 47 seconds, so computation is about 34 seconds (and it is easy to verify that this napkin math works for u-batches of 8k and 4k). If you would go to u-batch size of 32k tokens, computation for the batch will at least double, offload time will stay the same, so it will be taking about 81 seconds, so performance will be in the range of 390 t/s. In reality when batch sizes become very large, computing performance goes down due to limited caches, etc, so I'm guessing you will saturate around 350-360 t/s. If I look at the 8k u-batch size, I estimate you have in the range of 30 GB of unused VRAM. Hence, you could have uploaded 5 or 6 layers of experts to the GPU. That would slightly increase your PP performance, and will also boost your TG performance by about 10%.

> 👤 **zts9989** replied on **2025-06-26** at **13:02:20**
> 
> I just gave it a try.
> My GPU is connected via PCIe 4.0 x16, so the bandwidth is around 30 GB/s. 347 t/s really seems to be the current limit for my setup. I experimented with a batch size of 32,768 tokens, but performance actually decreased. I also tried pre-loading experts into the available GPU VRAM – the gain was minimal (just from 17.3 to 17.5 t/s).
> 
> Thanks for the suggestions though. I've now secured a runtime environment with higher-performance PP.

> 👤 **ikawrakow** replied on **2025-06-26** at **17:37:09**
> 
> Does PR [#560](https://github.com/ikawrakow/ik_llama.cpp/issues/560) let you compute the context that fails on the main branch with batch/u-batch of 16k tokens?

> 👤 **zts9989** replied on **2025-06-27** at **02:46:28**
> 
> > Does PR [#560](https://github.com/ikawrakow/ik_llama.cpp/issues/560) let you compute the context that fails on the main branch with batch/u-batch of 16k tokens?
> 
> I tried this version, and it still crashed after 131,072. This time it wasn't an error in the cuda cpy, but in the cuda compute. It might really be exceeding the limit.
> 
> Thank you a lot. 
> ![Screenshot_2025-06-27_09-25-51](https://github.com/user-attachments/assets/6d8f2cd4-9e59-4943-ad69-9c472f7dad08)

---

👤 **eous** commented on **2025-07-10** at **21:20:45**

Just a couple benchmark dumps.

Compiled with `cmake -B ./build -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1 -DCMAKE_CUDA_ARCHITECTURES="120"`
```
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
  Device 1: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
llm_load_tensors: ggml ctx size =    1.40 MiB
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   497.11 MiB
llm_load_tensors:      CUDA0 buffer size = 65593.61 MiB
llm_load_tensors:      CUDA1 buffer size = 70014.23 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 20480
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   697.50 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   675.00 MiB
llama_new_context_with_model: KV self size  = 1372.50 MiB, c^KV (f16): 1372.50 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  2872.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  2712.03 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   432.05 MiB
llama_new_context_with_model: graph nodes  = 8184
llama_new_context_with_model: graph splits = 3

main: n_kv_max = 20480, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 99, n_threads = 1, n_threads_batch = 1

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    4.994 |   820.25 |   24.307 |    42.13 |
|  4096 |   1024 |   4096 |    6.440 |   636.07 |   24.893 |    41.14 |
|  4096 |   1024 |   8192 |    8.033 |   509.89 |   26.175 |    39.12 |
|  4096 |   1024 |  12288 |    9.646 |   424.65 |   27.750 |    36.90 |
|  4096 |   1024 |  16384 |   11.407 |   359.09 |   28.304 |    36.18 |
```
Compiled with `cmake -B ./build -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=1 -DCMAKE_CUDA_ARCHITECTURES="120"`
```
main: n_kv_max = 20480, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 99, n_threads = 1, n_threads_batch = 1

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    5.002 |   818.89 |   23.962 |    42.73 |
|  4096 |   1024 |   4096 |    6.496 |   630.53 |   24.954 |    41.03 |
|  4096 |   1024 |   8192 |    8.334 |   491.49 |   26.183 |    39.11 |
|  4096 |   1024 |  12288 |    9.765 |   419.47 |   27.661 |    37.02 |
|  4096 |   1024 |  16384 |   11.547 |   354.71 |   28.253 |    36.24 |
```

Do not really see a difference with `-DGGML_CUDA_IQK_FORCE_BF16=1` on my setup but that is sort of expected since on mainline at least bf16 is treated like fp32 and fp32 isn't using the faster fp32 accumulation available in modern cuda (https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation-support-overview) last I looked

Hardware
---
```
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
  Device 1: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, compute capability 12.0, VMM: yes
```
```
vendor_id	: AuthenticAMD
cpu family	: 25
model		: 24
model name	: AMD Ryzen Threadripper PRO 7975WX 32-Cores
stepping	: 1
cpu MHz		: 4790.945
cache size	: 1024 KB
physical id	: 0
siblings	: 64
core id		: 31
cpu cores	: 32
apicid		: 63
initial apicid	: 63
fpu		: yes
fpu_exception	: yes
cpuid level	: 16
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid overflow_recov succor smca fsrm flush_l1d sev sev_es debug_swap
bugs		: sysret_ss_attrs spectre_v1 spectre_v2 spec_store_bypass srso
bogomips	: 7988.12
TLB size	: 3584 4K pages
clflush size	: 64
cache_alignment	: 64
address sizes	: 46 bits physical, 57 bits virtual
power management: ts ttp tm hwpstate cpb eff_freq_ro [13] [14]
```
```
$ free -h
               total        used        free      shared  buff/cache   available
Mem:           750Gi        32Gi        32Gi       182Mi       690Gi       718Gi
Swap:          8.0Gi       859Mi       7.2Gi
```
```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
```

> 👤 **ikawrakow** replied on **2025-07-11** at **04:57:13**
> 
> What is the model in these benchmarks?

> 👤 **ubergarm** replied on **2025-07-11** at **06:22:06**
> 
> @ikawrakow 
> 
> I believe it is [ubergarm/DeepSeek-TNG-R1T2-Chimera/IQ1_S at 132.915 GiB (1.699 BPW) quant](https://huggingface.co/ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF#-iq1_s-132915-gib-1699-bpw)
> 
> This was the command psure:
> 
> ```bash
> ./build/bin/llama-sweep-bench \
>     --model /mnt/models/llama/DeepSeek-TNG-R1T2-Chimera-IQ1_S/DeepSeek-TNG-R1T2-Chimera-IQ1_S-00001-of-00003.gguf \
>     -fa -mla 3 -fmoe -amb 512 -mg 0 \
>     --ctx-size 20480 \
>     -ngl 99 \
>     --threads 1 \
>     -ub 4096 -b 4096 \
>     --warmup-batch
> ```
> 
> We had some discussions over on [level1techs forum here](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/287) where I got this info.
> 
> @eous 
> 
> Thanks for your report!
> 
> I tried my hand at a recipe optimized (hopefully) for your dual RTX PRO 6000 Blackwell's if you are interested in testing. The [ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF/IQ2_XXS](https://huggingface.co/ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF#-iq2_xxs-169590-gib-2168-bpw) weighs in at 169.590 GiB (2.168 BPW). I believe it will fit full 160k context with full offload on your 192GB VRAM. I'm not sure if it will have enough for full context *and* `-ub 4096 -b 4096` but hopefully.
> 
> It is a blend of two of the smallest yet faster CUDA inferencing quants, IQ2_KS and slightly smaller IQ2_XXS for the routed exps. The perplexity is better too at around ~4.0 so should be a little "smarter" than the smaller IQ1_S.
> 
> Uploading now, should be live within a couple hours!

> 👤 **ikawrakow** replied on **2025-07-11** at **07:15:12**
> 
> Oh, I see. That's why it is fully loaded in VRAM. Very impressive.
> 
> Can one get 800 t/s PP and 40+ t/s TG with any of llama.cpp, KTransformers, vLLM, sglang, ... with this setup?
> 
> @ubergarm If you are targeting a fully offloaded setup, isn't `IQ2_KT` the best option? It beets `IQ2_XXS` and `IQ2_KS` in terms of PPL and GPU performance.

> 👤 **ubergarm** replied on **2025-07-11** at **14:55:34**
> 
> @ikawrakow 
> 
> > Can one get 800 t/s PP and 40+ t/s TG with any of llama.cpp, KTransformers, vLLM, sglang, ... with this setup?
> 
> eous [previously submitted llama-sweep-bench results here](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/153) for mainline llama.cpp running the slightly larger similar quality `DeepSeek-R1-0528-UD-IQ1_S` and was peaking out around ~450 tok/sec PP and almost ~50 tok/sec TG.
> 
> > If you are targeting a fully offloaded setup, isn't IQ2_KT the best option? It beets IQ2_XXS and IQ2_KS in terms of PPL and GPU performance.
> 
> I was thinking hard about that `IQ2_KT` and believe its 2.125 bpw is about right compared to the ~2.1025 blend of IQ2_KS down + IQ2_XXS (gate|up). IQ2_KT is the fastest for PP as I recall, with IQ2_KS just behind. I just wasn't sure about TG performance however as I don't recall a recent comparison for full CUDA offload.
> 
> The rig is already setup with some time available today so I'll give it a try adjusting the attn/shexp to use similar BPW `KT` quants as well. I'll leave that output "head" at iq5_k though I suppose.
> 
> It will take a bit longer to cook and calculate perplexity as I can't offload it all, but I'm too curious now not to try! Thanks!
> 
> PS. I'm still not sure the best way to handle that odd shaped `attn_k_b.*=q4_0`... It could go to `iq4_nl` but I'm honestly not even sure if it is actually used or if the corresponding versions of that tensor are used.

> 👤 **ikawrakow** replied on **2025-07-11** at **15:21:02**
> 
> `IQ2_KT` TG performance on CUDA is pretty good, at least on my RTX-4080. It is in the same ballpark as `IQ2_XXS/IQ2_KS`. 
> 
> The `attn_k_b` and `attn_v_b` tensors get used for TG. The `attn_kv_b` tensors that `ik_llama.cpp` creates on-the-fly are used for PP (when MLA = 2, 3). To avoid potential accuracy loss due to re-quantization, the `attn_kv_b` tensors get created as `Q8_0`. 
> 
> Surprised to see `llama.cpp` pulling ahead for TG. I guess one needs to see the exact compositions of these models as theirs may be larger on disk, but use fewer bits during inference.
> 
> What about KTransformers? They for sure can do `IQ1_S` after copy/pasting it from here.

> 👤 **ubergarm** replied on **2025-07-11** at **21:21:31**
> 
> @ikawrakow 
> 
> > The attn_k_b and attn_v_b tensors get used for TG. The attn_kv_b tensors that ik_llama.cpp creates on-the-fly are used for PP (when MLA = 2, 3). To avoid potential accuracy loss due to re-quantization, the attn_kv_b tensors get created as Q8_0.
> 
> Interesting, so perhaps I should modify my recipes to make `attn_k_b` and `attn_v_b` larger e.g. q8_0 and try to prune off or shrink the `attn_kv_b` as it is not even used with ik_llama.cpp mla=2/3 then? I've seen some folks suggest offloading it to CPU to free up a little more VRAM...
> 
> > IQ2_KT TG performance on CUDA is pretty good, at least on my RTX-4080. It is in the same ballpark as IQ2_XXS/IQ2_KS.
> 
> Yeah for full offload I believe IQ2_KT will be the way to go. While I'm only able to offload about half the model, still competitive performance despite the trellis running on CPU during TG. Maybe @eous can try the IQ2_KT fully offloaded on those 2x 6000 PRO blackwells for likely now the best available perplexity and speed combination.
> 
> <img width="4176" height="2217" alt="sweep-bench-TNG-R1T2-Chimera-IQ2_KT-vs-IQ2_XXS" src="https://github.com/user-attachments/assets/7f7d0e93-d220-4c48-bfd9-bf6538ac19d2" />
> 
> <details>
> 
> <summary>👈 llama-sweep-bench command and data</summary>
> 
> ```bash
> model=/mnt/raid/hf/DeepSeek-TNG-R1T2-Chimera-GGUF/IQ2_KT/DeepSeek-TNG-R1T2-Chimera-IQ2_KT-00001-of-00004.gguf
> #model=/mnt/raid/hf/DeepSeek-TNG-R1T2-Chimera-GGUF/IQ2_XXS/DeepSeek-TNG-R1T2-Chimera-IQ2_XXS-00001-of-00004.gguf
> 
> ./build/bin/llama-sweep-bench \
>     --model "$model" \
>     --no-mmap \
>     --ctx-size 12288 \
>     -ctk q8_0 \
>     -fa -fmoe \
>     -mla 3 -amb 512 \
>     -ngl 99 \
>     -ot "blk\.(3|4|5|6|7|8|9|10|11|12|13|14|15)\.ffn_.*=CUDA0" \
>     -ot "blk\.(16|17|18|19|20|21|22|23|24|25|26|27|28)\.ffn_.*=CUDA1" \
>     -ot exps=CPU \
>     -ub 4096 -b 4096 \
>     --warmup-batch \
>     --threads 24
> ```
> 
> ## IQ2_KT 171.146 GiB (2.188 BPW) +26 exps offload PPL=3.8887
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |    9.737 |   420.64 |   76.938 |    13.31 |
> |  4096 |   1024 |   4096 |   11.808 |   346.89 |   78.850 |    12.99 |
> |  4096 |   1024 |   8192 |   14.321 |   286.02 |   82.925 |    12.35 |
> 
> ## IQ2_XXS 169.590 GiB (2.168 BPW) +26 exps offload PPL=4.0078
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |    9.864 |   415.27 |   64.423 |    15.90 |
> |  4096 |   1024 |   4096 |   12.038 |   340.27 |   67.079 |    15.27 |
> |  4096 |   1024 |   8192 |   14.536 |   281.79 |   71.132 |    14.40 |
> 
> </details>
> 
> *UPDATE* two great reports of running this IQ2_KT fully offloaded: https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/296

---

👤 **magikRUKKOLA** commented on **2025-07-10** at **21:31:25**

MOVED: https://github.com/ikawrakow/ik_llama.cpp/discussions/258#discussioncomment-13726226

> 👤 **ubergarm** replied on **2025-07-10** at **23:41:21**
> 
> @magikRUKKOLA 
> 
> Thanks for bringing the discussion over here, explaining your goal of running as much context as possible up to 160k (model max) on the least VRAM possible, and showing your hardware setup.
> 
> > hence the for the full context in ik_llama.cpp its required to have at least 48 GB VRAM which is not ideal.
> 
> I'm not sure how you came to this conclusion? I just ran [ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF/IQ2_KS](https://huggingface.co/ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF) at full 160k context using only 13830MiB VRAM with q8_0 quantized kv-cache... The TG speeds are suffering a bit because I'm not offloading any layers/weights to GPU, but if I were to really run this I'd optimize by offloading some more layers to fill remaining VRAM and increasing `-ub 4096 -b 4096` etc...
> 
> <details>
> 
> <summary>👈How to run 160k context in under 14GB VRAM + ~200GB RAM</summary>
> 
> 
> ```bash
> export model=/mnt/raid/hf/DeepSeek-TNG-R1T2-Chimera-GGUF/IQ2_KS/DeepSeek-TNG-R1T2-Chimera-IQ2_KS-00001-of-00005.gguf
> CUDA_VISIBLE_DEVICES="0" \
> ./build/bin/llama-server \
>     --model "$model" \
>     --alias ubergarm/DeepSeek-TNG-R1T2-Chimera-IQ2_KS \
>     -fa \
>     -mla 3 -fmoe -amb 512 \
>     --ctx-size 163840 \
>     -ctk q8_0 \
>     -ngl 0 \
>     --parallel 1 \
>     --threads 24 \
>     --host 127.0.0.1 \
>     --port 8080
> .
> .
> .
> 
>   Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
> llm_load_tensors: ggml ctx size =    0.47 MiB
> llm_load_tensors: offloading 0 repeating layers to GPU
> llm_load_tensors: offloaded 0/62 layers to GPU
> llm_load_tensors:        CPU buffer size = 42314.45 MiB
> llm_load_tensors:        CPU buffer size = 42634.02 MiB
> llm_load_tensors:        CPU buffer size = 42634.02 MiB
> llm_load_tensors:        CPU buffer size = 42634.02 MiB
> llm_load_tensors:        CPU buffer size = 38222.26 MiB
> ....................................................................................................
> llama_new_context_with_model: n_ctx      = 163840
> llama_new_context_with_model: n_batch    = 2048
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 3
> llama_new_context_with_model: attn_max_b = 512
> llama_new_context_with_model: fused_moe  = 1
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:  CUDA_Host KV buffer size =  5833.12 MiB
> llama_new_context_with_model: KV self size  = 5833.12 MiB, c^KV (q8_0): 5833.12 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     0.99 MiB
> llama_new_context_with_model:      CUDA0 compute buffer size = 13569.14 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =   334.01 MiB
> ```
> 
> </details>
> 
> So you have 3x 3090s and how much RAM? You can easily achieve full 160k context while offloading additional layers for max PP and TG speeds.

> 👤 **magikRUKKOLA** replied on **2025-07-10** at **23:48:40**
> 
> @ubergarm 
> 
> > I'm not sure how you came to this conclusion?
> 
> Uh oh.  I assumed that the first three layers are always getting loaded onto the main gpu. :)
> 
> > So you have 3x 3090s and how much RAM?
> 
> 512 GB RAM

> 👤 **ubergarm** replied on **2025-07-10** at **23:52:42**
> 
> lmao so sorry, I realized after refreshing that it was moved over *there* so replied there for the next step! xD
> 
> yeah you have plenty of ram and VRAM, we can get u going 160k context no problemo

---

👤 **magikRUKKOLA** commented on **2025-07-14** at **15:07:55**

Lets update the perplexity vs llm size graph.  I suggest we use svg.

[EDIT]: this is an old graph.  But it contains the code of the generator in the details.  The latest version of the code will be here but the most up-to-date graphs could be elsewhere.  For example, for the Deepseek-R1-0528 its here: https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13779135
And for the Kimi-K2 is here: https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13776504

with qr codes [following[ to the huggingface['s short-version domain name hf.co (to save on the QR data)]:
![ppl-log](https://github.com/user-attachments/assets/3709b863-ba89-43c1-ae8f-cd951757bedf)



[INSTRUCTIONS TO GENETATE SVG]
* the colours for the figures are generated deterministically, via the name of the quant in the config.
* the trendline goes via the pareto-optimal quants.

<details>
To generate the svg graph perplexity vs llm size keep the data in config.json:

```
{
  "title": "DeepSeek-R1-0528 (671B) Quantization Analysis",
  "subtitle": "Lower perplexity = Better performance",
  "model_parameters": 671000000000,
  "data": [
    {"name": "IQ1_S_R4", "bpw": 1.664, "ppl": 4.8831, "url": "https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ1_S_R4"},
    {"name": "IQ2_KT", "bpw": 2.514, "ppl": 3.6378},
    {"name": "IQ2_K_R4", "bpw": 2.799, "ppl": 3.5069, "url": "https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ2_K_R4"},
    {"name": "UD_Q2_K_XL", "bpw": 2.994, "ppl": 3.5278, "url": "https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/UD-Q2_K_XL"},
    {"name": "IQ3_KT", "bpw": 3.483, "ppl": 3.3056, "url": "https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ3_KT"},
    {"name": "IQ3_KS", "bpw": 3.598, "ppl": 3.2991, "url": "https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ3_KS"},
    {"name": "IQ3_K_R4", "bpw": 3.847, "ppl": 3.2730, "url": "https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ3_K_R4"},
    {"name": "q4_0", "bpw": 4.508, "ppl": 3.2895, "url": "https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/Q4_0"},
    {"name": "IQ4_XS (unsloth)", "bpw": 4.2683, "ppl": 3.2598, "url": "https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/IQ4_XS"},
    {"name": "UD_Q4_K_XL", "bpw": 4.578, "ppl": 3.2483, "url": "https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/UD-Q4_K_XL"},
    {"name": "IQ4_KS_R4", "bpw": 4.701, "ppl": 3.2286, "url": "https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ4_KS_R4"},
    {"name": "DQ4_K_R4", "bpw": 5.289, "ppl": 3.2276, "url": "https://huggingface.co/anikifoss/DeepSeek-R1-0528-DQ4_K_R4"},
    {"name": "Q8_0", "bpw": 8.5259260, "ppl": 3.2130, "url": "https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/Q8_0"}
  ]
}

```

and use the make.sh script ( ./make.sh --logscale config.json > ppl-log.svg ) to generate the svg file:
```bash
#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# 1.  CLI
# ------------------------------------------------------------------
logscale=0
[[ $# -ge 1 && $1 == "--logscale" ]] && { logscale=1; shift; }
[[ $# -eq 1 ]] || { echo "Usage: $0 [--logscale] config.json" >&2; exit 1; }
config=$1
[[ -f $config ]] || { echo "Config file not found" >&2; exit 1; }

# ------------------------------------------------------------------
# 2.  QR-codes  (never touch stdin)
# ------------------------------------------------------------------
qr_dir="qrcodes"
mkdir -p "$qr_dir"
while IFS= read -r url; do
    [[ -z $url ]] && continue
    short=${url//https:\/\/huggingface.co\//hf.co/}
    hash=$(printf '%s' "$short" | md5sum | awk '{print $1}')
    file="$qr_dir/$hash.svg"
    [[ -f $file ]] && continue
    tmp="$qr_dir/${hash}_tmp.svg"
    qrencode --inline -t svg -l L -s 1 -m 0 "$short" -o "$tmp"
    svgo --multipass -q "$tmp" -o "$file" 2>/dev/null
    rm -f "$tmp"
done < <(jq -r '.data[] | select(.url) | .url' "$config")

# ------------------------------------------------------------------
# 3.  Pre-compute .size and limits
# ------------------------------------------------------------------
mp=$(jq -r '.model_parameters' "$config")
min_ppl=$(jq -r '.data | min_by(.ppl).ppl' "$config")
max_ppl=$(jq -r '.data | max_by(.ppl).ppl' "$config")

sizes=()
while IFS= read -r size; do
    sizes+=("$size")
done < <(jq -r --arg mp "$mp" '.data[] | .bpw * ($mp|tonumber) / 8 / 1024 / 1024 / 1024 | round * 1.0' "$config")

max_sz=0
for size in "${sizes[@]}"; do
    if (( $(echo "$size > $max_sz" | bc -l) )); then
        max_sz=$size
    fi
done
max_round=$(awk -v m="$max_sz" 'BEGIN{r=int((m+63)/64)*64; print (r<64?64:r)}')

title=$(jq -r '.title // "Quantization Analysis"' "$config")
subtitle=$(jq -r '.subtitle // "Lower perplexity = better"' "$config")
[[ $logscale -eq 1 ]] && subtitle+=" (log-difference scale)"

if [[ $logscale -eq 1 ]]; then
    rng=$(awk -v min="$min_ppl" -v max="$max_ppl" 'BEGIN{print max-min}')
    eps=$(awk -v r="$rng" 'BEGIN{print r/100}')
    t_min=$(awk -v e="$eps" 'BEGIN{print log(e)/log(10)}')
    t_range=$(awk -v min="$min_ppl" -v max="$max_ppl" -v e="$eps" \
        'BEGIN{tmax=log(max-min+e)/log(10); print tmax-log(e)/log(10)}')
else
    ppl_rng=$(awk -v min="$min_ppl" -v max="$max_ppl" 'BEGIN{print max-min}')
fi

# ------------------------------------------------------------------
# 4.  Pareto indices
# ------------------------------------------------------------------
pareto_i=()
item_count=$(jq '.data | length' "$config")

for ((i=0; i<item_count; i++)); do
    item=$(jq --argjson i "$i" '.data[$i]' "$config")
    bpw=$(jq -r '.bpw' <<<"$item")
    ppl=$(jq -r '.ppl' <<<"$item")
    size=$(bc <<< "scale=4; $bpw * $mp / 8 / 1024 / 1024 / 1024")
    size=$(printf "%.1f" "$size")

    is_pareto=1
    for ((j=0; j<item_count; j++)); do
        [[ $j -eq $i ]] && continue
        j_item=$(jq --argjson j "$j" '.data[$j]' "$config")
        j_bpw=$(jq -r '.bpw' <<<"$j_item")
        j_ppl=$(jq -r '.ppl' <<<"$j_item")
        j_size=$(bc <<< "scale=4; $j_bpw * $mp / 8 / 1024 / 1024 / 1024")
        j_size=$(printf "%.1f" "$j_size")

        if (( $(echo "$j_ppl <= $ppl" | bc -l) && $(echo "$j_size <= $size" | bc -l) )); then
            if (( $(echo "$j_ppl < $ppl" | bc -l) || $(echo "$j_size < $size" | bc -l) )); then
                is_pareto=0
                break
            fi
        fi
    done
    [[ $is_pareto -eq 1 ]] && pareto_i+=("$i")
done

# ------------------------------------------------------------------
# 5.  SVG header & grid
# ------------------------------------------------------------------
top=100; h=400; gap=50
leg_h=$((70 + item_count * 40))
tot=$((top + h + gap + leg_h + 50))

cat <<EOF
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 $tot">
<defs>
    <radialGradient id="halo" cx="50%" cy="50%" r="50%">
        <stop offset="0%"  stop-color="#00c853" stop-opacity="0.20"/>
        <stop offset="100%" stop-color="#00c853" stop-opacity="0"/>
    </radialGradient>
</defs>
<style>
    .axis{stroke:#555;stroke-width:1.5}
    .grid{stroke:#eee;stroke-width:.5}
    .label{font:14px sans-serif;fill:#333}
    .title{font:bold 18px sans-serif;fill:#111}
    .triangle{stroke-width:1.5}
    .legend-item{font:12px monospace}
    .legend-title{font:bold 13px sans-serif}
</style>
<rect width="100%" height="100%" fill="white"/>
<text class="title" x="400" y="30" text-anchor="middle">$title</text>
<text class="label" x="400" y="55" text-anchor="middle" fill="#666">$subtitle</text>
<line x1="100" y1="100" x2="100" y2="500" class="axis"/>
<line x1="100" y1="500" x2="700" y2="500" class="axis"/>
<text class="label" x="400" y="540" text-anchor="middle">Model Size (GB)</text>
<text class="label" x="20" y="300" text-anchor="middle" transform="rotate(-90,20,300)">Perplexity (lower is better)</text>
EOF

# Y-axis grid
if [[ $logscale -eq 1 ]]; then
    for i in {0..4}; do
        frac=$(awk -v i="$i" 'BEGIN{print i/4}')
        ppl=$(awk -v min="$min_ppl" -v eps="$eps" -v tr="$t_range" -v f="$frac" \
              'BEGIN{printf "%.3f", min + 10**(log(eps)/log(10) + f*tr) - eps}')
        y=$(awk -v f="$frac" 'BEGIN{printf "%.1f",500-400*f}')
        text_y=$(awk -v y="$y" 'BEGIN{printf "%.1f", y+5}')
        echo "    <line x1=\"100\" y1=\"$y\" x2=\"700\" y2=\"$y\" class=\"grid\"/>"
        echo "    <text class=\"label\" x=\"80\" y=\"$text_y\" text-anchor=\"end\">$ppl</text>"
    done
else
    for i in {0..4}; do
        ppl=$(awk -v min="$min_ppl" -v max="$max_ppl" -v i="$i" -v r="$ppl_rng" \
              'BEGIN{printf "%.1f",max-i*r/4}')
        y=$((100+i*100))
        text_y=$((y+5))
        echo "    <line x1=\"100\" y1=\"$y\" x2=\"700\" y2=\"$y\" class=\"grid\"/>"
        echo "    <text class=\"label\" x=\"80\" y=\"$text_y\" text-anchor=\"end\">$ppl</text>"
    done
fi

# X-axis grid
for i in $(seq 0 64 "$max_round"); do
    x=$(awk -v s="$i" -v mr="$max_round" 'BEGIN{printf "%.1f",100+(s/mr)*600}')
    echo "    <line x1=\"$x\" y1=\"100\" x2=\"$x\" y2=\"500\" class=\"grid\"/>"
    [[ $((i%256)) -eq 0 || $i -eq $max_round ]] && \
        echo "    <text class=\"label\" x=\"$x\" y=\"520\" text-anchor=\"middle\">$i</text>"
done

# ------------------------------------------------------------------
# 6.  Helpers
# ------------------------------------------------------------------
to_xy() {
    local sz=$1 pl=$2 x y
    x=$(awk -v s="$sz" -v mr="$max_round" 'BEGIN{printf "%.1f",100+(s/mr)*600}')
    if [[ $logscale -eq 1 ]]; then
        y=$(awk -v p="$pl" -v min="$min_ppl" -v eps="$eps" \
            -v tmin="$(awk -v e="$eps" 'BEGIN{print log(e)/log(10)}')" -v tr="$t_range" \
            'BEGIN{d=p-min;printf "%.1f",500-400*((log(d+eps)/log(10)-tmin)/tr)}')
    else
        y=$(awk -v p="$pl" -v min="$min_ppl" -v r="$ppl_rng" \
            'BEGIN{printf "%.1f",500-(p-min)*400/r}')
    fi
    echo "$x $y"
}

trend="M"
sorted_pareto_i=($(for i in "${pareto_i[@]}"; do
    item=$(jq --argjson i "$i" '.data[$i]' "$config")
    bpw=$(jq -r '.bpw' <<<"$item")
    size=$(bc <<< "scale=4; $bpw * $mp / 8 / 1024 / 1024 / 1024")
    size=$(printf "%.1f" "$size")
    echo "$size $i"
done | sort -n | awk '{print $2}'))

for i in "${sorted_pareto_i[@]}"; do
    item=$(jq --argjson i "$i" '.data[$i]' "$config")
    bpw=$(jq -r '.bpw' <<<"$item")
    ppl=$(jq -r '.ppl' <<<"$item")
    size=$(bc <<< "scale=4; $bpw * $mp / 8 / 1024 / 1024 / 1024")
    size=$(printf "%.1f" "$size")
    read x y < <(to_xy "$size" "$ppl")
    trend+=" $x $y"
done

# ------------------------------------------------------------------
# 7.  Draw points (ascending ppl)
# ------------------------------------------------------------------
sorted_indices=($(jq -r '.data | sort_by(.ppl) | keys_unsorted[]' "$config"))
ly=$((top + h + gap + 70))

for idx in "${sorted_indices[@]}"; do
    item=$(jq --argjson i "$idx" '.data[$i]' "$config")
    name=$(jq -r '.name' <<<"$item")
    bpw=$(jq -r '.bpw' <<<"$item")
    ppl=$(jq -r '.ppl' <<<"$item")
    size=$(bc <<< "scale=4; $bpw * $mp / 8 / 1024 / 1024 / 1024")
    size=$(printf "%.1f" "$size")
    url=$(jq -r '.url // ""' <<<"$item")
    read x y < <(to_xy "$size" "$ppl")

    xl=$(awk -v x="$x" 'BEGIN{printf "%.1f",x-10}')
    yt=$(awk -v y="$y" 'BEGIN{printf "%.1f",y-10}')
    xr=$(awk -v x="$x" 'BEGIN{printf "%.1f",x+10}')

    c=$(printf '%s' "$name" | md5sum | awk '{print "#"substr($1,1,6)}')
    dc=$(printf '%s' "${c#?}" | awk '{printf "#%02x%02x%02x", strtonum("0x"substr($0,1,2))*8/10, strtonum("0x"substr($0,3,2))*8/10, strtonum("0x"substr($0,5,2))*8/10}')

    is_pareto=0
    for i in "${pareto_i[@]}"; do
        [[ $i -eq $idx ]] && { is_pareto=1; break; }
    done

    if [[ $is_pareto -eq 1 ]]; then
        echo "    <circle cx=\"$x\" cy=\"$y\" r=\"14\" fill=\"url(#halo)\"/>"
        echo "    <polygon points=\"$x,$y $xl,$yt $xr,$yt\" class=\"triangle\" fill=\"$c\" stroke=\"$dc\"/>"
    else
        echo "    <polygon points=\"$x,$y $xl,$yt $xr,$yt\" class=\"triangle\" fill=\"$c\" stroke=\"$dc\"/>"
    fi

    qr=""
    if [[ -n $url ]]; then
        short=${url//https:\/\/huggingface.co\//hf.co/}
        hsh=$(printf '%s' "$short" | md5sum | awk '{print $1}')
        [[ -f $qr_dir/$hsh.svg ]] && qr_base64=$(base64 -w 0 "$qr_dir/$hsh.svg" 2>/dev/null || base64 "$qr_dir/$hsh.svg" | tr -d '\n')
        qr="<image x=\"450\" y=\"$((ly-10))\" width=\"32\" height=\"32\" href=\"data:image/svg+xml;base64,$qr_base64\"/>"
    fi

    points+="\n        <polygon points=\"70,$ly 60,$((ly+10)) 80,$((ly+10))\" fill=\"$c\""
    if [[ $is_pareto -eq 0 ]]; then
        points+=" stroke=\"#ff000050\" stroke-width=\"2\""
    else
        points+=" stroke=\"$dc\""
    fi
    points+="/>"
    points+="\n        <text class=\"legend-item\" x=\"100\" y=\"$((ly+10))\">$name: $bpw bpw, $ppl ppl</text>"
    [[ -n $qr ]] && points+="\n        $qr"
    ly=$((ly+40))
done
points=$(echo -e "$points")

# ------------------------------------------------------------------
# 8.  Trendline & legend
# ------------------------------------------------------------------
[[ ${#trend} -gt 1 ]] && \
    echo "    <path d=\"$trend\" fill=\"none\" stroke=\"#00c853\" stroke-width=\"1.5\" stroke-dasharray=\"6,3\" stroke-opacity=\"0.5\"/>"

cat <<EOF
    <rect x="50" y="$((top + h + gap))" width="700" height="$((leg_h-5))" fill="#f8fafc" stroke="#e2e8f0" rx="5"/>
    <text class="legend-title" x="70" y="$((top + h + gap + 25))">Quantization Details</text>
    <g class="legend">$points
    </g>
</svg>
EOF

```
<details>

> 👤 **ikawrakow** replied on **2025-07-14** at **15:11:10**
> 
> My recommendation would be to use a log scale for perplexity, else you see nothing when you add low-but quants and expand the plot range accordingly.

> 👤 **magikRUKKOLA** replied on **2025-07-15** at **06:40:15**
> 
> @ikawrakow okay cool, its done.  The json and bash files to generate the graph are provided.

> 👤 **saood06** replied on **2025-07-15** at **06:49:46**
> 
> Can you add the PPL values reported in the first post specifically:
> > `DeepSeek-R1-0528-Q8_0` 666GiB
> > `Final estimate: PPL = 3.2130 +/- 0.01698`
> 
> > `DeepSeek-R1-0528-IQ3_K_R4`  301GiB
> >  `Final estimate: PPL = 3.2730 +/- 0.01738`
> 
> > `DeepSeek-R1-0528-IQ2_K_R4`  220GiB
> >  `Final estimate: PPL = 3.5069 +/- 0.01893`
> 
> Q8_0 I think is generic, but his IQ3_K_R4 and IQ2_K_R4 are mixes so should be marked like UD is.
> 
> Having Q8_0 adds a reference point.

> 👤 **magikRUKKOLA** replied on **2025-07-15** at **07:01:08**
> 
> > but his IQ3_K_R4 and IQ2_K_R4 are mixes so should be marked like UD is.
> 
> UD?  I thought is stands for "unsloth dynamic".  No?

> 👤 **saood06** replied on **2025-07-15** at **07:04:10**
> 
> I meant in a manner similar. "UG" could work, but pick another shorthand if you have one in mind.

> 👤 **magikRUKKOLA** replied on **2025-07-15** at **07:04:31**
> 
> > Can you add the PPL values reported in the first post specifically:
> 
> Feel free to build the graph yourself.  See the details above. :)

> 👤 **magikRUKKOLA** replied on **2025-07-15** at **07:06:13**
> 
> > I meant in a manner similar. "UG" could work, but pick another shorthand if you have one in mind.
> 
> Feel free to provide the link to the naming convention of the quants etc. :)

> 👤 **saood06** replied on **2025-07-15** at **07:07:18**
> 
> >Feel free to build the graph yourself. See the details above. :)
> 
> Thank you for adding them.

> 👤 **saood06** replied on **2025-07-15** at **07:12:04**
> 
> > > I meant in a manner similar. "UG" could work, but pick another shorthand if you have one in mind.
> > 
> > Feel free to provide the link to the naming convention of the quants etc. :)
> 
> I think mixes that aren't generated when passing the quant name to generate it should be marked different from ones that are, how that is accomplished (naming is just one way to do it) I don't mind. You do mark Unsloth custom recipes with a name, which is why I suggested a name for this. But again this is just my opinion on how things should be represented.

> 👤 **ikawrakow** replied on **2025-07-15** at **07:44:57**
> 
> It is not so that Unsloth invented "dynamic" quants. I added the ability to use different bpw for the various tensors in a model in the initial `llama.cpp` [k-quants commit](https://github.com/ggml-org/llama.cpp/pull/1684) (and in fact, at some point someone added the `--pure` command line option to `llama-quantize` to be able to have "non-dynamic" k- and i-quants). So, while the entire Internet just knows that Unsloth created "dynamic" quants, I'd rather not have that myth perpetuated in my own repository. There are the Unsloth-specific quantization recipes, Ubergarm-specific quantization recipes, `llama.cpp` default quantization recipes, `ik_llama.cpp` default quantization recipes, etc. There is nothing "dynamic" in Unsloth's quantization [1]. If a recipe is named `IQ3_K_R4`, it basically means that this is the predominant quantization type. One could add a bpw to that (e.g., `IQ3_K_R4_3.844`). `UD_Q2_K_XL` would then become simply `Q2_K_2.994`. If the creators of the quantized models (a.k.a., "quant cooks") prefer to have their names recorded in the model type name, then it could be `IQ3_K_R4_3.844_Ubergarm` and `Q2_K_2.994_Unsloth`. 
> 
> [1] Before Unsloth came along, it wasn't my concept that one needs to have studied physics to know that "dynamic" is something that changes with time. As opposed to "static", which remains the same. So that, to consider a quantization "dynamic", it should somehow change during run time depending on context.

> 👤 **magikRUKKOLA** replied on **2025-07-15** at **08:05:27**
> 
> > It is not so that Unsloth invented "dynamic" quants.
> 
> Yeah, understood.  I was pointing out about the cases such that:
> 
> unsloth/DeepSeek-R1-0528-**IQ4_XS**-00001-of-00008.gguf
> and
> ubergarm/DeepSeek-R1-0528-**IQ3_KS**-00001-of-00007.gguf
> 
> IQ3_**K**S and IQ4_**X**S.  So they look very similar.  Someone can easily confuse where to get the exact quant of interest.  The bpw is already present in the legend and that would not answer the question.
> Apparently the only thing that is left is to append the suffix with the name of an author to the [current name of the quant].  Or the QR code to the huggingface repo?

> 👤 **ikawrakow** replied on **2025-07-15** at **08:19:59**
> 
> > Or the QR code to the huggingface repo?
> 
> A link to the HF repo is of course useful.

> 👤 **magikRUKKOLA** replied on **2025-07-15** at **15:34:46**
> 
> > > Or the QR code to the huggingface repo?
> > 
> > A link to the HF repo is of course useful.
> 
> Ok done.

> 👤 **magikRUKKOLA** replied on **2025-07-16** at **12:03:14**
> 
> ![kimi-log-ppl](https://github.com/user-attachments/assets/987c99df-c010-4405-ba7d-1f04a43bde92)
> 
> 
> ```json
> {
>   "title": "Kimi-K2-Instruct (1026B) Quantization Analysis",
>   "subtitle": "Lower perplexity = Better performance",
>   "model_parameters": 1026000000000,
>   "data": [
>     {"name": "smol-IQ1_KT", "bpw": 1.792, "ppl": 4.3623, "url": "https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF"},
>     {"name": "IQ1_KT", "bpw": 1.915, "ppl": 4.1310, "url": "https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF"},
>     {"name": "UD-IQ1_S", "bpw": 2.192, "ppl": 4.3331, "url": "https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-IQ1_S"},
>     {"name": "IQ2_KS", "bpw": 2.398, "ppl": 3.7922, "url": "https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF/tree/main/IQ2_KS"},
>     {"name": "UD-IQ2_XXS", "bpw": 2.558, "ppl": 3.5258, "url": "https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-IQ2_XXS"},
>     {"name": "IQ2_KL", "bpw": 2.892, "ppl": 3.2741, "url": "https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF/tree/main/IQ2_KL"},
>     {"name": "Q2_K", "bpw": 2.906, "ppl": 4.9829, "url": "https://huggingface.co/gabriellarson/Kimi-K2-Instruct-GGUF"},
>     {"name": "bigattnshexpdense-IQ2_KL", "bpw": 2.923, "ppl": 3.1813, "url": "full q8_0 attn/shexp/blk.0.ffn"},
>     {"name": "chonk-IQ2_KL", "bpw": 3.057, "ppl": 3.2095, "url": "blk.(1|2|3|4|5|6|59|60).ffn_down_exps.weight=iq4_ks and blk.(1|2|3|4|5|6|59|60).ffn_(gate|up)_exps.weight=iq4_kss"},
>     {"name": "UD-IQ3_XXS", "bpw": 3.247, "ppl": 3.1467, "url": "https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-IQ3_XXS"},
>     {"name": "UD-Q3_K_XL", "bpw": 3.524, "ppl": 3.2695, "url": "https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-Q3_K_XL"},
>     {"name": "IQ3_KS", "bpw": 3.573, "ppl": 3.1395, "url": "https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF/tree/main/IQ3_KS"},
>     {"name": "UD-Q4_K_XL", "bpw": 4.581, "ppl": 3.0612, "url": "https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/UD-Q4_K_XL"},
>     {"name": "IQ4_KS", "bpw": 4.604, "ppl": 3.0438, "url": "https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF/tree/main/IQ4_KS"},
>     {"name": "DQ4_K", "bpw": 5.229, "ppl": 2.9691, "url": "https://huggingface.co/anikifoss/Kimi-K2-Instruct-DQ4_K"},
>     {"name": "Q8_0", "bpw": 8.504, "ppl": 2.9507, "url": "https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/tree/main/Q8_0"}
>   ]
> }
> ```

> 👤 **magikRUKKOLA** replied on **2025-07-16** at **12:52:26**
> 
> But why not cook the quants such so it would be close to 256 or 512 GB -- as to put the weights on RAM and KV cache on GPU (for as much longer context as possible)?   Or, it doesn't really work like that?

> 👤 **ubergarm** replied on **2025-07-16** at **13:21:21**
> 
> @magikRUKKOLA nice graphs, thanks for pulling together the data and hf links! Just got the `Kimi-K2-Instruct-IQ2_KS` 286.624 GiB (2.398 BPW) uploaded to https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF
> 
> Final Perplexity just came in: `Final estimate: PPL = 3.7922 +/- 0.02045`
> 
> > But why not cook the quants such so it would be close to 256 or 512 GB -- as to put the weights on RAM and KV cache on GPU (for as much longer context as possible)? Or, it doesn't really work like that?
> 
> I do try to target hardware breakpoints like 256 / 384 / 512 GB RAM assuming some combination of GPUs or not. But there is a wide variety of user hardware configurations that I've seen now doing this a couple months. So I try to strike a balance between general usability, accuracy, and speed given the best quality quants currently available.
> 
> Most importantly I try to keep it fun hehe...

> 👤 **ikawrakow** replied on **2025-07-16** at **13:32:10**
> 
> Yes, thanks for the nice graphs.
> 
> The DeepSeek-R1-0528 graph does show pretty well how Unsloth quants (and even more so `Q4_0`) are not on the Pareto frontier of the quality vs size compromise. 
> 
> If we could somehow get our hands on the PPL of the Unsloth models and put the data on the same graph to see how things stack up there, that would be cool.

> 👤 **ubergarm** replied on **2025-07-16** at **13:39:40**
> 
> I think @Panchovix has been collecting data on various models as well: https://www.reddit.com/r/LocalLLaMA/comments/1lz1s8x/ and might have some more values to fill in the graphs.

> 👤 **magikRUKKOLA** replied on **2025-07-16** at **14:32:08**
> 
> > Just got the `Kimi-K2-Instruct-IQ2_KS`
> 
> cool thanks.  updated

> 👤 **ubergarm** replied on **2025-07-17** at **01:00:39**
> 
> @magikRUKKOLA if you want a couple experimental quants based on ik's new IQ1_KS 1.75 BPW SOTA trellis quant implantation i have the numbers. these are not yet available on HF as its not merged into main and could possibly change. also the KT quants tend to run faster TG on CUDA backend as calculating the trellis on CPU actually breaks the rule of "TG is limited by ram bandwidth" hahah
> 
> * Kimi-K2-Instruct-smol-IQ1_KT 
>   - 214.182 GiB (1.792 BPW)
>   - Final estimate: PPL = 4.3623 +/- 0.02432
> * Kimi-K2-Instruct-IQ1_KT
>   - 228.948 GiB (1.915 BPW)
>   - Final estimate: PPL = 4.1310 +/- 0.02266
> 
> the -smol here is how i indicate the ffn_down_exps was also IQ1_KT same size as the ffn_(up|gate)_exps. The "normal" IQ1_KT used IQ2_KT for ffn_down_exps as i usually would do.

> 👤 **ubergarm** replied on **2025-07-17** at **23:16:44**
> 
> @magikRUKKOLA 
> 
> Hey curious where you got that `Kimi-K2-Instruct-UD-IQ3_XXS` perplexity from? I was trying to get data on their Kimi-K2-Instruct-UD-IQ1_S but its broken, but got a tip to disable `-fmoe` which got it running perplexity correctly again. I think I cc'd you over ont hat thread too hah sorry so many tiny little comment boxes to get lost in!
> 
> I'll share more data on what I find! thanks!

> 👤 **magikRUKKOLA** replied on **2025-07-17** at **23:31:17**
> 
> > Hey curious where you got that `Kimi-K2-Instruct-UD-IQ3_XXS` perplexity from?
> 
> from ik_llama.cpp as usual

> 👤 **magikRUKKOLA** replied on **2025-07-19** at **19:53:40**
> 
> > The DeepSeek-R1-0528 graph does show pretty well how Unsloth quants (and even more so `Q4_0`) are not on the Pareto frontier of the quality vs size compromise.
> 
> Ok cool, now the trendline goes only via the Pareto-compatible quants (and the non-Pareto quants are highlighted in reddish border in the legend).

---

👤 **magikRUKKOLA** commented on **2025-07-16** at **15:56:08**

R1 stats (THIREUS quants added).

![r1-0528-ppl-log](https://github.com/user-attachments/assets/114e5b25-e46b-4c9d-a7b5-920acc3d1640)


```json
{
  "title": "DeepSeek-R1-0528 (671B) Quantization Analysis",
  "subtitle": "Lower perplexity = Better performance",
  "model_parameters": 671000000000,
  "data": [
    {"name": "IQ1_S_R4", "bpw": 1.664, "ppl": 4.8831, "url": "https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ1_S_R4"},
    {"name": "THIREUS-1.9364", "bpw": 1.9364, "ppl": 4.3533, "url": "https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-1.9364bpw-4.3533ppl.151GB-GGUF_11GB-GPU_140GB-CPU.3c88ec6_9fd615d.recipe"},
    {"name": "IQ2_KT", "bpw": 2.514, "ppl": 3.6378},
    {"name": "THIREUS-2.7840", "bpw": 2.7840, "ppl": 3.4341, "url": "https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-2.7840bpw-3.4341ppl.217GB-GGUF_14GB-GPU_203GB-CPU.3c88ec6_02247be.recipe"},
    {"name": "IQ2_K_R4", "bpw": 2.799, "ppl": 3.5069, "url": "https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ2_K_R4"},
    {"name": "JWNoctis/R1-0528/IQ2_KL", "bpw": 2.930, "ppl": 3.4379, "url": "https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/354"},
    {"name": "UD_Q2_K_XL", "bpw": 2.994, "ppl": 3.5278, "url": "https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/UD-Q2_K_XL"},
    {"name": "THIREUS-3.1027", "bpw": 3.1027, "ppl": 3.3372, "url": "https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-3.1027bpw-3.3372ppl.242GB-GGUF_11GB-GPU_231GB-CPU.3c88ec6_adc8101.recipe"},
    {"name": "THIREUS-3.1446", "bpw": 3.1446, "ppl": 3.3257, "url": "https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-3.1446bpw-3.3257ppl.246GB-GGUF_15GB-GPU_231GB-CPU.3c88ec6_7d1efe1.recipe"},
    {"name": "THIREUS-3.1447", "bpw": 3.1447, "ppl": 3.3269, "url": "https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-3.1447bpw-3.3269ppl.246GB-GGUF_15GB-GPU_231GB-CPU.3c88ec6_4b1254a.recipe"},
    {"name": "THIREUS-3.1525", "bpw": 3.1525, "ppl": 3.3251, "url": "https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-3.1525bpw-3.3251ppl.246GB-GGUF_15GB-GPU_231GB-CPU.3c88ec6_5a3fc0f.recipe"},
    {"name": "THIREUS-3.1740", "bpw": 3.1740, "ppl": 3.3253, "url": "https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-3.1740bpw-3.3253ppl.248GB-GGUF_17GB-GPU_231GB-CPU.3c88ec6_6cf3a72.recipe"},
    {"name": "THIREUS-3.1858", "bpw": 3.1858, "ppl": 3.3261, "url": "https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-3.1858bpw-3.3261ppl.249GB-GGUF_18GB-GPU_231GB-CPU.3c88ec6_027b7ff.recipe"},
    {"name": "THIREUS-3.2564", "bpw": 3.2564, "ppl": 3.2985, "url": "https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-3.2564bpw-3.2985ppl.254GB-GGUF_15GB-GPU_239GB-CPU.3c88ec6_7c0be1e.recipe"},
    {"name": "IQ3_KT", "bpw": 3.483, "ppl": 3.3056, "url": "https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ3_KT"},
    {"name": "THIREUS-3.5652", "bpw": 3.5652, "ppl": 3.2734, "url": "https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-3.5652bpw-3.2734ppl.278GB-GGUF_14GB-GPU_264GB-CPU.3c88ec6_9b5660b.recipe"},
    {"name": "IQ3_KS", "bpw": 3.598, "ppl": 3.2991, "url": "https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ3_KS"},
    {"name": "THIREUS-3.6766", "bpw": 3.6766, "ppl": 3.2741, "url": "https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13781700"},
    {"name": "IQ3_K_R4", "bpw": 3.847, "ppl": 3.2730, "url": "https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ3_K_R4"},
    {"name": "THIREUS-3.976", "bpw": 3.976, "ppl": 3.2452, "url": "https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13798329"},
    {"name": "IQ4_XS (unsloth)", "bpw": 4.2683, "ppl": 3.2598, "url": "https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/IQ4_XS"},
    {"name": "q4_0", "bpw": 4.508, "ppl": 3.2895, "url": "https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/Q4_0"},
    {"name": "UD_Q4_K_XL", "bpw": 4.578, "ppl": 3.2483, "url": "https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/UD-Q4_K_XL"},
    {"name": "IQ4_KS_R4", "bpw": 4.701, "ppl": 3.2286, "url": "https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/tree/main/IQ4_KS_R4"},
    {"name": "DQ4_K_R4", "bpw": 5.289, "ppl": 3.2276, "url": "https://huggingface.co/anikifoss/DeepSeek-R1-0528-DQ4_K_R4"},
    {"name": "THIREUS-6.2478", "bpw": 6.2478, "ppl": 3.2240, "url": "https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13781560"},
    {"name": "Q8_0", "bpw": 8.5259260, "ppl": 3.2130, "url": "https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF/tree/main/Q8_0"}
  ]
}


```

> 👤 **Panchovix** replied on **2025-07-16** at **17:17:17**
> 
> Those Thireus ones look pretty impressive, are they posted somewhere? Do they work on lcpp or only on iklcpp?

> 👤 **magikRUKKOLA** replied on **2025-07-16** at **18:51:48**
> 
> > Those Thireus ones look pretty impressive
> 
> If you have only 256GB RAM?  Well, yeah.  But for me the clear winner is IQ4_KS_R4.  Its relatively fast and pretty precise.
> 
> [EDIT]:
> But ... THIREUS-3.5652 looks nice if you do have 512GB and want a longer context.  I need to test if it handles 160k with 3 x 24GB GPU or not.

> 👤 **Panchovix** replied on **2025-07-16** at **18:53:08**
> 
> I have just total 400GB between ram and VRAM, so can't quite run that model.

> 👤 **Thireus** replied on **2025-07-16** at **19:01:13**
> 
> @magikRUKKOLA, what is you available RAM and VRAM?
> @Panchovix, what is your available RAM and VRAM?

> 👤 **Panchovix** replied on **2025-07-16** at **19:03:23**
> 
> I have 192GB RAM (but about 180GB usable) and 208GB VRAM (about 180GB usage because the multigpu overhead), between 7 GPUs. For example I can load iq4_XS which weights 333GB. Maybe my limit is 340GB on weights or so.

> 👤 **magikRUKKOLA** replied on **2025-07-16** at **19:04:47**
> 
> > @magikRUKKOLA, what is you available RAM and VRAM?
> 
> I have 512GB ECC DDR4 (2933 MT/s and 3200 MT/s) with various number of 24GB RTX 3090 (either 2 or 3).
> 
> [EDIT]: will have 4 GPUs as soon as I figure out the water cooling.

> 👤 **Thireus** replied on **2025-07-16** at **19:26:20**
> 
> @magikRUKKOLA - try this [recipe](https://colab.research.google.com/github/Thireus/GGUF-Tool-Suite/blob/c2e1782cb037936d0ce1bbfc075da3d226d6e630/quant_recipe_pipeline.ipynb):
> 
> ```
> ## Quant mix recipe created using Thireus' GGUF Tool Suite - https://gguf.thireus.com/
> # Model name: DeepSeek-R1-0528
> # Link to the original model: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528
> 
> ## Model head & embeddings — qbits: 32 8 
> output_norm\.weight=f32
> token_embd\.weight=q8_0
> output\.weight=q8_0
> 
> ## Special attention kernels — single-quant only (llama-quantize takes care of it) — qbits: 8 
> blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0
> 
> ## Multi-headed attention parameters — qbits: 32 4 
> blk\.([0-9]|[1-5][0-9]|60)\.attn_v_b\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_mqa\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_output\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_b\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_b\.weight=iq4_xs
> 
> ## Core FFN weights — qbits: 32 8 6 5 
> blk\.2\.ffn_gate\.weight=q8_0
> blk\.(0|2)\.ffn_up\.weight=iq6_k
> blk\.([0-9]|[1-5][0-9]|60)\.ffn_norm\.weight=f32
> blk\.[0-1]\.ffn_gate\.weight=iq6_k
> blk\.1\.ffn_down\.weight=iq6_k
> blk\.2\.ffn_down\.weight=iq5_k_r4
> blk\.1\.ffn_up\.weight=iq5_k_r4
> blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_inp\.weight=f32
> blk\.0\.ffn_down\.weight=q8_0
> 
> ## Other tensors — qbits: 32 
> blk\.([3-9]|[1-5][0-9]|60)\.exp_probs_b\.bias=f32
> 
> ## GPU-loaded ffn_*_shexp
> # ffn_down_shexp (down-projection) — qbits: 8 6 5 
> blk\.(11|17|19|29|36|39|44|60|2[6-7]|2[0-4]|3[0-1]|3[3-4])\.ffn_down_shexp\.weight=q8_0
> blk\.([3-8]|10|12|25|28|32|35|3[7-8]|1[4-6]|4[5-9]|4[0-3]|5[0-8])\.ffn_down_shexp\.weight=iq6_k
> blk\.(9|13|18|59)\.ffn_down_shexp\.weight=iq5_k_r4
> 
> # ffn_up_shexp (up-projection) — qbits: 8 6 5 
> blk\.(6|15|18|30|37|39|41|50|54|60|2[1-4]|3[2-4]|2[6-9])\.ffn_up_shexp\.weight=q8_0
> blk\.([3-5]|[8-9]|19|20|25|31|38|40|58|4[2-9]|1[6-7]|1[0-4]|3[5-6]|5[5-6]|5[1-3])\.ffn_up_shexp\.weight=iq6_k
> blk\.(7|57|59)\.ffn_up_shexp\.weight=iq5_k_r4
> 
> # ffn_gate_shexp (gate-projection) — qbits: 8 6 5 
> blk\.(16|20|29|54|60|5[6-8]|5[0-2]|4[1-2]|4[4-9]|1[8-9]|2[3-6]|3[3-4])\.ffn_gate_shexp\.weight=q8_0
> blk\.([3-5]|[7-9]|17|21|40|43|53|55|3[0-2]|2[7-8]|3[5-9]|1[1-5])\.ffn_gate_shexp\.weight=iq6_k
> blk\.(6|10|22|59)\.ffn_gate_shexp\.weight=iq5_k_r4
> 
> ## CPU-loaded ffn_*_exps
> # ffn_down_exps (down-extraction) — qbits: 8 5 3 
> blk\.(51|53|3[2-9]|4[0-9])\.ffn_down_exps\.weight=q8_0
> blk\.([3-9]|50|52|60|5[4-9]|1[0-4]|2[0-9]|3[0-1]|1[6-9])\.ffn_down_exps\.weight=iq5_k_r4
> blk\.15\.ffn_down_exps\.weight=iq3_k
> 
> # ffn_up_exps (up-extraction) — qbits: 8 5 4 
> blk\.(35|53|55|4[7-8]|5[0-1]|4[3-4])\.ffn_up_exps\.weight=q8_0
> blk\.([3-9]|49|52|54|60|4[0-2]|1[1-9]|3[0-4]|2[0-9]|4[5-6]|3[6-9]|5[6-9])\.ffn_up_exps\.weight=iq5_k_r4
> blk\.10\.ffn_up_exps\.weight=iq4_ks
> 
> # ffn_gate_exps (gate-extraction) — qbits: 8 5 4 
> blk\.(35|39|41|60|5[0-5]|4[3-9])\.ffn_gate_exps\.weight=q8_0
> blk\.([3-7]|9|[1-2][0-9]|40|42|3[6-8]|3[0-4]|5[6-9])\.ffn_gate_exps\.weight=iq5_k_r4
> blk\.8\.ffn_gate_exps\.weight=iq4_ks
> 
> ## Summary of tensor sizes per class
> # GPU Total: 11.744 GiB (95.1%) | 12.34 GiB max, if all were q8_0 | 10.39 GiB min, if all were iq5_k_r4
> # CPU Total: 477.066 GiB (73.7%) | 647.06 GiB max, if all were q8_0 | 261.68 GiB min, if all were iq3_k
> # GPU+CPU Total: 488.811 GiB (84.4%)
> 
> ## Summary of tensor counts and bpw per qtype
> #
> # GPU-loaded quants:
> # QTYPE		Count	BPW	Assigned GiB	% Assigned	Max GiB (all)
> # +f32       	361	32.0  	  0.40 GiB	-		-
> # +q8_0      	61 	8.5   	  0.51 GiB	-		-
> # q8_0      	71 	8.5   	  3.07 GiB	55.4%		5.54
> # iq6_k     	101	6.625 	  1.60 GiB	37.0%		4.32
> # iq5_k_r4  	13 	5.5   	  0.27 GiB	7.6%		3.58
> # +iq4_xs    	366	4.25  	  5.90 GiB	-		-
> #
> # CPU-loaded quants:
> # QTYPE		Count	BPW	Assigned GiB	% Assigned	Max GiB (all)
> # q8_0      	46 	8.5   	171.06 GiB	26.4%		647.06
> # iq5_k_r4  	125	5.5   	300.78 GiB	71.8%		418.69
> # iq4_ks    	2  	4.25  	  3.72 GiB	1.1%		323.53
> # iq3_k     	1  	3.4375	  1.50 GiB	0.6%		261.68
> #
> # -Average BPW: 6.2478
> #
> # -Notes:
> # - '+' means user-defined pre-assigned tensors and f32 tensors
> # - Recipe produced on the 2025-07-16 19:21:22 UTC+0000 using Thireus' GGUF tools (https://gguf.thireus.com/)
> # - Script SHA-256: 3c88ec66185ed0999d6be95e1d8e5fb2d22000c404863f0c2fa301a44160f8c3
> # - Command used:
> # quant_assign.py ppl_results.csv --tolerance 0.01 --cpu-irq-k 1.5 --gpu-irq-k 1.5 --gpu-assign-qtype iq4_xs \
> # --cpu-tensors-max-size 500 --gpu-tensors-max-size 95% --exponential-factor 8 --cpu-tensors \
> # 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' \
> # 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' --gpu-tensors '.*' --cpu-quants iq4_ks iq3_k iq5_k_r4 q8_0 \
> # --gpu-quants q8_0 iq5_k_r4 iq6_k --gpu-assign-tensors 'blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0'
> 
> ## THE END!
> # Saved recipe to file: DeepSeek-R1-0528.ROOT-6.2478bpw-0.0000ppl.488GB-GGUF_11GB-GPU_477GB-CPU.3c88ec6_c3039f4.recipe
> ```
> 
> Save it inside a file named `~/DeepSeek-R1-0528.ROOT-6.2478bpw-0.0000ppl.488GB-GGUF_11GB-GPU_477GB-CPU.3c88ec6_c3039f4.recipe`
> 
> ```
> git clone https://github.com/Thireus/GGUF-Tool-Suite/
> cd GGUF-Tool-Suite
> mkdir DeepSeek-R1-0528.ROOT-6.2478bpw
> cd DeepSeek-R1-0528.ROOT-6.2478bpw
> ../quant_downloader.sh ~/DeepSeek-R1-0528.ROOT-6.2478bpw-0.0000ppl.488GB-GGUF_11GB-GPU_477GB-CPU.3c88ec6_c3039f4.recipe
> ```
> 
> Then run the latest version of ik_llama on the DeepSeek-R1-0528.ROOT-6.2478bpw model folder. But make sure to invoke ulimit -n 9999 before running it if you are on Linux) or use these [releases](https://github.com/Thireus/ik_llama.cpp/releases) for Windows.
> 
> ```
> ulimit -n 9999
> cd DeepSeek-R1-0528.ROOT-6.2478bpw
> ~/llama-cli \
>   -m DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf
> ```
> 
> Please if you can report back the ppl that'd be nice, thanks!
> 
> (I should have added q6_K tensors but I have not uploaded them yet, it should give a better ppl once available)

> 👤 **Thireus** replied on **2025-07-16** at **19:39:30**
> 
> @Panchovix, same instructions but your [recipe](https://colab.research.google.com/github/Thireus/GGUF-Tool-Suite/blob/main/quant_recipe_pipeline.ipynb) file is this one:
> 
> ```
> ## Quant mix recipe created using Thireus' GGUF Tool Suite - https://gguf.thireus.com/
> # Model name: DeepSeek-R1-0528
> # Link to the original model: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528
> 
> ## Model head & embeddings — qbits: 32 8 
> output_norm\.weight=f32
> token_embd\.weight=q8_0
> output\.weight=q8_0
> 
> ## Special attention kernels — single-quant only (llama-quantize takes care of it) — qbits: 8 
> blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0
> 
> ## Multi-headed attention parameters — qbits: 32 4 
> blk\.([0-9]|[1-5][0-9]|60)\.attn_v_b\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_mqa\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_output\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_b\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_b\.weight=iq4_xs
> 
> ## Core FFN weights — qbits: 32 8 6 5 
> blk\.2\.ffn_gate\.weight=q8_0
> blk\.(0|2)\.ffn_up\.weight=iq6_k
> blk\.([0-9]|[1-5][0-9]|60)\.ffn_norm\.weight=f32
> blk\.[0-1]\.ffn_gate\.weight=iq6_k
> blk\.1\.ffn_down\.weight=iq6_k
> blk\.2\.ffn_down\.weight=iq5_k_r4
> blk\.1\.ffn_up\.weight=iq5_k_r4
> blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_inp\.weight=f32
> blk\.0\.ffn_down\.weight=q8_0
> 
> ## Other tensors — qbits: 32 
> blk\.([3-9]|[1-5][0-9]|60)\.exp_probs_b\.bias=f32
> 
> ## GPU-loaded ffn_*_shexp
> # ffn_down_shexp (down-projection) — qbits: 8 6 5 
> blk\.(11|17|19|29|36|39|44|60|2[6-7]|2[0-4]|3[0-1]|3[3-4])\.ffn_down_shexp\.weight=q8_0
> blk\.([3-8]|10|12|25|28|32|35|3[7-8]|1[4-6]|4[5-9]|4[0-3]|5[0-8])\.ffn_down_shexp\.weight=iq6_k
> blk\.(9|13|18|59)\.ffn_down_shexp\.weight=iq5_k_r4
> 
> # ffn_up_shexp (up-projection) — qbits: 8 6 5 
> blk\.(6|15|18|30|37|39|41|50|54|60|2[1-4]|3[2-4]|2[6-9])\.ffn_up_shexp\.weight=q8_0
> blk\.([3-5]|[8-9]|19|20|25|31|38|40|58|4[2-9]|1[6-7]|1[0-4]|3[5-6]|5[5-6]|5[1-3])\.ffn_up_shexp\.weight=iq6_k
> blk\.(7|57|59)\.ffn_up_shexp\.weight=iq5_k_r4
> 
> # ffn_gate_shexp (gate-projection) — qbits: 8 6 5 
> blk\.(16|20|29|54|60|5[6-8]|5[0-2]|4[1-2]|4[4-9]|1[8-9]|2[3-6]|3[3-4])\.ffn_gate_shexp\.weight=q8_0
> blk\.([3-5]|[7-9]|17|21|40|43|53|55|3[0-2]|2[7-8]|3[5-9]|1[1-5])\.ffn_gate_shexp\.weight=iq6_k
> blk\.(6|10|22|59)\.ffn_gate_shexp\.weight=iq5_k_r4
> 
> ## CPU-loaded ffn_*_exps
> # ffn_down_exps (down-extraction) — qbits: 4 3 2 1 
> blk\.(51|53|3[2-9]|4[0-9])\.ffn_down_exps\.weight=iq4_ks
> blk\.([4-9]|50|52|60|5[4-9]|1[0-4]|2[0-9]|3[0-1]|1[6-9])\.ffn_down_exps\.weight=iq3_k
> blk\.3\.ffn_down_exps\.weight=iq2_k
> blk\.15\.ffn_down_exps\.weight=iq1_m_r4
> 
> # ffn_up_exps (up-extraction) — qbits: 4 3 2 
> blk\.(35|53|55|4[7-8]|5[0-1]|4[3-4])\.ffn_up_exps\.weight=iq4_ks
> blk\.([3-9]|49|52|54|60|4[0-2]|1[1-9]|3[0-4]|2[0-9]|4[5-6]|3[6-9]|5[6-9])\.ffn_up_exps\.weight=iq3_k
> blk\.10\.ffn_up_exps\.weight=iq2_k
> 
> # ffn_gate_exps (gate-extraction) — qbits: 4 3 2 
> blk\.(35|39|41|60|5[0-5]|4[3-9])\.ffn_gate_exps\.weight=iq4_ks
> blk\.([3-7]|9|[1-2][0-9]|40|42|3[6-8]|3[0-4]|5[6-9])\.ffn_gate_exps\.weight=iq3_k
> blk\.8\.ffn_gate_exps\.weight=iq2_k
> 
> ## Summary of tensor sizes per class
> # GPU Total: 11.744 GiB (95.1%) | 12.34 GiB max, if all were q8_0 | 10.39 GiB min, if all were iq5_k_r4
> # CPU Total: 275.898 GiB (85.3%) | 323.53 GiB max, if all were iq4_ks | 133.22 GiB min, if all were iq1_m_r4
> # GPU+CPU Total: 287.643 GiB (90.2%)
> 
> ## Summary of tensor counts and bpw per qtype
> #
> # GPU-loaded quants:
> # QTYPE		Count	BPW	Assigned GiB	% Assigned	Max GiB (all)
> # +f32       	361	32.0  	  0.40 GiB	-		-
> # +q8_0      	61 	8.5   	  0.51 GiB	-		-
> # q8_0      	71 	8.5   	  3.07 GiB	55.4%		5.54
> # iq6_k     	101	6.625 	  1.60 GiB	37.0%		4.32
> # iq5_k_r4  	13 	5.5   	  0.27 GiB	7.6%		3.58
> # +iq4_xs    	366	4.25  	  5.90 GiB	-		-
> #
> # CPU-loaded quants:
> # QTYPE		Count	BPW	Assigned GiB	% Assigned	Max GiB (all)
> # iq4_ks    	46 	4.25  	 85.53 GiB	26.4%		323.53
> # iq3_k     	124	3.4375	186.48 GiB	71.3%		261.68
> # iq2_k     	3  	2.375 	  3.12 GiB	1.7%		180.80
> # iq1_m_r4  	1  	1.75  	  0.77 GiB	0.6%		133.22
> #
> # -Average BPW: 3.6766
> #
> # -Notes:
> # - '+' means user-defined pre-assigned tensors and f32 tensors
> # - Recipe produced on the 2025-07-16 19:30:15 UTC+0000 using Thireus' GGUF tools (https://gguf.thireus.com/)
> # - Script SHA-256: 3c88ec66185ed0999d6be95e1d8e5fb2d22000c404863f0c2fa301a44160f8c3
> # - Command used:
> # quant_assign.py ppl_results.csv --tolerance 0.01 --cpu-irq-k 1.5 --gpu-irq-k 1.5 --gpu-assign-qtype iq4_xs \
> # --cpu-tensors-max-size 323 --gpu-tensors-max-size 95% --exponential-factor 8 --cpu-tensors \
> # 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' \
> # 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' --gpu-tensors '.*' --cpu-quants iq4_ks iq3_k iq2_k iq1_m_r4 \
> # --gpu-quants q8_0 iq5_k_r4 iq6_k --gpu-assign-tensors 'blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0'
> 
> ## THE END!
> # Saved recipe to file: DeepSeek-R1-0528.ROOT-3.6766bpw-0.0000ppl.286GB-GGUF_11GB-GPU_275GB-CPU.3c88ec6_97df301.recipe
> ```
> 
> Save it as `~/DeepSeek-R1-0528.ROOT-3.6766bpw-0.0000ppl.286GB-GGUF_11GB-GPU_275GB-CPU.3c88ec6_97df301.recipe`.
> 
> Since you have a crazy lot of VRAM you'll need to offload many tensors to your GPUs. Please if you can report back the ppl that'd be nice.

> 👤 **magikRUKKOLA** replied on **2025-07-16** at **21:06:02**
> 
> @Thireus ha what a cool workflow!  Apparently it downloads the shards so that in case I would want to try out a different recipe it would not [re-]download the quants/shards that were already downloaded -- is that correct?

> 👤 **Thireus** replied on **2025-07-16** at **21:06:57**
> 
> > @Thireus ha what a cool workflow!  Apparently it downloads the shards so that in case I would want to try out a different recipe it would not [re-]download the quants/shards that were already downloaded -- is that correct?
> 
> Yes that's correct, as long as you use the same directory.

> 👤 **Thireus** replied on **2025-07-17** at **00:06:35**
> 
> @magikRUKKOLA - Here's `DeepSeek-R1-0528.ROOT-6.1382bpw-0.0000ppl.507GB-GGUF_12GB-GPU_495GB-CPU.3c88ec6_090cc31.recipe` with q6_K if you'd like to try. Without knowing how your system performs for each quant involved this is pretty much a guessing game. The perplexity on the other hand isn't, and the expectation is that these recipes perform as close as possible to the theoretical max PPL for the given size.
> 
> ```
> ## Quant mix recipe created using Thireus' GGUF Tool Suite - https://gguf.thireus.com/
> # Model name: DeepSeek-R1-0528
> # Link to the original model: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528
> 
> ## Model head & embeddings — qbits: 32 8 
> output_norm\.weight=f32
> token_embd\.weight=q8_0
> output\.weight=q8_0
> 
> ## Special attention kernels — single-quant only (llama-quantize takes care of it) — qbits: 8 
> blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0
> 
> ## Multi-headed attention parameters — qbits: 32 4 
> blk\.([0-9]|[1-5][0-9]|60)\.attn_v_b\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_mqa\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_output\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_b\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_b\.weight=iq4_xs
> 
> ## Core FFN weights — qbits: 32 8 6 
> blk\.[1-2]\.ffn_gate\.weight=q8_0
> blk\.[0-1]\.ffn_up\.weight=iq6_k
> blk\.([0-9]|[1-5][0-9]|60)\.ffn_norm\.weight=f32
> blk\.0\.ffn_gate\.weight=iq6_k
> blk\.[1-2]\.ffn_down\.weight=iq6_k
> blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_inp\.weight=f32
> blk\.0\.ffn_down\.weight=q8_0
> blk\.2\.ffn_up\.weight=q8_0
> 
> ## Other tensors — qbits: 32 
> blk\.([3-9]|[1-5][0-9]|60)\.exp_probs_b\.bias=f32
> 
> ## GPU-loaded ffn_*_shexp
> # ffn_down_shexp (down-projection) — qbits: 8 6 
> blk\.([3-5]|8|19|39|40|49|51|57|60|1[4-7]|4[2-7]|3[0-7]|1[1-2]|2[0-9]|5[3-4])\.ffn_down_shexp\.weight=q8_0
> blk\.([6-7]|9|10|13|18|38|41|48|50|52|5[8-9]|5[5-6])\.ffn_down_shexp\.weight=iq6_k
> 
> # ffn_up_shexp (up-projection) — qbits: 8 6 5 
> blk\.([5-6]|8|18|45|58|60|1[0-5]|2[0-4]|3[0-9]|5[0-1]|5[3-4]|4[0-3]|4[7-9]|2[6-9])\.ffn_up_shexp\.weight=q8_0
> blk\.([3-4]|7|9|19|25|44|46|52|5[5-7]|1[6-7])\.ffn_up_shexp\.weight=iq6_k
> blk\.59\.ffn_up_shexp\.weight=iq5_k_r4
> 
> # ffn_gate_shexp (gate-projection) — qbits: 8 6 5 
> blk\.(5|7|60|4[1-9]|[2-3][0-1]|[1-3][3-9]|5[0-8])\.ffn_gate_shexp\.weight=q8_0
> blk\.([3-4]|6|[8-9]|22|32|40|59|1[1-2])\.ffn_gate_shexp\.weight=iq6_k
> blk\.10\.ffn_gate_shexp\.weight=iq5_k_r4
> 
> ## CPU-loaded ffn_*_exps
> # ffn_down_exps (down-extraction) — qbits: 8 6 5 4 
> blk\.(39|4[0-2]|3[2-7]|4[4-9])\.ffn_down_exps\.weight=q8_0
> blk\.(14|38|43|60|1[0-2]|2[0-9]|5[0-9]|1[8-9]|3[0-1])\.ffn_down_exps\.weight=q6_K
> blk\.(5|7|16)\.ffn_down_exps\.weight=iq5_k_r4
> blk\.([3-4]|6|[8-9]|13|15|17)\.ffn_down_exps\.weight=iq4_ks
> 
> # ffn_up_exps (up-extraction) — qbits: 8 6 5 4 
> blk\.(44|47|50)\.ffn_up_exps\.weight=q8_0
> blk\.(5|12|15|[2-3][0-9]|60|5[1-9]|1[7-8]|4[8-9]|4[5-6]|4[0-3])\.ffn_up_exps\.weight=q6_K
> blk\.(3|6|[8-9]|11|16|19|1[3-4])\.ffn_up_exps\.weight=iq5_k_r4
> blk\.(4|7|10)\.ffn_up_exps\.weight=iq4_ks
> 
> # ffn_gate_exps (gate-extraction) — qbits: 8 6 5 4 
> blk\.(41|44|4[6-9]|5[4-5])\.ffn_gate_exps\.weight=q8_0
> blk\.(16|20|22|40|45|60|3[0-9]|1[8-9]|4[2-3]|5[0-3]|2[7-9]|5[6-9]|2[4-5])\.ffn_gate_exps\.weight=q6_K
> blk\.([4-5]|9|17|21|26|1[0-5])\.ffn_gate_exps\.weight=iq5_k_r4
> blk\.(3|[6-8]|23)\.ffn_gate_exps\.weight=iq4_ks
> 
> ## Summary of tensor sizes per class
> # GPU Total: 12.062 GiB (97.7%) | 12.34 GiB max, if all were q8_0 | 10.39 GiB min, if all were iq5_k_r4
> # CPU Total: 495.113 GiB (76.5%) | 647.06 GiB max, if all were q8_0 | 323.53 GiB min, if all were iq4_ks
> # GPU+CPU Total: 507.176 GiB (87.1%)
> 
> ## Summary of tensor counts and bpw per qtype
> #
> # GPU-loaded quants:
> # QTYPE		Count	BPW	Assigned GiB	% Assigned	Max GiB (all)
> # +f32       	361	32.0  	  0.40 GiB	-		-
> # +q8_0      	61 	8.5   	  0.51 GiB	-		-
> # q8_0      	138	8.5   	  4.27 GiB	77.2%		5.54
> # iq6_k     	45 	6.625 	  0.96 GiB	22.3%		4.32
> # iq5_k_r4  	2  	5.5   	  0.02 GiB	0.5%		3.58
> # +iq4_xs    	366	4.25  	  5.90 GiB	-		-
> #
> # CPU-loaded quants:
> # QTYPE		Count	BPW	Assigned GiB	% Assigned	Max GiB (all)
> # q8_0      	27 	8.5   	100.41 GiB	15.5%		647.06
> # q6_K      	107	6     	307.21 GiB	61.5%		499.57
> # iq5_k_r4  	24 	5.5   	 57.75 GiB	13.8%		418.69
> # iq4_ks    	16 	4.25  	 29.75 GiB	9.2%		323.53
> #
> # -Average BPW: 6.1382
> #
> # -Notes:
> # - '+' means user-defined pre-assigned tensors and f32 tensors
> # - Recipe produced on the 2025-07-17 00:01:49 UTC+0000 using Thireus' GGUF tools (https://gguf.thireus.com/)
> # - Script SHA-256: 3c88ec66185ed0999d6be95e1d8e5fb2d22000c404863f0c2fa301a44160f8c3
> # - Command used:
> # quant_assign.py ppl_results.csv --tolerance 0.01 --cpu-irq-k 1.5 --gpu-irq-k 1.5 --gpu-assign-qtype iq4_xs \
> # --cpu-tensors-max-size 500 --gpu-tensors-max-size 99% --exponential-factor 8 --cpu-tensors \
> # 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' \
> # 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' --gpu-tensors '.*' --cpu-quants iq4_ks iq5_k_r4 q8_0 q6_K \
> # --gpu-quants q8_0 iq5_k_r4 iq6_k --gpu-assign-tensors 'blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0'
> 
> ## THE END!
> # Saved recipe to file: DeepSeek-R1-0528.ROOT-6.1382bpw-0.0000ppl.507GB-GGUF_12GB-GPU_495GB-CPU.3c88ec6_090cc31.recipe
> ```

> 👤 **Panchovix** replied on **2025-07-17** at **00:36:36**
> 
> @Thireus wondering here, to make this quant we need the original R1 0528 model at BF16/FP16 and then quant? Asking as rn I am not able to get 1.2TB+ available in storage haha.

> 👤 **Thireus** replied on **2025-07-17** at **01:41:17**
> 
> @Panchovix - You don't need to make the quant. You simply need to download the shards which quant_downloader.sh will do for you.
> So you need ~287 GB of available space.

> 👤 **Panchovix** replied on **2025-07-17** at **01:43:40**
> 
> @Thireus perfect! I don't have 500GB left either at the moment but I will see what can I do. Does that quant method you developed support V3 0324, or only R1 0528 at the moment?

> 👤 **Thireus** replied on **2025-07-17** at **01:49:02**
> 
> Only R1-0528 at the moment, I'm focusing on DeepSeek-TNG-R1T2-Chimera and Kimi-K2 for now, which should be ready in 3 weeks.

> 👤 **Thireus** replied on **2025-07-17** at **01:50:52**
> 
> @Panchovix, sorry I got confused, your model recipe is actually 287 GB in size, not 500GB.

> 👤 **magikRUKKOLA** replied on **2025-07-17** at **15:53:23**
> 
> @Thireus 
> 
> Ha!  It crashed right before the output of the PPL:
> 
> ```
> 
> llama_new_context_with_model: n_ctx      = 4096
> llama_new_context_with_model: n_batch    = 4096
> llama_new_context_with_model: n_ubatch   = 2048
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 3
> llama_new_context_with_model: attn_max_b = 512
> llama_new_context_with_model: fused_moe  = 1
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:      CUDA0 KV buffer size =    74.12 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =    71.73 MiB
> llama_new_context_with_model: KV self size  =  145.83 MiB, c^KV (q8_0):  145.83 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     3.95 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size =  7802.00 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size =  1144.02 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =    88.02 MiB
> llama_new_context_with_model: graph nodes  = 3568
> llama_new_context_with_model: graph splits = 159
> 
> system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
> perplexity: tokenizing the input ..
> perplexity: tokenization took 1114.81 ms
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=4096, n_seq=8
> perplexity: 65.25 seconds per pass - ETA 1 hours 16.25 minutes
> [1]2.5262,[2]3.2173,[3]2.3303,[4]1.9474,[5]1.7635,[6]1.6256,[7]1.5334,[8]1.4696,[9]1.4223,[10]1.3829,[11]1.3669,[12]1.3820,[13]1.3936,[14]1.5115,[15]1.6382,[16]1.6926,[17]1.8469,[18]1.9682,[19]1.9334,[20]1.9204,[21]2.0207,[22]1.9939,[23]1.9680,[24]1.9806,[25]1.9529,[26]1.9327,[27]1.9760,[28]1.9863,[29]2.0318,[30]2.0619,[31]2.0922,[32]2.1086,[33]2.1451,[34]2.1883,[35]2.2342,[36]2.2833,[37]2.3195,[38]2.3655,[39]2.4074,[40]2.4649,[41]2.5011,[42]2.5128,[43]2.5580,[44]2.5722,[45]2.6489,[46]2.6966,[47]2.6546,[48]2.6115,[49]2.5883,[50]2.6049,[51]2.6470,[52]2.6614,[53]2.7126,[54]2.7264,[55]2.7573,[56]2.7867,[57]2.7992,[58]2.8302,[59]2.8405,[60]2.8832,[61]2.9222,[62]2.9688,[63]3.0005,[64]3.0406,[65]3.0504,[66]3.0356,[67]3.0123,[68]3.0379,[69]3.0348,[70]3.0454,[71]3.0638,[72]3.0789,[73]3.0928,[74]3.1162,[75]3.0962,[76]3.0529,[77]3.0120,[78]3.0063,[79]2.9854,[80]2.9675,[81]2.9331,[82]2.9354,[83]2.9063,[84]2.8739,[85]2.8419,[86]2.8192,[87]2.8135,[88]2.7881,[89]2.7719,[90]2.7484,[91]2.7210,[92]2.6977,[93]2.6728,[94]2.6488,[95]2.6287,[96]2.6265,[97]2.6331,[98]2.6189,[99]2.6029,[100]2.6041,[101]2.5964,[102]2.6123,[103]2.6361,[104]2.6539,[105]2.6512,[106]2.6737,[107]2.6978,[108]2.7173,[109]2.7495,[110]2.7826,[111]2.8010,[112]2.7770,[113]2.7642,[114]2.7435,[115]2.7293,[116]2.7161,[117]2.6950,[118]2.6752,[119]2.6553,[120]2.6377,[121]2.6221,[122]2.6059,[123]2.5896,[124]2.5713,[125]2.5545,[126]2.5389,[127]2.5259,[128]2.5160,[129]2.5049,[130]2.4926,[131]2.4847,[132]2.4901,[133]2.4994,[134]2.5050,[135]2.5150,[136]2.5301,[137]2.5430,[138]2.5509,[139]2.5616,[140]2.5631,[141]2.5648,[142]2.5641,[143]2.5654,[144]2.5631,[145]2.5557,[146]2.5545,[147]2.5593,[148]2.5597,[149]2.5612,[150]2.5563,[151]2.5546,[152]2.5523,[153]2.5490,[154]2.5494,[155]2.5534,[156]2.5554,[157]2.5615,[158]2.5698,[159]2.5723,[160]2.5812,[161]2.5893,[162]2.5989,[163]2.6027,[164]2.6221,[165]2.6445,[166]2.6611,[167]2.6725,[168]2.6956,[169]2.7174,[170]2.7375,[171]2.7594,[172]2.7446,[173]2.7292,[174]2.7166,[175]2.7044,[176]2.6933,[177]2.6826,[178]2.6708,[179]2.6579,[180]2.6615,[181]2.6755,[182]2.6905,[183]2.7042,[184]2.7172,[185]2.7273,[186]2.7432,[187]2.7583,[188]2.7724,[189]2.7830,[190]2.7839,[191]2.7913,[192]2.7945,[193]2.7994,[194]2.8184,[195]2.8271,[196]2.8401,[197]2.8502,[198]2.8546,[199]2.8599,[200]2.8590,[201]2.8731,[202]2.8678,[203]2.8732,[204]2.8765,[205]2.8763,[206]2.8793,[207]2.8868,[208]2.8957,[209]2.9047,[210]2.9050,[211]2.9009,[212]2.9018,[213]2.9091,[214]2.9108,[215]2.9158,[216]2.9164,[217]2.9114,[218]2.9117,[219]2.9129,[220]2.9128,[221]2.9133,[222]2.9133,[223]2.9138,[224]2.9185,[225]2.9202,[226]2.9127,[227]2.9100,[228]2.9120,[229]2.9160,[230]2.9221,[231]2.9286,[232]2.9210,[233]2.9138,[234]2.9147,[235]2.9127,[236]2.9209,[237]2.9285,[238]2.9373,[239]2.9472,[240]2.9561,[241]2.9670,[242]2.9807,[243]2.9921,[244]2.9999,[245]3.0108,[246]3.0214,[247]3.0200,[248]3.0160,[249]3.0138,[250]3.0079,[251]3.0059,[252]3.0085,[253]3.0122,[254]3.0193,[255]3.0252,[256]3.0289,[257]3.0314,[258]3.0325,[259]3.0360,[260]3.0384,[261]3.0400,[262]3.0394,[263]3.0444,[264]3.0467,[265]3.0470,[266]3.0486,[267]3.0507,[268]3.0539,[269]3.0569,[270]3.0564,[271]3.0548,[272]3.0485,[273]3.0479,[274]3.0416,[275]3.0314,[276]3.0210,[277]3.0230,[278]3.0328,[279]3.0385,[280]3.0461,[281]3.0535,[282]3.0593,[283]3.0654,[284]3.0715,[285]3.0850,[286]3.0873,[287]3.0905,[288]3.0954,[289]3.0978,[290]3.0903,[291]3.0815,[292]3.0791,[293]3.0786,[294]3.0760,[295]3.0738,[296]3.0756,[297]3.0762,[298]3.0815,[299]3.0869,[300]3.0896,[301]3.0934,[302]3.0953,[303]3.0966,[304]3.0962,[305]3.1074,[306]3.1146,[307]3.1252,[308]3.1147,[309]3.1095,[310]3.1006,[311]3.1032,[312]3.1045,[313]3.1092,[314]3.1115,[315]3.1147,[316]3.1162,[317]3.1180,[318]3.1184,[319]3.1190,[320]3.1229,[321]3.1229,[322]3.1243,[323]3.1305,[324]3.1313,[325]3.1363,[326]3.1404,[327]3.1441,[328]3.1465,[329]3.1481,[330]3.1545,[331]3.1574,[332]3.1619,[333]3.1609,[334]3.1614,[335]3.1621,[336]3.1621,[337]3.1631,[338]3.1630,[339]3.1654,[340]3.1688,[341]3.1742,[342]3.1828,[343]3.1915,[344]3.1963,[345]3.1881,[346]3.1808,[347]3.1756,[348]3.1686,[349]3.1646,[350]3.1634,[351]3.1679,[352]3.1819,[353]3.1909,[354]3.2030,[355]3.2115,[356]3.2168,[357]3.2282,[358]3.2376,[359]3.2407,[360]3.2466,[361]3.2556,[362]3.2637,[363]3.2689,[364]3.2752,[365]3.2807,[366]3.2904,[367]3.2988,[368]3.3054,[369]3.3128,[370]3.3208,[371]3.3335,[372]3.3415,[373]3.3450,[374]3.3482,[375]3.3525,[376]3.3646,[377]3.3750,[378]3.3777,[379]3.3776,[380]3.3744,[381]3.3787,[382]3.3843,[383]3.3874,[384]3.3915,[385]3.3953,[386]3.4006,[387]3.4060,[388]3.4090,[389]3.3993,[390]3.3906,[391]3.3808,[392]3.3757,[393]3.3665,[394]3.3582,[395]3.3497,[396]3.3402,[397]3.3320,[398]3.3229,[399]3.3132,[400]3.3050,[401]3.2956,[402]3.2859,[403]3.2780,[404]3.2685,[405]3.2596,[406]3.2504,[407]3.2418,[408]3.2333,[409]3.2252,[410]3.2196,[411]3.2202,[412]3.2156,[413]3.2169,[414]3.2183,[415]3.2150,[416]3.2148,[417]3.2166,[418]3.2108,[419]3.2118,[420]3.2092,[421]3.2081,[422]3.2088,[423]3.2083,[424]3.2120,[425]3.2119,[426]3.2120,[427]3.2111,[428]3.2134,[429]3.2145,[430]3.2169,[431]3.2177,[432]3.2166,[433]3.2131,[434]3.2131,[435]3.2060,[436]3.2001,[437]3.1964,[438]3.1949,[439]3.1921,[440]3.1968,[441]3.2019,[442]3.2092,[443]3.2071,[444]3.2077,[445]3.2084,[446]3.2122,[447]3.2152,[448]3.2173,[449]3.2203,[450]3.2240,[451]3.2271,[452]3.2289,[453]3.2302,[454]3.2289,[455]3.2312,[456]3.2318,[457]3.2346,[458]3.2395,[459]3.2400,[460]3.2401,[461]3.2373,[462]3.2406,[463]3.2474,[464]3.2521,[465]3.2457,[466]3.2437,[467]3.2421,[468]3.2436,[469]3.2412,[470]3.2384,[471]3.2389,[472]3.2396,[473]3.2387,[474]3.2378,[475]3.2388,[476]3.2373,[477]3.2364,[478]3.2372,[479]3.2387,[480]3.2411,[481]3.2374,[482]3.2408,[483]3.2403,[484]3.2439,[485]3.2500,[486]3.2531,[487]3.2566,[488]3.2617,[489]3.2642,[490]3.2685,[491]3.2742,[492]3.2782,[493]3.2778,[494]3.2789,[495]3.2810,[496]3.2829,[497]3.2857,[498]3.2864,[499]3.2860,[500]3.2896,[501]3.2942,[502]3.2929,[503]3.2917,[504]3.2936,[505]3.2969,[506]3.3045,[507]3.3075,[508]3.3108,[509]3.3039,[510]3.2985,[511]3.2923,[512]3.2879,[513]3.2819,[514]3.2802,[515]3.2819,[516]3.2771,[517]3.2771,[518]3.2760,[519]3.2760,[520]3.2797,[521]3.2783,[522]3.2769,[523]3.2819,[524]3.2805,[525]3.2789,[526]3.2744,[527]3.2696,[528]3.2664,[529]3.2636,[530]3.2608,[531]3.2582,[532]3.2529,[533]3.2472,[534]3.2429,[535]3.2437,[536]3.2461,[537]3.2490,[538]3.2509,[539]3.2534,[540]3.2584,[541]3.2613,[542]3.2636,[543]3.2584,[544]3.2544,[545]3.2541,[546]3.2480,[547]3.2419,[548]3.2359,[549]3.2297,[550]3.2239,[551]3.2182,[552]3.2128,[553]3.2073,[554]3.2056,[555]3.2039,[556]3.2068,[557]3.2104,[558]3.2163,[559]3.2204,[560]3.2257,/opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error/opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> 
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> fatal error
> fatal error/opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal errorfatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error/opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> 
> fatal error
> 
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> fatal error/opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error/opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> 
> 
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error/opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254:
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> 
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> ```
> 
> [EDIT]: let me try with 1 batch size.
> 
> [EDIT2]: i wasn't able to initialize the GPUs after that crash probably because I installed a proprietary driver some time ago ... so after that the nvidia-smi wasn't able to detect any GPUs.  The installatioin of the latest linux kernel and the subsequent installation of the opensource nvidia drivers seems to fix the issue.  So let me try again.  I can try some different seeds, but overall, it shouldn't matter much (EVEN IF it actually applies to the perplexity calculations).
> 
> [EDIT3]: so after the unfortunate reboot I am starting the PPL calculations again.
> The PPL of every batch is exactly the same as it was before:
> ```
> perplexity: 72.76 seconds per pass - ETA 1 hours 25.03 minutes
> [1]2.5262,[2]3.2173,[3]2.3303,[4]1.9474,[5]1.7635,[6]1.6256,[7]1.5334,[8]1.4696,
> ```
> so the low PPL wasn't a fluke apparently.   I will report the final PPL eventually within an hour or so.
> 
> [EDIT4]: dammit it was the same error in the end once again.  I guess its because it tries to allocate some pinned buffer and fails to do so.  So I added the 64GB swapfile ... let's see what's up.

> 👤 **ikawrakow** replied on **2025-07-17** at **16:07:42**
> 
> This is really strange. I just don't see how it can compute 560 batches and then fail. Nothing is different in the 561'th batch.
> 
> Either way, this assert is triggered when computing on the CPU. It can only be triggered if a GEMM or dequantization kernel is missing. Can you post the list of quantization types being used? Thanks.

> 👤 **magikRUKKOLA** replied on **2025-07-17** at **16:16:37**
> 
> > Either way, this assert is triggered when computing on the CPU.
> 
> Its might be due to the insufficient RAM (I don't have any swap). :)

> 👤 **Thireus** replied on **2025-07-17** at **16:28:02**
> 
> @magikRUKKOLA, I saw you had initially posted about llama not being able to load more than 62 shards. I believe you have discovered that you need you must compile the latest ik_llama.cpp with `-DGGML_MAX_CONTEXTS=2048` - see pull requests: [#611](https://github.com/ikawrakow/ik_llama.cpp/issues/611), [#620](https://github.com/ikawrakow/ik_llama.cpp/issues/620) and [#622](https://github.com/ikawrakow/ik_llama.cpp/issues/622).
> 
> Curious issue you have here with the PPL computation. The resulting PPL is the average of all the PPLs computed for each chunk. So your PPL should have been very close to: 2.9532

> 👤 **Thireus** replied on **2025-07-17** at **16:33:16**
> 
> @magikRUKKOLA - not sure if this is already the case but please use --seed 1337 -f [wiki.test.raw](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/wiki.test.raw). That ppl is oddly low.

> 👤 **ikawrakow** replied on **2025-07-17** at **16:36:04**
> 
> There is no sampling in PPL calculations, so the seed should have no impact on the computed perplexity.

> 👤 **Thireus** replied on **2025-07-17** at **16:41:32**
> 
> @ikawrakow - Ah good to know! cc: @ubergarm

> 👤 **Panchovix** replied on **2025-07-17** at **16:46:46**
> 
> @Thireus I will try to do the quant you sent me here after quantization tweaks PR https://github.com/ikawrakow/ik_llama.cpp/pull/624, as I managed to get some storage by deleting some models that I may not use.
> 
> Would your script work when ikllamacpp gets updated or we have to wait on your end before?

> 👤 **Thireus** replied on **2025-07-17** at **16:51:19**
> 
> @Panchovix - You don't need to wait for this PR and since the quants have already been computed the model you'll download won't have these optimisations (I'd have to re-compute the quants which would take a few days). Make sure you compile the latest ik_llama with `-DGGML_MAX_CONTEXTS=2048` as mentioned above though.

> 👤 **Panchovix** replied on **2025-07-17** at **17:49:27**
> 
> @Thireus I got this error when trying to run the model, I'm running it wrong?
> 
> ```
> pancho@fedora:/run/media/pancho/60A2FCEDA2FCC894/ChatIAs/ik_llama.cpp/lenux/bin$ ./llama-perplexity -m '/run/media/pancho/60A2FCEDA2FCC894/models_llm/GGUF-Tool-Suite/kitchen/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf' -c 512 --no-mmap -ngl 999 -ot "blk.(0|1|2|3|4|5|6).ffn.=CUDA0" -ot "blk.(7|8|9|10).ffn.=CUDA1" -ot "blk.(11|12|13|14).ffn.=CUDA2" -ot "blk.(15|16|17|18|19).ffn.=CUDA3" -ot "blk.(20|21|22|23).ffn.=CUDA4" -ot "blk.(24|25|26|27).ffn.=CUDA5" -ot "blk.(28|29|30|31|32|33|34|35).ffn.=CUDA6" -ot exps=CPU -fa -mg 0 -mla 3 -fmoe -amb 256 --threads 8 -f wiki.test.raw
> ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
> ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
> ggml_cuda_init: found 7 CUDA devices:
>   Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
>   Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
>   Device 2: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
>   Device 3: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
>   Device 4: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 5: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 6: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
> main: build = 3806 (b94f3af5)
> main: built with cc (GCC) 14.3.1 20250523 (Red Hat 14.3.1-1) for x86_64-redhat-linux
> main: seed  = 1752774546
> llama_model_load: error loading model: tensor 'blk.5.ffn_gate_exps.weight' data is not within the file bounds, model is corrupted or incomplete
> llama_load_model_from_file: failed to load model
> llama_init_from_gpt_params: error: failed to load model '/run/media/pancho/60A2FCEDA2FCC894/models_llm/GGUF-Tool-Suite/kitchen/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf'
> main: error: unable to load model
> ```
> 
> Log from kitchen was at the end
> 
> ```
> [2025-07-17 13:41:35] Saved file id '01145' - tensor 'blk.60.ffn_gate_exps.weight' of qtype: 'iq4_ks'
> [2025-07-17 13:41:35] Fetching first shard separately
> [2025-07-17 13:41:35] Starting download of DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf into ./downloaded_shards
> [2025-07-17 13:41:35] Trying curl from https://huggingface.co/Thireus/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_SPLIT/resolve/main/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf?download=true
> [2025-07-17 13:41:36] Download complete, verifying…
> [2025-07-17 13:41:36] ✓ Verified and saved via curl (org: Thireus, banch: main) - ./downloaded_shards/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf (BF16)
> [2025-07-17 13:41:36] First shard saved
> [2025-07-17 13:41:36] Verifying shard sequence completeness
> [2025-07-17 13:41:36] All shards from 00002 to 01148 are present.
> Download and verification complete. Enjoy!
> ```

> 👤 **Thireus** replied on **2025-07-17** at **17:57:47**
> 
> @Panchovix, have you executed `ulimit -n 9999` before running llama-perplexity?

> 👤 **Panchovix** replied on **2025-07-17** at **18:00:42**
> 
> @Thireus do you mean on the console directly or as an argument? Tried it on the console and got the same error. Not sure how to use it as argument.
> 
> ```
> pancho@fedora:/run/media/pancho/60A2FCEDA2FCC894/ChatIAs/ik_llama.cpp/lenux/bin$ ulimit -n 9999
> pancho@fedora:/run/media/pancho/60A2FCEDA2FCC894/ChatIAs/ik_llama.cpp/lenux/bin$ ./llama-perplexity -m '/run/media/pancho/60A2FCEDA2FCC894/models_llm/GGUF-Tool-Suite/kitchen/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf' -c 512 --no-mmap -ngl 999 -ot "blk.(0|1|2|3|4|5|6).ffn.=CUDA0" -ot "blk.(7|8|9|10).ffn.=CUDA1" -ot "blk.(11|12|13|14).ffn.=CUDA2" -ot "blk.(15|16|17|18|19).ffn.=CUDA3" -ot "blk.(20|21|22|23).ffn.=CUDA4" -ot "blk.(24|25|26|27).ffn.=CUDA5" -ot "blk.(28|29|30|31|32|33|34|35).ffn.=CUDA6" -ot exps=CPU -fa -mg 0 -mla 3 -fmoe -amb 256 --threads 8 -f wiki.test.raw
> ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
> ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
> ggml_cuda_init: found 7 CUDA devices:
>   Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
>   Device 1: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
>   Device 2: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
>   Device 3: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
>   Device 4: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 5: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 6: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
> main: build = 3806 (b94f3af5)
> main: built with cc (GCC) 14.3.1 20250523 (Red Hat 14.3.1-1) for x86_64-redhat-linux
> main: seed  = 1752775192
> llama_model_load: error loading model: tensor 'blk.5.ffn_gate_exps.weight' data is not within the file bounds, model is corrupted or incomplete
> llama_load_model_from_file: failed to load model
> llama_init_from_gpt_params: error: failed to load model '/run/media/pancho/60A2FCEDA2FCC894/models_llm/GGUF-Tool-Suite/kitchen/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf'
> main: error: unable to load model
> ```

> 👤 **Thireus** replied on **2025-07-17** at **18:02:59**
> 
> Yes like this is good. I'm not familiar with fedora, so not sure if there might be another OS limit that prevents you from loading all shards.
> 
> This is the shard it is complaining about: -00099-of-01148.gguf, can you check you see it: `ls -l *-00099-of-01148.gguf`. And can you tell me the checksum please? `sha256sum *-00099-of-01148.gguf`.

> 👤 **Panchovix** replied on **2025-07-17** at **18:05:14**
> 
> @Thireus sure, here it is
> 
> ```
> pancho@fedora:/run/media/pancho/60A2FCEDA2FCC894/models_llm/GGUF-Tool-Suite/kitchen$ ls -l *-00099-of-01148.gguf
> -rwxr-xr-x. 1 pancho pancho 1325184345 Jul 17 13:01 DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00099-of-01148.gguf
> pancho@fedora:/run/media/pancho/60A2FCEDA2FCC894/models_llm/GGUF-Tool-Suite/kitchen$ sha256sum *-00099-of-01148.gguf
> e118e472db8e9726308cd5ee84cbc9bf31c2da0900d1b0f24827347d9a3b1084  DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00099-of-01148.gguf
> ```
> 
> Also for reference, I built with:
> 
> ```
> cmake -B lenux \
>     -DGGML_CUDA=ON \
>     -DGGML_CUDA_FA_ALL_QUANTS=ON \
>     -DGGML_BLAS=OFF \
>     -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
>     -DGGML_IQK_FA_ALL_QUANTS=1 \
>     -DGGML_SCHED_MAX_COPIES=1 \
>     -DGGML_CUDA_IQK_FORCE_BF16=1 \
>     -DGGML_MAX_CONTEXTS=2048 \
> ```

> 👤 **Thireus** replied on **2025-07-17** at **18:11:28**
> 
> @Panchovix - The hash of that file is wrong. It should have been `777f8de0b4de8216417da48ddfcd3e7de32bf83e289176c7b6c9b26f0d3943ed`. Could you try to run the quant_downloader again in the same directory? Don't delete the existing files, it will automatically replace the corrupted ones. Please tell me if you see any hash mismatch error in the quant_downloader output.

> 👤 **Panchovix** replied on **2025-07-17** at **19:07:01**
> 
> @Thireus okay just got back, sorry for the delay. After rerunning it found that shard having issues and redownloaded it, and also a few more ones. Sadly I forgot to output it to a log file and restarted the PC afterwards (had no do something in Windows and then returned to Fedora).
> 
> Model is loading now so I'm gonna test it's perplexity.
> 
> ```
> llm_load_tensors:        CPU buffer size = 130220.00 MiB
> llm_load_tensors:  CUDA_Host buffer size =   938.98 MiB
> llm_load_tensors:      CUDA0 buffer size = 20806.16 MiB
> llm_load_tensors:      CUDA1 buffer size = 18858.90 MiB
> llm_load_tensors:      CUDA2 buffer size = 19985.20 MiB
> llm_load_tensors:      CUDA3 buffer size = 23914.20 MiB
> llm_load_tensors:      CUDA4 buffer size = 20233.62 MiB
> llm_load_tensors:      CUDA5 buffer size = 20165.25 MiB
> llm_load_tensors:      CUDA6 buffer size = 43249.05 MiB
> .....................
> ```

> 👤 **Thireus** replied on **2025-07-17** at **19:09:11**
> 
> @Panchovix - Great, I'll try to see if there is a bug in the quant_downloader script because it should have identified the hash mismatch the first time you ran it. Glad it's working now. Yes, please let us know the ppl when you see it.

> 👤 **Thireus** replied on **2025-07-17** at **20:27:26**
> 
> @Panchovix (cc @magikRUKKOLA) I fixed this issue with quant_downloader: https://github.com/Thireus/GGUF-Tool-Suite/issues/5, if you obtain the latest version the new script will ensure the file hash is correctly checked (which was not the case when shards were downloaded for the first time). Thanks Panchovix for your help spotting the issue.

> 👤 **Panchovix** replied on **2025-07-17** at **21:24:27**
> 
> Okay ended the PPL test. model ended at 3.724 BPW, 291.378GB.
> 
> ```
> DeepSeek-R1-0528-3.6bpw_Thireuscustom.gguf
> Final estimate: PPL = 3.2741 +/- 0.01738
> 291.37 GB
> ```
> 
> Comparatively to other quants
> 
> ```
> DeepSeek-R1-0528-IQ4_XS-merged.gguf
> Final estimate: PPL = 3.2598 +/- 0.01727
> 333.1GB
> 
> DeepSeek-R1-0528-IQ3_K_R4-merged.gguf
> Final estimate: PPL = 3.2730 +/- 0.01738
> 300.9 GB
> 
> DeepSeek-R1-0528-IQ3_KS-merged.gguf
> Final estimate: PPL = 3.2983 +/- 0.01759
> 281.5 GB
> 
> DeepSeek-R1-0528-Q3_K_XL-merged.gguf
> Final estimate: PPL = 3.3324
> 275.6 GB
> ```
> 
> So pretty good!

> 👤 **Thireus** replied on **2025-07-17** at **21:30:38**
> 
> Cool, we may be able to get it down to 3.26. Let me know if you want to try another one and if there is still some spare RAM and VRAM available after you load the custom model you just tested.

> 👤 **Panchovix** replied on **2025-07-17** at **21:34:03**
> 
> For now I kept this one for normal usage, as it pretty damn near Q8 lol (3.2119). I have about 25GB left on RAM and not much VRAM as I use ubatch/batch 4096. Maybe a 300-310GB one or one at 260GB sounds interesting. 330GB is about the limit I can do (iq4_xs, 3.2598 PPL)
> 
> EDIT: Also these quants work only on iklcpp right? So I have to keep IQ4_XS and Q3_K_XL for normal lcpp.

> 👤 **Thireus** replied on **2025-07-17** at **21:51:48**
> 
> @Panchovix, if you'd like to give a go to this one: `DeepSeek-R1-0528.ROOT-3.9399bpw-0.0000ppl.308GB-GGUF_14GB-GPU_294GB-CPU.3c88ec6_ae5dd55.recipe`
> 
> ```
> ## Quant mix recipe created using Thireus' GGUF Tool Suite - https://gguf.thireus.com/
> # Model name: DeepSeek-R1-0528
> # Link to the original model: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528
> 
> ## Model head & embeddings — qbits: 32 8 
> output_norm\.weight=f32
> token_embd\.weight=q8_0
> output\.weight=q8_0
> 
> ## Special attention kernels — single-quant only (llama-quantize takes care of it) — qbits: 8 
> blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0
> 
> ## Multi-headed attention parameters — qbits: 32 5 
> blk\.([0-9]|[1-5][0-9]|60)\.attn_output\.weight=iq5_k_r4
> blk\.([0-9]|[1-5][0-9]|60)\.attn_v_b\.weight=iq5_k_r4
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_mqa\.weight=iq5_k_r4
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_b\.weight=iq5_k_r4
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a\.weight=iq5_k_r4
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_b\.weight=iq5_k_r4
> blk\.([0-9]|[1-5][0-9]|60)\.attn_norm\.weight=f32
> 
> ## Core FFN weights — qbits: 32 8 
> blk\.[0-2]\.ffn_gate\.weight=q8_0
> blk\.([0-9]|[1-5][0-9]|60)\.ffn_norm\.weight=f32
> blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_inp\.weight=f32
> blk\.[0-2]\.ffn_down\.weight=q8_0
> blk\.[0-2]\.ffn_up\.weight=q8_0
> 
> ## Other tensors — qbits: 32 
> blk\.([3-9]|[1-5][0-9]|60)\.exp_probs_b\.bias=f32
> 
> ## GPU-loaded ffn_*_shexp
> # ffn_down_shexp (down-projection) — qbits: 8 
> blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_shexp\.weight=q8_0
> 
> # ffn_up_shexp (up-projection) — qbits: 8 
> blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_shexp\.weight=q8_0
> 
> # ffn_gate_shexp (gate-projection) — qbits: 8 
> blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_shexp\.weight=q8_0
> 
> ## CPU-loaded ffn_*_exps
> # ffn_down_exps (down-extraction) — qbits: 5 4 3 2 
> blk\.(39|3[5-7]|4[0-2]|3[2-3]|4[7-9]|4[4-5])\.ffn_down_exps\.weight=iq5_k_r4
> blk\.(12|34|38|43|46|60|5[0-7]|2[0-9]|3[0-1])\.ffn_down_exps\.weight=iq4_ks
> blk\.(14|1[8-9]|5[8-9]|1[0-1])\.ffn_down_exps\.weight=iq3_k
> blk\.([3-9]|13|1[5-7])\.ffn_down_exps\.weight=iq2_k
> 
> # ffn_up_exps (up-extraction) — qbits: 5 4 3 2 
> blk\.50\.ffn_up_exps\.weight=iq5_k_r4
> blk\.(21|24|40|60|[3-4][2-9]|5[1-7]|2[6-7])\.ffn_up_exps\.weight=iq4_ks
> blk\.(5|8|20|25|41|1[2-5]|1[7-8]|2[8-9]|5[8-9]|3[0-1]|2[2-3])\.ffn_up_exps\.weight=iq3_k
> blk\.([3-4]|[6-7]|9|16|19|1[0-1])\.ffn_up_exps\.weight=iq2_k
> 
> # ffn_gate_exps (gate-extraction) — qbits: 5 4 3 2 
> blk\.(44|46|4[8-9])\.ffn_gate_exps\.weight=iq5_k_r4
> blk\.(24|45|47|60|5[0-9]|3[0-1]|4[0-3]|3[3-9]|2[7-9])\.ffn_gate_exps\.weight=iq4_ks
> blk\.(5|25|32|2[0-2]|1[8-9]|1[5-6]|1[2-3])\.ffn_gate_exps\.weight=iq3_k
> blk\.([3-4]|[6-9]|14|17|23|26|1[0-1])\.ffn_gate_exps\.weight=iq2_k
> 
> ## Summary of tensor sizes per class
> # GPU Total: 14.080 GiB (100.0%) | 14.08 GiB max, if all were q8_0 | 14.08 GiB min, if all were q8_0
> # CPU Total: 294.164 GiB (70.3%) | 418.69 GiB max, if all were iq5_k_r4 | 180.80 GiB min, if all were iq2_k
> # GPU+CPU Total: 308.244 GiB (85.1%)
> 
> ## Summary of tensor counts and bpw per qtype
> #
> # GPU-loaded quants:
> # QTYPE		Count	BPW	Assigned GiB	% Assigned	Max GiB (all)
> # +f32       	361	32.0  	  0.40 GiB	-		-
> # +q8_0      	61 	8.5   	  0.51 GiB	-		-
> # q8_0      	185	8.5   	  5.54 GiB	100.0%		5.54
> # +iq5_k_r4  	366	5.5   	  7.64 GiB	-		-
> #
> # CPU-loaded quants:
> # QTYPE		Count	BPW	Assigned GiB	% Assigned	Max GiB (all)
> # iq5_k_r4  	19 	5.5   	 45.72 GiB	10.9%		418.69
> # iq4_ks    	85 	4.25  	158.05 GiB	48.9%		323.53
> # iq3_k     	38 	3.4375	 57.15 GiB	21.8%		261.68
> # iq2_k     	32 	2.375 	 33.25 GiB	18.4%		180.80
> #
> # -Average BPW: 3.9399
> #
> # -Notes:
> # - '+' means user-defined pre-assigned tensors and f32 tensors
> # - Recipe produced on the 2025-07-17 21:49:55 UTC+0000 using Thireus' GGUF tools (https://gguf.thireus.com/)
> # - Script SHA-256: 3c88ec66185ed0999d6be95e1d8e5fb2d22000c404863f0c2fa301a44160f8c3
> # - Command used:
> # quant_assign.py ppl_results.csv --tolerance 0.01 --cpu-irq-k 1.5 --gpu-irq-k 1.5 --gpu-assign-qtype iq5_k_r4 \
> # --cpu-tensors-max-size 295 --gpu-tensors-max-size 100% --exponential-factor 8 --cpu-tensors \
> # 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' \
> # 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' --gpu-tensors '.*' --cpu-quants iq5_k_r4 iq4_ks iq3_k iq2_k \
> # --gpu-quants q8_0 --gpu-assign-tensors 'blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0'
> 
> ## THE END!
> # Saved recipe to file: DeepSeek-R1-0528.ROOT-3.9399bpw-0.0000ppl.308GB-GGUF_14GB-GPU_294GB-CPU.3c88ec6_ae5dd55.recipe
> ```
> 
> I'd suggest you use the same download directory (or copy it, so it doesn't download the shards that already match).

> 👤 **ubergarm** replied on **2025-07-17** at **23:10:08**
> 
> @magikRUKKOLA @ikawrakow @Thireus 
> 
> > /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> /opt/ik_llama.cpp/ik_llama.cpp/ggml/src/ggml.c:15254: fatal error
> 
> > Either way, this assert is triggered when computing on the CPU. It can only be triggered if a GEMM or dequantization kernel is missing. Can you post the list of quantization types being used? Thanks.
> 
> While not the same model, I've had a similar report come in and myself had an issue testing perplexity on CPU backend with Kimi-K2-Instruct-UD-IQ1_S [described here on hf](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF/discussions/1#687979526323c34af09d40c7)
> 
> jukofyork suggested omitting `-fmoe` and now it is running for me.
> 
> While the error looks similar, mine would *immedeately* crash, not compute some and crash later.

> 👤 **magikRUKKOLA** replied on **2025-07-17** at **23:33:55**
> 
> > While not the same model, I've had a similar report come in and myself had an issue testing perplexity on CPU backend with Kimi-K2-Instruct-UD-IQ1_S [described here on hf](https://huggingface.co/ubergarm/Kimi-K2-Instruct-GGUF/discussions/1#687979526323c34af09d40c7)
> 
> it might be the same issue.  mine llama-perplexity would crash right after the start if:
> ```
> -ub 512
> ```
> 
> but with this:
> ```
> -b $((8 * 512)) -ub $((4 * 512)) \
> ```
> 
> it would crash at 561th batch.  at least it happened two times already.  not sure why
> 
> [EDIT]: THREE TIMES.  It happened THREE TIMES in the row.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **00:01:31**
> 
> > jukofyork suggested omitting `-fmoe` and now it is running for me.
> 
> But that would affect the PPL, right?

> 👤 **Thireus** replied on **2025-07-18** at **00:08:18**
> 
> @magikRUKKOLA, please try another round of quant_downloader.sh using the version I uploaded, it is possible you have some corrupted shards.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **00:35:08**
> 
> @Thireus @ubergarm 
> 
> yeah, the removal of -fmoe seems to do the trick.  but i am not sure if the batches config 16k and 8k is correct (a least its calculating the ppl about three times faster than with a regular 2k 0.5k batches.
> 
> ```
> 
> ....................................................................................................
> llama_new_context_with_model: n_ctx      = 16384
> llama_new_context_with_model: n_batch    = 16384
> llama_new_context_with_model: n_ubatch   = 8192
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 3
> llama_new_context_with_model: attn_max_b = 512
> llama_new_context_with_model: fused_moe  = 0
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:      CUDA0 KV buffer size =   296.45 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =   286.89 MiB
> llama_new_context_with_model: KV self size  =  583.31 MiB, c^KV (q8_0):  583.31 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =    15.78 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size =  7848.00 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size =  4264.00 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =   736.09 MiB
> llama_new_context_with_model: graph nodes  = 5677
> llama_new_context_with_model: graph splits = 207
> 
> system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
> perplexity: tokenizing the input ..
> perplexity: tokenization took 1091.2 ms
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=16384, n_seq=32
> perplexity: 130.84 seconds per pass - ETA 38.22 minutes
> [1]2.5251,[2]3.2261,[3]2.3356,[4]1.9465,[5]1.7619,[6]1.6229,[7]1.5316,[8]1.4672,[9]1.4207,[10]1.3814,[11]1.3655,[12]1.3840,[13]1.3952,[14]1.5139,[15]1.6396,[16]1.6936,[17]1.8489,[18]1.9697,[19]1.9363,[20]1.9228,[21]2.0226,[22]1.9961,[23]1.9707,[24]1.9834,[25]1.9558,[26]1.9352,[27]1.9782,[28]1.9877,[29]2.0324,[30]2.0624,[31]2.0929,[32]2.1095,[33]2.1458,[34]2.1890,[35]2.2348,[36]2.2841,[37]2.3205,[38]2.3667,[39]2.4096,[40]2.4673,[41]2.5031,[42]2.5149,[43]2.5602,[44]2.5740,[45]2.6509,[46]2.6986,[47]2.6566,[48]2.6135,[49]2.5894,[50]2.6066,[51]2.6491,[52]2.6635,[53]2.7149,[54]2.7293,[55]2.7604,[56]2.7903,[57]2.8026,[58]2.8337,[59]2.8439,[60]2.8864,[61]2.9255,[62]2.9728,[63]3.0037,[64]3.0435,[65]3.0530,[66]3.0382,[67]3.0150,[68]3.0401,[69]3.0368,[70]3.0473,[71]3.0658,[72]3.0810,[73]3.0946,[74]3.1176,[75]3.0977,[76]3.0543,[77]3.0131,[78]3.0076,[79]2.9864,[80]2.9686,[81]2.9341,[82]2.9364,[83]2.9075,[84]2.8752,[85]2.8432,[86]2.8205,[87]2.8148,[88]2.7892,[89]2.7727,[90]2.7490,[91]2.7215,[92]2.6982,[93]2.6732,[94]2.6494,[95]2.6291,[96]2.6269,[97]2.6334,[98]2.6191,[99]2.6031,[100]2.6041,[101]2.5967,[102]2.6125,[103]2.6363,[104]2.6539,[105]2.6509,[106]2.6736,[107]2.6979,[108]2.7173,[109]2.7497,[110]2.7828,[111]2.8010,[112]2.7770,[113]2.7643,[114]2.7437,[115]2.7296,[116]2.7159,[117]2.6949,[118]2.6749,[119]2.6554,[120]2.6380,[121]2.6224,[122]2.6059,[123]2.5897,[124]2.5715,[125]2.5548,[126]2.5389,[127]2.5259,[128]2.5161,[129]2.5049,[130]2.4929,[131]2.4851,[132]2.4904,[133]2.4997,[134]2.5054,[135]2.5155,[136]2.5304,[137]2.5431,[138]2.5511,[139]2.5620,[140]2.5634,[141]2.5651,[142]2.5644,[143]2.5656,[144]2.5634,[145]2.5560,[146]2.5548,[147]2.5595,[148]2.5599,[149]2.5615,[150]2.5564,[151]2.5548,[152]2.5523,[153]2.5491,[154]2.5494,[155]2.5534,[156]2.5554,[157]2.5615,[158]2.5698,[159]2.5723,[160]2.5811,[161]2.5892,[162]2.5988,[163]2.6025,[164]2.6220,[165]2.6443,[166]2.6610,[167]2.6724,[168]2.6954,[169]2.7172,[170]2.7371,[171]2.7587,[172]2.7439,[173]2.7286,[174]2.7160,[175]2.7037,[176]2.6926,[177]2.6815,[178]2.6697,[179]2.6567,[180]2.6603,[181]2.6744,[182]2.6892,[183]2.7029,[184]2.7160,[185]2.7260,[186]2.7418,[187]2.7571,[188]2.7708,[189]2.7812,[190]2.7822,[191]2.7896,[192]2.7928,[193]2.7978,[194]2.8169,[195]2.8255,[196]2.8385,[197]2.8486,[198]2.8530,[199]2.8584,[200]2.8575,[201]2.8716,[202]2.8664,[203]2.8717,[204]2.8751,[205]2.8751,[206]2.8779,[207]2.8855,[208]2.8943,[209]2.9032,[210]2.9034,[211]2.8992,[212]2.9001,[213]2.9073,[214]2.9090,[215]2.9140,[216]2.9146,[217]2.9097,[218]2.9101,[219]2.9114,[220]2.9113,[221]2.9119,[222]2.9119,[223]2.9125,[224]2.9171,[225]2.9187,[226]2.9111,[227]2.9082,[228]2.9103,[229]2.9143,[230]2.9205,[231]2.9270,[232]2.9194,[233]2.9122,[234]2.9131,[235]2.9110,[236]2.9193,[237]2.9268,[238]2.9357,[239]2.9454,[240]2.9543,[241]2.9651,[242]2.9788,[243]2.9902,[244]2.9980,[245]3.0090,[246]3.0195,[247]3.0181,[248]3.0140,[249]3.0117,[250]3.0059,[251]3.0038,[252]3.0064,[253]3.0101,[254]3.0173,[255]3.0233,[256]3.0270,[257]3.0294,[258]3.0305,[259]3.0340,[260]3.0365,[261]3.0380,[262]3.0375,[263]3.0425,[264]3.0447,[265]3.0451,[266]3.0467,[267]3.0488,[268]3.0522,[269]3.0551,[270]3.0546,[271]3.0529,[272]3.0466,[273]3.0461,[274]3.0399,[275]3.0296,[276]3.0192,[277]3.0212,[278]3.0311,[279]3.0367,[280]3.0443,[281]3.0517,[282]3.0575,[283]3.0636,[284]3.0695,[285]3.0831,[286]3.0852,[287]3.0885,[288]3.0934,[289]3.0957,[290]3.0882,[291]3.0795,[292]3.0771,[293]3.0767,[294]3.0742,[295]3.0720,[296]3.0739,[297]3.0745,[298]3.0798,[299]3.0852,[300]3.0880,[301]3.0918,[302]3.0936,[303]3.0949,[304]3.0946,[305]3.1058,[306]3.1130,[307]3.1236,[308]3.1131,[309]3.1080,[310]3.0991,[311]3.1018,[312]3.1031,[313]3.1077,[314]3.1100,[315]3.1132,[316]3.1147,[317]3.1165,[318]3.1170,[319]3.1176,[320]3.1216,[321]3.1217,[322]3.1230,[323]3.1292,[324]3.1302,[325]3.1354,[326]3.1397,[327]3.1434,[328]3.1457,[329]3.1474,[330]3.1539,[331]3.1567,[332]3.1610,[333]3.1600,[334]3.1605,[335]3.1612,[336]3.1613,[337]3.1622,[338]3.1622,[339]3.1646,[340]3.1679,[341]3.1733,[342]3.1820,[343]3.1907,[344]3.1956,[345]3.1875,[346]3.1802,[347]3.1749,[348]3.1678,[349]3.1638,[350]3.1627,[351]3.1671,[352]3.1812,[353]3.1900,[354]3.2022,[355]3.2107,[356]3.2160,[357]3.2273,[358]3.2368,[359]3.2399,[360]3.2459,[361]3.2550,[362]3.2630,[363]3.2683,[364]3.2745,[365]3.2801,[366]3.2897,[367]3.2980,[368]3.3046,[369]3.3121,[370]3.3201,[371]3.3326,[372]3.3406,[373]3.3440,[374]3.3473,[375]3.3517,[376]3.3639,[377]3.3742,[378]3.3770,[379]3.3769,[380]3.3737,[381]3.3781,[382]3.3836,[383]3.3868,[384]3.3909,[385]3.3948,[386]3.4002,[387]3.4056,[388]3.4086,[389]3.3990,[390]3.3902,[391]3.3804,[392]3.3753,[393]3.3662,[394]3.3578,[395]3.3493,[396]3.3398,[397]3.3316,[398]3.3226,[399]3.3129,[400]3.3047,[401]3.2952,[402]3.2856,[403]3.2776,[404]3.2681,[405]3.2592,[406]3.2500,[407]3.2413,[408]3.2329,[409]3.2248,[410]3.2193,[411]3.2198,[412]3.2154,[413]3.2168,[414]3.2182,[415]3.2150,[416]3.2148,[417]3.2165,[418]3.2108,[419]3.2118,[420]3.2092,[421]3.2081,[422]3.2087,[423]3.2082,[424]3.2121,[425]3.2119,[426]3.2121,[427]3.2112,[428]3.2135,[429]3.2146,[430]3.2169,[431]3.2177,[432]3.2167,[433]3.2131,[434]3.2130,[435]3.2059,[436]3.2000,[437]3.1962,[438]3.1947,[439]3.1918,[440]3.1966,[441]3.2016,[442]3.2089,[443]3.2069,[444]3.2075,[445]3.2082,[446]3.2120,[447]3.2150,[448]3.2171,[449]3.2201,[450]3.2238,[451]3.2268,[452]3.2286,[453]3.2298,[454]3.2286,[455]3.2309,[456]3.2315,[457]3.2342,[458]3.2392,[459]3.2397,[460]3.2398,[461]3.2369,[462]3.2402,[463]3.2470,[464]3.2518,[465]3.2453,[466]3.2433,[467]3.2418,[468]3.2433,[469]3.2409,[470]3.2381,[471]3.2386,[472]3.2393,[473]3.2385,[474]3.2375,[475]3.2385,[476]3.2371,[477]3.2363,[478]3.2370,[479]3.2386,[480]3.2410,[481]3.2373,[482]3.2407,[483]3.2402,[484]3.2439,[485]3.2501,[486]3.2532,[487]3.2567,[488]3.2617,[489]3.2642,[490]3.2685,[491]3.2742,[492]3.2781,[493]3.2778,[494]3.2789,[495]3.2811,[496]3.2830,[497]3.2858,[498]3.2865,[499]3.2861,[500]3.2898,[501]3.2944,[502]3.2931,[503]3.2919,[504]3.2938,[505]3.2970,[506]3.3047,[507]3.3077,[508]3.3111,[509]3.3041,[510]3.2989,[511]3.2926,[512]3.2882,[513]3.2822,[514]3.2806,[515]3.2823,[516]3.2774,[517]3.2774,[518]3.2764,[519]3.2763,[520]3.2800,[521]3.2786,[522]3.2772,[523]3.2821,[524]3.2809,[525]3.2793,[526]3.2749,[527]3.2701,[528]3.2669,[529]3.2640,[530]3.2613,[531]3.2586,[532]3.2534,[533]3.2477,[534]3.2433,[535]3.2441,[536]3.2465,[537]3.2493,[538]3.2513,[539]3.2538,[540]3.2588,[541]3.2617,[542]3.2640,[543]3.2588,[544]3.2548,[545]3.2546,[546]3.2485,[547]3.2425,[548]3.2365,[549]3.2302,[550]3.2244,[551]3.2187,[552]3.2132,[553]3.2078,[554]3.2059,[555]3.2042,[556]3.2071,[557]3.2108,[558]3.2166,[559]3.2206,[560]3.2259,[561]3.2241,
> Final estimate: PPL = 3.2241 +/- 0.01704
> 
> llama_print_timings:        load time =   77787.27 ms
> llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings: prompt eval time = 1955430.02 ms / 287232 tokens (    6.81 ms per token,   146.89 tokens per second)
> llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings:       total time = 2015890.97 ms / 287233 tokens
> ```
> 
> as related to the quant_downloader.sh.  I updated the version but everything is correct.
> 
> ```
> [2025-07-18 00:30:09] Verifying shard sequence completeness
> [2025-07-18 00:30:09] All shards from 00001 to 01148 are present.
> Download and verification complete. Enjoy!
> ```

> 👤 **Thireus** replied on **2025-07-18** at **01:11:36**
> 
> Ah indeed, might be worth trying to reduce the batch size.
> 
> The Collab links won't display the recipe. I would suggest to upload the recipe files somewhere (adding the ppl in the name). I can also add them to the GitHub examples later.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **01:25:20**
> 
> > I can also add them to the GitHub examples later.
> 
> I bet that would be the easiest solution due to my laziness.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **01:35:35**
> 
> > Ah indeed, might be worth trying to reduce the batch size.
> 
> I have no idea what's going on with my system if I try to make the u_batch=512.  Everything is pretty strange.  Its either crashing right at the start or its doing something like this:
> 
> [EDIT]:  actually, the following might be unrelated since the ik_llama.cpp doesn't use the p2p functionality.  Its probably from the run of p2pBandwidthLatencyTest.
> [EDIT2]: actually, nope.  p2pBandwidthLatencyTest seems unrelated.  Very strange.
> [EDIT3]: likely was related to the absence of swap file.
> 
> ```
> 
> [Fri Jul 18 01:02:39 2025] NVRM: knvlinkCoreShutdownDeviceLinks_IMPL: Need to shutdown all links unilaterally for GPU1
> [Fri Jul 18 01:02:39 2025] NVRM: iovaspaceDestruct_IMPL: 4 left-over mappings in IOVAS 0x200
> [Fri Jul 18 01:02:39 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:592
> [Fri Jul 18 01:02:39 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:601
> [Fri Jul 18 01:02:39 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:592
> [Fri Jul 18 01:02:39 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:601
> [Fri Jul 18 01:02:39 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:592
> [Fri Jul 18 01:02:39 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:601
> [Fri Jul 18 01:02:39 2025] NVRM: nvAssertFailedNoLog: Assertion failed: Sysmemdesc outlived its attached pGpu @ mem_desc.c:1514
> [Fri Jul 18 01:02:39 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:592
> [Fri Jul 18 01:02:39 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:601
> [Fri Jul 18 01:02:39 2025] NVRM: knvlinkCoreShutdownDeviceLinks_IMPL: Need to shutdown all links unilaterally for GPU0
> [Fri Jul 18 01:12:12 2025] NVRM: knvlinkCoreShutdownDeviceLinks_IMPL: Need to shutdown all links unilaterally for GPU1
> [Fri Jul 18 01:12:12 2025] NVRM: iovaspaceDestruct_IMPL: 4 left-over mappings in IOVAS 0x200
> [Fri Jul 18 01:12:12 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:592
> [Fri Jul 18 01:12:12 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:601
> [Fri Jul 18 01:12:12 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:592
> [Fri Jul 18 01:12:12 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:601
> [Fri Jul 18 01:12:12 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:592
> [Fri Jul 18 01:12:12 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:601
> [Fri Jul 18 01:12:12 2025] NVRM: nvAssertFailedNoLog: Assertion failed: Sysmemdesc outlived its attached pGpu @ mem_desc.c:1514
> [Fri Jul 18 01:12:12 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:592
> [Fri Jul 18 01:12:12 2025] NVRM: nvAssertFailedNoLog: Assertion failed: pIOVAS != NULL @ io_vaspace.c:601
> [Fri Jul 18 01:12:13 2025] NVRM: knvlinkCoreShutdownDeviceLinks_IMPL: Need to shutdown all links unilaterally for GPU0
> ```
> 
> kernel parameters:
> ```
> cat /proc/cmdline
> BOOT_IMAGE=/vmlinuz-6.12.35+deb13-amd64 root=/dev/mapper/xxx ro quiet rd.auto=1 iommu=pt amd_iommu=on pci=realloc pcie_aspm=off nomodeset
> ```
> 
> Ha!  After the log above is started to output the data:
> 
> ```
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
> 
> perplexity: 649.05 seconds per pass - ETA 25 hours 17.13 minutes
> [1]2.5320,[2]3.2286,[3]2.3362,[4]1.9525,
> ```
> 
> The numbers with the 512 u_batch size seems to be somewhat [higher].  I have no idea why.

> 👤 **Panchovix** replied on **2025-07-18** at **02:21:10**
> 
> @Thireus just ended testing your new recipe, at about 4bpw
> 
> ```
> llm_load_print_meta: model size       = 311.039 GiB (3.976 BPW) 
> Final estimate: PPL = 3.2452 +/- 0.01719
> ```
> Impressive! Better than IQ4_XS while weighting ~22GB less. Also this one is "just" 1% worse than Q8_0 (or the Q8_0 is just 1% better).
> 
> Will keep this one as well, really good.
> 
> If someday you have a bit of time for DeepSeek V3 0324 it would be really appreciated. I can't quite run Kimi K2 at decent quality so V3 0324 is my way to go for non thinking models.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **02:47:16**
> 
> > I can't quite run Kimi K2 at decent quality
> 
> I tried it today.  Its flying (10 tps+ decode) with DDR4 and a few GPUs.
> 
> > so V3 0324 is my way to go for non thinking models.
> 
> Apparently the DeepSeek-TNG-R1T2-Chimera is a new thing (as the Thireus mentioned above).

> 👤 **Panchovix** replied on **2025-07-18** at **02:55:12**
> 
> @magikRUKKOLA I can run Kimi K2 at max 2.5-2.6bpw which is quite worse than any Deepseek v3 3.4bpw or more quant, so not much sense on trying to run it.
> 
> Chimera is like a mix IIRC, it still thinks but just less. It's pretty good though.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **03:54:39**
> 
> Note on THIREUS-6.2478 real performance with two RTX 3090:
> (28k prefill/prompt on 112k context;  100 tps prefill, ~5.4 tps decode)
> 
> ```
> llama_new_context_with_model: n_ctx      = 114688
> llama_new_context_with_model: n_batch    = 8192
> llama_new_context_with_model: n_ubatch   = 4096
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 3
> llama_new_context_with_model: attn_max_b = 256
> llama_new_context_with_model: fused_moe  = 1
> llama_new_context_with_model: ser        = -1, 0
> 
> INFO [           print_timings] prompt eval time     =  268774.70 ms / 27088 tokens (    9.92 ms per token,   100.78 tokens per second) | tid="140025814818816" timestamp=1752810557 id_slot=0 id_task=0 t_prompt_processing=268774.702 n_prompt_tokens_processed=27088 t_token=9.922279311872416 n_tokens_second=100.78329470159733
> INFO [           print_timings] generation eval time =  337804.37 ms /  1815 runs   (  186.12 ms per token,     5.37 tokens per second) | tid="140025814818816" timestamp=1752810557 id_slot=0 id_task=0 t_token_generation=337804.371 n_decoded=1815 t_token=186.11811074380165 n_tokens_second=5.372932252555134
> ```
> 
> I was unable to run it with 128k context with 4k/2k batch sizes.  Will try with three GPUs later on.
> 
> [EDIT]:
> 
> for a full context of 112k:
> 
> ```
> main: n_kv_max = 114688, n_batch = 8192, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 99, n_threads = 64, n_threads_batch = 64
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |   35.114 |   116.65 |  158.917 |     6.44 |
> |  4096 |   1024 |   4096 |   36.364 |   112.64 |  162.022 |     6.32 |
> |  4096 |   1024 |   8192 |   37.751 |   108.50 |  166.520 |     6.15 |
> |  4096 |   1024 |  12288 |   38.855 |   105.42 |  170.334 |     6.01 |
> |  4096 |   1024 |  16384 |   41.047 |    99.79 |  174.222 |     5.88 |
> |  4096 |   1024 |  20480 |   42.162 |    97.15 |  178.118 |     5.75 |
> |  4096 |   1024 |  24576 |   43.843 |    93.42 |  182.030 |     5.63 |
> |  4096 |   1024 |  28672 |   46.028 |    88.99 |  186.261 |     5.50 |
> |  4096 |   1024 |  32768 |   49.664 |    82.47 |  189.186 |     5.41 |
> |  4096 |   1024 |  36864 |   52.154 |    78.54 |  193.756 |     5.28 |
> |  4096 |   1024 |  40960 |   54.834 |    74.70 |  196.732 |     5.21 |
> |  4096 |   1024 |  45056 |   57.470 |    71.27 |  201.112 |     5.09 |
> |  4096 |   1024 |  49152 |   60.232 |    68.00 |  204.763 |     5.00 |
> |  4096 |   1024 |  53248 |   62.919 |    65.10 |  209.342 |     4.89 |
> |  4096 |   1024 |  57344 |   65.624 |    62.42 |  213.462 |     4.80 |
> |  4096 |   1024 |  61440 |   68.161 |    60.09 |  216.063 |     4.74 |
> |  4096 |   1024 |  65536 |   72.260 |    56.68 |  220.774 |     4.64 |
> |  4096 |   1024 |  69632 |   74.987 |    54.62 |  223.362 |     4.58 |
> |  4096 |   1024 |  73728 |   77.669 |    52.74 |  228.419 |     4.48 |
> |  4096 |   1024 |  77824 |   80.511 |    50.88 |  231.139 |     4.43 |
> |  4096 |   1024 |  81920 |   83.341 |    49.15 |  235.590 |     4.35 |
> |  4096 |   1024 |  86016 |   86.171 |    47.53 |  240.463 |     4.26 |
> |  4096 |   1024 |  90112 |   88.941 |    46.05 |  242.988 |     4.21 |
> |  4096 |   1024 |  94208 |   91.681 |    44.68 |  247.037 |     4.15 |
> |  4096 |   1024 |  98304 |   94.738 |    43.24 |  250.521 |     4.09 |
> |  4096 |   1024 | 102400 |   97.487 |    42.02 |  254.901 |     4.02 |
> |  4096 |   1024 | 106496 |  100.061 |    40.93 |  258.541 |     3.96 |
> |  4096 |   1024 | 110592 |  102.824 |    39.83 |  262.444 |     3.90 |
> ```

> 👤 **Thireus** replied on **2025-07-18** at **05:54:08**
> 
> @Panchovix - Very good, glad you like it. I think I must create a little guide to explain how to use the collab tool to generate these recipes. If you notice on the initial recipe I provided the quants were not quite well distributed, that's why the ppl wasn't optimum.
> 
> @magikRUKKOLA - Pretty sure your initial ppl computation was broken. If you can run it again at some point I would be really curious to know how much you get now. There is a chance it might end up being around 3.25 (which wouldn't be great), if that end up the case this would be due to the GPU tensors, so if you're willing to compromise a bit on context size we could bump these tensors a little more towards Q8. But there is still a chance you end up towards 3.23, which would be good.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **10:39:30**
> 
> > Pretty sure your initial ppl computation was broken. If you can run it again at some point I would be really curious to know how much you get now.
> 
> With the batch size = 512 and without -fmoe ?  Well, I am not sure if that one is correct because:
> 
> ```
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
> ggml_cuda_host_malloc: failed to allocate 505.00 MiB of pinned memory: initialization error
> perplexity: 60.24 seconds per pass - ETA 2 hours 20.82 minutes
> [1]2.5014,[2]3.2002,[3]2.3253,[4]1.9436,
> ```
> 
> ggml_cuda_host_malloc failed?  I am not sure that after that the calculations are correct.

> 👤 **ikawrakow** replied on **2025-07-18** at **10:41:58**
> 
> This message is totally harmless. I should remove it to avoid confusion.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **10:44:56**
> 
> > This message is totally harmless. I should remove it to avoid confusion.
> 
> okay I am re-running that quant then without -fmoe and with the -ub=512 then.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **10:47:45**
> 
> > so if you're willing to compromise a bit on context size we could bump these tensors a little more towards Q8. But there is still a chance you end up towards 3.23, which would be good.
> 
> Sure, why not? :)

> 👤 **Thireus** replied on **2025-07-18** at **11:20:06**
> 
> @magikRUKKOLA - what is your memory usage (VRAM and RAM) like when the model runs? Is there still any left?

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **15:42:13**
> 
> > perplexity: 649.05 seconds per pass - ETA 25 hours 17.13 minutes
> 
> A small update.  Apparently the extremely low speed above was related to the fact that I was rsync'ing some stuff to the USB3 connected storage.  Now with a small batch it run only three times slower.
> 
> ```
> llama_new_context_with_model: n_ctx      = 2048
> llama_new_context_with_model: n_batch    = 2048
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 3
> llama_new_context_with_model: attn_max_b = 256
> llama_new_context_with_model: fused_moe  = 0
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:      CUDA0 KV buffer size =    37.07 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =    35.87 MiB
> llama_new_context_with_model: KV self size  =   72.91 MiB, c^KV (q8_0):   72.91 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     1.97 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size =   503.00 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size =   477.50 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =   226.01 MiB
> llama_new_context_with_model: graph nodes  = 3664
> llama_new_context_with_model: graph splits = 119
> 
> system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
> perplexity: tokenizing the input ..
> perplexity: tokenization took 1448.08 ms
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
> perplexity: 37.84 seconds per pass - ETA 1 hours 28.45 minutes
> [1]2.5320,[2]3.2286,[3]2.3362,[4]1.9525,[5]1.7658,[6]1.6259,[7]1.5333,[8]1.4691,[9]1.4220,[10]1.3823,[11]1.3659,[12]1.3791,[13]1.3905,[14]1.5091,[15]1.6353,[16]1.6899,[17]1.8443,[18]1.9649,[19]1.9296,[20]1.9172,[21]2.0180,[22]1.9916,[23]1.9661,[24]1.9799,[25]1.9523,[26]1.9319,[27]1.9747,[28]1.9846,[29]2.0298,[30]2.0594,[31]2.0897,[32]2.1062,[33]2.1426,[34]2.1852,[35]2.2309,[36]2.2801,[37]2.3162,[38]2.3621,[39]2.4050,[40]2.4625,[41]2.4988,[42]2.5107,[43]2.5560,[44]2.5697,[45]2.6466,[46]2.6942,[47]2.6518,[48]2.6081,[49]2.5845,[50]2.6017,[51]2.6438,[52]2.6584,[53]2.7089,[54]2.7234,[55]2.7547,[56]2.7848,[57]2.7971,[58]2.8282,[59]2.8387,[60]2.8811,[61]2.9201,[62]2.9658,[63]2.9964,[64]3.0367,[65]3.0468,[66]3.0325,[67]3.0092,[68]3.0345,[69]3.0310,[70]3.0418,[71]3.0602,[72]3.0761,[73]3.0897,[74]3.1122,[75]3.0924,[76]3.0490,
> ```
> 
> but why now, without the:
> 
> ```
> ggml_cuda_host_malloc: failed to allocate 505.00 MiB of pinned memory: initialization error
> ```
> 
> it shows some different perplexities for each batch?  Before it was like:
> 
> ```
> [1]2.5014,[2]3.2002,[3]2.3253,[4]1.9436,
> ```
> 
> I don't believe I changed anything in the settings.  I didn't use the -fmoe and everything seemed to be the same.  But now the numbers are different.  Very strange.  Okay I will not f**k around with the batch sizes and only will turn off -fmoe if it crashes (but for what reason?).  But why the different batch size affects the ppl?  It should not, I gather?

> 👤 **Thireus** replied on **2025-07-18** at **15:46:58**
> 
> @magikRUKKOLA - something was corrupted in your first attempt, you were getting a ppl much lower than BF16 which didn't make sense.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **15:48:56**
> 
> > @magikRUKKOLA - what is your memory usage (VRAM and RAM) like when the model runs? Is there still any left?
> 
> Ah s**t.  I didn't include that information in the logs.  But as related to the VRAM -- usually I am trying to fill the VRAM of every GPU with KV-cache since I want a longer (160k for Deepseek ideally) context.  With that 6.2+bpw quant I was able to fit 112k context with 4k/2k batches.  As related to the RAM ... I will let you know everything for 6.2+bpw and 6.1+bpw quants a little bit later on once I will get the perplexities.

> 👤 **ubergarm** replied on **2025-07-18** at **15:56:27**
> 
> @magikRUKKOLA 
> 
> > But why the different batch size affects the ppl? It should not, I gather?
> 
> *EDIT*: Read down and ik shows you can indeed increase `-ub 4096 -b 4096` while preserving `n_ctx = 512` which is the important bit for comparing final perplexity.
> 
> I am very careful not to change things when measuring perplexity. I'm not 100% sure but changing batch stuff could change context stuff (~if you increase ub above ctx it forces ctx higher which effects PPL i think is what u are seeing~).  I keep it simple and use the same command consistent across all measurements. As pointed out though the `-seed` doesn't matter as its not used. And as @Panchovix says below the defaults are `-ub 512 -b 2048 -c 512` which is what i'm doing here, but i am just explicit on the context.
> 
> ```
> numactl -N 0 -m 0 \
> ./build/bin/llama-perplexity \
>     -m "$model" \
>     -f wiki.test.raw \
>     --seed 1337 \
>     -fa -fmoe \
>     -mla 3 \
>     --ctx-size 512 \
>     --numa numactl \
>     --threads 128 \
>     --threads-batch 192 \
>     --no-mmap
> ```

> 👤 **Panchovix** replied on **2025-07-18** at **15:56:54**
> 
> For PPL and compare to @ubergarm or my results, use batch size 2048 and ubatch 512 (aka default values for both)
> 
> If you change either of those results will be different.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **16:26:12**
> 
> > (if you increase ub above ctx it forces ctx higher which effects PPL i think is what u are seeing).
> 
> Please explain like I am 5 yo.  If I am increasing the ub above ctx (which is 512 tokens) [it means that the context is increased to ub].  Okay.  But how the increased context should affect the PPL if only 512 tokens from it is used?

> 👤 **ikawrakow** replied on **2025-07-18** at **16:30:54**
> 
> Batch and u-batch size does not affect PPL beyond numerical roundoff. In the early batches you may seem more significant differences, but as calculation progresses, the result should be (nearly) independent on batch and u-batch size. If at the end of the Wikitext2 calculation the difference is greater than 0.01 or so, you should file an issue with your logs.

> 👤 **Panchovix** replied on **2025-07-18** at **16:36:18**
> 
> I got different values last time i tested (i tried to cheat it with -ub 2048 to make it quicker), like, different scale of values. I think I was getting way lower PPL than R1 Q8_0 for example (with IQ3_XXS), but I adjusted the -c to 2048 as well (because I assume 512 fails, but maybe that is wrong).
> 
> Hmm...

> 👤 **ikawrakow** replied on **2025-07-18** at **16:39:23**
> 
> >  but I adjusted the -c to 2048 as well
> 
> Well, that's different. Context size does affect PPL quite a bit. But you can keep the context size at the default (512), and still use larger batch/u-batch.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **16:40:00**
> 
> > . I think I was getting way lower PPL than R1 Q8_0 for example (with IQ3_XXS)
> 
> You mean the final PPL or some of the intermediate values (for each batch) shown in the logs?

> 👤 **Panchovix** replied on **2025-07-18** at **16:41:37**
> 
> @ikawrakow oh that's nice! I though it would error with -b 2048 -ub 2048 and -c 512. Then if PPL remains the same it would save a lot of time.
> 
> @magikRUKKOLA final PPL, got like 2.5 or something near when Q8_0 is 3.2119.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **16:45:19**
> 
> @ikawrakow 
> 
> > Well, that's different. Context size does affect PPL quite a bit. But you can keep the context size at the default (512), and still use larger batch/u-batch.
> 
> But the context is automatically sets up to the batch_size, not u_batch_size.  Example:
> 
> ```
> CUDA_VISIBLE_DEVICES="0,1" \
> /opt/ik_llama.cpp/ik_llama.cpp/build/bin/llama-perplexity \
>     -f /opt/ik_llama.cpp/wiki.test.raw \
>     --model /opt/GGUF-Tool-Suite/GGUF-Tool-Suite/DeepSeek-R1-0528.ROOT-6.2478bpw/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf \
>     --alias THIREUS/DeepSeek-R1-0528-6.2478bpw \
>     --ctx-size $((512)) \
>     -ub $((512)) \
>     --mlock \
>     --seed 3407 \
>     --temp 0.5 --top-k 0 --top-p 1.0 --min-p 0.1 --repeat-penalty 1.0 \
>     -ctk q8_0 \
>     -mla 3 -fa \
>     -amb 256 \
>     --override-tensor exps=CPU \
>     --n-gpu-layers 99 \
>     --threads $(grep ^cpu\\scores /proc/cpuinfo | uniq | awk '{print $4}' | xargs -I{} echo "{}-0" | bc) \
>     --host 0.0.0.0 \
>     --port 8080 \
>     --lookup-cache-dynamic /mnt/data/ik_llama.kv.dump
> ```
> (these are the default settings for testing the PPL I gather)
> 
> That would output such in the logs:
> 
> ```
> llama_new_context_with_model: n_ctx      = 2048
> llama_new_context_with_model: n_batch    = 2048
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 3
> llama_new_context_with_model: attn_max_b = 256
> ```
> 
> so the n_ctx is 2k, not 0.5k as intended (?).  Is that the proper behaviour?

> 👤 **ikawrakow** replied on **2025-07-18** at **16:49:48**
> 
> No, it isn't.
> ```
> ./bin/llama-perplexity -m $model -f ../../llama.cpp/tests/wiki.test.raw -t 1 -ngl 100 -fa -b 4096 -ub 4096
> ...
> perplexity: tokenizing the input ..
> perplexity: tokenization took 551.551 ms
> perplexity: calculating perplexity over 655 chunks, n_ctx=512, batch_size=4096, n_seq=8
> perplexity: 0.77 seconds per pass - ETA 1.05 minutes
> [1]8.2845,[2]10.0334,[3]10.6426,[4]12.1269,[5]11.9298
> ```

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **16:53:00**
> 
> @Thireus @ikawrakow 
> 
> So I tried to retest the 6.2bpw quant with the default batch size as recommended above the result is about the same.  With the 8k/4k batches it was 3.2241 but with the settings above its about the same, its 3.2240.
> 
> ```
> 
> ....................................................................................................
> llama_new_context_with_model: n_ctx      = 2048
> llama_new_context_with_model: n_batch    = 2048
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 3
> llama_new_context_with_model: attn_max_b = 256
> llama_new_context_with_model: fused_moe  = 0
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:      CUDA0 KV buffer size =    37.07 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =    35.87 MiB
> llama_new_context_with_model: KV self size  =   72.91 MiB, c^KV (q8_0):   72.91 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     1.97 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size =   503.00 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size =   477.50 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =   226.01 MiB
> llama_new_context_with_model: graph nodes  = 3664
> llama_new_context_with_model: graph splits = 119
> 
> system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
> perplexity: tokenizing the input ..
> perplexity: tokenization took 1448.08 ms
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
> perplexity: 37.84 seconds per pass - ETA 1 hours 28.45 minutes
> [1]2.5320,[2]3.2286,[3]2.3362,[4]1.9525,[5]1.7658,[6]1.6259,[7]1.5333,[8]1.4691,[9]1.4220,[10]1.3823,[11]1.3659,[12]1.3791,[13]1.3905,[14]1.5091,[15]1.6353,[16]1.6899,[17]1.8443,[18]1.9649,[19]1.9296,[20]1.9172,[21]2.0180,[22]1.9916,[23]1.9661,[24]1.9799,[25]1.9523,[26]1.9319,[27]1.9747,[28]1.9846,[29]2.0298,[30]2.0594,[31]2.0897,[32]2.1062,[33]2.1426,[34]2.1852,[35]2.2309,[36]2.2801,[37]2.3162,[38]2.3621,[39]2.4050,[40]2.4625,[41]2.4988,[42]2.5107,[43]2.5560,[44]2.5697,[45]2.6466,[46]2.6942,[47]2.6518,[48]2.6081,[49]2.5845,[50]2.6017,[51]2.6438,[52]2.6584,[53]2.7089,[54]2.7234,[55]2.7547,[56]2.7848,[57]2.7971,[58]2.8282,[59]2.8387,[60]2.8811,[61]2.9201,[62]2.9658,[63]2.9964,[64]3.0367,[65]3.0468,[66]3.0325,[67]3.0092,[68]3.0345,[69]3.0310,[70]3.0418,[71]3.0602,[72]3.0761,[73]3.0897,[74]3.1122,[75]3.0924,[76]3.0490,[77]3.0082,[78]3.0025,[79]2.9815,[80]2.9636,[81]2.9294,[82]2.9316,[83]2.9026,[84]2.8703,[85]2.8382,[86]2.8157,[87]2.8098,[88]2.7844,[89]2.7681,[90]2.7441,[91]2.7167,[92]2.6931,[93]2.6689,[94]2.6449,[95]2.6249,[96]2.6226,[97]2.6284,[98]2.6143,[99]2.5982,[100]2.5992,[101]2.5916,[102]2.6072,[103]2.6310,[104]2.6488,[105]2.6460,[106]2.6682,[107]2.6923,[108]2.7118,[109]2.7441,[110]2.7771,[111]2.7952,[112]2.7713,[113]2.7585,[114]2.7379,[115]2.7237,[116]2.7103,[117]2.6893,[118]2.6695,[119]2.6499,[120]2.6324,[121]2.6169,[122]2.6008,[123]2.5846,[124]2.5665,[125]2.5498,[126]2.5341,[127]2.5212,[128]2.5114,[129]2.5006,[130]2.4886,[131]2.4805,[132]2.4859,[133]2.4952,[134]2.5009,[135]2.5106,[136]2.5256,[137]2.5384,[138]2.5464,[139]2.5572,[140]2.5586,[141]2.5604,[142]2.5595,[143]2.5608,[144]2.5587,[145]2.5514,[146]2.5501,[147]2.5548,[148]2.5553,[149]2.5569,[150]2.5519,[151]2.5503,[152]2.5479,[153]2.5447,[154]2.5452,[155]2.5492,[156]2.5512,[157]2.5574,[158]2.5657,[159]2.5682,[160]2.5773,[161]2.5853,[162]2.5950,[163]2.5987,[164]2.6182,[165]2.6404,[166]2.6572,[167]2.6687,[168]2.6919,[169]2.7136,[170]2.7334,[171]2.7552,[172]2.7404,[173]2.7251,[174]2.7126,[175]2.7004,[176]2.6895,[177]2.6784,[178]2.6667,[179]2.6539,[180]2.6575,[181]2.6716,[182]2.6865,[183]2.7004,[184]2.7135,[185]2.7235,[186]2.7393,[187]2.7545,[188]2.7684,[189]2.7791,[190]2.7801,[191]2.7873,[192]2.7904,[193]2.7954,[194]2.8144,[195]2.8231,[196]2.8361,[197]2.8462,[198]2.8507,[199]2.8561,[200]2.8552,[201]2.8693,[202]2.8640,[203]2.8692,[204]2.8726,[205]2.8726,[206]2.8754,[207]2.8830,[208]2.8918,[209]2.9008,[210]2.9011,[211]2.8969,[212]2.8979,[213]2.9051,[214]2.9068,[215]2.9117,[216]2.9123,[217]2.9074,[218]2.9078,[219]2.9087,[220]2.9087,[221]2.9093,[222]2.9094,[223]2.9100,[224]2.9146,[225]2.9163,[226]2.9089,[227]2.9060,[228]2.9082,[229]2.9121,[230]2.9182,[231]2.9247,[232]2.9171,[233]2.9097,[234]2.9104,[235]2.9083,[236]2.9166,[237]2.9243,[238]2.9333,[239]2.9431,[240]2.9520,[241]2.9629,[242]2.9767,[243]2.9882,[244]2.9959,[245]3.0068,[246]3.0173,[247]3.0159,[248]3.0118,[249]3.0097,[250]3.0038,[251]3.0017,[252]3.0044,[253]3.0082,[254]3.0155,[255]3.0215,[256]3.0251,[257]3.0277,[258]3.0288,[259]3.0325,[260]3.0350,[261]3.0366,[262]3.0360,[263]3.0412,[264]3.0435,[265]3.0438,[266]3.0453,[267]3.0474,[268]3.0507,[269]3.0537,[270]3.0531,[271]3.0516,[272]3.0453,[273]3.0448,[274]3.0387,[275]3.0286,[276]3.0182,[277]3.0202,[278]3.0300,[279]3.0357,[280]3.0433,[281]3.0507,[282]3.0565,[283]3.0626,[284]3.0687,[285]3.0823,[286]3.0845,[287]3.0879,[288]3.0928,[289]3.0951,[290]3.0877,[291]3.0790,[292]3.0766,[293]3.0761,[294]3.0736,[295]3.0714,[296]3.0732,[297]3.0738,[298]3.0790,[299]3.0847,[300]3.0874,[301]3.0912,[302]3.0930,[303]3.0945,[304]3.0942,[305]3.1056,[306]3.1128,[307]3.1235,[308]3.1130,[309]3.1078,[310]3.0989,[311]3.1017,[312]3.1030,[313]3.1076,[314]3.1099,[315]3.1131,[316]3.1147,[317]3.1166,[318]3.1171,[319]3.1176,[320]3.1216,[321]3.1217,[322]3.1231,[323]3.1293,[324]3.1302,[325]3.1354,[326]3.1394,[327]3.1431,[328]3.1454,[329]3.1471,[330]3.1537,[331]3.1563,[332]3.1606,[333]3.1596,[334]3.1601,[335]3.1608,[336]3.1609,[337]3.1620,[338]3.1619,[339]3.1644,[340]3.1678,[341]3.1732,[342]3.1817,[343]3.1904,[344]3.1953,[345]3.1871,[346]3.1798,[347]3.1746,[348]3.1674,[349]3.1633,[350]3.1622,[351]3.1668,[352]3.1808,[353]3.1897,[354]3.2019,[355]3.2104,[356]3.2157,[357]3.2270,[358]3.2364,[359]3.2396,[360]3.2455,[361]3.2545,[362]3.2627,[363]3.2680,[364]3.2742,[365]3.2798,[366]3.2894,[367]3.2978,[368]3.3044,[369]3.3118,[370]3.3197,[371]3.3324,[372]3.3403,[373]3.3437,[374]3.3469,[375]3.3514,[376]3.3635,[377]3.3739,[378]3.3766,[379]3.3765,[380]3.3733,[381]3.3777,[382]3.3832,[383]3.3863,[384]3.3904,[385]3.3943,[386]3.3997,[387]3.4052,[388]3.4082,[389]3.3986,[390]3.3899,[391]3.3802,[392]3.3751,[393]3.3659,[394]3.3576,[395]3.3491,[396]3.3396,[397]3.3313,[398]3.3224,[399]3.3126,[400]3.3044,[401]3.2949,[402]3.2854,[403]3.2774,[404]3.2679,[405]3.2590,[406]3.2498,[407]3.2411,[408]3.2327,[409]3.2246,[410]3.2190,[411]3.2195,[412]3.2150,[413]3.2165,[414]3.2178,[415]3.2146,[416]3.2143,[417]3.2161,[418]3.2103,[419]3.2114,[420]3.2088,[421]3.2078,[422]3.2084,[423]3.2079,[424]3.2116,[425]3.2115,[426]3.2117,[427]3.2108,[428]3.2131,[429]3.2142,[430]3.2166,[431]3.2175,[432]3.2164,[433]3.2128,[434]3.2127,[435]3.2056,[436]3.1997,[437]3.1960,[438]3.1945,[439]3.1916,[440]3.1963,[441]3.2015,[442]3.2087,[443]3.2066,[444]3.2072,[445]3.2079,[446]3.2117,[447]3.2147,[448]3.2168,[449]3.2198,[450]3.2235,[451]3.2266,[452]3.2284,[453]3.2297,[454]3.2284,[455]3.2307,[456]3.2313,[457]3.2340,[458]3.2390,[459]3.2394,[460]3.2395,[461]3.2366,[462]3.2400,[463]3.2468,[464]3.2515,[465]3.2451,[466]3.2431,[467]3.2415,[468]3.2430,[469]3.2405,[470]3.2376,[471]3.2381,[472]3.2388,[473]3.2380,[474]3.2370,[475]3.2380,[476]3.2365,[477]3.2357,[478]3.2365,[479]3.2381,[480]3.2404,[481]3.2368,[482]3.2402,[483]3.2397,[484]3.2433,[485]3.2495,[486]3.2526,[487]3.2560,[488]3.2611,[489]3.2635,[490]3.2679,[491]3.2736,[492]3.2776,[493]3.2773,[494]3.2784,[495]3.2805,[496]3.2823,[497]3.2851,[498]3.2858,[499]3.2853,[500]3.2890,[501]3.2935,[502]3.2923,[503]3.2910,[504]3.2929,[505]3.2963,[506]3.3039,[507]3.3070,[508]3.3103,[509]3.3034,[510]3.2981,[511]3.2919,[512]3.2875,[513]3.2815,[514]3.2799,[515]3.2816,[516]3.2768,[517]3.2768,[518]3.2758,[519]3.2757,[520]3.2794,[521]3.2780,[522]3.2766,[523]3.2816,[524]3.2803,[525]3.2787,[526]3.2742,[527]3.2694,[528]3.2662,[529]3.2634,[530]3.2607,[531]3.2580,[532]3.2528,[533]3.2471,[534]3.2427,[535]3.2436,[536]3.2460,[537]3.2488,[538]3.2509,[539]3.2533,[540]3.2584,[541]3.2613,[542]3.2636,[543]3.2584,[544]3.2543,[545]3.2542,[546]3.2481,[547]3.2420,[548]3.2360,[549]3.2297,[550]3.2240,[551]3.2183,[552]3.2129,[553]3.2074,[554]3.2057,[555]3.2040,[556]3.2069,[557]3.2106,[558]3.2165,[559]3.2206,[560]3.2258,[561]3.2240,
> Final estimate: PPL = 3.2240 +/- 0.01704
> 
> llama_print_timings:        load time =  259919.06 ms
> llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings: prompt eval time = 4934882.60 ms / 287232 tokens (   17.18 ms per token,    58.20 tokens per second)
> llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings:       total time = 4945116.03 ms / 287233 tokens                                                                                                                                                   
> ```

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **16:55:09**
> 
> > No, it isn't.
> 
> Well, I probably looked to the wrong li[n]e.  Hold on.
> 
> so this one:
> ```
> llama_new_context_with_model: n_ctx      = 2048
> ```
> 
> is not the same as:
> 
> ```
> n_ctx=512
> ```
> 
> ?
> 
> So everything looks good in my command above etc.?

> 👤 **Thireus** replied on **2025-07-18** at **17:02:02**
> 
> > So I tried to retest the 6.2bpw quant with the default batch size as recommended above the result is about the same.  With the 8k/4k batches it was 3.2241 but with the settings above its about the same, its 3.2240.
> 
> Good ppl better than what I expected. Might be able to reach 3.218 with some more tweaking, but you'll lose on the context size. Let me know if you want to try another one. But please do tell me how much VRAM and RAN you still have available when running this model at the context size of your liking.

> 👤 **ubergarm** replied on **2025-07-18** at **17:03:50**
> 
> Oh well I learned something new, thanks. One can increase `-ub 4096 -b 4096` while maintaining `-c 512` as shown by ik where it just adjusts the n_seq.
> 
> > perplexity: calculating perplexity over 655 chunks, n_ctx=512, batch_size=4096, n_seq=8
> 
> @magikRUKKOLA 
> 
> You want `n_ctx` to be 512 to be able to compare your results with mine.

> 👤 **ikawrakow** replied on **2025-07-18** at **17:07:20**
> 
> What its used as a context for the PPL calculation is printed in this line
> ```
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
> ```
> just before you start seeing PPL values. The other output is irrelevant (it only means that internally it will create a KV cache of that size so it can hold the tokens of a whole batch, but when running the inference the KQ mask will mask out tokens that are not part of the 512 token context window).

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **18:26:22**
> 
> @Thireus lol somehow the folder with 6.1+bpw quant weighs 510.8 GiB and with 6.2+bpw quant weighs 494.8 GiB.  I tried to test the 6.1+bpw one and its OOMed.  Hm ...
> 
> Interestingly, after the OOM I had to do:
> 
> ```
> nvidia-smi -r
> ```
> 
> to make cuda see the GPUs again.
> 
> Let me try once again...
> 
> ```
> ./run-ik_llama.cpp.sh
> ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
> ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
> ggml_cuda_init: found 2 CUDA devices:
>   Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
> main: build = 3808 (38012f72)
> main: built with cc (Debian 14.2.0-19) 14.2.0 for x86_64-linux-gnu
> main: seed  = 3407
> llama_model_loader: additional 1147 GGUFs metadata loaded.
> llama_model_loader: loaded meta data with 49 key-value pairs and 1147 tensors from /opt/GGUF-Tool-Suite/GGUF-Tool-Suite/DeepSeek-R1-0528.ROOT-6.1382bpw/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf (version GGUF V3 (latest))
> llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
> llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
> llama_model_loader: - kv   1:                               general.type str              = model
> llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
> llama_model_loader: - kv   3:                            general.version str              = 0528
> llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
> llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
> llama_model_loader: - kv   6:                            general.license str              = mit
> llama_model_loader: - kv   7:                      deepseek2.block_count u32              = 61
> llama_model_loader: - kv   8:                   deepseek2.context_length u32              = 163840
> llama_model_loader: - kv   9:                 deepseek2.embedding_length u32              = 7168
> llama_model_loader: - kv  10:              deepseek2.feed_forward_length u32              = 18432
> llama_model_loader: - kv  11:             deepseek2.attention.head_count u32              = 128
> llama_model_loader: - kv  12:          deepseek2.attention.head_count_kv u32              = 128
> llama_model_loader: - kv  13:                   deepseek2.rope.freq_base f32              = 10000.000000
> llama_model_loader: - kv  14: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
> llama_model_loader: - kv  15:                deepseek2.expert_used_count u32              = 8
> llama_model_loader: - kv  16:                          general.file_type u32              = 32
> llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 3
> llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 129280
> llama_model_loader: - kv  19:            deepseek2.attention.q_lora_rank u32              = 1536
> llama_model_loader: - kv  20:           deepseek2.attention.kv_lora_rank u32              = 512
> llama_model_loader: - kv  21:             deepseek2.attention.key_length u32              = 192
> llama_model_loader: - kv  22:           deepseek2.attention.value_length u32              = 128
> llama_model_loader: - kv  23:       deepseek2.expert_feed_forward_length u32              = 2048
> llama_model_loader: - kv  24:                     deepseek2.expert_count u32              = 256
> llama_model_loader: - kv  25:              deepseek2.expert_shared_count u32              = 1
> llama_model_loader: - kv  26:             deepseek2.expert_weights_scale f32              = 2.500000
> llama_model_loader: - kv  27:              deepseek2.expert_weights_norm bool             = true
> llama_model_loader: - kv  28:               deepseek2.expert_gating_func u32              = 2
> llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
> llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
> llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40.000000
> llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
> llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
> llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
> llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
> llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<...
> llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
> llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
> llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
> llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
> llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
> llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
> llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
> llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
> llama_model_loader: - kv  45:               general.quantization_version u32              = 2
> llama_model_loader: - kv  46:                                   split.no u16              = 0
> llama_model_loader: - kv  47:                                split.count u16              = 1148
> llama_model_loader: - kv  48:                        split.tensors.count i32              = 1147
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type q8_0:  287 tensors
> llama_model_loader: - type q6_K:  107 tensors
> llama_model_loader: - type iq4_xs:  305 tensors
> llama_model_loader: - type iq6_k:   45 tensors
> llama_model_loader: - type iq4_ks:   16 tensors
> llama_model_loader: - type iq5_k_r4:   26 tensors
> ```
> 
> [EDIT2]:  I should probably not use -mlock then.

> 👤 **Thireus** replied on **2025-07-18** at **19:49:22**
> 
> @magikRUKKOLA, indeed because not the same quants are used. In the 6.1bpw I've used Q6_K which isn't in the 6.2bpw. Still curious about the free VRAM/RAM that you're left with.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **20:11:57**
> 
> @Thireus 
> 
> ```
> llama-sweep-bench
> 
> RAM:
> 
> VIRT: 517G -> 596G
> RES: 492G -> 480G
> (loading -> running)
> 
> GPU #1: 23.353Gi/24.000Gi
> ~~GPU #2: 20.312Gi/24.000Gi~~
> GPU #2: 20.370Gi/24.000Gi
> ```
> 
> ```
> export MALLOC_CONF="background_thread:true,percpu_arena:phycpu,metadata_thp:auto,dirty_decay_ms:10000,muzzy_decay_ms:60000"
> export LD_PRELOAD=/usr/local/lib/libjemalloc.so
> 
> ulimit -n 9999
> CUDA_VISIBLE_DEVICES="0,1" \
> /opt/ik_llama.cpp/ik_llama.cpp/build/bin/llama-sweep-bench \
>     --warmup-batch \
>     --model /opt/GGUF-Tool-Suite/GGUF-Tool-Suite/DeepSeek-R1-0528.ROOT-6.2478bpw/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf \
>     --alias THIREUS/DeepSeek-R1-0528-6.2478bpw \
>     --ctx-size $((112 * 1024)) \
>     -b $((16 * 512)) -ub $((8 * 512)) \
>     --mlock \
>     --seed 3407 \
>     --temp 0.5 --top-k 0 --top-p 1.0 --min-p 0.1 --repeat-penalty 1.0 \
>     -ctk q8_0 \
>     -mla 3 -fa \
>     -amb 256 \
>     --override-tensor exps=CPU \
>     --n-gpu-layers 99 \
>     --threads $(grep ^cpu\\scores /proc/cpuinfo | uniq | awk '{print $4}' | xargs -I{} echo "{}-0" | bc) \
>     --host 0.0.0.0 \
>     --port 8080 \
>     --lookup-cache-dynamic /mnt/data/ik_llama.kv.dump
> ```
> 
> ```
> ...
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type q8_0:  239 tensors
> llama_model_loader: - type iq4_xs:  305 tensors
> llama_model_loader: - type iq3_k:    1 tensors
> llama_model_loader: - type iq6_k:  101 tensors
> llama_model_loader: - type iq4_ks:    2 tensors
> llama_model_loader: - type iq5_k_r4:  138 tensors
> ...
> llm_load_tensors:      CUDA0 buffer size =  7407.47 MiB
> llm_load_tensors:      CUDA1 buffer size =  7309.40 MiB
> ....................................................................................................
> llama_new_context_with_model: n_ctx      = 114688
> llama_new_context_with_model: n_batch    = 8192
> llama_new_context_with_model: n_ubatch   = 4096
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 3
> llama_new_context_with_model: attn_max_b = 256
> llama_new_context_with_model: fused_moe  = 0
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:      CUDA0 KV buffer size =  2075.08 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =  2008.14 MiB
> llama_new_context_with_model: KV self size  = 4083.19 MiB, c^KV (q8_0): 4083.19 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size = 12400.02 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size = 10612.03 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =  1904.05 MiB
> llama_new_context_with_model: graph nodes  = 45937
> llama_new_context_with_model: graph splits = 207
> 
> main: n_kv_max = 114688, n_batch = 8192, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 99, n_threads = 64, n_threads_batch = 64
> ...
> ```

> 👤 **Thireus** replied on **2025-07-18** at **20:19:28**
> 
> That's very ambitious to run such high context size with this amount of VRAM. But it seems you know what you are doing. If you ever need slightly lower RAM usage you can reduce the quant_assign.sh `--cpu-tensors-max-size` value (it's expressed in GB) from the command you see in the recipe file - you don't need to touch any of the other parameters.

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **22:52:57**
> 
> @Thireus 
> > That's very ambitious to run such high context size with this amount of VRAM.
> 
> Well, at least the ik_llama.cpp doesn't crash out of the blue and the multiple GPUs dramatically increase the prefill so its not slow.  So if the model supports say 160k the question is -- why not use the model fully?  At least in anything related to the software engineering the quality and the length of the context are of the utmost importance.
> 
> Thanks a lot for your work once again!  Your results are impressive!

> 👤 **magikRUKKOLA** replied on **2025-07-18** at **23:17:57**
> 
> Unsloth updated a wide range of quants such as UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL.  That happened exactly when I was downloading the last part of UD-Q3_K_XL.  Now I have to re-download everything again.  ~~What a stupid s**t.~~  Why he didn't use the revisions?
> 
> Now it means that we have to re-test all the updated quants.

> 👤 **ubergarm** replied on **2025-07-19** at **14:16:06**
> 
> @magikRUKKOLA 
> 
> I just noticed most of the unsloth quants were modified somehow, but not sure what changed. Is there a reddit post or blogpost explaining? Sometimes they do that just for the GGUF metadata and doesn't effect the tensors but still requires full upload psure.

> 👤 **ikawrakow** replied on **2025-07-19** at **14:41:41**
> 
> If you download new Unsloth quants, please first make a gguf-dump of the model you have before downloading the new model. Then do a gguf-dump on the new model, compare, and post the difference. I think many people will be curious to know what was changed that was so important that Unsloth felt it is worth making people re-download hundreds of GB of data.

> 👤 **firecoperana** replied on **2025-07-19** at **15:07:04**
> 
> https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/discussions/7
> It's tool calling related. Just need to re-download the first GGUF file.

> 👤 **magikRUKKOLA** replied on **2025-07-19** at **15:50:34**
> 
> > It's tool calling related. Just need to re-download the first GGUF file.
> 
> This story doesn't add up like AT ALL.
> Check this out:
> 
> https://huggingface.co/unsloth/Kimi-K2-Instruct-GGUF/commit/ac691362ab1d5c071d82a115b76ceb0b3ed3b4d3
> ```
> UD-IQ3_XXS/Kimi-K2-Instruct-UD-IQ3_XXS-00003-of-00009.gguf CHANGED
> version https://git-lfs.github.com/spec/v1
> oid sha256:7e756f2fb141dc6b9dc76905485b82997b03537594eeed6f00b000cb9ca8118e
> size 48137425536
> 
> version https://git-lfs.github.com/spec/v1
> oid sha256:2f0fd3546428437dc801c86b3cf5ee38c4b7043874dcc9c61c1c1df97c6fcf7d
> size 48711897728
> ```
> 
> Its plus ~~+6GB~~ +600MB to a third file.  Is it a tool calling template update?  No, its not.
> 
> Same thing goes to the quant I was downloaded (UD-Q3_K_XL).  It grew in size.

> 👤 **ubergarm** replied on **2025-07-19** at **16:26:19**
> 
> fwiw i've observed that Kimi-K2-Instruct is very sensitive to attn/shexp/blk.0.ffn.* quantization (or possibly just attn). i too would like to see the difference in the recipes. i've collected a lot more data and hope to update my graph soon after one more test quant finishes cooking.
> 
> i believe it is possible to do 'revisions' using git branches on hugging face and hope to figure that out to release some updated versions of my quants perhaps

> 👤 **magikRUKKOLA** replied on **2025-07-20** at **12:05:45**
> 
> @Thireus 
> 
> > Will try with three GPUs later on.
> 
> It looks like I was able to run the THIREUS-6.2478bpw with a full 160k context with three GPUs.
> 
> ```
> llm_load_tensors:      CUDA0 buffer size =  3551.63 MiB
> llm_load_tensors:      CUDA1 buffer size =  5350.88 MiB
> llm_load_tensors:      CUDA2 buffer size =  5814.36 MiB
> 
> llama_kv_cache_init:      CUDA0 KV buffer size =  1243.13 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =  2390.64 MiB
> llama_kv_cache_init:      CUDA2 KV buffer size =  2199.39 MiB
> llama_new_context_with_model: KV self size  = 5833.12 MiB, c^KV (q8_0): 5833.12 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size = 17512.02 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size = 14992.02 MiB
> llama_new_context_with_model:      CUDA2 compute buffer size = 14992.03 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =  2672.05 MiB
> 
>  Device 0 [NVIDIA GeForce RTX 3090] PCIe GEN 4@16x RX: 13.18 MiB/s TX: 600.0 KiB/s
>  GPU 1695MHz MEM 9501MHz TEMP  70°C  FAN  56%   POW 134 / 350 W
>  GPU[||                               6%] MEM[||||||||||||||||||23.767Gi/24.000Gi]
> 
>  Device 1 [NVIDIA GeForce RTX 3090] PCIe GEN 4@16x RX: 89.55 MiB/s TX: 350.0 KiB/s
>  GPU 1695MHz MEM 9501MHz TEMP  51°C  FAN  30%   POW 119 / 350 W
>  GPU[|||                              9%] MEM[||||||||||||||||||23.245Gi/24.000Gi]
> 
>  Device 2 [NVIDIA GeForce RTX 3090] PCIe GEN 4@16x RX: 104.5 MiB/s TX: 51.51 MiB/s
>  GPU 1695MHz MEM 9501MHz TEMP  66°C  FAN  39%   POW 137 / 350 W
>  GPU[|||                              9%] MEM[||||||||||||||||||23.511Gi/24.000Gi]
>  ```
> 
> The tensor-split was crucial in my case to spread the model upon the three 24GB VRAM GPUs:
> 
> ```
> 
> export MALLOC_CONF="background_thread:true,percpu_arena:phycpu,metadata_thp:auto,dirty_decay_ms:10000,muzzy_decay_ms:60000"
> export LD_PRELOAD=/usr/local/lib/libjemalloc.so
> 
> ulimit -n 9999
> CUDA_VISIBLE_DEVICES="0,1,2" \
> /opt/ik_llama.cpp/ik_llama.cpp/build/bin/llama-sweep-bench \
>     --warmup-batch \
>     --model /opt/GGUF-Tool-Suite/GGUF-Tool-Suite/DeepSeek-R1-0528.ROOT-6.2478bpw/DeepSeek-R1-0528-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf \
>     --alias THIREUS/DeepSeek-R1-0528-6.2478bpw \
>     --ctx-size $((160 * 1024)) \
>     -b $((16 * 512)) -ub $((8 * 512)) \
>     --mlock \
>     --seed 3407 \
>     --temp 0.5 --top-k 0 --top-p 1.0 --min-p 0.1 --repeat-penalty 1.0 \
>     -ctk q8_0 \
>     -mla 3 -fa \
>     -amb 512 \
>     --split-mode layer \
>     --tensor-split 1,2,2 \
>     --main-gpu 1 \
>     --override-tensor exps=CPU \
>     --n-gpu-layers 99 \
>     --threads $(grep ^cpu\\scores /proc/cpuinfo | uniq | awk '{print $4}' | xargs -I{} echo "{}-0" | bc) \
>     --host 0.0.0.0 \
>     --port 8080 \
>     --lookup-cache-dynamic /mnt/data/ik_llama.kv.dump
> ```
> 
> Let me check if it would actually run the full context without the crashes.
> 
> [EDIT]:
> ```
> main: n_kv_max = 163840, n_batch = 8192, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 99, n_threads = 64, n_threads_batch = 64
> 
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  4096 |   1024 |      0 |   35.022 |   116.96 |  161.903 |     6.32 |
> |  4096 |   1024 |   4096 |   35.661 |   114.86 |  165.473 |     6.19 |
> |  4096 |   1024 |   8192 |   36.277 |   112.91 |  170.106 |     6.02 |
> |  4096 |   1024 |  12288 |   37.890 |   108.10 |  173.710 |     5.89 |
> |  4096 |   1024 |  16384 |   39.613 |   103.40 |  178.062 |     5.75 |
> |  4096 |   1024 |  20480 |   41.999 |    97.53 |  181.954 |     5.63 |
> |  4096 |   1024 |  24576 |   44.429 |    92.19 |  185.710 |     5.51 |
> |  4096 |   1024 |  28672 |   46.932 |    87.28 |  190.164 |     5.38 |
> |  4096 |   1024 |  32768 |   49.930 |    82.04 |  193.364 |     5.30 |
> |  4096 |   1024 |  36864 |   52.424 |    78.13 |  197.987 |     5.17 |
> |  4096 |   1024 |  40960 |   54.932 |    74.57 |  200.872 |     5.10 |
> |  4096 |   1024 |  45056 |   57.564 |    71.16 |  210.702 |     4.86 |
> |  4096 |   1024 |  49152 |   59.976 |    68.29 |  209.510 |     4.89 |
> |  4096 |   1024 |  53248 |   62.569 |    65.46 |  214.199 |     4.78 |
> |  4096 |   1024 |  57344 |   65.047 |    62.97 |  218.920 |     4.68 |
> |  4096 |   1024 |  61440 |   67.577 |    60.61 |  221.387 |     4.63 |
> |  4096 |   1024 |  65536 |   73.532 |    55.70 |  225.942 |     4.53 |
> |  4096 |   1024 |  69632 |   76.194 |    53.76 |  229.005 |     4.47 |
> |  4096 |   1024 |  73728 |   78.875 |    51.93 |  233.785 |     4.38 |
> |  4096 |   1024 |  77824 |   81.289 |    50.39 |  237.003 |     4.32 |
> |  4096 |   1024 |  81920 |   83.952 |    48.79 |  240.589 |     4.26 |
> |  4096 |   1024 |  86016 |   86.502 |    47.35 |  243.834 |     4.20 |
> |  4096 |   1024 |  90112 |   89.140 |    45.95 |  247.685 |     4.13 |
> |  4096 |   1024 |  94208 |   91.803 |    44.62 |  251.893 |     4.07 |
> |  4096 |   1024 |  98304 |   94.444 |    43.37 |  255.264 |     4.01 |
> |  4096 |   1024 | 102400 |   97.225 |    42.13 |  259.784 |     3.94 |
> |  4096 |   1024 | 106496 |   99.876 |    41.01 |  262.594 |     3.90 |
> |  4096 |   1024 | 110592 |  102.649 |    39.90 |  266.853 |     3.84 |
> |  4096 |   1024 | 114688 |  105.461 |    38.84 |  270.749 |     3.78 |
> |  4096 |   1024 | 118784 |  108.210 |    37.85 |  274.773 |     3.73 |
> |  4096 |   1024 | 122880 |  110.963 |    36.91 |  278.748 |     3.67 |
> |  4096 |   1024 | 126976 |  113.851 |    35.98 |  282.601 |     3.62 |
> |  4096 |   1024 | 131072 |  120.226 |    34.07 |  287.020 |     3.57 |
> |  4096 |   1024 | 135168 |  122.992 |    33.30 |  290.092 |     3.53 |
> |  4096 |   1024 | 139264 |  126.097 |    32.48 |  294.785 |     3.47 |
> |  4096 |   1024 | 143360 |  129.193 |    31.70 |  297.875 |     3.44 |
> |  4096 |   1024 | 147456 |  132.032 |    31.02 |  302.351 |     3.39 |
> |  4096 |   1024 | 151552 |  135.192 |    30.30 |  305.997 |     3.35 |
> |  4096 |   1024 | 155648 |  138.069 |    29.67 |  310.111 |     3.30 |
> |  4096 |   1024 | 159744 |  141.029 |    29.04 |  314.032 |     3.26 |
> ...
> ```
> *note:  the TG t/s is a little bit lower than before because my Gigabyte MB doesn't support voltage setup for the RAM so I could only overclock to 3200 MT/s with the default timings with the default 1.2V.
> 
> [EDIT2]:
> temps:
> ```
>  GPU 1695MHz MEM 9501MHz TEMP  73°C  FAN  64%   POW 141 / 350 W
>  GPU 1695MHz MEM 9501MHz TEMP  53°C  FAN  30%   POW 125 / 350 W
>  GPU 1695MHz MEM 9501MHz TEMP  67°C  FAN  48%   POW 143 / 350 W
> 
> CPU0_TEMP        | 01h | ok  | 65.1 | 73 degrees C
> DIMMG0_TEMP      | 04h | ok  | 66.1 | 80 degrees C
> DIMMG1_TEMP      | 05h | ok  | 66.2 | 72 degrees C
> ```
> 
> [EDIT]:
> 
> all good, we're golden!

> 👤 **Thireus** replied on **2025-07-22** at **12:27:08**
> 
> @magikRUKKOLA, @Panchovix - R1T2 is now supported. I have uploaded an example. Since the architecture is the same as R1-0528, all you have to do is run the same parameters from the recipes I provided earlier but change these two parameters only:
> 
> ```
> Model name: DeepSeek-TNG-R1T2-Chimera
> Link to the original model: https://huggingface.co/tngtech/DeepSeek-TNG-R1T2-Chimera
> ```
> 
> As usual, for ease of use you can use Colab: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Thireus/GGUF-Tool-Suite/blob/main/quant_recipe_pipeline.ipynb).
> 
> Any other params can remain the same as the ones found at the bottom of your custom .recipe file.
> 
> I have published one recipe example: https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-TNG-R1T2-Chimera.ROOT-3.0624bpw-3.3657ppl.238GB-GGUF_11GB-GPU_227GB-CPU.13549e6_1ac857a.recipe

> 👤 **magikRUKKOLA** replied on **2025-07-23** at **02:37:51**
> 
> > all you have to do is run the same parameters from the recipes I provided earlier but change these two parameters only:
> 
> But it seems that the model name is not taken from the comments of the recipe.  Its rather set in the tensor_downloader.sh:
> 
> ```
>  41 # -----------------------------------------------------------------------------
>  42 # Default configuration (used if not overridden by download.conf)
>  43 MODEL_NAME="DeepSeek-R1-0528" # Name of the LLM model
> ```
> 
> So I have to use download.conf, right?
> 
> [EDIT]:  and the path to the generated download.conf is set via the:
> 
> ```
> CONFIG_FILE="$SCRIPT_DIR/download.conf"
> ```
> 
> ~~So I should copy the download.conf from the destination directory (the current directory) to the directory where the tensor_downloader.sh is placed.~~
> So I should copy the GGUF-Tool-Suite/models/DeepSeek-TNG-R1T2-Chimera/download.conf to the directory where the quant_downloader.sh is placed.  Right?

> 👤 **Thireus** replied on **2025-07-23** at **05:40:28**
> 
> > So I should copy the GGUF-Tool-Suite/models/DeepSeek-TNG-R1T2-Chimera/download.conf to the directory where the quant_downloader.sh is placed. Right?
> 
> That's right. The same must be done for the ppl_results.csv file. You will notice that the ppl_results.csv and download.conf at the root of the repo are currently symlinks to the R1-0528 - https://github.com/Thireus/GGUF-Tool-Suite/ they must be changed to point to R1T2 (or be replaced).

> 👤 **magikRUKKOLA** replied on **2025-07-24** at **15:32:17**
> 
> @Thireus 
> 
> Final estimate: PPL = 3.2381 +/- 0.01727
> 
> recipe:
> ```
> ## Quant mix recipe created using Thireus' GGUF Tool Suite - https://gguf.thireus.com/
> # Model name: DeepSeek-TNG-R1T2-Chimera
> # Link to the original model: https://huggingface.co/tngtech/DeepSeek-TNG-R1T2-Chimera
> 
> ## Model head & embeddings — qbits: 32 8
> output_norm\.weight=f32
> token_embd\.weight=q8_0
> output\.weight=q8_0
> 
> ## Special attention kernels — single-quant only (llama-quantize takes care of it) — qbits: 8
> blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0
> 
> ## Multi-headed attention parameters — qbits: 32 4
> blk\.([0-9]|[1-5][0-9]|60)\.attn_v_b\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_a_mqa\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_output\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_kv_b\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_norm\.weight=f32
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_a\.weight=iq4_xs
> blk\.([0-9]|[1-5][0-9]|60)\.attn_q_b\.weight=iq4_xs
> 
> ## Core FFN weights — qbits: 32 8 6 5
> blk\.2\.ffn_gate\.weight=q8_0
> blk\.(0|2)\.ffn_up\.weight=iq6_k
> blk\.([0-9]|[1-5][0-9]|60)\.ffn_norm\.weight=f32
> blk\.[0-1]\.ffn_gate\.weight=iq6_k
> blk\.1\.ffn_down\.weight=iq6_k
> blk\.2\.ffn_down\.weight=iq5_k_r4
> blk\.1\.ffn_up\.weight=iq5_k_r4
> blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_inp\.weight=f32
> blk\.0\.ffn_down\.weight=q8_0
> 
> ## Other tensors — qbits: 32
> blk\.([3-9]|[1-5][0-9]|60)\.exp_probs_b\.bias=f32
> 
> ## GPU-loaded ffn_*_shexp
> # ffn_down_shexp (down-projection) — qbits: 8 6 5
> blk\.(11|17|19|29|36|39|44|60|2[6-7]|2[0-4]|3[0-1]|3[3-4])\.ffn_down_shexp\.weight=q8_0
> blk\.([3-8]|10|12|25|28|32|35|3[7-8]|1[4-6]|4[5-9]|4[0-3]|5[0-8])\.ffn_down_shexp\.weight=iq6_k
> blk\.(9|13|18|59)\.ffn_down_shexp\.weight=iq5_k_r4
> 
> # ffn_up_shexp (up-projection) — qbits: 8 6 5
> blk\.(6|15|18|30|37|39|41|50|54|60|2[1-4]|3[2-4]|2[6-9])\.ffn_up_shexp\.weight=q8_0
> blk\.([3-5]|[8-9]|19|20|25|31|38|40|58|4[2-9]|1[6-7]|1[0-4]|3[5-6]|5[5-6]|5[1-3])\.ffn_up_shexp\.weight=iq6_k
> blk\.(7|57|59)\.ffn_up_shexp\.weight=iq5_k_r4
> 
> # ffn_gate_shexp (gate-projection) — qbits: 8 6 5
> blk\.(16|20|29|54|60|5[6-8]|5[0-2]|4[1-2]|4[4-9]|1[8-9]|2[3-6]|3[3-4])\.ffn_gate_shexp\.weight=q8_0
> blk\.([3-5]|[7-9]|17|21|40|43|53|55|3[0-2]|2[7-8]|3[5-9]|1[1-5])\.ffn_gate_shexp\.weight=iq6_k
> blk\.(6|10|22|59)\.ffn_gate_shexp\.weight=iq5_k_r4
> 
> ## CPU-loaded ffn_*_exps
> # ffn_down_exps (down-extraction) — qbits: 8 5 3
> blk\.(51|53|3[2-9]|4[0-9])\.ffn_down_exps\.weight=q8_0
> blk\.([3-9]|50|52|60|5[4-9]|1[0-4]|2[0-9]|3[0-1]|1[6-9])\.ffn_down_exps\.weight=iq5_k_r4
> blk\.15\.ffn_down_exps\.weight=iq3_k
> 
> # ffn_up_exps (up-extraction) — qbits: 8 5 4
> blk\.(35|53|55|4[7-8]|5[0-1]|4[3-4])\.ffn_up_exps\.weight=q8_0
> blk\.([3-9]|49|52|54|60|4[0-2]|1[1-9]|3[0-4]|2[0-9]|4[5-6]|3[6-9]|5[6-9])\.ffn_up_exps\.weight=iq5_k_r4
> blk\.10\.ffn_up_exps\.weight=iq4_ks
> 
> # ffn_gate_exps (gate-extraction) — qbits: 8 5 4
> blk\.(35|39|41|60|5[0-5]|4[3-9])\.ffn_gate_exps\.weight=q8_0
> blk\.([3-7]|9|[1-2][0-9]|40|42|3[6-8]|3[0-4]|5[6-9])\.ffn_gate_exps\.weight=iq5_k_r4
> blk\.8\.ffn_gate_exps\.weight=iq4_ks
> 
> ## Summary of tensor sizes per class
> # GPU Total: 11.744 GiB (95.1%) | 12.34 GiB max, if all were q8_0 | 10.39 GiB min, if all were iq5_k_r4
> # CPU Total: 477.066 GiB (73.7%) | 647.06 GiB max, if all were q8_0 | 261.68 GiB min, if all were iq3_k
> # GPU+CPU Total: 488.811 GiB (84.4%)
> 
> ## Summary of tensor counts and bpw per qtype
> #
> # GPU-loaded quants:
> # QTYPE         Count   BPW     Assigned GiB    % Assigned      Max GiB (all)
> # +f32          361     32.0      0.40 GiB      -               -
> # +q8_0         61      8.5       0.51 GiB      -               -
> # q8_0          71      8.5       3.07 GiB      55.4%           5.54
> # iq6_k         101     6.625     1.60 GiB      37.0%           4.32
> # iq5_k_r4      13      5.5       0.27 GiB      7.6%            3.58
> # +iq4_xs       366     4.25      5.90 GiB      -               -
> #
> # CPU-loaded quants:
> # QTYPE         Count   BPW     Assigned GiB    % Assigned      Max GiB (all)
> # q8_0          46      8.5     171.06 GiB      26.4%           647.06
> # iq5_k_r4      125     5.5     300.78 GiB      71.8%           418.69
> # iq4_ks        2       4.25      3.72 GiB      1.1%            323.53
> # iq3_k         1       3.4375    1.50 GiB      0.6%            261.68
> #
> # -Average BPW: 6.2478
> #
> # -Notes:
> # - '+' means user-defined pre-assigned tensors and f32 tensors
> # - Recipe produced on the 2025-07-16 19:21:22 UTC+0000 using Thireus' GGUF tools (https://gguf.thireus.com/)
> # - Command used:
> # quant_assign.py ppl_results.csv --tolerance 0.01 --cpu-irq-k 1.5 --gpu-irq-k 1.5 --gpu-assign-qtype iq4_xs \
> # --cpu-tensors-max-size 500 --gpu-tensors-max-size 95% --exponential-factor 8 --cpu-tensors \
> # 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_down_exps\.weight' 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_up_exps\.weight' \
> # 'blk\.([3-9]|[1-5][0-9]|60)\.ffn_gate_exps\.weight' --gpu-tensors '.*' --cpu-quants iq4_ks iq3_k iq5_k_r4 q8_0 \
> # --gpu-quants q8_0 iq5_k_r4 iq6_k --gpu-assign-tensors 'blk\.([0-9]|[1-5][0-9]|60)\.attn_k_b\.weight=q8_0'
> ```
> 
> symbolic links:
> ```
> ls -lah /opt/GGUF-Tool-Suite/GGUF-Tool-Suite/ | grep -- '->'
> lrwxrwxrwx  1 root root   46 Jul 23 14:51 download.conf -> models/DeepSeek-TNG-R1T2-Chimera/download.conf
> lrwxrwxrwx  1 root root   48 Jul 23 14:51 ppl_results.csv -> models/DeepSeek-TNG-R1T2-Chimera/ppl_results.csv
> ```
> 
> ppl command:
> ```
> export MALLOC_CONF="background_thread:true,percpu_arena:phycpu,metadata_thp:auto,dirty_decay_ms:10000,muzzy_decay_ms:60000"
> export LD_PRELOAD=/usr/local/lib/libjemalloc.so
> 
> ulimit -n 9999
> CUDA_VISIBLE_DEVICES="0,1" \
> /opt/ik_llama.cpp/ik_llama.cpp/build/bin/llama-perplexity \
>     -f /opt/ik_llama.cpp/wiki.test.raw \
>     --model /opt/GGUF-Tool-Suite/GGUF-Tool-Suite/DeepSeek-TNG-R1T2-Chimera.ROOT-6.2478bpw/DeepSeek-TNG-R1T2-Chimera-THIREUS-BF16-SPECIAL_TENSOR-00001-of-01148.gguf \
>     --alias THIREUS/DeepSeek-R1-0528-6.2478bpw \
>     --ctx-size $((512)) \
>     -ub $((512)) \
>     --mlock \
>     --seed 3407 \
>     --temp 0.5 --top-k 0 --top-p 1.0 --min-p 0.1 --repeat-penalty 1.0 \
>     -ctk q8_0 \
>     -mla 3 -fa \
>     -amb 256 \
>     --override-tensor exps=CPU \
>     --n-gpu-layers 99 \
>     --threads $(grep ^cpu\\scores /proc/cpuinfo | uniq | awk '{print $4}' | xargs -I{} echo "{}-0" | bc) \
>     --host 0.0.0.0 \
>     --port 8080 \
>     --lookup-cache-dynamic /mnt/data/ik_llama.kv.dump
> ```
> 
> output:
> ```
> 
> llm_load_tensors:      CUDA1 buffer size =  7309.40 MiB
> ....................................................................................................
> llama_new_context_with_model: n_ctx      = 2048
> llama_new_context_with_model: n_batch    = 2048
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 3
> llama_new_context_with_model: attn_max_b = 256
> llama_new_context_with_model: fused_moe  = 0
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:      CUDA0 KV buffer size =    37.07 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =    35.87 MiB
> llama_new_context_with_model: KV self size  =   72.91 MiB, c^KV (q8_0):   72.91 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     1.97 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size =   503.00 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size =   477.50 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =   226.01 MiB
> llama_new_context_with_model: graph nodes  = 3664
> llama_new_context_with_model: graph splits = 119
> 
> system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
> perplexity: tokenizing the input ..
> perplexity: tokenization took 1063.51 ms
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
> perplexity: 34.42 seconds per pass - ETA 1 hours 20.43 minutes
> [1]2.4351,[2]3.1897,[3]2.3205,[4]1.9458,[5]1.7609,[6]1.6196,[7]1.5299,[8]1.4672,[9]1.4199,[10]1.3818,[11]1.3660,[12]1.3907,[13]1.4017,[14]1.5253,[15]1.6514,[16]1.7064,[17]1.8628,[18]1.9873,[19]1.9522,[20]1.9395,[21]2.0402,[22]2.0125,[23]1.9875,[24]2.0011,[25]1.9732,[26]1.9527,[27]1.9981,[28]2.0069,[29]2.0518,[30]2.0816,[31]2.1125,[32]2.1295,[33]2.1668,[34]2.2087,[35]2.2552,[36]2.3050,[37]2.3396,[38]2.3842,[39]2.4249,[40]2.4829,
> [41]2.5208,[42]2.5342,[43]2.5822,[44]2.5977,[45]2.6753,[46]2.7253,[47]2.6833,[48]2.6399,[49]2.6157,[50]2.6325,[51]2.6774,[52]2.6906,[53]2.7387,[54]2.7510,[55]2.7807,[56]2.8087,[57]2.8209,[58]2.8541,[59]2.8657,[60]2.9098,[61]2.9479,[62]2.9947,[63]3.0254,[64]3.0669,[65]3.0752,[66]3.0599,[67]3.0367,[68]3.0642,[69]3.0591,[70]3.0714,[71]3.0888,[72]3.1040,[73]3.1180,[74]3.1399,[75]3.1196,[76]3.0758,[77]3.0346,[78]3.0299,[79]3.0087,[80]2.9913,[81]2.9569,[82]2.9600,[83]2.9304,[84]2.8972,[85]2.8645,[86]2.8421,[87]2.8368,[88]2.8102,[89]2.7942,[90]2.7699,[91]2.7421,[92]2.7182,[93]2.6928,[94]2.6682,[95]2.6466,[96]2.6447,[97]2.6512,[98]2.6366,[99]2.6195,[100]2.6214,[101]2.6132,[102]2.6284,[103]2.6526,[104]2.6704,[105]2.6673,[106]2.6886,[107]2.7128,[108]2.7322,[109]2.7648,[110]2.7977,[111]2.8168,[112]2.7924,[113]2.7797,[114]2.7587,[115]2.7442,[116]2.7299,[117]2.7081,[118]2.6884,[119]2.6681,[120]2.6501,[121]2.6349,[122]2.6184,[123]2.6017,[124]2.5834,[125]2.5665,[126]2.5504,[127]2.5366,[128]2.5267,[129]2.5157,[130]2.5029,[131]2.4946,[132]2.5008,[133]2.5102,[134]2.5164,[135]2.5266,[136]2.5417,[137]2.5549,[138]2.5630,[139]2.5736,[140]2.5748,[141]2.5769,[142]2.5760,[143]2.5770,[144]2.5745,[145]2.5666,[146]2.5651,[147]2.5695,[148]2.5698,[149]2.5711,[150]2.5661,[151]2.5645,[152]2.5621,[153]2.5584,[154]2.5589,[155]2.5630,[156]2.5649,[157]2.5707,[158]2.5792,[159]2.5814,[160]2.5899,[161]2.5981,[162]2.6080,[163]2.6119,[164]2.6312,[165]2.6536,[166]2.6702,[167]2.6816,[168]2.7050,[169]2.7269,[170]2.7477,[171]2.7695,[172]2.7545,[173]2.7390,[174]2.7265,[175]2.7139,[176]2.7024,[177]2.6910,[178]2.6792,[179]2.6665,[180]2.6700,[181]2.6838,[182]2.6983,[183]2.7121,[184]2.7255,[185]2.7354,[186]2.7514,[187]2.7666,[188]2.7800,[189]2.7904,[190]2.7911,[191]2.7983,[192]2.8016,[193]2.8067,[194]2.8256,[195]2.8340,[196]2.8471,[197]2.8569,[198]2.8614,[199]2.8669,[200]2.8662,[201]2.8807,[202]2.8758,[203]2.8812,[204]2.8846,[205]2.8843,[206]2.8869,[207]2.8949,[208]2.9037,[209]2.9127,[210]2.9132,[211]2.9087,[212]2.9092,[213]2.9166,[214]2.9183,[215]2.9235,[216]2.9241,[217]2.9196,[218]2.9197,[219]2.9206,[220]2.9205,[221]2.9211,[222]2.9212,[223]2.9219,[224]2.9263,[225]2.9281,[226]2.9206,[227]2.9179,[228]2.9199,[229]2.9237,[230]2.9295,[231]2.9355,[232]2.9278,[233]2.9203,[234]2.9207,[235]2.9187,[236]2.9271,[237]2.9351,[238]2.9441,[239]2.9537,[240]2.9626,[241]2.9735,[242]2.9867,[243]2.9985,[244]3.0064,[245]3.0171,[246]3.0272,[247]3.0260,[248]3.0220,[249]3.0199,[250]3.0140,[251]3.0118,[252]3.0143,[253]3.0179,[254]3.0247,[255]3.0308,[256]3.0345,[257]3.0371,[258]3.0383,[259]3.0416,[260]3.0439,[261]3.0452,[262]3.0447,[263]3.0500,[264]3.0524,[265]3.0529,[266]3.0546,[267]3.0570,[268]3.0607,[269]3.0637,[270]3.0631,[271]3.0616,[272]3.0553,[273]3.0548,[274]3.0483,[275]3.0381,[276]3.0274,[277]3.0293,[278]3.0392,[279]3.0452,[280]3.0525,[281]3.0599,[282]3.0659,[283]3.0719,[284]3.0779,[285]3.0910,[286]3.0933,[287]3.0966,[288]3.1015,[289]3.1039,[290]3.0964,[291]3.0878,[292]3.0855,[293]3.0845,[294]3.0816,[295]3.0793,[296]3.0811,[297]3.0818,[298]3.0870,[299]3.0928,[300]3.0955,[301]3.0993,[302]3.1011,[303]3.1029,[304]3.1025,[305]3.1138,[306]3.1210,[307]3.1314,[308]3.1209,[309]3.1157,[310]3.1066,[311]3.1095,[312]3.1112,[313]3.1166,[314]3.1189,[315]3.1220,[316]3.1234,[317]3.1252,[318]3.1258,[319]3.1262,[320]3.1302,[321]3.1304,[322]3.1321,[323]3.1383,[324]3.1391,[325]3.1443,[326]3.1487,[327]3.1527,[328]3.1553,[329]3.1570,[330]3.1634,[331]3.1666,[332]3.1710,[333]3.1699,[334]3.1701,[335]3.1708,[336]3.1709,[337]3.1719,[338]3.1720,[339]3.1744,[340]3.1780,[341]3.1831,[342]3.1915,[343]3.2004,[344]3.2053,[345]3.1970,[346]3.1897,[347]3.1843,[348]3.1771,[349]3.1734,[350]3.1722,[351]3.1767,[352]3.1909,[353]3.1997,[354]3.2119,[355]3.2202,[356]3.2256,[357]3.2369,[358]3.2463,[359]3.2494,[360]3.2554,[361]3.2645,[362]3.2727,[363]3.2781,[364]3.2844,[365]3.2902,[366]3.3000,[367]3.3082,[368]3.3149,[369]3.3223,[370]3.3302,[371]3.3431,[372]3.3513,[373]3.3545,[374]3.3577,[375]3.3624,[376]3.3746,[377]3.3853,[378]3.3881,[379]3.3880,[380]3.3847,[381]3.3890,[382]3.3945,[383]3.3979,[384]3.4021,[385]3.4060,[386]3.4118,[387]3.4175,[388]3.4205,[389]3.4108,[390]3.4021,[391]3.3920,[392]3.3868,[393]3.3776,[394]3.3693,[395]3.3607,[396]3.3512,[397]3.3428,[398]3.3338,[399]3.3240,[400]3.3156,[401]3.3062,[402]3.2965,[403]3.2885,[404]3.2789,[405]3.2701,[406]3.2608,[407]3.2520,[408]3.2435,[409]3.2354,[410]3.2297,[411]3.2303,[412]3.2256,[413]3.2272,[414]3.2290,[415]3.2260,[416]3.2259,[417]3.2281,[418]3.2224,[419]3.2237,[420]3.2213,[421]3.2203,[422]3.2210,[423]3.2203,[424]3.2242,[425]3.2239,[426]3.2242,[427]3.2233,[428]3.2258,[429]3.2270,[430]3.2294,[431]3.2303,[432]3.2294,[433]3.2259,[434]3.2258,[435]3.2184,[436]3.2123,[437]3.2085,[438]3.2070,[439]3.2038,[440]3.2088,[441]3.2141,[442]3.2214,[443]3.2194,[444]3.2200,[445]3.2210,[446]3.2253,[447]3.2283,[448]3.2305,[449]3.2334,[450]3.2371,[451]3.2401,[452]3.2420,[453]3.2434,[454]3.2423,[455]3.2444,[456]3.2449,[457]3.2475,[458]3.2525,[459]3.2530,[460]3.2531,[461]3.2502,[462]3.2537,[463]3.2606,[464]3.2654,[465]3.2587,[466]3.2568,[467]3.2550,[468]3.2563,[469]3.2537,[470]3.2510,[471]3.2514,[472]3.2519,[473]3.2511,[474]3.2502,[475]3.2512,[476]3.2500,[477]3.2493,[478]3.2500,[479]3.2515,[480]3.2541,[481]3.2503,[482]3.2537,[483]3.2531,[484]3.2566,[485]3.2626,[486]3.2657,[487]3.2691,[488]3.2743,[489]3.2767,[490]3.2813,[491]3.2871,[492]3.2913,[493]3.2911,[494]3.2922,[495]3.2945,[496]3.2962,[497]3.2991,[498]3.2997,[499]3.2992,[500]3.3031,[501]3.3076,[502]3.3064,[503]3.3050,[504]3.3069,[505]3.3102,[506]3.3182,[507]3.3211,[508]3.3245,[509]3.3175,[510]3.3122,[511]3.3059,[512]3.3016,[513]3.2956,[514]3.2941,[515]3.2957,[516]3.2909,[517]3.2907,[518]3.2897,[519]3.2899,[520]3.2938,[521]3.2926,[522]3.2912,[523]3.2966,[524]3.2954,[525]3.2937,[526]3.2892,[527]3.2845,[528]3.2811,[529]3.2784,[530]3.2757,[531]3.2728,[532]3.2675,[533]3.2618,[534]3.2573,[535]3.2580,[536]3.2605,[537]3.2634,[538]3.2654,[539]3.2679,[540]3.2730,[541]3.2761,[542]3.2782,[543]3.2729,[544]3.2688,[545]3.2686,[546]3.2625,[547]3.2563,[548]3.2503,[549]3.2440,[550]3.2382,[551]3.2325,[552]3.2270,[553]3.2214,[554]3.2197,[555]3.2181,[556]3.2209,[557]3.2247,[558]3.2304,[559]3.2346,[560]3.2398,[561]3.2381,
> Final estimate: PPL = 3.2381 +/- 0.01727
> 
> llama_print_timings:        load time =  265144.07 ms
> llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings: prompt eval time = 4844799.75 ms / 287232 tokens (   16.87 ms per token,    59.29 tokens per second)
> llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings:       total time = 4854801.43 ms / 287233 tokens
> ```

> 👤 **Thireus** replied on **2025-07-24** at **16:06:47**
> 
> Great! Do we have a ppl graph/metrics somewhere for R1T2-Chimera to compare?

> 👤 **magikRUKKOLA** replied on **2025-07-24** at **21:47:02**
> 
> > Great! Do we have a ppl graph/metrics somewhere for R1T2-Chimera to compare?
> 
> Nope.  That's the first quant I have tested.  And its likely that this is the quant I am going to use.
> That said, I can test some other quants and build the graphs.  But which ones?  Should I create perhaps a repo for the graphs?  Or I should just keep it in the comments of the discussions of the ik_llama.cpp?

> 👤 **magikRUKKOLA** replied on **2025-07-24** at **23:30:32**
> 
> @Thireus 
> > Great! Do we have a ppl graph/metrics somewhere for R1T2-Chimera to compare?
> 
> I propose that you would cook some recipes, I will download it and run it on my hardware to check the PPL.  I have about 4 machines with 512GB RAM (and building the fifth with faster RAM) so I can check it more-or-less quickly.  Then I can upload the JSON and the SVG over here (into the comments?).

> 👤 **saood06** replied on **2025-07-25** at **00:09:33**
> 
> >Should I create perhaps a repo for the graphs? Or I should just keep it in the comments of the discussions of the ik_llama.cpp?
> 
> Why not make a new Discussion? It feels a bit out of place having Kimi and other models PPL graphs here in the "DeepSeek-R1-0528 ik quants" discussion. (You could also make a repo if that is what you want).

> 👤 **magikRUKKOLA** replied on **2025-07-25** at **00:14:17**
> 
> @saood06 
> 
> > Why not make a new Discussion?
> 
> Yeah, why not.  Let me collect some data on Chimera quants first.
> 
> >  (You could also make a repo if that is what you want).
> 
> Not sure about that.  Unfortunately I can be too lazy for that -- first, I need to make git to use TOR proxy.  Not sure if it supports that from the box.

> 👤 **Thireus** replied on **2025-07-25** at **00:19:55**
> 
> That's really nice of you @magikRUKKOLA. I'd really like to know how these four recipes compare to @ubergarm's quants: https://huggingface.co/ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF
> 
> I wish unsloth could post their ppl... so I could also directly compare to them.
> 
> I've created three recipes that are on par (or slightly lower) with the bpw @ubergarm posted on his README:
> 
> ```
> * IQ3_KS 281.463 GiB (3.598 BPW)
> Final estimate: PPL = 3.3167 +/- 0.01789
> 
> * IQ2_KS 203.553 GiB (2.602 BPW)
> Final estimate: PPL = 3.6254 +/- 0.02001
> 
> * IQ2_KT 171.146 GiB (2.188 BPW)
> Final estimate: PPL = 3.8887 +/- 0.02191
> 
> * IQ2_XXS 169.590 GiB (2.168 BPW)
> Final estimate: PPL = 4.0078 +/- 0.02291
> 
> * IQ1_S 132.915 GiB (1.699 BPW)
> Final estimate: PPL = 4.9878 +/- 0.02999
> ```
> 
> This time I do not have any idea how they may perform. I went a bit wild on the quant spread.
> 
> [DeepSeek-TNG-R1T2-Chimera.20250725-recipes.zip](https://github.com/user-attachments/files/21420758/DeepSeek-TNG-R1T2-Chimera.20250725-recipes.zip)

> 👤 **saood06** replied on **2025-07-25** at **00:31:24**
> 
> >Yeah, why not. Let me collect some data on Chimera quants first.
> 
> Thanks. Your graphs really are nice, and it will be nice when they are more convenient to find.

> 👤 **Thireus** replied on **2025-07-26** at **17:52:56**
> 
> ~~Quick Note: Looks like Google Colab doesn't compute good mixes - so I don't know if the recipes I provided are good @magikRUKKOLA (I've generated them from Google Colab). I'm investigating the issue here: https://github.com/Thireus/GGUF-Tool-Suite/issues/15~~
> 
> Edit: issue identified, it's not affecting the DeepSeek quant mixes, only Kimi-K2.

> 👤 **magikRUKKOLA** replied on **2025-07-26** at **18:38:45**
> 
> @Thireus 
> 
> I am in the process of testing those.
> This is the first one:
> 
> DeepSeek-TNG-R1T2-Chimera-1.6693bpw-ppl.log
> ```
> 
> ....................................................................................................
> llama_new_context_with_model: n_ctx      = 2048
> llama_new_context_with_model: n_batch    = 2048
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 3
> llama_new_context_with_model: attn_max_b = 256
> llama_new_context_with_model: fused_moe  = 1
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:      CUDA0 KV buffer size =    25.11 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =    25.11 MiB
> llama_kv_cache_init:      CUDA2 KV buffer size =    22.72 MiB
> llama_new_context_with_model: KV self size  =   72.91 MiB, c^KV (q8_0):   72.91 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     1.97 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size =   503.00 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size =   477.50 MiB
> llama_new_context_with_model:      CUDA2 compute buffer size =   477.50 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =   226.01 MiB
> llama_new_context_with_model: graph nodes  = 3574
> llama_new_context_with_model: graph splits = 120
> 
> system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
> perplexity: tokenizing the input ..
> perplexity: tokenization took 1068.62 ms
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
> perplexity: 37.57 seconds per pass - ETA 1 hours 27.82 minutes
> [1]3.3547,[2]4.2632,[3]3.2154,[4]3.0465,[5]2.9922,[6]2.9360,[7]2.8575,[8]2.9308,[9]2.9686,[10]2.9051,[11]2.9850,[12]3.1540,[13]3.2077,[14]3.3146,[15]3.4845,[16]3.4521,[17]3.6308,[18]3.7577,[19]3.7164,[20]3.6877,[21]3.7809,[22]3.7042,[23]3.6014,[24]3.6152,[25]3.5232,[26]3.4597,[27]3.5111,[28]3.5033,[29]3.5621,[30]3.5715,[31]3.5956,[32]3.6042,[33]3.6625,[34]3.7224,[35]3.7911,[36]3.8496,[37]3.8650,[38]3.9139,[39]3.9491,[40]4.0080,[41]4.0543,[42]4.0490,[43]4.0879,[44]4.0773,[45]4.1642,[46]4.2138,[47]4.1957,[48]4.1911,[49]4.1825,[50]4.2029,[51]4.2522,[52]4.2617,[53]4.3398,[54]4.3588,[55]4.3859,[56]4.4228,[57]4.4413,[58]4.4750,[59]4.4766,[60]4.5208,[61]4.5652,[62]4.6172,[63]4.6575,[64]4.7058,[65]4.7163,[66]4.7177,[67]4.7026,[68]4.7332,[69]4.7611,[70]4.7896,[71]4.7968,[72]4.8004,[73]4.8086,[74]4.8325,[75]4.8144,[76]4.7689,[77]4.7276,[78]4.7265,[79]4.7301,[80]4.7400,[81]4.7095,[82]4.7298,[83]4.7264,[84]4.7074,[85]4.6881,[86]4.6737,[87]4.7103,[88]4.7008,[89]4.7041,[90]4.6953,[91]4.6846,[92]4.6699,[93]4.6573,[94]4.6558,[95]4.6478,[96]4.6641,[97]4.6792,[98]4.6746,[99]4.6619,[100]4.6568,[101]4.6471,[102]4.6632,[103]4.6963,[104]4.7288,[105]4.7241,[106]4.7569,[107]4.7775,[108]4.7943,[109]4.8345,[110]4.8705,[111]4.8952,[112]4.8630,[113]4.8579,[114]4.8448,[115]4.8238,[116]4.8302,[117]4.8144,[118]4.7993,[119]4.7741,[120]4.7512,[121]4.7390,[122]4.7074,[123]4.6896,[124]4.6716,[125]4.6514,[126]4.6241,[127]4.6210,[128]4.6145,[129]4.6143,[130]4.6160,[131]4.6121,[132]4.6119,[133]4.6128,[134]4.6247,[135]4.6362,[136]4.6548,[137]4.6688,[138]4.6678,[139]4.6771,[140]4.6616,[141]4.6502,[142]4.6342,[143]4.6218,[144]4.6013,[145]4.5789,[146]4.5646,[147]4.5594,[148]4.5464,[149]4.5381,[150]4.5173,[151]4.5045,[152]4.4921,[153]4.4737,[154]4.4621,[155]4.4607,[156]4.4502,[157]4.4491,[158]4.4519,[159]4.4498,[160]4.4565,[161]4.4622,[162]4.4702,[163]4.4806,[164]4.5014,[165]4.5311,[166]4.5506,[167]4.5672,[168]4.5952,[169]4.6212,[170]4.6539,[171]4.6777,[172]4.6564,[173]4.6331,[174]4.6326,[175]4.6215,[176]4.6177,[177]4.6124,[178]4.6049,[179]4.5982,[180]4.5995,[181]4.6129,[182]4.6304,[183]4.6431,[184]4.6550,[185]4.6614,[186]4.6774,[187]4.6916,[188]4.7078,[189]4.7163,[190]4.7109,[191]4.7139,[192]4.7099,[193]4.7110,[194]4.7320,[195]4.7450,[196]4.7574,[197]4.7652,[198]4.7632,[199]4.7637,[200]4.7509,[201]4.7661,[202]4.7523,[203]4.7542,[204]4.7538,[205]4.7537,[206]4.7534,[207]4.7595,[208]4.7709,[209]4.7788,[210]4.7725,[211]4.7595,[212]4.7570,[213]4.7626,[214]4.7584,[215]4.7634,[216]4.7579,[217]4.7480,[218]4.7451,[219]4.7436,[220]4.7351,[221]4.7330,[222]4.7259,[223]4.7234,[224]4.7249,[225]4.7219,[226]4.7092,[227]4.7051,[228]4.7018,[229]4.7033,[230]4.7065,[231]4.7096,[232]4.6995,[233]4.6978,[234]4.7009,[235]4.7050,[236]4.7123,[237]4.7191,[238]4.7277,[239]4.7402,[240]4.7486,[241]4.7590,[242]4.7760,[243]4.7887,[244]4.7989,[245]4.8138,[246]4.8272,[247]4.8211,[248]4.8121,[249]4.8028,[250]4.7891,[251]4.7802,[252]4.7780,[253]4.7775,[254]4.7835,[255]4.7866,[256]4.7866,[257]4.7844,[258]4.7814,[259]4.7806,[260]4.7808,[261]4.7783,[262]4.7715,[263]4.7760,[264]4.7784,[265]4.7723,[266]4.7713,[267]4.7699,[268]4.7728,[269]4.7758,[270]4.7710,[271]4.7661,[272]4.7524,[273]4.7571,[274]4.7524,[275]4.7421,[276]4.7355,[277]4.7340,[278]4.7434,[279]4.7478,[280]4.7555,[281]4.7616,[282]4.7662,[283]4.7736,[284]4.7790,[285]4.7950,[286]4.7949,[287]4.7941,[288]4.7969,[289]4.7953,[290]4.7882,[291]4.7833,[292]4.7904,[293]4.7952,[294]4.7954,[295]4.7949,[296]4.7977,[297]4.7969,[298]4.8009,[299]4.8082,[300]4.8083,[301]4.8095,[302]4.8108,[303]4.8078,[304]4.8028,[305]4.8151,[306]4.8210,[307]4.8322,[308]4.8212,[309]4.8199,[310]4.8123,[311]4.8180,[312]4.8216,[313]4.8285,[314]4.8297,[315]4.8305,[316]4.8279,[317]4.8268,[318]4.8234,[319]4.8198,[320]4.8225,[321]4.8198,[322]4.8175,[323]4.8252,[324]4.8237,[325]4.8274,[326]4.8300,[327]4.8309,[328]4.8295,[329]4.8287,[330]4.8335,[331]4.8378,[332]4.8408,[333]4.8369,[334]4.8338,[335]4.8314,[336]4.8277,[337]4.8268,[338]4.8251,[339]4.8249,[340]4.8269,[341]4.8316,[342]4.8403,[343]4.8510,[344]4.8551,[345]4.8522,[346]4.8495,[347]4.8514,[348]4.8449,[349]4.8436,[350]4.8408,[351]4.8466,[352]4.8643,[353]4.8751,[354]4.8898,[355]4.9000,[356]4.9082,[357]4.9228,[358]4.9361,[359]4.9385,[360]4.9435,[361]4.9537,[362]4.9627,[363]4.9670,[364]4.9732,[365]4.9785,[366]4.9896,[367]4.9978,[368]5.0052,[369]5.0117,[370]5.0197,[371]5.0355,[372]5.0447,[373]5.0463,[374]5.0477,[375]5.0514,[376]5.0655,[377]5.0783,[378]5.0792,[379]5.0758,[380]5.0701,[381]5.0751,[382]5.0803,[383]5.0817,[384]5.0855,[385]5.0895,[386]5.0952,[387]5.1003,[388]5.1019,[389]5.0897,[390]5.0815,[391]5.0705,[392]5.0647,[393]5.0617,[394]5.0529,[395]5.0484,[396]5.0469,[397]5.0387,[398]5.0288,[399]5.0191,[400]5.0121,[401]5.0006,[402]4.9916,[403]4.9824,[404]4.9697,[405]4.9592,[406]4.9504,[407]4.9401,[408]4.9343,[409]4.9267,[410]4.9215,[411]4.9248,[412]4.9269,[413]4.9330,[414]4.9390,[415]4.9367,[416]4.9366,[417]4.9405,[418]4.9344,[419]4.9369,[420]4.9316,[421]4.9302,[422]4.9324,[423]4.9312,[424]4.9357,[425]4.9341,[426]4.9342,[427]4.9326,[428]4.9368,[429]4.9359,[430]4.9402,[431]4.9407,[432]4.9384,[433]4.9325,[434]4.9344,[435]4.9296,[436]4.9242,[437]4.9200,[438]4.9170,[439]4.9182,[440]4.9234,[441]4.9298,[442]4.9386,[443]4.9348,[444]4.9333,[445]4.9334,[446]4.9387,[447]4.9415,[448]4.9435,[449]4.9449,[450]4.9484,[451]4.9511,[452]4.9538,[453]4.9553,[454]4.9516,[455]4.9530,[456]4.9510,[457]4.9530,[458]4.9577,[459]4.9582,[460]4.9559,[461]4.9513,[462]4.9532,[463]4.9609,[464]4.9658,[465]4.9612,[466]4.9624,[467]4.9636,[468]4.9655,[469]4.9626,[470]4.9594,[471]4.9588,[472]4.9599,[473]4.9579,[474]4.9550,[475]4.9544,[476]4.9534,[477]4.9513,[478]4.9517,[479]4.9543,[480]4.9566,[481]4.9522,[482]4.9560,[483]4.9530,[484]4.9556,[485]4.9622,[486]4.9645,[487]4.9668,[488]4.9721,[489]4.9725,[490]4.9767,[491]4.9834,[492]4.9875,[493]4.9858,[494]4.9850,[495]4.9859,[496]4.9869,[497]4.9888,[498]4.9881,[499]4.9855,[500]4.9888,[501]4.9919,[502]4.9906,[503]4.9880,[504]4.9903,[505]4.9921,[506]5.0018,[507]5.0035,[508]5.0061,[509]4.9980,[510]4.9971,[511]4.9941,[512]4.9911,[513]4.9866,[514]4.9880,[515]4.9907,[516]4.9859,[517]4.9875,[518]4.9874,[519]4.9886,[520]4.9940,[521]4.9904,[522]4.9871,[523]4.9937,[524]4.9921,[525]4.9918,[526]4.9896,[527]4.9819,[528]4.9810,[529]4.9755,[530]4.9714,[531]4.9656,[532]4.9543,[533]4.9490,[534]4.9477,[535]4.9493,[536]4.9525,[537]4.9565,[538]4.9604,[539]4.9648,[540]4.9712,[541]4.9776,[542]4.9813,[543]4.9810,[544]4.9770,[545]4.9779,[546]4.9708,[547]4.9684,[548]4.9630,[549]4.9593,[550]4.9564,[551]4.9538,[552]4.9505,[553]4.9481,[554]4.9520,[555]4.9486,[556]4.9506,[557]4.9544,[558]4.9609,[559]4.9652,[560]4.9711,[561]4.9676,
> Final estimate: PPL = 4.9676 +/- 0.03020
> 
> llama_print_timings:        load time =  142436.92 ms
> llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings: prompt eval time = 5254620.03 ms / 287232 tokens (   18.29 ms per token,    54.66 tokens per second)
> llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings:       total time = 5264058.14 ms / 287233 tokens
> ```
> 
> ~~[EDIT]:  yeah, the PPL is pretty high here.~~
> 
> [EDIT2]:
> 
> DeepSeek-TNG-R1T2-Chimera-2.1572bpw-ppl.log
> ```
> 
> .................................................................................................
> llama_new_context_with_model: n_ctx      = 2048
> llama_new_context_with_model: n_batch    = 2048
> llama_new_context_with_model: n_ubatch   = 512
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 3
> llama_new_context_with_model: attn_max_b = 256
> llama_new_context_with_model: fused_moe  = 0
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:      CUDA0 KV buffer size =    25.11 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =    25.11 MiB
> llama_kv_cache_init:      CUDA2 KV buffer size =    22.72 MiB
> llama_new_context_with_model: KV self size  =   72.91 MiB, c^KV (q8_0):   72.91 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =     1.97 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size =   503.00 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size =   477.50 MiB
> llama_new_context_with_model:      CUDA2 compute buffer size =   477.50 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =   226.01 MiB
> llama_new_context_with_model: graph nodes  = 3664
> llama_new_context_with_model: graph splits = 120
> 
> system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
> perplexity: tokenizing the input ..
> perplexity: tokenization took 1060.51 ms
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
> perplexity: 38.68 seconds per pass - ETA 1 hours 30.40 minutes
> [1]3.1395,[2]3.7800,[3]2.6898,[4]2.3015,[5]2.1432,[6]2.0281,[7]1.9105,[8]1.8493,[9]1.8018,[10]1.7491,[11]1.7756,[12]1.8668,[13]1.9075,[14]2.0470,[15]2.1921,[16]2.2339,[17]2.4061,[18]2.5351,[19]2.5051,[20]2.5122,[21]2.6119,[22]2.5630,[23]2.5174,[24]2.5376,[25]2.4927,[26]2.4540,[27]2.5016,[28]2.5118,[29]2.5748,[30]2.6013,[31]2.6353,[32]2.6529,[33]2.6998,[34]2.7546,[35]2.8127,[36]2.8734,[37]2.9058,[38]2.9556,[39]2.9967,[40]3.0583,[41]3.0976,[42]3.1059,[43]3.1579,[44]3.1663,[45]3.2504,[46]3.3049,[47]3.2730,[48]3.2419,[49]3.2409,[50]3.2596,[51]3.3055,[52]3.3175,[53]3.3829,[54]3.4018,[55]3.4327,[56]3.4671,[57]3.4869,[58]3.5242,[59]3.5329,[60]3.5831,[61]3.6263,[62]3.6812,[63]3.7176,[64]3.7644,[65]3.7768,[66]3.7741,[67]3.7547,[68]3.7872,[69]3.7996,[70]3.8211,[71]3.8365,[72]3.8481,[73]3.8580,[74]3.8790,[75]3.8552,[76]3.8001,[77]3.7534,[78]3.7543,[79]3.7489,[80]3.7473,[81]3.7131,[82]3.7287,[83]3.7070,[84]3.6790,[85]3.6461,[86]3.6306,[87]3.6450,[88]3.6300,[89]3.6256,[90]3.6086,[91]3.5860,[92]3.5675,[93]3.5393,[94]3.5255,[95]3.5109,[96]3.5207,[97]3.5359,[98]3.5266,[99]3.5093,[100]3.5088,[101]3.4992,[102]3.5192,[103]3.5468,[104]3.5697,[105]3.5683,[106]3.5975,[107]3.6229,[108]3.6422,[109]3.6795,[110]3.7152,[111]3.7359,[112]3.7012,[113]3.6905,[114]3.6727,[115]3.6578,[116]3.6607,[117]3.6447,[118]3.6224,[119]3.6002,[120]3.5749,[121]3.5563,[122]3.5360,[123]3.5186,[124]3.4976,[125]3.4775,[126]3.4584,[127]3.4518,[128]3.4436,[129]3.4383,[130]3.4295,[131]3.4227,[132]3.4241,[133]3.4315,[134]3.4426,[135]3.4540,[136]3.4697,[137]3.4848,[138]3.4905,[139]3.5008,[140]3.4960,[141]3.4934,[142]3.4857,[143]3.4822,[144]3.4734,[145]3.4603,[146]3.4523,[147]3.4534,[148]3.4480,[149]3.4470,[150]3.4366,[151]3.4299,[152]3.4225,[153]3.4141,[154]3.4099,[155]3.4116,[156]3.4091,[157]3.4119,[158]3.4174,[159]3.4188,[160]3.4268,[161]3.4337,[162]3.4421,[163]3.4523,[164]3.4741,[165]3.4998,[166]3.5180,[167]3.5322,[168]3.5585,[169]3.5837,[170]3.6082,[171]3.6306,[172]3.6100,[173]3.5894,[174]3.5814,[175]3.5700,[176]3.5593,[177]3.5481,[178]3.5357,[179]3.5233,[180]3.5276,[181]3.5416,[182]3.5578,[183]3.5716,[184]3.5840,[185]3.5927,[186]3.6090,[187]3.6246,[188]3.6392,[189]3.6488,[190]3.6477,[191]3.6532,[192]3.6537,[193]3.6569,[194]3.6785,[195]3.6893,[196]3.7022,[197]3.7111,[198]3.7142,[199]3.7173,[200]3.7115,[201]3.7263,[202]3.7172,[203]3.7202,[204]3.7215,[205]3.7225,[206]3.7244,[207]3.7323,[208]3.7436,[209]3.7526,[210]3.7500,[211]3.7406,[212]3.7390,[213]3.7460,[214]3.7457,[215]3.7519,[216]3.7498,[217]3.7422,[218]3.7398,[219]3.7400,[220]3.7366,[221]3.7363,[222]3.7341,[223]3.7339,[224]3.7368,[225]3.7357,[226]3.7246,[227]3.7230,[228]3.7237,[229]3.7265,[230]3.7316,[231]3.7369,[232]3.7267,[233]3.7221,[234]3.7244,[235]3.7257,[236]3.7344,[237]3.7418,[238]3.7511,[239]3.7615,[240]3.7703,[241]3.7809,[242]3.7967,[243]3.8091,[244]3.8177,[245]3.8317,[246]3.8420,[247]3.8386,[248]3.8320,[249]3.8269,[250]3.8167,[251]3.8114,[252]3.8120,[253]3.8137,[254]3.8202,[255]3.8252,[256]3.8274,[257]3.8285,[258]3.8274,[259]3.8289,[260]3.8298,[261]3.8293,[262]3.8265,[263]3.8318,[264]3.8347,[265]3.8330,[266]3.8337,[267]3.8341,[268]3.8363,[269]3.8390,[270]3.8364,[271]3.8334,[272]3.8231,[273]3.8257,[274]3.8201,[275]3.8102,[276]3.8046,[277]3.8055,[278]3.8151,[279]3.8210,[280]3.8280,[281]3.8353,[282]3.8404,[283]3.8471,[284]3.8527,[285]3.8672,[286]3.8683,[287]3.8696,[288]3.8741,[289]3.8744,[290]3.8660,[291]3.8608,[292]3.8643,[293]3.8661,[294]3.8657,[295]3.8647,[296]3.8675,[297]3.8692,[298]3.8738,[299]3.8812,[300]3.8822,[301]3.8850,[302]3.8878,[303]3.8881,[304]3.8863,[305]3.8982,[306]3.9054,[307]3.9166,[308]3.9028,[309]3.8984,[310]3.8896,[311]3.8946,[312]3.8983,[313]3.9050,[314]3.9071,[315]3.9101,[316]3.9092,[317]3.9097,[318]3.9090,[319]3.9080,[320]3.9115,[321]3.9099,[322]3.9100,[323]3.9168,[324]3.9160,[325]3.9205,[326]3.9242,[327]3.9270,[328]3.9285,[329]3.9294,[330]3.9356,[331]3.9391,[332]3.9428,[333]3.9403,[334]3.9387,[335]3.9380,[336]3.9371,[337]3.9374,[338]3.9362,[339]3.9382,[340]3.9412,[341]3.9467,[342]3.9558,[343]3.9654,[344]3.9709,[345]3.9664,[346]3.9613,[347]3.9592,[348]3.9520,[349]3.9497,[350]3.9479,[351]3.9537,[352]3.9696,[353]3.9795,[354]3.9931,[355]4.0033,[356]4.0096,[357]4.0231,[358]4.0342,[359]4.0372,[360]4.0430,[361]4.0529,[362]4.0618,[363]4.0676,[364]4.0739,[365]4.0796,[366]4.0900,[367]4.0985,[368]4.1052,[369]4.1127,[370]4.1211,[371]4.1354,[372]4.1445,[373]4.1472,[374]4.1501,[375]4.1543,[376]4.1668,[377]4.1784,[378]4.1800,[379]4.1785,[380]4.1753,[381]4.1797,[382]4.1850,[383]4.1875,[384]4.1921,[385]4.1961,[386]4.2019,[387]4.2080,[388]4.2100,[389]4.1979,[390]4.1875,[391]4.1762,[392]4.1705,[393]4.1630,[394]4.1533,[395]4.1447,[396]4.1357,[397]4.1270,[398]4.1163,[399]4.1048,[400]4.0953,[401]4.0826,[402]4.0713,[403]4.0604,[404]4.0475,[405]4.0359,[406]4.0246,[407]4.0145,[408]4.0047,[409]3.9946,[410]3.9882,[411]3.9903,[412]3.9870,[413]3.9907,[414]3.9939,[415]3.9915,[416]3.9920,[417]3.9936,[418]3.9874,[419]3.9898,[420]3.9859,[421]3.9845,[422]3.9866,[423]3.9860,[424]3.9905,[425]3.9902,[426]3.9903,[427]3.9897,[428]3.9931,[429]3.9936,[430]3.9977,[431]3.9984,[432]3.9967,[433]3.9914,[434]3.9923,[435]3.9869,[436]3.9821,[437]3.9787,[438]3.9761,[439]3.9765,[440]3.9815,[441]3.9876,[442]3.9955,[443]3.9933,[444]3.9931,[445]3.9937,[446]3.9988,[447]4.0023,[448]4.0049,[449]4.0073,[450]4.0117,[451]4.0147,[452]4.0175,[453]4.0191,[454]4.0166,[455]4.0190,[456]4.0185,[457]4.0209,[458]4.0255,[459]4.0254,[460]4.0246,[461]4.0204,[462]4.0236,[463]4.0310,[464]4.0369,[465]4.0308,[466]4.0309,[467]4.0314,[468]4.0336,[469]4.0315,[470]4.0284,[471]4.0288,[472]4.0300,[473]4.0285,[474]4.0271,[475]4.0274,[476]4.0259,[477]4.0243,[478]4.0247,[479]4.0268,[480]4.0291,[481]4.0246,[482]4.0281,[483]4.0265,[484]4.0301,[485]4.0369,[486]4.0396,[487]4.0424,[488]4.0474,[489]4.0495,[490]4.0538,[491]4.0602,[492]4.0642,[493]4.0639,[494]4.0642,[495]4.0662,[496]4.0673,[497]4.0696,[498]4.0695,[499]4.0678,[500]4.0712,[501]4.0752,[502]4.0736,[503]4.0721,[504]4.0742,[505]4.0768,[506]4.0859,[507]4.0881,[508]4.0909,[509]4.0826,[510]4.0808,[511]4.0764,[512]4.0734,[513]4.0678,[514]4.0673,[515]4.0699,[516]4.0657,[517]4.0663,[518]4.0654,[519]4.0661,[520]4.0709,[521]4.0688,[522]4.0665,[523]4.0725,[524]4.0715,[525]4.0696,[526]4.0662,[527]4.0598,[528]4.0576,[529]4.0534,[530]4.0509,[531]4.0464,[532]4.0387,[533]4.0315,[534]4.0283,[535]4.0300,[536]4.0333,[537]4.0367,[538]4.0397,[539]4.0432,[540]4.0487,[541]4.0530,[542]4.0561,[543]4.0528,[544]4.0490,[545]4.0494,[546]4.0428,[547]4.0392,[548]4.0324,[549]4.0257,[550]4.0204,[551]4.0154,[552]4.0096,[553]4.0053,[554]4.0052,[555]4.0027,[556]4.0055,[557]4.0095,[558]4.0159,[559]4.0198,[560]4.0255,[561]4.0228,
> Final estimate: PPL = 4.0228 +/- 0.02296
> 
> llama_print_timings:        load time =   32655.82 ms
> llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings: prompt eval time = 4981373.84 ms / 287232 tokens (   17.34 ms per token,    57.66 tokens per second)
> llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings:       total time = 4990750.04 ms / 287233 tokens
> ```

> 👤 **Thireus** replied on **2025-07-26** at **20:03:01**
> 
> @magikRUKKOLA - Thanks! Glad to see the first one performs okay compared to @ubergarm given the size. For the second one it's not so good - given that I have not put a lot of effort into checking the quant_assign params this is expected I guess...
> 
> You should be able to get much faster PPL compute if you bump the batch sizes: `-b 16384 -ub 8192`

> 👤 **magikRUKKOLA** replied on **2025-07-26** at **21:07:38**
> 
> [EDIT3]:
> 
> 
> File: /opt/THIREUS/DeepSeek-TNG-R1T2-Chimera-2.5961bpw-ppl.log
> ```
> 
> ...................................................................................................
> llama_new_context_with_model: n_ctx      = 16384
> llama_new_context_with_model: n_batch    = 16384
> llama_new_context_with_model: n_ubatch   = 16384
> llama_new_context_with_model: flash_attn = 1
> llama_new_context_with_model: mla_attn   = 3
> llama_new_context_with_model: attn_max_b = 256
> llama_new_context_with_model: fused_moe  = 0
> llama_new_context_with_model: ser        = -1, 0
> llama_new_context_with_model: freq_base  = 10000.0
> llama_new_context_with_model: freq_scale = 0.025
> llama_kv_cache_init:      CUDA0 KV buffer size =   200.82 MiB
> llama_kv_cache_init:      CUDA1 KV buffer size =   200.82 MiB
> llama_kv_cache_init:      CUDA2 KV buffer size =   181.70 MiB
> llama_new_context_with_model: KV self size  =  583.31 MiB, c^KV (q8_0):  583.31 MiB, kv^T: not used
> llama_new_context_with_model:  CUDA_Host  output buffer size =    15.78 MiB
> llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
> llama_new_context_with_model:      CUDA0 compute buffer size = 11520.06 MiB
> llama_new_context_with_model:      CUDA1 compute buffer size =  6208.06 MiB
> llama_new_context_with_model:      CUDA2 compute buffer size =  8528.00 MiB
> llama_new_context_with_model:  CUDA_Host compute buffer size =  1472.19 MiB
> llama_new_context_with_model: graph nodes  = 8361
> llama_new_context_with_model: graph splits = 218
> 
> system_info: n_threads = 64 / 128 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
> perplexity: tokenizing the input ..
> perplexity: tokenization took 1113.24 ms
> perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=16384, n_seq=32
> perplexity: 70.51 seconds per pass - ETA 20.60 minutes
> [1]2.7995,[2]3.5450,[3]2.5074,[4]2.1400,[5]1.9401,[6]1.7820,[7]1.6822,[8]1.6148,[9]1.5679,[10]1.5298,[11]1.5213,[12]1.6218,[13]1.6435,[14]1.7727,[15]1.9145,[16]1.9638,[17]2.1265,[18]2.2572,[19]2.2212,[20]2.2111,[21]2.3081,[22]2.2739,[23]2.2457,[24]2.2548,[25]2.2179,[26]2.1865,[27]2.2370,[28]2.2430,[29]2.3020,[30]2.3306,[31]2.3630,[32]2.3845,[33]2.4280,[34]2.4767,[35]2.5332,[36]2.5921,[37]2.6251,[38]2.6737,[39]2.7134,[40]2.7772,[41]2.8208,[42]2.8317,[43]2.8850,[44]2.8973,[45]2.9760,[46]3.0267,[47]2.9834,[48]2.9446,[49]2.9384,[50]2.9588,[51]3.0057,[52]3.0175,[53]3.0766,[54]3.0934,[55]3.1238,[56]3.1578,[57]3.1710,[58]3.2051,[59]3.2150,[60]3.2600,[61]3.3024,[62]3.3503,[63]3.3833,[64]3.4274,[65]3.4379,[66]3.4307,[67]3.4131,[68]3.4419,[69]3.4405,[70]3.4651,[71]3.4823,[72]3.4977,[73]3.5107,[74]3.5327,[75]3.5100,[76]3.4592,[77]3.4129,[78]3.4091,[79]3.3913,[80]3.3812,[81]3.3451,[82]3.3501,[83]3.3210,[84]3.2871,[85]3.2567,[86]3.2390,[87]3.2439,[88]3.2208,[89]3.2090,[90]3.1884,[91]3.1645,[92]3.1411,[93]3.1122,[94]3.0916,[95]3.0726,[96]3.0773,[97]3.0877,[98]3.0739,[99]3.0557,[100]3.0585,[101]3.0508,[102]3.0686,[103]3.0936,[104]3.1153,[105]3.1151,[106]3.1417,[107]3.1661,[108]3.1864,[109]3.2220,[110]3.2567,[111]3.2782,[112]3.2470,[113]3.2371,[114]3.2171,[115]3.2031,[116]3.2026,[117]3.1815,[118]3.1588,[119]3.1367,[120]3.1146,[121]3.0978,[122]3.0787,[123]3.0623,[124]3.0431,[125]3.0239,[126]3.0052,[127]2.9940,[128]2.9886,[129]2.9808,[130]2.9732,[131]2.9665,[132]2.9727,[133]2.9812,[134]2.9887,[135]2.9998,[136]3.0165,[137]3.0309,[138]3.0376,[139]3.0488,[140]3.0472,[141]3.0467,[142]3.0430,[143]3.0418,[144]3.0361,[145]3.0260,[146]3.0217,[147]3.0247,[148]3.0216,[149]3.0219,[150]3.0137,[151]3.0094,[152]3.0055,[153]2.9993,[154]2.9986,[155]3.0020,[156]3.0015,[157]3.0056,[158]3.0128,[159]3.0157,[160]3.0250,[161]3.0332,[162]3.0423,[163]3.0490,[164]3.0708,[165]3.0953,[166]3.1134,[167]3.1268,[168]3.1523,[169]3.1763,[170]3.1994,[171]3.2227,[172]3.2050,[173]3.1864,[174]3.1755,[175]3.1630,[176]3.1521,[177]3.1402,[178]3.1267,[179]3.1129,[180]3.1168,[181]3.1308,[182]3.1474,[183]3.1612,[184]3.1744,[185]3.1843,[186]3.2002,[187]3.2158,[188]3.2305,[189]3.2408,[190]3.2410,[191]3.2474,[192]3.2488,[193]3.2539,[194]3.2749,[195]3.2846,[196]3.2979,[197]3.3074,[198]3.3114,[199]3.3157,[200]3.3128,[201]3.3277,[202]3.3213,[203]3.3257,[204]3.3278,[205]3.3284,[206]3.3313,[207]3.3398,[208]3.3503,[209]3.3599,[210]3.3593,[211]3.3522,[212]3.3518,[213]3.3596,[214]3.3609,[215]3.3669,[216]3.3666,[217]3.3603,[218]3.3595,[219]3.3604,[220]3.3578,[221]3.3582,[222]3.3577,[223]3.3573,[224]3.3611,[225]3.3607,[226]3.3529,[227]3.3521,[228]3.3540,[229]3.3576,[230]3.3640,[231]3.3702,[232]3.3613,[233]3.3543,[234]3.3556,[235]3.3558,[236]3.3649,[237]3.3728,[238]3.3816,[239]3.3924,[240]3.4007,[241]3.4119,[242]3.4269,[243]3.4389,[244]3.4472,[245]3.4593,[246]3.4699,[247]3.4675,[248]3.4618,[249]3.4581,[250]3.4506,[251]3.4474,[252]3.4489,[253]3.4516,[254]3.4578,[255]3.4633,[256]3.4663,[257]3.4684,[258]3.4684,[259]3.4709,[260]3.4727,[261]3.4730,[262]3.4716,[263]3.4770,[264]3.4799,[265]3.4791,[266]3.4805,[267]3.4821,[268]3.4854,[269]3.4880,[270]3.4864,[271]3.4836,[272]3.4751,[273]3.4767,[274]3.4711,[275]3.4607,[276]3.4512,[277]3.4528,[278]3.4629,[279]3.4685,[280]3.4757,[281]3.4832,[282]3.4886,[283]3.4947,[284]3.5008,[285]3.5148,[286]3.5169,[287]3.5188,[288]3.5234,[289]3.5247,[290]3.5163,[291]3.5095,[292]3.5119,[293]3.5123,[294]3.5114,[295]3.5105,[296]3.5127,[297]3.5135,[298]3.5187,[299]3.5261,[300]3.5285,[301]3.5317,[302]3.5344,[303]3.5351,[304]3.5338,[305]3.5460,[306]3.5534,[307]3.5641,[308]3.5512,[309]3.5462,[310]3.5379,[311]3.5416,[312]3.5444,[313]3.5501,[314]3.5526,[315]3.5558,[316]3.5560,[317]3.5571,[318]3.5572,[319]3.5569,[320]3.5615,[321]3.5605,[322]3.5618,[323]3.5691,[324]3.5692,[325]3.5739,[326]3.5782,[327]3.5816,[328]3.5834,[329]3.5847,[330]3.5909,[331]3.5947,[332]3.5983,[333]3.5966,[334]3.5962,[335]3.5959,[336]3.5955,[337]3.5960,[338]3.5958,[339]3.5980,[340]3.6015,[341]3.6070,[342]3.6160,[343]3.6254,[344]3.6307,[345]3.6241,[346]3.6186,[347]3.6155,[348]3.6078,[349]3.6061,[350]3.6047,[351]3.6095,[352]3.6249,[353]3.6347,[354]3.6477,[355]3.6574,[356]3.6629,[357]3.6750,[358]3.6855,[359]3.6887,[360]3.6944,[361]3.7041,[362]3.7132,[363]3.7192,[364]3.7257,[365]3.7321,[366]3.7427,[367]3.7519,[368]3.7587,[369]3.7665,[370]3.7749,[371]3.7885,[372]3.7978,[373]3.8009,[374]3.8041,[375]3.8086,[376]3.8211,[377]3.8327,[378]3.8352,[379]3.8346,[380]3.8312,[381]3.8356,[382]3.8412,[383]3.8440,[384]3.8485,[385]3.8526,[386]3.8579,[387]3.8643,[388]3.8673,[389]3.8557,[390]3.8455,[391]3.8344,[392]3.8285,[393]3.8205,[394]3.8108,[395]3.8014,[396]3.7909,[397]3.7815,[398]3.7707,[399]3.7593,[400]3.7494,[401]3.7384,[402]3.7269,[403]3.7170,[404]3.7053,[405]3.6945,[406]3.6834,[407]3.6728,[408]3.6633,[409]3.6541,[410]3.6477,[411]3.6488,[412]3.6450,[413]3.6474,[414]3.6510,[415]3.6485,[416]3.6492,[417]3.6509,[418]3.6439,[419]3.6460,[420]3.6427,[421]3.6418,[422]3.6434,[423]3.6428,[424]3.6471,[425]3.6467,[426]3.6470,[427]3.6456,[428]3.6489,[429]3.6494,[430]3.6522,[431]3.6537,[432]3.6529,[433]3.6487,[434]3.6489,[435]3.6436,[436]3.6389,[437]3.6353,[438]3.6334,[439]3.6327,[440]3.6380,[441]3.6437,[442]3.6514,[443]3.6495,[444]3.6501,[445]3.6511,[446]3.6554,[447]3.6587,[448]3.6613,[449]3.6638,[450]3.6680,[451]3.6707,[452]3.6735,[453]3.6755,[454]3.6737,[455]3.6763,[456]3.6764,[457]3.6788,[458]3.6836,[459]3.6840,[460]3.6836,[461]3.6797,[462]3.6830,[463]3.6902,[464]3.6958,[465]3.6892,[466]3.6885,[467]3.6880,[468]3.6901,[469]3.6878,[470]3.6854,[471]3.6856,[472]3.6867,[473]3.6854,[474]3.6844,[475]3.6852,[476]3.6846,[477]3.6835,[478]3.6839,[479]3.6859,[480]3.6882,[481]3.6839,[482]3.6874,[483]3.6862,[484]3.6901,[485]3.6967,[486]3.6994,[487]3.7026,[488]3.7074,[489]3.7094,[490]3.7140,[491]3.7202,[492]3.7249,[493]3.7251,[494]3.7260,[495]3.7278,[496]3.7295,[497]3.7320,[498]3.7323,[499]3.7315,[500]3.7350,[501]3.7393,[502]3.7380,[503]3.7366,[504]3.7391,[505]3.7421,[506]3.7507,[507]3.7534,[508]3.7565,[509]3.7484,[510]3.7445,[511]3.7389,[512]3.7353,[513]3.7296,[514]3.7286,[515]3.7312,[516]3.7268,[517]3.7272,[518]3.7266,[519]3.7269,[520]3.7314,[521]3.7299,[522]3.7281,[523]3.7340,[524]3.7333,[525]3.7313,[526]3.7269,[527]3.7213,[528]3.7187,[529]3.7148,[530]3.7121,[531]3.7086,[532]3.7018,[533]3.6953,[534]3.6909,[535]3.6922,[536]3.6951,[537]3.6986,[538]3.7016,[539]3.7043,[540]3.7097,[541]3.7138,[542]3.7166,[543]3.7111,[544]3.7071,[545]3.7072,[546]3.7003,[547]3.6950,[548]3.6888,[549]3.6816,[550]3.6758,[551]3.6697,[552]3.6640,[553]3.6590,[554]3.6583,[555]3.6566,[556]3.6593,[557]3.6633,[558]3.6695,[559]3.6737,[560]3.6792,[561]3.6768,
> Final estimate: PPL = 3.6768 +/- 0.02052
> 
> llama_print_timings:        load time =   39812.02 ms
> llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings: prompt eval time = 1213011.59 ms / 287232 tokens (    4.22 ms per token,   236.79 tokens per second)
> llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
> llama_print_timings:       total time = 1222713.59 ms / 287233 tokens
> ```

---

👤 **magikRUKKOLA** commented on **2025-07-16** at **21:36:20**

Is there any way to predict the performance of the quant (prefill and decode) based solely on the types of the quants used?

---

👤 **anikifoss** commented on **2025-07-16** at **21:58:00**

> Is there any way to predict the performance of the quant (prefill and decode) based solely on the types of the quants used?

Yes: RAM bandwidth

Take `active_parameters * bits_per_parameter`: that's how much data you need to move from RAM to CPU per token.

For example, **Qwen3-30B-A3B** has 3 billion `active_parameters`:
- For Q8_0, you need to move `8 bits = 1 byte` per `active_parameter`
  - 3GB per token
- For Q4_0, you need to move `4 bits = 0.5 byte` per `active_parameter`
  -  1.5GB per token

You can then measure how many tokens per second you get with **Qwen3-30B-A3B**, and calculate your system's memory bandwidth (often this is around 80% of the theoretical possible bandwidth).

Once you have the system's effective memory bandwidth, you can then reverse the calculation to estimate tokens per second you will get with X active parameters:

`tokens_per_second = effective_system_bandwidth / (active_parameters * bits_per_parameter)`

Things get a little more tricky when you have a GPU in the mix. The same formula usually applies to GPU and VRAM (uncless the card is very weak at compute, like some older cards). However, if you have both GPU and CPU working together, then the slowest one (CPU) will be your bottleneck. Then you need to figure out how many active parameters will go on the GPU and how many will go on the CPU.

> 👤 **magikRUKKOLA** replied on **2025-07-16** at **22:08:55**
> 
> A stupid question -- are you basically saying that IQ1_S_R4 quant will be twise as fast as say, IQ3_KT ?  both in prefill and decode ? :)

> 👤 **anikifoss** replied on **2025-07-16** at **22:16:13**
> 
> > are you basically saying that IQ1_S_R4 quant will be twise as fast as say, IQ3_KT ? both in prefill and decode ? :)
> 
> Generally, yes. There is a small penalty you may have to pay for I-Quants, because they are little compute heavy.

> 👤 **ubergarm** replied on **2025-07-17** at **00:56:01**
> 
> It is generally more easy to predict TG by RAM bandwidth as aniki mentioned. 
> 
> PP can vary quite a bit depending on how you run it as right it is more CPU bottle-necked. 
> 
> And yeah some quants don't have MMQ kernels (basically can multiply it directly without dequantizting it which can be faster much of the time).. 
> 
> There are so many variables I tend to just pick a couple quants that fits in my rig at the desired context and llama-sweep-bench a bunch of combinations.
> 
> its usually all trade-offs like the most exciting things in engineering thereis not "one right answer" imo

> 👤 **saood06** replied on **2025-07-17** at **00:58:22**
> 
> >Things get a little more tricky when you have a GPU in the mix.
> 
> Things also get complicated when you have a NUMA system.

---

👤 **anikifoss** commented on **2025-07-16** at **22:20:13**

**Prompt Processing** uses a clever workaround to cheat the RAM bandwidth limitation. You multiply several tokens at the same time, that way you are re-using the data in the CPU cache, side-stepping the RAM bandwidth limit.

> 👤 **magikRUKKOLA** replied on **2025-07-16** at **22:30:27**
> 
> > **Prompt Processing**
> 
> Unrelated question:  have you seen MLA matrix absorbtion ( https://github.com/ikawrakow/ik_llama.cpp/discussions/599#discussion-8567748 ) implemented properly somewhere?

---

👤 **anikifoss** commented on **2025-07-17** at **16:54:59**

I have four units of MI50-32GB arriving soon (was a great deal, $150 each). Together with RTX-5090, it should give me 160GB of VRAM. So I can try benchmarking IQ1_S fully from VRAM.

Does anyone have experience with MI50s or running a mixed ROCM/CUDA setup?

If I can get MI50s working working, I'll try hooking up 22 of them into one system, for a total of 704GB VRAM. That should be enough to run my chunky Kimi-K2 quant. Will need to limit power consumption to 120W stay within 1600x2 Watts.

I found some articles online with mixed feedback about MI50s, would really appreciate if someone could share the first hand experience!

> 👤 **magikRUKKOLA** replied on **2025-07-17** at **19:14:37**
> 
> > If I can get MI50s working working, I'll try hooking up 22 of them into one system, for a total of 704GB VRAM
> 
> But exactly how?  The CPUs have a limited number of PCIe lanes that they support.

> 👤 **anikifoss** replied on **2025-07-17** at **19:17:39**
> 
> Threadripper's WRX90E has 6 PCI 5.0 x16 slots and one x8 slot. With x4x4x4x4 bifurcation, you get 26 x4 slots. I'm going via an x16 to 4 m.2 adapter, and then m.2 back to PCI x4 cable :crossed_fingers:

> 👤 **ubergarm** replied on **2025-07-17** at **19:40:56**
> 
> @anikifoss 
> 
> You are brave and i wish you the best getting AMD GPU's MI50-32GB working, and especially working with newer CUDA haha... The GPUs might be cheap, but I'm sure your time is not - it will likely be a rabbit hole, abliet possibly a fun one! hehe But honestly I hope you can get it working because lord knows we could use more cheap sources of VRAM!
> 
> Here on ik very recently vulkan support (3 days ago) was added haha... I tested the basics were working with flash attention. I tested on the following backends for it here both CUDA and AMD: https://github.com/ikawrakow/ik_llama.cpp/pull/608 You might ask @firecoperana also as they are the local vulkan enthusiast :hugs: 
> 
> I'm not sure how it would work but possibly you could use the CUDA backend for the RTX-5090 as that will be faster (in most situations) than the NVIDIA vulkan backend `NV_coopmat2` with CUDA Version: 12.9 and Driver Version: 575.64. If you have older CUDA drivers it could run `KHR_coopmat` albiet slower likely.
> 
> For the AMD stuff you can choose between RADV MESA open source community driver or the AMDVLK open source official amd driver. You can see people discussing more about benchmarking this stuff in this useful mainline thread: https://github.com/ggml-org/llama.cpp/discussions/10879
> 
> I don't think you can use ROCm/HIP here on ik's fork, but it is probably the best choice for AMD GPUs still on mainline in my very limited testing on an RX 7900 XTX 24GB VRAM GPU.
> 
> So if you want to just give it the old college try it might look something like this for enabling both backends simultaneously I guess? (after installing the `amdvlk` package or what not)
> 
> ```bash
> cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_VULKAN=ON -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1
> cmake --build build --config Release -j $(nproc)
> ```
> 
> You could test that MLA and fmoe and all that are working properly using the much easier to handle model https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite probably start with Q4_0 which seems to be well supported across all backends. Then you can iterate to other quant types and confirm speed and perplexity are okay.
> 
> Finally there are a couple discusions on vulkan and amd stuff here:
> * https://github.com/ikawrakow/ik_llama.cpp/discussions/590 (vote if u haven't seen it)
> * https://github.com/ikawrakow/ik_llama.cpp/discussions/562
> 
> :saluting_face:

> 👤 **Panchovix** replied on **2025-07-17** at **19:42:05**
> 
> I have a 7800X3D with 7 GPUs on a consumer board haha.
> 
> 5090x2, each at X8/X8 5.0 on PCIe slots to CPU
> 4090x2, each at X4/X4 4.0 (M2 to PCIe adapter, both adapter and slot support PCIe 5.0 but the 4090 does not) to CPU
> 3090x2, each at X4/X4 4.0 (M2 to PCIe adapter), chipset
> A6000 at X4 4.0, chipset.
> 
> The bottleneck is quite huge on llama/iklcpp lol. I hope by the end of the year to change to a TR system.

> 👤 **anikifoss** replied on **2025-07-17** at **19:57:58**
> 
> @Panchovix that's a cool setup! How much VRAM do you have in total, and what kind of numbers (tokens/second) are you getting when running LLMs?
> 
> I would be super interested to learn some ik_llama.cpp and vllm numbers.
> 
> I think vllm may have tensor-parallel support, so the performance from multiple accelerators should add up. With llama.cpp and forks, the split is by layer, so you'd only go as fast as the slowest GPU.

> 👤 **Panchovix** replied on **2025-07-17** at **20:03:46**
> 
> @anikifoss Total 208GB VRAM, and 192GB RAM at 6000Mhz (4x48GB), bandwidth about 56-60 GB/s. I posted some speeds here https://www.reddit.com/r/LocalLLaMA/comments/1lwnj5x/performance_benchmarks_on_deepseek/, and on full GPU (IQ2_XXS R1) I get these ones
> 
> main: n_kv_max = 16384, n_batch = 2048, n_ubatch = 1024, flash_attn = 1, n_gpu_layers = 999, n_threads = 8, n_threads_batch = 8
> 
> ```
> |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
> |-------|--------|--------|----------|----------|----------|----------|
> |  1024 |    256 |      0 |    2.005 |   510.77 |    8.588 |    29.81 |
> |  1024 |    256 |   1024 |    1.970 |   519.78 |    8.736 |    29.30 |
> |  1024 |    256 |   2048 |    2.138 |   478.86 |    8.845 |    28.94 |
> |  1024 |    256 |   3072 |    2.289 |   447.34 |    9.114 |    28.09 |
> |  1024 |    256 |   4096 |    2.490 |   411.23 |    9.248 |    27.68 |
> |  1024 |    256 |   5120 |    2.660 |   384.95 |    9.445 |    27.10 |
> |  1024 |    256 |   6144 |    2.832 |   361.63 |    9.669 |    26.48 |
> |  1024 |    256 |   7168 |    2.990 |   342.44 |    9.761 |    26.23 |
> |  1024 |    256 |   8192 |    3.250 |   315.04 |   10.047 |    25.48 |
> |  1024 |    256 |   9216 |    3.421 |   299.31 |   10.129 |    25.27 |
> |  1024 |    256 |  10240 |    3.593 |   284.96 |   10.222 |    25.04 |
> |  1024 |    256 |  11264 |    3.752 |   272.90 |   10.536 |    24.30 |
> |  1024 |    256 |  12288 |    3.923 |   261.02 |   10.635 |    24.07 |
> |  1024 |    256 |  13312 |    4.094 |   250.15 |   10.841 |    23.61 |
> |  1024 |    256 |  14336 |    4.273 |   239.62 |   10.954 |    23.37 |
> |  1024 |    256 |  15360 |    4.456 |   229.81 |   10.991 |    23.29 |
> ```
> 
> vLLM when usable with TP, it is either 2 or 4 GPUs. But it is wild faster vs GGUF in general when enabling tensor parallel. exllamav2 also supports it and it is also reallyyy fast, despite those slow PCIe 4.0 X4 lanes.
> 
> If not using TP (for example pipeline parallel to use the 7 GPUs) speeds are about the same.

> 👤 **RodriMora** replied on **2025-07-17** at **20:20:56**
> 
> > Threadripper's WRX90E has 6 PCI 5.0 x16 slots and one x8 slot. With x4x4x4x4 bifurcation, you get 26 x4 slots. I'm going via an x16 to 4 m.2 adapter, and then m.2 back to PCI x4 cable 🤞
> 
> one question, why are you going to m.2 and then to pcie? wouldn't it be easier to just do a pcie x16 to 4x4 bifurcation board?

> 👤 **anikifoss** replied on **2025-07-17** at **20:30:11**
> 
> > one question, why are you going to m.2 and the to pcie? wouldn't it be easier to just do a pcie x16 to 4x4 bifurcation board?
> 
> I used what I could find to order online. If you can find the device you are describing, please post the link :pray:

> 👤 **anikifoss** replied on **2025-07-17** at **20:40:09**
> 
> The test-kit arrived! ![test_kit](https://github.com/user-attachments/assets/f949f6c6-9250-48e7-895b-ee85d7a5a940)

> 👤 **RodriMora** replied on **2025-07-17** at **20:45:58**
> 
> > > one question, why are you going to m.2 and the to pcie? wouldn't it be easier to just do a pcie x16 to 4x4 bifurcation board?
> > 
> > I used what I could find to order online. If you can find the device you are describing, please post the link 🙏
> 
> you are right, for some reason there are no pcie x16 to x4x4x4x4 that run at pcie 4.0, all I can find is 3.0. Only this one is 4.0 but at 8x8 https://es.aliexpress.com/item/1005004963399212.html?spm=a2g0o.order_list.order_list_main.5.be65194diuISIK&gatewayAdapt=glo2esp
> The rest are 3.0, like this one: https://es.aliexpress.com/item/1005005590607272.html?spm=a2g0o.productlist.main.1.640aAbOLAbOL8n&algo_pvid=17ae5b6a-a6aa-4d1f-88e1-8559b0695de6&algo_exp_id=17ae5b6a-a6aa-4d1f-88e1-8559b0695de6-0&pdp_ext_f=%7B%22order%22%3A%2260%22%2C%22eval%22%3A%221%22%7D&pdp_npi=4%40dis%21EUR%2126.27%2124.19%21%21%2129.84%2127.48%21%40211b876e17527850121701226e1c8e%2112000033667057067%21sea%21ES%21169616054%21X&curPageLogUid=dS6OzzWevhG0&utparam-url=scene%3Asearch%7Cquery_from%3A

> 👤 **anikifoss** replied on **2025-07-17** at **20:57:31**
> 
> MI50s are from 2018, they are PCI 3.0, so the second device would work! There is a reason they are $150 each :smiling_face_with_tear:

> 👤 **Panchovix** replied on **2025-07-17** at **21:03:39**
> 
> For X16 to PCIe 4X4 4.0 you can get some PCIe X16 to 4X M2 adapters PCIe4.0 and then 4 M2 to PCIe adapters. It works fine. More expensive tho.

> 👤 **anikifoss** replied on **2025-07-17** at **23:01:58**
> 
> @Panchovix thanks for sharing the numbers, 23.29 tokens/sec with 15360 context is impressive!

> 👤 **Ph0rk0z** replied on **2025-07-19** at **15:46:58**
> 
> They are $150 due to the cooling and software hassle :P
> 
> I would have loved these instead of P40s, they were $500+ when the former were $150 themselves.

> 👤 **anikifoss** replied on **2025-07-19** at **16:03:14**
> 
> Any issues with P40? I considered those, but the RAM density was not high enough to load something like DeepSeek with the max loadout.

> 👤 **Ph0rk0z** replied on **2025-07-20** at **13:54:18**
> 
> They are getting dropped from the next cuda driver and have terrible FP16 performance, so everything has to be calculated in FP32 for things like transformers. They do have int8 and DP4A instructions, however and I think llama.cpp makes use of it.

> 👤 **Said-Akbar** replied on **2025-07-23** at **01:16:53**
> 
> @anikifoss how is MI50 card setup going? I tried vulkan and these cards are not fast with vulkan. I am using ROCm 6.2.4 and vllm with tensor parallelism gives good results. e.g. qwen3 72B 4bit runs at 20t/s initially with 2xMI50. At 32k context, TG is around 12 t/s and PP stays around 250t/s. 
> But unfortunately, I do not see the same performance in llama.cpp. For the same model at Q4_1, I see TG 12 t/s initially. PP is 50 t/s from the start and this is using a row split. Layer split is even slower (TG starts at 10 t/s). So, there is 50% drop in TG and 500% drop in PP in llama.cpp for the similar sized quants.

> 👤 **anikifoss** replied on **2025-07-23** at **16:55:11**
> 
> @Said-Akbar I got 4MI50 working in my system: 2 plugged in directly, and 2 via x16 to x4x4x4x4 m2 bifurcation adapter card.
> 
> TLDR:
> - These are great for the price, but the lower compute performance requires a good tensor-parallel implementation that can fully utilize multiple MI50s cards.
> - VLLM provides some tensor parellelims, but is unable to fully load more than 2 cards. I'm going to sit and wait until better tensor parallel support in llama.cpp.
> - There are [some recent efforts](https://github.com/ggml-org/llama.cpp/pull/13818) to add tensor paralell support to llama.cpp, hopefully they will lead somewhere.
> 
> More details:
> - ROCM installation took about one day of tinkering, and that's only because I did not follow pre and post install instructions (like adding the user to video and render groups, and manuall adding installation onto the path).
> - The cooling fans added to the cards are effective, but loud (and more cards = louder)
> - These card have generous VRAM, but the chip is about 4x to 6x slower (depending on the task) than a 5090. So the PP and TG speeds are not amazing.
> - VLLM works from a [custom vllm-gfx906 fork](https://github.com/nlzy/vllm-gfx906)
> - VLLM's tensor parallel does not boost the speed above 2x of one card, but to me this seems like a VLLM's task scheduling limitation rather than the fault of MI50s
> 
> ![mi50_test](https://github.com/user-attachments/assets/71888b96-1fdf-48cf-a6ba-f5966312f1aa)

---

👤 **ikawrakow** commented on **2025-07-18** at **17:09:16**

Btw, because many people in this thread are running calculations with models that contained `IQ1_M` quants, and that lead to a crash when using `-fmoe`, this is now fixed via PR [#630](https://github.com/ikawrakow/ik_llama.cpp/issues/630) that just got merged. I.e., you can now use `-fmoe` for those models again.