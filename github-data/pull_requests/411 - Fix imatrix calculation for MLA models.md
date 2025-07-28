## 🔀 [Pull Request #411](https://github.com/ikawrakow/ik_llama.cpp/pull/411) - Fix imatrix calculation for MLA models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_mla_imatrix` |
| **Target Branch** | `main` |
| **Created** | 2025-05-12 |
| **Updated** | 2025-05-30 |
| **Merged** | 2025-05-13 |

---

## 📄 Description

Mainline `llama.cpp` implemented MLA for DeepSeek models in [this PR](https://github.com/ggml-org/llama.cpp/pull/12801) 2.5 months after MLA was available here. The PR broke backwards compatibility with existing DeepSeek GGUFs. The incompatibility was handled in PR [#394](https://github.com/ikawrakow/ik_llama.cpp/issues/394), and the reduced prompt processing performance with `llama.cpp`-style MLA GGUFs was recovered in [#409](https://github.com/ikawrakow/ik_llama.cpp/issues/409).

This PR fixes imatrix calculation for `llama.cpp`-style MLA GGUFs. The mainline MLA implementation splits the original `attn_kv_b` 2D tensor into `attn_k_b` and `attn_v_b`, which are 3D and have the shape `128 x n_lora x n_head` (`attn_k_b`) and `n_lora x 128 x n_head` (`attn_v_b`). When the `imatrix` tool was written there were only 2D tensors in the models, so it does not really work for the new 3D MLA tensors. There are two issues:
* The first issue is that the activations are not contiguous, and this leads to a crash in the `imatrix` tool. The crash was fixed in mainline `llama.cpp` in [PR 13286](https://github.com/ggml-org/llama.cpp/pull/13286), and is fixed here with this PR
* The author of PR 13286 correctly noticed that 3D tensors are not handled, but didn't know what to do, so left the data collection the way it is. The result is that if one computes an imatrix for a DeepSeek model with any `llama.cpp` version after [PR 13286](https://github.com/ggml-org/llama.cpp/pull/13286) was merged, one will not be able to use this imatrix to quantize a model. This PR handles the situation the way it should be handled: the imatrix for the 3D tensors needs to have `128*n_head` (`attn_k_b`) or `512*n_head` (`attn_v_b`) entries.

It is now almost a month since the `llama.cpp` [MLA PR](https://github.com/ggml-org/llama.cpp/pull/12801) was merged, so I'm wondering what "quant cookers" (as @ubergarm likes to call them) have been doing for MLA models. Hence, pinging @bartowski1182 and @danielhanchen.

---

## 💬 Conversation

👤 **bartowski1182** commented on **2025-05-12** at **21:49:14**

I have been purposefully avoiding reuploading with MLA, not even with the awareness of this glaring issue :')

And of course even these changes you've made, despite me knowing your exact intentions, are black magic to me, so I personally wouldn't have been able to even consider making this change upstream

---

👤 **danielhanchen** commented on **2025-05-13** at **15:50:35**

Super nice work @ikawrakow ! I had to temporarily disable quantizing the _v and _b matrices and left them in Q8_0 - you're new changes are super good - nice again!

---

👤 **ThomasBaruzier** commented on **2025-05-13** at **19:30:11**

Thank you for this!
I would be very grateful if anyone have the time/compute to create an imatrix for DeepSeek V3 0324 from this PR and uploads it to HF. It would probably take a week or two on my hardware

---

👤 **ikawrakow** commented on **2025-05-14** at **11:01:31**

I don't have the hardware to play with DeepSeek-V3/R1, but I'm curious about potential performance gains one can get that way. Published quantized models tend to use high-bit quants for the attention tensors (and after the MLA changes in `llama.cpp` they are all `Q8_0`). This is fine in terms of model size. But for token generation attention tensors are in the range of 40% of the model weights that need to get fetched from RAM/VRAM, so a lower bpw quantization type is going to have a non-negligible positive impact on performance. With this PR a proper imatrix can be computed, so perhaps it is feasible to go to lower bpw quantization for attention tensors without significant decrease in quantized model quality. From quick experiments with DeepSeek-V2-16B, a high-quality 5-bit quantization such as `IQ5_K` for the attention tensors is on par with `Q8_0`.

---

👤 **saood06** commented on **2025-05-14** at **11:26:48**

>so perhaps it is feasible to go to lower bpw quantization for attention tensors without significant decrease in quantized model quality. From quick experiments with DeepSeek-V2-16B, a high-quality 5-bit quantization such as `IQ5_K` for the attention tensors is on par with `Q8_0`.

This is why I've been running pure IQ4_K and my next mix is going be a mix of IQ4_KS and Q4_K and IQ4_K.

---

👤 **ThomasBaruzier** commented on **2025-05-14** at **11:33:47**

> I don't have the hardware to play with DeepSeek-V3/R1

Do you accept donations? You could feature such a page on your README explaining the goal of investing in a test bench for your experiments with this fork. You already have a 4090 iirc, so a second-hand CPU server with ~256-512 GB of RAM for ~0.5-1k € on eBay could work. I believe you've helped enough people that some would be willing to help.

---

👤 **ikawrakow** commented on **2025-05-14** at **12:00:50**

> Do you accept donations?

There is a company that wanted to sponsor me to get my hands on a higher end system. It even seemed to go ahead, but it looks like things got lost on their end. I guess I have to remind them.

I even own a Ryzen-5975WX system that I inherited from the company I was working for when it died. It has 8 memory slots, but is currently configured with just 4 x 32 GB RAM. It used to be remote but circumstances changed and I got it home just 2-3 days ago. I guess, now I need to get organized, replace the RAM with 8 x 64 GB, and add a second larger SSD (the one currently inside is just 2 TB, and always full to 98% capacity). Oh, a second GPU would be good too so I can finally look into multi-GPU stuff.

---

👤 **ThomasBaruzier** commented on **2025-05-14** at **13:01:06**

Well, that's amazing news, even if your sponsor doesn't get back to you.
Quickly looking on eBay, you could get away with 512GB ECC RDIMM at 2666MHz for 450eur or 3200Mhz for 800eur
As for the GPU, I couldn't find a 4090 lower than 1.8k eur :(
Do you think TP is achievable here?

---

👤 **ikawrakow** commented on **2025-05-14** at **13:57:42**

> Well, that's amazing news, even if your sponsor doesn't get back to you.

Haha, this is because you don't know me, and so don't expect for how long I'm going to procrastinate on this.

> Do you think TP is achievable here?

What is TP?

---

👤 **ThomasBaruzier** commented on **2025-05-14** at **14:31:33**

Oh you mean procrastinate by instead submitting even more amazing PRs here lmao

TP is tensor parallelism, aiming at using 100% of each GPU during inference. But I guess it would require a tremendous amount of work to get there from a codebase that is not meant for such a feature. I don't even know if there would be significant gains because of hybrid inference bottlenecks.

https://github.com/turboderp-org/exllamav2/blob/master/exllamav2/exllamav2_ext/ext_tp.cpp

---

👤 **ikawrakow** commented on **2025-05-14** at **14:42:28**

Ah, OK, TP is one of the things I would look into if I had 2 or more GPUs. I wouldn't dare to do it in the CUDA code, but have some vague ideas how it could be done on the level of the compute graph. I have no idea if/how much performance one would gain. How much faster is exllamav2?

---

👤 **ThomasBaruzier** commented on **2025-05-14** at **14:52:13**

Without speculative decoding, 2x3090@275w:
- Llama 3.3 70B 4.5bpw, from 18.1 to 22.9 tok/s
- Mistral Large 123B 3.0bpw, from 15.5 to 22.3 tok/s

Exl3 is supposed to have even better TP performance, but it's not implemented yet.

---

👤 **ikawrakow** commented on **2025-05-14** at **15:02:49**

> Without speculative decoding, 2x3090@275w:
> 
>     * Llama 3.3 70B 4.5bpw, from 18.1 to 22.9 tok/s
> 
>     * Mistral Large 123B 3.0bpw, from 15.5 to 22.3 tok/s
> 
> 
> Exl3 is supposed to have even better TP performance, but it's not implemented yet.

So, barely faster than `llama.cpp`? I have a 4080 (717 GB/s), so less bandwidth than a 3090 (935 GB/s), and I get 125 t/s for Llama-8B at 4.5 bpw on the 4080. Napkin math: `125 * 8/70 * 935/717 = 18.6 t/s`.

---

👤 **ThomasBaruzier** commented on **2025-05-28** at **01:03:50**

Sorry for the long wait. I finally got the time to properly benchmark all the quants in this repo and multiple exl2 sizes of Llama-3.1-Nemotron-Nano-8B-v1 (maybe a bit too much, I tried to generate the exl quants based on the bpw of the equivalent gguf files, and as a result, small quants ended up a lot heavier than their gguf counterpart)

I was also curious to see how fast each quant is (for custom mixes), but I didn't convert with --pure for the sake of the benchmark.

I used basic standard parameters for both programs, and generated 1k token * 10 and averaged the result. Using ExllamaV2 0.3.1 and latest ik_llama.cpp. A single 350w RTX 3090 was used. I didn't benchmark tensor parralelism.

Commands used:

### TabbyAPI:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model-dir /home/user/exl --model-name <MODEL> --max-seq-len 4096
```

### llama-server:
```bash
CUDA_VISIBLE_DEVICES=0 ./ik_llama.cpp/llama-server -m <MODEL> -ngl 99 -mg 0 -c 4096
```

![model_size_speed_plot](https://github.com/user-attachments/assets/38bd1d0f-9dcc-4c3a-a1ef-66352548e19b)

 (yes I forgot that some types are aliases, and ended up benchmarking everything...)

<details>
<summary>Tables</summary>

### EXL2 Models

| Quant/Type | Size (MB) | Speed (tok/s) |
|------------|-----------|---------------|
| 2.38bpw | 3398 | 181.48 |
| 2.46bpw | 3464 | 178.52 |
| 2.59bpw | 3572 | 172.72 |
| 2.69bpw | 3656 | 169.28 |
| 2.74bpw | 3697 | 168.24 |
| 2.93bpw | 3855 | 159.45 |
| 2.99bpw | 3905 | 159.89 |
| 3.18bpw | 4063 | 155.24 |
| 3.27bpw | 4138 | 152.30 |
| 3.50bpw | 4330 | 145.92 |
| 3.59bpw | 4404 | 141.56 |
| 3.66bpw | 4463 | 140.75 |
| 3.78bpw | 4563 | 139.07 |
| 4.01bpw | 4754 | 134.07 |
| 4.02bpw | 4762 | 133.44 |
| 4.20bpw | 4912 | 130.74 |
| 4.32bpw | 5012 | 128.96 |
| 4.42bpw | 5095 | 130.60 |
| 4.43bpw | 5103 | 127.09 |
| 4.65bpw | 5286 | 123.61 |
| 4.68bpw | 5311 | 123.94 |
| 4.90bpw | 5494 | 121.76 |
| 5.10bpw | 5661 | 118.93 |
| 5.34bpw | 5860 | 120.08 |
| 5.52bpw | 6010 | 118.08 |
| 5.57bpw | 6052 | 117.36 |
| 5.58bpw | 6059 | 117.38 |
| 5.71bpw | 6168 | 115.87 |
| 6.0bpw | 6515 | 111.35 |
| 6.04bpw | 6548 | 110.86 |
| 6.50bpw | 6931 | 105.84 |
| 6.56bpw | 6981 | 105.06 |
| 6.63bpw | 7039 | 104.17 |
| 8.0bpw | 8177 | 91.66 |
| 8.01bpw | 8186 | 91.48 |
| 8.50bpw | 8210 | 91.17 |

### GGUF Models

| Quant/Type | Size (MB) | Speed (tok/s) |
|------------|-----------|---------------|
| IQ1_S | 1946 | 146.87 |
| IQ1_M | 2081 | 138.79 |
| IQ2_XXS | 2288 | 132.16 |
| IQ2_KS | 2361 | 122.87 |
| IQ2_XS | 2485 | 129.85 |
| IQ2_K | 2579 | 124.24 |
| IQ2_S | 2630 | 127.08 |
| IQ2_M | 2811 | 131.48 |
| Q2_K_S | 2866 | 133.95 |
| Q2_K | 3047 | 119.56 |
| IQ3_XXS | 3139 | 126.01 |
| IQ3_XS | 3355 | 120.20 |
| IQ3_K | 3445 | 106.26 |
| IQ3_S | 3511 | 119.32 |
| Q3_K_S | 3511 | 96.78 |
| IQ3_M | 3625 | 116.48 |
| Q3_K_M | 3848 | 102.49 |
| Q3_K | 3848 | 102.47 |
| IQ3_KL | 3855 | 106.91 |
| IQ4_KSS | 4027 | 107.83 |
| Q3_K_L | 4138 | 98.71 |
| IQ4_XS | 4241 | 108.82 |
| IQ4_KS | 4247 | 109.52 |
| Q4_0 | 4459 | 128.19 |
| IQ4_NL | 4461 | 105.69 |
| IQ4_K | 4461 | 99.11 |
| Q4_K_S | 4491 | 124.95 |
| Q4_K | 4700 | 120.44 |
| Q4_K_M | 4700 | 120.45 |
| Q4_1 | 4892 | 121.24 |
| IQ5_KS | 5121 | 97.28 |
| Q5_K_S | 5292 | 112.33 |
| IQ5_K | 5339 | 92.44 |
| Q5_0 | 5353 | 112.59 |
| Q5_K_M | 5475 | 109.01 |
| Q5_K | 5475 | 109.00 |
| Q5_1 | 5787 | 107.84 |
| Q6_0 | 6234 | 102.46 |
| Q6_K | 6290 | 96.95 |
| IQ6_K | 6350 | 91.02 |
| Q8_0 | 8145 | 84.93 |

</details>

For completeness, another plot with PPL metrics could have been useful, but I don't know any program that can compute PPL from an API

---

👤 **saood06** commented on **2025-05-28** at **01:17:16**

@ThomasBaruzier 

Thanks for the data. Did you accidentally create a second comment instead of editing the first? (I do appreciate the tables for raw data though).

Also this repo has three types of quants, the k-quants, i-quants which are also in mainline, and iqk-quants (see [this](https://github.com/ikawrakow/ik_llama.cpp/discussions/8)) which are not found on mainline. This is why some of the green-dots are especially close together or have sudden changes in performance as you are putting both i-quants and iqk-quants together, even though they are different types of quants.

---

👤 **ikawrakow** commented on **2025-05-28** at **07:44:31**

@ThomasBaruzier Thanks for the detailed comparison!

You are not using flash attention in `ik_llama.cpp`. For 1000 generated tokens this makes a noticeable difference in performance.

I did a quick comparison to your data for Llama-3.1-Nemotron-Nano-8B-v1 on my 4080 GPU.

First lets look at legacy (`Q4_0`, etc.) and k-quants (`Q4_K`, etc.)

 
![thomas1](https://github.com/user-attachments/assets/b6eff374-89d8-4b56-a12d-635c32540afa)

For 4+ bpw the behavior is as expected: the 4080 has less memory bandwidth, so performance is lower than your 3090. The difference decreases with decreasing bpw, that's most likely because you did not use FA. But something goes wrong on your GPU sub-4 bpw. k-quants have a very simple unpacking algorithm, so it would be unexpected if the calculation became compute bound so that the faster 4080 pulls ahead because of that.

Things go south for i- and  iqk-quants:

![thomas2](https://github.com/user-attachments/assets/5a083f78-cc9f-4b95-968a-34a34e9d46c9)

If I put all 4080 data on the same plot it looks like this:

![thomas3](https://github.com/user-attachments/assets/f49e71f5-9a82-4fe5-ae15-f97584b5fcd9)

Not much of a difference as TG is memory bound (apart from `IQ2_KS`, which is likely not fully optimized).
 
The only explanation for the massive performance difference below 4 bpw between the 4080 and the 3090 is that the 3090 somehow does not like lookup tables (all i- and iqk-quants use a non-linear mapping between quant index and dequantized model weight, and this requires lookup tables).

Here the `ik_llama.cpp` 4080 data for the above graphs

| model             |       size |          test |              t/s |
| ----------------- | ---------: | ------------: | ---------------: |
| llama 8B Q8_0     |   7.95 GiB |        tg1024 |     74.45 ± 0.20 |
| llama 8B Q6_0     |   6.08 GiB |        tg1024 |     94.15 ± 0.03 |
| llama 8B Q5_0     |   5.21 GiB |        tg1024 |    107.33 ± 0.05 |
| llama 8B Q4_0     |   4.33 GiB |        tg1024 |    124.87 ± 0.68 |
| llama 8B Q6_K     |   6.14 GiB |        tg1024 |     92.48 ± 0.31 |
| llama 8B Q5_K     |   5.16 GiB |        tg1024 |    107.81 ± 0.06 |
| llama 8B Q4_K     |   4.38 GiB |        tg1024 |    123.71 ± 0.12 |
| llama 8B Q3_K     |   3.42 GiB |        tg1024 |    139.14 ± 0.27 |
| llama 8B Q2_K     |   2.79 GiB |        tg1024 |    174.68 ± 0.10 |
| llama 8B IQ4_NL   |   4.37 GiB |        tg1024 |    121.61 ± 1.09 |
| llama 8B IQ4_XS   |   4.15 GiB |        tg1024 |    127.50 ± 0.11 |
| llama 8B IQ3_S    |   3.44 GiB |        tg1024 |    147.51 ± 0.72 |
| llama 8B IQ3_XXS  |   3.06 GiB |        tg1024 |    160.33 ± 2.03 |
| llama 8B IQ2_M    |   2.74 GiB |        tg1024 |    177.72 ± 0.03 |
| llama 8B IQ2_XS   |   2.42 GiB |        tg1024 |    190.64 ± 1.20 |
| llama 8B IQ2_XXS  |   2.23 GiB |        tg1024 |    195.61 ± 0.28 |
| llama 8B IQ1_M    |   2.03 GiB |        tg1024 |    208.15 ± 1.80 |
| llama 8B IQ1_S    |   1.89 GiB |        tg1024 |    213.89 ± 0.22 |
| llama 8B IQ6_K    |   6.19 GiB |        tg1024 |     91.90 ± 0.28 |
| llama 8B IQ5_K    |   5.16 GiB |        tg1024 |    106.67 ± 0.08 |
| llama 8B IQ5_KS   |   4.95 GiB |        tg1024 |    110.28 ± 0.37 |
| llama 8B IQ4_K    |   4.37 GiB |        tg1024 |    122.49 ± 0.09 |
| llama 8B IQ4_KS   |   4.16 GiB |        tg1024 |    127.42 ± 0.66 |
| llama 8B IQ3_K    |   3.37 GiB |        tg1024 |    146.39 ± 0.89 |
| llama 8B IQ2_K    |   2.53 GiB |        tg1024 |    178.06 ± 1.58 |
| llama 8B IQ2_KS   |   2.30 GiB |        tg1024 |    177.14 ± 0.07 |

---

👤 **ThomasBaruzier** commented on **2025-05-28** at **11:47:28**

Thanks for all the feedback!

FA helps with 4+bpw as you predicted, but for i- and iqk-quants, I'll investigate further another time, maybe a few param tweaks could help?

Here is a refined plot:
![ploty](https://github.com/user-attachments/assets/ea780039-1f21-4a73-b088-a8affe7630ad)

<details>
<summary>Tables</summary>

## EXL2 Models

| Quant/Type | Size (MB) | Speed (tok/s) |
|------------|-----------|---------------|
| 2.38bpw | 3398 | 182.65 |
| 2.59bpw | 3572 | 174.38 |
| 2.93bpw | 3855 | 163.01 |
| 3.18bpw | 4063 | 156.92 |
| 3.59bpw | 4404 | 143.35 |
| 4.02bpw | 4762 | 134.40 |
| 4.42bpw | 5095 | 131.70 |
| 4.65bpw | 5286 | 124.44 |
| 4.90bpw | 5494 | 122.50 |
| 5.57bpw | 6052 | 118.12 |
| 6.0bpw | 6515 | 112.21 |
| 6.56bpw | 6981 | 105.64 |
| 8.0bpw | 8177 | 92.36 |

## GGUF Models

| Quant/Type | Size (MB) | Speed (tok/s) |
|------------|-----------|---------------|
| IQ1_S | 1946 | 159.98 |
| IQ1_M | 2081 | 154.14 |
| IQ2_XXS | 2288 | 143.73 |
| IQ2_KS | 2361 | 133.46 |
| IQ2_XS | 2485 | 141.81 |
| IQ2_K | 2579 | 135.18 |
| IQ2_S | 2630 | 140.29 |
| IQ2_M | 2811 | 143.30 |
| Q2_K_S | 2866 | 147.24 |
| Q2_K | 3047 | 130.29 |
| IQ3_XXS | 3139 | 136.93 |
| IQ3_XS | 3355 | 132.24 |
| IQ3_K | 3445 | 115.67 |
| IQ3_S | 3511 | 131.65 |
| Q3_K_S | 3511 | 104.77 |
| IQ3_M | 3625 | 127.77 |
| Q3_K_M | 3848 | 111.58 |
| IQ3_KL | 3855 | 116.59 |
| IQ4_KSS | 4027 | 117.97 |
| Q3_K_L | 4138 | 107.06 |
| IQ4_XS | 4241 | 119.22 |
| IQ4_KS | 4247 | 120.30 |
| Q4_0 | 4459 | 141.52 |
| IQ4_NL | 4461 | 115.15 |
| IQ4_K | 4461 | 107.58 |
| Q4_K_S | 4491 | 138.01 |
| Q4_K_M | 4700 | 132.63 |
| Q4_1 | 4892 | 133.22 |
| IQ5_KS | 5121 | 105.39 |
| Q5_K_S | 5292 | 122.86 |
| IQ5_K | 5339 | 99.89 |
| Q5_0 | 5353 | 123.06 |
| Q5_K_M | 5475 | 119.12 |
| Q5_1 | 5787 | 117.32 |
| Q6_0 | 6234 | 111.12 |
| Q6_K | 6290 | 105.05 |
| IQ6_K | 6350 | 98.25 |
| Q8_0 | 8145 | 90.84 |

</details>

---

👤 **Ph0rk0z** commented on **2025-05-30** at **12:27:07**

Is it possible to repack mainline quants somehow to be ik_llama compatible? Rather than doing it on the fly to just save a "normal" version of the weights as a copy? That should regain memory lost from the work around?

>So, barely faster than llama.cpp? I have a 4080 (717 GB/s), so less bandwidth than a 3090 (935 GB/s), and I get 125 t/s for Llama-8B at 4.5 bpw on the 4080. Napkin math: 125 * 8/70 * 935/717 = 18.6 t/s

Nah. Regardless of whatever calculations, I can load 70b models in llama.cpp of all kinds. They are about as fast with pipeline parallel, but in tensor parallel it is a much larger difference as he showed. Plus that is 0 CTX speeds, as context builds, it output t/s falls much less. For multi GPU and dual CPU socket it is a worthy endeavor 100%. On larger models the responsiveness goes from bleh to wow.

---

👤 **ikawrakow** commented on **2025-05-30** at **12:40:07**

> Is it possible to repack mainline quants somehow to be ik_llama compatible?

What do you mean? All mainline quants apart from `TQ1_0` and `TQ2_0` can be used with `ik_llama.cpp`. `TQ1_0` and `TQ2_0` are BitNet specific, and there is a much faster implementation for BitNet here. If your question is if you can repack mainline quants to `*_R4` (or `*_R8`), yes, you can. You do it with
```
./bin/llama-quantize --repack $model $new_model X`
```
where `X` is some arbitrary  quantization type name (`iq4_k_r4`, etc.)

---

👤 **saood06** commented on **2025-05-30** at **12:47:37**

> > Is it possible to repack mainline quants somehow to be ik_llama compatible?
> 
> What do you mean? 

I'm guessing he means the wk_b tensors ([#259](https://github.com/ikawrakow/ik_llama.cpp/issues/259) uses the term on the fly as well). And as an answer to his question, a python script using gguf-py should be able to do it, assuming you have "donor" tensors. (on my system this on the fly generation came at a minor but measurable cost, and if I still had any "legacy" quants, that I needed to use extensively I would take this approach)

---

👤 **ikawrakow** commented on **2025-05-30** at **13:18:53**

> And as an answer to his question, a python script using gguf-py should be able to do it, assuming you have "donor" tensors. (on my system this on the fly generation came at a minor but measurable cost,

I suspect because the new tensors get created as `Q8_0`, while your original quants were IIRC 4 or 5 bit. The tensors are created as 8 bit to avoid possible accuracy loss when doing `dequantize -> transpose -> quantize without imatrix`. If you are content with potentially losing some accuracy (as you would in a python script that adds the tensors to an already quantized model), then one can add a command line option to do that on-the-fly as well.

---

👤 **saood06** commented on **2025-05-30** at **13:35:30**

> I suspect because the new tensors get created as `Q8_0`, while your original quants were IIRC 4 or 5 bit. The tensors are created as 8 bit to avoid possible accuracy loss when doing `dequantize -> transpose -> quantize without imatrix`. If you are content with potentially losing some accuracy (as you would in a python script that adds the tensors to an already quantized model), then one can add a command line option to do that on-the-fly as well.

I think I tested that theory and even accounting for that it was still a difference. I definitely have made quants that use `Q8_0` for those tensors, and I knew the on-the-fly ones were `Q8_0` at the time, but I'm not 100% sure if I did, and my notes aren't very thorough. My server is very picky about memory layout and placement.

---

👤 **ubergarm** commented on **2025-05-30** at **13:42:28**

If folks are looking for ik_llama.cpp quantized version of DeepSeek-R1-0528, I just got one cooked up and [released on huggingface here](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF).

Feel free to use the imatrix in the repo if you are making your own quants to save a step. Details on that are int he model card and it was generated from a the Q8_0.

* `DeepSeek-R1-0528-Q8_0.gguf` `Final estimate: PPL = 3.2130 +/- 0.01698`
* `DeepSeek-R1-0528-IQ3_K_R4.gguf` `Final estimate: PPL = 3.2730 +/- 0.01738`

Gonna definitely look into a smaller one now with attention tensors possibly `q6_K`/`q5_K` or maybe `iq5_ks` (which might be good now for both CUDA and CPU?). I'm guessing mainline quants probably still have to keep attention at Q8_0 since that imatrix code doesn't have this?

---

👤 **saood06** commented on **2025-05-30** at **13:50:19**

> If folks are looking for ik_llama.cpp quantized version of DeepSeek-R1-0528, I just got one cooked up and [released on huggingface here](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF).


Thank you for the imatrix. I was considering making a discussion thread for DeepSeek-R1-0528. The one we had for V3 was quite nice.

---

👤 **ikawrakow** commented on **2025-05-30** at **13:53:58**

> Gonna definitely look into a smaller one now with attention tensors possibly q6_K/q5_K or maybe iq5_ks (which might be good now for both CUDA and CPU?). I'm guessing mainline quants probably still have to keep attention at Q8_0 since that imatrix code doesn't have this?

I would be curious to see how much degradation in quality there is from using 6- or 5-bit quants for the attention tensors and shared experts. It would be also interesting to see how much mainline suffers when quantizing attention with less than `Q8_0` without having the correct imatrix. I think answering these question would be enough for a paper, so if I was a researcher desperate to get another paper on my CV, I would definitely do it.

---

👤 **saood06** commented on **2025-05-30** at **13:59:09**

> > Gonna definitely look into a smaller one now with attention tensors possibly q6_K/q5_K or maybe iq5_ks (which might be good now for both CUDA and CPU?). I'm guessing mainline quants probably still have to keep attention at Q8_0 since that imatrix code doesn't have this?
> 
> I would be curious to see how much degradation in quality there is from using 6- or 5-bit quants for the attention tensors and shared experts. It would be also interesting to see how much mainline suffers when quantizing attention with less than `Q8_0` without having the correct imatrix. I think answering these question would be enough for a paper, so if I was a researcher desperate to get another paper on my CV, I would definitely do it.

In theory if you had the compute and benchmarks, I think https://github.com/Just-Curieous/Curie would result in nice quants, but with a model this big the compute would might be very expensive.

---

👤 **ubergarm** commented on **2025-05-30** at **14:01:23**

> I would be curious to see how much degradation in quality there is from using 6- or 5-bit quants for the attention tensors and shared experts.

Yes, I wanted to do this after V3-0324, but I think now is the time to try it out. I'll probably go for `iq5_ks` given the recent improvements.

I see [unsloth's keeping k_b and v_b at Q8_0](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF?show_file_info=UD-IQ1_M%2FDeepSeek-R1-0528-UD-IQ1_M-00001-of-00005.gguf) but don't see the actual imatrix data file hrmm..
![unsloth-r1-0528-gguf-imatrix](https://github.com/user-attachments/assets/0a52cc89-f05f-47a7-9de8-49513f8ceb94)

---

👤 **Ph0rk0z** commented on **2025-05-30** at **14:05:04**

I thought there is still a penalty to memory, prompt processing and speed from using MLA containing mainline quants vs the old ones. Even if they load/work. 

As much as IQ3/Q4 quants sound nice, anything over 250gb is going to go down into unusable speeds on my system. Only get about ~50t/s PP and 10t/s using IQ2XXS as it is. If it gets much slower... Usability comes from cramming as much into the GPUs as possible because the CPUs/memory speed isn't that good.

---

👤 **saood06** commented on **2025-05-30** at **14:10:35**

> I thought there is still a penalty to memory, prompt processing and speed from using MLA containing mainline quants vs the old ones. Even if they load/work.
> 
[#394](https://github.com/ikawrakow/ik_llama.cpp/issues/394) and [#259](https://github.com/ikawrakow/ik_llama.cpp/issues/259) are different, but they both add support for methods that differ from what our convert script generates.

---

👤 **ikawrakow** commented on **2025-05-30** at **14:14:49**

> In theory if you had the compute and benchmarks, I think https://github.com/Just-Curieous/Curie would result in nice quants, but with a model this big the compute would might be very expensive.

Do we need an "AI" agent for this?
```bash
#! /bin/sh
model=...
imatrix=...
q_exps=...

for q in q6_K iq6_k q5_K iq5_k iq5_ks; do
    ./bin/llama-quantize --imatrix $imatrix --custom-q "attn=$q,shexps=$q" --custom-q $q_exps $model tmp.gguf iq3_k
    ./bin/llama-perplexity -m tmp.gguf >>log.out 2>&1
done
grep Final log.out
```

---

👤 **ikawrakow** commented on **2025-05-30** at **14:20:54**

> I thought there is still a penalty to memory, prompt processing and speed from using MLA containing mainline quants vs the old ones. Even if they load/work.

There shouldn't be after [#409](https://github.com/ikawrakow/ik_llama.cpp/issues/409). Just `-mla 3 -fa`, and it should be fine. If there is any difference in performance, it would be very minor. I don't see a real difference with the models I can run, but some systems are very finicky about where tensors end up in memory, and it that case there may be a small performance difference because the tensors created on the fly are not in the same contiguously allocated memory block as the other tensors.

---

👤 **Ph0rk0z** commented on **2025-05-30** at **14:29:29**

>https://github.com/ikawrakow/ik_llama.cpp/pull/394 and https://github.com/ikawrakow/ik_llama.cpp/pull/259 are different, but they both add support for methods that differ from what our convert script generates.

If I had the b/w to download the full model and use the script, I'd be golden. But sadly I have to go with what people upload. Losing several GB of GPU memory is another couple of tensors I can throw on there. Just trying to get a gauge of if I should avoid any new mainline quants. Unsloth was going to make some kind of 140gb one for the new R1. Even if quality is a little lower, speed is going to be like Qwen. 

>there shouldn't be after https://github.com/ikawrakow/ik_llama.cpp/pull/409. Just -mla 3 -fa, and it should be fine.

I use those settings, so it will be mostly the same memory footprint as a native quant? Single GPU for ctx, I see how it doesn't matter but for 4x24 it really does.

---

👤 **saood06** commented on **2025-05-30** at **14:31:11**

> > [#394](https://github.com/ikawrakow/ik_llama.cpp/issues/394) and [#259](https://github.com/ikawrakow/ik_llama.cpp/issues/259) are different, but they both add support for methods that differ from what our convert script generates.
> 
> If I had the b/w to download the full model and use the script, I'd be golden.

I could maybe do tensor surgery and upload just the donor parts to huggingface, if you want?

---

👤 **saood06** commented on **2025-05-30** at **14:35:02**

> Do we need an "AI" agent for this?

If you want to create a full almost continuous spectrum of quality to size trade-offs you kind of need to do a lot of experimenting. I know ubergarm and EAddario are working on trying to rank tensors/layers to achieve that goal as well, but I do not think a greedy algorithm is optimal, and doing anything more would require more than just using a ranking.

---

👤 **ubergarm** commented on **2025-05-30** at **18:06:07**

> I would be curious to see how much degradation in quality there is from using 6- or 5-bit quants for the attention tensors and shared experts.

While I don't have a Ph.D., I didn't have to vibe code this bash script to brute force check these 7 test cases varying attn and shexp but holding all else constant q4_0.

Its gonna take a long while to finish and then test perplexity on though. Will report back by later this weekend hopefully.

<details>

<summary>👈 Test Case Bash Script</summary>

```bash
#!/usr/bin/env bash

model=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-256x21B-0528-BF16-00001-of-00030.gguf
imatrix=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/imatrix-DeepSeek-R1-0528.dat
outdir=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF
basename=DeepSeek-R1-0528
base_q=q4_0

# iterate over list of tuples as attn_k_b shape requires qN_0 types
for q in q8_0,q8_0 q6_0,q6_K q6_0,iq6_k q5_0,q5_K q5_0,iq5_k q5_0,iq5_ks q4_0,q4_0
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
        2>&1 | tee -a logs/quantize-$basename-$base_q-attn-shexp-$2.gguf
done
```

</details>

> It would be also interesting to see how much mainline suffers when quantizing attention with less than Q8_0 without having the correct imatrix.

I haven't tried making MLA imatrix on mainline, but possibly there are some issues still with the 3D tensor shapes right? I'll not fuss with this for now, maybe someone else can figure this one out.

I'm gonna release a quant today with `q5_0/iq5_ks/iq4_ks` attn_k_b/attn/shexp before discovering thes results also just so there will be at least one quant available for folks to try without q8_0's for `k_b` and `v_b`.  Thanks!

---

👤 **Ph0rk0z** commented on **2025-05-30** at **19:24:55**

>I could maybe do tensor surgery and upload just the donor parts to huggingface, if you want?

So far I have smoothie qwen, 2 quants of regular qwen and the older V3 (3/24). Those all work. I wanted to get chimera but not sure there is a small enough one out there. The mini R1 from this week, I'm willing to gamble with the smallest quant, if it ever makes an appearance. 

For the future though, who knows. Might be worth it.

---

👤 **ubergarm** commented on **2025-05-30** at **20:13:14**

> Thank you for the imatrix. I was considering making a discussion thread for DeepSeek-R1-0528. The one we had for V3 was quite nice.

Good idea, I created one and will link it in my huggingface repo card to try to keep traffic directed there as any questions and discussion arise: https://github.com/ikawrakow/ik_llama.cpp/discussions/477