### 🐛 [#411](https://github.com/ikawrakow/ik_llama.cpp/pull/411) - Fix imatrix calculation for MLA models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-12 |
| **Updated** | 2025-05-30 |

---

#### Description

Mainline `llama.cpp` implemented MLA for DeepSeek models in [this PR](https://github.com/ggml-org/llama.cpp/pull/12801) 2.5 months after MLA was available here. The PR broke backwards compatibility with existing DeepSeek GGUFs. The incompatibility was handled in PR #394, and the reduced prompt processing performance with `llama.cpp`-style MLA GGUFs was recovered in #409.

This PR fixes imatrix calculation for `llama.cpp`-style MLA GGUFs. The mainline MLA implementation splits the original `attn_kv_b` 2D tensor into `attn_k_b` and `attn_v_b`, which are 3D and have the shape `128 x n_lora x n_head` (`attn_k_b`) and `n_lora x 128 x n_head` (`attn_v_b`). When the `imatrix` tool was written there were only 2D tensors in the models, so it does not really work for the new 3D MLA tensors. There are two issues:
* The first issue is that the activations are not contiguous, and this leads to a crash in the `imatrix` tool. The crash was fixed in mainline `llama.cpp` in [PR 13286](https://github.com/ggml-org/llama.cpp/pull/13286), and is fixed here with this PR
* The author of PR 13286 correctly noticed that 3D tensors are not handled, but didn't know what to do, so left the data collection the way it is. The result is that if one computes an imatrix for a DeepSeek model with any `llama.cpp` version after [PR 13286](https://github.com/ggml-org/llama.cpp/pull/13286) was merged, one will not be able to use this imatrix to quantize a model. This PR handles the situation the way it should be handled: the imatrix for the 3D tensors needs to have `128*n_head` (`attn_k_b`) or `512*n_head` (`attn_v_b`) entries.

It is now almost a month since the `llama.cpp` [MLA PR](https://github.com/ggml-org/llama.cpp/pull/12801) was merged, so I'm wondering what "quant cookers" (as @ubergarm likes to call them) have been doing for MLA models. Hence, pinging @bartowski1182 and @danielhanchen.

---

#### 💬 Conversation

👤 **bartowski1182** commented the **2025-05-12** at **21:49:14**:<br>

I have been purposefully avoiding reuploading with MLA, not even with the awareness of this glaring issue :')

And of course even these changes you've made, despite me knowing your exact intentions, are black magic to me, so I personally wouldn't have been able to even consider making this change upstream

---

👤 **ThomasBaruzier** commented the **2025-05-13** at **19:30:11**:<br>

Thank you for this!
I would be very grateful if anyone have the time/compute to create an imatrix for DeepSeek V3 0324 from this PR and uploads it to HF. It would probably take a week or two on my hardware

---

👤 **ikawrakow** commented the **2025-05-14** at **11:01:31**:<br>

I don't have the hardware to play with DeepSeek-V3/R1, but I'm curious about potential performance gains one can get that way. Published quantized models tend to use high-bit quants for the attention tensors (and after the MLA changes in `llama.cpp` they are all `Q8_0`). This is fine in terms of model size. But for token generation attention tensors are in the range of 40% of the model weights that need to get fetched from RAM/VRAM, so a lower bpw quantization type is going to have a non-negligible positive impact on performance. With this PR a proper imatrix can be computed, so perhaps it is feasible to go to lower bpw quantization for attention tensors without significant decrease in quantized model quality. From quick experiments with DeepSeek-V2-16B, a high-quality 5-bit quantization such as `IQ5_K` for the attention tensors is on par with `Q8_0`.

---

👤 **ThomasBaruzier** commented the **2025-05-14** at **11:33:47**:<br>

> I don't have the hardware to play with DeepSeek-V3/R1

Do you accept donations? You could feature such a page on your README explaining the goal of investing in a test bench for your experiments with this fork. You already have a 4090 iirc, so a second-hand CPU server with ~256-512 GB of RAM for ~0.5-1k € on eBay could work. I believe you've helped enough people that some would be willing to help.

---

👤 **ikawrakow** commented the **2025-05-14** at **12:00:50**:<br>

> Do you accept donations?

There is a company that wanted to sponsor me to get my hands on a higher end system. It even seemed to go ahead, but it looks like things got lost on their end. I guess I have to remind them.

I even own a Ryzen-5975WX system that I inherited from the company I was working for when it died. It has 8 memory slots, but is currently configured with just 4 x 32 GB RAM. It used to be remote but circumstances changed and I got it home just 2-3 days ago. I guess, now I need to get organized, replace the RAM with 8 x 64 GB, and add a second larger SSD (the one currently inside is just 2 TB, and always full to 98% capacity). Oh, a second GPU would be good too so I can finally look into multi-GPU stuff.

---

👤 **ThomasBaruzier** commented the **2025-05-14** at **13:01:06**:<br>

Well, that's amazing news, even if your sponsor doesn't get back to you.
Quickly looking on eBay, you could get away with 512GB ECC RDIMM at 2666MHz for 450eur or 3200Mhz for 800eur
As for the GPU, I couldn't find a 4090 lower than 1.8k eur :(
Do you think TP is achievable here?

---

👤 **ikawrakow** commented the **2025-05-14** at **13:57:42**:<br>

> Well, that's amazing news, even if your sponsor doesn't get back to you.

Haha, this is because you don't know me, and so don't expect for how long I'm going to procrastinate on this.

> Do you think TP is achievable here?

What is TP?

---

👤 **ikawrakow** commented the **2025-05-14** at **14:42:28**:<br>

Ah, OK, TP is one of the things I would look into if I had 2 or more GPUs. I wouldn't dare to do it in the CUDA code, but have some vague ideas how it could be done on the level of the compute graph. I have no idea if/how much performance one would gain. How much faster is exllamav2?

---

👤 **ThomasBaruzier** commented the **2025-05-14** at **14:52:13**:<br>

Without speculative decoding, 2x3090@275w:
- Llama 3.3 70B 4.5bpw, from 18.1 to 22.9 tok/s
- Mistral Large 123B 3.0bpw, from 15.5 to 22.3 tok/s

Exl3 is supposed to have even better TP performance, but it's not implemented yet.

---

👤 **ikawrakow** commented the **2025-05-14** at **15:02:49**:<br>

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

👤 **ThomasBaruzier** commented the **2025-05-28** at **01:03:50**:<br>

Sorry for the long wait. I finally got the time to properly benchmark all the quants in this repo and multiple exl2 sizes of Llama-3.1-Nemotron-Nano-8B-v1 (maybe a bit too much, I tried to generate the exl quants based on the bpw of the equivalent gguf files, and as a result, small quants ended up a lot heavier than their gguf counterpart)

I was also curious to see how fast each quant is (for custom mixes), but I didn't convert with --pure for the sake of the benchmark.

I used basic standard parameters for both programs, and generated 1k token * 10 and averaged the result. Using ExllamaV2 0.3.1 and latest ik_llama.cpp. I didn't benchmark tensor parralelism.

A single 350w RTX 3090 was used to perform all these tests:

![model_size_speed_plot](https://github.com/user-attachments/assets/38bd1d0f-9dcc-4c3a-a1ef-66352548e19b)

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

 (yes I forgot that some types are aliases, and ended up benchmarking everything...)

For completeness, another plot with PPL metrics could have been useful, but I don't know any program that can compute PPL from an API

---

👤 **ThomasBaruzier** commented the **2025-05-28** at **11:47:28**:<br>

Thanks for all the feedback!

FA helps with 4+bpw as you predicted, but for i- and iqk-quants, I'll investigate further another time, maybe a few param tweaks could help?

Here is a refined plot:
![ploty](https://github.com/user-attachments/assets/14a96d32-cc1b-460f-bd92-93b258f61af5)

---

👤 **saood06** commented the **2025-05-30** at **13:35:30**:<br>

> I suspect because the new tensors get created as `Q8_0`, while your original quants were IIRC 4 or 5 bit. The tensors are created as 8 bit to avoid possible accuracy loss when doing `dequantize -> transpose -> quantize without imatrix`. If you are content with potentially losing some accuracy (as you would in a python script that adds the tensors to an already quantized model), then one can add a command line option to do that on-the-fly as well.

I think I tested that theory and even accounting for that it was still a difference. I definitely have made quants that use `Q8_0` for those tensors, and I knew the on-the-fly ones were `Q8_0` at the time, but I'm not 100% sure if I did, and my notes aren't very thorough.

---

👤 **ubergarm** commented the **2025-05-30** at **13:42:28**:<br>

If folks are looking for ik_llama.cpp quantized version of DeepSeek-R1-0528, I just got one cooked up and [released on huggingface here](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF).

Feel free to use the imatrix in the repo if you are making your own quants to save a step. Details on that are int he model card and it was generated from a the Q8_0.

* `DeepSeek-R1-0528-Q8_0.gguf` `Final estimate: PPL = 3.2130 +/- 0.01698`
* `DeepSeek-R1-0528-IQ3_K_R4.gguf` `Final estimate: PPL = 3.2730 +/- 0.01738`

Gonna definitely look into a smaller one now with attention tensors possibly `q6_K`/`q5_K` or maybe `iq5_ks` (which might be good now for both CUDA and CPU?). I'm guessing mainline quants probably still have to keep attention at Q8_0 since that imatrix code doesn't have this?

---

👤 **saood06** commented the **2025-05-30** at **13:50:19**:<br>

> If folks are looking for ik_llama.cpp quantized version of DeepSeek-R1-0528, I just got one cooked up and [released on huggingface here](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF).


Thank you for the imatrix. I was considering making a discussion thread for DeepSeek-R1-0528. The one we had for V3 was quite nice.

---

👤 **ikawrakow** commented the **2025-05-30** at **14:14:49**:<br>

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

👤 **ikawrakow** commented the **2025-05-30** at **14:20:54**:<br>

> I thought there is still a penalty to memory, prompt processing and speed from using MLA containing mainline quants vs the old ones. Even if they load/work.

There shouldn't be after #409. Just `-mla 3 -fa`, and it should be fine. If there is any difference in performance, it would be very minor. I don't see a real difference with the models I can run, but some systems are very finicky about where tensors end up in memory, and it that case there may be a small performance difference because the tensors created on the fly are not in the same contiguously allocated memory block.

---

👤 **saood06** commented the **2025-05-30** at **14:35:02**:<br>

> Do we need an "AI" agent for this?

If you want to create a full almost continuous spectrum of quality to size trade-offs you kind of need to do a lot of experimenting. I know ubergarm and EAddario are working on trying to rank tensors/layers to achieve that goal as well.

---

👤 **Ph0rk0z** commented the **2025-05-30** at **19:24:55**:<br>

>I could maybe do tensor surgery and upload just the donor parts to huggingface, if you want?

So far I have smoothie qwen, 2 quants of regular qwen and the older V3 (3/24). Those all work. I wanted to get chimera but not sure there is a small enough one out there. The mini R1 from now I'm willing to gamble with the smallest quant if it ever makes an appearance. 

For the future though, who knows. Might be worth it.

---

👤 **ubergarm** commented the **2025-05-30** at **20:13:14**:<br>

> Thank you for the imatrix. I was considering making a discussion thread for DeepSeek-R1-0528. The one we had for V3 was quite nice.

Good idea, I created one and will link it in my huggingface repo card to try to keep traffic directed there as any questions and discussion arise: https://github.com/ikawrakow/ik_llama.cpp/discussions/477