### 🔀 [#495](https://github.com/ikawrakow/ik_llama.cpp/pull/495) - Check if ffn_up and ffn_gate are of the same type before using fmoe

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-06-06 |
| **Updated** | 2025-07-12 |

---

#### Description

Apparently some quant cookers are going as far as using different quantization types for `ffn_up` and `ffn_gate`. As this possibility is not correctly handled in the fused `ffn_up+ffn_gate` op, this PR adds a check and disables `fmoe` in these layers.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-06-06** at **10:10:35**:<br>

Oh, I see. The model contains `IQ1_M` quants. With partial offload TG will run on the CPU, and `IQ1_M` quants are not supported there with `-fmoe`. The fused `ffn_up+ffn_gate` op relies on the `IQK` GEMM/GEMV implementation, and there is no `IQK` implementation for `IQ1_M`.

I mistakenly thought it is because Unsloth have used different quantization types for `ffn_up_exps` and `ffn_gate_exps`, which this PR fixes.

Thanks for testing. So, for now, models containing `IQ1_M` quants cannot be used with `-fmoe`.

---

👤 **Thireus** commented the **2025-06-06** at **12:48:14**:<br>

Ah! Thank you for the clarification. Where can I find the list of quantisation type currently implemented in ik_llama? I'm thinking of attempting to reproduce Unsloth dynamic GGUF quants that would only include supported ik_llama quants.

---

👤 **Thireus** commented the **2025-06-06** at **13:03:28**:<br>

Yes sorry this is what I meant, I'm looking for the file/folder where the fast CPU matrix multiplication for IQ1_M would need to be implemented please. I plan to use other UD quants so I will need to see what has been implemented so far for fast CPU matrix multiplication.

---

👤 **Thireus** commented the **2025-06-06** at **14:04:05**:<br>

I see, not cool what happened here! ... 🫤

I with unsloth could make UD quants compatible with ik_llama. Their imatrix is quite good from what I could measure for my use-cases but they don't provide the calibration dataset they use... So I believe I have a few options here to get blasting fast speed with "unsloth", not sure if all are achievable/realistic or if they even make sense:

1. Get the imatrix from unsloth and produce my own quants for ik_llama
2. Implement IQ1_M and potentially others for higher unsloth quants (dunno if they use XS and XSS in their UD, would need to check)
3. Use the provided non-UD IQ from unsloth... knowing I would not benefit from UD quality boost. However, they only provide IQ4 which I cannot run because too big for my rig, so would need to ask them to produce lower ones. 🙁

I'm leaning towards 1. as I don't understand yet the benefits of using R4 quants. But may have to change my mind and go with option 1.

---
Summary of “Missing quant‐types” per bit
	•	1 bit: IQ1_M, IQ1_BN_R4
	•	2 bit: (none)
	• 3 bit: IQ3_XS, IQ3_XS_R4, IQ3_BN, IQ3_BN_R4
	•	4 bit: IQ4_XXS, IQ4_XXS_R4, IQ4_S, IQ4_S_R4, IQ4_XS_R4, IQ4_BN, IQ4_BN_R4
	•	5 bit: IQ5_XXS, IQ5_XXS_R4, IQ5_XS, IQ5_XS_R4, IQ5_S, IQ5_S_R4, IQ5_KT, IQ5_NL, IQ5_NL_R4, IQ5_BN, IQ5_BN_R4
	•	6 bit: IQ6_XXS, IQ6_XXS_R4, IQ6_XS, IQ6_XS_R4, IQ6_S, IQ6_S_R4, IQ6_K_R4, IQ6_KS, IQ6_KS_R4, IQ6_KT, IQ6_NL, IQ6_NL_R4, IQ6_BN, IQ6_BN_R4
	•	8 bit: Q8_K, Q8_K_R4, Q8_KS, Q8_KS_R4, Q8_KT, Q8_XXS, Q8_XXS_R4, Q8_XS, Q8_XS_R4, Q8_S, Q8_S_R4, Q8_NL, Q8_NL_R4, Q8_BN, Q8_BN_R4

---

👤 **ikawrakow** commented the **2025-06-06** at **14:13:13**:<br>

* IQ1_BN_R4 does not exist
* IQ3_XS, IQ3_XS_R4, IQ3_BN, IQ3_BN_R4 - they don't exist
*  IQ4_XXS, IQ4_XXS_R4, IQ4_S, IQ4_S_R4, IQ4_XS_R4, IQ4_BN, IQ4_BN_R4 - they don't exist
* IQ5_XXS, IQ5_XXS_R4, IQ5_XS, IQ5_XS_R4, IQ5_S, IQ5_S_R4, IQ5_KT, IQ5_NL, IQ5_NL_R4, IQ5_BN, IQ5_BN_R4 - they don't exist

To see what quantization types exist, take a look [here](https://github.com/ikawrakow/ik_llama.cpp/blob/ffd87f282e76ff9d34f47efd6d3f6af2071d416a/ggml/include/ggml.h#L366). Everything below `GGML_TYPE_Q4_0_8_8` is `ik_llama.cpp` specific, so you will not find UD quants with those. The `GGML_TYPE_Q4_0_4_4, GGML_TYPE_Q4_0_4_8, GGML_TYPE_Q4_0_8_8` no longer exist in mainline `llama.cpp` (I keep them around for testing purposes), so you will not find UD quants with those either.

---

👤 **Thireus** commented the **2025-06-07** at **13:39:51**:<br>

Hey @ubergarm, thank you for the kind words and most of all for sharing your knowledge here and there, it's been incredibly valuable. I am trying to ramp up my knowledge as fast as I can at the moment. I do not have well structured and scientific methodologies, but mainly rely on some quick tricks to build just enough evidence (to my own appreciation) about what my next steps should be to 1. get a GGUF tailored to my use cases, 2. make the most use of my current hardware in an attempt to avoid spending $20k+ on new hardware which may become obsolete in a couple of years and 3. gain sufficient knowledge to be comfortable with the (ik_)llama.cpp framework which appears to be the most flexible framework there is today for enthusiasts (I've explored exllama, vllm and a few others before).

My main target is indeed to be able to process large prompts, so my evals mainly rely on 100k+ prompt processing. And only a few quants are able to remain consistent and reason well at these large context sizes.

I'm almost done creating my first dynamic quant using unsloth's DeepSeek-R1-0528 imatrix (it's taking a few hours to produce the quantized GGUF). And I'll report back if there's any success and gains (both quality and speed).

I don't think unsloth have published any of their methodologies and calibration dataset. I trust it may be better than the ones produced using calibration_data_v5_rc.txt. And from what I understand as well, this isn't the only factor that plays a role into producing better than average quants.

So, baby steps first, I'm first reproducing other people's work, then will decide if it's worth diving further into the rabbit hole - it costs a lot of time... and there are all the other interesting aspects of LLMs that are worth exploring such as building the platform that uses these models or also creating custom refined models.

---
To answer my original question, using `llama-quantize -h` is also a quick way to list the supported quants.

---

👤 **ubergarm** commented the **2025-06-07** at **16:45:29**:<br>

@Thireus 

Enjoy the journey, no rush, and glad to see you're doing your own research and testing out what has been done before to figure out how you want to proceed.

> I'm almost done creating my first dynamic quant using unsloth's DeepSeek-R1-0528 imatrix (it's taking a few hours to produce the quantized GGUF). And I'll report back if there's any success and gains (both quality and speed).

Definitely keep us posted. I'm personally skeptical that that particular imatrix will be better given it was made with the previous weights despite the model arch being the same. Feel free to use my own [imatrix dat](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/blob/main/imatrix-DeepSeek-R1-0528.dat) which was made on the updated R1-0528 weights using both `calibration_data_v5_rc.txt` plus additional data from exllamav3 as listed in the model card for folks to recreate or modify their own if desired.

>  I trust it may be better than the ones produced using calibration_data_v5_rc.txt.

I find this sentiment common though don't understand nor agree with it personally. I'm happy to be proven wrong though! Its not my job to disabuse people of superiority of unsloth quants. lmao...

Cheers!

---

👤 **Thireus** commented the **2025-06-11** at **06:04:25**:<br>

Early observations using PPL: Using unsloth's imatrix into IQ1_S quants leads to slightly degraded results. `PPL = 4.9200 +/- 0.02917`

Unless I'm missing something, there are no mind-blowing results when evaluating mixture of quants. I have not evaluated the original UDs, but from what I can see the ones I've adapted to ik don't lead to surprising results. I have yet to do more eval, but I'm already noticing that for my specific hardware and use case (110k+ context size) I should target IQ3_XXS - I believe the PPL should be around 3.34. I'll give it a go and will report back.

![DeepSeek-R1-0528-GGUFs-PPL](https://thireus.com/GITHUB/DeepSeek-R1-0528-GGUFs-PPL-01.png)

---

👤 **ubergarm** commented the **2025-06-16** at **02:13:33**:<br>

> Would you know a model that uses the same arch as DeepSeek R1-0528 that is relatively small?

Yeah ik and folks use [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) which is ~16B MoE 2.4B active.

> Here are the results:

Oh interesting, you ran made a lot of quants of that little 0.6B, very cool! Is this running on all layers offloaded on a single CUDA device with `--threads 1`? The `_r4` variants were mainly for CPU inferencing and didn't even work on CUDA [until a few weeks ago since PR461](https://github.com/ikawrakow/ik_llama.cpp/pull/461).

For DeepSeek-V2 architechture (R1-0528 etc) my strategy is:
1. Keep all `attn/shexp` ready to run fully offloaded on GPU (iq5_ks is one of the best in my experiments in terms of speed/accuracy trade-offs). If someone wants to run pure-CPU, they can use `-rtr` or manually repack them to `_r4` for CPU optimizations.
2. I'm not for sure on `attn_k_b` but due to its shape you're restricted to like `q4_0` or `q6_0` etc. I believe it is technically redundant and I'm not for sure if it is possible to prune it or the corresponding `attn_` layers with the same data. More or less I keep it around the same BPW as my other `attn` tensors.
3. Keep all routed experts `-ot exps=CPU` as `_r4` variants assuming people will use hybrid inferencing with these layers on CPU/RAM. Originally when I did this, people could *not* add a few more layers onto GPU to fill up VRAM until ik bailed me out with the more recent PRs as mentioned. In the most ideal system customized to your exact hardware you'd calculate how many extra layers fit into your VRAM and quantize those as non `_r4` varieties leaving the remainder as `_r4`. This level of customization not practical for general purpose release to public huggingface though imo.
4. output.weight is also sometimes called "head" and often left at ~6bpw as it is not repeating. Seems like q6_K is fairly common, or iq6_k, or heck I'll leave it iq5_ks just to keep things consistent with my other tensors.
5. token_embd.weight is also not repeating and can be kept similar slightly higher BPW.

Hope that sheds some more light on things.

---

👤 **ikawrakow** commented the **2025-06-16** at **10:18:19**:<br>

@Thireus

I don't think this model is very useful for measuring performance. Most tensors in this models have row sizes that are not a multiple of 256, which is required for almost all quantization types except `Q4_0, Q4_1, Q5_0, Q5_1, Q6_0, Q8_0, IQ4_NL`. When a tensor is found that can not be quantized with the requested type, it gets quantized with one of the quants just listed. So, with this model you are not really measuring the performance of most quantization types.

Also, aren't you trying to benchmark CPU performance? (your results don't look like CPU performance at all).

Either way, here are the types that you can meaningfully benchmark with this model, along with their CPU performance on my Ryzen-7950X:

| type | PP-512 |
| ---: | ---: |
| bf16_r16 | 2824.71 ± 103.89 |
| bf16     | 2706.96 ± 33.88 |
| q8_0     | 2303.43 ± 27.07 |
| q8_0_r8  | 3245.95 ± 69.42 |
| q4_0     | 2199.27 ± 24.48 |
| q4_0_r8  | 3227.76 ± 85.51 |
| q4_1     | 2200.43 ± 65.13 |
| q5_0     | 2080.88 ± 108.83 |
| q5_0_r4  | 3013.45 ± 62.07 |
| q5_1     | 2053.47 ± 52.06 |
| q6_0     | 2103.14 ± 41.86 |
| q6_0_r4  | 2945.44 ± 94.24 |
| iq4_nl   | 2162.09 ± 83.69 |
| iq4_nl_r4| 3073.78 ± 48.64 |

I also don't think it is productive to blindly go through a list of names. One does need to understand what all these types are, do they need an imatrix or not, is it better to use an imatrix or is it OK tun run without, how many bits they use, etc.
For instance, as mentioned earlier, you should never ever, not even once, use `IQ1_BN, IQ2_BN` and `IQ2_BN_R4` to quantize models that are not BitNet models.

---

👤 **saood06** commented the **2025-06-16** at **11:41:57**:<br>

>     1. Learn about different quant methods (but first, find where this documentation is...)

For each quant type you want to learn more about you can search for it. The `README` lists a lot of the newer one's alongside the PR they were introduced but there are often follow-up PRs that increase their speed.

There is a method between the two in which you do the bruteforce method, but then focus your attention on select quants you want to learn more about.

---

👤 **ikawrakow** commented the **2025-06-16** at **11:52:52**:<br>

Your brute force method is unlikely to produce  a meaningful outcome. You don't want to just find the quantization type that runs fastest on your hardware, but the quantization mix that runs the fastest **and satisfies a minimum quantization quality requirement**. Because, you know, the absolutely fastest model is the one that does no computation at all.

---

👤 **Thireus** commented the **2025-06-19** at **15:56:45**:<br>

Thank you for all the feedback. I am making small progress and I'm working towards a combination of quants that brings high speed (both prompt eval and new tokens) as well as reduced PPL on my hardware. I'm on Intel x299 and there are a lot of quants that really kill the speed (hence my initial high failure rate).

The best model I was able to produce so far in terms of speed while maintaining a fair quality has the following characteristics:
- 214GB in size
- 3.5904 +/- 0.01953 PPL
- 140.62 PP-512 (t/s)
- 6.21 t/s new tokens

I have also found that I need a model that is around 240GB in size max. So I'm currently cooking some quant mixes to achieve this (this is where the gap on the diagram is).

![DeepSeek-R1-0528-GGUFs-PPL-02](https://thireus.com/GITHUB/DeepSeek-R1-0528-GGUFs-PPL-02.png)

tl;dr: Still cooking.

---

👤 **saood06** commented the **2025-06-19** at **19:52:26**:<br>

> > I don't get why they are called "secret recipes"
> 
> For myself at least, it is jest as I do my best to make my recipes known, easy to repeat, and provide imatrix data etc.

My question was not directed toward your use which I understood as a jest, it's just that I've seen some people use it more literally.

>And yes the gguf-dump is very useful. I'm not sure why huggingface throws "bad gguf magic number" for some of my quants but not others, as I like to look at a gguf before downloading it sometimes.

It might have something to do with this? https://github.com/ikawrakow/ik_llama.cpp/issues/432

> Anyway, thanks as always for sharing all of your experience and guidance, you are very generous.

And thank you for the work you did in polishing a lot of it up and popularizing it.

> Regarding "extra 26GB of budget" type stuff, I still wonder what the best way to add a little more fat to an otherwise fairly homogeneous quant.

Well it depends even within the constraint of "homogeneous quant" there is a world of difference between low and high bpw.

>I'm not sure how best to vary some layers over other layers other than lots of trial and error.

My solution was to try to learn from not only my own trial and error but also others. I know you can try to understand it more with theory, but it seems like people can end up with good results coming from either theory or intuition.

---

👤 **Thireus** commented the **2025-06-28** at **15:57:07**:<br>

Just wanted to share that I haven't given up, in fact I have made my first breakthrough today after a week of bruteforcing and auto-analysis to find the optimum quant combination, which allowed me to cook the following dynamic quant today:

- 236GB in size
- 3.3919 +/- 0.01826 PPL
- 110.45 PP-512 (t/s)
- 4.97 t/s new tokens

![DeepSeek-R1-0528-GGUFs-PPL-03.png](https://thireus.com/GITHUB/DeepSeek-R1-0528-GGUFs-PPL-03.png)

I still need ~ 2 weeks worth of computing to achieve better results in speed and quality than the above. Then, I plan to share the methodology, scripts and quants.

---

👤 **ubergarm** commented the **2025-06-28** at **16:31:22**:<br>

@Thireus 

Thanks for the report! You're exploring the heck out of that inflection "knee point" between 200 and 300 GiB and cool to see the updated plot.

Keep up the good work, and keep in mind it is somewhat of a moving target with recent PRs like 559 which have made `iq4_k` faster than `iq4_ks` when offloaded onto CUDA for PP at least on my test rig.

Looking back I'd definitely change a few things on my quants like probably standardize using `q4_K` or `iq4_k` for token_embd and `q6_K` or `iq6_k` for final output. Also maybe tweak the first 3 `ffn` just a touch etc. Always something to tweak and tinker with which keeps this hobby interesting lol...

Cheers!

---

👤 **Thireus** commented the **2025-07-02** at **22:20:22**:<br>

Yes, I keep feeding the new quants to my automated scripts as soon as they are released/improved, so they can ingest them and see if they are of any good use. I've also fed the latest iq3_ks. I've also experimented with _kt.

I've taken a lot of shortcuts (including interpolation of partial metrics and mathematical models based on partial or guessed data) to save time and cost and speed up the quant mix discovery and calibration process. I'm not yet entirely happy about the quality of some scripts nor some algorithms that can still be improved. Nevertheless, I believe the methodology is mature enough to provide near optimum quant mixes, competing against popular quants such as unsloth quants.

I have created a script that can produce optimum mix recipes given a VRAM and RAM GB target. So, I'm happy to report I was able to produce a mixture tonight that fits exactly 240GB which was my target, and fits 99% of my free RAM without incurring any speed loss. The PPL is also the lowest I've achieved so far.

- 240GB in size
- 3.3471 +/- 0.01783 PPL
- 99.68 PP-512 (t/s)
- 4.94 t/s new tokens

Since I run my scripts on partial metrics, full metrics will be available in about 5-6 more days (I had made a mistake in my calibration dataset last week and had to redo all the computation), so there is still a bit of hope that I can reach slightly lower PPL for this size.

In the meantime, here's a zero-shot screensaver created by that mixture of quants which I very much like (part of my own quality check testing, so can't disclose the prompt): https://thireus.com/GITHUB/screensaver.py

---

👤 **Thireus** commented the **2025-07-11** at **11:23:19**:<br>

MVP1 published - https://github.com/Thireus/GGUF-Tool-Suite

Example of quant mix recipe available [here](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-3.4064bpw-3.3372ppl.242GB-GGUF_11GB-GPU_231GB-CPU.254e1cf_c044584.recipe).

- 3.3372 +/- 0.01781 ppl
- 242GB Total size
- 11GB VRAM
- 231GB RAM
- 113.10 t/s PP eval
- 5.70 t/s eval

Config: 1x 5090 + 2x 3090 + i9 7980xe with 250GB DDR4

Custom recipes can be produced within minutes for different VRAM and RAM requirements, see README file for basic instructions. Article coming soon.