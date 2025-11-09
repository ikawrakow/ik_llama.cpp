## 🔀 [Pull Request #495](https://github.com/ikawrakow/ik_llama.cpp/pull/495) - Check if ffn_up and ffn_gate are of the same type before using fmoe

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `ik/check_up_gate_fmoe` |
| **Target Branch** | `main` |
| **Created** | 2025-06-06 |
| **Updated** | 2025-07-12 |

---

## 📄 Description

Apparently some quant cookers are going as far as using different quantization types for `ffn_up` and `ffn_gate`. As this possibility is not correctly handled in the fused `ffn_up+ffn_gate` op, this PR adds a check and disables `fmoe` in these layers.

---

## 💬 Conversation

👤 **Thireus** commented on **2025-06-06** at **08:43:25**

Thank you for looking into this. I'll test your change and will report back when finished. Model loads when `-fmoe` is specified now.

---

👤 **Thireus** commented on **2025-06-06** at **09:30:53**

It would appear that llama-sweep-bench and llama-cli don't like `-fmoe`.

# llama-bench - Works when `-fmoe 1` specified for unsloth model
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 ~/ik_llama-main-b3758-23c3e73-bin-win-cuda-12.8-x64/llama-bench -m DeepSeek-R1-0528-UD-IQ1_S-00001-of-00004.gguf -mla 3 -fa 1   -fmoe 1  -amb 1024   -ngl 99   -ctk f16   -ot ".ffn_(up|down)_exps.=CPU"   -b 4096 -ub 4096   --mmap 0   --threads 36   --main-gpu 0   -n 0
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 3 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
| model                          |       size |     params | backend    | ngl | threads | n_batch | n_ubatch | fa | mla |   amb | mmap | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | ------: | -------: | -: | --: | ----: | ---: | ---: | ------------: | ---------------: |
==========================================================================
Detected incompatible DeepSeek model.
Will try to fix, but there are no guarantees

*** Your prompt processing speed will be crippled ***

Consider making your own ik_llama.cpp compatible model or
ask the model provider to make one for you,
==========================================================================
Computed blk.0.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.1.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
...
| deepseek2 671B IQ1_S - 1.5625 bpw | 173.47 GiB |   672.05 B | CUDA       |  99 |      36 |    4096 |     4096 |  1 |   3 |  1024 |    0 |    1 |         pp512 |     23.32 ± 0.50 |

build: 23c3e73 (1)
```

# llama-cli - Doesn't work
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 ~/ik_llama-main-b3758-23c3e73-bin-win-cuda-12.8-x64/llama-cli -m DeepSeek-R1-0528-UD-IQ1_S-00001-of-00004.gguf  -mla 3 -fa \
  -amb 1024 \
  -fmoe \
  -ctk f16 \
  -c 16384 \
  -ngl 99 \
  -ot "blk\.(3|4)\.ffn_.*=CUDA0" -ot "blk\.(5)\.ffn_.*=CUDA1" -ot "blk\.(6)\.ffn_.*=CUDA2" \
  -ot exps=CPU \
  -b 4096 -ub 4096 \
  --warmup-batch \
  --no-mmap \
  --threads 36 \
  --main-gpu 0 \
  -p '<｜begin▁of▁sentence｜><｜User｜>What is the solution of x+5=-2?<｜Assistant｜><think>\n'
---
...
<|begin?of?sentence|><|User|>What is the solution of x+5=-2?<|Assistant|><think>
FirstD:\a\ik_llama.cpp\ik_llama.cpp\ggml\src\ggml.c:15189: fatal error
D:\a\ik_llama.cpp\ik_llama.cpp\ggml\src\ggml.c:15189: fatal error
D:\a\ik_llama.cpp\ik_llama.cpp\ggml\src\ggml.c:15189: fatal error
...
---
```

# llama-sweep-bench - Fatal error after the model loads when `-fmoe` specified for unsloth model
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 ~/ik_llama-main-b3758-23c3e73-bin-win-cuda-12.8-x64/llama-sweep-bench -m DeepSeek-R1-0528-UD-IQ1_S-00001-of-00004.gguf  -mla 3 -fa \
  -amb 1024 \
  -fmoe \
  -ctk f16 \
  -c 16384 \
  -ngl 99 \
  -ot "blk\.(3|4)\.ffn_.*=CUDA0" -ot "blk\.(5)\.ffn_.*=CUDA1" -ot "blk\.(6)\.ffn_.*=CUDA2" \
  -ot exps=CPU \
  -b 4096 -ub 4096 \
  --warmup-batch \
  --no-mmap \
  --threads 36 \
  --main-gpu 0
---
...
main: n_kv_max = 16384, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 99, n_threads = 36, n_threads_batch = 36

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
D:\a\ik_llama.cpp\ik_llama.cpp\ggml\src\ggml.c:15189: D:\a\ik_llama.cpp\ik_llama.cpp\ggml\src\ggml.c:15189: fatal error
D:\a\ik_llama.cpp\ik_llama.cpp\ggml\src\ggml.c:15189: fatal error
...
---
```

---

👤 **ikawrakow** commented on **2025-06-06** at **10:10:35**

Oh, I see. The model contains `IQ1_M` quants. With partial offload TG will run on the CPU, and `IQ1_M` quants are not supported there with `-fmoe`. The fused `ffn_up+ffn_gate` op relies on the `IQK` GEMM/GEMV implementation, and there is no `IQK` implementation for `IQ1_M`.

I mistakenly thought it is because Unsloth have used different quantization types for `ffn_up_exps` and `ffn_gate_exps`, which this PR fixes.

Thanks for testing. So, for now, models containing `IQ1_M` quants cannot be used with `-fmoe`.

---

👤 **Thireus** commented on **2025-06-06** at **12:48:14**

Ah! Thank you for the clarification. Where can I find the list of quantisation type currently implemented in ik_llama? I'm thinking of attempting to reproduce Unsloth dynamic GGUF quants that would only include supported ik_llama quants.

---

👤 **ikawrakow** commented on **2025-06-06** at **12:54:17**

> Ah! Thank you for the clarification. Where can I find the list of quantisation type currently implemented in ik_llama?

All types supported by `llama.cpp`  are also supported in `ik_llama.cpp` (+ another 3X extra types). The specific issue you have with `-fmoe` (also an `ik_llama.cpp` extension) is that `IQ1_M` does not have fast CPU matrix multiplications implemented. But it will work just fine without `-fmoe` the same way it does in `llama.cpp` (i.e., very slow).

---

👤 **Thireus** commented on **2025-06-06** at **13:03:28**

Yes sorry this is what I meant, I'm looking for the file/folder where the fast CPU matrix multiplication for IQ1_M would need to be implemented please. I plan to use other UD quants so I will need to see what has been implemented so far for fast CPU matrix multiplication.

Edit: I believe I found it - ggml/src/iqk/iqk_mul_mat.cpp

---

👤 **ikawrakow** commented on **2025-06-06** at **13:13:05**

Everything is implemented apart from `IQ1_M`. But if you want to take a look yourself, this is in the `ggml/src/iqk` folder. 
The function `MulMat::prepare()` in `iqk_mul_mat.cpp` will tell you which types are implemented.

I personally don't take `IQ1_S` and `IQ1_M` very seriously, so did not implement those. The only reason `IQ1_S` is implemented is that there was a user asking for `IQ1_S`, so I added it in PR [#212](https://github.com/ikawrakow/ik_llama.cpp/issues/212). It then turned out this specific user was only asking for it to copy the code into KTransformers.

---

👤 **Thireus** commented on **2025-06-06** at **14:04:05**

I see, not cool what happened here! ... 🫤

I with unsloth could make UD quants compatible with ik_llama. Their imatrix is quite good from what I could measure for my use-cases but they don't provide the calibration dataset they use... So I believe I have a few options here to get blasting fast speed with "unsloth", not sure if all are achievable/realistic or if they even make sense:

1. Get the imatrix from unsloth and produce my own quants for ik_llama
2. Implement IQ1_M and potentially others for higher unsloth quants (dunno if they use XS and XSS in their UD, would need to check)
3. Use the provided non-UD IQ from unsloth... knowing I would not benefit from UD quality boost. However, they only provide IQ4 which I cannot run because too big for my rig, so would need to ask them to produce lower ones. 🙁

I'm leaning towards 1. as I don't understand yet measured the benefits of using _R4 quants. But may have to change my mind and go with option 1.

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

👤 **ikawrakow** commented on **2025-06-06** at **14:13:13**

* IQ1_BN_R4 does not exist
* IQ3_XS, IQ3_XS_R4, IQ3_BN, IQ3_BN_R4 - they don't exist
*  IQ4_XXS, IQ4_XXS_R4, IQ4_S, IQ4_S_R4, IQ4_XS_R4, IQ4_BN, IQ4_BN_R4 - they don't exist
* IQ5_XXS, IQ5_XXS_R4, IQ5_XS, IQ5_XS_R4, IQ5_S, IQ5_S_R4, IQ5_KT, IQ5_NL, IQ5_NL_R4, IQ5_BN, IQ5_BN_R4 - they don't exist

To see what quantization types exist, take a look [here](https://github.com/ikawrakow/ik_llama.cpp/blob/ffd87f282e76ff9d34f47efd6d3f6af2071d416a/ggml/include/ggml.h#L366). Everything below `GGML_TYPE_Q4_0_8_8` is `ik_llama.cpp` specific, so you will not find UD quants with those. The `GGML_TYPE_Q4_0_4_4, GGML_TYPE_Q4_0_4_8, GGML_TYPE_Q4_0_8_8` no longer exist in mainline `llama.cpp` (I keep them around for testing purposes), so you will not find UD quants with those either.

---

👤 **ubergarm** commented on **2025-06-06** at **19:58:52**

Thanks for all the testing, this helps me understand why the temporary IQ1_S i had rolled seemed off in brief testing. The IQ1_S_R4 is definitely the way to go I see given the recent CUDA support.

@Thireus 

> I would not benefit from UD quality boost

## tl;dr;
You seem pretty smart, don't limit your imagination to what others like unsloth and myself have done already. ik_llama.cpp gives you a powerful palette of quant types and optimizations to come up with your own mixes and methodologies for your use case and hardware.

## Ramblings

Sounds like you've done some measurements and possibly observed the quant recipes and imatrix methodologies grouped broadly under the label of "unsloth dynamic 2.0" are good for your use cases? I'm curious how you are doing benchmarks, as it can be pretty challenging with these large models. (maybe u already posted on HF, catching up on messages now).

I'm genuinely curious as my understanding is that unsloth recently began to generate synthetic datasets including model specific tokens and using a larger context window e.g. 6-12k rather than the "normal" default 512 context. However it isn't always clear what methodology and imatrix corpus datasets were used on each quant and don't think they upload their imatrix dat files anymore either.

My own experience at least with Qwen3-30B-A3B in writing up [The Great Quant Wars of 2025](https://gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38) suggests that UD GGUFs are not necessarily "better" in a statistically measurable way at least using the usual methodologies which I try to share openly. 

In [another reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1ksw070/comment/mtq8ow7/) about why some folks seem to like unsloth quants, Daniel gave an interesting reply:

> The biggest difference I would say isn't the quants, but rather our bug fixes for every model!

I appreciate all the effort they are putting in, am very happy to have more quants to choose from, and honestly they helped get me into all this with the first very small DeepSeek-R1 quants only a few months ago now haha...  Their hard work fixing bugs is great too, and I'm glad they are trying out more methodologies, but it isn't clear to me how these effect actual model performance in common situations or that it is always "better". It definitely isn't better in all situations, such as the [128k 4x yarn quant GGUFs](https://huggingface.co/unsloth/Qwen3-30B-A3B-128K-GGUF/discussions/8) which change the Qwen recommended defaults - it might be better if all your prompts are actually 100-128k, but it gives [a measurable worse perplexity](https://github.com/vllm-project/llm-compressor/issues/1406#issuecomment-2937053069) in shorter context lengths as Qwen warns on the official model card. (link is some data I generated using exllamav3 in discussions with vllm-compressor AWQ quantizations).

Anyway, happy Friday and enjoy your weekend! I appreciate your enthusiasm and am looking forward to seeing what you cook up! Always feel free to holler at me as I'm trying to keep up with folks pushing the limits of this stuff like Ed Addario, Bartowski, Unsloth, myself, and now you and others!

Cheers!

---

👤 **Thireus** commented on **2025-06-07** at **13:39:51**

Hey @ubergarm, thank you for the kind words and most of all for sharing your knowledge here and there, it's been incredibly valuable. I am trying to ramp up my knowledge as fast as I can at the moment. I do not have well structured and scientific methodologies, but mainly rely on some quick tricks to build just enough evidence (to my own appreciation) about what my next steps should be to 1. get a GGUF tailored to my use cases, 2. make the most use of my current hardware in an attempt to avoid spending $20k+ on new hardware which may become obsolete in a couple of years and 3. gain sufficient knowledge to be comfortable with the (ik_)llama.cpp framework which appears to be the most flexible framework there is today for enthusiasts (I've explored exllama, vllm and a few others before).

My main target is indeed to be able to process large prompts, so my evals mainly rely on 100k+ prompt processing. And only a few quants are able to remain consistent and reason well at these large context sizes.

I'm almost done creating my first dynamic quant using unsloth's DeepSeek-R1-0528 imatrix (it's taking a few hours to produce the quantized GGUF). And I'll report back if there's any success and gains (both quality and speed).

I don't think unsloth have published any of their methodologies and calibration dataset. I trust it may be better than the ones produced using calibration_data_v5_rc.txt. And from what I understand as well, this isn't the only factor that plays a role into producing better than average quants.

So, baby steps first, I'm first reproducing other people's work, then will decide if it's worth diving further into the rabbit hole - it costs a lot of time... and there are all the other interesting aspects of LLMs that are worth exploring such as building the platform that uses these models or also creating custom refined models.

---
To answer my original question, using `llama-quantize -h` is also a quick way to list the supported quants.

---

👤 **ubergarm** commented on **2025-06-07** at **16:45:29**

@Thireus 

Enjoy the journey, no rush, and glad to see you're doing your own research and testing out what has been done before to figure out how you want to proceed.

> I'm almost done creating my first dynamic quant using unsloth's DeepSeek-R1-0528 imatrix (it's taking a few hours to produce the quantized GGUF). And I'll report back if there's any success and gains (both quality and speed).

Definitely keep us posted. I'm personally skeptical that that particular imatrix will be better given it was made with the previous weights despite the model arch being the same. Feel free to use my own [imatrix dat](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/blob/main/imatrix-DeepSeek-R1-0528.dat) which was made on the updated R1-0528 weights using both `calibration_data_v5_rc.txt` plus additional data from exllamav3 as listed in the model card for folks to recreate or modify their own if desired.

>  I trust it may be better than the ones produced using calibration_data_v5_rc.txt.

I find this sentiment common though don't understand nor agree with it personally. I'm happy to be proven wrong though! Its not my job to disabuse people of superiority of unsloth quants. lmao...

Cheers!

---

👤 **Thireus** commented on **2025-06-09** at **10:20:36**

@ubergarm - Thanks, I went with option 1. (Get the imatrix from unsloth and produce my own quants for ik_llama). I've adapted the quants they use in their model to be ik-optimised. I'll be testing the quality of the model.

---

👤 **Thireus** commented on **2025-06-11** at **06:04:25**

Early observations using PPL: Using unsloth's imatrix into IQ1_S quants leads to slightly degraded results. `PPL = 4.9200 +/- 0.02917`

Unless I'm missing something, there are no mind-blowing results when evaluating mixture of quants. I have not evaluated the original UDs, but from what I can see the ones I've adapted to ik don't lead to surprising results. I have yet to do more eval, but I'm already noticing that for my specific hardware and use case (110k+ context size) I should target IQ3_XXS - I believe the PPL should be around 3.34. I'll give it a go and will report back.

![DeepSeek-R1-0528-GGUFs-PPL](https://thireus.com/GITHUB/DeepSeek-R1-0528-GGUFs-PPL-01.png)

---

👤 **ubergarm** commented on **2025-06-11** at **14:15:47**

@Thireus 

Thanks for all the heavy lifting and number crunching to confirming some things. Your measured numbers for my IQ2_K_R4 and IQ3_K_R4 line up closely with my own so seems like you're methodology is sound. Nice job!

> (110k+ context size) I should target IQ3_XXS

You're in luck, because `IQ3_XXS` (and more, check the other closed PRs) just got a big boost in prompt processing today: https://github.com/ikawrakow/ik_llama.cpp/pull/516

My advice would be to consider using `iq5_ks` for all attn/shexp/token_embd based on [this experiment](https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13336629) (assuming you have enough VRAM to fit your desired large 110k context still).

Then maybe IQ3_XXS for ffn_down and IQ2_XXS for ffn_(gate|up) or something similar to hit the size for which you're aiming. `iq4_ks` for down and `iq3_xxs` (gate|up) may also be a good combo for a slightly larger quant now. I didn't run the final size numbers but you get the idea.

Again feel free to use my [R1-0528 imatrix](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/blob/main/imatrix-DeepSeek-R1-0528.dat) which was made including the fixes to imatrix MLA computation in recent [PR411](https://github.com/ikawrakow/ik_llama.cpp/pull/411) so likely the best you can find without making your own.

Have fun and keep us posted! I'd be interested in your `llama-sweep-bench` results as well comparing PP/TG between my IQ2_K_R4 with whatever IQ3_XXS mix you cook up. Cheers!

---

👤 **Thireus** commented on **2025-06-15** at **20:02:19**

I need some help to understand quant performance - how can I know which quant performs better than others? Are there metrics somewhere that I've missed?

For example, when using @ubergarm's quants:
```
# Token embedding and output tensors (GPU)
# note token_embd cannot be repacked quant type
token_embd\.weight=iq5_ks
output\.weight=iq5_ks
output_norm\.weight=iq5_ks

# First 3 dense layers (0-3) (GPU)
# Except blk.*.attn_k_b.weight is not divisible by 256 so only supports qN_0
blk\.[0-2]\.attn_k_b.*=q5_0
blk\.[0-2]\.attn_.*=iq5_ks
blk\.[0-2]\..*=iq5_ks

# All attention, norm weights, and bias tensors for MoE layers (3-60) (GPU)
# Except blk.*.attn_k_b.weight is not divisible by 256 so only supports qN_0
blk\.[3-9]\.attn_k_b.*=q5_0
blk\.[1-5][0-9]\.attn_k_b.*=q5_0
blk\.60\.attn_k_b.*=q5_0

blk\.[3-9]\.attn_.*=iq5_ks
blk\.[1-5][0-9]\.attn_.*=iq5_ks
blk\.60\.attn_.*=iq5_ks

blk\.[3-9]\.ffn_norm\.weight=iq5_ks
blk\.[1-5][0-9]\.ffn_norm\.weight=iq5_ks
blk\.60\.ffn_norm\.weight=iq5_ks

blk\.[3-9]\.exp_probs_b\.bias=iq5_ks
blk\.[1-5][0-9]\.exp_probs_b\.bias=iq5_ks
blk\.60\.exp_probs_b\.bias=iq5_ks

# Shared Experts (3-60) (GPU)
blk\.[3-9]\.ffn_down_shexp\.weight=iq5_ks
blk\.[1-5][0-9]\.ffn_down_shexp\.weight=iq5_ks
blk\.60\.ffn_down_shexp\.weight=iq5_ks

blk\.[3-9]\.ffn_(gate|up)_shexp\.weight=iq4_ks
blk\.[1-5][0-9]\.ffn_(gate|up)_shexp\.weight=iq4_ks
blk\.60\.ffn_(gate|up)_shexp\.weight=iq4_ks

# Routed Experts (3-60) (CPU)
blk\.[3-9]\.ffn_down_exps\.weight=iq3_k_r4
blk\.[1-5][0-9]\.ffn_down_exps\.weight=iq3_k_r4
blk\.60\.ffn_down_exps\.weight=iq3_k_r4

blk\.[3-9]\.ffn_(gate|up)_exps\.weight=iq2_k_r4
blk\.[1-5][0-9]\.ffn_(gate|up)_exps\.weight=iq2_k_r4
blk\.60\.ffn_(gate|up)_exps\.weight=iq2_k_r4
```

Perfs are great - pp and eval are through the roof, example: 5.50t/s for eval.

But now, if I decide to change the quant of the routed experts from `iq3_k_r4` to `iq3_xxs_r4`, pp and eval get divided by 10, example: 0.62t/s for eval.

Why is it that changing `iq3_k_r4` to `iq3_xxs_r4` results in such disproportionate and unexpected performance drop? I have in fact noticed this with other quants too, in fact any attempt that I've made at trying various quant mixes result in the same outcome: perf drops significantly making the model unusable. The only time I get great perfs if when using @ubergarm's secret recipe from https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF. At which point I am wondering if these are the only possible and usable recipes...

How do I know which quant will result in perf degradation in advance and which ones result in great perfs? And why is it that `iq3_xxs_r4` performs 10x worse than `iq3_k_r4` which is quite counter intuitive to me?

I'm really lost here.

Edit: ChatGPT to the rescue on that one - https://chatgpt.com/share/684f3874-83f4-800f-a3b6-e9ac15ec48cd but unsure how accurate the answer is.

---

👤 **ubergarm** commented on **2025-06-15** at **21:21:11**

@Thireus 

Great job you've come a long way in a short time! This is a great question. You're getting deep enough now that the answers are not simple! I'm no expert and will likely make some mistakes in this, but you'll get the gist.

> I need some help to understand quant performance

Given the context of your post my impression is you are interested in "performance" in terms of inference *speed* both token generation and prompt processing. (not in terms of quality like PPL/KLD similarity to the original model).

> Perfs are great - pp and eval are through the roof, example: 5.50t/s for eval.

Great to hear! I tried to choose my quants recipes based on a mix of quality and speed. Though given recent improvements in various quant inferencing implementations, there are probably other good combinations as well depending on your exact hardware.

Keep in mind the "best" quant in terms of speed depends on a number of things:
1. Overall size
  * Prompt Processing (PP) tends to be CPU limited, while Token Generation (TG) tends to be RAM i/o bandwidth limited. using smaller quants means fewer bits which can speed up TG for example, but at likely a trade-off of typically "worse quality" for lower BPW quants.
2. Hardware support
  * LLM inferencing is a game of finding the most optimized kernels/algorithms to multiply matricies and vectors making use of *exact* hardware registers and flags available for CPU and GPUs. Type `lscpu | grep avx2` to see if your CPU supports AVX2 instruction set (Zen4 and newer). Newer CUDA architectures like 4090 and up support native fp8 registers etc.
3. Software Implementation
  * Even if you have the hardware, it doesn't matter unless the software is optimized to take advantage of it. ik's project tends to focus on CPU optimizations with some CUDA optimizations as well from what I've seen.
  * Do a google search for MARLIN GEMM, and CUTLASS GEMM, and BLAS implementations, and you'll see there is an entire academic industrial complex built up around optimization of matrix math beginning around the early 80s with FORTRAN that continues today across multiple languages and target hardware.

> How do I know which quant will result in perf degradation in advance and which ones result in great perfs?

Right more specific to your exact problem at hand: "Which quants should I choose for my recipe to optimize speed?". I'm not sure there is a great way to know "in advance" honestly unless you look through the code to see which quants have like MMQ (quantized matrix multiplication psure) implementations for your target hardware. If it has to rely on fp16 and fp32 dtype registers, it will likely be slower especially on CPU etc.

Personally, I pick a small model of similar architecture and make a bunch of quants. Then test them with llama-sweep-bench to empirically discover which ones are faster e.g. `iq5_ks` tends to be faster than `iq5_k` given the block size allowing less time spent unpacking and more time processing.

Then I use what I learn in that experiment to inform how to quantize the larger models.

> And why is it that iq3_xxs_r4 performs 10x worse than iq3_k_r4 which is quite counter intuitive to me?

You saw recent updates to `iq3_xxs` in [PR516](https://github.com/ikawrakow/ik_llama.cpp/pull/516) and [PR524](https://github.com/ikawrakow/ik_llama.cpp/pull/524). Keep in mind that `iq3_xxs` is not the exact same implementation as `iq3_xxs_r4` and PR516 even says the new `iq3_xxs` is about 2x faster than `iq3_xxs_r4` given ik's specific llama-sweep-bench testing.

Anyway, you get the idea. So in conclusion my basic approach is:

1. Think about hardware break points for both GPU VRAM and CPU RAM.
2. Decide if the model will likely be 100% on GPU or hybrid inferencing (or even CPU only).
3. Test the biggest size quants that will allow me to hit my breakpoint targets for good quality.
4. Test out some of them and compare against baseline `q4_0` and `q8_0` versions to see what is faster and lower perplexity KLD as well.
5. Scale up with a larger model size and see if it works like I want.
6. Iterate

Also I'm very happy to fail. I've made many more quants that never saw the light of day than those that I upload to hugging face. Failure is half the fun. xD

Cheers!

---

👤 **Thireus** commented on **2025-06-15** at **23:11:20**

Thank you for the tips!

> I pick a small model of similar architecture and make a bunch of quants. Then test them with llama-sweep-bench to empirically discover which ones are faster

This! That was indeed going to be my next step. But I'm still very surprised to hear that there is not "general" universal quant benchmark, at least for CPU AVX-2 to give us an idea of what speed to expect for each quant. My assumption here is that it doesn't exist because would be vastly inaccurate and strongly dependent of config type... but I still find it surprising to be honest.

Would you know a model that uses the same arch as DeepSeek R1-0528 that is relatively small?

I just ran some benchmarks on: https://huggingface.co/Thireus/DeepSeek-R1-0528-CODER-DRAFT-0.6B-v1.0-BF16-GGUF

Here are the results:
```
GGUF Quant	PP (t/s) with llama-sweep-bench -c 512
q8_0	4936.13
q4_1	4610.79
iq4_nl	4604.94
q4_0	4568.61
q5_0	4473.73
q5_1	4347.12
q6_0	4334.24
iq3_xxs	4084.95
iq2_ks	3977.56
iq3_s	3908.1
iq4_xs	3890.67
iq1_bn	3884.23
iq6_k	3866.31
iq2_bn	3866.19
iq4_ks	3820.21
iq2_k	3803.67
iq3_k	3772.67
iq4_ks_r4	3753.78
iq1_m_r4	3749.02
iq5_ks_r4	3702.12
iq5_ks	3700.79
iq4_k	3628.3
iq5_k	3503.13
iq1_m	3284.37
iq2_k_r4	3202.56
iq3_k_r4	3178.56
iq4_kss	3093.66
bf16	3051.4
iq4_k_r4	3036.16
iq5_k_r4	2988.56
f32	2206.25
q8_k_r8	2197.11
q4_k_r4	2040.04
f16	1950.84
q2_k_r4	1886.5
q5_k_r4	1880.66
iq4_xs_r8	1764.99
q6_k_r4	1753
q3_k_r4	1725.95
iq2_xs_r4	1584.74
iq3_s_r4	1573.65
iq2_xxs_r4	1468.21
iq3_xxs_r4	1447.08
iq2_bn_r4	1362.26
q4_0_r8	1291.37
q5_0_r4	1050.08
q8_0_r8	1006.06
q6_0_r4	996.71
iq4_nl_r4	959.81
iq2_xxs	54.81
iq1_s	49.16
iq1_s_r4	44.45
iq2_xs	40.78
iq2_s	38.96
bf16_r16	DID NOT RUN
iq2_kt	DID NOT RUN
iq3_kt	DID NOT RUN
iq4_kt	DID NOT RUN
iq2_m	DID NOT QUANTIZE
iq2_m_r4	DID NOT QUANTIZE
iq3_kl	DID NOT QUANTIZE
iq3_m	DID NOT QUANTIZE
iq3_xs	DID NOT QUANTIZE
q2_k_s	DID NOT QUANTIZE
q3_k_l	DID NOT QUANTIZE
q3_k_m	DID NOT QUANTIZE
q3_k_s	DID NOT QUANTIZE
q4_0_4_4	DID NOT QUANTIZE
q4_0_4_8	DID NOT QUANTIZE
q4_0_8_8	DID NOT QUANTIZE
q4_k_m	DID NOT QUANTIZE
q4_k_s	DID NOT QUANTIZE
q5_k_m	DID NOT QUANTIZE
q5_k_s	DID NOT QUANTIZE
q8_kv	DID NOT QUANTIZE
q8_kv_r8	DID NOT QUANTIZE
```

I've quantised these layers, and left all the others at q8_0:
```
blk\.([0-9]|1[0-9]|2[0-3])\.ffn_down\.weight=$_quant
blk\.([0-9]|1[0-9]|2[0-3])\.ffn_gate\.weight=$_quant
blk\.([0-9]|1[0-9]|2[0-3])\.ffn_norm\.weight=$_quant
blk\.([0-9]|1[0-9]|2[0-3])\.ffn_up\.weight=$_quant
```

Basically I should avoid any quant below f32 from the bench results table above. But then there is `iq1_s_r4` which should have maybe been higher up in the list... but I suppose this is because the model's architecture is not the same...

---

👤 **ubergarm** commented on **2025-06-16** at **02:13:33**

> Would you know a model that uses the same arch as DeepSeek R1-0528 that is relatively small?

Yeah ik and folks use [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) which is ~16B MoE 2.4B active.

> Here are the results:

Oh interesting, you made a lot of quants of that little 0.6B, very cool! Is this running on all layers offloaded on a single CUDA device with `--threads 1`? The `_r4` variants were mainly for CPU inferencing and didn't even work on CUDA [until a few weeks ago since PR461](https://github.com/ikawrakow/ik_llama.cpp/pull/461).

For DeepSeek-V2 architechture (R1-0528 etc) my strategy is:
1. Keep all `attn/shexp` ready to run fully offloaded on GPU (iq5_ks is one of the best [in my experiments](https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13336629) in terms of speed/accuracy trade-offs). If someone wants to run pure-CPU, they can use `-rtr` or manually repack them to `_r4` for CPU optimizations.
2. I'm not for sure on `attn_k_b` but due to its shape you're restricted to like `q4_0` or `q6_0` etc. I believe it is technically redundant and I'm not for sure if it is possible to prune it or the corresponding `attn_` layers with the same data. More or less I keep it around the same BPW as my other `attn` tensors.
3. Keep all routed experts `-ot exps=CPU` as `_r4` variants assuming people will use hybrid inferencing with these layers on CPU/RAM. Originally when I did this, people could *not* add a few more layers onto GPU to fill up VRAM until ik bailed me out with the more recent PRs as mentioned. In the most ideal system customized to your exact hardware you'd calculate how many extra layers fit into your VRAM and quantize those as non `_r4` varieties leaving the remainder as `_r4`. This level of customization not practical for general purpose release to public huggingface though imo.
4. output.weight is also sometimes called "head" and often left at ~6bpw as it is not repeating. Seems like q6_K is fairly common, or iq6_k, or heck I'll leave it iq5_ks just to keep things consistent with my other tensors.
5. token_embd.weight is also not repeating and can be kept similar slightly higher BPW.

Hope that sheds some more light on things.

---

👤 **ikawrakow** commented on **2025-06-16** at **10:18:19**

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

👤 **Thireus** commented on **2025-06-16** at **11:36:12**

Thank you @ubergarm and @ikawrakow - I'll switch to DeepSeek-V2-Lite so it can be a better representation of R1-0528

The measurements I took were with partial offloading and latest ik_llama build. So I get a mix of GPU and CPU. But indeed those are not the speed of each quant, rather it gives an indication of which quant will slow down the overall speed perfs when used in a GPU+CPU mix.

```
for f in $(ls DeepSeek-R1-0528-CODER-DRAFT-0.6B-v1.0-THIREUS-*.gguf); do \
echo $f:;
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 ~/ik_llama-main-b3800-79fc7dd-bin-win-cuda-12.8-x64/llama-sweep-bench -m $f  -mla 3 -fa \
  -amb 1024 \
  -ctk f16 \
  -c 512 \
  -ngl 99 \
  -ot "blk\.(3|4|5|6)\.ffn_.*=CUDA0" -ot "blk\..*\.ffn_.*=CPU" \
  -b 4096 -ub 512 \
  --threads 36 \
  --main-gpu 0 2>&1 | grep "|"; \
done
```

My current strategy remains to save as much time as possible in this quest of producing the most optimised GGUF for my hardware. So, anything that removes the human factor to perform any pre-assessment of which quants to use or not based on the hardware/model_architecture/quant_theory would help.

I'm currently sitting on the Bruteforce method below:

# Bruteforce method - Effort: Minimal (measured in hours) - Full automation with scripts - Drawback: Limited knowledge gain

1. Loop through all quants and produce speed perf metrics for a small model with similar architecture for specific hardware that will run the LLM
2. Triage results - blacklist quants with poor perfs
3. Identify best quants based on speed and estimated resulting model size
4. Produce variations of the big LLM with these quants and measure the PPL
5. Identify best PPL/Size GGUF variant from resulting metrics

# Smart method - Effort: High (measured in weeks/months) - Full manual - Drawback: Time (which I don't have)

1. Learn about different quant methods (but first, find where this documentation is...)
2. Understand the maths and drawbacks (but first, find where to get this knowledge from...)
3. Dive into the llama code to understand what has been implemented and optimised
4. Understand hardware limitations and optimisations (but first, find where this information is...)
5. Identify the best theoretical quants for the model architecture and specific hardware that will run the LLM
6. Produce GGUF

---

👤 **saood06** commented on **2025-06-16** at **11:41:57**

>     1. Learn about different quant methods (but first, find where this documentation is...)

For each quant type you want to learn more about you can search for it [here](https://github.com/ikawrakow/ik_llama.cpp/pulls). The `README` lists a lot of the newer one's alongside the PR they were introduced but there are often follow-up PRs that increase their speed.

There is a method between the two in which you do the bruteforce method, but then focus your attention on select quants you want to learn more about.

---

👤 **ikawrakow** commented on **2025-06-16** at **11:52:52**

Your brute force method is unlikely to produce  a meaningful outcome. You don't want to just find the quantization type that runs fastest on your hardware, but the quantization mix that runs the fastest **and satisfies a minimum quantization quality requirement**. Because, you know, the absolutely fastest model is the one that does no computation at all.

---

👤 **ubergarm** commented on **2025-06-18** at **20:43:22**

@Thireus 

How you coming along? Things have changed a lot just in the past couple days with the enhanced CPU Prompt Processing in closed `PR531`, `PR533`, `PR534`. 

This seems to create three "tiers" of quant speed for CPU based PP from how I understand it reading `PR534` (specifically for CPUs supporting  `avx2` instructions). Might be useful thing to keep in mind when designing quants for hybrid GPU+CPU inferencing as you're doing with your R1-0528. I'm also experimenting with some ~72B dense models now myself.

Note that all three tiers are very optimized now relative to other forks. So this is mostly a distinction between the groups relative to each other on this fork.

While there is still some variation within each "tier", the easiest way to tell quickly besides pulling up those PRs, is grep the code like so:

<details>

<summary>👈 A Tier</summary>

```bash
$ cd ik_llama.cpp/ggml/src/iqk
$ grep Q8_K_R8 iqk_mul_mat.cpp | grep type
            case GGML_TYPE_IQ2_XXS: return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_XS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_S  : return nrc_y >= 16 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_XXS: return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_XS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_S  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ1_S  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ1_M  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_Q2_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_Q3_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ5_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ5_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ6_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
```

</details>

<details>

<summary>👈 B Tier</summary>

```bash
$ cd ik_llama.cpp/ggml/src/iqk
$ grep Q8_0_R8 iqk_mul_mat.cpp | grep type
            case GGML_TYPE_Q6_K   : return nrc_y >= 64 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_Q4_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_Q5_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_Q6_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ4_NL : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_Q8_0   : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ2_KT : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ3_KT : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ4_KT : return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ2_KT: return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
            case GGML_TYPE_IQ4_KT: return nrc_y >= 32 ? GGML_TYPE_Q8_0_R8 : type;
```

</details>

<details>

<summary>👈 C Tier</summary>

```bash
$ cd ik_llama.cpp/ggml/src/iqk
$ grep Q8_1 iqk_mul_mat.cpp | grep type
            case GGML_TYPE_Q4_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
            case GGML_TYPE_Q5_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
            case GGML_TYPE_Q4_1   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
            case GGML_TYPE_Q5_1   : return nrc_y >= 32 ? GGML_TYPE_Q8_1    : type;
```

</details>

There is more to take into consideration than just PP speed on CPUs with avx2 support of course, like the GPU speeds for offloaded layers, perplexity, overall BPW as TG is generally memory i/o bound, etc. Just wanted to check it with you and also write this up to help my own brain process the changes haha...

Finally no need to sweat it too much. I tested changed `token_embd/output` from the usual `q4_K/q6_K` to `iq4_k/iq6_k` and didn't see significant measurable differences in PPL/speed for just those two in my one test.

Cheers!

*EDIT*:

The biggest upshot here for me is that the `_r4` row interleaved quants are no longer fastest for CPU inference in many situations especially for dense models or where batch sizes are large enough for MoEs.

---

👤 **Thireus** commented on **2025-06-19** at **15:56:45**

Thank you for all the feedback. I am making small progress and I'm working towards a combination of quants that brings high speed (both prompt eval and new tokens) as well as reduced PPL on my hardware. I'm on Intel x299 and there are a lot of quants that really kill the CPU speed (hence my initial high failure rate).

The best model I was able to produce so far in terms of speed while maintaining a fair quality has the following characteristics:
- 214GB in size
- 3.5904 +/- 0.01953 PPL
- 140.62 PP-512 (t/s)
- 6.21 t/s new tokens

I have also found that I need a model that is around 240GB in size max. So I'm currently cooking some quant mixes to achieve this (this is where the gap on the graph is).

![DeepSeek-R1-0528-GGUFs-PPL-02](https://thireus.com/GITHUB/DeepSeek-R1-0528-GGUFs-PPL-02.png)

Once I find the most optimum mix I'll upload the model, including the eval results and the secret recipe.

tl;dr: Still cooking.

---

👤 **saood06** commented on **2025-06-19** at **18:03:41**

> Once I find the most optimum mix I'll upload the model, including the eval results and the secret recipe.

I don't get why they are called "secret recipes", even if not provided if a mix is, you can gguf-dump to get them (even if that is in a more inconvenient way than the custom regex used). 

If you share what your current working mix is then it would allow people to make suggestions on what you might want to change to use the ~26GB of extra budget you have. I have gone through the process you have with a lot of iterations optimizing for performance while maintaining my quality standard within a size budget (although my size budget was higher than yours).

---

👤 **ubergarm** commented on **2025-06-19** at **19:34:19**

> I don't get why they are called "secret recipes"

For myself at least, it is jest as I do my best to make my recipes known, easy to repeat, and provide imatrix data etc. And yes the gguf-dump is very useful. I'm not sure why huggingface throws "bad gguf magic number" for some of my quants but not others, as I like to look at a gguf before downloading it sometimes.

Anyway, thanks as always for sharing all of your experience and guidance, you are very generous.

Regarding "extra 26GB of budget" type stuff, I still wonder what the best way to add a little more fat to an otherwise fairly homogeneous quant. For example, using the normal pattern of ffn_down slightly larger than ffn_(gate|up) will hit a given size for a given quant type. If you want just a little more, is it best to increase like the first 8 layers one size? Then maybe the last few layers a little bigger? I've seen this done in some discussions, but even with the layer-similarity score I'm not sure how best to vary some layers over other layers other than lots of trial and error.

Thanks!

---

👤 **saood06** commented on **2025-06-19** at **19:52:26**

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

👤 **Thireus** commented on **2025-06-28** at **15:57:07**

Just wanted to share that I haven't given up, in fact I have made my first breakthrough today after a week of bruteforcing and auto-analysis to find the optimum quant combination, which allowed me to cook the following dynamic quant today:

- 236GB in size
- 3.3919 +/- 0.01826 PPL
- 110.45 PP-512 (t/s)
- 4.97 t/s new tokens

![DeepSeek-R1-0528-GGUFs-PPL-03.png](https://thireus.com/GITHUB/DeepSeek-R1-0528-GGUFs-PPL-03.png)

I still need ~ 2 weeks worth of computing to achieve better results in speed and quality than the above. Then, I plan to share the methodology, scripts and quants.

---

👤 **ubergarm** commented on **2025-06-28** at **16:31:22**

@Thireus 

Thanks for the report! You're exploring the heck out of that inflection "knee point" between 200 and 300 GiB and cool to see the updated plot.

Keep up the good work, and keep in mind it is somewhat of a moving target with recent PRs like 559 which have made `iq4_k` faster than `iq4_ks` when offloaded onto CUDA for PP at least on my test rig.

Looking back I'd definitely change a few things on my quants like probably standardize using `q4_K` or `iq4_k` for token_embd and `q6_K` or `iq6_k` for final output. Also maybe tweak the first 3 `ffn` just a touch etc. Always something to tweak and tinker with which keeps this hobby interesting lol...

Cheers!

---

👤 **Thireus** commented on **2025-07-02** at **22:20:22**

Yes, I keep feeding the new quants to my automated scripts as soon as they are released/improved, so they can ingest them and see if they are of any good use. I've also fed the latest iq3_ks. I've also experimented with _kt.

I've taken a lot of shortcuts (including interpolation of partial metrics and mathematical models based on partial or guessed data) to save time and cost and speed up the quant mix discovery and calibration process. I'm not yet entirely happy about the quality of some scripts nor some algorithms that can still be improved. Nevertheless, I believe the methodology is mature enough to provide near optimum quant mixes, competing against popular quants such as unsloth quants.

I have created a script that can produce optimum mix recipes given a VRAM and RAM GB target. So, I'm happy to report I was able to produce a mixture tonight that fits exactly 240GB which was my target, and fits 99% of my free RAM without incurring any speed loss. The PPL is also the lowest I've achieved so far.

- 240GB in size
- 3.3471 +/- 0.01783 PPL
- 99.68 PP-512 (t/s)
- 5.43 t/s new tokens

Since I run my scripts on partial metrics, full metrics will be available in about 5-6 more days (I had made a mistake in my calibration dataset last week and had to redo all the computation), so there is still a bit of hope that I can reach slightly lower PPL for this size.

In the meantime, here's a zero-shot screensaver created by that mixture of quants which I very much like (part of my own quality check testing, so can't disclose the prompt): https://thireus.com/GITHUB/screensaver.py

---

👤 **Thireus** commented on **2025-07-11** at **11:23:19**

MVP1 published - https://github.com/Thireus/GGUF-Tool-Suite

Example of quant mix recipe available [here](https://github.com/Thireus/GGUF-Tool-Suite/blob/main/recipe_examples/DeepSeek-R1-0528.THIREUS-3.4064bpw-3.3372ppl.242GB-GGUF_11GB-GPU_231GB-CPU.254e1cf_c044584.recipe).

- 3.3372 +/- 0.01781 ppl
- 242GB Total size
- 11GB VRAM
- 231GB RAM
- 113.10 t/s PP eval
- 5.70 t/s eval

Config: 1x 5090 + 2x 3090 + i9 9980xe with 256GB DDR4

Custom recipes can be produced within minutes for different VRAM and RAM requirements, see README file for basic instructions. Article coming soon.