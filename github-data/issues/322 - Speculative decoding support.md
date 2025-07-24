### 📝 [#322](https://github.com/ikawrakow/ik_llama.cpp/issues/322) - Speculative decoding support

| **Author** | `Lissanro` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-04-09 |
| **Updated** | 2025-06-03 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

A while ago a patch to support speculative decoding was merged to llama.cpp:
https://github.com/ggml-org/llama.cpp/pull/10455

I noticed that ik_llama.cpp has --model-draft and --gpu-layers-draft but they do not seem to do anything as far as I can tell (I see no speed up from using a draft model and nothing in the logs about the draft model being loaded), and ik_llama.cpp lacks options from the pull request that implements speculative decoding, like --draft-max, --draft-min, --device-draft and --draft-p-min, possibly some others.

### Motivation

Recently, a draft model specifically for R1 was made: https://huggingface.co/jukofyork/DeepSeek-R1-DRAFT-0.5B-v1.0-GGUF - it would be great if it was possible to use it with ik_llama.cpp. Potentially, it could provide 1.5-2 speed up for inference.

### Possible Implementation

_No response_

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-04-09** at **12:29:17**:<br>

I have never used or looked into speculative decoding, so it would be something new to learn and wrap my head around what needs to get done.

---

👤 **orca-zhang** commented the **2025-04-09** at **14:29:57**:<br>

That's great. I've tried to make a DRAFT model for speculative decoding but failed.

---

👤 **saood06** commented the **2025-04-10** at **03:32:44**:<br>

> I have never used or looked into speculative decoding, so it would be something new to learn and wrap my head around what needs to get done.

The speculative example exists here in ik_llama.cpp, but there are a few functional commits from mainline that we are behind (many commits are just refactorings or non functional tweaks), we also lack the speculative-simple and speculative support in server.

It was something I was interested in syncing after updating the cache_prompt (and maybe even adding some stuff to the API that front ends could benefit from for my usecases)

---

👤 **saood06** commented the **2025-04-10** at **03:32:44**:<br>

> I have never used or looked into speculative decoding, so it would be something new to learn and wrap my head around what needs to get done.

The speculative example exists here in ik_llama.cpp, but there are a few functional commits that are missing (many commits are just refactorings or non functional tweaks), the speculative-simple and speculative support in server are missing.

It was something I was interested in syncing after updating the cache_prompt (and maybe even adding some stuff to the API that front ends could benefit from for my usecases)

---

👤 **orca-zhang** commented the **2025-04-10** at **15:33:41**:<br>

I have tested it on the mainline, using UD-Q2_K_XL + DRAFT_0.5B_BF16 parameters `-ot=exp -ngl99 -ngld 99`. Although it is fast, the output quality is very poor, with almost no useful output. The draft model can run at 120 tokens/s, and the final tg can go from 9.35 -> 11.8 tokens/s, with a memory bandwidth of 608GB/s, 2S 6454s with a single 5080. Of course, it may also be a problem of parameter tuning.

---

👤 **Lissanro** commented the **2025-04-10** at **16:29:34**:<br>

Speculative decoding should have zero impact on quality of output, since this is its the most important feature, to provide performance boost without affecting the quality. At worst, the draft model will not provide any speed up if it is very unlucky at predicting tokens of the main model.

If there is any impact on quality of the output from the main model while using a draft model, it means there is a bug somewhere.

---

👤 **Lissanro** commented the **2025-04-10** at **16:29:34**:<br>

Speculative decoding should have zero impact on quality of output, since this is the most important feature of the speculative decoding, to provide performance boost without affecting the quality. At worst, the draft model will not provide any speed up if it is very unlucky at predicting tokens of the main model.

If there is any impact on quality of the output from the main model while using a draft model, it means there is a bug somewhere.

---

👤 **ikawrakow** commented the **2025-04-10** at **18:19:24**:<br>

Isn't this dependent on how it is implemented? If sampling is done without taking into account tokens predicted by the draft model, then sure, the draft model should not affect quality. But if someone was trying to be clever and somehow incorporate the draft tokens into the sampling (e.g., in order to increase acceptance rate), then it can lead to a disaster. I haven't checked how it is done in `llama.cpp`. But if @orca-zhang observes a much reduced quality of the generated output (I assume with otherwise identical parameters apart from using a draft model?), then either there is a bug, or it is not implemented correctly.

---

👤 **saood06** commented the **2025-06-01** at **07:45:24**:<br>

Interestingly Eagle-2 seems like it may be coming to llama.cpp see https://github.com/ggml-org/llama.cpp/pull/13908. I'm keeping my eye on how easy it would be to add support here once there is a working PR in llama.cpp.

---

👤 **ikawrakow** commented the **2025-06-01** at **09:04:08**:<br>

> Interestingly Eagle-2 seems like it may be coming to llama.cpp see [ggml-org/llama.cpp#13908](https://github.com/ggml-org/llama.cpp/pull/13908). I'm keeping my eye on how easy it would be to add support here once there is a working PR in llama.cpp.

I know you are very interested in getting Eagle-2 here, but I don't find the results they report particularly impressive..

They have run benchmarks on an RTX-4080, which is the GPU I have. I also have Qwen2.5-7B-Instruct handy (is this the model they mean when they say "Qwen2-7B-Instruct"?). With that model in `bf16` (or `f16`) precision and no speculation I get 45 t/s on today's mainline and also with `ik_llama.cpp`. Which would mean a 10% speedup, and not the 35% they report for zero temperature. I guess they compare to mainline speculative implementation, but on my book that comparison is bogus. What they need to compare to is `Max(speculation, no speculation)`. This applies also to the "2.1" speedup, which in reality is just `53/45`, so 18%. If the "baseline" is just 37 t/s, it basically means that the draft model just consumes GPU cycles without resulting in any successful drafts with the current mainline speculative implementation.

---

👤 **saood06** commented the **2025-06-01** at **09:58:50**:<br>

> I know you are very interested in getting Eagle-2 here, but I don't find the results they report particularly impressive..
> 
> They have run benchmarks on an RTX-4080, which is the GPU I have. I also have Qwen2.5-7B-Instruct handy (is this the model they mean when they say "Qwen2-7B-Instruct"?). With that model in `bf16` (or `f16`) precision and no speculation I get 45 t/s on today's mainline and also with `ik_llama.cpp`. Which would mean a 10% speedup, and not the 35% they report for zero temperature. I guess they compare to mainline speculative implementation, but on my book that comparison is bogus. What they need to compare to is `Max(speculation, no speculation)`. This applies also to the "2.1" speedup, which in reality is just `53/45`, so 18%. If the "baseline" is just 37 t/s, it basically means that the draft model just consumes GPU cycles without resulting in any successful drafts with the current mainline speculative implementation.

I didn't pay much attention to their performance results for a few reasons, first they haven't shared code yet, and hopefully aren't indicative of what the future PR allows for if used properly, and most importantly I have no idea why they are using such a large draft model, as that is far from optimal (even for the "naive" speculative implementation in llama.cpp and in here, I'm fairly certain the typical given advice is to use 10x smaller draft or even smaller for larger models [it is more complicated than that as picking the correct quant type matters]). 

~For reference they tested with a 2.7GB draft model as stated in the PR, and looking at available Eagle-3 draft models it is 850 MB for [this](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B/tree/main) 8B model, 1.28 GB for [this](https://huggingface.co/yuhuili/EAGLE3-Vicuna1.3-13B/tree/main) 13B model, and 3.15 GB for [this](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B/tree/main) 70B model. Their draft model is closest in size to the 70B when when they were drafting for a 7B model.~

The official Eagle based implementations perform well see: https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md.

Edit: See the comment below for a direct comparison, and an explanation for why the size differs.

---

👤 **saood06** commented the **2025-06-01** at **09:58:50**:<br>

> I know you are very interested in getting Eagle-2 here, but I don't find the results they report particularly impressive..
> 
> They have run benchmarks on an RTX-4080, which is the GPU I have. I also have Qwen2.5-7B-Instruct handy (is this the model they mean when they say "Qwen2-7B-Instruct"?). With that model in `bf16` (or `f16`) precision and no speculation I get 45 t/s on today's mainline and also with `ik_llama.cpp`. Which would mean a 10% speedup, and not the 35% they report for zero temperature. I guess they compare to mainline speculative implementation, but on my book that comparison is bogus. What they need to compare to is `Max(speculation, no speculation)`. This applies also to the "2.1" speedup, which in reality is just `53/45`, so 18%. If the "baseline" is just 37 t/s, it basically means that the draft model just consumes GPU cycles without resulting in any successful drafts with the current mainline speculative implementation.

I didn't pay much attention to their performance results for a few reasons, first they haven't shared code yet, and hopefully aren't indicative of what the future PR allows for if used properly, and most importantly I have no idea why they are using such a large draft model, as that is far from optimal (even for the "naive" speculative implementation in llama.cpp and in here, I'm fairly certain the typical given advice is to use 10x smaller draft or even smaller for larger models [it is more complicated than that as picking the correct quant type matters]). 

For reference they tested with a 2.7GB draft model as stated in the PR, and looking at available Eagle-3 draft models it is 850 MB for [this](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B/tree/main) 8B model, 1.28 GB for [this](https://huggingface.co/yuhuili/EAGLE3-Vicuna1.3-13B/tree/main) 13B model, and 3.15 GB for [this](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B/tree/main) 70B model. Their draft model is closest in size to the 70B when when they were drafting for a 7B model.

The official Eagle based implementations perform well see: https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md.

---

👤 **pockers21** commented the **2025-06-03** at **08:21:04**:<br>

> > I know you are very interested in getting Eagle-2 here, but I don't find the results they report particularly impressive..
> > They have run benchmarks on an RTX-4080, which is the GPU I have. I also have Qwen2.5-7B-Instruct handy (is this the model they mean when they say "Qwen2-7B-Instruct"?). With that model in `bf16` (or `f16`) precision and no speculation I get 45 t/s on today's mainline and also with `ik_llama.cpp`. Which would mean a 10% speedup, and not the 35% they report for zero temperature. I guess they compare to mainline speculative implementation, but on my book that comparison is bogus. What they need to compare to is `Max(speculation, no speculation)`. This applies also to the "2.1" speedup, which in reality is just `53/45`, so 18%. If the "baseline" is just 37 t/s, it basically means that the draft model just consumes GPU cycles without resulting in any successful drafts with the current mainline speculative implementation.
> 
> I didn't pay much attention to their performance results for a few reasons, first they haven't shared code yet, and hopefully aren't indicative of what the future PR allows for if used properly, and most importantly I have no idea why they are using such a large draft model, as that is far from optimal (even for the "naive" speculative implementation in llama.cpp and in here, I'm fairly certain the typical given advice is to use 10x smaller draft or even smaller for larger models [it is more complicated than that as picking the correct quant type matters]).
> 
> For reference they tested with a 2.7GB draft model as stated in the PR, and looking at available Eagle-3 draft models it is 850 MB for [this](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B/tree/main) 8B model, 1.28 GB for [this](https://huggingface.co/yuhuili/EAGLE3-Vicuna1.3-13B/tree/main) 13B model, and 3.15 GB for [this](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B/tree/main) 70B model. Their draft model is closest in size to the 70B when when they were drafting for a 7B model.
> 
> The official Eagle based implementations perform well see: https://github.com/hemingkx/Spec-Bench/blob/main/Leaderboard.md.

https://huggingface.co/yuhuili/EAGLE-Qwen2-7B-Instruct

This is the EAGLE-2 Qwen2 7B draft model repository, with a model size of 1.6GB.
However, this model doesn't include the lm_head output layer, because in the code implementation, this layer is passed as a parameter at 

https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets1.py#L673C54-L673C58

Since llama.cpp is not as flexible as Python and needs to specify this layer in the computation graph, 
I need to append the lm_head layer from the original Qwen2 7B Instruct model to the end of the draft model before converting it to GGUF format. 
This increases the model size from 1.6GB to 2.7GB. The smaller models you mentioned are EAGLE-3 draft models, not the EAGLE-2 I'm working with here.

---

👤 **saood06** commented the **2025-06-03** at **09:00:43**:<br>

> https://huggingface.co/yuhuili/EAGLE-Qwen2-7B-Instruct
> 
> This is the EAGLE-2 Qwen2 7B draft model repository, with a model size of 1.6GB. However, this model doesn't include the lm_head output layer, because in the code implementation, this layer is passed as a parameter at
> 
> https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets1.py#L673C54-L673C58
> 
> Since llama.cpp is not as flexible as Python and needs to specify this layer in the computation graph, I need to append the lm_head layer from the original Qwen2 7B Instruct model to the end of the draft model before converting it to GGUF format. This increases the model size from 1.6GB to 2.7GB. 

I see, thank you for the info on why the size is different. I've run into situations where mergekit generated safetensors were larger than expected because they added the lm_head tensor and the llama.cpp conversion script would fail (and in those situations the easiest fix was to remove them from the safetensors rather than fix the conversion script to ignore them).

Like I said, I'm (patiently) waiting to see the Phase-2 and Phase-3 submissions before I form any opinions on implementation and performance, I only commented about the size difference I saw since the conversion code and generated files for it are currently shared.

>The smaller models you mentioned are EAGLE-3 draft models, not the EAGLE-2 I'm working with here.

I definitely should have clarified that when I linked the other weights for reference. It's been a while since I've looked into Eagle and I forgot that EAGLE and EAGLE-2 share weights, and they have removed this line from their README ("Compared to EAGLE, EAGLE-2 does not require additional training and uses the same weights.") which would have reminded me, so I decided to reference the newer weights, but the most relevant reference would have been the one you linked. Sorry, that is my mistake, and I have edited my original comment to hopefully prevent anyone from being misled.

---

👤 **saood06** commented the **2025-06-03** at **09:00:43**:<br>

> https://huggingface.co/yuhuili/EAGLE-Qwen2-7B-Instruct
> 
> This is the EAGLE-2 Qwen2 7B draft model repository, with a model size of 1.6GB. However, this model doesn't include the lm_head output layer, because in the code implementation, this layer is passed as a parameter at
> 
> https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets1.py#L673C54-L673C58
> 
> Since llama.cpp is not as flexible as Python and needs to specify this layer in the computation graph, I need to append the lm_head layer from the original Qwen2 7B Instruct model to the end of the draft model before converting it to GGUF format. This increases the model size from 1.6GB to 2.7GB. 

I see, thank you for the info on why the size is different. I've run into situations where mergekit generated safetensors were larger than expected because they added the lm_head tensor and the llama.cpp conversion script would fail (and in those situations the easiest fix was to remove them from the safetensors).

Like I said, I'm (patiently) waiting to see the Phase-2 and Phase-3 submissions before I form any opinions on implementation and performance, I only commented about the size difference I saw since the conversion code and generated files for it where shared.

>The smaller models you mentioned are EAGLE-3 draft models, not the EAGLE-2 I'm working with here.

I definitely should have clarified that when I linked the other weights for reference. It's been a while since I've looked into Eagle and I forgot that EAGLE and EAGLE-2 share weights, and they have removed this line from their README ("Compared to EAGLE, EAGLE-2 does not require additional training and uses the same weights.") which would have reminded me, so I decided to reference the newer weights, but the most relevant reference would have been the one you linked. Sorry, that is my mistake.