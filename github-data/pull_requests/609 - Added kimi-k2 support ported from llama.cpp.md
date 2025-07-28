## 🔀 [Pull Request #609](https://github.com/ikawrakow/ik_llama.cpp/pull/609) - Added kimi-k2 support (ported from llama.cpp)

| **Author** | `anikifoss` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `kimi-k2-support` |
| **Target Branch** | `main` |
| **Created** | 2025-07-14 |
| **Updated** | 2025-07-15 |
| **Merged** | 2025-07-14 |

---

## 📄 Description

Ported kimi-k2 support from llama.cpp.

[Original patch](https://github.com/ggml-org/llama.cpp/pull/14654) by @gabriellarson

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [x] Medium
  - [ ] High

---

## 💬 Conversation

👤 **anikifoss** commented on **2025-07-14** at **16:40:34**

I see this warning when loading the model `Your prompt processing speed will be crippled`, and it appears to be true: the PP speed is indeed crippled.

---

👤 **ubergarm** commented on **2025-07-14** at **16:40:38**

@anikifoss 

Thanks for using your resources (both CPU and BRAIN) for hacking on this behemoth model!

I've successfully used the mainline PR version to convert_hf_to_gguf.py the bf16 safetensors created by fp8_cast_to_bf16.py deepseek script and the resulting Q8_0 seems to be working.

I'll try to use this PR on the same bf16 safetensors, and hope that the MLA stuff works out and that I don't get that `missing wkv_b tensor(s) hanging MLA from to 1` warning. Let me know if you have any luck getting `-mla 3` going on ik's fork! Hope to try it myslf today.

---

👤 **anikifoss** commented on **2025-07-14** at **16:41:44**

I haven't ported the python changes yet, just getting ik_llama to load the model.

---

👤 **ikawrakow** approved this pull request ✅ on **2025-07-14** at **16:43:15**

LGTM.

---

👤 **anikifoss** commented on **2025-07-14** at **16:44:11**

@ikawrakow sorry, I forgot to mark this as a draft. Still waiting for llama.cpp branch to merge...

---

👤 **anikifoss** commented on **2025-07-14** at **16:45:01**

I'll open a follow up PR to bring any changes as well as port the python script support.

---

👤 **ubergarm** commented on **2025-07-14** at **16:45:01**

@anikifoss 

Okay yeah I was thinking this might happen as I'd seen it trying to use the "mainline method" instead of the OG fairydreaming evshiron method to preserve the tensors. Yeah that warning is because the "mainline method" handles some MLA tensors differently. I always use the evshiron method for my ik specific quants.

So might need to look into the differences in what you have ported and with https://github.com/evshiron/llama.cpp

@saood06  and I have been discussing it'd be great to get this all into ik's fork.

---

👤 **anikifoss** commented on **2025-07-14** at **16:47:14**

@ubergarm I used unsloth's BF16 safetensors and then converted that to GGUF using llama.cpp, so I skipped the step that gives you the `missing wkv_b tensor(s) hanging MLA from to 1` warning.

I quantized using unpatched ik_llama, and it seems to be working.

---

👤 **ubergarm** commented on **2025-07-14** at **16:57:03**

> I quantized using unpatched ik_llama, and it seems to be working.

Okay, then I think my path forward looks something like:

1. Use this PR (now merged into main) to convert my bf16 safetensors to bf16 GGUF to test the code a little more lol
2. use ik_llama.cpp to quantize a Q8_0
3. confirm this Q8_0 is happy and no complaints about `missing wkv_b tensor(s)`
4. use ik_llama.cpp to generate an imatrix.dat
5. test out some mixes and relase some ik quants!

---

👤 **anikifoss** commented on **2025-07-14** at **16:58:37**

> Use this PR (now merged into main) to convert my bf16 safetensors to bf16 GGUF to test the code a little more lol

The conversion code is currently missing (this was a draft PR, I did not expect it to get merged so fast)

---

👤 **ubergarm** commented on **2025-07-14** at **17:01:21**

Ahh okie, things are indeed moving fast. I'm reading up on some more clues from ik [here](https://github.com/ikawrakow/ik_llama.cpp/issues/601#issuecomment-3070185792) so it might be okay. 

I'll just use my existing bf16 GGUF then and try it out on ik_llama.cpp and confirm the default behavior is `-mla 1` for imatrix.

Exciting monday lol :sweat_smile:

---

👤 **ikawrakow** commented on **2025-07-14** at **17:06:37**

> @ikawrakow sorry, I forgot to mark this as a draft. Still waiting for llama.cpp branch to merge...

It's OK. You can make a separate PR for the Python stuff. In the meantime if someone is really desperate to try the model with `ik_llama.cpp`, they can do it with a GGUF that has been created with mainline.

---

👤 **ubergarm** commented on **2025-07-14** at **17:07:37**

It'd sure be interesting if someone released an Kimi-K2-Instruct-1000B-A32B-IQ2_KL...

---

👤 **anikifoss** commented on **2025-07-14** at **17:11:54**

> It'd sure be interesting if someone released an Kimi-K2-Instruct-1000B-A32B-IQ2_KL...

That is YOUR job :sweat_smile: ... I'm sticking to q4+ quants with no imatrix. But not many have enough RAM to run those. My system is using 690G with the DQ4_K quant.

---

👤 **ubergarm** commented on **2025-07-14** at **19:44:28**

So yeah I tested this PR too using a "mainline style" Q8_0 i cooked and it is running at least single inference:

```
>>> User:

Count from 1 to 10 in French.

>>> Assistant:

1. un
2. deux
3. trois
4. quatre
5. cinq
6. six
7. sept
8. huit
9. neuf
10. dix
```

Despite quantizing my bf16 GGUF with ik_llama.cpp it still throws that warning, so there are some important details happening differntly in the convert_hf_to_gguf.py between [ik_llama.cpp's version](https://github.com/ikawrakow/ik_llama.cpp/blob/main/convert_hf_to_gguf.py#L3462-L3484) and [mainline's verison](https://github.com/gabriellarson/llama.cpp/blob/kimi-k2/convert_hf_to_gguf.py#L5705-L5725)

So I'm fussing to see if I can merge in just the changes needed from gabriellarson/llama.cpp/tree/kimi-k2 without messing up the MLA tensors so they stay the OG way... Then I will have a bf16 GGUF with the OG style MLA tensors and can go forward like normal haha...

---

👤 **anikifoss** commented on **2025-07-14** at **19:50:30**

@ubergarm I see the following message when running with ik_llama, is this the same issues you are looking at?
```
============ llm_prepare_mla: need to compute 61 wkv_b tensors
Computed blk.0.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.1.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.2.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.3.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.4.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.5.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.6.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.7.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.8.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.9.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.10.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.11.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.12.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.13.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.14.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.15.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.16.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.17.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.18.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.19.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.20.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.21.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.22.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.23.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.24.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.25.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.26.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.27.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.28.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.29.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.30.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.31.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.32.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.33.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.34.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.35.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.36.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.37.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.38.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.39.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.40.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.41.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.42.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.43.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.44.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.45.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.46.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.47.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.48.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.49.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.50.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.51.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.52.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.53.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.54.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.55.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.56.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.57.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.58.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.59.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
Computed blk.60.attn_kv_b.weight as 512 x 16384 and stored in buffer CUDA0
```

---

👤 **whatever1983** commented on **2025-07-14** at **19:53:14**

yo, guys, seriously, just had to comment on this model on two fronts:

First, the model is just 1Trillion, and you already have to deal with 2TB BF16 files.  Either you look at DFloat11 format and compress the matissa to 11.2bpw perfectly.  If not only for ssd savings.   I was begging ik to consider working with FP8/FP4 formats in another thread and got rejected.  Why go through the FP8->  2TB BF16 safetensors with triton-cpu -> q8_0 loss->requantize to 2-3bits, when FP4 checkpoints are out there @ 580GB k-l-lambda/Kimi-K2-Instruct-FP4 or baseten/Kimi-K2-Instruct-FP4?  I know it is a lot to implement for FP8/FP4. vllm already has a marlin FP4 kernel. SGlang has a petit-nvfp4 WIP kernel for ROCm.  What's missing is CPU based NVFP4/FP8 inferencing using bf16 recast.  Really, you work with 580GB of weights already done for you. 

Second comment is for the Kimi K2 model itself.  If you haven't read the README,  it is only 51 SWE-Bench Verified for non-agent, below R1-0528's 57points.  65 for single agent, but then you have to use tooling, which includes bash. ("Kimi K2 achieves 65.8% pass@1 on the SWE-bench Verified tests with bash/editor tools"  So if you want a SWE-bench 8 points higher than R1-0528, you have to expose your bash prompt.  Who knows what the bash prompt is calling HTTPS API endpoints, posting your data to which API endpoints?  It is such a security risk, are you going to sandbox your bash execution?  All I can speculate is that you could theoretically call the Anthropic API point to fudge the benchmark.  Then there is the 71 points for multiagent SWE-bench(aka cons=32 or 64).  Good luck running 10toks/sec on a 768GB DDR5 EPYC @ cons=64.   You could sleep all night and come back in the morning for a cons64 job.

Not that impressive 1Trillion model if you care about data security or claimed performance.  I suggest that you just either wait for OpenAI's open source model, which calls O3 via HTTP, or just pay 30dollars/month for grok4-coder cons=1 at SWE-bench=72.

---

👤 **saood06** commented on **2025-07-14** at **19:55:08**

> So I'm fussing to see if I can merge in just the changes needed from gabriellarson/llama.cpp/tree/kimi-k2 without messing up the MLA tensors so they stay the OG way... Then I will have a bf16 GGUF with the OG style MLA tensors and can go forward like normal haha...

Like I said on HF, if you take the ~2 TB BF16 safetensor you made, then you can just use the `ik_llama.cpp` convert script (with the kimi changes) and it should give you a GGUF with the MLA tensors you want.

---

👤 **ubergarm** commented on **2025-07-14** at **20:05:16**

@anikifoss 

I think I got it going now: https://github.com/ikawrakow/ik_llama.cpp/issues/601#issuecomment-3070800462

You'll have to download the ~1TB FP8 yourself and fp8_cast_bf16 them like I show in that hf repo discussion. And if my current test works, I'll open a PR with with the updated ik_llama.cpp convert_hf_to_gguf.py including the Kimi-K2 fixes. (or i could upload the 2TB bf16 with the correct MLA tensors, but would have to check if that is okay with the uplink ata first... haha... :sweat_smile: )

If you start with unsloth's bf16 they already have the mainline MLA stuff done to them.

---

👤 **ubergarm** commented on **2025-07-14** at **20:15:55**

@whatever1983  

>  I suggest that you just either wait for OpenAI's open source model, which calls O3 via HTTP, or just pay 30dollars/month for grok4-coder cons=1 at SWE-bench=72.

But where is the fun in that? ;p  And besides, I generally don't use LLMs I just enjoy making them go brrr....

---

👤 **anikifoss** commented on **2025-07-14** at **20:18:24**

Do we feed the trolls? :thinking:

---

👤 **anikifoss** commented on **2025-07-14** at **20:24:52**

Kimi-K2 has amazing VRAM savings, I can load the full 131k context!

I am **over the moon** with this model :new_moon_with_face:

---

👤 **saood06** commented on **2025-07-14** at **20:30:20**

> I am **over the moon** with this model 🌚

I haven't tried the model at all, but I have heard mixed feedback about it.

If you don't mind, how prone to refusals is it? That's the one area I'm most curious about (and will probably affect when/whether or not I end up trying the model locally).

---

👤 **whatever1983** commented on **2025-07-14** at **20:36:42**

@anikifoss:

Why do you call me a troll?  That's just not nice.  I am realistic.  What's the point of running DQ4KM at 690GB or IQ2K IQ3K levels further dropping SWE-bench, if you use it for real work?  It took me about a year messing with GGUF to realize that the GGUF format, even with IK's superb IQK quants is such a toy for client side home production, and I am forced to move to original FP4 safetensors format instead or just pay for the top tier models.   GGUF got started too early.  There's a BF16-> IQ6K compression saving, even at FP8->IQ6K.  The compression just disappears when GB200 trains FP4 models natively, no one is dumb enough to run FP4 trained/compressed model at IQ6K.

---

👤 **anikifoss** commented on **2025-07-14** at **20:36:46**

> If you don't mind, how prone to refusals is it?

Thanks I'll keep an eye on it. But so far it's been amazing at answering my usual benchmark questions. I'll try my goto roo-code project to see how well it does.

In terms of refusal, I noticed devstall-small-24b was refusing some of my suggestions. I suspect it's related to agentic lean, when models are taught to avoid uncertain actions to prevent getting into the weeds. Since Kimi-K2 is mainly developed for agentic use, it may have similar tendencies.

---

👤 **saood06** commented on **2025-07-14** at **20:55:05**

> Thanks I'll keep an eye on it. But so far it's been amazing at answering my usual benchmark questions. I'll try my goto roo-code project to see how well it does.
> 
> In terms of refusal, I noticed devstall-small-24b was refusing some of my suggestions. I suspect it's related to agentic lean, when models are taught to avoid uncertain actions to prevent getting into the weeds.

Never heard the term "agentic lean" before. 

If you are just using it for coding tasks, then I'm not sure you will hit the refusals I care about. It's not even the refusals I care about as bypassing them is rather trivial, but their existence and prevalence tend to correlate with training decisions which impact downstream quality which is what I care about. (Never refusing like abliterated models leads to worse quality from what I've seen, just like a model that refuses too often).

---

👤 **anikifoss** commented on **2025-07-14** at **21:02:47**

> Never heard the term "agentic lean" before.

Sorry, that sounds like something a tech bro would say. Perhaps I was primed somehow :sweat_smile:. Just sharing my thoughts that these models were both trained for agentic use-cases, so they may share simlar tendencies.

---

👤 **saood06** commented on **2025-07-14** at **21:07:19**

> > Never heard the term "agentic lean" before.
> 
> Sorry, that sounds like something a tech bro would say. Perhaps I was primed somehow 😅. 

Not calling you out, just was new vocabulary for me.

>Just sharing my thoughts that these models were both trained for agentic use-cases, so they may share simlar tendencies.

That does make sense. I do appreciate your thoughts, no need to apologize.

---

👤 **ubergarm** commented on **2025-07-14** at **22:48:17**

@anikifoss 

sorry i'm taking so long, still testing my convert_hf_to_gguf.py is working, its taking a while i had to restart for hardware stuff, hah... it is just the mainline changes for kimidev applied to the existing ik_llama.cpp fork's convert_hf_to_gguf.py - no need for the evshiron fork technically (though it is convenient to save a step and disk space, but outside this scope for me).

the mainline PR is still having some discussion, and from i heard in BeaverAIClub the chat template looks like this (with no newlines) (credit tofumagnate for this info) from converting the official template: https://huggingface.co/moonshotai/Kimi-K2-Base/blob/main/tokenizer_config.json#L154

```
<|im_system|>system<|im_middle|>example system prompt<|im_end|><|im_user|>user<|im_middle|>example user turn 1<|im_end|><|im_assistant|>assistant<|im_middle|>example assistant turn 1<|im_end|><|im_user|>user<|im_middle|>example user turn 2<|im_end|><|im_assistant|>assistant<|im_middle|>
```

So probably gonna need something around here: https://github.com/ikawrakow/ik_llama.cpp/blob/main/src/llama.cpp#L23236-L23259 for the chat completions endpoint to detect it and apply it on the server side...

*UPDATE*
The convert is getting close, over 80% I kept having tmux explode on me and then ran the process in `nohup` and its going like a champ (knock on wood). Random aside my rsync --progress had been doing the same thing with tmux panes suddenly closing and nothing in dmesg and no ram errors etc. Anyway, i gotta be careful of how I pipe tqdm progress bar style output with my terminal i guess maybe hopefully lol...

Anyway, if thing thing finishes finally I can get a Q8_0 that *should* not have the warning on this fork! What a day lol

---

👤 **saood06** commented on **2025-07-14** at **23:10:34**

> BeaverAIClub

Is that a discord?

> So probably gonna need something around here: https://github.com/ikawrakow/ik_llama.cpp/blob/main/src/llama.cpp#L23236-L23259 for the chat completions endpoint to detect it and apply it on the server side...

I never connected the dots that the chat completion endpoint needs that (probably because I prefer and almost always use the standard completion endpoint). Thanks.

---

👤 **ubergarm** commented on **2025-07-15** at **02:46:33**

@anikifoss 

I finally think I'm out of the woods with the convert script... My tmux was dying which would end the process, had to run it in a nohup lol... I think its `tqdm` progress bar messing with my terminal or something :crossed_fingers: 

Anyway, in the mean time I pushed a branch, but want to test it is working with a quant. I also added what I think will be the chat template which also needs testing. I could open a draft PR I suppose at least to have a place holder...

https://github.com/ubergarm/ik_llama.cpp/tree/ug/convert-kimi-k2

One step closer!

*UPDATE*: Went ahead and opened a draft PR https://github.com/ikawrakow/ik_llama.cpp/pull/612