### 🔀 [#609](https://github.com/ikawrakow/ik_llama.cpp/pull/609) - Added kimi-k2 support (ported from llama.cpp)

| **Author** | `anikifoss` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-14 |
| **Updated** | 2025-07-15 |

---

#### Description

Ported kimi-k2 support from llama.cpp.

[Original patch](https://github.com/ggml-org/llama.cpp/pull/14654) by @gabriellarson

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [x] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **anikifoss** commented the **2025-07-14** at **16:40:34**:<br>

I see this warning when loading the model `Your prompt processing speed will be crippled`, and it appears to be true: the PP speed is indeed crippled.

---

👤 **anikifoss** commented the **2025-07-14** at **16:41:44**:<br>

I haven't ported the python changes yet, just getting ik_llama to load the model.

---

👤 **ikawrakow** submitted a review the **2025-07-14** at **16:43:15**: ✅ `APPROVED`<br>

LGTM.

---

👤 **anikifoss** commented the **2025-07-14** at **16:44:11**:<br>

@ikawrakow sorry, I forgot to mark this as a draft. Still waiting for llama.cpp branch to merge...

---

👤 **ubergarm** commented the **2025-07-14** at **16:45:01**:<br>

@anikifoss 

Okay yeah I was thinking this might happen as I'd seen it trying to use the "mainline method" instead of the OG fairydreaming evshiron method to preserve the tensors. Yeah that warning is because the "mainline method" handles some MLA tensors differently. I always use the evshiron method for my ik specific quants.

So might need to look into the differences in what you have ported and with https://github.com/evshiron/llama.cpp

@saood06  and I have been discussing it'd be great to get this all into ik's fork.

---

👤 **anikifoss** commented the **2025-07-14** at **16:45:01**:<br>

I'll open a follow up PR to bring any changes as well as port the python script support.

---

👤 **anikifoss** commented the **2025-07-14** at **16:58:37**:<br>

> Use this PR (now merged into main) to convert my bf16 safetensors to bf16 GGUF to test the code a little more lol

The conversion code is currently missing (this was a draft PR, I did not expect it to get merged so fast)

---

👤 **ubergarm** commented the **2025-07-14** at **17:07:37**:<br>

It'd sure be interesting if someone released an Kimi-K2-Instruct-1000B-A32B-IQ2_KL...

---

👤 **whatever1983** commented the **2025-07-14** at **19:53:14**:<br>

yo, guys, seriously, just had to comment on this model on two fronts:

First, the model is just 1Trillion, and you already have to deal with 2TB BF16 files.  Either you look at DFloat11 format and compress the matissa to 11.2bpw perfectly.  If not only for ssd savings.   I was begging ik to consider working with FP8/FP4 formats in another thread and got rejected.  Why go through the FP8->  2TB BF16 safetensors with triton-cpu -> q8_0 loss->requantize to 2-3bits, when FP4 checkpoints are out there @ 580GB k-l-lambda/Kimi-K2-Instruct-FP4 or baseten/Kimi-K2-Instruct-FP4?  I know it is a lot to implement for FP8/FP4. vllm already has a marlin FP4 kernel. SGlang has a petit-nvfp4 WIP kernel for ROCm.  What's missing is CPU based NVFP4/FP8 inferencing using bf16 recast.  Really, you work with 580GB of weights already done for you. 

Second comment is for the Kimi K2 model itself.  If you haven't read the README,  it is only 51 SWE-Bench Verified for non-agent, below R1-0528's 57points.  65 for single agent, but then you have to use tooling, which includes bash. ("Kimi K2 achieves 65.8% pass@1 on the SWE-bench Verified tests with bash/editor tools"  So if you want a SWE-bench 8 points higher than R1-0528, you have to expose your bash prompt.  Who knows what the bash prompt is calling HTTPS API endpoints, posting your data to which API endpoints?  It is such a security risk, are you going to sandbox your bash execution?  All I can speculate is that you could theoretically call the Anthropic API point to fudge the benchmark.  Then there is the 71 points for multiagent SWE-bench(aka cons=32 or 64).  Good luck running 10toks/sec on a 768GB DDR5 EPYC @ cons=64.   You could sleep all night and come back in the morning for a cons64 job.

Not that impressive 1Trillion model if you care about data security or claimed performance.  I suggest that you just either wait for OpenAI's open source model, which calls O3 via HTTP, or just pay 30dollars/month for grok4-coder cons=1 at SWE-bench=72.

---

👤 **ubergarm** commented the **2025-07-14** at **20:15:55**:<br>

@whatever1983  

>  I suggest that you just either wait for OpenAI's open source model, which calls O3 via HTTP, or just pay 30dollars/month for grok4-coder cons=1 at SWE-bench=72.

But where is the fun in that? ;p  And besides, I generally don't use LLMs I just enjoy making them go brrr....

---

👤 **anikifoss** commented the **2025-07-14** at **21:02:47**:<br>

> Never heard the term "agentic lean" before.

Sorry, that sounds like something a tech bro would say. Perhaps I was primed somehow :sweat_smile:. Just sharing my thoughts that these models were both trained for agentic use-cases, so they may share simlar tendencies.

---

👤 **saood06** commented the **2025-07-14** at **21:07:19**:<br>

> > Never heard the term "agentic lean" before.
> 
> Sorry, that sounds like something a tech bro would say. Perhaps I was primed somehow 😅. 

Not calling you out, just was new vocabulary for me.

>Just sharing my thoughts that these models were both trained for agentic use-cases, so they may share simlar tendencies.

That does make sense. I do appreciate your thoughts, no need to apologize.

---

👤 **saood06** commented the **2025-07-14** at **23:10:34**:<br>

> BeaverAIClub

Is that a discord?

> So probably gonna need something around here: https://github.com/ikawrakow/ik_llama.cpp/blob/main/src/llama.cpp#L23236-L23259 for the chat completions endpoint to detect it and apply it on the server side...

I never connected the dots that the chat completion endpoint needs that (probably because I prefer and almost always use the standard completion endpoint). Thanks.

---

👤 **ubergarm** commented the **2025-07-15** at **02:46:33**:<br>

@anikifoss 

I finally think I'm out of the woods with the convert script... My tmux was dying which would end the process, had to run it in a nohup lol... I think its `tqdm` progress bar messing with my terminal or something :crossed_fingers: 

Anyway, in the mean time I pushed a branch, but want to test it is working with a quant. I also added what I think will be the chat template which also needs testing. I could open a draft PR I suppose at least to have a place holder...

https://github.com/ubergarm/ik_llama.cpp/tree/ug/convert-kimi-k2

One step closer!