### ✨ [#561](https://github.com/ikawrakow/ik_llama.cpp/issues/561) - Feature Request: Tencent Hunyuan-A13B model support

| **Author** | `Downtown-Case` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-27 |
| **Updated** | 2025-07-12 |

---

#### Description

80B/13B active MoE, good benchmarks. Seems right up ik_llama.cpp's alley, aka expert offloading like deepseek.

Uses a custom architecture with good old GQA and NTK rope scaling. At a glance it doesn't look like anything too exotic: https://huggingface.co/tencent/Hunyuan-A13B-Instruct/tree/main

Relevant main llama.cpp issue: https://github.com/ggml-org/llama.cpp/issues/14415

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-06-27** at **21:09:18**:<br>

I took a look at mainline's PR and it isn't quite working there yet. 

Here llama-server is bombing out a little earlier on the bf16 with:
```
llama_model_loader: - type  f32:  161 tensors
llama_model_loader: - type bf16:  321 tensors
llama_model_load: error loading model: error loading model vocabulary: invalid character
llama_load_model_from_file: failed to load model
```

I'll look at it again this weekend if I have some time.

---

👤 **saood06** commented the **2025-06-27** at **21:38:02**:<br>

>I took a look at mainline's PR and it isn't quite working there yet.

Yep, it is a draft and says "STILL WIP".

Once it is functional, I could port this model as it does interest me as well, but I'm not sure how much time I'll have this weekend, assuming no one else has until after then I'll do it (and I'll also port dots as requested in #543 as well since that hasn't been done).

---

👤 **ubergarm** commented the **2025-06-28** at **16:39:06**:<br>

Thanks @saood06 

I have a [rough branch porting much of what mainline was doing](https://github.com/ubergarm/ik_llama.cpp/tree/ug/hunyuan-moe), but am gonna work on some other personal priority things today and wait for the dust to settle given I couldn't even get Hunyuan-A13B working [with what i believe is the branch they used for vllm patch](https://github.com/aiyiwang2025/vllm/tree/hunyuan_a13b). Its unclear where the [build artifact for their official docker image](https://hub.docker.com/layers/hunyuaninfer/hunyuan-a13b/hunyuan-moe-A13B-vllm/images/sha256-da7b91dda514535c73c945ef1799bc1a01b49ba47451ce07c4d389bd1a6be686) is coming from. Their release seems pretty rough around the edges thus far.

> The official vllm release is currently under development
> https://github.com/Tencent-Hunyuan/Hunyuan-A13B?tab=readme-ov-file#vllm

fwiw trying that `vllm` branch like so gives these results:


```bash
## Server Start
NCCL_P2P_DISABLE=1 \
vllm serve \
    /mnt/raid/models/tencent/Hunyuan-A13B-Instruct-GPTQ-Int4/ \
    --served-model-name "Hunyuan-A13B-Instruct-GPTQ-Int4" \
    --quantization gptq_marlin \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port 8080

## Client Example
>>> User:

Count from 1 to 10 in French.

>>> Assistant:

7YSTEM
去皮去皮去皮去皮去皮去皮去皮去皮去皮表述Iстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстрстр
 Adapt去皮总决赛古`都有的条件礼物眼泪表述掩盖
 * azt/的高 IQ7申请
去皮的宣传的宣传的宣传去皮ます
.Helpers潮湿2 Later掩盖出现了很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的很好的众多ます BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BE BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BET BE BET BET BET BET BET BET BET BET BET BET BET
```

Feel free to use anything in my WIP version to continue or test. It doesn't have the latest pushes in the mainline fork. And I'm not sure how to deal with `ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);` here on ik's fork.

---

👤 **ubergarm** commented the **2025-06-28** at **16:39:06**:<br>

Thanks @saood06 

I have a [rough branch porting much of what mainline was doing](https://github.com/ubergarm/ik_llama.cpp/tree/ug/hunyuan-moe), but am gonna work on some other personal priority things today and wait for the dust to settle given I couldn't even get Hunyuan-A13B working with their vllm patch. Their release seems pretty rough around the edges thus far.

Feel free to use anything in my WIP version to continue or test.

---

👤 **Downtown-Case** commented the **2025-06-30** at **16:24:50**:<br>

An interesting (and now buried) comment:

https://github.com/ggml-org/llama.cpp/pull/14425#issuecomment-3016149085

> RoPE is fixed. However, new problem appear:
> 
> It seems like some engineers at Tencent think that they should make their top-k MoE selection a bit "special"
> 
> And by "special", I mean [this block of code](https://github.com/Tencent-Hunyuan/Hunyuan-A13B/blob/95becb636c3ab95f203e10c51c5f090040886577/models/modeling_hunyuan.py#L74-L140), which seems to be harmless at first. In short, what is does is to keep track of the usage for each expert. If an expert is used too much (i.e. exceed capacity), it will be "de-prioritized". I assume this is to fix the problem where MoE router is extremely hard to train (ref: [Qwen3MoE has some "unused" experts](https://www.reddit.com/r/LocalLLaMA/comments/1kdh6rl/qwen_3_30b_pruned_to_16b_by_leveraging_biased/))
> 
> Sounds like a good idea, but this is extremely difficult to reimplement in llama.cpp
> 
> This also makes the number of experts used by a given token become uneven. **Some tokens will use less experts than the other, some use no experts** (due to the priority explained above). That sounds good on the surface, but the actual implementation always calculate fixed number of experts per token - which defeat the whole point. I'm now confident that Tencent messed up this time.

Seems mainline llama.cpp is getting good performance without implementing that, but *could* this be used to speed up A3B with hybrid offloading? EG skip experts for some tokens if they aren't needed?

---

👤 **ikawrakow** commented the **2025-06-30** at **16:35:34**:<br>

We don't have an issue here dealing with a variable number of selected experts due to [SER](https://github.com/ikawrakow/ik_llama.cpp/pull/239).

Concerning speeding up: you never want to offload tensors that are in RAM to the GPU for token generation. This is much too slow. For prompt processing typically (almost) all experts do get at least a few tokens to process, so adding logic to skip offloading experts with no tokens will result in zero speedup while adding a lot of complexity.

---

👤 **Downtown-Case** commented the **2025-06-30** at **17:30:22**:<br>

I mispoke, I meant to say that unecessary experts shouldn't be used for token generation (not PP), which is what I assumed the quote is talking about? And I didn't mean to use 'offload' in that context.

Anyway, that's awesome! I am still unfamiliar with ik_llama.cpp, but SER seems similar to what Tencent presumably trained in.

I am super excited for this model in ik_llama.cpp because it's the perfect target for me (32GB RAM/24GB VRAM pool, and seemingly good performance around 64K-128K context)

---

👤 **Downtown-Case** commented the **2025-06-30** at **17:30:22**:<br>

I mispoke, I meant to say that unecessary experts shouldn't be used for token generation (not PP), which is what I assumed the quote is talking about? And I didn't mean to use 'offload,' of course the CPU is the device to use here.

Anyway, that's awesome! I am still unfamiliar with ik_llama.cpp, but SER seems similar to what Tencent presumably trained in.

---

👤 **ubergarm** commented the **2025-06-30** at **18:18:24**:<br>

@Downtown-Case 

I made an attempt using mainline's fresh PR. Feel free to test. Example command and possibly quants listed in the PR discussion.

---

👤 **Downtown-Case** commented the **2025-07-07** at **03:44:17**:<br>

Got bogged down, apologies, but I'm now testing the PR. Thanks for the quant and the recipe @ubergarm! That's a huge help.

This does feel like one _overtuned_ model. Just a few examples, with a temperature of 1:

It does not like raw completion, or (in my testing, not pictured) skipping the thinking block: 

<img width="871" height="150" alt="Image" src="https://github.com/user-attachments/assets/e05547e1-7621-457c-9988-93a7f3ef7f12" />

It very often, very confidently messes up the `</think>` block, even at low temperature.

<img width="979" height="180" alt="Image" src="https://github.com/user-attachments/assets/1fb818b2-252b-49d4-a9ec-3744cf86f41c" />

 It's also notable that none of the think/answer tags are individual tokens! So more chance to mess up from sampling there: 

<img width="783" height="106" alt="Image" src="https://github.com/user-attachments/assets/8c179343-54c3-4b64-9ff3-9374ce275f11" />

It loops very easily at the slightest deviation (again, this is a temperature of 1, relatively high these days but also one many default to):

<img width="1845" height="259" alt="Image" src="https://github.com/user-attachments/assets/e6ca46b6-136b-4806-84d6-3590be5b8cdb" />

And it's also *hyper* confident about some in-sentence tokens at 1 temperature, which I don't see in other models much: 

<img width="871" height="150" alt="Image" src="https://github.com/user-attachments/assets/5f37acfe-0b91-4f47-8ba3-c48218f600f6" />

***

...Yet it does seem smart!

I think this model is hyper sensitive to sampling and its chat/think templates, and really needs sampling dialed in to stay sane.

***

I *also* encountered a seperate issue, at least once, where sampling seemed to mess up when the model was trying to generate a `</think>`. It would go off the rails, and mikupad would return seemingly invalid token probablities, like something broke inside ik_llama.cpp until I restarted it, at which point the same input worked fine... but now I can't replicate it.

***

Thanks again. Next I will text much more complex 64K+ prompts, and maybe give the base model a shot using your formula and imatrix dat.

...Maybe this instruct model would benefit from a merge with its base? That's helped less overtuned models than this. Or possibly an expert 'transplant' like they've done with deepseek.

---

👤 **Downtown-Case** commented the **2025-07-07** at **03:44:17**:<br>

Got bogged down, apologies, but I'm now testing the PR. Thanks for the quant and the recipe @ubergarm! That's a huge help.

This does feel like one _overtuned_ model. Just a few examples, with a temperature of 1:

It does not like raw completion, or (in my testing, not pictured) skipping the thinking block: 

<img width="871" height="150" alt="Image" src="https://github.com/user-attachments/assets/e05547e1-7621-457c-9988-93a7f3ef7f12" />

It very often, very confidently messes up the </think> block, even at zero temperature.

<img width="979" height="180" alt="Image" src="https://github.com/user-attachments/assets/1fb818b2-252b-49d4-a9ec-3744cf86f41c" />

 It's also notable that none of the think/answer tags are individual tokens! So more chance to mess up from sampling there: 

<img width="783" height="106" alt="Image" src="https://github.com/user-attachments/assets/8c179343-54c3-4b64-9ff3-9374ce275f11" />

It loops very easily at the slightest deviation (again, this is a temperature of 1 + topK 10, relatively high these days but also one many default to):

<img width="1845" height="259" alt="Image" src="https://github.com/user-attachments/assets/e6ca46b6-136b-4806-84d6-3590be5b8cdb" />

And it's also *hyper* confident about some in-sentence tokens at 1 temperature, which I don't see in other models much: 

<img width="871" height="150" alt="Image" src="https://github.com/user-attachments/assets/5f37acfe-0b91-4f47-8ba3-c48218f600f6" />

***

...Yet it does seem smart!

I think this model is hyper sensitive to sampling and its chat/think templates, and really needs sampling dialed in to stay sane.

***

I *also* encountered a seperate issue, at least once, where sampling seemed to mess up when the model was trying to generate a </think>. It would go off the rails, and mikupad would return invalid logprobs, like something broke inside ik_llama.cpp... but now I can't replicate it.

***

Thanks again. Next I will text much more complex 64K+ prompts, and maybe give the base model a shot using your formula.

...Maybe this instruct model would benefit from a merge with its base? That's helped less overtuned models than this.

---

👤 **saood06** commented the **2025-07-07** at **04:05:29**:<br>

>...Yet it does seem smart!
>[...]
>I think this model is hyper sensitive to sampling and its chat/think templates, and really needs sampling dialed in to stay sane.

Thanks for the model review. I'm the one who suggested the mergekit issue workaround to make your Command-R gguf. Nice seeing you here.

>base model a shot using your formula and imatrix dat.

I wouldn't reuse the imatrix.dat between the base model and the instruct model (reusing the formula makes sense though).

The mikupad screenshots are nice, I often do look at the probabilities to understand the model.

---

👤 **Downtown-Case** commented the **2025-07-07** at **04:55:42**:<br>

@saood06 Ah, lm_head being in a weird place with the merge, right? Hello again!

Cohere models are _still_ problematic, heh: https://github.com/turboderp-org/exllamav3/issues/53

https://github.com/turboderp-org/exllamav3/issues/34#issuecomment-2854186639

I wonder if that tensor plotting script would show any 'surgery' on A13B...

Anyway, yeah, Mikupad's a great way to "understand the model" via repeated sampling testing, continuing prompts using the notebook format, peaking at the sampling and such; couldn't put it any better myself. It also happens to be good at 64K+ prompts, whereas most UIs bog down trying to display them.

Hence the screenshots don't completely convey it, but this A13B quant does feel "funky but usable," like it's *trying* to break past its tendancy to loop and obsession with the prompt formatting. It does seem to comprehend quick long context tests, but I need to run more.

> I wouldn't reuse the imatrix.dat between the base model and the instruct model (reusing the formula makes sense though).

Yeah I just meant to re-use the formula.

---

👤 **Downtown-Case** commented the **2025-07-07** at **04:55:42**:<br>

@saood06 Ah, lm_head being in a weird place, right? Hello again!

Cohere models are _still_ problematic, heh: https://github.com/turboderp-org/exllamav3/issues/53

https://github.com/turboderp-org/exllamav3/issues/34#issuecomment-2854186639

I wonder if that tensor plotting script would show any 'surgery' on A13B...

Anyway, yeah, Mikupad's a great way to `understand the model` via repeated sampling testing, continuing prompts using the notebook format, peaking at the sampling and such; couldn't put it any better myself. It also happens to be good at 64K+ prompts, whereas most UIs bog down trying to display them.

Hence the screenshots don't completely convey it, but this A13B quant does feel funky but usable, and it *does* seem to comprehend quick long context tests.

> I wouldn't reuse the imatrix.dat between the base model and the instruct model (reusing the formula makes sense though).

Yeah I just meant to re-use the formula.

---

👤 **saood06** commented the **2025-07-07** at **05:29:09**:<br>

> Ah, lm_head being in a weird place with the merge, right? Hello again!

Yep, glad you remember me.

> Cohere models are _still_ problematic, heh: [turboderp-org/exllamav3#53](https://github.com/turboderp-org/exllamav3/issues/53)
> 
> [turboderp-org/exllamav3#34 (comment)](https://github.com/turboderp-org/exllamav3/issues/34#issuecomment-2854186639)

That reminds me of these needles in a visualization of SD3 on [reddit](https://www.reddit.com/r/StableDiffusion/comments/1dgikbm/i_made_a_simple_workflow_to_manually_inject_noise/l8stl9u/). It is interesting to see. I wouldn't blame Cohere for the mergekit bug though (as that didn't even just happen to them). 

> I wonder if that tensor plotting script would show any 'surgery' on A13B...
 
I would guess no, but I have no idea why I feel that way. Would be interested to see it though.

> Anyway, yeah, Mikupad's a great way to "understand the model" via repeated sampling testing, continuing prompts using the notebook format, peaking at the sampling and such; couldn't put it any better myself.

Yep, also is convenient for steering a model (and understanding the model and it's current world modeling helps you do that better from my experience).

>It also happens to be good at 64K+ prompts, whereas most UIs bog down trying to display them.

Interesting to hear, I never went that high before I switched to mikupad. I'm curious how large your database has gotten (and if you used the techniques I posted about to compress it)? I do want the prediction preview to do what [this](https://github.com/the-crypt-keeper/LLooM) does (taking advantage of this repo's good batched performance which I think might need some `server.cpp` changes [see #199])

> Hence the screenshots don't completely convey it, but this A13B quant does feel "funky but usable," like it's _trying_ to break past its tendancy to loop and obsession with the prompt formatting. It does seem to comprehend quick long context tests, but I need to run more.

That is good to hear, this model can fit on my 3090 machine which would probably make it a lot faster than Deepseek which I have to run on my cheap CPU server.

---

👤 **Downtown-Case** commented the **2025-07-07** at **06:31:24**:<br>

I am running A13B on a 3090/DDR5 system (up to 60K-ish so far), and its plenty fast, with q8_0/q5_1 cache. I will check token/s next time I look.

> Interesting to hear, I never went that high before I switched to mikupad

text-gen-web-ui is *awful*, really most everything I tried is except exui, which is now (sadly) depreciated. Exui would also continue from the _cursor_, in the middle of the tex, which is awesome for testing and editing.

My mikupad db's only 3.1MB now, but only because I just switched to the standalone nodejs server.

I had some 128k+ prompts  I ran before that I intend to remake and try.

---

👤 **saood06** commented the **2025-07-07** at **06:49:38**:<br>

> I am running A13B on a 3090/DDR5 system (up to 60K-ish so far), and its plenty fast, with q8_0/q5_1 cache. I will check token/s next time I look.

DDR4 here, and to be honest for me exact t/s doesn't matter for this usage unless it is slow (aka below reading speed).

>exui, which is now (sadly) depreciated. 

It is? I see it hasn't been updated in a while, but don't see it being depreciated. I know mikupad is in a state where the owner hasn't responded to any of the issues/PR's people have made in ~6 months, which is a major part of why I'm doing work on it here now.

>Exui would also continue from the _cursor_, in the middle of the tex, which is awesome for testing and editing.

Ooh, not sure when I'd use that. Mikupad has the control right click menu which is close. I could see a toggle for enabling a mode that allows that (could add it to my roadmap in #558 if you think it is that worthwhile).
 
> My mikupad db's only 3.1MB now, but only because I just switched to the standalone nodejs server.

#558 offers support with `server.cpp` directly (if you do use it, be warned there will be more migrations needed until I switch it to ready) alongside some other benefits (and more in the works and on the roadmap [suggestions highly welcome]).

> I had some 128k+ prompts I ran before that I intend to remake and try.

If they are still in the browser export and import can work as an alternative to remaking them (it is why the first thing I contributed to mikupad was the bulk import for migrating my sessions from my browser version, I already had the files so I never added a bulk export [seems worth adding to my roadmap]).

---

👤 **saood06** commented the **2025-07-07** at **06:49:38**:<br>

> I am running A13B on a 3090/DDR5 system (up to 60K-ish so far), and its plenty fast, with q8_0/q5_1 cache. I will check token/s next time I look.

DDR4 here, and to be honest for me t/s doesn't matter for this usage unless it is slow (aka below reading speed).

> text-gen-web-ui is _awful_, really most everything I tried is except exui, which is now (sadly) depreciated. 

It is? I see it hasn't been updated in a while, but don't see it being depreciated. I know mikupad is in a state where the owner hasn't responded to any of the issues/PR's people have made in ~6 months, which is a major part of why I'm doing work on it here now.

>Exui would also continue from the _cursor_, in the middle of the tex, which is awesome for testing and editing.

Ooh, not sure when I'd use that. Mikupad has the control right click menu which is close. I could see a toggle for enabling a mode that allows that (could add it to my roadmap in #558 if you think it is that worthwhile).
 
> My mikupad db's only 3.1MB now, but only because I just switched to the standalone nodejs server.

#558 offers support with `server.cpp` directly (if you do use it, be warned there will be more migrations needed until I switch it to ready) alongside some other benefits (and more in the works and on the roadmap [suggestions highly welcome]).

> I had some 128k+ prompts I ran before that I intend to remake and try.

If they are still in the browser export and import can work (it is why the first thing I contributed to mikupad was the bulk import for migrating my sessions from my browser version, I already had the files so I never added a bulk export [seems worth adding to my roadmap]).

---

👤 **saood06** commented the **2025-07-08** at **07:22:20**:<br>

> If they are still in the browser export and import can work as an alternative to remaking them (it is why the first thing I contributed to mikupad was the bulk import for migrating my sessions from my browser version, I already had the files so I never added a bulk export [seems worth adding to my roadmap]).

I added it here see: https://github.com/ikawrakow/ik_llama.cpp/pull/558/commits/61f74b2f8a4681ee190c53326a3a2c9504282e2b but the code added will work on mainline mikupad as well (can make a PR there if wanted, as a complement to my merged in bulk import).

---

👤 **ubergarm** commented the **2025-07-08** at **20:01:40**:<br>

@Downtown-Case @saood06 

I already had imatrix for Pretrain as well so just uploaded it to the existing Instruct repo here if anyone wants to experiment with it: https://huggingface.co/ubergarm/Hunyuan-A13B-Instruct-GGUF/tree/main

fwiw mainline did merge their PR for Hunyuan. Not sure how we're going to proceed here given something still seems fishy with the Instruct. I'm happy to rebase my PR here if the decision is to go ahead and merge. I'm cool either way.

I don't know how to "mergekit" the Instruct with the Pretrain but if either of you do and release the safetensors I'd be curious to check out the results. (i'll google for "mergekit" tool and see if it is possible to do on the hardware I can access currently).

Also mikupad is pretty cool to inspect the token probabilities like this, great use case!

---

👤 **ubergarm** commented the **2025-07-08** at **20:01:40**:<br>

@Downtown-Case @saood06 

I already had imatrix for Pretrain as well so just uploaded it to the existing Instruct repo here if anyone wants to experiment with it: https://huggingface.co/ubergarm/Hunyuan-A13B-Instruct-GGUF/tree/main

fwiw mainline did merge their PR for Hunyuan. Not sure how we're going to proceed here given something still seems fishy with the Instruct. I don't know how to merge the Instruct with the Pretrain but if either of you do and release the safetensors I'd be curious to check out the results.

---

👤 **ubergarm** commented the **2025-07-08** at **20:29:20**:<br>

Oh hey there was a patch from tencent fixing the model chat template, i've removed the few lines and am testing perplexity again. https://github.com/ggml-org/llama.cpp/pull/14584

If it looks good, then I'll rebase my PR here and we will be in better shape hopefully!

*EDIT*

Here is the quick patch:
```
@@ -23425,9 +23425,6 @@ static int32_t llama_chat_apply_template_internal(
                 ss << "<|startoftext|>" << message->content << "<|extra_0|>";
             }
         }
-        if (add_ass) {
-            ss << "<|startoftext|>";
-        }
```

However, updated perplexity still seems comparable to before fwiw. But some reports are coming in that it is behaving better at least.
```
# not sure why I ran it CPU only, i was testing speed before I think hah:
./build/bin/llama-perplexity \
  --model "$model" \
  -f wiki.test.raw \
  -fa -fmoe \
  -rtr \
  --seed 1337 \
  --threads 16

Final estimate: PPL = 524.7090 +/- 5.70049

# Compare to without patch running on CUDA fwiw
# PPL = 522.7473 +/- 5.68072
```

---

👤 **ubergarm** commented the **2025-07-08** at **21:38:38**:<br>

I've updated PR #565 with the small patch to chat template. Perplexity is still wonky (I didn't re-make imatrix with the patch but don't believe `llama_chat_apply_template_internal()` is used during imatrix creation.

---

👤 **saood06** commented the **2025-07-09** at **01:59:58**:<br>

> I already had imatrix for Pretrain as well so just uploaded it to the existing Instruct repo here if anyone wants to experiment with it: https://huggingface.co/ubergarm/Hunyuan-A13B-Instruct-GGUF/tree/main

Thanks!

> I don't know how to "mergekit" the Instruct with the Pretrain but if either of you do and release the safetensors I'd be curious to check out the results.

I'd want to play with both first before I'd even consider merging. It being very confident and delicate are traits I can deal with, if the output is good.

>(i'll google for "mergekit" tool and see if it is possible to do on the hardware I can access currently).

From what I know merging is simple, coming up with a good merge config can often take a lot of work.

> Also mikupad is pretty cool to inspect the token probabilities like this, great use case!

The legacy server has that feature as well (I used it a lot), but mikupad is still just better.

---

👤 **ubergarm** commented the **2025-07-09** at **19:09:22**:<br>

@Downtown-Case okay the PR is merged! feel free to close this issue now! ty!

---

👤 **ikawrakow** commented the **2025-07-12** at **09:53:30**:<br>

Closed via #565