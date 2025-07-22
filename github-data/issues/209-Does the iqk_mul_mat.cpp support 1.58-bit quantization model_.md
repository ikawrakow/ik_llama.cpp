### 📝 [#209](https://github.com/ikawrakow/ik_llama.cpp/issues/209) - Does the iqk_mul_mat.cpp support 1.58-bit quantization model?

| **Author** | `godrosev` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-02-19 |
| **Updated** | 2025-03-21 |

---

#### Description

And I have another question.I found the "iqk_mul_mat.inc"file of the "llamafile" is very old. It cannot support any iq model. Do you have the plan to update the file? Thanks

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-02-19** at **05:43:20**:<br>

> Does the iqk_mul_mat.cpp support 1.58-bit quantization model?

Which 1.58-bit model? There is Unsloth's DeepSeek-R1 quantized with `IQ1_S` and sold as 1.58b, but there are also the BitNet ternary models, which actually are 1.58b.

>I found the "iqk_mul_mat.inc"file of the "llamafile" is very old. It cannot support any iq model. Do you have the plan to update the file? 

It cannot? Looking at the current `iqk_mul_mat.inc` I see `IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S` being supported in the code, see [this](https://github.com/Mozilla-Ocho/llamafile/blob/29b5f27172306da39a9c70fe25173da1b1564f82/llamafile/iqk_mul_mat.inc#L2999)

---

👤 **godrosev** commented the **2025-02-19** at **06:30:38**:<br>

Thank you very much for your answer.
Sorry, my previous questions may have been very unclear.
In question one, I mean 1.58b quantization model of Unsloth's Deepseek-r1.
Question 2,I'm also referring to IQ1_S and IQ1_M these.
Eventually, all I want to do is run Unsloth's Deepseek-r1 1.58b quantization model with llamafile. I haven't been able to do that yet, so I'd like to use the [method](https://github.com/ikawrakow/ik_llama.cpp/blob/d44aba79ea9bea07c22cbf2336b51a37ba823524/ggml/src/iqk/iqk_mul_mat.cpp#L13958C14-L13958C32) of ik_llama.cpp  and move it to llamafile, I don't know if that's possible.
I would like to ask you for advice

---

👤 **godrosev** commented the **2025-02-19** at **06:30:38**:<br>

Thank you very much for your answer.
Sorry, my previous questions may have been very unclear.
In question one, I mean 1.58b quantization model of Unsloth's Deepseek-r1.
Question 2,I'm also referring to IQ1_S and IQ1_M these.
Eventually, all I want to do is run Unsloth's Deepseek-r1 1.58b quantization model with llamafile. I haven't been able to do that yet, so I'd like to use the method of https://github.com/ikawrakow/ik_llama.cpp/iqk_mul_mat.cpp  and move it to llamafile, I don't know if that's possible.
I would like to ask you for advice

---

👤 **ikawrakow** commented the **2025-02-19** at **07:11:12**:<br>

You can run Unsloth's `IQ1_S` with this repository,  but it will be slow as I haven't added `IQ1_S` gemm/gemv kernels to `iqk_mul_mat.cpp`, so matrix multiplications will be done via the kernels in `ggml`. If you quantize the model to `IQ1_S_R4`, it will be slightly better (as measured by perplexity) than Unsloth's, it will be a few GB smaller, and will run faster. Nearly 4X faster for prompt processing (a.k.a. prefill), and I estimate about 20% faster for token generation. To quantize the model, you need to find an imatrix file for DeepSeek-R1 on the Internet, and then simply
```
./bin/llama-quantize --imatrix $the_imatrix_your_found --token-embedding-type q8_0 deepseek_model_file quantized_file iq1_s_r4
```
To quantize to `IQ4_M_R4`, just change `iq1_s_r4` to `iq1_m_r4` in the above command.

All other Unsloth quantizations will run here as is with much improved speed by using `-rtr` on the command line. However, model loading will be quite slow as model weights will be repacked for more efficient matrix multiplications while loading, and this takes some time for 670 billion parameters.

Updating `iqk_mul_mat.cpp` in llamafile: no, I don't have plans to do that at this point.

---

👤 **godrosev** commented the **2025-02-19** at **07:42:26**:<br>

Thank！！And I'll try the new method you advocate.

---

👤 **ikawrakow** commented the **2025-02-19** at **07:43:53**:<br>

Btw, what is the system you intend to run this on?

---

👤 **godrosev** commented the **2025-02-19** at **08:15:36**:<br>

Linux(debian) and windows

---

👤 **ikawrakow** commented the **2025-02-19** at **08:17:56**:<br>

I never use/test on Windows, so this may or may not work. But what I meant is the system specs (CPU, amount of RAM).

---

👤 **godrosev** commented the **2025-02-19** at **08:53:28**:<br>

Oh,I just misunderstood.
I have two device
One is a server:
Intel Xeon 6348 * 2 ，DDR4 3200 512GB，RTX3090*2
the other is a PC:
AMD AI MAX+395, LPDDR5 8000Mhz 128G;iGPU 40cu 8060s
Now i want to intend the 1.58b Deepseek 671B model on the ai max 395.
I use the [ktransformers](https://github.com/kvcache-ai/ktransformers).But they dont support the IQ1_S model(Because they use the llamafile).So I'd like to modify this part of the code myself.
How should I do it best? Can you give me some advice? Thank you very much

---

👤 **godrosev** commented the **2025-02-19** at **08:53:28**:<br>

Oh,I just misunderstood.
I have two device
One is a server:
Intel Xeon 6348 * 2 ，DDR4 3200 512GB，RTX3090*2
the other is a PC:
AMD AI MAX+395, LPDDR5 8000Mhz 128G;iGPU 40cu 8060s
Now i want to intend the 1.58b Deepseek 671B model on the ai max 395.
I use the ktransformers.But they dont support the IQ1_S model(Because they use the llamafile).So I'd like to modify this part of the code myself.

---

👤 **ikawrakow** commented the **2025-02-19** at **13:25:15**:<br>

What is the advantage of using KTransformers? Are you more familiar with Python?

---

👤 **saood06** commented the **2025-02-19** at **14:10:18**:<br>

> What is the advantage of using KTransformers? Are you more familiar with Python?

KTransformers offers the best performance for running Deepseek mostly on CPU (but they only support certain hardware configs and limited amount of KV). There is some performance for ik_llama.cpp running Deepseek here: #223 .They ran ik_llama in a lot of configs (default attention ,mla, mla+cuda, fa, fa+q8kv).

---

👤 **saood06** commented the **2025-02-19** at **14:10:18**:<br>

> What is the advantage of using KTransformers? Are you more familiar with Python?

KTransformers offers the best performance (but they only support certain hardware configs and limited amount of KV). There is a comparison between ik_llama.cpp, llama.cpp and ktransformers running Deepseek here: https://www.reddit.com/r/LocalLLaMA/comments/1iq6ngx/ktransformers_21_and_llamacpp_comparison_with/ .They ran ik_llama in a lot of configs (default attention ,mla, mla+cuda, fa, fa+q8kv).

---

👤 **ikawrakow** commented the **2025-02-19** at **14:22:59**:<br>

@saood06 

The comparison in the linked Reddit thread does not use run-time-repacking in `ik_llama.cpp`, correct? And then, where is it fair to compare performance at a context of 8K to performance at 64k tokens?

---

👤 **saood06** commented the **2025-02-19** at **14:32:22**:<br>

@ikawrakow 

The table is a little misleading, the context is only launch config in order to show RAM usage differences between the configs. All tests were done with 500 token prompt for prefill, and a 300 token response (not very deep in context which shows why the non MLA configs still look decent).

There is no -rtr and I did not ask the person to test with it, as -rtr with MoE models was only just fixed a few hours ago, I could ask the person to pull ik_llama.cpp and test that.

Two things I found interesting where FA reducing TG performance relative to the standard (which is not what you saw with Deepseek-lite),  and CUDA+mla leading to very poor PP (adding more evidence that there is a serious bottleneck in CUDA implementation).

Edit: I had mentioned using a IQ4_K_R4 to them but they ended up testing ik_llama.cpp MLA via downloading a quant from huggingface as conversion was hitting issues for them.

---

👤 **ikawrakow** commented the **2025-02-19** at **15:04:10**:<br>

So, the `V` cache is transposed without FA, so when you store a single token, it will go and touch the entire giant 64k context memory allocation. This out to have some impact on what stuff goes into what memory bank, thus affecting TG performance.  I must admit I don't quite understand why the resident memory is still so high when using FA (in that case tokens are stored consecutively, so I expect to see only the memory actually used reported). Clearly, something is not quite right there.

The person is using 6 experts or 8 with KTransformers?

> There is no -rtr and I did not ask the person to test with it, as -rtr with MoE models was only just fixed a few hours ago, I could ask the person to pull ik_llama.cpp and test that.

Yes , please. The "fix" you mention from PR #210 does improve things. But that's on top of the improvement that `-rtr` gives even without #210 

My napkin math tells me that something is not quite right in this testing. I now get 660 t/s for DeepSeek-Lite for a context of 500 tokens on my Ryzen-7950X for `IQ4_XS -rtr`. DeepSeek-Lite has 2.4B active parameters, DeepSeek-R1 has 37B (but otherwise the architecture is basically the same). So, I expect to see `660*2.4/37 = ~43 t/s` on my CPU. His CPU is ~2X my CPU, so I'm expecting in the range of 70-80 t/s for PP.  The other thing is that in the KTRansformers repo they brag about 97 t/s on a **dual EPYC with 6 experts**, but this guy is getting 83 t/s on a single EPYC? (with how many experts?)

I also don't get their low TG performance with FA. With a context of 500 tokens it should be about the same as no FA.

---

👤 **godrosev** commented the **2025-02-20** at **03:34:55**:<br>

> What is the advantage of using KTransformers? Are you more familiar with Python?

No, no, no, in fact I don't even like python very much.
I also don't think KT did a better job than ik_llamacpp in most of the optimizations.
Simply because their architecture can run most of the 671b's deepseek layers on CPU and memory, and only the active expert model (<37b) on the GPU and VRAM.
But their support for IQ is nowhere near as good as ik_llamacpp, so I wanted to give it a try.
Also, is there any chance that ik_llamacpp will also load the activated expert model into the VRAM as KT did above, so that I can use one 3090 running 671b Deepseek.
I think this should be pretty easy for you compared to other performance acceleration jobs you do

---

👤 **ikawrakow** commented the **2025-02-20** at **10:43:34**:<br>

@godrosev 

#212 has `iqk_mul_mat.cpp` implementation for `IQ1_S`

---

👤 **godrosev** commented the **2025-02-20** at **13:17:54**:<br>

Thank you very much indeed!

---

👤 **ikawrakow** commented the **2025-02-21** at **07:56:02**:<br>

> KTransformers offers the best performance for running Deepseek mostly on CPU (but they only support certain hardware configs and limited amount of KV)

So, they keep the attention tensors on the GPU and do the MoE part on the CPU. is that it? Or is there more to it? I didn't see anything in the code that would make it run faster without a GPU. Or am I missing something?

---

👤 **saood06** commented the **2025-02-21** at **23:48:25**:<br>

>is that it?

Basically yes.

There are people (me included) who have tried to use llama.cpp (via an unmerged PR) to effectively do the same thing with llama.cpp and place only the attention tensors on the GPU and leave the experts on the CPU with varying degrees of success ( other people reported better performance but I ran into performance degradation and that might be because my GPU was only able to be accessed via RPC). There was even someone who reported a lot of success with Mixtral 8x22 (+66% better TG and -26% PP vs normal offloading) and that seems even more promising as llama.cpp has a better CUDA implementation for that than Deepseek where offloading crashes PP performance.

I looked into porting that PR over to ik_llama.cpp but it looks like it would have to be basically rewritten and I haven't really put in any more time since then.

>Or is there more to it? I didn't see anything in the code that would make it run faster without a GPU. Or am I missing something?


Technically they do have some more features (and I'm not sure how much is in their source code as last I checked they did a binary only release initially of their latest version), but they aren't very relevant. They do better NUMA by duplicating the model on each node, avoiding any inter-socket model access but at the cost of double the memory footprint, and they also have AMX instruction support which is only relevant to a handful of CPUs.

---

👤 **saood06** commented the **2025-02-21** at **23:48:25**:<br>

>is that it?
Basically yes.
>Or is there more to it? I didn't see anything in the code that would make it run faster without a GPU. Or am I missing something?
Technically they do have some more features (and I'm not sure how much is in their source code as last I checked they did a binary only release initially of their latest version), but they aren't very relevant. They do better NUMA by duplicating the model on each node, avoiding any inter-socket model access but at the cost of double the memory footprint, and they also have AMX instruction support which is only relevant to a handful of CPUs.

There are people (me included) who have tried to use llama.cpp (via an unmerged PR) to effectively do the same thing with llama.cpp and place only the attention tensors on the GPU and leave the experts on the CPU with varying degrees of success ( other people reported better performance but I ran into performance degradation and that might be because my GPU was only able to be accessed via RPC and was not local). There was even someone who reported a lot of success with Mixtral 8x22 (+66% better TG and -26% PP vs normal offloading) and that seems even more promising as llama.cpp has a better CUDA implementation for that than Deepseek where offloading crashes PP performance.

I looked into porting that PR over to ik_llama.cpp but it looks like it would have to be basically rewritten and I haven't really put in any more time since then.

---

👤 **godrosev** commented the **2025-02-22** at **01:48:08**:<br>

> > KTransformers offers the best performance for running Deepseek mostly on CPU (but they only support certain hardware configs and limited amount of KV)
> 
> So, they keep the attention tensors on the GPU and do the MoE part on the CPU. is that it? Or is there more to it? I didn't see anything in the code that would make it run faster without a GPU. Or am I missing something?

You didn't miss it, the current version, they just implement such a feature, and there is nothing special other than that.
Version 0.3 claims to include the AMX instruction set to further increase speed, but only supports certain CPUs.
Other CPU instruction optimizations are based on LlamaFile acceleration (i.e., your iqk_mul_mat).
Therefore, I think the work that you do is the most important and crucial

---

👤 **ikawrakow** commented the **2025-03-21** at **12:38:49**:<br>

I think we can close this one.