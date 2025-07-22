### 🐛 [#474](https://github.com/ikawrakow/ik_llama.cpp/issues/474) - Bug: Perf Regression in PP throughput after Pull [#461](https://github.com/ikawrakow/ik_llama.cpp/issues/461) (...R4 CUDA impl)

| **Author** | `usrlocalben` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-30 |
| **Updated** | 2025-07-05 |

---

#### Description

### What happened?

While testing out an IQ4 quant of R1-0528 I noticed that PP throughput on my system was reduced e.g. 75/s -> 12/s, basically equal to TG throughput. With IQ4 and Q8 shared on GPU I expect PP > 60/s.

I compare with an all Q8_0 quant and see what I expect, PP >50/sec (on main/HEAD today.)

I bisected, and found that this problem was introduced with Pull #461 (commit 1429291).

However, my IQ4 quant **doesn't have any _R4 tensors**. It's Q8 shared, and IQ4_K for the remaining tensors.

Absence/presence of `--run-time-repack` doesn't cause nor avoid it.

CUDA device is RTX 8000 (Turing)

I glance over the commit and mostly see changes that seem clearly restricted to _R4 suffix components. There are some shared parts where _n_interleaved_ is propagated down the template stack (iqk_mmvq.cu) but at a casual glance nothing strikes me as odd, but I'm certainly not that familiar with it. The dot product interface changed to a mutating one taking an accumulator pointer (previously returning the computed result) and that could be curious.

aside, but maybe related -- there were recent PRs related to mla/fa that had some vague language wrt. Turing support. (Pulls #386  and #408 ) I say vague because 386 indicates turing is not supported, then 408 indicates that it is extended to Turing, but I'm not sure they're referring to the same thing, and the changes in 408 don't seem very significant. It's not clear what the proper mla/fa settings should be on Turing at this time. I currently use `-mla 2 -fa`


### What operating system are you seeing the problem on?

Linux

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-30** at **07:48:21**:<br>

> However, my IQ4 quant doesn't have any _R4 tensors. It's Q8 shared, and IQ4_K for the remaining tensors.

> Absence/presence of --run-time-repack doesn't cause nor avoid it.

To make sure I understand correctly, prior to #461 you observed the same good PP performance irrespective of using or not using `--run-time-repack`. But after #461 you observe the same bad bad PP performance with or without `--run-time-repack` ?

---

👤 **ikawrakow** commented the **2025-05-30** at **07:56:37**:<br>

Please also provide your full command line. This really makes it easier to diagnose the problem.

---

👤 **usrlocalben** commented the **2025-05-30** at **17:15:52**:<br>

```
ik_llama.cpp/build/bin/llama-server
-mla 2 -fa -fmoe
-amb 512
-c 65536
-np 1
--n-gpu-layers 99
-ctk q8_0
--run-time-repack
-ot "blk\.3\.ffn_up_exps=CUDA0, blk\.3\.ffn_gate_exps=CUDA0"
-ot "blk\.4\.ffn_up_exps=CUDA0, blk\.4\.ffn_gate_exps=CUDA0"
-ot "blk\.5\.ffn_up_exps=CUDA0, blk\.5\.ffn_gate_exps=CUDA0"
-ot "blk\.6\.ffn_up_exps=CUDA0, blk\.6\.ffn_gate_exps=CUDA0"
-ot "blk\.7\.ffn_up_exps=CUDA0, blk\.7\.ffn_gate_exps=CUDA0"
-ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU"
--host 127.0.0.1 --port 9999
--temp 0.6 --top-p 0.95
-m /path/to/model/DeepSeek-R1-0528-IQ4/data.gguf
```


```

commit 24c010b3 (last known good)

rtr=yes
prompt eval time     =  161791.56 ms / 10188 tokens (   15.88 ms per token,    62.97 tokens per second)
generation eval time =  115078.31 ms /  1012 runs   (  113.71 ms per token,     8.79 tokens per second)

rtr=no
prompt eval time     =  612061.95 ms / 10188 tokens (   60.08 ms per token,    16.65 tokens per second)
generation eval time =  144322.65 ms /  1268 runs   (  113.82 ms per token,     8.79 tokens per second)


commit 14292913 (CUDA _R4)

rtr=yes
prompt eval time     =  937934.38 ms / 10188 tokens (   92.06 ms per token,    10.86 tokens per second)
generation eval time =  122195.15 ms /  1065 runs   (  114.74 ms per token,     8.72 tokens per second)

rtr=no
prompt eval time     =  613312.38 ms / 10188 tokens (   60.20 ms per token,    16.61 tokens per second)
generation eval time =  163612.05 ms /  1437 runs   (  113.86 ms per token,     8.78 tokens per second)

```

---

👤 **ikawrakow** commented the **2025-05-31** at **04:35:34**:<br>

**Observations**:
* rtr=no has the same performance on 14292913 and on 24c010b3. In both versions, when rtr=no tensors stored in RAM get offloaded to the GPU to perform the matrix multiplication. 
* rtr=no is much slower that rtr=yes on the last know good 24c010b3. On that version, when rtr=yes tensors stored in RAM are not offloaded to the GPU because the CUDA back-end reports to not support matrix multiplications for the repacked types. 

Conclusion: PCE-E speed is very low, resulting in low PP performance when tensors stored in RAM are offloaded to the GPU. #461implemented CUDA matrix multiplications for repacked tensors, so after the PR all tensors stored in RAM get offloaded to the GPU to perform matrix multiplications, so performance drops.

**Mitigations**:
* If possible, use large u-batches. This allows more work to be done per amount of data copied to the GPU. If you have enough VRAM, `-b 4096 -ub 4096` will maximize PP performance.
* Avoid offloading tensors stored in RAM to the GPU. This is accomplished with `-op 26,0,27,0,29,0` where
    - `26,0` disables offloading matrix multiplications
    - `27,0` disables offloading indirect matrix multiplications (used in MoE models)
    - `29,0` disables  offloading fused `ffn_up+ffn_gate` operations (you get these in MoE models when using `-fmoe`)
 * You may want to experiment with `-op` (`op` stands for offload policy, see PR #405)
     - `-op 29,0 -rtr` should result in the exact same performance as you had on 24c010b3
     - If your PCI-E speed is so low as to give such bad performance with GPU offload enabled, adding `-op 27,0` to the above may improve performance compared to what you had on 24c010b3
     
Note that for most people not using `-op` and using large batches with `-b 4096 -ub 4096` maximizes PP performance.

---

👤 **usrlocalben** commented the **2025-06-01** at **23:41:22**:<br>

@ikawrakow 
Switching to b/ub=4096 indeed gives the perf that I observed prior to the CUDA _R4, or better. I've seen as high as 90+ t/s now. (And learned something new about how PP is implemented)

I'm not sure what to do with the Issue. It seems like the commit changed behavior in a way that is orthogonal to its description--but maybe I was just ignorant of the batch-size implications and the previous impl let me get away with it.

Additionally, it seems like the number of combinations of tensor/config/compile settings are quite numerous, and more so now after these changes. Is there a way to know what the optimal arrangement should be? e.g. IQ4_K for GPU-tensors, _R4 for cpu tensors, GGML_CUDA_IQK_FORCE_BF16=1 etc. ? Or is it all YMMV, tradeoffs between PP/TG perf, CUDA-arch etc?

---

👤 **ikawrakow** commented the **2025-06-02** at **06:00:39**:<br>

> I'm not sure what to do with the Issue. It seems like the commit changed behavior in a way that is orthogonal to its description--but maybe I was just ignorant of the batch-size implications and the previous impl let me get away with it.

The performance drop is unexpected and specific to your system. It most likely indicates extremely low PCI-E throughput and/or extremely high PCI-E latency.

> Additionally, it seems like the number of combinations of tensor/config/compile settings are quite numerous, and more so now after these changes. Is there a way to know what the optimal arrangement should be? e.g. IQ4_K for GPU-tensors, _R4 for cpu tensors, GGML_CUDA_IQK_FORCE_BF16=1 etc. ? Or is it all YMMV, tradeoffs between PP/TG perf, CUDA-arch etc?

I know. Writing simple and easy to follow instructions has never been one of my strengths. Models are different (there are big differences in optimum settings between dense and MoE models, and even for MoE models there are big differences between, say, DeepSeek and Maverick), users systems very between 100% GPU and 100% CPU, and anything in between, there are different quantization types with different tradeoffs, etc. Making it easy for the users would be the domain of product managers, marketing specialists, and technical support, none of which is present in a hobby project such as this one. Hence, it is basically up to the user base to come up with the cook book recipes. @ubergarm has done some of that [here](https://github.com/ikawrakow/ik_llama.cpp/discussions/258), but it is by no means complete (and things are moving and changing).

---

👤 **saood06** commented the **2025-06-02** at **07:36:45**:<br>

> I know. Writing simple and easy to follow instructions has never been one of my strengths. Models are different (there are big differences in optimum settings between dense and MoE models, and even for MoE models there are big differences between, say, DeepSeek and Maverick), users systems very between 100% GPU and 100% CPU, and anything in between, there are different quantization types with different tradeoffs, etc. Making it easy for the users would be the domain of product managers, marketing specialists, and technical support, none of which is present in a hobby project such as this one. Hence, it is basically up to the user base to come up with the cook book recipes. [@ubergarm](https://github.com/ubergarm) has done some of that [here](https://github.com/ikawrakow/ik_llama.cpp/discussions/258), but it is by no means complete (and things are moving and changing).

I don't think you should be so hard on yourself. The problem exists with mainline as well, which has FAR more people working on it. I'm fairly certain most users that use llama.cpp don't even know llama.cpp exists, they think of ollama (which has more stars than llama.cpp does), or one of the alternative front-ends which provides a streamlined experience (often but not always at a cost of less features/functionality/performance etc.).

You do a really good job of providing a lot of info in your PRs but there is no getting around the fact that there is way too much relevant information for the average user to take the time to read and understand (which might take them even longer since they may not have the prerequisite knowledge). You also do put in a LOT of effort to help people who end up here asking for help. I try to do the same on other platforms (since plenty of users do not even consider creating issues or discussions on github, which is why  I've ended up giving you bug reports on their behalf). It is really fortunate that the people that end up here for help, often give back in some way either by testing, or doing write-ups.

---

👤 **ubergarm** commented the **2025-06-02** at **16:18:04**:<br>

> Additionally, it seems like the number of combinations of tensor/config/compile settings are quite numerous, and more so now after these changes. Is there a way to know what the optimal arrangement should be? e.g. IQ4_K for GPU-tensors, _R4 for cpu tensors, GGML_CUDA_IQK_FORCE_BF16=1 etc. ? Or is it all YMMV, tradeoffs between PP/TG perf, CUDA-arch etc?

I'll piggy-back on what ik said in that things are moving and changing. I haven't read all the papers so have just picked up knowledge through experience and looking at the existing llama-quantize code and such. I've run a lot of my own a/b testing benchmarks which takes a long time and is not always fully conclusive, this is just a challenging area with lots of research happening as we speak.

ik is such a valuable resource and I've been impressed with the level of discussions that happen in many of these PRs. It really does take time to learn it even if all the information were put into a single guide. I've tried to distill some basic knowledge as mentioned, but even then a lot is lost.

There are even more variables to consider than you mention and you may have noticed a "friendly competition" with myself, bartowski, and unsloth "dynamic 2.0" quant recipes. We've all been experimenting with making some tensors bigger or smaller, some layers bigger or smaller, longer context or different imatrix corpus etc. All of these have some small impacts which may or may not really add up to noticeable or measurable improvements.

Also everything is trade-offs with no one-size-fits-all solution given the wide variety of hardware configurations. For example in general the more bpw the better quality but slower speeds. 

A few tips I can give you that I've gleaned:

1. the new `iq5_ks` quant and slightly smaller `iq4_ks` quants seem to be quite strong both in terms of quality and inferencing speed and are often a good choice. Better quant quality like this seems to deliever solid improvements in PPL/KLD measurement more than tweaking your imatrix dataset a little bit one way or the other imo.
2. you need to consider common hardware breakpoints in your quant design e.g. targeting 32GB RAM + 16GB VRAM system or maybe a 368GB RAM with no GPU system etc. Being just a few GiB too large to fit means much much slower performance than if you'd shaved off that little extra even if it costs a tiny bit of quality.
3. ease of use becomes an issue too especially with the pre-repacked `_r4` quants which until a week ago were useless for additional GPU offload making them very inflexible. I'm grateful ik added some functionality to run them on GPU to simplify my life by releasing a single quant that can be used more easily in a variety of configurations.
4. I'm not sure where this comes from, but in general people tend to make ffn_down a bit larger than ffn_(gate|up) tensors. It may be because down is used twice in the calculations, but honestly I'm very shaky in my implementation knowledge. You'll find this default pattern in the llama-quantize code where if you make a q4_0 quant it bumps up ffn_down to q4_1 or something like that.
5. Take a look at [turboderp's exllamav3 allocation code](https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/conversion/allocation.py#L34-L62) which has some heuristics about where to allocate the extra bpw between the attn qkvo tensors and the ffn gate/down/up tensors.
6. I try to make all of [my "secret recipes" public](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF#iq2_k_r4-2799-bpw-220gib) and share them so people can see what is going on under the hood and feel free to modify them and test it for themselves. Bartowski has a [public fork open to github](https://github.com/ggml-org/llama.cpp/pull/12727) where he keeps silently pushing updates that he uses for releasing his quants. You can also [look in the huggingface model card side-bar](https://huggingface.co/bartowski/deepseek-ai_DeepSeek-R1-0528-GGUF?show_file_info=deepseek-ai_DeepSeek-R1-0528-Q3_K_XL%2Fdeepseek-ai_DeepSeek-R1-0528-Q3_K_XL-00001-of-00009.gguf) for the information that you would get from `./gguf-py/gguf/scripts/gguf_dump.py` like the exact quantization of each tensor and layer. Unsloth used to keep their own public fork until mainline borked it by renaming the examples directory to tools directory at which point [i pushed a non-compiling version of their now missing branch](https://github.com/ubergarm/llama.cpp/tree/unsloths-old-quantization-branch) to my repo. It may be possible unsloth has since released their code, but in general they tend to be more secretive of their recipes and exact methodology like not posting their imatrix.dat file recently from what I can tell. It may be possible they are just busy and that is low priority for them, I dunno.

Okay, those are a few nuggets of wisdom I've picked up along the way. I have plenty more to learn every day and it is definitely and interesting field and glad to be playing one small part caught up with everyone in the incessant flow of the eternal dao. 😹 

Cheers!

---

👤 **ikawrakow** commented the **2025-07-05** at **13:13:00**:<br>

I think we can close this now.