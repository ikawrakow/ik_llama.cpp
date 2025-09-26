## 📌 [Issue #601](https://github.com/ikawrakow/ik_llama.cpp/issues/601) - Bug: llama-imatrix crashing

| **Author** | `Lissanro` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-12 |
| **Updated** | 2025-07-27 |

---

## 📄 Description

### What happened?

I wanted to create imatrix file for DeepSeek V3 but it keeps failing. In the past, I was able to create imatrix file with exactly the same command for R1 model. Did I do something wrong or is it a bug? Seems to be reproducible regardless of calibration dataset content.


### Name and Version

version: 3795 (c53cb652)
built with cc (Ubuntu 14.2.0-19ubuntu2) 14.2.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
> ~/pkgs/ik_llama.cpp/build/bin/llama-imatrix -m /mnt/neuro/DeepSeek-V3-0324/DeepSeek-V3-0324-Q8_0.gguf -f ~/pkgs/imatrix/all.txt --n-gpu-layers 62 --tensor-split 25,23,26,26 -mla 3 -fa -ctk q8_0 -amb 1024 -fmoe -ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" --threads 64
...
compute_imatrix: tokenizing the input ..
compute_imatrix: tokenization took 7579.07 ms
compute_imatrix: computing over 3660 chunks with batch_size 512
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
fatal error/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error

/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error

/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
fatal error
fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
/home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error
```

---

## 💬 Conversation

👤 **Lissanro** commented on **2025-07-12** at **02:56:11**

I should have checked with llama.cpp imatrix before reporting:

llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 1147, got 1025
llama_model_load_from_file_impl: failed to load model

It looks like I accidentally generated incomplete quant, not sure how that happened though, will try to regenerate (maybe the console where I was running it got closed by accident since cannot find the output).

Please consider this bug report as a request to for clearer error message... if the quant was incomplete, it would have saved me quite a bit of time if it reported the actual error instead of crashing. But if good error handling too hard to implement, please feel free to close this bug report.

---

👤 **ikawrakow** commented on **2025-07-12** at **06:45:17**

So, because of the issues around DeepSeek and the MLA tensors that can be different between mainline and `ik_llama.cpp` I disabled the tensor number check that triggers in mainline. That of course leads to the situation where faulty model will load but then crash because of missing tensors.

---

👤 **ubergarm** commented on **2025-07-12** at **15:49:53**

Heya @Lissanro here is the script I use that has worked on DeepSeek-R1, V3, V3-0324, R1-0528, and the new TNG Chimera models. Keep in mind if u got back to the `-fmoe` closed PR it mentions not to use that when doing imatrix to get data for the individual tensors. This is a dual socket intel xeon 6980P with 768GB RAM per numa node (SNC=Disable gives one numa node per socket):

```bash
numactl -N 0 -m 0 \
./build/bin/llama-imatrix \
    -m /mnt/raid/models/ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF/DeepSeek-TNG-R1T2-Chimera-Q8_0.gguf \
    -f ubergarm-imatrix-calibration-corpus-v02.txt \
    -o /mnt/raid/models/ubergarm/DeepSeek-TNG-R1T2-Chimera-GGUF/imatrix-DeepSeek-TNG-R1T2-Chimera-Q8_0.dat \
    --verbosity 1 \
    --ctx-size 512 \
    --layer-similarity \
    --numa numactl \
    --threads 128
```

I only ever convert fp8 safetensors via the evshiron llama.cpp fork (made from fairydreaming's original MLA stuf) plus triton-cpu to get bf16 GGUFs directly without need for > sm89 architechture GPU or any GPU at all.

Feel free to use my imatrix files which were made with ik's fork using this method which are on huggingface for each model I release.

P.S. I have done it the mainline way by casting the fp8 to bf16 safetensors then doing another step to go from bf16 safetensors to bf16 GGUF. You can do it with triton-cpu as well though its not documented anywhere besides a single post where I discussed it. However, I've made some quants for mainline but they kept throwing `nan` when testing perplexity so not sure what was going on and I abandoned that project for now hah... This was all mainline llama.cpp stuff, so the nans have nothing to do with this fork (with which I've had more success).

---

👤 **saood06** commented on **2025-07-12** at **21:04:32**

> llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 1147, got 1025 llama_model_load_from_file_impl: failed to load model
> 
> It looks like I accidentally generated incomplete quant, not sure how that happened though, will try to regenerate (maybe the console where I was running it got closed by accident since cannot find the output).

I don't know if it is an "incomplete quant", as 1025 tensors is what I see in my notes for the earliest GGUF I tested of Deepseek (with the more recent one's I use having 1147 from the extra MLA tensors).

---

👤 **Lissanro** commented on **2025-07-12** at **22:30:30**

@ubergarm
Thank you, I was making it work without crashing. As it turned out the issue wasn't missing tensors (rebuilding from scratch did not help), but it seems some extra options in my command were crashing it. When I used your command with some adjustment to my system (I have only 64 cores) and paths, it started working, however I tried without any GPUs for now. I will try carefully to add GPU options when I am not using another model actively.

@saood06
This is how I converted from fp8 to bf16:

```
> python3 /home/lissanro/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py \
--outtype bf16 \
--outfile /mnt/neuro/DeepSeek-V3-0324/DeepSeek-V3-0324-BF16.gguf \
/mnt/secondary/neuro/DeepSeek-V3-0324 --split-max-size 48G
```

At the end it says the total number of tensors is 1147. The next conversion command also reports back the same amount:

```
> ~/pkgs/ik_llama.cpp/build/bin/llama-quantize \
/mnt/neuro/DeepSeek-V3-0324/DeepSeek-V3-0324-BF16-00001-of-00030.gguf \
/mnt/neuro/DeepSeek-V3-0324/DeepSeek-V3-0324-Q8_0.gguf Q8_0
...
[1147/1147]                   output_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
```

I wonder, does that mean extra MLA tensors were bundled in the original FP8 model, or did /convert_hf_to_gguf.py add them? I did not take a note how many tensors R1 and R1T original FP8 models had when I was converting them, so not sure if this one is different or the same.

---

👤 **saood06** commented on **2025-07-12** at **23:05:01**

> I wonder, does that mean extra MLA tensors were bundled in the original FP8 model, or did /convert_hf_to_gguf.py add them? I did not take a note how many tensors R1 and R1T original FP8 models had when I was converting them, so not sure if this one is different or the same.

It really depends on your definition of bundled or added. It is doing as the name suggest and converting between the formats which have different layouts and conventions (MLA related things are not the only differences, for example GGUF currently packs multiple experts together, safetensors do not).

See my old comment [here](https://github.com/ikawrakow/ik_llama.cpp/discussions/354#discussioncomment-13054586) where I go over different ways these MLA tensors have been handled in GGUFs (and as the edit suggests the comment is outdated in terms of what is and is not supported here, just linking for reference to the different types).

Hopefully this helps you understand.

>Thank you, I was making it work without crashing. As it turned out the issue wasn't missing tensors (rebuilding from scratch did not help), but it seems some extra options in my command were crashing it. When I used your command with some adjustment to my system (I have only 64 cores) and paths, it started working, however I tried without any GPUs for now. I will try carefully to add GPU options when I am not using another model actively.

I have very limited experience in creating imatrix files, but I do remember `-fmoe` was stated as not compatible as "this option cannot be used when computing an imatrix because than the intermediate results remain in temporary work buffers, hence will not be propagated to collect activation statistics for the up_exps and gate_exps tensors." (from [#229](https://github.com/ikawrakow/ik_llama.cpp/issues/229)).

I'm not sure if that was the only issue, but it seems like it may have been an issue.

---

👤 **ubergarm** commented on **2025-07-12** at **23:22:43**

@Lissanro 

>  however I tried without any GPUs for now

Glad you're able to get it to run at least on CPU. Curious if it would work with CUDA too.

> This is how I converted from fp8 to bf16:

Wait are you using mainline llama.cpp to do the conversion `python3 /home/lissanro/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py` after using https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/fp8_cast_bf16.py (and possibly triton-cpu if you don't have a sm89 or newer GPU)?

And then ik to do the imatrix `~/pkgs/ik_llama.cpp/build/bin/llama-imatrix` ? 

I've only recently tried that once for an experiment trying a bunch of tiny IQ1_S quants with older quantization types possibly to run on AMD GPU but got distracted. I can't remember but some combination threw an error, either mainline llama.cpp imatrixing my usual method of `evshiron+triton-cpu` quant or vice versa... 

I did grab a gguf-dump of the first bf16 file for both methods if you'd like to look, I put both of them here:

https://gist.github.com/ubergarm/d9a3e89355199fc34d8c75882bcc3ab4

If I make a quant converted with the mainline two step cast method, it also shows up when starting on ik_llama.cpp with that error message `missing wkv_b tensor(s) changing MLA from %d to 1`.

---

👤 **saood06** commented on **2025-07-12** at **23:40:00**

> I only ever convert fp8 safetensors via the evshiron llama.cpp fork (made from fairydreaming's original MLA stuf) plus triton-cpu to get bf16 GGUFs directly without need for > sm89 architechture GPU or any GPU at all.

I don't like that this is the way I still resort to doing it (a goal of mine [even if I haven't been working at it at all recently] is to make using any convert script outside this repo not needed for making GGUFs for models supported by this repo*. Besides upcasting FP8 using triton, I know certain models like Gemma 3 and GLM-4 still aren't supported).

*Well besides the new bitnet model as they have their own standalone scripts [this](https://github.com/microsoft/BitNet/blob/main/utils/convert-ms-to-gguf-bitnet.py) and [this](https://github.com/microsoft/BitNet/blob/main/utils/convert-hf-to-gguf-bitnet.py) that I had issues using.

---

👤 **Lissanro** commented on **2025-07-13** at **00:25:30**

@ubergarm 

> Wait are you using mainline llama.cpp to do the conversion python3 /home/lissanro/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py

No, it does direct conversion from FP8 to BF16. As the directory name suggests it is special version that I only use to convert FP8 to BF16, since the official DeepSeek script never worked for me. I think the special version uses triton-cpu. My workflow to convert from FP8 to the final IQ4 quant is shared here:

https://github.com/ikawrakow/ik_llama.cpp/issues/383#issuecomment-2869544925

And according to it, having -fmoe wasn't causing crashes in the past when creating imatrix, this is why I was using it, I just wasn't aware it is not supported anymore for the imatrix creation (based on information in this thread, it sounds like maybe it was never really supported). Since my workflow shared in quite many places, once I test things out from start to finish with recent ik_llama.cpp, I will edit it to make sure it is up to date.

---

👤 **saood06** commented on **2025-07-13** at **01:43:19**

> And according to it, having -fmoe wasn't causing crashes in the past when creating imatrix, this is why I was using it, I just wasn't aware it is not supported anymore for the imatrix creation (based on information in this thread, it sounds like maybe it was never really supported).

Even if it wasn't causing crashing it might explain why your imatrix file was smaller than it should have been. (130 MB vs 987 MB), and potentially less than ideal (maybe this is why `IQ4_KS_R4` performed so poorly on the maze, but this is pure speculation at this point).

---

👤 **ubergarm** commented on **2025-07-14** at **15:51:51**

@saood06 

> I don't like that this is the way I still resort to doing it (a goal of mine [even if I haven't been working at it at all recently] is to make using any convert script outside this repo not needed for making GGUFs for models supported by this repo*. Besides upcasting FP8 using triton, I know certain models like Gemma 3 and GLM-4 still aren't supported).

Yeah I wasn't sure where ik_llama.cpp convert_hf_to_gguf.py stands and skipped porting over the python code on GLM-4 and also Hunyuan-A13B....

I can't remember your name on huggingface, but wanted to loop you in on [Kimi-K2-Instruct conversion and imatrix stuff](https://huggingface.co/gabriellarson/Kimi-K2-Instruct-GGUF/discussions/1#687522c60c755f6c912037a1).

My goal is to get a "small" Kimi-K2-Instruct GGUF using ik's SOTA quants. However, it is a slightly modified DeepSeek architecture with more routed exps, only one ffn dense layer up front (instead of 3), and less MLA heads I believe. 

I'm currently working through testing a mainline PR and it seems to be running there, but I'm not sure if I can use that bf16 GGUF or if I need to update the evshiron fork method to ensure not getting that `missing wkv_b tensor(s)` warning restricing us to `-mla 1`.

Details are in that hf link above, and I also decided to go with Compilade's unmerged imatrix GGUF PR as it still saves data even when the routed exps are not 100% (it was dropping a lot at first). Not sure on how compatible that "imatrix.gguf" will be here if I convert it back to ".dat"...

Not sure how it will pan out, but I think we'll get there eventually!

---

👤 **ikawrakow** commented on **2025-07-14** at **16:20:27**

> My goal is to get a "small" Kimi-K2-Instruct GGUF using ik's SOTA quants. However, it is a slightly modified DeepSeek architecture with more routed exps, only one ffn dense layer up front (instead of 3), and less MLA heads I believe.

As far as I can tell, the only thing that needs a change is the pre-tokenizer. The number of dense layers, total number of experts, etc., is all taken from the GGUF metadata, so such differences are irrelevant. Oh, one needs to also see if my hack to convert mainline's conventions on head dimensions and such to `ik_llama.cpp` works, given the change in number of heads.

> I'm currently working through testing a mainline PR and it seems to be running there, but I'm not sure if I can use that bf16 GGUF or if I need to update the evshiron fork method to ensure not getting that missing wkv_b tensor(s) warning restricing us to -mla 1.

So, the preferred way to calculate the imatrix is to use `mla = 1`. This gives you imatrix data for the `wk_b` and `wv_b` tensors, which is good. It is good because these two get used for TG, so you want them quantized with fewer bits if possible. If `wkv_b` is added to the GGUF, it should be quantized with `Q8_0`. If it is not added, `ik_llama.cpp` will (nearly) losslessly create `wkv_b` tensors as `Q8_0` from `wk_b` and `wv_b` while loading the model. `wkv_b` being `Q8_0` is fine because tit only gets used for PP, so the more bits don't matter for performance.

If you instead run the imatrix calculation with `mla = 3`, there will only be data for `wkv_b`. `wk_b` and `wk_v` will not have imatrix data, so need to be quantized with more bits, so this will result in lower TG performance.

Unless you are worried about model size and want to squeeze out the last bit possible. In that case you need to run the imatrix calculation twice (once with `mla = 3` and once with `mla = 1`), and somehow merge the two datasets.

---

👤 **saood06** commented on **2025-07-14** at **18:40:33**

> Yeah I wasn't sure where ik_llama.cpp convert_hf_to_gguf.py stands and skipped porting over the python code on GLM-4 and also Hunyuan-A13B....

That's reasonable, and you aren't the only one to skip that, I do try to port the conversion stuff and convert from source when bringing over models, but it isn't needed to inference with the model and is less used in general (most people download GGUF quants not source .safetensors).

I obviously wouldn't complain if you decide to go back to those models and update the python code but obviously you don't have to do that.

> I can't remember your name on huggingface, but wanted to loop you in on [Kimi-K2-Instruct conversion and imatrix stuff](https://huggingface.co/gabriellarson/Kimi-K2-Instruct-GGUF/discussions/1#687522c60c755f6c912037a1).

No worries, I saw the thread from when you linked it elsewhere here (and in general I check my notifications here far more than on HF).

> My goal is to get a "small" Kimi-K2-Instruct GGUF using ik's SOTA quants. 

Nice, I'm not sure how much I'll be doing with this model given my hardware (I do have a server with 1 TB of RAM but I haven't used it for a long time given it has some hardware instability, noise, and power issues).

>However, it is a slightly modified DeepSeek architecture with more routed exps, only one ffn dense layer up front (instead of 3), and less MLA heads I believe.

That's what I heard as well.

> Details are in that hf link above

Will read through that. (Edit: Gave a reply there as well).

>and I also decided to go with Compilade's unmerged imatrix GGUF PR as it still saves data even when the routed exps are not 100% (it was dropping a lot at first). Not sure on how compatible that "imatrix.gguf" will be here if I convert it back to ".dat"...

You mean to accomplish something similar to [#202](https://github.com/ikawrakow/ik_llama.cpp/issues/202). I've been saying on mainline that nicoboss's fork was based on this PR (since I was the one who reported the issue that lead to the creation of that PR and went back and told them and they made their fork based on that).

> Not sure how it will pan out, but I think we'll get there eventually!

Let me know if you need me to help with anything.

---

👤 **ikawrakow** commented on **2025-07-14** at **19:40:31**

> and I also decided to go with Compilade's unmerged imatrix GGUF PR as it still saves data even when the routed exps are not 100% (it was dropping a lot at first). Not sure on how compatible that "imatrix.gguf" will be here if I convert it back to ".dat"...

If you insist on calculating the imatrix with mainline, you absolutely need compilade's PR. Not because "it still saves data even when the routed exps are not 100%", but because without that PR mainline calculates broken self-attention imatrix data for MLA models (and has been doing that for the last 3 months, and before that it couldn't because it did not support MLA).

Having said that, there is nothing in compilade's PR that has not been solved here a long time ago. Given that [#609](https://github.com/ikawrakow/ik_llama.cpp/issues/609) has been merged, I would calculate the imatrix data with `ik_llama.cpp` if I were you.

---

👤 **saood06** commented on **2025-07-14** at **19:51:08**

>Having said that, there is nothing in compilade's PR that has not been solved here a long time ago. Given that [[#609](https://github.com/ikawrakow/ik_llama.cpp/issues/609)](https://github.com/ikawrakow/ik_llama.cpp/pull/609) has been merged, I would calculate the imatrix data with `ik_llama.cpp` if I were you.

I agree about generating the imatrix data with `ik_llama.cpp`, but the one thing that has not been solved (at least not ideally in my opinion) is turning the FP8 source file into BF16 but it seems like @ubergarm is already past that point based on the HF thread (also just to clarify this is a separate issue outside the scope of [#609](https://github.com/ikawrakow/ik_llama.cpp/issues/609) or the compilade PR).

---

👤 **ubergarm** commented on **2025-07-14** at **20:01:59**

Thanks y'all, and yes I *want* to use ik_llama.cpp imatrix!! 

I had never understood exactly what step messes up the MLA tensors with the "mainline fp8_cast_bf16.py -> convert_hf_to_gguf.py method" vs what I use here referred to as the "evshiron+triton-cpu direct fp8 -> bf16 gguf method".

But I think I finally understand it and got it going now... I'm using ik_llama.cpp's convert_hf_to_gguf.py now adapted with the mainline PR for Kimi-K2

```
INFO:gguf.gguf_writer:/mnt/raid/models/ubergarm/Kimi-K2-Instruct-GGUF/Kimi-K2-384x15B-Instruct-safetensors-BF16-00045-of-00045.gguf: n_tensors = 35, total_size = 45.7G
Shard (2/45):  25%|██▍       | 11.3G/45.4G [01:00<02:17, 248Mbyte/s]
Writing:   3%|▎         | 60.1G/2.05T [04:40<2:19:20, 239Mbyte/s]
```

This bf16 GGUF should have the right stuff in it so that my quants won't print out the `missing wkv_b tensor(s)` warning! 🤞 

I realize I could make imatrix with ik_llama.cpp using a mainline quant, but then I'm still stuck as the quants I cook would all throw that error without fixing the convert step. Thanks!

---

👤 **saood06** commented on **2025-07-14** at **20:06:39**

> I had never understood exactly what step messes up the MLA tensors with the "mainline fp8_cast_bf16.py -> convert_hf_to_gguf.py method" vs what I use here referred to as the "evshiron+triton-cpu direct fp8 -> bf16 gguf method".

The python script that converts the safetensors into a GGUF is the one that determines what MLA tensors you end up with.

---

👤 **ubergarm** commented on **2025-07-14** at **20:10:40**

> The python script that converts the safetensors into a GGUF is the one that determines what MLA tensors you end up with.

Yup, I never quite realized that as the evshiron method being a single step confused me. I never grokked where exactly things were happening until going through this all today.  Also it isn't apparent when I was looking in the `./gguf-py/scripts/gguf_dump.py` for "what is different between my quants and mainline quants" given the `wkv_b` tensors don't appear by that name in either one which also led me astray haha..

I link to the different code in question [in this comment here](https://github.com/ikawrakow/ik_llama.cpp/pull/609#issuecomment-3070754157)

Thanks for your patience I can be pretty slow on the uptake sometimes haha

---

👤 **saood06** commented on **2025-07-14** at **20:21:48**

> > The python script that converts the safetensors into a GGUF is the one that determines what MLA tensors you end up with.
> 
> Yup, I never quite realized that as the evshiron method being a single step confused me. 

Yes that isn't the most intuitive, but it is really convenient.

>Also it isn't apparent when I was looking in the `./gguf-py/scripts/gguf_dump.py` for "what is different between my quants and mainline quants" given the `wkv_b` tensors don't appear by that name in either one which also led me astray haha..

That is why I keep linking the comment which goes over three types and the differences between them because the differences might not be readily apparent.

> Thanks for your patience I can be pretty slow on the uptake sometimes haha

Thank you for doing all this. It helps a lot of people, so I'm glad to assist when I can.

---

👤 **ubergarm** commented on **2025-07-14** at **20:37:39**

> Yes that isn't the most intuitive, but it is really convenient.

Yeah, though fortunately now I have a method to use triton-cpu (with your help patching that) and use deepseek's fp8_cast_bf16.py directly to avoid needing enough VRAM or >=sm89 arch for fp8e4m3 support. 

At that point can just use the convert script here in ik's fork and so far so good... I'll know for sure in couple hours hah...


> That is why I keep linking the comment which goes over three types and the differences between them because the differences might not be readily apparent.

Ahh yes, I have definitely read this before, but it didn't sink in, and notes are scattered across so many platforms these days alas... Here it is again for my future self to stuble on it:

> So in conclusion if the model has all three attn_k_b.weight, attn_v_b.weight and attn_kv_b.weight or just attn_kv_b.weight it will work here, but if it has attn_k_b.weight and attn_v_b.weight but no attn_kv_b.weight it will not work here. *EDIT BY UBERGARM* To be clear ik_llama.cpp does support mainline quants despite mainline changing the MLA tensors!!! 

And just confirmed that the Q8_0 I quantized from the mainline convert script is indeed lacking `attn_kv_b`:

```bash
$ cat quantize-Kimi-K2-Instruct-mainline-Q8_0.log | grep attn_kv_b
# nothing
```

---

👤 **saood06** commented on **2025-07-14** at **20:43:12**

> Yeah, though fortunately now I have a method to use triton-cpu (with your help patching that) and use deepseek's fp8_cast_bf16.py directly to avoid needing enough VRAM or >=sm89 arch for fp8e4m3 support.

I never did that as once you have triton-cpu the evshiron method saves you a step so I always did that.

> Ahh yes, I have definitely read this before, but it didn't sink in, and notes are scattered across so many platforms these days alas... Here it is again for my future self to stuble on it:
> 
> > So in conclusion if the model has all three attn_k_b.weight, attn_v_b.weight and attn_kv_b.weight or just attn_kv_b.weight it will work here, but if it has attn_k_b.weight and attn_v_b.weight but no attn_kv_b.weight it will not work here.
> 

NO. The conclusion to that comment is outdated (and I say so in the comment).

The point to linking the old comment is not for the conclusion or even about compatibility, it is just about the differing MLA tensors amongst GGUFs that exist. The comment was written and edited with those things in mind but I'm linking it just for the differing model and what tensors they contain (I really should have just taken that info out instead of linking it, but I didn't think it would cause confusion).

---

👤 **ubergarm** commented on **2025-07-14** at **20:58:05**

> NO. The conclusion to that comment is outdated (and I say so in the comment).
> 
> The point to linking the old comment is not for the conclusion or even about compatibility, it is just about the differing MLA tensors amongst GGUFs that exist. 

I think I'm doing too many things at the same time, sorry to misunderstand yet again lol. I do understand and agree that since fairydreaming's early MLA PR that was not merged, there are indeed a variety of differing MLA tensors amongst GGUFs that exist.

---

👤 **saood06** commented on **2025-07-14** at **21:05:08**

> I think I'm doing too many things at the same time, sorry to misunderstand yet again lol. 

The big No was because the conclusion is outdated and wrong `ik_llama.cpp` now does work with models with tensors like that, and I don't want anyone getting confused about that.

>I do understand and agree that since fairydreaming's early MLA PR that was not merged, there are indeed a variety of differing MLA tensors amongst GGUFs that exist.

Yes (and some from even before any MLA implementation exists). I was linking it as an answer to people asking stuff like "what is different between my quants and mainline quants" which you also asked.

---

👤 **ubergarm** commented on **2025-07-14** at **21:10:13**

Right, I added edited the comment above and stuck this in there: `EDIT BY UBERGARM To be clear ik_llama.cpp does support mainline quants despite mainline changing the MLA tensors!!!`

Yeah ik supports a a lot of thing mainline does not, but definitely people outside of this github seem to get even more confused ideas than me! haha

Thanks!

---

👤 **Lissanro** commented on **2025-07-19** at **06:04:11**

I tried to rebuilt my quants avoid using -fmoe and -mla 3 options, but use just -mla 1 instead. I was successfully was able to rebuild V3 quant, but R1 gives me a trouble (get nan during imatrix), I would appreciate if anyone encountered similar issues or know how to debug this.

First, I create Q8 from BF16:

~/pkgs/ik_llama.cpp/build/bin/llama-quantize /mnt/secondary/neuro/DeepSeek-R1-0528/DeepSeek-R1-256x21B-0528-BF16.gguf /mnt/neuro/models/DeepSeek-R1-256x21B-0528-IQ4_K-163840seq/DeepSeek-R1-256x21B-0528-Q8_0.gguf Q8_0

Then I try to build imatrix:

~/pkgs/ik_llama.cpp/build/bin/llama-imatrix -m /mnt/neuro/models/DeepSeek-R1-256x21B-0528-IQ4_K-163840seq/DeepSeek-R1-256x21B-0528-Q8_0.gguf -f ~/pkgs/imatrix/compact.txt --n-gpu-layers 62 --tensor-split 25,23,26,26 -ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" --threads 64 -mla 1 -b 4096 -ub 4096
...
save_imatrix: stored collected data after 730 chunks in imatrix.dat
[730]4.8195,[731]4.8186,[732]4.8137,[733]4.8200,[734]4.8243,[735]4.8169,[736]4.8161,[737]4.8118,nan detected in blk.60.attn_output.weight

I also tried without "-mla 1 -b 4096 -ub 4096" and it crashed in a similar way. Maybe something wrong with my Q8 or maybe I missed some imatrix option that is needed, but could not figure this out just yet.

---

👤 **ikawrakow** commented on **2025-07-19** at **06:47:39**

This is a bummer. No-one has reported a problem such as this, so it could be useful to see the calibration data if it is not secret.

You can still use the matrix data that was saved before the NaN occurred.

---

👤 **ubergarm** commented on **2025-07-19** at **14:42:07**

@Lissanro 

Your command looks reasonable, and while i personally don't mix `-ts` and `-ot` it should be fine if its loading how you like onto your GPUs. I haven't used `-ub 4096 -b 4096` while doing imatrix, but it should be fine I just learned yesterday and still work at the default n_ctx 512 which I want.

I presume you compiled with `-DGGML_CUDA_IQK_FORCE_BF16=1` to avoid nans specifically with DeepSeek/MLA models e.g.:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_VULKAN=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1
cmake --build build --config Release -j $(nproc)
```

Otherwise yes bummer indeed.

---

👤 **ikawrakow** commented on **2025-07-19** at **18:05:54**

> I presume you compiled with -DGGML_CUDA_IQK_FORCE_BF16=1 to avoid nans specifically with DeepSeek/MLA models

That was relevant only for quants that did not have quantized matrix multiplications (a.k.a., MMQ), and hence dequantized to `f16` by default, which resulted in NaNs for DeepSeek. This is no longer relevant as all quants have MMQ now. It never was relevant for `Q8_0`.

---

👤 **ubergarm** commented on **2025-07-19** at **18:16:11**

> This is no longer relevant as all quants have MMQ now. It never was relevant for Q8_0.

Thanks for the clarification, the 😈 is in the details. I'll do my best to not perpetuate incorrect collective knowledge for all eternity... 😅

---

👤 **Lissanro** commented on **2025-07-22** at **15:47:48**

@ikawrakow Sure, I can share my imatrix data and from where I got it, I just wanted to try one more time to be sure before providing more details. I tried this time without `-mla 1` and without `-ub 4096 -b 4096` options (with them outcome is the same):

```
~/pkgs/ik_llama.cpp/build/bin/llama-imatrix \
-m /mnt/neuro/models/DeepSeek-R1-256x21B-0528-IQ4_K-163840seq/DeepSeek-R1-256x21B-0528-Q8_0.gguf \
-f ~/pkgs/imatrix/compact-imatrix.txt --n-gpu-layers 62 --tensor-split 25,23,26,26 \
-ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" --threads 64
...
save_imatrix: stored collected data after 720 chunks in imatrix.dat
[720]4.8196,[721]4.8264,[722]4.8189,[723]4.8177,[724]4.8175,[725]4.8171,[726]4.8170,[727]4.8097,[728]4.8111,[729]4.8132,
save_imatrix: stored collected data after 730 chunks in imatrix.dat
[730]4.8195,[731]4.8186,[732]4.8137,[733]4.8200,[734]4.8243,[735]4.8169,[736]4.8161,[737]4.8118,nan detected in blk.60.attn_output.weight

```

Full log: https://pastebin.com/CgFtLbWB (even though the working folder named IQ4, I am using Q8_0 GGUF quant, and did not get to making IQ4 yet).

My calibration data https://dragon.studio/2025/07/compact-imatrix.txt is a merge of these two:
https://huggingface.co/ThomasBaruzier/Qwen2.5-3B-Instruct-GGUF/blob/main/calibration_datav3.txt
https://huggingface.co/datasets/eaddario/imatrix-calibration/blob/main/combined_all_tiny.parquet (converted to txt first)

This attempt was with updated repo from few days ago (could not update before running again due to repo not being available at the time).

By the way, by using the imatrix data, do you mean I can use the resulted file as is, or can continue it somehow without starting over? Any ideas if it is something in my calibration data causing this and if yes, how to pinpoint it? It worked for V3 though, so it is a bit surprising that it fails for R1 0528.

---

👤 **ikawrakow** commented on **2025-07-22** at **16:05:57**

> By the way, by using the imatrix data, do you mean I can use the resulted file as is, or can continue it somehow without starting over? Any ideas if it is something in my calibration data causing this and if yes, how to pinpoint it? It worked for V3 though, so it is a bit surprising that it fails for R1 0528.

It got saved to `imatrix.dat` after 730 chunks. You can use this file to create quantized models. Rename it to something else to not lose it when you try the following:
* Add `--from-chunk 738` to your imatrix command line. This will start computing from chunk 738 thus skipping chunk 737 that produces the NaN
* If that goes well, you can combine the two (or more) imatrix files by using
```
./bin/imatrix --in-file file1 --in-file file2 [--in-file file3 ...] -o combined_imatrix.dat
```

---

👤 **magikRUKKOLA** commented on **2025-07-25** at **00:02:15**

> /home/lissanro/pkgs/ik_llama.cpp/ggml/src/ggml.c:15229: fatal error

Uh oh this nasty error.

I can confirm it still present when running the latest ik_llama.cpp master with some 6 bpw+ quants from the Thireus (for the INFERENCE).

In my case it was solved by removing the -fmoe from the command.  Tried to look into it, but failed :( which is sad.  It was somewhat related to munlock.

[EDIT]: pretty sure I can reproduce it for the PPL calculations with -fmoe, low-free-ram and some absend swap file.

---

👤 **Lissanro** commented on **2025-07-25** at **15:01:32**

@ikawrakow Thank you so much! It is good to know that it is possible to combine imatrix files like that, and besides bypassing NaN errors, opens up a possibility to build in few cycles rather than in one go. Once I completed the imatrix, I think it worked well - Q4_K_XS is about as good or better then Q4_K_M quant I had previously with wrongly built imatrix, and reasoning quality breakdown that I originally observed when testing XS is no longer happens.

* * *

By the way, does anyone here have an idea how to convert Kimi K2 FP8 to BF16? I thought it is based on DeepSeek architecture so thought at least quantizing it going to be easy, but so far I could not found any way to convert it to BF16 - 
for example, llama.cpp folk https://github.com/evshiron/llama.cpp that had script capable of converting FP8 to BF16 for DeepSeek V3 and R1 crashes when loading Kimi K2 safetensors:

```
python3 /home/lissanro/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py \
--outtype bf16 --outfile /mnt/Toshiba_Canvio_4TB/Kimi-K2-Instruct/Kimi-K2-Instruct-BF16.gguf \
/mnt/neuro/models/Kimi-K2-Instruct/
...
Traceback (most recent call last):
  File "/home/lissanro/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py", line 5244, in <module>
    main()
    ~~~~^^
  File "/home/lissanro/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py", line 5238, in main
    model_instance.write()
    ~~~~~~~~~~~~~~~~~~~~^^
  File "/home/lissanro/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py", line 440, in write
    self.prepare_metadata(vocab_only=False)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/home/lissanro/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py", line 433, in prepare_metadata
    self.set_vocab()
    ~~~~~~~~~~~~~~^^
  File "/home/lissanro/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py", line 4058, in set_vocab
    self._set_vocab_gpt2()
    ~~~~~~~~~~~~~~~~~~~~^^
  File "/home/lissanro/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py", line 728, in _set_vocab_gpt2
    tokens, toktypes, tokpre = self.get_vocab_base()
                               ~~~~~~~~~~~~~~~~~~~^^
  File "/home/lissanro/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py", line 522, in get_vocab_base
    tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
  File "/home/lissanro/.local/lib/python3.13/site-packages/transformers/models/auto/tokenization_auto.py", line 946, in from_pretrained
    tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
  File "/home/lissanro/.local/lib/python3.13/site-packages/transformers/models/auto/tokenization_auto.py", line 800, in get_tokenizer_config
    result = json.load(reader)
  File "/usr/lib/python3.13/json/__init__.py", line 293, in load
    return loads(fp.read(),
        cls=cls, object_hook=object_hook,
        parse_float=parse_float, parse_int=parse_int,
        parse_constant=parse_constant, object_pairs_hook=object_pairs_hook, **kw)
  File "/usr/lib/python3.13/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/usr/lib/python3.13/json/decoder.py", line 348, in decode
    raise JSONDecodeError("Extra data", s, end)
json.decoder.JSONDecodeError: Extra data: line 165 column 2 (char 5595)
```

(full log: https://pastebin.com/3GmeCK9d )

convert_hf_to_gguf.py script from ik_llama.cpp and original llama.cpp does not seem to support FP8 unfortunately (so I assume this is not actually a bug, hence this is why I do not open an issue for it, and why I am trying alternatives that can work with CPU or 3090 cards).

---

👤 **ubergarm** commented on **2025-07-26** at **18:20:07**

@Lissanro 

> convert_hf_to_gguf.py script from ik_llama.cpp and original llama.cpp does not seem to support FP8 unfortunately 

Both myself and anikifoss have converted Kimi-K2 using this repo's `convert_hf_to_gguf.py` script. I've written about all three methods to get fp8 safetensors into bf16 GGUFs here: https://github.com/ggml-org/llama.cpp/issues/14762#issuecomment-3098571703

If you have the disk space, the triton-cpu method works quite well though takes some time. Otherwise you can grab pre-made bf16 GGUFs from like unsloth or folks which have already been passed through DeepSeek's original fp8_cast_bf16 python script but not yet converted.

---

👤 **Lissanro** commented on **2025-07-26** at **18:51:10**

@ubergarm 
I actually managed to get conversion going today, but it was non-trivial and required some modifications. The error I shared above was the easiest to solve and caused by a change of the json in the upstream repository - my download script double checks that all files fully downloaded and the new version was slightly bigger so ended up as a corrupted file that I had to redownload.

But, then it complained about requiring trust_remote_code=True to the AutoTokenizer.from_pretrained function call, then some other things... later on, about some unprocessed experts. So, I ended up with this patch:

https://dragon.studio/2025/07/lama.cpp-fp8-to-bf16-patch.diff

It was based on the differences between the https://github.com/evshiron/llama.cpp fork and the upstream llama.cpp related to the conversion script and Kimi K2 updates.

I am still in the process of converting the model though, and then will need to build imatrix and final quant, so it will be a while before I actually test the result to see if it works as expected. But I though I share the patch here in case it is useful to someone. I used this command to start the conversion process:

```
python3 ~/pkgs/llama.cpp-fp8-to-bf16/llama.cpp/convert_hf_to_gguf.py \
--outtype bf16 \
--outfile /mnt/neuro/Kimi-K2-Instruct/Kimi-K2-Instruct-BF16.gguf \
/mnt/neuro/models/Kimi-K2-Instruct/
```

_(I am on 4G connection so even downloading original FP8 was quite a challenge; there is also a risk to get bandwidth throttled if I go beyond some undefined traffic limit, this is why I could not just download premade BF16 after downloading FP8 and had to figure out the solution)_

---

👤 **ubergarm** commented on **2025-07-26** at **19:14:45**

@Lissanro 

Oh amazing you got the Kimi-K2 fp8 safetensors over 4G! Yes wouldn't want to download the 2TB bf16 safetensors!  Glad you got it to convert and thanks for sharing your patch.

> and then will need to build imatrix

I assume you've been following along with some questions about imatrix for MLA tensors in the other threads, which I've added an issue for now here: https://github.com/ikawrakow/ik_llama.cpp/issues/651

I'd be curious if your imatrix run for kimi-k2-instruct gives you data for those attn_k_b and attn_v_b tensors. You're welcome to use my imatrix, however, I don't think it contains that data despite my running it with `-mla 1` as explained in the issue.

---

👤 **saood06** commented on **2025-07-27** at **01:21:24**

> But I though I share the patch here in case it is useful to someone. I used this command to start the conversion process:

Thanks. It may end up being useful in some way to me.