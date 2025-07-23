### 🗣️ [#532](https://github.com/ikawrakow/ik_llama.cpp/discussions/532) - Guidance on GPU Layer Offloading Strategy in ik_llama.cpp for Multi GPU Rig (2x5090 + 2x4090)

| **Author** | `mtcl` |
| :--- | :--- |
| **Created** | 2025-06-16 |
| **Updated** | 2025-06-24 |

---

#### Description

@ikawrakow or @ubergarm

I've recently expanded my GPU rig to include (2x RTX 5090 + 2x RTX 4090) and am seeking your expertise to develop a systematic approach for offloading layers across these GPUs.

While I have experience with hardware configurations, I'd like to avoid ad-hoc experimentation and instead follow best practices or documented methodologies specific to ik_llama.cpp's architecture. Could you please share recommendations regarding:

Which types of layers (e.g., attention, feed-forward) benefit most from GPU acceleration? How do i know which layer I need to offload? Currently I have been randomly offloading whatever i can.
Optimal strategies for distributing work across heterogeneous GPUs (5090 vs 4090)?
Are there built-in features/flags in ik_llama.cpp to control layer distribution?

I'm particularly interested in any rationale behind layer offloading decisions in GPU-accelerated LLMs. 

this is one of the commands that I used:

For some reason my nvidia-smi shows GPU 0 and 3 as NVIDIA 5090, but in reality CUDA_VISIBLE_DEVICES sees GPUs 2 and 3 as NVIDIA 5090. So I have arranged it such that the first and last -ot parameter is for NVIDIA 5090 and in between two -ot parameters are for NVIDIA 40900.

for Qwen3-235B

```bash
CUDA_VISIBLE_DEVICES="2,1,0,3" ./build/bin/llama-server \
  --model /home/mukul/dev-ai/models/unsloth/Qwen3-235B-A22B-128K-GGUF/Q4_K_M/Qwen3-235B-A22B-128K-Q4_K_M-00001-of-00003.gguf \
  --alias unsloth/Qwen3-235B-A22B-128K-Q4_K_M \
  --ctx-size 65536 \
  -ctk q8_0 -ctv q8_0 \
  -fa \
  -b 4096 -ub 4096 \
  -fmoe \
  --n-gpu-layers 100 \
  -ot "blk\.([0-9]|1[0-5])\.ffn=CUDA0" \
  -ot "blk\.(1[6-9]|2[0-7])\.ffn=CUDA1" \
  -ot "blk\.(2[8-9]|3[0-9])\.ffn=CUDA2" \
  -ot "blk\.(4[0-9]|5[0-5])\.ffn=CUDA3" \
  --override-tensor exps=CPU \
  --parallel 1 \
  --threads 56 \
  --host 0.0.0.0 \
  --port 10002

```

and for a DeepSeek-R1

```bash
CUDA_VISIBLE_DEVICES="2,1,0,3" ./build/bin/llama-server \
    --model /home/mukul/dev-ai/models/ubergarm/DeepSeek-R1-0528-GGUF/IQ4_KS_R4/DeepSeek-R1-0528-IQ4_KS_R4-00001-of-00009.gguf \
    --alias ubergarm/DeepSeek-R1-0528-IQ4_KS_R4 \
    --ctx-size 40960 \
    -ctk q8_0 \
    -mla 3 -fa \
    -b 4096 -ub 4096 \
    -amb 512 \
    -fmoe \
    -ngl 63 \
    -ot "blk\.[3-4]\.ffn_.*=CUDA0" \
    -ot "blk\.[5-6]\.ffn_.*=CUDA1" \
    -ot "blk\.[7-8]\.ffn_.*=CUDA2" \
    -ot "blk\.[9]\.ffn=CUDA3,blk\.1[0-1]\.ffn=CUDA3" \
    -ot exps=CPU \
    --parallel 1 \
    --threads 56 \
    --host 0.0.0.0 \
    --port 10002
```

---

#### 🗣️ Discussion

👤 **ubergarm** replied the **2025-06-17** at **01:09:25**:<br>

> I'd like to avoid ad-hoc experimentation and instead follow best practices or documented methodologies specific to ik_llama.cpp's architecture.

I personally *do* advise using ad-hoc experimentation like simple `llama-sweep-bench` a/b comparisons to find out what works best for your specific hardware configuration. There are a number of discussions you could search for such as [this discussion thread](https://github.com/ikawrakow/ik_llama.cpp/discussions/477#discussioncomment-13366794) or search of `CUDA2` etc for other multi-gpu enjoyers like yourself e.g. @Lissanro @Ph0rk0z @ciprianveg @Thireus @rodriMora @Panchovix And it might depend on your PCIe allocation per card and stuff like that too.

If you'd like to document some best practices for multi-GPU offloading strategies for multiple LLMs, that would be welcome contribution! However keep in mind, things change fast so honestly spend some time looking through recently closed and newly opened PRs as some quants are getting a big boost for PP etc.

> Which types of layers (e.g., attention, feed-forward) benefit most from GPU acceleration? 

> How do i know which layer I need to offload?

Offload early offload often! No really, if you can offload the whole thing that is great. If not put as much as possible on your fastest GPUs first. Try to keep kv-cache near the attn tensors probably or all on a single e.g. `--main-gpu 0` or whatever maybe?

Usually the dense ffn layers get offloaded to CPU first as they are just larger. Hopefully your quant has optimized those for CPU/RAM usage e.g. `_r4` quant types or use `-rtr` etc.

I don't recommend separating `ffn_(gate|up)` as the `-fmoe` is fusing those together psure. Usually I just put all attn/shexp on GPU and as many other ffn that will fit for DeepSeek-R1. Qwen has no shared experts so same thing basically but be aware the names/regex are different and there is no `-ot exps=CPU` for Qwen btw (so remove that from your command). You can read more on my [ubergarm/Qwen3-235B-A22B-GGUF discussions](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/1#6814ea55554bef6174d3bab1)

> Currently I have been randomly offloading whatever i can.

This works pretty well a lot of time!

> Optimal strategies for distributing work across heterogeneous GPUs (5090 vs 4090)?

See above, put more layers on the fast GPUs first.

> Are there built-in features/flags in ik_llama.cpp to control layer distribution?

I'd suggest some combination of this depending on which model you're running:
```
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1 -DGGML_CUDA_F16=ON
cmake --build ./build --config Release -j $(nproc)
```

Use the `-DGGML_CUDA_IQK_FORCE_BF16=1` if you're running DeepSeek models. The `-DGGML_CUDA_F16=ON` is fore the new experimental `_kt` quants at least and maybe other stuff I'm not sure. Personally I leave BLAS off and don't fuss with any of that stuff (even the experimental intel compiler stuff i tried once seemed slower so i don't mess with any of it nor AMX stuff).

> For some reason my nvidia-smi shows GPU 0 and 3 as NVIDIA 5090, but in reality CUDA_VISIBLE_DEVICES sees GPUs 2 and 3 as NVIDIA 5090. 

I believe there is some device ordering environment variables you can use to swap those around, I thought I saw some chatter near the linked discussion above. e.g. `CUDA_DEVICE_ORDER=<PCI_BUS_ID>`

Cheers!

---

👤 **Ph0rk0z** replied the **2025-06-17** at **13:28:43**:<br>

It's a good idea to view all the layers and their file sizes on the model. That way you know what you can fit onto your GPUs. Not all blocks have the same sized layers. I have used llama.cpp mainline to print the sizes and adjusted accordingly. The more you cram, the more t/s you generally get. Smaller AMB can get you smaller buffers and perhaps fit another layer. Benchmarking is your friend. Once you cache the model in sysram, its easy to re-load and try things. There is some variance in llama-sweep-bench, but you can still spot larger trends. After a lot of testing, it might be wise to dump your cache and re-test your best ones.

> 👤 **Panchovix** replied the **2025-06-17** at **13:56:49**:<br>
> What is the effect by reducing or increasing the amb, besides buffer size?
> 
> 👤 **Ph0rk0z** replied the **2025-06-18** at **01:11:46**:<br>
> From the original PR it runs the ops multiple times, if I understand correctly. Breaking them up into batches. On some systems that can be slower? In practice I found little difference.
> 
> 👤 **Panchovix** replied the **2025-06-18** at **01:17:28**:<br>
> I'm not sure how it exactly either, the PR is too technical for me haha. I guess in theory reducing amb reduces the buffer sizes?
> 
> 👤 **ubergarm** replied the **2025-06-18** at **02:14:07**:<br>
> So without `-amb` it used to just fill up a bunch of VRAM and cause trouble. But using `-amb 512` for example will set aside a buffer of fixed size 512MiB VRAM and yes as @Ph0rk0z says it will use that fixed buffer size and loop over processing all the data that previously filled up a bunch of VRAM.
> 
> So it is a trade-off, in that it puts a cap on the amount of VRAM used but there is a little overhead to setup and copy into it looping over all the data to process.
> 
> If you make it too small, e.g. `-amb 64` things can get slower or stop working. So in general I leave it at like `-amb 512` or if I am squeezing something in and need a little more I'll drop to `-amb 256` but generally don't go lower than that.
> 
> 👤 **Ph0rk0z** replied the **2025-06-18** at **12:31:23**:<br>
> 64 actually worked for me. Some fractions of t/s off at higher contexts and that's all. I think at some point the buffer doesn't get much smaller. Higher AMB also didn't produce higher performance. Had tested 1024 and 2048 but nada. Does it affect anything like PPL or quality tho? Ubatch supposedly has some of that for context.
> 
> 👤 **ikawrakow** replied the **2025-06-18** at **13:00:04**:<br>
> Few clarifications for this thread
> * `-amb` only has effect if we have MLA (DeepSeek models), and/or if we are not using flash attention for models with GQA (almost all other models). I.e., when using FA, it has no effect whatsoever on any model other than DeepSeek.
> * It has no effect on accuracy
> * It splits the self attention calculation into chunks over attention heads in such a way that the intermediate buffers required for `K*Q` or, in the case of MLA, for the K-cache times `attn_k_b` tensor, does not exceed the specified amount of memory
> * Obviously this is only possible up to a point. When a chunk reaches a single attention head no further reduction is possible
> * It only controls compute buffer sizes of buffers related to self attention. There are many other operations in an LLM compute graph that also require temporary buffers for storing results. Those are not effected by `-amb`, so the actual compute buffer size is almost always larger than what is specified with `-amb`.  
> * It is theoretically slower than no `-amb` because, instead of doing one big matrix matrix multiplication, we need to do several smaller matrix multiplications, and this is always slower. This is why it is an option rather than being on by default.
> * In practice people seem to observe no measurable effect when using `-amb` with DeepSeek-R1/V3.
> * I observe about a 3% performance penalty when running DeepSeek-Lite (a 16B parameter model with the same attention mechanism as DeepSeek-V3/R1) fully offloaded to the GPU, and also about that when running this model CPU only.
> 
> 👤 **Ph0rk0z** replied the **2025-06-18** at **13:05:28**:<br>
> >-amb only has effect if we have MLA (DeepSeek models)
> 
> Good to know because I was copying 512 for qwen models at first, since people put it in their command line. Monkey see, monkey do. Took it out only when testing multiple sizes and seeing no change.

---

👤 **mtcl** replied the **2025-06-23** at **03:15:56**:<br>

@ubergarm and @ikawrakow  I have recently obtained 2XBlackwell Pro 6000 GPUs. so I have 192 Gigs of VRAM. I am able to offload your Qwen3-235B model completely on the model and i get over 1000 prompt processing and 50 tk/sec generation speed. But for the Deepseek model, i cant get beyond the 12-13tk/second. Would you have any advice for me? Below is the video where i compare all the different options, I have chapters in the video so that you can see the right sections. But any help will be really appreciated. I am starting to give up on deepseek. if two blackwells aren't enough then what is :/

https://www.youtube.com/watch?v=cFddXR1nPLg

8:01 - Test 1: Qwen 3 235B (Fully GPU Offloaded)
10:55 - Qwen 3 235B Loaded - Insane Performance!
12:18 - Qwen 3 235B Benchmark: 58 tokens/sec
18:21 - Qwen 3 235B Pushing the Limit: 128k Context Test
21:14 - Test 2: DeepSeek MoE Model (Partial Offload)
26:43 - Experimenting with Layer Offloading
31:29 - DeepSeek Benchmark & Power Draw
35:27 - DeepSeek's Impressive Snake Game
41:35 - DeepSeek Performance Results (12 tokens/sec)
44:27 - Test 3: DeepSeek on iklama.cpp (IQ3 Quant)
59:36 - iklama.cpp Performance Results (15 tokens/sec)
1:08:31 - Test 4: Llama 4 Maverick MoE Model
1:20:22 - Maverick Performance Results (57 tokens/sec!)

> 👤 **Panchovix** replied the **2025-06-23** at **03:19:03**:<br>
> Not them but for 2bpw or more you may get benefits by offloading.
> 
> If you want to load 2-4bpw fully on GP without offloading, then you need 2 6000 Pros more haha.
> 
> It is not normal to get 12 t/s PP. How are you running the model? On that specific quant I get about 200-300 t/s PP on a 5090 (and other GPUs, offloading about 100GB to RAM).
> 
> 👤 **mtcl** replied the **2025-06-23** at **03:28:12**:<br>
> > Not them but for 2bpw or more you may get benefits by offloading.
> > 
> > If you want to load 2-4bpw fully on GP without offloading, then you need 2 6000 Pros more haha.
> > 
> > It is not normal to get 12 t/s PP. How are you running the model? On that specific quant I get about 200-300 t/s PP on a 5090 (and other GPUs, offloading about 100GB to RAM).
> 
> Oh 12 t/s is generation speed, PP is 150ish I might as well get two more blackwells lol
> 
> 👤 **Panchovix** replied the **2025-06-23** at **03:30:27**:<br>
> Oh then those 12 t/s are probably kinda normal, I get like 7-8.5 t/s haha. Not sure if there's a way to improve more TG t/s besides running fully on GPU.
> 
> 👤 **saood06** replied the **2025-06-23** at **04:47:59**:<br>
> >besides running fully on GPU.
> 
> A mix of IQ1_S_R4 and IQ2_KT could fit in 192 Gigs of VRAM (I think pure IQ2_KT would be too big). Some measurements of quants and PPL. https://github.com/ikawrakow/ik_llama.cpp/pull/529#issuecomment-2978837501 and https://github.com/ikawrakow/ik_llama.cpp/pull/495#issuecomment-2988574743
> 
> 👤 **Ph0rk0z** replied the **2025-06-23** at **12:29:14**:<br>
> TG limited by the CPU/RAM of the system you are using if it's not fully on GPU.
> 
> 👤 **ubergarm** replied the **2025-06-23** at **14:54:25**:<br>
> Heya @mtcl you selling your used 5090s now already too? haha... 
> 
> Check [this l1t thread for a guy offloading IQ1_S onto 2x Blackwell 6000's](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/153) he did some llama-sweep bench with various batch sizes.
> 
> And yeah as the others have said you will be limited by however much active weights are left on CPU/RAM as TG will be bottlenecked by ram bandwidth.
> 
> saood06 is correct that a pure IQ2_KT is a little too big, it is like 192GiB (can't find my notes for exact value). But you could make a custom quant that is IQ1_S and IQ2_KT etc to get it down a little bit. I've had some requests for a ~196GB RAM target quant and that IQ2_KT would be pretty good if you can offload it all.
> 
> 👤 **mtcl** replied the **2025-06-23** at **15:55:20**:<br>
> I might add both 5090s back on the server to load a slightly bigger model 😂 but I'm in MI this week, I'll be back home in MN on Friday. Till then I only have remote kvm access to my machine.

---

👤 **kirnat** replied the **2025-06-24** at **17:08:34**:<br>

Hi Mukul. Thanks for your helpful videos.
I just wanted to add some data points since we share the same motherboard + cpu setup.

Hardware:
Asus Pro W790 Sage
Intel engineering sample QYFS (Xeon 8480 equivalent)
8x48GB @ 4800 (Sadly memory clock is locked on the CPU, even if you can flip the switch in BIOS)
1x Blackwell 6000 Pro RTX

Command line options:
llama-sweep-bench \
    -m ./models/unsloth/R1/DeepSeek-R1-0528-UD-Q4_K_XL-00001-of-00008.gguf \
    -fa \
    -t 52 \
    -ngl 61 \
    -ot "blk\.[0-9]\.ffn_(gate)_exps.=CPU" \
    -ot "blk\.1[0-9]\.ffn_(gate)_exps.=CPU" \
    -ot ".ffn_(up|down)_exps.=CPU" \
    -mla 1 \
    -rtr \
    -fmoe \
    -ctk q8_0 \
    -ctv q8_0 \
    -b 1024 \
    -ub 1024 \
    -c 32768

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    6.705 |   152.71 |   16.386 |    15.62 |
|  1024 |    256 |   1024 |    6.743 |   151.85 |   16.558 |    15.46 |
|  1024 |    256 |   2048 |    6.811 |   150.35 |   16.605 |    15.42 |
|  1024 |    256 |   3072 |    6.886 |   148.70 |   16.656 |    15.37 |
|  1024 |    256 |   4096 |    6.962 |   147.09 |   16.696 |    15.33 |
|  1024 |    256 |   5120 |    7.042 |   145.40 |   16.756 |    15.28 |
|  1024 |    256 |   6144 |    7.078 |   144.68 |   17.024 |    15.04 |
|  1024 |    256 |   7168 |    7.164 |   142.93 |   17.034 |    15.03 |
|  1024 |    256 |   8192 |    7.241 |   141.42 |   17.057 |    15.01 |
|  1024 |    256 |   9216 |    7.309 |   140.10 |   17.089 |    14.98 |
|  1024 |    256 |  10240 |    7.386 |   138.64 |   17.108 |    14.96 |
|  1024 |    256 |  11264 |    7.462 |   137.24 |   17.141 |    14.94 |
|  1024 |    256 |  12288 |    7.535 |   135.90 |   17.423 |    14.69 |
|  1024 |    256 |  13312 |    7.607 |   134.61 |   17.482 |    14.64 |
|  1024 |    256 |  14336 |    7.679 |   133.34 |   17.495 |    14.63 |
|  1024 |    256 |  15360 |    7.750 |   132.13 |   17.519 |    14.61 |
|  1024 |    256 |  16384 |    7.833 |   130.73 |   17.545 |    14.59 |
|  1024 |    256 |  17408 |    7.907 |   129.51 |   17.589 |    14.55 |
|  1024 |    256 |  18432 |    7.982 |   128.29 |   17.746 |    14.43 |
|  1024 |    256 |  19456 |    8.057 |   127.09 |   17.772 |    14.40 |
|  1024 |    256 |  20480 |    8.133 |   125.91 |   17.777 |    14.40 |
|  1024 |    256 |  21504 |    8.218 |   124.60 |   17.795 |    14.39 |
|  1024 |    256 |  22528 |    8.292 |   123.49 |   17.827 |    14.36 |
|  1024 |    256 |  23552 |    8.376 |   122.25 |   17.840 |    14.35 |
|  1024 |    256 |  24576 |    8.464 |   120.99 |   18.187 |    14.08 |
|  1024 |    256 |  25600 |    8.535 |   119.98 |   18.205 |    14.06 |
|  1024 |    256 |  26624 |    8.608 |   118.96 |   18.235 |    14.04 |
|  1024 |    256 |  27648 |    8.686 |   117.89 |   18.235 |    14.04 |
|  1024 |    256 |  28672 |    8.753 |   116.99 |   18.253 |    14.03 |
|  1024 |    256 |  29696 |    8.833 |   115.92 |   18.286 |    14.00 |
|  1024 |    256 |  30720 |    8.902 |   115.03 |   18.444 |    13.88 |
|  1024 |    256 |  31744 |    8.979 |   114.04 |   18.457 |    13.87 |

I used Ubergarm's high quality DeepSeek V3 R4 quantized model before ik llama.cpp had support for that quantization type on GPU with a 4090 and all experts, except for the shared one on CPU with about 12 output tps <10000 context. I will try again with the latest model from Ubergarm later.